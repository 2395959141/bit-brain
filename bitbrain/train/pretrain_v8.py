##! 在v6的基础上，进一步添加训练过程中的测试，并添加权重衰减
##! 默认使用混合精度，并启动 torch.compile加速
##! 本版本从 DDP 修改为 FSDP 策略
import os
import sys
import argparse
import time
import math
import torch.compiler #! 导入 torch.compiler 以解决 CUDAGraphs 问题
from functools import partial #! FSDP CHANGE: 导入 partial 用于自动包装策略

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
#! FSDP CHANGE: 移除 DDP, 导入 FSDP 相关模块
# from torch.distributed.fsdp import (
#     FullyShardedDataParallel as FSDP,
#     ShardingStrategy,
#     MixedPrecision,
#     BackwardPrefetch,
# )
# from torch.distributed.fsdp.wrap import (
#     transformer_auto_wrap_policy,
# )
# from torch.distributed.fsdp.api import StateDictType

from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torch.amp import GradScaler, autocast #! autocast 仅用于非FSDP场景
from modelscope import AutoConfig, AutoModelForCausalLM, AutoTokenizer
# 导入 HuggingFace Datasets 库
from datasets import load_from_disk
from transformers import get_cosine_schedule_with_warmup
#! FSDP CHANGE: 导入模型内部的 Transformer Block，用于自动包装策略
# 请根据你的模型库和版本确认正确的导入路径
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer

from loguru import logger
from bitbrain.train.tools.mfu import get_gpu_peak_flops, estimate_model_flops, calculate_mfu_distributed
from bitbrain.train.tools.utils import test_model_on_prompts
from liger_kernel.transformers import AutoLigerKernelForCausalLM
import swanlab

#! 禁用CUDA Graph Trees
torch._inductor.config.triton.cudagraph_trees = False

#model_id = "Qwen/Qwen2-0.5B"
#model_id = "/DATA/disk2/yuhang/.cache/modelscope/models/Qwen/Qwen2.5-0.5B-Instruct"
model_id = "/DATA/disk2/yuhang/.cache/modelscope/models/Qwen/Qwen3-0.6B"

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=12)
parser.add_argument("--seq_len", type=int, default=2048, help="训练时使用的序列长度")
parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")
parser.add_argument("--world_size", type=int, default=1, help="Number of processes for distributed training")
parser.add_argument("--master_addr", type=str, default="localhost", help="Master address for distributed training")
parser.add_argument("--master_port", type=str, default="12355", help="Master port for distributed training")
parser.add_argument("--save_dir", type=str, default="./out", help="用于保存在epoch中途的检查点的目录。默认: 'checkpoints_in_epoch'")
parser.add_argument("--save_interval", type=int, default=5000, help="每N个原始批次（dataloader的批次）保存一次检查点。默认: 1000。如果为0，则禁用epoch中途保存。")
args = parser.parse_args()

def setup_distributed():
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        rank = args.local_rank
        world_size = args.world_size
        local_rank = args.local_rank
        os.environ['MASTER_ADDR'] = args.master_addr
        os.environ['MASTER_PORT'] = args.master_port
    
    dist.init_process_group(backend='nccl')
    torch.cuda.set_device(local_rank)
    return rank, world_size, local_rank

if torch.distributed.is_available() and torch.cuda.device_count() > 1:
    rank, world_size, local_rank = setup_distributed()
    is_distributed = True
    device = f"cuda:{local_rank}"
    if rank == 0:
        logger.info(f"分布式训练已启用 (FSDP): rank={rank}, world_size={world_size}, local_rank={local_rank}")
        logger.info(f"使用 {world_size} 个GPU进行训练")
else:
    rank = 0
    world_size = 1
    local_rank = 0
    is_distributed = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("使用单卡训练模式")

num_epochs = 1

def main_process_only(func):
    def wrapper(*args, **kwargs):
        if rank == 0:
            return func(*args, **kwargs)
        return None
    return wrapper

original_logger_info = logger.info
logger.info = main_process_only(original_logger_info)

logger.info(f"Loading tokenizer for {model_id}...")
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
logger.info(f"Tokenizer for {model_id} loaded successfully.")

# 使用 HuggingFace Datasets 库加载预处理好的数据集
logger.info("加载预处理的数据集...")
train_dataset = load_from_disk("/DATA/disk2/yuhang/.cache/bit_brain_data/step3_tokenizer_data")
logger.info(f"数据集加载完成，共有 {len(train_dataset)} 条数据")

# 设置数据集格式为 PyTorch 张量
train_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask'])

if is_distributed:
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler, 
                             collate_fn=lambda batch: {
                                 'input_ids': torch.stack([item['input_ids'] for item in batch]),
                                 'attention_mask': torch.stack([item['attention_mask'] for item in batch])
                             })
else:
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                             collate_fn=lambda batch: {
                                 'input_ids': torch.stack([item['input_ids'] for item in batch]),
                                 'attention_mask': torch.stack([item['attention_mask'] for item in batch])
                             })

gradient_accumulation_steps = 8
mixed_precision_dtype = torch.bfloat16
compile_mode = "default"

torch.set_float32_matmul_precision('high')

logger.info(f"Loading configuration for {model_id} from ModelScope...")
qwen_config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
logger.info(f"Configuration for {model_id} loaded successfully.")

qwen_config.rope_theta = 10000
qwen_config.max_position_embeddings = 4096
logger.info(f"Model's max_position_embeddings set to: {qwen_config.max_position_embeddings} from args.seq_len")

logger.info(f"Initializing a new model from configuration: {model_id} (training from scratch)...")
model = AutoLigerKernelForCausalLM.from_config(config=qwen_config, trust_remote_code=True)
logger.info(f"New model initialized successfully with random weights based on {model_id} configuration.")

is_gradient_checkpointing_enabled = False
logger.info("使用torch.compile时关闭梯度检查点，兼容性考虑")

config_to_save = qwen_config
model = model.to(device)
total_params = sum(p.numel() for p in model.parameters())
logger.info(f"Total parameters: {total_params / 1e6:.2f} M")

is_model_compiled = False
logger.info(f"尝试使用 torch.compile (模式: {compile_mode}) 编译模型以优化性能...")
try:
    if hasattr(torch, 'compile'):
        compile_start_time = time.time()
        model = torch.compile(model, mode=compile_mode, fullgraph=True)
        compile_duration = time.time() - compile_start_time
        logger.info(f"模型编译完成！编译耗时: {compile_duration:.2f}s")
        is_model_compiled = True
    else:
        logger.warning("当前 PyTorch 版本不支持 torch.compile，跳过编译优化")
except Exception as e:
    logger.error(f"模型编译失败，将继续使用未编译版本: {e}")

#! DDP CHANGE: 使用传统 DDP 实现 Zero-1
if is_distributed:
    logger.info("将模型包装为 DDP (Zero-1 策略)...")
    model = DDP(
        model,
        device_ids=[local_rank],
        output_device=local_rank,
        find_unused_parameters=False,  # 如果模型中有未使用的参数，设置为 True
    )
    logger.info(f"DDP 包装完成。策略: Zero-1 (传统DDP), 混合精度: {mixed_precision_dtype}")

# ... MFU and other initializations
seq_len = args.seq_len
vocab_size = qwen_config.vocab_size
model_flops = estimate_model_flops(model.module if is_distributed else model, logger) # FSDP unwrapping for MFU
single_gpu_peak_flops = get_gpu_peak_flops(logger)
total_tokens_processed = 0
total_training_time = 0

lr_scheduler_config = {
    "scheduler_type": "cosine_with_warmup",
    "max_lr": 1e-4,
    "min_lr": 1e-5,
    "warmup_steps": 2500,
    "warmup_ratio": 0.25,
}
weight_decay = 0.1

#! FSDP CHANGE: 优化器必须在 FSDP 包装模型之后创建
optimizer = torch.optim.AdamW(
    model.parameters(), 
    lr=lr_scheduler_config["max_lr"],
    weight_decay=weight_decay,
    betas=(0.9, 0.95)
)

steps_per_epoch = len(train_loader) // gradient_accumulation_steps
total_training_steps = steps_per_epoch
warmup_steps = lr_scheduler_config["warmup_steps"] or int(total_training_steps * lr_scheduler_config["warmup_ratio"])

logger.info(f"学习率调度配置: 总步数={total_training_steps}, 热身步数={warmup_steps}")

def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr_ratio=0.1, last_epoch=-1):
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            warmup_progress = float(current_step) / float(max(1, num_warmup_steps))
            return min_lr_ratio + (1.0 - min_lr_ratio) * warmup_progress
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)

min_lr_ratio = lr_scheduler_config["min_lr"] / lr_scheduler_config["max_lr"]
scheduler = get_cosine_schedule_with_warmup(
    optimizer, num_warmup_steps=warmup_steps, num_training_steps=total_training_steps, min_lr_ratio=min_lr_ratio
)

if rank == 0:
    swanlab_config = { "model_id": model_id, "total_params_M": total_params / 1e6, "vocab_size": vocab_size, "seq_len": seq_len, "batch_size": args.batch_size, "gradient_accumulation_steps": gradient_accumulation_steps, "effective_batch_size": args.batch_size * gradient_accumulation_steps, "global_effective_batch_size": args.batch_size * gradient_accumulation_steps * world_size, "num_epochs": num_epochs, "learning_rate_max": lr_scheduler_config["max_lr"], "learning_rate_min": lr_scheduler_config["min_lr"], "scheduler_type": lr_scheduler_config["scheduler_type"], "warmup_steps": warmup_steps, "total_training_steps": total_training_steps, "steps_per_epoch": steps_per_epoch, "weight_decay": weight_decay, "world_size": world_size, "is_distributed": is_distributed, "mixed_precision_dtype": str(mixed_precision_dtype), "gradient_checkpointing": is_gradient_checkpointing_enabled, "torch_compile": is_model_compiled, "compile_mode": compile_mode, "device": device, "gpu_count": world_size, "fsdp_strategy": "N/A" }
    swanlab_run = swanlab.init(project="bitbrain-pretrain_v1", experiment_name=f"qwen2-pretrain-{time.strftime('%Y%m%d_%H%M%S')}", config=swanlab_config, description="Qwen2-0.5B pre-training with FSDP, mixed precision, and compile.")
else:
    swanlab_run = None

#! DDP CHANGE: 恢复 GradScaler 用于混合精度训练
scaler = None
if mixed_precision_dtype == torch.float16:
    scaler = GradScaler()
    logger.info("使用 torch.float16 混合精度，已初始化 GradScaler。")
elif mixed_precision_dtype == torch.bfloat16:
    scaler = None
    logger.info("使用 torch.bfloat16 混合精度，GradScaler 设置为 None。")

logger.info(f"训练配置: 分布式={is_distributed}, 梯度累积={gradient_accumulation_steps}, torch.compile={is_model_compiled}")

#! DDP CHANGE: 修改检查点保存函数以支持 DDP
def save_checkpoint_helper(model, optimizer, scheduler, scaler, config_to_save, epoch, current_optimizer_step_in_epoch, global_optimizer_step, args_namespace, is_final_checkpoint=False):
    """
    保存检查点，适配单卡和 DDP 场景。
    """
    # 获取状态字典
    if is_distributed:
        model_state_dict = model.module.state_dict()  # DDP 需要使用 .module
    else:
        model_state_dict = model.state_dict()
    
    optimizer_state_dict = optimizer.state_dict()

    # 只有 rank 0 执行文件写入操作
    if rank == 0:
        if is_final_checkpoint:
            save_dir = "checkpoints"
            filename_prefix = f"qwen2_pretrain_epoch_{epoch}"
            checkpoint_filename = f"{filename_prefix}.pt"
        else:
            save_dir = args_namespace.save_dir
            timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
            filename_prefix = f'pretrain_epoch{epoch}_optstep{current_optimizer_step_in_epoch}'
            checkpoint_filename = f'{filename_prefix}_{timestamp}.pth'
        
        os.makedirs(save_dir, exist_ok=True)
        checkpoint_path = os.path.join(save_dir, checkpoint_filename)

        checkpoint_data = {
            "model_config": config_to_save,
            "model_state_dict": model_state_dict,
            "optimizer_state_dict": optimizer_state_dict,
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict() if scaler else None,
            "epoch": epoch,
            "global_optimizer_step": global_optimizer_step,
            "args": vars(args_namespace),
            "mixed_precision_dtype_at_save": str(mixed_precision_dtype),
            "weight_decay_at_save": weight_decay,
        }
        
        torch.save(checkpoint_data, checkpoint_path)
        log_prefix = "DDP " if is_distributed else ""
        original_logger_info(f"{log_prefix}检查点已保存至: {checkpoint_path} (Epoch {epoch}, Optimizer Step {current_optimizer_step_in_epoch})")

def train(model, optimizer, scheduler, train_loader, device,
          epoch, scaler, gradient_accumulation_steps=4,
          tokens_per_optimizer_step_local=0):
    model.train()
    accumulated_loss = 0
    if is_distributed:
        train_loader.sampler.set_epoch(epoch)
    
    epoch_start_time = time.time()
    step_start_time = None
    first_batch_done = False
    epoch_step_count = 0
    epoch_total_loss = 0
    num_optimizer_steps_per_epoch = len(train_loader) // gradient_accumulation_steps

    for batch_idx, batch in enumerate(train_loader):
        if (batch_idx) % gradient_accumulation_steps == 0:
            step_start_time = time.time()
        
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = input_ids.clone()
        
        #! DDP CHANGE: 对于 DDP，可以使用 autocast 进行混合精度训练
        if scaler is not None:  # float16
            with autocast(device_type='cuda', dtype=torch.float16):
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss = loss / gradient_accumulation_steps
            scaler.scale(loss).backward()
        else:  # bfloat16 或 float32
            if mixed_precision_dtype == torch.bfloat16:
                with autocast(device_type='cuda', dtype=torch.bfloat16):
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                    loss = outputs.loss
                    loss = loss / gradient_accumulation_steps
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
                loss = outputs.loss
                loss = loss / gradient_accumulation_steps
            loss.backward()
        
        accumulated_loss += loss.item()
        
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            if scaler is not None:
                # 使用 GradScaler 进行梯度裁剪和优化器更新
                scaler.unscale_(optimizer)
                total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                if torch.isfinite(total_norm):
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    logger.warning(f"Skipping optimizer step at epoch {epoch}, batch_idx {batch_idx} due to non-finite gradients (norm: {total_norm}).")
            else:
                # 直接进行梯度裁剪和优化器更新
                total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                if torch.isfinite(total_norm):
                    optimizer.step()
                else:
                    logger.warning(f"Skipping optimizer step at epoch {epoch}, batch_idx {batch_idx} due to non-finite gradients (norm: {total_norm}).")

            optimizer.zero_grad()
            scheduler.step()
            
            step_end_time = time.time()
            step_time = step_end_time - step_start_time
            
            epoch_step_count += 1
            epoch_total_loss += accumulated_loss
            
            global total_tokens_processed
            total_tokens_processed += tokens_per_optimizer_step_local * world_size

            current_optimizer_step_in_epoch = (batch_idx // gradient_accumulation_steps) + 1
            
            if current_optimizer_step_in_epoch % 10 == 0:
                tokens_per_sec_global = (tokens_per_optimizer_step_local * world_size) / step_time if step_time > 0 else 0
                mfu = calculate_mfu_distributed(
                    tokens_per_optimizer_step_local, step_time, model_flops, 
                    single_gpu_peak_flops, world_size
                ) if step_time > 0 and model_flops > 0 else 0
                
                if rank == 0:
                    current_lr = scheduler.get_last_lr()[0]
                    log_message = (
                        f'Epoch:[{epoch+1}/{num_epochs}] '
                        f'Step:[{current_optimizer_step_in_epoch}/{num_optimizer_steps_per_epoch}] '
                        f'Loss:{accumulated_loss:.4f} '
                        f'LR:{current_lr:.8f} '
                        f'Tokens/s (Global):{tokens_per_sec_global:.0f} '
                        f'StepTime:{step_time:.3f}s '
                        f'MFU:{mfu*100:.2f}%'
                    )
                    original_logger_info(log_message)
                    if swanlab_run:
                        global_step = epoch * num_optimizer_steps_per_epoch + current_optimizer_step_in_epoch
                        swanlab.log({
                            "train/loss": accumulated_loss, "train/learning_rate": current_lr, "train/epoch": epoch + 1,
                            "train/step": current_optimizer_step_in_epoch, "train/global_step": global_step,
                            "performance/tokens_per_second_global": tokens_per_sec_global, "performance/mfu_percent": mfu * 100,
                            "performance/step_time_seconds": step_time,
                        })
            
            accumulated_loss = 0
            
            global_optimizer_step = (epoch * num_optimizer_steps_per_epoch) + current_optimizer_step_in_epoch
            #! DDP CHANGE: 保存检查点需要在所有 rank 上调用
            if args.save_interval > 0 and global_optimizer_step % args.save_interval == 0:
                if rank == 0:
                    # 测试只在主进程进行
                    test_model_on_prompts(model, tokenizer, device, max_seq_len=80)
                
                # 保存函数现在需要在所有 rank 上调用，它内部处理 rank 0 的写入
                save_checkpoint_helper(
                    model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
                    config_to_save=config_to_save, epoch=epoch + 1,
                    current_optimizer_step_in_epoch=current_optimizer_step_in_epoch,
                    global_optimizer_step=global_optimizer_step, args_namespace=args,
                    is_final_checkpoint=False
                )
                
                # 确保所有进程都完成了保存操作后再继续
                if is_distributed:
                    dist.barrier()
                
                # 模型返回训练模式
                model.train()

    epoch_end_time = time.time()
    epoch_duration = epoch_end_time - epoch_start_time
    global total_training_time
    total_training_time += epoch_duration
    
    avg_epoch_loss = epoch_total_loss / epoch_step_count if epoch_step_count > 0 else 0
    if rank == 0 and swanlab_run:
        swanlab.log({"epoch/avg_loss": avg_epoch_loss, "epoch/epoch_num": epoch + 1})
    
    logger.info(f"Epoch {epoch + 1} 完成 - 用时: {epoch_duration:.2f}s")
    return avg_epoch_loss

# 训练主循环
logger.info(f"Starting pretraining for {num_epochs} epochs...")
total_start_time = time.time()
tokens_per_optimizer_step_local = seq_len * args.batch_size * gradient_accumulation_steps

for epoch in range(num_epochs):
    logger.info(f"Starting epoch {epoch + 1}/{num_epochs}")
    train_loss_avg = train(model, optimizer, scheduler, train_loader, device, epoch, 
                           scaler, gradient_accumulation_steps, 
                           tokens_per_optimizer_step_local=tokens_per_optimizer_step_local)
    
    if is_distributed:
        dist.barrier()
        
    # 保存 epoch 结束时的最终检查点
    final_global_optimizer_step = (epoch + 1) * steps_per_epoch
    save_checkpoint_helper(
        model=model, optimizer=optimizer, scheduler=scheduler, scaler=scaler,
        config_to_save=config_to_save, epoch=epoch + 1,
        current_optimizer_step_in_epoch=steps_per_epoch,
        global_optimizer_step=final_global_optimizer_step,
        args_namespace=args, is_final_checkpoint=True
    )
    if is_distributed:
        dist.barrier()

if rank == 0 and swanlab_run:
    logger.info("训练完成，正在结束SwanLab实验记录...")

if is_distributed:
    dist.destroy_process_group()