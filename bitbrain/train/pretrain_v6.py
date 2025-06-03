##
#!  在v5的基础上，读取提前tokenizer之后的arrow数据文件
#!  增加模型权重保存
import os
import sys
import argparse
import time  #! 添加时间模块用于性能监控
import math  #! 添加数学模块用于FLOPS计算
# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import torch
import torch.distributed as dist  #! 添加分布式训练支持
from torch.nn.parallel import DistributedDataParallel as DDP  #! 添加DDP支持
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler  #! 添加分布式采样器
from torch.amp import GradScaler, autocast  #! 添加混合精度训练支持
from modelscope import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from bitbrain.dataset.pretrain_dataset_arrow import PretrainDataset
from loguru import logger
#! 从新的 mfu.py 文件导入 MFU 相关函数
from bitbrain.train.tools.mfu import get_gpu_peak_flops, estimate_model_flops, calculate_mfu_distributed
#! 导入SwanLab用于实验跟踪
import swanlab

#model_id = "Qwen/Qwen2-0.5B"
model_id = "/DATA/disk2/yuhang/.cache/modelscope/models/Qwen/Qwen2___5-0___5B-Instruct"

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=26)
parser.add_argument("--seq_len", type=int, default=2048)
#! 添加分布式训练相关参数
parser.add_argument("--local_rank", type=int, default=-1, help="Local rank for distributed training")
parser.add_argument("--world_size", type=int, default=1, help="Number of processes for distributed training")
parser.add_argument("--master_addr", type=str, default="localhost", help="Master address for distributed training")
parser.add_argument("--master_port", type=str, default="12355", help="Master port for distributed training")
#! 添加模型保存相关参数
parser.add_argument("--save_dir", type=str, default="./out", help="用于保存在epoch中途的检查点的目录。默认: 'checkpoints_in_epoch'")
parser.add_argument("--save_interval", type=int, default=5000, help="每N个原始批次（dataloader的批次）保存一次检查点。默认: 1000。如果为0，则禁用epoch中途保存。")
args = parser.parse_args()

#! 初始化分布式训练环境
def setup_distributed():
    """初始化分布式训练环境"""
    # 检查是否在分布式环境中运行
    if 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        # 从环境变量获取分布式参数（推荐方式）
        rank = int(os.environ['RANK'])
        world_size = int(os.environ['WORLD_SIZE'])
        local_rank = int(os.environ['LOCAL_RANK'])
    else:
        # 从命令行参数获取（备用方式）
        rank = args.local_rank
        world_size = args.world_size
        local_rank = args.local_rank
        
        # 设置环境变量
        os.environ['MASTER_ADDR'] = args.master_addr
        os.environ['MASTER_PORT'] = args.master_port
    
    # 初始化进程组
    dist.init_process_group(backend='nccl')
    
    # 设置当前进程使用的GPU
    torch.cuda.set_device(local_rank)
    
    return rank, world_size, local_rank

#! 初始化分布式训练（如果可用）
if torch.distributed.is_available() and torch.cuda.device_count() > 1:
    rank, world_size, local_rank = setup_distributed()
    is_distributed = True
    device = f"cuda:{local_rank}"
    
    # 只在主进程输出初始化信息
    if rank == 0:
        logger.info(f"分布式训练已启用: rank={rank}, world_size={world_size}, local_rank={local_rank}")
        logger.info(f"使用 {world_size} 个GPU进行训练")
else:
    rank = 0
    world_size = 1
    local_rank = 0
    is_distributed = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    logger.info("使用单卡训练模式")

# 只在主进程输出日志的装饰器
def main_process_only(func):
    """装饰器：只在主进程执行函数"""
    def wrapper(*args, **kwargs):
        if rank == 0:
            return func(*args, **kwargs)
        return None
    return wrapper

# 包装logger的info方法
original_logger_info = logger.info
logger.info = main_process_only(original_logger_info)

#! (3) 加载与模型配置匹配的分词器
logger.info(f"Loading tokenizer for {model_id}...")
tokenizer = AutoTokenizer.from_pretrained(model_id,attn_implementation="flash_attention_2",trust_remote_code=True)
logger.info(f"Tokenizer for {model_id} loaded successfully.")

##! (0) 加载并配置训练数据集
train_dataset = PretrainDataset(data_path="/DATA/disk2/yuhang/.cache/steel_dataset/step3_tokenizer_data")
                                    

#train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [0.9, 0.1])

#! 使用分布式采样器
if is_distributed:
    train_sampler = DistributedSampler(train_dataset, num_replicas=world_size, rank=rank, shuffle=True)
    #val_sampler = DistributedSampler(val_dataset, num_replicas=world_size, rank=rank, shuffle=False)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler)
    #val_loader = DataLoader(val_dataset, batch_size=args.batch_size, sampler=val_sampler)
else:
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    #val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

#* 训练配置参数
gradient_accumulation_steps = 2  # 梯度累计步数，实际batch_size = batch_size * gradient_accumulation_steps
# use_mixed_precision 布尔标志已移除，混合精度默认启用
mixed_precision_dtype = torch.bfloat16  # 或者 torch.float16，根据需要配置
# use_gradient_checkpointing 布尔标志已移除，梯度检查点默认尝试启用
# use_torch_compile 布尔标志已移除，torch.compile 默认尝试启用
compile_mode = "default"  # 编译模式：'default', 'reduce-overhead', 'max-autotune'

torch.set_float32_matmul_precision('high')

#! (1) 加载模型配置
logger.info(f"Loading configuration for {model_id} from ModelScope...")
qwen_config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
logger.info(f"Configuration for {model_id} loaded successfully.")

#! 设置不使用sliding window attention
qwen_config.use_sliding_window_attention = False

#* 打印配置中的最大位置嵌入（通常是最大上下文长度）
logger.info(f"Model's max_position_embeddings from config: {qwen_config.max_position_embeddings}")

#! (2) 从配置初始化新模型（权重随机初始化）
logger.info(f"Initializing a new model from configuration: {model_id} (training from scratch)...")
model = AutoModelForCausalLM.from_config(config=qwen_config)
logger.info(f"New model initialized successfully with random weights based on {model_id} configuration.")
if hasattr(qwen_config, 'layer_types'):
    logger.info(f"原始 config.layer_types (前3个): {qwen_config.layer_types[:3]}")
    # 确保所有层都设置为 full attention
    qwen_config.layer_types = ["full_attention"] * qwen_config.num_hidden_layers
    logger.info(f"修改后的 config.layer_types (前3个): {qwen_config.layer_types[:3]}")

#! 尝试启用梯度检查点
is_gradient_checkpointing_enabled = False
logger.info("尝试启用梯度检查点以节省显存...")
try:
    model.gradient_checkpointing_enable()
    logger.info("梯度检查点启用成功.")
    is_gradient_checkpointing_enabled = True
except AttributeError:
    logger.warning("模型不支持梯度检查点，将不使用此功能.")
    # is_gradient_checkpointing_enabled 保持 False

#* 用于保存检查点
config_to_save = qwen_config

# 将模型移动到对应的设备
model = model.to(device)

total_params = sum(p.numel() for p in model.parameters())
logger.info(f"Total parameters: {total_params / 1e6:.2f} M")

#! 尝试 torch.compile 支持 (在DDP包装之前)
is_model_compiled = False
logger.info(f"尝试使用 torch.compile (模式: {compile_mode}) 编译模型以优化性能...")
try:
    # 检查 PyTorch 版本是否支持 compile
    if hasattr(torch, 'compile'):
        compile_start_time = time.time()
        # 先编译模型，再包装DDP
        model = torch.compile(model, mode=compile_mode)
        compile_end_time = time.time()
        compile_duration = compile_end_time - compile_start_time
        logger.info(f"模型编译完成！编译耗时: {compile_duration:.2f}s")
        logger.info("注意：首次前向传播可能会有额外的编译开销")
        is_model_compiled = True
    else:
        logger.warning("当前 PyTorch 版本不支持 torch.compile，跳过编译优化")
except Exception as e:
    logger.error(f"模型编译失败，将继续使用未编译版本: {e}")
    # is_model_compiled 保持 False

#! 包装模型为DDP（在编译之后）
if is_distributed:
    logger.info("将模型包装为DistributedDataParallel...")
    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=False)
    logger.info("DDP包装完成")

#! 添加性能监控初始化
# 获取序列长度（从数据集或配置中）
seq_len = qwen_config.max_position_embeddings if hasattr(qwen_config, 'max_position_embeddings') else 2048
vocab_size = qwen_config.vocab_size

# 估算模型FLOPS
#! 调用时传入 logger 对象
model_flops = estimate_model_flops(model, logger)

# 获取单个GPU理论峰值FLOPS
#! 调用时传入 logger 对象
single_gpu_peak_flops = get_gpu_peak_flops(logger)

# 性能统计变量
total_tokens_processed = 0
total_training_time = 0

# 在优化器配置部分添加学习率调度相关参数
lr_scheduler_config = {
    "scheduler_type": "cosine_with_warmup",  # 调度器类型
    "max_lr": 1e-4,                          # 最大学习率
    "min_lr": 1e-5,                          # 最小学习率（最大学习率的10%）
    "warmup_steps": 2500,                     # 热身步数
    "warmup_ratio": 0.25,                    # 热身比例（如果warmup_steps为None则使用此值）
}

#! (4) 设置优化器和改进的学习率调度器
optimizer = torch.optim.AdamW(model.parameters(), lr=lr_scheduler_config["max_lr"])

# 动态计算总训练步数
steps_per_epoch = len(train_loader) // gradient_accumulation_steps
total_training_steps = steps_per_epoch 

# 计算热身步数
if lr_scheduler_config.get("warmup_steps") is not None:
    warmup_steps = lr_scheduler_config["warmup_steps"]
else:
    warmup_steps = int(total_training_steps * lr_scheduler_config["warmup_ratio"])

logger.info(f"学习率调度配置:")
logger.info(f"  - 调度器类型: {lr_scheduler_config['scheduler_type']}")
logger.info(f"  - 最大学习率: {lr_scheduler_config['max_lr']}")
logger.info(f"  - 最小学习率: {lr_scheduler_config['min_lr']}")
logger.info(f"  - 总训练步数: {total_training_steps}")
logger.info(f"  - 热身步数: {warmup_steps}")
logger.info(f"  - 每轮步数: {steps_per_epoch}")

# 创建自定义的带热身的余弦退火调度器
def get_cosine_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, min_lr_ratio=0.1, last_epoch=-1):
    """
    创建带热身的余弦退火学习率调度器
    
    Args:
        optimizer: 优化器
        num_warmup_steps: 热身步数
        num_training_steps: 总训练步数
        min_lr_ratio: 最小学习率与初始学习率的比例
        last_epoch: 上次训练的epoch
    """
    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            # 热身阶段：线性增长
            return float(current_step) / float(max(1, num_warmup_steps))
        
        # 余弦退火阶段
        progress = float(current_step - num_warmup_steps) / float(max(1, num_training_steps - num_warmup_steps))
        cosine_decay = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr_ratio + (1.0 - min_lr_ratio) * cosine_decay
    
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda, last_epoch)

# 使用改进的调度器
min_lr_ratio = lr_scheduler_config["min_lr"] / lr_scheduler_config["max_lr"]
scheduler = get_cosine_schedule_with_warmup(
    optimizer, 
    num_warmup_steps=warmup_steps,
    num_training_steps=total_training_steps,
    min_lr_ratio=min_lr_ratio
)

# 如果需要使用带重启的余弦退火，可以使用以下代码替代上面的scheduler
# scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
#     optimizer, 
#     T_0=steps_per_epoch,  # 每个epoch重启一次
#     T_mult=1,             # 每次重启后周期长度的倍数
#     eta_min=lr_scheduler_config["min_lr"]  # 最小学习率
# )

# 更新SwanLab配置中的学习率相关信息
if rank == 0:
    # 在SwanLab配置中添加学习率调度信息
    swanlab_config_update = {
        "learning_rate_max": lr_scheduler_config["max_lr"],
        "learning_rate_min": lr_scheduler_config["min_lr"],
        "scheduler_type": lr_scheduler_config["scheduler_type"],
        "warmup_steps": warmup_steps,
        "total_training_steps": total_training_steps,
        "steps_per_epoch": steps_per_epoch,
    }
    


# 初始化混合精度训练的GradScaler
# 混合精度训练默认启用
if mixed_precision_dtype == torch.float16:
    scaler = GradScaler(enabled=True)
    logger.info("使用 GradScaler 进行 float16 混合精度训练.")
elif mixed_precision_dtype == torch.bfloat16:
    scaler = None # bfloat16 通常不需要 scaler
    logger.info("使用 bfloat16 混合精度训练 (通常不需要 GradScaler).")
else:
    scaler = None # 作为后备
    logger.warning(f"未知的 mixed_precision_dtype: {mixed_precision_dtype}。不使用 GradScaler。")

#! 在设置完分布式训练环境后，初始化SwanLab（只在主进程）
if rank == 0:
    # 初始化SwanLab实验跟踪
    logger.info("初始化SwanLab实验跟踪...")
    swanlab_run = swanlab.init(
        # 设置项目名称
        project="bitbrain-pretrain",
        # 设置实验名称（可选）
        experiment_name=f"qwen2-pretrain-{time.strftime('%Y%m%d_%H%M%S')}",
        # 记录超参数和实验配置
        config={
            # 模型相关参数
            "model_id": model_id,
            "total_params_M": total_params / 1e6,
            "vocab_size": vocab_size,
            "seq_len": seq_len,
            
            # 训练相关参数
            "batch_size": args.batch_size,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "effective_batch_size": args.batch_size * gradient_accumulation_steps,
            "global_effective_batch_size": args.batch_size * gradient_accumulation_steps * world_size,
            "learning_rate": lr_scheduler_config["max_lr"],
            "num_epochs": 1,  # 这里我们设置为训练的轮数
            
            # 分布式训练参数
            "world_size": world_size,
            "is_distributed": is_distributed,
            
            # 优化相关参数
            "mixed_precision_dtype": str(mixed_precision_dtype),
            "gradient_checkpointing": is_gradient_checkpointing_enabled,
            "torch_compile": is_model_compiled,
            "compile_mode": compile_mode,
            
            # 硬件相关
            "device": device,
            "gpu_count": world_size,
        },
        # 添加实验描述
        description="Qwen2-0.5B预训练实验，使用分布式训练和混合精度"
    )
    logger.info("SwanLab实验跟踪初始化完成")
else:
    swanlab_run = None

logger.info(f"训练配置:")
logger.info(f"  - 分布式训练: {is_distributed}")
if is_distributed:
    logger.info(f"  - World size: {world_size}")
    logger.info(f"  - Rank: {rank}")
logger.info(f"  - 梯度累积步数: {gradient_accumulation_steps}")
logger.info(f"  - 混合精度训练: 已启用")
logger.info(f"  - 混合精度数据类型: {mixed_precision_dtype}")
logger.info(f"  - 梯度检查点状态: {'已启用' if is_gradient_checkpointing_enabled else '尝试启用失败或模型不支持'}")
logger.info(f"  - Torch compile 状态: {'已启用 (模式: ' + compile_mode + ')' if is_model_compiled else ('尝试编译失败或PyTorch/模型不支持 (尝试模式: ' + compile_mode + ')' if hasattr(torch, 'compile') else 'PyTorch版本不支持 torch.compile')}")
logger.info(f"  - 单GPU有效批处理大小: {train_loader.batch_size * gradient_accumulation_steps}")
logger.info(f"  - 全局有效批处理大小: {train_loader.batch_size * gradient_accumulation_steps * world_size}")

#! (5) 修改训练循环支持分布式训练和SwanLab记录
def train(model, optimizer, scheduler, train_loader, device,
           epoch, scaler=None, gradient_accumulation_steps=4, token_per_step=0):
    model.train()
    total_loss = 0
    accumulated_loss = 0
    
    if is_distributed and hasattr(train_loader.sampler, 'set_epoch'):
        train_loader.sampler.set_epoch(epoch)
    
    epoch_start_time = time.time()
    step_start_time = None
    first_batch_done = False
    
    # 用于记录epoch内的统计信息
    epoch_step_count = 0
    epoch_total_loss = 0
    
    for batch_idx, (x, y, loss_mask) in enumerate(train_loader):
        if (batch_idx) % gradient_accumulation_steps == 0:
            step_start_time = time.time()
            if not first_batch_done and is_model_compiled: # 使用 is_model_compiled
                logger.info("开始首次前向传播 (模型已编译)，可能包含编译开销...")
        
        x, y, loss_mask = x.to(device), y.to(device), loss_mask.to(device)
        
        #! 前向传播 (混合精度默认启用)
        with autocast(device_type='cuda', dtype=mixed_precision_dtype):
            outputs = model(input_ids=x, labels=y)
            loss = outputs.loss
            loss = loss / gradient_accumulation_steps

        #! 反向传播 (混合精度默认启用)
        if scaler is not None: # 意味着是 float16
            scaler.scale(loss).backward()
        else: # bfloat16 (或 float16 但 scaler 配置不当)
            loss.backward()
        
        accumulated_loss += loss.item()
        
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            if scaler is not None: # float16
                scaler.unscale_(optimizer)
                total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                if torch.isfinite(total_norm):
                    scaler.step(optimizer)
                else:
                    logger.warning(f"Skipping optimizer step at epoch {epoch}, batch_idx {batch_idx} due to non-finite gradients (norm: {total_norm}).")
                scaler.update() # 无论是否跳过step，都需要update scaler
            else: # bfloat16 (或未使用scaler)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            optimizer.zero_grad()
            scheduler.step()
            
            step_end_time = time.time()
            step_time = step_end_time - step_start_time
            
            # 更新epoch统计
            epoch_step_count += 1
            epoch_total_loss += accumulated_loss
            
            if not first_batch_done and is_model_compiled: # 使用 is_model_compiled
                first_batch_done = True
                logger.info(f"首次步骤完成 (模型已编译，包含编译开销): {step_time:.3f}s")
            
            if ((batch_idx + 1) // gradient_accumulation_steps) % 10 == 0:
                tokens_per_sec = token_per_step / step_time if step_time > 0 else 0
                mfu = calculate_mfu_distributed(
                    token_per_step, step_time, model_flops, 
                    single_gpu_peak_flops, world_size
                ) if step_time > 0 and model_flops > 0 else 0
                actual_flops_per_sec = (token_per_step * model_flops) / step_time if step_time > 0 else 0
                current_step = (batch_idx + 1) // gradient_accumulation_steps
                total_steps = len(train_loader) // gradient_accumulation_steps
                current_lr = scheduler.get_last_lr()[0]
                
                if rank == 0:
                    log_info = [
                        f'Epoch:[{epoch+1}/{num_epochs}]',
                        f'Step:[{current_step}/{total_steps}]',
                        f'Loss:{accumulated_loss:.4f}',
                        f'LR:{current_lr:.8f}',
                        f'Tokens/s:{tokens_per_sec:.0f}',
                        f'StepTime:{step_time:.3f}s',
                        f'MFU:{mfu*100:.2f}%',
                        f'FLOPS:{actual_flops_per_sec/1e12:.2f}T',
                        f'GPUs:{world_size}'
                    ]
                    log_message = ' '.join([info for info in log_info if info])
                    original_logger_info(log_message)
                    
                    # 使用SwanLab记录训练指标
                    if swanlab_run:
                        # 计算全局步数
                        global_step = epoch * (len(train_loader) // gradient_accumulation_steps) + current_step
                        
                        swanlab.log({
                            # 训练指标
                            "train/loss": accumulated_loss,
                            "train/learning_rate": current_lr,
                            "train/epoch": epoch + 1,
                            "train/step": current_step,
                            "train/global_step": global_step,
                            
                            # 性能指标
                            "performance/tokens_per_second": tokens_per_sec,
                            "performance/mfu_percent": mfu * 100,
                            "performance/step_time_seconds": step_time,
                            "performance/flops_tera": actual_flops_per_sec / 1e12,
                            
                            # 硬件指标
                            "hardware/gpu_count": world_size,
                            "hardware/batch_size": args.batch_size,
                            "hardware/effective_batch_size": args.batch_size * gradient_accumulation_steps * world_size,
                        })
            
            accumulated_loss = 0
            if args.save_interval > 0 and (batch_idx + 1) % args.save_interval == 0 and (rank == 0):
                # 确保保存目录存在
                os.makedirs(args.save_dir, exist_ok=True)

                timestamp = time.strftime("%Y%m%d_%H%M%S", time.localtime())
                # 创建一个更具描述性的检查点文件名
                checkpoint_filename = f'pretrain_epoch{epoch+1}_batch{batch_idx+1}_{timestamp}.pth'
                checkpoint_path = os.path.join(args.save_dir, checkpoint_filename)

                # 获取模型状态字典，处理分布式训练的情况
                if isinstance(model, torch.nn.parallel.DistributedDataParallel):
                    model_state_dict_to_save = model.module.state_dict()
                else:
                    model_state_dict_to_save = model.state_dict()

                # 获取当前学习率
                # scheduler.get_last_lr() 返回一个列表，取第一个元素
                current_lr_at_save = scheduler.get_last_lr()[0] if scheduler.get_last_lr() else optimizer.param_groups[0]['lr']

                # 构建检查点字典
                checkpoint = {
                    "epoch": epoch + 1,  # 保存1-based的epoch号
                    "batch_idx": batch_idx + 1,  # 保存1-based的当前epoch的批次号
                    "current_optimizer_step_in_epoch": (batch_idx + 1) // gradient_accumulation_steps, # 当前epoch内的优化器步数
                    "model_state_dict": model_state_dict_to_save,
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "scaler_state_dict": scaler.state_dict() if scaler else None, # 保存GradScaler状态（如果使用）
                    "config": config_to_save,  # 保存模型配置，用于后续加载
                    "gradient_accumulation_steps": gradient_accumulation_steps,
                    "mixed_precision_dtype_used": str(mixed_precision_dtype), # 保存使用的混合精度类型 (如 'torch.bfloat16')
                    "gradient_checkpointing_status": is_gradient_checkpointing_enabled,
                    "torch_compile_status": is_model_compiled,
                    "torch_compile_mode_attempted": compile_mode,
                    "world_size": world_size, # 保存分布式训练的world_size
                    "current_learning_rate": current_lr_at_save, # 保存当前学习率
                    "args": vars(args) # 保存所有命令行参数，方便复现和恢复
                }
                
                # 保存检查点
                torch.save(checkpoint, checkpoint_path)
                # 使用原始logger记录，因为它只在主进程执行
                original_logger_info(f"Epoch中途检查点已保存至: {checkpoint_path} (Epoch {epoch+1}, Batch {batch_idx+1})")

                # 如果启用了SwanLab，记录保存事件
                if swanlab_run:
                    # steps_per_epoch = len(train_loader) // gradient_accumulation_steps (在主脚本中定义)
                    # global_optimizer_step = epoch * steps_per_epoch + ((batch_idx + 1) // gradient_accumulation_steps)
                    swanlab.log({
                        "checkpoint/in_epoch_saved": True,
                        "checkpoint/in_epoch_path": checkpoint_path,
                        "checkpoint/in_epoch_epoch": epoch + 1,
                        "checkpoint/in_epoch_batch_idx": batch_idx + 1,
                        # "checkpoint/in_epoch_global_optimizer_step": global_optimizer_step, # 如果需要可以计算并记录
                    })
                
                # 确保模型返回训练模式 (虽然通常torch.save不改变模式，但明确一下是好习惯)
                # 这行代码你已经有了，保持即可
                model.train()

    epoch_end_time = time.time()
    epoch_duration = epoch_end_time - epoch_start_time
    global total_training_time
    total_training_time += epoch_duration
    avg_tokens_per_sec = total_tokens_processed / total_training_time if total_training_time > 0 else 0
    
    # 记录epoch级别的指标
    if rank == 0 and swanlab_run:
        avg_epoch_loss = epoch_total_loss / epoch_step_count if epoch_step_count > 0 else 0
        swanlab.log({
            "epoch/duration_seconds": epoch_duration,
            "epoch/avg_loss": avg_epoch_loss,
            "epoch/avg_tokens_per_second": avg_tokens_per_sec,
            "epoch/steps_count": epoch_step_count,
        })
    
    logger.info(
        f"Epoch {epoch} 完成 - "
        f"用时: {epoch_duration:.2f}s, "
        f"平均处理速度: {avg_tokens_per_sec:.0f} tokens/s"
    )
    return total_loss

def eval(model, val_loader, device):
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for x, y, loss_mask in val_loader:
            x, y, loss_mask = x.to(device), y.to(device), loss_mask.to(device)
            
            # 混合精度默认启用
            with autocast(device_type='cuda', dtype=mixed_precision_dtype):
                outputs = model(input_ids=x, labels=y)
                loss = outputs.loss
                
            val_loss += loss.item()
    
    if is_distributed:
        val_loss_tensor = torch.tensor(val_loss, device=device)
        dist.all_reduce(val_loss_tensor, op=dist.ReduceOp.SUM)
        val_loss = val_loss_tensor.item() / world_size
    
    return val_loss

# 训练主循环
num_epochs = 1
logger.info(f"Starting pretraining for {num_epochs} epochs...")

# 记录总训练开始时间
total_start_time = time.time()

# 分布式训练中每个进程处理的token数量
tokens_per_step = 2048 * args.batch_size * gradient_accumulation_steps

for epoch in range(num_epochs):
    logger.info(f"Starting epoch {epoch + 1}/{num_epochs}")
    
    train_loss = train(model, optimizer, scheduler, train_loader, device, epoch, scaler, gradient_accumulation_steps, 
                       token_per_step=tokens_per_step)
    #val_loss = eval(model, val_loader, device)
    
    avg_train_loss = train_loss 
    #avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0 # Guard against empty val_loader
    
    current_time = time.time()
    elapsed_time = current_time - total_start_time
    # total_tokens_processed needs to be updated in train loop for this to be accurate.
    # This variable is global but not updated in the provided snippet.
    # Assume total_tokens_processed is correctly updated elsewhere or this is for future.
    overall_tokens_per_sec = total_tokens_processed / elapsed_time if elapsed_time > 0 else 0
    
    # 使用SwanLab记录验证指标
    if rank == 0 and swanlab_run:
        swanlab.log({
            #"validation/loss": avg_val_loss,
            "validation/epoch": epoch + 1,
            "summary/total_time_seconds": elapsed_time,
            "summary/total_tokens_processed": total_tokens_processed,
            "summary/overall_tokens_per_second": overall_tokens_per_sec,
        })
    
    logger.info(
        f"Epoch: {epoch + 1}, "
        # f"Train Loss: {avg_train_loss:.4f}, " # This will be 0.0000 with current train() return
        #f"Val Loss: {avg_val_loss:.4f}, "
        f"总用时: {elapsed_time:.2f}s, "
        f"总处理tokens: {total_tokens_processed:,}, " # Needs update
        f"整体吞吐量: {overall_tokens_per_sec:.0f} tokens/s" # Needs update
    )

    if rank == 0:
        model_state_dict = model.module.state_dict() if hasattr(model, 'module') else model.state_dict()
        
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": model_state_dict,
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "scaler_state_dict": scaler.state_dict() if scaler else None,
            "config": config_to_save,
            "gradient_accumulation_steps": gradient_accumulation_steps,
            "mixed_precision_dtype_used": mixed_precision_dtype, # 保存使用的混合精度类型
            "gradient_checkpointing_status": is_gradient_checkpointing_enabled, # 保存梯度检查点状态
            "torch_compile_status": is_model_compiled, # 保存编译状态
            "torch_compile_mode_attempted": compile_mode, # 保存尝试的编译模式
        }
        
        import os
        os.makedirs("checkpoints", exist_ok=True)
        
        checkpoint_path = f"checkpoints/qwen2_pretrain_epoch_{epoch}.pt"
        torch.save(checkpoint, checkpoint_path)
        
        # 使用SwanLab记录检查点保存事件
        if swanlab_run:
            swanlab.log({
                "checkpoint/saved": True,
                "checkpoint/path": checkpoint_path,
                "checkpoint/epoch": epoch + 1,
            })
        
        original_logger_info(f"Checkpoint saved to {checkpoint_path}")

# 训练完成后的清理
if rank == 0 and swanlab_run:
    logger.info("训练完成，正在结束SwanLab实验记录...")
    # SwanLab会自动在进程结束时完成实验

if is_distributed:
    dist.destroy_process_group()


