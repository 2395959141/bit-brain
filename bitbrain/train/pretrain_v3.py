#！ 在梯度累计；混合精度训练；和梯度检查点的基础上，进一步添加训练过程中的训练时间和MFU利用率估算
import os
import sys
import argparse
import time  #! 添加时间模块用于性能监控
import math  #! 添加数学模块用于FLOPS计算
# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

import torch
from torch.utils.data import DataLoader
from torch.amp import GradScaler, autocast  #! 添加混合精度训练支持
from modelscope import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from bitbrain.dataset.pretrain_dataset_jsonl import PretrainDataset_v3,PretrainDataset_v2,PretrainDataset_v1
from loguru import logger

model_id = "Qwen/Qwen2-0.5B"

parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--seq_len", type=int, default=2048)
args = parser.parse_args()



#! (3) 加载与模型配置匹配的分词器
logger.info(f"Loading tokenizer for {model_id}...")
tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
logger.info(f"Tokenizer for {model_id} loaded successfully.")

##! (0) 加载并配置训练数据集
train_dataset = PretrainDataset_v1(data_path="/home/ytllm/.cache/pretrain_data/data_jsonl/00127.jsonl",
                                    tokenizer=tokenizer,
                                    max_length=2048)

train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [0.9, 0.1])
train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False)

#* 训练配置参数
gradient_accumulation_steps = 4  # 梯度累计步数，实际batch_size = batch_size * gradient_accumulation_steps
use_mixed_precision = True  # 是否使用混合精度训练
mixed_precision_dtype = torch.bfloat16  # 或者 torch.float16，参考 pretrain_v2.py
use_gradient_checkpointing = True  # 是否使用梯度检查点（重计算）来节省显存
use_torch_compile = True  # 是否使用 torch.compile 编译优化（PyTorch 2.0+）
compile_mode = "default"  # 编译模式：'default', 'reduce-overhead', 'max-autotune'


def get_gpu_peak_flops():
    """
    获取GPU理论峰值FLOPS
    针对混合精度训练优化，优先使用Tensor Core性能
    """
    if not torch.cuda.is_available():
        return 0
    
    # 获取GPU信息
    gpu_name = torch.cuda.get_device_name()
    
    # GPU FLOPS估算（优先使用Tensor Core性能用于混合精度训练）
    gpu_flops_dict = {
        'V100': 125e12,     # 125 TFLOPS (Tensor Core FP16)
        'A100': 312e12,     # 312 TFLOPS (Tensor Core BF16/FP16)
        'H100': 989e12,     # 989 TFLOPS (Tensor Core FP16)
        '3090': 142e12,     # 142 TFLOPS (Tensor Core FP16) - 也需要更新
        '4090': 330e12,     # 330 TFLOPS (Tensor Core BF16/FP16) ← 修正！
        'T4': 130e12,       # 130 TFLOPS (Tensor Core FP16) - 也需要更新
    }
    
    # 尝试匹配GPU型号
    for gpu_type, flops in gpu_flops_dict.items():
        if gpu_type in gpu_name:
            logger.info(f"检测到GPU: {gpu_name}, Tensor Core峰值FLOPS: {flops/1e12:.1f} TFLOPS")
            return flops
    
    # 如果未找到匹配的GPU，使用保守估算
    logger.warning(f"未识别的GPU型号: {gpu_name}, 使用默认估算值")
    return 100e12  # 提高默认值到100 TFLOPS

#! 添加性能估算函数
def estimate_model_flops(model):
    """
    估算Transformer模型每个token的理论FLOPs
    
    参数:
        model: 模型实例
    
    返回:
        每个token的理论FLOPs数值
    """
    # 获取模型配置参数
    n_params = sum(p.numel() for p in model.parameters())
    
    #! 模型计算量近似估算公式
    #! 对于训练（前向+后向），每个token的FLOPs通常估算为 6 * n_params。
    #! 代码中使用了经验系数8，可能考虑了梯度重计算等因素。我们保留这个系数。
    flops_per_token = 6 * n_params
    logger.info(f"估算的每个token的训练计算量 (FLOPs per token, based on 6 * n_params): {flops_per_token:.0f}")
    
    return flops_per_token

def calculate_mfu(toknes_processed, step_time, flops_per_token, peak_flops):
    """
    计算MFU（Model FLOPS Utilization）
    
    参数:
        toknes_processed: 处理的token数量
        step_time: 处理这些token所用的时间 (秒)
        flops_per_token: 每个token的理论FLOPs
        peak_flops: 硬件理论峰值FLOPS/秒
    
    返回:
        MFU利用率（0-1之间的数值）
    """
    if step_time <= 0 or peak_flops <= 0 or flops_per_token <= 0:
        return 0
    
    # 计算实际FLOPS
    actual_flops_per_sec = (toknes_processed * flops_per_token) / step_time
    mfu = actual_flops_per_sec / peak_flops
    # 计算MFU
    return min(mfu, 1.0) # MFU最高为100%



#! (1) 加载模型配置
logger.info(f"Loading configuration for {model_id} from ModelScope...")
qwen_config = AutoConfig.from_pretrained(model_id, trust_remote_code=True)
logger.info(f"Configuration for {model_id} loaded successfully.")

#* 打印配置中的最大位置嵌入（通常是最大上下文长度）
logger.info(f"Model's max_position_embeddings from config: {qwen_config.max_position_embeddings}")

#! (2) 从配置初始化新模型（权重随机初始化）
logger.info(f"Initializing a new model from configuration: {model_id} (training from scratch)...")
model = AutoModelForCausalLM.from_config(config=qwen_config)
logger.info(f"New model initialized successfully with random weights based on {model_id} configuration.")

#! 启用梯度检查点
if use_gradient_checkpointing:
    logger.info("Enabling gradient checkpointing to save memory...")
    try:
        model.gradient_checkpointing_enable()
        logger.info("Gradient checkpointing enabled successfully.")
    except AttributeError:
        logger.warning("Model does not support gradient checkpointing, continuing without it.")
        use_gradient_checkpointing = False


#* 用于保存检查点 
config_to_save = qwen_config

device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)

total_params = sum(p.numel() for p in model.parameters())
logger.info(f"Total parameters: {total_params / 1e6:.2f} M")

#! 添加 torch.compile 支持
if use_torch_compile:
    logger.info("开始编译模型以优化性能...")
    try:
        # 检查 PyTorch 版本是否支持 compile
        if hasattr(torch, 'compile'):
            compile_start_time = time.time()
            
            # 编译模型
            # 不同的编译模式说明：
            # - 'default': 平衡编译时间和运行时性能
            # - 'reduce-overhead': 减少 Python 开销，适合小批量训练
            # - 'max-autotune': 最大化优化，编译时间较长但性能最佳
            model = torch.compile(model, mode=compile_mode)
            
            compile_end_time = time.time()
            compile_duration = compile_end_time - compile_start_time
            
            logger.info(f"模型编译完成！编译模式: {compile_mode}, 编译耗时: {compile_duration:.2f}s")
            logger.info("注意：首次前向传播可能会有额外的编译开销")
        else:
            logger.warning("当前 PyTorch 版本不支持 torch.compile，跳过编译优化")
            use_torch_compile = False
    except Exception as e:
        logger.error(f"模型编译失败，将继续使用未编译版本: {e}")
        use_torch_compile = False

#! 添加性能监控初始化
# 获取序列长度（从数据集或配置中）
seq_len = qwen_config.max_position_embeddings if hasattr(qwen_config, 'max_position_embeddings') else 2048
vocab_size = qwen_config.vocab_size

# 估算模型FLOPS
model_flops = estimate_model_flops(model)

# 获取GPU理论峰值FLOPS
peak_flops = get_gpu_peak_flops()

# 性能统计变量
total_tokens_processed = 0
total_training_time = 0

#! (4) 设置优化器和学习率调度器
optimizer = torch.optim.AdamW(model.parameters(), lr=3e-4)
# todo 可以设置T_max根据训练的batch 和 训练集的样本数动态计算
scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=150000)

# 初始化混合精度训练的GradScaler
# 注意：对于 BFloat16，通常不需要 GradScaler，或者其行为不同。
# GradScaler 主要用于 Float16。
if use_mixed_precision and mixed_precision_dtype == torch.float16:
    scaler = GradScaler(enabled=True) # 在 pretrain_v3 中 'cuda' 作为 autocast target 足够
    logger.info("Using GradScaler for float16 mixed precision training")
elif use_mixed_precision and mixed_precision_dtype == torch.bfloat16:
    scaler = None # bfloat16 通常不需要 scaler
    logger.info("Using bfloat16 mixed precision training (scaler not typically needed or handled differently by autocast)")
else:
    scaler = None

logger.info(f"Training configuration:")
logger.info(f"  - Gradient accumulation steps: {gradient_accumulation_steps}")
logger.info(f"  - Mixed precision training: {use_mixed_precision}")
if use_mixed_precision:
    logger.info(f"  - Mixed precision dtype: {mixed_precision_dtype}")
logger.info(f"  - Gradient checkpointing: {use_gradient_checkpointing}")
logger.info(f"  - Torch compile: {use_torch_compile}")
if use_torch_compile:
    logger.info(f"  - Compile mode: {compile_mode}")
logger.info(f"  - Effective batch size: {train_loader.batch_size * gradient_accumulation_steps}")

#! (5) 修改训练循环添加性能监控
def train(model, optimizer, scheduler, train_loader, val_loader, device,
           epoch, scaler=None, gradient_accumulation_steps=4, token_per_step=0):
    model.train()
    total_loss = 0
    accumulated_loss = 0  # 用于跟踪累计的损失
    
    # 性能监控变量
    epoch_start_time = time.time()  # 记录epoch开始时间
    step_start_time = None  # 每个梯度更新步骤的开始时间
    first_batch_done = False  # 标记第一个批次是否完成（用于监控编译开销）
    
    for batch_idx, (x, y, loss_mask) in enumerate(train_loader):
        # 在梯度累积步骤开始时记录时间
        if (batch_idx) % gradient_accumulation_steps == 0:
            step_start_time = time.time()
            # 如果是第一个批次且使用了编译，记录额外信息
            if not first_batch_done and use_torch_compile:
                logger.info("开始首次前向传播，可能包含编译开销...")
        
        # 将数据移到设备上
        x, y, loss_mask = x.to(device), y.to(device), loss_mask.to(device)
        
        #! 前向传播
        if use_mixed_precision:
            with autocast(device_type='cuda', dtype=mixed_precision_dtype):
                outputs = model(input_ids=x, labels=y)
                loss = outputs.loss
                
                # 如果需要使用loss_mask来过滤损失计算
                # 这里可以根据实际需求修改损失计算方式
                loss = loss / gradient_accumulation_steps
        else:
            outputs = model(input_ids=x, labels=y)
            loss = outputs.loss
            # 梯度累计：将损失除以累计步数
            loss = loss / gradient_accumulation_steps

        #! 反向传播
        if use_mixed_precision and scaler is not None: # 通常意味着是 float16
            #* 使用scaler进行混合精度的反向传播 (float16)
            scaler.scale(loss).backward()
        elif use_mixed_precision: # 通常意味着是 bfloat16 (scaler is None) 或其他不需要显式scaler的场景
            loss.backward()
        else: # 不使用混合精度
            loss.backward()
        
        accumulated_loss += loss.item()
        
        #! 梯度累计：只有达到累计步数时才更新参数
        if (batch_idx + 1) % gradient_accumulation_steps == 0:
            if use_mixed_precision and scaler is not None: # Float16 路径
                #! 梯度裁剪(注意处理顺序)
                scaler.unscale_(optimizer) # 在裁剪前 unscale 梯度
                total_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                
                # 检查梯度是否包含无限值
                if torch.isfinite(total_norm):
                    # 更新参数
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    logger.warning(f"Skipping optimizer step at epoch {epoch}, batch_idx {batch_idx} due to non-finite gradients (norm: {total_norm}).")
                    # 即使跳过 optimizer.step()，也可能需要调用 scaler.update()，
                    # 这取决于具体的 PyTorch 版本和 GradScaler 的行为。
                    # 通常，如果 scaler.step() 没有被调用，scaler.update() 也不应该改变 scale factor。
                    # 但为了安全，如果跳过了 step，可以考虑是否需要调整 update 调用或 optimizer.zero_grad() 的位置。
                    # 这里我们先保持 scaler.update()，因为它在 pretrain_v2 中即使跳过也会调用。
                    # optimizer.zero_grad() 应该在更新参数后或跳过更新后执行。
                    scaler.update() # 确保 scaler 状态被更新
            else: # BFloat16 或不使用混合精度的路径
                # 梯度裁剪（推荐用于大模型训练）
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
            
            # 清零梯度
            optimizer.zero_grad()
            # 调整学习率
            scheduler.step()
            
            # 计算单个梯度更新步骤的时间
            step_end_time = time.time()
            step_time = step_end_time - step_start_time  # 现在是单步时间！
            
            # 标记第一个批次完成，并记录编译影响
            if not first_batch_done and use_torch_compile:
                first_batch_done = True
                logger.info(f"首次步骤完成（包含编译开销）: {step_time:.3f}s")
            
            # 计算累计步骤的token数量
            
            # 每10个累计步骤打印详细性能信息
            if ((batch_idx + 1) // gradient_accumulation_steps) % 10 == 0:
                # 计算性能指标（现在使用正确的单步时间）
                tokens_per_sec = token_per_step / step_time if step_time > 0 else 0
                
                actual_flops_per_sec = 0
                mfu = 0
                if step_time > 0 and model_flops > 0:
                    actual_flops_per_sec = (token_per_step * model_flops) / step_time
                    if peak_flops > 0:
                        mfu = actual_flops_per_sec / peak_flops
                        mfu = min(mfu, 1.0)
                
                # 计算相关变量
                current_step = (batch_idx + 1) // gradient_accumulation_steps
                total_steps = len(train_loader) // gradient_accumulation_steps
                current_lr = scheduler.get_last_lr()[0]
                
                # 按照参考格式重新组织日志信息
                log_info = [
                    f'Epoch:[{epoch+1}/{num_epochs}]',
                    f'Step:[{current_step}/{total_steps}]',
                    f'Loss:{accumulated_loss:.4f}',
                    f'LR:{current_lr:.8f}',
                    f'Tokens/s:{tokens_per_sec:.0f}',
                    f'StepTime:{step_time:.3f}s',
                    f'MFU:{mfu*100:.2f}%',
                    f'FLOPS:{actual_flops_per_sec/1e12:.2f}T'
                ]
                
                # 过滤空字符串并连接
                log_message = ' '.join([info for info in log_info if info])
                logger.info(log_message)
            
            # 重置累计损失
            accumulated_loss = 0
            # 注意：不需要重置step_start_time，因为下一个梯度累积步骤开始时会重新设置

    # 计算epoch统计信息
    epoch_end_time = time.time()
    epoch_duration = epoch_end_time - epoch_start_time  # 使用epoch_start_time而不是全局start_time
    
    global total_training_time
    total_training_time += epoch_duration
    
    avg_tokens_per_sec = total_tokens_processed / total_training_time if total_training_time > 0 else 0
    
    logger.info(
        f"Epoch {epoch} 完成 - "
        f"用时: {epoch_duration:.2f}s, "
        f"平均处理速度: {avg_tokens_per_sec:.0f} tokens/s"
    )

    return total_loss


def eval(model, val_loader, device):
    # 验证
    model.eval()
    val_loss = 0
    with torch.no_grad():
        for x, y, loss_mask in val_loader:  # 修改这里
            x, y, loss_mask = x.to(device), y.to(device), loss_mask.to(device)  # 修改这里
            
            # 验证时也可以使用混合精度来节省显存
            if use_mixed_precision:
                with autocast(device_type='cuda', dtype=mixed_precision_dtype):
                    outputs = model(input_ids=x, labels=y)
                    loss = outputs.loss
            else:
                outputs = model(input_ids=x, labels=y)
                loss = outputs.loss
                
            val_loss += loss.item()
    return val_loss

# todo 使用 if __name__ == "__main__": 来执行训练 
# todo 使用 parser.add_argument 来设置训练参数
# 训练主循环
num_epochs = 2
logger.info(f"Starting pretraining for {num_epochs} epochs...")

# 记录总训练开始时间
total_start_time = time.time()

# todo 这是每次梯度更新时处理的token数量
total_per_step = 2048 * args.batch_size * gradient_accumulation_steps

for epoch in range(num_epochs):
    logger.info(f"Starting epoch {epoch + 1}/{num_epochs}")
    
    train_loss = train(model, optimizer, scheduler, train_loader, val_loader, device, epoch, scaler, gradient_accumulation_steps, 
                       token_per_step=total_per_step)
    val_loss = eval(model, val_loader, device)
    
    #! 计算梯度累计下的有效batch数量
    effective_train_batches = len(train_loader) // gradient_accumulation_steps
    avg_train_loss = train_loss / effective_train_batches
    avg_val_loss = val_loss / len(val_loader)
    
    # 计算当前的整体性能统计
    current_time = time.time()
    elapsed_time = current_time - total_start_time
    overall_tokens_per_sec = total_tokens_processed / elapsed_time if elapsed_time > 0 else 0
    
    logger.info(
        f"Epoch: {epoch + 1}, "
        f"Train Loss: {avg_train_loss:.4f}, "
        f"Val Loss: {avg_val_loss:.4f}, "
        f"总用时: {elapsed_time:.2f}s, "
        f"总处理tokens: {total_tokens_processed:,}, "
        f"整体吞吐量: {overall_tokens_per_sec:.0f} tokens/s"
    )

    # 保存模型检查点
    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "scaler_state_dict": scaler.state_dict() if scaler else None,  # 保存scaler状态
        "config": config_to_save,
        "train_loss": avg_train_loss,
        "val_loss": avg_val_loss,
        "gradient_accumulation_steps": gradient_accumulation_steps,  # 保存训练配置
        "use_mixed_precision": use_mixed_precision,
        "use_gradient_checkpointing": use_gradient_checkpointing,  # 保存梯度检查点配置
        "use_torch_compile": use_torch_compile,  # 保存编译配置
        "compile_mode": compile_mode if use_torch_compile else None,
        #! 添加性能统计信息到检查点
        "total_tokens_processed": total_tokens_processed,
        "total_training_time": total_training_time,
        "overall_tokens_per_sec": overall_tokens_per_sec,
    }
    
    # 确保checkpoints目录存在
    import os
    os.makedirs("checkpoints", exist_ok=True)
    
    # 保存每个epoch的模型
    checkpoint_path = f"checkpoints/qwen2_pretrain_epoch_{epoch}.pt"
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Checkpoint saved to {checkpoint_path}")


