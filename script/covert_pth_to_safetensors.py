import torch
from safetensors.torch import save_file
import os

# 输入的 .pth 文件路径
pytorch_model_path = "/home/chenyuhang/bit-brain/bitbrain/train/checkpoints/bitbrain_pretrain_epoch_1.pt"  # 例如: "./input_model/model.pth"

# 输出的 .safetensors 文件路径 (你希望保存转换后模型的位置)
safetensors_path = "/home/chenyuhang/bit-brain/bitbrain/models/Bitbran-0.6B-base"  # 例如: "./output_model/model.safetensors"

print(f"准备将 '{pytorch_model_path}' 转换为 '{safetensors_path}'")

print(f"正在从 {pytorch_model_path} 加载权重...")
try:
    # 加载 .pth 文件
    checkpoint = torch.load(pytorch_model_path, map_location='cpu',weights_only=False)
    print("权重加载成功！")
    print(f"检查点文件的键: {list(checkpoint.keys())}")
except Exception as e:
    print(f"加载 .pth 文件失败: {e}")
    print("请检查文件路径是否正确，以及文件是否损坏。")
    exit()

# --- 提取实际的模型权重 ---
# 判断是否包含 'model' 键
if 'model' in checkpoint:
    print("检测到检查点文件包含 'model' 键，正在提取实际的模型权重...")
    state_dict = checkpoint['model']
    print("已从 'model' 键中提取模型权重。")
elif isinstance(checkpoint, dict) and all(isinstance(v, torch.Tensor) for v in checkpoint.values()):
    print("检测到检查点文件直接包含模型权重（无嵌套结构）...")
    state_dict = checkpoint
    print("直接使用检查点文件中的权重。")
else:
    print("警告: 检查点文件结构不明确，尝试查找可能的权重字典...")
    # 尝试找到包含张量的字典
    tensor_dicts = []
    for key, value in checkpoint.items():
        if isinstance(value, dict) and all(isinstance(v, torch.Tensor) for v in value.values()):
            tensor_dicts.append((key, value))
    
    if len(tensor_dicts) == 1:
        key, state_dict = tensor_dicts[0]
        print(f"找到权重字典，键名为: '{key}'")
    elif len(tensor_dicts) > 1:
        print(f"找到多个可能的权重字典: {[key for key, _ in tensor_dicts]}")
        print("请手动指定要使用的键名。")
        exit()
    else:
        print("无法找到有效的模型权重。")
        exit()

# 验证 state_dict 的有效性
if not isinstance(state_dict, dict):
    print(f"错误: 提取的权重不是字典类型，而是 {type(state_dict)}")
    exit()

if not all(isinstance(v, torch.Tensor) for v in state_dict.values()):
    print("错误: 权重字典中包含非张量类型的值")
    exit()

print(f"成功提取模型权重，包含 {len(state_dict)} 个参数。")
print(f"参数示例: {list(state_dict.keys())[:5]}...")

# 添加权重名称清理功能
print("\n正在清理权重名称...")
cleaned_state_dict = {}

for name, tensor in state_dict.items():
    # 清理权重名称，去掉不必要的前缀
    cleaned_name = name
    
    # 去掉 'model._orig_mod.' 前缀
    if cleaned_name.startswith('model._orig_mod.'):
        cleaned_name = cleaned_name.replace('model._orig_mod.', '')
        print(f"清理权重名称: {name} -> {cleaned_name}")
    
    # 去掉 '_orig_mod.' 前缀（以防有其他情况）
    elif '_orig_mod.' in cleaned_name:
        cleaned_name = cleaned_name.replace('_orig_mod.', '')
        print(f"清理权重名称: {name} -> {cleaned_name}")
    
    cleaned_state_dict[cleaned_name] = tensor

print(f"权重名称清理完成，共处理 {len(cleaned_state_dict)} 个参数。")
print(f"清理后参数示例: {list(cleaned_state_dict.keys())[:5]}...")

print(f"\n正在将权重保存为 safetensors 格式到 {safetensors_path}...")
try:
    # 处理共享张量的问题
    # 创建一个新的状态字典，确保没有内存共享
    processed_state_dict = {}
    
    print("正在处理可能的共享张量...")
    for name, tensor in cleaned_state_dict.items():  # 使用清理后的权重字典
        # 创建张量的深拷贝，确保没有内存共享
        processed_state_dict[name] = tensor.clone().detach()
        
    print(f"已处理 {len(processed_state_dict)} 个参数，消除了内存共享问题。")
    
    # 确保输出路径是文件而不是目录
    if os.path.isdir(safetensors_path):
        # 如果是目录，添加默认文件名
        safetensors_file_path = os.path.join(safetensors_path, "model.safetensors")
    else:
        safetensors_file_path = safetensors_path
    
    # 确保目录存在
    os.makedirs(os.path.dirname(safetensors_file_path), exist_ok=True)
    
    # 保存为 .safetensors 文件
    save_file(processed_state_dict, safetensors_file_path)
    print(f"模型权重已成功保存为 {safetensors_file_path}")
    
except Exception as e:
    print(f"保存 .safetensors 文件失败: {e}")
    exit()

# --- 转换完成后的说明 ---
print("\n转换完成! 🎉")
print("\n重要提示:")
print(f"1. 你得到的 '{safetensors_path}' 文件现在包含纯粹的模型权重。")
print(f"   这是一个更安全、更高效的权重存储格式。")
print(f"2. 当你将来想要使用这个 '.safetensors' 文件加载模型时，你可以:")
print(f"   a. 使用 safetensors 库直接加载:")
print(f"      from safetensors.torch import load_file")
print(f"      state_dict = load_file('{safetensors_path}')")
print(f"      model.load_state_dict(state_dict)")
print(f"   b. 或者使用 Hugging Face Transformers (如果模型兼容):")
print(f"      from transformers import AutoModelForCausalLM")
print(f"      # 确保同目录下有 config.json 文件")
print(f"      model = AutoModelForCausalLM.from_pretrained('模型目录路径')")
print(f"3. safetensors 格式的优势:")
print(f"   - 更安全: 不会执行任意 Python 代码")
print(f"   - 更快速: 加载速度更快")
print(f"   - 跨平台: 更好的兼容性")
print(f"   - 内存友好: 支持内存映射加载")
