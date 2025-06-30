import torch
from safetensors.torch import load_file

safetensors_path = "/DATA/disk2/yuhang/.cache/ckpt/steel_llm/sft_ckpt/Code-Feedback_Infinity-Instruct_7M_OpenHermes_190000/checkpoint-47535/model.safetensors" # 例如: "./output_model/adapter_model.safetensors" 或 "./output_model/pytorch_model.safetensors"

# 输出的 .pth 文件路径 (你希望保存转换后模型的位置)
pytorch_model_path = "/DATA/disk2/yuhang/.cache/ckpt/steel_llm/sft_pth/checkpoint-47535.pth" # 例如: "./output_model/converted_model.pth"

print(f"准备将 '{safetensors_path}' 转换为 '{pytorch_model_path}'")


print(f"正在从 {safetensors_path} 加载权重...")
try:
    # state_dict 是一个 Python 字典，它包含了模型所有可学习的参数 (权重和偏置)。
    # 每一项的键是参数的名称 (例如 'layer.0.attention.self.query.weight')，值是参数对应的张量 (tensor)。
    state_dict = load_file(safetensors_path)
    print("权重加载成功！")
except Exception as e:
    print(f"加载 .safetensors 文件失败: {e}")
    print("请检查文件路径是否正确，以及文件是否损坏。")
    exit()

# --- 新增代码开始 ---
# 为了解决 "'model' key 未找到" 的错误，我们需要将加载的 state_dict 包装一下。
# 很多模型加载工具 (例如某些 Hugging Face Transformers 或 LlamaFactory 的加载流程)
# 期望在检查点文件中，实际的模型权重是存储在一个名为 'model' 的键 (key) 下的。
# 所以，我们创建一个新的字典，其中包含一个 'model' 键，
# 这个键对应的值就是我们从 .safetensors 文件中加载出来的 state_dict。
# 想象一下，state_dict 是我们要寄送的货物，structured_state_dict 就是一个贴了 "model" 标签的包裹，里面装着货物。
structured_state_dict = {'model': state_dict}
print(f"已将权重构建为 {{'model': actual_weights}} 的结构，以便兼容常见的加载方式。")
# --- 新增代码结束 ---


print(f"\n正在将结构化的权重保存到 {pytorch_model_path}...")
try:
    # 现在我们保存的是 structured_state_dict，而不是直接保存 state_dict。
    torch.save(structured_state_dict, pytorch_model_path)
    print(f"模型权重已成功保存为 {pytorch_model_path}")
except Exception as e:
    print(f"保存 .pth 文件失败: {e}")
    exit()

# --- 转换完成后的说明 ---
print("\n转换完成! 🎉")
print("\n重要提示:")
# --- 修改说明开始 ---
print(f"1. 你得到的 '{pytorch_model_path}' 文件现在包含一个 Python 字典。")
print(f"   这个字典里有一个键叫做 'model'，这个 'model' 键对应的值才是真正的模型权重 (state_dict)。")
print(f"   文件的结构大致是这样的：{{'model': {{'param1_name': param1_tensor, ...}} }}")
print(f"2. 当你将来想要使用这个 '.pth' 文件加载模型时，你需要：")
print(f"   a. 首先，根据原始模型的结构创建一个模型实例。")
print(f"      例如，使用 Hugging Face Transformers 库:")
print(f"      from transformers import AutoModelForCausalLM")
print(f"      model = AutoModelForCausalLM.from_pretrained('你的基础模型名称或路径或配置文件路径') # 替换为正确的模型配置来源")
print(f"   b. 然后，加载这个 '.pth' 文件中的权重到模型实例中:")
print(f"      # 首先，使用 torch.load() 加载整个 .pth 文件，这时你会得到包含 'model' 键的字典")
print(f"      checkpoint = torch.load('{pytorch_model_path}')")
print(f"      # 接着，从这个字典中提取出 'model' 键对应的值，这部分才是模型实际的权重")
print(f"      model_weights = checkpoint['model']")
print(f"      # 最后，将提取出的模型权重加载到你的模型实例中")
print(f"      model.load_state_dict(model_weights)")
print(f"      # 有些时候，你也可以简化为一行 (如果确定加载器不会做更复杂的事情):")
print(f"      # model.load_state_dict(torch.load('{pytorch_model_path}')['model'])")
# --- 修改说明结束 ---
print(f"3. 关于LlamaFactory的输出:")
print(f"   - 如果你的 '{safetensors_path}' 是全参数微调后的完整模型权重，那么转换后的 '{pytorch_model_path}' 中 'model' 键对应的值也是完整的模型权重。")
print(f"   - 如果 '{safetensors_path}' 只包含LoRA等适配器权重，那么 '{pytorch_model_path}' 中的 'model' 键对应的值也只包含这些适配器权重。")
print(f"     在这种情况下，加载模型时，你需要先加载基础模型，然后应用这些适配器权重。")
print(f"     LlamaFactory 通常也提供了将LoRA权重合并到基础模型并保存为完整模型的功能，建议优先使用该功能得到完整权重的 .safetensors 文件再进行转换。")