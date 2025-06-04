import torch
import sys
from pathlib import Path
import os

# 添加必要的路径，以便导入模型
parent_dir = Path(__file__).parent.parent.parent.resolve()
steel_model_dir = parent_dir / "pretrain_modify_from_TinyLlama" / "model" / "steel_modify_from_qwen_1_5"

# 添加tokenizer路径
tokenizer_dir = "/DATA/disk2/yuhang/.cache/modelscope/models/Qwen/Qwen2___5-0___5B-Instruct"

sys.path.append(str(parent_dir))
sys.path.append(str(steel_model_dir))

from transformers import AutoConfig, AutoTokenizer, TextStreamer, GenerationConfig,AutoModelForCausalLM

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用设备: {device}")
#! 
checkpoint_path = "bitbrain/train/out/pretrain_epoch1_batch45000_20250604_071749.pth"
#checkpoint_path = "/DATA/disk2/yuhang/.cache/ckpt/steel_llm/step-190000-iter-1520000-ckpt/state.pth"
print(f"从以下路径加载检查点: {checkpoint_path}")
state = torch.load(checkpoint_path, map_location=device)

print(f"从以下路径加载模型配置: {steel_model_dir}")
config_kwargs = {
    "trust_remote_code": True,
    "use_custom_rmsnorm": True,
}
if torch.cuda.is_available():
    config_kwargs["attn_implementation"] = "flash_attention_2"

config = AutoConfig.from_pretrained(
    str(steel_model_dir),
    **config_kwargs
)

# 使用配置中的 max_position_embeddings 作为 block_size
# 如果需要，可以覆盖： config.max_position_embeddings = new_value
print(f"模型配置加载完成。Block size (max_position_embeddings): {config.max_position_embeddings}")

print("初始化模型结构...")
model = AutoModelForCausalLM.from_config(config=config)
print("模型结构初始化完成。")

print("加载模型权重...")
model_state_dict = None
try:
    if 'model' in state:
        model_state_dict = state["model"]
    elif 'module' in state and 'model' in state['module']:
        model_state_dict = state['module']['model']
        print("从 state['module']['model'] 加载权重 (可能来自 DDP 检查点).")
    elif 'state_dict' in state:
         model_state_dict = state["state_dict"]
         print("从 state['state_dict'] 加载权重.")
    else:
        is_model_state_dict_at_top_level = all(
            any(key.startswith(p_name) for p_name in dict(model.named_parameters()).keys()) or
            any(p_name.startswith(key) for p_name in dict(model.named_parameters()).keys())
            for key in state.keys()
        )
        if len(state.keys()) > 0 and is_model_state_dict_at_top_level :
             print("警告: 'model' key 未在检查点中找到。尝试直接从顶层加载权重。")
             model_state_dict = state
        else:
            raise KeyError(f"'model' key 或可识别的模型状态字典未在检查点中找到。可用 keys: {state.keys()}")

    if model_state_dict is None:
        raise ValueError("未能从检查点中提取 model_state_dict。")

    load_result = model.load_state_dict(model_state_dict, strict=True)
    print(f"模型权重加载成功: {load_result}")
except Exception as e:
    print(f"加载模型权重时出错 (strict=True): {e}")
    if model_state_dict is not None:
        print("尝试使用 strict=False 进行加载...")
        try:
            load_result = model.load_state_dict(model_state_dict, strict=False)
            print(f"模型权重使用 strict=False 加载成功: {load_result}")
            if load_result.missing_keys:
                print(f"缺失的键: {load_result.missing_keys}")
            if load_result.unexpected_keys:
                print(f"非预期的键: {load_result.unexpected_keys}")
        except Exception as e2:
            print(f"使用 strict=False 加载模型权重时再次出错: {e2}")
            print("请检查检查点文件和模型结构是否匹配。")
            sys.exit(1)
    else:
        print("无法尝试 strict=False 加载，因为 model_state_dict 未被提取。")
        sys.exit(1)

model.to(device)
print(f"模型已移至设备: {device}")
print(f"模型原始数据类型: {next(model.parameters()).dtype}")
model = model.half()
print(f"模型已转换为半精度 (fp16)。当前数据类型: {next(model.parameters()).dtype}")
model.eval()
print("模型已设置为评估模式。")

print(f"从以下路径加载Tokenizer: {tokenizer_dir}")
tokenizer = AutoTokenizer.from_pretrained(
    str(tokenizer_dir),
    trust_remote_code=True,
    use_fast=True
)
if tokenizer.pad_token is None:
    if tokenizer.eos_token is not None:
        print(f"Tokenizer pad_token 未设置，将使用 eos_token ('{tokenizer.eos_token}') 作为 pad_token。")
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.pad_token_id = tokenizer.eos_token_id # 确保ID也同步
    else:
        print("警告: Tokenizer pad_token 和 eos_token 均未设置。添加一个默认的 pad_token '<|pad|>'")
        tokenizer.add_special_tokens({'pad_token': '<|pad|>'}) # 使用Qwen风格的特殊token格式
        model.resize_token_embeddings(len(tokenizer))
        # 确保config中的pad_token_id也更新，如果模型实现依赖它
        if hasattr(model.config, 'pad_token_id'):
            model.config.pad_token_id = tokenizer.pad_token_id


print(f"Tokenizer pad_token: '{tokenizer.pad_token}', ID: {tokenizer.pad_token_id}")
print(f"Tokenizer eos_token: '{tokenizer.eos_token}', ID: {tokenizer.eos_token_id}")

def build_prompt(history):
    prompt_parts = []
    for i, (query, response) in enumerate(history):
        prompt_parts.append(f"用户: {query}")
        if response: # 只有当回复存在时才添加
            prompt_parts.append(f"助手: {response}")

    current_prompt = "\n".join(prompt_parts)

    if history and not history[-1][1]: # 如果最后一轮是用户说的，且助手还没回复
        current_prompt += f"\n助手:"
    elif not history: # 如果历史为空（例如，直接开始一个新问题给助手）
        # 这部分可以根据需要调整，例如如果第一个输入就是问题，则不需要"助手:"
        pass # 或者 current_prompt = "助手:" 如果希望模型直接开始

    # 根据用户要求，为预训练测试添加 <|im_end|>
    # 注意: 如果 <|im_end|> 是 EOS token，模型可能在看到它后立即停止生成。
    current_prompt += "<|im_end|>"
    return current_prompt

@torch.no_grad()
def generate(current_model, current_tokenizer, prompt_text, max_new_tokens=100, temperature=0.8, top_p=0.9):
    input_ids = current_tokenizer.encode(prompt_text, return_tensors="pt", truncation=True, max_length=current_model.config.max_position_embeddings).to(device)

    # 初始化 streamer 为 None
    streamer_obj = None # 使用不同的变量名以示区分
    try:
        streamer_obj = TextStreamer(current_tokenizer, skip_prompt=True, skip_special_tokens=True)
        # print("TextStreamer 初始化成功。") # 用于调试
    except ImportError:
        print("transformers.TextStreamer 未找到或导入失败，不进行流式输出。")
    except Exception as e_streamer:
        print(f"初始化 TextStreamer 时出错: {e_streamer}，不进行流式输出。")

    generation_config = GenerationConfig(
        max_new_tokens=max_new_tokens,
        temperature=temperature if temperature > 0.01 else None,
        top_p=top_p if temperature > 0.01 else None,
        do_sample=True if temperature > 0.01 else False,
        pad_token_id=current_tokenizer.pad_token_id,
        eos_token_id=current_tokenizer.eos_token_id,
        repetition_penalty=1.05,
        use_cache=True
    )

    output_ids = current_model.generate(
        input_ids,
        generation_config=generation_config,
        streamer=streamer_obj # 直接传递 streamer_obj (可以是 None 或 TextStreamer 实例)
    )

    if streamer_obj is None: # 如果未使用流式输出 (streamer_obj 保持为 None)
        generated_ids = output_ids[0, input_ids.size(1):]
        generated_text = current_tokenizer.decode(generated_ids, skip_special_tokens=True)
        return generated_text
    else: # 如果使用了流式输出
        return "" # 流式输出时，文本已打印到控制台

def clear_screen():
    os.system('cls' if os.name == 'nt' else 'clear')

def interactive_chat():
    history = []
    clear_screen()
    print("=" * 50)
    print("欢迎使用 Steel-LLM 交互式对话系统")
    print("输入'退出'或'exit'结束对话")
    print("输入'清空'或'clear'清除对话历史")
    print("输入'设置'或'settings'调整生成参数")
    print("提示: 所有用户输入将用于构建一个包含<|im_end|>的提示以测试模型。")
    print("=" * 50)

    generation_params = {
        "max_new_tokens": 200,
        "temperature": 0.7,
        "top_p": 0.9
    }

    while True:
        user_input = input("\n用户: ").strip()
        if not user_input:
            continue
        if user_input.lower() in ['退出', 'exit', 'quit', 'q']:
            print("谢谢使用，再见！")
            break
        if user_input.lower() in ['清空', 'clear']:
            history = []
            clear_screen()
            print("对话历史已清空")
            continue
        if user_input.lower() in ['设置', 'settings']:
            try:
                temp_str = input(f"请输入温度参数(0.0-1.0，当前值: {generation_params['temperature']:.1f}, 按回车跳过): ").strip()
                if temp_str: generation_params["temperature"] = max(0.0, min(1.0, float(temp_str)))
                tokens_str = input(f"请输入最大生成长度(10-1024，当前值: {generation_params['max_new_tokens']}, 按回车跳过): ").strip()
                if tokens_str: generation_params["max_new_tokens"] = max(10, min(1024, int(tokens_str)))
                print(f"参数已更新: 温度={generation_params['temperature']:.2f}, 最大长度={generation_params['max_new_tokens']}")
            except ValueError:
                print("输入无效，保持原有设置")
            continue

        history.append((user_input, "")) # 助手回复暂时为空
        prompt = build_prompt(history)

        print(f"\n>>> 实际输入到模型的提示文本:\n---\n{prompt}\n---")
        print("\n助手: ", end="", flush=True)

        response_text = generate(
            model,
            tokenizer,
            prompt,
            max_new_tokens=generation_params["max_new_tokens"],
            temperature=generation_params["temperature"],
            top_p=generation_params["top_p"]
        )

        # 如果 generate 函数返回了文本 (即非流式)，则打印它
        if response_text: # "" (空字符串) 在布尔上下文中为 False
            print(response_text, end="", flush=True) # end="" 以防 TextStreamer 也打印换行
        print() # 确保在流式或非流式输出后都有换行

        cleaned_response = response_text.split("<|im_end|>")[0].strip() if response_text else ""
        history[-1] = (user_input, cleaned_response) # 更新历史记录中的助手回复

if __name__ == "__main__":
    print("\n" + "="*50)
    print("测试模型能力（每个输入都会添加<|im_end|>标记以测试预训练模型）:")
    print("="*50)

    test_cases = [
        ("你好，请介绍一下你自己", "测试对话格式:"),
        ("北京是中国的首都", "测试纯文本理解能力:"),
        ("请介绍一下人工智能的发展历程", "测试知识问答能力:")
    ]

    for test_input, test_name in test_cases:
        print(f"\n{test_name}")
        current_history = [] # 为每个测试用例重置历史
        if "助手:" not in test_input:
            current_history.append((test_input, ""))
        else:
            parts = test_input.split("助手:")
            user_q = parts[0].replace("用户:", "").strip()
            current_history.append((user_q, "")) # 假设助手部分是空的，让build_prompt处理

        test_prompt_built = build_prompt(current_history)

        print(f"用户输入（用于构建提示）: {test_input}")
        print(f">>> 实际输入到模型的提示文本:\n---\n{test_prompt_built}\n---")
        print("助手: ", end="", flush=True)

        test_response = generate(model, tokenizer, test_prompt_built, max_new_tokens=100, temperature=0.7)

        # 如果 generate 函数返回了文本 (即非流式)，则打印它
        if test_response: # "" (空字符串) 在布尔上下文中为 False
            print(test_response, end="", flush=True)
        print() # 确保换行
        print("-" * 30)

    print("\n" + "="*50)
    input("按回车键开始交互式对话...")
    # clear_screen() # 在进入交互前可以选择清屏
    print("="*50)
    interactive_chat()