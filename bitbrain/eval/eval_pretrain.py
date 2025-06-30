import torch
import argparse
import random
import warnings
import numpy as np
from transformers import AutoTokenizer, AutoModelForCausalLM, TextStreamer, AutoConfig
# from model.model_minimind import MiniMindConfig, MiniMindForCausalLM
# from model.model_lora import *

warnings.filterwarnings('ignore')

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"使用设备: {device}")

def init_model(args):
    tokenizer = AutoTokenizer.from_pretrained(
    str(args.tokenizer_dir),
    trust_remote_code=True,
    use_fast=True
)
    
    # 调试信息：打印分词器的特殊标记
    print(f"分词器特殊标记:")
    print(f"  bos_token: {repr(tokenizer.bos_token)}")
    print(f"  eos_token: {repr(tokenizer.eos_token)}")
    print(f"  pad_token: {repr(tokenizer.pad_token)}")

    config_kwargs = {
        "trust_remote_code": True,
        "use_custom_rmsnorm": False,
        "rope_theta": 10000,
        "max_position_embeddings": 4096
    }
# if torch.cuda.is_available():
#     config_kwargs["attn_implementation"] = "flash_attention_2"

    config = AutoConfig.from_pretrained(
        str(args.model_dir),
        **config_kwargs
    )

    if args.load == 0:
        modes = {0: 'pretrain', 1: 'full_sft', 2: 'rlhf', 3: 'reason', 4: 'grpo'}
        
        # 根据不同模式选择不同的checkpoint路径
        checkpoint_paths = {
            0: "/home/chenyuhang/bit-brain/bitbrain/train/checkpoints/bitbrain_pretrain_epoch_1.pt",  # 预训练模型
            1: '/home/chenyuhang/bit-brain/bitbrain/train/checkpoints/bitbrain_pretrain_epoch_1.pt',  # SFT模型
            2: '/path/to/rlhf/checkpoint/',  # RLHF模型路径（需要你提供实际路径）
            3: '/path/to/reason/checkpoint/',  # Reason模型路径
            4: '/path/to/grpo/checkpoint/'   # GRPO模型路径
        }
        
        current_mode = modes[args.model_mode]
        ckp = checkpoint_paths[args.model_mode]
        
        print(f"当前模式: {current_mode}")
        print(f"加载checkpoint: {ckp}")
        
        model = AutoModelForCausalLM.from_config(config=config)
        print("加载模型权重...")
        
        # 根据文件类型选择不同的加载方式
        if ckp.endswith(('.pth', '.pt')):
            # 预训练模型加载方式
            checkpoint = torch.load(ckp, map_location=args.device, weights_only=False)
            if 'model_state_dict' in checkpoint:
                state_dict = checkpoint['model_state_dict']
                # 去除_orig_mod.前缀
                clean_state_dict = {k.replace('_orig_mod.', ''): v for k, v in state_dict.items()}
            else:
                clean_state_dict = checkpoint
            model.load_state_dict(clean_state_dict, strict=True)
            # 将模型移动到指定的设备（例如 'cuda' 或 'cpu'）
            # 这是修复问题的关键步骤
            model.to(args.device)
        else:
            # SFT模型加载方式（从checkpoint目录加载）
            model = AutoModelForCausalLM.from_pretrained(
                ckp,
                config=config,
                torch_dtype=torch.float16,  # 使用半精度以节省显存
                device_map="auto" if torch.cuda.is_available() else None
            )
            
            # 当没有CUDA可用时，device_map为None，模型在CPU上
            # 此时需要这行代码确保模型在正确的设备（'cpu'）上
            if not torch.cuda.is_available():
                model.to(device)

    print(f'Bit-Brain_V1模型参数量: {sum(p.numel() for p in model.parameters() if p.requires_grad) / 1e6:.2f}M(illion)')
    return model.eval(), tokenizer


def get_prompt_datas(args):
    if args.model_mode == 0:
        # pretrain模型的接龙能力（无法对话）
        prompt_datas = [
            '马克思主义基本原理',
            '人类大脑的主要功能',
            '万有引力原理是',
            '世界上最高的山峰是',
            '二氧化碳在空气中',
            '地球上最大的动物有',
            '杭州市的美食有'
        ]
    elif args.model_mode == 1:  # SFT模式
        # SFT模型的对话测试问题
        prompt_datas = [
            '你好，请介绍一下自己。',
            '什么是人工智能？请详细解释。',
            '请帮我写一个Python的快速排序算法。',
            '解释一下什么是机器学习。',
            '如何学习编程？给我一些建议。',
            '请解释量子计算的基本原理。',
            '中国的首都是哪里？',
            '请用中文回答：什么是深度学习？',
            '帮我分析一下这个问题：如何提高工作效率？'
        ]
    else:
        if args.lora_name == 'None':
            # 通用对话问题
            prompt_datas = [
                '请介绍一下自己。',
                '你更擅长哪一个学科？',
                '鲁迅的《狂人日记》是如何批判封建礼教的？',
                '我咳嗽已经持续了两周，需要去医院检查吗？',
                '详细的介绍光速的物理概念。',
                '推荐一些杭州的特色美食吧。',
                '请为我讲解"大语言模型"这个概念。',
                '如何理解ChatGPT？',
                'Introduce the history of the United States, please.'
            ]
        else:
            # 特定领域问题
            lora_prompt_datas = {
                'lora_identity': [
                    "你是ChatGPT吧。",
                    "你叫什么名字？",
                    "你和openai是什么关系？"
                ],
                'lora_medical': [
                    '我最近经常感到头晕，可能是什么原因？',
                    '我咳嗽已经持续了两周，需要去医院检查吗？',
                    '服用抗生素时需要注意哪些事项？',
                    '体检报告中显示胆固醇偏高，我该怎么办？',
                    '孕妇在饮食上需要注意什么？',
                    '老年人如何预防骨质疏松？',
                    '我最近总是感到焦虑，应该怎么缓解？',
                    '如果有人突然晕倒，应该如何急救？'
                ],
            }
            prompt_datas = lora_prompt_datas[args.lora_name]

    return prompt_datas


# 设置可复现的随机种子
def setup_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def main():
    parser = argparse.ArgumentParser(description="Chat with Bit-Brain")
    parser.add_argument('--tokenizer_dir', default='/DATA/disk2/yuhang/.cache/modelscope/models/Qwen/Qwen3-0.6B',type=str)
    parser.add_argument('--model_dir', default='/DATA/disk2/yuhang/.cache/modelscope/models/Qwen/Qwen3-0.6B', type=str)
    parser.add_argument('--lora_name', default='None', type=str)
    parser.add_argument('--out_dir', default='out', type=str)
    parser.add_argument('--max_seq_len', default=2048, type=int)          
    parser.add_argument('--temperature', default=0.6, type=float)
    parser.add_argument('--top_p', default=0.95, type=float)
    parser.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', type=str)
    # 此处max_seq_len（最大输出长度）并不意味模型具有对应的长文本的性能，仅防止QA出现被截断的问题
    # 携带历史对话上下文条数
    # history_cnt需要设为偶数，即【用户问题, 模型回答】为1组；设置为0时，即当前query不携带历史上文
    # 模型未经过外推微调时，在更长的上下文的chat_template时难免出现性能的明显退化，因此需要注意此处设置
    parser.add_argument('--history_cnt', default=0, type=int)
    parser.add_argument('--load', default=0, type=int, help="0: 原生torch权重，1: transformers加载")
    parser.add_argument('--model_mode', default=0, type=int,  # 默认使用SFT模式
                        help="0: 预训练模型，1: SFT-Chat模型，2: RLHF-Chat模型，3: Reason模型，4: GRPO-Chat模型")
    args = parser.parse_args()

    model, tokenizer = init_model(args)
    
    # 显示当前测试模式
    modes = {0: 'pretrain', 1: 'full_sft', 2: 'rlhf', 3: 'reason', 4: 'grpo'}
    print(f"\n=== 当前测试模式: {modes[args.model_mode].upper()} ===")
    
    prompts = get_prompt_datas(args)
    test_mode = int(input('[0] 自动测试\n[1] 手动输入\n'))
    streamer = TextStreamer(tokenizer, skip_prompt=True, skip_special_tokens=True)

    messages = []
    for idx, prompt in enumerate(prompts if test_mode == 0 else iter(lambda: input('👶: '), '')):
        setup_seed(random.randint(0, 2048))
        # setup_seed(2025)  # 如需固定每次输出则换成【固定】的随机种子
        if test_mode == 0: print(f'👶: {prompt}')

        messages = messages[-args.history_cnt:] if args.history_cnt else []
        messages.append({"role": "user", "content": prompt})

        if args.model_mode != 0:
            new_prompt = tokenizer.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True
            )
        else:
            # 处理预训练模式的提示词格式
            bos_token = tokenizer.bos_token if tokenizer.bos_token is not None else ''
            new_prompt = bos_token + prompt

        inputs = tokenizer(
            new_prompt,
            return_tensors="pt",
            truncation=True
        ).to(args.device)

        print('🤖️: ', end='')
        generated_ids = model.generate(
            inputs["input_ids"],
            max_new_tokens=200,
            num_return_sequences=1,
            do_sample=True,
            attention_mask=inputs["attention_mask"],
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            streamer=streamer,
            top_p=args.top_p,
            temperature=args.temperature,
            repetition_penalty=1.4
        )

        response = tokenizer.decode(generated_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        messages.append({"role": "assistant", "content": response})
        print('\n\n')


if __name__ == "__main__":
    main()