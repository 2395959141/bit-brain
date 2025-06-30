import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from safetensors.torch import load_file

def quick_test(safetensors_path):
    """快速测试safetensors文件"""
    try:
        # 加载权重
        state_dict = load_file(safetensors_path)
        print(f"✅ 权重加载成功，参数数量: {len(state_dict)}")
        
        # 方法1: 通过AutoConfig修改配置
        config = AutoConfig.from_pretrained("/home/chenyuhang/bit-brain/bitbrain/models/Bitbran-0.6B-base")
        config.rope_theta = 10000.0
        config.max_position_embeddings = 4096
        
        print(f"📝 配置信息:")
        print(f"  - rope_theta: {config.rope_theta}")
        print(f"  - max_position_embeddings: {config.max_position_embeddings}")
        print(f"  - vocab_size: {config.vocab_size}")
        print(f"  - hidden_size: {config.hidden_size}")
        
        # 加载模型和分词器
        model = AutoModelForCausalLM.from_pretrained(
            "/home/chenyuhang/bit-brain/bitbrain/models/Bitbran-0.6B-base", 
            config=config,
            torch_dtype=torch.float16,
            trust_remote_code=True
        )
        tokenizer = AutoTokenizer.from_pretrained("/home/chenyuhang/bit-brain/bitbrain/models/Bitbran-0.6B-base")
        
        # 应用权重
        missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
        print(f"📊 权重加载结果:")
        print(f"  - 缺失键数量: {len(missing_keys)}")
        print(f"  - 多余键数量: {len(unexpected_keys)}")
        
        model.eval()
        
        # 简单测试
        prompt = "杭州的小吃有哪些？"
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=500,
                do_sample=True,
                temperature=0.7,
                top_k=20,
                top_p=0.7,
                repetition_penalty=1.2,
                pad_token_id=tokenizer.eos_token_id,
                #no_repeat_ngram_size=2,
            )
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"✅ 测试成功！输出: {response}")
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

# 方法2: 直接通过model_kwargs设置（替代方案）
def quick_test_alternative(safetensors_path):
    """使用model_kwargs参数的替代方法"""
    try:
        # 加载权重
        state_dict = load_file(safetensors_path)
        print(f"✅ 权重加载成功，参数数量: {len(state_dict)}")
        
        # 直接通过from_pretrained的参数设置
        model = AutoModelForCausalLM.from_pretrained(
            "Qwen/Qwen3-0.6B",
            torch_dtype=torch.float16,
            trust_remote_code=True,
            rope_theta=10000.0,
            max_position_embeddings=4096
        )
        
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen3-0.6B")
        
        # 验证配置
        print(f"📝 当前模型配置:")
        print(f"  - rope_theta: {model.config.rope_theta}")
        print(f"  - max_position_embeddings: {model.config.max_position_embeddings}")
        
        # 应用权重
        model.load_state_dict(state_dict, strict=False)
        model.eval()
        
        # 简单测试
        prompt = "马克思主义思想是什么？"
        inputs = tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            outputs = model.generate(**inputs, max_new_tokens=20)
        
        response = tokenizer.decode(outputs[0], skip_special_tokens=True)
        print(f"✅ 测试成功！输出: {response}")
        return True
        
    except Exception as e:
        print(f"❌ 测试失败: {e}")
        return False

# 使用示例
if __name__ == "__main__":
    safetensors_file = "/home/chenyuhang/bit-brain/bitbrain/models/pertrain_qwen3_0.6B/model.safetensors"
    
    print("方法1: 通过AutoConfig设置参数")
    success1 = quick_test(safetensors_file)
    
    print("\n" + "="*50)
    print("方法2: 通过model_kwargs设置参数")
    #success2 = quick_test_alternative(safetensors_file)
