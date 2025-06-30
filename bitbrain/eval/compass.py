# 从 opencompass 库中导入评测所需的模块和预定义好的数据集配置
# HuggingFaceBaseModel 是用来加载你这种本地模型的类
from opencompass.models import HuggingFaceBaseModel
# gsm8k_datasets 是一个预定义好的数学应用题数据集配置
from opencompass.configs.datasets.gsm8k.gsm8k_gen import gsm8k_datasets
# ceval_datasets 是一个预定义好的中文综合能力评测数据集配置
from opencompass.configs.datasets.ceval.ceval_gen import ceval_datasets
from mmengine.config import read_base
from opencompass.models import HuggingFaceCausalLM

# 使用 read_base 上下文管理器来导入预定义的数据集配置
# 这是 OpenCompass 推荐的标准做法，可以更好地处理配置的继承和懒加载
with read_base():
    from opencompass.configs.datasets.gsm8k.gsm8k_gen import gsm8k_datasets
    from opencompass.configs.datasets.ceval.ceval_gen import ceval_datasets

# 使用列表解包的方式将所有需要评测的数据集组合起来
# 这种写法清晰地表明我们将多个数据集列表合并成一个
datasets = [*ceval_datasets]

# --- 第一部分：定义你的模型 ---
# `models` 是一个 Python 列表，你可以在里面定义一个或多个模型进行评测
models = [
    dict(
        # 使用 HuggingFaceCausalLM，更精确地指定模型类型
        type=HuggingFaceCausalLM,
        
        # `abbr` 是你模型的简称，它会作为列名显示在最终的结果表格里，方便识别。
        abbr='bitbran-0.6b-base',
        
        # `path` 是模型的本地存放路径，这里我填入了你提供的绝对路径。
        path='/home/chenyuhang/bit-brain/bitbrain/models/Bitbran-0.6B-base',
        
        # `tokenizer_path` 指定了分词器的路径，对于你的模型来说，它和模型路径是一样的。
        tokenizer_path='/home/chenyuhang/bit-brain/bitbrain/models/Bitbran-0.6B-base',
        
        # `tokenizer_kwargs` 是一些关于分词器的额外参数设置。
        tokenizer_kwargs=dict(
            padding_side='left',
            truncation_side='left',
            use_fast=False,
        ),
        
        # 新增: 为模型生成添加额外参数，明确指定不进行采样，消除相关警告
        #! 对ceval数据集进行评测，不进行采样
        generation_kwargs=dict(
            do_sample=False,
        ),

        # `max_out_len` 限制了模型在生成答案时，能生成的最大 token 数量。
        # 对于 C-Eval 选择题，一个很小的值就足够了
        max_out_len=16,
        
        # `max_seq_len` 是模型能处理的最大输入序列长度（上下文长度）。
        max_seq_len=2048,
        
        # `batch_size` 是推理时每批次处理的样本数量。你可以根据你的显存大小来调整这个值。
        batch_size=8,
        
        # `run_cfg` 配置了运行任务所需的计算资源，这里我们指定使用 1 个 GPU。
        run_cfg=dict(num_gpus=1),
    )
]
