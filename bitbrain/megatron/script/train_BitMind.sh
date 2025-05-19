#!/bin/bash

# Runs the "175B" parameter model

export CUDA_DEVICE_MAX_CONNECTIONS=1
export HF_ENDPOINT=https://hf-mirror.com
export TRANSFORMERS_OFFLINE=1
export HF_DATASETS_OFFLINE=1
# 添加以下环境变量跳过C++编译
export MEGATRON_SKIP_CPP_COMPILE=1


GPUS_PER_NODE=4
# Change for multinode config
MASTER_ADDR=localhost
MASTER_PORT=6000
NUM_NODES=1
NODE_RANK=0
WORLD_SIZE=$(($GPUS_PER_NODE*$NUM_NODES))

# Use parameter substitution to provide default values if arguments are not passed
# This prevents the shell from trying to execute the path strings as commands.
CHECKPOINT_PATH=${1:-"/DATA/disk2/yuhang/.cache/Megatron/BitMind/CHECKPOINT"}
TENSORBOARD_LOGS_PATH=${2:-"/DATA/disk2/yuhang/.cache/Megatron/BitMind/TENSORBOARD_LOGS"}
VOCAB_FILE=${3:-"/DATA/disk2/yuhang/.cache/modelscope/models/Qwen/Qwen2___5-0___5B-Instruct/vocab.json"}
MERGE_FILE=${4:-"/DATA/disk2/yuhang/.cache/modelscope/models/Qwen/Qwen2___5-0___5B-Instruct/merges.txt"}
#TOKENIZER_CONFIG_FILE=${6:-"/DATA/disk2/yuhang/.cache/modelscope/models/Qwen/Qwen2___5-0___5B-Instruct/tokenizer_config.json"}
DATA_PATH=${5:-"/DATA/disk2/yuhang/.cache/Megatron/BitMind"}

# Determine the absolute path of the directory where the script is located
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
# Construct the absolute path to the pretrain_BitMind.py script
# pretrain_BitMind.py is expected to be in the parent directory of the script's directory
PYTHON_SCRIPT_PATH="$SCRIPT_DIR/../pretrain_BitMind.py"

# Construct the path to the directory containing the 'megatron' package.
# This is effectively the root of your 'bitbrain' project in this context,
# as 'megatron' is expected to be importable from within it.
PROJECT_BASE_DIR=$(cd "$SCRIPT_DIR/../.." && pwd)

# Prepend this directory to PYTHONPATH so Python can find the 'megatron' module.
# We also include the existing PYTHONPATH to avoid overriding it if it's already set.
export PYTHONPATH="${PROJECT_BASE_DIR}:${PYTHONPATH}"

DISTRIBUTED_ARGS=(
    --nproc_per_node $GPUS_PER_NODE
    --nnodes $NUM_NODES
    --master_addr $MASTER_ADDR
    --master_port $MASTER_PORT
)

GPT_MODEL_ARGS=(
    --num-layers 12
    --hidden-size 2048
    --num-attention-heads 16
    --seq-length 2048
    --max-position-embeddings 2048
    --attention-backend auto # Can use (flash/fused/unfused/local)
)

TRAINING_ARGS=(
    --micro-batch-size 1
    --global-batch-size 48
    #--rampup-batch-size 16 16 5859375 #! 初始的全局批量大小  每次增加的批量大小  达到多少迭代步数后停止
    --train-iters 500000
    --weight-decay 0.1
    --adam-beta1 0.9
    --adam-beta2 0.95
    --init-method-std 0.006
    --clip-grad 1.0
    --fp16
    --lr 3.0e-4
    --lr-decay-style cosine
    --min-lr 3.0e-5
    --lr-warmup-fraction .001
    --lr-decay-iters 430000  #! 学习率衰减发生的迭代次数
)

MODEL_PARALLEL_ARGS=(
	--tensor-model-parallel-size 1
	--pipeline-model-parallel-size 1
)

TOKENIZER_PATH="/DATA/disk2/yuhang/.cache/modelscope/models/Qwen/Qwen2___5-0___5B-Instruct"

DATA_ARGS=(
    --data-path $DATA_PATH
    --vocab-file $VOCAB_FILE
    --merge-file $MERGE_FILE
    --tokenizer-type HuggingFaceTokenizer
    --tokenizer-model $TOKENIZER_PATH
    --split 949,50,1
)

EVAL_AND_LOGGING_ARGS=(
    --log-interval 100
    --save-interval 10000
    --eval-interval 1000
    --save $CHECKPOINT_PATH
    --load $CHECKPOINT_PATH
    --eval-iters 10
    --tensorboard-dir $TENSORBOARD_LOGS_PATH
)

# Use the calculated PYTHON_SCRIPT_PATH
torchrun ${DISTRIBUTED_ARGS[@]} $PYTHON_SCRIPT_PATH \
    ${GPT_MODEL_ARGS[@]} \
    ${TRAINING_ARGS[@]} \
    ${MODEL_PARALLEL_ARGS[@]} \
    ${DATA_ARGS[@]} \
    ${EVAL_AND_LOGGING_ARGS[@]}
