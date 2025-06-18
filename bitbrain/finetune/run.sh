# llama factory start script
##########################
#CUDA_VISIBLE_DEVICES=5,6 python ../LLaMA-Factory/src/train.py \
export LLaMA_PATH=/home/chenyuhang/LLaMA-Factory
OUTPUT_DIR=/DATA/disk2/yuhang/.cache/ckpt/bit-brain-v2/sft
#export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=4,5,6,7 
export NCCL_P2P_LEVEL=NVL
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
#python  $LLaMA_PATH/src/train.py \
# max_steps / num_train_epochs
# --streaming True \
# 6卡配置：1,2,3,4,5,7
# 原始ckpt：--model_name_or_path /data/model/llm/Steel-LLM/steel-llm-step-1060000-ckpt \
# all data: 
#! 可设置混合策略：     --mix_strategy interleave_over\
#!    --interleave_probs 0.1,0.35,0.2,0.2,0.1,0.05 \

#! 可设置自定义的评估数据集  --eval_dataset ceval,cmmlu \

#! 设置了--eval-dataset就不能设置--val_size 
#! 先不使用--overwrite_cache \

#!     --streaming True \
#!    --max_steps 100000 \

#!  baai_instruct_70W,baai_instruct_682W,deepctl_200W \
#! --max_samples 2000000 \


FORCE_TORCHRUN=1 CUDA_VISIBLE_DEVICES=4,5,6,7  torchrun --nproc_per_node 4 $LLaMA_PATH/src/train.py \
    --stage sft \
    --do_train \
    --model_name_or_path /home/chenyuhang/bit-brain/bitbrain/models/pertrain_qwen3_0.6B \
    --cutoff_len 2048 \
    --dataset_dir /DATA/disk2/yuhang/.cache/steel_dataset/sft_data/llamafactory_input \
    --dataset deepctl_1120W_zh,deepctl_276W_en,baai_instruct_70W,ultrachat_200k,wanjuan_exam_399W \
    --max_samples 2000000 \
    --overwrite_cache \
    --packing False \
    --use_swanlab true \
    --report_to swanlab \
    --run_name sft_bit-brain-v2 \
    --preprocessing_num_workers 30 \
    --template qwen \
    --finetuning_type full \
    --output_dir ${OUTPUT_DIR}/lr2e-5 \
    --overwrite_output_dir \
    --per_device_train_batch_size 16 \
    --per_device_eval_batch_size 4 \
    --do_eval \
    --val_size 100 \
    --eval_strategy steps \
    --eval_steps 1000 \
    --flash_attn sdpa\
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --warmup_ratio 0.1 \
    --max_grad_norm 1.0 \
    --logging_steps 10 \
    --save_steps 500 \
    --learning_rate 2e-5 \
    --weight_decay 0.01 \
    --num_train_epochs 4.0 \
    --plot_loss \
    --bf16 \

    
    #--resume_from_checkpoint /DATA/disk2/yuhang/.cache/ckpt/bit-brain/sft_mix_v2/bit-brain-v1-full-sft/checkpoint-500


    #--evaluation_strategy 5000 \
    # --resume_from_checkpoint /DATA/disk2/yuhang/.cache/ckpt/bit-brain/sft/bit-brain-v1-full-sft/checkpoint-10000
##########################