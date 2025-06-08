##########################
# llama factory start script
##########################
#CUDA_VISIBLE_DEVICES=5,6 python ../LLaMA-Factory/src/train.py \
export LLaMA_PATH=/home/chenyuhang/LLaMA-Factory
OUTPUT_DIR=/DATA/disk2/yuhang/.cache/ckpt/bit-brain/sft
export CUDA_VISIBLE_DEVICES=4,5,6,7 
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
#python  $LLaMA_PATH/src/train.py \
# max_steps / num_train_epochs
# --streaming True \
# 6卡配置：1,2,3,4,5,7
# 原始ckpt：--model_name_or_path /data/model/llm/Steel-LLM/steel-llm-step-1060000-ckpt \
# all data: 
#! 可设置混合策略：   --mix_strategy interleave_over \
#!                  --interleave_probs 0.4,0.1,0.1,0.4 \
#! 可设置自定义的评估数据集  --eval_dataset ceval,cmmlu \

#! 设置了--eval-dataset就不能设置--val_size 
CUDA_VISIBLE_DEVICES=4,5,6,7 torchrun --nproc_per_node 4 $LLaMA_PATH/src/train.py \
    --stage sft \
    --do_train \
    --model_name_or_path /home/chenyuhang/bit-brain/bitbrain/models/pertrain_qwen3_0.6B \
    --cutoff_len 1024 \
    --dataset_dir /DATA/disk2/yuhang/.cache/steel_dataset/sft_data/llamafactory_input \
    --dataset baai_instruct_70W,baai_instruct_682W,openhermes_custom,wanjuan_exam_399W \
    --use_swanlab true \
    --report_to swanlab \
    --run_name sft_bit-brain-v1 \
    --preprocessing_num_workers 40 \
    --template qwen \
    --finetuning_type full \
    --output_dir ${OUTPUT_DIR}/bit-brain-v1-full-sft \
    --overwrite_output_dir \
    --overwrite_cache \
    --per_device_train_batch_size 24 \
    --per_device_eval_batch_size 4 \
    --do_eval \
    --val_size 0.01 \
    --eval_strategy steps \
    --eval_steps 1000 \
    --flash_attn fa2\
    --gradient_accumulation_steps 4 \
    --lr_scheduler_type cosine \
    --logging_steps 10 \
    --save_steps 5000 \
    --learning_rate 1.5e-4 \
    --weight_decay 0.0 \
    --num_train_epochs 4.0 \
    --plot_loss \
    --bf16


    #--evaluation_strategy 5000 \