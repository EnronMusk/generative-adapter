#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="../../"

GPU=2
GLOBAL_BATCH_SIZE=8
MODEL_NAME_OR_PATH="mistralai/Mistral-7B-Instruct-v0.2"
MODEL_TAG="mistral-7b-instruct"

ACCUM_STEP=$((GLOBAL_BATCH_SIZE / GPU))
TIMESTAMP=$(date +%Y%m%d-%H%M%S)

torchrun --nproc_per_node "$GPU" \
         --master_port 29501 \
         train.py \
         --run_name "fastlora_mistral_${TIMESTAMP}" \
         --output_dir "$PROJECT_ROOT/outputs/fastlora_mistral_run_${TIMESTAMP}" \
         --model_name_or_path "$MODEL_NAME_OR_PATH" \
         --train_data "$PROJECT_ROOT/data/pretrain/mistral-8K-1B/" \
         --eval_data "$PROJECT_ROOT/data/pretrain/val-8K-1M.json" \
         --deepspeed "$PROJECT_ROOT/data/deepspeed/stage2.json" \
         --per_device_train_batch_size 1 \
         --gradient_accumulation_steps "$ACCUM_STEP" \
         --num_train_epochs 1 \
         --max_length 4096 \
         --warmup_steps 100 \
         --learning_rate 1e-4 \
         --weight_decay 0.01 \
         --gradient_checkpointing \
         --use_reentrant False \
         --enable_lora \
         --lora_r 0 \
         --enable_fastlora \
         --fastlora_r 1024 \
         --fastlora_inter_size 1024 \
         --fastlora_window 1024 \
         --fastlora_max_rank 128 \
         --fastlora_attn_len 8192 \
         --fastlora_alpha 64 \
         --fastlora_dropout 0.0 \
         --fastlora_param o_proj \
         --fastlora_arch aassbb \
         --fastlora_norm svd \
         --fastlora_merge pre-norm-sum \
         --fastlora_training_attention_mask abcdabcd \
         --save_only_model \
         --save_strategy steps \
         --evaluation_strategy steps \
         --save_steps 0.19 \
         --eval_steps 500 \
         --eval_max_length 2048 \
         --logging_steps 1 \
         --report_to wandb \
         --wandb_watch_log_freq 1000 \
