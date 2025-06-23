#!/usr/bin/env bash
set -euo pipefail

# root of your repo
PROJECT_ROOT="../../"

# resources
GPU=2
GLOBAL_BATCH_SIZE=23

# start-from checkpoint
MODEL_NAME_OR_PATH="../../../../data/outputs/fastlora.Mistral7BInstructv02.mistral-8K-1B-com.w1024-pre-norm-sum.kinf.ri1024.r128.a64.o.svd.bs8.lr1e-4.pt-mix.20240916-110738/checkpoint-14332"

# template for system / user turns
CHAT_TEMPLATE="mistral"

# derived numbers
ACCUM_STEP=$((GLOBAL_BATCH_SIZE / GPU))
TIMESTAMP=$(date +%Y%m%d-%H%M%S)

torchrun --nproc_per_node "$GPU" \
         --master_port 29501 \
         train.py \
         --run_name "fastlora_mistral_sft_v4_${TIMESTAMP}" \
         --output_dir "$PROJECT_ROOT/data/outputs/run_${TIMESTAMP}" \
         --model_name_or_path "$MODEL_NAME_OR_PATH" \
         --train_data "$PROJECT_ROOT/data/sft/sft-v4.jsonl" \
         --eval_data "$PROJECT_ROOT/data/pretrain/val-8K-1M.json" \
         --chat_template "$CHAT_TEMPLATE" \
         --deepspeed "$PROJECT_ROOT/data/deepspeed/stage2.json" \
         --per_device_train_batch_size 1 \
         --gradient_accumulation_steps "$ACCUM_STEP" \
         --num_train_epochs 2 \
         --max_length 8192 \
         --fastlora_window 1024 \
         --warmup_steps 100 \
         --learning_rate 5e-5 \
         --weight_decay 0.01 \
         --gradient_checkpointing \
         --use_reentrant False \
         --save_only_model \
         --save_strategy steps \
         --evaluation_strategy steps \
         --save_steps 0.49 \
         --eval_steps 500 \
         --eval_max_length 2048 \
         --logging_steps 1 \
         --report_to wandb \
         --wandb_watch_log_freq 1000
