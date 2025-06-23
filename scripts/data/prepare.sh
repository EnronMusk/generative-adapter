PROJECT_ROOT="../../"

python prepare_pretrain_data.py \
    --config $PROJECT_ROOT/data/slimpajama/config.json \
    --train_data $PROJECT_ROOT/data/slimpajama/ \
    --output_dir mistral-7b-8K-1B \
    --dataset_cache_dir cache/ \
    --num_token 8192:1b \
    --model_name_or_path mistralai/Mistral-7B-Instruct-v0.2 

python prepare_pretrain_data.py \
    --config $PROJECT_ROOT/data/slimpajama/config.json \
    --train_data $PROJECT_ROOT/data/slimpajama/ \
    --output_dir llama2-7b-8K-1B \
    --dataset_cache_dir cache/ \
    --num_token 8192:1b \
    --model_name_or_path meta-llama/Llama-2-7b-chat-hf 

python prepare_pretrain_data.py \
    --config $PROJECT_ROOT/data/slimpajama/config.json \
    --train_data $PROJECT_ROOT/data/slimpajama/ \
    --output_dir llama3-8b-8K-1B \
    --dataset_cache_dir cache/ \
    --num_token 8192:1b \
    --model_name_or_path meta-llama/Meta-Llama-3-8B-Instruct 
