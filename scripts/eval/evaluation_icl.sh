
# eval-metaicl-0:
# 	CUDA_VISIBLE_DEVICES=0 \
# 	python eval_icl_metaicl.py \
# 		--model_name ../../data/models-dev/fastlora.Mistral7BInstructv02.sft-v4.w1024-pre-norm-sum.kinf.ri1024.r128.a64.o.svd.bs8.lr1e-4.pt-mix-sft-v4.20240927-003203/checkpoint-3428 \
# 		--seed 0 \
# 		--merge_strategy sequential \
# 		--window_size 1024 \
# 		--output_path ./results/metaicl/fastlora.mistral.sft-v4.w1024-pre-norm-sum.kinf.ri1024.r128.a64.o.svd.bs8.lr1e-4.pt-mix-sft-v4.sequential.json \


# eval-metaicl-1:
# 	CUDA_VISIBLE_DEVICES=1 \
# 	python eval_icl_metaicl.py \
# 		--model_name ../../data/models-dev/fastlora.llama2-7B-chat.sft-v4.w1024-pre-norm-sum.kinf.ri1024.r128.a64.o.svd.bs8.lr1e-4.pt-mix-sft-v4.20240929-161031/checkpoint-3416 \
# 		--seed 0 \
# 		--merge_strategy sequential \
# 		--window_size 1024 \
# 		--output_path ./results/metaicl/fastlora.llama2.sft-v4.w1024-pre-norm-sum.kinf.ri1024.r128.a64.o.svd.bs8.lr1e-4.pt-mix-sft-v4.sequential.json \


# eval-metaicl-2:
# 	CUDA_VISIBLE_DEVICES=1 \
# 	python eval_icl_metaicl.py \
# 		--model_name meta-llama/Llama-2-7b-chat-hf \
# 		--seed 0 \
# 		--use_chat \
# 		--decoding_setting zero-shot \
# 		--output_path ./results/metaicl/llama2.instruct.zero-shot.json \

# eval-metaicl-3:
# 	CUDA_VISIBLE_DEVICES=3 \
# 	python eval_icl_metaicl.py \
# 		--model_name Mistralai/Mistral-7B-Instruct-v0.2 \
# 		--seed 0 \
# 		--use_chat \
# 		--decoding_setting zero-shot \
# 		--output_path ./results/metaicl/mistral.instruct.zero-shot.json \

# eval-metaicl-2:
# 	CUDA_VISIBLE_DEVICES=1 \
# 	python eval_icl_metaicl.py \
# 		--model_name meta-llama/Llama-2-7b-chat-hf \
# 		--seed 0 \
# 		--use_chat \
# 		--output_path ./results/metaicl/llama2.instruct.no-instruct.json \

# eval-metaicl-3:
# 	CUDA_VISIBLE_DEVICES=3 \
# 	python eval_icl_metaicl.py \
# 		--model_name Mistralai/Mistral-7B-Instruct-v0.2 \
# 		--seed 0 \
# 		--use_chat \
# 		--output_path ./results/metaicl/mistral.instruct.no-instruct.json \

# eval-metaicl-4:
# 	CUDA_VISIBLE_DEVICES=0 \
# 	python eval_icl_metaicl.py \
# 		--model_name meta-llama/Llama-2-7b-chat-hf \
# 		--seed 0 \
# 		--use_chat \
# 		--finetune \
# 		--output_path ./results/metaicl/llama2.finetune-lr1e-5.no-instruct.json \

# 	# CUDA_VISIBLE_DEVICES=0 \
# 	# python eval_icl_metaicl.py \
# 	# 	--model_name Mistralai/Mistral-7B-Instruct-v0.2 \
# 	# 	--seed 0 \
# 	# 	--use_chat \
# 	# 	--finetune \
# 	# 	--output_path ./results/metaicl/mistral.finetune.no-instruct.json \
