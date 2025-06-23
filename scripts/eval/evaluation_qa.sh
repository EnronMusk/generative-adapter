## Llama2

### Without knowledge injection
python eval_streaming_qa.py \
	--data streamingqa \
	--model_name meta-llama/Llama-2-7b-chat-hf \
	--prompt_template context-instruction \
	--output_path results/streamingqa/llama2-7b-instruct.default.context-instruction.json

python eval_streaming_qa.py \
	--data streamingqa \
	--model_name meta-llama/Llama-2-7b-chat-hf \
	--prompt_template rag-instruction \
	--output_path results/streamingqa/llama2-7b-instruct.default.rag-instruction.json

python eval_streaming_qa.py \
	--data streamingqa \
	--model_name meta-llama/Llama-2-7b-chat-hf \
	--prompt_template instruction \
	--decoding_setting close-book \
	--output_path results/streamingqa/llama2-7b-instruct.question.json

python eval_streaming_qa.py \
	--data streamingqa \
	--model_name ../../libs/CaMeLS/outputs/eval/streamingqa/uniform-uniform/meta-llama-Llama-2-7b-chat-hf/sft/final_model \
	--prompt_template context-question \
	--output_path results/streamingqa/llama2-7b-instruct.sft.context-question.json \

python eval_streaming_qa.py \
	--data streamingqa \
	--model_name ../../libs/CaMeLS/outputs/eval/streamingqa/uniform-uniform/meta-llama-Llama-2-7b-chat-hf/sft/final_model \
	--prompt_template rag-question \
	--output_path results/streamingqa/llama2-7b-instruct.sft.rag-question.json \

python eval_streaming_qa.py \
	--data streamingqa \
	--model_name ../../libs/CaMeLS/outputs/eval/streamingqa/uniform-uniform/meta-llama-Llama-2-7b-chat-hf/sft/final_model \
	--prompt_template question \
	--decoding_setting close-book \
	--output_path results/streamingqa/llama2-7b-instruct.sft.question.json

### With knowledge injection

python eval_streaming_qa.py \
	--data streamingqa \
	--model_name ../../data/models-dev/fastlora.llama2-7B-chat.sft-v4.w1024-pre-norm-sum.kinf.ri1024.r128.a64.o.svd.bs8.lr1e-4.pt-mix-sft-v4.20240928-215611/checkpoint-1950 \
	--prompt_template context-question \
	--merge_strategy sequential \
	--output_path results/streamingqa/llama2-7b-instruct.fastlora.context-question.json \

python eval_streaming_qa.py \
	--data streamingqa \
	--model_name ../../data/models-dev/fastlora.llama2-7B-chat.sft-v4.w1024-pre-norm-sum.kinf.ri1024.r128.a64.o.svd.bs8.lr1e-4.pt-mix-sft-v4.20240928-215611/checkpoint-1950 \
	--prompt_template rag-question \
	--merge_strategy sequential \
	--output_path results/streamingqa/llama2-7b-instruct.fastlora.rag-question.json \

python eval_streaming_qa.py \
	--data squad \
	--model_name ../../data/models-dev/fastlora.llama2-7B-chat.sft-v4.w1024-pre-norm-sum.kinf.ri1024.r128.a64.o.svd.bs8.lr1e-4.pt-mix-sft-v4.20240928-215611/checkpoint-1950 \
	--prompt_template question \
	--merge_strategy sequential \
	--output_path results/streamingqa/llama2-7b-instruct.fastlora.question.json \

python eval_streaming_qa.py \
	--data streamingqa \
	--model_name ../../libs/CaMeLS/outputs/eval/streamingqa/uniform-uniform/meta-llama-Llama-2-7b-chat-hf/ep8-sft/final_model \
	--prompt_template question \
	--decoding_setting close-book \
	--output_path results/streamingqa/llama2-7b-instruct.cp-sft.question.json


## Mistral

### Without knowledge injection
python eval_streaming_qa.py \
	--data streamingqa \
	--model_name Mistralai/Mistral-7B-Instruct-v0.2 \
	--prompt_template context-instruction \
	--output_path results/streamingqa/mistral-7b-instruct.default.context-instruction.json \

python eval_streaming_qa.py \
	--data streamingqa \
	--model_name Mistralai/Mistral-7B-Instruct-v0.2 \
	--prompt_template rag-instruction \
	--output_path results/streamingqa/mistral-7b-instruct.default.rag-instruction.json \

python eval_streaming_qa.py \
	--data streamingqa \
	--model_name Mistralai/Mistral-7B-Instruct-v0.2 \
	--prompt_template instruction \
	--decoding_setting close-book \
	--output_path results/streamingqa/mistral-7b-instruct.close-book.json \

python eval_streaming_qa.py \
	--data streamingqa \
	--model_name ../../libs/CaMeLS/outputs/eval/squad/uniform-uniform/mistralai-Mistral-7B-Instruct-v0.2/sft/final_model \
	--prompt_template context-question \
	--output_path results/streamingqa/mistral-7b-instruct.sft.context-question.json

python eval_streaming_qa.py \
	--data streamingqa \
	--model_name ../../libs/CaMeLS/outputs/eval/squad/uniform-uniform/mistralai-Mistral-7B-Instruct-v0.2/sft/final_model \
	--prompt_template rag-question \
	--output_path results/streamingqa/mistral-7b-instruct.sft.rag-question.json \

python eval_streaming_qa.py \
	--data streamingqa \
	--model_name ../../libs/CaMeLS/outputs/eval/squad/uniform-uniform/mistralai-Mistral-7B-Instruct-v0.2/sft/final_model \
	--prompt_template question \
	--decoding_setting close-book \
	--output_path results/streamingqa/mistral-7b-instruct.sft.question.json \


### With knowledge injection

python eval_streaming_qa.py \
	--data streamingqa \
	--model_name ../../data/models-dev/fastlora.Mistral7BInstructv02.sft-v4.w1024-pre-norm-sum.kinf.ri1024.r128.a64.o.svd.bs8.lr1e-4.pt-mix-sft-v4.20240927-003203/checkpoint-3428 \
	--prompt_template context-question \
	--merge_strategy sequential \
	--output_path results/streamingqa/mistral-7b-instruct.fastlora.context-question.json \

python eval_streaming_qa.py \
	--data streamingqa \
	--model_name ../../data/models-dev/fastlora.Mistral7BInstructv02.sft-v4.w1024-pre-norm-sum.kinf.ri1024.r128.a64.o.svd.bs8.lr1e-4.pt-mix-sft-v4.20240927-003203/checkpoint-3428 \
	--prompt_template rag-question \
	--merge_strategy sequential \
	--output_path results/streamingqa/mistral-7b-instruct.fastlora.rag-question.json \

python eval_streaming_qa.py \
	--data squad \
	--streamingqa ../../data/models-dev/fastlora.Mistral7BInstructv02.sft-v4.w1024-pre-norm-sum.kinf.ri1024.r128.a64.o.svd.bs8.lr1e-4.pt-mix-sft-v4.20240927-003203/checkpoint-3428 \
	--prompt_template question \
	--merge_strategy sequential \
	--output_path results/streamingqa/mistral-7b-instruct.fastlora.question.json \


python eval_streaming_qa.py \
	--data streamingqa \
	--model_name ../../libs/CaMeLS/outputs/eval/squad/uniform-uniform/mistralai-Mistral-7B-Instruct-v0.2/ep8-sft/final_model \
	--prompt_template question \
	--decoding_setting close-book \
	--output_path results/streamingqa/mistral-7b-instruct.cp-sft.ep8-question.json \

