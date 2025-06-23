import json
import codecs
import random
from pathlib import Path
from typing import Any, Callable, Dict, List, NewType, Optional, Tuple, Union
import os
import math

import torch
from torch.utils.data import Dataset, Sampler

import json
import random
from datetime import datetime
from time import sleep
import logging
import argparse
from tqdm import tqdm
import csv
import os
import transformers
import torch.utils.data

import torch
from transformers import AutoTokenizer, AutoConfig, AutoModelForCausalLM

os.environ["TOKENIZERS_PARALLELISM"] = "false"  # To suppress warnings about parallelism in tokenizers
logger = logging.getLogger(__name__)


from transformers.trainer_callback import TrainerCallback, ExportableState
class CustomizedEarlyStoppingCallback(TrainerCallback, ExportableState):

    def __init__(self, early_stopping_patience: int = 1, early_stopping_threshold: Optional[float] = 0.0):
        self.early_stopping_patience = early_stopping_patience
        self.early_stopping_threshold = early_stopping_threshold
        # early_stopping_patience_counter denotes the number of times validation metrics failed to improve.
        self.early_stopping_patience_counter = 0
        self.best_metric = None

    def check_metric_value(self, args, state, control, metric_value):
        import numpy as np
        # best_metric is set by code for load_best_model
        operator = np.greater if args.greater_is_better else np.less
        if self.best_metric is None or (
            operator(metric_value, self.best_metric)
            and abs(metric_value - self.best_metric) > self.early_stopping_threshold
        ):
            self.early_stopping_patience_counter = 0
        else:
            self.early_stopping_patience_counter += 1
        
        if self.best_metric is None:
            self.best_metric = metric_value
        else:
            self.best_metric = max(self.best_metric, metric_value) if args.greater_is_better else min(self.best_metric, metric_value)
        # print(f"self.best_metric: {self.best_metric}, metric_value: {metric_value}, self.early_stopping_patience_counter: {self.early_stopping_patience_counter}")

    def on_train_begin(self, args, state, control, **kwargs):
        # assert args.load_best_model_at_end, "EarlyStoppingCallback requires load_best_model_at_end = True"
        assert (
            args.metric_for_best_model is not None
        ), "EarlyStoppingCallback requires metric_for_best_model is defined"
        # assert (
        #     args.eval_strategy != IntervalStrategy.NO
        # ), "EarlyStoppingCallback requires IntervalStrategy of steps or epoch"

    def on_evaluate(self, args, state, control, metrics, **kwargs):
        metric_to_check = args.metric_for_best_model
        if not metric_to_check.startswith("eval_"):
            metric_to_check = f"eval_{metric_to_check}"
        metric_value = metrics.get(metric_to_check)

        if metric_value is None:
            logger.warning(
                f"early stopping required metric_for_best_model, but did not find {metric_to_check} so early stopping"
                " is disabled"
            )
            return

        self.check_metric_value(args, state, control, metric_value)
        if self.early_stopping_patience_counter >= self.early_stopping_patience:
            control.should_training_stop = True

    def state(self) -> dict:
        return {
            "args": {
                "early_stopping_patience": self.early_stopping_patience,
                "early_stopping_threshold": self.early_stopping_threshold,
            },
            "attributes": {
                "early_stopping_patience_counter": self.early_stopping_patience_counter,
            },
        }


class MetaICLDataset(torch.utils.data.Dataset):

    TOKENIZER_DATA = transformers.AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')

    def __init__(self, path, tokenizer):
        test_dp = torch.load(path)
        self.n = len(test_dp['input'])
        input_text_list = [self.TOKENIZER_DATA.decode(test_dp['input'][i]) for i in range(self.n)]
        output_text_list = [self.TOKENIZER_DATA.decode(test_dp['output'][i]) for i in range(self.n)]
        self.input_ids = []
        self.labels = []
        for input_text, output_text in zip(input_text_list, output_text_list):
            input_ids = tokenizer.apply_chat_template(
                [{"role": "user", "content": "Input: " + input_text + "\nOutput:"}],
                tokenize=True, add_generation_prompt=True,
            )
            output_ids = tokenizer(output_text, add_special_tokens=False)["input_ids"] + [tokenizer.eos_token_id]
            self.input_ids.append(input_ids + output_ids)
            self.labels.append([-100] * len(input_ids) + output_ids)
        self.eos_token_id = tokenizer.eos_token_id

    def get_max_len(self):
        return max(len(ids) for ids in self.input_ids)

    def __getitem__(self, idx):
        # item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        # if self.labels:
        #     item["labels"] = torch.tensor(self.labels[idx])
        # return item
        return {
            "input_ids": self.input_ids[idx],
            "labels": self.labels[idx],
        }

    def __len__(self):
        # return len(self.encodings["input_ids"])
        return self.n
    
    def collate_fn(self, batch):
        input_ids = [torch.tensor(item["input_ids"]) for item in batch]
        labels = [torch.tensor(item["labels"]) for item in batch]
        input_ids = torch.nn.utils.rnn.pad_sequence(input_ids, batch_first=True, padding_value=self.eos_token_id)
        labels = torch.nn.utils.rnn.pad_sequence(labels, batch_first=True, padding_value=-100)
        return {
            "input_ids": input_ids,
            "labels": labels,
        }

def finetune(model_name, dataset_name, device='cuda'):
    from transformers import TrainingArguments, Trainer
    from transformers import EarlyStoppingCallback
    import wandb

    # if wandb is not init, init it
    if not wandb.run:
        wandb.init(project="meta-icl")

    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        torch_dtype=torch.bfloat16,
        attn_implementation="sdpa",
        # attn_implementation="flash_attention_2",
    )
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    print('model', model)

    train_dataset = MetaICLDataset(f'../../data/metaicl/llama/{dataset_name}/{dataset_name}_16_100_train.jsonl', tokenizer)
    val_dataset = MetaICLDataset(f'../../data/metaicl/llama/{dataset_name}/{dataset_name}_16_87_train.jsonl', tokenizer)
    
    local_batch_size = max(1, min(16, 1024 // train_dataset.get_max_len()))
    global_batch_size = 16

    args = TrainingArguments(
        output_dir="logs",
        report_to="wandb",
        gradient_checkpointing=True,
        per_device_train_batch_size=local_batch_size,
        # learning_rate=5e-6,
        learning_rate=1e-5,
        lr_scheduler_type="linear",
        warmup_ratio=0.1,
        num_train_epochs=40,
        gradient_accumulation_steps=global_batch_size // local_batch_size,
        metric_for_best_model="eval_loss",
        eval_strategy="epoch",
        save_strategy="no",
    )
    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=train_dataset.collate_fn,
        callbacks=[CustomizedEarlyStoppingCallback(early_stopping_patience=3)],
    )
    trainer.train()
    model = model.eval().to(device)
    return model, tokenizer

def eval_icl(model, tokenizer, dataset_name, n_train_shot, seed=0, enable_fastlora=False, enable_finetune=False, data_dir=None, merge_strategy=None, window_size=None, max_new_tokens=10, stop=["\n"], **kwargs):

    random.seed(seed)

    print("dataset_name", dataset_name)
    print("n_train_shot", n_train_shot)

    # assert data_dir is not None
    # dataset_dir = os.path.join(data_dir, dataset_name)
    # train_data = name_to_dataset[dataset_name](dataset_dir, mode='train')
    # dev_data = name_to_dataset[dataset_name](dataset_dir, mode='dev')

    tokenizer_data = transformers.AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf')
    test_dp = torch.load(f'../../data/metaicl/llama/{dataset_name}/{dataset_name}_16_100_test.jsonl')
    n_test = len(test_dp['input'])
    test_input_text = [tokenizer_data.decode(test_dp['input'][i]) for i in range(n_test)]
    test_output_text = [tokenizer_data.decode(test_dp['output'][i]) for i in range(n_test)]

    # FIXME: for efficency
    n_test = max(16, n_test // 10)
    test_input_text = test_input_text[:n_test]
    test_output_text = test_output_text[:n_test]
    print(f"n_test: {n_test}")

    all_test_cases_list = []

    if not enable_finetune:
        train_seed_list = [13, 21, 42, 87, 100]
    else:
        train_seed_list = [None] 

    for train_seed in tqdm(train_seed_list, desc="train_seed"):

        if enable_finetune:
            model, tokenizer = finetune(kwargs["model_name"], dataset_name)
            prompt_prefix = ""
        else:

            train_dp = torch.load(f'../../data/metaicl/llama/{dataset_name}/{dataset_name}_16_{train_seed}_train.jsonl')
            n_train = len(train_dp['input'])
            train_input_text = [tokenizer_data.decode(train_dp['input'][i]) for i in range(n_train)]
            train_output_text = [tokenizer_data.decode(train_dp['output'][i]) for i in range(n_train)]
            indices = random.sample(range(n_train), n_train_shot)
            train_input_text = [train_input_text[i] for i in indices]
            train_output_text = [train_output_text[i] for i in indices]

            print('n_train', n_train)

            # inference
            # train_data.subsamplebyshot(n_train_shot, seed)
            # logger.info(f"===== eval on {dev_data.__len__()} dev examples =====")
            # prompt_prefix = make_prompt(train_data, dataset_name, mode='train')
            # prompt_prefix = "You are given multiple examples of a text classification task.\n\n" + prompt_prefix

            prompt_prefix = "\n\n".join([f"Input: {train_input_text[j]}\nOutput: {train_output_text[j]}" for j in range(n_train_shot)])

        context_len = len(tokenizer.encode(prompt_prefix))
        if context_len > 4096:
            print(f"Warning: context length {context_len} exceeds 4096, may cause OOM error.")
            return {
                "acc": None,
                "data": None,
            }

        if enable_fastlora:
            from fastlora.eval_utils import fastlora_generate_adaptor
            lora_weights = fastlora_generate_adaptor(
                model, tokenizer, 
                prompt_prefix, 
                merge_strategy=merge_strategy, max_window_size=window_size,
            )
        else:
            lora_weights = None

        results_list = []
        # for i, ins in enumerate(tqdm(dev_data.data, total=dev_data.__len__())):
        for i, (input_text, answer_text) in enumerate(tqdm(zip(test_input_text, test_output_text))):
            # prompt = prompt_prefix + make_prompt(ins, dataset_name, mode='inference')
            # prompt_input = make_prompt(ins, dataset_name, mode='inference')
            
            # prompt_input = "Classify the following case using one of the labels demonstrated. Provide only the single-word label as your response.\n\n" + prompt_input

            prompt_input = "Input: " + input_text + "\nOutput:"
            if kwargs["prompt_template"] == 'instruction':
                # prompt_input = "## Instruction: Based on the demonstration above, provide a one-word answer only, without any explanation or additional words.\n\n" + prompt_input
                prompt_input = f"## Instruction: Based on the demonstration above, provide a short and concise answer, without any explanation or additional words.\n\n{prompt_input}"
            else:
                pass

            # gen_logits = llm_gen(model, prompt, tokenizer, max_context_len)
            # dev_pred.append(parse_response(gen_logits, tokenizer, id2verb))

            if enable_fastlora:
                from fastlora.eval_utils import fastlora_conditional_generate
                output_text, input_text_proc = fastlora_conditional_generate(
                    model, tokenizer, 
                    input_text=prompt_input, use_chat=True,
                    mode="weights", lora_weights=lora_weights, 
                    max_new_tokens=max_new_tokens,
                    stop=stop,
                    return_input_text=True,
                )
            elif "ultragist" in kwargs["model_name"]:
                from fastlora.eval_utils import ultragist_conditional_generate
                output_text, input_text_proc, metainfo = ultragist_conditional_generate(
                    model, tokenizer, 
                    context_text=prompt_prefix, 
                    input_text=input_text, use_chat=True,
                    max_new_tokens=max_new_tokens,
                    return_input_text=True,
                    stop=stop,
                )
                print(f"input_len: {len(tokenizer(input_text_proc).input_ids)}, metainfo: {metainfo}")
            else:
                from fastlora.eval_utils import default_conditional_generate
                output_text, input_text_proc = default_conditional_generate(
                    model, tokenizer, 
                    input_text=prompt_input, context_text=prompt_prefix, 
                    max_new_tokens=max_new_tokens,
                    stop=stop,
                    return_input_text=True,
                )
                # replace the occurence of prompt_prefix in the input_text with "CONTEXT" symbol
                if prompt_prefix:
                    input_text_proc = input_text_proc.replace(prompt_prefix, "{{CONTEXT}}")

            # assert isinstance(output_text, str)
            # # find the first occurrence of each class
            # first_occur = {label: output_text.lower().find(label) for label in id2verb}
            # # get the key with the smallest value
            # pred = min([idx for idx in range(len(id2verb)) if first_occur[id2verb[idx]] != -1], key=lambda x: first_occur[id2verb[x]], default=None)

            # dev_pred.append(pred)
            # dev_labels.append(label2id[ins['label']])

            if i == 0:
                print(f'prompt_prefix', prompt_prefix)
                print(f"input_text: {input_text_proc}")
                print(f"output_text: {output_text}")
                print(f"label: {answer_text}")

            output_text = output_text.replace("Output:", "").strip()

            acc = output_text.lower().strip() == answer_text.lower().strip()
            # f1 between output_text and answer_text
            from squad_utils import f1_score
            qa_f1 = f1_score(output_text, answer_text)
            # f1 = f1_score(output_text, answer)
            
            results_list.append({
                "input": input_text_proc,
                "output": output_text,
                "label": answer_text,
                "acc": acc,
                "qa_f1": qa_f1,
            })
        
        all_test_cases_list.append({
            "acc": sum([res["acc"] for res in results_list]) / len(results_list),
            "qa_f1": sum([res["qa_f1"] for res in results_list]) / len(results_list),
            "context": prompt_prefix,
            "context_len": len(tokenizer.encode(prompt_prefix)),
            "eval": results_list,
        })

        if enable_finetune:
            del model

    return {
        "acc": sum([res["acc"] for res in all_test_cases_list]) / len(all_test_cases_list),
        "qa_f1": sum([res["qa_f1"] for res in all_test_cases_list]) / len(all_test_cases_list),
        "data": all_test_cases_list,
    }


def main(args):

    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
    )
    logger.setLevel(logging.INFO)

    # device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    # # set pad token ids for batched inference cus gpt2 does not have one
    # tokenizer.padding_side = "left"
    # tokenizer.pad_token = tokenizer.eos_token
    # tokenizer.pad_token_id = tokenizer.eos_token_id
    # # model_config = AutoConfig.from_pretrained(args.model_name)
    # model = AutoModelForCausalLM.from_pretrained(args.model_name)
    # model.to(device)
    # model.eval()

    if args.dataset is None:
        with open("../../data/metaicl/hr_to_lr.json", "r") as f:
            config = json.load(f)
        dataset_list = config["test"]
    else:
        dataset_list = [args.dataset]
    
    if args.n_train_shot is None:
        n_train_shot_list = [1, 2, 4, 8, 16]
    else:
        n_train_shot_list = [args.n_train_shot]
    if args.decoding_setting == "zero-shot":
        n_train_shot_list = [0]
    if args.finetune:
        n_train_shot_list = [16]
    if "ultragist" in args.model_name:
        n_train_shot_list = [16]
        

    print("dataset_list", dataset_list)
    print("n_train_shot_list", n_train_shot_list)
    
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not args.finetune:
        from fastlora.eval_utils import load_model_and_tokenizer
        model, tokenizer = load_model_and_tokenizer(args.model_name, device=device, fastlora_params={"fastlora_merge": "pre-norm-sum"})
    
    else:
        model, tokenizer = None, None

    results_all = {}
    for dataset_name in dataset_list:

        for n_train_shot in n_train_shot_list:
            
            if "fastlora" in args.model_name:
                results = eval_icl(
                    model, tokenizer, dataset_name, n_train_shot=n_train_shot, seed=args.seed, enable_fastlora=True, 
                    data_dir=args.data_dir, merge_strategy=args.merge_strategy, window_size=args.window_size, max_new_tokens=10, stop=["\n"],
                    prompt_template=args.prompt_template, model_name=args.model_name,
                )
            else:
                results = eval_icl(
                    model, tokenizer, dataset_name, n_train_shot=n_train_shot, seed=args.seed, 
                    enable_fastlora=False, enable_finetune=args.finetune,
                    data_dir=args.data_dir, max_new_tokens=10, stop=["\n"], 
                    prompt_template=args.prompt_template, model_name=args.model_name,
                )
            acc = results["acc"]
            logger.info(f"Acc: {acc}")

            results_all[f"acc@{dataset_name}@{n_train_shot}"] = acc
            results_all[f"data@{dataset_name}@{n_train_shot}"] = results["data"]

            # save the results to a file
            Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(args.output_path, "w") as f:
                json.dump(results_all, f, indent=2)

            # logging
            save_results_file = args.output_path.removesuffix('.json') + '.csv'
            csv_exists = os.path.isfile(save_results_file)
            with open(save_results_file, 'a+', newline='') as csvfile:
                csvwriter = csv.writer(csvfile)
                if not csv_exists:
                    csvwriter.writerow(['dataset', 'llm', 'n_train_shot', 'seed', 'acc'])
                csvwriter.writerow([dataset_name,
                                    args.model_name,
                                    n_train_shot,
                                    args.seed,
                                    acc])

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="In-Context Learning baseline.")
    parser.add_argument(
        "--model_name",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default=None,
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--n_train_shot",
        type=int,
        default=None,
    )
    parser.add_argument(
        "--output_path",
        type=str,
        default=None,
    )
    parser.add_argument("--use_chat", action="store_true")
    parser.add_argument("--use_chat_context", action="store_true")
    parser.add_argument("--decoding_setting", type=str, default=None)
    parser.add_argument("--finetune", action="store_true")
    parser.add_argument("--prompt_template", type=str, default=None)
    parser.add_argument("--merge_strategy", choices=["concat", "parallel", "sequential", "sequential-long"], default="concat")
    parser.add_argument("--window_size", type=int, default=1024)
    args = parser.parse_args()
    main(args)