import numpy as np
import pandas as pd
from datasets import load_dataset
import torch
from fastlora.eval_utils import fastlora_generate_adaptor, fastlora_conditional_generate, default_conditional_generate
from tqdm import tqdm

from squad_utils import f1_score, exact_match_score, metric_max_over_ground_truths
from rouge import Rouge
from fastlora.eval_utils import normalize_text
from collections import defaultdict

class TextAndQuestionDataset:
    def __init__(self, max_text_len = 1024, max_question_len = 128, device = None, loc = False, qa_only = False, qa_for_generation=False, max_answer_len=24, tokenizer = 'gpt2', prompt_samples = -1, pad_qa_for_gen=True, include_eos = True):
        self.device = device if device is not None else ('cuda' if torch.cuda.is_available() else 'cpu')
        if isinstance(tokenizer, str):
            from transformers import AutoTokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer)
        else:
            self.tokenizer = tokenizer
        self.tokenizer.pad_token = self.tokenizer.eos_token 
        self.max_text_len = max_text_len
        self.qa_for_generation = qa_for_generation
        self.qa_only = qa_only
        self.max_question_len = max_question_len 
        self.max_answer_len = max_answer_len
        self.loc = loc
        self.prompt_samples = prompt_samples
        self.pad_qa_for_gen = pad_qa_for_gen
        self.include_eos = include_eos
    def __len__(self):
        raise NotImplementedError("Subclasses must implement __len__")
    
    def get_qa(self, idx):
        #return text corresponding to a question and answer pair at index idx
        #we expect answer to not have a space at the beginning
        raise NotImplementedError("Subclasses must implement get_qa")
    
    def get_text(self, idx):
        #return text corresponding to the passage with information at index idx
        raise NotImplementedError("Subclasses must implement get_text")
    
    def __getitem__(self, idx):
        qa_ids, qa_attention, qa_target_ids = self.tok_qa_for_training(idx)
        if self.loc:
            return {'loc_ids': qa_ids.squeeze().to(self.device), 
                    'loc_attention': qa_attention.squeeze().to(self.device), 
                    'loc_mask': torch.roll(qa_target_ids.squeeze().to(self.device) != -100, -1, 0)}
        if self.qa_only:
            return_dic =  {'idx': torch.tensor(idx).to(self.device),
                    'qa_ids': qa_ids.squeeze().to(self.device), 
                    'qa_attention': qa_attention.squeeze().to(self.device),
                    'qa_target_ids': qa_target_ids.squeeze().to(self.device)}
        else:
            text = self.tokenizer(self.get_text(idx), max_length=self.max_text_len ,padding='max_length', truncation=True, return_tensors="pt" )
            return_dic =  {'idx': torch.tensor(idx).to(self.device),
                    'text_ids': text['input_ids'].squeeze().to(self.device), 
                    'text_attention': text['attention_mask'].squeeze().to(self.device), 
                    'qa_ids': qa_ids.squeeze().to(self.device), 
                    'qa_attention': qa_attention.squeeze().to(self.device),
                    'qa_target_ids': qa_target_ids.squeeze().to(self.device)}
        if self.qa_for_generation:
            return_dic.update(self.tok_qa_for_generation(idx))
        
        return return_dic

    @staticmethod
    def shuffle_groups(df, group_col):
        """
        Shuffles the order of groups in a Pandas DataFrame without shuffling the order of items within each group.

        Parameters:
        - df: the input DataFrame
        - group_col: the name of the column containing the groups to be shuffled

        Returns:
        - a shuffled copy of the input DataFrame
        """
        # Get a list of unique groups
        groups = df[group_col].unique()

        # Shuffle the list of groups
        np.random.shuffle(groups)

        # Define a sorting key that sorts by the shuffled order of groups
        def sort_key(row):
            return np.argwhere(groups == row[group_col])[0][0]

        df['temp'] = df.apply(sort_key, axis=1)
        shuffled_df = df.sort_values('temp', kind='stable').drop('temp', axis=1).reset_index(drop=True)
        return shuffled_df

    #given a pd dataframe, return a head of the dataframe such that column column has k unique values
    @staticmethod
    def return_k_unique(df, k, column): 
        if k >= len(df[column].unique()):
            return df
        else:
            values_to_keep = df[column].unique()[:k]
            return df[df.apply(lambda x: x[column] in values_to_keep, axis=1)]

#%%
class StreamingQADataset(TextAndQuestionDataset):
    def __init__(self, csv_path, downsample_to = -1, **kwargs):
        self.csv_path = csv_path
        self.data_frame = pd.read_csv(csv_path)
        if downsample_to != -1 and downsample_to < len(self.data_frame):
            print('downsampling from ', len(self.data_frame), ' to ', downsample_to, ' examples')
            self.data_frame = self.data_frame.sample(downsample_to)
        else:
            self.data_frame = self.data_frame.sample(frac=1)
        super().__init__(**kwargs)

    def __len__(self):
        return len(self.data_frame)
    
    def get_qa(self, idx):
        row = self.data_frame.iloc[idx]
        answers = row['answers'].split("\\")
        answer = min(answers, key = len)
        question = row['question'].strip() 
        return question, answer
    
    def get_text(self, idx):
        return self.data_frame.iloc[idx]['text']
    
class SquadDataset(TextAndQuestionDataset):
    
    def __init__(self, split, start_idx = 0, end_idx = -1, shuffle_by='title', downsample_to=-1, downsample_by='context',**kwargs):
        squad_ds = load_dataset('squad', split=split)
        if end_idx == -1:
            end_idx = len(squad_ds)
        squad_ds = squad_ds.select(list(range(start_idx,end_idx)))
        self.data_frame = pd.DataFrame(squad_ds)
        # self.data_frame = self.shuffle_groups(self.data_frame, shuffle_by)
        if downsample_to > 0:
            self.data_frame = self.return_k_unique(self.data_frame, downsample_to, downsample_by)
        super().__init__(**kwargs)
        
    def __len__(self):
        return len(self.data_frame)
        
    def get_qa(self, idx):
        question = self.data_frame.iloc[idx]['question'].strip() 
        answer = min(self.data_frame.iloc[idx]['answers']['text'], key = len).strip()
        if answer[0].islower():
            answer = answer[0].upper() + answer[1:]
        return question, answer
    
    def get_text(self, idx):
        return self.data_frame.iloc[idx]['context']
  
class ArchivalQADataset(TextAndQuestionDataset):
    def __init__(self, csv_path, full_passage = False, shuffle_by='doc_id', downsample_to=-1,downsample_by='ans_paragraph', **kwargs):
        self.csv_path = csv_path
        self.full_passage = full_passage
        self.data_frame = pd.read_csv(csv_path)
        #we sort pre shuffle to make sure that for any given doc_id, the examples are in increasing order of para_num
        self.data_frame.sort_values('para_num', kind='stable', inplace=True)
        self.data_frame = self.shuffle_groups(self.data_frame, shuffle_by)
        if downsample_to > 0:
            self.data_frame = self.return_k_unique(self.data_frame, downsample_to, downsample_by)
        super().__init__(**kwargs)

    def __len__(self):
        return len(self.data_frame)
    
    def get_qa(self, idx):
        row = self.data_frame.iloc[idx]
        answer = row['answer']
        if answer[0].islower():
            answer = answer[0].upper() + answer[1:]
        question = row['question'].strip() 
        return question, answer
    
    def get_text(self, idx):
        if self.full_passage:
            return self.data_frame.iloc[idx]['ans_text']
        return self.data_frame.iloc[idx]['ans_paragraph']


@torch.inference_mode()
def eval_squad(dataset, model, tokenizer, context_len=None, max_new_tokens=20, max_tokens=None, stop=["\n"], enable_fastlora=False, fastlora_mode="weights", **kwargs):

    np.random.seed(0)
    torch.manual_seed(0)

    results = []
    rouge_scorer = Rouge()

    text_to_qa = defaultdict(list)
    for i in range(len(dataset)):
        context = dataset.get_text(i)
        question, answer = dataset.get_qa(i)
        text_to_qa[context].append((question, answer))
    all_context_list = list(text_to_qa.keys())
    all_qa_list = sum([text_to_qa[context] for context in all_context_list], start=[])

    if context_len is None:
        context_list_per_group = [None]
        num_groups = 1
    else:
        num_tokens = len(tokenizer("\n\n".join(all_context_list))["input_ids"])
        num_groups = (num_tokens + context_len - 1) // context_len
        context_list_per_group = np.array_split(all_context_list, num_groups)
    
    # FIXME: debug, first make sure at least evaluate 500 QA pairs for each context length. 10% qas. 10 -> 1, 100 -> 10
    context_list_per_group = context_list_per_group[:max(1, num_groups // 10)]

    for context_list in tqdm(context_list_per_group):
        if context_list is None:
            qa_list = all_qa_list
            context_text = None
        else:
            qa_list = sum([text_to_qa[context] for context in context_list], start=[])
            context_text = "\n\n".join(context_list)
            if enable_fastlora:
                lora_weights = fastlora_generate_adaptor(
                    model, tokenizer, 
                    context_text, 
                    merge_strategy=kwargs["merge_strategy"], max_window_size=kwargs["window_size"]
                )
        
        # # FIXME: debug, first make sure at least evaluate 500 QA pairs for each context length. 10% qas. 10 -> 1, 100 -> 10
        # qa_list = qa_list[:len(qa_list) // 10]

        disable_context = False
        results_per_group = []
        for question, answer in tqdm(qa_list):

            if kwargs["prompt_template"].endswith("question"):
                input_text = question
            elif kwargs["prompt_template"].endswith("instruction"):
                input_text = f"## Instruction: Answer the question based on the context above. Respond with a short phrase only. Keep the answer short and concise, without any explanation or additional words\n\nQuestion: {question}\nAnswer:"
            else:
                raise ValueError(f"Invalid prompt_template: {kwargs['prompt_template']}")

            if kwargs["prompt_template"].startswith("rag"):
                assert context_text is not None
                from rank_bm25 import BM25Okapi
                import nltk
                # split context_text into groups of 100-word chunks
                chunk_size = 100
                tokenized_context = nltk.word_tokenize(context_text.lower())
                chunks = [" ".join(tokenized_context[i:i+chunk_size]) for i in range(0, len(tokenized_context), chunk_size)]
                tokenized_chunks = [nltk.word_tokenize(chunk) for chunk in chunks]
                # Initialize BM25
                bm25 = BM25Okapi(tokenized_chunks)

                tokenized_query = nltk.word_tokenize(question.lower())
                # Get BM25 scores for the query
                scores = bm25.get_scores(tokenized_query)

                # Find the passage with the highest score
                best_score_index = scores.argmax()
                best_chunk = chunks[best_score_index]
                input_text = f"{best_chunk}\n\n" + input_text
                disable_context = True
            
            if kwargs["prompt_template"].startswith("context"):
                assert context_text is not None
                input_text = f"{context_text}\n\n" + input_text
                disable_context = True
            
            
            if enable_fastlora:
                assert not disable_context, "FastLoRA requires context to be provided."
                output_text, input_text_proc = fastlora_conditional_generate(
                    model, tokenizer, 
                    input_text=input_text, use_chat=True,
                    mode=fastlora_mode, lora_weights=lora_weights, 
                    max_new_tokens=max_new_tokens,
                    return_input_text=True,
                    stop=stop,
                )
            elif "ultragist" in kwargs["model_name"]:
                from fastlora.eval_utils import ultragist_conditional_generate
                output_text, input_text_proc, metainfo = ultragist_conditional_generate(
                    model, tokenizer, 
                    context_text=context_text if not disable_context else None,
                    input_text=input_text, use_chat=True,
                    max_new_tokens=max_new_tokens,
                    return_input_text=True,
                    stop=stop,
                )
                print(f"metainfo: {metainfo}")
            else:
                if max_tokens is not None:
                    input_text_input_ids = tokenizer(input_text)["input_ids"]
                    input_text_input_ids = input_text_input_ids[-max_tokens:]
                    input_text = tokenizer.decode(input_text_input_ids, skip_special_tokens=True)
                # if kwargs["use_chat_context"]:
                #     context_input_ids = tokenizer.apply_chat_template(
                #         [{"role": "user", "content": context_text}],
                #         tokenize=True,
                #     )
                output_text, input_text_proc = default_conditional_generate(
                    model, tokenizer, 
                    context_text=context_text if not disable_context else None,
                    input_text=input_text, use_chat=True,
                    max_new_tokens=max_new_tokens,
                    return_input_text=True,
                    stop=stop,
                )
                if context_text is not None:
                    input_text_proc = input_text_proc.replace(context_text, "{{CONTEXT}}")

            if normalize_text(answer) == "" or normalize_text(output_text) == "":
                f1, em, rouge_score = 0.0, 0.0, 0.0
            else:
                f1 = f1_score(output_text, answer)
                em = exact_match_score(output_text, answer)
                rouge_score = rouge_scorer.get_scores(normalize_text(output_text), normalize_text(answer), avg=True)["rouge-l"]["r"]

            # print(f'shape of input_ids: {input_ids.shape}')
            # print(f"f1: {f1}, em: {em}, rouge: {rouge_score}, prediction: {output_text}, ground truth: {answer}")

            results_per_group.append({
                "input": input_text_proc,
                "output": output_text,
                "answer": answer,
                "f1": f1,
                "em": em,
                "rouge": rouge_score,
            })
        results.append({
            "context": context_text,
            "qa": results_per_group,
        })
    get_metric = lambda x: np.mean([item[x] for group in results for item in group["qa"]])
    return {
        "f1": get_metric("f1"),
        "em": get_metric("em"),
        "rouge": get_metric("rouge"),
        "data": results,
    }


def main(args):
    from fastlora.eval_utils import load_model_and_tokenizer, fastlora_generate, normalize_text
    from pathlib import Path
    import json

    model, tokenizer = load_model_and_tokenizer(args.model_name, device=args.device, fastlora_params={"fastlora_merge": "pre-norm-sum"})
    # dataset_context = dataset.get_deduplicated_dataset()
    # num_docs_list = [8, 16, 32, 64, 128, 256, 512, 1024]
    # num_docs_list = [256, 512, 1024]
    # num_docs_list = [4]
    # context_len_list = [512, 1024, 2048, 4096, 8192, 16384, 32768, 65536, int(1e9)]
    # context_len_list = [512]
    max_new_tokens = 20
    stop = ["\n", "Question:"]

    np.random.seed(0)
    torch.manual_seed(0)

    if args.data == 'squad':
        dataset = SquadDataset(split='validation', start_idx=0, end_idx=-1, shuffle_by='title')
    elif args.data == 'streamingqa':
        dataset = StreamingQADataset(csv_path='streaming-qa/test.csv')

    if args.decoding_setting == "close-book":
        results = eval_squad(
            dataset, model, tokenizer, 
            context_len=None, 
            max_new_tokens=max_new_tokens, stop=stop, **vars(args)
        )
        # save the results to a file
        Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(args.output_path, "w") as f:
            json.dump(results, f, indent=2)
    else:
        context_len_list = [512, 1024, 2048, 4096, 8192, 16384, 32768]
        # context_len_list = [16384]
        # if args.merge_strategy == "sequential-long":
        #     # context_len_list = [1048576]
        #     context_len_list = [32768]
        enable_fastlora = "fastlora" in args.model_name.lower()
        
        # FIXME: temporary fix for Llama2 model
        max_tokens = 4096 if "llama-2" in args.model_name.lower() and not enable_fastlora else None

        output_dict = {}
        for context_len in context_len_list:
            results = eval_squad(
                dataset, model, tokenizer, 
                context_len=context_len, enable_fastlora=enable_fastlora,
                max_tokens=max_tokens,
                max_new_tokens=max_new_tokens, stop=stop, **vars(args)
            )
            output_dict[f"f1@{context_len}"] = results["f1"]
            output_dict[f"em@{context_len}"] = results["em"]
            output_dict[f"rouge@{context_len}"] = results["rouge"]
            output_dict[f"data@{context_len}"] = results["data"]

            # save the results to a file
            Path(args.output_path).parent.mkdir(parents=True, exist_ok=True)
            with open(args.output_path, "w") as f:
                json.dump(output_dict, f, indent=2)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default=None)
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--use_chat_context", action="store_true")
    parser.add_argument("--prompt_template", type=str, default=None)
    parser.add_argument("--merge_strategy", choices=["concat", "parallel", "sequential", "sequential-long"], default="concat")
    parser.add_argument("--window_size", type=int, default=1024)
    parser.add_argument("--decoding_setting", type=str, default=None)
    parser.add_argument("--device", type=str, default="cuda")
    parser.add_argument("--output_path", type=str, default=None)
    args = parser.parse_args()

    main(args)
