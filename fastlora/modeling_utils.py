import math
import torch
from tqdm import tqdm
from dataclasses import dataclass
from contextlib import nullcontext
from typing import Mapping, Optional, Tuple
from accelerate import Accelerator
from collections import defaultdict
from transformers.modeling_outputs import BaseModelOutputWithPast
from datasets import load_dataset


def optional_grad_ctx(with_grad=False):
    if with_grad:
        return nullcontext()
    else:
        return torch.no_grad()

def move_to_device(data, device):
    """
    Prepares one `data` before feeding it to the model, be it a tensor or a nested list/dictionary of tensors.
    """
    if isinstance(data, Mapping):
        return type(data)({k: move_to_device(v, device) for k, v in data.items()})
    elif isinstance(data, (tuple, list)):
        return type(data)(move_to_device(v, device) for v in data)
    elif isinstance(data, torch.Tensor):
        kwargs = {"device": device}
        return data.to(**kwargs)
    else:
        return data

def compute_loss(logits, labels, shift=False):
    """
    Returns:
        token_loss: batch_size, seq_length
    """
    if shift:
        logits = logits[:, :-1, :].contiguous()
        labels = labels[:, 1:].contiguous()

    labels = labels.to(logits.device)
    batch_size = logits.shape[0]

    # NOTE: the loss on -100 labels is 0 by default
    token_loss = torch.nn.functional.cross_entropy(
        logits.flatten(0, 1), 
        labels.reshape(-1), 
        reduction="none"
    ).reshape(batch_size, -1)   # batch_size, seq_len
    
    valid_token_num = (labels != -100).sum(-1)  # batch_size
    all_valid_token_num = valid_token_num.sum()
    
    if all_valid_token_num > 0:
        loss = token_loss.sum() / valid_token_num.sum()
    else:
        loss = token_loss.sum()

    batch_loss = token_loss.sum(-1) / valid_token_num
    # prevent nan
    if (valid_token_num == 0).any():
        batch_loss = batch_loss.masked_fill(valid_token_num == 0, 0.)

    return loss, batch_loss, valid_token_num


def compute_loss_context_input(model, input_ids_seq1, input_ids_seq2, attention_mask_seq1, attention_mask_seq2, label_seq1, label_seq2):
    from torch.nn.utils.rnn import pad_sequence
    import torch.nn.functional as F

    # cut the context into segments
    window_size = model.peft_config['default'].fastlora_window
    pad_token_id = model.config.pad_token_id if model.config.pad_token_id else model.config.eos_token_id
    
    number_windows = (input_ids_seq1.shape[-1] + window_size - 1) // window_size
    seq_len = (input_ids_seq1.shape[-1] + number_windows - 1) // number_windows
    input_ids_seq1 = F.pad(input_ids_seq1, (0, number_windows * seq_len - input_ids_seq1.shape[-1]), value=pad_token_id).reshape(-1, number_windows, seq_len)
    attention_mask_seq1 = F.pad(attention_mask_seq1, (0, number_windows * seq_len - attention_mask_seq1.shape[-1]), value=0).reshape(-1, number_windows, seq_len)
    label_seq1 = F.pad(label_seq1, (0, number_windows * seq_len - label_seq1.shape[-1]), value=-100).reshape(-1, number_windows, seq_len)
    
    # print("model.device", model.device)
    # print(f'{input_ids_seq1.shape} ({input_ids_seq1.dtype}, {input_ids_seq1.device}), {attention_mask_seq1.shape} ({attention_mask_seq1.dtype}, {attention_mask_seq1.device}), {label_seq1.shape} ({label_seq1.dtype}, {label_seq1.device})')
    # print(f'{input_ids_seq2.shape} ({input_ids_seq2.dtype}, {input_ids_seq2.device}), {attention_mask_seq2.shape} ({attention_mask_seq2.dtype}, {attention_mask_seq2.device}), {label_seq2.shape} ({label_seq2.dtype}, {label_seq2.device})')

    # >>> Mode 1: default
    assert input_ids_seq1.shape[0] == 1, "batch size should be 1"
    input_ids = pad_sequence([*input_ids_seq1.squeeze(0), input_ids_seq2.squeeze(0)], batch_first=True, padding_value=pad_token_id).unsqueeze(0)
    attention_mask = pad_sequence([*attention_mask_seq1.squeeze(0), attention_mask_seq2.squeeze(0)], batch_first=True, padding_value=0).unsqueeze(0)
    labels = pad_sequence([*label_seq1.squeeze(0), label_seq2.squeeze(0)], batch_first=True, padding_value=-100).unsqueeze(0)
    inputs = {}
    inputs["input_ids"] = input_ids
    inputs["attention_mask"] = attention_mask
    inputs["labels"] = labels

    # inputs["labels"] = inputs["input_ids"]
    # print(inputs["input_ids"].shape, inputs["input_ids"])
    # print(inputs["attention_mask"].shape, inputs["attention_mask"])
    # print(inputs["labels"].shape, inputs["labels"])
    # print(inputs)
    # for name, x in model.named_parameters():
    #     print(f"{name: ^80} {x.dtype}, {x.device}")

    outputs = model(**inputs)

    logits = outputs.logits
    loss, batch_loss, valid_token_num = compute_loss(logits.reshape((-1,) + logits.shape[-2:]), labels.reshape(-1, labels.shape[-1]), shift=True)
    batch_loss = batch_loss * valid_token_num
    batch_loss = batch_loss.reshape(logits.shape[0], logits.shape[1])
    valid_token_num = valid_token_num.reshape(logits.shape[0], logits.shape[1])
    valid_token_num = valid_token_num.sum(-1)
    batch_loss = batch_loss.sum(-1) / torch.clamp(valid_token_num, min=1)
    return loss, batch_loss, valid_token_num


@torch.no_grad()
def evaluate_perplexity(model, tokenizer, reconstruct_tokens=None, context_len=1024, input_len=1024):
    
    data = load_dataset("json", data_files="../../converted_data/eval_combined_qv_100.json")
    data = data["train"]

    # if accelerator is not None and type(dataloader) == torch.utils.data.DataLoader:
    #     # if the dataloader has been prepared, we shall not prepare it twice, especially in case of deepspeed
    #     dataloader = accelerator.prepare(dataloader)

    # if accelerator.process_index == 0:
    #     for name, x in model.named_parameters():
    #         print(f"{name: ^80} {x.dtype}")

    # all_loss = defaultdict(list)
    all_loss = []
    for i, x in enumerate(tqdm(data, desc="Computing Perplexity")):
        x = tokenizer(x["text"], return_tensors="pt")

        # prepare the context and the input
        input_ids_seq_1, input_ids_seq_2 = x["input_ids"][:, :context_len], x["input_ids"][:, context_len:]
        attention_mask_seq_1, attention_mask_seq_2 = x["attention_mask"][:, :context_len], x["attention_mask"][:, context_len:]
        label_seq_1, label_seq_2 = x["input_ids"][:, :context_len], x["input_ids"][:, context_len:]
        input_ids_seq_2 = input_ids_seq_2[:, :input_len]
        attention_mask_seq_2 = attention_mask_seq_2[:, :input_len]
        label_seq_2 = label_seq_2[:, :input_len]
        
        if reconstruct_tokens is not None:
            input_ids_seq_2, attention_mask_seq_2, label_seq_2 = input_ids_seq_1, attention_mask_seq_1, label_seq_1     # for reconstruction evaluation
            input_ids_seq_2, attention_mask_seq_2, label_seq_2 = input_ids_seq_2[:, :reconstruct_tokens], attention_mask_seq_2[:, :reconstruct_tokens], label_seq_2[:, :reconstruct_tokens]     # for short instruction evaluation
        # input_ids_seq_1, attention_mask_seq_1 = input_ids_seq_1[:, :64], attention_mask_seq_1[:, :64]     # for short context evaluation
        label_seq_1 = torch.full_like(input_ids_seq_1, -100)
        
        input_ids_seq_1 = input_ids_seq_1.to(model.device)
        input_ids_seq_2 = input_ids_seq_2.to(model.device)
        attention_mask_seq_1 = attention_mask_seq_1.to(model.device)
        attention_mask_seq_2 = attention_mask_seq_2.to(model.device)
        label_seq_1 = label_seq_1.to(model.device)
        label_seq_2 = label_seq_2.to(model.device)

        # print(input_ids_seq_1.shape, input_ids_seq_2.shape, attention_mask_seq_1.shape, attention_mask_seq_2.shape, label_seq_1.shape, label_seq_2.shape)

        loss, batch_loss, valid_token_num = compute_loss_context_input(model, input_ids_seq_1, input_ids_seq_2, attention_mask_seq_1, attention_mask_seq_2, label_seq_1, label_seq_2)

        all_loss.append(loss.item())
    
    perplexity = math.exp(sum(all_loss) / len(all_loss))
    return perplexity

@torch.no_grad()
def evaluate_squad(model, tokenizer):
    data = load_dataset("json", data_files="../../data/pretrain/eval-squad-100/data_eval_squad.jsonl")
    data = data["train"]

    all_loss = []
    # item = data[0]
    for item in tqdm(data):
        context_text = f"Title: {item['title']}\nPassage: {item['context']}"
        input_text = item['question']
        answer_text = item["answers"]["text"][0]

        context_text_ids = tokenizer(context_text, return_tensors="pt").input_ids
        input_text_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": input_text}],
            tokenize=True, add_generation_prompt=True, return_tensors="pt",
        )
        input_answer_text_ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": input_text}, {"role": "assistant", "content": answer_text}],
            tokenize=True, return_tensors="pt",
        )
        assert (input_answer_text_ids[:, :input_text_ids.shape[1]] == input_text_ids).all()

        num_label_tokens = input_answer_text_ids.shape[1] - input_text_ids.shape[1]

        context_text_ids = context_text_ids.to(model.device)
        input_answer_text_ids = input_answer_text_ids.to(model.device)
        input_ids_seq_1 = context_text_ids
        input_ids_seq_2 = input_answer_text_ids
        attention_mask_seq_1 = torch.ones_like(input_ids_seq_1)
        attention_mask_seq_2 = torch.ones_like(input_ids_seq_2)
        label_seq_1 = torch.full_like(input_ids_seq_1, -100)
        label_seq_2 = input_ids_seq_2.clone()
        label_seq_2[:, :-num_label_tokens] = -100

        loss, batch_loss, valid_token_num = compute_loss_context_input(model, input_ids_seq_1, input_ids_seq_2, attention_mask_seq_1, attention_mask_seq_2, label_seq_1, label_seq_2)

        all_loss.append(loss.item())

    return sum(all_loss) / len(all_loss)

@torch.no_grad()
def evaluate_generation(model, dataloader, accelerator:Optional[Accelerator]=None, tokenizer=None, return_new_tokens_only=True, return_decoded=True, **generation_config):
    if accelerator is not None and type(dataloader) == torch.utils.data.DataLoader:
        # if the dataloader has been prepared, we shall not prepare it twice, especially in case of deepspeed
        dataloader = accelerator.prepare(dataloader)

    all_indices = []
    all_outputs = []
    
    for i, x in enumerate(tqdm(dataloader, desc="Computing Generation")):
        # if i > 3:
        #     break
        
        # NOTE: important to reset memory for every batch
        if hasattr(model, "memory"):
            model.memory.reset()

        indices = x.pop("index")
        # length is used to group training data, no use here
        length = x.pop("length", None)

        outputs = model.generate(**x, **generation_config)
        if return_new_tokens_only:
            start_idx = x["input_ids"].shape[1]
            outputs = outputs[:, start_idx:]

        if accelerator is not None and accelerator.num_processes > 1:
            # must be contiguous
            outputs = accelerator.pad_across_processes(outputs.contiguous(), pad_index=tokenizer.pad_token_id, dim=1)
            outputs = accelerator.gather_for_metrics(outputs)
            indices = accelerator.gather_for_metrics(indices)

        outputs = outputs.tolist()
        indices = indices.tolist()
        if return_decoded:
            outputs = tokenizer.batch_decode(outputs, skip_special_tokens=True)
        all_indices.extend(indices)
        all_outputs.extend(outputs)

    return all_indices, all_outputs


@torch.no_grad()
def evaluate_generation_fastlora(model, dataloader, accelerator:Optional[Accelerator]=None, tokenizer=None, max_new_tokens=50, merge_strategy='concat', **generation_config):
    """
    Custom generation evaluation for FastLoRA with context.
    
    Expects data format with:
    - input_ids: [batch, num_segments, seq_len] where segments include context chunks + query+response
    - attention_mask: [batch, num_segments, seq_len]
    - labels: [batch, num_segments, seq_len] (ground truth assistant response, user part is -100)
    
    Process:
    1. Extract context segments (all but last)
    2. Extract query from last segment (only the user question part, where labels == -100)
    3. Encode context through model to get hidden states
    4. Generate response conditioned on context hidden states + query (user question only)
    5. Compare generated response with ground truth labels (assistant response)
    """
    if accelerator is not None and type(dataloader) == torch.utils.data.DataLoader:
        dataloader = accelerator.prepare(dataloader)

    all_indices = []
    all_outputs = []
    all_labels = []
    
    for i, x in enumerate(tqdm(dataloader, desc="Computing FastLoRA Generation")):
        indices = x.pop("index")
        length = x.pop("length", None)
        
        # Extract raw text fields if available
        ground_truth_texts = x.pop("ground_truth_text", None)
        query_texts = x.pop("query_text", None)
        context_texts = x.pop("context_text", None)
        
        input_ids = x["input_ids"]  # [B, num_segments, seq_len]
        attention_mask = x["attention_mask"]  # [B, num_segments, seq_len]
        labels = x.get("labels", None)  # [B, num_segments, seq_len]
        
        batch_size, num_segments, seq_len = input_ids.shape
        
        # Split into context (all segments except last) and query+response (last segment)
        if num_segments > 1:
            # Context: segments 0 to N-2 (KEEP WINDOWED FORMAT for FastLoRA!)
            context_input_ids = input_ids[:, :-1, :]  # [B, N-1, seq_len]
            context_attention_mask = attention_mask[:, :-1, :]  # [B, N-1, seq_len]
            
            # Last segment contains: user query (labels=-100) + assistant response (labels=token_ids)
            last_segment_ids = input_ids[:, -1, :]  # [B, seq_len]
            last_segment_mask = attention_mask[:, -1, :]
            last_segment_labels = labels[:, -1, :] if labels is not None else None
            
            # Extract ONLY the user query part (where labels == -100)
            # This is the prompt we'll use for generation
            query_input_ids_list = []
            query_attention_mask_list = []
            ground_truth_list = []
            
            for b in range(batch_size):
                if last_segment_labels is not None:
                    # Find where labels transition from -100 (user) to real tokens (assistant)
                    user_mask = last_segment_labels[b] == -100
                    assistant_mask = last_segment_labels[b] != -100
                    
                    # Find the last user token position
                    user_positions = torch.where(user_mask)[0]
                    if len(user_positions) > 0:
                        last_user_pos = user_positions[-1].item() + 1  # +1 to include the position
                        query_ids = last_segment_ids[b, :last_user_pos]
                        query_mask = last_segment_mask[b, :last_user_pos]
                    else:
                        # No user part, use everything (shouldn't happen)
                        query_ids = last_segment_ids[b]
                        query_mask = last_segment_mask[b]
                    
                    # Extract ground truth assistant response
                    # Get the actual token IDs from input_ids where labels are not -100
                    assistant_positions = torch.where(assistant_mask)[0]
                    if len(assistant_positions) > 0:
                        # Use input_ids at assistant positions, not labels
                        assistant_tokens = last_segment_ids[b][assistant_positions]
                        ground_truth = tokenizer.decode(assistant_tokens, skip_special_tokens=True)
                        
                        # Debug: print for first example
                        if b == 0 and i == 0:
                            print(f"\n[DEBUG] Label stats:")
                            print(f"  Total tokens: {len(last_segment_labels[b])}")
                            print(f"  User tokens (label=-100): {user_mask.sum().item()}")
                            print(f"  Assistant tokens (label!=100): {assistant_mask.sum().item()}")
                            print(f"  Assistant positions: {assistant_positions[:10].tolist()}...")
                            print(f"  Last segment full text: {tokenizer.decode(last_segment_ids[b], skip_special_tokens=True)[:200]}...")
                            print(f"  Ground truth extracted: {ground_truth[:100]}...")
                    else:
                        ground_truth = ""
                        if b == 0 and i == 0:
                            print(f"\n[DEBUG] No assistant tokens found! All labels are -100")
                else:
                    # No labels, use full segment
                    query_ids = last_segment_ids[b]
                    query_mask = last_segment_mask[b]
                    ground_truth = ""
                
                query_input_ids_list.append(query_ids)
                query_attention_mask_list.append(query_mask)
                ground_truth_list.append(ground_truth)
            
            # Pad queries to same length for batching
            from torch.nn.utils.rnn import pad_sequence
            query_input_ids = pad_sequence(query_input_ids_list, batch_first=True, padding_value=tokenizer.pad_token_id)
            query_attention_mask = pad_sequence(query_attention_mask_list, batch_first=True, padding_value=0)
            
            # Encode context to get hidden states (WITH WINDOWED FORMAT for FastLoRA!)
            context_outputs = model(
                input_ids=context_input_ids,  # [B, N-1, seq_len] - windowed!
                attention_mask=context_attention_mask,  # [B, N-1, seq_len]
                output_hidden_states=True,
            )
            # Extract the last layer's hidden states (required for fastlora_use_last mode)
            # context_outputs.hidden_states is a tuple of tensors (one per layer)
            # We need the last layer: tuple[-1] -> [B, N-1, seq_len, hidden_size]
            context_hidden_states_segmented = context_outputs.hidden_states[-1]
            
            context_attention_mask_seg = context_attention_mask  # Already [B, N-1, seq_len]
            
            # Generate conditioned on context hidden states + user query only
            outputs = model.generate(
                inputs=query_input_ids,
                attention_mask=query_attention_mask,
                fastlora_hidden_states_and_mask=(context_hidden_states_segmented, context_attention_mask_seg),
                max_new_tokens=250,
                pad_token_id=tokenizer.pad_token_id,  # FIXED: Use pad_token_id, not eos_token_id
                eos_token_id=tokenizer.eos_token_id,
                do_sample=False,
                **generation_config
            )
        else:
            assert False, "No context, just generate from query"
            # No context, just generate from query
            last_segment_ids = input_ids[:, 0, :]
            last_segment_mask = attention_mask[:, 0, :]
            last_segment_labels = labels[:, 0, :] if labels is not None else None
            
            # Extract user query and ground truth
            query_input_ids_list = []
            query_attention_mask_list = []
            ground_truth_list = []
            
            for b in range(batch_size):
                if last_segment_labels is not None:
                    user_mask = last_segment_labels[b] == -100
                    assistant_mask = last_segment_labels[b] != -100
                    
                    user_positions = torch.where(user_mask)[0]
                    if len(user_positions) > 0:
                        last_user_pos = user_positions[-1].item() + 1
                        query_ids = last_segment_ids[b, :last_user_pos]
                        query_mask = last_segment_mask[b, :last_user_pos]
                    else:
                        query_ids = last_segment_ids[b]
                        query_mask = last_segment_mask[b]
                    
                    # Use input_ids at assistant positions, not labels
                    assistant_positions = torch.where(assistant_mask)[0]
                    if len(assistant_positions) > 0:
                        assistant_tokens = last_segment_ids[b][assistant_positions]
                        ground_truth = tokenizer.decode(assistant_tokens, skip_special_tokens=True)
                    else:
                        ground_truth = ""
                else:
                    query_ids = last_segment_ids[b]
                    query_mask = last_segment_mask[b]
                    ground_truth = ""
                
                query_input_ids_list.append(query_ids)
                query_attention_mask_list.append(query_mask)
                ground_truth_list.append(ground_truth)
            
            from torch.nn.utils.rnn import pad_sequence
            query_input_ids = pad_sequence(query_input_ids_list, batch_first=True, padding_value=tokenizer.pad_token_id)
            query_attention_mask = pad_sequence(query_attention_mask_list, batch_first=True, padding_value=0)
            
            outputs = model.generate(
                inputs=query_input_ids,
                attention_mask=query_attention_mask,
                max_new_tokens=max_new_tokens,
                pad_token_id=tokenizer.pad_token_id,  # FIXED: Use pad_token_id, not eos_token_id
                eos_token_id=tokenizer.eos_token_id,
                do_sample=False,
                **generation_config
            )
        
        # Extract only new tokens (skip input query)
        start_idx = query_input_ids.shape[1]
        new_tokens = outputs[:, start_idx:]
        
        # Gather across processes if using distributed
        if accelerator is not None and accelerator.num_processes > 1:
            new_tokens = accelerator.pad_across_processes(new_tokens.contiguous(), pad_index=tokenizer.pad_token_id, dim=1)
            new_tokens = accelerator.gather_for_metrics(new_tokens)
            indices = accelerator.gather_for_metrics(indices)
        
        # Decode outputs
        decoded_outputs = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
        
        # Debug: Check if EOS tokens are present in raw output
        if i == 0:  # Only for first batch
            raw_outputs = tokenizer.batch_decode(new_tokens, skip_special_tokens=False)
            for b in range(min(len(raw_outputs), 1)):  # Just first example
                print(f"\n[DEBUG] Raw generated tokens: {raw_outputs[b][:100]}...")
                print(f"[DEBUG] Contains EOS token: {'<|im_end|>' in raw_outputs[b]}")
                print(f"[DEBUG] Token length: {new_tokens.shape[1]} tokens")
                print(f"[DEBUG] Last 3 token IDs: {new_tokens[b][-3:].tolist()}")
        
        all_outputs.extend(decoded_outputs)
        all_indices.extend(indices.tolist())
        
        # Use raw ground truth text if available, otherwise use extracted text
        if ground_truth_texts is not None:
            all_labels.extend(ground_truth_texts.tolist() if hasattr(ground_truth_texts, 'tolist') else ground_truth_texts)
        else:
            all_labels.extend(ground_truth_list)
        
        # Print first few examples for inspection
        if i < 3:  # Print first 3 batches
            print(f"\n{'='*80}")
            print(f"GENERATION EXAMPLE (Batch {i+1})")
            print(f"{'='*80}")
            for b in range(min(batch_size, 2)):  # Print first 2 in batch
                # Use raw text if available, otherwise decode
                if context_texts is not None:
                    context_text = context_texts[b] if hasattr(context_texts, '__getitem__') else str(context_texts)
                elif num_segments > 1:
                    context_text = tokenizer.decode(context_input_ids[b], skip_special_tokens=True)
                else:
                    context_text = "[No context]"
                
                if query_texts is not None:
                    query_text = query_texts[b] if hasattr(query_texts, '__getitem__') else str(query_texts)
                else:
                    query_text = tokenizer.decode(query_input_ids_list[b] if 'query_input_ids_list' in locals() else query_input_ids[b], skip_special_tokens=True)
                
                if ground_truth_texts is not None:
                    ground_truth = ground_truth_texts[b] if hasattr(ground_truth_texts, '__getitem__') else str(ground_truth_texts)
                else:
                    ground_truth = ground_truth_list[b] if b < len(ground_truth_list) else ""
                
                print(f"\n--- Example {b+1} ---")
                print(f"CONTEXT: {context_text}" if len(context_text) > 500 else f"CONTEXT: {context_text}")
                print(f"\nQUERY: {query_text}")
                print(f"\nGROUND TRUTH: {ground_truth}")
                print(f"\nGENERATED: {decoded_outputs[b]}")
                print(f"{'-'*80}")
    
    return all_indices, all_outputs, all_labels


@torch.no_grad()
def evaluate_nll(model, dataloader, accelerator:Optional[Accelerator]=None):
    if accelerator is not None and type(dataloader) == torch.utils.data.DataLoader:
        # if the dataloader has been prepared, we shall not prepare it twice, especially in case of deepspeed
        dataloader = accelerator.prepare(dataloader)

    # if accelerator.process_index == 0:
    #     for name, x in model.named_parameters():
    #         print(f"{name: ^80} {x.dtype}")

    all_loss = defaultdict(list)
    for i, x in enumerate(tqdm(dataloader, desc="Computing Perplexity")):
        # NOTE: important to reset memory for every batch
        if hasattr(model, "memory"):
            model.memory.reset()

        # the seq id
        index = x.pop("index")
        # length is used to group training data, no use here
        length = x.pop("length", None)

        output = model(**x)

        # NOTE: we need the loss for each element in the batch for accurate computation, because the number of valid tokens may differ among elements
        if hasattr(output, "batch_loss"):
            # output from our model has batch_loss by default
            batch_loss = output.batch_loss
            valid_token_num = output.valid_token_num
        else:
            # output from other models does not
            loss, batch_loss, valid_token_num = compute_loss(output.logits, x["labels"], shift=True)

        if accelerator is not None and accelerator.num_processes > 1:
            # num_device * batch_size
            index = accelerator.gather_for_metrics(index)
            batch_loss = accelerator.gather_for_metrics(batch_loss)
            valid_token_num = accelerator.gather_for_metrics(valid_token_num)

        for _id, _loss in zip(index.tolist(), batch_loss.tolist()):
            # loss times num is the total loss of all valid tokens
            all_loss[_id].append(_loss)

    return all_loss



@dataclass
class ModelOutput(BaseModelOutputWithPast):
    loss: Optional[torch.FloatTensor] = None
    batch_loss: Optional[torch.FloatTensor] = None
    valid_token_num: Optional[torch.LongTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


# ============================================================================
# Transformer Blocks for Hidden State Refinement
# ============================================================================

class TransformerRefinerBlock(torch.nn.Module):
    """
    A standard transformer block for refining hidden states before outer product computation.
    
    Applies self-attention and FFN with residual connections and layer normalization.
    Input/Output shape: [B, S, L, H] or [B, L, H] depending on context.
    """
    def __init__(self, hidden_size, num_heads=4, ffn_hidden_size=None, dropout_rate=0.1, dtype=torch.float32, activation_type="gelu"):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        
        # Validate multi-head attention setup
        assert hidden_size % num_heads == 0, f"hidden_size ({hidden_size}) must be divisible by num_heads ({num_heads})"
        
        self.norm1 = torch.nn.LayerNorm(hidden_size, dtype=dtype)
        self.self_attn = torch.nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout_rate, batch_first=True, dtype=dtype
        )
        
        self.norm2 = torch.nn.LayerNorm(hidden_size, dtype=dtype)
        ffn_hidden_size = ffn_hidden_size or 4 * hidden_size
        
        # Activation function
        if activation_type == "gelu":
            activation = torch.nn.GELU()
        elif activation_type == "silu":
            activation = torch.nn.SiLU()
        elif activation_type == "relu":
            activation = torch.nn.ReLU()
        else:
            activation = torch.nn.GELU()
        
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, ffn_hidden_size, dtype=dtype),
            activation,
            torch.nn.Dropout(dropout_rate),
            torch.nn.Linear(ffn_hidden_size, hidden_size, dtype=dtype),
            torch.nn.Dropout(dropout_rate),
        )
        
    def forward(self, x):
        """
        Args:
            x: [B, S, L, H] or [B, L, H] tensor
            
        Returns:
            refined_x: Same shape as input, refined by self-attention and FFN
        """
        original_shape = x.shape
        input_dtype = x.dtype
        
        # Handle both [B, S, L, H] and [B, L, H] cases by reshaping to 3D for MHA
        if len(x.shape) == 4:
            B, S, L, H = x.shape
            x = x.view(B * S, L, H)  # Reshape to [B*S, L, H]
        
        # Self-attention with residual
        x_norm = self.norm1(x)
        attn_output, _ = self.self_attn(x_norm, x_norm, x_norm)
        x = x + attn_output
        
        # FFN with residual
        x_norm = self.norm2(x)
        ffn_output = self.ffn(x_norm)
        x = x + ffn_output
        
        # Reshape back to original shape if needed
        if len(original_shape) == 4:
            B, S, L, H = original_shape
            x = x.view(B, S, L, H)
        
        return x.to(input_dtype)


class TransformerHiddenStateRefiner(torch.nn.Module):
    """
    Stack of transformer blocks to refine hidden states before outer product computation.
    
    This module is prepended to the FastLoRA outer product computation to refine
    context hidden states through self-attention and FFN layers, enabling better
    context understanding before weight prediction.
    """
    def __init__(self, hidden_size, num_layers, num_heads=4, ffn_hidden_size=None, dropout_rate=0.1, dtype=torch.float32, activation_type="gelu"):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        ffn_hidden_size = ffn_hidden_size or 4 * hidden_size
        
        self.transformer_blocks = torch.nn.ModuleList([
            TransformerRefinerBlock(
                hidden_size=hidden_size,
                num_heads=num_heads,
                ffn_hidden_size=ffn_hidden_size,
                dropout_rate=dropout_rate,
                dtype=dtype,
                activation_type=activation_type,
            )
            for _ in range(num_layers)
        ])
        
        self.final_norm = torch.nn.LayerNorm(hidden_size, dtype=dtype)
    
    def forward(self, hidden_states):
        """
        Args:
            hidden_states: [B, S, L, H] or [B, L, H] - Hidden states to refine
            
        Returns:
            refined_states: Same shape as input - Refined hidden states
        """
        # Apply transformer blocks sequentially
        for block in self.transformer_blocks:
            hidden_states = block(hidden_states) + hidden_states
        
        # Final normalization
        refined_states = self.final_norm(hidden_states)
        
        return refined_states


# ============================================================================
# Deep Context Refiner Architecture for FastLoRA
# ============================================================================

class FastLoraContextRefinerBlock(torch.nn.Module):
    """
    A single block in the Deep Context Refiner architecture.
    
    Takes hidden states H_in [B, S, L, H] and produces a delta_H update.
    
    Architecture:
    1. Project H_in through A2 and A3 to get K and V
    2. Compute ss = K^T @ V (the adaptation matrix) with bilinear alternation
    3. Project flattened ss back to hidden dimension via O (output projection)
    4. Process through FFN
    5. Return delta_H for residual update
    """
    def __init__(self, hidden_size, r1, r2, ffn_hidden_size, dropout_rate=0.1, dtype=torch.float32, fastlora_bilinear=False, activation_type="gelu"):
        super().__init__()
        self.hidden_size = hidden_size
        self.r1 = r1
        self.r2 = r2
        self.fastlora_bilinear = fastlora_bilinear
        self.activation_type = activation_type
        
        # Step counter for bilinear alternation (shared across all blocks)
        self.bilinear_step_counter = 0
        
        # A2 and A3 projections (like in original FastLoRA)
        self.A2 = torch.nn.Linear(hidden_size, r1, bias=False, dtype=dtype)
        self.A3 = torch.nn.Linear(hidden_size, r2, bias=False, dtype=dtype)
        
        # Output projection: maps ss [R1*R2] back to hidden dimension [H]
        self.output_proj = torch.nn.Linear(r1 * r2, hidden_size, bias=False, dtype=dtype)
        # Initialize to zero for stability (starts with identity-like behavior)
        torch.nn.init.zeros_(self.output_proj.weight)
        
        # FFN for non-linear transformation - now supports configurable activation
        if activation_type.lower() == "gelu":
            activation_fn = torch.nn.GELU()
        elif activation_type.lower() == "silu":
            activation_fn = torch.nn.SiLU()
        elif activation_type.lower() == "relu":
            activation_fn = torch.nn.ReLU()
        elif activation_type.lower() == "tanh":
            activation_fn = torch.nn.Tanh()
        else:
            raise ValueError(f"Unknown activation type: {activation_type}")
            
        self.ffn = torch.nn.Sequential(
            torch.nn.Linear(hidden_size, ffn_hidden_size, dtype=dtype),
            activation_fn,  # Now configurable!
            torch.nn.Linear(ffn_hidden_size, hidden_size, dtype=dtype),
            torch.nn.Dropout(dropout_rate),
        )
        
        # Layer norms
        self.norm1 = torch.nn.LayerNorm(hidden_size, dtype=dtype)
        self.norm2 = torch.nn.LayerNorm(hidden_size, dtype=dtype)
        self.dropout = torch.nn.Dropout(dropout_rate)
    
    def _compute_bilinear_ss(self, K, V):
        """Compute ss matrix with optional alternating gradient detachment for stability."""
        if self.fastlora_bilinear and self.training:
            # Increment step counter
            self.bilinear_step_counter += 1
            
            # Alternate which tensor gets detached based on step counter
            # Even steps: detach K, odd steps: detach V
            if self.bilinear_step_counter % 2 == 0:
                # Detach K - gradients flow through V only
                ss = K.detach().transpose(-2, -1) @ V
            else:
                # Detach V - gradients flow through K only  
                ss = K.transpose(-2, -1) @ V.detach()
        else:
            # Standard computation - gradients flow through both
            ss = K.transpose(-2, -1) @ V
            
        return ss
    
    def forward(self, H_in):
        """
        Args:
            H_in: [B, S, L, H] - Input hidden states
            
        Returns:
            delta_H: [B, S, 1, H] - Update to be added to H_in
        """
        B, S, L, H = H_in.shape
        
        # 1. Apply layer norm and project through A2, A3
        H_norm = self.norm1(H_in)  # [B, S, L, H]
        K = self.A2(H_norm)  # [B, S, L, R1]
        V = self.A3(H_norm)  # [B, S, L, R2]
        
        # 2. Compute ss = K^T @ V for each segment with bilinear alternation
        # K^T has shape [B, S, R1, L], V has shape [B, S, L, R2]
        ss = self._compute_bilinear_ss(K, V)  # [B, S, R1, R2]
        
        # 3. Flatten ss and project back to hidden dimension
        ss_flat = ss.view(B, S, -1)  # [B, S, R1*R2]
        h_pooled = self.output_proj(ss_flat)  # [B, S, H]
        
        # 4. Apply FFN with residual connection
        h_pooled_norm = self.norm2(h_pooled)  # [B, S, H]
        h_transformed = self.ffn(h_pooled_norm)  # [B, S, H]
        delta_h_pooled = h_pooled + self.dropout(h_transformed)  # [B, S, H]
        
        # 5. Broadcast to match input shape (add dimension for L)
        delta_H = delta_h_pooled.unsqueeze(2)  # [B, S, 1, H]
        
        return delta_H


class FastLoraContextRefiner(torch.nn.Module):
    """
    Stacked Deep Context Refiner for FastLoRA.
    
    This module acts as a preprocessing step that refines context hidden states
    before they are processed by the existing A2/A3 pipeline in FastLoraLinear.
    
    Architecture:
    1. Initial embedding: Project H [hidden_size] → [inter_size] for compact representation
    2. Stack of refiner blocks: Iteratively refine the embedded representation
    3. Output: Refined hidden states in inter_size dimension
    
    The output is then normalized and fed into the existing A2/A3 modules in model.py.
    """
    def __init__(self, hidden_size, inter_size, num_layers, r1, r2, ffn_hidden_size, dropout_rate=0.1, dtype=torch.float32, fastlora_bilinear=False, activation_type="gelu"):
        super().__init__()
        self.hidden_size = hidden_size
        self.inter_size = inter_size
        self.num_layers = num_layers
        
        # Initial embedding layer: project from base model hidden_size to inter_size
        # This allows the refiner to work in a more compact latent space
        self.initial_embedding = torch.nn.Linear(hidden_size, inter_size, bias=False, dtype=dtype)
        torch.nn.init.normal_(self.initial_embedding.weight, std=0.02)
        
        # Stack of refiner blocks (now operating in inter_size dimension)
        self.blocks = torch.nn.ModuleList([
            FastLoraContextRefinerBlock(
                hidden_size=inter_size,  # Now working in inter_size, not hidden_size!
                r1=r1,
                r2=r2,
                ffn_hidden_size=ffn_hidden_size,
                dropout_rate=dropout_rate,
                dtype=dtype,
                fastlora_bilinear=fastlora_bilinear,
                activation_type=activation_type  # Pass activation to each block
            )
            for _ in range(num_layers)
        ])
        
        # Final layer norm to normalize the refined hidden states
        self.final_norm = torch.nn.LayerNorm(inter_size, dtype=dtype)
    
    def forward(self, H_raw):
        """
        Args:
            H_raw: [B, S, L, hidden_size] - Raw hidden states from base model
            
        Returns:
            H_refined_norm: [B, S, L, inter_size] - Refined and normalized hidden states
                                                     ready for A2/A3 processing
        """
        # 1. Project to inter_size dimension
        H_embedded = self.initial_embedding(H_raw)  # [B, S, L, inter_size]
        
        # 2. Start with the embedded representation
        H_state = H_embedded
        
        # 3. Iteratively refine through each block
        # Each block sees the ORIGINAL embedded representation (full fidelity)
        for block in self.blocks:
            delta_H = block(H_embedded)  # Each block sees the ORIGINAL H_embedded
            H_state = H_state + delta_H  # Residual update (broadcasts across L)
        
        # 4. Final normalization
        H_refined_norm = self.final_norm(H_state)  # [B, S, L, inter_size]
        
        return H_refined_norm


