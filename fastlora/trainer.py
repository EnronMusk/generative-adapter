import os
import math
import torch
import datasets
import random
from tqdm import tqdm
from dataclasses import asdict
from typing import Any, Dict, List, Optional, Union
from torch.utils.data import Sampler, Dataset
from transformers.trainer import Trainer, is_datasets_available
from transformers.tokenization_utils import BatchEncoding
from transformers.utils import logging
import torch.nn.functional as F

from fastlora.modeling_utils import evaluate_generation, evaluate_generation_fastlora, evaluate_perplexity, evaluate_squad

logger = logging.get_logger(__name__)


class FastLoraTrainer(Trainer):
    def __init__(self, *args, model_args, file_logger, **kwargs):
        super().__init__(*args, **kwargs)
        self.model_args = model_args
        self.file_logger = file_logger

    def _compute_perplexity_from_dataloader(self, model, dataloader, reconstruct_tokens=None):
        """Compute perplexity using the existing evaluation dataloader instead of hardcoded files."""
        model.eval()
        all_losses = []
        
        for batch in tqdm(dataloader, desc="Computing Perplexity"):
            # Remove index and length if present
            batch.pop("index", None)
            batch.pop("length", None)
            
            # Move to device
            batch = {k: v.to(model.device) if hasattr(v, 'to') else v for k, v in batch.items()}
            
            with torch.no_grad():
                outputs = model(**batch)
                
                if hasattr(outputs, "loss") and outputs.loss is not None:
                    all_losses.append(outputs.loss.item())
        
        if all_losses:
            avg_loss = sum(all_losses) / len(all_losses)
            return math.exp(avg_loss)  # Convert to perplexity
        else:
            return float('inf')

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        # if "retrieval_span" in inputs:
        #     self.model.memory._retrieval_span = inputs['retrieval_span'][0]
        #     inputs.pop("retrieval_span")

        # Support both transformers 4.42.4 (no num_items_in_batch) and 4.51.0+ (has num_items_in_batch)
        import inspect
        sig = inspect.signature(super().compute_loss)
        if 'num_items_in_batch' in sig.parameters:
            outputs = super().compute_loss(model, inputs, return_outputs, num_items_in_batch=num_items_in_batch)
        else:
            outputs = super().compute_loss(model, inputs, return_outputs)

        # if hasattr(self.model, "memory") and hasattr(self.model.memory, "_retrieval_span"):
        #     del self.model.memory._retrieval_span
        #     del self.model.memory._retrieval_condensing_ratios
        return outputs

    def _prepare_inputs(self, inputs: Dict[str, Union[torch.Tensor, Any]]) -> Dict[str, Union[torch.Tensor, Any]]:
        """
        Prepare `inputs` before feeding them to the model, converting them to tensors if they are not already and
        handling potential state.
        """
        inputs.pop("length", None)
        inputs.pop("index", None)
        # move to GPU
        inputs = self._prepare_input(inputs)
        
        # window_size = self.model.peft_config['default'].fastlora_window
        # number_windows = (inputs.shape[-1] + window_size - 1) // window_size
        # seq_len = (inputs.shape[-1] + number_windows - 1) // number_windows
        # input_ids, attention_mask, labels = inputs["input_ids"], inputs["attention_mask"], inputs["labels"]
        # input_ids = F.pad(input_ids, (0, number_windows * seq_len - input_ids.shape[-1]), value=self.tokenizer.pad_token_id).reshape(-1, number_windows, seq_len)
        # attention_mask = F.pad(attention_mask, (0, number_windows * seq_len - attention_mask.shape[-1]), value=0).reshape(-1, number_windows, seq_len)
        # labels = F.pad(labels, (0, number_windows * seq_len - labels.shape[-1]), value=-100).reshape(-1, number_windows, seq_len)
        # inputs = {"input_ids": input_ids, "attention_mask": attention_mask, "labels": labels}

        return inputs
    
    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        # Build the sampler.
        if self.args.group_by_stride is not None:
            raise NotImplementedError("StrideGroupedSampler is not implemented yet!")
            # if is_datasets_available() and isinstance(self.train_dataset, datasets.Dataset):
            #     lengths = self.train_dataset[self.args.length_column_name]
            # else:
            #     lengths = None
            
            # model_input_name = self.tokenizer.model_input_names[0] if self.tokenizer is not None else None

            # return StrideGroupedSampler(
            #     # NOTE: multiply world size to get the total number of training instances across devices
            #     batch_size=self.args.train_batch_size * self.args.world_size,
            #     window=self.model.memory.ultragist_window,
            #     stride=self.model.memory.ultragist_stride,
            #     group=self.args.group_by_stride,
            #     sort=self.args.sort_by_stride,
            #     dataset=self.train_dataset,
            #     lengths=lengths,
            #     model_input_name=model_input_name,
            # )

        else:
            return super()._get_train_sampler()
    
    def _save(self, output_dir: Optional[str] = None, state_dict=None):
        outputs = super()._save(output_dir, state_dict)
        # NOTE: also save model_args
        self.model_args.save(os.path.join(output_dir, "model_args.json"))
        return outputs

    @torch.no_grad()
    def evaluate(self, eval_dataset: Dataset | None = None, ignore_keys: List[str] | None = None, metric_key_prefix: str = "eval") -> Dict[str, float]:        
        # memory metrics - must set up as early as possible
        self._memory_tracker.start()

        if eval_dataset is None and self.eval_dataset is None:
            return

        if self.args.eval_method == "generation":
            labels = self.eval_dataset["labels"]
            self.eval_dataset = self.eval_dataset.remove_columns(["labels"])

        dataloader = self.get_eval_dataloader()

        # self.model.memory.reset()
        # train_ultragist_ratio = self.model.memory.ultragist_ratio
        # train_ultragist_ratio_mix = self.model.memory.ultragist_ratio_mix
        # self.model.memory.set(
        #     ultragist_ratio=self.args.eval_ultragist_ratio,
        #     ultragist_ratio_mix=self.args.eval_ultragist_ratio_mix,
        # )

        model = self.model.eval()

        if self.args.eval_method == "perplexity":
            perplexity = evaluate_perplexity(model, self.tokenizer)
            perplexity_reconstruct = evaluate_perplexity(model, self.tokenizer, reconstruct_tokens=1024)
            loss_squad = evaluate_squad(model, self.tokenizer)
            metrics = {"perplexity": perplexity, "perplexity_reconstruct": perplexity_reconstruct, "loss_squad": loss_squad}
        elif self.args.eval_method == "generation":
            indices, outputs = evaluate_generation(
                model, 
                dataloader, 
                accelerator=self.accelerator, 
                tokenizer=self.tokenizer,
            )
            metrics = self.compute_metrics(outputs, labels, indices=indices)
        elif self.args.eval_method == "generation_fastlora":
            # Custom FastLoRA generation with context encoding
            indices, outputs, labels = evaluate_generation_fastlora(
                model, 
                dataloader, 
                accelerator=self.accelerator, 
                tokenizer=self.tokenizer,
                max_new_tokens=250,  # Reduced from 50 to prevent repetition
            )
            
            # Print first few generations for debug with context/query/target/generated
            if self.args.process_index == 0:
                print("\n=== GENERATION DEBUG (First 3 samples) ===")
                
                # Get the dataloader to extract context/query information
                eval_dataloader = self.get_eval_dataloader()
                
                # Get first few batches to match with outputs
                debug_batches = []
                for i, batch in enumerate(eval_dataloader):
                    if i >= 3:  # Only need first 3 samples worth
                        break
                    debug_batches.append(batch)
                
                sample_idx = 0
                for batch in debug_batches:
                    batch_size = batch['input_ids'].shape[0]
                    for b in range(batch_size):
                        if sample_idx >= min(3, len(outputs)):
                            break
                            
                        print(f"\n--- Sample {sample_idx + 1} (Index: {indices[sample_idx]}) ---")
                        
                        # Extract input_ids for this sample: [num_segments, seq_len] 
                        sample_input_ids = batch['input_ids'][b]  # [N, seq_len]
                        sample_labels = batch['labels'][b]        # [N, seq_len]
                        
                        # Context = all segments except last
                        context_segments = sample_input_ids[:-1]  # [N-1, seq_len]
                        context_text = ""
                        for seg in context_segments:
                            # Remove padding and decode
                            seg_clean = seg[seg != self.tokenizer.pad_token_id]
                            context_text += self.tokenizer.decode(seg_clean, skip_special_tokens=True) + " "
                        
                        # Query + Target = last segment
                        last_segment_ids = sample_input_ids[-1]   # [seq_len]
                        last_segment_labels = sample_labels[-1]   # [seq_len]
                        
                        # Query = where labels == -100 (user part)
                        query_mask = last_segment_labels == -100
                        query_ids = last_segment_ids[query_mask]
                        if len(query_ids) > 0:
                            query_text = self.tokenizer.decode(query_ids, skip_special_tokens=True)
                        else:
                            query_text = "[No query found]"
                        
                        # Target = where labels != -100 (assistant response)
                        target_mask = last_segment_labels != -100
                        target_ids = last_segment_labels[target_mask]  # Use labels, not input_ids
                        if len(target_ids) > 0:
                            target_text = self.tokenizer.decode(target_ids, skip_special_tokens=True)
                        else:
                            target_text = "[No target found]"
                        
                        # Generated response
                        generated_text = outputs[sample_idx]
                        
                        # Print all components
                        print(f"CONTEXT (KEY): {context_text.strip()[:300]}...")
                        print(f"QUERY: {query_text.strip()}")
                        print(f"TARGET VALUE: {target_text.strip()}")
                        print(f"GENERATED VALUE: {generated_text.strip()}")
                        
                        sample_idx += 1
                        if sample_idx >= min(3, len(outputs)):
                            break
                    
                    if sample_idx >= min(3, len(outputs)):
                        break
                
                print("=" * 50)
            
            # Skip compute_metrics since you don't want ROUGE
            metrics = {}
            
            # ONLY compute perplexity metrics using the existing eval dataloader (no hardcoded files)
            eval_dataloader = self.get_eval_dataloader()
            perplexity = self._compute_perplexity_from_dataloader(model, eval_dataloader)
            perplexity_reconstruct = self._compute_perplexity_from_dataloader(model, eval_dataloader, reconstruct_tokens=1024)
            
            # Add only perplexity metrics to the results
            metrics.update({
                "perplexity": perplexity,
                "perplexity_reconstruct": perplexity_reconstruct
            })
            
            # Print metrics for debug
            if self.args.process_index == 0:
                print(f"\n=== EVAL METRICS DEBUG ===")
                print(f"perplexity: {perplexity:.4f}")
                print(f"perplexity_reconstruct: {perplexity_reconstruct:.4f}")
                print("=" * 30)
        else:
            raise NotImplementedError(f"Eval method {self.args.eval_method} not implemented!")

        # self.model.memory.reset()
        # self.model.memory.set(
        #     ultragist_ratio=train_ultragist_ratio,
        #     ultragist_ratio_mix=train_ultragist_ratio_mix,
        # )

        # Prefix all keys with metric_key_prefix + '_'
        for key in list(metrics.keys()):
            if not key.startswith(f"{metric_key_prefix}_") and key != "epoch":
                metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

        self.log(metrics)
        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
        self._memory_tracker.stop_and_update_metrics(metrics)

        # log to file
        if self.args.process_index == 0:
            self.file_logger.log(
                metrics=metrics,
                Model_Args=asdict(self.model_args),
                Training_Args=asdict(self.args),
                Global_Steps=self.state.global_step
            )

        return metrics

    def on_after_backward(self) -> None:
        valid_gradients = True
        for name, param in self.model.named_parameters():
            if param.grad is not None:
                valid_gradients = not (torch.isnan(param.grad).any() or torch.isinf(param.grad).any())
                if not valid_gradients:
                    break

        if not valid_gradients:
            logger.warning(f'detected inf or nan values in gradients. not updating model parameters')
            self.zero_grad()
    
    def log(self, logs: Dict[str, float], start_time=None) -> None:
        """Override log to add weight norm statistics."""
        if self.state.global_step > 0:
            # Compute weight norms for FastLoRA adapters (hypernetwork parameters)
            try:
                weight_norms = []
                for name, module in self.model.named_modules():
                    # Look for FastLoRA adapter parameters
                    if hasattr(module, 'fastlora_A1'):
                        # Compute norms of the adapter weight matrices
                        if hasattr(module.fastlora_A1, 'weight'):
                            weight_norms.append(module.fastlora_A1.weight.norm().item())
                        if hasattr(module, 'fastlora_A2') and hasattr(module.fastlora_A2, 'weight'):
                            weight_norms.append(module.fastlora_A2.weight.norm().item())
                        if hasattr(module, 'fastlora_A3') and hasattr(module.fastlora_A3, 'weight'):
                            weight_norms.append(module.fastlora_A3.weight.norm().item())
                        if hasattr(module, 'fastlora_B') and hasattr(module.fastlora_B, 'weight'):
                            weight_norms.append(module.fastlora_B.weight.norm().item())
                
                if weight_norms:
                    logs['weight_norm_mean'] = sum(weight_norms) / len(weight_norms)
            except Exception as e:
                logger.warning(f"Failed to compute weight norms: {e}")

            # Get predicted weight norms from cached values (computed during forward pass)
            try:
                self._add_predicted_weight_norms_from_cache(logs)
            except Exception as e:
                logger.warning(f"Failed to get predicted weight norms from cache: {e}")

        # Handle different transformers versions
        import inspect
        sig = inspect.signature(super().log)
        if 'start_time' in sig.parameters:
            super().log(logs, start_time)
        else:
            super().log(logs)

    def _add_predicted_weight_norms_from_cache(self, logs: Dict[str, float]) -> None:
        """Add predicted weight norms from cached values computed during forward pass."""
        # Look for cached predicted weight norms in FastLoRA modules
        pred_a_norms = []
        pred_b_norms = []
        
        for name, module in self.model.named_modules():
            if hasattr(module, 'fastlora_A1') and hasattr(module, '_last_predicted_norms'):
                # Get cached norms from the last forward pass
                cached_norms = module._last_predicted_norms
                if 'pred_a_norm' in cached_norms:
                    pred_a_norms.append(cached_norms['pred_a_norm'])
                if 'pred_b_norm' in cached_norms:
                    pred_b_norms.append(cached_norms['pred_b_norm'])
        
        # Log A matrix norms
        if pred_a_norms:
            logs['pred_a_norm_mean'] = sum(pred_a_norms) / len(pred_a_norms)
            logs['pred_a_norm_max'] = max(pred_a_norms)
            logs['pred_a_norm_min'] = min(pred_a_norms)
        
        # Log B matrix norms
        if pred_b_norms:
            logs['pred_b_norm_mean'] = sum(pred_b_norms) / len(pred_b_norms)
            logs['pred_b_norm_max'] = max(pred_b_norms)
            logs['pred_b_norm_min'] = min(pred_b_norms)

class StrideGroupedSampler(Sampler):
    """Group """

    def __init__(
        self,
        batch_size: int,
        window: int,
        stride: int,
        group: str,
        sort: Optional[str] = None,
        dataset: Optional[Dataset] = None,
        lengths: Optional[List[int]] = None,
        model_input_name: Optional[str] = None
    ):
        if dataset is None and lengths is None:
            raise ValueError("One of dataset and lengths must be provided.")
        
        if group is None:
            raise ValueError("Group cannot be None!")

        if lengths is None:
            model_input_name = model_input_name if model_input_name is not None else "input_ids"
            if (
                not (isinstance(dataset[0], dict) or isinstance(dataset[0], BatchEncoding))
                or model_input_name not in dataset[0]
            ):
                raise ValueError(
                    "Can only automatically infer lengths for datasets whose items are dictionaries with an "
                    f"'{model_input_name}' key."
                )
            lengths = [len(feature[model_input_name]) for feature in dataset]
        elif isinstance(lengths, torch.Tensor):
            logger.info(
                "If lengths is a torch.Tensor, LengthGroupedSampler will be slow. Converting lengths to List[int]..."
            )
            lengths = lengths.tolist()

        indices = list(range(len(lengths)))

        # get number of strides for each data
        num_strides = []
        for length in lengths:
            num_stride = math.ceil((length - window) / stride) + 1
            num_strides.append(num_stride)

        indice_stride_pairs = list(zip(indices, num_strides))
        # NOTE: shuffle the indices in advance, otherwise the randomness may be lost when all num_strides are equal
        random.shuffle(indice_stride_pairs)

        # sort data according to the number of strides
        indice_stride_pairs = sorted(indice_stride_pairs, key=lambda x: x[1])

        # group data instances with the same number of strides into the same batch
        batches = []
        batch = []
        prev_num_stride = None
        for index, num_stride in indice_stride_pairs:
            if num_stride != prev_num_stride:
                # in strict mode, all instances in the batch are forced to have the same number of strides
                if group == "strict":
                    batch.clear()
                elif group == "relaxed":
                    pass
                else:
                    raise ValueError(f"Group method {group} must be in None, strict, relaxed!")

            batch.append(index)
            prev_num_stride = num_stride

            if len(batch) == batch_size:
                batches.append((batch.copy(), num_stride))
                batch.clear()

        if len(batch) and group == "relaxed":
            batches.append((batch.copy(), num_stride))

        if sort is None:
            random.shuffle(batches)
        elif sort == "ascend":
            batches = sorted(batches, key=lambda x: x[1])
        elif sort == "descend":
            batches = sorted(batches, key=lambda x: x[1], reverse=True)
        else:
            raise ValueError(f"Sort method {sort} must be in None, ascend, descend!")

        batches = [x[0] for x in batches]
        self.indices = sum(batches, [])

    def __len__(self):
        return len(self.indices)

    def __iter__(self):
        return iter(self.indices)
