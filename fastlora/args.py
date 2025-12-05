import os
import json
from dataclasses import dataclass, field, asdict
from transformers.training_args import TrainingArguments
from typing import Optional, List, Tuple, Union, Dict


@dataclass
class ModelArgs:
    model_cache_dir: str = field(
        default=None,
        metadata={'help': 'Default path to save language models.'}
    )
    dataset_cache_dir: str = field(
        default=None,
        metadata={'help': 'Default path to save huggingface datasets.'}
    )
    data_root: str = field(
        default="/data/long-llm", 
        metadata={'help': 'The base directory storing all data used for training and evaluation. If specified, make sure all train_data, eval_data, and corpus are path relative to data_root!'},
    )
    train_data: Optional[List[str]] = field(
        default=None,
        metadata={'help': 'Training json file or glob to match a list of files.'},
    )
    eval_data: Optional[str] = field(
        default=None,
        metadata={'help': 'Evaluation json file.'},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={'help': 'Maximum number of training samples to use (for debugging).'},
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={'help': 'Maximum number of evaluation samples to use (for debugging).'},
    )
    
    model_name_or_path: str = field(
        default='meta-llama/Llama-2-7b-chat-hf',
        metadata={'help': 'Path to pretrained model or model identifier from huggingface.co/models'}
    )
    padding_side: str = field(
        default="left",
        metadata={'help': 'Tokenizer padding side. Always use left padding.'}
    )
    no_use_fast: bool = field(
        default=False,
        metadata={'help': 'Do not use fast tokenizer?'}
    )
    access_token: Optional[str] = field(
        default=None,
        metadata={'help': 'Huggingface access token.'}
    )
    attn_impl: Optional[str] = field(
        default="flash_attention_2",
        metadata={'help': 'The implementation of attention.'}
    )

    max_length: int = field(
        default=4096,
        metadata={'help': 'How many tokens at maximum for each input.'},
    )
    chat_template: str = field(
        default="llama-2",
        metadata={'help': 'Instruction template name in fastchat.'}
    )

    max_position_embeddings: Optional[int] = field(
        default=None,
        metadata={'help': 'Maximum position.'},
    )
    mistral_sliding_window: Optional[int] = field(
        default=None,
        metadata={'help': 'Sliding window size in Mistral models.'},
    )
    rope_theta: Optional[float] = field(
        default=None,
        metadata={'help': 'RoPE base (theta).'},
    )
    rope_method: Optional[str] = field(
        default=None,
        metadata={'help': 'How to scale RoPE?'},
    )
    rope_factor: float = field(
        default=1.,
        metadata={'help': 'RoPE scaling factor.'},
    )
    
    lora: Optional[str] = field(
        default=None,
        metadata={'help': 'LoRA ID.'},
    )
    lora_unload: bool = field(
        default=True,
        metadata={'help': 'Merge and unload LoRA?'},
    )
    load_in_4_bit: bool = field(
        default=False,
        metadata={'help': 'Load model in 4 bits?'},
    )

    dtype: str = field(
        default="bf16",
        metadata={'help': 'Data type for embeddings.'}
    )
    device_map: Optional[str] = field(
        default=None,
        metadata={'help': 'Device map for loading the model. Set to auto to load across devices.'}
    )
    batch_size: int = field(
        default=1,
        metadata={'help': 'Evaluation batch size.'},
    )
    cpu: bool = field(
        default=False,
        metadata={'help': 'Use cpu?'}
    )

    enable_tp: bool = field(
        default=False,
        metadata={'help': 'Use tensor parallel to wrap the model?'}
    )

    # enable_ultragist: bool = field(
    #     default=False,
    #     metadata={'help': 'Enable activation ultragist?'}
    # )
    # ultragist_window: Optional[int] = field(
    #     default=None,
    #     metadata={'help': 'The initial sliding window size.'}
    # )
    # ultragist_stride: Optional[int] = field(
    #     default=None,
    #     metadata={'help': 'The stride of the sliding window.'}
    # )
    # ultragist_attn: Optional[str] = field(
    #     default=None,
    #     metadata={'help': 'How to assign attention masks of ultragist tokens? {segmentation, step-expansion, full-converage}'}
    # )
    # ultragist_ratio: Optional[List[int]] = field(
    #     default=None,
    #     metadata={'help': 'Condensing ratios for ultragists.'}
    # )
    # ultragist_ratio_mix: Optional[str] = field(
    #     default=None,
    #     metadata={'help': 'How to determine the ultragist_ratio for each input. {step-random, instance-random, adapt-x}'}
    # )
    # ultragist_param: Optional[List[str]] = field(
    #     default=None,
    #     metadata={'help': 'The introduced parameters for ultragist.'}
    # )
    # ultragist_embed_init: str = field(
    #     default="eos",
    #     metadata={'help': 'Initialize ultragist embedding from eos/bos embedding.'}
    # )
    # ultragist_sink_size: Optional[int] = field(
    #     default=None,
    #     metadata={'help': 'The number of activations that are always kept in the head of the sequence according to StreamingLLM.'}
    # )
    # ultragist_attend_prev: Optional[bool] = field(
    #     default=None,
    #     metadata={'help': 'Can ultragist tokens attend to previous ultragist tokens?'}
    # )

    enable_lora: bool = field(
        default=False,
        metadata={'help': 'Enable LoRA?'}
    )
    lora_r: Optional[int] = field(
        default=None,
        metadata={'help': 'LoRA rank.'}
    )
    lora_alpha: Optional[float] = field(
        default=None,
        metadata={'help': 'LoRA scaling factor.'}
    )
    lora_dropout: Optional[float] = field(
        default=None,
        metadata={'help': 'LoRA dropout p.'}
    )
    lora_param: Optional[List[str]] = field(
        default=None,
        metadata={'help': 'The introduced parameters for ultragist.'}
    )

    enable_fastlora: bool = field(
        default=False,
        metadata={'help': 'Enable FastLoRA?'}
    )
    fastlora_r: Optional[int] = field(
        default=None,
        metadata={'help': 'LoRA rank.'}
    )
    fastlora_inter_size: Optional[int] = field(
        default=None,
        metadata={'help': 'The dimension of the inter size.'}
    )
    fastlora_window: Optional[int] = field(
        default=None,
        metadata={'help': 'The sliding window size.'}
    )
    fastlora_max_rank: Optional[int] = field(
        default=None,
        metadata={'help': 'The maximum rank of FastLoRA.'}
    )
    fastlora_attn_len: Optional[int] = field(
        default=None,
        metadata={'help': 'The number of tokens used in generative LoRA.'}
    )
    fastlora_gist_len: Optional[int] = field(
        default=None,
        metadata={'help': 'The gist length for each segment.'}
    )
    fastlora_alpha: Optional[float] = field(
        default=None,
        metadata={'help': 'LoRA scaling factor.'}
    )
    fastlora_dropout: Optional[float] = field(
        default=None,
        metadata={'help': 'LoRA dropout p.'}
    )
    fastlora_param: Optional[List[str]] = field(
        default=None,
        metadata={'help': 'The introduced parameters for ultragist.', 'nargs': '+'}
    )
    fastlora_arch: Optional[str] = field(
        default=None,
        metadata={'help': 'The architecture of FastLoRA.'}
    )
    fastlora_norm: Optional[str] = field(
        default=None,
        metadata={'help': 'The normalization method of FastLoRA.'}
    )
    fastlora_init: Optional[str] = field(
        default=None,
        metadata={'help': 'The initialization method of FastLoRA.'}
    )
    fastlora_merge: Optional[str] = field(
        default=None,
        metadata={'help': 'The initialization method of FastLoRA.'}
    )
    fastlora_training_attention_mask: Optional[str] = field(
        default=None,
        metadata={'help': 'The target modules to apply FastLoRA.'}
    )
    fastlora_diag_fix: bool = field(
        default=False,
        metadata={'help': 'Use diagonal=0 instead of diagonal=-1 for causal merge. Allows segment 0 to use its own hidden states.'}
    )
    fastlora_use_mlp: bool = field(
        default=False,
        metadata={'help': 'Use MLP architecture (2-layer MLP for query/key projections) instead of simple linear A2/A3.'}
    )
    fastlora_normalize_ss: bool = field(
        default=False,
        metadata={'help': 'Normalize ss matrix by sqrt(token_count) after key@value multiplication to prevent magnitude explosion.'}
    )
    fastlora_normalize_ss_after_merge: bool = field(
        default=False,
        metadata={'help': 'Apply 1/sqrt(token_count) scaling after merging instead of before merging for final merged product normalization.'}
    )
    fastlora_sqrt_after_merge: bool = field(
        default=False,
        metadata={'help': 'Apply sqrt normalization to the final merged ss matrix instead of token-based scaling.'}
    )
    fastlora_use_linear_strategy: bool = field(
        default=False,
        metadata={'help': 'Use linear attention stabilization strategies: QK-Norm on A1/A2/A3, explicit denominator normalization, and spectral norm constraints.'}
    )
    fastlora_norm_a1a2a3: bool = field(
        default=False,
        metadata={'help': 'Apply LayerNorm to A1, A2, A3 outputs (QK-Norm) with conservative 0.1 scaling for projection stabilization.'}
    )
    fastlora_key_denominator_norm: bool = field(
        default=False,
        metadata={'help': 'Apply explicit denominator normalization using key sum magnitude (like linear attention denominator).'}
    )
    fastlora_add_embeddings: bool = field(
        default=False,
        metadata={'help': 'Add learnable module-type embeddings to allow different adaptation patterns for different module types (q_proj, k_proj, etc.).'}
    )
    fastlora_add_layer_embeddings: bool = field(
        default=False,
        metadata={'help': 'Add learnable layer-index embeddings to allow different adaptation patterns for different depths (early vs late layers).'}
    )
    
    # GELU activation flags
    fastlora_use_activations_a1: bool = field(
        default=False,
        metadata={'help': 'Add GELU activation after A1 projection for more expressive query representations.'}
    )
    fastlora_use_activations_a2_a3: bool = field(
        default=False,
        metadata={'help': 'Add GELU activations after A2 and A3 projections for more expressiveness in key/value generation.'}
    )
    fastlora_activation_type: str = field(
        default="gelu",
        metadata={'help': 'Type of activation to use: {gelu, silu, relu, swish}'}
    )
    fastlora_use_activations_after_ss: bool = field(
        default=False,
        metadata={'help': 'Add activation after the ss matrix multiplication (query @ ss) before applying B.'}
    )
    fastlora_use_activations_ss_softmax: bool = field(
        default=False,
        metadata={'help': 'Apply softmax activation to ss matrix for stability (like attention mechanism).'}
    )
    fastlora_use_activations_ss_tanh: bool = field(
        default=False,
        metadata={'help': 'Apply tanh activation to ss matrix for stability and bounded values.'}
    )
    fastlora_use_activations_ss_after_merge: bool = field(
        default=False,
        metadata={'help': 'Apply ss activations (softmax/tanh) after merging instead of before merging for final merged product activation.'}
    )
    fastlora_use_square_ss: bool = field(
        default=False,
        metadata={'help': 'Use square ss matrix [inter_size, inter_size] instead of [inter_size, r] for richer interactions.'}
    )
    fastlora_bilinear: bool = field(
        default=False,
        metadata={'help': 'Alternate between detaching c_key_states and c_value_states for gradient stability in bilinear ss computation.'}
    )
    fastlora_outer_product_norm: bool = field(
        default=False,
        metadata={'help': 'Apply learnable normalization to outer product matrix for gradient stability (like LayerNorm but for the entire outer product).'}
    )
    fastlora_learnable_frobenius_norm: bool = field(
        default=False,
        metadata={'help': 'Apply learnable Frobenius normalization to outer product matrix (normalizes magnitude while preserving mean structure).'}
    )
    fastlora_direct_hidden_outer_product: bool = field(
        default=False,
        metadata={'help': 'Bypass A2/A3 projections and compute outer product directly from hidden states (H^T @ H). Useful for testing if projections contribute to gradient explosion.'}
    )
    
    # Deep Context Refiner (Transformer-based hypernetwork) arguments
    fastlora_use_deep_refiner: bool = field(
        default=False,
        metadata={'help': 'Use Deep Context Refiner architecture (stacked transformer blocks for context processing).'}
    )
    fastlora_refiner_layers: Optional[int] = field(
        default=2,
        metadata={'help': 'Number of layers in the Deep Context Refiner stack.'}
    )
    fastlora_refiner_ffn_size: Optional[int] = field(
        default=None,
        metadata={'help': 'FFN hidden size in Deep Context Refiner blocks. Defaults to 4 * fastlora_inter_size.'}
    )
    fastlora_parallelize: bool = field(
        default=False,
        metadata={'help': 'Parallelize FastLoRA generator computations across all modules using batched operations for better GPU utilization.'}
    )
    fastlora_use_last: bool = field(
        default=False,
        metadata={'help': 'Share FastLoRA adapters across layers, using only the last layer\'s adapters with layer embeddings for differentiation. Automatically enables layer embeddings.'}
    )
    
    # Transformer blocks for hidden state refinement (prepended before outer product computation)
    fastlora_use_transformer_blocks: bool = field(
        default=False,
        metadata={'help': 'Use transformer blocks to refine hidden states before outer product weight prediction.'}
    )
    fastlora_transformer_layers: Optional[int] = field(
        default=2,
        metadata={'help': 'Number of transformer layers for hidden state refinement.'}
    )
    fastlora_transformer_heads: Optional[int] = field(
        default=4,
        metadata={'help': 'Number of attention heads in transformer blocks.'}
    )
    fastlora_transformer_ffn_size: Optional[int] = field(
        default=None,
        metadata={'help': 'FFN hidden size in transformer blocks. Defaults to 4 * hidden_size.'}
    )
    fastlora_transformer_dropout: Optional[float] = field(
        default=0.1,
        metadata={'help': 'Dropout rate in transformer blocks.'}
    )

    max_new_tokens: Optional[int] = field(
        default=None,
        metadata={'help': 'How many tokens at maximum to return?'},
    )
    do_sample: Optional[bool] = field(
        default=None,
        metadata={'help': 'Do sampling when decoding?'},
    )
    temperature: Optional[float] = field(
        default=None,
        metadata={'help': 'Sampling temperature.'},
    )
    top_p: Optional[float] = field(
        default=None,
        metadata={'help': "If set to float < 1, only the smallest set of most probable tokens with probabilities that add up to `top_p` or higher are kept for generation."}
    )

    # load baselines
    baseline: Optional[str] = field(
        default=None,
        metadata={'help': 'Load Longlora model, AutoCompressors model, Gisting model, and CCM model.'}
    )
    longlora_s2_attn: bool = field(
        default=True,
    )
    autocompr_segment_length: Optional[int] = field(
        default=None,
    )

    def resolve_path(self, path):
        """Resolve any path starting with 'ultragist:' to relative path against data_root."""
        pattern = "ultragist:"
        # resolve relative data paths when necessary
        if isinstance(path, list):
            for i, x in enumerate(path):
                if x.startswith(pattern):
                    path[i] = os.path.join(self.data_root, x.replace(pattern, ""))
        else:
            if path.startswith(pattern):
                path = os.path.join(self.data_root, path.replace(pattern, ""))

        return path
    
    def get_generation_config(self):
        generation_config = {}
        if self.max_new_tokens is not None:
            generation_config["max_new_tokens"] = self.max_new_tokens
        if self.do_sample is not None:
            generation_config["do_sample"] = self.do_sample
        if self.temperature is not None:
            generation_config["temperature"] = self.temperature
        if self.top_p is not None:
            generation_config["top_p"] = self.top_p
        return generation_config

    def to_dict(self):
        return asdict(self)

    def save(self, path):
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f)

    def __post_init__(self):
        if self.train_data is not None:
            self.train_data = self.resolve_path(self.train_data)

        if self.eval_data is not None:
            self.eval_data = self.resolve_path(self.eval_data)

        if hasattr(self, "output_dir") and self.output_dir is not None:
            self.output_dir = self.resolve_path(self.output_dir)

        if hasattr(self, "result_dir"):
            if self.result_dir is None: 
                if self.lora is not None:
                    name_or_path_components = [x for x in self.lora.split("/") if len(x)][-2:]
                else:
                    name_or_path_components = [x for x in self.model_name_or_path.split("/") if len(x)][-2:]
                self.result_dir = os.path.join(*name_or_path_components)
            else:
                self.result_dir = self.resolve_path(self.result_dir)


@dataclass
class TrainingArgs(TrainingArguments):
    # ==============================
    # Common arguments
    # ==============================
    output_dir: str = field(
        default="data/outputs/pretrain",
    )

    per_device_train_batch_size: int = field(
        default=1,
        metadata={'help': 'Train batch size.'}
    )
    per_device_eval_batch_size: int = field(
        default=1,
        metadata={'help': 'Evaluation batch size.'}
    )
    remove_unused_columns: bool = field(
        default=False,
        metadata={'help': 'Remove columns in the dataset that are not registered in the forward function?'}
    )
    ddp_find_unused_parameters: bool = field(
        default=False,
        metadata={'help': 'Find unusuable parameters?'}
    )
    # NOTE: essential to keep comuputation graph because we need gradients for ultragist tokens
    use_reentrant: Optional[bool] = field(
        default=None,
        metadata={'help': "Use reetrant in gradient checkpointing?"}
    )
    report_to: str = field(
        default="none",
        metadata={'help': 'Log results by external tools?'}
    )

    # ==============================
    # Customized arguments
    # ==============================
    min_length: int = field(
        default=0,
        metadata={'help': 'How many tokens at minimum for training?'}
    )

    group_by_stride: Optional[str] = field(
        default=None,
        metadata={'help': 'Group the training data instances by the number of strides in the ultragist model. {relaxed, strict}'}
    )
    sort_by_stride: Optional[str] = field(
        default=None,
        metadata={'help': 'Sort the training data instances by the number of strides in the ultragist model. {ascend, descend}'}
    )
    only_train_ultragist: bool = field(
        default=True,
        metadata={'help': 'Freeze LLM parameters when training ultragist parameters?'}
    )
    
    eval_method: str = field(
        default="perplexity",
        metadata={'help': 'How to evaluate during training? {perplexity, generation}'}
    )
    eval_max_length: int = field(
        default=4096,
        metadata={'help': 'How many tokens at maximum for each input in evaluation.'},
    )
    eval_min_length: int = field(
        default=0,
        metadata={'help': 'How many tokens at minimum for each input in evaluation.'},
    )
    eval_ultragist_ratio: List[int] = field(
        default_factory=lambda: [32],
        metadata={'help': 'Condensing ratios for ultragists in evaluation.'}
    )
    eval_ultragist_ratio_mix: str = field(
        default="adapt-1024",
        metadata={'help': 'How to determine the ultragist_ratio for each input. {step-random, instance-random, adapt-x}'}
    )
    max_eval_num: Optional[int] = field(
        default=None,
        metadata={'help': 'How many samples for validation?'}
    )

    # lora_tune: bool = field(
    #     default=False,
    #     metadata={"help": "Use LoRA fine-tuning?"},
    # )
    # lora_rank: int = field(
    #     default=32,
    #     metadata={'help': 'LoRA rank.'}
    # )
    # lora_alpha: int = field(
    #     default=16,
    #     metadata={'help': 'LoRA scaling factor.'}
    # )
    # lora_dropout: float = field(
    #     default=0.,
    #     metadata={'help': 'LoRA dropout p.'}
    # )
    # lora_targets: List[str] = field(
    #     default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"],
    #     metadata={"help": "Module name patterns to add LoRA."},
    # )
    # lora_extra_params: List[str] = field(
    #     default_factory=lambda: ["embed_tokens", "norm"],
    #     metadata={"help": "Extra trainable parameters except LoRA weights, if low rank training."},
    # )

    metrics: List[str] = field(
        default_factory=lambda: [],
        metadata={'help': 'List of metrics. {rouge, save_result}'}
    )
    log_path: str = field(
        default="data/outputs/metrics.log",
        metadata={'help': 'Log file path.'}
    )
    wandb_watch_log_freq: int = field(
        default=None,
        metadata={'help': 'Logging frequency for wandb.watch.'}
    )


    def __post_init__(self):
        if self.use_reentrant is not None:
            self.gradient_checkpointing_kwargs = {"use_reentrant": self.use_reentrant}
        return super().__post_init__()
