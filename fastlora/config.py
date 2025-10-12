from dataclasses import asdict, dataclass, field
from peft import PeftConfig

from typing import Optional, Union, List

@dataclass
class FastLoraConfig(PeftConfig):

    fastlora_r: Optional[int] = field(default=32, metadata={"help": "The number of attention heads."})
    fastlora_inter_size: Optional[int] = field(default=None, metadata={"help": "The number of attention heads."})
    fastlora_window: Optional[int] = field(default=1024, metadata={"help": "The number of attention heads."})
    fastlora_max_rank: Optional[int] = field(default=128, metadata={"help": "The number of attention heads."})
    fastlora_attn_len: Optional[int] = field(default=8192, metadata={"help": "The number of attention heads."})
    fastlora_gist_len: Optional[int] = field(default=0, metadata={"help": "The number of attention heads."})
    fastlora_alpha: Optional[int] = field(default=64, metadata={"help": "The number of attention heads."})
    fastlora_dropout: Optional[float] = field(default=0.1, metadata={"help": "The dropout ratio for the attention probabilities."})
    fastlora_arch: Optional[str] = field(default="aassbb", metadata={"help": "The number of attention heads."})
    fastlora_norm: Optional[str] = field(default="forbenius", metadata={"help": "The number of attention heads."})
    fastlora_init: Optional[str] = field(default="random", metadata={"help": "The number of attention heads."})
    fastlora_merge: Optional[str] = field(default="mean", metadata={"help": "The number of attention heads."})
    fastlora_param: Optional[List[str]] = field(default=None, metadata={"help": "the target modules to apply fastlora"})
    fastlora_training_attention_mask: Optional[str] = field(default=None, metadata={"help": "the target modules to apply fastlora"})
    fastlora_diag_fix: bool = field(default=False, metadata={"help": "Use diagonal=0 instead of diagonal=-1 for causal merge. Allows segment 0 to use its own hidden states."})
    fastlora_use_mlp: bool = field(default=False, metadata={"help": "Use MLP architecture (2-layer MLP for query/key projections) instead of simple linear A2/A3."})
    fastlora_normalize_ss: bool = field(default=False, metadata={"help": "Normalize ss matrix by sqrt(token_count) after key@value multiplication to prevent magnitude explosion."})
    fastlora_add_embeddings: bool = field(default=False, metadata={"help": "Add learnable module-type embeddings to allow different adaptation patterns for different module types (q_proj, k_proj, etc.)."})
    fastlora_add_layer_embeddings: bool = field(default=False, metadata={"help": "Add learnable layer-index embeddings to allow different adaptation patterns for different depths (early vs late layers)."})
    fastlora_use_activations_a2_a3: bool = field(default=False, metadata={"help": "Add GELU activations after A2 and A3 projections for more expressiveness in key/value generation."})
    fastlora_use_activations_a1: bool = field(default=False, metadata={"help": "Add GELU activation after A1 projection for more expressive query representations."})
    fastlora_activation_type: str = field(default="gelu", metadata={"help": "Type of activation to use: {gelu, silu, relu, swish}"})
    fastlora_use_activations_after_ss: bool = field(default=False, metadata={"help": "Add activation after the ss matrix multiplication (query @ ss) before applying B."})
    
    # Deep Context Refiner (Transformer-based hypernetwork) arguments
    fastlora_use_deep_refiner: bool = field(default=False, metadata={"help": "Use Deep Context Refiner architecture (stacked transformer blocks for context processing)."})
    fastlora_refiner_layers: int = field(default=2, metadata={"help": "Number of layers in the Deep Context Refiner stack."})
    fastlora_refiner_ffn_size: Optional[int] = field(default=None, metadata={"help": "FFN hidden size in Deep Context Refiner blocks. Defaults to 4 * fastlora_inter_size."})
    fastlora_parallelize: bool = field(default=False, metadata={"help": "Parallelize FastLoRA generator computations across all modules using batched operations for better GPU utilization."})
    fastlora_use_last: bool = field(default=False, metadata={"help": "Share FastLoRA adapters across layers, using only the last layer's adapters with layer embeddings for differentiation. Automatically enables layer embeddings."})

    target_modules: Optional[List[str]] = field(default=None, metadata={"help": "The target modules to apply fastlora."})

    # obsolete parameters, for compatibility with the original code
    lora_r: Optional[int] = field(default=0, metadata={"help": "The number of attention heads."})
    lora_alpha: Optional[int] = field(default=0, metadata={"help": "The number of attention heads."})
    lora_dropout: Optional[float] = field(default=0.0, metadata={"help": "The dropout ratio for the attention probabilities."})
    lora_param: Optional[List[str]] = field(default=None, metadata={"help": "The number of attention heads."})
    layer_replication: Optional[list[tuple[int, int]]] = field(default=None, metadata={"help": "Enables replicating layers in a model to expand it to a larger model."})
    megatron_config: Optional[dict] = field(default=None, metadata={"help": "Megatron configuration."})
    bias: Optional[str] = field(default='none', metadata={"help": "Whether to use bias in the attention layer."})
    