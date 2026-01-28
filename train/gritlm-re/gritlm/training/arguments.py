import os
from dataclasses import dataclass, field
from typing import Optional

from transformers import TrainingArguments


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"})
    config_name: Optional[str] = field(default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"})
    tokenizer_name: Optional[str] = field(default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"})
    pooling_method: str = field(default="weightedmean", metadata={"help": "Pooling method for sentences"})
    normalized: bool = field(default=True)
    attn_implementation: str = field(default="sdpa", metadata={"help": "eager/sdpa/flash_attention_2"})
    attn: str = field(
        default="bbcc",
        metadata={
            "help": "bidirectional/causal attn for emb inst., emb sample, gen inst., gen sample"
            " e.g. bbcc is bidirectional over both emb inst. & sample but causal over gen inst. & sample"
            " cccc is causal over all; bccc is bidirectional over emb inst. but causal over rest etc."
        },
    )
    projection: int = field(default=None, metadata={"help": "Optional linear learned embedding down projection"})


@dataclass
class DataArguments:
    train_data: str = field(
        default=None,
        metadata={
            "help": "Path to folder or file with training data. If the path is a folder, for each minibatch"
            " all samples will come from one file in the folder. You can use this to ensure"
            " in-batch negatives are very difficult."
        },
    )
    train_group_size: int = field(
        default=2,
        metadata={
            "help": "Number of positive & negatives for a query in training. There is always one"
            " positive, so this argument controls the number of negatives"
            " (#negatives=train_group_size-1). Note that the number of negatives should"
            " not be larger than the numbers of negatives in the data. Besides the negatives"
            " in this group, the in-batch negatives will also be used in fine-tuning."
        },
    )
    query_max_len: int = field(
        default=32,
        metadata={"help": "The maximum tokens for the query. Sequences longer" " than this will be truncated, sequences shorter will be padded."},
    )
    passage_max_len: int = field(
        default=128,
        metadata={"help": "The maximum tokens for passages (positives & negatives). Sequences longer" " than this will be truncated, sequences shorter will be padded."},
    )
    max_example_num_per_dataset: int = field(default=100_000_000, metadata={"help": "the max number of examples for each dataset"})
    num_samples: Optional[str] = field(default=None, metadata={"help": "path to json with number of samples per dataset"})

    def __post_init__(self):
        if not os.path.exists(self.train_data):
            raise FileNotFoundError(f"cannot find file: {self.train_data}, please set a true path")


@dataclass
class CustomTrainingArguments(TrainingArguments):
    negatives_cross_device: bool = field(default=False, metadata={"help": "Share the negatives across all GPUs. This argument will extend the number of negatives."})
    temperature: Optional[float] = field(
        default=0.02,
        metadata={"help": "Similarity will be sim = sim/temperature before using them to compute loss." " A higher temperature can reduce the value of similarity between texts in downstream tasks."},
    )
    lora: bool = field(default=False, metadata={"help": "Use LoRA PEFT"})
    qlora: bool = field(default=False, metadata={"help": "Use QLoRA PEFT"})
    save_safetensors: bool = field(default=False, metadata={"help": "Save in safetensors format"})
    split_emb: bool = field(default=False, metadata={"help": "Split embedding forward / backward pass"})
    split_emb_full: bool = field(default=False, metadata={"help": "Split embedding forward / backward pass"})
    emb_q_only: bool = field(default=False, metadata={"help": "Only backprop on q's"})
    emb_p_only: bool = field(default=False, metadata={"help": "Only backprop on p's (pos & neg)"})

    debiased: bool = field(default=False, metadata={"help": "Use debiased contrastive loss"})
    tau_plus: float = field(default=0.1, metadata={"help": "tau+ for debiased contrastive loss"})

    use_muon: bool = field(default=False, metadata={"help": "Use muon optimizer"})
    use_fused_adamw: bool = field(default=False, metadata={"help": "Use fused AdamW optimizer for 5-10% speedup"})

    # Performance optimizations
    torch_compile: bool = field(default=False, metadata={"help": "Use torch.compile for 20-50% speedup"})
    torch_compile_mode: str = field(default="reduce-overhead", metadata={"help": "Torch compile mode: default, reduce-overhead, max-autotune"})
    torch_compile_backend: str = field(default="inductor", metadata={"help": "Torch compile backend"})

    # Ring-based training (alternative to GradCache)
    use_ring_loss: bool = field(default=False, metadata={"help": "Use ring-based contrastive loss instead of GradCache (faster, more memory efficient)"})
    ring_head_dim: int = field(default=256, metadata={"help": "Head dimension for ring loss (must be 16, 32, 64, 128, or 256)"})
    use_inf_loss: bool = field(default=True, metadata={"help": "Use InfProb (Flash-optimized) vs RingProb for ring loss"})
