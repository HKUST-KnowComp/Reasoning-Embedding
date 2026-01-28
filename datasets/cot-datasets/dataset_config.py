from dataclasses import dataclass
from typing import Optional


@dataclass
class DatasetConfig:
    name: str
    subset: Optional[str]
    split: str = "train"
    random_sample: bool = False
    random_seed: Optional[int] = 42
    source_filter: Optional[str] = None  # For filtering by 'source' field (e.g., NuminaMath)


# Default hard math config: LiveMathBench hard split
DEFAULT_DATASET_CONFIG = DatasetConfig(
    name="opencompass/LiveMathBench",
    subset="v202505_hard_en",
    split="test",
)

# Additional preset: HuggingFaceH4/MATH-500 test split with random sampling
MATH500_DATASET_CONFIG = DatasetConfig(
    name="HuggingFaceH4/MATH-500",
    subset=None,
    split="test",
    random_sample=False,
    random_seed=42,
)

# NuminaMath-CoT dataset configs (filtered by source)
NUMINAMATH_CN_K12_CONFIG = DatasetConfig(
    name="AI-MO/NuminaMath-CoT",
    subset=None,
    split="train",
    random_sample=True,
    random_seed=42,
    source_filter="cn_k12",
)

NUMINAMATH_MATH_CONFIG = DatasetConfig(
    name="AI-MO/NuminaMath-CoT",
    subset=None,
    split="train",
    random_sample=True,
    random_seed=42,
    source_filter="math",
)

NUMINAMATH_OLYMPIADS_CONFIG = DatasetConfig(
    name="AI-MO/NuminaMath-CoT",
    subset=None,
    split="train",
    random_sample=True,
    random_seed=42,
    source_filter="olympiads",
)

# GSM8K dataset config
GSM8K_CONFIG = DatasetConfig(
    name="openai/gsm8k",
    subset="main",
    split="test",
    random_sample=True,
    random_seed=42,
)
