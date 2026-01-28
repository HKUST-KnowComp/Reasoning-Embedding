"""
Unified configuration classes for evaluation metrics.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

from hrsa.utils import canonical_dataset_slug

import os

from dotenv import load_dotenv

load_dotenv()

METRIC_RESULTS_FOLDER = os.getenv("METRIC_RESULTS_FOLDER", "metric_results")

@dataclass
class DatasetConfig:
    """Unified configuration for dataset specifications across all metrics."""

    label: str
    name: str  # dataset name or path
    text_column: str
    label_column: Optional[str] = None  # for KNN and probe metrics
    subset: str = "main"
    split: str = "train"
    num_sentences: int = 2000

    def slug(self) -> str:
        """Generate a canonical slug for the dataset name."""
        return canonical_dataset_slug(self.name)

    def describe(self) -> str:
        """Return a human-readable description of the configuration."""
        parts = [
            f"{self.label}: dataset={self.name}",
            f"column={self.text_column}",
            f"subset={self.subset}",
            f"split={self.split}",
            f"num_sentences={self.num_sentences}",
        ]
        if self.label_column:
            parts.insert(2, f"label_column={self.label_column}")
        return " | ".join(parts)


@dataclass
class ProbeDatasetConfig(DatasetConfig):
    """Configuration for linear probe evaluation with train/val/test splits."""

    dataset_config: Optional[str] = None  # HuggingFace dataset config name
    text_column: str = "text"
    task_type: str = "classification"  # "classification" or "regression"
    train_fraction: float = 0.8
    val_fraction: float = 0.1
    test_fraction: float = 0.1
    max_samples: Optional[int] = None
    min_class_samples: int = 100
    seed: int = 42

    def slug(self) -> str:
        """Generate a canonical slug for the dataset name."""
        return canonical_dataset_slug(self.name)

    def describe(self) -> str:
        """Return a human-readable description of the configuration."""
        return (
            f"{self.label}: dataset={self.name} | "
            f"task_type={self.task_type} | "
            f"text_column={self.text_column} | "
            f"label_column={self.label_column} | "
            f"train={self.train_fraction} | "
            f"val={self.val_fraction} | "
            f"test={self.test_fraction}"
        )
