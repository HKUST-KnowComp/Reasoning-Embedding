#!/usr/bin/env python3
"""
Train linear probes on one model's representations and evaluate their transfer to
another model. Useful for quantifying the functional similarity between
sentence-embedding models and large language models.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pickle

try:
    import joblib  # type: ignore
except ImportError:  # pragma: no cover - fallback when joblib is missing
    joblib = None  # type: ignore
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from colorama import Fore, Style
from datasets import load_dataset
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from hrsa.plot_style import get_palette
from hrsa.utils import (
    canonical_dataset_slug,
    sanitize_model_name,
)

def dump_artifact(obj: object, path: Path) -> None:
    if joblib is not None:
        joblib.dump(obj, path)
    else:  # pragma: no cover - executed only when joblib is missing
        with path.open("wb") as f:
            pickle.dump(obj, f)


def load_artifact(path: Path) -> object:
    if joblib is not None:
        return joblib.load(path)
    with path.open("rb") as f:  # pragma: no cover - executed only when joblib is missing
        return pickle.load(f)


def chunk_list(items: List[str], chunk_size: int) -> Iterable[List[str]]:
    """Yield successive chunk_size-sized chunks from items."""
    for i in range(0, len(items), chunk_size):
        yield items[i : i + chunk_size]


def resolve_device(device: str) -> torch.device:
    if device == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(device)


def resolve_dtype(dtype_name: Optional[str]) -> Optional[torch.dtype]:
    if dtype_name is None:
        return None
    name = dtype_name.lower()
    if name in {"fp16", "float16"}:
        return torch.float16
    if name in {"bf16", "bfloat16"}:
        return torch.bfloat16
    if name in {"fp32", "float32"}:
        return torch.float32
    raise ValueError(f"Unsupported dtype '{dtype_name}'.")


@dataclass
class DatasetMetadata:
    dataset_name: str
    dataset_config: Optional[str]
    dataset_split: str
    text_column: str
    label_column: str
    task_type: str
    label_mapping: Optional[Dict[str, str]] = None
    num_samples: Dict[str, int] = field(default_factory=dict)


class TaskDatasetPreparer:
    """Load, split, and cache datasets for downstream linear probes."""

    TARGET_COLUMN = "target_value"

    def __init__(
        self,
        dataset_name: str,
        dataset_config: Optional[str],
        dataset_split: str,
        task_type: str,
        text_column: str,
        label_column: str,
        task_name: Optional[str],
        cache_root: str,
        train_fraction: float,
        val_fraction: float,
        test_fraction: Optional[float],
        max_samples: Optional[int],
        min_class_samples: int,
        seed: int,
        force_refresh: bool = False,
    ) -> None:
        self.dataset_name = dataset_name
        self.dataset_config = dataset_config
        self.dataset_split = dataset_split
        self.task_type = task_type
        self.text_column = text_column
        self.label_column = label_column
        self.seed = seed
        self.force_refresh = force_refresh
        self.max_samples = max_samples
        self.min_class_samples = min_class_samples
        self.train_fraction = train_fraction
        self.val_fraction = val_fraction
        self.test_fraction = (
            test_fraction
            if test_fraction is not None
            else max(0.0, 1.0 - train_fraction - val_fraction)
        )

        total = self.train_fraction + self.val_fraction + self.test_fraction
        if not math.isclose(total, 1.0, abs_tol=1e-4):
            raise ValueError(
                f"Fractions must sum to 1.0 but got {total:.3f} "
                f"(train={self.train_fraction}, val={self.val_fraction}, test={self.test_fraction})."
            )

        dataset_slug = canonical_dataset_slug(dataset_name)
        derived_task = task_name or f"{dataset_slug}-{label_column}"
        self.task_slug = sanitize_model_name(derived_task)
        self.cache_dir = Path(cache_root) / self.task_slug
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.meta_path = self.cache_dir / "metadata.json"
        self.label_encoder: Optional[LabelEncoder] = None
        self.metadata: Optional[DatasetMetadata] = None
        self.id_to_label: Optional[Dict[int, str]] = None
        self.available_splits: List[str] = []

    def prepare(self) -> Dict[str, pd.DataFrame]:
        if not self.force_refresh and self._cache_exists():
            print(
                f"{Fore.GREEN}Loading cached dataset splits from {self.cache_dir}{Style.RESET_ALL}"
            )
            return self._load_cached()

        print(
            f"{Fore.BLUE}Loading dataset {self.dataset_name} (config={self.dataset_config}, split={self.dataset_split}){Style.RESET_ALL}"
        )
        
        if self.dataset_name.endswith(".jsonl"):
            dataset = load_dataset("json", data_files=self.dataset_name, split=self.dataset_split)
        else:
            dataset = load_dataset(
                self.dataset_name,
                self.dataset_config,
                split=self.dataset_split,
            )
        df = pd.DataFrame(dataset)

        if self.text_column not in df.columns or self.label_column not in df.columns:
            raise ValueError(
                f"Columns not found. Available columns: {list(df.columns)}"
            )

        df = df[[self.text_column, self.label_column]].dropna()
        print("Dataset after dropping NA values: %d rows", len(df))

        if self.task_type == "classification":
            df = self._filter_min_class_samples(df)
            self.label_encoder = LabelEncoder()
            df[self.TARGET_COLUMN] = self.label_encoder.fit_transform(
                df[self.label_column]
            )
            self.id_to_label = {
                int(i): str(label) for i, label in enumerate(self.label_encoder.classes_)
            }
        else:
            df[self.TARGET_COLUMN] = pd.to_numeric(
                df[self.label_column], errors="coerce"
            )
            df = df.dropna(subset=[self.TARGET_COLUMN])

        if self.max_samples and len(df) > self.max_samples:
            df = self._stratified_sample(df, self.max_samples)
            print("Sampled down to %d rows for efficiency.", len(df))

        splits = self._split_dataframe(df)
        self._cache_splits(splits)
        return splits

    def _cache_exists(self) -> bool:
        required = [self.cache_dir / "train.jsonl", self.cache_dir / "test.jsonl"]
        exists = all(path.exists() for path in required) and self.meta_path.exists()
        return exists

    def _load_cached(self) -> Dict[str, pd.DataFrame]:
        with self.meta_path.open("r", encoding="utf-8") as f:
            raw_meta = json.load(f)
        self.metadata = DatasetMetadata(**raw_meta)
        if self.metadata.label_mapping:
            self.id_to_label = {int(k): v for k, v in self.metadata.label_mapping.items()}

        splits: Dict[str, pd.DataFrame] = {}
        for split_name in ["train", "val", "test"]:
            path = self.cache_dir / f"{split_name}.jsonl"
            if path.exists():
                splits[split_name] = pd.read_json(path, lines=True)
        self.available_splits = list(splits.keys())
        return splits

    def _filter_min_class_samples(self, df: pd.DataFrame) -> pd.DataFrame:
        value_counts = df[self.label_column].value_counts()
        valid_classes = value_counts[value_counts >= self.min_class_samples].index
        filtered = df[df[self.label_column].isin(valid_classes)]
        print(
            "Kept %d classes (>= %d samples). Rows remaining: %d.",
            len(valid_classes),
            self.min_class_samples,
            len(filtered),
        )
        return filtered

    def _stratified_sample(self, df: pd.DataFrame, max_samples: int) -> pd.DataFrame:
        if self.task_type != "classification":
            return df.sample(max_samples, random_state=self.seed)

        n_classes = df[self.label_column].nunique()
        per_class = max(1, max_samples // n_classes)
        sampled = (
            df.groupby(self.label_column, group_keys=False)
            .apply(
                lambda x: x.sample(
                    min(len(x), per_class), random_state=self.seed
                )
            )
            .reset_index(drop=True)
        )
        if len(sampled) > max_samples:
            sampled = sampled.sample(max_samples, random_state=self.seed)
        return sampled

    def _split_dataframe(self, df: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        splits: Dict[str, pd.DataFrame] = {}
        stratify_labels = (
            df[self.TARGET_COLUMN] if self.task_type == "classification" else None
        )

        if self.val_fraction == 0:
            train_df, test_df = train_test_split(
                df,
                test_size=self.test_fraction,
                random_state=self.seed,
                stratify=stratify_labels,
            )
            splits["train"] = train_df.reset_index(drop=True)
            splits["test"] = test_df.reset_index(drop=True)
        else:
            train_df, temp_df = train_test_split(
                df,
                test_size=self.val_fraction + self.test_fraction,
                random_state=self.seed,
                stratify=stratify_labels,
            )
            temp_stratify = (
                temp_df[self.TARGET_COLUMN]
                if self.task_type == "classification"
                else None
            )
            val_ratio = self.val_fraction / (self.val_fraction + self.test_fraction)
            val_df, test_df = train_test_split(
                temp_df,
                test_size=1.0 - val_ratio,
                random_state=self.seed,
                stratify=temp_stratify,
            )
            splits["train"] = train_df.reset_index(drop=True)
            splits["val"] = val_df.reset_index(drop=True)
            splits["test"] = test_df.reset_index(drop=True)

        self.available_splits = list(splits.keys())
        self.metadata = DatasetMetadata(
            dataset_name=self.dataset_name,
            dataset_config=self.dataset_config,
            dataset_split=self.dataset_split,
            text_column=self.text_column,
            label_column=self.label_column,
            task_type=self.task_type,
            label_mapping=self.id_to_label,
            num_samples={split: len(df_split) for split, df_split in splits.items()},
        )
        return splits

    def _cache_splits(self, splits: Dict[str, pd.DataFrame]) -> None:
        for split_name, df_split in splits.items():
            path = self.cache_dir / f"{split_name}.jsonl"
            df_split.to_json(path, orient="records", lines=True)
        if self.metadata is None:
            raise ValueError("Metadata must be set before caching splits.")
        with self.meta_path.open("w", encoding="utf-8") as f:
            json.dump(self.metadata.__dict__, f, indent=2)

    @property
    def target_column(self) -> str:
        return self.TARGET_COLUMN


class RepresentationExtractor:
    """Uniform interface for extracting representations from different model families."""

    def __init__(
        self,
        model_name: str,
        model_type: str,
        device: str = "auto",
        layer: int = -1,
        pooling: str = "mean",
        batch_size: int = 16,
        max_length: int = 512,
        normalize: bool = True,
        dtype: Optional[str] = None,
    ) -> None:
        self.model_name = model_name
        self.model_type = model_type
        self.layer = layer
        self.pooling = pooling
        self.batch_size = batch_size
        self.max_length = max_length
        self.normalize = normalize
        self.device = resolve_device(device)
        self.dtype = resolve_dtype(dtype)

        if self.model_type == "sentence_transformer":
            self.model = SentenceTransformer(
                model_name,
                device=str(self.device),
                # model_kwargs={"torch_dtype": self.dtype} if self.dtype else None,
                model_kwargs={"dtype": "bfloat16"},
            )
            self.tokenizer = None
        elif self.model_type == "causal_lm":
            self.tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token or self.tokenizer.cls_token
            self.model = AutoModelForCausalLM.from_pretrained(
                model_name,
                dtype=self.dtype,
                output_hidden_states=True,
            ).to(self.device)
            self.model.eval()
        else:
            raise ValueError(
                f"Unsupported model_type '{model_type}'. "
                "Use 'sentence_transformer' or 'causal_lm'."
            )

    def encode(self, texts: List[str], desc: str) -> np.ndarray:
        if len(texts) == 0:
            return np.empty((0, 0), dtype=np.float32)

        print(
            f"{Fore.CYAN}Encoding {len(texts)} examples with {self.model_name} ({desc}){Style.RESET_ALL}"
        )

        if self.model_type == "sentence_transformer":
            embeddings = self.model.encode(
                texts,
                batch_size=self.batch_size,
                show_progress_bar=True,
                normalize_embeddings=self.normalize,
                convert_to_numpy=True,
            )
            return embeddings.astype(np.float32)

        features: List[np.ndarray] = []
        for batch in tqdm(
            list(chunk_list(texts, self.batch_size)),
            desc=f"{desc} ({self.model_type})",
        ):
            tokenized = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            ).to(self.device)
            with torch.no_grad():
                outputs = self.model(
                    **tokenized,
                    output_hidden_states=True,
                    use_cache=False,
                )
            hidden_states = outputs.hidden_states
            selected = hidden_states[self.layer]
            pooled = self._pool_hidden(selected, tokenized["attention_mask"])
            if self.normalize:
                pooled = F.normalize(pooled, p=2, dim=-1)
            features.append(pooled.detach().cpu().numpy())
            del tokenized, outputs, hidden_states, selected, pooled
            if self.device.type == "cuda":
                torch.cuda.empty_cache()
        return np.concatenate(features, axis=0).astype(np.float32)

    def _pool_hidden(self, hidden: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        if self.pooling == "cls":
            return hidden[:, 0]

        if self.pooling == "last":
            lengths = attention_mask.sum(dim=1) - 1
            lengths = torch.clamp(lengths, min=0)
            batch_indices = torch.arange(hidden.size(0), device=hidden.device)
            return hidden[batch_indices, lengths]

        # Default to attention-mask mean pooling
        mask = attention_mask.unsqueeze(-1)
        summed = (hidden * mask).sum(dim=1)
        denom = mask.sum(dim=1).clamp(min=1)
        return summed / denom


class FeatureProjector:
    """Optional projection/alignment so models with different dims share a feature basis."""

    def __init__(
        self,
        method: str = "none",
        shared_dim: Optional[int] = None,
        seed: int = 42,
    ) -> None:
        self.method = method
        self.shared_dim = shared_dim
        self.seed = seed
        self.max_dim: Optional[int] = None
        self.pca: Optional[PCA] = None

    def fit(self, feature_dict: Dict[str, np.ndarray]) -> None:
        if self.method == "none":
            dims = {feat.shape[1] for feat in feature_dict.values()}
            if len(dims) != 1:
                raise ValueError(
                    "Feature dimensions differ. Use --projector pca or provide shared_dim."
                )
            self.max_dim = dims.pop()
            return

        if self.method != "pca":
            raise ValueError(f"Unsupported projector method: {self.method}")

        self.max_dim = max(feat.shape[1] for feat in feature_dict.values())

        padded = [self._pad(feat) for feat in feature_dict.values()]
        stacked = np.vstack(padded)
        n_components = self.shared_dim or min(self.max_dim, stacked.shape[1])
        self.pca = PCA(n_components=n_components, random_state=self.seed)
        self.pca.fit(stacked)
        print(
            "Fitted PCA projector to %d samples (n_components=%d).",
            stacked.shape[0],
            n_components,
        )

    def transform(self, features: np.ndarray) -> np.ndarray:
        if self.method == "none":
            return features
        if self.pca is None or self.max_dim is None:
            raise ValueError("Projector must be fitted before calling transform.")
        padded = self._pad(features)
        transformed = self.pca.transform(padded)
        return transformed.astype(np.float32)

    def _pad(self, features: np.ndarray) -> np.ndarray:
        if self.max_dim is None:
            raise ValueError("Projector has not been fitted.")
        if features.shape[1] == self.max_dim:
            return features
        if features.shape[1] > self.max_dim:
            return features[:, : self.max_dim]
        pad_width = self.max_dim - features.shape[1]
        return np.pad(features, ((0, 0), (0, pad_width)))

    def save(self, path: Path) -> None:
        payload = {
            "method": self.method,
            "shared_dim": self.shared_dim,
            "seed": self.seed,
            "max_dim": self.max_dim,
            "pca": self.pca,
        }
        dump_artifact(payload, path)

    @classmethod
    def load(cls, path: Path) -> "FeatureProjector":
        payload = load_artifact(path)
        projector = cls(
            method=payload["method"],
            shared_dim=payload["shared_dim"],
            seed=payload["seed"],
        )
        projector.max_dim = payload["max_dim"]
        projector.pca = payload["pca"]
        return projector


class LinearProbeTrainer:
    """Wrapper around scikit-learn linear models with scaling and serialization."""

    def __init__(
        self,
        task_type: str,
        max_iter: int = 1000,
        c_value: float = 1.0,
        penalty: str = "l2",
        solver: str = "lbfgs",
        ridge_alpha: float = 1.0,
        n_jobs: int = -1,
        seed: int = 42,
    ) -> None:
        self.task_type = task_type
        self.max_iter = max_iter
        self.c_value = c_value
        self.penalty = penalty
        self.solver = solver
        self.ridge_alpha = ridge_alpha
        self.n_jobs = n_jobs
        self.seed = seed
        self.scaler: Optional[StandardScaler] = None
        self.model = None
        self.label_mapping: Optional[Dict[int, str]] = None

    def fit(self, features: np.ndarray, targets: np.ndarray) -> None:
        self.scaler = StandardScaler(with_mean=True, with_std=True)
        features_scaled = self.scaler.fit_transform(features)

        if self.task_type == "classification":
            self.model = LogisticRegression(
                penalty=self.penalty,
                C=self.c_value,
                solver=self.solver,
                max_iter=self.max_iter,
                n_jobs=self.n_jobs,
                random_state=self.seed,
                multi_class="auto",
            )
        else:
            self.model = Ridge(alpha=self.ridge_alpha, random_state=self.seed)

        self.model.fit(features_scaled, targets)
        
        print(f"{Fore.GREEN}Fitted linear probe for {self.task_type}{Style.RESET_ALL}")

    def predict(self, features: np.ndarray) -> np.ndarray:
        if self.scaler is None or self.model is None:
            raise ValueError("Model not fitted.")
        features_scaled = self.scaler.transform(features)
        return self.model.predict(features_scaled)

    def evaluate(self, features: np.ndarray, targets: np.ndarray) -> Dict[str, float]:
        if len(targets) == 0:
            return {}
        preds = self.predict(features)
        if self.task_type == "classification":
            return {
                "accuracy": float(accuracy_score(targets, preds)),
                "macro_f1": float(f1_score(targets, preds, average="macro")),
                "weighted_f1": float(f1_score(targets, preds, average="weighted")),
            }
        mse = mean_squared_error(targets, preds)
        mae = mean_absolute_error(targets, preds)
        return {
            "mae": float(mae),
            "mse": float(mse),
            "rmse": float(math.sqrt(mse)),
            "r2": float(r2_score(targets, preds)),
        }

    def save(self, path: Path) -> None:
        payload = {
            "task_type": self.task_type,
            "max_iter": self.max_iter,
            "c_value": self.c_value,
            "penalty": self.penalty,
            "solver": self.solver,
            "ridge_alpha": self.ridge_alpha,
            "n_jobs": self.n_jobs,
            "seed": self.seed,
            "scaler": self.scaler,
            "model": self.model,
            "label_mapping": self.label_mapping,
        }
        dump_artifact(payload, path)

    @classmethod
    def load(cls, path: Path) -> "LinearProbeTrainer":
        payload = load_artifact(path)
        trainer = cls(
            task_type=payload["task_type"],
            max_iter=payload["max_iter"],
            c_value=payload["c_value"],
            penalty=payload["penalty"],
            solver=payload["solver"],
            ridge_alpha=payload["ridge_alpha"],
            n_jobs=payload["n_jobs"],
            seed=payload["seed"],
        )
        trainer.scaler = payload["scaler"]
        trainer.model = payload["model"]
        trainer.label_mapping = payload.get("label_mapping")
        return trainer


def write_metrics_outputs(
    metrics: Dict[str, Dict[str, float]],
    metadata: DatasetMetadata,
    out_json: Path,
    out_tsv: Path,
) -> None:
    result_payload = {
        "dataset": metadata.__dict__,
        "metrics": metrics,
    }
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(result_payload, f, indent=2)

    rows = []
    for split_name, metric_dict in metrics.items():
        row = {"split": split_name}
        row.update(metric_dict)
        rows.append(row)
    if rows:
        pd.DataFrame(rows).to_csv(out_tsv, sep="\t", index=False)


def plot_primary_metric(
    metrics: Dict[str, Dict[str, float]],
    metric_name: str,
    output_path: Path,
) -> None:
    keys, values = [], []
    for split_key, metric_dict in metrics.items():
        if metric_name in metric_dict:
            keys.append(split_key)
            values.append(metric_dict[metric_name])

    if len(values) < 2:
        print(f"{Fore.YELLOW}Not enough metric points to plot {metric_name}{Style.RESET_ALL}")
        return

    palette = get_palette(len(keys))
    fig, ax = plt.subplots(figsize=(max(6, len(keys)), 4.5))
    bars = ax.bar(keys, values, color=palette)
    ax.axhline(0, color="#6c757d", linewidth=0.6)
    ax.set_ylabel(metric_name)
    ax.set_xticklabels(keys, rotation=45, ha="right")

    for bar, value in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height(),
            f"{value:.3f}",
            ha="center",
            va="bottom" if value >= 0 else "top",
            fontsize=9,
        )

    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
