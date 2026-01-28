"""
Base evaluation metric class with shared utilities for model comparison.
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import torch
from colorama import Fore, Style
from tqdm import trange

from hrsa.utils import get_model, clear_memory, sanitize_model_name
from hrsa.config import DatasetConfig

@dataclass
class ActivationBundle:
    """Container for activations and token bookkeeping for a single model."""

    model_name: str
    activations: torch.Tensor  # shape (num_layers, total_tokens, hidden_dim) on CPU
    token_counts: torch.Tensor  # per-sentence token counts on CPU (int32)

    @property
    def num_layers(self) -> int:
        return int(self.activations.shape[0])

    @property
    def num_tokens(self) -> int:
        return int(self.activations.shape[1]) if self.activations.ndim >= 2 else 0


@dataclass
class PairActivations:
    """Holds aligned activations for a model pair."""

    model_1: ActivationBundle
    model_2: ActivationBundle
    token_counts: torch.Tensor

    @property
    def pair(self) -> Tuple[str, str]:
        return (self.model_1.model_name, self.model_2.model_name)

    @property
    def sanitized_names(self) -> Tuple[str, str]:
        return (
            sanitize_model_name(self.model_1.model_name),
            sanitize_model_name(self.model_2.model_name),
        )


class BaseEvaluationMetric(ABC):
    """
    Abstract base class for evaluation metrics that compare two models.
    
    Provides shared utilities for activation collection, output directory management,
    and configuration saving. Subclasses must implement the compute() method.
    """

    def __init__(
        self,
        model_1: str,
        model_2: str,
        dataset_config: DatasetConfig,
        device: str,
        dtype: torch.dtype,
        batch_size: int,
        output_base_dir: Path,
        is_causal_attn: bool = False,
    ):
        """
        Initialize the base evaluation metric.
        
        Args:
            model_1: Path or name of the first model
            model_2: Path or name of the second model
            dataset_config: Dataset configuration
            device: Device to run model inference on
            dtype: Torch dtype for computation
            batch_size: Batch size for activation collection
            output_base_dir: Base directory for saving results
            is_causal_attn: Whether to use causal attention mask
        """
        self.model_1 = model_1
        self.model_2 = model_2
        self.dataset_config = dataset_config
        self.device = device
        self.dtype = dtype
        self.batch_size = batch_size
        self.output_base_dir = Path(output_base_dir)
        self.is_causal_attn = is_causal_attn

    @property
    def model_1_name(self) -> str:
        """Return the first model name."""
        return self.model_1

    @property
    def model_2_name(self) -> str:
        """Return the second model name."""
        return self.model_2

    @property
    def sanitized_names(self) -> Tuple[str, str]:
        """Return sanitized model names for file paths."""
        return (
            sanitize_model_name(self.model_1),
            sanitize_model_name(self.model_2),
        )

    @property
    @abstractmethod
    def metric_name(self) -> str:
        """Return the name of the metric (e.g., 'cka_evaluation')."""
        pass

    @abstractmethod
    def compute(self):
        """
        Compute the evaluation metric.
        
        This method must be implemented by subclasses to perform the actual
        metric computation. The return type varies by metric.
        """
        pass

    def collect_model_activations(
        self,
        model_name: str,
        sentences: Sequence[str],
    ) -> ActivationBundle:
        """
        Collect full-layer activations and per-sentence token counts for a model.
        
        Args:
            model_name: Model name or path
            sentences: List of input sentences
            
        Returns:
            ActivationBundle containing activations and token counts
        """
        print(f"\n{Fore.YELLOW}Collecting activations for {model_name}{Style.RESET_ALL}")
        f_model, model, tokenizer = get_model(
            model_name_or_path=model_name,
            device=self.device,
            is_causal_attn=self.is_causal_attn,
        )

        per_sentence_counts: List[int] = []
        per_layer_storage: List[List[torch.Tensor]] = []
        hidden_dim: Optional[int] = None
        num_layers: Optional[int] = None

        total_sentences = len(sentences)
        num_batches = (total_sentences + self.batch_size - 1) // self.batch_size

        for batch_idx in trange(num_batches, desc="Collecting activations"):
            start = batch_idx * self.batch_size
            end = min((batch_idx + 1) * self.batch_size, total_sentences)
            batch = sentences[start:end]

            hidden_states, attention_mask = f_model(batch)

            if num_layers is None:
                num_layers = len(hidden_states)
                per_layer_storage = [[] for _ in range(num_layers)]

            attn_mask = attention_mask.to(hidden_states[0].device)
            row_counts_t = attn_mask.sum(dim=1).to(dtype=torch.int32).detach().cpu()
            per_sentence_counts.extend(row_counts_t.tolist())

            for layer_idx, layer_outputs in enumerate(hidden_states):
                hidden_dim = int(layer_outputs.shape[-1])
                for row in range(layer_outputs.shape[0]):
                    count = int(row_counts_t[row].item())
                    if count <= 0:
                        continue
                    token_vectors = layer_outputs[row][attn_mask[row].to(torch.bool)]
                    per_layer_storage[layer_idx].append(
                        token_vectors.detach().to("cpu", dtype=self.dtype)
                    )

            del hidden_states, attention_mask, attn_mask
            clear_memory()

        model.to("cpu")
        del model, tokenizer
        clear_memory()

        if num_layers is None or hidden_dim is None:
            raise RuntimeError(f"No activations collected for {model_name}.")

        layer_arrays: List[torch.Tensor] = []
        for storage in per_layer_storage:
            if storage:
                layer_arrays.append(torch.cat(storage, dim=0))
            else:
                layer_arrays.append(torch.empty((0, hidden_dim), dtype=self.dtype))

        activations = torch.stack(layer_arrays, dim=0).to("cpu")
        token_counts = torch.tensor(per_sentence_counts, dtype=torch.int32).to("cpu")

        del layer_arrays, per_layer_storage
        clear_memory()

        print(
            f"{Fore.CYAN}{model_name}: layers={activations.shape[0]}, "
            f"tokens={activations.shape[1]}, hidden={activations.shape[2]}{Style.RESET_ALL}\n"
        )

        return ActivationBundle(
            model_name=model_name, activations=activations, token_counts=token_counts
        )

    def collect_pair_activations(
        self,
        sentences: Sequence[str],
    ) -> PairActivations:
        """
        Compute and align activations for a pair of models.
        
        Args:
            sentences: List of input sentences
            
        Returns:
            PairActivations containing aligned activations for both models
            
        Raises:
            ValueError: If token counts don't match between models
        """
        bundle1 = self.collect_model_activations(self.model_1, sentences)
        bundle2 = self.collect_model_activations(self.model_2, sentences)

        if bundle1.num_layers != bundle2.num_layers:
            print(
                f"{Fore.MAGENTA}Layer count mismatch ({bundle1.num_layers} vs "
                f"{bundle2.num_layers}); metrics will use min layers downstream."
                f"{Style.RESET_ALL}"
            )

        if not torch.equal(bundle1.token_counts, bundle2.token_counts):
            raise ValueError(
                f"Token count mismatch between {self.model_1} and {self.model_2}; "
                "cannot align activations."
            )

        return PairActivations(
            model_1=bundle1,
            model_2=bundle2,
            token_counts=bundle1.token_counts.clone(),
        )

    def get_output_dir(self) -> Path:
        """
        Get the output directory for this metric.
        
        Returns:
            Path to the output directory
        """
        model_1_safe, model_2_safe = self.sanitized_names
        dataset_slug = self.dataset_config.slug()
        output_dir = (
            self.output_base_dir
            / self.metric_name
            / f"{model_1_safe}_vs_{model_2_safe}"
            / dataset_slug
        )
        output_dir.mkdir(parents=True, exist_ok=True)
        return output_dir

    def save_config(self, output_dir: Path, additional_config: Optional[dict] = None):
        """
        Save configuration to JSON file.
        
        Args:
            output_dir: Directory to save configuration
            additional_config: Optional additional configuration to include
        """
        config_payload = {
            "model_1": self.model_1,
            "model_2": self.model_2,
            "dataset": self.dataset_config.name,
            "text_column": self.dataset_config.text_column,
            "dataset_subset": self.dataset_config.subset,
            "dataset_split": self.dataset_config.split,
            "num_sentences": self.dataset_config.num_sentences,
            "device": self.device,
            "dtype": str(self.dtype),
            "batch_size": self.batch_size,
        }
        
        if self.dataset_config.label_column:
            config_payload["label_column"] = self.dataset_config.label_column
        
        if additional_config:
            config_payload.update(additional_config)

        config_path = output_dir / "config.json"
        with open(config_path, "w") as handle:
            json.dump(config_payload, handle, indent=4)
