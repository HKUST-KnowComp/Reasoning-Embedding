"""
K-Nearest Neighbor overlap metric for comparing model representations.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional

import torch
from colorama import Fore, Style
from tqdm import trange

from hrsa.utils import load_sentences_and_labels
from hrsa.metrics.knn import (
    aggregate_label_stats,
    build_neighbor_graph,
    compute_jaccard_scores,
    expand_token_labels,
    maybe_subsample_tokens,
    plot_layerwise_curves,
    save_layerwise_outputs,
    save_single_layer_outputs,
)
from hrsa.base import BaseEvaluationMetric
from hrsa.config import DatasetConfig


class KNNNeighborOverlapMetric(BaseEvaluationMetric):
    """
    Compute k-nearest neighbor overlap between two models' representations.
    
    This metric measures the similarity of representation spaces by comparing
    the k-nearest neighbors of each token in both models' embedding spaces.
    High overlap indicates similar local structure.
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
        neighbor_ks: List[int] = None,
        neighbor_metric: str = "cosine",
        max_tokens: int = 50000,
        random_seed: int = 42,
        is_causal_attn: bool = False,
    ):
        """
        Initialize KNN overlap metric with additional parameters.
        
        Args:
            model_1: Path or name of the first model
            model_2: Path or name of the second model
            dataset_config: Dataset configuration
            device: Device to run model inference on
            dtype: Torch dtype for computation
            batch_size: Batch size for activation collection
            output_base_dir: Base directory for saving results
            neighbor_ks: List of k values for k-NN (default: [5, 10, 50])
            neighbor_metric: Distance metric for k-NN (default: "cosine")
            max_tokens: Maximum tokens to use (default: 50000)
            random_seed: Random seed for subsampling (default: 42)
            is_causal_attn: Whether to use causal attention mask
        """
        super().__init__(
            model_1=model_1,
            model_2=model_2,
            dataset_config=dataset_config,
            device=device,
            dtype=dtype,
            batch_size=batch_size,
            output_base_dir=output_base_dir,
            is_causal_attn=is_causal_attn,
        )
        self.neighbor_ks = neighbor_ks if neighbor_ks is not None else [5, 10, 50]
        self.neighbor_metric = neighbor_metric
        self.max_tokens = max_tokens
        self.random_seed = random_seed

    @property
    def metric_name(self) -> str:
        return "knn_overlap"

    def compute(self) -> List[Dict[str, object]]:
        """
        Compute k-NN overlap metrics across layers.
        
        Returns:
            List of dictionaries containing overlap results for each layer
        """
        # Load sentences and labels
        sentences, sentence_labels = load_sentences_and_labels(
            dataset_name_or_path=self.dataset_config.name,
            text_column=self.dataset_config.text_column,
            label_column=self.dataset_config.label_column,
            subset=self.dataset_config.subset,
            split=self.dataset_config.split,
            num_sentences=self.dataset_config.num_sentences,
        )
        print(
            f"{Fore.CYAN}Loaded {len(sentences)} sentences for k-NN overlap computation."
            f"{Style.RESET_ALL}"
        )

        # Collect activations for both models
        pair_activations = self.collect_pair_activations(sentences)

        # Prepare output directories
        model_1_safe, model_2_safe = self.sanitized_names
        output_root = (
            self.output_base_dir
            / self.metric_name
            / f"{model_1_safe}_vs_{model_2_safe}"
        )
        output_root.mkdir(parents=True, exist_ok=True)

        # Expand sentence labels to token labels
        token_labels = (
            expand_token_labels(sentence_labels, pair_activations.token_counts)
            if sentence_labels is not None
            else None
        )

        # Get activations
        emb1 = pair_activations.model_1.activations
        emb2 = pair_activations.model_2.activations

        # Handle layer mismatch
        min_layers = min(emb1.shape[0], emb2.shape[0])
        if emb1.shape[0] != emb2.shape[0]:
            print(
                f"{Fore.MAGENTA}Warning: layer mismatch ({emb1.shape[0]} vs "
                f"{emb2.shape[0]}). Using first {min_layers} layers.{Style.RESET_ALL}"
            )
        emb1 = emb1[:min_layers]
        emb2 = emb2[:min_layers]

        # Subsample tokens if necessary
        emb1, emb2, token_labels = maybe_subsample_tokens(
            emb1,
            emb2,
            token_labels,
            max_tokens=self.max_tokens,
            random_seed=self.random_seed,
        )

        # Validate token count
        total_tokens = emb1.shape[1]
        if total_tokens < 2:
            raise ValueError("Need at least two tokens to compute neighbor overlap.")

        # Filter valid k values
        max_possible_k = total_tokens - 1
        valid_neighbor_ks: List[int] = []
        for k in self.neighbor_ks:
            if k <= 0:
                continue
            if k > max_possible_k:
                print(
                    f"{Fore.RED}Skipping k={k} (only {total_tokens} tokens, "
                    f"max {max_possible_k}).{Style.RESET_ALL}"
                )
                continue
            valid_neighbor_ks.append(k)

        if not valid_neighbor_ks:
            raise ValueError("No valid neighbor sizes for k-NN overlap computation.")

        max_k = max(valid_neighbor_ks)
        token_labels_array = token_labels

        # Print configuration
        print(f"{Fore.YELLOW}Computing k-NN overlap{Style.RESET_ALL}")
        print(f"{Fore.CYAN}    Using {min_layers} layers{Style.RESET_ALL}")
        print(f"{Fore.CYAN}    Using {max_k} as max k{Style.RESET_ALL}")
        print(f"{Fore.CYAN}    Using {self.neighbor_metric} as metric{Style.RESET_ALL}")
        print(
            f"{Fore.CYAN}    Using {valid_neighbor_ks} as valid neighbor ks{Style.RESET_ALL}"
        )
        print(f"{Fore.CYAN}    Using {emb1.shape[0]} layers for model 1{Style.RESET_ALL}")
        print(f"{Fore.CYAN}    Using {emb2.shape[0]} layers for model 2{Style.RESET_ALL}")
        print(
            f"{Fore.CYAN}    Using {pair_activations.token_counts.sum()} total token counts"
            f"{Style.RESET_ALL}"
        )

        # Compute k-NN overlap for each layer
        layer_results: List[Dict[str, object]] = []
        for layer_position in trange(min_layers, desc="Computing k-NN overlap"):
            layer_emb1 = emb1[layer_position]
            layer_emb2 = emb2[layer_position]
            neighbors_model1 = build_neighbor_graph(
                layer_emb1, max_k, metric=self.neighbor_metric
            )
            neighbors_model2 = build_neighbor_graph(
                layer_emb2, max_k, metric=self.neighbor_metric
            )
            overlap_results = compute_jaccard_scores(
                neighbors_model1, neighbors_model2, valid_neighbor_ks
            )

            per_layer_label_stats: Dict[int, Optional[List[Dict[str, object]]]] = {}
            for k in valid_neighbor_ks:
                per_token = overlap_results[k]["per_token"]
                per_layer_label_stats[k] = aggregate_label_stats(
                    per_token, token_labels_array, k
                )
                del overlap_results[k]["per_token"]

            layer_results.append(
                {
                    "position": layer_position,
                    "model_1_layer_index": layer_position,
                    "model_2_layer_index": layer_position,
                    "results": {
                        str(k): {
                            "stats": overlap_results[k]["stats"],
                            "label_stats": per_layer_label_stats[k],
                        }
                        for k in valid_neighbor_ks
                    },
                }
            )

        # Prepare output directories
        split_dir = output_root / self.dataset_config.slug()
        split_dir.mkdir(parents=True, exist_ok=True)
        plots_dir = split_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        # Save results
        if min_layers == 1:
            single_layer_entry = layer_results[0]
            save_single_layer_outputs(
                split_dir=split_dir,
                dataset_config=self.dataset_config,
                ks=valid_neighbor_ks,
                overlap_results={
                    int(k): single_layer_entry["results"][str(k)]["stats"]  # type: ignore[index]
                    for k in valid_neighbor_ks
                },
                label_stats={
                    int(k): single_layer_entry["results"][str(k)]["label_stats"]  # type: ignore[index]
                    for k in valid_neighbor_ks
                },
            )
        else:
            plot_path = plots_dir / "layerwise_knn_overlap.png"
            plot_layerwise_curves(
                layer_results=layer_results, ks=valid_neighbor_ks, output_path=plot_path
            )
            save_layerwise_outputs(
                split_dir=split_dir,
                dataset_config=self.dataset_config,
                ks=valid_neighbor_ks,
                layer_results=layer_results,
                plot_path=plot_path,
            )

        # Save configuration
        additional_config = {
            "neighbor_ks": valid_neighbor_ks,
            "neighbor_metric": self.neighbor_metric,
            "max_tokens": self.max_tokens,
            "random_seed": self.random_seed,
        }
        self.save_config(split_dir, additional_config)

        print(f"{Fore.GREEN}Saved k-NN overlap metrics to {split_dir}{Style.RESET_ALL}\n")

        return layer_results
