"""
Dimension-wise correlation metric for comparing model representations.
"""
from __future__ import annotations

import json

import torch
from colorama import Fore, Style

from hrsa.utils import load_sentences_and_labels
from hrsa.metrics.correlation import (
    SplitActivations,
    compute_dimension_correlations,
    compute_layerwise_correlations,
    plot_correlation_diagonal,
    plot_correlation_heatmap,
    plot_correlation_histogram,
    prepare_output_dirs as prepare_correlation_output_dirs,
    summarize_correlations,
    write_correlation_tables,
    write_layerwise_statistics,
    write_run_summary,
)
from hrsa.base import BaseEvaluationMetric


class DimensionWiseCorrelationMetric(BaseEvaluationMetric):
    """
    Compute dimension-wise correlations between two models' representations.
    
    This metric computes Pearson correlations between corresponding dimensions
    of two models' hidden representations, providing insights into how similar
    the feature spaces are at a fine-grained level.
    """

    @property
    def metric_name(self) -> str:
        return "dimension_wise_correlation"

    def compute(self) -> torch.Tensor:
        """
        Compute layerwise correlation matrix.
        
        Returns:
            Correlation matrix of shape (num_layers, num_layers)
        """
        # Load sentences
        sentences, _ = load_sentences_and_labels(
            dataset_name_or_path=self.dataset_config.name,
            text_column=self.dataset_config.text_column,
            label_column=self.dataset_config.label_column,
            subset=self.dataset_config.subset,
            split=self.dataset_config.split,
            num_sentences=self.dataset_config.num_sentences,
        )
        print(
            f"{Fore.CYAN}Loaded {len(sentences)} sentences for correlation computation."
            f"{Style.RESET_ALL}"
        )

        # Collect activations for both models
        pair_activations = self.collect_pair_activations(sentences)

        # Prepare output directories
        model_1_safe, model_2_safe = self.sanitized_names
        output_root = prepare_correlation_output_dirs(
            self.output_base_dir / self.metric_name,
            model_1_safe,
            model_2_safe,
        )

        split_dir = output_root / self.dataset_config.slug()
        plots_dir = split_dir / "plots"
        tables_dir = split_dir / "tables"
        plots_dir.mkdir(parents=True, exist_ok=True)
        tables_dir.mkdir(parents=True, exist_ok=True)

        # Compute layerwise correlations (all layer pairs)
        print(f"{Fore.YELLOW}Computing layerwise correlations...{Style.RESET_ALL}")
        activations1 = pair_activations.model_1.activations
        activations2 = pair_activations.model_2.activations

        correlation_matrix, all_correlations = compute_layerwise_correlations(
            activations1, activations2
        )

        # Plot heatmap
        heatmap_path = plots_dir / f"corr_heatmap_{model_1_safe}_vs_{model_2_safe}.png"
        plot_correlation_heatmap(
            correlation_matrix, heatmap_path, model_1_safe, model_2_safe
        )

        # Plot diagonal
        diagonal_path = plots_dir / f"corr_diagonal_{model_1_safe}_vs_{model_2_safe}.png"
        plot_correlation_diagonal(
            correlation_matrix, diagonal_path, model_1_safe, model_2_safe
        )

        # Write statistics
        write_layerwise_statistics(
            correlation_matrix, tables_dir, model_1_safe, model_2_safe
        )

        # Save correlation matrix (torch)
        torch.save(correlation_matrix, tables_dir / "correlation_matrix.pt")

        # Compute and save summary for the diagonal (matching layers)
        diagonal_correlations = torch.diagonal(correlation_matrix)
        diagonal_summary = {
            "mean": float(diagonal_correlations.mean().item()),
            "median": float(diagonal_correlations.median().item()),
            "std": float(diagonal_correlations.std(unbiased=True).item())
            if diagonal_correlations.numel() > 1
            else 0.0,
            "min": float(diagonal_correlations.min().item()),
            "max": float(diagonal_correlations.max().item()),
            "num_layers": int(correlation_matrix.shape[0]),
        }

        with open(tables_dir / "layerwise_summary.json", "w") as handle:
            json.dump(
                {
                    "diagonal_summary": diagonal_summary,
                    "overall_matrix_mean": float(correlation_matrix.mean()),
                    "overall_matrix_std": float(correlation_matrix.std()),
                },
                handle,
                indent=4,
            )

        # Also save a histogram of the last layer correlations
        last_layer_idx = -1
        last_layer_act1 = activations1[last_layer_idx]
        last_layer_act2 = activations2[last_layer_idx]
        last_layer_correlations = compute_dimension_correlations(
            last_layer_act1, last_layer_act2
        )
        last_layer_summary = summarize_correlations(last_layer_correlations)

        split_result = SplitActivations(
            config=self.dataset_config,
            activations_model1=last_layer_act1,
            activations_model2=last_layer_act2,
        )

        histogram_path = (
            plots_dir / f"{self.dataset_config.slug()}_last_layer_correlations_hist.png"
        )
        plot_correlation_histogram(
            correlations=last_layer_correlations,
            output_path=histogram_path,
            bins=40,
            dataset_label=self.dataset_config.slug(),
        )
        write_correlation_tables(
            split_result,
            last_layer_summary,
            last_layer_correlations,
            tables_dir,
            save_full_vector=False,
        )
        write_run_summary(
            split_dir, self.dataset_config.label, self.dataset_config.name, last_layer_summary
        )

        # Save configuration
        additional_config = {
            "layerwise": True,
            "num_layers": int(correlation_matrix.shape[0]),
        }
        self.save_config(split_dir, additional_config)

        print(f"{Fore.GREEN}Saved Correlation metrics to {split_dir}{Style.RESET_ALL}\n")

        return correlation_matrix
