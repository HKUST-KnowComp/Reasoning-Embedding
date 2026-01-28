"""
Procrustes alignment metric for comparing model representations.
"""
from __future__ import annotations

from typing import Dict

import torch
from colorama import Fore, Style

from hrsa.utils import load_sentences_and_labels
from hrsa.metrics.procrustes import (
    SplitResult,
    compute_row_entropy,
    evaluate_split,
    fit_global_mapping,
    fit_layerwise_mappings,
    plot_o_matrices,
    plot_procrustes_metrics,
    prepare_output_dirs as prepare_procrustes_output_dirs,
    write_run_insights,
    write_split_tables,
)
from hrsa.base import BaseEvaluationMetric


class OrthogonalProcrustesMetric(BaseEvaluationMetric):
    """
    Compute Procrustes alignment metrics between two models' representations.
    
    Procrustes alignment finds orthogonal transformations that optimally align
    representation spaces. This metric computes both layer-wise and global
    alignment matrices and evaluates their quality.
    """

    @property
    def metric_name(self) -> str:
        return "orthogonal_procrustes"

    def compute(self) -> Dict[str, torch.Tensor]:
        """
        Compute Procrustes alignment metrics.
        
        Returns:
            Dictionary containing alignment metrics for each layer
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
            f"{Fore.CYAN}Loaded {len(sentences)} sentences for Procrustes computation."
            f"{Style.RESET_ALL}"
        )

        # Collect activations for both models
        pair_activations = self.collect_pair_activations(sentences)

        # Prepare output directories
        model_1_safe, model_2_safe = self.sanitized_names
        output_root = prepare_procrustes_output_dirs(
            self.output_base_dir / self.metric_name,
            model_1_safe,
            model_2_safe,
        )

        # Create split result
        calibration_result = SplitResult(
            config=self.dataset_config,
            activations_model1=pair_activations.model_1.activations,
            activations_model2=pair_activations.model_2.activations,
        )

        # Compute alignments
        print(f"{Fore.YELLOW}Computing Procrustes alignments...{Style.RESET_ALL}")
        computation_device = torch.device("cpu")  # Metrics computed on CPU
        
        layer_mappings = fit_layerwise_mappings(
            calibration_result.activations_model1,
            calibration_result.activations_model2,
            device=computation_device,
            dtype=self.dtype,
        )
        
        global_mapping = fit_global_mapping(
            calibration_result.activations_model1,
            calibration_result.activations_model2,
            device=computation_device,
            dtype=self.dtype,
        )

        # Prepare output directories
        split_dir = output_root / f"{self.dataset_config.label}_{self.dataset_config.slug()}"
        plots_dir = split_dir / "plots"
        tables_dir = split_dir / "tables"
        plots_dir.mkdir(parents=True, exist_ok=True)
        tables_dir.mkdir(parents=True, exist_ok=True)

        # Calculate row entropy and sparsity for the global O matrix
        absolute_global_o = global_mapping.detach().abs().to("cpu")
        hidden_size = int(absolute_global_o.shape[0])
        global_o_sparsity = float(
            (absolute_global_o < 1e-5).to(torch.float32).mean().item()
        )
        row_entropy_stats = compute_row_entropy(global_mapping)
        mean_row_inverse_entropy = row_entropy_stats["mean_row_inverse_entropy"]

        # Plot O matrices
        plot_o_matrices(
            layer_mappings=layer_mappings,
            global_mapping=global_mapping,
            calibration_config=self.dataset_config,
            output_root=output_root,
            mean_row_inverse_entropy=mean_row_inverse_entropy,
            global_o_sparsity=global_o_sparsity,
            hidden_size=hidden_size,
        )

        # Evaluate split
        metrics = evaluate_split(
            calibration_result,
            layer_mappings=layer_mappings,
            global_mapping=global_mapping,
            device=computation_device,
            dtype=self.dtype,
        )

        # Save plots and tables
        plot_procrustes_metrics(self.dataset_config.label, metrics, plots_dir)
        write_split_tables(calibration_result, metrics, tables_dir)
        write_run_insights(
            split_dir,
            [
                {
                    "label": self.dataset_config.label,
                    "dataset": calibration_result.config.name,
                    "metrics": metrics,
                }
            ],
            row_entropy_stats,
            global_o_sparsity,
            hidden_size,
        )

        # Save configuration
        additional_config = {
            "hidden_size": hidden_size,
            "row_entropy_stats": row_entropy_stats,
            "global_o_sparsity": global_o_sparsity,
        }
        self.save_config(split_dir, additional_config)

        print(f"{Fore.GREEN}Saved Procrustes metrics to {output_root}{Style.RESET_ALL}\n")

        return metrics
