"""
Linear CKA (Centered Kernel Alignment) metric for comparing model representations.
"""
from __future__ import annotations

import torch
from colorama import Fore, Style

from hrsa.metrics.cka import compute_layerwise_cka, plot_cka, write_statistics
from hrsa.utils import load_sentences_and_labels
from hrsa.base import BaseEvaluationMetric


class LinearCKAMetric(BaseEvaluationMetric):
    """
    Compute Linear CKA similarity between two models' layer representations.
    
    CKA measures the similarity between representation spaces by comparing
    their centered Gram matrices. This implementation uses the efficient
    linear version that operates directly on features rather than forming
    full Gram matrices.
    """

    @property
    def metric_name(self) -> str:
        return "linear_cka"

    def compute(self) -> torch.Tensor:
        """
        Compute layerwise Linear CKA similarity matrix.
        
        Returns:
            CKA matrix of shape (num_layers, num_layers) as a CPU torch.FloatTensor
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
            f"{Fore.CYAN}Loaded {len(sentences)} sentences for CKA computation.{Style.RESET_ALL}"
        )

        # Collect activations for both models
        pair_activations = self.collect_pair_activations(sentences)

        # Get output directory
        output_dir = self.get_output_dir()

        # Compute layerwise CKA
        print(f"{Fore.YELLOW}Computing Linear CKA...{Style.RESET_ALL}")
        cka_matrix = compute_layerwise_cka(
            pair_activations.model_1.activations,
            pair_activations.model_2.activations,
            use_float64=self.dtype == torch.float64,
            dtype=self.dtype,
        )

        # Plot and save results
        model_1_safe, model_2_safe = self.sanitized_names
        heatmap_path, _ = plot_cka(cka_matrix, output_dir, model_1_safe, model_2_safe)
        write_statistics(cka_matrix, heatmap_path)

        # Save configuration
        self.save_config(output_dir)

        print(f"{Fore.GREEN}Saved CKA metrics to {output_dir}{Style.RESET_ALL}\n")

        return cka_matrix
