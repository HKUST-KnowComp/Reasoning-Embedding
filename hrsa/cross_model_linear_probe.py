"""
Cross-model linear probe metric for evaluating representation transfer.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from colorama import Fore, Style

import torch

from hrsa.metrics.linear_probe import (
    FeatureProjector,
    LinearProbeTrainer,
    RepresentationExtractor,
    TaskDatasetPreparer,
    plot_primary_metric,
    write_metrics_outputs,
)
from hrsa.base import BaseEvaluationMetric
from hrsa.config import ProbeDatasetConfig
from hrsa.utils import sanitize_model_name


class CrossModelLinearProbeMetric(BaseEvaluationMetric):
    """
    Train a linear probe on one model's representations and evaluate transfer to another.
    
    This metric measures how well representations from a source model can be used
    to train classifiers/regressors that transfer to a target model, providing
    insights into functional similarity between models.
    """

    def __init__(
        self,
        model_1: str,
        model_2: str,
        probe_config: ProbeDatasetConfig,
        device: str,
        dtype: torch.dtype,
        batch_size: int,
        output_base_dir: Path,
        source_layer: int = -1,
        target_layer: int = -1,
        source_pooling: str = "mean",
        target_pooling: str = "mean",
        model_type: str = "sentence_transformer",
        projector: str = "none",
        shared_dim: Optional[int] = None,
        max_length: int = 32000,
        normalize_embeddings: bool = True,
        max_iter: int = 1000,
        c_value: float = 1.0,
        penalty: str = "l2",
        solver: str = "lbfgs",
        ridge_alpha: float = 1.0,
        n_jobs: int = -1,
        seed: int = 42,
        primary_metric: Optional[str] = None,
        skip_plot: bool = False,
    ):
        """Initialize cross-model linear probe metric."""
        super().__init__(
            model_1=model_1,
            model_2=model_2,
            dataset_config=probe_config,
            device=device,
            dtype=dtype,
            batch_size=batch_size,
            output_base_dir=output_base_dir,
        )

        # Probe-specific parameters
        self.source_layer = source_layer
        self.target_layer = target_layer
        self.source_pooling = source_pooling
        self.target_pooling = target_pooling
        self.model_type = model_type
        self.projector = projector
        self.shared_dim = shared_dim
        self.max_length = max_length
        self.normalize_embeddings = normalize_embeddings
        
        # Training parameters
        self.max_iter = max_iter
        self.c_value = c_value
        self.penalty = penalty
        self.solver = solver
        self.ridge_alpha = ridge_alpha
        self.n_jobs = n_jobs
        self.seed = seed
        
        # Output parameters
        self.primary_metric = primary_metric
        self.skip_plot = skip_plot

    @property
    def model_1_name(self) -> str:
        """Return the source model name."""
        return self.model_1

    @property
    def model_2_name(self) -> str:
        """Return the target model name."""
        return self.model_2

    @property
    def sanitized_names(self) -> Tuple[str, str]:
        """Return sanitized model names for file paths."""
        return (
            sanitize_model_name(self.model_1),
            sanitize_model_name(self.model_2),
        )

    @property
    def metric_name(self) -> str:
        return "cross_model_linear_probe"


    def save_config(self, output_dir: Path, additional_config: Optional[dict] = None):
        """Save linear probe configuration to JSON file."""
        config_payload = {
            "model_1": self.model_1,
            "model_2": self.model_2,
            "dataset": self.dataset_config.name,
            "dataset_config": self.dataset_config.dataset_config,
            "text_column": self.dataset_config.text_column,
            "label_column": self.dataset_config.label_column,
            "task_type": self.dataset_config.task_type,
            "train_fraction": self.dataset_config.train_fraction,
            "val_fraction": self.dataset_config.val_fraction,
            "test_fraction": self.dataset_config.test_fraction,
            "max_samples": self.dataset_config.max_samples,
            "min_class_samples": self.dataset_config.min_class_samples,
            "seed": self.dataset_config.seed,
            "device": self.device,
            "batch_size": self.batch_size,
            "model_type": self.model_type,
            "source_layer": self.source_layer,
            "target_layer": self.target_layer,
            "source_pooling": self.source_pooling,
            "target_pooling": self.target_pooling,
            "projector": self.projector,
            "shared_dim": self.shared_dim,
            "max_length": self.max_length,
            "normalize_embeddings": self.normalize_embeddings,
            "max_iter": self.max_iter,
            "c_value": self.c_value,
            "penalty": self.penalty,
            "solver": self.solver,
            "ridge_alpha": self.ridge_alpha,
            "n_jobs": self.n_jobs,
            "primary_metric": self.primary_metric,
            "skip_plot": self.skip_plot,
        }

        if additional_config:
            config_payload.update(additional_config)

        config_path = output_dir / "config.json"
        with config_path.open("w", encoding="utf-8") as handle:
            json.dump(config_payload, handle, indent=4)

    def compute(self) -> Dict[str, Dict[str, float]]:
        """Train probe on source model and evaluate transfer to target model."""
        # Initialize containers
        splits: Dict[str, pd.DataFrame] = {}
        split_texts: Dict[str, List[str]] = {}
        split_targets: Dict[str, np.ndarray] = {}
        source_features: Dict[str, np.ndarray] = {}
        target_features: Dict[str, np.ndarray] = {}

        # Prepare dataset with train/val/test splits
        print(f"{Fore.YELLOW}Preparing dataset splits...{Style.RESET_ALL}")
        dataset_manager = TaskDatasetPreparer(
            dataset_name=self.dataset_config.name,
            dataset_config=self.dataset_config.dataset_config,
            dataset_split="train",
            task_type=self.dataset_config.task_type,
            text_column=self.dataset_config.text_column,
            label_column=self.dataset_config.label_column,
            task_name=self.dataset_config.label,
            cache_root=str(self.output_base_dir / "cache"),
            train_fraction=self.dataset_config.train_fraction,
            val_fraction=self.dataset_config.val_fraction,
            test_fraction=self.dataset_config.test_fraction,
            max_samples=self.dataset_config.max_samples,
            min_class_samples=self.dataset_config.min_class_samples,
            seed=self.dataset_config.seed,
            force_refresh=False,
        )
        splits = dataset_manager.prepare()

        text_column = dataset_manager.text_column
        target_column = dataset_manager.target_column

        split_texts = {
            split: df[text_column].astype(str).tolist() for split, df in splits.items()
        }
        split_targets = {split: df[target_column].to_numpy() for split, df in splits.items()}

        # Extract features from source model
        print(f"{Fore.YELLOW}Extracting features from source model...{Style.RESET_ALL}")
        source_extractor = RepresentationExtractor(
            model_name=self.model_1,
            model_type=self.model_type,
            device=self.device,
            layer=self.source_layer,
            pooling=self.source_pooling,
            batch_size=self.batch_size,
            max_length=self.max_length,
            normalize=self.normalize_embeddings,
            dtype=None,
        )

        # Extract features from target model
        print(f"{Fore.YELLOW}Extracting features from target model...{Style.RESET_ALL}")
        target_extractor = RepresentationExtractor(
            model_name=self.model_2,
            model_type=self.model_type,
            device=self.device,
            layer=self.target_layer,
            pooling=self.target_pooling,
            batch_size=self.batch_size,
            max_length=self.max_length,
            normalize=self.normalize_embeddings,
            dtype=None,
        )

        for split_name in splits.keys():
            texts = split_texts[split_name]
            source_features[split_name] = source_extractor.encode(
                texts, desc=f"{split_name}::source"
            )
            target_features[split_name] = target_extractor.encode(
                texts, desc=f"{split_name}::target"
            )

        # Apply optional projection
        print(f"{Fore.YELLOW}Applying feature projection...{Style.RESET_ALL}")
        projector_obj = FeatureProjector(
            method=self.projector,
            shared_dim=self.shared_dim,
            seed=self.seed,
        )
        projector_obj.fit(
            {
                "source_train": source_features["train"],
                "target_train": target_features["train"],
            }
        )
        for split_name in splits.keys():
            source_features[split_name] = projector_obj.transform(source_features[split_name])
            target_features[split_name] = projector_obj.transform(target_features[split_name])

        # Train linear probe on source features
        print(f"{Fore.YELLOW}Training linear probe on source features...{Style.RESET_ALL}")
        trainer = LinearProbeTrainer(
            task_type=self.dataset_config.task_type,
            max_iter=self.max_iter,
            c_value=self.c_value,
            penalty=self.penalty,
            solver=self.solver,
            ridge_alpha=self.ridge_alpha,
            n_jobs=self.n_jobs,
            seed=self.seed,
        )
        trainer.label_mapping = dataset_manager.id_to_label
        trainer.fit(source_features["train"], split_targets["train"])
        print(f"{Fore.GREEN}Fitted linear probe{Style.RESET_ALL}")

        # Evaluate on all splits for both models
        print(f"{Fore.YELLOW}Evaluating probe on all splits...{Style.RESET_ALL}")
        metrics: Dict[str, Dict[str, float]] = {}
        for split_name in splits.keys():
            metrics[f"source_{split_name}"] = trainer.evaluate(
                source_features[split_name], split_targets[split_name]
            )
            metrics[f"target_{split_name}"] = trainer.evaluate(
                target_features[split_name], split_targets[split_name]
            )

        # Prepare output directory
        output_dir = self.get_output_dir()
        # Save models and results
        self.save_config(output_dir)
        trainer.save(output_dir / "linear_probe.joblib")
        projector_obj.save(output_dir / "feature_projector.joblib")
        task_slug = dataset_manager.task_slug
        write_metrics_outputs(
            metrics,
            dataset_manager.metadata,
            output_dir / f"{task_slug}.json",
            output_dir / f"{task_slug}.tsv",
        )

        # Plot primary metric
        primary_metric = self.primary_metric
        if primary_metric is None:
            primary_metric = (
                "accuracy" if self.dataset_config.task_type == "classification" else "mae"
            )
        if not self.skip_plot:
            plot_path = output_dir / f"{task_slug}_{primary_metric}.png"
            try:
                plot_primary_metric(metrics, primary_metric, plot_path)
            except Exception as exc:
                print(f"{Fore.YELLOW}Could not create plot: {exc}{Style.RESET_ALL}")

        print(
            f"{Fore.GREEN}Saved probe to {output_dir} and metrics to {output_dir / f'{task_slug}.json'}"
            f"{Style.RESET_ALL}\n"
        )

        return metrics
