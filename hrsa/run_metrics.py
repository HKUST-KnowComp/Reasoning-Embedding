#!/usr/bin/env python3
"""
Unified CLI for running evaluation metrics on model pairs.

This script provides a clean interface for computing various similarity metrics
between two models using a modular, object-oriented architecture.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import torch
from colorama import Fore, Style, init as color_init

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from hrsa import METRIC_REGISTRY
from hrsa.config import DatasetConfig, ProbeDatasetConfig, METRIC_RESULTS_FOLDER
from hrsa.utils import parse_compute_dtype, DTYPE_ALIASES

color_init(autoreset=True)

def parse_neighbor_ks(ks_str: str) -> list[int]:
    """Parse comma-separated list of k values."""
    return [int(k.strip()) for k in ks_str.split(",")]


def main():
    parser = argparse.ArgumentParser(
        description="Run evaluation metrics on model pairs",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run CKA metric
  python hrsa/run_metrics.py --metric linear_cka \\
      --model_1 Qwen/Qwen2.5-1.5B \\
      --model_2 hkust-nlp/Qwen-2.5-1.5B-SimpleRL-Zoo \\
      --dataset /data/wychanbu/re_data/mmlu_pro_100_samples.jsonl \\
      --text_column prompt --num_sentences 1400

  # Run linear probe metric
  python hrsa/run_metrics.py --metric linear_probe \\
      --model_1 Qwen/Qwen2.5-1.5B \\
      --model_2 hkust-nlp/Qwen-2.5-1.5B-SimpleRL-Zoo \\
      --dataset /data/wychanbu/re_data/ag_news_7p6k.jsonl \\
      --text_column text --label_column label \\
      --task_type classification --train_fraction 0.8
        """,
    )

    # Required arguments
    parser.add_argument(
        "--metric",
        type=str,
        required=True,
        choices=list(METRIC_REGISTRY.keys()),
        help="Metric to compute",
    )
    parser.add_argument("--model_1", type=str, required=True, help="First model (source)")
    parser.add_argument("--model_2", type=str, required=True, help="Second model (target)")
    parser.add_argument("--dataset", type=str, required=True, help="Dataset name or path")

    # Common dataset arguments
    parser.add_argument(
        "--text_column", type=str, default="text", help="Text column name (default: text)"
    )
    parser.add_argument(
        "--label_column",
        type=str,
        default=None,
        help="Label column name (for KNN and linear_probe)",
    )
    parser.add_argument(
        "--dataset_split", type=str, default="train", help="Dataset split (default: train)"
    )
    parser.add_argument(
        "--dataset_subset", type=str, default="default", help="Dataset subset (default: default)"
    )
    parser.add_argument(
        "--num_sentences", type=int, default=2000, help="Number of sentences (default: 2000)"
    )

    # Common model arguments
    parser.add_argument(
        "--batch_size", type=int, default=8, help="Batch size (default: 8)"
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device (default: cuda)"
    )
    parser.add_argument(
        "--compute_dtype",
        type=str,
        default="float32",
        choices=sorted(DTYPE_ALIASES.keys()),
        help="Compute dtype (default: float32)",
    )
    parser.add_argument(
        "--is_causal_attn",
        action="store_true",
        help="Use causal attention mask",
    )

    # Output argument
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (default: from hrsa.config.METRIC_RESULTS_FOLDER or metric_results)",
    )

    # KNN-specific arguments
    parser.add_argument(
        "--neighbor_ks",
        type=str,
        default="5,10,50",
        help="Comma-separated k values for KNN (default: 5,10,50)",
    )
    parser.add_argument(
        "--neighbor_metric",
        type=str,
        default="cosine",
        help="Distance metric for KNN (default: cosine)",
    )
    parser.add_argument(
        "--max_tokens",
        type=int,
        default=50000,
        help="Max tokens for KNN (default: 50000)",
    )
    parser.add_argument(
        "--random_seed", type=int, default=42, help="Random seed (default: 42)"
    )

    # Linear probe-specific arguments
    parser.add_argument(
        "--task_type",
        type=str,
        default="classification",
        choices=["classification", "regression"],
        help="Task type for linear_probe (default: classification)",
    )
    parser.add_argument(
        "--train_fraction",
        type=float,
        default=0.8,
        help="Train fraction for linear_probe (default: 0.8)",
    )
    parser.add_argument(
        "--val_fraction",
        type=float,
        default=0.1,
        help="Val fraction for linear_probe (default: 0.1)",
    )
    parser.add_argument(
        "--test_fraction",
        type=float,
        default=0.1,
        help="Test fraction for linear_probe (default: 0.1)",
    )
    parser.add_argument(
        "--max_samples",
        type=int,
        default=8000,
        help="Max samples for linear_probe (default: 8000)",
    )
    parser.add_argument(
        "--min_class_samples",
        type=int,
        default=100,
        help="Min class samples for linear_probe (default: 100)",
    )
    parser.add_argument(
        "--source_layer",
        type=int,
        default=-1,
        help="Source layer for linear_probe (default: -1)",
    )
    parser.add_argument(
        "--target_layer",
        type=int,
        default=-1,
        help="Target layer for linear_probe (default: -1)",
    )
    parser.add_argument(
        "--projector",
        type=str,
        default="none",
        choices=["none", "pca"],
        help="Projector for linear_probe (default: none)",
    )
    parser.add_argument(
        "--shared_dim",
        type=int,
        default=None,
        help="Shared dimension for projection (optional)",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=32000,
        help="Max length for linear_probe (default: 32000)",
    )

    args = parser.parse_args()

    # Parse dtype
    dtype = parse_compute_dtype(args.compute_dtype)

    # Get output directory
    output_base_dir = Path(args.output_dir) if args.output_dir else Path(METRIC_RESULTS_FOLDER)

    print(f"\n{Fore.CYAN}{'=' * 80}{Style.RESET_ALL}")
    print(f"{Fore.YELLOW}Running metric: {args.metric}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}Model 1: {args.model_1}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}Model 2: {args.model_2}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}Dataset: {args.dataset}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}{'=' * 80}{Style.RESET_ALL}\n")

    # Get metric class from registry
    metric_class = METRIC_REGISTRY[args.metric]

    # Instantiate and run metric
    try:
        if args.metric == "linear_probe":
            # Linear probe uses ProbeDatasetConfig
            probe_config = ProbeDatasetConfig(
                label=args.dataset_split,
                name=args.dataset,
                dataset_config=None,
                text_column=args.text_column,
                label_column=args.label_column or "label",
                task_type=args.task_type,
                train_fraction=args.train_fraction,
                val_fraction=args.val_fraction,
                test_fraction=args.test_fraction,
                max_samples=args.max_samples,
                min_class_samples=args.min_class_samples,
                seed=args.random_seed,
            )

            metric = metric_class(
                model_1=args.model_1,
                model_2=args.model_2,
                probe_config=probe_config,
                device=args.device,
                dtype=dtype,
                batch_size=args.batch_size,
                output_base_dir=output_base_dir,
                source_layer=args.source_layer,
                target_layer=args.target_layer,
                projector=args.projector,
                shared_dim=args.shared_dim,
                max_length=args.max_length,
                seed=args.random_seed,
            )
        elif args.metric == "knn_overlap":
            # KNN overlap has additional parameters
            dataset_config = DatasetConfig(
                label=args.dataset_split,
                name=args.dataset,
                text_column=args.text_column,
                label_column=args.label_column,
                subset=args.dataset_subset,
                split=args.dataset_split,
                num_sentences=args.num_sentences,
            )

            neighbor_ks = parse_neighbor_ks(args.neighbor_ks)

            metric = metric_class(
                model_1=args.model_1,
                model_2=args.model_2,
                dataset_config=dataset_config,
                device=args.device,
                dtype=dtype,
                batch_size=args.batch_size,
                output_base_dir=output_base_dir,
                neighbor_ks=neighbor_ks,
                neighbor_metric=args.neighbor_metric,
                max_tokens=args.max_tokens,
                random_seed=args.random_seed,
                is_causal_attn=args.is_causal_attn,
            )
        else:
            # Standard metrics (CKA, Procrustes, Correlation)
            dataset_config = DatasetConfig(
                label=args.dataset_split,
                name=args.dataset,
                text_column=args.text_column,
                label_column=args.label_column,
                subset=args.dataset_subset,
                split=args.dataset_split,
                num_sentences=args.num_sentences,
            )

            metric = metric_class(
                model_1=args.model_1,
                model_2=args.model_2,
                dataset_config=dataset_config,
                device=args.device,
                dtype=dtype,
                batch_size=args.batch_size,
                output_base_dir=output_base_dir,
                is_causal_attn=args.is_causal_attn,
            )

        # Compute the metric
        print(f"{Fore.YELLOW}Computing {args.metric}...{Style.RESET_ALL}")
        results = metric.compute()

        print(f"\n{Fore.GREEN}{'=' * 80}{Style.RESET_ALL}")
        print(f"{Fore.GREEN}Successfully computed {args.metric}!{Style.RESET_ALL}")
        print(f"{Fore.GREEN}{'=' * 80}{Style.RESET_ALL}\n")

    except Exception as exc:
        print(f"\n{Fore.RED}Error computing {args.metric}: {exc}{Style.RESET_ALL}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
