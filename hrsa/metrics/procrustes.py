#!/usr/bin/env python3
"""
Orthogonal Procrustes alignment for comparing hidden representations.

This script reuses the activation-extraction utilities from
`evaluation/compare_cka.py` to collect hidden states for a base model and a
reasoning-enhanced variant on calibration, held-out, and optional OOD corpora.
It then fits both layer-wise and global orthogonal mappings that minimize
‖H_base · O − H_reasoning‖_F, reports residual variance ratios (RVR) and cosine
similarities, visualizes the learned O matrices, and writes plots/tables plus a
narrative insight file under `metric_results/procrustes_evaluation/...`.

Interpreting the outputs:
    * RVR ≤ 2% with cosine ≥ 0.995 indicates near-isometry (“same coordinates”).
    * 2–5% highlights mild rotations but largely shared subspaces.
    * >5% flags noticeably different representations worth deeper inspection.
"""

from __future__ import annotations

import argparse
import json
import math

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from colorama import Fore, Style
from tqdm import tqdm

from hrsa.utils import clear_memory
from hrsa.config import DatasetConfig
from hrsa.plot_style import get_palette


@dataclass
class SplitResult:
    """Holds activations for a split."""

    config: DatasetConfig
    activations_model1: torch.Tensor  # CPU
    activations_model2: torch.Tensor  # CPU

    @property
    def num_layers(self) -> int:
        return min(self.activations_model1.shape[0], self.activations_model2.shape[0])


def build_dataset_config(
    args: argparse.Namespace,
    prefix: str,
    label: str,
    fallback: Optional[DatasetConfig] = None,
    required: bool = False,
) -> Optional[DatasetConfig]:
    """Construct a DatasetConfig from CLI args, optionally inheriting from a fallback."""

    dataset = getattr(args, f"{prefix}_dataset", None)
    if dataset is None:
        if required:
            raise ValueError(f"--{prefix}_dataset is required.")
        return None

    def resolve(attr: str, default: str) -> str:
        value = getattr(args, f"{prefix}_{attr}", None)
        if value is not None:
            return value
        if fallback is not None:
            return getattr(fallback, attr)
        return default

    num_sentences = getattr(args, f"{prefix}_num_sentences", None)
    if num_sentences is None:
        num_sentences = fallback.num_sentences if fallback else args.calib_num_sentences

    return DatasetConfig(
        label=label,
        name=dataset,
        text_column=resolve("text_column", "text"),
        subset=resolve("subset", "main"),
        split=resolve("split", "train"),
        num_sentences=num_sentences,
    )

def solve_orthogonal_procrustes(
    x_tokens: torch.Tensor,
    y_tokens: torch.Tensor,
    dtype: torch.dtype,
    device: torch.device,
) -> torch.Tensor:
    """
    Fit an orthogonal mapping O that minimizes ||XO - Y||_F.
    """

    if x_tokens.size(0) == 0 or y_tokens.size(0) == 0:
        raise ValueError("Cannot fit Procrustes mapping with zero tokens.")

    min_tokens = min(x_tokens.size(0), y_tokens.size(0))
    x = x_tokens[:min_tokens].to(device=device, dtype=dtype)
    y = y_tokens[:min_tokens].to(device=device, dtype=dtype)

    x = x - x.mean(dim=0, keepdim=True)
    y = y - y.mean(dim=0, keepdim=True)

    cross_cov = x.transpose(0, 1) @ y
    # NOTE: torch CPU SVD is not implemented for bf16/fp16. We try requested dtype first,
    # then fall back to float32 for the decomposition if needed.
    try:
        U, _, Vh = torch.linalg.svd(cross_cov, full_matrices=False)
        O = U @ Vh
    except NotImplementedError:
        U, _, Vh = torch.linalg.svd(cross_cov.to(torch.float32), full_matrices=False)
        O = (U @ Vh).to(dtype)

    del x, y, cross_cov, U, Vh
    clear_memory()

    return O.to("cpu")


def fit_layerwise_mappings(
    activations1: torch.Tensor,
    activations2: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> Dict[str, List]:
    """Fit O matrices for each overlapping layer (using the last N layers)."""

    num_layers = min(activations1.shape[0], activations2.shape[0])
    layer_indices = list(range(-num_layers, 0))
    layer_numbers = list(range(1, num_layers + 1))
    solutions: List[torch.Tensor] = []

    print(f"{Fore.YELLOW}Fitting layer-wise orthogonal mappings...{Style.RESET_ALL}")
    for idx in tqdm(layer_indices, desc="Layer Procrustes"):
        x = activations1[idx]
        y = activations2[idx]
        O = solve_orthogonal_procrustes(x, y, dtype=dtype, device=device)
        solutions.append(O)

    clear_memory()
    return {"layer_indices": layer_indices, "layer_numbers": layer_numbers, "solutions": solutions}


def fit_global_mapping(
    activations1: torch.Tensor,
    activations2: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Fit a single mapping on the final layer (global embedding space)."""

    print(f"{Fore.YELLOW}Fitting global mapping on final layer...{Style.RESET_ALL}")
    final_layer_idx = -1
    x = activations1[final_layer_idx]
    y = activations2[final_layer_idx]
    return solve_orthogonal_procrustes(x, y, dtype=dtype, device=device)


def compute_alignment_metrics(
    x_tokens: torch.Tensor,
    y_tokens: torch.Tensor,
    o_matrix: torch.Tensor,
    dtype: torch.dtype,
    device: torch.device,
) -> Dict[str, float]:
    """Compute residual variance ratio and cosine similarity statistics."""

    min_tokens = min(x_tokens.size(0), y_tokens.size(0))
    if min_tokens == 0:
        raise ValueError("Cannot compute metrics with zero tokens.")

    x = x_tokens[:min_tokens].to(device=device, dtype=dtype)
    y = y_tokens[:min_tokens].to(device=device, dtype=dtype)
    o = o_matrix.to(device=device, dtype=dtype)

    mapped = x @ o
    residual = mapped - y

    residual_norm = torch.linalg.norm(residual, ord="fro") ** 2
    target_norm = torch.linalg.norm(y, ord="fro") ** 2 + 1e-12
    residual_variance_ratio = (residual_norm / target_norm).item()

    per_token_cos = F.cosine_similarity(mapped, y, dim=1, eps=1e-12)
    mean_cosine = per_token_cos.mean().item()
    min_cosine = per_token_cos.min().item()
    max_cosine = per_token_cos.max().item()

    global_cosine = (
        (mapped * y).sum()
        / (torch.linalg.norm(mapped, ord="fro") * torch.linalg.norm(y, ord="fro") + 1e-12)
    ).item()

    del x, y, o, mapped, residual, per_token_cos
    clear_memory()

    return {
        "num_tokens": int(min_tokens),
        "residual_variance_ratio": residual_variance_ratio,
        "mean_cosine": mean_cosine,
        "min_cosine": min_cosine,
        "max_cosine": max_cosine,
        "global_cosine": global_cosine,
    }


def compute_row_entropy(o_matrix: torch.Tensor) -> Dict[str, float]:
    """
    Compute row entropy for O* using squared entries as probabilities.
    Returns mean/min/max entropy across rows (normalized to [0, 1]).
    """

    o_cpu = o_matrix.detach().to(dtype=torch.float32, device="cpu")
    probs = o_cpu.pow(2)
    row_sums = probs.sum(dim=1, keepdim=True).clamp_min(1e-12)
    print(f"{Fore.YELLOW}Row sums: {row_sums}{Style.RESET_ALL}")
    probs = probs / row_sums
    entropy = -(probs * torch.log(probs.clamp_min(1e-12))).sum(dim=1)
    num_cols = probs.size(1)
    max_entropy = math.log(max(num_cols, 1))
    if max_entropy > 0:
        entropy = entropy / max_entropy
    return {
        "mean_row_inverse_entropy": 1 - float(entropy.mean().item()),
        "min_row_inverse_entropy": 1 - float(entropy.min().item()),
        "max_row_inverse_entropy": 1 - float(entropy.max().item()),
    }


def evaluate_split(
    split: SplitResult,
    layer_mappings: Dict[str, List],
    global_mapping: torch.Tensor,
    device: torch.device,
    dtype: torch.dtype,
) -> Dict[str, List[Dict[str, float]]]:
    """Apply learned mappings to a split and collect metrics."""

    results: Dict[str, List[Dict[str, float]]] = {"layer_metrics": [], "global_metrics": []}
    layer_indices = layer_mappings["layer_indices"]
    layer_numbers = layer_mappings["layer_numbers"]
    layer_solutions = layer_mappings["solutions"]
    num_layers = len(layer_solutions)

    print(
        f"{Fore.YELLOW}Evaluating split '{split.config.label}' with {num_layers} layers...{Style.RESET_ALL}"
    )

    # for idx, o_matrix, layer_number in tqdm(zip(layer_indices, layer_solutions, layer_numbers), desc="Evaluating layers"):
    #     label = f"L{layer_number}"
    #     x = split.activations_model1[idx]
    #     y = split.activations_model2[idx]
    #     metrics = compute_alignment_metrics(x, y, o_matrix, dtype=dtype, device=device)
    #     metrics["layer_label"] = label
    #     metrics["layer_index"] = layer_number
    #     results["layer_metrics"].append(metrics)

    # Global (final layer) metrics
    x_final = split.activations_model1[-1]
    y_final = split.activations_model2[-1]
    global_metrics = compute_alignment_metrics(
        x_final, y_final, global_mapping, dtype=dtype, device=device
    )
    global_metrics["layer_label"] = "Final"
    results["global_metrics"].append(global_metrics)

    clear_memory()
    return results


def plot_procrustes_metrics(
    split_label: str,
    metrics: Dict[str, List[Dict[str, float]]],
    output_dir: Path,
):
    """Create individual plots for residual variance ratio and cosine scores."""

    layer_metrics = metrics["layer_metrics"]
    if not layer_metrics:
        return

    output_dir.mkdir(parents=True, exist_ok=True)
    layer_labels = [
        str(m.get("layer_index", idx + 1)) for idx, m in enumerate(layer_metrics)
    ]
    rvr = [m["residual_variance_ratio"] for m in layer_metrics]
    mean_cos = [m["mean_cosine"] for m in layer_metrics]
    palette = get_palette(2)

    def _save_plot(
        values: List[float],
        ylabel: str,
        title_suffix: str,
        path: Path,
        chart_type: str = "bar",
        baseline: float | None = None,
        baseline_label: str | None = None,
        color: str = palette[0],
    ) -> None:
        fig, ax = plt.subplots(figsize=(8, 6))
        if chart_type == "bar":
            ax.bar(layer_labels, values, color=color)
        else:
            ax.plot(layer_labels, values, marker="o", color=color)
        ax.set_xlabel("Layer")
        ax.set_ylabel(ylabel)
        ax.set_title(f"{split_label}: {title_suffix}")
        if baseline is not None:
            ax.axhline(baseline, color="#6c757d", linestyle="--", linewidth=1.2, label=baseline_label)
        if baseline_label:
            ax.legend()
        fig.tight_layout()
        fig.savefig(path, bbox_inches="tight")
        plt.close(fig)
        print(f"{Fore.GREEN}Saved plot to {path}{Style.RESET_ALL}")

    rvr_path = output_dir / f"{split_label}_rvr.png"
    _save_plot(
        rvr,
        "Residual variance ratio",
        "Residual Variance Ratio",
        rvr_path,
        chart_type="bar",
        baseline=0.02,
        baseline_label="RVR=0.02",
        color=palette[0],
    )

    cos_path = output_dir / f"{split_label}_mean_cosine.png"
    _save_plot(
        mean_cos,
        "Mean cosine similarity",
        "Mean Cosine Similarity",
        cos_path,
        chart_type="line",
        baseline=0.995,
        baseline_label="cos=0.995",
        color=palette[1],
    )


def plot_o_matrix(
    o_matrix_data: np.ndarray,
    output_path: Path,
    mean_row_inverse_entropy: float,
    hidden_size: int,
) -> None:
    """Visualize a learned O matrix via heatmap."""

    data = o_matrix_data
    fig, ax = plt.subplots(figsize=(6, 6))
    im = ax.imshow(data, cmap="coolwarm", vmin=-1.0, vmax=1.0, aspect="auto")
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("O value")
    ax.set_title(f"Mean row inverse entropy: {mean_row_inverse_entropy:.4f}\n", fontsize=14, y=0.99)
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"{Fore.GREEN}Saved O matrix plot to {output_path}{Style.RESET_ALL}")


def plot_o_matrices(
    layer_mappings: Dict[str, List],
    global_mapping: torch.Tensor,
    calibration_config: DatasetConfig,
    output_root: Path,
    mean_row_inverse_entropy: float,
    global_o_sparsity: float,
    hidden_size: int,
) -> None:
    """Persist visualizations for layer-wise and global O matrices."""
    global_mapping_np = global_mapping.detach().to(dtype=torch.float32, device="cpu").numpy()

    o_dir = (
        output_root
        / f"{calibration_config.label}_{calibration_config.slug()}"
        / "plots"
    )
    o_dir.mkdir(parents=True, exist_ok=True)

    plot_o_matrix(
        global_mapping_np,
        output_path=o_dir / "O_global.png",
        mean_row_inverse_entropy=mean_row_inverse_entropy,
        hidden_size=hidden_size,
    )


def write_split_tables(
    split: SplitResult,
    metrics: Dict[str, List[Dict[str, float]]],
    tables_dir: Path,
) -> None:
    """Persist metrics as JSON and text summary."""

    tables_dir.mkdir(parents=True, exist_ok=True)
    json_path = tables_dir / f"{split.config.label}_metrics.json"
    txt_path = tables_dir / f"{split.config.label}_summary.txt"

    with open(json_path, "w") as handle:
        json.dump(
            {
                "split": split.config.label,
                "dataset": split.config.name,
                "layer_metrics": metrics["layer_metrics"],
                "global_metrics": metrics["global_metrics"],
            },
            handle,
            indent=4,
        )

    with open(txt_path, "w") as handle:
        handle.write(f"PROCRUSTES SUMMARY :: {split.config.label.upper()}\n")
        handle.write("=" * 80 + "\n")
        # for layer_metric in metrics["layer_metrics"]:
        #     handle.write(
        #         f"{layer_metric['layer_label']}: "
        #         f"RVR={layer_metric['residual_variance_ratio']:.4e}, "
        #         f"mean_cos={layer_metric['mean_cosine']:.5f}, "
        #         f"tokens={layer_metric['num_tokens']}\n"
        #     )
        handle.write("-" * 80 + "\n")
        global_metric = metrics["global_metrics"][0]
        handle.write("Global / final layer:\n")
        handle.write(
            f"RVR={global_metric['residual_variance_ratio']:.4e}, "
            f"mean_cos={global_metric['mean_cosine']:.5f}, "
            f"global_cos={global_metric['global_cosine']:.5f}, "
            f"tokens={global_metric['num_tokens']}\n"
        )

    print(f"{Fore.GREEN}Saved tables to {tables_dir}{Style.RESET_ALL}")


def interpret_alignment_signal(rvr: float, cosine: float) -> str:
    """Return a short textual interpretation for the provided metrics."""

    if rvr <= 0.02 and cosine >= 0.995:
        return "near-isometric (shared coordinates)."
    if rvr <= 0.05 and cosine >= 0.99:
        return "mild rotation with strong overlap."
    return "noticeable drift; investigate these layers."


def write_run_insights(
    output_dir: Path,
    split_reports: List[Dict[str, object]],
    row_entropy_stats: Dict[str, float],
    global_o_sparsity: float,
    hidden_size: int,
) -> None:
    """Write a lightweight narrative summary of the collected metrics.
    
    row_entropy_stats: row entropy summary statistics for global O
    global_o_sparsity: the sparsity of the Global O matrix
    hidden_size: dimension of the hidden states / O matrix
    """

    if not split_reports:
        return

    lines = [
        "ORTHOGONAL PROCRUSTES ALIGNMENT :: INSIGHT SUMMARY",
        "=" * 80,
        "This run fits layer-wise and global orthogonal maps between the base and",
        "reasoning models. Residual variance ratio (RVR) quantifies unexplained",
        "variance after alignment; cosine measures directional agreement.",
        "",
        "Interpretation guide:",
        "  - RVR ≤ 0.02 with cosine ≥ 0.995 ⇒ near-isometry (shared coordinates).",
        "  - 0.02 < RVR ≤ 0.05 ⇒ mild drift but mostly overlapping subspaces.",
        "  - RVR > 0.05 ⇒ noticeable rotations worth deeper inspection.",
        "  - Low row entropy ⇒ basis preserved (feature i ↦ i).",
        "  - High row entropy ⇒ features smeared across dimensions.",
        "",
    ]

    for report in split_reports:
        label = str(report["label"]).capitalize()
        dataset = report["dataset"]
        metrics = report["metrics"]
        # layer_metrics = metrics["layer_metrics"]
        # if not layer_metrics:
        #     continue
        final_metric = metrics["global_metrics"][0]
        # worst_layer = max(layer_metrics, key=lambda m: m["residual_variance_ratio"])
        # best_cos_layer = max(layer_metrics, key=lambda m: m["mean_cosine"])
        signal = interpret_alignment_signal(
            final_metric["residual_variance_ratio"], final_metric["mean_cosine"]
        )

        lines.append(f"[{label}] dataset={dataset}")
        lines.append(
            f"  Final layer:: RVR={final_metric['residual_variance_ratio']:.4e}, "
            f"mean_cos={final_metric['mean_cosine']:.5f} → {signal}"
        )
        # lines.append(
        #     f"  Worst layer:: {worst_layer['layer_label']} "
        #     f"(RVR={worst_layer['residual_variance_ratio']:.4e}, "
        #     f"mean_cos={worst_layer['mean_cosine']:.5f})"
        # )
        # lines.append(
        #     f"  Highest cosine layer:: {best_cos_layer['layer_label']} "
        #     f"(mean_cos={best_cos_layer['mean_cosine']:.5f})"
        # )
        lines.append(f"  Tokens evaluated: {final_metric['num_tokens']}")
        lines.append("-" * 80)
    
    lines.append(f"\nHidden size: {hidden_size}")
    lines.append(
        "Row entropy (global O): "
        f"mean={row_entropy_stats['mean_row_inverse_entropy']:.6f}, "
        f"min={row_entropy_stats['min_row_inverse_entropy']:.6f}, "
        f"max={row_entropy_stats['max_row_inverse_entropy']:.6f}"
    )
    lines.append(f"Sparsity of the Global O matrix: {global_o_sparsity}\n")

    lines.append("All detailed plots/tables live under this run directory.")

    insights_path = output_dir / "insights.txt"
    with open(insights_path, "w") as handle:
        handle.write("\n".join(lines))

    print(f"{Fore.CYAN}Saved narrative insights to {insights_path}{Style.RESET_ALL}")


def prepare_output_dirs(
    base_output_dir: Path,
    model_1_name: str,
    model_2_name: str,
) -> Path:
    """Create the root output directory for results."""

    output_dir = base_output_dir / f"{model_1_name}_vs_{model_2_name}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir