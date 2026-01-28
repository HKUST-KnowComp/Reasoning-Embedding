#!/usr/bin/env python3
"""
Compute per-dimension Pearson correlations between token activations from two models.

The script mirrors `cluster_sentence_embeddings.py`: it gathers raw hidden states for a
dataset split, flattens all valid tokens, and then correlates each activation dimension
between the base model and a reasoning-enhanced variant.

Outputs include correlation histograms, JSON summaries, per-dimension vectors (optional),
and a text report describing alignment quality across embedding dimensions.
"""

from __future__ import annotations

import json

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import matplotlib.pyplot as plt
import torch
from colorama import Fore, Style

from hrsa.plot_style import get_palette
from hrsa.config import DatasetConfig

@dataclass
class SplitActivations:
    """Token activations for both models."""

    config: DatasetConfig
    activations_model1: torch.Tensor  # CPU
    activations_model2: torch.Tensor  # CPU


def ensure_matching_shapes(activations_model1: torch.Tensor, activations_model2: torch.Tensor) -> None:
    """Ensure both activation tensors have identical shapes."""

    if activations_model1.shape != activations_model2.shape:
        raise ValueError(
            "Activation tensors must share the same shape. "
            f"Model 1: {activations_model1.shape}, Model 2: {activations_model2.shape}"
        )


def compute_dimension_correlations(
    activations_model1: torch.Tensor,
    activations_model2: torch.Tensor,
    dtype: torch.dtype = torch.float32,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    Compute Pearson correlations per embedding dimension.
    
    Args:
        activations_model1: Activations from first model.
        activations_model2: Activations from second model.
        dtype: Computation precision (default: float32).
        device: Device for computation (default: CPU).
    
    Returns:
        Correlation values per dimension.
    """

    ensure_matching_shapes(activations_model1, activations_model2)
    activations_model1 = activations_model1.to(device, dtype=dtype)
    activations_model2 = activations_model2.to(device, dtype=dtype)
    num_samples = int(activations_model1.shape[0])
    if num_samples == 0:
        raise ValueError("No tokens were provided for correlation computation.")

    ddof = 1 if num_samples > 1 else 0

    mean1 = activations_model1.mean(dim=0)
    mean2 = activations_model2.mean(dim=0)
    centered1 = activations_model1 - mean1
    centered2 = activations_model2 - mean2

    denom_factor = max(num_samples - 1, 1)
    covariance = (centered1 * centered2).sum(dim=0) / denom_factor

    if ddof:
        var1 = (centered1 * centered1).sum(dim=0) / denom_factor
        var2 = (centered2 * centered2).sum(dim=0) / denom_factor
    else:
        denom0 = max(num_samples, 1)
        var1 = (centered1 * centered1).sum(dim=0) / denom0
        var2 = (centered2 * centered2).sum(dim=0) / denom0

    std1 = torch.sqrt(var1)
    std2 = torch.sqrt(var2)
    denom = std1 * std2

    correlations = torch.full((activations_model1.shape[1],), float("nan"), dtype=activations_model1.dtype)
    valid_mask = denom > 1e-12
    correlations[valid_mask] = torch.clamp(covariance[valid_mask] / denom[valid_mask], -1.0, 1.0)
    return correlations


def summarize_correlations(correlations: torch.Tensor) -> Dict[str, float]:
    """Aggregate summary statistics for the per-dimension correlations."""

    correlations = correlations.to("cpu")
    finite_mask = torch.isfinite(correlations)
    finite_values = correlations[finite_mask]

    if finite_values.numel():
        finite_f32 = finite_values.to(torch.float32)
        percentiles = torch.quantile(
            finite_f32,
            torch.tensor([0.05, 0.25, 0.50, 0.75, 0.95], dtype=torch.float32),
        ).tolist()
        summary = {
            "mean": float(finite_f32.mean().item()),
            "median": float(finite_f32.median().item()),
            "std": float(finite_f32.std(unbiased=True).item()) if finite_values.numel() > 1 else 0.0,
            "min": float(finite_f32.min().item()),
            "max": float(finite_f32.max().item()),
            "p05": float(percentiles[0]),
            "p25": float(percentiles[1]),
            "p75": float(percentiles[3]),
            "p95": float(percentiles[4]),
            "positive_fraction": float(
                (finite_values >= 0).sum().item() / max(finite_values.numel(), 1)
            ),
            "abs_mean": float(finite_f32.abs().mean().item()),
        }
    else:
        summary = {
            "mean": 0.0,
            "median": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "p05": 0.0,
            "p25": 0.0,
            "p75": 0.0,
            "p95": 0.0,
            "positive_fraction": 0.0,
            "abs_mean": 0.0,
        }

    summary.update(
        {
            "num_dimensions": int(correlations.numel()),
            "num_valid_dimensions": int(finite_values.numel()),
            "num_degenerate_dimensions": int(correlations.numel() - finite_values.numel()),
        }
    )
    return summary


def interpret_correlation_summary(summary: Dict[str, float]) -> str:
    """Provide a coarse textual interpretation."""

    mean_corr = summary.get("mean", 0.0)
    abs_mean = summary.get("abs_mean", 0.0)

    if mean_corr >= 0.9 and abs_mean >= 0.9:
        return "per-dimension features nearly identical"
    if mean_corr >= 0.7:
        return "strong correspondence across most dimensions"
    if mean_corr >= 0.4:
        return "moderate overlap with notable drift"
    return "weak alignment; models learn divergent features"


def compute_layerwise_correlations(
    activations_model1: torch.Tensor,
    activations_model2: torch.Tensor,
    dtype: torch.dtype = torch.float32,
    device: torch.device = torch.device("cpu"),
) -> Tuple[torch.Tensor, Dict[Tuple[int, int], torch.Tensor]]:
    """
    Compute pairwise mean correlations across all layers of both models.
    
    For each pair of layers (i from model1, j from model2), compute the per-dimension
    Pearson correlations and return the mean correlation as the similarity metric.
    
    Args:
        activations_model1: First model activations (num_layers, num_tokens, hidden_dim)
        activations_model2: Second model activations (num_layers, num_tokens, hidden_dim)
    
    Returns:
        Tuple of:
            - correlation_matrix: Mean correlations of shape (num_layers, num_layers)
            - all_correlations: Full per-dimension correlations for each layer pair
    """
    num_layers = min(activations_model1.shape[0], activations_model2.shape[0])
    print(f"{Fore.YELLOW}Computing layerwise correlations...{Style.RESET_ALL}")
    print(f"{Fore.MAGENTA}Number of layers: {num_layers}{Style.RESET_ALL}")
    
    activations_model1 = activations_model1.to("cpu")
    activations_model2 = activations_model2.to("cpu")
    correlation_matrix = torch.zeros((num_layers, num_layers), dtype=torch.float32)
    all_correlations: Dict[Tuple[int, int], torch.Tensor] = {}
    
    total_pairs = num_layers * num_layers
    from tqdm import tqdm
    pbar = tqdm(total=total_pairs, desc="Computing correlations", leave=True)
    
    for i in range(num_layers):
        layer_i_activations = activations_model1[i]
        
        for j in range(num_layers):
            layer_j_activations = activations_model2[j]
            
            # Handle token count mismatch
            if layer_i_activations.shape[0] != layer_j_activations.shape[0]:
                min_tokens = min(layer_i_activations.shape[0], layer_j_activations.shape[0])
                if i == 0 and j == 0:
                    print(
                        f"\n{Fore.RED}Warning: token count mismatch. "
                        f"Using first {min_tokens} tokens.{Style.RESET_ALL}"
                    )
                act1 = layer_i_activations[:min_tokens]
                act2 = layer_j_activations[:min_tokens]
            else:
                act1 = layer_i_activations
                act2 = layer_j_activations
            
            correlations = compute_dimension_correlations(act1, act2)
            summary = summarize_correlations(correlations)
            correlation_matrix[i, j] = summary["mean"]
            all_correlations[(i, j)] = correlations
            
            pbar.update(1)
    
    pbar.close()
    print(f"{Fore.CYAN}Correlation matrix shape: {tuple(correlation_matrix.shape)}{Style.RESET_ALL}")
    return correlation_matrix, all_correlations


def plot_correlation_heatmap(
    correlation_matrix: torch.Tensor,
    output_path: Path,
    model_1_name: str,
    model_2_name: str,
) -> Path:
    """Plot the correlation matrix as a heatmap."""
    corr_np = correlation_matrix.detach().cpu().to(torch.float32).numpy()
    num_layers = corr_np.shape[0]
    layer_labels = [str(idx) for idx in range(1, num_layers + 1)]
    
    print(f"{Fore.YELLOW}Saving correlation heatmap to {output_path}{Style.RESET_ALL}")
    
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(corr_np, vmin=-1.0, vmax=1.0, cmap="RdBu_r")
    ax.set_title("Layer-wise Mean Dimension Correlations")
    ax.set_xlabel("Reasoning layers")
    ax.set_ylabel("Base layers")
    ax.set_xticks(range(num_layers))
    ax.set_xticklabels(layer_labels, rotation=45, ha="right")
    ax.set_yticks(range(num_layers))
    ax.set_yticklabels(layer_labels)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Mean Pearson correlation")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    
    print(f"{Fore.GREEN}Saved correlation heatmap to {output_path}{Style.RESET_ALL}")
    return output_path


def plot_correlation_diagonal(
    correlation_matrix: torch.Tensor,
    output_path: Path,
    model_1_name: str,
    model_2_name: str,
) -> Path:
    """Plot the diagonal (matching-layer) correlations."""
    corr_np = correlation_matrix.detach().cpu().to(torch.float32).numpy()
    num_layers = corr_np.shape[0]
    layer_labels = [str(idx) for idx in range(1, num_layers + 1)]
    diagonal_scores = corr_np.diagonal()
    
    palette = get_palette(1)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(range(1, num_layers + 1), diagonal_scores, marker="o", color=palette[0])
    ax.set_ylim(-1.0, 1.0)
    ax.axhline(y=0, color='gray', linestyle='--', alpha=0.5)
    ax.set_xticks(range(1, num_layers + 1))
    ax.set_xticklabels(layer_labels, rotation=45, ha="right")
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Mean Pearson correlation")
    ax.set_title("Matching-layer Mean Correlations")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    
    print(f"{Fore.GREEN}Saved correlation diagonal plot to {output_path}{Style.RESET_ALL}")
    return output_path


def write_layerwise_statistics(
    correlation_matrix: torch.Tensor,
    output_dir: Path,
    model_1_name: str,
    model_2_name: str,
) -> None:
    """Write summary statistics for layerwise correlations."""
    stats_path = output_dir / f"correlation_stats_{model_1_name}_vs_{model_2_name}.txt"
    corr = correlation_matrix.detach().cpu().to(torch.float32)
    diagonal_scores = torch.diagonal(corr)
    mean_diag = float(diagonal_scores.mean().item())
    max_diag = float(diagonal_scores.max().item())
    min_diag = float(diagonal_scores.min().item())
    
    with open(stats_path, 'w') as handle:
        handle.write("LAYERWISE CORRELATION SUMMARY STATISTICS\n")
        handle.write("=" * 80 + "\n")
        handle.write(f"Mean diagonal correlation: {mean_diag:.4f}\n")
        handle.write(f"Max diagonal correlation: {max_diag:.4f}\n")
        handle.write(f"Min diagonal correlation: {min_diag:.4f}\n")
        handle.write(f"\nOverall matrix mean: {float(corr.mean().item()):.4f}\n")
        handle.write(f"Overall matrix std: {float(corr.std(unbiased=False).item()):.4f}\n")
        handle.write("\nLayer-wise correlations (matching layers):\n")
        num_layers = int(diagonal_scores.shape[0])
        for idx in range(num_layers):
            layer_label = idx + 1
            handle.write(f"Layer {layer_label}: {float(diagonal_scores[idx].item()):.4f}\n")
    
    print(f"{Fore.YELLOW}\n" + "=" * 80 + f"{Style.RESET_ALL}")
    print("LAYERWISE CORRELATION SUMMARY STATISTICS")
    print("=" * 80)
    print(f"{Fore.CYAN}Mean diagonal correlation: {mean_diag:.4f}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}Max diagonal correlation: {max_diag:.4f}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}Min diagonal correlation: {min_diag:.4f}{Style.RESET_ALL}")
    print("=" * 80)
    print(f"\n{Fore.GREEN}Stats saved to {stats_path}{Style.RESET_ALL}")


def plot_correlation_histogram(
    correlations: torch.Tensor,
    output_path: Path,
    bins: int,
    dataset_label: str,
) -> None:
    """Plot histogram of the correlation distribution."""

    corr = correlations.detach().cpu().to(torch.float32)
    finite_values = corr[torch.isfinite(corr)]
    if not finite_values.numel():
        print(f"{Fore.RED}No valid correlations to plot.{Style.RESET_ALL}")
        return

    palette = get_palette(1)
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.hist(
        finite_values.numpy(),
        bins=bins,
        range=(-1, 1),
        color=palette[0],
        alpha=0.85,
        edgecolor="white",
    )
    ax.set_title(f"Dimension-Wise Correlations")
    ax.set_xlabel("Pearson correlation")
    ax.set_ylabel("Frequency")
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"{Fore.GREEN}Saved correlation histogram to {output_path}{Style.RESET_ALL}")


def write_correlation_tables(
    split: SplitActivations,
    summary: Dict[str, float],
    correlations: torch.Tensor,
    tables_dir: Path,
    save_full_vector: bool,
) -> None:
    """Persist correlation summaries and optional vectors."""

    tables_dir.mkdir(parents=True, exist_ok=True)
    json_path = tables_dir / f"{split.config.slug()}_correlations.json"
    txt_path = tables_dir / f"{split.config.slug()}_correlations.txt"

    record = {
        "split": split.config.label,
        "dataset": split.config.name,
        "summary": summary,
        "interpretation": interpret_correlation_summary(summary),
    }
    with open(json_path, "w") as handle:
        json.dump(record, handle, indent=4)

    with open(txt_path, "w") as handle:
        handle.write(f"DIMENSION CORRELATION SUMMARY :: {split.config.label.upper()}\n")
        handle.write("=" * 80 + "\n")
        handle.write(f"Mean correlation={summary['mean']:.4f}\n")
        handle.write(f"Median correlation={summary['median']:.4f}\n")
        handle.write(
            f"Std={summary['std']:.4f}, min={summary['min']:.4f}, "
            f"max={summary['max']:.4f}\n"
        )
        handle.write(
            f"P05/P25/P75/P95={summary['p05']:.4f}/"
            f"{summary['p25']:.4f}/{summary['p75']:.4f}/{summary['p95']:.4f}\n"
        )
        handle.write(
            f"Valid dimensions={summary['num_valid_dimensions']} / "
            f"{summary['num_dimensions']} (degenerate={summary['num_degenerate_dimensions']})\n"
        )
        handle.write(
            f"Positive fraction={summary['positive_fraction']:.4f}, "
            f"Abs mean={summary['abs_mean']:.4f}\n"
        )
        handle.write("-" * 80 + "\n")
        handle.write(f"Interpretation: {record['interpretation']}\n")

    if save_full_vector:
        vector_path = tables_dir / f"{split.config.slug()}_correlations.pt"
        torch.save(correlations.detach().cpu(), vector_path)
        print(f"{Fore.GREEN}Saved correlation vector to {vector_path}{Style.RESET_ALL}")

    print(f"{Fore.GREEN}Saved correlation tables to {tables_dir}{Style.RESET_ALL}")


def write_run_summary(
    dataset_dir: Path,
    dataset_label: str,
    dataset_name: str,
    summary: Dict[str, float],
) -> None:
    """Write a lightweight insight file at the run root."""

    interpretation = interpret_correlation_summary(summary)
    lines = [
        "DIMENSION CORRELATION :: INSIGHT SUMMARY",
        "=" * 80,
        f"[{dataset_label}] dataset={dataset_name}",
        f"  mean={summary['mean']:.4f}, median={summary['median']:.4f}, "
        f"std={summary['std']:.4f}",
        f"  abs_mean={summary['abs_mean']:.4f}, positive_fraction={summary['positive_fraction']:.4f}",
        f"  valid_dims={summary['num_valid_dimensions']} / {summary['num_dimensions']}",
        f"  interpretation → {interpretation}",
        "",
        "Refer to the dataset directory for detailed tables and plots.",
    ]

    insights_path = dataset_dir / "insights.txt"
    with open(insights_path, "w") as handle:
        handle.write("\n".join(lines))

    print(f"{Fore.CYAN}Saved insight summary to {insights_path}{Style.RESET_ALL}")


def prepare_output_dirs(base_output_dir: Path, model_1_name: str, model_2_name: str) -> Path:
    """Create output directory root."""

    output_dir = base_output_dir / f"{model_1_name}_vs_{model_2_name}"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir