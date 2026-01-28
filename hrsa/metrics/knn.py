#!/usr/bin/env python3
"""
Compute k-NN neighbor-overlap scores between two models' token activations.

This script mirrors the CLI ergonomics of `cluster_sentence_embeddings.py`, but
it keeps every token-level representation (at a target hidden layer) so we can
measure how often each token's nearest neighbors agree between two models. The
main metric is the mean per-token Jaccard overlap across a list of k values.
Results are written as JSON/text summaries under `metric_results/`.
"""

from __future__ import annotations

import json

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import torch
import torch.nn.functional as F
from colorama import Fore, Style
import matplotlib.pyplot as plt

from hrsa.plot_style import get_palette
from hrsa.config import DatasetConfig

def maybe_subsample_tokens(
    activations_a: torch.Tensor,
    activations_b: torch.Tensor,
    token_labels: Optional[List[str]],
    max_tokens: Optional[int],
    random_seed: int,
) -> Tuple[torch.Tensor, torch.Tensor, Optional[List[str]]]:
    """Uniformly subsample tokens if requested."""

    if activations_a.ndim != 3 or activations_b.ndim != 3:
        raise ValueError("Activations must have shape (num_layers, num_tokens, hidden_dim).")

    total_tokens = activations_a.shape[1]
    if max_tokens is None or total_tokens <= max_tokens:
        return activations_a, activations_b, token_labels

    g = torch.Generator(device="cpu")
    g.manual_seed(int(random_seed))
    indices = torch.randperm(int(total_tokens), generator=g)[: int(max_tokens)].sort().values
    sub_a = activations_a[:, indices, :]
    sub_b = activations_b[:, indices, :]
    sub_labels = (
        [token_labels[int(idx)] for idx in indices.tolist()] if token_labels is not None else None
    )
    print(
        f"{Fore.MAGENTA}Subsampled tokens from {total_tokens} to {len(indices)} for computation."
        f"{Style.RESET_ALL}"
    )
    return sub_a, sub_b, sub_labels

def build_neighbor_graph(
    activations: torch.Tensor,
    max_k: int,
    metric: str = "cosine",
    chunk_size: int = 1024,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    Return neighbor indices of shape (num_tokens, max_k).
    
    Args:
        activations: Token activations (num_tokens, hidden_dim).
        max_k: Maximum number of neighbors to find.
        metric: Distance metric ('cosine' or 'euclidean').
        chunk_size: Chunk size for memory-efficient computation.
        device: Device for computation (default: CPU).
    
    Returns:
        Neighbor indices tensor (num_tokens, max_k).
    """

    activations = activations.to(device)
    if activations.shape[0] <= 1:
        raise ValueError("Need at least 2 tokens to compute neighbor overlap.")

    requested_k = min(max_k, activations.shape[0] - 1)
    if requested_k <= 0:
        raise ValueError("Not enough tokens for the requested neighbor sizes.")

    num_tokens = int(activations.shape[0])
    neighbors = torch.empty((num_tokens, requested_k), dtype=torch.long, device=device)

    if metric == "cosine":
        x = F.normalize(activations, p=2, dim=1)
        x_t = x.t()
        neg_inf = torch.tensor(-1e9, dtype=x.dtype, device=device)
        for start in range(0, num_tokens, chunk_size):
            end = min(start + chunk_size, num_tokens)
            chunk = x[start:end]
            sim = chunk @ x_t  # (chunk, num_tokens)
            row_ids = torch.arange(start, end, device=device)
            sim[torch.arange(end - start, device=device), row_ids] = neg_inf
            topk = torch.topk(sim, k=requested_k, dim=1, largest=True).indices
            neighbors[start:end] = topk.to(torch.long)
        return neighbors.to("cpu")

    if metric == "euclidean":
        x = activations
        x_norm = (x * x).sum(dim=1)  # (num_tokens,)
        x_t = x.t()
        pos_inf = torch.tensor(float("inf"), dtype=x.dtype, device=device)
        for start in range(0, num_tokens, chunk_size):
            end = min(start + chunk_size, num_tokens)
            chunk = x[start:end]
            chunk_norm = (chunk * chunk).sum(dim=1, keepdim=True)
            dist2 = chunk_norm + x_norm.unsqueeze(0) - 2 * (chunk @ x_t)
            row_ids = torch.arange(start, end, device=device)
            dist2[torch.arange(end - start, device=device), row_ids] = pos_inf
            topk = torch.topk(-dist2, k=requested_k, dim=1, largest=True).indices
            neighbors[start:end] = topk.to(torch.long)
        return neighbors.to("cpu")

    raise ValueError(f"Unknown metric '{metric}'. Expected 'cosine' or 'euclidean'.")


def compute_jaccard_scores(
    neighbors_a: torch.Tensor,
    neighbors_b: torch.Tensor,
    ks: Sequence[int],
) -> Dict[int, Dict[str, object]]:
    """Compute per-k overlap statistics."""

    results: Dict[int, Dict[str, object]] = {}
    neighbors_a = neighbors_a.to("cpu")
    neighbors_b = neighbors_b.to("cpu")
    num_tokens = int(neighbors_a.shape[0])

    for k in ks:
        k = int(k)
        if k <= 0:
            raise ValueError("k must be positive.")
        if k > int(neighbors_a.shape[1]):
            raise ValueError(
                f"Requested k={k} exceeds built neighbor limit={neighbors_a.shape[1]}."
            )
        subset_a = neighbors_a[:, :k]
        subset_b = neighbors_b[:, :k]

        # intersection count per token via broadcasting (n, k, k)
        eq = subset_a.unsqueeze(2) == subset_b.unsqueeze(1)
        intersection = eq.any(dim=2).sum(dim=1).to(torch.float32)
        union = (2 * k) - intersection
        overlaps_array = (intersection / union).to(torch.float32)

        stats = {
            "k": k,
            "num_tokens": num_tokens,
            "mean_overlap": float(overlaps_array.mean().item()),
            "median_overlap": float(overlaps_array.median().item()),
            "std_overlap": float(overlaps_array.std(unbiased=False).item()),
            "min_overlap": float(overlaps_array.min().item()),
            "max_overlap": float(overlaps_array.max().item()),
        }
        results[k] = {"stats": stats, "per_token": overlaps_array}
        print(
            f"{Fore.CYAN}k={k}: mean={stats['mean_overlap']:.4f}, "
            f"min={stats['min_overlap']:.4f}, max={stats['max_overlap']:.4f}{Style.RESET_ALL}"
        )

    return results


def aggregate_label_stats(
    overlaps: torch.Tensor,
    token_labels: Optional[List[str]],
    k: int,
) -> Optional[List[Dict[str, object]]]:
    """Compute label-conditioned statistics if labels are provided."""

    if token_labels is None:
        return None
    overlaps = overlaps.to("cpu").to(torch.float32)
    if len(token_labels) != int(overlaps.shape[0]):
        raise ValueError(
            f"Token label count ({len(token_labels)}) does not match overlaps ({int(overlaps.shape[0])})."
        )

    unique_labels = sorted(set(token_labels))
    stats: List[Dict[str, object]] = []
    for label in unique_labels:
        idxs = [i for i, lab in enumerate(token_labels) if lab == label]
        if not idxs:
            continue
        values = overlaps[idxs]
        stats.append(
            {
                "label": str(label),
                "count": int(len(idxs)),
                "mean_overlap": float(values.mean().item()),
                "min_overlap": float(values.min().item()),
                "max_overlap": float(values.max().item()),
            }
        )
    stats.sort(key=lambda x: x["mean_overlap"])
    return stats


def expand_token_labels(sentence_labels: Optional[List[str]], counts: torch.Tensor) -> Optional[List[str]]:
    """Broadcast sentence-level labels to tokens."""

    if sentence_labels is None:
        return None
    if len(sentence_labels) != len(counts):
        raise ValueError(
            f"Label count ({len(sentence_labels)}) does not match sentence count ({len(counts)})."
        )

    token_labels: List[str] = []
    for label, count in zip(sentence_labels, counts):
        token_labels.extend([label] * int(count))
    return token_labels


def save_single_layer_outputs(
    split_dir: Path,
    dataset_config: DatasetConfig,
    ks: Sequence[int],
    overlap_results: Dict[int, Dict[str, float]],
    label_stats: Dict[int, Optional[List[Dict[str, object]]]],
) -> None:
    """Write JSON + text summaries for single-layer runs."""

    metrics_path = split_dir / "knn_overlap_metrics.json"
    summary_path = split_dir / "summary.txt"

    serializable = {
        "mode": "single_layer",
        "dataset": dataset_config.name,
        "label_column": dataset_config.label_column,
        "num_sentences": dataset_config.num_sentences,
        "ks": list(map(int, ks)),
        "results": {
            str(k): {
                "stats": overlap_results[k],
                "label_stats": label_stats.get(k),
            }
            for k in ks
        },
    }

    with open(metrics_path, "w") as handle:
        json.dump(serializable, handle, indent=4)

    lines = [
        "K-NN NEIGHBOR OVERLAP SUMMARY",
        "=" * 80,
        f"Dataset: {dataset_config.name}",
        f"Split: {dataset_config.split} | Subset: {dataset_config.subset}",
        f"Sentences: {dataset_config.num_sentences}",
        "-" * 80,
    ]
    for k in ks:
        stats = overlap_results[k]
        lines.append(
            f"k={k}: mean={stats['mean_overlap']:.4f}, median={stats['median_overlap']:.4f}, "
            f"min={stats['min_overlap']:.4f}, max={stats['max_overlap']:.4f}, std={stats['std_overlap']:.4f}"
        )
        label_entries = label_stats.get(k)
        if label_entries:
            worst = label_entries[0]
            best = label_entries[-1]
            lines.append(
                f"  Label span: worst={worst['label']} ({worst['mean_overlap']:.4f}, n={worst['count']}), "
                f"best={best['label']} ({best['mean_overlap']:.4f}, n={best['count']})"
            )
        lines.append("-" * 80)

    with open(summary_path, "w") as handle:
        handle.write("\n".join(lines))

    print(f"{Fore.GREEN}Saved metrics to {metrics_path}{Style.RESET_ALL}")
    print(f"{Fore.GREEN}Saved summary to {summary_path}{Style.RESET_ALL}")


def save_layerwise_outputs(
    split_dir: Path,
    dataset_config: DatasetConfig,
    ks: Sequence[int],
    layer_results: List[Dict[str, object]],
    plot_path: Path,
) -> None:
    """Persist layer-wise metrics/summary."""

    metrics_path = split_dir / "knn_overlap_metrics.json"
    summary_path = split_dir / "summary.txt"

    payload = {
        "mode": "layerwise",
        "dataset": dataset_config.name,
        "label_column": dataset_config.label_column,
        "num_sentences": dataset_config.num_sentences,
        "ks": list(map(int, ks)),
        "num_layers": len(layer_results),
        "plot_path": str(plot_path),
        "layers": layer_results,
    }

    with open(metrics_path, "w") as handle:
        json.dump(payload, handle, indent=4)

    lines = [
        "K-NN NEIGHBOR OVERLAP :: LAYER-WISE SUMMARY",
        "=" * 80,
        f"Dataset: {dataset_config.name}",
        f"Split: {dataset_config.split} | Subset: {dataset_config.subset}",
        f"Sentences: {dataset_config.num_sentences}",
        f"Layers evaluated: {len(layer_results)}",
        "-" * 80,
    ]

    for k in ks:
        means = [
            (
                entry["position"],
                entry["results"][str(k)]["stats"]["mean_overlap"],  # type: ignore[index]
            )
            for entry in layer_results
        ]
        best_layer = max(means, key=lambda item: item[1])
        worst_layer = min(means, key=lambda item: item[1])
        best_pos = int(best_layer[0])
        worst_pos = int(worst_layer[0])
        best_model_idx = layer_results[best_pos].get("model_1_layer_index", best_pos)
        worst_model_idx = layer_results[worst_pos].get("model_1_layer_index", worst_pos)
        lines.append(
            f"k={k}: mean overlap spans [{worst_layer[1]:.4f}, {best_layer[1]:.4f}] "
            f"(worst layer idx={worst_model_idx}, best layer idx={best_model_idx})"
        )

    lines.append("-" * 80)
    lines.append(f"Layer-wise curves plotted at: {plot_path}")

    with open(summary_path, "w") as handle:
        handle.write("\n".join(lines))

    print(f"{Fore.GREEN}Saved layer metrics to {metrics_path}{Style.RESET_ALL}")
    print(f"{Fore.GREEN}Saved layer summary to {summary_path}{Style.RESET_ALL}")


def plot_layerwise_curves(
    layer_results: List[Dict[str, object]],
    ks: Sequence[int],
    output_path: Path,
) -> None:
    """Plot mean overlap vs. layer index for each k."""

    positions = [entry["position"] for entry in layer_results]
    tick_labels = [
        str(entry.get("model_1_layer_index", pos)) for pos, entry in zip(positions, layer_results)
    ]
    fig, ax = plt.subplots(figsize=(8, 6))
    palette = get_palette(len(ks))
    for idx, k in enumerate(ks):
        means = [
            entry["results"][str(k)]["stats"]["mean_overlap"]  # type: ignore[index]
            for entry in layer_results
        ]
        ax.plot(positions, means, marker="o", label=f"k={k}", color=palette[idx])

    ax.set_title("Layer-wise k-NN Overlap (mean Jaccard)")
    ax.set_xlabel("Layer index")
    ax.set_ylabel("Mean overlap")
    ax.set_ylim(0.0, 1.0)
    ax.set_xticks(positions, tick_labels, rotation=45)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, bbox_inches="tight")
    plt.close(fig)
    print(f"{Fore.GREEN}Saved layer plot to {output_path}{Style.RESET_ALL}")