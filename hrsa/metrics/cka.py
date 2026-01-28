#!/usr/bin/env python3
"""
Compare internal representations of two language models using linear CKA.

This script mirrors the CLI, data handling, and activation extraction flow of
`evaluation/compare_cca.py`, but swaps the SVCCA metric for Centered Kernel
Alignment (CKA) to quantify representational similarity.
"""

from pathlib import Path
from typing import Optional, Tuple

import torch
import matplotlib.pyplot as plt
from tqdm import tqdm
from colorama import Fore, Style
from tqdm import tqdm

from hrsa.plot_style import get_palette
from hrsa.utils import clear_memory

def compute_layerwise_cka(
    activations1: torch.Tensor,
    activations2: torch.Tensor,
    use_float64: bool = False,
    dtype: Optional[torch.dtype] = None,
):
    """
    Compute pairwise CKA for the last layers of two models with optimization.
    
    For centered data, let `X_c = H X` and `Y_c = H Y` with `H = I - 11ᵀ/n`. The centered Gram matrices are

    - `K_c = H X Xᵀ H = X_c X_cᵀ`
    - `L_c = H Y Yᵀ H = Y_c Y_cᵀ`

    The Hilbert–Schmidt inner product that appears in CKA is

    `HSIC = ⟨K_c, L_c⟩_F = trace(K_c L_c) = trace(X_c X_cᵀ Y_c Y_cᵀ)`.

    Using cyclic trace (`trace(AB) = trace(BA)`), this becomes

    `trace(X_cᵀ Y_c Y_cᵀ X_c) = ||X_cᵀ Y_c||_F²`.

    So the Frobenius norm of the “covariance” matrix `X_cᵀ Y_c` equals the Hilbert–Schmidt inner product of the centered Gram matrices. That’s why `linear_cka` in the script can center features directly and work with `cross_cov = X_cᵀ Y_c`: it computes the same quantity as explicitly forming `K_c = HXXᵀH` and `L_c = HYYᵀH`, but avoids materializing huge token-by-token Gram matrices.
    
    Args:
        activations1: First model activations (num_layers, num_tokens, hidden_dim) [torch, CPU]
        activations2: Second model activations (num_layers, num_tokens, hidden_dim) [torch, CPU]
        device: torch device for computation
        use_float64: Use float64 for higher precision (slower) vs float32 (faster)
        dtype: Optional torch dtype override (supports bfloat16/float16/float32/float64)
    
    Returns:
        CKA matrix of shape (num_layers, num_layers) as a CPU torch.FloatTensor
    """
    num_layers = min(activations1.shape[0], activations2.shape[0])
    print(f"{Fore.YELLOW}Computing CKA similarities (optimized)...{Style.RESET_ALL}")
    print(f"{Fore.MAGENTA}Number of layers: {num_layers}{Style.RESET_ALL}")
    
    if dtype is None:
        dtype = torch.float64 if use_float64 else torch.float32
    elif use_float64 and dtype != torch.float64:
        raise ValueError("Conflicting arguments: dtype overrides use_float64.")
    if dtype not in (torch.float16, torch.float32, torch.float64, torch.bfloat16):
        raise ValueError(f"Unsupported dtype for CKA computation: {dtype}")

    # Force CPU for metric computation; model forward may run elsewhere.
    activations1 = activations1.to("cpu")
    activations2 = activations2.to("cpu")

    cka_matrix = torch.zeros((num_layers, num_layers), dtype=torch.float32, device="cpu")

    layer_indices = list(range(-num_layers, 0))
    
    # Pre-load and center all layers at once to avoid redundant computation
    print(f"{Fore.YELLOW}Pre-centering activations...{Style.RESET_ALL}")
    act1_centered = []
    act2_centered = []
    cov_norms_1 = []
    cov_norms_2 = []
    
    for i in tqdm(range(num_layers), desc="Pre-processing layers"):
        layer_idx = layer_indices[i]
        
        # Model 1
        x_t = activations1[layer_idx].to("cpu", dtype=dtype)
        x_t = x_t - x_t.mean(dim=0, keepdim=True)
        cov_x = x_t.T @ x_t
        cov_norm_x = torch.linalg.norm(cov_x, ord='fro')
        act1_centered.append(x_t.to("cpu"))
        cov_norms_1.append(cov_norm_x.to("cpu"))
        
        # Model 2
        y_t = activations2[layer_idx].to("cpu", dtype=dtype)
        y_t = y_t - y_t.mean(dim=0, keepdim=True)
        cov_y = y_t.T @ y_t
        cov_norm_y = torch.linalg.norm(cov_y, ord='fro')
        act2_centered.append(y_t.to("cpu"))
        cov_norms_2.append(cov_norm_y.to("cpu"))
    
    # Compute pairwise CKA with pre-centered data
    print(f"{Fore.YELLOW}Computing pairwise CKA...{Style.RESET_ALL}")
    pbar = tqdm(total=num_layers * num_layers, desc="Computing CKA", leave=True)
    
    for i in range(num_layers):
        x_t = act1_centered[i].to("cpu")
        norm_x = cov_norms_1[i].to("cpu")
        
        for j in range(num_layers):
            y_t = act2_centered[j].to("cpu")
            norm_y = cov_norms_2[j].to("cpu")
            
            # Handle token count mismatch
            if x_t.shape[0] != y_t.shape[0]:
                min_tokens = min(x_t.shape[0], y_t.shape[0])
                if i == 0 and j == 0:  # Only print warning once
                    print(
                        f"\n{Fore.RED}Warning: token count mismatch. "
                        f"Using first {min_tokens} tokens.{Style.RESET_ALL}"
                    )
                x_layer = x_t[:min_tokens]
                y_layer = y_t[:min_tokens]
                # Recompute norms for truncated data
                cov_x = x_layer.T @ x_layer
                cov_y = y_layer.T @ y_layer
                norm_x_trunc = torch.linalg.norm(cov_x, ord='fro')
                norm_y_trunc = torch.linalg.norm(cov_y, ord='fro')
            else:
                x_layer = x_t
                y_layer = y_t
                norm_x_trunc = norm_x
                norm_y_trunc = norm_y
            
            # Compute cross-covariance
            cross_cov = x_layer.T @ y_layer
            numerator = torch.linalg.norm(cross_cov, ord='fro') ** 2
            denominator = norm_x_trunc * norm_y_trunc + 1e-12
            
            cka_value = (numerator / denominator).clamp(min=0.0, max=1.0)
            cka_matrix[i, j] = cka_value.to("cpu").item()
            pbar.update(1)
            
            del x_layer, y_layer, norm_x_trunc, norm_y_trunc, cross_cov, numerator, denominator, cka_value
            clear_memory()

    pbar.close()
    
    # Clean up
    del act1_centered, act2_centered, cov_norms_1, cov_norms_2
    clear_memory()
    
    print(f"{Fore.CYAN}CKA matrix shape: {tuple(cka_matrix.shape)}{Style.RESET_ALL}")
    return cka_matrix


def plot_cka(
    cka_matrix: torch.Tensor, output_dir: Path, model_1_name: str, model_2_name: str
) -> Tuple[Path, Path]:
    """Plot the CKA matrix heatmap and diagonal alignment curves as separate figures."""
    output_dir.mkdir(parents=True, exist_ok=True)
    cka_np = cka_matrix.detach().cpu().to(torch.float32).numpy()
    num_layers = cka_np.shape[0]
    layer_labels = [str(idx) for idx in range(1, num_layers + 1)]
    palette = get_palette(2)

    print(f"{Fore.YELLOW}Saving linear CKA visualizations to {output_dir}{Style.RESET_ALL}")

    heatmap_path = output_dir / f"cka_heatmap_{model_1_name}_vs_{model_2_name}.png"
    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(cka_np, vmin=0.0, vmax=1.0, cmap="viridis")
    ax.set_title("Layer-wise Linear CKA Similarity")
    ax.set_xlabel("Reasoning layers")
    ax.set_ylabel("Base layers")
    ax.set_xticks(range(num_layers), layer_labels, rotation=45, ha="right")
    ax.set_yticks(range(num_layers), layer_labels)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Linear CKA similarity")
    fig.tight_layout()
    fig.savefig(heatmap_path, bbox_inches="tight")
    plt.close(fig)

    diagonal_path = output_dir / f"cka_diagonal_{model_1_name}_vs_{model_2_name}.png"
    diagonal_scores = cka_np.diagonal()
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    ax2.plot(range(1, num_layers + 1), diagonal_scores, marker="o", color=palette[0])
    ax2.set_ylim(0.0, 1.0)
    ax2.set_xticks(range(1, num_layers + 1), layer_labels, rotation=45, ha="right")
    ax2.set_xlabel("Layer index")
    ax2.set_ylabel("Linear CKA similarity")
    ax2.set_title("Matching-layer CKA")
    fig2.tight_layout()
    fig2.savefig(diagonal_path, bbox_inches="tight")
    plt.close(fig2)

    print(f"{Fore.GREEN}Saved linear CKA heatmap to {heatmap_path}{Style.RESET_ALL}")
    print(f"{Fore.GREEN}Saved linear CKA diagonal plot to {diagonal_path}{Style.RESET_ALL}")
    return heatmap_path, diagonal_path


def write_statistics(cka_matrix: torch.Tensor, plot_path: Path):
    """Write summary statistics for the CKA scores."""
    stats_path = plot_path.with_suffix('.txt')
    diagonal_scores = torch.diagonal(cka_matrix.detach().cpu().to(torch.float32))
    mean_diag = float(diagonal_scores.mean().item())
    max_diag = float(diagonal_scores.max().item())

    with open(stats_path, 'w') as handle:
        handle.write("Linear CKA SUMMARY STATISTICS\n")
        handle.write("=" * 80 + "\n")
        handle.write(f"Mean diagonal CKA: {mean_diag:.4f}\n")
        handle.write(f"Max diagonal CKA: {max_diag:.4f}\n")
        handle.write("\nLayer-wise Linear CKA (matching layers):\n")
        num_layers = int(diagonal_scores.shape[0])
        for idx in range(num_layers):
            layer_label = num_layers - idx
            handle.write(f"Layer {layer_label}: {float(diagonal_scores[idx].item()):.4f}\n")

    print(f"{Fore.YELLOW}\n" + "=" * 80 + f"{Style.RESET_ALL}")
    print("Linear CKA SUMMARY STATISTICS")
    print("=" * 80)
    print(f"{Fore.CYAN}Mean diagonal CKA: {mean_diag:.4f}{Style.RESET_ALL}")
    print(f"{Fore.CYAN}Max diagonal CKA: {max_diag:.4f}{Style.RESET_ALL}")
    print("=" * 80)
    print(f"\n{Fore.GREEN}Done! Plot saved to {plot_path}{Style.RESET_ALL}")