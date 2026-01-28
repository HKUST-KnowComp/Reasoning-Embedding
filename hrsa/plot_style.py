#!/usr/bin/env python3
"""
Shared plotting style utilities to keep figures consistent across evaluation scripts.
"""
from __future__ import annotations

from typing import Iterable, List

import matplotlib.pyplot as plt

_BASE_RC = {
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.titlesize": 13,
    "axes.labelsize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 160,
    "savefig.dpi": 200,
    "legend.frameon": False,
}

_DEFAULT_STYLE = "seaborn-v0_8-whitegrid"
# _PALETTE = ["#0D3B66", "#007F5F", "#B5179E", "#F77F00", "#219EBC", "#EE6C4D"]

_PALETTE = ["#0f52ba", "#52d137", "#ffd700", "#B5179E", "#F77F00", "#22a9c9"]
_STYLE_APPLIED = False


def apply_plot_style() -> None:
    """Apply a consistent set of matplotlib rcParams once per process."""
    global _STYLE_APPLIED
    if _STYLE_APPLIED:
        return
    try:
        plt.style.use(_DEFAULT_STYLE)
    except OSError:
        plt.style.use("seaborn-whitegrid")
    plt.rcParams.update(_BASE_RC)
    _STYLE_APPLIED = True


def get_palette(count: int | None = None) -> List[str]:
    """
    Return a color palette with at least ``count`` colors.

    Args:
        count: Minimum number of colors required. If None, returns the base palette.
    """
    if count is None or count <= len(_PALETTE):
        return _PALETTE[:count] if count else list(_PALETTE)

    repeats = (count + len(_PALETTE) - 1) // len(_PALETTE)
    extended: List[str] = (_PALETTE * repeats)[:count]
    return extended

