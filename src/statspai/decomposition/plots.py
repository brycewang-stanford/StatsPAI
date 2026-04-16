"""
Plots for decomposition analysis.

Every plot function returns (fig, ax) (or (fig, axes) for multi-axis)
so the user can further customize. All functions degrade gracefully
when matplotlib is unavailable.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

import numpy as np
import pandas as pd

if TYPE_CHECKING:
    import matplotlib.pyplot as plt


def _require_mpl():
    try:
        import matplotlib.pyplot as plt
        return plt
    except ImportError as err:
        raise ImportError(
            "matplotlib is required for plotting. "
            "Install with: pip install matplotlib"
        ) from err


# ════════════════════════════════════════════════════════════════════════
# Generic waterfall for detailed decompositions
# ════════════════════════════════════════════════════════════════════════

def detailed_waterfall(
    df: pd.DataFrame,
    value_col: str = "contribution",
    label_col: str = "variable",
    title: str = "Decomposition",
    figsize=(8, 5),
    color_pos: str = "#2196F3",
    color_neg: str = "#FF5722",
):
    """Horizontal waterfall bar chart of per-variable contributions."""
    plt = _require_mpl()
    data = df.copy().sort_values(value_col)
    colors = [color_pos if v >= 0 else color_neg for v in data[value_col]]
    fig, ax = plt.subplots(figsize=figsize)
    ax.barh(data[label_col].astype(str), data[value_col],
            color=colors, edgecolor="white")
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel(value_col)
    ax.set_title(title)
    fig.tight_layout()
    return fig, ax


# ════════════════════════════════════════════════════════════════════════
# DFL plot (overall decomposition as horizontal bars)
# ════════════════════════════════════════════════════════════════════════

def dfl_plot(result, figsize=(7, 4)):
    """Summary bar chart: gap, composition, structure."""
    plt = _require_mpl()
    labels = ["Total gap", "Composition", "Structure"]
    values = [result.gap, result.composition, result.structure]
    errs = None
    if result.se is not None:
        errs = [result.se.get("gap", 0),
                result.se.get("composition", 0),
                result.se.get("structure", 0)]
    fig, ax = plt.subplots(figsize=figsize)
    colors = ["#455A64", "#2196F3", "#FF5722"]
    ax.bar(labels, values, yerr=errs, color=colors, capsize=4,
           edgecolor="white")
    ax.axhline(0, color="black", linewidth=0.8)
    name = result.stat
    if result.stat == "quantile":
        name = f"quantile(τ={result.tau})"
    ax.set_title(f"DFL Decomposition — {name}")
    fig.tight_layout()
    return fig, ax


# ════════════════════════════════════════════════════════════════════════
# FFL waterfall (detailed composition + structure side by side)
# ════════════════════════════════════════════════════════════════════════

def ffl_waterfall(result, figsize=(10, 6)):
    """Two-panel bar chart: composition (left) vs structure (right)."""
    plt = _require_mpl()
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=False)
    comp = result.detailed_composition.sort_values("composition")
    struct = result.detailed_structure.sort_values("structure")
    axes[0].barh(comp["variable"], comp["composition"],
                 color=["#2196F3" if v >= 0 else "#FF5722" for v in comp["composition"]])
    axes[0].set_title("Composition (X effect)")
    axes[0].axvline(0, color="black", linewidth=0.8)
    axes[1].barh(struct["variable"], struct["structure"],
                 color=["#4CAF50" if v >= 0 else "#FF9800" for v in struct["structure"]])
    axes[1].set_title("Structure (β effect)")
    axes[1].axvline(0, color="black", linewidth=0.8)
    name = result.stat
    if result.stat == "quantile":
        name = f"quantile(τ={result.tau})"
    fig.suptitle(f"FFL Detailed Decomposition — {name}")
    fig.tight_layout()
    return fig, axes


# ════════════════════════════════════════════════════════════════════════
# Quantile process plot (Machado-Mata / Melly / CFM)
# ════════════════════════════════════════════════════════════════════════

def quantile_process_plot(
    result, figsize=(9, 5), show_gap: bool = True,
):
    """Plot gap, composition, structure as a function of τ."""
    plt = _require_mpl()
    g = result.quantile_grid
    fig, ax = plt.subplots(figsize=figsize)
    if show_gap:
        ax.plot(g["tau"], g["gap"], marker="o", label="Total gap",
                color="#263238", linewidth=2)
    ax.plot(g["tau"], g["composition"], marker="s",
            label="Composition", color="#2196F3")
    ax.plot(g["tau"], g["structure"], marker="^",
            label="Structure", color="#FF5722")
    ax.axhline(0, color="black", linewidth=0.5, linestyle="--")
    ax.set_xlabel("τ (quantile)")
    ax.set_ylabel("Effect")
    ax.set_title("Quantile Process Decomposition")
    ax.legend()
    fig.tight_layout()
    return fig, ax


# ════════════════════════════════════════════════════════════════════════
# CDF counterfactual overlay
# ════════════════════════════════════════════════════════════════════════

def counterfactual_cdf_plot(result, figsize=(9, 5)):
    """Overlay observed A, observed B, counterfactual CDFs."""
    plt = _require_mpl()
    c = result.cdf_grid
    fig, ax = plt.subplots(figsize=figsize)
    ax.step(c["y"], c["cdf_a"], label="F_A (observed)", color="#2196F3")
    ax.step(c["y"], c["cdf_b"], label="F_B (observed)", color="#FF5722")
    ax.step(c["y"], c["cdf_cf"], label="F_cf (counterfactual)",
            color="#4CAF50", linestyle="--")
    ax.set_xlabel("y")
    ax.set_ylabel("F(y)")
    ax.set_title("Counterfactual CDF Decomposition (CFM)")
    ax.legend()
    fig.tight_layout()
    return fig, ax


# ════════════════════════════════════════════════════════════════════════
# Inequality subgroup plot
# ════════════════════════════════════════════════════════════════════════

def inequality_subgroup_plot(result, figsize=(8, 5)):
    """Stacked bar: between + within per group contribution."""
    plt = _require_mpl()
    fig, ax = plt.subplots(figsize=figsize)
    labels = ["Between", "Within"]
    values = [result.between, result.within]
    if result.overlap is not None:
        labels.append("Overlap")
        values.append(result.overlap)
    colors = ["#2196F3", "#FF5722", "#9C27B0"][: len(labels)]
    ax.bar(labels, values, color=colors, edgecolor="white")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(f"Subgroup Decomposition — {result.index}")
    fig.tight_layout()
    return fig, ax


# ════════════════════════════════════════════════════════════════════════
# Gap closing plot
# ════════════════════════════════════════════════════════════════════════

def gap_closing_plot(result, figsize=(7, 4)):
    """Show observed gap, counterfactual gap, and closed portion."""
    plt = _require_mpl()
    fig, ax = plt.subplots(figsize=figsize)
    ax.bar(["Observed", "Counterfactual", "Closed"],
           [result.observed_gap, result.counterfactual_gap, result.closed_gap],
           color=["#455A64", "#4CAF50", "#2196F3"], edgecolor="white")
    ax.axhline(0, color="black", linewidth=0.8)
    ax.set_title(f"Gap Closing — {result.method.upper()}")
    fig.tight_layout()
    return fig, ax


# ════════════════════════════════════════════════════════════════════════
# RIF contribution heatmap (variable × quantile)
# ════════════════════════════════════════════════════════════════════════

def rif_heatmap(
    grid_df: pd.DataFrame,
    variable_col: str = "variable",
    tau_col: str = "tau",
    value_col: str = "contribution",
    figsize=(9, 5),
    cmap: str = "RdBu_r",
):
    """
    Heatmap of RIF/FFL contributions across variables and quantiles.

    `grid_df` should be a long-format DataFrame with columns for
    variable, tau, and contribution.
    """
    plt = _require_mpl()
    pivot = grid_df.pivot(index=variable_col, columns=tau_col, values=value_col)
    fig, ax = plt.subplots(figsize=figsize)
    lim = max(abs(pivot.to_numpy()).max(), 1e-6)
    im = ax.imshow(pivot.to_numpy(), cmap=cmap, aspect="auto",
                   vmin=-lim, vmax=lim)
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{t:.2f}" for t in pivot.columns], rotation=45)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels(pivot.index)
    ax.set_xlabel("τ")
    ax.set_ylabel("Variable")
    ax.set_title("RIF contribution heatmap")
    fig.colorbar(im, ax=ax, shrink=0.8)
    fig.tight_layout()
    return fig, ax
