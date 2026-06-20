"""
Publication-quality plots for :mod:`sp.decomposition`.

Every plot returns ``(fig, ax)`` (or ``(fig, axes)`` for multi-axis
layouts) so callers can post-edit titles, legends, labels and saving.
A unified palette and minimalist style live in
:mod:`statspai.decomposition._results` (``DECOMP_PALETTE``,
``apply_decomp_style``) so that figures from different methods look
like part of the same family.

Plots that take a result object accept anything that exposes the
relevant attributes (``overall``, ``detailed``, ``quantile_grid``,
``cdf_grid``, ``observed_gap`` / ``counterfactual_gap`` / ``closed_gap``,
etc.). When error / SE columns are present we render 95% confidence
whiskers; otherwise the bars are unadorned.

Functions degrade gracefully when matplotlib is unavailable: they raise
a clear ``ImportError`` rather than a deep import-time failure.
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from ._results import DECOMP_PALETTE, apply_decomp_style


def _require_mpl() -> Any:
    try:
        import matplotlib.pyplot as plt

        return plt
    except ImportError as err:  # pragma: no cover
        raise ImportError(
            "matplotlib is required for plotting. "
            "Install with: pip install matplotlib"
        ) from err


def _ci_whiskers(
    values: Sequence[float], ses: Sequence[float], alpha: float = 0.05
) -> Optional[np.ndarray]:
    """Return symmetric whisker half-widths (z * SE) or None."""
    ses_arr = np.asarray(ses, dtype=float)
    if not np.any(ses_arr > 0):
        return None
    z = float(stats.norm.ppf(1 - alpha / 2))
    return z * ses_arr


# ════════════════════════════════════════════════════════════════════════
# Generic waterfall (per-variable contributions, signed-colour bars)
# ════════════════════════════════════════════════════════════════════════


def detailed_waterfall(
    df: pd.DataFrame,
    *,
    value_col: str = "contribution",
    label_col: str = "variable",
    se_col: Optional[str] = "se",
    title: str = "Decomposition",
    figsize: tuple[float, float] = (8, 5),
    alpha: float = 0.05,
) -> tuple[Any, Any]:
    """Horizontal bar chart of per-variable contributions with optional 95% CI.

    Parameters
    ----------
    df : DataFrame
        Per-variable table; must contain ``value_col`` and ``label_col``.
    se_col : str or None
        Standard-error column. Whiskers are drawn at ``±z·se``; if
        absent or zero, bars are unadorned.
    """
    plt = _require_mpl()
    data = df.copy().sort_values(value_col)
    colors = [
        DECOMP_PALETTE["pos"] if v >= 0 else DECOMP_PALETTE["neg"]
        for v in data[value_col]
    ]
    fig, ax = plt.subplots(figsize=figsize)
    whisker = None
    if se_col and se_col in data.columns:
        whisker = _ci_whiskers(data[value_col], data[se_col], alpha)
    ax.barh(
        data[label_col].astype(str),
        data[value_col],
        color=colors,
        edgecolor="white",
        zorder=2,
        xerr=whisker if whisker is not None else None,
        error_kw=dict(ecolor=DECOMP_PALETTE["ci"], capsize=3, lw=1.0),
    )
    ax.axvline(0, color="black", linewidth=0.8, zorder=3)
    ax.set_xlabel(value_col.replace("_", " ").title())
    ax.set_title(title)
    apply_decomp_style(ax)
    fig.tight_layout()
    return fig, ax


# ════════════════════════════════════════════════════════════════════════
# Forest plot (variable estimates with CIs and a zero reference line)
# ════════════════════════════════════════════════════════════════════════


def forest_plot(
    df: pd.DataFrame,
    *,
    value_col: str = "contribution",
    label_col: str = "variable",
    se_col: str = "se",
    title: str = "Decomposition",
    figsize: tuple[float, float] = (8, 5),
    alpha: float = 0.05,
) -> tuple[Any, Any]:
    """Forest-style plot: point estimates + 95% CI per variable.

    Useful for emphasising which contributions are statistically
    different from zero. Significant rows (CI not crossing zero) are
    coloured by sign; non-significant rows are grey.
    """
    plt = _require_mpl()
    data = df.copy().sort_values(value_col).reset_index(drop=True)
    z = float(stats.norm.ppf(1 - alpha / 2))
    se = (
        data[se_col].to_numpy(dtype=float)
        if se_col in data.columns
        else np.zeros(len(data))
    )
    val = data[value_col].to_numpy(dtype=float)
    lo, hi = val - z * se, val + z * se

    sig = (lo > 0) | (hi < 0)
    colors = [
        (
            DECOMP_PALETTE["pos"]
            if (s and v >= 0)
            else DECOMP_PALETTE["neg"] if (s and v < 0) else DECOMP_PALETTE["ci"]
        )
        for s, v in zip(sig, val)
    ]

    fig, ax = plt.subplots(figsize=figsize)
    y = np.arange(len(data))
    ax.hlines(y, lo, hi, colors=colors, linewidth=2, zorder=2)
    ax.scatter(val, y, c=colors, s=42, zorder=3, edgecolors="white", linewidth=0.8)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--", zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels(data[label_col].astype(str))
    ax.set_xlabel(value_col.replace("_", " ").title())
    ax.set_title(title)
    apply_decomp_style(ax)
    fig.tight_layout()
    return fig, ax


# ════════════════════════════════════════════════════════════════════════
# Aggregate decomposition bar chart (gap / composition / structure)
# ════════════════════════════════════════════════════════════════════════


def dfl_plot(
    result: Any,
    figsize: tuple[float, float] = (7, 4),
) -> tuple[Any, Any]:
    """Summary bar chart: gap, composition, structure with 95% CI whiskers."""
    plt = _require_mpl()
    labels = ["Total gap", "Composition", "Structure"]
    values = [result.gap, result.composition, result.structure]
    se_dict = getattr(result, "se", None) or {}
    z = float(stats.norm.ppf(0.975))
    errs = [
        z * se_dict.get("gap", 0.0),
        z * se_dict.get("composition", 0.0),
        z * se_dict.get("structure", 0.0),
    ]
    yerr = errs if any(errs) else None
    fig, ax = plt.subplots(figsize=figsize)
    colors = [DECOMP_PALETTE["accent"], DECOMP_PALETTE["a"], DECOMP_PALETTE["b"]]
    ax.bar(
        labels,
        values,
        yerr=yerr,
        color=colors,
        capsize=4,
        edgecolor="white",
        error_kw=dict(ecolor=DECOMP_PALETTE["ci"]),
    )
    ax.axhline(0, color="black", linewidth=0.8)
    name = getattr(result, "stat", "mean")
    if name == "quantile":
        name = f"quantile(τ={getattr(result, 'tau', 0.5):.2f})"
    ax.set_title(f"DFL Decomposition — {name}")
    apply_decomp_style(ax)
    fig.tight_layout()
    return fig, ax


# ════════════════════════════════════════════════════════════════════════
# FFL waterfall (composition + structure side by side)
# ════════════════════════════════════════════════════════════════════════


def ffl_waterfall(
    result: Any,
    figsize: tuple[float, float] = (11, 6),
) -> tuple[Any, Any]:
    """Two-panel forest chart: composition (left) vs structure (right)."""
    plt = _require_mpl()
    fig, axes = plt.subplots(1, 2, figsize=figsize, sharey=False)
    comp = result.detailed_composition.sort_values("composition")
    struct = result.detailed_structure.sort_values("structure")

    z = float(stats.norm.ppf(0.975))
    comp_se = comp["se"].to_numpy() if "se" in comp.columns else np.zeros(len(comp))
    struct_se = (
        struct["se"].to_numpy() if "se" in struct.columns else np.zeros(len(struct))
    )

    axes[0].barh(
        comp["variable"],
        comp["composition"],
        color=[
            DECOMP_PALETTE["pos"] if v >= 0 else DECOMP_PALETTE["neg"]
            for v in comp["composition"]
        ],
        xerr=z * comp_se if comp_se.any() else None,
        error_kw=dict(ecolor=DECOMP_PALETTE["ci"], capsize=3),
        zorder=2,
    )
    axes[0].set_title("Composition (X effect)")
    axes[0].axvline(0, color="black", linewidth=0.8, zorder=3)

    axes[1].barh(
        struct["variable"],
        struct["structure"],
        color=[
            DECOMP_PALETTE["a"] if v >= 0 else DECOMP_PALETTE["b"]
            for v in struct["structure"]
        ],
        xerr=z * struct_se if struct_se.any() else None,
        error_kw=dict(ecolor=DECOMP_PALETTE["ci"], capsize=3),
        zorder=2,
    )
    axes[1].set_title("Structure (β effect)")
    axes[1].axvline(0, color="black", linewidth=0.8, zorder=3)

    for ax in axes:
        apply_decomp_style(ax)
    name = getattr(result, "stat", "mean")
    if name == "quantile":
        name = f"quantile(τ={getattr(result, 'tau', 0.5):.2f})"
    fig.suptitle(f"FFL Detailed Decomposition — {name}")
    fig.tight_layout()
    return fig, axes


# ════════════════════════════════════════════════════════════════════════
# Quantile process plot (Machado-Mata / Melly / CFM / RIF over τ)
# ════════════════════════════════════════════════════════════════════════


def quantile_process_plot(
    result: Any,
    *,
    figsize: tuple[float, float] = (9, 5),
    show_gap: bool = True,
    show_ci: bool = True,
    alpha: float = 0.05,
) -> tuple[Any, Any]:
    """Plot total gap, composition and structure as functions of τ.

    If the result's ``quantile_grid`` carries SE columns
    (``gap_se`` / ``composition_se`` / ``structure_se``) we shade the
    95% normal-approx CI bands.
    """
    plt = _require_mpl()
    g = result.quantile_grid
    z = float(stats.norm.ppf(1 - alpha / 2))
    fig, ax = plt.subplots(figsize=figsize)

    series = []
    if show_gap and "gap" in g.columns:
        series.append(("gap", "Total gap", DECOMP_PALETTE["accent"], "o"))
    if "composition" in g.columns:
        series.append(("composition", "Composition", DECOMP_PALETTE["a"], "s"))
    if "structure" in g.columns:
        series.append(("structure", "Structure", DECOMP_PALETTE["b"], "^"))

    for name, label, color, marker in series:
        ax.plot(g["tau"], g[name], marker=marker, label=label, color=color, linewidth=2)
        se_col = f"{name}_se"
        if show_ci and se_col in g.columns and (g[se_col] > 0).any():
            ax.fill_between(
                g["tau"],
                g[name] - z * g[se_col],
                g[name] + z * g[se_col],
                color=color,
                alpha=0.18,
                linewidth=0,
            )
    ax.axhline(0, color="black", linewidth=0.6, linestyle="--")
    ax.set_xlabel("τ (quantile)")
    ax.set_ylabel("Effect")
    ax.set_title("Quantile Process Decomposition")
    ax.legend(frameon=False, loc="best")
    apply_decomp_style(ax)
    fig.tight_layout()
    return fig, ax


# ════════════════════════════════════════════════════════════════════════
# Counterfactual CDF (CFM)
# ════════════════════════════════════════════════════════════════════════


def counterfactual_cdf_plot(
    result: Any,
    figsize: tuple[float, float] = (9, 5),
) -> tuple[Any, Any]:
    """Overlay observed A, observed B, and counterfactual CDFs."""
    plt = _require_mpl()
    c = result.cdf_grid
    fig, ax = plt.subplots(figsize=figsize)
    ax.step(
        c["y"],
        c["cdf_a"],
        label="F_A (observed)",
        color=DECOMP_PALETTE["a"],
        linewidth=1.8,
    )
    ax.step(
        c["y"],
        c["cdf_b"],
        label="F_B (observed)",
        color=DECOMP_PALETTE["b"],
        linewidth=1.8,
    )
    ax.step(
        c["y"],
        c["cdf_cf"],
        label="F_cf (counterfactual)",
        color=DECOMP_PALETTE["cf"],
        linewidth=1.8,
        linestyle="--",
    )
    ax.set_xlabel("y")
    ax.set_ylabel("F(y)")
    ax.set_title("Counterfactual CDF Decomposition (CFM)")
    ax.legend(frameon=False)
    apply_decomp_style(ax)
    fig.tight_layout()
    return fig, ax


# ════════════════════════════════════════════════════════════════════════
# Inequality subgroup plot (between / within / overlap)
# ════════════════════════════════════════════════════════════════════════


def inequality_subgroup_plot(
    result: Any,
    figsize: tuple[float, float] = (8, 5),
) -> tuple[Any, Any]:
    """Bar chart of subgroup inequality decomposition."""
    plt = _require_mpl()
    fig, ax = plt.subplots(figsize=figsize)
    labels = ["Between", "Within"]
    values = [result.between, result.within]
    if getattr(result, "overlap", None) is not None:
        labels.append("Overlap")
        values.append(result.overlap)
    colors = [DECOMP_PALETTE[k] for k in ("between", "within", "overlap")]
    colors = colors[: len(labels)]
    ax.bar(labels, values, color=colors, edgecolor="white", zorder=2)
    ax.axhline(0, color="black", linewidth=0.8, zorder=3)
    ax.set_title(f"Subgroup Decomposition — {result.index}")
    apply_decomp_style(ax)
    fig.tight_layout()
    return fig, ax


# ════════════════════════════════════════════════════════════════════════
# Gap closing — observed vs counterfactual vs closed (with CI whiskers)
# ════════════════════════════════════════════════════════════════════════


def gap_closing_plot(
    result: Any,
    figsize: tuple[float, float] = (7, 4),
    alpha: float = 0.05,
) -> tuple[Any, Any]:
    """Bars for observed, counterfactual, and closed gap with whiskers."""
    plt = _require_mpl()
    fig, ax = plt.subplots(figsize=figsize)
    se = getattr(result, "se", None) or {}
    z = float(stats.norm.ppf(1 - alpha / 2))
    values = [result.observed_gap, result.counterfactual_gap, result.closed_gap]
    errs = [
        z * se.get("observed", 0.0),
        z * se.get("counterfactual", 0.0),
        z * se.get("closed", 0.0),
    ]
    yerr = errs if any(errs) else None
    ax.bar(
        ["Observed", "Counterfactual", "Closed"],
        values,
        yerr=yerr,
        color=[DECOMP_PALETTE["accent"], DECOMP_PALETTE["cf"], DECOMP_PALETTE["a"]],
        edgecolor="white",
        capsize=4,
        zorder=2,
        error_kw=dict(ecolor=DECOMP_PALETTE["ci"]),
    )
    ax.axhline(0, color="black", linewidth=0.8, zorder=3)
    ax.set_title(f"Gap Closing — {result.method.upper()}")
    apply_decomp_style(ax)
    fig.tight_layout()
    return fig, ax


# ════════════════════════════════════════════════════════════════════════
# Mediation forest (NDE / NIE / total + their CIs)
# ════════════════════════════════════════════════════════════════════════


def mediation_forest(
    result: Any,
    *,
    figsize: tuple[float, float] = (7, 3.5),
    alpha: float = 0.05,
) -> tuple[Any, Any]:
    """Forest plot of NDE / NIE / total effect with 95% CI."""
    plt = _require_mpl()
    z = float(stats.norm.ppf(1 - alpha / 2))
    rows = []
    for label, key in (
        ("Total", "total_effect"),
        ("NDE", "nde"),
        ("NIE", "nie"),
    ):
        v = getattr(result, key, None)
        if v is None:
            continue
        se_dict = getattr(result, "se", None) or {}
        s = float(se_dict.get(key, 0.0))
        rows.append((label, float(v), s))
    if not rows:
        raise ValueError("MediationDecompResult has no NDE/NIE/total effect.")
    labels, vals_raw, ses_raw = zip(*rows)
    vals = np.asarray(vals_raw, dtype=float)
    ses = np.asarray(ses_raw, dtype=float)
    fig, ax = plt.subplots(figsize=figsize)
    y = np.arange(len(labels))
    lo = vals - z * ses
    hi = vals + z * ses
    sig = (lo > 0) | (hi < 0)
    colors = [
        (
            DECOMP_PALETTE["pos"]
            if (s and v >= 0)
            else DECOMP_PALETTE["neg"] if (s and v < 0) else DECOMP_PALETTE["ci"]
        )
        for s, v in zip(sig, vals)
    ]
    ax.hlines(y, lo, hi, colors=colors, linewidth=2)
    ax.scatter(vals, y, c=colors, s=42, edgecolors="white", linewidth=0.8, zorder=3)
    ax.axvline(0, color="black", linewidth=0.8, linestyle="--", zorder=1)
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.set_xlabel("Effect")
    ax.set_title("Mediation Decomposition")
    apply_decomp_style(ax)
    fig.tight_layout()
    return fig, ax


# ════════════════════════════════════════════════════════════════════════
# RIF contribution heatmap (variable × quantile)
# ════════════════════════════════════════════════════════════════════════


def rif_heatmap(
    grid_df: pd.DataFrame,
    *,
    variable_col: str = "variable",
    tau_col: str = "tau",
    value_col: str = "contribution",
    figsize: tuple[float, float] = (9, 5),
    cmap: str = "RdBu_r",
) -> tuple[Any, Any]:
    """Heatmap of per-variable RIF contributions across quantiles.

    ``grid_df`` is expected in long form with three columns: variable,
    tau, contribution.
    """
    plt = _require_mpl()
    pivot = grid_df.pivot(index=variable_col, columns=tau_col, values=value_col)
    fig, ax = plt.subplots(figsize=figsize)
    lim = max(abs(pivot.to_numpy()).max(), 1e-6)
    im = ax.imshow(pivot.to_numpy(), cmap=cmap, aspect="auto", vmin=-lim, vmax=lim)
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


# ════════════════════════════════════════════════════════════════════════
# Yu-Elwert mechanism plot (prevalence / impact / selection)
# ════════════════════════════════════════════════════════════════════════


def yu_elwert_mechanisms_plot(
    result: Any,
    *,
    figsize: tuple[float, float] = (8, 4.5),
    alpha: float = 0.05,
) -> tuple[Any, Any]:
    """Bar chart of the three Yu-Elwert (2025) decomposition mechanisms.

    Bars: total disparity, prevalence, treatment-effect, selection,
    plus the residual baseline. Whiskers are normal-approx 95% CIs.
    """
    plt = _require_mpl()
    z = float(stats.norm.ppf(1 - alpha / 2))
    components = [
        ("Disparity", "disparity"),
        ("Baseline", "baseline"),
        ("Prevalence", "prevalence"),
        ("Effect", "effect"),
        ("Selection", "selection"),
    ]
    labels: list[str] = []
    vals: list[float] = []
    errs: list[float] = []
    se_dict = getattr(result, "se", None) or {}
    for label, key in components:
        v = getattr(result, key, None)
        if v is None:
            continue
        labels.append(label)
        vals.append(float(v))
        s = float(se_dict.get(key, 0.0))
        errs.append(z * s)
    yerr = errs if any(errs) else None
    fig, ax = plt.subplots(figsize=figsize)
    colors = [
        DECOMP_PALETTE["accent"],
        DECOMP_PALETTE["ci"],
        DECOMP_PALETTE["a"],
        DECOMP_PALETTE["b"],
        DECOMP_PALETTE["cf"],
    ][: len(labels)]
    ax.bar(
        labels,
        vals,
        yerr=yerr,
        color=colors,
        edgecolor="white",
        capsize=4,
        zorder=2,
        error_kw=dict(ecolor=DECOMP_PALETTE["ci"]),
    )
    ax.axhline(0, color="black", linewidth=0.8, zorder=3)
    ax.set_title("Yu–Elwert (2025) Causal Decomposition")
    ax.set_ylabel("Contribution to disparity")
    apply_decomp_style(ax)
    fig.tight_layout()
    return fig, ax
