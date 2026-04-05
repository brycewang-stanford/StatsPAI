"""
Unified plotting for all Synthetic Control variants.

``synthplot(result, type=...)`` auto-detects the SCM variant from
``model_info`` and dispatches to the appropriate visualisation.

Plot types
----------
* **trajectory** — treated vs synthetic outcome over time (all variants)
* **gap** — treatment effect (gap) over time (all variants)
* **both** — two-panel: trajectory on top, gap on bottom
* **weights** — horizontal bar chart of donor weights
* **placebo** — placebo distribution + treated rank
* **conformal** — period-level effects with conformal CI bands
* **staggered** — cohort-specific ATT comparison
* **factors** — latent factor loadings (gsynth only)
* **compare** — overlay multiple ``CausalResult`` objects
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from ..core.results import CausalResult


# ====================================================================== #
#  Colour palette — publication-quality defaults
# ====================================================================== #

_PALETTE = {
    "treated": "#2C3E50",
    "synthetic": "#E74C3C",
    "gap_fill": "#3498DB",
    "treatment_line": "#95A5A6",
    "placebo": "#BDC3C7",
    "ci_band": "#E74C3C",
    "bar": "#3498DB",
    "cohort_colors": [
        "#E74C3C", "#3498DB", "#2ECC71", "#9B59B6",
        "#F39C12", "#1ABC9C", "#E67E22", "#34495E",
    ],
}


# ====================================================================== #
#  Main entry point
# ====================================================================== #

def synthplot(
    result: Union[CausalResult, List[CausalResult]],
    type: str = "trajectory",
    ax=None,
    figsize: Optional[tuple] = None,
    title: Optional[str] = None,
    top_n: int = 15,
    labels: Optional[List[str]] = None,
    **kwargs,
):
    """
    Unified plot function for all Synthetic Control variants.

    Automatically detects the SCM variant and renders the appropriate
    visualisation. Works with results from ``synth(method=...)``,
    ``sdid()``, ``augsynth()``, ``gsynth()``, ``staggered_synth()``,
    ``conformal_synth()``, and all other variants.

    Parameters
    ----------
    result : CausalResult or list of CausalResult
        Output of any ``synth()`` variant. Pass a list for ``type='compare'``.
    type : str, default 'trajectory'
        Plot type:

        * ``'trajectory'`` — treated vs synthetic over time.
        * ``'gap'`` — effect (gap) over time.
        * ``'both'`` — two-panel: trajectory + gap.
        * ``'weights'`` — donor weight bar chart.
        * ``'placebo'`` — placebo ATT distribution.
        * ``'placebo_gap'`` — placebo gap spaghetti plot.
        * ``'conformal'`` — period-level effects + conformal CIs.
        * ``'staggered'`` — cohort-level ATT comparison.
        * ``'factors'`` — latent factor loadings (gsynth only).
        * ``'compare'`` — overlay multiple results.
    ax : matplotlib Axes, optional
        Pre-existing axes for single-panel plots.
    figsize : tuple, optional
        Figure size. Auto-selected if None.
    title : str, optional
        Override the auto-generated title.
    top_n : int, default 15
        Number of donors to show in weight plots.
    labels : list of str, optional
        Labels for ``type='compare'``.
    **kwargs
        Additional arguments passed to individual plotters.

    Returns
    -------
    (fig, ax) or (fig, axes)

    Examples
    --------
    >>> result = sp.synth(df, ..., method='demeaned')
    >>> sp.synthplot(result)                    # trajectory
    >>> sp.synthplot(result, type='gap')        # gap plot
    >>> sp.synthplot(result, type='both')       # two-panel
    >>> sp.synthplot(result, type='weights')    # donor weights
    >>> sp.synthplot(result, type='placebo')    # placebo distribution

    Compare methods:

    >>> r1 = sp.synth(df, ..., method='classic')
    >>> r2 = sp.synth(df, ..., method='demeaned')
    >>> r3 = sp.synth(df, ..., method='sdid')
    >>> sp.synthplot([r1, r2, r3], type='compare',
    ...             labels=['Classic', 'De-meaned', 'SDID'])
    """
    _ensure_matplotlib()

    type_ = type.lower().strip()

    # --- Multi-result: compare ---
    if type_ == "compare":
        results = result if isinstance(result, list) else [result]
        return _plot_compare(results, labels=labels, ax=ax,
                             figsize=figsize or (11, 7), title=title, **kwargs)

    # Single result from here
    if isinstance(result, list):
        result = result[0]

    mi = result.model_info

    # --- Dispatch ---
    if type_ == "both":
        return _plot_both(result, figsize=figsize or (10, 9),
                          title=title, **kwargs)
    if type_ == "trajectory":
        return _plot_trajectory(result, ax=ax,
                                figsize=figsize or (10, 6.5),
                                title=title, **kwargs)
    if type_ == "gap":
        return _plot_gap(result, ax=ax, figsize=figsize or (10, 5),
                         title=title, **kwargs)
    if type_ == "weights":
        return _plot_weights(result, top_n=top_n, ax=ax,
                             figsize=figsize or (8, 5), title=title, **kwargs)
    if type_ in ("placebo", "placebo_dist"):
        return _plot_placebo(result, ax=ax, figsize=figsize or (9, 6),
                             title=title, **kwargs)
    if type_ == "placebo_gap":
        return _plot_placebo_gap(result, ax=ax,
                                 figsize=figsize or (10, 6),
                                 title=title, **kwargs)
    if type_ == "conformal":
        return _plot_conformal(result, ax=ax,
                               figsize=figsize or (10, 6),
                               title=title, **kwargs)
    if type_ == "staggered":
        return _plot_staggered(result, ax=ax,
                               figsize=figsize or (9, 6),
                               title=title, **kwargs)
    if type_ in ("factors", "loadings"):
        return _plot_factors(result, ax=ax,
                             figsize=figsize or (10, 5),
                             title=title, **kwargs)

    raise ValueError(
        f"Unknown plot type {type_!r}. Choose from: 'trajectory', 'gap', "
        f"'both', 'weights', 'placebo', 'placebo_gap', 'conformal', "
        f"'staggered', 'factors', 'compare'."
    )


# ====================================================================== #
#  Data extraction — normalise model_info across all variants
# ====================================================================== #

def _extract_trajectories(result: CausalResult):
    """
    Extract (times, Y_treated, Y_synth, treatment_time, label) from
    any SCM variant's CausalResult.
    """
    mi = result.model_info

    # --- Classic / demeaned / robust / conformal ---
    if "gap_table" in mi:
        gt = mi["gap_table"]
        return (
            gt["time"].values,
            gt["treated"].values,
            gt["synthetic"].values,
            mi.get("treatment_time"),
            mi.get("treated_unit", "Treated"),
        )

    # --- SDID ---
    if "Y_obs" in mi and isinstance(mi["Y_obs"], pd.Series):
        return (
            np.array(mi["all_times"]),
            mi["Y_obs"].values,
            mi["Y_synth"].values,
            mi.get("treat_time"),
            str(mi.get("treated_units", ["Treated"])[0])
            if isinstance(mi.get("treated_units"), list)
            else "Treated",
        )

    # --- GSynth ---
    if "trajectory" in mi:
        tj = mi["trajectory"]
        return (
            tj["time"].values if hasattr(tj["time"], "values") else np.array(tj["time"]),
            tj["treated"].values if hasattr(tj["treated"], "values") else np.array(tj["treated"]),
            tj["synthetic"].values if hasattr(tj["synthetic"], "values") else np.array(tj["synthetic"]),
            mi.get("treatment_time"),
            mi.get("treated_unit", "Treated"),
        )

    # --- Augmented ---
    if "effects_by_period" in mi and "Y_synth" not in mi:
        ep = mi["effects_by_period"]
        return (
            ep["time"].values,
            ep["treated"].values,
            ep["counterfactual"].values,
            mi.get("treatment_time"),
            mi.get("treated_unit", "Treated"),
        )

    # --- Fallback: raw arrays ---
    if "Y_treated" in mi and "Y_synth" in mi:
        return (
            np.array(mi["times"]),
            np.asarray(mi["Y_treated"]),
            np.asarray(mi["Y_synth"]),
            mi.get("treatment_time"),
            mi.get("treated_unit", "Treated"),
        )

    raise ValueError("Cannot extract trajectory data from this result")


def _extract_weights(result: CausalResult):
    """Extract (unit_names, weights) as sorted arrays."""
    mi = result.model_info

    # DataFrame with 'unit' and 'weight' columns
    for key in ("weights", "unit_weights"):
        w = mi.get(key)
        if isinstance(w, pd.DataFrame) and "weight" in w.columns:
            w = w[w["weight"].abs() > 1e-6].copy()
            w = w.sort_values("weight", ascending=False, key=abs)
            return w["unit"].values, w["weight"].values

    # Dict
    if isinstance(mi.get("weights"), dict):
        items = sorted(mi["weights"].items(), key=lambda x: abs(x[1]),
                       reverse=True)
        items = [(k, v) for k, v in items if abs(v) > 1e-6]
        if items:
            return (
                np.array([k for k, v in items]),
                np.array([v for k, v in items]),
            )

    raise ValueError("No weight data found in this result")


# ====================================================================== #
#  Individual plotters
# ====================================================================== #

def _plot_trajectory(result, ax=None, figsize=(10, 6.5), title=None, **kw):
    import matplotlib.pyplot as plt

    times, y_tr, y_syn, t_time, label = _extract_trajectories(result)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    ax.plot(times, y_tr, color=_PALETTE["treated"], linewidth=2.2,
            label=str(label), zorder=3)
    ax.plot(times, y_syn, color=_PALETTE["synthetic"], linewidth=2.2,
            linestyle="--", label="Synthetic", zorder=3)

    # Shade post-treatment gap
    if t_time is not None:
        post = np.array([t >= t_time for t in times])
        if post.any():
            ax.fill_between(
                np.asarray(times)[post], y_tr[post], y_syn[post],
                alpha=0.12, color=_PALETTE["ci_band"],
                label="Treatment effect", zorder=2,
            )
        ax.axvline(x=t_time, color=_PALETTE["treatment_line"],
                   linestyle=":", linewidth=1.2, alpha=0.8, zorder=1)

    # SDID time weight shading
    mi = result.model_info
    if mi.get("estimator") == "sdid" and "time_weights" in mi:
        tw = mi["time_weights"]
        pre_t = mi["pre_times"]
        max_w = tw.max() if tw.max() > 0 else 1
        for t_val, w_val in zip(pre_t, tw.values):
            ax.axvspan(t_val - 0.4, t_val + 0.4,
                       alpha=0.06 * (w_val / max_w), color="blue")

    att = result.estimate
    method_short = _method_short_label(result)
    ax.set_title(title or f"{method_short}: ATT = {att:.3f}", fontsize=13)
    ax.set_xlabel("Time", fontsize=11)
    ax.set_ylabel("Outcome", fontsize=11)
    ax.legend(fontsize=9.5, frameon=False)
    _clean_spines(ax)
    fig.tight_layout()
    return fig, ax


def _plot_gap(result, ax=None, figsize=(10, 5), title=None, **kw):
    import matplotlib.pyplot as plt

    times, y_tr, y_syn, t_time, label = _extract_trajectories(result)
    gap = y_tr - y_syn

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    ax.plot(times, gap, color=_PALETTE["treated"], linewidth=2, zorder=3)
    ax.fill_between(times, 0, gap, alpha=0.12, color=_PALETTE["gap_fill"],
                    zorder=2)
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8, zorder=1)

    if t_time is not None:
        ax.axvline(x=t_time, color=_PALETTE["treatment_line"],
                   linestyle=":", linewidth=1.2, alpha=0.8, zorder=1)

    # Conformal: add per-period CI bands if available
    mi = result.model_info
    if "period_results" in mi:
        pr = mi["period_results"]
        ax.fill_between(
            pr["time"].values,
            pr["ci_lower"].values,
            pr["ci_upper"].values,
            alpha=0.15, color=_PALETTE["ci_band"],
            label=f"{int((1 - result.alpha) * 100)}% Conformal CI",
            zorder=2,
        )
        ax.legend(fontsize=9, frameon=False)

    method_short = _method_short_label(result)
    ax.set_title(
        title or f"{method_short}: Gap (ATT = {result.estimate:.3f})",
        fontsize=13,
    )
    ax.set_xlabel("Time", fontsize=11)
    ax.set_ylabel("Gap (Treated \u2212 Synthetic)", fontsize=11)
    _clean_spines(ax)
    fig.tight_layout()
    return fig, ax


def _plot_both(result, figsize=(10, 9), title=None, **kw):
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 1, figsize=figsize, sharex=True,
                             gridspec_kw={"height_ratios": [1.2, 1]})

    _plot_trajectory(result, ax=axes[0], **kw)
    _plot_gap(result, ax=axes[1], **kw)

    # Remove duplicate x-label on top panel
    axes[0].set_xlabel("")

    method_short = _method_short_label(result)
    if title:
        fig.suptitle(title, fontsize=14, y=1.01)
    fig.tight_layout()
    return fig, axes


def _plot_weights(result, top_n=15, ax=None, figsize=(8, 5), title=None, **kw):
    import matplotlib.pyplot as plt

    names, weights = _extract_weights(result)

    # Limit to top_n
    if len(names) > top_n:
        names = names[:top_n]
        weights = weights[:top_n]

    # Sort ascending for horizontal bar
    order = np.argsort(np.abs(weights))
    names = names[order]
    weights = weights[order]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    colors = [_PALETTE["bar"] if w >= 0 else "#E74C3C" for w in weights]
    ax.barh([str(n) for n in names], weights, color=colors, edgecolor="white",
            linewidth=0.5)
    ax.axvline(x=0, color="gray", linewidth=0.5, zorder=0)
    ax.set_xlabel("Weight", fontsize=11)

    method_short = _method_short_label(result)
    ax.set_title(title or f"Donor Weights ({method_short})", fontsize=13)
    _clean_spines(ax)
    fig.tight_layout()
    return fig, ax


def _plot_placebo(result, ax=None, figsize=(9, 6), title=None, **kw):
    """Placebo ATT distribution: histogram + treated unit marker."""
    import matplotlib.pyplot as plt

    mi = result.model_info
    placebos = mi.get("placebo_atts") or mi.get("placebo_distribution")
    if placebos is None:
        raise ValueError(
            "No placebo data. Run with placebo=True to generate."
        )
    placebos = np.asarray(placebos)
    att = result.estimate

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Histogram
    n_bins = min(max(len(placebos) // 2, 8), 30)
    ax.hist(placebos, bins=n_bins, color=_PALETTE["placebo"],
            edgecolor="white", alpha=0.7, label="Placebo ATTs", zorder=2)

    # Treated unit line
    ax.axvline(x=att, color=_PALETTE["synthetic"], linewidth=2.5,
               linestyle="--", label=f"Treated: {att:.3f}", zorder=3)

    # Rank annotation
    rank = int(np.sum(np.abs(placebos) >= abs(att))) + 1
    total = len(placebos) + 1
    ax.annotate(
        f"Rank: {rank}/{total}\np = {result.pvalue:.3f}",
        xy=(att, ax.get_ylim()[1] * 0.85),
        fontsize=10, fontweight="bold",
        ha="left" if att > np.median(placebos) else "right",
        color=_PALETTE["synthetic"],
    )

    method_short = _method_short_label(result)
    ax.set_title(title or f"Placebo Distribution ({method_short})",
                 fontsize=13)
    ax.set_xlabel("ATT", fontsize=11)
    ax.set_ylabel("Count", fontsize=11)
    ax.legend(fontsize=9.5, frameon=False)
    _clean_spines(ax)
    fig.tight_layout()
    return fig, ax


def _plot_placebo_gap(result, ax=None, figsize=(10, 6), title=None, **kw):
    """
    Spaghetti plot: placebo gap paths overlaid with treated gap.

    Requires the full placebo SCM to have been stored — otherwise
    we fall back to the treated gap only.
    """
    import matplotlib.pyplot as plt

    times, y_tr, y_syn, t_time, label = _extract_trajectories(result)
    gap = y_tr - y_syn
    mi = result.model_info

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # We don't typically store full placebo trajectories, so we show
    # the treated gap prominently with a note about the placebo distribution
    ax.plot(times, gap, color=_PALETTE["treated"], linewidth=2.5,
            label=str(label), zorder=3)

    # If we have placebo ATTs, draw horizontal reference lines
    placebos = mi.get("placebo_atts") or mi.get("placebo_distribution")
    if placebos is not None:
        placebos = np.asarray(placebos)
        # Draw quantile bands
        for q, alpha_ in [(0.05, 0.06), (0.25, 0.06), (0.75, 0.06), (0.95, 0.06)]:
            val = np.quantile(placebos, q)
            ax.axhline(y=val, color=_PALETTE["placebo"], linewidth=0.8,
                       alpha=0.5 + alpha_, zorder=1)
        # Shade interquartile range of placebos
        q25, q75 = np.quantile(placebos, 0.25), np.quantile(placebos, 0.75)
        ax.axhspan(q25, q75, alpha=0.08, color=_PALETTE["placebo"],
                   label="Placebo IQR", zorder=1)

    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8, zorder=1)
    if t_time is not None:
        ax.axvline(x=t_time, color=_PALETTE["treatment_line"],
                   linestyle=":", linewidth=1.2, alpha=0.8, zorder=1)

    method_short = _method_short_label(result)
    ax.set_title(title or f"Gap + Placebo Range ({method_short})",
                 fontsize=13)
    ax.set_xlabel("Time", fontsize=11)
    ax.set_ylabel("Gap", fontsize=11)
    ax.legend(fontsize=9.5, frameon=False)
    _clean_spines(ax)
    fig.tight_layout()
    return fig, ax


def _plot_conformal(result, ax=None, figsize=(10, 6), title=None, **kw):
    """Period-level effects with conformal confidence intervals."""
    import matplotlib.pyplot as plt

    mi = result.model_info
    pr = mi.get("period_results")
    if pr is None:
        raise ValueError(
            "No conformal period_results. Use inference='conformal'."
        )

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    times = pr["time"].values
    effects = pr["effect"].values
    ci_lo = pr["ci_lower"].values
    ci_hi = pr["ci_upper"].values
    pvals = pr["pvalue"].values

    # Effect line + markers coloured by significance
    sig = pvals < result.alpha
    ax.plot(times, effects, color=_PALETTE["treated"], linewidth=2, zorder=3)
    ax.scatter(times[sig], effects[sig], color=_PALETTE["synthetic"],
               s=50, zorder=4, label=f"p < {result.alpha}")
    ax.scatter(times[~sig], effects[~sig], color=_PALETTE["placebo"],
               s=50, zorder=4, label=f"p \u2265 {result.alpha}")

    # CI band
    ax.fill_between(times, ci_lo, ci_hi, alpha=0.15,
                    color=_PALETTE["ci_band"],
                    label=f"{int((1 - result.alpha) * 100)}% Conformal CI")

    ax.axhline(y=0, color="gray", linestyle="--", linewidth=0.8, zorder=1)

    ax.set_title(
        title or f"Conformal Inference: ATT = {result.estimate:.3f}",
        fontsize=13,
    )
    ax.set_xlabel("Time", fontsize=11)
    ax.set_ylabel("Treatment Effect", fontsize=11)
    ax.legend(fontsize=9.5, frameon=False)
    _clean_spines(ax)
    fig.tight_layout()
    return fig, ax


def _plot_staggered(result, ax=None, figsize=(9, 6), title=None, **kw):
    """Cohort-level ATT comparison for staggered adoption."""
    import matplotlib.pyplot as plt

    mi = result.model_info
    cohort_df = mi.get("cohort_effects")
    unit_df = mi.get("unit_effects")

    if cohort_df is None and unit_df is None:
        raise ValueError(
            "No staggered cohort data. Use method='staggered'."
        )

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    if cohort_df is not None and len(cohort_df) > 0:
        colors = _PALETTE["cohort_colors"]
        cohort_times = cohort_df["cohort_time"].values
        cohort_atts = cohort_df["att"].values

        # Bar chart of cohort ATTs
        bar_colors = [colors[i % len(colors)] for i in range(len(cohort_times))]
        bars = ax.bar(
            [str(t) for t in cohort_times], cohort_atts,
            color=bar_colors, edgecolor="white", linewidth=0.5, zorder=3,
        )

        # Add unit-level jitter points if available
        if unit_df is not None:
            for i, (ct, _) in enumerate(zip(cohort_times, cohort_atts)):
                units = unit_df[unit_df["cohort_time"] == ct]
                x_jitter = np.random.default_rng(42).uniform(
                    -0.15, 0.15, len(units)
                )
                ax.scatter(
                    i + x_jitter, units["att"].values,
                    color="white", edgecolor=bar_colors[i],
                    s=40, linewidth=1.2, zorder=4,
                )

        # Overall ATT line
        ax.axhline(y=result.estimate, color=_PALETTE["treated"],
                   linestyle="--", linewidth=1.5, zorder=2,
                   label=f"Overall ATT = {result.estimate:.3f}")

        ax.axhline(y=0, color="gray", linestyle="-", linewidth=0.5, zorder=1)
        ax.set_xlabel("Cohort (adoption time)", fontsize=11)
        ax.set_ylabel("ATT", fontsize=11)
    else:
        # Fallback: just unit-level
        ax.barh(
            unit_df["unit"].astype(str),
            unit_df["att"],
            color=_PALETTE["bar"],
        )
        ax.axvline(x=result.estimate, color=_PALETTE["treated"],
                   linestyle="--", linewidth=1.5,
                   label=f"Overall ATT = {result.estimate:.3f}")
        ax.set_xlabel("ATT", fontsize=11)

    ax.set_title(title or "Staggered SCM: Cohort Effects", fontsize=13)
    ax.legend(fontsize=9.5, frameon=False)
    _clean_spines(ax)
    fig.tight_layout()
    return fig, ax


def _plot_factors(result, ax=None, figsize=(10, 5), title=None, **kw):
    """Latent factor time paths for gsynth."""
    import matplotlib.pyplot as plt

    mi = result.model_info
    F_pre = mi.get("factors_pre")
    F_post = mi.get("factors_post")

    if F_pre is None:
        raise ValueError("No factor data. Use method='gsynth'.")

    pre_times = mi.get("times", [])[:F_pre.shape[0]] if "times" not in mi else mi["times"]
    n_factors = F_pre.shape[1]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    # Combine pre + post factors
    all_times = mi.get("times", np.arange(F_pre.shape[0] + F_post.shape[0]))
    T0 = F_pre.shape[0]

    colors = _PALETTE["cohort_colors"]
    for k in range(n_factors):
        F_all = np.concatenate([F_pre[:, k], F_post[:, k]])
        ax.plot(all_times, F_all, color=colors[k % len(colors)],
                linewidth=1.8, label=f"Factor {k + 1}", zorder=3)

    t_time = mi.get("treatment_time")
    if t_time is not None:
        ax.axvline(x=t_time, color=_PALETTE["treatment_line"],
                   linestyle=":", linewidth=1.2, alpha=0.8, zorder=1)

    ax.set_title(title or f"Latent Factors ({n_factors})", fontsize=13)
    ax.set_xlabel("Time", fontsize=11)
    ax.set_ylabel("Factor Value", fontsize=11)
    ax.legend(fontsize=9.5, frameon=False)
    _clean_spines(ax)
    fig.tight_layout()
    return fig, ax


def _plot_compare(
    results: List[CausalResult],
    labels: Optional[List[str]] = None,
    ax=None,
    figsize=(11, 7),
    title=None,
    **kw,
):
    """Overlay treated vs synthetic for multiple methods."""
    import matplotlib.pyplot as plt

    if labels is None:
        labels = [_method_short_label(r) for r in results]

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    colors = _PALETTE["cohort_colors"]
    t_time = None
    y_treated = None

    for i, (res, lab) in enumerate(zip(results, labels)):
        times, y_tr, y_syn, tt, unit_label = _extract_trajectories(res)

        # Plot treated line only once
        if y_treated is None:
            ax.plot(times, y_tr, color=_PALETTE["treated"], linewidth=2.5,
                    label=str(unit_label), zorder=4)
            y_treated = y_tr
            t_time = tt

        # Plot each synthetic
        c = colors[i % len(colors)]
        att = res.estimate
        ax.plot(times, y_syn, color=c, linewidth=1.8, linestyle="--",
                label=f"{lab} (ATT={att:.2f})", zorder=3)

    if t_time is not None:
        ax.axvline(x=t_time, color=_PALETTE["treatment_line"],
                   linestyle=":", linewidth=1.2, alpha=0.8, zorder=1)

    ax.set_title(title or "Method Comparison", fontsize=13)
    ax.set_xlabel("Time", fontsize=11)
    ax.set_ylabel("Outcome", fontsize=11)
    ax.legend(fontsize=9, frameon=False, loc="best")
    _clean_spines(ax)
    fig.tight_layout()
    return fig, ax


# ====================================================================== #
#  Helpers
# ====================================================================== #

def _method_short_label(result: CausalResult) -> str:
    """Extract a short method label from the result."""
    mi = result.model_info
    # Check common keys
    if mi.get("variant"):
        v = mi["variant"]
        labels = {
            "demeaned": "De-meaned SCM",
            "detrended": "De-trended SCM",
            "unconstrained": "Unconstrained SCM",
            "elastic_net": "Elastic-Net SCM",
            "penalized": "Penalized SCM",
        }
        return labels.get(v, v)
    if mi.get("estimator_label"):
        return mi["estimator_label"]
    if mi.get("inference_method") == "conformal":
        return "Conformal SCM"
    if mi.get("n_factors") is not None:
        return f"GSynth ({mi['n_factors']}F)"
    if mi.get("n_cohorts") is not None:
        return "Staggered SCM"
    if "model_type" in mi and "Augmented" in str(mi["model_type"]):
        return "Augmented SCM"

    # Fallback: parse result.method
    m = result.method
    if "Demeaned" in m or "De-meaned" in m:
        return "De-meaned SCM"
    if "De-trended" in m or "Detrended" in m:
        return "De-trended SCM"
    if "Unconstrained" in m:
        return "Unconstrained SCM"
    if "Augmented" in m:
        return "Augmented SCM"
    if "Generalized" in m:
        return "GSynth"
    if "Staggered" in m:
        return "Staggered SCM"
    if "Conformal" in m:
        return "Conformal SCM"
    if "SDID" in m or "Synthetic Difference" in m:
        return "SDID"
    return "SCM"


def _clean_spines(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def _ensure_matplotlib():
    try:
        import matplotlib
    except ImportError:
        raise ImportError(
            "matplotlib is required for plotting. "
            "Install: pip install matplotlib"
        )
