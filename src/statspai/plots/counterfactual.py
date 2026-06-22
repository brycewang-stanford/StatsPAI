"""Unified counterfactual contract and plot.

CausalPy's signature output is the same picture for every quasi-experimental
design: the observed series, the model's counterfactual (no-intervention)
series, an uncertainty band, and a vertical marker at the intervention. StatsPAI
already produces this *per design* (``impactplot`` for causal impact,
``synthplot`` for synthetic control) but each reads a bespoke layout.

This module gives StatsPAI a single contract instead. :func:`counterfactual_data`
normalises any result that carries an observed-vs-counterfactual time series into
one tidy frame; :func:`counterfactual_plot` renders it. New designs that store
the same series (e.g. Bayesian synthetic control / Bayesian ITS) join the
contract for free.

Supported result types
----------------------
- Causal impact (``sp.causal_impact``) — ``CausalResult`` with an ``actual`` /
  ``predicted`` detail table.
- Synthetic control family (``sp.synth`` and friends) — reconstructed from the
  treated / synthetic series.
- Interrupted time series (``sp.its``) — reads the observed / counterfactual
  series stored on ``ITSResult.detail``.

The tidy frame columns are always:

``time, observed, counterfactual, point_effect`` plus, when available,
``cf_lower, cf_upper`` (counterfactual band), ``effect_lower, effect_upper``
(pointwise-effect band), ``post`` (bool, post-intervention), and
``cumulative_effect`` (running sum of ``point_effect`` over the post period).
"""

from __future__ import annotations

from typing import Any, Optional, Sequence

import numpy as np
import pandas as pd

__all__ = ["counterfactual_data", "counterfactual_plot"]

_SUPPORTED = (
    "Supported: sp.causal_impact results, the sp.synth family, and sp.its "
    "(ITSResult). The result must carry an observed-vs-counterfactual series."
)


def _coerce_times(times: Any, n: int) -> list:
    """Return a length-``n`` list of time labels, falling back to a range."""
    if times is None:
        return list(range(n))
    if isinstance(times, pd.Index):
        times = times.tolist()
    elif isinstance(times, pd.Series):
        times = times.tolist()
    else:
        times = list(np.asarray(times).ravel())
    if len(times) != n:
        return list(range(n))
    return list(times)


def _add_cumulative(df: pd.DataFrame) -> pd.DataFrame:
    """Attach a post-period running sum of ``point_effect`` (NaN pre-period)."""
    cum = np.full(len(df), np.nan)
    if "post" in df.columns and df["post"].any():
        mask = df["post"].to_numpy(dtype=bool)
        eff = df["point_effect"].to_numpy(dtype=float)
        cum[mask] = np.nancumsum(np.where(mask, eff, 0.0))[mask]
    df = df.copy()
    df["cumulative_effect"] = cum
    return df


def _from_causal_impact(result: Any) -> Optional[pd.DataFrame]:
    detail = getattr(result, "detail", None)
    if not isinstance(detail, pd.DataFrame):
        return None
    cols = set(detail.columns)
    if not {"actual", "predicted"}.issubset(cols):
        return None
    out = pd.DataFrame(
        {
            "time": _coerce_times(detail.get("time"), len(detail)),
            "observed": detail["actual"].to_numpy(dtype=float),
            "counterfactual": detail["predicted"].to_numpy(dtype=float),
        }
    )
    out["point_effect"] = out["observed"] - out["counterfactual"]
    if {"effect_lower", "effect_upper"}.issubset(cols):
        out["effect_lower"] = detail["effect_lower"].to_numpy(dtype=float)
        out["effect_upper"] = detail["effect_upper"].to_numpy(dtype=float)
        # Counterfactual band is the pointwise-effect band reflected through the
        # observed series, so it inherits the result's own CI level.
        out["cf_lower"] = out["observed"] - out["effect_upper"]
        out["cf_upper"] = out["observed"] - out["effect_lower"]
    if "post_intervention" in cols:
        out["post"] = detail["post_intervention"].to_numpy(dtype=bool)
    return _add_cumulative(out)


def _from_its(result: Any) -> Optional[pd.DataFrame]:
    if type(result).__name__ != "ITSResult":
        return None
    detail = getattr(result, "detail", None)
    if not isinstance(detail, dict) or "observed" not in detail:
        return None
    observed = np.asarray(detail["observed"], dtype=float)
    counterfactual = np.asarray(detail.get("counterfactual"), dtype=float)
    out = pd.DataFrame(
        {
            "time": _coerce_times(detail.get("time"), len(observed)),
            "observed": observed,
            "counterfactual": counterfactual,
        }
    )
    out["point_effect"] = out["observed"] - out["counterfactual"]
    if "post" in detail:
        out["post"] = np.asarray(detail["post"], dtype=bool)
    elif getattr(result, "intervention_time", None) is not None:
        out["post"] = np.asarray(out["time"]) >= result.intervention_time
    if "cf_lower" in detail and "cf_upper" in detail:
        out["cf_lower"] = np.asarray(detail["cf_lower"], dtype=float)
        out["cf_upper"] = np.asarray(detail["cf_upper"], dtype=float)
    return _add_cumulative(out)


def _from_bayes_series(result: Any) -> Optional[pd.DataFrame]:
    """Bayesian time-series results (bayes_its / bayes_synth) store the
    observed / counterfactual series + credible bands on ``model_info``."""
    mi = getattr(result, "model_info", None)
    if not isinstance(mi, dict):
        return None
    detail = mi.get("detail")
    if not isinstance(detail, dict) or "observed" not in detail:
        return None
    observed = np.asarray(detail["observed"], dtype=float)
    out = pd.DataFrame(
        {
            "time": _coerce_times(detail.get("time"), len(observed)),
            "observed": observed,
            "counterfactual": np.asarray(detail["counterfactual"], dtype=float),
        }
    )
    out["point_effect"] = out["observed"] - out["counterfactual"]
    if "post" in detail:
        out["post"] = np.asarray(detail["post"], dtype=bool)
    if "cf_lower" in detail and "cf_upper" in detail:
        out["cf_lower"] = np.asarray(detail["cf_lower"], dtype=float)
        out["cf_upper"] = np.asarray(detail["cf_upper"], dtype=float)
    return _add_cumulative(out)


def _from_synth(result: Any) -> Optional[pd.DataFrame]:
    try:
        from ..synth.exports import _gap_table
    except Exception:  # pragma: no cover - synth is always importable
        return None
    gap = _gap_table(result)
    if not isinstance(gap, pd.DataFrame) or gap.empty:
        return None
    out = pd.DataFrame(
        {
            "time": list(gap["time"]),
            "observed": gap["treated"].to_numpy(dtype=float),
            "counterfactual": gap["synthetic"].to_numpy(dtype=float),
        }
    )
    out["point_effect"] = (
        gap["gap"].to_numpy(dtype=float)
        if "gap" in gap.columns
        else out["observed"] - out["counterfactual"]
    )
    if "post_treatment" in gap.columns:
        out["post"] = gap["post_treatment"].to_numpy(dtype=bool)
    return _add_cumulative(out)


def counterfactual_data(result: Any) -> pd.DataFrame:
    """Normalise a fitted result into a tidy counterfactual frame.

    Parameters
    ----------
    result : object
        A StatsPAI result carrying an observed-vs-counterfactual time series:
        a ``sp.causal_impact`` result, any ``sp.synth`` family result, or an
        ``sp.its`` ``ITSResult``.

    Returns
    -------
    pandas.DataFrame
        Columns ``time, observed, counterfactual, point_effect`` plus, when the
        result provides them, ``cf_lower, cf_upper, effect_lower, effect_upper,
        post, cumulative_effect``.

    Raises
    ------
    TypeError
        If ``result`` does not expose a recognised counterfactual contract.

    Examples
    --------
    >>> import statspai as sp  # doctest: +SKIP
    >>> res = sp.its(df, y="y", time="t", intervention=30)  # doctest: +SKIP
    >>> sp.counterfactual_data(res).columns.tolist()  # doctest: +SKIP
    ['time', 'observed', 'counterfactual', 'point_effect', 'post', ...]
    """
    for reader in (
        _from_causal_impact,
        _from_its,
        _from_bayes_series,
        _from_synth,
    ):
        out = reader(result)
        if out is not None:
            return out
    raise TypeError(
        "counterfactual_data() could not find an observed-vs-counterfactual "
        f"series on {type(result).__name__}. {_SUPPORTED}"
    )


def counterfactual_plot(
    result: Any,
    *,
    ax: Any = None,
    figsize: Sequence[float] = (10.0, 7.0),
    bands: bool = True,
    show_effect: bool = True,
    title: Optional[str] = None,
) -> Any:
    """Plot observed vs counterfactual with an uncertainty band.

    The figure has a trajectory panel (observed solid, counterfactual dashed,
    shaded band, intervention marker) and, by default, a lower panel of the
    pointwise treatment effect with a zero reference line.

    Parameters
    ----------
    result : object
        Any result supported by :func:`counterfactual_data`.
    ax : matplotlib.axes.Axes, optional
        Draw the trajectory panel into this axis instead of a new figure. When
        given, the effect panel is suppressed (single-axis mode).
    figsize : sequence of float, default (10, 7)
        Figure size when a new figure is created.
    bands : bool, default True
        Shade the counterfactual / effect uncertainty bands when available.
    show_effect : bool, default True
        Add the lower pointwise-effect panel.
    title : str, optional
        Figure title. Defaults to the result's method.

    Returns
    -------
    matplotlib.figure.Figure
        The figure (or the parent figure of ``ax``).

    Examples
    --------
    >>> import statspai as sp  # doctest: +SKIP
    >>> res = sp.causal_impact(df, ...)  # doctest: +SKIP
    >>> fig = sp.counterfactual_plot(res)  # doctest: +SKIP
    """
    import matplotlib.pyplot as plt

    data = counterfactual_data(result)
    time = data["time"].tolist()
    method = title or getattr(result, "method", "Counterfactual")

    single_axis = ax is not None
    if single_axis:
        ax_top = ax
        fig = ax_top.figure
        ax_bot = None
    elif show_effect:
        fig, (ax_top, ax_bot) = plt.subplots(
            2,
            1,
            figsize=tuple(figsize),
            sharex=True,
            gridspec_kw={"height_ratios": [2, 1]},
        )
    else:
        fig, ax_top = plt.subplots(figsize=tuple(figsize))
        ax_bot = None

    ax_top.plot(time, data["observed"], color="black", lw=1.6, label="Observed")
    ax_top.plot(
        time,
        data["counterfactual"],
        color="#1f77b4",
        lw=1.6,
        ls="--",
        label="Counterfactual",
    )
    if bands and {"cf_lower", "cf_upper"}.issubset(data.columns):
        ax_top.fill_between(
            time,
            data["cf_lower"],
            data["cf_upper"],
            color="#1f77b4",
            alpha=0.2,
            label="Counterfactual band",
        )
    _mark_intervention(ax_top, data)
    ax_top.set_ylabel("Outcome")
    ax_top.set_title(method)
    ax_top.legend(loc="best", fontsize=9)

    if ax_bot is not None:
        ax_bot.axhline(0.0, color="grey", lw=0.8, ls=":")
        ax_bot.plot(time, data["point_effect"], color="#d62728", lw=1.4)
        if bands and {"effect_lower", "effect_upper"}.issubset(data.columns):
            ax_bot.fill_between(
                time,
                data["effect_lower"],
                data["effect_upper"],
                color="#d62728",
                alpha=0.2,
            )
        _mark_intervention(ax_bot, data)
        ax_bot.set_ylabel("Pointwise effect")
        ax_bot.set_xlabel("Time")
    else:
        ax_top.set_xlabel("Time")

    fig.tight_layout()
    return fig


def _mark_intervention(ax: Any, data: pd.DataFrame) -> None:
    """Draw a vertical line at the first post-intervention period."""
    if "post" not in data.columns or not data["post"].any():
        return
    first_post = int(np.argmax(data["post"].to_numpy(dtype=bool)))
    if first_post <= 0:
        return
    ax.axvline(
        data["time"].iloc[first_post],
        color="grey",
        lw=1.0,
        ls="-.",
        alpha=0.7,
    )
