"""Binned residuals for models of a binary or count outcome."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

from ..exceptions import DataInsufficient, MethodIncompatibility


def _fitted_and_residuals(result: Any) -> tuple:
    info = getattr(result, "data_info", None) or {}
    fitted = info.get("fitted_values")
    resid = info.get("residuals")
    if fitted is None or resid is None:
        predict = getattr(result, "predict", None)
        model = getattr(result, "_model", None)
        if callable(predict) and model is not None and hasattr(model, "y"):
            fitted = np.asarray(predict(), dtype=float)
            observed = np.asarray(model.y, dtype=float)
            trials = getattr(model, "trials", None)
            if trials is not None:  # grouped binomial: residual of the share
                observed = observed / np.asarray(trials, dtype=float)
            resid = observed - fitted
        else:
            raise MethodIncompatibility(
                "This result does not expose fitted values and residuals. "
                "Pass them directly: sp.binned_residuals(fitted, residuals)."
            )
    return np.asarray(fitted, dtype=float), np.asarray(resid, dtype=float)


def binned_residuals(
    x: Any,
    residuals: Optional[Any] = None,
    n_bins: Optional[int] = None,
    by: Optional[Any] = None,
    band: str = "empirical",
) -> pd.DataFrame:
    """Average residuals within bins of the fitted value or a regressor.

    The residuals of a binary regression take two values per fitted
    probability and a plot of them shows nothing. Averaged within bins
    holding equal numbers of observations they should scatter around
    zero inside ``+/- 2 sd / sqrt(n)``; a run of bins outside the band,
    or a trend across bins, shows where the mean model is wrong.

    Parameters
    ----------
    x : fitted model or array-like
        A fitted model (``sp.logit``, ``sp.probit``, ``sp.glm``,
        ``sp.poisson``, ``sp.bayes_regress``): its fitted values are
        binned and its response residuals averaged. Or the values to bin
        on, with ``residuals``.
    residuals : array-like, optional
        Observed minus fitted. Required when ``x`` is an array.
    n_bins : int, optional
        Number of bins. Default ``floor(sqrt(n))``.
    by : array-like, optional
        Bin on this variable instead of the fitted values of a model,
        e.g. a regressor, to see whether it needs a transformation.

    band : {'empirical', 'model'}, default 'empirical'
        ``'empirical'``: ``2 sd / sqrt(n)`` of the residuals in the bin,
        the band of ``arm::binnedplot``. Where the outcome hardly varies
        (fitted probabilities near 0 or 1) those residuals are nearly
        equal, the band collapses and the bin is flagged for no reason.
        ``'model'``: ``2 sqrt(sum p (1 - p)) / n``, the standard error the
        fitted binary model implies for the bin average, which does not
        collapse. Needs a fitted model of a 0 / 1 outcome.

    Returns
    -------
    pd.DataFrame
        One row per bin: ``xbar`` and ``ybar`` (means of the binning
        variable and of the residuals), ``n``, ``x_lo``, ``x_hi``,
        ``two_se`` (``2 sd / sqrt(n)`` of the residuals in the bin) and
        ``outside`` (``|ybar| > two_se``). ``attrs['share_outside']``
        is the share of bins outside the band; about 5 percent is
        expected from a correct model.

    Notes
    -----
    Bins are cut at the order statistics ``floor(n i / n_bins)`` of the
    binning variable and are closed on the right, so tied values stay
    together and bin sizes can differ by a few observations (the rule of
    R ``arm::binned.resids``).

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> df = pd.DataFrame({"x": rng.normal(size=400)})
    >>> df["y"] = (rng.uniform(size=400) < 1 / (1 + np.exp(-df["x"]))).astype(int)
    >>> out = sp.binned_residuals(sp.logit("y ~ x", df))
    >>> len(out)
    20

    References
    ----------
    gelman2006data, gelman2020regression
    """
    if band not in ("empirical", "model"):
        raise MethodIncompatibility("band must be 'empirical' or 'model'.")
    prob: Optional[np.ndarray] = None
    if residuals is None:
        fitted, resid = _fitted_and_residuals(x)
        xv = fitted if by is None else np.asarray(by, dtype=float).reshape(-1)
        if band == "model":
            outcome = fitted + resid
            binary = np.all(np.isclose(outcome, 0.0) | np.isclose(outcome, 1.0))
            if not binary or fitted.min() < 0 or fitted.max() > 1:
                raise MethodIncompatibility(
                    "band='model' is for a fitted model of a 0 / 1 outcome."
                )
            prob = fitted
    else:
        if band == "model":
            raise MethodIncompatibility(
                "band='model' needs the fitted model, not arrays."
            )
        if by is not None:
            raise MethodIncompatibility("Pass by= only with a fitted model.")
        xv = np.asarray(x, dtype=float).reshape(-1)
        resid = np.asarray(residuals, dtype=float).reshape(-1)
    if xv.shape != resid.shape:
        raise MethodIncompatibility(
            f"The binning variable has {xv.size} values and the residuals "
            f"{resid.size}."
        )
    keep = np.isfinite(xv) & np.isfinite(resid)
    xv, resid = xv[keep], resid[keep]
    if prob is not None:
        prob = prob[keep]
    n = xv.size
    if n < 4:
        raise DataInsufficient("Binned residuals need at least four observations.")
    bins = int(np.floor(np.sqrt(n))) if n_bins is None else int(n_bins)
    if not 2 <= bins <= n:
        raise MethodIncompatibility(
            f"n_bins must be between 2 and the number of observations; got {bins}."
        )
    cut_index = np.floor(n * np.arange(1, bins) / bins).astype(int)
    cuts = np.sort(xv)[cut_index - 1]
    # right-closed intervals: the bin of x is the number of cuts below it
    which = np.searchsorted(cuts, xv, side="left")
    rows = []
    for b in range(bins):
        inside = which == b
        m = int(inside.sum())
        if m == 0:
            continue
        r = resid[inside]
        sd = float(r.std(ddof=1)) if m > 1 else float("nan")
        if prob is not None:
            q = prob[inside]
            sd = float(np.sqrt((q * (1.0 - q)).mean()))
        rows.append(
            {
                "xbar": float(xv[inside].mean()),
                "ybar": float(r.mean()),
                "n": m,
                "x_lo": float(xv[inside].min()),
                "x_hi": float(xv[inside].max()),
                "two_se": 2.0 * sd / np.sqrt(m),
            }
        )
    out = pd.DataFrame(rows)
    out["outside"] = out["ybar"].abs() > out["two_se"]
    out.attrs["share_outside"] = float(out["outside"].mean())
    out.attrs["n_obs"] = int(n)
    return out


def binned_residuals_plot(
    x: Any,
    residuals: Optional[Any] = None,
    n_bins: Optional[int] = None,
    by: Optional[Any] = None,
    ax: Any = None,
    xlabel: Optional[str] = None,
    band: str = "empirical",
) -> Any:
    """Plot of :func:`binned_residuals` with its ``+/- 2 se`` band.

    Parameters are those of :func:`binned_residuals` (including
    ``band``), plus an optional matplotlib ``ax`` and ``xlabel``.

    Returns
    -------
    matplotlib.figure.Figure

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> df = pd.DataFrame({"x": rng.normal(size=400)})
    >>> df["y"] = (rng.uniform(size=400) < 1 / (1 + np.exp(-df["x"]))).astype(int)
    >>> fig = sp.binned_residuals_plot(sp.logit("y ~ x", df))

    References
    ----------
    gelman2006data
    """
    import matplotlib.pyplot as plt

    table = binned_residuals(x, residuals, n_bins=n_bins, by=by, band=band)
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 4))
    else:
        fig = ax.figure
    ax.axhline(0.0, color="grey", linewidth=0.8)
    ax.plot(table["xbar"], table["two_se"], color="grey", linewidth=0.8)
    ax.plot(table["xbar"], -table["two_se"], color="grey", linewidth=0.8)
    ax.scatter(table["xbar"], table["ybar"], s=14, color="black")
    default = "fitted value" if residuals is None and by is None else "x"
    ax.set_xlabel(default if xlabel is None else xlabel)
    ax.set_ylabel("average residual")
    fig.tight_layout()
    return fig


__all__ = ["binned_residuals", "binned_residuals_plot"]
