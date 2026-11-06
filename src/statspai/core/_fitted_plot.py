"""``EconometricResults.plot(type='fitted')``: fitted against observed values.

Each observation of the estimation sample is drawn at its observed value,
with the fitted value and a pointwise interval for it. On a date-indexed
sample the horizontal axis is the date; otherwise the observations are
ordered by their fitted value, so the fitted values form a rising line and
the interval a band around it.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

from ..exceptions import MethodIncompatibility


def _time_like(index: Optional[pd.Index]) -> bool:
    return isinstance(index, (pd.DatetimeIndex, pd.PeriodIndex)) and bool(
        index.is_monotonic_increasing
    )


def fitted_plot(
    result: Any,
    *,
    interval: str = "confidence",
    alpha: Optional[float] = None,
    data: Optional[pd.DataFrame] = None,
    ax: Any = None,
) -> Any:
    """Observed values, fitted values and a ``1 - alpha`` interval.

    Parameters
    ----------
    result : EconometricResults
    interval : {'confidence', 'prediction', 'none'}, default 'confidence'
        ``'confidence'``: the interval for the mean ``E[y | x]``;
        ``'prediction'``: for a new observation (linear models, unweighted
        fits); ``'none'``: no band. Both come from ``result.predict``.
    alpha : float, optional
        One minus the coverage; the fit's ``alpha`` by default.
    data : DataFrame, optional
        The estimation data. Needed after a fit with absorbed fixed
        effects, whose stored design is demeaned; the rows of the
        estimation sample are taken from it by label.
    ax : matplotlib Axes, optional

    Returns
    -------
    (fig, ax)
    """
    if interval not in ("confidence", "prediction", "none"):
        raise MethodIncompatibility(
            f"plot(type='fitted'): interval must be 'confidence', "
            f"'prediction' or 'none'; got {interval!r}.",
            recovery_hint="Use interval='confidence'.",
        )
    fitted = result.fitted_values()
    resid = result.residuals()
    if fitted is None or resid is None:
        raise MethodIncompatibility(
            "plot(type='fitted'): this fit stores no fitted values or " "residuals.",
            recovery_hint="Use plot() for the coefficient plot.",
        )
    yhat = np.asarray(fitted, dtype=float).ravel()
    observed = yhat + np.asarray(resid, dtype=float).ravel()
    level = result.alpha if alpha is None else float(alpha)
    index = result.sample_index

    lower = upper = None
    if interval != "none":
        rows = None
        if data is not None:
            if index is None or not pd.Index(index).isin(data.index).all():
                raise MethodIncompatibility(
                    "plot(type='fitted'): data= does not hold the rows of the "
                    "estimation sample.",
                    recovery_hint="Pass the DataFrame the model was fitted on.",
                )
            rows = data.loc[index]
        try:
            band = result.predict(rows, what=interval, alpha=level)
        except (MethodIncompatibility, NotImplementedError) as exc:
            raise MethodIncompatibility(
                f"plot(type='fitted'): no {interval} interval for this fit "
                f"({exc}).",
                recovery_hint="Pass the estimation data as data=, or use "
                "interval='none'.",
            ) from exc
        if not np.allclose(
            band["yhat"].to_numpy(dtype=float),
            yhat,
            rtol=1e-8,
            atol=1e-10 * max(1.0, float(np.max(np.abs(yhat)))),
        ):
            raise MethodIncompatibility(
                "plot(type='fitted'): the predictions for the estimation "
                "sample do not reproduce the stored fitted values.",
                recovery_hint="Pass the DataFrame the model was fitted on as "
                "data=, or use interval='none'.",
            )
        lower = band["lower"].to_numpy(dtype=float)
        upper = band["upper"].to_numpy(dtype=float)

    import matplotlib.pyplot as plt

    if ax is None:
        fig, ax = plt.subplots(figsize=(7.0, 4.2))
    else:
        fig = ax.figure
    if _time_like(index):
        x: Any = index
        xlabel = "date"
        order = np.arange(yhat.size)
    else:
        order = np.argsort(yhat, kind="stable")
        x = np.arange(1, yhat.size + 1)
        xlabel = "observation, ordered by fitted value"
    if lower is not None and upper is not None:
        ax.fill_between(
            x,
            lower[order],
            upper[order],
            color="C0",
            alpha=0.25,
            lw=0,
            label=f"{100 * (1 - level):g}% {interval} interval",
        )
    ax.plot(x, yhat[order], color="C0", lw=1.4, label="fitted")
    ax.scatter(x, observed[order], s=9, color="C1", zorder=3, label="observed")
    ax.set_xlabel(xlabel)
    dep = (getattr(result, "data_info", None) or {}).get("dependent_var")
    ax.set_ylabel(str(dep) if dep else "outcome")
    ax.legend(loc="best", frameon=False)
    return fig, ax
