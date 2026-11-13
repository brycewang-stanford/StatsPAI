"""Diebold-Mariano test of equal predictive accuracy.

Two sets of forecasts of the same outcomes are compared through the loss
differential ``d_t = |e1_t|**power - |e2_t|**power``. Under the null of
equal expected loss its mean is zero; the test is a t ratio of the mean
of ``d`` with a long-run variance that allows for the serial correlation
an ``h``-step forecast error has by construction (up to lag ``h - 1``).
"""

from __future__ import annotations

from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from scipy import stats

from ..core._call_record import records_call
from ..exceptions import DataInsufficient, MethodIncompatibility, NumericalInstability

__all__ = ["dm_test"]

_ALTERNATIVES = {
    "two-sided": "two-sided",
    "two_sided": "two-sided",
    "two.sided": "two-sided",
    "less": "less",
    "greater": "greater",
}


def _errors(e: Any, data: Optional[pd.DataFrame], name: str) -> np.ndarray:
    if isinstance(e, str):
        if data is None:
            raise MethodIncompatibility(
                f"dm_test: {name}={e!r} is a column name but data= was not given.",
                recovery_hint="Pass data=, or the forecast errors as arrays.",
            )
        if e not in data.columns:
            raise MethodIncompatibility(
                f"dm_test: column {e!r} is not in data.",
                recovery_hint="Name a column that holds forecast errors.",
                diagnostics={"missing": e},
            )
        e = data[e]
    out = np.asarray(e, dtype=float)
    if out.ndim != 1:
        raise MethodIncompatibility(
            f"dm_test: {name} must be one series of forecast errors; got an "
            f"array of shape {out.shape}.",
            recovery_hint="Pass the errors of one horizon, e.g. "
            "np.asarray(sp.tscv(...).errors)[:, h - 1].",
        )
    return out


@records_call("dm_test")
def dm_test(
    e1: Any,
    e2: Any,
    *,
    h: int = 1,
    power: float = 2,
    alternative: str = "two-sided",
    varestimator: str = "acf",
    harvey: bool = True,
    data: Optional[pd.DataFrame] = None,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Diebold-Mariano test that two forecasts are equally accurate.

    Parameters
    ----------
    e1, e2 : array-like or str
        Forecast errors (outcome minus forecast) of the two methods over
        the same periods, in time order; column names when ``data`` is
        given. Periods where either is missing are left out; they must be
        at the start or the end (the first origins of a rolling
        evaluation), not inside the sample.
    h : int, default 1
        Forecast horizon the errors belong to. Errors of ``h``-step
        forecasts are autocorrelated up to lag ``h - 1``, and that many
        autocovariances of the loss differential enter its variance.
    power : float, default 2
        Loss is ``|e|**power``: 2 compares mean squared errors, 1 mean
        absolute errors.
    alternative : {'two-sided', 'less', 'greater'}, default 'two-sided'
        ``'greater'``: method 2 is more accurate than method 1 (its
        expected loss is smaller). ``'less'``: method 2 is less accurate.
    varestimator : {'acf', 'bartlett'}, default 'acf'
        Long-run variance of the loss differential. ``'acf'`` sums the
        autocovariances up to lag ``h - 1`` with weight one, the estimator
        of the original paper, which can be negative. ``'bartlett'``
        weights lag ``k`` by ``1 - k / h``, which cannot.
    harvey : bool, default True
        Apply the small-sample correction of Harvey, Leybourne and Newbold
        (1997): the statistic is multiplied by ``sqrt((n + 1 - 2h +
        h(h - 1)/n) / n)`` and referred to a t distribution with ``n - 1``
        degrees of freedom. ``False`` gives the original statistic with a
        standard normal reference.
    data : pandas.DataFrame, optional
        Frame holding the columns named by ``e1`` and ``e2``.
    alpha : float, default 0.05
        Level used for ``reject``.

    Returns
    -------
    dict
        ``statistic``, ``pvalue``, ``df`` (``n - 1``, or ``None`` with
        ``harvey=False``), ``mean_loss_diff`` (mean of ``|e1|**power -
        |e2|**power``: negative when method 1 has the smaller loss),
        ``loss1`` and ``loss2`` (the two mean losses), ``lrvar`` (the
        long-run variance of the loss differential), ``n_obs``, ``h``,
        ``power``, ``alternative``, ``varestimator``, ``harvey``,
        ``reject`` and ``test``.

    Raises
    ------
    MethodIncompatibility
        Unknown ``alternative`` or ``varestimator``, ``h < 1``, a
        non-positive ``power``, series of different lengths, or a missing
        value inside the sample.
    DataInsufficient
        Fewer than ``h + 2`` complete periods.
    NumericalInstability
        The variance estimate is not positive (identical forecasts, or
        ``varestimator='acf'`` with ``h > 1`` on a short sample).

    Notes
    -----
    With the defaults this is R's ``forecast::dm.test(e1, e2, alternative,
    h, power, varestimator)``. Autocovariances have divisor ``n``.

    The test compares forecasts, not models. For nested models estimated
    on the same data the loss differential degenerates under the null as
    the estimation sample grows, and the statistic is not asymptotically
    normal; rolling or expanding out-of-sample errors of non-nested
    methods are the setting the test was derived for.

    Examples
    --------
    >>> import numpy as np
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> y = rng.normal(size=200)
    >>> e_good = y - 0.0                         # forecast: the true mean
    >>> e_bad = y - rng.normal(scale=1.0, size=200)   # a noisy forecast
    >>> res = sp.dm_test(e_good, e_bad, alternative="less")
    >>> bool(res["mean_loss_diff"] < 0) and bool(res["reject"])
    True
    >>> res["df"]
    199

    References
    ----------
    diebold1995comparing, harvey1997testing
    """
    alt = _ALTERNATIVES.get(str(alternative).lower())
    if alt is None:
        raise MethodIncompatibility(
            f"dm_test: alternative={alternative!r} is not 'two-sided', 'less' "
            "or 'greater'.",
            recovery_hint="'greater' says method 2 is the more accurate one.",
        )
    var_key = str(varestimator).lower()
    if var_key not in ("acf", "bartlett"):
        raise MethodIncompatibility(
            f"dm_test: varestimator={varestimator!r} is not 'acf' or " "'bartlett'.",
            recovery_hint="Use 'bartlett' when 'acf' gives a negative variance.",
        )
    if int(h) != h or int(h) < 1:
        raise MethodIncompatibility(
            f"dm_test: h must be a positive integer, got {h!r}.",
            recovery_hint="h is the forecast horizon of the errors.",
        )
    h = int(h)
    if not float(power) > 0:
        raise MethodIncompatibility(
            f"dm_test: power must be positive, got {power!r}.",
            recovery_hint="power=2 for squared errors, 1 for absolute errors.",
        )
    if not 0.0 < float(alpha) < 1.0:
        raise MethodIncompatibility(
            f"dm_test: alpha must lie strictly between 0 and 1, got {alpha!r}.",
            recovery_hint="alpha=0.05 is the usual level.",
        )
    a = _errors(e1, data, "e1")
    b = _errors(e2, data, "e2")
    if a.shape[0] != b.shape[0]:
        raise MethodIncompatibility(
            f"dm_test: e1 has {a.shape[0]} errors and e2 has {b.shape[0]}; "
            "the two forecasts must cover the same periods.",
            recovery_hint="Align the two error series on their dates first.",
            diagnostics={"n_e1": int(a.shape[0]), "n_e2": int(b.shape[0])},
        )
    if np.isinf(a).any() or np.isinf(b).any():
        raise MethodIncompatibility(
            "dm_test: the forecast errors contain infinite values.",
            recovery_hint="Check the forecasts that produced them.",
        )
    ok = np.isfinite(a) & np.isfinite(b)
    if not ok.any():
        raise DataInsufficient(
            "dm_test: no period has both forecast errors.",
            recovery_hint="Check the two series for missing values.",
        )
    kept = np.flatnonzero(ok)
    first, last = int(kept[0]), int(kept[-1])
    holes = int((~ok[first : last + 1]).sum())
    if holes:
        raise MethodIncompatibility(
            f"dm_test: {holes} period(s) inside the sample lack a forecast "
            "error; the autocovariances of the loss differential need "
            "consecutive periods.",
            recovery_hint="Use a stretch without gaps.",
            diagnostics={"n_missing_inside": holes},
        )
    a, b = a[first : last + 1], b[first : last + 1]
    n = a.shape[0]
    if n < h + 2:
        raise DataInsufficient(
            f"dm_test: {n} forecast error(s) are too few for horizon h={h}.",
            recovery_hint="More forecast origins are needed.",
            diagnostics={"n_obs": int(n), "h": h},
        )
    la, lb = np.abs(a) ** float(power), np.abs(b) ** float(power)
    d = la - lb
    dbar = float(d.mean())
    dc = d - dbar
    gamma = np.array([float(dc[k:] @ dc[: n - k]) / n for k in range(h)])
    if var_key == "acf":
        weights = np.ones(h)
    else:
        weights = 1.0 - np.arange(h) / h
    lrvar = float(gamma[0] + 2.0 * (weights[1:] @ gamma[1:]))
    # degenerate when the two loss series coincide up to rounding
    scale = max(float(np.abs(d).max()), float(la.mean()), float(lb.mean()))
    if not lrvar > 1e-20 * max(scale * scale, np.finfo(float).tiny):
        raise NumericalInstability(
            "dm_test: the estimated variance of the mean loss differential "
            f"is not positive ({lrvar:.3g}).",
            recovery_hint=(
                "With h > 1 use varestimator='bartlett', which cannot be "
                "negative; if the two forecasts are identical there is "
                "nothing to test."
            ),
            diagnostics={"lrvar": lrvar, "h": h, "varestimator": var_key},
        )
    stat = dbar / np.sqrt(lrvar / n)
    df: Optional[int]
    if harvey:
        stat *= np.sqrt((n + 1.0 - 2.0 * h + h * (h - 1.0) / n) / n)
        df = n - 1
        dist: Any = stats.t(df)
    else:
        df = None
        dist = stats.norm()
    if alt == "two-sided":
        p = float(2.0 * dist.sf(abs(stat)))
    elif alt == "less":
        p = float(dist.cdf(stat))
    else:
        p = float(dist.sf(stat))
    return {
        "test": (
            "Diebold-Mariano test"
            + (" (Harvey-Leybourne-Newbold correction)" if harvey else "")
        ),
        "statistic": float(stat),
        "pvalue": p,
        "df": df,
        "mean_loss_diff": dbar,
        "loss1": float(la.mean()),
        "loss2": float(lb.mean()),
        "lrvar": lrvar,
        "n_obs": int(n),
        "h": h,
        "power": float(power),
        "alternative": alt,
        "varestimator": var_key,
        "harvey": bool(harvey),
        "reject": bool(p < alpha),
    }
