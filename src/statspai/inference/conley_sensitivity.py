"""Conley standard errors across distance cutoffs.

The Conley (1999) spatial HAC standard error depends on a bandwidth that
theory does not pin down, so applied papers report it at several cutoffs.
:func:`conley_sensitivity` runs :func:`statspai.conley` once per cutoff on
the same fit (OLS or 2SLS) and tabulates the standard error, interval and
p-value of each coefficient. Nothing is re-estimated and no kernel is
re-implemented here: every number comes from ``sp.conley``.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
from scipy import stats

from .._result_serialize import ResultProtocolMixin
from ..exceptions import MethodIncompatibility

__all__ = ["conley_sensitivity", "ConleySensitivityResult"]

_INTERCEPT_NAMES = ("Intercept", "(Intercept)", "const", "_cons")


class ConleySensitivityResult(ResultProtocolMixin):
    """Table of Conley standard errors by distance cutoff.

    Attributes
    ----------
    table : DataFrame
        Long format, one row per (cutoff, term): ``cutoff_km``, ``term``,
        ``estimate``, ``se``, ``ci_lower``, ``ci_upper``, ``pvalue``,
        ``fit_se`` (the standard error the fit itself reports) and
        ``se_ratio`` (``se / fit_se``).

    Examples
    --------
    >>> import statspai as sp
    >>> sp.ConleySensitivityResult.__name__
    'ConleySensitivityResult'
    """

    def __init__(
        self,
        table: pd.DataFrame,
        settings: Dict[str, Any],
    ) -> None:
        self.table = table
        self.settings = settings

    def wide(self, value: str = "se") -> pd.DataFrame:
        """Cutoffs as rows and terms as columns for one column of ``table``."""
        return self.table.pivot(index="cutoff_km", columns="term", values=value)

    def to_dict(self) -> Dict[str, Any]:
        return {"settings": dict(self.settings), "table": self.table.to_dict("records")}

    def summary(self) -> str:
        s = self.settings
        head = (
            f"Conley spatial HAC by cutoff ({s['kernel']} kernel, {s['distance']} "
            f"distance, {s['inference']} intervals at {100 * (1 - s['alpha']):g}%)"
        )
        body = self.table.to_string(index=False, float_format=lambda v: f"{v:.6g}")
        return "\n".join([head, "=" * len(head), body])

    def __repr__(self) -> str:
        return self.summary()

    def plot(self, terms: Optional[Sequence[str]] = None, ax: Any = None) -> Any:
        """Estimate with its interval at each cutoff, one panel per term.

        Returns the matplotlib figure.
        """
        import matplotlib.pyplot as plt

        names = list(terms) if terms is not None else list(self.table["term"].unique())
        if ax is None:
            fig, axes = plt.subplots(
                1, len(names), figsize=(4.0 * len(names), 3.2), squeeze=False
            )
            axes = axes[0]
        else:
            axes = np.atleast_1d(ax)
            fig = axes[0].figure
            if len(axes) != len(names):
                raise MethodIncompatibility(
                    f"plot: {len(names)} term(s) but {len(axes)} axes.",
                    recovery_hint="Pass one axis per term, or terms=.",
                )
        for a, name in zip(axes, names):
            sub = self.table[self.table["term"] == name]
            x = sub["cutoff_km"].to_numpy(dtype=float)
            est = sub["estimate"].to_numpy(dtype=float)
            lo = sub["ci_lower"].to_numpy(dtype=float)
            hi = sub["ci_upper"].to_numpy(dtype=float)
            a.vlines(x, lo, hi, color="#3b6ea5", linewidth=2)
            a.plot(x, est, "o", color="#1f2933")
            fit_se = float(sub["fit_se"].iloc[0])
            crit = float(self.settings["critical_value"])
            if np.isfinite(fit_se):
                a.axhspan(
                    est[0] - crit * fit_se,
                    est[0] + crit * fit_se,
                    color="#d9d9d9",
                    alpha=0.5,
                    label="fit's own SE",
                )
            a.axhline(0.0, color="#888888", linewidth=0.8)
            a.set_title(str(name))
            a.set_xlabel("distance cutoff (km)")
        axes[0].set_ylabel("estimate and interval")
        fig.tight_layout()
        return fig


def conley_sensitivity(
    result: Union[Any, str],
    data: pd.DataFrame,
    lat: str,
    lon: str,
    cutoffs: Sequence[float],
    *,
    terms: Optional[Sequence[str]] = None,
    kernel: str = "uniform",
    distance: str = "planar",
    alpha: float = 0.05,
    inference: str = "normal",
    method: str = "ols",
    fit_kwargs: Optional[Dict[str, Any]] = None,
) -> ConleySensitivityResult:
    """Conley standard errors, intervals and p-values across distance cutoffs.

    Parameters
    ----------
    result : fitted result or str
        An ``sp.regress`` / ``sp.feols`` (OLS) or ``sp.iv`` / ``sp.ivreg``
        2SLS result, or a formula, which is then fitted on ``data`` with
        ``sp.regress`` (``method='ols'``) or ``sp.iv`` (``method='iv'``).
    data : DataFrame
        Rows aligned with the estimation sample (the frame the model was
        fitted on, after dropping incomplete rows), with the coordinates.
    lat, lon : str
        Latitude and longitude columns, in degrees.
    cutoffs : sequence of float
        Distance cutoffs in km, each a positive number.
    terms : sequence of str, optional
        Coefficients to report (default all but the intercept).
    kernel : {'uniform', 'bartlett'}, default 'uniform'
        Spatial kernel, as ``sp.conley`` (``acreg`` without / with
        ``bartlett``).
    distance : {'planar', 'haversine'}, default 'planar'
        ``'planar'`` is Stata ``acreg``'s 111 km per degree convention, the
        one ``sp.regress`` / ``sp.iv(vce='conley')`` use; ``'haversine'`` is
        great-circle distance (``sp.conley``'s own default).
    alpha : float, default 0.05
    inference : {'normal', 't'}, default 'normal'
        Reference distribution of intervals and p-values. ``'normal'`` is
        what ``acreg`` and ``sp.iv(vce='conley')`` report; ``'t'`` uses
        ``t(n - k)`` as ``sp.conley`` does. The standard errors are the same.
    method : {'ols', 'iv'}, default 'ols'
        Estimator when ``result`` is a formula.
    fit_kwargs : dict, optional
        Extra arguments for that fit.

    Returns
    -------
    ConleySensitivityResult
        ``.table`` (long), ``.wide('se')``, ``.summary()``, ``.plot()``,
        ``.to_dict()``.

    Raises
    ------
    MethodIncompatibility
        Bad cutoffs or options, a likelihood (non least-squares) fit, data
        not aligned with the fit (raised by ``sp.conley``).

    Notes
    -----
    The uniform kernel does not guarantee a positive semi-definite
    covariance; at wide cutoffs a variance can come out negative. That
    standard error is reported as NaN with a warning (``acreg`` shows a
    missing value), never as zero.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> n = 300
    >>> df = pd.DataFrame({"lat": rng.uniform(50, 54, n),
    ...                    "lon": rng.uniform(-4, 1, n), "x": rng.normal(size=n)})
    >>> df["y"] = 0.5 * df.x + rng.normal(size=n)
    >>> fit = sp.regress("y ~ x", data=df)
    >>> tab = sp.conley_sensitivity(fit, df, "lat", "lon", [25, 50, 100])
    >>> tab.wide("se").shape
    (3, 1)

    References
    ----------
    [@conley1999estimation]
    """
    import statspai as sp

    cut = [float(c) for c in np.atleast_1d(np.asarray(cutoffs, dtype=float))]
    if not cut or any(not np.isfinite(c) or c <= 0 for c in cut):
        raise MethodIncompatibility(
            f"conley_sensitivity: cutoffs must be positive distances in km, got "
            f"{list(cutoffs)!r}.",
            recovery_hint="e.g. cutoffs=[25, 50, 100, 200].",
        )
    if len(set(cut)) != len(cut):
        raise MethodIncompatibility(
            "conley_sensitivity: cutoffs contain duplicates.",
            recovery_hint="List each cutoff once.",
        )
    if inference not in ("normal", "t"):
        raise MethodIncompatibility(
            f"conley_sensitivity: inference must be 'normal' or 't', got "
            f"{inference!r}.",
            recovery_hint="inference='normal' (acreg) or 't' (sp.conley).",
        )
    if not (0.0 < float(alpha) < 1.0):
        raise MethodIncompatibility(
            f"conley_sensitivity: alpha must be in (0, 1), got {alpha!r}.",
            recovery_hint="Use alpha=0.05.",
        )
    if isinstance(result, str):
        if method not in ("ols", "iv"):
            raise MethodIncompatibility(
                f"conley_sensitivity: method must be 'ols' or 'iv', got {method!r}.",
                recovery_hint="method='iv' for a formula with an IV block.",
            )
        fn = sp.regress if method == "ols" else sp.iv
        fit = fn(result, data=data, **dict(fit_kwargs or {}))
    else:
        if fit_kwargs:
            raise MethodIncompatibility(
                "conley_sensitivity: fit_kwargs= applies only when result is a "
                "formula.",
                recovery_hint="Drop fit_kwargs=, or pass the formula.",
            )
        fit = result

    params = pd.Series(fit.params, dtype=float)
    fit_se = pd.Series(fit.std_errors, dtype=float).reindex(params.index)
    if terms is None:
        names: List[str] = [c for c in params.index if c not in _INTERCEPT_NAMES]
    else:
        names = [str(t) for t in terms]
        unknown = [t for t in names if t not in params.index]
        if unknown:
            raise MethodIncompatibility(
                f"conley_sensitivity: {unknown} are not coefficients of the fit.",
                recovery_hint=f"Choose from {list(params.index)}.",
            )

    rows = []
    df_t = None
    for c in cut:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            out = sp.conley(
                fit,
                data=data,
                lat=lat,
                lon=lon,
                dist_cutoff=c,
                kernel=kernel,
                alpha=alpha,
                distance=distance,
            )
        for w in caught:  # re-issue, tagged with the cutoff
            warnings.warn(
                f"conley_sensitivity (cutoff {c:g} km): {w.message}",
                w.category,
                stacklevel=2,
            )
        df_t = float(out.data_info.get("df_resid"))
        se = pd.Series(out.std_errors, dtype=float)
        for t in names:
            rows.append(
                {
                    "cutoff_km": c,
                    "term": t,
                    "estimate": float(params[t]),
                    "se": float(se[t]),
                    "fit_se": float(fit_se[t]),
                }
            )
    table = pd.DataFrame(rows)
    if inference == "normal":
        crit = float(stats.norm.ppf(1 - alpha / 2))
        tail = stats.norm.sf
    else:
        crit = float(stats.t.ppf(1 - alpha / 2, df_t))

        def tail(v: np.ndarray) -> np.ndarray:
            return stats.t.sf(v, df_t)

    with np.errstate(invalid="ignore", divide="ignore"):
        tt = table["estimate"] / table["se"]
        table["ci_lower"] = table["estimate"] - crit * table["se"]
        table["ci_upper"] = table["estimate"] + crit * table["se"]
        table["pvalue"] = 2 * tail(np.abs(tt.to_numpy(dtype=float)))
        table["se_ratio"] = table["se"] / table["fit_se"]
    table = table[
        [
            "cutoff_km",
            "term",
            "estimate",
            "se",
            "ci_lower",
            "ci_upper",
            "pvalue",
            "fit_se",
            "se_ratio",
        ]
    ]
    settings = {
        "lat": lat,
        "lon": lon,
        "cutoffs": cut,
        "kernel": kernel,
        "distance": distance,
        "alpha": float(alpha),
        "inference": inference,
        "critical_value": crit,
        "df_t": df_t,
        "n": int(len(data)),
        "fit_vcov": (getattr(fit, "model_info", None) or {}).get("vcov_type")
        or (getattr(fit, "model_info", None) or {}).get("robust"),
    }
    return ConleySensitivityResult(table, settings)
