"""Gain curves for a CATE ranking on randomized data, any treatment type.

:func:`cate_eval` scores a ranking with RATE on doubly robust scores of a
0/1 treatment. With a dose (a discount, a price, a credit line) there is no
such score, but on data where the treatment was randomized the effect in any
subset is still the slope of the outcome on the treatment in that subset.
The curves here are built from that slope, so they need nothing but a
ranking, an outcome and a randomized treatment.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .._result_serialize import ResultProtocolMixin
from ..exceptions import DataInsufficient, MethodIncompatibility

__all__ = ["cate_gain_curve", "CATEGainCurveResult"]


def _slope(y: np.ndarray, t: np.ndarray) -> float:
    """OLS slope of ``y`` on ``t``; NaN when ``t`` does not vary."""
    tc = t - t.mean()
    denom = float(np.sum(tc**2))
    if denom <= 0:
        return float("nan")
    return float(np.sum(tc * y) / denom)


@dataclass(repr=False)
class CATEGainCurveResult(ResultProtocolMixin):
    """Output of :func:`cate_gain_curve`.

    Attributes
    ----------
    auc : float
        Sum of the (normalized) cumulative gain curve over its steps: zero
        in expectation for a ranking unrelated to the effect, positive when
        the units ranked first respond more.
    ate : float
        Slope of the outcome on the treatment in the whole sample.
    curve : pd.DataFrame
        One row per step: ``share`` of the sample (the top-ranked units),
        ``n``, ``cumulative_effect`` (slope among them) and
        ``cumulative_gain`` (``share`` times that slope, minus ``share``
        times ``ate`` when normalized).
    by_quantile : pd.DataFrame
        Slope within each quantile group of the ranking, lowest first.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> df = pd.DataFrame({"x": rng.uniform(0, 1, 2000),
    ...                    "dose": rng.uniform(0, 10, 2000)})
    >>> df["y"] = (1 + 2 * df["x"]) * df["dose"] + rng.normal(size=2000)
    >>> res = sp.cate_gain_curve(df, cate="x", y="y", treat="dose")
    >>> type(res).__name__
    'CATEGainCurveResult'
    >>> list(res.curve.columns)
    ['share', 'n', 'cumulative_effect', 'cumulative_gain']
    """

    auc: float
    ate: float
    curve: pd.DataFrame
    by_quantile: pd.DataFrame
    normalize: bool = True
    n_obs: int = 0
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"CATEGainCurveResult(auc={self.auc:.6g}, ate={self.ate:.6g}, "
            f"n_obs={self.n_obs}, steps={len(self.curve)})"
        )

    def summary(self) -> str:
        lines = [
            "CATE ranking: cumulative gain on randomized data",
            "-" * 52,
            f"  N                 : {self.n_obs:,}",
            f"  Average effect    : {self.ate:.6g}",
            f"  Area under curve  : {self.auc:.6g}"
            + ("  (normalized)" if self.normalize else ""),
            "",
            "  Effect by quantile of the ranking (lowest first):",
            self.by_quantile.to_string(index=False),
        ]
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "auc": float(self.auc),
            "ate": float(self.ate),
            "normalize": bool(self.normalize),
            "n_obs": int(self.n_obs),
            "curve": self.curve.to_dict(orient="list"),
            "by_quantile": self.by_quantile.to_dict(orient="list"),
        }

    def plot(self, ax: Any = None) -> Any:
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=(7, 4))
        ax.plot(100 * self.curve["share"], self.curve["cumulative_gain"])
        ax.axhline(0.0, linestyle="--", color="black", linewidth=1)
        ax.set_xlabel("Top % by predicted effect")
        ax.set_ylabel("Cumulative gain" + (" (normalized)" if self.normalize else ""))
        return ax


def cate_gain_curve(
    data: pd.DataFrame,
    cate: Any,
    y: str,
    treat: str,
    n_steps: int = 100,
    n_quantiles: int = 10,
    normalize: bool = True,
    ascending: bool = False,
) -> CATEGainCurveResult:
    """Cumulative gain curve of a CATE ranking on randomized data.

    The sample is sorted by the prediction, largest first. At each step the
    effect among the top-ranked units is the OLS slope of the outcome on the
    treatment in that subset (a difference in means when the treatment is
    0/1), and the gain is that slope times the share of the sample used. A
    ranking that puts responsive units first has a curve above the straight
    line from zero to the average effect. The normalized curve subtracts
    that line, and its sum over the steps is the reported area.

    Parameters
    ----------
    data : pd.DataFrame
        Evaluation data in which ``treat`` was randomized. It should not be
        the data the predictions were fitted on.
    cate : str or array-like
        Predicted effects (or any score that ranks units by effect): a
        column name or one value per row of ``data``.
    y : str
        Outcome column.
    treat : str
        Treatment column, binary or continuous.
    n_steps : int, default 100
        Number of points on the curve.
    n_quantiles : int, default 10
        Number of groups in the effect-by-quantile table.
    normalize : bool, default True
        Subtract the random-ranking line from the gain curve.
    ascending : bool, default False
        Rank the smallest predictions first (for an effect where lower is
        better).

    Returns
    -------
    CATEGainCurveResult

    Notes
    -----
    The slope identifies the effect in a subset only if the treatment is
    independent of the potential outcomes there, which randomization
    delivers and observational data does not. On observational data the
    curve ranks units by a confounded association. The slope also averages
    a nonlinear dose response with weights that depend on the distribution
    of the dose in the subset. No standard errors are computed; to compare
    two rankings resample the evaluation data and recompute both areas.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> n = 4000
    >>> x = rng.uniform(0, 1, n)
    >>> dose = rng.uniform(0, 10, n)
    >>> df = pd.DataFrame({"x": x, "dose": dose,
    ...                    "y": (1 + 2 * x) * dose + rng.normal(size=n)})
    >>> good = sp.cate_gain_curve(df, cate="x", y="y", treat="dose")
    >>> noise = sp.cate_gain_curve(df, cate=rng.normal(size=n), y="y",
    ...                            treat="dose")
    >>> bool(good.auc > abs(noise.auc))
    True
    """
    missing = [c for c in (y, treat) if c not in data.columns]
    if missing:
        raise MethodIncompatibility(
            f"cate_gain_curve: columns not found in data: {missing}",
            diagnostics={"missing": missing},
        )
    if isinstance(cate, str):
        if cate not in data.columns:
            raise MethodIncompatibility(
                f"cate_gain_curve: column {cate!r} not found in data."
            )
        score = data[cate].to_numpy(dtype=float)
    else:
        score = np.asarray(cate, dtype=float).ravel()
        if score.shape[0] != len(data):
            raise MethodIncompatibility(
                f"cate_gain_curve: cate has {score.shape[0]} values for "
                f"{len(data)} rows."
            )
    y_arr = data[y].to_numpy(dtype=float)
    t_arr = data[treat].to_numpy(dtype=float)
    ok = np.isfinite(score) & np.isfinite(y_arr) & np.isfinite(t_arr)
    score, y_arr, t_arr = score[ok], y_arr[ok], t_arr[ok]
    n = int(score.shape[0])
    n_steps, n_quantiles = int(n_steps), int(n_quantiles)
    if n_steps < 1 or n_quantiles < 1:
        raise MethodIncompatibility("n_steps and n_quantiles must be positive.")
    if n < max(2 * n_steps, 2 * n_quantiles, 4):
        raise DataInsufficient(
            f"cate_gain_curve: {n} complete rows are too few for "
            f"n_steps={n_steps} and n_quantiles={n_quantiles}.",
            diagnostics={"n_obs": n},
        )
    ate = _slope(y_arr, t_arr)
    if not np.isfinite(ate):
        raise DataInsufficient("cate_gain_curve: the treatment does not vary.")

    order = np.argsort(-score if not ascending else score, kind="stable")
    y_s, t_s = y_arr[order], t_arr[order]
    sizes = np.round(np.linspace(n / n_steps, n, n_steps)).astype(int)
    sizes = np.clip(sizes, 2, n)
    effects = np.array([_slope(y_s[:k], t_s[:k]) for k in sizes])
    share = sizes / n
    gain = (effects - (ate if normalize else 0.0)) * share
    curve = pd.DataFrame(
        {
            "share": share,
            "n": sizes,
            "cumulative_effect": effects,
            "cumulative_gain": gain,
        }
    )
    auc = float(np.nansum(gain))

    groups = pd.qcut(pd.Series(score), q=n_quantiles, duplicates="drop")
    rows: List[Dict[str, Any]] = []
    for interval, idx in pd.Series(np.arange(n)).groupby(groups, observed=True):
        ii = idx.to_numpy()
        rows.append(
            {
                "quantile": len(rows) + 1,
                "score_low": float(interval.left),
                "score_high": float(interval.right),
                "n": int(ii.size),
                "effect": _slope(y_arr[ii], t_arr[ii]),
            }
        )
    by_quantile = pd.DataFrame(rows)

    n_flat: Optional[int] = int(np.sum(~np.isfinite(effects)))
    return CATEGainCurveResult(
        auc=auc,
        ate=float(ate),
        curve=curve,
        by_quantile=by_quantile,
        normalize=bool(normalize),
        n_obs=n,
        diagnostics={
            "n_dropped": int((~ok).sum()),
            "n_steps_without_treatment_variation": n_flat,
        },
    )
