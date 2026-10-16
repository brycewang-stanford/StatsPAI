"""Gain curves for a CATE ranking on randomized data, any treatment type.

:func:`cate_eval` scores a ranking with RATE on doubly robust scores of a
0/1 treatment. With a dose (a discount, a price, a credit line) there is no
such score, but on data where the treatment was randomized the effect in any
subset is still the slope of the outcome on the treatment in that subset.
The curves here are built from that slope, so they need nothing but a
ranking, an outcome and a randomized treatment.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .._input_validation import check_columns, missing_named_columns
from .._result_serialize import ResultProtocolMixin
from ..exceptions import ColumnNotFound, DataInsufficient, MethodIncompatibility

__all__ = ["cate_gain_curve", "CATEGainCurveResult"]


def _slope(y: np.ndarray, t: np.ndarray) -> float:
    """OLS slope of ``y`` on ``t``; NaN when ``t`` does not vary."""
    tc = t - t.mean()
    denom = float(np.sum(tc**2))
    if denom <= 0:
        return float("nan")
    return float(np.sum(tc * y) / denom)


def _gain(
    score: np.ndarray,
    y: np.ndarray,
    t: np.ndarray,
    n_steps: int,
    normalize: bool,
    ascending: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Step sizes, cumulative effects and gains of one ranking."""
    n = score.shape[0]
    key = score if ascending else -score
    order = np.argsort(key, kind="stable")
    y_s, t_s, key_s = y[order], t[order], key[order]
    sizes = np.clip(np.round(np.linspace(n / n_steps, n, n_steps)).astype(int), 2, n)
    full = _slope(y, t)
    effects = np.array([_top_slope(y_s, t_s, key_s, int(k), full) for k in sizes])
    base = full if normalize else 0.0
    return sizes, effects, (effects - base) * (sizes / n)


def _top_slope(
    y_s: np.ndarray, t_s: np.ndarray, key_s: np.ndarray, k: int, full: float
) -> float:
    """Slope among the top ``k`` of a sorted sample, tied scores together.

    When the cut at ``k`` falls inside a group of equal scores, which of
    them make the top ``k`` is an accident of row order. The group then
    enters whole, each member with weight (places left) / (group size), so
    the result does not depend on the order of the rows. A cut between
    distinct scores, or inside a group of identical observations, is the
    plain slope of the first ``k`` rows.
    """
    n = key_s.shape[0]
    if k >= n or key_s[k - 1] != key_s[k]:
        return _slope(y_s[:k], t_s[:k])
    lo = int(np.searchsorted(key_s, key_s[k - 1], side="left"))
    hi = int(np.searchsorted(key_s, key_s[k - 1], side="right"))
    y_g, t_g = y_s[lo:hi], t_s[lo:hi]
    if bool(np.all(y_g == y_g[0]) and np.all(t_g == t_g[0])):
        return _slope(y_s[:k], t_s[:k])
    if lo == 0:
        # Only the tied group is in play, equally weighted.
        return full if hi == n else _slope(y_g, t_g)
    w = np.ones(hi)
    w[lo:] = (k - lo) / (hi - lo)
    y_h, t_h = y_s[:hi], t_s[:hi]
    tc = t_h - float(np.sum(w * t_h) / np.sum(w))
    denom = float(np.sum(w * tc**2))
    if denom <= 0:
        return float("nan")
    return float(np.sum(w * tc * y_h) / denom)


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
    auc_se, auc_ci : float and (float, float), or None
        Bootstrap standard error and percentile interval of ``auc``; ``None``
        unless ``n_boot`` was set.

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
    auc_se: Optional[float] = None
    auc_ci: Optional[Tuple[float, float]] = None
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
            + ("  (normalized)" if self.normalize else "")
            + (
                f"  [SE {self.auc_se:.4g}, CI {self.auc_ci[0]:.4g} to "
                f"{self.auc_ci[1]:.4g}]"
                if self.auc_se is not None and self.auc_ci is not None
                else ""
            ),
            "",
            "  Effect by quantile of the ranking (lowest first):",
            self.by_quantile.to_string(index=False),
        ]
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "auc": float(self.auc),
            "auc_se": self.auc_se,
            "auc_ci": None if self.auc_ci is None else list(self.auc_ci),
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
    n_boot: int = 0,
    cluster: Optional[str] = None,
    alpha: float = 0.05,
    seed: Optional[int] = None,
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
    n_boot : int, default 0
        Bootstrap replications for the standard error and the percentile
        interval of the area. ``0`` computes neither.
    cluster : str, optional
        Column whose groups are resampled whole (repeated days of one
        store, say). Default: rows are resampled.
    alpha : float, default 0.05
        Level of the bootstrap interval.
    seed : int, optional
        Seed for the bootstrap.

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
    of the dose in the subset. The bootstrap of ``n_boot`` resamples the
    evaluation data with the predictions held fixed, so it describes the
    sampling error of the evaluation, not that of the model behind the
    predictions.

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
    missing = missing_named_columns(
        {"y": y, "treat": treat, "cate": cate if isinstance(cate, str) else None},
        data.columns,
    )
    if missing:
        raise ColumnNotFound.from_missing(
            missing,
            list(data.columns),
            context="sp.cate_gain_curve",
            extra_diagnostics={"missing": list(missing.values())},
        )
    if isinstance(cate, str):
        score = data[cate].to_numpy(dtype=float)
    else:
        if isinstance(cate, pd.Series) and not cate.index.equals(data.index):
            same = (
                len(cate) == len(data)
                and cate.index.is_unique
                and data.index.is_unique
                and cate.index.isin(data.index).all()
            )
            if not same:
                raise MethodIncompatibility(
                    "cate_gain_curve: the index of the cate Series does not "
                    "match the index of data, so the predictions cannot be "
                    "paired with rows.",
                    recovery_hint=(
                        "Give the Series the index of data, or pass "
                        "cate.to_numpy() if the values are already in row "
                        "order."
                    ),
                )
            cate = cate.reindex(data.index)
        score = np.asarray(cate, dtype=float).ravel()
        if score.shape[0] != len(data):
            raise MethodIncompatibility(
                f"cate_gain_curve: cate has {score.shape[0]} values for "
                f"{len(data)} rows."
            )
    arrays = []
    for role, col in (("outcome", y), ("treatment", treat)):
        try:
            arrays.append(data[col].to_numpy(dtype=float))
        except (TypeError, ValueError) as exc:
            raise MethodIncompatibility(
                f"cate_gain_curve: the {role} column {col!r} is not numeric "
                f"(dtype {data[col].dtype}).",
                recovery_hint=(
                    "Code the column as numbers: 0/1 for a two-arm "
                    "treatment, or the dose."
                ),
                diagnostics={"column": col, "dtype": str(data[col].dtype)},
            ) from exc
    y_arr, t_arr = arrays
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

    sizes, effects, gain = _gain(score, y_arr, t_arr, n_steps, normalize, ascending)
    share = sizes / n
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

    auc_se: Optional[float] = None
    auc_ci: Optional[Tuple[float, float]] = None
    n_boot = int(n_boot)
    if n_boot > 0:
        if not 0.0 < alpha < 1.0:
            raise MethodIncompatibility("alpha must lie strictly between 0 and 1.")
        rng = np.random.default_rng(seed)
        blocks: Optional[List[np.ndarray]] = None
        if cluster is not None:
            check_columns(data, {"cluster": cluster}, function="sp.cate_gain_curve")
            codes = pd.factorize(data[cluster].to_numpy()[ok])[0]
            blocks = [np.flatnonzero(codes == g) for g in range(codes.max() + 1)]
        draws = np.empty(n_boot)
        for b in range(n_boot):
            if blocks is None:
                idx = rng.integers(0, n, n)
            else:
                pick = rng.integers(0, len(blocks), len(blocks))
                idx = np.concatenate([blocks[g] for g in pick])
            _, _, g_b = _gain(
                score[idx], y_arr[idx], t_arr[idx], n_steps, normalize, ascending
            )
            draws[b] = np.nansum(g_b)
        auc_se = float(np.std(draws, ddof=1))
        lo, hi = np.quantile(draws, [alpha / 2, 1 - alpha / 2])
        auc_ci = (float(lo), float(hi))

    n_flat: Optional[int] = int(np.sum(~np.isfinite(effects)))
    if n_flat:
        warnings.warn(
            f"cate_gain_curve: the treatment does not vary among the "
            f"top-ranked units at {n_flat} of {len(sizes)} step(s) of the "
            "curve (all of them are in one arm), so the cumulative effect "
            "there is NaN and those steps add nothing to the area.",
            UserWarning,
            stacklevel=2,
        )
    return CATEGainCurveResult(
        auc=auc,
        ate=float(ate),
        curve=curve,
        by_quantile=by_quantile,
        normalize=bool(normalize),
        n_obs=n,
        auc_se=auc_se,
        auc_ci=auc_ci,
        diagnostics={
            "n_dropped": int((~ok).sum()),
            "n_steps_without_treatment_variation": n_flat,
        },
    )
