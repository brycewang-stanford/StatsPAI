"""
Counterfactual-Fair Conformal Prediction (arXiv 2510.08724, 2025).

Adds counterfactual fairness constraints to conformal ITE intervals:
the interval coverage rate should be the same across protected
subgroups (e.g. race, sex). Achieved by stratified conformalization —
compute calibration quantiles separately per subgroup and report
both per-group and pooled intervals.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from .._result_serialize import ResultProtocolMixin
from ..exceptions import DataInsufficient


@dataclass
class FairConformalResult(ResultProtocolMixin):
    """Fairness-aware conformal ITE intervals.

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np
    >>> import pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> n = 240
    >>> g = rng.integers(0, 2, size=n)
    >>> x1 = rng.normal(size=n)
    >>> x2 = rng.normal(size=n)
    >>> d = rng.binomial(1, 0.5, size=n)
    >>> y = 1.0 + 0.7 * x1 + 0.4 * x2 + 1.2 * d + 0.5 * g + rng.normal(0, 0.5, n)
    >>> df = pd.DataFrame({"y": y, "d": d, "x1": x1, "x2": x2, "grp": g})
    >>> res = sp.conformal_fair_ite(
    ...     df, y="y", treat="d", covariates=["x1", "x2"],
    ...     protected="grp", seed=1)
    >>> bool(res.point_estimate.shape == (240,))
    True
    >>> sorted(res.group_coverage_targets.keys())
    ['0', '1']
    """

    intervals: np.ndarray
    point_estimate: np.ndarray
    group_assignment: np.ndarray
    group_widths: Dict[str, float]
    group_coverage_targets: Dict[str, float]
    coverage_target: float

    def summary(self) -> str:
        rows = [
            "Counterfactual-Fair Conformal ITE",
            "=" * 42,
            f"  Target coverage : {1 - self.coverage_target:.2f}",
            "  Per-group widths:",
        ]
        for g, w in self.group_widths.items():
            rows.append(f"    Group {g!r:>10s}: width = {w:.4f}")
        return "\n".join(rows)


def conformal_fair_ite(
    data: pd.DataFrame,
    y: str,
    treat: str,
    covariates: List[str],
    protected: str,
    test_data: Optional[pd.DataFrame] = None,
    alpha: float = 0.1,
    seed: int = 0,
) -> FairConformalResult:
    """
    Counterfactual-fair conformal ITE intervals.

    Parameters
    ----------
    data : pd.DataFrame
    y, treat : str
    covariates : list of str
        Predictive features. ``protected`` is **excluded** from the
        outcome regression to preserve counterfactual fairness, but
        used downstream for stratified calibration.
    protected : str
        Protected attribute column (categorical).
    test_data : pd.DataFrame, optional
    alpha : float, default 0.1
        Miscoverage of the effect interval. Within each protected group,
        each arm's residuals get a split-conformal quantile at
        ``1 - alpha / 2`` and the half-width is their sum, so the interval
        targets coverage of ``Y(1) - Y(0)`` of at least ``1 - alpha`` in
        each group (union bound; conservative) when treatment is
        randomised given the covariates.
    seed : int

    Returns
    -------
    FairConformalResult

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np
    >>> import pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> n = 240
    >>> g = rng.integers(0, 2, size=n)
    >>> x1 = rng.normal(size=n)
    >>> x2 = rng.normal(size=n)
    >>> d = rng.binomial(1, 0.5, size=n)
    >>> y = 1.0 + 0.7 * x1 + 0.4 * x2 + 1.2 * d + 0.5 * g + rng.normal(0, 0.5, n)
    >>> df = pd.DataFrame({"y": y, "d": d, "x1": x1, "x2": x2, "grp": g})
    >>> res = sp.conformal_fair_ite(
    ...     df, y="y", treat="d", covariates=["x1", "x2"],
    ...     protected="grp", alpha=0.1, seed=1)
    >>> isinstance(res, sp.FairConformalResult)
    True
    >>> res.intervals.shape
    (240, 2)
    >>> sorted(res.group_widths.keys())
    ['0', '1']
    >>> widths = res.intervals[:, 1] - res.intervals[:, 0]
    >>> bool((widths > 0).all())
    True
    >>> print(res.summary())  # doctest: +SKIP
    """
    from sklearn.linear_model import LinearRegression

    cov_no_protected = [c for c in covariates if c != protected]
    df = data[[y, treat, protected] + cov_no_protected].dropna().reset_index(drop=True)
    if df[treat].nunique() != 2:
        raise ValueError("Fair conformal requires binary treatment.")
    Y = df[y].to_numpy(float)
    D = df[treat].to_numpy(int)
    X = df[cov_no_protected].to_numpy(float)
    G = df[protected].to_numpy()
    n = len(df)
    rng = np.random.default_rng(seed)

    # 50/50 split
    perm = rng.permutation(n)
    train = perm[: n // 2]
    cal = perm[n // 2 :]

    m1 = LinearRegression().fit(X[train][D[train] == 1], Y[train][D[train] == 1])
    m0 = LinearRegression().fit(X[train][D[train] == 0], Y[train][D[train] == 0])

    # Per-group calibration. An interval for the effect Y(1) - Y(0) has to
    # hold both potential outcomes: each arm gets its own split-conformal
    # quantile of |Y - mu_d(X)| at level 1 - alpha/2, and the half-width is
    # their sum (union bound), as in sp.conformal_cate. Before the Oct 2026
    # scan one quantile of the pooled single-arm residuals at 1 - alpha was
    # used, which bounds one potential outcome only: a nominal 90% interval
    # covered the effect about 74% of the time with equal-variance noise.
    #
    # A group with fewer than 5 calibration points in an arm falls back to
    # the *conservative* value -- the largest half-width among the groups
    # that have enough data -- rather than blending groups, which would
    # destroy the group-specific coverage exactly for the small groups fair
    # conformal is designed to protect.
    def _arm_quantile(resid: np.ndarray) -> float:
        k = int(np.ceil((len(resid) + 1) * (1 - alpha / 2)))
        return float(np.sort(resid)[min(k, len(resid)) - 1])

    def _half_width(mask_cal: np.ndarray) -> Optional[float]:
        q = 0.0
        for arm, model in ((1, m1), (0, m0)):
            sel = mask_cal & (D[cal] == arm)
            if sel.sum() < 5:
                return None
            q += _arm_quantile(np.abs(Y[cal][sel] - model.predict(X[cal][sel])))
        return q

    group_q: Dict = {}
    group_widths: Dict[str, float] = {}
    group_fallback: Dict[str, bool] = {}
    for g in np.unique(G):
        q_g = _half_width(G[cal] == g)
        group_q[g] = q_g
        group_fallback[str(g)] = q_g is None
        if q_g is not None:
            group_widths[str(g)] = 2 * q_g

    valid_qs = [q for q in group_q.values() if q is not None]
    if valid_qs:
        fallback_q = float(max(valid_qs))
    else:
        import warnings as _warnings

        _warnings.warn(
            "conformal_fair_ite: every group has fewer than 5 calibration "
            "points in an arm; falling back to the pooled quantile. "
            "Per-group coverage guarantee does NOT hold.",
            stacklevel=2,
        )
        pooled = _half_width(np.ones(len(cal), dtype=bool))
        if pooled is None:
            raise DataInsufficient(
                "conformal_fair_ite: fewer than 5 calibration points in a "
                "treatment arm.",
                recovery_hint="Use more data or a larger calibration share.",
            )
        fallback_q = pooled
    for g, q in list(group_q.items()):
        if q is None:
            group_q[g] = fallback_q
            group_widths[str(g)] = 2 * fallback_q

    test_df = test_data if test_data is not None else df
    test_df = test_df[cov_no_protected + [protected]].dropna().reset_index(drop=True)
    Xt = test_df[cov_no_protected].to_numpy(float)
    Gt = test_df[protected].to_numpy()
    point = m1.predict(Xt) - m0.predict(Xt)

    intervals = np.zeros((len(test_df), 2))
    for i, g in enumerate(Gt):
        q = group_q.get(g, np.median(list(group_q.values())))
        intervals[i] = [point[i] - q, point[i] + q]

    group_targets = {str(g): 1 - alpha for g in np.unique(G)}

    _result = FairConformalResult(
        intervals=intervals,
        point_estimate=point,
        group_assignment=Gt,
        group_widths=group_widths,
        group_coverage_targets=group_targets,
        coverage_target=alpha,
    )
    try:
        from ..output._lineage import attach_provenance as _attach_prov

        _attach_prov(
            _result,
            function="sp.conformal_causal.conformal_fair_ite",
            params={
                "y": y,
                "treat": treat,
                "covariates": list(covariates),
                "protected": protected,
                "alpha": alpha,
                "seed": seed,
            },
            data=data,
            overwrite=False,
        )
    except Exception:  # pragma: no cover
        pass
    return _result
