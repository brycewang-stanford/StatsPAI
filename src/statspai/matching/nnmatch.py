"""Nearest-neighbour matching on covariates (Abadie & Imbens 2006, 2011).

This is the estimator Stata ships as ``teffects nnmatch`` (and, before it,
``nnmatch``): match each unit to its ``m`` nearest units of the opposite
treatment arm in a covariate metric, impute the missing potential outcome by
their mean, optionally correct the bias that inexact matches leave behind
with a within-arm regression, and report the variance that accounts for
controls being used more than once.

Notation follows "Methods and formulas" of ``[CAUSAL] teffects nnmatch``
(Stata 18). For unit ``i`` with treatment ``t_i``:

* ``Omega_m(i)`` -- the ``m`` nearest opposite-arm units, **every tie at the
  ``m``-th distance included** (so the set can hold more than ``m`` units);
* ``yhat_{1-t_i, i}`` -- the mean outcome over ``Omega_m(i)``;
* ``K_m(i)`` -- the matching weight unit ``i`` carries as a match for others,
  ``sum_j 1[i in Omega_m(j)] / |Omega_m(j)|``, and ``K'_m(i)`` the same sum
  with ``1 / |Omega_m(j)|^2``.

Then::

    ATE   tau   = (1/N)  sum_i (yhat_1i - yhat_0i)
    ATET  delta = (1/N1) sum_{i: t_i = 1} (y_i - yhat_0i)

    Var(tau)   = sum_i [ (yhat_1i - yhat_0i - tau)^2
                         + xi2_i {K_m(i)^2 + 2 K_m(i) - K'_m(i)} ] / N^2
    Var(delta) = sum_i [ t_i (y_i - yhat_0i - delta)^2
                         + (1 - t_i) xi2_i {K_m(i)^2 - K'_m(i)} ] / N1^2

``xi2_i`` is the conditional outcome variance. ``vce='robust'`` estimates it
unit by unit as the sample variance of the outcome over the unit and its
``h`` nearest *same-arm* units, ties kept (``vce_nn = h``, Stata's
``vce(robust, nn(h))``, default 2). ``vce='iid'`` assumes it constant and
estimates it as half the mean squared discrepancy between a unit and each
of its matches. Both were pinned by reconstructing ``e(V)`` term by term on
the NSW-CPS sample: the same-arm set of ``nn(2)`` holds three units, not
two, and the homoskedastic estimate averages the squared pairwise
differences, not the square of the averaged one.

The bias adjustment replaces ``y_j`` in the imputation by
``y_j + (x_i - x_j)' beta_{t_j}``, with ``beta_t`` from the regression of the
outcome on the adjustment covariates among arm ``t``, weighted by
``K_m``, so only units that serve as matches contribute.

References
----------
abadie2006large, abadie2011bias
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

from ..core.results import CausalResult
from ..exceptions import DataInsufficient, MethodIncompatibility

__all__ = ["nnmatch"]

_METRICS = ("mahalanobis", "ivariance", "euclidean")
# Rows per block of the neighbour search; 64 x n x k doubles at a time.
_CHUNK = 64


def _metric_transform(X: np.ndarray, metric: str) -> np.ndarray:
    """Columns rescaled so that the metric is Euclidean distance on them.

    ``mahalanobis`` uses the full-sample covariance of the matching
    covariates, ``ivariance`` its diagonal, ``euclidean`` the identity.
    """
    if metric == "euclidean":
        return X.copy()
    if X.shape[0] < 2:  # pragma: no cover - guarded by the caller
        return X.copy()
    S = np.atleast_2d(np.cov(X, rowvar=False, ddof=1))
    if metric == "ivariance":
        sd = np.sqrt(np.diag(S))
        if np.any(sd <= 0):
            raise MethodIncompatibility(
                "nnmatch: a matching covariate is constant, so the "
                "inverse-variance metric is undefined.",
                recovery_hint="Drop the constant covariate.",
                diagnostics={"constant_columns": np.flatnonzero(sd <= 0).tolist()},
            )
        return X / sd
    try:
        L = np.linalg.cholesky(np.linalg.inv(S))
    except np.linalg.LinAlgError as exc:
        raise MethodIncompatibility(
            "nnmatch: the covariance matrix of the matching covariates is "
            "singular, so the Mahalanobis metric is undefined.",
            recovery_hint=(
                "Drop collinear covariates, or use metric='ivariance' / " "'euclidean'."
            ),
        ) from exc
    return X @ L


def _neighbour_sets(
    Xw_rows: np.ndarray,
    Xw_pool: np.ndarray,
    k: int,
    dtol: float,
    *,
    exact_rows: Optional[np.ndarray] = None,
    exact_pool: Optional[np.ndarray] = None,
    caliper: Optional[float] = None,
    self_offset: Optional[int] = None,
) -> List[np.ndarray]:
    """For each row, the positions in the pool of its ``k`` nearest units,
    ties at the ``k``-th distance (within ``dtol``) included.

    ``self_offset`` marks the rows as the pool's own rows starting at that
    position, for same-arm searches; the unit itself then sits at distance
    zero and counts towards ``k``, as Stata's ``nn(h)`` counts it.
    """
    del self_offset  # the unit is found at distance zero like any other
    out: List[np.ndarray] = []
    n_pool = Xw_pool.shape[0]
    k_eff = int(min(max(k, 1), n_pool))
    for start in range(0, Xw_rows.shape[0], _CHUNK):
        block = Xw_rows[start : start + _CHUNK]
        diff = block[:, None, :] - Xw_pool[None, :, :]
        dist = np.sqrt(np.einsum("ijk,ijk->ij", diff, diff))
        if exact_rows is not None and exact_pool is not None:
            dist[exact_rows[start : start + _CHUNK][:, None] != exact_pool[None, :]] = (
                np.inf
            )
        if caliper is not None:
            dist[dist > caliper] = np.inf
        thr = np.partition(dist, k_eff - 1, axis=1)[:, k_eff - 1]
        for r in range(block.shape[0]):
            if not np.isfinite(thr[r]):
                # Fewer than k admissible units: keep whatever is admissible.
                out.append(np.flatnonzero(np.isfinite(dist[r])))
            else:
                out.append(np.flatnonzero(dist[r] <= thr[r] + dtol))
    return out


def _wls_slopes(X: np.ndarray, y: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Slopes (constant excluded) of the ``w``-weighted regression of ``y``
    on ``X`` over the rows with positive weight."""
    use = w > 0
    if use.sum() <= X.shape[1]:
        raise DataInsufficient(
            "nnmatch: too few matched units to fit the bias-adjustment " "regression.",
            recovery_hint="Use fewer bias-adjustment covariates.",
            diagnostics={"n_matched": int(use.sum()), "n_covariates": X.shape[1]},
        )
    sw = np.sqrt(w[use])
    A = np.column_stack([np.ones(use.sum()), X[use]]) * sw[:, None]
    beta = np.linalg.lstsq(A, y[use] * sw, rcond=None)[0]
    return np.asarray(beta[1:], dtype=float)


def _weighted_moments(x: np.ndarray, w: np.ndarray) -> Tuple[float, float]:
    """Mean and variance with frequency-style weights (``sum(w) - 1``)."""
    sw = float(w.sum())
    mean = float(np.sum(w * x) / sw)
    var = float(np.sum(w * (x - mean) ** 2) / (sw - 1.0)) if sw > 1 else float("nan")
    return mean, var


def _balance_table(
    X: np.ndarray,
    t: np.ndarray,
    w_matched: np.ndarray,
    names: Sequence[str],
) -> pd.DataFrame:
    """Standardized differences and variance ratios, raw and matched.

    ``std_diff = (mean_1 - mean_0) / sqrt((var_1 + var_0) / 2)`` and
    ``var_ratio = var_1 / var_0``, as Stata ``tebalance summarize``. The
    matched columns weight every unit by how often it enters the matched
    sample.
    """
    rows = []
    ones = np.ones(len(t))
    for j, name in enumerate(names):
        rec: Dict[str, Any] = {"variable": name}
        for label, w in (("raw", ones), ("matched", w_matched)):
            m1, v1 = _weighted_moments(X[t == 1, j], w[t == 1])
            m0, v0 = _weighted_moments(X[t == 0, j], w[t == 0])
            pooled = np.sqrt((v1 + v0) / 2.0)
            rec[f"std_diff_{label}"] = (m1 - m0) / pooled if pooled > 0 else 0.0
            rec[f"var_ratio_{label}"] = v1 / v0 if v0 > 0 else float("nan")
        rows.append(rec)
    return pd.DataFrame(rows)


def _column_list(value: Any, name: str) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    try:
        cols = list(value)
    except TypeError as exc:
        raise MethodIncompatibility(
            f"nnmatch: {name} must be a column name or a list of column names.",
            diagnostics={name: repr(value)},
        ) from exc
    if not all(isinstance(c, str) for c in cols):
        raise MethodIncompatibility(
            f"nnmatch: {name} must contain column names.",
            diagnostics={name: repr(value)},
        )
    return cols


def nnmatch(
    data: pd.DataFrame,
    y: str,
    treat: str,
    covariates: Sequence[str],
    *,
    estimand: str = "ATT",
    n_matches: int = 1,
    metric: str = "mahalanobis",
    exact: Optional[Union[str, Sequence[str]]] = None,
    caliper: Optional[float] = None,
    bias_adjust: Optional[Union[bool, str, Sequence[str]]] = None,
    vce: str = "robust",
    vce_nn: int = 2,
    dtol: float = 1e-8,
    alpha: float = 0.05,
) -> CausalResult:
    """Abadie-Imbens nearest-neighbour matching on covariates.

    Reached as ``sp.match(..., method='nnmatch')``. Reproduces Stata
    ``teffects nnmatch (y x) (treat)``: matching with replacement, every tie
    kept, the Abadie-Imbens (2006) variance and the Abadie-Imbens (2011)
    bias adjustment.

    Parameters
    ----------
    data : pd.DataFrame
    y, treat : str
        Outcome and 0/1 treatment columns.
    covariates : list of str
        Matching covariates.
    estimand : {'ATT', 'ATE'}, default 'ATT'
        ``'ATT'`` is Stata's ``atet``; Stata's own default is the ATE.
    n_matches : int, default 1
        Matches per unit, ``nneighbor(#)``. Ties at the last distance are
        all kept, so a unit can have more.
    metric : {'mahalanobis', 'ivariance', 'euclidean'}, default 'mahalanobis'
        Distance on the matching covariates: the inverse of their full-sample
        covariance, the inverse of its diagonal, or the identity. Note that
        ``sp.match(distance='euclidean')`` standardises the covariates first,
        which is the ``'ivariance'`` metric here; ``'euclidean'`` here is on
        the raw scale, as in Stata.
    exact : str or list of str, optional
        Covariates on which a match must agree exactly, ``ematch()``. They
        need not be among ``covariates``.
    caliper : float, optional
        Largest admissible distance, in units of the metric.
    bias_adjust : bool, str or list of str, optional
        Covariates of the bias-adjustment regression, ``biasadj()``.
        ``True`` uses the matching covariates.
    vce : {'robust', 'iid'}, default 'robust'
        ``'robust'`` estimates the conditional outcome variance unit by unit
        from ``vce_nn`` nearest same-arm units; ``'iid'`` assumes it
        constant.
    vce_nn : int, default 2
        ``h`` in Stata's ``vce(robust, nn(h))``: same-arm neighbours used,
        with the unit itself, for its conditional variance.
    dtol : float, default 1e-8
        Distances within ``dtol`` of the ``n_matches``-th are ties
        (Stata's ``dtolerance()``).
    alpha : float, default 0.05

    Returns
    -------
    CausalResult
        ``estimate`` / ``se`` / ``ci`` use the normal distribution, as
        ``teffects`` does. ``detail`` is the balance table (standardized
        differences and variance ratios, raw and matched). ``model_info``
        carries ``matches_min`` / ``matches_max``, ``n_treated``,
        ``n_control`` and ``match_weights`` (``K_m`` by unit, a Series on the
        index of the estimation sample).

    Raises
    ------
    DataInsufficient
        When a unit has no admissible match under ``exact`` or ``caliper``.
        The exception's ``diagnostics['n_unmatched']`` counts them and its
        ``unmatched`` attribute is a boolean Series flagging them, the
        analogue of Stata's ``osample()``.

    Examples
    --------
    >>> import statspai as sp
    >>> df = sp.datasets.nsw_dw()
    >>> res = sp.match(df, y='re78', treat='treat',
    ...                covariates=['age', 'education', 're74', 're75'],
    ...                method='nnmatch', bias_adjust=True)
    >>> res.estimand
    'ATT'

    References
    ----------
    abadie2006large, abadie2011bias
    """
    estimand_key = str(estimand).upper()
    if estimand_key == "ATET":
        estimand_key = "ATT"
    if estimand_key not in ("ATT", "ATE"):
        raise MethodIncompatibility(
            f"nnmatch: estimand must be 'ATT' or 'ATE', got {estimand!r}.",
            diagnostics={"estimand": estimand},
        )
    metric_key = str(metric).lower()
    if metric_key not in _METRICS:
        raise MethodIncompatibility(
            f"nnmatch: metric must be one of {_METRICS}, got {metric!r}.",
            diagnostics={"metric": metric},
        )
    vce_key = str(vce).lower()
    if vce_key not in ("robust", "iid"):
        raise MethodIncompatibility(
            f"nnmatch: vce must be 'robust' or 'iid', got {vce!r}.",
            diagnostics={"vce": vce},
        )
    m = int(n_matches)
    h = int(vce_nn)
    if m < 1:
        raise MethodIncompatibility("nnmatch: n_matches must be at least 1.")
    if h < 1:
        raise MethodIncompatibility("nnmatch: vce_nn must be at least 1.")
    if caliper is not None and not float(caliper) > 0:
        raise MethodIncompatibility("nnmatch: caliper must be positive.")

    cov_cols = _column_list(covariates, "covariates")
    if not cov_cols:
        raise MethodIncompatibility("nnmatch: covariates must not be empty.")
    exact_cols = _column_list(exact, "exact")
    if bias_adjust is True:
        bias_cols = list(cov_cols)
    elif bias_adjust is False or bias_adjust is None:
        bias_cols = []
    else:
        bias_cols = _column_list(bias_adjust, "bias_adjust")

    needed = [y, treat, *cov_cols, *exact_cols, *bias_cols]
    missing = [c for c in dict.fromkeys(needed) if c not in data.columns]
    if missing:
        raise MethodIncompatibility(
            f"nnmatch: column(s) not found in data: {missing}.",
            diagnostics={"missing_columns": missing},
        )
    clean = data[list(dict.fromkeys(needed))].dropna()
    t = clean[treat].to_numpy()
    if not np.isin(t, (0, 1)).all():
        raise MethodIncompatibility(
            f"nnmatch: treatment column {treat!r} must be coded 0/1.",
            diagnostics={"values": np.unique(t)[:10].tolist()},
        )
    t = t.astype(int)
    n = len(clean)
    idx1, idx0 = np.flatnonzero(t == 1), np.flatnonzero(t == 0)
    n1, n0 = len(idx1), len(idx0)
    if n1 < 2 or n0 < 2:
        raise DataInsufficient(
            "nnmatch: need at least two treated and two control units.",
            diagnostics={"n_treated": n1, "n_control": n0},
        )

    Y = clean[y].to_numpy(dtype=float)
    X = clean[cov_cols].to_numpy(dtype=float)
    Xw = _metric_transform(X, metric_key)
    exact_code = None
    if exact_cols:
        exact_code = (
            clean[exact_cols]
            .apply(tuple, axis=1)
            .astype("category")
            .cat.codes.to_numpy()
        )
    cal = None if caliper is None else float(caliper)

    arms = {1: idx1, 0: idx0}
    # Units whose missing potential outcome must be imputed.
    targets = [1] if estimand_key == "ATT" else [1, 0]

    omega: Dict[int, np.ndarray] = {}
    unmatched = np.zeros(n, dtype=bool)
    for g in targets:
        own, other = arms[g], arms[1 - g]
        sets = _neighbour_sets(
            Xw[own],
            Xw[other],
            m,
            dtol,
            exact_rows=None if exact_code is None else exact_code[own],
            exact_pool=None if exact_code is None else exact_code[other],
            caliper=cal,
        )
        for pos, s in zip(own, sets):
            if s.size == 0:
                unmatched[pos] = True
            omega[int(pos)] = other[s]
    if unmatched.any():
        why = " and ".join(
            w
            for w, on in (("exact=", bool(exact_cols)), ("caliper=", cal is not None))
            if on
        )
        err = DataInsufficient(
            f"nnmatch: {int(unmatched.sum())} observation(s) have no "
            f"admissible match under {why or 'the matching rule'}.",
            recovery_hint=(
                "Relax the restriction, or drop the flagged observations "
                "(the exception's `unmatched` attribute) and re-estimate."
            ),
            diagnostics={"n_unmatched": int(unmatched.sum()), "n_obs": n},
        )
        err.unmatched = pd.Series(unmatched, index=clean.index, name="unmatched")
        raise err

    # Matching weights K_m, K'_m.
    K = np.zeros(n)
    Kp = np.zeros(n)
    n_set = np.zeros(n)
    for pos, s in omega.items():
        K[s] += 1.0 / len(s)
        Kp[s] += 1.0 / len(s) ** 2
        n_set[pos] = len(s)

    # Bias-adjustment slopes by donor arm.
    beta: Dict[int, np.ndarray] = {}
    Xb = None
    if bias_cols:
        Xb = clean[bias_cols].to_numpy(dtype=float)
        for g in targets:
            donor = 1 - g
            rows = arms[donor]
            beta[donor] = _wls_slopes(Xb[rows], Y[rows], K[rows])

    # Imputed potential outcomes. `pair_ms` keeps, per target, the mean
    # squared difference between its outcome and each match's (adjusted)
    # outcome, which is what the homoskedastic variance averages.
    y_obs_arm = {1: np.full(n, np.nan), 0: np.full(n, np.nan)}
    pair_vals: Dict[int, np.ndarray] = {}
    for pos, s in omega.items():
        donor = 1 - int(t[pos])
        vals = Y[s]
        if Xb is not None:
            vals = vals + (Xb[pos] - Xb[s]) @ beta[donor]
        y_obs_arm[donor][pos] = float(vals.mean())
        pair_vals[pos] = vals
    for g in (1, 0):
        y_obs_arm[g][arms[g]] = Y[arms[g]]

    if estimand_key == "ATT":
        diff = Y[idx1] - y_obs_arm[0][idx1]
        est = float(diff.mean())
        denom = float(n1)
    else:
        diff = y_obs_arm[1] - y_obs_arm[0]
        est = float(diff.mean())
        denom = float(n)

    # Conditional outcome variance xi2.
    if vce_key == "robust":
        # Sample variance of the outcome over the unit and its h nearest
        # same-arm units (ties kept): h + 1 units when there are no ties.
        # The neighbours must agree on `exact` as the matches do.
        xi2 = np.full(n, np.nan)
        for g in (1, 0):
            rows = arms[g]
            code = None if exact_code is None else exact_code[rows]
            sets = _neighbour_sets(
                Xw[rows], Xw[rows], h + 1, dtol, exact_rows=code, exact_pool=code
            )
            yg = Y[rows]
            for local, s in enumerate(sets):
                xi2[rows[local]] = float(np.var(yg[s], ddof=1)) if len(s) > 1 else 0.0
    else:
        # Homoskedastic: half the mean, over targets, of the average squared
        # discrepancy between a unit and each of its matches. With one
        # match per unit this is half the mean squared matched difference;
        # with several it is larger than squaring the averaged difference.
        sq = [
            float(np.mean(((2 * t[pos] - 1) * (Y[pos] - vals) - est) ** 2))
            for pos, vals in pair_vals.items()
        ]
        xi2 = np.full(n, 0.5 * float(np.mean(sq)))

    if estimand_key == "ATT":
        var = (
            float(np.sum((diff - est) ** 2)) + float(np.sum(xi2 * (K**2 - Kp)))
        ) / denom**2
    else:
        var = (
            float(np.sum((diff - est) ** 2))
            + float(np.sum(xi2 * (K**2 + 2.0 * K - Kp)))
        ) / denom**2
    se = float(np.sqrt(var)) if var >= 0 else float("nan")

    z = est / se if se > 0 else float("nan")
    pvalue = float(2 * stats.norm.sf(abs(z))) if np.isfinite(z) else float("nan")
    crit = stats.norm.ppf(1 - alpha / 2)
    ci = (est - crit * se, est + crit * se)

    # Matched-sample weights: a unit counts once for itself when it is a
    # matching target, plus K_m for the times it is used as a match.
    w_matched = K.copy()
    for g in targets:
        w_matched[arms[g]] += 1.0
    balance = _balance_table(X, t, w_matched, cov_cols)

    counts = n_set[np.concatenate([arms[g] for g in targets])]
    model_info: Dict[str, Any] = {
        "estimator": "nnmatch",
        "reference": "Stata teffects nnmatch",
        "metric": metric_key,
        "n_matches": m,
        "matches_min": int(counts.min()),
        "matches_max": int(counts.max()),
        "exact": exact_cols,
        "caliper": cal,
        "bias_adjust": bias_cols,
        "vce": vce_key,
        "vce_nn": h if vce_key == "robust" else None,
        "n_treated": n1,
        "n_control": n0,
        "n_matched_treated": float(w_matched[idx1].sum()),
        "n_matched_control": float(w_matched[idx0].sum()),
        "match_weights": pd.Series(K, index=clean.index, name="K_m"),
        "inference": "z",
    }
    return CausalResult(
        method="Nearest-neighbour matching (Abadie-Imbens)",
        estimand=estimand_key,
        estimate=est,
        se=se,
        pvalue=pvalue,
        ci=ci,
        alpha=alpha,
        n_obs=n,
        detail=balance,
        model_info=model_info,
        _citation_key="matching",
    )
