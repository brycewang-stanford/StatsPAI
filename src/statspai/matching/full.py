"""Optimal full matching (Rosenbaum 1991; Hansen 2004).

Every treated and every comparison unit is placed in a matched set that
holds either one treated unit and one or more comparison units, or one
comparison unit and one or more treated units, and the sum of the
within-set treated-comparison distances is as small as it can be. No unit
is discarded, which is the difference from pair matching.

Solved exactly. An optimal full matching is a minimum-cost edge cover of
the bipartite graph of treated and comparison units, and a minimum-cost
edge cover is obtained from a maximum-weight matching on the reduced gains
``min_i + min_j - c_ij`` plus each uncovered unit's cheapest edge. The
matching is one call to ``scipy.optimize.linear_sum_assignment``.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import optimize, stats

from .._aliases import accepts_aliases
from .._result_serialize import ResultProtocolMixin
from ..core._covariates import expands_categorical_covariates as _expands_categorical
from ..exceptions import DataInsufficient, MethodIncompatibility

__all__ = ["full_match", "FullMatchResult", "full_match_sets", "matched_set_effect"]


def full_match_sets(
    cost: np.ndarray, caliper: Optional[float] = None
) -> Tuple[np.ndarray, np.ndarray, float]:
    """Optimal full matching on a treated x comparison distance matrix.

    Parameters
    ----------
    cost : ndarray, shape (n_treated, n_control)
        Non-negative distances. ``inf`` forbids a pairing.
    caliper : float, optional
        Pairings with a distance above ``caliper`` are forbidden.

    Returns
    -------
    set_treated : ndarray of int, shape (n_treated,)
    set_control : ndarray of int, shape (n_control,)
        Matched-set label of each unit, ``-1`` for a unit with no
        permitted partner.
    total : float
        Sum of the distances of the matched pairs.
    """
    c = np.array(cost, dtype=float)
    if c.ndim != 2 or c.size == 0:
        raise MethodIncompatibility(
            "full_match_sets: cost must be a non-empty 2-D matrix.",
            recovery_hint="Pass distances with treated units in rows.",
        )
    if np.isnan(c).any() or (c[np.isfinite(c)] < 0).any():
        raise MethodIncompatibility(
            "full_match_sets: distances must be non-negative and not NaN.",
            recovery_hint="Use inf to forbid a pairing.",
        )
    if caliper is not None:
        c[c > float(caliper)] = np.inf
    n1, n0 = c.shape
    ok_t = np.isfinite(c).any(axis=1)
    ok_c = np.isfinite(c).any(axis=0)
    set_t = np.full(n1, -1, dtype=np.int64)
    set_c = np.full(n0, -1, dtype=np.int64)
    rows, cols = np.flatnonzero(ok_t), np.flatnonzero(ok_c)
    if rows.size == 0 or cols.size == 0:
        return set_t, set_c, 0.0
    sub = c[np.ix_(rows, cols)]
    finite = np.isfinite(sub)
    min_t = np.where(finite, sub, np.inf).min(axis=1)
    min_c = np.where(finite, sub, np.inf).min(axis=0)
    gain = min_t[:, None] + min_c[None, :] - np.where(finite, sub, np.inf)
    gain = np.where(finite & (gain > 0), gain, 0.0)
    ri, ci = optimize.linear_sum_assignment(gain, maximize=True)
    keep = gain[ri, ci] > 0
    edges = set(zip(ri[keep].tolist(), ci[keep].tolist()))
    covered_t = np.zeros(len(rows), dtype=bool)
    covered_c = np.zeros(len(cols), dtype=bool)
    covered_t[ri[keep]] = True
    covered_c[ci[keep]] = True
    arg_t = np.where(finite, sub, np.inf).argmin(axis=1)
    arg_c = np.where(finite, sub, np.inf).argmin(axis=0)
    for i in np.flatnonzero(~covered_t):
        edges.add((int(i), int(arg_t[i])))
    for j in np.flatnonzero(~covered_c):
        edges.add((int(arg_c[j]), int(j)))
    # With zero-distance ties the cover can hold an edge whose two ends are
    # both covered by other edges; dropping it keeps the cost and leaves
    # every component a star.
    deg_t = np.zeros(len(rows), dtype=np.int64)
    deg_c = np.zeros(len(cols), dtype=np.int64)
    for i, j in edges:
        deg_t[i] += 1
        deg_c[j] += 1
    for i, j in sorted(edges, key=lambda e: -sub[e[0], e[1]]):
        if deg_t[i] > 1 and deg_c[j] > 1:
            edges.discard((i, j))
            deg_t[i] -= 1
            deg_c[j] -= 1
    # Components of the cover are the matched sets.
    parent = np.arange(len(rows) + len(cols))

    def find(a: int) -> int:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = int(parent[a])
        return a

    total = 0.0
    for i, j in edges:
        total += float(sub[i, j])
        ra, rb = find(i), find(len(rows) + j)
        if ra != rb:
            parent[rb] = ra
    roots = np.array([find(k) for k in range(len(parent))])
    _, labels = np.unique(roots, return_inverse=True)
    set_t[rows] = labels[: len(rows)]
    set_c[cols] = labels[len(rows) :]
    return set_t, set_c, total


def matched_set_effect(
    y: np.ndarray,
    treat: np.ndarray,
    sets: np.ndarray,
    estimand: str = "ATT",
) -> Dict[str, Any]:
    """Effect and cluster-robust standard error from matched sets.

    The estimate is the weighted difference in means with the matching
    weights of the estimand: for the ATT a treated unit has weight one and
    a comparison unit ``n_treated_in_set / n_control_in_set``; for the ATE
    a unit has weight ``set size / own-arm count in the set``. It equals
    the average of the within-set differences in means, weighted by the
    number of treated units in the set (ATT) or by set size (ATE).

    The standard error is that of the weighted regression of the outcome
    on the treatment, clustered on matched set, with the small-sample
    factor ``G / (G - 1) * (N - 1) / (N - 2)``: ``sandwich::vcovCL`` on
    ``lm(y ~ treat, weights = w)``, which is what the ``MatchIt``
    documentation prescribes after full matching.

    Units with ``sets < 0`` are left out.
    """
    y = np.asarray(y, dtype=float)
    d = np.asarray(treat, dtype=float)
    s = np.asarray(sets)
    est = str(estimand).upper()
    if est not in ("ATT", "ATE", "ATC"):
        raise MethodIncompatibility(
            f"matched_set_effect: estimand must be ATT, ATC or ATE, got "
            f"{estimand!r}.",
            recovery_hint="Use estimand='ATT'.",
        )
    keep = s >= 0
    y, d, s = y[keep], d[keep], s[keep]
    _, code = np.unique(s, return_inverse=True)
    G = int(code.max()) + 1 if len(code) else 0
    n1s = np.bincount(code, weights=d, minlength=G)
    n0s = np.bincount(code, weights=1.0 - d, minlength=G)
    if G < 2 or (n1s == 0).any() or (n0s == 0).any():
        raise DataInsufficient(
            "matched_set_effect: every matched set needs a treated and a "
            "comparison unit, and there must be at least two sets.",
            recovery_hint="Check the matched-set labels.",
        )
    if est == "ATT":
        w = np.where(d == 1, 1.0, (n1s / n0s)[code])
    elif est == "ATC":
        w = np.where(d == 1, (n0s / n1s)[code], 1.0)
    else:
        ns = n1s + n0s
        w = np.where(d == 1, (ns / n1s)[code], (ns / n0s)[code])
    n = len(y)
    m1 = float(np.sum(w * d * y) / np.sum(w * d))
    m0 = float(np.sum(w * (1 - d) * y) / np.sum(w * (1 - d)))
    tau = m1 - m0
    X = np.column_stack([np.ones(n), d])
    resid = y - (m0 + tau * d)
    bread = np.linalg.inv(X.T @ (X * w[:, None]))
    score = X * (w * resid)[:, None]
    sums = np.zeros((G, 2))
    np.add.at(sums, code, score)
    meat = sums.T @ sums
    V = bread @ meat @ bread * (G / (G - 1.0)) * ((n - 1.0) / (n - 2.0))
    return {
        "estimate": tau,
        "se": float(np.sqrt(V[1, 1])),
        "weights": w,
        "mean_treated": m1,
        "mean_control": m0,
        "n_sets": G,
        "n": n,
        "kept": keep,
    }


@dataclass
class FullMatchResult(ResultProtocolMixin):
    """Result of :func:`full_match`.

    Attributes
    ----------
    estimate, se, ci, pvalue : float
        Effect on the outcome and its matched-set-clustered standard
        error; ``nan`` when no outcome was given.
    estimand : str
    subclass : pandas.Series
        Matched-set label of every row of the input (``<NA>`` for a row
        left out by the caliper or by missing values).
    weights : pandas.Series
        Matching weights for the estimand (0 for a row left out).
    balance : pandas.DataFrame
        Standardised mean differences before and after matching.
    total_distance : float
        Sum of the matched distances, the quantity that is minimised.
    n_sets, n_treated, n_control, n_unmatched : int

    Examples
    --------
    >>> import numpy as np, pandas as pd, statspai as sp
    >>> rng = np.random.default_rng(1)
    >>> n = 300
    >>> x = rng.normal(size=n)
    >>> d = rng.binomial(1, 1 / (1 + np.exp(-x)))
    >>> df = pd.DataFrame({"x": x, "d": d, "y": 1 + x + 2 * d + rng.normal(size=n)})
    >>> fit = sp.full_match(df, "y", "d", ["x"])
    >>> isinstance(fit, sp.FullMatchResult)
    True
    >>> bool(fit.n_treated + fit.n_control == n)
    True
    >>> sorted(fit.matched_data(df).columns)
    ['d', 'subclass', 'weights', 'x', 'y']
    """

    _citation_keys = ("rosenbaum1991characterization", "hansen2004full")

    estimate: float
    se: float
    ci: Tuple[float, float]
    pvalue: float
    estimand: str
    subclass: pd.Series
    weights: pd.Series
    balance: pd.DataFrame
    total_distance: float
    n_sets: int
    n_treated: int
    n_control: int
    n_unmatched: int
    distance: str
    model_info: Dict[str, Any] = field(default_factory=dict)

    @property
    def att(self) -> float:
        return self.estimate

    def matched_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """``data`` with ``subclass`` and ``weights`` columns, matched rows
        only."""
        out = data.loc[self.subclass.index].copy()
        out["subclass"] = self.subclass
        out["weights"] = self.weights
        return out[out["weights"] > 0]

    def summary(self) -> str:
        sizes = self.model_info.get("set_sizes", {})
        lines = [
            "Optimal full matching",
            "-" * 44,
            f"Distance          : {self.distance}",
            f"Estimand          : {self.estimand}",
            f"Treated / control : {self.n_treated} / {self.n_control}",
            f"Matched sets      : {self.n_sets}",
            f"Unmatched units   : {self.n_unmatched}",
            f"Total distance    : {self.total_distance:.6g}",
            f"Largest set       : {sizes.get('max', '')}",
        ]
        if np.isfinite(self.estimate):
            lines.append(
                f"{self.estimand:<18}: {self.estimate:.4f}  (SE = {self.se:.4f}, "
                "clustered on matched set)"
            )
        lines.append("")
        lines.append(self.balance.round(4).to_string())
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()


def _smd(X: np.ndarray, d: np.ndarray, w: np.ndarray, denom: np.ndarray) -> np.ndarray:
    t, c = d == 1, d == 0
    m1 = np.average(X[t], axis=0, weights=w[t])
    m0 = np.average(X[c], axis=0, weights=w[c])
    return np.asarray((m1 - m0) / denom, dtype=float)


@accepts_aliases(treatment="treat", outcome="y")
@_expands_categorical("covariates")
def full_match(
    data: pd.DataFrame,
    y: Optional[str] = None,
    treat: Optional[str] = None,
    covariates: Optional[Sequence[str]] = None,
    *,
    distance: str = "propensity",
    pscore: Optional[str] = None,
    estimand: str = "ATT",
    caliper: Optional[float] = None,
    caliper_scale: str = "sd",
    ps_model: str = "logit",
    alpha: float = 0.05,
) -> FullMatchResult:
    """Optimal full matching.

    Places every treated and every comparison unit in a matched set with
    one treated and several comparison units, or one comparison and
    several treated units, so that the total within-set distance is
    minimal. Unlike pair matching it uses the whole sample, and unlike
    subclassification the sets are chosen by the data.

    Parameters
    ----------
    data : pandas.DataFrame
    y : str, optional
        Outcome. Without it only the matching (sets, weights, balance) is
        returned.
    treat : str
        0/1 treatment indicator.
    covariates : sequence of str
        Variables to match on, and the variables of the balance table.
    distance : {'propensity', 'logit', 'mahalanobis', 'euclidean'}
        ``'propensity'``: absolute difference of the fitted propensity
        score (``MatchIt``'s default, ``distance = "glm"``). ``'logit'``:
        difference on the linear-predictor scale. ``'mahalanobis'`` uses
        the pooled within-group covariance, as ``MatchIt`` does.
    pscore : str, optional
        Column holding a score computed elsewhere (a boosted or forest
        propensity score, say). Matching is on the absolute difference of
        this column and ``covariates`` are used for the balance table only.
    estimand : {'ATT', 'ATC', 'ATE'}, default 'ATT'
        Sets the matching weights; the matched sets are the same.
    caliper : float, optional
        Largest permitted distance. A unit with no partner inside the
        caliper is left unmatched and the estimand then refers to the
        matched units.
    caliper_scale : {'sd', 'raw'}, default 'sd'
        ``'sd'`` reads ``caliper`` in standard deviations of the score
        (score distances only), ``'raw'`` in the units of the distance.
    ps_model : {'logit', 'probit'}, default 'logit'
    alpha : float, default 0.05

    Returns
    -------
    FullMatchResult

    Notes
    -----
    The matching is exact: the reported ``total_distance`` is the minimum
    over all full matchings. ``optmatch::fullmatch`` (and ``MatchIt`` with
    ``method = "full"``, which calls it) solves the same problem on
    distances rounded to a tolerance, so its total can be slightly larger
    and the matched sets, hence the estimate, can differ a little.

    Matched sets can be very unequal in size when overlap is poor; a few
    comparison units then carry most of the weight. Inspect
    ``model_info['set_sizes']`` and the effective sample size
    ``model_info['ess_control']``.

    Examples
    --------
    >>> import numpy as np, pandas as pd, statspai as sp
    >>> rng = np.random.default_rng(1)
    >>> n = 300
    >>> x = rng.normal(size=n)
    >>> d = rng.binomial(1, 1 / (1 + np.exp(-x)))
    >>> df = pd.DataFrame({"x": x, "d": d, "y": 1 + x + 2 * d + rng.normal(size=n)})
    >>> fit = sp.full_match(df, "y", "d", ["x"])
    >>> int(fit.n_unmatched)
    0
    >>> bool(abs(fit.balance.loc["x", "smd_matched"]) < 0.1)
    True

    References
    ----------
    [@rosenbaum1991characterization]
    [@hansen2004full]
    """
    if treat is None or covariates is None:
        raise MethodIncompatibility(
            "full_match: treat= and covariates= are required.",
            recovery_hint="Call sp.full_match(df, y, treat, covariates).",
        )
    covariates = [covariates] if isinstance(covariates, str) else list(covariates)
    est = str(estimand).upper()
    if est not in ("ATT", "ATC", "ATE"):
        raise MethodIncompatibility(
            f"full_match: estimand must be ATT, ATC or ATE, got {estimand!r}.",
            recovery_hint="Use estimand='ATT'.",
        )
    dist = str(distance).lower()
    if dist not in ("propensity", "logit", "mahalanobis", "euclidean"):
        raise MethodIncompatibility(
            f"full_match: unknown distance {distance!r}.",
            recovery_hint="Use 'propensity', 'logit', 'mahalanobis' or " "'euclidean'.",
        )
    if caliper_scale not in ("sd", "raw"):
        raise MethodIncompatibility(
            "full_match: caliper_scale must be 'sd' or 'raw'.",
            recovery_hint="Use caliper_scale='sd'.",
        )
    cols = [c for c in (y, treat, pscore) if c is not None] + covariates
    missing = [c for c in cols if c not in data.columns]
    if missing:
        raise MethodIncompatibility(
            f"full_match: column(s) not in data: {missing}.",
            diagnostics={"missing_columns": missing},
        )
    df = data[list(dict.fromkeys(cols))].dropna()
    d = pd.to_numeric(df[treat], errors="coerce").to_numpy(dtype=float)
    if not np.isin(np.unique(d), (0.0, 1.0)).all():
        raise MethodIncompatibility(
            "full_match: treat must be coded 0/1.",
            recovery_hint="Recode the treatment to a 0/1 indicator.",
        )
    X = df[covariates].to_numpy(dtype=float)
    t_idx, c_idx = np.flatnonzero(d == 1), np.flatnonzero(d == 0)
    if len(t_idx) == 0 or len(c_idx) == 0:
        raise DataInsufficient(
            "full_match: need both treated and comparison units.",
            recovery_hint="Check the treatment column.",
        )

    score: Optional[np.ndarray] = None
    if pscore is not None:
        score = df[pscore].to_numpy(dtype=float)
        label = f"|{pscore}| (supplied score)"
    elif dist in ("propensity", "logit"):
        from ._binary_fit import fit_binary_index

        if ps_model not in ("logit", "probit"):
            raise MethodIncompatibility(
                f"full_match: ps_model must be 'logit' or 'probit', got "
                f"{ps_model!r}.",
                recovery_hint="Use ps_model='logit'.",
            )
        beta = np.asarray(fit_binary_index(X, d, ps_model)["beta"], dtype=float)
        index = beta[0] + X @ beta[1:]
        if dist == "logit":
            score = index
        elif ps_model == "logit":
            score = 1.0 / (1.0 + np.exp(-index))
        else:
            score = stats.norm.cdf(index)
        label = f"{ps_model} propensity score" + (
            " (linear predictor)" if dist == "logit" else ""
        )
    if score is not None:
        cost = np.abs(score[t_idx][:, None] - score[c_idx][None, :])
    else:
        from .optimal import _distance_matrix

        cost = _distance_matrix(X[t_idx], X[c_idx], metric=dist)
        label = dist
    cal = None
    if caliper is not None:
        cal = float(caliper)
        if caliper_scale == "sd":
            if score is None:
                raise MethodIncompatibility(
                    "full_match: caliper_scale='sd' needs a score distance.",
                    recovery_hint="Use caliper_scale='raw' with a "
                    "covariate distance.",
                )
            cal *= float(np.std(score, ddof=1))
    set_t, set_c, total = full_match_sets(cost, caliper=cal)
    sets = np.full(len(d), -1, dtype=np.int64)
    sets[t_idx], sets[c_idx] = set_t, set_c
    matched = sets >= 0
    n_un = int((~matched).sum())
    if not matched.any():
        raise DataInsufficient(
            "full_match: the caliper leaves no unit with a partner.",
            recovery_hint="Use a wider caliper.",
        )
    if n_un:
        warnings.warn(
            f"full_match: {n_un} unit(s) have no partner within the caliper "
            "and are left out; the estimand refers to the matched units.",
            RuntimeWarning,
            stacklevel=2,
        )
    y_arr = (
        df[y].to_numpy(dtype=float) if y is not None else np.zeros(len(d), dtype=float)
    )
    eff = matched_set_effect(y_arr, d, sets, est)
    w = np.zeros(len(d))
    w[matched] = eff["weights"]

    # Balance: the denominator is the standard deviation in the group the
    # estimand refers to (pooled for the ATE), computed before matching and
    # used for both columns.
    sd1 = X[d == 1].std(axis=0, ddof=1)
    sd0 = X[d == 0].std(axis=0, ddof=1)
    denom = {"ATT": sd1, "ATC": sd0, "ATE": np.sqrt((sd1**2 + sd0**2) / 2.0)}[est]
    denom = np.where(denom > 0, denom, np.nan)
    ones = np.ones(len(d))
    balance = pd.DataFrame(
        {
            "smd_unmatched": _smd(X, d, ones, denom),
            "smd_matched": _smd(X[matched], d[matched], w[matched], denom),
        },
        index=list(covariates),
    )

    sizes = np.bincount(sets[matched])
    sizes = sizes[sizes > 0]
    wc = w[(d == 0) & matched]
    wt = w[(d == 1) & matched]
    z = float(stats.norm.ppf(1 - alpha / 2))
    if y is None:
        estimate = se = pval = float("nan")
        ci = (float("nan"), float("nan"))
    else:
        estimate, se = float(eff["estimate"]), float(eff["se"])
        pval = float(2 * stats.norm.sf(abs(estimate / se))) if se > 0 else float("nan")
        ci = (estimate - z * se, estimate + z * se)
    subclass = pd.Series(pd.array(sets, dtype="Int64"), index=df.index, name="subclass")
    subclass[~matched] = pd.NA
    info: Dict[str, Any] = {
        "estimator": "Optimal full matching",
        "se_method": "weighted regression, clustered on matched set",
        "set_sizes": {
            "min": int(sizes.min()),
            "median": float(np.median(sizes)),
            "max": int(sizes.max()),
        },
        "ess_treated": float(wt.sum() ** 2 / np.sum(wt**2)),
        "ess_control": float(wc.sum() ** 2 / np.sum(wc**2)),
        "caliper": cal,
        "n_obs": int(matched.sum()),
        "covariates": list(covariates),
    }
    return FullMatchResult(
        estimate=estimate,
        se=se,
        ci=ci,
        pvalue=pval,
        estimand=est,
        subclass=subclass,
        weights=pd.Series(w, index=df.index, name="weights"),
        balance=balance,
        total_distance=float(total),
        n_sets=int(len(sizes)),
        n_treated=int((d[matched] == 1).sum()),
        n_control=int((d[matched] == 0).sum()),
        n_unmatched=n_un,
        distance=label,
        model_info=info,
    )
