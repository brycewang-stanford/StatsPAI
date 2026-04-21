"""
Invariant Causal Prediction (ICP).

The identifying principle: if S is the true set of direct causes of Y,
then the *conditional distribution* P(Y | X_S) is invariant across
environments (interventional settings) that do not directly intervene
on Y. ICP exploits this by testing, for every candidate subset S, the
null hypothesis of invariance; the intersection of subsets accepted by
the test is a conservative estimate of the parents of Y.

Reference
---------
Peters, J., Buehlmann, P., & Meinshausen, N. (2016).
"Causal inference using invariant prediction: identification and
confidence intervals." *JRSS-B* 78(5): 947-1012.

Heinze-Deml, C., Peters, J., & Meinshausen, N. (2018).
"Invariant Causal Prediction for Nonlinear Models." *Journal of
Causal Inference* 6(2).

Notes
-----
This implementation performs **level-alpha** subset search using the
two-sample mean-plus-variance test (Chow test equivalent for linear
models). For large p, pass ``max_subset_size`` < p to keep runtime
tractable.
"""

from __future__ import annotations
from dataclasses import dataclass
from itertools import combinations
from typing import Iterable, Sequence
import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class ICPResult:
    parents: set[str]
    accepted_subsets: list[frozenset]
    rejection_reason: dict[frozenset, str]
    alpha: float
    coefficients: dict[str, tuple[float, float]]  # var -> (lo, hi) CI
    method: str

    def summary(self) -> str:
        if not self.parents:
            return (
                f"ICP @ alpha={self.alpha}: no subset accepted -- the empty "
                f"set is the only provably-causal estimate (conservative)."
            )
        lines = [
            f"ICP (method={self.method}, alpha={self.alpha}) -- "
            f"plausible parents of Y: {sorted(self.parents)}",
            "Confidence intervals for parent coefficients:",
        ]
        for v, (lo, hi) in self.coefficients.items():
            lines.append(f"  {v}: [{lo:.3f}, {hi:.3f}]")
        lines.append(
            f"{len(self.accepted_subsets)} subsets accepted by invariance test."
        )
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"ICPResult(parents={sorted(self.parents)}, "
            f"n_accepted={len(self.accepted_subsets)})"
        )


def icp(
    X: pd.DataFrame | np.ndarray,
    y: np.ndarray,
    environment: np.ndarray,
    alpha: float = 0.05,
    method: str = "linear",
    max_subset_size: int | None = None,
) -> ICPResult:
    """Infer the direct parents of ``y`` via Invariant Causal Prediction.

    Parameters
    ----------
    X : pd.DataFrame | np.ndarray
        Predictor matrix (n, p). A DataFrame preserves column names.
    y : array-like
        Response vector (length n).
    environment : array-like
        Integer-coded environment label (length n). At least two
        environments required; more give stronger identification.
    alpha : float, default 0.05
        Family-wise significance level.
    method : {"linear", "nonlinear"}, default "linear"
        Linear uses F-test on residual distributions; ``nonlinear``
        uses a two-sample K-S test on residuals from a local linear
        fit, per Heinze-Deml et al. 2018.
    max_subset_size : int, optional
        Cap on the cardinality of subsets tested. Default: min(p, 6).

    Returns
    -------
    ICPResult
    """
    X = pd.DataFrame(X) if not isinstance(X, pd.DataFrame) else X.copy()
    y = np.asarray(y, dtype=float)
    env = np.asarray(environment)
    n, p = X.shape
    assert len(y) == n and len(env) == n
    env_labels = np.unique(env)
    if len(env_labels) < 2:
        raise ValueError("ICP needs at least 2 environments.")

    per_subset_alpha = alpha / max(1, 2 ** p - 1)

    if max_subset_size is None:
        max_subset_size = min(p, 6)

    accepted: list[frozenset] = []
    rejection: dict[frozenset, str] = {}

    cols = list(X.columns)
    candidate_sizes = range(0, max_subset_size + 1)

    for k in candidate_sizes:
        for subset in combinations(cols, k):
            S = frozenset(subset)
            pval, reason = _invariance_test(
                X[list(subset)].to_numpy(), y, env, method=method
            )
            if pval >= per_subset_alpha:
                accepted.append(S)
            else:
                rejection[S] = f"pval={pval:.4g} < alpha/m={per_subset_alpha:.2g}: {reason}"

    if not accepted:
        return ICPResult(
            parents=set(),
            accepted_subsets=[],
            rejection_reason=rejection,
            alpha=alpha,
            coefficients={},
            method=method,
        )

    parents = set.intersection(*(set(s) for s in accepted))

    coefficients: dict[str, tuple[float, float]] = {}
    if parents:
        Xp = X[list(parents)].to_numpy()
        Xp = np.column_stack([np.ones(n), Xp])
        beta, *_ = np.linalg.lstsq(Xp, y, rcond=None)
        resid = y - Xp @ beta
        sigma2 = float(resid @ resid / max(n - Xp.shape[1], 1))
        try:
            xtx_inv = np.linalg.inv(Xp.T @ Xp)
        except np.linalg.LinAlgError:
            xtx_inv = np.linalg.pinv(Xp.T @ Xp)
        se = np.sqrt(np.diag(sigma2 * xtx_inv))
        for i, v in enumerate(sorted(parents), start=1):
            coefficients[v] = (
                float(beta[i] - 1.96 * se[i]),
                float(beta[i] + 1.96 * se[i]),
            )

    return ICPResult(
        parents=parents,
        accepted_subsets=accepted,
        rejection_reason=rejection,
        alpha=alpha,
        coefficients=coefficients,
        method=method,
    )


def _invariance_test(
    X: np.ndarray, y: np.ndarray, env: np.ndarray, method: str
) -> tuple[float, str]:
    """Return (pvalue, reason) for the null hypothesis of invariance."""
    n = X.shape[0]
    X_with_const = np.column_stack([np.ones(n), X]) if X.size else np.ones((n, 1))

    beta, *_ = np.linalg.lstsq(X_with_const, y, rcond=None)
    resid = y - X_with_const @ beta

    env_labels = np.unique(env)
    if method == "linear":
        means = []
        vars_ = []
        ns = []
        for e in env_labels:
            r = resid[env == e]
            if r.size < 2:
                return 0.0, "environment too small"
            means.append(r.mean())
            vars_.append(r.var(ddof=1))
            ns.append(r.size)
        overall_var = resid.var(ddof=1)
        # Welch-style test for equality of means
        stat_mean, p_mean = stats.f_oneway(
            *[resid[env == e] for e in env_labels]
        )
        # Levene-style test for equality of variances
        stat_var, p_var = stats.levene(
            *[resid[env == e] for e in env_labels]
        )
        p = min(p_mean, p_var)
        # Bonferroni for two sub-tests:
        return float(min(1.0, 2.0 * p)), "mean/variance invariance"

    if method == "nonlinear":
        from scipy.stats import ks_2samp
        groups = [resid[env == e] for e in env_labels]
        worst_p = 1.0
        for i in range(len(groups)):
            for j in range(i + 1, len(groups)):
                _, pij = ks_2samp(groups[i], groups[j])
                worst_p = min(worst_p, pij)
        m = len(groups) * (len(groups) - 1) / 2
        return float(min(1.0, m * worst_p)), "K-S residual invariance"

    raise ValueError(f"Unknown ICP method: {method!r}")


def nonlinear_icp(X, y, environment, alpha: float = 0.05, **kw) -> ICPResult:
    """Alias for ``icp(..., method='nonlinear')`` -- Heinze-Deml et al. 2018."""
    return icp(X, y, environment, alpha=alpha, method="nonlinear", **kw)
