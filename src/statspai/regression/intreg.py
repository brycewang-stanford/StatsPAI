"""Interval regression (Stata ``intreg``).

The outcome ``y* = x'b + e``, ``e ~ N(0, sigma^2)``, is known only to lie in
``[lo, hi]``: a point when ``lo == hi``, left-censored when ``lo`` is
missing, right-censored when ``hi`` is missing, an interval otherwise.
Rows with both bounds missing carry no information and are dropped, as in
Stata. The log-likelihood contributions are

* point: ``ln phi((y - x'b) / s) - ln s``;
* left-censored: ``ln Phi((hi - x'b) / s)``;
* right-censored: ``ln Phi((x'b - lo) / s)``;
* interval: ``ln [Phi((hi - x'b) / s) - Phi((lo - x'b) / s)]``,

maximised over ``(b, ln s)`` -- Stata's ``/lnsigma`` -- by BFGS from OLS on
the interval midpoints, then Newton on complex-step scores.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import optimize, special, stats

from .._aliases import accepts_aliases
from ..core._call_record import records_call
from ..core._vcov import ml_vcov
from ..core.results import CausalResult
from ..exceptions import ColumnNotFound, DataInsufficient, MethodIncompatibility
from ._limited_dep_result import LimitedDepResult
from ._optim_helpers import (
    complex_step_scores,
    inverse_information,
    ml_newton_polish,
    require_two_clusters,
    robust_convergence,
    se_from_vcov,
)

__all__ = ["intreg"]


def _log_prob_between(a: Any, b: Any) -> Any:
    """``ln(Phi(b) - Phi(a))`` for ``a < b``, accurate in both tails and
    complex-step safe (the branch is chosen on the real part)."""
    upper = np.real(a) > 0
    # lower tail: Phi(b) [1 - Phi(a)/Phi(b)]; upper tail: by symmetry
    lo_side = special.log_ndtr(b) + np.log1p(
        -np.exp(special.log_ndtr(a) - special.log_ndtr(b))
    )
    up_side = special.log_ndtr(-a) + np.log1p(
        -np.exp(special.log_ndtr(-b) - special.log_ndtr(-a))
    )
    return np.where(upper, up_side, lo_side)


def _rhs_columns(formula: str, data: pd.DataFrame) -> Any:
    """Regressor columns of ``'lo + hi ~ rhs'`` without dropping rows on the
    (partly missing) bounds."""
    from ..core.utils import formula_to_columns

    lhs, rhs = formula.split("~", 1)
    bounds = [t.strip() for t in lhs.split("+")]
    if len(bounds) != 2 or not all(bounds):
        raise MethodIncompatibility(
            f"intreg: the left-hand side of {formula!r} must name the lower "
            "and the upper bound, 'lo + hi ~ x1 + x2'.",
        )
    one = "_intreg_one"
    while one in data.columns:
        one = "_" + one
    frame, _, x = formula_to_columns(f"{one} ~ {rhs}", data.assign(**{one: 1.0}))
    return frame.drop(columns=[one]), bounds[0], bounds[1], x


@records_call("intreg")
@accepts_aliases(robust="vce")
def intreg(
    data: pd.DataFrame,
    y_lower: Optional[str] = None,
    y_upper: Optional[str] = None,
    x: Optional[List[str]] = None,
    vce: Optional[str] = None,
    cluster: Optional[str] = None,
    weights: Optional[str] = None,
    alpha: float = 0.05,
    *,
    formula: Optional[str] = None,
) -> CausalResult:
    """
    Interval regression: a linear model for an outcome observed as an
    interval, by maximum likelihood.

    Equivalent to Stata's ``intreg lo hi x``. Each row's outcome lies in
    ``[lo, hi]``: equal bounds are an exact value, a missing lower bound
    is left-censoring at ``hi``, a missing upper bound is right-censoring
    at ``lo``. Tobit is the special case with one censoring point.

    Parameters
    ----------
    data : pd.DataFrame
    y_lower, y_upper : str
        Columns with the lower and upper bounds (``NaN`` = unbounded).
    x : list of str
        Regressors (a constant is added).
    vce : {None, 'oim', 'robust', 'cluster'}, optional
        Observed information (default), sandwich, or cluster sandwich
        (``vce='cluster firm'`` or ``cluster='firm'``).
    cluster : str, optional
    weights : str, optional
        Sampling weights (``[pw=]``), which imply robust standard errors.
    alpha : float, default 0.05
    formula : str, optional
        ``'lo + hi ~ x1 + C(g)'`` instead of ``y_lower`` / ``y_upper`` /
        ``x``; factor and transformed terms become columns.

    Returns
    -------
    CausalResult
        ``params`` holds the coefficients and ``sigma``; ``model_info``
        counts the uncensored, left-, right- and interval-censored rows and
        gives ``lnsigma``, the log-likelihood and the Wald test of the
        slopes. ``sp.margins`` reports Stata's ``predict`` statistics
        (``xb`` by default, ``e(a,b)``, ``ystar(a,b)``, ``pr(a,b)``).

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> df = pd.DataFrame({"x": rng.normal(size=500)})
    >>> y = 1 + 2 * df["x"] + rng.normal(size=500)
    >>> df["lo"] = np.floor(y)          # income reported in bands of 1
    >>> df["hi"] = df["lo"] + 1
    >>> df.loc[y < -1, "lo"] = np.nan   # bottom band open
    >>> fit = sp.intreg(df, y_lower="lo", y_upper="hi", x=["x"])
    >>> round(float(fit.params["x"]), 1)  # doctest: +SKIP
    2.0
    """
    from ..core._vcov_spec import parse_se_request

    se_req = parse_se_request(
        vce, cluster, function="intreg", supported=("nonrobust", "robust", "cluster")
    )
    se_kind, cluster = se_req.kind, se_req.cluster
    if weights is not None and se_kind == "nonrobust":
        se_kind = "robust"  # Stata: pweights imply vce(robust)
    formula_text = formula
    if formula is not None:
        if y_lower is not None or y_upper is not None or x is not None:
            raise MethodIncompatibility(
                "intreg: pass formula= or y_lower= / y_upper= / x=, not both.",
            )
        data, y_lower, y_upper, x = _rhs_columns(formula, data)
    if y_lower is None or y_upper is None or x is None:
        raise MethodIncompatibility(
            "intreg: the two bounds and the regressors are needed.",
            recovery_hint="Pass y_lower=, y_upper= and x=, or "
            "formula='lo + hi ~ x1 + x2'.",
        )
    x = [x] if isinstance(x, str) else list(x)
    needed = [y_lower, y_upper] + x + [c for c in (cluster, weights) if c]
    missing = [c for c in needed if c not in data.columns]
    if missing:
        raise ColumnNotFound.from_missing(missing, list(data.columns), context="intreg")
    df = data.loc[data[x + [c for c in (cluster, weights) if c]].notna().all(axis=1)]
    lo = df[y_lower].to_numpy(dtype=float)
    hi = df[y_upper].to_numpy(dtype=float)
    keep = ~(np.isnan(lo) & np.isnan(hi))
    df, lo, hi = df.loc[keep], lo[keep], hi[keep]
    bad = ~np.isnan(lo) & ~np.isnan(hi) & (lo > hi)
    if bad.any():
        raise MethodIncompatibility(
            f"intreg: {int(bad.sum())} row(s) have {y_lower} > {y_upper}.",
            recovery_hint="Swap or fix the bounds; Stata drops these rows "
            "with a note, which hides a data error.",
            diagnostics={"n_inverted": int(bad.sum())},
        )
    from ..core._collinear import drop_collinear_names

    x, omitted = drop_collinear_names(df, x, "intreg")
    n = len(df)
    X = np.column_stack([np.ones(n)] + [df[v].to_numpy(dtype=float) for v in x])
    k = X.shape[1]
    point = lo == hi
    left = np.isnan(lo) & ~np.isnan(hi)
    right = ~np.isnan(lo) & np.isnan(hi)
    inter = ~point & ~left & ~right
    if int(point.sum() + inter.sum()) == 0 and n < k + 1:
        raise DataInsufficient("intreg: too few observations.")
    wt = np.ones(n)
    if weights is not None:
        wt = df[weights].to_numpy(dtype=float)
        if not np.all(np.isfinite(wt)) or np.any(wt <= 0):
            raise MethodIncompatibility(
                "intreg: weights must be finite and strictly positive.",
                diagnostics={"weights": weights},
            )
        wt = wt * (n / wt.sum())
    lo0 = np.where(np.isnan(lo), 0.0, lo)
    hi0 = np.where(np.isnan(hi), 0.0, hi)
    half_log_2pi = 0.5 * np.log(2 * np.pi)

    def obs_loglik(theta: np.ndarray) -> np.ndarray:
        xb = X @ theta[:k]
        ln_s = theta[k]
        s = np.exp(ln_s)
        out = np.zeros(n, dtype=np.result_type(theta, float))
        if point.any():
            r = (lo0[point] - xb[point]) / s
            out[point] = -half_log_2pi - ln_s - 0.5 * r * r
        if left.any():
            out[left] = special.log_ndtr((hi0[left] - xb[left]) / s)
        if right.any():
            out[right] = special.log_ndtr((xb[right] - lo0[right]) / s)
        if inter.any():
            out[inter] = _log_prob_between(
                (lo0[inter] - xb[inter]) / s, (hi0[inter] - xb[inter]) / s
            )
        return wt * out

    def neg_total(theta: np.ndarray) -> float:
        with np.errstate(all="ignore"):
            v = -float(np.sum(obs_loglik(theta)))
        return v if np.isfinite(v) else 1e300

    def neg_grad(theta: np.ndarray) -> np.ndarray:
        grad: np.ndarray = -complex_step_scores(obs_loglik, theta).sum(axis=0)
        return grad

    mid = np.where(point | inter, 0.5 * (lo0 + hi0), np.where(left, hi0, lo0))
    b0 = np.linalg.lstsq(X, mid, rcond=None)[0]
    s0 = max(float(np.std(mid - X @ b0)), 1e-3)
    opt = optimize.minimize(
        neg_total,
        np.append(b0, np.log(s0)),
        jac=neg_grad,
        method="BFGS",
        options={"maxiter": 2000},
    )
    converged, _ = robust_convergence(opt)
    theta, scores, H, _ = ml_newton_polish(obs_loglik, opt.x)
    grad_norm = float(np.max(np.abs(scores.sum(axis=0))))
    converged = bool(converged or grad_norm < 1e-6)
    clusters = df[cluster].to_numpy() if se_kind == "cluster" else None
    if clusters is not None:
        require_two_clusters(clusters, "intreg")
    V = ml_vcov(
        inverse_information(H),
        scores if se_kind != "nonrobust" else None,
        kind=se_kind,
        clusters=clusters,
    )
    se = se_from_vcov(V)
    beta, sigma = theta[:k], float(np.exp(theta[k]))
    names = ["const"] + x
    with np.errstate(divide="ignore", invalid="ignore"):
        z = beta / se[:k]
    pv = 2 * stats.norm.sf(np.abs(z))
    detail = pd.DataFrame(
        {
            "variable": names + ["sigma"],
            "coefficient": np.append(beta, sigma),
            "se": np.append(se[:k], se[k] * sigma),
            "z": np.append(z, np.nan),
            "pvalue": np.append(pv, np.nan),
        }
    )
    slopes = slice(1, k)
    wald = (
        float(beta[slopes] @ np.linalg.solve(V[slopes, slopes], beta[slopes]))
        if k > 1
        else np.nan
    )
    model_info: Dict[str, Any] = {
        "omitted": omitted,
        "method": "Interval regression (ML)",
        "sigma": sigma,
        "lnsigma": float(theta[k]),
        "lnsigma_se": float(se[k]),
        "n_uncensored": int(point.sum()),
        "n_left_censored": int(left.sum()),
        "n_right_censored": int(right.sum()),
        "n_interval": int(inter.sum()),
        "log_likelihood": float(np.sum(obs_loglik(theta))),
        "wald_chi2": wald,
        "wald_df": k - 1,
        "wald_pvalue": float(stats.chi2.sf(wald, k - 1)) if k > 1 else np.nan,
        "converged": converged,
        "gradient_norm": grad_norm,
        "vce": se_kind,
        "cluster": cluster if se_kind == "cluster" else None,
        "n_clusters": (
            int(pd.Series(clusters).nunique()) if clusters is not None else None
        ),
        "weights": weights,
    }
    z_crit = stats.norm.ppf(1 - alpha / 2)
    j = 1 if k > 1 else 0
    est, s_e = float(beta[j]), float(se[j])
    res = LimitedDepResult(
        method="Interval regression",
        estimand=f"beta_{names[j]}",
        estimate=est,
        se=s_e,
        pvalue=float(pv[j]),
        ci=(est - z_crit * s_e, est + z_crit * s_e),
        alpha=alpha,
        n_obs=n,
        detail=detail,
        model_info=model_info,
        _citation_key="intreg",
    )
    res._ml_state = {
        "model": "intreg",
        "theta": np.asarray(theta, dtype=float),
        "vcov": np.asarray(V, dtype=float),
        "terms": names,
        "ll": None,
        "ul": None,
        "bounds": [y_lower, y_upper],
        "formula": formula_text,
        "nobs": n,
    }
    return res
