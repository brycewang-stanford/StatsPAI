"""Einav-Finkelstein-Cullen welfare with non-linear demand curves.

The Online Appendix of Einav, Finkelstein and Cullen (2010), Table A2,
re-estimates the welfare cost of adverse selection with a probit demand
curve (row 2), linear demand constrained to pass through ``(Q, P) =
(1, $0)`` or ``(0, $800)`` (rows 3 and 4) and a quadratic demand curve
constrained through both points (row 5); average cost stays linear in
price in every row. This module computes those specifications:
``demand='probit'`` (``D(p) = Phi(alpha + beta p)``), ``demand='quadratic'``
(``D(p) = alpha + beta p + beta2 p^2``) and ``demand_through=[(q, p), ...]``,
restricted least squares that forces the linear or quadratic curve through
the given points.

With ``AC(p) = gamma + delta p`` the competitive equilibrium price solves
``p = AC(p)`` whatever the demand curve, so only the coverage at it moves.
The surplus of covering the share ``q`` with the highest willingness to pay
is ``S(q) = integral_0^q P(s) ds - q AC(P(q))`` with ``P`` the inverse
demand; the efficient allocation maximises it, the deadweight loss is
``S(q_eff) - S(q_eq)``. The integral is evaluated as
``q P(q) + integral_{P(q)}^{choke} D(p) dp`` in closed form for every
demand family.

References
----------
[@einav2010estimating]
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import optimize, stats

from ..exceptions import (
    AssumptionWarning,
    DataInsufficient,
    IdentificationFailure,
    MethodIncompatibility,
    NumericalInstability,
)
from ._efc_core import (
    _EFC_ROWS,
    _efc_bootstrap,
    _numeric_jacobian,
    _ols_influence,
    _stacked_vcov,
)
from ._results import EFCWelfareResult

__all__ = ["efc_nonlinear", "efc_quantities", "inverse_demand"]

_GRID = 4001


def rows_for(demand: str) -> Tuple[str, ...]:
    """Row labels of the result table for a demand family."""
    curve = ("alpha", "beta", "beta2") if demand == "quadratic" else ("alpha", "beta")
    return curve + _EFC_ROWS[2:]


class _Demand:
    """A downward-sloping demand curve with its inverse and upper area."""

    def __init__(self, demand: str, coef: Sequence[float]) -> None:
        self.kind = demand
        self.a = float(coef[0])
        self.b = float(coef[1])
        self.b2 = float(coef[2]) if demand == "quadratic" else 0.0
        if demand == "quadratic" and self.b2 != 0.0:
            # Decreasing branch: D'(p) = b + 2 b2 p < 0.
            self.peak = -self.b / (2.0 * self.b2)
            if self.b2 > 0 and self.peak <= 0:
                raise IdentificationFailure(
                    "the quadratic demand curve slopes up at every positive price",
                    recovery_hint="Constrain it with demand_through= or use "
                    "demand='linear' / 'probit'.",
                )
        elif self.b >= 0:
            raise IdentificationFailure(
                f"estimated demand does not slope down (beta = {self.b:.4g})",
                recovery_hint="Check that the price variation is exogenous given "
                "the covariates, and that price is coded as the price of coverage.",
            )
        if demand == "probit":
            self.q_lo, self.q_hi = 0.0, 1.0
            self.choke = np.inf
        elif demand == "linear" or self.b2 == 0.0:
            self.q_lo, self.q_hi = 0.0, 1.0
            self.choke = -self.a / self.b
        else:
            self.choke = self.inverse(0.0)
            if not np.isfinite(self.choke):
                raise IdentificationFailure(
                    "the quadratic demand curve never reaches zero coverage",
                    recovery_hint="Constrain it with demand_through= (e.g. a choke "
                    "price) or use demand='linear' / 'probit'.",
                )
            q_top = self.D(self.peak) if self.b2 < 0 else np.inf
            self.q_lo, self.q_hi = 0.0, float(min(1.0, q_top))

    def D(self, p: Any) -> Any:
        if self.kind == "probit":
            return stats.norm.cdf(self.a + self.b * p)
        return self.a + self.b * p + self.b2 * p * p

    def inverse(self, q: Any) -> Any:
        q = np.asarray(q, dtype=float)
        if self.kind == "probit":
            with np.errstate(divide="ignore"):
                return (stats.norm.ppf(q) - self.a) / self.b
        if self.b2 == 0.0:
            return (q - self.a) / self.b
        disc = self.b * self.b - 4.0 * self.b2 * (self.a - q)
        with np.errstate(invalid="ignore"):
            return (-self.b - np.sqrt(disc)) / (2.0 * self.b2)

    def inverse_slope(self, q: Any) -> Any:
        p = self.inverse(q)
        if self.kind == "probit":
            return 1.0 / (self.b * stats.norm.pdf(self.a + self.b * p))
        return 1.0 / (self.b + 2.0 * self.b2 * p)

    def area_above(self, p0: Any) -> Any:
        """``integral_{p0}^{choke} D(p) dp``."""
        if self.kind == "probit":
            u = self.a + self.b * p0
            return (u * stats.norm.cdf(u) + stats.norm.pdf(u)) / (-self.b)

        def prim(p: Any) -> Any:
            return self.a * p + 0.5 * self.b * p * p + self.b2 * p**3 / 3.0

        return prim(self.choke) - prim(p0)


def _surplus(dem: _Demand, gamma: float, delta: float, q: Any) -> Any:
    q = np.asarray(q, dtype=float)
    p = dem.inverse(q)
    with np.errstate(invalid="ignore"):
        s = q * (1.0 - delta) * p - q * gamma + dem.area_above(p)
    return np.where(q == 0.0, 0.0, s)


def _surplus_slope(dem: _Demand, gamma: float, delta: float, q: Any) -> Any:
    """``P(q) - MC(q)`` with ``MC(q) = AC(P(q)) + delta q P'(q)``."""
    return (1.0 - delta) * dem.inverse(q) - gamma - delta * q * dem.inverse_slope(q)


def efc_quantities(
    demand: str, coef: Sequence[float], gamma: float, delta: float
) -> np.ndarray:
    """Equilibrium, efficient allocation and deadweight loss.

    ``coef`` holds the demand coefficients at the covariate means
    (``alpha, beta`` or ``alpha, beta, beta2``); the output follows
    :func:`rows_for`.
    """
    dem = _Demand(demand, coef)
    if abs(1.0 - delta) < 1e-10:
        raise NumericalInstability(
            "average cost moves one for one with price, so the equilibrium is "
            "not unique",
            recovery_hint="Inspect the cost regression.",
        )
    p_star = gamma / (1.0 - delta)
    q_eq = float(np.clip(dem.D(p_star), dem.q_lo, dem.q_hi))
    p_at_eq = float(dem.inverse(q_eq))
    p_eq = gamma + delta * p_at_eq if np.isfinite(p_at_eq) else np.nan
    # Efficient coverage: the best of zero, the finite upper end and every
    # interior point where demand crosses marginal cost from above.
    lo, hi = dem.q_lo, dem.q_hi
    eps = 1e-12 if demand == "probit" else 0.0
    grid = np.linspace(lo + eps, hi - eps, _GRID)
    if demand == "probit":
        grid = stats.norm.cdf(np.linspace(-8.0, 8.0, _GRID))
    slope = _surplus_slope(dem, gamma, delta, grid)
    cands: List[float] = [lo]
    if demand != "probit":
        cands.append(hi)
    for j in np.flatnonzero((slope[:-1] > 0) & (slope[1:] <= 0)):
        if slope[j + 1] == 0.0:
            cands.append(float(grid[j + 1]))
            continue
        root = optimize.brentq(
            lambda x: float(_surplus_slope(dem, gamma, delta, x)),
            grid[j],
            grid[j + 1],
            xtol=1e-15,
            rtol=4 * np.finfo(float).eps,
        )
        cands.append(float(root))
    vals = [float(_surplus(dem, gamma, delta, c)) for c in cands]
    q_eff = cands[int(np.nanargmax(vals))]
    p_eff = float(dem.inverse(q_eff))
    s_eff = float(_surplus(dem, gamma, delta, q_eff))
    dwl = s_eff - float(_surplus(dem, gamma, delta, q_eq))
    share = dwl / s_eff if s_eff > 0 else np.nan
    return np.array(
        list(coef) + [gamma, delta, p_eq, q_eq, p_eff, q_eff, dwl, s_eff, share],
        dtype=float,
    )


def inverse_demand(demand: str, curves: Dict[str, float], q: np.ndarray) -> np.ndarray:
    """Inverse demand of a fitted result, for plotting."""
    coef = [curves["alpha"], curves["beta"]] + (
        [curves["beta2"]] if demand == "quadratic" else []
    )
    return np.asarray(_Demand(demand, coef).inverse(q))


# ---------------------------------------------------------------------
# Estimation
# ---------------------------------------------------------------------


def _restrict(
    X: np.ndarray, b: np.ndarray, infl: np.ndarray, R: np.ndarray, r: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Restricted least squares ``R b = r`` and its influence rows."""
    A = np.linalg.inv(X.T @ X)
    K = A @ R.T @ np.linalg.inv(R @ A @ R.T)
    b_r = b - K @ (R @ b - r)
    M = np.eye(len(b)) - K @ R
    return b_r, infl @ M.T


def _probit_fit(
    X: np.ndarray, d: np.ndarray, n_total: int
) -> Tuple[np.ndarray, np.ndarray]:
    """Probit coefficients (sp.probit) and influence rows from the observed
    Hessian, so that the stacked variance is the HC0 sandwich."""
    from ..regression.logit_probit import probit as _probit

    cols = [f"x{j}" for j in range(1, X.shape[1])]
    frame = pd.DataFrame(X[:, 1:], columns=cols)
    frame["_d"] = d
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = _probit(data=frame, y="_d", x=cols, tol=1e-12)
    if fit.model_info.get("n_perfect_prediction_dropped"):
        raise DataInsufficient(
            "the probit demand predicts some choices perfectly",
            recovery_hint="Use demand='linear', or check the price groups.",
        )
    b = fit.params.reindex(["Intercept"] + cols).to_numpy(dtype=float)
    if np.isnan(b).any():
        raise NumericalInstability(
            "the probit demand dropped collinear regressors",
            recovery_hint="Drop collinear covariates.",
        )
    xb = X @ b
    s = 2.0 * d - 1.0
    lam = s * stats.norm.pdf(xb) / stats.norm.cdf(s * xb)
    hess = -(X * (lam * (lam + xb))[:, None]).T @ X
    score = X * lam[:, None]
    return b, n_total * score @ np.linalg.inv(-hess)


def _fit(
    df: pd.DataFrame,
    price: str,
    coverage: str,
    cost: str,
    xs: List[str],
    demand: str,
    through: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray]:
    n = len(df)
    p = df[price].to_numpy(dtype=float)
    Z = [df[c].to_numpy(dtype=float) for c in xs]
    X = np.column_stack([np.ones(n), p] + Z)
    Xd = np.column_stack([np.ones(n), p, p * p] + Z) if demand == "quadratic" else X
    d = df[coverage].to_numpy(dtype=float)
    covered = d == 1
    if demand == "probit":
        b_d, if_d = _probit_fit(Xd, d, n)
    else:
        b_d, if_d = _ols_influence(Xd, d, n, "demand")
        if through is not None:
            R = np.column_stack(
                [np.ones(len(through)), through[:, 1]]
                + ([through[:, 1] ** 2] if demand == "quadratic" else [])
            )
            b_d, if_d = _restrict(Xd, b_d, if_d, R, through[:, 0])
    b_c, if_c_sub = _ols_influence(
        X[covered], df.loc[covered, cost].to_numpy(dtype=float), n, "average cost"
    )
    if_c = np.zeros((n, X.shape[1]))
    if_c[covered] = if_c_sub
    xbar = X[:, 2:].mean(axis=0)
    theta = np.concatenate([b_d, b_c, xbar])
    return theta, np.hstack([if_d, if_c, X[:, 2:] - xbar])


def _from_theta(theta: np.ndarray, m: int, demand: str) -> np.ndarray:
    kd = (3 if demand == "quadratic" else 2) + m
    b_d, b_c, xbar = theta[:kd], theta[kd : kd + 2 + m], theta[kd + 2 + m :]
    slopes = b_d[1 : kd - m]
    coef = [b_d[0] + float(xbar @ b_d[kd - m :])] + list(slopes)
    gamma = b_c[0] + float(xbar @ b_c[2:])
    return efc_quantities(demand, coef, gamma, b_c[1])


def efc_nonlinear(
    df: pd.DataFrame,
    price: str,
    coverage: str,
    cost: str,
    xs: List[str],
    cluster: Optional[str],
    se_method: str,
    n_boot: int,
    seed: Optional[int],
    alpha: float,
    demand: str,
    demand_through: Optional[Sequence[Tuple[float, float]]],
) -> EFCWelfareResult:
    """Body of :func:`statspai.efc_welfare` for the Online Appendix specs."""
    through = None
    if demand_through is not None:
        through = np.asarray(demand_through, dtype=float).reshape(-1, 2)
        n_coef = 3 if demand == "quadratic" else 2
        if demand == "probit":
            raise MethodIncompatibility(
                "demand_through= applies to linear and quadratic demand",
                recovery_hint="A probit cannot pass through coverage 0 or 1; drop "
                "demand_through or use demand='quadratic'.",
            )
        if xs:
            raise MethodIncompatibility(
                "demand_through= cannot be combined with covariates",
                recovery_hint="The constraint fixes the curve at the covariate "
                "means, which are estimated; drop the covariates.",
            )
        if len(through) >= n_coef:
            raise MethodIncompatibility(
                f"{len(through)} points leave no {demand} demand coefficient free",
                recovery_hint=f"Pass at most {n_coef - 1} points.",
            )
    clusters = df[cluster].to_numpy() if cluster is not None else None
    m = len(xs)
    theta, infl = _fit(df, price, coverage, cost, xs, demand, through)
    est = _from_theta(theta, m, demand)
    labels = rows_for(demand)
    z = stats.norm.ppf(1 - alpha / 2)
    n_failed = 0
    if se_method == "delta":
        vcov = _stacked_vcov(infl, clusters)
        grad = _numeric_jacobian(lambda t: _from_theta(t, m, demand), theta)
        with np.errstate(invalid="ignore"):
            se = np.sqrt(np.maximum(np.einsum("ij,jk,ik->i", grad, vcov, grad), 0.0))
        lo, hi = est - z * se, est + z * se
    else:

        def estimate(sub: pd.DataFrame) -> np.ndarray:
            t, _ = _fit(sub, price, coverage, cost, xs, demand, through)
            return _from_theta(t, m, demand)

        draws, n_failed = _efc_bootstrap(
            df,
            price,
            coverage,
            cost,
            xs,
            clusters,
            n_boot,
            seed,
            estimate=estimate,
            n_out=len(labels),
            failures=(
                NumericalInstability,
                IdentificationFailure,
                DataInsufficient,
                np.linalg.LinAlgError,
            ),
        )
        se = np.nanstd(draws, axis=0, ddof=1)
        lo = np.nanquantile(draws, alpha / 2, axis=0)
        hi = np.nanquantile(draws, 1 - alpha / 2, axis=0)
    table = pd.DataFrame(
        {"estimate": est, "se": se, "ci_lower": lo, "ci_upper": hi}, index=list(labels)
    )
    curves = {k: float(v) for k, v in zip(labels, est) if k in labels[:-7]}
    p_obs = df[price].to_numpy(dtype=float)
    p_range = (float(p_obs.min()), float(p_obs.max()))
    q_eq, q_eff = float(table.loc["coverage_eq", "estimate"]), float(
        table.loc["coverage_eff", "estimate"]
    )
    dem = _Demand(demand, [curves[k] for k in labels[: len(labels) - 9]])
    diagnostics: Dict[str, Any] = {
        "selection": "adverse" if curves["delta"] > 0 else "advantageous",
        "equilibrium_corner": bool(q_eq in (dem.q_lo, dem.q_hi)),
        "efficient_corner": bool(q_eff in (dem.q_lo, dem.q_hi)),
        "n_boot_failed": n_failed,
        "demand": demand,
        "demand_through": None if through is None else through.tolist(),
        "max_coverage": dem.q_hi,
    }
    outside = [
        name
        for name in ("price_eq", "price_eff")
        if not p_range[0] <= float(table.loc[name, "estimate"]) <= p_range[1]
    ]
    diagnostics["extrapolated"] = outside
    if outside:
        warnings.warn(
            AssumptionWarning(
                f"{', '.join(outside)} lies outside the observed prices "
                f"[{p_range[0]:g}, {p_range[1]:g}]; the curves are extrapolated",
                recovery_hint="Read the welfare numbers as an extrapolation, "
                "or collect wider price variation.",
                diagnostics={"extrapolated": outside},
            ),
            stacklevel=3,
        )
    return EFCWelfareResult(
        table=table,
        curves=curves,
        se_method=se_method,
        alpha=alpha,
        n_obs=len(df),
        n_covered=int((df[coverage].to_numpy(dtype=float) == 1).sum()),
        price_range=p_range,
        diagnostics=diagnostics,
        demand=demand,
    )
