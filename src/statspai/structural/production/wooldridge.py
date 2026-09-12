"""
Wooldridge (2009) production function estimator.

Two equations share the output elasticities ``b`` and the control function
``h`` (a polynomial of degree ``polynomial_degree`` in state and proxy, no
intercept), with ``x = (free, state)``:

    Eq 1 (level):  y_it = a0 + x_it'b + h_it + eta_it
    Eq 2 (Markov): y_it = c2 + x_it'b + g(h_{i,t-1}) + xi_it + eta_it

``convention="statspai"`` (default) leaves ``g(s) = rho_1 s + ... + rho_d s**d``
free and estimates everything jointly by GMM. Equation 1 is instrumented by
``(1, free, h-terms)``: ``eta`` is not known when inputs are chosen. Equation 2
is instrumented by ``(1, state, lagged free, lagged h-terms)``: ``xi`` arrives
after the lagged choices. The one-step weight is
``diag((Z1'Z1)^{-1}, (Z2'Z2)^{-1})``, with a sandwich covariance clustered by
firm.

``convention="prodest"`` reproduces Stata ``prodest, method(wrdg)``: ``g(s) =
s``, so productivity is a random walk with drift ``c2 - a0``, and both
equations are stacked into one 2SLS in which the free inputs are instrumented
by their lag. When productivity is persistent but stationary (``rho < 1``),
the unit slope leaves ``-(1 - rho) * omega_{t-1}`` in the error of equation 2.
That term moves with current capital, so the capital elasticity is biased:
on the parity panel capital comes out at -0.64 against a true 0.30. R
``prodest::prodestWRDG`` additionally drops ``c2`` and leaves the constant out
of the instrument set; ``test_prodest_parity.py`` rebuilds it from the same
design.

Reference
---------
Wooldridge, J.M. (2009). On estimating firm-level production functions
using proxy variables to control for unobservables. Economics Letters,
104(3), 112-114. [@wooldridge2009estimating]
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import optimize

from ...exceptions import ConvergenceFailure
from ._core import polynomial_basis
from ._result import ProductionResult
from .op_lp_acf import (
    _PID,
    _firm_bootstrap,
    _lags,
    _min_obs,
    _n_gaps,
    _prepare_panel,
    _resolve_inputs,
)

_VCE = ("cluster", "robust", "unadjusted")
_CONVENTIONS = ("statspai", "prodest")


def _wrdg_blocks(
    df: pd.DataFrame,
    output: str,
    free: List[str],
    state: List[str],
    proxy: str,
    polynomial_degree: int,
) -> Dict[str, Any]:
    """Row blocks of both equations on the rows where the lags exist."""
    hvars = [*state, proxy]
    H, _ = polynomial_basis(
        df[hvars].to_numpy(dtype=float),
        degree=polynomial_degree,
        include_intercept=False,
    )
    H_lag, _ = polynomial_basis(
        _lags(df, hvars), degree=polynomial_degree, include_intercept=False
    )
    F_lag = _lags(df, free)
    valid = np.isfinite(H_lag).all(axis=1) & np.isfinite(F_lag).all(axis=1)
    n = int(valid.sum())
    _min_obs(n)
    return {
        "y1": df[output].to_numpy(dtype=float)[valid],
        "F": df[free].to_numpy(dtype=float)[valid],
        "F_lag": F_lag[valid],
        "S": df[state].to_numpy(dtype=float)[valid],
        "H": H[valid],
        "H_lag": H_lag[valid],
        "ids": df[_PID].to_numpy()[valid],
        "valid": valid,
        "n": n,
    }


def _wrdg_design(
    df: pd.DataFrame,
    output: str,
    free: List[str],
    state: List[str],
    proxy: str,
    polynomial_degree: int,
) -> Dict[str, Any]:
    """Stata prodest's stacked system (eq 1 rows over eq 2 rows).

    Columns of ``X``: ``const, e0, free..., state..., h-terms...``; ``Z``
    replaces the free inputs by their lags.
    """
    B = _wrdg_blocks(df, output, free, state, proxy, polynomial_degree)
    n = B["n"]
    one, zero = np.ones(n), np.zeros(n)
    X = np.vstack(
        [
            np.column_stack([one, zero, B["F"], B["S"], B["H"]]),
            np.column_stack([one, one, B["F"], B["S"], B["H_lag"]]),
        ]
    )
    Z = np.vstack(
        [
            np.column_stack([one, zero, B["F_lag"], B["S"], B["H"]]),
            np.column_stack([one, one, B["F_lag"], B["S"], B["H_lag"]]),
        ]
    )
    return {
        **B,
        "y": np.concatenate([B["y1"], B["y1"]]),
        "X": X,
        "Z": Z,
        "cluster": np.concatenate([B["ids"], B["ids"]]),
        "n_free": B["F"].shape[1],
        "n_state": B["S"].shape[1],
        "n_h": B["H"].shape[1],
    }


def _two_sls(design: Dict[str, Any], vce: str) -> Dict[str, Any]:
    X, Z, Y = design["X"], design["Z"], design["y"]
    Xhat = Z @ np.linalg.lstsq(Z, X, rcond=None)[0]
    A = Xhat.T @ X
    b = np.linalg.solve(A, Xhat.T @ Y)
    u = Y - X @ b
    A_inv = np.linalg.inv(A)
    N = X.shape[0]
    if vce == "unadjusted":
        # Stata ivregress's default VCE for wmatrix(unadjusted): u'u / N.
        V = (u @ u / N) * A_inv
    elif vce == "robust":
        meat = (Xhat * u[:, None]).T @ (Xhat * u[:, None])
        V = A_inv @ meat @ A_inv.T
    else:
        scores = pd.DataFrame(Xhat * u[:, None]).groupby(design["cluster"]).sum()
        G = scores.shape[0]
        g = scores.to_numpy()
        V = (G / (G - 1)) * A_inv @ (g.T @ g) @ A_inv.T
    return {"b": b, "V": V, "u": u}


class _WooldridgeGMM:
    """Joint GMM for the two equations with a free Markov polynomial."""

    def __init__(self, blocks: Dict[str, Any], degree: int) -> None:
        n = blocks["n"]
        one = np.ones(n)
        self.y = blocks["y1"]
        self.X = np.column_stack([blocks["F"], blocks["S"]])
        self.H, self.H_lag = blocks["H"], blocks["H_lag"]
        self.Z1 = np.column_stack([one, blocks["F"], self.H])
        self.Z2 = np.column_stack([one, blocks["S"], blocks["F_lag"], self.H_lag])
        self.W1 = np.linalg.inv(self.Z1.T @ self.Z1)
        self.W2 = np.linalg.inv(self.Z2.T @ self.Z2)
        self.L1 = np.linalg.cholesky(self.W1)
        self.L2 = np.linalg.cholesky(self.W2)
        self.ids = blocks["ids"]
        self.n, self.p, self.nh, self.d = n, self.X.shape[1], self.H.shape[1], degree

    def split(
        self, th: np.ndarray
    ) -> Tuple[float, float, np.ndarray, np.ndarray, np.ndarray]:
        p, nh = self.p, self.nh
        return th[0], th[1], th[2 : 2 + p], th[2 + p : 2 + p + nh], th[2 + p + nh :]

    def residuals(self, th: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        a0, c2, b, g, rho = self.split(th)
        s = self.H_lag @ g
        u1 = self.y - a0 - self.X @ b - self.H @ g
        u2 = (
            self.y - c2 - self.X @ b - sum(rho[j] * s ** (j + 1) for j in range(self.d))
        )
        return u1, u2, s

    def jacobians(self, th: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        _, _, _, _, rho = self.split(th)
        _, _, s = self.residuals(th)
        one, zero = np.ones(self.n), np.zeros(self.n)
        J1 = np.column_stack([-one, zero, -self.X, -self.H, np.zeros((self.n, self.d))])
        ds = sum((j + 1) * rho[j] * s**j for j in range(self.d))
        powers = np.column_stack([s ** (j + 1) for j in range(self.d)])
        J2 = np.column_stack(
            [zero, -one, -self.X, -(ds[:, None] * self.H_lag), -powers]
        )
        return J1, J2

    def _scaled(self, th: np.ndarray) -> np.ndarray:
        u1, u2, _ = self.residuals(th)
        return np.concatenate(
            [self.L1.T @ (self.Z1.T @ u1), self.L2.T @ (self.Z2.T @ u2)]
        )

    def _scaled_jac(self, th: np.ndarray) -> np.ndarray:
        J1, J2 = self.jacobians(th)
        return np.vstack([self.L1.T @ (self.Z1.T @ J1), self.L2.T @ (self.Z2.T @ J2)])

    def solve(self, start: np.ndarray) -> optimize.OptimizeResult:
        fit = optimize.least_squares(
            self._scaled,
            start,
            jac=self._scaled_jac,
            method="lm",
            xtol=1e-15,
            ftol=1e-15,
            gtol=1e-15,
            max_nfev=10000,
        )
        if fit.status <= 0 or not np.all(np.isfinite(fit.x)):
            raise ConvergenceFailure(f"Wooldridge GMM did not converge: {fit.message}")
        return fit

    def vcov(self, th: np.ndarray, vce: str) -> np.ndarray:
        u1, u2, _ = self.residuals(th)
        J1, J2 = self.jacobians(th)
        G = np.vstack([self.Z1.T @ J1, self.Z2.T @ J2])
        q1 = self.Z1.shape[1]
        W = np.zeros((G.shape[0], G.shape[0]))
        W[:q1, :q1], W[q1:, q1:] = self.W1, self.W2
        scores = np.column_stack([self.Z1 * u1[:, None], self.Z2 * u2[:, None]])
        factor = 1.0
        if vce == "cluster":
            summed = pd.DataFrame(scores).groupby(self.ids).sum()
            n_cl = summed.shape[0]
            scores = summed.to_numpy()
            factor = n_cl / (n_cl - 1)
        S = scores.T @ scores
        A_inv = np.linalg.inv(G.T @ W @ G)
        return factor * A_inv @ (G.T @ W @ S @ W @ G) @ A_inv


def _gmm_start(design: Dict[str, Any], degree: int) -> np.ndarray:
    """prodest's 2SLS (g(s) = s) as the starting point: rho_1 = 1."""
    b = _two_sls(design, "unadjusted")["b"]
    rho0 = np.zeros(degree)
    rho0[0] = 1.0
    return np.concatenate([[b[0], b[0] + b[1]], b[2:], rho0])


def wooldridge_prod(
    data: pd.DataFrame,
    output: str = "y",
    free: Sequence[str] | str | None = None,
    state: Sequence[str] | str | None = None,
    proxy: str = "m",
    panel_id: str = "id",
    time: str = "year",
    polynomial_degree: int = 3,
    productivity_degree: Optional[int] = None,
    functional_form: str = "cobb-douglas",
    boot_reps: int = 0,
    seed: Optional[int] = None,
    vce: str = "cluster",
    convention: str = "statspai",
) -> ProductionResult:
    """Wooldridge (2009) production function estimator.

    Estimates the level equation and the productivity-substituted equation
    jointly; they share the input elasticities and the control-function
    polynomial ``h(state, proxy)``. By default the Markov process
    ``omega_t = g(omega_{t-1}) + xi_t`` has a free polynomial ``g`` and the
    system is estimated by GMM (equation 1 instrumented by current free
    inputs and ``h``, equation 2 by state, lagged free inputs and lagged
    ``h``). ``convention="prodest"`` reproduces Stata ``prodest,
    method(wrdg)``, which imposes a unit slope (a random walk) and runs one
    stacked 2SLS. When productivity mean-reverts, that restriction biases the
    capital elasticity (see the module docstring).

    Parameters
    ----------
    data : DataFrame
        Long panel with one row per (firm, year).
    output : str, default ``"y"``
        Log output column.
    free : str or list, default ``["l"]``
        Free inputs (labor).
    state : str or list, default ``["k"]``
        State inputs (capital).
    proxy : str, default ``"m"``
        Productivity proxy (typically intermediate input).
    panel_id, time : str
        Firm and (numeric) year identifiers; lags are calendar lags.
    polynomial_degree : int, default 3
        Degree of ``h(state, proxy)`` (Stata ``prodest``'s default).
    productivity_degree : int, optional
        Degree of ``g`` (default 3, as OP / LP / ACF). Must be left ``None``
        with ``convention="prodest"``, which fixes ``g(s) = s``.
    functional_form : {'cobb-douglas'}, default 'cobb-douglas'
        Translog raises ``NotImplementedError``.
    boot_reps : int, default 0
        If positive, firm-cluster bootstrap standard errors replace the
        analytic ones.
    seed : int, optional
        Bootstrap RNG seed.
    vce : {"cluster", "robust", "unadjusted"}, default "cluster"
        Analytic covariance. ``"cluster"`` clusters by firm (``G/(G-1)``):
        both equations of a firm-year share ``eta``. ``"robust"`` treats
        firm-years as independent. ``"unadjusted"`` (``convention="prodest"``
        only) is Stata ``prodest``'s, ``u'u/N`` over the stacked rows.
    convention : {"statspai", "prodest"}, default "statspai"
        Estimator, as described above.

    Returns
    -------
    ProductionResult

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> rows = []
    >>> for fid in range(60):
    ...     k = rng.normal(2.0, 0.5)
    ...     omega = rng.normal(0.0, 0.3)
    ...     w = rng.normal(0.0, 0.3)
    ...     for yr in range(8):
    ...         omega = 0.7 * omega + rng.normal(0.0, 0.2)
    ...         w = 0.8 * w + rng.normal(0.0, 0.3)
    ...         l = 0.6 * k + omega - 0.5 * w + rng.normal(1.0, 0.05)
    ...         m = 0.5 * k + omega + rng.normal(0.5, 0.05)
    ...         y = 0.6 * l + 0.3 * k + omega + rng.normal(0.0, 0.1)
    ...         rows.append(dict(id=fid, year=yr, y=y, l=l, k=k, m=m))
    ...         k = 0.9 * k + 0.3 * rng.normal(1.0, 0.3)
    >>> df = pd.DataFrame(rows)
    >>> res = sp.wooldridge_prod(df, output="y", free="l", state="k", proxy="m")
    >>> sorted(res.coef.keys())
    ['k', 'l']
    >>> res.method
    'wrdg'

    References
    ----------
    Wooldridge, J.M. (2009). On estimating firm-level production
    functions using proxy variables to control for unobservables.
    Economics Letters, 104(3), 112-114.
    """
    if functional_form.lower().replace("_", "-") not in ("cobb-douglas", "cd"):
        raise NotImplementedError(
            "wooldridge_prod currently only supports functional_form="
            "'cobb-douglas'. For translog use sp.ackerberg_caves_frazer."
        )
    if convention not in _CONVENTIONS:
        raise ValueError(
            f"convention must be one of {_CONVENTIONS}; got {convention!r}."
        )
    if vce not in _VCE:
        raise ValueError(f"vce must be one of {_VCE}; got {vce!r}.")
    if convention == "prodest" and productivity_degree is not None:
        raise ValueError(
            "convention='prodest' fixes g(s) = s (unit slope); leave "
            "productivity_degree=None."
        )
    if convention == "statspai" and vce == "unadjusted":
        raise ValueError(
            "vce='unadjusted' is available with convention='prodest' only."
        )
    degree = 3 if productivity_degree is None else int(productivity_degree)
    if degree < 1:
        raise ValueError(
            f"productivity_degree must be >= 1; got {productivity_degree!r}."
        )

    free, state = _resolve_inputs(free, state, ["l"], ["k"])
    clash = sorted(set(free) & ({proxy} | set(state)))
    if clash:
        raise ValueError(
            f"{clash} appear among the free inputs and the state/proxy "
            "variables; they must be distinct."
        )
    df = _prepare_panel(data, output, free, state, proxy, panel_id, time)
    p = len(free) + len(state)

    def estimate(frame: pd.DataFrame, cov_type: Optional[str]) -> Dict[str, Any]:
        design = _wrdg_design(frame, output, free, state, proxy, polynomial_degree)
        if convention == "prodest":
            fit = _two_sls(design, cov_type or "unadjusted")
            b, n = fit["b"], design["n"]
            return {
                "beta": b[2 : 2 + p],
                "V": fit["V"][2 : 2 + p, 2 : 2 + p],
                "a0": b[0],
                "h_coef": b[2 + p :],
                "u1": fit["u"][:n],
                "u2": fit["u"][n:],
                "markov": [float(b[1]), 1.0],
                "design": design,
                "converged": True,
            }
        gmm = _WooldridgeGMM(design, degree)
        fit = gmm.solve(_gmm_start(design, degree))
        a0, c2, beta, h_coef, rho = gmm.split(fit.x)
        u1, u2, _ = gmm.residuals(fit.x)
        V = gmm.vcov(fit.x, cov_type)[2 : 2 + p, 2 : 2 + p] if cov_type else None
        return {
            "beta": beta,
            "V": V,
            "a0": a0,
            "h_coef": h_coef,
            "u1": u1,
            "u2": u2,
            "markov": [float(c2 - a0), *map(float, rho)],
            "design": design,
            "converged": bool(fit.status > 0),
        }

    est = estimate(df, vce)
    beta = est["beta"]
    cov: Optional[np.ndarray] = est["V"]
    se = np.sqrt(np.diag(cov))

    n_boot = 0
    se_source = f"analytic ({vce})"
    if boot_reps and boot_reps > 0:
        se_b, cov_b, n_boot = _firm_bootstrap(
            df,
            lambda frame: estimate(frame, None)["beta"],
            boot_reps,
            seed,
            "Wooldridge",
        )
        se = se_b if se_b.size == p else np.full(p, np.nan)
        cov = cov_b
        se_source = "firm bootstrap"

    design = est["design"]
    omega = est["a0"] + design["H"] @ est["h_coef"]
    u1, u2 = est["u1"], est["u2"]
    xi = u2 - u1
    sigma_xi = float(np.std(xi, ddof=1))
    rho = est["markov"][1]
    names = [*free, *state]
    sample = df.loc[design["valid"]].reset_index(drop=True)
    sample["omega"] = omega
    sample["eta"] = u1

    diagnostics = {
        "convention": convention,
        "vce": vce,
        "se_source": se_source,
        # g(s) = markov_coef[0] + markov_coef[1] s + ..., s = h_{t-1}; the
        # intercept is the drift relative to a0.
        "markov_coef": est["markov"],
        "markov_intercept": est["markov"][0],
        "ar_rho": rho,
        "ar_sigma_xi": sigma_xi,
        "stage2_converged": est["converged"],
        "polynomial_degree": int(polynomial_degree),
        "productivity_degree": 1 if convention == "prodest" else degree,
        "n_obs_stacked": int(2 * design["n"]),
        "n_clusters": int(np.unique(design["ids"]).size),
        "boot_reps_effective": n_boot,
        "n_calendar_gaps": _n_gaps(df),
    }

    return ProductionResult(
        method="wrdg",
        params=pd.Series(beta, index=names, name="elasticity"),
        std_errors=pd.Series(se, index=names, name="std_error"),
        coef={name: float(beta[i]) for i, name in enumerate(names)},
        tfp=omega,
        residuals=u1,
        productivity_process={"rho": rho, "sigma": sigma_xi},
        sample=sample,
        diagnostics=diagnostics,
        model_info={
            "free_inputs": free,
            "state_inputs": state,
            "proxy": proxy,
            "functional_form": "cobb-douglas",
            "convention": convention,
        },
        cov=cov,
    )
