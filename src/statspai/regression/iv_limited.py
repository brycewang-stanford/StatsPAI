"""
Probit and tobit models with continuous endogenous regressors.

    y1* = y2'beta + x1'gamma + u            (outcome, latent)
    y2  = x1'Pi1 + x2'Pi2 + v               (reduced form, one row per y2)
    (u, v) ~ N(0, Sigma)

``y1 = 1[y1* > 0]`` for :func:`ivprobit` (``Var(u) = 1``) and ``y1`` is
``y1*`` censored at ``ll`` / ``ul`` for :func:`ivtobit`. Both follow
Stata's ``ivprobit`` and ``ivtobit``: maximum likelihood by default,
Newey's minimum chi-squared estimator with ``method='twostep'``.

The likelihood, the ancillary parameters and their names
(``/athrho2_1``, ``/lnsigma2``: equation 1 is the outcome, equations
2, 3, ... the endogenous regressors) are Stata's, so a fit can be laid
next to Stata output line by line.

References
----------
[@newey1987efficient] for the minimum chi-squared estimator,
[@rivers1988limited] for the probit control function and its exogeneity
test, [@smith1986exogeneity] for the tobit counterpart.
"""

from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy import optimize, special, stats

from .._aliases import accepts_aliases
from ..core._vcov import ml_vcov
from ..core.results import CausalResult, EconometricResults
from ..exceptions import DataInsufficient, MethodIncompatibility
from ._optim_helpers import (
    complex_step_scores,
    inverse_information,
    ml_newton_polish,
    se_from_vcov,
)

__all__ = ["ivprobit", "ivtobit"]

_LOG_2PI = float(np.log(2.0 * np.pi))


def _as_list(value: Union[str, Sequence[str], None]) -> List[str]:
    if value is None:
        return []
    if isinstance(value, str):
        return [value]
    return list(value)


class _Design:
    """Aligned arrays for one fit; column order follows Stata's output."""

    def __init__(
        self,
        data: pd.DataFrame,
        y: str,
        x: Union[str, Sequence[str], None],
        endog: Union[str, Sequence[str], None],
        instruments: Union[str, Sequence[str], None],
        cluster: Optional[str],
        function: str,
    ) -> None:
        self.y_name = y
        self.x = _as_list(x)
        self.endog = _as_list(endog)
        self.instruments = _as_list(instruments)
        if not self.endog:
            raise MethodIncompatibility(
                f"{function}: endog= is empty. Without an endogenous regressor "
                f"this is sp.{function[2:]}; name at least one in endog=."
            )
        needed = [y] + self.endog + self.x + self.instruments
        overlap = sorted(
            (set(self.endog) & set(self.x))
            | (set(self.endog) & set(self.instruments))
            | (set(self.x) & set(self.instruments))
        )
        if overlap:
            raise MethodIncompatibility(
                f"{function}: {overlap} appear in more than one of x=, endog=, "
                "instruments=. Each variable belongs to exactly one list.",
                diagnostics={"overlap": overlap},
            )
        extra = [cluster] if isinstance(cluster, str) else []
        missing = [c for c in needed + extra if c not in data]
        if missing:
            raise MethodIncompatibility(
                f"{function}: columns not found in data: {missing}",
                diagnostics={"missing": missing},
            )
        if len(self.instruments) < len(self.endog):
            raise MethodIncompatibility(
                f"{function}: {len(self.endog)} endogenous regressor(s) but only "
                f"{len(self.instruments)} excluded instrument(s); the order "
                "condition for identification fails.",
                diagnostics={
                    "n_endog": len(self.endog),
                    "n_instruments": len(self.instruments),
                },
            )
        df = data[list(dict.fromkeys(needed + extra))].dropna()
        self.df = df
        n = len(df)
        one = np.ones(n)
        self.Y1 = df[y].to_numpy(dtype=float)
        self.Y2 = df[self.endog].to_numpy(dtype=float)
        ex = [df[v].to_numpy(dtype=float) for v in self.x]
        # z: right-hand side of the outcome equation; X: all exogenous.
        self.Z = np.column_stack([self.Y2] + ex + [one])
        self.X = np.column_stack(
            ex + [df[v].to_numpy(dtype=float) for v in self.instruments] + [one]
        )
        self.z_names = self.endog + self.x + ["_cons"]
        self.x_names = self.x + self.instruments + ["_cons"]
        self.n, self.p = n, len(self.endog)
        self.kz, self.kx = self.Z.shape[1], self.X.shape[1]
        if n <= self.kx + self.kz:
            raise DataInsufficient(
                f"{function}: {n} complete observations for "
                f"{self.kx + self.kz} mean parameters."
            )
        if np.linalg.matrix_rank(self.X) < self.kx:
            raise MethodIncompatibility(
                f"{function}: the exogenous variables and instruments are "
                "collinear; drop the redundant column."
            )
        # First stage: OLS of every endogenous regressor on X.
        self.Pi = np.linalg.lstsq(self.X, self.Y2, rcond=None)[0]  # (kx, p)
        self.Vhat = self.Y2 - self.X @ self.Pi
        rank_pi = np.linalg.matrix_rank(self.Pi[len(self.x) : -1, :])
        if rank_pi < self.p:
            raise MethodIncompatibility(
                f"{function}: the excluded instruments do not move every "
                "endogenous regressor independently (rank condition fails)."
            )
        self.clusters = df[cluster].to_numpy() if isinstance(cluster, str) else None


# ---------------------------------------------------------------------------
# Single-equation building blocks (probit and tobit on a given design)
# ---------------------------------------------------------------------------


def _probit_fit(y: np.ndarray, W: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Probit by Newton-Raphson. Returns (coefficients, inverse information)."""
    q = 2.0 * y - 1.0
    b = np.zeros(W.shape[1])
    H = np.eye(W.shape[1])
    for _ in range(100):
        xb = W @ b
        lam = q * np.exp(stats.norm.logpdf(q * xb) - special.log_ndtr(q * xb))
        H = (W * (lam * (xb + lam))[:, None]).T @ W
        step = np.linalg.solve(H, W.T @ lam)
        b = b + step
        if np.max(np.abs(step)) < 1e-12 * (1.0 + np.max(np.abs(b))):
            break
    xb = W @ b
    lam = q * np.exp(stats.norm.logpdf(q * xb) - special.log_ndtr(q * xb))
    H = (W * (lam * (xb + lam))[:, None]).T @ W
    return b, np.linalg.inv(H)


def _censor_masks(
    y: np.ndarray, ll: Optional[float], ul: Optional[float]
) -> Tuple[np.ndarray, np.ndarray]:
    lo = y <= ll if ll is not None else np.zeros(len(y), dtype=bool)
    hi = y >= ul if ul is not None else np.zeros(len(y), dtype=bool)
    return lo, hi


def _tobit_obs_loglik(
    y: np.ndarray,
    index: np.ndarray,
    ln_s: Any,
    lo: np.ndarray,
    hi: np.ndarray,
    ll: Optional[float],
    ul: Optional[float],
) -> np.ndarray:
    """Censored-normal log density around ``index`` with sd ``exp(ln_s)``."""
    s = np.exp(ln_s)
    out = np.zeros(len(y), dtype=np.result_type(index, ln_s, float))
    mid = ~lo & ~hi
    r = (y[mid] - index[mid]) / s
    out[mid] = -0.5 * _LOG_2PI - ln_s - 0.5 * r * r
    if lo.any():
        out[lo] = special.log_ndtr((ll - index[lo]) / s)
    if hi.any():
        out[hi] = special.log_ndtr((index[hi] - ul) / s)
    return out


def _tobit_fit(
    y: np.ndarray,
    W: np.ndarray,
    lo: np.ndarray,
    hi: np.ndarray,
    ll: Optional[float],
    ul: Optional[float],
) -> Tuple[np.ndarray, np.ndarray]:
    """Tobit on a design matrix. Returns ([beta, ln sigma], inverse information)."""
    k = W.shape[1]
    mid = ~lo & ~hi
    b0 = np.linalg.lstsq(W[mid], y[mid], rcond=None)[0]
    s0 = max(float(np.std(y[mid] - W[mid] @ b0)), 1e-3)

    def obs(theta: np.ndarray) -> np.ndarray:
        return _tobit_obs_loglik(y, W @ theta[:k], theta[k], lo, hi, ll, ul)

    theta = _maximise(obs, np.append(b0, np.log(s0)))
    theta, _, H, _ = ml_newton_polish(obs, theta)
    return theta, inverse_information(H)


def _maximise(obs: Callable[[np.ndarray], np.ndarray], start: np.ndarray) -> np.ndarray:
    """BFGS on the summed log-likelihood with complex-step gradients."""

    def fun(t: np.ndarray) -> float:
        with np.errstate(all="ignore"):
            v = float(np.sum(np.real(obs(t))))
        return -v if np.isfinite(v) else 1e300

    def jac(t: np.ndarray) -> np.ndarray:
        with np.errstate(all="ignore"):
            g = -complex_step_scores(obs, t).sum(axis=0)
        return np.asarray(np.where(np.isfinite(g), g, 0.0))

    res = optimize.minimize(
        fun, start, jac=jac, method="BFGS", options={"maxiter": 500, "gtol": 1e-7}
    )
    return np.asarray(res.x, dtype=float)


# ---------------------------------------------------------------------------
# Joint likelihood
# ---------------------------------------------------------------------------


class _Layout:
    """Positions of the parameter blocks inside the optimisation vector.

    ``delta`` (outcome equation), ``Pi`` (reduced forms, stacked by
    endogenous regressor), the Cholesky factor ``L`` of ``Var(v)`` (log
    diagonal, then the strict lower triangle by rows), the loading ``t``
    of ``u`` on the standardised reduced-form errors and, for the tobit,
    the log of the conditional standard deviation.
    """

    def __init__(self, d: _Design, tobit: bool) -> None:
        p, kz, kx = d.p, d.kz, d.kx
        self.p, self.kz, self.kx, self.tobit = p, kz, kx, tobit
        self.i_delta = slice(0, kz)
        self.i_pi = slice(kz, kz + p * kx)
        a = kz + p * kx
        self.i_ldiag = slice(a, a + p)
        self.n_off = p * (p - 1) // 2
        self.i_loff = slice(a + p, a + p + self.n_off)
        a = a + p + self.n_off
        self.i_t = slice(a, a + p)
        self.i_lns = a + p if tobit else None
        self.size = a + p + (1 if tobit else 0)
        self.tril = np.tril_indices(p, -1)

    def chol(self, theta: np.ndarray) -> np.ndarray:
        L = np.zeros((self.p, self.p), dtype=theta.dtype)
        L[np.diag_indices(self.p)] = np.exp(theta[self.i_ldiag])
        if self.n_off:
            L[self.tril] = theta[self.i_loff]
        return L


def _joint_obs_loglik(
    d: _Design,
    lay: _Layout,
    lo: Optional[np.ndarray],
    hi: Optional[np.ndarray],
    ll: Optional[float],
    ul: Optional[float],
) -> Callable[[np.ndarray], np.ndarray]:
    q = 2.0 * d.Y1 - 1.0

    def obs(theta: np.ndarray) -> np.ndarray:
        theta = np.asarray(theta)
        Pi = theta[lay.i_pi].reshape(lay.p, lay.kx).T
        L = lay.chol(theta)
        V = d.Y2 - d.X @ Pi
        # e = L^{-1} v: standardised reduced-form errors, identity covariance.
        E = np.linalg.solve(L, V.T).T
        log_fv = (
            -0.5 * lay.p * _LOG_2PI
            - np.sum(theta[lay.i_ldiag])
            - 0.5 * np.sum(E * E, axis=1)
        )
        t = theta[lay.i_t]
        zd = d.Z @ theta[lay.i_delta]
        if lay.tobit:
            assert lo is not None and hi is not None
            return np.asarray(
                log_fv
                + _tobit_obs_loglik(d.Y1, zd + E @ t, theta[lay.i_lns], lo, hi, ll, ul)
            )
        # u = l'e + l_uu * eps with |l|^2 + l_uu^2 = 1 and l = t * l_uu, so
        # Var(u) = 1 holds for every t and no constraint is needed.
        index = np.sqrt(1.0 + np.sum(t * t)) * zd + E @ t
        return np.asarray(log_fv + special.log_ndtr(q * index))

    return obs


def _stata_ancillary(
    lay: _Layout,
) -> Tuple[Callable[[np.ndarray], np.ndarray], List[str]]:
    """Map the optimisation vector to Stata's reported parameters.

    Stata numbers the outcome equation 1 and the endogenous regressors
    2, ..., p+1, and reports ``atanh`` of every pairwise correlation of
    (u, v) and the log standard deviations.
    """
    p = lay.p
    names = [f"/athrho{j + 2}_{i + 1}" for j in range(p) for i in range(j + 1)]
    names += (["/lnsigma1"] if lay.tobit else []) + [
        f"/lnsigma{j + 2}" for j in range(p)
    ]

    def g(theta: np.ndarray) -> np.ndarray:
        theta = np.asarray(theta)
        L = lay.chol(theta)
        t = theta[lay.i_t]
        if lay.tobit:
            a = t
            var_u = np.sum(a * a) + np.exp(2.0 * theta[lay.i_lns])
        else:
            a = t / np.sqrt(1.0 + np.sum(t * t))
            var_u = 1.0 + 0.0 * theta[0]
        S = np.zeros((p + 1, p + 1), dtype=theta.dtype)
        S[0, 0] = var_u
        S[1:, 0] = S[0, 1:] = L @ a
        S[1:, 1:] = L @ L.T
        sd = np.sqrt(np.diag(S))
        ath = [
            np.arctanh(S[j + 1, i] / (sd[j + 1] * sd[i]))
            for j in range(p)
            for i in range(j + 1)
        ]
        lns = ([np.log(sd[0])] if lay.tobit else []) + [
            np.log(sd[j + 1]) for j in range(p)
        ]
        head = theta[: lay.i_pi.stop]
        return np.concatenate([head, np.array(ath + lns, dtype=theta.dtype)])

    return g, names


def _complex_jacobian(
    g: Callable[[np.ndarray], np.ndarray], theta: np.ndarray
) -> np.ndarray:
    h = 1e-20
    cols = []
    for j in range(len(theta)):
        tc = theta.astype(complex)
        tc[j] += 1j * h
        cols.append(np.imag(g(tc)) / h)
    return np.column_stack(cols)


def _fit_mle(
    d: _Design,
    tobit: bool,
    ll: Optional[float],
    ul: Optional[float],
    se_kind: str,
) -> Dict[str, Any]:
    lay = _Layout(d, tobit)
    lo = hi = None
    if tobit:
        lo, hi = _censor_masks(d.Y1, ll, ul)
    obs = _joint_obs_loglik(d, lay, lo, hi, ll, ul)

    # Start from the control-function fit: the outcome model on z and the
    # standardised first-stage residuals.
    S22 = d.Vhat.T @ d.Vhat / d.n
    L0 = np.linalg.cholesky(S22)
    E0 = np.linalg.solve(L0, d.Vhat.T).T
    W = np.column_stack([d.Z, E0])
    start = np.zeros(lay.size)
    start[lay.i_pi] = d.Pi.T.ravel()
    start[lay.i_ldiag] = np.log(np.diag(L0))
    if lay.n_off:
        start[lay.i_loff] = L0[lay.tril]
    if tobit:
        assert lo is not None and hi is not None
        th, _ = _tobit_fit(d.Y1, W, lo, hi, ll, ul)
        start[lay.i_delta] = th[: d.kz]
        start[lay.i_t] = th[d.kz : d.kz + d.p]
        start[lay.i_lns] = th[-1]
    else:
        b, _ = _probit_fit(d.Y1, W)
        t0 = b[d.kz :]
        start[lay.i_delta] = b[: d.kz] / np.sqrt(1.0 + t0 @ t0)
        start[lay.i_t] = t0

    theta = _maximise(obs, start)
    theta, scores, H, _ = ml_newton_polish(obs, theta)
    grad_norm = float(np.max(np.abs(scores.sum(axis=0))))
    V_theta = ml_vcov(
        inverse_information(H),
        scores if se_kind != "nonrobust" else None,
        kind=se_kind,
        clusters=d.clusters if se_kind == "cluster" else None,
    )
    g, anc_names = _stata_ancillary(lay)
    J = _complex_jacobian(g, theta)
    b_out = np.real(g(theta.astype(complex)))
    V_out = J @ V_theta @ J.T
    return {
        "b": b_out,
        "V": V_out,
        "anc_names": anc_names,
        "ll": float(np.sum(obs(theta))),
        "grad_norm": grad_norm,
        "converged": bool(grad_norm < 1e-5 * max(1.0, d.n**0.5)),
        "n_ancillary": len(anc_names),
    }


# ---------------------------------------------------------------------------
# Newey's minimum chi-squared estimator
# ---------------------------------------------------------------------------


def _fit_twostep(
    d: _Design, tobit: bool, ll: Optional[float], ul: Optional[float]
) -> Dict[str, Any]:
    """Newey (1987) as laid out in the Stata manual for ``ivprobit``.

    The reduced form ``y1* = X alpha + vhat lambda + e`` is fitted with the
    first-stage residuals as extra regressors; ``alpha = D(Pi) delta`` with
    ``D = [Pi, I1]`` ties it to the structural parameters, and ``delta`` is
    recovered by minimum distance with the efficient weight.
    """
    n, p, kx, kz = d.n, d.p, d.kx, d.kz
    lo = hi = None
    if tobit:
        lo, hi = _censor_masks(d.Y1, ll, ul)

    def fit(W: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if tobit:
            assert lo is not None and hi is not None
            return _tobit_fit(d.Y1, W, lo, hi, ll, ul)
        return _probit_fit(d.Y1, W)

    # Reduced form: alpha on X, lambda on the residuals.
    th_rf, V_rf = fit(np.column_stack([d.X, d.Vhat]))
    alpha_t = th_rf[:kx]
    lam = th_rf[kx : kx + p]
    J_inv = V_rf[:kx, :kx]
    # 2SIV: the structural index with the residuals as control function.
    th_cf, V_cf = fit(np.column_stack([d.Z, d.Vhat]))
    beta_t = th_cf[:p]
    lam_cf = th_cf[kz : kz + p]
    V_lam_cf = V_cf[kz : kz + p, kz : kz + p]

    # Var(alpha~) picks up the first-stage noise through y2 (lambda - beta).
    w = d.Y2 @ (lam - beta_t)
    XtX_inv = np.linalg.inv(d.X.T @ d.X)
    resid = w - d.X @ (XtX_inv @ (d.X.T @ w))
    omega = J_inv + (resid @ resid / (n - kx)) * XtX_inv

    # D maps delta = (beta, gamma, const) to alpha: Pi for the endogenous
    # regressors, a selection of X's own columns for the exogenous ones.
    D = np.zeros((kx, kz))
    D[:, :p] = d.Pi
    for j in range(len(d.x)):
        D[j, p + j] = 1.0
    D[kx - 1, kz - 1] = 1.0
    om_inv = np.linalg.inv(omega)
    V = np.linalg.inv(D.T @ om_inv @ D)
    delta = V @ (D.T @ om_inv @ alpha_t)
    chi2_exog = float(lam_cf @ np.linalg.solve(V_lam_cf, lam_cf))
    return {"b": delta, "V": V, "chi2_exog": chi2_exog}


# ---------------------------------------------------------------------------
# Result assembly
# ---------------------------------------------------------------------------


def _assemble(
    d: _Design,
    function: str,
    method: str,
    tobit: bool,
    ll: Optional[float],
    ul: Optional[float],
    se_kind: str,
    cluster: Optional[str],
    alpha: float,
) -> EconometricResults:
    y = d.y_name
    struct_names = [f"{y}:{v}" for v in d.z_names]
    info: Dict[str, Any] = {
        "alpha": alpha,
        "model_type": "IV tobit" if tobit else "IV probit",
        "citation_key": function,
        "method": method,
        "vce": se_kind,
        "cluster": cluster if se_kind == "cluster" else None,
        "endog": list(d.endog),
        "instruments": list(d.instruments),
    }
    if tobit:
        lo, hi = _censor_masks(d.Y1, ll, ul)
        info.update(
            lower_limit=ll,
            upper_limit=ul,
            n_left_censored=int(lo.sum()),
            n_right_censored=int(hi.sum()),
            n_uncensored=int((~lo & ~hi).sum()),
        )
    if method == "twostep":
        fit = _fit_twostep(d, tobit, ll, ul)
        names = list(d.z_names)
        b, V = fit["b"], fit["V"]
        chi2_exog = fit["chi2_exog"]
        info["normalisation"] = (
            "Var(u | v) = 1 (Newey); the maximum likelihood fit sets "
            "Var(u) = 1, so the two coefficient vectors differ by the "
            "factor sqrt(1 - rho'rho)"
            if not tobit
            else "none (same scale as maximum likelihood)"
        )
    else:
        fit = _fit_mle(d, tobit, ll, ul, se_kind)
        rf_names = [f"{e}:{v}" for e in d.endog for v in d.x_names]
        names = struct_names + rf_names + fit["anc_names"]
        b, V = fit["b"], fit["V"]
        # Wald test that every correlation between u and v is zero.
        anc0 = d.kz + d.p * d.kx
        idx = [anc0 + fit["anc_names"].index(f"/athrho{j + 2}_1") for j in range(d.p)]
        bb = b[idx]
        chi2_exog = float(bb @ np.linalg.solve(V[np.ix_(idx, idx)], bb))
        info.update(
            log_likelihood=fit["ll"],
            ll=fit["ll"],
            converged=fit["converged"],
            gradient_norm=fit["grad_norm"],
            aic=-2.0 * fit["ll"] + 2.0 * len(b),
            bic=-2.0 * fit["ll"] + np.log(d.n) * len(b),
        )
        for j, e in enumerate(d.endog):
            info[f"rho_{e}"] = float(np.tanh(b[idx[j]]))
            info[f"sigma_{e}"] = float(
                np.exp(b[anc0 + fit["anc_names"].index(f"/lnsigma{j + 2}")])
            )
        if tobit:
            info["sigma"] = float(np.exp(b[anc0 + fit["anc_names"].index("/lnsigma1")]))
        if se_kind == "cluster" and d.clusters is not None:
            info["n_clusters"] = int(pd.unique(d.clusters).size)
    p_exog = float(stats.chi2.sf(chi2_exog, d.p))
    # Wald test that every slope of the outcome equation is zero.
    k_slope = d.kz - 1
    bs = b[:k_slope]
    wald = float(bs @ np.linalg.solve(V[:k_slope, :k_slope], bs))
    info.update(
        exogeneity_chi2=chi2_exog,
        exogeneity_df=d.p,
        exogeneity_pvalue=p_exog,
        wald_chi2=wald,
        wald_df=k_slope,
        wald_pvalue=float(stats.chi2.sf(wald, k_slope)),
    )
    se = se_from_vcov(V)
    params = pd.Series(b, index=names)
    std_errors = pd.Series(se, index=names)
    return EconometricResults(
        params=params,
        std_errors=std_errors,
        model_info=info,
        data_info={
            "nobs": d.n,
            "df_model": k_slope,
            "df_resid": d.n - len(b),
            "dependent_var": y,
            "var_cov": V,
            "var_names": names,
            "inference": "z",
        },
        diagnostics={
            "Wald test of exogeneity (chi2)": chi2_exog,
            "Prob > chi2 (exogeneity)": p_exog,
            "Wald chi2": wald,
            **({"Log-Likelihood": info["log_likelihood"]} if method == "mle" else {}),
        },
    )


def _resolve(
    function: str, method: str, vce: Any, cluster: Any
) -> Tuple[str, str, Optional[str]]:
    from ..core._vcov_spec import parse_se_request

    m = str(method).lower()
    m = {"ml": "mle", "two-step": "twostep", "2step": "twostep"}.get(m, m)
    if m not in ("mle", "twostep"):
        raise MethodIncompatibility(
            f"{function}: method={method!r} is not available; use 'mle' or "
            "'twostep'."
        )
    req = parse_se_request(
        vce,
        cluster,
        function=function,
        supported=("nonrobust", "robust", "cluster"),
    )
    if m == "twostep" and req.kind != "nonrobust":
        raise MethodIncompatibility(
            f"{function}: method='twostep' has one covariance matrix, the "
            "minimum chi-squared one; robust and cluster variants are defined "
            "for method='mle' only (as in Stata)."
        )
    return m, req.kind, req.cluster


@accepts_aliases(robust="vce", covariates="x")
def ivprobit(
    data: pd.DataFrame,
    y: str,
    x: Union[str, Sequence[str], None] = None,
    endog: Union[str, Sequence[str], None] = None,
    instruments: Union[str, Sequence[str], None] = None,
    method: str = "mle",
    vce: Optional[str] = None,
    cluster: Optional[str] = None,
    alpha: float = 0.05,
) -> EconometricResults:
    """
    Probit with continuous endogenous regressors.

    Equivalent to Stata's ``ivprobit y x (endog = instruments)``, with
    ``method='twostep'`` for ``ivprobit ..., twostep``.

    Parameters
    ----------
    data : pd.DataFrame
    y : str
        Binary outcome (0 / 1).
    x : str or list of str, optional
        Exogenous regressors of the outcome equation. A constant is added.
    endog : str or list of str
        Continuous endogenous regressors.
    instruments : str or list of str
        Excluded instruments, at least as many as ``endog``.
    method : {'mle', 'twostep'}, default 'mle'
        ``'mle'`` maximises the joint likelihood of the outcome and the
        reduced forms. ``'twostep'`` is Newey's minimum chi-squared
        estimator; its coefficients are scaled by ``Var(u | v) = 1``
        instead of ``Var(u) = 1``, so they are larger in magnitude than
        the maximum likelihood ones by ``1 / sqrt(1 - rho'rho)``.
    vce : {None, 'robust', 'cluster'}, optional
        Covariance of the maximum likelihood estimator: observed
        information (default), sandwich, or cluster sandwich
        (``vce='cluster firm'`` or ``cluster='firm'``).
    cluster : str, optional
    alpha : float, default 0.05

    Returns
    -------
    EconometricResults
        ``params`` holds the outcome equation (``y:var``), then for
        ``'mle'`` the reduced forms (``endog:var``) and Stata's ancillary
        parameters ``/athrho2_1``, ``/lnsigma2``, ... ``model_info``
        carries the Wald test of exogeneity (``exogeneity_chi2``,
        ``exogeneity_pvalue``), the log-likelihood and, per endogenous
        regressor, ``rho_<name>`` and ``sigma_<name>``.

    Examples
    --------
    >>> import statspai as sp
    >>> res = sp.ivprobit(df, y="works", x=["age", "kids"],
    ...                   endog="other_income", instruments=["husband_educ"])
    >>> print(res.summary())  # doctest: +SKIP
    >>> res.model_info["exogeneity_pvalue"]  # doctest: +SKIP

    Notes
    -----
    A discrete endogenous regressor does not fit this model: the reduced
    form is a linear regression with a normal error. For a binary
    endogenous regressor use :func:`biprobit` (recursive bivariate
    probit).

    References
    ----------
    [@newey1987efficient], [@rivers1988limited]
    """
    m, se_kind, cl = _resolve("ivprobit", method, vce, cluster)
    d = _Design(data, y, x, endog, instruments, cl, "ivprobit")
    values = np.unique(d.Y1)
    if not np.all(np.isin(values, (0.0, 1.0))) or values.size < 2:
        raise MethodIncompatibility(
            f"ivprobit: y={y!r} must be binary 0/1 with both values present; "
            f"found {values[:6].tolist()}.",
            diagnostics={"values": values[:10].tolist()},
        )
    return _assemble(d, "ivprobit", m, False, None, None, se_kind, cl, alpha)


@accepts_aliases(robust="vce", covariates="x")
def ivtobit(
    data: pd.DataFrame,
    y: str,
    x: Union[str, Sequence[str], None] = None,
    endog: Union[str, Sequence[str], None] = None,
    instruments: Union[str, Sequence[str], None] = None,
    ll: Optional[float] = 0,
    ul: Optional[float] = None,
    method: str = "mle",
    vce: Optional[str] = None,
    cluster: Optional[str] = None,
    alpha: float = 0.05,
) -> EconometricResults:
    """
    Tobit with continuous endogenous regressors.

    Equivalent to Stata's ``ivtobit y x (endog = instruments), ll(0)``,
    with ``method='twostep'`` for ``ivtobit ..., twostep``.

    Parameters
    ----------
    data : pd.DataFrame
    y : str
        Censored outcome.
    x : str or list of str, optional
        Exogenous regressors of the outcome equation. A constant is added.
    endog : str or list of str
        Continuous endogenous regressors.
    instruments : str or list of str
        Excluded instruments, at least as many as ``endog``.
    ll, ul : float, optional
        Lower and upper censoring limits. Observations with ``y <= ll``
        (``y >= ul``) are treated as censored. ``ll=0`` by default, as in
        :func:`tobit`; pass ``None`` for no limit on that side.
    method : {'mle', 'twostep'}, default 'mle'
        ``'twostep'`` is Newey's minimum chi-squared estimator.
    vce : {None, 'robust', 'cluster'}, optional
        Covariance of the maximum likelihood estimator.
    cluster : str, optional
    alpha : float, default 0.05

    Returns
    -------
    EconometricResults
        ``params`` holds the outcome equation (``y:var``), then for
        ``'mle'`` the reduced forms and ``/athrho2_1``, ``/lnsigma1``,
        ``/lnsigma2``, ... ``model_info`` carries the Wald test of
        exogeneity and the censoring counts.

    Examples
    --------
    >>> import statspai as sp
    >>> res = sp.ivtobit(df, y="hours", x=["age", "kids"],
    ...                  endog="wage", instruments=["experience"], ll=0)
    >>> print(res.summary())  # doctest: +SKIP

    References
    ----------
    [@newey1987efficient], [@smith1986exogeneity]
    """
    m, se_kind, cl = _resolve("ivtobit", method, vce, cluster)
    d = _Design(data, y, x, endog, instruments, cl, "ivtobit")
    if ll is None and ul is None:
        raise MethodIncompatibility(
            "ivtobit: both ll and ul are None, so nothing is censored; that "
            "model is linear IV (sp.ivreg)."
        )
    lo, hi = _censor_masks(d.Y1, ll, ul)
    if int((~lo & ~hi).sum()) <= d.kz + d.p:
        raise DataInsufficient("ivtobit: not enough uncensored observations.")
    return _assemble(d, "ivtobit", m, True, ll, ul, se_kind, cl, alpha)


# Citation: both estimators report Newey's (1987) two-step and the joint
# likelihood he takes as the benchmark. Mirrors paper.bib.
_NEWEY_1987 = (
    "@article{newey1987efficient,\n"
    "  title={Efficient Estimation of Limited Dependent Variable Models with "
    "Endogenous Explanatory Variables},\n"
    "  author={Newey, Whitney K.},\n"
    "  journal={Journal of Econometrics},\n"
    "  volume={36},\n"
    "  number={3},\n"
    "  pages={231--250},\n"
    "  year={1987},\n"
    "  doi={10.1016/0304-4076(87)90001-7}\n"
    "}"
)
CausalResult._CITATIONS["ivprobit"] = _NEWEY_1987
CausalResult._CITATIONS["ivtobit"] = _NEWEY_1987
