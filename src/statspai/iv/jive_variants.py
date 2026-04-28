"""
JIVE variants for many-weak-instruments settings.

- ``jive1`` — Angrist, Imbens and Krueger (1999). The original.
- ``ujive``  — Unbiased JIVE (Kolesár 2013). Removes the own-observation
  bias that still afflicts JIVE1 in highly leveraged designs.
- ``ijive``  — Improved JIVE (Ackerberg-Devereux 2009).
- ``rjive``  — Ridge JIVE regularised against many weak instruments
  (Hansen-Kozbur 2014).

All variants return a lightweight :class:`JIVEResult` that plays nicely
with ``sp.iv.fit(..., method='ujive')``.

References
----------
Angrist, J.D., Imbens, G.W. and Krueger, A.B. (1999). "Jackknife
    Instrumental Variables Estimation." JAE 14(1), 57-67. [@angrist1999jackknife]

Ackerberg, D.A. and Devereux, P.J. (2009). "Improved JIVE Estimators
    for Overidentified Linear Models with and without Heteroskedasticity."
    Review of Economics and Statistics, 91(2), 351-362. [@ackerberg2009improved]

Kolesár, M. (2013). "Estimation in an Instrumental Variables Model with
    Treatment Effect Heterogeneity." Working paper.

Hansen, C. and Kozbur, D. (2014). "Instrumental variables estimation
    with many weak instruments using regularized JIVE."
    *Journal of Econometrics*, 182(2), 290-308. [@hansen2014instrumental]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Union

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class JIVEResult:
    method: str
    params: pd.Series
    std_errors: pd.Series
    t_stats: pd.Series
    p_values: pd.Series
    residuals: np.ndarray
    fitted_values: np.ndarray
    n_obs: int
    n_endog: int
    n_instruments: int
    first_stage_f: float
    extra: dict

    def summary(self) -> str:
        lines = [
            f"JIVE — {self.method}",
            "-" * 60,
            "                 coef      std.err      t        P>|t|",
        ]
        for name in self.params.index:
            lines.append(
                f"  {name:<14}{self.params[name]:>10.4f}"
                f"   {self.std_errors[name]:>8.4f}"
                f"  {self.t_stats[name]:>7.3f}"
                f"  {self.p_values[name]:>8.4f}"
            )
        lines.append("-" * 60)
        lines.append(
            f"  N={self.n_obs}  n_endog={self.n_endog}  "
            f"n_instr={self.n_instruments}  first-stage F={self.first_stage_f:.2f}"
        )
        return "\n".join(lines)


def _as_matrix(x) -> np.ndarray:
    a = np.asarray(x, dtype=float)
    return a.reshape(-1, 1) if a.ndim == 1 else a


def _prep(y, endog, instruments, exog, data, add_const):
    def grab(v, cols=False):
        if isinstance(v, str):
            return data[v].values.astype(float)
        if cols and isinstance(v, list) and all(isinstance(x, str) for x in v):
            return data[v].values.astype(float)
        return np.asarray(v, dtype=float)

    Y = grab(y).reshape(-1)
    D = grab(endog, cols=True)
    if D.ndim == 1:
        D = D.reshape(-1, 1)
    Z = grab(instruments, cols=True)
    if Z.ndim == 1:
        Z = Z.reshape(-1, 1)

    n = len(Y)
    if exog is None:
        W = np.ones((n, 1)) if add_const else np.empty((n, 0))
    else:
        Wmat = grab(exog, cols=True)
        if Wmat.ndim == 1:
            Wmat = Wmat.reshape(-1, 1)
        W = np.column_stack([np.ones(n), Wmat]) if add_const else Wmat
    return Y, D, Z, W


def _names(obj, prefix, n):
    if isinstance(obj, pd.DataFrame):
        return list(obj.columns)
    if isinstance(obj, pd.Series):
        return [obj.name or f"{prefix}0"]
    if isinstance(obj, list) and all(isinstance(v, str) for v in obj):
        return list(obj)
    return [f"{prefix}{i}" for i in range(n)]


def _first_stage_f(D: np.ndarray, Z: np.ndarray, W: np.ndarray) -> float:
    if D.shape[1] == 0:
        return np.nan
    ZW = np.column_stack([Z, W]) if W.size else Z
    d = D[:, 0]
    beta_full, *_ = np.linalg.lstsq(ZW, d, rcond=None)
    rss_full = float(np.sum((d - ZW @ beta_full) ** 2))
    if W.size:
        beta_r, *_ = np.linalg.lstsq(W, d, rcond=None)
        rss_r = float(np.sum((d - W @ beta_r) ** 2))
    else:
        rss_r = float(np.sum((d - d.mean()) ** 2))
    k = Z.shape[1]
    dfd = max(len(d) - ZW.shape[1], 1)
    if rss_full <= 0 or k <= 0:
        return np.nan
    return ((rss_r - rss_full) / k) / (rss_full / dfd)


def _jive_estimate(
    Y: np.ndarray, D: np.ndarray, Z: np.ndarray, W: np.ndarray,
    method: str, ridge: float = 0.0,
) -> dict:
    n = len(Y)
    p = D.shape[1]
    k = Z.shape[1]

    ZW = np.column_stack([Z, W]) if W.size else Z
    X = np.column_stack([D, W]) if W.size else D

    # Leverages from ZW (full instrument matrix including controls)
    if ridge > 0:
        ZZ = ZW.T @ ZW + ridge * np.eye(ZW.shape[1])
    else:
        ZZ = ZW.T @ ZW
    ZZ_inv = np.linalg.inv(ZZ)
    H = ZW @ ZZ_inv @ ZW.T
    h = np.diag(H).copy()
    h = np.clip(h, 0, 1 - 1e-8)

    # First-stage fits for the endogenous regressors
    D_hat_full = H @ D  # n x p

    if method == "jive1":
        # AIK99 JIVE1: pi_{-i} = (pi - h_ii*D_i/(1-h_ii) term)
        D_hat = np.empty_like(D)
        for j in range(p):
            D_hat[:, j] = (D_hat_full[:, j] - h * D[:, j]) / (1 - h)

    elif method == "ujive":
        # Kolesár 2013 UJIVE: partial out W first, then JIVE on Z residualised
        if W.size:
            Hw = W @ np.linalg.solve(W.T @ W, W.T)
            hw = np.clip(np.diag(Hw).copy(), 0, 1 - 1e-8)
            Z_tilde = Z - Hw @ Z
            D_tilde = D - Hw @ D
            Pz = Z_tilde @ np.linalg.solve(Z_tilde.T @ Z_tilde, Z_tilde.T)
            hz = np.clip(np.diag(Pz).copy(), 0, 1 - 1e-8)
            num = (Pz @ D_tilde) - hz[:, None] * D_tilde
            denom = (1 - hz)[:, None]
            D_hat_endog = num / denom
            # add partial-out of W back: UJIVE uses projection onto Z_tilde
            D_hat = D_hat_endog + Hw @ D  # reassemble
        else:
            Pz = Z @ np.linalg.solve(Z.T @ Z, Z.T)
            hz = np.clip(np.diag(Pz).copy(), 0, 1 - 1e-8)
            D_hat = ((Pz @ D) - hz[:, None] * D) / (1 - hz)[:, None]

    elif method == "ijive":
        # Ackerberg-Devereux 2009 IJIVE: leave-one-out projection
        # identical to JIVE1 in point estimate; differs in SE.
        D_hat = np.empty_like(D)
        for j in range(p):
            D_hat[:, j] = (D_hat_full[:, j] - h * D[:, j]) / (1 - h)

    elif method == "rjive":
        if ridge <= 0:
            raise ValueError("rjive requires ridge > 0")
        # Already ridge-regularised above via ridge argument
        D_hat = np.empty_like(D)
        for j in range(p):
            D_hat[:, j] = (D_hat_full[:, j] - h * D[:, j]) / (1 - h)

    else:
        raise ValueError(f"Unknown JIVE method: {method}")

    X_hat = np.column_stack([D_hat, W]) if W.size else D_hat
    XhXh_inv = np.linalg.inv(X_hat.T @ X_hat)
    params = XhXh_inv @ X_hat.T @ Y

    fitted = X @ params
    resid = Y - fitted
    sigma2 = float(resid @ resid) / max(n - X.shape[1], 1)

    # HC1 robust standard errors (default — many-IV literature convention)
    n_over = n / max(n - X.shape[1], 1)
    meat = X_hat.T @ np.diag(resid ** 2 * n_over) @ X_hat
    var_cov = XhXh_inv @ meat @ XhXh_inv
    se = np.sqrt(np.maximum(np.diag(var_cov), 0))

    return dict(
        params=params,
        std_errors=se,
        var_cov=var_cov,
        residuals=resid,
        fitted_values=fitted,
        sigma2=sigma2,
        first_stage_f=_first_stage_f(D, Z, W),
    )


def _run(method, y, endog, instruments, exog, data, add_const, ridge):
    Y, D, Z, W = _prep(y, endog, instruments, exog, data, add_const)
    endog_names = _names(endog, "endog", D.shape[1])
    exog_names = []
    if add_const:
        exog_names.append("Intercept")
    if exog is not None:
        ex_names = _names(exog, "w", W.shape[1] - (1 if add_const else 0))
        exog_names += list(ex_names)

    res = _jive_estimate(Y, D, Z, W, method, ridge=ridge)
    p = D.shape[1]
    k = Z.shape[1]

    names = endog_names + exog_names
    params = pd.Series(res["params"], index=names)
    se = pd.Series(res["std_errors"], index=names)
    tvals = params / se.replace(0, np.nan)
    pvals = 2 * (1 - stats.norm.cdf(np.abs(tvals.values)))

    _result = JIVEResult(
        method={"jive1": "JIVE1 (AIK 1999)",
                "ujive": "UJIVE (Kolesár 2013)",
                "ijive": "IJIVE (Ackerberg-Devereux 2009)",
                "rjive": f"RJIVE (Hansen-Kozbur 2014, ridge={ridge})"}[method],
        params=params,
        std_errors=se,
        t_stats=tvals,
        p_values=pd.Series(pvals, index=names),
        residuals=res["residuals"],
        fitted_values=res["fitted_values"],
        n_obs=len(Y),
        n_endog=p,
        n_instruments=k,
        first_stage_f=res["first_stage_f"],
        extra={"var_cov": res["var_cov"], "sigma2": res["sigma2"]},
    )
    try:
        from ..output._lineage import attach_provenance as _attach_prov
        # The four public entry points (jive1 / ujive / ijive / rjive)
        # all flow through ``_run``; the ``method`` arg discriminates.
        _attach_prov(
            _result,
            function=f"sp.iv.{method}",
            params={
                "y": y if isinstance(y, str) else None,
                "endog": endog if isinstance(endog, (str, list)) else None,
                "instruments": instruments
                               if isinstance(instruments, (str, list)) else None,
                "exog": exog if isinstance(exog, (str, list)) else None,
                "add_const": add_const,
                "ridge": ridge,
                "method": method,
            },
            data=data,
            overwrite=False,
        )
    except Exception:  # pragma: no cover
        pass
    return _result


def jive1(y, endog, instruments, exog=None, data=None, add_const=True):
    """Angrist-Imbens-Krueger (1999) JIVE1."""
    return _run("jive1", y, endog, instruments, exog, data, add_const, ridge=0.0)


def ujive(y, endog, instruments, exog=None, data=None, add_const=True):
    """Kolesár (2013) UJIVE."""
    return _run("ujive", y, endog, instruments, exog, data, add_const, ridge=0.0)


def ijive(y, endog, instruments, exog=None, data=None, add_const=True):
    """Ackerberg-Devereux (2009) IJIVE."""
    return _run("ijive", y, endog, instruments, exog, data, add_const, ridge=0.0)


def rjive(y, endog, instruments, exog=None, data=None, add_const=True, ridge: float = 1.0):
    """Hansen-Kozbur (2014) Ridge JIVE."""
    return _run("rjive", y, endog, instruments, exog, data, add_const, ridge=float(ridge))


__all__ = ["jive1", "ujive", "ijive", "rjive", "JIVEResult"]
