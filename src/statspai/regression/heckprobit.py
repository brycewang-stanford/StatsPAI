"""Probit with sample selection (Stata ``heckprobit``).

A binary outcome ``y = 1(x'b + e1 > 0)`` is observed only when
``s = 1(z'g + e2 > 0)``, with ``(e1, e2)`` standard bivariate normal with
correlation ``rho``. The per-observation log-likelihood is

* not selected: ``ln Phi(-z'g)``;
* selected, ``y = 1``: ``ln Phi2(x'b, z'g; rho)``;
* selected, ``y = 0``: ``ln Phi2(-x'b, z'g; -rho)``,

maximised over ``theta = (b, g, atanh rho)`` -- Stata's parameterisation, so
the coefficient vector and its covariance are ``e(b)`` / ``e(V)``. The fit
runs BFGS from the two separate probits and finishes with Newton steps on
complex-step scores (the shared ML path in ``_optim_helpers``); standard
errors are ``vce(oim)`` / ``vce(robust)`` / ``vce(cluster)`` through
:func:`statspai.core._vcov.ml_vcov`.
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
from ..exceptions import DataInsufficient, MethodIncompatibility
from ._bivariate_normal import binormal_cdf
from ._limited_dep_result import LimitedDepResult
from ._optim_helpers import (
    complex_step_scores,
    inverse_information,
    ml_newton_polish,
    require_two_clusters,
    robust_convergence,
    se_from_vcov,
)

__all__ = ["heckprobit"]


def _probit(y: np.ndarray, X: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Weighted probit coefficients (Newton), the starting values."""
    b = np.zeros(X.shape[1])
    for _ in range(100):
        xb = X @ b
        q = 2.0 * y - 1.0
        lam = q * np.exp(stats.norm.logpdf(q * xb) - special.log_ndtr(q * xb))
        g = X.T @ (w * lam)
        H = -(X * (w * lam * (lam + xb))[:, None]).T @ X
        try:
            step = np.linalg.solve(H, g)
        except np.linalg.LinAlgError:
            break
        b = b - step
        if np.max(np.abs(step)) < 1e-10:
            break
    return b


@records_call("heckprobit")
@accepts_aliases(robust="vce")
def heckprobit(
    data: pd.DataFrame,
    y: str,
    x: List[str],
    select: Optional[str] = None,
    z: Optional[List[str]] = None,
    vce: Optional[str] = None,
    cluster: Optional[str] = None,
    weights: Optional[str] = None,
    alpha: float = 0.05,
) -> CausalResult:
    """
    Probit model with sample selection, by maximum likelihood.

    Equivalent to Stata's ``heckprobit y x, select(s = z)``: a binary
    outcome observed only for a selected subsample, with errors of the
    outcome and selection equations jointly normal with correlation
    ``rho``. With ``rho = 0`` the outcome equation is a probit on the
    selected rows; otherwise that probit is inconsistent.

    Parameters
    ----------
    data : pd.DataFrame
    y : str
        Binary outcome (0 / 1), observed when ``select = 1``.
    x : list of str
        Regressors of the outcome equation (a constant is added).
    select : str, optional
        Binary selection indicator. When omitted, a row is selected when
        ``y`` is not missing (Stata ``select(z1 z2)``).
    z : list of str
        Variables of the selection equation (a constant is added); include
        at least one that is not in ``x``.
    vce : {None, 'oim', 'robust', 'cluster'}, optional
        Observed information (default), sandwich, or cluster sandwich
        (``vce='cluster firm'`` or ``cluster='firm'``).
    cluster : str, optional
        Cluster column.
    weights : str, optional
        Sampling weights (``[pw=]``), which imply robust standard errors as
        in Stata.
    alpha : float, default 0.05

    Returns
    -------
    CausalResult
        ``params`` holds the outcome equation (``const``, ``x``...), the
        selection equation (``select:const``, ``select:z``...), ``athrho``
        and the derived ``rho``. ``model_info`` carries the
        log-likelihood, the Wald test of ``rho = 0`` (on ``athrho``, as
        Stata reports it) and, with the default variance, the likelihood
        ratio test of independent equations. ``sp.margins`` reports
        Stata's ``predict`` statistics (``pmargin`` by default, ``p11``,
        ``pcond``, ``psel`` ...).

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> n = 1000
    >>> df = pd.DataFrame({"x": rng.normal(size=n), "z": rng.normal(size=n)})
    >>> e = rng.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], size=n)
    >>> df["s"] = (0.2 + df["z"] + 0.5 * df["x"] + e[:, 1] > 0).astype(int)
    >>> df["y"] = np.where(df["s"] == 1, (0.3 + df["x"] + e[:, 0] > 0), np.nan)
    >>> fit = sp.heckprobit(df, y="y", x=["x"], select="s", z=["z", "x"])
    >>> round(fit.model_info["rho"], 1)  # doctest: +SKIP
    0.5

    References
    ----------
    vandeven1981demand
    """
    from ..core._vcov_spec import parse_se_request

    se_req = parse_se_request(
        vce,
        cluster,
        function="heckprobit",
        supported=("nonrobust", "robust", "cluster"),
    )
    se_kind, cluster = se_req.kind, se_req.cluster
    if weights is not None and se_kind == "nonrobust":
        se_kind = "robust"  # Stata: pweights imply vce(robust)
    if not z:
        raise MethodIncompatibility(
            "heckprobit: z= lists the variables of the selection equation.",
            recovery_hint="Pass z=[...]; include at least one variable that "
            "is not in x (an exclusion restriction).",
        )
    x = [x] if isinstance(x, str) else list(x)
    z = [z] if isinstance(z, str) else list(z)
    from .._input_validation import check_columns

    check_columns(
        data,
        {
            "y": y,
            "select": select,
            "x": x,
            "z": z,
            "cluster": cluster,
            "weights": weights,
        },
        function="heckprobit",
    )
    user_select = select
    if select is None:
        select = "_selected"
        while select in data.columns:
            select = "_" + select
        data = data.assign(**{select: data[y].notna().astype(float)})

    extra = [c for c in (cluster, weights) if c is not None]
    need_all = list(dict.fromkeys([select] + z + extra))
    df = data.loc[data[need_all].notna().all(axis=1)]
    d_all = df[select].to_numpy(dtype=float)
    if not np.isin(d_all, (0.0, 1.0)).all():
        raise MethodIncompatibility(
            f"heckprobit: the selection indicator {select!r} must be 0/1.",
            diagnostics={"select": select},
        )
    sel_ok = df[[y] + x].notna().all(axis=1).to_numpy()
    df = df.loc[(d_all == 0) | sel_ok]
    D = df[select].to_numpy(dtype=float)
    sel = D == 1
    n = len(df)
    Yv = np.where(sel, df[y].to_numpy(dtype=float, na_value=0.0), 0.0)
    if not np.isin(Yv[sel], (0.0, 1.0)).all():
        raise MethodIncompatibility(
            f"heckprobit: the outcome {y!r} must be 0/1 on the selected rows.",
            recovery_hint="For a continuous outcome use sp.heckman.",
        )
    n_sel = int(sel.sum())
    if n_sel == 0 or n_sel == n:
        raise DataInsufficient(
            "heckprobit: the selection indicator must vary (some rows "
            "selected, some not).",
            diagnostics={"n_selected": n_sel, "n": n},
        )
    n_pos = int(Yv[sel].sum())
    if n_pos in (0, n_sel):
        # Stata: "outcome does not vary". The likelihood has no maximum; the
        # optimiser used to stop at a 17.4 coefficient with SE 3e30 and
        # report convergence (2026-10-20 probe).
        raise DataInsufficient(
            f"heckprobit: the outcome {y!r} is {int(Yv[sel][0])} on every "
            "selected row; it must vary.",
            recovery_hint="Check the outcome and selection coding; with an "
            "outcome that does not vary there is nothing to estimate.",
            diagnostics={"n_selected": n_sel, "n_outcome_1": n_pos},
        )
    Z = np.column_stack([np.ones(n)] + [df[v].to_numpy(dtype=float) for v in z])
    X = np.column_stack(
        [np.ones(n)] + [df[v].to_numpy(dtype=float, na_value=0.0) for v in x]
    )
    X[~sel] = 0.0
    kx, kz = X.shape[1], Z.shape[1]
    wt = np.ones(n)
    if weights is not None:
        wt = df[weights].to_numpy(dtype=float)
        if not np.all(np.isfinite(wt)) or np.any(wt <= 0):
            raise MethodIncompatibility(
                "heckprobit: weights must be finite and strictly positive.",
                diagnostics={"weights": weights},
            )
        wt = wt * (n / wt.sum())
    q = np.where(sel, 2.0 * Yv - 1.0, 0.0)
    q_sel = q[sel]

    def obs_loglik(theta: np.ndarray) -> np.ndarray:
        rho = np.tanh(theta[kx + kz])
        zg = Z @ theta[kx : kx + kz]
        out = np.zeros(n, dtype=np.result_type(theta, float))
        xb = X[sel] @ theta[:kx]
        out[sel] = np.log(binormal_cdf(q_sel * xb, zg[sel], q_sel * rho))
        out[~sel] = special.log_ndtr(-zg[~sel])
        return wt * out

    def neg_total(theta: np.ndarray) -> float:
        with np.errstate(all="ignore"):
            v = -float(np.sum(obs_loglik(theta)))
        return v if np.isfinite(v) else 1e300

    def neg_grad(theta: np.ndarray) -> np.ndarray:
        grad: np.ndarray = -complex_step_scores(obs_loglik, theta).sum(axis=0)
        return grad

    gamma0 = _probit(D, Z, wt)
    beta0 = _probit(Yv[sel], X[sel], wt[sel])
    theta0 = np.concatenate([beta0, gamma0, [0.0]])
    opt = optimize.minimize(
        neg_total, theta0, jac=neg_grad, method="BFGS", options={"maxiter": 2000}
    )
    converged, _ = robust_convergence(opt)
    theta, scores, H, _ = ml_newton_polish(obs_loglik, opt.x)
    grad_norm = float(np.max(np.abs(scores.sum(axis=0))))
    converged = bool(converged or grad_norm < 1e-6)

    clusters = df[cluster].to_numpy() if se_kind == "cluster" else None
    if clusters is not None:
        require_two_clusters(clusters, "heckprobit")
    V = ml_vcov(
        inverse_information(H),
        scores if se_kind != "nonrobust" else None,
        kind=se_kind,
        clusters=clusters,
    )
    se = se_from_vcov(V)
    t_idx = kx + kz
    athrho = float(theta[t_idx])
    rho = float(np.tanh(athrho))
    se_rho = float((1.0 - rho**2) * se[t_idx])
    loglik = float(np.sum(obs_loglik(theta)))

    names = ["const"] + x + ["select:const"] + [f"select:{v}" for v in z] + ["athrho"]
    coef = np.append(theta, rho)
    ses = np.append(se, se_rho)
    with np.errstate(divide="ignore", invalid="ignore"):
        zstat = coef / ses
    zstat[-1] = np.nan  # rho is reported, tested through athrho
    pvals = 2.0 * stats.norm.sf(np.abs(zstat))
    detail = pd.DataFrame(
        {
            "variable": names + ["rho"],
            "coefficient": coef,
            "se": ses,
            "z": zstat,
            "pvalue": pvals,
        }
    )
    wald = float((athrho / se[t_idx]) ** 2) if se[t_idx] > 0 else np.nan
    model_info: Dict[str, Any] = {
        "method": "Heckman probit (ML)",
        "n_total": n,
        "n_selected": n_sel,
        "n_censored": n - n_sel,
        "selection_vars": z,
        "rho": rho,
        "log_likelihood": loglik,
        "wald_rho0": wald,
        "wald_rho0_pvalue": (
            float(stats.chi2.sf(wald, 1)) if np.isfinite(wald) else np.nan
        ),
        "converged": converged,
        "gradient_norm": grad_norm,
        "vce": se_kind,
        "cluster": cluster if se_kind == "cluster" else None,
        "n_clusters": (
            int(pd.Series(clusters).nunique()) if clusters is not None else None
        ),
        "weights": weights,
    }
    if se_kind == "nonrobust" and weights is None:
        # LR test of independent equations: the joint fit against the
        # selection probit plus the outcome probit on the selected rows.
        def _probit_ll(yy: np.ndarray, XX: np.ndarray, b: np.ndarray) -> float:
            return float(np.sum(special.log_ndtr((2.0 * yy - 1.0) * (XX @ b))))

        ll0 = _probit_ll(D, Z, gamma0) + _probit_ll(Yv[sel], X[sel], beta0)
        lr = max(2.0 * (loglik - ll0), 0.0)
        model_info["lr_rho0"] = lr
        model_info["lr_rho0_pvalue"] = float(stats.chi2.sf(lr, 1))

    z_crit = stats.norm.ppf(1 - alpha / 2)
    main, main_se = float(theta[1]) if kx > 1 else float(theta[0]), float(
        se[1] if kx > 1 else se[0]
    )
    res = LimitedDepResult(
        method="Probit with sample selection (ML)",
        estimand=f"beta_{x[0]}" if x else "beta_const",
        estimate=main,
        se=main_se,
        pvalue=float(2 * stats.norm.sf(abs(main / main_se))) if main_se > 0 else np.nan,
        ci=(main - z_crit * main_se, main + z_crit * main_se),
        alpha=alpha,
        n_obs=n,
        detail=detail,
        model_info=model_info,
        _citation_key="heckprobit",
    )
    res._ml_state = {
        "model": "heckprobit",
        "theta": np.asarray(theta, dtype=float),
        "vcov": np.asarray(V, dtype=float),
        "out_terms": ["const"] + x,
        "sel_terms": ["const"] + z,
        "select": user_select,
        "y": y,
        "nobs": n,
    }
    return res
