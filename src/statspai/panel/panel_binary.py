"""
Panel binary-choice models: logit and probit for panel data.

Estimators
----------
- ``panel_logit``  method='fe'  — Conditional FE logit (Chamberlain 1980)
- ``panel_logit``  method='re'  — Random Effects logit (Gauss-Hermite quadrature)
- ``panel_logit``  method='cre' — Correlated Random Effects (Mundlak) logit
- ``panel_probit`` method='re'  — Random Effects probit
- ``panel_probit`` method='cre' — Correlated Random Effects probit

References
----------
Chamberlain, G. (1980). "Analysis of Covariance with Qualitative Data."
Mundlak, Y. (1978). "On the Pooling of Time Series and Cross Section Data."
Wooldridge, J.M. (2010). Econometric Analysis of Cross Section and Panel Data.
[@chamberlain1980analysis]
"""

from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import optimize, special, stats

from .._aliases import accepts_aliases
from ..core._vcov_spec import markout_clusters
from ..core.results import EconometricResults

# --------------- helpers ---------------

_logit_cdf = special.expit
_probit_cdf = stats.norm.cdf


def _group_panel(
    df: pd.DataFrame,
    y: str,
    x: List[str],
    id_col: str,
    drop_no_variation: bool = False,
) -> Tuple[List[Any], Dict[Any, np.ndarray], Dict[Any, np.ndarray], int]:
    """Group panel data by unit.  Returns groups, x_groups, y_groups, n_dropped."""
    groups, x_groups, y_groups, n_dropped = [], {}, {}, 0
    for gid, gdf in df.groupby(id_col):
        yi = gdf[y].values.astype(float)
        if drop_no_variation and (yi.sum() == 0 or yi.sum() == len(yi)):
            n_dropped += 1
            continue
        groups.append(gid)
        x_groups[gid] = gdf[x].values.astype(float)
        y_groups[gid] = yi
    return groups, x_groups, y_groups, n_dropped


def _add_mundlak_means(
    data: pd.DataFrame,
    x: List[str],
    id_col: str,
) -> Tuple[pd.DataFrame, List[str]]:
    """Add within-unit means as additional regressors (Mundlak device)."""
    df = data.copy()
    mean_names = [f"{v}_mean" for v in x]
    means = df.groupby(id_col)[x].transform("mean")
    if isinstance(means, pd.Series):
        df[mean_names[0]] = means
    else:
        for orig, mn in zip(x, mean_names):
            df[mn] = means[orig]
    return df, mean_names


# --------------- Conditional FE logit (Chamberlain 1980) ---------------


def _logaddexp_c(a: Any, b: Any) -> Any:
    """``log(exp(a) + exp(b))``: stable, and complex-step safe."""
    m = np.maximum(np.real(a), np.real(b))
    return m + np.log(np.exp(a - m) + np.exp(b - m))


def _log_sum_combinations(scores: np.ndarray, T: int, d: int) -> Any:
    """Log-sum-exp over all d-combinations of scores via DP (complex-step safe)."""
    NEG_INF = -1e30
    dp = np.full((T + 1, d + 1), NEG_INF, dtype=np.result_type(scores, float))
    dp[0, 0] = 0.0
    for j in range(1, T + 1):
        sj = scores[j - 1]
        for k in range(min(j, d) + 1):
            val = dp[j - 1, k]
            if k > 0:
                val = _logaddexp_c(val, dp[j - 1, k - 1] + sj)
            dp[j, k] = val
    return dp[T, d]


def _conditional_logit_group_ll(
    beta: np.ndarray,
    groups: List[Any],
    xg: Dict[Any, np.ndarray],
    yg: Dict[Any, np.ndarray],
) -> np.ndarray:
    """Per-panel conditional log-likelihood for FE logit (complex-step safe)."""
    out = []
    for g in groups:
        xi, yi = xg[g], yg[g]
        di, Ti = int(yi.sum()), len(yi)
        if di == 0 or di == Ti:
            out.append(0.0)  # pragma: no cover - dropped upstream
            continue
        scores = xi @ beta
        out.append(scores[yi == 1].sum() - _log_sum_combinations(scores, Ti, di))
    return np.array(out)


def _conditional_logit_nll(
    beta: np.ndarray,
    groups: List[Any],
    xg: Dict[Any, np.ndarray],
    yg: Dict[Any, np.ndarray],
) -> float:
    """Negative conditional log-likelihood for FE logit."""
    return float(-np.sum(np.real(_conditional_logit_group_ll(beta, groups, xg, yg))))


def _conditional_logit_grad(
    beta: np.ndarray,
    groups: List[Any],
    xg: Dict[Any, np.ndarray],
    yg: Dict[Any, np.ndarray],
) -> np.ndarray:
    """Gradient of conditional negative log-likelihood (forward-backward DP)."""
    NEG_INF = -1e30
    k = len(beta)
    grad = np.zeros(k)
    for g in groups:
        xi, yi = xg[g], yg[g]
        di, Ti = int(yi.sum()), len(yi)
        if di == 0 or di == Ti:
            continue  # pragma: no cover
        scores = xi @ beta
        grad -= xi[yi == 1].sum(axis=0)
        # forward DP
        dp_f = np.full((Ti + 1, di + 1), NEG_INF)
        dp_f[0, 0] = 0.0
        for j in range(1, Ti + 1):
            sj = scores[j - 1]
            for m in range(min(j, di) + 1):
                v = dp_f[j - 1, m]
                if m > 0:
                    v = np.logaddexp(v, dp_f[j - 1, m - 1] + sj)
                dp_f[j, m] = v
        log_denom = dp_f[Ti, di]
        # backward DP
        dp_b = np.full((Ti + 2, di + 1), NEG_INF)
        dp_b[Ti + 1, 0] = 0.0
        for j in range(Ti, 0, -1):
            sj = scores[j - 1]
            for m in range(min(Ti - j + 1, di) + 1):
                v = dp_b[j + 1, m]
                if m > 0:
                    v = np.logaddexp(v, dp_b[j + 1, m - 1] + sj)
                dp_b[j, m] = v
        # marginal inclusion probabilities
        for j in range(Ti):
            for m in range(min(j, di - 1) + 1):
                rem = di - m - 1
                if rem < 0 or rem > Ti - j - 1:
                    continue
                lp = dp_f[j, m] + scores[j] + dp_b[j + 2, rem] - log_denom
                if lp > NEG_INF + 100:
                    grad += np.exp(lp) * xi[j]
    return grad


def _fit_fe_logit(
    data: pd.DataFrame,
    y: str,
    x: List[str],
    id_col: str,
    maxiter: int,
    tol: float,
) -> Tuple[np.ndarray, np.ndarray, float, int, int, int, np.ndarray, bool]:
    """Fit conditional FE logit: BFGS, then exact Newton steps.

    The observed information comes from complex-step derivatives of the
    per-panel conditional log-likelihood; the second-difference Hessian used
    before carried ~2e-6 relative error against Stata's ``xtlogit, fe``.
    """
    from ..regression._optim_helpers import (
        inverse_information,
        ml_newton_polish,
        se_from_vcov,
    )

    df = data[[id_col, y] + x].dropna()
    groups, xg, yg, n_dropped = _group_panel(df, y, x, id_col, drop_no_variation=True)
    n_units = len(groups)
    n_obs = sum(len(yg[g]) for g in groups)
    res = optimize.minimize(
        _conditional_logit_nll,
        np.zeros(len(x)),
        args=(groups, xg, yg),
        jac=_conditional_logit_grad,
        method="BFGS",
        options={"maxiter": maxiter, "gtol": tol},
    )
    beta, scores, H, _ = ml_newton_polish(
        lambda b: _conditional_logit_group_ll(b, groups, xg, yg),
        np.asarray(res.x, dtype=float),
    )
    vcov = inverse_information(H)
    se = se_from_vcov(vcov)
    ll = float(np.sum(np.real(_conditional_logit_group_ll(beta, groups, xg, yg))))
    converged = bool(res.success or np.linalg.norm(scores.sum(axis=0)) < 1e-6)
    return beta, se, ll, n_obs, n_units, n_dropped, vcov, converged


# --------------- RE logit / probit via Gauss-Hermite quadrature ---------------


def _re_panel_group_ll(
    theta: np.ndarray,
    groups: List[Any],
    xg: Dict[Any, np.ndarray],
    yg: Dict[Any, np.ndarray],
    n_quad: int,
    link: str,
) -> np.ndarray:
    """Per-panel RE log-likelihood by Gauss-Hermite quadrature.

    ``theta = [beta..., log_sigma_u]``. Complex-step safe: the complex path
    uses ``log1p(exp(.))`` and the real path the overflow-proof ``logaddexp``.
    """
    beta, sigma_u = theta[:-1], np.exp(theta[-1])
    nodes, weights = np.polynomial.hermite.hermgauss(n_quad)
    log_w = np.log(weights) - 0.5 * np.log(np.pi)
    complex_path = np.iscomplexobj(theta)
    out = []
    for g in groups:
        xi, yi = xg[g], yg[g]
        eta = (xi @ beta)[None, :] + np.sqrt(2.0) * sigma_u * nodes[:, None]  # (Q, T)
        if link == "logit":
            if complex_path:
                log_p, log_q = -np.log1p(np.exp(-eta)), -np.log1p(np.exp(eta))
            else:
                log_p, log_q = -np.logaddexp(0.0, -eta), -np.logaddexp(0.0, eta)
        else:
            log_p, log_q = special.log_ndtr(eta), special.log_ndtr(-eta)
        ll_q = log_w + np.sum(yi * log_p + (1 - yi) * log_q, axis=1)
        m = np.max(np.real(ll_q))
        out.append(m + np.log(np.sum(np.exp(ll_q - m))))
    return np.array(out)


def _link_name(link_cdf: Callable[[Any], Any]) -> str:
    return "logit" if link_cdf is _logit_cdf else "probit"


def _re_panel_nll(
    theta: np.ndarray,
    groups: List[Any],
    xg: Dict[Any, np.ndarray],
    yg: Dict[Any, np.ndarray],
    n_quad: int,
    link_cdf: Callable[[Any], Any],
) -> float:
    """Negative log-likelihood for RE binary panel model.
    theta = [beta..., log_sigma_u]
    """
    group_ll = _re_panel_group_ll(theta, groups, xg, yg, n_quad, _link_name(link_cdf))
    return float(-np.sum(np.real(group_ll)))


def _fit_re_binary(
    data: pd.DataFrame,
    y: str,
    x: List[str],
    id_col: str,
    n_quad: int,
    link_cdf: Callable[[Any], Any],
    maxiter: int,
    tol: float,
    se_kind: str = "nonrobust",
    cluster: Optional[str] = None,
) -> Tuple[np.ndarray, np.ndarray, float, float, float, int, int, np.ndarray, bool]:
    """Fit RE binary panel model via MLE with Gauss-Hermite quadrature.

    ``se_kind`` follows Stata's ``xtlogit`` / ``xtprobit, re``:
    ``'nonrobust'`` is the observed information; ``'robust'`` is
    cluster-robust at the panel level with G/(G-1); ``'cluster'`` clusters
    on the ``cluster`` column, within which panels must nest.
    """
    from ..core._vcov import ml_vcov
    from ..exceptions import MethodIncompatibility
    from ..regression._optim_helpers import (
        inverse_information,
        ml_newton_polish,
        se_from_vcov,
    )

    extra = (
        [cluster]
        if cluster is not None and cluster not in [id_col, y] + list(x)
        else []
    )
    df = data[[id_col, y] + x + extra].dropna()
    # The random-effects likelihood needs an intercept. Conditional FE logit
    # does not -- the constant is differenced out -- and this fitter reused
    # the same grouping helper, which builds the design from `x` alone. So
    # the RE logit and RE probit were fitted with NO constant term, which
    # biases every slope in a nonlinear model. Measured against Stata's
    # `xtprobit, re` on a balanced N=60, T=12 panel: 0.39%, and stubbornly
    # 0.39% at any number of quadrature points, which is what identified it
    # as a design-matrix problem rather than an integration one. On a design
    # whose regressors are not centred the error is unbounded.
    df = df.copy()
    const_name = "_cons"
    while const_name in df.columns:  # pragma: no cover - name collision
        const_name += "_"
    df[const_name] = 1.0
    x_design = [const_name] + list(x)
    groups, xg, yg, _ = _group_panel(df, y, x_design, id_col)
    n_units, n_obs = len(groups), sum(len(yg[g]) for g in groups)
    theta0 = np.zeros(len(x_design) + 1)
    res = optimize.minimize(
        _re_panel_nll,
        theta0,
        args=(groups, xg, yg, n_quad, link_cdf),
        method="BFGS",
        options={"maxiter": maxiter, "gtol": tol},
    )
    link = _link_name(link_cdf)

    def group_ll(t: np.ndarray) -> np.ndarray:
        return _re_panel_group_ll(t, groups, xg, yg, n_quad, link)

    # Exact Newton steps, then the observed information and per-panel scores
    # from complex-step derivatives (the second-difference Hessian used
    # before carried ~1e-5 relative error).
    theta, scores, H, _ = ml_newton_polish(group_ll, np.asarray(res.x, dtype=float))
    beta, sigma_u = theta[:-1], np.exp(theta[-1])

    kind, clusters = se_kind, None
    if se_kind == "robust":
        # Stata xtlogit/xtprobit, re vce(robust): clusters are the panels.
        kind, clusters = "cluster", np.arange(n_units)
    elif se_kind == "cluster":
        per_panel = df.groupby(id_col)[cluster]
        if (per_panel.nunique() > 1).any():
            raise MethodIncompatibility(
                f"panel {id_col!r} spans more than one {cluster!r} cluster; "
                "panels must be nested within clusters.",
                recovery_hint="Cluster on a variable that is constant within panel.",
            )
        first = per_panel.first()
        clusters = np.asarray([first[g] for g in groups])
    vcov_full = ml_vcov(inverse_information(H), scores, kind=kind, clusters=clusters)
    se_full = se_from_vcov(vcov_full)
    se_beta = se_full[:-1]
    se_sigma_u = sigma_u * se_full[-1]  # delta method
    converged = bool(res.success or np.linalg.norm(scores.sum(axis=0)) < 1e-6)
    return (
        beta,
        se_beta,
        float(sigma_u),
        float(se_sigma_u),
        float(np.sum(np.real(group_ll(theta)))),
        n_obs,
        n_units,
        vcov_full[:-1, :-1],
        converged,
    )


# --------------- Result wrappers ---------------


def _wrap_re_result(
    data: pd.DataFrame,
    y: str,
    x_vars: List[str],
    id_col: str,
    n_quad: int,
    link_cdf: Callable[[Any], Any],
    maxiter: int,
    tol: float,
    alpha: float,
    model_name: str,
    method_tag: str,
    link: str = "logit",
    original_x: Optional[List[str]] = None,
    mean_names: Optional[List[str]] = None,
    se_kind: str = "nonrobust",
    cluster: Optional[str] = None,
) -> EconometricResults:
    """Fit RE/CRE binary model and wrap into EconometricResults."""
    beta, se, sigma_u, se_sigma_u, ll, n_obs, n_units, vcov, ok = _fit_re_binary(
        data,
        y,
        x_vars,
        id_col,
        n_quad,
        link_cdf,
        maxiter,
        tol,
        se_kind=se_kind,
        cluster=cluster,
    )
    # _fit_re_binary prepends the constant to the design, so the first
    # coefficient is `_cons`. It is reported last, matching Stata's layout
    # and the rest of this package.
    coef_names = list(x_vars) + ["_cons"]
    order = list(range(1, len(beta))) + [0]
    beta = np.asarray(beta, dtype=float)[order]
    se = np.asarray(se, dtype=float)[order]
    vcov = np.asarray(vcov, dtype=float)[np.ix_(order, order)]
    scale = np.pi**2 / 3 if link == "logit" else 1.0
    rho = sigma_u**2 / (sigma_u**2 + scale)
    n_params = len(coef_names) + 1
    model_info = {
        "model": model_name,
        "method": method_tag,
        "link": link,
        "dep_var": y,
        "converged": ok,
        "log_likelihood": ll,
        "sigma_u": sigma_u,
        "se_sigma_u": se_sigma_u,
        "rho": rho,
        "n_quadrature": n_quad,
        "alpha": alpha,
    }
    if original_x is not None:
        model_info["original_x"] = original_x
        model_info["mundlak_means"] = mean_names
    data_info = {
        "n_obs": n_obs,
        "n_units": n_units,
        "n_vars": len(x_vars),
        "df_resid": n_obs - n_params,
        "alpha": alpha,
        # z / chi2 inference, as Stata's xtlogit / xtprobit, re.
        "inference": "z",
        "var_cov": vcov,
    }
    diagnostics = {
        "aic": -2 * ll + 2 * n_params,
        "bic": -2 * ll + np.log(n_obs) * n_params,
    }
    return EconometricResults(
        pd.Series(beta, index=coef_names),
        pd.Series(se, index=coef_names),
        model_info,
        data_info,
        diagnostics,
    )


# ====================== Public API ======================


@accepts_aliases(vce="robust")
@markout_clusters
def panel_logit(
    data: pd.DataFrame,
    y: str,
    x: List[str],
    id: str = "id",
    time: str = "time",
    method: str = "fe",
    n_quadrature: int = 12,
    robust: str = "nonrobust",
    cluster: Optional[str] = None,
    maxiter: int = 200,
    tol: float = 1e-8,
    alpha: float = 0.05,
) -> EconometricResults:
    """Panel logit model.

    Parameters
    ----------
    data : DataFrame
        Panel data in long format.
    y : str
        Binary dependent variable (0/1).
    x : list of str
        Regressors.
    id, time : str
        Unit and time identifier columns.
    method : str
        'fe' (conditional FE logit), 're' (random effects), 'cre' (Mundlak).
    n_quadrature : int
        Gauss-Hermite quadrature points (RE/CRE only).
    robust : str
        'nonrobust' or 'robust'.
    cluster : str or None
        Column for cluster-robust SEs.
    maxiter : int
        Maximum optimizer iterations.
    tol : float
        Gradient tolerance.
    alpha : float
        Significance level for confidence intervals.

    Returns
    -------
    EconometricResults

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np
    >>> import pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> rows = []
    >>> for i in range(60):
    ...     a = rng.normal()  # unit effect
    ...     for t in range(5):
    ...         xit = rng.normal()
    ...         p = 1.0 / (1.0 + np.exp(-(0.8 * xit + a)))
    ...         rows.append({"id": i, "time": t,
    ...                      "y": int(rng.uniform() < p), "x": xit})
    >>> df = pd.DataFrame(rows)
    >>> res = sp.panel_logit(df, y="y", x=["x"], id="id", time="time", method="fe")
    >>> bool("x" in res.params.index)
    True
    """
    from ..core._vcov_spec import parse_se_request
    from ..exceptions import MethodIncompatibility

    method = method.lower()
    if method not in ("fe", "re", "cre"):
        raise ValueError("method must be 'fe', 're', or 'cre'")
    id_col, x_vars = id, list(x)

    # Stata grammar. robust= and cluster= used to be accepted and ignored.
    # xtlogit, re takes vce(robust) (clusters are the panels) and
    # vce(cluster c); xtlogit, fe takes neither (Stata rc 198).
    se_req = parse_se_request(
        robust,
        cluster,
        function="panel_logit",
        supported=("nonrobust", "robust", "cluster"),
    )
    if method == "fe" and se_req.kind != "nonrobust":
        raise MethodIncompatibility(
            "panel_logit(method='fe') reports observed-information standard "
            f"errors only (as Stata's xtlogit, fe); vce={robust!r} / "
            f"cluster={cluster!r} is not available.",
            recovery_hint=(
                f"Use sp.clogit(formula, data, group={id_col!r}, "
                f"vce='cluster {id_col}') for cluster-robust FE logit SEs."
            ),
        )

    if method == "cre":
        data, mn = _add_mundlak_means(data, x_vars, id_col)
        return _wrap_re_result(
            data,
            y,
            x_vars + mn,
            id_col,
            n_quadrature,
            _logit_cdf,
            maxiter,
            tol,
            alpha,
            "Panel Logit (CRE/Mundlak)",
            "cre",
            original_x=x_vars,
            mean_names=mn,
            se_kind=se_req.kind,
            cluster=se_req.cluster,
        )
    if method == "re":
        return _wrap_re_result(
            data,
            y,
            x_vars,
            id_col,
            n_quadrature,
            _logit_cdf,
            maxiter,
            tol,
            alpha,
            "Panel Logit (RE)",
            "re",
            se_kind=se_req.kind,
            cluster=se_req.cluster,
        )

    # --- FE ---
    beta, se, ll, n_obs, n_units, n_dropped, vcov, ok = _fit_fe_logit(
        data, y, x_vars, id_col, maxiter, tol
    )
    k = len(x_vars)
    return EconometricResults(
        pd.Series(beta, index=x_vars),
        pd.Series(se, index=x_vars),
        model_info={
            "alpha": alpha,
            "model": "Panel Logit (Conditional FE)",
            "method": "fe",
            "link": "logit",
            "dep_var": y,
            "converged": ok,
            "log_likelihood": ll,
            "n_dropped_units": n_dropped,
        },
        data_info={
            "n_obs": n_obs,
            "n_units": n_units,
            "n_vars": k,
            "df_resid": n_obs - k,
            "alpha": alpha,
            # z / chi2 inference, as Stata's xtlogit, fe.
            "inference": "z",
            "var_cov": vcov,
        },
        diagnostics={
            "aic": -2 * ll + 2 * k,
            "bic": -2 * ll + np.log(n_obs) * k,
        },
    )


@accepts_aliases(vce="robust")
@markout_clusters
def panel_probit(
    data: pd.DataFrame,
    y: str,
    x: List[str],
    id: str = "id",
    time: str = "time",
    method: str = "re",
    n_quadrature: int = 12,
    robust: str = "nonrobust",
    cluster: Optional[str] = None,
    maxiter: int = 200,
    tol: float = 1e-8,
    alpha: float = 0.05,
) -> EconometricResults:
    """Panel probit model.

    Parameters
    ----------
    data : DataFrame
        Panel data in long format.
    y : str
        Binary dependent variable (0/1).
    x : list of str
        Regressors.
    id, time : str
        Unit and time identifier columns.
    method : str
        're' (random effects) or 'cre' (Mundlak).
        FE probit not supported (incidental parameters problem).
    n_quadrature : int
        Gauss-Hermite quadrature points.
    robust : str
        'nonrobust' or 'robust'.
    cluster : str or None
        Column for cluster-robust SEs.
    maxiter : int
        Maximum optimizer iterations.
    tol : float
        Gradient tolerance.
    alpha : float
        Significance level for confidence intervals.

    Returns
    -------
    EconometricResults

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np
    >>> import pandas as pd
    >>> from scipy.stats import norm
    >>> rng = np.random.default_rng(0)
    >>> rows = []
    >>> for i in range(50):
    ...     a = rng.normal()  # unit effect
    ...     for t in range(4):
    ...         xit = rng.normal()
    ...         p = norm.cdf(0.7 * xit + a)
    ...         rows.append({"id": i, "time": t,
    ...                      "y": int(rng.uniform() < p), "x": xit})
    >>> df = pd.DataFrame(rows)
    >>> res = sp.panel_probit(df, y="y", x=["x"], id="id", time="time",
    ...                       method="re", n_quadrature=8)
    >>> bool("x" in res.params.index)
    True
    """
    from ..core._vcov_spec import parse_se_request

    method = method.lower()
    if method not in ("re", "cre"):
        raise ValueError(  # pragma: no cover
            "method must be 're' or 'cre'. FE probit is not supported "
            "due to the incidental parameters problem."
        )
    id_col, x_vars = id, list(x)

    # Stata grammar (xtprobit, re: vce(robust) clusters on the panels).
    # robust= and cluster= used to be accepted and ignored.
    se_req = parse_se_request(
        robust,
        cluster,
        function="panel_probit",
        supported=("nonrobust", "robust", "cluster"),
    )

    if method == "cre":
        data, mn = _add_mundlak_means(data, x_vars, id_col)
        return _wrap_re_result(
            data,
            y,
            x_vars + mn,
            id_col,
            n_quadrature,
            _probit_cdf,
            maxiter,
            tol,
            alpha,
            "Panel Probit (CRE/Mundlak)",
            "cre",
            link="probit",
            original_x=x_vars,
            mean_names=mn,
            se_kind=se_req.kind,
            cluster=se_req.cluster,
        )
    return _wrap_re_result(
        data,
        y,
        x_vars,
        id_col,
        n_quadrature,
        _probit_cdf,
        maxiter,
        tol,
        alpha,
        "Panel Probit (RE)",
        "re",
        link="probit",
        se_kind=se_req.kind,
        cluster=se_req.cluster,
    )
