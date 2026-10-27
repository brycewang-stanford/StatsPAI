"""Treatment effects on survival time: Stata's ``stteffects`` (RA, WRA, IPW,
IPWRA) for a binary treatment.

Each estimator is a set of estimating equations solved jointly (Stata
[CAUSAL] stteffects, Methods and formulas): the censoring model
(``gamma``), the treatment model (``alpha``), one outcome model per arm
(``beta_0``, ``beta_1``) and the potential-outcome means. The point
estimates come from solving the blocks in order; the variance is the
sandwich ``A^-1 B A^-T / N`` of the stacked moments, ``A`` their Jacobian
(central differences of the analytic moments) and ``B`` the mean of their
outer products, which is the robust VCE Stata reports.

Weights (Stata's notation; ``c`` censored, ``p`` the treatment
probability, ``G`` the censoring survivor at the observed time):

* IPW and WAC-IPWRA, ATE: ``tau (1 - c) / {p G}`` and ``(1 - tau)(1 - c) /
  {(1 - p) G}``; ATET: ``tau (1 - c) / G`` and ``p (1 - tau)(1 - c) /
  {(1 - p) G}``.
* LAC-IPWRA (no censoring model): the same without ``(1 - c) / G``; the
  censored rows enter the outcome likelihood through ``log S``.
* WRA: ``(1 - c) / G`` within each arm.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from ..core.results import CausalResult
from ..exceptions import DataInsufficient, MethodIncompatibility
from . import _param_surv as ps

METHODS = ("ra", "wra", "ipw", "ipwra")


def _design(df: pd.DataFrame, cols: Sequence[str]) -> np.ndarray:
    X = df[list(cols)].to_numpy(dtype=float) if cols else np.zeros((len(df), 0))
    return np.column_stack([X, np.ones(len(df))])


def stteffects(
    data: pd.DataFrame,
    duration: str,
    event: str,
    treat: str,
    covariates: Optional[Sequence[str]] = None,
    method: str = "ra",
    estimand: str = "ATE",
    treat_covariates: Optional[Sequence[str]] = None,
    censor_covariates: Optional[Sequence[str]] = None,
    dist: str = "weibull",
    censor_dist: str = "weibull",
    treat_model: str = "logit",
    alpha: float = 0.05,
) -> CausalResult:
    """Average treatment effect on survival time (Stata ``stteffects``).

    The effect is on the *mean* survival time, from parametric models of the
    outcome, the treatment and / or the censoring time, for a binary
    treatment and right-censored data without delayed entry.

    Parameters
    ----------
    data : pd.DataFrame
        One row per subject.
    duration : str
        Observed time (failure or censoring).
    event : str
        1 = failure observed, 0 = censored.
    treat : str
        Binary 0/1 treatment.
    covariates : list of str, optional
        Outcome-model covariates (``ra`` / ``wra`` / ``ipwra``). A constant
        is added.
    method : {'ra', 'wra', 'ipw', 'ipwra'}, default 'ra'
        ``'ra'``: a survival model per arm fitted with the censored rows in
        the likelihood; the potential-outcome means average its predicted
        mean times (``stteffects ra``). ``'wra'``: the same models fitted on
        the failures weighted by the inverse probability of not being
        censored (``stteffects wra``; needs ``censor_covariates``).
        ``'ipw'``: weighted means of the observed failure times, weights the
        inverse of the probability of the treatment received times that of
        not being censored (``stteffects ipw``; needs ``censor_covariates``
        when some rows are censored). ``'ipwra'``: the outcome models
        fitted with treatment weights; with ``censor_covariates`` the
        censoring is handled by weights (WAC), without it by the likelihood
        (LAC), as ``stteffects ipwra``.
    estimand : {'ATE', 'ATET'}, default 'ATE'
        ``'ATET'`` averages over the treated.
    treat_covariates : list of str, optional
        Treatment-model covariates (``ipw`` / ``ipwra``); defaults to
        ``covariates``.
    censor_covariates : list of str, optional
        Censoring-model covariates; pass ``[]`` for a model with a constant
        only. The list may include the treatment variable, as Stata's
        ``cmodel()`` allows.
    dist : {'weibull', 'exponential', 'lognormal'}, default 'weibull'
        Outcome model (Stata's ``omodel``).
    censor_dist : {'weibull', 'exponential', 'lognormal'}, default 'weibull'
        Censoring model (Stata's ``cmodel``).
    treat_model : {'logit', 'probit'}, default 'logit'
        Treatment model.
    alpha : float, default 0.05
        Significance level.

    Returns
    -------
    CausalResult
        ``estimate`` / ``se`` of the ATE (or ATET); ``detail`` holds the
        effect and both potential-outcome means (``POmean0``,
        ``POmean1``); ``model_info['coefficients']`` the nuisance models;
        ``model_info['vcov']`` the joint covariance of the effect and the
        means.

    Notes
    -----
    Numbers match Stata 18 ``stteffects`` (see
    ``tests/reference_parity/test_stteffects_stata_parity.py``). An
    ancillary (shape) equation with covariates, multi-valued treatments
    and the gamma model are not implemented. The estimators do not allow
    delayed entry, for the reason Stata gives: the truncation process
    would itself need a model.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> n = 600
    >>> x = rng.normal(size=n)
    >>> a = rng.binomial(1, 1 / (1 + np.exp(-0.5 * x)))
    >>> t = np.exp(1 + 0.5 * a + 0.3 * x) * rng.weibull(1.5, n)
    >>> c = rng.uniform(0, 12, n)
    >>> df = pd.DataFrame({"t": np.minimum(t, c), "d": (t <= c).astype(int),
    ...                    "a": a, "x": x})
    >>> r = sp.stteffects(df, "t", "d", "a", covariates=["x"], method="ra")
    >>> r.estimand
    'ATE'
    """
    method = str(method).lower()
    if method not in METHODS:
        raise MethodIncompatibility(
            f"stteffects: method must be one of {list(METHODS)}; got {method!r}.",
            recovery_hint="Use 'ra', 'wra', 'ipw' or 'ipwra'.",
        )
    estimand = str(estimand).upper()
    if estimand not in ("ATE", "ATET"):
        raise MethodIncompatibility(
            f"stteffects: estimand must be 'ATE' or 'ATET'; got {estimand!r}.",
            recovery_hint="Use estimand='ATE' or 'ATET'.",
        )
    dist = ps.check_dist(dist, "dist")
    censor_dist = ps.check_dist(censor_dist, "censor_dist")
    treat_model = str(treat_model).lower()
    if treat_model not in ("logit", "probit"):
        raise MethodIncompatibility(
            "stteffects: treat_model must be 'logit' or 'probit'; got "
            f"{treat_model!r}.",
            recovery_hint="Use treat_model='logit'.",
        )
    covariates = list(covariates or [])
    if treat_covariates is None:
        treat_covariates = covariates
    treat_covariates = list(treat_covariates)
    cens_cols = None if censor_covariates is None else list(censor_covariates)
    if method == "ra" and cens_cols is not None:
        raise MethodIncompatibility(
            "stteffects: method='ra' handles censoring through the outcome "
            "likelihood and takes no censoring model.",
            recovery_hint="Drop censor_covariates=, or use method='wra' / 'ipwra'.",
        )

    cols = [duration, event, treat] + covariates + treat_covariates + (cens_cols or [])
    cols = list(dict.fromkeys(cols))
    missing = [c for c in cols if c not in data.columns]
    if missing:
        raise DataInsufficient(
            f"stteffects: columns not found: {missing}",
            recovery_hint="Check the column names.",
        )
    df = data[cols].dropna()
    t = df[duration].to_numpy(dtype=float)
    d = df[event].to_numpy(dtype=float)
    from ..core._validate import treatment_as_float

    tr = treatment_as_float(df[treat], function="sp.stteffects")
    if not set(np.unique(tr)) <= {0.0, 1.0} or len(np.unique(tr)) < 2:
        raise MethodIncompatibility(
            "stteffects: the treatment must be binary 0/1 with both levels " "present.",
            recovery_hint="Recode the treatment to 0/1.",
        )
    if not set(np.unique(d)) <= {0.0, 1.0}:
        raise MethodIncompatibility(
            "stteffects: the event indicator must be 0/1.",
            recovery_hint="Recode the event column (1 = failure).",
        )
    if np.any(t <= 0):
        raise DataInsufficient(
            "stteffects: survival times must be positive.",
            recovery_hint="Drop rows with zero or negative time.",
        )
    n = len(t)
    logt = np.log(t)
    censored = (d == 0).any()
    if method in ("wra", "ipw") and cens_cols is None and censored:
        raise MethodIncompatibility(
            f"stteffects: method={method!r} with censored rows needs a "
            "censoring model (censor_covariates=, [] for a constant only).",
            recovery_hint="Pass censor_covariates=, or use method='ra' / 'ipwra'.",
        )
    use_cens = cens_cols is not None and (censored or method == "wra")
    if cens_cols is not None and not censored:
        raise MethodIncompatibility(
            "stteffects: no row is censored, so a censoring model is not "
            "identified.",
            recovery_hint="Drop censor_covariates=.",
        )

    X = _design(df, covariates)
    Z = _design(df, treat_covariates)
    W = _design(df, cens_cols or [])
    one = np.ones(n)
    uses_treat = method in ("ipw", "ipwra")
    uses_out = method != "ipw"
    atet = estimand == "ATET"

    # ---- blocks: sizes ------------------------------------------------
    k_c = ps.n_params(censor_dist, W.shape[1]) if use_cens else 0
    k_t = Z.shape[1] if uses_treat else 0
    k_o = ps.n_params(dist, X.shape[1]) if uses_out else 0
    sl_c = slice(0, k_c)
    sl_t = slice(k_c, k_c + k_t)
    sl_0 = slice(k_c + k_t, k_c + k_t + k_o)
    sl_1 = slice(k_c + k_t + k_o, k_c + k_t + 2 * k_o)
    i_pom0 = k_c + k_t + 2 * k_o
    i_eff = i_pom0 + 1
    k = i_eff + 1

    def weights(theta: np.ndarray) -> Dict[str, np.ndarray]:
        G = ps.survivor(theta[sl_c], W, logt, censor_dist) if use_cens else one
        if uses_treat:
            p, _ = ps.binary_choice(theta[sl_t], Z, treat_model)
        else:
            p = None
        out: Dict[str, np.ndarray] = {"G": G}
        if p is not None:
            if atet:
                w1, w0 = tr, p * (1 - tr) / (1 - p)
            else:
                w1, w0 = tr / p, (1 - tr) / (1 - p)
            out["w1"], out["w0"] = w1, w0
            out["p"] = p
        return out

    def outcome_rows(
        theta: np.ndarray, arm: int, wts: Dict[str, np.ndarray]
    ) -> np.ndarray:
        beta = theta[sl_1 if arm == 1 else sl_0]
        in_arm = tr if arm == 1 else 1 - tr
        if method == "ra":
            return ps.score_rows(beta, X, logt, d, in_arm, dist, True)
        if method == "wra":
            return ps.score_rows(beta, X, logt, d, in_arm / wts["G"], dist, False)
        w_arm = wts["w1"] if arm == 1 else wts["w0"]
        if use_cens:  # WAC-IPWRA
            return ps.score_rows(beta, X, logt, d, w_arm / wts["G"], dist, False)
        return ps.score_rows(beta, X, logt, d, w_arm, dist, True)  # LAC

    def moments(theta: np.ndarray) -> np.ndarray:
        psi = np.zeros((n, k))
        wts = weights(theta)
        if use_cens:
            psi[:, sl_c] = ps.score_rows(
                theta[sl_c], W, logt, 1 - d, one, censor_dist, True
            )
        if uses_treat:
            psi[:, sl_t] = ps.binary_choice(theta[sl_t], Z, treat_model)[1](tr)
        pom0, eff = theta[i_pom0], theta[i_eff]
        if uses_out:
            psi[:, sl_0] = outcome_rows(theta, 0, wts)
            psi[:, sl_1] = outcome_rows(theta, 1, wts)
            m0 = ps.mean_time(theta[sl_0], X, dist)
            m1 = ps.mean_time(theta[sl_1], X, dist)
            sel = tr if atet else one
            psi[:, i_pom0] = sel * (m0 - pom0)
            psi[:, i_eff] = sel * (m1 - pom0 - eff)
        else:  # ipw: weighted means of the observed failure times
            g = wts["G"] if use_cens else one
            om0 = wts["w0"] * d / g
            om1 = wts["w1"] * d / g
            psi[:, i_pom0] = om0 * (t - pom0)
            psi[:, i_eff] = om1 * (t - pom0 - eff)
        return psi

    # ---- point estimates, block by block -----------------------------
    theta = np.zeros(k)
    if use_cens:
        theta[sl_c] = ps.fit(W, logt, 1 - d, one, censor_dist, True, "censoring model")
    if uses_treat:
        theta[sl_t] = ps.fit_binary(Z, tr, treat_model)
    wts = weights(theta)
    if uses_out:
        for arm, sl in ((0, sl_0), (1, sl_1)):
            in_arm = tr if arm == 1 else 1 - tr
            if method == "ra":
                wv, cterm = in_arm, True
            elif method == "wra":
                wv, cterm = in_arm / wts["G"], False
            elif use_cens:
                wv, cterm = (wts["w1"] if arm else wts["w0"]) / wts["G"], False
            else:
                wv, cterm = (wts["w1"] if arm else wts["w0"]), True
            theta[sl] = ps.fit(
                X, logt, d, wv, dist, cterm, f"outcome model (arm {arm})"
            )
        sel = tr if atet else one
        m0 = ps.mean_time(theta[sl_0], X, dist)
        m1 = ps.mean_time(theta[sl_1], X, dist)
        theta[i_pom0] = float((sel * m0).sum() / sel.sum())
        theta[i_eff] = float((sel * m1).sum() / sel.sum()) - theta[i_pom0]
    else:
        g = wts["G"] if use_cens else one
        om0 = wts["w0"] * d / g
        om1 = wts["w1"] * d / g
        theta[i_pom0] = float((om0 * t).sum() / om0.sum())
        theta[i_eff] = float((om1 * t).sum() / om1.sum()) - theta[i_pom0]

    # ---- sandwich of the stacked moments ------------------------------
    psi = moments(theta)
    A = np.empty((k, k))
    for j in range(k):
        h = 1e-6 * max(1.0, abs(theta[j]))
        e = np.zeros(k)
        e[j] = h
        A[:, j] = (moments(theta + e).sum(axis=0) - moments(theta - e).sum(axis=0)) / (
            2 * h * n
        )
    B = psi.T @ psi / n
    try:
        A_inv = np.linalg.inv(A)
    except np.linalg.LinAlgError as exc:
        raise MethodIncompatibility(
            "stteffects: the Jacobian of the estimating equations is singular.",
            recovery_hint="Drop collinear covariates or check overlap.",
        ) from exc
    V = A_inv @ B @ A_inv.T / n

    z = stats.norm.ppf(1 - alpha / 2)
    est = float(theta[i_eff])
    se = float(np.sqrt(V[i_eff, i_eff]))
    pom0 = float(theta[i_pom0])
    se0 = float(np.sqrt(V[i_pom0, i_pom0]))
    pom1 = est + pom0
    se1 = float(np.sqrt(V[i_eff, i_eff] + V[i_pom0, i_pom0] + 2 * V[i_eff, i_pom0]))
    rows: List[Dict[str, Any]] = []
    for name, b, s in (
        (estimand, est, se),
        ("POmean0", pom0, se0),
        ("POmean1", pom1, se1),
    ):
        rows.append(
            {
                "term": name,
                "estimate": b,
                "se": s,
                "z": b / s,
                "p_value": float(2 * stats.norm.sf(abs(b / s))),
                "ci_lower": b - z * s,
                "ci_upper": b + z * s,
            }
        )
    detail = pd.DataFrame(rows)
    coefs: Dict[str, Any] = {}
    if use_cens:
        coefs["censoring"] = dict(
            zip(_names(cens_cols or [], censor_dist), theta[sl_c].tolist())
        )
    if uses_treat:
        coefs["treatment"] = dict(
            zip(list(treat_covariates) + ["_cons"], theta[sl_t].tolist())
        )
    if uses_out:
        coefs["outcome0"] = dict(zip(_names(covariates, dist), theta[sl_0].tolist()))
        coefs["outcome1"] = dict(zip(_names(covariates, dist), theta[sl_1].tolist()))
    label = {
        "ra": "regression adjustment",
        "wra": "weighted regression adjustment",
        "ipw": "inverse-probability weighting",
        "ipwra": "IPW regression adjustment"
        + (
            " (weighted-adjusted censoring)"
            if use_cens
            else " (likelihood-adjusted censoring)"
        ),
    }[method]
    model_info = {
        "estimator": f"stteffects {method}",
        "description": f"Survival-time treatment effects: {label}",
        "outcome_model": dist if uses_out else None,
        "censoring_model": censor_dist if use_cens else None,
        "treatment_model": treat_model if uses_treat else None,
        "coefficients": coefs,
        "vcov": pd.DataFrame(
            V[np.ix_([i_eff, i_pom0], [i_eff, i_pom0])],
            index=[estimand, "POmean0"],
            columns=[estimand, "POmean0"],
        ),
        "n_treated": int(tr.sum()),
        "n_failures": int(d.sum()),
    }
    return CausalResult(
        method=f"stteffects {method}",
        estimand=estimand,
        estimate=est,
        se=se,
        pvalue=float(2 * stats.norm.sf(abs(est / se))),
        ci=(est - z * se, est + z * se),
        alpha=alpha,
        n_obs=n,
        detail=detail,
        model_info=model_info,
    )


def _names(cols: Sequence[str], dist: str) -> List[str]:
    out = list(cols) + ["_cons"]
    if dist != "exponential":
        out.append("ln_shape" if dist == "weibull" else "ln_sigma")
    return out
