"""
Surrogate-index estimators for long-term treatment effects.

Implements three closely-related identification strategies:

1. **Classical surrogate index** (Athey, Chetty, Imbens, Pollmann, Taubinsky
   2019) — under the surrogacy assumption ``Y ⟂ T | S``, the long-term ATE
   in the experiment equals

       ATE = E[ f(S) | T=1 ] - E[ f(S) | T=0 ],  f(s) = E[Y | S=s]

   where ``f`` is fit on the observational sample.

2. **Long-term from short-term experiments** (Ghassami et al. arXiv:2311.08527,
   2024) — iterates the above over multiple waves to handle long-term
   *treatments*, not just long-term outcomes.

3. **Proximal surrogate index** (Imbens, Kallus, Mao arXiv:2601.17712, 2026)
   — relaxes surrogacy to allow an unobserved ``U`` confounding ``S → Y``,
   using a proxy ``W``. Identifies the bridge function via a two-stage
   least-squares style moment condition.

Standard errors use an analytic two-sample delta-method asymptotic
variance (Athey et al. 2019, Theorem 1) by default, with a nonparametric
bootstrap fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
from scipy import stats

from ..core.results import CausalResult


__all__ = [
    "surrogate_index",
    "long_term_from_short",
    "proximal_surrogate_index",
    "SurrogateResult",
]


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass
class SurrogateResult:
    """Structured container for surrogate-index estimation artefacts."""

    estimate: float
    se: float
    ci: tuple
    n_exp: int
    n_obs: int
    method: str
    surrogate_fn: Optional[Callable] = None
    diagnostics: Optional[Dict[str, Any]] = None

    def summary(self) -> str:
        lo, hi = self.ci
        return (
            f"Surrogate-Index Estimator: {self.method}\n"
            f"  Long-term ATE estimate : {self.estimate:.6f}\n"
            f"  Standard error        : {self.se:.6f}\n"
            f"  95% CI                : [{lo:.6f}, {hi:.6f}]\n"
            f"  n (experimental)      : {self.n_exp}\n"
            f"  n (observational)     : {self.n_obs}\n"
        )


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _as_matrix(df: pd.DataFrame, cols: Sequence[str]) -> np.ndarray:
    arr = df[list(cols)].to_numpy(dtype=float, copy=True)
    if np.isnan(arr).any():
        raise ValueError(
            "Surrogate/covariate columns contain NaN. Drop or impute before "
            "calling sp.surrogate.surrogate_index()."
        )
    return arr


def _fit_outcome_model(
    S: np.ndarray,
    Y: np.ndarray,
    model: Union[str, Any],
    weights: Optional[np.ndarray] = None,
) -> Callable[[np.ndarray], np.ndarray]:
    """Fit E[Y | S] on the observational sample and return a predictor.

    ``model`` may be:

    - ``'ols'`` (default): linear regression with optional weights.
    - any sklearn-style object exposing ``fit(X, y)`` + ``predict(X)``.
    """
    if model == "ols":
        # Closed-form weighted least squares with intercept.
        n, p = S.shape
        X = np.column_stack([np.ones(n), S])
        w = np.ones(n) if weights is None else np.asarray(weights, dtype=float)
        W = w[:, None]
        XtWX = X.T @ (W * X)
        XtWy = X.T @ (w * Y)
        beta, *_ = np.linalg.lstsq(XtWX, XtWy, rcond=None)

        def predict(Snew: np.ndarray) -> np.ndarray:
            Xnew = np.column_stack([np.ones(Snew.shape[0]), Snew])
            return Xnew @ beta

        return predict

    # sklearn-style estimator
    fit_fn = getattr(model, "fit", None)
    predict_fn = getattr(model, "predict", None)
    if fit_fn is None or predict_fn is None:
        raise TypeError(
            "`model` must be 'ols' or an object with .fit(X,y) and .predict(X) "
            f"methods; got {type(model).__name__}."
        )
    try:
        if weights is not None:
            fit_fn(S, Y, sample_weight=weights)
        else:
            fit_fn(S, Y)
    except TypeError:
        fit_fn(S, Y)
    return predict_fn


def _delta_variance(
    h: np.ndarray,
    T: np.ndarray,
    resid_obs: np.ndarray,
    h_pred_obs: np.ndarray,
    n_exp: int,
    n_obs: int,
) -> float:
    """Athey-Chetty-Imbens two-sample variance.

    var(ATE_hat) = Var[h | T=1] / n1 + Var[h | T=0] / n0
                 + E[h_pred'(S) * (Y - f(S))]^2 * sigma^2 / n_obs (approx)

    We implement the dominant experimental-sample term plus a correction
    using residual variance from the observational regression.
    """
    T = T.astype(bool)
    h1 = h[T]
    h0 = h[~T]
    n1, n0 = max(h1.size, 1), max(h0.size, 1)
    var_exp = np.var(h1, ddof=1) / n1 + np.var(h0, ddof=1) / n0

    # First-order correction for observational-sample variability.
    var_obs = float(np.var(resid_obs, ddof=1)) * float(np.var(h_pred_obs)) / max(n_obs, 1)
    return var_exp + var_obs


# ---------------------------------------------------------------------------
# 1) Classical surrogate index (Athey-Chetty-Imbens 2019)
# ---------------------------------------------------------------------------


def surrogate_index(
    experimental: pd.DataFrame,
    observational: pd.DataFrame,
    *,
    treatment: str,
    surrogates: Sequence[str],
    long_term_outcome: str,
    covariates: Optional[Sequence[str]] = None,
    model: Union[str, Any] = "ols",
    alpha: float = 0.05,
    n_boot: int = 0,
    random_state: Optional[int] = None,
) -> CausalResult:
    """Athey-Chetty-Imbens surrogate-index estimator for the long-term ATE.

    Parameters
    ----------
    experimental : DataFrame
        Experimental sample. Must contain ``treatment`` and ``surrogates``.
        ``long_term_outcome`` need *not* be present — that is the whole point.
    observational : DataFrame
        Observational / historical sample with ``surrogates`` and
        ``long_term_outcome``. Need not contain ``treatment``.
    treatment : str
        Name of the binary treatment column in ``experimental``.
    surrogates : sequence of str
        Names of short-term surrogates — present in both samples.
    long_term_outcome : str
        Name of the long-term outcome column in ``observational``.
    covariates : sequence of str, optional
        Optional pre-treatment covariates appended to the surrogate vector.
    model : {'ols'} or sklearn-style estimator, default ``'ols'``
        How to fit ``E[Y | S]`` on the observational sample.
    alpha : float, default 0.05
    n_boot : int, default 0
        If ``> 0``, use ``n_boot`` paired-bootstrap replications instead of
        the analytic delta-method variance.
    random_state : int, optional

    Returns
    -------
    CausalResult
        ``estimand='ATE'``, ``method='surrogate_index'``.

    Notes
    -----
    The key identifying assumption is **surrogacy**: ``Y ⟂ T | S, X`` —
    conditional on the surrogate(s) and covariates, the treatment has no
    direct effect on the long-term outcome. This is strictly stronger than
    ignorability and should be defended explicitly (e.g. with placebo
    long-term outcomes in a validation sample).

    References
    ----------
    Athey, Chetty, Imbens, Pollmann, Taubinsky (2019). NBER WP 26463.
    """
    if not isinstance(experimental, pd.DataFrame):
        raise TypeError("`experimental` must be a pandas DataFrame.")
    if not isinstance(observational, pd.DataFrame):
        raise TypeError("`observational` must be a pandas DataFrame.")
    cols_e = {treatment, *surrogates}
    cols_o = {long_term_outcome, *surrogates}
    if covariates:
        cols_e |= set(covariates)
        cols_o |= set(covariates)
    missing_e = cols_e - set(experimental.columns)
    missing_o = cols_o - set(observational.columns)
    if missing_e:
        raise ValueError(f"experimental missing columns: {sorted(missing_e)}")
    if missing_o:
        raise ValueError(f"observational missing columns: {sorted(missing_o)}")

    feat_cols: List[str] = list(surrogates) + (list(covariates) if covariates else [])

    # --- Fit f(S) on observational sample -------------------------------
    S_o = _as_matrix(observational, feat_cols)
    Y_o = observational[long_term_outcome].to_numpy(dtype=float)
    predict = _fit_outcome_model(S_o, Y_o, model=model)
    h_pred_obs = predict(S_o)
    resid_obs = Y_o - h_pred_obs

    # --- Predict h on experimental sample and compute ATE ---------------
    S_e = _as_matrix(experimental, feat_cols)
    T_e = experimental[treatment].to_numpy(dtype=float)
    if set(np.unique(T_e)) - {0.0, 1.0}:
        raise ValueError(
            f"`treatment` '{treatment}' must be binary 0/1; found "
            f"{sorted(np.unique(T_e))}."
        )
    h = predict(S_e)
    T_bool = T_e.astype(bool)
    est = float(h[T_bool].mean() - h[~T_bool].mean())

    # --- Variance -------------------------------------------------------
    if n_boot > 0:
        rng = np.random.default_rng(random_state)
        boots = np.empty(n_boot)
        n_e, n_o = len(experimental), len(observational)
        for b in range(n_boot):
            ix_o = rng.integers(0, n_o, size=n_o)
            ix_e = rng.integers(0, n_e, size=n_e)
            pred_b = _fit_outcome_model(S_o[ix_o], Y_o[ix_o], model=model)
            h_b = pred_b(S_e[ix_e])
            T_b = T_e[ix_e].astype(bool)
            if T_b.all() or (~T_b).all():
                boots[b] = np.nan
                continue
            boots[b] = h_b[T_b].mean() - h_b[~T_b].mean()
        boots = boots[~np.isnan(boots)]
        if boots.size < 10:
            raise RuntimeError(
                "Bootstrap produced <10 valid replicates; check treatment "
                "overlap in experimental sample."
            )
        se = float(boots.std(ddof=1))
        ci = (
            float(np.quantile(boots, alpha / 2)),
            float(np.quantile(boots, 1 - alpha / 2)),
        )
        pval = 2.0 * min(
            (boots <= 0).mean() + 0.5 / boots.size,
            (boots >= 0).mean() + 0.5 / boots.size,
        )
    else:
        var = _delta_variance(
            h=h, T=T_e, resid_obs=resid_obs,
            h_pred_obs=h_pred_obs,
            n_exp=len(experimental), n_obs=len(observational),
        )
        se = float(np.sqrt(max(var, 0.0)))
        z = stats.norm.ppf(1 - alpha / 2)
        ci = (est - z * se, est + z * se)
        pval = (
            2 * (1 - stats.norm.cdf(abs(est) / se))
            if se > 0 else float("nan")
        )

    return CausalResult(
        method="surrogate_index",
        estimand="ATE",
        estimate=est,
        se=se,
        pvalue=pval,
        ci=ci,
        alpha=alpha,
        n_obs=int(len(experimental) + len(observational)),
        model_info={
            "n_exp": int(len(experimental)),
            "n_obs": int(len(observational)),
            "surrogates": list(surrogates),
            "covariates": list(covariates) if covariates else [],
            "outcome_model": str(model),
            "inference": "bootstrap" if n_boot > 0 else "delta_method",
            "resid_std_obs": float(resid_obs.std(ddof=1)),
        },
    )


# ---------------------------------------------------------------------------
# 2) Long-term effects of long-term treatments (Ghassami et al. 2024)
# ---------------------------------------------------------------------------


def long_term_from_short(
    experimental: pd.DataFrame,
    observational: pd.DataFrame,
    *,
    treatment: str,
    surrogates_waves: Sequence[Sequence[str]],
    long_term_outcome: str,
    covariates: Optional[Sequence[str]] = None,
    model: Union[str, Any] = "ols",
    alpha: float = 0.05,
    n_boot: int = 200,
    random_state: Optional[int] = None,
) -> CausalResult:
    """Long-term ATE under multi-wave short-term surrogates.

    Extends the classical surrogate index by chaining K surrogate *waves*
    — successive short-term measurements — so you can estimate the effect
    of a *sustained* treatment from a short experiment.

    Parameters
    ----------
    surrogates_waves : sequence of sequences
        ``[wave_1_cols, wave_2_cols, ..., wave_K_cols]`` where each
        ``wave_k_cols`` is a list of column names. Wave ``k`` is assumed
        measured in both samples.
    ...

    Notes
    -----
    Uses the iterated-expectation estimator (Ghassami et al. 2024, Eq. 3):

        f_K(s_K) = E[Y | S_K = s_K]                    in observational
        f_{k}(s_k) = E[f_{k+1}(S_{k+1}) | S_k = s_k]   in observational
        ATE = E[ f_1(S_1) | T=1 ] - E[ f_1(S_1) | T=0 ] in experimental

    Inference is bootstrap-based because the iterated delta variance is
    unwieldy in closed form.

    References
    ----------
    Ghassami, Yang, Shpitser, Tchetgen Tchetgen (arXiv:2311.08527, 2024).
    """
    if not surrogates_waves:
        raise ValueError("`surrogates_waves` must contain at least one wave.")
    flat_surr: List[str] = [c for wave in surrogates_waves for c in wave]
    base_cols_e = {treatment, *flat_surr}
    base_cols_o = {long_term_outcome, *flat_surr}
    if covariates:
        base_cols_e |= set(covariates)
        base_cols_o |= set(covariates)
    missing_e = base_cols_e - set(experimental.columns)
    missing_o = base_cols_o - set(observational.columns)
    if missing_e:
        raise ValueError(f"experimental missing columns: {sorted(missing_e)}")
    if missing_o:
        raise ValueError(f"observational missing columns: {sorted(missing_o)}")
    if n_boot < 50:
        raise ValueError("`n_boot` >= 50 required for multi-wave surrogate index.")

    cov_list = list(covariates) if covariates else []
    K = len(surrogates_waves)

    def _point_estimate(obs_df: pd.DataFrame, exp_df: pd.DataFrame) -> float:
        # Backward induction: start with Y on final wave, then iterate.
        current_target = obs_df[long_term_outcome].to_numpy(dtype=float)
        preds_cache: List[Callable] = []
        for k in range(K - 1, -1, -1):
            feat_k = list(surrogates_waves[k]) + cov_list
            X_k = _as_matrix(obs_df, feat_k)
            pred = _fit_outcome_model(X_k, current_target, model=model)
            preds_cache.append(pred)
            if k > 0:
                # Update target for next iteration: predict current layer
                # using wave k features on the observational sample.
                current_target = pred(X_k)
        # Final: apply wave-1 predictor to experimental sample.
        wave1_cols = list(surrogates_waves[0]) + cov_list
        f1_exp = preds_cache[-1](_as_matrix(exp_df, wave1_cols))
        T = exp_df[treatment].to_numpy(dtype=float).astype(bool)
        if T.all() or (~T).all():
            raise ValueError("Experimental sample has no treatment overlap.")
        return float(f1_exp[T].mean() - f1_exp[~T].mean())

    est = _point_estimate(observational, experimental)

    rng = np.random.default_rng(random_state)
    n_e, n_o = len(experimental), len(observational)
    boots = np.empty(n_boot)
    n_ok = 0
    for b in range(n_boot):
        ix_o = rng.integers(0, n_o, size=n_o)
        ix_e = rng.integers(0, n_e, size=n_e)
        try:
            boots[n_ok] = _point_estimate(
                observational.iloc[ix_o].reset_index(drop=True),
                experimental.iloc[ix_e].reset_index(drop=True),
            )
            n_ok += 1
        except ValueError:
            continue
    boots = boots[:n_ok]
    if n_ok < max(50, int(0.8 * n_boot)):
        raise RuntimeError(
            f"Only {n_ok}/{n_boot} bootstrap replicates succeeded; check "
            "treatment overlap and surrogate availability."
        )
    se = float(boots.std(ddof=1))
    ci = (
        float(np.quantile(boots, alpha / 2)),
        float(np.quantile(boots, 1 - alpha / 2)),
    )
    pval = 2.0 * min(
        (boots <= 0).mean() + 0.5 / n_ok,
        (boots >= 0).mean() + 0.5 / n_ok,
    )

    return CausalResult(
        method="long_term_from_short",
        estimand="ATE",
        estimate=est,
        se=se,
        pvalue=pval,
        ci=ci,
        alpha=alpha,
        n_obs=int(len(experimental) + len(observational)),
        model_info={
            "n_exp": int(len(experimental)),
            "n_obs": int(len(observational)),
            "n_waves": K,
            "waves": [list(w) for w in surrogates_waves],
            "covariates": cov_list,
            "outcome_model": str(model),
            "inference": "bootstrap",
            "n_boot_effective": int(n_ok),
        },
    )


# ---------------------------------------------------------------------------
# 3) Proximal surrogate index (Imbens-Kallus-Mao 2026)
# ---------------------------------------------------------------------------


def proximal_surrogate_index(
    experimental: pd.DataFrame,
    observational: pd.DataFrame,
    *,
    treatment: str,
    surrogates: Sequence[str],
    proxies: Sequence[str],
    long_term_outcome: str,
    covariates: Optional[Sequence[str]] = None,
    alpha: float = 0.05,
    n_boot: int = 200,
    random_state: Optional[int] = None,
) -> CausalResult:
    """Proximal surrogate index — long-term ATE under unobserved confounding.

    Relaxes the surrogacy assumption ``Y ⟂ T | S`` by allowing an
    unobserved ``U`` that confounds ``S → Y``. Identification uses a proxy
    ``W`` satisfying the two-stage completeness conditions of Imbens,
    Kallus & Mao (2026). In linear-Gaussian form the bridge function
    ``h(s, x)`` solves

        E[Y | S, X, W] = W' * α + β * h(S, X)

    which we estimate by two-stage least squares with ``W`` instrumenting
    for the unobserved structure.

    Parameters
    ----------
    proxies : sequence of str
        Names of proxy variables ``W`` — present in the *observational*
        sample only. Proxies must be (a) related to ``U`` and (b) excluded
        from the direct effect on ``Y`` (classical IV-style exclusion).

    Notes
    -----
    The linear-Gaussian implementation below is faithful to the paper's
    proposition 3.1 (the bridge equation) but makes strong parametric
    assumptions. For nonparametric bridges, use the ``model`` hooks in
    :func:`surrogate_index` with a kernel/NN estimator and pass a custom
    2SLS wrapper.

    References
    ----------
    Imbens, Kallus, Mao (arXiv:2601.17712, 2026).
    """
    if len(proxies) == 0:
        raise ValueError(
            "`proxies` must contain at least one proxy variable W; "
            "otherwise reduce to sp.surrogate.surrogate_index()."
        )
    cov_list = list(covariates) if covariates else []
    feat_s = list(surrogates) + cov_list
    cols_o = {long_term_outcome, *surrogates, *proxies, *cov_list}
    cols_e = {treatment, *surrogates, *cov_list}
    missing_o = cols_o - set(observational.columns)
    missing_e = cols_e - set(experimental.columns)
    if missing_o:
        raise ValueError(f"observational missing columns: {sorted(missing_o)}")
    if missing_e:
        raise ValueError(f"experimental missing columns: {sorted(missing_e)}")

    def _bridge_predict(obs_df: pd.DataFrame) -> Callable[[np.ndarray], np.ndarray]:
        """Solve E[Y|S,X,W] = Wα + h(S,X) by 2SLS.

        First stage: regress S on W (and X) → S_hat(W, X)
        Second stage: regress Y on W, S_hat(W, X), X
        Bridge function h(s, x) is defined by the S-coefficients in stage 2
        applied to the *observed* (not projected) s.
        """
        S_o = _as_matrix(obs_df, feat_s)  # n_o × p_s+p_x
        W_o = _as_matrix(obs_df, list(proxies))  # n_o × p_w
        Y_o = obs_df[long_term_outcome].to_numpy(dtype=float)
        n_o = S_o.shape[0]
        ones = np.ones((n_o, 1))
        # Stage 1: S_hat ~ [1, W, X]
        X_stage1 = np.column_stack([ones, W_o])
        if cov_list:
            X_stage1 = np.column_stack([X_stage1, _as_matrix(obs_df, cov_list)])
        beta1, *_ = np.linalg.lstsq(X_stage1, S_o, rcond=None)
        S_hat_o = X_stage1 @ beta1
        # Stage 2: Y ~ [1, W, S_hat, X]
        X_stage2 = np.column_stack([ones, W_o, S_hat_o])
        if cov_list:
            X_stage2 = np.column_stack([X_stage2, _as_matrix(obs_df, cov_list)])
        beta2, *_ = np.linalg.lstsq(X_stage2, Y_o, rcond=None)
        # Extract the S-part coefficients: after [1, W]
        p_w = W_o.shape[1]
        s_start = 1 + p_w
        s_end = s_start + S_hat_o.shape[1]
        beta_s = beta2[s_start:s_end]
        intercept = beta2[0]
        beta_x = beta2[s_end:] if cov_list else None

        def predict(S_exp: np.ndarray, X_exp: Optional[np.ndarray] = None) -> np.ndarray:
            val = intercept + S_exp @ beta_s
            if beta_x is not None and X_exp is not None:
                val = val + X_exp @ beta_x
            return val

        return predict

    def _point_estimate(obs_df: pd.DataFrame, exp_df: pd.DataFrame) -> float:
        predict = _bridge_predict(obs_df)
        S_e = _as_matrix(exp_df, list(surrogates))
        X_e = _as_matrix(exp_df, cov_list) if cov_list else None
        h_e = predict(S_e, X_e)
        T = exp_df[treatment].to_numpy(dtype=float).astype(bool)
        if T.all() or (~T).all():
            raise ValueError("Experimental sample has no treatment overlap.")
        return float(h_e[T].mean() - h_e[~T].mean())

    est = _point_estimate(observational, experimental)

    rng = np.random.default_rng(random_state)
    n_e, n_o = len(experimental), len(observational)
    boots = np.empty(n_boot)
    n_ok = 0
    for b in range(n_boot):
        ix_o = rng.integers(0, n_o, size=n_o)
        ix_e = rng.integers(0, n_e, size=n_e)
        try:
            boots[n_ok] = _point_estimate(
                observational.iloc[ix_o].reset_index(drop=True),
                experimental.iloc[ix_e].reset_index(drop=True),
            )
            n_ok += 1
        except (ValueError, np.linalg.LinAlgError):
            continue
    boots = boots[:n_ok]
    if n_ok < max(50, int(0.7 * n_boot)):
        raise RuntimeError(
            f"Only {n_ok}/{n_boot} bootstrap replicates succeeded; check "
            "proxy strength and overlap."
        )
    se = float(boots.std(ddof=1))
    ci = (
        float(np.quantile(boots, alpha / 2)),
        float(np.quantile(boots, 1 - alpha / 2)),
    )
    pval = 2.0 * min(
        (boots <= 0).mean() + 0.5 / n_ok,
        (boots >= 0).mean() + 0.5 / n_ok,
    )

    return CausalResult(
        method="proximal_surrogate_index",
        estimand="ATE",
        estimate=est,
        se=se,
        pvalue=pval,
        ci=ci,
        alpha=alpha,
        n_obs=int(len(experimental) + len(observational)),
        model_info={
            "n_exp": int(len(experimental)),
            "n_obs": int(len(observational)),
            "surrogates": list(surrogates),
            "proxies": list(proxies),
            "covariates": cov_list,
            "inference": "bootstrap",
            "n_boot_effective": int(n_ok),
        },
    )
