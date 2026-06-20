"""
Surrogate-index estimators for long-term treatment effects.

Implements three closely-related identification strategies:

1. **Classical surrogate index** (Athey, Chetty, Imbens & Kang 2019,
   NBER WP 26463) — under the surrogacy assumption ``Y ⟂ T | S``, the long-term ATE
   in the experiment equals

       ATE = E[ f(S) | T=1 ] - E[ f(S) | T=0 ],  f(s) = E[Y | S=s]

   where ``f`` is fit on the observational sample.

2. **Long-term from short-term experiments** (Tran, Bibaut & Kallus
   arXiv:2311.08527, 2023) — iterates the above over multiple waves to
   handle long-term *treatments*, not just long-term outcomes.

3. **Proximal surrogate index** (Imbens, Kallus, Mao & Wang 2025, JRSS-B
   87(2); arXiv:2202.07234) — relaxes surrogacy to allow an unobserved
   ``U`` confounding ``S → Y``, using a proxy ``W``. Identifies the
   bridge function via a two-stage least-squares style moment condition.

Standard errors use an analytic two-sample delta-method asymptotic
variance (Athey, Chetty, Imbens & Kang 2019, Theorem 1) by default,
with a nonparametric bootstrap fallback.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
from scipy import stats

from ..core.results import CausalResult
from ..exceptions import ConvergenceFailure, DataInsufficient, MethodIncompatibility

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


def _require_dataframe(value: Any, name: str) -> pd.DataFrame:
    if not isinstance(value, pd.DataFrame):
        raise MethodIncompatibility(
            f"`{name}` must be a pandas DataFrame.",
            diagnostics={name: value.__class__.__name__},
        )
    if value.empty:
        raise DataInsufficient(
            f"`{name}` must contain at least one row.",
            diagnostics={name: len(value)},
        )
    return value


def _require_column_name(value: Any, name: str) -> str:
    if not isinstance(value, str) or not value:
        raise MethodIncompatibility(
            f"`{name}` must be a non-empty column name.",
            diagnostics={name: repr(value)},
        )
    return value


def _coerce_column_list(
    columns: Any,
    name: str,
    *,
    allow_empty: bool = False,
) -> List[str]:
    if isinstance(columns, str):
        out = [columns]
    else:
        try:
            out = list(columns)
        except TypeError as exc:
            raise MethodIncompatibility(
                f"`{name}` must be a column name or list of column names.",
                diagnostics={name: repr(columns)},
            ) from exc
    if not allow_empty and not out:
        raise MethodIncompatibility(
            f"`{name}` must contain at least one column name.",
            diagnostics={name: out},
        )
    bad = [c for c in out if not isinstance(c, str) or not c]
    if bad:
        raise MethodIncompatibility(
            f"`{name}` must contain only non-empty string column names.",
            diagnostics={name: out, "invalid_columns": bad},
        )
    return out


def _coerce_optional_column_list(columns: Any, name: str) -> List[str]:
    if columns is None:
        return []
    return _coerce_column_list(columns, name, allow_empty=True)


def _coerce_waves(waves: Any) -> List[List[str]]:
    if isinstance(waves, str):
        raise MethodIncompatibility(
            "`surrogates_waves` must be a sequence of per-wave column lists.",
            diagnostics={"surrogates_waves": repr(waves)},
        )
    try:
        out = [
            _coerce_column_list(wave, f"surrogates_waves[{i}]")
            for i, wave in enumerate(waves)
        ]
    except TypeError as exc:
        raise MethodIncompatibility(
            "`surrogates_waves` must be a sequence of per-wave column lists.",
            diagnostics={"surrogates_waves": repr(waves)},
        ) from exc
    if not out:
        raise MethodIncompatibility(
            "`surrogates_waves` must contain at least one wave.",
            diagnostics={"n_waves": 0},
        )
    return out


def _require_open_unit_float(value: Any, name: str) -> float:
    if isinstance(value, (bool, np.bool_)):
        raise MethodIncompatibility(
            f"`{name}` must be a number in (0, 1).",
            diagnostics={name: repr(value)},
        )
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise MethodIncompatibility(
            f"`{name}` must be a number in (0, 1).",
            diagnostics={name: repr(value)},
        ) from exc
    if not np.isfinite(out) or not 0.0 < out < 1.0:
        raise MethodIncompatibility(
            f"`{name}` must be in (0, 1).",
            diagnostics={name: out},
        )
    return out


def _require_int_at_least(value: Any, name: str, minimum: int) -> int:
    if isinstance(value, (bool, np.bool_)):
        raise MethodIncompatibility(
            f"`{name}` must be an integer >= {minimum}.",
            diagnostics={name: repr(value), "minimum": minimum},
        )
    try:
        out = int(value)
    except (TypeError, ValueError) as exc:
        raise MethodIncompatibility(
            f"`{name}` must be an integer >= {minimum}.",
            diagnostics={name: repr(value), "minimum": minimum},
        ) from exc
    if out != value or out < minimum:
        raise MethodIncompatibility(
            f"`{name}` must be an integer >= {minimum}.",
            diagnostics={name: repr(value), "minimum": minimum},
        )
    return out


def _raise_missing_columns(sample: str, missing: set[str]) -> None:
    if missing:
        raise MethodIncompatibility(
            f"{sample} missing columns: {sorted(missing)}",
            diagnostics={sample: sorted(missing)},
        )


def _require_finite_vector(values: np.ndarray, name: str) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0 or np.any(~np.isfinite(arr)):
        raise DataInsufficient(
            f"`{name}` must contain finite numeric values.",
            diagnostics={"name": name, "n": int(arr.size)},
        )
    return arr


def _require_binary_treatment(values: np.ndarray, name: str) -> np.ndarray:
    arr = _require_finite_vector(values, name)
    uniques = np.unique(arr)
    if set(uniques) - {0.0, 1.0}:
        raise MethodIncompatibility(
            f"`treatment` '{name}' must be binary 0/1; found {sorted(uniques)}.",
            diagnostics={"treatment": name, "values": sorted(uniques.tolist())},
        )
    mask = arr.astype(bool)
    if mask.all() or (~mask).all():
        raise DataInsufficient(
            "Experimental sample has no treatment overlap.",
            recovery_hint="Include both treated and control observations.",
            diagnostics={"treatment": name, "n_treated": int(mask.sum())},
        )
    return mask


def _as_matrix(df: pd.DataFrame, cols: Sequence[str]) -> np.ndarray:
    arr = df[list(cols)].to_numpy(dtype=float, copy=True)
    if np.any(~np.isfinite(arr)):
        raise DataInsufficient(
            "Surrogate/covariate columns contain NaN. Drop or impute before "
            "calling sp.surrogate.surrogate_index().",
            recovery_hint="Drop or impute missing surrogate/covariate values.",
            diagnostics={"columns": list(cols)},
        )
    return np.asarray(arr)


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
            return np.asarray(Xnew @ beta)

        return predict

    # sklearn-style estimator
    fit_fn = getattr(model, "fit", None)
    predict_fn = getattr(model, "predict", None)
    if fit_fn is None or predict_fn is None:
        raise MethodIncompatibility(
            "`model` must be 'ols' or an object with .fit(X,y) and .predict(X) "
            f"methods; got {type(model).__name__}.",
            diagnostics={"model": type(model).__name__},
        )
    try:
        if weights is not None:
            fit_fn(S, Y, sample_weight=weights)
        else:
            fit_fn(S, Y)
    except TypeError:
        fit_fn(S, Y)
    predict_callable: Callable[[np.ndarray], np.ndarray] = predict_fn
    return predict_callable


def _delta_variance(
    h: np.ndarray,
    T: np.ndarray,
    resid_obs: np.ndarray,
    h_pred_obs: np.ndarray,
    n_exp: int,
    n_obs: int,
) -> float:
    """Athey-Chetty-Imbens-Kang two-sample variance.

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
    var_obs = (
        float(np.var(resid_obs, ddof=1)) * float(np.var(h_pred_obs)) / max(n_obs, 1)
    )
    return float(var_exp + var_obs)


# ---------------------------------------------------------------------------
# 1) Classical surrogate index (Athey-Chetty-Imbens-Kang 2019)
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
    """Athey-Chetty-Imbens-Kang surrogate-index estimator for the long-term ATE.

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

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> # Observational sample: surrogate S drives the long-term outcome Y.
    >>> S_o = rng.normal(size=400)
    >>> obs = pd.DataFrame({"S": S_o, "Y": 1.5 * S_o + rng.normal(scale=0.5, size=400)})
    >>> # Experimental sample: treatment shifts S; Y is never observed here.
    >>> T = rng.integers(0, 2, size=300)
    >>> exp = pd.DataFrame({"T": T, "S": 0.8 * T + rng.normal(size=300)})
    >>> res = sp.surrogate_index(
    ...     exp, obs, treatment="T", surrogates=["S"], long_term_outcome="Y",
    ... )
    >>> bool(res.estimate > 0)  # positive long-term ATE recovered
    True

    References
    ----------
    Athey, S., Chetty, R., Imbens, G. W., & Kang, H. (2019).
    "The Surrogate Index: Combining Short-Term Proxies to Estimate
    Long-Term Treatment Effects More Rapidly and Precisely."
    NBER Working Paper 26463. [@athey2019surrogate]
    """
    experimental = _require_dataframe(experimental, "experimental")
    observational = _require_dataframe(observational, "observational")
    treatment = _require_column_name(treatment, "treatment")
    surrogates = _coerce_column_list(surrogates, "surrogates")
    long_term_outcome = _require_column_name(long_term_outcome, "long_term_outcome")
    cov_list = _coerce_optional_column_list(covariates, "covariates")
    alpha = _require_open_unit_float(alpha, "alpha")
    n_boot = _require_int_at_least(n_boot, "n_boot", 0)
    cols_e = {treatment, *surrogates}
    cols_o = {long_term_outcome, *surrogates}
    if cov_list:
        cols_e |= set(cov_list)
        cols_o |= set(cov_list)
    missing_e = cols_e - set(experimental.columns)
    missing_o = cols_o - set(observational.columns)
    _raise_missing_columns("experimental", missing_e)
    _raise_missing_columns("observational", missing_o)

    feat_cols: List[str] = list(surrogates) + cov_list

    # --- Fit f(S) on observational sample -------------------------------
    S_o = _as_matrix(observational, feat_cols)
    Y_o = _require_finite_vector(
        observational[long_term_outcome].to_numpy(dtype=float), long_term_outcome
    )
    predict = _fit_outcome_model(S_o, Y_o, model=model)
    h_pred_obs = predict(S_o)
    resid_obs = Y_o - h_pred_obs

    # --- Predict h on experimental sample and compute ATE ---------------
    S_e = _as_matrix(experimental, feat_cols)
    T_e = experimental[treatment].to_numpy(dtype=float)
    T_bool = _require_binary_treatment(T_e, treatment)
    h = predict(S_e)
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
            raise ConvergenceFailure(
                "Bootstrap produced <10 valid replicates; check treatment "
                "overlap in experimental sample.",
                recovery_hint="Increase n_boot or ensure both treatment arms resample.",
                diagnostics={"valid_replicates": int(boots.size), "n_boot": n_boot},
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
            h=h,
            T=T_e,
            resid_obs=resid_obs,
            h_pred_obs=h_pred_obs,
            n_exp=len(experimental),
            n_obs=len(observational),
        )
        se = float(np.sqrt(max(var, 0.0)))
        z = stats.norm.ppf(1 - alpha / 2)
        ci = (est - z * se, est + z * se)
        pval = 2 * (1 - stats.norm.cdf(abs(est) / se)) if se > 0 else float("nan")

    _result = CausalResult(
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
            "covariates": cov_list,
            "outcome_model": str(model),
            "inference": "bootstrap" if n_boot > 0 else "delta_method",
            "resid_std_obs": float(resid_obs.std(ddof=1)),
        },
    )
    try:
        from ..output._lineage import attach_provenance as _attach_prov

        _attach_prov(
            _result,
            function="sp.surrogate.surrogate_index",
            params={
                "treatment": treatment,
                "surrogates": list(surrogates),
                "long_term_outcome": long_term_outcome,
                "covariates": cov_list if cov_list else None,
                "model": str(model),
                "alpha": alpha,
                "n_boot": n_boot,
                "random_state": random_state,
            },
            data=experimental,
            overwrite=False,
        )
    except Exception:  # pragma: no cover
        pass
    return _result


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
        measured in both samples (and so on for each of the ``K`` waves).

    Notes
    -----
    Uses the iterated-expectation estimator (Ghassami et al. 2024, Eq. 3):

        f_K(s_K) = E[Y | S_K = s_K]                    in observational
        f_{k}(s_k) = E[f_{k+1}(S_{k+1}) | S_k = s_k]   in observational
        ATE = E[ f_1(S_1) | T=1 ] - E[ f_1(S_1) | T=0 ] in experimental

    Inference is bootstrap-based because the iterated delta variance is
    unwieldy in closed form.

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(1)
    >>> # Two surrogate waves: S1 -> S2 -> Y in the observational sample.
    >>> S1 = rng.normal(size=400)
    >>> S2 = 0.9 * S1 + rng.normal(scale=0.4, size=400)
    >>> obs = pd.DataFrame({"S1": S1, "S2": S2,
    ...                     "Y": 1.2 * S2 + rng.normal(scale=0.4, size=400)})
    >>> T = rng.integers(0, 2, size=300)
    >>> S1e = 0.7 * T + rng.normal(size=300)
    >>> exp = pd.DataFrame({"T": T, "S1": S1e,
    ...                     "S2": 0.9 * S1e + rng.normal(scale=0.4, size=300)})
    >>> res = sp.long_term_from_short(
    ...     exp, obs, treatment="T",
    ...     surrogates_waves=[["S1"], ["S2"]],
    ...     long_term_outcome="Y", n_boot=100, random_state=0,
    ... )
    >>> bool(res.estimate > 0)
    True

    References
    ----------
    Tran, A., Bibaut, A., & Kallus, N. (2023). "Inferring the Long-Term
    Causal Effects of Long-Term Treatments from Short-Term Experiments."
    arXiv:2311.08527.
    """
    experimental = _require_dataframe(experimental, "experimental")
    observational = _require_dataframe(observational, "observational")
    treatment = _require_column_name(treatment, "treatment")
    surrogates_waves = _coerce_waves(surrogates_waves)
    long_term_outcome = _require_column_name(long_term_outcome, "long_term_outcome")
    cov_list = _coerce_optional_column_list(covariates, "covariates")
    alpha = _require_open_unit_float(alpha, "alpha")
    n_boot = _require_int_at_least(n_boot, "n_boot", 50)
    flat_surr: List[str] = [c for wave in surrogates_waves for c in wave]
    base_cols_e = {treatment, *flat_surr}
    base_cols_o = {long_term_outcome, *flat_surr}
    if cov_list:
        base_cols_e |= set(cov_list)
        base_cols_o |= set(cov_list)
    missing_e = base_cols_e - set(experimental.columns)
    missing_o = base_cols_o - set(observational.columns)
    _raise_missing_columns("experimental", missing_e)
    _raise_missing_columns("observational", missing_o)

    K = len(surrogates_waves)

    def _point_estimate(obs_df: pd.DataFrame, exp_df: pd.DataFrame) -> float:
        # Backward induction: start with Y on final wave, then iterate.
        current_target = _require_finite_vector(
            obs_df[long_term_outcome].to_numpy(dtype=float), long_term_outcome
        )
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
        T = _require_binary_treatment(
            exp_df[treatment].to_numpy(dtype=float), treatment
        )
        return float(f1_exp[T].mean() - f1_exp[~T].mean())

    est = _point_estimate(observational, experimental)

    rng = np.random.default_rng(random_state)
    n_e, n_o = len(experimental), len(observational)
    boots: np.ndarray = np.empty(n_boot)
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
        raise ConvergenceFailure(
            f"Only {n_ok}/{n_boot} bootstrap replicates succeeded; check "
            "treatment overlap and surrogate availability.",
            recovery_hint="Increase n_boot or inspect treatment/surrogate support.",
            diagnostics={"valid_replicates": int(n_ok), "n_boot": int(n_boot)},
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
# 3) Proximal surrogate index (Imbens-Kallus-Mao-Wang 2025 JRSS-B)
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
    Kallus, Mao & Wang (2025, JRSS-B). In linear-Gaussian form the
    bridge function ``h(s, x)`` solves

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

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(2)
    >>> # Unobserved U confounds S -> Y; proxy W stands in for U.
    >>> U = rng.normal(size=500)
    >>> S = 0.8 * U + rng.normal(scale=0.5, size=500)
    >>> W = 0.9 * U + rng.normal(scale=0.5, size=500)
    >>> obs = pd.DataFrame({"S": S, "W": W,
    ...                     "Y": 1.0 * S + 0.7 * U + rng.normal(scale=0.5, size=500)})
    >>> T = rng.integers(0, 2, size=300)
    >>> exp = pd.DataFrame({"T": T, "S": 0.6 * T + rng.normal(size=300)})
    >>> res = sp.proximal_surrogate_index(
    ...     exp, obs, treatment="T", surrogates=["S"], proxies=["W"],
    ...     long_term_outcome="Y", n_boot=100, random_state=0,
    ... )
    >>> bool(res.estimate > 0)
    True

    References
    ----------
    Imbens, G., Kallus, N., Mao, X., & Wang, Y. (2025).
    "Long-term Causal Inference Under Persistent Confounding via Data
    Combination." Journal of the Royal Statistical Society Series B,
    87(2), 362-388. arXiv:2202.07234. [@imbens2025long]
    """
    experimental = _require_dataframe(experimental, "experimental")
    observational = _require_dataframe(observational, "observational")
    treatment = _require_column_name(treatment, "treatment")
    surrogates = _coerce_column_list(surrogates, "surrogates")
    proxies = _coerce_column_list(proxies, "proxies", allow_empty=True)
    long_term_outcome = _require_column_name(long_term_outcome, "long_term_outcome")
    cov_list = _coerce_optional_column_list(covariates, "covariates")
    alpha = _require_open_unit_float(alpha, "alpha")
    n_boot = _require_int_at_least(n_boot, "n_boot", 50)
    if len(proxies) == 0:
        raise MethodIncompatibility(
            "`proxies` must contain at least one proxy variable W; "
            "otherwise reduce to sp.surrogate.surrogate_index().",
            alternative_functions=["sp.surrogate_index"],
        )
    feat_s = list(surrogates) + cov_list
    cols_o = {long_term_outcome, *surrogates, *proxies, *cov_list}
    cols_e = {treatment, *surrogates, *cov_list}
    missing_o = cols_o - set(observational.columns)
    missing_e = cols_e - set(experimental.columns)
    _raise_missing_columns("observational", missing_o)
    _raise_missing_columns("experimental", missing_e)

    def _bridge_predict(
        obs_df: pd.DataFrame,
    ) -> Callable[[np.ndarray, Optional[np.ndarray]], np.ndarray]:
        """Solve E[Y|S,X,W] = Wα + h(S,X) by 2SLS.

        First stage: regress S on W (and X) → S_hat(W, X)
        Second stage: regress Y on W, S_hat(W, X), X
        Bridge function h(s, x) is defined by the S-coefficients in stage 2
        applied to the *observed* (not projected) s.
        """
        S_o = _as_matrix(obs_df, feat_s)  # n_o × p_s+p_x
        W_o = _as_matrix(obs_df, list(proxies))  # n_o × p_w
        Y_o = _require_finite_vector(
            obs_df[long_term_outcome].to_numpy(dtype=float), long_term_outcome
        )
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

        def predict(
            S_exp: np.ndarray, X_exp: Optional[np.ndarray] = None
        ) -> np.ndarray:
            val = intercept + S_exp @ beta_s
            if beta_x is not None and X_exp is not None:
                val = val + X_exp @ beta_x
            return np.asarray(val)

        return predict

    def _point_estimate(obs_df: pd.DataFrame, exp_df: pd.DataFrame) -> float:
        predict = _bridge_predict(obs_df)
        S_e = _as_matrix(exp_df, list(surrogates))
        X_e = _as_matrix(exp_df, cov_list) if cov_list else None
        h_e = predict(S_e, X_e)
        T = _require_binary_treatment(
            exp_df[treatment].to_numpy(dtype=float), treatment
        )
        return float(h_e[T].mean() - h_e[~T].mean())

    est = _point_estimate(observational, experimental)

    rng = np.random.default_rng(random_state)
    n_e, n_o = len(experimental), len(observational)
    boots: np.ndarray = np.empty(n_boot)
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
        raise ConvergenceFailure(
            f"Only {n_ok}/{n_boot} bootstrap replicates succeeded; check "
            "proxy strength and overlap.",
            recovery_hint="Increase n_boot or inspect proxy strength and overlap.",
            diagnostics={"valid_replicates": int(n_ok), "n_boot": int(n_boot)},
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
