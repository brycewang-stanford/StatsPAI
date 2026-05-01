"""LLM-annotator measurement-error correction (Egami et al. 2024).

When a downstream causal estimate uses a *treatment* indicator that
came from an LLM (or any imperfect classifier) rather than from a
human, the resulting OLS / IPW / DR coefficient is *attenuated* by the
LLM's misclassification rate.  Egami, Hinck, Stewart & Wei (2024)
formalise this and propose corrections that recover the true
coefficient when a small human-validated subset is available.

This module implements two correction paths:

* **Binary treatment** — Hausman-style correction (Aigner 1973;
  Hausman, Abrevaya & Scott-Morton 1998).  The key identity is

      β_obs = (1 - p_01 - p_10) · β_true

  where ``p_01 = P(T_obs=1 | T_true=0)`` and
  ``p_10 = P(T_obs=0 | T_true=1)``.  Estimate both rates on the
  human-validated subset and divide.

* **Multi-class treatment** (``K ≥ 3``) — inverse-confusion-matrix
  correction.  The KxK confusion matrix ``M[i, j] = P(T_obs=j |
  T_true=i)`` and the validation-set marginals ``π[i] = P(T_true=i)``
  pin down the conditional distribution
  ``Q[i, j] = P(T_true=i | T_obs=j)`` via Bayes.  Under non-
  differential measurement error,

      θ_obs = T · θ_true,

  where ``θ`` collects the dummy-coded coefficients and ``T`` is a
  KxK transformation built from ``Q`` (intercept and K-1 contrasts).
  Recovering ``θ_true = T⁻¹ θ_obs`` yields per-class corrected
  contrasts; the head-line ``.estimate`` uses the smallest non-
  reference class and the full vector ships in ``.detail``.

Standard-error options
----------------------
The first-order standard error divides ``se_naive`` by the same
attenuation (binary) or applies the linear transformation ``T⁻¹`` to
the naive covariance (multi-class).  Both ignore validation-set
sampling noise.

Two extras lift this:

1. ``model_info['se_inflation_factor']`` — multiplicative factor (>= 1)
   the user can apply to the first-order SE for an honest delta-method
   accounting of validation-set noise.  Always populated.
2. ``bootstrap=True`` — joint resample of the full sample
   (validation rows + unlabeled rows), re-run the entire correction
   pipeline ``n_bootstrap`` times, and report bias-corrected
   percentile-bootstrap CIs.  Replaces ``.ci`` / ``.se`` on the
   returned object; the first-order versions remain available in
   ``model_info['first_order_se']`` / ``model_info['first_order_ci']``.

References
----------
Egami, N., Hinck, M., Stewart, B., & Wei, H. (2024). "Using imperfect
surrogates for downstream inference: Design-based supervised learning
for social science applications of large language models."  *NeurIPS*.
arXiv:2306.04746. [@egami2023imperfect]

Hausman, J., Abrevaya, J., & Scott-Morton, F. (1998). "Misclassification
of the dependent variable in a discrete-response setting."  *Journal of
Econometrics*, 87, 239–269. [@hausman1998misclassification]
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from ..core.results import CausalResult


__all__ = ['llm_annotator_correct', 'LLMAnnotatorResult']


# ----------------------------------------------------------------------
# Result container
# ----------------------------------------------------------------------


class LLMAnnotatorResult(CausalResult):
    """Output of :func:`llm_annotator_correct`.

    Inherits the agent-native CausalResult API.  Adds annotator-specific
    fields ``naive_estimate``, ``correction_factor``, and
    ``annotator_diagnostics`` (false-positive / false-negative rates,
    validation-set size, agreement rate, optional confusion matrix and
    inflation factor) on the instance.
    """

    def __init__(
        self, *, method: str, estimand: str, estimate: float, se: float,
        pvalue: float, ci: tuple, alpha: float, n_obs: int,
        naive_estimate: float, naive_se: float,
        correction_factor: float,
        annotator_diagnostics: Dict[str, Any],
        detail: Optional[pd.DataFrame] = None,
        model_info: Optional[Dict[str, Any]] = None,
    ):
        # Flatten annotator diagnostics into model_info so they show up
        # via CausalResult's inherited `.diagnostics` property; also
        # keep the self-contained sub-dict for explicit access.
        flat = dict(annotator_diagnostics)
        mi = dict(model_info or {})
        mi.update(flat)
        mi['llm_annotator_diagnostics'] = dict(annotator_diagnostics)
        super().__init__(
            method=method, estimand=estimand, estimate=estimate, se=se,
            pvalue=pvalue, ci=ci, alpha=alpha, n_obs=n_obs,
            detail=detail, model_info=mi,
        )
        self.naive_estimate = float(naive_estimate)
        self.naive_se = float(naive_se)
        self.correction_factor = float(correction_factor)
        self.annotator_diagnostics = dict(annotator_diagnostics)

    def summary(self) -> str:  # pragma: no cover (cosmetic)
        d = self.annotator_diagnostics
        n_classes = d.get('n_classes', 2)
        lines = [
            "LLMAnnotatorResult",
            "=" * 60,
            f"  Method            : {self.method}",
            f"  Estimand          : {self.estimand}",
            f"  N classes         : {n_classes}",
            f"  Naive estimate    : {self.naive_estimate:.4f} "
            f"(SE = {self.naive_se:.4f})",
            f"  Correction factor : {self.correction_factor:.4f}",
            f"  Corrected estimate: {self.estimate:.4f} "
            f"(SE = {self.se:.4f})",
            f"  {int((1-self.alpha)*100)}% CI            : "
            f"[{self.ci[0]:.4f}, {self.ci[1]:.4f}]",
            f"  p-value           : {self.pvalue:.4f}",
            f"  N obs             : {self.n_obs}",
            f"  Validation N      : {d.get('n_validation', 'NA')}",
            f"  Agreement rate    : {d.get('agreement', float('nan')):.4f}",
        ]
        if n_classes == 2:
            lines.append(
                f"  P(T_obs=1|T=0)    : "
                f"{d.get('p_01', float('nan')):.4f}"
            )
            lines.append(
                f"  P(T_obs=0|T=1)    : "
                f"{d.get('p_10', float('nan')):.4f}"
            )
        infl = d.get('se_inflation_factor')
        if infl is not None and np.isfinite(infl):
            lines.append(
                f"  SE inflation      : x{infl:.3f} "
                "(apply for validation-set-aware SE)"
            )
        se_corr = d.get('se_correction', 'first_order')
        lines.append(f"  SE correction     : {se_corr}")
        if d.get('status'):
            lines.append(f"  Status            : {d['status']}")
        return "\n".join(lines)


# ----------------------------------------------------------------------
# Input validation + frame assembly
# ----------------------------------------------------------------------


def _validate_inputs(
    annotations_llm: pd.Series,
    annotations_human: Optional[pd.Series],
    outcome: Optional[pd.Series],
    covariates: Optional[pd.DataFrame],
) -> None:
    if annotations_human is None:
        from ..exceptions import DataInsufficient
        raise DataInsufficient(
            "annotations_human is required for measurement-error "
            "correction. Pass a Series with NaN where the human label "
            "is missing."
        )
    if outcome is None:
        raise ValueError("outcome series is required.")


def _build_frames(
    annotations_llm: pd.Series,
    annotations_human: pd.Series,
    outcome: pd.Series,
    covariates: Optional[pd.DataFrame],
) -> Tuple[pd.DataFrame, pd.DataFrame, List[str]]:
    """Return (full_sample_df, validation_df, covariate_columns).

    ``full_sample_df`` contains every row with non-NaN T_llm, y, and
    covariates. ``validation_df`` is the further subset whose human
    label is also non-NaN.
    """
    df = pd.DataFrame({
        'T_llm': annotations_llm.values,
        'T_human': annotations_human.values,
        'y': outcome.values,
    })
    if covariates is not None:
        cov_arr = covariates.reset_index(drop=True)
        for c in cov_arr.columns:
            df[f'_cov_{c}'] = cov_arr[c].values
    cov_cols = [c for c in df.columns if c.startswith('_cov_')]

    use_full = df.dropna(subset=['T_llm', 'y']).copy()
    if cov_cols:
        use_full = use_full[use_full[cov_cols].notna().all(axis=1)]
    if len(use_full) < 20:
        from ..exceptions import DataInsufficient
        raise DataInsufficient(
            f"At least 20 rows required for the OLS step; got "
            f"{len(use_full)}."
        )

    val = use_full.dropna(subset=['T_human']).copy()
    if len(val) < 30:
        from ..exceptions import DataInsufficient
        raise DataInsufficient(
            f"At least 30 validation rows (with both LLM and human "
            f"labels) recommended for stable correction; got {len(val)}."
        )
    return use_full, val, cov_cols


# ----------------------------------------------------------------------
# Naive OLS with HC1 SEs (binary T_llm regressor or K-1 dummies)
# ----------------------------------------------------------------------


def _ols_hc1(
    X: np.ndarray, y: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Solve OLS with HC1 (small-sample-corrected) sandwich covariance.

    Returns ``(beta, cov)``.
    """
    n, p = X.shape
    XtX = X.T @ X
    try:
        XtX_inv = np.linalg.pinv(XtX)
    except np.linalg.LinAlgError as exc:
        raise RuntimeError(
            f"OLS normal equations singular: {exc}"
        )
    beta = XtX_inv @ (X.T @ y)
    resid = y - X @ beta
    df_resid = max(n - p, 1)
    Omega = (X.T * (resid ** 2)) @ X
    cov = (n / df_resid) * (XtX_inv @ Omega @ XtX_inv)
    return beta, cov


def _design_binary(
    df: pd.DataFrame, cov_cols: List[str]
) -> Tuple[np.ndarray, np.ndarray, int]:
    """Design matrix for binary path: [1, T_llm, X]. Returns (X, y,
    treat_idx)."""
    n = len(df)
    cov_mat = (df[cov_cols].astype(np.float64).values if cov_cols
               else np.zeros((n, 0)))
    X = np.hstack([
        np.ones((n, 1)),
        df['T_llm'].astype(np.float64).values.reshape(-1, 1),
        cov_mat,
    ])
    y = df['y'].astype(np.float64).values
    return X, y, 1  # treatment column index


def _design_multiclass(
    df: pd.DataFrame, cov_cols: List[str], classes: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Design matrix for K-class path: [1, D_1, ..., D_{K-1}, X].

    Reference category is ``classes[0]`` (the smallest label).  Returns
    ``(X, y, treat_idx)`` where ``treat_idx`` is a length-(K-1) array
    of column positions for the treatment dummies.
    """
    n = len(df)
    K = len(classes)
    t_obs = df['T_llm'].values
    dummies = np.zeros((n, K - 1), dtype=np.float64)
    for k in range(1, K):
        dummies[:, k - 1] = (t_obs == classes[k]).astype(np.float64)
    cov_mat = (df[cov_cols].astype(np.float64).values if cov_cols
               else np.zeros((n, 0)))
    X = np.hstack([
        np.ones((n, 1)),
        dummies,
        cov_mat,
    ])
    y = df['y'].astype(np.float64).values
    treat_idx = np.arange(1, K)  # columns for D_1..D_{K-1}
    return X, y, treat_idx


# ----------------------------------------------------------------------
# Confusion matrix + transformation
# ----------------------------------------------------------------------


def _confusion(
    val: pd.DataFrame, classes: np.ndarray
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute the K x K confusion matrix M[i,j] = P(T_obs=j | T_true=i)
    and validation marginals ``π[i] = P(T_true=i)``.

    Both human and LLM labels are aligned to ``classes`` order.  Empty
    rows of ``M`` (a true class never observed in validation) raise.
    """
    K = len(classes)
    th = val['T_human'].values
    tl = val['T_llm'].values
    M = np.zeros((K, K))
    pi = np.zeros(K)
    for i, ci in enumerate(classes):
        mask_i = (th == ci)
        n_i = int(mask_i.sum())
        pi[i] = n_i / len(val)
        if n_i == 0:
            from ..exceptions import DataInsufficient
            raise DataInsufficient(
                f"Validation sample lacks the class {ci!r} of T_human "
                "(every true class must appear at least once)."
            )
        for j, cj in enumerate(classes):
            M[i, j] = float(((th == ci) & (tl == cj)).sum() / n_i)
    return M, pi


def _bayes_posterior(M: np.ndarray, pi: np.ndarray) -> np.ndarray:
    """Compute Q[i, j] = P(T_true=i | T_obs=j) via Bayes' rule.

    ``Q.shape == M.shape``.  Columns of ``Q`` sum to 1.  If a column of
    ``M`` is identically zero (a class never produced by the LLM), the
    corresponding ``Q`` column is left as zeros — downstream callers
    must ensure such classes do not appear in the OLS dummy set.
    """
    p_obs = pi @ M  # P(T_obs = j) for each j
    Q = np.zeros_like(M)
    for j in range(M.shape[1]):
        if p_obs[j] > 0:
            Q[:, j] = M[:, j] * pi / p_obs[j]
    return Q


def _coef_transform(Q: np.ndarray) -> np.ndarray:
    """Build the (K x K) coefficient transformation T such that
    ``θ_obs = T @ θ_true`` when the regressors are an intercept + K-1
    dummies for non-reference classes (reference = class 0).

    Block layout::

        T[0, 0] = 1
        T[0, i+1] = Q[i, 0]                  for i = 1..K-1
        T[k, 0] = 0                          for k = 1..K-1
        T[k, i+1] = Q[i, k] - Q[i, 0]        for k, i = 1..K-1
    """
    K = Q.shape[0]
    T = np.zeros((K, K))
    T[0, 0] = 1.0
    for i in range(1, K):
        T[0, i] = Q[i, 0]
    for k in range(1, K):
        for i in range(1, K):
            T[k, i] = Q[i, k] - Q[i, 0]
    return T


# ----------------------------------------------------------------------
# Binary correction (Hausman 1998)
# ----------------------------------------------------------------------


def _correct_binary(
    use_full: pd.DataFrame, val: pd.DataFrame, cov_cols: List[str],
    alpha: float,
) -> Dict[str, Any]:
    """Binary-T Hausman correction. Returns a dict with the result
    payload (estimate, se, ci, pvalue, naive_*, diagnostics, detail).
    """
    val_t_human = val['T_human'].astype(int).values
    val_t_llm = val['T_llm'].astype(int).values
    n_human0 = int((val_t_human == 0).sum())
    n_human1 = int((val_t_human == 1).sum())
    if n_human0 == 0 or n_human1 == 0:
        from ..exceptions import DataInsufficient
        raise DataInsufficient(
            "Validation sample lacks both human-label classes "
            "(need at least one row each of T_human=0 and T_human=1)."
        )
    p_01 = float(
        ((val_t_human == 0) & (val_t_llm == 1)).sum() / n_human0
    )
    p_10 = float(
        ((val_t_human == 1) & (val_t_llm == 0)).sum() / n_human1
    )
    correction_factor = 1.0 - p_01 - p_10
    if correction_factor <= 0:
        from ..exceptions import IdentificationFailure
        raise IdentificationFailure(
            f"Misclassification rates (p_01={p_01:.3f}, "
            f"p_10={p_10:.3f}) imply the LLM label has no information "
            "about the true treatment (1 - p_01 - p_10 <= 0). "
            "Correction is not identified; consider re-prompting the "
            "LLM or hand-labelling."
        )
    agreement = float((val_t_llm == val_t_human).mean())

    X, y, treat_idx = _design_binary(use_full, cov_cols)
    n = len(use_full)
    beta, cov = _ols_hc1(X, y)
    se_naive = float(np.sqrt(max(cov[treat_idx, treat_idx], 0.0)))
    naive_estimate = float(beta[treat_idx])

    corrected_estimate = naive_estimate / correction_factor
    corrected_se = se_naive / abs(correction_factor)

    # Validation-set inflation (delta method).  Variance of p_01_hat is
    # p_01 (1-p_01) / n_human0 by binomial sampling; same for p_10.
    var_p01 = p_01 * (1.0 - p_01) / max(n_human0, 1)
    var_p10 = p_10 * (1.0 - p_10) / max(n_human1, 1)
    extra_var = (corrected_estimate ** 2) * (var_p01 + var_p10)
    base_var = corrected_se ** 2
    if base_var > 0:
        infl = float(np.sqrt(1.0 + extra_var / base_var))
    else:
        infl = float('nan')

    diag: Dict[str, Any] = {
        'p_01': p_01,
        'p_10': p_10,
        'p_01_se': float(np.sqrt(var_p01)),
        'p_10_se': float(np.sqrt(var_p10)),
        'correction_factor': float(correction_factor),
        'agreement': agreement,
        'n_validation': int(len(val)),
        'n_full': int(n),
        'n_classes': 2,
        'classes': [0, 1],
        'se_inflation_factor': infl,
        'se_correction': 'first_order',
    }
    return {
        'estimate': corrected_estimate,
        'se': corrected_se,
        'naive_estimate': naive_estimate,
        'naive_se': se_naive,
        'correction_factor': float(correction_factor),
        'diag': diag,
        'detail': None,
    }


# ----------------------------------------------------------------------
# Multi-class correction (inverse confusion matrix)
# ----------------------------------------------------------------------


def _correct_multiclass(
    use_full: pd.DataFrame, val: pd.DataFrame, cov_cols: List[str],
    classes: np.ndarray, alpha: float,
) -> Dict[str, Any]:
    """K-class inverse-confusion-matrix correction.  Returns a payload
    dict; ``estimate`` corresponds to the smallest non-reference class
    (``classes[1]``), and the full per-class corrected vector ships in
    ``detail``.
    """
    K = len(classes)
    M, pi = _confusion(val, classes)
    Q = _bayes_posterior(M, pi)
    T = _coef_transform(Q)
    try:
        T_inv = np.linalg.inv(T)
    except np.linalg.LinAlgError as exc:
        from ..exceptions import IdentificationFailure
        raise IdentificationFailure(
            "Coefficient transformation matrix is singular "
            f"({exc}). LLM labels are uninformative about T_true; "
            "correction is not identified."
        )
    # Numerical guard: if T's smallest singular value is tiny relative
    # to its largest, treat as ill-conditioned (near-unidentified).
    sv = np.linalg.svd(T, compute_uv=False)
    if sv.min() / max(sv.max(), 1e-300) < 1e-8:
        from ..exceptions import IdentificationFailure
        raise IdentificationFailure(
            f"Coefficient transformation matrix is near-singular "
            f"(cond={sv.max() / max(sv.min(), 1e-300):.2e}); "
            "the LLM is essentially uninformative for some class."
        )

    X, y, treat_idx = _design_multiclass(use_full, cov_cols, classes)
    n = len(use_full)
    beta, cov = _ols_hc1(X, y)

    # Build the K-vector θ_obs = [α; β_1; ...; β_{K-1}] from the OLS
    # coefficients (intercept + K-1 dummies; covariates trail).
    theta_obs = np.empty(K)
    theta_obs[0] = beta[0]
    for k in range(1, K):
        theta_obs[k] = beta[treat_idx[k - 1]]

    # Selection matrix S (KxP) so that θ_obs = S @ beta — covariate
    # coefficients are *not* affected by the correction (non-
    # differential ME assumption); we only transform [α; β_1..β_{K-1}].
    P = X.shape[1]
    S = np.zeros((K, P))
    S[0, 0] = 1.0
    for k in range(1, K):
        S[k, treat_idx[k - 1]] = 1.0

    cov_theta_obs = S @ cov @ S.T
    cov_theta_true = T_inv @ cov_theta_obs @ T_inv.T
    theta_true = T_inv @ theta_obs

    naive_betas = theta_obs[1:].copy()      # K-1 contrasts
    corrected_betas = theta_true[1:].copy()
    se_betas_naive = np.sqrt(np.maximum(np.diag(cov_theta_obs)[1:], 0.0))
    se_betas_corrected = np.sqrt(
        np.maximum(np.diag(cov_theta_true)[1:], 0.0)
    )

    # Headline contrast: smallest non-reference class.
    head_idx = 0
    naive_estimate = float(naive_betas[head_idx])
    se_naive = float(se_betas_naive[head_idx])
    corrected_estimate = float(corrected_betas[head_idx])
    corrected_se = float(se_betas_corrected[head_idx])
    correction_factor = (
        naive_estimate / corrected_estimate
        if abs(corrected_estimate) > 1e-12 else float('nan')
    )

    # Detail frame: per-class corrected contrasts.
    detail = pd.DataFrame({
        'class': [classes[k] for k in range(1, K)],
        'naive_estimate': naive_betas,
        'naive_se': se_betas_naive,
        'corrected_estimate': corrected_betas,
        'corrected_se': se_betas_corrected,
    })
    detail['t'] = detail['corrected_estimate'] / detail['corrected_se']
    z = sp_stats.norm.ppf(1 - alpha / 2)
    detail['ci_low'] = (
        detail['corrected_estimate'] - z * detail['corrected_se']
    )
    detail['ci_high'] = (
        detail['corrected_estimate'] + z * detail['corrected_se']
    )
    detail['pvalue'] = 2 * (
        1 - sp_stats.norm.cdf(np.abs(detail['t']))
    )

    # Inflation factor (delta method on M).  For multi-class we only
    # report a coarse upper-bound version: treat each off-diagonal
    # M[i, j] as an independent multinomial estimate with variance
    # M[i,j](1-M[i,j])/(pi[i] * n_val), and use the Frobenius norm of
    # ∂θ_true/∂vec(M) as a scalar inflation. This is a heuristic — for
    # honest CIs use bootstrap=True.
    n_val = len(val)
    var_M_entries = M * (1.0 - M) / np.maximum(pi[:, None] * n_val, 1.0)
    # Squared sensitivity ‖∂θ_true,head / ∂M‖_F^2 via finite differ.
    head_row = head_idx + 1
    eps = 1e-4
    sens_sq = 0.0
    for i in range(K):
        for j in range(K):
            if M[i, j] == 0 and (i != j):
                continue
            M_pert = M.copy()
            # Perturb M[i, j] keeping row i a probability vector by
            # subtracting eps from M[i, i] (or another non-zero entry
            # in the same row when i == j).
            offset_target = i if i != j else (j + 1) % K
            M_pert[i, j] += eps
            M_pert[i, offset_target] -= eps
            try:
                Q_p = _bayes_posterior(M_pert, pi)
                T_p = _coef_transform(Q_p)
                T_p_inv = np.linalg.inv(T_p)
                theta_p = T_p_inv @ theta_obs
                d = (theta_p[head_row] - theta_true[head_row]) / eps
            except np.linalg.LinAlgError:
                d = 0.0
            sens_sq += (d ** 2) * var_M_entries[i, j]
    base_var = corrected_se ** 2
    if base_var > 0 and np.isfinite(sens_sq) and sens_sq >= 0:
        infl = float(np.sqrt(1.0 + sens_sq / base_var))
    else:
        infl = float('nan')

    diag: Dict[str, Any] = {
        'confusion_matrix': M.tolist(),
        'classes': [
            int(c) if isinstance(c, (np.integer, int)) else c
            for c in classes.tolist()
        ],
        'n_classes': int(K),
        'pi_validation': pi.tolist(),
        'q_posterior': Q.tolist(),
        'transform_matrix': T.tolist(),
        'condition_number': float(sv.max() / max(sv.min(), 1e-300)),
        'agreement': float(
            (val['T_human'].values == val['T_llm'].values).mean()
        ),
        'n_validation': int(len(val)),
        'n_full': int(n),
        'correction_factor': float(correction_factor),
        'se_inflation_factor': infl,
        'se_correction': 'first_order',
        'headline_contrast': (
            f"class={classes[head_idx + 1]} vs ref={classes[0]}"
        ),
    }

    return {
        'estimate': corrected_estimate,
        'se': corrected_se,
        'naive_estimate': naive_estimate,
        'naive_se': se_naive,
        'correction_factor': float(correction_factor),
        'diag': diag,
        'detail': detail,
    }


# ----------------------------------------------------------------------
# Bootstrap (joint resampling of validation + unlabeled rows)
# ----------------------------------------------------------------------


def _run_pipeline(
    use_full: pd.DataFrame, cov_cols: List[str],
    classes: np.ndarray, n_classes: int, alpha: float,
) -> Optional[float]:
    """Re-execute the entire correction pipeline on a (resampled)
    full-sample DataFrame.  Returns the headline estimate, or ``None``
    if the resample is degenerate (e.g. a class missing from the
    validation set).  Used by the bootstrap loop.
    """
    val = use_full.dropna(subset=['T_human'])
    if len(val) < 2:
        return None
    try:
        if n_classes == 2:
            payload = _correct_binary(
                use_full, val, cov_cols, alpha,
            )
        else:
            payload = _correct_multiclass(
                use_full, val, cov_cols, classes, alpha,
            )
    except Exception:
        return None
    return float(payload['estimate'])


def _bootstrap_ci(
    use_full: pd.DataFrame, cov_cols: List[str],
    classes: np.ndarray, n_classes: int, alpha: float,
    point: float, n_bootstrap: int, seed: Optional[int],
) -> Tuple[float, float, float, np.ndarray, int]:
    """Joint resample the full sample and re-run the pipeline; report
    bias-corrected percentile bootstrap CIs.

    Returns ``(se_boot, ci_low, ci_high, draws, n_failed)``.
    """
    rng = np.random.default_rng(seed)
    n = len(use_full)
    arr_full = use_full.reset_index(drop=True)
    draws = np.empty(n_bootstrap)
    n_failed = 0
    for b in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot_df = arr_full.iloc[idx].reset_index(drop=True)
        est = _run_pipeline(
            boot_df, cov_cols, classes, n_classes, alpha,
        )
        if est is None or not np.isfinite(est):
            draws[b] = np.nan
            n_failed += 1
        else:
            draws[b] = est
    valid = draws[~np.isnan(draws)]
    if len(valid) < max(50, n_bootstrap // 4):
        from ..exceptions import DataInsufficient
        raise DataInsufficient(
            f"Bootstrap produced only {len(valid)} valid draws out of "
            f"{n_bootstrap}; correction is too unstable for resampling."
        )
    se_boot = float(np.std(valid, ddof=1))
    # Bias-corrected percentile bootstrap (Efron & Tibshirani 1993, §14).
    p0 = float((valid < point).mean())
    p0 = min(max(p0, 1.0 / (len(valid) + 1)),
             1.0 - 1.0 / (len(valid) + 1))
    z0 = sp_stats.norm.ppf(p0)
    z_a = sp_stats.norm.ppf(alpha / 2)
    z_b = sp_stats.norm.ppf(1 - alpha / 2)
    a_lo = sp_stats.norm.cdf(2 * z0 + z_a)
    a_hi = sp_stats.norm.cdf(2 * z0 + z_b)
    ci_lo = float(np.quantile(valid, a_lo))
    ci_hi = float(np.quantile(valid, a_hi))
    return se_boot, ci_lo, ci_hi, valid, n_failed


# ----------------------------------------------------------------------
# Public entry point
# ----------------------------------------------------------------------


def llm_annotator_correct(
    *,
    annotations_llm: pd.Series,
    outcome: pd.Series,
    annotations_human: Optional[pd.Series] = None,
    covariates: Optional[pd.DataFrame] = None,
    method: str = 'hausman',
    bootstrap: bool = False,
    n_bootstrap: int = 500,
    bootstrap_seed: Optional[int] = None,
    alpha: float = 0.05,
) -> LLMAnnotatorResult:
    """Correct a downstream causal coefficient for LLM annotation noise.

    Implements two correction paths sharing the same API:

    * **Binary T** — Hausman (1998) ``β_corrected = β_obs / (1 - p_01
      - p_10)``.
    * **Multi-class T** (``K ≥ 3``) — inverse-confusion-matrix
      correction.  The treatment is dummy-encoded with the smallest
      label as reference; per-class contrasts are recovered via a
      linear transformation built from the validation-set Bayes
      posterior ``Q[i, j] = P(T_true=i | T_obs=j)``.

    Parameters
    ----------
    annotations_llm : pd.Series
        LLM-derived annotation for every row.  Binary or multi-class
        (numeric class labels).
    outcome : pd.Series
        Outcome variable.
    annotations_human : pd.Series, optional
        Human annotation; ``NaN`` where unavailable.  At least 30 rows
        with both LLM and human labels are required, and every true
        class must appear at least once.
    covariates : pd.DataFrame, optional
        Additional control variables for the OLS regression.
    method : {'hausman'}, default 'hausman'
        Correction method.  Future versions will add full SAR with
        super learners (Egami et al. 2024 §3).
    bootstrap : bool, default False
        If True, run a bias-corrected percentile bootstrap that
        resamples the full sample (validation rows + unlabeled rows
        jointly) ``n_bootstrap`` times.  CIs / SE on the result reflect
        validation-set sampling uncertainty; the first-order versions
        ship in ``model_info['first_order_se']`` and
        ``['first_order_ci']``.
    n_bootstrap : int, default 500
        Bootstrap replicates.
    bootstrap_seed : int, optional
        Seed for the bootstrap RNG (NumPy ``default_rng``).
    alpha : float, default 0.05
        CI level (1 - alpha confidence).

    Returns
    -------
    LLMAnnotatorResult

    Examples
    --------
    >>> import statspai as sp, pandas as pd, numpy as np
    >>> n, n_val = 1000, 100
    >>> rng = np.random.default_rng(0)
    >>> T_true = (rng.random(n) > 0.5).astype(int)
    >>> noise = (rng.random(n) < 0.15).astype(int)
    >>> T_llm = (T_true ^ noise).astype(int)            # 15% misclass.
    >>> y = 1.0 * T_true + rng.standard_normal(n)        # true ATE 1.0
    >>> human = pd.Series([T_true[i] if i < n_val else np.nan
    ...                    for i in range(n)])
    >>> r = sp.llm_annotator_correct(
    ...     annotations_llm=pd.Series(T_llm),
    ...     annotations_human=human,
    ...     outcome=pd.Series(y),
    ... )
    >>> r.estimate    # ~1.0 (corrected from naive ~0.7)

    References
    ----------
    Egami, Hinck, Stewart & Wei (NeurIPS 2024) — arXiv:2306.04746.
    Hausman, Abrevaya & Scott-Morton (J. Econometrics 1998).
    """
    if method not in {'hausman'}:
        raise ValueError(
            f"Unknown method={method!r}. Currently supported: 'hausman'."
        )
    _validate_inputs(
        annotations_llm, annotations_human, outcome, covariates,
    )
    use_full, val, cov_cols = _build_frames(
        annotations_llm, annotations_human, outcome, covariates,
    )

    # Determine class set from the union of LLM and human labels.
    classes_llm = pd.Series(use_full['T_llm']).dropna().unique()
    classes_h = pd.Series(val['T_human']).dropna().unique()
    classes_all = np.unique(np.concatenate([classes_llm, classes_h]))
    classes_all = np.sort(classes_all)
    n_classes = len(classes_all)
    if n_classes < 2:
        from ..exceptions import DataInsufficient
        raise DataInsufficient(
            f"Need at least 2 distinct classes in (T_llm ∪ T_human); "
            f"got {n_classes}."
        )

    if n_classes == 2:
        payload = _correct_binary(use_full, val, cov_cols, alpha)
    else:
        payload = _correct_multiclass(
            use_full, val, cov_cols, classes_all, alpha,
        )

    point = float(payload['estimate'])
    first_order_se = float(payload['se'])
    z = sp_stats.norm.ppf(1 - alpha / 2)
    fo_ci_lo = point - z * first_order_se
    fo_ci_hi = point + z * first_order_se

    diag = dict(payload['diag'])
    diag['method'] = method
    diag['method_family'] = 'llm-annotator-mec (Egami et al. 2024)'
    diag['first_order_se'] = first_order_se
    diag['first_order_ci'] = (float(fo_ci_lo), float(fo_ci_hi))

    if bootstrap:
        if n_bootstrap < 50:
            raise ValueError(
                f"n_bootstrap={n_bootstrap} is too small; require >= 50 "
                "for stable bias correction."
            )
        se_b, lo_b, hi_b, draws, n_failed = _bootstrap_ci(
            use_full, cov_cols, classes_all, n_classes, alpha,
            point=point, n_bootstrap=n_bootstrap, seed=bootstrap_seed,
        )
        report_se = se_b
        ci_lo, ci_hi = lo_b, hi_b
        diag['bootstrap'] = {
            'n_bootstrap': int(n_bootstrap),
            'n_valid': int(len(draws)),
            'n_failed': int(n_failed),
            'seed': bootstrap_seed,
            'method': 'bias_corrected_percentile',
            'se': float(se_b),
            'ci': (float(lo_b), float(hi_b)),
            'mean': float(np.mean(draws)),
            'median': float(np.median(draws)),
        }
        diag['se_correction'] = 'bias_corrected_bootstrap'
    else:
        report_se = first_order_se
        ci_lo, ci_hi = fo_ci_lo, fo_ci_hi

    if report_se > 0:
        pval = float(2 * (1 - sp_stats.norm.cdf(abs(point / report_se))))
    else:
        pval = float('nan')

    diag['status'] = 'experimental'

    return LLMAnnotatorResult(
        method='llm_annotator_correct',
        estimand='ATE' if n_classes == 2 else 'ATE_first_contrast',
        estimate=point,
        se=float(report_se),
        pvalue=pval,
        ci=(float(ci_lo), float(ci_hi)),
        alpha=alpha,
        n_obs=int(payload['diag']['n_full']),
        naive_estimate=payload['naive_estimate'],
        naive_se=payload['naive_se'],
        correction_factor=payload['correction_factor'],
        annotator_diagnostics=diag,
        detail=payload['detail'],
    )
