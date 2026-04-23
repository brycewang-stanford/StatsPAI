"""LLM-annotator measurement-error correction (Egami et al. 2024) — MVP.

When a downstream causal estimate uses a *treatment* indicator that
came from an LLM (or any imperfect classifier) rather than from a
human, the resulting OLS / IPW / DR coefficient is *attenuated* by the
LLM's misclassification rate.  Egami, Hinck, Stewart & Wei (2024)
formalise this and propose corrections that recover the true
coefficient when a small human-validated subset is available.

This module implements the simplest defensible version: a Hausman-style
correction for **binary** treatment misclassification.  The key
identity (Aigner 1973; Hausman, Abrevaya & Scott-Morton 1998) is that
for OLS of ``y`` on a binary treatment ``T_obs``,

    β_obs = (1 - p_01 - p_10) · β_true

where ``p_01 = P(T_obs=1 | T_true=0)`` and ``p_10 = P(T_obs=0 |
T_true=1)`` are the false-positive and false-negative rates.  Estimate
both rates on the human-validated subset, then correct:

    β_corrected = β_obs / (1 - p_01 - p_10)

A first-order standard error correction divides ``se_obs`` by the same
attenuation factor (this ignores the additional uncertainty from the
finite validation set; the result flags this with a diagnostics
``se_correction='first_order'`` entry so callers can apply a
multiplicative inflation if needed).

Status: **experimental.**  More sophisticated approaches (full SAR
with super learners; bias-corrected bootstrap) are deferred to v1.7.

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

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from ..core.results import CausalResult


__all__ = ['llm_annotator_correct', 'LLMAnnotatorResult']


class LLMAnnotatorResult(CausalResult):
    """Output of :func:`llm_annotator_correct`.

    Inherits the agent-native CausalResult API.  Adds annotator-specific
    fields ``naive_estimate``, ``correction_factor``, and
    ``annotator_diagnostics`` (false-positive / false-negative rates,
    validation-set size, agreement rate) on the instance.
    """

    def __init__(
        self, *, method: str, estimand: str, estimate: float, se: float,
        pvalue: float, ci: tuple, alpha: float, n_obs: int,
        naive_estimate: float, naive_se: float,
        correction_factor: float,
        annotator_diagnostics: Dict[str, Any],
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
            model_info=mi,
        )
        self.naive_estimate = float(naive_estimate)
        self.naive_se = float(naive_se)
        self.correction_factor = float(correction_factor)
        self.annotator_diagnostics = dict(annotator_diagnostics)

    def summary(self) -> str:  # pragma: no cover (cosmetic)
        d = self.annotator_diagnostics
        lines = [
            "LLMAnnotatorResult (experimental)",
            "=" * 60,
            f"  Method            : {self.method}",
            f"  Estimand          : {self.estimand}",
            f"  Naive estimate    : {self.naive_estimate:.4f} "
            f"(SE = {self.naive_se:.4f})",
            f"  Correction factor : {self.correction_factor:.4f}",
            f"  Corrected estimate: {self.estimate:.4f} "
            f"(SE = {self.se:.4f})",
            f"  95% CI            : [{self.ci[0]:.4f}, {self.ci[1]:.4f}]",
            f"  p-value           : {self.pvalue:.4f}",
            f"  N obs             : {self.n_obs}",
            f"  Validation N      : {d.get('n_validation', 'NA')}",
            f"  Agreement rate    : {d.get('agreement', float('nan')):.4f}",
            f"  P(T_obs=1|T=0)    : {d.get('p_01', float('nan')):.4f}",
            f"  P(T_obs=0|T=1)    : {d.get('p_10', float('nan')):.4f}",
            "  SE correction     : first_order (validation-set noise "
            "ignored)",
            "  Status            : experimental",
        ]
        return "\n".join(lines)


def _validate_inputs(
    annotations_llm: pd.Series,
    annotations_human: Optional[pd.Series],
    outcome: Optional[pd.Series],
    covariates: Optional[pd.DataFrame],
) -> None:
    if annotations_human is None:
        from ..exceptions import DataInsufficient
        raise DataInsufficient(
            "annotations_human is required for measurement-error correction. "
            "Pass a Series with NaN where the human label is missing."
        )
    if outcome is None:
        raise ValueError("outcome series is required.")


def llm_annotator_correct(
    *,
    annotations_llm: pd.Series,
    outcome: pd.Series,
    annotations_human: Optional[pd.Series] = None,
    covariates: Optional[pd.DataFrame] = None,
    method: str = 'hausman',
    alpha: float = 0.05,
) -> LLMAnnotatorResult:
    """Correct a downstream causal coefficient for LLM annotation noise.

    Implements the Hausman-style correction for binary treatment
    misclassification.  Given a small subset of rows with both LLM and
    human labels, estimate the false-positive (``p_01``) and
    false-negative (``p_10``) rates, and use them to inflate the naive
    OLS coefficient back to the true value.

    Parameters
    ----------
    annotations_llm : pd.Series of {0, 1}
        LLM-derived binary annotation for every row.
    outcome : pd.Series
        Outcome variable.
    annotations_human : pd.Series, optional
        Human annotation; ``NaN`` where unavailable.  At least 30 rows
        with both LLM and human labels are required for stable
        correction.
    covariates : pd.DataFrame, optional
        Additional control variables for the OLS regression.
    method : {'hausman'}, default 'hausman'
        Correction method.  Future versions will add 'sar' (super-
        learner with adjustment) and 'bootstrap'.
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
    >>> T_llm = (T_true ^ noise).astype(int)            # 15% misclassification
    >>> y = 1.0 * T_true + rng.standard_normal(n)        # true ATE = 1.0
    >>> human = pd.Series([T_true[i] if i < n_val else np.nan
    ...                    for i in range(n)])
    >>> r = sp.llm_annotator_correct(
    ...     annotations_llm=pd.Series(T_llm),
    ...     annotations_human=human,
    ...     outcome=pd.Series(y),
    ... )
    >>> r.estimate    # ~1.0 (corrected from naive ~0.7)
    """
    if method not in {'hausman'}:
        raise ValueError(
            f"Unknown method={method!r}. Currently supported: 'hausman'."
        )
    _validate_inputs(annotations_llm, annotations_human, outcome,
                     covariates)

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

    # Full sample for the naive OLS.
    use_full = df.dropna(subset=['T_llm', 'y']).copy()
    use_full = use_full[
        use_full[cov_cols].notna().all(axis=1) if cov_cols
        else np.full(len(use_full), True)
    ]
    if len(use_full) < 20:
        from ..exceptions import DataInsufficient
        raise DataInsufficient(
            f"At least 20 rows required for the OLS step; got "
            f"{len(use_full)}."
        )

    # Validation sample: rows with both labels.
    val = df.dropna(subset=['T_llm', 'T_human']).copy()
    if len(val) < 30:
        from ..exceptions import DataInsufficient
        raise DataInsufficient(
            f"At least 30 validation rows (with both LLM and human "
            f"labels) recommended for stable correction; got {len(val)}."
        )

    # Estimate p_01 and p_10 on the validation set.
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
    p_01 = float(((val_t_human == 0) & (val_t_llm == 1)).sum() / n_human0)
    p_10 = float(((val_t_human == 1) & (val_t_llm == 0)).sum() / n_human1)
    correction_factor = 1.0 - p_01 - p_10
    if correction_factor <= 0:
        # Misclassification too severe — coefficient is unidentified.
        from ..exceptions import IdentificationFailure
        raise IdentificationFailure(
            f"Misclassification rates (p_01={p_01:.3f}, p_10={p_10:.3f}) "
            "imply the LLM label has no information about the true "
            "treatment (1 - p_01 - p_10 <= 0). Correction is not "
            "identified; consider re-prompting the LLM or hand-labelling."
        )
    agreement = float((val_t_llm == val_t_human).mean())

    # Naive OLS on the full sample.
    n = len(use_full)
    cov_mat = (use_full[cov_cols].astype(np.float64).values if cov_cols
               else np.zeros((n, 0)))
    X = np.hstack([
        np.ones((n, 1)),
        use_full['T_llm'].astype(np.float64).values.reshape(-1, 1),
        cov_mat,
    ])
    y = use_full['y'].astype(np.float64).values
    treat_idx = 1
    XtX = X.T @ X
    try:
        XtX_inv = np.linalg.pinv(XtX)
    except np.linalg.LinAlgError as exc:
        raise RuntimeError(
            f"OLS normal equations singular: {exc}"
        )
    beta = XtX_inv @ (X.T @ y)
    yhat = X @ beta
    resid = y - yhat
    p = X.shape[1]
    df_resid = max(n - p, 1)
    Omega = (X.T * (resid ** 2)) @ X
    cov_hc1 = (n / df_resid) * (XtX_inv @ Omega @ XtX_inv)
    se = np.sqrt(np.maximum(np.diag(cov_hc1), 0.0))

    naive_estimate = float(beta[treat_idx])
    naive_se = float(se[treat_idx])

    corrected_estimate = naive_estimate / correction_factor
    corrected_se = naive_se / abs(correction_factor)
    z = sp_stats.norm.ppf(1 - alpha / 2)
    ci_lo = corrected_estimate - z * corrected_se
    ci_hi = corrected_estimate + z * corrected_se
    pval = float(2 * (1 - sp_stats.norm.cdf(
        abs(corrected_estimate / corrected_se)
    ))) if corrected_se > 0 else float('nan')

    diagnostics: Dict[str, Any] = {
        'p_01': p_01,
        'p_10': p_10,
        'correction_factor': float(correction_factor),
        'agreement': agreement,
        'n_validation': int(len(val)),
        'n_full': int(n),
        'method': method,
        'se_correction': 'first_order',
        'status': 'experimental',
        'method_family': 'llm-annotator-mec (Egami et al. 2024 MVP)',
    }

    return LLMAnnotatorResult(
        method='llm_annotator_correct',
        estimand='ATE',
        estimate=corrected_estimate,
        se=corrected_se,
        pvalue=pval,
        ci=(float(ci_lo), float(ci_hi)),
        alpha=alpha,
        n_obs=int(n),
        naive_estimate=naive_estimate,
        naive_se=naive_se,
        correction_factor=float(correction_factor),
        annotator_diagnostics=diagnostics,
    )
