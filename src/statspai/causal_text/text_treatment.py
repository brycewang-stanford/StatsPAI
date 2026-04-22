"""Text-as-treatment estimation (Veitch-Wang-Blei 2020) — MVP.

Estimate the average treatment effect of a (typically binary)
treatment indicator while controlling for the confounding variation
captured in a low-dimensional text embedding.

Approach (simplified MVP):

    1. Embed each document via :func:`statspai.causal_text._common.embed_texts`
       to an ``(n, k)`` real-valued matrix.
    2. Treat the embedding columns as additional confounders.
    3. Estimate the ATE via outcome regression — OLS on the joined
       ``[treatment, embedding, covariates]`` design matrix.  The
       coefficient on ``treatment`` is the ATE.
    4. Compute heteroskedasticity-robust (HC1) standard errors via the
       sandwich formula computed locally (so we don't pull in
       statsmodels for the MVP).

Status: **experimental.**  The full Veitch et al. (2020) recipe uses a
neural causal effect variational autoencoder (CEVAE) with a separate
embedding space for treatment-relevant vs outcome-relevant text
variation.  The MVP uses a single shared low-dimensional projection;
this is a coarser approximation but produces an unbiased ATE estimate
under the (strong) assumption that all text-based confounding is
captured by the projection.

References
----------
Veitch, V., Sridhar, D., & Blei, D. M. (2019). "Adapting text embeddings
for causal inference." *UAI*. arXiv:1905.12741.
"""
from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from ..core.results import CausalResult
from ._common import embed_texts


__all__ = ['text_treatment_effect', 'TextTreatmentResult']


class TextTreatmentResult(CausalResult):
    """ATE result for text-as-treatment estimation.

    Subclasses :class:`CausalResult` so it inherits ``.tidy()``,
    ``.to_latex()``, ``.cite()``, and the agent-native ``.to_dict()``
    when P0's additions are present.  Exposes embedding-specific
    metadata on the instance and inside ``model_info['text_diagnostics']``
    so agents can introspect what happened.
    """

    def __init__(
        self, *, method: str, estimand: str, estimate: float, se: float,
        pvalue: float, ci: tuple, alpha: float, n_obs: int,
        embedding_dim: int, embedder_name: str,
        diagnostics: Optional[Dict[str, Any]] = None,
        model_info: Optional[Dict[str, Any]] = None,
    ):
        # Flatten text-specific diagnostics into model_info so they
        # surface via CausalResult's inherited `.diagnostics` property
        # (which returns model_info verbatim). The full block is also
        # kept under `model_info['text_diagnostics']` for callers that
        # want a self-contained sub-dict.
        flat = dict(diagnostics or {})
        flat.update({
            'embedding_dim': int(embedding_dim),
            'embedder_name': str(embedder_name),
            'status': 'experimental',
        })
        mi = dict(model_info or {})
        mi.update(flat)
        mi['text_diagnostics'] = dict(flat)
        super().__init__(
            method=method, estimand=estimand, estimate=estimate, se=se,
            pvalue=pvalue, ci=ci, alpha=alpha, n_obs=n_obs,
            model_info=mi,
        )
        self.embedding_dim = int(embedding_dim)
        self.embedder_name = str(embedder_name)

    def summary(self) -> str:  # pragma: no cover (cosmetic)
        lines = [
            "TextTreatmentResult (experimental)",
            "=" * 60,
            f"  Method            : {self.method}",
            f"  Estimand          : {self.estimand}",
            f"  Estimate          : {self.estimate:.4f}",
            f"  SE                : {self.se:.4f}",
            f"  95% CI            : [{self.ci[0]:.4f}, {self.ci[1]:.4f}]",
            f"  p-value           : {self.pvalue:.4f}",
            f"  N obs             : {self.n_obs}",
            f"  Embedding dim     : {self.embedding_dim}",
            f"  Embedder          : {self.embedder_name}",
            "  Status            : experimental",
        ]
        return "\n".join(lines)


def text_treatment_effect(
    data: pd.DataFrame,
    *,
    text_col: str,
    outcome: str,
    treatment: str,
    covariates: Optional[List[str]] = None,
    embedder: Union[str, Callable] = 'hash',
    n_components: int = 20,
    seed: int = 0,
    alpha: float = 0.05,
) -> TextTreatmentResult:
    """Estimate the ATE of a text-derived treatment via embedding adjustment.

    Parameters
    ----------
    data : pd.DataFrame
    text_col : str
        Column containing the document text (string).
    outcome : str
        Outcome column (numeric).
    treatment : str
        Treatment column (binary or continuous).  This is the variable
        whose coefficient is interpreted as the ATE.
    covariates : list of str, optional
        Additional non-text covariates to include in the adjustment
        regression.
    embedder : {'hash', 'sbert'} or callable, default 'hash'
        Text embedder.  See :func:`statspai.causal_text._common.embed_texts`.
    n_components : int, default 20
        Embedding dimensionality (for 'hash'); ignored for 'sbert'.
    seed : int, default 0
    alpha : float, default 0.05
        CI level (1 - alpha confidence).

    Returns
    -------
    TextTreatmentResult

    Examples
    --------
    >>> import statspai as sp, pandas as pd
    >>> df = pd.DataFrame({
    ...     'text': ['great product', 'terrible bug', 'okay tool', 'love it'],
    ...     'treatment': [1, 0, 0, 1],
    ...     'outcome': [4.5, 1.2, 2.8, 4.7],
    ... })
    >>> r = sp.text_treatment_effect(df, text_col='text', outcome='outcome',
    ...                              treatment='treatment', n_components=4)
    >>> r.estimate
    """
    for col in (text_col, outcome, treatment):
        if col not in data.columns:
            raise ValueError(f"Column {col!r} not in data.")
    cov = list(covariates or [])
    for c in cov:
        if c not in data.columns:
            raise ValueError(f"Covariate {c!r} not in data.")

    # Drop rows with NaN in any required column.
    use = data[[text_col, outcome, treatment, *cov]].copy()
    use = use.dropna(subset=[outcome, treatment, *cov])
    if len(use) < max(20, n_components + 4):
        from ..exceptions import DataInsufficient
        raise DataInsufficient(
            f"Need at least {max(20, n_components + 4)} non-missing rows "
            f"for n_components={n_components}; got {len(use)}."
        )
    use[text_col] = use[text_col].fillna("")

    # Embed text.
    Z = embed_texts(
        use[text_col].astype(str).values,
        embedder=embedder, n_components=n_components, seed=seed,
    )
    if Z.shape[1] < 1:
        raise ValueError("Embedder returned 0 components.")
    actual_k = Z.shape[1]

    y = use[outcome].astype(np.float64).values
    t = use[treatment].astype(np.float64).values
    n = len(y)
    cov_mat = (use[cov].astype(np.float64).values if cov
               else np.zeros((n, 0)))

    # Build design matrix [intercept, t, cov, Z].
    X = np.hstack([
        np.ones((n, 1)),
        t.reshape(-1, 1),
        cov_mat,
        Z,
    ])
    treat_idx = 1  # column index of `treatment` in X

    # OLS via lstsq.
    XtX = X.T @ X
    try:
        XtX_inv = np.linalg.pinv(XtX)
    except np.linalg.LinAlgError as exc:
        raise RuntimeError(
            f"OLS normal equations singular: {exc}. "
            "Try lowering n_components or removing collinear covariates."
        )
    beta = XtX_inv @ (X.T @ y)
    yhat = X @ beta
    resid = y - yhat

    # HC1 sandwich standard errors.
    p = X.shape[1]
    df_resid = max(n - p, 1)
    Omega = (X.T * (resid ** 2)) @ X
    cov_hc1 = (n / df_resid) * (XtX_inv @ Omega @ XtX_inv)
    se = np.sqrt(np.maximum(np.diag(cov_hc1), 0.0))

    estimate = float(beta[treat_idx])
    se_t = float(se[treat_idx])
    z = sp_stats.norm.ppf(1 - alpha / 2)
    ci_lo = estimate - z * se_t
    ci_hi = estimate + z * se_t
    pval = float(2 * (1 - sp_stats.norm.cdf(abs(estimate / se_t))))

    diagnostics: Dict[str, Any] = {
        'n_text_components': int(actual_k),
        'embedder': embedder if isinstance(embedder, str) else 'callable',
        'n_obs': int(n),
        'status': 'experimental',
        'method_family': 'text-as-treatment (Veitch-Wang-Blei 2020 MVP)',
    }

    return TextTreatmentResult(
        method='text_treatment_effect',
        estimand='ATE',
        estimate=estimate,
        se=se_t,
        ci=(float(ci_lo), float(ci_hi)),
        pvalue=pval,
        alpha=alpha,
        n_obs=int(n),
        embedding_dim=int(actual_k),
        embedder_name=embedder if isinstance(embedder, str) else 'callable',
        diagnostics=diagnostics,
    )
