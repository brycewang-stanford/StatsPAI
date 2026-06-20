"""Shared GLM-fit primitives (CLAUDE.md §4: no duplicated estimator basics).

Currently hosts the *safe statsmodels logistic fit* used by covariate-adjusted
nuisance models (front-door mediator model, principal-score model). Both
previously carried byte-identical private copies of this wrapper.

NOTE ON SCOPE — this intentionally does NOT subsume the custom Newton/IRLS
logits in ``censoring/ipcw``, ``gformula/mc``, ``longitudinal/analyze``,
``iv/ivmte_lp`` or the sklearn-regularised logit in ``tmle/ltmle``: those use
deliberately different solvers / tolerances / regularisation, so merging them
would shift their numerical output. Unifying onto one canonical IRLS is a
parity-sensitive follow-up, not a drop-in dedup.
"""

from __future__ import annotations

from typing import Any

import numpy as np

__all__ = ["safe_logit_fit", "safe_logit_predict"]


def safe_logit_fit(y: Any, X: Any) -> Any:
    """Fit a logistic regression via statsmodels with an added constant.

    Returns the fitted results object, or ``None`` when the fit fails
    (singular / separated design, non-convergence). Callers are responsible
    for surfacing the ``None`` fallback loudly rather than swallowing it
    (CLAUDE.md §7) — see ``front_door`` / ``principal_strat``.
    """
    try:
        import statsmodels.api as sm

        design = sm.add_constant(X, has_constant="add")
        return sm.Logit(y, design).fit(disp=0, maxiter=200, warn_convergence=False)
    except Exception:
        return None


def safe_logit_predict(fit: Any, X: Any, fallback: float) -> np.ndarray:
    """Predict P(y=1 | X) from a ``safe_logit_fit`` result.

    When ``fit is None`` (the fit failed) returns the constant ``fallback``
    for every row — the *unadjusted* marginal probability. Callers must warn
    when this path is taken on the main sample.
    """
    if fit is None:
        return np.full(X.shape[0], fallback)
    import statsmodels.api as sm

    design = sm.add_constant(X, has_constant="add")
    return np.asarray(np.clip(fit.predict(design), 1e-6, 1 - 1e-6))
