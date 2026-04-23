"""
Inverse Probability of Censoring Weights (IPCW).

IPCW re-weights non-censored observations to recover the distribution
that would have been observed without informative censoring. Combined
with standard estimators (Cox, pooled logistic, g-computation) it
restores consistency under the assumption of *conditional independent
censoring* given measured covariates.

References
----------
* Robins & Finkelstein (2000). "Correcting for Noncompliance and
  Dependent Censoring in an AIDS Clinical Trial with IPCW Log-Rank
  Tests."
* Hernan & Robins. *Causal Inference: What If* (Chapter 17). [@robins2000correcting]
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Sequence
import numpy as np
import pandas as pd


@dataclass
class IPCWResult:
    """Result of an IPCW fit.

    Attributes
    ----------
    weights : np.ndarray
        Per-observation IPC weights (uncensored obs only; censored rows
        receive weight 0 by convention but are returned for alignment).
    stabilized : bool
        Whether stabilized weights are reported.
    summary_stats : dict
        Basic diagnostics — mean, max, share above common thresholds.
    method : str
        Nuisance model used for the censoring hazard.
    """

    weights: np.ndarray
    stabilized: bool
    summary_stats: dict
    method: str
    fitted_hazards: np.ndarray = field(default_factory=lambda: np.empty(0))

    def diagnose(self) -> pd.DataFrame:
        """Common weight diagnostics — flags extreme IPCW values."""
        w = self.weights
        w = w[np.isfinite(w) & (w > 0)]
        out = pd.DataFrame(
            {
                "metric": [
                    "n_obs",
                    "mean",
                    "sd",
                    "min",
                    "max",
                    "share > 10",
                    "share > 20",
                    "effective_sample_size",
                ],
                "value": [
                    int(w.size),
                    float(w.mean()) if w.size else np.nan,
                    float(w.std(ddof=1)) if w.size > 1 else np.nan,
                    float(w.min()) if w.size else np.nan,
                    float(w.max()) if w.size else np.nan,
                    float((w > 10).mean()) if w.size else np.nan,
                    float((w > 20).mean()) if w.size else np.nan,
                    float(w.sum() ** 2 / (w ** 2).sum()) if w.size else np.nan,
                ],
            }
        )
        return out


def ipcw(
    data: pd.DataFrame,
    time: str,
    event: str,
    censor_covariates: Sequence[str],
    treatment_covariates: Sequence[str] | None = None,
    stabilize: bool = True,
    method: str = "pooled_logistic",
    truncate: tuple[float, float] | None = (0.01, 0.99),
) -> IPCWResult:
    """Compute inverse probability of censoring weights.

    Parameters
    ----------
    data : pd.DataFrame
        Long-format (person-time) or wide-format (one row per subject)
        dataframe. For pooled logistic pathway use long format with
        columns ``(id, t, time, event, ...)``.
    time : str
        Follow-up time column.
    event : str
        Event indicator — 1 if event, 0 if censored (administrative or
        informative), right-censored at ``time``.
    censor_covariates : list[str]
        Covariates predicting censoring.
    treatment_covariates : list[str], optional
        Extra covariates used for the numerator of stabilized weights
        (usually baseline / pre-treatment only).
    stabilize : bool, default True
        Return stabilized weights
        :math:`sw_i = \\hat P(C=0 \\mid V) / \\hat P(C=0 \\mid V, L)`.
    method : {"pooled_logistic", "cox_ph"}, default "pooled_logistic"
        Nuisance model for the censoring hazard. Pooled logistic is
        recommended when follow-up is long and outcomes are rare.
    truncate : tuple[float, float] | None, default (0.01, 0.99)
        Truncate weights at these quantiles to curb extreme values
        (Cole & Hernan 2008). ``None`` disables truncation.

    Returns
    -------
    IPCWResult
    """
    if time not in data.columns or event not in data.columns:
        raise KeyError("time/event columns not in data.")
    missing = [c for c in censor_covariates if c not in data.columns]
    if missing:
        raise KeyError(f"Missing censor covariates: {missing}")

    n = len(data)
    t = data[time].to_numpy(dtype=float)
    d = data[event].to_numpy(dtype=int)
    X = data[list(censor_covariates)].to_numpy(dtype=float)
    X = np.column_stack([np.ones(n), X])

    if method == "pooled_logistic":
        beta = _fit_logit(_censor_indicator(d), X)
        eta = X @ beta
        p_uncensored_cond = _sigmoid(eta)
    elif method == "cox_ph":
        p_uncensored_cond = _cox_uncensored_survival(t, d, X[:, 1:])
    else:
        raise ValueError(
            "method must be 'pooled_logistic' or 'cox_ph', got "
            f"{method!r}."
        )

    p_uncensored_cond = np.clip(p_uncensored_cond, 1e-8, 1.0)

    if stabilize:
        if treatment_covariates:
            V = data[list(treatment_covariates)].to_numpy(dtype=float)
            V = np.column_stack([np.ones(n), V])
        else:
            V = np.ones((n, 1))
        beta_num = _fit_logit(_censor_indicator(d), V)
        p_uncensored_marg = np.clip(_sigmoid(V @ beta_num), 1e-8, 1.0)
        w = p_uncensored_marg / p_uncensored_cond
    else:
        w = 1.0 / p_uncensored_cond

    w = w * (d >= 0).astype(float)
    w = np.where(d == 1, w, w)

    if truncate is not None:
        lo, hi = np.quantile(w[np.isfinite(w) & (w > 0)], list(truncate))
        w = np.clip(w, lo, hi)

    summary = {
        "mean": float(np.nanmean(w)),
        "max": float(np.nanmax(w)),
        "min": float(np.nanmin(w)),
        "effective_sample_size": float(w.sum() ** 2 / (w ** 2).sum()),
    }

    return IPCWResult(
        weights=w,
        stabilized=stabilize,
        summary_stats=summary,
        method=method,
        fitted_hazards=1.0 - p_uncensored_cond,
    )


def _censor_indicator(event: np.ndarray) -> np.ndarray:
    """Return 1 if NOT censored (event or still at risk), 0 if censored."""
    return (event == 1).astype(float)


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -35, 35)
    return 1.0 / (1.0 + np.exp(-z))


def _fit_logit(y: np.ndarray, X: np.ndarray, max_iter: int = 50, tol: float = 1e-8) -> np.ndarray:
    """Plain Newton-Raphson IRLS logistic regression; no external deps."""
    n, p = X.shape
    beta = np.zeros(p)
    for _ in range(max_iter):
        eta = X @ beta
        mu = _sigmoid(eta)
        W = mu * (1.0 - mu) + 1e-8
        gradient = X.T @ (y - mu)
        H = -(X.T * W) @ X
        try:
            step = np.linalg.solve(H, gradient)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(H, gradient, rcond=None)[0]
        beta_new = beta - step
        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new
            break
        beta = beta_new
    return beta


def _cox_uncensored_survival(
    t: np.ndarray, d: np.ndarray, X: np.ndarray
) -> np.ndarray:
    """Very light Breslow Cox estimator — good enough to derive
    :math:`\\hat S_C(t|X)` = prob still uncensored by t given X.

    For heavy lifting users should plug ``lifelines`` / sp.survival.cox.
    """
    n, p = X.shape
    censor_event = (d == 0).astype(float)
    order = np.argsort(t)
    X_ord, t_ord, ce_ord = X[order], t[order], censor_event[order]

    beta = np.zeros(p)
    for _ in range(25):
        eta = X_ord @ beta
        exp_eta = np.exp(np.clip(eta, -35, 35))
        cum_risk = np.flip(np.cumsum(np.flip(exp_eta)))
        wmean = np.flip(np.cumsum(np.flip(X_ord * exp_eta[:, None]))) / np.clip(
            np.repeat(cum_risk[:, None], p, axis=1), 1e-12, None
        )
        score = (X_ord - wmean)[ce_ord == 1].sum(axis=0)
        hess = np.zeros((p, p))
        for i in np.where(ce_ord == 1)[0]:
            diff = X_ord[i:] - wmean[i]
            w = exp_eta[i:] / np.clip(cum_risk[i], 1e-12, None)
            hess += (diff * w[:, None]).T @ diff
        hess = -hess - 1e-6 * np.eye(p)
        try:
            step = np.linalg.solve(hess, score)
        except np.linalg.LinAlgError:
            step = np.linalg.lstsq(hess, score, rcond=None)[0]
        beta_new = beta - step
        if np.max(np.abs(beta_new - beta)) < 1e-6:
            beta = beta_new
            break
        beta = beta_new

    eta = X @ beta
    haz_baseline = np.cumsum(
        censor_event / np.clip(
            np.array([np.sum(np.exp(np.clip(X[t >= ti] @ beta, -35, 35))) for ti in t]),
            1e-12, None,
        )
    )
    surv = np.exp(-haz_baseline * np.exp(np.clip(eta, -35, 35)))
    return np.clip(surv, 1e-6, 1.0)
