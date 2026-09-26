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

from .._result_serialize import ResultProtocolMixin


@dataclass
class IPCWResult(ResultProtocolMixin):
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

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> n = 200
    >>> age = rng.normal(50, 10, n)
    >>> biomarker = rng.normal(0, 1, n)
    >>> p = 1.0 / (1.0 + np.exp(-(0.5 - 0.5 * biomarker)))
    >>> df = pd.DataFrame({
    ...     "time": rng.exponential(5, n),
    ...     "event": rng.binomial(1, p),
    ...     "age": age,
    ...     "biomarker": biomarker,
    ... })
    >>> res = sp.ipcw(df, time="time", event="event",
    ...               censor_covariates=["age", "biomarker"])
    >>> isinstance(res, sp.IPCWResult)
    True
    >>> diag = res.diagnose()        # weight diagnostics table
    >>> list(diag.columns)
    ['metric', 'value']
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
                    float(w.sum() ** 2 / (w**2).sum()) if w.size else np.nan,
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
        Nuisance model for being uncensored.

        * ``"pooled_logistic"``: a logistic regression of the observed
          indicator ``event == 1`` on the covariates, row by row. It uses
          no time information, so it is the inverse probability of an
          *observed outcome* (complete-case IPW), not a hazard model; for
          person-time data with a per-interval censoring indicator build
          the cumulative product yourself or use ``sp.target_trial``.
        * ``"cox_ph"``: one row per subject. A Cox model (Breslow ties) is
          fitted to the censoring times (``event == 0`` as the "event")
          and each subject gets ``1 / S_C(T_i- | X_i)`` with the Breslow
          baseline cumulative hazard -- the standard IPCW. With
          ``stabilize=True`` the numerator is the same Cox model on
          ``treatment_covariates`` (the Nelson-Aalen marginal ``S_C`` when
          none are given). Matches ``survival::coxph(ties = "breslow")``
          with ``basehaz(centered = FALSE)``.
    truncate : tuple[float, float] | None, default (0.01, 0.99)
        Truncate the uncensored rows' weights at these quantiles to curb
        extreme values (Cole & Hernan 2008). ``None`` disables truncation.

    Returns
    -------
    IPCWResult
        Censored rows (``event == 0``) carry weight 0; the summary
        statistics are over the uncensored rows. (Before 1.29 censored
        rows received the same nonzero weight formula as uncensored ones,
        contrary to this documented convention, and ``method="cox_ph"``
        raised a broadcasting error for more than one covariate and
        accumulated the baseline hazard in data order rather than time
        order.)

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> n = 200
    >>> biomarker = rng.normal(0, 1, n)
    >>> p = 1.0 / (1.0 + np.exp(-(0.5 - 0.5 * biomarker)))
    >>> df = pd.DataFrame({
    ...     "time": rng.exponential(5, n),
    ...     "event": rng.binomial(1, p),
    ...     "age": rng.normal(50, 10, n),
    ...     "biomarker": biomarker,
    ... })
    >>> res = sp.ipcw(df, time="time", event="event",
    ...               censor_covariates=["age", "biomarker"])
    >>> bool(res.stabilized)
    True
    >>> sorted(res.summary_stats)
    ['effective_sample_size', 'max', 'mean', 'min']
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
            "method must be 'pooled_logistic' or 'cox_ph', got " f"{method!r}."
        )

    p_uncensored_cond = np.clip(p_uncensored_cond, 1e-8, 1.0)

    if stabilize:
        if treatment_covariates:
            V_raw = data[list(treatment_covariates)].to_numpy(dtype=float)
        else:
            V_raw = np.zeros((n, 0))
        if method == "cox_ph":
            p_uncensored_marg = _cox_uncensored_survival(t, d, V_raw)
        else:
            V = np.column_stack([np.ones(n), V_raw])
            beta_num = _fit_logit(_censor_indicator(d), V)
            p_uncensored_marg = _sigmoid(V @ beta_num)
        p_uncensored_marg = np.clip(p_uncensored_marg, 1e-8, 1.0)
        w = p_uncensored_marg / p_uncensored_cond
    else:
        w = 1.0 / p_uncensored_cond

    # Censored rows contribute nothing to the weighted (complete-case)
    # analysis. The line here used to read ``np.where(d == 1, w, w)``.
    observed = d == 1
    w = np.where(observed, w, 0.0)

    if truncate is not None and observed.any():
        lo, hi = np.quantile(w[observed], list(truncate))
        w = np.where(observed, np.clip(w, lo, hi), 0.0)

    wo = w[observed]
    summary = {
        "mean": float(np.mean(wo)) if wo.size else float("nan"),
        "max": float(np.max(wo)) if wo.size else float("nan"),
        "min": float(np.min(wo)) if wo.size else float("nan"),
        "effective_sample_size": (
            float(wo.sum() ** 2 / (wo**2).sum()) if wo.size else float("nan")
        ),
    }

    _result = IPCWResult(
        weights=w,
        stabilized=stabilize,
        summary_stats=summary,
        method=method,
        fitted_hazards=1.0 - p_uncensored_cond,
    )
    try:
        from ..output._lineage import attach_provenance as _attach_prov

        _attach_prov(
            _result,
            function="sp.censoring.ipcw",
            params={
                "time": time,
                "event": event,
                "censor_covariates": list(censor_covariates),
                "treatment_covariates": (
                    list(treatment_covariates) if treatment_covariates else None
                ),
                "stabilize": stabilize,
                "method": method,
                "truncate": list(truncate) if truncate else None,
            },
            data=data,
            overwrite=False,
        )
    except Exception:  # pragma: no cover
        pass
    return _result


def _censor_indicator(event: np.ndarray) -> np.ndarray:
    """Return 1 if NOT censored (event or still at risk), 0 if censored."""
    return np.asarray((event == 1).astype(float))


def _sigmoid(z: np.ndarray) -> np.ndarray:
    z = np.clip(z, -35, 35)
    return np.asarray(1.0 / (1.0 + np.exp(-z)))


def _fit_logit(
    y: np.ndarray, X: np.ndarray, max_iter: int = 50, tol: float = 1e-8
) -> np.ndarray:
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


def _cox_uncensored_survival(t: np.ndarray, d: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Cox model for the censoring time; returns ``S_C(t_i- | x_i)``.

    Censoring (``d == 0``) is the event. Coefficients maximise the Breslow
    partial likelihood (Newton-Raphson); the baseline cumulative hazard is
    Breslow's ``H0(s) = sum_{censoring times u <= s} 1 / sum_{t_l >= u}
    exp(x_l b)`` and the survival is taken just before ``t_i``, so a
    subject's own censoring does not enter its weight. With no covariates
    this is the Nelson-Aalen estimator of the censoring distribution.
    """
    t = np.asarray(t, dtype=float)
    X = np.asarray(X, dtype=float).reshape(len(t), -1)
    n, p = X.shape
    ce = (np.asarray(d) == 0).astype(float)

    order = np.argsort(t, kind="mergesort")
    ts, Xs, ces = t[order], X[order], ce[order]
    # Risk sets: rows with t_l >= t_i. With ties, all tied rows share the
    # risk set that starts at the first of them (Breslow).
    first_at = np.searchsorted(ts, ts, side="left")

    beta = np.zeros(p)
    if p:
        for _ in range(100):
            r = np.exp(np.clip(Xs @ beta, -700, 700))
            S0 = np.cumsum(r[::-1])[::-1][first_at]
            S1 = np.cumsum((Xs * r[:, None])[::-1], axis=0)[::-1][first_at]
            S2 = np.cumsum(
                (Xs[:, :, None] * Xs[:, None, :] * r[:, None, None])[::-1], axis=0
            )[::-1][first_at]
            ev = ces == 1
            xbar = S1[ev] / S0[ev, None]
            score = (Xs[ev] - xbar).sum(axis=0)
            info = (S2[ev] / S0[ev, None, None]).sum(axis=0) - np.einsum(
                "ij,ik->jk", xbar, xbar
            )
            step = np.linalg.solve(info, score)
            beta = beta + step
            if np.max(np.abs(step)) < 1e-12:
                break

    r = np.exp(np.clip(Xs @ beta, -700, 700))
    S0 = np.cumsum(r[::-1])[::-1][first_at]
    # Breslow increments at censoring times, then H0 just before each t_i.
    inc = np.where(ces == 1, 1.0 / S0, 0.0)
    H_cum = np.cumsum(inc)
    # H0(t_i-) = sum of increments at censoring times strictly below t_i.
    H_before = np.concatenate([[0.0], H_cum])[first_at]
    surv_sorted = np.exp(-H_before * np.exp(np.clip(Xs @ beta, -700, 700)))
    out = np.empty(n)
    out[order] = surv_sorted
    return np.asarray(out)
