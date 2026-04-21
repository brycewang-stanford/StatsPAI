"""
Target trial emulation — end-to-end pipeline wiring a protocol to data.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Callable
import numpy as np
import pandas as pd

from .protocol import TargetTrialProtocol


@dataclass
class TargetTrialResult:
    """Result of target trial emulation.

    Attributes
    ----------
    protocol : TargetTrialProtocol
        The protocol that was emulated.
    estimate : float
        Point estimate of the causal contrast under the protocol.
    se : float
        Analytic standard error (IPW sandwich or bootstrap).
    ci : tuple[float, float]
        95% confidence interval.
    n_eligible : int
        Subjects passing the eligibility criterion at time zero.
    n_excluded_immortal : int
        Subjects excluded to prevent immortal time bias.
    weights : np.ndarray
        IP weights used (baseline + censoring combined).
    method : str
        Which analysis plan was executed.
    """

    protocol: TargetTrialProtocol
    estimate: float
    se: float
    ci: tuple[float, float]
    n_eligible: int
    n_excluded_immortal: int
    weights: np.ndarray
    method: str

    def summary(self) -> str:
        return (
            f"Target Trial Emulation -- {self.protocol.causal_contrast}\n"
            f"  Estimate: {self.estimate:.4f}  SE: {self.se:.4f}  "
            f"95% CI: [{self.ci[0]:.4f}, {self.ci[1]:.4f}]\n"
            f"  n eligible: {self.n_eligible}  "
            f"n excluded (immortal-time prevention): {self.n_excluded_immortal}\n"
            f"  Method: {self.method}"
        )

    def to_paper(self, fmt: str = "markdown", title: str | None = None) -> str:
        """Render a publication-ready Methods/Results block.

        See :func:`statspai.target_trial.to_paper` for details.
        """
        from .report import to_paper as _to_paper
        return _to_paper(self, fmt=fmt, title=title)


def emulate(
    protocol: TargetTrialProtocol,
    data: pd.DataFrame,
    outcome_col: str,
    treatment_col: str,
    time_zero_filter: Callable[[pd.DataFrame], pd.Series] | None = None,
    weights: np.ndarray | None = None,
) -> TargetTrialResult:
    """Emulate the target trial defined by ``protocol``.

    Analysis-stage wrapper: enforces time-zero alignment, applies
    eligibility, computes (or accepts) weights, and returns an
    IPW-weighted mean-difference point estimate for the protocol-
    defined contrast.

    For more complex analysis plans (pooled logistic + IPCW, g-formula,
    LTMLE), users should call those estimators directly and pass the
    protocol as documentation.
    """
    n_total = len(data)

    if time_zero_filter is not None:
        mask_values = np.asarray(time_zero_filter(data), dtype=bool)
    elif isinstance(protocol.eligibility, str):
        subset = data.query(protocol.eligibility)
        mask_values = data.index.isin(subset.index)
    elif callable(protocol.eligibility):
        mask_values = data.apply(protocol.eligibility, axis=1).to_numpy(dtype=bool)
    else:
        mask_values = np.ones(n_total, dtype=bool)

    mask = pd.Series(mask_values, index=data.index)
    eligible = data.loc[mask].copy()
    n_eligible = int(len(eligible))
    n_excluded = int(n_total - n_eligible)

    if treatment_col not in eligible.columns or outcome_col not in eligible.columns:
        raise KeyError("treatment_col / outcome_col must be in data")

    a = eligible[treatment_col].to_numpy(dtype=float)
    y = eligible[outcome_col].to_numpy(dtype=float)
    if weights is None:
        w = np.ones_like(a)
    else:
        w_full = np.asarray(weights, dtype=float)
        w = w_full[mask_values]

    sum_w1 = max(float(np.sum(w * a)), 1e-12)
    sum_w0 = max(float(np.sum(w * (1 - a))), 1e-12)
    m1 = float(np.sum(w * a * y) / sum_w1)
    m0 = float(np.sum(w * (1 - a) * y) / sum_w0)
    estimate = m1 - m0

    v1 = float(np.sum(w * a * (y - m1) ** 2) / sum_w1 ** 2)
    v0 = float(np.sum(w * (1 - a) * (y - m0) ** 2) / sum_w0 ** 2)
    se = float(np.sqrt(v1 + v0))
    ci = (estimate - 1.96 * se, estimate + 1.96 * se)

    return TargetTrialResult(
        protocol=protocol,
        estimate=estimate,
        se=se,
        ci=(float(ci[0]), float(ci[1])),
        n_eligible=n_eligible,
        n_excluded_immortal=n_excluded,
        weights=w,
        method=protocol.analysis_plan,
    )
