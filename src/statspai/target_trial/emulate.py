"""
Target trial emulation — end-to-end pipeline wiring a protocol to data.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Callable, Literal, cast

import numpy as np
import pandas as pd

from .._result_serialize import ResultProtocolMixin
from ..exceptions import AssumptionWarning, DataInsufficient, MethodIncompatibility
from .protocol import TargetTrialProtocol


@dataclass
class TargetTrialResult(ResultProtocolMixin):
    """Result of target trial emulation.

    Attributes
    ----------
    protocol : TargetTrialProtocol
        The protocol that was emulated.
    estimate : float
        Point estimate of the causal contrast under the protocol.
    se : float
        Sandwich (HC0) standard error of the weighted mean difference,
        weights treated as known (``lm(y ~ a, weights = w)`` with
        ``sandwich::vcovHC(type = "HC0")``).
    ci : tuple[float, float]
        95% confidence interval.
    n_eligible : int
        Subjects passing the eligibility criterion at time zero.
    n_excluded_immortal : int
        Subjects who did not meet the eligibility criterion at time zero
        and were left out (also available as ``n_excluded``). Applying
        eligibility at time zero is what keeps immortal time out of the
        comparison; the count itself is of ineligible subjects.
    weights : np.ndarray
        IP weights used (baseline + censoring combined).
    method : str
        What was computed: the protocol's analysis plan when ``weights``
        were supplied, ``"unadjusted difference in means ..."`` when they
        were not.

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np, pandas as pd
    >>> proto = sp.target_trial_protocol(
    ...     eligibility="age >= 40 and ldl > 130",
    ...     treatment_strategies=["statin", "no statin"],
    ...     assignment="randomization",
    ...     time_zero="first eligible visit",
    ...     followup_end="5 years",
    ...     outcome="incident MI",
    ... )
    >>> rng = np.random.default_rng(0)
    >>> n = 400
    >>> df = pd.DataFrame({
    ...     "age": rng.integers(40, 70, n),
    ...     "ldl": rng.normal(150, 20, n),
    ...     "statin": rng.integers(0, 2, n),
    ... })
    >>> df["mi"] = (rng.random(n) < 0.2).astype(int)
    >>> res = sp.target_trial_emulate(
    ...     proto, df, outcome_col="mi", treatment_col="statin")
    >>> type(res).__name__
    'TargetTrialResult'
    >>> res.n_eligible
    335
    >>> res.n_excluded_immortal
    65
    """

    protocol: TargetTrialProtocol
    estimate: float
    se: float
    ci: tuple[float, float]
    n_eligible: int
    n_excluded_immortal: int
    weights: np.ndarray
    method: str

    @property
    def n_excluded(self) -> int:
        """Subjects not eligible at time zero (``n_excluded_immortal``)."""
        return self.n_excluded_immortal

    def summary(self) -> str:
        return (
            f"Target Trial Emulation -- {self.protocol.causal_contrast}\n"
            f"  Estimate: {self.estimate:.4f}  SE: {self.se:.4f}  "
            f"95% CI: [{self.ci[0]:.4f}, {self.ci[1]:.4f}]\n"
            f"  n eligible: {self.n_eligible}  "
            f"n excluded (not eligible at time zero): {self.n_excluded_immortal}\n"
            f"  Method: {self.method}"
        )

    def to_paper(self, fmt: str = "markdown", title: str | None = None) -> str:
        """Render a structured Methods/Results block.

        See :func:`statspai.target_trial.to_paper` for details.
        """
        from .report import to_paper as _to_paper

        fmt_lit = cast(
            Literal["markdown", "latex", "text", "target", "jama", "bmj"],
            fmt,
        )
        return _to_paper(self, fmt=fmt_lit, title=title)


def emulate(
    protocol: TargetTrialProtocol,
    data: pd.DataFrame,
    outcome_col: str,
    treatment_col: str,
    time_zero_filter: Callable[[pd.DataFrame], pd.Series] | None = None,
    weights: np.ndarray | None = None,
) -> TargetTrialResult:
    """Emulate the target trial defined by ``protocol``.

    Analysis-stage wrapper: applies the eligibility criterion at time
    zero and returns the weighted difference in mean outcome between the
    two strategies among the eligible, with ``weights`` the inverse-
    probability weights of the analysis plan.

    No weights are estimated here. Without ``weights`` the estimate is
    the unadjusted difference in means: the right analysis of a
    randomized protocol, and a crude association for an observational
    emulation. The result's ``method`` then says so (it does not repeat
    the protocol's analysis plan), and an observational protocol gets an
    ``AssumptionWarning``. For an emulation, estimate the weights first,
    e.g. ``sp.ps_weights(sp.propensity_score(data, treat=..., covariates=
    protocol.baseline_covariates), data[treat])`` or the stabilized
    weights of ``sp.stabilized_weights`` / ``sp.clone_censor_weight``,
    and pass them. For analysis plans that are not a weighted mean
    difference (pooled logistic + IPCW, g-formula, LTMLE), call those
    estimators directly and keep the protocol as documentation.

    Parameters
    ----------
    protocol : TargetTrialProtocol
        The protocol from :func:`sp.target_trial_protocol`.
    data : pd.DataFrame
        One row per subject at time zero.
    outcome_col : str
        Outcome column (observed for every eligible subject).
    treatment_col : str
        Strategy indicator, coded 0/1.
    time_zero_filter : callable, optional
        Function of ``data`` returning a boolean mask of the eligible
        rows; replaces the protocol's ``eligibility``.
    weights : array-like, optional
        One inverse-probability weight per row of ``data`` (treatment,
        censoring, or their product). Treated as known in the standard
        error.

    Returns
    -------
    TargetTrialResult

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np, pandas as pd
    >>> proto = sp.target_trial_protocol(
    ...     eligibility="age >= 40 and ldl > 130",
    ...     treatment_strategies=["statin", "no statin"],
    ...     assignment="randomization",
    ...     time_zero="first eligible visit",
    ...     followup_end="5 years",
    ...     outcome="incident MI",
    ... )
    >>> rng = np.random.default_rng(0)
    >>> n = 400
    >>> df = pd.DataFrame({
    ...     "age": rng.integers(40, 70, n),
    ...     "ldl": rng.normal(150, 20, n),
    ...     "statin": rng.integers(0, 2, n),
    ... })
    >>> df["mi"] = (rng.random(n) < 0.2).astype(int)
    >>> res = sp.target_trial_emulate(
    ...     proto, df, outcome_col="mi", treatment_col="statin")
    >>> res.n_eligible  # eligibility filter applied at time zero
    335
    >>> bool(res.ci[0] <= res.estimate <= res.ci[1])
    True

    References
    ----------
    hernan2016using
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
    bad_a = ~np.isin(a, (0.0, 1.0))
    if bad_a.any():
        raise MethodIncompatibility(
            f"target_trial_emulate: {treatment_col!r} must be coded 0/1 with "
            f"no missing values among the eligible; {int(bad_a.sum())} row(s) "
            "are not.",
            recovery_hint="Recode the two strategies as 0 and 1 and settle "
            "the rows with a missing strategy before emulating.",
            diagnostics={"n_not_binary": int(bad_a.sum())},
        )
    if len(a) == 0 or a.min() == a.max():
        raise DataInsufficient(
            "target_trial_emulate: the eligible sample needs subjects under "
            "both strategies.",
            recovery_hint="Check the eligibility criterion and the treatment "
            "column.",
            diagnostics={"n_eligible": n_eligible},
        )
    n_missing_y = int(np.isnan(y).sum())
    if n_missing_y:
        raise DataInsufficient(
            f"target_trial_emulate: {outcome_col!r} is missing for "
            f"{n_missing_y} eligible subject(s); a mean difference over the "
            "rest would condition on being observed.",
            recovery_hint="Restrict to subjects with an observed outcome and "
            "pass inverse-probability-of-censoring weights (times the "
            "treatment weights) as weights=.",
            diagnostics={"n_missing_outcome": n_missing_y},
        )
    if weights is None:
        w = np.ones_like(a)
        method = "unadjusted difference in means (no weights supplied)"
        if protocol.assignment == "observational emulation":
            warnings.warn(
                "target_trial_emulate: no weights were supplied, so the "
                "estimate is the unadjusted difference in means, not the "
                f"protocol's analysis plan ({protocol.analysis_plan!r}). For "
                "an observational emulation pass inverse-probability weights "
                "(e.g. sp.ps_weights on a propensity score fitted to the "
                "protocol's baseline covariates).",
                AssumptionWarning,
                stacklevel=2,
            )
    else:
        w_full = np.asarray(weights, dtype=float)
        if w_full.shape != (n_total,):
            raise MethodIncompatibility(
                f"target_trial_emulate: weights must hold one value per row "
                f"of data ({n_total}); got shape {w_full.shape}.",
                recovery_hint="Pass the weights of all rows; the eligibility "
                "filter is applied to them here.",
            )
        w = w_full[mask_values]
        if not np.all(np.isfinite(w)) or (w < 0).any():
            raise MethodIncompatibility(
                "target_trial_emulate: weights must be finite and "
                "non-negative among the eligible.",
                recovery_hint="Check the weight model (a fitted probability "
                "of 0 or 1 gives an infinite weight).",
            )
        method = str(protocol.analysis_plan)

    sum_w1 = max(float(np.sum(w * a)), 1e-12)
    sum_w0 = max(float(np.sum(w * (1 - a))), 1e-12)
    m1 = float(np.sum(w * a * y) / sum_w1)
    m0 = float(np.sum(w * (1 - a) * y) / sum_w0)
    estimate = m1 - m0

    # Sandwich (HC0) variance of the two weighted means, weights treated as
    # known: sum(w^2 (y - m)^2) / (sum w)^2 per arm. Before 1.43 the
    # numerator carried w instead of w^2, so the SE changed when the
    # weights were rescaled; with unit weights the two agree.
    v1 = float(np.sum(w**2 * a * (y - m1) ** 2) / sum_w1**2)
    v0 = float(np.sum(w**2 * (1 - a) * (y - m0) ** 2) / sum_w0**2)
    se = float(np.sqrt(v1 + v0))
    ci = (estimate - 1.96 * se, estimate + 1.96 * se)

    _result = TargetTrialResult(
        protocol=protocol,
        estimate=estimate,
        se=se,
        ci=(float(ci[0]), float(ci[1])),
        n_eligible=n_eligible,
        n_excluded_immortal=n_excluded,
        weights=w,
        method=method,
    )
    try:
        from ..output._lineage import attach_provenance as _attach_prov

        _attach_prov(
            _result,
            function="sp.target_trial.emulate",
            params={
                "outcome_col": outcome_col,
                "treatment_col": treatment_col,
                "n_eligible": int(n_eligible),
                "analysis_plan": str(protocol.analysis_plan),
            },
            data=data,
            overwrite=False,
        )
    except Exception:  # pragma: no cover
        pass
    return _result
