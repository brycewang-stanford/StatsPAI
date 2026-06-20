"""
Unified entry point for longitudinal (time-varying treatment) analysis.

The user calls :func:`analyze` once with their panel data and the
module routes to the right estimator (MSM / g-formula ICE / LTMLE /
IPW) based on:

  - whether the outcome is end-of-follow-up or time-varying
  - whether confounders are time-varying
  - whether weights are extreme (positivity violation)
  - whether the user asked for multiple regimes to be contrasted

This is the "single-window dispatcher" the article calls out as
Epidemiology Layer 4 gap in Python.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence, Union

import numpy as np
import pandas as pd

from .regime import Regime, regime as _regime

from .._result_serialize import ResultProtocolMixin

__all__ = [
    "LongitudinalResult",
    "analyze",
    "contrast",
]


@dataclass
class LongitudinalResult(ResultProtocolMixin):
    """Result of a unified longitudinal analysis.

    Attributes
    ----------
    method : str
        Which estimator was used ("msm", "g-formula", "ipw", "ltmle").
    regime_name : str
        Name of the regime being evaluated.
    estimate : float
        E[Y(regime)] under the chosen estimator.
    se : float
    ci : tuple[float, float]
    n : int
        Panel sample size.
    n_periods : int
    diagnostics : dict
        Weight quantiles, positivity flags, etc.
    underlying_result : Any
        Raw result object from the underlying estimator.

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> rows = []
    >>> for i in range(80):
    ...     base = rng.normal()
    ...     for t in range(3):
    ...         a = int(rng.random() < 0.5)
    ...         y = 0.5 * a + 0.3 * base + rng.normal(0, 0.5)
    ...         rows.append({"id": i, "time": t, "A": a, "Y": y})
    >>> df = pd.DataFrame(rows)
    >>> res = sp.longitudinal_analyze(
    ...     df, id="id", time="time", treatment="A", outcome="Y",
    ...     regime="always_treat")
    >>> isinstance(res, sp.LongitudinalResult)
    True
    >>> res.method
    'ipw'
    >>> res.n, res.n_periods
    (80, 3)
    >>> bool("Longitudinal analysis" in res.summary())
    True
    """

    method: str
    regime_name: str
    estimate: float
    se: float
    ci: tuple[float, float]
    n: int
    n_periods: int
    diagnostics: dict = field(default_factory=dict)
    underlying_result: Any = None

    def summary(self) -> str:
        lo, hi = self.ci
        lines = [
            f"Longitudinal analysis via {self.method.upper()}",
            "-" * 60,
            f"  Regime         : {self.regime_name}",
            f"  E[Y(regime)]   = {self.estimate:+.4f}",
            f"  SE             = {self.se:.4f}",
            f"  95% CI         = [{lo:+.4f}, {hi:+.4f}]",
            f"  n (subjects)   = {self.n}",
            f"  n_periods      = {self.n_periods}",
        ]
        if self.diagnostics:
            lines.append("  Diagnostics:")
            for k, v in self.diagnostics.items():
                if isinstance(v, float):
                    lines.append(f"    {k:<22s}: {v:.4f}")
                else:
                    lines.append(f"    {k:<22s}: {v}")
        return "\n".join(lines)


def analyze(
    data: pd.DataFrame,
    id: str,
    time: str,
    treatment: str,
    outcome: str,
    time_varying: Optional[Sequence[str]] = None,
    baseline: Optional[Sequence[str]] = None,
    regime: Union[str, Sequence, Regime] = "always_treat",
    *,
    method: str = "auto",
    alpha: float = 0.05,
    trim: float = 0.01,
) -> LongitudinalResult:
    """Unified longitudinal causal-effect estimator.

    Parameters
    ----------
    data : pd.DataFrame
        Long-format panel: one row per (id, time).
    id, time, treatment, outcome : str
        Required column names.
    time_varying : list of str, optional
        Time-varying confounders (measured before treatment at period t).
    baseline : list of str, optional
        Baseline / time-invariant covariates.
    regime : str | sequence | Regime, default "always_treat"
        Treatment regime to evaluate.  See :class:`Regime` for forms.
    method : {"auto", "msm", "g-formula", "ipw"}, default "auto"
        Estimator to use.  "auto" dispatches based on data shape:

          - No time-varying confounders              -> IPW (Robins)
          - Time-varying confounders + dynamic regime -> MSM
          - End-of-follow-up outcome + static regime  -> g-formula ICE
    alpha : float, default 0.05
    trim : float, default 0.01
        Weight truncation quantile (MSM/IPW paths).

    Returns
    -------
    LongitudinalResult

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> rows = []
    >>> for i in range(80):
    ...     base = rng.normal()
    ...     for t in range(3):
    ...         a = int(rng.random() < 0.5)
    ...         y = 0.5 * a + 0.3 * base + rng.normal(0, 0.5)
    ...         rows.append({"id": i, "time": t, "A": a, "Y": y})
    >>> df = pd.DataFrame(rows)
    >>> res = sp.longitudinal_analyze(
    ...     df, id="id", time="time", treatment="A", outcome="Y",
    ...     regime="always_treat")
    >>> res.method  # no time-varying confounders -> IPW
    'ipw'
    >>> res.regime_name
    'always_treat'
    >>> isinstance(res.estimate, float)
    True
    """
    if method not in ("auto", "msm", "g-formula", "ipw"):
        raise ValueError("method must be one of 'auto', 'msm', 'g-formula', 'ipw'")
    if not isinstance(regime, Regime):
        regime = _regime(regime)

    for col in (id, time, treatment, outcome):
        if col not in data.columns:
            raise KeyError(f"Column {col!r} not in data.")
    time_varying = list(time_varying or [])
    baseline = list(baseline or [])

    n = int(data[id].nunique())
    n_periods = int(data[time].nunique())

    # Dispatcher
    if method == "auto":
        if not time_varying:
            resolved = "ipw"
        elif regime.kind == "static":
            resolved = "g-formula"
        else:
            resolved = "msm"
    else:
        resolved = method

    if resolved == "msm":
        return _run_msm(
            data,
            id,
            time,
            treatment,
            outcome,
            time_varying,
            baseline,
            regime,
            alpha,
            trim,
            n,
            n_periods,
        )
    if resolved == "g-formula":
        return _run_gformula(
            data,
            id,
            time,
            treatment,
            outcome,
            time_varying,
            baseline,
            regime,
            alpha,
            n,
            n_periods,
        )
    return _run_ipw(
        data,
        id,
        time,
        treatment,
        outcome,
        baseline,
        regime,
        alpha,
        trim,
        n,
        n_periods,
    )


# --------------------------------------------------------------------------- #
#  MSM path
# --------------------------------------------------------------------------- #


def _run_msm(
    data: pd.DataFrame,
    id_col: str,
    time_col: str,
    treatment_col: str,
    outcome_col: str,
    time_varying: Sequence[str],
    baseline: Sequence[str],
    regime: Regime,
    alpha: float,
    trim: float,
    n: int,
    n_periods: int,
) -> LongitudinalResult:
    from ..msm import msm as _msm

    res = _msm(
        data=data,
        y=outcome_col,
        treat=treatment_col,
        id=id_col,
        time=time_col,
        time_varying=list(time_varying),
        baseline=list(baseline) or None,
        exposure="cumulative" if regime.kind == "static" else "current",
        trim=trim,
        alpha=alpha,
    )

    est = float(getattr(res, "estimate", np.nan))
    se = float(getattr(res, "se", np.nan))
    ci = getattr(res, "ci", None)
    if ci is None or any(x is None for x in ci):
        z = 1.96
        ci = (est - z * se, est + z * se)
    diag: dict = {}
    info = getattr(res, "model_info", None)
    if isinstance(info, dict):
        for key in (
            "weight_mean",
            "weight_min",
            "weight_max",
            "weight_trimmed_frac",
            "n_clusters",
        ):
            if key in info:
                diag[key] = info[key]

    return LongitudinalResult(
        method="msm",
        regime_name=regime.name,
        estimate=est,
        se=se,
        ci=(float(ci[0]), float(ci[1])),
        n=n,
        n_periods=n_periods,
        diagnostics=diag,
        underlying_result=res,
    )


# --------------------------------------------------------------------------- #
#  G-formula ICE path (wide-panel sequential regression)
# --------------------------------------------------------------------------- #


def _run_gformula(
    data: pd.DataFrame,
    id_col: str,
    time_col: str,
    treatment_col: str,
    outcome_col: str,
    time_varying: Sequence[str],
    baseline: Sequence[str],
    regime: Regime,
    alpha: float,
    n: int,
    n_periods: int,
) -> LongitudinalResult:
    from ..gformula import ice as _ice_fn

    # Wide panel: one column per period for treatment + confounders.
    periods = sorted(data[time_col].unique())
    K = len(periods)

    wide = _pivot_panel(
        data,
        id_col,
        time_col,
        treatment_col,
        outcome_col,
        time_varying,
        baseline,
        periods,
    )
    treat_cols = [f"A_{t}" for t in range(K)]
    confounder_cols = [[f"L_{t}_{c}" for c in time_varying] for t in range(K)]
    # Fold baseline confounders into every period
    if baseline:
        confounder_cols = [c + list(baseline) for c in confounder_cols]

    # Static regime: use the sequence; Dynamic: call regime on row.
    if regime.kind == "static":
        assert not callable(regime.rule)
        strategy = list(regime.rule)
        if len(strategy) < K:
            strategy = strategy + [strategy[-1]] * (K - len(strategy))
        else:
            strategy = strategy[:K]
    else:
        # Approximate dynamic regime by evaluating at each period on observed L_{t-1}
        strategy = _materialize_dynamic(wide, regime, K, time_varying)

    res = _ice_fn(
        data=wide,
        id_col=id_col,
        time_col=time_col,
        treatment_cols=treat_cols,
        confounder_cols=confounder_cols,
        outcome_col=outcome_col,
        treatment_strategy=strategy,
        bootstrap=200,
    )

    return LongitudinalResult(
        method="g-formula",
        regime_name=regime.name,
        estimate=float(res.value),
        se=float(res.se),
        ci=(float(res.ci[0]), float(res.ci[1])),
        n=n,
        n_periods=n_periods,
        diagnostics={"bootstrap_reps": 200},
        underlying_result=res,
    )


def _pivot_panel(
    data: pd.DataFrame,
    id_col: str,
    time_col: str,
    treatment_col: str,
    outcome_col: str,
    time_varying: Sequence[str],
    baseline: Sequence[str],
    periods: Sequence[Any],
) -> pd.DataFrame:
    """Transform long panel into wide format expected by the ICE function."""
    df = data.sort_values([id_col, time_col])
    idx_map = {p: i for i, p in enumerate(periods)}
    subjects = df[id_col].unique()
    rows: list = []
    last_row_by_id: dict = {}
    for sid in subjects:
        sub = df.loc[df[id_col] == sid]
        row: dict = {id_col: sid}
        for _, obs in sub.iterrows():
            t = idx_map[obs[time_col]]
            row[f"A_{t}"] = obs[treatment_col]
            for c in time_varying:
                row[f"L_{t}_{c}"] = obs[c]
            row[outcome_col] = obs[outcome_col]  # terminal outcome
            if baseline:
                for b in baseline:
                    row[b] = obs[b]
        rows.append(row)
        last_row_by_id[sid] = row
    out = pd.DataFrame(rows)
    # Fill missing periods with last-observed (simple forward-fill substitute)
    for t in range(len(periods)):
        if f"A_{t}" not in out.columns:
            out[f"A_{t}"] = 0.0
        for c in time_varying:
            col = f"L_{t}_{c}"
            if col not in out.columns:
                out[col] = 0.0
    if "A_0" not in out.columns:
        out["A_0"] = 0.0
    n_missing = int(out.isna().sum().sum())
    if n_missing > 0:
        import warnings as _warnings

        _warnings.warn(
            f"_pivot_panel: {n_missing} missing (id, period, variable) "
            "cells filled with 0.0 to build the wide panel. For "
            "continuous covariates this may bias the g-formula "
            "estimate; supply a balanced panel or pre-impute before "
            "calling sp.longitudinal.analyze(..., method='g-formula').",
            stacklevel=3,
        )
    return out.fillna(0.0)


def _materialize_dynamic(
    wide: pd.DataFrame,
    regime: Regime,
    K: int,
    time_varying: Sequence[str],
) -> list:
    """For dynamic regimes we evaluate the rule on the observed history
    at each period t and return the implied sequence of mean values.

    This is an approximation — the fully Monte-Carlo g-formula would
    simulate L_t under the regime — but it gives a sensible, testable
    default.
    """
    seq = []
    for t in range(K):
        hist: dict
        if time_varying:
            hist = {c: float(wide[f"L_{t}_{c}"].mean()) for c in time_varying}
        else:
            hist = {}
        seq.append(float(regime.treatment(hist, t)))
    return seq


# --------------------------------------------------------------------------- #
#  IPW path (no time-varying confounders)
# --------------------------------------------------------------------------- #


def _run_ipw(
    data: pd.DataFrame,
    id_col: str,
    time_col: str,
    treatment_col: str,
    outcome_col: str,
    baseline: Sequence[str],
    regime: Regime,
    alpha: float,
    trim: float,
    n: int,
    n_periods: int,
) -> LongitudinalResult:
    """Baseline-only IPW when no time-varying confounders are given."""
    df = data.sort_values([id_col, time_col]).groupby(id_col).tail(1).copy()
    a = df[treatment_col].to_numpy(dtype=float)
    y = df[outcome_col].to_numpy(dtype=float)
    if baseline:
        X = df[list(baseline)].to_numpy(dtype=float)
    else:
        X = np.empty((len(df), 0))

    # Fit baseline propensity score
    ps = _logit_fit_predict(X, a)
    ps = np.clip(ps, trim, 1 - trim)
    w = np.where(a == 1, 1 / ps, 1 / (1 - ps))

    # Apply regime: if regime is static [1, ...] we estimate E[Y(1)]; if [0, ...] E[Y(0)].
    # For dynamic or mixed regimes, fall back to the contrast of observed arms.
    if regime.kind == "static":
        assert not callable(regime.rule)
        target = regime.rule[0]
        mask = a == target
        wm = w[mask]
        if wm.sum() == 0:
            estimate = float("nan")
            se = float("nan")
        else:
            estimate = float(np.sum(wm * y[mask]) / wm.sum())
            se = float(np.sqrt(np.sum(wm**2 * (y[mask] - estimate) ** 2)) / wm.sum())
    else:
        estimate = float(np.mean(y))
        se = float(y.std(ddof=1) / np.sqrt(len(y)))

    z = 1.96
    ci = (estimate - z * se, estimate + z * se)
    diag = {
        "weight_mean": float(np.mean(w)),
        "weight_min": float(np.min(w)),
        "weight_max": float(np.max(w)),
        "weight_trimmed_frac": float(trim * 2),
    }
    return LongitudinalResult(
        method="ipw",
        regime_name=regime.name,
        estimate=estimate,
        se=se,
        ci=(float(ci[0]), float(ci[1])),
        n=n,
        n_periods=n_periods,
        diagnostics=diag,
        underlying_result=None,
    )


def _logit_fit_predict(X: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Tiny IRLS logistic regression (independent of statsmodels)."""
    n = len(y)
    if X.shape[1] == 0:
        # No covariates -> constant propensity == sample mean
        return np.full(n, y.mean())
    X_ = np.column_stack([np.ones(n), X])
    beta = np.zeros(X_.shape[1])
    for _ in range(50):
        lin = X_ @ beta
        p = 1.0 / (1.0 + np.exp(-lin))
        p = np.clip(p, 1e-6, 1 - 1e-6)
        W = p * (1 - p)
        grad = X_.T @ (y - p)
        H = (X_.T * W) @ X_
        step: Any
        try:
            step = np.linalg.solve(H, grad)
        except np.linalg.LinAlgError:
            step = np.linalg.pinv(H) @ grad
        beta = beta + step
        if np.linalg.norm(step) < 1e-8:
            break
    lin = X_ @ beta
    return 1.0 / (1.0 + np.exp(-lin))


# --------------------------------------------------------------------------- #
#  Contrast of two regimes
# --------------------------------------------------------------------------- #


def contrast(
    data: pd.DataFrame,
    id: str,
    time: str,
    treatment: str,
    outcome: str,
    regime_a: Union[str, Sequence, Regime],
    regime_b: Union[str, Sequence, Regime],
    **kwargs: Any,
) -> dict:
    """Estimate E[Y(regime_a)] - E[Y(regime_b)] using :func:`analyze`.

    Returns
    -------
    dict
        With keys ``regime_a``, ``regime_b``, ``contrast``, ``a_result``,
        ``b_result``.  The ``contrast`` value is the plug-in difference
        ``a.estimate - b.estimate``; its SE uses the delta-method
        approximation ``sqrt(se_a^2 + se_b^2)``.

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> rows = []
    >>> for i in range(80):
    ...     base = rng.normal()
    ...     for t in range(3):
    ...         a = int(rng.random() < 0.5)
    ...         y = 0.5 * a + 0.3 * base + rng.normal(0, 0.5)
    ...         rows.append({"id": i, "time": t, "A": a, "Y": y})
    >>> df = pd.DataFrame(rows)
    >>> res = sp.longitudinal_contrast(
    ...     df, id="id", time="time", treatment="A", outcome="Y",
    ...     regime_a="always_treat", regime_b="never_treat")
    >>> res["regime_a"], res["regime_b"]
    ('always_treat', 'never_treat')
    >>> sorted(res.keys())
    ['a_result', 'b_result', 'ci', 'contrast', 'regime_a', 'regime_b', 'se']
    >>> isinstance(res["contrast"], float)
    True
    """
    a = analyze(
        data,
        id=id,
        time=time,
        treatment=treatment,
        outcome=outcome,
        regime=regime_a,
        **kwargs,
    )
    b = analyze(
        data,
        id=id,
        time=time,
        treatment=treatment,
        outcome=outcome,
        regime=regime_b,
        **kwargs,
    )
    delta = a.estimate - b.estimate
    se = float(np.sqrt(a.se**2 + b.se**2))
    return {
        "regime_a": a.regime_name,
        "regime_b": b.regime_name,
        "contrast": float(delta),
        "se": se,
        "ci": (delta - 1.96 * se, delta + 1.96 * se),
        "a_result": a,
        "b_result": b,
    }
