"""``sp.fast.event_study`` on the Phase 1+ HDFE stack.

Phase 6 deliverable. This is a homogeneous-effects event-study estimator
(post-treatment leads/lags interacted with treatment), built on top of:

* ``sp.fast.within``    — the cached HDFE residualizer.
* ``sp.fast.i``         — event-time dummies with an explicit reference.
* ``sp.fast.crve``      — cluster-robust standard errors.

It is **not** a replacement for ``sp.callaway_santanna`` /
``sp.sun_abraham`` / ``sp.borusyak_jaravel_spiess`` — those handle
heterogeneous treatment effects with the appropriate group-specific
weights, and they are unchanged. ``sp.fast.event_study`` ships the
TWFE-style estimator on the new Rust path, suitable for designs with
homogeneous effects and as a baseline against which the heterogeneous
estimators can be diff-checked.

Design assumptions
------------------

* Two-way fixed effects: unit + time.
* Single binary treatment turning on at unit-specific event time.
* Symmetric event window around the event time; missing event-times
  outside the window are pooled into never-event indicators.
* Reference period defaults to ``t = -1`` (one period before treatment)
  per the canonical event-study convention (Borusyak-Jaravel-Spiess
  2024 §3.1).

References
----------
de Chaisemartin, C., D'Haultfœuille, X. (2020). Two-way fixed effects
estimators with heterogeneous treatment effects. AER 110(9): 2964–2996.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple, cast

import numpy as np
import pandas as pd

from ..exceptions import (
    DataInsufficient,
    MethodIncompatibility,
    NumericalInstability,
)
from .within import within as _within
from .inference import crve as _crve
from ._result_protocol import jsonable as _jsonable
from ._result_protocol import tidy_records as _tidy_records


@dataclass
class EventStudyResult:
    """Outcome of :func:`event_study`."""

    formula: str
    event_times: np.ndarray
    coefs: np.ndarray
    ses: np.ndarray
    n_obs: int
    n_kept: int
    n_clusters: Optional[int]
    cluster_var: Optional[str]
    reference_event_time: int

    def tidy(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "event_time": self.event_times,
                "Estimate": self.coefs,
                "Std. Error": self.ses,
                "ci_lower": self.coefs - 1.96 * self.ses,
                "ci_upper": self.coefs + 1.96 * self.ses,
            }
        )

    def plot(self, ax: Any = None) -> Any:  # pragma: no cover  - cosmetic
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise RuntimeError("matplotlib is required for .plot()") from exc
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 4))
        td = self.tidy()
        ax.errorbar(
            td["event_time"],
            td["Estimate"],
            yerr=1.96 * td["Std. Error"],
            fmt="o-",
            capsize=3,
        )
        ax.axhline(0, color="black", linewidth=0.5)
        ax.axvline(-0.5, color="grey", linestyle="--", linewidth=0.5)
        ax.set_xlabel("Event time")
        ax.set_ylabel("Estimated effect")
        ax.set_title("Event study (TWFE)")
        return ax

    def summary(self) -> str:
        table = str(
            self.tidy().to_string(
                index=False,
                float_format=lambda v: f"{v:.4f}",
            )
        )
        return (
            f"sp.fast.event_study  |  N={self.n_obs:,}, kept={self.n_kept:,}, "
            f"event_times={list(self.event_times)}\n" + table
        )

    def to_dict(self) -> dict[str, Any]:
        """Lossless JSON-safe payload for the fast TWFE event study."""
        payload = {
            "kind": "fast_event_study_result",
            "model": "twfe_event_study",
            "formula": self.formula,
            "n_obs": self.n_obs,
            "n_kept": self.n_kept,
            "n_clusters": self.n_clusters,
            "cluster_var": self.cluster_var,
            "reference_event_time": self.reference_event_time,
            "event_times": self.event_times,
            "coefficients": self.coefs,
            "standard_errors": self.ses,
            "tidy": _tidy_records(self.tidy()),
        }
        return cast(dict[str, Any], _jsonable(payload))

    def to_agent_summary(self, *, max_event_times: int = 20) -> dict[str, Any]:
        """Bounded agent-facing summary for fast TWFE event-study results."""
        n_terms = len(self.event_times)
        limit = max(int(max_event_times), 0)
        rows = _tidy_records(self.tidy().head(limit))
        payload = {
            "kind": "fast_event_study_agent_summary",
            "model": "twfe_event_study",
            "formula": self.formula,
            "n_obs": self.n_obs,
            "n_kept": self.n_kept,
            "n_clusters": self.n_clusters,
            "cluster_var": self.cluster_var,
            "reference_event_time": self.reference_event_time,
            "event_times": rows,
            "n_event_times": n_terms,
            "truncated_event_times": max(n_terms - limit, 0),
        }
        return cast(dict[str, Any], _jsonable(payload))


def event_study(
    data: pd.DataFrame,
    *,
    y: str,
    unit: str,
    time: str,
    event_time: str,
    window: Optional[Tuple[int, int]] = None,
    reference: int = -1,
    cluster: Optional[str] = None,
    drop_singletons: bool = True,
) -> EventStudyResult:
    """Two-way-FE event study on the Phase 1+ HDFE stack.

    Parameters
    ----------
    data : pd.DataFrame
    y : str
        Outcome column name.
    unit : str
        Unit identifier (e.g. firm id). Absorbed as a fixed effect.
    time : str
        Time identifier (e.g. year). Absorbed as a fixed effect.
    event_time : str
        Per-row column giving each row's relative time vs. its unit's
        event date. For never-treated units pass NaN; for treated units
        pass an integer offset (negative = pre-treatment, 0 = treatment
        period, positive = post-treatment).
    window : (lo, hi), optional
        Truncate event-time dummies to ``[lo, hi]``. Outside the window,
        rows are kept (so the unit FE still soaks up their level) but
        not given individual coefficients.
    reference : int, default -1
        Event-time period to use as the omitted reference category.
    cluster : str, optional
        Column name to cluster standard errors on. Default: cluster on
        ``unit`` (the conventional choice for panel DiD).
    drop_singletons : bool

    Returns
    -------
    EventStudyResult
    """
    if not isinstance(data, pd.DataFrame):
        raise MethodIncompatibility("fast.event_study: data must be a DataFrame")
    if len(data) < 1:
        raise DataInsufficient("fast.event_study: data must contain at least one row")
    roles = {"y": y, "unit": unit, "time": time, "event_time": event_time}
    bad_roles = [
        name for name, col in roles.items() if not isinstance(col, str) or not col
    ]
    if bad_roles:
        raise MethodIncompatibility(
            "fast.event_study: column arguments must be non-empty strings: "
            f"{bad_roles}"
        )
    missing = [col for col in roles.values() if col not in data.columns]
    if missing:
        raise MethodIncompatibility(
            f"fast.event_study: data missing columns: {missing}"
        )

    cluster_col = cluster if cluster is not None else unit
    if not isinstance(cluster_col, str) or not cluster_col:
        raise MethodIncompatibility(
            "fast.event_study: cluster must be a non-empty string column name"
        )
    if cluster_col not in data.columns:
        raise MethodIncompatibility(
            f"fast.event_study: data missing cluster column {cluster_col!r}"
        )
    if not isinstance(reference, (int, np.integer)):
        raise MethodIncompatibility(
            "fast.event_study: reference must be an integer event-time offset"
        )
    reference = int(reference)
    if window is not None:
        try:
            lo, hi = window
        except (TypeError, ValueError) as exc:
            raise MethodIncompatibility(
                "fast.event_study: window must be a (lo, hi) pair"
            ) from exc
        if isinstance(window, (str, bytes)):
            raise MethodIncompatibility(
                "fast.event_study: window must be a (lo, hi) pair"
            )
        if not isinstance(lo, (int, np.integer)) or not isinstance(
            hi, (int, np.integer)
        ):
            raise MethodIncompatibility(
                "fast.event_study: window bounds must be integer " "event-time offsets"
            )
        lo = int(lo)
        hi = int(hi)
        if lo > hi:
            raise MethodIncompatibility(
                "fast.event_study: window lower bound must be <= upper bound"
            )
        window = (lo, hi)

    df = data.copy()
    n_obs = len(df)
    y_values = df[y].to_numpy(dtype=np.float64)
    if not np.isfinite(y_values).all():
        raise MethodIncompatibility(
            f"fast.event_study: outcome column {y!r} has non-finite values"
        )

    # Build event-time dummies (one-hot), masking NaN rows out.
    try:
        et = df[event_time].to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise MethodIncompatibility(
            f"fast.event_study: event_time column {event_time!r} must be "
            "numeric with NaN for never-treated rows"
        ) from exc
    if np.isinf(et).any():
        raise MethodIncompatibility(
            f"fast.event_study: event_time column {event_time!r} contains "
            "infinite values; use NaN only for never-treated rows"
        )
    finite = np.isfinite(et)
    rounded_et = np.rint(et[finite])
    if not np.allclose(et[finite], rounded_et, atol=1e-10, rtol=0.0):
        raise MethodIncompatibility(
            f"fast.event_study: event_time column {event_time!r} must contain "
            "integer event-time offsets; non-integer finite values would be "
            "silently truncated"
        )
    # cast finite rows to int for dummy labels; non-finite get a sentinel
    et_int = np.full(et.shape, np.iinfo(np.int64).min, dtype=np.int64)
    et_int[finite] = rounded_et.astype(np.int64)
    if window is not None:
        lo, hi = window
        et_int = np.where(
            finite & (et_int >= lo) & (et_int <= hi),
            et_int,
            np.iinfo(np.int64).min,
        )
        finite &= (et_int >= lo) & (et_int <= hi)

    # Use pd.Categorical so we control the level set
    levels = sorted({v for v in et_int[finite] if v != reference})
    if not levels:
        raise DataInsufficient(
            "fast.event_study: no event-time dummies after filtering; "
            "check event_time / window"
        )
    dummies = pd.DataFrame(
        {f"et_{lv}": ((et_int == lv) & finite).astype(np.float64) for lv in levels},
        index=df.index,
    )
    dummy_cols = list(dummies.columns)
    df_aug = pd.concat([df, dummies], axis=1)

    # FE residualisation
    wt = _within(df_aug, fe=[unit, time], drop_singletons=drop_singletons)
    y_dem, _ = wt.transform(y_values)
    X_dem = wt.transform_columns(df_aug, dummy_cols).to_numpy()

    # OLS on residualised
    XtX = X_dem.T @ X_dem
    Xty = X_dem.T @ y_dem
    try:
        bread = np.linalg.inv(XtX)
    except np.linalg.LinAlgError as exc:
        raise NumericalInstability(
            "event_study normal equations are singular. Likely cause: "
            "event-time dummies are perfectly collinear with the absorbed "
            "unit/time fixed effects after filtering."
        ) from exc
    beta = bread @ Xty
    resid = y_dem - X_dem @ beta

    # Cluster-robust SE on the residualised system. The CR1 small-sample
    # factor must charge the absorbed FE rank against residual DOF — same
    # convention as ``reghdfe`` / ``fixest``. Without it, SEs are
    # systematically too small (the FE rank is "free" parameters).
    cluster_arr = df_aug.loc[wt.keep_mask, cluster_col].to_numpy()
    fe_dof = sum(int(g) - 1 for g in wt.n_fe)
    V = _crve(
        X_dem,
        resid,
        cluster_arr,
        bread=bread,
        type="cr1",
        extra_df=fe_dof,
    )
    se = np.sqrt(np.diag(V))

    formula = f"{y} ~ event_time | {unit} + {time}  " f"(cluster: {cluster_col})"
    return EventStudyResult(
        formula=formula,
        event_times=np.asarray(levels),
        coefs=beta,
        ses=se,
        n_obs=n_obs,
        n_kept=int(wt.n_kept),
        n_clusters=int(np.unique(cluster_arr).size),
        cluster_var=cluster_col,
        reference_event_time=reference,
    )


__all__ = ["event_study", "EventStudyResult"]
