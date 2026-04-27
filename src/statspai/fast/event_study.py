"""``sp.fast.event_study`` — fast event-study (TWFE) on the Phase 1+ HDFE stack.

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
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

from .within import within as _within
from .dsl import i as _i
from .inference import crve as _crve


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
        return pd.DataFrame({
            "event_time": self.event_times,
            "Estimate": self.coefs,
            "Std. Error": self.ses,
            "ci_lower": self.coefs - 1.96 * self.ses,
            "ci_upper": self.coefs + 1.96 * self.ses,
        })

    def plot(self, ax=None):  # pragma: no cover  - cosmetic
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise RuntimeError("matplotlib is required for .plot()") from exc
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 4))
        td = self.tidy()
        ax.errorbar(
            td["event_time"], td["Estimate"],
            yerr=1.96 * td["Std. Error"], fmt="o-", capsize=3,
        )
        ax.axhline(0, color="black", linewidth=0.5)
        ax.axvline(-0.5, color="grey", linestyle="--", linewidth=0.5)
        ax.set_xlabel("Event time")
        ax.set_ylabel("Estimated effect")
        ax.set_title("Event study (TWFE)")
        return ax

    def summary(self) -> str:
        return (
            f"sp.fast.event_study  |  N={self.n_obs:,}, kept={self.n_kept:,}, "
            f"event_times={list(self.event_times)}\n" +
            self.tidy().to_string(index=False, float_format=lambda v: f"{v:.4f}")
        )


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
    for col in (y, unit, time, event_time):
        if col not in data.columns:
            raise KeyError(f"data missing column {col!r}")

    cluster_col = cluster if cluster is not None else unit
    if cluster_col not in data.columns:
        raise KeyError(f"data missing cluster column {cluster_col!r}")

    df = data.copy()
    n_obs = len(df)

    # Build event-time dummies (one-hot), masking NaN rows out.
    et = df[event_time].to_numpy()
    finite = np.isfinite(et)
    # cast finite rows to int for dummy labels; non-finite get a sentinel
    et_int = np.where(finite, et, np.iinfo(np.int64).min).astype(np.int64)
    if window is not None:
        lo, hi = window
        et_int = np.where(
            finite & (et_int >= lo) & (et_int <= hi),
            et_int, np.iinfo(np.int64).min,
        )
        finite &= (et_int >= lo) & (et_int <= hi)

    # Use pd.Categorical so we control the level set
    levels = sorted({v for v in et_int[finite] if v != reference})
    if not levels:
        raise ValueError(
            "no event-time dummies after filtering — check event_time / window"
        )
    dummies = pd.DataFrame({
        f"et_{lv}": ((et_int == lv) & finite).astype(np.float64)
        for lv in levels
    }, index=df.index)
    dummy_cols = list(dummies.columns)
    df_aug = pd.concat([df, dummies], axis=1)

    # FE residualisation
    wt = _within(df_aug, fe=[unit, time], drop_singletons=drop_singletons)
    y_dem, _ = wt.transform(df_aug[y].to_numpy(dtype=np.float64))
    X_dem = wt.transform_columns(df_aug, dummy_cols).to_numpy()

    # OLS on residualised
    XtX = X_dem.T @ X_dem
    Xty = X_dem.T @ y_dem
    beta = np.linalg.solve(XtX, Xty)
    resid = y_dem - X_dem @ beta

    # Cluster-robust SE on the residualised system. The CR1 small-sample
    # factor must charge the absorbed FE rank against residual DOF — same
    # convention as ``reghdfe`` / ``fixest``. Without it, SEs are
    # systematically too small (the FE rank is "free" parameters).
    cluster_arr = df_aug.loc[wt.keep_mask, cluster_col].to_numpy()
    bread = np.linalg.inv(XtX)
    fe_dof = sum(int(g) - 1 for g in wt.n_fe)
    V = _crve(
        X_dem, resid, cluster_arr,
        bread=bread, type="cr1", extra_df=fe_dof,
    )
    se = np.sqrt(np.diag(V))

    return EventStudyResult(
        formula=f"{y} ~ event_time | {unit} + {time}  (cluster: {cluster_col})",
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
