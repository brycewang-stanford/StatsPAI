"""
Bridge: DiD ≡ Synthetic Control (Sun, Xie & Zhang 2025, arXiv 2503.11375).

Identifies the ATT under either parallel trends (DiD) or the SC
factor-model condition. Both paths target the same ATT; agreement
implies the doubly-robust estimate is well-defined.
"""

from __future__ import annotations

from typing import List, Optional

import numpy as np
import pandas as pd

from .core import BridgeResult, _agreement_test, _dr_combine, _register


@_register("did_sc")
def did_sc_bridge(
    data: pd.DataFrame,
    y: str,
    unit: str,
    time: str,
    treated_unit,
    treatment_time: int,
    covariates: Optional[List[str]] = None,
    alpha: float = 0.05,
) -> BridgeResult:
    """
    Compare DiD (path A: parallel trends) against Synthetic Control
    (path B: factor-model fit) on the same single-treated-unit panel.

    Parameters
    ----------
    data : pd.DataFrame
        Long-format panel.
    y : str
        Outcome column.
    unit : str
        Unit identifier column.
    time : str
        Time period column (numeric).
    treated_unit : scalar or list
        Treated unit identifier. Single-unit case for clean comparison.
    treatment_time : int
        First treatment period.
    covariates : list[str], optional
        Pre-treatment covariates (passed through to both estimators).
    alpha : float, default 0.05
        Significance level (kept for API consistency; reported on the
        BridgeResult).
    """
    # Normalise treated unit to a single id for this bridge.
    if isinstance(treated_unit, (list, tuple, set)):
        treated_units = list(treated_unit)
    else:
        treated_units = [treated_unit]

    # ---------- Path A: classical 2×2 DID ---------- #
    df = data.copy()
    df['_post'] = (df[time] >= treatment_time).astype(int)
    df['_treat'] = df[unit].isin(treated_units).astype(int)

    # Mean outcomes in the four cells
    cell_means = df.groupby(['_treat', '_post'])[y].mean()
    try:
        m_t1_post = cell_means.loc[(1, 1)]
        m_t1_pre = cell_means.loc[(1, 0)]
        m_t0_post = cell_means.loc[(0, 1)]
        m_t0_pre = cell_means.loc[(0, 0)]
    except KeyError as e:  # pragma: no cover - validates user input
        raise ValueError(
            f"DiD path: missing cell {e!r}. Check that treated_unit "
            f"and treatment_time produce non-empty pre/post × treat/control "
            f"groups."
        )
    att_did = float((m_t1_post - m_t1_pre) - (m_t0_post - m_t0_pre))

    # SE for DID via cluster bootstrap on units (200 reps for speed)
    rng = np.random.default_rng(0)
    units = df[unit].unique()
    n_units = len(units)
    n_boot = 200
    boot = np.full(n_boot, np.nan)
    for b in range(n_boot):
        sample_units = rng.choice(units, size=n_units, replace=True)
        sub = pd.concat(
            [df[df[unit] == u] for u in sample_units], ignore_index=True
        )
        try:
            cm = sub.groupby(['_treat', '_post'])[y].mean()
            boot[b] = (
                (cm.loc[(1, 1)] - cm.loc[(1, 0)])
                - (cm.loc[(0, 1)] - cm.loc[(0, 0)])
            )
        except KeyError:
            continue
    se_did = float(np.nanstd(boot, ddof=1))

    # ---------- Path B: Synthetic Control ---------- #
    from ..synth.scm import SyntheticControl

    sc_estimate = np.nan
    sc_se = np.nan
    sc_detail = {}
    try:
        sc = SyntheticControl(
            outcome=y, unit=unit, time=time,
            treated_unit=treated_units[0],
            treatment_time=treatment_time,
            covariates=covariates,
        )
        sc.fit(data)
        # ATT ≡ average post-treatment gap in SC
        gap = sc.gap_  # Series indexed by time
        sc_estimate = float(np.mean(gap[gap.index >= treatment_time]))
        # Placebo SE: refit SC on each donor, compute post-mean gap
        donor_units = [u for u in units if u not in treated_units]
        placebo = []
        for du in donor_units[: min(20, len(donor_units))]:
            try:
                placebo_sc = SyntheticControl(
                    outcome=y, unit=unit, time=time,
                    treated_unit=du,
                    treatment_time=treatment_time,
                    covariates=covariates,
                )
                placebo_sc.fit(data)
                pg = placebo_sc.gap_
                placebo.append(np.mean(pg[pg.index >= treatment_time]))
            except Exception:
                continue
        if len(placebo) >= 3:
            sc_se = float(np.std(placebo, ddof=1))
        sc_detail = {'n_placebo': len(placebo)}
    except Exception as e:  # pragma: no cover - upstream import edge cases
        sc_detail = {'sc_error': f"{type(e).__name__}: {e}"}
        sc_estimate = att_did
        sc_se = se_did

    # ---------- Agreement test + DR combine ---------- #
    diff, diff_se, diff_p = _agreement_test(
        att_did, se_did, sc_estimate, sc_se if not np.isnan(sc_se) else se_did
    )
    est_dr, se_dr = _dr_combine(
        att_did, se_did,
        sc_estimate, sc_se if not np.isnan(sc_se) else se_did,
        diff_p,
    )

    _result = BridgeResult(
        kind="did_sc",
        path_a_name="DiD (parallel trends)",
        path_b_name="Synthetic Control (factor model)",
        estimate_a=att_did,
        estimate_b=float(sc_estimate),
        se_a=se_did,
        se_b=float(sc_se) if not np.isnan(sc_se) else se_did,
        diff=diff,
        diff_se=diff_se,
        diff_p=diff_p,
        estimate_dr=est_dr,
        se_dr=se_dr,
        n_obs=len(df),
        detail=sc_detail,
        reference="Sun, Xie & Zhang (2025), arXiv 2503.11375",
    )
    try:
        from ..output._lineage import attach_provenance as _attach_prov
        _attach_prov(
            _result,
            function="sp.bridge.did_sc_bridge",
            params={
                "y": y, "unit": unit, "time": time,
                "treatment_time": int(treatment_time),
                "covariates": list(covariates) if covariates else None,
                "alpha": alpha,
            },
            data=data,
            overwrite=False,
        )
    except Exception:  # pragma: no cover
        pass
    return _result
