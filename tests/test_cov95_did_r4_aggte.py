"""Coverage round-4 — aggregation of Callaway-Sant'Anna ATT(g,t).

Drives ``sp.aggte`` across every aggregation type (simple / dynamic /
group / calendar), the multiplier bootstrap *and* the analytic-SE
fallback, the ``min_e`` / ``max_e`` / ``balance_e`` event-time filters,
``na_rm`` NaN-dropping, and the input-validation raises.  The DGP has a
constant +2 ATT switched on at each cohort's adoption date, so every
consistent aggregation must recover an overall ATT near 2.  Assertions
check real properties (sign, magnitude window, p-values in [0,1], CI
ordering, frame shapes) and that bad inputs raise — never fabricated
numbers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp

TRUE_ATT = 2.0


def _cs_panel(seed=0, n_units=120, n_periods=10, att=TRUE_ATT):
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_units):
        g = [0, 5, 7][u % 3]
        fe = rng.normal()
        x1 = rng.normal()
        for t in range(1, n_periods + 1):
            treated_now = (g > 0) and (t >= g)
            te = att if treated_now else 0.0
            y = fe + 0.3 * t + 0.5 * x1 + te + rng.normal(0, 0.4)
            rows.append({"unit": u, "time": t, "y": y, "g": g, "x1": x1})
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def cs_result():
    df = _cs_panel()
    return sp.callaway_santanna(df, y="y", g="g", t="time", i="unit")


def _ci_ok(res):
    lo, hi = res.ci
    assert lo <= res.estimate <= hi
    assert 0.0 <= res.pvalue <= 1.0


@pytest.mark.parametrize("typ", ["simple", "dynamic", "group", "calendar"])
def test_aggte_types_recover_att(cs_result, typ):
    agg = sp.aggte(cs_result, type=typ, n_boot=150, random_state=0)
    assert abs(agg.estimate - TRUE_ATT) < 0.6
    _ci_ok(agg)
    assert np.isfinite(agg.se) and agg.se > 0
    # per-cell detail frame is populated
    assert len(agg.detail) >= 1
    assert (agg.detail["pvalue"].between(0, 1)).all()


def test_aggte_dynamic_has_cband(cs_result):
    agg = sp.aggte(cs_result, type="dynamic", cband=True, n_boot=200, random_state=1)
    assert "cband_lower" in agg.detail.columns
    assert "cband_upper" in agg.detail.columns
    # uniform band is at least as wide as the pointwise CI
    d = agg.detail
    assert (d["cband_lower"] <= d["ci_lower"] + 1e-9).all()
    assert (d["cband_upper"] >= d["ci_upper"] - 1e-9).all()


def test_aggte_simple_no_cband(cs_result):
    agg = sp.aggte(cs_result, type="simple", n_boot=100, random_state=0)
    assert "cband_lower" not in agg.detail.columns


def test_aggte_event_time_window(cs_result):
    agg = sp.aggte(
        cs_result, type="dynamic", min_e=-2, max_e=3, n_boot=100, random_state=0
    )
    rt = agg.detail["time"] if "time" in agg.detail.columns else None
    # the dynamic detail labels event time in the dim column ('time')
    assert (agg.detail.iloc[:, 0] >= -2).all()
    assert (agg.detail.iloc[:, 0] <= 3).all()
    _ci_ok(agg)


def test_aggte_balance_e(cs_result):
    agg = sp.aggte(cs_result, type="dynamic", balance_e=2, n_boot=100, random_state=0)
    # balanced window keeps only e in [.., 2]
    assert (agg.detail.iloc[:, 0] <= 2).all()
    _ci_ok(agg)


def test_aggte_analytic_fallback(cs_result):
    # bstrap=False -> conservative analytic SEs (no influence-fn path)
    agg = sp.aggte(cs_result, type="group", bstrap=False)
    assert np.isfinite(agg.se) and agg.se > 0
    _ci_ok(agg)


def test_aggte_na_rm_drops_nan_cell(cs_result):
    # Inject a NaN ATT cell into a copy of the result; na_rm should drop it.
    import copy

    res = copy.copy(cs_result)
    det = cs_result.detail.copy()
    det.loc[det.index[0], "att"] = np.nan
    res.detail = det
    agg = sp.aggte(res, type="simple", na_rm=True, bstrap=False)
    assert np.isfinite(agg.estimate)


def test_aggte_bad_type_raises(cs_result):
    with pytest.raises(ValueError):
        sp.aggte(cs_result, type="nonsense")


def test_aggte_bad_boottype_raises(cs_result):
    with pytest.raises(NotImplementedError):
        sp.aggte(cs_result, boot_type="bayesian")


def test_aggte_non_cs_result_raises():
    # A plain DiD result has no ATT(g,t) grid -> aggte must reject it.
    df = sp.dgp_did(n_units=40, n_periods=6, effect=1.0, seed=0)
    r = sp.did(df, y="y", treat="treated", time="time", id="unit", method="twfe")
    with pytest.raises((ValueError, AttributeError, TypeError)):
        sp.aggte(r, type="simple")
