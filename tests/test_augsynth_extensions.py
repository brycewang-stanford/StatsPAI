"""sp.augsynth: several treated units, progfunc / scm / fixedeff, refusals.

Numbers against R ``augsynth::augsynth`` are in
``tests/reference_parity/test_synth_family_R_parity.py``; these tests pin
known-truth and boundary behaviour.
"""

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import MethodIncompatibility


def _panel(effect=1.5, T=16, t0=10, seed=0):
    rng = np.random.default_rng(seed)
    ctrl = {f"c{j}": np.cumsum(rng.normal(0, 1, T)) + 2 * j for j in range(7)}
    mixes = {"a": (0.5, 0.5, 0, 0, 0, 0, 0), "b": (0, 0, 0.3, 0.7, 0, 0, 0)}
    rows = []
    for u, v in ctrl.items():
        rows += [(u, t, v[t]) for t in range(T)]
    for u, mix in mixes.items():
        base = sum(m * ctrl[f"c{k}"] for k, m in enumerate(mix))
        rows += [(u, t, base[t] + effect * (t >= t0)) for t in range(T)]
    return pd.DataFrame(rows, columns=["unit", "time", "y"])


KW = dict(outcome="y", unit="unit", time="time", treatment_time=10, placebo=False)


@pytest.mark.parametrize("progfunc", ["ridge", "none"])
def test_two_treated_units_recover_exact_effect(progfunc):
    # both treated units are exact convex combinations of donors, so is
    # their average: SCM reproduces it and the ridge correction is zero
    r = sp.augsynth(_panel(), treated_unit=["a", "b"], progfunc=progfunc, **KW)
    assert abs(r.estimate - 1.5) < 1e-6
    assert r.model_info["n_treated_units"] == 2
    assert set(r.model_info["weights_dict"]) == {f"c{j}" for j in range(7)}


def test_treated_average_equals_averaged_unit():
    """Fitting [a, b] is fitting one unit whose path is their average."""
    df = _panel(seed=3)
    avg = (
        df[df.unit.isin(["a", "b"])].groupby("time", as_index=False)["y"].mean()
    ).assign(unit="ab")
    df2 = pd.concat([df[~df.unit.isin(["a", "b"])], avg], ignore_index=True)
    r_list = sp.augsynth(df, treated_unit=["a", "b"], ridge_lambda=0.7, **KW)
    r_avg = sp.augsynth(df2, treated_unit="ab", ridge_lambda=0.7, **KW)
    np.testing.assert_allclose(r_list.estimate, r_avg.estimate, rtol=1e-12)


def test_single_element_list_equals_scalar():
    df = _panel(seed=1)
    r1 = sp.augsynth(df, treated_unit="a", **KW)
    r2 = sp.augsynth(df.query("unit != 'b'"), treated_unit=["a"], **KW)
    r3 = sp.augsynth(df.query("unit != 'b'"), treated_unit="a", **KW)
    np.testing.assert_allclose(r2.estimate, r3.estimate, rtol=1e-12)
    assert np.isfinite(r1.estimate)


def test_fixedeff_removes_level_shift():
    """A constant level shift of the treated unit is absorbed by fixedeff."""
    df = _panel(seed=2)
    df.loc[df.unit == "a", "y"] += 40.0
    r = sp.augsynth(
        df.query("unit != 'b'"), treated_unit="a", fixedeff=True, progfunc="none", **KW
    )
    assert abs(r.estimate - 1.5) < 1e-6


def test_placebo_runs_with_several_treated_units():
    df = _panel(seed=4)
    r = sp.augsynth(df, treated_unit=["a", "b"], **{**KW, "placebo": True})
    assert r.model_info["n_placebos"] == 7
    assert np.isfinite(r.se)


@pytest.mark.parametrize("pf", ["EN", "gsyn", "MCP", "rf"])
def test_unsupported_progfunc_is_refused(pf):
    with pytest.raises(MethodIncompatibility) as ei:
        sp.augsynth(_panel(), treated_unit="a", progfunc=pf, **KW)
    assert "ridge" in str(ei.value) and "none" in str(ei.value)


def test_staggered_dates_are_refused():
    with pytest.raises(MethodIncompatibility, match="staggered_synth|dates"):
        sp.augsynth(
            _panel(),
            outcome="y",
            unit="unit",
            time="time",
            treated_unit=["a", "b"],
            treatment_time=[10, 12],
        )


def test_common_date_list_is_accepted():
    r = sp.augsynth(
        _panel(),
        outcome="y",
        unit="unit",
        time="time",
        treated_unit=["a", "b"],
        treatment_time=[10, 10],
        placebo=False,
    )
    assert abs(r.estimate - 1.5) < 1e-6


def test_r_backend_refuses_extensions():
    with pytest.raises(NotImplementedError):
        sp.augsynth(_panel(), treated_unit=["a", "b"], backend="r", **KW)
    with pytest.raises(NotImplementedError):
        sp.augsynth(_panel(), treated_unit="a", fixedeff=True, backend="r", **KW)


def test_unknown_treated_unit():
    with pytest.raises(ValueError, match="not found"):
        sp.augsynth(_panel(), treated_unit=["a", "zz"], **KW)


def test_progfunc_none_is_scm_whatever_scm_says():
    # 2026-10-20 probe: progfunc='none', scm=False returned the average of
    # uniform donor weights (6.878 on the probe panel) while R
    # augsynth(progfunc = "None", scm = FALSE) returns the SCM estimate
    # (3.33428, the same as scm = TRUE; run in R augsynth on that panel).
    # Here: the SCM fit of 'a' is exact, so both settings give 1.5.
    df = _panel(seed=5).query("unit != 'b'")
    a = sp.augsynth(df, treated_unit="a", progfunc="none", scm=True, **KW)
    b = sp.augsynth(df, treated_unit="a", progfunc="none", scm=False, **KW)
    assert b.estimate == a.estimate
    assert abs(b.estimate - 1.5) < 1e-6
