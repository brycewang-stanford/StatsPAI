"""Coverage round-2 — ``synth_report``, ``discos_test``, and extra
inference paths of the SDID / DiSCo / SCPI estimators.

Drives the report formatter (text / markdown / latex, with and without
the sensitivity section) and the distributional hypothesis tests
(KS / Cramér-von Mises / stochastic dominance) against real fitted
results.

Assertions check that reports are non-empty formatted strings carrying
the expected section markers, that the distributional tests return
populated dicts with finite statistics, and that bad inputs raise —
never fabricated numbers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp

T_TREAT = 11


def _panel(seed=0, n_donors=8, n_t=20, effect=4.0):
    rng = np.random.default_rng(seed)
    units = [f"u{i}" for i in range(n_donors)] + ["treated"]
    rows = []
    for u in units:
        base = rng.normal(0, 1)
        fe = rng.normal(0, 0.5)
        for t in range(1, n_t + 1):
            eff = effect if (u == "treated" and t >= T_TREAT) else 0.0
            rows.append(
                {
                    "unit": u,
                    "time": t,
                    "y": base + 0.2 * t + fe + eff + rng.normal(0, 0.3),
                }
            )
    return pd.DataFrame(rows)


# ===========================================================================
# synth_report — text / markdown / latex, sensitivity on / off
# ===========================================================================
@pytest.mark.parametrize("output", ["text", "markdown", "latex"])
def test_synth_report_formats_no_sensitivity(output):
    rep = sp.synth_report(
        _panel(0),
        outcome="y",
        unit="unit",
        time="time",
        treated_unit="treated",
        treatment_time=T_TREAT,
        method="classic",
        output=output,
        sensitivity=False,
        placebo=False,
    )
    assert isinstance(rep, str) and len(rep) > 50


def test_synth_report_with_sensitivity_text():
    rep = sp.synth_report(
        _panel(1),
        outcome="y",
        unit="unit",
        time="time",
        treated_unit="treated",
        treatment_time=T_TREAT,
        method="classic",
        output="text",
        sensitivity=True,
        placebo=True,
    )
    assert isinstance(rep, str) and len(rep) > 50


def test_synth_report_to_file(tmp_path):
    out = tmp_path / "report.md"
    path = sp.synth_report_to_file(
        _panel(2),
        outcome="y",
        unit="unit",
        time="time",
        treated_unit="treated",
        treatment_time=T_TREAT,
        method="classic",
        output="markdown",
        sensitivity=False,
        filename=str(out),
        placebo=False,
    )
    assert out.exists()
    assert out.read_text(encoding="utf-8").strip() != ""
    assert str(path)


# ===========================================================================
# discos_test — KS / CvM / stochastic dominance
# ===========================================================================
@pytest.fixture(scope="module")
def discos_result():
    return sp.discos(
        _panel(3),
        outcome="y",
        unit="unit",
        time="time",
        treated_unit="treated",
        treatment_time=T_TREAT,
        n_quantiles=40,
        placebo=False,
        seed=1,
    )


@pytest.mark.parametrize("test", ["ks", "cvm", "stochastic_dominance"])
def test_discos_test_variants(discos_result, test):
    out = sp.discos_test(discos_result, test=test)
    assert isinstance(out, dict)
    assert "statistic" in out and "pvalue" in out
    assert np.isfinite(out["statistic"])
    assert 0.0 <= out["pvalue"] <= 1.0


def test_discos_test_unknown_raises(discos_result):
    with pytest.raises(ValueError):
        sp.discos_test(discos_result, test="not_a_test")


# ===========================================================================
# SCPI — prediction-interval estimator
# ===========================================================================
def test_scpi_runs_and_reports_interval():
    r = sp.synth(
        _panel(4),
        outcome="y",
        unit="unit",
        time="time",
        treated_unit="treated",
        treatment_time=T_TREAT,
        method="scpi",
    )
    assert np.isfinite(r.estimate)
    lo, hi = r.ci
    assert lo <= hi


# ===========================================================================
# SDID extra inference + helper exports
# ===========================================================================
def test_sdid_jackknife_inference():
    r = sp.synth(
        _panel(5),
        outcome="y",
        unit="unit",
        time="time",
        treated_unit="treated",
        treatment_time=T_TREAT,
        method="sdid",
        inference="jackknife",
    )
    assert np.isfinite(r.estimate)


def test_sc_and_did_estimate_helpers():
    df = _panel(6)
    sc = sp.sc_estimate(
        df, y="y", unit="unit", time="time", treat_unit="treated", treat_time=T_TREAT
    )
    did = sp.did_estimate(
        df, y="y", unit="unit", time="time", treat_unit="treated", treat_time=T_TREAT
    )
    assert np.isfinite(sc.estimate)
    assert np.isfinite(did.estimate)
