"""Coverage round-3 (final) — synth report formatting and BSTS validation.

Targets the report-builder branches of ``synth/report.py`` (text /
markdown / latex output, fit-quality labels, weight / period truncation,
sensitivity sections) and the validation guards of the BSTS /
CausalImpact estimators (``causal_impact`` wide-format + ``bsts_synth``
long-format panel interface).

All pure-numpy (BSTS uses a hand-rolled Kalman filter — no pymc/tfp).
File output is written to pytest ``tmp_path`` only, never the repo root.
Assertions check structural properties (non-empty report bodies, correct
exceptions); no estimator numbers are fabricated.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from statspai.exceptions import DataInsufficient, MethodIncompatibility
from statspai.synth.report import synth_report, synth_report_to_file
from statspai.synth.bsts import causal_impact, bsts_synth

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
# synth_report — output formats + sensitivity
# ===========================================================================
@pytest.mark.parametrize("output", ["text", "markdown", "latex"])
def test_synth_report_output_formats(output):
    rep = synth_report(
        _panel(0),
        outcome="y",
        unit="unit",
        time="time",
        treated_unit="treated",
        treatment_time=T_TREAT,
        method="classic",
        output=output,
        sensitivity=False,
    )
    assert isinstance(rep, str) and len(rep) > 50


def test_synth_report_with_sensitivity():
    rep = synth_report(
        _panel(1),
        outcome="y",
        unit="unit",
        time="time",
        treated_unit="treated",
        treatment_time=T_TREAT,
        method="classic",
        output="text",
        sensitivity=True,
    )
    assert isinstance(rep, str)
    assert "Synthetic Control" in rep or "ATT" in rep or len(rep) > 50


@pytest.mark.parametrize("output", ["markdown", "latex"])
def test_synth_report_sensitivity_sections(output):
    # Drives the markdown / latex "Sensitivity Analysis" section bodies.
    rep = synth_report(
        _panel(5),
        outcome="y",
        unit="unit",
        time="time",
        treated_unit="treated",
        treatment_time=T_TREAT,
        method="classic",
        output=output,
        sensitivity=True,
    )
    assert isinstance(rep, str) and len(rep) > 50


def test_synth_report_unknown_output_raises():
    with pytest.raises(ValueError):
        synth_report(
            _panel(2),
            outcome="y",
            unit="unit",
            time="time",
            treated_unit="treated",
            treatment_time=T_TREAT,
            output="xml",
        )


def test_synth_report_to_file_writes_tmp(tmp_path):
    # CRITICAL: pass an absolute tmp_path filename so the default
    # ``report.md`` is NEVER written to the repo root.
    out_path = tmp_path / "synth_report.md"
    synth_report_to_file(
        _panel(3),
        outcome="y",
        unit="unit",
        time="time",
        treated_unit="treated",
        treatment_time=T_TREAT,
        output="markdown",
        sensitivity=False,
        filename=str(out_path),
    )
    assert out_path.exists()
    body = out_path.read_text(encoding="utf-8")
    assert len(body) > 50


def test_synth_report_large_donor_pool_truncation():
    # Many donors -> the ">15 weights / >20 periods" truncation branches.
    rep = synth_report(
        _panel(4, n_donors=25, n_t=40),
        outcome="y",
        unit="unit",
        time="time",
        treated_unit="treated",
        treatment_time=21,
        method="classic",
        output="text",
        sensitivity=False,
    )
    assert isinstance(rep, str) and len(rep) > 50


# ===========================================================================
# causal_impact (wide format) — validation
# ===========================================================================
def _wide(n=20, seed=0):
    rng = np.random.default_rng(seed)
    return pd.DataFrame(
        {
            "y": np.arange(n, dtype=float) + rng.normal(0, 0.1, n),
            "x1": np.arange(n, dtype=float) + 0.5,
            "x2": np.arange(n, dtype=float) * 0.9,
        },
        index=range(n),
    )


def test_causal_impact_missing_outcome_column_raises():
    wide = _wide()
    with pytest.raises(MethodIncompatibility):
        causal_impact(
            wide, pre_period=(0, 9), post_period=(10, 19), outcome="not_a_col"
        )


def test_causal_impact_missing_covariate_raises():
    wide = _wide()
    with pytest.raises(MethodIncompatibility):
        causal_impact(
            wide,
            pre_period=(0, 9),
            post_period=(10, 19),
            outcome="y",
            covariates=["nope"],
        )


def test_causal_impact_empty_period_raises():
    wide = _wide()
    with pytest.raises(DataInsufficient):
        # pre-period outside the index -> no observations
        causal_impact(wide, pre_period=(100, 109), post_period=(10, 19))


def test_causal_impact_overlapping_periods_raises():
    wide = _wide()
    with pytest.raises(MethodIncompatibility):
        causal_impact(wide, pre_period=(0, 15), post_period=(10, 19))


# ===========================================================================
# bsts_synth (long-format panel) — validation
# ===========================================================================
def test_bsts_synth_runs():
    r = bsts_synth(
        _panel(0),
        outcome="y",
        unit="unit",
        time="time",
        treated_unit="treated",
        treatment_time=T_TREAT,
        n_simulations=200,
        seed=1,
    )
    assert np.isfinite(r.estimate)


def test_bsts_synth_bad_data_type_raises():
    with pytest.raises(MethodIncompatibility):
        bsts_synth(
            [1, 2, 3],
            outcome="y",
            unit="unit",
            time="time",
            treated_unit="treated",
            treatment_time=T_TREAT,
        )


def test_bsts_synth_missing_column_raises():
    df = _panel(1).drop(columns=["y"])
    with pytest.raises(MethodIncompatibility):
        bsts_synth(
            df,
            outcome="y",
            unit="unit",
            time="time",
            treated_unit="treated",
            treatment_time=T_TREAT,
        )


def test_bsts_synth_missing_treated_raises():
    with pytest.raises(DataInsufficient):
        bsts_synth(
            _panel(2),
            outcome="y",
            unit="unit",
            time="time",
            treated_unit="ghost",
            treatment_time=T_TREAT,
        )


def test_bsts_synth_missing_covariate_raises():
    with pytest.raises(MethodIncompatibility):
        bsts_synth(
            _panel(3),
            outcome="y",
            unit="unit",
            time="time",
            treated_unit="treated",
            treatment_time=T_TREAT,
            covariates=["does_not_exist"],
        )
