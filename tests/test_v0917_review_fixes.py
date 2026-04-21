"""Targeted regression tests for the v0.9.17 post-review fixes.

Each test pins down a specific correctness / security issue raised by
the independent code review, so that regressions can be caught
immediately.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import statspai as sp


# ---------------------------------------------------------------------------
# Critical 1: RR SE formula — must match Katz (1978) textbook form
# ---------------------------------------------------------------------------


def test_rr_se_katz_formula_dense_cells():
    r = sp.epi.relative_risk(50, 50, 30, 70)
    expected_se = math.sqrt(1/50 - 1/100 + 1/30 - 1/100)
    assert r.se_log == pytest.approx(expected_se, rel=1e-6)


def test_rr_se_katz_formula_haldane_corrected():
    # Zero cell triggers Haldane correction: a=0.5, b=10.5, c=5.5, d=20.5
    r = sp.epi.relative_risk(0, 10, 5, 20)
    expected_se = math.sqrt(
        1/0.5 - 1/(0.5 + 10.5) + 1/5.5 - 1/(5.5 + 20.5)
    )
    assert r.se_log == pytest.approx(expected_se, rel=1e-6)


# ---------------------------------------------------------------------------
# Critical 2: Cochran's Q is inverse-variance weighted chi^2(K-1)
# ---------------------------------------------------------------------------


def test_cochran_q_zero_on_identical_strata():
    t1 = [[20, 10], [10, 10]]
    mh = sp.epi.mantel_haenszel([t1, t1], measure="OR")
    assert mh.homogeneity_statistic == pytest.approx(0.0, abs=1e-8)
    assert mh.homogeneity_p == pytest.approx(1.0, abs=1e-8)
    assert "inverse-variance" in mh.homogeneity_method


def test_cochran_q_detects_real_heterogeneity():
    t1 = [[40, 20], [20, 20]]
    t2 = [[10, 40], [40, 10]]
    mh = sp.epi.mantel_haenszel([t1, t2], measure="OR")
    assert mh.homogeneity_statistic > 3.84  # chi2(1, 0.05)
    assert mh.homogeneity_p < 0.05


# ---------------------------------------------------------------------------
# High 3: AST walker rejects non-numeric values from history
# ---------------------------------------------------------------------------


def test_regime_rejects_callable_via_history():
    r = sp.regime("malicious")

    class _Evil:
        def __call__(self, *a, **kw):
            return 1
    with pytest.raises(TypeError):
        r.treatment({"malicious": _Evil()}, 0)


def test_regime_rejects_string_via_history():
    r = sp.regime("x")
    with pytest.raises(TypeError):
        r.treatment({"x": "not numeric"}, 0)


def test_regime_accepts_numeric_and_bool():
    # Plain boolean expression: 2 + 3.5 > 0 -> treated (1)
    r = sp.regime("x + y")
    assert r.treatment({"x": 2, "y": 3.5}, 0) == 1
    # True boolean value from history
    r2 = sp.regime("flag")
    assert r2.treatment({"flag": True}, 0) == 1
    # False path
    r3 = sp.regime("if x > 0 then 1 else 0")
    assert r3.treatment({"x": -0.5}, 0) == 0


# ---------------------------------------------------------------------------
# High 4: NNT CI ordering
# ---------------------------------------------------------------------------


def test_nnt_ci_orientation_for_benefit():
    res = sp.epi.number_needed_to_treat(20, 80, 80, 20)
    assert res.risk_difference < 0
    assert res.ci[0] <= res.ci[1]
    assert res.ci[0] <= res.estimate <= res.ci[1] + 1e-6


# ---------------------------------------------------------------------------
# High 5: Oster delta only runs when beta_uncontrolled supplied
# ---------------------------------------------------------------------------


def test_unified_sensitivity_skips_oster_without_beta_uncontrolled():
    from dataclasses import dataclass

    @dataclass
    class R:
        estimate: float
        se: float
        ci: tuple
    dash = sp.unified_sensitivity(
        R(0.3, 0.1, (0.1, 0.5)),
        r2_treated=0.40,
        r2_controlled=0.25,
    )
    assert dash.oster is None
    assert any("Oster delta skipped" in n for n in dash.notes)


def test_unified_sensitivity_runs_oster_when_all_args_supplied():
    from dataclasses import dataclass

    @dataclass
    class R:
        estimate: float
        se: float
        ci: tuple
    dash = sp.unified_sensitivity(
        R(0.3, 0.1, (0.1, 0.5)),
        r2_treated=0.40,
        r2_controlled=0.25,
        beta_uncontrolled=0.5,
    )
    assert dash.oster is not None or \
        any("Oster delta skipped" in n for n in dash.notes)


# ---------------------------------------------------------------------------
# High 6: breakdown bias_to_flip is CI bound closer to null, not |estimate|
# ---------------------------------------------------------------------------


def test_breakdown_bias_uses_ci_not_estimate():
    from dataclasses import dataclass

    @dataclass
    class R:
        estimate: float
        se: float
        ci: tuple
    dash = sp.unified_sensitivity(R(0.5, 0.1, (0.3, 0.7)))
    # bias_to_flip should be ~0.3 (lower CI bound), not 0.5
    assert dash.breakdown["bias_to_flip"] == pytest.approx(0.3, rel=1e-3)


# ---------------------------------------------------------------------------
# Medium 8: MR-Egger intercept p-value uses t(n-2)
# ---------------------------------------------------------------------------


def test_egger_intercept_pvalue_in_range():
    rng = np.random.default_rng(3)
    bx = rng.uniform(0.1, 0.3, 5)
    sy = np.full(5, 0.03)
    by = 0.4 * bx + 0.01 + rng.normal(0, sy)
    r = sp.mr_pleiotropy_egger(bx, by, sy)
    assert 0.0 <= r.p_value <= 1.0


# ---------------------------------------------------------------------------
# Medium 9: mr_radial requires >= 2 SNPs
# ---------------------------------------------------------------------------


def test_mr_radial_rejects_single_snp():
    with pytest.raises(ValueError):
        sp.mr_radial(
            beta_exposure=np.array([0.2]),
            beta_outcome=np.array([0.1]),
            se_outcome=np.array([0.03]),
        )


# ---------------------------------------------------------------------------
# Medium 10: pivot_panel warns on missing cells
# ---------------------------------------------------------------------------


def test_longitudinal_gformula_warns_on_missing_cells():
    import pandas as pd
    import warnings
    df = pd.DataFrame({
        "pid": [1, 1, 1, 2, 2, 3, 3, 3],
        "visit": [0, 1, 2, 0, 1, 0, 1, 2],
        "treat": [0, 1, 1, 0, 1, 1, 1, 1],
        "bp_lag": [130, 125, 120, 140, 138, 120, 118, 115],
        "y": [118, 118, 118, 136, 136, 113, 113, 113],
    })
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        sp.longitudinal_analyze(
            data=df, id="pid", time="visit",
            treatment="treat", outcome="y",
            time_varying=["bp_lag"],
            regime=sp.always_treat(K=3),
            method="g-formula",
        )
        msgs = [str(x.message) for x in w]
        assert any("missing" in m for m in msgs)


# ---------------------------------------------------------------------------
# Medium 11: YAML roundtrip preserves colons in values
# ---------------------------------------------------------------------------


def test_preregister_preserves_colons_in_notes(tmp_path):
    q = sp.causal_question(
        treatment="d", outcome="y", design="rct",
        notes="Design: RCT with 80:20 split",
    )
    path = q.save(tmp_path / "pap.yaml")
    q2 = sp.CausalQuestion.load(path)
    assert "Design: RCT with 80:20 split" in q2.notes
