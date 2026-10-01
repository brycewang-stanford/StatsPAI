"""Smoke tests for v0.10 staggered DiD frontiers."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp


@pytest.fixture
def staggered_panel():
    rng = np.random.default_rng(42)
    n_units, n_time = 20, 10
    rows = []
    for u in range(n_units):
        first_treat = 0 if u < 8 else (4 if u < 14 else 6)  # 3 cohorts
        for t in range(n_time):
            post = first_treat > 0 and t >= first_treat
            x = rng.standard_normal()
            y = (
                u * 0.1
                + t * 0.2
                + (2.0 if post else 0.0)
                + 0.3 * x
                + rng.standard_normal()
            )
            rows.append(
                {
                    "unit": f"u{u}",
                    "time": t,
                    "y": y,
                    "x": x,
                    "first_treat": first_treat,
                }
            )
    return pd.DataFrame(rows)


def test_did_bcf(staggered_panel):
    res = sp.did_bcf(
        staggered_panel,
        y="y",
        treat="first_treat",
        time="time",
        id="unit",
        covariates=["x"],
        n_trees=20,
        seed=0,
    )
    assert hasattr(res, "estimate")
    # True ATT ≈ 2.0
    assert -1.0 < res.estimate < 5.0
    assert res.method.startswith("DiD-BCF")
    assert "catt_by_cohort" in res.model_info
    # Known truth: ATT = 2.0.  The pre-1.29 implementation reported
    # 1.875 with SE 0.055 on this 20-unit panel -- a standard error built
    # from the dispersion of fitted CATEs, several times too small for
    # 8 never-treated and 12 treated units with unit-period noise sd 1.
    assert 1.0 < res.estimate < 3.0
    assert 0.1 < res.se < 1.0
    assert res.ci[0] < 2.0 < res.ci[1]
    assert set(res.model_info["catt_by_cohort"]) == {4.0, 6.0}
    assert "bootstrap" in res.model_info["se_method"]


def test_cohort_anchored(staggered_panel):
    res = sp.cohort_anchored_event_study(
        staggered_panel,
        y="y",
        treat="first_treat",
        time="time",
        id="unit",
        leads=2,
        lags=2,
    )
    assert hasattr(res, "estimate")
    es = res.model_info["event_study"]
    assert isinstance(es, pd.DataFrame)
    assert "rel_time" in es.columns
    # Should produce post-treatment effects in the right ballpark
    assert -1.0 < res.estimate < 5.0


def test_design_robust(staggered_panel):
    res = sp.design_robust_event_study(
        staggered_panel,
        y="y",
        treat="first_treat",
        time="time",
        id="unit",
        leads=2,
        lags=2,
    )
    assert hasattr(res, "estimate")
    es = res.model_info["event_study"]
    assert isinstance(es, pd.DataFrame)
    assert "diagnostics" in res.model_info
    # ⚠️ correctness fix (2026-07): the headline SE is now sqrt(w'Vw) over
    # the post-period block of the cluster-robust vcov instead of the
    # independence approximation sqrt(sum se_k^2)/m.  Cross-validated
    # against a 400-draw cluster bootstrap of the full procedure on this
    # fixture: bootstrap SE 0.31009 vs analytic 0.30652 (the old pinned
    # 0.24139 was ~22% below the bootstrap truth).
    #
    # ⚠️ correctness fix (2026-10): treated rows later than ``lags`` now get
    # their own unreported dummy. They used to sit in the omitted category
    # and act as comparisons, which pulled this fixture's estimate to
    # 1.2399 against a true effect of 2.0 (2.5 SE low). The 2026-07
    # bootstrap check validated the SE of that procedure, not its target.
    assert res.ci[0] < 2.0 < res.ci[1]
    assert res.model_info["diagnostics"]["n_obs_binned_post"] == 24
    np.testing.assert_allclose(
        [res.estimate, res.se, res.pvalue, res.ci[0], res.ci[1]],
        [
            1.9267159263975826,
            0.2806863920186107,
            6.681753714428635e-12,
            1.3765807070906146,
            2.4768511457045506,
        ],
        atol=1e-12,
    )
    np.testing.assert_allclose(
        es[["rel_time", "att", "se"]].to_numpy(),
        np.array(
            [
                [-2.0, 0.288411, 0.407313],
                [0.0, 1.935775, 0.373040],
                [1.0, 1.888157, 0.341940],
                [2.0, 1.956215, 0.474201],
            ]
        ),
        atol=5e-7,
    )


def test_did_misclassified_no_correction(staggered_panel):
    # pi_misclass=0, anticipation=0 → should reproduce naive ATT
    res = sp.did_misclassified(
        staggered_panel,
        y="y",
        treat="first_treat",
        time="time",
        id="unit",
        pi_misclass=0.0,
        anticipation_periods=0,
    )
    assert hasattr(res, "estimate")
    assert abs(res.model_info["naive_att"] - res.estimate) < 1e-6


def test_did_misclassified_with_correction(staggered_panel):
    # pi_misclass=0.1 → corrected estimate larger in magnitude
    res = sp.did_misclassified(
        staggered_panel,
        y="y",
        treat="first_treat",
        time="time",
        id="unit",
        pi_misclass=0.1,
        anticipation_periods=1,
    )
    assert res.model_info["misclass_factor"] > 1.0


def test_did_misclassified_invalid_pi():
    df = pd.DataFrame(
        {
            "y": np.random.randn(20),
            "first_treat": [0] * 10 + [3] * 10,
            "time": list(range(10)) * 2,
            "unit": [f"u{i}" for i in range(2) for _ in range(10)],
        }
    )
    with pytest.raises(ValueError, match="pi_misclass"):
        sp.did_misclassified(
            df, y="y", treat="first_treat", time="time", id="unit", pi_misclass=0.6
        )


def test_did_bcf_compares_cohorts_over_their_own_calendar_windows():
    """Regression: never-treated units were split at the median period.

    With a common *non-linear* trend, a cohort's long difference (post mean
    minus pre mean over its own adoption date) differs from the controls'
    long difference taken over a different split, so the old implementation
    returned cohort CATTs of 0.01 and 1.64 on this design (true effect 1 for
    both). Linear trends cancel under any split, so the trend must be
    non-linear for the test to bite.
    """
    rng = np.random.default_rng(0)
    n_units, n_time = 600, 10
    g = rng.choice([0, 3, 8], size=n_units, p=[0.4, 0.3, 0.3])
    frames = []
    for t in range(1, n_time + 1):
        d = (g > 0) & (t >= g)
        y = rng.normal(size=n_units) + 0.1 * t**2 + 1.0 * d
        frames.append(pd.DataFrame({"id": np.arange(n_units), "t": t, "g": g, "y": y}))
    df = pd.concat(frames, ignore_index=True)
    res = sp.did_bcf(df, y="y", treat="g", time="t", id="id")
    catt = res.model_info["catt_by_cohort"]
    se = res.model_info["se_by_cohort"]
    for cohort in (3.0, 8.0):
        assert abs(catt[cohort] - 1.0) < 4 * se[cohort], (cohort, catt[cohort])
    assert abs(res.estimate - 1.0) < 4 * res.se


def test_did_bcf_accepts_an_absorbing_indicator_and_rejects_switching():
    rng = np.random.default_rng(1)
    rows = []
    for i in range(80):
        g = 0 if i < 40 else 4
        for t in range(1, 7):
            d = int(g > 0 and t >= g)
            rows.append({"id": i, "t": t, "d": d, "y": 0.2 * t + d + rng.normal()})
    df = pd.DataFrame(rows)
    res = sp.did_bcf(df, y="y", treat="d", time="t", id="id")
    assert set(res.model_info["catt_by_cohort"]) == {4.0}
    df.loc[(df["id"] == 50) & (df["t"] == 6), "d"] = 0
    with pytest.raises(sp.MethodIncompatibility, match="switches off"):
        sp.did_bcf(df, y="y", treat="d", time="t", id="id")
