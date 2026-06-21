"""Coverage for previously-untested synthetic-control helpers.

All tests run on ``sp.dgp_synth`` panels with a known injected treatment
effect, so the recovery checks are anchored to ground truth: every estimator
in this family should return an ATT close to the planted effect, the scest
simplex weights must form a convex combination, and the post-treatment
distributional tests (discos / stochastic dominance) should fire when the
effect is large and strictly positive.

Functions exercised: ``scdata``, ``scest``, ``qqsynth``, ``robust_synth``,
``demeaned_synth``, ``staggered_synth``, ``synth_donor_sensitivity``,
``synth_rmspe_filter``, ``discos_test``, ``stochastic_dominance``.
"""

import numpy as np
import pytest

import statspai as sp

EFFECT = 6.0
TREAT_T = 20
N_PERIODS = 30
N_UNITS = 15


@pytest.fixture
def panel():
    return sp.dgp_synth(
        n_units=N_UNITS,
        n_periods=N_PERIODS,
        treated_unit=0,
        treatment_time=TREAT_T,
        effect=EFFECT,
        seed=1,
    )


@pytest.fixture
def base_kwargs(panel):
    return dict(
        data=panel,
        outcome="y",
        unit="unit",
        time="time",
        treated_unit=0,
        treatment_time=TREAT_T,
    )


# --------------------------------------------------------------------------
# scpi-style data prep + estimate
# --------------------------------------------------------------------------
def test_scdata_partitions_times_and_donors(base_kwargs):
    sc = sp.scdata(**base_kwargs)
    assert len(sc["donor_names"]) == N_UNITS - 1
    assert len(sc["pre_times"]) == TREAT_T
    assert len(sc["post_times"]) == N_PERIODS - TREAT_T
    assert sc["Y_pre"].shape[0] == TREAT_T
    assert sc["Y_post"].shape[0] == N_PERIODS - TREAT_T
    np.testing.assert_allclose(
        sc["Y_pre"][:5],
        [0.48864085, 0.48765462, 0.08854805, 0.17332318, 0.60413802],
        atol=5e-9,
    )
    np.testing.assert_allclose(
        sc["Y_post"][:5],
        [6.60812636, 6.38461966, 5.88108975, 5.98176705, 6.65810089],
        atol=5e-9,
    )


def test_scest_simplex_weights_and_effect_recovery(base_kwargs):
    sc = sp.scest(**base_kwargs, w_constr="simplex")
    w = np.asarray(sc["weights"])
    # Simplex constraint: non-negative weights summing to one.
    assert w.min() >= -1e-9
    assert w.sum() == pytest.approx(1.0, abs=1e-6)
    assert np.mean(sc["effects"]) == pytest.approx(EFFECT, abs=1.5)
    assert sc["pre_rmspe"] >= 0.0


# --------------------------------------------------------------------------
# Estimator variants all recover the planted ATT
# --------------------------------------------------------------------------
@pytest.mark.parametrize("fn_name", ["qqsynth", "robust_synth", "demeaned_synth"])
def test_synth_variants_recover_effect(base_kwargs, fn_name):
    fn = getattr(sp, fn_name)
    res = fn(**base_kwargs)
    assert res.estimate == pytest.approx(EFFECT, abs=1.5)


def test_robust_and_demeaned_numeric_contracts(base_kwargs):
    robust = sp.robust_synth(**base_kwargs)
    demeaned = sp.demeaned_synth(**base_kwargs)
    np.testing.assert_allclose(
        [robust.estimate, robust.se, robust.ci[0], robust.ci[1]],
        [
            6.151221649507683,
            3.6951747725571136,
            -1.0911878212852457,
            13.393631120300611,
        ],
        atol=1e-12,
    )
    np.testing.assert_allclose(
        robust.model_info["weights"].head(3).to_numpy(),
        np.array([[1.0, 0.53462887], [3.0, 0.50228474], [7.0, 0.38153420]]),
        atol=5e-7,
    )
    np.testing.assert_allclose(
        [demeaned.estimate, demeaned.se, demeaned.ci[0], demeaned.ci[1]],
        [6.035430332614562, 0.8730595645484532, 4.324265029741372, 7.746595635487753],
        atol=1e-12,
    )
    np.testing.assert_allclose(
        demeaned.model_info["weights"].head(3).to_numpy(),
        np.array([[1.0, 0.24598025], [13.0, 0.22607612], [7.0, 0.19569769]]),
        atol=1e-6,
    )


def test_staggered_synth_recovers_effect(panel):
    df = panel.copy()
    df["treat"] = ((df["unit"] == 0) & (df["time"] >= TREAT_T)).astype(int)
    res = sp.staggered_synth(
        data=df, outcome="y", unit="unit", time="time", treatment="treat"
    )
    assert res.estimate == pytest.approx(EFFECT, abs=1.5)


# --------------------------------------------------------------------------
# Robustness / placebo helpers return well-formed frames
# --------------------------------------------------------------------------
def test_donor_sensitivity_frame(base_kwargs):
    out = sp.synth_donor_sensitivity(**base_kwargs, n_samples=30, seed=0)
    assert {"iteration", "donors_used", "att", "pre_rmse"} <= set(out.columns)
    assert len(out) == 30
    # Resampling donors should keep the ATT in the right ballpark on average.
    assert out["att"].median() == pytest.approx(EFFECT, abs=2.0)


def test_rmspe_filter_pvalues_are_probabilities(base_kwargs):
    out = sp.synth_rmspe_filter(**base_kwargs)
    assert {"threshold", "n_placebos", "pvalue"} <= set(out.columns)
    assert ((out["pvalue"] >= 0) & (out["pvalue"] <= 1)).all()
    assert (out["n_placebos"] >= 0).all()
    np.testing.assert_allclose(
        out[["threshold", "n_placebos", "pvalue", "treated_pre_rmspe"]]
        .head(3)
        .to_numpy(),
        np.array(
            [
                [1.0, 1.0, 0.5, 0.17714],
                [2.0, 10.0, 0.09090909, 0.17714],
                [5.0, 14.0, 0.06666667, 0.17714],
            ]
        ),
        atol=5e-8,
    )


# --------------------------------------------------------------------------
# Distributional post-tests on the synth result
# --------------------------------------------------------------------------
def test_discos_test_fires_on_large_effect(base_kwargs):
    res = sp.qqsynth(**base_kwargs, seed=0)
    out = sp.discos_test(res, test="ks")
    assert 0.0 <= out["pvalue"] <= 1.0
    # A 6-unit shift makes the pre/post distributions clearly distinct.
    assert out["reject"] is True
    np.testing.assert_allclose(
        [out["statistic"], out["pvalue"]],
        [1.0, 2.2087606931995054e-59],
        atol=1e-70,
    )


def test_stochastic_dominance_under_positive_effect(base_kwargs):
    res = sp.qqsynth(**base_kwargs, seed=0)
    out = sp.stochastic_dominance(res, order=1)
    # Every post-period gap is positive, so first-order dominance holds.
    assert out["dominates"] is True
    assert out["fraction_positive"] == pytest.approx(1.0, abs=1e-9)
