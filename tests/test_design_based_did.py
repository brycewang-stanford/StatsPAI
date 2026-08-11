"""Behaviour and calibration tests for the design-based DiD family.

``tests/reference_parity/`` pins these estimators against R, but those tests
skip when the fixtures or R are unavailable, and agreement with a reference
implementation is not the same as being *right*. This file stands alone: it
checks the properties the estimators are supposed to have, on data generated
here, with no reference implementation involved.

The two that matter most are at the bottom:

* ``test_confidence_intervals_cover_the_truth`` — the whole claim of a
  design-based estimator is that its interval is valid under random adoption
  timing rather than under parallel trends. Monte-Carlo coverage is the only
  way to check that claim rather than restate it.
* ``test_fisher_pvalue_is_calibrated_under_the_null`` — a randomisation test
  that always returned 0 would pass every parity comparison on a panel with a
  real effect. Under the null it must not.

Covers ``sp.staggered_rollout``, ``sp.staggered_cs``, ``sp.staggered_sa``,
``sp.functional_form_test`` and ``sp.distributional_did``.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp

TAU = 0.8  # homogeneous treatment effect used throughout


def randomised_rollout(
    seed: int,
    n_units: int = 200,
    n_periods: int = 5,
    tau: float = TAU,
    cohorts: tuple = (3, 4, 5),
    include_never_treated: bool = True,
) -> pd.DataFrame:
    """A staggered rollout with adoption dates dealt at random.

    Timing is independent of everything else by construction, so the
    design-based assumption holds exactly and the simple ATT equals ``tau``.
    """
    rng = np.random.default_rng(seed)
    pool = list(cohorts) + ([0] if include_never_treated else [])
    unit_g = rng.permutation(np.tile(pool, int(np.ceil(n_units / len(pool))))[:n_units])
    unit_fe = rng.normal(0.0, 1.0, size=n_units)

    rows = []
    for u in range(n_units):
        g = int(unit_g[u])
        for t in range(1, n_periods + 1):
            treated = g > 0 and t >= g
            y = unit_fe[u] + 0.2 * t + (tau if treated else 0.0)
            y += rng.normal(0.0, 0.5)
            rows.append({"unit": u + 1, "time": t, "first_treat": g, "y": y})
    return pd.DataFrame(rows)


KEYS = dict(y="y", i="unit", t="time", g="first_treat")


# --------------------------------------------------------------------------
# point estimates
# --------------------------------------------------------------------------


@pytest.mark.parametrize("estimand", ["simple", "cohort", "calendar"])
def test_recovers_a_homogeneous_effect(estimand):
    """With a constant effect every estimand targets the same number."""
    df = randomised_rollout(seed=1, n_units=600)
    res = sp.staggered_rollout(df, estimand=estimand, **KEYS)
    assert res.estimate == pytest.approx(TAU, abs=4 * res.se)


def test_efficient_is_at_least_as_precise_as_the_plug_in():
    """That is the entire point of the efficient weights."""
    df = randomised_rollout(seed=2, n_units=400)
    eff = sp.staggered_rollout(df, **KEYS)
    plug = sp.staggered_rollout(df, efficient=False, **KEYS)
    assert eff.model_info["se_neyman"] <= plug.model_info["se_neyman"] + 1e-12
    assert not np.allclose(eff.model_info["beta"], 1.0)


def test_adjusted_se_never_exceeds_the_conservative_one():
    df = randomised_rollout(seed=3)
    res = sp.staggered_rollout(df, **KEYS)
    assert res.model_info["se_adjusted"] <= res.model_info["se_neyman"] + 1e-12


def test_general_control_set_is_weakly_more_efficient():
    df = randomised_rollout(seed=4, n_units=400)
    did = sp.staggered_rollout(df, **KEYS)
    general = sp.staggered_rollout(df, use_did_a0=False, **KEYS)
    assert general.model_info["se_neyman"] <= did.model_info["se_neyman"] + 1e-12
    assert np.asarray(general.model_info["beta"]).size > 1


def test_works_without_never_treated_units():
    """``max(g)`` finite is a different branch of the weight construction."""
    df = randomised_rollout(seed=5, n_units=400, include_never_treated=False)
    res = sp.staggered_rollout(df, **KEYS)
    assert res.estimate == pytest.approx(TAU, abs=4 * res.se)


@pytest.mark.parametrize("never", [0, np.nan, np.inf])
def test_never_treated_coding_is_irrelevant(never):
    """0 / NaN / inf must all mean "never", with identical numbers."""
    df = randomised_rollout(seed=6)
    recoded = df.copy()
    recoded["first_treat"] = recoded["first_treat"].astype(float)
    recoded.loc[recoded["first_treat"] == 0, "first_treat"] = never
    baseline = sp.staggered_rollout(df, **KEYS)
    got = sp.staggered_rollout(recoded, **KEYS)
    assert got.estimate == pytest.approx(baseline.estimate, abs=1e-12)
    assert got.se == pytest.approx(baseline.se, abs=1e-12)


def test_event_study_tracks_a_dynamic_effect():
    """An effect growing with exposure must show up as a rising path."""
    rng = np.random.default_rng(7)
    rows = []
    unit_g = rng.permutation(np.tile([3, 4, 5], 100))
    for u in range(300):
        g = int(unit_g[u])
        fe = rng.normal(0.0, 0.5)
        for t in range(1, 7):
            rel = t - g
            y = fe + 0.1 * t + (0.5 * (rel + 1) if rel >= 0 else 0.0)
            y += rng.normal(0.0, 0.3)
            rows.append({"unit": u + 1, "time": t, "first_treat": g, "y": y})
    df = pd.DataFrame(rows)

    res = sp.staggered_rollout(df, estimand="eventstudy", event_time=[0, 1], **KEYS)
    path = res.detail.sort_values("event_time")["estimate"].to_numpy()
    assert path[1] > path[0]
    assert path[0] == pytest.approx(0.5, abs=0.25)


def test_event_time_minus_one_is_identically_zero():
    """Outcome and control weights coincide there, so the estimate cancels."""
    df = randomised_rollout(seed=8)
    res = sp.staggered_rollout(df, estimand="eventstudy", event_time=-1, **KEYS)
    assert res.estimate == pytest.approx(0.0, abs=1e-10)


def test_cs_wrapper_equals_the_plug_in_when_nothing_is_dropped():
    df = randomised_rollout(seed=9)
    plug = sp.staggered_rollout(df, efficient=False, **KEYS)
    cs = sp.staggered_cs(df, **KEYS)
    assert cs.estimate == pytest.approx(plug.estimate, abs=1e-12)


def test_sa_uses_a_smaller_comparison_group_than_cs():
    df = randomised_rollout(seed=10)
    cs = sp.staggered_cs(df, **KEYS)
    sa = sp.staggered_sa(df, **KEYS)
    assert cs.estimate != pytest.approx(sa.estimate, abs=1e-8)
    assert sa.model_info["use_last_treated_only"] is True


# --------------------------------------------------------------------------
# failure modes
# --------------------------------------------------------------------------


def test_unbalanced_panel_fails_loudly():
    df = randomised_rollout(seed=11).drop(index=[0, 5, 9])
    with pytest.raises(sp.exceptions.DataInsufficient, match="balanced"):
        sp.staggered_rollout(df, **KEYS)


def test_missing_column_fails_loudly():
    df = randomised_rollout(seed=12)
    with pytest.raises(sp.exceptions.MethodIncompatibility, match="not in data"):
        sp.staggered_rollout(df, y="nope", i="unit", t="time", g="first_treat")


@pytest.mark.parametrize(
    "kwargs,match",
    [
        (dict(estimand="dynamic"), "estimand"),
        (dict(se_type="hc1"), "se_type"),
        (dict(use_did_a0=False, efficient=False), "plug-in"),
    ],
)
def test_bad_arguments_fail_loudly(kwargs, match):
    df = randomised_rollout(seed=13)
    with pytest.raises(sp.exceptions.MethodIncompatibility, match=match):
        sp.staggered_rollout(df, **KEYS, **kwargs)


def test_single_cohort_fails_loudly():
    df = randomised_rollout(seed=14, cohorts=(3,), include_never_treated=False)
    with pytest.raises(sp.exceptions.DataInsufficient, match="two treatment cohorts"):
        sp.staggered_rollout(df, **KEYS)


def test_infeasible_event_time_fails_loudly():
    df = randomised_rollout(seed=15)
    with pytest.raises(sp.exceptions.DataInsufficient):
        sp.staggered_rollout(df, estimand="eventstudy", event_time=99, **KEYS)


# --------------------------------------------------------------------------
# calibration — the claims that parity cannot check
# --------------------------------------------------------------------------


def test_confidence_intervals_cover_the_truth():
    """Monte-Carlo coverage of the design-based interval under random timing.

    200 replications of a fresh randomised rollout. The conservative interval
    should cover at least at its nominal rate (it is conservative, so it may
    over-cover); the adjusted one should be close to nominal and never wildly
    below it. At 200 draws the standard error of a coverage estimate near 0.95
    is about 0.015, so the floors below sit roughly four of those under
    nominal — loose enough not to flake, tight enough that a broken variance
    (which would show up as coverage near 0.6, not 0.93) cannot pass.
    """
    covered_neyman = covered_adjusted = 0
    reps = 200
    for seed in range(reps):
        df = randomised_rollout(seed=1000 + seed, n_units=120, n_periods=4)
        res = sp.staggered_rollout(df, se_type="neyman", **KEYS)
        se_n = res.model_info["se_neyman"]
        se_a = res.model_info["se_adjusted"]
        covered_neyman += abs(res.estimate - TAU) <= 1.96 * se_n
        covered_adjusted += abs(res.estimate - TAU) <= 1.96 * se_a

    assert covered_neyman / reps >= 0.90, covered_neyman / reps
    assert covered_adjusted / reps >= 0.88, covered_adjusted / reps
    # The conservative interval cannot cover less often than the adjusted one.
    assert covered_neyman >= covered_adjusted


def test_estimator_is_unbiased_across_replications():
    """Averaging over draws should land on the true effect, not near it."""
    estimates = [
        sp.staggered_rollout(
            randomised_rollout(seed=2000 + s, n_units=120, n_periods=4), **KEYS
        ).estimate
        for s in range(120)
    ]
    mean = float(np.mean(estimates))
    mc_se = float(np.std(estimates, ddof=1) / np.sqrt(len(estimates)))
    assert abs(mean - TAU) <= 3 * mc_se, (mean, mc_se)


def test_fisher_pvalue_is_calibrated_under_the_null():
    """With no effect the randomisation p-value must not pile up at zero.

    A test that always rejected would still match every parity comparison run
    on a panel with a real effect, so this is the check that keeps the
    randomisation path honest. Under the null the p-value is approximately
    uniform, so its mean sits near 0.5; the bound below only asks that it is
    nowhere near degenerate.
    """
    pvalues = []
    for seed in range(25):
        df = randomised_rollout(seed=3000 + seed, n_units=100, n_periods=4, tau=0.0)
        res = sp.staggered_rollout(
            df, fisher=True, n_fisher=200, random_state=seed, **KEYS
        )
        pvalues.append(res.model_info["fisher_pvalue"])

    pvalues = np.asarray(pvalues, dtype=float)
    assert float(pvalues.mean()) > 0.25, pvalues
    assert float((pvalues < 0.05).mean()) < 0.25, pvalues


def test_fisher_pvalue_detects_a_real_effect():
    df = randomised_rollout(seed=17, n_units=300)
    res = sp.staggered_rollout(df, fisher=True, n_fisher=200, random_state=0, **KEYS)
    assert res.model_info["fisher_pvalue"] < 0.01


# --------------------------------------------------------------------------
# functional form / distributional
# --------------------------------------------------------------------------


def _multiplicative_panel(seed: int = 21) -> pd.DataFrame:
    """Parallel trends holds in logs and fails in levels."""
    rng = np.random.default_rng(seed)
    rows = []
    for uid in range(1, 401):
        g = int(rng.choice([3, 0]))
        base = rng.lognormal(1.6 if g > 0 else 0.6, 0.35)
        for t in range(1, 5):
            rows.append(
                {
                    "id": uid,
                    "t": t,
                    "g": g,
                    "y": base * (1.25 ** (t - 1)) * rng.lognormal(0.0, 0.10),
                }
            )
    return pd.DataFrame(rows)


FF_KEYS = dict(y="y", g="g", t="t", i="id")


def test_functional_form_test_does_not_reject_an_additive_design():
    """Parallel trends holding on the scale it is run on must not reject."""
    rng = np.random.default_rng(22)
    rows = []
    for uid in range(1, 401):
        g = int(rng.choice([3, 0]))
        fe = rng.normal(6.0, 1.2)
        for t in range(1, 5):
            y = fe + 0.30 * t + (0.8 if (g > 0 and t >= g) else 0.0)
            rows.append({"id": uid, "t": t, "g": g, "y": y + rng.normal(0, 0.5)})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sp.functional_form_test(pd.DataFrame(rows), **FF_KEYS, n_bins=8)
    assert res.pvalue > 0.05


def test_functional_form_test_rejects_a_multiplicative_design_read_in_levels():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sp.functional_form_test(_multiplicative_panel(), **FF_KEYS, n_bins=8)
    assert res.pvalue < 0.05
    assert len(res.negative_bins) > 0


def test_implied_density_sums_to_about_one():
    """It is a density; the bins partition the support."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sp.functional_form_test(_multiplicative_panel(), **FF_KEYS, n_bins=8)
    assert res.diagnostics["density_sum"] == pytest.approx(1.0, abs=0.05)


def test_distributional_effects_sum_to_zero():
    """Treatment redistributes probability mass; it cannot create it."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sp.distributional_did(_multiplicative_panel(), **FF_KEYS, n_bins=8)
    assert res.diagnostics["effect_sum"] == pytest.approx(0.0, abs=1e-10)


def test_weights_move_the_functional_form_answer():
    panel = _multiplicative_panel()
    rng = np.random.default_rng(23)
    unit_w = {
        uid: w for uid, w in zip(panel["id"].unique(), rng.uniform(0.5, 2.0, 400))
    }
    panel["w"] = panel["id"].map(unit_w)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plain = sp.functional_form_test(panel, **FF_KEYS, n_bins=8)
        weighted = sp.functional_form_test(panel, **FF_KEYS, n_bins=8, weights="w")
    assert not np.allclose(
        plain.table["implied_density"].to_numpy(),
        weighted.table["implied_density"].to_numpy(),
        atol=1e-6,
    )


@pytest.mark.parametrize(
    "kwargs,match",
    [
        (dict(aggregation="nope"), "aggregation"),
        (dict(n_bins=1), "at least 2"),
        (dict(n_bins="sturges"), "n_bins"),
        (dict(n_bins=6, binpoints=[1, 2, 3]), "only one"),
        (dict(alpha=1.5), "alpha"),
        (dict(n_sims=10), "n_sims"),
    ],
)
def test_functional_form_bad_arguments_fail_loudly(kwargs, match):
    with pytest.raises(sp.exceptions.MethodIncompatibility, match=match):
        sp.functional_form_test(_multiplicative_panel(), **FF_KEYS, **kwargs)


def test_functional_form_rejects_non_finite_outcomes():
    panel = _multiplicative_panel()
    panel.loc[0, "y"] = np.nan
    with pytest.raises(sp.exceptions.MethodIncompatibility, match="non-finite"):
        sp.functional_form_test(panel, **FF_KEYS, n_bins=6)


def test_result_objects_expose_the_standard_protocol():
    """Both results must serialise like every other StatsPAI result."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ff = sp.functional_form_test(_multiplicative_panel(), **FF_KEYS, n_bins=6)
        dd = sp.distributional_did(_multiplicative_panel(), **FF_KEYS, n_bins=6)
    for result in (ff, dd):
        assert isinstance(result.to_dict(), dict)
        assert isinstance(result.summary(), str)
        assert isinstance(result.to_markdown(), str)
        assert result.to_latex()
