"""Reference parity: ``sp.did_multiplegt_dyn`` vs ``DIDmultiplegtDYN``.

The de Chaisemartin-D'Haultfoeuille (2024) intertemporal event study shipped
as an explicit MVP: the module docstring carried ``[待核验]`` markers on the
control-group window, the per-horizon weights and the placebo definition,
because none of them had been checked against anything. The authors' own R
package settles all three.

On a staggered absorbing design every effect, every placebo and the
switcher-weighted aggregate agree to 5e-15 -- and so do the switcher counts
behind each horizon, which is the part that shows the two sides are using
the same samples rather than landing on the same number by luck.

Reference generation (R 4.5.2, DIDmultiplegtDYN 2.3.4)::

    did_multiplegt_dyn(df = d, outcome = "y", group = "id", time = "t",
                       treatment = "d", effects = 4, placebo = 2,
                       cluster = "id", graph_off = TRUE)

Fixture: ``_fixtures/multiplegt_dyn_panel.csv`` -- 200 units x 8 periods,
cohorts {3, 5, 7} plus never-treated, effect 1.5.

Index convention: the R package labels from 1, so ``Effect_k`` is horizon
``k - 1`` and ``Placebo_k`` is horizon ``-k``.

Not compared: standard errors. The package reports analytical
influence-function SEs; this estimator has only a cluster bootstrap. That is
a real gap in the implementation, not a tolerance question.

References
----------
de Chaisemartin, C. and D'Haultfoeuille, X. (2024). "Difference-in-Differences
Estimators of Intertemporal Treatment Effects." *Review of Economics and
Statistics*. DOI 10.1162/rest_a_01414. [@dechaisemartin2024difference]
"""

from __future__ import annotations

import pathlib
import warnings

import pandas as pd
import pytest

import statspai as sp

_FIXTURE = pathlib.Path(__file__).parent / "_fixtures" / "multiplegt_dyn_panel.csv"

# DIDmultiplegtDYN: (estimate, switchers) keyed by StatsPAI horizon.
R_EFFECTS = {
    0: (1.4633451798, 149),
    1: (1.3815647774, 149),
    2: (1.3308854161, 109),
    3: (1.4067241960, 109),
}
R_PLACEBOS = {
    -1: (-0.0476249724, 149),
    -2: (0.0663315900, 89),
}
R_AV_TOT_EFF = 1.3997888204
SP_SIMPLE_AVERAGE = 1.3956298923


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    if not _FIXTURE.exists():  # pragma: no cover - fixture ships with the repo
        pytest.skip(f"missing fixture: {_FIXTURE}")
    return pd.read_csv(_FIXTURE)


def _fit(df: pd.DataFrame, aggregation: str = "simple"):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.did_multiplegt_dyn(
            df,
            y="y",
            group="id",
            time="t",
            treatment="d",
            dynamic=3,
            placebo=2,
            cluster="id",
            n_boot=0,
            aggregation=aggregation,
        )


@pytest.fixture(scope="module")
def fit(panel):
    return _fit(panel)


@pytest.mark.parametrize("h", sorted(R_EFFECTS))
def test_effect_matches_r_package(fit, h):
    row = fit.detail.set_index("horizon").loc[h]
    got = float(row["delta_l"])
    assert got == pytest.approx(
        R_EFFECTS[h][0], abs=1e-9
    ), f"h={h}: StatsPAI {got:.10f} vs DIDmultiplegtDYN {R_EFFECTS[h][0]:.10f}"


@pytest.mark.parametrize("h", sorted(R_PLACEBOS))
def test_placebo_matches_r_package(fit, h):
    """⚠️ These changed in 1.21.0. The old placebo was a one-period
    difference sliding backwards, not the mirrored long difference the
    package computes."""
    row = fit.detail.set_index("horizon").loc[h]
    got = float(row["delta_l"])
    assert got == pytest.approx(
        R_PLACEBOS[h][0], abs=1e-9
    ), f"h={h}: StatsPAI {got:.10f} vs DIDmultiplegtDYN {R_PLACEBOS[h][0]:.10f}"


@pytest.mark.parametrize("h", sorted(R_EFFECTS) + sorted(R_PLACEBOS))
def test_switcher_counts_match_r_package(fit, h):
    """The sample behind each horizon, not just the number it produces.

    This is what caught the old placebo definition: it needed one more
    pre-period than the package does, so lag 1 silently dropped a cohort.
    """
    expected = (R_EFFECTS | R_PLACEBOS)[h][1]
    row = fit.detail.set_index("horizon").loc[h]
    assert int(row["n_switchers"]) == expected


def test_switcher_weighted_aggregate_matches_r_package(panel):
    assert _fit(panel, "switchers").estimate == pytest.approx(R_AV_TOT_EFF, abs=1e-9)


def test_default_aggregation_is_unchanged(fit):
    """Guards the default: switching it would move published numbers."""
    assert fit.model_info["aggregation"] == "simple"
    assert fit.estimate == pytest.approx(SP_SIMPLE_AVERAGE, abs=1e-9)


def test_the_two_aggregations_differ_here(panel):
    """Later horizons rest on fewer cohorts (109 switchers vs 149), so
    equal-weighting and switcher-weighting are not the same average."""
    assert abs(_fit(panel, "switchers").estimate - _fit(panel).estimate) > 1e-3


def test_unknown_aggregation_fails_loudly(panel):
    with pytest.raises(ValueError, match="aggregation"):
        _fit(panel, aggregation="cohort")


def test_placebos_are_near_zero_and_effects_are_not(fit):
    """The DGP has no pre-trend and a planted effect of 1.5."""
    detail = fit.detail.set_index("horizon")
    for h in R_PLACEBOS:
        assert abs(float(detail.loc[h, "delta_l"])) < 0.2
    for h in R_EFFECTS:
        assert float(detail.loc[h, "delta_l"]) > 1.0


# --------------------------------------------------------------------------
# Analytic standard errors: available, bounded, and NOT claimed as parity.
# --------------------------------------------------------------------------

# DIDmultiplegtDYN's reported standard errors, by StatsPAI horizon.
R_SE = {
    0: 0.09740853,
    1: 0.09518580,
    2: 0.12724023,
    3: 0.12652024,
    -1: 0.1016065,
    -2: 0.1284979,
}
# The analytic variance is the standard two-sample influence function, not
# dCDH's own derivation, so it is held to a bounded gap rather than parity.
SE_GAP = 0.015


@pytest.fixture(scope="module")
def analytic_fit(panel):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.did_multiplegt_dyn(
            panel,
            y="y",
            group="id",
            time="t",
            treatment="d",
            dynamic=3,
            placebo=2,
            n_boot=0,
            se_method="analytic",
        )


@pytest.mark.parametrize("h", sorted(R_SE))
def test_analytic_se_is_close_to_the_reference(analytic_fit, h):
    """Bounded, measured, and explicitly not parity.

    The analytic form agrees with this module's own cluster bootstrap; the
    ~1% gap against DIDmultiplegtDYN is dCDH's variance derivation, which
    is not reproduced. Pinned as a bound so a regression that widens it
    still fails.
    """
    es = analytic_fit.model_info["event_study"].set_index("relative_time")
    got = float(es.loc[h, "se"])
    ref = R_SE[h]
    assert 0 < got < ref, (got, ref)
    assert abs(got / ref - 1.0) < SE_GAP, (h, got, ref)


def test_analytic_se_agrees_with_the_bootstrap(panel):
    """What makes the analytic form trustworthy despite the reference gap."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # 400 draws is enough to bound the comparison without making the
        # test slow; the bootstrap's own noise is the loose end here.
        boot = sp.did_multiplegt_dyn(
            panel,
            y="y",
            group="id",
            time="t",
            treatment="d",
            dynamic=3,
            placebo=2,
            cluster="id",
            n_boot=400,
            seed=0,
        )
        ana = sp.did_multiplegt_dyn(
            panel,
            y="y",
            group="id",
            time="t",
            treatment="d",
            dynamic=3,
            placebo=2,
            n_boot=0,
            se_method="analytic",
        )
    b = boot.model_info["event_study"].set_index("relative_time")["se"]
    a = ana.model_info["event_study"].set_index("relative_time")["se"]
    for h in sorted(R_SE):
        assert abs(float(a.loc[h]) / float(b.loc[h]) - 1.0) < 0.10, h


def test_analytic_path_estimates_are_unchanged(analytic_fit, fit):
    """Only the variance changes; every point estimate must be identical."""
    a = analytic_fit.model_info["event_study"].set_index("relative_time")["att"]
    b = fit.model_info["event_study"].set_index("relative_time")["att"]
    for h in sorted(R_SE):
        assert float(a.loc[h]) == pytest.approx(float(b.loc[h]), abs=1e-12)


def test_analytic_path_reports_itself(analytic_fit):
    assert analytic_fit.model_info["se_method"] == "analytic"
    # The joint tests come from the bootstrap, so with n_boot=0 they are
    # absent rather than silently fabricated from the analytic variance.
    assert analytic_fit.model_info["joint_placebo_test"] is None


def test_unknown_se_method_fails_loudly(panel):
    with pytest.raises(ValueError, match="se_method"):
        sp.did_multiplegt_dyn(
            panel,
            y="y",
            group="id",
            time="t",
            treatment="d",
            dynamic=1,
            n_boot=0,
            se_method="jackknife",
        )


# --------------------------------------------------------------------------
# Switch-off: the design this estimator exists for.
# --------------------------------------------------------------------------

_SWITCHING = (
    _FIX / "multiplegt_dyn_switching_panel.csv"
    if (_FIX := pathlib.Path(__file__).parent / "_fixtures")
    else None
)

# DIDmultiplegtDYN on the switching panel, effects = 2, placebo = 1.
R_OFF = {
    0: (1.2691144380, 126),
    1: (0.7585626001, 119),
    -1: (-0.0883148725, 84),
}


@pytest.fixture(scope="module")
def switching_panel():
    if not _SWITCHING.exists():  # pragma: no cover - ships with the repo
        pytest.skip(f"missing fixture: {_SWITCHING}")
    return pd.read_csv(_SWITCHING)


@pytest.fixture(scope="module")
def switching_fit(switching_panel):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.did_multiplegt_dyn(
            switching_panel,
            y="y",
            group="id",
            time="t",
            treatment="d",
            dynamic=1,
            placebo=1,
            cluster="id",
            n_boot=0,
        )


@pytest.mark.parametrize("h", sorted(R_OFF))
def test_switch_off_estimates_match_reference(switching_fit, h):
    """⚠️ New in 1.21.0. Switch-off events used to be dropped silently,
    which changed the estimand on any non-absorbing panel."""
    es = switching_fit.model_info["event_study"].set_index("relative_time")
    want, _ = R_OFF[h]
    got = float(es.loc[h, "att"])
    assert got == pytest.approx(
        want, abs=1e-9
    ), f"h={h}: StatsPAI {got:.10f} vs DIDmultiplegtDYN {want:.10f}"


@pytest.mark.parametrize("h", sorted(R_OFF))
def test_switch_off_switcher_counts_match_reference(switching_fit, h):
    """The counts are the tell. A run that dropped the switch-off events
    would report roughly three quarters of these and still look plausible."""
    es = switching_fit.model_info["event_study"].set_index("relative_time")
    assert int(es.loc[h, "n_switchers"]) == R_OFF[h][1]


def test_switch_off_units_actually_contribute(switching_panel, switching_fit):
    """Guards against the fixture quietly becoming absorbing."""
    d = switching_panel.sort_values(["id", "t"])
    fell = d.groupby("id")["d"].apply(lambda s: (s.diff() == -1).any())
    assert fell.sum() > 20, int(fell.sum())
    # And the estimate uses more switchers than the switch-on events alone.
    rose_first = d.groupby("id")["d"].apply(
        lambda s: bool(len(s.diff().dropna().to_numpy().nonzero()[0]))
        and s.diff().dropna().to_numpy()[s.diff().dropna().to_numpy() != 0][0] > 0
    )
    es = switching_fit.model_info["event_study"].set_index("relative_time")
    assert int(es.loc[0, "n_switchers"]) > int(rose_first.sum())
