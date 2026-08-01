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
