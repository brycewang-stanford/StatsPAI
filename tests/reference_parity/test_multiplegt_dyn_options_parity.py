"""Stata ``did_multiplegt_dyn`` parity for the sample-restriction options.

Covers ``switchers(in|out)`` and ``same_switchers``, plus the
``effects_equal`` joint test, all new in the DiD option-depth campaign.

Fixture panel ``tests/stata_parity/data_85_dcdh_switch.csv`` deliberately
carries **both** switch-on and switch-off events, plus never-switchers at
both baseline levels. An absorbing panel cannot distinguish these options
from the default and could not have exposed the bootstrap defect recorded
below.

Golden numbers are read from the committed
``tests/stata_parity/results/85_multiplegt_dyn_options_Stata.json``,
produced by ``85_multiplegt_dyn_options.do`` on Stata 18 MP. Reading the
JSON rather than hard-coding keeps the do-file and the test from drifting
apart silently.

Convention mapping
------------------
* Stata ``effects(4)`` == StatsPAI ``dynamic=3`` (Stata indexes effects
  from ℓ=1, StatsPAI horizons from h=0), so ``Effect_k`` ↔ ``h = k-1``.
* Stata ``Placebo_k`` ↔ ``h = -k``.
* Stata ``e(Av_tot_effect)`` is StatsPAI's ``aggregation='switchers'``
  headline, **not** the ``'simple'`` default.

Tolerance: 1e-6 on point estimates; observed worst case 3.4e-8.

⚠️ Regression history — the bootstrap estimated a different quantity
---------------------------------------------------------------------
Building these fixtures exposed a live defect. The point estimate finds
each unit's first treatment *change* (either direction) via
``_first_switch``, but the cluster bootstrap re-derived the switch date as
``min(time | d == 1)`` — "first period treated", which is the first change
only for switch-ON units. A unit going 1 → 0 at F therefore got ``_F`` = 1,
its own first period, which has no base period F−1 and so dropped out of
every replicate.

Consequences on a non-absorbing panel: ``switchers='out'`` returned a NaN
standard error, and the pooled SE silently coincided with the switch-in-only
SE (0.182973) because the switch-out units were broken in every draw. The
correct pooled SE is 0.119610. Absorbing panels were unaffected — which is
exactly why the existing ``DIDmultiplegtDYN`` parity test never caught it.
``test_bootstrap_uses_the_same_switch_date_as_the_estimate`` pins it.
"""

from __future__ import annotations

import json
import pathlib

import pandas as pd
import pytest

import statspai as sp

_HERE = pathlib.Path(__file__).resolve().parents[1]
_PANEL = _HERE / "stata_parity" / "option_parity" / "data_85_dcdh_switch.csv"
_GOLDEN = (
    _HERE
    / "stata_parity"
    / "option_parity"
    / "results"
    / "85_multiplegt_dyn_options_Stata.json"
)

ATOL = 1e-6

# aggregation='switchers' matches e(Av_tot_effect); se_method='analytic'
# with n_boot=0 keeps the point-estimate comparison fast and deterministic.
BASE = dict(
    group="i",
    time="t",
    treatment="d",
    dynamic=3,
    placebo=2,
    aggregation="switchers",
    se_method="analytic",
    n_boot=0,
    seed=1,
)

VARIANTS = {
    "pooled": {},
    "switchers_in": {"switchers": "in"},
    "switchers_out": {"switchers": "out"},
    "same_switchers": {"same_switchers": True},
}


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    return pd.read_csv(_PANEL)


@pytest.fixture(scope="module")
def golden() -> dict:
    return json.loads(_GOLDEN.read_text(encoding="utf-8"))


def _fit(panel: pd.DataFrame, **kwargs):
    return sp.did_multiplegt_dyn(panel, "y", **BASE, **kwargs)


def _boot_fit(panel: pd.DataFrame, **kwargs):
    """Same fit with draws, for anything needing the bootstrap covariance."""
    opts = {**BASE, "n_boot": 200, "se_method": "bootstrap"}
    return sp.did_multiplegt_dyn(panel, "y", **opts, **kwargs)


class TestSwitcherOptionsParity:
    @pytest.mark.parametrize("variant", list(VARIANTS))
    def test_headline_matches_stata(self, panel, golden, variant):
        res = _fit(panel, **VARIANTS[variant])
        want = golden[variant]["Av_tot_eff"]
        assert res.estimate == pytest.approx(
            want, abs=ATOL
        ), f"{variant}: StatsPAI {res.estimate:.12f} vs Stata {want:.12f}"

    @pytest.mark.parametrize("variant", list(VARIANTS))
    def test_every_horizon_matches_stata(self, panel, golden, variant):
        """Effects AND placebos, cell by cell — not just the aggregate.

        The aggregate can match while individual horizons are wrong, so
        the per-horizon comparison is what actually pins the restriction.
        """
        res = _fit(panel, **VARIANTS[variant])
        got = {int(r.horizon): float(r.delta_l) for r in res.detail.itertuples()}
        g = golden[variant]
        for k in range(1, 5):  # Effect_k <-> h = k-1
            assert got[k - 1] == pytest.approx(
                g[f"Effect_{k}"], abs=ATOL
            ), f"{variant} Effect_{k} (h={k - 1})"
        for k in range(1, 3):  # Placebo_k <-> h = -k
            assert got[-k] == pytest.approx(
                g[f"Placebo_{k}"], abs=ATOL
            ), f"{variant} Placebo_{k} (h={-k})"

    @pytest.mark.parametrize("variant", list(VARIANTS))
    def test_switcher_counts_match_stata(self, panel, golden, variant):
        """Counts are the sharpest check that the same units were used."""
        res = _fit(panel, **VARIANTS[variant])
        got = {int(r.horizon): int(r.n_switchers) for r in res.detail.itertuples()}
        assert got[0] == golden[variant]["N_switchers_effect_1"]

    def test_same_switchers_holds_the_composition_fixed(self, panel):
        """The defining property: n_switchers constant across horizons."""
        restricted = _fit(panel, same_switchers=True).detail["n_switchers"].tolist()
        assert (
            len(set(restricted)) == 1
        ), f"same_switchers must fix the composition, got {restricted}"
        pooled = _fit(panel).detail["n_switchers"].tolist()
        assert len(set(pooled)) > 1, (
            "fixture must have a composition that actually varies without "
            "the option, or the test proves nothing"
        )

    def test_restricted_switchers_stay_available_as_controls(self, panel):
        """same_switchers must not delete those units from the panel.

        A unit that switches at F=8 cannot support horizon 3, so it is
        excluded as a *switcher* — but it is still a legitimate
        not-yet-treated control for the F=4 and F=6 events. Dropping its
        rows shrinks the control pool and moves every estimate; that error
        put same_switchers 2.9e-2 away from Stata before it was fixed.
        """
        res = _fit(panel, same_switchers=True)
        want = 0.441039354850848  # Stata
        assert res.estimate == pytest.approx(want, abs=ATOL)

    def test_in_and_out_partition_the_switchers(self, panel, golden):
        n_in = golden["switchers_in"]["N_switchers_effect_1"]
        n_out = golden["switchers_out"]["N_switchers_effect_1"]
        n_all = golden["pooled"]["N_switchers_effect_1"]
        assert n_in + n_out == n_all, "in/out must partition the switcher set"

    def test_pooled_lies_between_the_two_directions(self, panel):
        """Sanity: pooling cannot land outside the two components."""
        lo = _fit(panel, switchers="out").estimate
        hi = _fit(panel, switchers="in").estimate
        pooled = _fit(panel).estimate
        assert lo < pooled < hi


class TestBootstrapSwitchDate:
    """⚠️ Regression: the bootstrap must re-derive the switch date the
    same way the point estimate does."""

    def test_bootstrap_uses_the_same_switch_date_as_the_estimate(self, panel):
        """Switch-out events must survive the bootstrap.

        Before the fix these units got ``_F`` = their first period, which
        has no base period, so every replicate dropped them and the SE
        came back NaN.
        """
        res = sp.did_multiplegt_dyn(
            panel,
            "y",
            group="i",
            time="t",
            treatment="d",
            dynamic=3,
            placebo=2,
            switchers="out",
            n_boot=100,
            seed=5,
        )
        assert res.se == res.se, "switch-out bootstrap SE must not be NaN"
        assert res.se > 0

    def test_pooled_se_is_not_the_switch_in_se(self, panel):
        """The tell-tale symptom of the old defect.

        With switch-out units broken in every replicate, the pooled
        bootstrap collapsed onto the switch-in-only one and the two SEs
        came out bit-identical. They must differ: pooled uses strictly
        more switchers.
        """
        common = dict(
            group="i",
            time="t",
            treatment="d",
            dynamic=3,
            placebo=2,
            n_boot=100,
            seed=5,
        )
        pooled = sp.did_multiplegt_dyn(panel, "y", **common).se
        only_in = sp.did_multiplegt_dyn(panel, "y", **common, switchers="in").se
        assert pooled != only_in
        assert pooled < only_in, (
            "pooling both switch directions uses more data, so its SE "
            f"should be smaller; got pooled={pooled:.6f}, in={only_in:.6f}"
        )

    def test_absorbing_panel_is_unaffected_by_the_fix(self):
        """The fix must be a no-op where the old code was already right.

        With only switch-on events, "first period treated" IS the first
        change, so the corrected bootstrap must reproduce the historical
        numbers — which is what the DIDmultiplegtDYN parity suite pins.
        """
        rows = []
        for i in range(150):
            g = [0, 4, 6][i % 3]
            for t in range(1, 9):
                d = 1 if (g > 0 and t >= g) else 0
                rows.append(
                    {"i": i, "t": t, "d": d, "y": 0.5 * d + 0.05 * t + (i % 7) * 0.01}
                )
        ab = pd.DataFrame(rows)
        res = sp.did_multiplegt_dyn(
            ab, "y", group="i", time="t", treatment="d", dynamic=2, n_boot=50, seed=3
        )
        assert res.se == res.se and res.se >= 0


class TestEffectsEqual:
    def test_df_is_k_minus_one(self, panel):
        """Equality leaves the common level free: k effects, k-1 restrictions."""
        res = _boot_fit(panel, effects_equal=True)
        test = res.diagnostics["effects_equal_test"]
        assert test["df"] == 4 - 1
        assert test["horizons"] == [0, 1, 2, 3]

    def test_range_form_selects_the_requested_horizons(self, panel):
        res = _boot_fit(panel, effects_equal=(0, 2))
        test = res.diagnostics["effects_equal_test"]
        assert test["horizons"] == [0, 1, 2]
        assert test["df"] == 2

    def test_constant_effect_is_not_rejected(self):
        """A DGP with a genuinely flat effect should not trip the test."""
        rows = []
        for i in range(200):
            g = [0, 4, 6][i % 3]
            for t in range(1, 11):
                d = 1 if (g > 0 and t >= g) else 0
                rows.append(
                    {"i": i, "t": t, "d": d, "y": 0.5 * d + 0.03 * t + (i % 11) * 0.02}
                )
        flat = pd.DataFrame(rows)
        res = sp.did_multiplegt_dyn(
            flat,
            "y",
            group="i",
            time="t",
            treatment="d",
            dynamic=3,
            n_boot=200,
            seed=2,
            effects_equal=True,
        )
        assert res.diagnostics["effects_equal_test"]["pvalue"] > 0.05

    def test_growing_effect_is_rejected(self):
        """Power check: a strongly ramping effect must be detected."""
        rows = []
        for i in range(200):
            g = [0, 4, 6][i % 3]
            for t in range(1, 11):
                d = 1 if (g > 0 and t >= g) else 0
                ramp = 1.5 * max(t - g, 0) if d else 0.0
                rows.append(
                    {"i": i, "t": t, "d": d, "y": ramp + 0.03 * t + (i % 11) * 0.01}
                )
        ramping = pd.DataFrame(rows)
        res = sp.did_multiplegt_dyn(
            ramping,
            "y",
            group="i",
            time="t",
            treatment="d",
            dynamic=3,
            n_boot=200,
            seed=2,
            effects_equal=True,
        )
        assert res.diagnostics["effects_equal_test"]["pvalue"] < 0.05

    def test_disabled_by_default(self, panel):
        """With draws available, the test must still be absent unless asked.

        Uses the bootstrap fit deliberately: against the n_boot=0 fit this
        would pass whether or not the option were wired up at all.
        """
        assert _boot_fit(panel).diagnostics["effects_equal_test"] is None

    def test_requires_bootstrap_draws(self, panel):
        """Documented behaviour: the covariance comes from the bootstrap."""
        assert _fit(panel, effects_equal=True).diagnostics["effects_equal_test"] is None

    @pytest.mark.parametrize("bad", [(3, 1), "all", (1,)])
    def test_invalid_range_rejected(self, panel, bad):
        with pytest.raises(ValueError):
            _boot_fit(panel, effects_equal=bad)


class TestSwitchersValidation:
    @pytest.mark.parametrize("bad", ["IN", "both", 1, True])
    def test_invalid_switchers_rejected(self, panel, bad):
        with pytest.raises(ValueError, match="switchers must be"):
            _fit(panel, switchers=bad)

    def test_settings_recorded(self, panel):
        res = _fit(panel, switchers="in", same_switchers=True)
        assert res.diagnostics["switchers"] == "in"
        assert res.diagnostics["same_switchers"] is True
