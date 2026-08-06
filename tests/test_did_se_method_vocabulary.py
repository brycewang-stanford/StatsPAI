"""One ``se_method=`` vocabulary across the staggered DiD estimators.

StatsPAI grew four spellings for the same question, because each
estimator copied whichever reference package it was aligned against:

===================== =========================================
estimator             native spelling
===================== =========================================
callaway_santanna     ``bstrap=`` (R ``did``)
did_imputation        ``vce=`` (Stata)
did_multiplegt_dyn    ``se_method=``
sun_abraham           none — analytic only
===================== =========================================

``se_method=`` is a synonym layer over those, **not** a replacement:
every native spelling still works and every default is untouched. That
"additive, no default moves" property is what most of this module pins,
because a silent default change here would move published standard
errors.

Note this is a different axis from ``statspai.core._vcov_spec``, which
normalizes *which sandwich* (HC0/HC1/CR1/CR2/CR3) a regression uses. Here
the question is which *procedure* produces the variance at all.

``'auto'`` switches to a bootstrap at or below ``FEW_CLUSTERS`` = 30, the
top of the range over which Cameron, Gelbach & Miller (2008),
doi:10.1162/rest.90.3.414, document over-rejection by cluster-robust
asymptotics.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.did._core import FEW_CLUSTERS, normalize_se_method

_MPDTA = (
    pathlib.Path(__file__).resolve().parent
    / "orig_parity"
    / "data"
    / "02_mpdta_original.csv"
)


@pytest.fixture(scope="module")
def mpdta() -> pd.DataFrame:
    return pd.read_csv(_MPDTA)


def _small_panel(n_units: int = 18, seed: int = 3) -> pd.DataFrame:
    """A panel with fewer than FEW_CLUSTERS clusters."""
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_units):
        g = [0, 4, 6][u % 3]
        for t in range(1, 9):
            d = 1 if (g > 0 and t >= g) else 0
            rows.append(
                {
                    "unit": u,
                    "time": t,
                    "gv": g,
                    "d": d,
                    "y": 0.5 * d + 0.05 * t + rng.normal(0, 0.3),
                }
            )
    return pd.DataFrame(rows)


class TestNormalizer:
    @pytest.mark.parametrize(
        "spelling,expected",
        [
            ("analytic", "analytic"),
            ("ANALYTIC", "analytic"),
            (" analytic ", "analytic"),
            ("asymptotic", "analytic"),
            ("influence", "analytic"),
            ("if", "analytic"),
            ("bootstrap", "bootstrap"),
            ("cluster", "bootstrap"),
            ("pairs", "bootstrap"),
            ("multiplier", "multiplier"),
            ("wild", "multiplier"),
            ("wboot", "multiplier"),
            ("wild-bootstrap", "multiplier"),
            ("wild bootstrap", "multiplier"),
        ],
    )
    def test_aliases_resolve(self, spelling, expected):
        got = normalize_se_method(
            spelling,
            supported=("analytic", "bootstrap", "multiplier"),
            function="t",
        )
        assert got == expected

    def test_unknown_spelling_rejected(self):
        with pytest.raises(Exception, match="unknown se_method"):
            normalize_se_method("hc1", supported=("analytic",), function="t")

    def test_unsupported_procedure_rejected_not_downgraded(self):
        """Silently falling back would hand back the narrower interval."""
        with pytest.raises(Exception, match="does not implement"):
            normalize_se_method(
                "wboot", supported=("analytic", "bootstrap"), function="t"
            )

    def test_non_string_rejected(self):
        with pytest.raises(Exception, match="must be a string"):
            normalize_se_method(True, supported=("analytic",), function="t")

    @pytest.mark.parametrize(
        "n_clusters,expected",
        [
            (5, "multiplier"),
            (FEW_CLUSTERS, "multiplier"),
            (FEW_CLUSTERS + 1, "analytic"),
            (1000, "analytic"),
            (None, "analytic"),
        ],
    )
    def test_auto_switches_at_the_documented_threshold(self, n_clusters, expected):
        got = normalize_se_method(
            "auto",
            supported=("analytic", "multiplier"),
            function="t",
            n_clusters=n_clusters,
        )
        assert got == expected

    def test_auto_with_unknown_cluster_count_does_not_guess(self):
        """None means "I don't know", which must resolve to analytic."""
        assert (
            normalize_se_method(
                "auto", supported=("analytic", "bootstrap"), function="t"
            )
            == "analytic"
        )

    def test_auto_prefers_multiplier_over_pairs_bootstrap(self):
        got = normalize_se_method(
            "auto",
            supported=("analytic", "bootstrap", "multiplier"),
            function="t",
            n_clusters=10,
        )
        assert got == "multiplier"


class TestCallawaySantanna:
    def test_default_unchanged(self, mpdta):
        res = sp.callaway_santanna(
            mpdta, y="lemp", g="first_treat", t="year", i="countyreal"
        )
        assert res.diagnostics["se_method"] == "analytic"

    def test_wboot_is_bit_identical_to_bstrap(self, mpdta):
        """The synonym must not be a re-implementation."""
        keys = dict(y="lemp", g="first_treat", t="year", i="countyreal")
        native = sp.callaway_santanna(
            mpdta, **keys, bstrap=True, biters=200, random_state=1
        )
        synonym = sp.callaway_santanna(
            mpdta, **keys, se_method="wboot", biters=200, random_state=1
        )
        assert synonym.se == native.se
        assert synonym.estimate == native.estimate

    def test_conflicting_spellings_rejected(self, mpdta):
        with pytest.raises(Exception, match="not both"):
            sp.callaway_santanna(
                mpdta,
                y="lemp",
                g="first_treat",
                t="year",
                i="countyreal",
                bstrap=True,
                se_method="analytic",
            )

    def test_auto_picks_analytic_on_many_clusters(self, mpdta):
        res = sp.callaway_santanna(
            mpdta,
            y="lemp",
            g="first_treat",
            t="year",
            i="countyreal",
            se_method="auto",
        )
        assert res.diagnostics["se_method"] == "analytic"

    def test_auto_picks_multiplier_on_few_clusters(self):
        panel = _small_panel()
        res = sp.callaway_santanna(
            panel,
            y="y",
            g="gv",
            t="time",
            i="unit",
            se_method="auto",
            biters=100,
            random_state=1,
        )
        assert res.diagnostics["se_method"] == "multiplier"

    def test_resolved_choice_is_recorded(self, mpdta):
        res = sp.callaway_santanna(
            mpdta,
            y="lemp",
            g="first_treat",
            t="year",
            i="countyreal",
            se_method="wboot",
            biters=100,
            random_state=1,
        )
        assert res.diagnostics["se_method"] == "multiplier"


class TestDidImputation:
    def test_default_unchanged(self, mpdta):
        keys = dict(
            y="lemp", group="countyreal", time="year", first_treat="first_treat"
        )
        assert sp.did_imputation(mpdta, **keys).se == pytest.approx(
            sp.did_imputation(mpdta, **keys, se_method="analytic").se, rel=1e-12
        )

    def test_conflicting_spellings_rejected(self, mpdta):
        with pytest.raises(ValueError, match="not both"):
            sp.did_imputation(
                mpdta,
                y="lemp",
                group="countyreal",
                time="year",
                first_treat="first_treat",
                vce="bootstrap",
                se_method="analytic",
            )

    def test_auto_moves_to_bootstrap_on_few_clusters(self):
        """BJS's analytic SE is anti-conservative, so auto must escalate."""
        panel = _small_panel()
        keys = dict(y="y", group="unit", time="time", first_treat="gv")
        analytic = sp.did_imputation(panel, **keys).se
        auto = sp.did_imputation(panel, **keys, se_method="auto", n_boot=50).se
        assert auto != analytic
        assert auto > analytic, (
            "the bootstrap should be wider than the known-anti-conservative "
            f"analytic SE; got auto={auto:.6f} vs analytic={analytic:.6f}"
        )


class TestDidMultiplegtDyn:
    def test_native_spelling_still_works(self):
        panel = _small_panel()
        res = sp.did_multiplegt_dyn(
            panel,
            "y",
            group="unit",
            time="time",
            treatment="d",
            dynamic=2,
            se_method="analytic",
        )
        assert res.diagnostics["se_method"] == "analytic"

    def test_auto_picks_bootstrap_on_few_clusters(self):
        panel = _small_panel()
        res = sp.did_multiplegt_dyn(
            panel,
            "y",
            group="unit",
            time="time",
            treatment="d",
            dynamic=2,
            n_boot=50,
            seed=1,
            se_method="auto",
        )
        assert res.diagnostics["se_method"] == "bootstrap"

    def test_multiplier_rejected_because_unimplemented(self):
        panel = _small_panel()
        with pytest.raises(Exception, match="does not implement"):
            sp.did_multiplegt_dyn(
                panel,
                "y",
                group="unit",
                time="time",
                treatment="d",
                dynamic=2,
                se_method="wboot",
            )
