"""Unit tests for CS multiplier-bootstrap inference and influence export.

Covers the batch-D additions:
- ``callaway_santanna(bstrap=, biters=, cband=, clustervars=,
  boot_weight_type=, random_state=)``
- ``did._core.multiplier_bootstrap`` (weight types, clustering)
- ``sp.influence_functions`` / ``sp.aggte_from_influence``

Numerical parity vs R ``did`` lives in
``tests/reference_parity/test_cs_inference_parity.py``; here we lock in
API behavior, error paths, and internal consistency.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.did._core import multiplier_bootstrap
from statspai.exceptions import MethodIncompatibility


def _staggered_panel(n_units=120, n_periods=8, n_clusters=12, rho=0.0, seed=42):
    """Three-cohort staggered panel; optional within-cluster correlation."""
    rng = np.random.default_rng(seed)
    cluster_effects = rng.normal(size=n_clusters) * rho
    rows = []
    for unit in range(n_units):
        g_val = [4, 6, 0][unit % 3]
        cl = rng.integers(0, n_clusters)  # random, so clusters mix cohorts
        for period in range(1, n_periods + 1):
            te = max(0, period - g_val + 1) if g_val > 0 else 0
            rows.append(
                {
                    "i": unit,
                    "t": period,
                    "y": te + cluster_effects[cl] + rng.normal(),
                    "g": g_val,
                    "cl": cl,
                }
            )
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def panel():
    return _staggered_panel()


@pytest.fixture(scope="module")
def cs_analytic(panel):
    return sp.callaway_santanna(panel, y="y", g="g", t="t", i="i")


@pytest.fixture(scope="module")
def cs_boot(panel):
    return sp.callaway_santanna(
        panel,
        y="y",
        g="g",
        t="t",
        i="i",
        bstrap=True,
        cband=True,
        biters=1999,
        random_state=7,
    )


# ----------------------------------------------------------------------
# bstrap / cband on callaway_santanna
# ----------------------------------------------------------------------


class TestBstrap:
    def test_point_estimates_unchanged(self, cs_analytic, cs_boot):
        # bstrap only touches inference, never the point estimates.
        np.testing.assert_allclose(
            cs_boot.detail["att"].values, cs_analytic.detail["att"].values
        )
        assert np.isclose(cs_boot.estimate, cs_analytic.estimate)

    def test_bootstrap_se_near_analytic(self, cs_analytic, cs_boot):
        # On a clean iid DGP the multiplier bootstrap must reproduce the
        # analytic (delta-method) SEs up to Monte Carlo noise.
        ratio = cs_boot.detail["se"].values / cs_analytic.detail["se"].values
        assert np.all(ratio > 0.75) and np.all(ratio < 1.30)

    def test_cband_columns_and_width(self, cs_boot):
        d = cs_boot.detail
        assert {"cband_lower", "cband_upper"} <= set(d.columns)
        crit = cs_boot.model_info["crit_val_uniform"]
        assert crit >= 1.959  # never narrower than pointwise normal
        # Uniform band strictly contains the pointwise CI.
        finite = np.isfinite(d["se"].values)
        assert np.all(d.loc[finite, "cband_lower"] <= d.loc[finite, "ci_lower"])
        assert np.all(d.loc[finite, "cband_upper"] >= d.loc[finite, "ci_upper"])

    def test_reproducible_with_seed(self, panel):
        kw = dict(
            data=panel,
            y="y",
            g="g",
            t="t",
            i="i",
            bstrap=True,
            biters=499,
            random_state=11,
        )
        r1 = sp.callaway_santanna(**kw)
        r2 = sp.callaway_santanna(**kw)
        np.testing.assert_array_equal(r1.detail["se"].values, r2.detail["se"].values)

    def test_event_study_uses_bootstrap_ses(self, cs_analytic, cs_boot):
        es_a = cs_analytic.model_info["event_study"]
        es_b = cs_boot.model_info["event_study"]
        np.testing.assert_allclose(es_a["att"].values, es_b["att"].values)
        # SEs shift under the bootstrap (same estimand, different variance
        # estimator) but must stay in the same ballpark.
        ratio = es_b["se"].values / es_a["se"].values
        assert np.all(ratio > 0.75) and np.all(ratio < 1.30)
        assert not np.allclose(es_a["se"].values, es_b["se"].values)

    def test_mammen_option_runs(self, panel):
        r = sp.callaway_santanna(
            panel,
            y="y",
            g="g",
            t="t",
            i="i",
            bstrap=True,
            boot_weight_type="mammen",
            biters=499,
            random_state=3,
        )
        assert np.isfinite(r.se) and r.se > 0

    def test_cband_without_bstrap_raises(self, panel):
        with pytest.raises(MethodIncompatibility, match="cband"):
            sp.callaway_santanna(panel, y="y", g="g", t="t", i="i", cband=True)

    def test_bad_weight_type_raises(self, panel):
        with pytest.raises(MethodIncompatibility, match="boot_weight_type"):
            sp.callaway_santanna(
                panel,
                y="y",
                g="g",
                t="t",
                i="i",
                bstrap=True,
                boot_weight_type="webb",
            )

    def test_rcs_bstrap_now_supported(self, panel):
        """bstrap used to raise under panel=False; the RCS influence
        functions now feed the multiplier bootstrap."""
        r = sp.callaway_santanna(
            panel,
            y="y",
            g="g",
            t="t",
            i="i",
            panel=False,
            estimator="reg",
            bstrap=True,
            biters=200,
            random_state=0,
        )
        assert r.se > 0

    def test_rcs_supports_the_clustered_bootstrap(self, panel):
        """Repeated cross-sections used to refuse clustervars; now they run.

        Clustering beyond the unit still requires bstrap=True: the
        analytic per-cell SEs cannot express within-cluster dependence, so
        reporting them under clustervars would understate uncertainty.
        """
        with pytest.raises(MethodIncompatibility, match="requires bstrap=True"):
            sp.callaway_santanna(
                panel,
                y="y",
                g="g",
                t="t",
                i="i",
                panel=False,
                estimator="reg",
                clustervars=["i", "cl"],
            )
        r = sp.callaway_santanna(
            panel,
            y="y",
            g="g",
            t="t",
            i="i",
            panel=False,
            estimator="reg",
            clustervars=["i", "cl"],
            bstrap=True,
            biters=299,
            random_state=3,
        )
        assert r.model_info["se_method"] == "multiplier"
        assert np.all(np.isfinite(r.detail["se"].to_numpy()))


# ----------------------------------------------------------------------
# clustervars
# ----------------------------------------------------------------------


class TestClustervars:
    def test_cluster_inflates_se_under_correlation(self):
        # Strong within-cluster correlation → clustered SE must exceed
        # the unit-level bootstrap SE.
        df = _staggered_panel(rho=2.0, seed=5)
        kw = dict(
            data=df,
            y="y",
            g="g",
            t="t",
            i="i",
            bstrap=True,
            biters=1999,
            random_state=9,
        )
        r_unit = sp.callaway_santanna(**kw)
        r_cl = sp.callaway_santanna(**kw, clustervars=["i", "cl"])
        assert r_cl.se > r_unit.se * 1.2

    def test_unit_id_only_equals_no_clustering(self, panel):
        kw = dict(
            data=panel,
            y="y",
            g="g",
            t="t",
            i="i",
            bstrap=True,
            biters=499,
            random_state=13,
        )
        r_plain = sp.callaway_santanna(**kw)
        r_id = sp.callaway_santanna(**kw, clustervars="i")
        np.testing.assert_array_equal(
            r_plain.detail["se"].values, r_id.detail["se"].values
        )

    def test_clustervars_without_bstrap_raises(self, panel):
        with pytest.raises(MethodIncompatibility, match="bstrap"):
            sp.callaway_santanna(
                panel, y="y", g="g", t="t", i="i", clustervars=["i", "cl"]
            )

    def test_two_extra_clustervars_raise(self, panel):
        df = panel.assign(cl2=panel["cl"])
        with pytest.raises(MethodIncompatibility, match="at most one"):
            sp.callaway_santanna(
                df,
                y="y",
                g="g",
                t="t",
                i="i",
                bstrap=True,
                clustervars=["i", "cl", "cl2"],
            )

    def test_time_varying_cluster_raises(self, panel):
        df = panel.copy()
        df["cl_tv"] = df["cl"] + (df["t"] > 4).astype(int)  # varies within unit
        with pytest.raises(MethodIncompatibility, match="time-varying"):
            sp.callaway_santanna(
                df,
                y="y",
                g="g",
                t="t",
                i="i",
                bstrap=True,
                clustervars=["i", "cl_tv"],
            )

    def test_missing_cluster_column_raises(self, panel):
        with pytest.raises(MethodIncompatibility, match="not found"):
            sp.callaway_santanna(
                panel,
                y="y",
                g="g",
                t="t",
                i="i",
                bstrap=True,
                clustervars=["i", "nope"],
            )

    def test_aggte_inherits_clustering(self):
        df = _staggered_panel(rho=2.0, seed=5)
        kw = dict(
            data=df,
            y="y",
            g="g",
            t="t",
            i="i",
            bstrap=True,
            biters=1999,
            random_state=9,
        )
        r_unit = sp.callaway_santanna(**kw)
        r_cl = sp.callaway_santanna(**kw, clustervars=["i", "cl"])
        a_unit = sp.aggte(r_unit, type="group", n_boot=1999, random_state=17)
        a_cl = sp.aggte(r_cl, type="group", n_boot=1999, random_state=17)
        # Same point estimates, wider clustered SEs.
        np.testing.assert_allclose(
            a_unit.detail["att"].values, a_cl.detail["att"].values
        )
        assert np.all(a_cl.detail["se"].values > a_unit.detail["se"].values)


# ----------------------------------------------------------------------
# did._core.multiplier_bootstrap
# ----------------------------------------------------------------------


class TestMultiplierBootstrap:
    def test_matches_analytic_iid(self):
        rng = np.random.default_rng(0)
        psi = rng.normal(size=(2000, 3))
        analytic = psi.std(axis=0) / np.sqrt(2000)
        se, crit = multiplier_bootstrap(psi, 2000, 0.05, 4000, 1)
        np.testing.assert_allclose(se, analytic, rtol=0.10)
        assert crit >= 1.959

    def test_cluster_recovers_true_se(self):
        # Shared cluster effect: unit-level bootstrap understates, the
        # clustered bootstrap must recover the true SE of the mean.
        rng = np.random.default_rng(1)
        n, m = 1200, 10
        n_cl = n // m
        cl = np.repeat(np.arange(n_cl), m)
        psi = np.repeat(rng.normal(size=(n_cl, 1)), m, axis=0)
        psi = psi + rng.normal(size=(n, 1)) * 0.5
        sums = psi.reshape(n_cl, m).sum(axis=1)
        true_se = np.sqrt((sums**2).sum()) / n
        se_u, _ = multiplier_bootstrap(psi, n, 0.05, 4000, 2)
        se_c, _ = multiplier_bootstrap(psi, n, 0.05, 4000, 2, cluster_ids=cl)
        assert abs(se_c[0] / true_se - 1) < 0.10
        assert se_u[0] < 0.5 * se_c[0]  # unit-level badly understates

    def test_weight_types_agree_asymptotically(self):
        rng = np.random.default_rng(3)
        psi = rng.normal(size=(1500, 2))
        se_r, _ = multiplier_bootstrap(
            psi, 1500, 0.05, 4000, 4, weight_type="rademacher"
        )
        se_m, _ = multiplier_bootstrap(psi, 1500, 0.05, 4000, 4, weight_type="mammen")
        np.testing.assert_allclose(se_r, se_m, rtol=0.10)

    def test_bad_weight_type_raises(self):
        with pytest.raises(ValueError, match="weight_type"):
            multiplier_bootstrap(np.ones((10, 1)), 10, 0.05, 99, 0, weight_type="webb")

    def test_cluster_length_mismatch_raises(self):
        with pytest.raises(ValueError, match="cluster_ids"):
            multiplier_bootstrap(
                np.ones((10, 1)), 10, 0.05, 99, 0, cluster_ids=np.arange(7)
            )


# ----------------------------------------------------------------------
# influence_functions / aggte_from_influence
# ----------------------------------------------------------------------


class TestInfluenceExport:
    def test_shape_and_columns(self, cs_analytic, panel):
        rif = sp.influence_functions(cs_analytic)
        n_units = panel["i"].nunique()
        n_pairs = len(cs_analytic.detail)
        assert len(rif) == n_units * n_pairs
        assert {
            "unit",
            "unit_cohort",
            "group",
            "time",
            "relative_time",
            "att",
            "influence",
        } <= set(rif.columns)
        # Per-unit cohort labels must match the DGP assignment.
        cohorts = rif.drop_duplicates("unit").set_index("unit")["unit_cohort"]
        expected = panel.groupby("i")["g"].first()
        pd.testing.assert_series_equal(
            cohorts.sort_index(),
            expected.sort_index(),
            check_names=False,
            check_dtype=False,
        )

    def test_roundtrip_matches_aggte_exactly(self, cs_analytic):
        rif = sp.influence_functions(cs_analytic)
        for agg_type in ("simple", "dynamic", "group", "calendar"):
            a_direct = sp.aggte(cs_analytic, type=agg_type, n_boot=499, random_state=23)
            a_rif = sp.aggte_from_influence(
                rif, type=agg_type, n_boot=499, random_state=23
            )
            np.testing.assert_allclose(
                a_direct.detail["att"].values,
                a_rif.detail["att"].values,
                atol=1e-12,
            )
            np.testing.assert_allclose(
                a_direct.detail["se"].values,
                a_rif.detail["se"].values,
                atol=1e-12,
            )
            assert np.isclose(a_direct.estimate, a_rif.estimate)
            assert np.isclose(a_direct.se, a_rif.se)

    def test_csv_roundtrip(self, cs_analytic, tmp_path):
        path = tmp_path / "rif.csv"
        sp.influence_functions(cs_analytic, path=path)
        assert path.exists()
        a1 = sp.aggte(cs_analytic, type="dynamic", n_boot=299, random_state=5)
        a2 = sp.aggte_from_influence(
            str(path), type="dynamic", n_boot=299, random_state=5
        )
        np.testing.assert_allclose(
            a1.detail["se"].values, a2.detail["se"].values, atol=1e-9
        )

    def test_cluster_column_preserved(self):
        df = _staggered_panel(rho=2.0, seed=5)
        r = sp.callaway_santanna(
            df,
            y="y",
            g="g",
            t="t",
            i="i",
            bstrap=True,
            clustervars=["i", "cl"],
            biters=299,
            random_state=1,
        )
        rif = sp.influence_functions(r)
        assert "cluster" in rif.columns
        a1 = sp.aggte(r, type="group", n_boot=499, random_state=6)
        a2 = sp.aggte_from_influence(rif, type="group", n_boot=499, random_state=6)
        np.testing.assert_allclose(
            a1.detail["se"].values, a2.detail["se"].values, atol=1e-12
        )

    def test_non_cs_result_raises(self, panel):
        r = sp.did_2x2(
            panel.assign(
                post=(panel["t"] >= 4).astype(int), treat=(panel["g"] == 4).astype(int)
            ),
            y="y",
            treat="treat",
            time="post",
        )
        with pytest.raises(MethodIncompatibility):
            sp.influence_functions(r)

    def test_missing_columns_raise(self):
        bad = pd.DataFrame({"unit": [1], "influence": [0.0]})
        with pytest.raises(MethodIncompatibility, match="missing required"):
            sp.aggte_from_influence(bad)

    def test_bad_source_type_raises(self):
        with pytest.raises(MethodIncompatibility, match="DataFrame or a file"):
            sp.aggte_from_influence(42)
