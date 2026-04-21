"""
Tests for the v1.5 family dispatchers.

Covers:
- ``sp.mr(method=...)`` → single-exposure point estimators, multi-
  exposure extensions, diagnostics, and the all-methods wrapper.
- ``sp.conformal(kind=...)`` → Lei-Candès 2021 core + 2025-2026 frontier.
- ``sp.interference(design=...)`` → partial / network / cluster RCT.

Each dispatcher's output is compared to the direct top-level call —
the two paths must produce identical numbers up to seed-controlled
RNG noise.  Unknown-method / kind / design routes must raise
``ValueError`` with the available-list in the message.

The tests also verify that the dispatchers are registered for
``sp.describe_function()`` and that the 5 previously-unregistered
family functions (network_exposure / peer_effects /
weighted_conformal_prediction / conformal_counterfactual /
conformal_ite_interval) now surface in ``sp.list_functions()``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp


# ======================================================================
# Shared fixtures
# ======================================================================


def _mr_arrays(seed: int = 0, n_snps: int = 10):
    rng = np.random.default_rng(seed)
    bx = np.abs(rng.normal(0.1, 0.04, size=n_snps))
    by = 2.0 * bx + rng.normal(0, 0.02, size=n_snps)
    sx = np.full(n_snps, 0.02)
    sy = np.full(n_snps, 0.08)
    return bx, by, sx, sy


def _conformal_df(n: int = 300, seed: int = 0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 2))
    d = (rng.random(n) < 0.5).astype(int)
    y = 0.2 * X[:, 0] - 0.1 * X[:, 1] + 0.3 * d + rng.normal(0, 0.5, n)
    return pd.DataFrame({"y": y, "d": d, "x1": X[:, 0], "x2": X[:, 1]})


def _spillover_df(n_households: int = 80, seed: int = 0):
    rng = np.random.default_rng(seed)
    households = np.repeat(np.arange(n_households), 4)
    n = len(households)
    d = (rng.random(n) < 0.5).astype(int)
    peer_share = (
        pd.DataFrame({"h": households, "d": d})
        .groupby("h")["d"]
        .transform(lambda s: (s.sum() - s) / 3)
    )
    y = -0.4 * d - 0.6 * peer_share.values + rng.normal(0, 0.3, n)
    return pd.DataFrame({"y": y, "d": d, "h": households})


# ======================================================================
# sp.mr dispatcher
# ======================================================================


class TestMRDispatcher:
    def test_ivw_matches_direct_call(self):
        bx, by, sx, sy = _mr_arrays()
        r_disp = sp.mr("ivw", beta_exposure=bx, beta_outcome=by,
                       se_exposure=sx, se_outcome=sy)
        r_direct = sp.mr_ivw(bx, by, sx, sy)
        assert r_disp["estimate"] == pytest.approx(r_direct["estimate"])
        assert r_disp["se"] == pytest.approx(r_direct["se"])

    def test_egger_matches_direct_call(self):
        bx, by, sx, sy = _mr_arrays()
        r_disp = sp.mr("egger", beta_exposure=bx, beta_outcome=by,
                       se_exposure=sx, se_outcome=sy)
        r_direct = sp.mr_egger(bx, by, sx, sy)
        assert r_disp["estimate"] == pytest.approx(r_direct["estimate"])
        assert r_disp["intercept"] == pytest.approx(r_direct["intercept"])

    def test_median_matches_direct_call(self):
        bx, by, sx, sy = _mr_arrays()
        r_disp = sp.mr("median", beta_exposure=bx, beta_outcome=by,
                       se_exposure=sx, se_outcome=sy,
                       n_boot=100, seed=123)
        r_direct = sp.mr_median(bx, by, sx, sy, n_boot=100, seed=123)
        assert r_disp["estimate"] == pytest.approx(r_direct["estimate"])

    def test_penalized_median_sets_kwarg(self):
        """penalized_median alias should flip penalized=True."""
        bx, by, sx, sy = _mr_arrays()
        r_disp = sp.mr("penalized_median", beta_exposure=bx, beta_outcome=by,
                       se_exposure=sx, se_outcome=sy, n_boot=100, seed=123)
        r_direct = sp.mr_median(bx, by, sx, sy,
                                penalized=True, n_boot=100, seed=123)
        assert r_disp["estimate"] == pytest.approx(r_direct["estimate"])

    def test_mode_matches_direct_call(self):
        bx, by, sx, sy = _mr_arrays()
        r_disp = sp.mr("mode", beta_exposure=bx, beta_outcome=by,
                       se_exposure=sx, se_outcome=sy, n_boot=100, seed=0)
        r_direct = sp.mr_mode(bx, by, sx, sy, n_boot=100, seed=0)
        assert r_disp.estimate == pytest.approx(r_direct.estimate)

    def test_simple_mode_alias(self):
        """simple_mode alias should set method='simple'."""
        bx, by, sx, sy = _mr_arrays()
        r_disp = sp.mr("simple_mode", beta_exposure=bx, beta_outcome=by,
                       se_exposure=sx, se_outcome=sy, n_boot=100, seed=0)
        r_direct = sp.mr_mode(bx, by, sx, sy,
                              method="simple", n_boot=100, seed=0)
        assert r_disp.estimate == pytest.approx(r_direct.estimate)

    def test_all_methods_routes_to_mendelian_randomization(self):
        """`method='all'` returns an MRResult with IVW+Egger+Median rows."""
        bx, by, sx, sy = _mr_arrays()
        df = pd.DataFrame({"bx": bx, "by": by, "sx": sx, "sy": sy})
        r = sp.mr("all", data=df,
                  beta_exposure="bx", beta_outcome="by",
                  se_exposure="sx", se_outcome="sy")
        assert hasattr(r, "estimates")
        methods = set(r.estimates["method"].tolist())
        assert {"IVW", "MR-Egger", "Weighted Median"} <= methods

    def test_mvmr_routes_correctly(self):
        rng = np.random.default_rng(0)
        n_snps = 40
        bx1 = np.abs(rng.normal(0.1, 0.04, n_snps))
        bx2 = np.abs(rng.normal(0.1, 0.04, n_snps))
        by = 1.5 * bx1 + 0.5 * bx2 + rng.normal(0, 0.02, n_snps)
        df = pd.DataFrame({"bx1": bx1, "bx2": bx2, "by": by,
                           "sy": np.full(n_snps, 0.08)})
        r_disp = sp.mr("mvmr", snp_associations=df,
                       outcome="by", outcome_se="sy",
                       exposures=["bx1", "bx2"])
        r_direct = sp.mr_multivariable(df, outcome="by", outcome_se="sy",
                                        exposures=["bx1", "bx2"])
        assert r_disp.direct_effect.equals(r_direct.direct_effect)

    def test_unknown_method_raises(self):
        bx, by, sx, sy = _mr_arrays()
        with pytest.raises(ValueError, match="Unknown MR method"):
            sp.mr("totally_fake", beta_exposure=bx, beta_outcome=by,
                  se_exposure=sx, se_outcome=sy)

    def test_available_methods_is_nonempty_sorted(self):
        methods = sp.mr_available_methods()
        assert len(methods) > 10
        assert methods == sorted(methods)
        assert "ivw" in methods and "egger" in methods and "mode" in methods

    def test_dispatcher_in_registry(self):
        assert "mr" in sp.list_functions()
        spec = sp.describe_function("mr")
        assert "dispatcher" in spec["tags"] or "mr" in spec["tags"]


# ======================================================================
# sp.conformal dispatcher
# ======================================================================


class TestConformalDispatcher:
    def test_cate_matches_direct_call(self):
        df = _conformal_df(n=200)
        r_disp = sp.conformal("cate", data=df, y="y", treat="d",
                              covariates=["x1", "x2"],
                              alpha=0.1, random_state=0)
        r_direct = sp.conformal_cate(data=df, y="y", treat="d",
                                      covariates=["x1", "x2"],
                                      alpha=0.1, random_state=0)
        assert r_disp.estimate == pytest.approx(r_direct.estimate)

    def test_ite_routes_to_ite_interval(self):
        df = _conformal_df(n=200)
        r_disp = sp.conformal("ite", data=df, y="y", treat="d",
                              covariates=["x1", "x2"],
                              alpha=0.1, random_state=0)
        r_direct = sp.conformal_ite_interval(
            data=df, y="y", treat="d", covariates=["x1", "x2"],
            alpha=0.1, random_state=0,
        )
        # same point estimates and same bounds under same seed
        assert np.allclose(r_disp.point, r_direct.point)
        assert np.allclose(r_disp.lower, r_direct.lower)

    def test_counterfactual_matches_direct_call(self):
        df = _conformal_df(n=200)
        r_disp = sp.conformal("counterfactual", data=df, y="y", treat="d",
                              covariates=["x1", "x2"],
                              alpha=0.1, random_state=0)
        r_direct = sp.conformal_counterfactual(
            data=df, y="y", treat="d", covariates=["x1", "x2"],
            alpha=0.1, random_state=0,
        )
        assert np.allclose(r_disp.lower_Y1, r_direct.lower_Y1)

    def test_debiased_matches_direct_call(self):
        df = _conformal_df(n=200)
        r_disp = sp.conformal("debiased", data=df, y="y", treat="d",
                              covariates=["x1", "x2"], alpha=0.1, seed=0)
        r_direct = sp.conformal_debiased_ml(
            data=df, y="y", treat="d", covariates=["x1", "x2"],
            alpha=0.1, seed=0,
        )
        assert np.allclose(r_disp.intervals, r_direct.intervals)

    def test_continuous_matches_direct_call(self):
        rng = np.random.default_rng(0)
        n = 200
        X = rng.normal(size=(n, 2))
        dose = rng.uniform(0, 1, n)
        y = 0.5 * dose + 0.2 * X[:, 0] + rng.normal(0, 0.3, n)
        train = pd.DataFrame({"y": y, "dose": dose,
                               "x1": X[:, 0], "x2": X[:, 1]})
        test = train.head(20).copy()
        r_disp = sp.conformal("continuous", data=train, y="y",
                              treatment="dose", covariates=["x1", "x2"],
                              test_data=test, alpha=0.1, random_state=0)
        r_direct = sp.conformal_continuous(
            data=train, y="y", treatment="dose",
            covariates=["x1", "x2"], test_data=test,
            alpha=0.1, random_state=0,
        )
        assert r_disp.quantile == pytest.approx(r_direct.quantile)

    def test_unknown_kind_raises(self):
        df = _conformal_df(n=200)
        with pytest.raises(ValueError, match="Unknown conformal kind"):
            sp.conformal("totally_fake", data=df, y="y", treat="d",
                         covariates=["x1", "x2"])

    def test_available_kinds_is_sorted(self):
        kinds = sp.conformal_available_kinds()
        assert kinds == sorted(kinds)
        assert "cate" in kinds
        assert "ite" in kinds
        assert "continuous" in kinds

    def test_dispatcher_in_registry(self):
        assert "conformal" in sp.list_functions()
        spec = sp.describe_function("conformal")
        assert "dispatcher" in spec["tags"]


# ======================================================================
# sp.interference dispatcher
# ======================================================================


class TestInterferenceDispatcher:
    def test_partial_matches_direct_call(self):
        df = _spillover_df(n_households=60)
        r_disp = sp.interference("partial", data=df, y="y", treat="d",
                                  cluster="h", n_bootstrap=50, random_state=0)
        r_direct = sp.spillover(data=df, y="y", treat="d", cluster="h",
                                 n_bootstrap=50, random_state=0)
        assert r_disp.estimate == pytest.approx(r_direct.estimate)

    def test_network_hte_matches_direct_call(self):
        rng = np.random.default_rng(0)
        n = 300
        x1 = rng.normal(0, 1, n)
        x2 = rng.normal(0, 1, n)
        d = rng.binomial(1, 0.5, n).astype(float)
        e = np.clip(0.3 * d + 0.4 * rng.uniform(size=n), 0, 1)
        y = 0.2 * x1 - 0.1 * x2 + 0.4 * d + 0.3 * e + rng.normal(0, 0.3, n)
        df = pd.DataFrame({"y": y, "d": d, "e": e, "x1": x1, "x2": x2})
        r_disp = sp.interference("network_hte", data=df, y="y",
                                  treatment="d", neighbor_exposure="e",
                                  covariates=["x1", "x2"],
                                  n_folds=5, random_state=0)
        r_direct = sp.network_hte(data=df, y="y", treatment="d",
                                   neighbor_exposure="e",
                                   covariates=["x1", "x2"],
                                   n_folds=5, random_state=0)
        assert r_disp.direct_effect == pytest.approx(r_direct.direct_effect)

    def test_network_exposure_matches_direct_call(self):
        rng = np.random.default_rng(0)
        n = 50
        # ring graph
        A = np.zeros((n, n), dtype=int)
        for i in range(n):
            A[i, (i + 1) % n] = 1
            A[(i + 1) % n, i] = 1
        Z = (rng.random(n) < 0.5).astype(int)
        Y = (0.3 * Z + 0.2 * (A @ Z > 0).astype(float)
             + rng.normal(0, 0.2, n))
        r_disp = sp.interference("network_exposure",
                                  Y=Y, Z=Z, adjacency=A,
                                  p_treat=0.5, n_sim=200, seed=0)
        r_direct = sp.network_exposure(
            Y=Y, Z=Z, adjacency=A, p_treat=0.5, n_sim=200, seed=0,
        )
        assert r_disp.n_obs == r_direct.n_obs
        assert r_disp.estimates.equals(r_direct.estimates)

    def test_unknown_design_raises(self):
        df = _spillover_df(n_households=20)
        with pytest.raises(ValueError, match="Unknown interference design"):
            sp.interference("totally_fake", data=df, y="y", treat="d",
                            cluster="h")

    def test_available_designs_is_sorted(self):
        designs = sp.interference_available_designs()
        assert designs == sorted(designs)
        assert "partial" in designs
        assert "network_exposure" in designs
        assert "cluster_staggered" in designs

    def test_dispatcher_in_registry(self):
        assert "interference" in sp.list_functions()
        spec = sp.describe_function("interference")
        assert "dispatcher" in spec["tags"]


# ======================================================================
# Registry coverage fixes — these 5 were previously sp.attributes but
# not in sp.list_functions()
# ======================================================================


class TestRegistryCoverageFixes:
    @pytest.mark.parametrize("name", [
        "network_exposure",
        "peer_effects",
        "weighted_conformal_prediction",
        "conformal_counterfactual",
        "conformal_ite_interval",
    ])
    def test_family_function_now_registered(self, name):
        assert hasattr(sp, name), f"sp.{name} must exist"
        assert name in sp.list_functions(), (
            f"sp.{name} must be in sp.list_functions() so agents can "
            "discover it via sp.help / sp.describe_function."
        )
