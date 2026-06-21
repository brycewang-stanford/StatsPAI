"""Tests for the v1.13 ML+Causal polish wave.

Covers:

1. ``forest.CausalForest.best_linear_projection`` now uses AIPW DR scores
   (Semenova-Chernozhukov 2021) — the SE is HC1 from the DR regression
   and the recovered slopes match the heterogeneity DGP.
2. ``mediation.mediate`` no longer silently substitutes the point
   estimate on bootstrap fit failures; ``model_info`` exposes
   ``n_boot_successful`` / ``n_boot_failed`` / ``boot_failure_rate``.
3. OPE namespace deduplication: ``sp.OPEResult`` is the canonical
   ``ope.estimators.OPEResult`` and ``isinstance(sp.direct_method(...),
   sp.OPEResult)`` is True.
4. Causal-discovery results expose ``.to_networkx()`` / ``.to_dot()`` /
   ``.plot()`` / ``.edge_list()`` for both dataclass results
   (LiNGAMResult, GESResult, …) and dict-shaped wrappers (DAGDict from
   ``notears`` / ``pc_algorithm``).
5. ``policy_tree`` returns a ``PolicyTreeResult`` (subclass of dict) with
   IF-based SE on ``value_policy``, plus ``summary`` / ``plot_tree`` /
   ``to_latex`` / ``cite``; legacy ``result['policy']`` access still
   works.
6. ``sp.dml_sensitivity`` (Chernozhukov-Cinelli-Newey 2022) returns RV_q,
   RV_qa, bias bound, and benchmark covariate effects.
7. ``sp.dml_diagnostics`` returns overlap, score density, balance,
   orthogonality test.
8. ``sp.cate_eval`` evaluates RATE/AUTOC/Qini for any CATE array
   (backbone-agnostic Yadlowsky 2025).
9. ``CausalResult.to_docx`` writes a publication-style .docx (existing
   API; the v1.13 polish wave audited it in place).
"""

from __future__ import annotations

import os
import tempfile
import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp

# ===================================================================== #
# 1. Forest BLP — AIPW DR scores
# ===================================================================== #


class TestForestBLP:

    def test_blp_recovers_heterogeneity_with_hc1_se(self):
        """Y = X1*T + N(0,1) ⇒ BLP coefficient on X1 ≈ 1, SE > 0."""
        rng = np.random.default_rng(0)
        n = 800
        X = rng.normal(size=(n, 3))
        T = rng.binomial(1, 0.5, n)
        Y = X[:, 0] * T + X[:, 1] + rng.normal(size=n)
        df = pd.DataFrame({"Y": Y, "T": T, "X1": X[:, 0], "X2": X[:, 1], "X3": X[:, 2]})
        cf = sp.causal_forest(
            formula="Y ~ T | X1 + X2 + X3", data=df, n_estimators=50, random_state=0
        )
        blp = cf.best_linear_projection()
        # Expected columns from the new DR-based implementation
        for col in ("coef", "se", "t", "p", "ci_lower", "ci_upper"):
            assert col in blp.columns
        # The heterogeneity is driven by feature index 0 (X0/X1)
        het_coef = blp.iloc[1]["coef"]
        het_se = blp.iloc[1]["se"]
        assert abs(het_coef - 1.0) < 0.40, f"BLP slope on X0 = {het_coef:.3f}"
        assert het_se > 0, "HC1 SE should be > 0"
        # diagnostics now records propensity-clip count
        assert "blp_n_clipped_propensities" in cf.diagnostics


# ===================================================================== #
# 2. Mediation bootstrap — no silent fallback
# ===================================================================== #


class TestMediationBootstrap:

    def test_no_silent_fallback_diagnostics_present(self):
        rng = np.random.default_rng(0)
        n = 400
        T = rng.binomial(1, 0.5, n)
        M = 0.5 * T + rng.normal(size=n)
        Y = 0.3 * M + 0.4 * T + rng.normal(size=n)
        df = pd.DataFrame({"Y": Y, "T": T, "M": M})
        res = sp.mediate(df, y="Y", treat="T", mediator="M", n_boot=200, seed=0)
        info = res.model_info
        assert "n_boot_requested" in info
        assert "n_boot_successful" in info
        assert "n_boot_failed" in info
        assert "boot_failure_rate" in info
        assert info["n_boot_requested"] == 200
        # On clean data we expect zero failures.
        assert info["n_boot_failed"] == 0
        assert info["boot_failure_rate"] == 0.0


# ===================================================================== #
# 3. OPE namespace deduplication
# ===================================================================== #


class TestOPEDedup:

    def test_canonical_OPEResult(self):
        from statspai.ope.estimators import OPEResult as Canonical
        from statspai.policy_learning.ope import OPEResult as PLAlias

        assert sp.OPEResult is Canonical
        assert PLAlias is Canonical

    def test_isinstance_works_across_entry_points(self):
        rng = np.random.default_rng(0)
        n = 200
        X = rng.normal(size=(n, 3))
        A = rng.integers(0, 3, size=n)
        R = rng.normal(size=n)
        pi_target = rng.dirichlet(np.ones(3), size=n)
        res = sp.direct_method(X, A, R, pi_target)
        assert isinstance(res, sp.OPEResult)
        assert res.method == "direct"
        assert res.estimator == "direct"  # back-compat property
        assert res.n_obs == n


# ===================================================================== #
# 4. Causal-discovery graph viz
# ===================================================================== #


@pytest.fixture
def small_dag_data():
    rng = np.random.default_rng(0)
    n = 200
    x1 = rng.normal(size=n)
    x2 = 0.6 * x1 + rng.normal(size=n)
    x3 = 0.4 * x2 - 0.3 * x1 + rng.normal(size=n)
    return pd.DataFrame({"x1": x1, "x2": x2, "x3": x3})


class TestCausalDiscoveryViz:

    def test_lingam_to_networkx(self, small_dag_data):
        res = sp.lingam(small_dag_data)
        try:
            import networkx as nx  # noqa: F401
        except ImportError:
            pytest.skip("networkx not installed")
        G = res.to_networkx()
        assert hasattr(G, "nodes") and hasattr(G, "edges")
        assert set(G.nodes) == {"x1", "x2", "x3"}

    def test_lingam_to_dot(self, small_dag_data):
        res = sp.lingam(small_dag_data)
        dot = res.to_dot()
        assert dot.startswith("digraph")
        assert "x1" in dot

    def test_notears_dagdict_methods(self, small_dag_data):
        res = sp.notears(small_dag_data)
        # DAGDict still behaves like a dict
        assert isinstance(res, dict)
        assert "adjacency" in res
        assert "variables" in res
        # …but also has the viz methods
        assert hasattr(res, "to_networkx")
        assert hasattr(res, "to_dot")
        assert hasattr(res, "plot")
        edges = res.edge_list()
        assert isinstance(edges, list)
        for e in edges:
            assert len(e) == 3  # (parent, child, weight)

    def test_pc_algorithm_dagdict_methods(self, small_dag_data):
        res = sp.pc_algorithm(small_dag_data)
        assert isinstance(res, dict)
        assert hasattr(res, "to_networkx") and hasattr(res, "edge_list")

    def test_module_exports_helpers(self):
        from statspai.causal_discovery import (
            to_networkx,
            to_dot,
            plot_dag,
            edge_list,
            shd,
        )

        # Standalone usage on a 3-node DAG
        A = np.array([[0, 0.5, 0.0], [0, 0.0, 0.4], [0, 0.0, 0.0]])
        names = ["a", "b", "c"]
        edges = edge_list(A, names)
        assert ("a", "b", 0.5) in edges
        assert ("b", "c", 0.4) in edges
        assert shd(A, A) == 0
        B = A.copy()
        B[0, 1] = 0.0
        assert shd(B, A) == 1


# ===================================================================== #
# 5. PolicyTreeResult — rich result class with IF-based SE
# ===================================================================== #


class TestPolicyTreeResult:

    @pytest.fixture
    def policy_data(self):
        rng = np.random.default_rng(0)
        n = 400
        X = rng.normal(size=(n, 3))
        T = rng.binomial(1, 0.5, n)
        # CATE varies with X1; treating helps when X1 > 0
        mu = 2 * X[:, 0]
        Y = T * mu + 0.5 * rng.normal(size=n)
        return pd.DataFrame(
            {"Y": Y, "T": T, "X1": X[:, 0], "X2": X[:, 1], "X3": X[:, 2]}
        )

    def test_returns_PolicyTreeResult(self, policy_data):
        res = sp.policy_tree(
            policy_data,
            y="Y",
            treat="T",
            covariates=["X1", "X2", "X3"],
            max_depth=2,
            min_leaf_size=30,
            n_folds=3,
        )
        assert isinstance(res, sp.PolicyTreeResult)
        # Backwards compat
        assert isinstance(res, dict)
        for key in ("policy", "value_policy", "rules", "fraction_treated"):
            assert key in res

    def test_value_policy_has_se_and_ci(self, policy_data):
        res = sp.policy_tree(
            policy_data,
            y="Y",
            treat="T",
            covariates=["X1", "X2", "X3"],
            max_depth=2,
            min_leaf_size=30,
            n_folds=3,
        )
        assert np.isfinite(res.value_policy_se)
        assert res.value_policy_se > 0
        lo, hi = res.value_policy_ci
        assert lo < res.value_policy < hi

    def test_summary_and_cite(self, policy_data):
        res = sp.policy_tree(
            policy_data,
            y="Y",
            treat="T",
            covariates=["X1", "X2", "X3"],
            max_depth=2,
            min_leaf_size=30,
            n_folds=3,
        )
        s = res.summary()
        assert "Policy Tree" in s
        bib = res.cite()
        assert "athey2021policy" in bib


# ===================================================================== #
# 6. DML-OVB sensitivity
# ===================================================================== #


@pytest.fixture
def dml_data():
    rng = np.random.default_rng(0)
    n = 800
    X = rng.normal(size=(n, 4))
    T = (X[:, 0] + X[:, 1] + 0.5 * rng.normal(size=n) > 0).astype(int)
    Y = 0.5 * T + 0.4 * X[:, 0] + 0.2 * X[:, 1] + rng.normal(size=n)
    return pd.DataFrame({"Y": Y, "T": T, **{f"X{j+1}": X[:, j] for j in range(4)}})


class TestDMLSensitivity:

    def test_basic_rv_q(self, dml_data):
        res = sp.dml(
            data=dml_data,
            y="Y",
            d="T",
            covariates=["X1", "X2", "X3", "X4"],
            model="plr",
            n_folds=3,
            random_state=0,
        )
        sens = sp.dml_sensitivity(res, q=1.0, cf_y=0.10, cf_d=0.10)
        assert 0.0 <= sens.rv_q <= 1.0
        assert 0.0 <= sens.rv_qa <= sens.rv_q + 1e-9  # rv_qa ≤ rv_q
        assert sens.bias_bound > 0
        assert sens.adjusted_estimate_low <= sens.estimate
        assert sens.adjusted_estimate_high >= sens.estimate

    def test_benchmarks(self, dml_data):
        res = sp.dml(
            data=dml_data,
            y="Y",
            d="T",
            covariates=["X1", "X2", "X3", "X4"],
            model="plr",
            n_folds=3,
            random_state=0,
        )
        sens = sp.dml_sensitivity(
            res,
            q=1.0,
            cf_y=0.05,
            cf_d=0.05,
            benchmark_covariates=["X1", "X2"],
        )
        assert not sens.benchmarks.empty
        assert set(sens.benchmarks["variable"]) == {"X1", "X2"}

    def test_summary_string(self, dml_data):
        res = sp.dml(
            data=dml_data,
            y="Y",
            d="T",
            covariates=["X1", "X2", "X3", "X4"],
            model="plr",
            n_folds=3,
            random_state=0,
        )
        sens = sp.dml_sensitivity(res, q=1.0)
        text = sens.summary()
        assert "Robustness value" in text
        assert "Long Story Short" in text

    def test_missing_residuals_raises(self):
        # A bare CausalResult without DML residuals should be rejected.
        from statspai.core.results import CausalResult

        bare = CausalResult(
            method="OLS",
            estimand="ATE",
            estimate=1.0,
            se=0.1,
            pvalue=0.0,
            ci=(0.8, 1.2),
            alpha=0.05,
            n_obs=100,
            model_info={},
        )
        with pytest.raises(ValueError, match="post-fit residuals"):
            sp.dml_sensitivity(bare)


# ===================================================================== #
# 7. DML diagnostics
# ===================================================================== #


class TestDMLDiagnostics:

    def test_basic_report(self, dml_data):
        res = sp.dml(
            data=dml_data,
            y="Y",
            d="T",
            covariates=["X1", "X2", "X3", "X4"],
            model="plr",
            n_folds=3,
            random_state=0,
        )
        diag = sp.dml_diagnostics(res)
        assert diag.method == "PLR"
        assert diag.score_sd > 0
        # Orthogonality score should be approximately zero by construction
        # for PLR's psi (mean-centred residual moment).
        assert abs(diag.orth_stat) < 1e-6
        # Balance table should include all four covariates
        assert set(diag.balance_table["variable"]) == {"X1", "X2", "X3", "X4"}

    def test_summary_string(self, dml_data):
        res = sp.dml(
            data=dml_data,
            y="Y",
            d="T",
            covariates=["X1", "X2", "X3", "X4"],
            model="plr",
            n_folds=3,
            random_state=0,
        )
        diag = sp.dml_diagnostics(res)
        s = diag.summary()
        assert "Overlap" in s
        assert "Score density" in s
        assert "Orthogonality" in s


# ===================================================================== #
# 8. sp.cate_eval — RATE/AUTOC/Qini for any CATE array
# ===================================================================== #


class TestCATEEval:

    def test_good_cate_has_positive_AUTOC(self):
        """A CATE that ranks correctly should have positive AUTOC."""
        rng = np.random.default_rng(0)
        n = 1000
        X = rng.normal(size=(n, 4))
        T = rng.binomial(1, 0.5, n)
        Y = T * 2 * X[:, 0] + 0.4 * X[:, 1] + rng.normal(size=n)
        cate_good = 2 * X[:, 0]
        res = sp.cate_eval(cate_good, Y, T, X=X, n_folds=3, random_state=0)
        assert res.autoc > 0
        # 95% CI shouldn't include zero on this strong signal
        assert res.autoc_ci[0] > 0
        assert res.toc_curve.shape[1] == 2  # columns: q, toc

    def test_random_cate_has_AUTOC_near_zero(self):
        rng = np.random.default_rng(1)
        n = 1000
        X = rng.normal(size=(n, 4))
        T = rng.binomial(1, 0.5, n)
        Y = T * 2 * X[:, 0] + rng.normal(size=n)
        cate_random = rng.normal(size=n)
        res = sp.cate_eval(cate_random, Y, T, X=X, n_folds=3, random_state=0)
        # Random CATE shouldn't show a large signal — assert it's not
        # meaningfully nonzero (95% CI brackets zero in expectation).
        assert res.autoc_ci[0] <= 0 <= res.autoc_ci[1]

    def test_requires_X_or_pre_computed_nuisances(self):
        rng = np.random.default_rng(0)
        Y = rng.normal(size=100)
        T = rng.binomial(1, 0.5, 100)
        cate = rng.normal(size=100)
        with pytest.raises(ValueError, match="Provide X"):
            sp.cate_eval(cate, Y, T)


# ===================================================================== #
# 9. CausalResult.to_docx — Word export (existing API verified for the
# DML CausalResult path during the v1.13 polish audit).
# ===================================================================== #


class TestToDocx:

    def test_dml_to_docx(self, dml_data):
        res = sp.dml(
            data=dml_data,
            y="Y",
            d="T",
            covariates=["X1", "X2", "X3", "X4"],
            model="plr",
            n_folds=3,
            random_state=0,
        )
        try:
            import docx  # noqa: F401
        except ImportError:
            pytest.skip("python-docx not installed")
        with tempfile.NamedTemporaryFile(suffix=".docx", delete=False) as f:
            path = f.name
        try:
            res.to_docx(path)
            assert os.path.getsize(path) > 1000  # well-formed .docx is > 1KB
        finally:
            os.remove(path)
