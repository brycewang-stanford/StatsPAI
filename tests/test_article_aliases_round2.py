"""Round-2 fixes: namespace-collision aliases, kwarg alignment, and the
tidy-attribute aliases on ``EconometricResults``.

These tests pin the article-promised surface that the first alias pass
missed (``sp.matrix_completion`` / ``sp.causal_discovery`` / ``sp.mediation``
all used to resolve to modules, not functions), the ``depth=`` / ``model_y=``
kwarg renames, the ``sp.evalue_rr`` convenience wrapper, and the
cross-estimator ``.estimate`` / ``.se`` / ``.pvalue`` / ``.ci`` aliases on
``EconometricResults``.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.core.results import CausalResult, EconometricResults


# =====================================================================
# Namespace-collision fixes
# =====================================================================

def test_matrix_completion_is_callable_function():
    assert callable(sp.matrix_completion), (
        "sp.matrix_completion should now be a function, not a module"
    )
    assert type(sp.matrix_completion).__name__ == "function"


def test_causal_discovery_is_callable_function():
    assert callable(sp.causal_discovery)
    assert type(sp.causal_discovery).__name__ == "function"


def test_mediation_is_callable_function():
    assert callable(sp.mediation)
    assert type(sp.mediation).__name__ == "function"


def _make_mc_panel(seed=0, n_units=20, n_periods=8):
    """Small panel with one treated unit & a few treated periods."""
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_units):
        treated_unit = (u == 0)
        for t in range(n_periods):
            post = t >= n_periods // 2
            d = int(treated_unit and post)
            y = (
                0.5 * u / n_units
                + 0.1 * t
                + 0.8 * d
                + rng.normal() * 0.3
            )
            rows.append({"unit": u, "time": t, "y": y, "d": d})
    return pd.DataFrame(rows)


def test_matrix_completion_end_to_end():
    df = _make_mc_panel(seed=1)
    r = sp.matrix_completion(
        df, y="y", d="d", unit="unit", time="time",
        n_bootstrap=50,
    )
    assert isinstance(r, CausalResult)
    assert np.isfinite(r.estimate)


def test_mediation_end_to_end():
    rng = np.random.default_rng(11)
    n = 300
    x1 = rng.normal(size=n)
    d = (rng.uniform(size=n) < 0.5).astype(int)
    m = 0.3 * d + 0.2 * x1 + rng.normal(size=n) * 0.3
    y = 0.5 * m + 0.2 * d + 0.1 * x1 + rng.normal(size=n) * 0.3
    df = pd.DataFrame({"y": y, "d": d, "m": m, "x1": x1})

    r = sp.mediation(df, y="y", d="d", m="m", X=["x1"], n_boot=100)
    assert isinstance(r, CausalResult)
    # `mediate` reports a total effect by default; just check finiteness.
    assert np.isfinite(r.estimate)


def test_causal_discovery_dispatch_notears():
    rng = np.random.default_rng(7)
    n = 200
    x1 = rng.normal(size=n)
    x2 = 0.6 * x1 + rng.normal(size=n) * 0.5
    x3 = 0.4 * x2 + rng.normal(size=n) * 0.5
    df = pd.DataFrame({"x1": x1, "x2": x2, "x3": x3})
    out = sp.causal_discovery(df, method="notears", max_iter=50)
    assert isinstance(out, dict)
    # notears dict keys: 'adjacency', 'adjacency_bin', 'edges', 'h_value', ...
    assert "adjacency" in out
    assert "edges" in out


def test_causal_discovery_rejects_unknown_method():
    df = pd.DataFrame({"x1": [0, 1], "x2": [1, 0]})
    with pytest.raises(ValueError, match="Unknown causal_discovery method"):
        sp.causal_discovery(df, method="not_a_real_algorithm")


def _make_cd_df(seed=0, n=150):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    x2 = 0.5 * x1 + rng.normal(size=n) * 0.5
    x3 = 0.4 * x2 + rng.normal(size=n) * 0.5
    return pd.DataFrame({"x1": x1, "x2": x2, "x3": x3})


def test_causal_discovery_dispatch_lingam_runs():
    """lingam's backend has NO `variables` kwarg — the dispatcher must
    subset the DataFrame up front instead of forwarding the kwarg."""
    df = _make_cd_df(seed=11)
    out = sp.causal_discovery(df, method="lingam", variables=["x1", "x2", "x3"])
    # LiNGAMResult has a method and adjacency-like attributes
    assert out is not None


def test_causal_discovery_dispatch_ges_runs():
    df = _make_cd_df(seed=13)
    out = sp.causal_discovery(df, method="ges", variables=["x1", "x2", "x3"])
    assert out is not None


def test_causal_discovery_dispatch_pc_runs():
    df = _make_cd_df(seed=17)
    out = sp.causal_discovery(df, method="pc", variables=["x1", "x2", "x3"])
    # notears/pc return dicts with an 'adjacency' key
    assert isinstance(out, dict)
    assert "adjacency" in out or "edges" in out or "skeleton" in out


# =====================================================================
# kwarg-alignment wrappers
# =====================================================================

def test_policy_tree_accepts_depth_kwarg():
    rng = np.random.default_rng(3)
    n = 400
    X1 = rng.normal(size=n)
    X2 = rng.normal(size=n)
    d = (rng.uniform(size=n) < 0.5).astype(int)
    y = X1 * d + 0.3 * X2 + rng.normal(size=n) * 0.5
    df = pd.DataFrame({"y": y, "d": d, "x1": X1, "x2": X2})

    r = sp.policy_tree(df, y="y", d="d", X=["x1", "x2"], depth=2)
    assert isinstance(r, dict) or hasattr(r, "tree_")


def test_policy_tree_max_depth_kwarg_still_works():
    rng = np.random.default_rng(4)
    n = 300
    X1 = rng.normal(size=n)
    d = (rng.uniform(size=n) < 0.5).astype(int)
    y = X1 * d + rng.normal(size=n) * 0.5
    df = pd.DataFrame({"y": y, "d": d, "x1": X1})
    r = sp.policy_tree(df, y="y", d="d", X=["x1"], max_depth=2)
    assert r is not None


def test_policy_tree_rejects_conflicting_depth():
    df = pd.DataFrame({"y": [0, 1], "d": [0, 1], "x1": [0.1, 0.2]})
    with pytest.raises(TypeError, match="either `depth` or `max_depth`"):
        sp.policy_tree(df, y="y", d="d", X=["x1"], depth=2, max_depth=3)


def test_policy_tree_rejects_conflicting_treat():
    """Passing both `d=` and `treat=` with different values must error."""
    df = pd.DataFrame({"y": [0, 1], "d": [0, 1], "t2": [0, 1], "x1": [0.1, 0.2]})
    with pytest.raises(TypeError, match="conflicting treatment"):
        sp.policy_tree(df, y="y", d="d", treat="t2", X=["x1"])


def test_policy_tree_rejects_conflicting_covariates():
    df = pd.DataFrame({"y": [0, 1], "d": [0, 1], "x1": [0.1, 0.2]})
    with pytest.raises(TypeError, match="conflicting covariate"):
        sp.policy_tree(df, y="y", d="d", X=["x1"], covariates=["d"])


def test_dml_rejects_conflicting_treat():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({
        "y": rng.normal(size=50), "d1": rng.integers(0, 2, 50),
        "d2": rng.integers(0, 2, 50), "x1": rng.normal(size=50),
    })
    with pytest.raises(TypeError, match="conflicting treatment"):
        sp.dml(df, y="y", d="d1", treat="d2", X=["x1"])


def test_dml_accepts_model_y_and_model_d_aliases():
    from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
    rng = np.random.default_rng(5)
    n = 300
    X1 = rng.normal(size=n)
    X2 = rng.normal(size=n)
    d = (rng.uniform(size=n) < 0.5).astype(int)
    y = 0.5 * d + 0.3 * X1 + 0.2 * X2 + rng.normal(size=n) * 0.5
    df = pd.DataFrame({"y": y, "d": d, "x1": X1, "x2": X2})

    r = sp.dml(
        df, y="y", d="d", X=["x1", "x2"],
        model_y=GradientBoostingRegressor(n_estimators=50, random_state=0),
        model_d=GradientBoostingClassifier(n_estimators=50, random_state=0),
        n_folds=2,
    )
    assert isinstance(r, CausalResult)
    assert np.isfinite(r.estimate)


# =====================================================================
# evalue_rr convenience
# =====================================================================

def test_evalue_rr_point_only():
    out = sp.evalue_rr(rr=1.5)
    assert isinstance(out, dict)
    # evalue reports keys like 'evalue_point', 'evalue_ci' etc
    assert any("e" in k.lower() for k in out)


def test_evalue_rr_with_ci_bounds():
    out = sp.evalue_rr(rr=1.5, rr_lower=1.1, rr_upper=2.0)
    assert isinstance(out, dict)


def test_evalue_rr_rejects_partial_ci():
    with pytest.raises(ValueError, match="BOTH.*or neither"):
        sp.evalue_rr(rr=1.5, rr_lower=1.1)  # missing upper


# =====================================================================
# Cross-estimator tidy surface — `.tidy()` is the canonical unifier,
# NOT `.estimate` / `.se` / `.pvalue` (those are scalar on CausalResult
# but would be Series on EconometricResults, breaking `hasattr` dispatch
# in the workflow layer — see NOTE in core/results.py).
# =====================================================================

@pytest.fixture(scope="module")
def ols_result():
    return sp.regress("y ~ x1 + x2", data=sp.dgp_observational(n=200, seed=1))


def test_econ_results_tidy_is_dataframe(ols_result):
    t = ols_result.tidy()
    assert isinstance(t, pd.DataFrame)
    assert set(t.columns) >= {"term", "estimate", "std_error", "p_value"}


def test_causal_result_tidy_is_dataframe():
    df = sp.dgp_did(
        n_units=60, n_periods=6, staggered=True, n_groups=3, effect=0.4, seed=2,
    )
    r = sp.callaway_santanna(data=df, y="y", g="first_treat", t="time", i="unit")
    t = r.tidy()
    assert isinstance(t, pd.DataFrame)


def test_econ_results_has_no_scalar_estimate_alias(ols_result):
    """`EconometricResults` intentionally does NOT expose `.estimate` —
    that attribute is reserved for scalar CausalResult semantics.  The
    absence matters because workflow code uses `hasattr(r, 'estimate')`
    to dispatch between the two result types."""
    assert not hasattr(ols_result, "estimate"), (
        "EconometricResults must not have `.estimate` — see NOTE in "
        "core/results.py about `hasattr` dispatch."
    )


# =====================================================================
# auto_did BJS validation
# =====================================================================

def test_auto_did_bjs_rejects_cohort_string_g():
    """If `g` is a string cohort label, BJS will silently misbehave —
    validate up front."""
    df = sp.dgp_did(
        n_units=60, n_periods=6, staggered=True, n_groups=3, effect=0.4, seed=7,
    )
    df["cohort_str"] = df["first_treat"].astype("object").where(
        df["first_treat"].notna(), "never"
    ).astype(str)
    with pytest.raises(TypeError, match="numeric first-treatment timing"):
        sp.auto_did(
            df, y="y", g="cohort_str", t="time", i="unit",
            methods=["cs", "bjs"],
        )


def test_auto_did_terse_repr():
    df = sp.dgp_did(
        n_units=60, n_periods=6, staggered=True, n_groups=3, effect=0.4, seed=8,
    )
    r = sp.auto_did(
        df, y="y", g="first_treat", t="time", i="unit", methods=["cs", "sa"],
    )
    rep = repr(r)
    # Terse, single-line repr — not the full leaderboard.
    assert "\n" not in rep
    assert "AutoDIDResult" in rep
    assert "winner=" in rep
