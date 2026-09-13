"""Placebo p-value convention and nested-solver fast path for sp.synth.

The permutation p-value ranks the treated unit together with its J
placebos and divides by J+1 (Abadie, Diamond & Hainmueller 2010). Before
this fix every placebo-based synth estimator computed
``mean(placebo >= treated)`` (denominator J, treated unit not counted)
and floored the result at ``1/(J+1)``, so a treated unit ranked third of
39 reported 2/38 = 0.0526 instead of 3/39 = 0.0769.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import statspai as sp
from statspai.synth import _core
from statspai.synth._core import placebo_rank_pvalue

# ---------------------------------------------------------------------------
# Helper: exact rank formula
# ---------------------------------------------------------------------------


def test_rank_pvalue_counts_treated_unit_in_numerator_and_denominator():
    placebos = np.arange(38, dtype=float)  # 0..37
    # Treated most extreme -> rank 1 of 39.
    assert placebo_rank_pvalue(100.0, placebos) == pytest.approx(1 / 39)
    # Two placebos (36, 37) at least as extreme -> rank 3 of 39, not 2/38.
    assert placebo_rank_pvalue(35.5, placebos) == pytest.approx(3 / 39)
    # Treated least extreme -> p = 1.
    assert placebo_rank_pvalue(-1.0, placebos) == pytest.approx(1.0)


def test_rank_pvalue_ties_count_against_treated():
    assert placebo_rank_pvalue(2.0, [1.0, 2.0, 2.0, 3.0]) == pytest.approx(4 / 5)


def test_rank_pvalue_edge_cases():
    assert np.isnan(placebo_rank_pvalue(1.0, []))
    assert np.isnan(placebo_rank_pvalue(np.nan, [1.0, 2.0]))
    # NaN placebos are unusable: dropped from numerator and denominator.
    assert placebo_rank_pvalue(1.5, [np.nan, 1.0, 2.0]) == pytest.approx(2 / 3)
    # Perfect pre-fit gives an infinite ratio: most extreme.
    assert placebo_rank_pvalue(np.inf, [1.0, 5.0, 0.0]) == pytest.approx(1 / 4)


# ---------------------------------------------------------------------------
# Classic SCM on California Proposition 99 (the README example)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def prop99():
    return sp.datasets.california_prop99()


@pytest.fixture(scope="module")
def readme_fit(prop99):
    return sp.synth(
        data=prop99,
        outcome="cigsale",
        unit="state",
        time="year",
        treated_unit="California",
        treatment_time=1989,
    )


def test_scm_pvalue_is_treated_rank_over_units(readme_fit):
    mi = readme_fit.model_info
    ratios = np.asarray(mi["placebo_ratios"])
    rank = 1 + int(np.sum(ratios >= mi["treated_ratio"]))
    n_units = len(ratios) + 1
    assert n_units == 39
    assert readme_fit.pvalue == pytest.approx(rank / n_units, abs=1e-15)
    # California ranks third of 39 under the outcome-only default spec.
    assert rank == 3
    assert readme_fit.pvalue == pytest.approx(3 / 39, abs=1e-15)
    assert mi["placebo_failures"] == []


def test_scm_readme_numbers(readme_fit):
    """Pins the numbers printed in README.md / README_CN.md."""
    assert readme_fit.estimate == pytest.approx(-19.760529, abs=5e-7)
    assert readme_fit.se == pytest.approx(11.233914, abs=5e-7)
    w = dict(
        zip(
            readme_fit.model_info["weights"]["unit"],
            readme_fit.model_info["weights"]["weight"],
        )
    )
    expected = {
        "Utah": 0.3768,
        "Montana": 0.2831,
        "Nevada": 0.1881,
        "Connecticut": 0.0690,
        "New Hampshire": 0.0439,
        "Colorado": 0.0391,
    }
    assert set(w) == set(expected)
    for unit, weight in expected.items():
        assert w[unit] == pytest.approx(weight, abs=5e-5)


def test_scm_failed_placebo_is_reported_not_swallowed(prop99, monkeypatch):
    sc = sp.SyntheticControl(
        prop99,
        outcome="cigsale",
        unit="state",
        time="year",
        treated_unit="California",
        treatment_time=1989,
    )
    real_solve = sc._solve_weights
    calls = {"n": 0}

    def flaky(*args, **kwargs):
        calls["n"] += 1
        if calls["n"] == 3:  # 1 = main fit, 2.. = placebos
            raise np.linalg.LinAlgError("synthetic failure")
        return real_solve(*args, **kwargs)

    monkeypatch.setattr(sc, "_solve_weights", flaky)
    with pytest.warns(RuntimeWarning, match="1 of 38 in-space placebo"):
        res = sc.fit()
    mi = res.model_info
    assert mi["n_placebos"] == 37
    assert len(mi["placebo_failures"]) == 1
    assert mi["placebo_failures"][0]["error_type"] == "LinAlgError"
    ratios = np.asarray(mi["placebo_ratios"])
    rank = 1 + int(np.sum(ratios >= mi["treated_ratio"]))
    assert res.pvalue == pytest.approx(rank / 38, abs=1e-15)


# ---------------------------------------------------------------------------
# Parallel placebo loop (n_jobs)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def small_panel(prop99):
    keep = ["California"] + sorted(
        s for s in prop99["state"].unique() if s != "California"
    )[:9]
    return prop99[prop99["state"].isin(keep)]


def _fit_small(panel, **kwargs):
    return sp.SyntheticControl(
        panel,
        outcome="cigsale",
        unit="state",
        time="year",
        treated_unit="California",
        treatment_time=1989,
        **kwargs,
    ).fit()


def test_parallel_placebo_loop_is_bit_identical_to_serial(small_panel):
    # Nested V-W (covariates) is the path n_jobs exists for.
    kw = dict(
        covariates=["lnincome", "retprice", "age15to24", "beer"],
        n_random_starts=0,
    )
    serial = _fit_small(small_panel, **kw)
    parallel = _fit_small(small_panel, n_jobs=2, **kw)
    ms, mp_ = serial.model_info, parallel.model_info
    assert ms["placebo_n_jobs"] == 1
    assert mp_["placebo_n_jobs"] == 2
    assert ms["placebo_parallel_fallback"] is None
    assert mp_["placebo_parallel_fallback"] is None
    # Exact equality, not approx: same code, same inputs, same order.
    assert parallel.pvalue == serial.pvalue
    assert parallel.se == serial.se
    assert mp_["placebo_units"] == ms["placebo_units"]
    assert mp_["placebo_ratios"] == ms["placebo_ratios"]
    assert mp_["placebo_atts"] == ms["placebo_atts"]
    np.testing.assert_array_equal(mp_["placebo_gaps"], ms["placebo_gaps"])


def test_sp_synth_forwards_n_jobs(small_panel):
    kw = dict(
        data=small_panel,
        outcome="cigsale",
        unit="state",
        time="year",
        treated_unit="California",
        treatment_time=1989,
    )
    serial = sp.synth(**kw)
    parallel = sp.synth(**kw, n_jobs=2)
    assert parallel.model_info["placebo_n_jobs"] == 2
    assert parallel.pvalue == serial.pvalue
    assert parallel.model_info["placebo_ratios"] == serial.model_info["placebo_ratios"]


def test_unstartable_process_pool_falls_back_to_serial_loudly(small_panel, monkeypatch):
    from statspai.synth import scm as scm_mod

    def no_processes(*args, **kwargs):
        raise OSError("process creation not permitted")

    monkeypatch.setattr(scm_mod, "_map_in_processes", no_processes)
    with pytest.warns(RuntimeWarning, match="fell back to serial"):
        res = _fit_small(small_panel, n_jobs=4)
    assert res.model_info["placebo_n_jobs"] == 1
    assert "OSError" in res.model_info["placebo_parallel_fallback"]
    assert res.pvalue == _fit_small(small_panel).pvalue


@pytest.mark.parametrize("bad", [0, -2, 1.5, True, "2"])
def test_n_jobs_rejects_invalid_values(small_panel, bad):
    from statspai.exceptions import MethodIncompatibility

    with pytest.raises(MethodIncompatibility, match="n_jobs"):
        sp.SyntheticControl(
            small_panel,
            outcome="cigsale",
            unit="state",
            time="year",
            treated_unit="California",
            treatment_time=1989,
            n_jobs=bad,
        )


def test_n_jobs_minus_one_uses_every_cpu(small_panel):
    import os

    sc = sp.SyntheticControl(
        small_panel,
        outcome="cigsale",
        unit="state",
        time="year",
        treated_unit="California",
        treatment_time=1989,
        n_jobs=-1,
    )
    assert sc.n_jobs == max(1, os.cpu_count() or 1)


# ---------------------------------------------------------------------------
# perfect_fit: treated predictors inside the donors' convex hull
# ---------------------------------------------------------------------------


def _in_hull_panel(seed: int = 0):
    """Treated covariates AND pre-period outcomes are the same convex
    combination ``w_true`` of the donors, so the treated unit is in the
    covariate hull and ``w_true`` is the unique zero-loss synthetic control
    (T0 = 20 > J = 8, donor outcomes in general position)."""
    import pandas as pd

    rng = np.random.default_rng(seed)
    J, T, T0 = 8, 26, 20
    w_true = np.array([0.5, 0.3, 0.2, 0.0, 0.0, 0.0, 0.0, 0.0])
    Y0 = rng.normal(size=(T, J)).cumsum(axis=0) + 10.0
    C0 = rng.normal(size=(2, J))
    y1 = Y0 @ w_true
    y1[T0:] += 3.0  # post-treatment effect
    c1 = C0 @ w_true
    rows = []
    for j in range(J + 1):
        name = "treated" if j == J else f"d{j}"
        ys = y1 if j == J else Y0[:, j]
        cs = c1 if j == J else C0[:, j]
        for t in range(T):
            rows.append({"unit": name, "time": t, "y": ys[t], "c1": cs[0], "c2": cs[1]})
    return pd.DataFrame(rows), w_true, T0


def _fit_in_hull(**kwargs):
    df, w_true, T0 = _in_hull_panel()
    res = sp.SyntheticControl(
        df,
        outcome="y",
        unit="unit",
        time="time",
        treated_unit="treated",
        treatment_time=T0,
        covariates=["c1", "c2"],
        **kwargs,
    ).fit(placebo=False)
    return res, w_true


def test_default_is_the_adh_search_and_flags_the_hull():
    res, _ = _fit_in_hull(n_random_starts=0)
    mi = res.model_info
    assert mi["perfect_fit"] == "legacy"
    assert mi["in_predictor_hull"] is True
    assert mi["v_identified"] is None
    assert mi["solver_best_start"] in {"equal", "regression"}


def test_exact_balance_recovers_the_exact_synthetic_control():
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # the certified path must not warn
        res, w_true = _fit_in_hull(perfect_fit="exact_balance")
    mi = res.model_info
    assert mi["in_predictor_hull"] is True
    assert mi["v_identified"] is False
    assert mi["solver_best_start"] == "exact_balance"
    assert mi["converged"] is True
    w = dict(zip(mi["weights"]["unit"], mi["weights"]["weight"]))
    # atol 1e-6: trust-constr stops at gtol 1e-12 on a well-conditioned QP.
    for j, wj in enumerate(w_true):
        assert w.get(f"d{j}", 0.0) == pytest.approx(wj, abs=1e-6)
    assert mi["pre_treatment_rmse"] == pytest.approx(0.0, abs=1e-5)
    assert res.estimate == pytest.approx(3.0, abs=1e-5)


def test_exact_balance_is_off_under_ridge_penalization():
    res, _ = _fit_in_hull(
        perfect_fit="exact_balance", penalization=0.01, n_random_starts=0
    )
    assert res.model_info["in_predictor_hull"] is True
    assert res.model_info["solver_best_start"] in {"equal", "regression"}


def test_out_of_hull_fit_is_flagged_and_untouched(prop99):
    """California is outside the four-covariate hull, so ``exact_balance``
    cannot apply (its optimum is pinned in the nested-solver test below)."""
    sc = sp.SyntheticControl(
        prop99,
        outcome="cigsale",
        unit="state",
        time="year",
        treated_unit="California",
        treatment_time=1989,
        covariates=["lnincome", "retprice", "age15to24", "beer"],
    )
    X1s, X0s, _ = _core.standardize_predictors(sc.X_treated, sc.X_donors)
    assert _core._hull_feasibility(X1s, X0s) == (False, None)


def test_perfect_fit_rejects_unknown_rule():
    from statspai.exceptions import MethodIncompatibility

    with pytest.raises(MethodIncompatibility, match="perfect_fit"):
        _fit_in_hull(perfect_fit="outcome_mspe")


# ---------------------------------------------------------------------------
# Other placebo-based estimators share the helper
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("method", ["demeaned", "sparse", "mc", "gsynth"])
def test_variant_pvalues_live_on_the_rank_grid(prop99, method):
    # 15 states keeps the placebo loops fast; the rank grid is over 15.
    keep = ["California"] + sorted(
        s for s in prop99["state"].unique() if s != "California"
    )[:14]
    panel = prop99[prop99["state"].isin(keep)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sp.synth(
            data=panel,
            outcome="cigsale",
            unit="state",
            time="year",
            treated_unit="California",
            treatment_time=1989,
            method=method,
        )
    if not np.isfinite(res.pvalue):
        pytest.skip(f"{method} produced no placebo p-value on this panel")
    n_donors = panel["state"].nunique() - 1
    scaled = res.pvalue * (n_donors + 1)
    # With no dropped placebos the p-value is an integer rank over J+1.
    assert scaled == pytest.approx(round(scaled), abs=1e-9)
    assert 1 <= round(scaled) <= n_donors + 1


# ---------------------------------------------------------------------------
# Nested V-W solver: the faster inner QP lands on the same optimum
# ---------------------------------------------------------------------------


def test_nested_solver_with_covariates_recovers_pre_speedup_optimum(prop99):
    """Abadie's four covariates, main fit only.

    Reference values come from the solver before the exact-Jacobian speed-up
    (outer loss 744.1002662498, weights Colorado 0.5079 / Connecticut
    0.4921, identical across all six starts). Supplying the exact
    adding-up Jacobian to SLSQP must land on the same optimum.
    """
    sc = sp.SyntheticControl(
        prop99,
        outcome="cigsale",
        unit="state",
        time="year",
        treated_unit="California",
        treatment_time=1989,
        covariates=["lnincome", "retprice", "age15to24", "beer"],
    )
    out = _core.solve_synth_weights_adh(
        sc.X_treated,
        sc.X_donors,
        sc.Y_treated[sc.pre_mask],
        sc.Y_donors[sc.pre_mask],
    )
    # rtol 1e-9: the loss agrees with the cold-start run to its printed
    # 10 decimals; any genuine change of optimum moves it by >> 1e-6.
    assert out["loss"] == pytest.approx(744.1002662498, rel=1e-9)
    w = dict(zip(sc.donor_units, out["w"]))
    active = {u: x for u, x in w.items() if x > 1e-4}
    assert set(active) == {"Colorado", "Connecticut"}
    # abs 5e-5: the reference weights are known to 4 decimals.
    assert active["Colorado"] == pytest.approx(0.5079, abs=5e-5)
    assert active["Connecticut"] == pytest.approx(0.4921, abs=5e-5)
