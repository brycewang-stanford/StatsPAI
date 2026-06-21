"""Coverage round-3 (final) — cluster-SCM, forward-DID, and ``_core``.

Targets the still-uncovered validation guards and small fall-through
branches of ``cluster_synth``, ``fdid`` (forward difference-in-differences,
Li 2024), and the shared simplex / ADH optimisation primitives in
``synth/_core.py``.

All pure-numpy. Assertions check real properties (finite ATT, weights on
the simplex, correct exceptions); no estimator numbers are fabricated.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.synth.cluster import cluster_synth
from statspai.synth.fdid import fdid
from statspai.synth import _core

T_TREAT = 11


def _panel(seed=0, n_donors=10, n_t=20, effect=4.0):
    rng = np.random.default_rng(seed)
    units = [f"u{i}" for i in range(n_donors)] + ["treated"]
    rows = []
    for u in units:
        base = rng.normal(0, 1)
        fe = rng.normal(0, 0.5)
        for t in range(1, n_t + 1):
            eff = effect if (u == "treated" and t >= T_TREAT) else 0.0
            rows.append(
                {
                    "unit": u,
                    "time": t,
                    "y": base + 0.2 * t + fe + eff + rng.normal(0, 0.3),
                }
            )
    return pd.DataFrame(rows)


# ===========================================================================
# cluster_synth
# ===========================================================================
def test_cluster_synth_kmeans_with_augment():
    r = cluster_synth(
        _panel(0),
        outcome="y",
        unit="unit",
        time="time",
        treated_unit="treated",
        treatment_time=T_TREAT,
        n_clusters=3,
        cluster_method="kmeans",
        augment=True,
        placebo=True,
        seed=1,
    )
    assert np.isfinite(r.estimate)


def test_cluster_synth_auto_n_clusters():
    r = cluster_synth(
        _panel(1),
        outcome="y",
        unit="unit",
        time="time",
        treated_unit="treated",
        treatment_time=T_TREAT,
        n_clusters=None,
        placebo=False,
        seed=2,
    )
    assert np.isfinite(r.estimate)


def test_cluster_synth_missing_treated_raises():
    with pytest.raises(ValueError):
        cluster_synth(
            _panel(2),
            outcome="y",
            unit="unit",
            time="time",
            treated_unit="ghost",
            treatment_time=T_TREAT,
        )


def test_cluster_synth_bad_treatment_time_raises():
    with pytest.raises(ValueError):
        cluster_synth(
            _panel(3),
            outcome="y",
            unit="unit",
            time="time",
            treated_unit="treated",
            treatment_time=9999,
        )


def test_cluster_synth_missing_covariate_raises():
    with pytest.raises((ValueError, KeyError)):
        cluster_synth(
            _panel(4),
            outcome="y",
            unit="unit",
            time="time",
            treated_unit="treated",
            treatment_time=T_TREAT,
            covariates=["nope"],
        )


def test_cluster_synth_spectral_method():
    r = cluster_synth(
        _panel(5),
        outcome="y",
        unit="unit",
        time="time",
        treated_unit="treated",
        treatment_time=T_TREAT,
        n_clusters=3,
        cluster_method="spectral",
        placebo=False,
        seed=3,
    )
    assert np.isfinite(r.estimate)


# ===========================================================================
# fdid — forward difference-in-differences
# ===========================================================================
def test_fdid_forward_selection():
    r = fdid(
        _panel(0),
        outcome="y",
        unit="unit",
        time="time",
        treated_unit="treated",
        treatment_time=T_TREAT,
        method="forward",
        placebo=True,
    )
    assert np.isfinite(r.estimate)


def test_fdid_max_donors_cap():
    r = fdid(
        _panel(1),
        outcome="y",
        unit="unit",
        time="time",
        treated_unit="treated",
        treatment_time=T_TREAT,
        method="forward",
        max_donors=3,
        placebo=False,
    )
    assert np.isfinite(r.estimate)


def test_fdid_forward_cv_method():
    # Enough pre-periods for the expanding-window CV branch.
    r = fdid(
        _panel(5, n_t=24),
        outcome="y",
        unit="unit",
        time="time",
        treated_unit="treated",
        treatment_time=13,
        method="forward_cv",
        placebo=False,
    )
    assert np.isfinite(r.estimate)


def test_fdid_best_subset_method():
    r = fdid(
        _panel(6, n_donors=6),
        outcome="y",
        unit="unit",
        time="time",
        treated_unit="treated",
        treatment_time=T_TREAT,
        method="best_subset",
        max_donors=2,
        placebo=False,
    )
    assert np.isfinite(r.estimate)


def test_fdid_missing_treated_raises():
    with pytest.raises(ValueError):
        fdid(
            _panel(2),
            outcome="y",
            unit="unit",
            time="time",
            treated_unit="ghost",
            treatment_time=T_TREAT,
        )


def test_fdid_bad_treatment_time_raises():
    with pytest.raises(ValueError):
        fdid(
            _panel(3),
            outcome="y",
            unit="unit",
            time="time",
            treated_unit="treated",
            treatment_time=9999,
        )


def test_fdid_unknown_method_raises():
    with pytest.raises(ValueError):
        fdid(
            _panel(4),
            outcome="y",
            unit="unit",
            time="time",
            treated_unit="treated",
            treatment_time=T_TREAT,
            method="not_a_method",
        )


# ===========================================================================
# _core — simplex / ADH primitives
# ===========================================================================
def test_solve_simplex_weights_single_donor():
    y = np.array([1.0, 2.0, 3.0])
    X = np.array([[1.0], [2.0], [3.0]])  # one donor
    w = _core.solve_simplex_weights(y, X)
    assert w.shape == (1,)
    assert abs(w.sum() - 1.0) < 1e-9


def test_solve_simplex_weights_zero_donors_raises():
    y = np.array([1.0, 2.0, 3.0])
    X = np.empty((3, 0))
    with pytest.raises(ValueError):
        _core.solve_simplex_weights(y, X)


def test_solve_simplex_weights_multi_donor_on_simplex():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(15, 4))
    true_w = np.array([0.4, 0.3, 0.2, 0.1])
    y = X @ true_w + rng.normal(0, 0.01, 15)
    w = _core.solve_simplex_weights(y, X)
    assert w.min() >= -1e-6
    assert abs(w.sum() - 1.0) < 1e-6


def test_standardize_predictors_shape_mismatch_raises():
    X1 = np.array([1.0, 2.0])
    X0 = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])  # K mismatch
    with pytest.raises(ValueError):
        _core.standardize_predictors(X1, X0)


def test_solve_synth_weights_adh_full_run():
    rng = np.random.default_rng(1)
    K, J, T0 = 3, 5, 12
    X0 = rng.normal(size=(K, J))
    true_w = rng.dirichlet(np.ones(J))
    X1 = X0 @ true_w + rng.normal(0, 0.05, K)
    Z0 = rng.normal(size=(T0, J))
    Z1 = Z0 @ true_w + rng.normal(0, 0.05, T0)
    out = _core.solve_synth_weights_adh(X1, X0, Z1, Z0, n_random_starts=2)
    w = np.asarray(out["w"])
    assert w.min() >= -1e-6
    assert abs(w.sum() - 1.0) < 1e-4
    assert np.isfinite(out["loss"])


def test_solve_synth_weights_adh_shape_validation():
    K, J, T0 = 2, 4, 8
    X0 = np.zeros((K, J))
    X1 = np.zeros(K)
    Z1 = np.zeros(T0)
    # Z0 with wrong donor count -> ValueError (L320)
    with pytest.raises(ValueError):
        _core.solve_synth_weights_adh(X1, X0, Z1, np.zeros((T0, J + 1)))
    # Z0 with wrong pre-period count -> ValueError (L324)
    with pytest.raises(ValueError):
        _core.solve_synth_weights_adh(X1, X0, Z1, np.zeros((T0 + 1, J)))


def test_solve_synth_weights_adh_no_standardize():
    rng = np.random.default_rng(2)
    K, J, T0 = 2, 3, 10
    X0 = rng.normal(size=(K, J))
    X1 = X0.mean(axis=1)
    Z0 = rng.normal(size=(T0, J))
    Z1 = Z0.mean(axis=1)
    out = _core.solve_synth_weights_adh(
        X1,
        X0,
        Z1,
        Z0,
        standardize=False,
        n_random_starts=1,
    )
    assert np.all(np.asarray(out["scale"]) == 1.0)
