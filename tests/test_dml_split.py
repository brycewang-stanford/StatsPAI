"""Tests for the split-out DoubleMLPLR/IRM/PLIV/IIVM classes."""

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.dml import DoubleMLPLR, DoubleMLIRM, DoubleMLPLIV, DoubleMLIIVM
from statspai.core.results import CausalResult
from statspai.exceptions import MethodIncompatibility


def test_plr_class_matches_dispatcher():
    rng = np.random.default_rng(42)
    n = 1200
    X1 = rng.normal(0, 1, n)
    X2 = rng.normal(0, 1, n)
    D = np.cos(X1) + X2 + rng.normal(0, 0.5, n)
    Y = 2.0 * D + np.sin(X1) + X2**2 + rng.normal(0, 0.5, n)
    df = pd.DataFrame({"y": Y, "d": D, "x1": X1, "x2": X2})

    r_dispatch = sp.dml(df, y="y", treat="d", covariates=["x1", "x2"])
    r_class = DoubleMLPLR(data=df, y="y", treat="d", covariates=["x1", "x2"]).fit()
    assert isinstance(r_class, CausalResult)
    # Same seed path → identical estimate
    assert abs(r_dispatch.estimate - r_class.estimate) < 1e-10


def test_plr_accepts_explicit_fold_indices_matching_kfold():
    from sklearn.model_selection import KFold

    rng = np.random.default_rng(43)
    n = 600
    X1 = rng.normal(size=n)
    X2 = rng.normal(size=n)
    D = 0.6 * X1 - 0.4 * X2 + rng.normal(size=n)
    Y = 1.25 * D + 0.5 * X1 + rng.normal(size=n)
    df = pd.DataFrame({"y": Y, "d": D, "x1": X1, "x2": X2})

    n_folds = 5
    seed = 321
    fold_indices = np.empty(n, dtype=int)
    for fold, (_, test_idx) in enumerate(
        KFold(n_splits=n_folds, shuffle=True, random_state=seed).split(df)
    ):
        fold_indices[test_idx] = fold

    default = sp.dml(
        df,
        y="y",
        d="d",
        X=["x1", "x2"],
        model_y="linear",
        model_d="linear",
        n_folds=n_folds,
        random_state=seed,
    )
    explicit = sp.dml(
        df,
        y="y",
        d="d",
        X=["x1", "x2"],
        model_y="linear",
        model_d="linear",
        n_folds=n_folds,
        random_state=999,
        fold_indices=fold_indices,
    )

    assert explicit.model_info["fold_source"] == "user"
    assert default.model_info["fold_source"] == "kfold"
    assert explicit.estimate == pytest.approx(default.estimate, abs=1e-14)
    assert explicit.se == pytest.approx(default.se, abs=1e-14)


def test_plr_explicit_fold_indices_validate_shape():
    df = pd.DataFrame(
        {
            "y": np.arange(10.0),
            "d": np.arange(10.0),
            "x": np.arange(10.0),
        }
    )
    with pytest.raises(ValueError, match="fold_indices"):
        sp.dml(
            df,
            y="y",
            d="d",
            X=["x"],
            model_y="linear",
            model_d="linear",
            fold_indices=np.arange(9),
        )


def test_explicit_fold_indices_are_plr_only_for_now():
    df = pd.DataFrame(
        {
            "y": [0, 1, 0, 1, 0, 1, 0, 1],
            "d": [0, 1, 0, 1, 0, 1, 0, 1],
            "x": np.arange(8.0),
        }
    )
    with pytest.raises(MethodIncompatibility, match="model=.plr. only"):
        sp.dml(
            df,
            y="y",
            d="d",
            X=["x"],
            model="irm",
            fold_indices=np.array([0, 0, 1, 1, 2, 2, 3, 3]),
            n_folds=4,
        )


def test_irm_class_direct():
    rng = np.random.default_rng(42)
    n = 1500
    X1 = rng.normal(0, 1, n)
    X2 = rng.normal(0, 1, n)
    logit = 0.5 * X1 + X2
    D = rng.binomial(1, 1 / (1 + np.exp(-logit)), n).astype(float)
    Y = 3.0 * D + X1 + X2**2 + rng.normal(0, 0.5, n)
    df = pd.DataFrame({"y": Y, "d": D, "x1": X1, "x2": X2})

    r = DoubleMLIRM(data=df, y="y", treat="d", covariates=["x1", "x2"]).fit()
    assert abs(r.estimate - 3.0) < 0.4
    assert r.estimand == "ATE"
    assert r.model_info["dml_model"] == "IRM"


def test_iivm_class_direct():
    rng = np.random.default_rng(42)
    n = 2500
    X = rng.normal(0, 1, n)
    Z = rng.binomial(1, 0.5, n).astype(float)
    u = rng.uniform(0, 1, n)
    D = np.where(u < 0.7, Z, 1.0).astype(float)
    Y = 1.5 * D + 0.5 * X + rng.normal(0, 0.3, n)
    df = pd.DataFrame({"y": Y, "d": D, "z": Z, "x": X})

    r = DoubleMLIIVM(
        data=df,
        y="y",
        treat="d",
        covariates=["x"],
        instrument="z",
    ).fit()
    assert r.estimand == "LATE"
    assert abs(r.estimate - 1.5) < 0.3


def test_legacy_DoubleML_still_works():
    """Back-compat: the old DoubleML class still accepts model= strings."""
    rng = np.random.default_rng(42)
    n = 1000
    X = rng.normal(0, 1, n)
    D = np.cos(X) + rng.normal(0, 0.5, n)
    Y = 2.0 * D + np.sin(X) + rng.normal(0, 0.5, n)
    df = pd.DataFrame({"y": Y, "d": D, "x": X})

    old = sp.DoubleML(data=df, y="y", treat="d", covariates=["x"], model="plr")
    result = old.fit()
    assert isinstance(result, CausalResult)
    assert old.model == "plr"
    # Legacy attributes should still resolve
    assert old.n_folds == 5
    assert old.covariates == ["x"]


def test_pliv_rejects_list_of_multiple_instruments():
    df = pd.DataFrame(
        {
            "y": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
            "d": [0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0],
            "z1": [1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0],
            "z2": [0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.1, 1.2, 1.3, 1.4],
            "x": [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        }
    )
    with pytest.raises(ValueError, match="single scalar instrument"):
        DoubleMLPLIV(
            data=df,
            y="y",
            treat="d",
            covariates=["x"],
            instrument=["z1", "z2"],
        )


def test_pliv_accepts_sample_weight_array():
    rng = np.random.default_rng(123)
    n = 2200
    X1 = rng.normal(size=n)
    X2 = rng.normal(size=n)
    Z = 0.7 * X1 + rng.normal(size=n)
    u = rng.normal(size=n)
    D = 0.6 * Z + 0.4 * X2 + 0.4 * u + rng.normal(scale=0.3, size=n)
    Y = 1.5 * D + np.sin(X1) + 0.5 * X2**2 + 1.2 * u + rng.normal(size=n)
    weights = rng.uniform(0.3, 2.5, size=n)
    df = pd.DataFrame(
        {
            "y": Y,
            "d": D,
            "z": Z,
            "x1": X1,
            "x2": X2,
        }
    )

    r = DoubleMLPLIV(
        data=df,
        y="y",
        treat="d",
        covariates=["x1", "x2"],
        instrument="z",
        sample_weight=weights,
    ).fit()
    assert np.isfinite(r.estimate)
    assert np.isfinite(r.se) and r.se > 0
    assert abs(r.estimate - 1.5) < 0.35
    assert r.model_info["diagnostics"]["weighted"] is True


def test_pliv_accepts_sample_weight_column_name():
    rng = np.random.default_rng(321)
    n = 1800
    X = rng.normal(size=n)
    Z = 0.8 * X + rng.normal(size=n)
    u = rng.normal(size=n)
    D = 0.5 * Z + 0.3 * X + 0.5 * u + rng.normal(scale=0.3, size=n)
    Y = 1.4 * D + np.cos(X) + u + rng.normal(scale=0.4, size=n)
    df = pd.DataFrame(
        {
            "y": Y,
            "d": D,
            "z": Z,
            "x": X,
            "w": rng.uniform(0.5, 1.8, size=n),
        }
    )

    r = sp.dml(
        df,
        y="y",
        treat="d",
        covariates=["x"],
        model="pliv",
        instrument="z",
        sample_weight="w",
    )
    assert np.isfinite(r.estimate)
    assert np.isfinite(r.se) and r.se > 0
    assert abs(r.estimate - 1.4) < 0.4
    assert r.model_info["diagnostics"]["weighted"] is True


def test_pliv_weight_scale_invariant():
    rng = np.random.default_rng(777)
    n = 1600
    X = rng.normal(size=(n, 2))
    Z = 0.9 * X[:, 0] + rng.normal(size=n)
    u = rng.normal(size=n)
    D = 0.7 * Z + 0.2 * X[:, 1] + 0.4 * u + rng.normal(scale=0.2, size=n)
    Y = 1.2 * D + X[:, 0] - 0.3 * X[:, 1] + u + rng.normal(scale=0.3, size=n)
    w = rng.uniform(0.4, 1.9, size=n)
    df = pd.DataFrame(
        {
            "y": Y,
            "d": D,
            "z": Z,
            "x1": X[:, 0],
            "x2": X[:, 1],
        }
    )

    a = sp.dml(
        df,
        y="y",
        treat="d",
        covariates=["x1", "x2"],
        model="pliv",
        instrument="z",
        sample_weight=w,
        ml_g="linear",
        ml_m="linear",
        ml_r="linear",
    )
    b = sp.dml(
        df,
        y="y",
        treat="d",
        covariates=["x1", "x2"],
        model="pliv",
        instrument="z",
        sample_weight=10.0 * w,
        ml_g="linear",
        ml_m="linear",
        ml_r="linear",
    )
    assert abs(a.estimate - b.estimate) < 1e-10
    assert abs(a.se - b.se) < 1e-10
