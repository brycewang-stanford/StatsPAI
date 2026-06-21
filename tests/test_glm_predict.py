"""Prediction contract tests for formula-backed GLM models."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from statspai.exceptions import DataInsufficient, MethodIncompatibility
from statspai.regression.glm import GLMRegression, glm


@pytest.fixture
def poisson_data() -> pd.DataFrame:
    rng = np.random.default_rng(20260617)
    n = 240
    x = rng.normal(size=n)
    z = rng.normal(size=n)
    g = np.where(np.arange(n) % 2 == 0, "a", "b")
    eta = 0.15 + 0.35 * x - 0.25 * z + 0.4 * (g == "b")
    y = rng.poisson(np.exp(eta))
    return pd.DataFrame({"y": y, "x": x, "z": z, "g": g})


def test_glm_predict_out_of_sample_without_response(poisson_data):
    model = GLMRegression(
        formula="y ~ x + z",
        data=poisson_data,
        family="poisson",
    )
    result = model.fit()

    new_data = poisson_data[["x", "z"]].head(12)
    eta = model.predict(new_data, type="link")
    expected_eta = (
        result.params["Intercept"]
        + result.params["x"] * new_data["x"].to_numpy()
        + result.params["z"] * new_data["z"].to_numpy()
    )
    np.testing.assert_allclose(eta, expected_eta)

    response = model.predict(new_data, type="response")
    variance = model.predict(new_data, type="variance")
    np.testing.assert_allclose(response, np.exp(expected_eta))
    np.testing.assert_allclose(variance, response)
    np.testing.assert_allclose(
        model.predict(new_data, type="link", offset=0.5),
        expected_eta + 0.5,
    )


def test_glm_predict_reuses_patsy_design_info_for_categories(poisson_data):
    model = GLMRegression(
        formula="y ~ x + C(g)",
        data=poisson_data,
        family="poisson",
    )
    result = model.fit()

    new_data = pd.DataFrame({"x": [0.0, 1.0, -1.0], "g": ["a", "b", "a"]})
    eta = model.predict(new_data, type="link")
    g_b = (new_data["g"] == "b").astype(float).to_numpy()
    expected_eta = (
        result.params["Intercept"]
        + result.params["x"] * new_data["x"].to_numpy()
        + result.params["C(g)[T.b]"] * g_b
    )
    np.testing.assert_allclose(eta, expected_eta)


def test_glm_predict_unseen_category_raises_taxonomy(poisson_data):
    model = GLMRegression(
        formula="y ~ x + C(g)",
        data=poisson_data,
        family="poisson",
    )
    model.fit()

    new_data = pd.DataFrame({"x": [0.0], "g": ["c"]})
    with pytest.raises(
        MethodIncompatibility, match="prediction design matrix"
    ) as excinfo:
        model.predict(new_data)
    assert excinfo.value.diagnostics["formula"] == "y ~ x + C(g)"


def test_glm_predict_errors_use_exception_taxonomy(poisson_data):
    unfitted = GLMRegression(
        formula="y ~ x + z",
        data=poisson_data,
        family="poisson",
    )
    with pytest.raises(MethodIncompatibility, match="fitted before prediction"):
        unfitted.predict(poisson_data[["x", "z"]].head(2))

    model = GLMRegression(
        formula="y ~ x + z",
        data=poisson_data,
        family="poisson",
    )
    model.fit()
    new_data = poisson_data[["x", "z"]].head(2)

    with pytest.raises(MethodIncompatibility, match="`type`"):
        model.predict(new_data, type="bogus")

    with pytest.raises(MethodIncompatibility, match="pandas DataFrame"):
        model.predict(np.ones((2, 2)))

    with pytest.raises(MethodIncompatibility, match="prediction design matrix"):
        model.predict(poisson_data[["x"]].head(2))

    with pytest.raises(MethodIncompatibility, match="offset length"):
        model.predict(new_data, offset=np.ones(3))

    x = np.column_stack(
        [
            np.ones(len(poisson_data)),
            poisson_data["x"].to_numpy(),
            poisson_data["z"].to_numpy(),
        ]
    )
    raw = GLMRegression(
        y=poisson_data["y"].to_numpy(),
        X=x,
        family="poisson",
    )
    raw.fit()
    with pytest.raises(MethodIncompatibility, match="fit with a formula"):
        raw.predict(new_data)


def test_glm_fit_aligns_auxiliary_columns_after_formula_dropna(poisson_data):
    dirty = poisson_data.copy()
    dirty["w"] = np.linspace(0.5, 2.0, len(dirty))
    dirty["off"] = np.linspace(-0.1, 0.1, len(dirty))
    dirty["cluster"] = np.arange(len(dirty)) % 12
    dirty.loc[5, "x"] = np.nan

    dirty_model = GLMRegression(
        formula="y ~ x",
        data=dirty,
        family="poisson",
    )
    dirty_result = dirty_model.fit(
        weights="w",
        offset="off",
        cluster="cluster",
    )

    clean = dirty.dropna(subset=["y", "x"])
    clean_model = GLMRegression(
        formula="y ~ x",
        data=clean,
        family="poisson",
    )
    clean_result = clean_model.fit(
        weights="w",
        offset="off",
        cluster="cluster",
    )

    assert dirty_result.data_info["nobs"] == len(clean)
    np.testing.assert_allclose(dirty_result.params, clean_result.params)
    np.testing.assert_allclose(
        dirty_result.std_errors,
        clean_result.std_errors,
    )


def test_glm_fit_rejects_bad_auxiliary_columns(poisson_data):
    df = poisson_data.copy()
    df["w"] = 1.0
    df["off"] = 0.0
    df["exposure"] = 1.0
    df["cluster"] = np.arange(len(df)) % 4

    with pytest.raises(MethodIncompatibility, match="weights column"):
        GLMRegression(
            formula="y ~ x",
            data=df,
            family="poisson",
        ).fit(weights="missing_weight")

    bad_weight = df.copy()
    bad_weight.loc[0, "w"] = np.nan
    with pytest.raises(MethodIncompatibility, match="weights contains missing"):
        GLMRegression(
            formula="y ~ x",
            data=bad_weight,
            family="poisson",
        ).fit(weights="w")

    bad_exposure = df.copy()
    bad_exposure.loc[0, "exposure"] = 0.0
    with pytest.raises(MethodIncompatibility, match="strictly positive"):
        GLMRegression(
            formula="y ~ x",
            data=bad_exposure,
            family="poisson",
        ).fit(exposure="exposure")

    one_cluster = df.copy()
    one_cluster["cluster"] = "only"
    with pytest.raises(MethodIncompatibility, match="at least two clusters"):
        GLMRegression(
            formula="y ~ x",
            data=one_cluster,
            family="poisson",
        ).fit(cluster="cluster")

    x = np.column_stack([np.ones(len(df)), df["x"].to_numpy()])
    raw = GLMRegression(y=df["y"].to_numpy(), X=x, family="poisson")
    with pytest.raises(MethodIncompatibility, match="weights requires data"):
        raw.fit(weights="w")


def test_glm_entry_errors_use_exception_taxonomy(poisson_data):
    with pytest.raises(MethodIncompatibility, match="pandas DataFrame"):
        GLMRegression(formula="y ~ x", data=[1, 2, 3], family="poisson")

    with pytest.raises(MethodIncompatibility, match="Unknown family"):
        GLMRegression(formula="y ~ x", data=poisson_data, family="bad")

    with pytest.raises(MethodIncompatibility, match="Unknown link"):
        GLMRegression(formula="y ~ x", data=poisson_data, family="poisson", link="bad")

    with pytest.raises(MethodIncompatibility, match="Must provide either"):
        GLMRegression(family="poisson").fit()

    with pytest.raises(MethodIncompatibility, match="Unknown robust option"):
        GLMRegression(
            formula="y ~ x",
            data=poisson_data,
            family="poisson",
        ).fit(robust="hc9")

    with pytest.raises(MethodIncompatibility, match="maxiter"):
        GLMRegression(
            formula="y ~ x",
            data=poisson_data,
            family="poisson",
        ).fit(maxiter=0)

    empty = poisson_data.iloc[0:0].copy()
    with pytest.raises(DataInsufficient, match="No rows remain"):
        GLMRegression(formula="y ~ x", data=empty, family="poisson").fit()


def test_public_glm_y_x_contract_and_scalar_x(poisson_data):
    res = glm(y="y", x="x", data=poisson_data, family="poisson")
    assert list(res.params.index) == ["Intercept", "x"]

    with pytest.raises(MethodIncompatibility, match="Must provide"):
        glm(family="poisson")

    with pytest.raises(MethodIncompatibility, match="x"):
        glm(y="y", x=[1], data=poisson_data, family="poisson")
