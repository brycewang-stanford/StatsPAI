"""Panel negative-binomial regression guards."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import MethodIncompatibility


def _panel_nb_data(seed: int = 20260507, n_g: int = 24, n_t: int = 6) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    gid = np.repeat(np.arange(n_g), n_t)
    group_effect = np.linspace(-0.8, 0.8, n_g)
    x = np.repeat(group_effect, n_t) + rng.normal(scale=0.35, size=n_g * n_t)
    eta = 0.15 + 0.35 * x + group_effect[gid]
    mu = np.exp(eta)
    alpha = 0.6
    size = 1.0 / alpha
    y = rng.negative_binomial(size, size / (size + mu))
    return pd.DataFrame({"y": y, "x": x, "id": gid, "t": np.tile(np.arange(n_t), n_g)})


def test_nbreg_formula_fixed_effects_are_explicit_not_ignored():
    df = _panel_nb_data()

    result = sp.nbreg("y ~ x | id", data=df, maxiter=60, tol=1e-6)

    assert result.model_info["fixed_effects"] == ["id"]
    assert result.model_info["n_fe_levels"] == {"id": df["id"].nunique()}
    assert result.model_info["n_fe_params"] == df["id"].nunique() - 1
    assert any(name.startswith("C(id)[") for name in result.params.index)


def test_xtnbreg_ufe_wraps_nbreg_with_entity_metadata():
    """model='ufe' is the unconditional dummy-variable NB-2 (the pre-1.28.x
    meaning of model='fe'); model='fe' is now Stata's HHG conditional FE."""
    df = _panel_nb_data()

    result = sp.xtnbreg(
        "y ~ x",
        data=df,
        entity="id",
        model="ufe",
        maxiter=60,
        tol=1e-6,
    )

    assert result.model_info["panel_model"] == "unconditional_fixed_effects"
    assert result.model_info["entity"] == "id"
    assert result.model_info["cluster"] == "id"
    assert result.model_info["fixed_effects"] == ["id"]


def test_xtnbreg_fe_is_the_conditional_hhg_model():
    df = _panel_nb_data()
    result = sp.xtnbreg("y ~ x", data=df, entity="id", model="fe")
    assert result.model_info["panel_model"] == "fixed_effects"
    assert result.model_info["stata_equivalent"] == "xtnbreg, fe"
    assert list(result.params.index) == ["x", "_cons"]


def test_xtnbreg_allows_formula_panel_part_without_entity_argument():
    df = _panel_nb_data(n_g=12, n_t=5)

    result = sp.xtnbreg("y ~ 1 | id", data=df, model="ufe", maxiter=40, tol=1e-6)

    assert result.model_info["entity"] == "id"
    assert result.model_info["fixed_effects"] == ["id"]
    np.testing.assert_allclose(
        result.model_info["n_fe_params"],
        df["id"].nunique() - 1,
    )
    fe = sp.xtnbreg("y ~ x | id", data=df, model="fe")
    assert fe.model_info["entity"] == "id"


def test_nbreg_input_errors_use_exception_taxonomy():
    df = _panel_nb_data()

    with pytest.raises(MethodIncompatibility, match="Must provide") as excinfo:
        sp.nbreg(data=df)
    assert isinstance(excinfo.value, ValueError)

    with pytest.raises(MethodIncompatibility, match="missing column"):
        sp.nbreg("y ~ missing", data=df, maxiter=5)

    with pytest.raises(MethodIncompatibility, match="fixed effects"):
        sp.nbreg("y ~ x | missing_id", data=df, maxiter=5)

    missing_fe = df.copy()
    missing_fe.loc[0, "id"] = np.nan
    with pytest.raises(MethodIncompatibility, match="fixed-effect column"):
        sp.nbreg("y ~ x | id", data=missing_fe, maxiter=5)


def test_count_models_reject_nonpositive_exposure_taxonomy():
    df = _panel_nb_data()
    df["exposure"] = 1.0
    df.loc[0, "exposure"] = 0.0

    with pytest.raises(MethodIncompatibility, match="exposure"):
        sp.nbreg("y ~ x", data=df, exposure="exposure", maxiter=5)

    with pytest.raises(MethodIncompatibility, match="exposure"):
        sp.poisson("y ~ x", data=df, exposure="exposure", maxiter=5)


def test_xtnbreg_configuration_errors_use_exception_taxonomy():
    df = _panel_nb_data()

    with pytest.raises(MethodIncompatibility, match="data"):
        sp.xtnbreg("y ~ x", data=None, entity="id")

    with pytest.raises(MethodIncompatibility, match="either `formula` or `y=`"):
        sp.xtnbreg(data=df)

    with pytest.raises(MethodIncompatibility, match="fixed-effects xtnbreg"):
        sp.xtnbreg("y ~ x", data=df, model="fe")

    with pytest.raises(MethodIncompatibility, match="time_effects=True"):
        sp.xtnbreg("y ~ x", data=df, entity="id", time_effects=True)

    with pytest.raises(MethodIncompatibility, match="random-effects xtnbreg"):
        sp.xtnbreg("y ~ x", data=df, model="re")

    with pytest.raises(MethodIncompatibility, match="model must be"):
        sp.xtnbreg("y ~ x", data=df, entity="id", model="bad")

    bad_exposure = df.copy()
    bad_exposure["exposure"] = 1.0
    bad_exposure.loc[0, "exposure"] = 0.0
    with pytest.raises(MethodIncompatibility, match="exposure"):
        sp.xtnbreg(
            "y ~ x",
            data=bad_exposure,
            entity="id",
            model="re",
            exposure="exposure",
        )
