"""``sp.feols`` / ``fepois`` / ``feglm`` validate input before any backend.

The pyfixest fallback used to surface its own errors on degenerate input:
"All variables are collinear" on zero rows or an all-NaN regressor,
``ZeroDivisionError`` when p >= n (rOpenSci audit probe P6). The checks now
give the typed errors ``sp.regress`` gives.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import DataInsufficient, NumericalInstability
from statspai.fixest._input_checks import (
    _n_params_plain,
    check_fixest_input,
    referenced_columns,
    translate_backend_failure,
)


@pytest.fixture(scope="module")
def frame() -> pd.DataFrame:
    rng = np.random.default_rng(3)
    n = 200
    df = pd.DataFrame(
        {"x": rng.normal(size=n), "z": rng.normal(size=n), "g": rng.integers(0, 5, n)}
    )
    df["y"] = 1 + df.x + rng.normal(size=n)
    df["c"] = rng.poisson(2.0, n)
    df["b"] = (df.y > 1).astype(float)
    return df


@pytest.mark.parametrize(
    "call",
    [
        lambda d: sp.feols("y ~ x", data=d),
        lambda d: sp.fepois("c ~ x", data=d),
        lambda d: sp.feglm("b ~ x", data=d, family="logit"),
    ],
    ids=["feols", "fepois", "feglm"],
)
def test_zero_rows(frame, call):
    with pytest.raises(DataInsufficient, match="no rows") as err:
        call(frame.iloc[:0])
    assert err.value.diagnostics["n_rows"] == 0


def test_all_missing_regressor(frame):
    d = frame.assign(x=np.nan)
    with pytest.raises(DataInsufficient) as err:
        sp.feols("y ~ x + z", data=d)
    assert err.value.diagnostics["all_missing_columns"] == ["x"]


@pytest.mark.parametrize(
    "n,fml", [(3, "y ~ x + z"), (1, "y ~ x"), (3, "y ~ x + z + g")]
)
def test_no_residual_degrees_of_freedom(frame, n, fml):
    with pytest.raises(
        DataInsufficient, match="more observations than parameters"
    ) as err:
        sp.feols(fml, data=frame.iloc[:n])
    k = err.value.diagnostics["n_params"]
    assert err.value.diagnostics["n_complete"] == n <= k


def test_same_message_as_regress(frame):
    d = frame.iloc[:3]
    with pytest.raises(DataInsufficient) as e_reg:
        sp.regress("y ~ x + z", data=d)
    with pytest.raises(DataInsufficient) as e_fe:
        sp.feols("y ~ x + z", data=d)
    tail = "nobs=3, parameters=3, residual df=0."
    assert str(e_reg.value).split(" -> ")[0].strip().endswith(tail)
    assert str(e_fe.value).split(" -> ")[0].strip().endswith(tail)


def test_one_more_row_than_parameters_fits(frame):
    res = sp.feols("y ~ x + z", data=frame.iloc[:4])
    ref = sp.regress("y ~ x + z", data=frame.iloc[:4])
    np.testing.assert_allclose(res.params.to_numpy(), ref.params.to_numpy(), rtol=1e-9)


def test_inf_in_cluster_or_weights_is_refused(frame):
    d = frame.assign(w=1.0)
    d.loc[2, "w"] = np.inf
    with pytest.raises(DataInsufficient, match="'w'"):
        sp.feols("y ~ x", data=d, weights="w")


def test_referenced_columns_skips_function_names():
    df = pd.DataFrame({"log": [1.0], "x": [1.0], "y": [1.0], "fe": [1]})
    assert referenced_columns("y ~ log(x) | fe", df, ["w", None]) == ["y", "x", "fe"]


@pytest.mark.parametrize(
    "fml,k",
    [("y ~ x + z", 3), ("y ~ x + z - 1", 2), ("y ~ x | fe", None), ("y ~ C(g)", None)],
)
def test_parameter_count_of_plain_formulas(fml, k):
    assert _n_params_plain(fml) == k


def test_check_is_a_no_op_on_non_frames():
    check_fixest_input("y ~ x", {"y": [1]}, entry="sp.feols")
    check_fixest_input(None, pd.DataFrame({"y": [1.0]}), entry="sp.feols")


def test_translate_backend_failure():
    with pytest.raises(DataInsufficient, match="sp.feols"):
        translate_backend_failure(ZeroDivisionError("division by zero"), "sp.feols")
    with pytest.raises(DataInsufficient):
        translate_backend_failure(
            ValueError("All variables are collinear. Maybe ..."), "sp.feols"
        )
    with pytest.raises(NumericalInstability):
        translate_backend_failure(np.linalg.LinAlgError("Matrix is singular."), "x")
    # anything else is left for the caller to re-raise unchanged
    assert translate_backend_failure(ValueError("other"), "sp.feols") is None
