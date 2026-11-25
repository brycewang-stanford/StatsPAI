"""``a:b`` / ``a*b`` products and Boolean regressors on the native fixest
engine (iangow/era_statpai#17): ``backend='auto'`` used to hand them to
pyfixest, and ``backend='native'`` raised ``IndexError`` without fixed
effects."""

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import MethodIncompatibility


@pytest.fixture(scope="module")
def frame():
    rng = np.random.default_rng(11)
    n = 3000
    df = pd.DataFrame(
        {
            "treat": rng.random(n) < 0.5,
            "post": rng.random(n) < 0.5,
            "x": rng.normal(size=n),
            "w": rng.normal(size=n),
            "g": rng.integers(0, 30, n),
            "h": rng.integers(0, 12, n),
        }
    )
    df["ti"] = df["treat"].astype(float)
    df["pi"] = df["post"].astype(float)
    df["tp"] = df["ti"] * df["pi"]
    df["xw"] = df["x"] * df["w"]
    df["y"] = (
        1 + 2 * df["tp"] + df["x"] + 0.5 * df["xw"] + 0.1 * df["g"] + rng.normal(size=n)
    )
    df["cnt"] = rng.poisson(np.exp(0.2 + 0.3 * df["tp"] + 0.2 * df["x"]))
    return df


def _fit(fml, df, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.feols(fml, data=df, **kw)


CASES = [
    # formula as written, the same model on precomputed columns, names
    (
        "y ~ treat + post + treat:post",
        "y ~ ti + pi + tp",
        ["Intercept", "treat", "post", "treat:post"],
    ),
    (
        "y ~ treat*post",
        "y ~ ti + pi + tp",
        ["Intercept", "treat", "post", "treat:post"],
    ),
    ("y ~ treat*post | g", "y ~ ti + pi + tp | g", ["treat", "post", "treat:post"]),
    ("y ~ x:w + x + w | g + h", "y ~ x + w + xw | g + h", ["x", "w", "x:w"]),
    ("y ~ x*w + ti", "y ~ x + w + ti + xw", ["Intercept", "x", "w", "ti", "x:w"]),
    ("y ~ ti:pi | g", "y ~ tp | g", ["ti:pi"]),
    ("y ~ treat + x | g", "y ~ ti + x | g", ["treat", "x"]),
    ("y ~ x + x*w", "y ~ x + w + xw", ["Intercept", "x", "w", "x:w"]),
]


@pytest.mark.parametrize("vcov", ["iid", "hetero", {"CRV1": "g"}])
@pytest.mark.parametrize("written,explicit,names", CASES)
def test_product_equals_the_precomputed_column(frame, written, explicit, names, vcov):
    for backend in ("auto", "native"):
        got = _fit(written, frame, vcov=vcov, backend=backend)
        want = _fit(explicit, frame, vcov=vcov, backend="native")
        assert got.model_info["backend"] == "statspai-native"
        assert [str(c) for c in got.params.index] == names
        np.testing.assert_allclose(
            got.params.to_numpy(), want.params.to_numpy(), rtol=1e-12, atol=1e-14
        )
        np.testing.assert_allclose(
            got.std_errors.to_numpy(),
            want.std_errors.to_numpy(),
            rtol=1e-12,
            atol=1e-14,
        )


@pytest.mark.parametrize("written,explicit,names", CASES)
def test_product_equals_pyfixest(frame, written, explicit, names):
    pytest.importorskip("pyfixest")
    native = _fit(written, frame, vcov="hetero", backend="native")
    other = _fit(written, frame, vcov="hetero", backend="pyfixest")
    assert [str(c) for c in other.params.index] == names
    np.testing.assert_allclose(
        native.params.to_numpy(), other.params.to_numpy(), rtol=1e-8, atol=1e-10
    )
    np.testing.assert_allclose(
        native.std_errors.to_numpy(),
        other.std_errors.to_numpy(),
        rtol=1e-8,
        atol=1e-10,
    )


def test_missing_values_in_a_product_component_drop_the_row(frame):
    df = frame.copy()
    df.loc[df.index[:25], "w"] = np.nan
    got = _fit("y ~ x*w | g", df, backend="native")
    df["xw"] = df["x"] * df["w"]
    want = _fit("y ~ x + w + xw | g", df, backend="native")
    assert got.data_info["nobs"] == want.data_info["nobs"] == len(df) - 25
    np.testing.assert_allclose(
        got.params.to_numpy(), want.params.to_numpy(), rtol=1e-12
    )


def test_the_caller_frame_is_not_changed(frame):
    before = frame.dtypes.copy()
    _fit("y ~ treat*post | g", frame)
    assert frame.dtypes.equals(before)
    assert frame["treat"].dtype == bool


def test_fepois_reads_a_product(frame):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        got = sp.fepois("cnt ~ treat*post + x | h", data=frame)
        want = sp.fepois("cnt ~ ti + pi + x + tp | h", data=frame)
    assert [str(c) for c in got.params.index] == ["treat", "post", "x", "treat:post"]
    np.testing.assert_allclose(
        got.params.to_numpy(), want.params.to_numpy(), rtol=1e-9, atol=1e-11
    )


def test_native_refuses_what_it_cannot_parse_with_a_named_error(frame):
    # was: IndexError: list index out of range
    with pytest.raises(MethodIncompatibility, match="backend='auto'"):
        sp.feols("y ~ x:w:ti", data=frame, backend="native")


def test_a_boolean_outcome_is_still_left_to_the_delegate(frame):
    pytest.importorskip("pyfixest")
    df = frame.assign(yb=frame["y"] > 1)
    res = _fit("yb ~ x | g", df)
    assert res.model_info["backend"] == "pyfixest"


@pytest.mark.parametrize(
    "kw,label",
    [
        ({"vcov": {"CRV1": "g"}}, "g"),
        ({"vcov": {"CRV1": "g + h"}}, "g, h"),
        ({"cluster": "g"}, "g"),
    ],
)
def test_both_engines_print_the_same_cluster_row(frame, kw, label):
    # A fit pyfixest ran recorded no clustering columns, so its table had
    # no "Cluster SE" row where the native fit of the same call had one.
    pytest.importorskip("pyfixest")
    rows = {}
    for backend in ("native", "pyfixest"):
        fit = _fit("y ~ x + w | g", frame, backend=backend, **kw)
        assert fit.model_info["backend"].endswith(backend)
        assert fit.model_info["clustervar"] == label.split(", ")
        rows[backend] = [
            line for line in str(sp.regtable(fit)).splitlines() if "Cluster" in line
        ]
    assert rows["native"] == rows["pyfixest"]
    assert (
        len(rows["native"]) == 1
        and rows["native"][0].split()[-len(label.split()) :] == label.split()
    )


def test_an_unclustered_delegate_fit_prints_no_cluster_row(frame):
    pytest.importorskip("pyfixest")
    fit = _fit("y ~ x | g", frame, backend="pyfixest", vcov="hetero")
    assert "clustervar" not in fit.model_info
    assert "Cluster" not in str(sp.regtable(fit))
