"""Configuration-level evidence for the likelihood-based regression family.

Pins what ``sp.validation_scope`` reports for fits of ``logit`` / ``probit`` /
``poisson`` / ``nbreg`` / ``glm`` / ``tobit`` / ``qreg`` / ``heckman``
(``statspai/_validation_scope_ldv.py``): which configurations a reference
ran, and that a configuration no reference ran is not reported as covered.
"""

import pathlib
import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp

_FIX = pathlib.Path(__file__).parent / "reference_parity" / "_fixtures"


@pytest.fixture(scope="module")
def d():
    df = pd.read_csv(_FIX / "ml_sandwich_data.csv")
    df["g"] = np.arange(len(df)) % 25
    rng = np.random.default_rng(1)
    df["yy"] = df["x1"] + rng.normal(size=len(df))
    df["s"] = (df["x2"] + rng.normal(size=len(df)) > 0).astype(int)
    df["yh"] = np.where(df["s"] == 1, df["yy"], np.nan)
    return df


def _scope(fn, *a, **k):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.validation_scope(fn(*a, **k))


CASES = [
    ("logit", lambda d: _scope(sp.logit, "yb ~ x1 + x2", data=d), "covered"),
    (
        "logit",
        lambda d: _scope(sp.logit, "yb ~ x1 + x2", data=d, robust="hc0", weights="w"),
        "covered",
    ),
    (
        "logit",
        lambda d: _scope(
            sp.logit, "yb ~ x1 + x2", data=d, vce="cluster g", weights="w"
        ),
        "covered",
    ),
    # R's sandwich uses expected information: only a disclosure for the SE.
    (
        "probit",
        lambda d: _scope(sp.probit, "yb ~ x1 + x2", data=d, robust="hc0"),
        "estimate_only",
    ),
    (
        "poisson",
        lambda d: _scope(
            sp.poisson, "c ~ x1 + x2", data=d, exposure="expo", robust="hc0"
        ),
        "covered",
    ),
    (
        "poisson",
        lambda d: _scope(
            sp.poisson, "c ~ x1 + x2", data=d, exposure="expo", robust="hc1"
        ),
        "estimate_only",
    ),
    (
        "nbreg",
        lambda d: _scope(
            sp.nbreg, "c ~ x1 + x2", data=d, dispersion="constant", vce="robust"
        ),
        "covered",
    ),
    (
        "glm",
        lambda d: _scope(
            sp.glm,
            "yb ~ x1 + x2",
            data=d,
            family="binomial",
            link="probit",
            robust="hc0",
        ),
        "estimate_only",
    ),
    (
        "glm",
        lambda d: _scope(
            sp.glm,
            "yb ~ x1 + x2",
            data=d,
            family="binomial",
            link="probit",
            robust="hc0",
            information="expected",
        ),
        "covered",
    ),
    (
        "tobit",
        lambda d: _scope(sp.tobit, d, y="yy", x=["x1"], ll=None, ul=0.5),
        "covered",
    ),
    ("qreg", lambda d: _scope(sp.qreg, d, formula="yy ~ x1", weights="w"), "covered"),
    (
        "heckman",
        lambda d: _scope(sp.heckman, d, y="yh", x=["x1"], select="s", z=["x1", "x2"]),
        "covered",
    ),
]


@pytest.mark.parametrize("function,make,status", CASES)
def test_status(d, function, make, status):
    scope = make(d)
    assert scope["function"] == function
    assert scope["status"] == status


def test_tobit_robust_with_weights_and_right_censoring_reads_its_configuration(d):
    scope = _scope(sp.tobit, d, y="yy", x=["x1"], ll=None, ul=0.5, cluster="g")
    assert scope["configuration"] == {
        "method": "mle",
        "vce": "cluster",
        "weights": "none",
        "censoring": "right",
    }


def test_glm_information_is_recorded(d):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = sp.glm("yb ~ x1 + x2", data=d, family="binomial", information="expected")
    assert fit.model_info["information"] == "expected"
