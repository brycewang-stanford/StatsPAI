"""Configuration-level evidence for ``sp.feols`` and ``sp.ppmlhdfe``."""

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp


@pytest.fixture(scope="module")
def d():
    rng = np.random.default_rng(0)
    n = 600
    df = pd.DataFrame(
        {
            "x": rng.normal(size=n),
            "f": rng.integers(0, 30, n),
            "h": rng.integers(0, 10, n),
            "g": rng.integers(0, 40, n),
            "w": rng.uniform(0.5, 2, n),
        }
    )
    df["y"] = df["x"] + rng.normal(size=n)
    df["c"] = rng.poisson(np.exp(0.3 * df["x"]))
    return df


def _scope(fn, *a, **k):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.validation_scope(fn(*a, **k))


CASES = [
    (
        lambda d: _scope(sp.feols, "y ~ x | f + h", data=d, vce="cluster g"),
        "covered",
        "cr1",
    ),
    (
        lambda d: _scope(sp.feols, "y ~ x | f", data=d, vce="robust", weights="w"),
        "covered",
        "hetero",
    ),
    (lambda d: _scope(sp.feols, "y ~ x", data=d), "covered", "iid"),
    (
        lambda d: _scope(sp.feols, "y ~ x | f + h", data=d, vce="cluster g h"),
        "covered",
        "cluster_multiway",
    ),
    # pyfixest's CRV3 has no reference row.
    (
        lambda d: _scope(sp.feols, "y ~ x | f", data=d, vce="cr3 g"),
        "estimate_only",
        "cr3",
    ),
]


@pytest.mark.parametrize("make,status,vcov", CASES)
def test_feols_status(d, make, status, vcov):
    scope = make(d)
    assert scope["function"] == "feols"
    assert scope["configuration"]["vcov"] == vcov
    assert scope["status"] == status


def test_feols_custom_ssc_is_its_own_cell(d):
    scope = _scope(sp.feols, "y ~ x | f", data=d, vce="robust", ssc=sp.ssc("reghdfe"))
    assert scope["configuration"]["ssc"] == "custom"
    assert scope["status"] == "estimate_only"


def test_ppmlhdfe_status(d):
    a = _scope(sp.ppmlhdfe, "c ~ x | f", data=d, cluster="g")
    assert a["function"] == "ppmlhdfe" and a["status"] == "covered"
    b = _scope(sp.ppmlhdfe, "c ~ x | f + h", data=d, robust="robust", ssc="fixest")
    assert b["configuration"]["ssc"] == "fixest" and b["status"] == "covered"
