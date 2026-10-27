"""Option scan, round 5: ``sp.cross_validate``.

Every engine must fit the model the caller specified; an engine that cannot
must say so (status ``skipped``) instead of fitting another model.
"""

import shutil

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.crossval._external import _stata_command
from statspai.crossval._spec import EstimandSpec

pytest.importorskip("pyfixest")
pytest.importorskip("linearmodels")


@pytest.fixture(scope="module")
def wdf():
    rng = np.random.default_rng(0)
    n = 400
    df = pd.DataFrame(
        {
            "x": rng.normal(size=n),
            "z": rng.normal(size=n),
            "g1": rng.integers(0, 20, n),
            "g2": rng.integers(0, 15, n),
            "w": rng.uniform(0.1, 5, n),
        }
    )
    a = rng.normal(size=20)[df.g1] * 2
    df["d"] = df.z + a + rng.normal(size=n)
    df["y"] = (
        1 + 0.5 * df.x + df.x * (df.w - 2) * 0.5 + 0.7 * df.d + 3 * a
    ) + rng.normal(size=n)
    return df


def _by_engine(cv):
    return {e.engine: e for e in cv.estimates}


def test_weights_reach_every_engine(wdf):
    truth = sp.regress("y ~ x", data=wdf, weights="w")
    cv = sp.cross_validate(
        wdf,
        "ols",
        formula="y ~ x",
        treatment="x",
        weights="w",
        engines=["statspai", "pyfixest", "linearmodels"],
    )
    for e in cv.estimates:
        assert e.ok, e.message
        assert e.coef == pytest.approx(truth.params["x"], rel=1e-10)
        assert e.se == pytest.approx(truth.std_errors["x"], rel=1e-10)
    assert cv.verdict == "AGREE"


@pytest.mark.skipif(shutil.which("Rscript") is None, reason="needs R")
def test_weights_reach_r_fixest(wdf):
    truth = sp.regress("y ~ x", data=wdf, weights="w")
    cv = sp.cross_validate(
        wdf, "ols", formula="y ~ x", treatment="x", weights="w", engines=["R"]
    )
    est = cv.estimates[0]
    if est.status == "unavailable":
        pytest.skip(est.message)
    assert est.coef == pytest.approx(truth.params["x"], rel=1e-10)


def test_engine_without_weights_is_skipped(wdf):
    df = wdf.assign(t=(wdf.z > 0).astype(int))
    cv = sp.cross_validate(
        df,
        "dml",
        y="y",
        treatment="t",
        covariates=["x"],
        weights="w",
        engines=["statspai", "doubleml"],
    )
    for e in cv.estimates:
        assert e.status in ("skipped", "unavailable")


def test_linearmodels_declines_multiway_cluster(wdf):
    cv = sp.cross_validate(
        wdf,
        "ols",
        formula="y ~ x",
        treatment="x",
        cluster=["g1", "g2"],
        engines=["pyfixest", "linearmodels"],
    )
    lm = _by_engine(cv)["linearmodels"]
    assert lm.status == "skipped"


def test_statspai_iv_engine_keeps_fixed_effects(wdf):
    import pyfixest as pf

    ref = pf.feols("y ~ x | g1 | d ~ z", data=wdf, vcov="iid").tidy().loc["d"]
    cv = sp.cross_validate(
        wdf, "iv", formula="y ~ x | g1 | d ~ z", engines=["statspai", "pyfixest"]
    )
    e = _by_engine(cv)["statspai"]
    assert e.coef == pytest.approx(ref["Estimate"], rel=1e-10)
    assert e.se == pytest.approx(ref["Std. Error"], rel=1e-8)
    assert cv.verdict == "AGREE"


def test_ols_with_fixed_effect_block_fits_fixed_effects(wdf):
    import pyfixest as pf

    ref = pf.feols("y ~ d + x | g1", data=wdf).coef()["d"]
    cv = sp.cross_validate(
        wdf,
        "ols",
        formula="y ~ d + x | g1",
        treatment="d",
        engines=["statspai", "pyfixest", "linearmodels"],
    )
    assert cv.estimand == "feols"
    for e in cv.estimates:
        assert e.coef == pytest.approx(ref, rel=1e-8), e.engine


def test_doubleml_engine_follows_model(wdf):
    pytest.importorskip("doubleml")
    df = wdf.assign(t=(wdf.z > 0).astype(int))
    cv = sp.cross_validate(
        df,
        "dml",
        y="y",
        treatment="t",
        covariates=["x", "z"],
        model="irm",
        engines=["doubleml"],
    )
    e = cv.estimates[0]
    assert e.ok, e.message
    assert e.extra["model"] == "irm"

    import doubleml as dml
    from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

    np.random.seed(42)
    data = dml.DoubleMLData(df, y_col="y", d_cols="t", x_cols=["x", "z"])
    irm = dml.DoubleMLIRM(
        data,
        RandomForestRegressor(n_estimators=100, random_state=42),
        RandomForestClassifier(n_estimators=100, random_state=42),
        n_folds=5,
    )
    irm.fit()
    assert e.coef == pytest.approx(float(irm.coef[0]), rel=1e-12)


def test_doubleml_engine_declines_unknown_model(wdf):
    pytest.importorskip("doubleml")
    cv = sp.cross_validate(
        wdf,
        "dml",
        y="y",
        treatment="d",
        covariates=["x"],
        model="pliv",
        engines=["doubleml"],
    )
    assert cv.estimates[0].status == "skipped"


# -- the Stata do-file asks for the specified model ------------------------


def _spec(wdf, **kw):
    return EstimandSpec.from_kwargs(wdf, **kw)


def test_stata_command_carries_vce_and_weights(wdf):
    s = _spec(wdf, estimand="ols", formula="y ~ x", treatment="x", vcov="robust")
    assert _stata_command(s) == "regress y x, vce(robust)"
    s = _spec(wdf, estimand="ols", formula="y ~ x", treatment="x", cluster="g1")
    assert _stata_command(s) == "regress y x, vce(cluster g1)"
    s = _spec(wdf, estimand="ols", formula="y ~ x", treatment="x", weights="w")
    assert _stata_command(s) == "regress y x [aweight=w]"
    s = _spec(wdf, estimand="feols", formula="y ~ x | g1", treatment="x", cluster="g1")
    assert _stata_command(s) == "reghdfe y x, absorb(g1) vce(cluster g1)"
    s = _spec(wdf, estimand="iv", formula="y ~ x | d ~ z", vcov="hc1")
    assert _stata_command(s) == "ivregress 2sls y x (d = z), small vce(robust)"


def test_stata_do_reads_doubles_and_writes_full_precision(wdf):
    from statspai.crossval._external import _stata_do

    s = _spec(wdf, estimand="ols", formula="y ~ x", treatment="x")
    do = _stata_do(s, "x", "in.csv", "out.csv")
    assert do.index("set type double") < do.index("import delimited")
    assert "%21.0g" in do
