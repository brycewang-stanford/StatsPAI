"""HC0-HC3 sandwich SEs of the likelihood models against R ``sandwich``.

Fills cells of the validation-scope maps that had no reference: the HC0 /
HC1 conventions of ``sp.logit`` / ``sp.probit`` / ``sp.poisson`` (weighted
and not), HC0-HC3 of ``sp.glm``, and ``sp.poisson(exposure=)``.

Reference: R 4.5 ``glm`` + ``sandwich::vcovHC`` on
``_fixtures/ml_sandwich_data.csv`` (written by
``_generate_ml_sandwich_data.py``), numbers in ``ml_sandwich_R.json``
(``_generate_ml_sandwich_R.R``). Weighted fits are R prior weights, which is
Stata's ``[iw]`` / ``[pw]`` for the point estimate.

One documented difference: ``sandwich`` builds its bread from R ``glm``'s
*expected* information. On the canonical links (logit, poisson) that is the
observed information too; on probit it is not, and ``sp.probit`` follows
Stata (observed). The probit rows therefore pin the 2-3% gap and show that
``sp.glm(..., information='expected')`` closes it.
"""

import json
import pathlib
import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp

_FIX = pathlib.Path(__file__).parent / "_fixtures"
R = json.loads((_FIX / "ml_sandwich_R.json").read_text(encoding="utf-8"))
TOL = 1e-8


@pytest.fixture(scope="module")
def data() -> pd.DataFrame:
    return pd.read_csv(_FIX / "ml_sandwich_data.csv", encoding="utf-8")


def _fit(fn, formula, data, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(formula, data=data, **kw)


MODELS = {
    "logit": (lambda f, **kw: sp.logit(f, **kw), "yb ~ x1 + x2"),
    "poisson": (lambda f, **kw: sp.poisson(f, **kw), "c ~ x1 + x2"),
}


@pytest.mark.parametrize("vce", ["hc0", "hc1"])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("model", sorted(MODELS))
def test_canonical_link_sandwich(data, model, weighted, vce):
    fn, formula = MODELS[model]
    kw = {"weights": "w"} if weighted else {}
    fit = _fit(fn, formula, data, robust=vce, **kw)
    ref = R[f"{model}_{'w' if weighted else 'none'}"]
    np.testing.assert_allclose(fit.params.to_numpy(), ref["coef"], rtol=TOL)
    np.testing.assert_allclose(fit.std_errors.to_numpy(), ref[vce.upper()], rtol=TOL)


@pytest.mark.parametrize("vce", ["hc0", "hc1"])
@pytest.mark.parametrize("weighted", [False, True])
def test_probit_sandwich_differs_from_r_by_the_information_matrix(data, weighted, vce):
    kw = {"weights": "w"} if weighted else {}
    fit = _fit(lambda f, **k: sp.probit(f, **k), "yb ~ x1 + x2", data, robust=vce, **kw)
    ref = R[f"probit_{'w' if weighted else 'none'}"]
    np.testing.assert_allclose(fit.params.to_numpy(), ref["coef"], rtol=1e-7)
    gap = np.abs(fit.std_errors.to_numpy() / np.asarray(ref[vce.upper()]) - 1)
    # Observed (Stata, this package) vs expected (R glm) information.
    assert 1e-3 < gap.max() < 0.05


def test_glm_expected_information_reproduces_r_probit_sandwich(data):
    fit = _fit(
        lambda f, **k: sp.glm(f, **k),
        "yb ~ x1 + x2",
        data,
        family="binomial",
        link="probit",
        robust="hc0",
        information="expected",
    )
    np.testing.assert_allclose(
        fit.std_errors.to_numpy(), R["probit_none"]["HC0"], rtol=TOL
    )


@pytest.mark.parametrize("vce", ["hc0", "hc1", "hc2", "hc3"])
@pytest.mark.parametrize(
    "model,family", [("logit", "binomial"), ("poisson", "poisson")]
)
def test_glm_hc_family(data, model, family, vce):
    _, formula = MODELS[model]
    fit = _fit(lambda f, **k: sp.glm(f, **k), formula, data, family=family, robust=vce)
    np.testing.assert_allclose(
        fit.std_errors.to_numpy(), R[f"{model}_none"][vce.upper()], rtol=TOL
    )


@pytest.mark.parametrize("vce,key", [("nonrobust", "se_model"), ("hc0", "HC0")])
def test_poisson_exposure(data, vce, key):
    fit = _fit(
        lambda f, **k: sp.poisson(f, **k),
        "c ~ x1 + x2",
        data,
        exposure="expo",
        robust=vce,
    )
    ref = R["poisson_exposure"]
    np.testing.assert_allclose(fit.params.to_numpy(), ref["coef"], rtol=TOL)
    np.testing.assert_allclose(fit.std_errors.to_numpy(), ref[key], rtol=TOL)
