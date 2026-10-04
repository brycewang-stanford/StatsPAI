"""``sp.nlogit`` against Stata 18 ``nlogit`` and R ``mlogit``.

Fixtures: ``_fixtures/_generate_nlogit_mlogit.R`` simulates choices from a
nested logit (two nests of two alternatives, dissimilarity 0.5 and 0.8) and
fits ``mlogit::mlogit(nests = ...)``; ``_generate_nlogit_stata.do`` fits
``nlogit`` on the same CSV.

* Stata, every block at 1e-6: coefficients 1e-10, standard errors 4e-8
  under ``vce(oim)``, 8e-8 under ``vce(robust)`` and ``vce(cluster)``, the
  log-likelihood to 1e-11 and the LR test of IIA to 1e-12.
* R: coefficients at 1e-6 and log-likelihoods at 1e-8, for separate
  dissimilarity parameters, a common one, and no constants.

mlogit's standard errors are not compared as equal. They differ from
Stata's by up to 4% on the same estimates, and StatsPAI's, which come from
the exact Hessian, are Stata's. The test asserts both facts.
"""

from __future__ import annotations

import json
import pathlib
import warnings

import numpy as np
import pandas as pd
import pytest

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import statspai as sp

_FIX = pathlib.Path(__file__).parent / "_fixtures"
NESTS = {"A": ["a1", "a2"], "B": ["b1", "b2"]}
KW = dict(y="chosen", x=["x1", "x2"], chid="id", alt="alt", nests=NESTS)
# sp: x1, x2, constants, lambdas. mlogit: constants, x1, x2, iv.
R_ORDER = [3, 4, 0, 1, 2, 5, 6]


@pytest.fixture(scope="module")
def data() -> pd.DataFrame:
    return pd.read_csv(_FIX / "nlogit_data.csv", skipinitialspace=True)


@pytest.fixture(scope="module")
def stata() -> dict:
    return json.loads((_FIX / "nlogit_stata.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def mlogit() -> dict:
    return json.loads((_FIX / "nlogit_mlogit.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def fit(data):
    return sp.nlogit(data, **KW)


@pytest.mark.parametrize(
    "block, extra",
    [
        ("separate", {}),
        ("separate_robust", {"vce": "robust"}),
        ("separate_cluster", {"cluster": "clust"}),
    ],
)
def test_matches_stata(block, extra, data, stata):
    res, ref = sp.nlogit(data, **KW, **extra), stata[block]
    assert list(res.params.index) == [
        "x1",
        "x2",
        "_cons:a2",
        "_cons:b1",
        "_cons:b2",
        "lambda:A",
        "lambda:B",
    ]
    np.testing.assert_allclose(res.params.values, ref["b"], rtol=1e-6)
    np.testing.assert_allclose(res.std_errors.values, ref["se"], rtol=1e-6)
    assert res.model_info["ll"] == pytest.approx(ref["ll"], abs=1e-8)
    assert res.model_info["converged"]


def test_iia_test_matches_stata(fit, stata, mlogit):
    info = fit.model_info
    assert info["lr_iia_chi2"] == pytest.approx(stata["separate"]["chi2_c"], rel=1e-8)
    assert info["lr_iia_df"] == 2
    assert info["ll_clogit"] == pytest.approx(mlogit["conditional"]["ll"], abs=1e-8)
    assert info["lr_iia_pvalue"] < 1e-10


def test_matches_mlogit_coefficients(fit, data, mlogit):
    ref = mlogit["separate"]
    np.testing.assert_allclose(
        fit.params.values, np.array(ref["b"])[R_ORDER], rtol=1e-6
    )
    assert fit.model_info["ll"] == pytest.approx(ref["ll"], abs=1e-8)
    common = sp.nlogit(data, **KW, common_lambda=True)
    ref = mlogit["common"]
    np.testing.assert_allclose(
        common.params.values, np.array(ref["b"])[[3, 4, 0, 1, 2, 5]], rtol=1e-6
    )
    assert common.model_info["ll"] == pytest.approx(ref["ll"], abs=1e-8)
    assert list(common.params.index)[-1] == "lambda"
    bare = sp.nlogit(data, **KW, constants=False)
    ref = mlogit["no_constants"]
    np.testing.assert_allclose(bare.params.values, ref["b"], rtol=1e-6)
    assert bare.model_info["ll"] == pytest.approx(ref["ll"], abs=1e-8)


def test_mlogit_standard_errors_are_the_odd_one_out(fit, stata, mlogit):
    ours = fit.std_errors.values
    theirs = np.array(mlogit["separate"]["se"])[R_ORDER]
    assert np.max(np.abs(ours / theirs - 1.0)) > 0.02
    np.testing.assert_allclose(ours, stata["separate"]["se"], rtol=1e-6)


def test_recovers_the_truth(fit):
    """The R script's DGP: 0.8, -0.6, constants 0.3 / -0.2 / 0.4, lambda 0.5 / 0.8."""
    truth = np.array([0.8, -0.6, 0.3, -0.2, 0.4, 0.5, 0.8])
    z = (fit.params.values - truth) / fit.std_errors.values
    assert np.all(np.abs(z) < 2.5)


def test_lambda_one_is_the_conditional_logit(data):
    """With IIA true the dissimilarity parameters are near 1 and the LR is small."""
    rng = np.random.default_rng(5)
    n, alts = 1500, ["a1", "a2", "b1", "b2"]
    d = pd.DataFrame({"id": np.repeat(np.arange(n), 4), "alt": alts * n})
    d["x1"] = rng.normal(size=len(d))
    u = 0.7 * d["x1"].to_numpy() + rng.gumbel(size=len(d))
    d["chosen"] = (
        (u.reshape(n, 4) == u.reshape(n, 4).max(axis=1, keepdims=True))
        .ravel()
        .astype(int)
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sp.nlogit(d, y="chosen", x=["x1"], chid="id", alt="alt", nests=NESTS)
    lam = res.params[["lambda:A", "lambda:B"]]
    assert np.all(np.abs(lam - 1.0) < 3.0 * res.std_errors[["lambda:A", "lambda:B"]])
    assert res.model_info["lr_iia_pvalue"] > 0.01


def test_degenerate_nest_has_no_parameter(data):
    res = sp.nlogit(
        data,
        y="chosen",
        x=["x1", "x2"],
        chid="id",
        alt="alt",
        nests={"A": ["a1", "a2", "b1"], "solo": ["b2"]},
    )
    assert [n for n in res.params.index if n.startswith("lambda")] == ["lambda:A"]


def test_refusals(data):
    bad = sp.exceptions.MethodIncompatibility
    with pytest.raises(bad, match="required"):
        sp.nlogit(data, y="chosen", x=["x1"], chid="id", alt="alt")
    with pytest.raises(bad, match="do not match"):
        sp.nlogit(
            data, y="chosen", x=["x1"], chid="id", alt="alt", nests={"A": ["a1", "a2"]}
        )
    with pytest.raises(bad, match="two nests"):
        sp.nlogit(
            data,
            y="chosen",
            x=["x1"],
            chid="id",
            alt="alt",
            nests={"A": ["a1", "a2"], "B": ["a2", "b1", "b2"]},
        )
    with pytest.raises(bad, match="exactly once"):
        sp.nlogit(data.iloc[1:], **KW)
    with pytest.raises(bad, match="exactly one chosen"):
        sp.nlogit(data.assign(chosen=1), **KW)
    with pytest.raises(bad, match="conditional logit"):
        sp.nlogit(
            data,
            y="chosen",
            x=["x1"],
            chid="id",
            alt="alt",
            nests={a: [a] for a in ["a1", "a2", "b1", "b2"]},
        )


def test_cite(fit):
    assert "heiss2002structural" in fit.cite()
