"""IV mediation against Stata's ``ivmediate`` (Dippel, Ferrara and Heblich).

``sp.iv_mediate`` is compared with ``ivmediate`` 1.0.4 (SSC) on the same CSV
bytes. The reference numbers are in ``_fixtures/iv_mediate_Stata.json``,
written by ``_fixtures/_generate_iv_mediate_Stata.do`` (Stata/MP 18) from
``_fixtures/iv_mediate.csv`` (600 rows, simulated by
``_generate_iv_mediate_data.py``: true total 0.82, direct 0.4, indirect 0.42).
Seven specifications: no controls; two covariates with iid, robust and
cluster variances; plus a 40-level fixed effect with iid, robust and cluster.

Tolerances. Coefficients are closed-form IV on the same partialled data
(observed <= 1e-14), ``rtol=1e-9``. The variances come out of Stata's
``gmm`` and differ from the closed-form sandwich by a uniform ~1e-11
relative (observed <= 1.6e-11 on every element), ``rtol=1e-9``. The F
statistics are closed form (observed <= 1.3e-15), ``rtol=1e-9``.
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import (
    AssumptionWarning,
    IdentificationFailure,
    MethodIncompatibility,
)

FIX = Path(__file__).parent / "_fixtures"
REF = json.loads((FIX / "iv_mediate_Stata.json").read_text(encoding="utf-8"))
RTOL = 1e-9
COV = ["w1", "w2"]

SPECS = {
    "plain": {},
    "cov": {"covariates": COV},
    "cov_robust": {"covariates": COV, "robust": True},
    "cov_cluster": {"covariates": COV, "cluster": "cl"},
    "fe": {"covariates": COV, "absorb": "g"},
    "fe_cluster": {"covariates": COV, "absorb": "g", "cluster": "g"},
    "fe_robust": {"covariates": COV, "absorb": "g", "robust": True},
}


@pytest.fixture(scope="module")
def data():
    return pd.read_csv(FIX / "iv_mediate.csv")


def _fit(data, **kw):
    return sp.iv_mediate(data, y="y", treat="t", mediator="m", instrument="z", **kw)


@pytest.mark.filterwarnings("ignore::statspai.exceptions.AssumptionWarning")
@pytest.mark.parametrize("key", list(SPECS))
def test_matches_ivmediate(data, key):
    res = _fit(data, **SPECS[key])
    np.testing.assert_allclose(res.estimates["coef"], REF[f"{key}_b"], rtol=RTOL)
    V = np.array([REF[f"{key}_V{i}"] for i in (1, 2, 3)])
    np.testing.assert_allclose(res.vcov.to_numpy(), V, rtol=RTOL, atol=0)
    assert res.fstat_treatment == pytest.approx(REF[f"{key}_fstat1"], rel=RTOL)
    assert res.fstat_mediator == pytest.approx(REF[f"{key}_fstat2"], rel=RTOL)
    assert 100 * res.share_mediated == pytest.approx(REF[f"{key}_mepct"], rel=RTOL)
    assert res.n_obs == REF[f"{key}_N"]
    if f"{key}_N_clust" in REF and "cluster" in SPECS[key]:
        assert res.n_clusters == REF[f"{key}_N_clust"]


def test_vce_string_is_the_same_as_the_arguments(data):
    a = _fit(data, covariates=COV, vce="cluster cl")
    b = _fit(data, covariates=COV, cluster="cl")
    pd.testing.assert_frame_equal(a.estimates, b.estimates)
    c = _fit(data, covariates=COV, vce="robust")
    np.testing.assert_allclose(
        c.vcov.to_numpy().ravel(),
        np.array([REF[f"cov_robust_V{i}"] for i in (1, 2, 3)]).ravel(),
        rtol=RTOL,
    )


def test_fixed_effect_equals_dummies(data):
    """``absorb`` is FWL: the same as adding the group dummies as covariates
    (the F statistics differ only because the dummies count as covariates)."""
    fe = _fit(data, covariates=COV, absorb="g")
    d = data.assign(gc=data["g"].astype(str))
    dum = _fit(d, covariates=COV + ["gc"])
    np.testing.assert_allclose(dum.estimates["coef"], fe.estimates["coef"], rtol=1e-10)
    np.testing.assert_allclose(dum.vcov.to_numpy(), fe.vcov.to_numpy(), rtol=1e-9)
    assert dum.fstat_treatment == pytest.approx(fe.fstat_treatment, rel=1e-10)


def test_recovers_known_effects_in_large_sample():
    """Analytic check on the paper's identifying structure."""
    rng = np.random.default_rng(7)
    n = 200_000
    z, v, u = rng.standard_normal((3, n))
    t = 0.8 * z + v + 0.5 * rng.standard_normal(n)
    m = 0.6 * t + v + u + 0.5 * rng.standard_normal(n)
    y = 0.4 * t + 0.7 * m + u + rng.standard_normal(n)
    res = sp.iv_mediate(
        pd.DataFrame({"y": y, "t": t, "m": m, "z": z}),
        y="y",
        treat="t",
        mediator="m",
        instrument="z",
        robust=True,
    )
    est = res.estimates
    # 5 standard errors either way: a failure is a bias, not noise.
    for name, truth in (("total", 0.82), ("direct", 0.4), ("indirect", 0.42)):
        assert abs(est.loc[name, "coef"] - truth) < 5 * est.loc[name, "se"]


def test_weak_instrument_warns():
    rng = np.random.default_rng(3)
    n = 400
    z, v, u = rng.standard_normal((3, n))
    t = 0.02 * z + v + rng.standard_normal(n)
    m = 0.6 * t + v + u
    y = 0.4 * t + 0.7 * m + u + rng.standard_normal(n)
    df = pd.DataFrame({"y": y, "t": t, "m": m, "z": z})
    with pytest.warns(AssumptionWarning, match="weak instrument"):
        res = sp.iv_mediate(df, y="y", treat="t", mediator="m", instrument="z")
    assert "treatment" in res.diagnostics["weak_instrument"]


def test_collinear_mediator_raises(data):
    d = data.assign(m2=2.0 * data["t"] - 0.5 * data["w1"] + 1.0)
    with pytest.raises(IdentificationFailure, match="linear function"):
        sp.iv_mediate(
            d, y="y", treat="t", mediator="m2", instrument="z", covariates=COV
        )


def test_instrument_absorbed_by_fixed_effect_raises(data):
    d = data.assign(zg=data.groupby("g")["z"].transform("mean"))
    with pytest.raises(IdentificationFailure, match="no variation"):
        sp.iv_mediate(d, y="y", treat="t", mediator="m", instrument="zg", absorb="g")


def test_argument_validation(data):
    with pytest.raises(MethodIncompatibility, match="exactly one instrument"):
        sp.iv_mediate(data, y="y", treat="t", mediator="m", instrument=["z", "w1"])
    with pytest.raises(MethodIncompatibility, match="both as"):
        sp.iv_mediate(
            data, y="y", treat="t", mediator="m", instrument="z", covariates=["t"]
        )
    with pytest.raises(MethodIncompatibility, match="not in data"):
        sp.iv_mediate(data, y="y", treat="t", mediator="m", instrument="zz")
    with pytest.raises(MethodIncompatibility, match="one column name"):
        sp.iv_mediate(
            data, y="y", treat="t", mediator="m", instrument="z", absorb=["g", "cl"]
        )


@pytest.mark.filterwarnings("ignore::statspai.exceptions.AssumptionWarning")
def test_missing_rows_dropped_listwise(data):
    d = data.copy()
    d.loc[[0, 5], "w1"] = np.nan
    with pytest.warns(RuntimeWarning, match="2 row"):
        res = _fit(d, covariates=COV)
    ref = _fit(d.dropna(), covariates=COV)
    pd.testing.assert_frame_equal(res.estimates, ref.estimates)
    assert res.n_obs == len(data) - 2


def test_result_protocol(data):
    res = _fit(data, covariates=COV, absorb="g", cluster="g")
    assert "total effect" in res.summary()
    assert list(res.tidy()["effect"]) == ["total", "direct", "indirect"]
    assert res.cite()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res.to_dict()
