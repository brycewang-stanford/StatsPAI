"""sp.xtivreg against Stata 18 MP ``xtivreg`` on an unbalanced panel.

90 panels of 2 to 8 periods (478 rows) with a time-invariant exogenous
regressor ``w``; reference ``tests/fixtures/xtivreg/stata_xtivreg.do``.
Three conventions were pinned by testing against Stata's output where its
manual is ambiguous or says otherwise:

* the between regression (``be``, and the one behind ``sigma_u`` in
  ``re``) has one row per panel, unweighted;
* in ``re`` the ``K`` of ``s2_nu = SSR_within / (N - n - K + 1)`` counts
  every regressor, the time-invariant ones included;
* the EC2SLS instruments are the within parts and panel means of ``X``
  plus the GLS-transformed constant ``1 - theta_i``.
"""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

import statspai as sp
from statspai.exceptions import MethodIncompatibility

_FIX = Path(__file__).parent / "fixtures" / "xtivreg"
_spec = importlib.util.spec_from_file_location("_xtiv_data", _FIX / "_data.py")
_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_mod)
STATA = json.loads((_FIX / "stata_xtivreg.json").read_text(encoding="utf-8"))

CASES = {
    "re_g2sls": dict(method="re", x=["x1", "w"]),
    "re_ec2sls": dict(method="re", x=["x1", "w"], ec2sls=True),
    "re_nosa": dict(method="re", x=["x1", "w"], nosa=True),
    "re_cluster": dict(method="re", x=["x1", "w"], vce="cluster", cluster="c"),
    "fe": dict(method="fe", x=["x1"]),
    "fe_robust": dict(method="fe", x=["x1"], vce="robust"),
    "fe_cluster": dict(method="fe", x=["x1"], vce="cluster", cluster="c"),
    "be": dict(method="be", x=["x1", "w"]),
}


@pytest.fixture(scope="module")
def df():
    return _mod.make_panel()


@pytest.mark.parametrize("tag", list(CASES))
def test_matches_stata_xtivreg(df, tag):
    r = sp.xtivreg(
        df, y="y", endog="d", instruments=["z1", "z2"], id="id", **CASES[tag]
    )
    ref = STATA[tag]
    names = list(r.params.index)
    assert names[0] == "d" and names[-1] == "_cons"  # Stata's order
    np.testing.assert_allclose(r.params, [ref[k][0] for k in names], rtol=1e-12)
    np.testing.assert_allclose(r.std_errors, [ref[k][1] for k in names], rtol=1e-12)
    mi = r.model_info
    for ours, theirs in [
        ("sigma_u", "e(sigma_u)"),
        ("sigma_e", "e(sigma_e)"),
        ("rho", "e(rho)"),
        ("r2_within", "e(r2_w)"),
        ("r2_between", "e(r2_b)"),
        ("r2_overall", "e(r2_o)"),
    ]:
        if ref.get(theirs, [None])[0] is not None and ours in mi:
            np.testing.assert_allclose(mi[ours], ref[theirs][0], rtol=1e-12)
    assert r.data_info["nobs"] == ref["e(N)"][0] == 478
    assert mi["n_groups"] == ref["e(N_g)"][0] == 90


def test_robust_is_clustering_on_the_panel(df):
    kw = dict(y="y", x="x1", endog="d", instruments=["z1", "z2"], id="id", method="fe")
    a = sp.xtivreg(df, vce="robust", **kw)
    b = sp.xtivreg(df, vce="cluster", cluster="id", **kw)
    np.testing.assert_allclose(a.std_errors, b.std_errors, rtol=1e-14)


def test_balanced_panel_theta_is_one_number(df):
    bal = df[df.groupby("id")["t"].transform("size") >= 4]
    bal = bal[bal["t"] <= 4]
    r = sp.xtivreg(bal, y="y", x="x1", endog="d", instruments=["z1", "z2"], id="id")
    mi = r.model_info
    T = 4
    expected = 1 - np.sqrt(
        mi["sigma_e"] ** 2 / (T * mi["sigma_u"] ** 2 + mi["sigma_e"] ** 2)
    )
    np.testing.assert_allclose(mi["theta"], expected, rtol=1e-12)


def test_exactly_identified_g2sls_is_2sls_on_the_gls_transform(df):
    # with as many instruments as regressors 2SLS is the IV estimator,
    # whatever the projection: X*'(y* - Z* d) = 0
    r = sp.xtivreg(df, y="y", x="x1", endog="d", instruments="z1", id="id")
    th_i = df.groupby("id")["y"].transform("size").to_numpy()
    s2u, s2e = r.model_info["sigma_u"] ** 2, r.model_info["sigma_e"] ** 2
    th = 1 - np.sqrt(s2e / (th_i * s2u + s2e))

    def star(c):
        v = df[c].to_numpy(float) if c != "1" else np.ones(len(df))
        m = df.assign(_v=v).groupby("id")["_v"].transform("mean").to_numpy()
        return v - th * m

    Zs = np.column_stack([star("d"), star("x1"), star("1")])
    Xs = np.column_stack([star("z1"), star("x1"), star("1")])
    np.testing.assert_allclose(
        Xs.T @ (star("y") - Zs @ r.params.to_numpy()), 0, atol=1e-9
    )


def test_underidentified_is_refused(df):
    with pytest.raises(MethodIncompatibility, match="excluded instruments"):
        sp.xtivreg(df, y="y", x="x1", endog=["d", "w"], instruments="z1", id="id")


def test_ec2sls_needs_re(df):
    with pytest.raises(MethodIncompatibility, match="method='re'"):
        sp.xtivreg(
            df,
            y="y",
            x="x1",
            endog="d",
            instruments="z1",
            id="id",
            method="fe",
            ec2sls=True,
        )


def test_fe_reports_time_invariant_regressors_as_omitted(df):
    r = sp.xtivreg(
        df,
        y="y",
        x=["x1", "w"],
        endog="d",
        instruments=["z1", "z2"],
        id="id",
        method="fe",
    )
    assert r.model_info["omitted"] == ["w"]
    np.testing.assert_allclose(r.params["d"], STATA["fe"]["d"][0], rtol=1e-12)
