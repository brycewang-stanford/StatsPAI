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

A second reference, ``tests/fixtures/xtivreg/stata_xtivreg_tinv.do`` (that
panel and a balanced one), pins the time-invariant regressor / instrument
conventions of ``re`` (2026-10-20):

* with no excluded instrument varying within panels Stata estimates
  ``sigma_e`` from the within regression without the endogenous regressors
  (it raised before);
* on a balanced panel the ``K`` behind ``sigma_e`` counts only the
  regressors the within regression uses, and the between regression behind
  ``sigma_u`` leaves out the excluded instruments constant within panels.
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


# --------------------------------------------------------------------------
# time-invariant regressors and instruments (stata_xtivreg_tinv.do)
# --------------------------------------------------------------------------
STATA_TINV = json.loads((_FIX / "stata_xtivreg_tinv.json").read_text(encoding="utf-8"))
_W = ["x1", "wd"]
TINV_CASES = {
    "unb_only_tinv": ("unb", dict(x=["x1", "w"], instruments=["zb"])),
    "unb_only_tinv_robust": (
        "unb",
        dict(x=["x1", "w"], instruments=["zb"], vce="robust"),
    ),
    "unb_only_tinv_ec2sls": (
        "unb",
        dict(x=["x1", "w"], instruments=["zb"], ec2sls=True),
    ),
    "bal_g2sls": ("bal", dict(x=_W, instruments=["z1", "zb"])),
    "bal_ec2sls": ("bal", dict(x=_W, instruments=["z1", "zb"], ec2sls=True)),
    "bal_nosa": ("bal", dict(x=_W, instruments=["z1", "zb"], nosa=True)),
    "bal_cluster": (
        "bal",
        dict(x=_W, instruments=["z1", "zb"], vce="cluster", cluster="c"),
    ),
    "bal_plain": ("bal", dict(x=["x1"], instruments=["z1", "z2"])),
    "bal_only_tinv": ("bal", dict(x=_W, instruments=["zb"])),
    "bal_only_tinv_robust": ("bal", dict(x=_W, instruments=["zb"], vce="robust")),
    "bal_w_cont": ("bal", dict(x=["x1", "w"], instruments=["z1"])),
}


@pytest.fixture(scope="module")
def tinv_panels():
    return {
        "unb": _mod.add_panel_mean(_mod.make_panel()),
        "bal": _mod.make_balanced_panel(),
    }


def _fit_quiet(df, **kw):
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.xtivreg(df, y="y", endog="d", id="id", **kw)


@pytest.mark.parametrize("tag", list(TINV_CASES))
def test_time_invariant_cases_match_stata(tinv_panels, tag):
    kind, kw = TINV_CASES[tag]
    r = _fit_quiet(tinv_panels[kind], **kw)
    ref = STATA_TINV[tag]
    names = list(r.params.index)
    np.testing.assert_allclose(r.params, [ref[k][0] for k in names], rtol=1e-11)
    np.testing.assert_allclose(r.std_errors, [ref[k][1] for k in names], rtol=1e-11)
    for ours, theirs in [("sigma_e", "e(sigma_e)"), ("sigma_u", "e(sigma_u)")]:
        np.testing.assert_allclose(
            r.model_info[ours], ref[theirs][0], rtol=1e-11, atol=1e-14
        )
    assert r.data_info["nobs"] == ref["e(N)"][0]


def test_no_time_varying_instrument_warns_and_names_the_omission(tinv_panels):
    with pytest.warns(UserWarning, match="no excluded instrument varies"):
        r = sp.xtivreg(
            tinv_panels["unb"],
            y="y",
            x=["x1", "w"],
            endog="d",
            instruments=["zb"],
            id="id",
        )
    assert r.model_info["within_omitted"] == ["d"]


def test_too_few_time_varying_instruments_is_refused(tinv_panels):
    df = tinv_panels["bal"].assign(d2=lambda f: f["d"] + f["x1"] ** 2)
    with pytest.raises(MethodIncompatibility, match="vary within panels"):
        sp.xtivreg(
            df, y="y", x="x1", endog=["d", "d2"], instruments=["z1", "zb"], id="id"
        )


def test_stata_robust_k_counts_a_continuous_time_invariant_regressor(tinv_panels):
    """Documented difference (T4): with vce(robust) on a balanced panel,
    Stata's K counts a continuous time-invariant regressor (its within
    transform keeps rounding noise; a dyadic one, ``wd``, is not counted,
    see bal_only_tinv_robust). Its sigma_e is ours with K one larger:
    sigma_e^2 (N - n - K + 1) is the same within sum of squares."""
    df = tinv_panels["bal"]
    ours = _fit_quiet(df, x=["x1", "w"], instruments=["z1"], vce="robust")
    N, n, K = len(df), df["id"].nunique(), 3  # d, x1, constant
    ssr = ours.model_info["sigma_e"] ** 2 * (N - n - K + 1)
    stata = STATA_TINV["bal_w_cont_robust"]["e(sigma_e)"][0]
    np.testing.assert_allclose(stata**2 * (N - n - (K + 1) + 1), ssr, rtol=1e-12)


# --------------------------------------------------------------------------
# two endogenous regressors and the remaining vce cases
# (stata_xtivreg_more.do, 2026-11-05)
# --------------------------------------------------------------------------
STATA_MORE = json.loads((_FIX / "stata_xtivreg_more.json").read_text(encoding="utf-8"))
_E2, _I3 = ["d1", "d2"], ["z1", "z2", "z3"]
_ONE = dict(endog="d", instruments=["z1", "z2"])
_TWO = dict(endog=_E2, instruments=_I3)
MORE_CASES = {
    "re_robust": ("one", dict(x=["x1", "w"], vce="robust", **_ONE)),
    "re_ec2sls_robust": ("one", dict(x=["x1", "w"], ec2sls=True, vce="robust", **_ONE)),
    "re_nosa_cluster": (
        "one",
        dict(x=["x1", "w"], nosa=True, vce="cluster", cluster="c", **_ONE),
    ),
    "two_re_g2sls": ("two", dict(x=["x1", "w"], **_TWO)),
    "two_re_ec2sls": ("two", dict(x=["x1", "w"], ec2sls=True, **_TWO)),
    "two_re_nosa": ("two", dict(x=["x1", "w"], nosa=True, **_TWO)),
    "two_re_cluster": ("two", dict(x=["x1", "w"], vce="cluster", cluster="c", **_TWO)),
    "two_re_robust": ("two", dict(x=["x1", "w"], vce="robust", **_TWO)),
    "two_re_ec2sls_cluster": (
        "two",
        dict(x=["x1", "w"], ec2sls=True, vce="cluster", cluster="c", **_TWO),
    ),
    "two_re_exact": ("two", dict(x=["x1", "w"], endog=_E2, instruments=["z1", "z2"])),
    "two_fe": ("two", dict(x=["x1"], method="fe", **_TWO)),
    "two_fe_robust": ("two", dict(x=["x1"], method="fe", vce="robust", **_TWO)),
    "two_fe_cluster": (
        "two",
        dict(x=["x1"], method="fe", vce="cluster", cluster="c", **_TWO),
    ),
    "two_fe_exact": (
        "two",
        dict(x=["x1"], method="fe", endog=_E2, instruments=["z1", "z2"]),
    ),
    "two_be": ("two", dict(x=["x1", "w"], method="be", **_TWO)),
    "twobal_re_g2sls": ("twobal", dict(x=["x1", "w"], **_TWO)),
    "twobal_re_ec2sls": ("twobal", dict(x=["x1", "w"], ec2sls=True, **_TWO)),
    "twobal_re_now": ("twobal", dict(x=["x1"], **_TWO)),
    "twobal_fe": ("twobal", dict(x=["x1"], method="fe", **_TWO)),
    "twobal_be": ("twobal", dict(x=["x1", "w"], method="be", **_TWO)),
}
# Stata's xtivreg, re on an unbalanced panel counts two or more endogenous
# regressors as one in the K of s2_nu = SSR_within / (N - n - K + 1) -- and
# only there: its sigma_u uses the full K, as its manual (K = g2 + k1) says
# for both. That count is not followed (see _re_sigma_e_k); these cases are
# compared under it below, and the effect of the difference is bounded.
_STATA_K_DIFFERS = {
    "two_re_g2sls",
    "two_re_ec2sls",
    "two_re_cluster",
    "two_re_robust",
    "two_re_ec2sls_cluster",
    "two_re_exact",
}


@pytest.fixture(scope="module")
def more_panels():
    return {
        "one": _mod.make_panel(),
        "two": _mod.make_two_endog_panel(),
        "twobal": _mod.make_two_endog_balanced(),
    }


def _check_more(r, ref, rtol):
    names = list(r.params.index)
    np.testing.assert_allclose(r.params, [ref[k][0] for k in names], rtol=rtol)
    np.testing.assert_allclose(r.std_errors, [ref[k][1] for k in names], rtol=rtol)
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
            np.testing.assert_allclose(mi[ours], ref[theirs][0], rtol=rtol)
    assert r.data_info["nobs"] == ref["e(N)"][0]
    assert mi["n_groups"] == ref["e(N_g)"][0]


@pytest.mark.parametrize("tag", [t for t in MORE_CASES if t not in _STATA_K_DIFFERS])
def test_more_cases_match_stata(more_panels, tag):
    kind, kw = MORE_CASES[tag]
    r = sp.xtivreg(more_panels[kind], y="y", id="id", **kw)
    # closed form given the data; observed <= 2e-14
    _check_more(r, STATA_MORE[tag], rtol=1e-12)


@pytest.mark.parametrize("tag", sorted(_STATA_K_DIFFERS))
def test_two_endogenous_re_matches_stata_under_its_k(more_panels, tag, monkeypatch):
    xtiv = importlib.import_module("statspai.panel.xtivreg")
    ours = xtiv._re_sigma_e_k

    def stata_k(n_cols, n_keep, n_within_omitted, balanced, se_kind, n_endog):
        k = ours(n_cols, n_keep, n_within_omitted, balanced, se_kind, n_endog)
        return k if balanced else k - (n_endog - 1)

    kind, kw = MORE_CASES[tag]
    df = more_panels[kind]
    default = sp.xtivreg(df, y="y", id="id", **kw)
    monkeypatch.setattr(xtiv, "_re_sigma_e_k", stata_k)
    pinned = sp.xtivreg(df, y="y", id="id", **kw)
    ref = STATA_MORE[tag]
    # every number of Stata's fit, once its K is used (observed <= 1e-14)
    _check_more(pinned, ref, rtol=1e-12)
    # the default: same within sum of squares, Stata's divisor one larger
    N, n = len(df), df["id"].nunique()
    K = 5  # d1, d2, x1, w, constant
    ssr = default.model_info["sigma_e"] ** 2 * (N - n - K + 1)
    np.testing.assert_allclose(
        ref["e(sigma_e)"][0] ** 2 * (N - n - (K - 1) + 1), ssr, rtol=1e-12
    )
    # and what the difference does to the default fit (observed: 1e-3 on
    # sigma_e, 1.2e-3 on rho, 5e-5 on the coefficients, 8e-4 on the SEs)
    names = list(default.params.index)
    np.testing.assert_allclose(default.params, [ref[k][0] for k in names], rtol=1e-4)
    np.testing.assert_allclose(
        default.std_errors, [ref[k][1] for k in names], rtol=2e-3
    )


def test_two_endogenous_balanced_robust_counts_the_time_invariant_regressor(
    more_panels, monkeypatch
):
    """The documented balanced-panel vce(robust) difference above, with two
    endogenous regressors: Stata's K counts the continuous time-invariant
    ``w`` (rounding noise in its within transform), so its sigma_e is ours
    with K one larger."""
    df = more_panels["twobal"]
    ours = sp.xtivreg(df, y="y", id="id", **MORE_CASES["twobal_re_g2sls"][1])
    rob = sp.xtivreg(
        df, y="y", id="id", **dict(MORE_CASES["twobal_re_g2sls"][1], vce="robust")
    )
    assert rob.model_info["sigma_e"] == pytest.approx(
        ours.model_info["sigma_e"], rel=1e-14
    )
    N, n, K = len(df), df["id"].nunique(), 4  # d1, d2, x1, constant
    ssr = ours.model_info["sigma_e"] ** 2 * (N - n - K + 1)
    stata = STATA_MORE["twobal_re_robust"]["e(sigma_e)"][0]
    np.testing.assert_allclose(stata**2 * (N - n - (K + 1) + 1), ssr, rtol=1e-12)
    # with that count every other number of Stata's fit follows
    xtiv = importlib.import_module("statspai.panel.xtivreg")
    base = xtiv._re_sigma_e_k
    monkeypatch.setattr(xtiv, "_re_sigma_e_k", lambda *a: base(*a) + 1)
    pinned = sp.xtivreg(
        df, y="y", id="id", **dict(MORE_CASES["twobal_re_g2sls"][1], vce="robust")
    )
    _check_more(pinned, STATA_MORE["twobal_re_robust"], rtol=1e-12)


def test_more_fixture_records_the_stata_build():
    meta = STATA_MORE["about"]["_meta"]
    assert meta["stata_version"] == "18" and meta["edition"] == "MP"


# --------------------------------------------------------------------------
# R plm (within and between IV) on the same CSVs (r_plm_xtivreg.R)
# --------------------------------------------------------------------------
R_PLM = json.loads((_FIX / "r_plm_xtivreg.json").read_text(encoding="utf-8"))
_R_CASES = {
    "fe": ("one", dict(x=["x1"], method="fe", **_ONE)),
    "fe_robust": ("one", dict(x=["x1"], method="fe", vce="robust", **_ONE)),
    "be": ("one", dict(x=["x1", "w"], method="be", **_ONE)),
    "two_fe": ("two", dict(x=["x1"], method="fe", **_TWO)),
    "two_fe_robust": ("two", dict(x=["x1"], method="fe", vce="robust", **_TWO)),
    "two_be": ("two", dict(x=["x1", "w"], method="be", **_TWO)),
}


@pytest.mark.parametrize("tag", list(_R_CASES))
def test_within_and_between_match_r_plm(more_panels, tag):
    kind, kw = _R_CASES[tag]
    df = more_panels[kind]
    r = sp.xtivreg(df, y="y", id="id", **kw)
    ref = R_PLM[tag]
    # plm's within fit reports no intercept. Its vcovHC(method="arellano",
    # type="sss") is the cluster factor G/(G-1) (N-1)/(N-K) with K counting
    # the slopes only; Stata's K (and ours) counts the constant too, so the
    # variances differ by (N-K)/(N-K-1) exactly (1.05e-3 on the SE here).
    k_slopes = len(r.params) - 1
    N = len(df)
    factor = (N - k_slopes - 1) / (N - k_slopes) if "robust" in tag else 1.0
    # observed <= 3e-14 (R's read.csv is not correctly rounded: 1e-16 floor)
    for name, (b, se) in ref.items():
        np.testing.assert_allclose(r.params[name], b, rtol=1e-12)
        np.testing.assert_allclose(r.std_errors[name] ** 2 * factor, se**2, rtol=1e-12)
    assert R_PLM["about"]["plm"] == "2.6.7"
