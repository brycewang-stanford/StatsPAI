"""Reference parity: RD kink / plot / density functions vs R, on identical bytes.

Functions pinned here against the R implementation each one names:

* ``sp.rdrobust(deriv=1)`` and ``sp.rkd`` -- R ``rdrobust(deriv = 1)``
  (rdrobust 4.0.0), sharp and fuzzy, ``p`` 1 and default, ``vce`` nn / hc1,
  fixed and selected bandwidths, uniform kernel, clustered; and Stata
  ``rdrobust ..., deriv(1)`` (same authors) on six of those cells
  (``_fixtures/rd_iv_Stata.json``, from ``_generate_rd_iv_stata.do``).
* ``sp.rdplot`` -- R ``rdrobust::rdplot``: all eight ``binselect`` rules,
  manual bins with a triangular kernel, covariate adjustment, and the Senate
  data (with missing outcomes).
* ``sp.rdplotdensity`` -- R ``rddensity::rdplotdensity(rddensity(X), X)``,
  i.e. ``lpdensity`` on each side with the ``rddensity`` bandwidths, on a
  running variable with mass points and on the Senate margin.
* ``sp.mccrary_test`` -- R ``rdd::DCdensity`` (CRAN archive 0.57).
* ``sp.rdhte`` / ``sp.rdbwhte`` / ``sp.rdhte_lincom`` -- R ``rdhte``:
  continuous and 0/1 (subgroup) moderators, HC0-HC3 and CR1, p = 2,
  three kernels, selected and fixed bandwidths, linear combinations.
* ``sp.rd_bias_aware_fuzzy`` -- R ``RDHonest(y | d ~ x)`` for every
  ingredient (jumps, nearest-neighbour or cluster variance, worst-case bias,
  rule-of-thumb ``M``, default bandwidth) and for RDHonest's own linearised
  interval, returned as ``model_info['bias_aware']['rdhonest']``. The
  Anderson-Rubin-type set built from those ingredients has no reference
  implementation; it is pinned by its defining identity instead.
* ``sp.rdsensitivity`` / ``sp.rdrbounds`` -- R ``rdlocrand``. Randomization
  p-values are Monte-Carlo on both sides (T3): compared within four pooled
  binomial standard errors at 4000 draws each, never as parity. The
  per-window estimates of ``rdsensitivity`` are deterministic and pinned
  to ``rdrandinf``'s observed statistic at 1e-9.

Fixture: ``_generate_rd_iv_rd_R.R`` (writes ``rd_iv_kink.csv``, ``rd_iv_hte.csv``,
``rd_iv_density.csv`` and ``rd_iv_rd_R.json``; reads ``rdsenate.csv``).

Conventions each number depends on
----------------------------------
* ``rdrobust(deriv=1)`` without ``p`` uses ``p = 2`` (R: ``p <- deriv + 1``);
  an explicit ``p = 1`` is honoured. Through 1.28.0 ``sp.rdrobust``
  silently raised any ``p <= deriv`` to ``deriv + 1``, so the ``p1`` rows
  were the ``p = 2`` estimate (1.3843 vs R's 0.9933 on ``sharp_p1_nn``).
* ``sp.rkd`` is ``rdrobust(deriv=1, vce='hc1')``: headline = conventional
  estimate and HC1 (or cluster) SE; ``model_info['robust']`` = R's robust
  bias-corrected row. Its default bandwidth is R's ``mserd``.
* ``rdplot`` R indexes, returned by StatsPAI 0-based: the fixture keeps
  every 37th point of R's 2 x 500 polynomial grid with R's 1-based index.
  Mass-point adjustment (``masspoints='adjust'``) is R's default and is
  exercised by the Senate data.
* ``rdplotdensity``: ``plotN = 10`` in R, ``n_grid=10`` here; side scale
  ``n_side / (n - 1)`` (R's, not ``n``); ``q = p + 1``.
* ``DCdensity``: bin width ``2 sd(x) n^(-1/2)`` and McCrary's bandwidth rule
  when not supplied.
* ``rdhte``: ``coef`` is the order-p fit, ``coef_bc`` / ``se_rb`` / ``vcov``
  the order-(p+1) fit on the same window with ``sandwich::vcovCL`` (whose
  HC0 carries ``n/(n-1)``, HC1 ``n/(n-k)``, HC2/HC3 no factor, CR1
  ``G/(G-1) (n-1)/(n-k)``). Default bandwidth = ``rdbwselect`` on x
  (per subgroup for a 0/1 moderator); ``vce`` defaults to HC3, CR1 with a
  cluster. ``rdhte_lincom`` estimates ``L coef`` and tests ``L coef_bc``.

Tolerances: closed forms and least squares at rel 1e-9 unless stated. The
p = 4 global polynomial coefficients of ``rdplot`` are ill-conditioned on the
raw scale (the design has ``(x - c)^4``); they agree to 1e-10 and are held at
1e-8. Bandwidths chosen by the CCT cascade are held at 1e-8 (two
implementations of the same closed-form cascade; observed ~1e-12).
"""

from __future__ import annotations

import json
import pathlib
import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

import statspai as sp  # noqa: E402

_FIX = pathlib.Path(__file__).parent / "_fixtures"


def _rel(a, b) -> float:
    a = np.atleast_1d(np.asarray(a, dtype=float))
    b = np.atleast_1d(np.asarray(b, dtype=float))
    assert a.shape == b.shape, (a.shape, b.shape)
    return float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-300)))


@pytest.fixture(scope="module")
def R():
    return json.loads((_FIX / "rd_iv_rd_R.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def kink():
    return pd.read_csv(_FIX / "rd_iv_kink.csv")


@pytest.fixture(scope="module")
def dens():
    return pd.read_csv(_FIX / "rd_iv_density.csv")


@pytest.fixture(scope="module")
def senate():
    return pd.read_csv(_FIX / "rdsenate.csv")


# --------------------------------------------------------------------------- #
#  rdrobust(deriv = 1)
# --------------------------------------------------------------------------- #
_KINK = {
    "sharp_p1_nn": dict(p=1, vce="nn"),
    "sharp_pdef_nn": dict(vce="nn"),
    "sharp_p1_hc1": dict(p=1, vce="hc1"),
    "sharp_p1_hc1_h04": dict(p=1, vce="hc1", h=0.4),
    "sharp_p1_uni_nn": dict(p=1, vce="nn", kernel="uniform"),
    "sharp_p1_hc1_cl": dict(p=1, vce="hc1", cluster="g"),
    "fuzzy_p1_nn": dict(p=1, vce="nn", fuzzy="t"),
    "fuzzy_pdef_nn": dict(vce="nn", fuzzy="t"),
    "fuzzy_p1_hc1": dict(p=1, vce="hc1", fuzzy="t"),
    "fuzzy_p1_hc1_h04": dict(p=1, vce="hc1", fuzzy="t", h=0.4),
}


@pytest.mark.parametrize("key", list(_KINK))
def test_rdrobust_deriv1_matches_r(key, R, kink):
    ref = R["kink"][key]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = sp.rdrobust(
            kink, y="y", x="x", deriv=1, manipulation_test=False, **_KINK[key]
        )
    mi = r.model_info
    assert _rel(mi["conventional"]["estimate"], ref["coef"][0]) < 1e-9
    assert _rel(mi["robust"]["estimate"], ref["coef"][2]) < 1e-9
    assert _rel(mi["conventional"]["se"], ref["se"][0]) < 1e-9
    assert _rel(mi["robust"]["se"], ref["se"][2]) < 1e-9
    assert _rel(mi["bandwidth_h"], ref["h"][0]) < 1e-8
    assert _rel(mi["bandwidth_b"], ref["b"][0]) < 1e-8
    assert mi["polynomial_p"] == (_KINK[key].get("p") or 2)


@pytest.fixture(scope="module")
def ST():
    return json.loads((_FIX / "rd_iv_Stata.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize(
    "key",
    [
        "sharp_p1_hc1",
        "sharp_pdef_nn",
        "sharp_p1_hc1_h04",
        "fuzzy_p1_hc1",
        "fuzzy_pdef_nn",
        "fuzzy_p1_hc1_h04",
    ],
)
def test_rdrobust_deriv1_matches_stata(key, ST, kink):
    ref = ST["kink"][key]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = sp.rdrobust(
            kink, y="y", x="x", deriv=1, manipulation_test=False, **_KINK[key]
        )
    mi = r.model_info
    assert (
        _rel([mi["conventional"]["estimate"], mi["robust"]["estimate"]], ref["coef"])
        < 1e-9
    )
    assert _rel([mi["conventional"]["se"], mi["robust"]["se"]], ref["se"]) < 1e-9
    assert _rel([mi["bandwidth_h"], mi["bandwidth_b"]], [ref["h"], ref["b"]]) < 1e-8
    assert mi["polynomial_p"] == ref["p"]
    if "fuzzy" not in _KINK[key] and _KINK[key].get("vce") == "hc1":
        rk = sp.rkd(kink, y="y", x="x", h=_KINK[key].get("h"))
        assert _rel([rk.estimate, rk.se], [ref["coef"][0], ref["se"][0]]) < 1e-9


def test_rdrobust_rejects_deriv_above_p(kink):
    with pytest.raises(Exception, match="must not exceed"):
        sp.rdrobust(kink, y="y", x="x", deriv=2, p=1, manipulation_test=False)


def test_rdrobust_refuses_empty_window(kink):
    """Through 1.28.0 h=1e-4 returned an 'estimate' with SE 6e-22 and p=0."""
    with pytest.raises(ValueError, match="Insufficient observations"):
        sp.rdrobust(kink, y="y", x="x", h=1e-4, manipulation_test=False)


# --------------------------------------------------------------------------- #
#  rkd = rdrobust(deriv=1, vce='hc1'), conventional headline
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "key,kw",
    [
        ("sharp_p1_hc1", {}),
        ("sharp_p1_hc1_h04", dict(h=0.4)),
        ("sharp_p1_hc1_cl", dict(cluster="g")),
        ("fuzzy_p1_hc1", dict(treatment="t")),
        ("fuzzy_p1_hc1_h04", dict(treatment="t", h=0.4)),
    ],
)
def test_rkd_matches_r(key, kw, R, kink):
    ref = R["kink"][key]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = sp.rkd(kink, y="y", x="x", **kw)
    assert _rel(r.estimate, ref["coef"][0]) < 1e-9
    # Through 1.28.0 the fuzzy SE was a delta method without the covariance
    # of the two kinks (3.6% off R on the h = 0.4 row).
    assert _rel(r.se, ref["se"][0]) < 1e-9
    assert _rel(r.model_info["robust"]["estimate"], ref["coef"][2]) < 1e-9
    assert _rel(r.model_info["robust"]["se"], ref["se"][2]) < 1e-9
    assert _rel(r.model_info["bandwidth"], ref["h"][0]) < 1e-8
    if "treatment" not in kw and "cluster" not in kw:
        # Identity: the sharp kink is the difference of the two side slopes.
        assert r.estimate == pytest.approx(r.model_info["kink_outcome"], rel=1e-10)


# --------------------------------------------------------------------------- #
#  rdplot
# --------------------------------------------------------------------------- #
_PLOT = {
    f"kink_{b}": dict(binselect=b)
    for b in ("es", "espr", "esmv", "esmvpr", "qs", "qspr", "qsmv", "qsmvpr")
}
_PLOT["kink_tri_p2_h05_nbins"] = dict(kernel="triangular", p=2, h=0.5, nbins=(7, 9))
_PLOT["kink_covs"] = dict(covs=["t"])


def _rdplot_check(res, ref):
    vb = res["vars_bins"]
    assert list(res["J"]) == ref["J"]
    assert list(res["J_IMSE"]) == ref["J_IMSE"]
    assert list(res["J_MV"]) == ref["J_MV"]
    assert list(vb["rdplot_N"]) == ref["N"]
    for ours, theirs in (
        ("rdplot_mean_bin", "mean_bin"),
        ("rdplot_mean_x", "mean_x"),
        ("rdplot_mean_y", "mean_y"),
        ("rdplot_se_y", "se_y"),
        ("rdplot_ci_l", "ci_l"),
        ("rdplot_ci_r", "ci_r"),
    ):
        assert _rel(vb[ours], ref[theirs]) < 1e-9, ours
    # p = 4 on the raw scale: see the module docstring.
    assert _rel(res["coef"].ravel(order="F"), ref["coef"]) < 1e-8
    idx = np.asarray(ref["poly_idx"]) - 1
    assert _rel(res["vars_poly"]["rdplot_y"][idx], ref["poly_y"]) < 1e-9


@pytest.mark.parametrize("key", list(_PLOT))
def test_rdplot_matches_r(key, R, kink):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fig, _ = sp.rdplot(kink, y="y", x="x", **_PLOT[key])
    _rdplot_check(fig.rdplot_data, R["rdplot"][key])
    plt.close(fig)


def test_rdplot_senate_matches_r(R, senate):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fig, _ = sp.rdplot(senate, y="vote", x="margin")
    res = fig.rdplot_data
    _rdplot_check(res, R["rdplot"]["senate_default"])
    # Identity: every complete row lands in exactly one bin.
    ok = senate[["vote", "margin"]].dropna()
    assert int(np.sum(res["vars_bins"]["rdplot_N"])) == len(ok)
    plt.close(fig)


# --------------------------------------------------------------------------- #
#  rdplotdensity
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("key,col", [("density", "x"), ("senate", "margin")])
def test_rdplotdensity_matches_r(key, col, R, dens, senate):
    ref = R["rdplotdensity"][key]
    data = dens if key == "density" else senate
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fig, _ = sp.rdplotdensity(data, x=col, n_grid=10)
    out = fig.rdplotdensity_data
    assert _rel(out["h"], ref["h"]) < 1e-8
    for side, est in (("left", out["Estl"]), ("right", out["Estr"])):
        for k, col_name in enumerate(ref["cols"]):
            tol = 1e-8 if col_name in ("grid", "bw") else 1e-9
            if col_name == "nh":
                assert list(est["nh"]) == ref[side][k]
            else:
                # Bandwidth rel 1e-12 propagates into the density at ~1e-10.
                assert _rel(est[col_name], ref[side][k]) < max(tol, 1e-8), (
                    side,
                    col_name,
                )
    plt.close(fig)


# --------------------------------------------------------------------------- #
#  McCrary -- rdd::DCdensity
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize(
    "key,frame,col,kw",
    [
        ("density_c0", "dens", "x", dict(c=0)),
        ("density_c05_fixed", "dens", "x", dict(c=0.5, bw=0.4, bin_width=0.05)),
        ("senate_c0", "senate", "margin", dict(c=0)),
    ],
)
def test_mccrary_matches_dcdensity(key, frame, col, kw, R, dens, senate):
    ref = R["mccrary"][key]
    data = dens if frame == "dens" else senate
    r = sp.mccrary_test(data, x=col, **kw)
    mi = r.model_info
    assert _rel(r.estimate, ref["theta"]) < 1e-9
    assert _rel(r.se, ref["se"]) < 1e-9
    assert _rel(mi["z"], ref["z"]) < 1e-9
    assert _rel(r.pvalue, ref["p"]) < 1e-8
    assert _rel(mi["bin_width"], ref["bin"]) < 1e-12
    assert _rel(mi["bandwidth"], ref["bw"]) < 1e-9
    # Identity: McCrary's SE formula from the two side densities.
    n = int(data[col].notna().sum())
    se = np.sqrt(
        24
        / (5 * n * mi["bandwidth"])
        * (1 / mi["density_left"] + 1 / mi["density_right"])
    )
    assert r.se == pytest.approx(se, rel=1e-12)


# --------------------------------------------------------------------------- #
#  rdhte / rdbwhte / rdhte_lincom -- R rdhte
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def hte():
    return pd.read_csv(_FIX / "rd_iv_hte.csv")


_HTE = {
    "cont_h04_hc1": dict(z="z", h=0.4, vce="hc1"),
    "cont_h04_hc3": dict(z="z", h=0.4),
    "cont_h04_hc0": dict(z="z", h=0.4, vce="hc0"),
    "cont_h04_hc2": dict(z="z", h=0.4, vce="hc2"),
    "cont_default": dict(z="z"),
    "cont_cluster": dict(z="z", cluster="cl"),
    "cont_p2_h05": dict(z="z", p=2, h=0.5),
    "cont_epa_h04": dict(z="z", kernel="epanechnikov", h=0.4),
    "cont_uni_h04": dict(z="z", kernel="uniform", h=0.4),
    "sub_default": dict(z="zb"),
    "sub_h04": dict(z="zb", h=0.4),
}


@pytest.mark.parametrize("key", list(_HTE))
def test_rdhte_matches_r(key, R, hte):
    ref = R["rdhte"][key]
    r = sp.rdhte(hte, y="y", x="x", **_HTE[key])
    mi = r.model_info
    assert _rel(mi["coef"], ref["coef"]) < 1e-9
    assert _rel(mi["coef_bc"], ref["coef_bc"]) < 1e-9
    assert _rel(mi["se_rb"], ref["se_rb"]) < 1e-9
    V = np.asarray(mi["vcov"])
    Vr = np.asarray(ref["vcov"]).reshape(V.shape, order="F")
    assert _rel(np.diag(V), np.diag(Vr)) < 1e-9
    if V.shape[0] > 1 and abs(Vr[0, 1]) > 1e-10:
        assert _rel(V[0, 1], Vr[0, 1]) < 1e-9
    h_ref = np.asarray(ref["h"])  # (level, side) row-major
    h_ours = np.array([v for lv in mi["bandwidth_by_level"].values() for v in lv])
    assert _rel(h_ours, h_ref) < 1e-8
    # Identity: the headline is the average of the detail rows.
    assert r.estimate == pytest.approx(float(r.detail["cate"].mean()), rel=1e-12)


def test_rdbwhte_matches_r(R, hte):
    h = sp.rdbwhte(hte, y="y", x="x", z="z")
    assert _rel([h, h], R["rdhte"]["rdbwhte_cont"]) < 1e-8
    hs = sp.rdbwhte(hte, y="y", x="x", z="zb")
    got = hs[["h_left", "h_right"]].to_numpy().ravel()
    assert _rel(got, R["rdhte"]["rdbwhte_sub"]) < 1e-8


@pytest.mark.parametrize(
    "key,z,L",
    [("lincom_sub", "zb", [-1.0, 1.0]), ("lincom_cont", "z", [1.0, 1.0])],
)
def test_rdhte_lincom_matches_r(key, z, L, R, hte):
    ref = R["rdhte"][key]
    r = sp.rdhte(hte, y="y", x="x", z=z)
    out = sp.rdhte_lincom(r, linfct=L)
    assert _rel(out["estimate"], ref["estimate"]) < 1e-9
    assert _rel(out["z"], ref["z"]) < 1e-9
    assert _rel(out["ci"], ref["ci"]) < 1e-9
    assert _rel(out["joint"]["statistic"], ref["joint"]) < 1e-9
    if ref["p"] > 0:
        assert _rel(out["pvalue"], ref["p"]) < 1e-8


# --------------------------------------------------------------------------- #
#  rd_bias_aware_fuzzy -- R RDHonest (fuzzy)
# --------------------------------------------------------------------------- #
_FRD = {
    "fixed": (dict(M_y=2.0, M_d=0.5, h=0.4), 1e-9),
    "fixed_uniform": (dict(M_y=2.0, M_d=0.5, h=0.4, kernel="uniform"), 1e-9),
    "cluster": (dict(M_y=2.0, M_d=0.5, h=0.4, cluster="g"), 1e-9),
    # Selected h: both sides minimise a flat objective numerically
    # (observed 1.5e-8 on the bandwidth), as for sp.rd_honest.
    "default": ({}, 1e-6),
}


@pytest.fixture(scope="module")
def frd():
    return pd.read_csv(_FIX / "rd_iv_frd.csv")


@pytest.mark.parametrize("key", list(_FRD))
def test_rd_bias_aware_fuzzy_matches_rdhonest(key, R, frd):
    kw, tol = _FRD[key]
    ref = R["rdhonest_fuzzy"][key]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = sp.rd_bias_aware_fuzzy(frd, y="y", x="x", fuzzy="d", **kw)
    b = r.model_info["bias_aware"]
    ours = b["rdhonest"]
    for k in ("estimate", "std.error", "maximum.bias", "conf.low", "conf.high"):
        assert _rel(ours[k], ref[k]) < tol, k
    assert _rel(b["bandwidth"], ref["bandwidth"]) < tol
    assert _rel([b["M_y"], b["M_d"]], [ref["M.rf"], ref["M.fs"]]) < tol
    assert _rel(b["delta_d"], ref["first.stage"]) < tol
    if ref["p.value"] > 1e-300:
        assert _rel(ours["p.value"], ref["p.value"]) < max(tol, 1e-8)


def test_rd_bias_aware_fuzzy_ar_set_solves_its_equation(frd):
    """Identity: at each endpoint |T(t)| = cv(b(t)), with RDHonest's pieces."""
    from statspai.rd._rdhonest import cv_bias

    r = sp.rd_bias_aware_fuzzy(frd, y="y", x="x", fuzzy="d", M_y=2.0, M_d=0.5, h=0.4)
    b = r.model_info["bias_aware"]
    V = np.asarray(b["variance_2x2"])
    for t in b["bias_aware_ci"]:
        se = np.sqrt(V[0] - 2 * t * V[1] + t * t * V[3])
        stat = abs(b["delta_y"] - t * b["delta_d"]) / se
        bias = (b["M_y"] + abs(t) * b["M_d"]) * b["bias_per_unit_M"]
        assert stat == pytest.approx(cv_bias(bias / se, 0.05), rel=1e-9)
    # At the Wald estimate the AR statistic's bias and noise are RDHonest's
    # maximum.bias and std.error scaled by |first stage|.
    hon = b["rdhonest"]
    tau = hon["estimate"]
    se_tau = np.sqrt(V[0] - 2 * tau * V[1] + tau * tau * V[3]) / abs(b["delta_d"])
    assert se_tau == pytest.approx(hon["std.error"], rel=1e-12)


# --------------------------------------------------------------------------- #
#  rdsensitivity / rdrbounds -- R rdlocrand (Monte Carlo: T3)
# --------------------------------------------------------------------------- #
def _mc_close(ours, theirs, reps):
    """Within 4 pooled binomial SEs (both sides draw ``reps`` times)."""
    p = max((ours + theirs) / 2, 1.0 / reps)
    return abs(ours - theirs) <= 4 * np.sqrt(2 * p * (1 - p) / reps)


def test_rdsensitivity_matches_rdlocrand(R):
    ref = R["locrand"]
    lr = pd.read_csv(_FIX / "rd_iv_locrand.csv")
    out = sp.rdsensitivity(
        lr, y="y", x="x", wlist=ref["sens_windows"], n_perms=ref["reps"], seed=1
    )
    assert _rel(out["estimate"], ref["obs_stat"]) < 1e-9
    for ours, theirs in zip(out["pvalue"], ref["sens_pvalues"]):
        assert _mc_close(float(ours), float(theirs), ref["reps"]), (ours, theirs)


def test_rdrbounds_matches_rdlocrand(R):
    ref = R["locrand"]
    lr = pd.read_csv(_FIX / "rd_iv_locrand.csv")
    tab = sp.rdrbounds(
        lr,
        y="y",
        x="x",
        wl=-0.15,
        wr=0.15,
        gamma_list=ref["rbounds_gamma"],
        n_perms=ref["reps"],
        seed=1,
    )
    for ours, theirs in zip(tab["pvalue_upper"], ref["rbounds_upper"]):
        assert _mc_close(float(ours), float(theirs), ref["reps"]), (ours, theirs)
    for ours, theirs in zip(tab["pvalue_lower"], ref["rbounds_lower"]):
        assert _mc_close(float(ours), float(theirs), ref["reps"]), (ours, theirs)
    # The upper bound grows with Gamma (more room for hidden bias).
    assert list(tab["pvalue_upper"]) == sorted(tab["pvalue_upper"])
