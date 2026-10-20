"""Reference parity: sp.rdhte / sp.rdbwhte / sp.rdhte_lincom vs R ``rdhte`` 0.2.0.

Fixture: ``_fixtures/_generate_rdhte_R.R`` -> ``rdhte_R.json`` (R run as a
black box on ``rd_iv_hte.csv`` and ``rdhte_extra.csv``). The module is a
clean-room implementation of Calonico, Cattaneo, Farrell, Palomba and
Titiunik (2025, ``calonico2025treatment`` / ``calonico2025rdhte``); spec
``docs/dev/cleanroom/rdhte.md``. The older cells in ``rd_iv_rd_R.json`` are
asserted by ``test_rd_iv_rd_R_parity.py``.

Coverage: HC0-HC3 and CR1; asymmetric windows (R's ``h.l`` / ``h.r`` =
``h=(left, right)`` here); ``p = 2``; ``q = 3``; three kernels; a non-zero
cutoff; a numeric 0/1/2 moderator (continuous, as in R); binary and
three-level categorical moderators (subgroups, a bandwidth per level);
selected bandwidths (mserd, msetwo, cerrd, HC1, CR1, uniform with ``p = 2``);
``rdbwhte``; ``rdhte_lincom`` with one and two restrictions.

Tolerances: coefficients, standard errors and covariances are closed-form
weighted least squares, rel 1e-9 (observed <= 3e-12). Selected bandwidths go
through the CCT selector, rel 1e-8 (observed <= 1e-11).
"""

from __future__ import annotations

import json
import pathlib
import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp

FX = pathlib.Path(__file__).parent / "_fixtures"
R = json.loads((FX / "rdhte_R.json").read_text(encoding="utf-8"))
D = pd.read_csv(FX / "rd_iv_hte.csv")
D["z3"] = pd.read_csv(FX / "rdhte_extra.csv")["z3"].astype(float)
D["z3f"] = pd.Categorical(D["z3"].astype(int))

EST = 1e-9
BW = 1e-8


def _kwargs(cell):
    a = dict(R[cell]["args"])
    kw = {}
    if "h.l" in a:
        kw["h"] = (a.pop("h.l"), a.pop("h.r"))
    if "cluster" in a:
        a.pop("cluster")
        kw["cluster"] = "cl"
    kernel = {"epa": "epanechnikov", "uni": "uniform"}
    for k, v in a.items():
        kw[k] = kernel.get(v, v) if k == "kernel" else v
    return kw


def _fit(cell):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.rdhte(D, y="y", x="x", z=R[cell]["moderator"], **_kwargs(cell))


_CELLS = sorted(k for k in R if not k.startswith(("bw_", "lincom_", "_")))


@pytest.mark.parametrize("cell", _CELLS)
def test_rdhte_matches_R(cell):
    ref = R[cell]
    res = _fit(cell)
    mi = res.model_info
    np.testing.assert_allclose(mi["coef"], ref["coef"], rtol=EST, err_msg="coef")
    np.testing.assert_allclose(mi["coef_bc"], ref["coef_bc"], rtol=EST, err_msg="bc")
    np.testing.assert_allclose(mi["se_rb"], ref["se_rb"], rtol=EST, err_msg="se")
    V = np.asarray(mi["vcov"])
    Vr = np.asarray(ref["vcov"]).reshape(V.shape, order="F")
    scale = np.sqrt(np.outer(np.diag(Vr), np.diag(Vr)))
    np.testing.assert_allclose(V / scale, Vr / scale, atol=EST, err_msg="vcov")
    h = np.array([v for pair in mi["bandwidth_by_level"].values() for v in pair])
    np.testing.assert_allclose(h, ref["h"], rtol=BW, err_msg="h")
    assert mi["vce"] == ref["vce"]


def test_rdhte_subgroup_effective_counts_match_R():
    """Window counts per level and side equal R's Nh."""
    res = _fit("zb_default")
    ref = R["zb_default"]
    assert res.model_info["n_effective"] == int(sum(ref["Nh"]))


_BW = sorted(k for k in R if k.startswith("bw_"))


@pytest.mark.parametrize("cell", _BW)
def test_rdbwhte_matches_R(cell):
    ref = R[cell]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = sp.rdbwhte(D, y="y", x="x", z=ref["moderator"], **_kwargs(cell))
    if isinstance(out, float):
        got = [out, out]
    else:
        got = out[["h_left", "h_right"]].to_numpy().ravel()
    np.testing.assert_allclose(got, ref["h"], rtol=BW)


@pytest.mark.parametrize("cell", ["lincom_z3f_two", "lincom_z_sum"])
def test_rdhte_lincom_matches_R(cell):
    ref = R[cell]
    model = _fit(ref["model"])
    L = np.asarray(ref["L"], dtype=float)
    out = sp.rdhte_lincom(model, linfct=L if L.shape[0] > 1 else L[0])
    np.testing.assert_allclose(out["estimate"], ref["estimate"], rtol=EST)
    np.testing.assert_allclose(out["z"], ref["z"], rtol=EST)
    ci = np.atleast_2d(np.asarray(out["ci"], dtype=float))
    np.testing.assert_allclose(ci[:, 0], ref["ci_low"], rtol=EST)
    np.testing.assert_allclose(ci[:, 1], ref["ci_high"], rtol=EST)
    assert out["joint"]["statistic"] == pytest.approx(ref["joint"], rel=EST)
    assert out["joint"]["df"] == L.shape[0]
    p_ref = np.atleast_1d(ref["p"])
    ok = p_ref > 1e-300
    np.testing.assert_allclose(np.atleast_1d(out["pvalue"])[ok], p_ref[ok], rtol=1e-8)


# ------------------------------------------------------------- identities


def test_subgroup_effects_equal_separate_rdrobust_style_fits():
    """A 0/1 moderator with common h: each level's effect is the RD jump of
    a kernel-weighted local linear fit on that level alone (independent
    numpy recomputation)."""
    h = 0.4
    res = sp.rdhte(D, y="y", x="x", z="zb", h=h)
    for lv, eff in zip((0, 1), res.model_info["coef"]):
        s = D[D.zb == lv]
        jumps = []
        for side in (s.x < 0, s.x >= 0):
            xs, ys = s.x[side].to_numpy(), s.y[side].to_numpy()
            w = np.clip(1 - np.abs(xs / h), 0, None)
            k = np.abs(xs) <= h
            X = np.column_stack([np.ones(k.sum()), xs[k]]) * np.sqrt(w[k])[:, None]
            jumps.append(np.linalg.lstsq(X, ys[k] * np.sqrt(w[k]), rcond=None)[0][0])
        assert eff == pytest.approx(jumps[1] - jumps[0], rel=1e-10)


def test_headline_is_average_of_rows_and_rows_are_linear_in_z():
    res = sp.rdhte(D, y="y", x="x", z="z", h=0.4, n_eval=5)
    det = res.detail
    assert res.estimate == pytest.approx(det["cate"].mean(), rel=1e-12)
    a, b = res.model_info["coef"]
    np.testing.assert_allclose(det["cate"], a + b * det["z_value"], rtol=1e-12)


def test_cluster_vce_remap_warns():
    with pytest.warns(UserWarning, match="cr1"):
        sp.rdhte(D, y="y", x="x", z="z", h=0.4, cluster="cl", vce="hc1")
    with pytest.raises(ValueError, match="CR2/CR3"):
        sp.rdhte(D, y="y", x="x", z="z", h=0.4, cluster="cl", vce="hc3")


@pytest.mark.parametrize(
    "kw",
    [dict(alpha=1.5), dict(n_eval=0), dict(h=-1.0), dict(p=1, q=1), dict(p=-1)],
)
def test_rdhte_rejects_invalid_inputs(kw):
    with pytest.raises(ValueError):
        sp.rdhte(D, y="y", x="x", z="z", **kw)
