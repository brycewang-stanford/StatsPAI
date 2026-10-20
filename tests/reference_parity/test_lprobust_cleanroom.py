"""Clean-room rewrite of ``sp.lprobust_at_point`` / ``sp.lpbwselect_*`` (H5).

Two kinds of evidence, both independent of the replaced code:

* **R black box.** ``_fixtures/_generate_lprobust_selectors_R.R`` runs the
  exported functions of R ``nprobust`` 1.0.0 (GPL-3, never read) on the
  committed CSV and records their outputs: every selector at interior
  points, three polynomial orders, four kernels, two ``bwcheck`` /
  ``bwregul`` settings, and the estimator at supplied bandwidths. The
  existing Stata fixture (``test_lprobust_parity.py``) stays the pin at the
  boundary.
* **Identities.** Exactness of the bias-corrected estimator on polynomials,
  the Nadaraya-Watson closed form, the hand-computed nearest-neighbour
  variance with ties, and the closed form behind ``mse-dpi``.

Tolerance: relative 1e-9; the rule-of-thumb selectors at ``p = 3`` get
1e-8. Those fit a degree-6 global polynomial, and a 50-digit evaluation of
the same formula (``docs/dev/cleanroom/lprobust.md``) shows the reference's
own rounding there reaches 4e-9 while this implementation is within 1e-12
of the high-precision value.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import DataInsufficient, MethodIncompatibility
from statspai.nonparametric import _lprobust_bw as bw

_FIX = pathlib.Path(__file__).resolve().parent / "_fixtures"
_DATA = pd.read_csv(_FIX / "lprobust_selectors_data.csv")
_R = json.loads((_FIX / "lprobust_selectors_R.json").read_text(encoding="utf-8"))


def _xy(design):
    s = _DATA[_DATA["design"] == design]
    return s["x"].to_numpy(float), s["y"].to_numpy(float)


def _rtol(selector, p):
    return 1e-8 if selector.endswith("rot") and p == 3 else 1e-9


def _case_id(r):
    keys = (
        "design",
        "selector",
        "kernel",
        "eval",
        "p",
        "deriv",
        "bwcheck",
        "bwregul",
        "imsegrid",
    )
    return "-".join(str(r[k]) for k in keys if k in r)


def test_fixture_is_the_black_box_output():
    assert _R["meta"]["nprobust"] == "1.0.0"
    assert len(_R["pointwise"]) == 342 and len(_R["global"]) == 66
    assert len(_R["at_point"]) == 192


@pytest.mark.parametrize("row", _R["pointwise"], ids=_case_id)
def test_pointwise_selectors_match_r(row):
    x, y = _xy(row["design"])
    kw = dict(kernel=row["kernel"], p=row["p"], deriv=row["deriv"])
    if row["selector"] == "mse-rot":
        got = sp.lpbwselect_mse_rot(x, y, row["eval"], **kw)
    else:
        fn = (
            sp.lpbwselect_mse_dpi
            if row["selector"] == "mse-dpi"
            else sp.lpbwselect_ce_rot
        )
        got = fn(
            x, y, row["eval"], bwcheck=row["bwcheck"], bwregul=row["bwregul"], **kw
        )
    rtol = _rtol(row["selector"], row["p"])
    assert got["h"] == pytest.approx(row["h"], rel=rtol)
    if row["b"] is not None:
        assert got["b"] == pytest.approx(row["b"], rel=rtol)


@pytest.mark.parametrize("row", _R["global"], ids=_case_id)
def test_integrated_selectors_match_r(row):
    x, y = _xy(row["design"])
    kw = dict(kernel=row["kernel"], p=row["p"], deriv=row["deriv"])
    if row["selector"] == "imse-dpi":
        got = sp.lpbwselect_imse_dpi(x, y, imsegrid=row["imsegrid"], **kw)
        assert got["b"] == pytest.approx(row["b"], rel=1e-9)
    else:
        got = sp.lpbwselect_imse_rot(x, y, **kw)
    assert got["h"] == pytest.approx(row["h"], rel=_rtol(row["selector"], row["p"]))


@pytest.mark.parametrize("row", _R["at_point"], ids=_case_id)
def test_estimator_at_supplied_bandwidths_matches_r(row):
    x, y = _xy(row["design"])
    got = sp.lprobust_at_point(
        x,
        y,
        row["eval"],
        h=row["h"],
        b=row["b"],
        kernel=row["kernel"],
        p=row["p"],
        deriv=row["deriv"],
        n_neighbors=row["nnmatch"],
    )
    assert got.n_eff == row["N"]
    for field in ("tau_us", "tau_bc", "se_us", "se_rb"):
        assert getattr(got, field) == pytest.approx(row[field], rel=1e-9, abs=1e-13)


# ----------------------------------------------------------------------
# Identities
# ----------------------------------------------------------------------


@pytest.mark.parametrize("p,deriv", [(0, 0), (1, 0), (1, 1), (2, 1), (3, 0)])
@pytest.mark.parametrize("x0", [0.0, 0.9])
def test_bias_corrected_estimate_is_exact_on_degree_p_plus_1(p, deriv, x0):
    """beta_bc removes the (p+1)-th order term exactly, so a noiseless
    polynomial of degree p+1 is recovered without error at any point."""
    rng = np.random.default_rng(3)
    x = rng.uniform(0.0, 2.0, 400)
    coef = np.array([1.0, -0.7, 0.4, 0.25, -0.1])[: p + 2]
    y = np.polynomial.polynomial.polyval(x, coef)
    truth = np.polynomial.polynomial.polyval(
        x0, np.polynomial.polynomial.polyder(coef, deriv) if deriv else coef
    )
    fit = sp.lprobust_at_point(x, y, x0, h=0.7, b=0.9, p=p, deriv=deriv)
    assert fit.tau_bc == pytest.approx(truth, abs=1e-9)


def test_conventional_estimate_is_exact_on_degree_p():
    x = np.linspace(-1.0, 1.0, 101)
    y = 2.0 - 3.0 * x
    fit = sp.lprobust_at_point(x, y, 0.3, h=0.5, p=1)
    assert fit.tau_us == pytest.approx(2.0 - 0.9, abs=1e-12)


def test_local_constant_uniform_is_the_window_mean():
    """p = 0 with the uniform kernel is the average of y over |x - x0| <= h,
    and its variance is sum(v_i) / n_h^2."""
    rng = np.random.default_rng(8)
    x = rng.uniform(0, 1, 200)
    y = rng.normal(size=200)
    x0, h = 0.5, 0.2
    fit = sp.lprobust_at_point(x, y, x0, h=h, b=h, p=0, kernel="uniform")
    inside = np.abs(x - x0) <= h
    order = np.argsort(x[inside], kind="stable")
    xs, ys = x[inside][order], y[inside][order]
    assert fit.tau_us == pytest.approx(ys.mean(), abs=1e-13)
    v = bw.nn_variance(xs, ys, 3)
    assert fit.se_us == pytest.approx(np.sqrt(v.sum()) / xs.size, rel=1e-12)


def test_nearest_neighbour_variance_matches_ties_whole():
    """Hand computation of v_i = J/(J+1) (y_i - mean of matched y)^2 where
    the match set is every row within the n_neighbors-th smallest distance."""
    x = np.array([0.0, 1.0, 2.0, 2.0, 5.0])
    y = np.array([1.0, 2.0, 4.0, 6.0, 3.0])
    v = bw.nn_variance(x, y, 3)
    # x = 0: neighbours at distance 1, 2, 2 -> mean (2 + 4 + 6) / 3 = 4
    assert v[0] == pytest.approx(3 / 4 * (1.0 - 4.0) ** 2)
    # x = 1: distances 1 (x=0), 1 (x=2), 1 (x=2), 4 -> three tied at 1
    assert v[1] == pytest.approx(3 / 4 * (2.0 - (1.0 + 4.0 + 6.0) / 3) ** 2)
    # x = 2 (first): 0 (other 2), 1 (x=1), 2 (x=0) -> J = 3
    assert v[2] == pytest.approx(3 / 4 * (4.0 - (6.0 + 2.0 + 1.0) / 3) ** 2)
    # with n_neighbors = 1 at x = 1 the three rows tied at distance 1 enter
    v1 = bw.nn_variance(x, y, 1)
    assert v1[1] == pytest.approx(3 / 4 * (2.0 - (1.0 + 4.0 + 6.0) / 3) ** 2)


def test_mse_dpi_is_its_closed_form():
    """With bwregul = 0, h = (V_h / (n B_h))^(1/(2p+3)) whenever neither
    the cap nor the bwcheck floor binds, and b is the same closed form
    capped at the largest distance to the data."""
    x, y = _xy("smooth")
    got = sp.lpbwselect_mse_dpi(x, y, 0.4, bwregul=0.0)
    assert got["h"] == pytest.approx(
        (got["V_h"] / (x.size * got["B_h"])) ** 0.2, rel=1e-12
    )
    # here the stage-2 closed form exceeds the largest distance to the data,
    # so b is capped there
    far = np.max(np.abs(x - 0.4))
    closed_b = (got["V_b"] / (x.size * got["B_b"])) ** (1 / 7)
    assert closed_b > far
    assert got["b"] == pytest.approx(far, rel=1e-15)


def test_bwcheck_equal_to_n_pins_every_bandwidth_at_the_farthest_point():
    """The floor (distance to the n-th nearest point) equals the cap."""
    x, y = _xy("smooth")
    got = sp.lpbwselect_mse_dpi(x, y, 0.4, bwcheck=x.size)
    far = np.max(np.abs(x - 0.4))
    assert got["h"] == pytest.approx(far) and got["b"] == pytest.approx(far)


def test_ce_rot_floor_acts_before_rescaling():
    x, y = _xy("smooth")
    n = x.size
    mse = sp.lpbwselect_mse_dpi(x, y, 0.4, bwcheck=200)
    ce = sp.lpbwselect_ce_rot(x, y, 0.4, bwcheck=200)
    assert ce["h"] == pytest.approx(mse["h"] * n ** (-1 / 20), rel=1e-14)


@pytest.mark.parametrize("bad", [0, -3, 2.5])
def test_bwcheck_must_be_a_positive_integer(bad):
    x, y = _xy("smooth")
    with pytest.raises(MethodIncompatibility, match="bwcheck"):
        sp.lpbwselect_mse_dpi(x, y, 0.4, bwcheck=bad)


def test_too_few_distinct_values_in_the_window_raise():
    x = np.repeat(np.arange(6.0), 10)
    y = np.sin(x) + np.linspace(0, 1, x.size)
    with pytest.raises(MethodIncompatibility, match="singular"):
        sp.lprobust_at_point(x, y, 0.0, h=1.0, p=1)  # epa: only x = 0 has weight


def test_window_count_rule():
    """A p-th order fit with bias correction needs more than p + 2 rows."""
    x = np.array([0.0, 0.1, 0.2, 0.3, 5.0, 6.0])
    y = np.array([0.0, 1.0, 0.5, 2.0, 3.0, 1.0])
    with pytest.raises(DataInsufficient, match="fall inside the bandwidth"):
        sp.lprobust_at_point(x, y, 0.0, h=0.25, p=1)  # 3 rows have weight
    assert sp.lprobust_at_point(x, y, 0.0, h=0.35, p=1).n_eff == 4
