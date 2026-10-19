"""sp.binscatter(method='binsreg') against R binsreg 2.2.

R ``binsreg`` (Cattaneo, Crump, Farrell, Feng; GPL-3) was used as a black
box: the numbers below are its outputs on the data generated here, written
to CSV with 17 significant digits and read back by R (2026-10-19, R 4.5,
``binsregselect(y, x[, w], masspoints = "off")`` and ``binsreg(y, x, w,
nbins = 8, ci = c(1, 1), cb = c(1, 1), nsims = 2000, simsgrid = 20)``).

* The number of bins (rule of thumb, its floor, direct plug-in) is an
  integer and must equal R's on all 40 designs.
* Dots and pointwise intervals are deterministic: rtol 1e-10 (the two
  least-squares solvers differ in the last digits only).
* The band's critical value is simulated by both programs (S-level
  screen): R's five seeds gave 3.000-3.040, so ours must fall in
  [2.9, 3.15]. The band's centre and the ratio of its half-width to the
  standard error are deterministic and are checked against R exactly.
"""

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.plots._binsreg import binsreg_knots, binsreg_series, select_nbins


def _design(seed):
    r = np.random.default_rng(seed)
    n = int(r.choice([200, 400, 800, 1500, 3000]))
    kind = seed % 4
    # all four are drawn (in this order) and one is kept
    draws = [r.uniform(0, 1, n), r.normal(size=n), r.exponential(size=n)]
    draws.append(r.beta(2, 5, n))
    x = draws[kind]
    w = r.normal(size=n) + 0.5 * x
    with np.errstate(invalid="ignore"):
        f = {0: np.sin(3 * x), 1: x**2, 2: np.log1p(x)}[seed % 3]
    y = f + 0.5 * w + r.normal(scale=0.3 + 0.2 * (seed % 5), size=n)
    return pd.DataFrame({"x": x, "w": w, "y": y}).dropna()


# seed: (rot_poly, rot, dpi) without w, then with w -- binsregselect
R_SELECT = {
    10: (94, 94, 265, 139, 139, 355),
    11: (4, 8, 5, 5, 8, 7),
    12: (20, 20, 21, 22, 22, 21),
    13: (64, 64, 115, 70, 70, 130),
    14: (7, 8, 7, 8, 8, 8),
    15: (18, 19, 23, 25, 25, 31),
    16: (12, 12, 15, 13, 13, 14),
    17: (22, 22, 37, 23, 23, 41),
    18: (11, 19, 37, 11, 19, 41),
    19: (7, 12, 14, 6, 12, 12),
    20: (15, 19, 21, 19, 19, 21),
    21: (8, 10, 25, 5, 10, 28),
    22: (77, 77, 232, 87, 87, 255),
    23: (6, 8, 7, 6, 8, 7),
    24: (11, 11, 11, 13, 13, 13),
    25: (57, 57, 109, 93, 93, 185),
    26: (22, 22, 39, 20, 20, 28),
    27: (6, 8, 9, 8, 8, 9),
    28: (10, 15, 15, 10, 15, 14),
    29: (23, 23, 36, 23, 23, 38),
}


@pytest.mark.parametrize("seed", sorted(R_SELECT))
def test_imse_number_of_bins_matches_binsregselect(seed):
    d = _design(seed)
    a = select_nbins(d.x.values, d.y.values)
    b = select_nbins(d.x.values, d.y.values, d.w.values)
    got = (a["rot_poly"], a["rot"], a["dpi"], b["rot_poly"], b["rot"], b["dpi"])
    assert a["n_unique"] == len(d)
    assert got == R_SELECT[seed]


# binsreg(y, x, w, nbins = 8, ci = c(1, 1)) on design 12:
# data.dots$x, data.dots$fit, data.ci$ci.l, data.ci$ci.r
R_DOTS = np.array(
    [
        [
            0.057928353346622719,
            0.33201953660150302,
            0.24037638477378387,
            0.41413943272025389,
        ],
        [
            0.18034621997130629,
            0.57083186025007748,
            0.54578705923306436,
            0.70744623792203998,
        ],
        [
            0.30455777567163583,
            1.0003605995081852,
            0.83210309525019033,
            0.99199313703894643,
        ],
        [
            0.43411945506716593,
            1.0447967605256752,
            1.0191445852912362,
            1.172898742882579,
        ],
        [
            0.5636686646475878,
            1.0972833234895634,
            1.0311077311126708,
            1.1909203585272352,
        ],
        [
            0.70072317671374862,
            1.0487950582988783,
            0.91277980787446011,
            1.0677683051257838,
        ],
        [
            0.81507155411994536,
            0.79376162815994711,
            0.72973100660719503,
            0.88513416100342701,
        ],
        [
            0.93895758577501054,
            0.4603788054483568,
            0.38320104826958729,
            0.57262642990777313,
        ],
    ]
)
R_COUNTS = [188, 187, 188, 187, 188, 187, 188, 187]


def test_dots_and_intervals_match_binsreg_with_covariates():
    d = _design(12)
    out = binsreg_series(
        d.x.values, d.y.values, d.w.values, 8, True, False, 0.95, 2000, 20, 0
    )
    dots = out["dots"]
    assert dots["n"].tolist() == R_COUNTS
    got = dots[["x_mean", "y_mean", "ci_lower", "ci_upper"]].to_numpy()
    np.testing.assert_allclose(got, R_DOTS, rtol=1e-10)
    # the first knot is the type-2 quantile: x_(188) of n = 1500
    assert binsreg_knots(d.x.values, 8)[1] == np.sort(d.x.values)[187]


# set.seed(1); binsreg(..., cb = c(1, 1), nsims = 2000, simsgrid = 20):
# cval.by, then data.cb rows 1, 21, 22, 100, 167 (x, cb.l, cb.r)
R_CVAL = 2.9993677295765302
R_BAND = {
    0: (0.0057715130851139426, -0.087242115945805249, 0.40203807743142617),
    20: (0.11507938212344754, 0.31681904143814893, 0.70994592246601251),
    21: (0.12122670156360441, 0.33932904606894609, 0.70876640102514232),
    99: (0.60199321959247121, 0.96783707350619597, 1.2711606536402598),
    166: (0.99395420670540491, -0.035987548635587563, 0.51473990184496676),
}


def test_band_grid_centre_and_width_match_binsreg():
    d = _design(12)
    out = binsreg_series(
        d.x.values, d.y.values, d.w.values, 8, False, True, 0.95, 2000, 20, 0
    )
    band = out["band"]
    assert len(band) == 8 * 20 + 7  # 20 points inside each bin and the knots
    cval = out["cb_critical_value"]
    assert 2.9 <= cval <= 3.15  # simulated: S-level screen against R's 3.00-3.04
    for row, (x, lo, hi) in R_BAND.items():
        assert band["x"].iloc[row] == pytest.approx(x, rel=1e-12)
        assert band["fit"].iloc[row] == pytest.approx((lo + hi) / 2, rel=1e-9)
        se_r = (hi - lo) / 2 / R_CVAL
        se_ours = (band["cb_upper"].iloc[row] - band["fit"].iloc[row]) / cval
        assert se_ours == pytest.approx(se_r, rel=1e-9)


def test_binscatter_binsreg_method_uses_the_imse_bins_and_covariates():
    pytest.importorskip("matplotlib")
    import matplotlib

    matplotlib.use("Agg")
    d = _design(12)
    fig, ax, bins = sp.binscatter(
        d, y="y", x="x", controls=["w"], method="binsreg", ci=True
    )
    info = bins.attrs["binsreg"]
    assert info["n_bins"] == 21 and info["selection"]["dpi"] == 21
    # the dots are the bin coefficients at mean(w), not residualised means
    fixed = binsreg_series(
        d.x.values, d.y.values, d.w.values, 21, True, False, 0.95, 500, 20, 0
    )["dots"]
    np.testing.assert_allclose(bins["y_mean"], fixed["y_mean"], rtol=1e-12)
    import matplotlib.pyplot as plt

    plt.close(fig)


def test_binsreg_dots_recover_a_known_step_function():
    # y is exactly the bin's level plus 2 w: the dots are the levels
    rng = np.random.default_rng(3)
    x = rng.uniform(size=400)
    w = rng.normal(size=400) + x
    knots = binsreg_knots(x, 4)
    level = np.array([1.0, -2.0, 0.5, 3.0])
    idx = np.searchsorted(knots[1:-1], x, side="left")
    y = level[idx] + 2 * w
    out = binsreg_series(x, y, w, 4, False, False, 0.95, 500, 20, 0)
    np.testing.assert_allclose(
        out["dots"]["y_mean"], level + 2 * w.mean(), rtol=1e-10, atol=1e-12
    )


def test_binsreg_method_refuses_what_it_does_not_implement():
    d = _design(12).assign(g=lambda f: (f.x > 0.5).astype(int))
    with pytest.raises(sp.MethodIncompatibility, match="absorb"):
        sp.binscatter(d, y="y", x="x", absorb=["g"], method="binsreg")
    with pytest.raises(sp.MethodIncompatibility, match="weights"):
        sp.binscatter(d, y="y", x="x", weights="w", method="binsreg")


def test_repeated_x_values_warn_that_mass_points_are_not_adjusted():
    rng = np.random.default_rng(0)
    x = rng.integers(0, 40, 600).astype(float)
    y = np.sin(x / 10) + rng.normal(size=600)
    with pytest.warns(sp.AssumptionWarning, match="mass points"):
        sel = select_nbins(x, y)
    assert sel["n_unique"] == len(np.unique(x))
