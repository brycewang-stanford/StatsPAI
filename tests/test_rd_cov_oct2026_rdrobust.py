"""``statspai.rd.rdrobust`` and ``_rdplot_core`` branches not reached elsewhere.

* the legacy plug-in bandwidth selector's eight ``bwselect`` rules, as
  identities between them (it is the documented fallback when the CCT
  selector fails, so its rules must stay mutually consistent);
* observation weights: validation, and the donut filter applied to them;
* the plotting numbers behind ``sp.rdplot`` / ``sp.rdplotdensity`` against
  independent least-squares computations written out here;
* the studentised bootstrap's refusal paths.
"""

import sys
import warnings

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from scipy import stats  # noqa: E402

import statspai as sp  # noqa: E402
from statspai.exceptions import (  # noqa: E402
    ConvergenceFailure,
    DataInsufficient,
    MethodIncompatibility,
)
from statspai.rd import _rdplot_core as core  # noqa: E402
from statspai.rd import rdrobust as rr  # noqa: E402


@pytest.fixture(scope="module")
def df():
    rng = np.random.default_rng(1)
    n = 500
    x = rng.uniform(-1, 1, n)
    curve = np.where(x < 0, 3 * x**2, -2 * x**2)
    y = 2.0 * (x >= 0) + 0.5 * x + curve + rng.normal(0, 0.3, n)
    return pd.DataFrame(
        {
            "y": y,
            "x": x,
            "w": rng.uniform(0.5, 1.5, n),
            "cl": rng.integers(0, 30, n),
            "z": rng.normal(size=n),
        }
    )


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


# ------------------------------------------------------------ legacy selector


def test_legacy_selector_rules_are_consistent_with_each_other(df):
    y, x = df["y"].to_numpy(), df["x"].to_numpy()
    left, right = x < 0, x >= 0

    def bw(rule):
        return rr._select_bandwidth(y, x, left, right, 1, "triangular", rule)

    common = bw("mserd")
    h_l, h_r = bw("msetwo")
    # Curvature differs by side in this DGP (+6 left, -4 right), so the three
    # MSE bandwidths are distinct and the combination rules can be told apart.
    assert len({round(v, 12) for v in (common, h_l, h_r)}) == 3
    assert bw("msecomb1") == min(common, h_l, h_r)
    assert bw("msecomb2") == float(np.median([common, h_l, h_r]))

    # CER = MSE x n^{-1/((2p+3)(2p+5))}; for p = 1 the exponent is 1/35.
    n = len(y)
    cer = n ** (-1.0 / 35.0)
    assert rr._cer_factor(n, 1) == pytest.approx(cer, rel=1e-14)
    assert rr._cer_factor(n, 2) == pytest.approx(n ** (-1.0 / 63.0), rel=1e-14)
    assert rr._cer_factor(1, 1) == 1.0  # no shrinkage is defined for n <= 1
    assert bw("cerrd") == pytest.approx(common * cer, rel=1e-14)
    assert bw("certwo") == pytest.approx((h_l * cer, h_r * cer), rel=1e-14)
    assert bw("cercomb1") == pytest.approx(min(common, h_l, h_r) * cer, rel=1e-14)
    assert bw("cercomb2") == pytest.approx(
        float(np.median([common, h_l, h_r])) * cer, rel=1e-14
    )
    # an unrecognised rule falls through to the common MSE bandwidth
    assert bw("not-a-rule") == common
    # n_total defaults to the sample size
    assert (
        rr._select_bandwidth(y, x, left, right, 1, "triangular", "mserd", n) == common
    )


def test_legacy_selector_pilot_pieces_on_thin_windows():
    y = np.array([1.0, 2.0, 4.0])
    x = np.array([0.1, 0.2, 0.3])
    # fewer than five points in the window: the unconditional variance
    assert rr._local_residual_var(y, x, 1.0, "triangular") == np.var(y)
    assert rr._local_residual_var(np.array([]), np.array([]), 1.0, "triangular") == 1.0
    # fewer than six: no curvature estimate, which keeps the pilot bandwidth
    assert (
        rr._estimate_second_deriv(np.arange(5.0), np.arange(5.0) / 10, 1.0, "uniform")
        == 0.0
    )
    # an exact quadratic is recovered: m''(0) = 2 * 1.5
    xs = np.linspace(0.01, 0.9, 40)
    got = rr._estimate_second_deriv(1 + 2 * xs + 1.5 * xs**2, xs, 1.0, "uniform")
    # exact fit up to the conditioning of a cubic Vandermonde on [0, 0.9]
    assert got == pytest.approx(3.0, abs=1e-8)


def test_one_support_point_a_side_is_refused_before_bandwidth_selection():
    # One support point per side: a local line is not identified. This input
    # used to make the CCT selector divide by zero, fall back to the legacy
    # bandwidth and return -1.0 (p = 0) where the two cell means differ by +1.
    rng = np.random.default_rng(1)
    x = np.repeat([-1.0, 1.0], 100)
    y = 1.0 * (x >= 0) + rng.normal(0, 0.3, x.size)
    frame = pd.DataFrame({"y": y, "x": x})
    with pytest.raises(DataInsufficient, match="too few distinct values"):
        sp.rdrobust(
            frame, y="y", x="x", warn_mass_points=False, manipulation_test=False
        )


# ------------------------------------------------------------ weights


def test_weights_column_is_validated(df):
    with pytest.raises(MethodIncompatibility, match="weights column 'nope' not found"):
        sp.rdrobust(df, y="y", x="x", weights="nope")
    bad = df.copy()
    bad.loc[3, "w"] = np.inf
    with pytest.raises(MethodIncompatibility, match="must be finite"):
        sp.rdrobust(bad, y="y", x="x", weights="w")


@pytest.mark.parametrize("bwselect", ["mserd", "cct"])
def test_donut_filters_the_weights_with_the_rows(df, bwselect):
    if bwselect == "cct":
        pytest.importorskip("rdrobust")
    kw = dict(y="y", x="x", weights="w", bwselect=bwselect, manipulation_test=False)
    donut = sp.rdrobust(df, donut=0.05, **kw)
    dropped = sp.rdrobust(df[np.abs(df["x"]) > 0.05], **kw)
    # Same rows, same weights: the same arithmetic, so exact equality.
    assert donut.estimate == dropped.estimate
    assert donut.se == dropped.se
    unweighted = sp.rdrobust(
        df, y="y", x="x", donut=0.05, bwselect=bwselect, manipulation_test=False
    )
    assert donut.estimate != unweighted.estimate


def test_column_list_coercion_refuses_an_empty_required_list():
    assert rr._coerce_column_list("a", "covs") == ["a"]
    assert rr._coerce_column_list(("a", "b"), "covs") == ["a", "b"]
    assert rr._coerce_column_list([], "covs", allow_empty=True) == []
    with pytest.raises(MethodIncompatibility, match="at least one column name"):
        rr._coerce_column_list([], "covs")


def test_parse_data_mask_is_optional(df):
    holed = df.copy()
    holed.loc[[0, 7], "y"] = np.nan
    four = rr._parse_data(holed, "y", "x", 0.25, None, ["z"])
    five = rr._parse_data(holed, "y", "x", 0.25, None, ["z"], return_mask=True)
    assert len(four) == 4 and len(five) == 5
    for a, b in zip(four, five[:4]):
        np.testing.assert_array_equal(a, b)
    assert five[4].sum() == len(df) - 2
    # centred at the cutoff; covariates demeaned
    np.testing.assert_allclose(four[1], holed["x"][five[4]] - 0.25)
    assert abs(four[3].mean()) < 1e-12


# ------------------------------------------------------------ manipulation test


def test_a_density_test_that_cannot_run_is_recorded_not_raised():
    rng = np.random.default_rng(1)
    x = np.r_[rng.uniform(-1, 0, 6), rng.uniform(0, 1, 6)]
    y = 1.0 * (x >= 0) + 0.2 * x + rng.normal(0, 0.3, x.size)
    res = sp.rdrobust(pd.DataFrame({"y": y, "x": x}), y="y", x="x", h=2.5)
    mc = res.model_info["mccrary"]
    assert mc["pvalue"] is None
    assert mc["test"] == "rddensity"
    assert mc["error"] == "DataInsufficient"
    assert np.isfinite(res.estimate)


# ------------------------------------------------------------ bootstrap


def _boot_kwargs(y, x, **over):
    kw = dict(
        Y=y,
        X_c=x,
        D=None,
        Z=None,
        left=x < 0,
        right=x >= 0,
        h=0.5,
        b=0.5,
        p=1,
        q=2,
        kernel="triangular",
        deriv=0,
        cluster_vals=None,
        alpha=0.05,
        n_boot=99,
        random_state=0,
        tau_bc=0.0,
        se_robust=1.0,
    )
    kw.update(over)
    return kw


def test_rbc_bootstrap_refuses_an_empty_window_and_degenerate_replicates():
    rng = np.random.default_rng(1)
    x = rng.uniform(-1, 1, 200)
    y = 1.0 * (x >= 0) + rng.normal(0, 0.3, 200)
    with pytest.raises(DataInsufficient, match="too few observations") as err:
        rr._rbc_bootstrap(**_boot_kwargs(y, x, h=1e-4, b=1e-4))
    assert err.value.diagnostics["minimum_per_side"] == 3

    # A constant outcome has zero (or rounding-noise) residual variance, so
    # some replicates have no usable studentised statistic; with n_boot=99
    # every one is needed.
    with pytest.raises(ConvergenceFailure, match=r"only produced \d+/99 valid"):
        rr._rbc_bootstrap(**_boot_kwargs(np.ones(200), x))

    out = rr._rbc_bootstrap(**_boot_kwargs(y, x, tau_bc=1.0, se_robust=0.1))
    lo, hi = out["ci"]
    q_lo, q_hi = out["quantiles"]
    # CI = tau - q_{1-a/2} se, tau - q_{a/2} se: the percentile-t inversion
    assert lo == pytest.approx(1.0 - q_hi * 0.1, rel=1e-12)
    assert hi == pytest.approx(1.0 - q_lo * 0.1, rel=1e-12)
    assert out["_n_ok"] == 99


# ------------------------------------------------------------ rdplot


def test_rdplot_refuses_an_unknown_kernel(df):
    with pytest.raises(MethodIncompatibility, match="kernel must be 'uniform'"):
        sp.rdplot(df, y="y", x="x", kernel="gaussian")


@pytest.mark.parametrize("fn", ["rdplot", "rdplotdensity"])
def test_plots_name_the_missing_dependency(df, fn, monkeypatch):
    # Simulates an environment without matplotlib; no numerical path involved.
    monkeypatch.setitem(sys.modules, "matplotlib.pyplot", None)
    with pytest.raises(ImportError, match="matplotlib required"):
        if fn == "rdplot":
            sp.rdplot(df, y="y", x="x")
        else:
            sp.rdplotdensity(df, x="x")


def test_rdplot_weights_enter_the_shaded_band(df):
    fig_w, ax_w = sp.rdplot(df, y="y", x="x", weights="w", kernel="triangular", p=2)
    fig_u, ax_u = sp.rdplot(df, y="y", x="x", kernel="triangular", p=2)
    # two shaded bands (left, right) + the error-bar line collection
    bands_w = [c for c in ax_w.collections if hasattr(c, "get_paths")][:2]
    bands_u = [c for c in ax_u.collections if hasattr(c, "get_paths")][:2]
    vw = bands_w[0].get_paths()[0].vertices
    vu = bands_u[0].get_paths()[0].vertices
    assert vw.shape == vu.shape
    assert not np.allclose(vw, vu)
    # ... and the plotted polynomial is the weighted one
    assert not np.allclose(
        fig_w.rdplot_data["vars_poly"]["rdplot_y"],
        fig_u.rdplot_data["vars_poly"]["rdplot_y"],
    )

    # the band is the weighted polynomial fit +/- z * se on the plotting grid
    X, Y, W = df["x"].to_numpy(), df["y"].to_numpy(), df["w"].to_numpy()
    left = X < 0
    hl = fig_w.rdplot_data["h"][0]
    k = np.clip(1 - np.abs(X[left] / hl), 0, None) / hl * W[left]
    grid = fig_w.rdplot_data["vars_poly"]["rdplot_x"]
    grid_l = grid[: len(grid) // 2]
    fit, lo, hi = rr._weighted_poly_fit_ci(X[left], Y[left], 2, grid_l, 0.95, k)
    coef = np.polyfit(X[left], Y[left], 2, w=np.sqrt(k))
    # same weighted least-squares problem solved by a different routine
    np.testing.assert_allclose(fit, np.polyval(coef, grid_l), rtol=1e-8)
    np.testing.assert_allclose(vw[:, 1].min(), lo.min(), rtol=1e-10)
    np.testing.assert_allclose(vw[:, 1].max(), hi.max(), rtol=1e-10)


def test_rdplot_shades_bandwidth_and_donut_with_labels(df):
    fig, ax = sp.rdplot(df, y="y", x="x", show_bw=True, donut=0.1)
    labels = ax.get_legend_handles_labels()[1]
    h = sp.rdrobust(df, y="y", x="x", p=1, manipulation_test=False).model_info[
        "bandwidth_h"
    ]
    assert labels == [f"Bandwidth h = {h:.3f}", "Donut ±0.1"]
    spans = [p for p in ax.patches if p.get_label() in labels]
    widths = sorted(p.get_width() for p in spans)
    assert widths == pytest.approx([0.2, 2 * h], rel=1e-9)


def test_unweighted_polynomial_band_matches_the_classical_ols_formula():
    rng = np.random.default_rng(5)
    xv = rng.uniform(0, 1, 60)
    yv = 1 + 2 * xv - xv**2 + rng.normal(0, 0.2, 60)
    grid = np.linspace(0, 1, 7)
    fit, lo, hi = rr._weighted_poly_fit_ci(xv, yv, 2, grid, 0.9)
    coef, cov = np.polyfit(xv, yv, 2, cov=True)  # cov scaled by RSS / (n - 3)
    V = np.vander(grid, 3)
    se = np.sqrt(np.einsum("ij,jk,ik->i", V, cov, V))
    z = stats.norm.ppf(0.95)
    # two solvers for the same normal equations
    np.testing.assert_allclose(fit, np.polyval(coef, grid), rtol=1e-9)
    np.testing.assert_allclose(lo, fit - z * se, rtol=1e-7)
    np.testing.assert_allclose(hi, fit + z * se, rtol=1e-7)

    # fewer than three points: no fit, and NaN rather than a made-up band
    out = rr._weighted_poly_fit_ci(xv[:2], yv[:2], 2, grid, 0.9)
    assert all(np.isnan(a).all() and a.shape == grid.shape for a in out)


def test_rdplotdensity_takes_one_bandwidth_per_side(df):
    fig_pair, _ = sp.rdplotdensity(df, x="x", h=(0.3, 0.3))
    fig_one, _ = sp.rdplotdensity(df, x="x", h=0.3)
    for side in ("Estl", "Estr"):
        np.testing.assert_array_equal(
            fig_pair.rdplotdensity_data[side]["f_p"],
            fig_one.rdplotdensity_data[side]["f_p"],
        )
    fig_two, _ = sp.rdplotdensity(df, x="x", h=(0.3, 0.5))
    np.testing.assert_array_equal(
        fig_two.rdplotdensity_data["Estl"]["f_p"],
        fig_one.rdplotdensity_data["Estl"]["f_p"],
    )
    assert np.all(fig_two.rdplotdensity_data["Estr"]["bw"] == 0.5)
    # U(-1, 1): density 1/2 at the cutoff from the left. Local-quadratic
    # density estimate with n = 259 on that side and h = 0.3; its own SE is
    # about 0.1, so 0.25 is a 2.5-SE band.
    assert fig_one.rdplotdensity_data["Estl"]["f_p"][-1] == pytest.approx(0.5, abs=0.25)


# ------------------------------------------------------------ rdplot numbers


def test_rdplot_numbers_validation(df):
    y, x = df["y"].to_numpy(), df["x"].to_numpy()
    with pytest.raises(MethodIncompatibility, match="binselect must be one of"):
        core.rdplot_numbers(y, x, binselect="zz")
    with pytest.raises(MethodIncompatibility, match="within the range of x"):
        core.rdplot_numbers(y, x, c=5.0)


def test_rdplot_numbers_scale_multiplies_the_selected_bins(df):
    y, x = df["y"].to_numpy(), df["x"].to_numpy()
    base = core.rdplot_numbers(y, x)
    both = core.rdplot_numbers(y, x, scale=2)
    each = core.rdplot_numbers(y, x, scale=(2, 3))
    j_l, j_r = base["J"]
    assert both["J"] == (2 * j_l, 2 * j_r)
    assert each["J"] == (2 * j_l, 3 * j_r)
    assert len(each["vars_bins"]["rdplot_mean_bin"]) <= 2 * j_l + 3 * j_r


def test_rdplot_numbers_side_specific_support_and_kernels(df):
    y, x = df["y"].to_numpy(), df["x"].to_numpy()
    res = core.rdplot_numbers(y, x, h=(0.5, 0.7), kernel="epanechnikov", p=2)
    assert res["h"] == (0.5, 0.7)
    assert res["N_h"] == (
        int(((x < 0) & (x >= -0.5)).sum()),
        int(((x >= 0) & (x <= 0.7)).sum()),
    )
    # the polynomial is the Epanechnikov-weighted quadratic on that support
    m = (x < 0) & (x >= -0.5)
    k = 0.75 * (1 - (x[m] / 0.5) ** 2) / 0.5
    np.testing.assert_allclose(core._kweight(x[m], 0.0, 0.5, "epa"), k, rtol=1e-14)
    coef = np.polyfit(x[m], y[m], 2, w=np.sqrt(k))[::-1]
    # rdplot_numbers solves by Cholesky, polyfit by SVD: 1e-8 relative
    np.testing.assert_allclose(np.asarray(res["coef"])[:, 0], coef, rtol=1e-8)


def test_rdplot_numbers_one_covariate_as_a_vector(df):
    y, x, z = (df[c].to_numpy() for c in ("y", "x", "z"))
    vec = core.rdplot_numbers(y, x, covs=z)
    mat = core.rdplot_numbers(y, x, covs=z[:, None])
    np.testing.assert_array_equal(vec["coef_covs"], mat["coef_covs"])
    np.testing.assert_array_equal(
        vec["vars_poly"]["rdplot_y"], mat["vars_poly"]["rdplot_y"]
    )
    # z is independent noise: its partial coefficient is near zero
    # (se about 0.3 / sqrt(500) = 0.013 times the residual scale; 0.1 is loose)
    assert abs(float(np.ravel(vec["coef_covs"])[0])) < 0.1


def test_rdplot_numbers_mass_points_switch_the_bin_rule():
    rng = np.random.default_rng(3)
    x = rng.integers(-10, 10, 400) / 10.0 + 0.05  # 20 support points
    y = 1.0 * (x >= 0) + x + rng.normal(0, 0.2, 400)
    adj = core.rdplot_numbers(y, x, binselect="es", masspoints="adjust")
    assert adj["mass_points_adjusted"] is True
    assert adj["binselect"] == "espr"  # spacings -> polynomial-regression rule
    off = core.rdplot_numbers(y, x, binselect="es", masspoints="off")
    assert off["mass_points_adjusted"] is False
    assert off["binselect"] == "es"


def test_xx_inv_reports_a_singular_gram_matrix():
    assert core._xx_inv(np.ones((5, 2))) is None
    X = np.column_stack([np.ones(5), np.arange(5.0)])
    np.testing.assert_allclose(core._xx_inv(X), np.linalg.inv(X.T @ X), rtol=1e-12)


# ------------------------------------------------------------ lpdensity numbers


@pytest.mark.parametrize("kernel", ["triangular", "uniform", "epanechnikov"])
def test_lpdensity_is_the_kernel_weighted_slope_of_the_empirical_cdf(kernel):
    rng = np.random.default_rng(7)
    data = np.sort(rng.normal(size=400))
    grid = np.array([-0.5, 0.0, 0.8])
    bw, p = 0.9, 2
    out = core.lpdensity_numbers(data, grid, bw, p=p, kernel=kernel, mass_points=False)
    Fn = np.arange(1, 401) / 400
    want = []
    for g in grid:
        u = (data - g) / bw
        ins = np.abs(u) <= 1
        k = {
            "triangular": 1 - np.abs(u),
            "uniform": np.full_like(u, 0.5),
            "epanechnikov": 0.75 * (1 - u**2),
        }[kernel]
        Xd = np.vander(u[ins], p + 1, increasing=True)
        sw = np.sqrt(k[ins])
        beta = np.linalg.lstsq(Xd * sw[:, None], Fn[ins] * sw, rcond=None)[0]
        want.append(beta[1] / bw)  # dF/dx = (dF/du) / h
    # normal equations vs. least squares on a well-conditioned 3-column design
    np.testing.assert_allclose(out["f_p"], want, rtol=1e-9)
    # N(0, 1) density, n = 400, h = 0.9: smoothing bias and noise both a few
    # hundredths; 0.08 is about three reported standard errors.
    np.testing.assert_allclose(out["f_p"], stats.norm.pdf(grid), atol=0.08)
    assert np.all(out["se_p"] > 0) and np.all(out["se_p"] < 0.05)

    # Without ties the mass-point adjustment has nothing to adjust.
    tied = core.lpdensity_numbers(data, grid, bw, p=p, kernel=kernel)
    np.testing.assert_allclose(tied["f_p"], out["f_p"], rtol=1e-10)
    np.testing.assert_allclose(tied["se_p"], out["se_p"], rtol=1e-10)
    np.testing.assert_array_equal(
        out["nh"], [(np.abs(data - g) <= bw).sum() for g in grid]
    )


def test_lpdensity_returns_nan_where_the_window_is_empty():
    data = np.linspace(0, 1, 50)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = core.lpdensity_numbers(
            data, np.array([0.5, 9.0]), 0.3, p=2, mass_points=False
        )
        tied = core.lpdensity_numbers(data, np.array([0.5, 9.0]), 0.3, p=2)
    for res in (out, tied):
        assert res["nh"][1] == 0
        assert np.isfinite(res["f_p"][0]) and np.isnan(res["f_p"][1])
        # uniform grid on [0, 1]: density 1 in the interior, exactly, because
        # the ECDF of an equally spaced sample is linear
        assert res["f_p"][0] == pytest.approx(1.0, rel=0.03)
