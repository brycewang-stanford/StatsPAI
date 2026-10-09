"""Branch tests for ``statspai.rd.rdrobust`` not reached elsewhere.

Weights validation and the weights + donut combination of ``sp.rdrobust``,
the legacy rule-of-thumb bandwidth selector that backs the fallback path
(all eight ``bwselect`` routes), the polynomial band helper of
``sp.rdplot``, and the plotting options that the main suite leaves out.
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import (
    ConvergenceFailure,
    DataInsufficient,
    MethodIncompatibility,
)
from statspai.rd import rdrobust as rr

matplotlib.use("Agg")


def _sharp(n=800, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, n)
    y = 1.0 * (x >= 0) + 0.5 * x + 0.8 * x**2 + rng.normal(0, 0.3, n)
    w = rng.uniform(0.5, 2.0, n)
    return pd.DataFrame({"y": y, "x": x, "w": w})


# ------------------------------------------------------- validation


def test_empty_column_list_is_refused():
    with pytest.raises(MethodIncompatibility, match="at least one column"):
        rr._coerce_column_list([], "covs")
    assert rr._coerce_column_list([], "covs", allow_empty=True) == []


def test_weights_column_must_exist_and_be_finite():
    df = _sharp(300)
    with pytest.raises(MethodIncompatibility, match="'nope' not found"):
        sp.rdrobust(df, y="y", x="x", weights="nope")
    bad = df.copy()
    bad.loc[3, "w"] = np.inf
    with pytest.raises(MethodIncompatibility, match="must be finite"):
        sp.rdrobust(bad, y="y", x="x", weights="w")


def test_weights_with_donut_equals_dropping_the_donut_rows():
    df = _sharp()
    kw = dict(y="y", x="x", weights="w", h=0.4, manipulation_test=False)
    a = sp.rdrobust(df, donut=0.05, **kw)
    b = sp.rdrobust(df[df["x"].abs() > 0.05], **kw)
    assert a.estimate == pytest.approx(b.estimate, rel=1e-12)
    assert a.se == pytest.approx(b.se, rel=1e-12)
    # and the weights matter: the unweighted donut fit is a different number
    c = sp.rdrobust(df, y="y", x="x", h=0.4, donut=0.05, manipulation_test=False)
    assert abs(a.estimate - c.estimate) > 1e-6


def test_parse_data_without_mask_drops_incomplete_rows():
    df = _sharp(50)
    df.loc[0, "y"] = np.nan
    out = rr._parse_data(df, "y", "x", 0.25, None, None)
    assert len(out) == 4
    Y, Xc, D, Z = out
    assert D is None and Z is None
    np.testing.assert_allclose(Y, df["y"].to_numpy()[1:])
    np.testing.assert_allclose(Xc, df["x"].to_numpy()[1:] - 0.25)


def test_failed_density_test_is_recorded_not_raised():
    # 16 observations: enough for a fixed-bandwidth fit, too few for the
    # density test. The estimate is returned and the failure is on record.
    rng = np.random.default_rng(21)
    x = np.linspace(-0.95, 0.95, 16)
    df = pd.DataFrame({"x": x, "y": (x >= 0) + rng.normal(0, 0.1, 16)})
    res = sp.rdrobust(df, y="y", x="x", h=0.9)
    mc = res.model_info["mccrary"]
    assert mc["pvalue"] is None and mc["test"] == "rddensity"
    assert mc["error"] == "DataInsufficient"
    assert abs(res.estimate - 1.0) < 0.5


# ------------------------------------------------- legacy fallback path


def _fuzzy(n=1500, seed=3):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, n)
    d = (rng.uniform(size=n) < 0.1 + 0.8 * (x >= 0)).astype(float)
    y = 2.0 * d + 0.5 * x + rng.normal(0, 0.3, n)
    return pd.DataFrame({"y": y, "x": x, "d": d, "w": rng.uniform(0.5, 2, n)})


def _fail_bias_correction(monkeypatch):
    """Fault injection: make the CCT bias-corrected fit report a singular design.

    Only the failure is injected; every number asserted below is produced by
    the real legacy estimator that the fallback switches to.
    """
    from statspai.rd import _cct_bandwidth

    def boom(*args, **kwargs):
        raise np.linalg.LinAlgError("singular (injected)")

    monkeypatch.setattr(_cct_bandwidth, "cct_bias_corrected", boom)


def test_bias_corrected_fallback_warns_and_is_recorded(monkeypatch):
    df = _fuzzy()
    kw = dict(y="y", x="x", h=0.4, b=0.6, manipulation_test=False)
    ref = sp.rdrobust(df, **kw)
    assert ref.model_info["legacy_fallbacks"] == []
    _fail_bias_correction(monkeypatch)
    with pytest.warns(Warning, match="rdrobust bias-corrected fit"):
        res = sp.rdrobust(df, **kw)
    assert res.model_info["legacy_fallbacks"] == ["bias_corrected_fit"]
    # the legacy conventional fit is the same local-linear regression
    conv_ref = ref.model_info["conventional"]["estimate"]
    assert res.model_info["conventional"]["estimate"] == pytest.approx(
        conv_ref, rel=1e-8
    )


def test_fuzzy_fallback_forms_the_wald_ratio_once(monkeypatch):
    df = _fuzzy()
    kw = dict(y="y", x="x", fuzzy="d", h=0.4, b=0.6, manipulation_test=False)
    ref = sp.rdrobust(df, **kw)
    itt = sp.rdrobust(df, y="y", x="x", h=0.4, b=0.6, manipulation_test=False)
    fs = sp.rdrobust(df, y="d", x="x", h=0.4, b=0.6, manipulation_test=False)
    _fail_bias_correction(monkeypatch)
    with pytest.warns(Warning, match="legacy local-polynomial estimate"):
        res = sp.rdrobust(df, **kw)
    assert "bias_corrected_fit" in res.model_info["legacy_fallbacks"]
    wald = (
        itt.model_info["conventional"]["estimate"]
        / fs.model_info["conventional"]["estimate"]
    )
    assert res.model_info["conventional"]["estimate"] == pytest.approx(wald, rel=1e-8)
    assert res.model_info["conventional"]["estimate"] == pytest.approx(
        ref.model_info["conventional"]["estimate"], rel=1e-8
    )
    assert abs(res.estimate - 2.0) < 0.3


def test_weighted_fit_never_falls_back_to_the_unweighted_legacy_path(monkeypatch):
    df = _fuzzy()
    _fail_bias_correction(monkeypatch)
    with pytest.raises(np.linalg.LinAlgError, match="injected"):
        sp.rdrobust(
            df, y="y", x="x", weights="w", h=0.4, b=0.6, manipulation_test=False
        )


def test_weighted_bandwidth_selection_failure_is_not_swallowed(monkeypatch):
    from statspai.rd import _cct_bandwidth

    def boom(*args, **kwargs):
        raise ZeroDivisionError("empty side (injected)")

    monkeypatch.setattr(_cct_bandwidth, "cct_bandwidth", boom)
    with pytest.raises(ZeroDivisionError, match="injected"):
        sp.rdrobust(_fuzzy(), y="y", x="x", weights="w", manipulation_test=False)


# ------------------------------------------ legacy bandwidth selector


def test_cer_factor_is_one_for_a_single_observation():
    assert rr._cer_factor(1) == 1.0
    assert rr._cer_factor(1000, p=1) == pytest.approx(1000 ** (-1 / 35))


def test_legacy_selector_routes_are_consistent_with_each_other():
    df = _sharp(1500, seed=4)
    Y, Xc = df["y"].to_numpy(), df["x"].to_numpy()
    left, right = Xc < 0, Xc >= 0

    def sel(bw):
        return rr._select_bandwidth(Y, Xc, left, right, 1, "triangular", bwselect=bw)

    h = sel("mserd")
    hl, hr = sel("msetwo")
    cer = rr._cer_factor(len(Y), 1)
    assert 0 < h < 2 and 0 < hl < 2 and 0 < hr < 2
    assert sel("msecomb1") == pytest.approx(min(h, hl, hr))
    assert sel("msecomb2") == pytest.approx(np.median([h, hl, hr]))
    assert sel("cerrd") == pytest.approx(h * cer)
    assert sel("certwo") == pytest.approx((hl * cer, hr * cer))
    assert sel("cercomb1") == pytest.approx(min(h, hl, hr) * cer)
    assert sel("cercomb2") == pytest.approx(np.median([h, hl, hr]) * cer)
    # an unrecognised name falls through to the common MSE bandwidth
    assert sel("something-else") == h
    # the common bandwidth is the closed form when curvature differs by side
    assert h != pytest.approx(1.06 * Xc.std() * len(Y) ** (-0.2))


def test_side_optimal_bandwidth_closed_form_and_clipping():
    h = rr._side_optimal_bw(
        sigma2=0.25, m2=2.0, f_c=0.5, n_side=400, C_K=3.0, h_pilot=0.3, x_range=2.0
    )
    assert h == pytest.approx((3.0 * 0.25 / (0.5 * 4.0 * 400)) ** 0.2)
    # no curvature, or too few observations: the pilot bandwidth
    assert rr._side_optimal_bw(0.25, 0.0, 0.5, 400, 3.0, 0.3, 2.0) == 0.3
    assert rr._side_optimal_bw(0.25, 2.0, 0.5, 4, 3.0, 0.3, 2.0) == 0.3
    # clipped to 98% of the range
    assert rr._side_optimal_bw(0.25, 0.0, 0.5, 400, 3.0, 5.0, 2.0) == 1.96


def test_pilot_helpers_with_too_few_points_in_the_window():
    y = np.array([1.0, 2.0, 4.0])
    x = np.array([-0.1, -0.2, -0.3])
    assert rr._local_residual_var(y, x, 1.0, "triangular") == pytest.approx(np.var(y))
    assert rr._local_residual_var(y[:0], x[:0], 1.0, "triangular") == 1.0
    assert rr._estimate_second_deriv(y, x, 1.0, "triangular") == 0.0
    # an exact quadratic has second derivative 2 * 3 = 6
    xg = np.linspace(-1, 0, 40)
    assert rr._estimate_second_deriv(
        1 + xg + 3 * xg**2, xg, 2.0, "uniform"
    ) == pytest.approx(6.0, abs=1e-8)


# ----------------------------------------------------------- rbc bootstrap


def _boot_args(Y, Xc, h=0.5, D=None, n_boot=99):
    return dict(
        Y=Y,
        X_c=Xc,
        D=D,
        Z=None,
        left=Xc < 0,
        right=Xc >= 0,
        h=h,
        b=h,
        p=1,
        q=2,
        kernel="triangular",
        deriv=0,
        cluster_vals=None,
        alpha=0.05,
        n_boot=n_boot,
        random_state=0,
        tau_bc=1.0,
        se_robust=0.1,
    )


def test_rbc_bootstrap_fails_loudly_when_no_replicate_is_usable():
    df = _sharp(200)
    Xc = df["x"].to_numpy()
    # a constant outcome has a zero standard error in every replicate
    with pytest.raises(ConvergenceFailure):
        rr._rbc_bootstrap(**_boot_args(np.ones(200), Xc))
    # a first stage that is identically zero leaves no Wald ratio to form
    with pytest.raises(ConvergenceFailure):
        rr._rbc_bootstrap(**_boot_args(df["y"].to_numpy(), Xc, D=np.zeros(200)))


def test_rbc_bootstrap_refuses_an_empty_side():
    df = _sharp(200)
    Y, Xc = df["y"].to_numpy(), df["x"].to_numpy()
    with pytest.raises(DataInsufficient, match="too few observations") as err:
        rr._rbc_bootstrap(**_boot_args(Y, Xc, h=1e-6))
    assert err.value.diagnostics["minimum_per_side"] == 3


# ------------------------------------------------------------- plots


def test_weighted_poly_fit_ci_unweighted_matches_ols_band():
    rng = np.random.default_rng(2)
    x = rng.uniform(0, 1, 60)
    y = 1 + 2 * x + rng.normal(0, 0.2, 60)
    grid = np.array([0.0, 0.5, 1.0])
    fit, lo, hi = rr._weighted_poly_fit_ci(x, y, 1, grid, 0.95)
    X = np.column_stack([np.ones(60), x])
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    np.testing.assert_allclose(fit, beta[0] + beta[1] * grid, atol=1e-10)
    s2 = np.sum((y - X @ beta) ** 2) / 58
    G = np.column_stack([np.ones(3), grid])
    se = np.sqrt(np.einsum("ij,jk,ik->i", G, s2 * np.linalg.inv(X.T @ X), G))
    np.testing.assert_allclose((hi - lo) / 2, 1.959963984540054 * se, rtol=1e-6)
    np.testing.assert_allclose((hi + lo) / 2, fit, atol=1e-10)


def test_weighted_poly_fit_ci_needs_three_points():
    grid = np.linspace(0, 1, 4)
    out = rr._weighted_poly_fit_ci(
        np.array([0.1, 0.9]), np.array([1.0, 2.0]), 2, grid, 0.95
    )
    for arr in out:
        assert arr.shape == (4,) and np.isnan(arr).all()


def test_rdplot_rejects_unknown_kernel():
    with pytest.raises(MethodIncompatibility, match="kernel"):
        sp.rdplot(_sharp(200), y="y", x="x", kernel="gaussian")


def test_rdplot_show_bw_warns_when_no_bandwidth_can_be_computed():
    # 20 observations: enough to draw the plot, too few for the selected
    # bandwidth to hold a local-linear fit on both sides.
    rng = np.random.default_rng(6)
    x = rng.uniform(-1, 1, 20)
    df = pd.DataFrame({"x": x, "y": (x >= 0) + rng.normal(0, 0.1, 20)})
    with pytest.warns(RuntimeWarning, match="no window is shaded"):
        fig, ax = sp.rdplot(df, y="y", x="x", show_bw=True)
    try:
        labels = [t.get_text() for t in ax.get_legend().get_texts()]
        assert not any("Bandwidth" in lab for lab in labels)
    finally:
        plt.close(fig)


def test_rdplot_weighted_band_differs_from_unweighted():
    df = _sharp(600, seed=7)
    # weights that favour one end of each side change the polynomial band
    df["w"] = np.where(df["x"].abs() > 0.5, 5.0, 0.2)
    fig_w, ax_w = sp.rdplot(df, y="y", x="x", weights="w", shade_ci=True)
    fig_u, ax_u = sp.rdplot(df, y="y", x="x", shade_ci=True)
    try:
        assert len(ax_w.collections) >= 2  # the two shaded bands

        def band(ax):
            return ax.collections[0].get_paths()[0].vertices[:, 1]

        bw_, bu_ = band(ax_w), band(ax_u)
        assert np.isfinite(bw_).all()
        assert not np.allclose(bw_, bu_)
        assert not np.allclose(
            fig_w.rdplot_data["vars_poly"]["rdplot_y"],
            fig_u.rdplot_data["vars_poly"]["rdplot_y"],
        )
    finally:
        plt.close(fig_w)
        plt.close(fig_u)


def test_rdplotdensity_takes_one_bandwidth_per_side():
    df = _sharp(600)
    fig, ax = sp.rdplotdensity(df, x="x", h=(0.2, 0.4))
    try:
        d = fig.rdplotdensity_data
        np.testing.assert_allclose(d["Estl"]["bw"], 0.2)
        np.testing.assert_allclose(d["Estr"]["bw"], 0.4)
        # the grid spans 3h on each side, clipped to the data
        assert d["Estl"]["grid"].min() == pytest.approx(-0.6)
        assert d["Estr"]["grid"].max() == pytest.approx(min(1.2, df["x"].max()))
    finally:
        plt.close(fig)
