"""Correctness of ``sp.rd_honest`` (``statspai/rd/honest_ci.py``).

Every number the result reports is recomputed here from first principles:

* the point estimate, from local-linear weights built in the test;
* the worst-case bias of *those* weights under a second-derivative bound
  ``M`` -- ``M/2 * sum |w_i| x_i^2`` for the Taylor class, and
  ``-M/2 * sum w_i x_i^2`` for the Holder class (Armstrong & Kolesar's
  closed form for a boundary local-linear estimator);
* the critical value ``cv_{1-alpha}(b)``, the ``1 - alpha`` quantile of
  ``|N(b, 1)|``, through the noncentral chi-square: ``|N(b,1)|^2`` is
  ``chi2(1, b^2)``;
* the p-value, the ``alpha`` at which the honest interval's edge sits on
  zero: ``P(|N(b, 1)| > |tau| / se)``.
"""

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import statspai as sp
from statspai.rd.honest_ci import _honest_pvalue

M, H = 2.0, 0.4
KERNELS = {
    "triangular": lambda u: 1 - np.abs(u),
    "uniform": lambda u: np.ones_like(u),
    "epanechnikov": lambda u: 1 - u**2,
}


@pytest.fixture(scope="module")
def df():
    rng = np.random.default_rng(42)
    n = 400
    x = rng.uniform(-1, 1, n)
    y = 0.5 * x + 2.0 * (x >= 0) + 0.8 * x**2 + rng.normal(0, 0.4, n)
    return pd.DataFrame({"y": y, "x": x, "g": rng.integers(0, 40, n)})


def _intercept_weights(xs, k):
    """Weights w with w @ y = intercept of the k-weighted linear fit."""
    X = np.column_stack([np.ones(len(xs)), xs])
    return np.linalg.solve((X * k[:, None]).T @ X, (X * k[:, None]).T)[0]


def _sides(df, kernel):
    x, y = df["x"].to_numpy(), df["y"].to_numpy()
    mr, ml = (x >= 0) & (x <= H), (x < 0) & (x >= -H)
    wr = _intercept_weights(x[mr], KERNELS[kernel](x[mr] / H))
    wl = _intercept_weights(x[ml], KERNELS[kernel](x[ml] / H))
    return (x[ml], y[ml], wl), (x[mr], y[mr], wr)


def _folded_normal_cv(b, alpha):
    return float(np.sqrt(stats.ncx2.ppf(1 - alpha, 1, b**2)))


@pytest.mark.parametrize("kernel", sorted(KERNELS))
@pytest.mark.parametrize("sclass", ["H", "T"])
def test_estimate_bias_and_interval_match_first_principles(df, kernel, sclass):
    alpha = 0.1
    res = sp.rd_honest(
        df, y="y", x="x", M=M, h=H, kernel=kernel, sclass=sclass, alpha=alpha
    )
    mi = res.model_info
    (xl, yl, wl), (xr, yr, wr) = _sides(df, kernel)

    # two linear solves of the same 2x2 system: rounding only
    assert res.estimate == pytest.approx(wr @ yr - wl @ yl, abs=1e-12)

    if sclass == "T":
        bias = M / 2 * (np.abs(wr) @ xr**2 + np.abs(wl) @ xl**2)
    else:
        bias = -M / 2 * (wr @ xr**2 + wl @ xl**2)
    # a sum of ~160 products against the same sum in another order
    assert mi["bias_bound"] == pytest.approx(bias, rel=1e-10)

    b = bias / res.se
    assert mi["bias_noise_ratio"] == pytest.approx(b, rel=1e-10)
    cv = _folded_normal_cv(b, alpha)
    # root-finder in the package vs. scipy's quantile function
    assert mi["ak_critical_value"] == pytest.approx(cv, rel=1e-8)
    # The interval is tau +/- cv * se. The bias is inside cv; adding it again
    # (tau +/- (cv * se + bias)) was the 1.7x-too-wide interval fixed earlier.
    lo, hi = res.ci
    assert hi == pytest.approx(res.estimate + cv * res.se, rel=1e-9)
    assert lo == pytest.approx(res.estimate - cv * res.se, rel=1e-9)
    assert mi["honest_ci"] == res.ci
    assert hi - lo < 2 * (cv * res.se + bias)

    z = stats.norm.ppf(1 - alpha / 2)
    assert mi["naive_ci"] == pytest.approx(
        (res.estimate - z * res.se, res.estimate + z * res.se), rel=1e-12
    )
    # cv(b) > z for b > 0, and below the "bias + z * se" shortcut
    assert z < mi["ak_critical_value"] < z + b
    assert mi["n_left"] == len(xl) and mi["n_right"] == len(xr)
    assert mi["sclass"] == sclass and mi["kernel"] == kernel
    assert mi["M"] == M and mi["M_estimated"] is False
    assert "(supplied)" in mi["summary_str"]
    assert f"Honest {int((1 - alpha) * 100)}% CI" in mi["summary_str"]


def test_taylor_class_is_never_narrower_than_holder(df):
    kw = dict(y="y", x="x", M=M, h=H)
    hold = sp.rd_honest(df, sclass="holder", **kw)
    tayl = sp.rd_honest(df, sclass="Taylor", **kw)
    # the long spellings are accepted and normalised
    assert hold.model_info["sclass"] == "H" and tayl.model_info["sclass"] == "T"
    assert hold.estimate == tayl.estimate and hold.se == tayl.se
    assert tayl.model_info["bias_bound"] > hold.model_info["bias_bound"]
    assert tayl.ci[1] - tayl.ci[0] > hold.ci[1] - hold.ci[0]


def test_bias_scales_linearly_in_M_and_vanishes_with_it(df):
    kw = dict(y="y", x="x", h=H)
    one = sp.rd_honest(df, M=1.0, **kw)
    four = sp.rd_honest(df, M=4.0, **kw)
    assert four.model_info["bias_bound"] == pytest.approx(
        4 * one.model_info["bias_bound"], rel=1e-12
    )
    assert four.estimate == one.estimate and four.se == one.se
    tiny = sp.rd_honest(df, M=1e-12, **kw)
    # no curvature allowed: the honest interval is the naive one
    assert tiny.ci == pytest.approx(tiny.model_info["naive_ci"], rel=1e-9)
    assert tiny.model_info["ak_critical_value"] == pytest.approx(
        stats.norm.ppf(0.975), rel=1e-9
    )


def test_honest_pvalue_inverts_the_honest_interval():
    # P(|N(b, 1)| > t), via the noncentral chi-square
    for tau, se, bias in [(0.30, 0.10, 0.05), (-0.25, 0.10, 0.12), (0.5, 0.4, 0.0)]:
        want = stats.ncx2.sf((tau / se) ** 2, 1, (bias / se) ** 2) if bias else None
        got = _honest_pvalue(tau, se, bias)
        if want is None:
            want = 2 * stats.norm.sf(abs(tau) / se)  # no bias: the z-test
        # 200 bisection steps on a monotone function; limited by the root
        # tolerance of the critical-value solver
        assert got == pytest.approx(want, rel=1e-6)
        # sign of the estimate is irrelevant
        assert _honest_pvalue(-tau, se, bias) == got
    # |tau| far inside the bias bound: zero is all but impossible to exclude
    assert _honest_pvalue(0.01, 0.10, 0.50) == pytest.approx(
        stats.ncx2.sf(0.1**2, 1, 5.0**2), rel=1e-9
    )
    assert _honest_pvalue(0.0, 0.10, 0.05) == 1.0
    # more allowed bias can only raise the p-value
    assert _honest_pvalue(0.3, 0.1, 0.10) > _honest_pvalue(0.3, 0.1, 0.02)


def test_reported_pvalue_agrees_with_the_interval_about_zero():
    rng = np.random.default_rng(3)
    n = 300
    x = rng.uniform(-1, 1, n)
    y = 0.15 * (x >= 0) + 0.3 * x + rng.normal(0, 0.5, n)
    frame = pd.DataFrame({"y": y, "x": x})
    res = sp.rd_honest(frame, y="y", x="x", M=1.0, h=0.5)
    b = res.model_info["bias_bound"] / res.se
    want = stats.ncx2.sf((res.estimate / res.se) ** 2, 1, b**2)
    assert 0.01 < want < 0.99  # an interior p-value, so the check has teeth
    assert res.pvalue == pytest.approx(want, rel=1e-6)
    # ... and strictly larger than the naive z-test's, which ignores the bias
    assert res.pvalue > 2 * stats.norm.sf(abs(res.estimate) / res.se)
    # duality: the level-p interval has an endpoint at zero
    at_p = sp.rd_honest(frame, y="y", x="x", M=1.0, h=0.5, alpha=res.pvalue)
    edge = min(abs(at_p.ci[0]), abs(at_p.ci[1]))
    # the interval moves by ~se per unit of cv; 1e-5 * se is the solver noise
    assert edge < 1e-5 * res.se
    for a in (res.pvalue * 0.5, min(res.pvalue * 1.5, 0.99)):
        r = sp.rd_honest(frame, y="y", x="x", M=1.0, h=0.5, alpha=a)
        excludes_zero = r.ci[0] > 0 or r.ci[1] < 0
        assert excludes_zero == (res.pvalue < a)


def test_bandwidth_criteria_and_the_rule_of_thumb_M(df):
    kw = dict(y="y", x="x", M=M)
    fits = {
        oc: sp.rd_honest(df, opt_criterion=oc.upper(), **kw)
        for oc in ("mse", "flci", "oci")
    }
    hs = {oc: f.model_info["bandwidth"] for oc, f in fits.items()}
    assert len({round(h, 8) for h in hs.values()}) == 3
    # the one-sided criterion trades less bias for more variance than MSE
    assert hs["oci"] < hs["mse"]
    for oc, f in fits.items():
        assert f.model_info["opt_criterion"] == oc  # lower-cased on the way in
        # true jump 2.0; each interval must cover it (M = 2 bounds the true
        # second derivative 1.6, so this is the guarantee, not luck)
        assert f.ci[0] < 2.0 < f.ci[1]
        # refitting at the selected bandwidth reproduces the fit
        again = sp.rd_honest(df, h=hs[oc], **kw)
        assert again.estimate == pytest.approx(f.estimate, rel=1e-10)
        assert again.se == pytest.approx(f.se, rel=1e-10)

    auto = sp.rd_honest(df, y="y", x="x")
    assert auto.model_info["M_estimated"] is True
    assert "(estimated)" in auto.model_info["summary_str"]
    # the rule of thumb is conservative here: above the true bound 1.6
    assert auto.model_info["M"] > 1.6
    assert auto.ci[0] < 2.0 < auto.ci[1]


def test_cluster_option_changes_only_the_variance(df):
    kw = dict(y="y", x="x", M=M, h=H)
    plain = sp.rd_honest(df, **kw)
    clus = sp.rd_honest(df, cluster="g", **kw)
    assert clus.estimate == plain.estimate
    assert clus.model_info["bias_bound"] == plain.model_info["bias_bound"]
    assert clus.se != plain.se
    assert clus.model_info["n_clusters"] == df["g"].nunique()
    assert plain.model_info["n_clusters"] is None
    # rows with a missing cluster id are dropped with the rest
    holed = df.copy()
    holed["g"] = holed["g"].astype(float)
    holed.loc[holed.index[:30], "g"] = np.nan
    a = sp.rd_honest(holed, cluster="g", **kw)
    b = sp.rd_honest(holed.iloc[30:], cluster="g", **kw)
    assert a.estimate == b.estimate and a.se == b.se and a.n_obs == len(df) - 30


@pytest.mark.parametrize(
    "kwargs, fragment",
    [
        (dict(opt_criterion="amse"), "opt_criterion must be 'mse', 'flci' or 'oci'"),
        (dict(sclass="G"), "sclass must be 'H'"),
    ],
)
def test_invalid_options_are_refused(df, kwargs, fragment):
    with pytest.raises(ValueError, match=fragment):
        sp.rd_honest(df, y="y", x="x", M=M, **kwargs)
