"""Option scan, round 6: ``sp.estat`` and the logit diagnostics after a
weighted or non-linear fit.

Reference numbers are Stata 18 (``regress y x1 x2 [aw=w]`` followed by the
``estat`` command named in each test) on the data built by ``_data()``,
written with ``DataFrame.to_csv`` (shortest round-trip repr, so Stata reads
the same doubles) and read back with ``set type double``.
"""

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import MethodIncompatibility


def _data():
    rng = np.random.default_rng(1)
    n = 200
    df = pd.DataFrame(
        {
            "x1": rng.normal(size=n),
            "x2": rng.uniform(size=n),
            "w": rng.uniform(0.2, 3, size=n),
        }
    )
    df["y"] = 1 + df.x1 + 0.5 * df.x2 + rng.normal(size=n) * (1 + df.x2)
    return df


@pytest.fixture(scope="module")
def fits():
    df = _data()
    return {
        "aw": sp.regress("y ~ x1 + x2", data=df, weights="w"),
        "pw": sp.regress("y ~ x1 + x2", data=df, weights="w", weights_as="sampling"),
        "ols": sp.regress("y ~ x1 + x2", data=df),
    }


def _estat(res, test, **kw):
    return sp.estat(res, test, print_results=False, **kw)


# ---------------------------------------------------------------- hettest
@pytest.mark.parametrize(
    "kw, stata",
    [
        # estat hettest, rhs iid
        ({"variables": "rhs", "version": "iid"}, 16.8051384368403),
        # estat hettest, rhs
        ({"variables": "rhs", "version": "normal"}, 18.4690131260306),
        # estat hettest (fitted values, normal)
        ({"variables": "fitted", "version": "normal"}, 9.41446122607371),
        # estat hettest, iid
        ({"variables": "fitted", "version": "iid"}, 8.5663117532494),
        # estat hettest x1
        ({"variables": ["x1"], "version": "normal"}, 5.40453767909347),
        # estat hettest x1, iid
        ({"variables": ["x1"], "version": "iid"}, 4.91764250014397),
    ],
)
def test_hettest_after_analytic_weights_matches_stata(fits, kw, stata):
    out = _estat(fits["aw"], "hettest", **kw)
    assert out["statistic"] == pytest.approx(stata, rel=1e-10)


def test_hettest_fstat_after_analytic_weights_matches_stata(fits):
    # estat hettest, rhs fstat -> F(2, 197) = 9.03576728028514
    out = _estat(fits["aw"], "hettest", variables="rhs", version="fstat")
    assert (out["df1"], out["df2"]) == (2, 197)
    assert out["statistic"] == pytest.approx(9.03576728028514, rel=1e-10)


def test_hettest_after_sampling_weights_is_refused(fits):
    # Stata: "hettest not appropriate with pweighted data"
    with pytest.raises(MethodIncompatibility):
        _estat(fits["pw"], "hettest")


# ------------------------------------------------------------ white / imtest
@pytest.mark.parametrize("test", ["white", "imtest"])
def test_white_and_imtest_after_weights_are_refused(fits, test):
    # Stata: "imtest does not support weights"
    with pytest.raises(MethodIncompatibility):
        _estat(fits["aw"], test)


# ------------------------------------------------------------------ reset
def test_reset_after_analytic_weights_matches_stata(fits):
    # regress y x1 x2 z2 z3 z4 [aw=w]; test z2 z3 z4, with z the fitted
    # values on unit scale: F = 2.39223212144352. `estat ovtest` prints
    # 2.39223346553226; it rescales the fitted values in a way that moves
    # the sixth digit, and the explicit regression is the exact statistic.
    out = _estat(fits["aw"], "reset", powers=4)
    assert (out["df1"], out["df2"]) == (3, 194)
    assert out["statistic"] == pytest.approx(2.39223212144352, rel=1e-9)
    assert out["statistic"] == pytest.approx(2.39223346553226, rel=1e-6)


def test_reset_rhs_after_analytic_weights_matches_stata(fits):
    # estat ovtest, rhs -> F(6, 191) = 1.44053122013518
    out = _estat(fits["aw"], "reset", powers=4, rhs=True)
    assert (out["df1"], out["df2"]) == (6, 191)
    assert out["statistic"] == pytest.approx(1.44053122013518, rel=1e-6)


def test_reset_after_sampling_weights_is_the_robust_wald(fits):
    # regress ... z2 z3 z4 [pw=w]; test z2 z3 z4 -> 3.31056215678665
    # (estat ovtest after [pw=w] prints 3.31055779188871)
    out = _estat(fits["pw"], "reset", powers=4)
    assert out["statistic"] == pytest.approx(3.31056215678665, rel=1e-8)


# -------------------------------------------------------------------- vif
def test_vif_after_analytic_weights_matches_stata(fits):
    # estat vif after [aw=w] and after [pw=w]: 1.03125287945777 for both
    for key in ("aw", "pw"):
        table = _estat(fits[key], "vif")["vif_table"]
        assert table["VIF"].to_numpy() == pytest.approx(
            [1.03125287945777] * 2, rel=1e-12
        )


# --------------------------------------------------------------- linktest
def test_linktest_after_analytic_weights_matches_stata(fits):
    # predict yh; regress y yh yh^2 [aw=w]
    out = _estat(fits["aw"], "linktest")
    assert out["coef_hatsq"] == pytest.approx(-0.343688744668963, rel=1e-10)
    assert out["se_hatsq"] == pytest.approx(0.14158166431005, rel=1e-10)
    assert out["statistic"] == pytest.approx(-2.42749473488544, rel=1e-10)


def _logit_data():
    rng = np.random.default_rng(2)
    n = 300
    df = pd.DataFrame(
        {
            "x1": rng.normal(size=n),
            "c": rng.normal(size=n),
            "w": rng.uniform(0.3, 3, n),
            "g": rng.integers(0, 30, n),
        }
    )
    df["yb"] = (0.3 + df.x1 + rng.logistic(size=n) > 0).astype(int)
    return df


def test_linktest_after_sampling_weights_is_robust(fits):
    # regress y yh yh^2 [pw=w]: the same coefficient, robust SE
    out = _estat(fits["pw"], "linktest")
    assert out["coef_hatsq"] == pytest.approx(-0.343688744668963, rel=1e-10)
    assert out["se_hatsq"] == pytest.approx(0.170796692556939, rel=1e-9)


@pytest.mark.parametrize(
    "fit, coef, se",
    [
        # logit yb x1 c; linktest (it printed .0384553 (.1369572), z = 0.28
        # where sp.estat ran OLS of y on the fitted probabilities)
        (
            lambda d: sp.logit("yb ~ x1 + c", data=d),
            0.038455254371249,
            0.136957196676904,
        ),
        # probit yb x1 c; predict xb; probit yb xb xb^2
        (
            lambda d: sp.probit("yb ~ x1 + c", data=d),
            0.0436618984971089,
            0.209891313824275,
        ),
        # gen cnt = mod(g, 5); poisson cnt x1 c; predict xb; poisson cnt xb
        # xb^2, tolerance(1e-14) ltolerance(1e-14) nrtolerance(1e-14): the
        # square is nearly collinear with the index and Stata's default
        # stopping rule leaves it at -1.24746688565294 (1.44707644535628)
        (
            lambda d: sp.poisson("cnt ~ x1 + c", data=d.assign(cnt=d.g % 5)),
            -1.24747016352303,
            1.44707664797374,
        ),
    ],
)
def test_linktest_after_likelihood_model_refits_that_model(fit, coef, se):
    res = fit(_logit_data())
    out = _estat(res, "linktest")
    assert out["statistic_label"] == "z"
    assert out["coef_hatsq"] == pytest.approx(coef, rel=1e-6)
    assert out["se_hatsq"] == pytest.approx(se, rel=1e-6)


# ----------------------------------------- unweighted fits stay unchanged
def test_unweighted_estat_values_unchanged(fits):
    # Stata after the unweighted regress: estat ovtest -> .715828848385817
    # (the same sixth-digit rescaling as in the weighted case above)
    out = _estat(fits["ols"], "reset", powers=4)
    assert out["statistic"] == pytest.approx(0.715828848385817, rel=1e-6)
    # estat hettest, rhs iid on the unweighted fit equals N R^2 of the
    # auxiliary OLS regression
    df = _data()
    X = np.column_stack([np.ones(len(df)), df.x1, df.x2])
    e = df.y.to_numpy() - X @ np.linalg.lstsq(X, df.y.to_numpy(), rcond=None)[0]
    e2 = e**2
    r = e2 - X @ np.linalg.lstsq(X, e2, rcond=None)[0]
    r2 = 1 - (r @ r) / np.sum((e2 - e2.mean()) ** 2)
    out = _estat(fits["ols"], "hettest")
    assert out["statistic"] == pytest.approx(len(df) * r2, rel=1e-12)


# ------------------------------------------- tests with no weighted form
@pytest.mark.parametrize("test", ["leverage", "normality"])
def test_residual_diagnostics_after_weights_are_refused(fits, test):
    with pytest.raises(MethodIncompatibility):
        _estat(fits["aw"], test)


def test_estat_all_after_weights_runs_the_weighted_tests(fits):
    outs = _estat(fits["aw"], "all")
    het = next(o for o in outs if o["test"].startswith("Breusch-Pagan"))
    assert het["statistic"] == pytest.approx(16.8051384368403, rel=1e-10)
    assert not any(o["test"].startswith("White") for o in outs)


@pytest.mark.parametrize("test", ["hettest", "white", "reset", "bgodfrey"])
def test_linear_specification_tests_refuse_a_logit(test):
    res = sp.logit("yb ~ x1 + c", data=_logit_data())
    with pytest.raises(MethodIncompatibility):
        _estat(res, test)


# ------------------------------------------------ logit diagnostics + weights
@pytest.mark.parametrize(
    "fn",
    [
        lambda r: sp.logit_gof(r),
        lambda r: sp.logit_influence(r),
        lambda r: sp.estat(r, "classification", print_results=False),
    ],
)
def test_logit_diagnostics_after_weights_are_refused(fn):
    # Stata refuses estat gof / estat classification after [pw=]; and
    # sp.logit_gof used to return the unweighted Pearson statistic (295.72)
    # where Stata's [iw=] statistic is 491.83.
    res = sp.logit("yb ~ x1 + c", data=_logit_data(), weights="w")
    with pytest.raises(MethodIncompatibility):
        fn(res)
