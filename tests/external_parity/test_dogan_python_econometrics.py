"""Dogan, *Introduction to Econometrics with Python*
(https://osmdogan.github.io/Python_Book), a Python companion to Stock &
Watson whose examples are written for statsmodels, linearmodels,
scikit-learn and ``arch``.

Each test reruns one of the book's computations with StatsPAI on the book's
own data and compares it with the library the book used, run here, or with
Stata 18 output recorded in the comment next to the number.

The data are not redistributed. Set ``STATSPAI_DOGAN_DIR`` to the folder
holding the book's ``datasets`` directory to run this; skipped otherwise.
"""

import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import statspai as sp

ROOT = os.environ.get("STATSPAI_DOGAN_DIR")
pytestmark = pytest.mark.skipif(
    not ROOT or not (Path(ROOT) / "datasets").is_dir(),
    reason="set STATSPAI_DOGAN_DIR to the folder holding the book's datasets/",
)


def _path(name):
    return Path(ROOT) / "datasets" / name


@pytest.fixture(scope="module")
def growth():
    g = pd.read_excel(_path("GrowthRate.xlsx"))
    g.index = pd.date_range("1960", periods=len(g), freq="QS")
    t = pd.read_csv(_path("TermSpread.csv"))
    t.index = pd.date_range("1960-01-01", periods=len(t), freq="QS")
    return pd.merge(g, t, left_index=True, right_index=True)


@pytest.fixture(scope="module")
def caschool():
    return pd.read_excel(_path("caschool.xlsx"), sheet_name="caschool")


# --- chapters 4-8: regression, robust errors, F tests ------------------


def test_ch8_polynomial_and_log_regressions_match_statsmodels(caschool):
    import statsmodels.formula.api as smf

    for formula in (
        "testscr ~ avginc + I(avginc**2) + I(avginc**3)",
        "np.log(testscr) ~ np.log(avginc)",
        "testscr ~ str + el_pct + str*el_pct",
    ):
        ref = smf.ols(formula, caschool).fit(cov_type="HC1")
        got = sp.regress(formula, caschool, robust="hc1")
        assert list(got.params.index) == list(ref.params.index)
        # the same least-squares problem and the same HC1 sandwich
        np.testing.assert_allclose(got.params, ref.params, rtol=1e-8)
        np.testing.assert_allclose(got.std_errors, ref.bse, rtol=1e-7)
    cubic = sp.regress(
        "testscr ~ avginc + I(avginc**2) + I(avginc**3)", caschool, robust="hc1"
    )
    # the book: model3.f_test("I(avginc ** 2)=0, I(avginc ** 3)=0"), F = 37.69
    out = sp.test(cubic, "I(avginc**2) = 0, I(avginc**3) = 0")
    assert out["statistic"] == pytest.approx(37.6908, abs=1e-3)
    assert out["df"] == (2, 416)


# --- chapter 12: instrumental variables --------------------------------


def test_ch12_cigarette_demand_in_the_books_own_formula():
    from linearmodels.iv import IV2SLS

    m = pd.read_excel(_path("cig_ch12.xlsx"))
    m["rprice"] = m.avgprs / m.cpi
    m["salestax"] = (m.taxs - m.tax) / m.cpi
    m["rincome"] = m.income / m["pop"] / m.cpi
    m95 = m[m.year == 1995]
    formula = "np.log(packpc) ~ 1 + np.log(rincome) + [np.log(rprice) ~ salestax]"
    ref = IV2SLS.from_formula(formula, m95).fit(cov_type="robust")
    got = sp.ivreg(formula, m95, robust="hc0")
    # linearmodels' "robust" applies no small-sample factor: HC0
    np.testing.assert_allclose(
        np.sort(got.params.to_numpy()), np.sort(ref.params.to_numpy()), rtol=1e-9
    )
    np.testing.assert_allclose(
        np.sort(got.std_errors.to_numpy()),
        np.sort(ref.std_errors.to_numpy()),
        rtol=1e-8,
    )


# --- chapter 14: prediction with many regressors ------------------------

_MAIN = (
    "str_s med_income_z te_avgyr_s exp_1000_1999_d frpm_frac_s ell_frac_s "
    "freem_frac_s enrollment_s fep_frac_s edi_s re_aian_frac_s re_asian_frac_s "
    "re_baa_frac_s re_fil_frac_s re_hl_frac_s re_hpi_frac_s re_tom_frac_s "
    "re_nr_frac_s te_fte_s te_1yr_frac_s te_2yr_frac_s te_tot_fte_rat_s "
    "exp_2000_2999_d exp_3000_3999_d exp_4000_4999_d exp_5000_5999_d "
    "exp_6000_6999_d exp_7000_7999_d exp_8000_8999_d expoc_1000_1999_d "
    "expoc_2000_2999_d expoc_3000_3999_d expoc_4000_4999_d expoc_5000_5999_d "
    "revoc_8010_8099_d revoc_8100_8299_d revoc_8300_8599_d revoc_8600_8799_d"
).split()


@pytest.fixture(scope="module")
def schools():
    ins = pd.read_stata(_path("ca_school_testscore_insample.dta"))
    oos = pd.read_stata(_path("ca_school_testscore_outofsample.dta"))
    both = pd.concat([ins, oos], ignore_index=True)
    cols = {}
    for i, a in enumerate(_MAIN):
        cols[f"x_{i + 1}"] = both[a]
        for j in range(i, len(_MAIN)):
            cols[f"xx_{i + 1}_{j + 1}"] = both[a] * both[_MAIN[j]]
        cols[f"xxx_{i + 1}"] = both[a] ** 3
    big = pd.DataFrame(cols).astype(float).drop(columns="xx_1_19")
    names = list(big.columns)
    big["testscore"] = both["testscore"].to_numpy(dtype=float)
    return big.iloc[: len(ins)], big.iloc[len(ins) :], names


def test_ch14_ridge_and_principal_components_with_816_predictors(schools):
    from sklearn.decomposition import PCA
    from sklearn.linear_model import Ridge

    train, hold, names = schools
    assert len(names) == 816 and len(train) == 1966
    Z = ((train[names] - train[names].mean()) / train[names].std(ddof=1)).to_numpy()
    yc = (train.testscore - train.testscore.mean()).to_numpy()

    # the book's cross-validated ridge penalty, as a given value
    ridge = sp.shrinkage(train, "testscore", names, penalty=1413.0, n_folds=0)
    ref = Ridge(alpha=1413.0, fit_intercept=False, solver="svd").fit(Z, yc)
    np.testing.assert_allclose(ridge.params.to_numpy(), ref.coef_, rtol=1e-8)

    pcr = sp.shrinkage(
        train, "testscore", names, method="pcr", n_components=51, n_folds=0
    )
    pca = PCA(n_components=51, svd_solver="full").fit(Z)
    gamma = np.linalg.lstsq(pca.transform(Z), yc, rcond=None)[0]
    np.testing.assert_allclose(
        pcr.params.to_numpy(), pca.components_.T @ gamma, rtol=1e-6, atol=1e-9
    )

    # The chapter's finding: with 816 predictors least squares predicts
    # badly out of sample and shrinkage does not (its Tables 24.4 to 24.7
    # report about 39.5 for ridge against 64 for OLS).
    cv_ridge = sp.shrinkage(train, "testscore", names, method="ridge")
    ols = sp.shrinkage(train, "testscore", names, method="ols", n_folds=0)
    assert 38.0 < cv_ridge.cv_rmspe < 41.0
    assert 38.0 < cv_ridge.rmspe(hold) < 41.0
    assert ols.rmspe(hold) > 55.0 > cv_ridge.rmspe(hold)


# --- chapter 15: autoregressions ---------------------------------------


def test_ch15_arima_is_the_exact_mle_stata_reports(growth):
    y = growth.loc["1962-01-01":"2017-07-01", "YGROWTH"].dropna()
    assert len(y) == 223
    # Stata 18: `arima y, ar(1)` and `arima y, ar(1/2)` on these 223 rows.
    stata = {
        (1, 0, 0): ([2.9798, 0.3365603], 3.047469, -564.997267),
        (2, 0, 0): ([2.98797, 0.2772972, 0.1757696], 3.000033, -561.492361),
        (0, 0, 2): ([2.979422, 0.2764447, 0.1918223], 3.028517, -563.587700),
    }
    for order, (coefs, sigma, ll) in stata.items():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = sp.arima(y, order=order)
        got = np.asarray(res.params, dtype=float)
        # Stata prints seven significant digits; its optimiser stops at
        # its own tolerance, so the fifth digit is the comparison
        np.testing.assert_allclose(got[:-1], coefs, rtol=2e-4)
        # the likelihood is flat in sigma near its maximum: Stata stops
        # with sigma 1e-4 away while the two log-likelihoods agree to 1e-5
        assert np.sqrt(got[-1]) == pytest.approx(sigma, rel=5e-4)
        assert res.log_likelihood == pytest.approx(ll, abs=1e-4)


def test_ch15_lags_written_as_shift_in_the_formula(growth):
    import statsmodels.formula.api as smf

    data = growth.loc["1962-01-01":"2017-07-01", ["YGROWTH", "RSPREAD"]].dropna()
    formula = (
        "YGROWTH ~ YGROWTH.shift(1) + YGROWTH.shift(2) "
        "+ RSPREAD.shift(1) + RSPREAD.shift(2)"
    )
    ref = smf.ols(formula, data).fit(cov_type="HC1")
    got = sp.regress(formula, data, robust="hc1")
    np.testing.assert_allclose(got.params.to_numpy(), ref.params, rtol=1e-9)
    np.testing.assert_allclose(got.std_errors.to_numpy(), ref.bse, rtol=1e-9)
    # the book's Granger F statistic, 3.98 on (2, 216)
    out = sp.test(got, "RSPREAD.shift(1) = 0, RSPREAD.shift(2) = 0")
    assert out["statistic"] == pytest.approx(3.9806, abs=1e-3)
    adl = sp.ardl(data.reset_index(), y="YGROWTH", x="RSPREAD", lags=2, x_lags=2)
    np.testing.assert_allclose(adl.params.to_numpy(), ref.params, rtol=1e-8)


# --- chapter 16: distributed lags with HAC errors ----------------------


def test_ch16_orange_juice_distributed_lag_with_newey_west_errors():
    import statsmodels.formula.api as smf

    df = pd.read_csv(_path("FrozenJuice.csv"))
    df.index = pd.date_range("1950-01-01", periods=len(df), freq="ME")
    df["rprice"] = df.price / df.ppi
    df["dprice"] = 100 * (np.log(df.rprice) - np.log(df.rprice.shift(1)))
    df["dFDD"] = df.fdd.diff()
    cumulative = (
        "dprice ~ "
        + " + ".join(f"dFDD.shift({i})" for i in range(18))
        + " + fdd.shift(18)"
    )
    for lags in (7, 14):
        ref = smf.ols(cumulative, df).fit(cov_type="HAC", cov_kwds={"maxlags": lags})
        got = sp.regress(cumulative, df, robust="hac", hac_lags=lags)
        assert got.nobs == 594
        np.testing.assert_allclose(got.params.to_numpy(), ref.params, rtol=1e-9)
        np.testing.assert_allclose(got.std_errors.to_numpy(), ref.bse, rtol=1e-9)


# --- chapter 17: vector error correction --------------------------------


def test_ch17_vector_error_correction_model_matches_statsmodels():
    from statsmodels.tsa.vector_ar.vecm import VECM

    rates = pd.read_stata(_path("FRED-QD.dta"))[["tb3ms", "gs10"]]
    rates = rates.dropna().astype(float).reset_index(drop=True)
    ref = VECM(rates, k_ar_diff=3, coint_rank=1, deterministic="ci").fit()
    got = sp.vec(rates, lags=3, rank=1, trend="rc")
    # the same reduced-rank regression; the book prints
    # alpha = (-0.0945, 0.0688) and beta = (1, -1.0072)
    np.testing.assert_allclose(
        np.asarray(got.alpha, dtype=float).ravel(), ref.alpha.ravel(), rtol=1e-8
    )
    beta = np.asarray(got.beta, dtype=float).ravel()
    np.testing.assert_allclose(beta[:2], ref.beta.ravel(), rtol=1e-8)
    assert beta[2] == pytest.approx(float(ref.det_coef_coint.ravel()[0]), rel=1e-8)
    assert got.log_likelihood == pytest.approx(ref.llf, abs=1e-6)
