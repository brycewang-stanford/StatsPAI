"""``sp.esize``, ``sp.power_ttest`` and ``sp.cor_test`` against Stata 18 and R.

The data are the car-sales experiment of Das, *Causal Inference in R*
(Packt), ch. 8: 200 salespeople per strategy, cars sold a Poisson count. Only
the two columns the tests need are kept, as the frequency table of the count
by strategy, so the sample is reproduced exactly.

Stata 18 (on the same 400 rows)::

    esize twosample cars_sold, by(strategy) all
    esize twosample cars_sold, by(strategy) all level(90) unequal
    power twomeans 0 0.5, n(128)            // and the other calls below

R 4.5.2: ``effsize::cohen.d`` 0.8.1, ``pwr::pwr.t.test`` 1.3-0.

All three are closed forms or one-dimensional root-finding on the noncentral
t distribution, so the tolerances are those of the references' own solvers.
"""

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from scipy import stats

import statspai as sp
from statspai.exceptions import DataInsufficient, MethodIncompatibility

COUNTS = {
    "A": {0: 14, 1: 27, 2: 37, 3: 51, 4: 37, 5: 16, 6: 11, 7: 2, 8: 3, 9: 1, 10: 1},
    "B": {0: 2, 1: 20, 2: 30, 3: 37, 4: 25, 5: 45, 6: 20, 7: 13, 8: 5, 9: 3},
}


@pytest.fixture(scope="module")
def sales():
    rows = [
        (g, float(v))
        for g, tab in COUNTS.items()
        for v, k in tab.items()
        for _ in range(k)
    ]
    df = pd.DataFrame(rows, columns=["strategy", "cars_sold"])
    assert len(df) == 400
    return df


# r() scalars of `esize twosample cars_sold, by(strategy) all`
STATA_ESIZE = {
    "cohens_d": (-0.4987737422331154, -0.6975032320971195, -0.2994358475911826),
    "hedges_g": (-0.4978331526214401, -0.6961878775812961, -0.2988711702159333),
    "glass_delta1": (-0.5177767279035572, -0.7196475297296105, -0.3146836709359342),
    "glass_delta2": (-0.4817203515944913, -0.6827694335655836, -0.2795249255047998),
    "point_biserial_r": (
        -0.2425472083313851,
        -0.3300365450981612,
        -0.1484310057992675,
    ),
}
# ... , level(90) unequal
STATA_ESIZE_UNEQUAL_90 = {
    "cohens_d": (-0.4987737422331154, -0.6655118369142139, -0.3314169074775062),
    "hedges_g": (-0.4978331526214401, -0.6642568119053875, -0.3307919200856054),
    "glass_delta1": (-0.5177767279035572, -0.6870782345766231, -0.3472218385366549),
    "glass_delta2": (-0.4817203515944913, -0.6503412724702481, -0.3119278053421928),
    "point_biserial_r": (
        -0.2431389118786071,
        -0.317185673649869,
        -0.1642917967738206,
    ),
}


@pytest.mark.parametrize(
    "kwargs, reference",
    [({}, STATA_ESIZE), ({"unequal": True, "alpha": 0.10}, STATA_ESIZE_UNEQUAL_90)],
)
def test_esize_equals_stata(sales, kwargs, reference):
    res = sp.esize(sales, "cars_sold", by="strategy", **kwargs)
    for name, row in reference.items():
        # Stata solves for the noncentrality to about 1e-10
        np.testing.assert_allclose(
            res.table.loc[name].to_numpy(), row, rtol=1e-8, err_msg=name
        )
    assert res.statistic == pytest.approx(reference["cohens_d"][0], rel=1e-12)
    assert res.estimates["n1"] == 200 and res.estimates["n2"] == 200


def test_esize_point_estimates_equal_effsize(sales):
    # effsize::cohen.d(cars_sold ~ strategy, pooled = TRUE): -0.498773742233115;
    # with hedges.correction = TRUE: -0.497833251204392, which uses the
    # approximation 1 - 3 / (4 m - 1) to the exact factor.
    res = sp.esize(sales, "cars_sold", by="strategy")
    assert res.table.loc["cohens_d", "estimate"] == pytest.approx(
        -0.498773742233115, rel=1e-13
    )
    g = res.table.loc["hedges_g", "estimate"]
    assert g == pytest.approx(-0.497833251204392, rel=1e-6)
    assert g != pytest.approx(-0.497833251204392, rel=1e-9)


def test_esize_interval_covers_the_true_effect():
    rng = np.random.default_rng(20261006)
    reps, covered = 400, 0
    for _ in range(reps):
        df = pd.DataFrame({"g": np.repeat([0, 1], [12, 18])})
        df["y"] = rng.standard_normal(30) - 0.8 * df["g"]  # d = +0.8 (0 minus 1)
        t = sp.esize(df, "y", by="g").table.loc["cohens_d"]
        covered += t["ci_lower"] <= 0.8 <= t["ci_upper"]
    # 400 draws of a 95% interval: above 92% with probability > 0.99
    assert covered / reps >= 0.92


def test_esize_refuses_what_it_cannot_standardise(sales):
    three = sales.assign(strategy=np.resize(["A", "B", "C"], len(sales)))
    with pytest.raises(MethodIncompatibility, match="exactly 2"):
        sp.esize(three, "cars_sold", by="strategy")
    flat = pd.DataFrame({"g": [0, 0, 0, 1, 1, 1], "y": [1.0, 1, 1, 2, 3, 4]})
    with pytest.raises(DataInsufficient, match="does not vary"):
        sp.esize(flat, "y", by="g")


# (kwargs, quantity, Stata value) -- `power twomeans / onemean / pairedmeans`
STATA_POWER = [
    (dict(n=128, delta=0.5), "power", 0.801459557922254),
    (dict(n=120, delta=0.5, ratio=2), "power", 0.726069919977493),
    (
        dict(n=60, delta=2, sd=4, alpha=0.01, alternative="greater"),
        "power",
        0.331813777097685,
    ),
    (dict(n=30, delta=0.4, type="one-sample"), "power", 0.562813607141057),
    (dict(n=128, power=0.8), "delta", 0.499069177965821),
]


@pytest.mark.parametrize("kwargs, quantity, expected", STATA_POWER)
def test_power_ttest_equals_stata(kwargs, quantity, expected):
    res = sp.power_ttest(**kwargs)
    got = res.power if quantity == "power" else res.params["delta"]
    assert got == pytest.approx(expected, rel=1e-8)


def test_power_ttest_sample_sizes_equal_stata_and_pwr():
    # Stata rounds each group up; pwr.t.test reports the fractional solution
    # for one group (uniroot, about 1e-7).
    res = sp.power_ttest(delta=0.5, power=0.8)
    assert (res.n, res.params["n1"], res.params["n2"]) == (128, 64, 64)
    assert res.power >= 0.8

    res = sp.power_ttest(delta=0.5, power=0.9, ratio=2)
    assert (res.n, res.params["n1"], res.params["n2"]) == (192, 64, 128)

    res = sp.power_ttest(delta=0.498773742233115, power=0.8)
    assert res.params["n_exact"] / 2 == pytest.approx(64.0746683564785, rel=1e-6)
    assert res.n == 130

    res = sp.power_ttest(delta=0.4, power=0.8, type="one-sample", alternative="greater")
    assert res.n == 41  # Stata
    assert res.params["n_exact"] == pytest.approx(40.0290847589618, rel=1e-6)

    res = sp.power_ttest(delta=0.3, power=0.9, alpha=0.01, type="paired")
    assert res.n == 169  # Stata
    assert res.params["n_exact"] == pytest.approx(168.657454308265, rel=1e-6)


def test_power_ttest_is_the_rejection_rate_of_the_test():
    rng = np.random.default_rng(7)
    n1, n2, delta, reps = 12, 20, 0.9, 4000
    a = rng.standard_normal((reps, n1))
    b = rng.standard_normal((reps, n2)) + delta
    p = stats.ttest_ind(a, b, axis=1).pvalue
    predicted = sp.power_ttest(n=n1 + n2, delta=delta, ratio=n2 / n1).power
    # binomial standard error at 4000 draws is under 0.008
    assert np.mean(p < 0.05) == pytest.approx(predicted, abs=0.025)
    # the normal approximation is too optimistic here
    assert sp.power_rct(n=n1 + n2, effect_size=delta, ratio=n2 / n1).power > predicted


def test_power_ttest_round_trip_and_sides():
    res = sp.power_ttest(n=90, delta=0.45, alternative="greater")
    back = sp.power_ttest(n=90, power=res.power, alternative="greater")
    assert back.params["delta"] == pytest.approx(0.45, rel=1e-8)
    less = sp.power_ttest(n=90, delta=-0.45, alternative="less")
    assert less.power == pytest.approx(res.power, rel=1e-12)
    assert sp.power_ttest(n=90, delta=0.45, alternative="less").power < 0.001


def test_power_ttest_arguments_are_validated():
    with pytest.raises(MethodIncompatibility, match="exactly two"):
        sp.power_ttest(delta=0.5)
    with pytest.raises(MethodIncompatibility, match="exactly two"):
        sp.power_ttest(n=100, delta=0.5, power=0.8)
    with pytest.raises(MethodIncompatibility, match="between alpha"):
        sp.power_ttest(delta=0.5, power=0.03)
    with pytest.raises(MethodIncompatibility, match="wrong side"):
        sp.power_ttest(delta=-0.5, power=0.8, alternative="greater")
    with pytest.raises(MethodIncompatibility, match="unknown type"):
        sp.power_ttest(n=50, delta=0.5, type="welch")


def _corr_data(seed=3, n=250):
    rng = np.random.default_rng(seed)
    z1, z2 = rng.standard_normal((2, n))
    x = 0.6 * z1 - 0.4 * z2 + rng.standard_normal(n)
    y = 0.3 * x + 0.5 * z1 + rng.standard_normal(n)
    return pd.DataFrame({"x": x, "y": y, "z1": z1, "z2": z2})


def test_cor_test_equals_scipy_pearsonr():
    df = _corr_data()
    res = sp.cor_test(df, "x", "y")
    ref = stats.pearsonr(df["x"], df["y"])
    ci = ref.confidence_interval(0.95)
    assert res.statistic == pytest.approx(ref.statistic, rel=1e-12)
    assert res.pvalue == pytest.approx(ref.pvalue, rel=1e-9)
    assert res.estimates["ci_lower"] == pytest.approx(ci.low, rel=1e-10)
    assert res.estimates["ci_upper"] == pytest.approx(ci.high, rel=1e-10)
    assert res.df == len(df) - 2


def test_cor_test_t_statistic_in_a_small_sample():
    df = pd.DataFrame(
        {
            "x": [1.2, 2.4, 3.1, 4.8, 5.0, 6.7, 7.1, 8.9],
            "y": [2.1, 1.9, 3.8, 3.2, 5.9, 5.1, 7.7, 7.0],
        }
    )
    res = sp.cor_test(df, "x", "y")
    ref = stats.pearsonr(df["x"], df["y"])
    t = ref.statistic * np.sqrt(6 / (1 - ref.statistic**2))
    assert res.estimates["t"] == pytest.approx(t, rel=1e-12)
    assert res.pvalue == pytest.approx(2 * stats.t.sf(abs(t), 6), rel=1e-12)


def test_partial_correlation_t_is_the_regression_t():
    """The t statistic of a partial correlation equals the t statistic of x
    in the regression of y on x and the covariates."""
    df = _corr_data()
    res = sp.cor_test(df, "x", "y", covariates=["z1", "z2"])
    ols = sm.OLS(df["y"], sm.add_constant(df[["x", "z1", "z2"]])).fit()
    assert res.estimates["t"] == pytest.approx(ols.tvalues["x"], rel=1e-10)
    assert res.pvalue == pytest.approx(ols.pvalues["x"], rel=1e-9)
    assert res.df == len(df) - 4
    assert res.estimates["ci_lower"] < res.statistic < res.estimates["ci_upper"]


def test_conditioning_on_a_collider_creates_a_correlation():
    rng = np.random.default_rng(100)
    n = 3000
    income, frugality = rng.standard_normal((2, n))
    spent = 0.5 * income - 0.3 * frugality + rng.standard_normal(n)
    df = pd.DataFrame({"income": income, "frugality": frugality, "spent": spent})
    assert sp.cor_test(df, "income", "frugality").pvalue > 0.01
    partial = sp.cor_test(df, "income", "frugality", covariates="spent")
    assert partial.pvalue < 1e-6 and partial.statistic > 0


def test_cor_test_handles_missing_rows_and_bad_input():
    df = _corr_data()
    df.loc[:9, "z1"] = np.nan
    assert sp.cor_test(df, "x", "y").n_obs == len(df)
    assert sp.cor_test(df, "x", "y", covariates=["z1"]).n_obs == len(df) - 10
    with pytest.raises(MethodIncompatibility, match="different columns"):
        sp.cor_test(df, "x", "x")
    with pytest.raises(DataInsufficient, match="constant"):
        sp.cor_test(df.assign(c=1.0), "x", "c")
    with pytest.raises(MethodIncompatibility, match="not numeric"):
        sp.cor_test(df.assign(s="a"), "x", "s")
