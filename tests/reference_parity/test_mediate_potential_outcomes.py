"""``sp.mediate(inference='robust')`` against Stata 18 ``mediate``.

Data: ``sp.datasets.nsw_dw()`` with ``lre78 = log(re78 + 1)`` as outcome,
``emp75 = 1[re75 > 0]`` as mediator and ``age education black married`` as
covariates -- chosen for the numbers, not for the causal story. Stata read
the same values from a .dta written by pandas::

    mediate (lre78 age education black married) ///
            (emp75 age education black married [, logit | probit]) (treat), all
    mediate (...) (...) (treat), nointeraction
    estat proportion

Evidence tier. Point estimates are the same closed-form function of the same
least-squares / maximum-likelihood fits and agree to 1e-9. The covariance of
the model coefficients agrees to 1e-9 as well. The standard errors of the
*effects* agree only to about 2e-4, and that gap is Stata's: its variance
uses numerical derivatives of the potential-outcome means, and the gradient
it implies has non-zero entries where the analytic gradient is exactly zero
(on the textbook data that prompted this work, 0.0065 for a covariate the
natural indirect effect does not depend on). Stata's own standard errors
move by 1e-4 when a covariate is centred, an equivalent model; the analytic
sandwich here does not move at all, which
``test_standard_errors_do_not_depend_on_how_covariates_are_scaled`` pins.
The one effect whose standard error involves no such derivative -- the
direct effect without interaction, a regression coefficient -- agrees to
3e-11.
"""

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import MethodIncompatibility

X = ["age", "education", "black", "married"]

# Point estimates: closed form on both sides; the logit / probit rows carry
# Stata's optimiser tolerance (2e-9 observed).
RTOL_EST = 1e-8
# Standard errors: bounded by the numerical-derivative noise in Stata's
# variance, 1.9e-4 at worst on these rows (see the module docstring).
RTOL_SE = 5e-4

# label: (kwargs, {effect: (estimate, se)})
STATA = {
    "linear_interaction": (
        {"interaction": True},
        {
            "NIE": (-0.358549695957, 0.384541640787),
            "NDE": (-1.884561700961, 0.474647480896),
            "PNIE": (-0.535672299602, 0.132053565122),
            "TNDE": (-1.707439097316, 0.313032464952),
            "TE": (-2.243111396918, 0.285273684047),
            "Prop. Mediated": (0.159844801489, 0.172268310234),
        },
    ),
    "logit_interaction": (
        {"interaction": True, "mediator_model": "logit"},
        {
            "NIE": (-0.339427581252, 0.364813257099),
            "NDE": (-1.884562559189, 0.474648411072),
            "PNIE": (-0.507103910693, 0.129252667358),
            "TNDE": (-1.716886229748, 0.310853505597),
            "TE": (-2.223990140441, 0.286433060414),
            "Prop. Mediated": (0.152620991919, 0.166154946158),
        },
    ),
    "probit_interaction": (
        {"interaction": True, "mediator_model": "probit"},
        {
            "NIE": (-0.344349921946, 0.369803482758),
            "NDE": (-1.884553152540, 0.474635619871),
            "PNIE": (-0.514457874700, 0.129680601982),
            "TNDE": (-1.714445199785, 0.311277877045),
            "TE": (-2.228903074486, 0.285909955032),
            "Prop. Mediated": (0.154492999668, 0.167721834891),
        },
    ),
    "linear_no_interaction": (
        {},
        {
            "NIE": (-0.506045337897, 0.128484751304),
            "NDE": (-1.731981147855, 0.309934700031),
            "TE": (-2.238026485752, 0.285784858786),
            "Prop. Mediated": (0.226112309715, 0.063512041129),
        },
    ),
}


@pytest.fixture(scope="module")
def df() -> pd.DataFrame:
    d = sp.datasets.nsw_dw().copy()
    d["emp75"] = (d["re75"] > 0).astype(int)
    d["lre78"] = np.log(d["re78"] + 1)
    return d


def _fit(df, covariates=None, **kwargs):
    return sp.mediate(
        df,
        "lre78",
        "treat",
        "emp75",
        X if covariates is None else covariates,
        inference="robust",
        **kwargs,
    )


@pytest.mark.parametrize("label", sorted(STATA))
def test_effects_match_stata(df, label):
    kwargs, reference = STATA[label]
    detail = _fit(df, **kwargs).detail.set_index("effect")
    assert list(detail.index) == list(reference)
    for effect, (est, se) in reference.items():
        assert detail.loc[effect, "estimate"] == pytest.approx(est, rel=RTOL_EST)
        assert detail.loc[effect, "se"] == pytest.approx(se, rel=RTOL_SE)


def test_direct_effect_without_interaction_is_the_regression_coefficient(df):
    res = _fit(df)
    nde = res.detail.set_index("effect").loc["NDE"]
    # No numerical derivative on Stata's side for this one, so it is exact.
    assert nde["se"] == pytest.approx(0.309934700031, rel=1e-9)
    ols = sp.regress("lre78 ~ treat + emp75 + " + " + ".join(X), df, vce="hc0")
    assert nde["estimate"] == pytest.approx(ols.params["treat"], rel=1e-12)
    assert nde["se"] == pytest.approx(ols.std_errors["treat"], rel=1e-9)


def test_potential_outcome_means_match_stata(df):
    po = _fit(df, interaction=True).model_info["po_means"].set_index("effect")
    # e(b) of the `all` fit, %14.10f.
    for name, value in {
        "Y0M0": 9.1181033191,
        "Y1M0": 7.2335416181,
        "Y0M1": 8.5824310195,
        "Y1M1": 6.8749919222,
    }.items():
        assert po.loc[name, "estimate"] == pytest.approx(value, abs=5e-10)


def test_decomposition_identities(df):
    res = _fit(df, interaction=True, mediator_model="logit")
    e = res.detail.set_index("effect")["estimate"]
    assert e["NIE"] + e["NDE"] == pytest.approx(e["TE"], rel=1e-12)
    assert e["PNIE"] + e["TNDE"] == pytest.approx(e["TE"], rel=1e-12)
    assert e["Prop. Mediated"] == pytest.approx(e["NIE"] / e["TE"], rel=1e-12)
    assert res.estimate == e["NIE"] and res.estimand == "NIE"

    plain = _fit(df)
    info = plain.model_info
    product = info["outcome_coef"]["emp75"] * info["mediator_coef"]["treat"]
    assert plain.estimate == pytest.approx(product, rel=1e-12)


def test_standard_errors_do_not_depend_on_how_covariates_are_scaled(df):
    # The same model in a different basis: Stata's standard errors change at
    # 1e-4 under this, the sandwich of the estimating equations must not.
    shifted = df.assign(
        age=(df["age"] - 30.0) / 7.0, education=df["education"] * 1000.0
    )
    for kwargs in (
        {"interaction": True},
        {"interaction": True, "mediator_model": "logit"},
    ):
        a = _fit(df, **kwargs).detail.set_index("effect")
        b = _fit(shifted, **kwargs).detail.set_index("effect")
        # rtol: only floating-point conditioning separates the two fits.
        np.testing.assert_allclose(a["estimate"], b["estimate"], rtol=1e-8)
        np.testing.assert_allclose(a["se"], b["se"], rtol=1e-7)


def test_recovers_known_effects_with_interaction():
    # Linear model with a treatment-mediator interaction: the effects are
    # NDE = bd + bdm * E[M(0)], NIE = (bm + bdm) * ad, PNIE = bm * ad.
    rng = np.random.default_rng(20261002)
    n = 200_000
    x = rng.normal(size=n)
    d = rng.binomial(1, 0.5, n)
    m = 0.3 + 0.8 * d + 0.5 * x + rng.normal(size=n)
    y = 1.0 + 0.6 * d + 0.7 * m + 0.4 * d * m + 0.2 * x + rng.normal(size=n)
    data = pd.DataFrame({"y": y, "d": d, "m": m, "x": x})
    res = sp.mediate(data, "y", "d", "m", ["x"], inference="robust", interaction=True)
    e = res.detail.set_index("effect")
    truth = {
        "NIE": (0.7 + 0.4) * 0.8,
        "PNIE": 0.7 * 0.8,
        "NDE": 0.6 + 0.4 * 0.3,
        "TNDE": 0.6 + 0.4 * (0.3 + 0.8),
    }
    for name, value in truth.items():
        # Four standard errors: a miss would be a 1-in-16,000 event per row.
        assert abs(e.loc[name, "estimate"] - value) < 4 * e.loc[name, "se"]


def test_robust_coverage_under_heteroskedasticity():
    # Error variance grows with the covariate; the delta method assumes it
    # constant, the stacked-equation sandwich does not.
    rng = np.random.default_rng(7)
    n, reps, truth = 600, 300, 0.7 * 0.8
    hits = 0
    for _ in range(reps):
        x = rng.normal(size=n)
        d = rng.binomial(1, 0.5, n)
        m = 0.8 * d + 0.5 * x + rng.normal(size=n) * (1 + np.abs(x))
        y = 0.6 * d + 0.7 * m + 0.2 * x + rng.normal(size=n) * (1 + np.abs(m))
        data = pd.DataFrame({"y": y, "d": d, "m": m, "x": x})
        res = sp.mediate(data, "y", "d", "m", ["x"], inference="robust")
        hits += res.ci[0] <= truth <= res.ci[1]
    # 300 replications: a 95% interval covers 0.95 +/- 0.025 (two binomial
    # standard errors), so anything below 0.90 is undercoverage.
    assert hits / reps > 0.90


@pytest.mark.parametrize(
    "kwargs",
    [
        {"inference": "delta", "interaction": True},
        {"inference": "bootstrap", "mediator_model": "logit"},
    ],
)
def test_interaction_and_binary_mediator_need_robust(df, kwargs):
    with pytest.raises(MethodIncompatibility, match="inference='robust'"):
        sp.mediate(df, "lre78", "treat", "emp75", X, **kwargs)


def test_continuous_treatment_is_refused(df):
    with pytest.raises(MethodIncompatibility, match="0/1"):
        sp.mediate(df, "lre78", "age", "emp75", ["education"], inference="robust")


def test_logit_needs_a_binary_mediator(df):
    with pytest.raises(MethodIncompatibility, match="0/1 mediator"):
        sp.mediate(
            df,
            "lre78",
            "treat",
            "re75",
            X,
            inference="robust",
            mediator_model="logit",
        )


def test_collinear_covariates_are_refused(df):
    dup = df.assign(age_copy=df["age"])
    with pytest.raises(MethodIncompatibility, match="rank"):
        _fit(dup, covariates=X + ["age_copy"])


def test_unknown_mediator_model_is_refused(df):
    with pytest.raises(MethodIncompatibility, match="mediator_model"):
        _fit(df, mediator_model="poisson")


def test_categorical_covariate_equals_hand_made_dummies(df):
    # `i.g` in Stata, `C(g)` here: one indicator per level but the lowest.
    banded = df.assign(
        band=np.select([df.age < 22, df.age < 28, df.age < 35], [1, 2, 3], 4)
    )
    dummies = pd.get_dummies(banded["band"], prefix="b", drop_first=True).astype(float)
    wide = pd.concat([banded, dummies], axis=1)
    by_term = sp.mediate(
        banded,
        "lre78",
        "treat",
        "emp75",
        ["education", "C(band)"],
        inference="robust",
        interaction=True,
    )
    by_hand = sp.mediate(
        wide,
        "lre78",
        "treat",
        "emp75",
        ["education", "b_2", "b_3", "b_4"],
        inference="robust",
        interaction=True,
    )
    np.testing.assert_allclose(
        by_term.detail["estimate"], by_hand.detail["estimate"], rtol=1e-12
    )
    np.testing.assert_allclose(by_term.detail["se"], by_hand.detail["se"], rtol=1e-12)
    assert "band[2]" in by_term.model_info["outcome_coef"].index


def test_stata_line_runs_through_sp_stata(df):
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sp.stata(
            "mediate(lre78 age education black married)"
            "(emp75 age education black married, logit)(treat), all",
            df,
        )
    detail = res.detail.set_index("effect")
    reference = STATA["logit_interaction"][1]
    for effect, (est, se) in reference.items():
        assert detail.loc[effect, "estimate"] == pytest.approx(est, rel=RTOL_EST)
        assert detail.loc[effect, "se"] == pytest.approx(se, rel=RTOL_SE)
