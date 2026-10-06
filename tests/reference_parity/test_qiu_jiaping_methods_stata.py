"""Stata 18 references for the methods of Qiu Jiaping's causal-inference
textbook: propensity score with blocks (``pscore``), nearest-neighbour
matching on a given score (``attnd``, ``psmatch2, pscore()``), the
intraclass correlation (``loneway``) and the selection equation of
``heckman, twostep``.

Every number below was printed by Stata 18 (MP) on
``sp.datasets.nsw_lalonde()`` written to a .dta file, or on the small
frames built here. The commands are quoted next to the numbers. ``pscore``
and ``attnd`` are Becker and Ichino's (st0026_2, Stata Journal 5-3);
``psmatch2`` is Leuven and Sianesi's.

The textbook's own data are not used: they cannot be redistributed. The
opt-in replay of its logs is ``tests/external_parity/test_qiu_jiaping_logs.py``.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp

COVARIATES = ["age", "educ", "black", "hispanic", "married", "nodegree", "re74", "re75"]
#: powers and an interaction of earnings: columns of order 1e9 next to dummies
POLYNOMIAL = [
    "age", "age2", "age3", "educ", "educ2", "black", "hispanic", "married",
    "nodegree", "re74", "re742", "re75", "re752", "educre74",
]  # fmt: skip


@pytest.fixture(scope="module")
def lalonde() -> pd.DataFrame:
    return sp.datasets.nsw_lalonde()


@pytest.fixture(scope="module")
def polynomial(lalonde) -> pd.DataFrame:
    d = lalonde
    return d.assign(
        age2=d.age**2.0, age3=d.age**3.0, educ2=d.educ**2.0, re742=d.re74**2,
        re752=d.re75**2, educre74=d.educ * d.re74,
    )  # fmt: skip


def _counts(result) -> list:
    return result.blocks[["n_control", "n_treated"]].to_numpy().tolist()


# ------------------------------------------------------------------ pscore
def test_pscore_logit_with_common_support(lalonde):
    """pscore treat age educ black hispanic married nodegree re74 re75,
    pscore(ps1) blockid(b1) logit comsup"""
    res = sp.pscore(lalonde, "treat", COVARIATES, common_support=True)
    assert res.loglik == pytest.approx(-243.92197, abs=5e-6)
    assert res.support_range == pytest.approx((0.02495178, 0.85315285), abs=5e-9)
    assert res.n_blocks == 7
    assert not res.balanced
    assert res.unbalanced[["variable", "block"]].to_numpy().tolist() == [["re75", 2]]
    assert _counts(res) == [[206, 10], [60, 14], [27, 11], [13, 7], [8, 24],
                            [58, 114], [0, 5]]  # fmt: skip
    assert res.blocks["lower"].to_numpy() == pytest.approx(
        [0, 0.1, 0.2, 0.4, 0.5, 0.6, 0.8], abs=1e-7
    )
    assert int(res.support.sum()) == 557
    # the block is missing outside the region of common support
    assert res.block.notna().sum() == 557
    coef = res.coefficients
    assert coef.loc["black", "coef"] == pytest.approx(3.065368, abs=5e-7)
    assert coef.loc["black", "se"] == pytest.approx(0.2865262, abs=5e-8)
    assert coef.loc["_cons", "coef"] == pytest.approx(-4.728649, abs=5e-7)


def test_pscore_probit(lalonde):
    """pscore treat age educ black hispanic married nodegree re74 re75,
    pscore(ps2) blockid(b2)"""
    res = sp.pscore(lalonde, "treat", COVARIATES, ps_model="probit")
    assert res.loglik == pytest.approx(-243.06433, abs=5e-6)
    assert res.n_blocks == 8
    assert res.unbalanced[["variable", "block"]].to_numpy().tolist() == [["re75", 3]]
    assert _counts(res) == [[165, 3], [93, 7], [62, 14], [27, 9], [14, 9],
                            [9, 24], [59, 115], [0, 4]]  # fmt: skip
    assert res.coefficients.loc["educ", "coef"] == pytest.approx(0.0921263, abs=5e-8)
    # Stata's probit stops when the scaled gradient is below 1e-5: its
    # standard errors agree with the converged ones to about six digits
    assert res.coefficients.loc["educ", "se"] == pytest.approx(0.0379201, rel=5e-6)


def test_pscore_other_start_and_level(lalonde):
    """pscore treat age educ married re74, pscore(ps3) blockid(b3) logit
    numblo(3) level(0.05)"""
    res = sp.pscore(
        lalonde, "treat", ["age", "educ", "married", "re74"], n_blocks=3, level=0.05
    )
    assert res.n_blocks == 8
    assert res.unbalanced[["variable", "block"]].to_numpy().tolist() == [
        ["married", 1], ["age", 2], ["age", 5], ["educ", 5], ["educ", 7], ["re74", 7],
    ]  # fmt: skip
    assert _counts(res) == [[150, 12], [102, 39], [9, 7], [38, 8], [66, 24],
                            [32, 28], [21, 56], [11, 11]]  # fmt: skip
    assert res.blocks["lower"].to_numpy() == pytest.approx(
        [0, 0.1666667, 0.3333333, 0.375, 0.4166667, 0.4375, 0.4583333, 0.5], abs=5e-8
    )


def test_pscore_assign_and_refusals(lalonde):
    res = sp.pscore(lalonde, "treat", COVARIATES, common_support=True)
    out = res.assign(lalonde, pscore="ps", block="b", support=None)
    assert list(out.columns.difference(lalonde.columns)) == ["b", "ps"]
    assert out["ps"].between(0, 1).all()
    with pytest.raises(sp.exceptions.MethodIncompatibility, match="ps_model"):
        sp.pscore(lalonde, "treat", COVARIATES, ps_model="cloglog")
    with pytest.raises(sp.exceptions.MethodIncompatibility, match="binary"):
        sp.pscore(lalonde, "age", ["educ"])
    with pytest.raises(sp.exceptions.MethodIncompatibility, match="not found"):
        sp.pscore(lalonde, "treat", ["nope"])


# ------------------------------------------------- matching on a given score
def test_match_on_a_given_score(lalonde):
    """psmatch2 treat, pscore(ps1) outcome(re78) neighbor(1) [ties]
    attnd re78 treat, pscore(ps1) [comsup]"""
    data = sp.pscore(lalonde, "treat", COVARIATES).assign(lalonde, pscore="ps1")
    ties = sp.psmatch2(data, treat="treat", pscore="ps1", outcome="re78", ties=True)
    # r(att), r(seatt) of psmatch2 and r(attnd), r(seattnd) of attnd
    assert ties.att == pytest.approx(1968.7997158559, rel=1e-11)
    assert ties.se == pytest.approx(1008.4082506356, rel=1e-9)
    first = sp.psmatch2(data, treat="treat", pscore="ps1", outcome="re78")
    assert first.att == pytest.approx(1967.9396843243, rel=1e-11)
    assert first.se == pytest.approx(1056.4867568065, rel=1e-9)
    # attnd, comsup: controls outside the range of the treated scores are set
    # aside (attnd holds its sums in single precision: nine digits)
    region = sp.psmatch2(data, treat="treat", pscore="ps1", outcome="re78",
                         ties=True, common_support="treated")  # fmt: skip
    assert region.att == pytest.approx(1981.076824530195, rel=5e-9)
    assert region.se == pytest.approx(1007.293531898701, rel=5e-9)
    used = region.matched_data
    controls = (
        (used["_treated"] == 0) & (used["_support"] == 1) & used["_weight"].notna()
    )
    assert int(controls.sum()) == 88  # r(ncnd)
    # the estimated score gives the same matches as the column holding it
    fitted = sp.psmatch2(lalonde, treat="treat", covariates=COVARIATES,
                         outcome="re78", ties=True)  # fmt: skip
    assert fitted.att == ties.att


def test_given_score_refusals(lalonde):
    data = sp.pscore(lalonde, "treat", COVARIATES).assign(lalonde, pscore="ps1")
    with pytest.raises(sp.exceptions.MethodIncompatibility, match="ties=True"):
        sp.psmatch2(data, treat="treat", pscore="ps1", outcome="re78",
                    common_support="treated")  # fmt: skip
    with pytest.raises(sp.exceptions.MethodIncompatibility, match="no score is"):
        sp.match(data, y="re78", treat="treat", covariates=COVARIATES,
                 pscore="ps1", se_method="abadie_imbens_2016")  # fmt: skip


def test_bootstrap_standard_error_with_ties(lalonde):
    data = sp.pscore(lalonde, "treat", COVARIATES).assign(lalonde, pscore="ps1")
    res = sp.psmatch2(data, treat="treat", pscore="ps1", outcome="re78", ties=True,
                      se="bootstrap", bootstrap_reps=60, bootstrap_seed=3)  # fmt: skip
    assert res.att == pytest.approx(1968.7997158559, rel=1e-11)
    info = res.result.model_info
    assert info["se_analytic"] == pytest.approx(1008.4082506356, rel=1e-9)
    # a bootstrap of 60 draws: the same order as the analytic one
    assert 0.6 * info["se_analytic"] < res.se < 1.6 * info["se_analytic"]


# ----------------------------------------------- the treatment model itself
def test_logit_with_columns_of_very_different_size(polynomial):
    """logit treat age age2 age3 educ educ2 black hispanic married nodegree
    re74 re742 re75 re752 educre74

    Columns of order 1e9 next to dummies: the coefficients range over
    nine orders of magnitude.
    """
    res = sp.pscore(polynomial, "treat", POLYNOMIAL)
    assert res.loglik == pytest.approx(-204.5182504419, abs=1e-9)
    stata = {
        "age": 2.060996305, "age2": -0.05399179273, "age3": 0.0004308909694,
        "educ": 1.068982643, "educ2": -0.05781829552, "black": 2.93338326,
        "hispanic": 0.6062857059, "married": -1.532266936,
        "nodegree": 0.3841697983, "re74": -0.0003206696223,
        "re742": 7.749457412e-09, "re75": -0.00002780274518,
        "re752": 5.312659192e-09, "educre74": 7.289714300e-06,
        "_cons": -29.77167836,
    }  # fmt: skip
    for name, value in stata.items():
        # Stata stops when the scaled gradient is below 1e-5
        assert res.coefficients.loc[name, "coef"] == pytest.approx(value, rel=5e-6)


def test_matching_on_that_score(polynomial):
    """teffects psmatch (re78) (treat age age2 ... educre74, logit), atet

    Two controls with the same covariates are at the same distance from a
    treated unit and both are its matches. The comparison that finds them
    squared the cut-off with the C library's pow and the distances with
    numpy; the two differ in the last place for about one number in 600,
    and the estimate was 1148.68.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sp.match(polynomial, y="re78", treat="treat", covariates=POLYNOMIAL,
                       distance="propensity", estimand="ATT", ties="all",
                       se_method="abadie_imbens_2016")  # fmt: skip
    assert res.estimate == pytest.approx(1170.9538992973, rel=1e-11)
    assert res.se == pytest.approx(1025.0810245086, rel=1e-8)


def test_redundant_covariate_next_to_large_columns(polynomial):
    """gen double u74 = re74 == 0 ; gen double u74b = u74
    teffects psmatch (re78) (treat <polynomial> u74 u74b, logit), atet
    psmatch2 treat <polynomial> u74 u74b, logit outcome(re78) ties

    Stata omits u74b. The fit here used to solve its Newton step from the
    singular Hessian; next to re74^2 the error of that step reached the
    scores (off by 1.7e-3) and the estimate was 605.80.
    """
    data = polynomial.assign(u74=(polynomial.re74 == 0).astype(float))
    data["u74b"] = data["u74"]
    covariates = POLYNOMIAL + ["u74", "u74b"]
    res = sp.pscore(data, "treat", covariates)
    assert res.loglik == pytest.approx(-193.3257071979, abs=1e-9)
    assert res.coefficients.loc["u74b", "coef"] == 0.0
    assert np.isnan(res.coefficients.loc["u74b", "se"])
    assert res.coefficients.loc["u74", "coef"] == pytest.approx(1.79573, abs=5e-6)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        te = sp.match(data, y="re78", treat="treat", covariates=covariates,
                      distance="propensity", estimand="ATT", ties="all",
                      se_method="abadie_imbens_2016")  # fmt: skip
        pm = sp.psmatch2(data, treat="treat", covariates=covariates,
                         outcome="re78", ties=True)  # fmt: skip
        one = sp.match(data, y="re78", treat="treat", covariates=covariates[:-1],
                       distance="propensity", estimand="ATT", ties="all")  # fmt: skip
    assert te.estimate == pytest.approx(468.1029554595, rel=1e-10)
    assert te.se == pytest.approx(1118.1602696816, rel=1e-7)
    assert pm.att == pytest.approx(468.1029554595, rel=1e-10)
    assert pm.se == pytest.approx(1420.4575884865, rel=1e-9)
    # the redundant column changes nothing
    assert one.estimate == pytest.approx(te.estimate, rel=1e-12)


def test_collinear_covariate_is_left_out(lalonde):
    from statspai.matching._binary_fit import fit_binary_index

    X = lalonde[["age", "educ", "married"]].to_numpy(float)
    t = lalonde["treat"].to_numpy(float)
    base = fit_binary_index(X, t)
    twice = fit_binary_index(np.column_stack([X, X[:, 0] + X[:, 1], X[:, 2]]), t)
    assert twice["omitted"] == [3, 4]
    assert twice["beta"][:4] == pytest.approx(base["beta"], rel=1e-9)
    assert twice["beta"][4:] == pytest.approx([0.0, 0.0])
    assert twice["loglik"] == pytest.approx(base["loglik"], rel=1e-12)
    assert base["converged"]


def test_scores_are_not_clipped():
    """A treated unit with a score of 1e-9 is matched to the control
    nearest to it, not to every control below 1e-6."""
    rng = np.random.default_rng(0)
    n = 4000
    x = rng.normal(size=n)
    d = (rng.random(n) < 1 / (1 + np.exp(-(-9 + 6 * x)))).astype(int)
    frame = pd.DataFrame({"x": x, "d": d, "y": x + rng.normal(size=n)})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sp.match(frame, y="y", treat="d", covariates=["x"],
                       distance="propensity", estimand="ATT")  # fmt: skip
    score = res.matched_data["_pscore"].to_numpy()
    assert score.min() < 1e-6
    assert np.unique(score[score < 1e-6]).size > 10


# ----------------------------------------------------------------- loneway
@pytest.mark.parametrize(
    "by, rho, se, lb, ub, sd_b, sd_w, rho_t",
    [
        # loneway re78 educ
        ("educ", 0.0223832869685335, 0.0205997774913334, 0.0, 0.0627581089410858,
         1119.431483681766, 7398.094841804445, 0.4019178598304161),
        # loneway re78 nodegree
        ("nodegree", 0.0391847029054058, 0.057853736784734, 0.0, 0.1525759433745446,
         1494.535669499729, 7400.617526410916, 0.9210741115060707),
    ],
)  # fmt: skip
def test_loneway_groups_of_unequal_size(
    lalonde, by, rho, se, lb, ub, sd_b, sd_w, rho_t
):
    e = sp.loneway(lalonde, "re78", by=by).estimates
    assert e["icc"] == pytest.approx(rho, rel=1e-12)
    assert e["icc_se"] == pytest.approx(se, rel=1e-12)
    assert e["icc_ci"] == pytest.approx((lb, ub), rel=1e-12)
    assert e["sd_between"] == pytest.approx(sd_b, rel=1e-12)
    assert e["sd_within"] == pytest.approx(sd_w, rel=1e-12)
    assert e["reliability"] == pytest.approx(rho_t, rel=1e-11)


def test_loneway_balanced_exact_and_level():
    """input y g ... ; replace y = y + mod(_n, 4)
    loneway y g, exact / loneway y g, level(90)"""
    y = np.array([1.0, 2, 3, 11, 12, 13, 21, 22, 23]) + np.arange(1, 10) % 4
    frame = pd.DataFrame({"y": y, "g": np.repeat([1, 2, 3], 3)})
    exact = sp.loneway(frame, "y", by="g", exact=True)
    assert exact.statistic == pytest.approx(90.25)
    e = exact.estimates
    assert e["icc"] == pytest.approx(0.967479674796748, rel=1e-13)
    assert e["icc_se"] == pytest.approx(0.0367371180606702, rel=1e-12)
    assert e["icc_ci"] == pytest.approx(
        (0.7921196235301388, 0.9991553255230533), rel=1e-12
    )
    assert e["sd_between"] == pytest.approx(9.620579793107874, rel=1e-13)
    assert e["sd_within"] == pytest.approx(1.763834207376394, rel=1e-13)
    assert e["reliability"] == pytest.approx(0.9889196675900277, rel=1e-13)
    # the normal interval is not cut at one
    ninety = sp.loneway(frame, "y", by="g", alpha=0.10).estimates["icc_ci"]
    assert ninety == pytest.approx((0.9070524929109103, 1.027906856682586), rel=1e-12)


def test_loneway_truncation_and_refusals():
    """A between-group mean square below the within-group one: the
    correlation is reported as zero and the group effect has no standard
    deviation (Stata prints `.`)."""
    frame = pd.DataFrame({"y": [1.0, 5, 9, 2, 5, 8, 1, 5, 9.5],
                          "g": [1, 1, 1, 2, 2, 2, 3, 3, 3]})  # fmt: skip
    e = sp.loneway(frame, "y", by="g").estimates
    assert e["icc"] == 0.0 and e["icc_truncated"]
    assert np.isnan(e["sd_between"])
    assert e["icc_ci"][0] == 0.0 and e["icc_ci"][1] > 0
    uneven = frame.iloc[:-1]
    with pytest.raises(sp.exceptions.MethodIncompatibility, match="equal size"):
        sp.loneway(uneven, "y", by="g", exact=True)
    with pytest.raises(sp.exceptions.MethodIncompatibility, match="alpha"):
        sp.loneway(frame, "y", by="g", alpha=1.5)


# ----------------------------------------------------------------- heckman
def test_heckman_twostep_reports_the_selection_equation(lalonde):
    """gen double y = re78 if re78 > 0
    heckman y age educ, select(age educ married) twostep"""
    data = lalonde.assign(y=lalonde.re78.where(lalonde.re78 > 0))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sp.heckman(data, y="y", x=["age", "educ"], z=["age", "educ", "married"])
    select = res.model_info["selection_equation"].set_index("variable")
    stata = {  # e(b) and the square roots of the diagonal of e(V)
        "age": (-0.01901229688, 0.0059545357),
        "educ": (0.0301499392, 0.0211553051),
        "married": (0.2397367265, 0.1242094756),
        "const": (0.8535162042, 0.2880902824),
    }
    for name, (coef, se) in stata.items():
        assert select.loc[name, "coefficient"] == pytest.approx(coef, rel=1e-8)
        # six digits: Stata's probit stops at its own tolerance (three
        # separate fits here, to machine precision, give 0.00595454327)
        assert select.loc[name, "se"] == pytest.approx(se, rel=5e-6)
    outcome = res.detail.set_index("variable")["coefficient"]
    assert outcome["age"] == pytest.approx(335.6192939, rel=1e-9)
    assert outcome["educ"] == pytest.approx(283.5991755, rel=1e-9)
    assert outcome["lambda (IMR)"] == pytest.approx(-20577.82141, rel=1e-9)
    assert res.model_info["wald_df"] == 2


def test_first_stage_summary_after_iv():
    data = sp.datasets.card_1995()
    res = sp.ivreg("lwage ~ exper + black + (educ ~ nearc4)", data=data)
    out = sp.estat(res, "firststage", print_results=False)
    stage = res.model_info["first_stage"][0]
    assert out["partial_r2"] == pytest.approx(stage["partial_r_squared"])
    assert out["statistic_label"] == f"F({out['df1']}, {out['df2']})"
    assert 0 <= out["pvalue"] <= 1
