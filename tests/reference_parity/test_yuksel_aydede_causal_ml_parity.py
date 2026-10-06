"""R parity for the causal chapters of Yuksel and Aydede, *Causal Inference
and Machine Learning: In Economics, Social, and Health Sciences*
(https://www.causalmlbook.com).

The book walks from randomised experiments through regression adjustment,
matching, weighting and doubly robust estimation to double machine
learning, meta-learners, causal forests and synthetic control, each time
with the R package that is standard for the step. This file pins the
StatsPAI function for each step to that package on the book's own
data-generating processes (at a smaller sample size).

Reference numbers: ``_fixtures/yuksel_aydede_causal_ml_R.json``, written by
``_fixtures/_generate_yuksel_aydede_causal_ml.R``; both sides read the same
CSV bytes.

Tolerances. Everything deterministic is held to the strict parity budget of
``CLAUDE.md`` section 5.1 (relative 1e-6), and most of it to 1e-9. Four
items are not parity claims and say so where they are tested:

* **Full matching.** ``MatchIt`` calls ``optmatch``, which solves the
  matching on distances rounded to a tolerance. StatsPAI solves it exactly,
  so its total distance is never larger, and the matched sets (hence the
  estimate) can differ. Pinned: the estimator given ``optmatch``'s own sets
  (1e-9), the optimality of the total distance, and agreement of the
  estimates to a few percent.
* **HC2 / HC3 for 2SLS.** ``estimatr`` takes the leverage of the
  second-stage regression and ``AER`` + ``sandwich`` the diagonal of the
  oblique projection. StatsPAI follows ``estimatr``; the test shows both.
* **glmnet.** ``sp.shrinkage`` standardises with the ``n - 1`` divisor and
  states its penalty on the residual sum of squares. The mapping to
  ``glmnet``'s ``lambda`` is exact and is the content of the test.
* **Bootstrap standard errors** of the generalized synthetic control are
  random on both sides (a screen, not parity).
"""

from __future__ import annotations

import itertools
import json
import warnings
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import statspai as sp

FIX = Path(__file__).parent / "_fixtures"
COV = ["income", "cont", "bin"]
TIGHT = 1e-9

pytestmark = pytest.mark.skipif(
    not (FIX / "yuksel_aydede_causal_ml_R.json").exists(),
    reason="Yuksel-Aydede fixture is not materialized",
)


@pytest.fixture(scope="module")
def ref():
    path = FIX / "yuksel_aydede_causal_ml_R.json"
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def sel():
    return pd.read_csv(FIX / "yuksel_aydede_selection.csv")


@pytest.fixture(scope="module")
def dml():
    return pd.read_csv(FIX / "yuksel_aydede_dml.csv")


@pytest.fixture(scope="module")
def gs():
    return pd.read_csv(FIX / "yuksel_aydede_gsynth.csv")


def _quiet(fn, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*args, **kwargs)


# --- randomised experiments ------------------------------------------------


def test_neyman_hc2_and_lin(sel, ref):
    r = ref["rct"]
    tt = sp.ttest(sel, "Y", by="D", unequal=True)
    assert tt.se == pytest.approx(r["neyman_se"], rel=TIGHT)
    ols = sp.regress("Y ~ D", sel, robust="hc2")
    assert ols.params["D"] == pytest.approx(r["hc2"][0], rel=TIGHT)
    assert ols.std_errors["D"] == pytest.approx(r["hc2"][1], rel=TIGHT)
    assert r["hc2"][1] == pytest.approx(r["neyman_se"], rel=1e-12)  # the identity
    lin = sp.lm_lin(sel, "Y", "D", ["income"], vce="hc2")
    assert lin.estimate == pytest.approx(r["lin"][0], rel=TIGHT)
    assert lin.se == pytest.approx(r["lin"][1], rel=TIGHT)


# --- subclassification and matching -----------------------------------------


@pytest.mark.parametrize("estimand", ["ATT", "ATE"])
def test_subclassification(sel, ref, estimand):
    df = sel.assign(stratum=ref["subclass"]["strata"])
    fit = sp.match(
        df,
        "Y",
        "D",
        ["income"],
        method="stratify",
        strata="stratum",
        estimand=estimand,
    )
    assert fit.estimate == pytest.approx(ref["subclass"][estimand.lower()], rel=TIGHT)


def test_matchit_nearest_mahalanobis_caliper(sel, ref):
    r = ref["match"]
    ps = sp.propensity_score(sel, "D", COV).to_numpy()
    assert ps == pytest.approx(np.array(r["ps"]), rel=1e-8)
    # MatchIt's default: 1:1 without replacement, treated units taken in
    # decreasing order of the score.
    nn = _quiet(
        sp.match,
        sel,
        "Y",
        "D",
        COV,
        distance="propensity",
        replace=False,
        m_order="largest",
    )
    assert nn.estimate == pytest.approx(r["nearest"], rel=TIGHT)
    mh = _quiet(
        sp.match,
        sel,
        "Y",
        "D",
        COV,
        distance="mahalanobis",
        replace=False,
        m_order="data",
    )
    assert mh.estimate == pytest.approx(r["mahalanobis"], rel=TIGHT)
    cal = _quiet(
        sp.match,
        sel,
        "Y",
        "D",
        COV,
        distance="propensity",
        replace=False,
        m_order="largest",
        caliper=0.1,
        caliper_scale="sd",
    )
    assert cal.estimate == pytest.approx(r["caliper"], rel=TIGHT)
    assert cal.model_info["n_matched_treated"] == r["caliper_n"]


@pytest.mark.parametrize("key", ["att", "ate", "att_tight", "ate_tight"])
def test_full_matching_estimator_given_optmatch_sets(sel, ref, key):
    """Weights, estimate and matched-set-clustered SE are MatchIt's
    (weighted lm + sandwich::vcovCL) when the sets are optmatch's."""
    from statspai.matching.full import matched_set_effect

    r = ref["full"][key]
    out = matched_set_effect(
        sel["Y"].to_numpy(),
        sel["D"].to_numpy(),
        np.array(r["subclass"]),
        key[:3].upper(),
    )
    assert out["estimate"] == pytest.approx(r["estimate"], rel=TIGHT)
    assert out["se"] == pytest.approx(r["se"], rel=TIGHT)
    assert out["n_sets"] == r["n_sets"]


@pytest.mark.parametrize("estimand", ["ATT", "ATE"])
def test_full_matching_is_at_least_as_good_as_optmatch(sel, ref, estimand):
    fit = sp.match(sel, "Y", "D", COV, method="full", estimand=estimand)
    assert isinstance(fit, sp.FullMatchResult)
    loose = ref["full"][estimand.lower()]
    tight = ref["full"][estimand.lower() + "_tight"]
    # exact optimum: never above optmatch, and equal to it (to its
    # tolerance) once that tolerance is tightened
    assert fit.total_distance <= loose["total_distance"] + 1e-12
    assert fit.total_distance <= tight["total_distance"] + 1e-12
    assert fit.total_distance == pytest.approx(tight["total_distance"], abs=1e-7)
    assert fit.n_unmatched == 0
    # the optimum is nearly flat, so the sets are not unique and the
    # estimates agree only roughly; this is not a parity claim
    assert fit.estimate == pytest.approx(tight["estimate"], rel=0.05)
    assert fit.se == pytest.approx(tight["se"], rel=0.05)
    assert np.abs(fit.balance["smd_matched"]).max() < 0.15
    assert np.abs(fit.balance["smd_unmatched"]).max() > 0.5


def test_full_matching_is_optimal_by_enumeration():
    """Every full matching of a small problem, enumerated as edge covers."""
    from statspai.matching.full import full_match_sets

    def brute(c):
        n1, n0 = c.shape
        edges = [(i, j) for i in range(n1) for j in range(n0)]
        best = np.inf
        for k in range(max(n1, n0), n1 + n0):
            for sub in itertools.combinations(edges, k):
                if len({i for i, _ in sub}) == n1 and len({j for _, j in sub}) == n0:
                    best = min(best, sum(c[i, j] for i, j in sub))
        return best

    rng = np.random.default_rng(0)
    for rep in range(40):
        n1, n0 = int(rng.integers(1, 4)), int(rng.integers(1, 5))
        c = rng.random((n1, n0))
        if rep % 2:
            c = np.round(c, 1)  # ties, including zero distances
        st, sc, total = full_match_sets(c)
        assert total == pytest.approx(brute(c), abs=1e-12)
        for s in np.unique(st):  # each set is a star
            assert (st == s).sum() == 1 or (sc == s).sum() == 1
        assert (st >= 0).all() and (sc >= 0).all()


def test_full_matching_caliper_and_supplied_score(sel):
    scored = sel.assign(score=sp.propensity_score(sel, "D", COV).to_numpy())
    a = sp.full_match(sel, "Y", "D", COV)
    b = sp.full_match(scored, "Y", "D", COV, pscore="score")
    assert b.total_distance == pytest.approx(a.total_distance, rel=1e-9)
    with pytest.warns(RuntimeWarning, match="no partner within the caliper"):
        c = sp.full_match(sel, "Y", "D", COV, caliper=0.005, caliper_scale="raw")
    assert c.n_unmatched > 0
    assert c.subclass.isna().sum() == c.n_unmatched
    assert (c.weights[c.subclass.isna()] == 0).all()
    only = sp.full_match(sel, None, "D", COV)
    assert np.isnan(only.estimate) and only.n_sets == a.n_sets


# --- weighting and doubly robust estimation ---------------------------------


def test_hajek_ipw_and_textbook_aipw(sel, ref):
    r = ref["weighting"]
    ipw = _quiet(sp.ipw, sel, "Y", "D", COV, se_method="sandwich")
    assert ipw.estimate == pytest.approx(r["hajek"], rel=1e-8)
    fit = sp.aipw(sel, "Y", "D", COV, cross_fit=False, trim=0)
    assert fit.estimate == pytest.approx(r["aipw"], rel=1e-8)
    assert fit.se == pytest.approx(r["aipw_se"], rel=1e-8)
    assert fit.model_info["n_propensity_clipped"] == 0


def test_aipw_reports_the_scores_it_clips(sel, ref):
    """Before 1.39 the count was always 0 and nothing was said."""
    r = ref["weighting"]
    with pytest.warns(RuntimeWarning, match="clipped"):
        fit = sp.aipw(sel, "Y", "D", COV, cross_fit=False)
    assert fit.model_info["n_propensity_clipped"] == r["n_outside_01_99"] > 0
    assert fit.model_info["trim"] == 0.01
    assert abs(fit.estimate - r["aipw"]) > 1e-3  # clipping moves the estimate
    from statspai.exceptions import MethodIncompatibility

    with pytest.raises(MethodIncompatibility, match="trim"):
        sp.aipw(sel, "Y", "D", COV, trim=0.6)


# --- double machine learning -------------------------------------------------


def test_dml_plr_and_pliv_with_rlasso(dml, ref):
    xs = [f"X{j}" for j in range(1, 7)]
    labels = dml["fold"].to_numpy()
    plr = _quiet(
        sp.dml,
        dml,
        "Y",
        "D",
        xs,
        model="plr",
        ml_g=sp.RlassoRegressor(),
        ml_m=sp.RlassoRegressor(),
        fold_indices=labels,
    )
    assert plr.estimate == pytest.approx(ref["dml"]["plr"][0], rel=TIGHT)
    assert plr.se == pytest.approx(ref["dml"]["plr"][1], rel=TIGHT)
    pliv = _quiet(
        sp.dml,
        dml,
        "Y",
        "D",
        xs,
        model="pliv",
        instrument="Z",
        ml_g=sp.RlassoRegressor(),
        ml_m=sp.RlassoRegressor(),
        ml_r=sp.RlassoRegressor(),
        fold_indices=labels,
    )
    assert pliv.estimate == pytest.approx(ref["dml"]["pliv"][0], rel=TIGHT)
    assert pliv.se == pytest.approx(ref["dml"]["pliv"][1], rel=TIGHT)


def test_dml_reads_scikit_learn_splits(dml):
    xs = [f"X{j}" for j in range(1, 7)]
    labels = dml["fold"].to_numpy()
    splits = [
        (np.flatnonzero(labels != k), np.flatnonzero(labels == k)) for k in range(1, 6)
    ]
    kw = dict(model="plr", ml_g="linear", ml_m="linear")
    a = _quiet(sp.dml, dml, "Y", "D", xs, fold_indices=labels, **kw)
    b = _quiet(sp.dml, dml, "Y", "D", xs, fold_indices=splits, **kw)
    assert (a.estimate, a.se) == (b.estimate, b.se)
    from statspai.exceptions import MethodIncompatibility

    bad = [(tr[:-1], te) for tr, te in splits]
    with pytest.raises(MethodIncompatibility, match="complement"):
        sp.dml(dml, "Y", "D", xs, fold_indices=bad, **kw)
    overlap = [splits[0], splits[0], *splits[2:]]
    with pytest.raises(MethodIncompatibility, match="overlap"):
        sp.dml(dml, "Y", "D", xs, fold_indices=overlap, **kw)


def test_iv_hc_leverage_convention(ref):
    res = pd.DataFrame(ref["dml"]["resid"])
    r = ref["dml"]["iv_se"]
    se = {
        t: sp.iv("Yr ~ (Dr ~ Zr)", res, robust=t).std_errors["Dr"]
        for t in ("hc1", "hc2", "hc3")
    }
    assert se["hc1"] == pytest.approx(r["hc1"], rel=TIGHT)
    assert se["hc2"] == pytest.approx(r["estimatr_hc2"], rel=TIGHT)
    assert se["hc3"] == pytest.approx(r["estimatr_hc3"], rel=TIGHT)
    # AER + sandwich use another leverage; the difference is real and small
    assert 1e-5 < abs(se["hc3"] / r["aer_hc3"] - 1) < 5e-3


@pytest.mark.parametrize("method", ["lasso", "ridge"])
def test_shrinkage_reproduces_glmnet_under_the_stated_mapping(dml, ref, method):
    """glmnet minimises RSS / (2n) + lambda * penalty on predictors scaled
    by the divisor-n standard deviation, and for the ridge on an outcome
    scaled to unit variance as well. In sp.shrinkage's terms:
    lasso  penalty = 2 n lambda sqrt((n - 1) / n);
    ridge  penalty = n (lambda / sd_y) (n - 1) / n."""
    xs = [f"X{j}" for j in range(1, 7)]
    g = ref["glmnet"]
    n = len(dml)
    coefs = np.array(g[method])
    for j, lam in enumerate(g["lambda"]):
        pen = (
            2 * n * lam * np.sqrt((n - 1) / n)
            if method == "lasso"
            else n * lam / g["sd_y"] * (n - 1) / n
        )
        fit = sp.shrinkage(dml, "Y", xs, method=method, penalty=pen)
        r_pred = coefs[0, j] + dml[xs].to_numpy() @ coefs[1:, j]
        assert np.asarray(fit.predict(dml)) == pytest.approx(r_pred, abs=1e-8)


# --- meta-learners ------------------------------------------------------------


def test_meta_learners_take_a_learner_per_arm(ref):
    from sklearn.linear_model import LinearRegression, LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import FunctionTransformer

    df = pd.read_csv(FIX / "yuksel_aydede_meta.csv")

    def quad():
        return make_pipeline(
            FunctionTransformer(lambda a: np.column_stack([a, a[:, 0] ** 2])),
            LinearRegression(),
        )

    t = _quiet(
        sp.metalearner,
        df,
        "Y",
        "W",
        ["X1", "X2"],
        learner="t",
        outcome_model=(LinearRegression(), quad()),
    )
    assert np.asarray(t.model_info["cate"]) == pytest.approx(
        np.array(ref["metalearner"]["t"]), abs=1e-9
    )
    x = _quiet(
        sp.metalearner,
        df,
        "Y",
        "W",
        ["X1", "X2"],
        learner="x",
        outcome_model=(LinearRegression(), quad()),
        cate_model=(quad(), LinearRegression()),
        propensity_model=LogisticRegression(penalty=None, tol=1e-12, max_iter=5000),
    )
    assert np.asarray(x.model_info["cate"]) == pytest.approx(
        np.array(ref["metalearner"]["x"]), abs=1e-6
    )
    from statspai.exceptions import MethodIncompatibility

    with pytest.raises(MethodIncompatibility, match="pair"):
        sp.metalearner(
            df,
            "Y",
            "W",
            ["X1", "X2"],
            learner="s",
            outcome_model=(LinearRegression(), LinearRegression()),
        )


# --- generalized synthetic control ---------------------------------------------


@pytest.mark.parametrize("r", [0, 1, 2, 3])
def test_gsynth_several_treated_units_matches_r(gs, ref, r):
    with_x = ref["gsynth"]["cov"][r]
    a = sp.gsynth(
        gs,
        "Y",
        "id",
        "time",
        treat="D",
        covariates=["X1", "X2"],
        n_factors=r,
        inference="none",
    )
    assert a.estimate == pytest.approx(with_x["att_avg"], rel=1e-8)
    assert a.model_info["beta"].to_numpy() == pytest.approx(with_x["beta"], rel=1e-7)
    assert a.detail["att"].to_numpy() == pytest.approx(with_x["att"][-10:], abs=1e-7)
    without = ref["gsynth"]["nocov"][r]
    b = sp.gsynth(gs, "Y", "id", "time", treat="D", n_factors=r, inference="none")
    assert b.estimate == pytest.approx(without["att_avg"], rel=TIGHT)


def test_gsynth_staggered_adoption_matches_r(gs, ref):
    r = ref["gsynth"]["staggered"]
    fit = sp.gsynth(
        gs,
        "Y",
        "id",
        "time",
        treat="D_stag",
        covariates=["X1", "X2"],
        n_factors=2,
        inference="none",
    )
    assert fit.estimate == pytest.approx(r["att_avg"], rel=1e-8)
    assert sorted(fit.model_info["n_pre_periods"].values()) == [20, 20, 20, 23, 25]
    assert fit.detail["n_units"].tolist() == [5] * 5 + [4, 4] + [3] * 3


def test_gsynth_single_unit_with_covariates_is_now_xu_2017(gs, ref):
    """⚠️ Correctness: the covariate coefficients used to come from a pooled
    regression without fixed effects (6.68 on these data, against 5.70)."""
    r = ref["gsynth"]["one_unit"]
    others = gs.loc[(gs["D"] == 1) & (gs["id"] != r["unit"]), "id"].unique()
    one = gs[~gs["id"].isin(others)]
    fit = sp.gsynth(
        one,
        "Y",
        "id",
        "time",
        r["unit"],
        21,
        covariates=["X1", "X2"],
        n_factors=2,
        inference="none",
    )
    assert fit.estimate == pytest.approx(r["att_avg"], rel=1e-8)
    assert fit.model_info["beta"].to_numpy() == pytest.approx(r["beta"], rel=1e-7)


def test_gsynth_cross_validation_and_bootstrap(gs, ref):
    """The number of factors is the reference's (and the truth, 2). The
    parametric bootstrap SE is random on both sides: a screen."""
    r = ref["gsynth"]["cv"]
    fit = sp.gsynth(
        gs, "Y", "id", "time", treat="D", covariates=["X1", "X2"], seed=1, n_boot=100
    )
    assert fit.model_info["n_factors"] == r["r_cv"] == 2
    assert fit.model_info["inference"] == "parametric"
    assert fit.estimate == pytest.approx(r["att_avg"], rel=1e-3)  # R: tol 1e-3
    assert 0.6 < fit.se / r["se"] < 1.6
    assert fit.ci[0] < fit.estimate < fit.ci[1]
    via = sp.synth(
        gs,
        outcome="Y",
        unit="id",
        time="time",
        treatment="D",
        method="gsynth",
        covariates=["X1", "X2"],
        n_factors=2,
        n_boot=5,
        seed=1,
    )
    assert via.estimate == pytest.approx(ref["gsynth"]["cov"][2]["att_avg"], rel=1e-8)


def test_gsynth_refuses_what_it_cannot_do(gs):
    from statspai.exceptions import MethodIncompatibility

    with pytest.raises(MethodIncompatibility, match="not both"):
        sp.gsynth(gs, "Y", "id", "time", 101, 21, treat="D")
    with pytest.raises(MethodIncompatibility, match="say who is treated"):
        sp.gsynth(gs, "Y", "id", "time")
    reversal = gs.copy()
    reversal.loc[(reversal["id"] == 101) & (reversal["time"] == 30), "D"] = 0
    with pytest.raises(MethodIncompatibility, match="switches off"):
        sp.gsynth(reversal, "Y", "id", "time", treat="D")
    with pytest.raises(MethodIncompatibility, match="balanced"):
        sp.gsynth(gs.iloc[1:], "Y", "id", "time", treat="D")


# --- causal forest ---------------------------------------------------------------


@pytest.mark.parametrize(
    "key,clustered,equalize",
    [
        ("rows", False, False),
        ("clusters", True, False),
        ("clusters_equalized", True, True),
    ],
)
def test_forest_subset_average_matches_grf(ref, key, clustered, equalize):
    """grf::average_treatment_effect(subset=) on grf's own forest outputs."""
    from statspai.forest import _grf_inference as gi
    from statspai.forest._grf_fit import observation_weights

    df = pd.read_csv(FIX / "grf_cluster_operator_data.csv")
    n = len(df)
    codes = np.unique(df["cluster"].to_numpy(), return_inverse=True)[1].astype(np.int64)
    forest = SimpleNamespace(
        _engine=object(),
        fe=None,
        _X_original=df[["x1", "x2", "x3"]].to_numpy(),
        _Y_original=df["Y"].to_numpy(),
        _T_original=df["W"].to_numpy(),
        _m_insample=df["Y_hat"].to_numpy(),
        _e_insample=df["W_hat"].to_numpy(),
        _oob_tau=df["tau_oob"].to_numpy(),
        _clusters=codes if clustered else None,
        _observation_weight=(
            observation_weights(codes, equalize, n) if clustered else np.ones(n)
        ),
    )
    mask = (df["x1"] > 0).to_numpy()
    for row in ref["grf_subset"][key]:
        got = gi.average_effect(
            forest, row["target"], alpha=0.05, clip=0.0, subset=mask
        )
        assert got["estimate"] == pytest.approx(row["estimate"], rel=TIGHT)
        assert got["se"] == pytest.approx(row["se"], rel=TIGHT)
        again = gi.average_effect(
            forest, row["target"], alpha=0.05, clip=0.0, subset=np.flatnonzero(mask)
        )
        assert again["estimate"] == got["estimate"]
        assert got["n"] == int(mask.sum()) and got["n_fit"] == n


def test_forest_subset_and_categorical_columns_end_to_end():
    from statspai.exceptions import DataInsufficient, MethodIncompatibility

    rng = np.random.default_rng(0)
    n = 600
    df = pd.DataFrame(
        {
            "x": rng.normal(size=n),
            "region": rng.choice(["north", "south", "west"], n),
            "w": rng.binomial(1, 0.5, n),
        }
    )
    tau = 1.0 + 2.0 * (df["region"] == "south")
    df["y"] = df["x"] + tau * df["w"] + rng.normal(size=n)
    cf = sp.causal_forest(
        data=df, y="y", d="w", x=["x", "region"], n_estimators=300, random_state=1
    )
    assert cf._feature_names == ["x", "region[north]", "region[south]", "region[west]"]
    south = (df["region"] == "south").to_numpy()
    hi = cf.average_treatment_effect(subset=south)
    lo = cf.average_treatment_effect(subset=~south)
    assert hi["estimate"] - lo["estimate"] > 1.0  # truth: 2
    whole = cf.average_treatment_effect()
    assert cf.average_treatment_effect(subset=np.ones(n, bool))["estimate"] == (
        whole["estimate"]
    )
    # prediction rebuilds the indicators from the raw column
    new = df.head(5)[["x", "region"]]
    assert cf.effect(new).shape == (5,)
    with pytest.raises(MethodIncompatibility, match="not in the fitted data"):
        cf.effect(new.assign(region="east"))
    with pytest.raises(DataInsufficient, match="no rows"):
        cf.average_treatment_effect(subset=np.zeros(n, bool))
    with pytest.raises(MethodIncompatibility, match="one entry per"):
        cf.average_treatment_effect(subset=np.ones(n - 1, bool))
