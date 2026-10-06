"""``sp.dml_did`` against the ``DoubleML`` Python package.

``DoubleML`` is the reference implementation of the double machine learning
difference-in-differences estimators (``DoubleMLDID`` for a two-period
panel, ``DoubleMLDIDCS`` for repeated cross-sections). With the sample split
fixed by a fold column and deterministic learners on both sides, the two
compute the same sums; the fixture is written by
``_fixtures/_generate_dml_did_doubleml.py`` and both sides read the same
CSV bytes.

Tolerance. ``rtol = 1e-8``: the only iterative step is the unpenalised
logit, solved to ``tol = 1e-12``; a formula difference (a normalisation, a
fold-specific versus full-sample treated share) would show at 1e-3 or more.
The remaining tests need no reference: an algebraic identity with the score
of Chang (2020), the long-panel layout, and recovery of a known effect with
nominal coverage under a non-linear trend that a linear model gets wrong.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from sklearn.linear_model import LinearRegression, LogisticRegression

import statspai as sp

FIX = Path(__file__).parent / "_fixtures"
X = ["x1", "x2", "x3", "x4"]
RTOL = 1e-8

pytestmark = pytest.mark.skipif(
    not (FIX / "dml_did_doubleml.json").exists(),
    reason="DoubleML DiD fixture is not materialized",
)


@pytest.fixture(scope="module")
def ref():
    return json.loads((FIX / "dml_did_doubleml.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def df():
    return pd.read_csv(FIX / "dml_did_doubleml.csv")


def _logit():
    return LogisticRegression(penalty=None, tol=1e-12, max_iter=5000)


@pytest.mark.parametrize("row", range(4))
def test_panel_matches_doubleml(df, ref, row):
    r = ref["panel"][row]
    fit = sp.dml_did(
        df,
        "dy",
        "d",
        X,
        ml_g=LinearRegression(),
        ml_m=_logit(),
        score=r["score"],
        in_sample_normalization=r["in_sample_normalization"],
        fold_indices="fold",
    )
    assert fit.estimate == pytest.approx(r["estimate"], rel=RTOL)
    assert fit.se == pytest.approx(r["se"], rel=RTOL)


@pytest.mark.parametrize("row", range(4))
def test_repeated_cross_sections_match_doubleml(df, ref, row):
    r = ref["rcs"][row]
    fit = sp.dml_did(
        df,
        "y",
        "d",
        X,
        time="t",
        ml_g=LinearRegression(),
        ml_m=_logit(),
        score=r["score"],
        in_sample_normalization=r["in_sample_normalization"],
        fold_indices="fold",
    )
    assert fit.model_info["layout"] == "repeated cross-sections"
    assert fit.estimate == pytest.approx(r["estimate"], rel=RTOL)
    assert fit.se == pytest.approx(r["se"], rel=RTOL)


def test_unnormalised_score_is_changs_estimator(df):
    """theta = mean((D - m) / (1 - m) * (dY - g0)) / mean(D): the treated
    share cancels, so this is Chang's (2020) estimator with the share taken
    on the full sample."""
    fit = sp.dml_did(
        df,
        "dy",
        "d",
        X,
        ml_g=LinearRegression(),
        ml_m=_logit(),
        in_sample_normalization=False,
        fold_indices="fold",
        trimming_threshold=0.0,
    )
    d = df["d"].to_numpy(float)
    dy = df["dy"].to_numpy(float)
    g0 = np.empty(len(df))
    m = np.empty(len(df))
    for k in range(5):
        tr, te = (df["fold"] != k).to_numpy(), (df["fold"] == k).to_numpy()
        ctrl = tr & (d == 0)
        g0[te] = (
            LinearRegression().fit(df.loc[ctrl, X], dy[ctrl]).predict(df.loc[te, X])
        )
        m[te] = _logit().fit(df.loc[tr, X], d[tr]).predict_proba(df.loc[te, X])[:, 1]
    theta = np.mean((d - m) / (1 - m) * (dy - g0)) / np.mean(d)
    assert fit.estimate == pytest.approx(theta, rel=1e-12)


def test_long_panel_equals_differenced_input(df):
    """A long two-period panel is differenced within unit; treat may be the
    group indicator or group x post."""
    wide = df.assign(id=np.arange(len(df)))
    pre = wide[["id", *X, "d"]].assign(period=2019, outcome=0.25 * wide["x1"])
    post = wide[["id", *X, "d"]].assign(
        period=2021, outcome=0.25 * wide["x1"] + wide["dy"]
    )
    # covariates measured after treatment must be ignored: perturb them
    post[X] = post[X] + 5.0
    long = pd.concat([post, pre], ignore_index=True).sample(frac=1.0, random_state=0)
    kw = dict(ml_g=LinearRegression(), ml_m=_logit(), random_state=3)
    a = sp.dml_did(df, "dy", "d", X, **kw)
    b = sp.dml_did(
        long.sort_values(["id", "period"]),
        "outcome",
        "d",
        X,
        time="period",
        id="id",
        **kw,
    )
    assert b.model_info["layout"] == "panel"
    assert b.estimate == pytest.approx(a.estimate, rel=1e-10)
    assert b.se == pytest.approx(a.se, rel=1e-10)
    long["d_post"] = long["d"] * (long["period"] == 2021)
    c = sp.dml_did(
        long.sort_values(["id", "period"]),
        "outcome",
        "d_post",
        X,
        time="period",
        id="id",
        **kw,
    )
    assert c.estimate == pytest.approx(a.estimate, rel=1e-10)


def test_recovers_known_att_where_linear_did_does_not():
    """Known truth: ATT = 2, with a trend and a propensity that are quadratic
    in a confounder. Regression DiD with the confounder entered linearly is
    biased by about one; with learners that can represent the quadratic the
    cross-fitted estimator is centred on 2 and its interval covers at close
    to the nominal rate. 200 replications: the Monte Carlo SE of the mean
    estimate is 0.012 and of the coverage 0.016.

    A random forest is not used here on purpose. On this design, with
    leaves of 10 and 20 observations, its bias in the two nuisances leaves
    the estimate at 2.69 (40 replications, Monte Carlo SE 0.03); that falls
    slowly with n, to 2.26 at n = 5,000 and 2.12 at n = 20,000. The learner
    has to be able to fit the nuisance for the theory to bite."""
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import PolynomialFeatures

    def g():
        return make_pipeline(PolynomialFeatures(2), LinearRegression())

    def m():
        return make_pipeline(
            PolynomialFeatures(2), LogisticRegression(penalty=None, max_iter=2000)
        )

    rng = np.random.default_rng(0)
    est, cover, lin = [], [], []
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(200):
            n = 600
            x = rng.normal(size=(n, 2))
            d = rng.binomial(1, 1 / (1 + np.exp(-0.8 * x[:, 0] ** 2 + 0.8)))
            dy = 1 + 1.5 * x[:, 0] ** 2 + 2.0 * d + rng.normal(size=n)
            df = pd.DataFrame(x, columns=["a", "b"]).assign(d=d, dy=dy)
            fit = sp.dml_did(df, "dy", "d", ["a", "b"], ml_g=g(), ml_m=m(), n_folds=3)
            est.append(fit.estimate)
            cover.append(fit.ci[0] <= 2.0 <= fit.ci[1])
            X1 = np.column_stack([np.ones(n), d, x])
            lin.append(np.linalg.lstsq(X1, dy, rcond=None)[0][1])
    assert abs(np.mean(est) - 2.0) < 0.05
    assert 0.88 < np.mean(cover) <= 1.0
    assert np.mean(lin) - 2.0 > 0.5  # the linear specification is badly off


def test_inputs_fail_loudly(df):
    from statspai.exceptions import DataInsufficient, MethodIncompatibility

    with pytest.raises(MethodIncompatibility, match="score"):
        sp.dml_did(df, "dy", "d", X, score="aipw")
    with pytest.raises(MethodIncompatibility, match="n_rep=1"):
        sp.dml_did(df, "dy", "d", X, fold_indices="fold", n_rep=2)
    with pytest.raises(MethodIncompatibility, match="exactly two"):
        sp.dml_did(df.assign(t=np.arange(len(df)) % 3), "y", "d", X, time="t")
    with pytest.raises(MethodIncompatibility, match="0/1"):
        sp.dml_did(df.assign(d=df["d"] * 2), "dy", "d", X)
    with pytest.raises(DataInsufficient, match="pre-period"):
        sp.dml_did(df.assign(dd=df["d"] * df["t"]), "y", "dd", X, time="t")
    with pytest.raises(MethodIncompatibility, match="at least one covariate"):
        sp.dml_did(df, "dy", "d", [])


def test_clipped_propensities_are_reported():
    rng = np.random.default_rng(1)
    n = 500
    x = rng.normal(size=n)
    d = rng.binomial(1, 1 / (1 + np.exp(-4 * x)))
    df = pd.DataFrame({"x": x, "d": d, "dy": x + d + rng.normal(size=n)})
    with pytest.warns(RuntimeWarning, match="clipped"):
        fit = sp.dml_did(df, "dy", "d", ["x"], ml_g="linear", ml_m="logit")
    assert fit.model_info["n_propensity_clipped"] > 0


def test_repetitions_aggregate_by_median(df):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = sp.dml_did(df, "dy", "d", X, ml_g="linear", ml_m="logit", n_rep=5)
    reps = np.array(fit.model_info["theta_reps"])
    ses = np.array(fit.model_info["se_reps"])
    assert fit.estimate == pytest.approx(np.median(reps))
    assert fit.se == pytest.approx(
        np.sqrt(np.median(ses**2 + (reps - np.median(reps)) ** 2))
    )
