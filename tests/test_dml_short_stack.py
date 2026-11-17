"""Short stacking inside ``sp.dml(model='plr')``.

Three kinds of evidence, all identities or known answers:

* ``sp.dml(ml_g='stack', ml_m='stack')`` is ``sp.dml_model_averaging``
  (the estimator the 1.45 default was announced as), on every option the
  two share;
* a roster of one learner is that learner, on every path of ``sp.dml``
  that the single-learner estimator has (IV-type score, two-way clusters,
  survey design, weights, repeated cross-fitting);
* the stacking weights are the constrained least-squares solution, so the
  stacked nuisance cannot have a larger risk than any candidate.
"""

import warnings

import numpy as np
import pandas as pd
import pytest
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LassoCV, LinearRegression, RidgeCV

import statspai as sp
from statspai.exceptions import MethodIncompatibility

warnings.filterwarnings("ignore")

COV = [f"x{j}" for j in range(5)]
THETA = 1.5


def _data(n=360, seed=0):
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, 5))
    g1 = rng.integers(0, 18, n)
    g2 = rng.integers(0, 15, n)
    a1, a2 = rng.normal(size=18), rng.normal(size=15)
    D = X[:, 0] + 0.5 * X[:, 1] ** 2 + 0.3 * a1[g1] + rng.normal(size=n)
    Y = (
        THETA * D
        + np.sin(X[:, 0])
        + X[:, 1] * X[:, 2]
        + 0.3 * a1[g1]
        + 0.3 * a2[g2]
        + rng.normal(size=n)
    )
    df = pd.DataFrame({"y": Y, "d": D, **{c: X[:, j] for j, c in enumerate(COV)}})
    df["g1"], df["g2"] = g1, g2
    df["w"] = rng.uniform(0.5, 2.0, n)
    df["stratum"] = g1 % 3
    return df


def _roster():
    return [
        ("lasso", LassoCV(cv=3)),
        ("ridge", RidgeCV()),
        ("rf", RandomForestRegressor(n_estimators=25, random_state=0, n_jobs=1)),
    ]


def _triples():
    return [(est, est, label) for label, est in _roster()]


# ---------------------------------------------------------------------------
# sp.dml with a roster == sp.dml_model_averaging
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "extra",
    [
        {},
        {"sample_weight": "w"},
        {"cluster": "g1"},
        {"n_rep": 3},
        {"cluster": "g1", "n_rep": 2, "sample_weight": "w"},
    ],
    ids=["plain", "weights", "cluster", "n_rep", "cluster+n_rep+weights"],
)
def test_roster_in_sp_dml_equals_dml_model_averaging(extra):
    df = _data()
    r = sp.dml(
        df,
        y="y",
        treat="d",
        covariates=COV,
        ml_g=_roster(),
        ml_m=_roster(),
        n_folds=3,
        random_state=7,
        **extra,
    )
    ma = sp.dml_model_averaging(
        df,
        y="y",
        treat="d",
        covariates=COV,
        candidates=_triples(),
        n_folds=3,
        seed=7,
        **extra,
    )
    assert r.estimate == pytest.approx(ma.estimate, rel=1e-12)
    assert r.se == pytest.approx(ma.se, rel=1e-12)
    for side in ("ml_g", "ml_m"):
        w = r.model_info["stacking"][side]["weights"]
        ref = ma.model_info["weights_g" if side == "ml_g" else "weights_m"]
        assert list(w) == ["lasso", "ridge", "rf"]
        assert [w[k] for k in w] == pytest.approx([ref[k] for k in w], abs=1e-12)


def test_stack_alias_is_the_announced_default_roster():
    """``'stack'`` is the default roster of ``sp.dml_model_averaging``, the
    estimator the 1.45 change was announced as."""
    df = _data(n=240, seed=3)
    r = sp.dml(
        df,
        y="y",
        treat="d",
        covariates=COV,
        ml_g="stack",
        ml_m="stack",
        n_folds=2,
        random_state=1,
    )
    ma = sp.dml_model_averaging(df, y="y", treat="d", covariates=COV, n_folds=2, seed=1)
    assert r.estimate == pytest.approx(ma.estimate, rel=1e-12)
    assert r.se == pytest.approx(ma.se, rel=1e-12)
    assert r.model_info["stacking"]["ml_g"]["candidates"] == [
        "lasso",
        "ridge",
        "rf",
        "gbm",
    ]
    assert r.model_info["ml_g"] == "ShortStack"
    assert "learner_note" not in r.model_info  # not the gradient-boosting default


# ---------------------------------------------------------------------------
# a roster of one learner is that learner, on every path
# ---------------------------------------------------------------------------


def _design(df):
    return sp.svydesign(df, weights="w", strata="stratum", cluster="g1", nest=True)


PATHS = {
    "iv_type": lambda df: {"score": "IV-type"},
    "twoway_cluster": lambda df: {"cluster": ["g1", "g2"]},
    "design": lambda df: {"design": _design(df)},
    "iv_type+twoway": lambda df: {"score": "IV-type", "cluster": ["g1", "g2"]},
    "weights+cluster": lambda df: {"sample_weight": "w", "cluster": "g1"},
    "twoway+n_rep": lambda df: {"cluster": ["g1", "g2"], "n_rep": 2},
}


@pytest.mark.parametrize("path", list(PATHS))
def test_roster_of_one_is_the_single_learner(path):
    df = _data()
    rf = RandomForestRegressor(n_estimators=25, random_state=0, n_jobs=1)
    common = dict(y="y", treat="d", covariates=COV, n_folds=3, random_state=11)
    single = sp.dml(df, ml_g=rf, ml_m=rf, **common, **PATHS[path](df))
    stack = sp.dml(df, ml_g=[rf], ml_m=[rf], **common, **PATHS[path](df))
    assert stack.estimate == pytest.approx(single.estimate, rel=1e-12)
    assert stack.se == pytest.approx(single.se, rel=1e-12)
    assert stack.ci[0] == pytest.approx(single.ci[0], rel=1e-12)
    assert stack.model_info["stacking"]["ml_m"]["weights"] == {
        "RandomForestRegressor": 1.0
    }


def test_iv_type_with_linear_candidates_is_partialling_out():
    """With a linear smoother the IV-type and partialling-out scores give
    one estimate (the identity ``plr.py`` documents). It holds for a roster
    only if the third nuisance of the IV-type score goes through the
    roster too, which is what ``ml_g_iv`` records."""
    df = _data()
    kw = dict(y="y", treat="d", covariates=COV, ml_g=["ols"], ml_m=["ols"], n_folds=3)
    po = sp.dml(df, **kw)
    iv = sp.dml(df, score="IV-type", **kw)
    assert iv.estimate == pytest.approx(po.estimate, rel=1e-10)
    assert "ml_g_iv" in iv.model_info["stacking"]
    assert "ml_g_iv" not in po.model_info["stacking"]


# ---------------------------------------------------------------------------
# the weights are the constrained least-squares solution
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("path", ["plain", "iv_type", "twoway_cluster", "design"])
def test_stacked_risk_is_not_above_any_candidate(path):
    df = _data()
    extra = {} if path == "plain" else PATHS[path](df)
    r = sp.dml(
        df,
        y="y",
        treat="d",
        covariates=COV,
        ml_g=_roster(),
        ml_m=_roster(),
        n_folds=3,
        **extra,
    )
    assert np.isfinite(r.estimate) and r.se > 0
    info = r.model_info["stacking"]
    sw = df["w"].to_numpy() if path == "design" else np.ones(len(df))
    for side, resid in (("ml_g", "_y_resid"), ("ml_m", "_d_resid")):
        w = np.array(list(info[side]["weights"].values()))
        assert w.min() >= 0 and w.sum() == pytest.approx(1.0, abs=1e-12)
        stacked = float(np.sum(sw * r.model_info[resid] ** 2) / sw.sum())
        assert stacked <= min(info[side]["mse"].values()) * (1 + 1e-12)
    assert ("ml_g_iv" in info) == (path == "iv_type")


def test_recovers_a_known_coefficient_on_the_new_paths():
    """theta = 1.5; each interval is checked against the truth."""
    df = _data(n=1500, seed=5)
    for extra in (
        {"score": "IV-type"},
        {"cluster": ["g1", "g2"]},
        {"design": _design(df)},
    ):
        r = sp.dml(
            df,
            y="y",
            treat="d",
            covariates=COV,
            ml_g=_roster(),
            ml_m=_roster(),
            n_folds=3,
            **extra,
        )
        assert abs(r.estimate - THETA) < 4 * r.se, extra
        assert r.se < 0.1


# ---------------------------------------------------------------------------
# what is refused, and what stays as it was
# ---------------------------------------------------------------------------


def test_sensitivity_benchmark_refits_the_roster():
    """The benchmark's ``delta_theta`` is the short fit minus the long fit;
    the short fit must stack the same roster on the same partition."""
    df = _data()
    kw = dict(y="y", treat="d", ml_g=["ols", "ridge"], ml_m=["ols", "ridge"], n_folds=3)
    long = sp.dml(df, covariates=COV, **kw)
    short = sp.dml(df, covariates=COV[1:], **kw)
    s = sp.dml_sensitivity(long, cf_y=0.03, cf_d=0.03, benchmark_covariates=["x0"])
    assert float(s.benchmarks["delta_theta"].iloc[0]) == pytest.approx(
        short.estimate - long.estimate, rel=1e-10
    )


def test_several_treatments_take_a_roster():
    df = _data()
    df["d2"] = df["x2"] + np.random.default_rng(1).normal(size=len(df))
    out = sp.dml(
        df,
        y="y",
        treat=["d", "d2"],
        covariates=COV,
        ml_g=["ols", "ridge"],
        ml_m=["ols", "ridge"],
        n_folds=3,
    )
    one = sp.dml(
        df,
        y="y",
        treat="d",
        covariates=COV + ["d2"],
        ml_g=["ols", "ridge"],
        ml_m=["ols", "ridge"],
        n_folds=3,
    )
    assert out.params["d"] == pytest.approx(one.estimate, rel=1e-12)


def test_oof_options_stay_irm_only_under_a_roster():
    """``store_oof`` / ``external_predictions`` are IRM-ATE features; a PLR
    fit refuses them whatever its learners are."""
    df = _data(n=120)
    with pytest.raises(NotImplementedError, match="model='irm'"):
        sp.dml(
            df,
            y="y",
            treat="d",
            covariates=COV,
            ml_g="stack",
            ml_m="stack",
            store_oof=True,
        )
    with pytest.raises(NotImplementedError, match="model='irm'"):
        sp.dml(df, y="y", treat="d", covariates=COV, store_oof=True)


def test_rosters_are_refused_outside_plr_and_when_malformed():
    df = _data(n=120)
    df["t"] = (df["d"] > 0).astype(int)
    with pytest.raises(MethodIncompatibility, match="short stacking"):
        sp.dml(df, y="y", treat="t", covariates=COV, model="irm", ml_g="stack")
    with pytest.raises(MethodIncompatibility, match="short stacking"):
        sp.dml(
            df,
            y="y",
            treat="d",
            covariates=COV,
            model="pliv",
            instrument="x4",
            ml_r=[LinearRegression()],
        )
    with pytest.raises(MethodIncompatibility, match="empty list"):
        sp.dml(df, y="y", treat="d", covariates=COV, ml_g=[], ml_m="ols")
    with pytest.raises(MethodIncompatibility, match="neither"):
        sp.dml(df, y="y", treat="d", covariates=COV, ml_g=[3.0], ml_m="ols")
    with pytest.raises(MethodIncompatibility, match="another roster"):
        sp.dml(df, y="y", treat="d", covariates=COV, ml_g=["stack"], ml_m="ols")


def test_one_stacked_nuisance_and_duplicate_labels():
    df = _data()
    r = sp.dml(
        df,
        y="y",
        treat="d",
        covariates=COV,
        ml_g=[LinearRegression(), LinearRegression(fit_intercept=False)],
        ml_m="ols",
        n_folds=3,
    )
    assert set(r.model_info["stacking"]) == {"rule", "ml_g"}
    assert r.model_info["stacking"]["ml_g"]["candidates"] == [
        "LinearRegression",
        "LinearRegression_2",
    ]
    ref = sp.dml(
        df, y="y", treat="d", covariates=COV, ml_g="ols", ml_m="ols", n_folds=3
    )
    # same partition and same ml_m, so the treatment residuals are shared
    assert np.allclose(r.model_info["_d_resid"], ref.model_info["_d_resid"])


def test_announcement_is_silent_for_a_roster_and_names_it():
    df = _data(n=120)
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        sp.dml(df, y="y", treat="d", covariates=COV, ml_g=["ols"], ml_m=["ols"])
    with pytest.warns(FutureWarning, match="ml_g='stack', ml_m='stack'"):
        sp.dml(df, y="y", treat="d", covariates=COV, n_folds=2)
