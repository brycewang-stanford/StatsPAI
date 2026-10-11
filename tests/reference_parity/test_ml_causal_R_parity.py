"""Reference parity: grf post-fit operators and CDDF BLP / GATES, forest held fixed.

Fixture: ``_fixtures/ml_causal_R.json`` written by
``tests/reference_parity/_generate_ml_causal_R.R`` from
``_fixtures/ml_causal_cate.csv`` (``_fixtures/_generate_ml_causal_data.py``).
R 4.5.2, grf 2.6.1, GenericML 0.2.3, sandwich 3.1.1 (versions recorded in
the fixture's ``meta``).

What is pinned, and why it can be exact
---------------------------------------
A causal forest factorises into the forest (stochastic, not pinnable
across implementations) and the operators applied to its outputs. The R
script grows ONE ``grf::causal_forest`` (2000 trees, seed 42) and writes
its OOB ``tau.hat``, ``Y.hat`` and ``W.hat``. The tests below hand those
exact vectors to StatsPAI -- through the public functions, via a frozen
forest object whose ``effect()`` returns grf's predictions -- and compare
every operator:

* ``sp.average_treatment_effect`` all / treated / control / overlap vs
  ``grf::average_treatment_effect`` (overlap: R-learner OLS with an
  intercept, HC3 SE). ``clip=0``: grf does not clip; on this draw
  ``W.hat`` lies in (0.25, 0.72) so StatsPAI's default ``clip=0.01`` is
  inert and is asserted to be.
* ``sp.calibration_test`` vs ``grf::test_calibration`` (HC3 default and
  HC1): coefficients, SEs, t statistics and grf's one-sided p-values
  (Student t, n - 2 df).
* ``sp.rate`` AUTOC / QINI point estimates and the TOC curve on
  q = 0.1..1 vs ``grf::rank_average_treatment_effect`` with priorities =
  tau.hat, and ``rate_from_scores`` on a priority vector with 30 tie
  groups vs ``grf::rank_average_treatment_effect.fit`` (grf averages the
  scores within tied priorities).
* ``sp.blp_test`` / ``sp.gate_test`` vs ``GenericML::BLP`` /
  ``GenericML::GATES`` (``monotonize = FALSE``) with the same Y, D,
  propensity, BCA proxy and CATE proxy: weighted least squares with
  weights 1/(p(1-p)), homoskedastic (GenericML's default ``vcovHC(type =
  "const")``) and HC1 covariances, normal-theory p-values.

Tolerances: 1e-10 relative on every deterministic number. They are
closed-form least-squares / sorting operators on identical doubles; the
observed agreement is 1e-13 or better.

Not pinned: grf's RATE standard error is a half-sample bootstrap. It is
compared against StatsPAI's analytic (rank-corrected) influence-function
SE at the Monte Carlo error of grf's R = 2000 bootstrap (relative MC
error of an SD from R draws ~ 1/sqrt(2R) = 1.6%; tolerance 6%, i.e.
well under 4 MC standard errors).
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.forest.forest_inference import grf_calibration, rate_from_scores

_HERE = pathlib.Path(__file__).parent
_FIXTURE = _HERE / "_fixtures" / "ml_causal_R.json"
_DATA = _HERE / "_fixtures" / "ml_causal_cate.csv"

pytestmark = pytest.mark.skipif(
    not _FIXTURE.exists(), reason="ml_causal_R.json fixture is not materialized"
)

REL = 1e-10


@pytest.fixture(scope="module")
def ref():
    return json.loads(_FIXTURE.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def data():
    return pd.read_csv(_DATA)


class _FrozenForest:
    """A fitted-forest stand-in carrying grf's own outputs.

    It replaces only the stochastic factor (tree growing); every operator
    under test runs its real code path on these arrays.
    """

    fitted_ = True
    honest = True

    def __init__(self, X, Y, W, tau, y_hat, w_hat):
        self._X_original = np.asarray(X, dtype=float)
        self._Y_original = np.asarray(Y, dtype=float)
        self._T_original = np.asarray(W, dtype=float)
        self._m_insample = np.asarray(y_hat, dtype=float)
        self._e_insample = np.asarray(w_hat, dtype=float)
        self._tau = np.asarray(tau, dtype=float)

    def effect(self, X):
        X = np.asarray(X, dtype=float)
        assert np.array_equal(X, self._X_original)
        return self._tau.copy()


@pytest.fixture(scope="module")
def forest(ref, data):
    X = data[[f"x{j}" for j in range(1, 6)]].to_numpy()
    return _FrozenForest(
        X, data["y"], data["w"], ref["tau_hat"], ref["y_hat"], ref["w_hat"]
    )


def _rel(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-300))


# ---------------------------------------------------------------- ATE family
@pytest.mark.parametrize("target", ["all", "treated", "control", "overlap"])
def test_average_treatment_effect_matches_grf(ref, forest, target):
    out = sp.average_treatment_effect(forest, target_sample=target, clip=0.0)
    assert out["estimate"] == pytest.approx(ref["ate"][target]["estimate"], rel=REL)
    assert out["se"] == pytest.approx(ref["ate"][target]["se"], rel=REL)


@pytest.mark.parametrize("target", ["all", "treated", "control", "overlap"])
def test_default_clip_is_inert_on_this_fixture(forest, target):
    a = sp.average_treatment_effect(forest, target_sample=target)
    b = sp.average_treatment_effect(forest, target_sample=target, clip=0.0)
    assert a["estimate"] == b["estimate"] and a["se"] == b["se"]


def test_overlap_is_r_learner_regression_identity(ref, forest, data):
    """Known-truth identity: the ATO slope is cov(Wres, Yres) / var(Wres)."""
    yr = data["y"].to_numpy() - np.asarray(ref["y_hat"])
    wr = data["w"].to_numpy() - np.asarray(ref["w_hat"])
    slope = np.cov(wr, yr, ddof=1)[0, 1] / np.var(wr, ddof=1)
    out = sp.average_treatment_effect(forest, target_sample="overlap")
    assert out["estimate"] == pytest.approx(slope, rel=1e-12)
    assert out["method"] == "r_learner_hc3"


def test_out_of_sample_rows_fall_back_loudly(forest):
    X = forest._X_original[:50] + 0.0
    X[0, 0] += 1.0
    forest_new = _FrozenForest(
        forest._X_original,
        forest._Y_original,
        forest._T_original,
        forest._tau,
        forest._m_insample,
        forest._e_insample,
    )
    forest_new.effect = lambda Z: np.zeros(len(Z))
    with pytest.warns(Warning, match="plug-in"):
        out = sp.average_treatment_effect(
            forest_new, X=X, T=forest._T_original[:50], target_sample="all"
        )
    assert out["method"] == "plug_in"


# ------------------------------------------------------------- calibration
@pytest.mark.parametrize("vcov", ["HC3", "HC1"])
def test_calibration_test_matches_grf(ref, forest, vcov):
    r = ref["calibration"][vcov]
    out = sp.calibration_test(forest, vce=vcov)
    assert list(out.index) == [
        "mean_forest_prediction",
        "differential_forest_prediction",
    ]
    assert _rel(out["coef"], r["coef"]) < REL
    assert _rel(out["se"], r["se"]) < REL
    assert _rel(out["t"], r["t"]) < REL
    assert _rel(out["p"], r["p_one_sided"]) < 1e-8  # tiny p-values: t-tail evaluation
    alias = sp.test_calibration(forest, vce=vcov)
    assert alias.equals(out)


@pytest.mark.parametrize("vcov", ["HC3", "HC1"])
def test_grf_calibration_matches_grf(ref, data, vcov):
    """The public operator, handed grf's own vectors rather than a forest."""
    r = ref["calibration"][vcov]
    out = sp.grf_calibration(
        Y=data["y"],
        W=data["w"],
        Y_hat=ref["y_hat"],
        W_hat=ref["w_hat"],
        tau_hat=ref["tau_hat"],
        vce=vcov,
    )
    assert _rel(out["coef"], r["coef"]) < REL
    assert _rel(out["se"], r["se"]) < REL
    assert _rel(out["t"], r["t"]) < REL
    assert _rel(out["p"], r["p_one_sided"]) < 1e-8  # as above


def test_calibration_default_is_grf_hc3(ref, forest):
    out = sp.calibration_test(forest)
    assert _rel(out["se"], ref["calibration"]["HC3"]["se"]) < REL


def test_calibration_operator_recovers_known_slopes():
    """Identity: noiseless Y - Y_hat = (W - W_hat) * tau gives b1 = b2 = 1."""
    rng = np.random.default_rng(3)
    n = 500
    w_hat = rng.uniform(0.2, 0.8, n)
    W = rng.binomial(1, w_hat).astype(float)
    tau = 1.0 + rng.normal(size=n)
    Y_hat = rng.normal(size=n)
    Y = Y_hat + (W - w_hat) * tau
    out = grf_calibration(Y=Y, W=W, Y_hat=Y_hat, W_hat=w_hat, tau_hat=tau)
    np.testing.assert_allclose(out["coef"], [1.0, 1.0], rtol=0, atol=1e-12)


# -------------------------------------------------------------------- RATE
@pytest.mark.parametrize("target", ["autoc", "qini"])
def test_rate_point_and_toc_match_grf(ref, forest, target):
    r = ref["rate"][target]
    out = sp.rate(forest, target=target.upper(), q_grid=10)
    assert out["estimate"] == pytest.approx(r["estimate"], rel=REL)
    toc = out["toc_curve"]
    np.testing.assert_allclose(toc[:, 0], r["toc_q"], rtol=0, atol=1e-12)
    # The last TOC point is 0 up to roundoff on both sides (1e-15).
    np.testing.assert_allclose(toc[:-1, 1], r["toc_estimate"][:-1], rtol=REL, atol=0)
    assert abs(toc[-1, 1]) < 1e-12


@pytest.mark.parametrize("target", ["autoc", "qini"])
def test_rate_with_tied_priorities_matches_grf_fit(ref, target):
    r = ref["rate"][f"fit_{target}_ties"]
    scores = np.asarray(ref["dr_scores"])
    prio = np.round(np.asarray(ref["tau_hat"]), 1)
    assert len(np.unique(prio)) == ref["rate"]["n_distinct_ties"]
    out = rate_from_scores(scores, prio, target.upper(), np.asarray(r["toc_q"]))
    assert out["estimate"] == pytest.approx(r["estimate"], rel=REL)
    np.testing.assert_allclose(out["toc"][:-1], r["toc_estimate"][:-1], rtol=REL)


def test_rate_scores_are_grf_get_scores(ref, forest):
    """sp.rate scores units with grf's DR scores (grf::get_scores)."""
    psi = sp.forest.forest_inference.aipw_scores(
        tau=forest._tau,
        T=forest._T_original,
        e_hat=forest._e_insample,
        m_hat=forest._m_insample,
        Y=forest._Y_original,
        target="all",
    )
    np.testing.assert_allclose(psi, ref["dr_scores"], rtol=0, atol=1e-12)


@pytest.mark.parametrize("target", ["autoc", "qini"])
def test_rate_se_agrees_with_grf_half_sample_bootstrap(ref, forest, target):
    """T3: analytic rank-corrected IF SE vs grf's R = 2000 half-sample SD."""
    grf_se = ref["rate"][f"{target}_R2000"]["se"]
    out = sp.rate(forest, target=target.upper())
    assert out["se"] == pytest.approx(grf_se, rel=0.06)


def test_rate_half_sample_option_is_grf_procedure(ref, forest):
    out = sp.rate(
        forest, target="AUTOC", se_method="half_sample", n_bootstrap=2000, seed=7
    )
    assert out["se"] == pytest.approx(ref["rate"]["autoc_R2000"]["se"], rel=0.06)


def test_rate_flat_toc_gives_zero():
    """Identity: constant scores => TOC == 0 => AUTOC == QINI == 0."""
    scores = np.full(101, 2.5)
    prio = np.random.default_rng(0).normal(size=101)
    for tgt in ("AUTOC", "QINI"):
        assert abs(rate_from_scores(scores, prio, tgt)["estimate"]) < 1e-13


def test_rate_accepts_external_priorities(ref, forest):
    prio = np.asarray(ref["tau_hat"])
    a = sp.rate(forest, priorities=prio)
    b = sp.rate(forest)
    assert a["estimate"] == b["estimate"]
    c = sp.rate(forest, priorities=-prio)
    assert c["estimate"] < 0 < b["estimate"]


def test_cate_eval_matches_grf_given_grf_nuisances(ref, data):
    """sp.cate_eval shares the RATE operator; fed grf's nuisances it is grf."""
    tau = np.asarray(ref["tau_hat"])
    y_hat = np.asarray(ref["y_hat"])
    w_hat = np.asarray(ref["w_hat"])
    res = sp.cate_eval(
        tau,
        data["y"].to_numpy(),
        data["w"].to_numpy(),
        e_hat=w_hat,
        m_hat=y_hat,
        mu1_hat=y_hat + (1 - w_hat) * tau,
        mu0_hat=y_hat - w_hat * tau,
        q_grid=10,
    )
    assert res.autoc == pytest.approx(ref["rate"]["autoc"]["estimate"], rel=REL)
    assert res.qini == pytest.approx(ref["rate"]["qini"]["estimate"], rel=REL)
    np.testing.assert_allclose(
        res.toc_curve["toc"].to_numpy()[:-1],
        ref["rate"]["autoc"]["toc_estimate"][:-1],
        rtol=REL,
    )


# ----------------------------------------------------------- honest_variance
def test_honest_variance_se_is_sampling_se_of_the_mean(ref, forest):
    """Half-sample spread -> sd(tau)/sqrt(n); it must not shrink with splits."""
    tau = np.asarray(ref["tau_hat"])
    target = tau.std(ddof=1) / np.sqrt(len(tau))
    se = sp.honest_variance(forest, n_splits=4000, seed=1)["se"]
    assert se == pytest.approx(target, rel=0.05)
    se_small = sp.honest_variance(forest, n_splits=50, seed=1)["se"]
    assert se_small == pytest.approx(target, rel=0.3)


def test_forest_diagnostics_identities(ref, forest):
    d = sp.forest_diagnostics(forest)
    tau = np.asarray(ref["tau_hat"])
    w_hat = np.asarray(ref["w_hat"])
    assert d["n"] == len(tau)
    assert d["cate_mean"] == pytest.approx(tau.mean(), rel=1e-14)
    assert d["cate_sd"] == pytest.approx(tau.std(ddof=1), rel=1e-14)
    assert d["pscore_min"] == w_hat.min() and d["pscore_max"] == w_hat.max()
    assert d["overlap_share"] == pytest.approx(
        np.mean((w_hat >= 0.05) & (w_hat <= 0.95)), abs=0
    )


# ------------------------------------------------- CDDF BLP / GATES (GenericML)
def _cddf_inputs(ref):
    tau = np.asarray(ref["tau_hat"])
    y_hat = np.asarray(ref["y_hat"])
    w_hat = np.asarray(ref["w_hat"])
    return tau, w_hat, y_hat - w_hat * tau  # proxy S, propensity p, BCA B


@pytest.mark.parametrize(
    "key,use_b,vcov",
    [
        ("blp_B_const", True, "const"),
        ("blp_B_hc1", True, "HC1"),
        ("blp_none_hc1", False, "HC1"),
    ],
)
def test_blp_test_matches_genericml(ref, data, key, use_b, vcov):
    S, p, B = _cddf_inputs(ref)
    r = ref["generic_ml"][key]
    out = sp.blp_test(
        S,
        data,
        y="y",
        treat="w",
        covariates=[f"x{j}" for j in range(1, 6)],
        propensity=p,
        baseline=B if use_b else None,
        vce=vcov,
    )
    assert out["proxy_source"] == "supplied"
    assert _rel([out["beta1"], out["beta2"]], r["beta"]) < REL
    assert _rel([out["beta1_se"], out["beta2_se"]], r["se"]) < REL
    assert (
        _rel([out["beta1_pvalue_right"], out["beta2_pvalue_right"]], r["p_right"])
        < 1e-8
    )
    assert _rel([out["beta1_ci"][0], out["beta2_ci"][0]], r["ci_lo"]) < REL


@pytest.mark.parametrize(
    "key,vcov", [("gates_B_const", "const"), ("gates_B_hc1", "HC1")]
)
def test_gate_test_matches_genericml(ref, data, key, vcov):
    S, p, B = _cddf_inputs(ref)
    r = ref["generic_ml"][key]
    out = sp.gate_test(
        S,
        data,
        by="cate",
        n_groups=4,
        y="y",
        treat="w",
        covariates=[f"x{j}" for j in range(1, 6)],
        propensity=p,
        baseline=B,
        vce=vcov,
    )
    tab = out["gate_table"]
    assert _rel(tab["gate"], r["estimate"][:4]) < REL
    assert _rel(tab["se"], r["se"][:4]) < REL
    assert out["top_vs_bottom_diff"] == pytest.approx(r["estimate"][4], rel=REL)
    assert out["top_vs_bottom_se"] == pytest.approx(r["se"][4], rel=REL)


def test_gate_groups_are_genericml_quantile_groups(ref):
    from statspai.metalearners._cddf import quantile_groups

    S = np.asarray(ref["tau_hat"])
    labels = quantile_groups(S, 4) + 1
    np.testing.assert_array_equal(labels, ref["generic_ml"]["membership"])


def test_blp_known_truth_on_noiseless_design():
    """Identity: Y = (D - p) * S exactly => beta1 = mean(S), beta2 = 1."""
    rng = np.random.default_rng(5)
    n = 400
    p = rng.uniform(0.3, 0.7, n)
    D = rng.binomial(1, p)
    S = 1.0 + rng.normal(size=n)
    df = pd.DataFrame({"y": 2.0 + (D - p) * S, "d": D, "x": rng.normal(size=n)})
    out = sp.blp_test(S, df, y="y", treat="d", covariates=["x"], propensity=p)
    assert out["beta1"] == pytest.approx(S.mean(), abs=1e-10)
    assert out["beta2"] == pytest.approx(1.0, abs=1e-10)
