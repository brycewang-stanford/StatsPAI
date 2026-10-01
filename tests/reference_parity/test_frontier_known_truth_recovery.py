"""Known-truth recovery for frontier estimators with no reference package.

Each test plants a parameter in a simulated design, runs one StatsPAI
estimator, and requires the estimate to land within four standard errors
of the planted value (the convention in REFERENCES.md). Where an estimator
reports no standard error, the band is four Monte Carlo standard deviations
measured over twelve or more seeds when the test was written; the measured
value is quoted next to each such tolerance.

These are T1 anchors: they show the estimator targets the right quantity on
a design where its assumptions hold. They are not comparisons with R or
Stata, and they say nothing about designs outside the one simulated.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp

pytestmark = pytest.mark.filterwarnings("ignore")


def _within(estimate: float, truth: float, se: float, n_sigma: float = 4.0) -> bool:
    return bool(np.isfinite(se) and se > 0 and abs(estimate - truth) <= n_sigma * se)


# Parametrised tests dispatch through these tables so that every estimator
# is still called by its public name (the parity index credits a function
# when it sees ``sp.<name>(`` in this directory).
_G_ESTIMATORS = {
    "a_learning": lambda df, **kw: sp.a_learning(df, **kw),
    "snmm": lambda df, **kw: sp.snmm(df, **kw),
}
_PROXY_ESTIMATORS = {
    "opreg": lambda df, **kw: sp.opreg(df, **kw),
    "levpet": lambda df, **kw: sp.levpet(df, **kw),
    "acf": lambda df, **kw: sp.acf(df, **kw),
}
_CONFORMAL_ITE = {
    "conformal_debiased_ml": lambda df, **kw: sp.conformal_debiased_ml(df, **kw),
    "conformal_density_ite": lambda df, **kw: sp.conformal_density_ite(df, **kw),
}


# --------------------------------------------------------------------- #
#  Staggered difference-in-differences
# --------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def staggered_panel() -> pd.DataFrame:
    """Two cohorts (t = 4, 6) and a never-treated third; ATT = 2.0."""
    rng = np.random.default_rng(0)
    rows = []
    for i in range(300):
        g = [0, 4, 6][i % 3]
        a = rng.normal()
        for t in range(1, 9):
            d = 1.0 if (g and t >= g) else 0.0
            rows.append((i, t, g, a + 0.3 * t + 2.0 * d + rng.normal(0, 0.5)))
    return pd.DataFrame(rows, columns=["id", "t", "g", "y"])


def test_did_misclassified_without_correction_recovers_att(staggered_panel):
    res = sp.did_misclassified(staggered_panel, y="y", treat="g", time="t", id="id")
    assert _within(res.estimate, 2.0, res.se)
    # With no misclassification and no anticipation nothing is adjusted.
    assert res.model_info["misclass_factor"] == pytest.approx(1.0)
    assert res.model_info["naive_att"] == pytest.approx(res.estimate, abs=1e-12)


def test_cohort_anchored_event_study_recovers_att(staggered_panel):
    res = sp.cohort_anchored_event_study(
        staggered_panel, y="y", treat="g", time="t", id="id", leads=2, lags=2
    )
    assert _within(res.estimate, 2.0, res.se)
    es = res.model_info["event_study"].set_index("rel_time")
    for k in (0, 1, 2):
        assert _within(es.loc[k, "att"], 2.0, es.loc[k, "se"])
    assert _within(es.loc[-2, "att"], 0.0, es.loc[-2, "se"])


def test_overlap_weighted_did_recovers_a_constant_effect():
    rng = np.random.default_rng(0)
    n = 2000
    x = rng.normal(0, 1, n)
    treat = rng.binomial(1, 1 / (1 + np.exp(-x)))
    base = 1.0 + 0.5 * x + rng.normal(0, 1, n)
    post = base + 0.4 + 1.5 * treat + rng.normal(0, 1, n)
    df = pd.DataFrame(
        {
            "y": np.concatenate([base, post]),
            "treat": np.tile(treat, 2),
            "time": np.repeat([0, 1], n),
            "x": np.tile(x, 2),
        }
    )
    res = sp.overlap_weighted_did(
        df, y="y", treat="treat", time="time", covariates=["x"]
    )
    # A constant effect is the overlap-weighted ATT whatever the weights.
    # The reported SE treats the two periods as independent samples, so on
    # a panel it is conservative (about 1.9x the Monte Carlo SD here).
    assert _within(res.estimate, 1.5, res.se)


def test_cs_jackknife_recovers_the_aggregate_att():
    rng = np.random.default_rng(0)
    rows = []
    for u in range(90):
        g = [0, 3, 4][u % 3]
        a = rng.normal()
        for tt in range(1, 6):
            d = 1.0 if g and tt >= g else 0.0
            rows.append((u, tt, g, a + 0.2 * tt + 0.5 * d + rng.normal(0, 0.3)))
    df = pd.DataFrame(rows, columns=["unit", "time", "g", "y"])
    res = sp.cs_jackknife(df, y="y", g="g", time="time", id="unit", estimator="reg")
    assert _within(res.estimate, 0.5, res.se)
    assert res.model_info["n_replicates"] == 90


def test_dnc_gnn_did_recovers_att():
    rng = np.random.default_rng(0)
    rows = []
    for uid in range(200):
        treat_time = 3 if uid < 100 else 0
        u = rng.normal()
        for t in range(1, 6):
            post = 1 if (treat_time > 0 and t >= treat_time) else 0
            rows.append(
                {
                    "id": uid,
                    "time": t,
                    "treat": treat_time,
                    "y": u + 0.5 * t + 1.2 * post + rng.normal(0, 0.3),
                    "nc_y": u + 0.4 * t + rng.normal(0, 0.3),
                    "nc_d": u + rng.normal(0, 0.3),
                }
            )
    res = sp.dnc_gnn_did(
        pd.DataFrame(rows),
        y="y",
        treat="treat",
        time="time",
        id="id",
        nc_outcome=["nc_y"],
        nc_exposure=["nc_d"],
        n_boot=100,
        seed=1,
    )
    assert _within(res.estimate, 1.2, res.se)


# --------------------------------------------------------------------- #
#  Proxies and negative controls
# --------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def confounded_frame() -> pd.DataFrame:
    """Y = 1.5 D + U + e with U unobserved and four noisy proxies of U."""
    rng = np.random.default_rng(0)
    n = 4000
    U = rng.normal(size=n)
    D = (0.5 * U + rng.normal(size=n) > 0).astype(int)
    return pd.DataFrame(
        {
            "Y": 1.5 * D + U + rng.normal(size=n),
            "D": D,
            "X": rng.normal(size=n),
            "NCE": 0.8 * U + 0.5 * rng.normal(size=n),
            "NCO": 0.8 * U + 0.5 * rng.normal(size=n),
            "Z": 0.7 * U + 0.5 * rng.normal(size=n),
            "W": 0.7 * U + 0.5 * rng.normal(size=n),
        }
    )


def test_double_negative_control_removes_the_confounding(confounded_frame):
    df = confounded_frame
    res = sp.double_negative_control(df, y="Y", treat="D", nce="NCE", nco="NCO")
    assert _within(res.estimate, 1.5, res.se)
    # The unadjusted contrast is far from the truth, so the anchor has power.
    naive = df.loc[df.D == 1, "Y"].mean() - df.loc[df.D == 0, "Y"].mean()
    assert naive - 1.5 > 10 * res.se


def test_proximal_regression_removes_the_confounding(confounded_frame):
    res = sp.proximal_regression(
        confounded_frame, y="Y", treat="D", z_proxy="Z", w_proxy="W", covariates=["X"]
    )
    assert _within(res.ate, 1.5, res.se)


def test_zero_first_stage_recovers_the_structural_coefficient():
    rng = np.random.default_rng(0)
    n = 3000
    desert = rng.integers(0, 2, size=n).astype(bool)
    z = rng.normal(size=n)
    u = rng.normal(size=n)
    d = np.where(desert, 0.0, 0.9 * z) + 0.6 * u + rng.normal(size=n)
    y = -0.5 * d + 0.7 * u + rng.normal(size=n)
    res = sp.zero_first_stage(
        pd.DataFrame({"y": y, "d": d, "z": z, "desert": desert}),
        y="y",
        endog="d",
        instrument="z",
        zfs="desert",
        n_boot=0,
    )
    assert _within(res.beta_iv, -0.5, float(res.beta_iv_se))
    # The instrument has no first stage and no reduced form in the
    # zero-first-stage sample, as the DGP dictates.
    assert _within(res.first_stage_zfs, 0.0, res.first_stage_zfs_se)
    assert _within(res.reduced_form_zfs, 0.0, res.reduced_form_zfs_se)
    assert _within(res.first_stage_main, 0.9, res.first_stage_main_se)


# --------------------------------------------------------------------- #
#  Interference
# --------------------------------------------------------------------- #


def test_inward_outward_spillover_recovers_both_channels():
    rng = np.random.default_rng(0)
    n = 3000
    d = (rng.uniform(size=n) < 0.5).astype(float)
    e_in = rng.uniform(size=n)
    e_out = rng.uniform(size=n)
    y = 1.0 + 0.6 * d + 0.4 * e_in + 0.2 * e_out + rng.normal(scale=0.5, size=n)
    res = sp.inward_outward_spillover(
        pd.DataFrame({"y": y, "d": d, "e_in": e_in, "e_out": e_out}),
        y="y",
        treatment="d",
        inward_exposure="e_in",
        outward_exposure="e_out",
    )
    assert _within(res.inward_effect, 0.4, res.inward_se)
    assert _within(res.outward_effect, 0.2, res.outward_se)


def test_network_hte_recovers_direct_and_spillover_effects():
    rng = np.random.default_rng(0)
    n = 3000
    d = (rng.uniform(size=n) < 0.5).astype(float)
    e = rng.uniform(size=n)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    y = 1.0 + 0.8 * d + 0.5 * e + 0.3 * x1 + rng.normal(scale=0.5, size=n)
    res = sp.network_hte(
        pd.DataFrame({"y": y, "d": d, "e": e, "x1": x1, "x2": x2}),
        y="y",
        treatment="d",
        neighbor_exposure="e",
        covariates=["x1", "x2"],
        n_folds=3,
        random_state=0,
    )
    assert _within(res.direct_effect, 0.8, res.direct_se)
    assert _within(res.spillover_effect, 0.5, res.spillover_se)


# --------------------------------------------------------------------- #
#  Regression discontinuity variants
# --------------------------------------------------------------------- #


def test_rd_lasso_recovers_the_jump_with_irrelevant_covariates():
    rng = np.random.default_rng(0)
    n = 1500
    x = rng.uniform(-1, 1, n)
    Z = rng.normal(size=(n, 8))
    y = 1.0 * (x >= 0) + 0.5 * x + 0.7 * Z[:, 0] + rng.normal(0, 0.5, n)
    df = pd.DataFrame({"y": y, "x": x})
    covs = [f"z{i}" for i in range(8)]
    for i, name in enumerate(covs):
        df[name] = Z[:, i]
    res = sp.rd_lasso(df, y="y", x="x", c=0, covs=covs)
    assert _within(res.estimate, 1.0, res.se)
    assert "z0" in res.model_info["selected_covariates"]


def test_rd_flex_recovers_the_jump():
    rng = np.random.default_rng(0)
    n = 1500
    margin = rng.uniform(-1, 1, n)
    w1 = 0.5 + 0.3 * margin + rng.normal(0, 0.1, n)
    w2 = 0.4 + 0.2 * margin + rng.normal(0, 0.1, n)
    vote = 0.3 * margin + 0.25 * (margin >= 0) + 0.4 * w1 + rng.normal(0, 0.1, n)
    res = sp.rd_flex(
        pd.DataFrame({"vote": vote, "margin": margin, "w1": w1, "w2": w2}),
        y="vote",
        x="margin",
        c=0.0,
        W=["w1", "w2"],
        learner="ridge",
        n_folds=5,
        random_state=0,
    )
    assert _within(res.estimate, 0.25, res.se)


def test_rdit_recovers_a_level_shift():
    rng = np.random.default_rng(0)
    dates = pd.date_range("2010-01-01", "2019-12-01", freq="MS")
    t = np.arange(len(dates))
    post = (dates >= pd.Timestamp("2015-01-01")).astype(float)
    y = 100 + 0.3 * t + 8.0 * post + rng.normal(0, 2.0, len(dates))
    res = sp.rdit(
        pd.DataFrame({"date": dates, "y": y}), y="y", time="date", cutoff="2015-01-01"
    )
    assert _within(res.estimate, 8.0, res.se)


def test_kink_unified_separates_a_jump_from_a_kink():
    rng = np.random.default_rng(0)
    n = 3000
    x = rng.uniform(-1.0, 1.0, n)
    y = 0.5 * x + 0.8 * (x >= 0) + rng.normal(0, 0.3, n)
    res = sp.kink_unified(
        pd.DataFrame({"earnings": y, "income": x}),
        y="earnings",
        running="income",
        cutoff=0.0,
    )
    # A level jump of 0.8 and no change in slope.
    assert _within(res.rdd_effect, 0.8, res.rdd_se)
    assert _within(res.rkd_effect, 0.0, res.rkd_se)


# --------------------------------------------------------------------- #
#  Prediction-powered inference
# --------------------------------------------------------------------- #


def test_ppi_mean_recovers_the_mean_and_beats_the_labeled_sample():
    rng = np.random.default_rng(7)
    n, N = 100, 5000
    y = 2.0 + rng.normal(0, 1, n + N)
    f = y + rng.normal(0, 0.5, n + N)
    res = sp.ppi_mean(y=y[:n], yhat=f[:n], yhat_unlabeled=f[n:])
    assert _within(res.estimate, 2.0, res.se)
    assert res.se < y[:n].std(ddof=1) / np.sqrt(n)


def test_ppi_ols_recovers_the_slope():
    rng = np.random.default_rng(3)
    n, N = 150, 4000
    x = rng.normal(size=n + N)
    y = 1.0 + 2.0 * x + rng.normal(0, 1, n + N)
    f = y + rng.normal(0, 0.5, n + N)
    res = sp.ppi_ols(
        X=pd.DataFrame({"x": x[:n]}),
        y=y[:n],
        yhat=f[:n],
        X_unlabeled=pd.DataFrame({"x": x[n:]}),
        yhat_unlabeled=f[n:],
    )
    assert _within(res.estimate, 2.0, res.se)
    const = res.detail.set_index("term").loc["const"]
    assert abs(float(const.iloc[0]) - 1.0) < 0.5


# --------------------------------------------------------------------- #
#  Dynamic treatment regimes
# --------------------------------------------------------------------- #


def _two_stage(seed: int, n: int = 20_000) -> pd.DataFrame:
    """Blips: stage 1 = 1 + x1, stage 2 = 0.5 - x2. Both actions randomised."""
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    a1 = rng.integers(0, 2, n)
    x2 = x1 + rng.normal(size=n)
    a2 = rng.integers(0, 2, n)
    y = x1 + a1 * (1 + x1) + a2 * (0.5 - x2) + rng.normal(size=n)
    return pd.DataFrame({"x1": x1, "a1": a1, "x2": x2, "a2": a2, "y": y})


#: E[x1 + (1 + x1)+ + (0.5 - x2)+] with x1 ~ N(0, 1), x2 ~ N(0, 2):
#: 0 + [phi(1) + Phi(1)] + [sqrt(2) phi(0.5 / sqrt(2)) + 0.5 Phi(0.5 / sqrt(2))].
_OPTIMAL_VALUE = 1.9324


@pytest.mark.parametrize(
    "fn_name, attr", [("a_learning", "psi"), ("snmm", "blip_params")]
)
def test_g_estimation_recovers_both_blip_functions(fn_name, attr):
    res = _G_ESTIMATORS[fn_name](
        _two_stage(seed=0),
        y="y",
        actions=["a1", "a2"],
        stage_covariates=[["x1"], ["x2"]],
    )
    stage1, stage2 = (np.asarray(p, dtype=float) for p in getattr(res, attr))
    # Bands are 4x the across-seed SD at n = 20,000 (stage 1: 0.03, 0.02;
    # stage 2: 0.025, 0.05, 0.054, 0.038).
    np.testing.assert_allclose(stage1, [1.0, 1.0], atol=0.12)
    np.testing.assert_allclose(stage2, [0.5, 0.0, 0.0, -1.0], atol=0.22)


def test_a_learning_value_is_the_optimal_regime_value():
    res = sp.a_learning(
        _two_stage(seed=0),
        y="y",
        actions=["a1", "a2"],
        stage_covariates=[["x1"], ["x2"]],
    )
    # Across-seed SD of the value at n = 20,000 is 0.013.
    assert res.value == pytest.approx(_OPTIMAL_VALUE, abs=0.052)


def test_q_learning_single_stage_rule_and_value():
    rng = np.random.default_rng(0)
    n = 20_000
    x = rng.normal(size=n)
    a = rng.integers(0, 2, n)
    y = x + a * (1 + x) + rng.normal(size=n)
    res = sp.q_learning(
        pd.DataFrame({"x": x, "a": a, "y": y}),
        y="y",
        actions=["a"],
        stage_covariates=[["x"]],
    )
    # Optimal rule: treat when 1 + x > 0. P = Phi(1) = 0.8413 and the value
    # is E[x + (1 + x)+] = phi(1) + Phi(1) = 1.0833. Across-seed SD of the
    # value is 0.018.
    assert res.value == pytest.approx(1.0833, abs=0.072)
    assert res.stage_coefs[0]["fraction_treat_optimal"] == pytest.approx(
        0.8413, abs=0.02
    )
    # The learned rule agrees with the optimal one except near 1 + x = 0.
    assert np.mean(res.optimal_actions[:, 0] == (1 + x > 0)) > 0.98


def test_q_learning_two_stage_value_under_a_correct_working_model():
    rng = np.random.default_rng(0)
    n = 20_000
    x1 = rng.normal(size=n)
    a1 = rng.integers(0, 2, n)
    x2 = x1 + rng.normal(size=n)
    a2 = rng.integers(0, 2, n)
    # Stage-1 effect constant, so the linear working model for Q2 holds.
    y = x1 + 0.7 * a1 + a2 * (0.5 - x2) + rng.normal(size=n)
    res = sp.q_learning(
        pd.DataFrame({"x1": x1, "a1": a1, "x2": x2, "a2": a2, "y": y}),
        y="y",
        actions=["a1", "a2"],
        stage_covariates=[["x1"], ["x2"]],
    )
    # Value = 0.7 + E[(0.5 - x2)+] = 0.7 + 0.8491; across-seed SD 0.011.
    assert res.value == pytest.approx(1.5491, abs=0.044)
    fractions = [c["fraction_treat_optimal"] for c in res.stage_coefs]
    assert fractions[0] == pytest.approx(1.0, abs=1e-12)
    assert fractions[1] == pytest.approx(0.638, abs=0.02)  # P(x2 < 0.5)


# --------------------------------------------------------------------- #
#  Production functions
# --------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def firm_panel() -> pd.DataFrame:
    """Cobb-Douglas with beta_l = 0.6, beta_k = 0.3.

    Both proxies are exact, strictly increasing functions of productivity
    given capital (the scalar-unobservable condition of Olley-Pakes and
    Levinsohn-Petrin), labour carries its own shock, and capital is chosen
    a period ahead.
    """
    rng = np.random.default_rng(0)
    rows = []
    for fid in range(600):
        k = rng.normal(2.0, 0.5)
        omega = rng.normal(0, 0.3)
        for yr in range(10):
            omega = 0.7 * omega + rng.normal(0, 0.2)
            labour = 0.3 * k + 0.5 * omega + rng.normal(1.0, 0.3)
            materials = 0.5 * k + omega
            investment = 0.4 * k + omega + 2.0
            y = 0.6 * labour + 0.3 * k + omega + rng.normal(0, 0.1)
            rows.append(
                dict(id=fid, year=yr, y=y, l=labour, k=k, m=materials, i=investment)
            )
            k = 0.8 * k + 0.1 * investment + rng.normal(0, 0.1)
    return pd.DataFrame(rows)


@pytest.mark.parametrize(
    "fn_name, proxy, tol_l, tol_k",
    [
        ("opreg", "i", 0.02, 0.03),
        ("levpet", "m", 0.02, 0.03),
        # ACF moves the labour coefficient into the second-stage GMM and is
        # noisier there.
        ("acf", "m", 0.03, 0.05),
    ],
)
def test_proxy_variable_estimators_recover_the_elasticities(
    firm_panel, fn_name, proxy, tol_l, tol_k
):
    res = _PROXY_ESTIMATORS[fn_name](
        firm_panel, output="y", free="l", state="k", proxy=proxy
    )
    assert res.coef["l"] == pytest.approx(0.6, abs=tol_l)
    assert res.coef["k"] == pytest.approx(0.3, abs=tol_k)
    # OLS on the same panel is biased by the transmission of productivity
    # to labour, so the anchor is not one that any regression would pass.
    X = np.column_stack([np.ones(len(firm_panel)), firm_panel["l"], firm_panel["k"]])
    ols = np.linalg.lstsq(X, firm_panel["y"].to_numpy(), rcond=None)[0]
    assert ols[1] - 0.6 > 0.1


# --------------------------------------------------------------------- #
#  Conformal intervals: marginal coverage on fresh draws
# --------------------------------------------------------------------- #


def test_conformal_counterfactual_covers_both_potential_outcomes():
    cover = []
    for s in range(12):
        rng = np.random.default_rng(s)
        n = 1000
        x1, x2 = rng.normal(size=n), rng.normal(size=n)
        t = rng.binomial(1, 0.5, size=n)
        y1 = 3.0 + 0.5 * x1 + rng.normal(scale=0.5, size=n)
        y0 = 1.0 + 0.5 * x1 + rng.normal(scale=0.5, size=n)
        df = pd.DataFrame({"y": np.where(t == 1, y1, y0), "t": t, "x1": x1, "x2": x2})
        res = sp.conformal_counterfactual(
            df.iloc[:600],
            y="y",
            treat="t",
            covariates=["x1", "x2"],
            X_test=df.iloc[600:][["x1", "x2"]].to_numpy(),
            alpha=0.1,
            random_state=s,
        )
        fr = res.to_frame()
        cover.append(
            (
                np.mean(
                    (fr["Y1_lower"].values <= y1[600:])
                    & (y1[600:] <= fr["Y1_upper"].values)
                ),
                np.mean(
                    (fr["Y0_lower"].values <= y0[600:])
                    & (y0[600:] <= fr["Y0_upper"].values)
                ),
            )
        )
    # 4,800 test points per arm: the MC SE of the coverage rate is 0.005.
    c1, c0 = np.mean(cover, axis=0)
    assert c1 >= 0.87 and c0 >= 0.87
    assert c1 <= 0.95 and c0 <= 0.95  # not vacuous


def test_conformal_continuous_covers_new_outcomes():
    cover = []
    for s in range(12):
        rng = np.random.default_rng(s)
        n = 1200
        t = rng.uniform(0, 5, n)
        x = rng.normal(size=n)
        y = 2.0 + 0.7 * t + 0.5 * x + rng.normal(0, 0.5, n)
        df = pd.DataFrame({"y": y, "t": t, "x": x})
        res = sp.conformal_continuous(
            df.iloc[:800],
            y="y",
            treatment="t",
            covariates=["x"],
            test_data=df.iloc[800:][["t", "x"]],
            alpha=0.1,
            random_state=s,
        )
        pred = res.predictions
        cover.append(
            np.mean((pred["lo"].values <= y[800:]) & (y[800:] <= pred["hi"].values))
        )
    assert 0.87 <= np.mean(cover) <= 0.95


@pytest.mark.parametrize("fn_name", ["conformal_debiased_ml", "conformal_density_ite"])
def test_conformal_ite_intervals_cover_the_true_effects(fn_name):
    cover = []
    for s in range(8):
        rng = np.random.default_rng(s)
        n = 1500
        x1 = rng.normal(size=n)
        x2 = rng.normal(size=n)
        d = rng.binomial(1, 1 / (1 + np.exp(-0.5 * x1)))
        y1 = 2.5 + 0.8 * x1 + 0.5 * x2 + rng.normal(0, 0.5, n)
        y0 = 1.0 + 0.8 * x1 + 0.5 * x2 + rng.normal(0, 0.5, n)
        df = pd.DataFrame({"y": np.where(d == 1, y1, y0), "d": d, "x1": x1, "x2": x2})
        res = _CONFORMAL_ITE[fn_name](
            df.iloc[:1000],
            y="y",
            treat="d",
            covariates=["x1", "x2"],
            test_data=df.iloc[1000:],
            alpha=0.1,
            seed=s,
        )
        ite = (y1 - y0)[1000:]
        iv = res.intervals
        cover.append(np.mean((iv[:, 0] <= ite) & (ite <= iv[:, 1])))
    # The guarantee is "at least 1 - alpha"; the debiased-ML band is
    # conservative (0.98 here), the density band close to nominal (0.91).
    assert np.mean(cover) >= 0.87


# --------------------------------------------------------------------- #
#  Causal discovery
# --------------------------------------------------------------------- #


def test_icp_returns_the_true_parent_and_never_the_child():
    exact, child_included = 0, 0
    reps = 20
    for s in range(reps):
        rng = np.random.default_rng(s)
        n = 600
        env = np.r_[np.zeros(n // 2, dtype=int), np.ones(n // 2, dtype=int)]
        x1 = 3.0 * env + rng.normal(size=n)
        y = 1.5 * x1 + rng.normal(size=n)
        x2 = y + rng.normal(size=n)  # a child of y
        res = sp.icp(pd.DataFrame({"X1": x1, "X2": x2}), y, env, alpha=0.05)
        exact += sorted(res.parents) == ["X1"]
        child_included += "X2" in res.parents
    # ICP controls the probability of a false parent at alpha.
    assert child_included <= 3
    assert exact / reps >= 0.8


@pytest.fixture(scope="module")
def lagged_chain() -> pd.DataFrame:
    """X(t-1) -> Y(t) -> Z(t+1), plus own lags of X and Y."""
    rng = np.random.default_rng(0)
    T = 600
    X, Y, Z = np.zeros(T), np.zeros(T), np.zeros(T)
    for t in range(1, T):
        X[t] = 0.5 * X[t - 1] + rng.normal(0, 0.5)
        Y[t] = 0.4 * X[t - 1] + 0.3 * Y[t - 1] + rng.normal(0, 0.5)
        Z[t] = 0.6 * Y[t - 1] + rng.normal(0, 0.5)
    return pd.DataFrame({"X": X, "Y": Y, "Z": Z})


def test_pcmci_finds_the_lagged_chain(lagged_chain):
    res = sp.pcmci(lagged_chain, variables=["X", "Y", "Z"], tau_max=2, pc_alpha=0.05)
    links = res.discovered_links()
    found = {(r.source, r.target, int(r.lag)) for r in links.itertuples()}
    for edge in [("X", "X", 1), ("X", "Y", 1), ("Y", "Y", 1), ("Y", "Z", 1)]:
        assert edge in found
    # Nothing runs against the arrow of the chain.
    assert not any(src == "Z" and tgt in ("X", "Y") for src, tgt, _ in found)
    strong = links[links["p_value"] < 1e-6]
    assert {(r.source, r.target) for r in strong.itertuples()} == {
        ("X", "X"),
        ("X", "Y"),
        ("Y", "Y"),
        ("Y", "Z"),
    }


def test_lpcmci_orients_the_lagged_chain(lagged_chain):
    res = sp.lpcmci(lagged_chain, variables=["X", "Y", "Z"], tau_max=2, alpha=0.05)
    idx = {"X": 0, "Y": 1, "Z": 2}
    for src, tgt in [("X", "Y"), ("Y", "Z"), ("X", "X"), ("Y", "Y")]:
        assert res.edge_types[1, idx[src], idx[tgt]] == "-->"
    for src, tgt in [("Z", "X"), ("Z", "Y"), ("Y", "X")]:
        assert res.edge_types[1, idx[src], idx[tgt]] in ("", None)


def test_dynotears_recovers_the_structural_var_on_unit_scale_data():
    rng = np.random.default_rng(0)
    T = 2000
    x, z, w = np.zeros(T), np.zeros(T), np.zeros(T)
    for t in range(1, T):
        x[t] = 0.6 * x[t - 1] + rng.normal()
        z[t] = 0.5 * x[t - 1] + rng.normal()
        w[t] = 0.4 * z[t] + rng.normal()
    res = sp.dynotears(pd.DataFrame({"x": x, "z": z, "w": w}), lag=1, threshold=0.1)
    W, A = np.asarray(res.W), np.asarray(res.A[0])
    # Planted: lagged x -> x (0.6), lagged x -> z (0.5), contemporaneous
    # z -> w (0.4). The L1 penalty shrinks each by a few hundredths. The
    # default penalties assume roughly unit-variance series; at an
    # innovation SD of 0.3 they remove the contemporaneous edge.
    assert A[0, 0] == pytest.approx(0.6, abs=0.1)
    assert A[0, 1] == pytest.approx(0.5, abs=0.1)
    assert W[1, 2] == pytest.approx(0.4, abs=0.1)
    planted_A = np.zeros((3, 3), bool)
    planted_A[0, 0] = planted_A[0, 1] = True
    planted_W = np.zeros((3, 3), bool)
    planted_W[1, 2] = True
    assert np.all(A[~planted_A] == 0)
    assert np.all(W[~planted_W] == 0)


# --------------------------------------------------------------------- #
#  Mendelian randomisation, bandits, multiple testing
# --------------------------------------------------------------------- #


def test_mr_clust_separates_two_causal_clusters():
    rng = np.random.default_rng(8)
    bx = rng.uniform(0.1, 0.5, 60)
    sx = np.full(60, 0.01)
    sy = np.full(60, 0.01)
    theta = np.r_[np.full(30, 0.3), np.full(30, -0.4)]
    by = theta * bx + rng.normal(0, sy)
    res = sp.mr_clust(bx, by, sx, sy, K_range=(1, 4))
    est = res.cluster_estimates
    substantive = est[est["n_snps"] > 0].sort_values("estimate")
    assert substantive["n_snps"].tolist() == [30, 30]
    low, high = substantive["estimate"].tolist()
    se_low, se_high = substantive["se"].tolist()
    assert _within(low, -0.4, se_low)
    assert _within(high, 0.3, se_high)


def test_causal_bandit_estimates_arm_means_and_picks_the_best():
    rng = np.random.default_rng(0)
    true = {"A": 1.0, "B": 0.3, "C": 0.6}
    res = sp.causal_bandit(
        ["A", "B", "C"],
        reward_fn=lambda arm, context: true[arm] + rng.normal(0, 0.5),
        n_samples=3000,
        rng_seed=0,
    )
    assert res.arm_labels[res.optimal_arm] == "A"
    # 3,000 pulls spread over three arms: SE at most 0.5 / sqrt(300) = 0.03
    # for any arm pulled 300 times or more.
    np.testing.assert_allclose(
        res.expected_rewards, [true[a] for a in res.arm_labels], atol=0.12
    )


def test_westfall_young_controls_the_familywise_error_rate():
    reject_adjusted, reject_raw = 0, 0
    reps = 80
    for s in range(reps):
        rng = np.random.default_rng(s)
        df = pd.DataFrame({"Z": rng.permutation([0, 1] * 40)})
        outcomes = [f"y{j}" for j in range(5)]
        for name in outcomes:
            df[name] = rng.normal(size=80)  # every null is true
        out = sp.westfall_young(df, y=outcomes, treat="Z", n_perms=300, seed=s)
        # Step-down adjustment never lowers a p-value.
        assert (out["p_wy"] >= out["p_perm"] - 1e-12).all()
        reject_adjusted += out["p_wy"].min() < 0.05
        reject_raw += out["p_perm"].min() < 0.05
    # With a true FWER of 5%, more than 9 rejections in 80 has probability
    # below 0.01. The unadjusted minimum p-value rejects about 1 - 0.95**5
    # = 23% of the time, so the adjustment is doing the work.
    assert reject_adjusted <= 9
    assert reject_raw >= 10


def test_westfall_young_detects_a_real_effect():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"Z": rng.permutation([0, 1] * 100)})
    df["y1"] = 0.8 * df["Z"] + rng.normal(size=200)
    df["y2"] = rng.normal(size=200)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = sp.westfall_young(df, y=["y1", "y2"], treat="Z", n_perms=1000, seed=1)
    out = out.set_index("outcome")
    assert out.loc["y1", "p_wy"] < 0.01
    assert out.loc["y2", "p_wy"] > 0.05
