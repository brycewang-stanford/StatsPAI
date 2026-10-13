"""Options and degenerate inputs of the weighting / outcome-model estimators.

``sp.ipw``, ``sp.g_computation``, ``sp.aipw`` and ``sp.pate``: every test
compares with a computation written out here (stacked estimating equations
differentiated numerically, a hand-rolled bootstrap with the same draws, a
closed form) or asserts an identity.
"""

import itertools
import warnings

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm
from scipy import optimize, stats

import statspai as sp
from statspai.exceptions import (
    ConvergenceWarning,
    DataInsufficient,
    MethodIncompatibility,
    StatsPAIError,
)


def _obs(seed: int = 7, n: int = 600) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    z = rng.normal(size=n)
    p = 1 / (1 + np.exp(-(0.9 * x - 0.6 * z)))
    d = (rng.uniform(size=n) < p).astype(float)
    y = 1 + 2 * d + x + d * z + rng.normal(size=n)
    return pd.DataFrame(
        {
            "y": y,
            "d": d,
            "x": x,
            "z": z,
            "w": rng.integers(1, 4, n).astype(float),
            "g": rng.integers(0, 40, n),
        }
    )


def _raw_ipw_weights(e, t, estimand):
    if estimand == "ATE":
        return t / e, (1 - t) / (1 - e)
    if estimand == "ATT":
        return t, (1 - t) * e / (1 - e)
    if estimand == "ATC":
        return t * (1 - e) / e, 1 - t
    if estimand == "ATO":
        return t * (1 - e), (1 - t) * e
    m = np.minimum(e, 1 - e)
    return t * m / e, (1 - t) * m / (1 - e)


def _ipw_m_estimate(df, estimand, normalize, trim, weighted, clustered, model):
    """Point estimate and sandwich SE from the stacked estimating equations.

    The propensity score equation, then either the two Hajek means or the
    Horvitz-Thompson contrast. The Jacobian is a central difference, the
    meat the outer product of the (cluster-summed) rows, divisor ``n``.
    """
    n = len(df)
    y = df["y"].to_numpy()
    d = df["d"].to_numpy()
    cl = df["g"].to_numpy()
    Xc = np.column_stack([np.ones(n), df["x"], df["z"]])
    ww = df["w"].to_numpy() * n / df["w"].sum() if weighted else np.ones(n)
    link = (lambda v: 1 / (1 + np.exp(-v))) if model == "logit" else stats.norm.cdf

    def score(gam):
        idx = Xc @ gam
        e = link(idx)
        s = d - e if model == "logit" else (d - e) * stats.norm.pdf(idx) / (e * (1 - e))
        return Xc * (ww * s)[:, None]

    gam = optimize.root(lambda gm: score(gm).mean(0), np.zeros(3), tol=1e-13).x

    def arms(gm):
        e = link(Xc @ gm)
        if trim > 0:
            e = np.clip(e, trim, 1 - trim)
        return _raw_ipw_weights(e, d, estimand)

    a1, a0 = arms(gam)
    target = {"ATT": d, "ATC": 1 - d}.get(estimand, np.ones(n))
    if normalize:
        theta = np.r_[
            gam,
            np.sum(ww * a1 * y) / np.sum(ww * a1),
            np.sum(ww * a0 * y) / np.sum(ww * a0),
        ]

        def psi(t):
            b1, b0 = arms(t[:3])
            return np.column_stack(
                [score(t[:3]), ww * b1 * (y - t[3]), ww * b0 * (y - t[4])]
            )

        c = np.r_[0, 0, 0, 1, -1]
    else:
        theta = np.r_[gam, np.sum(ww * (a1 - a0) * y) / np.sum(ww * target)]

        def psi(t):
            b1, b0 = arms(t[:3])
            return np.column_stack(
                [score(t[:3]), ww * (b1 - b0) * y - t[3] * ww * target]
            )

        c = np.r_[0, 0, 0, 1]
    k = len(theta)
    A = np.zeros((k, k))
    for j in range(k):
        h = 1e-6
        tp, tm = theta.copy(), theta.copy()
        tp[j] += h
        tm[j] -= h
        A[:, j] = (psi(tp).mean(0) - psi(tm).mean(0)) / (2 * h)
    P = psi(theta)
    if clustered:
        P = np.array([P[cl == u].sum(0) for u in np.unique(cl)])
    Ai = np.linalg.inv(A)
    V = Ai @ (P.T @ P / n) @ Ai.T / n
    return float(c @ theta), float(np.sqrt(c @ V @ c))


class TestIpwSandwich:
    @pytest.mark.parametrize("estimand", ["ATE", "ATT", "ATC", "ATO", "ATM"])
    def test_sandwich_is_the_stacked_m_estimation_variance(self, estimand):
        df = _obs()
        grid = itertools.product(
            [True, False],
            [0.0, 0.25],
            [False, True],
            [False, True],
            ["logit", "probit"],
        )
        for normalize, trim, weighted, clustered, model in grid:
            if estimand in ("ATO", "ATM") and not normalize:
                continue
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit = sp.ipw(
                    df,
                    y="y",
                    treat="d",
                    covariates=["x", "z"],
                    estimand=estimand,
                    normalize=normalize,
                    trim=trim,
                    weights="w" if weighted else None,
                    cluster="g" if clustered else None,
                    ps_model=model,
                    se_method="sandwich",
                )
            est, se = _ipw_m_estimate(
                df, estimand, normalize, trim, weighted, clustered, model
            )
            label = (estimand, normalize, trim, weighted, clustered, model)
            # The reference Jacobian is a central difference with step 1e-6,
            # good to about 1e-8 relative; the point estimate is a root find.
            assert fit.estimate == pytest.approx(est, abs=1e-7), label
            assert fit.se == pytest.approx(se, rel=1e-6), label

    def test_atc_is_the_att_with_the_arms_swapped(self):
        df = _obs()
        swapped = df.assign(d=1 - df["d"])
        kw = dict(y="y", treat="d", covariates=["x", "z"], se_method="sandwich")
        atc = sp.ipw(df, estimand="ATC", **kw)
        att = sp.ipw(swapped, estimand="ATT", **kw)
        # Relabelling the arms flips the sign of the contrast and nothing else.
        assert atc.estimate == pytest.approx(-att.estimate, abs=1e-9)
        assert atc.se == pytest.approx(att.se, rel=1e-8)
        assert sp.ipw(df, estimand="ATU", **kw).estimate == atc.estimate

    def test_sampling_weights_equal_duplicated_rows_clustered_on_the_unit(self):
        # An integer weight w is w copies of the row. The pweighted sandwich
        # sums w * psi per unit, which is what clustering the copies on the
        # unit does, and both divide by the same total, so the two standard
        # errors agree up to the logit solver's tolerance.
        df = _obs(n=300).assign(unit=np.arange(300))
        long = df.loc[df.index.repeat(df["w"].astype(int))].reset_index(drop=True)
        for estimand, normalize in [("ATE", True), ("ATT", True), ("ATE", False)]:
            kw = dict(
                y="y",
                treat="d",
                covariates=["x", "z"],
                estimand=estimand,
                normalize=normalize,
                se_method="sandwich",
            )
            weighted = sp.ipw(df, weights="w", **kw)
            dup = sp.ipw(long, cluster="unit", **kw)
            assert weighted.estimate == pytest.approx(dup.estimate, abs=1e-8)
            assert weighted.se == pytest.approx(dup.se, rel=1e-7)

    def test_weights_are_scale_free(self):
        df = _obs(n=300)
        kw = dict(y="y", treat="d", covariates=["x", "z"], se_method="sandwich")
        a = sp.ipw(df, weights="w", **kw)
        b = sp.ipw(df.assign(w=df["w"] * 37.5), weights="w", **kw)
        assert a.estimate == pytest.approx(b.estimate, abs=1e-9)
        assert a.se == pytest.approx(b.se, rel=1e-8)


class TestIpwPointEstimate:
    def test_known_propensity_equal_to_the_treated_share_makes_ht_and_hajek_agree(
        self,
    ):
        df = _obs(n=200)
        share = float(df["d"].mean())
        kw = dict(
            y="y", treat="d", covariates=["x"], propensity=share, se_method="sandwich"
        )
        hajek = sp.ipw(df, normalize=True, **kw)
        ht = sp.ipw(df, normalize=False, **kw)
        diff = df.loc[df.d == 1, "y"].mean() - df.loc[df.d == 0, "y"].mean()
        # sum(T / p) = n exactly when p is the treated share, so the
        # Horvitz-Thompson divisor n is the Hajek divisor.
        assert hajek.estimate == pytest.approx(diff, abs=1e-12)
        assert ht.estimate == pytest.approx(diff, abs=1e-12)

    @pytest.mark.parametrize("estimand", ["ATE", "ATT", "ATC"])
    def test_trim_clips_the_fitted_scores(self, estimand):
        df = _obs(n=300)
        X = sm.add_constant(df[["x", "z"]].to_numpy())
        e = sm.Logit(df["d"].to_numpy(), X).fit(disp=0).predict(X)
        e = np.clip(e, 0.2, 0.8)
        a1, a0 = _raw_ipw_weights(e, df["d"].to_numpy(), estimand)
        y = df["y"].to_numpy()
        expected = np.sum(a1 * y) / a1.sum() - np.sum(a0 * y) / a0.sum()
        fit = sp.ipw(
            df,
            y="y",
            treat="d",
            covariates=["x", "z"],
            estimand=estimand,
            trim=0.2,
            se_method="sandwich",
        )
        # Two logit solvers; they agree to their convergence tolerance.
        assert fit.estimate == pytest.approx(expected, abs=1e-7)
        assert fit.model_info["pscore_min"] >= 0.2 - 1e-12
        assert fit.model_info["pscore_max"] <= 0.8 + 1e-12

    def test_rows_with_missing_values_are_dropped_listwise(self):
        df = _obs(n=300)
        holes = df.copy()
        holes.loc[holes.index[:7], "x"] = np.nan
        holes.loc[holes.index[7:12], "w"] = np.nan
        kw = dict(
            y="y", treat="d", covariates=["x", "z"], weights="w", se_method="sandwich"
        )
        a = sp.ipw(holes, **kw)
        b = sp.ipw(holes.dropna().reset_index(drop=True), **kw)
        assert a.n_obs == 288
        assert a.estimate == pytest.approx(b.estimate, abs=1e-12)
        assert a.se == pytest.approx(b.se, rel=1e-12)

    def test_categorical_covariate_is_expanded_into_dummies(self):
        df = _obs(n=300)
        cat = pd.Categorical(np.array(["a", "b", "c"])[np.arange(300) % 3])
        with_cat = df.assign(c=cat)
        manual = df.assign(cb=(cat == "b").astype(float), cc=(cat == "c").astype(float))
        kw = dict(y="y", treat="d", se_method="sandwich")
        a = sp.ipw(with_cat, covariates=["x", "c"], **kw)
        b = sp.ipw(manual, covariates=["x", "cb", "cc"], **kw)
        assert a.estimate == pytest.approx(b.estimate, abs=1e-10)
        assert a.se == pytest.approx(b.se, rel=1e-9)


class TestIpwBootstrap:
    def test_bootstrap_se_is_the_hand_rolled_loop_with_the_same_draws(self):
        df = _obs(n=150)
        y, d = df["y"].to_numpy(), df["d"].to_numpy()
        X = sm.add_constant(df[["x"]].to_numpy())
        rng = np.random.RandomState(11)
        draws = []
        for _ in range(40):
            idx = rng.choice(150, size=150, replace=True)
            e = sm.GLM(d[idx], X[idx], family=sm.families.Binomial()).fit()
            e = e.predict(X[idx])
            a1, a0 = _raw_ipw_weights(e, d[idx], "ATE")
            draws.append(
                np.sum(a1 * y[idx]) / a1.sum() - np.sum(a0 * y[idx]) / a0.sum()
            )
        fit = sp.ipw(df, y="y", treat="d", covariates=["x"], n_bootstrap=40, seed=11)
        # Same generator, same draws, same logit; only solver round-off.
        assert fit.se == pytest.approx(np.std(draws, ddof=1), rel=1e-8)

    def test_cluster_bootstrap_resamples_whole_clusters(self):
        df = _obs(n=150)
        y, d = df["y"].to_numpy(), df["d"].to_numpy()
        codes = pd.factorize(df["g"])[0]
        members = [np.flatnonzero(codes == c) for c in range(codes.max() + 1)]
        rng = np.random.RandomState(5)
        draws = []
        for _ in range(30):
            pick = rng.choice(len(members), size=len(members), replace=True)
            idx = np.concatenate([members[c] for c in pick])
            a1, a0 = _raw_ipw_weights(np.full(len(idx), 0.4), d[idx], "ATE")
            draws.append(
                np.sum(a1 * y[idx]) / a1.sum() - np.sum(a0 * y[idx]) / a0.sum()
            )
        fit = sp.ipw(
            df,
            y="y",
            treat="d",
            covariates=["x"],
            propensity=0.4,
            cluster="g",
            n_bootstrap=30,
            seed=5,
        )
        assert fit.se == pytest.approx(np.std(draws, ddof=1), rel=1e-10)
        assert fit.model_info["n_clusters"] == len(members)

    def test_bootstrap_se_does_not_depend_on_the_level_of_the_outcome(self):
        # The Hajek estimate is unchanged when a constant is added to the
        # outcome (the docstring says so), and so is every bootstrap
        # replicate that has both arms. With one treated unit in 40 about a
        # third of the resamples have none.
        rng = np.random.default_rng(3)
        x = rng.normal(size=40)
        d = np.zeros(40)
        d[0] = 1.0
        df = pd.DataFrame({"y": 1 + x + rng.normal(size=40), "d": d, "x": x})
        kw = dict(y="y", treat="d", covariates=["x"], n_bootstrap=200, seed=0)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            base = sp.ipw(df, **kw)
            shifted = sp.ipw(df.assign(y=df["y"] + 1000.0), **kw)
        assert shifted.estimate == pytest.approx(base.estimate, abs=1e-8)
        assert shifted.se == pytest.approx(base.se, rel=1e-6)

    def test_one_bootstrap_replicate_is_refused(self):
        with pytest.raises(ValueError):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sp.ipw(_obs(n=100), y="y", treat="d", covariates=["x"], n_bootstrap=1)


class TestIpwValidation:
    @pytest.mark.parametrize(
        "kwargs, exc",
        [
            (dict(estimand="LATE"), ValueError),
            (dict(se_method="hc1"), MethodIncompatibility),
            (dict(ps_model="cloglog"), MethodIncompatibility),
            (dict(estimand="ATO", normalize=False), MethodIncompatibility),
            (dict(propensity=1.0), MethodIncompatibility),
            (dict(weights="neg"), MethodIncompatibility),
            (dict(cluster="one"), MethodIncompatibility),
            (dict(covariates=["x", "nope"]), MethodIncompatibility),
        ],
    )
    def test_invalid_options_are_refused(self, kwargs, exc):
        df = _obs(n=100).assign(neg=-1.0, one=1)
        call = dict(y="y", treat="d", covariates=["x"], se_method="sandwich")
        call.update(kwargs)
        with pytest.raises(exc):
            sp.ipw(df, **call)

    def test_three_valued_and_single_arm_treatments_are_refused(self):
        df = _obs(n=100)
        with pytest.raises(ValueError, match="binary"):
            sp.ipw(df.assign(d=np.arange(100) % 3), y="y", treat="d", covariates=["x"])
        with pytest.raises(ValueError, match="both treated and control"):
            sp.ipw(df.assign(d=1.0), y="y", treat="d", covariates=["x"])

    @pytest.mark.parametrize("trim", [0.5, 0.6, -0.1])
    def test_trim_outside_zero_to_one_half_is_refused(self, trim):
        with pytest.raises(ValueError):
            sp.ipw(
                _obs(n=100),
                y="y",
                treat="d",
                covariates=["x"],
                trim=trim,
                se_method="sandwich",
            )

    def test_perfect_separation_is_flagged(self):
        rng = np.random.default_rng(3)
        x = rng.normal(size=300)
        d = (x > 0).astype(float)
        df = pd.DataFrame({"y": 1 + 2 * d + x + rng.normal(size=300), "d": d, "x": x})
        flagged = False
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            try:
                sp.ipw(df, y="y", treat="d", covariates=["x"], se_method="sandwich")
            except StatsPAIError:
                flagged = True
        flagged = flagged or any(
            "statspai" in type(w.message).__module__ for w in caught
        )
        assert flagged


class TestAlphaIsValidated:
    """``alpha`` outside (0, 1) must not produce an inverted or NaN interval."""

    def test_ipw(self):
        with pytest.raises(ValueError):
            sp.ipw(
                _obs(n=100),
                y="y",
                treat="d",
                covariates=["x"],
                se_method="sandwich",
                alpha=2,
            )

    def test_aipw(self):
        with pytest.raises(ValueError):
            sp.aipw(_obs(n=100), y="y", treat="d", covariates=["x"], alpha=3)

    def test_g_computation(self):
        with pytest.raises(ValueError):
            sp.g_computation(_obs(n=100), "y", "d", ["x"], n_boot=20, seed=1, alpha=1.5)


# --------------------------------------------------------------------- #
#  g-computation
# --------------------------------------------------------------------- #


def _ra_m_estimate(df, estimand, ps):
    """Regression adjustment (optionally IPW-weighted) as stacked moments."""
    n = len(df)
    y = df["y"].to_numpy()
    d = df["d"].to_numpy()
    X = np.column_stack([np.ones(n), df["x"], df["z"]])
    Z = X[:, :2] if ps else None
    att = estimand == "ATT"

    def wts(gam):
        if Z is None:
            return np.ones(n), np.ones(n)
        p = 1 / (1 + np.exp(-(Z @ gam)))
        return (np.ones(n), p / (1 - p)) if att else (1 / p, 1 / (1 - p))

    def psi(t):
        gam, b1, b0, tau = t[:2], t[2:5], t[5:8], t[8]
        w1, w0 = wts(gam)
        blocks = []
        if Z is not None:
            p = 1 / (1 + np.exp(-(Z @ gam)))
            blocks.append(Z * (d - p)[:, None])
        else:
            blocks.append(np.zeros((n, 2)) + gam)  # inert placeholder rows
        blocks.append(X * (d * w1 * (y - X @ b1))[:, None])
        blocks.append(X * ((1 - d) * w0 * (y - X @ b0))[:, None])
        contrast = X @ b1 - X @ b0 - tau
        blocks.append((d * contrast if att else contrast)[:, None])
        return np.column_stack(blocks)

    start = np.zeros(9)
    theta = optimize.root(lambda t: psi(t).mean(0), start, tol=1e-13).x
    k = len(theta)
    A = np.zeros((k, k))
    for j in range(k):
        h = 1e-6
        tp, tm = theta.copy(), theta.copy()
        tp[j] += h
        tm[j] -= h
        A[:, j] = (psi(tp).mean(0) - psi(tm).mean(0)) / (2 * h)
    P = psi(theta)
    Ai = np.linalg.inv(A)
    V = Ai @ (P.T @ P / n) @ Ai.T / n
    return float(theta[8]), float(np.sqrt(V[8, 8]))


class TestGComputation:
    def test_additive_ols_gives_the_coefficient_for_every_estimand(self):
        df = _obs(n=300)
        design = sm.add_constant(df[["d", "x", "z"]].to_numpy())
        coef = sm.OLS(df["y"].to_numpy(), design).fit().params[1]
        for estimand in ("ATE", "ATT", "ATC", "ATU"):
            fit = sp.g_computation(
                df, "y", "d", ["x", "z"], estimand=estimand, n_boot=10, seed=0
            )
            # Without an interaction Q(1, X) - Q(0, X) is the coefficient
            # on every row, whichever rows are averaged.
            assert fit.estimate == pytest.approx(coef, abs=1e-10)

    def test_bootstrap_is_the_hand_rolled_loop_with_the_same_draws(self):
        df = _obs(n=120)
        y = df["y"].to_numpy()
        design = sm.add_constant(df[["d", "x"]].to_numpy())
        rng = np.random.default_rng(4)
        draws = []
        for _ in range(50):
            idx = rng.integers(0, 120, size=120)
            draws.append(sm.OLS(y[idx], design[idx]).fit().params[1])
        fit = sp.g_computation(df, "y", "d", ["x"], n_boot=50, seed=4, alpha=0.1)
        assert fit.se == pytest.approx(np.std(draws, ddof=1), rel=1e-9)
        lo, hi = np.percentile(draws, [5, 95])
        assert fit.ci[0] == pytest.approx(lo, abs=1e-9)
        assert fit.ci[1] == pytest.approx(hi, abs=1e-9)
        assert fit.model_info["n_boot_success"] == 50

    def test_sklearn_outcome_model_matches_the_default_ols(self):
        from sklearn.linear_model import LinearRegression

        df = _obs(n=120)
        a = sp.g_computation(df, "y", "d", ["x", "z"], n_boot=20, seed=2)
        b = sp.g_computation(
            df, "y", "d", ["x", "z"], n_boot=20, seed=2, ml_Q=LinearRegression()
        )
        # The same least-squares fit through two libraries.
        assert b.estimate == pytest.approx(a.estimate, abs=1e-9)
        assert b.se == pytest.approx(a.se, rel=1e-7)
        assert b.model_info["ml_Q"] == "LinearRegression"

    def test_dose_response_of_a_linear_model_is_a_line(self):
        rng = np.random.default_rng(0)
        n = 200
        x = rng.normal(size=n)
        dose = rng.uniform(0, 4, n) + 0.3 * x
        df = pd.DataFrame({"y": 1 + 0.7 * dose + x + rng.normal(size=n), "d": dose})
        df["x"] = x
        design = sm.add_constant(df[["d", "x"]].to_numpy())
        b = sm.OLS(df["y"].to_numpy(), design).fit().params
        grid = [0.0, 1.5, 4.0]
        fit = sp.g_computation(
            df,
            "y",
            "d",
            ["x"],
            estimand="dose_response",
            treat_values=grid,
            n_boot=15,
            seed=0,
        )
        expected = b[0] + b[1] * np.array(grid) + b[2] * x.mean()
        np.testing.assert_allclose(fit.detail["estimate"], expected, atol=1e-10)
        np.testing.assert_allclose(fit.detail["dose"], grid)
        assert fit.estimate == pytest.approx(expected[0], abs=1e-10)

    def test_by_arm_is_the_fully_interacted_regression(self):
        df = _obs(n=300)
        y, d = df["y"].to_numpy(), df["d"].to_numpy()
        X = sm.add_constant(df[["x", "z"]].to_numpy())
        b1 = np.linalg.lstsq(X[d == 1], y[d == 1], rcond=None)[0]
        b0 = np.linalg.lstsq(X[d == 0], y[d == 0], rcond=None)[0]
        kw = dict(by_arm=True, se_method="analytic")
        ate = sp.g_computation(df, "y", "d", ["x", "z"], **kw)
        att = sp.g_computation(df, "y", "d", ["x", "z"], estimand="ATT", **kw)
        assert ate.estimate == pytest.approx(np.mean(X @ (b1 - b0)), abs=1e-10)
        # On the treated the arm-1 fit averages to the arm-1 mean.
        expected_att = y[d == 1].mean() - np.mean(X[d == 1] @ b0)
        assert att.estimate == pytest.approx(expected_att, abs=1e-10)
        assert att.model_info["pomean0"] == pytest.approx(
            np.mean(X[d == 1] @ b0), abs=1e-10
        )

    @pytest.mark.parametrize("estimand", ["ATE", "ATT"])
    @pytest.mark.parametrize("ps", [False, True])
    def test_analytic_se_is_the_stacked_sandwich(self, estimand, ps):
        df = _obs(n=400)
        fit = sp.g_computation(
            df,
            "y",
            "d",
            ["x", "z"],
            estimand=estimand,
            by_arm=True,
            ps_covariates=["x"] if ps else None,
            se_method="analytic",
        )
        est, se = _ra_m_estimate(df, estimand, ps)
        # The reference's ATT moment is d * (m1 - m0 - tau); the share of
        # treated cancels between its Jacobian and its meat.
        assert fit.estimate == pytest.approx(est, abs=1e-7)
        # Central-difference Jacobian, step 1e-6.
        assert fit.se == pytest.approx(se, rel=1e-5)
        z = stats.norm.ppf(0.975)
        assert fit.ci[0] == pytest.approx(fit.estimate - z * fit.se, abs=1e-12)

    @pytest.mark.parametrize(
        "kwargs, exc",
        [
            (dict(estimand="late"), ValueError),
            (dict(se_method="hc1"), MethodIncompatibility),
            (dict(ps_covariates=["x"]), MethodIncompatibility),
            (dict(se_method="analytic"), MethodIncompatibility),
            (dict(by_arm=True, estimand="ATC"), MethodIncompatibility),
            (dict(estimand="dose_response"), ValueError),
            (dict(estimand="dose_response", treat_values=[]), ValueError),
        ],
    )
    def test_invalid_option_combinations_are_refused(self, kwargs, exc):
        with pytest.raises(exc):
            sp.g_computation(_obs(n=80), "y", "d", ["x"], n_boot=5, **kwargs)

    def test_non_binary_treatment_needs_dose_response(self):
        df = _obs(n=80)
        df = df.assign(d=np.arange(80) % 3)
        with pytest.raises(ValueError, match="binary"):
            sp.g_computation(df, "y", "d", ["x"], n_boot=5)
        with pytest.raises(ValueError, match="binary"):
            sp.g_computation(df, "y", "d", ["x"], by_arm=True, se_method="analytic")

    def test_reported_bootstrap_successes_exclude_empty_target_arms(self):
        rng = np.random.default_rng(3)
        n = 30
        x = rng.normal(size=n)
        d = np.zeros(n)
        d[0] = 1.0
        df = pd.DataFrame({"y": 1 + x + rng.normal(size=n), "d": d, "x": x})
        draw = np.random.default_rng(1)
        usable = sum(bool(d[draw.integers(0, n, size=n)].sum() > 0) for _ in range(40))
        assert usable < 40  # the design does produce empty resamples
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = sp.g_computation(
                df, "y", "d", ["x"], estimand="ATT", n_boot=40, seed=1
            )
        assert fit.model_info["n_boot_success"] == usable

    def test_single_arm_treatment_is_refused_with_a_clear_message(self):
        df = _obs(n=80).assign(d=1.0)
        with pytest.raises(ValueError, match="(?i)control|untreated|both|arm"):
            sp.g_computation(df, "y", "d", ["x"], n_boot=5)


# --------------------------------------------------------------------- #
#  AIPW
# --------------------------------------------------------------------- #


def _aipw_by_hand(df, folds=None, weights=None):
    n = len(df)
    y, d = df["y"].to_numpy(), df["d"].to_numpy()
    X = sm.add_constant(df[["x", "z"]].to_numpy(), has_constant="add")
    e, m1, m0 = np.zeros(n), np.zeros(n), np.zeros(n)
    splits = (
        [(np.ones(n, bool), np.ones(n, bool))]
        if folds is None
        else [(folds != f, folds == f) for f in np.unique(folds)]
    )
    for tr, te in splits:
        e[te] = sm.Logit(d[tr], X[tr]).fit(disp=0).predict(X[te])
        t1, t0 = tr & (d == 1), tr & (d == 0)
        m1[te] = X[te] @ np.linalg.lstsq(X[t1], y[t1], rcond=None)[0]
        m0[te] = X[te] @ np.linalg.lstsq(X[t0], y[t0], rcond=None)[0]
    return y, d, e, m1, m0


class TestAipw:
    def test_full_sample_ate_and_influence_se(self):
        df = _obs(n=300)
        y, d, e, m1, m0 = _aipw_by_hand(df)
        psi = m1 - m0 + d * (y - m1) / e - (1 - d) * (y - m0) / (1 - e)
        fit = sp.aipw(df, "y", "d", ["x", "z"], cross_fit=False, trim=0.0)
        assert fit.estimate == pytest.approx(psi.mean(), abs=1e-8)
        assert fit.se == pytest.approx(np.sqrt(psi.var(ddof=1) / 300), rel=1e-7)
        means = fit.model_info["potential_outcome_means"]
        assert means[1] - means[0] == pytest.approx(fit.estimate, abs=1e-12)

    def test_cross_fitting_uses_the_seeded_fold_assignment(self):
        df = _obs(n=300)
        folds = np.random.default_rng(9).choice(4, size=300)
        y, d, e, m1, m0 = _aipw_by_hand(df, folds)
        psi = m1 - m0 + d * (y - m1) / e - (1 - d) * (y - m0) / (1 - e)
        fit = sp.aipw(df, "y", "d", ["x", "z"], n_folds=4, seed=9, trim=0.0)
        assert fit.estimate == pytest.approx(psi.mean(), abs=1e-8)
        assert fit.se == pytest.approx(np.sqrt(psi.var(ddof=1) / 300), rel=1e-7)

    def test_att_is_the_outcome_regression_plus_weighted_control_residuals(self):
        df = _obs(n=300)
        y, d, e, _, m0 = _aipw_by_hand(df)
        p = d.mean()
        psi = d * (y - m0) / p - (1 - d) * e * (y - m0) / ((1 - e) * p)
        fit = sp.aipw(
            df, "y", "d", ["x", "z"], estimand="ATT", cross_fit=False, trim=0.0
        )
        assert fit.estimate == pytest.approx(psi.mean(), abs=1e-8)
        resid = psi - psi.mean() * d / p
        assert fit.se == pytest.approx(np.sqrt(resid.var(ddof=1) / 300), rel=1e-7)

    def test_trim_clips_and_counts_the_scores_outside_the_bounds(self):
        df = _obs(n=300)
        y, d, e, m1, m0 = _aipw_by_hand(df)
        outside = int(np.sum((e < 0.25) | (e > 0.75)))
        ec = np.clip(e, 0.25, 0.75)
        psi = m1 - m0 + d * (y - m1) / ec - (1 - d) * (y - m0) / (1 - ec)
        with pytest.warns(RuntimeWarning, match="clipped"):
            fit = sp.aipw(df, "y", "d", ["x", "z"], cross_fit=False, trim=0.25)
        assert outside > 0
        assert fit.model_info["n_propensity_clipped"] == outside
        assert fit.estimate == pytest.approx(psi.mean(), abs=1e-8)

    @pytest.mark.parametrize("estimand", ["ATE", "ATT"])
    def test_sampling_weights_equal_duplicated_rows_clustered_on_the_unit(
        self, estimand
    ):
        # Same argument as for sp.ipw: an integer pweight is that many
        # copies of the row, clustered on the unit.
        df = _obs(n=300).assign(unit=np.arange(300))
        long = df.loc[df.index.repeat(df["w"].astype(int))].reset_index(drop=True)
        kw = dict(estimand=estimand, cross_fit=False, trim=0.0)
        a = sp.aipw(df, "y", "d", ["x", "z"], weights="w", **kw)
        b = sp.aipw(long, "y", "d", ["x", "z"], cluster="unit", **kw)
        assert a.estimate == pytest.approx(b.estimate, abs=1e-8)
        assert a.se == pytest.approx(b.se, rel=1e-7)

    def test_sandwich_with_weights_equals_duplicated_rows(self):
        df = _obs(n=300).assign(unit=np.arange(300))
        long = df.loc[df.index.repeat(df["w"].astype(int))].reset_index(drop=True)
        kw = dict(cross_fit=False, trim=0.0, se_method="sandwich")
        a = sp.aipw(df, "y", "d", ["x", "z"], weights="w", **kw)
        b = sp.aipw(long, "y", "d", ["x", "z"], cluster="unit", **kw)
        assert a.estimate == pytest.approx(b.estimate, abs=1e-8)
        assert a.se == pytest.approx(b.se, rel=1e-7)

    def test_known_propensity_is_used_as_given(self):
        df = _obs(n=300)
        y, d, _, m1, m0 = _aipw_by_hand(df)
        psi = m1 - m0 + d * (y - m1) / 0.4 - (1 - d) * (y - m0) / 0.6
        fit = sp.aipw(df, "y", "d", ["x", "z"], cross_fit=False, propensity=0.4)
        assert fit.estimate == pytest.approx(psi.mean(), abs=1e-9)
        assert fit.model_info["n_propensity_clipped"] == 0
        assert fit.model_info["propensity"] == "known"

    @pytest.mark.parametrize(
        "kwargs",
        [
            dict(se_method="bootstrap"),
            dict(se_method="sandwich"),  # cross_fit defaults to True
            dict(se_method="sandwich", cross_fit=False, estimand="ATT"),
            dict(se_method="sandwich", cross_fit=False, propensity=0.5),
            dict(outcome_model="tobit"),
            dict(trim=0.5),
            dict(trim=-0.01),
            dict(propensity=0.0),
            dict(cluster="one"),
            dict(weights="neg"),
            dict(outcome_model="logit"),  # outcome outside [0, 1]
        ],
    )
    def test_invalid_options_are_refused(self, kwargs):
        df = _obs(n=100).assign(one=1, neg=-1.0)
        with pytest.raises(MethodIncompatibility):
            sp.aipw(df, "y", "d", ["x"], **kwargs)

    def test_unknown_estimand_and_non_binary_treatment(self):
        df = _obs(n=100)
        with pytest.raises(ValueError, match="estimand"):
            sp.aipw(df, "y", "d", ["x"], estimand="ATC")
        with pytest.raises(ValueError, match="binary"):
            sp.aipw(df.assign(d=np.arange(100) % 3), "y", "d", ["x"])

    def test_covariate_constant_within_a_fold_does_not_break_cross_fitting(self):
        rng = np.random.default_rng(3)
        n = 300
        x = rng.normal(size=n)
        z = rng.normal(size=n)
        d = (rng.uniform(size=n) < 1 / (1 + np.exp(-x))).astype(float)
        y = 1 + 2 * d + x + d * z + rng.normal(size=n)
        dummy = (rng.uniform(size=n) < 0.97).astype(float)
        df = pd.DataFrame({"y": y, "d": d, "x": x, "z": z, "dum": dummy})
        folds = np.random.default_rng(1).choice(5, size=n)
        # the design: some test fold has the dummy equal to one throughout
        assert any(dummy[folds == f].min() == 1.0 for f in range(5))
        X = sm.add_constant(df[["x", "z", "dum"]].to_numpy(), has_constant="add")
        e, m1, m0 = np.zeros(n), np.zeros(n), np.zeros(n)
        for f in range(5):
            tr, te = folds != f, folds == f
            e[te] = sm.Logit(d[tr], X[tr]).fit(disp=0).predict(X[te])
            t1, t0 = tr & (d == 1), tr & (d == 0)
            m1[te] = X[te] @ np.linalg.lstsq(X[t1], y[t1], rcond=None)[0]
            m0[te] = X[te] @ np.linalg.lstsq(X[t0], y[t0], rcond=None)[0]
        psi = m1 - m0 + d * (y - m1) / e - (1 - d) * (y - m0) / (1 - e)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = sp.aipw(df, "y", "d", ["x", "z", "dum"], seed=1, trim=0.0)
        assert fit.estimate == pytest.approx(psi.mean(), abs=1e-7)

    def test_single_arm_treatment_is_refused(self):
        with pytest.raises(ValueError):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sp.aipw(_obs(n=100).assign(d=1.0), "y", "d", ["x"])

    def test_one_fold_cross_fitting_is_refused(self):
        with pytest.raises(ValueError):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sp.aipw(_obs(n=100), "y", "d", ["x"], n_folds=1)


# --------------------------------------------------------------------- #
#  PATE
# --------------------------------------------------------------------- #


def _trial_and_target():
    rng = np.random.default_rng(0)
    ne, nt = 400, 600
    age = rng.normal(45, 8, ne)
    edu = rng.normal(13, 2, ne)
    tr = rng.binomial(1, 0.5, ne).astype(float)
    out = 2 * tr + 0.05 * age * tr + 0.1 * edu + rng.normal(size=ne)
    trial = pd.DataFrame({"y": out, "d": tr, "age": age, "edu": edu})
    target = pd.DataFrame({"age": rng.normal(40, 8, nt), "edu": rng.normal(14, 2, nt)})
    return trial, target


def _hajek_contrast(y, d, w):
    return np.average(y[d == 1], weights=w[d == 1]) - np.average(
        y[d == 0], weights=w[d == 0]
    )


class TestPate:
    def _odds(self, trial, target):
        Xp = sm.add_constant(
            np.vstack([trial[["age", "edu"]].to_numpy(), target.to_numpy()])
        )
        s = np.r_[np.ones(len(trial)), np.zeros(len(target))]
        ps = sm.Logit(s, Xp).fit(disp=0).predict(Xp)[: len(trial)]
        return (1 - ps) / ps

    def test_ipw_is_the_inverse_odds_of_participation_contrast(self):
        trial, target = _trial_and_target()
        w = self._odds(trial, target)
        expected = _hajek_contrast(trial["y"].to_numpy(), trial["d"].to_numpy(), w)
        fit = sp.pate(
            trial, target, "y", "d", ["age", "edu"], n_boot=12, seed=0, trim=0.0
        )
        # sp.pate fits the participation logit by L-BFGS on the raw
        # covariates; it stops within about 1e-6 of the Newton solution.
        assert fit.estimate == pytest.approx(expected, abs=1e-5)
        assert int(fit.detail["n_boot_valid"].iloc[0]) == 12
        assert fit.estimand == "PATE"

    def test_aipw_adds_the_weighted_residuals_to_the_projected_effect(self):
        trial, target = _trial_and_target()
        w = self._odds(trial, target)
        y, d = trial["y"].to_numpy(), trial["d"].to_numpy()
        X = trial[["age", "edu"]].to_numpy()

        def design(xm, dv):
            return np.column_stack([np.ones(len(xm)), xm, dv, dv[:, None] * xm])

        beta = np.linalg.lstsq(design(X, d), y, rcond=None)[0]
        nt = len(target)
        proj = design(target.to_numpy(), np.ones(nt)) @ beta
        proj = proj - design(target.to_numpy(), np.zeros(nt)) @ beta
        r1 = y - design(X, np.ones(len(y))) @ beta
        r0 = y - design(X, np.zeros(len(y))) @ beta
        expected = (
            proj.mean()
            + np.average(r1[d == 1], weights=w[d == 1])
            - np.average(r0[d == 0], weights=w[d == 0])
        )
        fit = sp.pate(
            trial,
            target,
            "y",
            "d",
            ["age", "edu"],
            method="aipw",
            n_boot=12,
            seed=0,
            trim=0.0,
        )
        assert fit.estimate == pytest.approx(expected, abs=1e-5)

    def test_calibration_weights_reproduce_the_target_means(self):
        trial, target = _trial_and_target()
        est = sp.PATEEstimator(
            trial, target, "y", "d", ["age", "edu"], method="calibration", n_boot=10
        )
        X = trial[["age", "edu"]].to_numpy()
        w = est._entropy_balance(X, target.to_numpy())
        # L-BFGS on the unscaled dual stops at a gradient of about 1e-5.
        np.testing.assert_allclose(
            (w @ X) / w.sum(), target.mean().to_numpy(), rtol=1e-5
        )
        assert w.mean() == pytest.approx(1.0, abs=1e-12)
        fit = est.fit()
        expected = _hajek_contrast(trial["y"].to_numpy(), trial["d"].to_numpy(), w)
        assert fit.estimate == pytest.approx(expected, abs=1e-12)

    @pytest.mark.parametrize("method", ["ipw", "aipw", "calibration"])
    def test_target_equal_to_the_trial_gives_the_trial_estimate(self, method):
        trial, _ = _trial_and_target()
        target = trial[["age", "edu"]].copy()
        y, d = trial["y"].to_numpy(), trial["d"].to_numpy()
        fit = sp.pate(
            trial, target, "y", "d", ["age", "edu"], method=method, n_boot=10, seed=0
        )
        if method == "aipw":
            # the interacted regression's average effect on the trial rows
            X = trial[["age", "edu"]].to_numpy()
            D = np.column_stack([np.ones(len(y)), X, d, d[:, None] * X])
            b = np.linalg.lstsq(D, y, rcond=None)[0]
            expected = b[3] + X.mean(0) @ b[4:]
        else:
            # no covariate shift: constant weights, the difference in means
            expected = y[d == 1].mean() - y[d == 0].mean()
        # L-BFGS leaves slopes of order 1e-7 instead of exactly zero.
        assert fit.estimate == pytest.approx(expected, abs=1e-5)

    def test_invalid_inputs_are_refused(self):
        trial, target = _trial_and_target()
        with pytest.raises(ValueError, match="method"):
            sp.pate(trial, target, "y", "d", ["age"], method="matching")
        with pytest.raises(ValueError, match="data_target"):
            sp.pate(trial, target.drop(columns="edu"), "y", "d", ["age", "edu"])
        with pytest.raises(ValueError, match="data_experiment"):
            sp.pate(trial.drop(columns="y"), target, "y", "d", ["age"])
        with pytest.raises(ValueError, match="binary"):
            sp.pate(trial.assign(d=trial["d"] + 1), target, "y", "d", ["age"])

    @pytest.mark.parametrize("column", ["age", "y"])
    def test_missing_values_are_dropped_or_refused(self, column):
        trial, target = _trial_and_target()
        holes = trial.copy()
        holes.loc[holes.index[0], column] = np.nan
        kw = dict(n_boot=10, seed=0)
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                fit = sp.pate(holes, target, "y", "d", ["age", "edu"], **kw)
        except ValueError:
            return
        clean = sp.pate(
            holes.dropna().reset_index(drop=True),
            target,
            "y",
            "d",
            ["age", "edu"],
            **kw,
        )
        assert fit.estimate == pytest.approx(clean.estimate, abs=1e-10)

    def test_single_arm_trial_is_refused(self):
        trial, target = _trial_and_target()
        with pytest.raises(ValueError):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                sp.pate(trial.assign(d=1.0), target, "y", "d", ["age"], n_boot=10)

    def test_trim_of_one_half_or_more_is_refused(self):
        trial, target = _trial_and_target()
        with pytest.raises(ValueError):
            sp.pate(trial, target, "y", "d", ["age"], n_boot=10, trim=0.7)


# --------------------------------------------------------------------- #
#  behaviour added with the fixes of this scan
# --------------------------------------------------------------------- #


class TestIpwDegenerateFits:
    def _quasi_separated(self):
        rng = np.random.default_rng(5)
        n = 400
        x = rng.normal(size=n)
        c = (rng.uniform(size=n) < 0.15).astype(float)
        d = (rng.uniform(size=n) < 1 / (1 + np.exp(-0.5 * x))).astype(float)
        d[c == 1] = 1.0  # the category holds treated units only
        y = 1 + 2 * d + x + rng.normal(size=n)
        return pd.DataFrame({"y": y, "d": d, "x": x, "c": c, "w": rng.uniform(1, 2, n)})

    @pytest.mark.parametrize(
        "kwargs", [{}, {"weights": "w"}, {"ps_model": "probit"}, {"trim": 0.05}]
    )
    def test_quasi_complete_separation_warns_and_is_recorded(self, kwargs):
        df = self._quasi_separated()
        with pytest.warns(ConvergenceWarning, match="quasi-complete"):
            fit = sp.ipw(df, "y", "d", ["x", "c"], se_method="sandwich", **kwargs)
        assert fit.model_info["separation"] == "quasi-complete"

    def test_ordinary_fit_records_no_separation_and_no_failed_replicates(self):
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            fit = sp.ipw(_obs(n=200), "y", "d", ["x", "z"], n_bootstrap=20, seed=1)
        assert "separation" not in fit.model_info
        assert "n_bootstrap_failed" not in fit.model_info

    def test_complete_separation_is_refused_whatever_the_options(self):
        rng = np.random.default_rng(3)
        x = rng.normal(size=120)
        d = (x > 0).astype(float)
        df = pd.DataFrame({"y": 1 + 2 * d + x + rng.normal(size=120), "d": d, "x": x})
        for kwargs in ({}, {"trim": 0.1}, {"ps_model": "probit"}, {"estimand": "ATT"}):
            with pytest.raises(DataInsufficient, match="separate"):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    sp.ipw(df, "y", "d", ["x"], se_method="sandwich", **kwargs)
        # a known design probability involves no fitted model
        fit = sp.ipw(df, "y", "d", ["x"], se_method="sandwich", propensity=0.5)
        assert np.isfinite(fit.se)

    def test_single_arm_resamples_are_excluded_and_counted(self):
        # The replicates are those of the seeded generator that hold both
        # arms; with a known propensity each is a closed form.
        rng = np.random.default_rng(3)
        n = 40
        x = rng.normal(size=n)
        d = np.zeros(n)
        d[:2] = 1.0
        y = 1 + x + rng.normal(size=n)
        df = pd.DataFrame({"y": y, "d": d, "x": x})
        rs = np.random.RandomState(0)
        draws, failed = [], 0
        for _ in range(100):
            idx = rs.choice(n, size=n, replace=True)
            if d[idx].min() == d[idx].max():
                failed += 1
                continue
            draws.append(y[idx][d[idx] == 1].mean() - y[idx][d[idx] == 0].mean())
        assert failed > 0
        with pytest.warns(RuntimeWarning, match="single treatment arm"):
            fit = sp.ipw(df, "y", "d", ["x"], propensity=0.3, n_bootstrap=100, seed=0)
        assert fit.model_info["n_bootstrap_failed"] == failed
        assert fit.se == pytest.approx(np.std(draws, ddof=1), rel=1e-10)


class TestAipwConstantCovariates:
    def test_covariate_constant_in_the_sample_changes_nothing(self):
        # collinear with the intercept, so the fit is that without it
        df = _obs(n=300)
        base = sp.aipw(df, "y", "d", ["x", "z"], seed=2)
        for value in (1.0, 0.0, 3.5):
            with warnings.catch_warnings():
                warnings.simplefilter("error", ConvergenceWarning)
                fit = sp.aipw(df.assign(k=value), "y", "d", ["x", "z", "k"], seed=2)
            assert fit.estimate == pytest.approx(base.estimate, abs=1e-8)
            assert fit.se == pytest.approx(base.se, abs=1e-8)

    def test_two_constant_covariates(self):
        df = _obs(n=300)
        base = sp.aipw(df, "y", "d", ["x", "z"], seed=2)
        with warnings.catch_warnings():
            warnings.simplefilter("error", ConvergenceWarning)
            fit = sp.aipw(
                df.assign(k=1.0, m=2.0), "y", "d", ["x", "z", "k", "m"], seed=2
            )
        assert fit.estimate == pytest.approx(base.estimate, abs=1e-8)


class TestGComputationDegenerate:
    def test_constant_covariate_changes_nothing(self):
        df = _obs(n=200)
        base = sp.g_computation(df, "y", "d", ["x", "z"], n_boot=20, seed=3)
        fit = sp.g_computation(
            df.assign(k=1.0), "y", "d", ["x", "z", "k"], n_boot=20, seed=3
        )
        assert fit.estimate == pytest.approx(base.estimate, abs=1e-9)
        assert fit.se == pytest.approx(base.se, abs=1e-9)

    def test_ate_resamples_with_one_arm_are_failed_replicates(self):
        rng = np.random.default_rng(3)
        n = 30
        x = rng.normal(size=n)
        d = np.zeros(n)
        d[0] = 1.0
        df = pd.DataFrame({"y": 1 + x + rng.normal(size=n), "d": d, "x": x})
        draw = np.random.default_rng(1)
        usable = sum(bool(d[draw.integers(0, n, size=n)].sum() > 0) for _ in range(40))
        with pytest.warns(RuntimeWarning, match="single treatment arm"):
            fit = sp.g_computation(df, "y", "d", ["x"], n_boot=40, seed=1)
        assert fit.model_info["n_boot_success"] == usable
        assert fit.model_info["n_boot_failed"] == 40 - usable


class TestPateDroppedRows:
    def test_dropped_rows_are_reported(self):
        trial, target = _trial_and_target()
        trial.loc[trial.index[:3], "edu"] = np.nan
        target.loc[target.index[:2], "age"] = np.nan
        with pytest.warns(UserWarning, match="dropped 3 of 400 .* 2 of 600"):
            fit = sp.pate(trial, target, "y", "d", ["age", "edu"], n_boot=10, seed=0)
        assert fit.n_obs == 397
        assert fit.model_info["n_target"] == 598


class TestNumpyScalarOptions:
    @pytest.mark.parametrize("trim", [np.float32(0.25), np.float64(0.25)])
    def test_ipw_takes_a_numpy_trim(self, trim):
        df = _obs(n=200)
        kw = dict(y="y", treat="d", covariates=["x"], se_method="sandwich")
        got = sp.ipw(df, trim=trim, **kw)
        ref = sp.ipw(df, trim=0.25, **kw)
        assert got.estimate == ref.estimate and got.se == ref.se

    def test_ipw_takes_a_numpy_integer_zero(self):
        df = _obs(n=200)
        kw = dict(y="y", treat="d", covariates=["x"], se_method="sandwich")
        assert sp.ipw(df, trim=np.int64(0), **kw).estimate == (
            sp.ipw(df, trim=0, **kw).estimate
        )

    def test_ipw_refuses_a_boolean_trim(self):
        with pytest.raises(MethodIncompatibility, match="trim"):
            sp.ipw(_obs(n=200), y="y", treat="d", covariates=["x"], trim=True)

    def test_pate_takes_a_numpy_trim(self):
        trial, target = _trial_and_target()
        kw = dict(n_boot=4, seed=0)
        got = sp.pate(trial, target, "y", "d", ["age", "edu"], trim=np.int64(0), **kw)
        ref = sp.pate(trial, target, "y", "d", ["age", "edu"], trim=0.0, **kw)
        assert got.estimate == ref.estimate
        f32 = sp.pate(
            trial, target, "y", "d", ["age", "edu"], trim=np.float32(0.25), **kw
        )
        f64 = sp.pate(trial, target, "y", "d", ["age", "edu"], trim=0.25, **kw)
        assert f32.estimate == f64.estimate
