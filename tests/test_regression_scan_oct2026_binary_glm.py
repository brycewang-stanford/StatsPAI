"""Branch tests for ``statspai.regression.logit_probit`` and
``statspai.regression.glm`` written while scanning their uncovered lines
for defects (October 2026).

Numerical assertions compare with likelihood algebra written here in numpy
(score, information, sandwich) or with identities: frequency weights equal
duplicated rows, ``offset=log(e)`` equals ``exposure=e``, grouped binomial
counts equal the ungrouped 0/1 rows. Tests marked ``xfail(strict=True)``
assert the correct behaviour of a defect that is not fixed yet.
"""

import signal
import warnings
from contextlib import contextmanager

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from scipy.special import expit

import statspai as sp
from statspai.exceptions import MethodIncompatibility, StatsPAIError

# Both sides solve the same smooth likelihood to a 1e-8 change in the
# log-likelihood / deviance, which leaves the coefficients good to about
# 1e-6 relative; standard errors inherit that.
RTOL = 1e-5


def _data(seed: int = 2, n: int = 300) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
            "g": rng.integers(0, 12, n),
            "w": rng.integers(1, 4, n).astype(float),
            "e": rng.uniform(0.5, 3.0, n),
        }
    )
    df["b"] = (0.3 + 0.8 * df.x1 - 0.5 * df.x2 + rng.logistic(size=n) > 0).astype(float)
    df["loge"] = np.log(df.e)
    df["cnt"] = rng.poisson(np.exp(0.3 + 0.5 * df.x1 - 0.3 * df.x2) * df.e).astype(
        float
    )
    df["pos"] = np.exp(0.5 + 0.3 * df.x1 + 0.4 * rng.normal(size=n))
    return df


def _dup(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[df.index.repeat(df.w.astype(int))].reset_index(drop=True)


def _X(df: pd.DataFrame) -> np.ndarray:
    return np.column_stack([np.ones(len(df)), df.x1, df.x2])


def _quiet(fn, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*args, **kwargs)


@contextmanager
def _deadline(seconds: float):
    """Fail instead of hanging: one defect below is an infinite loop."""
    if not hasattr(signal, "setitimer"):  # pragma: no cover - Windows
        pytest.skip("needs POSIX interval timers")

    def _raise(*_):
        raise TimeoutError(f"no result after {seconds} s")

    old = signal.signal(signal.SIGALRM, _raise)
    signal.setitimer(signal.ITIMER_REAL, seconds)
    try:
        yield
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, old)


# ===================================================================== #
#  logit / probit / cloglog
# ===================================================================== #


class TestLogitClosedForm:
    def test_score_information_and_sandwiches(self):
        df = _data()
        n, X, y = len(df), _X(df), df.b.to_numpy()
        res = sp.logit("b ~ x1 + x2", df)
        beta = res.params.to_numpy()
        p = expit(X @ beta)
        # first-order condition of the logit likelihood, relative to the
        # scale of the score terms (which are O(n))
        assert np.abs(X.T @ (y - p)).max() < 1e-6 * n
        bread = np.linalg.inv(X.T @ (X * (p * (1 - p))[:, None]))
        np.testing.assert_allclose(res.std_errors, np.sqrt(np.diag(bread)), rtol=RTOL)
        s = X * (y - p)[:, None]
        hc0 = bread @ (s.T @ s) @ bread
        # Stata's ML conventions: vce(robust) carries N/(N-1), hc1 N/(N-K),
        # vce(cluster) G/(G-1) only
        for kind, factor in (
            ("hc0", 1.0),
            ("robust", n / (n - 1)),
            ("hc1", n / (n - 3)),
        ):
            got = sp.logit("b ~ x1 + x2", df, robust=kind).std_errors
            np.testing.assert_allclose(got, np.sqrt(np.diag(hc0) * factor), rtol=RTOL)
        sums = pd.DataFrame(s).groupby(df.g.to_numpy()).sum().to_numpy()
        G = sums.shape[0]
        cl = bread @ (sums.T @ sums) @ bread * G / (G - 1)
        got = _quiet(sp.logit, "b ~ x1 + x2", df, cluster="g")
        np.testing.assert_allclose(got.std_errors, np.sqrt(np.diag(cl)), rtol=RTOL)
        assert got.model_info["n_clusters"] == G

    @pytest.mark.parametrize("fn", [sp.logit, sp.probit, sp.cloglog])
    def test_frequency_weights_equal_duplicated_rows(self, fn):
        df = _data()
        a = _quiet(fn, "b ~ x1 + x2", df, weights="w")
        b = _quiet(fn, "b ~ x1 + x2", _dup(df))
        np.testing.assert_allclose(a.params, b.params, rtol=RTOL)
        np.testing.assert_allclose(a.std_errors, b.std_errors, rtol=RTOL)
        assert a.model_info["ll"] == pytest.approx(b.model_info["ll"], rel=1e-8)
        assert a.model_info["weights"] == "w"

    def test_weighted_cluster_equals_duplicated_rows(self):
        df = _data()
        a = _quiet(sp.logit, "b ~ x1 + x2", df, weights="w", cluster="g")
        b = _quiet(sp.logit, "b ~ x1 + x2", _dup(df), cluster="g")
        # cluster sums of w * score equal cluster sums over duplicated rows
        np.testing.assert_allclose(a.std_errors, b.std_errors, rtol=RTOL)

    def test_yx_path_follows_the_complete_rows(self):
        df = _data()
        df.loc[[3, 9], "x1"] = np.nan
        a = _quiet(sp.logit, data=df, y="b", x=["x1", "x2"], weights="w", cluster="g")
        b = _quiet(sp.logit, "b ~ x1 + x2", df.dropna(), weights="w", cluster="g")
        assert a.data_info["nobs"] == 298
        np.testing.assert_allclose(a.params.to_numpy(), b.params.to_numpy(), rtol=RTOL)
        np.testing.assert_allclose(
            a.std_errors.to_numpy(), b.std_errors.to_numpy(), rtol=RTOL
        )


class TestLogitReporting:
    def test_marginal_effects_are_density_times_coefficient(self):
        df = _data()
        X = _X(df)
        res = sp.logit("b ~ x1 + x2", df, marginal_effects="average")
        beta = res.params.to_numpy()
        p = expit(X @ beta)
        ame = res.model_info["marginal_effects"]["dy/dx"].to_numpy()
        np.testing.assert_allclose(ame, np.mean(p * (1 - p)) * beta, rtol=1e-10)

        pbar = expit(X.mean(axis=0) @ beta)
        mem = sp.logit("b ~ x1 + x2", df, marginal_effects="mean")
        np.testing.assert_allclose(
            mem.model_info["marginal_effects"]["dy/dx"], pbar * (1 - pbar) * beta, 1e-10
        )
        at_default = sp.logit("b ~ x1 + x2", df, marginal_effects="at")
        np.testing.assert_allclose(
            at_default.model_info["marginal_effects"]["dy/dx"],
            pbar * (1 - pbar) * beta,
            rtol=1e-10,
        )
        xr = X.mean(axis=0).copy()
        xr[1] = 2.0
        pr = expit(xr @ beta)
        at = sp.logit("b ~ x1 + x2", df, marginal_effects="at", at_values={"x1": 2.0})
        np.testing.assert_allclose(
            at.model_info["marginal_effects"]["dy/dx"], pr * (1 - pr) * beta, rtol=1e-10
        )

    def test_probit_marginal_effect_uses_the_normal_density(self):
        df = _data()
        res = sp.probit("b ~ x1 + x2", df, marginal_effects="average")
        beta = res.params.to_numpy()
        expected = np.mean(stats.norm.pdf(_X(df) @ beta)) * beta
        np.testing.assert_allclose(
            res.model_info["marginal_effects"]["dy/dx"], expected, rtol=1e-10
        )

    def test_odds_ratios_and_alpha(self):
        df = _data()
        res = sp.logit("b ~ x1 + x2", df, odds_ratio=True, alpha=0.10)
        table = res.model_info["odds_ratio"]
        beta, se = res.params.to_numpy(), res.std_errors.to_numpy()
        z = stats.norm.ppf(0.95)
        np.testing.assert_allclose(table["OR"], np.exp(beta), rtol=1e-12)
        np.testing.assert_allclose(table["Std. Err."], np.exp(beta) * se, rtol=1e-12)
        np.testing.assert_allclose(table.iloc[:, 2], np.exp(beta - z * se), rtol=1e-12)
        np.testing.assert_allclose(table.iloc[:, 3], np.exp(beta + z * se), rtol=1e-12)
        ci = res.conf_int().to_numpy()
        np.testing.assert_allclose(ci[:, 0], beta - z * se, rtol=1e-8)
        with pytest.raises(TypeError, match="odds_ratio"):
            sp.probit("b ~ x1 + x2", df, odds_ratio=True)

    def test_predict_types_and_delta_method_interval(self):
        df = _data()
        res = sp.logit("b ~ x1 + x2", df)
        beta = res.params.to_numpy()
        new = df.head(4)
        xb = _X(new) @ beta
        np.testing.assert_allclose(res.predict(new), expit(xb), rtol=1e-12)
        np.testing.assert_allclose(res.predict(new, pred_type="link"), xb, rtol=1e-12)
        np.testing.assert_allclose(res.predict(new, what="link"), xb, rtol=1e-12)
        np.testing.assert_array_equal(
            res.predict(new, pred_type="class", cutoff=0.7), (expit(xb) >= 0.7) * 1.0
        )
        np.testing.assert_allclose(res.predict(_X(new)), expit(xb), rtol=1e-12)
        V = np.asarray(res.data_info["var_cov"])
        se_link = np.sqrt(np.einsum("ij,jk,ik->i", _X(new), V, _X(new)))
        z = stats.norm.ppf(0.95)
        frame = res.predict(data=new, what="confidence", alpha=0.10)
        p = expit(xb)
        # the density is a central difference with step 1e-6
        np.testing.assert_allclose(frame["se"], p * (1 - p) * se_link, rtol=1e-6)
        np.testing.assert_allclose(frame["lower"], expit(xb - z * se_link), rtol=1e-10)
        np.testing.assert_allclose(frame["upper"], expit(xb + z * se_link), rtol=1e-10)

    def test_predict_validation(self):
        df = _data()
        res = sp.logit("b ~ x1 + x2", df)
        with pytest.raises(ValueError, match="Unknown predict type"):
            res.predict(df.head(), pred_type="zz")
        with pytest.raises(MethodIncompatibility, match="not available for a binary"):
            res.predict(df.head(), what="prediction")
        with pytest.raises(MethodIncompatibility, match="has shape"):
            res.predict(np.ones((3, 2)))
        with pytest.raises(MethodIncompatibility, match="'x2' is not in the data"):
            res.predict(df[["x1"]].head())

    def test_classification_table_at_another_cutoff(self):
        df = _data()
        res = sp.logit("b ~ x1 + x2", df)
        p = expit(_X(df) @ res.params.to_numpy())
        y = df.b.to_numpy()
        table = res.classification_table(cutoff=0.3)
        assert table["tp"] == int(((p >= 0.3) & (y == 1)).sum())
        assert table["fp"] == int(((p >= 0.3) & (y == 0)).sum())
        assert table["sensitivity"] == pytest.approx(table["tp"] / (y == 1).sum())
        assert res.classification_table()["cutoff"] == 0.5


class TestLogitDegenerateInputs:
    def test_perfect_predictor_is_dropped_with_its_rows(self):
        df = _data()
        df["d"] = (np.arange(len(df)) < 8).astype(float)
        df.loc[df.d == 1, "b"] = 1.0
        with pytest.warns(UserWarning, match="predicts success perfectly"):
            a = sp.logit("b ~ x1 + d", df, weights="w", cluster="g")
        b = _quiet(sp.logit, "b ~ x1", df[df.d == 0], weights="w", cluster="g")
        assert a.data_info["nobs"] == len(df) - 8
        assert a.model_info["perfect_prediction_omitted"] == ["d"]
        np.testing.assert_allclose(a.params.to_numpy(), b.params.to_numpy(), rtol=RTOL)
        # weights and cluster keys follow the rows that were kept
        np.testing.assert_allclose(
            a.std_errors.to_numpy(), b.std_errors.to_numpy(), rtol=RTOL
        )

    def test_keeping_a_perfect_predictor_leaves_the_other_terms_alone(self):
        df = _data()
        df["d"] = (np.arange(len(df)) < 8).astype(float)
        df.loc[df.d == 1, "b"] = 1.0
        kept = _quiet(sp.logit, "b ~ x1 + d", df, perfect_prediction="keep")
        dropped = _quiet(sp.logit, "b ~ x1 + d", df)
        assert "d" in kept.params.index
        assert kept.data_info["nobs"] == len(df)
        # as the coefficient on d diverges its rows stop contributing to the
        # score of the other terms, so those converge to the fit without
        # them; the iteration stops once the likelihood gain is below 1e-8,
        # which leaves them within about 1e-6 of that limit
        np.testing.assert_allclose(
            kept.params[["Intercept", "x1"]], dropped.params, rtol=1e-4
        )
        with pytest.raises(MethodIncompatibility, match="'drop' or 'keep'"):
            sp.logit("b ~ x1 + d", df, perfect_prediction="zz")

    def test_input_validation(self):
        df = _data()
        with pytest.raises(ValueError, match="must be binary"):
            sp.logit("y2 ~ x1", df.assign(y2=df.b * 2))
        with pytest.raises(ValueError, match="Provide either"):
            sp.logit()
        with pytest.raises(MethodIncompatibility, match="not available"):
            sp.logit("b ~ x1", df, robust="hc2")

    def test_nonconvergence_is_announced(self):
        df = _data()
        with pytest.warns(UserWarning, match="did not converge after 1 iter"):
            sp.logit("b ~ x1 + x2", df, maxiter=1)


# ===================================================================== #
#  glm
# ===================================================================== #


class TestGlmClosedForm:
    def test_poisson_score_information_and_sandwiches(self):
        df = _data()
        n, X, y = len(df), _X(df), df.cnt.to_numpy()
        res = sp.glm("cnt ~ x1 + x2", df, family="poisson")
        beta = res.params.to_numpy()
        mu = np.exp(X @ beta)
        assert np.abs(X.T @ (y - mu)).max() < 1e-6 * n
        bread = np.linalg.inv(X.T @ (X * mu[:, None]))
        np.testing.assert_allclose(res.std_errors, np.sqrt(np.diag(bread)), rtol=RTOL)
        s = X * (y - mu)[:, None]
        hc0 = bread @ (s.T @ s) @ bread
        np.testing.assert_allclose(
            sp.glm("cnt ~ x1 + x2", df, family="poisson", robust="hc0").std_errors,
            np.sqrt(np.diag(hc0)),
            rtol=RTOL,
        )
        np.testing.assert_allclose(
            sp.glm("cnt ~ x1 + x2", df, family="poisson", robust="hc1").std_errors,
            np.sqrt(np.diag(hc0) * n / (n - 3)),
            rtol=RTOL,
        )
        # HC2 / HC3 divide each squared score by (1 - h) and (1 - h)^2,
        # with h the leverage of the IRLS-weighted design
        h = mu * np.einsum("ij,jk,ik->i", X, bread, X)
        for kind, power in (("hc2", 1), ("hc3", 2)):
            meat = (s / ((1 - h) ** (power / 2))[:, None]).T @ (
                s / ((1 - h) ** (power / 2))[:, None]
            )
            np.testing.assert_allclose(
                sp.glm("cnt ~ x1 + x2", df, family="poisson", robust=kind).std_errors,
                np.sqrt(np.diag(bread @ meat @ bread)),
                rtol=RTOL,
            )
        sums = pd.DataFrame(s).groupby(df.g.to_numpy()).sum().to_numpy()
        G = sums.shape[0]
        cl = bread @ (sums.T @ sums) @ bread * G / (G - 1)  # Stata glm: G/(G-1)
        got = _quiet(sp.glm, "cnt ~ x1 + x2", df, family="poisson", cluster="g")
        np.testing.assert_allclose(got.std_errors, np.sqrt(np.diag(cl)), rtol=RTOL)

    def test_gaussian_identity_is_ols(self):
        df = _data()
        a = sp.glm("pos ~ x1 + x2", df, family="gaussian")
        b = sp.regress("pos ~ x1 + x2", df)
        np.testing.assert_allclose(a.params, b.params, rtol=1e-8)
        np.testing.assert_allclose(a.std_errors, b.std_errors, rtol=1e-8)

    def test_binomial_links_agree_with_the_dedicated_estimators(self):
        df = _data()
        for link, fn in (("logit", sp.logit), ("probit", sp.probit)):
            a = sp.glm("b ~ x1 + x2", df, family="binomial", link=link)
            b = fn("b ~ x1 + x2", df)
            np.testing.assert_allclose(a.params, b.params, rtol=RTOL)
            # both report the observed information (Stata's vce(oim))
            np.testing.assert_allclose(a.std_errors, b.std_errors, rtol=1e-4)

    @pytest.mark.parametrize("family, yvar", [("poisson", "cnt"), ("binomial", "b")])
    def test_frequency_weights_equal_duplicated_rows(self, family, yvar):
        df = _data()
        a = sp.glm(f"{yvar} ~ x1 + x2", df, family=family, weights="w")
        b = sp.glm(f"{yvar} ~ x1 + x2", _dup(df), family=family)
        np.testing.assert_allclose(a.params, b.params, rtol=RTOL)
        np.testing.assert_allclose(a.std_errors, b.std_errors, rtol=RTOL)

    def test_weighted_cluster_equals_duplicated_rows(self):
        df = _data()
        kw = dict(family="poisson", cluster="g")
        a = _quiet(sp.glm, "cnt ~ x1 + x2", df, weights="w", **kw)
        b = _quiet(sp.glm, "cnt ~ x1 + x2", _dup(df), **kw)
        np.testing.assert_allclose(a.std_errors, b.std_errors, rtol=RTOL)

    def test_grouped_binomial_equals_the_ungrouped_rows(self):
        rng = np.random.default_rng(7)
        x = np.repeat(np.linspace(-1, 1, 12), 1)
        trials = rng.integers(3, 9, 12)
        succ = rng.binomial(trials, expit(0.4 + 0.9 * x))
        grouped = pd.DataFrame({"x": x, "s": succ, "f": trials - succ})
        rows = [(xi, 1.0) for xi, si in zip(x, succ) for _ in range(si)] + [
            (xi, 0.0) for xi, fi in zip(x, trials - succ) for _ in range(fi)
        ]
        long = pd.DataFrame(rows, columns=["x", "y"])
        a = sp.glm("cbind(s, f) ~ x", grouped, family="binomial")
        b = sp.glm("y ~ x", long, family="binomial")
        np.testing.assert_allclose(a.params.to_numpy(), b.params.to_numpy(), rtol=RTOL)
        np.testing.assert_allclose(
            a.std_errors.to_numpy(), b.std_errors.to_numpy(), rtol=RTOL
        )


class TestGlmOffsetAndScale:
    def test_exposure_is_a_log_offset_and_the_two_add(self):
        df = _data()
        kw = dict(family="poisson")
        a = sp.glm("cnt ~ x1 + x2", df, offset="loge", **kw)
        b = sp.glm("cnt ~ x1 + x2", df, exposure="e", **kw)
        np.testing.assert_allclose(a.params, b.params, rtol=1e-10)
        np.testing.assert_allclose(a.std_errors, b.std_errors, rtol=1e-10)
        # the offset enters the mean with coefficient one
        X, y = _X(df), df.cnt.to_numpy()
        mu = np.exp(X @ a.params.to_numpy() + df.loge.to_numpy())
        assert np.abs(X.T @ (y - mu)).max() < 1e-6 * len(df)
        both = sp.glm("cnt ~ x1 + x2", df, offset="loge", exposure="e", **kw)
        twice = sp.glm("cnt ~ x1 + x2", df.assign(l2=2 * df.loge), offset="l2", **kw)
        np.testing.assert_allclose(both.params, twice.params, rtol=1e-10)

    def test_predict_on_new_data_keeps_the_offset(self):
        df = _data()
        res = sp.glm("cnt ~ x1 + x2", df, family="poisson", exposure="e")
        new = df.head(5)
        expected = np.exp(_X(new) @ res.params.to_numpy()) * new.e.to_numpy()
        np.testing.assert_allclose(np.asarray(res.predict(new)), expected, rtol=1e-10)
        np.testing.assert_allclose(
            np.asarray(res.fitted_values())[:5], expected, rtol=1e-10
        )

    def test_scale_rescales_the_model_based_covariance(self):
        df = _data()
        n, X, y = len(df), _X(df), df.cnt.to_numpy()
        base = sp.glm("cnt ~ x1 + x2", df, family="poisson")
        mu = np.exp(X @ base.params.to_numpy())
        x2 = np.sum((y - mu) ** 2 / mu) / (n - 3)
        dev = (
            2
            * np.sum(
                np.where(y > 0, y * np.log(np.where(y > 0, y, 1) / mu), 0) - (y - mu)
            )
            / (n - 3)
        )
        for scale, phi in (("x2", x2), ("dev", dev), (2.0, 2.0)):
            got = sp.glm("cnt ~ x1 + x2", df, family="poisson", scale=scale)
            np.testing.assert_allclose(
                got.std_errors, base.std_errors * np.sqrt(phi), rtol=RTOL
            )
            np.testing.assert_allclose(got.params, base.params, rtol=1e-12)
        quasi = sp.glm("cnt ~ x1 + x2", df, family="quasipoisson")
        np.testing.assert_allclose(
            quasi.std_errors, base.std_errors * np.sqrt(x2), rtol=RTOL
        )

    def test_scale_validation(self):
        df = _data()
        kw = dict(family="poisson")
        with pytest.raises(MethodIncompatibility, match="does not use it"):
            sp.glm("cnt ~ x1", df, scale="x2", robust="hc0", **kw)
        with pytest.raises(MethodIncompatibility, match="not 'x2', 'dev'"):
            sp.glm("cnt ~ x1", df, scale="zz", **kw)
        with pytest.raises(MethodIncompatibility, match="must be positive"):
            sp.glm("cnt ~ x1", df, scale=-1.0, **kw)

    def test_expected_information_with_pearson_scale_for_gamma_log(self):
        # log-link gamma: IRLS weight is 1, so the expected-information
        # covariance is phi * (X'X)^-1 with phi the Pearson dispersion
        df = _data()
        n, X, y = len(df), _X(df), df.pos.to_numpy()
        kw = dict(family="gamma", link="log")
        res = sp.glm("pos ~ x1 + x2", df, information="expected", scale="x2", **kw)
        mu = np.exp(X @ res.params.to_numpy())
        phi = np.sum((y - mu) ** 2 / mu**2) / (n - 3)
        se = np.sqrt(np.diag(phi * np.linalg.inv(X.T @ X)))
        np.testing.assert_allclose(res.std_errors, se, rtol=RTOL)
        observed = sp.glm("pos ~ x1 + x2", df, scale="x2", **kw)
        # observed information adds X' diag((y - mu)/mu) X / phi to the bread
        info = X.T @ (X * (y / mu)[:, None])
        se_obs = np.sqrt(np.diag(phi * np.linalg.inv(info)))
        np.testing.assert_allclose(observed.std_errors, se_obs, rtol=RTOL)
        with pytest.raises(MethodIncompatibility, match="'observed' or 'expected'"):
            sp.glm("pos ~ x1 + x2", df, information="zz", **kw)


class TestGlmValidation:
    def test_auxiliary_columns_are_checked(self):
        df = _data()
        kw = dict(family="poisson")
        f = "cnt ~ x1 + x2"
        with pytest.raises(MethodIncompatibility, match="weights contains missing"):
            sp.glm(
                f, df.assign(w=np.r_[np.nan, df.w.to_numpy()[1:]]), weights="w", **kw
            )
        with pytest.raises(MethodIncompatibility, match="non-negative"):
            sp.glm(f, df.assign(w=-df.w), weights="w", **kw)
        with pytest.raises(MethodIncompatibility, match="at least one positive"):
            sp.glm(f, df.assign(w=0.0), weights="w", **kw)
        with pytest.raises(MethodIncompatibility, match="offset contains missing"):
            sp.glm(
                f,
                df.assign(loge=np.r_[np.nan, df.loge.to_numpy()[1:]]),
                offset="loge",
                **kw,
            )
        with pytest.raises(MethodIncompatibility, match="strictly positive"):
            sp.glm(f, df.assign(e=-1.0), exposure="e", **kw)
        with pytest.raises(MethodIncompatibility, match="at least two clusters"):
            sp.glm(f, df.assign(g=1), cluster="g", **kw)

    def test_option_validation(self):
        df = _data()
        f = "cnt ~ x1 + x2"
        with pytest.raises(MethodIncompatibility, match="Unknown family"):
            sp.glm(f, df, family="zz")
        with pytest.raises(MethodIncompatibility, match="Unknown link"):
            sp.glm(f, df, family="poisson", link="zz")
        with pytest.raises(MethodIncompatibility, match="alpha"):
            sp.glm(f, df, family="poisson", alpha=2)
        with pytest.raises(MethodIncompatibility, match="maxiter"):
            sp.glm(f, df, family="poisson", maxiter=0)
        with pytest.raises(MethodIncompatibility, match="tol"):
            sp.glm(f, df, family="poisson", tol=-1)

    def test_missing_cluster_label_drops_the_row(self):
        df = _data()
        df["g"] = df.g.astype(float)
        df.loc[[3, 30], "g"] = np.nan
        kw = dict(family="poisson", cluster="g")
        a = _quiet(sp.glm, "cnt ~ x1 + x2", df, **kw)
        b = _quiet(sp.glm, "cnt ~ x1 + x2", df.dropna(), **kw)
        np.testing.assert_allclose(a.std_errors, b.std_errors, rtol=1e-10)

    def test_yx_path_and_alpha(self):
        df = _data()
        kw = dict(family="poisson", offset="loge", weights="w")
        a = sp.glm(data=df, y="cnt", x=["x1", "x2"], **kw)
        b = sp.glm("cnt ~ x1 + x2", df, **kw)
        np.testing.assert_allclose(a.params.to_numpy(), b.params.to_numpy(), 1e-10)
        one = sp.glm(data=df, y="cnt", x="x1", family="poisson")
        assert len(one.params) == 2
        z = stats.norm.ppf(0.95)
        res = sp.glm("cnt ~ x1 + x2", df, family="poisson", alpha=0.10)
        ci = res.conf_int().to_numpy()
        np.testing.assert_allclose(
            ci[:, 1] - ci[:, 0], 2 * z * res.std_errors.to_numpy(), rtol=1e-8
        )


# --------------------------------------------------------------------- #
#  Defects (assert the correct behaviour; remove the marker with the fix)
# --------------------------------------------------------------------- #


def test_glm_hac_is_newey_west_on_the_scores():
    df = _data()
    n, X, y = len(df), _X(df), df.cnt.to_numpy()
    res = sp.glm("cnt ~ x1 + x2", df, family="poisson", robust="hac")
    mu = np.exp(X @ res.params.to_numpy())
    bread = np.linalg.inv(X.T @ (X * mu[:, None]))
    s = X * (y - mu)[:, None]
    lags = int(np.floor(4 * (n / 100) ** (2 / 9)))
    S = s.T @ s
    for j in range(1, lags + 1):
        gj = s[j:].T @ s[:-j]
        S += (1 - j / (lags + 1)) * (gj + gj.T)
    expected = np.sqrt(np.diag(bread @ S @ bread))
    np.testing.assert_allclose(res.std_errors, expected, rtol=RTOL)


@pytest.mark.parametrize(
    "family, shift, scale",
    [("poisson", -3.0, 1.0), ("binomial", 0.0, 2.0), ("negative_binomial", -3.0, 1.0)],
)
def test_glm_refuses_an_outcome_outside_the_support(family, shift, scale):
    df = _data()
    base = df.b if family == "binomial" else df.cnt
    with pytest.raises(StatsPAIError):
        _quiet(sp.glm, "yy ~ x1", df.assign(yy=base * scale + shift), family=family)


@pytest.mark.xfail(
    strict=True,
    reason="the default dispersion of the gamma / inverse-gaussian families "
    "is deviance / df; Stata's glm (scale(x2) for continuous families) and "
    "R's summary.glm both use the Pearson statistic, so default standard "
    "errors differ from both references (4% on this sample).",
)
def test_glm_gamma_default_dispersion_is_pearson():
    df = _data()
    n, X, y = len(df), _X(df), df.pos.to_numpy()
    res = sp.glm(
        "pos ~ x1 + x2", df, family="gamma", link="log", information="expected"
    )
    mu = np.exp(X @ res.params.to_numpy())
    phi = np.sum((y - mu) ** 2 / mu**2) / (n - 3)
    expected = np.sqrt(np.diag(phi * np.linalg.inv(X.T @ X)))
    np.testing.assert_allclose(res.std_errors, expected, rtol=RTOL)


@pytest.mark.parametrize("kind", ["average", "mean"])
def test_logit_marginal_effects_use_the_weights(kind):
    df = _data()
    a = _quiet(sp.logit, "b ~ x1 + x2", df, weights="w", marginal_effects=kind)
    b = _quiet(sp.logit, "b ~ x1 + x2", _dup(df), marginal_effects=kind)
    np.testing.assert_allclose(
        a.model_info["marginal_effects"]["dy/dx"],
        b.model_info["marginal_effects"]["dy/dx"],
        rtol=RTOL,
    )


@pytest.mark.parametrize(
    "kw",
    [
        dict(marginal_effects="atmeans"),
        dict(marginal_effects="at", at_values={"not_a_regressor": 2.0}),
    ],
)
def test_logit_refuses_a_marginal_effect_request_it_cannot_read(kw):
    with pytest.raises((StatsPAIError, ValueError)):
        sp.logit("b ~ x1 + x2", _data(), **kw)


def test_logit_missing_weight_does_not_hang():
    df = _data(n=60)
    df.loc[3, "w"] = np.nan
    with _deadline(2.0):
        try:
            res = _quiet(sp.logit, "b ~ x1", df, weights="w")
        except (StatsPAIError, ValueError):
            return  # refusing is fine
    # dropping the row, as Stata does with a missing weight, is fine too
    ref = _quiet(sp.logit, "b ~ x1", df.dropna(), weights="w")
    np.testing.assert_allclose(res.params, ref.params, rtol=RTOL)


@pytest.mark.parametrize("value", [-1.0, 0.0])
def test_logit_refuses_weights_that_are_not_positive(value):
    df = _data().assign(w=value)
    with pytest.raises((StatsPAIError, ValueError)):
        _quiet(sp.logit, "b ~ x1 + x2", df, weights="w")


@pytest.mark.parametrize("fn", [sp.logit, sp.probit])
def test_binary_single_cluster_is_refused(fn):
    df = _data().assign(g=1)
    with pytest.raises((StatsPAIError, ValueError)):
        _quiet(fn, "b ~ x1 + x2", df, cluster="g")


def test_logit_constant_outcome_has_a_clear_error():
    df = _data().assign(b=1.0)
    with pytest.raises((StatsPAIError, ValueError)):
        _quiet(sp.logit, "b ~ x1 + x2", df)
