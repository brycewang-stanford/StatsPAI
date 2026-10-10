"""Branch tests for ``statspai.regression.count``, ``zeroinflated`` and
``multinomial`` written while scanning their uncovered lines for defects
(October 2026).

Every likelihood is written out again here with numpy / scipy and the
reported estimate is checked against it: the gradient vanishes there and the
standard errors are the inverse of its numerically differentiated Hessian.
Other assertions are identities (frequency weights equal duplicated rows,
``offset=log(e)`` equals ``exposure=e``, absorbing a factor equals its
dummies). Tests marked ``xfail(strict=True)`` assert the correct behaviour
of a defect that is not fixed yet.
"""

import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import stats
from scipy.special import expit, gammaln, logsumexp

import statspai as sp
from statspai.exceptions import MethodIncompatibility, StatsPAIError

# Estimates come from iterative solvers stopped at a 1e-8 criterion;
# coefficients and standard errors are good to about 1e-6 relative.
RTOL = 1e-5


def _quiet(fn, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*args, **kwargs)


def _grad(f, theta, h=1e-5):
    theta = np.asarray(theta, dtype=float)
    out = np.empty_like(theta)
    for j in range(theta.size):
        e = np.zeros_like(theta)
        e[j] = h
        out[j] = (f(theta + e) - f(theta - e)) / (2 * h)
    return out


def _hess(f, theta, h=1e-4):
    theta = np.asarray(theta, dtype=float)
    k = theta.size
    H = np.empty((k, k))
    for i in range(k):
        for j in range(k):
            ei, ej = np.zeros(k), np.zeros(k)
            ei[i], ej[j] = h, h
            H[i, j] = (
                f(theta + ei + ej)
                - f(theta + ei - ej)
                - f(theta - ei + ej)
                + f(theta - ei - ej)
            ) / (4 * h * h)
    return H


def _assert_stationary(f, theta, n):
    """The log-likelihood gradient at ``theta`` is zero.

    Scaled by ``n`` because each term of the score is O(1) per observation;
    1e-5 * n is two orders above the central-difference error and four
    below the score at a wrong estimate.
    """
    assert np.abs(_grad(f, theta)).max() < 1e-5 * n


def _count_data(seed: int = 3, n: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
            "g": rng.integers(0, 12, n),
            "h": rng.integers(0, 8, n),
            "w": rng.integers(1, 4, n).astype(float),
            "e": rng.uniform(0.5, 3.0, n),
            "o": 0.5 * rng.normal(size=n),
        }
    )
    df["loge"] = np.log(df.e)
    mu = np.exp(0.3 + 0.5 * df.x1 - 0.3 * df.x2) * df.e
    df["y"] = rng.poisson(mu * rng.gamma(2, 0.5, n)).astype(float)
    return df


def _dup(df: pd.DataFrame) -> pd.DataFrame:
    return df.loc[df.index.repeat(df.w.astype(int))].reset_index(drop=True)


def _X(df: pd.DataFrame) -> np.ndarray:
    return np.column_stack([np.ones(len(df)), df.x1, df.x2])


F = "y ~ x1 + x2"


# ===================================================================== #
#  poisson / nbreg
# ===================================================================== #


class TestPoisson:
    def test_score_information_and_sandwiches(self):
        df = _count_data()
        n, X, y = len(df), _X(df), df.y.to_numpy()
        res = sp.poisson(F, df)
        beta = res.params.to_numpy()
        mu = np.exp(X @ beta)
        assert np.abs(X.T @ (y - mu)).max() < 1e-6 * n
        bread = np.linalg.inv(X.T @ (X * mu[:, None]))
        np.testing.assert_allclose(res.std_errors, np.sqrt(np.diag(bread)), rtol=RTOL)
        s = X * (y - mu)[:, None]
        hc0 = np.diag(bread @ (s.T @ s) @ bread)
        # Stata poisson: vce(robust) carries N/(N-1), vce(cluster) G/(G-1)
        for kind, factor in (
            ("hc0", 1.0),
            ("robust", n / (n - 1)),
            ("hc1", n / (n - 3)),
        ):
            got = sp.poisson(F, df, robust=kind).std_errors
            np.testing.assert_allclose(got, np.sqrt(hc0 * factor), rtol=RTOL)
        sums = pd.DataFrame(s).groupby(df.g.to_numpy()).sum().to_numpy()
        G = sums.shape[0]
        cl = np.diag(bread @ (sums.T @ sums) @ bread) * G / (G - 1)
        got = _quiet(sp.poisson, F, df, cluster="g").std_errors
        np.testing.assert_allclose(got, np.sqrt(cl), rtol=RTOL)

    def test_irr_reports_exponentiated_coefficients(self):
        df = _count_data()
        base = sp.poisson(F, df)
        irr = sp.poisson(F, df, irr=True)
        np.testing.assert_allclose(irr.params, np.exp(base.params), rtol=1e-10)
        # delta method
        np.testing.assert_allclose(
            irr.std_errors, np.exp(base.params) * base.std_errors, rtol=1e-10
        )

    def test_alpha_sets_the_interval_level(self):
        df = _count_data()
        res = sp.poisson(F, df, alpha=0.10)
        ci = res.conf_int().to_numpy()
        z = stats.norm.ppf(0.95)
        np.testing.assert_allclose(
            ci[:, 1] - ci[:, 0], 2 * z * res.std_errors.to_numpy(), rtol=1e-8
        )

    def test_predict_on_new_data_keeps_the_exposure(self):
        df = _count_data()
        res = sp.poisson(F, df, exposure="e")
        new = df.head(5)
        expected = np.exp(_X(new) @ res.params.to_numpy()) * new.e.to_numpy()
        np.testing.assert_allclose(np.asarray(res.predict(new)), expected, rtol=1e-10)

    def test_collinear_regressor_is_omitted(self):
        df = _count_data().assign(c=2.0)
        with pytest.warns(UserWarning, match="c omitted because of collinearity"):
            res = sp.poisson("y ~ x1 + c", df)
        assert list(res.params.index) == ["_cons", "x1"]

    def test_nonconvergence_is_announced(self):
        with pytest.warns(Warning, match="did not converge in 1 iter"):
            sp.poisson(F, _count_data(), maxiter=1)


@pytest.mark.parametrize(
    "fn, kw",
    [
        (sp.poisson, {}),
        (sp.nbreg, {}),
        (sp.nbreg, {"dispersion": "constant"}),
    ],
    ids=["poisson", "nb2", "nb1"],
)
class TestCountOptionsShared:
    def test_frequency_weights_equal_duplicated_rows(self, fn, kw):
        df = _count_data()
        a = _quiet(fn, F, df, weights="w", offset="loge", **kw)
        b = _quiet(fn, F, _dup(df), offset="loge", **kw)
        np.testing.assert_allclose(a.params, b.params, rtol=RTOL)
        np.testing.assert_allclose(a.std_errors, b.std_errors, rtol=RTOL)

    def test_weighted_cluster_equals_duplicated_rows(self, fn, kw):
        df = _count_data()
        a = _quiet(fn, F, df, weights="w", cluster="g", **kw)
        b = _quiet(fn, F, _dup(df), cluster="g", **kw)
        np.testing.assert_allclose(a.std_errors, b.std_errors, rtol=RTOL)

    def test_exposure_is_a_log_offset(self, fn, kw):
        df = _count_data()
        a = _quiet(fn, F, df, offset="loge", robust="robust", **kw)
        b = _quiet(fn, F, df, exposure="e", robust="robust", **kw)
        np.testing.assert_allclose(a.params, b.params, rtol=1e-9)
        np.testing.assert_allclose(a.std_errors, b.std_errors, rtol=1e-9)
        c = _quiet(fn, F, df, **kw)
        # ... and it is not ignored: the constant absorbs mean log exposure
        assert abs(c.params["_cons"] - a.params["_cons"]) > 0.3

    def test_yx_path_is_the_formula_path(self, fn, kw):
        df = _count_data()
        opts = dict(offset="loge", weights="w", cluster="g", **kw)
        a = _quiet(fn, data=df, y="y", x=["x1", "x2"], **opts)
        b = _quiet(fn, F, df, **opts)
        np.testing.assert_allclose(a.params.to_numpy(), b.params.to_numpy(), 1e-10)
        np.testing.assert_allclose(
            a.std_errors.to_numpy(), b.std_errors.to_numpy(), rtol=1e-10
        )

    def test_missing_cluster_label_drops_the_row(self, fn, kw):
        df = _count_data()
        df["g"] = df.g.astype(float)
        df.loc[[3, 30], "g"] = np.nan
        a = _quiet(fn, F, df, cluster="g", **kw)
        b = _quiet(fn, F, df.dropna(), cluster="g", **kw)
        assert a.data_info["nobs"] == len(df) - 2
        np.testing.assert_allclose(a.std_errors, b.std_errors, rtol=1e-9)

    def test_exposure_validation(self, fn, kw):
        df = _count_data()
        with pytest.raises(MethodIncompatibility, match="strictly positive"):
            fn(F, df.assign(e=0.0), exposure="e", **kw)
        with pytest.raises(StatsPAIError, match="NaN or infinite"):
            fn(F, df.assign(e=np.r_[np.nan, df.e.to_numpy()[1:]]), exposure="e", **kw)
        with pytest.raises(MethodIncompatibility, match="not available"):
            fn(F, df, robust="hc3", **kw)


class TestNegativeBinomialLikelihood:
    def test_nb2_estimate_maximises_the_nb2_likelihood(self):
        df = _count_data()
        n, X, y = len(df), _X(df), df.y.to_numpy()
        res = sp.nbreg(F, df)
        alpha = res.model_info["dispersion"]
        assert res.model_info["dispersion_type"] == "NB2"

        def ll(theta):
            mu, a = np.exp(X @ theta[:3]), np.exp(theta[3])
            r = 1 / a
            return np.sum(
                gammaln(y + r)
                - gammaln(r)
                - gammaln(y + 1)
                + r * np.log(r / (r + mu))
                + y * np.log(mu / (r + mu))
            )

        theta = np.r_[res.params.to_numpy(), np.log(alpha)]
        _assert_stationary(ll, theta, n)
        se = np.sqrt(np.diag(np.linalg.inv(-_hess(ll, theta))))
        # 1e-4: the reference Hessian is a second difference with step 1e-4
        np.testing.assert_allclose(res.std_errors, se[:3], rtol=1e-4)
        assert res.model_info["se_lnalpha"] == pytest.approx(se[3], rel=1e-4)

    def test_nb1_estimate_maximises_the_nb1_likelihood(self):
        df = _count_data()
        n, X, y = len(df), _X(df), df.y.to_numpy()
        res = sp.nbreg(F, df, dispersion="constant")
        delta = res.model_info["dispersion"]
        assert res.model_info["dispersion_type"] == "NB1"

        def ll(theta):
            # Var(y) = mu (1 + delta): size mu/delta, success prob 1/(1+delta)
            mu, d = np.exp(X @ theta[:3]), np.exp(theta[3])
            r = mu / d
            return np.sum(
                gammaln(y + r)
                - gammaln(r)
                - gammaln(y + 1)
                - r * np.log1p(d)
                + y * np.log(d / (1 + d))
            )

        theta = np.r_[res.params.to_numpy(), np.log(delta)]
        _assert_stationary(ll, theta, n)
        se = np.sqrt(np.diag(np.linalg.inv(-_hess(ll, theta))))
        np.testing.assert_allclose(res.std_errors, se[:3], rtol=1e-4)
        assert res.model_info["se_lndelta"] == pytest.approx(se[3], rel=1e-4)


# ===================================================================== #
#  ppmlhdfe / xtnbreg
# ===================================================================== #


class TestPpmlHdfe:
    def test_absorbing_a_factor_equals_its_dummies(self):
        df = _count_data()
        a = sp.ppmlhdfe(F, df, absorb="g")
        b = _quiet(sp.poisson, "y ~ x1 + x2 + C(g)", df, robust="robust")
        np.testing.assert_allclose(a.params, b.params[["x1", "x2"]], rtol=RTOL)
        np.testing.assert_allclose(a.std_errors, b.std_errors[["x1", "x2"]], rtol=RTOL)

    def test_weights_and_cluster_reach_the_absorbed_fit(self):
        df = _count_data()
        a = _quiet(sp.ppmlhdfe, F, df, absorb="g", weights="w", cluster="h")
        b = _quiet(sp.poisson, "y ~ x1 + x2 + C(g)", df, weights="w", cluster="h")
        np.testing.assert_allclose(a.params, b.params[["x1", "x2"]], rtol=RTOL)
        np.testing.assert_allclose(a.std_errors, b.std_errors[["x1", "x2"]], rtol=RTOL)
        c = _quiet(sp.ppmlhdfe, F, _dup(df), absorb="g", cluster="h")
        np.testing.assert_allclose(a.std_errors, c.std_errors, rtol=RTOL)

    def test_without_absorb_it_is_poisson(self):
        df = _count_data()
        a = sp.ppmlhdfe(F, df)
        b = sp.poisson(F, df)
        np.testing.assert_allclose(a.params.to_numpy(), b.params.to_numpy(), rtol=RTOL)

    def test_missing_weight_drops_the_row(self):
        df = _count_data()
        df.loc[[3, 9, 50], "w"] = np.nan
        a = _quiet(sp.ppmlhdfe, F, df, absorb="g", weights="w")
        b = _quiet(sp.ppmlhdfe, F, df.dropna(), absorb="g", weights="w")
        assert a.data_info["nobs"] == len(df) - 3
        np.testing.assert_allclose(a.params, b.params, rtol=1e-9)
        np.testing.assert_allclose(a.std_errors, b.std_errors, rtol=1e-9)

    def test_option_validation(self):
        df = _count_data()
        with pytest.raises(MethodIncompatibility, match="Unknown robust option"):
            sp.ppmlhdfe(F, df, absorb="g", robust="zz")
        with pytest.raises(ValueError, match="ssc must be one of"):
            sp.ppmlhdfe(F, df, absorb="g", ssc="zz")
        with pytest.raises(MethodIncompatibility, match="not recognised"):
            sp.ppmlhdfe(F, df, absorb="g", separation="zz")
        for vce in ("cr2", "wild"):
            with pytest.raises(MethodIncompatibility, match="does not support weights"):
                sp.ppmlhdfe(F, df, absorb="g", cluster="h", vce=vce, weights="w")


class TestXtnbregRefusals:
    def _panel(self):
        df = _count_data()
        return df.assign(id=np.repeat(np.arange(40), 10), t=np.tile(np.arange(10), 40))

    def test_fixed_effects_model_refuses_what_it_cannot_do(self):
        df = self._panel()
        kw = dict(entity="id", time="t")
        with pytest.raises(MethodIncompatibility, match="does not support weights"):
            sp.xtnbreg(F, df, weights="w", **kw)
        with pytest.raises(MethodIncompatibility, match="observed-information"):
            sp.xtnbreg(F, df, robust="robust", **kw)
        with pytest.raises(MethodIncompatibility, match="observed-information"):
            sp.xtnbreg(F, df, cluster="id", **kw)
        with pytest.raises(MethodIncompatibility, match="model must be one of"):
            sp.xtnbreg(F, df, model="pa", **kw)
        with pytest.raises(MethodIncompatibility, match="requires `entity=`"):
            sp.xtnbreg(F, df)
        with pytest.raises(MethodIncompatibility):
            sp.xtnbreg(F, df, offset="loge", exposure="e", **kw)

    def test_exposure_is_a_log_offset_and_changes_the_fit(self):
        df = self._panel()
        kw = dict(entity="id", time="t")
        a = _quiet(sp.xtnbreg, F, df, offset="loge", **kw)
        b = _quiet(sp.xtnbreg, F, df, exposure="e", **kw)
        np.testing.assert_allclose(a.params, b.params, rtol=1e-9)
        c = _quiet(sp.xtnbreg, F, df, **kw)
        assert abs(c.params["_cons"] - a.params["_cons"]) > 0.05

    def test_time_effects_add_period_dummies(self):
        df = self._panel()
        res = _quiet(sp.xtnbreg, F, df, entity="id", time="t", time_effects=True)
        assert [c for c in res.params.index if c.startswith("t=")] == [
            f"t={k}" for k in range(1, 10)
        ]


# ===================================================================== #
#  zero-inflated and hurdle models
# ===================================================================== #


def _zi_data(seed: int = 5, n: int = 600) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
            "z": rng.normal(size=n),
            "g": rng.integers(0, 15, n),
        }
    )
    zero = rng.uniform(size=n) < expit(-0.5 + 0.8 * df.z)
    lam = np.exp(0.5 + 0.4 * df.x1 - 0.3 * df.x2)
    df["y"] = np.where(zero, 0, rng.poisson(lam)).astype(float)
    nb = rng.negative_binomial(2, 2 / (2 + 1.4 * lam))
    df["ynb"] = np.where(zero, 0, nb).astype(float)
    return df


class TestZeroInflatedPoisson:
    def _ll_obs(self, df):
        X, Y = _X(df), df.y.to_numpy()
        Z = np.column_stack([np.ones(len(df)), df.z])

        def ll_obs(t):
            mu, p = np.exp(X @ t[:3]), expit(Z @ t[3:])
            zero = np.log(p + (1 - p) * np.exp(-mu))
            pos = np.log1p(-p) + Y * np.log(mu) - mu - gammaln(Y + 1)
            return np.where(Y == 0, zero, pos)

        return ll_obs

    def test_estimate_and_every_variance_from_the_likelihood(self):
        df = _zi_data()
        n = len(df)
        ll_obs = self._ll_obs(df)

        def ll(t):
            return ll_obs(t).sum()

        res = sp.zip_model(F, df, inflate=["z"])
        theta = res.params.to_numpy()
        assert list(res.params.index) == [
            "const",
            "x1",
            "x2",
            "inflate_const",
            "inflate_z",
        ]
        _assert_stationary(ll, theta, n)
        bread = np.linalg.inv(-_hess(ll, theta))
        np.testing.assert_allclose(res.std_errors, np.sqrt(np.diag(bread)), rtol=1e-4)

        # per-observation scores by central differences
        k, h = theta.size, 1e-5
        S = np.empty((n, k))
        for j in range(k):
            e = np.zeros(k)
            e[j] = h
            S[:, j] = (ll_obs(theta + e) - ll_obs(theta - e)) / (2 * h)
        hc0 = np.diag(bread @ (S.T @ S) @ bread)
        for kind, factor in (
            ("hc0", 1.0),
            ("robust", n / (n - 1)),
            ("hc1", n / (n - k)),
        ):
            got = sp.zip_model(F, df, inflate=["z"], robust=kind).std_errors
            np.testing.assert_allclose(got, np.sqrt(hc0 * factor), rtol=1e-4)
        sums = pd.DataFrame(S).groupby(df.g.to_numpy()).sum().to_numpy()
        G = sums.shape[0]
        cl = np.diag(bread @ (sums.T @ sums) @ bread) * G / (G - 1)
        got = _quiet(sp.zip_model, F, df, inflate=["z"], cluster="g").std_errors
        np.testing.assert_allclose(got, np.sqrt(cl), rtol=1e-4)

    def test_inflate_defaults_to_the_count_regressors(self):
        df = _zi_data()
        a = sp.zip_model(F, df)
        b = sp.zip_model(F, df, inflate=["x1", "x2"])
        assert list(a.params.index)[3:] == ["inflate_const", "inflate_x1", "inflate_x2"]
        np.testing.assert_allclose(a.params, b.params, rtol=1e-10)

    def test_yx_path_and_cluster_keys_on_dropped_rows(self):
        df = _zi_data()
        a = sp.zip_model(data=df, y="y", x=["x1", "x2"], inflate=["z"])
        b = sp.zip_model(F, df, inflate=["z"])
        np.testing.assert_allclose(a.params, b.params, rtol=1e-10)
        df.loc[[3, 40, 77], "x1"] = np.nan
        c = _quiet(sp.zip_model, F, df, inflate=["z"], cluster="g")
        d = _quiet(sp.zip_model, F, df.dropna(), inflate=["z"], cluster="g")
        np.testing.assert_allclose(c.std_errors, d.std_errors, rtol=1e-9)

    def test_input_validation(self):
        df = _zi_data()
        with pytest.raises(ValueError, match="non-negative integers"):
            sp.zip_model("yy ~ x1", df.assign(yy=df.y - 1))
        with pytest.raises(ValueError, match="non-negative integers"):
            sp.zip_model("yy ~ x1", df.assign(yy=df.y + 0.5))
        with pytest.raises(ValueError, match="`data` must be provided"):
            sp.zip_model(F)
        with pytest.raises(ValueError, match="Provide either"):
            sp.zip_model(data=df)
        with pytest.raises(MethodIncompatibility, match="not available"):
            sp.zip_model(F, df, robust="hc3")

    def test_no_zero_inflation_is_reported_as_a_flat_likelihood(self):
        df = _zi_data()
        with pytest.warns(Warning, match="likelihood is flat along"):
            sp.zip_model("yy ~ x1", df.assign(yy=df.y + 1), inflate=["z"])


class TestZeroInflatedNegativeBinomial:
    def test_estimate_maximises_the_zinb_likelihood(self):
        df = _zi_data()
        n, X, Y = len(df), _X(df), df.ynb.to_numpy()
        Z = np.column_stack([np.ones(n), df.z])

        def ll(t):
            mu, p, a = np.exp(X @ t[:3]), expit(Z @ t[3:5]), np.exp(t[5])
            r = 1 / a
            lognb = (
                gammaln(Y + r)
                - gammaln(r)
                - gammaln(Y + 1)
                + r * np.log(r / (r + mu))
                + Y * np.log(mu / (r + mu))
            )
            zero = np.log(p + (1 - p) * (r / (r + mu)) ** r)
            return np.sum(np.where(Y == 0, zero, np.log1p(-p) + lognb))

        res = sp.zinb("ynb ~ x1 + x2", df, inflate=["z"])
        assert res.params.index[-1] == "ln_alpha"
        theta = res.params.to_numpy()
        _assert_stationary(ll, theta, n)
        se = np.sqrt(np.diag(np.linalg.inv(-_hess(ll, theta))))
        np.testing.assert_allclose(res.std_errors, se, rtol=1e-4)


class TestHurdle:
    def test_binary_part_is_the_logit_of_any_positive_count(self):
        df = _zi_data()
        res = sp.hurdle(F, df)
        ref = sp.logit("p ~ x1 + x2", df.assign(p=(df.y > 0).astype(float)))
        np.testing.assert_allclose(
            res.params[["hurdle_const", "hurdle_x1", "hurdle_x2"]],
            ref.params,
            rtol=RTOL,
        )
        np.testing.assert_allclose(
            res.std_errors[["hurdle_const", "hurdle_x1", "hurdle_x2"]],
            ref.std_errors,
            rtol=1e-4,
        )

    def test_count_part_is_the_zero_truncated_poisson_mle(self):
        df = _zi_data()
        pos = df[df.y > 0]
        X, Y = _X(pos), pos.y.to_numpy()

        def ll(b):
            mu = np.exp(X @ b)
            return np.sum(Y * np.log(mu) - mu - gammaln(Y + 1) - np.log1p(-np.exp(-mu)))

        res = sp.hurdle(F, df)
        b = res.params[["count_const", "count_x1", "count_x2"]].to_numpy()
        _assert_stationary(ll, b, len(pos))
        se = np.sqrt(np.diag(np.linalg.inv(-_hess(ll, b))))
        np.testing.assert_allclose(
            res.std_errors[["count_const", "count_x1", "count_x2"]], se, rtol=1e-4
        )

    def test_negbin_spellings_select_the_truncated_nb2(self):
        df = _zi_data()
        a = sp.hurdle("ynb ~ x1 + x2", df, count_model="negbin")
        b = sp.hurdle("ynb ~ x1 + x2", df, count_model="nb")
        assert a.params.index[-1] == "ln_alpha"
        np.testing.assert_allclose(a.params, b.params, rtol=1e-10)
        pos = df[df.ynb > 0]
        X, Y = _X(pos), pos.ynb.to_numpy()

        def ll(t):
            mu, r = np.exp(X @ t[:3]), 1 / np.exp(t[3])
            lognb = (
                gammaln(Y + r)
                - gammaln(r)
                - gammaln(Y + 1)
                + r * np.log(r / (r + mu))
                + Y * np.log(mu / (r + mu))
            )
            return np.sum(lognb - np.log1p(-((r / (r + mu)) ** r)))

        theta = a.params[["count_const", "count_x1", "count_x2", "ln_alpha"]]
        _assert_stationary(ll, theta.to_numpy(), len(pos))

    def test_cluster_keys_on_dropped_rows(self):
        df = _zi_data()
        df.loc[[3, 40, 77], "x1"] = np.nan
        a = _quiet(sp.hurdle, F, df, cluster="g")
        b = _quiet(sp.hurdle, F, df.dropna(), cluster="g")
        np.testing.assert_allclose(a.std_errors, b.std_errors, rtol=1e-9)


# ===================================================================== #
#  mlogit / ologit / oprobit / clogit
# ===================================================================== #


def _choice_data(seed: int = 5, n: int = 600) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
            "g": rng.integers(0, 15, n),
        }
    )
    util = np.column_stack(
        [np.zeros(n), 0.5 * df.x1, -0.5 * df.x1 + 0.4 * df.x2]
    ) + rng.gumbel(size=(n, 3))
    df["c"] = util.argmax(axis=1)
    latent = 0.8 * df.x1 - 0.5 * df.x2 + rng.logistic(size=n)
    df["o"] = np.digitize(latent, [-1.0, 0.5])
    return df


class TestMultinomialLogit:
    def _ll_obs(self, df):
        X, c = _X(df), df.c.to_numpy()

        def ll_obs(t):
            eta = np.column_stack([np.zeros(len(df)), X @ t[:3], X @ t[3:]])
            return eta[np.arange(len(df)), c] - logsumexp(eta, axis=1)

        return ll_obs

    def test_estimate_and_variances_from_the_likelihood(self):
        df = _choice_data()
        n = len(df)
        ll_obs = self._ll_obs(df)

        def ll(t):
            return ll_obs(t).sum()

        res = sp.mlogit("c ~ x1 + x2", df)
        theta = res.params.to_numpy()
        assert list(res.params.index[:3]) == ["[1]_cons", "[1]x1", "[1]x2"]
        _assert_stationary(ll, theta, n)
        bread = np.linalg.inv(-_hess(ll, theta))
        np.testing.assert_allclose(res.std_errors, np.sqrt(np.diag(bread)), rtol=1e-4)
        k, h = theta.size, 1e-5
        S = np.empty((n, k))
        for j in range(k):
            e = np.zeros(k)
            e[j] = h
            S[:, j] = (ll_obs(theta + e) - ll_obs(theta - e)) / (2 * h)
        rob = np.diag(bread @ (S.T @ S) @ bread) * n / (n - 1)
        got = sp.mlogit("c ~ x1 + x2", df, robust="robust").std_errors
        np.testing.assert_allclose(got, np.sqrt(rob), rtol=1e-4)
        sums = pd.DataFrame(S).groupby(df.g.to_numpy()).sum().to_numpy()
        G = sums.shape[0]
        cl = np.diag(bread @ (sums.T @ sums) @ bread) * G / (G - 1)
        got = _quiet(sp.mlogit, "c ~ x1 + x2", df, cluster="g").std_errors
        np.testing.assert_allclose(got, np.sqrt(cl), rtol=1e-4)

    def test_changing_the_base_category_is_a_reparameterisation(self):
        df = _choice_data()
        a = sp.mlogit("c ~ x1 + x2", df)
        b = sp.mlogit("c ~ x1 + x2", df, base=2)
        for term in ("_cons", "x1", "x2"):
            assert b.params[f"[0]{term}"] == pytest.approx(
                -a.params[f"[2]{term}"], rel=RTOL
            )
            assert b.params[f"[1]{term}"] == pytest.approx(
                a.params[f"[1]{term}"] - a.params[f"[2]{term}"], rel=RTOL
            )
        with pytest.raises(ValueError, match="base must be in"):
            sp.mlogit("c ~ x1 + x2", df, base=7)

    def test_rrr_table_and_predicted_probabilities(self):
        df = _choice_data()
        res = sp.mlogit("c ~ x1 + x2", df, rrr=True)
        base = sp.mlogit("c ~ x1 + x2", df)
        np.testing.assert_allclose(res.params, np.exp(base.params), rtol=1e-10)
        np.testing.assert_allclose(
            res.std_errors, np.exp(base.params) * base.std_errors, rtol=1e-10
        )
        new = df.head(6)
        t = base.params.to_numpy()
        eta = np.column_stack([np.zeros(6), _X(new) @ t[:3], _X(new) @ t[3:]])
        expected = np.exp(eta - logsumexp(eta, axis=1, keepdims=True))
        np.testing.assert_allclose(np.asarray(base.predict(new)), expected, rtol=1e-10)

    def test_labels_validation_and_collinearity(self):
        df = _choice_data()
        named = df.assign(cs=df.c.map({0: "a", 1: "b", 2: "c"}))
        res = sp.mlogit("cs ~ x1 + x2", named)
        assert list(res.params.index[:1]) == ["[b]_cons"]
        with pytest.raises(ValueError, match="J >= 3"):
            sp.mlogit("c ~ x1", df.assign(c=1))
        with pytest.warns(UserWarning, match="k omitted because of collinearity"):
            dropped = sp.mlogit("c ~ x1 + k", df.assign(k=2.0))
        assert list(dropped.params.index) == [
            "[1]_cons",
            "[1]x1",
            "[2]_cons",
            "[2]x1",
        ]
        with pytest.raises(MethodIncompatibility, match="not available"):
            sp.mlogit("c ~ x1", df, robust="hc3")


class TestOrderedModels:
    @pytest.mark.parametrize(
        "fn, cdf", [(sp.ologit, expit), (sp.oprobit, stats.norm.cdf)]
    )
    def test_estimate_and_standard_errors_from_the_likelihood(self, fn, cdf):
        df = _choice_data()
        n, o = len(df), df.o.to_numpy()
        Xs = np.column_stack([df.x1, df.x2])

        def ll(t):
            xb = Xs @ t[:2]
            cuts = np.r_[-np.inf, t[2], t[3], np.inf]
            p = cdf(cuts[o + 1] - xb) - cdf(cuts[o] - xb)
            return np.sum(np.log(p))

        res = fn("o ~ x1 + x2", df)
        assert list(res.params.index) == ["x1", "x2", "/cut1", "/cut2"]
        theta = res.params.to_numpy()
        _assert_stationary(ll, theta, n)
        se = np.sqrt(np.diag(np.linalg.inv(-_hess(ll, theta))))
        np.testing.assert_allclose(res.std_errors, se, rtol=1e-4)

    @pytest.mark.parametrize("fn", [sp.ologit, sp.oprobit])
    def test_validation_collinearity_and_dropped_rows(self, fn):
        df = _choice_data()
        with pytest.raises(ValueError, match="J >= 3"):
            fn("b ~ x1", df.assign(b=(df.o > 0).astype(int)))
        with pytest.warns(UserWarning, match="k omitted because of collinearity"):
            res = fn("o ~ x1 + k", df.assign(k=2.0))
        assert list(res.params.index) == ["x1", "/cut1", "/cut2"]
        df.loc[[3, 40], "x1"] = np.nan
        a = _quiet(fn, "o ~ x1 + x2", df, cluster="g")
        b = _quiet(fn, "o ~ x1 + x2", df.dropna(), cluster="g")
        np.testing.assert_allclose(a.std_errors, b.std_errors, rtol=1e-9)


class TestConditionalLogit:
    def _data(self):
        rng = np.random.default_rng(8)
        m = 150
        df = pd.DataFrame(
            {
                "grp": np.repeat(np.arange(m), 4),
                "x1": rng.normal(size=4 * m),
                "x2": rng.normal(size=4 * m),
            }
        )
        u = 0.7 * df.x1 - 0.4 * df.x2 + rng.gumbel(size=4 * m)
        df["ch"] = (u.groupby(df.grp).transform("max") == u).astype(int)
        return df

    def test_estimate_and_standard_errors_from_the_likelihood(self):
        df = self._data()
        m = df.grp.nunique()
        Xs = df[["x1", "x2"]].to_numpy().reshape(m, 4, 2)
        chosen = df.ch.to_numpy().reshape(m, 4).argmax(axis=1)

        def ll(b):
            eta = Xs @ b
            return np.sum(eta[np.arange(m), chosen] - logsumexp(eta, axis=1))

        res = sp.clogit("ch ~ x1 + x2", df, group="grp")
        b = res.params.to_numpy()
        _assert_stationary(ll, b, m)
        se = np.sqrt(np.diag(np.linalg.inv(-_hess(ll, b))))
        np.testing.assert_allclose(res.std_errors, se, rtol=1e-4)

    def test_validation(self):
        df = self._data()
        with pytest.raises(ValueError, match="'group' must be specified"):
            sp.clogit("ch ~ x1 + x2", df)
        two = df.assign(ch=np.where(np.arange(len(df)) % 4 < 2, 1, 0))
        with pytest.raises(ValueError, match="exactly one chosen"):
            sp.clogit("ch ~ x1 + x2", two, group="grp")


# --------------------------------------------------------------------- #
#  Defects (assert the correct behaviour; remove the marker with the fix)
# --------------------------------------------------------------------- #


@pytest.mark.parametrize("fn", [sp.poisson, sp.nbreg])
def test_offset_is_not_dropped_when_exposure_is_also_given(fn):
    df = _count_data()
    try:
        both = _quiet(fn, F, df, offset="o", exposure="e")
    except StatsPAIError:
        return  # refusing the combination is fine
    total = _quiet(fn, F, df.assign(tot=df.o + df.loge), offset="tot")
    np.testing.assert_allclose(both.params, total.params, rtol=RTOL)


@pytest.mark.parametrize(
    "name",
    ["poisson", "nbreg", "ppmlhdfe", "zip", "hurdle", "mlogit", "ologit", "clogit"],
)
def test_single_cluster_is_refused(name):
    df = _count_data().assign(one=1)
    zi = _zi_data().assign(one=1)
    ch = _choice_data().assign(one=1)
    with pytest.raises((StatsPAIError, ValueError)):
        if name == "poisson":
            _quiet(sp.poisson, F, df, cluster="one")
        elif name == "nbreg":
            _quiet(sp.nbreg, F, df, cluster="one")
        elif name == "ppmlhdfe":
            _quiet(sp.ppmlhdfe, F, df, absorb="g", cluster="one")
        elif name == "zip":
            _quiet(sp.zip_model, F, zi, inflate=["z"], cluster="one")
        elif name == "hurdle":
            _quiet(sp.hurdle, F, zi, cluster="one")
        elif name == "mlogit":
            _quiet(sp.mlogit, "c ~ x1 + x2", ch, cluster="one")
        elif name == "ologit":
            _quiet(sp.ologit, "o ~ x1 + x2", ch, cluster="one")
        else:
            cl = TestConditionalLogit()._data().assign(one=1)
            _quiet(sp.clogit, "ch ~ x1 + x2", cl, group="grp", cluster="one")


@pytest.mark.parametrize("which", ["nbreg", "hurdle"])
def test_unknown_model_choice_is_refused(which):
    with pytest.raises((StatsPAIError, ValueError)):
        if which == "nbreg":
            _quiet(sp.nbreg, F, _count_data(), dispersion="meen")
        else:
            _quiet(sp.hurdle, F, _zi_data(), count_model="negbinomial")


@pytest.mark.parametrize(
    "case", ["negative_y", "negative_weights", "nan_weights", "nan_offset", "zero_y"]
)
@pytest.mark.parametrize("fn", [sp.poisson, sp.nbreg])
def test_count_inputs_are_validated(fn, case):
    df = _count_data(n=120)
    kw = {}
    if case == "negative_y":
        df = df.assign(y=df.y - 3)
    elif case == "zero_y":
        df = df.assign(y=0.0)
    elif case == "negative_weights":
        df, kw = df.assign(w=-df.w), {"weights": "w"}
    elif case == "nan_weights":
        df.loc[3, "w"] = np.nan
        kw = {"weights": "w"}
    else:
        df.loc[3, "loge"] = np.nan
        kw = {"offset": "loge"}
    try:
        res = _quiet(fn, F, df, **kw)
    except (StatsPAIError, ValueError) as exc:
        # a LinAlgError is a ValueError subclass in numpy; it is not a
        # diagnosis of the input
        assert not isinstance(exc, np.linalg.LinAlgError)
        return
    # dropping the incomplete row, as Stata does, is the other valid outcome
    assert case in ("nan_weights", "nan_offset")
    ref = _quiet(fn, F, df.dropna(), **kw)
    np.testing.assert_allclose(res.params, ref.params, rtol=RTOL)


def test_hurdle_without_positive_counts_is_refused():
    df = _zi_data().assign(y=0.0)
    with pytest.raises((StatsPAIError, ValueError)):
        _quiet(sp.hurdle, "y ~ x1", df)


def test_zip_omits_a_collinear_regressor():
    df = _zi_data().assign(c=1.0)
    try:
        res = _quiet(sp.zip_model, "y ~ x1 + c", df, inflate=["z"])
    except StatsPAIError:
        return
    assert "c" not in res.params.index
    assert np.isfinite(res.std_errors.to_numpy()).all()


@pytest.mark.parametrize("kw", [dict(formula="y ~ x1 + C(g)"), dict(inflate=["qq"])])
def test_zip_unknown_term_has_a_clear_error(kw):
    args = {"formula": F, "inflate": ["z"], **kw}
    with pytest.raises(StatsPAIError):
        sp.zip_model(data=_zi_data(), **args)
