"""Branch tests for ``statspai.regression.iv`` written while scanning the
uncovered lines of the module for defects (October 2026).

Numerical assertions compare with closed-form 2SLS / k-class algebra written
here in numpy, or with identities (weight rescaling, pre-filtering, the
two-part formula). Tests marked ``xfail(strict=True)`` assert the correct
behaviour of a defect that is not fixed yet.
"""

import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import statspai as sp
from statspai.exceptions import DataInsufficient, MethodIncompatibility, StatsPAIError
from statspai.regression.iv import IVEstimator, IVRegression
from statspai.regression.iv import iv as iv_fit

# Closed-form comparisons on a 4-instrument, 3-regressor design; both sides
# are a few float64 products, so 1e-8 only allows for the lstsq / inverse
# difference in how the same projection is solved.
RTOL = 1e-8
F = "y ~ (d ~ z1 + z2) + x"


def _data(seed: int = 1, n: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "z1": rng.normal(size=n),
            "z2": rng.normal(size=n),
            "x": rng.normal(size=n),
            "g": rng.integers(0, 15, n),
            "h": rng.integers(0, 9, n),
            "w": rng.uniform(0.5, 3.0, n),
        }
    )
    u = rng.normal(size=n)
    df["d"] = 0.7 * df.z1 + 0.4 * df.z2 + 0.5 * u + rng.normal(size=n)
    df["y"] = 1 + 1.5 * df.d + 0.5 * df.x + u * (1 + 0.5 * np.abs(df.x))
    return df


def _tsls(df: pd.DataFrame, weighted: bool = False):
    """2SLS by hand on the (optionally sqrt(w)-scaled) arrays.

    Returns beta (Intercept, x, d), residuals, X-hat and (X-hat'X-hat)^-1.
    """
    n = len(df)
    w = df.w.to_numpy() if weighted else np.ones(n)
    sw = np.sqrt(w * n / w.sum())[:, None]
    W = np.column_stack([np.ones(n), df.x, df.z1, df.z2]) * sw
    X = np.column_stack([np.ones(n), df.x, df.d]) * sw
    y = df.y.to_numpy() * sw[:, 0]
    Xh = W @ np.linalg.solve(W.T @ W, W.T @ X)
    bread = np.linalg.inv(Xh.T @ Xh)
    beta = bread @ Xh.T @ y
    return beta, y - X @ beta, Xh, bread


def _fit(*args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.iv(*args, **kwargs)


class TestTwoStageLeastSquaresClosedForm:
    @pytest.mark.parametrize("weighted", [False, True])
    def test_point_estimates_and_every_variance(self, weighted):
        df = _data()
        n, k = len(df), 3
        kw = {"weights": "w"} if weighted else {}
        beta, e, Xh, bread = _tsls(df, weighted)
        order = ["Intercept", "x", "d"]

        res = _fit(F, df, **kw)
        np.testing.assert_allclose(res.params[order], beta, rtol=RTOL)
        se = np.sqrt(np.diag(bread) * (e @ e) / (n - k))
        np.testing.assert_allclose(res.std_errors[order], se, rtol=RTOL)

        meat = (Xh * (e**2)[:, None]).T @ Xh
        hc0 = np.sqrt(np.diag(bread @ meat @ bread))
        np.testing.assert_allclose(
            _fit(F, df, robust="hc0", **kw).std_errors[order], hc0, rtol=RTOL
        )
        np.testing.assert_allclose(
            _fit(F, df, robust="hc1", **kw).std_errors[order],
            hc0 * np.sqrt(n / (n - k)),
            rtol=RTOL,
        )

        sums = pd.DataFrame(Xh * e[:, None]).groupby(df.g.to_numpy()).sum().to_numpy()
        G = sums.shape[0]
        cl = np.sqrt(np.diag(bread @ (sums.T @ sums) @ bread))
        res_cl = _fit(F, df, cluster="g", **kw)
        np.testing.assert_allclose(
            res_cl.std_errors[order],
            cl * np.sqrt(G / (G - 1) * (n - 1) / (n - k)),
            rtol=RTOL,
        )

        # small=False is ivregress without `small`: N divisor, HC0, no
        # cluster factor, and normal critical values
        np.testing.assert_allclose(
            _fit(F, df, small=False, **kw).std_errors[order],
            np.sqrt(np.diag(bread) * (e @ e) / n),
            rtol=RTOL,
        )
        np.testing.assert_allclose(
            _fit(F, df, small=False, robust="hc1", **kw).std_errors[order],
            hc0,
            rtol=RTOL,
        )
        big = _fit(F, df, small=False, cluster="g", **kw)
        np.testing.assert_allclose(big.std_errors[order], cl, rtol=RTOL)
        z = beta / cl
        np.testing.assert_allclose(
            big.pvalues[order], 2 * stats.norm.sf(np.abs(z)), rtol=1e-6
        )

    def test_alpha_sets_the_interval_level(self):
        df = _data()
        beta, e, _, bread = _tsls(df)
        n, k = len(df), 3
        se = np.sqrt(np.diag(bread) * (e @ e) / (n - k))
        ci = _fit(F, df, alpha=0.10).conf_int().loc["d"].to_numpy()
        half = stats.t.ppf(0.95, n - k) * se[2]
        np.testing.assert_allclose(ci, [beta[2] - half, beta[2] + half], rtol=1e-7)

    def test_two_part_formula_is_the_same_model(self):
        df = _data()
        a = _fit("y ~ x + d | x + z1 + z2", df)
        b = _fit(F, df)
        np.testing.assert_allclose(a.params[b.params.index], b.params, rtol=RTOL)
        np.testing.assert_allclose(
            a.std_errors[b.params.index], b.std_errors, rtol=RTOL
        )

    def test_redundant_instrument_is_dropped_with_a_warning(self):
        df = _data().assign(z1b=lambda t: 2 * t.z1)
        with pytest.warns(UserWarning, match="z1b"):
            a = sp.iv("y ~ (d ~ z1 + z1b) + x", df)
        b = _fit("y ~ (d ~ z1) + x", df)
        np.testing.assert_allclose(a.params, b.params, rtol=RTOL)
        np.testing.assert_allclose(a.std_errors, b.std_errors, rtol=RTOL)
        assert "z1b" in str(a.model_info["omitted_instruments"])


class TestKClass:
    def _kappa(self, df):
        n = len(df)
        Xx = np.column_stack([np.ones(n), df.x])
        W = np.column_stack([Xx, df.z1, df.z2])
        Y = np.column_stack([df.y, df.d])

        def resid(A, B):
            return B - A @ np.linalg.lstsq(A, B, rcond=None)[0]

        Yx, Yw = resid(Xx, Y), resid(W, Y)
        ev = np.linalg.eigvals(np.linalg.solve(Yw.T @ Yw, Yx.T @ Yx))
        return float(np.min(ev.real)), W

    def test_liml_is_the_k_class_estimator_at_the_smallest_root(self):
        df = _data()
        n = len(df)
        kappa, W = self._kappa(df)
        assert kappa > 1.0
        X = np.column_stack([np.ones(n), df.x, df.d])
        PX = W @ np.linalg.solve(W.T @ W, W.T @ X)
        AX = (1 - kappa) * X + kappa * PX
        beta = np.linalg.solve(AX.T @ X, AX.T @ df.y.to_numpy())
        res = _fit(F, df, method="liml")
        np.testing.assert_allclose(res.params[["Intercept", "x", "d"]], beta, rtol=1e-7)

    def test_liml_equals_2sls_when_just_identified(self):
        df = _data()
        a = _fit("y ~ (d ~ z1) + x", df, method="liml")
        b = _fit("y ~ (d ~ z1) + x", df)
        # kappa = 1 up to the eigen-solver's rounding
        np.testing.assert_allclose(a.params, b.params, rtol=1e-6)

    def test_fuller_with_zero_constant_is_liml(self):
        df = _data()
        a = _fit(F, df, method="fuller", fuller_alpha=0.0)
        b = _fit(F, df, method="liml")
        np.testing.assert_allclose(a.params, b.params, rtol=RTOL)
        c = _fit(F, df, method="fuller", fuller_alpha=4.0)
        # the Fuller constant moves the estimate towards OLS, by a little
        assert abs(c.params["d"] - b.params["d"]) < 0.05
        assert c.params["d"] != pytest.approx(b.params["d"], rel=1e-9)

    def test_just_identified_gmm_is_2sls(self):
        df = _data()
        a = _fit("y ~ (d ~ z1) + x", df, method="gmm", robust="hc1")
        b = _fit("y ~ (d ~ z1) + x", df)
        np.testing.assert_allclose(a.params[b.params.index], b.params, rtol=1e-7)


class TestWeights:
    @pytest.mark.parametrize("method", ["2sls", "liml", "fuller", "gmm", "jive"])
    def test_weights_are_scale_invariant(self, method):
        df = _data()
        kw = {"robust": "hc1"} if method == "gmm" else {}
        a = _fit(F, df, method=method, weights="w", **kw)
        b = _fit(F, df.assign(w=df.w * 12.5), method=method, weights="w", **kw)
        np.testing.assert_allclose(a.params, b.params, rtol=1e-7)
        np.testing.assert_allclose(a.std_errors, b.std_errors, rtol=1e-7)

    @pytest.mark.parametrize("method", ["2sls", "liml", "fuller", "gmm", "jive"])
    def test_constant_weights_are_no_weights(self, method):
        df = _data().assign(w=3.0)
        kw = {"robust": "hc1"} if method == "gmm" else {}
        a = _fit(F, df, method=method, weights="w", **kw)
        b = _fit(F, df, method=method, **kw)
        np.testing.assert_allclose(a.params, b.params, rtol=1e-7)
        np.testing.assert_allclose(a.std_errors, b.std_errors, rtol=1e-7)

    def test_weight_column_follows_the_fitted_rows(self):
        df = _data()
        df.loc[[5, 60], "x"] = np.nan
        a = _fit(F, df, weights="w", cluster="g")
        b = _fit(F, df.dropna(), weights="w", cluster="g")
        np.testing.assert_allclose(a.params, b.params, rtol=RTOL)
        np.testing.assert_allclose(a.std_errors, b.std_errors, rtol=RTOL)
        with pytest.raises(MethodIncompatibility, match="weights length"):
            _fit(F, df, weights=df.w.to_numpy())

    def test_invalid_weights_are_refused(self):
        df = _data()
        with pytest.raises(DataInsufficient, match="NaN or infinite"):
            _fit(F, df.assign(w=np.r_[np.nan, df.w.to_numpy()[1:]]), weights="w")
        with pytest.raises(MethodIncompatibility, match="strictly positive"):
            _fit(F, df.assign(w=0.0), weights="w")
        with pytest.raises(MethodIncompatibility, match="not a column"):
            _fit(F, df, weights="nope")

    def test_weights_refused_on_paths_that_would_ignore_them(self):
        df = _data().assign(lat=0.1, lon=0.2)
        for kw in (
            dict(vce="cr2", cluster="g"),
            dict(vce="cr3", cluster="g"),
            dict(cluster=["g", "h"]),
            dict(vce="wild", cluster="g", wild_reps=19, seed=1),
            dict(vce="conley", conley_lat="lat", conley_lon="lon", conley_cutoff=5),
        ):
            with pytest.raises(MethodIncompatibility, match="weighted fits"):
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    sp.ivreg(F, df, weights="w", **kw)
        with pytest.raises(MethodIncompatibility, match="not supported"):
            _fit(F, df, absorb="g", weights="w")


class TestClustersAndSmallSample:
    def test_ivreg_drops_rows_with_a_missing_cluster_label(self):
        df = _data()
        df["g"] = df.g.astype(float)
        df.loc[[5, 17, 40], "g"] = np.nan
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            a = sp.ivreg(F, df, cluster="g")
            b = sp.ivreg(F, df.dropna(), cluster="g")
        assert a.data_info["nobs"] == 397
        np.testing.assert_allclose(a.std_errors, b.std_errors, rtol=RTOL)

    def test_single_cluster_is_refused(self):
        df = _data().assign(g=1)
        with pytest.raises(DataInsufficient, match="at least two clusters"):
            _fit(F, df, cluster="g")

    def test_multiway_is_inclusion_exclusion_with_one_factor(self):
        df = _data()
        n, k = len(df), 3
        beta, e, Xh, bread = _tsls(df)
        scores = pd.DataFrame(Xh * e[:, None])

        def meat(keys):
            s = scores.groupby(keys).sum().to_numpy()
            return s.T @ s

        pair = (df.g.astype(str) + "_" + df.h.astype(str)).to_numpy()
        M = meat(df.g.to_numpy()) + meat(df.h.to_numpy()) - meat(pair)
        gmin = min(df.g.nunique(), df.h.nunique())
        V = gmin / (gmin - 1) * (n - 1) / (n - k) * bread @ M @ bread
        # negative eigenvalues of the inclusion-exclusion matrix are set to
        # zero (the ivreg2 / reghdfe adjustment the docstring names)
        ev, evec = np.linalg.eigh(V)
        V = evec @ np.diag(np.maximum(ev, 0.0)) @ evec.T
        res = _fit(F, df, cluster=["g", "h"])
        np.testing.assert_allclose(
            res.std_errors[["Intercept", "x", "d"]], np.sqrt(np.diag(V)), rtol=1e-7
        )
        same = _fit(F, df, cluster="g + h")
        np.testing.assert_allclose(same.std_errors, res.std_errors, rtol=RTOL)

    def test_small_false_is_refused_where_ivregress_has_no_counterpart(self):
        df = _data()
        with pytest.raises(MethodIncompatibility, match="True or False"):
            _fit(F, df, small=1)
        with pytest.raises(MethodIncompatibility, match="2SLS, LIML and GMM"):
            _fit(F, df, small=False, method="jive")
        with pytest.raises(MethodIncompatibility, match="no HC3 variance"):
            _fit(F, df, small=False, robust="hc3")
        with pytest.raises(MethodIncompatibility, match="not defined with absorb"):
            _fit(F, df, small=False, absorb="g")
        with pytest.raises(MethodIncompatibility, match="one-way clustering"):
            _fit(F, df, small=False, cluster=["g", "h"])

    def test_absorb_equals_dummies(self):
        df = _data()
        a = _fit(F, df, absorb="g")
        b = _fit("y ~ (d ~ z1 + z2) + x + C(g)", df)
        np.testing.assert_allclose(a.params[["x", "d"]], b.params[["x", "d"]], 1e-7)
        np.testing.assert_allclose(
            a.std_errors[["x", "d"]], b.std_errors[["x", "d"]], rtol=1e-7
        )


class TestValidationAndPredict:
    def test_option_validation(self):
        df = _data()
        for bad in (0.0, 1.5, "x", np.nan):
            with pytest.raises(MethodIncompatibility, match="alpha must be"):
                _fit(F, df, alpha=bad)
        with pytest.raises(MethodIncompatibility, match="cannot be combined"):
            _fit(F, df, robust="hc3", cluster="g")
        with pytest.raises(MethodIncompatibility, match="Under-identified"):
            _fit("y ~ (d + x ~ z1)", df)
        with pytest.raises(MethodIncompatibility, match="must be a string"):
            iv_fit(3, df)
        with pytest.raises(MethodIncompatibility, match="requires cluster"):
            sp.ivreg(F, df, vce="wild")
        with pytest.raises(MethodIncompatibility, match="requires cluster"):
            sp.ivreg(F, df, vce="cr2")
        with pytest.raises(MethodIncompatibility, match="conley_lat"):
            sp.ivreg(F, df, vce="conley")
        with pytest.raises(TypeError, match="unexpected keyword"):
            sp.ivreg(F, df, small=False)

    def test_predict_plugs_observed_regressors_into_the_structural_form(self):
        df = _data()
        model = IVRegression(formula=F, data=df)
        with pytest.raises(MethodIncompatibility, match="must be fitted first"):
            model.first_stage
        with pytest.raises(MethodIncompatibility, match="must be fitted"):
            model.predict()
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            res = model.fit()
        new = df.head(6)
        expected = (
            res.params["Intercept"] + res.params["x"] * new.x + res.params["d"] * new.d
        )
        np.testing.assert_allclose(model.predict(new), expected, rtol=1e-12)
        np.testing.assert_allclose(
            model.predict(), np.asarray(res.fitted_values()), rtol=1e-12
        )
        with pytest.raises(MethodIncompatibility, match="missing columns"):
            model.predict(new[["x"]])
        with pytest.raises(MethodIncompatibility, match="pandas DataFrame"):
            model.predict(new.to_numpy())
        with pytest.raises(MethodIncompatibility, match="must be numeric"):
            model.predict(new.assign(x="a"))
        assert len(model.first_stage) == 1
        assert model.hausman_test is not None

    def test_legacy_estimator_is_2sls(self):
        df = _data()
        n = len(df)
        Xx = np.column_stack([np.ones(n), df.x])
        est = IVEstimator()
        with pytest.raises(MethodIncompatibility, match="requires X_endog and Z"):
            est.estimate(df.y.to_numpy(), Xx)
        out = est.estimate(
            df.y.to_numpy(),
            Xx,
            X_endog=df[["d"]].to_numpy(),
            Z=df[["z1", "z2"]].to_numpy(),
            robust=True,
        )
        beta, e, Xh, bread = _tsls(df)
        np.testing.assert_allclose(out["params"], beta, rtol=RTOL)
        hc1 = bread @ ((Xh * (e**2)[:, None]).T @ Xh) @ bread * n / (n - 3)
        np.testing.assert_allclose(out["std_errors"], np.sqrt(np.diag(hc1)), rtol=RTOL)
        with pytest.raises(MethodIncompatibility, match="Unknown robust option"):
            est.estimate(
                df.y.to_numpy(),
                Xx,
                X_endog=df[["d"]].to_numpy(),
                Z=df[["z1", "z2"]].to_numpy(),
                robust="hc9",
            )


# --------------------------------------------------------------------- #
#  Defects (assert the correct behaviour; remove the marker with the fix)
# --------------------------------------------------------------------- #


def test_jive_is_the_leave_one_out_instrument_estimator():
    rng = np.random.default_rng(0)
    n, K = 120, 8
    Z = rng.normal(size=(n, K))
    u = rng.normal(size=n)
    d = Z @ np.full(K, 0.15) + 0.8 * u + 0.6 * rng.normal(size=n)
    y = 1.0 * d + u
    df = pd.DataFrame(Z, columns=[f"z{i}" for i in range(K)]).assign(d=d, y=y)
    # leave-one-out first-stage fitted values by refitting n times
    W = np.column_stack([np.ones(n), Z])
    d_loo = np.empty(n)
    for i in range(n):
        keep = np.arange(n) != i
        d_loo[i] = W[i] @ np.linalg.lstsq(W[keep], d[keep], rcond=None)[0]
    Xh = np.column_stack([np.ones(n), d_loo])
    X = np.column_stack([np.ones(n), d])
    jive1 = np.linalg.solve(Xh.T @ X, Xh.T @ y)[1]
    formula = "y ~ (d ~ " + " + ".join(f"z{i}" for i in range(K)) + ")"
    res = _fit(formula, df, method="jive")
    # same closed form, so agreement should be at rounding level
    assert res.params["d"] == pytest.approx(jive1, rel=1e-8)


def test_iv_missing_cluster_labels_are_not_merged_into_another_cluster():
    df = _data()
    df["g"] = df.g.astype(float)
    df.loc[[5, 17, 40], "g"] = np.nan
    a = _fit(F, df, cluster="g")
    b = _fit(F, df.dropna(), cluster="g")
    np.testing.assert_allclose(a.std_errors, b.std_errors, rtol=RTOL)


def test_unknown_gmm_vcov_is_refused():
    with pytest.raises(StatsPAIError):
        _fit(F, _data(), method="gmm", gmm_vcov="eficient")


def test_ivreg_small_sample_cluster_paths_say_they_are_2sls_only():
    with pytest.raises(MethodIncompatibility):
        sp.ivreg(F, _data(), cluster="g", vce="cr2", method="liml")
