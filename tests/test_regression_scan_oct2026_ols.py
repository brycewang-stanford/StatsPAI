"""Branch tests for ``statspai.regression.ols`` written while scanning the
uncovered lines of the module for defects (October 2026).

Every numerical assertion compares with an independent computation written
here (closed-form WLS / sandwich algebra in numpy) or with an identity.
Tests marked ``xfail(strict=True)`` assert the correct behaviour of a defect
that is not fixed yet; the fix only has to remove the marker.
"""

import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import statspai as sp
from statspai.exceptions import (
    DataInsufficient,
    MethodIncompatibility,
    NumericalInstability,
    StatsPAIError,
)
from statspai.regression.ols import OLSEstimator, OLSRegression

# Closed-form comparisons: both sides are a handful of float64 matrix
# products on a well-conditioned 3-column design, so they agree far below
# this; 1e-9 leaves room for a different BLAS summation order.
RTOL = 1e-9


def _data(seed: int = 0, n: int = 200) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=n),
            "g": rng.integers(0, 12, n),
            "h": rng.integers(0, 7, n),
            "w": rng.uniform(0.5, 3.0, n),
        }
    )
    df["y"] = 1 + 2 * df.x1 - df.x2 + rng.normal(size=n) * (1 + df.x1**2)
    return df


def _wls_pieces(df: pd.DataFrame, weighted: bool = True):
    """beta, residuals, bread and the sqrt(w)-scaled design, by hand."""
    n = len(df)
    X = np.column_stack([np.ones(n), df.x1, df.x2])
    y = df.y.to_numpy()
    w = df.w.to_numpy() if weighted else np.ones(n)
    w = w * n / w.sum()  # Stata aweight normalisation
    Xs, ys = X * np.sqrt(w)[:, None], y * np.sqrt(w)
    bread = np.linalg.inv(Xs.T @ Xs)
    beta = bread @ Xs.T @ ys
    return beta, ys - Xs @ beta, bread, Xs


class TestWeightedVarianceMenu:
    """weights= reaches every variance branch (it used to be dropped)."""

    @pytest.mark.parametrize("kind", ["nonrobust", "hc0", "hc1", "hc2", "hc3", "hc4"])
    def test_weighted_se_is_the_sandwich_on_the_scaled_design(self, kind):
        df = _data()
        n, k = len(df), 3
        beta, e, bread, Xs = _wls_pieces(df)
        lev = np.einsum("ij,jk,ik->i", Xs, bread, Xs)
        if kind == "nonrobust":
            V = bread * (e @ e) / (n - k)
        else:
            scale = {
                "hc0": np.ones(n),
                "hc1": np.full(n, n / (n - k)),
                "hc2": 1 / (1 - lev),
                "hc3": 1 / (1 - lev) ** 2,
                # Cribari-Neto (2004): exponent min(4, n h / k)
                "hc4": 1 / (1 - lev) ** np.minimum(4.0, n * lev / k),
            }[kind]
            V = bread @ ((Xs * (scale * e**2)[:, None]).T @ Xs) @ bread
        res = sp.regress("y ~ x1 + x2", df, robust=kind, weights="w")
        np.testing.assert_allclose(res.params.to_numpy(), beta, rtol=RTOL)
        np.testing.assert_allclose(
            res.std_errors.to_numpy(), np.sqrt(np.diag(V)), rtol=RTOL
        )

    def test_weights_are_scale_invariant_and_unit_weights_are_ols(self):
        df = _data()
        a = sp.regress("y ~ x1 + x2", df, weights="w", robust="hc1")
        b = sp.regress(
            "y ~ x1 + x2", df.assign(w=df.w * 7.5), weights="w", robust="hc1"
        )
        np.testing.assert_allclose(a.params, b.params, rtol=RTOL)
        np.testing.assert_allclose(a.std_errors, b.std_errors, rtol=RTOL)
        c = sp.regress("y ~ x1 + x2", df.assign(w=4.0), weights="w")
        d = sp.regress("y ~ x1 + x2", df)
        np.testing.assert_allclose(c.params, d.params, rtol=RTOL)
        np.testing.assert_allclose(c.std_errors, d.std_errors, rtol=RTOL)

    def test_weighted_cluster_se_and_its_t_reference(self):
        df = _data()
        n, k = len(df), 3
        beta, e, bread, Xs = _wls_pieces(df)
        scores = Xs * e[:, None]
        sums = pd.DataFrame(scores).groupby(df.g.to_numpy()).sum().to_numpy()
        G = sums.shape[0]
        V = (G / (G - 1)) * ((n - 1) / (n - k)) * bread @ (sums.T @ sums) @ bread
        se = np.sqrt(np.diag(V))
        res = sp.regress("y ~ x1 + x2", df, cluster="g", weights="w")
        np.testing.assert_allclose(res.std_errors.to_numpy(), se, rtol=RTOL)
        # Stata regress, vce(cluster): t(G - 1), not t(N - K)
        p = 2 * stats.t.sf(np.abs(beta / se), G - 1)
        np.testing.assert_allclose(res.pvalues.to_numpy(), p, rtol=1e-7)
        assert res.model_info["n_clusters"] == G

    def test_robust_model_f_is_the_wald_statistic_on_the_slopes(self):
        df = _data()
        n, k = len(df), 3
        beta, e, bread, Xs = _wls_pieces(df)
        V = (n / (n - k)) * bread @ ((Xs * (e**2)[:, None]).T @ Xs) @ bread
        wald = beta[1:] @ np.linalg.solve(V[1:, 1:], beta[1:]) / 2
        res = sp.regress("y ~ x1 + x2", df, robust="hc1", weights="w")
        assert res.diagnostics["F-statistic"] == pytest.approx(wald, rel=1e-8)
        assert res.diagnostics["Prob (F-statistic)"] == pytest.approx(
            stats.f.sf(wald, 2, n - k), rel=1e-6
        )

    def test_no_constant_weighted_r2_is_uncentred(self):
        df = _data()
        n = len(df)
        X = np.column_stack([df.x1, df.x2])
        w = df.w.to_numpy() * n / df.w.sum()
        beta = np.linalg.solve(X.T @ (X * w[:, None]), X.T @ (w * df.y))
        r = df.y.to_numpy() - X @ beta
        r2 = 1 - np.sum(w * r**2) / np.sum(w * df.y.to_numpy() ** 2)
        res = sp.regress("y ~ x1 + x2 - 1", df, weights="w")
        assert res.diagnostics["R-squared"] == pytest.approx(r2, rel=1e-10)

    def test_weights_column_stays_aligned_when_a_row_is_dropped(self):
        df = _data()
        df.loc[[3, 50], "x1"] = np.nan
        a = sp.regress("y ~ x1 + x2", df, weights="w", cluster="g")
        b = sp.regress("y ~ x1 + x2", df.dropna(), weights="w", cluster="g")
        np.testing.assert_allclose(a.params, b.params, rtol=RTOL)
        np.testing.assert_allclose(a.std_errors, b.std_errors, rtol=RTOL)

    def test_weight_array_of_the_wrong_length_is_refused(self):
        df = _data()
        df.loc[3, "x1"] = np.nan  # 199 rows are fitted, the array has 200
        with pytest.raises(MethodIncompatibility, match="weights length"):
            sp.regress("y ~ x1 + x2", df, weights=df.w.to_numpy())

    @pytest.mark.parametrize(
        "bad, exc, match",
        [
            (np.nan, DataInsufficient, "NaN or infinite"),
            (np.inf, DataInsufficient, "NaN or infinite"),
            (0.0, MethodIncompatibility, "strictly positive"),
            (-1.0, MethodIncompatibility, "strictly positive"),
        ],
    )
    def test_invalid_weight_values_are_refused(self, bad, exc, match):
        df = _data()
        df.loc[5, "w"] = bad
        with pytest.raises(exc, match=match):
            sp.regress("y ~ x1 + x2", df, weights="w")

    def test_weights_refused_where_the_variance_would_ignore_them(self):
        df = _data().assign(lat=0.1, lon=0.2)
        with pytest.raises(MethodIncompatibility, match="not implemented with"):
            sp.regress(
                "y ~ x1 + x2",
                df,
                vce="conley",
                conley_lat="lat",
                conley_lon="lon",
                conley_cutoff=5,
                weights="w",
            )
        with pytest.raises(MethodIncompatibility, match="not implemented with"):
            sp.regress("y ~ x1 + x2", df, vce="wild", cluster="g", weights="w")
        with pytest.raises(MethodIncompatibility, match="not implemented with"):
            sp.regress("y ~ x1 + x2", df, robust="hc2", dfadjust=True, weights="w")


class TestHacOptions:
    def _nw(self, df, lags, weighted=False):
        beta, e, bread, Xs = _wls_pieces(df, weighted=weighted)
        m = Xs * e[:, None]
        S = m.T @ m
        for j in range(1, lags + 1):
            gj = m[j:].T @ m[:-j]
            S += (1 - j / (lags + 1)) * (gj + gj.T)
        return bread @ S @ bread

    @pytest.mark.parametrize("weighted", [False, True])
    def test_newey_west_matches_the_bartlett_formula(self, weighted):
        df = _data()
        n, k = len(df), 3
        V = self._nw(df, 3, weighted)
        kw = {"weights": "w"} if weighted else {}
        res = sp.regress("y ~ x1 + x2", df, robust="hac", hac_lags=3, **kw)
        np.testing.assert_allclose(res.std_errors, np.sqrt(np.diag(V)), rtol=RTOL)
        small = sp.regress(
            "y ~ x1 + x2", df, robust="hac", hac_lags=3, hac_small=True, **kw
        )
        # Stata newey / NeweyWest(adjust=TRUE): the same matrix times N/(N-K)
        np.testing.assert_allclose(
            small.std_errors, np.sqrt(np.diag(V) * n / (n - k)), rtol=RTOL
        )
        assert res.model_info["hac_lags"] == 3
        assert small.model_info["hac_small"] is True

    def test_zero_lags_is_hc0_and_default_is_the_1994_rule(self):
        df = _data()
        a = sp.regress("y ~ x1 + x2", df, robust="hac", hac_lags=0)
        b = sp.regress("y ~ x1 + x2", df, robust="hc0")
        np.testing.assert_allclose(a.std_errors, b.std_errors, rtol=RTOL)
        d = sp.regress("y ~ x1 + x2", df, robust="hac")
        rule = int(np.floor(4 * (len(df) / 100) ** (2 / 9)))
        assert d.model_info["hac_lags"] == rule
        np.testing.assert_allclose(
            d.std_errors, np.sqrt(np.diag(self._nw(df, rule))), rtol=RTOL
        )

    def test_panel_hac_pairs_rows_within_units_at_exact_lags(self):
        rng = np.random.default_rng(4)
        units, periods, lags = 10, 12, 2
        df = pd.DataFrame(
            {
                "i": np.repeat(np.arange(units), periods),
                "t": np.tile(np.arange(periods), units),
                "x": rng.normal(size=units * periods),
            }
        )
        df["y"] = 1 + 2 * df.x + rng.normal(size=len(df))
        # a gap: the pair (t=4, t=6) of unit 0 is two periods apart
        df = df[~((df.i == 0) & (df.t == 5))].reset_index(drop=True)
        X = np.column_stack([np.ones(len(df)), df.x])
        bread = np.linalg.inv(X.T @ X)
        e = df.y.to_numpy() - X @ (bread @ X.T @ df.y.to_numpy())
        m = X * e[:, None]
        S = m.T @ m
        for a in range(len(df)):
            for b in range(len(df)):
                lag = df.t[a] - df.t[b]
                if df.i[a] == df.i[b] and 1 <= lag <= lags:
                    wt = 1 - lag / (lags + 1)
                    S += wt * (np.outer(m[a], m[b]) + np.outer(m[b], m[a]))
        expected = np.sqrt(np.diag(bread @ S @ bread))
        kw = dict(robust="hac", hac_lags=lags)
        res = sp.regress("y ~ x", df, hac_panel=("i", "t"), **kw)
        np.testing.assert_allclose(res.std_errors, expected, rtol=RTOL)
        # unsorted rows and a date-typed period give the same matrix
        shuffled = df.sample(frac=1, random_state=1)
        res_s = sp.regress("y ~ x", shuffled, hac_panel=("i", "t"), **kw)
        np.testing.assert_allclose(res_s.std_errors, expected, rtol=RTOL)
        assert res.model_info["hac_panel"] == ["i", "t"]

    def test_hac_option_validation(self):
        df = _data().assign(t=np.arange(200))
        f = "y ~ x1 + x2"
        with pytest.raises(MethodIncompatibility, match="non-negative integer"):
            sp.regress(f, df, robust="hac", hac_lags=1.5)
        with pytest.raises(MethodIncompatibility, match="only apply to"):
            sp.regress(f, df, hac_lags=1)
        with pytest.raises(MethodIncompatibility, match="only apply to"):
            sp.regress(f, df, robust="hc1", hac_small=True)
        with pytest.raises(DataInsufficient, match="autocovariances"):
            sp.regress(f, df, robust="hac", hac_lags=500)
        with pytest.raises(MethodIncompatibility, match="panel and the time"):
            sp.regress(f, df, robust="hac", hac_panel="g")
        with pytest.raises(MethodIncompatibility, match="not in the data"):
            sp.regress(f, df, robust="hac", hac_panel=("g", "zz"))
        with pytest.raises(MethodIncompatibility, match="do not identify the rows"):
            sp.regress(f, df, robust="hac", hac_panel=("g", "g"))
        missing_t = df.assign(t=np.r_[np.nan, df.t.to_numpy()[1:]])
        with pytest.raises(MethodIncompatibility, match="missing values"):
            sp.regress(f, missing_t, robust="hac", hac_panel=("g", "t"))

    def test_ewc_option_validation(self):
        df = _data()
        f = "y ~ x1 + x2"
        with pytest.raises(MethodIncompatibility, match="only applies to"):
            sp.regress(f, df, ewc_df=3)
        with pytest.raises(MethodIncompatibility, match="positive integer"):
            sp.regress(f, df, robust="ewc", ewc_df=0)
        with pytest.raises(DataInsufficient, match="cosine terms"):
            sp.regress(f, df, robust="ewc", ewc_df=500)
        with pytest.raises(MethodIncompatibility, match="cannot be combined"):
            sp.regress(f, df, robust="ewc", cluster="g")
        res = sp.regress(f, df, robust="ewc", ewc_df=9)
        assert res.model_info["ewc_df"] == 9
        assert res.data_info["df_inference"] == 9


class TestClusterMenuOnMissingRows:
    """Cluster keys must follow the rows that were fitted."""

    @pytest.mark.parametrize("vce", ["cluster", "cr2", "cr3", "jackknife"])
    def test_dropped_rows_equal_prefiltering(self, vce):
        df = _data()
        df.loc[[3, 77], "x1"] = np.nan
        kw = {} if vce == "cluster" else {"vce": vce}
        a = sp.regress("y ~ x1 + x2", df, cluster="g", **kw)
        b = sp.regress("y ~ x1 + x2", df.dropna(), cluster="g", **kw)
        np.testing.assert_allclose(a.std_errors, b.std_errors, rtol=RTOL)

    def test_missing_cluster_label_drops_the_row(self):
        df = _data()
        df["g"] = df.g.astype(float)
        df.loc[[3, 77], "g"] = np.nan
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            a = sp.regress("y ~ x1 + x2", df, cluster="g", weights="w")
        b = sp.regress("y ~ x1 + x2", df.dropna(), cluster="g", weights="w")
        assert a.data_info["nobs"] == 198
        np.testing.assert_allclose(a.params, b.params, rtol=RTOL)
        np.testing.assert_allclose(a.std_errors, b.std_errors, rtol=RTOL)

    def test_two_way_with_dropped_rows_and_weights(self):
        df = _data()
        df.loc[[3, 77], "x1"] = np.nan
        a = sp.regress("y ~ x1 + x2", df, cluster=["g", "h"], weights="w")
        b = sp.regress("y ~ x1 + x2", df.dropna(), cluster=["g", "h"], weights="w")
        np.testing.assert_allclose(a.std_errors, b.std_errors, rtol=RTOL)
        # inclusion-exclusion with the single G_min factor, by hand
        d = df.dropna().reset_index(drop=True)
        n, k = len(d), 3
        beta, e, bread, Xs = _wls_pieces(d)
        scores = pd.DataFrame(Xs * e[:, None])

        def meat(keys):
            s = scores.groupby(keys).sum().to_numpy()
            return s.T @ s

        pair = d.g.astype(str) + "_" + d.h.astype(str)
        M = meat(d.g.to_numpy()) + meat(d.h.to_numpy()) - meat(pair.to_numpy())
        gmin = min(d.g.nunique(), d.h.nunique())
        V = (gmin / (gmin - 1)) * ((n - 1) / (n - k)) * bread @ M @ bread
        # the inclusion-exclusion matrix is not PSD on this sample; negative
        # eigenvalues are set to zero (Cameron-Gelbach-Miller 2011), which is
        # announced by a RuntimeWarning and recorded in the diagnostics
        ev, evec = np.linalg.eigh(V)
        assert ev.min() < 0
        V = evec @ np.diag(np.maximum(ev, 0.0)) @ evec.T
        np.testing.assert_allclose(a.std_errors, np.sqrt(np.diag(V)), rtol=1e-8)
        assert a.diagnostics["Two-way VCOV negative eigenvalues"] == 1
        assert a.model_info["cluster"] == ["g", "h"]

    def test_single_cluster_is_refused_on_every_cluster_path(self):
        df = _data().assign(g=1)
        for kw in ({}, {"vce": "cr2"}, {"vce": "cr3"}):
            with pytest.raises(DataInsufficient, match="at least two clusters"):
                sp.regress("y ~ x1 + x2", df, cluster="g", **kw)
        with pytest.raises(DataInsufficient, match="at least two clusters"):
            sp.regress("y ~ x1 + x2", df, cluster=["g", "h"])

    def test_vcov_dict_spelling_is_the_same_estimator(self):
        df = _data()
        a = sp.regress("y ~ x1 + x2", df, vcov={"CRV3": "g"})
        b = sp.regress("y ~ x1 + x2", df, vce="cr3", cluster="g")
        np.testing.assert_allclose(a.std_errors, b.std_errors, rtol=RTOL)
        c = sp.regress("y ~ x1 + x2", df, vcov="hetero")
        d = sp.regress("y ~ x1 + x2", df, robust="hc1")
        np.testing.assert_allclose(c.std_errors, d.std_errors, rtol=RTOL)

    def test_dfadjust_keeps_the_hc2_standard_errors(self):
        df = _data()
        a = sp.regress("y ~ x1 + x2", df, robust="hc2", dfadjust=True)
        b = sp.regress("y ~ x1 + x2", df, robust="hc2")
        np.testing.assert_allclose(a.std_errors, b.std_errors, rtol=1e-8)
        assert a.model_info["dfadjust"] is True
        # joint tests keep the ordinary denominator df (Stata `test`)
        assert a.data_info["df_resid"] == len(df) - 3
        c = sp.regress("y ~ x1 + x2", df, vce="cr2", cluster="g", dfadjust=True)
        d = sp.regress("y ~ x1 + x2", df, vce="cr2", cluster="g")
        np.testing.assert_allclose(c.std_errors, d.std_errors, rtol=1e-8)
        assert c.data_info["df_resid"] == df.g.nunique() - 1


class TestRegressValidation:
    def test_requests_that_cannot_be_honoured_are_refused(self):
        df = _data()
        f = "y ~ x1 + x2"
        cases = [
            (dict(dfadjust=True), "Bell-McCaffrey"),
            (dict(vce="cr2"), "requires cluster"),
            (dict(vce="wild"), "requires cluster"),
            (dict(vce="conley"), "conley_lat"),
            (dict(cluster="zz"), "not a column"),
            (dict(robust="hc3", cluster="g"), "cannot be combined"),
        ]
        for kw, match in cases:
            with pytest.raises(MethodIncompatibility, match=match):
                sp.regress(f, df, **kw)

    def test_input_errors_name_the_problem(self):
        df = _data()
        with pytest.raises(TypeError, match="pandas DataFrame"):
            sp.regress("y ~ x1", {"y": [1.0], "x1": [2.0]})
        with pytest.raises(ValueError, match="empty"):
            sp.regress("y ~ x1", df.iloc[:0])
        with pytest.raises(ValueError, match=r"not found in data: \['q'\]"):
            sp.regress("y ~ x1 + q", df)
        with pytest.raises(ValueError, match="entirely NaN"):
            sp.regress("y ~ x1", df.assign(y=np.nan))
        with pytest.raises(TypeError, match="unexpected keyword"):
            sp.regress("y ~ x1", df, subset=df.x1 > 0)

    def test_collinear_regressor_is_omitted_or_raises(self):
        df = _data().assign(c=3.0)
        df["x3"] = df.x1 + df.x2
        with pytest.warns(UserWarning, match="omitted because of collinearity"):
            res = sp.regress("y ~ x1 + x2 + x3 + c", df)
        assert list(res.params.index) == ["Intercept", "x1", "x2"]
        assert {o["variable"] for o in res.model_info["omitted"]} == {"x3", "c"}
        ref = sp.regress("y ~ x1 + x2", df)
        np.testing.assert_allclose(res.params, ref.params, rtol=1e-8)
        with pytest.raises(NumericalInstability):
            sp.regress("y ~ x1 + x2 + x3", df, collinear="raise")
        with pytest.raises(MethodIncompatibility, match="'omit' or 'raise'"):
            sp.regress("y ~ x1", df, collinear="drop")

    def test_constant_outcome_warns(self):
        df = _data().assign(y=3.0)
        with pytest.warns(UserWarning, match="zero variance"):
            res = sp.regress("y ~ x1", df)
        assert np.isnan(res.diagnostics["R-squared"])


class TestEstimatorKernel:
    def _arrays(self):
        rng = np.random.default_rng(0)
        n = 120
        X = np.column_stack([np.ones(n), rng.normal(size=n)])
        g = np.repeat(np.arange(10), 12)
        y = X @ [1.0, 2.0] + rng.normal(size=n) + np.repeat(rng.normal(size=10), 12)
        return y, X, g

    def test_robust_aliases(self):
        y, X, _ = self._arrays()
        est = OLSEstimator()
        hc1 = est.estimate(y, X, robust="hc1")["std_errors"]
        for alias in (True, "robust", "HC1"):
            np.testing.assert_allclose(
                est.estimate(y, X, robust=alias)["std_errors"], hc1, rtol=RTOL
            )
        iid = est.estimate(y, X)["std_errors"]
        for alias in (False, "none", "iid", "classical"):
            np.testing.assert_allclose(
                est.estimate(y, X, robust=alias)["std_errors"], iid, rtol=RTOL
            )
        with pytest.raises(MethodIncompatibility, match="Unknown robust option"):
            est.estimate(y, X, robust="zzz")

    def test_array_validation(self):
        y, X, g = self._arrays()
        est = OLSEstimator()
        with pytest.raises(DataInsufficient, match="more observations than"):
            est.estimate(y[:2], X[:2])
        with pytest.raises(DataInsufficient, match="non-finite"):
            est.estimate(np.r_[np.nan, y[1:]], X)
        with pytest.raises(MethodIncompatibility, match="must be 2-D"):
            est.estimate(y, X[:, 1])
        with pytest.raises(MethodIncompatibility, match="rows"):
            est.estimate(y[:-1], X)
        with pytest.raises(MethodIncompatibility, match="cluster length"):
            est.estimate(y, X, cluster=g[:-1])
        with pytest.raises(DataInsufficient, match="non-missing cluster"):
            est.estimate(y, X, cluster=np.r_[np.nan, g[1:].astype(float)])
        # a column vector for y is accepted and read as a vector
        a = est.estimate(y.reshape(-1, 1), X)["params"]
        np.testing.assert_allclose(a, est.estimate(y, X)["params"], rtol=RTOL)

    def test_cluster_array_matches_the_formula_path(self):
        y, X, g = self._arrays()
        df = pd.DataFrame({"y": y, "x": X[:, 1], "g": g})
        a = OLSEstimator().estimate(y, X, cluster=g)["std_errors"]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            b = sp.regress("y ~ x", df, cluster="g").std_errors
        np.testing.assert_allclose(a, b, rtol=RTOL)

    def test_model_class_validation(self):
        y, X, _ = self._arrays()
        with pytest.raises(ValueError, match="Must provide either"):
            OLSRegression().fit()
        with pytest.raises(MethodIncompatibility, match="var_names has 1"):
            OLSRegression(y=y, X=X, var_names=["a"]).fit()
        with pytest.raises(ValueError, match="not a column"):
            OLSRegression(y=y, X=X).fit(weights="w")
        model = OLSRegression(y=y, X=X)
        with pytest.raises(MethodIncompatibility, match="must be fitted"):
            model.predict()
        res = model.fit()
        assert list(res.params.index) == ["x0", "x1"]
        with pytest.raises(MethodIncompatibility, match="fit with a formula"):
            model.predict(pd.DataFrame({"x": [1.0]}))


class TestPredict:
    def _fit(self, **kw):
        df = _data()
        model = OLSRegression("y ~ x1 + x2", df)
        return df, model, model.fit(**kw)

    def test_confidence_and_prediction_intervals_closed_form(self):
        df, model, res = self._fit()
        n, k = len(df), 3
        beta, e, bread, _ = _wls_pieces(df, weighted=False)
        s2 = e @ e / (n - k)
        new = df.head(5)
        Xn = np.column_stack([np.ones(5), new.x1, new.x2])
        var_mean = np.einsum("ij,jk,ik->i", Xn, s2 * bread, Xn)
        tcrit = stats.t.ppf(0.95, n - k)
        ci = model.predict(new, what="confidence", alpha=0.10)
        np.testing.assert_allclose(ci["yhat"], Xn @ beta, rtol=RTOL)
        np.testing.assert_allclose(
            ci["upper"] - ci["lower"], 2 * tcrit * np.sqrt(var_mean), rtol=1e-8
        )
        pi = model.predict(new, what="prediction", alpha=0.10)
        np.testing.assert_allclose(
            pi["upper"] - pi["lower"], 2 * tcrit * np.sqrt(var_mean + s2), rtol=1e-8
        )

    def test_in_sample_paths(self):
        df, model, res = self._fit()
        fitted = np.asarray(res.fitted_values()).ravel()
        np.testing.assert_allclose(model.predict(), fitted, rtol=RTOL)
        frame = model.predict(return_df=True)
        assert list(frame.columns) == ["yhat"]
        ci = model.predict(what="confidence")
        np.testing.assert_allclose(ci["yhat"], fitted, rtol=RTOL)
        assert (ci["upper"] > ci["lower"]).all()

    def test_predict_validation(self):
        df, model, _ = self._fit()
        with pytest.raises(MethodIncompatibility, match="`what` must be"):
            model.predict(df, what="interval")
        for bad in (0.0, 1.0, np.nan, "x"):
            with pytest.raises(MethodIncompatibility, match="`alpha` must be"):
                model.predict(df, what="confidence", alpha=bad)
        with pytest.raises(MethodIncompatibility, match="prediction design"):
            model.predict(df[["x1"]])

    def test_categorical_level_unseen_at_fit_is_refused(self):
        df = _data()
        df["c"] = np.where(df.g < 6, "a", "b")
        model = OLSRegression("y ~ x1 + C(c)", df)
        model.fit()
        new = pd.DataFrame({"x1": [0.0], "c": ["zzz"]})
        with pytest.raises(MethodIncompatibility, match="prediction design"):
            model.predict(new)


# --------------------------------------------------------------------- #
#  Defects (assert the correct behaviour; remove the marker with the fix)
# --------------------------------------------------------------------- #


def test_array_fit_does_not_silently_drop_a_cluster_request():
    rng = np.random.default_rng(0)
    n = 120
    X = np.column_stack([np.ones(n), rng.normal(size=n)])
    y = X @ [1.0, 2.0] + rng.normal(size=n)
    with pytest.raises(StatsPAIError):
        OLSRegression(y=y, X=X).fit(cluster="g")


def test_three_way_cluster_request_has_a_clear_error():
    df = _data()
    with pytest.raises(MethodIncompatibility):
        sp.regress("y ~ x1 + x2", df, cluster=["g", "h", "w"])


def test_array_fit_hac_panel_has_a_clear_error():
    rng = np.random.default_rng(0)
    X = np.column_stack([np.ones(50), rng.normal(size=50)])
    y = rng.normal(size=50)
    with pytest.raises(StatsPAIError):
        OLSRegression(y=y, X=X).fit(robust="hac", hac_panel=("i", "t"))
