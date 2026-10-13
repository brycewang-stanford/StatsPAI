"""Option and degenerate-input tests for ``sp.var``, ``sp.irf``,
``sp.granger_causality``, the VAR variance decomposition and ``sp.svar``
(scan round 3).

Numerical assertions compare with algebra written here: equation-by-equation
OLS, the moving-average recursion from the coefficient matrices, a numerical
delta method for the asymptotic bands, a residual bootstrap rebuilt from the
same random draws, Wald statistics from the coefficients and their
covariance, and Cholesky factors for the structural identifications.
"""

import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import statspai as sp
from statspai.exceptions import (
    DataInsufficient,
    IdentificationFailure,
    MethodIncompatibility,
    StatsPAIError,
)

NAMES = list("abc")
K, P, H = 3, 2, 4
NAN = np.nan


@pytest.fixture(scope="module")
def frame():
    rng = np.random.default_rng(3)
    T = 150
    A = np.array([[0.5, 0.1, 0.0], [0.2, 0.4, 0.1], [0.0, 0.2, 0.3]])
    L = np.array([[1.0, 0, 0], [0.3, 1, 0], [0.2, 0.1, 1]])
    Y = np.zeros((T, 3))
    for t in range(1, T):
        Y[t] = A @ Y[t - 1] + L @ rng.normal(size=3) + [1, 2, 3]
    out = pd.DataFrame(Y, columns=NAMES)
    out["z"] = rng.normal(size=T)
    return out


def _design(frame, trend, exog):
    Y = frame[NAMES].to_numpy()
    lhs = Y[P:]
    n = len(lhs)
    cols = [Y[1:-1], Y[:-2]]
    if trend != "n":
        cols.append(np.ones((n, 1)))
    if trend == "ct":
        cols.append(np.arange(1.0, n + 1).reshape(-1, 1))
    if exog:
        cols.append(frame[exog].to_numpy()[P:])
    return lhs, np.column_stack(cols)


def _ols(frame, trend="c", exog=None):
    lhs, X = _design(frame, trend, exog)
    B = np.linalg.lstsq(X, lhs, rcond=None)[0]
    U = lhs - X @ B
    return B, U, X, lhs


def _ma(B, periods):
    A1, A2 = B[:K].T, B[K : 2 * K].T
    phi = [np.eye(K)]
    for s in range(1, periods + 1):
        acc = phi[s - 1] @ A1
        if s >= 2:
            acc = acc + phi[s - 2] @ A2
        phi.append(acc)
    return phi


def _irf_array(out, key="irf"):
    arr = np.zeros((len(out["periods"]), K, K))
    for j, shock in enumerate(NAMES):
        for i, resp in enumerate(NAMES):
            arr[:, i, j] = out[key][f"{shock} -> {resp}"]
    return arr


SPECS = [("c", None), ("ct", None), ("n", None), ("c", ["z"]), ("ct", ["z"])]


class TestReducedForm:
    @pytest.mark.parametrize("trend, exog", SPECS)
    @pytest.mark.parametrize("se_df", ["stata", "r"])
    def test_equation_by_equation_ols(self, frame, trend, exog, se_df):
        v = sp.var(frame, variables=NAMES, lags=P, trend=trend, exog=exog, se_df=se_df)
        B, U, X, lhs = _ols(frame, trend, exog)
        n, m = X.shape
        S_ml = U.T @ U / n
        S_se = S_ml if se_df == "stata" else U.T @ U / (n - m)
        bread = np.linalg.inv(X.T @ X)
        for i, name in enumerate(NAMES):
            table = v.coefs[name]
            # same normal equations; 1e-9 leaves room for inv against lstsq
            np.testing.assert_allclose(table["coef"], B[:, i], atol=1e-9)
            np.testing.assert_allclose(
                table["se"], np.sqrt(S_se[i, i] * np.diag(bread)), rtol=1e-9
            )
        expected_names = [f"L{lag}.{n_}" for lag in (1, 2) for n_ in NAMES]
        expected_names += ["_cons"] * (trend != "n") + ["_trend"] * (trend == "ct")
        expected_names += exog or []
        assert list(v.coefs["a"].index) == expected_names
        np.testing.assert_allclose(v.sigma_u.to_numpy(), S_ml, rtol=1e-9)
        ll = -(n * K / 2) * (1 + np.log(2 * np.pi)) - n / 2 * np.log(
            np.linalg.det(S_ml)
        )
        assert v.log_likelihood == pytest.approx(ll, rel=1e-10)
        assert v.aic == pytest.approx(-2 * ll / n + 2 * K * m / n, rel=1e-10)
        assert v.bic == pytest.approx(-2 * ll / n + np.log(n) * K * m / n, rel=1e-10)
        assert v.n_obs == n

    def test_default_variables_are_the_numeric_columns(self, frame):
        mixed = frame.assign(label="x")
        v = sp.var(mixed, lags=1)
        assert v.var_names == NAMES + ["z"]

    def test_option_errors(self, frame):
        with pytest.raises(ValueError, match="se_df must be"):
            sp.var(frame, variables=NAMES, se_df="x")
        with pytest.raises(MethodIncompatibility, match="exog="):
            sp.var(frame, variables=NAMES, exog=["nope"])
        with pytest.raises(MethodIncompatibility, match="exog="):
            sp.var(frame, variables=NAMES, exog=["a"])
        # three usable rows for ten coefficients: refused under every se_df
        with pytest.raises(DataInsufficient, match="saturated"):
            sp.var(frame.iloc[:6], variables=NAMES, lags=3, se_df="r")


class TestImpulseResponses:
    @pytest.mark.parametrize("trend, exog", [("c", None), ("ct", ["z"]), ("n", None)])
    @pytest.mark.parametrize("se_df", ["stata", "r"])
    def test_closed_form_from_the_coefficient_matrices(self, frame, trend, exog, se_df):
        v = sp.var(frame, variables=NAMES, lags=P, trend=trend, exog=exog, se_df=se_df)
        B, U, X, _ = _ols(frame, trend, exog)
        n, m = X.shape
        phi = np.stack(_ma(B, 3))
        divisor = n if se_df == "stata" else n - m
        chol = np.linalg.cholesky(U.T @ U / divisor)
        np.testing.assert_allclose(
            _irf_array(v.irf(3, orthogonal=False)), phi, atol=1e-10
        )
        np.testing.assert_allclose(_irf_array(v.irf(3)), phi @ chol, atol=1e-10)
        np.testing.assert_allclose(
            _irf_array(sp.irf(v, 3, cumulative=True)),
            np.cumsum(phi @ chol, axis=0),
            atol=1e-10,
        )
        # sigma_df overrides the divisor the fit implies
        other = "unbiased" if se_df == "stata" else "ml"
        chol_o = np.linalg.cholesky(U.T @ U / (n - m if other == "unbiased" else n))
        np.testing.assert_allclose(
            _irf_array(sp.irf(v, 3, sigma_df=other)), phi @ chol_o, atol=1e-10
        )

    def test_impulse_and_response_select_one_path(self, frame):
        v = sp.var(frame, variables=NAMES, lags=P)
        full = v.irf(3)
        one = v.irf(3, impulse="b", response="c")
        assert list(one["irf"]) == ["b -> c"]
        np.testing.assert_array_equal(one["irf"]["b -> c"], full["irf"]["b -> c"])
        assert one["periods"] == [0, 1, 2, 3]
        zero = v.irf(0)
        assert zero["irf"]["b -> a"].shape == (1,) and zero["irf"]["b -> a"][0] == 0.0

    @staticmethod
    def _numeric_delta(fn, B, S, bread, n, with_sigma):
        """Delta-method standard errors with numerical Jacobians.

        ``Var(vec A) = S kron (X'X)^{-1}`` for the lag block and
        ``Cov(s_ij, s_lm) = (s_il s_jm + s_im s_jl) / n`` for the residual
        covariance, the two blocks independent.
        """
        kp = K * P
        a0 = B[:kp].copy()
        f0 = fn(a0, S).ravel()
        h = 1e-6
        Ja = np.zeros((f0.size, kp * K))
        for c in range(K):
            for r in range(kp):
                d = np.zeros_like(a0)
                d[r, c] = h
                Ja[:, c * kp + r] = (fn(a0 + d, S) - fn(a0 - d, S)).ravel() / (2 * h)
        var = np.einsum("ij,jk,ik->i", Ja, np.kron(S, bread[:kp, :kp]), Ja)
        if with_sigma:
            idx = [(i, j) for j in range(K) for i in range(j, K)]
            Js = np.zeros((f0.size, len(idx)))
            for col, (i, j) in enumerate(idx):
                d = np.zeros((K, K))
                d[i, j] = d[j, i] = h
                Js[:, col] = (fn(a0, S + d) - fn(a0, S - d)).ravel() / (2 * h)
            cov_s = np.array(
                [
                    [(S[i, l] * S[j, m] + S[i, m] * S[j, l]) / n for (l, m) in idx]
                    for (i, j) in idx
                ]
            )
            var = var + np.einsum("ij,jk,ik->i", Js, cov_s, Js)
        return np.sqrt(np.clip(var, 0, None)).reshape(H + 1, K, K)

    @pytest.mark.parametrize(
        "trend, exog, se_df",
        [("c", None, "stata"), ("ct", ["z"], "stata"), ("n", None, "r")],
    )
    @pytest.mark.parametrize(
        "orthogonal, cumulative", [(False, False), (True, False), (True, True)]
    )
    def test_asymptotic_band_is_the_delta_method(
        self, frame, trend, exog, se_df, orthogonal, cumulative
    ):
        v = sp.var(frame, variables=NAMES, lags=P, trend=trend, exog=exog, se_df=se_df)
        B, U, X, _ = _ols(frame, trend, exog)
        n, m = X.shape
        S = U.T @ U / (n if se_df == "stata" else n - m)

        def theta(a, sig):
            phi = np.stack(_ma(a, H))
            out = phi @ np.linalg.cholesky(sig) if orthogonal else phi
            return np.cumsum(out, axis=0) if cumulative else out

        expect = self._numeric_delta(theta, B, S, np.linalg.inv(X.T @ X), n, orthogonal)
        out = sp.irf(
            v, H, orthogonal=orthogonal, cumulative=cumulative, ci="asymptotic"
        )
        # central differences with step 1e-6 on smooth functions (observed
        # agreement 4e-11)
        np.testing.assert_allclose(_irf_array(out, "se"), expect, atol=1e-8)
        z = stats.norm.ppf(0.975)
        np.testing.assert_allclose(
            _irf_array(out, "upper"),
            _irf_array(out) + z * _irf_array(out, "se"),
            rtol=1e-12,
        )
        assert out["ci"] == {"ci": "asymptotic", "alpha": 0.05}

    def test_alpha_sets_the_width_of_the_asymptotic_band(self, frame):
        v = sp.var(frame, variables=NAMES, lags=P)
        wide = sp.irf(v, 2, ci="asymptotic", alpha=0.01)
        narrow = sp.irf(v, 2, ci="asymptotic", alpha=0.20)
        ratio = stats.norm.ppf(0.995) / stats.norm.ppf(0.90)
        w = _irf_array(wide, "upper") - _irf_array(wide, "lower")
        nar = _irf_array(narrow, "upper") - _irf_array(narrow, "lower")
        mask = nar > 1e-12
        np.testing.assert_allclose(w[mask] / nar[mask], ratio, rtol=1e-10)


class TestBootstrapBands:
    REPS, SEED = 40, 7

    def _by_hand(self, frame, trend, exog):
        """The residual bootstrap rebuilt from the same generator draws:
        centred residuals drawn with replacement, the series regenerated
        from its first ``P`` observations, the VAR re-estimated."""
        B, U, X, _ = _ols(frame, trend, exog)
        n = X.shape[0]
        kp = K * P
        resid = U - U.mean(axis=0)
        det = X[:, kp:] @ B[kp:]
        start = frame[NAMES].to_numpy()[:P]
        rng = np.random.default_rng(self.SEED)
        draws = []
        for _ in range(self.REPS):
            e = resid[rng.integers(0, n, size=n)]
            y = np.empty((n + P, K))
            y[:P] = start
            for t in range(n):
                y[t + P] = np.r_[y[t + 1], y[t]] @ B[:kp] + det[t] + e[t]
            Xb = X.copy()
            Xb[:, :K] = y[1:-1]
            Xb[:, K:kp] = y[:-2]
            Bb = np.linalg.lstsq(Xb, y[P:], rcond=None)[0]
            ub = y[P:] - Xb @ Bb
            draws.append(np.stack(_ma(Bb, 3)) @ np.linalg.cholesky(ub.T @ ub / n))
        return np.stack(draws), np.stack(_ma(B, 3)) @ np.linalg.cholesky(U.T @ U / n)

    @pytest.mark.parametrize("trend, exog", [("c", None), ("ct", ["z"])])
    def test_percentile_band_equals_a_hand_rolled_bootstrap(self, frame, trend, exog):
        v = sp.var(frame, variables=NAMES, lags=P, trend=trend, exog=exog)
        draws, theta = self._by_hand(frame, trend, exog)
        out = sp.irf(v, 3, ci="bootstrap", boot="efron", reps=self.REPS, seed=self.SEED)
        # identical draws and algebra; lstsq against inv only
        np.testing.assert_allclose(
            _irf_array(out, "lower"), np.quantile(draws, 0.025, axis=0), atol=1e-8
        )
        np.testing.assert_allclose(
            _irf_array(out, "upper"), np.quantile(draws, 0.975, axis=0), atol=1e-8
        )
        np.testing.assert_allclose(
            _irf_array(out, "se"), draws.std(axis=0, ddof=1), atol=1e-8
        )
        hall = sp.irf(v, 3, ci="bootstrap", boot="hall", reps=self.REPS, seed=self.SEED)
        np.testing.assert_allclose(
            _irf_array(hall, "lower"),
            2 * theta - np.quantile(draws, 0.975, axis=0),
            atol=1e-8,
        )
        assert out["ci"]["boot"] == "efron" and out["ci"]["reps"] == self.REPS

    def test_default_band_is_kilian_and_reproducible(self, frame):
        v = sp.var(frame, variables=NAMES, lags=P)
        a = sp.irf(v, 2, ci="bootstrap", reps=30, seed=1)
        b = sp.irf(v, 2, ci="bootstrap", reps=30, seed=1)
        assert a["ci"]["boot"] == "kilian"
        np.testing.assert_array_equal(_irf_array(a, "lower"), _irf_array(b, "lower"))
        lo, hi = _irf_array(a, "lower"), _irf_array(a, "upper")
        assert np.all(lo <= hi)

    def test_bootstrap_option_errors(self, frame):
        v = sp.var(frame, variables=NAMES, lags=P)
        with pytest.raises(MethodIncompatibility, match="ci='bogus'"):
            sp.irf(v, 2, ci="bogus")
        with pytest.raises(MethodIncompatibility, match="boot must be"):
            sp.irf(v, 2, ci="bootstrap", boot="bogus", reps=30)
        with pytest.raises(MethodIncompatibility, match="too few"):
            sp.irf(v, 2, ci="bootstrap", reps=5)
        with pytest.raises(MethodIncompatibility, match="alpha must be in"):
            sp.irf(v, 2, ci="asymptotic", alpha=2)
        with pytest.raises(MethodIncompatibility, match="sigma_df must be"):
            sp.irf(v, 2, sigma_df="bogus")


class TestVarianceDecomposition:
    def test_shares_from_the_orthogonalised_responses(self, frame):
        v = sp.var(frame, variables=NAMES, lags=P)
        B, U, X, _ = _ols(frame)
        theta = np.stack(_ma(B, H)) @ np.linalg.cholesky(U.T @ U / len(U))
        cum = np.cumsum(theta**2, axis=0)
        share = cum / cum.sum(axis=2, keepdims=True)
        table = v.fevd(H)
        for j, shock in enumerate(NAMES):
            for i, resp in enumerate(NAMES):
                got = table[(table.shock == shock) & (table.response == resp)]
                assert got["fevd"].iloc[0] == 0.0
                np.testing.assert_allclose(
                    got["fevd"].to_numpy()[1:], share[:H, i, j], atol=1e-10
                )
        totals = table[table.period > 0].groupby(["response", "period"])["fevd"].sum()
        np.testing.assert_allclose(totals, 1.0, rtol=1e-12)
        first = table[
            (table.shock == "a") & (table.response == "a") & (table.period == 1)
        ]
        assert first["fevd"].iloc[0] == pytest.approx(1.0, abs=1e-12)

    def test_asymptotic_standard_errors_are_the_delta_method(self, frame):
        v = sp.var(frame, variables=NAMES, lags=P)
        B, U, X, _ = _ols(frame)
        n = len(U)
        S = U.T @ U / n

        def shares(a, sig):
            theta = np.stack(_ma(a, H)) @ np.linalg.cholesky(sig)
            cum = np.cumsum(theta**2, axis=0)
            out = np.zeros_like(theta)
            out[1:] = cum[:-1] / cum[:-1].sum(axis=2, keepdims=True)
            return out

        expect = TestImpulseResponses._numeric_delta(
            shares, B, S, np.linalg.inv(X.T @ X), n, True
        )
        table = v.fevd(H, ci="asymptotic")
        got = np.zeros((H + 1, K, K))
        for j, shock in enumerate(NAMES):
            for i, resp in enumerate(NAMES):
                rows = table[(table.shock == shock) & (table.response == resp)]
                got[:, i, j] = rows["se"].to_numpy()
        np.testing.assert_allclose(got, expect, atol=1e-8)

    def test_bootstrap_bands_stay_in_the_unit_interval(self, frame):
        v = sp.var(frame, variables=NAMES, lags=P)
        table = v.fevd(3, ci="bootstrap", reps=30, seed=0, boot="efron")
        assert (table["lower"] >= 0).all() and (table["upper"] <= 1).all()
        assert (table["lower"] <= table["upper"]).all()
        with pytest.raises(MethodIncompatibility, match="ci='bogus'"):
            v.fevd(3, ci="bogus")
        with pytest.raises(MethodIncompatibility, match="boot must be"):
            v.fevd(3, ci="bootstrap", boot="hall", reps=30)
        with pytest.raises(MethodIncompatibility, match="alpha must be in"):
            v.fevd(3, ci="asymptotic", alpha=0)


class TestGranger:
    @pytest.mark.parametrize("se_df", ["stata", "r"])
    @pytest.mark.parametrize("trend, exog", [("c", None), ("ct", ["z"])])
    def test_wald_statistic_by_hand(self, frame, se_df, trend, exog):
        v = sp.var(frame, variables=NAMES, lags=P, trend=trend, exog=exog, se_df=se_df)
        B, U, X, _ = _ols(frame, trend, exog)
        n, m = X.shape
        s2 = (U[:, 0] @ U[:, 0]) / (n if se_df == "stata" else n - m)
        idx = [1, 2, 4, 5]  # both lags of b and c in the equation of a
        V = s2 * np.linalg.inv(X.T @ X)[np.ix_(idx, idx)]
        wald = float(B[idx, 0] @ np.linalg.solve(V, B[idx, 0]))
        g = sp.granger_causality(v, caused="a", causing=["b", "c"])
        assert g["chi2"] == pytest.approx(wald, rel=1e-9)
        assert g["F_stat"] == pytest.approx(wald / 4, rel=1e-9)
        assert (g["df1"], g["df2"]) == (4, n - m)
        assert g["p_value"] == pytest.approx(stats.f.sf(wald / 4, 4, n - m), rel=1e-8)
        assert g["chi2_p_value"] == pytest.approx(stats.chi2.sf(wald, 4), rel=1e-8)
        assert g["reject"] == (g["p_value"] < 0.05)

    def test_data_interface_fits_the_bivariate_var(self, frame):
        direct = sp.granger_causality(data=frame, caused="a", causing="b", lags=2)
        fitted = sp.var(frame, variables=["a", "b"], lags=2).granger_test("a", "b")
        assert direct["chi2"] == pytest.approx(fitted["chi2"], rel=1e-12)
        assert direct["df1"] == 2

    def test_argument_errors(self, frame):
        v = sp.var(frame, variables=NAMES, lags=P)
        with pytest.raises(MethodIncompatibility, match="requires both"):
            sp.granger_causality(v, caused="a")
        with pytest.raises(MethodIncompatibility, match="Provide var_result or data"):
            sp.granger_causality(caused="a", causing="b")
        with pytest.raises(MethodIncompatibility, match="must not contain the caused"):
            sp.granger_causality(v, caused="a", causing=["a", "b"])


class TestForecast:
    @pytest.mark.parametrize("trend", ["c", "ct", "n"])
    def test_dynamic_recursion_and_known_coefficient_errors(self, frame, trend):
        v = sp.var(frame, variables=NAMES, lags=P, trend=trend)
        B, U, X, _ = _ols(frame, trend)
        Y = frame[NAMES].to_numpy()
        n = len(U)

        def step(recent, older, h):
            x = np.r_[recent, older]
            if trend != "n":
                x = np.r_[x, 1.0]
            if trend == "ct":
                x = np.r_[x, n + h]
            return x @ B

        f1 = step(Y[-1], Y[-2], 1)
        f2 = step(f1, Y[-1], 2)
        fc = v.forecast(2, alpha=0.10)
        np.testing.assert_allclose(fc.loc[1, NAMES].to_numpy(float), f1, atol=1e-10)
        np.testing.assert_allclose(fc.loc[2, NAMES].to_numpy(float), f2, atol=1e-10)
        S = U.T @ U / n
        A1 = B[:K].T
        mse2 = S + A1 @ S @ A1.T
        assert fc.loc[1, "b_se"] == pytest.approx(np.sqrt(S[1, 1]), rel=1e-9)
        assert fc.loc[2, "b_se"] == pytest.approx(np.sqrt(mse2[1, 1]), rel=1e-9)
        z = stats.norm.ppf(0.95)
        assert fc.loc[2, "b_upper"] == pytest.approx(
            f2[1] + z * np.sqrt(mse2[1, 1]), rel=1e-9
        )

    def test_exogenous_variables_and_zero_steps_are_refused(self, frame):
        with pytest.raises(MethodIncompatibility, match="future values are not known"):
            sp.var(frame, variables=["a", "b"], exog=["z"]).forecast(2)
        with pytest.raises(MethodIncompatibility, match="steps must be at least 1"):
            sp.var(frame, variables=["a", "b"]).forecast(0)


class TestStructural:
    @pytest.fixture(scope="class")
    def fit(self, frame):
        return sp.var(frame, variables=NAMES, lags=P)

    def test_recursive_identification_is_the_cholesky_factor(self, fit):
        S = fit.sigma_u.to_numpy()
        res = sp.svar(fit, B=np.tril(np.full((K, K), NAN)))
        np.testing.assert_allclose(
            np.abs(res.impact.to_numpy()), np.abs(np.linalg.cholesky(S)), atol=1e-7
        )
        assert res.log_likelihood == pytest.approx(fit.log_likelihood, abs=1e-7)
        assert res.overid is None
        table = res.fevd(3)
        totals = table[table.period > 0].groupby(["response", "period"])["fevd"].sum()
        np.testing.assert_allclose(totals, 1.0, rtol=1e-10)

    def test_standard_errors_are_the_inverse_information(self, fit):
        S = fit.sigma_u.to_numpy()
        n = fit.n_obs
        res = sp.svar(fit, B=np.tril(np.full((K, K), NAN)))
        idx = [(i, j) for j in range(K) for i in range(j, K)]
        est = np.asarray(res.B, dtype=float)
        theta0 = np.array([est[i, j] for i, j in idx])

        def loglik(theta):
            Bm = np.zeros((K, K))
            for value, (i, j) in zip(theta, idx):
                Bm[i, j] = value
            M = np.linalg.inv(Bm)
            return (
                -n
                / 2
                * (
                    K * np.log(2 * np.pi)
                    - np.log(np.linalg.det(M) ** 2)
                    + np.trace(M.T @ M @ S)
                )
            )

        h = 1e-4
        Hm = np.zeros((len(idx), len(idx)))
        for a in range(len(idx)):
            for b in range(len(idx)):
                ea, eb = np.zeros(len(idx)), np.zeros(len(idx))
                ea[a], eb[b] = h, h
                Hm[a, b] = (
                    loglik(theta0 + ea + eb)
                    - loglik(theta0 + ea - eb)
                    - loglik(theta0 - ea + eb)
                    + loglik(theta0 - ea - eb)
                ) / (4 * h * h)
        se = np.sqrt(np.diag(np.linalg.inv(-Hm)))
        got = np.array([res.table.loc[f"B[{i + 1},{j + 1}]", "se"] for i, j in idx])
        # Exactly identified: observed and expected information coincide at
        # the estimate. The Hessian here is a second difference with step
        # 1e-4, good to about six digits.
        np.testing.assert_allclose(got, se, rtol=1e-4)

    def test_diagonal_model_is_overidentified_with_the_lr_statistic(self, fit):
        S = fit.sigma_u.to_numpy()
        n = fit.n_obs
        res = sp.svar(fit, B=np.where(np.eye(K, dtype=bool), NAN, 0.0))
        np.testing.assert_allclose(
            np.diag(res.impact.to_numpy()) ** 2, np.diag(S), rtol=1e-7
        )
        lr = n * (np.log(np.prod(np.diag(S))) - np.log(np.linalg.det(S)))
        assert res.overid["statistic"] == pytest.approx(lr, rel=1e-7)
        assert res.overid["df"] == 3.0
        assert res.overid["pvalue"] == pytest.approx(stats.chi2.sf(lr, 3), rel=1e-6)

    def test_a_only_pattern_reproduces_the_covariance(self, fit):
        S = fit.sigma_u.to_numpy()
        res = sp.svar(fit, A=[[1, 0, 0], [NAN, 1, 0], [NAN, NAN, 1]])
        impact = res.impact.to_numpy()
        np.testing.assert_allclose(impact @ impact.T, S, atol=1e-7)

    def test_long_run_identification_is_the_cholesky_of_the_long_run_variance(
        self, fit
    ):
        S = fit.sigma_u.to_numpy()
        B = fit._B
        total = np.eye(K) - B[:K].T - B[K : 2 * K].T
        a_bar = np.linalg.inv(total)
        C = np.linalg.cholesky(a_bar @ S @ a_bar.T)
        res = sp.svar(fit, long_run=np.tril(np.full((K, K), NAN)))
        np.testing.assert_allclose(np.abs(np.asarray(res.C)), C, atol=1e-6)
        np.testing.assert_allclose(
            np.abs(res.impact.to_numpy()), np.abs(total @ C), atol=1e-6
        )
        cum = res.irf(300, cumulative=True)
        last = (
            cum[cum.period == 300]
            .pivot(index="response", columns="shock", values="irf")
            .loc[NAMES]
            .to_numpy()
        )
        # the accumulated response converges to C (largest root about 0.7)
        np.testing.assert_allclose(np.abs(last), C, atol=1e-6)

    def test_sign_restricted_draws_are_rotations_that_satisfy_the_signs(self, fit):
        S = fit.sigma_u.to_numpy()
        res = sp.svar(fit, sign={"s1": {"a": "+", "b": "-"}}, n_draws=40, seed=0)
        draws = res._draws
        assert res.shock_names == ["s1", "shock2", "shock3"]
        assert draws.shape == (40, K, K) and res.n_accepted == 40
        for impact in draws:
            np.testing.assert_allclose(impact @ impact.T, S, atol=1e-10)
        assert (draws[:, 0, 0] > 0).all() and (draws[:, 1, 0] < 0).all()
        table = res.irf(2)
        assert (table["lower"] <= table["irf"]).all()
        assert (table["irf"] <= table["upper"]).all()
        again = sp.svar(fit, sign={"s1": {"a": "+", "b": "-"}}, n_draws=40, seed=0)
        np.testing.assert_array_equal(again._draws, draws)

    def test_structural_bootstrap_of_a_recursive_model_is_the_var_bootstrap(self, fit):
        res = sp.svar(fit, B=np.tril(np.full((K, K), NAN)))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            band = res.irf(3, ci="bootstrap", reps=30, seed=1, boot="efron")
            hall = res.irf(3, ci="bootstrap", reps=30, seed=1, boot="hall")
            ref = sp.irf(fit, 3, ci="bootstrap", reps=30, seed=1, boot="efron")
        rows = band[(band.shock == "shock2") & (band.response == "c")]
        # the same draws, identified by maximum likelihood instead of a
        # Cholesky factorisation
        np.testing.assert_allclose(rows["lower"], ref["lower"]["b -> c"], atol=1e-6)
        np.testing.assert_allclose(rows["upper"], ref["upper"]["b -> c"], atol=1e-6)
        np.testing.assert_allclose(
            hall["lower"], 2 * band["irf"] - band["upper"], atol=1e-12
        )

    def test_identification_uses_the_ml_covariance_whatever_the_var_reports(
        self, frame, fit
    ):
        small = sp.var(frame, variables=NAMES, lags=P, se_df="r")
        pattern = np.tril(np.full((K, K), NAN))
        np.testing.assert_allclose(
            sp.svar(small, B=pattern).impact.to_numpy(),
            sp.svar(fit, B=pattern).impact.to_numpy(),
            atol=1e-10,
        )

    @pytest.mark.parametrize(
        "kw, exc, match",
        [
            ({}, MethodIncompatibility, "exactly one kind"),
            (
                {"B": np.eye(3), "long_run": np.eye(3)},
                MethodIncompatibility,
                "exactly one kind",
            ),
            ({"B": np.eye(2)}, MethodIncompatibility, "3 x 3 matrix"),
            ({"B": np.eye(3)}, MethodIncompatibility, "nothing to estimate"),
            ({"B": np.full((3, 3), NAN)}, IdentificationFailure, "9 free parameters"),
            ({"sign": {"s": {"q": "+"}}}, MethodIncompatibility, "not a variable"),
            ({"sign": {"s": {"a": "up"}}}, MethodIncompatibility, "must be '\\+' or"),
            (
                {"sign": {f"s{i}": {"a": "+"} for i in range(4)}},
                MethodIncompatibility,
                "4 shocks for 3",
            ),
            (
                {"B": np.tril(np.full((3, 3), NAN)), "alpha": 1.5},
                MethodIncompatibility,
                "alpha must be in",
            ),
        ],
    )
    def test_restriction_errors(self, fit, kw, exc, match):
        with pytest.raises(exc, match=match):
            sp.svar(fit, **kw)

    def test_result_method_errors(self, fit):
        res = sp.svar(fit, B=np.tril(np.full((K, K), NAN)))
        with pytest.raises(MethodIncompatibility, match="not None or 'bootstrap'"):
            res.irf(2, ci="asymptotic")
        with pytest.raises(MethodIncompatibility, match="non-negative"):
            res.irf(-1)
        with pytest.raises(MethodIncompatibility, match="boot must be"):
            res.irf(2, ci="bootstrap", boot="x", reps=30)
        with pytest.raises(MethodIncompatibility, match="too few"):
            res.fevd(2, ci="bootstrap", reps=3)
        sign = sp.svar(fit, sign={"s1": {"a": "+"}}, n_draws=20, seed=0)
        with pytest.raises(MethodIncompatibility, match="identified set"):
            sign.irf(2, ci="bootstrap")
        with pytest.raises(MethodIncompatibility, match="fitted by sp.var"):
            sp.svar(fit.sigma_u, B=np.eye(3))


class TestDegenerateInputs:
    def test_missing_row_is_refused_or_its_lags_are_left_out(self, frame):
        hole = frame.copy()
        hole.loc[70, "b"] = np.nan
        try:
            v = sp.var(hole, variables=NAMES, lags=1)
        except StatsPAIError:
            return
        Y = hole[NAMES].to_numpy()
        ok = ~np.isnan(Y).any(axis=1)
        rows = np.flatnonzero(ok[1:] & ok[:-1]) + 1  # t and t - 1 observed
        X = np.column_stack([Y[rows - 1], np.ones(len(rows))])
        B = np.linalg.lstsq(X, Y[rows], rcond=None)[0]
        assert v.n_obs == len(rows)
        np.testing.assert_allclose(v.coefs["a"]["coef"], B[:, 0], atol=1e-9)

    def test_unknown_trend_is_named(self, frame):
        with pytest.raises(StatsPAIError, match="trend"):
            sp.var(frame, variables=NAMES, lags=1, trend="t")

    def test_saturated_var_is_refused(self, frame):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(StatsPAIError):
                sp.var(frame.iloc[:5], variables=NAMES, lags=2)

    def test_collinear_variables_are_refused(self, frame):
        twin = frame.assign(d=2 * frame["a"])
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            with pytest.raises(StatsPAIError):
                sp.var(twin, variables=["a", "b", "d"], lags=1)

    def test_zero_lags_is_named(self, frame):
        with pytest.raises(StatsPAIError, match="lags"):
            sp.var(frame, variables=NAMES, lags=0)

    def test_irf_argument_errors_are_named(self, frame):
        v = sp.var(frame, variables=NAMES, lags=1)
        with pytest.raises(StatsPAIError, match="impulse"):
            v.irf(3, impulse="q")
        with pytest.raises(StatsPAIError, match="periods"):
            v.irf(-1)


class TestGaps:
    def test_rows_missing_at_either_end_are_trimmed(self, frame):
        ragged = frame.copy()
        ragged.loc[[0, 1, 149], "a"] = np.nan
        v = sp.var(ragged, variables=NAMES, lags=1)
        ref = sp.var(frame.iloc[2:149], variables=NAMES, lags=1)
        np.testing.assert_array_equal(
            v.coefs["a"].to_numpy(), ref.coefs["a"].to_numpy()
        )

    def test_error_says_how_many_rows_and_where(self, frame):
        hole = frame.copy()
        hole.loc[[70, 90], "b"] = np.nan
        with pytest.raises(MethodIncompatibility, match="2 row.*position 70"):
            sp.var(hole, variables=NAMES, lags=1)
        with pytest.raises(MethodIncompatibility, match="2 row.*position 70"):
            sp.varsoc(hole, NAMES, maxlag=2)

    def test_exog_column_that_repeats_the_constant_is_refused(self, frame):
        with pytest.raises(MethodIncompatibility, match="collinear"):
            sp.var(frame.assign(one=1.0), variables=NAMES, lags=1, exog=["one"])


class TestArgumentRefusals:
    @pytest.mark.parametrize("rows", [2, 3])
    def test_no_more_rows_than_lags_is_refused(self, frame, rows):
        with pytest.raises(DataInsufficient, match=f"{rows} complete observation"):
            sp.var(frame.iloc[:rows], variables=NAMES, lags=3)

    @pytest.mark.parametrize("lags", [1.5, True, "2"])
    def test_lags_must_be_an_integer(self, frame, lags):
        with pytest.raises((MethodIncompatibility, TypeError), match="lags|str"):
            sp.var(frame, variables=NAMES, lags=lags)

    def test_irf_names_an_unknown_response_and_a_fractional_horizon(self, frame):
        v = sp.var(frame, variables=NAMES, lags=1)
        with pytest.raises(MethodIncompatibility, match="response='q'"):
            sp.irf(v, response="q")
        with pytest.raises(MethodIncompatibility, match="periods"):
            sp.irf(v, periods=2.5)
