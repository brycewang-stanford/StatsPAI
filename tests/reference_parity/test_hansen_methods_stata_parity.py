"""Stata parity for the methods added or corrected while walking through
Hansen's *Econometrics* (2022).

The reference numbers are real Stata 18 output on three committed synthetic
datasets (``_fixtures/_generate_hansen_methods_stata.do`` reads the same CSV
bytes): ``cnsreg``, ``nl``, ``pca``, ``factor``, ``estat overid`` after 2SLS
and LIML, collinear instruments, ``jackknife``, ``xtreg, re`` with regressors
that do not vary within panel, ``xthtaylor``, ``var`` / ``varsoc`` with exogenous
variables, ``irf table`` and ``dfuller`` p-values of explosive series.
Model averaging is compared with R's ``quadprog`` on the same bytes.

Tolerances
----------
* 1e-9 relative by default: the quantities are closed-form given the data.
* 1e-6 / 1e-5 for the parameters / standard errors of ``nl``: both programs
  iterate, and both differentiate the regression function numerically.
* 1e-6 for jackknife standard errors: Stata keeps the leave-one-out values
  in single precision unless told otherwise, so its variance carries about
  seven digits.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import statspai as sp

FIX = Path(__file__).parent / "_fixtures"
PCA_VARS = ["y", "endog", "x1", "x2", "z1", "z2"]


@pytest.fixture(scope="module")
def G():
    table = pd.read_csv(FIX / "hansen_methods_Stata.csv", skipinitialspace=True)
    values = pd.to_numeric(table["value"], errors="coerce")  # Stata's "." is NaN
    return {k: float(v) for k, v in zip(table["key"], values)}


@pytest.fixture(scope="module")
def cs():
    data = pd.read_csv(FIX / "textbook_cs.csv")
    data["xp"] = np.exp(data["x1"] / 2)
    data["ynl"] = 2 + 3 * data["xp"] ** 0.5 + data["z2"] / 4
    data["z3"] = data["z1"] + data["z2"]
    data["z4"] = 2 * data["x1"] - 1
    return data


@pytest.fixture(scope="module")
def panel():
    data = pd.read_csv(FIX / "textbook_panel.csv")
    data["ti"] = (data["id"] % 3).astype(float)
    data["hi"] = (data["id"] > 20).astype(float)
    return data


@pytest.fixture(scope="module")
def ts():
    return pd.read_csv(FIX / "textbook_ts.csv")


def close(ours, stata, rtol=1e-9, atol=0.0):
    assert np.isclose(ours, stata, rtol=rtol, atol=atol), (ours, stata)


# ------------------------------------------------- constrained least squares
@pytest.mark.parametrize(
    "tag, kwargs",
    [("ols", {}), ("rob", {"vce": "robust"}), ("clu", {"cluster": "g"})],
)
def test_cnsreg_matches_stata(G, cs, tag, kwargs):
    fit = sp.cnsreg("y ~ x1 + x2 + d", cs, ["x1 + x2 = 1", "d = 0.5"], **kwargs)
    close(fit.params["x1"], G[f"cns.{tag}.b_x1"])
    close(fit.params["x2"], G[f"cns.{tag}.b_x2"])
    close(fit.params["Intercept"], G[f"cns.{tag}.b_cons"])
    close(fit.std_errors["x1"], G[f"cns.{tag}.se_x1"])
    close(fit.std_errors["Intercept"], G[f"cns.{tag}.se_cons"])
    close(fit.diagnostics["Root MSE"], G[f"cns.{tag}.rmse"])
    assert fit.params["d"] == pytest.approx(0.5, abs=1e-12)
    assert fit.std_errors["d"] == pytest.approx(0.0, abs=1e-9)
    if tag != "ols":  # Stata leaves the classical F of this model missing
        close(fit.diagnostics["F-statistic"], G[f"cns.{tag}.F"])


def test_minimum_distance_is_cls_under_the_classical_covariance(cs):
    """With V = s^2 (X'X)^{-1} the minimum distance projection is the
    constrained least squares estimator; with a robust V it is not."""
    from statspai.regression.cnsreg import _robust_cov

    cls = sp.cnsreg("y ~ x1 + x2 + d", cs, "x1 + x2 = 1")
    ols = sp.regress("y ~ x1 + x2 + d", data=cs)
    X, y = ols.data_info["X"], ols.data_info["y"]
    b = ols.params.to_numpy()
    V = _robust_cov(X, y - X @ b, "ols", None, X.shape[1])
    R = cls.model_info["R"]
    projected = b - V @ R.T @ np.linalg.solve(R @ V @ R.T, R @ b - cls.model_info["c"])
    np.testing.assert_allclose(projected, cls.params.to_numpy(), rtol=1e-10)

    emd = sp.cnsreg("y ~ x1 + x2 + d", cs, "x1 + x2 = 1", method="emd")
    assert emd.params["x1"] + emd.params["x2"] == pytest.approx(1.0, abs=1e-12)
    assert abs(emd.params["x1"] - cls.params["x1"]) > 1e-6
    # the constraint direction has no variance left
    assert abs(R @ emd.data_info["var_cov"] @ R.T).max() < 1e-12


def test_cnsreg_refuses_what_it_cannot_fit(cs):
    with pytest.raises(sp.exceptions.MethodIncompatibility):
        sp.cnsreg("y ~ x1 + x2", cs, ["x1 = 1", "x1 = 2"])
    with pytest.raises(sp.exceptions.MethodIncompatibility):
        sp.cnsreg("y ~ x1", cs, ["x1 = 1", "Intercept = 0"])
    with pytest.raises(sp.exceptions.MethodIncompatibility):
        sp.cnsreg("y ~ x1 + x2", cs, "x1 = x2", method="gls")


# -------------------------------------------------- nonlinear least squares
@pytest.mark.parametrize(
    "tag, kwargs",
    [("ols", {}), ("rob", {"vce": "robust"}), ("clu", {"cluster": "g"})],
)
def test_nls_matches_stata_nl(G, cs, tag, kwargs):
    fit = sp.nls(
        "ynl ~ {a} + {b} * xp^{c}", cs, start={"a": 1, "b": 1, "c": 1}, **kwargs
    )
    for name in ("a", "b", "c"):
        close(fit.params[name], G[f"nl.{tag}.{name}"], rtol=1e-6)
        close(fit.std_errors[name], G[f"nl.{tag}.se_{name}"], rtol=1e-5)
    close(fit.diagnostics["Residual SS"], G[f"nl.{tag}.rss"], rtol=1e-10)
    close(fit.diagnostics["R-squared"], G[f"nl.{tag}.r2"], rtol=1e-10)
    close(fit.diagnostics["Root MSE"], G[f"nl.{tag}.rmse"], rtol=1e-10)
    assert fit.model_info["has_constant"] and fit.model_info["converged"]


def test_nls_formula_and_function_agree(cs):
    by_formula = sp.nls("ynl ~ {a} + {b} * xp^{c=1}", cs, vce="robust")
    by_function = sp.nls(
        lambda p, d: p["a"] + p["b"] * d["xp"] ** p["c"],
        cs,
        start={"a": 1.0, "b": 1.0, "c": 1.0},
        y="ynl",
        vce="robust",
    )
    np.testing.assert_allclose(by_formula.params, by_function.params, rtol=1e-8)
    np.testing.assert_allclose(by_formula.std_errors, by_function.std_errors, rtol=1e-6)
    # post-estimation works on the result
    assert np.isfinite(sp.nlcom(by_formula, "_b[a] + _b[b]")["estimate"])


def test_nls_fails_loudly(cs):
    with pytest.raises(sp.exceptions.MethodIncompatibility):
        sp.nls("ynl ~ a + b * xp", cs)  # no parameter in braces
    # a parameter may share its name with a column (the fixture has `b`)
    assert "b" in cs.columns
    with pytest.raises(sp.exceptions.MethodIncompatibility):
        sp.nls("ynl ~ {a} + {b} * xp", cs, start={"zz": 1})
    with pytest.raises(sp.exceptions.NumericalInstability):
        # c is not identified when the regressor is a constant
        sp.nls("ynl ~ {a} + {b} * 1^{c}", cs)


# ------------------------------------------- principal components, factors
def test_pca_matches_stata(G, cs):
    res = sp.pca(cs, PCA_VARS)
    for j in range(6):
        close(res.eigenvalues["eigenvalue"].iloc[j], G[f"pca.ev{j + 1}"])
        close(res.loadings.iloc[j, 0], G[f"pca.l{j + 1}1"])
        close(res.loadings.iloc[j, 1], G[f"pca.l{j + 1}2"])
    assert res.eigenvalues["proportion"].sum() == pytest.approx(1.0)
    scores = res.scores(cs)
    # scores are uncorrelated with variances equal to the eigenvalues
    np.testing.assert_allclose(
        np.cov(scores.to_numpy(), rowvar=False),
        np.diag(res.eigenvalues["eigenvalue"]),
        atol=1e-10,
    )


def test_principal_factors_match_stata(G, cs):
    res = sp.factor(cs, PCA_VARS)
    assert res.n_factors == int(G["pf.k"])
    for j in range(6):
        close(res.eigenvalues["eigenvalue"].iloc[j], G[f"pf.ev{j + 1}"])
        close(res.loadings.iloc[j, 0], G[f"pf.l{j + 1}1"])
        close(res.uniqueness.iloc[j], G[f"pf.u{j + 1}"])


def test_principal_component_factors_match_stata(G, cs):
    res = sp.factor(cs, PCA_VARS, method="pcf", n_factors=2)
    for j in range(6):
        close(res.loadings.iloc[j, 0], G[f"pcf.l{j + 1}1"])
        close(res.loadings.iloc[j, 1], G[f"pcf.l{j + 1}2"])
        close(res.uniqueness.iloc[j], G[f"pcf.u{j + 1}"])


def test_ml_factor_statistics_and_recovery(G, cs):
    """The test of independence does not depend on the factor solution, so
    it is compared on the fixture (whose one-factor solution is a Heywood
    case, where programs differ by where they stop). The loadings are
    checked on data that follow a two-factor model."""
    res = sp.factor(cs, PCA_VARS, method="ml", n_factors=1)
    close(res.lr_independence["chi2"], G["ml.chi2_i"])
    assert res.lr_factors["df"] == G["ml.df_1"]

    rng = np.random.default_rng(3)
    load = np.array(
        [[0.8, 0], [0.7, 0.2], [0.6, 0.3], [0, 0.8], [0.2, 0.7], [0.3, 0.6]]
    )
    f = rng.normal(size=(20000, 2))
    x = f @ load.T + rng.normal(size=(20000, 6)) * np.sqrt(1 - (load**2).sum(1))
    fit = sp.factor(pd.DataFrame(x, columns=list("abcdef")), method="ml", n_factors=2)
    implied = fit.loadings.to_numpy() @ fit.loadings.to_numpy().T
    np.testing.assert_allclose(
        implied[np.triu_indices(6, 1)],
        (load @ load.T)[np.triu_indices(6, 1)],
        atol=0.03,
    )
    np.testing.assert_allclose(fit.uniqueness, 1 - (load**2).sum(1), atol=0.03)
    assert not fit.heywood and fit.lr_factors["pvalue"] > 0.001


# ------------------------------------------------------- overidentification
def test_overid_statistics_after_2sls_and_liml(G, cs):
    formula = "y ~ x1 + x2 + (endog ~ z1 + z2 + b)"
    plain = sp.estat(sp.iv(formula, data=cs, method="2sls"), "overid")
    close(plain["statistic"], G["oid.sargan"])
    close(plain["pvalue"], G["oid.p_sargan"])
    close(plain["iid_errors"]["basmann"], G["oid.basmann"])
    close(plain["iid_errors"]["basmann_pvalue"], G["oid.p_basmann"])
    assert "score" not in plain

    robust = sp.estat(sp.iv(formula, data=cs, method="2sls", robust="robust"), "overid")
    close(robust["score"], G["oid.score"])
    close(robust["score_pvalue"], G["oid.p_score"])
    # the i.i.d. statistics stay available after a robust fit
    close(robust["iid_errors"]["sargan"], G["oid.sargan"])

    liml = sp.estat(sp.iv(formula, data=cs, method="liml"), "overid")
    close(liml["iid_errors"]["anderson_rubin"], G["oid.ar"])
    close(liml["iid_errors"]["anderson_rubin_pvalue"], G["oid.p_ar"])


def test_collinear_instruments_are_dropped_with_a_warning(G, cs):
    with pytest.warns(sp.exceptions.StatsPAIWarning, match="omitted"):
        fit = sp.iv("y ~ x1 + x2 + (endog ~ z1 + z2 + z3 + z4)", data=cs, small=False)
    assert fit.model_info["omitted_instruments"] == ["z3", "z4"]
    assert fit.diagnostics["N instruments"] == 2
    close(fit.params["endog"], G["ivc.b_endog"])
    close(fit.std_errors["endog"], G["ivc.se_endog"])


def test_2sls_keeps_its_digits_on_an_ill_conditioned_design():
    """A just-identified model with a cubic in the instrument: Z'X has a
    condition number near 1e9. The data are integers and y = X b exactly, so
    the estimate must return b; a solve through (W'W)^{-1} lost seven
    digits."""
    z = np.arange(1.0, 41.0)
    frame = pd.DataFrame({"z1": z, "z2": z**2, "z3": z**3})
    shift = np.where(np.arange(40) % 2 == 0, 1.0, -1.0)
    frame["d1"] = frame.z1 + shift
    frame["d2"] = frame.z2 + 3 * shift
    frame["d3"] = frame.z3 - 2 * shift
    beta = np.array([34.0, -61.0, 29.0])
    frame["y"] = 5 + frame[["d1", "d2", "d3"]].to_numpy() @ beta
    assert np.linalg.cond(frame[["z1", "z2", "z3"]].T @ frame[["d1", "d2", "d3"]]) > 1e7
    fit = sp.iv("y ~ (d1 + d2 + d3 ~ z1 + z2 + z3)", data=frame)
    np.testing.assert_allclose(fit.params[["d1", "d2", "d3"]], beta, rtol=1e-8)


# ----------------------------------------------------------------- jackknife
def test_jackknife_matches_stata(G, cs):
    sub = cs[cs["g"] <= 3]
    assert len(sub) == G["jk.n"]
    fit = sp.jackknife(sub, lambda d: sp.regress("y ~ x1 + x2", data=d))
    close(fit.std_errors["x1"], G["jk.se_x1"], rtol=1e-6)
    close(fit.std_errors["x2"], G["jk.se_x2"], rtol=1e-6)
    close(fit.std_errors["Intercept"], G["jk.se_cons"], rtol=1e-6)
    assert fit.data_info["df_resid"] == len(sub) - 1

    def ratio(d):
        b = sp.regress("y ~ x1 + x2", data=d).params
        return b["x1"] / b["x2"]

    one = sp.jackknife(sub, ratio)
    close(one.estimate, G["jk.ratio"])
    close(one.se, G["jk.ratio_se"], rtol=1e-6)
    close(sp.jackknife(sub, ratio, mse=True).se, G["jk.ratio_se_mse"], rtol=1e-6)

    def s2(d):
        info = sp.regress("y ~ x1 + x2", data=d).data_info
        return info["rss"] / info["nobs"]

    by_cluster = sp.jackknife(sub, s2, cluster="g")
    close(by_cluster.estimate, G["jk.s2"])
    close(by_cluster.se, G["jk.s2_se"], rtol=1e-6)
    assert by_cluster.n_reps == 3


def test_jackknife_through_sp_stata(G, cs):
    sub = cs[cs["g"] <= 3]
    fit = sp.stata("regress y x1 x2, vce(jackknife)", data=sub)
    close(fit.std_errors["x1"], G["jk.se_x1"], rtol=1e-6)
    one = sp.stata("jackknife (_b[x1] / _b[x2]): regress y x1 x2", data=sub)
    close(one.se, G["jk.ratio_se"], rtol=1e-6)


# ----------------------------------------------- random effects, rank counts
def test_random_effects_with_time_invariant_regressors(G, panel):
    fit = sp.panel(
        panel, "y ~ x1 + x2 + ti + hi", entity="id", time="year", method="re",
        ssc="stata",
    )  # fmt: skip
    close(fit.params["x1"], G["re.b_x1"])
    close(fit.params["ti"], G["re.b_ti"])
    close(fit.params["hi"], G["re.b_hi"])
    close(fit.std_errors["x1"], G["re.se_x1"])
    close(fit.std_errors["ti"], G["re.se_ti"])
    # sigma_e is the within estimate, which the two swept-out regressors
    # cannot change: the fixed-effects fit without them gives the same number
    assert G["re.sigma_e"] == G["re.fe_sigma_e"]
    from statspai.panel.xt_tools import xt_statistics

    xt = xt_statistics(
        panel, "y", ["x1", "x2", "ti", "hi"], id="id",
        params=fit.params.to_dict(), method="re",
    )  # fmt: skip
    close(xt["sigma_e"], G["re.sigma_e"])
    close(xt["sigma_u"], G["re.sigma_u"])

    robust = sp.panel(
        panel, "y ~ x1 + x2 + ti + hi", entity="id", time="year", method="re",
        ssc="stata", robust="robust",
    )  # fmt: skip
    close(robust.std_errors["x1"], G["re.rob_se_x1"])
    close(robust.std_errors["ti"], G["re.rob_se_ti"])


# ------------------------------------------------------------ Hausman-Taylor
@pytest.fixture(scope="module")
def ht_panel(panel):
    data = panel.copy()
    data["zi"] = data.groupby("id")["w"].transform("mean") + 0.3 * data["ti"]
    return data


@pytest.mark.parametrize("tag, kwargs", [("conv", {}), ("rob", {"vce": "robust"})])
def test_hausman_taylor_matches_stata(G, ht_panel, tag, kwargs):
    fit = sp.xthtaylor(
        "y ~ x1 + x2 + w + ti + zi", ht_panel, id="id", endog=["x2", "zi"], **kwargs
    )
    for name in ("x1", "w", "x2", "ti", "zi"):
        close(fit.params[name], G[f"ht.{tag}.b_{name}"])
    close(fit.params["_cons"], G[f"ht.{tag}.b_cons"])
    for name in ("x1", "x2", "ti", "zi"):
        close(fit.std_errors[name], G[f"ht.{tag}.se_{name}"])
    close(fit.std_errors["_cons"], G[f"ht.{tag}.se_cons"])
    close(fit.model_info["sigma_u"], G[f"ht.{tag}.sigma_u"])
    close(fit.model_info["sigma_e"], G[f"ht.{tag}.sigma_e"])
    if tag == "conv":
        close(fit.diagnostics["Wald chi2"], G["ht.conv.chi2"])
    assert fit.model_info["tv_exogenous"] == ["x1", "w"]
    assert fit.model_info["ti_endogenous"] == ["zi"]
    # through the translator, with the panel declared by xtset
    via = sp.stata(
        "xtset id year\nxthtaylor y x1 x2 w ti zi, endog(x2 zi)"
        + (" vce(robust)" if kwargs else ""),
        data=ht_panel,
    )
    pd.testing.assert_series_equal(via.std_errors, fit.std_errors)


def test_hausman_taylor_does_not_depend_on_the_base_period(G, ht_panel):
    """With period dummies in an unbalanced panel Stata's estimate of the
    time-invariant endogenous coefficient moves with the year it leaves out
    (-0.248 without 2001, +0.015 without 2005 on these data). The fit here
    is the same for every base."""
    assert abs(G["ht.base2001.b_zi"] - G["ht.base2005.b_zi"]) > 0.2
    fits = [
        sp.xthtaylor(
            f"y ~ x1 + x2 + ti + zi + C(year, Treatment({base}))",
            ht_panel, id="id", endog=["x2", "zi"],
        )  # fmt: skip
        for base in (2001, 2005, 2008)
    ]
    for other in fits[1:]:
        for name in ("x1", "x2", "ti", "zi"):
            close(other.params[name], fits[0].params[name], rtol=1e-9)
            close(other.std_errors[name], fits[0].std_errors[name], rtol=1e-9)


def test_hausman_taylor_refuses_what_is_not_identified(ht_panel):
    with pytest.raises(sp.exceptions.MethodIncompatibility, match="under-identified"):
        sp.xthtaylor("y ~ x2 + ti + zi", ht_panel, id="id", endog=["x2", "zi"])
    with pytest.raises(sp.exceptions.MethodIncompatibility, match="endog"):
        sp.xthtaylor("y ~ x1 + ti", ht_panel, id="id", endog=[])
    with pytest.raises(sp.exceptions.MethodIncompatibility, match="constant within"):
        sp.xthtaylor("y ~ x1 + x2", ht_panel, id="id", endog=["x2"])
    with pytest.raises(sp.exceptions.MethodIncompatibility, match="not regressors"):
        sp.xthtaylor("y ~ x1 + ti", ht_panel, id="id", endog=["nope"])


# ------------------------------------------------- VAR with exogenous terms
def test_var_with_exogenous_variables(G, ts):
    fit = sp.var(ts, ["z1", "z2"], lags=2, exog=["x1", "d"])
    close(fit.log_likelihood, G["var.ll"])
    close(fit.coefs["z1"].loc["L1.z1", "coef"], G["var.b11"])
    close(fit.coefs["z1"].loc["x1", "coef"], G["var.b1x"])
    close(fit.coefs["z2"].loc["d", "coef"], G["var.b2d"])
    close(fit.coefs["z1"].loc["x1", "se"], G["var.se1x"])
    close(fit.aic, G["var.aic"])
    path = sp.irf(fit, periods=4, orthogonal=True)["irf"]["z1 -> z2"]
    for s in range(5):
        close(path[s], G[f"var.oirf{s}"])
    with pytest.raises(sp.exceptions.MethodIncompatibility, match="exogenous"):
        fit.forecast(2)

    table = sp.varsoc(ts, ["z1", "z2"], maxlag=3, exog=["x1", "d"])
    for p in range(4):
        close(table.loc[p, "LL"], G[f"soc.ll{p}"])
        close(table.loc[p, "AIC"], G[f"soc.aic{p}"])
        close(table.loc[p, "SBIC"], G[f"soc.sbic{p}"])


def test_irf_table_through_sp_stata(G, ts):
    out = sp.stata(
        """
        tsset t
        var z1 z2, lag(1/2) exog(x1 d)
        irf create m1, step(4)
        irf table oirf, impulse(z1) response(z2)
        """,
        data=ts,
    )
    for s in range(5):
        close(out["oirf"].loc[s], G[f"var.oirf{s}"])


# ------------------------------------------------------------ model averaging
def test_model_averaging_matches_quadprog(cs):
    """Criteria and weights against R (``quadprog::solve.QP``) on the same
    bytes, with the definitions of the programs of Hansen's chapter 28
    (``_fixtures/_generate_hansen_model_average.R``)."""
    table = pd.read_csv(FIX / "hansen_model_average_R.csv")
    R = dict(zip(table["key"], table["value"]))
    formulas = [
        "y ~ x1",
        "y ~ x1 + x2",
        "y ~ x1 + x2 + d",
        "y ~ x1 + x2 + d + z1 + z2",
        "y ~ x1 + I(x1**2) + x2 + I(x2**2) + d + z1 + z2",
    ]
    res = sp.model_average(formulas, cs, method="mma")
    for m in range(5):
        row = res.table.iloc[m]
        close(row["aic"], R[f"aic{m + 1}"], rtol=1e-12)
        close(row["bic"], R[f"bic{m + 1}"], rtol=1e-12)
        close(row["cv"], R[f"cv{m + 1}"], rtol=1e-11)
        close(row["w_aic"], R[f"waic{m + 1}"], rtol=1e-9)
        close(row["w_bic"], R[f"wbic{m + 1}"], rtol=1e-9, atol=1e-300)
        # quadratic programmes: both are exact on the support
        close(row["w_mma"], R[f"wmma{m + 1}"], rtol=1e-8, atol=1e-12)
        close(row["w_jma"], R[f"wjma{m + 1}"], rtol=1e-8, atol=1e-12)
    assert res.weights.sum() == pytest.approx(1.0, abs=1e-12)
    # the averaged coefficients and fit are the weighted candidates
    by_hand = sum(
        w * fit.params.reindex(res.params.index).fillna(0.0)
        for w, fit in zip(res.weights, res.fits)
    )
    np.testing.assert_allclose(res.params, by_hand, rtol=1e-12)
    np.testing.assert_allclose(res.predict(cs), res.fitted, rtol=1e-10)
    slope = res.average(lambda fit: fit.params["x1"])
    assert slope == pytest.approx(
        sum(w * f.params["x1"] for w, f in zip(res.weights, res.fits))
    )


def test_model_averaging_weights_are_the_minimum(cs):
    """No point of the simplex does better than the returned weights."""
    formulas = ["y ~ x1", "y ~ x1 + x2", "y ~ x1 + x2 + d", "y ~ x1 + d + z1"]
    res = sp.model_average(formulas, cs)
    E = np.column_stack([f.data_info["residuals"] for f in res.fits])
    lev = [(np.linalg.qr(f.data_info["X"])[0] ** 2).sum(axis=1) for f in res.fits]
    Rm = E / (1 - np.column_stack(lev))
    best = res.table["w_jma"].to_numpy()
    value = best @ Rm.T @ Rm @ best
    rng = np.random.default_rng(0)
    for w in rng.dirichlet(np.ones(4), size=2000):
        assert w @ Rm.T @ Rm @ w >= value - 1e-9
    assert res.selected["cv"] == res.table["cv"].idxmin()
    with pytest.raises(sp.exceptions.MethodIncompatibility):
        sp.model_average(["y ~ x1"], cs)
    with pytest.raises(sp.exceptions.MethodIncompatibility):
        sp.model_average(["y ~ x1", "endog ~ x1"], cs)


# -------------------------------------------- Dickey-Fuller upper-tail p-value
def test_dickey_fuller_p_value_of_an_explosive_series(G, ts):
    """Above the fitted range of MacKinnon's surface the p-value is 1; the
    cubic left on its own turns down (0.06 at +3.25 with a trend)."""
    from statspai.panel.unit_root import mackinnon1994_pvalue

    e1 = np.zeros(len(ts))
    z1 = ts["z1"].to_numpy()
    e1[0] = z1[0]
    for t in range(1, len(ts)):
        e1[t] = 1.08 * e1[t - 1] + z1[t]
    frame = pd.DataFrame({"e1": e1})
    with_trend = sp.unitroot(frame, y="e1", test="adf", lags=1, trend="ct")
    close(with_trend.statistic, G["df.ct_stat"], rtol=1e-7)
    assert with_trend.pvalue == G["df.ct_p"] == 1.0
    constant = sp.unitroot(frame, y="e1", test="adf", lags=1, trend="c")
    close(constant.statistic, G["df.c_stat"], rtol=1e-7)
    assert constant.pvalue == G["df.c_p"] == 1.0
    # inside the range nothing changes (values printed by Stata 18)
    close(mackinnon1994_pvalue(0.48008745, "ct"), 0.9968134395, rtol=1e-9)
    close(mackinnon1994_pvalue(2.67036504, "c"), 0.9990848786, rtol=1e-9)
    assert mackinnon1994_pvalue(3.25, "ct") == 1.0


# ------------------------------------------- dynamic panel with period dummies
def _dpd(data, **kwargs):
    # Stata's xtdpd: the MA(1) weight on the differenced rows (h=2) and each
    # iv() variable differenced in one equation and in levels in the other
    return sp.xtdpdsys(
        data,
        y="n",
        id="id",
        time="year",
        lags=2,
        time_dummies=True,
        h=2,
        iv_equation="both",
        **kwargs,
    )


def test_system_gmm_with_period_dummies_matches_stata_xtdpd(G):
    """Two lags take the first periods out of the differenced equation. The
    dummies of those periods, and one more because the rest sum to the
    constant, leave the regressors; before, all of them stayed and the fit
    went through a pseudo-inverse (L1.n 1.1548 where Stata prints 1.1665)."""
    data = pd.read_csv(FIX / "dynpanel_abdata.csv")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        one = _dpd(data, twostep=False, robust=False)
        rob = _dpd(data, twostep=False, robust=True)
        two = _dpd(data, twostep=True, robust=True)
    t = one.detail.set_index("variable")
    assert list(t.index) == ["L1.n", "L2.n"] + [
        f"_T{year}" for year in range(1979, 1984)
    ] + ["_cons"]
    assert len(t) == G["dpd.one.k"]
    assert one.model_info["n_instruments"] == G["dpd.one.zrank"]
    close(t.loc["L1.n", "coefficient"], G["dpd.one.b_l1"])
    close(t.loc["L2.n", "coefficient"], G["dpd.one.b_l2"])
    close(t.loc["_cons", "coefficient"], G["dpd.one.b_cons"])
    close(t.loc["_T1979", "coefficient"], G["dpd.one.b_yr1979"])
    close(t.loc["_T1983", "coefficient"], G["dpd.one.b_yr1983"])
    close(t.loc["L1.n", "se"], G["dpd.one.se_l1"])
    close(t.loc["_cons", "se"], G["dpd.one.se_cons"])
    close(one.model_info["sargan_stat"], G["dpd.one.sargan"])
    r = rob.detail.set_index("variable")
    close(r.loc["L1.n", "se"], G["dpd.rob.se_l1"])
    close(r.loc["L2.n", "se"], G["dpd.rob.se_l2"])
    close(r.loc["_cons", "se"], G["dpd.rob.se_cons"])
    w = two.detail.set_index("variable")
    close(w.loc["L1.n", "coefficient"], G["dpd.two.b_l1"], rtol=1e-8)
    close(w.loc["L2.n", "coefficient"], G["dpd.two.b_l2"], rtol=1e-8)
    close(w.loc["_cons", "coefficient"], G["dpd.two.b_cons"], rtol=1e-8)
    close(w.loc["L1.n", "se"], G["dpd.two.se_l1"], rtol=1e-8)
    close(w.loc["_cons", "se"], G["dpd.two.se_cons"], rtol=1e-8)


def test_two_step_gmm_with_a_singular_weight_matches_stata(G):
    """Six firms are observed in the first two years and the moments dated
    there outnumber them, so the two-step weight is a generalized inverse
    and the estimate depends on which one. Stata sweeps on the largest
    remaining diagonal (Mata's `invsym`) and leaves the moments it set to
    zero out of the one-step covariance in Windmeijer's correction. With the
    Moore-Penrose inverse the first lag is 1.1638 here, not 1.1574."""
    data = pd.read_csv(FIX / "dynpanel_abdata.csv")
    data = data[~((data["year"] < 1978) & (data["id"] > 6))]

    def fit(robust):
        return sp.xtdpdsys(
            data,
            y="n",
            id="id",
            time="year",
            lags=2,
            h=2,
            iv_equation="both",
            twostep=True,
            robust=robust,
        )

    with pytest.warns(UserWarning, match="two-step weight matrix.*singular"):
        conventional = fit(False)
    with pytest.warns(UserWarning, match="two-step weight matrix.*singular"):
        corrected = fit(True)
    t = conventional.detail.set_index("variable")
    close(t.loc["L1.n", "coefficient"], G["dpd.sing.b_l1"], rtol=1e-9)
    close(t.loc["L2.n", "coefficient"], G["dpd.sing.b_l2"], rtol=1e-9)
    close(t.loc["_cons", "coefficient"], G["dpd.sing.b_cons"], rtol=1e-9)
    close(t.loc["L1.n", "se"], G["dpd.sing.se_l1"], rtol=1e-9)
    close(t.loc["_cons", "se"], G["dpd.sing.se_cons"], rtol=1e-9)
    # Stata's two-step Sargan statistic is Hansen's J
    close(conventional.model_info["hansen_stat"], G["dpd.sing.sargan"], rtol=1e-9)
    c = corrected.detail.set_index("variable")
    close(c.loc["L1.n", "se"], G["dpd.sing.wc_l1"], rtol=1e-9)
    close(c.loc["L2.n", "se"], G["dpd.sing.wc_l2"], rtol=1e-9)
    close(c.loc["_cons", "se"], G["dpd.sing.wc_cons"], rtol=1e-9)


def test_xtdpd_through_sp_stata(G):
    """`dgmmiv()` variables are instrumented GMM-style, `iv()` variables are
    their own instruments, `i.year` in both is the set of period dummies."""
    data = pd.read_csv(FIX / "dynpanel_abdata.csv")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        fit = sp.stata(
            """
            xtset id year
            xi: xtdpd L(0/2).n L(0/1).w k i.year, iv(k i.year) ///
                dgmmiv(n w, lag(2 4)) lgmmiv(n w) twostep vce(robust)
            """,
            data=data,
        )
    t = fit.detail.set_index("variable")
    assert fit.model_info["n_instruments"] == G["dpd.endo.zrank"]
    for ours, key in (
        ("L1.n", "l1"),
        ("L2.n", "l2"),
        ("w", "w"),
        ("L1.w", "w_l1"),
        ("k", "k"),
        ("_cons", "cons"),
    ):
        close(t.loc[ours, "coefficient"], G[f"dpd.endo.b_{key}"], rtol=1e-8)
    close(t.loc["L1.n", "se"], G["dpd.endo.se_l1"], rtol=1e-8)
    close(t.loc["w", "se"], G["dpd.endo.se_w"], rtol=1e-8)
    close(t.loc["k", "se"], G["dpd.endo.se_k"], rtol=1e-8)
    close(t.loc["_cons", "se"], G["dpd.endo.se_cons"], rtol=1e-8)
    # the one-line form carries the panel as i() t()
    one = sp.from_stata(
        "xtdpd n n_L1 k, iv(k) dgmmiv(n, lagrange(2 3)) lgmmiv(n) i(id) t(year)"
    )
    assert one["arguments"] == {
        "y": "n",
        "id": "id",
        "lags": 1,
        "gmm_lags": (2, 3),
        "time": "year",
        "x": ["k"],
        "twostep": False,
        "robust": False,
        "h": 2,
        "iv_equation": "both",
    }
    for line, why in (
        ("xtdpd L(0/1).n, dgmmiv(n)", "lgmmiv"),
        ("xtdpd L(0/1).n k, dgmmiv(n) lgmmiv(n)", "has to be in iv"),
        ("xtdpd L(0/1).n, dgmmiv(n) lgmmiv(n) noconstant", "noconstant"),
        ("xtdpd L(0/1).n, dgmmiv(n) lgmmiv(n, lag(2))", "lag"),
        ("xtdpd L(0/1).n k, dgmmiv(n) lgmmiv(n) div(k)", "div"),
        ("xtdpd L(0/1).n, dgmmiv(n) lgmmiv(n) artests(3)", "artests"),
    ):
        with pytest.raises(sp.exceptions.MethodIncompatibility, match=why):
            sp.stata("xtset id year\n" + line, data=data)


def test_a_weight_matrix_singular_to_rounding_is_reported():
    """`inv` raises only on an exactly zero pivot. Three firms observed early
    carry more moments than firms, the two-step weight has no inverse, and
    what `inv` returned for it gave standard errors of order 1e14."""
    from statspai.gmm._dynpanel._estimate import safe_inv, sweep_ginv

    rng = np.random.default_rng(17)
    g = rng.normal(size=(3, 6))
    m = g.T @ g
    with pytest.warns(UserWarning, match="singular"):
        a = safe_inv(m, "weight")
    # a generalized inverse (M A M = M) of the rank of M, zero on the
    # moments it dropped
    np.testing.assert_allclose(m @ a @ m, m, atol=1e-9)
    assert np.linalg.matrix_rank(a) == 3
    assert int((np.diag(a) == 0).sum()) == 3
    np.testing.assert_array_equal(a, sweep_ginv(m))
    full = rng.normal(size=(40, 6))
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        np.testing.assert_allclose(
            safe_inv(full.T @ full, "weight"), np.linalg.inv(full.T @ full)
        )
    np.testing.assert_allclose(
        sweep_ginv(full.T @ full), np.linalg.inv(full.T @ full), rtol=1e-9
    )


# ------------------------------------------------- threshold and kink models
@pytest.fixture(scope="module")
def thr(ts):
    data = ts.copy()
    data["yt"] = 1 + 0.5 * data.x1 + (data.x2 > 0.3) * (1 + data.x1) + 0.5 * data.z1
    data["yk"] = (
        1
        + 0.5 * data.x1
        - np.minimum(data.x2 - 0.3, 0)
        + 2 * np.maximum(data.x2 - 0.3, 0)
        + 0.3 * data.z1
    )
    return data


def _region_variance(fit, name):
    """Variance of a coefficient above the threshold: base plus change."""
    names = list(fit.params.index)
    V = fit.data_info["var_cov"]
    i = names.index(name)
    j = names.index("above" if name == "Intercept" else f"above:{name}")
    return V[i, i] + V[j, j] + 2 * V[i, j]


def test_threshold_regression_matches_stata_threshold(G, thr):
    """Stata searches the sample values that leave floor(n * trim)
    observations on each side; its default covariance is the classical one
    and its `vce(robust)` is HC0."""
    fit = sp.threshold("yt ~ x1", thr, "x2", vce="ols")
    regimes = fit.model_info["regimes"]
    assert fit.model_info["threshold"] == G["thr.ols.gamma"]
    close(fit.diagnostics["Residual SS"], G["thr.ols.ssr"], rtol=1e-12)
    close(regimes.loc["x1", "below"], G["thr.ols.b1_x1"], rtol=1e-11)
    close(regimes.loc["Intercept", "below"], G["thr.ols.b1_cons"], rtol=1e-11)
    close(regimes.loc["x1", "above"], G["thr.ols.b2_x1"], rtol=1e-11)
    close(regimes.loc["Intercept", "above"], G["thr.ols.b2_cons"], rtol=1e-11)
    close(fit.std_errors["x1"] ** 2, G["thr.ols.v1_x1"], rtol=1e-10)
    close(_region_variance(fit, "x1"), G["thr.ols.v2_x1"], rtol=1e-10)
    close(_region_variance(fit, "Intercept"), G["thr.ols.v2_cons"], rtol=1e-10)

    rob = sp.threshold("yt ~ x1 + z2", thr, "x2", regime=["x1"], trim=0.15, vce="hc0")
    regimes = rob.model_info["regimes"]
    assert rob.model_info["threshold"] == G["thr.rob.gamma"]
    close(rob.diagnostics["Residual SS"], G["thr.rob.ssr"], rtol=1e-12)
    close(rob.params["z2"], G["thr.rob.b_z2"], rtol=1e-11)
    close(regimes.loc["x1", "below"], G["thr.rob.b1_x1"], rtol=1e-11)
    close(regimes.loc["Intercept", "below"], G["thr.rob.b1_cons"], rtol=1e-11)
    close(regimes.loc["x1", "above"], G["thr.rob.b2_x1"], rtol=1e-11)
    close(regimes.loc["Intercept", "above"], G["thr.rob.b2_cons"], rtol=1e-11)
    close(rob.std_errors["z2"] ** 2, G["thr.rob.v_z2"], rtol=1e-10)
    close(rob.std_errors["x1"] ** 2, G["thr.rob.v1_x1"], rtol=1e-10)
    close(_region_variance(rob, "x1"), G["thr.rob.v2_x1"], rtol=1e-10)
    close(_region_variance(rob, "Intercept"), G["thr.rob.v2_cons"], rtol=1e-10)
    close(regimes.loc["x1", "se_above"] ** 2, G["thr.rob.v2_x1"], rtol=1e-10)
    close(regimes.loc["x1", "se_below"] ** 2, G["thr.rob.v1_x1"], rtol=1e-10)


def test_threshold_through_sp_stata(G, thr):
    fit = sp.stata(
        """
        tsset t
        threshold yt z2, regionvars(x1) threshvar(x2) trim(15) vce(robust)
        """,
        data=thr,
    )
    assert fit.model_info["threshold"] == G["thr.rob.gamma"]
    close(fit.std_errors["z2"] ** 2, G["thr.rob.v_z2"], rtol=1e-10)
    with pytest.raises(sp.exceptions.MethodIncompatibility, match="nthresholds"):
        sp.stata("tsset t\nthreshold yt, threshvar(x2) nthresholds(2)", data=thr)


def test_regression_kink_matches_stata_nl(G, thr):
    """The kink point is root-n normal jointly with the slopes, so the fit is
    nonlinear least squares and `nl` is the reference (run with eps(1e-14);
    its estimate of the kink point is converged to about 1e-9)."""
    fit = sp.threshold("yk ~ x1", thr, "x2", kink=True)
    for ours, key in (
        ("x2:below", "below"),
        ("x2:above", "above"),
        ("x1", "x1"),
        ("Intercept", "cons"),
    ):
        close(fit.params[ours], G[f"kink.b_{key}"], rtol=1e-7)
        close(fit.std_errors[ours], G[f"kink.se_{key}"], rtol=1e-7)
    close(fit.params["threshold"], G["kink.gamma"], rtol=1e-7)
    close(fit.std_errors["threshold"], G["kink.se_gamma"], rtol=1e-7)
    close(fit.diagnostics["Residual SS"], G["kink.rss"], rtol=1e-12)
    assert fit.diagnostics["Residual SS"] <= G["kink.rss"] * (1 + 1e-14)
