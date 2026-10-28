"""Tests for sp.positive_correlation_test, sp.efc_welfare and sp.pooled_price_ratio.

The Stata ``biprobit`` parity of the bivariate-probit version lives in
``tests/reference_parity/test_positive_correlation_test_stata_parity.py``.
Here every assertion is a known truth: an identity (the residual score is
the derivative of the bivariate probit likelihood), a closed form (linear
curves built exactly into the data; conditional means of uniform and beta
distributions) or an equivalence (weights against duplicated rows).
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import special, stats

import statspai as sp
from statspai.exceptions import (
    AssumptionWarning,
    ColumnNotFound,
    DataInsufficient,
    IdentificationFailure,
    MethodIncompatibility,
)

# ---------------------------------------------------------------------
# positive_correlation_test
# ---------------------------------------------------------------------


def _pcp_data(n: int = 1500, rho: float = 0.0, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    z = rng.normal(size=n)
    u = rng.normal(size=n)
    a = np.sqrt(abs(rho))
    e1 = a * u + np.sqrt(1 - a**2) * rng.normal(size=n)
    e2 = np.sign(rho) * a * u + np.sqrt(1 - a**2) * rng.normal(size=n)
    return pd.DataFrame(
        {
            "x": x,
            "z": z,
            "cover": (0.2 + 0.5 * x + e1 > 0).astype(int),
            "claim": (-0.4 + 0.3 * x + 0.4 * z + e2 > 0).astype(int),
            "w": rng.integers(1, 4, size=n).astype(float),
            "g": rng.integers(0, 60, size=n),
        }
    )


def _probit_index(df, y, x):
    fit = sp.probit(data=df, y=y, x=x)
    X = np.column_stack([np.ones(len(df)), df[x].to_numpy(float)])
    return X @ fit.params.reindex(["Intercept"] + x).to_numpy()


def test_residual_score_is_biprobit_rho_derivative_at_zero():
    # sum e1 e2 is d/d rho of the bivariate probit log likelihood at rho = 0,
    # evaluated at the separate probit estimates (central difference, h=1e-5;
    # the derivative is smooth, so the error is O(h^2) ~ 1e-10 relative).
    from statspai.regression._bivariate_normal import binormal_cdf

    df = _pcp_data(rho=0.3)
    a1 = _probit_index(df, "cover", ["x"])
    a2 = _probit_index(df, "claim", ["x", "z"])
    q1 = 2 * df["cover"].to_numpy(float) - 1
    q2 = 2 * df["claim"].to_numpy(float) - 1

    def ll(rho):
        return np.sum(np.log(binormal_cdf(q1 * a1, q2 * a2, q1 * q2 * rho)))

    h = 1e-5
    deriv = (ll(h) - ll(-h)) / (2 * h)
    res = sp.positive_correlation_test(
        df, "cover", "claim", ["x"], method="residuals", claim_covariates=["x", "z"]
    )
    assert res.detail["score"] == pytest.approx(deriv, rel=1e-6)
    # W is the squared score over the outer-product variance.
    assert res.statistic == pytest.approx(
        res.detail["score"] ** 2 / res.detail["score_variance"], rel=1e-12
    )


def test_residuals_detect_positive_correlation_and_not_negative():
    pos = _pcp_data(n=3000, rho=0.4, seed=1)
    neg = _pcp_data(n=3000, rho=-0.4, seed=1)
    rp = sp.positive_correlation_test(pos, "cover", "claim", ["x"], method="residuals")
    rn = sp.positive_correlation_test(neg, "cover", "claim", ["x"], method="residuals")
    assert rp.pvalue_greater < 1e-6 and rp.estimate > 0
    # Negative correlation is significant two-sided but not one-sided.
    assert rn.pvalue < 1e-6 and rn.pvalue_greater > 0.999


def test_residuals_weights_equal_duplicated_rows_for_score_and_correlation():
    # Integer weights: the probit fits, the score sum w e1 e2 and the weighted
    # residual correlation equal those on rows duplicated w times. (The
    # variance differs on purpose: sampling weights, not frequency weights.)
    df = _pcp_data(n=600, rho=0.2, seed=3)
    dup = df.loc[df.index.repeat(df["w"].astype(int))].reset_index(drop=True)
    rw = sp.positive_correlation_test(
        df, "cover", "claim", ["x"], method="residuals", weights="w"
    )
    rd = sp.positive_correlation_test(dup, "cover", "claim", ["x"], method="residuals")
    assert rw.detail["score"] == pytest.approx(rd.detail["score"], rel=1e-6)
    assert rw.estimate == pytest.approx(rd.estimate, rel=1e-6)


def test_residuals_symmetric_in_coverage_and_claim():
    df = _pcp_data(rho=0.2, seed=4)
    a = sp.positive_correlation_test(df, "cover", "claim", ["x"], method="residuals")
    b = sp.positive_correlation_test(df, "claim", "cover", ["x"], method="residuals")
    assert a.statistic == pytest.approx(b.statistic, rel=1e-12)


def test_cluster_variance_is_sum_of_cluster_score_squares():
    df = _pcp_data(rho=0.2, seed=5)
    res = sp.positive_correlation_test(
        df, "cover", "claim", ["x"], method="residuals", cluster="g"
    )
    a1 = _probit_index(df, "cover", ["x"])
    a2 = _probit_index(df, "claim", ["x"])

    def gr(y, a):
        q = 2 * y - 1
        return q * stats.norm.pdf(a) / stats.norm.cdf(q * a)

    s = gr(df["cover"].to_numpy(float), a1) * gr(df["claim"].to_numpy(float), a2)
    var = (pd.Series(s).groupby(df["g"]).sum() ** 2).sum()
    assert res.detail["score_variance"] == pytest.approx(var, rel=1e-8)
    assert res.detail["n_clusters"] == df["g"].nunique()


def test_no_covariates_uses_the_marginal_probit():
    df = _pcp_data(rho=0.3, seed=6)
    res = sp.positive_correlation_test(df, "cover", "claim", method="residuals")
    # With an intercept only, the probit index is Phi^{-1}(mean).
    p1, p2 = df["cover"].mean(), df["claim"].mean()
    y1, y2 = df["cover"].to_numpy(float), df["claim"].to_numpy(float)
    e1 = (y1 - p1) * stats.norm.pdf(stats.norm.ppf(p1)) / (p1 * (1 - p1))
    e2 = (y2 - p2) * stats.norm.pdf(stats.norm.ppf(p2)) / (p2 * (1 - p2))
    assert res.detail["score"] == pytest.approx(np.sum(e1 * e2), rel=1e-10)


@pytest.mark.slow
def test_residuals_size_under_independence():
    # 200 replications under independence; a 5% test rejects with probability
    # 0.05, so the count lies in [3, 19] with probability > 0.999 (binomial).
    rng = np.random.default_rng(7)
    rej = 0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for _ in range(200):
            df = _pcp_data(n=300, rho=0.0, seed=int(rng.integers(1 << 31)))
            r = sp.positive_correlation_test(
                df, "cover", "claim", ["x"], method="residuals"
            )
            rej += r.pvalue < 0.05
    assert 3 <= rej <= 19


def test_pcp_input_errors():
    df = _pcp_data(n=200)
    with pytest.raises(MethodIncompatibility):
        sp.positive_correlation_test(df, "cover", "claim", ["x"], weights="w")
    bad = df.assign(cover=df["cover"] * 2)
    with pytest.raises(MethodIncompatibility):
        sp.positive_correlation_test(bad, "cover", "claim", ["x"], method="residuals")
    with pytest.raises(ColumnNotFound):
        sp.positive_correlation_test(df, "cover", "claim", ["nope"])
    const = df.assign(claim=1)
    with pytest.raises(DataInsufficient):
        sp.positive_correlation_test(const, "cover", "claim", ["x"])
    with pytest.raises(MethodIncompatibility):
        sp.positive_correlation_test(df, "cover", "claim", ["x"], method="ols")


def test_pcp_result_contract():
    df = _pcp_data(n=500, rho=0.3)
    res = sp.positive_correlation_test(df, "cover", "claim", ["x"])
    d = res.to_dict(detail="minimal")
    assert d["estimate"] == res.estimate and d["citation_key"] == "chiappori2000testing"
    assert "Likelihood-ratio" in res.summary()


# ---------------------------------------------------------------------
# efc_welfare
# ---------------------------------------------------------------------

# D(p) = 0.9 - 0.1 p, AC(p) = 2 + 0.2 p, built exactly into the data.
A, B, G, DL = 0.9, -0.1, 2.0, 0.2


def _efc_exact_data(per_cell: int = 1000) -> pd.DataFrame:
    rows = []
    for p in (0.0, 1.0, 2.0, 3.0, 4.0):
        n_cov = int(round(per_cell * (A + B * p)))
        cov = np.r_[np.ones(n_cov), np.zeros(per_cell - n_cov)]
        # Zero-mean noise within each cell keeps the cost regression exact.
        noise = np.where(np.arange(n_cov) % 2 == 0, 1.0, -1.0)
        if n_cov % 2:
            noise[-1] = 0.0
        cost = np.r_[G + DL * p + noise, np.full(per_cell - n_cov, np.nan)]
        rows.append(pd.DataFrame({"p": p, "buy": cov, "cost": cost}))
    return pd.concat(rows, ignore_index=True)


def test_efc_closed_form_interior_case():
    res = sp.efc_welfare(_efc_exact_data(), "p", "buy", "cost")
    t = res.table["estimate"]
    for key, val in zip(("alpha", "beta", "gamma", "delta"), (A, B, G, DL)):
        assert t[key] == pytest.approx(val, abs=1e-12)
    # Equilibrium p = AC(p): p = G / (1 - DL); efficient p = MC(p):
    # p = (G + DL A / B) / (1 - 2 DL).
    p_eq = G / (1 - DL)
    p_eff = (G + DL * A / B) / (1 - 2 * DL)
    d_eq, d_eff = A + B * p_eq, A + B * p_eff
    # DWL triangle: base d_eff - d_eq, height AC - MC at d_eq = -DL d_eq / B.
    dwl = 0.5 * (d_eff - d_eq) * (-DL * d_eq / B)
    assert t["price_eq"] == pytest.approx(p_eq, rel=1e-10)
    assert t["coverage_eq"] == pytest.approx(d_eq, rel=1e-10)
    assert t["price_eff"] == pytest.approx(p_eff, rel=1e-10)
    assert t["coverage_eff"] == pytest.approx(d_eff, rel=1e-10)
    assert t["dwl"] == pytest.approx(dwl, rel=1e-10)
    assert dwl == pytest.approx(0.140833333333, rel=1e-9)
    assert res.diagnostics["selection"] == "adverse"


def test_efc_corner_full_coverage_is_efficient():
    # With costs flat in price (no selection), MC = AC; efficient coverage
    # then equals the equilibrium and the DWL is zero.
    df = _efc_exact_data()
    df["cost"] = np.where(df["buy"] == 1, 2.0 + (df.index % 2 - 0.5), np.nan)
    df["cost"] = df.groupby("p")["cost"].transform(lambda s: s - s.mean() + 2.0)
    res = sp.efc_welfare(df, "p", "buy", "cost")
    assert res.table.loc["delta", "estimate"] == pytest.approx(0.0, abs=1e-12)
    assert res.estimate == pytest.approx(0.0, abs=1e-12)
    assert res.table.loc["coverage_eff", "estimate"] == pytest.approx(
        res.table.loc["coverage_eq", "estimate"], abs=1e-12
    )


def test_efc_delta_se_of_beta_is_hc0_ols():
    rng = np.random.default_rng(11)
    n = 3000
    p = rng.choice([1.0, 2.0, 3.0, 4.0], size=n)
    wtp = rng.uniform(0, 8, size=n)
    buy = (wtp > p + rng.normal(size=n)).astype(int)
    cost = np.where(buy == 1, 0.5 * wtp + rng.normal(size=n), np.nan)
    df = pd.DataFrame({"p": p, "buy": buy, "cost": cost})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", AssumptionWarning)  # corner, extrapolated
        res = sp.efc_welfare(df, "p", "buy", "cost")
    ols = sp.regress("buy ~ p", data=df, robust="hc0")
    assert res.table.loc["beta", "se"] == pytest.approx(ols.std_errors["p"], rel=1e-6)
    cov = df[df["buy"] == 1]
    ols_c = sp.regress("cost ~ p", data=cov, robust="hc0")
    # The cost regression's influence rows are scaled to the full sample.
    assert res.table.loc["delta", "se"] == pytest.approx(
        ols_c.std_errors["p"], rel=1e-6
    )


def test_efc_bootstrap_reproducible_and_close_to_delta():
    rng = np.random.default_rng(12)
    n = 3000
    p = rng.choice([1.0, 2.0, 3.0, 4.0], size=n)
    wtp = rng.uniform(0, 8, size=n)
    buy = (wtp > p + rng.normal(size=n)).astype(int)
    cost = np.where(buy == 1, 0.5 * wtp + rng.normal(size=n), np.nan)
    df = pd.DataFrame({"p": p, "buy": buy, "cost": cost})
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", AssumptionWarning)
        a = sp.efc_welfare(df, "p", "buy", "cost", se_method="bootstrap", n_boot=60)
        b = sp.efc_welfare(df, "p", "buy", "cost", se_method="bootstrap", n_boot=60)
        d = sp.efc_welfare(df, "p", "buy", "cost")
    pd.testing.assert_frame_equal(a.table, b.table)
    # Screen, not a parity claim: 60 draws put the bootstrap SE of beta within
    # ~25% of the delta-method SE with high probability.
    assert a.table.loc["beta", "se"] == pytest.approx(
        d.table.loc["beta", "se"], rel=0.25
    )


def test_efc_errors():
    df = _efc_exact_data()
    with pytest.raises(IdentificationFailure):
        sp.efc_welfare(df.assign(p=df["p"] * -1.0), "p", "buy", "cost")
    with pytest.raises(IdentificationFailure):
        sp.efc_welfare(df.assign(p=1.0), "p", "buy", "cost")
    holed = df.copy()
    holed.loc[holed.index[0], "cost"] = np.nan  # first row is covered
    with pytest.raises(DataInsufficient):
        sp.efc_welfare(holed, "p", "buy", "cost")
    with pytest.raises(MethodIncompatibility):
        sp.efc_welfare(df, "p", "buy", "cost", se_method="jackknife")


def test_efc_corner_warns_under_delta():
    rng = np.random.default_rng(1)
    n = 4000
    p = rng.choice([2.0, 3.0, 4.0, 5.0], size=n)
    risk = rng.uniform(0, 10, size=n)
    buy = (risk > p + rng.normal(0, 1, n)).astype(int)
    cost = np.where(buy == 1, 0.6 * risk + rng.normal(0, 1, n), np.nan)
    df = pd.DataFrame({"p": p, "buy": buy, "cost": cost})
    with pytest.warns(AssumptionWarning):
        res = sp.efc_welfare(df, "p", "buy", "cost")
    assert res.diagnostics["efficient_corner"]
    assert res.table.loc["coverage_eff", "estimate"] == 1.0


# ---------------------------------------------------------------------
# pooled_price_ratio
# ---------------------------------------------------------------------


def test_ppr_discrete_exact():
    # p = 0.1 (weight 2), 0.2, 0.4. At r = 0.1: E[P|P>=r] = 0.8 / 4 = 0.2 and
    # T = (0.2/0.8) / (0.1/0.9) = 2.25. At 0.2: 0.3, T = (0.3/0.7)/(0.2/0.8).
    res = sp.pooled_price_ratio([0.1, 0.2, 0.4], weights=[2, 1, 1])
    c = res.curve
    assert c["quantile"].tolist() == pytest.approx([0.0, 0.5, 0.75])
    assert c["mean_above"].tolist() == pytest.approx([0.2, 0.3, 0.4])
    t_02 = (0.3 / 0.7) / 0.25
    assert c["T"].tolist() == pytest.approx([2.25, t_02, 1.0], rel=1e-12)
    assert res.estimate == pytest.approx(0.0, abs=1e-12)  # top point included
    r6 = sp.pooled_price_ratio([0.1, 0.2, 0.4], weights=[2, 1, 1], upto_quantile=0.6)
    assert r6.estimate == pytest.approx(t_02 - 1, rel=1e-12)
    assert r6.argmin_r == 0.2


def test_ppr_markup_no_trade():
    r = sp.pooled_price_ratio(
        [0.1, 0.2, 0.4], weights=[2, 1, 1], upto_quantile=0.6, markup=0.5
    )
    assert r.no_trade is False  # the top type is always servable alone
    assert r.no_trade_upto is True  # t = 1.25 and 0.714 >= 0.5


def test_ppr_uniform_closed_form():
    # P ~ U(a, b): E[P | P >= r] = (r + b) / 2. Midpoint quantile sample of
    # size n reproduces it to O((b - a) / n).
    a, b, n = 0.02, 0.3, 20000
    pv = a + (b - a) * (np.arange(n) + 0.5) / n
    res = sp.pooled_price_ratio(pv)
    c = res.curve.iloc[::1000]
    m = (c["r"] + b) / 2
    expected = (m / (1 - m)) / (c["r"] / (1 - c["r"]))
    np.testing.assert_allclose(c["T"], expected, rtol=1e-4)


def test_ppr_beta_t80_closed_form():
    # P ~ Beta(2, 30): E[P 1{P >= r}] = mean * (1 - I_r(3, 30)).
    a_, b_, n = 2.0, 30.0, 40000
    pv = stats.beta.ppf((np.arange(n) + 0.5) / n, a_, b_)
    res = sp.pooled_price_ratio(pv)
    r = np.linspace(stats.beta.ppf(0.001, a_, b_), stats.beta.ppf(0.8, a_, b_), 4000)
    mean = a_ / (a_ + b_)
    m = mean * (1 - special.betainc(a_ + 1, b_, r)) / (1 - special.betainc(a_, b_, r))
    t = (m / (1 - m)) / (r / (1 - r)) - 1
    # Grid and sample discretisation: agreement to 2e-3 relative.
    assert res.estimate == pytest.approx(t.min(), rel=2e-3)


def test_ppr_weights_equal_duplicated_rows():
    rng = np.random.default_rng(2)
    pv = np.round(rng.beta(2, 20, size=300), 3)
    w = rng.integers(1, 4, size=300)
    a = sp.pooled_price_ratio(pv, weights=w)
    b = sp.pooled_price_ratio(np.repeat(pv, w))
    pd.testing.assert_frame_equal(a.curve, b.curve, rtol=1e-12)
    assert a.estimate == pytest.approx(b.estimate, rel=1e-12)


def test_ppr_column_and_bootstrap():
    rng = np.random.default_rng(3)
    df = pd.DataFrame({"risk": rng.beta(2, 30, size=2000)})
    a = sp.pooled_price_ratio("risk", data=df, n_boot=30, seed=4)
    b = sp.pooled_price_ratio(df["risk"].to_numpy(), n_boot=30, seed=4)
    assert a.se == b.se and a.ci == b.ci
    assert a.ci[0] <= a.estimate <= a.ci[1]


def test_ppr_errors():
    with pytest.raises(MethodIncompatibility):
        sp.pooled_price_ratio([0.1, 1.2])
    with pytest.raises(MethodIncompatibility):
        sp.pooled_price_ratio("risk")
    with pytest.raises(DataInsufficient):
        sp.pooled_price_ratio([0.0, 0.0, 1.0])
    with pytest.raises(MethodIncompatibility):
        sp.pooled_price_ratio([0.1, 0.2], upto_quantile=0.0)


# ---------------------------------------------------------------------
# positive_correlation_test with ordered outcomes
# ---------------------------------------------------------------------


def _ordered_data(n: int = 1200, rho: float = 0.0, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    z = rng.normal(size=n)
    a = np.sqrt(abs(rho))
    u = rng.normal(size=n)
    e1 = a * u + np.sqrt(1 - a**2) * rng.normal(size=n)
    e2 = a * u + np.sqrt(1 - a**2) * rng.normal(size=n)
    return pd.DataFrame(
        {
            "x": x,
            "z": z,
            "cover": np.digitize(0.4 * x + e1, [-0.4, 0.5]),
            "nclaim": np.digitize(0.3 * x + 0.4 * z + e2, [0.4, 1.3]),
            "w": rng.integers(1, 4, size=n).astype(float),
        }
    )


def _rect_prob(l1, u1, l2, u2, rho):
    """P(l1 < e1 <= u1, l2 < e2 <= u2) for a standard bivariate normal."""
    from scipy import integrate

    s = np.sqrt(1 - rho**2)

    def f(t):
        return stats.norm.pdf(t) * (
            stats.norm.cdf((u2 - rho * t) / s) - stats.norm.cdf((l2 - rho * t) / s)
        )

    return integrate.quad(f, l1, u1, epsabs=1e-14, epsrel=1e-13)[0]


def test_ordered_residual_score_is_bivariate_ordered_probit_rho_derivative():
    # sum e1 e2 equals d logL / d rho at rho = 0 of the bivariate model with
    # ordered probit margins held at their separate fits (central
    # difference, step 1e-4: truncation error about 1e-8 relative).
    df = _ordered_data(n=120, seed=3)
    res = sp.positive_correlation_test(
        df, "cover", "nclaim", ["x"], method="residuals", claim_covariates=["x", "z"]
    )
    bounds = []
    for y, x in (("cover", ["x"]), ("nclaim", ["x", "z"])):
        fit = sp.oprobit(data=df, y=y, x=x)
        cuts = fit.model_info["cutpoints"]
        k = np.r_[-np.inf, [cuts[f"cut{j + 1}"] for j in range(len(cuts))], np.inf]
        xb = df[x].to_numpy() @ fit.params[x].to_numpy()
        j = df[y].to_numpy()
        bounds.append((k[j] - xb, k[j + 1] - xb))
    (l1, u1), (l2, u2) = bounds

    def loglik(rho):
        return sum(
            np.log(_rect_prob(l1[i], u1[i], l2[i], u2[i], rho)) for i in range(len(df))
        )

    h = 1e-4
    deriv = (loglik(h) - loglik(-h)) / (2 * h)
    assert res.detail["score"] == pytest.approx(deriv, rel=1e-6)
    assert set(res.detail["ordered"]) == {"cover", "nclaim"}


def test_ordered_residual_with_two_levels_is_probit_residual():
    from statspai.adverse_selection._ordered import ordered_generalized_residual
    from statspai.adverse_selection.tests import _generalized_residual, _probit_index

    df = _pcp_data(n=500, seed=4)
    e_ord = ordered_generalized_residual(df, "claim", [])
    e_bin = _generalized_residual(
        df["claim"].to_numpy(float), _probit_index(df, "claim", [], None)
    )
    np.testing.assert_allclose(e_ord, e_bin, rtol=1e-12)


def test_ordered_residual_test_size_and_power():
    # 200 null samples: rejection rate at 5% within [0.02, 0.09] (binomial
    # SE 0.015). Under rho = 0.4 the one-sided test rejects.
    rej = [
        sp.positive_correlation_test(
            _ordered_data(n=400, seed=100 + r),
            "cover",
            "nclaim",
            ["x"],
            method="residuals",
            claim_covariates=["x", "z"],
        ).pvalue
        < 0.05
        for r in range(200)
    ]
    assert 0.02 <= np.mean(rej) <= 0.09
    res = sp.positive_correlation_test(
        _ordered_data(rho=0.4, seed=5),
        "cover",
        "nclaim",
        ["x"],
        method="residuals",
        claim_covariates=["x", "z"],
    )
    assert res.z > 0 and res.pvalue_greater < 1e-4


def test_ordered_outcomes_rejections():
    df = _ordered_data(n=300, seed=6)
    with pytest.raises(MethodIncompatibility):
        sp.positive_correlation_test(df, "cover", "nclaim", ["x"])  # biprobit
    with pytest.raises(MethodIncompatibility):
        sp.positive_correlation_test(
            df, "cover", "nclaim", ["x"], method="residuals", weights="w"
        )
    with pytest.raises(MethodIncompatibility):
        sp.positive_correlation_test(df, "cover", "z", ["x"], method="residuals")
    two = df.assign(c2=(df["cover"] > 0).astype(int) + 1)  # coded 1/2
    with pytest.raises(MethodIncompatibility, match="0/1"):
        sp.positive_correlation_test(two, "c2", "nclaim", ["x"], method="residuals")
