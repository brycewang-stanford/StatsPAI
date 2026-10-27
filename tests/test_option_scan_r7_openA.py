"""Round-7 option scan, line openA: the items round 6 left open.

Each test states the correct behaviour against a known truth, an identity
or a reference implementation (Stata 18 numbers are pinned inline with the
command that produced them).
"""

from __future__ import annotations

import pathlib
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import statspai as sp
from statspai.core.results import CausalResult
from statspai.exceptions import MethodIncompatibility

_FIX = pathlib.Path(__file__).parent / "fixtures"


def _frame(seed: int = 0, n: int = 400) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "x": rng.normal(size=n),
            "z": rng.normal(size=n),
            "f": rng.integers(0, 20, n),
            "h": rng.integers(0, 15, n),
            "g": rng.choice(list("abc"), n),
        }
    )
    df["d"] = (rng.uniform(size=n) < 1 / (1 + np.exp(-0.5 * df["x"]))).astype(int)
    df["y"] = 1 + 0.5 * df["d"] + df["x"] + rng.normal(size=n)
    df["c"] = rng.poisson(np.exp(0.3 * df["x"]))
    return df


# --------------------------------------------------------------------------
# 1. CausalResult.conf_int at another level uses the fit's own distribution
# --------------------------------------------------------------------------


def _t_result(df_resid: float | None) -> CausalResult:
    """A headline estimate whose stored interval is t(12) at 95%."""
    info = {} if df_resid is None else {"df_resid": df_resid}
    crit = stats.norm.ppf(0.975) if df_resid is None else stats.t.ppf(0.975, df_resid)
    return CausalResult(
        method="test",
        estimand="ATE",
        estimate=1.5,
        se=0.5,
        pvalue=0.01,
        ci=(1.5 - crit * 0.5, 1.5 + crit * 0.5),
        alpha=0.05,
        n_obs=14,
        model_info=info,
    )


def test_conf_int_at_another_level_uses_t_df_resid() -> None:
    r = _t_result(12)
    lo, hi = r.conf_int(0.10).iloc[0]
    crit = stats.t.ppf(0.95, 12)
    np.testing.assert_allclose([lo, hi], [1.5 - crit * 0.5, 1.5 + crit * 0.5])
    main = r.tidy(conf_level=0.90).query("type == 'main'").iloc[0]
    assert (lo, hi) == (main["conf_low"], main["conf_high"])


def test_conf_int_tidy_and_summary_agree_at_every_level() -> None:
    r = _t_result(12)
    for a in (0.01, 0.10, 0.20):
        ci = r.conf_int(a).iloc[0].to_numpy()
        main = r.tidy(conf_level=1 - a).query("type == 'main'")
        np.testing.assert_array_equal(ci, main[["conf_low", "conf_high"]].to_numpy()[0])
        line = next(
            s for s in str(r.summary(alpha=a, digits=6)).splitlines() if "% CI]" in s
        )
        printed = [float(v) for v in line.split("[", 2)[2].strip(" ]").split(",")]
        np.testing.assert_allclose(printed, ci, atol=5e-7)


def test_conf_int_without_df_resid_stays_normal() -> None:
    r = _t_result(None)
    lo, hi = r.conf_int(0.10).iloc[0]
    z = stats.norm.ppf(0.95)
    np.testing.assert_allclose([lo, hi], [1.5 - z * 0.5, 1.5 + z * 0.5], rtol=1e-15)
    # at the fit level the stored interval is returned as it is
    assert tuple(r.conf_int().iloc[0]) == tuple(r.ci)


# --------------------------------------------------------------------------
# 2. sp.ipw drops exactly collinear propensity covariates
# --------------------------------------------------------------------------


def _dup_frame() -> pd.DataFrame:
    df = _frame(0)
    return df.assign(x2=2 * df["x"])


def test_ipw_sandwich_with_a_duplicated_covariate_equals_the_reduced_fit() -> None:
    df = _dup_frame()
    with pytest.warns(UserWarning, match="x2 omitted because of collinearity"):
        full = sp.ipw(
            df, y="y", treat="d", covariates=["x", "x2"], se_method="sandwich"
        )
    reduced = sp.ipw(df, y="y", treat="d", covariates=["x"], se_method="sandwich")
    assert full.estimate == reduced.estimate
    assert full.se == reduced.se
    assert full.model_info["omitted"] == [
        {"variable": "x2", "reason": "collinear with 'x'"}
    ]
    assert "omitted" not in reduced.model_info


def test_ipw_bootstrap_with_a_duplicated_covariate_equals_the_reduced_fit() -> None:
    # 1.41: bootstrap SE 0.4296 with the duplicate, 0.1260 without it
    rng = np.random.default_rng(0)
    n = 300
    df = pd.DataFrame({"x": rng.normal(size=n), "z": rng.normal(size=n)})
    df["x2"] = 2 * df["x"]
    df["d"] = (rng.uniform(size=n) < 1 / (1 + np.exp(-0.8 * df["x"]))).astype(int)
    df["y"] = 1 + 0.5 * df["d"] + df["x"] + rng.normal(size=n)
    kw = dict(y="y", treat="d", seed=1, n_bootstrap=200)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        full = sp.ipw(df, covariates=["x", "z", "x2"], **kw)
    reduced = sp.ipw(df, covariates=["x", "z"], **kw)
    assert (full.estimate, full.se) == (reduced.estimate, reduced.se)


def test_ipw_seed_and_random_state_reproduce_the_bootstrap() -> None:
    df = _frame(1)
    kw = dict(y="y", treat="d", covariates=["x", "z"], n_bootstrap=40)
    a = sp.ipw(df, seed=11, **kw).se
    b = sp.ipw(df, seed=11, **kw).se
    c = sp.ipw(df, random_state=11, **kw).se
    assert a == b == c


# --------------------------------------------------------------------------
# 3. sp.qreg accepts Stata's vce(...) wrapper
# --------------------------------------------------------------------------


@pytest.mark.parametrize("bare", ["cluster f", "robust", "iid", "nid"])
def test_qreg_vce_wrapper_equals_bare_spelling(bare: str) -> None:
    df = _frame(2)
    a = sp.qreg(df, "y ~ x", vce=bare)
    b = sp.qreg(df, "y ~ x", vce=f"vce({bare})")
    np.testing.assert_array_equal(a.std_errors.to_numpy(), b.std_errors.to_numpy())


# --------------------------------------------------------------------------
# 4. vcov={"CRV1": "f+h"} is two-way clustering
# --------------------------------------------------------------------------


@pytest.mark.parametrize("spec", [{"CRV1": "f+h"}, {"CRV1": ["f", "h"]}])
def test_regress_crv1_plus_is_two_way_clustering(spec: dict) -> None:
    df = _frame(3)
    ref = sp.regress("y ~ x", data=df, cluster=["f", "h"])
    got = sp.regress("y ~ x", data=df, vcov=spec)
    np.testing.assert_array_equal(got.std_errors.to_numpy(), ref.std_errors.to_numpy())
    np.testing.assert_array_equal(got.params.to_numpy(), ref.params.to_numpy())


def test_multiway_cr3_is_refused_with_a_hint() -> None:
    df = _frame(3)
    with pytest.raises(MethodIncompatibility, match="one-way only"):
        sp.regress("y ~ x", data=df, vcov={"CRV3": "f+h"})


def test_ivreg_crv1_plus_equals_its_cluster_list() -> None:
    df = _frame(5)
    df["x"] = df["z"] + 0.5 * df["x"]  # a strong instrument
    ref = sp.ivreg("y ~ (x ~ z)", data=df, cluster=["f", "h"])
    got = sp.ivreg("y ~ (x ~ z)", data=df, vcov={"CRV1": "f+h"})
    np.testing.assert_array_equal(got.std_errors.to_numpy(), ref.std_errors.to_numpy())
    assert np.isfinite(ref.std_errors.to_numpy()).all()


# --------------------------------------------------------------------------
# 5. Count models omit the collinear term regress / logit omit
# --------------------------------------------------------------------------


def _collinear_factor_frame() -> pd.DataFrame:
    rng = np.random.default_rng(5)
    n = 500
    df = pd.DataFrame({"x": rng.normal(size=n), "g": rng.choice(list("abc"), n)})
    df["z"] = (df["g"] == "b").astype(float)
    df["c"] = rng.poisson(np.exp(0.3 * df["x"] + 0.2 * df["z"]))
    df["d"] = (rng.uniform(size=n) < 0.4 + 0.2 * df["z"]).astype(int)
    df["f"] = rng.integers(0, 10, n)
    return df


def _omitted(result) -> list:
    return [o["variable"] for o in result.model_info["omitted"]]


@pytest.mark.parametrize("rhs", ["z + C(g) + x", "C(g) + z + x", "x + C(g) + z"])
def test_count_models_follow_written_order(rhs: str) -> None:
    df = _collinear_factor_frame()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        want = _omitted(sp.regress(f"c ~ {rhs}", data=df))
        assert _omitted(sp.logit(f"d ~ {rhs}", data=df)) == want
        assert _omitted(sp.poisson(f"c ~ {rhs}", data=df)) == want
        assert _omitted(sp.nbreg(f"c ~ {rhs}", data=df)) == want


def test_nbreg_with_absorbed_effects_follows_written_order() -> None:
    df = _collinear_factor_frame()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sp.nbreg("c ~ z + C(g) + x | f", data=df)
    assert _omitted(res) == ["C(g)[T.b]"]
    assert "z" in res.params.index


# --------------------------------------------------------------------------
# 6. The logit AUC is weighted by the fit's weights (Stata lroc)
# --------------------------------------------------------------------------


def _auc_frame() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    n = 300
    df = pd.DataFrame({"x": rng.normal(size=n), "z": rng.normal(size=n)})
    df["d"] = (rng.uniform(size=n) < 1 / (1 + np.exp(-(0.3 + 0.8 * df["x"])))).astype(
        int
    )
    df["w"] = rng.integers(1, 6, n).astype(float)
    return df


def test_weighted_logit_auc_matches_stata_lroc() -> None:
    # Stata 18, the same rows written with %.17g:
    #   logit d x z [fw=fw]  ; lroc, nograph  -> r(area) = .650319412662089813
    #   logit d x z [iw=w]   ; lroc, nograph  -> r(area) = .650319412662089813
    #   logit d x z          ; lroc, nograph  -> r(area) = .650176840751457985
    df = _auc_frame()
    weighted = sp.logit("d ~ x + z", data=df, weights="w")
    assert weighted.model_info["auc"] == pytest.approx(0.650319412662089813, abs=1e-14)
    plain = sp.logit("d ~ x + z", data=df)
    assert plain.model_info["auc"] == pytest.approx(0.650176840751457985, abs=1e-14)


def test_weighted_auc_equals_the_auc_of_the_expanded_data() -> None:
    from statspai.regression.logit_probit import _roc_auc

    rng = np.random.default_rng(8)
    y = rng.integers(0, 2, 200).astype(float)
    p = np.round(rng.uniform(size=200), 1)  # ties within and across arms
    w = rng.integers(0, 4, 200).astype(float)
    expanded = _roc_auc(np.repeat(y, w.astype(int)), np.repeat(p, w.astype(int)))
    assert _roc_auc(y, p, w) == pytest.approx(expanded, abs=1e-14)
    assert _roc_auc(y, p, np.ones(200)) == pytest.approx(_roc_auc(y, p), abs=1e-14)


# --------------------------------------------------------------------------
# 7. sp.fast.fepois default tolerance
# --------------------------------------------------------------------------


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_fast_fepois_default_is_within_1e8_of_the_converged_fit(seed: int) -> None:
    rng = np.random.default_rng(500 + seed)
    n, G, T = 3000, 300, 30
    i, t = rng.integers(0, G, n), rng.integers(0, T, n)
    df = pd.DataFrame(
        {"i": i, "t": t, "x1": rng.normal(size=n), "x2": rng.binomial(1, 0.05, n)}
    )
    lam = np.exp(
        0.3 * df["x1"]
        + 2.0 * df["x2"]
        + 2 * rng.normal(size=G)[i]
        + rng.normal(size=T)[t]
    )
    df["y"] = rng.poisson(lam)
    df["w"] = np.exp(rng.normal(0, 2, n))
    wt = "w" if seed == 1 else None
    default = sp.fast.fepois("y ~ x1 + x2 | i + t", data=df, weights=wt)
    tight = sp.fast.fepois(
        "y ~ x1 + x2 | i + t",
        data=df,
        weights=wt,
        tol=1e-14,
        fe_tol=1e-12,
        maxiter=300,
        fe_maxiter=20000,
    )
    assert np.max(np.abs(default.coef_vec / tight.coef_vec - 1)) < 1e-8


# --------------------------------------------------------------------------
# 8. sp.fast.boottest p-value types (Stata boottest ptype())
# --------------------------------------------------------------------------

_BT = _FIX / "boottest_ptype"


def test_boottest_p_value_types_match_stata_on_its_distribution() -> None:
    # boottest_ptype.do: boottest x, ptype(...) reps(999) seed(42) svmat(t)
    from statspai.fast.inference import _boot_t_pvalue

    dist = pd.read_csv(_BT / "stata_dist.csv")["dd1"].to_numpy()
    t = -0.7452775697490591211
    want = {
        "symmetric": 0.595595595595595606,
        "equal-tailed": 0.580580580580580596,
        "lower": 0.290290290290290298,
        "upper": 0.709709709709709702,
    }
    for kind, p in want.items():
        assert _boot_t_pvalue(dist, t, kind) == pytest.approx(p, abs=1e-15)


def _bt_inputs():
    d = pd.read_csv(_BT / "data.csv")
    X = np.column_stack([np.ones(len(d)), d["x"].to_numpy()])
    return X, d["y"].to_numpy(), d["g"].to_numpy()


def test_boottest_t_obs_matches_stata_and_p_types_use_the_draws() -> None:
    X, y, g = _bt_inputs()
    res = {
        k: sp.fast.boottest(X, y, g, null_coef=1, B=399, seed=3, p_val_type=k)
        for k in ("symmetric", "equal-tailed", "lower", "upper")
    }
    assert res["symmetric"].t_obs == pytest.approx(-0.7452775697490591211, rel=1e-12)
    draws, t = res["symmetric"].boot_t_dist, res["symmetric"].t_obs
    gap = 1e-10 * abs(t)  # beyond round-off of a tie
    assert res["symmetric"].pvalue == np.mean(np.abs(draws) > abs(t) + gap)
    lower, upper = np.mean(draws < t - gap), np.mean(draws > t + gap)
    assert res["lower"].pvalue == lower and res["upper"].pvalue == upper
    assert res["equal-tailed"].pvalue == min(1.0, 2 * min(lower, upper))
    assert sp.fast.boottest(X, y, g, null_coef=1, B=399, seed=3).pvalue == (
        res["symmetric"].pvalue
    )
    assert res["lower"].to_dict()["p_val_type"] == "lower"


@pytest.mark.parametrize("weights", ["rademacher", "webb"])
def test_boottest_constant_weight_draws_are_ties_not_exceedances(weights) -> None:
    """Stata boottest counts ``|t*| > |t|`` strictly.

    With four clusters an eighth of the Rademacher draws have constant
    weights; each reproduces ``±t_obs`` and is a tie. Until 1.42 the
    symmetric p-value counted them through ``>=`` (or not, by the last bit
    of the refit).
    """
    rng = np.random.default_rng(12)
    g = np.repeat(np.arange(4), 25)
    x = rng.normal(size=100)
    y = 0.4 * x + rng.normal(size=4)[g] + rng.normal(size=100)
    X = np.column_stack([np.ones(100), x])
    res = sp.fast.boottest(X, y, g, null_coef=1, B=400, seed=1, weights=weights)
    draws, t = res.boot_t_dist, res.t_obs
    ties = np.isclose(np.abs(draws), abs(t), rtol=1e-10, atol=0)
    assert ties.sum() > 0  # constant-weight draws were drawn
    assert res.pvalue == np.mean((np.abs(draws) > abs(t)) & ~ties)
    if weights == "rademacher":
        # Stata 18 on these rows (%.17g): regress y x, cluster(g);
        # boottest x  -> all 16 Rademacher draws, t = 2.164410241825723613,
        # p = 0 (the two constant draws are ties). 1.41 reported 0.1275.
        assert t == pytest.approx(2.164410241825723613, rel=1e-12)
        assert res.pvalue == 0.0
    for kind, want in (
        ("lower", np.mean((draws < t) & ~np.isclose(draws, t, rtol=1e-10, atol=0))),
        ("upper", np.mean((draws > t) & ~np.isclose(draws, t, rtol=1e-10, atol=0))),
    ):
        got = sp.fast.boottest(
            X, y, g, null_coef=1, B=400, seed=1, weights=weights, p_val_type=kind
        )
        assert got.pvalue == want


def test_boottest_extra_df_moves_t_but_not_the_p_value() -> None:
    X, y, g = _bt_inputs()
    a = sp.fast.boottest(X, y, g, null_coef=1, B=299, seed=4)
    b = sp.fast.boottest(X, y, g, null_coef=1, B=299, seed=4, extra_df=6)
    n, k = X.shape
    # t scales by sqrt of the ratio of the CR1 factors (n - k) / (n - k - 6)
    assert b.t_obs / a.t_obs == pytest.approx(np.sqrt((n - k - 6) / (n - k)), rel=1e-12)
    assert a.pvalue == b.pvalue


def test_boottest_unknown_p_val_type_is_refused() -> None:
    X, y, g = _bt_inputs()
    with pytest.raises(MethodIncompatibility, match="p_val_type"):
        sp.fast.boottest(X, y, g, null_coef=1, B=9, p_val_type="two-sided")
