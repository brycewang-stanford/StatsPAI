"""Option scan of ``mendelian/`` (2026-10-27).

Each test states the correct behaviour; the defects they found are fixed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import MethodIncompatibility


def _snps(n: int = 40, seed: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    bx = rng.uniform(0.05, 0.3, n)
    return pd.DataFrame(
        {
            "bx": bx,
            "by": 0.4 * bx + rng.normal(0, 0.01, n),
            "sx": rng.uniform(0.01, 0.02, n),
            "sy": rng.uniform(0.01, 0.02, n),
        }
    )


def test_mendelian_randomization_unknown_method_is_refused():
    with pytest.raises(ValueError):
        sp.mendelian_randomization(
            data=_snps(),
            beta_exposure="bx",
            beta_outcome="by",
            se_exposure="sx",
            se_outcome="sy",
            methods=["ivw", "median"],  # 'median' is not a recognised name here
        )


# --------------------------------------------------------------------------- #
#  MVMR conditional F (Sanderson & Windmeijer): how much variation in the
#  variant-exposure associations of X_j is left once the other exposures are
#  conditioned on, relative to its sampling error. Collinearity
#  (bx2 = 2 bx1 + noise of sd 1e-5, sampling SEs 0.01) leaves about none:
#  the conditional F of X1 is of order (0.5e-5 / 0.011)^2, i.e. 2e-7.
#  (Exact collinearity leaves the MVMR itself unidentified.)
# --------------------------------------------------------------------------- #


def _mvmr_collinear(seed: int = 9) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n = 40
    bx1 = rng.uniform(0.05, 0.3, n)
    bx2 = 2.0 * bx1 + rng.normal(0, 1e-5, n)  # collinear up to 1e-5
    return pd.DataFrame(
        {
            "beta_x1": bx1,
            "beta_x2": bx2,
            "se_x1": np.full(n, 0.01),
            "se_x2": np.full(n, 0.01),
            "beta_y": 0.3 * bx1 + rng.normal(0, 0.01, n),
            "se_y": np.full(n, 0.01),
        }
    )


def test_mvmr_conditional_f_is_near_zero_under_collinearity():
    df = _mvmr_collinear()
    res = sp.mr_multivariable(
        df,
        outcome="beta_y",
        outcome_se="se_y",
        exposures=["beta_x1", "beta_x2"],
        exposure_se=["se_x1", "se_x2"],
    )
    assert 0 <= res.conditional_f_stats["beta_x1"] < 1e-5
    assert 0 <= res.conditional_f_stats["beta_x2"] < 1e-5


def test_f_statistic_r2_needs_sample_size():
    df = _snps()
    res = sp.mr_f_statistic(df["bx"].to_numpy(), df["sx"].to_numpy())
    assert np.isnan(res.r2_mean)


def test_mvmr_without_exposure_se_reports_no_conditional_f():
    df = _mvmr_collinear()
    res = sp.mr_multivariable(df, exposures=["beta_x1", "beta_x2"])
    assert all(np.isnan(v) for v in res.conditional_f_stats.values())
    assert "pass exposure_se=" in res.summary()


def test_mvmr_conditional_f_closed_form_two_exposures():
    # K = 2: d is the least-squares slope of b1 on b2 (no intercept),
    # Q = sum (b1 - d b2)^2 / (se1^2 + d^2 se2^2), F = Q / (L - 1)
    rng = np.random.default_rng(21)
    L = 50
    b1 = rng.normal(0, 0.2, L)
    b2 = 0.5 * b1 + rng.normal(0, 0.1, L)
    s1 = rng.uniform(0.01, 0.03, L)
    s2 = rng.uniform(0.01, 0.03, L)
    df = pd.DataFrame(
        {
            "beta_x1": b1,
            "beta_x2": b2,
            "se_x1": s1,
            "se_x2": s2,
            "beta_y": 0.3 * b1 + rng.normal(0, 0.01, L),
            "se_y": np.full(L, 0.01),
        }
    )
    res = sp.mr_multivariable(
        df, exposures=["beta_x1", "beta_x2"], exposure_se=["se_x1", "se_x2"]
    )
    d = np.sum(b1 * b2) / np.sum(b2**2)
    q = np.sum((b1 - d * b2) ** 2 / (s1**2 + d**2 * s2**2))
    assert res.conditional_f_stats["beta_x1"] == pytest.approx(q / (L - 1), rel=1e-12)


def test_mvmr_exposure_se_length_mismatch_is_refused():
    with pytest.raises(MethodIncompatibility):
        sp.mr_multivariable(
            _mvmr_collinear(), exposures=["beta_x1", "beta_x2"], exposure_se=["se_x1"]
        )


def test_mr_ivw_refuses_missing_values():
    df = _snps()
    df.loc[3, "by"] = np.nan
    with pytest.raises(MethodIncompatibility):
        sp.mr_ivw(df["bx"], df["by"], df["sx"], df["sy"])


def test_mr_ivw_series_equals_arrays():
    df = _snps()
    a = sp.mr_ivw(df["bx"], df["by"], df["sx"], df["sy"])
    b = sp.mr_ivw(*(df[c].to_numpy() for c in ("bx", "by", "sx", "sy")))
    assert a["estimate"] == b["estimate"] and a["se"] == b["se"]


def test_f_statistic_with_sample_size():
    df = _snps()
    res = sp.mr_f_statistic(df["bx"].to_numpy(), df["sx"].to_numpy(), n_samples=10000)
    f = (df["bx"] / df["sx"]) ** 2
    assert res.f_mean == pytest.approx(f.mean(), rel=1e-14)
    assert res.r2_mean == pytest.approx((f / (f + 9998)).mean(), rel=1e-14)


# R MVMR 0.4 (Sanderson, Spiller & Bowden's package, mrcieu r-universe),
# strength_mvmr(format_mvmr(...), gencov = 0), run as a black box on
# 2026-10-27 on these exact doubles (written with %.17g and read back).
_MVMR_K2 = {"beta_x1": 18.283939040033296, "beta_x2": 15.996838812986697}
_MVMR_K3 = {
    "beta_x1": 16.049148126687719,
    "beta_x2": 13.804753951155973,
    "beta_x3": 16.616296831801812,
}


def _mvmr_k2() -> pd.DataFrame:
    rng = np.random.default_rng(21)
    L = 50
    b1 = rng.normal(0, 0.2, L)
    b2 = 0.5 * b1 + rng.normal(0, 0.1, L)
    s1 = rng.uniform(0.01, 0.03, L)
    s2 = rng.uniform(0.01, 0.03, L)
    return pd.DataFrame(
        {
            "beta_x1": b1,
            "beta_x2": b2,
            "se_x1": s1,
            "se_x2": s2,
            "beta_y": 0.3 * b1 + rng.normal(0, 0.01, L),
            "se_y": np.full(L, 0.01),
        }
    )


def _mvmr_k3() -> pd.DataFrame:
    rng = np.random.default_rng(33)
    L = 60
    b1 = rng.normal(0, 0.2, L)
    b2 = 0.4 * b1 + rng.normal(0, 0.1, L)
    b3 = -0.3 * b1 + 0.5 * b2 + rng.normal(0, 0.08, L)
    return pd.DataFrame(
        {
            "beta_x1": b1,
            "beta_x2": b2,
            "beta_x3": b3,
            "se_x1": rng.uniform(0.01, 0.03, L),
            "se_x2": rng.uniform(0.01, 0.03, L),
            "se_x3": rng.uniform(0.01, 0.03, L),
            "beta_y": 0.3 * b1 - 0.2 * b3 + rng.normal(0, 0.01, L),
            "se_y": np.full(L, 0.01),
        }
    )


@pytest.mark.parametrize("make, ref", [(_mvmr_k2, _MVMR_K2), (_mvmr_k3, _MVMR_K3)])
def test_mvmr_conditional_f_matches_r_mvmr_strength(make, ref):
    df = make()
    exps = list(ref)
    res = sp.mr_multivariable(
        df, exposures=exps, exposure_se=[e.replace("beta_", "se_") for e in exps]
    )
    for e in exps:
        assert res.conditional_f_stats[e] == pytest.approx(ref[e], rel=1e-12)
