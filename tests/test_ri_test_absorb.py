"""``sp.ri_test`` with a continuous treatment, absorbed FEs and ``interact=``.

The design of Zheng, Huang & Zhu (2026) Section 16A: occupation-level AI
exposure times post, four absorbed FE groups, exposure permuted across
occupations. The statistic must equal the ``sp.hdfe_ols`` coefficient refitted
on the permuted exposure (FWL identity: exact up to the absorber tolerance).
"""

import numpy as np
import pandas as pd
import pytest

import statspai as sp


@pytest.fixture(scope="module")
def panel():
    rng = np.random.default_rng(0)
    n, G = 3000, 30
    occ = rng.integers(0, G, n)
    exposure = rng.uniform(0, 1, G)
    df = pd.DataFrame(
        {
            "occ": occ,
            "firm": rng.integers(0, 150, n),
            "city": rng.integers(0, 8, n),
            "q": rng.integers(0, 6, n),
            "exposure": exposure[occ],
        }
    )
    df["post"] = (df.q >= 3).astype(float)
    df["x"] = rng.normal(size=n)
    df["y"] = (
        -0.3 * df.exposure * df.post
        + 0.2 * df.x
        + rng.normal(size=150)[df.firm]
        + rng.normal(size=n)
    )
    df["d"] = df.exposure * df.post
    return df


FE = ["firm", "occ", "city^q"]


def _refit(df, col="d"):
    r = sp.hdfe_ols(
        f"y ~ {col} + x | firm + occ + city^q", df, cluster="occ", tol=1e-12
    )
    return float(r.coef[col]), float(r.tvalues[col])


def test_observed_statistic_is_the_hdfe_coefficient(panel):
    out = sp.ri_test(
        panel,
        y="y",
        treat="exposure",
        interact="post",
        stat="ols",
        absorb=FE,
        covariates=["x"],
        cluster="occ",
        n_perms=50,
        seed=1,
    )
    b, _ = _refit(panel)
    assert out["observed"] == pytest.approx(b, rel=1e-7)


def test_permuted_statistic_equals_refit_on_permuted_exposure(panel):
    """Draw the same cluster permutation by hand and refit."""
    kw = dict(
        y="y",
        treat="exposure",
        interact="post",
        absorb=FE,
        covariates=["x"],
        cluster="occ",
        n_perms=3,
        seed=7,
    )
    ob = sp.ri_test(panel, stat="ols", **kw)
    ot = sp.ri_test(panel, stat="ols_t", **kw)
    rng = np.random.default_rng(7)
    ids = np.unique(panel.occ)
    level = panel.groupby("occ").exposure.first().loc[ids].to_numpy()
    perm = rng.permutation(level)
    p = panel.assign(d=perm[np.searchsorted(ids, panel.occ)] * panel.post)
    b, t = _refit(p)
    assert ob["perm_distribution"][0] == pytest.approx(b, rel=1e-7)
    # the t differs from hdfe_ols's by the constant small-sample factor
    ratio = ot["perm_distribution"][0] / t
    _, t_obs = _refit(panel)
    assert ot["observed"] / t_obs == pytest.approx(ratio, rel=1e-7)


def test_row_level_permutation_with_absorb(panel):
    out = sp.ri_test(panel, y="y", treat="d", stat="ols", absorb=FE, n_perms=5, seed=3)
    b = float(sp.hdfe_ols("y ~ d | firm + occ + city^q", panel, tol=1e-12).coef["d"])
    assert out["observed"] == pytest.approx(b, rel=1e-7)
    assert out["perm_distribution"].size == 5


def test_treatment_varying_within_cluster_fails_loudly(panel):
    with pytest.raises(sp.exceptions.MethodIncompatibility, match="varies within"):
        sp.ri_test(panel, y="y", treat="d", stat="ols", cluster="occ", n_perms=5)


def test_binary_statistics_reject_continuous_treatment(panel):
    with pytest.raises(sp.exceptions.MethodIncompatibility, match="binary"):
        sp.ri_test(panel, y="y", treat="exposure", stat="diff_means", n_perms=5)


def test_absorb_needs_regression_statistic(panel):
    with pytest.raises(sp.exceptions.MethodIncompatibility, match="ols"):
        sp.ri_test(panel, y="y", treat="d", stat="t", absorb="firm", n_perms=5)


def test_ols_then_ols_t_reuses_the_sweep_without_changing_results(panel):
    from statspai.inference import randomization as ri

    kw = dict(
        y="y",
        treat="exposure",
        interact="post",
        absorb=FE,
        covariates=["x"],
        cluster="occ",
        n_perms=40,
        seed=5,
    )
    ri._SWEEP_CACHE.clear()
    sp.ri_test(panel, stat="ols", **kw)
    assert len(ri._SWEEP_CACHE) == 1
    reused = sp.ri_test(panel, stat="ols_t", **kw)
    ri._SWEEP_CACHE.clear()
    fresh = sp.ri_test(panel, stat="ols_t", **kw)
    np.testing.assert_array_equal(
        reused["perm_distribution"], fresh["perm_distribution"]
    )
    assert reused["observed"] == fresh["observed"]
    # a different outcome is a different design: no stale reuse
    other = sp.ri_test(panel.assign(y=panel.y * 2), stat="ols", **kw)
    assert other["observed"] == pytest.approx(
        2 * sp.ri_test(panel, stat="ols", **kw)["observed"], rel=1e-9
    )
