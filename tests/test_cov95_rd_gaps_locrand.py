"""Branch tests for ``statspai.rd.locrand`` not reached elsewhere.

Covers the small statistic helpers (kept so the reference-parity test can
feed them R's own draws), the validation branches of ``sp.rdrandinf`` /
``sp.rdwinselect`` / ``sp.rdrbounds``, the Anderson-Rubin interval of a
fuzzy design, Bernoulli draws that leave one arm empty, and the on-request
figure of ``sp.rdsensitivity``.
"""

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
from scipy import stats

import statspai as sp
from statspai.exceptions import DataInsufficient, MethodIncompatibility
from statspai.rd import locrand as lr

matplotlib.use("Agg")


def _sharp(n=400, seed=0, tau=1.0):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, n)
    y = tau * (x >= 0) + 0.3 * x + rng.normal(0, 0.5, n)
    return pd.DataFrame({"y": y, "x": x, "z1": rng.normal(size=n)})


def _fuzzy(n=600, seed=1):
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1, 1, n)
    d = (rng.uniform(size=n) < 0.1 + 0.8 * (x >= 0)).astype(float)
    y = 2.0 * d + rng.normal(0, 0.5, n)
    return pd.DataFrame({"y": y, "x": x, "d": d})


# ------------------------------------------------------------ helpers


def test_polynomial_residuals_are_orthogonal_to_the_design():
    rng = np.random.default_rng(3)
    x = rng.normal(size=60)
    cov = rng.normal(size=60)
    y = 1 + 2 * x - x**2 + 0.5 * cov + rng.normal(size=60)
    # p = 2 with a one-dimensional covariate array
    r = lr._polynomial_residuals(y, x, 2, cov)
    for col in (np.ones(60), x, x**2, cov):
        assert abs(r @ col) < 1e-9
    # nothing to partial out: plain demeaning
    r0 = lr._polynomial_residuals(y, x, 0, np.empty((60, 0)))
    np.testing.assert_allclose(r0, y - y.mean(), atol=1e-14)


def test_statistic_helpers_and_degenerate_ranksum():
    y = np.array([1.0, 2.0, 3.0, 10.0, 11.0, 12.0])
    d = np.array([0, 0, 0, 1, 1, 1])
    assert lr._diffmeans(y, d) == pytest.approx(9.0)
    assert lr._ks_stat(y, d) == pytest.approx(1.0)
    assert lr._compute_stat(y, d, "diffmeans") == pytest.approx(9.0)
    assert lr._compute_stat(y, d, "ksmirnov") == pytest.approx(1.0)
    # one arm empty, and a constant outcome (zero rank variance)
    assert lr._ranksum_stat(y, np.ones(6, dtype=int)) == 0.0
    assert lr._ranksum_stat(np.full(6, 2.0), d) == 0.0


def test_pvalue_from_assignments_counts_exactly():
    y = np.array([1.0, 2.0, 3.0, 10.0, 11.0, 12.0])
    d = np.array([0, 0, 0, 1, 1, 1])
    obs = lr._diffmeans(y, d)
    draws = [
        d,  # equal to observed
        1 - d,  # -9: as extreme two-sided, not one-sided
        np.array([0, 1, 0, 1, 0, 1]),  # 1/3
        np.array([1, 0, 1, 0, 1, 0]),
    ]
    assert lr._pvalue_from_assignments(y, obs, draws, "diffmeans") == 0.5
    assert (
        lr._pvalue_from_assignments(y, obs, draws, "diffmeans", two_sided=False) == 0.25
    )


def test_permutation_pvalue_is_exact_share_of_its_own_draws():
    y = np.array([1.0, 2.0, 3.0, 10.0, 11.0, 12.0])
    d = np.array([0, 0, 0, 1, 1, 1])
    obs, pval = lr._permutation_pvalue(y, d, "diffmeans", 400, np.random.default_rng(5))
    assert obs == pytest.approx(9.0)
    # only 2 of the 20 assignments are as extreme: p should be near 0.1
    rng = np.random.default_rng(5)
    again = [rng.permutation(d) for _ in range(400)]
    share = np.mean([abs(lr._diffmeans(y, a)) >= 9.0 - 1e-12 for a in again])
    assert pval == pytest.approx(share)
    assert 0.03 < pval < 0.2


def test_asymptotic_pvalue_diffmeans_is_welch_z():
    rng = np.random.default_rng(8)
    y = rng.normal(size=50)
    d = (np.arange(50) < 20).astype(int)
    y[d == 1] += 0.7
    diff, pval = lr._asymptotic_pvalue(y, d, "diffmeans")
    y1, y0 = y[d == 1], y[d == 0]
    se = np.sqrt(y1.var(ddof=1) / 20 + y0.var(ddof=1) / 30)
    assert diff == pytest.approx(y1.mean() - y0.mean())
    assert pval == pytest.approx(2 * stats.norm.cdf(-abs(diff / se)))


def test_asymptotic_pvalue_degenerate_cases():
    # one treated unit: no variance estimate
    y = np.array([1.0, 2.0, 3.0, 9.0])
    diff, pval = lr._asymptotic_pvalue(y, np.array([0, 0, 0, 1]), "diffmeans")
    assert diff == pytest.approx(7.0) and np.isnan(pval)
    # no within-arm variation: certain rejection or certain non-rejection
    d = np.array([0, 0, 1, 1])
    assert lr._asymptotic_pvalue(np.array([1.0, 1, 3, 3]), d, "diffmeans") == (2.0, 0.0)
    assert lr._asymptotic_pvalue(np.ones(4), d, "diffmeans") == (0.0, 1.0)
    z = d.astype(float)
    assert np.isnan(lr._asymptotic_for("diffmeans", y, z, 1.0, np.nan))
    assert lr._asymptotic_for("diffmeans", y, z, 1.0, 0.0) == 0.0
    assert lr._asymptotic_for("diffmeans", y, z, 0.0, 0.0) == 1.0


# ------------------------------------------------------- rdrandinf


@pytest.mark.parametrize(
    "kwargs, fragment",
    [
        ({"p": -1}, "non-negative"),
        ({"fuzzy": "d", "fuzzy_stat": "wald"}, "fuzzy_stat"),
        ({"p": 1, "bernoulli": 0.5}, "bernoulli= needs p=0"),
        ({"fuzzy": "d", "kernel": "triangular"}, "fuzzy= needs p=0"),
        ({"bernoulli": 1.0}, "strictly in (0, 1)"),
        ({"ci": [0.5]}, "at least two effects"),
        (
            {"fuzzy": "d", "fuzzy_stat": "tsls", "ci": [0.0, 1.0, 2.0]},
            "large-sample",
        ),
    ],
)
def test_rdrandinf_rejects_incompatible_options(kwargs, fragment):
    df = _fuzzy()
    with pytest.raises(MethodIncompatibility) as err:
        sp.rdrandinf(df, y="y", x="x", wl=-0.5, wr=0.5, n_perms=50, **kwargs)
    assert fragment in str(err.value)


def test_rdrandinf_fuzzy_anderson_rubin_interval_comes_from_the_grid():
    df = _fuzzy()
    grid = np.linspace(1.0, 3.0, 201)
    res = sp.rdrandinf(
        df, y="y", x="x", wl=-0.5, wr=0.5, fuzzy="d", ci=grid, n_perms=200
    )
    lo, hi = res.ci
    # the interval is a set of grid points, and it holds the true effect
    assert np.min(np.abs(grid - lo)) < 1e-12 and np.min(np.abs(grid - hi)) < 1e-12
    assert lo < 2.0 < hi
    assert res.model_info["ci_method"] == "randomization test inversion"
    assert res.model_info["ci_grid_truncated"] is False
    assert abs(res.estimate - 2.0) < 0.3


def test_rdrandinf_bernoulli_drops_draws_with_an_empty_arm():
    # Six units: about 3% of Bernoulli(1/2) draws put everyone in one arm,
    # where a difference in means does not exist. Those draws are dropped.
    df = pd.DataFrame(
        {
            "x": [-0.3, -0.2, -0.1, 0.1, 0.2, 0.3],
            "y": [0.0, 0.2, 0.1, 5.0, 5.1, 5.3],
        }
    )
    res = sp.rdrandinf(
        df, y="y", x="x", wl=-0.5, wr=0.5, bernoulli=0.5, n_perms=2000, ci=False
    )
    assert res.estimate == pytest.approx(5.0333333, abs=1e-6)
    # exactly 2 of the 62 two-armed assignments are as extreme as observed
    assert res.pvalue == pytest.approx(2 / 62, abs=0.012)


# ----------------------------------------------- rdwinselect & friends


def test_rdwinselect_ttest_is_an_alias_and_unknown_statistic_is_refused():
    df = _sharp()
    kw = dict(x="x", covs=["z1"], nwindows=3, approx=True)
    a = sp.rdwinselect(df, statistic="ttest", **kw)
    b = sp.rdwinselect(df, statistic="diffmeans", **kw)
    pd.testing.assert_frame_equal(a, b)
    with pytest.raises(MethodIncompatibility, match="tatistic"):
        sp.rdwinselect(df, statistic="median", **kw)


def test_rdwinselect_needs_both_sides():
    df = _sharp()
    with pytest.raises(DataInsufficient, match="both sides"):
        sp.rdwinselect(df[df["x"] > 0], x="x", covs=["z1"])


def test_balance_pvalue_branches():
    rng = np.random.default_rng(4)
    xc = rng.uniform(-1, 1, 80)
    z = (xc >= 0).astype(int)
    vals = rng.normal(size=80)
    common = dict(bw=(1.0, 1.0), n_perms=50, rng=rng)
    # a constant covariate cannot be unbalanced
    assert np.isnan(
        lr._balance_pvalue(
            np.ones(80),
            xc,
            z,
            statistic="diffmeans",
            p=0,
            kernel="uniform",
            approx=True,
            **common,
        )
    )
    with pytest.raises(MethodIncompatibility, match="need p=0"):
        lr._balance_pvalue(
            vals,
            xc,
            z,
            statistic="ranksum",
            p=1,
            kernel="uniform",
            approx=True,
            **common,
        )
    # large-sample rank-sum p-value is the normal tail of the statistic
    pv = lr._balance_pvalue(
        vals, xc, z, statistic="ranksum", p=0, kernel="uniform", approx=True, **common
    )
    assert pv == pytest.approx(
        2 * stats.norm.cdf(-abs(lr._ranksum_stat(vals, z))), rel=1e-12
    )


def test_rdsensitivity_attaches_a_two_panel_figure_only_on_request():
    df = _sharp(300)
    kw = dict(y="y", x="x", wlist=[0.3, 0.6], n_perms=100, seed=1)
    quiet = sp.rdsensitivity(df, **kw)
    assert "figure" not in quiet.attrs
    out = sp.rdsensitivity(df, plot=True, **kw)
    fig = out.attrs["figure"]
    assert len(fig.axes) == 2
    line = fig.axes[0].lines[0]
    np.testing.assert_allclose(line.get_xdata(), [0.3, 0.6])
    np.testing.assert_allclose(line.get_ydata(), out["estimate"].to_numpy())
    np.testing.assert_allclose(
        fig.axes[1].lines[0].get_ydata(), out["pvalue"].to_numpy()
    )
    plt.close(fig)
    pd.testing.assert_frame_equal(out, quiet)


def test_rdrbounds_validation_and_diffmeans_statistic():
    df = _sharp(200)
    with pytest.raises(MethodIncompatibility, match="wl and wr"):
        sp.rdrbounds(df, y="y", x="x")
    with pytest.raises(MethodIncompatibility, match="ranksum"):
        sp.rdrbounds(df, y="y", x="x", wl=-0.3, wr=0.3, statistic="ksmirnov")
    tab = sp.rdrbounds(
        df,
        y="y",
        x="x",
        wl=-0.3,
        wr=0.3,
        gamma_list=[1.0, 2.0, 6.0],
        statistic="diffmeans",
        n_perms=200,
    )
    assert (tab["pvalue_upper"] >= tab["pvalue_lower"]).all()
    # more hidden bias can only widen the bounds
    assert tab["pvalue_upper"].is_monotonic_increasing
    assert tab["pvalue_lower"].is_monotonic_decreasing
