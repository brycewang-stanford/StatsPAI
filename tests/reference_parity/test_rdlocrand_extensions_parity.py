"""Local randomization beyond the unadjusted difference in means.

The options exercised by Cattaneo, Idrobo & Titiunik (2024), *A Practical
Introduction to Regression Discontinuity Designs: Extensions*: kernels,
polynomial adjustment, a non-zero null, Bernoulli assignment, fuzzy designs,
test-inversion intervals, window selection and the binomial tests of
``rddensity``. Reference: R ``rdlocrand`` 2.0 and ``rddensity`` 2.6
(``_fixtures/_generate_rdlocrand_extensions_R.R``).

Evidence, by kind
-----------------
* **T2, same bytes.** Observed statistics, large-sample p-values, window
  counts and binomial p-values are deterministic and are held to 1e-9.
* **S, stochastic screen.** A randomization p-value is an RNG draw. The
  mean over 60 seeds is compared with R's mean over its own 60 seeds; the
  tolerance is stated in the test. This is a screen, not parity.
* **T4, documented divergence, with independent evidence.** Two places:

  1. *Polynomial adjustment* (``p > 0``). ``rdlocrand`` re-randomizes
     treatment labels with the scores held fixed. The observed statistic
     extrapolates each side's fit to the cutoff; a relabelled sample fits
     through the interior. On data with no effect that test rejects far
     above its level (``test_label_permutation_overrejects_with_p1``),
     while permuting outcomes against (score, assignment) pairs holds it
     (``test_rdrandinf_p1_holds_its_level``). The statistic and the
     large-sample p-value still agree with R to 1e-9.
  2. *The first window of* ``rdwinselect``. ``rdlocrand`` 2.0 starts one
     observation short on the left of the cutoff. Its documentation, and
     the output printed in the book (Snippet 2.5), describe the rule
     implemented here (``test_first_window_holds_obsmin_on_each_side``).
     With the windows given, everything else agrees to 1e-9.
"""

from __future__ import annotations

import json
import pathlib
import warnings

import numpy as np
import pandas as pd
import pytest
from scipy import stats

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import statspai as sp

_FIX = pathlib.Path(__file__).parent / "_fixtures"
RTOL = 1e-9


@pytest.fixture(scope="module")
def rjson():
    path = _FIX / "rdlocrand_extensions_R.json"
    if not path.exists():  # pragma: no cover
        pytest.skip("run _generate_rdlocrand_extensions_R.R to build the fixture")
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def senate():
    df = pd.read_csv(_FIX / "rdsenate.csv")
    # The fixture's deterministic take-up rule: every fourth unit defies.
    took = (df["margin"] >= 0).astype(float).to_numpy()
    flip = (np.arange(1, len(df) + 1) % 4) == 0
    took[flip] = 1 - took[flip]
    df["took"] = took
    return df


def _fit(senate, **kw):
    kw.setdefault("n_perms", 20)
    kw.setdefault("ci", False)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.rdrandinf(senate, y="vote", x="margin", **kw)


# ── rdrandinf: deterministic quantities ─────────────────────────────────

_CASES = {
    "tri": dict(wl=-5, wr=5, kernel="triangular"),
    "epan": dict(wl=-5, wr=5, kernel="epan"),
    "tri_asym": dict(wl=-3, wr=5, kernel="triangular"),
    "p1": dict(wl=-5, wr=5, p=1),
    "p2": dict(wl=-5, wr=5, p=2),
    "tri_p1": dict(wl=-5, wr=5, p=1, kernel="triangular"),
    "p1_eval": dict(wl=-5, wr=5, p=1, evall=-2, evalr=2),
    "null3": dict(wl=-5, wr=5, nulltau=3),
    "placebo_c2": dict(c=2, wl=0.5, wr=3.5),
}


@pytest.mark.parametrize("case", sorted(_CASES))
def test_observed_statistic_matches_r(rjson, senate, case):
    res = _fit(senate, **_CASES[case])
    got = res.model_info["results_by_stat"]["diffmeans"]["observed_stat"]
    assert got == pytest.approx(rjson[case]["obs_stat"], rel=RTOL)


@pytest.mark.parametrize("case", sorted(_CASES))
def test_large_sample_pvalue_matches_r(rjson, senate, case):
    """HC2 variance of the side-specific weighted fits, normal reference."""
    res = _fit(senate, **_CASES[case])
    assert res.model_info["pvalue_asymptotic"] == pytest.approx(
        rjson[case]["asy_pvalue"], rel=RTOL
    )


def test_kolmogorov_smirnov_under_a_nonzero_null(rjson, senate):
    res = _fit(senate, wl=-5, wr=5, nulltau=3, statistic="ksmirnov")
    got = res.model_info["results_by_stat"]["ksmirnov"]
    assert got["observed_stat"] == pytest.approx(
        rjson["ks_null3"]["obs_stat"], rel=RTOL
    )
    assert got["pvalue_asymptotic"] == pytest.approx(
        rjson["ks_null3"]["asy_pvalue"], rel=RTOL
    )


def test_window_endpoints_are_on_the_score_scale(rjson, senate):
    """``wl`` / ``wr`` are endpoints, not offsets: a placebo cutoff at 2."""
    mi = _fit(senate, c=2, wl=0.5, wr=3.5).model_info
    assert (mi["n_left"], mi["n_right"]) == (
        rjson["placebo_c2"]["Nl"],
        rjson["placebo_c2"]["Nr"],
    )
    assert mi["window"] == (0.5, 3.5)


def test_window_that_misses_the_cutoff_is_refused(senate):
    """The old offset convention must fail loudly, not pick another window."""
    with pytest.raises(ValueError, match="does not contain the cutoff"):
        _fit(senate, c=50, wl=-5, wr=5)


def test_nonzero_null_shifts_the_statistic_not_the_estimate(senate):
    base = _fit(senate, wl=-5, wr=5)
    shifted = _fit(senate, wl=-5, wr=5, nulltau=3)
    assert shifted.estimate == pytest.approx(base.estimate)
    stat = shifted.model_info["results_by_stat"]["diffmeans"]["observed_stat"]
    assert stat == pytest.approx(base.estimate - 3)


def test_power_is_the_two_sided_z_test_power(senate):
    res = _fit(senate, wl=-5, wr=5)
    mi = res.model_info
    assert mi["power_d"] == pytest.approx(0.5 * mi["sd_left"])
    z = stats.norm.ppf(0.975)
    ratio = mi["power_d"] / res.se
    want = 1 - stats.norm.cdf(z - ratio) + stats.norm.cdf(-z - ratio)
    assert mi["power"] == pytest.approx(want, rel=1e-12)
    assert _fit(senate, wl=-5, wr=5, d=7.0).model_info["power_d"] == 7.0


def test_rank_statistics_refuse_an_adjusted_fit(senate):
    with pytest.raises(ValueError, match="not defined for"):
        _fit(senate, wl=-5, wr=5, p=1, statistic="ranksum")
    with pytest.raises(ValueError, match="not defined for"):
        _fit(senate, wl=-5, wr=5, kernel="triangular", statistic="all")


# ── fuzzy ───────────────────────────────────────────────────────────────


def test_fuzzy_itt_matches_r(rjson, senate):
    res = _fit(senate, wl=-5, wr=5, fuzzy="took")
    mi = res.model_info
    assert mi["itt"] == pytest.approx(rjson["fuzzy_itt"]["obs_stat"], rel=RTOL)
    assert mi["pvalue_asymptotic"] == pytest.approx(
        rjson["fuzzy_itt"]["asy_pvalue"], rel=RTOL
    )
    # The estimate is the effect on compliers; the test behind `pvalue` is
    # the randomization test on the reduced form.
    assert res.estimate == pytest.approx(rjson["fuzzy_tsls"]["obs_stat"], rel=RTOL)
    assert res.pvalue == mi["pvalue_permutation"]


def test_fuzzy_tsls_matches_r(rjson, senate):
    res = _fit(senate, wl=-5, wr=5, fuzzy="took", fuzzy_stat="tsls")
    assert res.estimate == pytest.approx(rjson["fuzzy_tsls"]["obs_stat"], rel=RTOL)
    assert res.pvalue == pytest.approx(rjson["fuzzy_tsls"]["asy_pvalue"], rel=RTOL)
    # A large-sample statistic has no randomization p-value; R prints NA.
    assert np.isnan(res.model_info["pvalue_permutation"])


def test_fuzzy_pvalue_is_not_the_permuted_wald_ratio(senate):
    """The defect this replaced.

    Permuting the instrument and recomputing the ratio puts a first stage
    near zero in the denominator of most draws, so the "null distribution"
    is dominated by exploded ratios and the p-value says nothing about the
    effect: here the reduced form is significant at any level and that
    procedure returned a p-value above one half.
    """
    res = _fit(senate, wl=-5, wr=5, fuzzy="took", n_perms=1000)
    assert res.pvalue < 0.01
    assert res.model_info["pvalue_tsls"] < 0.01


def test_fuzzy_with_no_first_stage_is_refused(senate):
    df = senate.assign(flat=1.0)
    with pytest.raises(ValueError, match="first stage"):
        _fit(df, wl=-5, wr=5, fuzzy="flat")


# ── randomization p-values: a stochastic screen ─────────────────────────


def _seed_mean(senate, **kw):
    ps = [
        sp.rdrandinf(
            senate,
            y="termshouse",
            x="margin",
            wl=-1,
            wr=1,
            seed=s,
            ci=False,
            **kw,
        ).pvalue
        for s in range(60)
    ]
    return float(np.mean(ps)), float(np.std(ps, ddof=1))


@pytest.mark.parametrize(
    "name,kw",
    [
        ("diffmeans", {}),
        ("ranksum", {"statistic": "ranksum"}),
        ("triangular", {"kernel": "triangular"}),
        ("bernoulli", {"bernoulli": 0.5}),
    ],
)
def test_randomization_pvalue_seed_mean_screen(rjson, senate, name, kw):
    """S: the 60-seed mean against R's own 60-seed mean.

    Each p-value is a mean of 1000 Bernoulli draws, so a 60-seed mean has
    Monte Carlo SD of about ``sqrt(p (1 - p) / 60000)`` <= 0.002 on each
    side. 0.02 is roughly seven of those; it screens for a wrong
    randomization scheme, which moves the p-value by far more.
    """
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mean, _ = _seed_mean(senate, **kw)
    assert mean == pytest.approx(rjson["seedmean"][name], abs=0.02)


# ── polynomial adjustment: the documented divergence ────────────────────


def _null_draw(rng, n=40):
    x = rng.uniform(-1, 1, n)
    return pd.DataFrame({"x": x, "y": rng.normal(size=n)})


def test_rdrandinf_p1_holds_its_level():
    """No effect, 40 observations, p = 1: the 5% test rejects about 5%."""
    rng = np.random.default_rng(20261004)
    reject, n_sim = 0, 600
    for i in range(n_sim):
        df = _null_draw(rng)
        res = sp.rdrandinf(
            df, y="y", x="x", wl=-1, wr=1, p=1, n_perms=199, seed=i, ci=False
        )
        reject += res.pvalue <= 0.05
    rate = reject / n_sim
    # Binomial(600, 0.05) has SD 0.0089; the band is +/- 3 SD.
    assert 0.02 <= rate <= 0.08, rate


def test_label_permutation_overrejects_with_p1():
    """What re-randomizing labels with scores fixed does on the same design.

    The independent evidence for the divergence: the observed statistic is
    a boundary extrapolation, the relabelled ones are interior fits with a
    quarter of its variance, so a true null is rejected several times too
    often. R's seed-mean p-value for p = 1 sits far below its own
    large-sample p-value for the same reason (fixture ``seedmean``).
    """

    def intercept_gap(y, x, t):
        out = []
        for side in (True, False):
            m = t == side
            design = np.column_stack([np.ones(m.sum()), x[m]])
            out.append(np.linalg.lstsq(design, y[m], rcond=None)[0][0])
        return out[0] - out[1]

    rng = np.random.default_rng(20261004)
    reject, n_sim = 0, 300
    for _ in range(n_sim):
        df = _null_draw(rng)
        x, y = df["x"].to_numpy(), df["y"].to_numpy()
        t = x >= 0
        if t.sum() < 5 or (~t).sum() < 5:
            continue
        obs = abs(intercept_gap(y, x, t))
        draws = [abs(intercept_gap(y, x, rng.permutation(t))) for _ in range(199)]
        reject += np.mean(np.array(draws) >= obs) <= 0.05
    assert reject / n_sim > 0.20


def test_reference_p1_randomization_pvalue_departs_from_its_own_asymptotic(rjson):
    """The same thing, read off the reference's own numbers."""
    sm = rjson["seedmean"]
    assert sm["p1"] < 0.5 * sm["p1_asy"]


def test_p1_randomization_pvalue_tracks_the_large_sample_one(rjson, senate):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mean, _ = _seed_mean(senate, p=1)
    assert abs(mean - rjson["seedmean"]["p1_asy"]) < 0.1


def test_pooled_residualization_is_gone(senate):
    """The earlier ``p=1``: residualize on one pooled trend, then compare.

    The score is collinear with treatment inside the window, so the pooled
    trend absorbed the effect: 2.9 on the Senate data where the difference
    in intercepts is 13.3.
    """
    res = _fit(senate, wl=-2.5, wr=2.5, p=1)
    m = senate["margin"].between(-2.5, 2.5)
    x, y = senate.loc[m, "margin"].to_numpy(), senate.loc[m, "vote"].to_numpy()
    fits = [np.polyfit(x[s], y[s], 1)[1] for s in (x >= 0, x < 0)]
    assert res.estimate == pytest.approx(fits[0] - fits[1], rel=1e-10)


# ── confidence interval by test inversion ───────────────────────────────


def test_interval_is_the_set_of_effects_not_rejected(senate):
    grid = np.arange(0.0, 20.01, 0.25)
    res = _fit(senate, wl=-2.5, wr=2.5, ci=grid, n_perms=2000, seed=3)
    lo, hi = res.ci
    assert res.model_info["ci_method"] == "randomization test inversion"
    assert lo in grid and hi in grid
    for tau0, inside in ((lo, True), (hi, True), (lo - 1.0, False), (hi + 1.0, False)):
        p = _fit(senate, wl=-2.5, wr=2.5, nulltau=tau0, n_perms=2000, seed=11).pvalue
        assert (p > 0.05) == inside, (tau0, p)


def test_truncated_grid_warns(senate):
    with pytest.warns(UserWarning, match="edge of the grid"):
        sp.rdrandinf(
            senate, y="vote", x="margin", wl=-2.5, wr=2.5, ci=np.linspace(8, 10, 21)
        )


def test_grid_that_misses_the_interval_warns(senate):
    with pytest.warns(UserWarning, match="does not cover"):
        res = sp.rdrandinf(
            senate, y="vote", x="margin", wl=-2.5, wr=2.5, ci=[100.0, 101.0]
        )
    assert res.model_info["ci_method"].startswith("large-sample")


# ── rdwinselect ─────────────────────────────────────────────────────────

_COVS = ["class", "termshouse", "termssenate"]


def test_winselect_fixed_sequence_matches_r(rjson, senate):
    """Counts, binomial test, minimum balance p-value and who attains it."""
    ref = rjson["winselect_approx"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = sp.rdwinselect(
            senate,
            x="margin",
            covs=_COVS,
            wmin=0.5,
            wstep=0.25,
            nwindows=12,
            approx=True,
        )
    np.testing.assert_allclose(out["window_right"], ref["w_right"], rtol=1e-12)
    assert out["n_left"].tolist() == ref["Nl"]
    assert out["n_right"].tolist() == ref["Nr"]
    np.testing.assert_allclose(out["binom_pvalue"], ref["binom"], rtol=RTOL)
    np.testing.assert_allclose(out["p_value"], ref["p_value"], rtol=RTOL)
    assert out["variable"].tolist() == [_COVS[i - 1] for i in ref["variable"]]


def test_no_window_is_recommended_when_the_first_fails(rjson, senate):
    """R returns no window here either: the first p-value is 0.113 < 0.15."""
    assert rjson["winselect_approx"]["rec_left"] is None
    with pytest.warns(UserWarning, match="smallest window already fails"):
        out = sp.rdwinselect(
            senate,
            x="margin",
            covs=_COVS,
            wmin=0.5,
            wstep=0.25,
            nwindows=12,
            approx=True,
        )
    assert out.attrs["recommended_window"] is None


def test_recommended_window_is_the_last_before_balance_fails(senate):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = sp.rdwinselect(
            senate,
            x="margin",
            covs=_COVS,
            wmin=0.5,
            wstep=0.25,
            nwindows=12,
            approx=True,
            alpha=0.11,
        )
    passed = (out["p_value"] >= 0.11).to_numpy()
    assert passed.all(), "at level 0.11 every window on this grid passes"
    assert out.attrs["recommended_window"] == (-3.25, 3.25)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = sp.rdwinselect(
            senate,
            x="margin",
            covs=_COVS,
            wmin=0.5,
            wstep=0.25,
            nwindows=12,
            approx=True,
            alpha=0.18,
        )
    # p-values run 0.113, 0.171, 0.191, 0.126, ...: at 0.18 the first window
    # fails, and a later pass does not rescue it.
    assert out.attrs["recommended_window"] is None


def test_first_window_holds_obsmin_on_each_side(senate):
    """The documented rule, and where rdlocrand 2.0 departs from it."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = sp.rdwinselect(senate, x="margin", covs=_COVS, wobs=2, approx=True)
    x = senate["margin"].to_numpy()
    left = np.sort(-x[x < 0])
    right = np.sort(x[x >= 0])
    first = out["window_right"].iloc[0]
    assert first == pytest.approx(max(left[9], right[9]))
    assert (left <= first).sum() >= 10 and (right <= first).sum() >= 10
    # Smallest such window: one notch narrower loses the tenth observation.
    narrower = np.nextafter(first, 0)
    assert min((left <= narrower).sum(), (right <= narrower).sum()) < 10


def test_each_step_adds_wobs_on_each_side(senate):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = sp.rdwinselect(senate, x="margin", wobs=3, nwindows=8)
    x = senate["margin"].to_numpy()
    left = np.sort(-x[x < 0])
    right = np.sort(x[x >= 0])
    n_l = [(left <= w).sum() for w in out["window_right"]]
    n_r = [(right <= w).sum() for w in out["window_right"]]
    assert all(b - a >= 3 for a, b in zip(n_l, n_l[1:]))
    assert all(b - a >= 3 for a, b in zip(n_r, n_r[1:]))
    # ... and by no more than needed: one side gains exactly wobs, unless
    # tied scores bring extra observations with them.
    for (a_l, b_l), (a_r, b_r) in zip(zip(n_l, n_l[1:]), zip(n_r, n_r[1:])):
        assert min(b_l - a_l, b_r - a_r) <= 3 + 1


def test_without_covariates_only_the_binomial_test_is_reported(senate):
    out = sp.rdwinselect(senate, x="margin", wmin=1.0, nwindows=1)
    row = out.iloc[0]
    assert np.isnan(row["p_value"]) and row["variable"] is None
    want = stats.binomtest(int(row["n_left"]), int(row["n_left"] + row["n_right"]))
    assert row["binom_pvalue"] == pytest.approx(want.pvalue, rel=1e-12)
    assert out.attrs["recommended_window"] is None


def test_asymmetric_windows_grow_each_side_separately():
    x = np.r_[-np.arange(1, 41, dtype=float), np.arange(40) + 0.5]
    df = pd.DataFrame({"x": x})
    out = sp.rdwinselect(df, x="x", obsmin=3, wobs=2, wasymmetric=True, nwindows=4)
    assert out["window_left"].tolist() == [-3.0, -5.0, -7.0, -9.0]
    assert out["window_right"].tolist() == [2.5, 4.5, 6.5, 8.5]
    assert out["n_left"].tolist() == out["n_right"].tolist() == [3, 5, 7, 9]


def test_mass_points_enter_whole():
    """A window never splits a mass point, so counts can exceed the target."""
    x = np.r_[
        np.repeat(-np.arange(1, 11, dtype=float), 3), np.repeat(np.arange(10) + 0.5, 2)
    ]
    out = sp.rdwinselect(pd.DataFrame({"x": x}), x="x", obsmin=3, wobs=2, nwindows=4)
    assert out["window_right"].tolist() == [1.5, 2.5, 3.5, 4.5]
    assert out["n_left"].tolist() == [3, 6, 9, 12]
    assert out["n_right"].tolist() == [4, 6, 8, 10]


def test_window_sequence_stops_when_the_data_run_out():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"x": rng.uniform(-1, 1, 60)})
    out = sp.rdwinselect(df, x="x", nwindows=50)
    assert 1 <= len(out) < 50


# ── rddensity binomial tests ────────────────────────────────────────────


def test_binomial_windows_set_by_the_user_match_r(rjson, senate):
    ref = rjson["bino_asym"]
    tab = sp.rddensity(
        senate, x="margin", bino_w=(1, 2), bino_wstep=(0.5, 1), bino_nw=3
    ).model_info["binomial_tests"]
    assert tab["n_left"].tolist() == ref["Nl"]
    assert tab["n_right"].tolist() == ref["Nr"]
    np.testing.assert_allclose(tab["pvalue"], ref["pval"], rtol=RTOL)
    assert tab["half_width"].tolist() == [1.0, 1.5, 2.0]
    assert tab["half_width_right"].tolist() == [2.0, 3.0, 4.0]


def test_binomial_null_other_than_one_half_matches_r(rjson, senate):
    ref = rjson["bino_p04"]
    tab = sp.rddensity(
        senate, x="margin", bino_w=0.75, bino_nw=4, bino_p=0.4
    ).model_info["binomial_tests"]
    assert tab["n_left"].tolist() == ref["Nl"]
    assert tab["n_right"].tolist() == ref["Nr"]
    np.testing.assert_allclose(tab["pvalue"], ref["pval"], rtol=RTOL)


def test_binomial_window_must_be_positive(senate):
    from statspai.exceptions import MethodIncompatibility

    with pytest.raises(MethodIncompatibility):
        sp.rddensity(senate, x="margin", bino_w=-1.0)
