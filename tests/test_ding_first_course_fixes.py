"""Corrections found while working through Ding's *A First Course in Causal
Inference* (2024). See ``docs/dev/2026-10-05-ding-first-course-review.md``.

* The confidence interval of ``sp.fisher_exact`` is the exact inversion of
  the randomization test.
* Lee / Zhang-Rubin bounds say so when ties at the trimming quantile leave
  ``trimming='quantile'`` unable to trim.
* Principal-score weighting handles one-sided noncompliance.
* The g-formula accepts a covariate named at more than one time.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp

# ---------------------------------------------------------------------------
# sp.fisher_exact: the interval is the set of effects the test does not reject
# ---------------------------------------------------------------------------


def _small_experiment(seed=3, n=14):
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "d": rng.permutation(np.r_[np.ones(n // 2), np.zeros(n - n // 2)]),
            "x": rng.normal(size=n),
        }
    )
    df["y"] = 1.0 * df.d + 0.8 * df.x + rng.normal(size=n)
    return df


@pytest.mark.parametrize("controls", [None, ["x"]])
def test_interval_is_the_exact_inversion_of_the_test(controls):
    # C(14, 7) = 3432 assignments are enumerated, so both the interval and
    # the brute-force p-values below are exact. Just inside each end the test
    # of Y - tau0 * D must not reject, just outside it must.
    df = _small_experiment()
    alpha = 0.10
    res = sp.fisher_exact(df, "y", "d", controls=controls, n_perm=5000, alpha=alpha)
    assert res.n_perm == 3432
    lo, hi = res.ci
    assert lo < res.statistic < hi

    def pval(tau0):
        shifted = df.assign(y=df.y - tau0 * df.d)
        return sp.fisher_exact(
            shifted, "y", "d", controls=controls, n_perm=5000
        ).p_value

    eps = 1e-6
    assert pval(lo + eps) >= alpha and pval(hi - eps) >= alpha
    assert pval(lo - eps) < alpha and pval(hi + eps) < alpha


def test_interval_agrees_with_neyman_in_a_large_experiment():
    # With 400 units the randomization interval and the Neyman interval are
    # close. The grid interval of 1.38.0 was quantised at 6% of sd(Y) and
    # came out too short on data like these.
    rng = np.random.default_rng(11)
    n = 400
    d = rng.permutation(np.r_[np.ones(160), np.zeros(240)])
    y = 2.0 * d + rng.normal(0, 3, n)
    df = pd.DataFrame({"y": y, "d": d})
    res = sp.fisher_exact(df, "y", "d", n_perm=4000, seed=5)
    neyman = sp.difference_in_means(df, "y", "d")
    assert res.ci[0] == pytest.approx(neyman.ci[0], abs=0.12 * neyman.se * 2)
    assert res.ci[1] == pytest.approx(neyman.ci[1], abs=0.12 * neyman.se * 2)
    # and the same seed gives the same interval
    again = sp.fisher_exact(df, "y", "d", n_perm=4000, seed=5)
    assert again.ci == res.ci


def test_interval_covers_a_constant_effect():
    # 300 experiments with a constant effect of 1: a 90% interval should
    # cover about 270 times. Binomial(300, 0.9) has sd 5.2; the bounds are
    # four sd either side.
    rng = np.random.default_rng(2026)
    cover = 0
    for _ in range(300):
        d = rng.permutation(np.r_[np.ones(15), np.zeros(15)])
        y = 1.0 * d + rng.exponential(size=30)
        res = sp.fisher_exact(
            pd.DataFrame({"y": y, "d": d}),
            "y",
            "d",
            n_perm=400,
            alpha=0.10,
            seed=int(rng.integers(1 << 30)),
        )
        cover += res.ci[0] <= 1.0 <= res.ci[1]
    assert 249 <= cover <= 291


def test_interval_is_unbounded_when_the_design_cannot_reject():
    # Two treated out of four: six assignments, smallest two-sided p = 1/3.
    df = pd.DataFrame({"d": [1, 1, 0, 0], "y": [3.0, 4.0, 1.0, 2.0]})
    with pytest.warns(UserWarning, match="unbounded"):
        res = sp.fisher_exact(df, "y", "d")
    assert res.ci == (-np.inf, np.inf)
    assert res.p_value == pytest.approx(1 / 3)


# ---------------------------------------------------------------------------
# Lee / Zhang-Rubin bounds on a binary outcome
# ---------------------------------------------------------------------------


def _yang_small():
    """The 2x2x2 table of chapter 26 (truncation by death, binary outcome)."""
    z = np.r_[np.ones(431), np.zeros(429)]
    m = np.r_[np.ones(322), np.zeros(109), np.ones(277), np.zeros(152)]
    y = np.r_[
        np.ones(54),
        np.zeros(268),
        np.full(109, np.nan),
        np.ones(59),
        np.zeros(218),
        np.full(152, np.nan),
    ]
    return pd.DataFrame({"z": z, "m": m, "y": y})


def _sharp_bounds():
    pi11, pi00 = 277 / 429, 109 / 431
    pi10 = 1 - pi11 - pi00
    mu11, mu01 = 54 / 322, 59 / 277
    lo = ((pi11 + pi10) * mu11 - pi10) / pi11 - mu01
    hi = (pi11 + pi10) * mu11 / pi11 - mu01
    return lo, hi


def test_exact_trimming_gives_the_sharp_bounds_on_a_binary_outcome():
    lo, hi = _sharp_bounds()
    assert (lo, hi) == pytest.approx((-0.17602, -0.01896), abs=5e-6)
    df = _yang_small()
    res = sp.lee_bounds(df, "y", "z", "m", n_bootstrap=20, trimming="exact")
    assert res.model_info["lower_bound"] == pytest.approx(lo, rel=1e-10)
    assert res.model_info["upper_bound"] == pytest.approx(hi, rel=1e-10)
    ps = sp.principal_strat(df, "y", "z", "m", n_boot=20, seed=0, trimming="exact")
    assert ps.bounds["estimate"].to_numpy() == pytest.approx([lo, hi], rel=1e-10)


@pytest.mark.parametrize("which", ["lee", "principal_strat", "sace"])
def test_quantile_trimming_warns_when_ties_defeat_it(which):
    df = _yang_small()
    calls = {
        "lee": lambda tr: sp.lee_bounds(df, "y", "z", "m", n_bootstrap=10, trimming=tr),
        "principal_strat": lambda tr: sp.principal_strat(
            df, "y", "z", "m", n_boot=10, seed=0, trimming=tr
        ),
        "sace": lambda tr: sp.survivor_average_causal_effect(
            df, "y", "z", "m", n_boot=10, seed=0, trimming=tr
        ),
    }
    with pytest.warns(UserWarning, match="tied at the trimming quantile"):
        calls[which]("quantile")
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        calls[which]("exact")


def test_quantile_trimming_is_silent_on_a_continuous_outcome():
    rng = np.random.default_rng(0)
    n = 2000
    d = rng.integers(0, 2, n)
    s = (rng.uniform(size=n) < 0.5 + 0.2 * d).astype(int)
    y = np.where(s == 1, rng.normal(size=n) + d, np.nan)
    df = pd.DataFrame({"d": d, "s": s, "y": y})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = sp.lee_bounds(df, "y", "d", "s", n_bootstrap=10)
    assert res.model_info["lower_bound"] < res.model_info["upper_bound"]


# ---------------------------------------------------------------------------
# Principal scores under one-sided noncompliance
# ---------------------------------------------------------------------------


def test_principal_score_with_one_sided_noncompliance():
    # Nobody assigned to control takes the treatment, so there are no
    # always-takers and the control-arm principal score is not a regression.
    # Truth: effect 1 on compliers, 0 on never-takers, compliance depends on x
    # and so does the outcome (principal ignorability holds given x).
    rng = np.random.default_rng(7)
    n = 6000
    x = rng.normal(size=n)
    z = rng.integers(0, 2, n)
    complier = rng.uniform(size=n) < 1 / (1 + np.exp(-(0.3 + x)))
    m = (z * complier).astype(int)
    y = 5.0 + 1.5 * x + 1.0 * m + rng.normal(size=n)
    df = pd.DataFrame({"y": y, "z": z, "m": m, "x": x})
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # no perfect-separation warnings
        res = sp.principal_strat(
            df,
            "y",
            "z",
            "m",
            covariates=["x"],
            method="principal_score",
            n_boot=30,
            seed=1,
        )
    eff = res.effects.set_index("stratum")["estimate"]
    assert res.strata_proportions["always-taker"] == 0.0
    assert np.isnan(eff["Always-taker PCE"])
    assert eff["Complier PCE"] == pytest.approx(1.0, abs=0.12)
    assert eff["Never-taker PCE"] == pytest.approx(0.0, abs=0.12)
    assert res.model_info["principal_score_degraded"] is False

    # The complier estimate is the Hajek form of Ding and Lu's weighting
    # estimator: treated compliers against controls weighted by their
    # principal score.
    import statsmodels.api as sm

    fit = sm.Logit(m[z == 1], sm.add_constant(x[z == 1])).fit(disp=0)
    e = np.asarray(fit.predict(sm.add_constant(x)))
    c = z == 0
    by_hand = y[(z == 1) & (m == 1)].mean() - np.sum(y[c] * e[c]) / np.sum(e[c])
    assert eff["Complier PCE"] == pytest.approx(by_hand, abs=1e-5)


# ---------------------------------------------------------------------------
# g-formula: a covariate listed at more than one time
# ---------------------------------------------------------------------------


def _two_period(seed=0, n=20000):
    """The simulation of chapter 29: the effect of (1, 1) against (0, 0) is 3."""
    rng = np.random.default_rng(seed)
    x0 = rng.normal(size=n)
    z1 = rng.binomial(1, 1 / (1 + np.exp(-x0)))
    x1 = z1 + x0 + rng.normal(size=n)
    z2 = rng.binomial(1, 1 / (1 + np.exp(-(-0.5 + z1 + x1 + 0.5 * x0))))
    y = z2 + z1 + x1 + x0 + rng.normal(size=n)
    return pd.DataFrame(
        {"id": np.arange(n), "x0": x0, "z1": z1, "x1": x1, "z2": z2, "y": y}
    )


def test_gformula_recovers_the_effect_of_a_treatment_sequence():
    df = _two_period()

    def contrast(conf):
        on = sp.gformula_ice_fn(df, "id", "id", ["z1", "z2"], conf, "y", [1, 1])
        off = sp.gformula_ice_fn(df, "id", "id", ["z1", "z2"], conf, "y", [0, 0])
        return on.value - off.value, on.se

    est, se = contrast([["x0"], ["x1"]])
    assert est == pytest.approx(3.0, abs=4 * np.sqrt(2) * se)
    # x1 is affected by the first treatment, so conditioning on it in one
    # regression removes part of the effect: about 2 instead of 3.
    ols = sp.regress("y ~ z2 + z1 + x1 + x0", df).params
    assert ols["z1"] + ols["z2"] == pytest.approx(2.0, abs=0.1)
    # Naming the baseline covariate again at the second time is the same
    # model. It used to make the sandwich singular (LinAlgError).
    again, se_again = contrast([["x0"], ["x0", "x1"]])
    assert again == pytest.approx(est, rel=1e-12)
    assert se_again == pytest.approx(se, rel=1e-12)


def test_gformula_flat_confounder_list():
    rng = np.random.default_rng(0)
    n = 300
    l0 = rng.normal(size=n)
    a0, a1 = rng.binomial(1, 0.5, n), rng.binomial(1, 0.5, n)
    y = 1 + 0.8 * a0 + 1.2 * a1 + 0.5 * l0 + rng.normal(size=n)
    df = pd.DataFrame({"id": range(n), "L0": l0, "A0": a0, "A1": a1, "Y": y})
    flat = sp.gformula_ice_fn(df, "id", "id", ["A0", "A1"], ["L0"], "Y", [1, 1])
    nested = sp.gformula_ice_fn(df, "id", "id", ["A0", "A1"], [["L0"], []], "Y", [1, 1])
    assert np.isfinite(flat.se) and flat.se > 0
    assert flat.value == pytest.approx(nested.value, rel=1e-12)
    assert flat.se == pytest.approx(nested.se, rel=1e-12)
    assert flat.value == pytest.approx(3.0, abs=4 * flat.se)
