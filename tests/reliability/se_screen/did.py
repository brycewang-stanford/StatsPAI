"""SE calibration screen: difference-in-differences beyond the Track B rows.

Staggered design: 300 units, 10 periods, cohorts first treated in period 5
and period 7 (100 units each) and 100 never-treated units. Unit and period
effects, AR(1) errors within unit (rho 0.5), parallel trends exact, and a
constant treatment effect of 1.0, so every overall ATT and every post-period
event-study coefficient has true value 1.0 and every lead has true value 0.
Inference is clustered by unit, each estimator's default.

The synthetic-control rows use a block design (40 units, 20 periods, 10
units treated from period 15) with a one-factor outcome model.

``sp.callaway_santanna`` and ``sp.sun_abraham`` overall ATT are Track B rows
(``tests/coverage_monte_carlo``) and are not repeated here.

    python tests/reliability/se_screen/did.py [--only name] [--confirm]
"""

# flake8: noqa: E501
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _harness import Row, coef, main, tidy_row, wald  # noqa: E402,F401

FAMILY = "did"
NU, NT = 300, 10
TAU = 1.0
RHO = 0.5


def _sp():
    import statspai as sp

    return sp


def _ar1(rng, nu, nt, rho):
    e = np.zeros((nu, nt))
    e[:, 0] = rng.normal(size=nu) / np.sqrt(1 - rho**2)
    for j in range(1, nt):
        e[:, j] = rho * e[:, j - 1] + rng.normal(size=nu)
    return e


def d_staggered(rng, nu=NU, nt=NT, cohorts=(5, 7, 0)):
    unit = np.repeat(np.arange(nu), nt)
    t = np.tile(np.arange(1, nt + 1), nu)
    g = np.array(cohorts)[np.arange(nu) % len(cohorts)][unit]
    a = rng.normal(size=nu)[unit]
    lam = rng.normal(size=nt)[t - 1]
    d = ((g > 0) & (t >= g)).astype(int)
    y = a + lam + TAU * d + _ar1(rng, nu, nt, RHO).ravel()
    return pd.DataFrame(
        {
            "unit": unit,
            "t": t,
            "g": g,
            "d": d,
            "y": y,
            "rel": np.where(g > 0, t - g, np.nan),
        }
    )


def d_single(rng):
    """One adoption date (period 6) for half the units."""
    return d_staggered(rng, cohorts=(6, 0))


def d_block_parallel(rng):
    """The block design with the same loading distribution in both groups."""
    return d_block(rng, shift=0.0)


def d_block(rng, nu=40, nt=20, n_tr=10, t0=15, shift=0.5):
    """One-factor model; treated units load more on the factor."""
    unit = np.repeat(np.arange(nu), nt)
    t = np.tile(np.arange(1, nt + 1), nu)
    treated = (np.arange(nu) < n_tr)[unit]
    load = (rng.normal(size=nu) + shift * (np.arange(nu) < n_tr))[unit]
    f = rng.normal(size=nt)[t - 1]
    a = rng.normal(size=nu)[unit]
    d = (treated & (t >= t0)).astype(int)
    y = a + 0.5 * t / nt + load * f + TAU * d + 0.5 * rng.normal(size=nu * nt)
    return pd.DataFrame(
        {"unit": unit, "t": t, "d": d, "y": y, "treated": treated.astype(int)}
    )


# ------------------------------------------------------------------- fits
def fit_stag(est, **kw):
    def fit(df):
        r = getattr(_sp(), est)(
            df, y="y", group="unit", time="t", first_treat="g", **kw
        )
        return coef(r)

    return fit


def fit_event_study(k):
    def fit(df):
        d = df.copy()
        d["tt"] = np.where(d.g > 0, d.g, np.nan)
        r = _sp().event_study(
            d, y="y", treat_time="tt", time="t", unit="unit", window=(-4, 4)
        )
        row = r.detail[r.detail["relative_time"] == k].iloc[0]
        return row["estimate"], row["se"], row["ci_lower"], row["ci_upper"]

    return fit


def d_staggered_small(rng):
    return d_staggered(rng, nu=120)


def fit_dcdh(df):
    r = _sp().did_multiplegt(
        df, y="y", group="unit", time="t", treatment="d", n_boot=50, seed=1
    )
    return coef(r)


def fit_sdid(se_method, method="sdid"):
    def fit(df):
        r = _sp().sdid(
            df,
            outcome="y",
            unit="unit",
            time="t",
            treat="d",
            method=method,
            se_method=se_method,
            n_reps=200,
            seed=1,
        )
        return coef(r)

    return fit


def fit_gsynth(df):
    r = _sp().gsynth(
        df, outcome="y", unit="unit", time="t", treat="d", n_factors=1, seed=1
    )
    return coef(r)


def R(
    name,
    estimator,
    call,
    dgp,
    fit,
    truth=TAU,
    n=NU * NT,
    B=None,
    design="",
    interval_only=False,
):
    return Row(name, estimator, call, n, truth, dgp, fit, B, design, interval_only)


BLOCK = "40 units, 20 periods, 10 treated from period 15, one factor"

ROWS = [
    R(
        "did_imputation/default",
        "did_imputation",
        "did_imputation(y, group, time, first_treat)",
        d_staggered,
        fit_stag("did_imputation"),
    ),
    R(
        "gardner_did/default",
        "gardner_did",
        "gardner_did(y, group, time, first_treat)",
        d_staggered,
        fit_stag("gardner_did"),
    ),
    R(
        "etwfe/default",
        "etwfe",
        "etwfe(y, group, time, first_treat)",
        d_staggered,
        fit_stag("etwfe"),
    ),
    R(
        "wooldridge_did/default",
        "wooldridge_did",
        "wooldridge_did(y, group, time, first_treat)",
        d_staggered,
        fit_stag("wooldridge_did"),
    ),
    R(
        "stacked_did/default",
        "stacked_did",
        "stacked_did(y, group, time, first_treat)",
        d_staggered,
        fit_stag("stacked_did", window=(-3, 3)),
    ),
    R(
        "event_study/lag1",
        "event_study",
        "event_study(window=(-4, 4)), event time +1",
        d_single,
        fit_event_study(1),
        design="single adoption date, half the units treated",
    ),
    R(
        "event_study/lead2",
        "event_study",
        "event_study(window=(-4, 4)), event time -2",
        d_single,
        fit_event_study(-2),
        truth=0.0,
        design="single adoption date, half the units treated",
    ),
    R(
        "did_multiplegt/default",
        "did_multiplegt",
        "did_multiplegt(n_boot=50)",
        d_staggered_small,
        fit_dcdh,
        n=1200,
        B=200,
        design="120 units; bootstrap SE with 50 replications; B = 200",
    ),
    R(
        "sdid/bootstrap",
        "sdid",
        'sdid(se_method="bootstrap")',
        d_block,
        fit_sdid("bootstrap"),
        n=800,
        B=200,
        design=BLOCK + "; B = 200",
    ),
    R(
        "sdid/jackknife",
        "sdid",
        'sdid(se_method="jackknife")',
        d_block,
        fit_sdid("jackknife"),
        n=800,
        design=BLOCK,
    ),
    R(
        "sdid/placebo-10-treated",
        "sdid",
        'sdid(se_method="placebo")',
        d_block,
        fit_sdid("placebo"),
        n=800,
        B=200,
        design=BLOCK + "; B = 200",
    ),
    R(
        "sdid/did-jackknife",
        "sdid",
        'sdid(method="did", se_method="jackknife")',
        d_block,
        fit_sdid("jackknife", "did"),
        n=800,
        design=BLOCK + " (DiD is biased here by design: loadings differ)",
    ),
    R(
        "sdid/did-jackknife-parallel",
        "sdid",
        'sdid(method="did", se_method="jackknife")',
        d_block_parallel,
        fit_sdid("jackknife", "did"),
        n=800,
        design="block design with equal loading distributions: parallel trends hold",
    ),
    R(
        "gsynth/default",
        "gsynth",
        "gsynth(n_factors=1)",
        d_block,
        fit_gsynth,
        n=800,
        design=BLOCK,
    ),
]


# ---------------------------------------------------------------- options
def d_one_treated(rng, nu=31, nt=35, t0=31):
    """One treated unit drawn from the same factor model as its 30 controls."""
    unit = np.repeat(np.arange(nu), nt)
    t = np.tile(np.arange(1, nt + 1), nu)
    load = rng.uniform(0.5, 1.5, size=nu)[unit]
    f = (0.3 * rng.normal(size=nt).cumsum() + rng.normal(size=nt))[t - 1]
    a = rng.normal(size=nu)[unit]
    d = ((unit == 0) & (t >= t0)).astype(int)
    y = a + load * f + TAU * d + 0.3 * rng.normal(size=nu * nt)
    return pd.DataFrame({"unit": unit, "t": t, "d": d, "y": y})


def fit_synth(**kw):
    def fit(df):
        r = _sp().synth(
            df,
            outcome="y",
            unit="unit",
            time="t",
            treated_unit=0,
            treatment_time=31,
            **kw,
        )
        return coef(r)

    return fit


def cover_synth(**kw):
    def fit(df):
        r = _sp().synth(
            df,
            outcome="y",
            unit="unit",
            time="t",
            treated_unit=0,
            treatment_time=31,
            **kw,
        )
        lo, hi = r.ci
        return lo <= TAU <= hi

    return fit


def fit_event_study_kw(k, **kw):
    def fit(df):
        d = df.copy()
        d["tt"] = np.where(d.g > 0, d.g, np.nan)
        r = _sp().event_study(
            d, y="y", treat_time="tt", time="t", unit="unit", window=(-4, 4), **kw
        )
        row = r.detail[r.detail["relative_time"] == k].iloc[0]
        return row["estimate"], row["se"], row["ci_lower"], row["ci_upper"]

    return fit


ONE = "one treated unit, 30 controls, 30 pre-periods, 5 post-periods, one factor"

OPTION_ROWS = [
    R(
        "did_imputation/bootstrap",
        "did_imputation",
        'did_imputation(vce="bootstrap", n_boot=50)',
        d_staggered_small,
        fit_stag("did_imputation", vce="bootstrap", n_boot=50),
        n=1200,
        B=200,
        design="120 units; cluster bootstrap with 50 replications; B = 200",
    ),
    R(
        "gardner_did/stage2",
        "gardner_did",
        'gardner_did(vce="stage2")',
        d_staggered,
        fit_stag("gardner_did", vce="stage2"),
        design="second-stage clustered variance without the first-stage correction",
    ),
    R(
        "gardner_did/bootstrap",
        "gardner_did",
        'gardner_did(vce="bootstrap", n_boot=50)',
        d_staggered_small,
        fit_stag("gardner_did", vce="bootstrap", n_boot=50),
        n=1200,
        B=200,
        design="120 units; cluster bootstrap with 50 replications; B = 200",
    ),
    R(
        "etwfe/nevertreated",
        "etwfe",
        'etwfe(cgroup="nevertreated")',
        d_staggered,
        fit_stag("etwfe", cgroup="nevertreated"),
    ),
    R(
        "stacked_did/pooled",
        "stacked_did",
        'stacked_did(spec="pooled")',
        d_staggered,
        fit_stag("stacked_did", window=(-3, 3), spec="pooled"),
    ),
    R(
        "stacked_did/treated-obs",
        "stacked_did",
        'stacked_did(aggregation="treated_obs")',
        d_staggered,
        fit_stag("stacked_did", window=(-3, 3), aggregation="treated_obs"),
    ),
    R(
        "event_study/lag1-normal",
        "event_study",
        'event_study(crit_dist="normal"), event time +1',
        d_single,
        fit_event_study_kw(1, crit_dist="normal"),
        design="single adoption date, half the units treated",
    ),
    R(
        "sdid/sc-jackknife",
        "sdid",
        'sdid(method="sc", se_method="jackknife")',
        d_block,
        fit_sdid("jackknife", "sc"),
        n=800,
        design=BLOCK,
    ),
    R(
        "gsynth/nonparametric",
        "gsynth",
        'gsynth(n_factors=1, inference="nonparametric")',
        d_block,
        lambda df: coef(
            _sp().gsynth(
                df,
                outcome="y",
                unit="unit",
                time="t",
                treat="d",
                n_factors=1,
                seed=1,
                inference="nonparametric",
            )
        ),
        n=800,
        design=BLOCK,
    ),
    R(
        "synth/classic-placebo-ci",
        "synth",
        "synth(treated_unit, treatment_time)",
        d_one_treated,
        cover_synth(),
        n=1085,
        interval_only=True,
        design=ONE + "; placebo test-inversion interval, which may be unbounded",
    ),
    R(
        "synth/augmented",
        "synth",
        'synth(method="augmented")',
        d_one_treated,
        fit_synth(method="augmented"),
        n=1085,
        design=ONE,
    ),
    R(
        "synth/conformal",
        "synth",
        'synth(inference="conformal")',
        d_one_treated,
        cover_synth(inference="conformal"),
        n=1085,
        B=200,
        interval_only=True,
        design=ONE + "; B = 200",
    ),
]

ROWS = ROWS + OPTION_ROWS

MECHANISMS: dict = {
    "sdid/did-jackknife": (
        "Design, not estimator. Treated units load 0.5 more on the factor, so parallel "
        "trends fail given the factor path: the DiD estimate moves with the realised "
        "post-minus-pre factor change, which is common to all units and which a jackknife "
        "over units cannot see. The estimate is unbiased over draws and its unit jackknife "
        "SE equals R synthdid's (0.0773 on one draw). With the same loading distribution in "
        "both groups (row sdid/did-jackknife-parallel) the SE is calibrated."
    ),
    "sdid/sc-jackknife": (
        "Property of the fixed-weight jackknife applied to synthetic control, shared with "
        "R synthdid 0.0.9 (same SE to four digits on one draw: 0.5185, against 0.107 from "
        "the placebo method). The jackknife holds the unit weights fixed. SC has no "
        "intercept, so dropping one treated unit changes the treated level that the fixed "
        "weights were chosen to match, and the pseudo-values pick up the spread of the unit "
        "effects, which a refitted SC would absorb. The SE is about four times the sampling "
        "SD and the interval always covers. Use se_method='placebo' or 'bootstrap' with "
        "method='sc'. sp.sdid does not warn about this combination (open item)."
    ),
    "gardner_did/stage2": (
        "Documented option. vce='stage2' is the second-stage clustered variance without "
        "the correction for the estimated first-stage fixed effects; the default "
        "vce='analytic' (row gardner_did/default) carries the correction and is calibrated."
    ),
    "synth/augmented": (
        "Heavy tails, not the SE: one treated unit. The median SE over the MAD-based "
        "spread is 1.05 and coverage is inside the band; a few draws with a poor "
        "pre-period fit inflate the Monte Carlo SD."
    ),
    "sdid/jackknife": (
        "Few treated units. The jackknife variance is driven by the 10 treated units, so it "
        "is itself noisy and the normal critical value is too small for it (a t with 9 "
        "degrees of freedom would give 2.26). The SE equals R synthdid's on one draw "
        "(0.0959). The bootstrap and placebo rows on the same design are inside the band."
    ),
    "gsynth/nonparametric": (
        "At the edge of the band (0.917 on 1,000 draws against a limit of 0.918) with a "
        "calibrated SE (SE/SD 0.96). The reported interval is the percentile interval of "
        "200 bootstrap draws over 40 units; a normal interval from the same SE covers 0.93."
    ),
}

if __name__ == "__main__":
    main(__file__, FAMILY, ROWS, MECHANISMS)
