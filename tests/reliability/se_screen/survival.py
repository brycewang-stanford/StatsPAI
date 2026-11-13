"""SE calibration screen: survival analysis.

n = 1,000, independent censoring at about 30%. Regression rows target the
coefficient on ``x1`` (0.5 on the log-hazard scale, or the stated
accelerated-failure-time value). The non-regression rows target a survival
probability, a cumulative incidence, or a restricted mean, each with a
closed-form truth under exponential hazards.

    python tests/reliability/se_screen/survival.py [--only name] [--confirm]
"""

# flake8: noqa: E501
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _harness import Row, coef, main, tidy_row, wald  # noqa: E402,F401

FAMILY = "survival"
N = 1000
BETA = 0.5
G = 100
SHAPE = 1.5  # Weibull shape in the parametric rows
T0 = 1.0  # evaluation time of the survival / incidence rows
TAU = 2.0  # restriction time of the RMST rows


def _sp():
    import statspai as sp

    return sp


def _xs(rng, n=N):
    return pd.DataFrame({"x1": rng.normal(size=n), "x2": rng.normal(size=n)})


def _censor(df, t, rng, rate=0.35):
    c = rng.exponential(1 / rate, size=len(df))
    df["time"] = np.minimum(t, c)
    df["event"] = (t <= c).astype(int)
    return df


# --------------------------------------------------------------------- DGPs
def d_ph(rng):
    """Exponential baseline, hazard exp(0.5 x1 - 0.5 x2)."""
    df = _xs(rng)
    t = rng.exponential(size=N) / np.exp(BETA * df.x1 - 0.5 * df.x2)
    return _censor(df, t, rng)


def d_ph_strata(rng):
    df = _xs(rng)
    df["s"] = rng.integers(0, 4, size=N)
    base = np.array([0.5, 1.0, 2.0, 4.0])[df.s]
    t = rng.exponential(size=N) / (base * np.exp(BETA * df.x1 - 0.5 * df.x2))
    return _censor(df, t, rng)


def d_ph_cluster(rng):
    """Marginal proportional hazards with dependence inside clusters (Gaussian copula)."""
    from scipy.stats import norm

    g = np.repeat(np.arange(G), N // G)
    df = pd.DataFrame({"g": g})
    df["x1"] = 0.7 * rng.normal(size=G)[g] + 0.7 * rng.normal(size=N)
    df["x2"] = rng.normal(size=N)
    u = norm.cdf(
        np.sqrt(0.4) * rng.normal(size=G)[g] + np.sqrt(0.6) * rng.normal(size=N)
    )
    t = -np.log(u) / np.exp(BETA * df.x1 - 0.5 * df.x2)
    return _censor(df, t, rng)


def d_weibull(rng):
    """Weibull with shape 1.5: PH coefficient 0.5, AFT coefficient -0.5 / 1.5."""
    df = _xs(rng)
    t = (rng.exponential(size=N) / np.exp(BETA * df.x1 - 0.5 * df.x2)) ** (1 / SHAPE)
    return _censor(df, t, rng)


def d_lognormal(rng):
    df = _xs(rng)
    t = np.exp(BETA * df.x1 - 0.5 * df.x2 + 0.8 * rng.normal(size=N))
    return _censor(df, t, rng, rate=0.25)


def d_loglogistic(rng):
    df = _xs(rng)
    t = np.exp(BETA * df.x1 - 0.5 * df.x2 + 0.5 * rng.logistic(size=N))
    return _censor(df, t, rng, rate=0.25)


def d_frailty(rng):
    """Gamma shared frailty (variance 0.5); 0.5 is the conditional log hazard ratio."""
    g = np.repeat(np.arange(G), N // G)
    df = pd.DataFrame({"g": g, "x1": rng.normal(size=N), "x2": rng.normal(size=N)})
    w = rng.gamma(shape=2.0, scale=0.5, size=G)[g]
    t = rng.exponential(size=N) / (w * np.exp(BETA * df.x1 - 0.5 * df.x2))
    return _censor(df, t, rng)


def d_finegray(rng, p=0.5):
    """Fine and Gray (1999) design: subdistribution hazard of cause 1 is PH in x1."""
    df = _xs(rng)
    e = np.exp(BETA * df.x1 - 0.5 * df.x2).to_numpy()
    p1 = 1 - (1 - p) ** e
    cause1 = rng.uniform(size=N) < p1
    u = rng.uniform(size=N)
    t1 = -np.log(1 - (1 - (1 - u * p1) ** (1 / e)) / p)
    t2 = rng.exponential(size=N) / np.exp(-0.3 * df.x1.to_numpy())
    t = np.where(cause1, t1, t2)
    c = rng.uniform(0, 6, size=N)
    df["time"] = np.minimum(t, c)
    df["event"] = np.where(t <= c, np.where(cause1, 1, 2), 0)
    return df


def d_two_group(rng):
    """Exponential survival, rate 1.0 in group 0 and 0.6 in group 1."""
    df = pd.DataFrame({"grp": rng.integers(0, 2, size=N)})
    t = rng.exponential(size=N) / np.where(df.grp == 1, 0.6, 1.0)
    return _censor(df, t, rng)


def d_one_group(rng):
    df = pd.DataFrame({"one": np.ones(N)})
    return _censor(df, rng.exponential(size=N), rng)


def d_competing(rng):
    """Two independent exponential causes, rates 0.6 and 0.4."""
    df = pd.DataFrame({"one": np.ones(N)})
    t1 = rng.exponential(1 / 0.6, size=N)
    t2 = rng.exponential(1 / 0.4, size=N)
    t = np.minimum(t1, t2)
    c = rng.exponential(1 / 0.35, size=N)
    df["time"] = np.minimum(t, c)
    df["event"] = np.where(t <= c, np.where(t1 <= t2, 1, 2), 0)
    return df


def d_stte(rng, rate=0.04):
    """Exponential outcome, mean exp(0.2 + 0.5 x1 + 0.4 d); treatment logit in x1.

    Censoring is exponential at ``rate``. The censoring-weighted estimators
    need E[T^2 / G(T)] finite, i.e. ``2 * rate`` below the failure hazard
    ``exp(-(0.2 + 0.5 x1 + 0.4 d))``. At rate 0.04 that fails only for
    x1 > 3.8; at rate 0.15 it fails for x1 > 1.2, one unit in nine.
    """
    df = _xs(rng, 2000)
    df["d"] = (rng.uniform(size=2000) < 1 / (1 + np.exp(-0.5 * df.x1))).astype(int)
    t = np.exp(0.2 + 0.5 * df.x1 + 0.4 * df.d) * rng.exponential(size=2000)
    c = rng.exponential(1 / rate, size=2000)
    df["time"] = np.minimum(t, c)
    df["event"] = (t <= c).astype(int)
    return df


def d_stte_heavy(rng):
    return d_stte(rng, rate=0.15)


STTE_TRUTH = float(np.exp(0.2 + 0.125) * (np.exp(0.4) - 1))
RMST_TRUTH = float((1 - np.exp(-0.6 * TAU)) / 0.6 - (1 - np.exp(-TAU)))
CIF_TRUTH = float(0.6 * (1 - np.exp(-T0)))
KM_TRUTH = float(np.exp(-T0))


# ------------------------------------------------------------------- fits
def _table_row(row):
    """(estimate, se, lower, upper) from one row of a result table."""

    def get(*names):
        for n in names:
            if n in row.index:
                return float(row[n])
        raise KeyError(f"none of {names} in {list(row.index)}")

    return (
        get("estimate", "survival", "cif", "rmst"),
        get("se", "std_err", "std_error"),
        get("ci_lower", "conf_low", "lower"),
        get("ci_upper", "conf_high", "upper"),
    )


def _no_interval(r, term="x1"):
    """AFTResult and FrailtyResult report an SE and no interval: normal interval."""
    t = r.tidy().set_index("term").loc[term]
    return wald(t["estimate"], t["std_error"])


def fit_cox(**kw):
    def fit(df):
        r = _sp().cox(
            data=df,
            duration="time",
            event="event",
            x=["x1", "x2"],
            hazard_ratio=False,
            **kw,
        )
        return coef(r, "x1")

    return fit


def fit_survreg(dist, **kw):
    def fit(df):
        r = _sp().survreg(
            data=df, duration="time", event="event", x=["x1", "x2"], dist=dist, **kw
        )
        return coef(r, "x1")

    return fit


def fit_aft(family):
    def fit(df):
        r = _sp().aft("time + event ~ x1 + x2", df, family=family)
        return _no_interval(r)

    return fit


def fit_frailty(df):
    r = _sp().cox_frailty("time + event ~ x1 + x2", df, cluster="g")
    return _no_interval(r)


def fit_finegray(**kw):
    def fit(df):
        r = _sp().finegray(
            df, duration="time", event="event", x=["x1", "x2"], cause=1, **kw
        )
        t = r.tidy().set_index("term").loc["x1"]
        return t["coef"], t["std_err"], np.log(t["shr_lower"]), np.log(t["shr_upper"])

    return fit


def fit_rmst(**kw):
    def fit(df):
        r = _sp().rmst(df, duration="time", event="event", group="grp", tau=TAU, **kw)
        return _table_row(r.contrasts.iloc[0])

    return fit


def fit_km(**kw):
    def fit(df):
        r = _sp().kaplan_meier(df, duration="time", event="event", **kw)
        t = r.survival_table
        return _table_row(t[t["time"] <= T0].iloc[-1])

    return fit


def fit_cuminc(**kw):
    def fit(df):
        r = _sp().cuminc(df, duration="time", event="event", **kw)
        t = r.cif_table
        t = t[(t["cause"] == 1) & (t["time"] <= T0)]
        return _table_row(t.iloc[-1])

    return fit


def fit_stte(method):
    def fit(df):
        kw = {"censor_covariates": []} if method != "ra" else {}
        r = _sp().stteffects(
            df,
            duration="time",
            event="event",
            treat="d",
            covariates=["x1"],
            method=method,
            dist="exponential",
            **kw,
        )
        return coef(r)

    return fit


def R(name, estimator, call, dgp, fit, truth=BETA, n=N, B=None, design=""):
    return Row(name, estimator, call, n, truth, dgp, fit, B, design)


AFT_W = -BETA / SHAPE

ROWS = [
    R("cox/default", "cox", 'cox("time + event ~ x1 + x2")', d_ph, fit_cox()),
    R("cox/breslow", "cox", 'cox(..., ties="breslow")', d_ph, fit_cox(ties="breslow")),
    R("cox/robust", "cox", 'cox(..., robust="robust")', d_ph, fit_cox(robust="robust")),
    R(
        "cox/cluster",
        "cox",
        'cox(..., cluster="g")',
        d_ph_cluster,
        fit_cox(cluster="g"),
        design="100 clusters of 10, copula dependence 0.4, marginal PH exact",
    ),
    R(
        "cox/strata",
        "cox",
        'cox(..., strata="s")',
        d_ph_strata,
        fit_cox(strata="s"),
        design="four strata with baseline hazards 0.5 to 4",
    ),
    R(
        "survreg/weibull",
        "survreg",
        'survreg(dist="weibull")',
        d_weibull,
        fit_survreg("weibull"),
        truth=AFT_W,
        design="Weibull shape 1.5; AFT coefficient -1/3",
    ),
    R(
        "survreg/exponential",
        "survreg",
        'survreg(dist="exponential")',
        d_ph,
        fit_survreg("exponential"),
        truth=-BETA,
        design="AFT coefficient -0.5",
    ),
    R(
        "survreg/weibull-robust",
        "survreg",
        'survreg(dist="weibull", robust="robust")',
        d_weibull,
        fit_survreg("weibull", robust="robust"),
        truth=AFT_W,
        design="Weibull shape 1.5; AFT coefficient -1/3",
    ),
    R(
        "survreg/lognormal",
        "survreg",
        'survreg(dist="lognormal")',
        d_lognormal,
        fit_survreg("lognormal"),
    ),
    R(
        "survreg/loglogistic",
        "survreg",
        'survreg(dist="loglogistic")',
        d_loglogistic,
        fit_survreg("loglogistic"),
    ),
    R(
        "aft/weibull",
        "aft",
        'aft(family="weibull")',
        d_weibull,
        fit_aft("weibull"),
        truth=AFT_W,
        design="Weibull shape 1.5; AFT coefficient -1/3",
    ),
    R(
        "aft/lognormal",
        "aft",
        'aft(family="lognormal")',
        d_lognormal,
        fit_aft("lognormal"),
    ),
    R(
        "aft/loglogistic",
        "aft",
        'aft(family="loglogistic")',
        d_loglogistic,
        fit_aft("loglogistic"),
    ),
    R(
        "cox_frailty/default",
        "cox_frailty",
        'cox_frailty(..., cluster="g")',
        d_frailty,
        fit_frailty,
        design="100 clusters of 10, gamma frailty variance 0.5",
    ),
    R(
        "finegray/default",
        "finegray",
        "finegray(duration, event, x, cause=1)",
        d_finegray,
        fit_finegray(),
        design="Fine-Gray design, uniform censoring",
    ),
    R(
        "finegray/model",
        "finegray",
        'finegray(..., vce="model")',
        d_finegray,
        fit_finegray(vce="model"),
        design="Fine-Gray design, uniform censoring",
    ),
    R(
        "rmst/difference",
        "rmst",
        "rmst(group=, tau=2)",
        d_two_group,
        fit_rmst(),
        truth=RMST_TRUTH,
        design="difference in restricted mean, exponential rates 1.0 and 0.6",
    ),
    R(
        "kaplan_meier/loglog",
        "kaplan_meier",
        "kaplan_meier(duration, event)",
        d_one_group,
        fit_km(),
        truth=KM_TRUTH,
        design="S(1) of a unit exponential",
    ),
    R(
        "kaplan_meier/plain",
        "kaplan_meier",
        'kaplan_meier(conf_type="plain")',
        d_one_group,
        fit_km(conf_type="plain"),
        truth=KM_TRUTH,
        design="S(1) of a unit exponential",
    ),
    R(
        "cuminc/default",
        "cuminc",
        "cuminc(duration, event)",
        d_competing,
        fit_cuminc(),
        truth=CIF_TRUTH,
        design="cumulative incidence of cause 1 at t = 1",
    ),
    R(
        "stteffects/ra",
        "stteffects",
        'stteffects(method="ra")',
        d_stte,
        fit_stte("ra"),
        truth=STTE_TRUTH,
        n=2000,
        design="ATE on mean survival time, censoring rate 0.04",
    ),
    R(
        "stteffects/ipw",
        "stteffects",
        'stteffects(method="ipw")',
        d_stte,
        fit_stte("ipw"),
        truth=STTE_TRUTH,
        n=2000,
        design="ATE on mean survival time, censoring rate 0.04",
    ),
    R(
        "stteffects/ipwra",
        "stteffects",
        'stteffects(method="ipwra")',
        d_stte,
        fit_stte("ipwra"),
        truth=STTE_TRUTH,
        n=2000,
        design="ATE on mean survival time, censoring rate 0.04",
    ),
    R(
        "stteffects/wra",
        "stteffects",
        'stteffects(method="wra")',
        d_stte,
        fit_stte("wra"),
        truth=STTE_TRUTH,
        n=2000,
        design="ATE on mean survival time, censoring rate 0.04",
    ),
    R(
        "stteffects/ipw-heavy-censoring",
        "stteffects",
        'stteffects(method="ipw")',
        d_stte_heavy,
        fit_stte("ipw"),
        truth=STTE_TRUTH,
        n=2000,
        design="censoring rate 0.15: long failure times are almost always censored",
    ),
]


# ---------------------------------------------------------------- options
def d_ph_w(rng):
    df = d_ph(rng)
    df["w"] = rng.uniform(0.5, 3.0, size=N)
    return df


def d_ph_ties(rng):
    """Proportional hazards observed on a grid of width 0.1 (many tied times)."""
    df = d_ph(rng)
    df["time"] = np.ceil(df.time / 0.1) * 0.1
    return df


def d_ph_entry(rng):
    """Delayed entry: subjects come under observation at a uniform time."""
    df = _xs(rng, 1500)
    t = rng.exponential(size=1500) / np.exp(BETA * df.x1 - 0.5 * df.x2)
    df = _censor(df, t, rng)
    df["entry"] = rng.uniform(0, 0.4, size=1500)
    return df[df.time > df.entry].head(N).reset_index(drop=True)


def d_weibull_cluster(rng):
    from scipy.stats import norm

    g = np.repeat(np.arange(G), N // G)
    df = pd.DataFrame({"g": g})
    df["x1"] = 0.7 * rng.normal(size=G)[g] + 0.7 * rng.normal(size=N)
    df["x2"] = rng.normal(size=N)
    u = norm.cdf(
        np.sqrt(0.4) * rng.normal(size=G)[g] + np.sqrt(0.6) * rng.normal(size=N)
    )
    t = (-np.log(u) / np.exp(BETA * df.x1 - 0.5 * df.x2)) ** (1 / SHAPE)
    return _censor(df, t, rng)


def _atet_truth():
    """E[exp(0.2 + 0.5 x)(exp(0.4) - 1) | d = 1] with P(d = 1 | x) = expit(0.5 x)."""
    x, w = np.polynomial.hermite_e.hermegauss(80)
    w = w / w.sum()
    p = 1 / (1 + np.exp(-0.5 * x))
    return float(
        (w * p * np.exp(0.2 + 0.5 * x)).sum() / (w * p).sum() * (np.exp(0.4) - 1)
    )


def fit_cox_start(df):
    r = _sp().cox(
        data=df,
        duration="time",
        event="event",
        x=["x1", "x2"],
        start="entry",
        hazard_ratio=False,
    )
    return coef(r, "x1")


def fit_stte_atet(df):
    r = _sp().stteffects(
        df,
        duration="time",
        event="event",
        treat="d",
        covariates=["x1"],
        method="ra",
        dist="exponential",
        estimand="ATET",
    )
    return coef(r)


TIES = "event times recorded on a grid of width 0.1"

OPTION_ROWS = [
    R(
        "cox/pweights",
        "cox",
        'cox(..., weights="w", weights_as="sampling")',
        d_ph_w,
        fit_cox(weights="w", weights_as="sampling"),
        design="weights independent of the data",
    ),
    R(
        "cox/robust-stata",
        "cox",
        'cox(..., robust="robust", ssc="stata")',
        d_ph,
        fit_cox(robust="robust", ssc="stata"),
    ),
    R(
        "cox/efron-ties",
        "cox",
        'cox(..., ties="efron")',
        d_ph_ties,
        fit_cox(ties="efron"),
        design=TIES,
    ),
    R(
        "cox/breslow-ties",
        "cox",
        'cox(..., ties="breslow")',
        d_ph_ties,
        fit_cox(ties="breslow"),
        design=TIES,
    ),
    R(
        "cox/delayed-entry",
        "cox",
        'cox(..., start="entry")',
        d_ph_entry,
        fit_cox_start,
        design="uniform entry times on [0, 0.4]",
    ),
    R(
        "survreg/cluster",
        "survreg",
        'survreg(dist="weibull", cluster="g")',
        d_weibull_cluster,
        fit_survreg("weibull", cluster="g"),
        truth=AFT_W,
        design="100 clusters of 10, copula dependence 0.4, Weibull shape 1.5",
    ),
    R(
        "survreg/lognormal-robust",
        "survreg",
        'survreg(dist="lognormal", robust="robust")',
        d_lognormal,
        fit_survreg("lognormal", robust="robust"),
    ),
    R(
        "aft/exponential",
        "aft",
        'aft(family="exponential")',
        d_ph,
        fit_aft("exponential"),
        truth=-BETA,
        design="AFT coefficient -0.5",
    ),
    R(
        "kaplan_meier/log",
        "kaplan_meier",
        'kaplan_meier(conf_type="log")',
        d_one_group,
        fit_km(conf_type="log"),
        truth=KM_TRUTH,
        design="S(1) of a unit exponential",
    ),
    R(
        "kaplan_meier/ij",
        "kaplan_meier",
        'kaplan_meier(se_method="ij")',
        d_one_group,
        fit_km(se_method="ij"),
        truth=KM_TRUTH,
        design="S(1) of a unit exponential",
    ),
    R(
        "rmst/ij",
        "rmst",
        'rmst(group=, tau=2, se_method="ij")',
        d_two_group,
        fit_rmst(se_method="ij"),
        truth=RMST_TRUTH,
        design="difference in restricted mean, exponential rates 1.0 and 0.6",
    ),
    R(
        "cuminc/gray",
        "cuminc",
        'cuminc(variance="gray")',
        d_competing,
        fit_cuminc(variance="gray"),
        truth=CIF_TRUTH,
        design="cumulative incidence of cause 1 at t = 1",
    ),
    R(
        "cuminc/log-log",
        "cuminc",
        'cuminc(conf_type="log-log")',
        d_competing,
        fit_cuminc(conf_type="log-log"),
        truth=CIF_TRUTH,
        design="cumulative incidence of cause 1 at t = 1",
    ),
    R(
        "finegray/small-sample",
        "finegray",
        "finegray(..., small_sample=True)",
        d_finegray,
        fit_finegray(small_sample=True),
        design="Fine-Gray design, uniform censoring",
    ),
    R(
        "stteffects/ra-atet",
        "stteffects",
        'stteffects(method="ra", estimand="ATET")',
        d_stte,
        fit_stte_atet,
        truth=_atet_truth(),
        n=2000,
        design="ATET on mean survival time, censoring rate 0.04",
    ),
]

ROWS = ROWS + OPTION_ROWS

MECHANISMS: dict = {
    "stteffects/ipw-heavy-censoring": (
        "Design breaks the overlap that censoring weights need. With censoring rate 0.15 "
        "the second moment of T / G(T) does not exist for units with x1 > 1.2: long "
        "failure times are almost always censored and the rare uncensored one carries a "
        "huge weight. The estimator is heavy-tailed, so SE/SD does not settle (0.66, 0.27 "
        "and 0.38 in three runs) while coverage stays at 92% to 96%. At rate 0.04 (row "
        "stteffects/ipw) SE/SD is 0.98. sp.stteffects now records the censoring weights in "
        "model_info and its notes describe the condition."
    ),
    "cox/breslow-ties": (
        "Known property of the Breslow approximation with heavy ties, not the SE (SE/SD "
        "1.07). Event times are recorded on a grid of width 0.1, so about a tenth of the "
        "risk set fails at each tied time, and Breslow's partial likelihood attenuates the "
        "coefficient (bias -0.9 SD). The Efron default on the same data (row "
        "cox/efron-ties) is inside the band."
    ),
}

if __name__ == "__main__":
    main(__file__, FAMILY, ROWS, MECHANISMS)
