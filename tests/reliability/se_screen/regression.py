"""SE calibration screen: regression family (likelihood and moment models).

Every row is a correctly specified model at a sample size where first-order
asymptotics should hold (n = 1,000 unless the row says otherwise). The
target is the slope on ``x1`` (true value 0.5) unless stated. Designs were
fixed before the first run; see ``_harness.py`` for the statistics and the
flagging rule.

    python tests/reliability/se_screen/regression.py            # all rows
    python tests/reliability/se_screen/regression.py --only tobit
    python tests/reliability/se_screen/regression.py --confirm  # flagged, B = 1000
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _harness import Row, coef, main, tidy_row, wald  # noqa: E402

FAMILY = "regression"
N = 1000
BETA = 0.5  # slope on x1 in every design unless the row says otherwise
G = 100  # clusters in the clustered designs (10 observations each)


def _sp():
    import statspai as sp

    return sp


def _expit(v):
    return 1.0 / (1.0 + np.exp(-v))


def _xs(rng, n=N):
    return pd.DataFrame({"x1": rng.normal(size=n), "x2": rng.normal(size=n)})


# --------------------------------------------------------------------- DGPs
def d_linear(rng):
    df = _xs(rng)
    df["y"] = 1 + BETA * df.x1 - 0.5 * df.x2 + rng.normal(size=N)
    return df


def d_linear_het(rng):
    df = _xs(rng)
    df["y"] = 1 + BETA * df.x1 - 0.5 * df.x2 + rng.normal(size=N) * np.exp(0.5 * df.x1)
    return df


def d_linear_cluster(rng):
    g = np.repeat(np.arange(G), N // G)
    df = pd.DataFrame({"g": g})
    df["x1"] = rng.normal(size=G)[g] * 0.7 + rng.normal(size=N) * 0.7
    df["x2"] = rng.normal(size=N)
    df["y"] = (
        1
        + BETA * df.x1
        - 0.5 * df.x2
        + rng.normal(size=G)[g] * 0.7
        + rng.normal(size=N)
    )
    return df


def d_logit(rng):
    df = _xs(rng)
    df["y"] = (rng.uniform(size=N) < _expit(-0.3 + BETA * df.x1 - 0.5 * df.x2)).astype(
        int
    )
    return df


def d_logit_cluster(rng):
    """Marginal logit holds exactly; a cluster-level uniform shifts every member."""
    from scipy.stats import norm

    g = np.repeat(np.arange(G), N // G)
    df = pd.DataFrame({"g": g})
    df["x1"] = rng.normal(size=G)[g] * 0.7 + rng.normal(size=N) * 0.7
    df["x2"] = rng.normal(size=N)
    u = norm.cdf(
        np.sqrt(0.3) * rng.normal(size=G)[g] + np.sqrt(0.7) * rng.normal(size=N)
    )
    df["y"] = (u < _expit(-0.3 + BETA * df.x1 - 0.5 * df.x2)).astype(int)
    df["t"] = np.tile(np.arange(N // G), G)
    return df


def d_probit(rng):
    df = _xs(rng)
    df["y"] = (-0.3 + BETA * df.x1 - 0.5 * df.x2 + rng.normal(size=N) > 0).astype(int)
    return df


def d_cloglog(rng):
    df = _xs(rng)
    p = 1 - np.exp(-np.exp(-0.5 + BETA * df.x1 - 0.5 * df.x2))
    df["y"] = (rng.uniform(size=N) < p).astype(int)
    return df


def d_poisson(rng):
    df = _xs(rng)
    df["y"] = rng.poisson(np.exp(0.5 + BETA * df.x1 - 0.3 * df.x2))
    return df


def d_negbin(rng):
    """Gamma-mixed Poisson, mean exp(xb), variance mu + 0.5 mu^2."""
    df = _xs(rng)
    mu = np.exp(0.5 + BETA * df.x1 - 0.3 * df.x2)
    df["y"] = rng.poisson(mu * rng.gamma(shape=2.0, scale=0.5, size=N))
    return df


def d_gamma(rng):
    df = _xs(rng)
    mu = np.exp(0.5 + BETA * df.x1 - 0.3 * df.x2)
    df["y"] = rng.gamma(shape=2.0, scale=mu / 2.0)
    return df


def d_tobit(rng):
    df = _xs(rng)
    df["y"] = np.maximum(0.0, 0.5 + BETA * df.x1 - 0.5 * df.x2 + rng.normal(size=N))
    return df


def d_truncreg(rng):
    df = _xs(rng, 1400)
    df["y"] = 1 + BETA * df.x1 - 0.5 * df.x2 + rng.normal(size=1400)
    return df[df.y > 0].head(N).reset_index(drop=True)


def d_intreg(rng):
    df = _xs(rng)
    ystar = 1 + BETA * df.x1 - 0.5 * df.x2 + rng.normal(size=N)
    df["lo"] = np.floor(ystar)
    df["hi"] = df.lo + 1.0
    return df


def d_heckman(rng):
    df = _xs(rng)
    df["z"] = rng.normal(size=N)
    u = rng.normal(size=N)
    e = 0.5 * u + np.sqrt(0.75) * rng.normal(size=N)
    df["s"] = (0.5 + df.z + 0.5 * df.x1 + u > 0).astype(int)
    df["y"] = np.where(df.s == 1, 1 + BETA * df.x1 - 0.5 * df.x2 + e, np.nan)
    return df


def d_heckprobit(rng):
    df = _xs(rng, 2000)
    df["z"] = rng.normal(size=2000)
    u = rng.normal(size=2000)
    e = 0.5 * u + np.sqrt(0.75) * rng.normal(size=2000)
    df["s"] = (0.5 + df.z + 0.5 * df.x1 + u > 0).astype(int)
    df["y"] = np.where(
        df.s == 1, (BETA * df.x1 - 0.5 * df.x2 + e > 0).astype(float), np.nan
    )
    return df


def d_ordered_logit(rng):
    df = _xs(rng)
    lat = BETA * df.x1 - 0.5 * df.x2 + rng.logistic(size=N)
    df["y"] = np.digitize(lat, [-1.0, 0.0, 1.0])
    return df


def d_ordered_probit(rng):
    df = _xs(rng)
    lat = BETA * df.x1 - 0.5 * df.x2 + rng.normal(size=N)
    df["y"] = np.digitize(lat, [-1.0, 0.0, 1.0])
    return df


def d_mlogit(rng):
    df = _xs(rng)
    u = np.column_stack(
        [
            rng.gumbel(size=N),
            0.2 + BETA * df.x1 - 0.5 * df.x2 + rng.gumbel(size=N),
            -0.2 - 0.5 * df.x1 + 0.3 * df.x2 + rng.gumbel(size=N),
        ]
    )
    df["y"] = u.argmax(axis=1)
    return df


def d_clogit(rng):
    """250 choice sets of four alternatives, one chosen."""
    ng, j = 250, 4
    g = np.repeat(np.arange(ng), j)
    df = pd.DataFrame(
        {"g": g, "x1": rng.normal(size=ng * j), "x2": rng.normal(size=ng * j)}
    )
    u = (BETA * df.x1 - 0.5 * df.x2 + rng.gumbel(size=ng * j)).to_numpy().reshape(ng, j)
    y = np.zeros((ng, j), dtype=int)
    y[np.arange(ng), u.argmax(axis=1)] = 1
    df["y"] = y.ravel()
    return df


def d_qreg_het(rng):
    """Location-scale with a symmetric error: the median slope is 0.5."""
    df = pd.DataFrame({"x1": rng.uniform(0, 2, size=N), "x2": rng.normal(size=N)})
    df["y"] = 1 + BETA * df.x1 - 0.5 * df.x2 + (0.5 + df.x1) * rng.normal(size=N)
    return df


def d_zip(rng):
    df = _xs(rng)
    zero = rng.uniform(size=N) < _expit(-1.0 + 0.5 * df.x2)
    df["y"] = np.where(zero, 0, rng.poisson(np.exp(0.5 + BETA * df.x1)))
    return df


def d_zinb(rng):
    df = _xs(rng, 2000)
    zero = rng.uniform(size=2000) < _expit(-1.0 + 0.5 * df.x2)
    mu = np.exp(1.0 + BETA * df.x1)
    df["y"] = np.where(zero, 0, rng.poisson(mu * rng.gamma(2.0, 0.5, size=2000)))
    return df


def d_hurdle(rng):
    """Logit for y > 0, zero-truncated Poisson for the positive part."""
    df = _xs(rng)
    pos = rng.uniform(size=N) < _expit(0.5 + 0.5 * df.x2)
    mu = np.exp(0.5 + BETA * df.x1).to_numpy()
    y = np.zeros(N, dtype=int)
    for i in np.where(pos)[0]:
        v = 0
        while v == 0:
            v = rng.poisson(mu[i])
        y[i] = v
    df["y"] = y
    return df


def d_beta(rng):
    df = _xs(rng)
    mu = _expit(BETA * df.x1 - 0.5 * df.x2)
    df["y"] = rng.beta(mu * 5.0, (1 - mu) * 5.0)
    return df


def d_biprobit(rng):
    df = _xs(rng)
    e1 = rng.normal(size=N)
    e2 = 0.5 * e1 + np.sqrt(0.75) * rng.normal(size=N)
    df["y1"] = (BETA * df.x1 + e1 > 0).astype(int)
    df["y2"] = (-0.5 * df.x2 + e2 > 0).astype(int)
    return df


def d_etregress(rng):
    df = _xs(rng)
    df["z"] = rng.normal(size=N)
    u = rng.normal(size=N)
    e = 0.5 * u + np.sqrt(0.75) * rng.normal(size=N)
    df["d"] = (0.8 * df.z + 0.3 * df.x1 + u > 0).astype(int)
    df["y"] = 1 + 0.3 * df.x1 + BETA * df.d + e
    return df


def d_etpoisson(rng):
    df = _xs(rng, 2000)
    df["z"] = rng.normal(size=2000)
    u = rng.normal(size=2000)
    e = 0.5 * (0.5 * u + np.sqrt(0.75) * rng.normal(size=2000))
    df["d"] = (0.8 * df.z + 0.3 * df.x1 + u > 0).astype(int)
    df["y"] = rng.poisson(np.exp(0.2 + 0.3 * df.x1 + BETA * df.d + e))
    return df


def _endog(rng, n=N):
    df = _xs(rng, n)
    df["z"] = rng.normal(size=n)
    v = rng.normal(size=n)
    df["w"] = 0.7 * df.z + 0.3 * df.x1 + v
    return df, v


def d_ivprobit(rng):
    df, v = _endog(rng)
    e = 0.5 * v + np.sqrt(0.75) * rng.normal(size=N)
    df["y"] = (0.2 + BETA * df.w + 0.3 * df.x1 + e > 0).astype(int)
    return df


def d_ivtobit(rng):
    df, v = _endog(rng)
    e = 0.5 * v + np.sqrt(0.75) * rng.normal(size=N)
    df["y"] = np.maximum(0.0, 0.5 + BETA * df.w + 0.3 * df.x1 + e)
    return df


def d_ivpoisson_add(rng):
    """E[y - exp(xb) | z, x1] = 0 with an error that moves with the first stage.

    y is Poisson with mean exp(xb) m(v), m = 1 + 0.6 tanh(v - 0.5). Because
    exp(0.5 v) phi(v) is proportional to phi(v - 0.5), E[exp(0.5 v)(m - 1)] = 0.
    """
    df, v = _endog(rng, 2000)
    m = 1 + 0.6 * np.tanh(v - 0.5)
    df["y"] = rng.poisson(np.exp(0.5 + BETA * df.w + 0.3 * df.x1) * m)
    return df


def d_ivpoisson_mult(rng):
    """E[y / exp(xb) - 1 | z, x1] = 0; the endogenous error is multiplicative."""
    df, v = _endog(rng, 2000)
    eps = np.exp(0.4 * v - 0.08)
    df["y"] = rng.poisson(np.exp(0.5 + BETA * df.w + 0.3 * df.x1) * eps)
    return df


def d_gee_gauss(rng):
    df = d_linear_cluster(rng)
    df["t"] = np.tile(np.arange(N // G), G)
    return df


def d_prais(rng):
    t = 500
    e = np.zeros(t)
    x = np.zeros(t)
    ie, ix = rng.normal(size=t), rng.normal(size=t)
    e[0] = ie[0] / np.sqrt(1 - 0.25)
    x[0] = ix[0] / np.sqrt(1 - 0.25)
    for i in range(1, t):
        e[i] = 0.5 * e[i - 1] + ie[i]
        x[i] = 0.5 * x[i - 1] + ix[i]
    return pd.DataFrame({"t": np.arange(t), "x1": x, "y": 1 + BETA * x + e})


def d_cns(rng):
    df = _xs(rng)
    df["y"] = 1 + BETA * df.x1 + BETA * df.x2 + rng.normal(size=N)
    return df


def d_nls(rng):
    df = pd.DataFrame({"x1": rng.uniform(0, 2, size=N)})
    df["y"] = 2.0 * np.exp(BETA * df.x1) + rng.normal(size=N)
    return df


def d_eiv(rng):
    """x1 is observed with error; reliability 0.8."""
    df = _xs(rng)
    df["y"] = 1 + BETA * df.x1 - 0.5 * df.x2 + rng.normal(size=N)
    df["x1"] = df.x1 + rng.normal(size=N) * 0.5
    return df


def d_sur(rng):
    df = _xs(rng)
    df["x3"] = rng.normal(size=N)
    e1 = rng.normal(size=N)
    e2 = 0.6 * e1 + 0.8 * rng.normal(size=N)
    df["y1"] = 1 + BETA * df.x1 + e1
    df["y2"] = -1 + 0.5 * df.x2 - 0.5 * df.x3 + e2
    return df


def d_3sls(rng):
    """y1 depends on endogenous y2; z1, z2 are excluded from the y1 equation."""
    df = _xs(rng)
    df["z1"] = rng.normal(size=N)
    df["z2"] = rng.normal(size=N)
    e1 = rng.normal(size=N)
    e2 = 0.6 * e1 + 0.8 * rng.normal(size=N)
    df["y2"] = 0.5 * df.x2 + 0.7 * df.z1 + 0.7 * df.z2 + e2
    df["y1"] = 1 + BETA * df.y2 + 0.3 * df.x1 + e1
    return df


def d_ppml_fe(rng):
    g = np.repeat(np.arange(G), N // G)
    df = pd.DataFrame({"g": g})
    a = rng.normal(size=G)[g] * 0.5
    df["x1"] = 0.5 * a + rng.normal(size=N)
    df["y"] = rng.poisson(np.exp(0.5 + a + BETA * df.x1))
    return df


def d_gam(rng):
    df = _xs(rng)
    df["y"] = 1 + BETA * df.x1 + np.sin(2 * df.x2) + rng.normal(size=N)
    return df


def d_fmm(rng):
    df = _xs(rng)
    cls = rng.uniform(size=N) < 0.4
    df["y"] = (
        np.where(cls, -3 + BETA * df.x1, 3 - 1.0 * df.x1) + rng.normal(size=N) * 0.7
    )
    return df


# ------------------------------------------------------------------- fits
def f(est, term="x1", **kw):
    """Formula estimator ``sp.<est>("y ~ x1 + x2", data, **kw)``."""

    def fit(df):
        return coef(getattr(_sp(), est)("y ~ x1 + x2", data=df, **kw), term)

    return fit


def fx(est, term="x1", x=("x1", "x2"), **kw):
    """Column-list estimator ``sp.<est>(data, y="y", x=[...], **kw)``."""

    def fit(df):
        return coef(getattr(_sp(), est)(data=df, y="y", x=list(x), **kw), term)

    return fit


def fit_intreg(df):
    return coef(_sp().intreg(df, y_lower="lo", y_upper="hi", x=["x1", "x2"]), "x1")


def fit_heckman(method):
    def fit(df):
        r = _sp().heckman(
            df, y="y", x=["x1", "x2"], select="s", z=["z", "x1"], method=method
        )
        return coef(r, "x1")

    return fit


def fit_heckprobit(df):
    return coef(
        _sp().heckprobit(df, y="y", x=["x1", "x2"], select="s", z=["z", "x1"]), "x1"
    )


def fit_gologit(df):
    r = _sp().gologit(df, y="y", x=["x1", "x2"], pl=True)
    return coef(r, "0:x1")


def fit_mlogit(df):
    r = _sp().mlogit("y ~ x1 + x2", data=df)
    return coef(r, [t for t in r.params.index if "x1" in str(t)][0])


def fit_clogit(df):
    return coef(_sp().clogit("y ~ x1 + x2", data=df, group="g"), "x1")


def fit_sqreg(df):
    r = _sp().sqreg(
        df, y="y", x=["x1", "x2"], quantiles=[0.25, 0.5, 0.75], reps=100, seed=1
    )
    return _sq(r)


def _sq(r):
    if isinstance(r, pd.DataFrame):
        raise TypeError(f"sqreg returned a frame with columns {list(r.columns)}")
    term = [t for t in r.params.index if "x1" in str(t)][1]
    return coef(r, term)


def fit_zi(est):
    def fit(df):
        r = getattr(_sp(), est)("y ~ x1", data=df, inflate=["x2"])
        return coef(r, [t for t in r.params.index if "x1" in str(t)][0])

    return fit


def fit_hurdle(df):
    r = _sp().hurdle("y ~ x1 + x2", data=df)
    return coef(r, [t for t in r.params.index if "x1" in str(t)][-1])


def fit_biprobit(df):
    r = _sp().biprobit(df, y1="y1", y2="y2", x1=["x1"], x2=["x2"])
    return coef(r, [t for t in r.params.index if "x1" in str(t)][0])


def fit_etregress(method):
    def fit(df):
        r = _sp().etregress(
            df, y="y", x=["x1"], treatment="d", z=["z", "x1"], method=method
        )
        return coef(r, [t for t in r.params.index if str(t).endswith("d")][0])

    return fit


def fit_etpoisson(df):
    r = _sp().etpoisson(df, y="y", x=["x1"], treat="d", z=["z", "x1"])
    return coef(r, [t for t in r.params.index if str(t).endswith("d")][0])


def fit_iv(est, **kw):
    def fit(df):
        r = getattr(_sp(), est)(df, y="y", x=["x1"], endog="w", instruments="z", **kw)
        return coef(r, [t for t in r.params.index if str(t).endswith("w")][0])

    return fit


def fit_gee(family, corstr):
    def fit(df):
        r = _sp().gee("y ~ x1 + x2", df, id="g", family=family, corstr=corstr, time="t")
        return coef(r, "x1")

    return fit


def fit_prais(method):
    def fit(df):
        return coef(_sp().prais("y ~ x1", df, method=method, time="t"), "x1")

    return fit


def fit_cnsreg(df):
    return coef(_sp().cnsreg("y ~ x1 + x2", df, "x1 = x2"), "x1")


def fit_nls(df):
    r = _sp().nls("y ~ {b0}*exp({b1}*x1)", df, start={"b0": 1.0, "b1": 0.1})
    return coef(r, [t for t in r.params.index if "b1" in str(t)][0])


def fit_eivreg(df):
    return coef(_sp().eivreg("y ~ x1 + x2", df, reliability={"x1": 0.8}), "x1")


def fit_sureg(df):
    r = _sp().sureg({"eq1": ("y1", ["x1"]), "eq2": ("y2", ["x2", "x3"])}, df)
    return _sys(r, "x1")


def fit_3sls(df):
    r = _sp().three_sls(
        {"eq1": ("y1", ["x1"], ["y2"]), "eq2": ("y2", ["x2", "z1", "z2"], [])}, df
    )
    return _sys(r, "y2")


def _sys(r, term):
    """SURResult reports coefficients and SEs but no interval: normal interval."""
    eq = r.equations["eq1"] if isinstance(r.equations, dict) else r.equations[0]
    return wald(eq["params"][term], eq["se"][term])


def fit_ppml(df):
    return coef(_sp().ppmlhdfe("y ~ x1", data=df, absorb="g"), "x1")


def fit_gam(df):
    r = _sp().gam("y ~ x1 + s(x2)", df)  # no interval reported for parametric terms
    return wald(r.params["x1"], r.std_errors["x1"])


def fit_fmm(df):
    return tidy_row(
        _sp().fmm("y ~ x1", df, k=2, seed=1),
        equation="outcome",
        term="x1",
        **{"class": 1},
    )


# ------------------------------------------------------------------- rows
def R(name, estimator, call, dgp, fit, truth=BETA, n=N, B=None, design=""):
    return Row(name, estimator, call, n, truth, dgp, fit, B, design)


ROWS = [
    R("regress/default", "regress", 'regress("y ~ x1 + x2")', d_linear, f("regress")),
    R(
        "regress/hc1",
        "regress",
        'regress(..., robust="hc1")',
        d_linear_het,
        f("regress", robust="hc1"),
        design="sd of the error is exp(0.5 x1)",
    ),
    R(
        "regress/hc3",
        "regress",
        'regress(..., robust="hc3")',
        d_linear_het,
        f("regress", robust="hc3"),
        design="sd of the error is exp(0.5 x1)",
    ),
    R(
        "regress/cluster",
        "regress",
        'regress(..., cluster="g")',
        d_linear_cluster,
        f("regress", cluster="g"),
        design="100 clusters of 10, cluster effects in x1 and error",
    ),
    R("logit/default", "logit", 'logit("y ~ x1 + x2")', d_logit, f("logit")),
    R(
        "logit/robust",
        "logit",
        'logit(..., robust="robust")',
        d_logit,
        f("logit", robust="robust"),
    ),
    R(
        "logit/cluster",
        "logit",
        'logit(..., cluster="g")',
        d_logit_cluster,
        f("logit", cluster="g"),
        design="100 clusters of 10, marginal logit exact, copula 0.3",
    ),
    R("probit/default", "probit", 'probit("y ~ x1 + x2")', d_probit, f("probit")),
    R(
        "probit/robust",
        "probit",
        'probit(..., robust="robust")',
        d_probit,
        f("probit", robust="robust"),
    ),
    R("cloglog/default", "cloglog", 'cloglog("y ~ x1 + x2")', d_cloglog, f("cloglog")),
    R("poisson/default", "poisson", 'poisson("y ~ x1 + x2")', d_poisson, f("poisson")),
    R(
        "poisson/robust",
        "poisson",
        'poisson(..., robust="robust")',
        d_negbin,
        f("poisson", robust="robust"),
        design="overdispersed counts (negative binomial)",
    ),
    R("nbreg/default", "nbreg", 'nbreg("y ~ x1 + x2")', d_negbin, f("nbreg")),
    R(
        "nbreg/robust",
        "nbreg",
        'nbreg(..., robust="robust")',
        d_negbin,
        f("nbreg", robust="robust"),
    ),
    R(
        "glm/gamma-log",
        "glm",
        'glm(..., family="gamma", link="log")',
        d_gamma,
        f("glm", family="gamma", link="log"),
    ),
    R(
        "glm/gamma-log-robust",
        "glm",
        'glm(..., family="gamma", link="log", robust="robust")',
        d_gamma,
        f("glm", family="gamma", link="log", robust="robust"),
    ),
    R(
        "glm/binomial",
        "glm",
        'glm(..., family="binomial")',
        d_logit,
        f("glm", family="binomial"),
    ),
    R(
        "glm/poisson-overdispersed",
        "glm",
        'glm(..., family="poisson", robust="robust")',
        d_negbin,
        f("glm", family="poisson", robust="robust"),
    ),
    R(
        "tobit/default",
        "tobit",
        'tobit(formula="y ~ x1 + x2", ll=0)',
        d_tobit,
        lambda df: coef(_sp().tobit(df, formula="y ~ x1 + x2", ll=0), "x1"),
    ),
    R(
        "tobit/robust",
        "tobit",
        'tobit(..., vce="robust")',
        d_tobit,
        lambda df: coef(
            _sp().tobit(df, formula="y ~ x1 + x2", ll=0, vce="robust"), "x1"
        ),
    ),
    R(
        "truncreg/default",
        "truncreg",
        'truncreg(formula="y ~ x1 + x2", ll=0)',
        d_truncreg,
        lambda df: coef(_sp().truncreg(df, formula="y ~ x1 + x2", ll=0), "x1"),
    ),
    R(
        "intreg/default",
        "intreg",
        "intreg(y_lower=, y_upper=, x=)",
        d_intreg,
        fit_intreg,
        design="latent outcome observed in unit-width bins",
    ),
    R(
        "heckman/twostep",
        "heckman",
        'heckman(..., method="twostep")',
        d_heckman,
        fit_heckman("twostep"),
        design="selection rate 0.63, rho 0.5, one excluded variable",
    ),
    R(
        "heckman/mle",
        "heckman",
        'heckman(..., method="mle")',
        d_heckman,
        fit_heckman("mle"),
        design="selection rate 0.63, rho 0.5, one excluded variable",
    ),
    R(
        "heckprobit/default",
        "heckprobit",
        "heckprobit(y, x, select, z)",
        d_heckprobit,
        fit_heckprobit,
        n=2000,
    ),
    R(
        "ologit/default",
        "ologit",
        'ologit("y ~ x1 + x2")',
        d_ordered_logit,
        f("ologit"),
    ),
    R(
        "oprobit/default",
        "oprobit",
        'oprobit("y ~ x1 + x2")',
        d_ordered_probit,
        f("oprobit"),
    ),
    R(
        "gologit/parallel",
        "gologit",
        "gologit(y, x, pl=True)",
        d_ordered_logit,
        fit_gologit,
    ),
    R("mlogit/default", "mlogit", 'mlogit("y ~ x1 + x2")', d_mlogit, fit_mlogit),
    R(
        "clogit/default",
        "clogit",
        'clogit("y ~ x1 + x2", group="g")',
        d_clogit,
        fit_clogit,
        design="250 choice sets of four",
    ),
    R(
        "qreg/default",
        "qreg",
        'qreg(formula="y ~ x1 + x2")',
        d_linear,
        lambda df: coef(_sp().qreg(df, "y ~ x1 + x2"), "x1"),
    ),
    R(
        "qreg/robust-het",
        "qreg",
        'qreg(..., vce="robust")',
        d_qreg_het,
        lambda df: coef(_sp().qreg(df, "y ~ x1 + x2", vce="robust"), "x1"),
        design="location-scale, sd 0.5 + x1",
    ),
    R(
        "qreg/q75",
        "qreg",
        "qreg(..., quantile=0.75)",
        d_linear,
        lambda df: coef(_sp().qreg(df, "y ~ x1 + x2", quantile=0.75), "x1"),
    ),
    R(
        "sqreg/median",
        "sqreg",
        "sqreg(quantiles=[.25,.5,.75], reps=100)",
        d_linear,
        fit_sqreg,
        B=200,
        design="bootstrap with 100 replications; B = 200",
    ),
    R(
        "zip_model/default",
        "zip_model",
        'zip_model("y ~ x1", inflate=["x2"])',
        d_zip,
        fit_zi("zip_model"),
    ),
    R(
        "zinb/default",
        "zinb",
        'zinb("y ~ x1", inflate=["x2"])',
        d_zinb,
        fit_zi("zinb"),
        n=2000,
    ),
    R(
        "hurdle/default",
        "hurdle",
        'hurdle("y ~ x1 + x2")',
        d_hurdle,
        fit_hurdle,
        design="count-part slope",
    ),
    R("fracreg/default", "fracreg", "fracreg(y, x)", d_beta, fx("fracreg")),
    R("betareg/default", "betareg", "betareg(y, x)", d_beta, fx("betareg")),
    R(
        "biprobit/default",
        "biprobit",
        "biprobit(y1, y2, x1, x2)",
        d_biprobit,
        fit_biprobit,
    ),
    R(
        "etregress/mle",
        "etregress",
        'etregress(..., method="mle")',
        d_etregress,
        fit_etregress("mle"),
        design="treatment coefficient",
    ),
    R(
        "etregress/twostep",
        "etregress",
        'etregress(..., method="twostep")',
        d_etregress,
        fit_etregress("twostep"),
        design="treatment coefficient",
    ),
    R(
        "etpoisson/default",
        "etpoisson",
        "etpoisson(y, x, treat, z)",
        d_etpoisson,
        fit_etpoisson,
        n=2000,
        B=200,
        design="treatment coefficient; B = 200 (quadrature likelihood)",
    ),
    R(
        "ivprobit/mle",
        "ivprobit",
        'ivprobit(..., method="mle")',
        d_ivprobit,
        fit_iv("ivprobit"),
    ),
    R(
        "ivprobit/twostep",
        "ivprobit",
        'ivprobit(..., method="twostep")',
        d_ivprobit,
        fit_iv("ivprobit", method="twostep"),
        truth=BETA / np.sqrt(0.75),
        design="Newey two-step estimates beta / sd(e | v)",
    ),
    R(
        "ivtobit/mle",
        "ivtobit",
        'ivtobit(..., method="mle")',
        d_ivtobit,
        fit_iv("ivtobit"),
    ),
    R(
        "ivtobit/twostep",
        "ivtobit",
        'ivtobit(..., method="twostep")',
        d_ivtobit,
        fit_iv("ivtobit", method="twostep"),
    ),
    R(
        "ivpoisson/additive",
        "ivpoisson",
        'ivpoisson(..., errors="additive")',
        d_ivpoisson_add,
        fit_iv("ivpoisson", errors="additive"),
        n=2000,
    ),
    R(
        "ivpoisson/multiplicative",
        "ivpoisson",
        'ivpoisson(..., errors="multiplicative")',
        d_ivpoisson_mult,
        fit_iv("ivpoisson", errors="multiplicative"),
        n=2000,
    ),
    R(
        "gee/gaussian-exchangeable",
        "gee",
        'gee(..., corstr="exchangeable")',
        d_gee_gauss,
        fit_gee("gaussian", "exchangeable"),
        design="100 clusters of 10",
    ),
    R(
        "gee/binomial-exchangeable",
        "gee",
        'gee(..., family="binomial", corstr="exchangeable")',
        d_logit_cluster,
        fit_gee("binomial", "exchangeable"),
        design="100 clusters of 10",
    ),
    R(
        "gee/binomial-independence",
        "gee",
        'gee(..., family="binomial")',
        d_logit_cluster,
        fit_gee("binomial", "independence"),
        design="100 clusters of 10",
    ),
    R(
        "prais/prais",
        "prais",
        'prais("y ~ x1", time="t")',
        d_prais,
        fit_prais("prais"),
        n=500,
        design="AR(1) error and regressor, rho 0.5, T = 500",
    ),
    R(
        "prais/corc",
        "prais",
        'prais(..., method="corc")',
        d_prais,
        fit_prais("corc"),
        n=500,
        design="AR(1) error and regressor, rho 0.5, T = 500",
    ),
    R(
        "cnsreg/default",
        "cnsreg",
        'cnsreg("y ~ x1 + x2", "x1 = x2")',
        d_cns,
        fit_cnsreg,
    ),
    R("nls/default", "nls", 'nls("y = {b0}*exp({b1}*x1)")', d_nls, fit_nls),
    R("robreg/mm", "robreg", 'robreg("y ~ x1 + x2")', d_linear, f("robreg")),
    R(
        "eivreg/default",
        "eivreg",
        'eivreg(..., reliability={"x1": 0.8})',
        d_eiv,
        fit_eivreg,
        design="classical error in x1, reliability 0.8 known",
    ),
    R("sureg/fgls", "sureg", "sureg({eq1, eq2})", d_sur, fit_sureg),
    R(
        "three_sls/default",
        "three_sls",
        "three_sls({eq1, eq2})",
        d_3sls,
        fit_3sls,
        design="coefficient on the endogenous regressor",
    ),
    R(
        "ppmlhdfe/default",
        "ppmlhdfe",
        'ppmlhdfe("y ~ x1", absorb="g")',
        d_ppml_fe,
        fit_ppml,
        design="100 groups of 10, group effects absorbed",
    ),
    R("gam/parametric-term", "gam", 'gam("y ~ x1 + s(x2)")', d_gam, fit_gam),
    R(
        "fmm/gaussian-2",
        "fmm",
        'fmm("y ~ x1", k=2)',
        d_fmm,
        fit_fmm,
        B=200,
        design="two well separated classes; slope of the first class; B = 200",
    ),
]

# Mechanism notes for flagged rows, written into the JSON by the harness.
MECHANISMS: dict = {}

if __name__ == "__main__":
    main(__file__, FAMILY, ROWS, MECHANISMS)
