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

# flake8: noqa: E501
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


# ------------------------------------------------- DGPs for the option rows
def _cluster_frame(rng):
    g = np.repeat(np.arange(G), N // G)
    df = pd.DataFrame({"g": g, "t": np.tile(np.arange(N // G), G)})
    df["x1"] = rng.normal(size=G)[g] * 0.7 + rng.normal(size=N) * 0.7
    df["x2"] = rng.normal(size=N)
    return df, g


def _cluster_normal(rng, g, rho=0.3):
    """Unit-variance normal error with within-cluster correlation ``rho``."""
    return np.sqrt(rho) * rng.normal(size=G)[g] + np.sqrt(1 - rho) * rng.normal(size=N)


def _cluster_uniform(rng, g, rho=0.3):
    from scipy.stats import norm

    return norm.cdf(_cluster_normal(rng, g, rho))


def d_aweights(rng):
    """Error variance 1 / w with w known: analytic weights are the efficient ones."""
    df = _xs(rng)
    df["w"] = rng.uniform(0.5, 3.0, size=N)
    df["y"] = 1 + BETA * df.x1 - 0.5 * df.x2 + rng.normal(size=N) / np.sqrt(df.w)
    return df


def d_pweights(rng, kind="linear"):
    """Sampling weights unrelated to the outcome: the weighted estimator is consistent."""
    df = {"linear": d_linear_het, "logit": d_logit, "poisson": d_negbin}[kind](rng)
    df["w"] = rng.uniform(0.5, 3.0, size=N)
    return df


def d_probit_cluster(rng):
    df, g = _cluster_frame(rng)
    df["y"] = (-0.3 + BETA * df.x1 - 0.5 * df.x2 + _cluster_normal(rng, g) > 0).astype(
        int
    )
    return df


def d_count_cluster(rng):
    """Poisson mean exp(xb) times a mean-one cluster effect and gamma noise."""
    df, g = _cluster_frame(rng)
    a = np.exp(0.4 * rng.normal(size=G) - 0.08)[g]
    mu = np.exp(0.5 + BETA * df.x1 - 0.3 * df.x2)
    df["y"] = rng.poisson(mu * a * rng.gamma(shape=4.0, scale=0.25, size=N))
    return df


def d_poisson_exposure(rng):
    df = _xs(rng)
    df["e"] = rng.uniform(0.5, 5.0, size=N)
    df["y"] = rng.poisson(df.e * np.exp(-0.5 + BETA * df.x1 - 0.3 * df.x2))
    return df


def d_nb1(rng):
    """NB1: variance (1 + 1.0) mu, gamma shape proportional to the mean."""
    df = _xs(rng)
    mu = np.exp(0.5 + BETA * df.x1 - 0.3 * df.x2)
    df["y"] = rng.poisson(rng.gamma(shape=mu / 1.0, scale=1.0))
    return df


def d_tobit_cluster(rng):
    df, g = _cluster_frame(rng)
    df["y"] = np.maximum(
        0.0, 0.5 + BETA * df.x1 - 0.5 * df.x2 + _cluster_normal(rng, g)
    )
    return df


def d_tobit_two(rng):
    df = _xs(rng)
    df["y"] = np.clip(0.8 + BETA * df.x1 - 0.5 * df.x2 + rng.normal(size=N), 0.0, 2.0)
    return df


def d_tobit_het(rng):
    """Symmetric heteroskedastic error: the case symmetrically censored LS is for."""
    df = _xs(rng)
    e = rng.normal(size=N) * np.exp(0.3 * df.x2)
    df["y"] = np.maximum(0.0, 1.5 + BETA * df.x1 - 0.5 * df.x2 + e)
    return df


def d_heckman_cluster(rng):
    df, g = _cluster_frame(rng)
    df["z"] = rng.normal(size=N)
    u = _cluster_normal(rng, g)
    e = 0.5 * u + np.sqrt(0.75) * _cluster_normal(rng, g)
    df["s"] = (0.5 + df.z + 0.5 * df.x1 + u > 0).astype(int)
    df["y"] = np.where(df.s == 1, 1 + BETA * df.x1 - 0.5 * df.x2 + e, np.nan)
    return df


def d_ologit_cluster(rng):
    df, g = _cluster_frame(rng)
    u = _cluster_uniform(rng, g)
    lat = BETA * df.x1 - 0.5 * df.x2 + np.log(u / (1 - u))
    df["y"] = np.digitize(lat, [-1.0, 0.0, 1.0])
    return df


def d_mlogit_cluster(rng):
    df, g = _cluster_frame(rng)
    gum = [-np.log(-np.log(_cluster_uniform(rng, g))) for _ in range(3)]
    u = np.column_stack(
        [
            gum[0],
            0.2 + BETA * df.x1 - 0.5 * df.x2 + gum[1],
            -0.2 - 0.5 * df.x1 + 0.3 * df.x2 + gum[2],
        ]
    )
    df["y"] = u.argmax(axis=1)
    return df


def d_qreg_cluster(rng):
    df, g = _cluster_frame(rng)
    df["y"] = 1 + BETA * df.x1 - 0.5 * df.x2 + _cluster_normal(rng, g)
    return df


def d_gee_ar1(rng):
    df, g = _cluster_frame(rng)
    m = N // G
    e = np.zeros((G, m))
    e[:, 0] = rng.normal(size=G)
    for j in range(1, m):
        e[:, j] = 0.6 * e[:, j - 1] + np.sqrt(1 - 0.36) * rng.normal(size=G)
    df["y"] = 1 + BETA * df.x1 - 0.5 * df.x2 + e.ravel()
    return df


def d_gee_poisson(rng):
    return d_count_cluster(rng)


def d_hurdle_nb(rng):
    df = _xs(rng, 2000)
    pos = rng.uniform(size=2000) < _expit(0.5 + 0.5 * df.x2)
    mu = np.exp(1.0 + BETA * df.x1).to_numpy()
    y = np.zeros(2000, dtype=int)
    for i in np.where(pos)[0]:
        v = 0
        while v == 0:
            v = rng.poisson(mu[i] * rng.gamma(2.0, 0.5))
        y[i] = v
    df["y"] = y
    return df


def d_fracprobit(rng):
    from scipy.stats import norm

    df = _xs(rng)
    mu = norm.cdf(BETA * df.x1 - 0.5 * df.x2)
    df["y"] = rng.beta(mu * 5.0, (1 - mu) * 5.0)
    return df


def d_nls_het(rng):
    df = pd.DataFrame({"x1": rng.uniform(0, 2, size=N)})
    df["y"] = 2.0 * np.exp(BETA * df.x1) + rng.normal(size=N) * (0.5 + df.x1)
    return df


def d_cns_het(rng):
    df = _xs(rng)
    df["y"] = 1 + BETA * df.x1 + BETA * df.x2 + rng.normal(size=N) * np.exp(0.5 * df.x1)
    return df


def d_outliers(rng):
    """Symmetric contamination: 10% of the errors have sd 5."""
    df = _xs(rng)
    e = np.where(rng.uniform(size=N) < 0.1, 5.0, 1.0) * rng.normal(size=N)
    df["y"] = 1 + BETA * df.x1 - 0.5 * df.x2 + e
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


def fit_tobit(**kw):
    def fit(df):
        return coef(_sp().tobit(df, formula="y ~ x1 + x2", **kw), "x1")

    return fit


def fit_qreg(**kw):
    def fit(df):
        return coef(_sp().qreg(df, "y ~ x1 + x2", **kw), "x1")

    return fit


def fit_heckman_vce(**kw):
    def fit(df):
        r = _sp().heckman(
            df, y="y", x=["x1", "x2"], select="s", z=["z", "x1"], method="ml", **kw
        )
        return coef(r, "x1")

    return fit


def fit_mlogit_kw(**kw):
    def fit(df):
        r = _sp().mlogit("y ~ x1 + x2", data=df, **kw)
        return coef(r, [t for t in r.params.index if "x1" in str(t)][0])

    return fit


def fit_gee_kw(family="gaussian", **kw):
    def fit(df):
        return coef(
            _sp().gee("y ~ x1 + x2", df, id="g", family=family, time="t", **kw), "x1"
        )

    return fit


def fit_hurdle_kw(**kw):
    def fit(df):
        r = _sp().hurdle("y ~ x1 + x2", data=df, **kw)
        return coef(r, [t for t in r.params.index if "x1" in str(t)][-1])

    return fit


def fit_zi_kw(est, **kw):
    def fit(df):
        r = getattr(_sp(), est)("y ~ x1", data=df, inflate=["x2"], **kw)
        return coef(r, [t for t in r.params.index if "x1" in str(t)][0])

    return fit


def fit_nls_kw(**kw):
    def fit(df):
        r = _sp().nls("y ~ {b0}*exp({b1}*x1)", df, start={"b0": 1.0, "b1": 0.1}, **kw)
        return coef(r, [t for t in r.params.index if "b1" in str(t)][0])

    return fit


def fit_cnsreg_kw(**kw):
    def fit(df):
        return coef(_sp().cnsreg("y ~ x1 + x2", df, "x1 = x2", **kw), "x1")

    return fit


def fit_prais_kw(**kw):
    def fit(df):
        return coef(_sp().prais("y ~ x1", df, time="t", **kw), "x1")

    return fit


def fit_iv_kw(est, term_end="w", **kw):
    return fit_iv(est, **kw)


def fit_biprobit_kw(**kw):
    def fit(df):
        r = _sp().biprobit(df, y1="y1", y2="y2", x1=["x1"], x2=["x2"], **kw)
        return coef(r, [t for t in r.params.index if "x1" in str(t)][0])

    return fit


def fit_gam_kw(**kw):
    def fit(df):
        r = _sp().gam("y ~ x1 + s(x2)", df, **kw)
        return wald(r.params["x1"], r.std_errors["x1"])

    return fit


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

CL = "100 clusters of 10, within-cluster correlation 0.3 in the error"

OPTION_ROWS = [
    R(
        "regress/hc2",
        "regress",
        'regress(..., robust="hc2")',
        d_linear_het,
        f("regress", robust="hc2"),
        design="sd of the error is exp(0.5 x1)",
    ),
    R(
        "regress/hc4",
        "regress",
        'regress(..., robust="hc4")',
        d_linear_het,
        f("regress", robust="hc4"),
        design="sd of the error is exp(0.5 x1)",
    ),
    R(
        "regress/cr2",
        "regress",
        'regress(..., cluster="g", vce="cr2")',
        d_linear_cluster,
        f("regress", cluster="g", vce="cr2"),
        design="100 clusters of 10",
    ),
    R(
        "regress/cr3",
        "regress",
        'regress(..., cluster="g", vce="cr3")',
        d_linear_cluster,
        f("regress", cluster="g", vce="cr3"),
        design="100 clusters of 10",
    ),
    R(
        "regress/jackknife",
        "regress",
        'regress(..., cluster="g", vce="jackknife")',
        d_linear_cluster,
        f("regress", cluster="g", vce="jackknife"),
        design="100 clusters of 10",
    ),
    R(
        "regress/aweights",
        "regress",
        'regress(..., weights="w")',
        d_aweights,
        f("regress", weights="w"),
        design="error variance 1 / w",
    ),
    R(
        "regress/pweights",
        "regress",
        'regress(..., weights="w", weights_as="sampling")',
        lambda rng: d_pweights(rng, "linear"),
        f("regress", weights="w", weights_as="sampling"),
        design="weights independent of the data, heteroskedastic error",
    ),
    R(
        "logit/hc1",
        "logit",
        'logit(..., robust="hc1")',
        d_logit,
        f("logit", robust="hc1"),
    ),
    R(
        "logit/pweights",
        "logit",
        'logit(..., weights="w", weights_as="sampling")',
        lambda rng: d_pweights(rng, "logit"),
        f("logit", weights="w", weights_as="sampling"),
        design="weights independent of the data",
    ),
    R(
        "probit/cluster",
        "probit",
        'probit(..., cluster="g")',
        d_probit_cluster,
        f("probit", cluster="g"),
        design=CL,
    ),
    R(
        "cloglog/robust",
        "cloglog",
        'cloglog(..., robust="robust")',
        d_cloglog,
        f("cloglog", robust="robust"),
    ),
    R(
        "poisson/cluster",
        "poisson",
        'poisson(..., cluster="g")',
        d_count_cluster,
        f("poisson", cluster="g"),
        design="100 clusters of 10, mean-one cluster effect",
    ),
    R(
        "poisson/pweights",
        "poisson",
        'poisson(..., weights="w", weights_as="sampling")',
        lambda rng: d_pweights(rng, "poisson"),
        f("poisson", weights="w", weights_as="sampling"),
        design="weights independent of the data, overdispersed counts",
    ),
    R(
        "poisson/exposure",
        "poisson",
        'poisson(..., exposure="e")',
        d_poisson_exposure,
        f("poisson", exposure="e"),
    ),
    R(
        "nbreg/cluster",
        "nbreg",
        'nbreg(..., cluster="g")',
        d_count_cluster,
        f("nbreg", cluster="g"),
        design="100 clusters of 10, mean-one cluster effect",
    ),
    R(
        "nbreg/constant-dispersion",
        "nbreg",
        'nbreg(..., dispersion="constant")',
        d_nb1,
        f("nbreg", dispersion="constant"),
        design="NB1 counts, variance 2 mu",
    ),
    R(
        "glm/binomial-cluster",
        "glm",
        'glm(..., family="binomial", cluster="g")',
        d_logit_cluster,
        f("glm", family="binomial", cluster="g"),
        design=CL,
    ),
    R(
        "glm/poisson-exposure",
        "glm",
        'glm(..., family="poisson", exposure="e")',
        d_poisson_exposure,
        f("glm", family="poisson", exposure="e"),
    ),
    R(
        "tobit/cluster",
        "tobit",
        'tobit(..., vce="cluster g")',
        d_tobit_cluster,
        fit_tobit(ll=0, vce="cluster g"),
        design=CL,
    ),
    R(
        "tobit/two-limit",
        "tobit",
        "tobit(..., ll=0, ul=2)",
        d_tobit_two,
        fit_tobit(ll=0, ul=2),
    ),
    R(
        "tobit/scls",
        "tobit",
        'tobit(..., method="scls")',
        d_tobit_het,
        fit_tobit(ll=0, method="scls"),
        design="symmetric heteroskedastic error, 10% censored",
    ),
    R(
        "truncreg/robust",
        "truncreg",
        'truncreg(..., robust="robust")',
        d_truncreg,
        lambda df: coef(
            _sp().truncreg(df, formula="y ~ x1 + x2", ll=0, robust="robust"), "x1"
        ),
    ),
    R(
        "intreg/robust",
        "intreg",
        'intreg(..., vce="robust")',
        d_intreg,
        lambda df: coef(
            _sp().intreg(df, y_lower="lo", y_upper="hi", x=["x1", "x2"], vce="robust"),
            "x1",
        ),
    ),
    R(
        "heckman/mle-robust",
        "heckman",
        'heckman(..., method="ml", vce="robust")',
        d_heckman,
        fit_heckman_vce(vce="robust"),
    ),
    R(
        "heckman/mle-cluster",
        "heckman",
        'heckman(..., method="ml", vce="cluster g")',
        d_heckman_cluster,
        fit_heckman_vce(vce="cluster g"),
        design=CL,
    ),
    R(
        "ologit/robust",
        "ologit",
        'ologit(..., robust="robust")',
        d_ordered_logit,
        f("ologit", robust="robust"),
    ),
    R(
        "ologit/cluster",
        "ologit",
        'ologit(..., cluster="g")',
        d_ologit_cluster,
        f("ologit", cluster="g"),
        design=CL,
    ),
    R(
        "oprobit/robust",
        "oprobit",
        'oprobit(..., robust="robust")',
        d_ordered_probit,
        f("oprobit", robust="robust"),
    ),
    R(
        "mlogit/robust",
        "mlogit",
        'mlogit(..., robust="robust")',
        d_mlogit,
        fit_mlogit_kw(robust="robust"),
    ),
    R(
        "mlogit/cluster",
        "mlogit",
        'mlogit(..., cluster="g")',
        d_mlogit_cluster,
        fit_mlogit_kw(cluster="g"),
        design=CL,
    ),
    R(
        "clogit/robust",
        "clogit",
        'clogit(..., robust="robust")',
        d_clogit,
        lambda df: coef(
            _sp().clogit("y ~ x1 + x2", data=df, group="g", robust="robust"), "x1"
        ),
    ),
    R(
        "qreg/nid",
        "qreg",
        'qreg(..., vce="nid")',
        d_qreg_het,
        fit_qreg(vce="nid"),
        design="location-scale, sd 0.5 + x1",
    ),
    R(
        "qreg/kernel",
        "qreg",
        'qreg(..., vce="kernel")',
        d_qreg_het,
        fit_qreg(vce="kernel"),
        design="location-scale, sd 0.5 + x1",
    ),
    R(
        "qreg/cluster",
        "qreg",
        'qreg(..., vce="cluster g")',
        d_qreg_cluster,
        fit_qreg(vce="cluster g"),
        design=CL,
    ),
    R(
        "zip_model/robust",
        "zip_model",
        'zip_model(..., robust="robust")',
        d_zip,
        fit_zi_kw("zip_model", robust="robust"),
    ),
    R(
        "zinb/robust",
        "zinb",
        'zinb(..., robust="robust")',
        d_zinb,
        fit_zi_kw("zinb", robust="robust"),
        n=2000,
    ),
    R(
        "hurdle/robust",
        "hurdle",
        'hurdle(..., robust="robust")',
        d_hurdle,
        fit_hurdle_kw(robust="robust"),
    ),
    R(
        "hurdle/negbin",
        "hurdle",
        'hurdle(..., count_model="negbin")',
        d_hurdle_nb,
        fit_hurdle_kw(count_model="negbin"),
        n=2000,
        design="zero-truncated negative binomial",
    ),
    R(
        "fracreg/probit",
        "fracreg",
        'fracreg(..., link="probit")',
        d_fracprobit,
        fx("fracreg", link="probit"),
    ),
    R(
        "betareg/robust",
        "betareg",
        'betareg(..., robust="robust")',
        d_beta,
        fx("betareg", robust="robust"),
    ),
    R(
        "biprobit/robust",
        "biprobit",
        'biprobit(..., robust="robust")',
        d_biprobit,
        fit_biprobit_kw(robust="robust"),
    ),
    R(
        "etregress/robust",
        "etregress",
        'etregress(..., robust="robust")',
        d_etregress,
        lambda df: (
            lambda r: coef(r, [t for t in r.params.index if str(t).endswith("d")][0])
        )(
            _sp().etregress(
                df, y="y", x=["x1"], treatment="d", z=["z", "x1"], robust="robust"
            )
        ),
        design="treatment coefficient",
    ),
    R(
        "ivprobit/robust",
        "ivprobit",
        'ivprobit(..., vce="robust")',
        d_ivprobit,
        fit_iv("ivprobit", vce="robust"),
    ),
    R(
        "ivtobit/robust",
        "ivtobit",
        'ivtobit(..., vce="robust")',
        d_ivtobit,
        fit_iv("ivtobit", vce="robust"),
    ),
    R(
        "ivpoisson/onestep",
        "ivpoisson",
        'ivpoisson(..., method="onestep")',
        d_ivpoisson_add,
        fit_iv("ivpoisson", method="onestep"),
        n=2000,
    ),
    R(
        "gee/gaussian-ar1",
        "gee",
        'gee(..., corstr="ar1")',
        d_gee_ar1,
        fit_gee_kw(corstr="ar1"),
        design="100 clusters of 10, AR(1) errors rho 0.6",
    ),
    R(
        "gee/gaussian-ar1-model",
        "gee",
        'gee(..., corstr="ar1", vce="model")',
        d_gee_ar1,
        fit_gee_kw(corstr="ar1", vce="model"),
        design="100 clusters of 10, AR(1) errors rho 0.6: the working correlation is true",
    ),
    R(
        "gee/gaussian-unstructured",
        "gee",
        'gee(..., corstr="unstructured")',
        d_gee_ar1,
        fit_gee_kw(corstr="unstructured"),
        design="100 clusters of 10, AR(1) errors rho 0.6",
    ),
    R(
        "gee/poisson-exchangeable",
        "gee",
        'gee(..., family="poisson", corstr="exchangeable")',
        d_gee_poisson,
        fit_gee_kw("poisson", corstr="exchangeable"),
        design="100 clusters of 10, mean-one cluster effect",
    ),
    R(
        "prais/robust",
        "prais",
        'prais(..., vce="robust")',
        d_prais,
        fit_prais_kw(vce="robust"),
        n=500,
        design="AR(1) error and regressor, rho 0.5, T = 500",
    ),
    R(
        "prais/twostep",
        "prais",
        "prais(..., twostep=True)",
        d_prais,
        fit_prais_kw(twostep=True),
        n=500,
        design="AR(1) error and regressor, rho 0.5, T = 500",
    ),
    R(
        "cnsreg/robust",
        "cnsreg",
        'cnsreg(..., vce="robust")',
        d_cns_het,
        fit_cnsreg_kw(vce="robust"),
        design="sd of the error is exp(0.5 x1)",
    ),
    R(
        "nls/robust",
        "nls",
        'nls(..., vce="robust")',
        d_nls_het,
        fit_nls_kw(vce="robust"),
        design="error sd 0.5 + x1",
    ),
    R(
        "nls/hc3",
        "nls",
        'nls(..., vce="hc3")',
        d_nls_het,
        fit_nls_kw(vce="hc3"),
        design="error sd 0.5 + x1",
    ),
    R(
        "robreg/mm-contaminated",
        "robreg",
        'robreg("y ~ x1 + x2")',
        d_outliers,
        f("robreg"),
        design="10% of the errors have sd 5",
    ),
    R(
        "robreg/m",
        "robreg",
        'robreg(..., method="m")',
        d_outliers,
        f("robreg", method="m"),
        design="10% of the errors have sd 5",
    ),
    R("gam/robust", "gam", 'gam(..., vce="robust")', d_gam, fit_gam_kw(vce="robust")),
    R(
        "sureg/mle",
        "sureg",
        'sureg({eq1, eq2}, method="mle")',
        d_sur,
        lambda df: _sys(
            _sp().sureg(
                {"eq1": ("y1", ["x1"]), "eq2": ("y2", ["x2", "x3"])}, df, method="mle"
            ),
            "x1",
        ),
    ),
]

ROWS = ROWS + OPTION_ROWS

# Mechanism notes for flagged rows, written into the JSON by the harness.
MECHANISMS: dict = {
    "gee/gaussian-unstructured": (
        "Known small-sample property of the sandwich with an unstructured working "
        "correlation: 45 correlations are estimated from 100 clusters, the fitted weights "
        "are noisy and the robust SE is biased down (SE/SD 0.89). In 7% of the draws the "
        "estimated correlation matrix is not positive definite and sp.gee refuses it with "
        "a typed error. The AR(1) and exchangeable rows on the same data are calibrated."
    ),
}

if __name__ == "__main__":
    main(__file__, FAMILY, ROWS, MECHANISMS)
