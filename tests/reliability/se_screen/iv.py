"""SE calibration screen: instrumental variables.

n = 1,000, one endogenous regressor ``d`` with true coefficient 0.5, one
exogenous control. Three instrument designs, fixed before the first run:

* strong: one or three instruments, first-stage F in the hundreds;
* many: 30 instruments, concentration parameter 300 (first-stage F about 11);
* weak: concentration parameter about 5. Only the weak-instrument robust
  sets are screened there, and only for coverage, which is what they claim.

    python tests/reliability/se_screen/iv.py [--only name] [--confirm]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _harness import Row, coef, main, tidy_row, wald  # noqa: E402,F401

FAMILY = "iv"
N = 1000
BETA = 0.5
G = 100
K_MANY = 30


def _sp():
    import statspai as sp

    return sp


def _iv(rng, k=1, pi=0.5, het=False, cluster=False, n=N):
    z = rng.normal(size=(n, k))
    df = pd.DataFrame(z, columns=[f"z{j + 1}" for j in range(k)])
    df["x1"] = rng.normal(size=n)
    v = rng.normal(size=n)
    e = rng.normal(size=n)
    if cluster:
        g = np.repeat(np.arange(G), n // G)
        df["g"] = g
        zc = rng.normal(size=(G, k))[g]
        for j in range(k):
            df[f"z{j + 1}"] = 0.7 * zc[:, j] + 0.7 * df[f"z{j + 1}"]
        v = 0.7 * rng.normal(size=G)[g] + 0.7 * v
        e = 0.7 * rng.normal(size=G)[g] + 0.7 * e
    u = 0.5 * v + np.sqrt(0.75) * e
    if het:
        u = u * np.exp(0.4 * df["z1"])
    zs = df[[f"z{j + 1}" for j in range(k)]].to_numpy()
    df["d"] = pi * zs.sum(axis=1) + 0.3 * df.x1 + v
    df["y"] = 1 + BETA * df.d + 0.3 * df.x1 + u
    return df


def d_strong(rng):
    return _iv(rng)


def d_strong_het(rng):
    return _iv(rng, het=True)


def d_strong_cluster(rng):
    return _iv(rng, cluster=True)


def d_over(rng):
    return _iv(rng, k=3, pi=0.3)


def d_over_het(rng):
    return _iv(rng, k=3, pi=0.3, het=True)


def d_many(rng):
    return _iv(rng, k=K_MANY, pi=0.1)


def d_weak1(rng):
    return _iv(rng, k=1, pi=0.07)


def d_weak3(rng):
    return _iv(rng, k=3, pi=0.04)


def d_sparse(rng, k=50):
    """50 candidate instruments, three matter."""
    df = _iv(rng, k=k, pi=0.0)
    v = df.d - 0.3 * df.x1
    df["d"] = 0.4 * (df.z1 + df.z2 + df.z3) + 0.3 * df.x1 + v
    u = df.y - 1 - BETA * (0.3 * df.x1 + v) - 0.3 * df.x1
    df["y"] = 1 + BETA * df.d + 0.3 * df.x1 + u
    return df


def d_ivqreg(rng):
    df = _iv(rng)
    return df


# ------------------------------------------------------------------- fits
def _fml(k):
    return "y ~ (d ~ " + " + ".join(f"z{j + 1}" for j in range(k)) + ") + x1"


def _zs(k):
    return [f"z{j + 1}" for j in range(k)]


def fit_iv(k=1, **kw):
    def fit(df):
        return coef(_sp().iv(_fml(k), data=df, **kw), "d")

    return fit


def fit_ivreg(k=1, **kw):
    def fit(df):
        return coef(_sp().ivreg(_fml(k), df, **kw), "d")

    return fit


def fit_liml(k, **kw):
    def fit(df):
        r = _sp().liml(data=df, y="y", x_endog=["d"], x_exog=["x1"], z=_zs(k), **kw)
        return coef(r, "d")

    return fit


def fit_jive(k, **kw):
    def fit(df):
        r = _sp().jive(df, y="y", x_endog=["d"], x_exog=["x1"], z=_zs(k), **kw)
        return coef(r, "d")

    return fit


def fit_jive_variant(name, k):
    def fit(df):
        r = getattr(_sp().iv, name)("y", "d", _zs(k), exog=["x1"], data=df)
        return _jr(r)

    return fit


def _jr(r):
    """JIVEResult reports coefficients and SEs but no interval: normal interval."""
    if hasattr(r, "ci"):
        return coef(r)
    return wald(r.params["endog0"], r.std_errors["endog0"])


def fit_jive_mw(df):
    r = _sp().iv.jive_mw(df, y="y", endog="d", instruments=_zs(K_MANY), exog=["x1"])
    return _jr(r)


def fit_gmm_generic(df):
    X = np.column_stack([np.ones(len(df)), df.d, df.x1])
    Z = np.column_stack([np.ones(len(df)), df.z1, df.z2, df.z3, df.x1])
    y = df.y.to_numpy()

    def g(theta, data):
        return Z * (y - X @ theta)[:, None]

    r = _sp().gmm(g, np.zeros(3), param_names=["const", "d", "x1"])
    return coef(r, "d")


def fit_lasso_iv(df):
    r = _sp().iv.lasso_iv(df, y="y", x_endog=["d"], x_exog=["x1"], z=_zs(50))
    return coef(r, "d")


def fit_rlasso_iv(df):
    r = _sp().rlasso_iv("y", "d", _zs(50), x=["x1"], data=df)
    lo, hi = np.ravel(np.asarray(r.conf_int()))[:2]
    return float(np.ravel(r.coef)[0]), float(np.ravel(r.se)[0]), float(lo), float(hi)


def fit_ivqreg(df):
    r = _sp().ivqreg(df, y="y", endog="d", instruments="z1", exog=["x1"], tau=0.5)
    return coef(r, "d")


def _in_set(cs, truth=BETA):
    if cs.is_empty:
        return False
    return any(lo <= truth <= hi for lo, hi in cs.as_intervals())


def cover(fn, k, **kw):
    def fit(df):
        cs = getattr(_sp().iv, fn)("y", "d", _zs(k), exog=["x1"], data=df, **kw)
        return _in_set(cs)

    return fit


def cover_many_weak_ar(df):
    r = _sp().iv.many_weak_ar(
        df, y="y", endog="d", instruments=_zs(K_MANY), exog=["x1"]
    )
    lo, hi = r.ci
    return lo <= BETA <= hi


def R(
    name,
    estimator,
    call,
    dgp,
    fit,
    truth=BETA,
    n=N,
    B=None,
    design="",
    interval_only=False,
):
    return Row(name, estimator, call, n, truth, dgp, fit, B, design, interval_only)


STRONG = "one instrument, first-stage F about 250"
OVER = "three instruments, first-stage F about 90"
MANY = "30 instruments, concentration parameter 300"

ROWS = [
    R("iv/2sls", "iv", 'iv("y ~ (d ~ z1) + x1")', d_strong, fit_iv(), design=STRONG),
    R(
        "iv/2sls-robust",
        "iv",
        'iv(..., robust="robust")',
        d_strong_het,
        fit_iv(robust="robust"),
        design=STRONG + ", error sd exp(0.4 z1)",
    ),
    R(
        "iv/2sls-cluster",
        "iv",
        'iv(..., cluster="g")',
        d_strong_cluster,
        fit_iv(cluster="g"),
        design="100 clusters of 10, cluster components in instrument and errors",
    ),
    R(
        "ivreg/2sls-over",
        "ivreg",
        'ivreg("y ~ (d ~ z1 + z2 + z3) + x1")',
        d_over,
        fit_ivreg(3),
        design=OVER,
    ),
    R(
        "iv/liml",
        "iv",
        'iv(..., method="liml")',
        d_over,
        fit_iv(3, method="liml"),
        design=OVER,
    ),
    R(
        "iv/fuller",
        "iv",
        'iv(..., method="fuller")',
        d_over,
        fit_iv(3, method="fuller"),
        design=OVER,
    ),
    R(
        "iv/gmm",
        "iv",
        'iv(..., method="gmm")',
        d_over_het,
        fit_iv(3, method="gmm"),
        design=OVER + ", error sd exp(0.4 z1)",
    ),
    R(
        "iv/gmm-robust",
        "iv",
        'iv(..., method="gmm", robust="robust")',
        d_over_het,
        fit_iv(3, method="gmm", robust="robust"),
        design=OVER + ", error sd exp(0.4 z1)",
    ),
    R(
        "iv/jive",
        "iv",
        'iv(..., method="jive")',
        d_many,
        fit_iv(K_MANY, method="jive"),
        design=MANY,
    ),
    R(
        "liml/default",
        "liml",
        "liml(y, x_endog, x_exog, z)",
        d_over,
        fit_liml(3),
        design=OVER,
    ),
    R(
        "liml/robust",
        "liml",
        'liml(..., robust="robust")',
        d_over_het,
        fit_liml(3, robust="robust"),
        design=OVER + ", error sd exp(0.4 z1)",
    ),
    R(
        "liml/many",
        "liml",
        "liml(...), 30 instruments",
        d_many,
        fit_liml(K_MANY),
        design=MANY,
    ),
    R(
        "liml/fuller1",
        "liml",
        "liml(..., fuller=1)",
        d_many,
        fit_liml(K_MANY, fuller=1.0),
        design=MANY,
    ),
    R(
        "jive/jive1",
        "jive",
        "jive(y, x_endog, x_exog, z)",
        d_many,
        fit_jive(K_MANY),
        design=MANY,
    ),
    R(
        "jive/robust",
        "jive",
        'jive(..., robust="robust")',
        d_many,
        fit_jive(K_MANY, robust="robust"),
        design=MANY,
    ),
    R(
        "iv.ujive/default",
        "iv.ujive",
        "iv.ujive(y, d, z, exog)",
        d_many,
        fit_jive_variant("ujive", K_MANY),
        design=MANY,
    ),
    R(
        "iv.ijive/default",
        "iv.ijive",
        "iv.ijive(y, d, z, exog)",
        d_many,
        fit_jive_variant("ijive", K_MANY),
        design=MANY,
    ),
    R(
        "iv.rjive/default",
        "iv.rjive",
        "iv.rjive(y, d, z, exog)",
        d_many,
        fit_jive_variant("rjive", K_MANY),
        design=MANY,
    ),
    R(
        "iv.jive1/default",
        "iv.jive1",
        "iv.jive1(y, d, z, exog)",
        d_many,
        fit_jive_variant("jive1", K_MANY),
        design=MANY,
    ),
    R(
        "iv.jive_mw/default",
        "iv.jive_mw",
        "iv.jive_mw(y, endog, instruments, exog)",
        d_many,
        fit_jive_mw,
        design=MANY,
    ),
    R(
        "gmm/linear-iv-twostep",
        "gmm",
        "gmm(moment_fn, theta0) with Z'(y - Xb)",
        d_over_het,
        fit_gmm_generic,
        design=OVER + ", error sd exp(0.4 z1)",
    ),
    R(
        "lasso_iv/default",
        "lasso_iv",
        "iv.lasso_iv(y, x_endog, x_exog, z)",
        d_sparse,
        fit_lasso_iv,
        design="50 candidate instruments, three relevant",
    ),
    R(
        "rlasso_iv/default",
        "rlasso_iv",
        "rlasso_iv(y, d, z, x)",
        d_sparse,
        fit_rlasso_iv,
        design="50 candidate instruments, three relevant",
    ),
    R(
        "anderson_rubin_ci/weak-1",
        "anderson_rubin_ci",
        "anderson_rubin_ci(y, d, [z1])",
        d_weak1,
        cover("anderson_rubin_ci", 1),
        interval_only=True,
        design="one instrument, concentration parameter 4.9",
    ),
    R(
        "anderson_rubin_ci/weak-3",
        "anderson_rubin_ci",
        "anderson_rubin_ci(y, d, [z1, z2, z3])",
        d_weak3,
        cover("anderson_rubin_ci", 3),
        interval_only=True,
        design="three instruments, concentration parameter 4.8",
    ),
    R(
        "anderson_rubin_ci/strong",
        "anderson_rubin_ci",
        "anderson_rubin_ci(y, d, [z1, z2, z3])",
        d_over,
        cover("anderson_rubin_ci", 3),
        interval_only=True,
        design=OVER,
    ),
    R(
        "conditional_lr_ci/weak-3",
        "conditional_lr_ci",
        "conditional_lr_ci(y, d, [z1, z2, z3])",
        d_weak3,
        cover("conditional_lr_ci", 3),
        interval_only=True,
        B=200,
        design="three instruments, concentration parameter 4.8; B = 200",
    ),
    R(
        "k_test_ci/weak-3",
        "k_test_ci",
        "k_test_ci(y, d, [z1, z2, z3])",
        d_weak3,
        cover("k_test_ci", 3),
        interval_only=True,
        design="three instruments, concentration parameter 4.8",
    ),
    R(
        "many_weak_ar/many",
        "iv.many_weak_ar",
        "iv.many_weak_ar(y, endog, instruments)",
        d_many,
        cover_many_weak_ar,
        interval_only=True,
        design=MANY,
    ),
]


def d_many_weak(rng):
    """30 instruments, concentration parameter 60 (first-stage F about 3)."""
    return _iv(rng, k=K_MANY, pi=0.045)


def d_weak_cluster(rng):
    return _iv(rng, k=1, pi=0.1, cluster=True)


def d_strong_w(rng):
    df = _iv(rng, het=True)
    df["wt"] = rng.uniform(0.5, 3.0, size=N)
    return df


def cover_ar_cluster(df):
    cs = _sp().iv.anderson_rubin_ci("y", "d", ["z1"], exog=["x1"], data=df, cluster="g")
    return _in_set(cs)


def fit_jive_mw_weak(df):
    r = _sp().iv.jive_mw(df, y="y", endog="d", instruments=_zs(K_MANY), exog=["x1"])
    return _jr(r)


MANY_WEAK = "30 instruments, concentration parameter 60"
CLUSTER = "100 clusters of 10, cluster components in instrument and errors"

OPTION_ROWS = [
    R(
        "ivreg/hc3",
        "ivreg",
        'ivreg(..., robust="hc3")',
        d_strong_het,
        fit_ivreg(robust="hc3"),
        design=STRONG + ", error sd exp(0.4 z1)",
    ),
    R(
        "ivreg/hc0",
        "ivreg",
        'ivreg(..., robust="hc0")',
        d_strong_het,
        fit_ivreg(robust="hc0"),
        design=STRONG + ", error sd exp(0.4 z1)",
    ),
    R(
        "ivreg/pweights",
        "ivreg",
        'ivreg(..., weights="wt", weights_as="sampling")',
        d_strong_w,
        fit_ivreg(weights="wt", weights_as="sampling"),
        design=STRONG + ", weights independent of the data",
    ),
    R(
        "iv/liml-cluster",
        "iv",
        'iv(..., method="liml", cluster="g")',
        d_strong_cluster,
        fit_iv(method="liml", cluster="g"),
        design=CLUSTER,
    ),
    R(
        "iv/gmm-cluster",
        "iv",
        'iv(..., method="gmm", cluster="g")',
        d_strong_cluster,
        fit_iv(method="gmm", cluster="g"),
        design=CLUSTER,
    ),
    R(
        "liml/cluster",
        "liml",
        'liml(..., cluster="g")',
        d_strong_cluster,
        lambda df: coef(
            _sp().liml(
                data=df, y="y", x_endog=["d"], x_exog=["x1"], z=["z1"], cluster="g"
            ),
            "d",
        ),
        design=CLUSTER,
    ),
    R(
        "jive/cluster",
        "jive",
        'jive(..., cluster="g")',
        d_strong_cluster,
        lambda df: coef(
            _sp().jive(df, y="y", x_endog=["d"], x_exog=["x1"], z=["z1"], cluster="g"),
            "d",
        ),
        design=CLUSTER,
    ),
    R(
        "jive/jive2",
        "jive",
        'jive(..., variant="jive2")',
        d_many,
        fit_jive(K_MANY, variant="jive2"),
        design=MANY,
    ),
    R(
        "iv.jive_mw/many-weak",
        "iv.jive_mw",
        "iv.jive_mw(y, endog, instruments, exog)",
        d_many_weak,
        fit_jive_mw_weak,
        design=MANY_WEAK,
    ),
    R(
        "many_weak_ar/many-weak",
        "iv.many_weak_ar",
        "iv.many_weak_ar(y, endog, instruments)",
        d_many_weak,
        cover_many_weak_ar,
        interval_only=True,
        design=MANY_WEAK,
    ),
    R(
        "anderson_rubin_ci/cluster",
        "anderson_rubin_ci",
        'anderson_rubin_ci(..., cluster="g")',
        d_weak_cluster,
        cover_ar_cluster,
        interval_only=True,
        design="100 clusters of 10, one instrument, concentration parameter about 10",
    ),
    R(
        "conditional_lr_ci/weak-1",
        "conditional_lr_ci",
        "conditional_lr_ci(y, d, [z1])",
        d_weak1,
        cover("conditional_lr_ci", 1),
        interval_only=True,
        design="one instrument, concentration parameter 4.9",
    ),
]

ROWS = ROWS + OPTION_ROWS

MECHANISMS: dict = {}

if __name__ == "__main__":
    main(__file__, FAMILY, ROWS, MECHANISMS)
