"""SE calibration screen: multilevel (mixed-effects) models.

100 groups of 10 observations (200 groups for the binary and ordinal rows),
a random intercept with sd 0.7, and a within-group regressor ``x1`` whose
conditional (subject-specific) coefficient is 0.5.

    python tests/reliability/se_screen/multilevel.py [--only name] [--confirm]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _harness import Row, coef, main, tidy_row, wald  # noqa: E402,F401

FAMILY = "multilevel"
G, M = 100, 10
BETA = 0.5
SD_U = 0.7


def _sp():
    import statspai as sp

    return sp


def _base(rng, g=G, m=M):
    grp = np.repeat(np.arange(g), m)
    df = pd.DataFrame(
        {"g": grp, "x1": rng.normal(size=g * m), "x2": rng.normal(size=g)[grp]}
    )
    u = SD_U * rng.normal(size=g)[grp]
    return df, u


def d_gauss(rng):
    df, u = _base(rng)
    df["y"] = 1 + BETA * df.x1 + 0.3 * df.x2 + u + rng.normal(size=len(df))
    return df


def d_slope(rng):
    df, u = _base(rng)
    b = 0.4 * rng.normal(size=G)[df.g]
    df["y"] = 1 + (BETA + b) * df.x1 + 0.3 * df.x2 + u + rng.normal(size=len(df))
    return df


def d_logit(rng):
    df, u = _base(rng, 200)
    eta = -0.2 + BETA * df.x1 + 0.3 * df.x2 + u
    df["y"] = (rng.logistic(size=len(df)) < eta).astype(int)
    return df


def d_poisson(rng):
    df, u = _base(rng)
    df["y"] = rng.poisson(np.exp(0.5 + BETA * df.x1 + 0.3 * df.x2 + u))
    return df


def d_negbin(rng):
    df, u = _base(rng, 200)
    mu = np.exp(0.5 + BETA * df.x1 + 0.3 * df.x2 + u)
    df["y"] = rng.poisson(mu * rng.gamma(2.0, 0.5, size=len(df)))
    return df


def d_gamma(rng):
    df, u = _base(rng)
    mu = np.exp(0.5 + BETA * df.x1 + 0.3 * df.x2 + u)
    df["y"] = rng.gamma(shape=2.0, scale=mu / 2.0)
    return df


def d_ologit(rng):
    df, u = _base(rng, 200)
    lat = BETA * df.x1 + 0.3 * df.x2 + u + rng.logistic(size=len(df))
    df["y"] = np.digitize(lat, [-1.0, 0.0, 1.0])
    return df


def _fx(r):
    """Fixed-effect row for x1 from a mixed-model result."""
    if hasattr(r, "bse"):  # MEGLMResult
        lo, hi = np.asarray(r.conf_int().loc["x1"])[:2]
        return r.params["x1"], r.bse["x1"], lo, hi
    return coef(r, "x1")


def fit_mixed(**kw):
    def fit(df):
        return _fx(_sp().mixed(df, y="y", x_fixed=["x1", "x2"], group="g", **kw))

    return fit


def fit_me(est, **kw):
    def fit(df):
        return _fx(
            getattr(_sp(), est)(df, y="y", x_fixed=["x1", "x2"], group="g", **kw)
        )

    return fit


def R(name, estimator, call, dgp, fit, truth=BETA, n=G * M, B=None, design=""):
    return Row(name, estimator, call, n, truth, dgp, fit, B, design)


ROWS = [
    R(
        "mixed/random-intercept",
        "mixed",
        "mixed(y, x_fixed, group)",
        d_gauss,
        fit_mixed(),
    ),
    R("mixed/ml", "mixed", 'mixed(..., method="ml")', d_gauss, fit_mixed(method="ml")),
    R(
        "mixed/random-slope",
        "mixed",
        'mixed(..., x_random=["x1"])',
        d_slope,
        fit_mixed(x_random=["x1"]),
        design="random slope on x1 with sd 0.4",
    ),
    R(
        "meglm/gaussian",
        "meglm",
        'meglm(..., family="gaussian")',
        d_gauss,
        fit_me("meglm", family="gaussian"),
    ),
    R(
        "melogit/laplace",
        "melogit",
        "melogit(y, x_fixed, group)",
        d_logit,
        fit_me("melogit"),
        n=2000,
        design="200 groups of 10",
    ),
    R(
        "melogit/agq7",
        "melogit",
        "melogit(..., nAGQ=7)",
        d_logit,
        fit_me("melogit", nAGQ=7),
        n=2000,
        design="200 groups of 10",
    ),
    R(
        "mepoisson/laplace",
        "mepoisson",
        "mepoisson(y, x_fixed, group)",
        d_poisson,
        fit_me("mepoisson"),
    ),
    R(
        "menbreg/laplace",
        "menbreg",
        "menbreg(y, x_fixed, group)",
        d_negbin,
        fit_me("menbreg"),
        n=2000,
        design="200 groups of 10, overdispersion 0.5",
    ),
    R(
        "megamma/laplace",
        "megamma",
        "megamma(y, x_fixed, group)",
        d_gamma,
        fit_me("megamma"),
    ),
    R(
        "meologit/laplace",
        "meologit",
        "meologit(y, x_fixed, group)",
        d_ologit,
        fit_me("meologit"),
        n=2000,
        B=200,
        design="200 groups of 10, four categories; 3 s a fit, B = 200",
    ),
]


# ---------------------------------------------------------------- options
def d_pois_slope(rng):
    df, u = _base(rng)
    b = 0.3 * rng.normal(size=G)[df.g]
    df["y"] = rng.poisson(np.exp(0.5 + (BETA + b) * df.x1 + 0.3 * df.x2 + u))
    return df


def d_three_level(rng, g3=40, g2=5, m=5):
    """40 schools, 5 classes each, 5 pupils each; effects at both levels."""
    school = np.repeat(np.arange(g3), g2 * m)
    klass = np.repeat(np.arange(g3 * g2), m)
    n = g3 * g2 * m
    df = pd.DataFrame(
        {
            "school": school,
            "klass": klass,
            "x1": rng.normal(size=n),
            "x2": rng.normal(size=g3)[school],
        }
    )
    df["y"] = (
        1
        + BETA * df.x1
        + 0.3 * df.x2
        + 0.6 * rng.normal(size=g3)[school]
        + 0.5 * rng.normal(size=g3 * g2)[klass]
        + rng.normal(size=n)
    )
    return df


def fit_three_level(df):
    return _fx(_sp().mixed(df, y="y", x_fixed=["x1", "x2"], group=["school", "klass"]))


OPTION_ROWS = [
    R(
        "mixed/random-slope-diagonal",
        "mixed",
        'mixed(..., x_random=["x1"], cov_type="diagonal")',
        d_slope,
        fit_mixed(x_random=["x1"], cov_type="diagonal"),
        design="random slope on x1 with sd 0.4, independent of the intercept",
    ),
    R(
        "mixed/three-level",
        "mixed",
        'mixed(..., group=["school", "klass"])',
        d_three_level,
        fit_three_level,
        design="40 schools, 5 classes each, 5 pupils each",
    ),
    R(
        "mixed/between-regressor",
        "mixed",
        "mixed(y, x_fixed, group), group-level x2",
        d_gauss,
        lambda df: (lambda r: coef(r, "x2"))(
            _sp().mixed(df, y="y", x_fixed=["x1", "x2"], group="g")
        ),
        truth=0.3,
        design="coefficient on the group-level regressor, 100 groups",
    ),
    R(
        "mepoisson/agq7",
        "mepoisson",
        "mepoisson(..., nAGQ=7)",
        d_poisson,
        fit_me("mepoisson", nAGQ=7),
    ),
    R(
        "mepoisson/random-slope",
        "mepoisson",
        'mepoisson(..., x_random=["x1"])',
        d_pois_slope,
        fit_me("mepoisson", x_random=["x1"]),
        B=200,
        design="random slope on x1 with sd 0.3; 6 s a fit, B = 200",
    ),
    R(
        "meglm/binomial",
        "meglm",
        'meglm(..., family="binomial")',
        d_logit,
        fit_me("meglm", family="binomial"),
        n=2000,
        design="200 groups of 10",
    ),
    R(
        "melogit/between-regressor",
        "melogit",
        "melogit(y, x_fixed, group), group-level x2",
        d_logit,
        lambda df: (
            lambda r: (
                r.params["x2"],
                r.bse["x2"],
                *np.asarray(r.conf_int().loc["x2"])[:2],
            )
        )(_sp().melogit(df, y="y", x_fixed=["x1", "x2"], group="g")),
        truth=0.3,
        n=2000,
        design="coefficient on the group-level regressor, 200 groups",
    ),
]

ROWS = ROWS + OPTION_ROWS

MECHANISMS: dict = {}

if __name__ == "__main__":
    main(__file__, FAMILY, ROWS, MECHANISMS)
