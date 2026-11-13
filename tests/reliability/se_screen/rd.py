"""SE calibration screen: regression discontinuity beyond sharp ``rdrobust``.

n = 2,000, running variable uniform on [-1, 1], cutoff 0, conditional mean
0.5 x + 0.25 x^2 on both sides, error sd 0.5, constant effect 1.0. Sharp
``rdrobust`` with defaults is a Track B row and mass points are a separate
study (``tests/reliability/rd_mass_points.py``); neither is repeated.

    python tests/reliability/se_screen/rd.py [--only name] [--confirm]
"""

# flake8: noqa: E501
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from _harness import Row, coef, main, tidy_row, wald  # noqa: E402,F401

FAMILY = "rd"
N = 2000
TAU = 1.0
KINK = 1.0
W = 0.15  # half-width of the local-randomization window


def _sp():
    import statspai as sp

    return sp


def _m(x):
    return 0.5 * x + 0.25 * x**2


def d_sharp(rng, n=N):
    x = rng.uniform(-1, 1, size=n)
    df = pd.DataFrame({"x": x, "z": rng.normal(size=n)})
    df["d"] = (x >= 0).astype(int)
    df["y"] = _m(x) + TAU * df.d + 0.5 * df.z + 0.5 * rng.normal(size=n)
    return df


def d_fuzzy(rng, n=4000):
    """Take-up jumps from 0.2 to 0.8 at the cutoff; constant effect of take-up."""
    x = rng.uniform(-1, 1, size=n)
    df = pd.DataFrame({"x": x})
    df["d"] = (rng.uniform(size=n) < np.where(x >= 0, 0.8, 0.2)).astype(int)
    df["y"] = _m(x) + TAU * df.d + 0.5 * rng.normal(size=n)
    return df


def d_cluster(rng, n=N, g=200):
    cl = rng.integers(0, g, size=n)
    x = rng.uniform(-1, 1, size=n)
    df = pd.DataFrame({"x": x, "g": cl})
    df["y"] = (
        _m(x) + TAU * (x >= 0) + 0.4 * rng.normal(size=g)[cl] + 0.4 * rng.normal(size=n)
    )
    return df


def d_kink(rng, n=4000):
    """Continuous at the cutoff; the slope rises by 1.0 there."""
    x = rng.uniform(-1, 1, size=n)
    df = pd.DataFrame({"x": x})
    df["y"] = _m(x) + KINK * np.maximum(x, 0.0) + 0.3 * rng.normal(size=n)
    return df


def d_locrand(rng, n=N, tau=TAU):
    """Inside |x| < 0.15 the outcome does not depend on the score."""
    x = rng.uniform(-1, 1, size=n)
    df = pd.DataFrame({"x": x})
    df["y"] = (
        np.where(np.abs(x) < W, 0.0, _m(x)) + tau * (x >= 0) + 0.5 * rng.normal(size=n)
    )
    return df


def d_locrand_null(rng):
    return d_locrand(rng, tau=0.0)


def d_multi(rng, n=3000):
    """Two cutoffs (-0.4 and 0.4), each unit facing one; effect 1.0 at both."""
    x = rng.uniform(-1, 1, size=n)
    c = np.where(rng.uniform(size=n) < 0.5, -0.4, 0.4)
    df = pd.DataFrame({"x": x, "c": c})
    df["y"] = _m(x) + TAU * (x >= c) + 0.5 * rng.normal(size=n)
    return df


# ------------------------------------------------------------------- fits
def fit_rdrobust(**kw):
    def fit(df):
        return coef(_sp().rdrobust(df, y="y", x="x", c=0, **kw))

    return fit


def fit_rkd(df):
    return coef(_sp().rkd(df, y="y", x="x", c=0))


def fit_honest(df):
    return coef(_sp().rd_honest(df, y="y", x="x", c=0.0, M=0.5))


def fit_randinf(df):
    r = _sp().rdrandinf(df, y="y", x="x", c=0, wl=-W, wr=W, n_perms=500, seed=1)
    lo, hi = r.ci
    return lo <= TAU <= hi


def fit_randinf_null(df):
    r = _sp().rdrandinf(df, y="y", x="x", c=0, wl=-W, wr=W, n_perms=500, seed=1)
    return r.pvalue > 0.05


def fit_rdmc(df):
    r = _sp().rdmc(df, y="y", x="x", cutoffs=[-0.4, 0.4], cutoff_var="c")
    # rdmc reports the conventional pooled estimate next to the robust SE and
    # the robust interval, which is centred on the bias-corrected estimate.
    # The SE belongs to the bias-corrected estimate, so that is the pair here.
    return r.pooled_estimate_robust, r.pooled_se, r.pooled_ci[0], r.pooled_ci[1]


def R(
    name,
    estimator,
    call,
    dgp,
    fit,
    truth=TAU,
    n=N,
    B=None,
    design="",
    interval_only=False,
):
    return Row(name, estimator, call, n, truth, dgp, fit, B, design, interval_only)


ROWS = [
    R(
        "rdrobust/fuzzy",
        "rdrobust",
        'rdrobust(fuzzy="d")',
        d_fuzzy,
        fit_rdrobust(fuzzy="d"),
        n=4000,
        design="take-up jumps 0.2 to 0.8",
    ),
    R(
        "rdrobust/covs",
        "rdrobust",
        'rdrobust(covs=["z"])',
        d_sharp,
        fit_rdrobust(covs=["z"]),
        design="one covariate with coefficient 0.5",
    ),
    R(
        "rdrobust/cluster",
        "rdrobust",
        'rdrobust(cluster="g")',
        d_cluster,
        fit_rdrobust(cluster="g"),
        design="200 clusters crossing the cutoff",
    ),
    R(
        "rdrobust/kink",
        "rdrobust",
        "rdrobust(deriv=1)",
        d_kink,
        fit_rdrobust(deriv=1),
        truth=KINK,
        n=4000,
        design="slope change of 1.0, no jump",
    ),
    R(
        "rdrobust/uniform-cerrd",
        "rdrobust",
        'rdrobust(kernel="uniform", bwselect="cerrd")',
        d_sharp,
        fit_rdrobust(kernel="uniform", bwselect="cerrd"),
    ),
    R(
        "rdrobust/hc3",
        "rdrobust",
        'rdrobust(vce="hc3")',
        d_sharp,
        fit_rdrobust(vce="hc3"),
    ),
    R("rdrobust/p2", "rdrobust", "rdrobust(p=2)", d_sharp, fit_rdrobust(p=2)),
    R(
        "rkd/default",
        "rkd",
        "rkd(y, x)",
        d_kink,
        fit_rkd,
        truth=KINK,
        n=4000,
        design="slope change of 1.0, no jump",
    ),
    R(
        "rd_honest/M-known",
        "rd_honest",
        "rd_honest(M=0.5)",
        d_sharp,
        fit_honest,
        design="true second derivative 0.5 equals the bound M",
    ),
    R(
        "rdrandinf/ci",
        "rdrandinf",
        "rdrandinf(wl=-0.15, wr=0.15), test-inversion interval",
        d_locrand,
        fit_randinf,
        interval_only=True,
        design="outcome flat in the score inside the window",
    ),
    R(
        "rdrandinf/size",
        "rdrandinf",
        "rdrandinf(wl=-0.15, wr=0.15), p > 0.05 under no effect",
        d_locrand_null,
        fit_randinf_null,
        truth=0.0,
        interval_only=True,
        design="no effect; share of draws not rejecting at 5%",
    ),
    R(
        "rdmc/pooled",
        "rdmc",
        "rdmc(cutoffs=[-0.4, 0.4])",
        d_multi,
        fit_rdmc,
        n=3000,
        design="two cutoffs, effect 1.0 at both",
    ),
]


# ---------------------------------------------------------------- options
def d_fuzzy_covs(rng):
    df = d_fuzzy(rng)
    df["z"] = rng.normal(size=len(df))
    df["y"] = df.y + 0.5 * df.z
    return df


def fit_honest_kw(**kw):
    def fit(df):
        return coef(_sp().rd_honest(df, y="y", x="x", c=0.0, M=0.5, **kw))

    return fit


OPTION_ROWS = [
    R(
        "rdrobust/hc0",
        "rdrobust",
        'rdrobust(vce="hc0")',
        d_sharp,
        fit_rdrobust(vce="hc0"),
    ),
    R(
        "rdrobust/hc1",
        "rdrobust",
        'rdrobust(vce="hc1")',
        d_sharp,
        fit_rdrobust(vce="hc1"),
    ),
    R(
        "rdrobust/hc2",
        "rdrobust",
        'rdrobust(vce="hc2")',
        d_sharp,
        fit_rdrobust(vce="hc2"),
    ),
    R(
        "rdrobust/msetwo",
        "rdrobust",
        'rdrobust(bwselect="msetwo")',
        d_sharp,
        fit_rdrobust(bwselect="msetwo"),
    ),
    R(
        "rdrobust/certwo",
        "rdrobust",
        'rdrobust(bwselect="certwo")',
        d_sharp,
        fit_rdrobust(bwselect="certwo"),
    ),
    R(
        "rdrobust/msecomb2",
        "rdrobust",
        'rdrobust(bwselect="msecomb2")',
        d_sharp,
        fit_rdrobust(bwselect="msecomb2"),
    ),
    R(
        "rdrobust/epanechnikov",
        "rdrobust",
        'rdrobust(kernel="epanechnikov")',
        d_sharp,
        fit_rdrobust(kernel="epanechnikov"),
    ),
    R("rdrobust/manual-h", "rdrobust", "rdrobust(h=0.3)", d_sharp, fit_rdrobust(h=0.3)),
    R(
        "rdrobust/fuzzy-covs",
        "rdrobust",
        'rdrobust(fuzzy="d", covs=["z"])',
        d_fuzzy_covs,
        fit_rdrobust(fuzzy="d", covs=["z"]),
        n=4000,
        design="take-up jumps 0.2 to 0.8",
    ),
    R(
        "rdrobust/cluster-cr2",
        "rdrobust",
        'rdrobust(cluster="g", vce="cr2")',
        d_cluster,
        fit_rdrobust(cluster="g", vce="cr2"),
        design="200 clusters crossing the cutoff",
    ),
    R(
        "rdrobust/donut",
        "rdrobust",
        "rdrobust(donut=0.02)",
        d_sharp,
        fit_rdrobust(donut=0.02),
    ),
    R(
        "rd_honest/taylor",
        "rd_honest",
        'rd_honest(M=0.5, sclass="T")',
        d_sharp,
        fit_honest_kw(sclass="T"),
    ),
    R(
        "rd_honest/flci",
        "rd_honest",
        'rd_honest(M=0.5, opt_criterion="flci")',
        d_sharp,
        fit_honest_kw(opt_criterion="flci"),
    ),
    R(
        "rd_honest/uniform",
        "rd_honest",
        'rd_honest(M=0.5, kernel="uniform")',
        d_sharp,
        fit_honest_kw(kernel="uniform"),
    ),
]

ROWS = ROWS + OPTION_ROWS

MECHANISMS: dict = {
    "rkd/default": (
        "Property of the conventional estimate at an MSE-optimal bandwidth. sp.rkd reports "
        "the conventional kink estimate and its SE as the headline; that bandwidth leaves a "
        "smoothing bias of the same order as the SE (bias/SD 0.23), and the data-driven "
        "bandwidth gives the estimate heavier tails than its SE describes (median SE over "
        "MAD-based spread 1.06). The robust bias-corrected interval, which sp.rkd puts in "
        "model_info['robust'] and sp.rdrobust(deriv=1) reports as its headline (row "
        "rdrobust/kink), is inside the band. Owner decision: make it the headline of sp.rkd."
    ),
}

if __name__ == "__main__":
    main(__file__, FAMILY, ROWS, MECHANISMS)
