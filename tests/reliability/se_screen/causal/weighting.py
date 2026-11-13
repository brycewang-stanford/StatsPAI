"""SE screen: balancing and weighting estimators.

Design constants (fixed before the first run, 2026-11-13):

* ``N = 1000``, design ``_causal_dgp.obs`` (logit propensity, linear outcome with
  effect ``1 + 0.5 x1``). Entropy balancing, CBPS, stable balancing
  weights and overlap weights are all consistent here: the propensity is
  logit-linear and the outcome is linear in the balanced covariates.
* ``B = 400``; rows whose standard error is a bootstrap that refits a
  slow optimiser (CBPS over-identified) use fewer bootstrap draws and
  ``B = 200``, and say so.
* Truths by quadrature (``_causal_dgp.OBS_TRUTH``); transport rows have a target
  population with ``x1 ~ N(0, 1)`` so the target ATE is 1.

Run: ``PYTHONPATH=src python tests/reliability/se_screen/causal/weighting.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from _causal_dgp import OBS_TRUTH, obs, sample_truth  # noqa: E402
from _causal_harness import Row, ci_from, main  # noqa: E402

import statspai as sp  # noqa: E402

N = 1000
X = ["x1", "x2"]
T = OBS_TRUTH


def _fit(fn, target="ATE", n=N, seeded=False, **kw):
    def fit(rng):
        df = obs(rng, n)
        if seeded:
            kw["seed"] = int(rng.integers(2**31))
        out = ci_from(fn(df, "y", "d", X, **kw))
        out["truth_s"] = sample_truth(df, target)
        return out

    return fit


def _ebal_base_weights(rng):
    df = obs(rng, N)
    df["w"] = rng.uniform(0.5, 1.5, size=N)
    return ci_from(sp.ebalance(df, "y", "d", X, weights="w"))


def _sbw(target, **kw):
    def fit(rng):
        df = obs(rng, N)
        out = ci_from(sp.sbw(df, "d", X, y="y", **kw))
        out["truth_s"] = sample_truth(df, target)
        return out

    return fit


def _transport(method):
    """Experiment on a shifted population, target population N(0, I)."""

    def fit(rng):
        n, m = 800, 2000
        x1 = rng.normal(0.5, 1, n)
        x2 = rng.normal(size=n)
        d = rng.integers(0, 2, n)
        y = 1 + x1 + 0.5 * x2 + (1 + 0.5 * x1) * d + rng.normal(size=n)
        exp = pd.DataFrame({"y": y, "d": d, "x1": x1, "x2": x2})
        tgt = pd.DataFrame({"x1": rng.normal(size=m), "x2": rng.normal(size=m)})
        r = sp.pate(exp, tgt, "y", "d", X, method=method, seed=int(rng.integers(2**31)))
        return ci_from(r)

    return fit


def _front_door(rng):
    """Unobserved confounder u; D -> M -> Y with effect 0.8 * 1.5 = 1.2."""
    n = N
    u = rng.normal(size=n)
    d = (rng.uniform(size=n) < 1 / (1 + np.exp(-u))).astype(int)
    m = 0.5 + 0.8 * d + rng.normal(size=n)
    y = 1 + 1.5 * m + u + rng.normal(size=n)
    df = pd.DataFrame({"y": y, "d": d, "m": m})
    r = sp.front_door(df, "y", "d", "m", n_boot=200, seed=int(rng.integers(2**31)))
    return ci_from(r)


D = "_causal_dgp.obs"
ROWS = [
    Row(
        "ebalance.att",
        "ebalance",
        "sp.ebalance(df, y, d, X)",
        N,
        T["ATT"],
        _fit(sp.ebalance, "ATT"),
        design=D,
    ),
    Row(
        "ebalance.att.naive",
        "ebalance",
        "sp.ebalance(..., vce='naive')",
        N,
        T["ATT"],
        _fit(sp.ebalance, "ATT", vce="naive"),
        design=D,
    ),
    Row(
        "ebalance.att.moments2",
        "ebalance",
        "sp.ebalance(..., moments=2)",
        N,
        T["ATT"],
        _fit(sp.ebalance, "ATT", moments=2),
        design=D,
    ),
    Row(
        "ebalance.ate",
        "ebalance",
        "sp.ebalance(..., estimand='ATE')",
        N,
        T["ATE"],
        _fit(sp.ebalance, "ATE", estimand="ATE"),
        design=D,
    ),
    Row(
        "ebalance.atc",
        "ebalance",
        "sp.ebalance(..., estimand='ATC')",
        N,
        T["ATC"],
        _fit(sp.ebalance, "ATC", estimand="ATC"),
        design=D,
    ),
    Row(
        "ebalance.att.weights",
        "ebalance",
        "sp.ebalance(..., weights='w')",
        N,
        T["ATT"],
        _ebal_base_weights,
        design=D + " + independent sampling weights U(0.5, 1.5)",
    ),
    Row(
        "cbps.ate.over",
        "cbps",
        "sp.cbps(df, y, d, X, n_bootstrap=200)",
        500,
        T["ATE"],
        _fit(sp.cbps, n=500, seeded=True, n_bootstrap=200),
        B=200,
        design=D + " (n = 500, 200 bootstrap draws, B = 200)",
    ),
    Row(
        "cbps.ate.exact",
        "cbps",
        "sp.cbps(..., variant='exact', n_bootstrap=200)",
        N,
        T["ATE"],
        _fit(sp.cbps, seeded=True, variant="exact", n_bootstrap=200),
        design=D + " (200 bootstrap draws)",
    ),
    Row(
        "cbps.att.exact",
        "cbps",
        "sp.cbps(..., variant='exact', estimand='ATT', n_bootstrap=200)",
        N,
        T["ATT"],
        _fit(
            sp.cbps,
            "ATT",
            seeded=True,
            variant="exact",
            estimand="ATT",
            n_bootstrap=200,
        ),
        design=D + " (200 bootstrap draws)",
    ),
    Row(
        "overlap_weights.ato",
        "overlap_weights",
        "sp.overlap_weights(df, y, d, X, n_bootstrap=200)",
        N,
        T["ATO"],
        _fit(sp.overlap_weights, seeded=True, n_bootstrap=200),
        design=D,
    ),
    Row(
        "overlap_weights.att",
        "overlap_weights",
        "sp.overlap_weights(..., estimand='ATT', n_bootstrap=200)",
        N,
        T["ATT"],
        _fit(sp.overlap_weights, "ATT", seeded=True, estimand="ATT", n_bootstrap=200),
        design=D,
    ),
    Row(
        "overlap_weights.matching",
        "overlap_weights",
        "sp.overlap_weights(..., estimand='matching', n_bootstrap=200)",
        N,
        T["matching"],
        _fit(sp.overlap_weights, seeded=True, estimand="matching", n_bootstrap=200),
        design=D,
    ),
    Row(
        "overlap_weights.entropy",
        "overlap_weights",
        "sp.overlap_weights(..., estimand='entropy', n_bootstrap=200)",
        N,
        T["entropy"],
        _fit(sp.overlap_weights, seeded=True, estimand="entropy", n_bootstrap=200),
        design=D,
    ),
    Row("sbw.att", "sbw", "sp.sbw(df, d, X, y=y)", N, T["ATT"], _sbw("ATT"), design=D),
    Row(
        "sbw.ate",
        "sbw",
        "sp.sbw(..., estimand='ate')",
        N,
        T["ATE"],
        _sbw("ATE", estimand="ate"),
        design=D,
    ),
    Row(
        "sbw.atc",
        "sbw",
        "sp.sbw(..., estimand='atc')",
        N,
        T["ATC"],
        _sbw("ATC", estimand="atc"),
        design=D,
    ),
    Row(
        "sbw.att.naive",
        "sbw",
        "sp.sbw(..., vce='naive')",
        N,
        T["ATT"],
        _sbw("ATT", vce="naive"),
        design=D,
    ),
    Row(
        "sbw.att.tight",
        "sbw",
        "sp.sbw(..., delta=0.001)",
        N,
        T["ATT"],
        _sbw("ATT", delta=0.001),
        design=D,
    ),
    Row(
        "residual_balance.ate",
        "residual_balance",
        "sp.residual_balance(df, y, d, X)",
        N,
        T["ATE"],
        _fit(sp.residual_balance, "ATE"),
        design=D,
        sample_target=True,
    ),
    Row(
        "residual_balance.att",
        "residual_balance",
        "sp.residual_balance(..., estimand='ATT')",
        N,
        T["ATT"],
        _fit(sp.residual_balance, "ATT", estimand="ATT"),
        design=D,
        sample_target=True,
    ),
    Row(
        "pate.ipw",
        "pate",
        "sp.pate(experiment, target, y, d, X)",
        800,
        1.0,
        _transport("ipw"),
        design="experiment x1 ~ N(0.5, 1), n = 800; target x1 ~ N(0, 1), m = 2000",
    ),
    Row(
        "pate.aipw",
        "pate",
        "sp.pate(..., method='aipw')",
        800,
        1.0,
        _transport("aipw"),
        design="as pate.ipw",
    ),
    Row(
        "pate.calibration",
        "pate",
        "sp.pate(..., method='calibration')",
        800,
        1.0,
        _transport("calibration"),
        design="as pate.ipw",
    ),
    Row(
        "front_door",
        "front_door",
        "sp.front_door(df, y, d, m, n_boot=200)",
        N,
        1.2,
        _front_door,
        B=200,
        design="unobserved u confounds D and Y; M = 0.5 + 0.8 D + e, "
        "Y = 1 + 1.5 M + u + e (B = 200, 200 bootstrap draws)",
    ),
]

_SBW_TOLERANCE = (
    "Bias, a property of approximate balance; the standard error is in "
    "band. With delta = 0.02 the minimum-dispersion weights stop on the "
    "tolerance boundary, so each covariate keeps an imbalance of 0.02 "
    "standard deviations in the direction of the raw imbalance, and the "
    "estimate keeps that imbalance times the outcome slopes (about 0.02 "
    "to 0.03 here, a quarter to four tenths of a standard error at n = "
    "1000). Where the row is also flagged for coverage, that bias is the "
    "reason: the interval has the right width and sits off centre. The "
    "bias is gone at delta = 0.001 (sbw.att.tight) and, for a fixed "
    "delta, grows relative to the standard error with n. Before this "
    "screen the standard error was 1.5 times too large and hid it (see "
    "before_fix). The docstring of delta now states the bias."
)
MECHANISMS: dict = {
    "ebalance.att.naive": (
        "Documented option. vce='naive' takes the entropy-balancing "
        "weights as fixed and measures each arm around its weighted mean, "
        "so it counts outcome variation that exact balance removes; the "
        "docstring says so and the default vce='mestimation' is in band "
        "(ebalance.att)."
    ),
    "sbw.att.naive": (
        "Documented option since this screen. vce='naive' is the "
        "fixed-weights variance sp.sbw reported through 1.43.1; it counts "
        "outcome variation that balancing removes. The default is now the "
        "linearized variance (sbw.att, sbw.ate, sbw.atc). The bias flag is "
        "the tolerance bias described for those rows."
    ),
    "sbw.att": _SBW_TOLERANCE,
    "sbw.ate": _SBW_TOLERANCE,
    "sbw.atc": _SBW_TOLERANCE,
}

# What a row measured before a fix that this screen led to: the default
# standard error through 1.43.1 (now vce='naive').
_SBW_BEFORE = (
    "The variance took the weights as fixed and measured each arm around "
    "its weighted mean, which counts the outcome variation that balancing "
    "the covariates has removed from the estimate."
)
HISTORY = {
    "sbw.att": {
        "statspai_version": "1.43.1",
        "B": 400,
        "coverage": 0.9975,
        "se_over_sd": 1.467,
        "bias_over_sd": 0.261,
        "mechanism": _SBW_BEFORE,
    },
    "sbw.ate": {
        "statspai_version": "1.43.1",
        "B": 400,
        "coverage": 0.99,
        "se_over_sd": 1.494,
        "bias_over_sd": 0.427,
        "mechanism": _SBW_BEFORE,
    },
    "sbw.atc": {
        "statspai_version": "1.43.1",
        "B": 400,
        "coverage": 0.995,
        "se_over_sd": 1.496,
        "bias_over_sd": 0.255,
        "mechanism": _SBW_BEFORE,
    },
}

if __name__ == "__main__":
    main(__file__, ROWS, MECHANISMS, HISTORY)
