"""SE screen: causal forest and the GRF family.

Design constants (fixed before the first run, 2026-11-13):

* ``N = 1000``, design ``_dgp.obs`` (effect ``1 + 0.5 x1``), 2,000 trees
  (the default), ``n_jobs = 1``. Forest seed and data seed are both drawn
  from the row's ``SeedSequence`` child, so neighbouring replications
  share nothing.
* ``B = 400`` for rows that reuse one fitted forest's doubly robust
  scores; ``B = 200`` for the family members that take seconds per fit.
* Pointwise rows (the forest's own interval for ``tau(x)``) are a
  different claim from the averages: honest forests are biased toward
  the local average, so those rows record how far the pointwise interval
  is from nominal at this ``n`` rather than test a formula.

Run: ``PYTHONPATH=src python tests/reliability/se_screen/causal/forest.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from _dgp import OBS_TRUTH, obs, obs_clustered  # noqa: E402
from _harness import Row, main  # noqa: E402
from scipy import stats  # noqa: E402

import statspai as sp  # noqa: E402

N = 1000
ATE, ATT, ATC, ATO = (OBS_TRUTH[k] for k in ("ATE", "ATT", "ATC", "ATO"))
HALF_MEAN = stats.norm.pdf(0) / 0.5  # E[x1 | x1 > 0]
Z = stats.norm.ppf(0.975)


def _forest(rng, clustered=False):
    df = obs_clustered(rng, 100, 10) if clustered else obs(rng, N)
    kw = {"clusters": df["g"].to_numpy()} if clustered else {}
    cf = sp.causal_forest(
        data=df,
        formula="y ~ d | x1 + x2",
        random_state=int(rng.integers(2**31)),
        n_jobs=1,
        **kw,
    )
    return df, cf


def _ate(target, clustered=False):
    def fit(rng):
        df, cf = _forest(rng, clustered)
        a = sp.average_treatment_effect(cf, target_sample=target)
        return {
            "est": a["estimate"],
            "se": a["se"],
            "lo": a["ci_low"],
            "hi": a["ci_high"],
        }

    return fit


def _blp(term):
    def fit(rng):
        df, cf = _forest(rng)
        b = sp.best_linear_projection(cf, df[["x1"]]).loc[term]
        return {
            "est": b["coef"],
            "se": b["se"],
            "lo": b["ci_lower"],
            "hi": b["ci_upper"],
        }

    return fit


def _calibration(rng):
    df, cf = _forest(rng)
    c = sp.calibration_test(cf).loc["mean_forest_prediction"]
    est, se = float(c["coef"]), float(c["se"])
    return {"est": est, "se": se, "lo": est - Z * se, "hi": est + Z * se}


def _group(rng):
    df, cf = _forest(rng)
    g = sp.forest_group_effects(cf, by=(df["x1"] > 0).to_numpy()).loc[True]
    return {"est": g["estimate"], "se": g["se"], "lo": g["ci_low"], "hi": g["ci_high"]}


def _pointwise(x):
    def fit(rng):
        df, cf = _forest(rng)
        pt = np.array([x])
        est = float(cf.effect(pt)[0])
        lo, hi = cf.effect_interval(pt)
        return {
            "est": est,
            "se": (float(hi[0]) - float(lo[0])) / (2 * Z),
            "lo": float(lo[0]),
            "hi": float(hi[0]),
        }

    return fit


def _iv_forest(rng):
    """Randomised binary instrument, 60% compliers, LATE = 1."""
    n = N
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    z = rng.integers(0, 2, n)
    comp = rng.uniform(size=n) < 0.6
    d = z * comp
    y = 1 + x1 + (1 + 0.5 * x1) * d + 0.5 * comp + rng.normal(size=n)
    df = pd.DataFrame({"y": y, "d": d, "z": z, "x1": x1, "x2": x2})
    r = sp.iv_forest(
        df, "y", "d", "z", ["x1", "x2"], n_jobs=1, random_state=int(rng.integers(2**31))
    )
    a = r.average_treatment_effect()
    return _from_ate(a)


def _from_ate(a):
    if isinstance(a, pd.DataFrame):
        a = a.iloc[0]
    est, se = float(a["estimate"]), float(a["se"])
    lo = float(a["ci_low"]) if "ci_low" in a else est - Z * se
    hi = float(a["ci_high"]) if "ci_high" in a else est + Z * se
    return {"est": est, "se": se, "lo": lo, "hi": hi}


def _multi_arm(rng):
    """Three randomised arms; arm 2 against arm 0 has effect 2 + 0.5 x1."""
    n = N
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    arm = rng.integers(0, 3, n)
    y = 1 + x1 + (arm == 1) * 1.0 + (arm == 2) * (2 + 0.5 * x1) + rng.normal(size=n)
    df = pd.DataFrame({"y": y, "a": arm, "x1": x1, "x2": x2})
    r = sp.multi_arm_forest(
        df, "y", "a", ["x1", "x2"], n_jobs=1, random_state=int(rng.integers(2**31))
    )
    a = r.average_treatment_effect()
    return _from_ate(a.iloc[[-1]] if isinstance(a, pd.DataFrame) else a)


ROWS = [
    Row(
        "causal_forest.ate",
        "average_treatment_effect",
        "sp.average_treatment_effect(sp.causal_forest(...))",
        N,
        ATE,
        _ate("all"),
        design="_dgp.obs",
    ),
    Row(
        "causal_forest.att",
        "average_treatment_effect",
        "sp.average_treatment_effect(cf, target_sample='treated')",
        N,
        ATT,
        _ate("treated"),
        design="_dgp.obs",
    ),
    Row(
        "causal_forest.atc",
        "average_treatment_effect",
        "sp.average_treatment_effect(cf, target_sample='control')",
        N,
        ATC,
        _ate("control"),
        design="_dgp.obs",
    ),
    Row(
        "causal_forest.overlap",
        "average_treatment_effect",
        "sp.average_treatment_effect(cf, target_sample='overlap')",
        N,
        ATO,
        _ate("overlap"),
        design="_dgp.obs",
    ),
    Row(
        "causal_forest.ate.clusters",
        "average_treatment_effect",
        "sp.average_treatment_effect(sp.causal_forest(..., clusters=g))",
        N,
        ATE,
        _ate("all", clustered=True),
        design="_dgp.obs_clustered, 100 clusters of 10",
    ),
    Row(
        "best_linear_projection.slope",
        "best_linear_projection",
        "sp.best_linear_projection(cf, df[['x1']]).loc['x1']",
        N,
        0.5,
        _blp("x1"),
        design="_dgp.obs (the projection of 1 + 0.5 x1 on x1 is exact)",
    ),
    Row(
        "best_linear_projection.intercept",
        "best_linear_projection",
        "sp.best_linear_projection(cf, df[['x1']]).loc['Intercept']",
        N,
        1.0,
        _blp("Intercept"),
        design="_dgp.obs",
    ),
    Row(
        "calibration_test.mean",
        "calibration_test",
        "sp.calibration_test(cf).loc['mean_forest_prediction'] +/- 1.96 se",
        N,
        1.0,
        _calibration,
        design="_dgp.obs (a coefficient of 1 means the mean "
        "forest prediction is right)",
    ),
    Row(
        "forest_group_effects.by",
        "forest_group_effects",
        "sp.forest_group_effects(cf, by=x1 > 0).loc[True]",
        N,
        1 + 0.5 * HALF_MEAN,
        _group,
        design="_dgp.obs; group = rows with x1 > 0",
    ),
    Row(
        "causal_forest.pointwise.centre",
        "causal_forest",
        "cf.effect_interval([[0, 0]])",
        N,
        1.0,
        _pointwise([0.0, 0.0]),
        design="_dgp.obs; tau(0, 0) = 1",
    ),
    Row(
        "causal_forest.pointwise.off_centre",
        "causal_forest",
        "cf.effect_interval([[1, 0]])",
        N,
        1.5,
        _pointwise([1.0, 0.0]),
        design="_dgp.obs; tau(1, 0) = 1.5",
    ),
    Row(
        "iv_forest.late",
        "iv_forest",
        "sp.iv_forest(df, y, d, z, X).average_treatment_effect()",
        N,
        1.0,
        _iv_forest,
        B=200,
        design="randomised binary instrument, 60% compliers independent of x "
        "(B = 200)",
    ),
    Row(
        "multi_arm_forest.ate",
        "multi_arm_forest",
        "sp.multi_arm_forest(df, y, a, X).average_treatment_effect(), arm 2 vs 0",
        N,
        2.0,
        _multi_arm,
        B=200,
        design="three randomised arms, effect of arm 2 is 2 + 0.5 x1 (B = 200)",
    ),
]

MECHANISMS: dict = {}

if __name__ == "__main__":
    main(__file__, ROWS, MECHANISMS)
