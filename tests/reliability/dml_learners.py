"""Coverage of the DML partially linear estimate as the learner changes.

Question, fixed before the first run: at a nominal 95% level, how often
does the interval of ``sp.dml(model='plr')`` cover the treatment
coefficient, as the first-stage learner and the shape of the nuisance
functions change?

Design (20 cells, ``B`` replications each, one seed per replication):

* ``n`` in 500, 2000; ten independent standard-normal covariates;
* ``d = m(x) + v`` and ``y = 0.5 d + g(x) + u`` with ``u``, ``v``
  independent standard normal, under two shapes:

  - ``linear``: ``m = 0.5 x0 + 0.5 x1``, ``g = x0 + 0.5 x2``;
  - ``nonlinear``: ``m = sin(x0) + 0.5 x1^2``,
    ``g = 2 sin(x0) + x1^2 + 0.5 x2``, so the part of the confounding a
    linear learner misses is shared by treatment and outcome;

* four learners, the same for both nuisance functions, five folds:

  - ``ols``: ``LinearRegression``;
  - ``lasso``: ``LassoCV(cv=3)``;
  - ``rf``: ``RandomForestRegressor(100 trees, min_samples_leaf=5)``;
  - ``gbm``: ``GradientBoostingRegressor()`` at scikit-learn's defaults,
    which is what ``sp.dml`` uses when no learner is passed;

* and ``stacking``: ``sp.dml_model_averaging`` at its defaults (short
  stacking over lasso, ridge, random forest and gradient boosting), the
  alternative the ``sp.dml`` result points to.

Reported per cell: mean bias, the standard deviation of the estimates,
the mean reported standard error, coverage of the 95% interval and its
Monte Carlo standard error. With ``B = 300`` a correct interval lands in
0.950 +/- 0.025.

Run: ``python tests/reliability/dml_learners.py [B] [workers] [learner]``
(cells run in parallel; about half an hour at B = 300 for the four
learners and an hour for stacking; naming a learner reruns its cells
and keeps the rest of the stored file). Writes
``dml_learners_results.json`` next to this file.
``tests/test_reliability_dml_learners.py`` recomputes one cell on its
first 30 replications.
"""

from __future__ import annotations

import json
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "dml_learners_results.json"
TRUTH = 0.5
PREFIX = 30
P = 10
N_VALUES = (500, 2000)
SHAPES = ("linear", "nonlinear")
COVARIATES = [f"x{j}" for j in range(P)]


def _ols(seed: int) -> Any:
    from sklearn.linear_model import LinearRegression

    return LinearRegression()


def _lasso(seed: int) -> Any:
    from sklearn.linear_model import LassoCV

    return LassoCV(cv=3, random_state=seed)


def _rf(seed: int) -> Any:
    from sklearn.ensemble import RandomForestRegressor

    return RandomForestRegressor(
        n_estimators=100, min_samples_leaf=5, random_state=seed
    )


def _gbm(seed: int) -> Any:
    from sklearn.ensemble import GradientBoostingRegressor

    return GradientBoostingRegressor(random_state=seed)


LEARNERS: Dict[str, Callable[[int], Any]] = {
    "ols": _ols,
    "lasso": _lasso,
    "rf": _rf,
    "gbm": _gbm,
}
METHODS = tuple(LEARNERS) + ("stacking",)


def draw(n: int, shape: str, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(n, P))
    if shape == "linear":
        m = 0.5 * X[:, 0] + 0.5 * X[:, 1]
        g = X[:, 0] + 0.5 * X[:, 2]
    elif shape == "nonlinear":
        m = np.sin(X[:, 0]) + 0.5 * X[:, 1] ** 2
        g = 2.0 * np.sin(X[:, 0]) + X[:, 1] ** 2 + 0.5 * X[:, 2]
    else:
        raise ValueError(shape)
    d = m + rng.normal(size=n)
    y = TRUTH * d + g + rng.normal(size=n)
    df = pd.DataFrame(X, columns=COVARIATES)
    df["d"] = d
    df["y"] = y
    return df


def cell_seed(n: int, shape: str, rep: int) -> int:
    return 2_000_003 * n + 1_019 * SHAPES.index(shape) + 104_729 * rep


def fit(df: pd.DataFrame, learner: str, seed: int) -> Tuple[float, float, float, float]:
    import statspai as sp

    if learner == "stacking":
        avg = sp.dml_model_averaging(
            df, y="y", treat="d", covariates=COVARIATES, seed=seed
        )
        return (
            float(avg.estimate),
            float(avg.se),
            float(avg.ci[0]),
            float(avg.ci[1]),
        )
    make = LEARNERS[learner]
    res = sp.dml(
        df,
        y="y",
        treat="d",
        covariates=COVARIATES,
        model="plr",
        ml_g=make(seed),
        ml_m=make(seed),
        random_state=seed,
    )
    return float(res.estimate), float(res.se), float(res.ci[0]), float(res.ci[1])


def run_cell(n: int, shape: str, learner: str, B: int) -> Dict[str, object]:
    est: List[float] = []
    ses: List[float] = []
    hits: List[int] = []
    for rep in range(B):
        seed = cell_seed(n, shape, rep)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            b, se, lo, hi = fit(draw(n, shape, seed), learner, seed % (2**31 - 1))
        est.append(b)
        ses.append(se)
        hits.append(int(lo <= TRUTH <= hi))
    h = np.asarray(hits)
    rate = float(h.mean())
    return {
        "n": n,
        "shape": shape,
        "learner": learner,
        "B": B,
        "bias": float(np.mean(est) - TRUTH),
        "sd": float(np.std(est, ddof=1)) if B > 1 else float("nan"),
        "mean_se": float(np.mean(ses)),
        "coverage": rate,
        "mc_se": float(np.sqrt(rate * (1 - rate) / B)),
        "prefix_hits": int(h[:PREFIX].sum()),
    }


def _job(args: Tuple[int, str, str, int]) -> Dict[str, object]:
    return run_cell(*args)


def main() -> None:
    import statspai as sp

    B = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    only = sys.argv[3] if len(sys.argv) > 3 else None
    jobs = [
        (n, s, k, B)
        for s in SHAPES
        for n in N_VALUES
        for k in METHODS
        if only in (None, k)
    ]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        cells = list(pool.map(_job, jobs))
    if only is not None:
        stored = json.loads(OUT.read_text(encoding="utf-8"))
        if stored["B"] != B:
            raise SystemExit(f"stored file has B={stored['B']}, not {B}")
        cells = [c for c in stored["cells"] if c["learner"] != only] + cells
        order = {
            (s, n, k): i
            for i, (n, s, k, _) in enumerate(
                (n, s, k, B) for s in SHAPES for n in N_VALUES for k in METHODS
            )
        }
        cells.sort(key=lambda c: order[(c["shape"], c["n"], c["learner"])])
    for c in cells:
        print(
            f"{c['shape']:9s} n={c['n']:4d} {c['learner']:5s} "
            f"bias={c['bias']:+.3f} sd={c['sd']:.3f} se={c['mean_se']:.3f} "
            f"cov={c['coverage']:.3f}",
            flush=True,
        )
    payload = {
        "study": "dml_learners",
        "level": 0.95,
        "truth": TRUTH,
        "B": B,
        "prefix": PREFIX,
        "statspai_version": sp.__version__,
        "cells": cells,
    }
    OUT.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
