"""Which learner should replace the ``sp.dml(model='plr')`` default?

1.44.0 announced that the gradient-boosting default (scikit-learn's
``GradientBoostingRegressor``, 100 trees, depth 3) will be replaced no
earlier than 1.45. This study picks the replacement. Everything below was
fixed before the first run.

Candidates (each is used for both nuisances, five cross-fitting folds,
``random_state=42`` wherever a learner takes one; core scikit-learn only,
so the default needs no optional dependency):

* ``gbm``: today's default, the baseline;
* ``gbm_es``: ``GradientBoostingRegressor(n_estimators=1000,
  learning_rate=0.05, max_depth=3, validation_fraction=0.2,
  n_iter_no_change=20)``, the same model with the number of trees chosen
  by early stopping on an inner validation split instead of fixed at 100;
* ``hgb_es``: ``HistGradientBoostingRegressor(max_iter=1000,
  learning_rate=0.05, early_stopping=True, validation_fraction=0.2,
  n_iter_no_change=20)``, the histogram variant with the same stopping;
* ``lasso_splines``: cubic B-splines with five knots per covariate, then
  ``LassoCV(cv=5)``, an additive model;
* ``stack``: ``StackingRegressor`` of ``LassoCV(cv=5)`` and ``gbm_es``
  with a non-negative least-squares combiner (``cv=3``), so a linear fit
  and a flexible fit are weighed inside each fold.

Designs (``B`` replications per cell, one seed per replication):

* the ten-covariate designs of ``dml_learners.py`` with ``theta = 0.5``,
  ``n`` in 500 and 2000, under three shapes of confounding:

  - ``linear``: ``m = 0.5 x0 + 0.5 x1``, ``g = x0 + 0.5 x2``;
  - ``nonlinear``: ``m = sin(x0) + 0.5 x1^2``,
    ``g = 2 sin(x0) + x1^2 + 0.5 x2`` (additive);
  - ``interaction``: ``m = 0.5 x0 + 0.5 x1 x2``,
    ``g = x0 + x1 x2 + 0.5 x3``, which no additive learner can fit;

* the Track B design of
  ``tests/coverage_monte_carlo/mechanisms/dml_plr_learners.py`` at
  ``n = 500``: two covariates, ``theta = 1``, ``m = 0.5 x1 + 0.3 sin(2 x2)``,
  ``g = x1 + 0.5 x2^2``.

Decision rule:

1. The score of a candidate is its lowest coverage over the seven cells.
2. The replacement is the candidate with the highest score, provided it
   beats ``gbm`` by more than 0.025 (two Monte Carlo standard errors at
   ``B = 300``). If none does, the default stays and the announcement is
   withdrawn.
3. Candidates whose scores are within 0.02 of the best are ranked by mean
   ``|bias| / sd`` over the cells, then by run time.

Added after the first run, and what was decided:

* ``short_stacking``: ``sp.dml_model_averaging`` at its defaults (short
  stacking over Lasso, Ridge, random forest and gradient boosting). It
  was left out of the first run by oversight, although it is the
  alternative ``dml_learners.py`` and the JSS manuscript's six-design
  table (``Paper-JSS/replication/scripts/dml_plr_designs.py``) measured.
* The rule above does not separate the candidates: in the interaction
  cells every learner covers under 0.04, so every score is near 0 and no
  candidate beats ``gbm`` by 0.025. Without those cells the best of the
  first five, ``lasso_splines`` (0.880 against 0.863), is still inside
  the margin.
* Here short stacking covers better than ``gbm`` with linear confounding
  (0.943 and 0.950 against 0.907 and 0.923) and on the Track B design
  (0.960 against 0.887), worse with smooth additive confounding (0.720
  and 0.860 against 0.863 and 0.893), not at all with an interaction, and
  takes six to seven times as long to fit.
* The owner decided on 2026-11-17 to replace the default with short
  stacking, on the cells of all three studies where some learner covers
  (the manuscript's six designs, the four of ``dml_learners.py`` and the
  five non-interaction cells here): short stacking covers better on
  eleven of the fifteen, and its worst coverage there is 0.720 against
  0.540. The four it loses are the smooth additive designs.

Run: ``python tests/reliability/dml_default_candidates.py [B] [workers]
[method]`` (chunks of replications run in parallel; about an hour at
``B = 300`` on eight cores for the first five; naming a method reruns its
cells and keeps the rest of the stored file). Writes
``dml_default_candidates_results.json`` next to this file;
``tests/test_reliability_dml_default_candidates.py`` recomputes the first
25 replications of two cells.
"""

from __future__ import annotations

import json
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Any, Callable, Dict, List, Tuple

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np  # noqa: E402  (thread caps above must come first)
import pandas as pd  # noqa: E402

HERE = Path(__file__).resolve().parent
OUT = HERE / "dml_default_candidates_results.json"
CHUNK = 25
P = 10
CELLS: Tuple[Tuple[str, int], ...] = (
    ("linear", 500),
    ("linear", 2000),
    ("nonlinear", 500),
    ("nonlinear", 2000),
    ("interaction", 500),
    ("interaction", 2000),
    ("trackb", 500),
)


def _gbm() -> Any:
    from sklearn.ensemble import GradientBoostingRegressor

    return GradientBoostingRegressor(
        n_estimators=100, max_depth=3, learning_rate=0.1, random_state=42
    )


def _gbm_es() -> Any:
    from sklearn.ensemble import GradientBoostingRegressor

    return GradientBoostingRegressor(
        n_estimators=1000,
        learning_rate=0.05,
        max_depth=3,
        validation_fraction=0.2,
        n_iter_no_change=20,
        random_state=42,
    )


def _hgb_es() -> Any:
    from sklearn.ensemble import HistGradientBoostingRegressor

    return HistGradientBoostingRegressor(
        max_iter=1000,
        learning_rate=0.05,
        early_stopping=True,
        validation_fraction=0.2,
        n_iter_no_change=20,
        random_state=42,
    )


def _lasso_splines() -> Any:
    from sklearn.linear_model import LassoCV
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import SplineTransformer

    return make_pipeline(
        SplineTransformer(n_knots=5, degree=3), LassoCV(cv=5, random_state=42)
    )


def _stack() -> Any:
    from sklearn.ensemble import StackingRegressor
    from sklearn.linear_model import LassoCV, LinearRegression

    return StackingRegressor(
        estimators=[("lasso", LassoCV(cv=5, random_state=42)), ("gbm", _gbm_es())],
        final_estimator=LinearRegression(positive=True),
        cv=3,
    )


LEARNERS: Dict[str, Callable[[], Any]] = {
    "gbm": _gbm,
    "gbm_es": _gbm_es,
    "hgb_es": _hgb_es,
    "lasso_splines": _lasso_splines,
    "stack": _stack,
}
METHODS = tuple(LEARNERS) + ("short_stacking",)


def truth(shape: str) -> float:
    return 1.0 if shape == "trackb" else 0.5


def draw(shape: str, n: int, seed: int) -> Tuple[pd.DataFrame, List[str]]:
    rng = np.random.default_rng(seed)
    if shape == "trackb":
        x1, x2 = rng.normal(size=n), rng.normal(size=n)
        d = 0.5 * x1 + 0.3 * np.sin(2 * x2) + rng.normal(size=n)
        y = 1.0 * d + x1 + 0.5 * x2**2 + rng.normal(size=n)
        return pd.DataFrame({"y": y, "d": d, "x1": x1, "x2": x2}), ["x1", "x2"]
    X = rng.normal(size=(n, P))
    if shape == "linear":
        m = 0.5 * X[:, 0] + 0.5 * X[:, 1]
        g = X[:, 0] + 0.5 * X[:, 2]
    elif shape == "nonlinear":
        m = np.sin(X[:, 0]) + 0.5 * X[:, 1] ** 2
        g = 2.0 * np.sin(X[:, 0]) + X[:, 1] ** 2 + 0.5 * X[:, 2]
    elif shape == "interaction":
        m = 0.5 * X[:, 0] + 0.5 * X[:, 1] * X[:, 2]
        g = X[:, 0] + X[:, 1] * X[:, 2] + 0.5 * X[:, 3]
    else:
        raise ValueError(shape)
    d = m + rng.normal(size=n)
    y = 0.5 * d + g + rng.normal(size=n)
    cols = [f"x{j}" for j in range(P)]
    df = pd.DataFrame(X, columns=cols)
    df["d"] = d
    df["y"] = y
    return df, cols


def cell_seed(shape: str, n: int, rep: int) -> int:
    shapes = ("linear", "nonlinear", "interaction", "trackb")
    return 7_000_003 * n + 1_019 * shapes.index(shape) + 104_729 * rep


def _chunk(args: Tuple[str, int, str, int, int]) -> Dict[str, Any]:
    import statspai as sp

    shape, n, learner, start, stop = args
    out: Dict[str, Any] = {"est": [], "se": [], "hit": [], "wall": 0.0}
    t0 = time.perf_counter()
    for rep in range(start, stop):
        seed = cell_seed(shape, n, rep)
        df, cols = draw(shape, n, seed)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            if learner == "short_stacking":
                r = sp.dml_model_averaging(
                    df, y="y", treat="d", covariates=cols, seed=seed % (2**31 - 1)
                )
            else:
                r = sp.dml(
                    df,
                    y="y",
                    treat="d",
                    covariates=cols,
                    model="plr",
                    ml_g=LEARNERS[learner](),
                    ml_m=LEARNERS[learner](),
                    random_state=seed % (2**31 - 1),
                )
        lo, hi = float(r.ci[0]), float(r.ci[1])
        out["est"].append(float(r.estimate))
        out["se"].append(float(r.se))
        out["hit"].append(int(lo <= truth(shape) <= hi))
    out["wall"] = time.perf_counter() - t0
    return out


def main() -> None:
    import sklearn

    import statspai as sp

    B = int(sys.argv[1]) if len(sys.argv) > 1 else 300
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 8
    only = sys.argv[3] if len(sys.argv) > 3 else None
    if only is not None and only not in METHODS:
        raise SystemExit(f"unknown method {only!r}; one of {METHODS}")
    keys = [(s, n, k) for s, n in CELLS for k in METHODS if only in (None, k)]
    jobs = [
        (s, n, k, a, min(a + CHUNK, B)) for s, n, k in keys for a in range(0, B, CHUNK)
    ]
    # Longest jobs first so the pool does not idle on a straggler.
    order = sorted(range(len(jobs)), key=lambda i: -jobs[i][1])
    with ProcessPoolExecutor(max_workers=workers) as pool:
        done = dict(zip(order, pool.map(_chunk, [jobs[i] for i in order])))
    cells: List[Dict[str, Any]] = []
    for s, n, k in keys:
        parts = [done[i] for i, j in enumerate(jobs) if j[:3] == (s, n, k)]
        est = np.concatenate([p["est"] for p in parts])
        se = np.concatenate([p["se"] for p in parts])
        hit = np.concatenate([p["hit"] for p in parts])
        sd = float(np.std(est, ddof=1))
        rate = float(hit.mean())
        cells.append(
            {
                "shape": s,
                "n": n,
                "learner": k,
                "B": int(len(est)),
                "bias": float(est.mean() - truth(s)),
                "sd": sd,
                "mean_se": float(se.mean()),
                "coverage": rate,
                "mc_se": float(np.sqrt(rate * (1 - rate) / len(est))),
                "prefix_hits": int(hit[:CHUNK].sum()),
                "seconds_per_fit": float(sum(p["wall"] for p in parts) / len(est)),
            }
        )
        c = cells[-1]
        print(
            f"{s:11s} n={n:4d} {k:13s} bias={c['bias']:+.3f} sd={c['sd']:.3f} "
            f"se={c['mean_se']:.3f} cov={c['coverage']:.3f} "
            f"t={c['seconds_per_fit']:.2f}s",
            flush=True,
        )
    if only is not None:
        stored = json.loads(OUT.read_text(encoding="utf-8"))
        if stored["B"] != B:
            raise SystemExit(f"stored file has B={stored['B']}, not {B}")
        rank = {
            key: i for i, key in enumerate((s, n, k) for s, n in CELLS for k in METHODS)
        }
        cells = [c for c in stored["cells"] if c["learner"] != only] + cells
        cells.sort(key=lambda c: rank[(c["shape"], c["n"], c["learner"])])
    payload = {
        "study": "dml_default_candidates",
        "level": 0.95,
        "B": B,
        "statspai_version": sp.__version__,
        "sklearn_version": sklearn.__version__,
        "cells": cells,
    }
    OUT.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
