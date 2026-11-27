"""Why does DML PLR under-cover on the Track B design?

Track B's PLR row (``run_b1000.coverage_dml_plr``: n = 500, theta = 1,
gradient-boosting nuisances, the default until 1.45) covers 0.883 with bias -0.026 and
SE/SD 0.88. ``sp.dml`` returns DoubleML's numbers to 1e-16 on the same folds
and learners, so the question is about the procedure, not the code. This
script varies one thing at a time on the same DGP (B draws each):

* the nuisance learner: the default gradient boosting, a random forest,
  cross-validated Lasso on a spline basis, and the *oracle* nuisances
  (the true E[Y|X] and E[D|X]) -- which removes nuisance error entirely;
* the sample size: n = 500 and n = 2,000 with the default learner.

If coverage recovers with the oracle and improves with n or a better-suited
learner, the mechanism is first-stage (regularisation) error in the
nuisances, which DML's orthogonality makes second order but not zero in
finite samples. Writes ``results_dml_plr_learners.json``.
"""

from __future__ import annotations

import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LassoCV
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import SplineTransformer

import statspai as sp

HERE = Path(__file__).resolve().parent
TRUTH = 1.0
B = int(sys.argv[1]) if len(sys.argv) > 1 else 500


class _Oracle:
    """A 'learner' that returns the true conditional mean (for the benchmark)."""

    def __init__(self, fn):
        self.fn = fn

    def fit(self, X, y):
        return self

    def predict(self, X):
        X = np.asarray(X)
        return self.fn(X[:, 0], X[:, 1])

    def get_params(self, deep=True):
        return {"fn": self.fn}

    def set_params(self, **p):
        self.fn = p.get("fn", self.fn)
        return self


def _e_y(x1, x2):  # E[Y|X] = theta*E[D|X] + x1 + 0.5 x2^2
    return TRUTH * (0.5 * x1 + 0.3 * np.sin(2 * x2)) + x1 + 0.5 * x2**2


def _e_d(x1, x2):
    return 0.5 * x1 + 0.3 * np.sin(2 * x2)


LEARNERS = {
    "gbm_default": lambda: None,
    "random_forest": lambda: RandomForestRegressor(n_estimators=300, min_samples_leaf=5, random_state=0),
    "lasso_splines": lambda: make_pipeline(SplineTransformer(n_knots=8, degree=3), LassoCV(cv=5)),
}


def run(n: int, learner: str) -> dict:
    est, se, cov = [], [], 0
    t0 = time.time()
    for seed in range(B):
        rng = np.random.default_rng(seed)
        x1, x2 = rng.normal(size=n), rng.normal(size=n)
        d = _e_d(x1, x2) + rng.normal(size=n)
        y = TRUTH * d + x1 + 0.5 * x2**2 + rng.normal(size=n)
        df = pd.DataFrame({"y": y, "d": d, "x1": x1, "x2": x2})
        kw = {}
        if learner == "oracle":
            kw = {"model_y": _Oracle(_e_y), "model_d": _Oracle(_e_d)}
        elif learner == "gbm_default":
            # Gradient boosting was the PLR default until 1.45; naming it
            # keeps these rows on the learner they were run with.
            kw = {"ml_g": "gbm", "ml_m": "gbm"}
        elif LEARNERS[learner]() is not None:
            kw = {"model_y": LEARNERS[learner](), "model_d": LEARNERS[learner]()}
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            r = sp.dml(df, y="y", d="d", X=["x1", "x2"], model="plr", n_folds=5,
                       random_state=seed, **kw)
        est.append(r.estimate)
        se.append(r.se)
        cov += r.ci[0] <= TRUTH <= r.ci[1]
    est, se = np.asarray(est), np.asarray(se)
    return {
        "n": n, "learner": learner, "B": B, "coverage": cov / B,
        "bias": float(est.mean() - TRUTH), "mc_sd": float(est.std(ddof=1)),
        "mean_se": float(se.mean()), "se_sd_ratio": float(se.mean() / est.std(ddof=1)),
        "bias_over_sd": float((est.mean() - TRUTH) / est.std(ddof=1)),
        "wall_s": round(time.time() - t0, 1),
    }


def main() -> None:
    rows = [run(500, "gbm_default"), run(500, "oracle"), run(500, "random_forest"),
            run(500, "lasso_splines"), run(2000, "gbm_default")]
    for r in rows:
        print(r, flush=True)
    (HERE / "results_dml_plr_learners.json").write_text(json.dumps(rows, indent=2) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
