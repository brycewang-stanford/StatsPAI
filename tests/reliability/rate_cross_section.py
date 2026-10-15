"""Size of ``sp.rate`` on a cross-sectional forest's own ranking.

Question, fixed before the run: with no treatment-effect heterogeneity,
how often does ``sp.rate(cf)`` -- which ranks units by the forest's own
out-of-bag CATE predictions and scores them with the same forest's
doubly-robust scores -- reject AUTOC = 0 at a nominal 5% level, and how
does ``sp.rate_split`` (rule fitted on one half, scored on the other) do
on the same data?

Design: ``n = 800``, four standard-normal covariates, ``T ~ Bernoulli(0.5)``,
``y = x0 + 0.5 x1 + 1.0 T + e``, ``e ~ N(0, 1)``. The effect is the
constant 1, so the population RATE of every ranking is 0.
``sp.causal_forest`` with its defaults, ``random_state = r``; data seed
``1000 + 100000 r`` (spaced, as CLAUDE.md requires for forest studies).
``B = 400`` replications for ``own``; ``rate_split`` with ``n_splits = 5``
on the first 100.

Result (2026-10-16, StatsPAI 1.39.6 + this branch): ``own`` averaged
AUTOC -0.104 and rejected 36.0%; ``rate_split`` averaged 0.009 and
rejected 0%. The cross-sectional path now warns, as the fixed-effects
path already did.

Run: ``PYTHONPATH=src python tests/reliability/rate_cross_section.py``
(writes ``rate_cross_section_results.json``; about 30 minutes).
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

import statspai as sp

B_OWN = 400
B_SPLIT = 100


def main() -> None:
    warnings.filterwarnings("ignore")
    own, split = [], []
    for r in range(B_OWN):
        rng = np.random.default_rng(1000 + 100000 * r)
        n = 800
        X = rng.normal(size=(n, 4))
        T = rng.binomial(1, 0.5, n)
        Y = X[:, 0] + 0.5 * X[:, 1] + 1.0 * T + rng.normal(size=n)
        df = pd.DataFrame({"y": Y, "d": T, **{f"x{j}": X[:, j] for j in range(4)}})
        cf = sp.causal_forest(
            data=df, formula="y ~ d | x0 + x1 + x2 + x3", random_state=r
        )
        a = sp.rate(cf)
        own.append([float(a["estimate"]), float(a["se"])])
        if r < B_SPLIT:
            b = sp.rate_split(cf, n_splits=5, random_state=r)
            split.append([float(b["estimate"]), float(b.get("p", np.nan))])
    o, s = np.array(own), np.array(split)

    def cell(method: str, rows: np.ndarray, rate: float) -> dict:
        b = len(rows)
        return {
            "n": 800,
            "method": method,
            "B": b,
            "rejection_rate": rate,
            "mc_se": float(np.sqrt(rate * (1 - rate) / b)),
            "mean_autoc": float(rows[:, 0].mean()),
        }

    out = {
        "study": "rate_cross_section",
        "alpha": 0.05,
        "B": B_OWN,
        "statspai_version": sp.__version__,
        "cells": [
            cell(
                "rate_own_ranking",
                o,
                float(np.mean(np.abs(o[:, 0] / o[:, 1]) > 1.959964)),
            ),
            cell("rate_split", s, float(np.mean(s[:, 1] < 0.05))),
        ],
    }
    path = Path(__file__).with_name("rate_cross_section_results.json")
    path.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
