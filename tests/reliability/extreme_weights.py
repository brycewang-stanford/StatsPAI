"""Coverage of weighted-regression intervals when the weights are extreme.

Question, fixed before the first run: at a nominal 95% level, how often
does the confidence interval of a slope in ``sp.regress(weights=)`` cover
the truth, as the dispersion of the weights grows, under each variance
option?

Design (12 cells, ``B`` replications each, one seed per replication):

* sample size ``n`` in 200, 1000;
* weights log-normal with sigma in 0 (all equal), 1, 2. The Kish
  effective sample size ``(sum w)^2 / sum w^2`` is about ``n``,
  ``n / 2.7`` and ``n / 55`` in expectation;
* two error models for ``y = 1 + 0.5 x + e``:

  - ``precision``: ``Var(e) = 1 / w``, the model analytic weights assume,
    under which the classical weighted variance is correct;
  - ``sampling``: ``Var(e) = 1`` whatever the weight, the survey reading,
    under which only a sandwich variance is.

Variances, each as a user would ask for it: classical (``weights=``
alone), ``hc1``, ``hc2``, ``hc3``.

Reported per cell and variance: coverage of the 95% interval for the
slope and its Monte Carlo standard error, plus the mean Kish effective
sample size. With ``B = 2000`` a correct interval lands in 0.950 +/- 0.010.

Run: ``python tests/reliability/extreme_weights.py [B]`` (a few minutes at
B = 2000). Writes ``extreme_weights_results.json`` next to this file.
``tests/test_reliability_extreme_weights.py`` recomputes one cell on its
first 60 replications and checks it against the stored prefix.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

import statspai as sp

HERE = Path(__file__).resolve().parent
OUT = HERE / "extreme_weights_results.json"
TRUTH = 0.5
PREFIX = 60
N_VALUES = (200, 1000)
SIGMAS = (0.0, 1.0, 2.0)
ERRORS = ("precision", "sampling")
VARIANCES = {
    "classical": {},
    "hc1": {"robust": "hc1"},
    "hc2": {"vce": "hc2"},
    "hc3": {"vce": "hc3"},
}


def kish(w: np.ndarray) -> float:
    return float(w.sum() ** 2 / (w**2).sum())


def draw(n: int, sigma: float, errors: str, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    w = np.exp(rng.normal(scale=sigma, size=n)) if sigma > 0 else np.ones(n)
    x = rng.normal(size=n)
    scale = 1.0 / np.sqrt(w) if errors == "precision" else np.ones(n)
    y = 1.0 + TRUTH * x + scale * rng.normal(size=n)
    return pd.DataFrame({"y": y, "x": x, "w": w})


def covers(df: pd.DataFrame) -> Dict[str, int]:
    out: Dict[str, int] = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name, kw in VARIANCES.items():
            fit = sp.regress("y ~ x", df, weights="w", **kw)
            lo = float(fit.conf_int_lower["x"])
            hi = float(fit.conf_int_upper["x"])
            out[name] = int(lo <= TRUTH <= hi)
    return out


def cell_seed(n: int, sigma: float, errors: str, rep: int) -> int:
    return (
        7_000_003 * n
        + 1_009 * int(round(10 * sigma))
        + (31 if errors == "sampling" else 0)
        + 104_729 * rep
    )


def run_cell(n: int, sigma: float, errors: str, B: int) -> Dict[str, object]:
    hits: Dict[str, List[int]] = {v: [] for v in VARIANCES}
    n_eff: List[float] = []
    for rep in range(B):
        df = draw(n, sigma, errors, cell_seed(n, sigma, errors, rep))
        n_eff.append(kish(df["w"].to_numpy()))
        for name, hit in covers(df).items():
            hits[name].append(hit)
    cell: Dict[str, object] = {
        "n": n,
        "sigma": sigma,
        "errors": errors,
        "B": B,
        "kish_n_mean": float(np.mean(n_eff)),
        "kish_n_median": float(np.median(n_eff)),
    }
    for name in VARIANCES:
        h = np.asarray(hits[name])
        rate = float(h.mean())
        cell[name] = {
            "coverage": rate,
            "mc_se": float(np.sqrt(rate * (1 - rate) / B)),
            "prefix_hits": int(h[:PREFIX].sum()),
        }
    return cell


def main() -> None:
    B = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    cells = []
    for errors in ERRORS:
        for n in N_VALUES:
            for sigma in SIGMAS:
                cell = run_cell(n, sigma, errors, B)
                cells.append(cell)
                print(
                    f"{errors:9s} n={n:4d} sigma={sigma:.0f} "
                    f"kish={cell['kish_n_median']:7.1f} "
                    + " ".join(
                        f"{v}={cell[v]['coverage']:.3f}"  # type: ignore[index]
                        for v in VARIANCES
                    ),
                    flush=True,
                )
    payload = {
        "study": "extreme_weights",
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
