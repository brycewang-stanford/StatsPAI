"""Coverage of ``sp.rdrobust`` when the running variable is discrete.

Question, fixed before the first run: at a nominal 95% level, how often
does the robust bias-corrected interval cover a known jump, as the
running variable goes from continuous to a handful of support points on
each side of the cutoff?

Design (30 cells, ``B`` replications each, one seed per replication):

* sample size ``n`` in 1000, 4000;
* running variable uniform on (-1, 1), either left continuous or rounded
  to ``M`` equally spaced support points per side, ``M`` in 50, 20, 10, 5.
  No support point sits at the cutoff: the nearest are at +/- 1 / (2 M);
* outcome ``y = 1 * 1(x >= 0) + 0.5 x + 2 x^2 - 1.5 x^3 + e`` with
  ``sd(e) = 0.5``, so the regression function is smooth and the jump at
  the cutoff is exactly 1.

Methods, each as a user would call it:

* ``adjust``  -- ``sp.rdrobust(...)``: the default ``masspoints='adjust'``;
* ``off``     -- ``masspoints='off'``: bandwidths as for continuous data;
* ``cluster`` -- the default plus ``cluster=`` on the running variable's
  own values (one cluster per support point); with a continuous running
  variable every row is its own cluster and this cell is left out.

Reported per cell and method: coverage of the robust 95% interval, its
Monte Carlo standard error, the mean interval length, the bias of the
conventional estimate, and how many fits ``sp.rdrobust`` refused, by
error class (coverage is over the fits that ran). With ``B = 1000`` a correct interval lands in
0.950 +/- 0.014.

Run: ``python tests/reliability/rd_mass_points.py [B]`` (about half an
hour at B = 1000). Writes ``rd_mass_points_results.json`` next to this
file. ``tests/test_reliability_rd_mass_points.py`` recomputes one cell on
its first 40 replications and checks it against the stored prefix.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

import statspai as sp

HERE = Path(__file__).resolve().parent
OUT = HERE / "rd_mass_points_results.json"
TRUTH = 1.0
PREFIX = 40
N_VALUES = (1000, 4000)
SUPPORT = (0, 50, 20, 10, 5)  # 0 = continuous
METHODS = ("adjust", "off", "cluster")


def draw(n: int, M: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.uniform(-1.0, 1.0, size=n)
    if M:
        x = (np.floor(x * M) + 0.5) / M
    m = 0.5 * x + 2.0 * x**2 - 1.5 * x**3
    y = TRUTH * (x >= 0) + m + rng.normal(scale=0.5, size=n)
    return pd.DataFrame({"x": x, "y": y, "xv": x})


def fit(df: pd.DataFrame, method: str) -> Optional[Dict[str, float]]:
    kw: Dict[str, object] = {}
    if method == "off":
        kw["masspoints"] = "off"
    elif method == "cluster":
        kw["cluster"] = "xv"
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sp.rdrobust(df, y="y", x="x", c=0, **kw)
    lo, hi = float(res.ci[0]), float(res.ci[1])
    return {
        "hit": float(lo <= TRUTH <= hi),
        "length": hi - lo,
        "conventional": float(res.detail["estimate"][0]),
    }


def cell_seed(n: int, M: int, rep: int) -> int:
    return 3_000_017 * n + 10_007 * M + 104_729 * rep


def run_cell(n: int, M: int, method: str, B: int) -> Dict[str, object]:
    hits: List[float] = []
    lengths: List[float] = []
    conv: List[float] = []
    failures: Dict[str, int] = {}
    for rep in range(B):
        df = draw(n, M, cell_seed(n, M, rep))
        try:
            out = fit(df, method)
        except (ValueError, RuntimeError) as exc:
            # StatsPAI's taxonomy errors: the fit refused. Counted by
            # class and reported; coverage is over the fits that ran.
            name = type(exc).__name__
            failures[name] = failures.get(name, 0) + 1
            continue
        hits.append(out["hit"])
        lengths.append(out["length"])
        conv.append(out["conventional"])
    h = np.asarray(hits)
    rate = float(h.mean()) if h.size else float("nan")
    return {
        "n": n,
        "support_per_side": M,
        "method": method,
        "B": B,
        "refused": failures,
        "n_fitted": int(h.size),
        "coverage": rate,
        "mc_se": float(np.sqrt(rate * (1 - rate) / max(h.size, 1))),
        "mean_length": float(np.mean(lengths)) if lengths else float("nan"),
        "bias_conventional": float(np.mean(conv) - TRUTH) if conv else float("nan"),
        "prefix_hits": int(h[:PREFIX].sum()),
    }


def main() -> None:
    B = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    cells = []
    for n in N_VALUES:
        for M in SUPPORT:
            for method in METHODS:
                if method == "cluster" and M == 0:
                    continue
                cell = run_cell(n, M, method, B)
                cells.append(cell)
                print(
                    f"n={n:4d} M={M:2d} {method:7s} cov={cell['coverage']:.3f} "
                    f"len={cell['mean_length']:.3f} "
                    f"bias={cell['bias_conventional']:+.3f} "
                    f"fitted={cell['n_fitted']} refused={cell['refused']}",
                    flush=True,
                )
    payload = {
        "study": "rd_mass_points",
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
