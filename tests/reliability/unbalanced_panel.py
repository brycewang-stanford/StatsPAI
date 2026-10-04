"""Coverage of staggered-adoption DiD estimators on an unbalanced panel.

Question, fixed before the first run: at a nominal 95% level, how often
does the interval for the overall ATT cover the truth when unit-period
cells are missing, under each way the cells can go missing and each way
an estimator can use what is left?

Design (8 cells, ``B`` replications each, one seed per replication):

* ``N`` units in 100, 400, observed over 8 periods; 30% adopt in period
  4, 30% in period 6, 40% never;
* ``y_it = a_i + 0.2 t + 1 * D_it + e_it`` with ``a_i`` standard normal
  and ``e_it`` AR(1) with coefficient 0.5 and unit variance. The effect
  is 1 in every treated cell, so every aggregation has the same truth
  and the comparison is about the missing cells alone;
* four patterns of missing cells:

  - ``balanced``: none;
  - ``mcar``: each cell dropped with probability 0.3;
  - ``attrit_level``: adopting units with ``a_i > 0`` are not observed
    from period 7 on (attrition on the unit's level, which unit fixed
    effects absorb);
  - ``attrit_outcome``: a treated cell is dropped when its shock
    ``e_it`` is below the 30th percentile of its distribution (attrition
    on the outcome itself, which no estimator here can undo).

Estimators, each as a user would call it:

* ``cs``: ``sp.callaway_santanna`` (within-unit differences) and
  ``sp.aggte(type='simple')``;
* ``cs_rcs``: the same with ``allow_unbalanced_panel=True``
  (repeated-cross-section estimators that keep every observed row);
* ``bjs``: ``sp.did_imputation(autosample=True)``;
* ``twfe``: ``sp.panel(method='twoway', cluster=unit)``.

Reported per cell and estimator: mean bias, coverage of the 95% interval
and its Monte Carlo standard error, and how many fits raised an error
(by class). With ``B = 1000`` a correct interval lands in 0.950 +/- 0.014.

Run: ``python tests/reliability/unbalanced_panel.py [B]``. Writes
``unbalanced_panel_results.json`` next to this file.
``tests/test_reliability_unbalanced_panel.py`` recomputes one cell on its
first 40 replications.
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path
from typing import Callable, Dict, List, Tuple

import numpy as np
import pandas as pd

import statspai as sp

HERE = Path(__file__).resolve().parent
OUT = HERE / "unbalanced_panel_results.json"
TRUTH = 1.0
PREFIX = 40
T = 8
N_VALUES = (100, 400)
PATTERNS = ("balanced", "mcar", "attrit_level", "attrit_outcome")
SHOCK_Q30 = -0.5244005127080407  # 30th percentile of N(0, 1)


def draw(N: int, pattern: str, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    g = rng.choice([4, 6, 0], size=N, p=[0.3, 0.3, 0.4])
    a = rng.normal(size=N)
    e = np.empty((N, T))
    e[:, 0] = rng.normal(size=N)
    for t in range(1, T):
        e[:, t] = 0.5 * e[:, t - 1] + np.sqrt(0.75) * rng.normal(size=N)
    period = np.arange(1, T + 1)
    d = ((g[:, None] > 0) & (period[None, :] >= g[:, None])).astype(float)
    y = a[:, None] + 0.2 * period[None, :] + TRUTH * d + e
    keep = np.ones((N, T), dtype=bool)
    if pattern == "mcar":
        keep = rng.random((N, T)) > 0.3
    elif pattern == "attrit_level":
        keep[np.ix_((g > 0) & (a > 0), period >= 7)] = False
    elif pattern == "attrit_outcome":
        keep = ~((d == 1) & (e < SHOCK_Q30))
    elif pattern != "balanced":
        raise ValueError(pattern)
    df = pd.DataFrame(
        {
            "id": np.repeat(np.arange(N), T),
            "t": np.tile(period, N),
            "g": np.repeat(g, T),
            "d": d.ravel(),
            "y": y.ravel(),
        }
    )
    return df[keep.ravel()].reset_index(drop=True)


def _cs(df: pd.DataFrame, **kw: object) -> Tuple[float, float, float]:
    fit = sp.callaway_santanna(df, y="y", g="g", t="t", i="id", **kw)
    agg = sp.aggte(fit, type="simple")
    return float(agg.estimate), float(agg.ci[0]), float(agg.ci[1])


def fit_cs(df: pd.DataFrame) -> Tuple[float, float, float]:
    return _cs(df)


def fit_cs_rcs(df: pd.DataFrame) -> Tuple[float, float, float]:
    return _cs(df, allow_unbalanced_panel=True)


def fit_bjs(df: pd.DataFrame) -> Tuple[float, float, float]:
    fit = sp.did_imputation(
        df, y="y", group="id", time="t", first_treat="g", autosample=True
    )
    return float(fit.estimate), float(fit.ci[0]), float(fit.ci[1])


def fit_twfe(df: pd.DataFrame) -> Tuple[float, float, float]:
    fit = sp.panel(df, "y ~ d", entity="id", time="t", method="twoway", cluster="id")
    return (
        float(fit.params["d"]),
        float(fit.conf_int_lower["d"]),
        float(fit.conf_int_upper["d"]),
    )


ESTIMATORS: Dict[str, Callable[[pd.DataFrame], Tuple[float, float, float]]] = {
    "cs": fit_cs,
    "cs_rcs": fit_cs_rcs,
    "bjs": fit_bjs,
    "twfe": fit_twfe,
}


def cell_seed(N: int, pattern: str, rep: int) -> int:
    return 3_000_017 * N + 1_013 * PATTERNS.index(pattern) + 104_729 * rep


def run_cell(N: int, pattern: str, B: int) -> Dict[str, object]:
    est: Dict[str, List[float]] = {k: [] for k in ESTIMATORS}
    hit: Dict[str, List[int]] = {k: [] for k in ESTIMATORS}
    refused: Dict[str, Dict[str, int]] = {k: {} for k in ESTIMATORS}
    share: List[float] = []
    for rep in range(B):
        df = draw(N, pattern, cell_seed(N, pattern, rep))
        share.append(len(df) / (N * T))
        for name, fit in ESTIMATORS.items():
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    b, lo, hi = fit(df)
                except Exception as exc:  # counted by class, never dropped
                    key = type(exc).__name__
                    refused[name][key] = refused[name].get(key, 0) + 1
                    continue
            if not (np.isfinite(b) and np.isfinite(lo) and np.isfinite(hi)):
                refused[name]["non_finite"] = refused[name].get("non_finite", 0) + 1
                continue
            est[name].append(b)
            hit[name].append(int(lo <= TRUTH <= hi))
    cell: Dict[str, object] = {
        "N": N,
        "pattern": pattern,
        "B": B,
        "share_observed": float(np.mean(share)),
    }
    for name in ESTIMATORS:
        h = np.asarray(hit[name])
        n_fit = int(h.size)
        rate = float(h.mean()) if n_fit else float("nan")
        cell[name] = {
            "n_fitted": n_fit,
            "refused": refused[name],
            "bias": float(np.mean(est[name]) - TRUTH) if n_fit else float("nan"),
            "sd": float(np.std(est[name], ddof=1)) if n_fit > 1 else float("nan"),
            "coverage": rate,
            "mc_se": float(np.sqrt(rate * (1 - rate) / n_fit)) if n_fit else None,
            "prefix_hits": int(h[:PREFIX].sum()),
        }
    return cell


def main() -> None:
    B = int(sys.argv[1]) if len(sys.argv) > 1 else 1000
    cells = []
    for pattern in PATTERNS:
        for N in N_VALUES:
            cell = run_cell(N, pattern, B)
            cells.append(cell)
            print(
                f"{pattern:15s} N={N:3d} obs={cell['share_observed']:.2f} "
                + " ".join(
                    f"{k}={cell[k]['bias']:+.3f}/{cell[k]['coverage']:.3f}"  # type: ignore[index]
                    + (
                        f"(r{sum(cell[k]['refused'].values())})"  # type: ignore[index]
                        if cell[k]["refused"]  # type: ignore[index]
                        else ""
                    )
                    for k in ESTIMATORS
                ),
                flush=True,
            )
    payload = {
        "study": "unbalanced_panel",
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
