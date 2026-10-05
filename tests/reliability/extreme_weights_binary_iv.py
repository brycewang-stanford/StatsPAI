"""Extreme weights in ``sp.logit``, ``sp.probit``, ``sp.glm`` and ``sp.iv``.

The fourth weight study, with the design fixed before the first run. It
asks of the four remaining regression entry points what the first three
asked of the others: under sampling weights, does the default interval
cover, and does the robust one?

Every design tests a regressor whose true coefficient is zero with the
package's own 5% test; the reported number is the share of replications
that do not reject, which is the coverage of the 95% interval at the
truth.

* ``logit`` and ``probit``: ``y = 1[0.2 + 0.8 z + e > 0]`` with logistic
  or normal ``e``; regressors ``x`` (coefficient zero) and ``z``.
* ``glm``: ``sp.glm(family='binomial')`` on the logit design.
* ``iv``: ``sp.iv('y ~ z + (d ~ q)')`` with ``d = q + 0.5 u + v`` and
  ``y = 0.5 z + u``: the endogenous regressor ``d`` has a true
  coefficient of zero and a strong instrument.

``n`` in 200, 1000; observation weights log-normal with sigma in 0, 1, 2;
neither the outcome nor the errors depend on the weight (the sampling
reading). Variances: the default, and ``robust='robust'`` (``'hc1'`` for
``sp.iv``). With ``B = 2000`` a correct test lands in 0.950 +/- 0.010.

Run: ``python tests/reliability/extreme_weights_binary_iv.py [B] [workers]``.
Writes ``extreme_weights_binary_iv_results.json`` next to this file.
``tests/test_reliability_extreme_weights_binary_iv.py`` recomputes one
cell of each model on its first 60 replications.
"""

from __future__ import annotations

import json
import sys
import warnings
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
OUT = HERE / "extreme_weights_binary_iv_results.json"
ALPHA = 0.05
PREFIX = 60
SIGMAS = (0.0, 1.0, 2.0)
N_VALUES = (200, 1000)
MODELS = ("logit", "probit", "glm", "iv")
VARIANCES = ("classical", "robust")
TESTED = {"logit": "x", "probit": "x", "glm": "x", "iv": "d"}


def kish(w: np.ndarray) -> float:
    return float(w.sum() ** 2 / (w**2).sum())


def draw(model: str, n: int, sigma: float, seed: int) -> Tuple[pd.DataFrame, float]:
    rng = np.random.default_rng(seed)
    w = np.exp(rng.normal(scale=sigma, size=n)) if sigma > 0 else np.ones(n)
    x, z = rng.normal(size=n), rng.normal(size=n)
    if model == "iv":
        q, u, v = rng.normal(size=n), rng.normal(size=n), rng.normal(size=n)
        d = q + 0.5 * u + v
        y = 0.5 * z + u
        return pd.DataFrame({"y": y, "d": d, "q": q, "z": z, "w": w}), kish(w)
    e = rng.normal(size=n) if model == "probit" else rng.logistic(size=n)
    y = (0.2 + 0.8 * z + e > 0).astype(int)
    return pd.DataFrame({"y": y, "x": x, "z": z, "w": w}), kish(w)


def pvalue(model: str, df: pd.DataFrame, variance: str) -> float:
    import statspai as sp

    robust = variance == "robust"
    if model == "iv":
        kw = {"robust": "hc1"} if robust else {}
        fit = sp.iv("y ~ z + (d ~ q)", data=df, weights="w", **kw)
    elif model == "glm":
        kw = {"robust": "robust"} if robust else {}
        fit = sp.glm("y ~ x + z", df, family="binomial", weights="w", **kw)
    else:
        kw = {"robust": "robust"} if robust else {}
        fit = getattr(sp, model)("y ~ x + z", df, weights="w", **kw)
    return float(fit.pvalues[TESTED[model]])


def cell_seed(model: str, n: int, sigma: float, rep: int) -> int:
    return (
        17 * MODELS.index(model)
        + 6_000_011 * n
        + 1_009 * int(round(10 * sigma))
        + 104_729 * rep
    )


def run_cell(model: str, n: int, sigma: float, B: int) -> Dict[str, object]:
    hits: Dict[str, List[int]] = {v: [] for v in VARIANCES}
    refused: Dict[str, Dict[str, int]] = {v: {} for v in VARIANCES}
    n_eff: List[float] = []
    for rep in range(B):
        df, k = draw(model, n, sigma, cell_seed(model, n, sigma, rep))
        n_eff.append(k)
        for v in VARIANCES:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                try:
                    p = pvalue(model, df, v)
                except Exception as exc:  # counted by class, never dropped
                    key = type(exc).__name__
                    refused[v][key] = refused[v].get(key, 0) + 1
                    continue
            if not np.isfinite(p):
                refused[v]["non_finite"] = refused[v].get("non_finite", 0) + 1
                continue
            hits[v].append(int(p >= ALPHA))
    cell: Dict[str, object] = {
        "model": model,
        "n": n,
        "sigma": sigma,
        "B": B,
        "kish_median": float(np.median(n_eff)),
    }
    for v in VARIANCES:
        h = np.asarray(hits[v])
        rate = float(h.mean()) if h.size else float("nan")
        cell[v] = {
            "n_fitted": int(h.size),
            "refused": refused[v],
            "coverage": rate,
            "mc_se": float(np.sqrt(rate * (1 - rate) / h.size)) if h.size else None,
            "prefix_hits": int(h[:PREFIX].sum()),
        }
    return cell


def _job(args: Tuple[str, int, float, int]) -> Dict[str, object]:
    return run_cell(*args)


def main() -> None:
    import statspai as sp

    B = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    workers = int(sys.argv[2]) if len(sys.argv) > 2 else 4
    jobs = [(m, n, s, B) for m in MODELS for n in N_VALUES for s in SIGMAS]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        cells = list(pool.map(_job, jobs))
    for c in cells:
        print(
            f"{c['model']:7s} n={c['n']:4d} sigma={c['sigma']:.0f} "
            f"kish={c['kish_median']:7.1f} "
            + " ".join(
                f"{v}={c[v]['coverage']:.3f}"  # type: ignore[index]
                + (
                    f"(r{sum(c[v]['refused'].values())})"  # type: ignore[index]
                    if c[v]["refused"]  # type: ignore[index]
                    else ""
                )
                for v in VARIANCES
            ),
            flush=True,
        )
    payload = {
        "study": "extreme_weights_binary_iv",
        "level": 0.95,
        "B": B,
        "prefix": PREFIX,
        "statspai_version": sp.__version__,
        "cells": cells,
    }
    OUT.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
