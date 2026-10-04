"""Extreme weights in ``sp.fast.feols``, ``sp.fast.fepois`` and ``sp.nbreg``.

The third weight study, with the design fixed before the first run. The
first two cover ``sp.regress`` and ``sp.panel`` / ``sp.poisson``; this
one asks the same question of the entry points they left out.

Every design has a regressor ``x`` whose true coefficient is zero and a
second regressor ``z`` that does matter. The reported number is the
share of replications in which the package's own 5% test of ``x`` does
not reject, which is the coverage of its 95% interval at the truth
(these result objects print a p-value and no interval).

* ``fast_feols``: ``sp.fast.feols('y ~ x + z | id', weights=)`` on ``G``
  units observed 5 times, ``y = a_i + 0.5 z + e``, ``e`` standard normal.
  Unit-level sampling weights, log-normal with sigma in 0, 1, 2; ``G`` in
  50, 200. Variances ``iid``, ``hc1``, ``cr1`` on the unit.
* ``fast_fepois``: ``sp.fast.fepois('y ~ x + z | id', weights=)`` on the
  same panel with ``y ~ Poisson(exp(0.3 a_i + 0.5 z))``. The same three
  variances.
* ``nbreg``: ``sp.nbreg('y ~ x + z', weights=)`` on ``n`` in 200, 1000
  observations, ``y`` negative binomial (NB2, alpha = 0.5) with mean
  ``exp(0.2 + 0.5 z)``, observation weights with the same sigmas.
  Variances: classical (the default) and ``robust='robust'``.

The errors never depend on the weight (the sampling reading). With
``B = 2000`` a correct test lands in 0.950 +/- 0.010.

Run: ``python tests/reliability/extreme_weights_fast.py [B] [workers]``
(cells in parallel). Writes ``extreme_weights_fast_results.json`` next
to this file. ``tests/test_reliability_extreme_weights_fast.py``
recomputes one cell of each model on its first 60 replications.
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
OUT = HERE / "extreme_weights_fast_results.json"
ALPHA = 0.05
PREFIX = 60
T = 5
SIGMAS = (0.0, 1.0, 2.0)
SIZES = {"fast_feols": (50, 200), "fast_fepois": (50, 200), "nbreg": (200, 1000)}
VARIANCES: Dict[str, Tuple[str, ...]] = {
    "fast_feols": ("iid", "hc1", "cr1"),
    "fast_fepois": ("iid", "hc1", "cr1"),
    "nbreg": ("classical", "robust"),
}
MODELS = tuple(SIZES)


def kish(w: np.ndarray) -> float:
    return float(w.sum() ** 2 / (w**2).sum())


def _weights(rng: np.random.Generator, size: int, sigma: float) -> np.ndarray:
    return np.exp(rng.normal(scale=sigma, size=size)) if sigma > 0 else np.ones(size)


def draw(model: str, size: int, sigma: float, seed: int) -> Tuple[pd.DataFrame, float]:
    rng = np.random.default_rng(seed)
    if model == "nbreg":
        w = _weights(rng, size, sigma)
        x, z = rng.normal(size=size), rng.normal(size=size)
        mu = np.exp(0.2 + 0.5 * z)
        y = rng.negative_binomial(2.0, 2.0 / (2.0 + mu))  # NB2, alpha = 0.5
        return pd.DataFrame({"y": y, "x": x, "z": z, "w": w}), kish(w)
    w_unit = _weights(rng, size, sigma)
    unit = np.repeat(np.arange(size), T)
    a = np.repeat(rng.normal(size=size), T)
    x, z = rng.normal(size=size * T), rng.normal(size=size * T)
    if model == "fast_feols":
        y = a + 0.5 * z + rng.normal(size=size * T)
    elif model == "fast_fepois":
        y = rng.poisson(np.exp(0.3 * a + 0.5 * z))
    else:
        raise ValueError(model)
    df = pd.DataFrame({"y": y, "x": x, "z": z, "id": unit, "w": w_unit[unit]})
    return df, kish(w_unit)


def pvalue(model: str, df: pd.DataFrame, variance: str) -> float:
    import statspai as sp

    if model == "nbreg":
        kw = {} if variance == "classical" else {"robust": "robust"}
        return float(sp.nbreg("y ~ x + z", df, weights="w", **kw).pvalues["x"])
    kw = {"vcov": variance}
    if variance == "cr1":
        kw["cluster"] = "id"
    fit = (sp.fast.feols if model == "fast_feols" else sp.fast.fepois)(
        "y ~ x + z | id", df, weights="w", **kw
    )
    row = fit.tidy().loc["x"]
    return float(row[[c for c in row.index if c.startswith("Pr(")][0]])


def cell_seed(model: str, size: int, sigma: float, rep: int) -> int:
    return (
        13 * MODELS.index(model)
        + 4_000_037 * size
        + 1_009 * int(round(10 * sigma))
        + 104_729 * rep
    )


def run_cell(model: str, size: int, sigma: float, B: int) -> Dict[str, object]:
    hits: Dict[str, List[int]] = {v: [] for v in VARIANCES[model]}
    refused: Dict[str, Dict[str, int]] = {v: {} for v in VARIANCES[model]}
    n_eff: List[float] = []
    for rep in range(B):
        df, k = draw(model, size, sigma, cell_seed(model, size, sigma, rep))
        n_eff.append(k)
        for v in VARIANCES[model]:
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
        "size": size,
        "sigma": sigma,
        "B": B,
        "kish_median": float(np.median(n_eff)),
    }
    for v in VARIANCES[model]:
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
    jobs = [(m, n, s, B) for m in MODELS for n in SIZES[m] for s in SIGMAS]
    with ProcessPoolExecutor(max_workers=workers) as pool:
        cells = list(pool.map(_job, jobs))
    for c in cells:
        print(
            f"{c['model']:11s} size={c['size']:4d} sigma={c['sigma']:.0f} "
            f"kish={c['kish_median']:7.1f} "
            + " ".join(
                f"{v}={c[v]['coverage']:.3f}"  # type: ignore[index]
                + (
                    f"(r{sum(c[v]['refused'].values())})"  # type: ignore[index]
                    if c[v]["refused"]  # type: ignore[index]
                    else ""
                )
                for v in VARIANCES[str(c["model"])]
            ),
            flush=True,
        )
    payload = {
        "study": "extreme_weights_fast",
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
