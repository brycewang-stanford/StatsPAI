"""Coverage under extreme weights beyond OLS: fixed effects and Poisson.

``extreme_weights.py`` answered the question for ``sp.regress``. This
study asks it, with the design fixed before the first run, for the two
other places a user passes ``weights=``:

* ``panel_fe``: ``sp.panel(method='fe', weights=)`` on ``G`` units
  observed 5 times, ``y = a_i + 0.5 x + e`` with ``e`` independent
  standard normal. The weight is a unit-level sampling weight (constant
  within unit, log-normal with sigma in 0, 1, 2); ``G`` in 50, 200.
  Variances: classical, ``robust='robust'``, ``cluster='id'``.
* ``poisson``: ``sp.poisson(weights=)`` on ``n`` in 200, 1000
  observations, ``y ~ Poisson(exp(0.2 + 0.5 x))`` (correctly specified),
  observation weights log-normal with the same sigmas. Variances:
  classical (the default) and ``robust='robust'``.

In both the errors do not depend on the weight (the sampling reading).
Reported per cell and variance: coverage of the 95% interval for the
slope, its Monte Carlo standard error, and the median Kish effective
size of the weights (over units for the panel, over observations for
Poisson). With ``B = 2000`` a correct interval lands in 0.950 +/- 0.010.

Run: ``python tests/reliability/extreme_weights_models.py [B]``. Writes
``extreme_weights_models_results.json`` next to this file.
``tests/test_reliability_extreme_weights_models.py`` recomputes one cell
of each model on its first 60 replications.
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
OUT = HERE / "extreme_weights_models_results.json"
TRUTH = 0.5
PREFIX = 60
T = 5
SIGMAS = (0.0, 1.0, 2.0)
SIZES = {"panel_fe": (50, 200), "poisson": (200, 1000)}
VARIANCES: Dict[str, Dict[str, Dict[str, object]]] = {
    "panel_fe": {
        "classical": {},
        "robust": {"robust": "robust"},
        "cluster": {"cluster": "id"},
    },
    "poisson": {"classical": {}, "robust": {"robust": "robust"}},
}


def kish(w: np.ndarray) -> float:
    return float(w.sum() ** 2 / (w**2).sum())


def _weights(rng: np.random.Generator, size: int, sigma: float) -> np.ndarray:
    return np.exp(rng.normal(scale=sigma, size=size)) if sigma > 0 else np.ones(size)


def draw_panel(G: int, sigma: float, seed: int) -> Tuple[pd.DataFrame, float]:
    rng = np.random.default_rng(seed)
    w_unit = _weights(rng, G, sigma)
    unit = np.repeat(np.arange(G), T)
    x = rng.normal(size=G * T) + np.repeat(rng.normal(size=G), T)
    y = np.repeat(rng.normal(size=G), T) + TRUTH * x + rng.normal(size=G * T)
    df = pd.DataFrame(
        {"y": y, "x": x, "id": unit, "t": np.tile(np.arange(T), G), "w": w_unit[unit]}
    )
    return df, kish(w_unit)


def draw_poisson(n: int, sigma: float, seed: int) -> Tuple[pd.DataFrame, float]:
    rng = np.random.default_rng(seed)
    w = _weights(rng, n, sigma)
    x = rng.normal(size=n)
    y = rng.poisson(np.exp(0.2 + TRUTH * x))
    return pd.DataFrame({"y": y, "x": x, "w": w}), kish(w)


def fit_panel(df: pd.DataFrame, **kw: object):
    return sp.panel(df, "y ~ x", entity="id", time="t", method="fe", weights="w", **kw)


def fit_poisson(df: pd.DataFrame, **kw: object):
    return sp.poisson("y ~ x", df, weights="w", **kw)


MODELS: Dict[str, Tuple[Callable, Callable]] = {
    "panel_fe": (draw_panel, fit_panel),
    "poisson": (draw_poisson, fit_poisson),
}


def cell_seed(model: str, size: int, sigma: float, rep: int) -> int:
    return (
        (11 if model == "poisson" else 0)
        + 5_000_011 * size
        + 1_009 * int(round(10 * sigma))
        + 104_729 * rep
    )


def run_cell(model: str, size: int, sigma: float, B: int) -> Dict[str, object]:
    draw, fit = MODELS[model]
    hits: Dict[str, List[int]] = {v: [] for v in VARIANCES[model]}
    n_eff: List[float] = []
    for rep in range(B):
        df, k = draw(size, sigma, cell_seed(model, size, sigma, rep))
        n_eff.append(k)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            for name, kw in VARIANCES[model].items():
                res = fit(df, **kw)
                lo = float(res.conf_int_lower["x"])
                hi = float(res.conf_int_upper["x"])
                hits[name].append(int(lo <= TRUTH <= hi))
    cell: Dict[str, object] = {
        "model": model,
        "size": size,
        "sigma": sigma,
        "B": B,
        "kish_median": float(np.median(n_eff)),
    }
    for name in VARIANCES[model]:
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
    for model in MODELS:
        for size in SIZES[model]:
            for sigma in SIGMAS:
                cell = run_cell(model, size, sigma, B)
                cells.append(cell)
                print(
                    f"{model:9s} size={size:4d} sigma={sigma:.0f} "
                    f"kish={cell['kish_median']:7.1f} "
                    + " ".join(
                        f"{v}={cell[v]['coverage']:.3f}"  # type: ignore[index]
                        for v in VARIANCES[model]
                    ),
                    flush=True,
                )
    payload = {
        "study": "extreme_weights_models",
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
