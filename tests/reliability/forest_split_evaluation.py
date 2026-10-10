"""Size and power of split-sample evaluation for fixed-effects forests.

Question, fixed before the first run: when the same panel is used to grow
a causal forest and to ask whether its ranking (``sp.rate``) or a rule
built on it (``sp.forest_policy_tree``) finds real heterogeneity, how often
is a true null rejected at a nominal 5% level, and how often is a real
effect found?

Design (one seed per replication, ``B`` replications per cell):

* a staggered-adoption panel, ``N`` units by 8 periods, one unit in five
  never treated, adoption date selected on the unit effect;
* ``y = a_i + 0.25 t + tau_i d + e`` with ``sd(e) = 0.6`` and
  ``tau_i = 0.3 + b z_i``, ``z`` standard normal and independent of
  adoption. A second covariate ``w`` is pure noise;
* the forest is ``sp.causal_forest(fe="twoway")`` on ``z`` and ``w`` with
  250 trees, clustered by unit.

Because ``z`` is independent of adoption the population RATE of the ideal
ranking is known: AUTOC ``0.9032 b`` and QINI ``0.2821 b``. With ``b = 0``
it is 0 for every ranking, and with ``cost = 0.3`` the gain of every
treatment rule over treating everyone is exactly 0.

Cells:

* ``rate``   (N = 150): ``b`` in 0, 0.5; targets AUTOC and QINI. Methods:
  ``own`` -- ``sp.rate`` ranked by the forest's own out-of-bag predictions;
  ``split_bjs`` / ``split_forest`` -- ``sp.rate_split`` with its default 21
  splits and ``variance="bjs"`` (the default) or ``"forest"``.
* ``policy`` (N = 200): ``b`` in 0, 0.8 with ``cost = 0.3``;
  ``sp.forest_policy_tree`` with its defaults. The oracle gain at
  ``b = 0.8`` is ``0.8 * phi(0) = 0.3191``.

Reported per cell and method: the mean estimate, the share of 95%
intervals that exclude zero (size when the truth is 0, power otherwise),
and the share that cover the population value. The first is stored as
``rejection_rate`` when the truth is 0 and as ``power`` otherwise. With
``B = 400`` a correct 5% test lands in 0.050 +/- 0.011.

The halves are refitted exactly as the forest was (same controls, same
nuisance estimates); through 1.39.3 they were not, so figures from those
releases are not comparable.

Run: ``python tests/reliability/forest_split_evaluation.py [B]`` (about
ten minutes at B = 400 on eight cores). Writes
``forest_split_evaluation_results.json`` next to this file.
``tests/test_reliability_forest_split_evaluation.py`` recomputes the first
replications of one cell and checks them against the stored prefix.
"""

from __future__ import annotations

import json
import multiprocessing as mp
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

OUT = Path(__file__).with_name("forest_split_evaluation_results.json")
PREFIX = 3
AUTOC_PER_B = 0.9032
QINI_PER_B = 0.2821
POLICY_COST = 0.3
ORACLE_GAIN_PER_B = 0.3989422804014327  # phi(0)


def panel(seed: int, n_units: int, het: float, n_periods: int = 8) -> pd.DataFrame:
    """Staggered adoption selected on the unit effect; tau = 0.3 + het * z."""
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_units):
        a, z, w = rng.normal(), rng.normal(), rng.normal()
        never = i % 5 == 0
        g = 10**6 if never else int(np.clip(3 + round(a), 2, n_periods))
        for t in range(1, n_periods + 1):
            d = 1.0 * (t >= g)
            y = a + 0.25 * t + (0.3 + het * z) * d + rng.normal(0, 0.6)
            rows.append((i, t, y, d, z, w))
    return pd.DataFrame(rows, columns=["id", "t", "y", "d", "z", "w"])


def _forest(seed: int, n_units: int, het: float) -> Any:
    import statspai as sp

    df = panel(seed, n_units, het)
    return sp.causal_forest(
        "y ~ d | z + w",
        data=df,
        fe="twoway",
        unit="id",
        time="t",
        clusters=df["id"].to_numpy(),
        n_estimators=250,
        random_state=seed,
    )


def _triple(res: Dict[str, Any]) -> Tuple[float, float, float]:
    return float(res["estimate"]), float(res["ci_low"]), float(res["ci_high"])


def one_rate(args: Tuple[int, float]) -> Dict[str, Tuple[float, float, float]]:
    seed, het = args
    import statspai as sp

    out: Dict[str, Tuple[float, float, float]] = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cf = _forest(seed, 150, het)
        for target in ("AUTOC", "QINI"):
            out[f"own/{target}"] = _triple(sp.rate(cf, target=target))
            for variance in ("bjs", "forest"):
                res = sp.rate_split(cf, target, variance=variance, random_state=seed)
                out[f"split_{variance}/{target}"] = _triple(res)
    return out


def one_policy(args: Tuple[int, float]) -> Dict[str, Tuple[float, float, float]]:
    seed, het = args
    import statspai as sp

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cf = _forest(seed, 200, het)
        res = sp.forest_policy_tree(cf, cost=POLICY_COST, random_state=seed)
    return {"split/gain": _triple(res["gain_over_treat_all"])}


def summarise(
    draws: List[Dict[str, Tuple[float, float, float]]],
    truth: Dict[str, float],
    B: int,
) -> Dict[str, Dict[str, Any]]:
    """One entry per method. The share of intervals excluding zero is stored
    as ``rejection_rate`` when the truth is zero (a size) and as ``power``
    otherwise, the layout ``scripts/build_evidence_inventory.py`` reads."""
    table: Dict[str, Dict[str, Any]] = {}
    for key in draws[0]:
        arr = np.array([d[key] for d in draws], dtype=float)
        ok = np.isfinite(arr).all(axis=1)
        est, lo, hi = arr[ok].T
        target = truth[key.split("/")[1]]
        share = float(np.mean((lo > 0) | (hi < 0)))
        table[key] = {
            "n_fitted": int(ok.sum()),
            "refused": {"non-finite interval": int(B - ok.sum())},
            "truth": target,
            "mean": float(est.mean()),
            "sd": float(est.std(ddof=1)),
            ("rejection_rate" if target == 0 else "power"): share,
            "covers_truth": float(np.mean((lo <= target) & (target <= hi))),
            "prefix": [[float(v) for v in row] for row in arr[:PREFIX]],
        }
    return table


def main() -> None:
    B = int(sys.argv[1]) if len(sys.argv) > 1 else 400
    import statspai as sp

    cells = []
    ctx = mp.get_context("spawn")
    with ctx.Pool() as pool:
        for het in (0.0, 0.5):
            draws = pool.map(one_rate, [(s, het) for s in range(B)])
            truth = {"AUTOC": AUTOC_PER_B * het, "QINI": QINI_PER_B * het}
            cell = {"quantity": "rate", "b": het, "model": f"tau = 0.3 + {het} z"}
            cell.update({"N": 150, "B": B})
            cell.update(summarise(draws, truth, B))
            cells.append(cell)
        for het in (0.0, 0.8):
            draws = pool.map(one_policy, [(s, het) for s in range(B)])
            truth = {"gain": ORACLE_GAIN_PER_B * het}
            cell = {"quantity": "policy", "b": het, "model": f"tau = 0.3 + {het} z"}
            cell.update({"N": 200, "B": B})
            cell.update(summarise(draws, truth, B))
            cells.append(cell)
    for cell in cells:
        for key, row in cell.items():
            if isinstance(row, dict):
                print(f"{cell['quantity']} b={cell['b']} {key:22s} {row}", flush=True)
    payload = {
        "study": "forest_split_evaluation",
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
