"""Size of cluster-robust tests in ``sp.regress`` with few clusters.

Question, fixed before the first run: at a nominal 5% level, how often
does each inference method reject a true null on a cluster-level
regressor, as the number of clusters, the share of treated clusters and
the balance of cluster sizes vary?

Design (16 cells, ``B`` replications each, one seed per replication):

* clusters ``G`` in 6, 10, 20, 40;
* treated clusters: half of them, or exactly two (the few-treated case);
* cluster sizes: 30 each, or one cluster holding half the sample and the
  rest sharing the other half;
* outcome ``y = 1 + 0 * d + a_g + e`` with a cluster effect of variance
  0.3 and unit noise of variance 0.7 (intra-cluster correlation 0.3), so
  the null ``beta_d = 0`` is true.

Methods, each as a user would call it:

* ``cr1``  -- ``sp.regress(cluster=)``: CR1, t(G - 1) reference;
* ``cr2``  -- ``sp.regress(vce='cr2', cluster=)``: CR2, normal reference;
* ``cr3``  -- ``sp.regress(vce='cr3', cluster=)``: CR3, normal reference;
* ``wild`` -- ``sp.wild_cluster_bootstrap``: WCR bootstrap p-value, 999
  draws, Rademacher weights (Webb weights below 12 clusters, as the
  function itself recommends).

Reported per cell and method: the rejection rate and its Monte Carlo
standard error ``sqrt(p (1 - p) / B)``. With ``B = 2000`` a correctly
sized test lands in 0.050 +/- 0.010 (two standard errors).

Run: ``python tests/reliability/few_clusters.py [B]`` (about a quarter of
an hour at B = 2000). Writes ``few_clusters_results.json`` next to this
file. ``tests/test_reliability_few_clusters.py`` recomputes one cell on
its first 60 replications and checks it against the stored prefix.
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
OUT = HERE / "few_clusters_results.json"
ALPHA = 0.05
N_BOOT = 999
PREFIX = 60  # replications stored separately for the regression test
G_VALUES = (6, 10, 20, 40)
TREATED = ("half", "two")
SIZES = ("balanced", "unbalanced")
METHODS = ("cr1", "cr2", "cr3", "wild")


def cluster_sizes(G: int, sizes: str) -> np.ndarray:
    if sizes == "balanced":
        return np.full(G, 30)
    total = 30 * G
    big = total // 2
    rest = np.full(G - 1, (total - big) // (G - 1))
    return np.concatenate([[big], rest])


def draw(G: int, treated: str, sizes: str, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_g = cluster_sizes(G, sizes)
    g = np.repeat(np.arange(G), n_g)
    n_treated = G // 2 if treated == "half" else 2
    # treated clusters are drawn at random, so the large cluster is
    # treated with probability n_treated / G
    d_cluster = np.zeros(G, dtype=int)
    d_cluster[rng.choice(G, size=n_treated, replace=False)] = 1
    a = rng.normal(scale=np.sqrt(0.3), size=G)
    e = rng.normal(scale=np.sqrt(0.7), size=g.size)
    return pd.DataFrame({"y": 1.0 + a[g] + e, "d": d_cluster[g], "g": g})


def pvalues(df: pd.DataFrame, seed: int) -> Dict[str, float]:
    G = int(df["g"].nunique())
    out: Dict[str, float] = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out["cr1"] = float(sp.regress("y ~ d", df, cluster="g").pvalues["d"])
        out["cr2"] = float(sp.regress("y ~ d", df, vce="cr2", cluster="g").pvalues["d"])
        out["cr3"] = float(sp.regress("y ~ d", df, vce="cr3", cluster="g").pvalues["d"])
        wild = sp.wild_cluster_bootstrap(
            df,
            y="y",
            x=["d"],
            cluster="g",
            test_var="d",
            n_boot=N_BOOT,
            weight_type="webb" if G < 12 else "rademacher",
            seed=seed,
        )
        out["wild"] = float(wild["p_boot"])
    return out


def run_cell(G: int, treated: str, sizes: str, B: int) -> Dict[str, object]:
    rejections: Dict[str, List[int]] = {m: [] for m in METHODS}
    for rep in range(B):
        # one integer per (cell, replication), stable across runs
        seed = 1_000_003 * G + 7919 * rep + (17 if treated == "two" else 0)
        seed += 104_729 if sizes == "unbalanced" else 0
        p = pvalues(draw(G, treated, sizes, seed), seed)
        for m in METHODS:
            rejections[m].append(int(p[m] < ALPHA))
    cell: Dict[str, object] = {"G": G, "treated": treated, "sizes": sizes, "B": B}
    for m in METHODS:
        r = np.asarray(rejections[m])
        rate = float(r.mean())
        cell[m] = {
            "rejection_rate": rate,
            "mc_se": float(np.sqrt(rate * (1 - rate) / B)),
            "prefix_rejections": int(r[:PREFIX].sum()),
        }
    return cell


def main() -> None:
    B = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    cells = []
    for G in G_VALUES:
        for treated in TREATED:
            for sizes in SIZES:
                cell = run_cell(G, treated, sizes, B)
                cells.append(cell)
                print(
                    f"G={G:2d} treated={treated:4s} {sizes:10s} "
                    + " ".join(
                        f"{m}={cell[m]['rejection_rate']:.3f}"  # type: ignore[index]
                        for m in METHODS
                    ),
                    flush=True,
                )
    payload = {
        "study": "few_clusters",
        "alpha": ALPHA,
        "B": B,
        "n_boot": N_BOOT,
        "prefix": PREFIX,
        "statspai_version": sp.__version__,
        "cells": cells,
    }
    OUT.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
