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
* ``cr2``  -- ``sp.regress(vce='cr2', cluster=)``: CR2, t(G - 1);
* ``cr3``  -- ``sp.regress(vce='cr3', cluster=)``: CR3, t(G - 1) (both were
  on a normal reference when this study was first run; its CR2 column at
  six balanced clusters, 12%, is what prompted the change);
* ``wild`` -- ``sp.wild_cluster_bootstrap``: WCR bootstrap p-value, 999
  draws, Rademacher weights (Webb weights below 12 clusters, as the
  function itself recommends).

Reported per cell and method: the rejection rate and its Monte Carlo
standard error ``sqrt(p (1 - p) / B)``. With ``B = 2000`` a correctly
sized test lands in 0.050 +/- 0.010 (two standard errors).

A second block asks what the *number* of clusters hides. With 40 or 60
clusters, half of them treated, cluster sizes are drawn as ``2:1`` (half
the clusters twice the size of the rest) or log-normal with sigma 0.5 or
1. For each design the file records the effective number of clusters by
size, ``(sum n_g)^2 / sum n_g^2`` (the inverse Herfindahl index of the
cluster shares), next to the rejection rates of CR1 and CR3. That is the
evidence behind the ``n_clusters_effective`` diagnostic and warning in
``sp.regress``.

A third block repeats the dominant-cluster question for a fixed-effects
panel, where the regressor varies within units: 40 units, unit effects
absorbed, AR(1) regressor and error (rho = 0.6), clustered by unit with
``sp.panel(method='fe', ssc='stata')``; either every unit has 8 periods
or one unit has 312 and the other 39 have 8.

A fourth block is the difference-in-differences version of "few treated":
40 units over 10 periods, unit and period effects, AR(1) errors
(rho = 0.5), and 1, 2, 5, 10 or 20 units treated from period 6 on with a
true effect of zero. The treatment varies within units, so it is not a
cluster-level regressor; what is few is the number of units ever treated.
Two tests of the true null: the two-way fixed-effects coefficient with
unit-clustered errors (``sp.panel(method='twoway', cluster=unit)``) and
the placebo test of ``sp.did_few_treated``.

Run: ``python tests/reliability/few_clusters.py [B]`` (about a quarter of
an hour at B = 2000; ``python tests/reliability/few_clusters.py B did``
reruns the fourth block alone and keeps the rest of the stored file). Writes ``few_clusters_results.json`` next to this
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


DISPERSION_G = (40, 60)
DISPERSION_KINDS = ("two_to_one", "lognormal_0.5", "lognormal_1")


def dispersed_sizes(G: int, kind: str, rng: np.random.Generator) -> np.ndarray:
    if kind == "two_to_one":
        w = np.where(np.arange(G) < G // 2, 2.0, 1.0)
    else:
        w = np.exp(rng.normal(scale=float(kind.split("_")[1]), size=G))
    return np.maximum((w / w.sum() * 30 * G).round().astype(int), 2)


def run_dispersion_cell(G: int, kind: str, B: int) -> Dict[str, object]:
    rejections = {"cr1": 0, "cr3": 0}
    effective: List[float] = []
    for rep in range(B):
        rng = np.random.default_rng(900_000 + rep + 13 * G)
        n_g = dispersed_sizes(G, kind, rng)
        g = np.repeat(np.arange(G), n_g)
        d_cluster = np.zeros(G, dtype=int)
        d_cluster[rng.choice(G, size=G // 2, replace=False)] = 1
        a = rng.normal(scale=np.sqrt(0.3), size=G)
        e = rng.normal(scale=np.sqrt(0.7), size=g.size)
        df = pd.DataFrame({"y": 1.0 + a[g] + e, "d": d_cluster[g], "g": g})
        effective.append(float(n_g.sum() ** 2 / (n_g**2).sum()))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            p1 = sp.regress("y ~ d", df, cluster="g").pvalues["d"]
            p3 = sp.regress("y ~ d", df, vce="cr3", cluster="g").pvalues["d"]
        rejections["cr1"] += int(p1 < ALPHA)
        rejections["cr3"] += int(p3 < ALPHA)
    out: Dict[str, object] = {
        "G": G,
        "sizes": kind,
        "B": B,
        "effective_clusters_mean": float(np.mean(effective)),
    }
    for m, count in rejections.items():
        rate = count / B
        out[m] = {
            "rejection_rate": rate,
            "mc_se": float(np.sqrt(rate * (1 - rate) / B)),
        }
    return out


PANEL_KINDS = ("balanced", "dominant")
PANEL_G = 40


def run_panel_cell(kind: str, B: int) -> Dict[str, object]:
    sizes = (
        [8] * PANEL_G
        if kind == "balanced"
        else [8 * (PANEL_G - 1)] + [8] * (PANEL_G - 1)
    )
    rho = 0.6
    scale = np.sqrt(1 - rho**2)
    rejections = 0
    for rep in range(B):
        rng = np.random.default_rng(5_000 + rep)
        frames = []
        for unit, T in enumerate(sizes):
            x = np.empty(T)
            e = np.empty(T)
            x[0], e[0] = rng.normal(), rng.normal()
            for t in range(1, T):
                x[t] = rho * x[t - 1] + scale * rng.normal()
                e[t] = rho * e[t - 1] + scale * rng.normal()
            frames.append(
                pd.DataFrame(
                    {"id": unit, "t": np.arange(T), "x": x, "y": rng.normal() + e}
                )
            )
        df = pd.concat(frames, ignore_index=True)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            fit = sp.panel(
                df,
                "y ~ x",
                entity="id",
                time="t",
                method="fe",
                cluster="id",
                ssc="stata",
            )
        rejections += int(float(fit.pvalues["x"]) < ALPHA)
    n = np.asarray(sizes, dtype=float)
    rate = rejections / B
    return {
        "G": PANEL_G,
        "sizes": kind,
        "B": B,
        "effective_clusters": float(n.sum() ** 2 / (n**2).sum()),
        "cr1": {
            "rejection_rate": rate,
            "mc_se": float(np.sqrt(rate * (1 - rate) / B)),
        },
    }


DID_G = 40
DID_T = 10
DID_TREATED = (1, 2, 5, 10, 20)
DID_METHODS = ("cr1", "placebo")


def draw_did(n_treated: int, seed: int) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rho = 0.5
    e = np.empty((DID_G, DID_T))
    e[:, 0] = rng.normal(size=DID_G)
    for t in range(1, DID_T):
        e[:, t] = rho * e[:, t - 1] + np.sqrt(1 - rho**2) * rng.normal(size=DID_G)
    unit = np.repeat(np.arange(DID_G), DID_T)
    period = np.tile(np.arange(1, DID_T + 1), DID_G)
    d = ((unit < n_treated) & (period >= 6)).astype(float)
    y = np.repeat(rng.normal(size=DID_G), DID_T) + 0.1 * period + e.ravel()
    return pd.DataFrame({"id": unit, "t": period, "d": d, "y": y})


def did_pvalues(df: pd.DataFrame) -> Dict[str, float]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        twfe = sp.panel(
            df, "y ~ d", entity="id", time="t", method="twoway", cluster="id"
        )
        placebo = sp.did_few_treated(df, y="y", unit="id", time="t", treat="d")
    return {"cr1": float(twfe.pvalues["d"]), "placebo": float(placebo.pvalue)}


def run_did_cell(n_treated: int, B: int) -> Dict[str, object]:
    rej: Dict[str, List[int]] = {m: [] for m in DID_METHODS}
    for rep in range(B):
        p = did_pvalues(draw_did(n_treated, 9_000_000 + 1_000 * n_treated + rep))
        for m in DID_METHODS:
            rej[m].append(int(p[m] < ALPHA))
    cell: Dict[str, object] = {"G": DID_G, "treated": str(n_treated), "B": B}
    for m in DID_METHODS:
        r = np.asarray(rej[m])
        rate = float(r.mean())
        cell[m] = {
            "rejection_rate": rate,
            "mc_se": float(np.sqrt(rate * (1 - rate) / B)),
            "prefix_rejections": int(r[:PREFIX].sum()),
        }
    return cell


def run_did_block(B: int) -> List[Dict[str, object]]:
    out = []
    for n_treated in DID_TREATED:
        cell = run_did_cell(n_treated, B)
        out.append(cell)
        print(
            f"DiD G={DID_G} treated={n_treated:2d} "
            + " ".join(
                f"{m}={cell[m]['rejection_rate']:.3f}"  # type: ignore[index]
                for m in DID_METHODS
            ),
            flush=True,
        )
    return out


def main() -> None:
    B = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    if len(sys.argv) > 2 and sys.argv[2] == "did":
        stored = json.loads(OUT.read_text(encoding="utf-8"))
        if stored["B"] != B:
            raise SystemExit(f"stored file has B={stored['B']}, not {B}")
        stored["did_few_treated"] = run_did_block(B)
        OUT.write_text(json.dumps(stored, indent=1) + "\n", encoding="utf-8")
        print(f"wrote {OUT}")
        return
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
    dispersion = []
    for G in DISPERSION_G:
        for kind in DISPERSION_KINDS:
            cell = run_dispersion_cell(G, kind, B)
            dispersion.append(cell)
            print(
                f"G={G:2d} sizes={kind:14s} effective="
                f"{cell['effective_clusters_mean']:.1f} "
                f"cr1={cell['cr1']['rejection_rate']:.3f} "  # type: ignore[index]
                f"cr3={cell['cr3']['rejection_rate']:.3f}",  # type: ignore[index]
                flush=True,
            )
    panel_cells = []
    for kind in PANEL_KINDS:
        cell = run_panel_cell(kind, B)
        panel_cells.append(cell)
        print(
            f"panel FE G={PANEL_G} sizes={kind:9s} effective="
            f"{cell['effective_clusters']:.1f} "
            f"cr1={cell['cr1']['rejection_rate']:.3f}",  # type: ignore[index]
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
        "size_dispersion": dispersion,
        "panel_fixed_effects": panel_cells,
        "did_few_treated": run_did_block(B),
    }
    OUT.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
