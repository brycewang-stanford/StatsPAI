"""Random graphs and queries for the ``sp.dag`` / do-calculus comparison
with R ``dagitty``.

Writes ``dag_dagitty_cases.json``: 120 acyclic graphs on 4 to 9 named nodes
(edge probability 0.2 to 0.6), a third of them with one or two declared
latent nodes and a third with one to three bidirected edges. Each graph
carries an exposure / outcome pair, twelve d-separation queries and six
do-calculus queries (disjoint sets Y, X, Z and W, W possibly empty).
``_generate_dag_dagitty_R.R`` answers them with dagitty and writes
``dag_dagitty_R.json``, which is what the test reads; it repeats the
graphs, so ``dag_dagitty_cases.json`` is an intermediate file and is not
committed.

Run from this directory::

    python _generate_dag_dagitty_cases.py
    Rscript _generate_dag_dagitty_R.R
"""

from __future__ import annotations

import json
import pathlib

import numpy as np

HERE = pathlib.Path(__file__).parent


def make_cases(n_cases: int = 120):
    cases = []
    for k in range(n_cases):
        rng = np.random.default_rng(20261113 + k)
        p = int(rng.integers(4, 10))
        names = [f"V{i + 1}" for i in range(p)]
        order = list(rng.permutation(p))
        dens = float(rng.uniform(0.2, 0.6))
        directed = []
        for a in range(p):
            for b in range(a + 1, p):
                if rng.random() < dens:
                    directed.append([names[order[a]], names[order[b]]])
        kind = k % 3  # 0 plain, 1 latent nodes, 2 bidirected edges
        latent, bidirected = [], []
        if kind == 1:
            # keep at least four observed nodes for the do-calculus queries
            n_lat = min(int(rng.integers(1, 3)), p - 4)
            latent = sorted(str(v) for v in rng.choice(names, n_lat, replace=False))
        if kind == 2:
            pairs = {tuple(sorted(e)) for e in directed}
            free = [
                (names[a], names[b])
                for a in range(p)
                for b in range(a + 1, p)
                if (names[a], names[b]) not in pairs
            ]
            n_bi = min(len(free), int(rng.integers(1, 4)))
            for j in rng.choice(len(free), n_bi, replace=False):
                bidirected.append(list(free[int(j)]))
        observed = [v for v in names if v not in latent]
        # exposure before outcome in the causal order, both observed
        pos = {names[order[i]]: i for i in range(p)}
        obs_sorted = sorted(observed, key=pos.get)
        i = int(rng.integers(0, len(obs_sorted) - 1))
        j = int(rng.integers(i + 1, len(obs_sorted)))
        exposure, outcome = obs_sorted[i], obs_sorted[j]
        dsep = []
        for _ in range(12):
            x, y = (str(v) for v in rng.choice(observed, 2, replace=False))
            rest = [v for v in observed if v not in (x, y)]
            m = int(rng.integers(0, min(3, len(rest)) + 1))
            z = sorted(str(v) for v in rng.choice(rest, m, replace=False))
            dsep.append({"x": x, "y": y, "z": z})
        rules = []
        for _ in range(6):
            perm = [str(v) for v in rng.permutation(observed)]
            ny = 1
            nx = 1 if len(perm) < 5 or rng.random() < 0.7 else 2
            nz = 1 if len(perm) - ny - nx < 2 or rng.random() < 0.7 else 2
            left = len(perm) - ny - nx - nz
            nw = int(rng.integers(0, min(2, left) + 1)) if left > 0 else 0
            cut = np.cumsum([ny, nx, nz, nw])
            rules.append(
                {
                    "Y": sorted(perm[: cut[0]]),
                    "X": sorted(perm[cut[0] : cut[1]]),
                    "Z": sorted(perm[cut[1] : cut[2]]),
                    "W": sorted(perm[cut[2] : cut[3]]),
                }
            )
        cases.append(
            {
                "id": f"g{k:03d}",
                "nodes": names,
                "directed": directed,
                "bidirected": bidirected,
                "latent": latent,
                "exposure": exposure,
                "outcome": outcome,
                "dsep": dsep,
                "rules": rules,
            }
        )
    return cases


if __name__ == "__main__":
    out = HERE / "dag_dagitty_cases.json"
    out.write_text(json.dumps(make_cases(), indent=0) + "\n", encoding="utf-8")
    print(out)
