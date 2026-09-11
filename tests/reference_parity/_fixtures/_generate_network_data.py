"""Write the network parity fixture CSVs (deterministic).

Run from the repository root:

    python tests/reference_parity/_fixtures/_generate_network_data.py
    Rscript tests/reference_parity/_fixtures/_generate_network_R.R

The karate and Florentine adjacencies are NOT written here: the R generator
writes them from igraph's ``make_graph("Zachary")`` and ergm's
``flomarriage``, so the test compares StatsPAI's bundled datasets against
independent copies rather than against themselves.
"""

from __future__ import annotations

import itertools
import pathlib

import numpy as np
import pandas as pd

OUT = pathlib.Path(__file__).parent


def _write(name: str, arr: np.ndarray, fmt: str = "%d") -> None:
    np.savetxt(OUT / name, arr, fmt=fmt, delimiter=",")


def main() -> None:
    # Directed graph: sparse random arcs plus a ring so it is weakly connected.
    rng = np.random.default_rng(20260912)
    n = 40
    D = (rng.random((n, n)) < 0.08).astype(int)
    np.fill_diagonal(D, 0)
    for i in range(n):
        D[i, (i + 1) % n] = 1
    _write("network_directed.csv", D)

    # Disconnected undirected graph: three connected blocks and three isolates,
    # the case the Wasserman-Faust closeness correction exists for.
    rng = np.random.default_rng(55)
    n = 30
    A = np.zeros((n, n), dtype=int)
    for block in (range(0, 12), range(12, 22), range(22, 27)):
        b = list(block)
        for a, c in itertools.combinations(b, 2):
            if rng.random() < 0.35:
                A[a, c] = A[c, a] = 1
        for k in range(len(b) - 1):
            A[b[k], b[k + 1]] = A[b[k + 1], b[k]] = 1
    _write("network_disconnected.csv", A)

    # Node attributes for the karate ERGM terms.
    rng = np.random.default_rng(3434)
    pd.DataFrame(
        {"grp": rng.integers(0, 2, 34), "x": np.round(rng.normal(size=34), 6)}
    ).to_csv(OUT / "network_karate_attrs.csv", index=False)

    # QAP matrices for netlm / netlogit.
    rng = np.random.default_rng(99)
    n = 25
    X1 = rng.normal(size=(n, n))
    X2 = (rng.random((n, n)) < 0.3).astype(float)
    np.fill_diagonal(X1, 0)
    np.fill_diagonal(X2, 0)
    Y = 0.5 + 1.2 * X1 - 0.8 * X2 + rng.normal(size=(n, n))
    np.fill_diagonal(Y, 0)
    Yb = (Y > np.median(Y)).astype(float)
    np.fill_diagonal(Yb, 0)
    for name, M in (("Y", Y), ("Yb", Yb), ("X1", X1), ("X2", X2)):
        _write(f"network_qap_{name}.csv", M, fmt="%.17g")

    # Dyadic data with node effects, undirected (one row per pair) and
    # directed (both orientations): the directed file is the case that
    # separates the ASA indicator from a shared-node count.
    rng = np.random.default_rng(8)
    n = 20
    a = rng.normal(size=n)
    for tag, pairs in (
        ("und", itertools.combinations(range(n), 2)),
        ("dir", itertools.permutations(range(n), 2)),
    ):
        rows = []
        for i, j in pairs:
            x = rng.normal()
            rows.append((i, j, x, 1 + 0.5 * x + a[i] + a[j] + rng.normal()))
        df = pd.DataFrame(rows, columns=["i", "j", "x", "y"])
        df["pair"] = [f"{min(p, q)}_{max(p, q)}" for p, q in zip(df.i, df.j)]
        df.to_csv(OUT / f"network_dyad_{tag}.csv", index=False, float_format="%.17g")


if __name__ == "__main__":
    main()
