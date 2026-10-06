"""Simulated linear-Gaussian SEMs for the ``sp.pc_algorithm`` / pcalg parity.

Writes ``pc_pcalg_data.csv``: 24 data sets stacked in one file (column
``case``; a case with fewer than 11 variables leaves the rest empty). The
graphs are random DAGs of 5, 8 or 11 nodes at two densities, the samples are
small (120 or 400 rows) so the conditional independence tests make mistakes
and colliders conflict -- the cases in which a wrong search order or a wrong
conflict rule shows up.

Run from the repository root, then run ``_generate_pc_pcalg.R``:

    python tests/reference_parity/_fixtures/_generate_pc_pcalg_data.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
P_MAX = 11


def main() -> None:
    frames = []
    case = 0
    for p in (5, 8, 11):
        for n in (120, 400):
            for density in (0.25, 0.5):
                for _ in range(2):
                    rng = np.random.default_rng(20261006 + case)
                    present = rng.random((p, p)) < density
                    size = rng.uniform(0.3, 1.2, (p, p))
                    sign = rng.choice([-1.0, 1.0], (p, p))
                    B = np.triu(present * size * sign, 1)
                    X = np.zeros((n, p))
                    for j in range(p):
                        noise = rng.standard_normal(n) * rng.uniform(0.5, 1.5)
                        X[:, j] = X @ B[:, j] + noise
                    # Shuffle the columns so the causal order is not the
                    # column order.
                    X = np.round(X[:, rng.permutation(p)], 3)
                    df = pd.DataFrame(X, columns=[f"v{i}" for i in range(p)])
                    df.insert(0, "case", case)
                    frames.append(df)
                    case += 1
    out = pd.concat(frames, ignore_index=True)
    out = out[["case"] + [f"v{i}" for i in range(P_MAX)]]
    out.to_csv(HERE / "pc_pcalg_data.csv", index=False)
    print(f"{case} cases, {len(out)} rows")


if __name__ == "__main__":
    main()
