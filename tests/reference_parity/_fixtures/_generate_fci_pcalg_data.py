"""Linear-Gaussian data with latent common causes, for the FCI parity test.

Writes ``fci_latent_data.csv``: 12 data sets stacked (column ``case``). Each
is a random DAG on 9 or 11 nodes of which two or three root-side nodes are
hidden after the data are drawn, so the observed variables have latent
confounders. 300 or 1000 rows.

Run from the repository root, then ``_generate_fci_pcalg.R``:

    python tests/reference_parity/_fixtures/_generate_fci_pcalg_data.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).parent
P_MAX = 8


def main() -> None:
    frames = []
    case = 0
    for p_total, n_hidden in ((9, 2), (11, 3)):
        for n in (300, 1000):
            for _ in range(3):
                rng = np.random.default_rng(20261007 + case)
                present = rng.random((p_total, p_total)) < 0.35
                size = rng.uniform(0.4, 1.2, (p_total, p_total))
                sign = rng.choice([-1.0, 1.0], (p_total, p_total))
                B = np.triu(present * size * sign, 1)
                X = np.zeros((n, p_total))
                for j in range(p_total):
                    X[:, j] = X @ B[:, j] + rng.standard_normal(n)
                # hide nodes early in the causal order: they are the ones
                # with children, hence the common causes
                hidden = rng.choice(p_total // 2, n_hidden, replace=False)
                keep = [j for j in range(p_total) if j not in set(hidden)]
                obs = np.round(X[:, rng.permutation(keep)], 3)
                df = pd.DataFrame(obs, columns=[f"v{i}" for i in range(obs.shape[1])])
                df.insert(0, "case", case)
                frames.append(df)
                case += 1
    out = pd.concat(frames, ignore_index=True)
    out = out[["case"] + [f"v{i}" for i in range(P_MAX)]]
    out.to_csv(HERE / "fci_latent_data.csv", index=False)
    print(f"{case} cases, {len(out)} rows")


if __name__ == "__main__":
    main()
