"""Simulated data for tests/reference_parity/test_mmhc_bnlearn_parity.py.

Twenty-four random DAGs on five to eight variables, alternately
linear-Gaussian and categorical (two or three levels), with 500, 1,000 or
3,000 rows. ``datasets()`` returns them; run as a script it writes one CSV
each into the directory given, for the R side to read:

    python tests/reference_parity/_fixtures/_generate_mmhc_data.py <dir>
    Rscript tests/reference_parity/_fixtures/_generate_mmhc_bnlearn.R <dir>

Values are rounded to four decimals before use on both sides, so the CSV
round trip changes nothing.
"""

import sys
from typing import Dict

import numpy as np
import pandas as pd

N_SETS = 24


def datasets() -> Dict[str, pd.DataFrame]:
    out = {}
    for s in range(N_SETS):
        rng = np.random.default_rng(900 + s)
        d = int(rng.integers(5, 9))
        n = int(rng.choice([500, 1000, 3000]))
        order = rng.permutation(d)
        pa = {j: [] for j in range(d)}
        for a in range(d):
            for b in range(a + 1, d):
                if rng.random() < 0.3 and len(pa[order[b]]) < 3:
                    pa[order[b]].append(order[a])
        if s % 2 == 0:
            X = np.zeros((n, d))
            for j in order:
                X[:, j] = sum(
                    rng.uniform(0.4, 1.0) * rng.choice([-1, 1]) * X[:, k] for k in pa[j]
                ) + rng.normal(size=n)
            df = pd.DataFrame(np.round(X, 4), columns=[f"v{i}" for i in range(d)])
        else:
            lev = rng.integers(2, 4, size=d)
            C = np.zeros((n, d), dtype=int)
            for j in order:
                ps = pa[j]
                ncfg = int(np.prod([lev[k] for k in ps])) if ps else 1
                tab = rng.dirichlet(np.ones(lev[j]) * 0.6, size=ncfg)
                cfg = np.zeros(n, dtype=int)
                for k in ps:
                    cfg = cfg * lev[k] + C[:, k]
                u = rng.random(n)
                C[:, j] = (
                    (u[:, None] > np.cumsum(tab[cfg], axis=1))
                    .sum(axis=1)
                    .clip(0, lev[j] - 1)
                )
            df = pd.DataFrame(
                {f"v{i}": np.array(list("abc"))[C[:, i]] for i in range(d)}
            )
        out[f"d{s:02d}"] = df
    return out


if __name__ == "__main__":
    for name, frame in datasets().items():
        frame.to_csv(f"{sys.argv[1]}/{name}.csv", index=False)
