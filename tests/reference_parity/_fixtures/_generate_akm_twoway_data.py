"""Synthetic worker-firm panel for ``sp.akm`` reference checks.

Writes ``akm_twoway.csv``: a person-year panel with sorting (high-alpha
workers move to high-psi firms more often), a year effect, one covariate,
heteroskedastic errors, plus three structures the connected-set logic must
handle:

* workers observed once (stayers with one observation: P_ii = 1);
* a two-firm island linked by its own movers but to no other firm (a
  second connected component, dropped by ``connected='largest'``);
* a firm reached by a single mover with a single observation there (a
  bridge: in the largest connected set, not in the leave-one-out set).

The same file feeds the Stata reference (``_generate_akm_twoway_Stata.do``:
``reghdfe ..., absorb(wfe=worker ffe=firm)`` and ``felsdvreg``) and the
pytwoway reference (``_generate_akm_twoway_pytwoway.py``).

Run from the repository root::

    python tests/reference_parity/_fixtures/_generate_akm_twoway_data.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).with_name("akm_twoway.csv")


def make(seed: int = 20261028) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    n_workers, n_firms, n_years = 360, 30, 5
    psi = rng.normal(0.0, 0.3, n_firms)
    order = np.argsort(psi)
    alpha = rng.normal(0.0, 0.4, n_workers)
    year_fx = np.linspace(0.0, 0.2, n_years)
    rows = []
    for w in range(n_workers):
        # sorting: the worker's alpha rank tilts the firm draw
        q = (np.argsort(np.argsort(alpha))[w] + 0.5) / n_workers
        centre = q * (n_firms - 1)
        p = np.exp(-0.5 * ((np.arange(n_firms) - centre) / 6.0) ** 2)
        p /= p.sum()
        firm = order[rng.choice(n_firms, p=p)]
        for t in range(n_years):
            if t > 0 and rng.uniform() < 0.25:
                firm = order[rng.choice(n_firms, p=p)]
            rows.append((w, int(firm), t))
    df = pd.DataFrame(rows, columns=["worker", "firm", "year"])
    # single-observation workers: keep only the first year of 15 workers
    single = rng.choice(n_workers, 15, replace=False)
    df = df[~(df.worker.isin(single) & (df.year > 0))]
    # island: firms 100 and 101, linked by four movers of their own
    island = []
    for k in range(4):
        w = 1000 + k
        for t in range(n_years):
            island.append((w, 100 + (t + k) % 2, t))
    # bridge firm 200: reached by worker 2000 once (a mover with one year
    # there) and by two stayers with two years each
    bridge = [(2000, 200, 0), (2000, int(order[0]), 1), (2000, int(order[0]), 2)]
    bridge += [(2001, 200, 0), (2001, 200, 1), (2002, 200, 3), (2002, 200, 4)]
    extra = pd.DataFrame(island + bridge, columns=["worker", "firm", "year"])
    df = pd.concat([df, extra], ignore_index=True)
    alpha_all = {
        w: alpha[w] if w < n_workers else rng.normal(0, 0.4) for w in df.worker.unique()
    }
    psi_all = {
        f: psi[f] if f < n_firms else rng.normal(0, 0.3) for f in df.firm.unique()
    }
    a = df.worker.map(alpha_all).to_numpy()
    f = df.firm.map(psi_all).to_numpy()
    x = rng.normal(0.0, 1.0, len(df)) + 0.5 * a
    sd = 0.15 + 0.15 * np.abs(f)  # heteroskedastic in the firm effect
    df["x"] = np.round(x, 6)
    df["lnw"] = np.round(
        1.0 + a + f + year_fx[df.year.to_numpy()] + 0.3 * x + rng.normal(0, sd), 6
    )
    return df.sort_values(["worker", "year"]).reset_index(drop=True)


if __name__ == "__main__":
    make().to_csv(OUT, index=False)
    print(f"wrote {OUT}")
