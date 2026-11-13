"""Data for ``test_icp_R_parity.py``: 16 designs in one long file.

Each design has two to four environments of 80 to 150 rows, a response
``y`` with two true parents ``x1`` and ``x2`` (the environments shift
``x1``, and in some designs ``x2``), a child ``x3`` of ``y`` and zero to
two pure-noise columns. In the designs with ``case % 4 == 3`` the
environments also shift ``y`` directly, so that invariance fails for
every subset. Unused columns are empty. Six decimals.

    python tests/reference_parity/_fixtures/_generate_icp_data.py
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).parent

if __name__ == "__main__":
    frames = []
    for case in range(16):
        rng = np.random.default_rng(880000 + case)
        K = 2 + case % 3
        m = int(rng.integers(80, 151))
        env = np.repeat(np.arange(K), m)
        n = env.size
        shift = rng.uniform(0.8, 2.0, K) * (np.arange(K) % 2)
        x1 = shift[env] + rng.normal(size=n) * (1 + 0.5 * (env == K - 1))
        x2 = rng.normal(size=n) + (0.7 * (env == 0) if case % 2 else 0.0)
        direct = 0.9 * (env == K - 1) if case % 4 == 3 else 0.0
        y = 1.2 * x1 - 0.7 * x2 + direct + rng.normal(size=n)
        x3 = y + 0.6 * x1 + rng.normal(size=n) * (1 + (env == 1))
        cols = {"case": case, "env": env, "y": y, "x1": x1, "x2": x2, "x3": x3}
        for j in range(case % 3):
            cols[f"x{4 + j}"] = rng.normal(size=n)
        frames.append(pd.DataFrame(cols))
    out = pd.concat(frames, ignore_index=True)
    num = [c for c in out.columns if c not in ("case", "env")]
    out[num] = out[num].round(6)
    out.to_csv(HERE / "icp_data.csv", index=False)
    print(out.groupby("case").size().to_dict())
