"""Panel for the ``sp.hdfe_ols`` nested-FE / collinear-regressor fixture.

100 units in 10 regions over 10 years. ``regionXyear`` nests ``year``;
``v_exact`` is spanned by the region x year effects and ``v_near`` is the same
column plus 1e-7 noise (the float32 storage case of the Web of Power data).

    python tests/reference_parity/_fixtures/_generate_hdfe_nested_collinear_data.py
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

OUT = pathlib.Path(__file__).parent / "hdfe_nested_collinear.csv"


def main() -> None:
    rng = np.random.default_rng(20260929)
    d = pd.DataFrame(
        [(c, c // 10, t) for c in range(100) for t in range(10)],
        columns=["unit", "region", "year"],
    )
    d["regionXyear"] = d.region * 100 + d.year
    post = (d.year >= 5).astype(float)
    d["x"] = rng.normal(size=100)[d.unit] * post + rng.normal(size=len(d)) * 0.1
    d["x2"] = rng.normal(size=len(d))
    d["v_exact"] = rng.normal(size=10)[d.region] * post
    d["v_near"] = d.v_exact + rng.normal(size=len(d)) * 1e-7
    d["y"] = 0.5 * d.x + 0.3 * d.x2 + rng.normal(size=len(d))
    d.to_csv(OUT, index=False, float_format="%.17g")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
