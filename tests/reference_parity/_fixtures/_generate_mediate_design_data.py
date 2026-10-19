"""Data for the design-based sp.mediate fixture: the weighted mediation data
(mediate_weights.csv) plus 6 strata and 48 PSUs.

    python tests/reference_parity/_fixtures/_generate_mediate_design_data.py
"""

import pathlib

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).parent


def main():
    d = pd.read_csv(HERE / "mediate_weights.csv")
    rng = np.random.default_rng(3)
    n = len(d)
    d["st"] = np.repeat(np.arange(6), n // 6)[:n]
    d["psu"] = d["st"] * 100 + rng.integers(0, 8, n)
    d.to_csv(HERE / "mediate_design.csv", index=False, float_format="%.17g")


if __name__ == "__main__":
    main()
