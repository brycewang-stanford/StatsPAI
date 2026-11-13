"""Data for the ``sp.dml_sensitivity(benchmark_covariates=)`` and one-sided
IIVM references.

Run with StatsPAI importable::

    python tests/reference_parity/_fixtures/_generate_dml_benchmark_data.py

Writes ``dml_benchmark_data.csv``: the Lalonde sample shipped with StatsPAI
(``sp.datasets.nsw_lalonde()``, 614 rows) plus

* ``fold``: a fixed five-fold partition, so both packages cross-fit on the
  same rows;
* ``z``: a binary instrument drawn independently of everything;
* ``dz``: a treatment with one-sided non-compliance, ``dz = z * 1{u < 0.6 +
  0.2 black}``: nobody with ``z = 0`` is treated, which is the case in which
  the first-stage probability ``P(D = 1 | Z = 0, X)`` is exactly zero;
* ``yz``: an outcome for that design, ``re78 + 1500 dz``.

``_generate_dml_benchmark_doubleml.py`` reads the CSV and writes the
reference numbers.
"""

from __future__ import annotations

import pathlib

import numpy as np

import statspai as sp

HERE = pathlib.Path(__file__).parent


def main() -> None:
    df = sp.datasets.nsw_lalonde().drop(columns=["race"]).reset_index(drop=True)
    n = len(df)
    rng = np.random.default_rng(20261113)
    df["fold"] = rng.permutation(n) % 5
    df["z"] = rng.integers(0, 2, size=n)
    u = rng.uniform(size=n)
    df["dz"] = (df["z"] * (u < 0.6 + 0.2 * df["black"])).astype(int)
    df["yz"] = df["re78"] + 1500.0 * df["dz"]
    df.to_csv(HERE / "dml_benchmark_data.csv", index=False, float_format="%.17g")
    print("wrote dml_benchmark_data.csv", df.shape)


if __name__ == "__main__":
    main()
