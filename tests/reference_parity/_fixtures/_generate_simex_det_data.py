"""Data and simulation draws for tests/reference_parity/test_simex_r_parity.py.

Both R ``simex`` 1.8 and ``sp.simex`` are fed the same pseudo-errors and the
same re-misclassified vectors, so the comparison is deterministic:

* ``simex_det_data.csv``: ``w = x + N(0, 0.5^2)`` with linear ``y``, logit
  ``d`` and Poisson ``cnt`` outcomes; a binary ``xb`` observed as ``xs``
  through ``P_x = [[0.85, 0.10], [0.15, 0.90]]`` with outcomes ``yb``
  (linear) and ``db`` (logit); a logit outcome ``db`` observed as ``dm``
  through ``P_y = [[0.90, 0.20], [0.10, 0.80]]``. Columns of ``P`` are the
  true values, as in R ``mcsimex``.
* ``simex_det_draws.csv.gz``: standard normal pseudo-errors, one column per
  simulated data set in the order R draws them (lambda outer, B inner),
  rounded to 6 decimals so both languages parse the same doubles.
* ``simex_det_mc.csv.gz``: the re-misclassified ``xs`` (columns ``x_*``)
  and ``dm`` (columns ``y_*``) for each simulated data set, drawn from
  ``P ** lambda`` with ``statspai.measurement_error._simex_misclass``.
* ``simex_det_factor_data.csv`` / ``simex_det_factor_mc.csv.gz``: a
  three-level factor ``xf`` observed through the 3 x 3 matrix ``P_F`` with
  a linear outcome ``yf`` and a logit outcome ``df``, and its
  re-misclassified codes (columns ``f*``). A separate random stream, so
  the files above did not change when the factor case was added.

Run from the repository root with ``PYTHONPATH=src``.
"""

from pathlib import Path

import numpy as np
import pandas as pd

from statspai.measurement_error import _simex_misclass as mc

HERE = Path(__file__).parent
GZ = {"method": "gzip", "mtime": 0}  # byte-reproducible archives
N, B = 300, 20
LAMBDAS = (0.5, 1.0, 1.5, 2.0)
P_X = np.array([[0.85, 0.10], [0.15, 0.90]])
P_Y = np.array([[0.90, 0.20], [0.10, 0.80]])
P_F = np.array([[0.80, 0.10, 0.05], [0.10, 0.80, 0.15], [0.10, 0.10, 0.80]])


def _flip(rng, true, P):
    return mc.misclassify(true.astype(float), P, rng).astype(int)


def main() -> None:
    rng = np.random.default_rng(20261029)
    x = rng.normal(size=N)
    z = rng.normal(size=N)
    xb = (rng.uniform(size=N) < 0.4).astype(int)
    db = (rng.uniform(size=N) < 1 / (1 + np.exp(-(-0.3 + 1.2 * xb + 0.5 * z)))).astype(
        int
    )
    df = pd.DataFrame(
        {
            "y": 1.0 + 2.0 * x + 0.5 * z + rng.normal(size=N),
            "d": (rng.uniform(size=N) < 1 / (1 + np.exp(-(-0.5 + x + 0.5 * z)))).astype(
                int
            ),
            "cnt": rng.poisson(np.exp(0.3 + 0.4 * x + 0.2 * z)),
            "w": x + rng.normal(scale=0.5, size=N),
            "z": z,
            "xs": _flip(rng, xb, P_X),
            "yb": 1.0 + 1.5 * xb + 0.5 * z + rng.normal(size=N),
            "db": db,
            "dm": _flip(rng, db, P_Y),
        }
    )
    df.to_csv(HERE / "simex_det_data.csv", index=False, float_format="%.8f")

    n_sets = B * len(LAMBDAS)
    draws = pd.DataFrame(
        rng.standard_normal((N, n_sets)), columns=[f"e{i}" for i in range(n_sets)]
    )
    draws.to_csv(
        HERE / "simex_det_draws.csv.gz",
        index=False,
        float_format="%.6f",
        compression=GZ,
    )

    cols = {}
    for name, var, P in (("x", "xs", P_X), ("y", "dm", P_Y)):
        i = 0
        for lam in LAMBDAS:
            P_lam = mc.mc_power(P, lam)
            for _ in range(B):
                cols[f"{name}{i}"] = mc.misclassify(
                    df[var].to_numpy(dtype=float), P_lam, rng
                ).astype(int)
                i += 1
    pd.DataFrame(cols).to_csv(HERE / "simex_det_mc.csv.gz", index=False, compression=GZ)
    factor_case()


def factor_case() -> None:
    rng = np.random.default_rng(20261030)
    xt = rng.integers(0, 3, size=N)
    z = rng.normal(size=N)
    eta = -0.4 + 0.6 * (xt == 1) + 1.3 * (xt == 2) + 0.5 * z
    df = pd.DataFrame(
        {
            "xf": mc.misclassify(xt.astype(float), P_F, rng).astype(int),
            "z": z,
            "yf": 1.0
            + 0.5 * (xt == 1)
            + 1.2 * (xt == 2)
            + 0.5 * z
            + rng.normal(size=N),
            "df": (rng.uniform(size=N) < 1 / (1 + np.exp(-eta))).astype(int),
        }
    )
    df.to_csv(HERE / "simex_det_factor_data.csv", index=False, float_format="%.8f")
    cols = {}
    i = 0
    for lam in LAMBDAS:
        P_lam = mc.mc_power(P_F, lam)
        for _ in range(B):
            cols[f"f{i}"] = mc.misclassify(
                df["xf"].to_numpy(dtype=float), P_lam, rng
            ).astype(int)
            i += 1
    pd.DataFrame(cols).to_csv(
        HERE / "simex_det_factor_mc.csv.gz", index=False, compression=GZ
    )


if __name__ == "__main__":
    main()
