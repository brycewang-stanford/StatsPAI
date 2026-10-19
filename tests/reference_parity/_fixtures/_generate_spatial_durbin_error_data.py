"""Data for the SDEM / GNS fixture (Stata spregress, ml).

Sample 1: 8 x 10 rook grid (symmetric contiguity). Sample 2: 70 random
points, 4 nearest neighbours (asymmetric weights). Columns w1..wN hold the
binary weights matrix row by row (Stata spmatrix fromdata, normalize(row)).

    python tests/reference_parity/_fixtures/_generate_spatial_durbin_error_data.py
"""

import pathlib

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).parent


def simulate(A, rng, rho, lam):
    n = len(A)
    W = A / A.sum(1, keepdims=True)
    x1 = rng.normal(size=n)
    x2 = rng.normal(size=n)
    X = np.column_stack([np.ones(n), x1, x2])
    u = np.linalg.solve(np.eye(n) - lam * W, rng.normal(scale=0.8, size=n))
    rhs = X @ [1, 0.8, -0.5] + W @ np.column_stack([x1, x2]) @ [0.4, 0.3] + u
    y = np.linalg.solve(np.eye(n) - rho * W, rhs)
    df = pd.DataFrame({"id": np.arange(1, n + 1), "y": y, "x1": x1, "x2": x2})
    for j in range(n):
        df[f"w{j + 1}"] = A[:, j]
    return df


def main():
    rng = np.random.default_rng(20261019)
    R, C = 8, 10
    n = R * C
    A = np.zeros((n, n))
    for r in range(R):
        for c in range(C):
            for dr, dc in ((1, 0), (-1, 0), (0, 1), (0, -1)):
                if 0 <= r + dr < R and 0 <= c + dc < C:
                    A[r * C + c, (r + dr) * C + c + dc] = 1
    simulate(A, rng, 0.35, 0.3).to_csv(
        HERE / "spatial_durbin_error_1.csv", index=False, float_format="%.17g"
    )
    m = 70
    pts = rng.random((m, 2))
    d = np.linalg.norm(pts[:, None] - pts[None], axis=2)
    np.fill_diagonal(d, np.inf)
    B = np.zeros((m, m))
    for i in range(m):
        B[i, np.argsort(d[i])[:4]] = 1
    simulate(B, rng, 0.2, 0.45).to_csv(
        HERE / "spatial_durbin_error_2.csv", index=False, float_format="%.17g"
    )


if __name__ == "__main__":
    main()
