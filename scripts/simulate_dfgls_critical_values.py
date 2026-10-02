"""Finite-sample critical values of the DF-GLS unit-root statistic.

The DF-GLS statistic of Elliott, Rothenberg and Stock (1996) is a
Dickey-Fuller t statistic on GLS-detrended data. Its null distribution
depends on the sample size ``T`` and, in finite samples, on the number of
lagged differences ``k`` in the test regression: with only a constant the
5% point is -1.95 asymptotically but about -2.27 at ``T = 50``.

This script simulates the null distribution (driftless Gaussian random walk)
on a grid of ``(T, k)`` and fits, for each deterministic specification and
level, the response surface

    cv(T, k) = b0 + b1/T + b2/T**2 + b3/T**3
                  + c1*(k/T) + c2*(k/T)**2 + c3*(k/T**2)

over ``T >= 25``. The coefficients it prints are the ``DFGLS_SURFACE`` table
in ``src/statspai/timeseries/_critvals.py``. The simulated quantiles of the
committed run are in
``tests/reference_parity/_fixtures/dfgls_null_quantiles.json``;
``tests/test_unitroot.py`` checks that the table is the fit to that file,
and checks the table against a fresh simulation.

Usage::

    # full run, about 90 minutes; keeps the simulated quantiles
    python scripts/simulate_dfgls_critical_values.py --points q.json
    # quick look
    python scripts/simulate_dfgls_critical_values.py --reps 20000
    # refit quantiles saved earlier, no simulation
    python scripts/simulate_dfgls_critical_values.py --refit q.json

The statistic is computed exactly as ``sp.unitroot(test="dfgls")`` does:
quasi-difference with ``a = 1 + c/T`` (``c = -7`` constant, ``-13.5``
trend), regress on the quasi-differenced deterministics, then run the
no-deterministic ADF regression with ``k`` lags on the ``T - 1 - k`` usable
observations.
"""

from __future__ import annotations

import argparse
import json
import sys

import numpy as np

T_GRID = (20, 25, 30, 40, 50, 60, 80, 100, 150, 200, 300, 500, 1000)
K_GRID = tuple(range(0, 13))
LEVELS = (0.01, 0.05, 0.10)
FIT_MIN_T = 25
CBAR = {"c": -7.0, "ct": -13.5}


def dfgls_batch(y: np.ndarray, trend: str, lags: int) -> np.ndarray:
    """DF-GLS statistics of each row of ``y`` (shape ``(R, T)``)."""
    R, T = y.shape
    a = 1.0 + CBAR[trend] / T
    z = np.ones((T, 1))
    if trend == "ct":
        z = np.column_stack([np.ones(T), np.arange(1.0, T + 1.0)])
    yq = np.concatenate([y[:, :1], y[:, 1:] - a * y[:, :-1]], axis=1)
    zq = np.vstack([z[:1], z[1:] - a * z[:-1]])
    delta = np.linalg.solve(zq.T @ zq, zq.T @ yq.T).T  # (R, q)
    yd = y - delta @ z.T
    dy = np.diff(yd, axis=1)  # (R, T-1)
    n = T - 1 - lags
    Y = dy[:, lags:]
    cols = [yd[:, lags:-1]] + [dy[:, lags - j : T - 1 - j] for j in range(1, lags + 1)]
    X = np.stack(cols, axis=2)  # (R, n, 1 + lags)
    XtX = np.einsum("rni,rnj->rij", X, X)
    XtY = np.einsum("rni,rn->ri", X, Y)
    beta = np.linalg.solve(XtX, XtY[..., None])[..., 0]
    resid = Y - np.einsum("rni,ri->rn", X, beta)
    s2 = (resid**2).sum(axis=1) / (n - X.shape[2])
    v00 = np.linalg.inv(XtX)[:, 0, 0]
    return beta[:, 0] / np.sqrt(s2 * v00)


def usable(T: int, k: int) -> bool:
    # keep at least four observations per estimated coefficient
    return (T - 1 - k) >= 4 * (k + 1) and k <= T // 4


def simulate(reps: int, seed: int, batch: int = 10000) -> dict:
    rng = np.random.default_rng(seed)
    out = {}
    for trend in ("c", "ct"):
        for T in T_GRID:
            ks = [k for k in K_GRID if usable(T, k)]
            stats = {k: [] for k in ks}
            done = 0
            while done < reps:
                r = min(batch, reps - done)
                y = np.cumsum(rng.standard_normal((r, T)), axis=1)
                for k in ks:
                    stats[k].append(dfgls_batch(y, trend, k))
                done += r
            for k in ks:
                q = np.quantile(np.concatenate(stats[k]), LEVELS)
                out[(trend, T, k)] = tuple(float(v) for v in q)
            print(f"  simulated {trend} T={T}", file=sys.stderr)
    return out


def design(T: np.ndarray, k: np.ndarray) -> np.ndarray:
    return np.column_stack(
        [np.ones_like(T), 1 / T, 1 / T**2, 1 / T**3, k / T, (k / T) ** 2, k / T**2]
    )


def fit(points: dict) -> dict:
    table = {}
    for trend in ("c", "ct"):
        keys = [key for key in points if key[0] == trend and key[1] >= FIT_MIN_T]
        T = np.array([key[1] for key in keys], dtype=float)
        k = np.array([key[2] for key in keys], dtype=float)
        A = design(T, k)
        for j, level in enumerate(LEVELS):
            cv = np.array([points[key][j] for key in keys])
            coef, *_ = np.linalg.lstsq(A, cv, rcond=None)
            err = cv - A @ coef
            table[(trend, int(round(level * 100)))] = {
                "coef": [float(f"{c:.6g}") for c in coef],
                "max_abs_error": round(float(np.abs(err).max()), 4),
                "rmse": round(float(np.sqrt((err**2).mean())), 4),
            }
    return table


def load_points(raw: dict) -> dict:
    out = {}
    for key, value in raw.items():
        trend, T, k = key.split("|")
        out[(trend, int(T), int(k))] = tuple(value)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--reps", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=19960701)
    ap.add_argument("--points", help="write the simulated quantiles here (JSON)")
    ap.add_argument("--refit", help="fit quantiles saved by --points; no simulation")
    args = ap.parse_args()
    if args.refit:
        with open(args.refit, encoding="utf-8") as fh:
            points = load_points(json.load(fh))
    else:
        points = simulate(args.reps, args.seed)
    table = fit(points)
    print("DFGLS_SURFACE = {")
    for (trend, level), row in table.items():
        print(
            f'    ("{trend}", {level}): {tuple(row["coef"])},'
            f'  # rmse {row["rmse"]}, max |err| {row["max_abs_error"]}'
        )
    print("}")
    if args.points:
        with open(args.points, "w", encoding="utf-8") as fh:
            json.dump({f"{k[0]}|{k[1]}|{k[2]}": v for k, v in points.items()}, fh)


if __name__ == "__main__":
    main()
