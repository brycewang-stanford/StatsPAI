"""Recompute the sup-F p-value response surface of Hansen (1997) by simulation.

Hansen (1997, JBES 15(1), 60-67, [@hansen1997approximate]) approximates the
asymptotic p-value function of Andrews' (1993) sup-Wald statistic,

    SupF(pi_0) = sup_{pi_0 <= tau <= 1 - pi_0} B(tau)'B(tau) / (tau (1 - tau)),

with ``B`` an ``m``-vector Brownian bridge (his eqs. 2, 5), by

    p(x | theta) = 1 - chi2(theta_0 + theta_1 x ; eta)          (eq. 8, v = 1)

and chooses ``theta = (theta_0, theta_1, eta)`` to minimise the weighted
L^8 distance between ``p(Q(q) | theta)`` and ``q`` over
``q = .001, .002, ..., .999`` (eqs. 11-12; ``w(q) = 1`` below .1,
``((.8 - q) / .7)^2`` on [.1, .8], 0 above). ``Q`` is the empirical
quantile function of the simulated statistic.

The paper prints the coefficients for pi_0 = .01, .05, .15, .25, .35 only,
to two decimals (Table 2), and says the complete table is "too many to
print". This script regenerates the complete table independently, from the
paper's description alone, as Hansen describes it: Brownian motion on a grid
of 1,000 evenly spaced points, pi_0 = .01, .03, ..., .49 and m = 1, ..., 40.
``REPS`` is 200,000, four times the paper's 50,000, which halves the Monte
Carlo error of the fitted surface. Seed and settings are fixed below; the
output is
``src/statspai/timeseries/_hansen_supf_table.py``.

Usage::

    python scripts/generate_hansen_supf_table.py            # write the table
    python scripts/generate_hansen_supf_table.py --check    # compare to Table 2

It takes about half an hour and 1 GB of memory.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
from scipy import optimize, stats

SEED = 19970101
REPS = 200_000
GRID = 1_000
M_MAX = 40
PI0 = np.round(np.arange(0.49, 0.0, -0.02), 2)  # .49, .47, ..., .01 (25 rows)
BATCH = 250
OUT = (
    Path(__file__).resolve().parents[1]
    / "src"
    / "statspai"
    / "timeseries"
    / "_hansen_supf_table.py"
)

# Hansen (1997), Table 2: (theta_0, theta_1, eta) printed for these pi_0 and
# m, read from the published article (two decimals; eta one decimal).
TABLE2 = {
    0.01: {
        1: (-1.79, 1.17, 4.5),
        5: (-6.39, 1.18, 9.4),
        10: (-11.01, 1.14, 13.3),
        20: (-18.99, 1.02, 17.0),
        40: (-34.24, 0.97, 24.8),
    },
    0.05: {
        1: (-1.39, 1.07, 3.6),
        5: (-4.84, 1.15, 9.3),
        10: (-8.84, 1.11, 13.2),
        20: (-16.09, 0.99, 17.0),
        40: (-29.24, 0.98, 28.0),
    },
    0.15: {
        1: (-0.99, 1.02, 3.0),
        5: (-3.46, 1.07, 8.3),
        10: (-6.47, 1.06, 12.8),
        20: (-12.14, 0.97, 18.3),
        40: (-21.65, 1.05, 36.7),
    },
    0.25: {
        1: (-0.73, 0.98, 2.5),
        5: (-2.63, 1.02, 7.5),
        10: (-4.97, 1.01, 12.0),
        20: (-8.87, 1.00, 20.5),
        40: (-14.18, 1.07, 42.7),
    },
    0.35: {
        1: (-0.50, 0.96, 2.1),
        5: (-1.82, 1.00, 7.0),
        10: (-2.92, 1.05, 13.0),
        20: (-5.94, 1.00, 21.3),
        40: (-11.95, 0.94, 35.0),
    },
}


def simulate(seed: int = SEED, reps: int = REPS) -> np.ndarray:
    """``out[r, m - 1, i]`` = SupF(PI0[i]) for ``m`` regressors, replication r."""
    rng = np.random.default_rng(seed)
    t = np.arange(1, GRID + 1) / GRID
    denom = t[:-1] * (1.0 - t[:-1])  # tau = 1 is outside every window
    lo = np.rint(PI0 * GRID).astype(int)  # tau_j = j / GRID, j = lo .. GRID - lo
    hi = GRID - lo
    out = np.empty(
        (reps, M_MAX, PI0.size), dtype=np.float32
    )  # quantiles need ~7 digits
    for start in range(0, reps, BATCH):
        b = min(BATCH, reps - start)
        w = np.cumsum(rng.standard_normal((b, M_MAX, GRID)), axis=2) / np.sqrt(GRID)
        bridge = w - t * w[:, :, -1:]
        f = np.cumsum(bridge[:, :, :-1] ** 2, axis=1) / denom  # sum over first m
        for i in range(PI0.size):
            out[start : start + b, :, i] = f[:, :, lo[i] - 1 : hi[i]].max(axis=2)
    return out


Q = np.arange(1, 1000) / 1000.0
W = np.where(Q <= 0.1, 1.0, np.where(Q <= 0.8, ((0.8 - Q) / 0.7) ** 2, 0.0))
KEEP = W > 0


def _loss(theta: np.ndarray, x: np.ndarray) -> float:
    t0, t1, eta = theta
    if eta <= 0:
        return np.inf
    p = stats.chi2.sf(t0 + t1 * x, eta)
    return float(np.sum(np.abs(p - Q[KEEP]) ** 8 * W[KEEP]) ** 0.125)


def fit(sims: np.ndarray) -> np.ndarray:
    """``beta[m - 1, i] = (theta_0, theta_1, eta)`` for ``PI0[i]``."""
    beta = np.empty((M_MAX, PI0.size, 3))
    for m in range(1, M_MAX + 1):
        start = np.array([0.0, 1.0, float(m)])  # pi_0 = .5 is chi2(m) exactly
        for i in range(PI0.size):
            x = np.quantile(sims[:, m - 1, i], 1.0 - Q[KEEP])
            res = optimize.minimize(
                _loss,
                start,
                args=(x,),
                method="Nelder-Mead",
                options={
                    "xatol": 1e-10,
                    "fatol": 1e-14,
                    "maxiter": 20_000,
                    "maxfev": 40_000,
                },
            )
            beta[m - 1, i] = res.x
            start = res.x  # warm start for the next (smaller) pi_0
    return beta


def write(beta: np.ndarray) -> None:
    lines = [
        '"""Sup-F p-value response surface, recomputed by simulation.',
        "",
        "Generated by ``scripts/generate_hansen_supf_table.py`` (seed",
        f"{SEED}, {REPS:,} replications, {GRID:,}-point grid) from the method of",
        "Hansen (1997) [@hansen1997approximate], eqs. (5), (8), (11)-(12). Do not",
        "edit by hand.",
        "",
        "``HANSEN_SUP_BETA[m]`` (m = 1..40): 25 rows ``(theta_0, theta_1, eta)``",
        "for pi_0 = .49, .47, ..., .01; the p-value of a Wald-scale statistic",
        "``x`` is ``chi2.sf(theta_0 + theta_1 x, eta)``.",
        '"""',
        "",
        "HANSEN_SUP_BETA = {",
    ]
    for m in range(1, M_MAX + 1):
        lines.append(f"    {m}: (")
        for row in beta[m - 1]:
            lines.append("        (" + ", ".join(f"{v:.8f}" for v in row) + "),")
        lines.append("    ),")
    lines.append("}")
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")


def check(beta: np.ndarray) -> float:
    """Worst |p(ours) - p(Table 2)| over the printed entries, x on a grid."""
    worst = 0.0
    for pi0, rows in TABLE2.items():
        i = int(np.flatnonzero(np.isclose(PI0, pi0))[0])
        for m, (a0, a1, eta) in rows.items():
            b0, b1, e = beta[m - 1, i]
            # x over the upper 50% .. 0.1% of the printed distribution
            x = (
                stats.chi2.isf(np.array([0.5, 0.2, 0.1, 0.05, 0.01, 0.001]), eta) - a0
            ) / a1
            gap = np.max(
                np.abs(stats.chi2.sf(b0 + b1 * x, e) - stats.chi2.sf(a0 + a1 * x, eta))
            )
            worst = max(worst, gap)
            print(
                f"pi0={pi0:.2f} m={m:2d} ours=({b0:7.2f},{b1:5.2f},{e:5.1f}) "
                f"Table2=({a0:7.2f},{a1:5.2f},{eta:5.1f}) max|dp|={gap:.4f}"
            )
    return worst


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--check", action="store_true", help="compare the committed table with Table 2"
    )
    ap.add_argument("--reps", type=int, default=REPS)
    args = ap.parse_args()
    if args.check:
        sys.path.insert(0, str(OUT.parents[2]))
        from statspai.timeseries._hansen_supf_table import HANSEN_SUP_BETA

        beta = np.array([HANSEN_SUP_BETA[m] for m in range(1, M_MAX + 1)])
        print(f"worst p-value gap to Table 2: {check(beta):.4f}")
        return 0
    sims = simulate(reps=args.reps)
    beta = fit(sims)
    write(beta)
    print(f"wrote {OUT}; worst p-value gap to Table 2: {check(beta):.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
