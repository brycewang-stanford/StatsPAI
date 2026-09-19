"""Write the inputs of tests/reference_parity/test_survival_epi_R_parity.py.

Run before ``../_generate_survival_epi_R.R`` and
``_generate_survival_epi_stata.do``. Deterministic (fixed seeds); the CSVs
are committed so the Python, R and Stata sides read identical bytes. Values
are written with at most 6 decimals so every side parses the same doubles.
"""

from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).parent


def _write(df: pd.DataFrame, name: str) -> None:
    df.to_csv(OUT / name, index=False, float_format="%.6f")


# Competing risks: two causes, a group, one continuous and one binary
# covariate. Times are rounded up to quarter units so that events of both
# causes and censorings tie with one another -- the case where the left-
# versus right-continuity of the censoring KM matters for Fine-Gray.
rng = np.random.default_rng(20260918)
n = 300
g = rng.integers(0, 2, n)
x1 = np.round(rng.normal(size=n), 6)
x2 = rng.integers(0, 2, n)
l1 = 0.25 * np.exp(0.5 * x1 + 0.4 * g)
l2 = 0.2 * np.exp(-0.3 * x2)
t1 = rng.exponential(1 / l1)
t2 = rng.exponential(1 / l2)
c = rng.uniform(0, 8, n)
t = np.minimum(np.minimum(t1, t2), c)
status = np.where(t == c, 0, np.where(t1 < t2, 1, 2))
time = np.ceil(t * 4) / 4
_write(
    pd.DataFrame(
        {"time": time, "status": status, "grp": g, "g3": g + x2, "x1": x1, "x2": x2}
    ),
    "survival_epi_cr.csv",
)

# Shared gamma frailty (variance 0.5), 40 clusters of 4-9, continuous
# times (no ties, so Efron = Breslow).
rng = np.random.default_rng(7)
rows = []
for k in range(40):
    w = rng.gamma(1 / 0.5, 0.5)
    for _ in range(int(rng.integers(4, 10))):
        a = rng.normal()
        b = int(rng.integers(0, 2))
        tt = rng.exponential(1 / (0.1 * w * np.exp(0.6 * a - 0.4 * b)))
        cc = rng.exponential(12)
        rows.append((k + 1, round(min(tt, cc), 6), int(tt <= cc), round(a, 6), b))
_write(
    pd.DataFrame(rows, columns=["cid", "time", "event", "x1", "x2"]),
    "survival_epi_frailty.csv",
)

# ROC: a continuous score and the same score rounded to one decimal
# (many ties between cases and controls).
rng = np.random.default_rng(31)
n = 240
y = rng.integers(0, 2, n)
s = 0.8 * y + rng.normal(size=n)
_write(
    pd.DataFrame({"y": y, "score": np.round(s, 6), "score_tied": np.round(s, 1)}),
    "survival_epi_roc.csv",
)

# Kernel density / local polynomial: n = 200 (so n p / 100 is an integer
# at the quartiles and Stata's percentile rule averages two order stats).
rng = np.random.default_rng(11)
n = 200
x = np.round(rng.gamma(3.0, 1.2, n), 6)
yy = np.round(np.sin(x) + 0.3 * rng.normal(size=n), 6)
_write(pd.DataFrame({"x": x, "y": yy}), "survival_epi_kd.csv")
_write(pd.DataFrame({"at": np.linspace(0.5, 8.0, 16)}), "survival_epi_at.csv")

# Rate standardisation: eight age strata of a study population, the
# standard population, and reference-population events for the SMR.
_write(
    pd.DataFrame(
        {
            "age": np.arange(1, 9),
            "events": [3, 7, 12, 25, 41, 66, 90, 118],
            "pop": [9200, 8800, 8100, 7600, 6400, 5100, 3300, 1900],
            "std_pop": [13818, 14464, 13920, 14982, 16263, 13335, 8716, 4454],
            "ref_events": [40, 71, 130, 290, 520, 840, 1210, 1520],
            "ref_pop": [150000, 142000, 139000, 131000, 118000, 96000, 70000, 41000],
        }
    ),
    "survival_epi_std.csv",
)

# Breslow-Day: four strata of [[a, b], [c, d]] (exposure x outcome) with
# odds ratios 3.6, 1.2, 2.7 and 7.9, one stratum with small cells.
_write(
    pd.DataFrame(
        {
            "stratum": [1, 2, 3, 4],
            "a": [30, 18, 12, 9],
            "b": [70, 82, 40, 3],
            "c": [12, 15, 10, 5],
            "d": [101, 82, 90, 13],
        }
    ),
    "survival_epi_bd.csv",
)

# Diagnostic 2x2 tables (tp, fn, fp, tn), one with a zero cell.
_write(
    pd.DataFrame(
        {
            "tab": [1, 2, 3],
            "tp": [80, 45, 12],
            "fn": [15, 5, 0],
            "fp": [20, 30, 7],
            "tn": [85, 120, 31],
        }
    ),
    "survival_epi_diag.csv",
)
