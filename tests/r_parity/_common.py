"""Shared helpers for the StatsPAI <-> R Track A parity harness.

Each module pair (NN_<name>.py + NN_<name>.R) follows the protocol:

  1. Python side dumps the calibrated replica to ``data/<name>.csv``
     (or reads an already-dumped CSV) so R sees identical bytes.
  2. Python side writes ``results/<name>_py.json`` with point estimate,
     SE, CI, and any diagnostics relevant to the parity contract.
  3. R side reads the same CSV, runs the canonical reference, and
     writes ``results/<name>_R.json`` with the same fields.
  4. ``compare.py`` reads both JSONs, computes abs / rel diff, and
     emits ``results/parity_table.{md,tex}``.

Tolerance budget (pre-registered in NEXT-STEPS, mirroring the JSS
plan §5.2):

  * closed-form estimators (OLS, 2SLS, HDFE):  rel_diff < 1e-6
  * iterative / cross-fit (DiD, RD, SCM, DML): rel_diff < 1e-3
  * bootstrap / placebo CI half-widths:        abs_diff < 0.05 * SE
"""
from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Mapping

import pandas as pd


HERE = Path(__file__).resolve().parent
DATA_DIR = HERE / "data"
RESULTS_DIR = HERE / "results"

DATA_DIR.mkdir(parents=True, exist_ok=True)
RESULTS_DIR.mkdir(parents=True, exist_ok=True)


# Fixed seed for any cross-fit / bootstrap path so Python and R land on
# comparable Monte Carlo samples. R-side scripts mirror this with
# `set.seed(...)`.
PARITY_SEED = 42


@dataclass
class ParityRecord:
    """One row of the parity table.

    Both sides emit the same shape so that ``compare.py`` can join
    on (module, statistic) and compute abs / rel diffs directly.
    """

    module: str                       # e.g. "01_ols"
    side: str                         # "py" or "R"
    statistic: str                    # e.g. "beta_educ", "se_educ"
    estimate: float | None
    se: float | None = None
    ci_lo: float | None = None
    ci_hi: float | None = None
    n: int | None = None
    extra: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        # Replace NaN with None so JSON stays valid.
        for k, v in list(d.items()):
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                d[k] = None
        return d


def dump_csv(df: pd.DataFrame, name: str) -> Path:
    """Dump a calibrated replica to data/<name>.csv with a fixed
    floating-point precision so R reads identical bytes."""
    path = DATA_DIR / f"{name}.csv"
    df.to_csv(path, index=False, float_format="%.16g")
    return path


def write_results(
    module: str,
    side: str,
    rows: list[ParityRecord],
    *,
    extra: Mapping[str, Any] | None = None,
) -> Path:
    """Write a list of ParityRecord rows to results/<module>_<side>.json."""
    out = RESULTS_DIR / f"{module}_{side}.json"
    payload = {
        "module": module,
        "side": side,
        "rows": [r.to_dict() for r in rows],
        "extra": dict(extra or {}),
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out


def read_results(module: str, side: str) -> dict[str, Any]:
    path = RESULTS_DIR / f"{module}_{side}.json"
    return json.loads(path.read_text(encoding="utf-8"))
