"""Shared helpers for the StatsPAI <-> R/Stata original-data parity
harness.

Whereas tests/r_parity/ exercises calibrated replicas, this harness
runs StatsPAI on the *original public-domain data* shipped by R
packages (wooldridge::card, did::mpdta, rdrobust::rdrobust_RDsenate,
Synth::basque, ...) and verifies that the published numbers in the
canonical paper are recovered to the published precision.

Each module follows the same protocol:

  1. R-side dumps the public-package dataset to data/<name>.csv.
  2. Python side (sp.*) reads that CSV and runs the estimator.
  3. R-side runs the canonical R reference on the same data.
  4. compare_orig.py joins the two sides + the published-paper
     anchor and emits a Markdown rollup.

Tolerance for original-data parity is *looser* than the calibrated-
replica harness because:
  * sample-size and variable-construction conventions vary (a paper
    may report rounded or reported numbers we cannot bit-match);
  * the published-paper anchor itself is reported to typically 3
    decimal places.

Pre-registered tolerance: rel < 1e-2 against the published number,
rel < 1e-6 against the canonical R reference on the same bytes.
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


@dataclass
class OrigRecord:
    module: str
    side: str
    statistic: str
    estimate: float | None
    se: float | None = None
    n: int | None = None
    published: float | None = None
    citation: str = ""
    extra: Mapping[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        for k, v in list(d.items()):
            if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
                d[k] = None
        return d


def write_results(module: str, side: str, rows: list[OrigRecord], *,
                  extra: Mapping[str, Any] | None = None) -> Path:
    out = RESULTS_DIR / f"{module}_{side}.json"
    payload = {
        "module": module,
        "side": side,
        "rows": [r.to_dict() for r in rows],
        "extra": dict(extra or {}),
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out


def read_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(DATA_DIR / f"{name}.csv")
