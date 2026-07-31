#!/usr/bin/env python3
"""Fold ``dynpanel_stata_raw.csv`` into ``dynpanel_stata.json``.

``_generate_dynpanel_stata.do`` emits a long ``spec,key,value`` CSV because
that is the only shape Stata can write generically over ``e(scalars)``.
This script pivots it into the nested JSON the parity tests consume::

    {
      "<spec>": {
        "coef":    {"L.n": ...},
        "se":      {"L.n": ...},
        "e":       {"N": ..., "zrank": ..., "arm1": ...},
        "r":       {"chi2": ...}          # from estat sargan / estat abond
      }
    }

Run from this directory after the ``.do``::

    python3 _fold_dynpanel_stata.py

Values are parsed from Stata's ``%21.16e`` output, i.e. full double
precision; the JSON is written with ``repr``-grade floats so the round trip
Stata -> CSV -> JSON -> Python is exact to the last bit.
"""

from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

HERE = Path(__file__).resolve().parent
RAW = HERE / "dynpanel_stata_raw.csv"
OUT = HERE / "dynpanel_stata.json"

# Stata writes "." for a missing scalar (e.g. Sargan under vce(robust)).
MISSING = {".", "", "nan"}


def main() -> None:
    if not RAW.exists():
        raise SystemExit(
            f"{RAW.name} not found — run `stata -b do _generate_dynpanel_stata.do` first."
        )

    folded: dict = defaultdict(lambda: defaultdict(dict))
    with RAW.open(newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        if header != ["spec", "key", "value"]:
            raise SystemExit(f"unexpected header {header!r} in {RAW.name}")
        for spec, key, value in reader:
            if ":" not in key:
                raise SystemExit(f"malformed key {key!r} for spec {spec!r}")
            kind, name = key.split(":", 1)
            folded[spec][kind][name] = None if value in MISSING else float(value)

    payload = {
        "_meta": {
            "source": "Stata 18 MP",
            "generator": "_generate_dynpanel_stata.do",
            "data": "webuse abdata (Arellano-Bond 1991 UK employment panel)",
            "commands": "xtabond, xtabond2 (SSC), xtdpdsys",
            "n_specs": len(folded),
        },
        **{spec: {k: dict(v) for k, v in kinds.items()} for spec, kinds in folded.items()},
    }
    OUT.write_text(json.dumps(payload, indent=1, sort_keys=True), encoding="utf-8")
    print(f"wrote {OUT.name}: {len(folded)} specs")


if __name__ == "__main__":
    main()
