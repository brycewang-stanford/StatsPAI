"""Convert the tab-separated output of the Stata fixture do-files to JSON.

The do-files under ``tests/fixtures/{xtivreg,xttobit,xtpcse,xtregar,mmqreg}``
write one row per number with ``file write ... %24.17g`` (17 significant
digits, so every double survives the round trip). Two layouts:

* ``tag  name  b  se`` -> ``{tag: {name: [b, se]}}``. Panel-specific AR(1)
  coefficients written as ``rho1 .. rhoK`` are collected into
  ``{"rho": [[rho1, ..., rhoK], 0.0]}``.
* ``tag  matrix  row  col  value`` -> ``{tag: {matrix: {rows, cols, data}}}``.

Stata's missing value ``.`` becomes ``null``; a ``tag  _meta  key  text``
row (four fields whose last one is not a number) is kept as text under
``{tag: {"_meta": {key: text}}}``, which is how the do-files record the
Stata build and the versions of community commands.

``python tests/fixtures/_stata_tsv.py in.tsv out.json``
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path
from typing import Any, Dict

_RHO = re.compile(r"rho(\d+)$")


def _num(text: str) -> Any:
    """A number, ``None`` for Stata's ``.``, or a list for a macro holding
    several numbers (``xtpcse, correlation(psar1)`` stores ``e(rho)`` so)."""
    parts = text.split()
    if len(parts) > 1:
        return [_num(p) for p in parts]
    text = text.strip()
    return None if text == "." else float(text)


def read_tsv(path: Path) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    rhos: Dict[str, Dict[int, float]] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        f = line.split("\t")
        tag = f[0]
        block = out.setdefault(tag, {})
        if len(f) == 4 and f[1] == "_meta":
            block.setdefault("_meta", {})[f[2]] = f[3].strip()
        elif len(f) == 4:
            name, b, se = f[1], _num(f[2]), _num(f[3])
            m = _RHO.match(name)
            if m:
                rhos.setdefault(tag, {})[int(m.group(1))] = b
            else:
                block[name] = [b, se]
        elif len(f) == 5:
            mat = block.setdefault(f[1], {"rows": [], "cols": [], "cells": {}})
            r, c = f[2], f[3]
            if r not in mat["rows"]:
                mat["rows"].append(r)
            if c not in mat["cols"]:
                mat["cols"].append(c)
            mat["cells"][(r, c)] = _num(f[4])
        else:
            raise ValueError(f"{path}: cannot parse {line!r}")
    for tag, vals in rhos.items():
        out[tag]["rho"] = [[vals[k] for k in sorted(vals)], 0.0]
    for block in out.values():
        for name, mat in list(block.items()):
            if isinstance(mat, dict) and "cells" in mat:
                cells = mat.pop("cells")
                mat["data"] = [
                    [cells[(r, c)] for c in mat["cols"]] for r in mat["rows"]
                ]
    return out


if __name__ == "__main__":
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])
    dst.write_text(json.dumps(read_tsv(src), indent=1) + "\n", encoding="utf-8")
