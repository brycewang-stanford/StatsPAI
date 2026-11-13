"""Reference causal orders for ``sp.lingam`` from the ``lingam`` package.

``lingam.DirectLiNGAM()`` (measure ``pwling``, no prior knowledge) on the
40 data sets of ``_generate_lingam_data.py``. Writes
``lingam_package.json`` with the causal order, a checksum of the data and
the package versions. ``lingam`` is MIT licensed.

Run with an interpreter that has ``lingam`` installed (not the repo venv)::

    python tests/reference_parity/_fixtures/_generate_lingam_package.py
"""

from __future__ import annotations

import json
import pathlib
import sys

import lingam
import numpy as np

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))
from _generate_lingam_data import N_CASES, make  # noqa: E402

cases = {}
for seed in range(N_CASES):
    X, _ = make(seed)
    model = lingam.DirectLiNGAM()
    model.fit(X)
    cases[str(seed)] = {
        "shape": list(X.shape),
        "checksum": float(np.abs(X).sum()),
        "order": [int(v) for v in model.causal_order_],
    }
out = {
    "versions": {
        "python": sys.version.split()[0],
        "lingam": lingam.__version__,
        "numpy": np.__version__,
    },
    "cases": cases,
}
(HERE / "lingam_package.json").write_text(
    json.dumps(out, indent=1) + "\n", encoding="utf-8"
)
print(out["versions"])
