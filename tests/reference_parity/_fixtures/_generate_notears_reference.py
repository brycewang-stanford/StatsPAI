"""Reference weights for ``sp.notears`` from the authors' implementation.

``notears.linear.notears_linear(X, lambda1=0.1, loss_type='l2',
w_threshold=0)`` (github.com/xunzheng/notears, Apache-2.0, commit
4a9ab19) on the standardised data of ``_generate_notears_data.py``.
Writes ``notears_reference.json`` with the unthresholded weight matrix of
every design whose weights all lie at least 0.1 away from the 0.3
threshold, and a checksum of the data.

Run with an interpreter that has ``notears`` installed (not the repo
venv)::

    python tests/reference_parity/_fixtures/_generate_notears_reference.py
"""

from __future__ import annotations

import json
import pathlib
import sys
import warnings

import numpy as np
import scipy
from notears.linear import notears_linear

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))
from _generate_notears_data import CANDIDATES, make  # noqa: E402

cases = {}
for seed in CANDIDATES:
    X, Xs = make(seed)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        W = notears_linear(Xs, lambda1=0.1, loss_type="l2", w_threshold=0.0)
    margin = float(np.min(np.abs(np.abs(W) - 0.3)))
    if margin < 0.1:
        continue
    cases[str(seed)] = {
        "shape": list(X.shape),
        "checksum": float(np.abs(X).sum()),
        "W": W.tolist(),
    }
out = {
    "versions": {
        "python": sys.version.split()[0],
        "notears": "3.0 (github.com/xunzheng/notears 4a9ab19)",
        "scipy": scipy.__version__,
        "numpy": np.__version__,
    },
    "cases": cases,
}
(HERE / "notears_reference.json").write_text(json.dumps(out) + "\n", encoding="utf-8")
print(out["versions"], len(cases), "of", len(CANDIDATES), "designs kept")
