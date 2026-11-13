"""Reference output for ``sp.pcmci`` from ``tigramite``.

``PCMCI(dataframe, cond_ind_test=ParCorr(significance='analytic'))`` and
``run_pcmci(tau_min=1, tau_max=..., pc_alpha=...)`` on the 24 series of
``_generate_pcmci_data.py``: the parents chosen by the PC1 stage, and the
p-value and partial-correlation matrices of the MCI stage. tigramite is
GPL-3; it is run as a black box.

Run with an interpreter that has ``tigramite`` installed (not the repo
venv)::

    python tests/reference_parity/_fixtures/_generate_pcmci_tigramite.py
"""

from __future__ import annotations

import json
import pathlib
import sys

import numpy as np
import tigramite
from tigramite import data_processing as pp
from tigramite.independence_tests.parcorr import ParCorr
from tigramite.pcmci import PCMCI

HERE = pathlib.Path(__file__).parent
sys.path.insert(0, str(HERE))
from _generate_pcmci_data import N_CASES, make  # noqa: E402

cases = {}
for seed in range(N_CASES):
    X, tau_max, pc_alpha = make(seed)
    model = PCMCI(
        dataframe=pp.DataFrame(X),
        cond_ind_test=ParCorr(significance="analytic"),
        verbosity=0,
    )
    parents = model.run_pc_stable(tau_min=1, tau_max=tau_max, pc_alpha=pc_alpha)
    res = model.run_pcmci(tau_min=1, tau_max=tau_max, pc_alpha=pc_alpha)
    cases[str(seed)] = {
        "shape": list(X.shape),
        "checksum": float(np.abs(X).sum()),
        "tau_max": tau_max,
        "pc_alpha": pc_alpha,
        # parents[j]: (source, lag) pairs, strongest first
        "parents": {
            str(j): [[int(i), int(-lag)] for (i, lag) in parents[j]]
            for j in sorted(parents)
        },
        # tigramite layout [source, target, lag]
        "p_matrix": res["p_matrix"].tolist(),
        "val_matrix": res["val_matrix"].tolist(),
    }
out = {
    "versions": {
        "python": sys.version.split()[0],
        "tigramite": getattr(tigramite, "__version__", "5.2.10.1"),
        "numpy": np.__version__,
    },
    "cases": cases,
}
(HERE / "pcmci_tigramite.json").write_text(json.dumps(out) + "\n", encoding="utf-8")
print(out["versions"], len(cases))
