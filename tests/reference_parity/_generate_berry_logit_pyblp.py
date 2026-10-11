"""pyblp reference for tests/reference_parity/test_berry_logit_pyblp_parity.py.

pyblp is a Python package, so it is NOT installed into the StatsPAI
environment (it would be the reference grading itself against our own
dependency set). Install it into a scratch directory and run:

    pip install --target <scratch>/pylibs pyblp
    PYTHONPATH="<scratch>/pylibs" python tests/reference_parity/_generate_berry_logit_pyblp.py

Reads ``_fixtures/berry_logit_data.csv`` (``_generate_berry_logit_data.py``)
and writes ``_fixtures/berry_logit_pyblp.json``. StatsPAI is not imported:
every number in the JSON is pyblp's.

Conventions
-----------
* Instruments: ``pyblp.build_blp_instruments`` on ``1 + x1 + x2`` (the
  firm's other products, then rival products, for the count and each
  characteristic), plus the cost shifter ``w``. The nested logit adds the
  count of products in the nest, the instrument of the pyblp tutorial; it
  is the ``nest_count`` of ``sp.blp_instruments`` plus one.
* ``center_moments=False``: StatsPAI's step-2 weighting matrix does not
  demean the moments. ``W_type="robust"``, ``se_type="robust"``.
* ``method="1s"`` is the weighting matrix ``(Z'Z/N)^{-1}``, i.e. 2SLS.
* pyblp's objective is ``N gbar'W gbar``; StatsPAI's is ``N`` times that.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyblp

FIX = Path(__file__).parent / "_fixtures"
pyblp.options.verbose = False


def _own_and_market0(res, d):
    E = res.compute_elasticities()
    own = res.extract_diagonals(E).ravel().tolist()
    rows = np.where(d["market_ids"].to_numpy() == 0)[0]
    return own, E[rows][:, : len(rows)].tolist()


def main() -> None:
    raw = pd.read_csv(FIX / "berry_logit_data.csv")
    d = raw.rename(
        columns={
            "market_id": "market_ids",
            "firm": "firm_ids",
            "share": "shares",
            "price": "prices",
        }
    )
    blp_iv = pyblp.build_blp_instruments(pyblp.Formulation("1 + x1 + x2"), d)
    out = {
        "versions": {
            "pyblp": pyblp.__version__,
            "numpy": np.__version__,
            "python": sys.version.split()[0],
        },
        "N": int(len(d)),
        "blp_instruments": blp_iv.tolist(),
    }
    form = pyblp.Formulation("1 + x1 + x2 + prices")
    tight = pyblp.Optimization("l-bfgs-b", {"gtol": 1e-13, "ftol": 0})

    logit = d.copy()
    for k in range(blp_iv.shape[1]):
        logit[f"demand_instruments{k}"] = blp_iv[:, k]
    logit[f"demand_instruments{blp_iv.shape[1]}"] = d["w"]
    nested = logit.copy()
    nested["nesting_ids"] = d["nest"]
    nested[f"demand_instruments{blp_iv.shape[1] + 1}"] = nested.groupby(
        ["market_ids", "nesting_ids"]
    )["shares"].transform("size")

    for label, frame, kw in (
        ("logit", logit, {}),
        ("nested", nested, {"rho": 0.5, "optimization": tight}),
    ):
        problem = pyblp.Problem(form, frame)
        for method in ("1s", "2s"):
            res = problem.solve(
                method=method,
                center_moments=False,
                W_type="robust",
                se_type="robust",
                **kw,
            )
            own, e0 = _own_and_market0(res, frame)
            entry = {
                "beta": res.beta.ravel().tolist(),
                "beta_se": res.beta_se.ravel().tolist(),
                "objective": float(np.squeeze(res.objective)),
                "own_elasticities": own,
                "elasticities_market0": e0,
            }
            if label == "nested":
                entry["rho"] = float(np.squeeze(res.rho))
                entry["rho_se"] = float(np.squeeze(res.rho_se))
                entry["delta"] = res.delta.ravel().tolist()
                entry["projected_gradient_norm"] = float(
                    np.max(np.abs(res.projected_gradient))
                )
            out[f"{label}_{method}"] = entry

    (FIX / "berry_logit_pyblp.json").write_text(
        json.dumps(out, indent=1), encoding="utf-8"
    )
    print("wrote", FIX / "berry_logit_pyblp.json")


if __name__ == "__main__":
    main()
