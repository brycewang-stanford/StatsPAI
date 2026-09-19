"""pyblp reference for tests/reference_parity/test_blp_pyblp_parity.py.

pyblp is a Python package, so it is NOT installed into the StatsPAI
environment; install it into a scratch directory and put that on PYTHONPATH
for this script only::

    pip install --target <scratch>/pylibs pyblp
    PYTHONPATH="src:<scratch>/pylibs" python tests/reference_parity/_generate_blp_pyblp.py

Reads ``_fixtures/blp_pyblp_data.csv`` (``_generate_blp_pyblp_data.py``) and
writes ``_fixtures/blp_pyblp.json``.

Identical integration
---------------------
``sp.blp`` integrates with one set of scrambled-Halton standard-normal nodes
(``_halton_sequence(n_draws, K, seed)``), reused in every market with equal
weights. pyblp is handed exactly those nodes as ``agent_data`` (weights
``1 / R``). The nodes are written into the fixture and the test asserts that
StatsPAI regenerates them bit-for-bit, so the two sides integrate the same
finite-sample objective and the comparison is deterministic (not T3).

Conventions pinned
------------------
* estimation: ``X1 = [1, x1, prices]``, ``X2 = [x1]``, excluded instruments
  ``w1, w2, gh_x1, gh_w1`` (pyblp appends the exogenous X1 columns, as
  StatsPAI does). ``method="2s"``: step 1 ``W = (Z'Z/N)^{-1}``; step 2
  ``W = S^{-1}`` from step-1 residuals with ``center_moments=False`` (StatsPAI
  does not demean the moments; pyblp's default does). ``se_type="robust"``.
  Tight optimiser / contraction tolerances so both sides sit at the optimum.
* pyblp's objective is ``N * gbar' W gbar``; StatsPAI's ``gmm_objective`` is
  ``(Z'xi)' W (Z'xi) = N^2 gbar' W gbar`` -- the test compares
  ``gmm_objective / N``.
* elasticities at FIXED parameters (``sigma = diag(1.0, 0.3)`` on
  ``[x1, prices]``, ``beta = (1, 1, -2)``; ``optimization='return'`` with
  beta pinned by equal bounds): the price random coefficient enters every
  consumer's price coefficient, which is what the elasticity check targets.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pyblp

from statspai.structural.blp import _halton_sequence

HERE = Path(__file__).parent
FIX = HERE / "_fixtures"
IV = ["w1", "w2", "gh_x1", "gh_w1"]
R_DRAWS, SEED = 100, 7
pyblp.options.verbose = False


def _problem(d: pd.DataFrame, rc: list, draws: np.ndarray) -> pyblp.Problem:
    pdd = d.rename(
        columns={"market_id": "market_ids", "share": "shares", "price": "prices"}
    )
    for k, c in enumerate(IV):
        pdd[f"demand_instruments{k}"] = d[c]
    mk = np.unique(d["market_id"])
    agent = pd.DataFrame(
        {"market_ids": np.repeat(mk, len(draws)), "weights": 1.0 / len(draws)}
    )
    for k in range(draws.shape[1]):
        agent[f"nodes{k}"] = np.tile(draws[:, k], len(mk))
    return pyblp.Problem(
        (
            pyblp.Formulation("1 + x1 + prices"),
            pyblp.Formulation("0 + " + " + ".join(rc)),
        ),
        pdd,
        agent_data=agent,
    )


def main() -> None:
    d = pd.read_csv(FIX / "blp_pyblp_data.csv")
    out = {
        "versions": {
            "pyblp": pyblp.__version__,
            "numpy": np.__version__,
            "python": sys.version.split()[0],
        },
        "n_draws": R_DRAWS,
        "seed": SEED,
        "instruments": IV,
    }

    # ---- estimation, random coefficient on x1 --------------------------------
    draws1 = _halton_sequence(R_DRAWS, 1, seed=SEED)
    prob = _problem(d, ["x1"], draws1)
    res = prob.solve(
        sigma=0.5 * np.eye(1),
        method="2s",
        center_moments=False,
        W_type="robust",
        se_type="robust",
        optimization=pyblp.Optimization("l-bfgs-b", {"gtol": 1e-12, "ftol": 0}),
        iteration=pyblp.Iteration("squarem", {"atol": 1e-14}),
    )
    E = res.compute_elasticities()
    own = []
    for m in np.unique(d["market_id"]):
        rows = np.where(d["market_id"].to_numpy() == m)[0]
        own.extend(np.diag(E[rows][:, : len(rows)]).tolist())
    rows0 = np.where(d["market_id"].to_numpy() == 0)[0]
    out["est_x1"] = {
        "nodes": draws1[:, 0].tolist(),
        "beta": res.beta.ravel().tolist(),
        "beta_se": res.beta_se.ravel().tolist(),
        "sigma": float(res.sigma[0, 0]),
        "sigma_se": float(res.sigma_se[0, 0]),
        "objective": float(np.squeeze(res.objective)),
        "N": int(len(d)),
        "converged": bool(res.converged),
        "projected_gradient_norm": float(np.max(np.abs(res.projected_gradient))),
        "delta": res.delta.ravel().tolist(),
        "own_elasticities": own,
        "elasticities_market0": E[rows0][:, : len(rows0)].tolist(),
    }

    # ---- elasticities at fixed parameters, random coefficient on price -------
    draws2 = _halton_sequence(R_DRAWS, 2, seed=SEED)
    prob2 = _problem(d, ["x1", "prices"], draws2)
    beta_fix = np.array([[1.0], [1.0], [-2.0]])
    res2 = prob2.solve(
        sigma=np.diag([1.0, 0.3]),
        beta=beta_fix,
        beta_bounds=(beta_fix, beta_fix),
        method="1s",
        optimization=pyblp.Optimization("return"),
        iteration=pyblp.Iteration("squarem", {"atol": 1e-14}),
    )
    E2 = res2.compute_elasticities()
    own2 = []
    for m in np.unique(d["market_id"]):
        rows = np.where(d["market_id"].to_numpy() == m)[0]
        own2.extend(np.diag(E2[rows][:, : len(rows)]).tolist())
    out["elast_fixed"] = {
        "nodes": draws2.tolist(),
        "sigma": [1.0, 0.3],
        "beta": beta_fix.ravel().tolist(),
        "delta": res2.delta.ravel().tolist(),
        "own_elasticities": own2,
        "elasticities_market0": E2[rows0][:, : len(rows0)].tolist(),
    }

    (FIX / "blp_pyblp.json").write_text(
        json.dumps(out, indent=1, default=float) + "\n", encoding="utf-8"
    )
    print("wrote", FIX / "blp_pyblp.json")


if __name__ == "__main__":
    main()
