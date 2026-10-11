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
* diversion ratios and markups: pyblp puts the diversion to the outside good
  on the diagonal of ``compute_diversion_ratios()``; markups are
  ``prices - compute_costs()`` with two firms per market (``firm_ids``
  below: products alternate), multi-product Bertrand pricing.
* a second estimation puts random coefficients on the constant and ``x1``
  (``est_const_x1``), nodes column 0 for the constant.
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


def firm_ids(d: pd.DataFrame) -> np.ndarray:
    """Two firms per market: products alternate between them."""
    return (d["product_id"].to_numpy() % 2).astype(int)


def _stacked(M: np.ndarray, d: pd.DataFrame, market: int) -> np.ndarray:
    """One market's block of a pyblp ``(N, max J)`` stacked matrix."""
    rows = np.where(d["market_id"].to_numpy() == market)[0]
    return M[rows][:, : len(rows)]


def _post(res: "pyblp.ProblemResults", d: pd.DataFrame) -> dict:
    """Diversion ratios (outside good on the diagonal) and Bertrand markups."""
    div = res.compute_diversion_ratios()
    costs = res.compute_costs()
    return {
        "diversion_market0": _stacked(div, d, 0).tolist(),
        "diversion_market3": _stacked(div, d, 3).tolist(),
        "markups": (d["price"].to_numpy() - costs.ravel()).tolist(),
    }


def _problem(d: pd.DataFrame, rc: list, draws: np.ndarray) -> pyblp.Problem:
    pdd = d.rename(
        columns={"market_id": "market_ids", "share": "shares", "price": "prices"}
    )
    for k, c in enumerate(IV):
        pdd[f"demand_instruments{k}"] = d[c]
    pdd["firm_ids"] = firm_ids(d)
    mk = np.unique(d["market_id"])
    agent = pd.DataFrame(
        {"market_ids": np.repeat(mk, len(draws)), "weights": 1.0 / len(draws)}
    )
    for k in range(draws.shape[1]):
        agent[f"nodes{k}"] = np.tile(draws[:, k], len(mk))
    return pyblp.Problem(
        (
            pyblp.Formulation("1 + x1 + prices"),
            pyblp.Formulation(" + ".join(rc) if "1" in rc else "0 + " + " + ".join(rc)),
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
        **_post(res, d),
    }

    # ---- estimation, random coefficients on the constant and x1 -------------
    draws_c = _halton_sequence(R_DRAWS, 2, seed=SEED)
    res_c = _problem(d, ["1", "x1"], draws_c).solve(
        sigma=0.5 * np.eye(2),
        method="2s",
        center_moments=False,
        W_type="robust",
        se_type="robust",
        optimization=pyblp.Optimization("l-bfgs-b", {"gtol": 1e-12, "ftol": 0}),
        iteration=pyblp.Iteration("squarem", {"atol": 1e-14}),
    )
    out["est_const_x1"] = {
        "beta": res_c.beta.ravel().tolist(),
        "beta_se": res_c.beta_se.ravel().tolist(),
        "sigma": np.diag(res_c.sigma).tolist(),
        "sigma_se": np.diag(res_c.sigma_se).tolist(),
        "objective": float(np.squeeze(res_c.objective)),
        "converged": bool(res_c.converged),
        **_post(res_c, d),
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
        **_post(res2, d),
    }

    (FIX / "blp_pyblp.json").write_text(
        json.dumps(out, indent=1, default=float) + "\n", encoding="utf-8"
    )
    print("wrote", FIX / "blp_pyblp.json")


if __name__ == "__main__":
    main()
