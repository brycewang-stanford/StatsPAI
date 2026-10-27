"""cupid_matching reference for tests/reference_parity/test_choo_siow_cupid_parity.py.

cupid_matching (Bernard Salanie's package for Galichon and Salanie's
separable matching models; MIT licence) is NOT installed into the StatsPAI
environment: it pins numpy < 2 and would grade itself against our own
dependency set. Install it into a scratch environment and run:

    uv venv -p 3.11 <scratch>/refvenv
    uv pip install -p <scratch>/refvenv/bin/python "numpy<2" "scipy<1.14" \
        "scikit-learn<1.6" "pandas<3" "bs_python_utils>=0.6,<0.7"
    uv pip install -p <scratch>/refvenv/bin/python --no-deps cupid_matching==1.3
    <scratch>/refvenv/bin/python tests/reference_parity/_generate_choo_siow_cupid.py

Writes ``_fixtures/choo_siow_cupid.json``. StatsPAI is not imported: the
input table is cupid's own simulated sample (``ChooSiowPrimitives.simulate``,
seeded) and every output number is cupid's.

What is compared
----------------
* ``choo_siow_poisson_glm``: the weighted Poisson GLM estimate of ``beta``
  and its sandwich covariance (multinomial ``Var(mu_hat)``). cupid fits with
  scikit-learn's ``PoissonRegressor`` (L-BFGS, ``tol=1e-12``), so its
  ``beta`` is an optimiser output whose first-order conditions hold only to
  about 5e-6. The same function is therefore also run with scikit-learn's
  solver switched to Newton-Cholesky (``poisson_glm_newton``; cupid's code
  otherwise unchanged), which is the strict parity row.
* ``ipfp_homoskedastic_solver`` at ``tol=1e-14``: the equilibrium for the
  true surplus at new margins (the counterfactual solver).
* The nonparametric surplus ``log(mu_xy^2 / (mu_x0 mu_0y))`` evaluated by
  hand from the same table, as the check on the table round trip.
"""

from __future__ import annotations

import json
from importlib.metadata import version
from pathlib import Path

import cupid_matching.poisson_glm as cupid_glm
import numpy as np
from cupid_matching.ipfp_solvers import ipfp_homoskedastic_solver
from cupid_matching.model_classes import ChooSiowPrimitives
from cupid_matching.poisson_glm import choo_siow_poisson_glm

FIX = Path(__file__).parent / "_fixtures"


def quadratic_bases(X: int, Y: int) -> np.ndarray:
    """The quadratic basis of cupid's own docstring example (1, x, y, x^2, xy, y^2)."""
    B = np.zeros((X, Y, 6))
    B[:, :, 0] = 1.0
    for x in range(X):
        B[x, :, 1] = x
        B[x, :, 3] = x * x
        for y in range(Y):
            B[x, y, 4] = x * y
    for y in range(Y):
        B[:, y, 2] = y
        B[:, y, 5] = y * y
    return B


def main() -> None:
    X, Y = 5, 4
    B = quadratic_bases(X, Y)
    beta_true = np.array([1.0, -0.3, -0.2, -0.05, 0.25, -0.04])
    Phi = B @ beta_true
    n = np.array([1.0, 1.2, 0.9, 1.1, 0.8])
    m = np.array([1.1, 1.0, 0.95, 1.05])
    prim = ChooSiowPrimitives(Phi, n, m)
    sample = prim.simulate(200_000, seed=20261028)
    muxy, mux0, mu0y, n_s, m_s = sample.unpack()

    res = choo_siow_poisson_glm(sample, B, verbose=0)

    # Same cupid function, with scikit-learn's solver switched from L-BFGS
    # (which stops with first-order conditions ~5e-6 off) to Newton-Cholesky.
    # cupid's design, weights, normalisation and variance code run unchanged.
    original = cupid_glm.linear_model.PoissonRegressor

    class _NewtonPoisson(original):  # type: ignore[misc, valid-type]
        def __init__(self, **kw):
            kw["solver"] = "newton-cholesky"
            kw["tol"] = 1e-14
            super().__init__(**kw)

    cupid_glm.linear_model.PoissonRegressor = _NewtonPoisson
    try:
        res_newton = choo_siow_poisson_glm(sample, B, verbose=0)
    finally:
        cupid_glm.linear_model.PoissonRegressor = original

    n_new = np.array([300.0, 500.0, 400.0, 200.0, 350.0])
    m_new = np.array([450.0, 250.0, 300.0, 520.0])
    eq, err_x, err_y = ipfp_homoskedastic_solver(Phi, n_new, m_new, tol=1e-14)
    cf_xy, cf_x0, cf_0y, *_ = eq.unpack()

    phi_np = 2 * np.log(muxy) - np.log(mux0)[:, None] - np.log(mu0y)[None, :]

    out = {
        "versions": {
            "cupid_matching": version("cupid_matching"),
            "numpy": version("numpy"),
            "scikit-learn": version("scikit-learn"),
        },
        "inputs": {
            "muxy": muxy.tolist(),
            "mux0": mux0.tolist(),
            "mu0y": mu0y.tolist(),
            "basis": B.tolist(),
            "beta_true": beta_true.tolist(),
            "phi_true": Phi.tolist(),
            "cf_men": n_new.tolist(),
            "cf_women": m_new.tolist(),
        },
        "poisson_glm": {
            "beta": np.asarray(res.estimated_beta).tolist(),
            "varcov_beta": np.asarray(res.varcov_beta).tolist(),
            "stderrs_beta": np.asarray(res.stderrs_beta).tolist(),
        },
        "poisson_glm_newton": {
            "beta": np.asarray(res_newton.estimated_beta).tolist(),
            "varcov_beta": np.asarray(res_newton.varcov_beta).tolist(),
            "stderrs_beta": np.asarray(res_newton.stderrs_beta).tolist(),
        },
        "ipfp": {
            "muxy": cf_xy.tolist(),
            "mux0": cf_x0.tolist(),
            "mu0y": cf_0y.tolist(),
            "max_margin_error": float(
                max(np.max(np.abs(err_x)), np.max(np.abs(err_y)))
            ),
        },
        "nonparametric_phi": phi_np.tolist(),
    }
    FIX.mkdir(exist_ok=True)
    (FIX / "choo_siow_cupid.json").write_text(
        json.dumps(out, indent=1) + "\n", encoding="utf-8"
    )
    print("beta", res.estimated_beta, "se", res.stderrs_beta)
    print("beta (newton)", res_newton.estimated_beta)


if __name__ == "__main__":
    main()
