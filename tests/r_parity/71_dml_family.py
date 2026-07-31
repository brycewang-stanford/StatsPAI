"""StatsPAI DML family parity (Python side) -- Module 71.

Extends the DML coverage beyond module 08's partially-linear model to
the other three ``DoubleML`` model classes:

    sp.dml(model='irm')   vs  DoubleML::DoubleMLIRM    (binary D, ATE)
    sp.dml(model='pliv')  vs  DoubleML::DoubleMLPLIV   (endogenous D, IV Z)
    sp.dml(model='iivm')  vs  DoubleML::DoubleMLIIVM   (binary D and Z, LATE)

Why this module exists
----------------------
``sp.dml``'s parity grade was certified for ``model='plr'`` only, and the
index record said so explicitly ("Grade is variant-specific"). The other
three classes were pinned against *doubleml-for-py* in
``tests/external_parity/test_dml_python_parity.py``, which is an optional
import: on a machine without ``doubleml`` installed those tests skip
silently, and nothing on the R or Stata side covered them at all.

Shared-fold design
------------------
Cross-fitting is the dominant Monte Carlo term in a DML estimate, and it
is entirely an artefact of the sample split rather than of the
estimator. The CSV therefore carries a deterministic ``fold_id`` column,
and both engines consume it through their explicit sample-splitting APIs
(``fold_indices=`` on the StatsPAI side, ``set_sample_splitting()`` on
the ``DoubleML`` side). With the partition fixed by the data, the split
contributes nothing and the remaining gap is the estimator.

Making that work required teaching ``sp.dml`` to accept ``fold_indices``
for IRM / PLIV / IIVM; previously only PLR routed them, and the other
three raised rather than silently ignore the argument.

Convention alignment (recorded, not absorbed by a wide tolerance)
-----------------------------------------------------------------
* ``trimming_threshold``: ``DoubleML``'s default is 1e-12, StatsPAI's is
  1e-2. Both sides are set to 1e-12 here. The DGP keeps propensities
  well inside (0.2, 0.8), so no observation is trimmed on either side
  and the parameter is inert -- the point is to remove it as a
  confound, not to hide behind it.
* ``normalize_ipw``: ``DoubleML`` 1.0.2's IRM/IIVM constructors do not
  expose Hajek self-normalisation, so StatsPAI uses its default
  ``normalize_ipw=False`` to match.
* Nuisance learners are closed-form on both sides -- ``regr.lm`` /
  ``LinearRegression`` for the regressions, ``classif.log_reg``
  (``stats::glm``) / ``LogisticRegression(penalty=None)`` for the
  binary ones. The logistic fits are the same unpenalised MLE reached
  by different optimisers, which sets the achievable floor for the
  models that use them.

Registered tolerance (``compare.py``): rel_est < 1e-6 for PLIV (all
nuisances closed-form least squares) and rel_est < 1e-4 for IRM / IIVM,
whose propensity nuisance is an iteratively-solved logistic MLE.

References
----------
[@chernozhukov2018double], [@bach2022doubleml]
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import statspai as sp
from sklearn.linear_model import LinearRegression, LogisticRegression

from _common import PARITY_SEED, ParityRecord, dump_csv, write_results

MODULE = "71_dml_family"
N = 2000
K = 5
COVARIATES = [f"x{j + 1}" for j in range(K)]
N_FOLDS = 5
TRIMMING = 1e-12


def _reg():
    return LinearRegression()


def _clf():
    # Unpenalised logistic MLE -- the sklearn counterpart of stats::glm,
    # which is what mlr3's classif.log_reg wraps.
    return LogisticRegression(penalty=None, max_iter=5000, tol=1e-10)


def make_data(seed: int = PARITY_SEED) -> pd.DataFrame:
    """One frame serving all three model classes.

    ``d_bin``  -- binary treatment with e(X) bounded in (0.2, 0.8), for IRM.
    ``d_cont`` -- continuous endogenous treatment, instrumented by ``z_c``.
    ``d_iv``   -- binary treatment driven by the binary instrument ``z_b``
                  and by X, for IIVM (LATE).
    """
    rng = np.random.default_rng(seed)
    X = rng.normal(size=(N, K))
    df = pd.DataFrame(X, columns=COVARIATES)

    # --- IRM: binary treatment, ATE = 0.75 ---------------------------
    # tanh keeps the propensity strictly inside (0.2, 0.8), so trimming
    # is inert on both sides and the row grades the score, not overlap.
    e = 0.5 + 0.3 * np.tanh(0.8 * X[:, 0] - 0.5 * X[:, 1])
    d_bin = (rng.uniform(size=N) < e).astype(int)
    df["d_bin"] = d_bin
    df["y_irm"] = (
        0.75 * d_bin + X[:, 0] + 0.5 * X[:, 2] - 0.3 * X[:, 3]
        + rng.normal(scale=1.0, size=N)
    )

    # --- PLIV: continuous endogenous D, continuous instrument --------
    z_c = rng.normal(size=N)
    u = rng.normal(size=N)  # confounder entering both D and Y
    d_cont = (
        0.8 * z_c + 0.5 * X[:, 0] - 0.4 * X[:, 1] + u
        + rng.normal(scale=0.5, size=N)
    )
    df["z_c"] = z_c
    df["d_cont"] = d_cont
    df["y_pliv"] = (
        0.5 * d_cont + X[:, 1] + 0.4 * X[:, 3] + u + rng.normal(scale=0.5, size=N)
    )

    # --- IIVM: binary D, binary Z, LATE ------------------------------
    z_b = rng.binomial(1, 0.5, size=N)
    # Compliance: Z shifts the latent index; always-/never-takers exist.
    latent = -0.2 + 1.6 * z_b + 0.5 * X[:, 0] - 0.4 * X[:, 2]
    p_d = 1.0 / (1.0 + np.exp(-latent))
    d_iv = (rng.uniform(size=N) < p_d).astype(int)
    df["z_b"] = z_b
    df["d_iv"] = d_iv
    df["y_iivm"] = (
        1.0 * d_iv + 0.6 * X[:, 0] + 0.3 * X[:, 4] + rng.normal(scale=1.0, size=N)
    )

    # Deterministic folds shared with R.
    df["fold_id"] = np.arange(N) % N_FOLDS
    return df


def main() -> None:
    df = make_data()
    dump_csv(df, MODULE)
    folds = df["fold_id"].to_numpy()
    rows: list[ParityRecord] = []

    def _record(stat: str, fit) -> None:
        rows.append(
            ParityRecord(
                module=MODULE, side="py", statistic=stat,
                estimate=float(fit.estimate), se=float(fit.se),
                ci_lo=float(fit.ci[0]) if fit.ci is not None else None,
                ci_hi=float(fit.ci[1]) if fit.ci is not None else None,
                n=int(fit.n_obs),
            )
        )
        assert fit.model_info["fold_source"] == "user", fit.model_info["fold_source"]

    irm = sp.dml(
        data=df, y="y_irm", d="d_bin", X=COVARIATES, model="irm",
        model_y=_reg(), model_d=_clf(),
        n_folds=N_FOLDS, fold_indices=folds,
        trimming_threshold=TRIMMING, normalize_ipw=False,
    )
    _record("theta_DML_IRM", irm)

    pliv = sp.dml(
        data=df, y="y_pliv", d="d_cont", X=COVARIATES, model="pliv", instrument="z_c",
        model_y=_reg(), model_d=_reg(), ml_r=_reg(),
        n_folds=N_FOLDS, fold_indices=folds,
    )
    _record("theta_DML_PLIV", pliv)

    iivm = sp.dml(
        data=df, y="y_iivm", d="d_iv", X=COVARIATES, model="iivm", instrument="z_b",
        model_y=_reg(), model_d=_clf(), ml_r=_clf(),
        n_folds=N_FOLDS, fold_indices=folds,
        trimming_threshold=TRIMMING, normalize_ipw=False,
    )
    _record("theta_DML_IIVM", iivm)

    write_results(
        MODULE, "py", rows,
        extra={
            "n_folds": N_FOLDS,
            "seed": PARITY_SEED,
            "fold_column": "fold_id",
            "fold_source": "user",
            "trimming_threshold": TRIMMING,
            "normalize_ipw": False,
            "ml_regression": "LinearRegression",
            "ml_classification": "LogisticRegression(penalty=None)",
            "covariates": COVARIATES,
            "propensity_support": "e(X) = 0.5 + 0.3*tanh(.) in (0.2, 0.8)",
            "models": {
                "theta_DML_IRM": "y_irm ~ d_bin | X (ATE)",
                "theta_DML_PLIV": "y_pliv ~ d_cont | X, instrument z_c",
                "theta_DML_IIVM": "y_iivm ~ d_iv | X, instrument z_b (LATE)",
            },
            "note": (
                "Shared-fold parity across the three DoubleML model classes "
                "module 08 does not cover. Both engines consume the same "
                "fold_id column through their explicit sample-splitting "
                "APIs, so cross-fitting contributes no Monte Carlo term "
                "and the residual gap is the estimator. Trimming is set to "
                "DoubleML's 1e-12 on both sides and is inert on this DGP "
                "(propensities stay inside (0.2, 0.8)); normalize_ipw is "
                "False because DoubleML 1.0.2 does not expose Hajek "
                "normalisation on IRM/IIVM."
            ),
        },
    )


if __name__ == "__main__":
    main()
