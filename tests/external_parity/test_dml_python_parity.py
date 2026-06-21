"""External parity: ``sp.dml`` vs ``DoubleML/doubleml-for-py``.

This module pins ``sp.dml`` against the upstream Python reference
implementation (DoubleML by Bach, Chernozhukov, Kurz & Spindler, 2022,
JMLR 23(53)) — complementary to the R-side check in
``tests/reference_parity/test_dml_parity.py``.

The non-instrumented models (PLR, IRM) consume the in-repo seed=42 DGP
fixture (``tests/reference_parity/_fixtures/dml_data.csv``, n=1000,
p=10, true theta=0.5); the instrumented models (PLIV, IIVM) consume the
companion fixture (``dml_iv_data.csv``, n=2000, p=10, true theta=0.5,
with a continuous instrument ``z_c`` and a binary instrument ``z_b``).
Both engines are given identical scikit-learn nuisance learners
(``LassoCV(cv=5)`` for regression, ``LogisticRegressionCV(cv=5)`` for
binary propensity / compliance), so any numerical divergence reflects a
genuine implementation difference rather than learner choice or DGP
noise.

This covers all four DoubleML model classes one-to-one:
``DoubleMLPLR`` / ``DoubleMLIRM`` / ``DoubleMLPLIV`` / ``DoubleMLIIVM``.

Skipped automatically when ``doubleml`` is not installed — this is an
optional pin (DoubleML is not a runtime dependency of StatsPAI).

Tolerance discipline
--------------------
- PLR (continuous d): coefficient and SE agreement to 1e-3 absolute.
  The two implementations share the same Neyman-orthogonal score and
  the same scikit-learn folds under a fixed seed, so we expect
  bit-for-bit agreement at typical ``LassoCV`` precision (verified
  |Δ| ~ 1e-16).
- PLIV (continuous d, continuous instrument): coefficient and SE
  agreement to 1e-3 absolute. Like PLR this is the partialling-out
  score on a shared fold partition; the two engines agree to machine
  precision (verified |Δcoef| = 0, |Δse| ~ 1e-17).
- IRM (binary d, AIPW): coefficient agreement to 0.05 absolute. The
  two implementations make minor differences in propensity-score
  trimming and AIPW score normalization; on this DGP both estimates
  are statistically indistinguishable from zero (the truth).
- IIVM (binary d, binary instrument, LATE): coefficient agreement to
  0.05 absolute. As with IRM the interactive-IV AIPW score leaves
  fold-conditional construction details unspecified; the two engines
  agree to ~1.2e-2 (≈ 0.13 SE) and both land near the true LATE.

References
----------
- Bach, P., Chernozhukov, V., Kurz, M.S., Spindler, M. (2022).
  DoubleML — An Object-Oriented Implementation of Double Machine
  Learning in Python. *Journal of Machine Learning Research*,
  23(53), 1-6. [@bach2022doubleml]
- Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen,
  C., Newey, W. & Robins, J. (2018). Double/debiased machine learning
  for treatment and structural parameters. *The Econometrics
  Journal*, 21(1), C1-C68. [@chernozhukov2018double]
"""
from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
import pytest

import statspai as sp

doubleml = pytest.importorskip("doubleml")
from sklearn.linear_model import LassoCV, LogisticRegressionCV


_FIXTURE_DIR = pathlib.Path(__file__).parent.parent / "reference_parity" / "_fixtures"
_FIXTURE = _FIXTURE_DIR / "dml_data.csv"
_IV_FIXTURE = _FIXTURE_DIR / "dml_iv_data.csv"


@pytest.fixture(scope="module")
def dml_data() -> pd.DataFrame:
    """Same csv the R reference parity test consumes (PLR / IRM)."""
    return pd.read_csv(_FIXTURE)


@pytest.fixture(scope="module")
def iv_data() -> pd.DataFrame:
    """Instrumented DGP for PLIV / IIVM (see ``_generate_dml_iv_data.py``)."""
    return pd.read_csv(_IV_FIXTURE)


@pytest.fixture(scope="module")
def x_cols() -> list[str]:
    return [f"x{i}" for i in range(1, 11)]


def test_plr_matches_doubleml_for_py(dml_data, x_cols):
    """sp.dml(model='plr') ≡ doubleml.DoubleMLPLR under identical learners."""
    np.random.seed(42)
    sp_res = sp.dml(
        data=dml_data, y="y", treat="d", covariates=x_cols,
        model="plr",
        ml_g=LassoCV(cv=5, random_state=42),
        ml_m=LassoCV(cv=5, random_state=42),
        n_folds=5, n_rep=1, random_state=42,
    )
    sp_coef = float(sp_res.estimate)
    sp_se = float(sp_res.se if np.isscalar(sp_res.se) else np.asarray(sp_res.se)[0])

    np.random.seed(42)
    dml_data_obj = doubleml.DoubleMLData(dml_data, y_col="y", d_cols="d", x_cols=x_cols)
    dml_plr = doubleml.DoubleMLPLR(
        dml_data_obj,
        ml_l=LassoCV(cv=5, random_state=42),
        ml_m=LassoCV(cv=5, random_state=42),
        n_folds=5, n_rep=1,
    )
    dml_plr.fit()
    ref_coef = float(dml_plr.coef[0])
    ref_se = float(dml_plr.se[0])

    assert abs(sp_coef - ref_coef) < 1e-3, (
        f"PLR coef mismatch: sp.dml={sp_coef:.6f}, doubleml-py={ref_coef:.6f}"
    )
    assert abs(sp_se - ref_se) < 1e-3, (
        f"PLR SE mismatch: sp.dml={sp_se:.6f}, doubleml-py={ref_se:.6f}"
    )


def test_irm_matches_doubleml_for_py(dml_data, x_cols):
    """sp.dml(model='irm') agrees with doubleml.DoubleMLIRM (binary D, AIPW)."""
    np.random.seed(42)
    sp_res = sp.dml(
        data=dml_data, y="y", treat="d_bin", covariates=x_cols,
        model="irm",
        ml_g=LassoCV(cv=5, random_state=42),
        ml_m=LogisticRegressionCV(cv=5, random_state=42, max_iter=2000),
        n_folds=5, n_rep=1, random_state=42,
    )
    sp_coef = float(sp_res.estimate)

    np.random.seed(42)
    dml_data_obj = doubleml.DoubleMLData(
        dml_data, y_col="y", d_cols="d_bin", x_cols=x_cols
    )
    dml_irm = doubleml.DoubleMLIRM(
        dml_data_obj,
        ml_g=LassoCV(cv=5, random_state=42),
        ml_m=LogisticRegressionCV(cv=5, random_state=42, max_iter=2000),
        n_folds=5, n_rep=1,
    )
    dml_irm.fit()
    ref_coef = float(dml_irm.coef[0])

    # IRM differs by AIPW score normalization / propensity trimming; both
    # implementations land statistically at zero on this DGP. Allow 0.05
    # absolute (= 2/3 of one SE on this fixture).
    assert abs(sp_coef - ref_coef) < 0.05, (
        f"IRM coef diverged beyond 0.05 absolute: "
        f"sp.dml={sp_coef:.6f}, doubleml-py={ref_coef:.6f}"
    )


def test_pliv_matches_doubleml_for_py(iv_data, x_cols):
    """sp.dml(model='pliv') ≡ doubleml.DoubleMLPLIV under identical learners.

    Partially linear IV with a continuous instrument ``z_c``. Like PLR this
    is the partialling-out score on a shared KFold partition, so the two
    engines agree to machine precision on both coef and SE.
    """
    np.random.seed(42)
    sp_res = sp.dml(
        data=iv_data, y="y_pliv", d="d_c", X=x_cols,
        model="pliv", instrument="z_c",
        ml_g=LassoCV(cv=5, random_state=42),
        ml_m=LassoCV(cv=5, random_state=42),
        ml_r=LassoCV(cv=5, random_state=42),
        n_folds=5, n_rep=1, random_state=42,
    )
    sp_coef = float(sp_res.estimate)
    sp_se = float(sp_res.se if np.isscalar(sp_res.se) else np.asarray(sp_res.se)[0])

    np.random.seed(42)
    # DoubleML PLIV naming: ml_l=E[Y|X], ml_m=E[Z|X], ml_r=E[D|X]. We pass the
    # same LassoCV to all three, so the sp.dml (ml_g=Y, ml_m=D, ml_r=Z) naming
    # cross-over is numerically irrelevant here.
    dml_data_obj = doubleml.DoubleMLData(
        iv_data, y_col="y_pliv", d_cols="d_c", x_cols=x_cols, z_cols="z_c"
    )
    dml_pliv = doubleml.DoubleMLPLIV(
        dml_data_obj,
        ml_l=LassoCV(cv=5, random_state=42),
        ml_m=LassoCV(cv=5, random_state=42),
        ml_r=LassoCV(cv=5, random_state=42),
        n_folds=5, n_rep=1,
    )
    dml_pliv.fit()
    ref_coef = float(dml_pliv.coef[0])
    ref_se = float(dml_pliv.se[0])

    assert abs(sp_coef - ref_coef) < 1e-3, (
        f"PLIV coef mismatch: sp.dml={sp_coef:.6f}, doubleml-py={ref_coef:.6f}"
    )
    assert abs(sp_se - ref_se) < 1e-3, (
        f"PLIV SE mismatch: sp.dml={sp_se:.6f}, doubleml-py={ref_se:.6f}"
    )


def test_iivm_matches_doubleml_for_py(iv_data, x_cols):
    """sp.dml(model='iivm') agrees with doubleml.DoubleMLIIVM (LATE).

    Interactive IV with a binary instrument ``z_b`` and binary treatment
    ``d_b`` (imperfect, X-dependent compliance). As with IRM the AIPW-style
    score leaves fold-conditional construction unspecified, so the two
    engines agree within one-eighth of a standard error rather than to
    machine precision.
    """
    np.random.seed(42)
    sp_res = sp.dml(
        data=iv_data, y="y_iivm", d="d_b", X=x_cols,
        model="iivm", instrument="z_b",
        ml_g=LassoCV(cv=5, random_state=42),
        ml_m=LogisticRegressionCV(cv=5, random_state=42, max_iter=3000),
        ml_r=LogisticRegressionCV(cv=5, random_state=42, max_iter=3000),
        n_folds=5, n_rep=1, random_state=42,
    )
    sp_coef = float(sp_res.estimate)

    np.random.seed(42)
    dml_data_obj = doubleml.DoubleMLData(
        iv_data, y_col="y_iivm", d_cols="d_b", x_cols=x_cols, z_cols="z_b"
    )
    dml_iivm = doubleml.DoubleMLIIVM(
        dml_data_obj,
        ml_g=LassoCV(cv=5, random_state=42),
        ml_m=LogisticRegressionCV(cv=5, random_state=42, max_iter=3000),
        ml_r=LogisticRegressionCV(cv=5, random_state=42, max_iter=3000),
        n_folds=5, n_rep=1,
    )
    dml_iivm.fit()
    ref_coef = float(dml_iivm.coef[0])

    # IIVM LATE differs by interactive-IV AIPW score construction; both land
    # near the true LATE (0.5) and agree to ~1.2e-2 (≈ 0.13 SE) on this
    # fixture. Allow 0.05 absolute, matching the IRM discipline.
    assert abs(sp_coef - ref_coef) < 0.05, (
        f"IIVM coef diverged beyond 0.05 absolute: "
        f"sp.dml={sp_coef:.6f}, doubleml-py={ref_coef:.6f}"
    )
