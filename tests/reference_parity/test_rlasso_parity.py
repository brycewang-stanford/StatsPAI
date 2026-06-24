"""Reference parity: ``sp.rlasso`` / ``rlasso_effect`` / ``rlasso_iv`` vs R ``hdm``.

The **hdm** package (Chernozhukov, Hansen & Spindler, 2016, *The R
Journal* 8(2), 185-199) is the reference implementation of the rigorous
(data-driven) Lasso and of optimal-instrument IV (Belloni-Chen-
Chernozhukov-Hansen, 2012; Belloni-Chernozhukov-Hansen, 2014).
StatsPAI's :mod:`statspai.rlasso` is a faithful Python port.

Fixture lifecycle
-----------------
``_generate_rlasso.R`` writes both the inputs (CSV) and the hdm
reference outputs (``_fixtures/rlasso_R.json``).  Re-run it only when the
algorithm contract changes:

    Rscript tests/reference_parity/_generate_rlasso.R

Tolerance discipline
--------------------
Unlike the DoubleML parity tests (whose CV-glmnet folds inject
sqrt(n)-scale noise), ``hdm::rlasso`` is **deterministic** — there is no
cross-validation and no RNG on the default penalty path.  So the bar is
near machine precision, not "within 5%":

- core ``rlasso`` coefficients / loadings / residuals: ``atol=1e-6``
  (observed ~1e-13); selected support: **exact**.
- ``rlasso_effect`` alpha / se: ``atol=1e-6`` (observed ~1e-14).
- ``rlasso_iv`` on a well-conditioned design: ``atol=1e-6`` (observed
  ~1e-14) for every selection path.
- ``rlasso_iv`` on EminentDomain (rank-deficient control block): the
  Moore-Penrose pseudo-inverse SVD differs between LAPACK builds, so we
  allow ``atol=1e-4`` (observed ~1e-9).

References
----------
- Belloni, A., Chen, D., Chernozhukov, V. and Hansen, C. (2012). "Sparse
  Models and Methods for Optimal Instruments With an Application to
  Eminent Domain." *Econometrica*, 80(6), 2369-2429.
  [@belloni2012sparse]
- Belloni, A., Chernozhukov, V. and Hansen, C. (2014). "Inference on
  Treatment Effects After Selection Among High-Dimensional Controls."
  *Review of Economic Studies*, 81(2), 608-650. [@belloni2014inference]
- Chernozhukov, V., Hansen, C. and Spindler, M. (2016). "hdm:
  High-Dimensional Metrics." *The R Journal*, 8(2), 185-199.
  [@chernozhukov2016hdm]
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd
import pytest

from statspai.rlasso import rlasso, rlasso_effect, rlasso_effects, rlasso_iv

_FIXTURE_DIR = pathlib.Path(__file__).parent / "_fixtures"


@pytest.fixture(scope="module")
def R():
    with open(_FIXTURE_DIR / "rlasso_R.json", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def coreA():
    df = pd.read_csv(_FIXTURE_DIR / "rlasso_coreA.csv")
    y = df["y"].values
    X = df.drop(columns=["y"]).values
    cols = list(df.drop(columns=["y"]).columns)
    return X, y, cols


@pytest.fixture(scope="module")
def effect_df():
    df = pd.read_csv(_FIXTURE_DIR / "rlasso_effect.csv")
    return df["y"].values, df["d"].values, df.drop(columns=["y", "d"]).values


@pytest.fixture(scope="module")
def iv_synth():
    df = pd.read_csv(_FIXTURE_DIR / "rlasso_iv_synth.csv")
    xc = [c for c in df.columns if c.startswith("x")]
    zc = [c for c in df.columns if c.startswith("z")]
    return df["y"].values, df["d"].values, df[xc].values, df[zc].values


@pytest.fixture(scope="module")
def eminent():
    df = pd.read_csv(_FIXTURE_DIR / "hdm_eminent_logGDP.csv")
    xc = [c for c in df.columns if c.startswith("x")]
    zc = [c for c in df.columns if c.startswith("z")]
    return df["y"].values, df["d"].values, df[xc].values, df[zc].values


# ───────────────────────────── core rlasso ────────────────────────────────


@pytest.mark.parametrize(
    "key,kwargs",
    [
        ("post_true_intercept_true", dict(post=True, intercept=True)),
        ("post_false_intercept_true", dict(post=False, intercept=True)),
        ("post_true_intercept_false", dict(post=True, intercept=False)),
        (
            "homoscedastic_post_true",
            dict(post=True, intercept=True, penalty={"homoscedastic": True}),
        ),
    ],
)
def test_core_rlasso_matches_hdm(coreA, R, key, kwargs):
    X, y, cols = coreA
    exp = R["coreA"][key]
    fit = rlasso(X, y, colnames=cols, **kwargs)

    # exact support
    assert np.array_equal(fit.index, np.array(exp["index"], dtype=bool))
    assert fit.n_selected == exp["n_selected"]
    # near machine precision on every numeric field
    np.testing.assert_allclose(fit.beta, exp["beta"], atol=1e-6)
    np.testing.assert_allclose(fit.lambda0, exp["lambda0"], rtol=1e-8)
    np.testing.assert_allclose(fit.sigma, exp["sigma"], atol=1e-6)
    np.testing.assert_allclose(fit.loadings, exp["loadings"], atol=1e-6)
    np.testing.assert_allclose(fit.residuals, exp["residuals"], atol=1e-6)
    # hdm reports NA for the no-intercept fit (jsonlite serialises it as
    # the string "NA"); our port reports NaN there.
    if isinstance(exp["intercept"], (int, float)):
        np.testing.assert_allclose(fit.intercept, exp["intercept"], atol=1e-6)
    else:
        assert np.isnan(fit.intercept)


def test_core_predict_matches_hdm(coreA, R):
    X, y, cols = coreA
    fit = rlasso(X, y, post=True, colnames=cols)
    np.testing.assert_allclose(
        fit.predict(X)[:10], R["coreA"]["predict_first10"], atol=1e-6
    )


# ──────────────────────────── rlasso_effect ───────────────────────────────


@pytest.mark.parametrize(
    "method,key",
    [
        ("partialling out", "partialling_out"),
        ("double selection", "double_selection"),
    ],
)
def test_rlasso_effect_matches_hdm(effect_df, R, method, key):
    y, d, X = effect_df
    exp = R["effect"][key]
    res = rlasso_effect(X, y, d, method=method, post=True)
    np.testing.assert_allclose(res.alpha, exp["alpha"], atol=1e-6)
    np.testing.assert_allclose(res.se, exp["se"], atol=1e-6)
    np.testing.assert_allclose(res.tstat, exp["t"], atol=1e-5)
    np.testing.assert_allclose(res.pvalue, exp["pval"], atol=1e-6)


@pytest.mark.parametrize(
    "method,key",
    [
        ("partialling out", "partialling_out"),
        ("double selection", "double_selection"),
    ],
)
def test_rlasso_effects_multi_target_matches_hdm(effect_df, R, method, key):
    """sp.rlasso_effects (many targets) vs hdm::rlassoEffects."""
    y, _, X = effect_df
    exp = R["effects_multi"][key]
    # R index is 1-based c(1,2,3,4); Python is 0-based.
    out = rlasso_effects(X, y, index=[0, 1, 2, 3], method=method)
    vals = list(out.values())
    np.testing.assert_allclose([v.alpha for v in vals], exp["alpha"], atol=1e-6)
    np.testing.assert_allclose([v.se for v in vals], exp["se"], atol=1e-6)
    np.testing.assert_allclose([v.tstat for v in vals], exp["t"], atol=1e-5)


# ─────────────────────── rlasso_iv (synthetic, 4 paths) ────────────────────


@pytest.mark.parametrize(
    "key,kwargs",
    [
        ("selectZ", dict(select_Z=True, select_X=False)),
        ("selectX", dict(select_Z=False, select_X=True)),
        ("selectBoth", dict(select_Z=True, select_X=True)),
        ("plain_tsls", dict(select_Z=False, select_X=False)),
    ],
)
def test_rlasso_iv_synth_matches_hdm(iv_synth, R, key, kwargs):
    y, d, X, Z = iv_synth
    exp = R["iv_synth"][key]
    res = rlasso_iv(y=y, d=d, z=Z, x=X, **kwargs)
    np.testing.assert_allclose(res.coef[0], exp["coef"], atol=1e-6)
    np.testing.assert_allclose(res.se[0], exp["se"], atol=1e-6)


# ───────────────────── rlasso_iv on EminentDomain (real data) ──────────────


def test_eminent_selectZ_matches_hdm(eminent, R):
    """The headline case: BCH (2012) optimal-instrument IV on eminent
    domain.  StatsPAI's pre-port estimator was ~17x off (0.013 vs 0.227);
    the faithful port lands on hdm's 0.2274 / 0.2466."""
    y, d, X, Z = eminent
    exp = R["eminent_logGDP"]["selectZ"]
    res = rlasso_iv(y=y, d=d, z=Z, x=X, select_Z=True, select_X=False)
    np.testing.assert_allclose(res.coef[0], exp["coef"], atol=1e-4)
    np.testing.assert_allclose(res.se[0], exp["se"], atol=1e-4)
    assert res.selection["n_selected_Z"] == exp["n_selected"]


def test_eminent_selectBoth_matches_hdm(eminent, R):
    y, d, X, Z = eminent
    exp = R["eminent_logGDP"]["selectBoth"]
    res = rlasso_iv(y=y, d=d, z=Z, x=X, select_Z=True, select_X=True)
    np.testing.assert_allclose(res.coef[0], exp["coef"], atol=1e-4)
    np.testing.assert_allclose(res.se[0], exp["se"], atol=1e-4)


def test_eminent_core_selection_matches_hdm(eminent, R):
    """The first stage d ~ [z, x] selects exactly hdm's support."""
    y, d, X, Z = eminent
    exp = R["eminent_logGDP"]["lasso_d_on_zx"]
    fit = rlasso(np.column_stack([Z, X]), d, post=True)
    assert fit.n_selected == exp["n_selected"]
    # R is 1-based; selected_idx stored 1-based
    got = (np.where(fit.index)[0] + 1).tolist()
    assert got == exp["selected_idx"]


# ───────────── sp.dml(ml_g='rlasso') vs R DoubleML-PLR-with-hdm::rlasso ─────


@pytest.fixture(scope="module")
def dml_rlasso():
    df = pd.read_csv(_FIXTURE_DIR / "dml_rlasso_data.csv")
    with open(_FIXTURE_DIR / "dml_rlasso_R.json", encoding="utf-8") as f:
        ref = json.load(f)
    return df, ref


def test_dml_rlasso_learner_matches_r_doubleml(dml_rlasso):
    """sp.dml(model='plr', ml_g='rlasso', ml_m='rlasso') reproduces a manual
    Double-ML PLR estimator whose nuisances are hdm::rlasso, cross-fitted over
    the SAME fold partition (shared via the ``fold`` column).  Identical folds
    + a bit-exact rlasso engine ⇒ machine-precision agreement, not a tolerance
    band — a tight cross-ecosystem validation of the rlasso nuisance path."""
    import statspai as sp

    df, ref = dml_rlasso
    xcols = [c for c in df.columns if c.startswith("x")]
    folds = df["fold"].values.astype(int)
    res = sp.dml(
        data=df, y="y", treat="d", covariates=xcols,
        model="plr", ml_g="rlasso", ml_m="rlasso",
        n_folds=ref["n_folds"], fold_indices=folds,
    )
    assert res.model_info.get("fold_source") == "user"
    np.testing.assert_allclose(float(res.estimate), ref["theta"], atol=1e-6)
    np.testing.assert_allclose(float(res.se), ref["se"], atol=1e-6)
