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
  ~1e-14) for every selection path. The ``select_Z``-only path is pinned
  to hdm's own first stage followed by hdm's ``tsls`` with the intercept
  restored: ``rlassoIVselectZ`` 0.3.2 drops the second-stage intercept and
  transposes the sandwich bread (see ``sp.rlasso_iv`` Notes); hdm's raw
  numbers are kept in the fixture as ``selectZ_hdm_raw``.
- ``rlasso_iv`` ``select_Z``-only on EminentDomain: the reference is
  ``AER::ivreg`` + HC0 on hdm's first stage, an exact solve. hdm 0.3.2's
  ``ginv`` solve truncates 34 of 81 singular values of an ill-conditioned
  (cond 5e13) system and reports 0.2274; the IV estimate is 0.0154.

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

import statspai as sp

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
    fit = sp.rlasso(X, y, colnames=cols, **kwargs)

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
    fit = sp.rlasso(X, y, post=True, colnames=cols)
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
    res = sp.rlasso_effect(X, y, d, method=method, post=True)
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
    out = sp.rlasso_effects(X, y, index=[0, 1, 2, 3], method=method)
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
    res = sp.rlasso_iv(y=y, d=d, z=Z, x=X, **kwargs)
    np.testing.assert_allclose(res.coef[0], exp["coef"], atol=1e-6)
    np.testing.assert_allclose(res.se[0], exp["se"], atol=1e-6)


# ──────────── rlasso_iv selectZ with nonzero means (intercept, fixture E) ────


def test_rlasso_iv_selectZ_nonzero_mean_is_2sls(R):
    """``hdm`` 0.3.2's ``rlassoIVselectZ`` drops the intercept from the
    second stage (here -0.652 against a truth of -2). StatsPAI restores it;
    with both instruments selected the estimator is then exactly robust
    2SLS, which ``hdm::tsls(intercept=TRUE, homoscedastic=FALSE)`` gives."""
    df = pd.read_csv(_FIXTURE_DIR / "rlasso_iv_nonzero_mean.csv")
    exp = R["iv_nonzero_mean"]
    assert exp["n_selected"] == 2
    res = sp.rlasso_iv(
        y=df["y"].values,
        d=df["d"].values,
        z=df[["z1", "z2"]].values,
        select_Z=True,
        select_X=False,
    )
    assert res.selection["n_selected_Z"] == exp["n_selected"]
    np.testing.assert_allclose(res.coef[0], exp["tsls"]["coef"], rtol=1e-10)
    np.testing.assert_allclose(res.se[0], exp["tsls"]["se"], rtol=1e-8)
    # the known truth is -2; hdm's uncorrected value is 1.35 away from it
    assert abs(res.coef[0] - exp["beta_true"]) < 3 * res.se[0]


# ───────────────────── rlasso_iv on EminentDomain (real data) ──────────────


def test_eminent_selectZ_matches_hdm(eminent, R):
    """The headline case: BCH (2012) optimal-instrument IV on eminent
    domain, with hdm's first stage (same 5 selected columns). The second
    stage is the exact IV solve, 0.0154 (HC0 SE 0.0141) = AER::ivreg; hdm
    0.3.2 reports 0.2274 (SE 0.2466) from a ginv that truncates 34 of 81
    singular values (the controls hold a constant column, so no intercept
    is added)."""
    y, d, X, Z = eminent
    exp = R["eminent_logGDP"]["selectZ"]
    res = sp.rlasso_iv(y=y, d=d, z=Z, x=X, select_Z=True, select_X=False)
    # cond(x) ~7e6 bounds cross-LAPACK drift near 1e-9; observed 6e-13.
    np.testing.assert_allclose(res.coef[0], exp["coef"], atol=1e-8)
    np.testing.assert_allclose(res.se[0], exp["se"], atol=1e-8)
    assert res.selection["n_selected_Z"] == exp["n_selected"]
    # hdm's truncated-SVD number is kept in the fixture; we must not land
    # on it.
    assert abs(res.coef[0] - R["eminent_logGDP"]["selectZ_hdm_raw"]["coef"]) > 0.1


def test_eminent_selectBoth_matches_hdm(eminent, R):
    y, d, X, Z = eminent
    exp = R["eminent_logGDP"]["selectBoth"]
    res = sp.rlasso_iv(y=y, d=d, z=Z, x=X, select_Z=True, select_X=True)
    np.testing.assert_allclose(res.coef[0], exp["coef"], atol=1e-4)
    np.testing.assert_allclose(res.se[0], exp["se"], atol=1e-4)


def test_eminent_core_selection_matches_hdm(eminent, R):
    """The first stage d ~ [z, x] selects exactly hdm's support."""
    y, d, X, Z = eminent
    exp = R["eminent_logGDP"]["lasso_d_on_zx"]
    fit = sp.rlasso(np.column_stack([Z, X]), d, post=True)
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
        data=df,
        y="y",
        treat="d",
        covariates=xcols,
        model="plr",
        ml_g="rlasso",
        ml_m="rlasso",
        n_folds=ref["n_folds"],
        fold_indices=folds,
    )
    assert res.model_info.get("fold_source") == "user"
    np.testing.assert_allclose(float(res.estimate), ref["theta"], atol=1e-6)
    np.testing.assert_allclose(float(res.se), ref["se"], atol=1e-6)
