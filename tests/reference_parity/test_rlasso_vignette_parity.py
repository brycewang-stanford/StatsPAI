"""Reference parity: StatsPAI vs the canonical ``hdm`` *vignette* applications.

The `hdm` package (Chernozhukov, Hansen & Spindler, 2016, *The R Journal*
8(2), 185-199) ships headline worked examples in its vignette. This
module pins ``sp.rlasso_effect`` / ``sp.rlasso_iv`` against `hdm`'s own
output on the canonical Growth, AJR, and cps2012 applications, so the
reproduction is a hard contract, not prose:

- **Growth** — ``rlassoEffect`` of (log) initial GDP on growth, selecting
  among ~60 country controls (the conditional-convergence coefficient,
  ~ -0.05). Data: Barro & Lee growth panel (``hdm::GrowthData``).
- **AJR** — ``rlassoIV`` (select among high-dim controls, single
  instrument) of expropriation risk on GDP, instrumented by settler
  mortality (~ 0.85). Acemoglu, Johnson & Robinson (2001), AER
  91(5):1369-1401 (``hdm::AJR``).
- **cps2012** — ``rlassoEffects`` for the gender wage gap in CPS 2012.
  The committed fixture is a deterministic 800-row subsample; the full
  29,217-row sample is recorded in the reference JSON because the expanded
  design is too large to bundle.

These datasets are public economic facts. Fixtures + reference
numbers are produced by ``_generate_rlasso_vignette.R`` (re-run only on a
contract change); no R is needed at test time.

Tolerance discipline mirrors ``test_rlasso_parity.py``: ``hdm`` is
deterministic (data-driven penalty, no CV), so the bar is near machine
precision. Observed agreement on these examples is ~1e-10 or tighter; the
assertions use ``atol=1e-6``.
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
def vignette_ref():
    path = _FIXTURE_DIR / "rlasso_vignette_R.json"
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def growth_xyd():
    df = pd.read_csv(_FIXTURE_DIR / "hdm_growth_data.csv")
    y = df["Outcome"].to_numpy()
    d = df["gdpsh465"].to_numpy()
    X = df.drop(columns=["Outcome", "intercept", "gdpsh465"]).to_numpy()
    return X, y, d


@pytest.fixture(scope="module")
def ajr_xydz():
    df = pd.read_csv(_FIXTURE_DIR / "hdm_ajr_data.csv")
    y = df["GDP"].to_numpy()
    d = df["Exprop"].to_numpy()
    z = df["logMort"].to_numpy()
    X = df.drop(columns=["GDP", "Exprop", "logMort"]).to_numpy()
    return X, y, d, z


@pytest.mark.parametrize(
    "method, key",
    [("partialling out", "partialling_out"), ("double selection", "double_selection")],
)
def test_growth_rlasso_effect_matches_hdm(vignette_ref, growth_xyd, method, key):
    X, y, d = growth_xyd
    res = sp.rlasso_effect(X, y, d, method=method)
    exp = vignette_ref["growth"][key]
    np.testing.assert_allclose(res.alpha, exp["coef"], atol=1e-6)
    np.testing.assert_allclose(res.se, exp["se"], atol=1e-6)


def test_growth_convergence_sign_and_scale(vignette_ref, growth_xyd):
    # Sanity on the economics: conditional convergence is negative & small.
    X, y, d = growth_xyd
    res = sp.rlasso_effect(X, y, d, method="partialling out")
    assert res.alpha < 0
    assert abs(res.alpha) < 0.2


def test_ajr_rlasso_iv_matches_hdm(vignette_ref, ajr_xydz):
    X, y, d, z = ajr_xydz
    res = sp.rlasso_iv(y, d, z, x=X, select_Z=False, select_X=True)
    coef = float(np.ravel(res.coef)[0])
    se = float(np.ravel(res.se)[0])
    exp = vignette_ref["ajr"]
    np.testing.assert_allclose(coef, exp["coef"], atol=1e-6)
    np.testing.assert_allclose(se, exp["se"], atol=1e-6)


def test_ajr_institutions_effect_positive(ajr_xydz):
    # The headline AJR finding: institutions (lower expropriation risk)
    # raise GDP — a positive, significant coefficient.
    X, y, d, z = ajr_xydz
    res = sp.rlasso_iv(y, d, z, x=X, select_Z=False, select_X=True)
    coef = float(np.ravel(res.coef)[0])
    se = float(np.ravel(res.se)[0])
    assert coef > 0
    assert abs(coef / se) > 1.96


# ---------------------------------------------------------------------------
#  cps2012 — gender wage gap via rlassoEffects.
#
#  The full expanded design is ~29,217 x 116 (~27 MB), too large to bundle. The
#  committed fixture is a deterministic 800-row subsample on which hdm's
#  multi-target call estimates 16 effects. 15 of them are identified and
#  StatsPAI reproduces every one (estimate and SE) to <= 3e-12.
#
#  The 16th, ``female:hsd08``, is not identified: the interaction is non-zero
#  in a single row of the subsample, so after Lasso partialling-out its
#  residual variance is 4e-30 of its total. hdm then reports -4.7e13 (SE
#  3.8e13) and StatsPAI 1.9e-36 -- both are ratios of rounding noise, which
#  is why the two differ, and neither is an estimate. Earlier text here
#  described the divergence as affecting "rare categories" in general and
#  left the other interactions unpinned; measured, the gap is this one
#  column. StatsPAI now warns on it (hdm does not); the test asserts both
#  the warning and the other fifteen numbers.
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def cps_subsample():
    df = pd.read_csv(_FIXTURE_DIR / "hdm_cps2012_subsample.csv")
    Xdf = df.drop(columns=["lnw"])
    return Xdf.to_numpy(), df["lnw"].to_numpy(), list(Xdf.columns)


def test_cps2012_all_identified_targets_match_hdm(vignette_ref, cps_subsample):
    X, y, cols = cps_subsample
    targets = vignette_ref["cps2012"]["subsample"]["targets"]
    # R's column names use ":" where read.csv / check.names wrote ".".
    idx = [cols.index(t["name"].replace(":", ".")) for t in targets]
    with pytest.warns(RuntimeWarning, match="not identified"):
        out = list(sp.rlasso_effects(X, y, index=idx).values())
    for res, exp in zip(out, targets):
        if exp["name"] == "female:hsd08":
            continue  # unidentified; see the block comment above
        np.testing.assert_allclose(res.alpha, exp["coef"], rtol=1e-9)
        np.testing.assert_allclose(res.se, exp["se"], rtol=1e-9)


def test_cps2012_full_sample_gender_gap_recorded(vignette_ref):
    # The published vignette number reproduces exactly (verified at generation
    # time on the full sample; recorded here since the 27 MB design isn't
    # bundled). Guards against silent drift of the documented claim.
    full = vignette_ref["cps2012"]["full_sample"]
    assert full["n"] == 29217
    np.testing.assert_allclose(full["female_coef"], -0.15492328, atol=1e-6)
