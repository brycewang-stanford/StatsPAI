"""Cross-language parity: weak-IV-robust inference, VIF and meta-analysis.

Fixture: ``_fixtures/_generate_weakiv_R.R`` against R ``ivmodel`` 1.9.1,
``car`` 3.1.5 and ``metafor`` 5.0.1.

Three tolerances for three reasons:

* **Deterministic and pinned at machine level** — the AR statistic and its
  confidence set, ``sp.vif``, and both meta-analysis pooling methods.
* **Monte Carlo by construction** — ``sp.conditional_lr_ci``. ``ivmodel``
  integrates Moreira's conditional distribution; StatsPAI simulates it. The
  test asserts the error *shrinks with* ``n_sim`` rather than pinning a
  number, which is the only honest statement about a simulated critical
  value.
* **Not tested against R at all** — nothing here, deliberately: everything
  in this file has a reference.

Two defects this file was written to close:

* ``sp.vif`` returned ``VIF`` rounded to two decimals and ``1/VIF`` to
  four, in the returned frame. The conventional threshold is 10, and two
  decimals decide it in the fourth significant digit.
* ``sp.anderson_rubin_ci`` (and ``conditional_lr_ci``, ``k_test_ci``)
  reported the extreme *grid point* still inside the acceptance region as
  the interval endpoint, biasing the interval inward by up to one grid
  step — 8.1e-3 on this fixture. The endpoints are now bisected off-grid.
  The grid still decides emptiness, disconnection and unboundedness.
  ``sp.anderson_rubin_test`` computed the same interval analytically all
  along, so the package disagreed with itself about one quantity.
"""

from __future__ import annotations

import json
import pathlib
import warnings

import pandas as pd
import pytest

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import statspai as sp

_FIX = pathlib.Path(__file__).parent / "_fixtures"
IVKW = dict(endog="d", instruments=["z1", "z2"], exog=["w"])


@pytest.fixture(scope="module")
def rjson():
    path = _FIX / "weakiv_R.json"
    if not path.exists():  # pragma: no cover
        pytest.skip("run _generate_weakiv_R.R first")
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def data():
    path = _FIX / "weakiv_data.csv"
    if not path.exists():  # pragma: no cover
        pytest.skip("run _generate_weakiv_R.R first")
    return pd.read_csv(path)


@pytest.fixture(scope="module")
def meta_data():
    path = _FIX / "meta_data.csv"
    if not path.exists():  # pragma: no cover
        pytest.skip("run _generate_weakiv_R.R first")
    return pd.read_csv(path)


# ── Anderson-Rubin ────────────────────────────────────────────────────────


def test_anderson_rubin_statistic_matches_ivmodel(rjson, data):
    out = sp.anderson_rubin_test(data, y="y", **IVKW)
    ref = rjson["AR"]
    assert out["ar_stat"] == pytest.approx(ref["Fstat"], rel=1e-12)
    assert out["ar_pvalue"] == pytest.approx(ref["p"], rel=1e-10)
    assert out["ar_df"] == (ref["df1"], ref["df2"])


def test_anderson_rubin_test_confidence_set_matches_ivmodel(rjson, data):
    out = sp.anderson_rubin_test(data, y="y", **IVKW)
    lo, hi = out["ar_ci"]
    assert lo == pytest.approx(rjson["AR"]["ci_lower"], rel=1e-10)
    assert hi == pytest.approx(rjson["AR"]["ci_upper"], rel=1e-10)


def test_anderson_rubin_ci_matches_ivmodel(rjson, data):
    """The grid-inversion entry point, after off-grid endpoint refinement."""
    cs = sp.anderson_rubin_ci(
        data["y"], data["d"], data[["z1", "z2"]], exog=data[["w"]]
    )
    assert cs.lower == pytest.approx(rjson["AR"]["ci_lower"], rel=1e-10)
    assert cs.upper == pytest.approx(rjson["AR"]["ci_upper"], rel=1e-10)


def test_the_two_ar_confidence_sets_agree_with_each_other(data):
    """Regression guard: the package used to disagree with itself by 8e-3.

    Independent of the reference — if the grid path stops being refined,
    this fails whether or not the fixture is present.
    """
    grid = sp.anderson_rubin_ci(
        data["y"], data["d"], data[["z1", "z2"]], exog=data[["w"]]
    )
    analytic_lo, analytic_hi = sp.anderson_rubin_test(data, y="y", **IVKW)["ar_ci"]
    assert grid.lower == pytest.approx(analytic_lo, rel=1e-9)
    assert grid.upper == pytest.approx(analytic_hi, rel=1e-9)


def test_ar_endpoints_are_not_grid_points(data):
    """The endpoints must lie strictly between two grid nodes.

    A refinement that silently stopped working would put them back exactly
    on the grid, which this catches without needing a tolerance.
    """
    cs = sp.anderson_rubin_ci(
        data["y"], data["d"], data[["z1", "z2"]], exog=data[["w"]]
    )
    grid = cs.beta_grid
    step = float(grid[1] - grid[0])
    for endpoint in (cs.lower, cs.upper):
        nearest = min(abs(float(g) - endpoint) for g in grid)
        assert 0.0 < nearest < step


# ── CLR: Monte Carlo, so the assertion is about convergence ──────────────


def test_clr_confidence_set_converges_to_ivmodel(rjson, data):
    ref_lo = rjson["CLR"]["ci_lower"]
    ref_hi = rjson["CLR"]["ci_upper"]

    def err(n_sim):
        cs = sp.conditional_lr_ci(
            data["y"],
            data["d"],
            data[["z1", "z2"]],
            exog=data[["w"]],
            n_sim=n_sim,
            random_state=7,
        )
        return max(
            abs(cs.lower - ref_lo) / abs(ref_lo),
            abs(cs.upper - ref_hi) / abs(ref_hi),
        )

    coarse = err(5_000)
    fine = err(100_000)
    assert fine < coarse
    assert fine < 5e-3


# ── VIF ───────────────────────────────────────────────────────────────────


@pytest.mark.parametrize("name", ["d", "z1", "z2", "w"])
def test_vif_matches_car(rjson, data, name):
    frame = sp.vif(data, x=["d", "z1", "z2", "w"])
    got = float(frame.loc[frame["variable"] == name, "VIF"].iloc[0])
    assert got == pytest.approx(rjson["vif"][name], rel=1e-12)


def test_vif_is_not_rounded(rjson, data):
    """Two decimals decide the conventional threshold of 10 too coarsely."""
    frame = sp.vif(data, x=["d", "z1", "z2", "w"])
    got = float(frame.loc[frame["variable"] == "d", "VIF"].iloc[0])
    assert got != pytest.approx(round(got, 2), abs=0.0, rel=1e-15)


# ── meta-analysis ─────────────────────────────────────────────────────────


def test_meta_analysis_random_effects_matches_metafor(rjson, meta_data):
    res = sp.meta_analysis(
        meta_data["eff"].to_numpy(), meta_data["se"].to_numpy(), method="DL"
    )
    ref = rjson["meta_DL"]
    assert res.random_estimate == pytest.approx(ref["b"], rel=1e-12)
    assert res.random_se == pytest.approx(ref["se"], rel=1e-12)
    assert res.tau2 == pytest.approx(ref["tau2"], rel=1e-12)


def test_meta_analysis_fixed_effects_matches_metafor(rjson, meta_data):
    res = sp.meta_analysis(
        meta_data["eff"].to_numpy(), meta_data["se"].to_numpy(), method="fixed"
    )
    ref = rjson["meta_FE"]
    assert res.fixed_estimate == pytest.approx(ref["b"], rel=1e-12)
    assert res.fixed_se == pytest.approx(ref["se"], rel=1e-12)


def test_meta_analysis_heterogeneity_matches_metafor(rjson, meta_data):
    res = sp.meta_analysis(
        meta_data["eff"].to_numpy(), meta_data["se"].to_numpy(), method="DL"
    )
    ref = rjson["meta_DL"]
    assert res.q == pytest.approx(ref["QE"], rel=1e-12)
    assert res.q_pvalue == pytest.approx(ref["QEp"], rel=1e-10)
    # metafor reports I2 as a percentage; StatsPAI as a fraction.
    assert res.i2 * 100.0 == pytest.approx(ref["I2"], rel=1e-12)
    assert res.h2 == pytest.approx(ref["H2"], rel=1e-12)
