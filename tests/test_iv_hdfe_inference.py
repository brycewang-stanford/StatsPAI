"""Behavioural tests for HDFE-IV inference wiring.

Numerical agreement with Stata lives in
``tests/reference_parity/test_iv_hdfe_stata_parity.py``. This file covers
the API contracts around it: the cluster-inside-absorb case that used to
crash, multiway clustering, spatial SEs on an absorbed IV fit, and the
weak-IV-robust helpers accepting the same ``absorb``/``cluster`` spec as
the estimator.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import MethodIncompatibility


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    rng = np.random.default_rng(7)
    n_unit, n_per = 80, 15
    unit = np.repeat(np.arange(n_unit), n_per)
    t = np.tile(np.arange(n_per), n_unit)
    n = unit.size
    alpha = rng.normal(size=n_unit)[unit]
    z = rng.normal(size=n)
    u = rng.normal(size=n)
    d = 0.9 * z + 0.6 * u + alpha + rng.normal(size=n)
    y = -0.5 * d + 0.7 * u + alpha + rng.normal(size=n)
    return pd.DataFrame(
        {
            "y": y,
            "d": d,
            "z": z,
            "x": rng.normal(size=n),
            "unit": unit,
            "t": t,
            "region": unit // 8,
            "lat": np.repeat(rng.uniform(20, 50, n_unit), n_per),
            "lon": np.repeat(rng.uniform(80, 130, n_unit), n_per),
        }
    )


FORMULA = "y ~ (d ~ z) + x"


def test_clustering_on_an_absorbed_dimension_works(panel):
    """The canonical panel spec: absorb unit FE, cluster by unit.

    This raised ``ValueError: If using all scalar values`` before, because
    the shared column was selected twice into the working frame.
    """
    res = sp.iv(FORMULA, data=panel, absorb=["unit", "t"], cluster="unit")
    assert np.isfinite(res.params["d"])
    assert res.std_errors["d"] > 0
    assert res.model_info["n_clusters"] == 80


def test_cluster_spellings_agree(panel):
    a = sp.iv(FORMULA, data=panel, absorb="unit", cluster=["unit", "t"])
    b = sp.iv(FORMULA, data=panel, absorb="unit", cluster="unit + t")
    np.testing.assert_allclose(a.std_errors["d"], b.std_errors["d"], rtol=1e-14)


def test_multiway_cluster_is_order_invariant_and_psd(panel):
    a = sp.iv(FORMULA, data=panel, absorb="unit", cluster=["unit", "t"])
    b = sp.iv(FORMULA, data=panel, absorb="unit", cluster=["t", "unit"])
    np.testing.assert_allclose(a.std_errors["d"], b.std_errors["d"], rtol=1e-12)
    assert np.all(np.isfinite(a.std_errors.to_numpy()))
    assert np.all(a.std_errors.to_numpy() > 0)


def test_multiway_records_cluster_counts_per_dimension(panel):
    res = sp.iv(FORMULA, data=panel, cluster=["unit", "t"])
    assert res.model_info["n_clusters"] == [80, 15]


def test_absorb_rejects_array_cluster_with_a_usable_message(panel):
    with pytest.raises(MethodIncompatibility, match="cluster to name column"):
        sp.iv(
            FORMULA,
            data=panel,
            absorb="unit",
            cluster=panel["unit"].to_numpy(),
        )


def test_gmm_vcov_option_changes_only_the_variance(panel):
    sandwich = sp.iv(
        "y ~ (d ~ z + x)", data=panel, absorb="unit", cluster="unit", method="gmm"
    )
    efficient = sp.iv(
        "y ~ (d ~ z + x)",
        data=panel,
        absorb="unit",
        cluster="unit",
        method="gmm",
        gmm_vcov="efficient",
    )
    np.testing.assert_allclose(sandwich.params["d"], efficient.params["d"], rtol=1e-12)
    assert sandwich.std_errors["d"] != efficient.std_errors["d"]


def test_every_method_absorbs(panel):
    base = sp.iv("y ~ (d ~ z + x)", data=panel, absorb=["unit", "t"])
    for method in ("liml", "fuller", "gmm", "jive"):
        res = sp.iv("y ~ (d ~ z + x)", data=panel, absorb=["unit", "t"], method=method)
        assert np.isfinite(res.params["d"])
        np.testing.assert_allclose(res.params["d"], base.params["d"], rtol=0.1)


def test_conley_accepts_an_absorbed_iv_result(panel):
    """Spatial HAC used to fail on IV results with ``KeyError: 'X'``."""
    res = sp.iv(FORMULA, data=panel, absorb=["unit", "t"])
    spatial = sp.conley(res, panel, lat="lat", lon="lon", dist_cutoff=400)
    np.testing.assert_allclose(spatial.params["d"], res.params["d"], rtol=1e-12)
    assert spatial.std_errors["d"] > 0
    assert spatial.model_info["se_type"] == "conley_spatial"


def test_conley_spatiotemporal_on_iv(panel):
    res = sp.iv(FORMULA, data=panel, absorb=["unit", "t"])
    out = sp.conley(
        res,
        panel,
        lat="lat",
        lon="lon",
        dist_cutoff=400,
        time="t",
        lag_cutoff=2,
        unit="unit",
    )
    assert np.isfinite(out.std_errors["d"])


def test_ar_confidence_set_is_scaled_to_the_coefficient(panel):
    """The AR grid must resolve a coefficient far smaller than 1.

    A fixed +/-5 window with 401 nodes quantises endpoints at 0.025, which
    swamps a policy-index coefficient; the set must instead be scaled by
    the estimate's own standard error.
    """
    small = panel.assign(y=panel["y"] / 1000.0)
    ar = sp.anderson_rubin_test(
        data=small,
        y="y",
        endog="d",
        instruments=["z"],
        exog=["x"],
        absorb=["unit", "t"],
        cluster="unit",
    )
    lo, hi = ar["ar_ci"]
    assert np.isfinite(lo) and np.isfinite(hi)
    width = hi - lo
    assert 0 < width < 0.05, width  # not a 10-wide grid artefact
    iv = sp.iv(FORMULA, data=small, absorb=["unit", "t"], cluster="unit")
    assert lo < iv.params["d"] < hi


def test_ar_cluster_widens_relative_to_iid(panel):
    """Serial correlation has to show up in the AR set, not just in Wald."""
    common = dict(
        data=panel,
        y="y",
        endog="d",
        instruments=["z"],
        exog=["x"],
        absorb=["unit", "t"],
    )
    plain = sp.anderson_rubin_test(**common)
    clustered = sp.anderson_rubin_test(**common, cluster="unit")
    assert clustered["ar_stat"] != plain["ar_stat"]
    assert np.isfinite(clustered["ar_ci"][0])


def test_iv_diag_absorb_matches_the_estimator(panel):
    diag = sp.iv_diag(
        panel,
        y="y",
        endog="d",
        instruments=["z"],
        exog=["x"],
        absorb=["unit", "t"],
        cluster="unit",
        n_boot=25,
        random_state=0,
    )
    res = sp.iv(FORMULA, data=panel, absorb=["unit", "t"], cluster="unit")
    np.testing.assert_allclose(diag.beta_2sls, res.params["d"], rtol=1e-10)
    np.testing.assert_allclose(diag.se_2sls, res.std_errors["d"], rtol=1e-8)
    np.testing.assert_allclose(diag.kp_rk_f, res.diagnostics["KP rk Wald F"], rtol=1e-8)


def test_effective_f_absorb_matches_the_attached_diagnostic(panel):
    res = sp.iv(FORMULA, data=panel, absorb=["unit", "t"], cluster="unit")
    out = sp.effective_f_test(
        panel,
        endog="d",
        instruments=["z"],
        exog=["x"],
        absorb=["unit", "t"],
        cluster="unit",
    )
    np.testing.assert_allclose(
        out["F_eff"], res.diagnostics["Olea-Pflueger effective F"], rtol=1e-10
    )


def test_interacted_fixed_effects(panel):
    """``prov^t`` is the fixest spelling for an interacted fixed effect."""
    interacted = sp.iv(FORMULA, data=panel, absorb=["unit", "region^t"], cluster="unit")
    manual = panel.assign(
        region_t=panel["region"].astype(str) + "_" + panel["t"].astype(str)
    )
    explicit = sp.iv(FORMULA, data=manual, absorb=["unit", "region_t"], cluster="unit")
    np.testing.assert_allclose(interacted.params["d"], explicit.params["d"], rtol=1e-12)
    np.testing.assert_allclose(
        interacted.std_errors["d"], explicit.std_errors["d"], rtol=1e-12
    )
    # The label the caller wrote is what gets reported back.
    assert interacted.model_info["absorb"] == ["unit", "region^t"]


def test_interacted_fe_names_a_missing_column(panel):
    with pytest.raises(MethodIncompatibility, match="missing column"):
        sp.iv(FORMULA, data=panel, absorb=["unit", "region^nope"])
