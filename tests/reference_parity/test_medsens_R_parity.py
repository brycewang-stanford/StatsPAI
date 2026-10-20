"""``sp.mediate_sensitivity`` against R ``mediation::medsens`` 4.5.1 (black box).

Extends ``test_misc_sens_R_parity.py`` (one configuration) with the cases in
``_fixtures/medsens_R.json`` (generator ``_fixtures/_generate_medsens_R.R``,
data ``medsens_sim.csv`` from ``_generate_cbps_medsens_data.py``): a finer
rho grid, the FGLS fixed point, no covariates, a continuous treatment, and
the R^2 curves of Imai, Keele & Yamamoto (2010, Section 5.2). mediation is
GPL; only its documented arguments and outputs were used
(``docs/dev/cleanroom/mediate_sensitivity.md``).

Tolerances: both sides run the same deterministic iteration, so iterates
agree to rounding (1e-10 relative; interval limits near zero get an absolute
floor of 1e-12). Grid quantities (rho values, err.cr.d) are compared at 1e-12
absolute because seq() and linspace round differently in the last bit.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import statspai as sp

_FIX = Path(__file__).parent / "_fixtures"
R = json.loads((_FIX / "medsens_R.json").read_text(encoding="utf-8"))
DF = pd.read_csv(_FIX / "medsens_sim.csv")
MEDSENS_EPS = float(np.sqrt(np.finfo(float).eps))

CASES = {
    # name: (y, treat, mediator, covariates, eps)
    "bin_cov": ("y", "t", "m", ["x1", "x2"], MEDSENS_EPS),
    "bin_cov_fp": ("y", "t", "m", ["x1", "x2"], None),
    "bin_nocov": ("y", "t", "m", None, MEDSENS_EPS),
    "cont": ("yc", "tc", "mc", ["x1", "x2"], MEDSENS_EPS),
}


def _fit(name):
    y, t, m, cov, eps = CASES[name]
    ref = R[name]
    rho = np.asarray(ref["rho"])
    return sp.mediate_sensitivity(
        DF,
        y=y,
        treat=t,
        mediator=m,
        covariates=cov,
        rho_range=(rho[0], rho[-1]),
        n_grid=rho.size,
        eps=eps,
    )


@pytest.mark.parametrize("name", list(CASES))
def test_acme_curve_and_interval(name):
    ref = R[name]
    r = _fit(name)
    np.testing.assert_allclose(r.rho_grid, ref["rho"], rtol=0, atol=1e-12)
    np.testing.assert_allclose(r.acme_at_rho, ref["d0"], rtol=1e-10)
    # no treatment-mediator interaction: delta(0) = delta(1)
    np.testing.assert_allclose(ref["d1"], ref["d0"], rtol=1e-12)
    np.testing.assert_allclose(r.ci_lower, ref["lower"], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(r.ci_upper, ref["upper"], rtol=1e-10, atol=1e-12)
    assert r.rho_grid_zero == pytest.approx(ref["err_cr"], abs=1e-12)


@pytest.mark.parametrize("name", list(CASES))
def test_r2_parametrisation(name):
    """IKY (2010, Section 5.2): rho^2 = R2*_M R2*_Y and
    R~2_M R~2_Y = rho^2 (1 - R2_M)(1 - R2_Y), over the whole grid."""
    ref = R[name]
    r = _fit(name)
    assert r.r2_mediator == pytest.approx(ref["r2_m"], rel=1e-12)
    assert r.r2_outcome == pytest.approx(ref["r2_y"], rel=1e-12)
    rho = np.asarray(ref["rho"])
    np.testing.assert_allclose(ref["R2star_prod"], rho**2, rtol=1e-12, atol=1e-15)
    shrink = (1 - r.r2_mediator) * (1 - r.r2_outcome)
    np.testing.assert_allclose(
        ref["R2tilde_prod"], rho**2 * shrink, rtol=1e-12, atol=1e-15
    )
    # medsens reports the thresholds at its grid crossing
    assert r.rho_grid_zero**2 == pytest.approx(ref["R2star_thresh"], rel=1e-12)
    assert r.rho_grid_zero**2 * shrink == pytest.approx(
        ref["R2tilde_thresh"], rel=1e-12
    )
    # StatsPAI's thresholds are the same products at the exact crossing
    assert r.r2star_threshold == pytest.approx(r.rho_at_zero**2, rel=1e-14)
    assert r.r2tilde_threshold == pytest.approx(r.rho_at_zero**2 * shrink, rel=1e-14)


@pytest.mark.parametrize("name", list(CASES))
def test_exact_crossing(name):
    """Theorem 4: ACME(rho) = 0 iff rho = Corr(e_M, e_{Y ~ T + X})."""
    y, t, m, cov, _ = CASES[name]
    r = _fit(name)
    assert r.rho_at_zero == pytest.approx(R[name]["rho_tilde"], rel=1e-12)
    at = sp.mediate_sensitivity(
        DF,
        y=y,
        treat=t,
        mediator=m,
        covariates=cov,
        rho_range=(r.rho_at_zero, r.rho_at_zero),
        n_grid=1,
    )
    assert abs(at.acme_at_rho[0]) < 1e-12 * max(1.0, abs(r.acme_at_zero))


def test_default_is_the_fixed_point_and_medsens_default_stops_short():
    fp, short = R["bin_cov_fp"], R["bin_cov"]
    r = _fit("bin_cov_fp")
    np.testing.assert_allclose(r.acme_at_rho, fp["d0"], rtol=1e-10)
    gap = np.max(np.abs(np.array(short["d0"]) / np.array(fp["d0"]) - 1))
    assert gap > 1e-8  # medsens' default stopping rule is visibly short


def test_max_iter_warns_and_returns_the_last_iterate():
    with pytest.warns(sp.ConvergenceWarning, match="max_iter=1"):
        r = sp.mediate_sensitivity(
            DF,
            y="y",
            treat="t",
            mediator="m",
            covariates=["x1", "x2"],
            rho_range=(0.5, 0.5),
            n_grid=1,
            max_iter=1,
        )
    full = sp.mediate_sensitivity(
        DF,
        y="y",
        treat="t",
        mediator="m",
        covariates=["x1", "x2"],
        rho_range=(0.5, 0.5),
        n_grid=1,
    )
    assert np.isfinite(r.acme_at_rho[0])
    assert abs(r.acme_at_rho[0] - full.acme_at_rho[0]) > 1e-6


@pytest.mark.parametrize(
    "kw,match",
    [
        ({"rho_range": (-1.0, 0.5)}, "rho_range"),
        ({"rho_range": (0.5, -0.5)}, "rho_range"),
        ({"n_grid": 0}, "n_grid"),
        ({"conf_level": 1.0}, "conf_level"),
        ({"eps": 0.0}, "eps"),
        ({"covariates": ["x1", "nope"]}, "not found"),
    ],
)
def test_invalid_arguments_raise_typed_errors(kw, match):
    args = dict(data=DF, y="y", treat="t", mediator="m", covariates=["x1"])
    args.update(kw)
    with pytest.raises(sp.MethodIncompatibility, match=match):
        sp.mediate_sensitivity(**args)


def test_collinear_covariates_are_refused():
    with pytest.raises(sp.MethodIncompatibility, match="not full rank"):
        sp.mediate_sensitivity(
            DF.assign(x3=2 * DF.x1),
            y="y",
            treat="t",
            mediator="m",
            covariates=["x1", "x3"],
        )
