"""``sp.mediate(interaction=True)`` under the default ``inference='bootstrap'``.

Until 2026-10-19 the default raised ``MethodIncompatibility`` for
``interaction=True`` (and for non-linear models / ``treat_values``): those
run on the potential-outcome estimator, which only had the sandwich
variance of ``inference='robust'``. The bootstrap now re-runs that
estimator on case resamples, as R ``mediation::mediate(boot = TRUE)`` does.

R reference (mediation 4.5.1, R 4.5) on the CSV written from
:func:`_data`, ``lm(m ~ d + x)``, ``lm(y ~ d * m + x)``,
``mediate(..., treat = "d", mediator = "m", boot = TRUE, sims = 300)``,
seeds 1 and 2: ``d1`` = 0.428395171064 and ``d0`` = 0.227302456637 in both
runs (the ACMEs are deterministic for linear models). R's ADEs move with
the seed (its mediator integration is by simulation: ``z0`` 0.402788 /
0.403126), so they are not pinned here; ours are the exact integrals.
"""

from __future__ import annotations

import io
import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp

R_ACME_TREATED = 0.428395171064  # mediation d1 == our NIE
R_ACME_CONTROL = 0.227302456637  # mediation d0 == our PNIE


def _data(n=800, seed=1):
    r = np.random.default_rng(seed)
    x = r.normal(size=n)
    d = r.binomial(1, 0.5, n)
    m = 0.5 * d + 0.3 * x + r.normal(size=n)
    y = 0.4 * d + 0.6 * m + 0.5 * d * m + 0.2 * x + r.normal(size=n)
    df = pd.DataFrame(dict(y=y, d=d, m=m, x=x)).round(12)
    buf = io.StringIO()
    df.to_csv(buf, index=False)  # the bytes R read
    return pd.read_csv(io.StringIO(buf.getvalue()), float_precision="round_trip")


KW = dict(y="y", treat="d", mediator="m", covariates=["x"], interaction=True)


@pytest.fixture(scope="module")
def fits():
    df = _data()
    boot = sp.mediate(df, n_boot=400, seed=3, **KW)  # default inference
    robust = sp.mediate(df, inference="robust", **KW)
    return boot, robust


def test_default_inference_runs_and_matches_r_acmes(fits):
    boot, _ = fits
    assert boot.model_info["inference"] == "bootstrap"
    eff = boot.detail.set_index("effect")["estimate"]
    assert eff["NIE"] == pytest.approx(R_ACME_TREATED, rel=1e-9)
    assert eff["PNIE"] == pytest.approx(R_ACME_CONTROL, rel=1e-9)


def test_point_estimates_are_the_robust_ones(fits):
    boot, robust = fits
    a = boot.detail.set_index("effect")["estimate"]
    b = robust.detail.set_index("effect")["estimate"]
    pd.testing.assert_series_equal(a, b, rtol=1e-12)


def test_bootstrap_se_agrees_with_the_sandwich(fits):
    """Both estimate the same sampling SD. With B = 400 the bootstrap SD has
    a Monte Carlo relative error of about 1/sqrt(2B) = 3.5%; allow 4 times
    that."""
    boot, robust = fits
    a = boot.detail.set_index("effect")["se"]
    b = robust.detail.set_index("effect")["se"]
    for name in ["NIE", "NDE", "PNIE", "TNDE", "TE"]:
        assert a[name] == pytest.approx(b[name], rel=0.14)


def test_percentile_interval_and_reproducible_seed(fits):
    boot, _ = fits
    draws = boot.model_info["boot_draws"]["NIE"].to_numpy()
    lo, hi = np.percentile(draws, [2.5, 97.5])
    assert boot.ci == pytest.approx((lo, hi), rel=1e-12)
    again = sp.mediate(_data(), n_boot=400, seed=3, **KW)
    assert again.se == boot.se and again.ci == boot.ci


def test_known_truth_recovered():
    """NIE = (0.6 + 0.5) * 0.5 = 0.55, PNIE = 0.6 * 0.5 = 0.30,
    NDE = 0.4 + 0.5 * E[M(0)] = 0.40, TE = 0.95 (E[M(0)] = 0)."""
    df = _data(n=6000, seed=11)
    res = sp.mediate(df, n_boot=150, seed=0, **KW)
    t = res.detail.set_index("effect")
    for name, truth in [("NIE", 0.55), ("PNIE", 0.30), ("NDE", 0.40), ("TE", 0.95)]:
        assert abs(t.loc[name, "estimate"] - truth) < 3.5 * t.loc[name, "se"]


def test_delta_still_refuses_the_po_estimator():
    with pytest.raises(sp.MethodIncompatibility):
        sp.mediate(_data(), inference="delta", **KW)
