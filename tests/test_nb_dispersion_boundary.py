"""NB2 dispersion at its boundary alpha = 0 in sp.zinb and sp.hurdle.

When the counts are Poisson the dispersion estimate runs to zero and the
full Hessian is singular in ln(alpha). Inverting it returned rounding
noise whose sign depended on the platform (NaN standard errors on Linux
and macOS, a finite one on Windows CI on 2026-10-17). The boundary is now
decided from the estimate (overdispersion alpha * mean(mu) < 1e-4): ln_alpha
gets NaN with a warning, and the other standard errors hold alpha fixed,
which makes them the Poisson-limit model's standard errors.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import ConvergenceWarning


@pytest.fixture(scope="module")
def poisson_counts():
    # Seed 14 lands on the boundary (ln alpha about -12); on Poisson data
    # roughly half the seeds do, the rest give an interior ln alpha near -3.5.
    rng = np.random.default_rng(14)
    n = 1500
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    y = rng.poisson(np.exp(0.3 + 0.4 * x1))
    y[rng.uniform(size=n) < 1 / (1 + np.exp(2.5 - 0.5 * x2))] = 0
    return pd.DataFrame({"y": y, "x1": x1, "x2": x2})


def _quiet(f, *a, **k):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return f(*a, **k)


@pytest.mark.parametrize("robust", ["nonrobust", "robust"])
def test_zinb_at_the_boundary_reports_the_zip_standard_errors(poisson_counts, robust):
    with pytest.warns(ConvergenceWarning, match="not finite"):
        nb = sp.zinb("y ~ x1", data=poisson_counts, inflate=["x2"], robust=robust)
    assert nb.model_info["dispersion_at_boundary"] is True
    assert nb.model_info["nonfinite_se_terms"] == ["ln_alpha"]
    zip_ = _quiet(
        sp.zip_model, "y ~ x1", data=poisson_counts, inflate=["x2"], robust=robust
    )
    # Same model in the limit; the two optimisers stop at slightly different
    # points along the flat ln(alpha) direction, hence 1e-3 and not 1e-8.
    np.testing.assert_allclose(nb.params.iloc[:4], zip_.params, rtol=1e-3)
    np.testing.assert_allclose(nb.std_errors.iloc[:4], zip_.std_errors, rtol=1e-3)


def test_hurdle_negbin_at_the_boundary(poisson_counts):
    nb = _quiet(sp.hurdle, "y ~ x1", data=poisson_counts, count_model="negbin")
    pois = _quiet(sp.hurdle, "y ~ x1", data=poisson_counts, count_model="poisson")
    assert nb.model_info["dispersion_at_boundary"] is True
    assert np.isnan(nb.std_errors["ln_alpha"])
    np.testing.assert_allclose(nb.std_errors.iloc[:-1], pois.std_errors, rtol=1e-3)


def test_real_overdispersion_is_not_a_boundary():
    rng = np.random.default_rng(12)
    n = 1500
    x1, x2 = rng.normal(size=n), rng.normal(size=n)
    mu = np.exp(0.5 + 0.4 * x1)
    y = rng.negative_binomial(1 / 0.8, 1 / (1 + 0.8 * mu))
    y[rng.uniform(size=n) < 0.2] = 0
    d = pd.DataFrame({"y": y, "x1": x1, "x2": x2})
    fit = _quiet(sp.zinb, "y ~ x1", data=d, inflate=["x2"])
    assert fit.model_info["dispersion_at_boundary"] is False
    assert np.isfinite(fit.std_errors).all()
    # alpha = 0.8 in the design; well away from the boundary either way.
    assert np.exp(fit.params["ln_alpha"]) > 0.3
