"""Tests for the arviz ``_az_hdi_compat`` kwarg-compat shim."""
from __future__ import annotations

import numpy as np
import pytest

pymc = pytest.importorskip(
    "pymc",
    reason="PyMC is an optional dependency; skip Bayesian tests if missing.",
)

from statspai.bayes._base import _az_hdi_compat


def test_shim_accepts_hdi_prob_and_returns_length_2():
    """On the currently-installed arviz, shim forwards the
    ``hdi_prob`` kwarg and returns a length-2 numpy array."""
    rng = np.random.default_rng(0)
    samples = rng.normal(size=1000)
    out = _az_hdi_compat(samples, hdi_prob=0.95)
    assert isinstance(out, np.ndarray)
    assert out.shape == (2,)
    assert out[0] < out[1]
    # Rough sanity: HDI ≈ ±1.96 on standard normal
    assert -2.5 < out[0] < -1.5
    assert 1.5 < out[1] < 2.5


def test_shim_handles_typeerror_and_falls_back_to_prob(monkeypatch):
    """Simulate arviz ≥ 0.18 by monkey-patching ``az.hdi`` to reject
    the ``hdi_prob`` kwarg and only accept ``prob``. The shim must
    catch the TypeError and re-try with ``prob``, returning the same
    numerical answer."""
    import arviz as az

    original_hdi = az.hdi

    def _future_hdi(samples, prob=0.95, **kwargs):
        """Simulates future arviz: only accepts ``prob``."""
        if 'hdi_prob' in kwargs:
            raise TypeError(
                "hdi() got an unexpected keyword argument 'hdi_prob'"
            )
        return original_hdi(samples, hdi_prob=prob)

    monkeypatch.setattr(az, 'hdi', _future_hdi)

    rng = np.random.default_rng(1)
    samples = rng.normal(size=500)
    out = _az_hdi_compat(samples, hdi_prob=0.9)
    assert out.shape == (2,)
    assert out[0] < out[1]


def test_shim_matches_direct_az_hdi_on_current_arviz():
    """On the version in the current env the shim should return
    exactly what a direct ``az.hdi(...)`` call returns."""
    import arviz as az
    rng = np.random.default_rng(2)
    samples = rng.normal(size=500)
    via_shim = _az_hdi_compat(samples, hdi_prob=0.8)
    via_direct = np.asarray(az.hdi(samples, hdi_prob=0.8)).ravel()
    np.testing.assert_allclose(via_shim, via_direct, atol=1e-12)


def test_shim_handles_both_kwargs_rejected(monkeypatch):
    """If BOTH kwargs are rejected (some hypothetical future arviz),
    the shim propagates the TypeError rather than silently succeeding
    — users should hear about the breakage loudly."""
    import arviz as az

    def _broken_hdi(*args, **kwargs):
        raise TypeError("unexpected kwarg")

    monkeypatch.setattr(az, 'hdi', _broken_hdi)

    with pytest.raises(TypeError):
        _az_hdi_compat(np.array([1.0, 2.0, 3.0]), hdi_prob=0.95)
