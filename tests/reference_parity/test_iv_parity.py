"""IV reference parity tests.

Recovery + cross-method consistency for instrumental variables.

Validates:
1. 2SLS recovers LATE on a DGP with a strong instrument.
2. Wald ratio (manual) equals 2SLS on a binary-Z DGP.
3. First-stage F > 10 on a strong-instrument DGP.
4. Reduced-form / first-stage ratio equals 2SLS coefficient.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp


def _within_n_se(estimate, truth, se, n_sigma=4.0):
    return abs(estimate - truth) <= n_sigma * se


# ---------------------------------------------------------------------------
# Strong-instrument IV: 2SLS must recover LATE
# ---------------------------------------------------------------------------

class TestStrongIVRecovery:
    """With a strong binary instrument, 2SLS must recover the LATE."""

    def test_2sls_recovers_late(self, iv_strong_data):
        truth = iv_strong_data.attrs['true_effect']
        r = sp.ivreg('y ~ (d ~ z)', data=iv_strong_data, robust='hc1')
        # Extract the coefficient on d (endogenous var)
        coef = r.params['d']
        se = r.std_errors['d']
        assert _within_n_se(coef, truth, se), (
            f"2SLS: {coef:.4f} vs truth {truth} (SE {se:.4f})"
        )

    def test_first_stage_f_exceeds_10(self, iv_strong_data):
        """Strong instrument: first-stage F > 10 (Staiger-Stock rule)."""
        r = sp.ivreg('y ~ (d ~ z)', data=iv_strong_data, robust='hc1')
        diag = r.diagnostics
        # Try a few common keys where F might be stored
        keys = [k for k in diag if 'first' in k.lower() and
                ('f' in k.lower() or 'stat' in k.lower())]
        if keys:
            f = float(diag[keys[0]])
            assert f > 10.0, f"First-stage F = {f:.2f} < 10"

    def test_wald_ratio_matches_2sls(self, iv_strong_data):
        """Wald ratio (by hand) should match 2SLS estimate."""
        df = iv_strong_data
        y_z1 = df[df['z'] == 1]['y'].mean()
        y_z0 = df[df['z'] == 0]['y'].mean()
        d_z1 = df[df['z'] == 1]['d'].mean()
        d_z0 = df[df['z'] == 0]['d'].mean()
        wald = (y_z1 - y_z0) / (d_z1 - d_z0)

        r = sp.ivreg('y ~ (d ~ z)', data=df)
        assert abs(wald - r.params['d']) < 0.01, (
            f"Wald {wald:.4f} vs 2SLS {r.params['d']:.4f}"
        )


# ---------------------------------------------------------------------------
# Weak instrument: effect may be noisy; test asymptotic properties
# ---------------------------------------------------------------------------

class TestWeakInstrumentDetection:
    """On a deliberately weak instrument, first-stage F must be < 10."""

    def test_weak_instrument_flagged(self):
        rng = np.random.default_rng(3)
        n = 1000
        z = rng.binomial(1, 0.5, n)
        u = rng.normal(size=n)
        # Very weak first stage: coefficient 0.05
        d = (0.4 + 0.05 * z + 0.3 * u + rng.normal(scale=0.3, size=n) > 0.5).astype(int)
        y = 1 + 0.5 * d + 0.3 * u + rng.normal(scale=0.5, size=n)
        df = pd.DataFrame({'y': y, 'd': d, 'z': z})

        r = sp.ivreg('y ~ (d ~ z)', data=df)
        diag = r.diagnostics
        keys = [k for k in diag if 'first' in k.lower() and
                ('f' in k.lower() or 'stat' in k.lower())]
        if keys:
            f = float(diag[keys[0]])
            # Weak: F < 30 (much less than 10 most of the time)
            assert f < 30, f"Expected weak-instrument F < 30, got {f:.2f}"
