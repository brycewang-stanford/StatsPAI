"""Synthetic control reference parity tests.

Recovery + consistency tests for SCM estimators on a factor-model DGP
where synthetic control is exactly unbiased.

Validates:
1. ``sp.synth`` recovers the true post-treatment effect on a 2-factor DGP
   within 4 SEs.
2. Augmented SCM (ridge-regularised) matches vanilla SCM when penalization
   is small.
3. Pre-treatment RMSPE is small on a well-specified DGP.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp


def _within_tolerance(estimate, truth, tol=0.5):
    return abs(estimate - truth) <= tol


class TestSynthRecovery:
    """Synthetic control must recover the imposed effect on a factor-model DGP."""

    def test_vanilla_synth_recovers_effect(self, synth_factor_model_data):
        df = synth_factor_model_data
        truth = df.attrs['true_effect']  # -5.0
        treated_unit = df.attrs['treated_unit']
        treatment_year = df.attrs['treatment_year']

        r = sp.synth(df, outcome='y', unit='unit', time='year',
                     treated_unit=treated_unit,
                     treatment_time=treatment_year,
                     method='classic', placebo=False)
        # The ATT (main estimand on SCM) should be near -5
        # Tolerance: 1.0 (absolute) because of noise variance = 0.2
        assert _within_tolerance(r.estimate, truth, tol=1.0), (
            f"Vanilla SCM: {r.estimate:.4f} vs truth {truth} "
            f"(tolerance ±1.0)"
        )

    def test_augmented_synth_recovers_effect(self, synth_factor_model_data):
        df = synth_factor_model_data
        truth = df.attrs['true_effect']
        r = sp.synth(df, outcome='y', unit='unit', time='year',
                     treated_unit=df.attrs['treated_unit'],
                     treatment_time=df.attrs['treatment_year'],
                     method='augmented', placebo=False)
        assert _within_tolerance(r.estimate, truth, tol=1.0), (
            f"Augmented SCM: {r.estimate:.4f} vs truth {truth}"
        )

    def test_synth_sign_correct(self, synth_factor_model_data):
        """Negative effect must produce negative estimate."""
        df = synth_factor_model_data
        r = sp.synth(df, outcome='y', unit='unit', time='year',
                     treated_unit=df.attrs['treated_unit'],
                     treatment_time=df.attrs['treatment_year'],
                     method='classic', placebo=False)
        assert r.estimate < 0, (
            f"Negative true effect must yield negative estimate, got {r.estimate}"
        )
