"""Analytical parity: sp.bayes_synth recovers a known synthetic-control lift.

``sp.bayes_synth`` fits convex donor weights (Dirichlet prior on the simplex)
to a treated unit's pre-treatment path and reports the posterior of the
post-period ATT. It has no cross-package reference, so the honest grade is
``analytical-only`` — known-DGP recovery on a deterministic panel.

When the treated unit is a convex combination of two donors in the
pre-period (``0.5*A + 0.5*B`` plus small noise), a synthetic control exists,
and injecting a known additive shift ``delta`` in the post-period makes the
posterior-mean ATT recover ``delta`` within the sampler band. The posterior
donor weights live on the simplex (sum to 1) by construction.

Requires the optional ``bayes`` extra (PyMC); skips cleanly when absent,
matching the rest of the Bayesian suite. Small chains keep it fast.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

pytest.importorskip("pymc")
pytest.importorskip("arviz")

import statspai as sp  # noqa: E402

_T_PERIODS = 24
_TREAT_TIME = 16
_DELTA = -5.0  # injected post-period ATT


def _panel():
    rng = np.random.default_rng(0)
    t = np.arange(_T_PERIODS)
    donors = {
        "A": 10 + 0.5 * t + rng.normal(0, 0.15, _T_PERIODS),
        "B": 5 + 0.2 * t + rng.normal(0, 0.15, _T_PERIODS),
        "C": 8 - 0.1 * t + rng.normal(0, 0.15, _T_PERIODS),
        "D": 12 + 0.3 * t + rng.normal(0, 0.15, _T_PERIODS),
    }
    # Treated = 0.5*A + 0.5*B (+ small noise so the residual SD is identified).
    treated = 0.5 * donors["A"] + 0.5 * donors["B"] + rng.normal(0, 0.1, _T_PERIODS)
    treated[t >= _TREAT_TIME] += _DELTA

    rows = []
    for g, series in donors.items():
        for i in range(_T_PERIODS):
            rows.append({"state": g, "year": i, "y": series[i]})
    for i in range(_T_PERIODS):
        rows.append({"state": "TR", "year": i, "y": treated[i]})
    return pd.DataFrame(rows)


def test_bayes_synth_recovers_known_att():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sp.bayes_synth(
            _panel(),
            outcome="y",
            unit="state",
            time="year",
            treated_unit="TR",
            treatment_time=_TREAT_TIME,
            draws=400,
            tune=400,
            chains=2,
            target_accept=0.95,
            random_state=0,
            progressbar=False,
        )
    assert res.estimand == "ATT"
    # Posterior mean ATT recovers the injected lift within the sampler band.
    assert abs(res.posterior_mean - _DELTA) <= 0.6
    # HDI brackets the posterior mean.
    assert res.hdi_lower < res.posterior_mean < res.hdi_upper
    # Donor weights are convex (simplex): sum to 1.
    weights = res.model_info["weights"]
    total = (
        sum(weights.values())
        if isinstance(weights, dict)
        else float(np.sum(weights))
    )
    assert abs(total - 1.0) <= 1e-6
