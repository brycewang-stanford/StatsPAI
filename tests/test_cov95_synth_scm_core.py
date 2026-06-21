"""Coverage campaign — core paths + guards of ``statspai.synth.scm``.

Targets reachable branches of the classic synthetic-control implementation
that the existing cov95 synth suite leaves uncovered:

- the input-validation guards (missing treated unit / treatment time, an
  unknown treated unit, donors that are all-NaN in the pre-period);
- the placebo-inference path (``placebo=True`` → per-donor placebo gaps);
- classic SCM with covariates + a penalization term;
- the fitted-result accessors used downstream.

DGP: one treated unit with a clean +4 post-treatment jump over several
clearly-imperfect donors, so the classic SCM recovers a positive effect.
Assertions check that recovery / the correct loud failures, never
fabricated numbers.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp

TRUE_EFFECT = 4.0
T_TREAT = 11


def _scm_panel(seed=0, n_donors=8, n_t=20, effect=TRUE_EFFECT):
    rng = np.random.default_rng(seed)
    units = [f"u{i}" for i in range(n_donors)] + ["treated"]
    rows = []
    for u in units:
        base = rng.normal(0, 1)
        fe = rng.normal(0, 0.5)
        for t in range(1, n_t + 1):
            eff = effect if (u == "treated" and t >= T_TREAT) else 0.0
            y = base + 0.2 * t + fe + eff + rng.normal(0, 0.3)
            rows.append({"unit": u, "time": t, "y": y})
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def panel():
    return _scm_panel()


def test_classic_scm_with_placebo_recovers_effect(panel):
    r = sp.synth(
        panel,
        outcome="y",
        unit="unit",
        time="time",
        treated_unit="treated",
        treatment_time=T_TREAT,
        method="classic",
        placebo=True,
    )
    assert r.estimate > 1.5  # clear positive effect (true +4)
    # placebo inference populated the result (per-donor placebo distribution)
    mi = getattr(r, "model_info", {}) or {}
    assert any("placebo" in str(k).lower() for k in mi) or r.pvalue is not None


def test_classic_scm_with_covariates_and_penalization():
    rng = np.random.default_rng(2)
    units = [f"u{i}" for i in range(8)] + ["treated"]
    rows = []
    for u in units:
        base = rng.normal()
        fe = rng.normal(0, 0.5)
        w = rng.normal()
        for t in range(1, 21):
            eff = TRUE_EFFECT if (u == "treated" and t >= T_TREAT) else 0.0
            rows.append(
                {
                    "unit": u,
                    "time": t,
                    "z": w + 0.1 * t,
                    "y": base + 0.2 * t + fe + 0.3 * w + eff + rng.normal(0, 0.3),
                }
            )
    df = pd.DataFrame(rows)
    r = sp.synth(
        df,
        outcome="y",
        unit="unit",
        time="time",
        treated_unit="treated",
        treatment_time=T_TREAT,
        method="classic",
        covariates=["z"],
        penalization=0.1,
        placebo=False,
    )
    assert np.isfinite(r.estimate)


def test_scm_requires_treated_unit_and_time(panel):
    with pytest.raises((ValueError, TypeError)):
        sp.synth(
            panel, outcome="y", unit="unit", time="time", method="classic"
        )  # no treated_unit / treatment_time


def test_scm_unknown_treated_unit_raises(panel):
    with pytest.raises(ValueError):
        sp.synth(
            panel,
            outcome="y",
            unit="unit",
            time="time",
            treated_unit="does_not_exist",
            treatment_time=T_TREAT,
            method="classic",
            placebo=False,
        )


def test_scm_all_nan_donors_pre_period_raises():
    # Every donor is NaN in the pre-period ⇒ no usable donor pool ⇒ loud fail.
    rng = np.random.default_rng(5)
    rows = []
    for u in [f"u{i}" for i in range(5)] + ["treated"]:
        for t in range(1, 13):
            if u != "treated" and t < T_TREAT:
                val = np.nan  # donors unusable pre-period
            else:
                val = rng.normal() + (
                    TRUE_EFFECT if (u == "treated" and t >= T_TREAT) else 0.0
                )
            rows.append({"unit": u, "time": t, "y": val})
    df = pd.DataFrame(rows)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(ValueError):
            sp.synth(
                df,
                outcome="y",
                unit="unit",
                time="time",
                treated_unit="treated",
                treatment_time=T_TREAT,
                method="classic",
                placebo=False,
            )
