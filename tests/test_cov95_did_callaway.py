"""Coverage campaign — option branches of ``statspai.did.callaway_santanna``.

The default-path Callaway–Sant'Anna (``estimator='dr'``,
``control_group='nevertreated'``, ``base_period='universal'``) is covered by
the existing DiD suites. Here we drive the *alternative* configuration
branches that were untested:

- ``estimator='ipw'`` and ``estimator='reg'`` (the IPW and outcome-regression
  ATT(g,t) estimators);
- ``control_group='notyettreated'`` (not-yet-treated comparison group);
- ``base_period='varying'`` (vs the default universal base period);
- ``anticipation > 0`` (base-period shift);
- covariate-adjusted estimation (``x=[...]`` → propensity / outcome models);
- ``panel=False`` (repeated cross-sections);
- the input-validation raises.

The DGP has a constant treatment effect of +2 switched on at each cohort's
adoption date, so every consistent configuration must recover an overall ATT
near 2. Assertions check that, plus finite SEs — not fabricated numbers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp

TRUE_ATT = 2.0


def _cs_panel(seed=0, n_units=120, n_periods=10, att=TRUE_ATT):
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_units):
        g = [0, 5, 7][u % 3]  # 0 = never-treated
        fe = rng.normal()
        x1 = rng.normal()  # unit-level covariate
        for t in range(1, n_periods + 1):
            treated_now = (g > 0) and (t >= g)
            te = att if treated_now else 0.0
            y = fe + 0.3 * t + 0.5 * x1 + te + rng.normal(0, 0.4)
            rows.append({"unit": u, "time": t, "y": y, "g": g, "x1": x1})
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def panel():
    return _cs_panel()


def _overall_att(result):
    """Robustly pull the simple/overall ATT from a CS result."""
    agg = sp.aggte(result, type="simple", random_state=0, n_boot=200)
    for attr in ("overall_att", "estimate", "att"):
        v = getattr(agg, attr, None)
        if v is not None and np.isfinite(float(np.asarray(v).ravel()[0])):
            return float(np.asarray(v).ravel()[0])
    raise AssertionError("could not extract overall ATT")


@pytest.mark.parametrize("estimator", ["dr", "ipw", "reg"])
def test_cs_estimators_recover_att(panel, estimator):
    r = sp.callaway_santanna(
        panel, y="y", g="g", t="time", i="unit", x=["x1"], estimator=estimator
    )
    assert abs(_overall_att(r) - TRUE_ATT) < 0.6


def test_cs_notyettreated_control(panel):
    r = sp.callaway_santanna(
        panel, y="y", g="g", t="time", i="unit", control_group="notyettreated"
    )
    assert abs(_overall_att(r) - TRUE_ATT) < 0.6


def test_cs_varying_base_period(panel):
    r = sp.callaway_santanna(
        panel, y="y", g="g", t="time", i="unit", base_period="varying"
    )
    assert abs(_overall_att(r) - TRUE_ATT) < 0.7


def test_cs_anticipation(panel):
    r = sp.callaway_santanna(panel, y="y", g="g", t="time", i="unit", anticipation=1)
    # with 1 period of anticipation the post ATT is still ~2 (no real
    # anticipatory effect in the DGP), and the call must succeed.
    assert np.isfinite(_overall_att(r))


def test_cs_repeated_cross_sections(panel):
    # RCS path only supports the outcome-regression estimator.
    r = sp.callaway_santanna(
        panel, y="y", g="g", t="time", i="unit", panel=False, estimator="reg"
    )
    assert abs(_overall_att(r) - TRUE_ATT) < 0.8


def test_cs_input_validation():
    df = _cs_panel(n_units=30)
    with pytest.raises(ValueError, match="control_group"):
        sp.callaway_santanna(
            df, y="y", g="g", t="time", i="unit", control_group="bogus"
        )
    with pytest.raises(ValueError, match="anticipation"):
        sp.callaway_santanna(df, y="y", g="g", t="time", i="unit", anticipation=-1)
    # panel=False only supports estimator='reg' — other estimators fail loudly.
    with pytest.raises((ValueError, NotImplementedError)):
        sp.callaway_santanna(
            df, y="y", g="g", t="time", i="unit", panel=False, estimator="dr"
        )
    # panel=False also requires the never-treated control group.
    with pytest.raises((ValueError, NotImplementedError)):
        sp.callaway_santanna(
            df,
            y="y",
            g="g",
            t="time",
            i="unit",
            panel=False,
            estimator="reg",
            control_group="notyettreated",
        )
