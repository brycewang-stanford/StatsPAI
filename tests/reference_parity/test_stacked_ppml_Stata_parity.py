"""``sp.stacked_did(family='poisson', spec=, absorb=, control_group=)``.

A replication's stacked DID was a PPML with a single treated x post
coefficient, extra fixed effects on top of unit x cohort and year x
cohort, and later-treated firms as controls only before their own
treatment. ``sp.stacked_did`` was linear, event-study only, and could not
express that stacking rule, so the stack had to be built by hand.

Reference: Stata 18 on ``_fixtures/stacked_ppml.csv``, stacking by hand
exactly that way and fitting ``ppmlhdfe y tp x, absorb(uc tc [id
city#year]) vce(cluster city)`` and the event-study version with ``lincom``
of the post coefficients (``_generate_stacked_ppml_Stata.do``). Estimates,
SEs and N agree to 1e-9 (ppmlhdfe's own convergence tolerance is 1e-8).
"""

from __future__ import annotations

import json
import pathlib
import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp

_FIX = pathlib.Path(__file__).parent / "_fixtures"
REF = json.loads((_FIX / "stacked_ppml_Stata.json").read_text(encoding="utf-8"))
KW = dict(
    y="y",
    group="id",
    time="year",
    first_treat="g",
    window=(-3, 2),
    controls=["x"],
    cluster="city",
    family="poisson",
    control_group="notyettreated_rows",
)


@pytest.fixture(scope="module")
def data():
    return pd.read_csv(_FIX / "stacked_ppml.csv")


def _fit(data, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.stacked_did(data, **{**KW, **kw})


def test_pooled_ppml(data):
    r = _fit(data, spec="pooled")
    ref = REF["pooled"]
    np.testing.assert_allclose(r.estimate, ref["b"], rtol=1e-9)
    np.testing.assert_allclose(r.se, ref["se"], rtol=1e-9)
    assert r.n_obs == ref["N"]


def test_pooled_ppml_with_extra_fixed_effects(data):
    r = _fit(data, spec="pooled", absorb="id + city#year")
    ref = REF["pooled_absorb"]
    np.testing.assert_allclose(r.estimate, ref["b"], rtol=1e-9)
    np.testing.assert_allclose(r.se, ref["se"], rtol=1e-9)
    assert r.n_obs == ref["N"]


def test_event_study_ppml(data):
    r = _fit(data, spec="event_study")
    es = r.detail[r.detail["relative_time"] != -1]
    ref = REF["event"]
    np.testing.assert_allclose(es["att"], ref["b"], rtol=1e-9)
    np.testing.assert_allclose(es["se"], ref["se"], rtol=1e-9)
    np.testing.assert_allclose(r.estimate, REF["event_att"]["b"], rtol=1e-9)
    np.testing.assert_allclose(r.se, REF["event_att"]["se"], rtol=1e-9)


def test_linear_event_study_with_controls_runs(data):
    """The ATT delta method used the full covariance (event times + controls)
    against an event-time weight vector and crashed whenever controls= was
    given."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = sp.stacked_did(
            data,
            y="y",
            group="id",
            time="year",
            first_treat="g",
            window=(-3, 2),
            controls=["x"],
        )
        base = sp.stacked_did(
            data, y="y", group="id", time="year", first_treat="g", window=(-3, 2)
        )
    assert np.isfinite(r.se) and r.se > 0
    assert r.estimate != base.estimate


def test_rows_rule_differs_from_unit_rule(data):
    rows = _fit(data, spec="pooled")
    units = _fit(data, spec="pooled", control_group="notyettreated")
    assert rows.n_obs > units.n_obs
