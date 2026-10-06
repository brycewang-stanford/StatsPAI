"""``sp.callaway_santanna`` and ``sp.cs_jackknife`` say so when the
never-treated comparison group is very small.

R ``did`` (>= 2.5.0) stops when ``control_group = "nevertreated"`` and the
never-treated group has fewer than five units plus one per covariate. The
estimate is still defined, so StatsPAI reports it and warns. These tests pin
the cutoff, that the numbers do not move, and that the jackknife speaks up
when only its delete-one samples fall below the cutoff, which is the case R
``didjack`` cannot run.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import AssumptionWarning


def _panel(n_never: int, n_treated_per_cohort: int = 15, seed: int = 0):
    rng = np.random.default_rng(seed)
    rows = []
    cohorts = [0] * n_never + [3] * n_treated_per_cohort + [4] * n_treated_per_cohort
    for u, g in enumerate(cohorts):
        a, xv = rng.normal(), rng.normal()
        for tt in range(1, 6):
            d = 1.0 if g and tt >= g else 0.0
            rows.append((u, tt, g, xv, a + 0.2 * tt + 0.5 * d + rng.normal(0, 0.3)))
    return pd.DataFrame(rows, columns=["unit", "time", "g", "x", "y"])


def _never_warnings(caught):
    return [w for w in caught if "never-treated" in str(w.message)]


def _fit(df, **kw):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        fit = sp.callaway_santanna(
            df, y="y", g="g", t="time", i="unit", estimator="reg", **kw
        )
    return fit, _never_warnings(caught)


@pytest.mark.parametrize("n_never, warns", [(4, True), (5, False), (12, False)])
def test_cutoff_is_five_units(n_never, warns):
    _, hits = _fit(_panel(n_never))
    assert bool(hits) is warns
    if warns:
        assert issubclass(hits[0].category, AssumptionWarning)
        assert f"only {n_never} never-treated" in str(hits[0].message)


def test_each_covariate_raises_the_cutoff_by_one():
    df = _panel(5)
    assert not _fit(df)[1]
    assert _fit(df, x=["x"])[1]


def test_not_yet_treated_controls_do_not_warn():
    assert not _fit(_panel(4), control_group="notyettreated")[1]


def test_the_warning_does_not_change_the_estimate():
    df = _panel(4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        quiet = sp.callaway_santanna(
            df, y="y", g="g", t="time", i="unit", estimator="reg"
        )
    loud, hits = _fit(df)
    assert hits
    assert loud.estimate == quiet.estimate
    assert loud.se == quiet.se


def test_repeated_cross_sections_count_observations_per_period():
    df = _panel(4)
    df["row"] = np.arange(len(df))
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sp.callaway_santanna(
            df, y="y", g="g", t="time", i="row", estimator="reg", panel=False
        )
    hits = _never_warnings(caught)
    assert hits and "only 4 never-treated observations per period" in str(
        hits[0].message
    )


def test_jackknife_warns_when_only_the_replicates_fall_short():
    """Five never-treated units pass; deleting one leaves four."""
    df = _panel(5)
    assert not _fit(df)[1]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        jk = sp.cs_jackknife(df, y="y", g="g", time="time", id="unit", estimator="reg")
    hits = [w for w in caught if "cs_jackknife" in str(w.message)]
    assert len(hits) == 1
    assert "fall to 4" in str(hits[0].message)
    assert np.isfinite(jk.se) and jk.model_info["n_replicates"] == 35


def test_jackknife_is_silent_with_room_to_spare():
    df = _panel(12)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        sp.cs_jackknife(df, y="y", g="g", time="time", id="unit", estimator="reg")
    assert not _never_warnings(caught)
