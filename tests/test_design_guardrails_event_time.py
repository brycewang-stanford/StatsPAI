"""Event-time dummies are not "assigned at the cohort level".

In a staggered design every event-time dummy ``1{t - g = k}`` is constant
within the cohort x period cells of the adoption-date column ``g``.  For a
lead or lag that only some cohorts reach (``k = 3`` when only the earliest
cohort is observed three years after adoption) several cohorts share the
all-zero path, so the cohort column no longer has one treatment path per
level and the assignment-level check used to report it as the level the
treatment was assigned at, recommending clusters on a handful of cohorts.
Found on the personal-bankruptcy replication (267 cities, four cohorts, the
k = 3 dummy flagged as "assigned at the 'gvar' level (5 groups)").

Each test asserts a true proposition: the adoption-date column is never the
reported assignment level, and treatment assigned by state is still flagged
at the state level when the data also carry a cohort column.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.core._design_guardrails import _adoption_dates
from statspai.core._spec_counter import reset_specification_counter
from statspai.exceptions import AssumptionWarning


@pytest.fixture(autouse=True)
def _fresh_counter():
    reset_specification_counter()
    yield
    reset_specification_counter()


def _levels(result):
    return [
        v["value"]["assignment_level"]
        for v in result.violations()
        if v["test"] == "treatment_cluster_level"
    ]


def _cities(first=2015, never=0, seed=0):
    """Four cohorts (4 / 15 / 10 / 10 units) and 228 never-treated units."""
    rng = np.random.default_rng(seed)
    sizes = {first + 4: 4, first + 5: 15, first + 6: 10, first + 7: 10}
    cohorts = [g for g, n in sizes.items() for _ in range(n)] + [never] * 228
    rows = []
    for i, g in enumerate(cohorts):
        a = rng.normal()
        for t in range(first, first + 8):
            treated = g != never and t >= g
            rows.append((i, t, g, a + 0.01 * treated + rng.normal(0, 0.1)))
    df = pd.DataFrame(rows, columns=["city", "year", "gvar", "y"])
    rel = np.where(df.gvar != never, df.year - df.gvar, -99)
    for k in range(4):
        df[f"L{k}"] = (rel == k).astype(int)
    for k in range(2, 8):
        df[f"F{k}"] = (rel == -k).astype(int)
    return df


EVENT = [f"F{k}" for k in range(2, 8)] + [f"L{k}" for k in range(4)]


@pytest.mark.parametrize(
    "first, never",
    [(2015, 0), (1, 0), (2015, 9999)],
    ids=["years-never0", "periods-never0", "years-never-after-panel"],
)
@pytest.mark.parametrize("term", ["L3", "L0", "F5", "F7"])
def test_single_event_dummy_is_not_assigned_at_the_cohort(first, never, term):
    df = _cities(first=first, never=never)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", AssumptionWarning)
        res = sp.feols(f"y ~ {term} | city + year", df, cluster="city")
    assert _levels(res) == []


def test_full_event_study_is_not_assigned_at_the_cohort():
    df = _cities()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", AssumptionWarning)
        res = sp.feols(
            "y ~ " + " + ".join(EVENT) + " | city + year", df, cluster="city"
        )
    assert _levels(res) == []


def test_adoption_dates_recognise_event_dummies_and_absorbing_treatment():
    df = _cities()
    for term in EVENT:
        assert _adoption_dates(df.gvar, df.year, df[term].to_numpy(float))
    absorbing = ((df.gvar > 0) & (df.year >= df.gvar)).to_numpy(float)
    assert _adoption_dates(df.gvar, df.year, absorbing)
    # a dummy that is not a function of year - gvar
    other = ((df.gvar > 0) & (df.city % 2 == 0) & (df.year >= df.gvar)).to_numpy(float)
    assert not _adoption_dates(df.gvar, df.year, other)


def _state_panel(seed=1):
    """Individuals in 40 states, adoption by state at three dates."""
    rng = np.random.default_rng(seed)
    rows = []
    for s in range(40):
        adopt = 4 + s % 3 if s % 2 == 0 else 0
        for i in range(12):
            for t in range(1, 9):
                rows.append((s * 100 + i, s, adopt, t, int(adopt > 0 and t >= adopt)))
    df = pd.DataFrame(rows, columns=["id", "state", "g", "year", "treated"])
    shock = rng.normal(0, 1, (40, 9))
    df["y"] = 0.5 * df.treated + shock[df.state, df.year] + rng.normal(0, 1, len(df))
    df["e0"] = ((df.g > 0) & (df.year == df.g)).astype(int)
    return df


def test_state_assignment_is_still_flagged_with_a_cohort_column_present():
    df = _state_panel()
    assert not _adoption_dates(df.state, df.year, df.treated.to_numpy(float))
    for term in ("treated", "e0"):
        with pytest.warns(AssumptionWarning, match="assigned at the 'state' level"):
            res = sp.feols(f"y ~ {term} | id + year", df, cluster="id")
        assert _levels(res) == ["state"]
