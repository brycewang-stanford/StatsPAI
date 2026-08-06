"""The ATT must say so when it does not cover every treated unit.

Matching drops treated units for three reasons — an exhausted control pool
under ``replace=False``, a caliper with no admissible donor, and
common-support trimming — and the reported effect is then an average over a
*non-randomly selected* subset of the treated.  Before v1.22 all three were
silent: ``sp.match`` returned a number labelled "ATT" that could cover as
little as a quarter of the treated sample with no indication whatsoever.

The arithmetic that makes the ``replace=False`` case predictable: matching
without replacement needs ``n_matches * n_treated <= n_control`` before it
can even form the requested matches.  As that constraint binds, the units
matched last take the leftovers, and past it they get nothing.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp

_WARNS = "on-support treated units found no match"


def _make(n_treated: int, n_control: int, seed: int = 0) -> pd.DataFrame:
    """A frame with exactly the requested arm sizes and overlapping scores."""
    rng = np.random.default_rng(seed)
    n = n_treated + n_control
    d = np.zeros(n, dtype=int)
    d[:n_treated] = 1
    x1 = rng.normal(size=n) + 0.5 * d
    x2 = rng.normal(size=n)
    y = 1.0 + 2.0 * d + 0.7 * x1 - 0.3 * x2 + rng.normal(scale=0.5, size=n)
    return pd.DataFrame({"x1": x1, "x2": x2, "d": d, "y": y})


def _fit(df, **kw):
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        res = sp.match(
            df,
            y="y",
            treat="d",
            covariates=["x1", "x2"],
            method="psm",
            se_method="ai",
            **kw,
        )
    return res, [str(w.message) for w in caught]


class TestExhaustedControlPool:
    def test_warns_when_pool_cannot_cover_every_treated(self):
        # 60 treated, 40 controls: 20 treated cannot be matched 1:1.
        res, msgs = _fit(_make(60, 40), replace=False, n_matches=1)
        hits = [m for m in msgs if _WARNS in m]
        assert hits, "an exhausted control pool must not be silent"
        assert "EXCLUDED from the reported ATT" in hits[0]
        assert res.model_info["n_treated_unmatched"] == 20

    def test_warning_names_the_pool_arithmetic(self):
        _, msgs = _fit(_make(60, 40), replace=False, n_matches=1)
        hit = next(m for m in msgs if _WARNS in m)
        assert "replace=False needs n_matches x n_treated = 60" in hit
        assert "only 40 are on support" in hit
        assert "replace=True" in hit  # actionable

    def test_warning_reports_the_share_not_just_the_count(self):
        _, msgs = _fit(_make(60, 40), replace=False, n_matches=1)
        hit = next(m for m in msgs if _WARNS in m)
        assert "(33% of the treated sample)" in hit

    def test_k_multiplies_the_requirement(self):
        """k=4 needs four controls per treated, so it starves much earlier."""
        res, msgs = _fit(_make(30, 60), replace=False, n_matches=4)
        assert [m for m in msgs if _WARNS in m]
        assert res.model_info["n_treated_unmatched"] > 0

    def test_silent_when_the_pool_is_adequate(self):
        """The warning must not cry wolf on a well-specified design."""
        res, msgs = _fit(_make(30, 300), replace=False, n_matches=1)
        assert not [m for m in msgs if _WARNS in m]
        assert res.model_info["n_treated_unmatched"] == 0
        assert res.model_info["n_treated_partially_matched"] == 0

    def test_replacement_removes_the_constraint_entirely(self):
        """With replacement a single control can serve everyone."""
        res, msgs = _fit(_make(60, 40), replace=True, n_matches=1)
        assert not [m for m in msgs if _WARNS in m]
        assert res.model_info["n_treated_unmatched"] == 0

    def test_reported_att_is_over_the_matched_subset(self):
        """Document the estimand actually returned."""
        res, _ = _fit(_make(60, 40), replace=False, n_matches=1)
        md = res.matched_data
        matched_treated = (md["_treated"] == 1) & md["_y"].notna()
        assert int(matched_treated.sum()) == 40
        att = float(
            (md.loc[matched_treated, "y"] - md.loc[matched_treated, "_y"]).mean()
        )
        assert att == pytest.approx(res.estimate, abs=1e-9)


class TestCaliperDrops:
    def test_caliper_drops_are_also_reported(self):
        # A caliper this tight admits almost no donor.
        res, msgs = _fit(_make(40, 200), caliper=1e-6, replace=True)
        hits = [m for m in msgs if _WARNS in m]
        assert hits
        assert "caliper" in hits[0]
        assert res.model_info["n_treated_unmatched"] > 0

    def test_caliper_message_does_not_blame_the_pool(self):
        _, msgs = _fit(_make(40, 200), caliper=1e-6, replace=True)
        hit = next(m for m in msgs if _WARNS in m)
        assert "replace=False needs" not in hit


class TestDiagnosticsAreAlwaysRecorded:
    @pytest.mark.parametrize("replace", [True, False])
    def test_counts_present_even_when_nothing_is_dropped(self, replace):
        res, _ = _fit(_make(30, 300), replace=replace, n_matches=1)
        assert res.model_info["n_treated_unmatched"] == 0
        assert res.model_info["n_treated_partially_matched"] == 0

    def test_partial_matches_are_counted_separately_from_total_failures(self):
        """A unit with 2 of 4 requested matches is neither fully matched
        nor excluded, and the two states must not be conflated."""
        res, _ = _fit(_make(40, 60), replace=False, n_matches=4)
        info = res.model_info
        n_on = info["n_treated_on_support"]
        assert info["n_treated_unmatched"] + info["n_treated_partially_matched"] <= n_on
        assert info["n_matched_treated"] == n_on - info["n_treated_unmatched"]
