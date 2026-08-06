"""Stata ``eventstudyinteract`` parity for ``sun_abraham(control_cohort=)``.

``eventstudyinteract`` takes ``control_cohort(varname)`` — a *binary
variable* naming the control cohort, which "can be never-treated units or
last-treated units". StatsPAI previously inferred the reference group and
offered no way to nominate it, so a design whose never-treated group is
contaminated (or absent) could not be expressed.

Golden numbers from Stata 18 MP, ``eventstudyinteract`` v0.1 (Sun 2022) on
``mpdta``; generating do-file ``tests/stata_parity/83_sunab_control_cohort.do``.

Tolerances
----------
ATT is pinned at 1e-6 — the observed worst case is 5.1e-8, driven by
reghdfe's absorb tolerance versus StatsPAI's dense solve.

SE is pinned at 0.2% *relative*. What remains after the share-variance
fix (below) is a **uniform** offset — 0.020% for the control_cohort=2007
fit and 0.081% for the never-treated fit — reflecting reghdfe's
small-sample cluster correction, whose K counts absorbed fixed effects
differently from StatsPAI's sandwich. Uniformity across relative times is
the diagnostic that matters: a per-relative-time *pattern* in the gap
means a real estimator difference, not a scaling convention.

Regression history
------------------
These fixtures caught a live SE defect. StatsPAI computed only
``w' Var(β̂) w`` and dropped the cohort-share term ``β' Var(ŵ) β`` from
Sun & Abraham (2021) Prop. 3. Because Var(ŵ) is degenerate when a single
cohort is eligible, the omission was invisible at most relative times
(gap 0.02%) and surfaced only where two or more cohorts contributed —
2.01% at e=1 here, always **understating** the SE. The parametrized
single-vs-multi cohort test below pins the property directly so the term
cannot be dropped again unnoticed.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
import pytest

import statspai as sp

_MPDTA = (
    pathlib.Path(__file__).resolve().parents[1]
    / "orig_parity"
    / "data"
    / "02_mpdta_original.csv"
)

ATT_ATOL = 1e-6
SE_RTOL = 0.002

# eventstudyinteract ... control_cohort(never)
STATA_NEVERTREATED = {
    -4: (0.0033063516, 0.0245550964),
    -3: (0.0250218088, 0.0181950652),
    -2: (0.0244587152, 0.0142962529),
    0: (-0.0199318068, 0.0118761305),
    1: (-0.0509573570, 0.0169640080),
    2: (-0.1372586996, 0.0365894800),
    3: (-0.1008113715, 0.0345042783),
}

# eventstudyinteract ... control_cohort(c2007), c2007 = first_treat==2007.
# g_m4 comes back exactly 0 in Stata (no estimable cohort at that lead once
# 2007 becomes the reference); StatsPAI omits the row instead of reporting
# a spurious zero, so it is not part of the comparison.
STATA_CONTROL_2007 = {
    -3: (0.0045018093, 0.0309631845),
    -2: (0.0019392060, 0.0191071653),
    0: (-0.0034213824, 0.0134061135),
    1: (-0.0423726647, 0.0165548420),
    2: (-0.1362743085, 0.0355242405),
    3: (-0.0920698754, 0.0334977475),
}


def _mpdta() -> pd.DataFrame:
    return pd.read_csv(_MPDTA)


def _event_study(**kwargs) -> dict:
    res = sp.sun_abraham(
        _mpdta(), y="lemp", g="first_treat", t="year", i="countyreal", **kwargs
    )
    return {
        int(r.relative_time): (float(r.att), float(r.se))
        for r in res.detail.itertuples()
    }


def _assert_matches(got: dict, want: dict, label: str) -> None:
    missing = set(want) - set(got)
    assert not missing, f"{label}: missing relative times {sorted(missing)}"
    for e, (att, se) in want.items():
        got_att, got_se = got[e]
        assert got_att == pytest.approx(
            att, abs=ATT_ATOL
        ), f"{label}: ATT at e={e} is {got_att:.10f}, Stata {att:.10f}"
        assert got_se == pytest.approx(
            se, rel=SE_RTOL
        ), f"{label}: SE at e={e} is {got_se:.10f}, Stata {se:.10f}"


class TestControlCohortParity:
    def test_default_matches_eventstudyinteract_nevertreated(self):
        """The inherited default must still reproduce the Stata baseline."""
        _assert_matches(_event_study(), STATA_NEVERTREATED, "nevertreated")

    def test_control_cohort_value_matches_stata(self):
        """Nominating the 2007 cohort reproduces control_cohort(c2007)."""
        _assert_matches(
            _event_study(control_cohort=2007), STATA_CONTROL_2007, "control_cohort=2007"
        )

    def test_control_cohort_indicator_column_matches_value_form(self):
        """The Stata spelling (0/1 column) and the shorthand must agree."""
        data = _mpdta()
        data["c2007"] = (data["first_treat"] == 2007).astype(int)
        res_col = sp.sun_abraham(
            data,
            y="lemp",
            g="first_treat",
            t="year",
            i="countyreal",
            control_cohort="c2007",
        )
        by_col = {
            int(r.relative_time): float(r.att) for r in res_col.detail.itertuples()
        }
        by_val = {e: v[0] for e, v in _event_study(control_cohort=2007).items()}
        assert by_col == pytest.approx(by_val, abs=1e-12)

    def test_control_cohort_zero_reproduces_nevertreated_exactly(self):
        """control_cohort=0 selects the never-treated: must be bit-identical."""
        default = _event_study()
        explicit = _event_study(control_cohort=0)
        assert set(default) == set(explicit)
        for e in default:
            assert explicit[e][0] == default[e][0], f"ATT drifted at e={e}"
            assert explicit[e][1] == default[e][1], f"SE drifted at e={e}"

    def test_reference_cohort_is_excluded_from_estimated_cohorts(self):
        """The control cohort must not also be estimated as a treated cohort."""
        res = sp.sun_abraham(
            _mpdta(),
            y="lemp",
            g="first_treat",
            t="year",
            i="countyreal",
            control_cohort=2007,
        )
        assert 2007 not in res.diagnostics["cohorts"]
        assert res.diagnostics["control_cohort"] == "first_treat in [2007]"


class TestCohortShareVariance:
    """Sun & Abraham (2021) Prop. 3 term 2: β' Var(ŵ) β.

    Dropping this term is invisible wherever one cohort is eligible, so
    these tests target the multi-cohort cells specifically.
    """

    @pytest.mark.parametrize(
        "kwargs,stata",
        [
            (dict(control_cohort=2007), STATA_CONTROL_2007),
            (dict(), STATA_NEVERTREATED),
        ],
        ids=["control_cohort=2007", "nevertreated"],
    )
    def test_se_gap_is_uniform_across_cohort_counts(self, kwargs, stata):
        """The Stata/StatsPAI SE ratio must not depend on how many cohorts
        contribute.

        Before the share-variance term was added, this ratio was ~1.000 at
        single-cohort relative times and ~0.980 at two-cohort ones. A flat
        ratio is what says the remaining gap is a scaling convention.
        """
        res = sp.sun_abraham(
            _mpdta(), y="lemp", g="first_treat", t="year", i="countyreal", **kwargs
        )
        ratios, counts = [], []
        for row in res.detail.itertuples():
            e = int(row.relative_time)
            if e in stata:
                ratios.append(float(row.se) / stata[e][1])
                counts.append(int(row.n_cohorts))
        assert max(counts) >= 2, "fixture must exercise a multi-cohort cell"
        assert min(counts) == 1, "fixture must exercise a single-cohort cell"
        spread = max(ratios) - min(ratios)
        assert spread < 5e-4, (
            f"SE ratio varies with cohort count (spread {spread:.2e}); the "
            f"cohort-share variance term looks wrong. ratios={ratios}, "
            f"n_cohorts={counts}"
        )

    def test_share_term_strictly_increases_multi_cohort_se(self):
        """The added term is a quadratic form in a PSD matrix: SE can only rise."""
        from statspai.did.sun_abraham import _cohort_share_vcov

        shares = np.array([0.3, 0.5, 0.2])
        v = _cohort_share_vcov(shares, n_obs=500)
        eig = np.linalg.eigvalsh(v)
        assert eig.min() > -1e-12, "share covariance must be PSD"
        beta = np.array([1.0, -2.0, 0.5])
        assert float(beta @ v @ beta) > 0.0

    def test_share_vcov_is_degenerate_for_one_cohort(self):
        """ŵ ≡ 1 carries no uncertainty, so the term must vanish exactly."""
        from statspai.did.sun_abraham import _cohort_share_vcov

        v = _cohort_share_vcov(np.array([1.0]), n_obs=500)
        assert v.shape == (1, 1)
        assert v[0, 0] == 0.0

    def test_share_vcov_matches_closed_form_multinomial(self):
        """Pin the algebra that replaces eventstudyinteract's avar sandwich."""
        from statspai.did.sun_abraham import _cohort_share_vcov

        shares = np.array([0.25, 0.75])
        n = 400
        got = _cohort_share_vcov(shares, n_obs=n)
        assert got[0, 0] == pytest.approx(0.25 * 0.75 / n)
        assert got[1, 1] == pytest.approx(0.75 * 0.25 / n)
        assert got[0, 1] == pytest.approx(-0.25 * 0.75 / n)
        # rows sum to zero: shares are constrained to sum to one
        assert got.sum(axis=1) == pytest.approx(np.zeros(2), abs=1e-15)

    def test_zero_observations_is_handled_not_divided_by(self):
        from statspai.did.sun_abraham import _cohort_share_vcov

        v = _cohort_share_vcov(np.array([0.5, 0.5]), n_obs=0)
        assert np.all(v == 0.0)


class TestControlCohortValidation:
    def test_unknown_column_name_rejected(self):
        with pytest.raises(ValueError, match="not a column"):
            sp.sun_abraham(
                _mpdta(),
                y="lemp",
                g="first_treat",
                t="year",
                i="countyreal",
                control_cohort="no_such_column",
            )

    def test_non_binary_column_rejected(self):
        """A continuous column is a user error, not a silent truthiness cast."""
        with pytest.raises(ValueError, match="binary 0/1 indicator"):
            sp.sun_abraham(
                _mpdta(),
                y="lemp",
                g="first_treat",
                t="year",
                i="countyreal",
                control_cohort="lpop",
            )

    def test_absent_cohort_value_rejected(self):
        with pytest.raises(ValueError, match="do not occur"):
            sp.sun_abraham(
                _mpdta(),
                y="lemp",
                g="first_treat",
                t="year",
                i="countyreal",
                control_cohort=1999,
            )

    def test_selecting_every_cohort_leaves_nothing_to_estimate(self):
        with pytest.raises(ValueError, match="No non-reference cohorts"):
            sp.sun_abraham(
                _mpdta(),
                y="lemp",
                g="first_treat",
                t="year",
                i="countyreal",
                control_cohort=[0, 2004, 2006, 2007],
            )

    def test_multiple_control_cohorts_accepted(self):
        """A sequence is a legitimate spelling: pool 0 and 2007 as controls."""
        res = sp.sun_abraham(
            _mpdta(),
            y="lemp",
            g="first_treat",
            t="year",
            i="countyreal",
            control_cohort=[0, 2007],
        )
        assert sorted(res.diagnostics["cohorts"]) == [2004, 2006]
