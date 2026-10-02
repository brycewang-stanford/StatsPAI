"""Stata ``eventstudyinteract`` parity for ``sun_abraham(control_cohort=)``.

``eventstudyinteract`` takes ``control_cohort(varname)`` — a *binary
variable* naming the control cohort, which "can be never-treated units or
last-treated units". StatsPAI previously inferred the reference group and
offered no way to nominate it, so a design whose never-treated group is
contaminated (or absent) could not be expressed.

Golden numbers from Stata 18 MP, ``eventstudyinteract`` v0.1 (Sun 2022) on
``mpdta``; generating do-file
``tests/stata_parity/option_parity/83_sunab_control_cohort.do``.

Tolerances
----------
Relative 1e-9 on both the ATT and the SE, read from
``option_parity/results/83_sunab_control_cohort_Stata.json``. Observed
worst case: 1.9e-10 (ATT), 2.0e-11 (SE).

The first version of the fixture was held to 1e-6 absolute on the ATT and
0.2% on the SE. Two things have changed since. The SE offset then
attributed to reghdfe's small-sample correction went away with the
Sun-Abraham design-matrix fix (ghost cohort x event columns, nested time
effects in K). And the remaining 2e-5 relative gap in the ATT was Stata
running in single precision: ``import delimited`` stored ``lemp`` as
float. The do-file now imports ``asdouble`` under ``set type double``.

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

import json
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

RTOL = 1e-9

_STATA = json.loads(
    (
        pathlib.Path(__file__).resolve().parents[1]
        / "stata_parity"
        / "option_parity"
        / "results"
        / "83_sunab_control_cohort_Stata.json"
    ).read_text(encoding="utf-8")
)


def _rows(block: str, times: list) -> dict:
    """``{relative time: (att, se)}`` from the fixture's g_m<k> / g_p<k> keys."""
    out = {}
    for e in times:
        row = _STATA[block][f"g_{'m' if e < 0 else 'p'}{abs(e)}"]
        out[e] = (row["att"], row["se"])
    return out


# eventstudyinteract ... control_cohort(never)
STATA_NEVERTREATED = _rows("control_cohort_never", [-4, -3, -2, 0, 1, 2, 3])

# eventstudyinteract ... control_cohort(c2007), c2007 = first_treat==2007.
# g_m4 comes back exactly 0 in Stata (no estimable cohort at that lead once
# 2007 becomes the reference); StatsPAI omits the row instead of reporting
# a spurious zero, so it is not part of the comparison.
STATA_CONTROL_2007 = _rows("control_cohort_2007", [-3, -2, 0, 1, 2, 3])


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
            att, rel=RTOL
        ), f"{label}: ATT at e={e} is {got_att:.10f}, Stata {att:.10f}"
        assert got_se == pytest.approx(
            se, rel=RTOL
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
