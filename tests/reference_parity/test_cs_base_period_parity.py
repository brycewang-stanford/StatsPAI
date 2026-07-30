"""Reference parity: CS event-study pre-periods under ``base_period='varying'``.

``sp.callaway_santanna``'s post-treatment dynamics have long matched R ``did``
and Stata ``csdid`` three-way to machine precision.  The **pre-treatment**
placebos did not, for two separate reasons:

1. StatsPAI defaults to ``base_period='universal'`` while R ``did`` and Stata
   ``csdid`` default to ``'varying'``.  Under 'universal' every pre-period is
   differenced against the single reference g−1, so the placebo coefficients
   are a different (also valid) estimand — not a bug, a different convention.
2. Under ``'varying'`` StatsPAI *omitted* the e = −1 coefficient. The (g, t)
   grid builder skipped ``t == g − 1 − anticipation`` unconditionally, on the
   reasoning that it is the reference period.  That is only true under the
   universal scheme; under 'varying' the base for t = g−1 is g−2, so
   ATT(g, g−1) is a genuine estimable placebo.  ⚠️ corrected — the cell is
   now reported.

With ``base_period='varying'`` the whole event study — every pre- and
post-treatment coefficient — now agrees with both references.

Reference values
----------------
Three-way reconciliation on the canonical ``did::mpdta`` panel (500 counties,
2003-2007, cohorts 2004/2006/2007).  R ``did`` 2.3.0 and Stata ``csdid``
agree with each other to machine precision on every cell below.  Source:
the WP-A cross-software reconciliation (DiD reconciliation study,
``parity/DISCREPANCY_TAXONOMY.md`` §1), which reports them to 6 decimals.

Data provenance
---------------
``tests/orig_parity/data/02_mpdta_original.csv``, SHA256
``1b789c34e12ff490b2f432217a1f70af334117523eb44d20eb842ed92a574661`` —
verified byte-identical to a rebuild of ``did::mpdta`` from the R package.
The checksum is asserted here so the fixture cannot drift underneath the
pinned reference numbers.

References
----------
- Callaway, B. and Sant'Anna, P.H.C. (2021). "Difference-in-Differences with
  Multiple Time Periods." *Journal of Econometrics*, 225(2), 200-230.
  [@callaway2021difference]
"""

from __future__ import annotations

import hashlib
import pathlib

import pandas as pd
import pytest

import statspai as sp

_MPDTA = (
    pathlib.Path(__file__).resolve().parents[1]
    / "orig_parity"
    / "data"
    / "02_mpdta_original.csv"
)

_MPDTA_SHA256 = "1b789c34e12ff490b2f432217a1f70af334117523eb44d20eb842ed92a574661"

# R did 2.3.0 == Stata csdid, base_period='varying'.  Rounded to 6 dp by the
# reconciliation source, hence the 1e-6 absolute tolerance below.
R_STATA_DYNAMIC = {
    -3: 0.030507,
    -2: -0.000563,
    -1: -0.024459,
    0: -0.019932,
    1: -0.050957,
    2: -0.137259,
    3: -0.100811,
}


@pytest.fixture(scope="module")
def mpdta() -> pd.DataFrame:
    if not _MPDTA.exists():  # pragma: no cover - fixture shipped with the repo
        pytest.skip(f"locked mpdta fixture missing: {_MPDTA}")
    digest = hashlib.sha256(_MPDTA.read_bytes()).hexdigest()
    assert digest == _MPDTA_SHA256, (
        "the mpdta fixture changed — the pinned R/Stata reference numbers in "
        f"this module were locked against {_MPDTA_SHA256}, got {digest}"
    )
    return pd.read_csv(_MPDTA)


def _dynamic(df: pd.DataFrame, base_period: str) -> pd.DataFrame:
    cs = sp.callaway_santanna(
        df,
        y="lemp",
        g="first_treat",
        t="year",
        i="countyreal",
        base_period=base_period,
    )
    return sp.aggte(cs, type="dynamic", bstrap=False).detail


def test_varying_event_study_matches_r_and_stata(mpdta):
    """Every pre- and post-treatment coefficient matches both references."""
    detail = _dynamic(mpdta, "varying")
    got = dict(zip(detail["relative_time"], detail["att"]))

    assert set(got) == set(R_STATA_DYNAMIC), (
        f"event-time grid drifted: got {sorted(got)}, "
        f"expected {sorted(R_STATA_DYNAMIC)}"
    )
    for e, expected in R_STATA_DYNAMIC.items():
        assert got[e] == pytest.approx(
            expected, abs=1e-6
        ), f"e={e}: StatsPAI {got[e]:.6f} vs R/Stata {expected:.6f}"


def test_varying_reports_the_minus_one_placebo(mpdta):
    """Regression guard for the omitted e = −1 cell.

    Its base period is g−2, so it is estimable and must be reported; the old
    grid builder dropped it as if it were the universal reference.
    """
    detail = _dynamic(mpdta, "varying")
    assert -1 in set(detail["relative_time"]), "e=-1 placebo is missing again"

    row = detail.loc[detail["relative_time"] == -1].iloc[0]
    assert row["att"] == pytest.approx(-0.024459, abs=1e-6)
    assert row["se"] > 0


def test_universal_still_omits_the_reference_period(mpdta):
    """The universal scheme must keep excluding its own reference cell.

    Under 'universal' ATT(g, g−1) is zero by construction, so reporting it
    would add a degenerate row.  This pins that the fix above did not leak
    into the default path.
    """
    detail = _dynamic(mpdta, "universal")
    assert -1 not in set(detail["relative_time"])


def test_post_treatment_is_base_period_invariant(mpdta):
    """base_period changes only the placebos, never the causal coefficients."""
    varying = _dynamic(mpdta, "varying")
    universal = _dynamic(mpdta, "universal")

    for frame in (varying, universal):
        post = frame.loc[frame["relative_time"] >= 0]
        for e, att in zip(post["relative_time"], post["att"]):
            assert att == pytest.approx(R_STATA_DYNAMIC[e], abs=1e-6)
