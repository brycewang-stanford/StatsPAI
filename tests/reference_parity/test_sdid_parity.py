"""Reference parity: ``sp.sdid`` vs the authors' ``synthdid`` R package.

Synthetic Difference-in-Differences was introduced by Arkhangelsky, Athey,
Hirshberg, Imbens & Wager (2021), "Synthetic Difference-in-Differences",
*American Economic Review* 111(12):4088-4118 (doi:10.1257/aer.20190159),
together with the reference R package ``synthdid``.  We pin ``sp.sdid`` to
that package's estimate on StatsPAI's own California Prop 99 panel.

Result: on *identical* data the two implementations agree to ~1e-6 — the
SDID point estimate is **-17.8985** packs-per-capita (38 control states,
19 pre-treatment years).

Note on the famous "-15.6": that headline number is the ``synthdid``
package's *own bundled* ``california_prop99`` dataset, which uses a slightly
different donor panel than ``sp.california_prop99()``.  Run on identical
input the two packages coincide exactly, so the -17.9 vs -15.6 difference is
a *data* difference, not an estimator difference.

Tolerances
----------
  * point estimate: 1e-6 relative.  The SDID weight optimisation is
    deterministic; sp reproduces the synthdid package to ~10 significant
    figures.
  * standard error: NOT pinned.  synthdid's placebo variance estimator is
    randomisation-based (it varies run to run, ~2.4-2.5 here), so the test
    only sanity-checks that sp's SE lands in that documented band.

References
----------
- Arkhangelsky, D., Athey, S., Hirshberg, D. A., Imbens, G. W. & Wager, S.
  (2021). "Synthetic Difference-in-Differences." *American Economic Review*,
  111(12), 4088-4118. doi:10.1257/aer.20190159
  (Citation and page range verified via Crossref, 2026-06-08.)
"""

from __future__ import annotations

import json
import pathlib
import warnings

import pandas as pd
import pytest

import statspai as sp

_FIXTURE_DIR = pathlib.Path(__file__).parent / "_fixtures"
_USED_COLS = ["state", "year", "packspercapita", "treated"]


@pytest.fixture(scope="module")
def prop99():
    return sp.california_prop99()


@pytest.fixture(scope="module")
def r_reference():
    with open(_FIXTURE_DIR / "sdid_R.json", encoding="utf-8") as f:
        return json.load(f)


@pytest.fixture(scope="module")
def sdid_result(prop99):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.sdid(
            prop99,
            outcome="packspercapita",
            unit="state",
            time="year",
            treated_unit="California",
            treatment_time=1989,
        )


def test_prop99_fixture_matches_shipped_dataset(prop99):
    """The CSV the synthdid reference was generated from must still equal the
    shipped dataset; otherwise the frozen R number no longer describes the
    same panel."""
    frozen = pd.read_csv(_FIXTURE_DIR / "sdid_prop99_data.csv")[_USED_COLS]
    live = prop99[_USED_COLS].reset_index(drop=True)
    pd.testing.assert_frame_equal(
        frozen.reset_index(drop=True), live, check_dtype=False
    )


def test_sdid_estimate_matches_synthdid_package(sdid_result, r_reference):
    ref = r_reference["sdid"]["estimate"]
    est = float(sdid_result.estimate)
    assert est == pytest.approx(ref, rel=1e-6), (
        f"sp.sdid estimate {est:.10f} vs synthdid R package {ref:.10f} on "
        f"identical California Prop 99 data — these must coincide."
    )


def test_sdid_se_in_placebo_band(sdid_result):
    """synthdid's placebo SE is randomisation-based (~2.4-2.5); sp should land
    in the same neighbourhood, but this is a sanity band, not a parity pin."""
    se = float(sdid_result.se)
    assert 1.5 < se < 3.5, f"sp.sdid SE {se:.4f} outside placebo band [1.5, 3.5]"


def test_sdid_setup_dimensions(prop99, r_reference):
    """38 control states, 19 pre-treatment years (California treated 1989)."""
    n_states = prop99["state"].nunique()
    assert n_states - 1 == r_reference["sdid"]["N0"]  # all but California
    pre_years = prop99.loc[prop99["year"] < 1989, "year"].nunique()
    assert pre_years == r_reference["sdid"]["T0"]
