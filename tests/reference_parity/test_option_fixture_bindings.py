"""Guards on the option-level Stata fixtures as a set.

``tests/stata_parity/option_parity/results/`` holds Stata output for
option switches within an estimator. Two defects were found in it on
2026-10-02 and both are pinned here so they cannot return.

**Single precision.** These do-files do not source Track A's
``_common.do``, which forces ``set type double``. Without it
``import delimited`` stored every decimal column as float, rounding the
data in the 8th digit, so Stata was not estimating on the bytes StatsPAI
read. The recorded gaps (2e-5 relative for ``csdid``, 2e-6 for
``did_imputation``) had been explained as optimizer and absorption
tolerance; regenerated in double precision they are 5e-13 and 6e-8. Every
do-file now sets ``set type double`` and imports ``asdouble``, and every
fixture records ``"precision": "double"``.

**Unread fixtures.** Three fixtures were opened by no test: their
consumers carried the Stata numbers as literals. Each consumer now reads
its file, and each fixture must be named by a test.

The seven ``did_imputation`` standard errors in fixture 84, which no test
compared, are compared here.
"""

from __future__ import annotations

import json
import pathlib
import re

import pandas as pd
import pytest

import statspai as sp

_TESTS = pathlib.Path(__file__).resolve().parents[1]
_OPTION = _TESTS / "stata_parity" / "option_parity"
_RESULTS = _OPTION / "results"
_MPDTA = _TESTS / "orig_parity" / "data" / "02_mpdta_original.csv"

FIXTURES = sorted(p.name for p in _RESULTS.glob("*_Stata.json"))


def test_there_are_fixtures_to_guard():
    assert len(FIXTURES) >= 6


@pytest.mark.parametrize("fixture", FIXTURES)
def test_fixture_was_generated_in_double_precision(fixture):
    data = json.loads((_RESULTS / fixture).read_text(encoding="utf-8"))
    assert data["_meta"].get("precision") == "double", (
        f"{fixture} does not record double precision: regenerate it from a "
        "do-file that sets `set type double` and imports `asdouble`"
    )


@pytest.mark.parametrize("fixture", FIXTURES)
def test_generating_do_file_forces_double_precision(fixture):
    do_file = _OPTION / fixture.replace("_Stata.json", ".do")
    source = do_file.read_text(encoding="utf-8")
    code = "\n".join(
        line for line in source.splitlines() if not line.lstrip().startswith("*")
    )
    assert re.search(r"^\s*set type double\s*$", code, flags=re.M), do_file.name
    imports = re.findall(r"^\s*import delimited .*$", code, flags=re.M)
    assert imports, do_file.name
    assert all("asdouble" in line for line in imports), (do_file.name, imports)


@pytest.mark.parametrize("fixture", FIXTURES)
def test_fixture_is_opened_by_a_test(fixture):
    readers = [
        p.name
        for p in (_TESTS / "reference_parity").glob("test_*.py")
        if p.name != pathlib.Path(__file__).name
        and fixture in p.read_text(encoding="utf-8")
    ]
    assert readers, f"no test under tests/reference_parity reads {fixture}"


# ----------------------------------------------------------------------
# did_imputation standard errors under the Y(0)-model options
# ----------------------------------------------------------------------
_KEYS = dict(y="lemp", group="countyreal", time="year", first_treat="first_treat")

#: fixture key -> (use the identified subset?, did_imputation options)
_BJS_CASES = {
    "default": (False, {}),
    "fe_time_only": (False, {"fe": ["year"]}),
    "fe_none": (False, {"fe": []}),
    "timecontrols_lpop": (False, {"time_covariates": ["lpop"]}),
    "controls_lpop": (False, {"controls": ["lpop"]}),
    "default_identified_subset": (True, {}),
    "unitcontrols_year_subset": (True, {"unit_covariates": ["year"]}),
}


def _bjs_fixture() -> dict:
    return json.loads(
        (_RESULTS / "84_bjs_fe_covariates_Stata.json").read_text(encoding="utf-8")
    )


def test_bjs_cases_cover_the_whole_fixture():
    assert set(_BJS_CASES) == set(_bjs_fixture()) - {"_meta"}


@pytest.mark.parametrize("case", sorted(_BJS_CASES))
def test_bjs_option_standard_errors_match_stata(case):
    """rtol 1e-8; observed worst case 1.3e-9 (``unitcontrols(year)``).

    Against the single-precision fixture the same comparison sat at
    1.8e-7.
    """
    subset, options = _BJS_CASES[case]
    frame = pd.read_csv(_MPDTA)
    if subset:
        # Units with >= 2 untreated periods; see test_bjs_fe_covariates_parity.
        untreated = (frame["first_treat"] == 0) | (frame["year"] < frame["first_treat"])
        keep = untreated.groupby(frame["countyreal"]).transform("sum") >= 2
        frame = frame[keep].copy()
    res = sp.did_imputation(frame, **_KEYS, **options)
    assert res.se == pytest.approx(_bjs_fixture()[case]["se"], rel=1e-8)
