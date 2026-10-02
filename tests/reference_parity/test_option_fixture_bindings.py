"""The option-level Stata fixtures are bound to the tests that cite them.

``tests/stata_parity/option_parity/results/`` holds Stata output for
option switches within an estimator. Three of those files (82, 83, 84)
were read by no test: their consumers carry the Stata numbers as literals
in the test source, so the committed fixture and the asserted numbers
could drift apart unnoticed, and nothing hashes these files
(``scripts/build_evidence_inventory.py`` reports both facts). This module
closes the gap from both sides:

* every number in fixtures 82-84 must appear as a literal in the test
  that claims it, to the precision the literal was written at;
* the seven ``did_imputation`` standard errors in fixture 84, which no
  test compared, are compared here against the fixture itself.
"""

from __future__ import annotations

import ast
import json
import pathlib

import pandas as pd
import pytest

import statspai as sp

_TESTS = pathlib.Path(__file__).resolve().parents[1]
_RESULTS = _TESTS / "stata_parity" / "option_parity" / "results"
_MPDTA = _TESTS / "orig_parity" / "data" / "02_mpdta_original.csv"

#: fixture -> the test whose literals are that fixture's numbers.
BINDINGS = {
    "82_csdid_conventions_Stata.json": "test_csdid_conventions_stata_parity.py",
    "83_sunab_control_cohort_Stata.json": "test_sunab_control_cohort_parity.py",
    "84_bjs_fe_covariates_Stata.json": "test_bjs_fe_covariates_parity.py",
}

#: Literals are written to 10+ decimals; half a unit in the 10th is 5e-11.
LITERAL_ATOL = 1e-9


def _leaves(node, path=""):
    if isinstance(node, dict):
        for key, value in node.items():
            if key != "_meta":
                yield from _leaves(value, f"{path}/{key}")
    elif isinstance(node, list):
        for i, value in enumerate(node):
            yield from _leaves(value, f"{path}/{i}")
    elif isinstance(node, (int, float)) and not isinstance(node, bool):
        yield path, float(node)


def _float_literals(source: str):
    out = []
    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Constant) and isinstance(node.value, float):
            out.append(node.value)
            out.append(-node.value)  # a unary minus is a separate node
    return out


@pytest.mark.parametrize("fixture", sorted(BINDINGS))
def test_every_fixture_number_is_a_literal_in_its_consumer(fixture):
    data = json.loads((_RESULTS / fixture).read_text(encoding="utf-8"))
    source = (_TESTS / "reference_parity" / BINDINGS[fixture]).read_text(
        encoding="utf-8"
    )
    literals = _float_literals(source)
    leaves = list(_leaves(data))
    assert len(leaves) >= 14, "fixture unexpectedly small"
    unbound = [
        (path, value)
        for path, value in leaves
        if not any(abs(value - lit) <= LITERAL_ATOL for lit in literals)
        # Fixture 84's standard errors are compared below, from the file.
        and not (fixture.startswith("84_") and path.endswith("/se"))
    ]
    assert (
        not unbound
    ), f"{BINDINGS[fixture]} does not pin these values of {fixture}: {unbound}"


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


def test_bjs_cases_cover_the_whole_fixture():
    data = json.loads(
        (_RESULTS / "84_bjs_fe_covariates_Stata.json").read_text(encoding="utf-8")
    )
    assert set(_BJS_CASES) == set(data) - {"_meta"}


@pytest.mark.parametrize("case", sorted(_BJS_CASES))
def test_bjs_option_standard_errors_match_stata(case):
    """``did_imputation`` SEs under each Y(0)-model option.

    rtol 1e-6 is the default same-byte budget; the worst case measured
    when this test was written is 1.8e-7 (``default``), the iterative
    ``lsqr`` fit against Stata's ``reghdfe`` absorption.
    """
    data = json.loads(
        (_RESULTS / "84_bjs_fe_covariates_Stata.json").read_text(encoding="utf-8")
    )
    subset, options = _BJS_CASES[case]
    frame = pd.read_csv(_MPDTA)
    if subset:
        # Units with >= 2 untreated periods; see the consumer test.
        untreated = (frame["first_treat"] == 0) | (frame["year"] < frame["first_treat"])
        keep = untreated.groupby(frame["countyreal"]).transform("sum") >= 2
        frame = frame[keep].copy()
    res = sp.did_imputation(frame, **_KEYS, **options)
    assert res.se == pytest.approx(data[case]["se"], rel=1e-6)
