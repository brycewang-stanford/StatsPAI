"""Reference parity: sp.svymean / sp.svytotal vs R survey::svymean/svytotal.

R's ``survey`` package (Lumley) is the canonical reference for design-based
survey estimation. On a weights-only design (``ids = ~1``), StatsPAI reproduces
the Horvitz-Thompson / Hajek estimators and their Taylor-linearization standard
errors to machine precision.

Frozen reference: ``_fixtures/survey_R.json`` (R 4.5.2 / survey). Regenerate::

    Rscript tests/reference_parity/_generate_survey_R.R

References
----------
- Lumley, T. (2004). Analysis of Complex Survey Samples. *JSS* 9(8).
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd
import pytest

import statspai as sp

_FIXTURES = pathlib.Path(__file__).parent / "_fixtures"


@pytest.fixture(scope="module")
def design():
    df = pd.read_csv(_FIXTURES / "survey_data.csv")
    return sp.svydesign(df, weights="w")


@pytest.fixture(scope="module")
def r_reference():
    with open(_FIXTURES / "survey_R.json", encoding="utf-8") as fh:
        return json.load(fh)


def _scalar(x):
    return float(np.ravel(x)[0])


def test_svymean_matches_R_survey(design, r_reference):
    m = sp.svymean("y", design)
    ref = r_reference["svymean"]
    assert _scalar(m.estimate) == pytest.approx(_scalar(ref["estimate"]), abs=1e-10)
    assert _scalar(m.std_error) == pytest.approx(_scalar(ref["se"]), abs=1e-10)


def test_svytotal_matches_R_survey(design, r_reference):
    t = sp.svytotal("y", design)
    ref = r_reference["svytotal"]
    assert _scalar(t.estimate) == pytest.approx(_scalar(ref["estimate"]), rel=1e-12)
    assert _scalar(t.std_error) == pytest.approx(_scalar(ref["se"]), rel=1e-10)
