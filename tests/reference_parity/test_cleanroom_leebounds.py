"""Clean-room ``sp.lee_bounds(trimming='leebounds')`` against Stata ``leebounds``.

Black-box reference (``docs/dev/cleanroom/leebounds.md``): Stata 18 with
``set type double``, ``leebounds`` (Tauchmann, SSC v1.5) on
``_fixtures/leebounds_cleanroom.csv`` from
``_fixtures/_generate_leebounds_cleanroom.do``, plus the text Stata stores
for ``local m = v`` on ``_fixtures/leebounds_cleanroom_macro.csv``.

* The macro text is matched exactly (``_stata_local_macro``).
* Samples 1-2: outcomes of order 1e-7, held by Stata in scientific notation
  with 12 significant digits. ``leebounds`` also posts ``e(b)`` / ``e(V)``
  through that text, so those numbers carry 12 significant digits: bounds
  and variances are compared at a relative 1e-10.
* Samples 3-4: tightened bounds where no cell threshold coincides with an
  outcome; the between-cell divisor is then the overall number of selected
  units of the trimmed arm (asserted as that count).
* Sample 5: integer outcomes, one cell whose trimmed arm is the less
  retained (``cellsel == 'hetero'``); the divisor is the count of the last
  cell, whose threshold is tied.
* Sample 6: ``(1 - p) n`` a whole number (``_pctile`` averages).
"""

from __future__ import annotations

import json
import pathlib
import warnings

import pandas as pd
import pytest

import statspai as sp
from statspai.bounds.lee_manski import _stata_local_macro

_FIX = pathlib.Path(__file__).parent / "_fixtures"
REF = json.loads((_FIX / "leebounds_cleanroom_stata.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def data():
    return pd.read_csv(_FIX / "leebounds_cleanroom.csv", float_precision="round_trip")


def test_macro_text_is_stata_text():
    for value, text in REF["macro"]:
        assert _stata_local_macro(value) == float(text), (value, text)


@pytest.mark.parametrize("key", sorted(REF["samples"]))
def test_bounds_and_variances_match_leebounds(data, key):
    ref = REF["samples"][key]
    k, cov = key.split("_")
    d = data[data["sample"] == int(k)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        mi = sp.lee_bounds(
            d,
            "y",
            "d",
            "s",
            covariates=None if cov == "none" else ["x"],
            trimming="leebounds",
            se_method="analytic",
        ).model_info
    assert mi["lower_bound"] == pytest.approx(ref["lower"], rel=1e-10)
    assert mi["upper_bound"] == pytest.approx(ref["upper"], rel=1e-10)
    assert mi["se_lower"] ** 2 == pytest.approx(ref["var_lower"], rel=1e-10)
    assert mi["se_upper"] ** 2 == pytest.approx(ref["var_upper"], rel=1e-10)
    assert mi["trimming_fraction"] == pytest.approx(ref["trim"], rel=1e-12)
    if cov == "x":
        assert mi["n_cells"] == ref["cells"]
        assert mi["cell_selection_direction"] == ref["cellsel"]
        trimmed = d["d"] if ref["trimmed"] == "treatment" else 1 - d["d"]
        sel_t = (d["s"] == 1) & (trimmed == 1)
        if k in ("3", "4"):
            assert mi["between_cell_divisor"] == sel_t.sum()
        if k == "5":
            assert mi["between_cell_divisor"] == (sel_t & (d["x"] == 2)).sum()
        if "between_cell_divisor" in mi:  # deprecated alias, same number
            assert mi["leebounds_ntall"] == mi["between_cell_divisor"]
