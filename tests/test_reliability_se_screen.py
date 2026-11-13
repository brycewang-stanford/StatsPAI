"""The SE calibration screen: stored rows are in band or explained.

``tests/reliability/se_screen/<family>.py`` simulates, for each estimator
call, the coverage of the reported 95% interval, the ratio of the mean
reported SE to the Monte Carlo SD of the estimate, and the bias in SD
units, and writes ``<family>_results.json``. This file checks, for every
committed results file in that directory:

* each row has the declared fields and was run on at least 200 draws;
* the stored ``flagged`` value is the one the stored numbers imply;
* an unflagged row is inside the band for its number of draws;
* a flagged row carries a mechanism note, so that no row is left
  unexplained;
* one cheap row per family, recomputed on its first 20 draws, reproduces
  the stored sums (same seeds, so the file came from the script).
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

DIR = Path(__file__).resolve().parent / "reliability" / "se_screen"
RESULT_FILES = sorted(DIR.glob("*_results.json"))
REQUIRED = {
    "name",
    "estimator",
    "call",
    "n",
    "B",
    "truth",
    "coverage",
    "se_over_sd",
    "bias_over_sd",
    "flagged",
    "mechanism",
    "statspai_version",
}

# One fast, deterministic row per family for the reproduction check.
REPRODUCE = {
    "regression": "logit/default",
    "panel": "panel/fe",
    "iv": "iv/2sls",
    "survival": "cox/default",
    "timeseries": "regress/hac-4",
    "did": "etwfe/default",
    "rd": "rdrobust/covs",
    "multilevel": "mixed/random-intercept",
    "spatial": "slx/beta",
}


def _harness():
    spec = importlib.util.spec_from_file_location(
        "_se_screen_harness", DIR / "_harness.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module  # dataclasses look their module up there
    spec.loader.exec_module(module)
    return module


HARNESS = _harness()


def _rows():
    out = []
    for path in RESULT_FILES:
        payload = json.loads(path.read_text(encoding="utf-8"))
        for row in payload["rows"]:
            out.append(
                pytest.param(
                    payload["family"], row, id=f"{payload['family']}:{row['name']}"
                )
            )
    return out


def test_results_files_exist_and_name_their_script():
    assert RESULT_FILES, "no committed se_screen results"
    for path in RESULT_FILES:
        payload = json.loads(path.read_text(encoding="utf-8"))
        assert (DIR / payload["script"]).exists()
        names = [r["name"] for r in payload["rows"]]
        assert len(names) == len(set(names))


@pytest.mark.parametrize("family,row", _rows())
def test_row_is_in_band_or_explained(family, row):
    assert REQUIRED <= set(row), sorted(REQUIRED - set(row))
    assert row["B"] >= 200
    band = HARNESS.band(row["B"])
    out_of_band = HARNESS.is_flagged(row)
    too_many_failures = row.get("n_failed", 0) > 0.05 * row["B"]
    assert row["flagged"] == (out_of_band or too_many_failures)
    if row["flagged"]:
        assert row["mechanism"].strip(), f"{row['name']} is flagged without a mechanism"
        return
    lo, hi = band["coverage"]
    assert lo <= row["coverage"] <= hi
    if row["se_over_sd"] is not None:
        lo, hi = band["se_over_sd"]
        assert lo <= row["se_over_sd"] <= hi
        assert abs(row["bias_over_sd"]) <= HARNESS.BIAS_LIMIT


def test_band_widens_only_below_400_draws():
    assert HARNESS.band(400)["coverage"] == [0.918, 0.985]
    assert HARNESS.band(1000) == HARNESS.band(400)
    lo, hi = HARNESS.band(200)["coverage"]
    # Wilson 99% interval around 0.95 on 200 draws is (0.8939, 0.9772); the
    # upper limit never drops below the 0.985 used at 400 draws.
    assert lo == pytest.approx(0.8939, abs=1e-4) and hi == 0.985
    assert HARNESS.band(200)["se_over_sd"] == pytest.approx([0.8303, 1.1697], abs=1e-4)


@pytest.mark.parametrize(
    "path", RESULT_FILES, ids=[p.stem.replace("_results", "") for p in RESULT_FILES]
)
def test_one_row_reproduces_its_stored_prefix(path):
    """Same seeds, same sums: the stored file came from this script."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    name = REPRODUCE.get(payload["family"])
    if name is None:
        pytest.skip(f"no reproduction row declared for {payload['family']}")
    (stored,) = [r for r in payload["rows"] if r["name"] == name]
    fresh = HARNESS.prefix(str(DIR / payload["script"]), name)
    assert fresh["ok"] == stored["prefix"]["ok"]
    # Sums of 20 estimates and SEs: optimiser and BLAS differences across
    # platforms stay far below 1e-5 relative on these closed-form or
    # tightly converged fits.
    assert fresh["sum_estimate"] == pytest.approx(
        stored["prefix"]["sum_estimate"], rel=1e-5
    )
    assert fresh["sum_se"] == pytest.approx(stored["prefix"]["sum_se"], rel=1e-5)
