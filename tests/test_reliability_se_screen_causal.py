"""The SE calibration screen of the causal families: in band or explained.

``tests/reliability/se_screen/causal/<family>.py`` simulates, for each
estimator call, the coverage of the reported interval, the ratio of the
mean reported standard error to the Monte Carlo standard deviation of the
estimate, and the bias in standard-deviation units, and writes
``<family>_results.json``. This file checks, for every committed results
file in that directory:

* each row carries the declared fields and was run on at least 200 draws;
* the stored ``flagged`` value is the one the stored numbers imply;
* an unflagged row is inside the bands for its number of draws;
* a flagged row carries a mechanism note, so no row is left unexplained;
* one cheap, closed-form row per family, recomputed on its first 20
  draws, reproduces the stored sum (same seeds: the file came from the
  script);
* the statements the package now makes because of this screen are the
  ones the stored numbers support.

The regression, panel, IV and DiD families are screened by a sibling
study in ``tests/reliability/se_screen/`` with its own test file.
"""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path

import pytest

DIR = Path(__file__).resolve().parent / "reliability" / "se_screen" / "causal"
RESULT_FILES = sorted(DIR.glob("*_results.json"))
REQUIRED = {
    "id",
    "estimator",
    "call",
    "kind",
    "n",
    "B",
    "nominal",
    "truth",
    "coverage",
    "se_over_sd",
    "bias_over_sd",
    "flagged",
    "flag_reasons",
    "mechanism",
    "statspai_version",
}

# One fast row per family whose fit is closed-form or tightly converged.
REPRODUCE = {
    "treatment_effects": "difference_in_means",
    "matching": "match.att.stratify",
    "weighting": "ebalance.att",
    "ml_causal": "dml.plr.cluster",
    "conformal": "conformal_regression.split",
    "mediation": "mediate.acme.delta",
    "bounds": "lee_bounds.analytic",
    "dose_qte": "qte.firpo.no_covariates",
}


def _harness():
    sys.path.insert(0, str(DIR))
    spec = importlib.util.spec_from_file_location(
        "se_screen_causal_harness", DIR / "_causal_harness.py"
    )
    module = importlib.util.module_from_spec(spec)
    sys.modules["se_screen_causal_harness"] = module
    spec.loader.exec_module(module)
    return module


HARNESS = _harness()


def _family(path: Path) -> str:
    return path.stem[: -len("_results")]


def _payload(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _rows():
    out = []
    for path in RESULT_FILES:
        for row in _payload(path)["rows"]:
            out.append(pytest.param(row, id=f"{_family(path)}:{row['id']}"))
    return out


def _row(family: str, row_id: str) -> dict:
    (row,) = [
        r for r in _payload(DIR / f"{family}_results.json")["rows"] if r["id"] == row_id
    ]
    return row


def test_every_family_script_has_committed_results():
    scripts = {p.stem for p in DIR.glob("*.py") if not p.name.startswith("_")}
    assert scripts == {_family(p) for p in RESULT_FILES}
    for path in RESULT_FILES:
        ids = [r["id"] for r in _payload(path)["rows"]]
        assert len(ids) == len(set(ids)), path.name


def test_the_screen_is_at_least_sixty_estimator_rows():
    assert len(_rows()) >= 60


@pytest.mark.parametrize("row", _rows())
def test_row_is_in_band_or_explained(row):
    assert REQUIRED <= set(row), sorted(REQUIRED - set(row))
    assert row["B"] >= 200
    reasons = HARNESS.is_flagged(row)
    assert row["flag_reasons"] == reasons
    assert row["flagged"] == bool(reasons)
    if row["flagged"]:
        assert row[
            "mechanism"
        ].strip(), f"{row['id']} is flagged ({reasons}) without a mechanism"
        return
    band = HARNESS.bands(row["nominal"], row["B"])
    lo, hi = band["coverage"]
    assert lo <= row["coverage"]
    if not row.get("conservative_by_construction", False):
        assert row["coverage"] <= hi
    if row["kind"] == "ci":
        if row["se_over_sd"] is not None:
            lo, hi = band["se_over_sd"]
            assert lo <= row["se_over_sd"] <= hi
        assert abs(row["bias_over_sd"]) <= band["bias_over_sd"][1]


def test_bands_are_the_declared_ones():
    b = HARNESS.bands(0.95, 400)
    assert b["coverage"] == (0.918, 0.985)
    assert b["se_over_sd"] == (0.88, 1.12)
    assert HARNESS.bands(0.95, 1000) == b
    # Below 400 draws: every coverage whose Wilson 99% interval holds 0.95,
    # that is 0.95 +/- 2.576 sqrt(0.95 * 0.05 / B).
    lo, hi = HARNESS.bands(0.95, 200)["coverage"]
    assert lo == pytest.approx(0.9103, abs=1e-4)
    assert hi == pytest.approx(0.9897, abs=1e-4)
    lo, hi = HARNESS.bands(0.95, 200)["se_over_sd"]
    assert (lo, hi) == pytest.approx((0.8303, 1.1697), abs=1e-4)
    # A 90% prediction interval on 400 draws.
    lo, hi = HARNESS.bands(0.90, 400)["coverage"]
    assert (lo, hi) == pytest.approx((0.8614, 0.9386), abs=1e-4)


@pytest.mark.parametrize("family", sorted(REPRODUCE))
def test_one_row_reproduces_its_stored_prefix(family):
    """Same seeds, same sums: the stored file came from this script."""
    path = DIR / f"{family}_results.json"
    if not path.exists():
        pytest.skip(f"{family} has no committed results")
    stored = _row(family, REPRODUCE[family])["prefix"]
    fresh = HARNESS.run_prefix(str(DIR / f"{family}.py"), REPRODUCE[family])
    assert fresh["ok"] == stored["ok"] == 20
    # Sum of 20 interval endpoints (or coverage shares). These fits are
    # least squares, a converged Newton logit or sample quantiles, so
    # platform BLAS differences stay orders of magnitude below 1e-6.
    assert fresh["sum"] == pytest.approx(stored["sum"], rel=1e-6)
