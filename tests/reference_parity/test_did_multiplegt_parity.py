"""Reference parity for sp.did_multiplegt (dCDH 2020 DID_M).

Scope
-----
Compare ``sp.did_multiplegt`` output against a reference implementation
(R package ``DIDmultiplegt`` — dCDH 2020 version, NOT the dynamic ``_dyn``
package). Fixtures are expected to be saved JSON files containing the
exact numerical outputs from the reference package on fixed inputs, so
parity tests can run offline without invoking R.

Why this file exists as a skeleton
----------------------------------
As of 2026-04-23 we do not yet have the R reference outputs committed as
fixtures. This module defines the fixture schema, the parity assertions,
and the exact input shape the R script must produce — so "fill in the
fixture" is a well-scoped next action rather than an open-ended ask.

The real ``_dyn`` estimator (dCDH 2024 ReStat) has its own separate
parity file once ``sp.did_multiplegt_dyn`` lands; see
``docs/rfc/multiplegt_dyn.md``.

Fixture format (expected at tests/reference_parity/fixtures/did_multiplegt/*.json):

{
  "metadata": {
    "source": "R DIDmultiplegt <version>",
    "script": "path/to/R/script.R",
    "seed": 42,
    "generated_at": "<ISO date>",
    "reference_version": "DIDmultiplegt 4.x.y",
    "dcdh_paper_version": "2020 AER final"
  },
  "inputs": {
    "n_units": 200,
    "n_periods": 8,
    "seed": 42
  },
  "expected": {
    "did_m": <float>,
    "did_m_se": <float>,
    "n_switching_cells": <int>,
    "n_switchers": <int>,
    "placebo": [ {"lag": -1, "estimate": ..., "se": ...}, ... ],
    "dynamic":  [ {"horizon": 0, "estimate": ..., "se": ...}, ... ]
  }
}

Tolerances: atol=1e-4 on point estimates (dCDH 2020 closed-form), atol=1e-3 on
bootstrap SEs (bootstrap differs across seeds + RNG choice).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from statspai.did import did_multiplegt

FIXTURE_DIR = Path(__file__).parent / "fixtures" / "did_multiplegt"


def _available_fixtures() -> list[Path]:
    if not FIXTURE_DIR.exists():
        return []
    return sorted(FIXTURE_DIR.glob("*.json"))


def _build_dgp(n_units: int, n_periods: int, seed: int) -> pd.DataFrame:
    """Deterministic on-off switching panel matching the R fixture script.

    NOTE: the R side MUST use the same DGP to produce parity fixtures.
    When you write the R script, port this function verbatim using the
    same generator seeding and operation order.

    Constraints: ``n_periods >= 6`` so that a valid
    ``switch_off`` exists for any ``switch_on``.
    """
    if n_periods < 6:
        raise ValueError(
            f"n_periods={n_periods} too short for the on/off DGP; "
            "need at least 6 periods so switch_off > switch_on is feasible."
        )
    rng = np.random.default_rng(seed)
    rows = []
    for u in range(n_units):
        switch_on = int(rng.integers(3, 7))
        switch_off = int(rng.integers(switch_on + 1, n_periods + 2))
        unit_fx = rng.normal(scale=0.3)
        for t in range(1, n_periods + 1):
            d = int(switch_on <= t < switch_off)
            y = unit_fx + 0.15 * t + 0.6 * d + rng.normal()
            rows.append({"g": u, "t": t, "d": d, "y": y})
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Structural tests that run without a fixture (always executed)
# ---------------------------------------------------------------------------


def test_dgp_shape_matches_r_script_contract():
    """If the R fixture script changes DGP shape (columns / types), this test
    fails and signals that fixtures are stale."""
    df = _build_dgp(n_units=20, n_periods=8, seed=1)
    assert set(df.columns) == {"g", "t", "d", "y"}
    assert df["d"].dtype == np.int64 or df["d"].dtype == np.int32
    assert df["d"].isin([0, 1]).all()
    assert df["t"].min() == 1


def test_did_multiplegt_runs_on_reference_dgp():
    """Baseline regression: the estimator produces a finite result on the
    reference DGP. Failure here means something upstream of the R parity
    check broke."""
    df = _build_dgp(n_units=60, n_periods=8, seed=1)
    r = did_multiplegt(
        df,
        y="y",
        group="g",
        time="t",
        treatment="d",
        placebo=1,
        dynamic=2,
        n_boot=50,
        seed=1,
    )
    assert np.isfinite(r.estimate)
    assert np.isfinite(r.se) and r.se > 0


# ---------------------------------------------------------------------------
# Fixture-driven parity (skipped until fixtures are committed)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "fixture_path",
    _available_fixtures()
    or [
        pytest.param(
            None,
            marks=pytest.mark.skip(
                reason="No R DIDmultiplegt fixtures committed yet. Generate via "
                "tests/reference_parity/fixtures/did_multiplegt/generate.R "
                "(to be written — see docs/rfc/did_roadmap_gap_audit.md "
                "§5 step 3) and commit JSON outputs matching the schema "
                "documented in this file's module docstring."
            ),
        )
    ],
    ids=lambda p: p.name if isinstance(p, Path) else "no-fixtures",
)
def test_parity_against_r_didmultiplegt(fixture_path):
    """Parity test — runs only when a JSON fixture from R is present."""
    fixture = json.loads(Path(fixture_path).read_text(encoding="utf-8"))
    inp = fixture["inputs"]
    exp = fixture["expected"]

    df = _build_dgp(
        n_units=inp["n_units"],
        n_periods=inp["n_periods"],
        seed=inp["seed"],
    )

    # Match R's placebo/dynamic counts from the fixture.
    placebo = len(exp.get("placebo", []))
    dynamic = len(exp.get("dynamic", [])) - 1  # horizons are 0..L so count-1
    if dynamic < 0:
        dynamic = 0

    r = did_multiplegt(
        df,
        y="y",
        group="g",
        time="t",
        treatment="d",
        placebo=placebo,
        dynamic=dynamic,
        n_boot=fixture["metadata"].get("n_boot", 100),
        seed=inp["seed"],
    )

    # Point estimate: dCDH 2020 DID_M is a closed-form weighted mean ⇒
    # should match exactly up to numeric noise.
    assert r.estimate == pytest.approx(
        exp["did_m"], abs=1e-4
    ), "DID_M point estimate differs from R reference"

    # SE is bootstrap ⇒ looser tolerance and still sensitive to RNG
    # choice. When R and Python use different RNGs this will need
    # either (a) R to dump its bootstrap draws so we reuse them, or
    # (b) a bootstrap-free SE path (not yet implemented). Mark as soft
    # assert for now.
    if "did_m_se" in exp and exp["did_m_se"] is not None:
        # Relative tolerance rather than absolute because SE scale varies.
        se_ratio = r.se / exp["did_m_se"] if exp["did_m_se"] != 0 else np.nan
        assert 0.5 < se_ratio < 2.0, (
            f"DID_M SE ratio {se_ratio:.3f} outside [0.5, 2.0] — "
            f"bootstrap may be using different RNG than R reference"
        )

    if exp.get("n_switchers") is not None:
        assert r.model_info["n_switchers"] == exp["n_switchers"]
    if exp.get("n_switching_cells") is not None:
        assert r.model_info["n_switching_cells"] == exp["n_switching_cells"]

    # Placebo and dynamic per-horizon estimates.
    plac_py = r.model_info.get("placebo", [])
    for exp_row in exp.get("placebo", []):
        match = next(
            (p for p in plac_py if p.get("lag") == exp_row["lag"]),
            None,
        )
        assert match is not None, f"Placebo lag {exp_row['lag']} missing"
        assert match["estimate"] == pytest.approx(exp_row["estimate"], abs=1e-4)

    dyn_py = r.model_info.get("dynamic", [])
    for exp_row in exp.get("dynamic", []):
        match = next(
            (d for d in dyn_py if d.get("horizon") == exp_row["horizon"]),
            None,
        )
        assert match is not None, f"Dynamic horizon {exp_row['horizon']} missing"
        assert match["estimate"] == pytest.approx(exp_row["estimate"], abs=1e-4)


# ---------------------------------------------------------------------------
# Instructions for generating the R fixture (kept here so it's co-located
# with the parity test consuming it)
# ---------------------------------------------------------------------------

R_FIXTURE_SCRIPT_TEMPLATE = r"""
# tests/reference_parity/fixtures/did_multiplegt/generate.R
#
# Prerequisites:
#   install.packages("DIDmultiplegt")  # dCDH 2020 version, NOT DIDmultiplegtDYN
#   install.packages("jsonlite")
#
# Run: Rscript generate.R
#
library(DIDmultiplegt)
library(jsonlite)

make_dgp <- function(n_units, n_periods, seed) {
  set.seed(seed)
  # NOTE: this DGP must match _build_dgp() in test_did_multiplegt_parity.py.
  # If either side diverges, fixture generation will be silently broken.
  rows <- list()
  for (u in seq_len(n_units)) {
    switch_on  <- sample(3:6, 1)
    switch_off <- sample((switch_on + 1):(n_periods + 1), 1)
    unit_fx   <- rnorm(1, sd = 0.3)
    for (t in seq_len(n_periods)) {
      d <- as.integer(switch_on <= t & t < switch_off)
      y <- unit_fx + 0.15 * t + 0.6 * d + rnorm(1)
      rows[[length(rows) + 1L]] <- list(g = u, t = t, d = d, y = y)
    }
  }
  do.call(rbind.data.frame, rows)
}

# CRITICAL — R's RNG ≠ numpy's PCG64. The parity test above tolerates SE
# drift but point estimates should match because DID_M is closed-form.
# Consider regenerating the DGP in Python once and saving to CSV, then
# reading the CSV from both sides to eliminate RNG divergence entirely.

fixture <- function(n_units = 60, n_periods = 8, seed = 1) {
  df <- make_dgp(n_units, n_periods, seed)
  out <- did_multiplegt(
    df, Y = "y", G = "g", T = "t", D = "d",
    placebo = 1, dynamic = 2, brep = 100, cluster = "g"
  )
  list(
    metadata = list(
      source = paste0("R DIDmultiplegt ", packageVersion("DIDmultiplegt")),
      script = "tests/reference_parity/fixtures/did_multiplegt/generate.R",
      seed = seed,
      n_boot = 100,
      dcdh_paper_version = "2020 AER final"
    ),
    inputs = list(n_units = n_units, n_periods = n_periods, seed = seed),
    expected = list(
      did_m = out$effect,
      did_m_se = out$se_effect,
      n_switchers = out$N_switchers_effect,
      placebo = lapply(seq_along(out$placebo), function(i) {
        list(lag = -i, estimate = out$placebo[i], se = out$se_placebo[i])
      }),
      dynamic = lapply(seq_along(out$dynamic), function(i) {
        list(horizon = i - 1, estimate = out$dynamic[i], se = out$se_dynamic[i])
      })
    )
  )
}

fx <- fixture()
writeLines(toJSON(fx, auto_unbox = TRUE, pretty = TRUE), "did_multiplegt_basic.json")
"""


def test_r_fixture_script_template_is_present():
    """Meta-test: the R script template must remain in-source so the
    fixture is always reproducible. If this fails, someone accidentally
    stripped the template from the module docstring block."""
    assert "DIDmultiplegt" in R_FIXTURE_SCRIPT_TEMPLATE
    assert "did_multiplegt" in R_FIXTURE_SCRIPT_TEMPLATE.lower()
    assert "generate.R" in R_FIXTURE_SCRIPT_TEMPLATE
