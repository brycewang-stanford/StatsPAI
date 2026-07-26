"""Reference parity: CS multiplier-bootstrap inference vs R ``did``.

Covers the batch-D inference options of ``sp.callaway_santanna``:

- ``bstrap=True`` per-(g, t) bootstrap SEs
- ``cband=True`` uniform (sup-t) critical value
- ``clustervars`` beyond the unit id

Fixture: ``_fixtures/cs_inference_R.json`` from
``_generate_cs_inference_R.R`` (R ``did`` 2.3.0, ``biters=9999``), on the
same panel as ``test_callaway_santanna_parity.py`` plus a deterministic
cluster variable ``clust = id % 15``.

Tolerances
----------
  • ATT(g, t) point estimates: 1e-6 absolute (deterministic, same DGP)
  • Bootstrap SEs: 8% relative (independent multiplier draws — R and
    Python use different RNG streams; both use Rademacher weights and
    IQR rescaling, so the difference is pure Monte Carlo noise at
    biters=9999)
  • Uniform critical values: 5% relative
  • Clustered SEs: 10% relative (only 15 clusters — the multiplier
    distribution is discrete/lumpy, so the IQR functional itself has
    higher Monte Carlo variance; R's own draws deviate from the
    procedure's theoretical limit by up to ~9% here)

Drift beyond these bands signals a bug — first suspects are the
multiplier weight distribution (R draws Rademacher via
``BMisc::multiplier_bootstrap`` despite docs citing Mammen) and the
cluster-mean collapse in ``did._core.multiplier_bootstrap``.

References
----------
- Callaway, B. and Sant'Anna, P.H.C. (2021). "Difference-in-
  differences with multiple time periods." *Journal of Econometrics*,
  225(2), 200-230. [@callaway2021difference]
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd
import pytest

import statspai as sp

_FIXTURE_DIR = pathlib.Path(__file__).parent / "_fixtures"

BITERS = 9999
SEED = 42


@pytest.fixture(scope="module")
def cs_data():
    df = pd.read_csv(_FIXTURE_DIR / "cs_data.csv")
    df["clust"] = df["id"] % 15
    return df


@pytest.fixture(scope="module")
def r_reference():
    with open(_FIXTURE_DIR / "cs_inference_R.json", encoding="utf-8") as f:
        return json.load(f)


def _r_cells(block):
    """R (g, t) cells with finite SEs (drops universal-base zero rows)."""
    out = pd.DataFrame(
        {
            "g": block["group"],
            "t": block["t"],
            "att": pd.to_numeric(block["att"], errors="coerce"),
            "se": pd.to_numeric(block["se"], errors="coerce"),
        }
    )
    return out[np.isfinite(out["se"])]


@pytest.fixture(scope="module")
def py_unit(cs_data):
    return sp.callaway_santanna(
        cs_data,
        y="y",
        g="first_treat",
        t="year",
        i="id",
        bstrap=True,
        cband=True,
        biters=BITERS,
        random_state=SEED,
    )


@pytest.fixture(scope="module")
def py_clustered(cs_data):
    return sp.callaway_santanna(
        cs_data,
        y="y",
        g="first_treat",
        t="year",
        i="id",
        bstrap=True,
        cband=True,
        clustervars=["id", "clust"],
        biters=BITERS,
        random_state=SEED,
    )


def test_att_gt_point_estimates_match_R(py_unit, r_reference):
    d = py_unit.detail.set_index(["group", "time"])
    for row in _r_cells(r_reference["unit"]).itertuples(index=False):
        key = (int(row.g), int(row.t))
        assert key in d.index, f"missing (g,t) cell {key}"
        diff = abs(float(d.loc[key, "att"]) - row.att)
        assert diff < 1e-6, (
            f"ATT{key} drifted from R did: Python={d.loc[key, 'att']:.8f} "
            f"R={row.att:.8f} (point estimates are deterministic — this "
            f"is an estimator bug, not bootstrap noise)"
        )


def test_bootstrap_ses_match_R(py_unit, r_reference):
    d = py_unit.detail.set_index(["group", "time"])
    for row in _r_cells(r_reference["unit"]).itertuples(index=False):
        key = (int(row.g), int(row.t))
        rel = abs(float(d.loc[key, "se"]) / row.se - 1)
        assert rel < 0.08, (
            f"bootstrap SE{key} drifted from R did by {rel:.1%} "
            f"(Python={d.loc[key, 'se']:.5f}, R={row.se:.5f}). "
            f"Tolerance 8%: independent multiplier draws at biters={BITERS}."
        )


def test_uniform_critical_value_matches_R(py_unit, r_reference):
    crit_py = py_unit.model_info["crit_val_uniform"]
    crit_r = r_reference["unit"]["crit_val"]
    rel = abs(crit_py / crit_r - 1)
    assert rel < 0.05, (
        f"uniform crit drifted from R did by {rel:.1%} "
        f"(Python={crit_py:.4f}, R={crit_r:.4f})"
    )
    # sup-t band must be wider than pointwise normal.
    assert crit_py > 1.96


def test_clustered_ses_match_R(py_clustered, r_reference):
    d = py_clustered.detail.set_index(["group", "time"])
    for row in _r_cells(r_reference["clustered"]).itertuples(index=False):
        key = (int(row.g), int(row.t))
        rel = abs(float(d.loc[key, "se"]) / row.se - 1)
        assert rel < 0.10, (
            f"clustered SE{key} drifted from R did by {rel:.1%} "
            f"(Python={d.loc[key, 'se']:.5f}, R={row.se:.5f}). "
            f"Tolerance 10%: only 15 clusters — lumpy multiplier "
            f"distribution, see module docstring."
        )


def test_clustered_crit_matches_R(py_clustered, r_reference):
    crit_py = py_clustered.model_info["crit_val_uniform"]
    crit_r = r_reference["clustered"]["crit_val"]
    rel = abs(crit_py / crit_r - 1)
    assert rel < 0.05, (
        f"clustered uniform crit drifted from R did by {rel:.1%} "
        f"(Python={crit_py:.4f}, R={crit_r:.4f})"
    )


def test_overall_simple_att_matches_R(py_unit, r_reference):
    r_att = r_reference["agg_simple"]["att"]
    r_se = r_reference["agg_simple"]["se"]
    assert abs(py_unit.estimate - r_att) < 1e-6, (
        f"overall simple ATT drifted: Python={py_unit.estimate:.8f} "
        f"R={r_att:.8f} (deterministic — estimator bug)"
    )
    rel = abs(py_unit.se / r_se - 1)
    assert rel < 0.10, (
        f"overall bootstrap SE drifted from R did::aggte by {rel:.1%} "
        f"(Python={py_unit.se:.5f}, R={r_se:.5f})"
    )


def test_fixture_meta(r_reference):
    meta = r_reference["meta"]
    assert meta["biters"] == BITERS
    assert "did_version" in meta
