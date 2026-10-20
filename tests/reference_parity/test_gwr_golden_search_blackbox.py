"""GWR bandwidth search against R ``GWmodel`` run as a black box.

``statspai.spatial.gwr.bandwidth`` is a clean-room implementation (spec:
``docs/dev/cleanroom/gwr_bandwidth.md``). Fixture
``_fixtures/gwr_golden_search_R.json`` (generator
``_generate_gwr_golden_search_R.R``, GWmodel 2.4-1, GPL >= 2, exported
functions only):

* ``searches``: 64 runs of ``GWmodel::gold`` on objectives that record their
  argument (monotone, quadratic, wavy and rounded-flat objectives; integer
  and continuous searches; five and four intervals). Every evaluated point
  and the result must be identical (the integer searches are exact; the
  continuous ones are compared bitwise, which they meet).
* ``bw_gwr_sim`` / ``bw_gwr_georgia``: ``bw.gwr`` for three kernels x
  adaptive / fixed x AICc / CV on the simulated points of
  ``_generate_gwr_golden_search_data.py`` and on the Georgia data. The
  criterion values come from ``sp.gwr`` (checked against ``gwr.basic`` in
  ``test_spatial_survey_R_parity.py``), so the selected bandwidths agree to
  1e-10 relative (adaptive: exactly).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.spatial.gwr.bandwidth import _golden_section_search

_FIX = Path(__file__).parent / "_fixtures"
R = json.loads((_FIX / "gwr_golden_search_R.json").read_text(encoding="utf-8"))


def _objective(case):
    p = case["par"]
    return {
        "inc": lambda x: x,
        "dec": lambda x: -x,
        "quad": lambda x: (x - p) ** 2,
        "wav": lambda x: (x - p) ** 2 + 30 * np.sin(x / 3),
        "flat": lambda x: round((x - p) ** 2, 3),
    }[case["kind"]]


@pytest.mark.parametrize("i", range(len(R["searches"])))
def test_search_path_equals_gold(i):
    case = R["searches"][i]
    best, probes = _golden_section_search(
        _objective(case),
        float(case["xL"]),
        float(case["xU"]),
        integer=bool(case["adaptive"]),
    )
    calls = case["calls"] if isinstance(case["calls"], list) else [case["calls"]]
    assert probes == [float(c) for c in calls]
    assert best == float(case["res"])


def _sim():
    df = pd.read_csv(_FIX / "gwr_golden_search_points.csv")
    return df[["u", "v"]].to_numpy(), df["y"].to_numpy(), df[["x1", "x2"]].to_numpy()


def _georgia():
    df = pd.read_csv(_FIX / "spatial_survey_georgia.csv")
    X = df[["PctRural", "PctPov", "PctBlack"]].to_numpy()
    return df[["X", "Y"]].to_numpy(), df["PctBach"].to_numpy(), X


@pytest.mark.parametrize("data", ["sim", "georgia"])
def test_selected_bandwidth_equals_bw_gwr(data):
    coords, y, X = _sim() if data == "sim" else _georgia()
    for s in R[f"bw_gwr_{data}"]:
        bw = sp.gwr_bandwidth(
            coords,
            y,
            X,
            kernel=s["kernel"],
            fixed=not s["adaptive"],
            criterion=s["approach"],
        )
        if s["adaptive"]:
            assert bw == s["bw"], s
        else:
            np.testing.assert_allclose(bw, s["bw"], rtol=1e-10, err_msg=str(s))


def test_unknown_kernel_is_rejected():
    coords, y, X = _sim()
    with pytest.raises(sp.MethodIncompatibility, match="kernel"):
        sp.gwr_bandwidth(coords, y, X, kernel="tricube")


def test_selected_bandwidth_is_a_local_minimum_on_a_unimodal_criterion():
    # Reference-free: a continuous search on a smooth unimodal objective
    # converges to its minimiser within the search tolerance.
    best, probes = _golden_section_search(
        lambda x: (x - 3.7) ** 2, 0.0, 10.0, integer=False, tol=1e-12
    )
    assert abs(best - 3.7) < 1e-5
    assert len(probes) == len(set(probes))


def test_small_samples_search_from_k_plus_two():
    # [20, n] is empty for n <= 20: the default search is the one over
    # [k + 2, n] with k local coefficients (constant included).
    rng = np.random.default_rng(4)
    coords = rng.uniform(0, 10, (16, 2))
    X = rng.normal(size=(16, 2))
    y = X @ [1.0, -0.5] + rng.normal(size=16)
    for crit in ("AICc", "CV"):
        default = sp.gwr_bandwidth(coords, y, X, criterion=crit)
        explicit = sp.gwr_bandwidth(
            coords, y, X, criterion=crit, bw_min=5.0, bw_max=16.0
        )
        assert default == explicit


def test_infeasible_probes_score_inf_and_warn():
    # n = 13: the integer search drifts below its lower end (k + 2 = 5)
    # to one neighbour or fewer, where no GWR exists.
    rng = np.random.default_rng(26)
    coords = rng.uniform(0, 10, (13, 2))
    X = rng.normal(size=(13, 2))
    y = X @ [0.3, 1.0] + rng.normal(size=13)
    with pytest.warns(RuntimeWarning, match="admit no GWR fit"):
        bw = sp.gwr_bandwidth(coords, y, X, criterion="CV")
    assert 5.0 <= bw <= 13.0
