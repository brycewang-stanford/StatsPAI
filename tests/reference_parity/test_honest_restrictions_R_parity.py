"""Honest-DiD restriction families vs R ``HonestDiD`` 0.2.8.

``statspai.did._honest_sets`` builds HonestDiD's moment matrices for the
sign restrictions (``Delta^SDPB`` / ``SDNB``, ``Delta^RMPB`` / ``RMNB`` ...),
the shape restrictions (``Delta^SDI`` / ``SDD``, ``Delta^RMI`` / ``RMD`` ...)
and the relative magnitudes of second differences (``Delta^SDRM`` and its
sign / shape variants), lays out HonestDiD's default grids and inverts the
Andrews-Roth-Pakes tests of :mod:`statspai.did._arp`. Both packages report
the smallest and largest accepted grid point, so:

* ``Conditional`` involves no simulation: a correct port accepts the same
  grid points (``atol=1e-12``), with the grid laid out the same way (the
  identified set at ``beta = 0`` for the smoothness family). These are T2
  rows, run with ``max_expand=0`` (HonestDiD reports the grid end of a set
  that reaches it; ``sp.honest_did`` extends instead, asserted wider).
* ``C-F`` (conditional-FLCI hybrid): its first stage and its grid are the
  level ``alpha/10`` FLCI, whose centre the two FLCI solvers agree on only
  to ~1e-5 (``test_honest_did_options_R_parity.py``). Against HonestDiD with
  the exact folded-normal quantile the grid ends agree to 1e-4 and the
  bounds to one grid step; against the shipped package (simulated
  quantile) to two. T4.
* ``C-LF`` simulates its first-stage critical value with each side's own
  RNG (1,000 draws): HonestDiD's bound lies inside the spread of the native
  bound over ten seeds, plus one grid step. A stochastic screen (S).

Fixture: ``_fixtures/honest_restrictions_R.json`` from
``_fixtures/_generate_honest_restrictions_R.sh``.
"""

from __future__ import annotations

import json
import pathlib
import warnings

import numpy as np
import pytest

import statspai as sp
from statspai.did._honest_sets import (
    create_a_monotone,
    rm_family_confidence_set,
    sd_family_confidence_set,
)

_FIX = pathlib.Path(__file__).parent / "_fixtures"
REF = json.loads((_FIX / "honest_restrictions_R.json").read_text(encoding="utf-8"))
CASES = sorted(k for k in REF if not k.startswith("_"))

pytestmark = pytest.mark.slow

_DIR = {"PB": ("positive", None), "NB": ("negative", None)}
_DIR.update({"I": (None, "increasing"), "D": (None, "decreasing"), "": (None, None)})
_HYBRID = {"Conditional": "ARP", "C-F": "FLCI", "C-F|shipped": "FLCI", "C-LF": "LF"}


def _parse(delta: str):
    if delta.startswith("SDRM"):
        return "sdrm", _DIR[delta[4:]]
    if delta.startswith("RM"):
        return "rm", _DIR[delta[2:]]
    return "smoothness", _DIR[delta[2:]]


def _run(key: str, max_expand: int = 0, seed: int = 0):
    c = REF[key]
    b = np.asarray(c["betahat"], float)
    s = np.asarray(c["sigma"], float)
    npre, npost = int(c["npre"]), int(c["npost"])
    lv = np.atleast_1d(np.asarray(c["l_vec"], float))
    family, (bias, mono) = _parse(c["delta"])
    hybrid = _HYBRID[c["method"]]
    out = []
    grid_values = c["M"] if family == "smoothness" else c["Mbar"]
    for m in np.atleast_1d(grid_values):
        if family == "smoothness":
            res = sd_family_confidence_set(
                b,
                s,
                npre,
                npost,
                float(m),
                lv,
                hybrid=hybrid,
                bias=bias,
                mono=mono,
                max_expand=max_expand,
                seed=seed,
            )
        else:
            res = rm_family_confidence_set(
                b,
                s,
                npre,
                npost,
                float(m),
                lv,
                hybrid=hybrid,
                base=family,
                bias=bias,
                mono=mono,
                max_expand=max_expand,
                seed=seed,
            )
        out.append(res)
    return c, out


def _keys(method: str):
    return [k for k in CASES if k.endswith("|" + method)]


@pytest.mark.parametrize("key", _keys("Conditional"))
def test_conditional_set_is_identical_to_honestdid(key):
    c, out = _run(key)
    lb, ub = np.atleast_1d(c["lb"]), np.atleast_1d(c["ub"])
    for i, (lo, hi, grid, _) in enumerate(out):
        assert lo == pytest.approx(lb[i], abs=1e-12), (key, i)
        assert hi == pytest.approx(ub[i], abs=1e-12), (key, i)
        if "grid_lo" in c:  # the smoothness grid: identified set +/- 20 sd
            assert grid[0] == pytest.approx(np.atleast_1d(c["grid_lo"])[i], abs=1e-12)
            assert grid[-1] == pytest.approx(np.atleast_1d(c["grid_hi"])[i], abs=1e-12)


@pytest.mark.parametrize(
    "key", [k for k in _keys("Conditional") if "grid_lo" in REF[k]]
)
def test_extended_grid_contains_honestdid_set(key):
    """Where HonestDiD's set reaches its grid end, the extended grid closes
    it further out; elsewhere it is the same set."""
    c, out0 = _run(key)
    _, out = _run(key, max_expand=10)
    for i, ((lo0, hi0, _, acc0), (lo, hi, _, _)) in enumerate(zip(out0, out)):
        if acc0[0] == 1 or acc0[-1] == 1:
            assert lo <= lo0 and hi >= hi0, (key, i)
        else:
            assert (lo, hi) == (lo0, hi0), (key, i)


@pytest.mark.parametrize("key", _keys("C-F") + _keys("C-F|shipped"))
def test_cf_hybrid_within_grid_steps_of_honestdid(key):
    c, out = _run(key)
    steps = 2 if key.endswith("shipped") else 1
    lb, ub = np.atleast_1d(c["lb"]), np.atleast_1d(c["ub"])
    for i, (lo, hi, grid, _) in enumerate(out):
        step = grid[1] - grid[0]
        if not key.endswith("shipped"):
            span = grid[-1] - grid[0]
            assert grid[0] == pytest.approx(
                np.atleast_1d(c["grid_lo"])[i], abs=1e-4 * span
            )
            assert grid[-1] == pytest.approx(
                np.atleast_1d(c["grid_hi"])[i], abs=1e-4 * span
            )
        assert abs(lo - lb[i]) <= steps * step * (1 + 1e-6) + 1e-9, (key, i)
        assert abs(hi - ub[i]) <= steps * step * (1 + 1e-6) + 1e-9, (key, i)


@pytest.mark.parametrize("key", _keys("C-LF"))
def test_clf_hybrid_inside_its_seed_spread(key):
    """HonestDiD's bound lies within the spread of the native bound over ten
    seeds of the simulated first-stage critical value, widened by one grid
    step: the two sides differ by their RNG draws, nothing else."""
    c, _ = _run(key)
    runs = [_run(key, seed=seed)[1] for seed in range(10)]
    lb, ub = np.atleast_1d(c["lb"]), np.atleast_1d(c["ub"])
    for i in range(len(lb)):
        los = [r[i][0] for r in runs]
        his = [r[i][1] for r in runs]
        step = runs[0][i][2][1] - runs[0][i][2][0]
        assert min(los) - step * 1.000001 <= lb[i] <= max(los) + step * 1.000001, (
            key,
            i,
        )
        assert min(his) - step * 1.000001 <= ub[i] <= max(his) + step * 1.000001, (
            key,
            i,
        )


def test_monotone_matrix_with_one_pre_period_is_the_definition():
    """``Delta^SDI``: delta_{-2} <= delta_{-1} = 0 <= delta_0 <= delta_1.

    HonestDiD's ``.create_A_M(1, 2)`` loops over ``1:0`` in R and writes
    ``(1, -1, 0)`` into the first row, i.e. delta_{-2} <= delta_0; the
    intended row is ``(1, 0, 0)``. Each row here is one adjacent pair.
    """
    a = create_a_monotone(1, 2, "increasing")
    np.testing.assert_array_equal(
        a, np.array([[1.0, 0.0, 0.0], [0.0, -1.0, 0.0], [0.0, 1.0, -1.0]])
    )
    # Every increasing path through 0 satisfies A delta <= 0, and a path
    # that decreases anywhere violates it.
    assert np.all(a @ np.array([-0.3, 0.1, 0.4]) <= 0)
    assert np.any(a @ np.array([-0.3, 0.5, 0.4]) > 0)


def _result(key):
    c = REF[key]
    b = np.asarray(c["betahat"], float)
    s = np.asarray(c["sigma"], float)
    npre = int(c["npre"])
    times = list(range(-npre - 1, -1)) + list(range(int(c["npost"])))
    return b, s, times, np.atleast_1d(np.asarray(c["l_vec"], float))


@pytest.mark.parametrize(
    "key, kw",
    [
        (
            "corr33|SDPB|Conditional",
            dict(method="smoothness", bias_direction="positive"),
        ),
        (
            "corr33|SDD|Conditional",
            dict(method="smoothness", monotonicity_direction="decreasing"),
        ),
        ("corr33|SDRM|Conditional", dict(method="smoothness_relative_magnitude")),
        (
            "corr33|RMI|Conditional",
            dict(method="relative_magnitude", monotonicity_direction="increasing"),
        ),
    ],
)
def test_public_entry_point_matches_honestdid(key, kw):
    c = REF[key]
    b, s, times, lv = _result(key)
    grid = c["M"] if "M" in c else c["Mbar"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = sp.honest_did_from_moments(
            b,
            s,
            event_times=times,
            l_vec=lv,
            m_grid=list(np.atleast_1d(grid)),
            honestdid_method="Conditional",
            grid_expand=False,
            **kw,
        )
    assert out.attrs["interval"] == "arp_conditional"
    assert out.attrs["delta"] == "Delta" + c["delta"]
    np.testing.assert_allclose(out["ci_lower"], np.atleast_1d(c["lb"]), atol=1e-12)
    np.testing.assert_allclose(out["ci_upper"], np.atleast_1d(c["ub"]), atol=1e-12)
