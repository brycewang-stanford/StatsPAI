"""sp.evalue_rd's confidence-limit grid rule against R EValue (black box).

Fixture: ``_fixtures/evalue_rd_grid_R.json`` from
``_generate_evalue_rd_grid_R.R`` (EValue 4.1.4, R 4.5.2). Coarse grids make
the rule visible: the answer is the first grid bias factor ``1 + k * grid``
whose delta-method lower confidence limit reaches ``true`` (Ding &
VanderWeele 2016, eAppendix eq. A.5), converted to the E-value scale.

Tolerance 1e-12: the point E-value is closed form and the grid points agree
with R's ``seq`` to a few ulps.

One documented departure (docs/dev/cleanroom/evalue.md): when no grid point
below the point-estimate bias factor qualifies, EValue returns NA; StatsPAI
uses that bias factor as the last grid point, so the answer is the point
E-value, which still overshoots the exact root by less than one grid step.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

import statspai as sp

_FIX = Path(__file__).parent / "_fixtures" / "evalue_rd_grid_R.json"
R = json.loads(_FIX.read_text(encoding="utf-8"))
CASES = [c for c in R["cases"] if c["lower"] is not None]
NA_CASES = [c for c in R["cases"] if c["lower"] is None]


def _ids(c):
    return f"{c['cells']}_t{c['true']}_g{c['grid']}"


@pytest.mark.parametrize("case", CASES, ids=_ids)
def test_evalue_rd_matches_evalues_RD(case):
    out = sp.evalue_rd(
        *case["cells"], true=case["true"], alpha=case["alpha"], grid=case["grid"]
    )
    assert out["evalue_estimate"] == pytest.approx(case["est"], rel=1e-12)
    assert out["evalue_ci"] == pytest.approx(case["lower"], rel=1e-12)


def _lower_limit(bf, cells, alpha):
    n11, n10, n01, n00 = map(float, cells)
    n1, n0 = n11 + n10, n01 + n00
    f, p1, p0 = n1 / (n1 + n0), n11 / n1, n01 / n0
    v1, v0, vf = p1 * (1 - p1) / n1, p0 * (1 - p0) / n0, f * (1 - f) / (n1 + n0)
    w = f + (1 - f) / bf
    se = np.sqrt(
        (v1 + v0 * bf**2) * w**2 + (p1 - p0 * bf) ** 2 * (1 - 1 / bf) ** 2 * vf
    )
    return (p1 - p0 * bf) * w - stats.norm.ppf(1 - alpha / 2) * se


@pytest.mark.parametrize("case", NA_CASES, ids=_ids)
def test_no_qualifying_grid_point_uses_the_point_bias_factor(case):
    out = sp.evalue_rd(
        *case["cells"], true=case["true"], alpha=case["alpha"], grid=case["grid"]
    )
    bf = out["bias_factor"]
    grid_points = 1.0 + case["grid"] * np.arange(int((bf - 1.0) / case["grid"]) + 1)
    # The R NA is real: no grid point below the point bias factor qualifies ...
    assert np.all(
        _lower_limit(grid_points, case["cells"], case["alpha"]) > case["true"]
    )
    # ... while the point bias factor always does, and it is used instead.
    assert _lower_limit(bf, case["cells"], case["alpha"]) <= case["true"]
    assert out["evalue_ci"] == pytest.approx(case["est"], rel=1e-12)
    assert out["evalue_ci"] == pytest.approx(bf + np.sqrt(bf * (bf - 1)), rel=1e-14)


def test_point_bias_factor_moves_the_bound_to_true():
    """Identity (Proposition A.8): (p1 - p0 B)(f + (1 - f)/B) = true at B*."""
    cells, true = (397, 78557, 51, 108778), 0.001
    out = sp.evalue_rd(*cells, true=true)
    n11, n10, n01, n00 = cells
    f = (n11 + n10) / sum(cells)
    p1, p0 = n11 / (n11 + n10), n01 / (n01 + n00)
    b = out["bias_factor"]
    assert (p1 - p0 * b) * (f + (1 - f) / b) == pytest.approx(true, rel=1e-12)


def test_no_unexposed_cases_is_refused():
    with pytest.raises(sp.exceptions.MethodIncompatibility, match="unbounded"):
        sp.evalue_rd(10, 90, 0, 100)
