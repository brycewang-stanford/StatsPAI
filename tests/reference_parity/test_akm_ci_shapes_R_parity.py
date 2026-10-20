"""AKM / AKM0 rows on every shape of the AKM0 confidence set, against R.

``statspai.bartik._akm`` is a clean-room implementation (spec:
``docs/dev/cleanroom/akm.md``). Reference: R 4.5.2, ``ShiftShareSE`` 1.1.0
``ivreg_ss.fit`` / ``reg_ss.fit`` (GPL-3, run as a black box) on the six
designs of ``_fixtures/_generate_akm_ci_shapes_data.py`` -- strong to absent
first stages (bounded, two-half-line and whole-line AKM0 sets), location
weights, a duplicated share column -- at ``beta0 = 0.5`` and
``alpha = 0.10`` (``_fixtures/_generate_akm_ci_shapes_R.R``).

Tolerance 1e-9 relative (closed-form least squares on identical bytes;
observed 1e-12 or better), p-values with an absolute floor of 1e-15 (the
reference computes ``2 * (1 - Phi(|t|))``). Infinite endpoints must match
exactly. For the union shape both sides report ``ci_l > ci_r``: the set is
``(-inf, ci_r] U [ci_l, inf)``.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from statspai.bartik._akm import _akm_fit

_FIX = Path(__file__).parent / "_fixtures"
R = json.loads((_FIX / "akm_ci_shapes_R.json").read_text(encoding="utf-8"))
LOC = pd.read_csv(_FIX / "akm_ci_shapes_loc.csv")
SHARES = pd.read_csv(_FIX / "akm_ci_shapes_shares.csv", header=None).to_numpy()
ROWS = ("Homoscedastic", "EHW", "Reg. cluster", "AKM", "AKM0")


def _num(v):
    return {"Inf": np.inf, "-Inf": -np.inf, "NaN": np.nan, "NA": np.nan}.get(v, v)


def _close(ours, ref, atol=0.0):
    ref = float(_num(ref))
    if np.isinf(ref):
        assert ours == ref
    else:
        np.testing.assert_allclose(float(ours), ref, rtol=1e-9, atol=atol)


@pytest.mark.parametrize("design", range(6))
@pytest.mark.parametrize("mode", ["iv", "ols"])
def test_all_rows_match_shiftsharese(design, mode):
    ref = R[f"d{design}"][mode]
    mask = (LOC["design"] == design).to_numpy()
    L = LOC[mask]
    W = SHARES[mask]
    Z = np.column_stack([np.ones(len(L)), L["c1"].to_numpy()])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        k = _akm_fit(
            L["y"].to_numpy(),
            L["X"].to_numpy(),
            W,
            Z,
            y2=L["x"].to_numpy() if mode == "iv" else None,
            region_cvar=L["region"].to_numpy(),
            beta0=0.5,
            alpha=0.10,
            w=L["w"].to_numpy() if design >= 4 else None,
        )
    _close(k["beta"], ref["beta"])
    for row in ROWS:
        _close(k["se"][row], ref["se"][row])
        _close(k["p"][row], ref["p"][row], atol=1e-15)
        _close(k["ci_l"][row], ref["ci_l"][row])
        _close(k["ci_r"][row], ref["ci_r"][row])
    lo, hi = float(_num(ref["ci_l"]["AKM0"])), float(_num(ref["ci_r"]["AKM0"]))
    expected = "real_line" if np.isinf(lo) else ("union" if lo > hi else "bounded")
    assert k["akm0_ci_type"] == expected


def test_fixture_covers_every_set_shape():
    kinds = set()
    for d in range(6):
        for mode in ("iv", "ols"):
            ref = R[f"d{d}"][mode]
            lo, hi = float(_num(ref["ci_l"]["AKM0"])), float(_num(ref["ci_r"]["AKM0"]))
            kinds.add(
                "real_line" if np.isinf(lo) else ("union" if lo > hi else "bounded")
            )
    assert kinds == {"bounded", "union", "real_line"}


def test_akm0_set_is_where_the_null_imposed_test_does_not_reject():
    # Identity: b is in the AKM0 set iff |beta - b| <= z * se_null(b).
    mask = (LOC["design"] == 2).to_numpy()
    L = LOC[mask]
    Z = np.column_stack([np.ones(len(L)), L["c1"].to_numpy()])
    args = (L["y"].to_numpy(), L["X"].to_numpy(), SHARES[mask], Z)
    base = _akm_fit(*args, y2=L["x"].to_numpy(), alpha=0.10)
    assert base["akm0_ci_type"] == "union"
    lo, hi = base["ci_l"]["AKM0"], base["ci_r"]["AKM0"]
    for b in np.linspace(hi - 3, lo + 3, 41):
        p = _akm_fit(*args, y2=L["x"].to_numpy(), beta0=b, alpha=0.10)["p"]["AKM0"]
        inside = b <= hi or b >= lo
        if min(abs(b - hi), abs(b - lo)) > 1e-6:
            assert (p >= 0.10) == inside
