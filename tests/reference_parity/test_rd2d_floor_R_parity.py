"""Reference parity: ``sp.rd2d`` vs R ``rd2d`` 1.0.0 where the sample floors bind.

Fixture: ``_fixtures/_generate_rd2d_floor_R.R`` -> ``rd2d_floor_R.json`` on
``rd2d_floor.csv`` (n = 600, so ``bwcheck = 50 + p + 1`` binds in the
selectors and in the estimation windows). R was run as a black box; the
module is the clean-room implementation of Cattaneo, Titiunik and Yu
(``cattaneo2025boundary``), spec ``docs/dev/cleanroom/rd2d.md``. The main
fixture ``test_rd_open_R_parity.py`` (n = 2,500) rarely reaches the floors.

What is pinned, per cell (location: default, msetwo, imserd, certwo, a user
bandwidth below the floor, the radial kernel with selected and user
bandwidths, fuzzy, clusters with separate fits, unstandardised ``p = 2``;
distance: default, fuzzy, unknown and known kinks, imserd, certwo,
clusters): every row of R's ``$main`` table (estimates and standard errors
of orders p and q, bandwidths, effective sample sizes) and the covariance of
the bias-corrected estimates. Radial-kernel standard errors are not
compared: the reference rescales them by ``(h1^2 + h2^2) / (h1 h2)`` (the
documented class-4 defect of ``test_rd_open_R_parity.py``).

Tolerances: closed-form least squares rel 1e-9; bandwidths rel 1e-8.
"""

from __future__ import annotations

import json
import pathlib
import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp

FX = pathlib.Path(__file__).parent / "_fixtures"
R = json.loads((FX / "rd2d_floor_R.json").read_text(encoding="utf-8"))
DF = pd.read_csv(FX / "rd2d_floor.csv")
B = np.array(R["_meta"]["b"], dtype=float)

_RNAME = {"kink.unknown": "kink_unknown"}


def _kwargs(cell):
    ref = R[cell]
    a = dict(ref["args"]) if isinstance(ref["args"], dict) else {}
    kw = {"approach": ref["approach"]}
    if a.pop("fz", False):
        kw.update(y="yf", fuzzy="takeup")
    if "cluster" in cell:
        kw["cluster"] = "g"
    if "kink.position" in a:
        kw["kink_position"] = [int(a.pop("kink.position")) - 1]
    for k, v in a.items():
        kw[_RNAME.get(k, k)] = v
    return kw


_CELLS = sorted(k for k in R if not k.startswith("_"))


@pytest.mark.parametrize("cell", _CELLS)
def test_rd2d_matches_R_with_binding_floors(cell):
    kw = _kwargs(cell)
    y = kw.pop("y", "y")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sp.rd2d(DF, y=y, x1="x1", x2="x2", treatment="t", eval_points=B, **kw)
    det, ref = res.detail, R[cell]["main"]
    rad = kw.get("kernel_type") == "rad"
    cols = [("estimate.p", "estimate_p"), ("estimate.q", "estimate_q")]
    if not rad:
        cols += [("std.err.p", "std_err_p"), ("std.err.q", "std_err_q")]
    for rc, pc in cols:
        np.testing.assert_allclose(det[pc], ref[rc], rtol=1e-9, err_msg=rc)
    bw = ["h01", "h02", "h11", "h12"] if kw["approach"] == "location" else ["h0", "h1"]
    for c in bw:
        np.testing.assert_allclose(det[c], ref[c], rtol=1e-8, err_msg=c)
    np.testing.assert_array_equal(det["n_co"], np.asarray(ref["N.Co"], dtype=int))
    np.testing.assert_array_equal(det["n_tr"], np.asarray(ref["N.Tr"], dtype=int))
    if not rad:
        C = np.asarray(R[cell]["cov"], dtype=float)
        scale = np.sqrt(np.outer(np.diag(C), np.diag(C)))
        np.testing.assert_allclose(
            res.model_info["cov_q"] / scale, C / scale, atol=1e-9
        )


def test_user_bandwidth_below_the_floor_is_reported_as_given():
    """The floor widens the fit's window, not the reported bandwidth."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sp.rd2d(DF, y="y", x1="x1", x2="x2", treatment="t", eval_points=B, h=0.1)
    assert np.all(res.detail[["h01", "h02", "h11", "h12"]].to_numpy() == 0.1)
    used = res.model_info["h_used_p"]
    assert np.all(used > 0.1)
    # 52 observations reach the window edge; the edge one has zero weight
    assert np.all(res.detail[["n_co", "n_tr"]].to_numpy() == 51)
