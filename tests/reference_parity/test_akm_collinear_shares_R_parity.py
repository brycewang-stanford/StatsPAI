"""AKM inference with collinear share columns against R ``ShiftShareSE``.

Reference: R 4.5.2, ``ShiftShareSE`` 1.1.0 ``ivreg_ss.fit`` on
``_fixtures/akm_collinear_{loc,shares}.csv`` (300 locations, 40 industries;
an exact duplicate, an exact sum and a 1e-12 near-duplicate column), from
``_fixtures/_generate_akm_collinear_shares_R.R``.

``ShiftShareSE`` drops collinear shares with the column-pivoting QR of R's
``qr()`` (``tol = 1e-7``; vignette section 3.1). StatsPAI once used a
one-pass Gram-Schmidt test instead, which keeps a different set on nearly
collinear share matrices: on the ADH (AER 2013) replication's raw
794-industry shares it kept 781 columns to R's 776 and reported an AKM SE of
4.6e4 against R's 1.5e4. (Both are meaningless -- that matrix is
numerically singular, and StatsPAI says so -- but they should be the same
meaningless number.) The selection is a limited-pivoting Householder QR
written from that description (clean-room, ``docs/dev/cleanroom/akm.md``):
the kept columns equal R's exactly, and the AKM / AKM0 numbers agree to
1e-10.
"""

from __future__ import annotations

import importlib
import json
import pathlib
import warnings

import pandas as pd
import pytest

import statspai as sp

_FIX = pathlib.Path(__file__).parent / "_fixtures"


@pytest.fixture(scope="module")
def ref():
    return json.loads(
        (_FIX / "akm_collinear_shares_R.json").read_text(encoding="utf-8")
    )


@pytest.fixture(scope="module")
def data():
    loc = pd.read_csv(_FIX / "akm_collinear_loc.csv")
    W = pd.read_csv(_FIX / "akm_collinear_shares.csv", header=None).to_numpy()
    g = pd.read_csv(_FIX / "akm_collinear_shocks.csv")["g"].to_numpy()
    return loc, W, g


def test_kept_columns_equal_r_qr(ref, data):
    akm = importlib.import_module("statspai.bartik._akm")
    _, W, _ = data
    keep = akm._limited_pivot_rank_columns(W)
    assert len(keep) == ref["rank"]
    assert sorted(keep.tolist()) == ref["keep"]


def test_shift_share_se_matches_ivreg_ss(ref, data):
    loc, W, g = data
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sp.BartikIV(
            loc,
            y="y",
            endog="x",
            shares=W,
            shocks=g,
            covariates=["c1", "c2"],
            leave_one_out=False,
        ).fit()
        with pytest.warns(UserWarning, match="3 collinear share column"):
            ss = sp.shift_share_se(res, shares=W)
    assert float(res.params["x"]) == pytest.approx(ref["beta"], rel=1e-10)
    assert ss.diagnostics["SE (AKM)"] == pytest.approx(ref["se_akm"], rel=1e-10)


def test_clean_shares_do_not_warn_about_conditioning(data):
    akm = importlib.import_module("statspai.bartik._akm")
    _, W, _ = data
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        akm._independent_share_columns(W)
    assert not [m for m in w if "near-singular" in str(m.message)]
