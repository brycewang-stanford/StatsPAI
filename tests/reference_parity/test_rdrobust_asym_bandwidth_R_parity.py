"""``sp.rdrobust`` with user-supplied asymmetric bandwidths vs R rdrobust 4.0.0.

``h`` and ``b`` are typed ``float | (left, right)`` and the estimator handles
pairs, but the input check called ``float()`` on them, so a pair -- for
instance ``h`` / ``b`` from ``bwselect='msetwo'`` -- was rejected
(``MethodIncompatibility: `h` must be a finite number``; found on the
Watering Down replication, QJE 2020). Reference:
``_fixtures/_generate_rdrobust_asym_R.R`` -- ``rdrobust(h = c(14, 19),
b = c(22, 27))`` on ``rdsenate_params.csv``, with and without clusters.
"""

from __future__ import annotations

import json
import pathlib
import warnings

import pandas as pd
import pytest

import statspai as sp

_FIX = pathlib.Path(__file__).parent / "_fixtures"
R = json.loads((_FIX / "rdrobust_asym_R.json").read_text(encoding="utf-8"))["cases"]


@pytest.fixture(scope="module")
def senate():
    return pd.read_csv(_FIX / "rdsenate_params.csv").dropna(subset=["vote", "margin"])


@pytest.mark.parametrize("case", ["plain", "cluster"])
def test_asymmetric_bandwidths_match_r(senate, case):
    ref = R[case]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = sp.rdrobust(
            senate,
            y="vote",
            x="margin",
            c=0,
            h=(14, 19),
            b=(22, 27),
            cluster="clust" if case == "cluster" else None,
        )
    det = r.detail.set_index("method")
    assert r.model_info["bandwidth_h"] == (14.0, 19.0)
    assert det.loc["Conventional", "estimate"] == pytest.approx(ref["conv"], rel=1e-9)
    assert det.loc["Conventional", "se"] == pytest.approx(ref["se_conv"], rel=1e-9)
    assert det.loc["Robust", "estimate"] == pytest.approx(ref["bc"], rel=1e-9)
    assert det.loc["Robust", "se"] == pytest.approx(ref["se_rob"], rel=1e-9)


def test_bad_bandwidth_pair_raises(senate):
    with pytest.raises(sp.MethodIncompatibility, match="pair"):
        sp.rdrobust(senate, y="vote", x="margin", h=(1, 2, 3))
    with pytest.raises(sp.MethodIncompatibility, match=r"h\[1\]"):
        sp.rdrobust(senate, y="vote", x="margin", h=(1, -2))
