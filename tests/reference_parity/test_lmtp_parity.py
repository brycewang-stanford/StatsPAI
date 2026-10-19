"""``sp.lmtp`` vs R ``lmtp::lmtp_tmle`` 1.5.4 (AGPL, black box).

Fixtures from ``_fixtures/_generate_lmtp_R.R``: a point treatment and a
two-period design, continuous and binary outcomes, policy ``A + 0.5``,
``SL.glm`` learners, ``folds = 1``. The density ratios agree to 1e-11;
estimates and standard errors to 1e-7 relative (R's glm stops at its
deviance tolerance; the binary rows are the loosest at 1.3e-7).
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd
import pytest

import statspai as sp

_FIX = pathlib.Path(__file__).parent / "_fixtures"
REF = json.loads((_FIX / "lmtp_R.json").read_text(encoding="utf-8"))
CASES = {
    "point_continuous": ("lmtp_point.csv", "Y", ["A"], None),
    "point_binary": ("lmtp_point.csv", "Yb", ["A"], None),
    "long_continuous": ("lmtp_long.csv", "Y", ["A1", "A2"], [[], ["L2"]]),
    "long_binary": ("lmtp_long.csv", "Yb", ["A1", "A2"], [[], ["L2"]]),
}


def _fit(name):
    f, y, trt, tv = CASES[name]
    d = pd.read_csv(_FIX / f)
    return sp.lmtp(d, y, trt, shift=0.5, covariates_time=tv, baseline=["W"])


@pytest.mark.parametrize("name", list(CASES))
def test_lmtp_matches_r(name):
    r = _fit(name)
    ref = REF[name]
    np.testing.assert_allclose(
        r.model_info["density_ratios"],
        np.asarray(ref["ratios"], dtype=float).reshape(
            r.model_info["density_ratios"].shape
        ),
        rtol=1e-9,
    )
    assert abs(r.estimate / ref["est"] - 1) < 3e-7
    assert abs(r.se / ref["se"] - 1) < 3e-7


def test_zero_shift_is_the_natural_course():
    d = pd.read_csv(_FIX / "lmtp_point.csv")
    r = sp.lmtp(d, "Y", ["A"], shift=0.0, baseline=["W"])
    assert abs(r.estimate - d["Y"].mean()) < 1e-6 * abs(d["Y"].mean())
    assert abs(r.model_info["effect"]) < 1e-6


def test_callable_shift_equals_the_number():
    d = pd.read_csv(_FIX / "lmtp_long.csv")
    a = sp.lmtp(
        d, "Y", ["A1", "A2"], shift=0.5, covariates_time=[[], ["L2"]], baseline=["W"]
    )
    b = sp.lmtp(
        d,
        "Y",
        ["A1", "A2"],
        shift=lambda x: x + 0.5,
        covariates_time=[[], ["L2"]],
        baseline=["W"],
    )
    assert a.estimate == b.estimate and a.se == b.se
