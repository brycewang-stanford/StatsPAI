"""``sp.lmtp`` vs R ``lmtp::lmtp_tmle`` / ``lmtp_sdr`` 1.5.4 (AGPL, black box).

Fixtures from ``_fixtures/_generate_lmtp_R.R``: a point treatment and a
two-period design (also with right censoring), continuous and binary
outcomes, policy ``A + 0.5``, ``SL.glm`` learners, ``folds = 1``. The
density ratios agree to 1e-11; estimates and standard errors to 1e-7
relative (R's glm stops at its deviance tolerance; the binary TMLE rows
are the loosest at 1.9e-7).
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
BASE = {
    "point_continuous": ("lmtp_point.csv", "Y", ["A"], None, None),
    "point_binary": ("lmtp_point.csv", "Yb", ["A"], None, None),
    "long_continuous": ("lmtp_long.csv", "Y", ["A1", "A2"], [[], ["L2"]], None),
    "long_binary": ("lmtp_long.csv", "Yb", ["A1", "A2"], [[], ["L2"]], None),
    "cens_continuous": ("lmtp_cens.csv", "Y", ["A1", "A2"], [[], ["L2"]], ["C1", "C2"]),
    "cens_binary": ("lmtp_cens.csv", "Yb", ["A1", "A2"], [[], ["L2"]], ["C1", "C2"]),
}
CASES = dict(BASE)
CASES.update({f"sdr_{k}": v for k, v in BASE.items()})


def _fit(name):
    f, y, trt, tv, cens = CASES[name]
    d = pd.read_csv(_FIX / f)
    return sp.lmtp(
        d,
        y,
        trt,
        shift=0.5,
        covariates_time=tv,
        baseline=["W"],
        censoring=cens,
        estimator="sdr" if name.startswith("sdr_") else "tmle",
    )


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


def test_sdr_estimate_is_the_mean_of_its_influence_function():
    # SDR is a one-step estimator: the centred influence function sums to 0
    r = _fit("sdr_cens_continuous")
    eif = r.model_info["influence_function"]
    assert abs(eif.mean()) < 1e-12 * abs(r.estimate)


def test_censored_rows_get_ratio_zero_and_no_censoring_is_neutral():
    d = pd.read_csv(_FIX / "lmtp_cens.csv")
    r = _fit("cens_continuous")
    ratios = r.model_info["density_ratios"]
    assert np.all(ratios[d["C1"].to_numpy() == 0] == 0)
    assert np.all(ratios[d["C2"].to_numpy() == 0, 1] == 0)
    # indicators that are 1 throughout change nothing
    long = pd.read_csv(_FIX / "lmtp_long.csv").assign(C1=1, C2=1)
    a = sp.lmtp(
        long,
        "Y",
        ["A1", "A2"],
        shift=0.5,
        covariates_time=[[], ["L2"]],
        baseline=["W"],
    )
    b = sp.lmtp(
        long,
        "Y",
        ["A1", "A2"],
        shift=0.5,
        covariates_time=[[], ["L2"]],
        baseline=["W"],
        censoring=["C1", "C2"],
    )
    assert abs(a.estimate - b.estimate) < 1e-9 * abs(a.estimate)
    assert abs(a.se - b.se) < 1e-7 * a.se


def test_censoring_must_be_monotone_and_complete():
    d = pd.read_csv(_FIX / "lmtp_cens.csv")
    bad = d.copy()
    bad.loc[bad["C1"] == 0, "C2"] = 1
    with pytest.raises(sp.exceptions.MethodIncompatibility, match="monotone"):
        sp.lmtp(
            bad,
            "Y",
            ["A1", "A2"],
            shift=0.5,
            covariates_time=[[], ["L2"]],
            baseline=["W"],
            censoring=["C1", "C2"],
        )
    gap = d.copy()
    gap.loc[gap.index[gap["C2"] == 1][0], "Y"] = np.nan
    with pytest.raises(sp.exceptions.MethodIncompatibility, match="outcome"):
        sp.lmtp(
            gap,
            "Y",
            ["A1", "A2"],
            shift=0.5,
            covariates_time=[[], ["L2"]],
            baseline=["W"],
            censoring=["C1", "C2"],
        )
