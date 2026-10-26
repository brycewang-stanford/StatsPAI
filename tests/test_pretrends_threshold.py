"""``threshold=`` of sp.pretrends_power / sp.pretrends_slope_for_power.

Roth's ``pretrends`` package fixes the pre-test cutoff at 1.96 while
StatsPAI's default is ``qnorm(1 - alpha/2)``. The keyword reproduces the
package; the committed Track A module 76 R artifact computes the package's
quantities with an exact integrator and a 1e-14 root, so it is the truth the
``threshold=1.96`` path is held to.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import statspai as sp
from statspai.core.results import CausalResult
from statspai.exceptions import MethodIncompatibility

R_GOLDEN = (
    Path(__file__).resolve().parent / "r_parity" / "results" / "76_pretrends_R.json"
)

T_VEC = np.array([-4.0, -3.0, -2.0, 0.0, 1.0, 2.0])
SE = np.array([0.050, 0.045, 0.040, 0.100, 0.110, 0.120])
BETA = np.array([0.012, -0.008, 0.021, 0.180, 0.240, 0.310])


def _result() -> CausalResult:
    idx = np.arange(len(SE))
    sigma = 0.5 ** np.abs(idx[:, None] - idx[None, :]) * np.outer(SE, SE)
    pre = T_VEC < -1
    es = pd.DataFrame({"relative_time": T_VEC, "att": BETA, "se": SE})
    return CausalResult(
        method="ThresholdTest",
        estimand="ATT(0)",
        estimate=BETA[3],
        se=SE[3],
        pvalue=0.0,
        ci=(0.0, 1.0),
        alpha=0.05,
        n_obs=1000,
        model_info={"event_study": es, "vcv_pre": sigma[np.ix_(pre, pre)]},
    )


def _golden() -> dict[str, float]:
    rows = json.loads(R_GOLDEN.read_text(encoding="utf-8"))["rows"]
    return {r["statistic"]: r["estimate"] for r in rows}


def test_threshold_196_reproduces_pretrends_power_and_slope():
    res, golden = _result(), _golden()
    pre = T_VEC < -1
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for slope, tag in ((0.02, "0p02"), (0.05, "0p05")):
            out = sp.pretrends_power(
                res, delta=slope * (T_VEC[pre] + 1.0), threshold=1.96
            )
            # StatsPAI integrates the rectangle probability to 1e-7.
            assert out["power"] == pytest.approx(
                golden[f"power_slope_{tag}_t196"], rel=1e-6
            )
            assert out["threshold_tstat"] == 1.96
        for target, tag in ((0.5, "0p5"), (0.8, "0p8")):
            out = sp.pretrends_slope_for_power(res, target_power=target, threshold=1.96)
            assert out["slope"] == pytest.approx(
                golden[f"slope_for_power_{tag}_t196"], rel=1e-6
            )
            assert out["achieved_power"] == pytest.approx(target, abs=1e-6)


def test_default_threshold_is_the_normal_quantile():
    res = _result()
    delta = 0.03 * (T_VEC[T_VEC < -1] + 1.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        default = sp.pretrends_power(res, delta=delta)
        explicit = sp.pretrends_power(
            res, delta=delta, threshold=float(stats.norm.ppf(0.975))
        )
        cutoff_196 = sp.pretrends_power(res, delta=delta, threshold=1.96)
    assert default["threshold_tstat"] == explicit["threshold_tstat"]
    assert default["power"] == pytest.approx(explicit["power"], abs=2e-7)
    # A higher cutoff rejects less often, so power falls.
    assert cutoff_196["power"] < default["power"]


@pytest.mark.parametrize("bad", [0.0, -1.96, float("nan"), float("inf"), True, "1.96"])
def test_threshold_must_be_a_positive_finite_number(bad):
    res = _result()
    with pytest.raises(MethodIncompatibility):
        sp.pretrends_power(res, delta=np.zeros(3), threshold=bad)
    with pytest.raises(MethodIncompatibility):
        sp.pretrends_slope_for_power(res, target_power=0.5, threshold=bad)
