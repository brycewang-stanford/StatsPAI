"""``sp.feols`` inference conventions (found by the flagship workflow tests).

Three defects in the pyfixest adapter, each pinned here:

* intervals used t(N - k) with k the *regressors*; pyfixest / R fixest use
  N - K with the absorbed fixed-effect levels counted, and t(G - 1) under
  clustering -- clustered intervals were too narrow;
* the coefficient covariance was not stored, so ``sp.test`` / ``lincom`` /
  ``margins`` refused any multi-coefficient restriction after ``sp.feols``;
  with it, the joint test's F denominator is G - 1, as R ``fixest::wald``;
* a missing cluster value crashed inside pyfixest instead of being marked
  out as Stata ``vce(cluster)`` / every other StatsPAI estimator does.

R reference (fixest 0.14.0) on ``_fixtures/feols_inference_df_data.csv``:
``wald(feols(y ~ x + x2 | h, d, cluster = ~g), keep = "^x")`` ->
stat 197.177835344815, p 1.96373173923058e-13, df (2, 19), n 295
(h = g %% 7; 5 rows with missing g dropped).
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import statspai as sp

pf = pytest.importorskip("pyfixest")
_FIX = Path(__file__).parent / "reference_parity" / "_fixtures"


@pytest.fixture(scope="module")
def data():
    d = pd.read_csv(_FIX / "feols_inference_df_data.csv")
    d["h"] = d["g"] % 7
    return d


@pytest.mark.parametrize("vcov", ["iid", "hetero", {"CRV1": "g"}])
def test_intervals_match_pyfixest(data, vcov):
    d = data.dropna()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ours = sp.feols("y ~ x + x2 | h", d, vcov=vcov)
        ref = pf.feols("y ~ x + x2 | h", d, vcov=vcov)
    np.testing.assert_allclose(
        ours.conf_int_lower.values, ref.confint().iloc[:, 0], rtol=1e-12
    )
    np.testing.assert_allclose(
        ours.conf_int_upper.values, ref.confint().iloc[:, 1], rtol=1e-12
    )


def test_joint_test_matches_fixest_wald_and_marks_out_missing_clusters(data):
    with pytest.warns(sp.exceptions.StatsPAIWarning, match="missing"):
        fit = sp.feols("y ~ x + x2 | h", data, cluster="g")
    assert fit.data_info["nobs"] == 295
    w = sp.test(fit, "x = x2 = 0")
    assert w["statistic"] == pytest.approx(197.177835344815, rel=1e-12)
    assert w["df"] == (2, 19)
    assert w["pvalue"] == pytest.approx(1.96373173923058e-13, rel=1e-9)
