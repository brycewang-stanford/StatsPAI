"""sp.rd_optimized against optrdd, the method authors' R package.

Fixture: ``_fixtures/rd_optimized_optrdd.json``, produced by
``_fixtures/_generate_rd_optimized_R.R`` from the committed CSV (optrdd
1.0.2 from github.com/swager/optrdd with ``optimizer="quadprog"``, used
as a black box). Three cases: a continuous running variable at two
curvature bounds and a discrete one with sixteen support points. Both
sides are given the same curvature bound and the same variance
(``sigma.sq = 0.25``), so the weights solve the same minimax problem.

Evidence level
--------------
This is not a bit-for-bit comparison and is not described as one. The
two packages discretise the same infinite-dimensional programme in
different ways: optrdd represents the least favourable function on a
grid of the running variable, StatsPAI by a piecewise constant second
derivative between support points. What can be asked is that they land
on the same estimator up to that discretisation.

Observed differences (recorded when the fixture was made): estimates
differ by 0.2 to 1.6e-3, which is at most 0.008 standard errors; the
worst-case bias by 0.3%; the sum of squared weights by 0.2%; the weights
have a correlation above 0.9998. The tolerances below are about three
times those.

One difference is one-sided. optrdd's weights are orthogonal to the
running variable only to about 5e-5 on each side, so under the exact
formula their worst-case bias is unbounded (a steep enough slope breaks
them); the bias optrdd reports is the one of its discretised problem.
StatsPAI imposes the four moment conditions exactly and reports the
exact supremum, which the test also checks.
"""

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.rd.optimized import worst_case_bias

FIX = Path(__file__).parent / "_fixtures"
REF = json.loads((FIX / "rd_optimized_optrdd.json").read_text(encoding="utf-8"))
DATA = pd.read_csv(FIX / "rd_optimized_data.csv")
CASES = [
    ("continuous", "continuous"),
    ("continuous_tight", "continuous"),
    ("discrete", "discrete"),
]


@pytest.mark.parametrize("key, design", CASES)
def test_same_estimator_as_optrdd(key, design):
    ref = REF[key]
    df = DATA[DATA["design"] == design].reset_index(drop=True)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = sp.rd_optimized(df, "y", "x", M=ref["M"], sigma2=ref["sigma_sq"])
    mi = r.model_info
    gamma = np.zeros(len(df))
    gamma[mi["index"]] = mi["weights"]
    gamma_r = np.asarray(ref["gamma"])
    x = df["x"].to_numpy()

    # Same estimator up to the discretisation.
    assert abs(r.estimate - ref["tau_hat"]) < 0.03 * ref["sampling_se"]
    assert mi["max_bias"] == pytest.approx(ref["max_bias"], rel=0.01)
    assert gamma @ gamma == pytest.approx(gamma_r @ gamma_r, rel=0.006)
    assert np.corrcoef(gamma, gamma_r)[0, 1] > 0.9995

    # Worst-case mean squared error at the common variance: ours is
    # within 0.3% of the reference's own figure.
    ours = mi["max_bias"] ** 2 + ref["sigma_sq"] * gamma @ gamma
    theirs = ref["max_bias"] ** 2 + ref["sigma_sq"] * gamma_r @ gamma_r
    assert ours <= theirs * 1.003

    # The moment conditions hold exactly here, and the reported bias is
    # the exact supremum for these weights.
    right = x >= 0
    assert gamma[right] @ x[right] == pytest.approx(0.0, abs=1e-10)
    assert gamma[~right] @ x[~right] == pytest.approx(0.0, abs=1e-10)
    assert worst_case_bias(x, gamma, ref["M"]) == pytest.approx(mi["max_bias"])
    # The reference's weights miss the slope condition by about 5e-5.
    assert abs(gamma_r[right] @ x[right]) > 1e-6
