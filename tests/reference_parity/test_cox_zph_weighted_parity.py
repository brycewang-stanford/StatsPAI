"""``CoxResult.ph_test`` on weighted and counting-process fits vs R.

Reference: ``_fixtures/cox_zph_weighted_R.json`` written by
``_generate_cox_zph_weighted_R.R`` (survival 3.8-3, ``cox.zph`` score
test; LGPL, used as a black box: documentation and outputs only) on the
fixtures of the weighted ``sp.cox`` parity (``survival_weighted_data.csv``,
``survival_counting_data.csv``).

Conventions read off ``cox.zph``'s outputs (``$x``, ``$table``):

* case weights enter the score and the information exactly as they enter
  the weighted partial likelihood (each death scores ``w_i (x_i - xbar)``,
  Efron terms carry the mean death weight); the variance is the
  model-based information whatever ``robust=`` / ``cluster=`` the fit used
  (the robust and model-based fits give the same table);
* the time transform ignores the weights: ``km`` is one minus the
  *unweighted* left-continuous Kaplan-Meier, of the ``(start, stop]`` data
  when there is a start; ``rank`` ranks the stop times of all rows.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import MethodIncompatibility

_FIX = Path(__file__).parent / "_fixtures"
R = json.loads((_FIX / "cox_zph_weighted_R.json").read_text(encoding="utf-8"))
TAGS = [k for k in R if k != "version"]


@pytest.fixture(scope="module")
def frames():
    return (
        pd.read_csv(_FIX / "survival_weighted_data.csv"),
        pd.read_csv(_FIX / "survival_counting_data.csv"),
    )


def _fit(tag, df, cp):
    ties = tag.rsplit("_", 1)[1]
    one = dict(data=df, duration="time", event="event", x=["x1", "x2", "a"])
    cpx = dict(data=cp, duration="tstop", event="event", start="tstart")
    if tag.startswith("w_model"):
        return sp.cox(**one, weights="w", ties=ties)
    if tag.startswith("w_robust"):
        return sp.cox(**one, weights="w", ties=ties, robust="hc0")
    if tag.startswith("entry_w"):
        return sp.cox(**one, weights="w", start="entry", ties=ties, cluster="id")
    if tag.startswith("entry"):
        return sp.cox(**one, start="entry", ties=ties)
    if tag.startswith("cp_id"):
        return sp.cox(**cpx, x=["z", "x2", "a"], ties=ties, cluster="id")
    if tag.startswith("cp_strata"):
        return sp.cox(**cpx, x=["z", "a"], strata="x2", ties=ties)
    assert tag.startswith("cp_w")
    return sp.cox(**cpx, x=["z", "x2", "a"], weights="w", ties=ties, cluster="id")


@pytest.mark.parametrize("tag", TAGS)
def test_ph_test_matches_cox_zph(frames, tag):
    ref = R[tag]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = _fit(tag, *frames)
    np.testing.assert_allclose(fit.params.values, ref["coef"], rtol=1e-10)
    for transform in ("km", "identity", "rank", "log"):
        out = fit.ph_test(transform=transform)
        exp = ref[transform]
        assert list(out["df"]) == [int(v) for v in exp["df"]]
        # Tolerance: the closed form agrees to ~3e-8 relative although the
        # coefficients agree to 1e-15; cox.zph appears to take the
        # information from the fit's last iteration (an output-level
        # reading, consistent with the 1e-5 budget of the unweighted test).
        np.testing.assert_allclose(
            out["chi2"], exp["chisq"], rtol=1e-6, err_msg=f"{tag} {transform}"
        )
        np.testing.assert_allclose(out["p_value"], exp["p"], rtol=1e-5, atol=1e-12)


def test_frequency_weights_equal_replicated_rows(frames):
    """Integer case weights are frequencies: under the Breslow tie rule the
    weighted test equals the test on the data with each row repeated ``w``
    times (an identity, independent of R; Efron is not invariant to
    replication because its ``l/d`` fractions count deaths, and the ``km``
    / ``rank`` transforms ignore weights, so ``identity`` is used)."""
    df, _ = frames
    rng = np.random.default_rng(3)
    d = df.assign(fw=rng.integers(1, 4, len(df)))
    rep = d.loc[d.index.repeat(d["fw"])].reset_index(drop=True)
    kw = dict(duration="time", event="event", x=["x1", "x2", "a"])
    a = sp.cox(data=d, weights="fw", ties="breslow", **kw)
    b = sp.cox(data=rep, ties="breslow", **kw)
    np.testing.assert_allclose(
        a.ph_test(transform="identity")["chi2"],
        b.ph_test(transform="identity")["chi2"],
        rtol=1e-8,
    )


def test_approx_and_design_refused(frames):
    df, cp = frames
    fit = sp.cox(data=cp, duration="tstop", event="event", start="tstart", x=["z"])
    with pytest.raises(MethodIncompatibility, match="approx"):
        fit.ph_test(method="approx")
    des = sp.svydesign(df, weights="w")
    dfit = sp.cox(data=df, duration="time", event="event", x=["x1"], design=des)
    with pytest.raises(MethodIncompatibility, match="design"):
        dfit.ph_test()
