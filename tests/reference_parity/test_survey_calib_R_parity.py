"""sp.rake / sp.linear_calibration vs R survey and Stata svycal.

References, both read ``_fixtures/survey_calib_data.csv`` (n = 200, 4 strata
x 5 PSUs, design weights d, sex x agegrp categories, continuous income and
age):

* ``_fixtures/survey_calib_R.json`` -- R ``survey::rake`` (run to its fixed
  point), ``survey::calibrate(calfun = "raking")`` and
  ``calibrate(calfun = "linear")``, written by ``_generate_survey_calib_R.R``
  (versions in its ``provenance``).
* ``_fixtures/survey_calib_stata.json`` -- Stata 18 ``svycal rake`` /
  ``svycal regress``, written by ``_fixtures/_generate_survey_calib_stata.do``.

Conventions each number depends on
----------------------------------
* ``sp.rake`` returns weights that sum to 1; R ``rake`` / ``calibrate`` and
  Stata ``svycal`` return weights on the population scale (sum = N = 10000
  here).  Raking weights are compared after dividing the reference by its
  sum; the margins are passed as the same population counts.
* Raking: IPF (ours, R ``rake``) and Newton on the multiplicative distance
  (R ``calibrate(calfun = "raking")``, Stata ``svycal rake``) share one
  fixed point; R's two routes agree to 2e-15.  ``sp.rake`` stops when every
  category share is within ``tol`` (relative) of its target: default
  ``tol = 1e-10`` lands within ~1e-10 of the fixed point, ``tol = 1e-14``
  within ~1e-15.
* Linear: unbounded chi-squared distance, closed form; no intercept is
  added (a column of ones / a dummy column carry the N and category
  totals).  Exact, compared unscaled.
* Variance: ``sp.rake`` / ``sp.linear_calibration`` return weights only.
  Fed to ``sp.svydesign`` they are treated as fixed, which reproduces R /
  Stata svymean on a plain design with those weights -- NOT the
  calibration-adjusted linearisation SE of a calibrated design (R
  ``calibrate`` / Stata ``svyset, rake()``), which here is 7-18 % smaller.
  The gap is pinned below as a documented difference.

Tolerance: 1e-12 relative on converged / closed-form weights and on the
fixed-weights SE (observed ~1e-15); 1e-8 on default-``tol`` raking weights.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import statspai as sp

_FIX = Path(__file__).parent / "_fixtures"
R = json.loads((_FIX / "survey_calib_R.json").read_text(encoding="utf-8"))
S = json.loads((_FIX / "survey_calib_stata.json").read_text(encoding="utf-8"))
T = R["targets"]
MARGINS = {"sex": T["sex"], "agegrp": T["agegrp"]}


@pytest.fixture(scope="module")
def df():
    return pd.read_csv(_FIX / "survey_calib_data.csv")


def _share(w):
    w = np.asarray(w, dtype=float)
    return w / w.sum()


def _close(ours, ref, rtol):
    np.testing.assert_allclose(
        np.asarray(ours, dtype=float), np.asarray(ref, dtype=float), rtol=rtol, atol=0
    )


# --------------------------------------------------------------------------
# raking
# --------------------------------------------------------------------------


def test_R_rake_and_calibrate_raking_share_the_fixed_point():
    _close(R["calib_raking"]["weights"], R["rake_tight"]["weights"], rtol=1e-12)


def test_rake_matches_R_rake_fixed_point(df):
    ref = _share(R["rake_tight"]["weights"])
    tight = sp.rake(df, MARGINS, weight="d", tol=1e-14)
    assert tight.converged
    _close(tight.calibrated_weights, ref, rtol=1e-12)
    default = sp.rake(df, MARGINS, weight="d")
    assert default.converged
    _close(default.calibrated_weights, ref, rtol=1e-8)


def test_rake_matches_R_calibrate_raking_and_stata_svycal(df):
    w = sp.rake(df, MARGINS, weight="d", tol=1e-14).calibrated_weights
    _close(w, _share(R["calib_raking"]["weights"]), rtol=1e-12)
    _close(w, _share(S["w_rake"]), rtol=1e-12)


def test_rake_equal_start_matches_R(df):
    w = sp.rake(df, MARGINS, tol=1e-14).calibrated_weights
    _close(w, _share(R["rake_equal_start"]["weights"]), rtol=1e-12)


def test_rake_proportion_and_count_margins_agree(df):
    props = {c: {k: v / T["N"] for k, v in m.items()} for c, m in MARGINS.items()}
    a = sp.rake(df, MARGINS, weight="d", tol=1e-14).calibrated_weights
    b = sp.rake(df, props, weight="d", tol=1e-14).calibrated_weights
    _close(a, b, rtol=1e-13)


def test_rake_hits_margins_regardless_of_sample_size(df):
    """Regression: the old absolute-change criterion stopped early at large n.

    With the data replicated 500x (n = 100 000) the 1.28.0 loop declared
    convergence after two sweeps with a 4.3e-4 relative margin error.
    """
    big = pd.concat([df] * 500, ignore_index=True)
    res = sp.rake(big, MARGINS, weight="d")
    w = res.calibrated_weights
    for col, targets in MARGINS.items():
        for cat, count in targets.items():
            share = w[big[col].to_numpy() == cat].sum()
            assert abs(share - count / T["N"]) / (count / T["N"]) < 1e-10
    # replication leaves the fixed point unchanged (up to the 1/500 scale)
    ref = _share(R["rake_tight"]["weights"])
    _close(w[: len(df)] * 500, ref, rtol=1e-8)


def test_R_rake_default_control_stops_short():
    """R's own rake(control = default) is ~2e-7 from its fixed point here."""
    gap = np.max(
        np.abs(np.asarray(R["rake_default"]["weights"]) - R["rake_tight"]["weights"])
        / np.asarray(R["rake_tight"]["weights"])
    )
    assert 1e-8 < gap < 1e-6


def test_rake_input_validation(df):
    with pytest.raises(ValueError, match="have no target"):
        sp.rake(df, {"agegrp": {"a": 0.5, "b": 0.5}}, weight="d")
    with pytest.raises(ValueError, match="do not occur"):
        sp.rake(df, {"sex": {"F": 0.4, "M": 0.5, "X": 0.1}}, weight="d")
    with pytest.warns(UserWarning, match="different totals"):
        sp.rake(df, {"sex": T["sex"], "agegrp": {"a": 1, "b": 1, "c": 1}})
    with pytest.warns(RuntimeWarning, match="did not converge"):
        res = sp.rake(df, MARGINS, weight="d", max_iter=1)
    assert not res.converged


# --------------------------------------------------------------------------
# linear calibration
# --------------------------------------------------------------------------


def test_linear_calibration_matches_R_and_stata_no_intercept(df):
    res = sp.linear_calibration(
        df, {"income": T["income"], "age": T["age"]}, weight="d"
    )
    _close(res.calibrated_weights, R["linear_noint"]["weights"], rtol=1e-12)
    _close(res.calibrated_weights, S["w_lin_noint"], rtol=1e-12)
    # reference-free: the calibration equations hold
    w = res.calibrated_weights
    _close([w @ df["income"], w @ df["age"]], [T["income"], T["age"]], rtol=1e-12)


def test_linear_calibration_matches_R_and_stata_intercept_and_dummy(df):
    totals = {"one": T["N"], "male": T["sex"]["M"], "income": T["income"]}
    w = sp.linear_calibration(df, totals, weight="d").calibrated_weights
    _close(w, R["linear_int_factor"]["weights"], rtol=1e-12)
    _close(w, S["w_lin_int"], rtol=1e-12)


def test_linear_calibration_is_the_chi_square_projection(df):
    """Reference-free: w - d lies in span{d * x} (the stationarity condition)."""
    X = df[["income", "age"]].to_numpy()
    d = df["d"].to_numpy()
    w = sp.linear_calibration(
        df, {"income": T["income"], "age": T["age"]}, weight="d"
    ).calibrated_weights
    g_minus_1 = w / d - 1
    coef, *_ = np.linalg.lstsq(X, g_minus_1, rcond=None)
    _close(X @ coef, g_minus_1, rtol=1e-10)


def test_linear_calibration_collinear_warns(df):
    with pytest.warns(UserWarning, match="collinear"):
        sp.linear_calibration(
            df.assign(inc2=2 * df["income"]),
            {"income": T["income"], "inc2": 2 * T["income"]},
            weight="d",
        )


# --------------------------------------------------------------------------
# variance with calibrated weights: documented difference
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "key, fn",
    [
        ("rake_tight", "rake"),
        ("linear_noint", "linear"),
    ],
)
def test_calibrated_weights_in_svydesign_give_fixed_weight_se(df, key, fn):
    if fn == "rake":
        w = sp.rake(df, MARGINS, weight="d", tol=1e-14).calibrated_weights * T["N"]
    else:
        w = sp.linear_calibration(
            df, {"income": T["income"], "age": T["age"]}, weight="d"
        ).calibrated_weights
    d = sp.svydesign(
        df.assign(wc=w), weights="wc", strata="stratum", cluster="psu", nest=True
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        m = d.mean("y")
    ref = R[key]
    _close(m.estimate, [ref["mean_y"]], rtol=1e-12)
    _close(m.std_error, [ref["se_fixed_weights"]], rtol=1e-12)
    if key == "rake_tight":
        _close(m.std_error, [S["rake_se_fixed_weights"]], rtol=1e-12)
    # NOT the calibration-adjusted SE of R calibrate / Stata svyset rake()
    assert m.std_error.iloc[0] > 1.05 * ref["se_calibrated"]
