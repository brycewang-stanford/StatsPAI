"""R parity for the wave-2 synthetic-control extensions (``synth_family``).

References are frozen in ``_fixtures/synth_family_R.json``, written by
``_generate_synth_family_R.R`` from the CSVs of
``_generate_synth_family_data.py``. Package versions are in the fixture's
``meta``.

Conventions each number depends on
----------------------------------
* ``augsynth``: ``augsynth::augsynth`` 0.2.0 with its simplex QP re-run at
  OSQP eps 1e-12 (stock eps is 1e-8). Several treated units with one
  treatment date are averaged into one treated trajectory
  (``single_augsynth``); covariates use augsynth's default
  ``residualize = FALSE`` (pre-period means, centred at the donor mean,
  rescaled to ``sd(X_c)``, appended to the outcome columns); the CV
  holdouts run over every column of that augmented matrix, covariate
  columns included, as augsynth's ``get_lambda_errors`` does.

* ``fect`` tests: ``fect::fect`` 2.4.1, ``method = "fe"``,
  ``vartype = "jackknife"`` (deterministic), ``tol = 1e-12``. Placebo
  (``placebo.period = c(-2, 0)``), carryover (``carryover.period = c(1, 2)``)
  and leave-one-period-out; ``test.out`` / ``loo.test.out`` are
  ``fect:::diagtest``. The one documented difference: diagtest's F
  statistic uses ``cov()`` of the raw jackknife replicates, StatsPAI the
  jackknife covariance, ``(n - 1)^2 / n`` times larger; the test pins
  ``F_fect = F_StatsPAI * (n - 1)^2 / n`` and recomputes fect's p-values
  from fect's F.

Tolerance (fect): 1e-8 relative. Both sides stop the EM fit at relative
change 1e-12; observed differences are ~1e-11 on the ATT and below 1e-9 on
SEs and p-values.

Tolerance (augsynth): 1e-8 relative on the ATT, 1e-9 absolute on weights
and 2e-9 on the effect path. Observed 1e-15 to 2e-13 everywhere except
``one_fixedeff_cv``, where CV picks lambda = 1.3e-5: the ridge step solves
with ``X_c'X_c + lambda I`` nearly singular and the 1e-11 OSQP error of the
simplex weights grows to 1.1e-10 (ATT, relative) and 5e-10 (path).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import statspai as sp

FX = Path(__file__).parent / "_fixtures"
R = json.loads((FX / "synth_family_R.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def asc():
    return pd.read_csv(FX / "synth_family_asc.csv")


_ASC_ARGS = {
    "two_ridge_fixed": dict(treated=[1, 2], progfunc="ridge", ridge_lambda=0.5),
    "two_ridge_cv": dict(treated=[1, 2], progfunc="ridge"),
    "two_none": dict(treated=[1, 2], progfunc="none"),
    "one_fixedeff_cv": dict(treated=1, progfunc="ridge", fixedeff=True),
    "one_noscm_cv": dict(treated=1, progfunc="ridge", scm=False),
    "one_cov_fixed": dict(treated=1, progfunc="ridge", ridge_lambda=2.0, cov=True),
    "two_cov_cv": dict(treated=[1, 2], progfunc="ridge", cov=True),
    "two_cov_none_fe": dict(treated=[1, 2], progfunc="none", cov=True, fixedeff=True),
}


@pytest.mark.parametrize("case", R["augsynth"], ids=[c["name"] for c in R["augsynth"]])
def test_augsynth_matches_r(asc, case):
    a = dict(_ASC_ARGS[case["name"]])
    treated = a.pop("treated")
    cov = a.pop("cov", False)
    df = asc if isinstance(treated, list) else asc[asc.unit != 2]
    r = sp.augsynth(
        df,
        outcome="y",
        unit="unit",
        time="time",
        treated_unit=treated,
        treatment_time=14,
        covariates=["x1", "x2"] if cov else None,
        placebo=False,
        **a,
    )
    np.testing.assert_allclose(r.estimate, case["att"], rtol=1e-8)
    w = r.model_info["weights_dict"]
    np.testing.assert_allclose(
        [w[u] for u in case["donors"]], case["weights"], rtol=0, atol=1e-9
    )
    if case["lambda"] is None:
        assert r.model_info["ridge_lambda"] is None
    else:
        np.testing.assert_allclose(
            r.model_info["ridge_lambda"], case["lambda"], rtol=1e-9
        )
    # full effect path (pre and post), augsynth's predict(att = TRUE)
    gap = r.model_info["gap_table"]["gap"].to_numpy()
    np.testing.assert_allclose(gap, case["att_path"], rtol=0, atol=2e-9)


# --------------------------------------------------------------------------
# fect placebo / carryover / leave-one-period-out tests vs fect::fect
# --------------------------------------------------------------------------
FT = R["fect"]
FECT_KW = dict(
    y="Y",
    treat="D",
    unit="id",
    time="time",
    method="fe",
    tol=1e-12,
    max_iter=20000,
    vce="jackknife",
)
RT_FECT = 1e-8


@pytest.fixture(scope="module")
def fect_df():
    return pd.read_csv(FX / "synth_family_fect.csv")


@pytest.mark.parametrize(
    "kind,arg",
    [
        ("placebo", {"placebo_period": (-2, 0)}),
        ("carryover", {"carryover_period": (1, 2)}),
    ],
)
def test_fect_window_tests_match_r(fect_df, kind, arg):
    ref = FT[kind]
    r = sp.fect(fect_df, **FECT_KW, **arg)
    np.testing.assert_allclose(r.estimate, ref["att_avg"], rtol=RT_FECT)
    got = r.model_info[f"{kind}_test"]
    for key in ("estimate", "se", "pvalue"):
        np.testing.assert_allclose(got[key], ref["test"][key], rtol=RT_FECT)
    np.testing.assert_allclose(
        got["ci"], [ref["test"]["ci_lower"], ref["test"]["ci_upper"]], rtol=RT_FECT
    )
    np.testing.assert_allclose(
        got["equiv_pvalue"], ref["test_out"][f"{kind}.equiv.p"], rtol=RT_FECT
    )
    np.testing.assert_allclose(
        got["tost_threshold"], 0.36 * np.sqrt(ref["sigma2_fect"]), rtol=RT_FECT
    )


@pytest.fixture(scope="module")
def fect_loo(fect_df):
    return sp.fect(fect_df, **FECT_KW, loo=True)


def test_fect_loo_table_matches_r(fect_loo):
    ref = FT["loo"]
    np.testing.assert_allclose(fect_loo.estimate, ref["att_avg"], rtol=RT_FECT)
    tab = fect_loo.model_info["loo"]["table"]
    np.testing.assert_array_equal(tab["period"], ref["pre_period"])
    np.testing.assert_allclose(tab["att"], ref["pre_att"], rtol=RT_FECT)
    np.testing.assert_allclose(tab["se"], ref["pre_se"], rtol=RT_FECT)
    np.testing.assert_allclose(tab["pvalue"], ref["pre_p"], rtol=RT_FECT)
    np.testing.assert_array_equal(tab["count"], ref["pre_count"])
    # in-sample event study and its jackknife SEs
    np.testing.assert_allclose(fect_loo.detail["att"], ref["att"], rtol=RT_FECT)
    np.testing.assert_allclose(fect_loo.detail["se"], ref["se"], rtol=RT_FECT)


@pytest.mark.parametrize("which", ["pretrend", "loo"])
def test_fect_pretrend_tests_match_r(fect_loo, which):
    ref = FT["loo"]["test_out" if which == "pretrend" else "loo_test_out"]
    mi = fect_loo.model_info
    got = mi["pretrend_test"] if which == "pretrend" else mi["loo"]["test"]
    n = FT["loo"]["n_units"]  # jackknife replicates (every unit's fit succeeds)
    assert got["df1"] == ref["df1"] and got["df2"] == ref["df2"]
    assert got["N_bar"] == ref["N_bar"]
    np.testing.assert_allclose(
        got["tost_threshold"], ref["tost.threshold"], rtol=RT_FECT
    )
    np.testing.assert_allclose(got["tost_equiv_p"], ref["tost.equiv.p"], rtol=RT_FECT)
    # documented departure: fect's F uses cov() of the raw jackknife replicates
    np.testing.assert_allclose(
        got["f_stat"] * (n - 1) ** 2 / n, ref["f.stat"], rtol=RT_FECT
    )
    from scipy import stats

    k, d2 = ref["df1"], ref["df2"]
    np.testing.assert_allclose(stats.f.sf(ref["f.stat"], k, d2), ref["f.p"], rtol=1e-6)
    np.testing.assert_allclose(
        stats.ncf.cdf(ref["f.stat"], k, d2, ref["N_bar"] * ref["f.threshold"]),
        ref["f.equiv.p"],
        rtol=1e-6,
    )
