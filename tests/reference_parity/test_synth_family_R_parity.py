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
