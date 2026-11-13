"""``sp.zero_first_stage`` against its component regressions in Stata.

The zero-first-stage estimator (van Kippersluis and Rietveld 2018) has no
command of its own. It is a combination of four least-squares fits and one
IV fit, so the reference runs those in Stata 18 MP and combines them in
the do-file (``_fixtures/_generate_zero_first_stage_Stata.do`` on
``zero_first_stage.csv``, 900 main and 600 zero-first-stage rows):

* first stage and reduced form in the zero-first-stage sample
  (coefficient and standard error of the instrument);
* first stage and reduced form in the main sample;
* 2SLS in the main sample (coefficient and standard error);
* the corrected estimate ``(rho_main - gamma_zfs) / pi_main``.

Three designs: ``regress`` / ``ivregress 2sls`` with ``vce(robust)``, the
same with ``vce(cluster g)``, and ``reghdfe`` / ``ivreghdfe`` absorbing
``f`` with clustering on ``g`` (reghdfe 6.13.1, ivreghdfe 1.1.4).

Tolerance 1e-9 relative on every number (observed <= 1e-14). The
bootstrap standard error of the corrected estimate has no Stata
counterpart and is not compared.

Through 1.43.1 ``beta_iv_se`` was ``|beta_iv| * se(pi_main) / |pi_main|``,
which leaves out the sampling error of the reduced form: 0.0738 on the
robust design here against 0.0485 from ``ivregress``.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd
import pytest

import statspai as sp

_FIX = pathlib.Path(__file__).parent / "_fixtures"
STATA = json.loads((_FIX / "zero_first_stage_Stata.json").read_text(encoding="utf-8"))
DATA = pd.read_csv(_FIX / "zero_first_stage.csv")
DESIGNS = {
    "robust": {},
    "cluster": {"cluster": "g"},
    "absorb": {"cluster": "g", "absorb": "f"},
}
REL = 1e-9


def _fit(design, **kw):
    return sp.zero_first_stage(
        DATA,
        y="y",
        endog="d",
        instrument="z",
        zfs="zfs",
        exog="x",
        **{"n_boot": 0, **DESIGNS[design], **kw},
    )


def test_fixture_has_a_violation_to_correct():
    ref = STATA["robust"]
    # the reduced form in the zero-first-stage sample is far from zero, so
    # the corrected estimate is far from the IV estimate
    assert ref["gamma"] / ref["gamma_se"] > 4
    assert ref["beta_iv"] - ref["corrected"] > 0.4
    assert STATA["versions"]["ivreghdfe"] == "1.1.4"


@pytest.mark.parametrize("design", sorted(DESIGNS))
def test_components_and_corrected_estimate_match_stata(design):
    res = _fit(design)
    ref = STATA[design]
    got = {
        "pi_zfs": res.first_stage_zfs,
        "pi_zfs_se": res.first_stage_zfs_se,
        "gamma": res.reduced_form_zfs,
        "gamma_se": res.reduced_form_zfs_se,
        "pi_main": res.first_stage_main,
        "pi_main_se": res.first_stage_main_se,
        "rho": res.diagnostics["reduced_form_main"],
        "beta_iv": res.beta_iv,
        "beta_iv_se": res.beta_iv_se,
        "corrected": res.beta_zfs_corrected,
    }
    for key, value in got.items():
        assert value == pytest.approx(ref[key], rel=REL), key
    assert res.implied_bias == pytest.approx(ref["gamma"] / ref["pi_main"], rel=REL)
    assert (res.n_main, res.n_zfs) == (900, 600)


def test_iv_standard_error_is_the_sampling_error_of_the_iv_estimate():
    # 300 draws of a just-identified design: the reported standard error
    # averages to the standard deviation of the estimate. The pre-fix
    # formula averaged 0.029 against a standard deviation of 0.064.
    est, ses = [], []
    for seed in range(300):
        rng = np.random.default_rng(seed)
        n = 1000
        desert = rng.integers(0, 2, size=n).astype(bool)
        z = rng.normal(size=n)
        u = rng.normal(size=n)
        d = np.where(desert, 0.0, 0.9 * z) + 0.6 * u + rng.normal(size=n)
        y = -0.5 * d + 0.7 * u + rng.normal(size=n)
        res = sp.zero_first_stage(
            pd.DataFrame({"y": y, "d": d, "z": z, "desert": desert}),
            y="y",
            endog="d",
            instrument="z",
            zfs="desert",
            n_boot=0,
        )
        est.append(res.beta_iv)
        ses.append(res.beta_iv_se)
    # ratio of mean SE to Monte Carlo SD; its own sampling error at 300
    # draws is about 4 percent
    assert np.mean(ses) / np.std(est, ddof=1) == pytest.approx(1.0, abs=0.12)
