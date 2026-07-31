"""Golden-value regression lock for ``sp.xtabond``.

Purpose
-------
The dynamic-panel GMM module is being extended substantially (instrument
classes, ``collapse``, system GMM, forward orthogonal deviations, ...) and
refactored into a package.  Per the project rule "never silently change an
existing estimator's numerical output", this file pins the *current*
``sp.xtabond`` numbers for eight representative specifications so that any
refactor or feature addition that perturbs a default-path result fails
loudly instead of drifting.

The values were captured on 2026-07-31 from the implementation that is
bit-exact against Stata 18 ``xtabond`` on ``webuse abdata`` (see
``tests/reference_parity/test_xtabond_abdata_parity.py``), so they are a
lock on *validated* behaviour, not on arbitrary output.

If a change here is intentional it must be accompanied by a CHANGELOG
``⚠️ correctness`` entry and a ``MIGRATION.md`` note — update the expected
values only together with those.

**2026-07-31 revision.** The one-step rows originally pinned
``hansen = None``: the pre-v1.21 estimator only computed the
heteroskedasticity-robust Hansen J when ``twostep=True``. It is now always
reported, because the J is defined at the two-step optimum regardless of
which step is *reported* and it is the only over-identification test that
survives heteroskedasticity — ``xtabond2`` prints both Sargan and Hansen for
a one-step fit for exactly this reason. No previously-reported number
changed; a previously-``NaN`` field became informative.

Coverage of the lock
--------------------
one-step robust / one-step classical / two-step Windmeijer / two-step
conventional, ``lags=2``, a capped ``gmm_lags`` window, and the same on an
interior-gapped panel.  Each case pins the full coefficient vector, the
full SE vector, the differenced-sample size, the instrument count, both
Arellano-Bond serial-correlation z statistics, and the Sargan/Hansen
statistics — i.e. everything a user reads off the result object.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp

# --------------------------------------------------------------------------
# Deterministic DGP (identical construction to the capture script).
# --------------------------------------------------------------------------

RHO = 0.5
BETA = 1.0
SEED = 11


def _panel(seed: int = SEED, N: int = 60, T: int = 8, gap: bool = False):
    """y_it = RHO y_{i,t-1} + BETA x_it + alpha_i + e_it, burn-in 15.

    ``gap=True`` punches an *interior* hole (period 3 removed for every
    third unit), which exercises the gap-aware ``H`` matrix and the
    gapped-panel warning path.
    """
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(N):
        a = rng.normal()
        y = a / (1 - RHO) + rng.normal()
        for _ in range(15):
            x = rng.normal()
            y = RHO * y + BETA * x + a + rng.normal()
        for t in range(T):
            x = rng.normal()
            y = RHO * y + BETA * x + a + rng.normal()
            rows.append({"id": i, "time": t, "y": y, "x": x})
    df = pd.DataFrame(rows)
    if gap:
        df = df[~((df["id"] % 3 == 0) & (df["time"] == 3))]
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------
# Golden values captured 2026-07-31.
# --------------------------------------------------------------------------

GOLDEN = {
    "balanced_1step_robust": {
        "kwargs": dict(lags=1, twostep=False, robust=True),
        "gap": False,
        "coef": [0.5141995827346253, 1.101754610697651],
        "se": [0.06976943503431914, 0.07392658474422975],
        "n_obs": 360,
        "n_instruments": 22,
        "ar1_z": -4.9665375268669845,
        "ar2_z": -0.09289946292663842,
        "sargan": 8.601477807449212,
        "sargan_df": 20,
        "hansen": 13.876661606394379,
    },
    "balanced_1step_classic": {
        "kwargs": dict(lags=1, twostep=False, robust=False),
        "gap": False,
        "coef": [0.5141995827346253, 1.101754610697651],
        "se": [0.06440800663866755, 0.06310285138992316],
        "n_obs": 360,
        "n_instruments": 22,
        "ar1_z": -6.836038751090802,
        "ar2_z": -0.08556252982086893,
        "sargan": 8.601477807449212,
        "sargan_df": 20,
        "hansen": 13.876661606394379,
    },
    "balanced_2step_wc": {
        "kwargs": dict(lags=1, twostep=True, robust=True),
        "gap": False,
        "coef": [0.5459969700352048, 1.1172908942432118],
        "se": [0.06624292449276986, 0.07552771638919005],
        "n_obs": 360,
        "n_instruments": 22,
        "ar1_z": -5.22464775757775,
        "ar2_z": -0.04234896890317841,
        "sargan": 8.601477807449212,
        "sargan_df": 20,
        "hansen": 13.876661606394375,
    },
    "balanced_2step_conv": {
        "kwargs": dict(lags=1, twostep=True, robust=False),
        "gap": False,
        "coef": [0.5459969700352048, 1.1172908942432118],
        "se": [0.04930054984995701, 0.05205747802499575],
        "n_obs": 360,
        "n_instruments": 22,
        "ar1_z": -5.22464775757775,
        "ar2_z": -0.04234896890317841,
        "sargan": 8.601477807449212,
        "sargan_df": 20,
        "hansen": 13.876661606394375,
    },
    "balanced_lags2": {
        "kwargs": dict(lags=2, twostep=False, robust=True),
        "gap": False,
        "coef": [0.5688224717164603, 0.026124144143509775, 1.1080029390414994],
        "se": [0.08947700359324733, 0.05151887061624585, 0.08601216748516732],
        "n_obs": 300,
        "n_instruments": 21,
        "ar1_z": -4.507681739129073,
        "ar2_z": 0.06834095740309859,
        "sargan": 6.373452929335489,
        "sargan_df": 18,
        "hansen": 10.75669348284545,
    },
    "balanced_maxlag4": {
        "kwargs": dict(lags=1, gmm_lags=(2, 4), twostep=False, robust=True),
        "gap": False,
        "coef": [0.495890240201569, 1.093708181974179],
        "se": [0.07270981868638918, 0.0759259458818499],
        "n_obs": 360,
        "n_instruments": 16,
        "ar1_z": -4.903559180630543,
        "ar2_z": -0.1291005253234361,
        "sargan": 5.172499480954242,
        "sargan_df": 14,
        "hansen": 7.578957919382017,
    },
    "gapped_1step_robust": {
        "kwargs": dict(lags=1, twostep=False, robust=True),
        "gap": True,
        "coef": [0.5031027532962737, 1.0677343926396747],
        "se": [0.0777354940009115, 0.08127166179518786],
        "n_obs": 300,
        "n_instruments": 22,
        "ar1_z": -4.408637153609025,
        "ar2_z": 0.6202833869607753,
        "sargan": 11.159381117779452,
        "sargan_df": 20,
        "hansen": 15.492082302587583,
    },
    "gapped_2step_wc": {
        "kwargs": dict(lags=1, twostep=True, robust=True),
        "gap": True,
        "coef": [0.5424297051460742, 1.0824823248996744],
        "se": [0.08514239006731264, 0.10147058785235939],
        "n_obs": 300,
        "n_instruments": 22,
        "ar1_z": -4.625330432941938,
        "ar2_z": 0.7594177940512935,
        "sargan": 11.159381117779452,
        "sargan_df": 20,
        "hansen": 15.492082302587594,
    },
}

# Pure arithmetic on the same inputs: only BLAS-level reassociation should
# ever move these, so the band is 8 significant digits.
RTOL = 1e-8
ATOL = 1e-10


@pytest.mark.parametrize("case", sorted(GOLDEN))
def test_xtabond_golden(case):
    spec = GOLDEN[case]
    df = _panel(gap=spec["gap"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = sp.xtabond(df, y="y", x=["x"], id="id", time="time", **spec["kwargs"])

    np.testing.assert_allclose(
        r.detail["coefficient"].to_numpy(float),
        spec["coef"],
        rtol=RTOL,
        atol=ATOL,
        err_msg=f"[{case}] xtabond coefficients moved off the golden lock.",
    )
    np.testing.assert_allclose(
        r.detail["se"].to_numpy(float),
        spec["se"],
        rtol=RTOL,
        atol=ATOL,
        err_msg=f"[{case}] xtabond standard errors moved off the golden lock.",
    )

    mi = r.model_info
    assert mi["n_obs"] == spec["n_obs"], f"[{case}] differenced sample size changed."
    assert (
        mi["n_instruments"] == spec["n_instruments"]
    ), f"[{case}] instrument count changed."

    for key, want in (
        ("ar1_z", spec["ar1_z"]),
        ("ar2_z", spec["ar2_z"]),
        ("sargan_stat", spec["sargan"]),
    ):
        np.testing.assert_allclose(
            float(mi[key]),
            want,
            rtol=RTOL,
            atol=ATOL,
            err_msg=f"[{case}] {key} moved off the golden lock.",
        )
    assert mi["sargan_df"] == spec["sargan_df"], f"[{case}] Sargan df changed."

    if spec["hansen"] is None:
        assert not np.isfinite(
            mi["hansen_stat"]
        ), f"[{case}] Hansen J appeared where the one-step path reports none."
    else:
        np.testing.assert_allclose(
            float(mi["hansen_stat"]),
            spec["hansen"],
            rtol=RTOL,
            atol=ATOL,
            err_msg=f"[{case}] Hansen J moved off the golden lock.",
        )


def test_estimate_is_first_coefficient():
    """``result.estimate`` / ``result.se`` stay the lagged-Y row.

    Downstream code (``sp.panel(method='ab')``, report builders) reads the
    scalar fields; pin the contract that they are rho, not something else.
    """
    df = _panel()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = sp.xtabond(df, y="y", x=["x"], id="id", time="time")
    assert r.estimate == pytest.approx(r.detail["coefficient"].iloc[0], rel=1e-12)
    assert r.se == pytest.approx(r.detail["se"].iloc[0], rel=1e-12)
    assert r.detail["variable"].iloc[0] == "L1.y"
