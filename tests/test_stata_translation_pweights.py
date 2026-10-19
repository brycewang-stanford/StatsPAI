"""``[pweight=]`` in the Stata translator: pweights imply ``vce(robust)``.

Stata gives the same variance for ``cmd [pw=w]`` and ``cmd [pw=w],
vce(robust)`` for the linear, IV and likelihood estimators the translator
handles (``xtreg, fe``: ``vce(cluster panelvar)``). Until 2026-10-19 the
translator only knew this for commands whose sp function has a
``robust=`` argument, so ``reghdfe`` / ``ivreghdfe`` / ``ivregress`` /
``ivreg2`` / ``xtreg, fe`` / ``glm`` ran with the wrong variance:
``reghdfe y x1 x2 [pw=w], absorb(g)`` gave an x1 standard error 7% too
large, ``xtreg y x [pw=w], fe`` one 21% too small.

The reference numbers are Stata 18 MP (reghdfe 6, ivreghdfe, ivreg2),
``set type double``, on ``tests/stata_translation_holdout/holdout_cross.csv``
and ``holdout_panel.csv``::

    import delimited holdout_cross.csv, clear asdouble case(preserve)
    gen g2 = cond(_n <= 5, 100 + _n, g)        // five singleton groups
    reghdfe y x1 x2 [pw=w], absorb(g2)                     // A
    reghdfe y x1 x2 [aw=w], absorb(g2) vce(robust)         // B (= A)
    reghdfe y x1 x2 [pw=w], absorb(g2 k)                   // C
    ivreghdfe y x1 (x2 = z) [pw=w], absorb(g2)             // D
    ivreg2 y x1 (x2 = z) [pw=w]                            // F
    reghdfe y x1 x2 [pw=w], absorb(g2) keepsingletons      // G
    import delimited holdout_panel.csv, clear asdouble case(preserve)
    gen w = 0.5 + mod(id, 7) / 4
    xtset id t
    xtreg y x [pw=w], fe                                   // E

Coefficients and standard errors are printed with 15 significant digits;
the tolerance 1e-8 is far above that rounding and far below the 7% / 21%
errors this file guards against.
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import statspai as sp

HERE = Path(__file__).resolve().parent / "stata_translation_holdout"
RTOL = 1e-8

#: label -> (command, dataset, N, {name: (b, se)}) from the do-file above.
STATA = {
    "A": (
        "reghdfe y x1 x2 [pw=w], absorb(g2)",
        "cross",
        395,
        {
            "x1": (8.38413029191750e-01, 7.12550269784885e-02),
            "x2": (-5.07278522246260e-02, 6.90202590111918e-02),
        },
    ),
    "B": (
        "reghdfe y x1 x2 [aw=w], absorb(g2) vce(robust)",
        "cross",
        395,
        {
            "x1": (8.38413029191750e-01, 7.12550269784885e-02),
            "x2": (-5.07278522246260e-02, 6.90202590111918e-02),
        },
    ),
    "C": (
        "reghdfe y x1 x2 [pw=w], absorb(g2 k)",
        "cross",
        395,
        {
            "x1": (8.25360303807113e-01, 7.09253953624338e-02),
            "x2": (-5.78725195937165e-02, 6.96925416875953e-02),
        },
    ),
    "D": (
        "ivreghdfe y x1 (x2 = z) [pw=w], absorb(g2)",
        "cross",
        395,
        {
            "x1": (8.37378822779927e-01, 7.46016918831765e-02),
            "x2": (-4.55604018217265e-01, 1.22822976690504e-01),
        },
    ),
    "F": (
        "ivreg2 y x1 (x2 = z) [pw=w]",
        "cross",
        400,
        {
            "x1": (8.72493838637811e-01, 6.98149673485015e-02),
            "x2": (-4.69541071804579e-01, 1.11879188982437e-01),
        },
    ),
    "G": (
        "reghdfe y x1 x2 [pw=w], absorb(g2) keepsingletons",
        "cross",
        400,
        {
            "x1": (8.38413029191751e-01, 7.17045899680502e-02),
            "x2": (-5.07278522246262e-02, 6.94557223784397e-02),
        },
    ),
    "E": (
        "xtreg y x [pw=w], fe",
        "panel",
        360,
        {"x": (8.45284297795429e-01, 7.17633249168803e-02)},
    ),
}


@pytest.fixture(scope="module")
def frames():
    cross = pd.read_csv(HERE / "holdout_cross.csv")
    cross["g2"] = np.where(
        np.arange(len(cross)) < 5, 101 + np.arange(len(cross)), cross["g"]
    )
    panel = pd.read_csv(HERE / "holdout_panel.csv")
    panel["w"] = 0.5 + (panel["id"] % 7) / 4
    return {"cross": cross, "panel": panel}


def _nobs(result) -> int:
    for attr in ("nobs", "n_obs"):
        value = getattr(result, attr, None)
        if value is not None:
            return int(value)
    return int(result.data_info["nobs"])


@pytest.mark.parametrize("label", sorted(STATA))
def test_pweighted_fit_reproduces_stata(frames, label):
    pytest.importorskip("pyfixest")
    command, dataset, n, ref = STATA[label]
    script = ("xtset id t\n" if dataset == "panel" else "") + command
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = sp.stata(script, data=frames[dataset])
    assert _nobs(result) == n
    params, ses = dict(result.params), dict(result.std_errors)
    for name, (b, se) in ref.items():
        assert np.isclose(params[name], b, rtol=RTOL, atol=0), (label, name)
        assert np.isclose(ses[name], se, rtol=RTOL, atol=0), (label, name)


@pytest.mark.parametrize(
    "line, tool, expected",
    [
        (
            "reghdfe y x [pw=w], absorb(id)",
            "feols",
            {"vcov": "hetero", "fixef_rm": "singleton", "weights": "w"},
        ),
        (
            "reghdfe y x [pw=w], absorb(id) keepsingletons",
            "feols",
            {"vcov": "hetero", "fixef_rm": "none"},
        ),
        # a cluster is honoured by sp.hdfe_ols with weights: no detour
        (
            "reghdfe y x [pw=w], absorb(id) vce(cluster id)",
            "hdfe_ols",
            {"cluster": "id", "weights": "w"},
        ),
        ("ivregress 2sls y (x = z) [pw=w]", "iv", {"robust": "hc1", "small": False}),
        ("glm y x [pw=w], family(poisson)", "glm", {"robust": "robust"}),
        ("logit y x [pw=w]", "logit", {"robust": "robust"}),
        ("regress y x [pw=w]", "regress", {"robust": "hc1"}),
        ("xtreg y x [pw=w], fe i(id)", "feols", {"cluster": "id"}),
        # aweights do not imply a robust variance
        ("reghdfe y x [aw=w], absorb(id)", "hdfe_ols", {"weights": "w"}),
    ],
)
def test_translation_of_pweights(line, tool, expected):
    out = sp.from_stata(line)
    assert out["ok"], out
    assert out["tool"] == tool
    for key, value in expected.items():
        assert out["arguments"].get(key) == value, (key, out["arguments"])
    assert not out["untranslated_options"]
    # python_code names every argument the payload carries
    for key, value in out["arguments"].items():
        if key not in ("formula", "fml"):
            assert f"{key}={value!r}" in out["python_code"], (key, out["python_code"])


def test_aweighted_reghdfe_keeps_the_default_variance():
    out = sp.from_stata("reghdfe y x [aw=w], absorb(id)")
    assert "vce" not in out["arguments"] and "vcov" not in out["arguments"]


def test_qreg_python_code_carries_the_weights():
    """``sp.qreg`` takes the data positionally; the weights must still be in
    the printed call (they were in ``arguments`` only)."""
    out = sp.from_stata("qreg y x1 x2 [pw=w]")
    assert out["arguments"]["weights"] == "w"
    assert "weights='w'" in out["python_code"]


@pytest.mark.parametrize("line", ["newey y x [pw=w], lag(2)", "rdrobust y x [pw=w]"])
def test_pweights_stata_rejects_are_refused(line):
    out = sp.from_stata(line)
    assert out["ok"] is False
    assert "r(101)" in out["error"]


def test_weighted_robust_with_absorbed_slope_is_refused():
    out = sp.from_stata("reghdfe y x [pw=w], absorb(id##c.t)")
    assert out["ok"] is False
    assert "slope" in out["error"]
