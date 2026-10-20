"""``stteffects`` through ``sp.from_stata`` / ``sp.stata``, against Stata 18.

Executed equivalence: each Stata line runs through ``sp.stata`` after
``stset time, failure(event)`` on ``reference_parity/_fixtures/
stteffects_data.csv`` and its ATE / ATET, the control POmean and their
standard errors are compared with what Stata 18 printed for the same line.

* The twelve lines of ``reference_parity/_generate_stteffects_stata.do``
  (frozen in ``stteffects_Stata.csv``): estimates 1e-9, SEs 1e-7 relative,
  the tolerances of ``test_survival_causal_parity.py`` (our Jacobian is a
  central difference, Stata's analytic).
* Four more lines run in Stata 18 MP on 2026-10-20 (constant-only
  censoring model ``(, weibull)``, a treatment equation without covariates,
  an exponential censoring model with ``wra``, log-normal outcome and
  censoring with a probit treatment model, and ``pomeans``); Stata prints
  seven significant digits, so those compare at 5e-7 relative.

``stteffects ipwra, atet`` translates with ``atet`` flagged (``complete``
is False) and ``sp.stata`` refuses it: Stata averages over all rows, the
documented departure in ``docs/dev/modules/survival.md``.
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import MethodIncompatibility

FX = pathlib.Path(__file__).parent / "reference_parity" / "_fixtures"
STSET = "stset time, failure(event)\n"

FROZEN = {
    "ra_ate_weibull": "stteffects ra (x1 x2) (a)",
    "ra_atet_weibull": "stteffects ra (x1 x2) (a), atet",
    "ra_ate_exponential": "stteffects ra (x1 x2, exponential) (a)",
    "ra_ate_lognormal": "stteffects ra (x1 x2, lnormal) (a)",
    "wra_ate_weibull": "stteffects wra (x1 x2) (a) (x1)",
    "ipw_ate_weibull": "stteffects ipw (a x1 x2) (x1)",
    "ipw_atet_weibull": "stteffects ipw (a x1 x2) (x1), atet",
    "ipw_ate_probit_lognormal": "stteffects ipw (a x1 x2, probit) (x1, lnormal)",
    "ipwra_lac_ate_weibull": "stteffects ipwra (x1 x2) (a x1 x2)",
    "ipwra_wac_ate_weibull": "stteffects ipwra (x1 x2) (a x1 x2) (x1)",
}

# (line, effect, se, POmean0, se) as Stata 18 printed them (%9.0g / 7 digits)
PRINTED = [
    ("stteffects ipw (a x1 x2) (, weibull)", 1.795017, 0.729426, 2.703249, 0.1670795),
    ("stteffects ipw (a) (x1)", 1.666629, 0.4787391, 2.513938, 0.1288799),
    (
        "stteffects wra (x1 x2) (a) (x1, exponential)",
        0.706832,
        0.2220315,
        2.491951,
        0.1039553,
    ),
    (
        "stteffects ipwra (x1 x2, lnormal) (a x1 x2, probit) (x1, lnormal)",
        1.102161,
        0.3561594,
        2.86253,
        0.1366437,
    ),
    (
        "stteffects ra (x1 x2) (a), vce(robust)",
        1.782708,
        0.3289986,
        2.829005,
        0.1234335,
    ),
]


@pytest.fixture(scope="module")
def data():
    return pd.read_csv(FX / "stteffects_data.csv")


@pytest.fixture(scope="module")
def ref():
    return pd.read_csv(FX / "stteffects_Stata.csv").set_index("tag")


def _effect_and_pomean0(res):
    det = res.detail.set_index("term")
    return (
        res.estimate,
        res.se,
        det.loc["POmean0", "estimate"],
        det.loc["POmean0", "se"],
    )


@pytest.mark.parametrize("tag", sorted(FROZEN))
def test_frozen_stata_lines_run_to_statas_numbers(tag, data, ref):
    res = sp.stata(STSET + FROZEN[tag], data=data)
    b, se, b0, se0 = _effect_and_pomean0(res)
    r = ref.loc[tag]
    np.testing.assert_allclose([b, b0], [r.b1, r.b2], rtol=1e-9)
    np.testing.assert_allclose([se, se0], [r.se1, r.se2], rtol=1e-7)
    assert res.n_obs == int(r.N)


@pytest.mark.parametrize("line, b, se, b0, se0", PRINTED)
def test_further_lines_match_statas_printed_output(line, b, se, b0, se0, data):
    res = sp.stata(STSET + line, data=data)
    got = _effect_and_pomean0(res)
    np.testing.assert_allclose(got, [b, se, b0, se0], rtol=5e-7)


def test_pomeans_is_the_same_fit_with_both_means_in_detail(data):
    res = sp.stata(STSET + "stteffects ra (x1 x2) (a), pomeans", data=data)
    det = res.detail.set_index("term")
    # Stata 18: POmeans 0 = 2.829005 (.1234335), 1 = 4.611714 (.3127317)
    np.testing.assert_allclose(
        det.loc[["POmean0", "POmean1"], ["estimate", "se"]].to_numpy(),
        [[2.829005, 0.1234335], [4.611714, 0.3127317]],
        rtol=5e-7,
    )
    tr = sp.from_stata("stteffects ra (x1 x2) (a), pomeans")
    assert tr["complete"] and any("POmean1" in s for s in tr["semantics"])


def test_level_carries_to_alpha(data):
    res = sp.stata(STSET + "stteffects ra (x1 x2) (a), level(90)", data=data)
    # Stata's 90% interval is b -/+ invnormal(.95) * se
    z = 1.6448536269514722
    np.testing.assert_allclose(
        res.ci, [res.estimate - z * res.se, res.estimate + z * res.se]
    )


def test_ipwra_atet_is_flagged_and_refused_by_sp_stata(data):
    tr = sp.from_stata("stteffects ipwra (x1 x2) (a x1 x2), atet")
    assert tr["ok"] and tr["dropped"] == ["atet"] and tr["complete"] is False
    with pytest.raises(MethodIncompatibility, match="atet"):
        sp.stata(STSET + "stteffects ipwra (x1 x2) (a x1 x2), atet", data=data)


@pytest.mark.parametrize(
    "line, reason",
    [
        ("stteffects ra (x1 x2, gamma) (a)", "gamma"),
        ("stteffects ra (x1 x2, noconstant) (a)", "noconstant"),
        ("stteffects ra (x1 x2) (a x1)", "no treatment-model covariates"),
        ("stteffects ipw (a x1) ()", "expected"),
        ("stteffects aipw (x1) (a x1)", "not one of"),
        ("stteffects ra (i.x1 x2) (a)", "plain variable names"),
    ],
)
def test_refusals_name_the_reason(line, reason):
    out = sp.from_stata(line)
    assert out["ok"] is False and reason in out["error"]


def test_without_stset_sp_stata_refuses(data):
    with pytest.raises(MethodIncompatibility, match="not declared as survival"):
        sp.stata("stteffects ra (x1 x2) (a)", data=data)


def test_event_codes_other_than_0_1_are_refused(data):
    df = data.assign(event=data.event * 2)
    with pytest.raises(MethodIncompatibility, match="codes"):
        sp.stata(STSET + "stteffects ra (x1 x2) (a)", data=df)
