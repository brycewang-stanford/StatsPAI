"""``stcox ..., tvc() texp()`` through ``sp.from_stata`` / ``sp.stata``.

The executed translations are checked against the Stata 18 numbers that
``tests/reference_parity/_fixtures/_generate_cox_tvc_Stata.do`` wrote for
the same commands on the same CSV bytes (``cox_tvc_Stata.csv``); the
estimator itself is pinned in ``tests/reference_parity/test_cox_tvc_parity.py``.
Tolerance 1e-6 relative, the parity budget (worst observed 3.3e-7, from
Stata's stopping point; see that file).
"""

from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import statspai as sp

_FIX = Path(__file__).parent / "reference_parity" / "_fixtures"
STATA = pd.read_csv(_FIX / "cox_tvc_Stata.csv", skipinitialspace=True)

SCRIPTS = {
    "c2_efron_ln": (
        "cont",
        "stset time, failure(died)\nstcox x1 race, tvc(race) texp(ln(_t)) efron",
    ),
    "c3_efron_t2_robust": (
        "cont",
        "stset time, failure(died)\n"
        "stcox x1, tvc(race x1) texp(_t^2) efron vce(robust)",
    ),
    "c4_breslow_min": (
        "cont",
        "stset time, failure(died)\nstcox x1 race, tvc(race) texp(min(_t, 4))",
    ),
    "t1_efron_t": ("ties", "stset time, failure(died)\nstcox x1 race, tvc(race) efron"),
    "t2_breslow_ln_robust": (
        "ties",
        "stset time, failure(died)\n"
        "stcox x1 race, tvc(race x1) texp(ln(_t)) breslow vce(robust)",
    ),
    "t3_efron_strata_cluster": (
        "ties",
        "stset time, failure(died)\n"
        "stcox x1 race, tvc(race) efron strata(grp) vce(cluster cl)",
    ),
    "t5_efron_step": (
        "ties",
        "stset time, failure(died)\nstcox x1 race, tvc(race) texp(_t > 5) efron",
    ),
    "t4_breslow_pw": (
        "ties",
        "stset time [pweight = w], failure(died)\nstcox x1 race, tvc(race) breslow",
    ),
    "t6_efron_entry_ln": (
        "ties",
        "stset time, failure(died) time0(entry)\n"
        "stcox x1 race, tvc(race) texp(ln(_t)) efron",
    ),
}


@pytest.fixture(scope="module")
def frames():
    return {
        "cont": pd.read_csv(_FIX / "cox_tvc_cont.csv"),
        "ties": pd.read_csv(_FIX / "cox_tvc_ties.csv"),
    }


@pytest.mark.parametrize("spec", sorted(SCRIPTS))
def test_sp_stata_reproduces_stcox_tvc(frames, spec):
    which, script = SCRIPTS[spec]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sp.stata(script, data=frames[which])
    ref = STATA[STATA.spec == spec]
    names = [n.replace("main:", "") for n in ref.name]
    np.testing.assert_allclose(res.params[names], ref.b, rtol=1e-6, atol=0)
    np.testing.assert_allclose(res.std_errors[names], ref.se, rtol=1e-6, atol=0)


def test_from_stata_arguments():
    out = sp.from_stata("stcox x1 race, tvc(race x1) texp(ln(_t)) efron")
    assert out["ok"] and out["complete"]
    args = out["arguments"]
    assert args["tvc"] == ["race", "x1"] and args["texp"] == "ln(_t)"
    assert args["ties"] == "efron"
    assert "tvc=['race', 'x1']" in out["python_code"]
    assert "texp='ln(_t)'" in out["python_code"]
    # texp(_t) is the default of both programs and is not written out
    out = sp.from_stata("stcox x1, tvc(race) texp(_t)")
    assert out["complete"] and "texp" not in out["arguments"]
    assert out["arguments"]["tvc"] == ["race"]


@pytest.mark.parametrize(
    "line, why",
    [
        ("stcox x1, texp(ln(_t))", "without tvc"),
        ("stcox x1, tvc(i.race)", "plain variable names"),
        ("stcox x1, tvc(race) texp(x1*_t)", "texp"),
        ("stcox x1, tvc(race) texp(_t^2^2)", "texp"),
    ],
)
def test_untranslatable_tvc_options_are_refused(line, why):
    out = sp.from_stata(line)
    assert out["ok"] is False
    assert why in out["error"]
