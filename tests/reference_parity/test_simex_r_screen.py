"""``sp.simex`` against R ``simex`` 1.8: a stochastic screen (tier S).

Reference: ``_fixtures/simex_R.txt`` written by
``_fixtures/_generate_simex_R.R`` (R 4.5.2, simex 1.8, B = 2000) on
``_fixtures/simex_data.csv`` (``_generate_simex_data.py``). Linear and logit
naive models, error SD 0.5 on ``w``, lambda = (0.5, 1, 1.5, 2), quadratic
and linear extrapolants, jackknife variance extrapolated with the
quadratic.

Both sides simulate, with different random number generators, so this is
not parity: it checks that the two implementations compute the same
estimator up to Monte Carlo error. The tolerance per (model, extrapolant)
is 5 standard deviations of the difference of the two simulation means,
from a seed study of ``sp.simex`` on these data (8 seeds spaced by 100000,
B = 250, scaled to B): Monte Carlo SD at B = 1000 of the largest
coefficient was 0.0034 (gaussian quadratic), 0.0012 (gaussian linear),
0.0035 (logit quadratic), 0.0012 (logit linear), and of the largest
jackknife SE 0.0011 (gaussian), 0.0010 (logit). With B = 500 here and
2000 in R the difference has SD sqrt(2 + 0.5) times the B = 1000 value.
In units of the sampling SE (0.07 to 0.14) the tolerances are 0.1 to 0.4.
"""

from __future__ import annotations

import pathlib
import warnings
from functools import lru_cache
from typing import Dict

import numpy as np
import pandas as pd
import pytest

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import statspai as sp

_FIX = pathlib.Path(__file__).parent / "_fixtures"
B = 500
SCALE = 5 * np.sqrt(2.0 + 0.5)
MC_SD_B1000 = {
    ("gaussian", "quadratic"): 0.0034,
    ("gaussian", "linear"): 0.0012,
    ("logit", "quadratic"): 0.0035,
    ("logit", "linear"): 0.0012,
}
MC_SD_SE_B1000 = {"gaussian": 0.0011, "logit": 0.0010}
FORMULA = {"gaussian": "y ~ w + z", "logit": "d ~ w + z"}


@lru_cache(maxsize=1)
def _r() -> Dict[str, float]:
    out: Dict[str, float] = {}
    text = (_FIX / "simex_R.txt").read_text(encoding="utf-8")
    for line in text.splitlines():
        if line.strip():
            key, value = line.split()
            out[key] = float(value)
    return out


@pytest.mark.parametrize("case", sorted(MC_SD_B1000))
def test_simex_matches_r_up_to_simulation_error(case):
    family, extrapolant = case
    ref = _r()
    df = pd.read_csv(_FIX / "simex_data.csv")
    fit = sp.simex(
        FORMULA[family],
        df,
        error_sd={"w": 0.5},
        family=family,
        B=B,
        extrapolant=extrapolant,
        seed=20261028,
    )
    tol_b = SCALE * MC_SD_B1000[case]
    tol_se = SCALE * MC_SD_SE_B1000[family]
    for term in ("Intercept", "w", "z"):
        r_b = ref[f"{family}_{extrapolant}_b_{term}"]
        r_se = ref[f"{family}_{extrapolant}_se_{term}"]
        assert abs(fit.params[term] - r_b) <= tol_b, (term, fit.params[term], r_b)
        assert abs(fit.std_errors[term] - r_se) <= tol_se, (
            term,
            fit.std_errors[term],
            r_se,
        )
