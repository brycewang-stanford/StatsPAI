"""``sp.bart_causal`` vs R ``bcf`` 2.0.2 / ``bartCause`` 1.0.10: seed screen.

Tier S (CLAUDE.md §5.1): a stochastic screen, reported and not graded,
never parity. The two sides implement the same models with different
samplers and hyperprior details. Data: ``_fixtures/bart_causal_data.csv``
(the DGP of bcf's help page, n = 300, true propensity supplied to both
sides). Each side runs eight single-chain fits (1000 burn-in, 1000 kept)
with seeds ``1000 + 100000 k`` (``_generate_bart_causal_R.R``,
``_generate_bart_causal_py.py``); the R packages are GPL and were used as
black boxes.

Observed (2026-10-19): BCF posterior means of the ATE 0.06 posterior SDs
apart, posterior SDs within 1.2%, seed-averaged CATE correlation 0.995.
BART (bartCause response model, CATE type) was 0.21 posterior SDs apart,
beyond seed noise (seed SDs 0.003-0.004). Located 2026-10-20 in the prior,
from dbarts' documentation of ``bart2`` (which bartCause calls): 75 trees,
not 200, and cut points equally spaced over each covariate's range, not at
quantiles. 75 trees alone leave 0.14 SDs, the grid alone 0.05; with both
(the 'bart' defaults now) the means are 0.02 posterior SDs apart (within
the 0.03 SD standard error of a difference of two 8-seed means), posterior
SDs within 0.6%, CATE correlation 0.999. The bounds below are screens
around those values.
"""

from __future__ import annotations

import json
import pathlib
import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp

_FIX = pathlib.Path(__file__).parent / "_fixtures"
R = json.loads((_FIX / "bart_causal_R.json").read_text(encoding="utf-8"))
PY = json.loads((_FIX / "bart_causal_py.json").read_text(encoding="utf-8"))
BOUND_MEAN = {"bcf": 0.25, "bart": 0.25}  # in posterior SDs


def _stats(side, m):
    mean = np.array([x["ate_mean"] for x in side[m]])
    sd = np.array([x["ate_sd"] for x in side[m]])
    cate = np.mean([x["cate_mean"] for x in side[m]], axis=0)
    return mean, sd, cate


@pytest.mark.parametrize("m", ["bcf", "bart"])
def test_posterior_of_the_ate_screens_against_r(m):
    pm, ps, _ = _stats(PY, m)
    rm, rs, _ = _stats(R, m)
    assert abs(pm.mean() - rm.mean()) <= BOUND_MEAN[m] * rs.mean()
    assert 0.85 <= ps.mean() / rs.mean() <= 1.15


@pytest.mark.parametrize("m", ["bcf", "bart"])
def test_individual_effects_screen_against_r(m):
    _, _, pc = _stats(PY, m)
    _, _, rc = _stats(R, m)
    assert np.corrcoef(pc, rc)[0, 1] > 0.98


def test_a_fresh_bcf_fit_lands_in_the_r_seed_distribution():
    d = pd.read_csv(_FIX / "bart_causal_data.csv")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = sp.bart_causal(
            d,
            "y",
            "z",
            ["x1", "x2", "x3"],
            method="bcf",
            propensity="pi",
            draws=500,
            burnin=500,
            chains=1,
            seed=99,
        )
    rm, rs, _ = _stats(R, "bcf")
    assert abs(r.estimate - rm.mean()) <= 0.5 * rs.mean()
