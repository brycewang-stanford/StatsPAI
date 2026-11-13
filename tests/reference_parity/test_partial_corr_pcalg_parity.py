"""``sp.partial_corr_pvalue`` against ``pcalg::gaussCItest``.

Reference: pcalg 2.7.12 (GPL, run as a black box) on
``_fixtures/ness_gaussian.csv``: every pair of the eight columns with
conditioning sets of size 0 to 3, on all 500 rows and on the first 60
(``_fixtures/_generate_partial_corr_pcalg_R.R``, 224 p-values in
``_fixtures/partial_corr_pcalg_R.json``). Both sides compute the Fisher z
test of a zero partial correlation with ``sqrt(n - |Z| - 3)`` scaling.

Tolerance: 1e-9 relative (observed 1.5e-12); the p-values range from
3e-144 to 1. R gets the partial correlation from the inverse of a
correlation submatrix, StatsPAI from regression residuals, which is where
the last digits differ.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import statspai as sp

_FIX = pathlib.Path(__file__).parent / "_fixtures"
REF = json.loads((_FIX / "partial_corr_pcalg_R.json").read_text(encoding="utf-8"))
DATA = pd.read_csv(_FIX / "ness_gaussian.csv")


def _ours(row):
    d = DATA.iloc[: row["n"]]
    Z = d[row["z"]].to_numpy() if row["z"] else None
    return sp.partial_corr_pvalue(d[row["x"]].to_numpy(), d[row["y"]].to_numpy(), Z)


def test_fixture_is_pcalg_output_over_a_wide_range():
    assert REF["versions"]["pcalg"] == "2.7.12"
    p = np.array([r["pvalue"] for r in REF["rows"]])
    assert len(p) == 224
    assert p.min() < 1e-100 and (p > 0.5).sum() > 20
    assert {len(r["z"]) for r in REF["rows"]} == {0, 1, 2, 3}


@pytest.mark.parametrize("k", range(len(REF["rows"])))
def test_p_value_is_gaussCItest(k):
    row = REF["rows"][k]
    assert _ours(row) == pytest.approx(row["pvalue"], rel=1e-9, abs=1e-300)


def test_marginal_case_is_the_fisher_z_test_by_hand():
    x, y = DATA["X1"].to_numpy(), DATA["X3"].to_numpy()
    r = np.corrcoef(x, y)[0, 1]
    z = np.sqrt(len(x) - 3) * np.arctanh(r)
    assert sp.partial_corr_pvalue(x, y) == pytest.approx(
        2 * stats.norm.sf(abs(z)), rel=1e-12
    )
