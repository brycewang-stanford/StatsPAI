"""``sp.hdfe_ols`` fit statistics against Stata ``reghdfe``.

``hdfe_ols`` reported only the within R-squared, so a replication had to
rebuild ``reghdfe``'s adjusted R-squared by hand -- and could not, when a
fixed effect was nested in the cluster variable: ``reghdfe`` then moves
those effects out of ``e(df_a)`` into ``e(df_a_nested)`` for inference but
charges them back in ``e(r2_a)`` (``used_df_r = N - K - df_a -
df_a_nested``).

Reference: Stata 18 ``reghdfe`` 6.13.1 on
``_fixtures/reghdfe_fitstats.csv`` (unbalanced panel with singletons), from
``_generate_reghdfe_fitstats_Stata.do``: A ``absorb(id year)``; B the same
with ``vce(cluster id)`` (``id`` nested); C ``absorb(id city#year)
vce(cluster cy)`` with ``cy = group(city year)`` (the cluster itself
absorbed). Degrees of freedom exact; statistics to 1e-12.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd
import pytest

import statspai as sp

_FIX = pathlib.Path(__file__).parent / "_fixtures"
REF = json.loads((_FIX / "reghdfe_fitstats_Stata.json").read_text(encoding="utf-8"))
SPECS = {
    "A": ("y ~ x + z | id + year", None),
    "B": ("y ~ x + z | id + year", "id"),
    "C": ("y ~ x + z | id + city^year", "cy"),
}


@pytest.fixture(scope="module")
def data():
    d = pd.read_csv(_FIX / "reghdfe_fitstats.csv")
    d["cy"] = d.groupby(["city", "year"]).ngroup()
    return d


@pytest.mark.parametrize("spec", sorted(SPECS))
def test_fit_statistics_match_reghdfe(data, spec):
    formula, cluster = SPECS[spec]
    r = sp.hdfe_ols(formula, data=data, cluster=cluster)
    ref = REF[spec]
    assert r.n_obs == ref["N"]
    assert r.df_a == ref["df_a"]
    assert r.df_a_nested == ref["df_a_nested"]
    for key in ("r2", "r2_a", "r2_within", "r2_a_within", "rmse"):
        np.testing.assert_allclose(getattr(r, key), ref[key], rtol=1e-12, err_msg=key)
