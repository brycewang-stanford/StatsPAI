"""``sp.feols`` over its common configuration grid against R ``fixest``.

Absorbed effects {none, firm, firm + year} x vcov {iid, hetero, cluster g,
cluster g + h} x weights {none, w}: 24 cells, coefficients and SEs, written
in Stata spelling (``vce='robust'``, ``'cluster g h'``) on our side and
fixest's (``vcov = 'hetero'``, ``~g + h``) on R's. Reference:
``_fixtures/_generate_feols_grid_R.R`` on ``feols_grid_data.csv``.

With two absorbed effects the SEs agree to ~1e-8 rather than 1e-15: both
sides demean iteratively and stop at their own convergence tolerance.
"""

import json
import pathlib
import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp

_FIX = pathlib.Path(__file__).parent / "_fixtures"
R = json.loads((_FIX / "feols_grid_R.json").read_text(encoding="utf-8"))
RTOL = 1e-6  # CLAUDE.md §5.1 default budget; observed <= 1.2e-8

FES = {
    "none": "y ~ x1 + x2",
    "one": "y ~ x1 + x2 | firm",
    "multi": "y ~ x1 + x2 | firm + year",
}
VCE = {"iid": "ols", "hetero": "robust", "cr1": "cluster g", "twoway": "cluster g h"}


@pytest.fixture(scope="module")
def data():
    return pd.read_csv(_FIX / "feols_grid_data.csv")


@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("vce", sorted(VCE))
@pytest.mark.parametrize("fe", sorted(FES))
def test_feols_cell_matches_fixest(data, fe, vce, weighted):
    kw = {"weights": "w"} if weighted else {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = sp.feols(FES[fe], data=data, vce=VCE[vce], **kw)
    ref = R[f"{fe}_{vce}_{'w' if weighted else 'none'}"]
    np.testing.assert_allclose(fit.params[["x1", "x2"]], ref["coef"], rtol=RTOL)
    np.testing.assert_allclose(fit.std_errors[["x1", "x2"]], ref["se"], rtol=RTOL)
