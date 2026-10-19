"""sp.panel(method='fe', vce='wild') with clusters that cut across entities,
against Stata ``areg ..., absorb(id) vce(cluster c)`` + ``boottest``.

The entity effects are not nested in the clusters here, so the bootstrap
has to re-absorb them in every draw. Before, sp.panel bootstrapped the
already-demeaned design (valid only for nested effects): on an 8-cluster
version of this design it reported p = 0.0625 / 0.117 where boottest gives
0.078 / 0.180. With 12 clusters boottest enumerates all 4096 Rademacher
sign vectors, so its p-value is exact; the confidence-set bounds are held
to 0.3% of the width (boottest's ptol), as in
``test_hdfe_wild_boottest_parity.py``.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import statspai as sp

_FIX = Path(__file__).parent / "fixtures" / "panel_wild"
# stata_boottest.do: r(p) and r(CI) per variable
BOOTTEST = {
    "x1": (0.0654296875, -0.0199614603936035, 0.433168636396784),
    "x2": (0.00537109375, -0.360398739124929, -0.0702803456414),
}


@pytest.fixture(scope="module")
def fit():
    df = pd.read_csv(_FIX / "panel_wild.csv")
    return sp.panel(
        df,
        "y ~ x1 + x2",
        entity="id",
        time="t",
        method="fe",
        vce="wild",
        cluster="c",
        wild_reps=10000,
        wild_weight_type="rademacher",
        seed=1,
    )


@pytest.mark.parametrize("var", ["x1", "x2"])
def test_p_value_equals_boottest(fit, var):
    assert fit.pvalues[var] == BOOTTEST[var][0]


@pytest.mark.parametrize("var", ["x1", "x2"])
def test_confidence_set_matches_boottest(fit, var):
    p, lo, hi = BOOTTEST[var]
    width = hi - lo
    assert fit.conf_int_lower[var] == pytest.approx(lo, abs=3e-3 * width)
    assert fit.conf_int_upper[var] == pytest.approx(hi, abs=3e-3 * width)


def test_nested_clusters_keep_the_demeaned_bootstrap():
    # clustering on the entity: the effects are nested and the bootstrap on
    # the demeaned design is exact, identical to sp.regress on that design
    df = pd.read_csv(_FIX / "panel_wild.csv")
    kw = dict(entity="id", time="t", method="fe", vce="wild", wild_reps=999, seed=3)
    a = sp.panel(df, "y ~ x1 + x2", **kw)
    b = sp.panel(df, "y ~ x1 + x2", cluster="id", **kw)
    np.testing.assert_array_equal(a.pvalues, b.pvalues)
