"""``sp.spec_curve`` refuses variance names it does not implement.

An unknown ``se_types`` entry used to be computed as HC1 (or as the
classical variance with ``fe=``) under the unknown label, and ``'cluster'``
without ``cluster_var`` was dropped without a word.
"""

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import MethodIncompatibility


@pytest.fixture
def df():
    rng = np.random.default_rng(0)
    n = 200
    out = pd.DataFrame({"x": rng.normal(size=n), "c": rng.normal(size=n)})
    out["g"] = rng.integers(0, 10, n)
    out["y"] = 0.5 * out.x + out.c + rng.normal(size=n)
    return out


@pytest.mark.parametrize("fe", [None, [["g"]]])
def test_unknown_se_type_is_refused(df, fe):
    with pytest.raises(MethodIncompatibility, match="unknown se_types"):
        sp.spec_curve(df, y="y", x="x", controls=[["c"]], se_types=["hc3"], fe=fe)


def test_cluster_without_cluster_var_is_refused(df):
    with pytest.raises(MethodIncompatibility, match="cluster_var"):
        sp.spec_curve(df, y="y", x="x", controls=[["c"]], se_types=["cluster"])


def test_robust_alias_equals_hc1(df):
    a = sp.spec_curve(df, y="y", x="x", controls=[["c"]], se_types=["hc1"])
    b = sp.spec_curve(df, y="y", x="x", controls=[["c"]], se_types=["robust"])
    np.testing.assert_array_equal(
        a.results_df["se"].to_numpy(), b.results_df["se"].to_numpy()
    )
