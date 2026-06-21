"""Tests for ``sp.romano_wolf`` — Romano-Wolf stepdown multiple-testing
correction across outcomes (previously untested public function).

References
----------
Romano, J.P. and Wolf, M. (2005). "Stepwise Multiple Testing as Formalized
Data Snooping." Econometrica 73(4), 1237-1282.
"""

import numpy as np
import pandas as pd

import statspai as sp


def _multi_outcome_data(n=500, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    return pd.DataFrame(
        {
            "x": x,
            "y1": 2.0 * x + rng.normal(size=n),  # strong true effect
            "y2": rng.normal(size=n),  # null
            "y3": 0.1 * x + rng.normal(size=n),  # weak
        }
    )


def test_romano_wolf_table_contract():
    rw = sp.romano_wolf(
        _multi_outcome_data(), y=["y1", "y2", "y3"], x="x", n_boot=1000, seed=1
    )
    assert rw.n_outcomes == 3
    cols = set(rw.table.columns)
    assert {"outcome", "coef", "se", "p_value", "p_rw", "p_bonf"} <= cols
    adj = rw.table[["p_rw", "p_bonf", "p_holm", "p_bh"]].to_numpy()
    assert np.all((adj >= 0) & (adj <= 1))


def test_romano_wolf_no_more_conservative_than_bonferroni():
    rw = sp.romano_wolf(
        _multi_outcome_data(), y=["y1", "y2", "y3"], x="x", n_boot=2000, seed=1
    )
    t = rw.table
    # RW exploits dependence across outcomes -> never harsher than Bonferroni.
    assert np.all(t["p_rw"].to_numpy() <= t["p_bonf"].to_numpy() + 1e-9)


def test_romano_wolf_separates_signal_from_null():
    rw = sp.romano_wolf(
        _multi_outcome_data(), y=["y1", "y2", "y3"], x="x", n_boot=2000, seed=1
    )
    t = rw.table.set_index("outcome")
    assert t.loc["y1", "p_rw"] < 0.05  # strong effect survives correction
    assert t.loc["y2", "p_rw"] > 0.10  # null is not spuriously rejected


def test_romano_wolf_seed_reproducible():
    df = _multi_outcome_data()
    a = sp.romano_wolf(df, y=["y1", "y2", "y3"], x="x", n_boot=500, seed=99)
    b = sp.romano_wolf(df, y=["y1", "y2", "y3"], x="x", n_boot=500, seed=99)
    np.testing.assert_allclose(a.table["p_rw"].to_numpy(), b.table["p_rw"].to_numpy())
