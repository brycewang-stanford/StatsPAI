"""sp.bacon_decomposition refuses a treatment the two-way effects absorb.

The TWFE coefficient being decomposed is not identified when the treatment
column is a unit effect plus a period effect. Through 1.42.1 every path
returned ``beta_twfe = 0.0`` with no comparisons (and
``balance='observed_rows'`` crashed with a ``KeyError``). R ``bacondecomp``
0.1.1 and Stata ``bacondecomp`` / ``ddtiming`` stop with an error on the
same panel (checked 2026-11-05).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp


def _panel(first_treat: dict, T: int = 6, seed: int = 0) -> pd.DataFrame:
    """``first_treat[unit]``: first treated period, 0 = never treated."""
    rng = np.random.default_rng(seed)
    rows = []
    for unit, g in first_treat.items():
        for t in range(1, T + 1):
            d = int(g > 0 and t >= g)
            rows.append((unit, t, d, unit + 0.3 * t + 2.0 * d + rng.normal()))
    df = pd.DataFrame(rows, columns=["id", "t", "d", "y"])
    df["w"] = 1.0 + (df["id"] % 3)
    df["x"] = np.random.default_rng(seed + 1).normal(size=len(df))
    return df


# ever-treated indicator: 1 in every period for units 0-4, 0 for 5-9
EVER = {u: (1 if u < 5 else 0) for u in range(10)}


@pytest.mark.parametrize(
    "kwargs",
    [{}, {"weights": "w"}, {"covariates": ["x"]}, {"balance": "observed_rows"}],
    ids=["default", "weights", "covariates", "observed_rows"],
)
def test_treatment_constant_within_unit_raises(kwargs):
    df = _panel(EVER)
    assert (df.groupby("id")["d"].nunique() == 1).all()
    with pytest.raises(sp.IdentificationFailure) as err:
        sp.bacon_decomposition(df, y="y", treat="d", time="t", id="id", **kwargs)
    assert "constant within every unit" in str(err.value)
    assert err.value.diagnostics["n_switching_units"] == 0
    assert err.value.diagnostics["n_units"] == 10


def test_single_adoption_date_without_comparison_raises():
    # every unit adopts in period 4: the period effects absorb the treatment
    df = _panel({u: 4 for u in range(8)})
    with pytest.raises(sp.IdentificationFailure, match="period effects absorb"):
        sp.bacon_decomposition(df, y="y", treat="d", time="t", id="id")


def test_identified_designs_next_to_the_boundary_still_decompose():
    # one adoption date plus always-treated units is identified: the
    # decomposition is the single Later vs Always 2x2, equal to the TWFE
    # coefficient (Theorem 1)
    df = _panel({**{u: 4 for u in range(5)}, **{u: 1 for u in range(5, 9)}})
    out = sp.bacon_decomposition(df, y="y", treat="d", time="t", id="id")
    fe = sp.feols("y ~ d | id + t", data=df)
    assert out["beta_twfe"] == pytest.approx(float(fe.params["d"]), rel=1e-10)
    assert out["weighted_sum"] == pytest.approx(out["beta_twfe"], rel=1e-12)
    assert set(out["decomposition"]["type"]) == {"Later vs Always Treated"}
