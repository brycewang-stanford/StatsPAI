"""``sp.gate_test(clan=)`` vs R ``GenericML::CLAN`` 0.2.3 (GPL, black box).

Fixture ``_fixtures/clan_data.csv`` / ``clan_R.json`` from
``_fixtures/_generate_clan_R.R``. Given the same quantile groups of the
proxy, CLAN is a set of group means (``sd / sqrt(n_k)``) and the
most-minus-least difference (Welch), so both sides agree to rounding.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd

import statspai as sp

_FIX = pathlib.Path(__file__).parent / "_fixtures"
REF = json.loads((_FIX / "clan_R.json").read_text(encoding="utf-8"))


def test_clan_matches_genericml():
    d = pd.read_csv(_FIX / "clan_data.csv")
    res = sp.gate_test(
        d["proxy"].to_numpy(),
        d,
        by="cate",
        y="y",
        treat="d",
        covariates=["z1", "z2"],
        propensity=np.full(len(d), 0.5),
        proxy=d["proxy"].to_numpy(),
        clan=["z1", "z2", "z3"],
    )
    tab = res["clan_table"]
    for v in ("z1", "z2", "z3"):
        got = tab[tab["variable"] == v]
        np.testing.assert_allclose(
            got["estimate"], REF[v]["estimate"], rtol=1e-12, atol=1e-14
        )
        np.testing.assert_allclose(got["se"], REF[v]["se"], rtol=1e-12)
