"""``sp.regress`` omits collinear regressors the way Stata's ``regress`` does.

Through 1.32 a rank-deficient design raised (Busting the Princelings, QJE
2019: a dummy set plus a hand-made level dummy). Stata omits instead. The
fixture (``_generate_regress_collinear_Stata.do``, Stata 18) records which
member Stata omits and the coefficients and SEs of the rest.

Where factor variables are involved Stata scans the regressors as written
and omits the later member -- the ``_rmcoll`` / ``reghdfe`` rule -- and
``sp.regress`` reproduces the omitted set, coefficients and SEs to 1e-10.
With plain variables only, ``regress`` pivots on the data inside its solver
(``x d0 d1 d2`` omits ``d0``, not ``d2``); ``sp.regress`` keeps the
written-order rule there, so the kept set is a different normalisation of
the same model: identical R-squared, root MSE, and coefficient on every
regressor outside the collinear set.
"""

from __future__ import annotations

import json
import pathlib
import re
import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import NumericalInstability

_FIX = pathlib.Path(__file__).parent / "_fixtures"
STATA = json.loads((_FIX / "regress_collinear_Stata.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def cs():
    return pd.read_csv(_FIX / "regress_collinear_cs.csv")


@pytest.fixture(scope="module")
def es():
    return pd.read_csv(_FIX / "regress_collinear_es.csv")


def _stata_terms(key):
    """{our name: (b, se)} for the kept terms, and the omitted names."""
    ref = STATA[key]
    kept, omitted = {}, []
    for name, b, se in zip(ref["names"].split(), ref["b"], ref["se"]):
        m = re.fullmatch(r"(\d+)(b?)(o?)\.(\w+)", name)
        if m:  # factor level
            level, base, omit, var = m.groups()
            ours = f"C({var})[T.{level}]"
            if base:
                continue
            if omit:
                omitted.append(ours)
                continue
        elif name.startswith("o."):
            omitted.append(name[2:])
            continue
        else:
            ours = "Intercept" if name == "_cons" else name
        kept[ours] = (b, se)
    return kept, omitted


def _fit(formula, data, **kw):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        r = sp.regress(formula, data=data, **kw)
    notes = [
        str(x.message) for x in w if "omitted because of collinearity" in str(x.message)
    ]
    return r, notes


@pytest.mark.parametrize(
    "key, formula",
    [("fv_last", "y ~ x + C(g) + d1"), ("fv_first", "y ~ d1 + C(g) + x")],
)
def test_factor_variable_designs_match_stata(cs, key, formula):
    r, notes = _fit(formula, cs)
    kept, omitted = _stata_terms(key)
    assert [o["variable"] for o in r.model_info["omitted"]] == omitted
    assert notes, "omission must warn"
    assert set(r.params.index) == set(kept)
    for name, (b, se) in kept.items():
        assert r.params[name] == pytest.approx(b, rel=1e-10, abs=1e-12)
        assert r.std_errors[name] == pytest.approx(se, rel=1e-10)


def test_event_study_with_unit_and_period_dummies_matches_stata(es):
    lags = "em7 em6 em5 em4 em3 em2 ep0 ep1 ep2 ep3 ep4 ep5 ep6".split()
    r, _ = _fit("y ~ " + " + ".join(lags) + " + C(u) + C(t)", es, cluster="u")
    kept, omitted = _stata_terms("event_study")
    # Stata omits the last two period dummies (8o.t 9o.t), keeping every lag.
    assert [o["variable"] for o in r.model_info["omitted"]] == omitted
    assert omitted == ["C(t)[T.8]", "C(t)[T.9]"]
    for name in lags:
        b, se = kept[name]
        assert r.params[name] == pytest.approx(b, rel=1e-9)
        assert r.std_errors[name] == pytest.approx(se, rel=1e-9)


@pytest.mark.parametrize(
    "key, formula, outside",
    [
        ("plain_dummies", "y ~ x + d0 + d1 + d2", "x"),
        ("plain_continuous", "y ~ x + w1 + wsum", None),
    ],
)
def test_plain_variable_designs_are_the_same_model(cs, key, formula, outside):
    r, _ = _fit(formula, cs)
    ref = STATA[key]
    assert len(r.model_info["omitted"]) == 1
    assert r.r2 == pytest.approx(ref["r2"], rel=1e-12)
    rmse = float(
        np.sqrt(np.sum(r.data_info["residuals"] ** 2) / r.data_info["df_resid"])
    )
    assert rmse == pytest.approx(ref["rmse"], rel=1e-12)
    if outside is not None:
        kept, _ = _stata_terms(key)
        assert r.params[outside] == pytest.approx(kept[outside][0], rel=1e-10)
        assert r.std_errors[outside] == pytest.approx(kept[outside][1], rel=1e-10)


def test_raise_mode_keeps_the_old_behaviour(cs):
    with pytest.raises(NumericalInstability, match="collinear|linear combination"):
        sp.regress("y ~ x + d0 + d1 + d2", data=cs, collinear="raise")
    with pytest.raises(ValueError, match="collinear must be"):
        sp.regress("y ~ x", data=cs, collinear="drop")


def test_full_rank_design_is_untouched(cs):
    r, notes = _fit("y ~ x + w1 + C(g)", cs)
    assert not notes and "omitted" not in r.model_info
