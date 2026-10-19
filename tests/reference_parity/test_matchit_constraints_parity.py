"""``sp.match(exact=, covariate_calipers=, mahvars=)`` against R MatchIt.

Reference: ``MatchIt::matchit(method = "nearest")`` 4.7.2 with ``exact =``,
named entries of ``caliper =`` and ``mahvars =``, run by
``_fixtures/_generate_matchit_constraints_R.R`` on the synthetic
``_fixtures/matchit_constraints.csv`` (1,000 units, 394 treated). R is not
executed in CI.

Every configuration must reproduce MatchIt's matched pairs *exactly* (the
treated -> control assignment, row by row) and its ATT (the weighted
difference in means of ``match.data``) to 1e-10. The match sets are
discrete, so the pair comparison is the real test; the ATT tolerance only
absorbs summation order (observed differences are below 1e-14).

Translation: MatchIt's default ``m.order`` is ``"largest"`` for a
propensity-score distance and ``"data"`` for ``distance =
"mahalanobis"``; ``std.caliper = TRUE`` is ``caliper_scale='sd'``.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import statspai as sp

FIX = Path(__file__).parent / "_fixtures"

CONFIGS = {
    "exact_ps": dict(
        covariates=["x1", "x2", "g"], exact=["g"], replace=False, m_order="largest"
    ),
    "exact_ps_replace": dict(covariates=["x1", "x2", "g"], exact=["g"], replace=True),
    "exact_two_ps": dict(
        covariates=["x1", "x2", "g", "x3"],
        exact=["g", "x3"],
        replace=False,
        m_order="largest",
    ),
    "exact_mahal": dict(
        covariates=["x1", "x2"],
        distance="mahalanobis",
        exact="g",
        replace=False,
        m_order="data",
    ),
    "covcal_ps": dict(
        covariates=["x1", "x2", "g"],
        covariate_calipers={"x1": 0.1},
        caliper_scale="sd",
        replace=False,
        m_order="largest",
    ),
    "covcal_ps_raw": dict(
        covariates=["x1", "x2", "g"],
        covariate_calipers={"x2": 0.05},
        replace=False,
        m_order="largest",
    ),
    "mahvars": dict(
        covariates=["x1", "x2", "g"],
        mahvars=["x1", "x2"],
        caliper=0.25,
        caliper_scale="sd",
        replace=False,
        m_order="largest",
    ),
    "mahvars_exact_replace": dict(
        covariates=["x1", "x2", "g"],
        mahvars=["x1", "x2"],
        exact=["x3"],
        caliper=0.5,
        caliper_scale="sd",
        replace=True,
    ),
}


@pytest.fixture(scope="module")
def data():
    return pd.read_csv(FIX / "matchit_constraints.csv")


@pytest.fixture(scope="module")
def ref():
    return json.loads((FIX / "matchit_constraints_R.json").read_text(encoding="utf-8"))


def _pairs(result) -> dict:
    """treated row -> control row, 1-based like R row names."""
    md = result.matched_data
    row_of_id = dict(zip(md["_id"].to_numpy(), md.index.to_numpy() + 1))
    treated = md[(md["_treated"] == 1) & md["_n1"].notna()]
    return {
        int(i + 1): int(row_of_id[n1])
        for i, n1 in zip(treated.index, treated["_n1"].to_numpy())
    }


@pytest.mark.parametrize("name", list(CONFIGS))
def test_constrained_nearest_matching_reproduces_matchit(data, ref, name):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sp.match(data, y="y", treat="d", **CONFIGS[name])
    r = ref[name]
    expected = {
        int(t): int(c)
        for t, c in zip(r["treated_row"], r["control_row"])
        if c is not None and c != "NA"
    }
    assert _pairs(res) == expected
    assert res.model_info["n_matched_treated"] == r["n_matched"]
    assert abs(res.estimate - r["att"]) < 1e-10


def test_exact_cells_are_respected_and_counted(data):
    """Identity: every pair agrees on the exact variables; a treated unit
    whose cell has no control stays unmatched and is counted."""
    df = data.copy()
    # one treated unit alone in its own cell
    lonely = int(np.flatnonzero(df["d"].to_numpy() == 1)[0])
    df.loc[lonely, "g"] = 9
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sp.match(df, y="y", treat="d", covariates=["x1", "x2"], exact=["g"])
    pairs = _pairs(res)
    g = df["g"].to_numpy()
    assert all(g[t - 1] == g[c - 1] for t, c in pairs.items())
    assert lonely + 1 not in pairs
    assert res.model_info["n_treated_without_exact_control"] == 1


def test_pair_options_refused_outside_nearest(data):
    with pytest.raises(sp.MethodIncompatibility, match="nearest"):
        sp.match(data, y="y", treat="d", covariates=["x1"], method="cem", exact=["g"])
    with pytest.raises(sp.MethodIncompatibility, match="propensity"):
        sp.match(
            data,
            y="y",
            treat="d",
            covariates=["x1"],
            distance="mahalanobis",
            mahvars=["x2"],
        )
    with pytest.raises(sp.MethodIncompatibility, match="positive"):
        sp.match(
            data, y="y", treat="d", covariates=["x1"], covariate_calipers={"x1": 0}
        )


def test_unknown_match_keywords_name_the_right_spelling(data):
    with pytest.raises(sp.MethodIncompatibility, match="bias_adjust"):
        sp.match(
            data,
            y="y",
            treat="d",
            covariates=["x1"],
            method="nnmatch",
            bias_correction=True,
        )
    with pytest.raises(TypeError, match="bias_correction"):
        sp.match(data, y="y", treat="d", covariates=["x1"], bias_adjust=True)
    with pytest.raises(sp.MethodIncompatibility, match="exact="):
        sp.match(data, y="y", treat="d", covariates=["x1"], exactt=["g"])
