"""sp.shapley_r2 against Stata shapley2 and R domir / relaimpo.

References (same CSV bytes, ``_fixtures/shapley_r2_data.csv``):

* R 4.5.2, domir 1.3.0 (general dominance) and relaimpo 2.2.7 (LMG):
  ungrouped, grouped (``.set``), analytic weights and always-included
  controls (``.all`` / ``always=``).  Generator
  ``_fixtures/_generate_shapley_r2_R.R``.
* Stata 18, shapley2 1.5 (SSC) with ``stat(r2)``, ungrouped and with
  ``group()``.  Generator ``_fixtures/_generate_shapley_r2_Stata.do``.

Tolerances.  R: rtol 1e-10, because jsonlite writes 15 significant
digits (the observed gap is ~1e-14).  Stata: shapley2 keeps each subset
R-squared in a float variable (its ``svmat`` call uses the default
storage type), so its Shapley values carry float rounding, ~3e-7
relative here.  The parity row uses the default T2 budget rtol 1e-6, and
``test_stata_gap_is_float_storage`` pins the mechanism: rounding every
subset R-squared (computed independently here by least squares) to
float32 reproduces Stata's numbers to 1e-12.
"""

from __future__ import annotations

import itertools
import json
import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import (
    DataInsufficient,
    MethodIncompatibility,
    NumericalInstability,
)

FIX = Path(__file__).resolve().parent / "_fixtures"
X = [f"x{j}" for j in range(1, 7)]
GROUPS = {"g1": ["x1", "x2"], "g2": ["x3"], "g3": ["x4", "x5", "x6"]}


@pytest.fixture(scope="module")
def df():
    return pd.read_csv(FIX / "shapley_r2_data.csv")


@pytest.fixture(scope="module")
def ref_r():
    return json.loads((FIX / "shapley_r2_R.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def ref_stata():
    return json.loads((FIX / "shapley_r2_Stata.json").read_text(encoding="utf-8"))


# ---------------------------------------------------------------- R rows --


@pytest.mark.parametrize(
    "case, kwargs",
    [
        ("ungrouped", {}),
        ("grouped", {"groups": GROUPS}),
        ("weighted", {"weights": "w"}),
        ("fixed", {"covariates_fixed": ["z"]}),
        (
            "fixed_grouped_weighted",
            {"covariates_fixed": ["z"], "weights": "w", "groups": GROUPS},
        ),
    ],
)
def test_domir_parity(df, ref_r, case, kwargs):
    res = sp.shapley_r2(df, y="y", x=X, **kwargs)
    ref = ref_r[case]
    np.testing.assert_allclose(res.detailed.contribution, ref["shapley"], rtol=1e-10)
    np.testing.assert_allclose(res.r2, ref["r2"], rtol=1e-10)
    if "r2_fixed" in ref:
        np.testing.assert_allclose(res.r2_fixed, ref["r2_fixed"], rtol=1e-10)


@pytest.mark.parametrize(
    "key, kwargs",
    [
        ("relaimpo_lmg", {}),
        ("relaimpo_lmg_weighted", {"weights": "w"}),
        ("relaimpo_lmg_fixed", {"covariates_fixed": ["z"]}),
    ],
)
def test_relaimpo_lmg_parity(df, ref_r, key, kwargs):
    res = sp.shapley_r2(df, y="y", x=X, **kwargs)
    np.testing.assert_allclose(res.detailed.contribution, ref_r[key], rtol=1e-10)


# ------------------------------------------------------------ Stata rows --


def test_stata_shapley2_parity(df, ref_stata):
    res = sp.shapley_r2(df, y="y", x=X)
    np.testing.assert_allclose(res.r2, ref_stata["ungrouped"]["r2"], rtol=1e-12)
    np.testing.assert_allclose(
        res.detailed.contribution, ref_stata["ungrouped"]["shapley"], rtol=1e-6
    )
    g = sp.shapley_r2(df, y="y", x=X, groups=GROUPS)
    np.testing.assert_allclose(
        g.detailed.contribution, ref_stata["grouped"]["shapley"], rtol=1e-6
    )


def _ols_r2(df, cols):
    if not cols:
        return 0.0
    A = np.column_stack([np.ones(len(df)), df[list(cols)].to_numpy()])
    yv = df["y"].to_numpy()
    beta = np.linalg.lstsq(A, yv, rcond=None)[0]
    e = yv - A @ beta
    return 1.0 - e @ e / np.sum((yv - yv.mean()) ** 2)


def _shapley_float32(df, players):
    p = len(players)
    cache = {}

    def v(T):
        key = tuple(sorted(T))
        if key not in cache:
            cols = [c for i in key for c in players[i]]
            cache[key] = float(np.float32(_ols_r2(df, cols)))
        return cache[key]

    out = []
    for j in range(p):
        others = [h for h in range(p) if h != j]
        s = 0.0
        for r in range(p):
            w = math.factorial(r) * math.factorial(p - r - 1) / math.factorial(p)
            for T in itertools.combinations(others, r):
                s += w * (v(T + (j,)) - v(T))
        out.append(s)
    return np.array(out)


def test_stata_gap_is_float_storage(df, ref_stata):
    np.testing.assert_allclose(
        _shapley_float32(df, [[c] for c in X]),
        ref_stata["ungrouped"]["shapley"],
        rtol=1e-12,
    )
    np.testing.assert_allclose(
        _shapley_float32(df, list(GROUPS.values())),
        ref_stata["grouped"]["shapley"],
        rtol=1e-12,
    )


def test_single_regressor_r2_matches_stata(df, ref_stata):
    for j, c in enumerate(X):
        r = sp.shapley_r2(df, y="y", x=[c])
        np.testing.assert_allclose(r.r2, ref_stata["single_r2"][j], rtol=1e-12)
        assert r.detailed.contribution.iloc[0] == pytest.approx(r.r2, rel=1e-14)


# ------------------------------------------------------- analytic checks --


def test_efficiency_and_owen_identities(df):
    res = sp.shapley_r2(df, y="y", x=X, groups=GROUPS, covariates_fixed=["z"])
    # contributions sum to the decomposed R2, Owen values to group values
    assert res.detailed.contribution.sum() == pytest.approx(res.decomposed, abs=1e-14)
    assert res.r2 == pytest.approx(res.r2_fixed + res.decomposed, abs=1e-15)
    by_group = res.owen.groupby("group", sort=False).contribution.sum()
    np.testing.assert_allclose(
        by_group.loc[list(GROUPS)], res.detailed.contribution, atol=1e-14
    )
    # singleton groups reproduce the ungrouped Shapley values
    plain = sp.shapley_r2(df, y="y", x=X, covariates_fixed=["z"])
    single = sp.shapley_r2(
        df, y="y", x=X, covariates_fixed=["z"], groups={c: [c] for c in X}
    )
    np.testing.assert_allclose(
        single.detailed.contribution, plain.detailed.contribution, atol=1e-14
    )
    np.testing.assert_allclose(
        single.owen.contribution, plain.detailed.contribution, atol=1e-14
    )
    # one group holding everything: its Owen values are the plain Shapley values
    one = sp.shapley_r2(df, y="y", x=X, covariates_fixed=["z"], groups={"all": X})
    np.testing.assert_allclose(
        one.owen.contribution, plain.detailed.contribution, atol=1e-14
    )


def test_orthogonal_regressors_are_additive():
    rng = np.random.default_rng(5)
    Z = rng.normal(size=(200, 3))
    Z = Z - Z.mean(axis=0)
    Q = np.linalg.qr(Z)[0] * np.sqrt(200)
    d = pd.DataFrame(Q, columns=["a", "b", "c"])
    d["y"] = 1 + 0.5 * d.a - 0.2 * d.b + rng.normal(size=200)
    res = sp.shapley_r2(d, y="y", x=["a", "b", "c"])
    for c, v in zip("abc", res.detailed.contribution):
        assert v == pytest.approx(_ols_r2(d, [c]), abs=1e-13)


def test_weights_scale_invariant_and_missing_rows(df):
    a = sp.shapley_r2(df, y="y", x=X, weights="w")
    b = sp.shapley_r2(df, y="y", x=X, weights=df["w"].to_numpy() * 7.5)
    np.testing.assert_allclose(
        a.detailed.contribution, b.detailed.contribution, rtol=1e-12
    )
    d2 = df.copy()
    d2.loc[[0, 5, 9], "x3"] = np.nan
    r = sp.shapley_r2(d2, y="y", x=X)
    full = sp.shapley_r2(df.drop(index=[0, 5, 9]), y="y", x=X)
    assert r.n_obs == len(df) - 3 and r.diagnostics["n_dropped"] == 3
    np.testing.assert_allclose(
        r.detailed.contribution, full.detailed.contribution, rtol=1e-12
    )


# ----------------------------------------------------------------- edges --


def test_too_many_players_raises():
    rng = np.random.default_rng(0)
    d = pd.DataFrame(rng.normal(size=(100, 22)), columns=[f"v{i}" for i in range(22)])
    d["y"] = rng.normal(size=100)
    with pytest.raises(MethodIncompatibility) as ei:
        sp.shapley_r2(d, y="y", x=[f"v{i}" for i in range(21)])
    assert ei.value.recovery_hint
    # 22 regressors in 3 groups: group values exact, Owen values skipped
    groups = {"a": [f"v{i}" for i in range(8)], "b": [f"v{i}" for i in range(8, 16)]}
    with pytest.warns(UserWarning, match="Owen"):
        res = sp.shapley_r2(d, y="y", x=[f"v{i}" for i in range(22)], groups=groups)
    assert res.owen is None
    assert len(res.detailed) == 2 + 6
    assert res.detailed.contribution.sum() == pytest.approx(res.r2, abs=1e-14)


def test_collinear_constant_and_bad_groups(df):
    d = df.copy()
    d["x7"] = d.x1 + 2 * d.x2
    with pytest.raises(NumericalInstability):
        sp.shapley_r2(d, y="y", x=X + ["x7"])
    d["k"] = 3.0
    with pytest.raises(DataInsufficient):
        sp.shapley_r2(d, y="k", x=X)
    with pytest.raises(MethodIncompatibility):
        sp.shapley_r2(df, y="y", x=X, groups={"a": ["x1", "x2"], "b": ["x2"]})
    with pytest.raises(MethodIncompatibility):
        sp.shapley_r2(df, y="y", x=X, groups={"a": ["x1", "nope"]})
    with pytest.raises(MethodIncompatibility):
        sp.shapley_r2(df, y="y", x=X, weights=-df["w"])
    with pytest.raises(MethodIncompatibility):
        sp.shapley_r2(df, y="y", x=X, covariates_fixed=["x1"])
    with pytest.raises(DataInsufficient):
        sp.shapley_r2(df.head(6), y="y", x=X)


def test_result_surface(df):
    res = sp.shapley_r2(df, y="y", x=X, groups=GROUPS)
    assert list(res.tidy().columns) == ["term", "estimate", "share", "pct_of_r2"]
    assert res.detailed.share.sum() == pytest.approx(1.0, abs=1e-14)
    assert "Owen" in res.summary()
    assert res.cite("bibtex_keys")[0] == "shorrocks2013decomposition"
    assert "doi:10.1214/12-EJS710" in res.cite()
    assert json.dumps(res.to_dict(), default=str)
