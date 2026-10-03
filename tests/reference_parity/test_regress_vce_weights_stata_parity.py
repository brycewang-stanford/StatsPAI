"""``sp.regress`` under analytic weights and the cluster-type variances, vs Stata 18.

Fixture: ``_fixtures/_generate_regress_vce_weights_stata.do`` (official
commands only, double precision) on the translation holdout cross-section
(400 rows, 40 clusters ``g``, a second key ``k``).

What is pinned:

* ``weights=`` with the classical, HC1, HC2, HC3 and one-way cluster
  variances: coefficients and standard errors against
  ``regress ... [aw=w]`` at 1e-10.
* ``vce='cr2'`` against ``regress, vce(hc2 g)``, with and without weights.
* Two defects that were silent until 2026-10-03, each with its own test:

  - ``weights=`` together with ``vce='cr2'`` / ``'cr3'`` or two-way
    clustering was dropped: the call returned the *unweighted*
    coefficients and standard errors.
  - with a missing value in the formula, the cluster keys for those
    variances were taken from the first ``n`` rows of the data rather than
    from the rows that were fitted (standard errors off by 17% for CR2 and
    66% for two-way clustering on this fixture with four rows missing).

Two documented differences of convention, asserted as exact identities so
they cannot be mistaken for agreement:

* Two-way clustering. StatsPAI scales the whole inclusion-exclusion meat
  by ``G_min/(G_min - 1) * (N - 1)/(N - K)``. Stata 18's
  ``regress, vce(cluster a b)`` scales each of the three components by its
  own ``G/(G - 1) * (N - 1)/(N - K)``. Both are rebuilt here from the same
  cluster scores.
* ``vce='cr3'`` / ``'jackknife'`` is ``sum_g (b_(g) - b)(b_(g) - b)'``.
  Stata's ``vce(jackknife, cluster(g) mse)`` multiplies that by
  ``(G - 1)/G``. Stata stores the replicates in single precision, so the
  comparison holds to 1e-6, not 1e-10; the delete-one-cluster identity
  itself is checked by brute force at 1e-10.
"""

from __future__ import annotations

import json
import pathlib
import warnings

import numpy as np
import pandas as pd
import pytest

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import statspai as sp

from statspai.exceptions import MethodIncompatibility

_HERE = pathlib.Path(__file__).parent
_FIX = _HERE / "_fixtures" / "regress_vce_weights_stata.json"
_DATA = _HERE.parent / "stata_translation_holdout" / "holdout_cross.csv"
_NAMES = (("x1", "x1"), ("x2", "x2"), ("Intercept", "_cons"))
RTOL = 1e-10
FORMULA = "y ~ x1 + x2"


@pytest.fixture(scope="module")
def ref():
    if not _FIX.exists():  # pragma: no cover
        pytest.skip("run _generate_regress_vce_weights_stata.do first")
    return json.loads(_FIX.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def data():
    return pd.read_csv(_DATA)


def _fit(data, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.regress(FORMULA, data, **kw)


def _assert_matches(res, cell, *, rtol=RTOL, se_scale=1.0):
    for ours, theirs in _NAMES:
        assert float(res.params[ours]) == pytest.approx(cell[f"b_{theirs}"], rel=RTOL)
        assert float(res.std_errors[ours]) * se_scale == pytest.approx(
            cell[f"se_{theirs}"], rel=rtol
        )


# ── numbers against Stata ─────────────────────────────────────────────── #


@pytest.mark.parametrize(
    "key,kw",
    [
        ("aw_classical", dict(weights="w")),
        ("aw_hc1", dict(weights="w", vce="hc1")),
        ("aw_hc2", dict(weights="w", vce="hc2")),
        ("aw_hc3", dict(weights="w", vce="hc3")),
        ("aw_cr1", dict(weights="w", cluster="g")),
        ("aw_cr2", dict(weights="w", vce="cr2", cluster="g")),
        ("cr2", dict(vce="cr2", cluster="g")),
    ],
)
def test_matches_stata(ref, data, key, kw):
    _assert_matches(_fit(data, **kw), ref[key])


def test_cr3_is_stata_s_cluster_jackknife_up_to_its_factor(ref, data):
    g = data["g"].nunique()
    res = _fit(data, vce="cr3", cluster="g")
    _assert_matches(
        res, ref["jackknife_cluster_mse"], rtol=1e-6, se_scale=np.sqrt((g - 1) / g)
    )
    alias = _fit(data, vce="jackknife", cluster="g")
    np.testing.assert_allclose(alias.std_errors, res.std_errors, rtol=1e-14)


# ── two-way clustering: the difference from Stata is the factor, exactly ─ #


def _scores(data, weights):
    X = np.column_stack([np.ones(len(data)), data["x1"], data["x2"]])
    y = data["y"].to_numpy()
    w = np.ones(len(data)) if weights is None else data[weights].to_numpy()
    bread = np.linalg.inv(X.T @ (X * w[:, None]))
    resid = y - X @ (bread @ (X.T @ (w * y)))
    return X.shape, bread, X * (w * resid)[:, None]


def _meat(scores, keys):
    codes = pd.factorize(keys)[0]
    m = np.zeros((scores.shape[1],) * 2)
    for c in range(codes.max() + 1):
        s = scores[codes == c].sum(axis=0)
        m += np.outer(s, s)
    return m, codes.max() + 1


@pytest.mark.parametrize("key,weights", [("twoway", None), ("aw_twoway", "w")])
def test_two_way_differs_from_stata_by_the_small_sample_factor_only(
    ref, data, key, weights
):
    (n, k), bread, scores = _scores(data, weights)
    both = pd.Series(list(zip(data["g"], data["k"])))
    parts = [_meat(scores, data["g"]), _meat(scores, data["k"]), _meat(scores, both)]
    signs = (1.0, 1.0, -1.0)
    dof = (n - 1) / (n - k)

    g_min = min(parts[0][1], parts[1][1])
    ours = g_min / (g_min - 1) * dof * sum(s * m for s, (m, _) in zip(signs, parts))
    stata = sum(s * g / (g - 1) * dof * m for s, (m, g) in zip(signs, parts))
    se_ours = np.sqrt(np.diag(bread @ ours @ bread))
    se_stata = np.sqrt(np.diag(bread @ stata @ bread))

    res = _fit(data, cluster=["g", "k"], weights=weights)
    np.testing.assert_allclose(res.std_errors.to_numpy(), se_ours, rtol=RTOL)
    cell = ref[key]
    np.testing.assert_allclose(
        se_stata, [cell["se__cons"], cell["se_x1"], cell["se_x2"]], rtol=RTOL
    )
    for ours_name, theirs in _NAMES:
        assert float(res.params[ours_name]) == pytest.approx(
            cell[f"b_{theirs}"], rel=RTOL
        )
    # ... and the two conventions do differ on this fixture.
    assert np.max(np.abs(se_ours / se_stata - 1)) > 3e-3


# ── the silent defects ─────────────────────────────────────────────────── #


@pytest.mark.parametrize(
    "kw",
    [
        dict(vce="cr2", cluster="g"),
        dict(vce="cr3", cluster="g"),
        dict(cluster=["g", "k"]),
    ],
)
def test_weights_are_used_not_dropped(data, kw):
    plain = _fit(data, **kw)
    weighted = _fit(data, weights="w", **kw)
    wls = _fit(data, weights="w")
    np.testing.assert_allclose(weighted.params, wls.params, rtol=1e-12)
    assert np.max(np.abs(weighted.params / plain.params - 1)) > 1e-3
    assert np.max(np.abs(weighted.std_errors / plain.std_errors - 1)) > 1e-3
    assert weighted.model_info["weighted"] is True


@pytest.mark.parametrize("weights", [None, "w"])
def test_cr3_is_the_delete_one_cluster_jackknife(data, weights):
    res = _fit(data, vce="cr3", cluster="g", weights=weights)
    full = _fit(data, weights=weights).params.to_numpy()
    total = np.zeros((3, 3))
    for g in sorted(data["g"].unique()):
        b = _fit(data[data["g"] != g], weights=weights).params.to_numpy()
        total += np.outer(b - full, b - full)
    np.testing.assert_allclose(
        res.std_errors.to_numpy(), np.sqrt(np.diag(total)), rtol=RTOL
    )


@pytest.mark.parametrize(
    "kw",
    [
        dict(vce="cr2", cluster="g"),
        dict(vce="cr3", cluster="g"),
        dict(cluster=["g", "k"]),
        dict(cluster=["g", "k"], weights="w"),
        dict(vce="wild", cluster="g", seed=1, wild_reps=99),
    ],
)
def test_rows_dropped_for_missing_values_keep_their_own_cluster(data, kw):
    holed = data.copy()
    holed.loc[[3, 50, 51, 200], "x1"] = np.nan
    clean = holed.dropna(subset=["x1"]).reset_index(drop=True)
    a = _fit(holed, **kw)
    b = _fit(clean, **kw)
    assert int(a.data_info["nobs"]) == 396
    np.testing.assert_allclose(a.std_errors, b.std_errors, rtol=1e-12)


@pytest.mark.parametrize(
    "kw",
    [
        dict(vce="wild", cluster="g"),
        dict(vce="conley", conley_lat="x1", conley_lon="x2", conley_cutoff=1.0),
    ],
)
def test_weights_are_refused_where_they_are_not_implemented(data, kw):
    with pytest.raises(MethodIncompatibility, match="weights"):
        sp.regress(FORMULA, data, weights="w", **kw)


def test_a_missing_cluster_key_drops_the_row_as_stata_does(data):
    holed = data.copy()
    holed["k"] = holed["k"].astype(float)
    holed.loc[5, "k"] = np.nan
    with pytest.warns(sp.exceptions.StatsPAIWarning, match="missing cluster"):
        a = sp.regress(FORMULA, holed, cluster=["g", "k"])
    b = _fit(holed.drop(index=5).reset_index(drop=True), cluster=["g", "k"])
    assert int(a.data_info["nobs"]) == 399
    np.testing.assert_allclose(a.std_errors, b.std_errors, rtol=1e-12)


def test_the_weighted_fit_is_not_reported_as_the_unweighted_cell(data):
    scope = sp.validation_scope(_fit(data, weights="w", vce="hc1"))
    scope = scope if isinstance(scope, dict) else scope.to_dict()
    assert scope["configuration"]["weights"] == "set"


# ── the standalone helpers read the same sample and the same weights ───── #


@pytest.mark.parametrize("key,weights", [("twoway", None), ("aw_twoway", "w")])
def test_twoway_cluster_helper_matches_stata(ref, data, key, weights):
    """``sp.twoway_cluster`` uses the per-dimension factor, which is Stata's.

    On a weighted fit it used the unweighted scores: standard errors 11%
    to 17% below Stata's on this fixture.
    """
    res = sp.twoway_cluster(_fit(data, weights=weights), data, "g", "k")
    _assert_matches(res, ref[key])


@pytest.mark.parametrize("key,weights", [("cr2", None), ("aw_cr2", "w")])
def test_cr2_se_helper_matches_stata(ref, data, key, weights):
    res = sp.cr2_se(_fit(data, weights=weights), data, cluster="g")
    _assert_matches(res, ref[key])


def test_helpers_follow_the_fitted_rows(data):
    holed = data.copy()
    holed.loc[[3, 50, 51, 200], "x1"] = np.nan
    clean = holed.dropna(subset=["x1"]).reset_index(drop=True)
    for helper in (
        lambda f, d: sp.twoway_cluster(f, d, "g", "k"),
        lambda f, d: sp.cr2_se(f, d, cluster="g"),
    ):
        a = helper(_fit(holed, weights="w"), holed)
        b = helper(_fit(clean, weights="w"), clean)
        np.testing.assert_allclose(a.std_errors, b.std_errors, rtol=1e-12)
