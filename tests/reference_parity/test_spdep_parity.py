"""Cross-language parity: StatsPAI's spatial battery vs R ``spdep`` /
``spatialreg``.

Fixture: ``_fixtures/_generate_spdep_R.R`` (spdep + spatialreg), run on
``_fixtures/spdep_data.csv`` — a 10x10 rook lattice carrying a spatial-lag
process, a strictly positive transform for the Getis-Ord family and a
binary split for join counts.

**Weights first.** ``sp.W`` is binary until ``w.transform = "R"``; spdep's
``nb2listw`` is row-standardised by default. Comparing the two directly is
not a parity test, it is two different statistics. Every assertion below
names the style it uses, and the fixture emits both.

This file was written alongside four correctness fixes it found. Each has a
regression guard here that does not depend on spdep, so the property
survives even if the fixture is ever regenerated wrongly:

* ``sp.lm_tests`` computed ``T`` as ``tr(WW) + sum_ij W_ij (WW')_ij``
  where Anselin's statistic needs ``tr(W'W) + tr(WW)``, and built the
  lag statistic's ``J`` from ``M(Wy)`` where it needs ``M(W X beta)``. In
  both cases the comment above the line stated the correct formula. The
  robust LM-error statistic — the Anselin rule for choosing between a
  spatial lag and a spatial error model — read 20.49 (p = 6e-6) where the
  correct value is 0.0397 (p = 0.84).
* ``sp.join_counts`` halved the double sum for BB and WW but not for BW,
  so ``BB + WW + BW`` came to 70.75 where ``S0/2`` is 50.
* ``sp.getis_ord_local(star=False)`` standardised Gi with Gi*'s
  whole-sample moments; Gi needs the exclude-self moments of Ord and Getis
  (1995). ``star=True`` was already exact.
* ``sp.moran_residuals`` returned the exact statistic with a p-value from
  the raw-variable null rather than the regression-residual null.
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

_FIX = pathlib.Path(__file__).parent / "_fixtures"


@pytest.fixture(scope="module")
def rjson():
    path = _FIX / "spdep_R.json"
    if not path.exists():  # pragma: no cover
        pytest.skip("run _generate_spdep_R.R first")
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def data():
    path = _FIX / "spdep_data.csv"
    if not path.exists():  # pragma: no cover
        pytest.skip("run _generate_spdep_R.R first")
    return pd.read_csv(path)


@pytest.fixture(scope="module")
def lattice(data):
    gr = data["grid_row"].to_numpy()
    gc = data["grid_col"].to_numpy()
    A = (
        (np.abs(gr[:, None] - gr[None, :]) + np.abs(gc[:, None] - gc[None, :])) == 1
    ).astype(float)
    return A


def _binary_W(A):
    nb = {i: [int(j) for j in np.flatnonzero(A[i])] for i in range(A.shape[0])}
    return sp.W(nb)


def _row_std_W(A):
    w = _binary_W(A)
    w.transform = "R"
    return w


# ── global statistics, row-standardised ───────────────────────────────────


def test_moran_matches_spdep(rjson, data, lattice):
    res = sp.moran(data["y"].to_numpy(), _row_std_W(lattice), permutations=0)
    ref = rjson["moran"]
    assert res.value == pytest.approx(ref["I"], rel=1e-10)
    assert res.expectation == pytest.approx(ref["E"], rel=1e-10)
    assert res.variance == pytest.approx(ref["V"], rel=1e-10)
    assert res.z_score == pytest.approx(ref["z"], rel=1e-10)


def test_geary_statistic_matches_spdep(rjson, data, lattice):
    res = sp.geary(data["y"].to_numpy(), _row_std_W(lattice), permutations=0)
    assert res.value == pytest.approx(rjson["geary_rand"]["C"], rel=1e-10)
    assert res.expectation == pytest.approx(1.0, rel=1e-12)


def test_geary_analytic_variance_matches_spdep(rjson, data, lattice):
    """Added in 1.27.0. `geary(permutations=0)` used to return NaN here.

    Both spdep conventions are pinned: the default randomisation null and
    the normality null, which differ in the fourth-moment term.
    """
    y = data["y"].to_numpy()
    w = _row_std_W(lattice)
    rand = sp.geary(y, w, permutations=0)
    assert rand.variance == pytest.approx(rjson["geary_rand"]["V"], rel=1e-9)
    assert rand.z_score == pytest.approx(rjson["geary_rand"]["z"], rel=1e-9)
    norm = sp.geary(y, w, permutations=0, assumption="normality")
    assert norm.variance == pytest.approx(rjson["geary_norm"]["V"], rel=1e-9)
    assert norm.z_score == pytest.approx(rjson["geary_norm"]["z"], rel=1e-9)


def test_getis_ord_g_matches_spdep_on_binary_weights(rjson, data, lattice):
    res = sp.getis_ord_g(data["pos"].to_numpy(), _binary_W(lattice), permutations=0)
    assert res.value == pytest.approx(rjson["getis_g"]["G"], rel=1e-12)


# ── local statistics ──────────────────────────────────────────────────────


def test_local_moran_matches_spdep(rjson, data, lattice):
    out = sp.moran_local(data["y"].to_numpy(), _row_std_W(lattice), permutations=0)
    got = np.asarray(out["Is"], float)
    ref = np.asarray(rjson["localmoran"]["Ii"], float)
    assert np.allclose(got, ref, rtol=1e-10, atol=0.0)


def test_getis_ord_local_star_matches_spdep(rjson, data, lattice):
    out = sp.getis_ord_local(
        data["pos"].to_numpy(), _binary_W(lattice), star=True, permutations=0
    )
    got = np.asarray(out["Gs"], float)
    ref = np.asarray(rjson["localGstar"], float)
    assert np.allclose(got, ref, rtol=1e-10, atol=0.0)


def test_getis_ord_local_gi_matches_spdep(rjson, data, lattice):
    """``star=False`` used to reuse Gi*'s whole-sample standardisation."""
    out = sp.getis_ord_local(
        data["pos"].to_numpy(), _binary_W(lattice), star=False, permutations=0
    )
    got = np.asarray(out["Gs"], float)
    ref = np.asarray(rjson["localG"], float)
    assert np.allclose(got, ref, rtol=1e-9, atol=0.0)


def test_gi_and_gistar_are_different_statistics(data, lattice):
    """Regression guard, independent of spdep.

    If ``star=False`` ever borrows ``star=True``'s moments again the two
    will move back toward each other; on this fixture they are far apart.
    """
    y = data["pos"].to_numpy()
    w = _binary_W(lattice)
    gi = np.asarray(sp.getis_ord_local(y, w, star=False, permutations=0)["Gs"], float)
    gs = np.asarray(sp.getis_ord_local(y, w, star=True, permutations=0)["Gs"], float)
    assert not np.allclose(gi, gs, rtol=1e-3)


# ── join counts ───────────────────────────────────────────────────────────


def test_join_counts_match_spdep(rjson, data, lattice):
    out = sp.join_counts(data["bin"].to_numpy(), _binary_W(lattice), permutations=0)
    ref = rjson["joincount"]
    assert out["BB"] == pytest.approx(ref["BB"], rel=1e-12)
    assert out["WW"] == pytest.approx(ref["WW"], rel=1e-12)
    assert out["BW"] == pytest.approx(ref["BW"], rel=1e-12)


@pytest.mark.parametrize("style", ["binary", "row"])
def test_join_counts_sum_to_half_the_weight_total(data, lattice, style):
    """The identity that proves the BW defect without any reference.

    Every join is counted exactly once, so BB + WW + BW = S0/2 for any
    weights. Before 1.27.0, BW carried no 1/2 while BB and WW did, and the
    sum came to 70.75 against S0/2 = 50 on the binary lattice.
    """
    w = _binary_W(lattice) if style == "binary" else _row_std_W(lattice)
    out = sp.join_counts(data["bin"].to_numpy(), w, permutations=0)
    s0_half = float(w.sparse.sum()) / 2.0
    assert out["BB"] + out["WW"] + out["BW"] == pytest.approx(s0_half, rel=1e-12)


# ── the LM battery ────────────────────────────────────────────────────────

LM_MAP = {
    "LM_err": "RSerr",
    "LM_lag": "RSlag",
    "Robust_LM_err": "adjRSerr",
    "Robust_LM_lag": "adjRSlag",
    "SARMA": "SARMA",
}


@pytest.mark.parametrize("ours,theirs", sorted(LM_MAP.items()))
def test_lm_tests_match_spdep(rjson, data, lattice, ours, theirs):
    out = sp.lm_tests("y ~ x1 + x2", data, lattice, row_normalize=True)
    stat, pval = out[ours]
    assert stat == pytest.approx(rjson["lm_tests"][theirs]["stat"], rel=1e-9)
    assert pval == pytest.approx(rjson["lm_tests"][theirs]["p"], rel=1e-7)


def test_lm_error_trace_term_is_not_doubled(data, lattice):
    """Regression guard for the ``T`` defect, without spdep.

    ``LM_err`` is inversely proportional to ``T = tr(W'W + WW)``. For a
    row-standardised symmetric adjacency both traces equal ``sum_ij W_ij^2``,
    so T is computable here in one line; the old code returned half of it
    and doubled the statistic.
    """
    Wn = lattice / lattice.sum(1, keepdims=True)
    expected_T = float(np.sum(Wn * Wn) + np.sum(Wn.T * Wn))
    y = data["y"].to_numpy()
    X = np.column_stack(
        [np.ones(len(data)), data["x1"].to_numpy(), data["x2"].to_numpy()]
    )
    beta = np.linalg.solve(X.T @ X, X.T @ y)
    e = y - X @ beta
    s2 = float(e @ e) / len(y)
    lm_err_expected = ((e @ (Wn @ e)) / s2) ** 2 / expected_T
    got = sp.lm_tests("y ~ x1 + x2", data, lattice, row_normalize=True)["LM_err"][0]
    assert got == pytest.approx(lm_err_expected, rel=1e-12)


def test_moran_residuals_matches_lm_morantest(rjson, data, lattice):
    """Statistic *and* p-value.

    The statistic was always exact; the p-value came from the null
    distribution of Moran's I for an observed variable, which is not the
    null for OLS residuals — those depend on X through the hat matrix.
    """
    X = np.column_stack(
        [np.ones(len(data)), data["x1"].to_numpy(), data["x2"].to_numpy()]
    )
    resid = np.asarray(sp.regress("y ~ x1 + x2", data).residuals(), float).ravel()
    I, p = sp.moran_residuals(resid, lattice, row_normalize=True, X=X)
    assert I == pytest.approx(rjson["moran_resid"]["I"], rel=1e-12)
    # lm.morantest defaults to a one-sided "greater" p; we report two-sided,
    # so the fixture records both and this compares like with like.
    assert p == pytest.approx(rjson["moran_resid"]["p"], rel=1e-7)


def test_moran_residuals_without_X_uses_the_raw_variable_null(rjson, data, lattice):
    """The fallback is documented, so it is pinned as a fallback.

    Omitting ``X`` cannot give the regression-residual null -- the null
    depends on the design matrix -- so the function says so by returning a
    visibly different p rather than silently pretending.
    """
    resid = np.asarray(sp.regress("y ~ x1 + x2", data).residuals(), float).ravel()
    I_no, p_no = sp.moran_residuals(resid, lattice, row_normalize=True)
    assert I_no == pytest.approx(rjson["moran_resid"]["I"], rel=1e-12)
    assert p_no != pytest.approx(rjson["moran_resid"]["p"], rel=1e-3)


# ── spatial regressions ───────────────────────────────────────────────────


def test_slx_matches_lmSLX(rjson, data, lattice):
    res = sp.slx(lattice, data, "y ~ x1 + x2", row_normalize=True)
    ref = rjson["slx"]["coef"]
    mapping = {
        "const": "(Intercept)",
        "x1": "x1",
        "x2": "x2",
        "W_x1": "lag.x1",
        "W_x2": "lag.x2",
    }
    for ours, theirs in mapping.items():
        assert float(res.params[ours]) == pytest.approx(ref[theirs], rel=1e-10)


def test_sac_matches_sacsarlm(rjson, data, lattice):
    res = sp.sac(lattice, data, "y ~ x1 + x2", row_normalize=True)
    ref = rjson["sac"]
    assert float(res.params["rho"]) == pytest.approx(ref["rho"], rel=1e-5)
    assert float(res.params["lambda"]) == pytest.approx(ref["lambda"], rel=1e-5)
    assert float(res.params["x1"]) == pytest.approx(ref["coef"]["x1"], rel=1e-6)
    assert float(res.params["x2"]) == pytest.approx(ref["coef"]["x2"], rel=1e-6)


def test_sar_impacts_match_spatialreg(rjson, data, lattice):
    fit = sp.sar(lattice, data, "y ~ x1 + x2", row_normalize=True)
    assert float(fit.params["rho"]) == pytest.approx(rjson["sar"]["rho"], rel=1e-6)
    imp = sp.impacts(fit, n_sim=0)
    for row, direct, indirect, total in zip(
        ("x1", "x2"),
        rjson["sar"]["direct"],
        rjson["sar"]["indirect"],
        rjson["sar"]["total"],
    ):
        assert float(imp.loc[row, "Direct"]) == pytest.approx(direct, rel=1e-6)
        assert float(imp.loc[row, "Indirect"]) == pytest.approx(indirect, rel=1e-6)
        assert float(imp.loc[row, "Total"]) == pytest.approx(total, rel=1e-6)


# ── weights constructors ──────────────────────────────────────────────────


@pytest.fixture(scope="module")
def points():
    path = _FIX / "spdep_points.csv"
    if not path.exists():  # pragma: no cover
        pytest.skip("run _generate_spdep_R.R first")
    return pd.read_csv(path)


def test_knn_weights_match_knearneigh(rjson, points):
    w = sp.knn_weights(points[["cx", "cy"]].to_numpy(), k=4)
    for i, ref in enumerate(rjson["knn4"]):
        assert sorted(w.neighbors[i]) == sorted(int(j) for j in ref)


def test_distance_band_matches_dnearneigh(rjson, points):
    w = sp.distance_band(points[["cx", "cy"]].to_numpy(), threshold=0.25, binary=True)
    for i, ref in enumerate(rjson["dband025"]):
        assert sorted(w.neighbors[i]) == sorted(int(j) for j in ref)
