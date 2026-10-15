"""The variance-estimator menu of ``sp.regress`` / ``sp.iv`` / ``sp.ivreg``.

One cross-section (``_fixtures/vce_menu_data.csv``, 360 rows, 12 clusters)
is fitted under every covariance estimator the two entry points offer and
compared, coefficient by coefficient, with the implementation that defines
it:

* R (``_fixtures/_generate_vce_menu_R.R`` -> ``vce_menu_R.json``):
  ``sandwich`` HC0-HC4 / ``vcovCL`` / ``NeweyWest`` on ``lm`` and
  ``ivreg::ivreg``; ``clubSandwich`` CR2 / CR3; ``fixest`` two- and
  three-way clustering at its default ``ssc()``; ``fwildclusterboot``.
* Stata 18 (``tests/stata_parity/option_parity/92_vce_menu.do`` ->
  ``92_vce_menu_Stata.json``): ``ivregress 2sls|liml`` with and without
  ``small``; ``ivreg2`` Fuller(1) and two-way clustering; ``acreg``
  (Conley); ``newey``; ``boottest`` after ``regress`` (WCR) and ``ivreg2``
  (WRE).

Both wild bootstraps run with 12 clusters and 4096 draws, so every
Rademacher sign vector is enumerated on every side and the p-values are
deterministic multiples of 1/4096: they are compared exactly.

Estimates and standard errors are held to ``RTOL = 1e-9``; the largest gap
observed when the fixtures were generated was 1e-13.

Two cells are conventions, not parity, and are tested as such (T4):

* weighted CR2. Stata ``vce(hc2 g) [aw=w]`` and StatsPAI apply the CR2
  adjustment to the sqrt(w)-transformed design; ``clubSandwich`` uses the
  Pustejovsky-Tipton symmetric form. Both satisfy the CR2 unbiasedness
  condition ``A_g M_g A_g' = W_g^{-1}`` under the working model, and they
  coincide when the weights are constant within clusters. The test
  rebuilds clubSandwich's number from the fit's own design and checks the
  condition for both forms.
* multiway clustering after LIML / Fuller (and 2SLS through ``sp.iv``).
  The inclusion-exclusion matrix is not positive semi-definite on this
  data. ``sp.iv`` zeroes its negative eigenvalues (as ``fixest`` does, and
  matches it to 1e-13 for 2SLS); ``ivreg2`` and ``sp.ivreg`` report the raw
  matrix. The test rebuilds ``ivreg2``'s raw matrix from three one-way
  ``sp.iv`` fits and shows that its projection is ``sp.iv``'s.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Any, Dict

import numpy as np
import pandas as pd
import pytest
import scipy.linalg as sla

import statspai as sp
from statspai.exceptions import MethodIncompatibility

_FIX = Path(__file__).resolve().parent / "_fixtures"
_OPT = Path(__file__).resolve().parents[1] / "stata_parity" / "option_parity"

RTOL = 1e-9
F = "y ~ d + x1 + x2 + x3"
IV = {
    "over": "y ~ (d ~ z1 + z2) + x1 + x2 + x3",
    "just": "y ~ (d ~ z1) + x1 + x2 + x3",
}
TERMS = ("d", "x1", "x2", "x3", "Intercept")
CONLEY = dict(conley_lat="lat", conley_lon="lon", conley_cutoff=100)
_R_NAME = {"(Intercept)": "Intercept", "fit_d": "d"}


@pytest.fixture(scope="module")
def data() -> pd.DataFrame:
    return pd.read_csv(_FIX / "vce_menu_data.csv")


@pytest.fixture(scope="module")
def ref_r() -> Dict[str, Any]:
    return json.loads((_FIX / "vce_menu_R.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def ref_stata() -> Dict[str, Any]:
    path = _OPT / "results" / "92_vce_menu_Stata.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _quiet(fn, *args, **kwargs):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*args, **kwargs)


def _assert_r(fit: Any, ref: Dict[str, Any], *, se: bool = True) -> None:
    for name, b in ref["b"].items():
        term = _R_NAME.get(name, name)
        np.testing.assert_allclose(fit.params[term], b, rtol=RTOL, err_msg=term)
        if se:
            np.testing.assert_allclose(
                fit.std_errors[term], ref["se"][name], rtol=RTOL, err_msg=term
            )


def _assert_stata(fit: Any, ref: Dict[str, Any]) -> None:
    for term in TERMS:
        key = "_cons" if term == "Intercept" else term
        np.testing.assert_allclose(
            fit.params[term], ref[f"b_{key}"], rtol=RTOL, err_msg=term
        )
        np.testing.assert_allclose(
            fit.std_errors[term], ref[f"se_{key}"], rtol=RTOL, err_msg=term
        )


def _cell(fit: Any, **expected: str) -> None:
    """The fit lands in the validation-scope cell its row claims."""
    got = sp.validation_scope(fit)["configuration"]
    assert {k: got[k] for k in expected} == expected


# --------------------------------------------------------------------------- #
#  sp.regress
# --------------------------------------------------------------------------- #

#: R key (without the weight suffix) -> (sp.regress options, scope vce)
_OLS = {
    "ols_classical": ({}, "classical"),
    "ols_hc0": ({"robust": "hc0"}, "hc0"),
    "ols_hc1": ({"robust": "hc1"}, "hc1"),
    "ols_hc2": ({"robust": "hc2"}, "hc2"),
    "ols_hc3": ({"robust": "hc3"}, "hc3"),
    "ols_hc4": ({"robust": "hc4"}, "hc4"),
    "ols_cr1": ({"cluster": "g"}, "cr1"),
    "ols_cr3": ({"vce": "cr3", "cluster": "g"}, "cr3"),
    "ols_hac": ({"robust": "hac", "hac_lags": 4}, "hac"),
    "ols_hac_small": ({"robust": "hac", "hac_lags": 4, "hac_small": True}, "hac"),
    "ols_twoway": ({"cluster": ["g", "yr"]}, "cluster_multiway"),
}


@pytest.mark.parametrize("weighted", [False, True], ids=["unweighted", "aw"])
@pytest.mark.parametrize("key", sorted(_OLS))
def test_regress_matches_r(data, ref_r, key, weighted):
    """sandwich / clubSandwich / fixest on lm, unweighted and weighted."""
    opts, vce = _OLS[key]
    fit = _quiet(sp.regress, F, data, weights="w" if weighted else None, **opts)
    _assert_r(fit, ref_r[key + ("_w" if weighted else "")])
    _cell(fit, vce=vce, weights="set" if weighted else "none")


def test_regress_unweighted_cr2_matches_clubsandwich(data, ref_r):
    fit = _quiet(sp.regress, F, data, vce="cr2", cluster="g")
    _assert_r(fit, ref_r["ols_cr2"])
    _cell(fit, vce="cr2", weights="none")


@pytest.mark.parametrize("weighted", [False, True], ids=["unweighted", "aw"])
def test_regress_jackknife_is_cr3(data, ref_r, weighted):
    """vce='jackknife' is the CR3 variance, sum_g (b_(g) - b)(b_(g) - b)'."""
    fit = _quiet(
        sp.regress,
        F,
        data,
        vce="jackknife",
        cluster="g",
        weights="w" if weighted else None,
    )
    _assert_r(fit, ref_r["ols_cr3" + ("_w" if weighted else "")])


def test_regress_three_way_clustering_is_refused(data):
    """fixest's three-way numbers are in the fixture; sp.regress has no
    three-way path and must say so rather than fall back to one-way."""
    with pytest.raises(MethodIncompatibility):
        sp.regress(F, data, cluster=["g", "yr", "h"])


def _cr2_forms(data: pd.DataFrame) -> Dict[str, np.ndarray]:
    """Both weighted CR2 adjustments, rebuilt from the fitted design.

    With precision weights W, residual-maker rows M_g = W_g^-1 - X_g B X_g'
    (B = (X'WX)^-1), and the working model Var(e) = W^-1, an adjustment A_g
    is unbiased when A_g M_g A_g' = W_g^-1.

    * transformed design (Stata, StatsPAI): A_g = W_g^-1/2 (I - H~_gg)^-1/2
      W_g^1/2, with H~ the hat matrix of sqrt(W) X;
    * Pustejovsky-Tipton (clubSandwich, inverse_var=TRUE): A_g = D (D M_g
      D)^-1/2 D with D = W_g^-1/2.
    """
    X = np.column_stack([np.ones(len(data)), data[["d", "x1", "x2", "x3"]]])
    w = data["w"].to_numpy(float)
    y = data["y"].to_numpy(float)
    g = data["g"].to_numpy()
    B = np.linalg.inv(X.T @ (X * w[:, None]))
    e = y - X @ (B @ X.T @ (w * y))
    order = [1, 2, 3, 4, 0]  # d, x1, x2, x3, Intercept
    meats = {"transformed": np.zeros((5, 5)), "pustejovsky_tipton": np.zeros((5, 5))}
    for c in np.unique(g):
        i = g == c
        Xg, wg, eg = X[i], w[i], e[i]
        Mg = np.diag(1 / wg) - Xg @ B @ Xg.T
        sw = np.sqrt(wg)
        Ht = (Xg * sw[:, None]) @ B @ (Xg * sw[:, None]).T
        root = sla.sqrtm(np.linalg.inv(np.eye(len(wg)) - Ht)).real
        A_t = np.diag(1 / sw) @ root @ np.diag(sw)
        D = np.diag(1 / sw)
        A_p = D @ sla.sqrtm(np.linalg.inv(D @ Mg @ D)).real @ D
        for A in (A_t, A_p):  # the unbiasedness condition, both forms
            np.testing.assert_allclose(A @ Mg @ A.T, np.diag(1 / wg), atol=1e-10)
        for name, A in (("transformed", A_t), ("pustejovsky_tipton", A_p)):
            s = Xg.T @ (wg * (A @ eg))
            meats[name] += np.outer(s, s)
    return {k: np.sqrt(np.diag(B @ m @ B))[order] for k, m in meats.items()}


def test_regress_weighted_cr2_is_the_transformed_design_form(data, ref_r):
    """T4: the weighted CR2 adjustment is not unique.

    StatsPAI's number is the transformed-design form (the one Stata 18's
    regress [aw=w], vce(hc2 g) reports, matched to 1e-10 in
    test_regress_vce_weights_stata_parity.py); clubSandwich's
    inverse_var=TRUE number is the Pustejovsky-Tipton form. Both are rebuilt
    here from the same fit; the gap on this data is up to 0.53% (x3).
    """
    fit = _quiet(sp.regress, F, data, vce="cr2", cluster="g", weights="w")
    forms = _cr2_forms(data)
    se = fit.std_errors[list(TERMS)].to_numpy()
    np.testing.assert_allclose(se, forms["transformed"], rtol=1e-10)
    club = ref_r["ols_cr2_w_inverse_var"]["se"]
    club_se = np.array([club[{"Intercept": "(Intercept)"}.get(t, t)] for t in TERMS])
    np.testing.assert_allclose(forms["pustejovsky_tipton"], club_se, rtol=1e-10)
    _cell(fit, vce="cr2", weights="set")


def test_regress_wild_bootstrap_matches_enumerated_references(data, ref_r, ref_stata):
    """WCR with all 2^12 sign vectors: p-values equal fwildclusterboot and
    boottest exactly; the reported SE is the CR1 one (vcovCL / vce(cluster))."""
    fit = _quiet(sp.regress, F, data, vce="wild", cluster="g", wild_reps=4096, seed=1)
    for term in ("d", "x1", "x2", "x3"):
        p = float(fit.pvalues[term])
        assert p * 4096 == round(p * 4096)
        assert p == ref_r[f"ols_wild_{term}"]["p_val"], term
        assert p == ref_stata[f"ols_wild_{term}"]["p"], term
    # x3 has a zero coefficient; its p-value is not at either end
    assert 0.5 < float(fit.pvalues["x3"]) < 1.0
    _assert_r(fit, ref_r["ols_cr1"])
    _assert_stata(fit, ref_stata["ols_cr1"])
    _cell(fit, vce="wild", weights="none")


def test_regress_conley_matches_acreg(data, ref_stata):
    fit = _quiet(sp.regress, F, data, vce="conley", **CONLEY)
    _assert_stata(fit, ref_stata["ols_conley"])
    _cell(fit, vce="conley", weights="none")


@pytest.mark.parametrize("weighted", [False, True], ids=["unweighted", "aw"])
def test_regress_newey_small_matches_stata_newey(data, ref_stata, weighted):
    fit = _quiet(
        sp.regress,
        F,
        data,
        robust="hac",
        hac_lags=4,
        hac_small=True,
        weights="w" if weighted else None,
    )
    _assert_stata(fit, ref_stata["ols_newey" + ("_aw" if weighted else "")])
    _cell(fit, vce="hac", weights="set" if weighted else "none")


# --------------------------------------------------------------------------- #
#  sp.iv / sp.ivreg -- R
# --------------------------------------------------------------------------- #

#: R key suffix -> (entry point, options, scope vce)
_IV_R = {
    "classical": ("iv", {}, "classical"),
    "hc0": ("iv", {"robust": "hc0"}, "hc0"),
    "hc1": ("iv", {"robust": "hc1"}, "hc1"),
    "hc2": ("iv", {"robust": "hc2"}, "hc2"),
    "hc3": ("iv", {"robust": "hc3"}, "hc3"),
    "cr1": ("iv", {"cluster": "g"}, "cr1"),
    "cr2": ("ivreg", {"vce": "cr2", "cluster": "g"}, "cr2"),
    "cr3": ("ivreg", {"vce": "cr3", "cluster": "g"}, "cr3"),
    "twoway": ("iv", {"cluster": ["g", "yr"]}, "cluster_multiway"),
    "threeway": ("iv", {"cluster": ["g", "yr", "h"]}, "cluster_multiway"),
}


@pytest.mark.parametrize("ident", ["over", "just"])
@pytest.mark.parametrize("key", sorted(_IV_R))
def test_iv_2sls_matches_r(data, ref_r, key, ident):
    """ivreg::ivreg + sandwich / clubSandwich, and fixest multiway."""
    entry, opts, vce = _IV_R[key]
    fn = sp.iv if entry == "iv" else sp.ivreg
    fit = _quiet(fn, IV[ident], data, **opts)
    _assert_r(fit, ref_r[f"iv_{ident}_{key}"])
    _cell(fit, estimator="2sls", vce=vce, identification=ident, small="true")


@pytest.mark.parametrize("ident", ["over", "just"])
def test_ivreg_jackknife_is_cr3(data, ref_r, ident):
    fit = _quiet(sp.ivreg, IV[ident], data, vce="jackknife", cluster="g")
    _assert_r(fit, ref_r[f"iv_{ident}_cr3"])


# --------------------------------------------------------------------------- #
#  sp.iv / sp.ivreg -- Stata
# --------------------------------------------------------------------------- #

#: ivregress vce -> (sp.iv options, scope vce with small, without small)
_IVREGRESS = {
    "unadjusted": ({}, "classical", "classical"),
    "robust": ({"robust": "hc1"}, "hc1", "hc0"),
    "cluster": ({"cluster": "g"}, "cr1", "cr1"),
}


@pytest.mark.parametrize("small", [True, False], ids=["small", "nosmall"])
@pytest.mark.parametrize("vce", sorted(_IVREGRESS))
@pytest.mark.parametrize("method", ["2sls", "liml"])
@pytest.mark.parametrize("ident", ["over", "just"])
def test_iv_matches_ivregress(data, ref_stata, ident, method, vce, small):
    opts, cell_small, cell_large = _IVREGRESS[vce]
    fit = _quiet(sp.iv, IV[ident], data, method=method, small=small, **opts)
    tag = "small" if small else "nosmall"
    _assert_stata(fit, ref_stata[f"ivregress_{ident}_{method}_{vce}_{tag}"])
    _cell(
        fit,
        estimator=method,
        vce=cell_small if small else cell_large,
        identification=ident,
        small="true" if small else "false",
    )


@pytest.mark.parametrize("ident", ["over", "just"])
def test_liml_hc0_with_small_is_ivregress_robust_without_small(data, ref_stata, ident):
    """robust='hc0' carries no finite-sample factor, so its SE is ivregress
    liml, vce(robust)'s (only the t / z reference law differs)."""
    fit = _quiet(sp.iv, IV[ident], data, method="liml", robust="hc0")
    _assert_stata(fit, ref_stata[f"ivregress_{ident}_liml_robust_nosmall"])
    _cell(fit, estimator="liml", vce="hc0", identification=ident, small="true")


#: ivreg2 Fuller(1) row -> sp.iv options, scope vce
_FULLER = {
    "classical_small": ({}, "classical"),
    "robust_small": ({"robust": "hc1"}, "hc1"),
    "cluster_small": ({"cluster": "g"}, "cr1"),
    # without small ivreg2's robust variance is HC0; its classical variance
    # divides by N and its cluster variance has no finite-sample factor,
    # which sp.iv(method='fuller') does not offer (small=False is refused)
    "robust_nosmall": ({"robust": "hc0"}, "hc0"),
}


@pytest.mark.parametrize("row", sorted(_FULLER))
@pytest.mark.parametrize("ident", ["over", "just"])
def test_fuller_matches_ivreg2(data, ref_stata, ident, row):
    opts, vce = _FULLER[row]
    fit = _quiet(sp.iv, IV[ident], data, method="fuller", **opts)
    _assert_stata(fit, ref_stata[f"ivreg2_{ident}_fuller_{row}"])
    _cell(fit, estimator="fuller", vce=vce, identification=ident, small="true")


@pytest.mark.parametrize("ident", ["over", "just"])
def test_ivreg_two_way_matches_ivreg2(data, ref_stata, ident):
    """sp.ivreg(cluster=[a, b]) reports the raw CGM matrix, as ivreg2 does."""
    fit = _quiet(sp.ivreg, IV[ident], data, cluster=["g", "yr"])
    _assert_stata(fit, ref_stata[f"ivreg2_{ident}_2sls_twoway_small"])
    _cell(fit, estimator="2sls", vce="cluster_multiway", identification=ident)


def _raw_two_way(data: pd.DataFrame, formula: str, method: str) -> np.ndarray:
    """ivreg2's two-way matrix, rebuilt from three one-way sp.iv fits.

    Each one-way variance carries G/(G-1) (N-1)/(N-K); undo it, add the g and
    yr meats, subtract the g x yr meat and apply G_min/(G_min-1) (N-1)/(N-K).
    """
    df = data.assign(g_yr=data["g"] * 100 + data["yr"])
    n = len(df)
    parts = {}
    for c in ("g", "yr", "g_yr"):
        fit = _quiet(sp.iv, formula, df, method=method, cluster=c)
        k = len(fit.params)
        G = df[c].nunique()
        parts[c] = np.asarray(fit.data_info["var_cov"]) / (
            G / (G - 1) * (n - 1) / (n - k)
        )
    g_min = min(df["g"].nunique(), df["yr"].nunique())
    return (g_min / (g_min - 1) * (n - 1) / (n - k)) * (
        parts["g"] + parts["yr"] - parts["g_yr"]
    )


@pytest.mark.parametrize("method", ["2sls", "liml", "fuller"])
@pytest.mark.parametrize("ident", ["over", "just"])
def test_iv_multiway_is_the_psd_projection_of_ivreg2(data, ref_stata, ident, method):
    """T4: ivreg2 reports the raw inclusion-exclusion matrix; sp.iv zeroes
    its negative eigenvalues (fixest's fix; 2SLS matches fixest exactly in
    test_iv_2sls_matches_r)."""
    raw = _raw_two_way(data, IV[ident], method)
    fit = _quiet(sp.iv, IV[ident], data, method=method, cluster=["g", "yr"])
    names = list(fit.params.index)
    ref = ref_stata[f"ivreg2_{ident}_{method}_twoway_small"]
    want = np.array([ref["se__cons" if t == "Intercept" else f"se_{t}"] for t in names])
    np.testing.assert_allclose(np.sqrt(np.diag(raw)), want, rtol=1e-9)
    evals, evecs = np.linalg.eigh(0.5 * (raw + raw.T))
    assert evals.min() < 0  # why the two conventions differ on this data
    proj = evecs @ np.diag(np.maximum(evals, 0.0)) @ evecs.T
    np.testing.assert_allclose(
        fit.std_errors[names].to_numpy(), np.sqrt(np.diag(proj)), rtol=1e-9
    )
    _cell(fit, estimator=method, vce="cluster_multiway", identification=ident)


@pytest.mark.parametrize("ident", ["over", "just"])
def test_ivreg_conley_matches_acreg(data, ref_stata, ident):
    fit = _quiet(sp.ivreg, IV[ident], data, vce="conley", **CONLEY)
    _assert_stata(fit, ref_stata[f"acreg_{ident}_conley"])
    _cell(fit, estimator="2sls", vce="conley", identification=ident)


@pytest.mark.parametrize("ident", ["over", "just"])
def test_ivreg_wild_matches_enumerated_boottest(data, ref_stata, ident):
    """WRE on the endogenous coefficient, all 2^12 sign vectors: the p-value
    equals boottest's exactly; the SE is ivreg2's one-way cluster SE."""
    fit = _quiet(
        sp.ivreg, IV[ident], data, vce="wild", cluster="g", wild_reps=4096, seed=1
    )
    names = list(fit.params.index)
    p = float(np.asarray(fit.pvalues, dtype=float)[names.index("d")])
    assert p == ref_stata[f"iv_wild_{ident}_d"]["p"]
    assert 0.1 < p < 0.9
    ref = ref_stata[f"ivreg2_{ident}_2sls_cluster_small"]
    np.testing.assert_allclose(fit.params["d"], ref["b_d"], rtol=RTOL)
    np.testing.assert_allclose(fit.std_errors["d"], ref["se_d"], rtol=RTOL)
    _cell(fit, estimator="2sls", vce="wild", identification=ident)


# --------------------------------------------------------------------------- #
#  Negative variances: missing, never zero
# --------------------------------------------------------------------------- #


def test_ivreg_conley_negative_variance_is_nan_as_in_acreg(ref_stata):
    """Correctness fix: a negative Conley variance used to be clamped to 0,
    which reported se(d) = 0, t = inf, p = 0; acreg reports it as missing."""
    from statspai.inference.iv_wild import iv_conley_vcov

    df = pd.read_csv(_FIX / "vce_menu_nonpsd_conley.csv")
    formula = "y ~ (d ~ z) + x + C(county)"
    ref = ref_stata["nonpsd_conley"]
    with pytest.warns(RuntimeWarning, match="non-positive-definite"):
        fit = sp.ivreg(formula, df, vce="conley", **dict(CONLEY, conley_cutoff=500))
    np.testing.assert_allclose(fit.params["d"], ref["b_d"], rtol=RTOL)
    assert ref["se_d_missing"] == 1 and np.isnan(fit.std_errors["d"])
    assert np.isnan(
        np.asarray(fit.pvalues, dtype=float)[list(fit.params.index).index("d")]
    )
    np.testing.assert_allclose(fit.std_errors["x"], ref["se_x"], rtol=RTOL)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cv = iv_conley_vcov(sp.iv(formula, df), df, "lat", "lon", 500)
    V = pd.DataFrame(cv["vcov"], index=cv["var_names"], columns=cv["var_names"])
    np.testing.assert_allclose(V.loc["d", "d"], ref["V_d"], rtol=RTOL)
    np.testing.assert_allclose(V.loc["x", "x"], ref["V_x"], rtol=RTOL)


def test_ivreg_two_way_negative_variance_is_nan_as_in_ivreg2(ref_stata):
    """Correctness fix: the raw two-way matrix had negative variances for d
    and the intercept, reported as se = 0; ivreg2 reports them as missing.
    sp.iv's projected matrix gives finite standard errors on the same data."""
    from statspai.inference.iv_wild import iv_twoway_vcov

    df = pd.read_csv(_FIX / "vce_menu_nonpsd_twoway.csv")
    formula = "y ~ (d ~ z) + x"
    ref = ref_stata["nonpsd_twoway"]
    with pytest.warns(RuntimeWarning, match="negative variances"):
        fit = sp.ivreg(formula, df, cluster=["a", "b"])
    np.testing.assert_allclose(fit.params["d"], ref["b_d"], rtol=RTOL)
    assert ref["se_d_missing"] == 1 and ref["se__cons_missing"] == 1
    assert np.isnan(fit.std_errors["d"]) and np.isnan(fit.std_errors["Intercept"])
    np.testing.assert_allclose(fit.std_errors["x"], ref["se_x"], rtol=RTOL)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tw = iv_twoway_vcov(_quiet(sp.iv, formula, df, cluster="a"), df, "a", "b")
        projected = sp.iv(formula, df, cluster=["a", "b"])
    V = pd.DataFrame(tw["vcov"], index=tw["var_names"], columns=tw["var_names"])
    for term, key in (("d", "V_d"), ("x", "V_x"), ("Intercept", "V__cons")):
        np.testing.assert_allclose(V.loc[term, term], ref[key], rtol=RTOL)
    assert np.isfinite(projected.std_errors).all()
