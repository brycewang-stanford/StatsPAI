"""``sp.suest`` with factor regressors, absorbed fixed effects and two-way
clustering.

Motivation: a test of rational expectations compares the coefficient of the
same regressor across a realized-outcome and a forecast regression, each
with categorical controls, standard errors two-way clustered by month and
respondent. ``sp.suest`` took plain columns and one cluster variable.

Reference (Stata 18, ``_fixtures/_generate_suest_fe_Stata.do`` on
``_fixtures/suest_fe_data.csv``, an unbalanced respondent x month panel):

A. ``suest`` after ``regress`` with ``i.`` dummies, vce(robust) and
   vce(cluster r | m): coefficients and the full joint covariance to 1e-10
   (observed 7e-15).
B. Stata ``suest`` has no multi-way clustering and does not take absorbed
   effects, so the reference is the stacked regression it is equivalent to:
   one row per equation and observation, every regressor and fixed effect
   interacted with the equation, clustered on the same variables (rows of
   one respondent-month share every cluster). reghdfe 6.13.1 reports
   ``V_suest * (N_s - 1) / (N_s - df_m - df_a)`` (``N_s`` stacked rows, the
   absorbed levels not nested in a cluster), Stata 18 ``regress, vce(cluster
   r m)`` ``V_suest(per_term) * (N_s - 1) / (N_s - K_s)``. These factors are
   the whole difference; ``sp.suest`` keeps ``suest``'s ``G/(G-1)`` only.

When the inclusion-exclusion matrix has negative eigenvalues (cases B1, B2:
12 month clusters against 7 to 15 coefficients) all three set them to zero,
but reghdfe does it in the basis of its standardized regressors (each
divided by its standard deviation over the stacked sample) and on the slope
block alone, while ``sp.suest`` (like ``sp.hdfe_ols``) and ``regress`` do it
in the original units. The tests rebuild the unadjusted matrix from three
one-way fits and reproduce each program's adjustment from it.

Tolerances: rel 1e-9 on the FE cases (absorber run to 1e-12 here, reghdfe
to tol(1e-14); observed <= 2e-10); 1e-10 elsewhere.
"""

from __future__ import annotations

import json
import pathlib
import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp

_FIX = pathlib.Path(__file__).parent / "_fixtures"
R = json.loads((_FIX / "suest_fe_Stata.json").read_text(encoding="utf-8"))
EA = ["yr ~ x + w + i.f", "yf ~ x + C(f) + i.g"]
EB = ["yr ~ x + w | f", "yf ~ x + i.g | f + m"]
SL_A = ["yr:x", "yr:w", "yr:f::2", "yr:f::3", "yr:f::4"] + [
    "yf:x", "yf:f::2", "yf:f::3", "yf:f::4",
    "yf:g::20", "yf:g::30", "yf:g::40", "yf:g::50",
]  # fmt: skip
SL_B = ["yr:x", "yr:w", "yf:x", "yf:g::20", "yf:g::30", "yf:g::40", "yf:g::50"]


@pytest.fixture(scope="module")
def data():
    d = pd.read_csv(_FIX / "suest_fe_data.csv", float_precision="round_trip")
    d["rm"] = d.r * 100 + d.m  # the respondent-month intersection
    return d


def _V(key):
    v = np.asarray(R[key]["V"])
    k = int(round(np.sqrt(v.size)))
    return v.reshape(k, k)


def _q(key):
    """reghdfe's (N - 1) / (N - df_m - df_a), from its own e()."""
    r = R[key]
    return (r["N"] - 1) / (r["N"] - r["df_m"] - r["df_a"])


def _rel(a, b):
    return float(np.max(np.abs(np.asarray(a) - b)) / np.max(np.abs(b)))


def _clip(V):
    w, U = np.linalg.eigh((V + V.T) / 2)
    return (U * np.maximum(w, 0.0)) @ U.T


def _unadjusted_two_way(data, eqs, names):
    """``G_min/(G_min-1) (M_r + M_m - M_rm)`` from three one-way fits."""
    one = {c: sp.suest(data, eqs, cluster=c) for c in ("r", "m", "rm")}
    M = sum(
        s * one[c].vcov.loc[names, names].to_numpy() * (one[c].n_clusters - 1)
        / one[c].n_clusters
        for c, s in (("r", 1), ("m", 1), ("rm", -1))
    )  # fmt: skip
    g = min(one["r"].n_clusters, one["m"].n_clusters)
    return g / (g - 1) * M


@pytest.mark.parametrize(
    "key, cl", [("A_robust", None), ("A_cluster_r", "r"), ("A_cluster_m", "m")]
)
def test_factor_dummies_match_stata_suest(data, key, cl):
    r = sp.suest(data, EA, cluster=cl)
    assert len(r.params) == 15
    assert _rel(r.params, np.asarray(R[key]["b"])) < 1e-10
    assert _rel(r.vcov.to_numpy(), _V(key)) < 1e-10
    if key == "A_cluster_m":
        chi2 = r.test_equal("x")["chi2"]
        assert chi2 == pytest.approx(R["A_cluster_m_test_x"]["chi2"], rel=1e-10)


def test_two_way_equals_stacked_reghdfe_when_psd(data):
    r = sp.suest(data, ["yr ~ x", "yf ~ x"], cluster=["r", "m"])
    assert r.cluster_info["n_negative_eigenvalues"] == 0
    assert r.cluster_info["n_clusters"] == [60, 12] and r.n_clusters == 12
    V = r.vcov.loc[["yr:x", "yf:x"], ["yr:x", "yf:x"]].to_numpy()
    assert _rel(V * _q("B0_reghdfe_rm"), _V("B0_reghdfe_rm")) < 1e-10
    assert _rel(r.params[["yr:x", "yf:x"]], np.asarray(R["B0_reghdfe_rm"]["b"])) < 1e-12


def test_per_term_equals_stacked_regress(data):
    with pytest.warns(RuntimeWarning, match="positive semi-definite"):
        r = sp.suest(data, EA, cluster=["r", "m"], cluster_df="per_term")
    # regress: the 13 slopes and two equation intercepts, (N-1)/(N-K) on top;
    # both programs set the negative eigenvalues to zero in these units
    n, k = R["B1_regress_rm"]["N"], len(r.params)
    V = r.vcov.loc[SL_A, SL_A].to_numpy() * (n - 1) / (n - k)
    assert _rel(V, _V("B1_regress_rm")) < 1e-10


@pytest.mark.parametrize(
    "case, eqs, names", [("B1", EA, SL_A), ("B2", EB, SL_B)]
)  # fmt: skip
def test_min_two_way_and_reghdfe_differ_only_in_the_psd_basis(data, case, eqs, names):
    raw = _unadjusted_two_way(data, eqs, names)
    with pytest.warns(RuntimeWarning, match="positive semi-definite"):
        r = sp.suest(data, eqs, cluster=["r", "m"])
    assert r.cluster_info["n_negative_eigenvalues"] > 0
    # sp.suest: zero the negative eigenvalues of the matrix as it is (for B1
    # the intercepts are part of it, so compare the B2 matrix whole)
    if case == "B2":
        assert _rel(r.vcov.to_numpy(), _clip(raw)) < 1e-9
    # reghdfe: the same in the standardized basis, slopes only
    s = np.asarray(R[f"sd_{case}"])
    reghdfe = _clip(raw * np.outer(s, s)) / np.outer(s, s)
    key = f"{case}_reghdfe_rm"
    assert _rel(reghdfe * _q(key), _V(key)) < 1e-9
    # and that is not a rounding matter: in the original basis it is off
    assert _rel(_clip(raw) * _q(key), _V(key)) > 1e-3


@pytest.mark.parametrize("key, cl", [("B2_reghdfe_r", "r"), ("B2_reghdfe_id", None)])
def test_fixed_effects_one_way_equals_stacked_reghdfe(data, key, cl):
    # robust suest = the stacked regression clustered on the original row
    r = sp.suest(data, EB, cluster=cl)
    assert list(r.params.index) == SL_B
    assert r.absorbed == {"yr": ["f"], "yf": ["f", "m"]}
    assert _rel(r.params, np.asarray(R[key]["b"])) < 1e-9
    assert _rel(r.vcov.to_numpy() * _q(key), _V(key)) < 1e-9


# ---------------------------------------------------------------- identities


def test_absorbing_a_factor_equals_its_dummies(data):
    # FWL: the slope's influence function is the same with the dummies in the
    # regression or swept out, so the joint covariance block is too (before
    # any eigenvalue adjustment, which acts on the whole matrix)
    for cl in (None, "r", "rm"):
        a = sp.suest(data, ["yr ~ x + w | f", "yf ~ x | g"], cluster=cl)
        b = sp.suest(data, ["yr ~ x + w + i.f", "yf ~ x + i.g"], cluster=cl)
        names = ["yr:x", "yr:w", "yf:x"]
        np.testing.assert_allclose(a.params[names], b.params[names], rtol=1e-10)
        np.testing.assert_allclose(
            a.vcov.loc[names, names], b.vcov.loc[names, names], rtol=1e-9
        )


@pytest.mark.filterwarnings("ignore:sp.suest. the multi-way")
def test_c_and_i_spellings_and_vce_string_agree(data):
    a = sp.suest(data, ["yr ~ x + C(f)", "yf ~ x + i.g"], cluster=["r", "m"])
    b = sp.suest(data, ["yr ~ x + i.f", "yf ~ x + C(g)"], vce="cluster r m")
    np.testing.assert_allclose(a.vcov.to_numpy(), b.vcov.to_numpy(), rtol=1e-12)
    assert b.vce == "cluster(r m)"


def test_one_cluster_in_a_list_is_one_way(data):
    a = sp.suest(data, EA, cluster=["r"])
    b = sp.suest(data, EA, cluster="r")
    np.testing.assert_allclose(a.vcov.to_numpy(), b.vcov.to_numpy(), rtol=1e-14)


# ---------------------------------------------------------------- boundaries


def test_singletons_are_dropped_like_reghdfe(data):
    d = data.copy()
    d["h"] = np.arange(len(d)) // 4  # 484 rows: 121 groups of four
    d.loc[0, "h"] = -1  # a fixed-effect level with one row
    r = sp.suest(d, ["yr ~ x | h", "yf ~ x"], cluster="r")
    assert r.n_obs["yr"] == len(d) - 1


def test_collinear_factor_is_omitted_with_a_warning(data):
    # f is constant within respondent: i.f is absorbed by a respondent FE
    with pytest.warns(UserWarning, match="omitted"):
        r = sp.suest(data, ["yr ~ x + i.f | r", "yf ~ x"], cluster="m")
    assert r.omitted == {"yr": ["f::2", "f::3", "f::4"]}
    assert "yr:f::2" not in r.params.index
    assert "omitted (collinear) f::2" in r.summary()


def test_bad_inputs_are_refused(data):
    with pytest.raises(sp.MethodIncompatibility, match="cluster_df"):
        sp.suest(data, EA, cluster=["r", "m"], cluster_df="max")
    with pytest.raises(sp.MethodIncompatibility, match="C\\(x\\)"):
        sp.suest(data, ["yr ~ C(f, Treatment(2))", "yf ~ x"])
    with pytest.raises(sp.MethodIncompatibility, match="empty"):
        sp.suest(data, EA, cluster=[])
    d = data.assign(one=1)
    with pytest.raises(sp.MethodIncompatibility, match="single value"):
        sp.suest(d, EA, cluster=["r", "one"])


def test_plain_tuple_equations_keep_any_column_name(data):
    d = data.rename(columns={"x": "x rate"})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        r = sp.suest(d, [("yr", ["x rate", "w"]), ("yf", ["x rate"])], cluster="r")
    assert list(r.params.index[:3]) == ["yr:x rate", "yr:w", "yr:_cons"]
