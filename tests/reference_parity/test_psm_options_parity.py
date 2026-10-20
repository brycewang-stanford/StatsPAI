"""The option grid of ``sp.match`` against Stata 18 and R ``Matching``.

Data: ``_fixtures/psm_options_data.csv`` (500 rows, 190 treated; written by
``_generate_psm_options_data.py``). The covariates are continuous, so the
propensity score and the covariate distances have no ties and every
comparison is about the option, not about tie-breaking. Stata reads the
file with ``asdouble``; both sides compute on the same doubles.

References (each entry records its command):

* ``psm_options_Stata.json`` -- ``_generate_psm_options_stata.do``:
  ``psmatch2`` (Leuven and Sianesi; the reference for the psmatch2 SE, the
  Abadie-Imbens SE ``ai(J)``, kernel, radius and caliper conventions),
  ``teffects psmatch`` (the Abadie-Imbens 2016 estimated-score SE),
  ``teffects nnmatch`` and ``atts`` (Becker and Ichino).
* ``psm_options_R.json`` -- ``_generate_psm_options_R.R``:
  ``Matching::Match`` (bias adjustment; a second caliper reference).

Tolerance: rtol 1e-9 for every T2 row. The matches are discrete and the
outcome arithmetic is the same; what differs is the logit fit (both sides
converge to about 1e-12 in the score). The largest gap observed is 3e-14
(the smooth kernels, which see the score itself).

Two psmatch2 standard errors are reproduced only by rebuilding psmatch2's
own rule from our quantities, and are recorded as reference differences
(T4), not parity: with ``caliper()`` or ``noreplacement``, psmatch2 runs
the *same-arm* search behind ``ai(J)`` with that caliper and without
replacement too, and a unit left without a same-arm neighbour gets
``sigma^2 = y^2 / 2`` (its ``_self_y`` stays 0). Abadie and Imbens (2006)
estimate ``sigma^2(X_i)`` from the ``J`` nearest same-arm units with no
such restriction, which is what ``sp.match`` does.

Open item: ``Matching::Match(caliper=)`` drops the same five treated units
and gives the same estimate, but its sample-ATT variance (0.29037 at
caliper 0.01) is neither ours (0.30474) nor psmatch2's (0.31863), and the
variants tried (same-arm pool restricted to matched units, the caliper
applied to the same-arm search with three fallbacks) do not reproduce it.
Matching is GPL-3 and is run as a black box only; the gap is unexplained.
"""

from __future__ import annotations

import json
import pathlib
import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import MethodIncompatibility

_FIX = pathlib.Path(__file__).parent / "_fixtures"
X = ["x1", "x2", "x3", "x4"]
RTOL = 1e-9


@pytest.fixture(scope="module")
def df():
    return pd.read_csv(_FIX / "psm_options_data.csv")


@pytest.fixture(scope="module")
def stata():
    return json.loads((_FIX / "psm_options_Stata.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def rref():
    return json.loads((_FIX / "psm_options_R.json").read_text(encoding="utf-8"))


def _fit(df, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.match(df, y="y", treat="d", covariates=X, **kw)


# fixture key -> sp.match options
CASES = {
    **{f"psmatch2_nn{k}_ai1": {"n_matches": k} for k in range(1, 11)},
    **{
        f"teffects_psmatch_atet_nn{k}": {
            "n_matches": k,
            "se_method": "abadie_imbens_2016",
        }
        for k in range(1, 11)
    },
    "psmatch2_nn1_ai2": {"ai_matches": 2},
    "psmatch2_nn1": {"se_method": "psmatch2"},
    "psmatch2_nn1_cal0p01": {"caliper": 0.01, "se_method": "psmatch2"},
    "psmatch2_nn1_cal0p005": {"caliper": 0.005, "se_method": "psmatch2"},
    # psmatch2 matches the treated in ascending order of the score
    "psmatch2_nn1_norepl": {
        "replace": False,
        "m_order": "smallest",
        "se_method": "psmatch2",
    },
    "psmatch2_mahal_nn1_ai1": {"distance": "mahalanobis"},
    "psmatch2_mahal_nn3_ai1": {"distance": "mahalanobis", "n_matches": 3},
    **{
        f"psmatch2_kernel_{k}_bw0p06": {"method": "kernel", "kernel": k}
        for k in ("epan", "normal", "biweight", "uniform", "tricube")
    },
    "psmatch2_kernel_epan_bw0p02": {"method": "kernel", "bwidth": 0.02},
    "psmatch2_radius_cal0p02": {"method": "radius", "caliper": 0.02},
    "psmatch2_radius_cal0p05": {"method": "radius", "caliper": 0.05},
    "teffects_psmatch_ate_nn3": {
        "n_matches": 3,
        "estimand": "ATE",
        "se_method": "abadie_imbens_2016",
    },
    # teffects reads caliper() as a condition that 0.2 meets
    "teffects_psmatch_atet_nn1_cal0p2": {
        "caliper": 0.2,
        "se_method": "abadie_imbens_2016",
    },
    "teffects_psmatch_ate_nn1_cal0p2": {
        "caliper": 0.2,
        "estimand": "ATE",
        "se_method": "abadie_imbens_2016",
    },
    "teffects_nnmatch_atet_nn1": {"method": "nnmatch"},
    "teffects_nnmatch_atet_nn4": {"method": "nnmatch", "n_matches": 4},
    "teffects_nnmatch_atet_nn5": {"method": "nnmatch", "n_matches": 5},
    "teffects_nnmatch_ate_nn4": {
        "method": "nnmatch",
        "n_matches": 4,
        "estimand": "ATE",
    },
    "teffects_nnmatch_ate_nn5": {
        "method": "nnmatch",
        "n_matches": 5,
        "estimand": "ATE",
    },
    "teffects_nnmatch_atet_vnn3": {"method": "nnmatch", "vce_nn": 3},
    "teffects_nnmatch_ate_nn4_vnn3": {
        "method": "nnmatch",
        "n_matches": 4,
        "estimand": "ATE",
        "vce_nn": 3,
    },
    "teffects_nnmatch_atet_nn4_cal3": {
        "method": "nnmatch",
        "n_matches": 4,
        "caliper": 3.0,
    },
    "teffects_nnmatch_ate_nn1_cal3": {
        "method": "nnmatch",
        "estimand": "ATE",
        "caliper": 3.0,
    },
    # five quantile classes of the score (sp.match's default strata)
    "atts_quintiles": {"method": "stratify"},
}


@pytest.mark.parametrize("key", sorted(CASES))
def test_estimate_and_se_match_stata(df, stata, key):
    res = _fit(df, **CASES[key])
    ref = stata[key]
    assert res.estimate == pytest.approx(ref["est"], rel=RTOL), ref["cmd"]
    assert res.se == pytest.approx(ref["se"], rel=RTOL), ref["cmd"]


def test_every_stata_fit_is_compared(stata):
    fits = {k for k, v in stata.items() if isinstance(v, dict) and "cmd" in v}
    t4 = {
        "psmatch2_nn1_cal0p01_ai1",
        "psmatch2_nn1_cal0p005_ai1",
        "psmatch2_nn1_norepl_ai1",
    }
    assert fits - t4 == set(CASES)


# --------------------------------------------------------------- caliper ----
@pytest.mark.parametrize("cal, tag", [(0.01, "0p01"), (0.005, "0p005")])
def test_caliper_se_counts_only_matched_treated(df, stata, cal, tag):
    """⚠️ correctness fix: a treated unit the caliper leaves without a match
    is not in the estimate, and is not in its standard error either.

    psmatch2 sets its _support to 0. Through 1.39.6 both standard errors
    still counted it, in N1 and with a variance term: at caliper 0.01 the
    psmatch2 SE was 0.354559 (Stata 0.363576) and the Abadie-Imbens SE
    0.298989 (now 0.304739).
    """
    assert stata[f"psmatch2_nn1_cal{tag}_n_on_support"] < 190
    res = _fit(df, caliper=cal)
    md = res.matched_data
    t = md["d"].to_numpy()
    matched = (t == 1) & (md["_nn"].to_numpy() > 0)
    assert int(matched.sum()) == stata[f"psmatch2_nn1_cal{tag}_n_on_support"]
    # The estimate is sum_i c_i y_i over the matched treated and their
    # matches; the conditional variance is sum_i c_i^2 sigma_i^2 with the
    # same c_i (Abadie and Imbens 2006, eq. 14).
    y = md["y"].to_numpy()
    w = np.nan_to_num(md["_weight"].to_numpy(dtype=float))
    n1 = matched.sum()
    c = np.where(t == 1, matched, -w) / n1
    assert float(c @ y) == pytest.approx(res.estimate, rel=1e-12)
    sigma2 = _same_arm_sigma2(y, t, md["_pscore"].to_numpy(), caliper=None)
    assert res.se == pytest.approx(np.sqrt(np.sum(c**2 * sigma2)), rel=1e-12)


@pytest.mark.parametrize("cal, tag", [(0.01, "0p01"), (0.005, "0p005")])
def test_psmatch2_caliper_ai_se_is_its_same_arm_caliper(df, stata, cal, tag):
    """T4: psmatch2's ai(1) SE under caliper() rebuilt from our quantities.

    psmatch2 applies the caliper to the same-arm search too, and a unit
    with no same-arm unit inside it gets sigma^2 = y^2 / 2 -- a variance
    that moves with the location of y. Our SE uses the nearest same-arm
    unit; the estimates agree to the last digit.
    """
    ref = stata[f"psmatch2_nn1_cal{tag}_ai1"]
    res = _fit(df, caliper=cal)
    assert res.estimate == pytest.approx(ref["est"], rel=RTOL)
    md = res.matched_data
    t = md["d"].to_numpy()
    y = md["y"].to_numpy()
    matched = (t == 1) & (md["_nn"].to_numpy() > 0)
    w = np.nan_to_num(md["_weight"].to_numpy(dtype=float))
    c = np.where(t == 1, matched, -w) / matched.sum()
    sigma2 = _same_arm_sigma2(y, t, md["_pscore"].to_numpy(), caliper=cal)
    assert np.sqrt(np.sum(c**2 * sigma2)) == pytest.approx(ref["se"], rel=RTOL)
    assert abs(res.se / ref["se"] - 1) > 1e-3


def test_psmatch2_noreplacement_ai_se_draws_same_arm_without_replacement(df, stata):
    """T4: psmatch2 ai(1) with noreplacement also takes each unit's
    same-arm neighbour without replacement, in ascending score order. Its
    SE is rebuilt from our matches by that rule; ours uses the nearest
    same-arm unit."""
    ref = stata["psmatch2_nn1_norepl_ai1"]
    res = _fit(df, replace=False, m_order="smallest")
    assert res.estimate == pytest.approx(ref["est"], rel=RTOL)
    md = res.matched_data
    t = md["d"].to_numpy()
    y = md["y"].to_numpy()
    ps = md["_pscore"].to_numpy()
    self_y = np.zeros(len(y))
    for g in (0, 1):
        arm = np.flatnonzero(t == g)
        arm = arm[np.argsort(ps[arm], kind="stable")]
        used = np.zeros(len(y), dtype=bool)
        for i in arm:
            pool = arm[(arm != i) & ~used[arm]]
            k = pool[np.argmin(np.abs(ps[i] - ps[pool]))]
            self_y[i], used[k] = y[k], True
    w = np.nan_to_num(md["_weight"].to_numpy(dtype=float))
    c = np.where(t == 1, 1.0, -w) / (t == 1).sum()
    rebuilt = np.sqrt(np.sum(c**2 * 0.5 * (y - self_y) ** 2))
    assert rebuilt == pytest.approx(ref["se"], rel=RTOL)
    assert abs(res.se / ref["se"] - 1) > 1e-3


def _same_arm_sigma2(y, t, ps, caliper):
    """sigma^2 = (y - y_j)^2 / 2 with j the nearest same-arm unit; with a
    caliper, psmatch2's rule (no unit within it: y_j = 0)."""
    out = np.empty(len(y))
    for g in (0, 1):
        arm = np.flatnonzero(t == g)
        for i in arm:
            pool = arm[arm != i]
            d = np.abs(ps[i] - ps[pool])
            j = int(np.argmin(d))
            yj = 0.0 if caliper is not None and not d[j] < caliper else y[pool[j]]
            out[i] = 0.5 * (y[i] - yj) ** 2
    return out


def test_caliper_estimate_matches_matching_package(df, rref):
    for cal, tag in ((0.01, "0p01"), (0.005, "0p005")):
        ref = rref[f"match_nn1_cal{tag}"]
        res = _fit(df, caliper=cal)
        assert res.estimate == pytest.approx(ref["est"], rel=RTOL)
        assert res.model_info["n_treated_unmatched"] == ref["n_dropped"]


@pytest.mark.parametrize("estimand", ["ATT", "ATE"])
def test_ai2016_refuses_where_teffects_refuses(df, stata, estimand):
    """⚠️ correctness fix: teffects psmatch reads caliper() as a condition
    on every observation, of either arm: max(m, h) opposite-arm units and h
    same-arm units within it, or it stops. At 0.1 one treated unit has a
    single control within reach. Through 1.39.6 the ATT ignored the
    condition and the ATE asked for m opposite-arm units only, so both
    returned a standard error Stata does not report (0.251449 / 0.197897).
    """
    tag = "atet" if estimand == "ATT" else "ate"
    assert stata[f"teffects_psmatch_{tag}_nn1_cal0p1_rc"] == 459
    with pytest.raises(MethodIncompatibility, match="within caliper"):
        _fit(df, caliper=0.1, estimand=estimand, se_method="abadie_imbens_2016")


# ------------------------------------------------------ bias correction ----
def test_bias_correction_matches_matching_package(df, rref):
    """⚠️ correctness fix: the bias-adjustment regression is fitted on the
    units used as matches, weighted by the number of times each is used
    (Abadie and Imbens 2011; Matching::Match and teffects nnmatch do the
    same). Through 1.39.6 it was an unweighted fit on every control:
    1.740076 here, against Matching's 1.700019."""
    ref = rref["match_nn1_biasadj"]
    res = _fit(df, bias_correction=True)
    assert res.estimate == pytest.approx(ref["est"], rel=RTOL)
    # the sample-ATT variance does not involve the adjustment
    assert res.se == pytest.approx(ref["se"], rel=RTOL)
    plain = rref["match_nn1"]
    base = _fit(df)
    assert base.estimate == pytest.approx(plain["est"], rel=RTOL)
    assert base.se == pytest.approx(plain["se"], rel=RTOL)


def test_bias_correction_is_the_k_weighted_regression(df):
    res = _fit(df, bias_correction=True)
    md = _fit(df).matched_data
    t = md["d"].to_numpy()
    y = md["y"].to_numpy()
    Z = md[X].to_numpy()
    K = np.nan_to_num(md["_weight"].to_numpy(dtype=float))
    use = (t == 0) & (K > 0)
    A = np.column_stack([np.ones(use.sum()), Z[use]]) * np.sqrt(K[use])[:, None]
    beta = np.linalg.lstsq(A, y[use] * np.sqrt(K[use]), rcond=None)[0][1:]
    pos = {int(v): k for k, v in enumerate(md["_id"].to_numpy())}
    tr = np.flatnonzero(t == 1)
    j = np.array([pos[int(v)] for v in md["_n1"].to_numpy()[tr]])
    want = np.mean(y[tr] - y[j] - (Z[tr] - Z[j]) @ beta)
    assert res.estimate == pytest.approx(want, rel=1e-12)


# -------------------------------------------------------- nnmatch caliper ---
def test_nnmatch_caliper_shortfall_warns(df, stata):
    """teffects nnmatch stops when a unit has fewer than nneighbor()
    matches within the caliper; sp.match keeps the ones it has and now
    says so (the estimate is not one Stata reports)."""
    assert stata["teffects_nnmatch_atet_nn4_cal2_rc"] == 459
    with pytest.warns(UserWarning, match="fewer than n_matches=4"):
        res = sp.match(
            df, y="y", treat="d", covariates=X, method="nnmatch",
            n_matches=4, caliper=2.0,
        )  # fmt: skip
    assert res.model_info["matches_min"] < 4


def test_no_replacement_in_data_order_matches_matching_package(df, rref):
    # Match(replace = FALSE) takes the treated in data order; it reports
    # no standard error for matching without replacement.
    ref = rref["match_nn1_norepl"]
    res = _fit(df, replace=False, m_order="data")
    assert res.estimate == pytest.approx(ref["est"], rel=RTOL)
    assert ref["se"] is None


# ------------------------------------------- radius caliper in SD units ----
@pytest.fixture(scope="module")
def stata_radius_sd():
    return json.loads((_FIX / "psm_radius_sd_Stata.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("c, tag", [(0.1, "0p1"), (0.25, "0p25")])
def test_radius_caliper_sd_is_radius_at_c_times_score_sd(df, stata_radius_sd, c, tag):
    """⚠️ correctness fix: method='radius' used to read caliper on the raw
    scale whatever caliper_scale said. With caliper_scale='sd' the radius is
    caliper * sd(score) over the full sample (ddof 1), as for nearest-
    neighbour matching. psmatch2 has no SD caliper; the reference runs
    ``psmatch2 , radius caliper(c * sd(_pscore))`` (black box)."""
    ref = stata_radius_sd[f"radius_sd{tag}"]
    res = _fit(df, method="radius", caliper=c, caliper_scale="sd")
    assert res.estimate == pytest.approx(ref["est"], rel=RTOL)
    assert res.se == pytest.approx(ref["se"], rel=RTOL)
    # the same call on the raw scale at Stata's width is the same estimator
    raw = _fit(df, method="radius", caliper=ref["width"])
    assert res.estimate == pytest.approx(raw.estimate, rel=1e-12)
    # the fixture width is c times the score SD Stata reports
    assert ref["width"] == pytest.approx(c * stata_radius_sd["pscore_sd"], rel=1e-15)


def test_psmatch2_kernel_ignores_caliper(df, stata_radius_sd):
    """psmatch2 kernel matching does not read caliper() (black box: the ATT
    is unchanged at caliper(0.001)); sp.match kernel matching agrees."""
    ref = stata_radius_sd["kernel_epan_bw0p06"]
    assert ref["est_with_caliper0p001"] == ref["est"]
    res = _fit(df, method="kernel", bwidth=0.06)
    assert res.estimate == pytest.approx(ref["est"], rel=RTOL)
