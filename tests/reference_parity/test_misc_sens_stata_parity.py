"""Round-2 ``misc_sens`` family vs Stata 18.

Reference: ``_fixtures/misc_sens_stata.json`` written by
``_fixtures/_generate_misc_sens_stata.do`` (Stata 18 MP, ``set type
double``; ietoolkit 7.5 / leebounds 1.5 from SSC in a private ado directory)
on the CSVs of ``_fixtures/_generate_misc_sens_data.py`` and the five
``mice`` imputations the R generator writes.

Conventions every number below depends on:

* ``mi estimate: regress``: Barnard-Rubin small-sample df with
  ``nu_c = e(df_r)``, t(df) p-values and CIs -- identical to R ``mice``.
  Its FMI is Barnard & Rubin's (1999) small-sample
  ``1 - [l(df)/l(nu_c)] U/T``, ``l(u) = (u+1)/(u+3)``, which StatsPAI
  returns as ``fmi_barnard_rubin`` (``fmi`` is mice's formula).
* ``regress, vce(robust)`` = HC1 with t(N - k); ``vce(cluster)`` =
  ``G/(G-1) (N-1)/(N-k)`` with t(G - 1): the ``sp.ancova`` / ``sp.negd`` /
  ``sp.subgroup_analysis`` defaults.
* ``testparm`` reports ``F = W / q`` on (q, N - k); ``subgroup_analysis``
  reports both ``chi2 = W`` (asymptotic) and ``F`` / ``pvalue_F``.
* ``tabulate, chi2`` is Pearson without Yates (``attrition_test(...,
  correction=False)``).
* ``iebaltab``: pair test = OLS t of the variable on a group dummy
  (``balance_check(equal_var=True)``), difference = control - treatment;
  ``ftest`` = F of the treatment dummy on all balance variables.
* ``leebounds`` as shipped rounds its trimming thresholds through a local
  macro: here the lower bound keeps ``floor((1-q) n)`` observations and the
  upper bound takes the fractional tie branch, which is
  ``trimming='exact'`` (bound and analytic SE). StatsPAI's default is Lee's
  sample-quantile rule (round-1 decision, ``test_teffects_R_parity``);
  ``trimming='exact'`` is also checked against known truth in
  ``test_lee_bounds_exact_threshold.py``.

Tolerance 1e-10 relative unless stated.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.imputation.mice import MICEResult

_FIX = Path(__file__).parent / "_fixtures"
ST = json.loads((_FIX / "misc_sens_stata.json").read_text(encoding="utf-8"))
OVB = pd.read_csv(_FIX / "misc_sens_ovb.csv")
ATTR = pd.read_csv(_FIX / "misc_sens_attr.csv")

TIGHT = 1e-10


def _close(ours, ref, rtol=TIGHT, atol=0.0):
    np.testing.assert_allclose(
        np.asarray(ours, dtype=float),
        np.asarray(ref, dtype=float),
        rtol=rtol,
        atol=atol,
    )


# --------------------------------------------------------------------------
# mi estimate
# --------------------------------------------------------------------------


def test_mi_pooling_matches_stata_mi_estimate():
    long = pd.read_csv(_FIX / "misc_sens_mi_imputed.csv")
    ds = [
        long.loc[long["imp"] == i, ["y", "x1", "x2", "x3"]].reset_index(drop=True)
        for i in range(1, 6)
    ]
    c = sp.mi_estimate(
        MICEResult(ds, 5, len(ds[0]), {}, [], {}, True),
        sp.regress,
        formula="y ~ x1 + x2 + x3",
    )
    order = [1, 2, 3, 0]  # Stata: x1 x2 x3 _cons
    _close(np.asarray(c["params"])[order], ST["mi_b"], rtol=1e-12)
    _close(np.asarray(c["se"])[order], ST["mi_se"], rtol=1e-12)
    _close(np.asarray(c["df"])[order], ST["mi_df"], rtol=1e-12)
    _close(np.asarray(c["pvalues"])[order], ST["mi_p"], rtol=1e-9)
    _close(np.asarray(c["ci_lower"])[order], ST["mi_ci_lo"], rtol=1e-10)
    _close(np.asarray(c["ci_upper"])[order], ST["mi_ci_hi"], rtol=1e-10)
    _close(np.asarray(c["fmi_barnard_rubin"])[order], ST["mi_fmi"], rtol=1e-12)
    # mice's FMI is a different documented quantity (larger here).
    assert np.all(np.asarray(c["fmi"])[order] > np.asarray(ST["mi_fmi"]))


# --------------------------------------------------------------------------
# subgroup_analysis
# --------------------------------------------------------------------------


@pytest.mark.parametrize("vce, robust", [("robust", "hc1"), ("ols", "nonrobust")])
def test_subgroup_analysis_matches_regress_and_testparm(vce, robust):
    r = sp.subgroup_analysis(
        OVB, "y ~ d + x1 + x2", "d", {"X3": "x3", "G3": "g3"}, robust=robust
    )
    ov = ST[f"sub_{vce}_overall"]
    _close(r.overall_estimate, ov[0])
    _close(r.overall_se, ov[1])
    df = r.results_df.set_index("label")
    for g, name in (("x3", "X3"), ("g3", "G3")):
        levels = sorted(OVB[g].unique())
        for v in levels:
            ref = ST[f"sub_{vce}_{g}_{v}"]
            row = df.loc[f"{name}: {v}"]
            _close(
                [
                    row["estimate"],
                    row["se"],
                    row["pvalue"],
                    row["ci_lower"],
                    row["ci_upper"],
                ],
                ref[:5],
                rtol=1e-9,
            )
            assert row["nobs"] == ref[5]
        het = r.het_tests[name]
        F, q, df_r, p = ST[f"sub_{vce}_het_{g}"]
        _close(het["F"], F)
        _close(het["chi2"], q * F)
        assert het["df"] == q and het["df_resid"] == df_r
        _close(het["pvalue_F"], p, rtol=1e-9)


# --------------------------------------------------------------------------
# ancova / negd
# --------------------------------------------------------------------------


def _row(res):
    return [res.estimate, res.se, res.pvalue, res.ci[0], res.ci[1]]


def test_ancova_matches_regress_robust():
    r = sp.ancova(OVB, "post", "grp", ["pre", "x2", "region"])
    _close(_row(r), ST["ancova_hc1"][:5], rtol=1e-9)
    assert r.n_obs == ST["ancova_hc1"][5]


def test_ancova_matches_regress_cluster():
    r = sp.ancova(OVB, "post", "grp", ["pre", "x2", "region"], cluster="cl")
    _close(_row(r), ST["ancova_cluster"][:5], rtol=1e-9)


def test_negd_ancova_and_change_score_match_regress():
    _close(
        _row(sp.negd(OVB, "grp", pre="pre", post="post")), ST["negd_ancova"], rtol=1e-9
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)
        r = sp.negd(
            OVB, "grp", pre="pre", post="post", covariates=["x2"], method="change_score"
        )
    _close(_row(r), ST["negd_change"], rtol=1e-9)


# --------------------------------------------------------------------------
# attrition
# --------------------------------------------------------------------------


def test_attrition_test_matches_tabulate_and_regress():
    r = sp.attrition_test(ATTR, "treat", "obs", ["age", "inc"], correction=False)
    _close([r.diff_test_stat, r.diff_p_value], ST["attr_chi2"])
    t = r.covariate_tests.set_index("variable")
    for v in ("age", "inc"):
        _close(t.loc[v, ["coef", "se", "p_value"]].to_numpy(float), ST[f"attr_reg_{v}"])


def test_attrition_bounds_equals_lee_bounds_and_reconstructs_shipped_leebounds():
    r = sp.attrition_bounds(ATTR, "y", "treat", "obs")
    lb = sp.lee_bounds(ATTR, "y", "treat", "obs", se_method="analytic")
    assert r["lower_bound"] == lb.model_info["lower_bound"]
    assert r["upper_bound"] == lb.model_info["upper_bound"]
    # leebounds as shipped: lower keeps floor((1-q) n) treated outcomes
    # (the rounded threshold drops the quantile observation), upper is the
    # exact fractional trim.
    ref = ST["attr_lee"]
    q = ref[4]
    obs = ATTR[ATTR["obs"] == 1]
    y1 = np.sort(obs.loc[obs["treat"] == 1, "y"].to_numpy())
    y0 = obs.loc[obs["treat"] == 0, "y"].to_numpy()
    keep = int(np.floor((1 - q) * len(y1)))
    _close(y1[:keep].mean() - y0.mean(), ref[0], rtol=1e-12)
    ex = sp.attrition_bounds(ATTR, "y", "treat", "obs", trimming="exact")
    _close(ex["upper_bound"], ref[1], rtol=1e-12)
    # the shipped upper bound's tie branch is the exact rule's analytic SE
    # too (fractional kept set, variance divisor = kept mass)
    lb_ex = sp.lee_bounds(
        ATTR, "y", "treat", "obs", se_method="analytic", trimming="exact"
    )
    _close(lb_ex.model_info["se_upper"], ref[3], rtol=1e-10)
    # relabelling the arms flips the bounds
    flip = sp.attrition_bounds(ATTR.assign(t0=1 - ATTR["treat"]), "y", "t0", "obs")
    _close(
        [flip["lower_bound"], flip["upper_bound"]],
        [-r["upper_bound"], -r["lower_bound"]],
    )
    _close(ST["attr_lee_flip"], [-ref[1], -ref[0]])


# --------------------------------------------------------------------------
# balance_check vs iebaltab
# --------------------------------------------------------------------------


def test_balance_check_pooled_ttest_matches_iebaltab():
    b = sp.balance_check(ATTR, "treat", ["b1", "b2", "b3"], equal_var=True)
    t = b.table.set_index("variable")
    for v in ("b1", "b2", "b3"):
        m1, m0, v1, v0, diff01, t01, p01, nrmd01 = ST[f"bal_{v}"]
        row = t.loc[v]
        _close([row["mean_treat"], row["mean_control"]], [m1, m0], rtol=1e-12)
        _close(row["diff"], -diff01, rtol=1e-12)
        _close(row["t_stat"], -t01, rtol=1e-10)
        _close(row["p_value"], p01, rtol=1e-9)
        _close(row["norm_diff"], -nrmd01, rtol=1e-12)
    F, p, n = ST["bal_ftest"]
    _close([b.omnibus_f, b.omnibus_p], [F, p], rtol=1e-9)
    # Default (Welch) differs only in the t-test.
    w = sp.balance_check(ATTR, "treat", ["b1", "b2", "b3"]).table
    _close(w["norm_diff"], b.table["norm_diff"], rtol=1e-15)
