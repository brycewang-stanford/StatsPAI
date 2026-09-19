"""sp.svydesign (strata, clusters, weights, fpc, lonely PSUs) vs R and Stata.

References, both read ``_fixtures/survey_design_data.csv`` (6 strata with 2-6
PSUs, PSU ids restarting at 1 in every stratum, 4-12 elements per PSU,
unequal weights, fpc columns in count and fraction form):

* ``_fixtures/survey_design_R.json`` -- R ``survey`` (``svydesign`` /
  ``svymean`` / ``svytotal`` / ``svyglm`` / ``degf`` / ``confint``), written
  by ``_generate_survey_design_R.R`` (versions in its ``provenance``).
* ``_fixtures/survey_design_stata.json`` -- Stata 18 ``svyset`` + ``svy:
  mean / total / regress / logit / poisson`` + ``estat effects``, written by
  ``_fixtures/_generate_survey_design_stata.do``.

Conventions each number depends on
----------------------------------
* Variance: first-stage (ultimate-cluster) Taylor linearisation,
  sum_h (1 - f_h) n_h / (n_h - 1) sum_j (t_hj - tbar_h)^2 over PSU score
  totals, f_h = n_h / N_h with n_h the number of sampled PSUs (not
  elements) in the stratum.  R ``survey:::onestrat`` and Stata agree.
* PSUs are identified within strata.  R needs ``nest = TRUE`` for the
  repeated ids in this file (``nest = FALSE`` is an error there, recorded
  in the fixture); Stata ``svyset`` nests implicitly.  StatsPAI nests
  always and warns when ``nest=False`` meets repeated ids.
* Design df = #PSU - #strata (R ``degf``, Stata ``e(df_r)``).  Mean / total
  CIs use t with that df (Stata; R ``confint(., df = degf(des))`` -- R's
  own ``confint.svystat`` default is the normal quantile, also frozen).
  GLM t / p / CI: ``dof="design"`` (default) is Stata's df;
  ``dof="residual"`` is R ``summary.svyglm``'s degf + 1 - #coef.
* DEFF: ``deff="wor"`` (default) is R ``deff = TRUE`` (SRS without
  replacement, N = sum of weights) and Stata ``estat effects`` when an fpc
  is declared; ``deff="replace"`` is Stata's ``estat effects`` without fpc
  (R ``deff = "replace"``).
* GLMs: canonical links (R quasibinomial / quasipoisson; Stata logit /
  poisson).  Both references are frozen at the converged MLE (see the
  generators for why R's one-pass svyglm is ~2e-7 off in the logit SE).
* Lonely PSU: R ``survey.lonely.psu`` remove / certainty / adjust / average
  = Stata ``singleunit`` certainty / certainty / centered / scaled.

Tolerance: 1e-10 relative on every point estimate, SE, DEFF and CI bound
(observed 1e-13 or tighter, CI bounds up to ~1e-13 through the t
quantile); p-values 1e-9 relative (``t.sf`` near the tail).
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import statspai as sp

_FIX = Path(__file__).parent / "_fixtures"
R = json.loads((_FIX / "survey_design_R.json").read_text(encoding="utf-8"))
S = json.loads((_FIX / "survey_design_stata.json").read_text(encoding="utf-8"))
RTOL = 1e-10


@pytest.fixture(scope="module")
def df():
    d = pd.read_csv(_FIX / "survey_design_data.csv")
    d["psu_global"] = d["stratum"] * 100 + d["psu"]
    return d


DESIGNS = {
    "full": dict(strata="stratum", cluster="psu", fpc="fpc_N", nest=True),
    "frac": dict(strata="stratum", cluster="psu", fpc="fpc_f", nest=True),
    "nofpc": dict(strata="stratum", cluster="psu", nest=True),
    "element_fpc": dict(strata="stratum", fpc="fpc_el"),
    "cluster_only": dict(cluster="psu_global"),
}
# Stata estat effects uses the with-replacement SRS variance when no fpc
STATA_DEFF_MODE = {
    "full": "wor",
    "nofpc": "replace",
    "element_fpc": "wor",
    "cluster_only": "replace",
}


def _close(ours, ref, rtol=RTOL):
    np.testing.assert_allclose(
        np.ravel(np.asarray(ours, dtype=float)),
        np.ravel(np.asarray(ref, dtype=float)),
        rtol=rtol,
        atol=0.0,
    )


def _stata_se(block):
    return np.sqrt(np.diag(np.asarray(block["V"], dtype=float)))


def _design(df, key, **extra):
    return sp.svydesign(df, weights="w", **DESIGNS[key], **extra)


# --------------------------------------------------------------------------
# mean / total vs R survey
# --------------------------------------------------------------------------


@pytest.mark.parametrize("key", list(DESIGNS))
def test_design_df_matches_R_degf(df, key):
    d = _design(df, key)
    assert d.mean("y").dof == R[key]["degf"]
    assert d.glm("y ~ x").dof == R[key]["degf"]


@pytest.mark.parametrize("key", list(DESIGNS))
def test_svymean_matches_R(df, key):
    d = _design(df, key)
    ref = R[key]
    m = d.mean("y")
    _close(m.estimate, ref["mean_y"]["estimate"])
    _close(m.std_error, ref["mean_y"]["se"])
    _close(m.deff, ref["mean_y"]["deff"])
    _close([m.ci_lower.iloc[0], m.ci_upper.iloc[0]], ref["mean_y"]["ci_t"])
    m2 = d.mean(["y", "x"])
    _close(m2.estimate, ref["mean_yx"]["estimate"])
    _close(m2.std_error, ref["mean_yx"]["se"])
    _close(m2.deff, ref["mean_yx"]["deff"])


def test_svymean_normal_ci_is_R_confint_default(df):
    """R's confint.svystat uses the normal quantile; ours is t(degf)."""
    from scipy import stats

    m = _design(df, "full").mean("y")
    z = stats.norm.ppf(0.975)
    est, se = m.estimate.iloc[0], m.std_error.iloc[0]
    _close([est - z * se, est + z * se], R["full"]["mean_y"]["ci_normal"])


@pytest.mark.parametrize("key", list(DESIGNS))
def test_svytotal_matches_R(df, key):
    t = _design(df, key).total("y")
    ref = R[key]["total_y"]
    _close(t.estimate, ref["estimate"])
    _close(t.std_error, ref["se"])
    _close(t.deff, ref["deff"])
    _close([t.ci_lower.iloc[0], t.ci_upper.iloc[0]], ref["ci_t"])


# --------------------------------------------------------------------------
# GLM vs R survey::svyglm
# --------------------------------------------------------------------------

GLMS = [
    ("glm_gaussian", "y ~ x", "gaussian"),
    ("glm_binomial", "yb ~ x", "binomial"),
    ("glm_poisson", "yc ~ x", "poisson"),
]


R_GLM_CASES = [(k, *g) for k in DESIGNS for g in GLMS if g[0] in R[k]]


@pytest.mark.parametrize("key, block, formula, family", R_GLM_CASES)
def test_svyglm_matches_R(df, key, block, formula, family):
    ref = R[key][block]
    d = _design(df, key)
    g = d.glm(formula, family=family)
    _close(g.estimate, ref["coef"])
    _close(g.std_error, ref["se"])
    # R summary.svyglm: t with df = degf + 1 - #coef
    gr = d.glm(formula, family=family, dof="residual")
    assert gr.dof == ref["df_residual"]
    _close(gr.t_values, ref["t"])
    _close(gr.p_values, ref["p"], rtol=1e-9)
    _close(np.column_stack([gr.ci_lower, gr.ci_upper]), ref["ci"])


def test_svyglm_logit_R_default_control_gap_is_stale_weights(df):
    """R's one-pass svyglm (glm.control() defaults) is ~2e-7 off in the SE.

    glm.fit reports the IRLS working weights from before its final update;
    refitting from the converged coefficients (the frozen reference) removes
    it.  Pin the size so a change on either side is noticed.
    """
    g = _design(df, "full").glm("yb ~ x", family="binomial")
    ref = np.asarray(R["full"]["glm_binomial_default_control"]["se"])
    gap = np.max(np.abs(g.std_error.to_numpy() - ref) / ref)
    assert 1e-8 < gap < 1e-6


# --------------------------------------------------------------------------
# Stata svy
# --------------------------------------------------------------------------


@pytest.mark.parametrize("key", ["full", "nofpc", "element_fpc", "cluster_only"])
def test_svy_mean_matches_stata(df, key):
    d = _design(df, key)
    m = d.mean("y", deff=STATA_DEFF_MODE[key])
    st = S[f"{key}_mean_y"]
    _close(m.estimate, st["b"])
    _close(m.std_error, _stata_se(st))
    _close(m.deff, st["deff"])
    assert m.dof == st["df_r"] == st["N_psu"] - st["N_strata"]


def test_svy_mean_frac_fpc_matches_stata(df):
    m = _design(df, "frac").mean("y")
    _close(m.std_error, _stata_se(S["frac_mean_y"]))


def test_svy_mean_two_vars_and_total_match_stata(df):
    d = _design(df, "full")
    _close(d.mean(["y", "x"]).std_error, _stata_se(S["full_mean_yx"]))
    t = d.total("y")
    _close(t.estimate, S["full_total_y"]["b"])
    _close(t.std_error, _stata_se(S["full_total_y"]))
    _close(t.deff, S["full_total_y"]["deff"])
    _close(_design(df, "nofpc").total("y").std_error, _stata_se(S["nofpc_total_y"]))


STATA_GLMS = [
    ("regress", "y ~ x", "gaussian"),
    ("logit", "yb ~ x", "binomial"),
    ("poisson", "yc ~ x", "poisson"),
]


STATA_GLM_CASES = [
    (k, *g)
    for k in ["full", "nofpc", "element_fpc", "cluster_only"]
    for g in STATA_GLMS
    if f"{k}_{g[0]}" in S
]


@pytest.mark.parametrize("key, cmd, formula, family", STATA_GLM_CASES)
def test_svy_regression_matches_stata(df, key, cmd, formula, family):
    name = f"{key}_{cmd}"
    st = S[name]
    g = _design(df, key).glm(formula, family=family)
    # Stata orders (x, _cons); ours (Intercept, x)
    _close(g.estimate.to_numpy()[::-1], st["b"])
    _close(g.std_error.to_numpy()[::-1], _stata_se(st))
    assert g.dof == st["df_r"]  # Stata: design df for regressions too


# --------------------------------------------------------------------------
# fpc, nesting, lonely PSUs
# --------------------------------------------------------------------------


def test_fpc_count_and_fraction_forms_agree_and_shrink_se(df):
    full = _design(df, "full").mean("y").std_error.iloc[0]
    frac = _design(df, "frac").mean("y").std_error.iloc[0]
    nofpc = _design(df, "nofpc").mean("y").std_error.iloc[0]
    _close(full, frac, rtol=1e-12)
    assert full < nofpc
    # reference-free: sampling every PSU (f_h = 1) makes the variance zero
    census = df.assign(
        N_census=df.groupby("stratum")["psu"].transform("nunique").astype(float)
    )
    d = sp.svydesign(
        census, weights="w", strata="stratum", cluster="psu", fpc="N_census", nest=True
    )
    assert d.mean("y").std_error.iloc[0] == 0.0


def test_fpc_counts_psus_not_elements(df):
    """Regression: f_h used to be (#elements in h) / N_h -- a NaN SE here.

    fpc_N is the population number of PSUs; with 4-12 elements per PSU the
    old element count exceeded N_h, so 1 - f_h < 0 and sqrt gave NaN.
    """
    se = _design(df, "full").mean("y").std_error.iloc[0]
    assert np.isfinite(se)
    _close(se, R["full"]["mean_y"]["se"])


def test_fpc_input_validation(df):
    bad = df.assign(mixed=np.where(df["stratum"] == 1, 0.5, 20.0))
    with pytest.raises(ValueError, match="all population counts"):
        sp.svydesign(
            bad, weights="w", strata="stratum", cluster="psu", fpc="mixed", nest=True
        )
    small = df.assign(Nsmall=1.5)
    with pytest.raises(ValueError, match="100% sampling"):
        sp.svydesign(
            small, weights="w", strata="stratum", cluster="psu", fpc="Nsmall", nest=True
        )


def test_repeated_psu_ids_nested_within_strata(df):
    """R refuses nest=FALSE here; Stata nests; we nest and warn."""
    assert R["crossed_nest_false_error"].startswith("Clusters not nested in strata")
    with pytest.warns(UserWarning, match="repeat across strata"):
        d = sp.svydesign(df, weights="w", strata="stratum", cluster="psu", nest=False)
    m = d.mean("y")
    # the df used to count distinct raw ids (6) -> max(6 - 6, 1) = 1
    assert m.dof == R["nofpc"]["degf"] == 17
    _close(m.std_error, R["nofpc"]["mean_y"]["se"])
    _close([m.ci_lower.iloc[0], m.ci_upper.iloc[0]], R["nofpc"]["mean_y"]["ci_t"])


LONELY = [
    ("remove", "certainty"),
    ("certainty", "certainty"),
    ("adjust", "centered"),
    ("average", "scaled"),
]


@pytest.mark.parametrize("rule, stata_rule", LONELY)
def test_lonely_psu_rules_match_R_and_stata(df, rule, stata_rule):
    d = sp.svydesign(
        df, weights="w", strata="stratum_l", cluster="psu", nest=True, lonely_psu=rule
    )
    ref = R[f"lonely_{rule}"]
    with warnings.catch_warnings():
        warnings.simplefilter("error")  # explicit rule: no warning
        m = d.mean("y")
    assert m.dof == ref["degf"]
    _close(m.estimate, ref["mean_y"]["estimate"])
    _close(m.std_error, ref["mean_y"]["se"])
    m2 = d.mean(["y", "x"])
    _close(m2.std_error, np.sqrt(np.diag(ref["mean_yx"]["vcov"])))
    _close(m2.std_error, _stata_se(S[f"lonely_{stata_rule}_mean_yx"]))
    _close(d.total("y").std_error, ref["total_y"]["se"])
    g = d.glm("y ~ x")
    _close(g.std_error, ref["glm_gaussian"]["se"])
    _close(g.std_error.to_numpy()[::-1], _stata_se(S[f"lonely_{stata_rule}_regress"]))


def test_lonely_psu_default_warns_and_fail_raises(df):
    assert R["lonely_fail_error"].startswith("Stratum (7) has only one PSU")
    assert S["lonely_missing_se"] == "missing"
    d = sp.svydesign(df, weights="w", strata="stratum_l", cluster="psu", nest=True)
    with pytest.warns(UserWarning, match="single sampled PSU"):
        se = d.mean("y").std_error.iloc[0]
    _close(se, R["lonely_remove"]["mean_y"]["se"])
    d = sp.svydesign(
        df, weights="w", strata="stratum_l", cluster="psu", nest=True, lonely_psu="fail"
    )
    with pytest.raises(ValueError, match="single sampled PSU"):
        d.mean("y")


def test_deff_replace_is_invariant_to_weight_scale(df):
    """Reference-free: the with-replacement DEFF of a mean is scale-free.

    Before the fix the DEFF denominator was svyvar / sum(w), so multiplying
    the weights by 10 multiplied the reported DEFF by 10.
    """
    d1 = _design(df, "nofpc")
    d10 = sp.svydesign(df.assign(w=df["w"] * 10), weights="w", **DESIGNS["nofpc"])
    _close(
        d1.mean("y", deff="replace").deff,
        d10.mean("y", deff="replace").deff,
        rtol=1e-12,
    )


def test_non_gaussian_bread_uses_variance_function(df):
    """Reference-free sandwich identity for the logit SE.

    V = A^{-1} B A^{-1} with A = X' diag(w mu (1 - mu)) X.  The bread used
    to be (X' W X)^{-1} for every family, understating the logit SE by ~4x.
    """
    d = _design(df, "nofpc")
    g = d.glm("yb ~ x", family="binomial")
    X = np.column_stack([np.ones(len(df)), df["x"].to_numpy()])
    b = g.estimate.to_numpy()
    mu = 1 / (1 + np.exp(-X @ b))
    w = df["w"].to_numpy()
    # stationarity of the weighted score at the reported coefficients
    np.testing.assert_allclose(X.T @ (w * (df["yb"].to_numpy() - mu)), 0.0, atol=1e-8)
    A = (X * (w * mu * (1 - mu))[:, None]).T @ X
    scores = X * (w * (df["yb"].to_numpy() - mu))[:, None]
    B = np.zeros((2, 2))
    for _, grp in pd.DataFrame(scores, columns=["a", "b"]).groupby(df["stratum"]):
        t = grp.groupby(df.loc[grp.index, "psu"]).sum().to_numpy()
        dev = t - t.mean(axis=0)
        B += len(t) / (len(t) - 1) * dev.T @ dev
    Ainv = np.linalg.inv(A)
    _close(g.std_error, np.sqrt(np.diag(Ainv @ B @ Ainv)), rtol=1e-8)
