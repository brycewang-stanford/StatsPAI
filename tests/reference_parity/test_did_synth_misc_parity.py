"""``sp.harvest_did``, ``sp.spillover_did`` and ``sp.causal_impact`` vs R.

Reference: ``_fixtures/did_synth_misc_R.json`` written by
``_generate_did_synth_misc_R.R`` (R 4.5, did 2.3.0 / DRDID, fixest 0.14.0,
KFAS 1.6.0, CausalImpact 1.4.1 / bsts 0.9.11 -- exact versions are stored in
the fixture's ``meta``) from the CSVs written by
``_generate_did_synth_misc_data.py`` (fixed seeds, ``%.17g``), so both sides
read identical bytes.

harvest_did
    No package implements the estimator as a whole (the "harvesting" chapter
    it was once attributed to, Abadie, Angrist, Frandsen & Pischke 2025, is a
    survey and defines no such estimator). Its building blocks are
    Callaway--Sant'Anna cells: every (cohort g, horizon e) cell equals
    ``did::att_gt(control_group = "notyettreated", base_period =
    "universal")`` -- point estimate and analytic SE (influence function,
    divisor n). With ``weighting = "n_treated"`` the per-horizon event study
    is ``did::aggte(type = "dynamic")``, including the cohort-share weight
    influence function. The inverse-variance aggregate over horizons has no
    ``did`` counterpart and is not graded against R.

spillover_did
    Butts's ring estimator. Single cohort: the direct and ring effects are the
    coefficients of ``fixest::feols(dbar ~ treat + ring1 + ring2)`` on the
    unit-level average long difference (the regression in Butts's
    ``rings_example.R``) and the SEs are its HC0 errors (``vcov = "hetero"``,
    ``ssc(adj = FALSE)``). Every design, including staggered adoption with
    ring exposure starting when the bordering cluster is treated: each group
    equals ``did::att_gt(control_group = "nevertreated")`` +
    ``did::aggte(type = "simple")`` on that group plus the clean controls,
    ring units' cohort set to their exposure onset. The ring construction is
    recomputed independently in R from the coordinates.

causal_impact
    ``sp.causal_impact`` is a frequentist regression + AR(1) state-space model
    with plug-in parameters; R ``CausalImpact`` is a Bayesian local-level +
    spike-and-slab regression fitted by MCMC (``bsts`` with a hard-coded
    ``seed = 1``). Same estimand, different model: not a parity target
    (outcome class 6). What *is* pinned is the deterministic chain
    ``sp.causal_impact`` actually runs -- the plug-in parameters from their
    closed forms (``lm`` / ``cor`` / ``sd``) and the Kalman filter/forecast
    against ``KFAS::KFS`` -- plus a reference-free identity for the
    standard error of the average effect (dense Gaussian conditioning).

Tolerance: 1e-9 relative on every compared number (all land at 1e-12 or
tighter); ``atol`` 1e-12 only for values that are exactly zero on one side.
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
R = json.loads((_FIX / "did_synth_misc_R.json").read_text(encoding="utf-8"))

RTOL = 1e-9
ATOL = 1e-12
EDGES = (0.0, 2.0, 4.0)


def _close(ours, ref, rtol=RTOL, atol=ATOL):
    np.testing.assert_allclose(
        np.asarray(ours, dtype=float),
        np.asarray(ref, dtype=float),
        rtol=rtol,
        atol=atol,
    )


# --------------------------------------------------------------------------
# harvest_did
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def harvest():
    h = pd.read_csv(_FIX / "did_synth_misc_harvest.csv")
    res = sp.harvest_did(
        h,
        unit="id",
        time="t",
        outcome="y",
        cohort="g",
        horizons=range(-5, 6),  # every cell did produces
        weighting="n_treated",
    )
    return h, res


def test_harvest_cells_match_did_att_gt(harvest):
    _, res = harvest
    cells = pd.DataFrame(R["harvest"]["cells"]).dropna(subset=["se"])
    ours = res.detail.merge(
        cells, left_on=["cohort", "t2"], right_on=["group", "t"], suffixes=("", "_R")
    )
    # Every did cell except the universal base period (att 0, se NA) is
    # produced, and nothing else is.
    assert len(ours) == len(cells) == len(res.detail)
    _close(ours["att"], ours["att_R"])
    _close(ours["se"], ours["se_R"])


def test_harvest_event_study_matches_did_aggte_dynamic(harvest):
    _, res = harvest
    dyn = pd.DataFrame(R["harvest"]["dynamic"]).dropna(subset=["se"])
    es = res.model_info["event_study"].merge(
        dyn, left_on="relative_time", right_on="e", suffixes=("", "_R")
    )
    assert len(es) == len(dyn)
    _close(es["att"], es["att_R"])
    _close(es["se"], es["se_R"])


def test_harvest_placebo_cells_exclude_own_cohort(harvest):
    """Regression: a pre-period cell's control group (cohorts not yet treated
    at the base period g-1) used to include cohort g itself."""
    h, res = harvest
    g_of = h.drop_duplicates("id").set_index("id")["g"]
    for _, row in res.detail.iterrows():
        t_last = max(row.t1, row.t2)
        expected = int(((g_of == 0) | ((g_of > t_last) & (g_of != row.cohort))).sum())
        assert row.n_control == expected


def test_harvest_aggregate_uses_cross_horizon_covariance(harvest):
    """Reference-free: the aggregate SE is the SE of the weighted sum of
    horizon estimates under their joint covariance, which exceeds the
    independence formula here because horizons share treated units."""
    _, res = harvest
    es = res.model_info["event_study"]
    post = es[es.relative_time >= 0]
    w = np.array(
        [res.model_info["aggregate_horizon_weights"][e] for e in post.relative_time]
    )
    _close(w.sum(), 1.0)
    _close(res.estimate, np.sum(w * post.att))
    indep = np.sqrt(np.sum(w**2 * post.se**2))
    assert res.se > indep


# --------------------------------------------------------------------------
# spillover_did
# --------------------------------------------------------------------------


def _spill(file):
    d = pd.read_csv(_FIX / file)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.spillover_did(
            d,
            y="y",
            unit="id",
            time="t",
            cohort="g",
            coords=["cx", "cy"],
            ring_edges=EDGES,
        )


@pytest.mark.parametrize(
    "key,file",
    [
        ("spill_single", "did_synth_misc_spill_single.csv"),
        ("spill_stag", "did_synth_misc_spill_stag.csv"),
    ],
)
def test_spillover_matches_did_simple(key, file):
    res = _spill(file)
    ref = R[key]
    assert res.n_clean_controls == ref["n_clean"]
    _close(res.direct, ref["direct"]["att"])
    _close(res.direct_se, ref["direct"]["se"])
    for _, row in res.rings.iterrows():
        rr = ref[f"ring_{int(row.ring)}"]
        assert row.n_units == rr["n_units"] == ref["ring_counts"][int(row.ring) - 1]
        _close(row.estimate, rr["att"])
        _close(row.se, rr["se"])
    # Cell-level: every (group, onset cohort, period) ATT.
    det = res.detail
    for name in ("direct", "ring_1", "ring_2"):
        rr = ref[name]
        cells = pd.DataFrame(
            {"cohort": rr["cell_group"], "time": rr["cell_t"], "att_R": rr["cell_att"]}
        )
        m = det[det.group == name].merge(cells, on=["cohort", "time"])
        assert len(m) == len(cells) == int((det.group == name).sum())
        _close(m["estimate"], m["att_R"])


def test_spillover_single_cohort_matches_fixest_hc0():
    res = _spill("did_synth_misc_spill_single.csv")
    ref = R["spill_single_feols"]
    rings = res.rings.set_index("ring")
    _close(res.direct, ref["coef"]["treat"])
    _close(res.direct_se, ref["se"]["treat"])
    for r in (1, 2):
        _close(rings.loc[r, "estimate"], ref["coef"][f"ring{r}"])
        _close(rings.loc[r, "se"], ref["se"][f"ring{r}"])


def test_spillover_staggered_ring_onsets():
    """Regression: under staggered adoption each ring unit is compared from
    its own exposure onset. Before the fix every ring unit entered every
    cohort's cell, and the ring effects were biased toward zero."""
    res = _spill("did_synth_misc_spill_stag.csv")
    onsets = res.diagnostics["ring_onsets"]
    for r in (1, 2):
        ref_groups = set(R["spill_stag"][f"ring_{r}"]["cell_group"])
        assert {int(float(k)) for k in onsets[f"ring_{r}"]} == ref_groups
    assert res.diagnostics["n_ring_changes"] == 0


# --------------------------------------------------------------------------
# causal_impact
# --------------------------------------------------------------------------


@pytest.fixture(scope="module")
def impact():
    ci = pd.read_csv(_FIX / "did_synth_misc_impact.csv")
    res = sp.causal_impact(
        ci, y="y", time="t", intervention_time=71, covariates=["x1", "x2"]
    )
    return ci, res


def test_causal_impact_plugin_parameters(impact):
    _, res = impact
    f = R["impact_filter"]
    mp = res.model_info["model_params"]
    _close(mp["beta"], f["beta"])
    _close(mp["rho"], f["rho"])
    _close(mp["sigma_obs"], f["sigma_obs"])
    _close(mp["sigma_state"], f["sigma_state"])


def test_causal_impact_filter_matches_kfas(impact):
    _, res = impact
    f = R["impact_filter"]
    _close(res.detail["predicted"], f["y_pred"])
    _close(res.detail["predicted_se"], f["y_pred_se"])


def test_causal_impact_se_is_dense_gaussian_conditional(impact):
    """Reference-free identity: under the fitted model, the variance of the
    post-period sum of forecast errors given the pre-period data, computed by
    dense Gaussian conditioning, equals ``se_total**2``; the conditional mean
    equals the forecast. (Before the fix the errors were treated as
    independent.)"""
    ci, res = impact
    mp = res.model_info["model_params"]
    rho, so, ss = mp["rho"], mp["sigma_obs"], mp["sigma_state"]
    n = len(ci)
    pre = ci["t"].to_numpy() < 71
    v = np.empty(n)
    v[0] = ss**2
    for t in range(1, n):
        v[t] = rho**2 * v[t - 1] + ss**2
    idx = np.arange(n)
    lo = np.minimum(idx[:, None], idx[None, :])
    S = rho ** np.abs(idx[:, None] - idx[None, :]) * v[lo]
    Sy = S + so**2 * np.eye(n)
    X = np.column_stack([np.ones(n), ci[["x1", "x2"]].to_numpy()])
    r = ci["y"].to_numpy() - X @ np.asarray(mp["beta"])
    K = Sy[np.ix_(~pre, pre)] @ np.linalg.inv(Sy[np.ix_(pre, pre)])
    cond_mean = K @ r[pre]
    cond_cov = Sy[np.ix_(~pre, ~pre)] - K @ Sy[np.ix_(pre, ~pre)]
    _close(
        res.detail["predicted"].to_numpy()[~pre],
        (X @ mp["beta"])[~pre] + cond_mean,
        rtol=1e-10,
    )
    _close(res.model_info["se_total"] ** 2, cond_cov.sum(), rtol=1e-10)
    _close(res.se, np.sqrt(cond_cov.sum()) / (~pre).sum(), rtol=1e-10)


def test_causal_impact_reference_is_a_different_model():
    """Evidence for outcome class 6, not a parity claim: R CausalImpact's
    posterior point estimate is seed-invariant (bsts runs with a hard-coded
    seed = 1; the R seed only moves the posterior-predictive draws), so a
    multi-seed Monte Carlo comparison is not available either."""
    runs = R["causalimpact_evidence"]
    assert len({r["avg_abs_effect"] for r in runs}) == 1
    assert len({r["avg_abs_effect_sd"] for r in runs}) > 1


def test_causal_impact_n_seasons_warns(impact):
    ci, _ = impact
    with pytest.warns(UserWarning, match="n_seasons=7 is ignored"):
        sp.causal_impact(
            ci,
            y="y",
            time="t",
            intervention_time=71,
            covariates=["x1", "x2"],
            n_seasons=7,
        )
