"""Synthetic-control variants vs the R implementations that define them.

Reference: ``_fixtures/did_synth_synthvar_R.json`` written by
``_generate_did_synth_synthvar_R.R`` from augsynth 0.2.0 (``augsynth`` and
``multisynth``; OSQP 1.0.0), scpi 4.0.1, glmnet 4.1.10, DiSCos 0.1.4
(pracma 2.4.6 / quadprog 1.5.8 for its weight regression, CVXR 1.8.2 / SCS
for its CDF-mixture LP) and GLPK (Rglpk 0.6.5.1, via CVXR), on the CSVs
written by ``_generate_did_synth_synthvar_data.py``.

``sp.demeaned_synth`` — augsynth(progfunc = "None", fixedeff = TRUE): the
simplex SCM on unit-demeaned outcomes, the treated unit's pre-mean added
back. Weights and the whole 30-period gap path. augsynth's ``synth_qp``
hard-codes OSQP at eps 1e-8, so R's weights carry ~1e-9 noise (zero weights
come back as +-1e-9): weights atol 1e-8, gap / ATT rtol 1e-7.

``sp.robust_synth`` — ``variant='unconstrained', l2_penalty=0`` is OLS with
an intercept (Doudchenko & Imbens' unconstrained estimator), identical to
scpi ``scest(w.constr = list(name = "ols"))`` with ``constant = TRUE`` and to
``lm``: 1e-10. The penalised paths minimise
``||y - mu - Xw||^2 + l2 ||w||^2 + l1 ||w||_1`` with ``mu`` unpenalised;
glmnet solves the same problem once ``y`` is scaled to unit (1/n) SD and
``n lambda (1 - alpha) = l2``, ``2 n lambda alpha = l1 / sd(y)`` (the mapping
is in the generator); glmnet run at ``thresh = 1e-24``: 1e-8.

``sp.staggered_synth`` — multisynth's partially pooled QP (nu, lambda,
fixedeff, n_leads, n_lags, time_cohort) and its ``Average`` ATT, per-unit /
per-cohort ATTs, event-time ATTs, weight matrix, nu heuristic, imbalance
norms and leave-one-unit-out jackknife SE. multisynth is run with OSQP at
eps 1e-12; StatsPAI solves the QP exactly (active set): 1e-8 on effects and
SE, atol 1e-8 on weights.

``sp.discos`` — DiSCo(mixture = FALSE) weights (per pre-period and
averaged), target and counterfactual quantile functions on the G + 1 grid,
DiSCoTEA's quantile differences, with R's own random quantile nodes replayed
(generator header) and passed as ``q_nodes``: quadprog vs StatsPAI's active
set, 1e-10. DiSCo(mixture = TRUE): the CDF grid DiSCo used is passed as
``cdf_grid``; the LP weights match GLPK at 1e-12 and DiSCo's SCS solution at
SCS's eps (atol 5e-6), the counterfactual quantiles exactly.

Permutation test (T4, reference defect). DiSCos 0.1.4's ``DiSCo_per_iter``
fits each placebo with the original treated unit in the donor pool but
builds the pool's quantile matrix with that unit's column left at zero, so
every placebo counterfactual drops a donor that carries weight. StatsPAI
uses the full pool. The test shows (i) StatsPAI's placebo distances equal
the generator's one-line-patched copy of ``DiSCo_per_iter`` at 1e-9 on R's
replayed nodes, and (ii) zeroing that column in StatsPAI's own pieces
reproduces DiSCos' unpatched distances at 1e-9 — the mechanism, not a
tolerance. The p-values (rank / (J + 1)) coincide on this data.

Reference-free identities: exact-fit DGPs for ``demeaned_synth`` (both
variants), ``staggered_synth`` and ``discos``; continuity of the elastic-net
path at ``l1 -> 0``; and placebo-statistic consistency for the MSPE-ratio
p-values.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.synth.discos import _disco_quantile_weights, _discos_micro, _quant7_sorted

_FIX = Path(__file__).parent / "_fixtures"
R = json.loads((_FIX / "did_synth_synthvar_R.json").read_text(encoding="utf-8"))
SINGLE = pd.read_csv(_FIX / "did_synth_synthvar_single.csv")
STAG = pd.read_csv(_FIX / "did_synth_synthvar_stag.csv")
MICRO = pd.read_csv(_FIX / "did_synth_synthvar_micro.csv")


def _close(ours, ref, rtol=1e-10, atol=0.0):
    np.testing.assert_allclose(
        np.asarray(ours, dtype=float),
        np.asarray(ref, dtype=float),
        rtol=rtol,
        atol=atol,
    )


def test_fixture_records_versions():
    meta = R["meta"]
    assert meta["augsynth"] == "0.2.0"
    assert meta["DiSCos"] == "0.1.4"
    assert meta["scpi"] == "4.0.1"
    assert meta["R_version"].startswith("R version 4.5")


# --------------------------------------------------------------------------
# demeaned_synth vs augsynth(fixedeff = TRUE, progfunc = "None")
# --------------------------------------------------------------------------


def _demeaned(**kw):
    return sp.demeaned_synth(
        SINGLE,
        "y",
        "unit",
        "time",
        treated_unit=1,
        treatment_time=21,
        placebo=kw.pop("placebo", False),
        **kw,
    )


def test_demeaned_weights_match_augsynth():
    ref = R["demeaned"]
    w = _demeaned().model_info["weights"].set_index("unit")["weight"]
    ours = [float(w.get(u, 0.0)) for u in ref["control_units"]]
    _close(ours, ref["weights"], rtol=0.0, atol=1e-8)


def test_demeaned_gap_path_and_att_match_augsynth():
    ref = R["demeaned"]
    r = _demeaned()
    _close(r.model_info["gap_table"]["gap"].to_numpy(), ref["gap"], rtol=1e-7)
    _close(r.estimate, ref["att_post_mean"], rtol=1e-7)


def test_demeaned_placebo_statistic_is_consistent():
    """Treated and placebo use the same post/pre MSPE ratio.

    Before the fix the placebos used mean(post gap)^2 / pre MSPE while the
    treated unit used mean(post gap^2) / pre MSPE, which by Jensen inflates
    the treated statistic: under no effect the p-value hit its minimum 1/9.
    Donor 5's placebo equals the fit with unit 5 as the treated unit.
    """
    r = _demeaned(placebo=True)
    donors = sorted(u for u in SINGLE.unit.unique() if u != 1)
    k = donors.index(5)
    direct = sp.demeaned_synth(
        SINGLE,
        "y",
        "unit",
        "time",
        treated_unit=5,
        treatment_time=21,
        placebo=True,
    )
    _close(
        r.model_info["placebo_mspe_ratios"][k],
        direct.model_info["mspe_ratio"],
        rtol=1e-6,
    )


@pytest.mark.parametrize("variant", ["demeaned", "detrended"])
def test_demeaned_recovers_exact_effect(variant):
    """Treated = 0.3 d1 + 0.7 d2 + level (+ trend) + 2 after t0: exact."""
    rng = np.random.default_rng(3)
    T, t0 = 25, 15
    donors = {
        f"d{j}": np.cumsum(rng.normal(0, 1, T)) + rng.normal(0, 3) for j in range(5)
    }
    trend = 0.4 * np.arange(T) if variant == "detrended" else 0.0
    tr = 0.3 * donors["d0"] + 0.7 * donors["d1"] + 10.0 + trend
    tr = tr + 2.0 * (np.arange(T) >= t0)
    rows = [(u, t, v[t]) for u, v in {**donors, "tr": tr}.items() for t in range(T)]
    df = pd.DataFrame(rows, columns=["unit", "time", "y"])
    r = sp.demeaned_synth(
        df,
        "y",
        "unit",
        "time",
        treated_unit="tr",
        treatment_time=t0,
        variant=variant,
        placebo=False,
    )
    assert abs(r.estimate - 2.0) < 1e-6


def test_demeaned_rejects_covariates():
    with pytest.raises(NotImplementedError):
        _demeaned(covariates=["treated"])


# --------------------------------------------------------------------------
# robust_synth vs scpi / lm / glmnet
# --------------------------------------------------------------------------


def _robust(**kw):
    return sp.robust_synth(
        SINGLE,
        "y",
        "unit",
        "time",
        treated_unit=1,
        treatment_time=21,
        placebo=kw.pop("placebo", False),
        **kw,
    )


def test_robust_ols_matches_scpi_and_lm():
    r = _robust(variant="unconstrained", l2_penalty=0.0)
    w = r.model_info["weights"].set_index("unit")["weight"].sort_index()
    for ref in (R["robust"]["ols"], R["robust"]["ols_lm"]):
        _close(w.to_numpy(), ref["weights"])
        _close(r.model_info["intercept"], ref["intercept"])
    ols = R["robust"]["ols"]
    y_synth = r.model_info["Y_synth"]
    _close(y_synth[:20], ols["y_pre_fit"])
    _close(y_synth[20:], ols["y_post_fit"])


@pytest.mark.parametrize("case", range(4))
def test_robust_penalised_paths_match_glmnet(case):
    ref = R["robust"]["glmnet"][case]
    variant = "elastic_net" if ref["l1"] > 0 else "unconstrained"
    r = _robust(variant=variant, l2_penalty=ref["l2"], l1_penalty=ref["l1"])
    w = r.model_info["weights"].set_index("unit")["weight"].sort_index()
    # glmnet stops coordinate descent on a change criterion: ~3e-11 absolute
    # on the smallest ridge weight (0.002); StatsPAI's ridge is closed form.
    _close(w.to_numpy(), ref["weights"], rtol=1e-8, atol=1e-10)
    _close(r.model_info["intercept"], ref["intercept"], rtol=1e-8)


def test_robust_elastic_net_is_continuous_at_zero_l1():
    """Regression: the coordinate-descent path used to penalise the intercept
    (and double l1), so l1 = 1e-12 moved the intercept from 20.74 to 0.53."""
    a = _robust(variant="elastic_net", l2_penalty=5.0, l1_penalty=0.0)
    b = _robust(variant="elastic_net", l2_penalty=5.0, l1_penalty=1e-12)
    _close(b.model_info["intercept"], a.model_info["intercept"], rtol=1e-9)
    _close(b.estimate, a.estimate, rtol=1e-9)


def test_robust_placebo_statistic_is_consistent():
    r = _robust(placebo=True)
    donors = sorted(u for u in SINGLE.unit.unique() if u != 1)
    k = donors.index(4)
    direct = sp.robust_synth(
        SINGLE, "y", "unit", "time", treated_unit=4, treatment_time=21, placebo=True
    )
    _close(
        r.model_info["placebo_mspe_ratios"][k],
        direct.model_info["mspe_ratio"],
        rtol=1e-9,
    )


def test_robust_rejects_covariates():
    with pytest.raises(NotImplementedError):
        _robust(covariates=["treated"])


# --------------------------------------------------------------------------
# staggered_synth vs multisynth
# --------------------------------------------------------------------------


def _ms_kwargs(case):
    args = case["args"] or {}  # jsonlite writes an empty list as []
    return dict(
        method="pooled" if args.get("time_cohort") else "separate",
        nu=args.get("nu", "auto"),
        fixedeff=args.get("fixedeff", True),  # multisynth default TRUE
        n_leads=case["n_leads"],
        n_lags=case["n_lags"],
        penalization=args.get("lambda", 0.0),
    )


@pytest.mark.parametrize(
    "idx", range(len(R["multisynth"])), ids=[c["name"] for c in R["multisynth"]]
)
def test_staggered_matches_multisynth(idx):
    case = R["multisynth"][idx]
    jack = "jackknife_se" in case
    r = sp.staggered_synth(
        STAG,
        "y",
        "unit",
        "time",
        "treated",
        placebo=jack,
        se_method="jackknife",
        **_ms_kwargs(case),
    )
    mi = r.model_info
    _close(r.estimate, case["att"], rtol=1e-8)
    _close(mi["unit_effects"]["att"].to_numpy(), case["group_att"], rtol=1e-8)
    _close(mi["nu"], case["nu"], rtol=1e-8)
    _close(mi["global_l2"], case["global_l2"], rtol=1e-6)
    _close(mi["ind_l2"], case["ind_l2"], rtol=1e-6)
    es = mi["event_study"].set_index("event_time")["att"]
    ref_es = dict(zip(case["event_time"], case["event_att"]))
    common = [e for e in ref_es if e in es.index and ref_es[e] is not None]
    assert len(common) == len(es)
    _close([es[e] for e in common], [ref_es[e] for e in common], rtol=1e-8, atol=1e-10)
    _close(
        mi["weights"].to_numpy(),
        np.asarray(case["weights"], dtype=float),
        rtol=0.0,
        atol=1e-8,
    )
    if jack:
        _close(r.se, case["jackknife_se"], rtol=1e-8)


def test_staggered_default_donors_exclude_eventually_treated():
    """Regression: donors used to include not-yet-treated units whose own
    treatment falls inside the effect window. By default (window to the end
    of the panel) only never-treated units may carry weight."""
    r = sp.staggered_synth(STAG, "y", "unit", "time", "treated", placebo=False)
    W = r.model_info["weights"]
    treated_units = sorted(STAG.loc[STAG.treated == 1, "unit"].unique())
    assert np.all(W.loc[treated_units].to_numpy() == 0.0)


def test_staggered_recovers_exact_effect():
    """Treated units are exact convex combinations of never-treated units."""
    rng = np.random.default_rng(5)
    T = 16
    ctrl = {f"c{j}": np.cumsum(rng.normal(0, 1, T)) + 3 * j for j in range(6)}
    adopt = {"a": 8, "b": 8, "c": 11}
    mix = {"a": (0.5, 0.5, 0), "b": (0.2, 0, 0.8), "c": (0, 0.6, 0.4)}
    rows = []
    for u, v in ctrl.items():
        rows += [(u, t, v[t], 0) for t in range(T)]
    for u, g in adopt.items():
        base = sum(m * ctrl[f"c{k}"] for k, m in enumerate(mix[u]))
        rows += [(u, t, base[t] + 1.5 * (t >= g), int(t >= g)) for t in range(T)]
    df = pd.DataFrame(rows, columns=["unit", "time", "y", "d"])
    r = sp.staggered_synth(df, "y", "unit", "time", "d", placebo=False)
    assert abs(r.estimate - 1.5) < 1e-8


# --------------------------------------------------------------------------
# discos vs DiSCos::DiSCo
# --------------------------------------------------------------------------


def _disco_case(name):
    return next(c for c in R["discos"] if c["name"] == name)


def _period_matrix(res, col):
    bp = res.model_info["quantile_effects_by_period"]
    return np.vstack([bp.loc[bp.time == t, col].to_numpy() for t in range(1, 7)])


def _run_disco(case, placebo=False):
    kw = dict(
        method="mixture" if case["mixture"] else "quantile",
        simplex=case["simplex"],
        M=case["M"],
        n_quantiles=case["G"],
    )
    if case["mixture"]:
        kw["cdf_grid"] = [np.asarray(g) for g in case["cdf_grid"]]
    else:
        kw["q_nodes"] = [np.asarray(q) for q in case["q_nodes"]]
    if placebo:
        nodes = case["perm_nodes"]
        return _discos_micro(
            MICRO,
            outcome="y",
            unit="id",
            time="time",
            treated_unit=1,
            treatment_time=case["t0"],
            G=case["G"],
            M=case["M"],
            simplex=case["simplex"],
            method=kw["method"],
            q_nodes=kw.get("q_nodes"),
            cdf_grid=kw.get("cdf_grid"),
            placebo=True,
            alpha=0.05,
            placebo_node_fn=lambda i, t: np.asarray(nodes[i][t]),
        )
    return sp.discos(
        MICRO,
        "y",
        "id",
        "time",
        treated_unit=1,
        treatment_time=case["t0"],
        placebo=False,
        **kw,
    )


@pytest.mark.parametrize("name", ["quantile", "quantile_simplex"])
def test_discos_quantile_matches_disco(name):
    case = _disco_case(name)
    r = _run_disco(case)
    mi = r.model_info
    _close(list(mi["weights"].values()), case["weights"], rtol=0.0, atol=1e-11)
    _close(
        mi["period_weights"].to_numpy(), case["period_weights"], rtol=0.0, atol=1e-11
    )
    _close(_period_matrix(r, "treated"), case["target_quantiles"], rtol=1e-13)
    _close(
        _period_matrix(r, "counterfactual"),
        case["counterfactual_quantiles"],
        rtol=1e-10,
    )
    _close(_period_matrix(r, "effect"), case["quantile_diff"], rtol=0.0, atol=1e-10)
    post = np.asarray(case["quantile_diff"], dtype=float)[case["t0"] - 1 :]
    _close(r.estimate, post.mean(), rtol=1e-10)


def test_discos_mixture_matches_glpk_and_disco():
    case = _disco_case("mixture_simplex")
    r = _run_disco(case)
    pw = r.model_info["period_weights"].to_numpy()
    for t, g in enumerate(case["glpk"]):
        _close(pw[t], g["weights"], rtol=0.0, atol=1e-12)
        # DiSCo's SCS solution is feasible and within its eps of the optimum
        assert g["objective_scs"] >= g["objective_glpk"] - 1e-9
        assert g["objective_scs"] - g["objective_glpk"] < 1e-5
    _close(pw, case["period_weights"], rtol=0.0, atol=5e-6)
    _close(list(r.model_info["weights"].values()), case["weights"], rtol=0.0, atol=5e-6)
    _close(
        _period_matrix(r, "counterfactual"),
        case["counterfactual_quantiles"],
        rtol=1e-13,
    )
    _close(_period_matrix(r, "effect"), case["quantile_diff"], rtol=0.0, atol=1e-12)


def test_discos_permutation_matches_patched_disco():
    case = _disco_case("quantile")
    r = _run_disco(case, placebo=True)
    perm = r.model_info["permutation"]
    _close(perm["placebo_wasserstein_sq"], case["patched_perm_control_dist"], rtol=1e-9)
    _close(
        list(r.model_info["wasserstein_sq"].values()),
        case["perm_target_dist"],
        rtol=1e-9,
    )
    _close(perm["p_value"], case["patched_perm_p_value"], rtol=1e-12)
    _close(perm["p_value"], case["perm_p_value"], rtol=1e-12)


def test_discos_r_permutation_defect_mechanism():
    """Rebuild DiSCos 0.1.4's placebo distances from StatsPAI's pieces with
    the original treated unit's quantile column zeroed (its bug) -> R's
    unpatched numbers; the correct pool gives different numbers."""
    case = _disco_case("quantile")
    G, T0 = case["G"], case["t0"] - 1
    evgrid = np.linspace(0.0, 1.0, G + 1)
    ctrl = case["control_ids"]

    def cell(u, t):
        return np.sort(MICRO.loc[(MICRO.id == u) & (MICRO.time == t), "y"].to_numpy())

    for idx, ref_row in enumerate(case["perm_control_dist"]):
        pool_ids = [1] + [c for k, c in enumerate(ctrl) if k != idx]
        lam = np.mean(
            [
                _disco_quantile_weights(
                    [cell(u, t + 1) for u in pool_ids],
                    cell(ctrl[idx], t + 1),
                    np.asarray(case["perm_nodes"][idx][t]),
                    False,
                )
                for t in range(T0)
            ],
            axis=0,
        )
        buggy = []
        for t in range(1, 7):
            Q = np.column_stack([_quant7_sorted(cell(u, t), evgrid) for u in pool_ids])
            Q[:, 0] = 0.0
            tq = _quant7_sorted(cell(ctrl[idx], t), evgrid)
            buggy.append(np.mean((Q @ lam - tq) ** 2))
        _close(buggy, ref_row, rtol=1e-9)
    patched = np.asarray(case["patched_perm_control_dist"], dtype=float)
    unpatched = np.asarray(case["perm_control_dist"], dtype=float)
    assert np.max(np.abs(patched - unpatched) / unpatched) > 0.1


def test_discos_recovers_copy_of_a_donor():
    """Treated sample = donor 3's sample in every period: w = e_3, zero
    effect at every quantile (unique: donor quantile functions independent)."""
    df = MICRO[MICRO.id != 1].copy()
    twin = df[df.id == 3].assign(id=1)
    r = sp.discos(
        pd.concat([df, twin]),
        "y",
        "id",
        "time",
        treated_unit=1,
        treatment_time=4,
        M=200,
        n_quantiles=100,
        placebo=False,
    )
    w = r.model_info["weights"]
    _close([w[k] for k in sorted(w)], [0, 1, 0, 0, 0], rtol=0.0, atol=1e-9)
    assert abs(r.estimate) < 1e-9


def test_discos_aggregate_panel_falls_back_with_warning():
    df = sp.california_prop99()
    with pytest.warns(UserWarning, match="not the DiSCo estimator"):
        r = sp.discos(
            df,
            "packspercapita",
            "state",
            "year",
            treated_unit="California",
            treatment_time=1989,
            placebo=False,
        )
    assert r.model_info["estimator"] == "time_series_quantiles_fallback"


def test_discos_needs_enough_observations():
    df = MICRO[(MICRO.id != 1) | (MICRO.index % 125 == 0)]
    with pytest.raises(ValueError, match="at least"):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            sp.discos(df, "y", "id", "time", treated_unit=1, treatment_time=4)
