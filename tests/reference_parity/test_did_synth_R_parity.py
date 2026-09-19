"""did_synth family vs the R packages written by the methods' own authors.

Three reference fixtures, each written by a committed generator from bytes
committed here (``_fixtures/_generate_did_synth_data.py`` writes the
simulated inputs; the Track A mpdta and Prop. 99 bytes are reused as is):

``did_synth_R.json`` -- ``_generate_did_synth_R.R``, synthdid 0.0.9
    ``sp.sc_estimate`` / ``sp.did_estimate`` / ``sp.synthdid_estimate`` against
    ``synthdid::sc_estimate`` / ``did_estimate`` / ``synthdid_estimate`` on
    the Prop. 99 replica (Track A ``12_sdid.csv``, one treated unit) and on
    ``did_synth_sdid_panel.csv`` (five treated units). Estimate, unit weights
    omega, time weights lambda and the regularisation constants at 1e-9
    relative: Frank-Wolfe is run with the same start, step rule, stopping
    rule and sparsification on both sides, so the two agree to rounding (a
    looser budget than 1e-12 only because FW's stopping test compares a
    difference of objective values to ``min.decrease^2`` and a last-bit
    difference can move the stopping iteration). The jackknife SE is
    deterministic and pinned at the same budget. The placebo and bootstrap
    SEs are Monte-Carlo draws from R's stream; each of R's 40 recorded draws
    is replayed through the Python replication map and must reproduce R's
    replication estimate at 1e-9, which pins every convention of
    ``synthdid::vcov`` (warm start from the renormalised full-sample weights,
    frozen regularisation constants, which units play treated, divisor ``r``).
    The end-to-end seeded SE is compared with R's only within pooled
    Monte-Carlo error (T3).

``did_synth_twfew_R.json`` -- ``_generate_did_synth_twfew_R.R``,
TwoWayFEWeights 2.1.0 and bacondecomp 0.1.1
    ``statspai.did._twfe_weights.dcdh_fe_weights`` (de Chaisemartin &
    D'Haultfoeuille ``feTR`` weights) against ``twowayfeweights`` on mpdta and
    on an unbalanced grouped panel with and without sampling weights: every
    cell weight, beta, the counts and sums of positive / negative weights and
    both summary measures at 1e-10. The R side is run with fixest's demeaning
    tolerance tightened to 1e-11 (its floor is 2.2e-12); at the default 1e-6
    the reference itself carries ~2e-8 of convergence error, which the
    ``_default_tol`` block records and the test bounds. ``sp.twfe_decomposition``
    -- which does NOT yet use this primitive -- is checked against the same
    two references and the gaps are pinned as a strict ``xfail`` (see
    ``docs/dev/campaign_phase3/did_synth.md``: the function body lives in
    ``did/wooldridge_did.py``, owned by another line of work).

``did_synth_honest_R.json`` -- ``_generate_did_synth_honest_R.R``,
HonestDiD 0.2.8
    ``sp.breakdown_m`` / the FLCI under ``Delta^SD(M)`` against
    ``HonestDiD::findOptimalFLCI`` on a hand-crafted 3 + 3 event study, twice:
    HonestDiD as shipped, and HonestDiD with its Monte-Carlo folded-normal
    quantile (``.qfoldednormal``: 1e6 draws at seed 0) replaced by the exact
    quantile. Against the exact-quantile run the breakdown value agrees at
    1e-9 and the FLCI half-length at 1e-8; the CI *bounds* agree only to
    ~2e-5, because HonestDiD's derivative bisection over ``h`` stops at a
    step of ``(h0 - hMin) / 100`` and the estimator's centre moves to first
    order in ``h`` while the half-length -- minimised over ``h`` -- does not.
    Against the shipped HonestDiD the breakdown agrees to 5e-4, which is the
    Monte-Carlo error of that quantile (1.96224 vs 1.95996 at mu = 0).
"""

from __future__ import annotations

import importlib
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.did._flci import breakdown_m_sd, event_study_moments, flci_delta_sd
from statspai.did._twfe_weights import dcdh_fe_weights

_sdid_mod = importlib.import_module("statspai.synth.sdid")

_FIX = Path(__file__).parent / "_fixtures"
_TRACK_A = Path(__file__).resolve().parents[1] / "r_parity" / "data"


def _load(name: str) -> dict:
    return json.loads((_FIX / name).read_text(encoding="utf-8"))


def _rel(ours, ref) -> float:
    a = np.asarray(ours, dtype=float).ravel()
    b = np.asarray(ref, dtype=float).ravel()
    assert a.shape == b.shape, (a.shape, b.shape)
    return float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-300)))


# ==========================================================================
#  synthdid: sp.sc_estimate / sp.did_estimate / sp.synthdid_estimate
# ==========================================================================

SD = _load("did_synth_R.json")
_DATASETS = {
    "prop99": dict(
        path=_TRACK_A / "12_sdid.csv",
        unit="state",
        time="year",
        y="cigsale",
        treat="treated",
    ),
    "panel": dict(
        path=_FIX / "did_synth_sdid_panel.csv",
        unit="unit",
        time="year",
        y="y",
        treat="treated",
    ),
}
_WRAPPERS = {
    "sdid": sp.synthdid_estimate,
    "sc": sp.sc_estimate,
    "did": sp.did_estimate,
}


def _panel(ds: str):
    """(long frame, Y in R's row order, N0, T0, treated units, first period)."""
    meta = _DATASETS[ds]
    df = pd.read_csv(meta["path"])
    ref = SD[ds]
    wide = df.pivot_table(index=meta["unit"], columns=meta["time"], values=meta["y"])
    Y = wide.loc[ref["units"], ref["times"]].to_numpy(dtype=float)
    treated = ref["units"][ref["N0"] :]
    first = ref["times"][ref["T0"]]
    return df, Y, ref["N0"], ref["T0"], treated, first


def _fit(ds: str, method: str, **kw):
    meta = _DATASETS[ds]
    df, _, _, _, treated, first = _panel(ds)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return _WRAPPERS[method](
            df,
            y=meta["y"],
            unit=meta["unit"],
            time=meta["time"],
            treat_unit=treated if len(treated) > 1 else treated[0],
            treat_time=first,
            **kw,
        )


@pytest.mark.parametrize("ds", ["prop99", "panel"])
@pytest.mark.parametrize("method", ["sdid", "sc", "did"])
def test_synthdid_estimate_and_weights_match_r(ds, method):
    ref = SD[ds][method]
    res = _fit(ds, method, se_method="jackknife")
    assert _rel(res.estimate, ref["estimate"]) < 1e-9
    units = SD[ds]["units"][: SD[ds]["N0"]]
    w = res.model_info["unit_weights"].set_index("unit")["weight"]
    omega = w.loc[units].to_numpy()
    # Absolute on the weights: sparsified weights are exactly 0 on both sides.
    np.testing.assert_allclose(omega, ref["omega"], rtol=1e-9, atol=1e-12)
    lam = res.model_info["time_weights"].to_numpy()
    np.testing.assert_allclose(lam, ref["lambda"], rtol=1e-9, atol=1e-12)


@pytest.mark.parametrize("ds", ["prop99", "panel"])
@pytest.mark.parametrize("method", ["sdid", "sc", "did"])
def test_synthdid_regularisation_constants_match_r(ds, method):
    """zeta.omega / zeta.lambda / min.decrease -- the first thing to bisect."""
    ref = SD[ds][method]
    _, Y, N0, T0, _, _ = _panel(ds)
    opts = _sdid_mod._synthdid_opts(Y, N0, T0, method)
    assert _rel(opts["zeta_omega"], ref["zeta_omega"]) < 1e-12
    assert _rel(opts["zeta_lambda"], ref["zeta_lambda"]) < 1e-12
    assert _rel(opts["min_decrease"], ref["min_decrease"]) < 1e-12


@pytest.mark.parametrize("method", ["sdid", "sc", "did"])
def test_synthdid_jackknife_se_matches_r(method):
    """Fixed-weight leave-one-unit-out over treated AND control units."""
    ref = SD["panel"][method]
    res = _fit("panel", method, se_method="jackknife")
    assert _rel(res.se, ref["se_jackknife"]) < 1e-9


@pytest.mark.parametrize("method", ["sdid", "sc", "did"])
def test_synthdid_jackknife_undefined_with_one_treated_unit(method):
    """synthdid returns NA; so must we (with a warning), never a number."""
    assert SD["prop99"][method]["se_jackknife"] is None
    meta = _DATASETS["prop99"]
    df, _, _, _, treated, first = _panel("prop99")
    with pytest.warns(UserWarning, match="undefined"):
        res = _WRAPPERS[method](
            df,
            y=meta["y"],
            unit=meta["unit"],
            time=meta["time"],
            treat_unit=treated[0],
            treat_time=first,
            se_method="jackknife",
        )
    assert np.isnan(res.se) and np.isnan(res.pvalue)


def _replay(ds: str, method: str, kind: str):
    ref = SD[ds][method]
    res = _fit(ds, method, se_method="jackknife")
    _, Y, N0, T0, _, _ = _panel(ds)
    units = SD[ds]["units"][:N0]
    w = res.model_info["unit_weights"].set_index("unit")["weight"]
    omega = w.loc[units].to_numpy(dtype=float)
    lam = res.model_info["time_weights"].to_numpy(dtype=float)
    opts = _sdid_mod._synthdid_opts(Y, N0, T0, method)
    theta = (
        _sdid_mod._synthdid_placebo_theta
        if kind == "placebo"
        else _sdid_mod._synthdid_bootstrap_theta
    )
    ours = np.array(
        [
            theta(Y, N0, T0, omega, lam, opts, np.asarray(ind, dtype=int) - 1)
            for ind in ref[f"{kind}_ind"]
        ]
    )
    return ours, np.asarray(ref[f"{kind}_theta"], dtype=float), ref


@pytest.mark.parametrize("ds", ["prop99", "panel"])
@pytest.mark.parametrize("method", ["sdid", "sc", "did"])
def test_synthdid_placebo_replications_match_r_draw_for_draw(ds, method):
    ours, theirs, ref = _replay(ds, method, "placebo")
    assert ours.size == theirs.size == SD["provenance"]["n_draws"]
    assert _rel(ours, theirs) < 1e-9
    # The SE formula applied to those draws: sd with divisor r.
    r = ours.size
    se = np.sqrt((r - 1) / r) * np.std(ours, ddof=1)
    assert _rel(se, ref["placebo_se_draws"]) < 1e-9


@pytest.mark.parametrize("method", ["sdid", "sc", "did"])
def test_synthdid_bootstrap_replications_match_r_draw_for_draw(method):
    ours, theirs, ref = _replay("panel", method, "bootstrap")
    assert ours.size == theirs.size == SD["provenance"]["n_draws"]
    assert _rel(ours, theirs) < 1e-9
    r = ours.size
    se = np.sqrt((r - 1) / r) * np.std(ours, ddof=1)
    assert _rel(se, ref["bootstrap_se_draws"]) < 1e-9


@pytest.mark.parametrize("method", ["sdid", "did"])
def test_synthdid_placebo_se_within_monte_carlo_error_of_r(method):
    """T3 only: two independent 200-draw Monte-Carlo estimates of one SE.

    The SE of a placebo SE estimated from r draws is roughly se / sqrt(2 r)
    (normal-theory approximation); the pooled error of the difference of two
    independent r = 200 estimates is se / sqrt(r) = 7% of se. Four pooled
    standard errors (28%) is the budget -- a statement about simulation
    error, not parity. ``sc`` is left out only for run time (its Frank-Wolfe
    problem takes ~10^4 iterations per replication); its replication map is
    pinned exactly above.
    """
    ref = SD["panel"][method]
    res = _fit("panel", method, se_method="placebo", n_reps=200, seed=11)
    pooled = ref["placebo_se_200"] / np.sqrt(200)
    assert abs(res.se - ref["placebo_se_200"]) < 4 * pooled


def test_did_estimate_is_the_plain_difference_in_means():
    """Reference-free identity: uniform weights reduce to the 2x2 DID."""
    _, Y, N0, T0, _, _ = _panel("panel")
    tau = (Y[N0:, T0:].mean() - Y[N0:, :T0].mean()) - (
        Y[:N0, T0:].mean() - Y[:N0, :T0].mean()
    )
    res = _fit("panel", "did", se_method="jackknife")
    assert abs(res.estimate - tau) < 1e-12


def test_sdid_covariates_are_refused_not_ignored():
    df, _, _, _, treated, first = _panel("panel")
    df = df.assign(x=np.arange(len(df), dtype=float))
    with pytest.raises(sp.exceptions.MethodIncompatibility, match="covariate"):
        sp.sdid(
            df,
            outcome="y",
            unit="unit",
            time="year",
            treated_unit=treated,
            treatment_time=first,
            covariates=["x"],
        )


# ==========================================================================
#  de Chaisemartin-D'Haultfoeuille TWFE weights (TwoWayFEWeights)
# ==========================================================================

TW = _load("did_synth_twfew_R.json")


def _mpdta():
    df = pd.read_csv(_TRACK_A / "16_bjs.csv")
    df["D"] = ((df["first_treat"] > 0) & (df["year"] >= df["first_treat"])).astype(int)
    return df


_TWFEW_CASES = {
    "twfew_mpdta": (lambda: _mpdta(), ("lemp", "countyreal", "year", "D"), {}),
    "twfew_grouped": (
        lambda: pd.read_csv(_FIX / "did_synth_twfew_panel.csv"),
        ("y", "g", "t", "d"),
        {},
    ),
    "twfew_grouped_weighted": (
        lambda: pd.read_csv(_FIX / "did_synth_twfew_panel.csv"),
        ("y", "g", "t", "d"),
        {"weights": "w"},
    ),
}


@pytest.mark.parametrize("case", sorted(_TWFEW_CASES))
def test_dcdh_fe_weights_match_twowayfeweights(case):
    make, args, kw = _TWFEW_CASES[case]
    ours = dcdh_fe_weights(make(), *args, **kw)
    ref = TW[case]
    for key in ("nr_plus", "nr_minus", "nr_weights", "tot_cells"):
        assert ours[key] == ref[key], key
    for key in ("beta", "sum_plus", "sum_minus", "sensibility", "sensibility2"):
        assert _rel(ours[key], ref[key]) < 1e-10, key
    cells = ours["cells"].sort_values(["group", "time"])
    np.testing.assert_array_equal(cells["group"].to_numpy(), ref["cells"]["G"])
    np.testing.assert_array_equal(cells["time"].to_numpy(), ref["cells"]["T"])
    scale = np.max(np.abs(ref["cells"]["weight"]))
    assert np.max(np.abs(cells["weight"].to_numpy() - ref["cells"]["weight"])) < (
        1e-10 * scale
    )


def test_twowayfeweights_default_tolerance_is_solver_error_only():
    """The default-tolerance reference differs from the converged one by
    fixest's demeaning error (~2e-8 here), not by a convention: bound it."""
    ours = dcdh_fe_weights(
        pd.read_csv(_FIX / "did_synth_twfew_panel.csv"), "y", "g", "t", "d"
    )
    ref = TW["twfew_grouped_default_tol"]
    gap = max(
        _rel(ours[k], ref[k])
        for k in ("beta", "sum_plus", "sum_minus", "sensibility", "sensibility2")
    )
    assert 1e-12 < gap < 1e-6


def test_dcdh_weights_identities():
    """Reference-free: weights sum to one over treated cells and reproduce
    beta as sum_gt w_gt * ATT_gt when every cell effect is known."""
    df = pd.read_csv(_FIX / "did_synth_twfew_panel.csv")
    out = dcdh_fe_weights(df, "y", "g", "t", "d")
    assert abs(out["cells"]["weight"].sum() - 1.0) < 1e-12
    # Plant a known heterogeneous effect on a noiseless outcome: Y = a_g +
    # b_t + D * tau_gt. The TWFE coefficient must equal sum w_gt tau_gt.
    rng = np.random.default_rng(0)
    g_fe = dict(zip(df["g"].unique(), rng.normal(size=df["g"].nunique())))
    t_fe = dict(zip(df["t"].unique(), rng.normal(size=df["t"].nunique())))
    cell_mean_d = df.groupby(["g", "t"])["d"].transform("mean")
    tau = 1.0 + 0.2 * df["t"] + 0.1 * df["g"]
    df2 = df.assign(
        d=cell_mean_d,
        y=df["g"].map(g_fe) + df["t"].map(t_fe) + cell_mean_d * tau,
        tau=tau,
    )
    out2 = dcdh_fe_weights(df2, "y", "g", "t", "d")
    cells = out2["cells"].merge(
        df2.groupby(["g", "t"], as_index=False)["tau"].first(),
        left_on=["group", "time"],
        right_on=["g", "t"],
    )
    assert abs(out2["beta"] - float((cells["weight"] * cells["tau"]).sum())) < 1e-10


def test_twfe_decomposition_beta_matches_r():
    """The TWFE coefficient sp.twfe_decomposition reports is right."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sp.twfe_decomposition(
            _mpdta(),
            y="lemp",
            group="countyreal",
            time="year",
            first_treat="first_treat",
        )
    assert _rel(res.model_info["twfe_beta"], TW["twfew_mpdta"]["beta"]) < 1e-10
    assert _rel(TW["bacon_mpdta"]["weighted_sum"], TW["twfew_mpdta"]["beta"]) < 1e-10


@pytest.mark.xfail(
    strict=True,
    reason=(
        "DEFECT (open; fix belongs to the owner of did/wooldridge_did.py): "
        "sp.twfe_decomposition's 'Bacon' rows use ad hoc windows and "
        "n_units weights, its headline is not the TWFE beta, and its 'dCDH' "
        "weights are not de Chaisemartin-D'Haultfoeuille's. See "
        "docs/dev/campaign_phase3/did_synth.md."
    ),
)
def test_twfe_decomposition_matches_bacon_and_twowayfeweights():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sp.twfe_decomposition(
            _mpdta(),
            y="lemp",
            group="countyreal",
            time="year",
            first_treat="first_treat",
        )
    # Headline: a decomposition of beta must sum to beta.
    assert _rel(res.estimate, TW["bacon_mpdta"]["weighted_sum"]) < 1e-8
    ref = pd.DataFrame(TW["bacon_mpdta"])
    ref["control"] = ref["untreated"].where(ref["untreated"] != 99999, -1)
    det = res.detail.copy()
    det["control"] = pd.to_numeric(det["control_cohort"], errors="coerce").fillna(-1)
    got = det.set_index(["treated_cohort", "control"]).sort_index()
    exp = ref.set_index(["treated", "control"]).sort_index()
    np.testing.assert_allclose(got["estimate"], exp["estimate"], rtol=1e-8)
    np.testing.assert_allclose(got["weight"], exp["weight"], rtol=1e-8)
    # dCDH: the reference has 125 negative weights on mpdta.
    assert res.model_info["n_negative_weights_dcdh"] == TW["twfew_mpdta"]["nr_minus"]


# ==========================================================================
#  sp.breakdown_m / FLCI (HonestDiD)
# ==========================================================================

HD = _load("did_synth_honest_R.json")
_ES = pd.read_csv(_FIX / "did_synth_honest_es.csv")
_B = _ES["betahat"].to_numpy(dtype=float)
_S = _ES[[f"s{j}" for j in range(6)]].to_numpy(dtype=float)


@pytest.mark.parametrize("e", [0, 1, 2])
def test_breakdown_matches_honestdid_with_exact_quantile(e):
    bd = breakdown_m_sd(_B, _S, 3, 3, l_post=np.eye(3)[e], alpha=0.05)
    assert _rel(bd, HD["exactq"][f"e{e}"]["breakdown"]) < 1e-9


@pytest.mark.parametrize("e", [0, 1, 2])
def test_breakdown_matches_shipped_honestdid_within_its_quantile_error(e):
    """aligned, not bit-exact: HonestDiD simulates the folded-normal
    quantile (1.96224 vs exact 1.95996 at mu = 0, 1.2e-3)."""
    bd = breakdown_m_sd(_B, _S, 3, 3, l_post=np.eye(3)[e], alpha=0.05)
    assert _rel(bd, HD["default"][f"e{e}"]["breakdown"]) < 1e-3


@pytest.mark.parametrize("e", [0, 1, 2])
def test_flci_matches_honestdid_with_exact_quantile(e):
    ref = HD["exactq"][f"e{e}"]
    for i, m in enumerate(ref["M"]):
        c = flci_delta_sd(_B, _S, 3, 3, m, l_post=np.eye(3)[e], alpha=0.05)
        half_ref = (ref["ci_upper"][i] - ref["ci_lower"][i]) / 2
        assert _rel(c.half_length, half_ref) < 1e-8, m
        # Bounds: HonestDiD's h search stops at a step of (h0 - hMin)/100
        # and the centre moves to first order in h (module docstring).
        assert (
            _rel([c.ci_lower, c.ci_upper], [ref["ci_lower"][i], ref["ci_upper"][i]])
            < 5e-5
        )


def test_breakdown_is_where_the_flci_touches_zero():
    """Reference-free: just below M* the FLCI excludes 0, just above it does not."""
    bd = breakdown_m_sd(_B, _S, 3, 3, l_post=np.eye(3)[1], alpha=0.05)
    below = flci_delta_sd(_B, _S, 3, 3, bd * (1 - 1e-6), l_post=np.eye(3)[1])
    above = flci_delta_sd(_B, _S, 3, 3, bd * (1 + 1e-6), l_post=np.eye(3)[1])
    assert below.ci_lower > 0 >= above.ci_lower


def test_public_breakdown_m_routes_to_the_flci_breakdown():
    """sp.breakdown_m on a Callaway-Sant'Anna fit inverts the FLCI that
    sp.honest_did reports for the same fit -- not the old closed form."""
    df = pd.read_csv(_TRACK_A / "16_bjs.csv")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        cs = sp.callaway_santanna(
            df, y="lemp", g="first_treat", t="year", i="countyreal"
        )
    beta, sigma, times = event_study_moments(cs)
    post = times >= 0
    order = np.r_[np.where(~post)[0], np.where(post)[0]]
    expected = breakdown_m_sd(
        beta[order],
        sigma[np.ix_(order, order)],
        int((~post).sum()),
        int(post.sum()),
        l_post=(times[post] == 0).astype(float),
    )
    got = sp.breakdown_m(cs, e=0)
    assert got == pytest.approx(expected, rel=1e-12)
    es = cs.model_info["event_study"].set_index("relative_time")
    old_closed_form = (abs(es.loc[0, "att"]) - 1.959963984540054 * es.loc[0, "se"]) / 1
    assert abs(got - old_closed_form) / old_closed_form > 0.5
    # method is honoured: relative magnitudes gives a different number.
    with pytest.warns(UserWarning, match="relative_magnitude"):
        rm = sp.breakdown_m(cs, e=0, method="relative_magnitude")
    assert rm != pytest.approx(got, rel=1e-3)
