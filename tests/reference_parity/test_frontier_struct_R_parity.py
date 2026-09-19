"""Frontier / structural add-ons vs R ``sfaR`` / ``metafrontier`` and Stata ``markupest``.

References
----------
* ``_fixtures/frontier_struct_R.json`` -- ``_generate_frontier_struct_R.R``
  (R sfaR 1.0.1 ``sfalcmcross`` / ``sfacross``; metafrontier 0.3.1
  ``metafrontier``) on the CSVs written by
  ``_generate_frontier_struct_data.py``.
* ``_fixtures/frontier_struct_stata.json`` --
  ``_fixtures/_generate_frontier_struct_stata.do`` (Stata 18, SSC
  ``markupest`` 1.0.1 over ``prodest``) on ``_fixtures/prodest_panel.csv``.

What each block claims
----------------------
* ``sp.lcsf``: the same two-class half-normal latent-class likelihood as
  ``sfaR::sfalcmcross`` (class-1 probability ``logistic(z'theta)``). sfaR
  reports log VARIANCES, StatsPAI log standard deviations, so
  ``ln_sigma = Zu / 2`` and ``se(ln_sigma) = se(Zu) / 2``. Classes are matched
  by ascending sigma_u (StatsPAI's canonical labelling). Point estimates
  1e-6 rel (observed ~1e-8): both sides are optimisers, and the class-1
  sigma_u direction is flat (SE ~0.55 on ln sigma_u), so agreement is limited
  by the reference's own gradient (written in the fixture) -- the fixture
  point and StatsPAI's point have the same log-likelihood to 1e-9.
  SEs 1e-6 rel (observed ~9e-8): StatsPAI's OIM is a Richardson-extrapolated
  central-difference Hessian, sfaR's is analytic. (Before the fix, a plain
  h = 1e-5 difference left 1.4e-3 / 2.1e-3 relative error on the flat
  class's SEs and L-BFGS-B stopped 1.2e-3 / 1.4e-3 short on the estimates.)
* ``sp.metafrontier(envelope="own")``: ``metafrontier::metafrontier`` with
  ``engine="sfaR"``, ``objective="lp"``. Group frontiers are sfaR fits
  (1e-6), the LP metafrontier coefficients, TGR = exp(x'b_g - x'b*) and
  TE_meta = TE_group * TGR at 1e-6 (the LP vertex is a smooth function of the
  group betas, which carry the optimiser error). The default
  ``envelope="all"`` (envelopment of every group frontier at every
  observation) has no reference implementation; it is checked for its
  defining identities.
* ``sp.malmquist``: no package implements this parametric index (distance
  ``D^s(x, y) = exp(y - x'b_s)``, adjacent-period geometric mean). The
  per-period half-normal frontiers are pinned to ``sfaR::sfacross`` (1e-6),
  and the index arithmetic, re-done in R from sfaR's betas, is pinned at
  1e-6 (the frontier betas carry StatsPAI's ~1e-7 optimiser error).
* ``sp.markup``: Stata ``markupest, method(dlw) pmethod(lp)`` -- the free
  input's stage-1 OLS elasticity divided by its (corrected) expenditure share.
  Everything is closed-form given stage 1: 1e-10 rel (observed 8e-13
  corrected, 3e-15 uncorrected), on the 2 084 firm-years StatsPAI keeps;
  Stata also reports the 281 first-year rows StatsPAI's production sample
  drops.
* ``sp.zisf``: no R / Stata implementation (sfaR 1.0.1 exports no
  zero-inefficiency model; SSC ``zisf`` / ``sfzi`` return 601). Checked for
  nesting (log-likelihood >= the half-normal SFA it nests), a zero score at
  the reported optimum, and recovery of the known DGP.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.frontier import _core as _fc

_FIX = Path(__file__).parent / "_fixtures"
R = json.loads((_FIX / "frontier_struct_R.json").read_text(encoding="utf-8"))
ST = json.loads((_FIX / "frontier_struct_stata.json").read_text(encoding="utf-8"))


def _rel(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-300))


# ---------------------------------------------------------------------------
# Latent-class SFA
# ---------------------------------------------------------------------------


def _sfar_to_sp(block, has_z):
    """sfaR coefficient order -> StatsPAI order, ln-variance -> ln-sd."""
    c, s = block["coef"], block["se"]
    names = ["(Intercept)", "x1", "x2"]
    out_c, out_s = [], []
    for suffix in ("", ".1"):
        out_c += [c[n + suffix] for n in names]
        out_s += [s[n + suffix] for n in names]
        out_c += [c["Zv_(Intercept)" + suffix] / 2, c["Zu_(Intercept)" + suffix] / 2]
        out_s += [s["Zv_(Intercept)" + suffix] / 2, s["Zu_(Intercept)" + suffix] / 2]
    cl = ["Cl1_(Intercept)"] + (["Cl1_z"] if has_z else [])
    out_c += [c[k] for k in cl]
    out_s += [s[k] for k in cl]
    return np.array(out_c), np.array(out_s)


def _canonical(params, se, k_beta=3):
    """Order classes by ascending sigma_u, as sp.lcsf does."""
    b = k_beta + 2
    if params[k_beta + 1] > params[b + k_beta + 1]:
        params = np.r_[params[b : 2 * b], params[:b], -params[2 * b :]]
        se = np.r_[se[b : 2 * b], se[:b], se[2 * b :]]
    return params, se


@pytest.fixture(scope="module")
def lc_data():
    return pd.read_csv(_FIX / "frontier_struct_lcsf.csv")


@pytest.mark.parametrize(
    "key, cost, z", [("lcsf_prod_z", False, True), ("lcsf_cost_noz", True, False)]
)
def test_lcsf_matches_sfalcmcross(lc_data, key, cost, z):
    df = lc_data.assign(c=-lc_data["y"])
    fit = sp.lcsf(
        df,
        y="c" if cost else "y",
        x=["x1", "x2"],
        z_class=["z"] if z else None,
        cost=cost,
    )
    ref_p, ref_s = _canonical(*_sfar_to_sp(R[key], z))
    assert fit.diagnostics["log_likelihood"] >= R[key]["loglik"] - 1e-9
    assert _rel(fit.params.to_numpy(), ref_p) < 1e-6
    assert _rel(fit.std_errors.to_numpy(), ref_s) < 1e-6


def test_lcsf_cost_is_mirror_of_production(lc_data):
    """Identity: a cost frontier on -y is the production frontier on y."""
    prod = sp.lcsf(lc_data, y="y", x=["x1", "x2"])
    cost = sp.lcsf(lc_data.assign(y=-lc_data["y"]), y="y", x=["x1", "x2"], cost=True)
    p, c = prod.params.to_numpy(), cost.params.to_numpy()
    flip = np.ones_like(p)
    flip[[0, 1, 2, 5, 6, 7]] = -1.0  # betas change sign, variances do not
    np.testing.assert_allclose(c, flip * p, rtol=1e-6, atol=1e-8)
    assert (
        abs(prod.diagnostics["log_likelihood"] - cost.diagnostics["log_likelihood"])
        < 1e-8
    )


# ---------------------------------------------------------------------------
# Zero-inefficiency SFA (no reference implementation)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def zi_fit():
    df = pd.read_csv(_FIX / "frontier_struct_zisf.csv")
    return df, sp.zisf(df, y="y", x=["x1", "x2"])


def test_zisf_nests_half_normal_sfa(zi_fit):
    df, fit = zi_fit
    sfa = sp.frontier(df, y="y", x=["x1", "x2"])
    assert fit.diagnostics["log_likelihood"] >= sfa.diagnostics["log_likelihood"] - 1e-9


def test_zisf_score_is_zero_at_reported_optimum(zi_fit):
    df, fit = zi_fit
    y = df["y"].to_numpy()
    X = np.column_stack([np.ones(len(df)), df[["x1", "x2"]].to_numpy()])

    def nll(t):
        b, th, lsv, lsu = t[:3], t[3], t[4], t[5]
        eps = y - X @ b
        sv, su = np.exp(lsv), np.exp(lsu)
        from scipy import stats

        lf_eff = stats.norm.logpdf(eps, scale=sv)
        lf_in = _fc.loglik_halfnormal(eps, np.full(len(y), sv), np.full(len(y), su), -1)
        lp = -np.logaddexp(0.0, -th)
        l1p = -np.logaddexp(0.0, th)
        return -np.logaddexp(lp + lf_eff, l1p + lf_in).sum()

    t = fit.params.to_numpy()
    assert abs(nll(t) + fit.diagnostics["log_likelihood"]) < 1e-9
    g = _fc.central_gradient(nll, t)
    assert np.max(np.abs(g)) < 1e-4


def test_zisf_recovers_dgp(zi_fit):
    _, fit = zi_fit
    truth = {
        "_cons": 1.0,
        "x1": 0.6,
        "x2": 0.3,
        "p__cons": np.log(0.4 / 0.6),
        "ln_sigma_v": np.log(0.15),
        "ln_sigma_u": np.log(0.5),
    }
    for k, v in truth.items():
        assert abs(fit.params[k] - v) < 3.5 * fit.std_errors[k], k


# ---------------------------------------------------------------------------
# Metafrontier
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def meta_data():
    return pd.read_csv(_FIX / "frontier_struct_meta.csv")


def test_metafrontier_own_envelope_matches_R(meta_data):
    ref = R["meta"]
    fit = sp.metafrontier(
        meta_data, y="y", x=["x1", "x2"], group="group", envelope="own"
    )
    for g, b in ref["group_coef"].items():
        assert (
            _rel(fit.beta_groups[g].to_numpy(), [b["(Intercept)"], b["x1"], b["x2"]])
            < 1e-6
        )
    mc = ref["meta_coef"]
    assert (
        _rel(fit.beta_meta.to_numpy(), [mc["(Intercept)"], mc["x1"], mc["x2"]]) < 1e-6
    )
    assert list(meta_data["group"]) == ref["group"]
    assert _rel(fit.tgr.to_numpy(), ref["tgr"]) < 1e-6
    assert _rel(fit.te_group.to_numpy(), ref["te_group"]) < 1e-6
    assert _rel(fit.te_meta.to_numpy(), ref["te_meta"]) < 1e-6


def test_metafrontier_all_envelope_identities(meta_data):
    fit = sp.metafrontier(meta_data, y="y", x=["x1", "x2"], group="group")
    X = np.column_stack([np.ones(len(meta_data)), meta_data[["x1", "x2"]].to_numpy()])
    meta = X @ fit.beta_meta.to_numpy()
    for b in fit.beta_groups.values():
        assert np.all(meta >= X @ b.to_numpy() - 1e-7)
    own = sp.metafrontier(
        meta_data, y="y", x=["x1", "x2"], group="group", envelope="own"
    )
    # a superset of constraints can only raise the minimised sum of X'b*
    assert (
        X.sum(axis=0) @ fit.beta_meta.to_numpy()
        >= X.sum(axis=0) @ own.beta_meta.to_numpy() - 1e-9
    )
    assert np.all((fit.tgr > 0) & (fit.tgr <= 1))
    np.testing.assert_allclose(fit.te_meta, fit.te_group * fit.tgr, rtol=1e-12)


@pytest.mark.parametrize("envelope", ["all", "own"])
def test_metafrontier_cost_is_mirror_of_production(meta_data, envelope):
    """Regression: cost TGR was exp(own - meta) >= 1, clipped to 1 everywhere."""
    prod = sp.metafrontier(
        meta_data, y="y", x=["x1", "x2"], group="group", envelope=envelope
    )
    cost = sp.metafrontier(
        meta_data.assign(y=-meta_data["y"]),
        y="y",
        x=["x1", "x2"],
        group="group",
        cost=True,
        envelope=envelope,
    )
    np.testing.assert_allclose(cost.beta_meta, -prod.beta_meta, rtol=1e-6, atol=1e-8)
    np.testing.assert_allclose(cost.tgr, prod.tgr, rtol=1e-6)
    assert cost.tgr.min() < 0.5  # not the degenerate all-ones TGR


# ---------------------------------------------------------------------------
# Malmquist
# ---------------------------------------------------------------------------


def test_malmquist_matches_R_recomputation():
    df = pd.read_csv(_FIX / "frontier_struct_malm.csv")
    fit = sp.malmquist(df, y="y", x=["x1", "x2"], id="id", time="t")
    for t, f in fit.period_frontiers.items():
        ref = R["malmquist"]["betas"][str(t)]
        assert _rel(f.params[["_cons", "x1", "x2"]].to_numpy(), ref["beta"]) < 1e-6
    it = fit.index_table.sort_values(["id", "t_from"]).reset_index(drop=True)
    ri = pd.DataFrame(R["malmquist"]["index"])
    assert (it["id"].to_numpy() == ri["id"].to_numpy()).all()
    for col in ("m_index", "ec", "tc"):
        assert _rel(it[col].to_numpy(), ri[col].to_numpy()) < 1e-6, col
    # identity, independent of any reference
    np.testing.assert_allclose(it["m_index"], it["ec"] * it["tc"], rtol=1e-12)


# ---------------------------------------------------------------------------
# De Loecker-Warzynski markups
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def lp_fit():
    panel = pd.read_csv(_FIX / "prodest_panel.csv")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.levinsohn_petrin(
            panel,
            output="y",
            free="l",
            state="k",
            proxy="m",
            panel_id="id",
            time="year",
            polynomial_degree=3,
        )


@pytest.mark.parametrize("key, corr", [("corrected", True), ("uncorrected", False)])
def test_markup_matches_markupest(lp_fit, key, corr):
    ref = ST[key]
    assert abs(lp_fit.coef["l"] - ref["b_l"]) / abs(ref["b_l"]) < 1e-10
    mu = sp.markup(
        lp_fit, revenue="y", input_cost="l", flexible_input="l", correct_eta=corr
    )
    got = lp_fit.sample[["id", "year"]].assign(mu=mu.to_numpy())
    st = pd.DataFrame(ref["rows"], columns=["id", "year", "mu"])
    m = got.merge(st, on=["id", "year"], suffixes=("", "_st"))
    assert len(m) == len(got) == 2084
    assert ref["n"] == 2365
    assert _rel(m["mu"].to_numpy(), m["mu_st"].to_numpy()) < 1e-10


def test_markup_identity(lp_fit):
    """mu = theta * exp(log_rev - eta - log_cost), reference-free."""
    s = lp_fit.sample
    mu = sp.markup(lp_fit, revenue="y", input_cost="l", flexible_input="l")
    expect = lp_fit.coef["l"] * np.exp(s["y"] - s["eta"] - s["l"])
    np.testing.assert_allclose(mu.to_numpy(), expect.to_numpy(), rtol=1e-14)
