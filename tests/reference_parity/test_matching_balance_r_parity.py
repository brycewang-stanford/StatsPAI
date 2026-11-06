"""R parity for the propensity-score tools of ``statspai.matching``.

``sp.propensity_score``, ``sp.trimming``, ``sp.ps_weights``, ``sp.ess``,
``sp.energy_distance``, ``sp.implied_weights``, ``sp.balance_diagnostics``,
``sp.ps_balance``, ``sp.bal_tab`` (options the WeightIt / cobalt fixture
leaves out) and ``sp.l1_imbalance``, each against the R package that
defines the quantity: ``stats::glm``, ``propensity`` 0.1.0, ``halfmoon``
0.2.0, ``WeightIt`` 1.7.0, ``cobalt`` 4.6.2, ``lmw`` 0.0.2 and ``cem``
1.1.31 (all GPL, run as black boxes).

Reference numbers: ``_fixtures/matching_balance_R.json``, written by
``_fixtures/_generate_matching_balance_R.R`` on the simulated frame
``_fixtures/matching_balance_data.csv`` (written and read back by that
script, so both sides read the same CSV bytes; the versions it ran are in
the JSON).

Tolerances. Everything is deterministic and held to 1e-9 relative (most
gaps are 1e-14). Statistics that are zero by construction (the overlap
weights balance the means exactly) are compared on an absolute 1e-12.
The items that are not "same number", with their mechanism:

* Probit scores: 1e-8. R ``glm(binomial("probit"))`` stops with a score of
  order 1e-7 at ``epsilon = 1e-14``; ours is the maximum (score 1e-12,
  asserted).
* Crump trimming: the sample equation ``gamma = 2 mean(g | g <= gamma)``
  can have several roots. With one root (the probit fit here) the cutoff
  and the kept rows are R's. With several (the logit fit) both packages
  return a root. StatsPAI returns the root with the smallest sample
  variance objective ``sum(g[:k]) / k^2`` (smallest root on a tie; before
  1.43 it always took the smallest root). ``propensity`` and ``PSweight``
  agree with each other and pick the smallest, the largest or one in
  between depending on the data (60 multi-root draws, 2026-11-05); their
  pick is the objective minimiser in 27 of 43 multi-root draws
  (2026-11-06), so there is no rule to follow. Asserted: each cutoff
  solves the equation, ours minimises the objective over all roots, and
  the kept sets differ by the rows between the two.
* Continuous-exposure weights: ``propensity::wt_ate`` and
  ``WeightIt::weightit(method = "glm")`` give the same weights (7e-15).
  Both divide standard normal densities of standardized values: the
  residual scaled by its maximum-likelihood standard deviation
  (``propensity`` ignores ``.sigma=`` for a vector of means) and, when
  stabilised, the exposure scaled by its ML standard deviation. That is
  the ratio of the normal densities times the constant
  ``sd_ML(A) / sigma_ML``. ``sp.ps_weights`` is the ratio of the normal
  densities themselves, ``f(A) / f(A | X)`` as in Robins, Hernan and
  Brumback (2000), with the ``sigma`` it is given and ``ddof = 1`` in
  the numerator (owner decision 2026-11-06: kept, a documented
  difference). Asserted: R's weights rebuilt from ours under R's scale
  choice.
* A 0/1 covariate in ``sp.balance_diagnostics`` / ``sp.ps_balance`` has the
  sample variance; cobalt's ``binary = "std"`` uses ``p (1 - p)``
  (owner decision 10, 2026-10-20). cobalt's own ``col_w_*`` with
  ``bin.vars = FALSE`` is matched exactly; ``bal.tab``'s binary row is
  rebuilt from our means.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import statspai as sp
from statspai.matching.ps_diagnostics import _crump_alpha

FIX = Path(__file__).parent / "_fixtures"
X = ["x1", "x2", "x3", "b"]
TIGHT = 1e-9
ZERO = 1e-12


@pytest.fixture(scope="module")
def ref():
    return json.loads((FIX / "matching_balance_R.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def df():
    return pd.read_csv(FIX / "matching_balance_data.csv")


@pytest.fixture(scope="module")
def ps(ref):
    return np.asarray(ref["ps"]["logit"], dtype=float)


@pytest.fixture(scope="module")
def weights(ref):
    return {k: np.asarray(v, dtype=float) for k, v in ref["weights"].items()}


def _close(got, want, rel=TIGHT, zero=ZERO):
    got, want = np.asarray(got, dtype=float), np.asarray(want, dtype=float)
    assert got.shape == want.shape
    gap = np.abs(got - want)
    ok = (gap <= rel * np.abs(want)) | (gap <= zero)
    assert ok.all(), (got[~ok], want[~ok])


# --- propensity scores ------------------------------------------------------


def test_logit_scores_match_glm(df, ref):
    _close(sp.propensity_score(df, "t", X).to_numpy(), ref["ps"]["logit"], rel=1e-12)


def test_probit_scores_match_glm_and_are_the_maximum(df, ref):
    """⚠️ Through 1.42.1 the probit branch stopped BFGS at a gradient of
    1e-5 and was 6e-8 from glm; it now uses the shared Newton fit."""
    p = sp.propensity_score(df, "t", X, method="probit").to_numpy()
    _close(p, ref["ps"]["probit"], rel=1e-8)
    design = np.column_stack([np.ones(len(df)), df[X].to_numpy(dtype=float)])
    t = df["t"].to_numpy(dtype=float)
    eta = stats.norm.ppf(p)
    lam = np.where(
        t == 1, stats.norm.pdf(eta) / p, -stats.norm.pdf(eta) / (1 - p)
    )  # the probit score residual
    assert np.max(np.abs(design.T @ lam)) < 1e-10


# --- trimming -----------------------------------------------------------------


def _roots(p):
    g = np.sort(1.0 / (p * (1.0 - p)))
    run = 2.0 * np.cumsum(g) / np.arange(1, g.size + 1)
    return [run[k - 1] for k in range(1, g.size) if g[k - 1] <= run[k - 1] < g[k]]


def _crump_objective(g, gamma):
    """Sample variance objective of Crump et al. (2009) for the cutoff
    ``gamma``: ``mean(g 1{g <= gamma}) / mean(1{g <= gamma})^2``."""
    keep = g <= gamma
    return float(np.mean(g * keep) / np.mean(keep) ** 2)


def _grid_roots(g):
    """Roots of ``2 mean(g | g <= gamma) - gamma`` found on a grid.

    Between order statistics the function is linear with slope -1, so a
    sign change from + to - between two grid points brackets a root,
    which is then solved on that segment."""
    gs = np.sort(g)
    csum = np.cumsum(gs)
    grid = np.linspace(gs[0], gs[-1], 200_001)
    k = np.searchsorted(gs, grid, side="right")
    f = 2.0 * csum[k - 1] / k - grid
    cross = np.nonzero((f[:-1] >= 0) & (f[1:] < 0))[0]
    return [float(2.0 * csum[k[i] - 1] / k[i]) for i in cross]


def test_crump_cutoff_minimises_the_objective_over_several_roots():
    """⚠️ Through 1.42.1 the smallest root was returned. Here the equation
    has two roots, 12.5 and 53/3. They keep the 4 and the 6 smallest g,
    so the objective ``n sum(g[:k]) / k^2`` is 8 * 25 / 16 = 12.5 at the
    first and 8 * 53 / 36 at the second, which is the cutoff."""
    g = np.array([4.0, 5.0, 6.0, 10.0, 13.0, 15.0, 22.0, 26.0])
    p = 0.5 - np.sqrt(0.25 - 1.0 / g)
    p[::2] = 1.0 - p[::2]  # both tails, same g
    g = 1.0 / (p * (1.0 - p))
    roots = _grid_roots(g)
    assert roots == pytest.approx([12.5, 53.0 / 3.0], rel=1e-12)
    obj = [_crump_objective(g, r) for r in roots]
    assert obj == pytest.approx([8 * 25.0 / 16.0, 8 * 53.0 / 36.0], rel=1e-12)
    alpha = _crump_alpha(p)
    gamma = 1.0 / (alpha * (1.0 - alpha))
    assert gamma == pytest.approx(53.0 / 3.0, rel=1e-12)
    assert _crump_objective(g, gamma * (1 + 1e-12)) == pytest.approx(
        min(obj), rel=1e-12
    )
    df = pd.DataFrame({"t": [1, 0] * 4, "x": np.arange(8.0)})
    kept = sp.trimming(df, "t", ["x"], ps=pd.Series(p, index=df.index))
    assert sorted(kept.index) == sorted(np.nonzero(g <= 53.0 / 3.0 + 1e-9)[0])


def test_crump_trimming_with_one_root_matches_propensity(df, ref):
    p = np.asarray(ref["ps"]["probit"])
    assert len(_roots(p)) == 1
    r = ref["trim"]["crump_probit"]
    assert _crump_alpha(p) == pytest.approx(r["cutoff"], rel=1e-10)
    kept = sp.trimming(df, "t", X, ps_method="probit")
    assert (kept.index + 1).tolist() == r["kept"]
    given = sp.trimming(df, "t", X, ps=pd.Series(p, index=df.index))
    assert (given.index + 1).tolist() == r["kept"]


def test_crump_trimming_with_several_roots(df, ref, ps):
    """Both cutoffs solve the sample equation; ours has the smallest
    sample objective of all roots (here also the smallest root)."""
    roots = _roots(ps)
    assert len(roots) > 1
    theirs = ref["trim"]["crump_logit"]["cutoff"]
    ours = _crump_alpha(ps)
    g = 1.0 / (ps * (1.0 - ps))
    for alpha in (theirs, ours):
        gamma = 1.0 / (alpha * (1.0 - alpha))
        assert gamma == pytest.approx(
            2.0 * g[g <= gamma * (1 + 1e-12)].mean(), rel=1e-10
        )
    gamma_ours = 1.0 / (ours * (1.0 - ours))
    obj = [_crump_objective(g, r * (1 + 1e-12)) for r in roots]
    assert _crump_objective(g, gamma_ours * (1 + 1e-12)) == pytest.approx(
        min(obj), rel=1e-12
    )
    assert gamma_ours == pytest.approx(roots[int(np.argmin(obj))], rel=1e-12)
    assert any(abs(1.0 / (theirs * (1.0 - theirs)) - r) <= 1e-12 * r for r in roots)
    kept = set(sp.trimming(df, "t", X).index + 1)
    r_kept = set(ref["trim"]["crump_logit"]["kept"])
    gamma_ours = 1.0 / (ours * (1.0 - ours))
    gamma_r = 1.0 / (theirs * (1.0 - theirs))
    between = set(np.nonzero((g > gamma_ours) & (g <= gamma_r * (1 + 1e-12)))[0] + 1)
    assert kept < r_kept and r_kept - kept == between


def test_fixed_bounds_trimming_matches_propensity(df, ref):
    kept = sp.trimming(df, "t", X, method="sturmer")
    assert (kept.index + 1).tolist() == ref["trim"]["fixed_logit"]["kept"]


# --- weights and effective sample size ---------------------------------------


@pytest.mark.parametrize("estimand", ["ATE", "ATT", "ATC", "ATM", "ATO"])
def test_weights_and_ess(df, ref, ps, estimand):
    t = df["t"].to_numpy()
    w = sp.ps_weights(ps, t, estimand)
    _close(w, ref["weights"][estimand])
    weightit_ess, halfmoon_ess = ref["ess"][estimand]
    assert sp.ess(w) == pytest.approx(weightit_ess, rel=TIGHT)
    assert sp.ess(w) == pytest.approx(halfmoon_ess, rel=TIGHT)
    by = sp.ess(w, by=t)
    for g in (0, 1):
        assert by.loc[g] == pytest.approx(
            ref["ess_by_group"][estimand][str(g)], rel=TIGHT
        )


def test_stabilized_probit_and_truncated_weights(df, ref, ps):
    t = df["t"].to_numpy()
    _close(sp.ps_weights(ps, t, stabilize=True), ref["weights_stabilized"])
    _close(sp.ps_weights(np.asarray(ref["ps"]["probit"]), t), ref["weights_probit_ATE"])
    tr = ref["weights_truncated"]
    _close(sp.ps_weights(ps, t, truncate=(0.05, 0.9)), tr["ps_ATE"])
    _close(sp.ps_weights(ps, t, "ATT", truncate=(0.05, 0.9)), tr["ps_ATT"])
    q = dict(truncate=(0.02, 0.98), truncate_scale="quantile")
    _close(sp.ps_weights(ps, t, **q), tr["pctl_ATE"])
    _close(sp.ps_weights(ps, t, "ATO", **q), tr["pctl_ATO"])


def test_continuous_weights_rebuild_propensity(df, ref):
    c = ref["continuous"]
    mu, x = np.asarray(c["mu"]), df["dose"].to_numpy()
    # ours: the density ratio at the sigma it is given
    ours = sp.ps_weights(mu, x, exposure="continuous", sigma=c["sigma"])
    _close(ours, 1.0 / stats.norm.pdf(x, mu, c["sigma"]), rel=1e-12)
    # propensity: standard normal densities on the maximum-likelihood scales
    s_ml = np.sqrt(np.mean((x - mu) ** 2))
    unstab = sp.ps_weights(mu, x, exposure="continuous", sigma=s_ml) / s_ml
    _close(unstab, c["w"])
    z_a = (x - x.mean()) / x.std(ddof=0)
    _close(unstab * stats.norm.pdf(z_a), c["w_stab"])


def test_continuous_weights_rebuild_weightit(df, ref):
    """WeightIt's glm weights are our stabilised density ratio, taken at
    the ML scales, times sd_ML(A) / sigma_ML."""
    c = ref["continuous"]
    mu, x = np.asarray(c["mu"]), df["dose"].to_numpy()
    s_ml = np.sqrt(np.mean((x - mu) ** 2))
    a_ml, a_1 = x.std(ddof=0), x.std(ddof=1)
    ours = sp.ps_weights(mu, x, exposure="continuous", sigma=s_ml, stabilize=True)
    _close(
        ours,
        stats.norm.pdf(x, x.mean(), a_1) / stats.norm.pdf(x, mu, s_ml),
        rel=1e-12,
    )
    # numerator at the ML scale, then the dropped Jacobian sd_ML(A) / sigma
    at_ml = ours * stats.norm.pdf(x, x.mean(), a_ml) / stats.norm.pdf(x, x.mean(), a_1)
    _close(at_ml * a_ml / s_ml, c["weightit_glm"])
    _close(c["weightit_glm"], c["w_stab"], rel=1e-13)


# --- energy distance ------------------------------------------------------------


@pytest.mark.parametrize("key", ["raw", "ATE", "ATT", "ATO"])
def test_energy_distance_matches_halfmoon(df, ref, weights, key):
    w = None if key == "raw" else weights[key]
    got = sp.energy_distance(df, "t", X, weights=w)
    assert got == pytest.approx(ref["energy"][key], rel=TIGHT)


def test_energy_distance_on_a_subset_and_inside_balance_diagnostics(
    df, ref, weights, ps
):
    got = sp.energy_distance(df, "t", ["x1", "x2"], weights=weights["ATE"])
    assert got == pytest.approx(ref["energy"]["continuous_only_ATE"], rel=TIGHT)
    s = sp.balance_diagnostics(df, "t", X, weights=weights["ATE"], ps=ps).summary_stats
    assert s["energy_distance_raw"] == pytest.approx(ref["energy"]["raw"], rel=TIGHT)
    assert s["energy_distance_weighted"] == pytest.approx(
        ref["energy"]["ATE"], rel=TIGHT
    )


# --- balance tables ---------------------------------------------------------------


@pytest.mark.parametrize("estimand", ["ATE", "ATT", "ATO"])
@pytest.mark.parametrize("sd_denom", ["unweighted", "weighted"])
def test_balance_diagnostics_matches_cobalt(df, ref, weights, ps, estimand, sd_denom):
    B = ref["balance"][estimand]
    key = {"unweighted": "smd_unweighted_denom", "weighted": "smd_weighted_denom"}
    tab = sp.balance_diagnostics(
        df, "t", X, weights=weights[estimand], ps=ps, sd_denom=sd_denom
    ).table
    _close(tab["smd_raw"], B["smd_raw"])
    _close(tab["smd_weighted"], B[key[sd_denom]])
    _close(tab["variance_ratio_weighted"], B["vr"])
    _close(tab["ks_stat_weighted"], B["ks"])
    _close(tab["weighted_mean_treat"], B["mean_w_t"])
    _close(tab["weighted_mean_control"], B["mean_w_c"])
    s = sp.balance_diagnostics(
        df, "t", X, weights=weights[estimand], ps=ps
    ).summary_stats
    assert s["effective_sample_size_treated"] == pytest.approx(
        ref["ess_by_group"][estimand]["1"], rel=TIGHT
    )
    assert s["effective_sample_size_control"] == pytest.approx(
        ref["ess_by_group"][estimand]["0"], rel=TIGHT
    )


@pytest.mark.parametrize("estimand", ["ATE", "ATT", "ATO"])
def test_ps_balance_matches_cobalt(df, ref, weights, estimand):
    B = ref["balance"][estimand]
    for sd_denom, key in (
        ("weighted", "smd_weighted_denom"),
        ("unweighted", "smd_unweighted_denom"),
    ):
        tab = sp.ps_balance(
            df, "t", X, weights=weights[estimand], sd_denom=sd_denom
        ).table
        _close(tab["smd_raw"], B["smd_raw"])
        _close(tab["smd_weighted"], B[key])
        _close(tab["variance_ratio"], B["vr"])
        _close(tab["ks_stat"], B["ks"])


def test_ps_balance_default_weights_are_propensity_ate_weights(df, ref):
    B = ref["balance"]["ATE"]
    tab = sp.ps_balance(df, "t", X).table
    _close(tab["smd_weighted"], B["smd_weighted_denom"])
    _close(tab["variance_ratio"], B["vr"])
    _close(tab["ks_stat"], B["ks"])


@pytest.mark.parametrize("estimand", ["ATE", "ATT", "ATO"])
def test_baltab_rows_and_the_binary_convention(df, ref, weights, ps, estimand):
    """cobalt's bal.tab: continuous rows equal ours; its 0/1 row is ours
    with the p(1 - p) variance in place of the sample variance."""
    bt = pd.DataFrame(ref["balance"][estimand]["baltab"]).set_index("var")
    tab = sp.balance_diagnostics(
        df, "t", X, weights=weights[estimand], ps=ps, sd_denom="unweighted"
    ).table
    cont = ["x1", "x2", "x3"]
    _close(tab.loc[cont, "smd_raw"], bt.loc[cont, "Diff.Un"])
    _close(tab.loc[cont, "smd_weighted"], bt.loc[cont, "Diff.Adj"])
    _close(tab.loc[cont, "variance_ratio_weighted"], bt.loc[cont, "V.Ratio.Adj"])
    _close(tab.loc[cont, "ks_stat_weighted"], bt.loc[cont, "KS.Adj"])
    t = df["t"].to_numpy() == 1
    p1, p0 = df["b"][t].mean(), df["b"][~t].mean()
    sd_binary = np.sqrt((p1 * (1 - p1) + p0 * (1 - p0)) / 2)
    row = tab.loc["b"]
    _close(
        (row["mean_treat"] - row["mean_control"]) / sd_binary, bt.loc["b", "Diff.Un"]
    )
    _close(
        (row["weighted_mean_treat"] - row["weighted_mean_control"]) / sd_binary,
        bt.loc["b", "Diff.Adj"],
    )
    _close(row["ks_stat_weighted"], bt.loc["b", "KS.Adj"])


_BALTAB_COLS = {
    "Diff.Un": "diff_un",
    "Diff.Adj": "diff_adj",
    "V.Ratio.Un": "vr_un",
    "V.Ratio.Adj": "vr_adj",
    "KS.Un": "ks_un",
    "KS.Adj": "ks_adj",
}


@pytest.mark.parametrize(
    "key, kw",
    [
        ("sd_all", dict(s_d_denom="all")),
        ("sd_treated", dict(s_d_denom="treated", focal=1)),
        ("sd_control", dict(s_d_denom="control")),
        ("cont_raw", dict(continuous="raw")),
        ("bin_std_all", dict(binary="std", s_d_denom="all")),
    ],
)
def test_bal_tab_options_match_cobalt(df, ref, key, kw):
    rec = ref["bal_tab"]
    tab = sp.bal_tab(
        df, "t", X, weights=np.asarray(rec["w"]), estimand="ATE", **kw
    ).table
    want = pd.DataFrame(rec[key]).set_index("var")
    n = 0
    for col, mine in _BALTAB_COLS.items():
        w = want[col].astype(float)
        w = w[w.notna()]
        _close(tab.loc[w.index, mine], w)
        n += len(w)
    assert n >= 20


# --- L1 imbalance -------------------------------------------------------------------


@pytest.mark.parametrize(
    "key, kw",
    [
        ("sturges", {}),
        ("k3", dict(n_bins=3)),
        ("k5", dict(n_bins=5)),
        ("k12", dict(n_bins=12)),
    ],
)
def test_l1_imbalance_matches_cem(df, ref, key, kw):
    assert sp.l1_imbalance(df, "t", X, **kw) == pytest.approx(ref["l1"][key], rel=1e-12)


def test_weighted_and_partial_l1_imbalance_match_cem(df, ref, weights):
    L = ref["l1"]
    assert L["k_sturges"] == int(np.ceil(np.log2(len(df)) + 1))
    got = sp.l1_imbalance(df, "t", X, weights=weights["ATE"])
    assert got == pytest.approx(L["sturges_ATE"], rel=1e-12)
    got = sp.l1_imbalance(df, "t", X, weights=weights["ATO"], n_bins=5)
    assert got == pytest.approx(L["k5_ATO"], rel=1e-12)
    assert sp.l1_imbalance(df, "t", ["x1"]) == pytest.approx(L["only_x1"], rel=1e-12)
    got = sp.l1_imbalance(df, "t", ["x1", "x3"], n_bins=4)
    assert got == pytest.approx(L["x1_x3_k4"], rel=1e-12)


def test_reference_versions_are_recorded(ref):
    v = ref["_versions"]
    assert set(v) >= {"R", "propensity", "halfmoon", "WeightIt", "cobalt", "cem"}


# --- implied regression weights ---------------------------------------------------


@pytest.mark.parametrize(
    "key, kw",
    [
        ("uri", dict()),
        ("mri_ate", dict(interactions=True, estimand="ATE")),
        ("mri_att", dict(interactions=True, estimand="ATT")),
        ("mri_atc", dict(interactions=True, estimand="ATC")),
    ],
)
def test_implied_weights_match_lmw(df, ref, key, kw):
    got = sp.implied_weights(df, "t", X, **kw).to_numpy()
    _close(got, ref["lmw"][key])


def test_implied_weights_of_a_smaller_model_match_lmw(df, ref):
    got = sp.implied_weights(df, "t", ["x1", "x2"]).to_numpy()
    _close(got, ref["lmw"]["uri_x1x2"])
