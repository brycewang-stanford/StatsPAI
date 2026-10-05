"""Barrett, D'Agostino McGowan and Gerke, *Causal Inference in R*, on its
own running example.

The book (https://www.r-causal.org) asks one question through chapters 8 to
16: do Extra Magic Morning hours at Disney's Magic Kingdom change the
posted wait for the Seven Dwarfs Mine Train at 9 am? The data are
``touringplans::seven_dwarfs_train_2018`` (354 days). The answer key
``data/barrett_causal_inference_in_r_R.json`` was made by rerunning the
book's analyses with the packages it uses
(``barrett_causal_inference_in_r_reference.R`` next to this file). Here
every number is recomputed with the ``sp.*`` function a user would reach
for.

The data are not redistributed. Run the R script once with
``STATSPAI_BARRETT_DIR`` pointing at an empty folder (it exports the two
tables from the R package), then

    STATSPAI_BARRETT_DIR=/that/folder \\
        pytest tests/external_parity/test_barrett_causal_inference_in_r.py

It is skipped otherwise. The same computations run in CI on simulated data
in ``tests/reference_parity/test_barrett_causal_inference_in_r_parity.py``.
What the pass found is in
``docs/dev/2026-10-05-barrett-causal-inference-in-r-review.md``.

Tolerances. Weights, effective sample sizes, trimming, energy distance,
tipping points: 1e-9 relative. Anything through R's ``lm`` on a column of
seconds since midnight (``park_close``, about 8e4): 1e-7. Delta-method
standard errors from ``marginaleffects`` (numerical derivatives): 1e-6.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import statspai as sp

ROOT = os.environ.get("STATSPAI_BARRETT_DIR")
pytestmark = pytest.mark.skipif(
    not ROOT or not (Path(ROOT) / "seven_dwarfs_9.csv").is_file(),
    reason="set STATSPAI_BARRETT_DIR and run "
    "barrett_causal_inference_in_r_reference.R once",
)

Y = "wait_minutes_posted_avg"
T = "park_extra_magic_morning"
X = ["season_regular", "season_value", "park_close_num", "park_temperature_high"]
EXACT = 1e-9


@pytest.fixture(scope="module")
def R():
    path = Path(__file__).parent / "data" / "barrett_causal_inference_in_r_R.json"
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def d():
    df = pd.read_csv(Path(ROOT) / "seven_dwarfs_9.csv")
    df["over60"] = (df[Y] > 60).astype(float)
    df["emm"] = df[T].astype(int)
    return df


@pytest.fixture(scope="module")
def ps(d, R):
    p = sp.propensity_score(d, T, X).to_numpy()
    assert p == pytest.approx(np.array(R["ps"]), rel=1e-9)
    return p


# --- chapters 8 and 10: weights for five estimands ---------------------------


@pytest.mark.parametrize(
    "estimand,key",
    [("ATE", "ate"), ("ATT", "att"), ("ATU", "atu"), ("ATM", "atm"), ("ATO", "ato")],
)
def test_weights_and_effective_sample_size(d, ps, R, estimand, key):
    w = sp.ps_weights(ps, d[T].to_numpy(), estimand)
    assert w == pytest.approx(np.array(R["weights"][key]), rel=EXACT)
    assert sp.ess(w) == pytest.approx(R["ess"][key], rel=EXACT)


def test_stabilized_and_truncated_weights(d, ps, R):
    t = d[T].to_numpy()
    w = sp.ps_weights(ps, t, stabilize=True)
    assert w == pytest.approx(np.array(R["weights"]["ate_stab"]), rel=EXACT)
    assert w.mean() == pytest.approx(1.0, abs=0.02)
    w = sp.ps_weights(ps, t, truncate=(0.01, 0.99), truncate_scale="quantile")
    assert sp.ess(w) == pytest.approx(R["ess_trunc"], rel=EXACT)


def test_adaptive_trimming_drops_the_same_days(d, R):
    kept = sp.trimming(d, treatment=T, covariates=X)
    dropped = sorted(set(d.index + 1) - set(kept.index + 1))
    assert dropped == R["trim_adaptive"]["trimmed"]
    assert len(kept) == 318


# --- chapter 9: diagnostics ----------------------------------------------------


def test_balance_diagnostics(d, ps, R):
    w = np.array(R["weights"]["ate"])
    bal = sp.balance_diagnostics(d, T, X, weights=w, ps=ps)
    direct = R["bal_direct"]
    temp = bal.table.loc["park_temperature_high"]
    assert temp["variance_ratio_weighted"] == pytest.approx(
        direct["vr_temp_w"], rel=1e-8
    )
    assert temp["ks_stat_weighted"] == pytest.approx(direct["ks_temp_w"], rel=EXACT)
    s = bal.summary_stats
    assert s["energy_distance_raw"] == pytest.approx(R["energy"]["obs"], rel=EXACT)
    assert s["energy_distance_weighted"] == pytest.approx(
        R["energy"]["ate"], rel=EXACT
    )
    assert s["effective_sample_size_treated"] == pytest.approx(
        R["ess_by_group_ate"]["1"], rel=EXACT
    )
    assert s["effective_sample_size_control"] == pytest.approx(
        R["ess_by_group_ate"]["0"], rel=EXACT
    )
    # halfmoon standardizes by the unweighted spread with divisor n; the
    # same mean difference over the n - 1 spread is what is reported here
    unw = sp.balance_diagnostics(d, T, X, weights=w, ps=ps, sd_denom="unweighted")
    x = d["park_temperature_high"].to_numpy()
    t = d[T].to_numpy() == 1
    ratio = np.sqrt(
        (x[t].var(ddof=1) + x[~t].var(ddof=1)) / (x[t].var() + x[~t].var())
    )
    assert abs(
        unw.table.loc["park_temperature_high", "smd_weighted"]
    ) * ratio == pytest.approx(direct["smd_temp_w"], rel=1e-8)


def test_auc_of_the_propensity_model(d, ps, R):
    t = d[T].to_numpy()
    assert sp.auc(t, ps) == pytest.approx(R["auc"]["observed"], rel=EXACT)
    # weighted: near one half once the weights balance the score
    assert sp.auc(t, ps, weights=R["weights"]["ate"]) == pytest.approx(
        R["auc"]["w_ate"], abs=2e-3
    )


def test_implied_regression_weights(d, R):
    w = sp.implied_weights(d, T, X)
    assert w.to_numpy() == pytest.approx(np.array(R["lmw"]["w"]), rel=1e-7)
    assert sp.ess(w) == pytest.approx(R["lmw"]["ess"], rel=1e-7)
    wi = sp.implied_weights(d, T, X, interactions=True)
    assert wi.min() == pytest.approx(R["lmw_int"]["min"], rel=1e-7)
    assert wi.min() < 0  # the interacted model extrapolates for some days


# --- chapter 11: the weighted outcome model ------------------------------------


@pytest.mark.parametrize(
    "estimand,key",
    [("ATE", "ate"), ("ATT", "att"), ("ATM", "atm"), ("ATO", "ato")],
)
def test_ipw_with_the_m_estimation_variance(d, R, estimand, key):
    r = sp.ipw(d, Y, T, X, estimand=estimand, se_method="sandwich")
    ref = R["ipw"][key]["ipw"][0]
    n = len(d)
    assert r.estimate == pytest.approx(ref["estimate"], rel=EXACT)
    assert r.se * np.sqrt(n / (n - 1)) == pytest.approx(ref["std.err"], rel=1e-9)
    # the outcome-model sandwich that ignores the estimated weights is wider
    assert r.se < R["ipw"][key]["se_hc0"]


def test_ipw_effect_on_the_untreated(d, R):
    r = sp.ipw(d, Y, T, X, estimand="ATU", se_method="sandwich")
    assert r.estimate == pytest.approx(R["ipw"]["atu"]["est"], rel=EXACT)


# --- chapters 13 and 14: g-computation -----------------------------------------


def test_g_computation_linear(d, R):
    g = R["gcomp"]
    r = sp.g_computation(d, Y, T, X, n_boot=10, seed=0)
    assert r.estimate == pytest.approx(g["ate"][0], rel=1e-7)
    fit = sp.regress(
        f"{Y} ~ C(emm)*season_regular + C(emm)*season_value"
        " + park_close_num + park_temperature_high",
        data=d,
    )
    for subset, key in (
        (None, "ate_int"),
        ("emm == 1", "att_int"),
        ("emm == 0", "atc_int"),
    ):
        c = sp.contrast(fit, data=d, variable="emm", subset=subset).iloc[0]
        assert c["contrast"] == pytest.approx(g[key][0], rel=1e-7)
        assert c["se"] == pytest.approx(g[key][1], rel=1e-6)


def test_g_computation_binary_outcome(d, R):
    g = R["gcomp_bin"]
    fit = sp.logit(
        "over60 ~ C(emm) + season_regular + season_value"
        " + park_close_num + park_temperature_high",
        data=d,
    )
    for subset, key in ((None, "rd"), ("emm == 1", "att"), ("emm == 0", "atc")):
        c = sp.contrast(fit, data=d, variable="emm", subset=subset).iloc[0]
        assert c["contrast"] == pytest.approx(g[key][0], rel=1e-7)
        assert c["se"] == pytest.approx(g[key][1], rel=1e-6)
    rr = sp.contrast(fit, data=d, variable="emm", effect="ratio").iloc[0]
    assert rr[["contrast", "ci_lower", "ci_upper"]].tolist() == pytest.approx(
        g["rr"], rel=1e-6
    )
    orr = sp.contrast(fit, data=d, variable="emm", effect="odds_ratio").iloc[0]
    assert orr[["contrast", "ci_lower", "ci_upper"]].tolist() == pytest.approx(
        g["or"], rel=1e-6
    )
    p0 = sp.margins_at(fit, data=d, at={"emm": [0]}).iloc[0]
    assert p0["margin"] == pytest.approx(g["p0"][0], rel=1e-7)
    assert p0["se"] == pytest.approx(g["p0"][1], rel=1e-6)


# --- chapter 16: sensitivity -----------------------------------------------------


def test_tipping_points_of_the_chapter(R):
    t = R["tipr"]
    adj = sp.confounder_adjust(
        6.58, confounder_outcome_effect=-2.3, exposure_confounder_effect=-0.17
    )
    assert float(adj["effect_adjusted"].iloc[0]) == pytest.approx(
        t["adjust_coef"][0]["effect_adjusted"], rel=1e-12
    )
    for outcome_effect, key in ((-7, "tip_coef_1"), (-2.3, "tip_coef_2")):
        tip = sp.confounder_tip(6.58, confounder_outcome_effect=outcome_effect)
        assert float(tip["exposure_confounder_effect"].iloc[0]) == pytest.approx(
            t[key][0]["exposure_confounder_effect"], rel=1e-12
        )
    grid = sp.confounder_tip(-10.2, exposure_confounder_effect=[1, 2, 3, 4, 5])
    assert grid["confounder_outcome_effect"].tolist() == pytest.approx(
        [row["confounder_outcome_effect"] for row in t["tip_coef_grid"]]
    )
    assert sp.evalue(1.5)["evalue_estimate"] == pytest.approx(t["e_value"], rel=1e-9)


def test_dag_of_the_chapter(R):
    g = sp.dag("emm -> wait; close -> wait; season -> wait; temp -> wait; temp -> emm")
    every = sorted(sorted(s) for s in g.adjustment_sets("emm", "wait", minimal=False))
    ref = sorted(sorted([s] if isinstance(s, str) else s) for s in R["dag"]["adj_all"])
    assert every == ref
    cls = g.equivalence_class()
    assert cls["n_dags"] == R["dag"]["n_equiv"] == 2
    assert cls["undirected"] == [("emm", "temp")]
    assert len(g.implied_independencies()) == len(R["dag"]["ci"])
