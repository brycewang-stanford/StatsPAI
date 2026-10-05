"""Ness, *Causal AI* (Manning, 2025), on its own data.

The book's code and data are at https://github.com/altdeep/causalML. It
builds causal graphs with pgmpy, tests them, fits their kernels, identifies
effects with y0 and estimates them with DoWhy. Here each step is redone
with the ``sp.*`` function a user would reach for and compared with

* the numbers the book's notebooks print (pgmpy, DoWhy), quoted below with
  the listing they come from, and
* ``data/ness_causal_ai_R.json``, the same steps rerun in R with dagitty,
  bnlearn and pcalg (``ness_causal_ai_reference.R`` next to this file).

The data are not copied into this repository. Clone the book's repository
and run

    STATSPAI_NESS_DIR=/path/to/causalML/datasets \\
        pytest tests/external_parity/test_ness_causal_ai.py

It is skipped otherwise. The same computations run in CI on simulated data
in ``tests/reference_parity/test_ness_causal_ai_parity.py``. What the pass
found is in ``docs/dev/2026-10-06-ness-causal-ai-review.md``.

Tolerances. Counts, chi-square statistics and least squares: 1e-9
relative. Numbers the book prints to four decimals: 5e-5 absolute.
"""

from __future__ import annotations

import json
import os
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import statspai as sp

ROOT = os.environ.get("STATSPAI_NESS_DIR")
pytestmark = pytest.mark.skipif(
    not ROOT or not (Path(ROOT) / "transportation_survey.csv").is_file(),
    reason="set STATSPAI_NESS_DIR to the datasets folder of altdeep/causalML",
)

EXACT = 1e-9
TRANSPORT = "A -> E; S -> E; E -> O; E -> R; O -> T; R -> T"
Y, D = "In-game Purchases", "Side-quest Engagement"
X = ["Guild Membership", "Player Skill Level", "Time Spent Playing"]
GAMING_DOT = """digraph {
    "Prior Experience" -> "Player Skill Level";
    "Prior Experience" -> "Time Spent Playing";
    "Time Spent Playing" -> "Player Skill Level";
    "Guild Membership" -> "Side-quest Engagement";
    "Guild Membership" -> "In-game Purchases";
    "Player Skill Level" -> "Side-quest Engagement";
    "Player Skill Level" -> "In-game Purchases";
    "Time Spent Playing" -> "Side-quest Engagement";
    "Time Spent Playing" -> "In-game Purchases";
    "Side-quest Group Assignment" -> "Side-quest Engagement";
    "Customization Level" -> "Side-quest Engagement";
    "Side-quest Engagement" -> "Won Items";
    "Won Items" -> "In-game Purchases";
    "Won Items" -> "Total Inventory";
    "In-game Purchases" -> "Total Inventory";
}"""


@pytest.fixture(scope="module")
def R():
    path = Path(__file__).parent / "data" / "ness_causal_ai_R.json"
    return json.loads(path.read_text(encoding="utf-8"))


def _read(name):
    return pd.read_csv(Path(ROOT) / name)


@pytest.fixture(scope="module")
def transport():
    return _read("transportation_survey.csv")


@pytest.fixture(scope="module")
def gaming():
    return _read("online_game_example_do_why.csv")


# --- chapter 3: kernels and queries -----------------------------------------


def test_listing_3_4_kernels_match_bnlearn(R, transport):
    cpt = sp.bayes_net(TRANSPORT, transport).cpt("T")
    for cell in R["transport_cpt_T"]:
        assert cpt.loc[(cell["O"], cell["R"]), cell["T"]] == pytest.approx(
            cell["Freq"], rel=EXACT
        )


def test_listing_3_5_dirichlet_kernel_is_the_books(transport):
    # pgmpy BayesianEstimator(prior_type="dirichlet", pseudo_counts=1)
    car = sp.bayes_net(TRANSPORT, transport, prior=1).cpt("T")["car"]
    assert car[("emp", "big")] == pytest.approx(0.7007299270072993, rel=EXACT)
    assert car[("emp", "small")] == pytest.approx(0.5176470588235295, rel=EXACT)
    assert car[("self", "big")] == pytest.approx(0.4166666666666667, rel=EXACT)
    assert car[("self", "small")] == pytest.approx(0.5, rel=EXACT)


def test_listing_3_7_variable_elimination_is_the_books(transport):
    # The notebook runs 3.7 on the model as refitted in 3.5.
    net = sp.bayes_net(TRANSPORT, transport, prior=1)
    train = net.query("E", evidence={"T": "train"}).set_index("E")["prob"]
    car = net.query("E", evidence={"T": "car"}).set_index("E")["prob"]
    assert train["high"] == pytest.approx(0.6162, abs=5e-5)
    assert train["uni"] == pytest.approx(0.3838, abs=5e-5)
    assert car["high"] == pytest.approx(0.5586, abs=5e-5)
    assert car["uni"] == pytest.approx(0.4414, abs=5e-5)


# --- chapter 4: testing the graph --------------------------------------------


def test_listing_4_1_d_separation():
    g = sp.dag("I -> U; I -> M; M -> U; J -> V; J -> M; M -> V")
    assert not g.d_separated("U", "V", {"M"})
    assert g.d_separated("U", "V", {"M", "I", "J"})
    assert g.d_separated("U", "V", {"M", "I"})
    assert g.d_separated("U", "V", {"M", "J"})


def test_listing_4_4_chi_square_on_thirty_rows_is_pgmpys(transport):
    # pgmpy prints (1.1611111111111112, 0.5595873983053805, 2)
    out = sp.dag(TRANSPORT).test_implications(transport.iloc[:30])
    row = out[(out.x == "E") & (out.y == "T") & (out.given == "O, R")].iloc[0]
    assert row["statistic"] == pytest.approx(1.1611111111111112, rel=EXACT)
    assert row["p_value"] == pytest.approx(0.5595873983053805, rel=EXACT)
    assert row["df"] == 2


def test_listing_4_6_every_implication_matches_dagitty_and_bnlearn(R, transport):
    chi = sp.dag(TRANSPORT).test_implications(transport).set_index(["x", "y", "given"])
    g2 = sp.dag(TRANSPORT).test_implications(transport, test="g-test").set_index(["x", "y", "given"])
    assert len(chi) == len(R["transport_chisq"]) == 11
    for row in R["transport_chisq"]:
        key = (row["x"], row["y"], ", ".join(row["z"]))
        assert chi.loc[key, "statistic"] == pytest.approx(row["x2"], rel=EXACT)
        assert chi.loc[key, "df"] == row["df"]
        assert chi.loc[key, "p_value"] == pytest.approx(row["p"], rel=EXACT)
        assert g2.loc[key, "statistic"] == pytest.approx(row["g2"], rel=EXACT)
        assert g2.loc[key, "p_value"] == pytest.approx(row["g2_p"], rel=EXACT)
    # The data were simulated from this graph: nothing should be rejected.
    assert (chi["p_holm"] > 0.05).all()


def test_structure_learning_matches_pcalg(R, transport):
    for data, key in [(_read("structure_learning_test.csv"), "pc_structure_learning"), (transport, "pc_transport")]:
        out = sp.pc_algorithm(data, ci_test="chi-square")
        assert sorted(out["edges"]) == sorted(tuple(e) for e in R[key]["directed"])
        assert {tuple(sorted(e)) for e in out["undirected_edges"]} == {
            tuple(sorted(e)) for e in R[key]["undirected"]
        }
        assert out["orientation_conflicts"] == []


# --- chapter 7: seeing, doing and the experiment -----------------------------


def test_adjusting_for_guild_membership_recovers_the_experiment(R):
    obs = _read("sidequests_and_purchases_full_obs.csv")
    exp = _read("sidequests_and_purchases_exp.csv")
    means = exp.groupby(D)[Y].mean()
    truth = means["high"] - means["low"]
    assert truth == pytest.approx(R["experiment"]["diff"], rel=EXACT)

    df = pd.DataFrame(
        {
            "I": obs[Y],
            "E": (obs[D] == "high").astype(int),
            "G": (obs["Guild Membership"] == "member").astype(int),
        }
    )
    naive = df.groupby("E")["I"].mean()
    assert naive[1] - naive[0] > 30  # the association has the wrong sign
    g = sp.dag("G -> E; G -> I; E -> I")
    assert g.adjustment_sets("E", "I") == [{"G"}]
    aipw = sp.aipw(df, y="I", treat="E", covariates=["G"])
    ipw = sp.ipw(df, y="I", treat="E", covariates=["G"], se_method="sandwich")
    gcomp = sp.g_computation(
        df, y="I", treat="E", covariates=["G"], by_arm=True, n_boot=20, seed=0
    )
    # Saturated in a binary confounder: all three are the standardised difference.
    assert gcomp.estimate == pytest.approx(R["standardised"], rel=1e-9)
    assert ipw.estimate == pytest.approx(R["standardised"], rel=1e-6)
    assert aipw.estimate == pytest.approx(R["standardised"], abs=0.05)
    # and that is the experimental effect, within sampling error
    assert abs(aipw.estimate - truth) < 2 * np.hypot(aipw.se, R["experiment"]["se"])


def test_labelled_treatment_is_explained():
    obs = _read("sidequests_and_purchases_full_obs.csv")
    with pytest.raises(sp.MethodIncompatibility, match="holds labels"):
        sp.aipw(obs, y=Y, treat=D, covariates=["Guild Membership"])


# --- chapters 10 and 11: identification and estimation -----------------------


def test_listing_11_4_the_three_strategies(gaming):
    g = sp.dag(GAMING_DOT, latent=["Prior Experience"])
    assert sorted(g.observed_nodes) == sorted(gaming.columns)
    assert g.adjustment_sets(D, Y) == [set(X)]  # DoWhy's backdoor set
    assert g.frontdoor_sets(D, Y) == [{"Won Items"}]  # its front-door set
    instruments = {
        v for v in g.observed_nodes - {D, Y} if "instrument" in g.classify_variable(v, D, Y)
    }
    assert instruments == {"Side-quest Group Assignment", "Customization Level"}
    rec = g.recommend_estimator(D, Y)
    assert "sp.front_door" in " ".join(rec.alternatives) and "sp.iv(" in " ".join(rec.alternatives)


def test_listing_11_5_linear_regression_is_dowhys(gaming):
    # DoWhy backdoor.linear_regression: 178.0861711575792, [168.68114922, 187.4911931]
    g = sp.dag(GAMING_DOT, latent=["Prior Experience"])
    call = g.recommend_estimator(D, Y).sp_call.split("  #")[0]
    fit = eval(call, {"sp": sp, "df": gaming})  # noqa: S307 - our own string
    term = f'Q("{D}")'
    assert fit.params[term] == pytest.approx(178.0861711575792, rel=EXACT)
    low, high = fit.conf_int().loc[term]
    assert low == pytest.approx(168.68114922, rel=1e-9)
    assert high == pytest.approx(187.4911931, rel=1e-9)


def test_listing_11_12_front_door(gaming):
    # Nobody with low engagement won items: E[Y | E = 0, W = 1] has no data.
    assert ((gaming[D] == 0) & (gaming["Won Items"] == 1)).sum() == 0
    with pytest.raises(sp.IdentificationFailure, match="does not vary"):
        sp.front_door(gaming, y=Y, treat=D, mediator="Won Items", n_boot=10, seed=0)
    # DoWhy frontdoor.two_stage_regression prints 170.20560581290403: the
    # product of two regression coefficients, which is the additive model.
    res = sp.front_door(gaming, y=Y, treat=D, mediator="Won Items", n_boot=50, seed=0, outcome_model="additive")
    assert res.estimate == pytest.approx(170.20560581290403, rel=EXACT)


def test_listing_11_9_weights_rest_on_a_tenth_of_the_controls(gaming):
    with pytest.warns(sp.AssumptionWarning, match="effective sample"):
        res = sp.ipw(gaming, y=Y, treat=D, covariates=X, se_method="sandwich")
    assert res.model_info["ess_control"] / res.model_info["n_control"] < 0.15
    # The book remarks that this estimator "differs so dramatically"; the
    # doubly robust one agrees with the regression.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        aipw = sp.aipw(gaming, y=Y, treat=D, covariates=X)
    assert aipw.estimate == pytest.approx(178.09, abs=3.0)


def test_listings_11_15_to_11_18_refutations(gaming):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for method in ("data_subset", "random_common_cause", "placebo_treatment", "dummy_outcome"):
            res = sp.refute(sp.aipw, gaming, y=Y, treat=D, covariates=X, method=method, n_simulations=40, seed=0)
            assert not res.refuted, method
        # Listing 11.18's dummy outcome is a function of two confounders.
        fn = lambda f: 100.0 * f["Guild Membership"] + 50.0 * f["Player Skill Level"] + 50.0  # noqa: E731
        kept = sp.refute(sp.aipw, gaming, y=Y, treat=D, covariates=X, method="dummy_outcome", n_simulations=40, seed=0, outcome_function=fn)
        naive = sp.refute(
            lambda f, y, treat, **k: f.loc[f[treat] == 1, y].mean() - f.loc[f[treat] == 0, y].mean(),
            gaming, y=Y, treat=D, method="dummy_outcome", n_simulations=40, seed=0, outcome_function=fn,
        )
    assert not kept.refuted and naive.refuted


def test_front_door_estimand_flags_the_same_empty_cell(gaming):
    df = gaming.rename(columns={D: "E", "Won Items": "W"})
    df = df.assign(I=df[Y] > df[Y].median())
    res = sp.identify(sp.dag("E -> W -> I; U -> E; U -> I", latent=["U"]), "E", "I")
    with pytest.warns(sp.AssumptionWarning, match="positivity"):
        est = res.estimate(df)
    assert est.loc[est.E == 1, "prob"].isna().all()
    assert est.loc[est.E == 0, "prob"].notna().all()
