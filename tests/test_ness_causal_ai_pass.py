"""What working through Ness, *Causal AI* (Manning, 2025) changed.

Each test pins one behaviour the pass fixed or added, on data built here.
The cross-language checks are in
``tests/reference_parity/test_ness_causal_ai_parity.py``; the book's own
data and printed numbers are in
``tests/external_parity/test_ness_causal_ai.py``. The account of the pass
is ``docs/dev/2026-10-06-ness-causal-ai-review.md``.
"""

from __future__ import annotations

import itertools
import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import ColumnNotFound

# --------------------------------------------------------------------------- #
#  Graph specification
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "spec,edges",
    [
        ("X <- Z; X -> Y", [("X", "Y"), ("Z", "X")]),
        ("A -> B <- C", [("A", "B"), ("C", "B")]),
        ("X <- Z -> Y", [("Z", "X"), ("Z", "Y")]),
        ("{A B} -> C", [("A", "C"), ("B", "C")]),
        ('"a;b" -> c; c -> "d e"', [("a;b", "c"), ("c", "d e")]),
        ("dag { A -> B; B -> C }", [("A", "B"), ("B", "C")]),
        (
            'digraph G { rankdir=LR; node [shape=box]; "A b" -> "C d" [label="x"]; }',
            [("A b", "C d")],
        ),
    ],
)
def test_arrows_quotes_and_wrappers_are_read(spec, edges):
    assert sorted(sp.dag(spec).edges) == sorted(edges)


def test_a_reversed_arrow_used_to_be_dropped():
    # 'X <- Z' has no '->', so the confounder vanished and the empty set
    # came back as a valid adjustment set.
    g = sp.dag("X <- Z; Z -> Y; X -> Y")
    assert g.adjustment_sets("X", "Y") == [{"Z"}]


def test_a_name_alone_declares_a_node():
    g = sp.dag("X -> Y; Z")
    assert g.nodes == {"X", "Y", "Z"} and g.edges == [("X", "Y")]


def test_latent_declarations_survive_the_wrapper():
    g = sp.dag("dag { U [latent]; U -> X; U -> Y; X -> Y }")
    assert g.latent_nodes == {"U"} and g.adjustment_sets("X", "Y") == []


@pytest.mark.parametrize(
    "spec", ["X => Y", "X -- Y; X -> Z", "X -> ; Y", "X -> Y; Y -> X"]
)
def test_what_cannot_be_read_is_refused(spec):
    with pytest.raises(sp.MethodIncompatibility):
        sp.dag(spec)


# --------------------------------------------------------------------------- #
#  Paths and roles
# --------------------------------------------------------------------------- #

COLLIDER_AFTER_X = "Z -> X; Z -> Y; X -> M -> Y; M -> C; Y -> C"


def test_backdoor_paths_enter_the_exposure():
    g = sp.dag(COLLIDER_AFTER_X)
    assert g.backdoor_paths("X", "Y") == [["X", "Z", "Y"]]
    assert g.noncausal_paths("X", "Y") == [["X", "Z", "Y"], ["X", "M", "C", "Y"]]


def test_path_status_names_three_kinds_and_draws_the_arrows():
    status = {
        s["arrows"]: (s["type"], s["open"])
        for s in sp.dag(COLLIDER_AFTER_X).path_status("X", "Y")
    }
    assert status == {
        "X -> M -> Y": ("causal", True),
        "X <- Z -> Y": ("backdoor", True),
        "X -> M -> C <- Y": ("noncausal", False),
    }


def test_path_order_is_deterministic():
    g = sp.dag(COLLIDER_AFTER_X)
    assert [s["path"] for s in g.path_status("X", "Y")] == [
        s["path"] for s in sp.dag(COLLIDER_AFTER_X).path_status("X", "Y")
    ]


def test_a_mediator_is_not_called_a_confounder():
    g = sp.dag(COLLIDER_AFTER_X)
    assert "confounder" not in g.classify_variable("M", "X", "Y")
    assert "mediator" in g.classify_variable("M", "X", "Y")
    assert "confounder" in g.classify_variable("Z", "X", "Y")
    assert "collider" in g.classify_variable("C", "X", "Y")


def test_m_bias_parents_are_not_confounders():
    # X <- A -> C <- B -> Y is closed at C: nothing to adjust for.
    g = sp.dag("A -> X; A -> C; B -> C; B -> Y; X -> Y")
    assert "confounder" not in g.classify_variable("A", "X", "Y")
    assert "collider" in g.classify_variable("C", "X", "Y")
    assert g.adjustment_sets("X", "Y") == [set()]


def test_bad_controls_shows_the_path_it_would_open():
    bad = sp.dag(COLLIDER_AFTER_X).bad_controls("X", "Y")
    assert any("X -> M -> C <- Y" in r for r in bad["C"])
    assert "X <- Z -> Y" in sp.dag(COLLIDER_AFTER_X).summary("X", "Y")


# --------------------------------------------------------------------------- #
#  Adjustment sets beyond six variables
# --------------------------------------------------------------------------- #


def _confounders(k):
    return "X -> Y; " + "; ".join(f"Z{i} -> X; Z{i} -> Y" for i in range(k))


@pytest.mark.parametrize("k", [6, 7, 8, 12])
def test_many_confounders_still_have_an_adjustment_set(k):
    # The search stopped at six variables and reported that none existed.
    g = sp.dag(_confounders(k))
    assert g.adjustment_sets("X", "Y") == [{f"Z{i}" for i in range(k)}]
    assert g.recommend_estimator("X", "Y").estimator == "regress"


def test_the_large_set_is_pruned_of_what_it_does_not_need():
    # Eight confounders plus an instrument and a cause of the outcome only.
    g = sp.dag(_confounders(8) + "; W -> X; V -> Y")
    assert g.adjustment_sets("X", "Y") == [{f"Z{i}" for i in range(8)}]
    every = g.adjustment_sets("X", "Y", minimal=False)
    assert {f"Z{i}" for i in range(8)} | {"W", "V"} in every


def test_no_set_is_still_no_set():
    assert sp.dag(_confounders(8) + "; X <-> Y").adjustment_sets("X", "Y") == []


# --------------------------------------------------------------------------- #
#  Recommender
# --------------------------------------------------------------------------- #

GAMING = """
    "Prior Experience" -> "Player Skill Level"; "Prior Experience" -> "Time Spent Playing"
    "Time Spent Playing" -> "Player Skill Level"
    "Guild Membership" -> "Side-quest Engagement"; "Guild Membership" -> "In-game Purchases"
    "Player Skill Level" -> "Side-quest Engagement"; "Player Skill Level" -> "In-game Purchases"
    "Time Spent Playing" -> "Side-quest Engagement"; "Time Spent Playing" -> "In-game Purchases"
    "Side-quest Group Assignment" -> "Side-quest Engagement"
    "Customization Level" -> "Side-quest Engagement"
    "Side-quest Engagement" -> "Won Items"; "Won Items" -> "In-game Purchases"
    "Won Items" -> "Total Inventory"; "In-game Purchases" -> "Total Inventory"
"""


def test_recommender_lists_every_strategy_the_graph_licenses():
    g = sp.dag(GAMING, latent=["Prior Experience"])
    rec = g.recommend_estimator("Side-quest Engagement", "In-game Purchases")
    assert rec.adjustment_set == {
        "Guild Membership",
        "Player Skill Level",
        "Time Spent Playing",
    }
    text = " ".join(rec.alternatives)
    assert "sp.front_door" in text and "Won Items" in text
    assert "sp.iv(" in text and "instrument" in text


def test_recommended_call_runs_on_names_with_spaces():
    rng = np.random.default_rng(0)
    n = 300
    z = rng.normal(size=n)
    d = (z + rng.normal(size=n) > 0).astype(float)
    df = pd.DataFrame({"my y": 2 * d + z + rng.normal(size=n), "the d": d, "z-1": z})
    rec = sp.dag(
        '"z-1" -> "the d"; "z-1" -> "my y"; "the d" -> "my y"'
    ).recommend_estimator("the d", "my y")
    call = rec.sp_call.split("  #")[0]
    fit = eval(call, {"sp": sp, "df": df})  # noqa: S307 - our own string
    assert fit.params['Q("the d")'] == pytest.approx(2.0, abs=0.4)


# --------------------------------------------------------------------------- #
#  Identification
# --------------------------------------------------------------------------- #


def test_a_u_prefix_does_not_make_a_node_latent():
    g = sp.dag("U_rate -> X; U_rate -> Y; X -> Y")
    assert g.adjustment_sets("X", "Y") == [{"U_rate"}]
    res = sp.identify(g, "X", "Y")
    assert (
        res.identifiable
        and res.estimand == "sum_{U_rate} [P(U_rate) * P(Y | U_rate, X)]"
    )


def test_estimand_conditions_only_on_what_matters():
    res = sp.identify(
        sp.dag(GAMING, latent=["Prior Experience"]),
        "Side-quest Engagement",
        "In-game Purchases",
    )
    assert "P(Won Items | Side-quest Engagement)" in res.estimand
    assert "Customization Level" not in res.estimand
    assert "Side-quest Group Assignment" not in res.estimand


def _population(cells):
    """A data frame whose frequencies are exactly the given probabilities."""
    rows = []
    for values, prob in cells:
        count = prob * 20000
        assert abs(count - round(count)) < 1e-6
        rows.extend([values] * int(round(count)))
    return rows


@pytest.fixture(scope="module")
def front_door_population():
    # U -> X, U -> Y unobserved; X -> M -> Y. Effect of X on P(Y) is
    # 0.5 * (0.9 - 0.2) = 0.35.
    cells = []
    for u, x, m, y in itertools.product([0, 1], repeat=4):
        p = 0.5
        px = 0.8 if u else 0.2
        p *= px if x else 1 - px
        pm = 0.9 if x else 0.2
        p *= pm if m else 1 - pm
        py = 0.1 + 0.5 * m + 0.3 * u
        p *= py if y else 1 - py
        cells.append(((u, x, m, y), p))
    return pd.DataFrame(_population(cells), columns=["U", "X", "M", "Y"])


def test_front_door_estimand_recovers_the_effect_without_the_confounder(
    front_door_population,
):
    df = front_door_population.drop(columns="U")
    g = sp.dag("X -> M -> Y; U -> X; U -> Y", latent=["U"])
    est = sp.identify(g, "X", "Y").estimate(df).set_index(["X", "Y"])["prob"]
    assert est[(1, 1)] - est[(0, 1)] == pytest.approx(0.35, abs=1e-12)
    naive = df.groupby("X")["Y"].mean()
    assert naive[1] - naive[0] > 0.5  # the association is far off


def test_backdoor_estimand_equals_the_network_under_do(front_door_population):
    df = front_door_population
    g = sp.dag("U -> X; U -> Y; X -> M -> Y")
    est = sp.identify(g, "X", "Y").estimate(df).set_index(["X", "Y"])["prob"]
    net = sp.bayes_net(g, df)
    for x in (0, 1):
        assert est[(x, 1)] == pytest.approx(net.prob({"Y": 1}, do={"X": x}), abs=1e-12)
    assert est[(1, 1)] - est[(0, 1)] == pytest.approx(0.35, abs=1e-12)


def test_estimate_marks_what_the_data_cannot_determine():
    # Nobody with X = 0 has M = 1, so P(Y | X = 0, M = 1) does not exist.
    df = pd.DataFrame(
        [(0, 0, 0)] * 30
        + [(0, 0, 1)] * 20
        + [(1, 0, 0)] * 10
        + [(1, 0, 1)] * 5
        + [(1, 1, 0)] * 15
        + [(1, 1, 1)] * 20,
        columns=["X", "M", "Y"],
    )
    res = sp.identify(sp.dag("X -> M -> Y; X <-> Y"), "X", "Y")
    with pytest.warns(sp.AssumptionWarning, match="positivity"):
        est = res.estimate(df)
    assert est.loc[est.X == 1, "prob"].isna().all()
    assert est.loc[est.X == 0, "prob"].notna().all()


def test_estimate_bootstrap_and_refusals(front_door_population):
    df = front_door_population.drop(columns="U")
    res = sp.identify(sp.dag("X -> M -> Y; X <-> Y"), "X", "Y")
    est = res.estimate(df, n_boot=30, seed=1)
    assert {"se", "ci_lower", "ci_upper"} <= set(est.columns)
    assert ((est.ci_lower <= est.prob) & (est.prob <= est.ci_upper)).all()
    with pytest.raises(ColumnNotFound):
        res.estimate(df.drop(columns="M"))
    with pytest.raises(sp.MethodIncompatibility, match="distinct values"):
        res.estimate(df.assign(M=np.arange(len(df))))
    with pytest.raises(sp.IdentificationFailure):
        sp.identify(sp.dag("X -> Y; X <-> Y"), "X", "Y").estimate(df)


# --------------------------------------------------------------------------- #
#  Testable implications on categorical data
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def chain():
    rng = np.random.default_rng(3)
    z = rng.integers(0, 3, 900)
    x = (z + rng.integers(0, 2, 900)) % 3
    y = (x + rng.integers(0, 2, 900)) % 3
    lab = np.array(["a", "b", "c"])
    return pd.DataFrame({"Z": lab[z], "X": lab[x], "Y": lab[y]})


def test_labels_are_tested_with_chi_square(chain):
    out = sp.dag("Z -> X -> Y").test_implications(chain)
    assert list(out.columns) == [
        "x",
        "y",
        "given",
        "statistic",
        "df",
        "cramers_v",
        "p_value",
        "p_holm",
    ]
    assert out.loc[0, "p_value"] > 0.01
    wrong = sp.dag("Z -> X; Z -> Y").test_implications(chain)  # says X _||_ Y | Z
    assert wrong.loc[0, "p_value"] < 1e-6 and wrong.loc[0, "cramers_v"] > 0.3


def test_chi_square_is_the_sum_over_strata(chain):
    from scipy import stats

    out = sp.dag("Z -> X -> Y").test_implications(chain).iloc[0]
    stat, dof = 0.0, 0
    for _, sub in chain.groupby("X"):
        res = stats.chi2_contingency(pd.crosstab(sub["Y"], sub["Z"]), correction=False)
        stat += res[0]
        dof += res[2]
    assert out["statistic"] == pytest.approx(stat, rel=1e-12) and out["df"] == dof
    g = sp.dag("Z -> X -> Y").test_implications(chain, test="g-test").iloc[0]
    lr = sum(
        stats.chi2_contingency(
            pd.crosstab(s["Y"], s["Z"]), correction=False, lambda_="log-likelihood"
        )[0]
        for _, s in chain.groupby("X")
    )
    assert g["statistic"] == pytest.approx(lr, rel=1e-12)


def test_tests_refuse_the_wrong_kind_of_column(chain):
    g = sp.dag("Z -> X -> Y")
    with pytest.raises(sp.MethodIncompatibility, match="numeric"):
        g.test_implications(chain, test="fisher-z")
    numeric = pd.DataFrame(
        np.random.default_rng(0).normal(size=(200, 3)), columns=list("ZXY")
    )
    with pytest.raises(sp.MethodIncompatibility, match="distinct values"):
        g.test_implications(numeric, test="chi-square")
    with pytest.raises(sp.MethodIncompatibility):
        g.test_implications(chain, test="kendall")
    assert "partial_corr" in g.test_implications(numeric).columns


def test_an_untestable_implication_has_no_p_value():
    # X copies Z, so Z is constant within every stratum of X.
    df = pd.DataFrame(
        {"Z": list("ab") * 20, "X": list("ab") * 20, "Y": list("uvvu") * 10}
    )
    out = sp.dag("Z -> X -> Y").test_implications(df)
    assert (
        out.loc[0, "df"] == 0
        and np.isnan(out.loc[0, "p_value"])
        and np.isnan(out.loc[0, "p_holm"])
    )


# --------------------------------------------------------------------------- #
#  Discrete causal Bayesian networks
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def investment():
    # Ness, listings 12.1 to 12.3.
    return sp.bayes_net(
        "C -> X; C -> Y; X -> Y; Y -> U",
        cpts={
            "C": {"bear": 0.5, "bull": 0.5},
            "X": {
                "bear": {"debt": 0.8, "equity": 0.2},
                "bull": {"debt": 0.2, "equity": 0.8},
            },
            "Y": {
                ("bear", "debt"): {"failure": 0.3, "success": 0.7},
                ("bull", "debt"): {"failure": 0.9, "success": 0.1},
                ("bear", "equity"): {"failure": 0.7, "success": 0.3},
                ("bull", "equity"): {"failure": 0.6, "success": 0.4},
            },
            "U": lambda Y: -1000 if Y == "failure" else 99000,
        },
    )


def test_seeing_and_doing_give_the_books_expected_utilities(investment):
    # Listing 12.5 prints 57000, 37000, 39000 and 34000.
    assert investment.expectation("U", evidence={"X": "debt"}) == pytest.approx(57000)
    assert investment.expectation("U", evidence={"X": "equity"}) == pytest.approx(37000)
    assert investment.expectation("U", do={"X": "debt"}) == pytest.approx(39000)
    assert investment.expectation("U", do={"X": "equity"}) == pytest.approx(34000)
    payoff = {"failure": -1000, "success": 99000}
    assert investment.expectation(
        "Y", do={"X": "debt"}, values=payoff
    ) == pytest.approx(39000)


def test_newcomb_intervening_while_knowing_ones_intent():
    # Listings 12.8 to 12.12 print 51000, 50000, 951000 and 950000.
    net = sp.bayes_net(
        'intent -> "AI prediction"; intent -> choice; "AI prediction" -> "box B"; '
        'choice -> U; "box B" -> U',
        cpts={
            "intent": {"B": 0.5, "both": 0.5},
            "choice": lambda intent: intent,
            "AI prediction": {
                "B": {"B": 0.95, "both": 0.05},
                "both": {"B": 0.05, "both": 0.95},
            },
            # parents whose names are not identifiers arrive as one dict
            "box B": lambda p: 1_000_000 if p["AI prediction"] == "B" else 0,
            "U": lambda p: p["box B"] + (1000 if p["choice"] == "both" else 0),
        },
    )
    want = {
        ("both", "both"): 51000,
        ("both", "B"): 50000,
        ("B", "both"): 951000,
        ("B", "B"): 950000,
    }
    for (intent, choice), value in want.items():
        got = net.expectation("U", do={"choice": choice}, evidence={"intent": intent})
        assert got == pytest.approx(value)
    # Evidential reasoning one-boxes, causal reasoning two-boxes.
    assert net.expectation("U", evidence={"choice": "B"}) > net.expectation(
        "U", evidence={"choice": "both"}
    )
    assert net.expectation("U", do={"choice": "both"}) > net.expectation(
        "U", do={"choice": "B"}
    )


def _enumerate(net, names, evidence):
    total = {}
    for combo in itertools.product(*[net.states[v] for v in net.nodes]):
        a = dict(zip(net.nodes, combo))
        p = 1.0
        for v in net.nodes:
            p *= (
                net.cpt(v)
                .to_numpy()
                .reshape([len(net.states[u]) for u in (*net.parents(v), v)])[
                    tuple(net.states[u].index(a[u]) for u in (*net.parents(v), v))
                ]
            )
        if all(a[k] == val for k, val in evidence.items()):
            key = tuple(a[v] for v in names)
            total[key] = total.get(key, 0.0) + p
    z = sum(total.values())
    return {k: v / z for k, v in total.items()}


def test_variable_elimination_equals_enumeration():
    rng = np.random.default_rng(11)
    spec = "A -> C; B -> C; C -> D; B -> E; D -> F; E -> F; A -> F"
    g = sp.dag(spec)
    card = {"A": 2, "B": 3, "C": 2, "D": 3, "E": 2, "F": 2}

    def random_table(node):
        pa = sorted(g.parents(node))
        states = list(range(card[node]))
        if not pa:
            return dict(zip(states, rng.dirichlet(np.ones(card[node]))))
        return {
            cfg: dict(zip(states, rng.dirichlet(np.ones(card[node]))))
            for cfg in itertools.product(*[range(card[p]) for p in pa])
        }

    net = sp.bayes_net(g, cpts={v: random_table(v) for v in "ABCDEF"})
    for names, evidence, do in [
        (["A", "E"], {"F": 1, "D": 2}, None),
        (["F"], {"A": 0}, {"C": 1}),
        (["B", "D", "F"], {}, {"E": 0, "A": 1}),
    ]:
        got = net.query(names, evidence=evidence, do=do)
        want = _enumerate(net.do(do) if do else net, names, evidence)
        for row in got.itertuples(index=False):
            assert row.prob == pytest.approx(want[tuple(row[:-1])], abs=1e-13)
        assert got["prob"].sum() == pytest.approx(1.0, abs=1e-12)


def test_do_cuts_the_incoming_arrows(investment):
    cut = investment.do(X="equity")
    assert cut.parents("X") == () and investment.parents("X") == ("C",)
    assert cut.prob({"C": "bull"}, evidence={"X": "equity"}) == pytest.approx(0.5)
    assert investment.prob({"C": "bull"}, evidence={"X": "equity"}) == pytest.approx(
        0.8
    )


def _monty_hall():
    doors = [1, 2, 3]

    def host(car, first, coin):
        free = [d for d in doors if d not in (car, first)]
        return free[0] if len(free) == 1 or coin == "tails" else free[1]

    def second(first, host, strategy):
        if strategy == "stay":
            return first
        return next(d for d in doors if d not in (first, host))

    return sp.bayes_net(
        "car -> host; first -> host; coin -> host; first -> second; host -> second; "
        "strategy -> second; second -> win; car -> win",
        cpts={
            "car": {d: 1 / 3 for d in doors},
            "first": {d: 1 / 3 for d in doors},
            "coin": {"tails": 0.5, "heads": 0.5},
            "strategy": {"stay": 0.5, "switch": 0.5},
            "host": host,
            "second": second,
            "win": lambda second, car: "win" if second == car else "lose",
        },
    )


def test_monty_hall_counterfactuals_are_the_books():
    # Listings 9.9 and 9.10 print 0.6667, 1.0000 and 0.6667.
    net = _monty_hall()
    assert net.prob({"win": "win"}, evidence={"strategy": "switch"}) == pytest.approx(
        2 / 3
    )
    cf = net.counterfactual(
        "win", evidence={"strategy": "stay", "win": "lose"}, do={"strategy": "switch"}
    )
    assert cf.set_index("win").loc["win", "prob"] == pytest.approx(1.0)
    cf = net.counterfactual("win", evidence={"win": "lose"}, do={"strategy": "switch"})
    assert cf.set_index("win").loc["win", "prob"] == pytest.approx(2 / 3)
    # A node the intervention cannot reach keeps its factual value.
    same = net.counterfactual("host", evidence={"host": 2}, do={"strategy": "switch"})
    assert same.set_index("host").loc[2, "prob"] == pytest.approx(1.0)


def test_counterfactual_needs_a_structural_model(investment):
    with pytest.raises(sp.AssumptionViolation, match="not deterministic"):
        investment.counterfactual("Y", evidence={"Y": "failure"}, do={"X": "equity"})


def test_fit_smoothing_and_placeholders():
    df = pd.DataFrame({"A": list("xxyy") * 25, "B": list("uvuv") * 25})
    df = df[~((df.A == "y") & (df.B == "v"))]
    df = pd.concat([df, pd.DataFrame({"A": ["y"] * 5, "B": ["u"] * 5})])
    child = np.where(df.B == "u", "hi", "lo")
    df = df.assign(C=child)
    net = sp.bayes_net("A -> C; B -> C", df)
    assert net.unseen == {"C": 1} and net.n_obs == len(df)
    with pytest.warns(sp.AssumptionWarning, match="positivity"):
        net.query("C", do={"A": "y", "B": "v"})
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        net.query("C", do={"A": "x", "B": "v"})  # observed: no warning
    smooth = sp.bayes_net("A -> C; B -> C", df, prior=1)
    assert smooth.unseen == {} and smooth.cpt("C").loc[("y", "v")].tolist() == [
        0.5,
        0.5,
    ]
    assert smooth.cpt("C").loc[("x", "u"), "hi"] == pytest.approx(26 / 27)


def test_simulation_follows_the_network(investment):
    sim = investment.simulate(40000, seed=5, do={"X": "debt"})
    assert set(sim["X"]) == {"debt"}
    assert (sim["Y"] == "success").mean() == pytest.approx(0.4, abs=0.01)
    assert (sim["C"] == "bull").mean() == pytest.approx(0.5, abs=0.01)


def test_bool_and_integer_states_stay_apart():
    net = sp.bayes_net(
        "A -> B", cpts={"A": {True: 0.25, False: 0.75}, "B": lambda A: int(A)}
    )
    assert net.prob({"B": 1}) == pytest.approx(0.25)
    assert net.prob({"A": True}, evidence={"B": 1}) == pytest.approx(1.0)


def test_network_refusals(investment):
    df = pd.DataFrame({"A": list("ab") * 30, "B": np.arange(60)})
    with pytest.raises(sp.MethodIncompatibility, match="distinct values"):
        sp.bayes_net("A -> B", df)
    with pytest.raises(ColumnNotFound, match="unobserved"):
        sp.bayes_net("A -> B; U -> A; U -> B", df.assign(B=list("uv") * 30))
    with pytest.raises(sp.MethodIncompatibility, match="sum to one"):
        sp.bayes_net("A", cpts={"A": {"x": 0.5, "y": 0.6}})
    with pytest.raises(sp.MethodIncompatibility, match="no row"):
        sp.bayes_net("A -> B", cpts={"A": {"x": 0.5, "y": 0.5}, "B": {"x": {"u": 1.0}}})
    with pytest.raises(sp.MethodIncompatibility, match="no table"):
        sp.bayes_net("A -> B", cpts={"A": {"x": 1.0}})
    with pytest.raises(sp.MethodIncompatibility, match="not a state"):
        investment.query("Y", evidence={"X": "bonds"})
    with pytest.raises(ColumnNotFound):
        investment.query("Z")
    with pytest.raises(sp.MethodIncompatibility, match="both queried"):
        investment.query("Y", evidence={"Y": "success"})
    zero = sp.bayes_net("A -> B", cpts={"A": {"x": 1.0, "y": 0.0}, "B": lambda A: A})
    with pytest.raises(sp.AssumptionViolation, match="probability zero"):
        zero.query("A", evidence={"B": "y"})
    with pytest.raises(sp.MethodIncompatibility, match="values="):
        investment.expectation("Y")


# --------------------------------------------------------------------------- #
#  Front door without mediator variation in an arm
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def flat_arm():
    rng = np.random.default_rng(2)
    n = 2000
    u = rng.normal(size=n)
    d = (u + rng.normal(size=n) > -1).astype(int)
    m = np.where(d == 1, rng.binomial(1, 0.6, n), 0)  # untreated never have M = 1
    y = 3.0 * m + u + rng.normal(size=n)
    return pd.DataFrame({"y": y, "d": d, "m": m})


def test_front_door_refuses_an_arm_without_mediator_variation(flat_arm):
    # It used to fit y ~ m in that arm, get a minimum-norm slope of zero,
    # and return the effect shrunk by the arm's share.
    with pytest.raises(sp.IdentificationFailure, match="does not vary") as err:
        sp.front_door(flat_arm, y="y", treat="d", mediator="m", n_boot=10, seed=0)
    assert "additive" in err.value.recovery_hint
    assert err.value.diagnostics["arms_without_mediator_variation"] == [0]


def test_additive_front_door_is_the_product_of_two_coefficients(flat_arm):
    df = flat_arm
    res = sp.front_door(
        df, y="y", treat="d", mediator="m", n_boot=20, seed=0, outcome_model="additive"
    )
    a = np.polyfit(df["d"], df["m"], 1)[0]
    design = np.column_stack([np.ones(len(df)), df["m"], df["d"]])
    b = np.linalg.lstsq(design, df["y"].to_numpy(), rcond=None)[0][1]
    assert res.estimate == pytest.approx(a * b, rel=1e-10)
    assert res.estimate == pytest.approx(3.0 * 0.6, abs=0.15)
    assert res.model_info["outcome_model"] == "additive"


def test_front_door_models_agree_without_interaction():
    rng = np.random.default_rng(7)
    n = 4000
    u = rng.normal(size=n)
    d = rng.binomial(1, 1 / (1 + np.exp(-u)))
    m = rng.binomial(1, 1 / (1 + np.exp(1 - 2 * d)))
    df = pd.DataFrame({"y": 1.5 * m + u + rng.normal(size=n), "d": d, "m": m})
    by_arm = sp.front_door(df, y="y", treat="d", mediator="m", n_boot=10, seed=0)
    additive = sp.front_door(
        df, y="y", treat="d", mediator="m", n_boot=10, seed=0, outcome_model="additive"
    )
    assert by_arm.estimate == pytest.approx(additive.estimate, abs=0.05)
    with pytest.raises(ValueError, match="outcome_model"):
        sp.front_door(df, y="y", treat="d", mediator="m", outcome_model="saturated")
    with pytest.raises(sp.IdentificationFailure, match="linear function"):
        sp.front_door(
            df.assign(m=df["d"]),
            y="y",
            treat="d",
            mediator="m",
            outcome_model="additive",
        )


# --------------------------------------------------------------------------- #
#  Inverse probability weights that rest on a few units
# --------------------------------------------------------------------------- #


def _selected(slope, n=3000, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=n)
    t = rng.binomial(1, 1 / (1 + np.exp(-slope * x)))
    return pd.DataFrame({"y": x + t + rng.normal(size=n), "t": t, "x": x})


def test_ipw_reports_and_flags_a_thin_effective_sample():
    with pytest.warns(sp.AssumptionWarning, match="effective sample") as rec:
        res = sp.ipw(
            _selected(3.0),
            y="y",
            treat="t",
            covariates=["x"],
            estimand="ATT",
            se_method="sandwich",
        )
    info = res.model_info
    assert info["ess_ratio_min"] < 0.2
    assert info["ess_ratio_min"] == pytest.approx(
        min(
            info["ess_treated"] / info["n_treated"],
            info["ess_control"] / info["n_control"],
        )
    )
    assert "sp.aipw" in rec[0].message.alternative_functions
    assert any(v["test"] == "ipw_effective_sample_size" for v in res.violations())


def test_ipw_is_quiet_under_ordinary_selection():
    with warnings.catch_warnings():
        warnings.simplefilter("error", sp.AssumptionWarning)
        res = sp.ipw(
            _selected(0.5), y="y", treat="t", covariates=["x"], se_method="sandwich"
        )
    assert res.model_info["ess_ratio_min"] > 0.8
    assert not any(v["test"] == "ipw_effective_sample_size" for v in res.violations())


def test_effective_sample_size_is_kishs():
    df = _selected(1.0)
    res = sp.ipw(df, y="y", treat="t", covariates=["x"], se_method="sandwich")
    e = res.model_info["_pscore"]
    w = 1 / (1 - e[df.t == 0])
    assert res.model_info["ess_control"] == pytest.approx(
        w.sum() ** 2 / (w**2).sum(), rel=1e-10
    )


# --------------------------------------------------------------------------- #
#  A treatment column of labels
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize(
    "call",
    [
        lambda d: sp.ipw(d, y="y", treat="t", covariates=["x"], n_bootstrap=5),
        lambda d: sp.aipw(d, y="y", treat="t", covariates=["x"]),
        lambda d: sp.g_computation(d, y="y", treat="t", covariates=["x"]),
        lambda d: sp.tmle(d, y="y", treat="t", covariates=["x"]),
        lambda d: sp.metalearner(d, y="y", treat="t", covariates=["x"], learner="t"),
        lambda d: sp.front_door(d, y="y", treat="t", mediator="x", n_boot=5),
        lambda d: sp.ebalance(d, y="y", treat="t", covariates=["x"]),
        lambda d: sp.cbps(d, y="y", treat="t", covariates=["x"]),
        lambda d: sp.dml(d, y="y", treat="t", covariates=["x"]),
    ],
    ids=[
        "ipw",
        "aipw",
        "g_computation",
        "tmle",
        "metalearner",
        "front_door",
        "ebalance",
        "cbps",
        "dml",
    ],
)
def test_labelled_treatment_gets_a_named_error(call):
    df = _selected(0.5, n=200)
    df["t"] = np.where(df["t"] == 1, "high", "low")
    with pytest.raises(sp.MethodIncompatibility, match="holds labels") as err:
        call(df)
    assert (
        "== 'low'" in err.value.recovery_hint or "== 'high'" in err.value.recovery_hint
    )


def test_boolean_treatment_is_still_accepted():
    df = _selected(0.5, n=400)
    as_bool = sp.aipw(
        df.assign(t=df["t"].astype(bool)), y="y", treat="t", covariates=["x"]
    )
    assert as_bool.estimate == pytest.approx(
        sp.aipw(df, y="y", treat="t", covariates=["x"]).estimate
    )


# --------------------------------------------------------------------------- #
#  Refutation tests
# --------------------------------------------------------------------------- #


@pytest.fixture(scope="module")
def confounded():
    rng = np.random.default_rng(0)
    n = 800
    x = rng.normal(size=n)
    d = (x + rng.normal(size=n) > 0).astype(int)
    return pd.DataFrame({"y": 2.0 * d + 3.0 * x + rng.normal(size=n), "d": d, "x": x})


def _naive(frame, y, treat, **_):
    return (
        frame.loc[frame[treat] == 1, y].mean() - frame.loc[frame[treat] == 0, y].mean()
    )


@pytest.mark.parametrize(
    "method",
    ["placebo_treatment", "dummy_outcome", "random_common_cause", "data_subset"],
)
def test_an_adjusted_estimator_survives_every_refutation(confounded, method):
    res = sp.refute(
        sp.aipw,
        confounded,
        y="y",
        treat="d",
        covariates=["x"],
        method=method,
        n_simulations=40,
        seed=0,
    )
    assert isinstance(res, sp.RefutationResult) and not res.refuted
    assert res.estimate == pytest.approx(2.0, abs=0.4)
    if method in ("placebo_treatment", "dummy_outcome"):
        assert res.expected == 0.0 and abs(res.new_effect) < 0.3
    else:
        assert res.expected == res.estimate and res.new_effect == pytest.approx(
            res.estimate, abs=0.1
        )
    assert method in res.summary() and res.to_dict(detail="minimal")["method"] == method


def test_a_confounded_dummy_outcome_catches_missing_adjustment(confounded):
    kw = dict(
        y="y",
        treat="d",
        method="dummy_outcome",
        n_simulations=40,
        seed=0,
        outcome_function=lambda f: 3.0 * f["x"],
    )
    assert sp.refute(_naive, confounded, **kw).refuted
    assert not sp.refute(sp.aipw, confounded, covariates=["x"], **kw).refuted
    # Shuffling the outcome removes the confounding with it: the naive
    # comparison passes that weaker check.
    weak = sp.refute(
        _naive,
        confounded,
        y="y",
        treat="d",
        method="dummy_outcome",
        n_simulations=40,
        seed=0,
    )
    assert not weak.refuted and weak.details["outcome"] == "permuted"


def test_placebo_draws_give_a_randomisation_p_value(confounded):
    res = sp.refute(
        sp.aipw,
        confounded,
        y="y",
        treat="d",
        covariates=["x"],
        n_simulations=40,
        seed=0,
    )
    assert res.permutation_pvalue == pytest.approx(1 / 41)
    assert res.p_value >= 2 / 41 and res.interval[0] < 0 < res.interval[1]


def test_refute_reads_regression_results_and_numbers(confounded):
    ols = lambda f, y, treat, covariates: sp.regress(
        f"{y} ~ {treat} + " + " + ".join(covariates), data=f
    )  # noqa: E731
    res = sp.refute(
        ols,
        confounded,
        y="y",
        treat="d",
        covariates=["x"],
        method="random_common_cause",
        n_simulations=40,
        seed=0,
    )
    direct = sp.regress("y ~ d + x", data=confounded).params["d"]
    assert res.estimate == pytest.approx(direct) and not res.refuted
    assert "_random_cause" not in confounded.columns


def test_refute_refusals(confounded):
    base = dict(y="y", treat="d", covariates=["x"])
    with pytest.raises(sp.MethodIncompatibility, match="not one of"):
        sp.refute(sp.aipw, confounded, method="add_unobserved_common_cause", **base)
    with pytest.raises(sp.MethodIncompatibility, match="nothing could be refuted"):
        sp.refute(sp.aipw, confounded, n_simulations=38, **base)
    with pytest.raises(sp.MethodIncompatibility, match="covariates"):
        sp.refute(_naive, confounded, y="y", treat="d", method="random_common_cause")
    with pytest.raises(ColumnNotFound):
        sp.refute(sp.aipw, confounded, y="y", treat="d", covariates=["nope"])
    with pytest.raises(sp.MethodIncompatibility, match="scalar effect"):
        sp.refute(lambda f, **k: "text", confounded, y="y", treat="d")
    with pytest.raises(sp.MethodIncompatibility, match="subset_fraction"):
        sp.refute(
            sp.aipw, confounded, method="data_subset", subset_fraction=1.5, **base
        )

    calls = {"n": 0}

    def flaky(frame, **_):
        calls["n"] += 1
        if calls["n"] > 1:
            raise RuntimeError("cannot fit")
        return 1.0

    with pytest.raises(sp.DataInsufficient, match="reruns"):
        sp.refute(flaky, confounded, y="y", treat="d", n_simulations=40, seed=0)


# --------------------------------------------------------------------------- #
#  Structure learning
# --------------------------------------------------------------------------- #


def _support(result):
    cp = result["cpdag"].to_numpy()
    return ((cp + cp.T) > 0).astype(int)


def test_pc_keeps_every_skeleton_edge_when_colliders_clash():
    # Two colliders used to zero both directions of a shared edge, which
    # then left the graph: 85 of these 200 draws lost at least one edge.
    rng = np.random.default_rng(0)
    clashes = 0
    for _ in range(60):
        X = rng.normal(size=(120, 6))
        X[:, 2] += X[:, 0] + X[:, 1]
        X[:, 4] += X[:, 2] + X[:, 3]
        X[:, 5] += X[:, 4] + 0.5 * X[:, 1]
        out = sp.pc_algorithm(pd.DataFrame(X, columns=list("abcdef")), alpha=0.1)
        assert np.array_equal(out["skeleton"].to_numpy(), _support(out))
        assert out["n_edges"] == out["skeleton"].to_numpy().sum() // 2
        clashes += bool(out["orientation_conflicts"])
    assert clashes > 5  # the situation does arise


def test_pc_learns_from_categorical_columns():
    rng = np.random.default_rng(4)
    n = 4000
    a = rng.integers(0, 2, n)
    b = rng.integers(0, 2, n)
    c = (a + b + (rng.random(n) < 0.1)) % 3
    d = (c + (rng.random(n) < 0.15)) % 3
    lab = np.array(["u", "v", "w"])
    df = pd.DataFrame({"A": lab[a], "B": lab[b], "C": lab[c], "D": lab[d]})
    out = sp.pc_algorithm(df, ci_test="chi-square")
    assert sorted(out["edges"]) == [("A", "C"), ("B", "C"), ("C", "D")]
    assert out["undirected_edges"] == [] and out["ci_test"] == "chi-square"
    same = sp.pc_algorithm(df, ci_test="g-test")
    assert sorted(same["edges"]) == sorted(out["edges"])


def test_discovery_says_why_it_cannot_use_labels():
    df = pd.DataFrame(
        {"A": list("uv") * 50, "B": list("uuvv") * 25, "C": list("vuuv") * 25}
    )
    with pytest.raises(sp.MethodIncompatibility, match="non-numeric left out") as err:
        sp.pc_algorithm(df)
    assert "chi-square" in err.value.recovery_hint
    with pytest.raises(sp.MethodIncompatibility, match="numeric columns"):
        sp.pc_algorithm(df, variables=["A", "B", "C"])
    with pytest.raises(sp.MethodIncompatibility, match="not one of"):
        sp.pc_algorithm(df, ci_test="hsic")
    with pytest.raises(sp.MethodIncompatibility, match="Gaussian BIC"):
        sp.ges(df)
    with pytest.raises(ValueError, match="non-numeric column"):
        sp.notears(df)
    with pytest.raises(ValueError, match="non-numeric column"):
        sp.fci(df)
    with pytest.raises(sp.MethodIncompatibility, match="non-Gaussian noise"):
        sp.lingam(df)
    wide = pd.DataFrame(
        np.random.default_rng(0).normal(size=(100, 3)), columns=list("ABC")
    )
    with pytest.raises(sp.MethodIncompatibility, match="distinct values"):
        sp.pc_algorithm(wide, ci_test="chi-square")


def test_fci_tests_the_large_separating_set():
    rng = np.random.default_rng(0)
    n = 3000
    a, b, c = rng.normal(size=(3, n))
    df = pd.DataFrame(
        {
            "A": a,
            "B": b,
            "C": c,
            "X": a + b + c + rng.normal(size=n),
            "Y": a + b + c + rng.normal(size=n),
        }
    )
    assert sp.fci(df).skeleton.loc["X", "Y"] == 0


# --------------------------------------------------------------------------- #
#  Random graphs: the whole chain against the truth
# --------------------------------------------------------------------------- #


class _ExactJoint:
    """Stands in for the data: exact marginals of a known observed joint."""

    def __init__(self, names, joint, states):
        self.names, self.joint, self.states, self.n = names, joint, states, 1

    def marginal(self, want):
        drop = tuple(i for i, v in enumerate(self.names) if v not in want)
        arr = self.joint.sum(axis=drop)
        kept = [v for v in self.names if v in want]
        return np.transpose(arr, [kept.index(v) for v in want])


def test_identified_estimands_equal_the_interventional_distribution():
    # Random DAGs with latent nodes and random tables. Whenever the query
    # is identified, the estimand (after simplification) evaluated on the
    # exact joint of the observed variables must be P(Y | do(X)) of the
    # full model. The same graphs check that an adjustment set is reported
    # exactly when some subset of the candidates is one.
    from statspai.dag import identification as idm

    rng = np.random.default_rng(123)
    identified = 0
    for _ in range(120):
        k = int(rng.integers(4, 7))
        names = [f"V{i}" for i in range(k)]
        edges = [
            (names[i], names[j])
            for i in range(k)
            for j in range(i + 1, k)
            if rng.random() < 0.45
        ]
        latent = [v for v in names[:-1] if rng.random() < 0.3]
        observed = [v for v in names if v not in latent]
        if len(observed) < 2:
            continue
        spec = "; ".join([f"{a} -> {b}" for a, b in edges] + names)
        g = sp.dag(spec, latent=latent or None)
        ix, iy = sorted(int(i) for i in rng.choice(len(observed), 2, replace=False))
        X, Y = observed[ix], observed[iy]

        candidates = sorted(g.observed_nodes - {X, Y} - g.descendants(X))
        exists = any(
            g._is_valid_adjustment(X, Y, set(c))
            for r in range(len(candidates) + 1)
            for c in itertools.combinations(candidates, r)
        )
        found = g.adjustment_sets(X, Y)
        assert bool(found) == exists
        assert all(g._is_valid_adjustment(X, Y, s) for s in found)

        res = sp.identify(g, X, Y)
        if not res.identifiable:
            continue
        identified += 1
        card = {v: int(rng.integers(2, 4)) for v in names}
        cpts = {}
        for v in names:
            pa = sorted(g.parents(v))
            states = list(range(card[v]))
            if not pa:
                cpts[v] = dict(zip(states, rng.dirichlet(np.ones(card[v]))))
            else:
                cpts[v] = {
                    cfg: dict(zip(states, rng.dirichlet(np.ones(card[v]))))
                    for cfg in itertools.product(*[range(card[p]) for p in pa])
                }
        net = sp.bayes_net(sp.dag(spec), cpts=cpts)  # latents observed here
        obs = [v for v in net.nodes if v not in latent]
        joint = net.query(obs)["prob"].to_numpy().reshape([card[v] for v in obs])
        emp = _ExactJoint(obs, joint, {v: list(range(card[v])) for v in obs})
        scope, arr = idm._array(res._expression, emp, False)
        parts = [(scope, arr)]
        extra = tuple(v for v in scope if v not in (X, Y))
        if extra:
            parts.append((extra, emp.marginal(extra)))
        parts += [((v,), np.ones(card[v])) for v in (X, Y) if v not in scope]
        value = idm._contract(parts, (X, Y))
        for x in range(card[X]):
            truth = net.query(Y, do={X: x})["prob"].to_numpy()
            assert np.allclose(value[x], truth, atol=1e-12)
    assert identified > 80


def test_estimate_when_the_outcome_does_not_respond():
    # X has no path to Y: P(Y | do(X)) is P(Y) for every x.
    rng = np.random.default_rng(1)
    df = pd.DataFrame(
        {
            "X": rng.integers(0, 2, 500),
            "Y": rng.integers(0, 3, 500),
            "Z": rng.integers(0, 2, 500),
        }
    )
    est = sp.identify(sp.dag("Z -> X; Z -> Y"), "X", "Y").estimate(df)
    marginal = df["Y"].value_counts(normalize=True).sort_index().to_numpy()
    for x in (0, 1):
        assert np.allclose(est.loc[est.X == x, "prob"], marginal, atol=1e-12)
