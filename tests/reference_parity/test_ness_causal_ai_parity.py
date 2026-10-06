"""R parity for the graph-side workflow of Ness, *Causal AI* (Manning, 2025;
code at https://github.com/altdeep/causalML).

The book builds a causal DAG, lists the conditional independencies it
implies, tests them on categorical data, fits the causal Markov kernels,
and reads interventional distributions off the fitted network. It does so
with pgmpy and y0. The references here are R packages that implement the
same things independently: ``dagitty`` (implied independencies, their
chi-square tests, adjustment sets, instruments), ``bnlearn`` (the same
tests under another name, and the kernels), ``causaleffect`` (the
Shpitser-Pearl ID algorithm) and ``pcalg`` (PC and FCI).

The reference numbers are in ``_fixtures/ness_causal_ai_R.json``, written
by ``_fixtures/_generate_ness_causal_ai.R`` on data simulated there; both
sides read the same CSV bytes.

Tolerances. Everything is deterministic: counts, sums of counts and the
chi-square distribution function. Statistics and probabilities agree to
1e-12 relative; the bound is written as 1e-10.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import statspai as sp

FIX = Path(__file__).parent / "_fixtures"
SPEC = "A -> E; S -> E; E -> O; E -> R; O -> T; R -> T"
TIGHT = 1e-10


@pytest.fixture(scope="module")
def ref():
    return json.loads((FIX / "ness_causal_ai_R.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def df():
    return pd.read_csv(FIX / "ness_categorical.csv")


def _z(z):
    """jsonlite unboxes a one-element conditioning set to a string."""
    if z is None or z == [] or z == {}:
        return ""
    return z if isinstance(z, str) else ", ".join(sorted(z))


def _rows(block):
    """jsonlite writes a named R list as an object; the names are not needed."""
    return list(block.values()) if isinstance(block, dict) else block


def test_implied_independencies_are_dagittys(ref):
    ours = {
        (a, b, ", ".join(sorted(z)))
        for a, b, z in sp.dag(SPEC).implied_independencies()
    }
    theirs = {(i["x"], i["y"], _z(i.get("z"))) for i in _rows(ref["implied"])}
    assert ours == theirs


def test_chi_square_tests_match_dagitty(ref, df):
    out = sp.dag(SPEC).test_implications(df).set_index(["x", "y", "given"])
    assert len(out) == len(ref["dagitty_chisq"])
    for row in _rows(ref["dagitty_chisq"]):
        got = out.loc[(row["x"], row["y"], _z(row.get("z")))]
        assert got["statistic"] == pytest.approx(row["x2"], rel=TIGHT)
        assert int(got["df"]) == row["df"]
        assert got["p_value"] == pytest.approx(row["p"], rel=TIGHT)


@pytest.mark.parametrize("test,key", [("chi-square", "x2_adf"), ("g-test", "mi_adf")])
def test_categorical_tests_match_bnlearn(ref, df, test, key):
    out = sp.dag(SPEC).test_implications(df, test=test).set_index(["x", "y", "given"])
    for row in _rows(ref["bnlearn"]):
        got = out.loc[(row["x"], row["y"], _z(row.get("z")))]
        assert got["statistic"] == pytest.approx(row[key]["stat"], rel=TIGHT)
        assert int(got["df"]) == row[key]["df"]
        assert got["p_value"] == pytest.approx(row[key]["p"], rel=TIGHT)


def test_the_omitted_arrow_is_what_fails(df):
    # The fixture adds A -> T, which the graph leaves out.
    out = sp.dag(SPEC).test_implications(df)
    worst = out.sort_values("p_value").iloc[0]
    assert (worst["x"], worst["y"]) == ("A", "T")
    assert worst["p_holm"] < 0.01
    assert (out.loc[~((out.x == "A") & (out.y == "T")), "p_holm"] > 0.05).all()


def test_maximum_likelihood_kernels_match_bnlearn(ref, df):
    net = sp.bayes_net(SPEC, df)
    cpt_e = net.cpt("E")
    for cell in ref["cpt_mle"]["E"]:
        assert cpt_e.loc[(cell["A"], cell["S"]), cell["E"]] == pytest.approx(
            cell["Freq"], rel=TIGHT
        )
    cpt_t = net.cpt("T")
    for cell in ref["cpt_mle"]["T"]:
        assert cpt_t.loc[(cell["O"], cell["R"]), cell["T"]] == pytest.approx(
            cell["Freq"], rel=TIGHT
        )


def test_dirichlet_prior_is_one_pseudo_count_per_cell(ref, df):
    cpt = sp.bayes_net(SPEC, df, prior=1).cpt("T")
    for cell in ref["cpt_T_laplace"]:
        assert cpt.loc[(cell["O"], cell["R"]), cell["T"]] == pytest.approx(
            cell["Freq"], rel=TIGHT
        )


def test_do_query_is_the_truncated_factorisation(ref, df):
    net = sp.bayes_net(SPEC, df)
    for e, want in ref["do_E"].items():
        got = net.query("T", do={"E": e}).set_index("T")["prob"]
        for t, p in want.items():
            assert got[t] == pytest.approx(p, rel=TIGHT)
        # and the ID estimand, evaluated on the data, is the same number
    est = sp.identify(sp.dag(SPEC), "E", "T").estimate(df).set_index(["E", "T"])["prob"]
    for e, want in ref["do_E"].items():
        for t, p in want.items():
            assert est[(e, t)] == pytest.approx(p, rel=TIGHT)


def test_conditional_query_matches_enumeration_in_r(ref, df):
    got = sp.bayes_net(SPEC, df).query("E", evidence={"T": "train"}).set_index("E")
    for e, p in ref["E_given_T_train"].items():
        assert got.loc[e, "prob"] == pytest.approx(p, rel=TIGHT)


def test_eight_confounders_have_an_adjustment_set(ref):
    spec = "X -> Y; " + "; ".join(f"Z{i} -> X; Z{i} -> Y" for i in range(1, 9))
    ours = [sorted(s) for s in sp.dag(spec).adjustment_sets("X", "Y")]
    assert ours == [list(v) for v in ref["adjust_eight"].values()]


GAMING = """
    PE -> Skill; PE -> Time; Time -> Skill
    Guild -> Engage; Guild -> Buy; Skill -> Engage; Skill -> Buy
    Time -> Engage; Time -> Buy; Assign -> Engage; Custom -> Engage
    Engage -> Won; Won -> Buy; Won -> Inventory; Buy -> Inventory
    PE [latent]
"""


def test_gaming_graph_agrees_with_dagitty(ref):
    g = sp.dag(GAMING)
    assert [sorted(s) for s in g.adjustment_sets("Engage", "Buy")] == [
        list(v) for v in ref["adjust_gaming"].values()
    ]
    instruments = sorted(
        v
        for v in g.observed_nodes - {"Engage", "Buy"}
        if "instrument" in g.classify_variable(v, "Engage", "Buy")
    )
    assert instruments == sorted(ref["instruments_gaming"])
    assert len(g.implied_independencies()) == ref["implied_gaming_n"]


@pytest.mark.parametrize(
    "name,spec",
    [
        ("backdoor", "Z -> X; Z -> Y; X -> Y"),
        ("bow", "X -> Y; X <-> Y"),
        ("frontdoor", "X -> M -> Y; X <-> Y"),
        ("iv", "Z -> X -> Y; X <-> Y"),
        ("napkin", "W -> Z -> X -> Y; W <-> X; W <-> Y"),
        ("m_bias", "X -> Y; X <-> Z; Z <-> Y"),
        ("fd_broken", "X -> M -> Y; X <-> Y; M <-> Y"),
    ],
)
def test_identification_verdicts_match_causaleffect(ref, name, spec):
    assert sp.identify(sp.dag(spec), "X", "Y").identifiable is ref["identify"][name]


def test_estimates_sum_to_one_where_identified(df):
    est = sp.identify(sp.dag(SPEC), "E", "T").estimate(df)
    totals = est.groupby("E")["prob"].sum()
    assert np.allclose(totals, 1.0, atol=1e-12)


# --------------------------------------------------------------------------- #
#  Structure learning: PC-stable against pcalg
# --------------------------------------------------------------------------- #


def _sep(result):
    return {
        tuple(sorted(pair)): sorted(s) for pair, s in result["separating_sets"].items()
    }


def test_pc_on_gaussian_data_is_pcalgs(ref):
    df = pd.read_csv(FIX / "ness_gaussian.csv")
    out = sp.pc_algorithm(df, alpha=0.05)
    want = ref["pc_gaussian"]
    assert sorted(out["edges"]) == sorted(tuple(e) for e in want["directed"])
    assert sorted(tuple(sorted(e)) for e in out["undirected_edges"]) == sorted(
        tuple(sorted(e)) for e in want["undirected"]
    )
    assert out["orientation_conflicts"] == []
    ours = _sep(out)
    assert len(ours) == len(want["sepsets"])
    for row in want["sepsets"]:
        assert ours[tuple(sorted((row["x"], row["y"])))] == sorted(row["s"])


def test_pc_on_categorical_data_has_pcalgs_skeleton_and_separating_sets(ref, df):
    out = sp.pc_algorithm(df, ci_test="chi-square", alpha=0.05)
    skeleton = {tuple(sorted(e)) for e in [*out["edges"], *out["undirected_edges"]]}
    assert skeleton == {tuple(sorted(e)) for e in ref["pc_categorical"]["skeleton"]}
    ours = _sep(out)
    for row in ref["pc_categorical"]["sepsets"]:
        assert ours[tuple(sorted((row["x"], row["y"])))] == sorted(row["s"])
    # The skeleton survives into the CPDAG edge for edge, and the clash
    # between the two colliders on R - T is reported, not resolved silently.
    cp = out["cpdag"].to_numpy()
    assert np.array_equal(out["skeleton"].to_numpy(), ((cp + cp.T) > 0).astype(int))
    assert out["orientation_conflicts"]


def test_pc_does_not_depend_on_column_order():
    df = pd.read_csv(FIX / "ness_gaussian.csv")
    base = sp.pc_algorithm(df)
    flipped = sp.pc_algorithm(df[df.columns[::-1]])
    assert sorted(base["edges"]) == sorted(flipped["edges"])
    assert {tuple(sorted(e)) for e in base["undirected_edges"]} == {
        tuple(sorted(e)) for e in flipped["undirected_edges"]
    }


@pytest.mark.parametrize(
    "csv,key,alpha",
    [
        ("ness_gaussian.csv", "fci_gaussian", 0.05),
        ("ness_three_causes.csv", "fci_three_causes", 0.05),
        ("ness_latent.csv", "fci_latent", 0.01),
    ],
)
def test_fci_pag_is_pcalgs(ref, csv, key, alpha):
    # Every edge with the mark at each end: 'o->', '-->', 'o-o', '<->'.
    out = sp.fci(pd.read_csv(FIX / csv), alpha=alpha)
    assert sorted(out.edges) == sorted(tuple(e) for e in ref[key])


def test_fci_separates_two_variables_with_three_common_causes(ref):
    # The search used to stop before trying the set {A, B, C}.
    pairs = {tuple(sorted((a, b))) for a, _, b in ref["fci_three_causes"]}
    assert ("X", "Y") not in pairs
    sk = sp.fci(pd.read_csv(FIX / "ness_three_causes.csv")).skeleton
    assert sk.loc["X", "Y"] == 0


def test_fci_draws_a_latent_common_cause_as_bidirected(ref):
    marks = {(a, b): m for a, m, b in ref["fci_latent"]}
    assert marks[("Y", "Z")] == "<->"
    ours = {
        (a, b): m
        for a, m, b in sp.fci(pd.read_csv(FIX / "ness_latent.csv"), alpha=0.01).edges
    }
    assert ours[("Y", "Z")] == "<->"


# --------------------------------------------------------------------------- #
#  Counterfactual identification: ID* / IDC* against cfid
# --------------------------------------------------------------------------- #

_Y0 = [("Y", 1, {"X": 0})]
_X1 = [("X", 1)]
CF_QUERIES = {
    "ett_backdoor": ("Z -> X; Z -> Y; X -> Y", _Y0, _X1),
    "ett_bow": ("X -> Y; X <-> Y", _Y0, _X1),
    "ett_frontdoor": ("X -> W -> Y; X <-> Y", _Y0, _X1),
    "ett_iv": ("Z -> X -> Y; X <-> Y", _Y0, _X1),
    "effect": ("X -> Y; X <-> Z; Z -> Y", _Y0, None),
    "two_worlds": ("X -> Y", [("Y", 1, {"X": 0}), ("Y", 1, {"X": 1})], None),
    "book_ett": (
        "T -> W -> A; B -> V -> A; C -> T; C -> A; C -> B",
        [("A", 1, {"T": 0})],
        [("T", 1)],
    ),
    "paper_example": (
        "X -> W -> Y; D -> Z -> Y; X <-> Y",
        _Y0,
        [("X", 1), ("Z", 1, {"D": 1}), ("D", 1)],
    ),
}


@pytest.mark.parametrize("name", sorted(CF_QUERIES))
def test_counterfactual_verdicts_match_cfid(ref, name):
    spec, event, given = CF_QUERIES[name]
    ours = sp.identify_counterfactual(sp.dag(spec), event, given=given)
    assert ours.identifiable is ref["cfid"][name]


def test_probability_of_necessity_is_where_cfid_0_1_8_differs(ref):
    # cfid 0.1.8 (CRAN), which made the fixture, answers
    # P(Y_{X=0} = 0 | X = 1, Y = 1) with the formula "0". In the model
    # Y = X the probability is 1. Its development version 0.1.9 reports
    # the query as not identifiable, as here; the query is the textbook
    # example of a counterfactual no experiment identifies
    # (tests/test_counterfactual_identification.py shows two models that
    # agree on every experiment and differ on it).
    assert ref["cfid"]["necessity"] is True
    ours = sp.identify_counterfactual(
        sp.dag("X -> Y"), [("Y", 0, {"X": 0})], given=[("X", 1), ("Y", 1)]
    )
    assert not ours.identifiable
