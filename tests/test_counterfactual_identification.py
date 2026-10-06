"""Counterfactual identification: ``sp.identify_counterfactual`` (ID*, IDC*).

Three kinds of evidence. Formulas the algorithm returns are evaluated in
random structural causal models and compared with the counterfactual
probability computed there by enumeration (soundness). Queries known not
to be identifiable are shown so with two models that agree on every
experiment and disagree on the query. Verdicts on textbook queries are
compared with the R package ``cfid`` in
``tests/reference_parity/test_ness_causal_ai_parity.py``.
"""

from __future__ import annotations

import itertools

import numpy as np
import pandas as pd
import pytest

import statspai as sp

Y0 = [("Y", 1, {"X": 0})]
X1 = [("X", 1)]


# --------------------------------------------------------------------------- #
#  A structural causal model small enough to enumerate
# --------------------------------------------------------------------------- #


class _SCM:
    """Binary variables; every unit is one configuration of the roots."""

    def __init__(self, V, directed, bidirected, rng):
        self.V = V
        self.pa = {v: [a for a, b in directed if b == v] for v in V}
        self.roots = {f"L_{a}_{b}": 2 for a, b in bidirected}
        self.roots.update({f"N_{v}": 2 for v in V})
        self.ex = {
            v: [f"L_{a}_{b}" for a, b in bidirected if v in (a, b)] + [f"N_{v}"]
            for v in V
        }
        self.p = {r: rng.dirichlet(np.ones(k)) for r, k in self.roots.items()}
        self.f = {
            v: rng.integers(0, 2, size=[2] * (len(self.pa[v]) + len(self.ex[v])))
            for v in V
        }
        self.states = {v: [0, 1] for v in V}
        self._memo = {}

    def _world(self, u, do):
        val = {}
        for v in self.V:
            if v in do:
                val[v] = do[v]
            else:
                key = [val[p] for p in self.pa[v]] + [u[r] for r in self.ex[v]]
                val[v] = int(self.f[v][tuple(key)])
        return val

    def joint(self, terms):
        """P(all the events), each event (variable, value, do)."""
        names = list(self.roots)
        total = 0.0
        for combo in itertools.product(*[range(self.roots[r]) for r in names]):
            u = dict(zip(names, combo))
            worlds = {}
            for name, value, do in terms:
                key = tuple(sorted(do.items()))
                if key not in worlds:
                    worlds[key] = self._world(u, do)
                if worlds[key][name] != value:
                    break
            else:
                total += float(np.prod([self.p[r][u[r]] for r in names]))
        return total

    def prob(self, event, do=None):  # what CounterfactualIdentification.evaluate asks
        key = (tuple(sorted(event.items())), tuple(sorted((do or {}).items())))
        if key not in self._memo:
            self._memo[key] = self.joint(
                [(v, x, dict(do or {})) for v, x in event.items()]
            )
        return self._memo[key]


def _random_graph(rng):
    k = int(rng.integers(3, 5))
    V = [f"V{i}" for i in range(k)]
    pairs = list(itertools.combinations(V, 2))
    directed = [p for p in pairs if rng.random() < 0.5]
    bidirected = [p for p in pairs if rng.random() < 0.25]
    spec = "; ".join(
        [f"{a} -> {b}" for a, b in directed]
        + [f"{a} <-> {b}" for a, b in bidirected]
        + V
    )
    return V, directed, bidirected, spec


def _random_event(rng, V):
    v = V[int(rng.integers(len(V)))]
    do = {x: int(rng.integers(2)) for x in V if x != v and rng.random() < 0.35}
    return (v, int(rng.integers(2)), do)


def test_identified_formulas_equal_the_counterfactual_probability():
    rng = np.random.default_rng(2026)
    identified = refused = 0
    for _ in range(220):
        V, directed, bidirected, spec = _random_graph(rng)
        event = [_random_event(rng, V) for _ in range(int(rng.integers(1, 3)))]
        given = [_random_event(rng, V) for _ in range(int(rng.integers(0, 3)))]
        scm = _SCM(V, directed, bidirected, rng)
        try:
            res = sp.identify_counterfactual(sp.dag(spec), event, given=given or None)
        except sp.AssumptionViolation:
            assert scm.joint(given) == pytest.approx(0.0, abs=1e-12)
            continue
        if not res.identifiable:
            refused += 1
            continue
        den = scm.joint(given) if given else 1.0
        if den < 1e-9:
            continue
        identified += 1
        want = scm.joint(event + given) / den
        assert res.evaluate(scm) == pytest.approx(want, abs=1e-9), (
            spec,
            event,
            given,
            res.estimand,
        )
    assert identified > 100 and refused > 20


# --------------------------------------------------------------------------- #
#  Textbook queries
# --------------------------------------------------------------------------- #


def test_effect_of_treatment_on_the_treated():
    backdoor = sp.identify_counterfactual(sp.dag("Z -> X; Z -> Y; X -> Y"), Y0, X1)
    assert backdoor.identifiable and backdoor.from_observational_data
    assert backdoor.query == "P(Y[X=0]=1 | X=1)"
    front = sp.identify_counterfactual(sp.dag("X -> W -> Y; X <-> Y"), Y0, X1)
    assert front.identifiable and front.from_observational_data
    assert front.estimand == (
        "[sum_{W} [P(W | do(X=0)) * P(X=1, Y=1 | do(W))]] / [P(X=1)]"
    )
    for spec in ("X -> Y; X <-> Y", "Z -> X -> Y; X <-> Y"):
        res = sp.identify_counterfactual(sp.dag(spec), Y0, X1)
        assert not res.identifiable and "NOT IDENTIFIABLE" in res.summary()


def test_the_papers_worked_example():
    # Shpitser and Pearl (2008), section 4: P(y_x | x', z_d, d) is
    # sum_w P_{z,w}(y, x') P_x(w) / P(x').
    g = sp.dag("X -> W -> Y; D -> Z -> Y; X <-> Y")
    res = sp.identify_counterfactual(g, Y0, [("X", 1), ("Z", 1, {"D": 1}), ("D", 1)])
    assert res.estimand == (
        "[sum_{W} [P(W | do(X=0)) * P(X=1, Y=1 | do(W, Z=1))]] / [P(X=1)]"
    )


def test_the_books_recommendation_query():
    # Ness, listing 10.8: P(A_{T=-t} = +a | T = +t).
    g = sp.dag("T -> W -> A; B -> V -> A; C -> T; C -> A; C -> B")
    res = sp.identify_counterfactual(g, [("A", 1, {"T": 0})], [("T", 1)])
    assert res.identifiable and res.from_observational_data
    for term in (
        "P(A=1 | do(C, V, W))",
        "P(W | do(T=0))",
        "P(V | do(B))",
        "P(B | do(C))",
        "P(T=1 | do(C))",
        "P(C)",
    ):
        assert term in res.estimand


def test_interventional_query_agrees_with_identify():
    plain = sp.identify_counterfactual(sp.dag("X -> Y; X <-> Y"), Y0)
    assert plain.estimand == "P(Y=1 | do(X=0))"
    # With a confounder the formula factorises over it; the value is the
    # one sp.identify gives.
    g = sp.dag("Z -> X; Z -> Y; X -> Y")
    res = sp.identify_counterfactual(g, Y0)
    assert res.estimand == "sum_{Z} [P(Y=1 | do(X=0, Z)) * P(Z)]"
    rng = np.random.default_rng(0)
    z = rng.integers(0, 2, 4000)
    x = (rng.random(4000) < np.where(z == 1, 0.7, 0.3)).astype(int)
    y = (rng.random(4000) < 0.2 + 0.3 * x + 0.3 * z).astype(int)
    df = pd.DataFrame({"Z": z, "X": x, "Y": y})
    direct = sp.identify(g, "X", "Y").estimate(df)
    want = direct[(direct.X == 0) & (direct.Y == 1)]["prob"].item()
    assert res.estimate(df) == pytest.approx(want, abs=1e-12)


def test_events_that_are_certain_or_impossible():
    g = sp.dag("X -> Y")
    sure = sp.identify_counterfactual(g, [("X", 0, {"X": 0})])
    assert sure.estimand == "1"
    never = sp.identify_counterfactual(g, [("X", 1, {"X": 0})])
    assert never.estimand == "0"
    implied = sp.identify_counterfactual(g, [("Y", 1)], given=[("Y", 1), ("X", 0)])
    assert implied.estimand == "1"
    with pytest.raises(sp.AssumptionViolation, match="inconsistent"):
        sp.identify_counterfactual(g, Y0, given=[("X", 1), ("X", 0)])


def test_consistency_merges_the_two_worlds():
    # Y_{X=0} is Y on units whose X is 0: the query is observational.
    res = sp.identify_counterfactual(
        sp.dag("X -> Y; X <-> Y"), [("Y", 1, {"X": 0})], given=[("X", 0)]
    )
    assert res.identifiable and res.estimand == "[P(X=0, Y=1)] / [P(X=0)]"


# --------------------------------------------------------------------------- #
#  What no experiment determines
# --------------------------------------------------------------------------- #


def test_probability_of_necessity_is_not_identified():
    g = sp.dag("X -> Y")
    pn = [("Y", 0, {"X": 0})]
    seen = [("X", 1), ("Y", 1)]
    assert not sp.identify_counterfactual(g, pn, given=seen).identifiable

    # Two models with the same P(X), P(Y | do(X)) and so the same data from
    # any experiment, and different probabilities of necessity.
    def model(response):
        # response: probabilities of the four types (y if x=0, y if x=1)
        def joint(terms):
            total = 0.0
            for x_nat in (0, 1):
                for (y0, y1), p in response.items():
                    ok = True
                    for name, value, *rest in terms:
                        do = rest[0] if rest else {}
                        x = do.get("X", x_nat)
                        got = x if name == "X" else (y1 if x else y0)
                        ok &= got == value
                    total += 0.5 * p * ok
            return total

        return joint

    a = model({(0, 0): 0.25, (0, 1): 0.25, (1, 0): 0.25, (1, 1): 0.25})
    b = model({(0, 1): 0.5, (1, 0): 0.5})
    for x in (0, 1):
        for y in (0, 1):
            ev = [("Y", y, {"X": x})]
            assert a(ev) == pytest.approx(b(ev))  # every experiment agrees
            assert a([("X", x), ("Y", y)]) == pytest.approx(b([("X", x), ("Y", y)]))
    pn_a = a(pn + seen) / a(seen)
    pn_b = b(pn + seen) / b(seen)
    assert pn_a == pytest.approx(0.5) and pn_b == pytest.approx(1.0)


def test_one_variable_in_two_worlds_is_refused():
    for event in (
        [("Y", 1, {"X": 0}), ("Y", 1, {"X": 1})],
        [("Y", 0), ("Y", 0, {"X": 0})],
    ):
        assert not sp.identify_counterfactual(sp.dag("X -> Y"), event).identifiable


# --------------------------------------------------------------------------- #
#  Numbers
# --------------------------------------------------------------------------- #


def test_estimate_recovers_the_effect_on_the_treated():
    # Exact population: Z -> X, Z -> Y, X -> Y, with a different effect of
    # X in the two strata of Z, so that ATT and ATE differ.
    rows = []
    for z, x, y in itertools.product((0, 1), repeat=3):
        px = 0.8 if z else 0.2
        py = 0.1 + (0.6 if z else 0.2) * x + 0.1 * z
        p = 0.5 * (px if x else 1 - px) * (py if y else 1 - py)
        rows.extend([(z, x, y)] * int(round(p * 10000)))
    df = pd.DataFrame(rows, columns=["Z", "X", "Y"])
    g = sp.dag("Z -> X; Z -> Y; X -> Y")
    res = sp.identify_counterfactual(g, Y0, given=X1)
    y0_treated = res.estimate(df)
    # P(Z=1 | X=1) = 0.8: P(Y_0=1 | X=1) = 0.8 * 0.2 + 0.2 * 0.1
    assert y0_treated == pytest.approx(0.18, abs=1e-12)
    att = df.loc[df.X == 1, "Y"].mean() - y0_treated
    assert att == pytest.approx(0.8 * 0.6 + 0.2 * 0.2, abs=1e-12)  # not the ATE, 0.4
    net = sp.bayes_net(g, df)
    assert res.evaluate(net) == pytest.approx(y0_treated, abs=1e-12)


def test_estimate_refuses_what_needs_an_experiment():
    g = sp.dag("X -> Y; X <-> Y")
    res = sp.identify_counterfactual(g, Y0)
    assert res.identifiable and not res.from_observational_data
    assert res.observational == {"P(Y | do(X))": None}
    df = pd.DataFrame({"X": [0, 1] * 20, "Y": [0, 1, 1, 0] * 10})
    with pytest.raises(sp.IdentificationFailure, match="not identified from obs"):
        res.estimate(df)
    refused = sp.identify_counterfactual(g, Y0, given=X1)
    with pytest.raises(sp.IdentificationFailure, match="not identified"):
        refused.estimate(df)


def test_input_is_checked():
    g = sp.dag("X -> Y")
    with pytest.raises(sp.MethodIncompatibility, match="cannot read"):
        sp.identify_counterfactual(g, ["Y=1"])
    with pytest.raises(sp.MethodIncompatibility, match="no counterfactual event"):
        sp.identify_counterfactual(g, [])
    with pytest.raises(sp.MethodIncompatibility, match="sp.dag"):
        sp.identify_counterfactual("X -> Y", Y0)
    from statspai.exceptions import ColumnNotFound

    with pytest.raises(ColumnNotFound):
        sp.identify_counterfactual(g, [("W", 1)])
    one = sp.identify_counterfactual(g, ("Y", 1, {"X": 0}))  # a single event
    assert one.estimand == "P(Y=1 | do(X=0))"
    as_dict = sp.identify_counterfactual(g, [{"var": "Y", "value": 1, "do": {"X": 0}}])
    assert as_dict.estimand == one.estimand
