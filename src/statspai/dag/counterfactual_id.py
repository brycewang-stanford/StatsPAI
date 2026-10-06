"""
Identification of counterfactual queries: ID* and IDC*.

A counterfactual query mixes worlds. "Among those who took the treatment
and recovered, how many would have recovered without it?" is
``P(Y_{X=0} = 1 | X = 1, Y = 1)``: the evidence is about the world as it
was, the outcome about a world in which ``X`` was set to 0. Whether such
a quantity can be computed at all, and from which experiments, depends on
the graph.

:func:`identify_counterfactual` answers with the ID* and IDC* algorithms
[@shpitser2008complete]. It builds the counterfactual graph of the query
(the worlds it mentions, with every variable that must take the same
value in two worlds merged into one node), splits it into confounded
components, and returns a formula in *interventional* distributions
``P(v | do(x))`` or reports that no such formula exists. Each
interventional term is then passed to :func:`statspai.identify`; when all
of them reduce to the observational distribution, the query can be
estimated from observational data alone.

This is an independent implementation from the paper. The R package
``cfid`` and the Python package ``y0`` implement the same algorithms and
are used as references in the tests.
"""

from __future__ import annotations

import itertools
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, FrozenSet, List, Mapping, Optional, Tuple

from .._result_serialize import ResultProtocolMixin
from ..exceptions import (
    AssumptionViolation,
    ColumnNotFound,
    IdentificationFailure,
    MethodIncompatibility,
)

__all__ = ["identify_counterfactual", "CounterfactualIdentification"]

# A value is ("c", constant) or ("s", symbol id): a value fixed by the
# query, or one bound by a summation.
Value = Tuple[str, Any]
DoSet = FrozenSet[Tuple[str, Value]]
# One counterfactual event: variable, the interventions of its world, value.
Term = Tuple[str, DoSet, Value]


# --------------------------------------------------------------------------- #
#  Expression tree over interventional distributions
# --------------------------------------------------------------------------- #


@dataclass(frozen=True)
class _CDo:
    """``P(vars = values | do(do))``."""

    vars: Tuple[Tuple[str, Value], ...]
    do: Tuple[Tuple[str, Value], ...]


@dataclass(frozen=True)
class _CSum:
    over: Tuple[Tuple[int, str], ...]  # (symbol id, variable it ranges over)
    body: Any


@dataclass(frozen=True)
class _CProd:
    terms: Tuple[Any, ...]


@dataclass(frozen=True)
class _CFrac:
    num: Any
    den: Any


@dataclass(frozen=True)
class _CConst:
    value: float


class _Fail(Exception):
    """The query is not identifiable from interventional distributions."""


class _Undefined(Exception):
    """The conditioning event has probability zero."""


# --------------------------------------------------------------------------- #
#  The graph, as the algorithm needs it
# --------------------------------------------------------------------------- #


class _Graph:
    """Observed nodes, their observed parents and their latent parents."""

    def __init__(self, dag: Any) -> None:
        if getattr(dag, "_latent", None):
            dag = dag.latent_projection()
        self.dag = dag
        self.observed = sorted(v for v in dag.nodes if not v.startswith("_L_"))
        self.parents = {
            v: sorted(p for p in dag.parents(v) if not p.startswith("_L_"))
            for v in self.observed
        }
        self.latents = {
            v: frozenset(p for p in dag.parents(v) if p.startswith("_L_"))
            for v in self.observed
        }
        order: List[str] = []
        placed: set = set()
        pending = list(self.observed)
        while pending:
            ready = [v for v in pending if all(p in placed for p in self.parents[v])]
            order.extend(ready)
            placed.update(ready)
            pending = [v for v in pending if v not in placed]
        self.order = order


class _CG:
    """A counterfactual graph: the outcome of make-cg."""

    def __init__(self) -> None:
        self.nodes: List[Any] = []  # non-fixed nodes, ("var", name, world)
        self.fixed: List[Any] = []  # ("fix", name, value)
        self.parents: Dict[Any, List[Any]] = {}
        self.value: Dict[Any, Value] = {}  # values the query assigns
        self.query: List[Any] = []  # the nodes of gamma'
        self.term_nodes: List[Any] = []  # the node each input event became
        self.g: Optional[_Graph] = None

    def subscript(self, node: Any) -> Dict[str, Value]:
        """Interventions that are ancestors of *node*."""
        out: Dict[str, Value] = {}
        stack, seen = [node], set()
        while stack:
            n = stack.pop()
            if n in seen:
                continue
            seen.add(n)
            if n[0] == "fix":
                out[n[1]] = n[2]
            else:
                stack.extend(self.parents[n])
        return out

    def components(self) -> List[List[Any]]:
        """C-components of the non-fixed nodes.

        Two nodes are confounded when they share a latent parent of the
        original graph, or when they are the same variable in two worlds
        (they then share that variable's own disturbance).
        """
        assert self.g is not None
        left = list(self.nodes)
        comps: List[List[Any]] = []
        while left:
            comp = [left.pop(0)]
            grew = True
            while grew:
                grew = False
                for n in list(left):
                    if any(
                        n[1] == m[1] or self.g.latents[n[1]] & self.g.latents[m[1]]
                        for m in comp
                    ):
                        comp.append(n)
                        left.remove(n)
                        grew = True
            comps.append(comp)
        return comps


def _same_value(a: Optional[Value], b: Optional[Value]) -> bool:
    return a is not None and b is not None and a == b


def _make_cg(g: _Graph, gamma: List[Term], use_values: bool = True) -> Optional[_CG]:
    """Counterfactual graph of *gamma*, or ``None`` when it is inconsistent.

    With ``use_values=False`` two copies of a variable are merged only
    when their parents are the same nodes, not when the parents merely
    attain the same values in this query. The graph then describes the
    worlds whatever the events turn out to be, which is what a
    d-separation statement about it needs.
    """
    worlds: List[DoSet] = []
    for _, do, _ in gamma:
        if do not in worlds:
            worlds.append(do)
    fixed_in = [dict(do) for do in worlds]

    rep: Dict[Any, Any] = {}

    def find(n: Any) -> Any:
        while rep.get(n, n) != n:
            n = rep[n]
        return n

    assigned: Dict[Any, Value] = {}
    for name, do, val in gamma:
        node = ("var", name, worlds.index(do))
        if node in assigned and assigned[node] != val:
            if assigned[node][0] == "c" and val[0] == "c":
                return None
            raise _Fail()
        assigned[node] = val

    def parent_node(p: str, w: int) -> Any:
        if p in fixed_in[w]:
            return ("fix", p, fixed_in[w][p])
        return find(("var", p, w))

    def known(n: Any) -> Optional[Value]:
        return n[2] if n[0] == "fix" else assigned.get(n)

    # Merge, in topological order, the copies of a variable whose parents
    # are the same nodes or attain the same values (Lemmas 24 and 25).
    chosen_parents: Dict[Any, Dict[str, Any]] = {}
    for v in g.order:
        live = [w for w in range(len(worlds)) if v not in fixed_in[w]]
        for w in live:
            chosen_parents[("var", v, w)] = {p: parent_node(p, w) for p in g.parents[v]}
        for a, b in itertools.combinations(live, 2):
            na, nb = find(("var", v, a)), find(("var", v, b))
            if na == nb:
                continue
            pa, pb = chosen_parents[na], chosen_parents[nb]
            if all(
                pa[p] == pb[p]
                or (use_values and _same_value(known(pa[p]), known(pb[p])))
                for p in g.parents[v]
            ):
                va, vb = assigned.get(na), assigned.get(nb)
                if va is not None and vb is not None and va != vb:
                    if va[0] == "c" and vb[0] == "c":
                        return None
                    raise _Fail()
                # The merged node keeps, for each parent, a natural
                # (non-fixed) copy when one of the two has it: a variable
                # observed at x carries more than one set to x.
                merged = {
                    p: (pa[p] if pa[p][0] == "var" else pb[p]) for p in g.parents[v]
                }
                rep[nb] = na
                chosen_parents[na] = merged
                if vb is not None:
                    assigned[na] = vb

    cg = _CG()
    cg.g = g
    query = []
    for name, do, _ in gamma:
        n = find(("var", name, worlds.index(do)))
        cg.term_nodes.append(n)
        if n not in query:
            query.append(n)
    # Restrict to the ancestors of the query.
    stack, seen = list(query), set()
    while stack:
        n = stack.pop()
        if n in seen:
            continue
        seen.add(n)
        if n[0] == "fix":
            cg.fixed.append(n)
            continue
        ps = [find(q) if q[0] == "var" else q for q in chosen_parents[n].values()]
        cg.parents[n] = ps
        stack.extend(ps)
        cg.nodes.append(n)
    order = {v: i for i, v in enumerate(g.order)}
    cg.nodes.sort(key=lambda n: (order[n[1]], n[2]))
    cg.query = query
    cg.value = {n: assigned[n] for n in cg.nodes if n in assigned}
    return cg


# --------------------------------------------------------------------------- #
#  ID* and IDC*
# --------------------------------------------------------------------------- #


class _Solver:
    def __init__(self, g: _Graph) -> None:
        self.g = g
        self._next = 0

    def fresh(self) -> int:
        self._next += 1
        return self._next

    def id_star(self, gamma: List[Term]) -> Any:
        # Lines 1 to 3.
        kept: List[Term] = []
        for name, do, val in gamma:
            own = dict(do).get(name)
            if own is None:
                kept.append((name, do, val))
            elif own == val:
                continue  # y_{..y..} = y always holds
            elif own[0] == "c" and val[0] == "c":
                return _CConst(0.0)  # y_{..y'..} = y never does
            else:
                raise _Fail()
        if not kept:
            return _CConst(1.0)

        cg = _make_cg(self.g, kept)  # line 4
        if cg is None:
            return _CConst(0.0)  # line 5
        comps = cg.components()

        if len(comps) > 1:  # line 6
            bound: List[Tuple[int, str]] = []
            value = dict(cg.value)
            for n in cg.nodes:
                if n not in value:
                    sym = self.fresh()
                    value[n] = ("s", sym)
                    bound.append((sym, n[1]))
            factors = []
            for comp in comps:
                inside = set(comp)
                sub: List[Term] = []
                for n in comp:
                    # Everything outside the component is held fixed. For
                    # this node that means the outside nodes reached by
                    # walking up from it through the component.
                    held: Dict[str, Value] = {}
                    stack, seen = [n], set()
                    while stack:
                        m = stack.pop()
                        if m in seen:
                            continue
                        seen.add(m)
                        for q in cg.parents[m]:
                            if q in inside:
                                stack.append(q)
                                continue
                            val = q[2] if q[0] == "fix" else value[q]
                            if held.setdefault(q[1], val) != val:
                                raise _Fail()
                    sub.append((n[1], frozenset(held.items()), value[n]))
                factors.append(self.id_star(sub))
            if any(isinstance(f, _CConst) and f.value == 0.0 for f in factors):
                return _CConst(0.0)
            factors = [f for f in factors if not isinstance(f, _CConst)] or [
                _CConst(1.0)
            ]
            body = factors[0] if len(factors) == 1 else _CProd(tuple(factors))
            return _CSum(tuple(bound), body) if bound else body

        # Lines 7 to 9: one component.
        do_all: Dict[str, Value] = {}
        for n in cg.query:
            for x, val in cg.subscript(n).items():
                if x in do_all and do_all[x] != val:
                    raise _Fail()
                do_all[x] = val
        # A variable that is set in one world and left to its natural value
        # in another, inside the same confounded component, is a conflict
        # unless the natural value is known to equal the one set (line 8).
        for n in cg.nodes:
            if n[1] in do_all and cg.value.get(n) != do_all[n[1]]:
                raise _Fail()
        names = [n[1] for n in cg.query]
        if len(set(names)) < len(names):
            raise _Fail()  # one variable in two worlds that did not merge
        # A variable set to x whose natural value is also observed to be x
        # was not changed by the intervention (consistency): the event is
        # about its natural value, so it leaves the subscript.
        for n in cg.query:
            do_all.pop(n[1], None)
        return _CDo(
            tuple(sorted((n[1], cg.value[n]) for n in cg.query)),
            tuple(sorted(do_all.items())),
        )

    def idc_star(self, gamma: List[Term], delta: List[Term]) -> Any:
        if not delta:
            return self.id_star(gamma)
        try:
            den = self.id_star(delta)  # line 1
        except _Fail:
            den = None  # P(delta) is obtained from the joint below
        if isinstance(den, _CConst) and den.value == 0.0:
            raise _Undefined()
        joint = gamma + delta
        # Events that name their own variable in the subscript are either
        # certain or impossible; ID* disposes of them (lines 2 and 3).
        live = [
            (t, k < len(gamma)) for k, t in enumerate(joint) if t[0] not in dict(t[1])
        ]
        if live and _make_cg(self.g, [t for t, _ in live]) is None:
            return _CConst(0.0)  # lines 2 and 3
        # Line 4: evidence with no back-door path to the outcome can be
        # moved from the condition into the subscripts (rule 2 of the
        # do-calculus). The test is made on the graph of the worlds as
        # such: merging copies because their parents happen to take equal
        # values in this query would draw arrows that hold only on the
        # event, and the events keep their own subscripts for the same
        # reason.
        cg = _make_cg(self.g, [t for t, _ in live], use_values=False) if live else None
        if cg is not None:
            node = dict(zip(range(len(live)), cg.term_nodes))
            d_idx = [k for k, (_, is_g) in enumerate(live) if not is_g]
            d_nodes = {node[k] for k in d_idx}
            g_idx = [
                k for k, (_, is_g) in enumerate(live) if is_g and node[k] not in d_nodes
            ]
            if not g_idx:
                return _CConst(1.0)  # every outcome event is also evidence
            g_nodes = {node[k] for k in g_idx}
            for k in d_idx:
                d = node[k]
                others = d_nodes - {d}
                if not _separated_without_outgoing(cg, d, g_nodes, others):
                    continue
                desc = _descendants(cg, d)
                value = live[k][0][2]

                def moved(m: int) -> Term:
                    name, do, val = live[m][0]
                    if node[m] in desc:
                        do = frozenset({**dict(do), d[1]: value}.items())
                    return (name, do, val)

                return self.idc_star(
                    [moved(m) for m in g_idx],
                    [moved(m) for m in d_idx if node[m] != d],
                )
        num = self.id_star(joint)  # line 5
        if isinstance(num, _CConst) and num.value == 0.0:
            return num
        if den is None:
            # P'(delta): the joint summed over the values of the outcome
            # events, each value a bound symbol.
            bound = [(self.fresh(), t[0]) for t in gamma]
            free = [(t[0], t[1], ("s", sym)) for t, (sym, _) in zip(gamma, bound)]
            den = _CSum(tuple(bound), self.id_star(free + delta))
        return _CFrac(num, den)


def _descendants(cg: _CG, node: Any) -> set:
    out: set = set()
    grew = True
    while grew:
        grew = False
        for n in cg.nodes:
            if n not in out and any(p == node or p in out for p in cg.parents[n]):
                out.add(n)
                grew = True
    return out


def _separated_without_outgoing(cg: _CG, node: Any, targets: set, given: set) -> bool:
    """Whether *node* is d-separated from *targets* given the remaining
    evidence, once its outgoing arrows are removed (rule 2 of the
    do-calculus on the counterfactual graph)."""
    from .graph import DAG

    assert cg.g is not None
    label = {n: f"n{i}" for i, n in enumerate([*cg.nodes, *cg.fixed])}
    graph = DAG()
    for n in label.values():
        graph.add_node(n)
    for n in cg.nodes:
        for p in cg.parents[n]:
            if p != node:
                graph.add_edge(label[p], label[n])
    # Shared disturbances: the latent parents of the original graph, and
    # each variable's own disturbance across its copies.
    for a, b in itertools.combinations(cg.nodes, 2):
        if a[1] == b[1] or cg.g.latents[a[1]] & cg.g.latents[b[1]]:
            graph.add_bidirected(label[a], label[b])
    cond = {label[m] for m in given}
    return all(graph.d_separated(label[node], label[t], cond) for t in targets)


# --------------------------------------------------------------------------- #
#  Rendering and evaluation
# --------------------------------------------------------------------------- #


def _show_value(name: str, val: Value, names: Dict[int, str]) -> str:
    return f"{name}={val[1]!r}" if val[0] == "c" else names.get(val[1], name)


def _render(expr: Any, names: Optional[Dict[int, str]] = None) -> str:
    names = dict(names or {})
    if isinstance(expr, _CConst):
        return f"{expr.value:g}"
    if isinstance(expr, _CDo):
        head = ", ".join(_show_value(v, val, names) for v, val in expr.vars)
        if not expr.do:
            return f"P({head})"
        do = ", ".join(_show_value(v, val, names) for v, val in expr.do)
        return f"P({head} | do({do}))"
    if isinstance(expr, _CProd):
        return " * ".join(sorted(_render(t, names) for t in expr.terms))
    if isinstance(expr, _CFrac):
        return f"[{_render(expr.num, names)}] / [{_render(expr.den, names)}]"
    assert isinstance(expr, _CSum), type(expr).__name__
    taken = set(names.values())
    for sym, var in expr.over:
        label = var
        while label in taken:
            label += "'"
        names[sym] = label
        taken.add(label)
    over = ", ".join(sorted(names[s] for s, _ in expr.over))
    return f"sum_{{{over}}} [{_render(expr.body, names)}]"


def _leaves(expr: Any) -> List[_CDo]:
    if isinstance(expr, _CDo):
        return [expr]
    if isinstance(expr, _CProd):
        return [x for t in expr.terms for x in _leaves(t)]
    if isinstance(expr, _CFrac):
        return _leaves(expr.num) + _leaves(expr.den)
    if isinstance(expr, _CSum):
        return _leaves(expr.body)
    return []


def _value_of(
    expr: Any,
    prob: Callable[[Dict[str, Any], Dict[str, Any]], float],
    states: Mapping[str, List[Any]],
    env: Dict[int, Any],
) -> float:
    def concrete(val: Value) -> Any:
        return val[1] if val[0] == "c" else env[val[1]]

    if isinstance(expr, _CConst):
        return expr.value
    if isinstance(expr, _CDo):
        return prob(
            {v: concrete(val) for v, val in expr.vars},
            {v: concrete(val) for v, val in expr.do},
        )
    if isinstance(expr, _CProd):
        out = 1.0
        for t in expr.terms:
            out *= _value_of(t, prob, states, env)
            if out == 0.0:
                break
        return out
    if isinstance(expr, _CFrac):
        den = _value_of(expr.den, prob, states, env)
        if den == 0.0:
            raise AssumptionViolation(
                "identify_counterfactual: the conditioning event has "
                "probability zero.",
                recovery_hint="Check the values named in given=.",
            )
        return _value_of(expr.num, prob, states, env) / den
    assert isinstance(expr, _CSum), type(expr).__name__
    total = 0.0
    for combo in itertools.product(*[states[var] for _, var in expr.over]):
        inner = dict(env)
        inner.update({sym: s for (sym, _), s in zip(expr.over, combo)})
        total += _value_of(expr.body, prob, states, inner)
    return total


# --------------------------------------------------------------------------- #
#  Public API
# --------------------------------------------------------------------------- #


@dataclass
class CounterfactualIdentification(ResultProtocolMixin):
    """Outcome of :func:`identify_counterfactual`.

    Attributes
    ----------
    identifiable : bool
        Whether the query is a function of interventional distributions.
    estimand : str
        The formula, in terms ``P(v | do(x))``; a bare variable name
        stands for the value a summation ranges over.
    query : str
        The query as understood.
    experiments : list of str
        The distinct interventional distributions the formula uses.
    observational : dict
        For each of those, its formula in the observational distribution,
        or ``None`` when it needs an experiment.
    from_observational_data : bool
        True when every term has an observational formula, so that
        :meth:`estimate` can run on observational data.
    explanation : str

    Examples
    --------
    >>> import statspai as sp
    >>> g = sp.dag("Z -> X; Z -> Y; X -> Y")
    >>> res = sp.identify_counterfactual(
    ...     g, [("Y", 1, {"X": 0})], given=[("X", 1)])
    >>> isinstance(res, sp.CounterfactualIdentification), res.identifiable
    (True, True)
    """

    identifiable: bool
    estimand: str
    query: str
    experiments: List[str] = field(default_factory=list)
    observational: Dict[str, Optional[str]] = field(default_factory=dict)
    from_observational_data: bool = False
    explanation: str = ""

    _citation_keys = ("shpitser2008complete",)

    def summary(self) -> str:
        """A formatted account of the result.

        Examples
        --------
        >>> import statspai as sp
        >>> res = sp.identify_counterfactual(
        ...     sp.dag("X -> Y; X <-> Y"), [("Y", 1, {"X": 0})], given=[("X", 1)])
        >>> "NOT IDENTIFIABLE" in res.summary()
        True
        """
        width = 70
        verdict = "IDENTIFIABLE" if self.identifiable else "NOT IDENTIFIABLE"
        lines = [
            "=" * width,
            "  Counterfactual identification (ID* / IDC*)",
            "=" * width,
            f"  Query:    {self.query}",
            f"  Verdict:  {verdict} from interventional distributions",
        ]
        if self.identifiable:
            lines.append(f"  Estimand: {self.estimand}")
            if self.experiments:
                lines.append("-" * width)
                lines.append("  Interventional terms, and each from observational data")
                lines.append("-" * width)
                for term in self.experiments:
                    formula = self.observational.get(term)
                    lines.append(f"    {term}")
                    lines.append(
                        f"      = {formula}"
                        if formula is not None
                        else "      needs an experiment (no observational formula)"
                    )
        if self.explanation:
            lines.append("-" * width)
            lines.append(f"  {self.explanation}")
        lines.append("=" * width)
        return "\n".join(lines)

    def __repr__(self) -> str:
        status = "identifiable" if self.identifiable else "NOT identifiable"
        return f"CounterfactualIdentification({status}: {self.query})"

    # -- numeric evaluation -------------------------------------------- #

    def evaluate(self, net: Any) -> float:
        """
        Value of the estimand in a fully specified model.

        Parameters
        ----------
        net : BayesNet
            A :func:`statspai.bayes_net` over the graph, latent variables
            included. Its interventional queries supply every term.

        Returns
        -------
        float

        Examples
        --------
        >>> import statspai as sp
        >>> g = sp.dag("Z -> X; Z -> Y; X -> Y")
        >>> net = sp.bayes_net(g, cpts={
        ...     "Z": {0: 0.5, 1: 0.5},
        ...     "X": {0: {0: 0.8, 1: 0.2}, 1: {0: 0.3, 1: 0.7}},
        ...     "Y": {(0, 0): {0: 0.9, 1: 0.1}, (0, 1): {0: 0.6, 1: 0.4},
        ...           (1, 0): {0: 0.5, 1: 0.5}, (1, 1): {0: 0.2, 1: 0.8}}})
        >>> res = sp.identify_counterfactual(
        ...     g, [("Y", 1, {"X": 0})], given=[("X", 1)])
        >>> round(res.evaluate(net), 4)   # sum_z P(z | X=1) P(Y=1 | X=0, z)
        0.3333
        """
        expr = self._require()

        def prob(event: Dict[str, Any], do: Dict[str, Any]) -> float:
            return float(net.prob(event, do=do or None))

        return float(_value_of(expr, prob, net.states, {}))

    def estimate(self, data: Any) -> float:
        """
        Value of the estimand on categorical observational data.

        Every interventional term is replaced by its observational formula
        (:func:`statspai.identify`) evaluated on the data's frequencies.

        Parameters
        ----------
        data : pd.DataFrame
            One categorical column per variable the formulas use.

        Returns
        -------
        float

        Raises
        ------
        IdentificationFailure
            When a term needs an experiment. ``experiments`` and
            ``observational`` say which.

        Examples
        --------
        >>> import numpy as np, pandas as pd, statspai as sp
        >>> rng = np.random.default_rng(0)
        >>> n = 40000
        >>> z = rng.integers(0, 2, n)
        >>> x = (rng.random(n) < np.where(z == 1, 0.7, 0.2)).astype(int)
        >>> y = (rng.random(n) < 0.1 + 0.3 * x + 0.4 * z).astype(int)
        >>> df = pd.DataFrame({"Z": z, "X": x, "Y": y})
        >>> g = sp.dag("Z -> X; Z -> Y; X -> Y")
        >>> ett0 = sp.identify_counterfactual(
        ...     g, [("Y", 1, {"X": 0})], given=[("X", 1)]).estimate(df)
        >>> seen = df.loc[df.X == 1, "Y"].mean()
        >>> round(float(seen - ett0), 1)   # effect of treatment on the treated
        0.3
        """
        import numpy as np
        import pandas as pd

        from . import identification as idm

        expr = self._require()
        missing_terms = [t for t, f in self.observational.items() if f is None]
        if missing_terms:
            raise IdentificationFailure(
                "identify_counterfactual(...).estimate: these terms are not "
                f"identified from observational data: {missing_terms}.",
                recovery_hint=(
                    "They need experiments on the variables in do(...). With "
                    "a fully specified model use .evaluate(net)."
                ),
                diagnostics={"experiments_needed": missing_terms},
            )
        g: _Graph = getattr(self, "_graph")
        tables: Dict[Tuple[Tuple[str, ...], Tuple[str, ...]], Any] = {}
        used: set = set()
        plans = {}
        for leaf in _leaves(expr):
            key = (tuple(v for v, _ in leaf.do), tuple(v for v, _ in leaf.vars))
            if key in plans:
                continue
            X, Y = frozenset(key[0]), frozenset(key[1])
            V = frozenset(g.observed)
            if X:
                tree = idm._simplify(idm._ID(Y, X, idm._subgraph(g.dag, V), V), g.dag)
            else:
                tree = idm._Prob(tuple(sorted(Y)))
            plans[key] = tree
            used |= idm._all_vars(tree) | set(key[0]) | set(key[1])
        for sym_expr in [expr]:
            used |= {var for leaf in _sums(sym_expr) for _, var in leaf.over}
        absent = sorted(v for v in used if v not in data.columns)
        if absent:
            raise ColumnNotFound(
                f"identify_counterfactual(...).estimate: columns {absent} are "
                "not in the data.",
                diagnostics={"missing_columns": absent},
            )
        frame = data[sorted(used)].dropna()
        states: Dict[str, List[Any]] = {}
        codes: Dict[str, Any] = {}
        for v in sorted(used):
            code, uniques = pd.factorize(frame[v], sort=True)
            if len(uniques) > 50:
                raise MethodIncompatibility(
                    f"identify_counterfactual(...).estimate: column {v!r} "
                    f"takes {len(uniques)} distinct values; bin it first."
                )
            states[v] = [s.item() if isinstance(s, np.generic) else s for s in uniques]
            codes[v] = np.asarray(code)
        emp = idm._Empirical(codes, states)

        for key, tree in plans.items():
            scope, arr = idm._array(tree, emp, False)
            target = (*key[0], *key[1])
            parts = [(scope, arr)]
            extra = tuple(v for v in scope if v not in target)
            if extra:
                parts.append((extra, emp.marginal(extra)))
            for v in target:
                if v not in scope:
                    parts.append(((v,), np.ones(len(states[v]))))
            tables[key] = idm._contract(parts, target)

        def index(v: str, value: Any) -> int:
            for i, s in enumerate(states[v]):
                if s == value:
                    return i
            raise MethodIncompatibility(
                f"identify_counterfactual(...).estimate: {value!r} is not a "
                f"value of {v!r} in the data (values: {states[v]})."
            )

        def prob(event: Dict[str, Any], do: Dict[str, Any]) -> float:
            key = (tuple(sorted(do)), tuple(sorted(event)))
            idx = tuple(index(v, do[v]) for v in key[0]) + tuple(
                index(v, event[v]) for v in key[1]
            )
            return float(tables[key][idx])

        return float(_value_of(expr, prob, states, {}))

    def _require(self) -> Any:
        expr = getattr(self, "_expression", None)
        if not self.identifiable or expr is None:
            raise IdentificationFailure(
                "identify_counterfactual: the query is not identified, so "
                "there is no formula to evaluate.",
                recovery_hint=(
                    "Bound it instead (sp.manski_bounds), or state the "
                    "structural model and use sp.bayes_net(...).counterfactual "
                    "or sp.SCM."
                ),
            )
        return expr


def _sums(expr: Any) -> List[_CSum]:
    if isinstance(expr, _CSum):
        return [expr, *_sums(expr.body)]
    if isinstance(expr, _CProd):
        return [x for t in expr.terms for x in _sums(t)]
    if isinstance(expr, _CFrac):
        return _sums(expr.num) + _sums(expr.den)
    return []


def _parse_terms(terms: Any, g: _Graph, what: str) -> List[Term]:
    if terms is None:
        return []
    if isinstance(terms, tuple) and terms and isinstance(terms[0], str):
        terms = [terms]
    out: List[Term] = []
    for t in terms:
        if isinstance(t, Mapping):
            name, val, do = t.get("var"), t.get("value"), t.get("do")
        elif isinstance(t, (tuple, list)) and len(t) in (2, 3):
            name, val = t[0], t[1]
            do = t[2] if len(t) == 3 else None
        else:
            raise MethodIncompatibility(
                f"identify_counterfactual: cannot read {t!r} in {what}=.",
                recovery_hint=(
                    "Write each event as (variable, value) or "
                    "(variable, value, {intervened: value, ...})."
                ),
            )
        do = dict(do or {})
        unknown = [v for v in (name, *do) if v not in g.observed]
        if unknown:
            raise ColumnNotFound(
                f"identify_counterfactual: {unknown} in {what}= are not "
                "observed nodes of the graph.",
                recovery_hint=f"Observed nodes are {g.observed}.",
            )
        events = frozenset((str(k), ("c", v)) for k, v in do.items())
        out.append((str(name), events, ("c", val)))
    return out


def _show_term(t: Term) -> str:
    name, do, val = t
    if not do:
        return f"{name}={val[1]!r}"
    sub = ", ".join(f"{k}={v[1]!r}" for k, v in sorted(do))
    return f"{name}[{sub}]={val[1]!r}"


def identify_counterfactual(
    dag: Any,
    event: Any,
    given: Any = None,
) -> CounterfactualIdentification:
    """
    Decide whether a counterfactual probability is identifiable, and from
    what (ID* and IDC*, Shpitser and Pearl).

    Parameters
    ----------
    dag : DAG
        The causal graph. Latent variables are declared with
        ``latent=[...]`` or as bidirected edges ``A <-> B``.
    event : list of tuple
        The counterfactual events whose joint probability is wanted. Each
        is ``(variable, value)`` for the world as it is, or ``(variable,
        value, {intervened: value, ...})`` for the world in which those
        variables are set. ``[("Y", 1, {"X": 0})]`` is ``Y_{X=0} = 1``.
    given : list of tuple, optional
        Events to condition on, written the same way. They may refer to
        other worlds than ``event`` does.

    Returns
    -------
    CounterfactualIdentification
        ``identifiable``, the ``estimand`` in interventional
        distributions, which of its terms reduce to observational data,
        and ``evaluate(net)`` / ``estimate(data)`` for a number.

    Notes
    -----
    Identifiable here means: computable from the collection of all
    interventional distributions, that is from experiments. That is the
    most one can ask of a counterfactual without stating the structural
    functions. ``from_observational_data`` then says whether those
    experiments are themselves replaceable by observational data in this
    graph.

    Examples
    --------
    The effect of treatment on the treated is the standard example.
    ``P(Y_{X=0} = y | X = 1)`` is identified whenever the effect of X on Y
    has no unblocked confounding, and not otherwise:

    >>> import statspai as sp
    >>> ok = sp.identify_counterfactual(
    ...     sp.dag("Z -> X; Z -> Y; X -> Y"), [("Y", 1, {"X": 0})], given=[("X", 1)])
    >>> ok.identifiable, ok.from_observational_data
    (True, True)
    >>> print(ok.estimand.split(" / ")[0])
    [sum_{Z} [P(X=1 | do(Z)) * P(Y=1 | do(X=0, Z)) * P(Z)]]
    >>> bow = sp.identify_counterfactual(
    ...     sp.dag("X -> Y; X <-> Y"), [("Y", 1, {"X": 0})], given=[("X", 1)])
    >>> bow.identifiable
    False

    A probability of necessity, ``P(Y_{X=0} = 0 | X = 1, Y = 1)``, is not
    identified even without confounding: it needs the joint of ``Y`` in
    two worlds, which no experiment reveals.

    >>> pn = sp.identify_counterfactual(
    ...     sp.dag("X -> Y"), [("Y", 0, {"X": 0})], given=[("X", 1), ("Y", 1)])
    >>> pn.identifiable
    False

    References
    ----------
    [@shpitser2008complete]
    """
    from . import identification as idm
    from .graph import DAG
    from .identification import _NotIdentifiable
    from .identification import _render as _render_obs

    if not isinstance(dag, DAG):
        raise MethodIncompatibility(
            "identify_counterfactual: dag must be an sp.dag(...) graph."
        )
    g = _Graph(dag)
    gamma = _parse_terms(event, g, "event")
    delta = _parse_terms(given, g, "given")
    if not gamma:
        raise MethodIncompatibility(
            "identify_counterfactual: event= names no counterfactual event."
        )
    query = "P(" + ", ".join(_show_term(t) for t in gamma)
    query += (" | " + ", ".join(_show_term(t) for t in delta) if delta else "") + ")"

    solver = _Solver(g)
    try:
        expr = solver.idc_star(gamma, delta)
    except _Undefined:
        raise AssumptionViolation(
            f"identify_counterfactual: the conditioning event of {query} is "
            "inconsistent (probability zero), so the query is undefined.",
            recovery_hint="Check the values in given=.",
        ) from None
    except _Fail:
        return CounterfactualIdentification(
            identifiable=False,
            estimand="",
            query=query,
            explanation=(
                "No formula in interventional distributions exists: the query "
                "needs the joint behaviour of one variable in two worlds that "
                "share its unobserved causes. Experiments on this graph "
                "cannot determine it; a structural model or bounds can."
            ),
        )

    experiments: List[str] = []
    observational: Dict[str, Optional[str]] = {}
    V = frozenset(g.observed)
    seen: set = set()
    for leaf in _leaves(expr):
        X = tuple(v for v, _ in leaf.do)
        Yv = tuple(v for v, _ in leaf.vars)
        if (X, Yv) in seen:
            continue
        seen.add((X, Yv))
        label = f"P({', '.join(Yv)}" + (f" | do({', '.join(X)}))" if X else ")")
        experiments.append(label)
        if not X:
            observational[label] = label
            continue
        try:
            tree = idm._simplify(
                idm._ID(frozenset(Yv), frozenset(X), idm._subgraph(g.dag, V), V),
                g.dag,
            )
            observational[label] = _render_obs(tree)
        except _NotIdentifiable:
            observational[label] = None

    from_obs = all(f is not None for f in observational.values())
    result = CounterfactualIdentification(
        identifiable=True,
        estimand=_render(expr),
        query=query,
        experiments=experiments,
        observational=observational,
        from_observational_data=from_obs,
        explanation=(
            "Every interventional term is itself identified from the "
            "observational distribution, so observational data suffice."
            if from_obs
            else "Some interventional terms are not identified from the "
            "observational distribution; they need experiments."
        ),
    )
    object.__setattr__(result, "_expression", expr)
    object.__setattr__(result, "_graph", g)
    return result
