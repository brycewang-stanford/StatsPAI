"""
Discrete causal Bayesian networks.

A causal DAG whose variables are categorical, with one conditional
probability table per node (its *causal Markov kernel*). The tables are
estimated from data or written by hand. Any probability the network
implies is then computed exactly by variable elimination:

- a conditional, ``P(Y | X = x)``;
- an interventional one, ``P(Y | do(X = x))``, by replacing the kernel of
  ``X`` with a point mass and cutting its incoming arrows (the truncated
  factorisation);
- a counterfactual, ``P(Y_{X = x} | evidence)``, when every node other
  than the roots is a deterministic function of its parents, so that the
  roots carry all the randomness and the network is a structural causal
  model. The factual and the hypothetical world then share the roots
  (a twin network).

Every node of the graph must be a column of the data: a table cannot be
estimated for an unobserved variable. With latent variables, identify the
query first (:func:`statspai.identify`) and evaluate the estimand with
:meth:`statspai.IdentificationResult.estimate`.
"""

from __future__ import annotations

import inspect
import itertools
import warnings
from typing import Any, Callable, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from ..exceptions import (
    AssumptionViolation,
    AssumptionWarning,
    ColumnNotFound,
    MethodIncompatibility,
)
from .graph import DAG

__all__ = ["BayesNet", "bayes_net"]

#: A numeric column with more distinct values than this is refused rather
#: than tabulated one value per state.
_MAX_STATES = 50

_Factor = Tuple[Tuple[str, ...], np.ndarray]


class BayesNet:
    """A fitted or hand-specified discrete causal Bayesian network.

    Build one with :func:`statspai.bayes_net`.

    Attributes
    ----------
    dag : DAG
        The graph.
    states : dict
        ``{node: list of its states}``, in table order.
    nodes : list of str
        Nodes in a topological order.
    unseen : dict
        ``{node: number of parent configurations with no data}`` for the
        nodes that have any. The table rows for those configurations are
        uniform placeholders, and a query whose answer depends on them
        warns.

    Examples
    --------
    >>> import statspai as sp
    >>> net = sp.bayes_net(
    ...     "C -> X; C -> Y; X -> Y",
    ...     cpts={
    ...         "C": {"bear": 0.5, "bull": 0.5},
    ...         "X": {"bear": {"debt": 0.8, "equity": 0.2},
    ...               "bull": {"debt": 0.2, "equity": 0.8}},
    ...         "Y": {("bear", "debt"): {"fail": 0.3, "success": 0.7},
    ...               ("bull", "debt"): {"fail": 0.9, "success": 0.1},
    ...               ("bear", "equity"): {"fail": 0.7, "success": 0.3},
    ...               ("bull", "equity"): {"fail": 0.6, "success": 0.4}},
    ...     },
    ... )
    >>> round(net.prob({"Y": "success"}, evidence={"X": "debt"}), 4)
    0.58
    >>> round(net.prob({"Y": "success"}, do={"X": "debt"}), 4)
    0.4
    """

    def __init__(
        self,
        dag: DAG,
        states: Dict[str, List[Any]],
        tables: Dict[str, np.ndarray],
        parents: Dict[str, Tuple[str, ...]],
        unseen: Optional[Dict[str, np.ndarray]] = None,
        n_obs: Optional[int] = None,
    ) -> None:
        self.dag = dag
        self.states = states
        self._tables = tables  # axes: (*parents, node)
        self._parents = parents
        self._unseen = unseen or {}
        self.n_obs = n_obs
        self.nodes = _topological(parents)

    # ------------------------------------------------------------------ #
    #  Inspection
    # ------------------------------------------------------------------ #

    @property
    def unseen(self) -> Dict[str, int]:
        return {v: int(m.sum()) for v, m in self._unseen.items() if m.any()}

    def parents(self, node: str) -> Tuple[str, ...]:
        """Parents of *node*, in the order of its table's leading axes."""
        self._check_nodes([node])
        return self._parents[node]

    def cpt(self, node: str) -> pd.DataFrame:
        """
        The conditional probability table of *node*.

        Returns
        -------
        pd.DataFrame
            One column per state of *node*; one row per configuration of
            its parents (a single row labelled ``''`` for a root).

        Examples
        --------
        >>> import statspai as sp
        >>> net = sp.bayes_net("A -> B", cpts={
        ...     "A": {"no": 0.7, "yes": 0.3},
        ...     "B": {"no": {"lo": 0.9, "hi": 0.1}, "yes": {"lo": 0.4, "hi": 0.6}}})
        >>> float(net.cpt("B").loc["yes", "hi"])
        0.6
        """
        self._check_nodes([node])
        pa = self._parents[node]
        table = self._tables[node]
        cols = pd.Index(self.states[node], name=node)
        if not pa:
            return pd.DataFrame(table.reshape(1, -1), index=[""], columns=cols)
        flat = table.reshape(-1, table.shape[-1])
        if len(pa) == 1:
            index: pd.Index = pd.Index(self.states[pa[0]], name=pa[0])
        else:
            index = pd.MultiIndex.from_product(
                [self.states[p] for p in pa], names=list(pa)
            )
        return pd.DataFrame(flat, index=index, columns=cols)

    def summary(self) -> str:
        """Nodes, their states and parents, and the size of the tables.

        Examples
        --------
        >>> import statspai as sp
        >>> net = sp.bayes_net("A -> B", cpts={
        ...     "A": {"no": 0.7, "yes": 0.3},
        ...     "B": {"no": {"lo": 0.9, "hi": 0.1}, "yes": {"lo": 0.4, "hi": 0.6}}})
        >>> print(net.summary().splitlines()[0])
        Discrete causal Bayesian network: 2 nodes, 3 free parameters
        """
        free = sum(
            int(np.prod(t.shape[:-1])) * (t.shape[-1] - 1)
            for t in self._tables.values()
        )
        lines = [
            f"Discrete causal Bayesian network: {len(self.nodes)} nodes, "
            f"{free} free parameters"
            + (f", fitted on {self.n_obs} rows" if self.n_obs is not None else "")
        ]
        for v in self.nodes:
            pa = self._parents[v]
            shown = ", ".join(str(s) for s in self.states[v][:6])
            more = ", ..." if len(self.states[v]) > 6 else ""
            given = f" | {', '.join(pa)}" if pa else ""
            lines.append(f"  P({v}{given}): states [{shown}{more}]")
        for v, k in self.unseen.items():
            lines.append(
                f"  note: {k} parent configuration(s) of {v} have no data; "
                "their rows are uniform placeholders"
            )
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"BayesNet({len(self.nodes)} nodes: {', '.join(self.nodes)})"

    # ------------------------------------------------------------------ #
    #  Interventions
    # ------------------------------------------------------------------ #

    def do(
        self, interventions: Optional[Mapping[str, Any]] = None, **kw: Any
    ) -> "BayesNet":
        """
        The network after an ideal intervention.

        Each intervened node loses its parents and takes its value with
        probability one; every other kernel is unchanged.

        Parameters
        ----------
        interventions : mapping, optional
            ``{node: state}``. Keyword arguments are accepted for nodes
            whose names are identifiers: ``net.do(X="debt")``.

        Returns
        -------
        BayesNet

        Examples
        --------
        >>> import statspai as sp
        >>> net = sp.bayes_net("Z -> X; Z -> Y; X -> Y", cpts={
        ...     "Z": {0: 0.5, 1: 0.5},
        ...     "X": {0: {0: 0.9, 1: 0.1}, 1: {0: 0.1, 1: 0.9}},
        ...     "Y": {(0, 0): {0: 0.9, 1: 0.1}, (0, 1): {0: 0.5, 1: 0.5},
        ...           (1, 0): {0: 0.5, 1: 0.5}, (1, 1): {0: 0.1, 1: 0.9}}})
        >>> net.do(X=1).parents("X")
        ()
        """
        todo = dict(interventions or {}, **kw)
        self._check_nodes(todo)
        tables = dict(self._tables)
        parents = dict(self._parents)
        unseen = dict(self._unseen)
        edges = [(p, c) for c in self.nodes for p in self._parents[c] if c not in todo]
        for node, value in todo.items():
            point = np.zeros(len(self.states[node]))
            point[self._index(node, value)] = 1.0
            tables[node] = point
            parents[node] = ()
            unseen.pop(node, None)
        graph = DAG()
        for v in self.nodes:
            graph.add_node(v)
        for p, c in edges:
            graph.add_edge(p, c)
        return BayesNet(graph, self.states, tables, parents, unseen, self.n_obs)

    # ------------------------------------------------------------------ #
    #  Queries
    # ------------------------------------------------------------------ #

    def query(
        self,
        variables: Union[str, Sequence[str]],
        evidence: Optional[Mapping[str, Any]] = None,
        do: Optional[Mapping[str, Any]] = None,
    ) -> pd.DataFrame:
        """
        The joint distribution of *variables* given evidence, under an
        intervention if one is named.

        Parameters
        ----------
        variables : str or sequence of str
        evidence : mapping, optional
            ``{node: observed state}``.
        do : mapping, optional
            ``{node: state it is set to}``. Evidence is then evidence
            about the intervened system: ``P(variables | do(...),
            evidence)``. For evidence about the world as it was, with the
            intervention hypothetical, use :meth:`counterfactual`.

        Returns
        -------
        pd.DataFrame
            One row per combination of states, with a ``prob`` column.

        Examples
        --------
        >>> import statspai as sp
        >>> net = sp.bayes_net("Z -> X; Z -> Y; X -> Y", cpts={
        ...     "Z": {0: 0.5, 1: 0.5},
        ...     "X": {0: {0: 0.9, 1: 0.1}, 1: {0: 0.1, 1: 0.9}},
        ...     "Y": {(0, 0): {0: 0.9, 1: 0.1}, (0, 1): {0: 0.5, 1: 0.5},
        ...           (1, 0): {0: 0.5, 1: 0.5}, (1, 1): {0: 0.1, 1: 0.9}}})
        >>> seen = net.query("Y", evidence={"X": 1})
        >>> done = net.query("Y", do={"X": 1})
        >>> round(seen.loc[seen.Y == 1, "prob"].item(), 2)   # association
        0.86
        >>> round(done.loc[done.Y == 1, "prob"].item(), 2)   # effect
        0.7
        """
        names = [variables] if isinstance(variables, str) else list(variables)
        if not names:
            raise MethodIncompatibility("bayes_net.query: no variables were named.")
        net = self.do(do) if do else self
        net._check_nodes(names)
        evidence = dict(evidence or {})
        net._check_nodes(evidence)
        overlap = [v for v in names if v in evidence]
        if overlap:
            raise MethodIncompatibility(
                f"bayes_net.query: {overlap} are both queried and given as " "evidence."
            )
        values = net._infer(names, evidence, net._tables)
        if any(m.any() for m in net._unseen.values()):
            alt = net._infer(names, evidence, net._placeholder_tables())
            if not np.allclose(values, alt, atol=1e-10, rtol=0):
                gaps = ", ".join(f"{v} ({k})" for v, k in net.unseen.items())
                warnings.warn(
                    AssumptionWarning(
                        "bayes_net.query: the answer depends on parent "
                        "configurations that never occur in the data "
                        f"[{gaps}]. Their probabilities are placeholders, "
                        "so this query is not determined by the data "
                        "(positivity fails).",
                        recovery_hint=(
                            "Coarsen the states so that every configuration "
                            "the query needs is observed, or fit with "
                            "prior= to make the smoothing explicit."
                        ),
                        diagnostics={"unseen_parent_configurations": net.unseen},
                    ),
                    stacklevel=2,
                )
        combos = list(itertools.product(*[net.states[v] for v in names]))
        out = pd.DataFrame(combos, columns=names)
        out["prob"] = values.reshape(-1)
        return out

    def prob(
        self,
        event: Mapping[str, Any],
        evidence: Optional[Mapping[str, Any]] = None,
        do: Optional[Mapping[str, Any]] = None,
    ) -> float:
        """Probability of one joint *event*, ``{node: state, ...}``.

        Examples
        --------
        >>> import statspai as sp
        >>> net = sp.bayes_net("A -> B", cpts={
        ...     "A": {"no": 0.7, "yes": 0.3},
        ...     "B": {"no": {"lo": 0.9, "hi": 0.1}, "yes": {"lo": 0.4, "hi": 0.6}}})
        >>> round(net.prob({"A": "yes"}, evidence={"B": "hi"}), 4)
        0.72
        """
        table = self.query(list(event), evidence=evidence, do=do)
        mask = np.ones(len(table), dtype=bool)
        for v, val in event.items():
            self._index(v, val)
            mask &= (table[v] == val).to_numpy()
        return float(table.loc[mask, "prob"].sum())

    def expectation(
        self,
        variable: str,
        evidence: Optional[Mapping[str, Any]] = None,
        do: Optional[Mapping[str, Any]] = None,
        values: Optional[Mapping[Any, float]] = None,
    ) -> float:
        """
        Expected value of a node.

        Parameters
        ----------
        variable : str
        evidence, do : mapping, optional
            As in :meth:`query`.
        values : mapping, optional
            ``{state: number}``, a utility or payoff for each state.
            Without it the states themselves must be numbers.

        Examples
        --------
        >>> import statspai as sp
        >>> net = sp.bayes_net("X -> Y", cpts={
        ...     "X": {"debt": 0.5, "equity": 0.5},
        ...     "Y": {"debt": {"fail": 0.6, "success": 0.4},
        ...           "equity": {"fail": 0.66, "success": 0.34}}})
        >>> payoff = {"fail": -1000, "success": 99000}
        >>> round(net.expectation("Y", do={"X": "debt"}, values=payoff))
        39000
        """
        table = self.query(variable, evidence=evidence, do=do)
        try:
            if values is None:
                numbers = np.asarray(table[variable], dtype=float)
            else:
                numbers = np.asarray([values[s] for s in table[variable]], dtype=float)
        except (KeyError, TypeError, ValueError) as exc:
            raise MethodIncompatibility(
                f"bayes_net.expectation: the states of {variable!r} are "
                f"{self.states[variable]}; give a number for each with "
                "values={state: number}.",
            ) from exc
        return float(numbers @ table["prob"].to_numpy())

    def counterfactual(
        self,
        variables: Union[str, Sequence[str]],
        evidence: Mapping[str, Any],
        do: Mapping[str, Any],
    ) -> pd.DataFrame:
        """
        ``P(variables_{do} | evidence)``: what the variables would have
        been under the intervention, for units on which the evidence was
        observed without it.

        Abduction, action and prediction are done in one pass on a twin
        network: the descendants of the intervened nodes are duplicated,
        the copies get the intervention, and both worlds share every other
        node and so the same roots.

        This is defined only when every node that is not a root is a
        deterministic function of its parents, so that the network is a
        structural causal model with its exogenous variables as roots.
        Conditional probability tables alone do not fix a counterfactual:
        two structural models can imply the same tables and disagree
        about it.

        Parameters
        ----------
        variables : str or sequence of str
        evidence : mapping
            What was observed in the factual world.
        do : mapping
            The hypothetical intervention.

        Returns
        -------
        pd.DataFrame
            As :meth:`query`.

        Examples
        --------
        Monty Hall as a structural model. For a player who stayed and
        lost, switching would certainly have won:

        >>> import statspai as sp
        >>> doors = [1, 2, 3]
        >>> def host(car, first, coin):
        ...     free = [d for d in doors if d not in (car, first)]
        ...     return free[0] if len(free) == 1 or coin == "tails" else free[1]
        >>> def second(first, host, strategy):
        ...     if strategy == "stay":
        ...         return first
        ...     return next(d for d in doors if d not in (first, host))
        >>> net = sp.bayes_net(
        ...     "car -> host; first -> host; coin -> host; first -> second; "
        ...     "host -> second; strategy -> second; second -> win; car -> win",
        ...     cpts={"car": {d: 1 / 3 for d in doors},
        ...           "first": {d: 1 / 3 for d in doors},
        ...           "coin": {"tails": 0.5, "heads": 0.5},
        ...           "strategy": {"stay": 0.5, "switch": 0.5},
        ...           "host": host, "second": second,
        ...           "win": lambda second, car: second == car})
        >>> cf = net.counterfactual("win", evidence={"strategy": "stay", "win": False},
        ...                         do={"strategy": "switch"})
        >>> float(cf.loc[cf.win == True, "prob"].item())
        1.0
        """
        names = [variables] if isinstance(variables, str) else list(variables)
        self._check_nodes(names)
        self._check_nodes(evidence)
        self._check_nodes(do)
        noisy = [
            v
            for v in self.nodes
            if self._parents[v]
            and not np.all((self._tables[v] == 0) | (self._tables[v] == 1))
        ]
        if noisy:
            raise AssumptionViolation(
                "bayes_net.counterfactual: the kernels of "
                f"{noisy} are not deterministic, so the network does not say "
                "how a unit's value would change with its parents; a "
                "counterfactual needs a structural model.",
                recovery_hint=(
                    "Give each such node an exogenous root parent that "
                    "carries its noise and make the node a function of its "
                    "parents (cpts={node: callable}); or use sp.SCM. The "
                    "interventional query net.query(..., do=...) needs no "
                    "such assumption."
                ),
                diagnostics={"stochastic_nodes": noisy},
            )

        # Nodes downstream of the intervention differ across worlds.
        affected: set = set()
        for v in self.nodes:
            if v in do or any(p in affected for p in self._parents[v]):
                affected.add(v)
        # A queried node the intervention cannot reach keeps its factual
        # value; it gets a copy all the same, so that it can be asked about
        # while the factual one is conditioned on.
        copied = {v for v in names if v not in affected}
        suffix = "*"
        while any(f"{v}{suffix}" in self.states for v in affected | copied):
            suffix += "*"

        def twin(v: str) -> str:
            return f"{v}{suffix}" if v in affected or v in copied else v

        states = dict(self.states)
        tables = dict(self._tables)
        parents = dict(self._parents)
        for v in self.nodes:
            if v not in affected:
                continue
            states[twin(v)] = self.states[v]
            if v in do:
                point = np.zeros(len(self.states[v]))
                point[self._index(v, do[v])] = 1.0
                tables[twin(v)] = point
                parents[twin(v)] = ()
            else:
                tables[twin(v)] = self._tables[v]
                parents[twin(v)] = tuple(
                    f"{p}{suffix}" if p in affected else p for p in self._parents[v]
                )
        for v in copied:
            states[twin(v)] = self.states[v]
            tables[twin(v)] = np.eye(len(self.states[v]))
            parents[twin(v)] = (v,)
        graph = DAG()
        for v, pa in parents.items():
            graph.add_node(v)
            for p in pa:
                graph.add_edge(p, v)
        both = BayesNet(graph, states, tables, parents)
        out = both.query([twin(v) for v in names], evidence=dict(evidence))
        return out.rename(columns={twin(v): v for v in names})

    # ------------------------------------------------------------------ #
    #  Simulation
    # ------------------------------------------------------------------ #

    def simulate(
        self,
        n: int,
        seed: Optional[int] = None,
        do: Optional[Mapping[str, Any]] = None,
    ) -> pd.DataFrame:
        """
        Draw *n* rows from the network, or from the network under an
        intervention.

        Examples
        --------
        >>> import statspai as sp
        >>> net = sp.bayes_net("A -> B", cpts={
        ...     "A": {"no": 0.7, "yes": 0.3},
        ...     "B": {"no": {"lo": 0.9, "hi": 0.1}, "yes": {"lo": 0.4, "hi": 0.6}}})
        >>> net.simulate(4, seed=0, do={"A": "yes"})["A"].tolist()
        ['yes', 'yes', 'yes', 'yes']
        """
        net = self.do(do) if do else self
        rng = np.random.default_rng(seed)
        codes: Dict[str, np.ndarray] = {}
        for v in net.nodes:
            table = net._tables[v]
            if net._parents[v]:
                rows = table[tuple(codes[p] for p in net._parents[v])]
            else:
                rows = np.broadcast_to(table, (n, table.shape[-1]))
            cum = np.cumsum(rows, axis=1)
            u = rng.random(n)[:, None] * cum[:, -1:]
            codes[v] = np.minimum((u > cum).sum(axis=1), table.shape[-1] - 1)
        out = {}
        for v in net.nodes:
            labels = np.empty(len(net.states[v]), dtype=object)
            labels[:] = net.states[v]
            out[v] = pd.Series(labels[codes[v]]).infer_objects()
        return pd.DataFrame(out)

    # ------------------------------------------------------------------ #
    #  Internals
    # ------------------------------------------------------------------ #

    def _check_nodes(self, names: Any) -> None:
        missing = [v for v in names if v not in self._tables]
        if missing:
            raise ColumnNotFound(
                f"bayes_net: {missing} are not nodes of the network.",
                recovery_hint=f"Nodes are {self.nodes}.",
                diagnostics={"missing": missing},
            )

    def _index(self, node: str, value: Any) -> int:
        value = _plain(value)
        for i, s in enumerate(self.states[node]):
            if _same(s, value):
                return i
        for i, s in enumerate(self.states[node]):
            if s == value:  # 1 for True, when no state is the number itself
                return i
        raise MethodIncompatibility(
            f"bayes_net: {value!r} is not a state of {node!r}.",
            recovery_hint=f"States of {node!r} are {self.states[node]}.",
            diagnostics={"node": node, "states": [repr(s) for s in self.states[node]]},
        )

    def _placeholder_tables(self) -> Dict[str, np.ndarray]:
        """Tables with the unseen rows set to a different placeholder."""
        out = dict(self._tables)
        for v, mask in self._unseen.items():
            if not mask.any():
                continue
            table = self._tables[v].copy()
            point = np.zeros(table.shape[-1])
            point[0] = 1.0
            table[mask] = point
            out[v] = table
        return out

    def _infer(
        self,
        names: List[str],
        evidence: Mapping[str, Any],
        tables: Mapping[str, np.ndarray],
    ) -> np.ndarray:
        """Normalised array over *names* by variable elimination."""
        # Only the ancestors of the query and the evidence matter.
        relevant: set = set()
        stack = list(names) + list(evidence)
        while stack:
            v = stack.pop()
            if v not in relevant:
                relevant.add(v)
                stack.extend(self._parents[v])

        factors: List[_Factor] = []
        for v in self.nodes:
            if v not in relevant:
                continue
            scope = (*self._parents[v], v)
            arr = tables[v]
            keep: List[str] = []
            index: List[Any] = []
            for name in scope:
                if name in evidence:
                    index.append(self._index(name, evidence[name]))
                else:
                    index.append(slice(None))
                    keep.append(name)
            factors.append((tuple(keep), np.asarray(arr[tuple(index)], dtype=float)))

        hidden = [v for v in relevant if v not in names and v not in evidence]
        while hidden:
            # Eliminate the variable whose product table is smallest.
            def cost(v: str) -> int:
                scope = {u for vs, _ in factors if v in vs for u in vs}
                return int(np.prod([len(self.states[u]) for u in scope]))

            v = min(hidden, key=lambda u: (cost(u), u))
            hidden.remove(v)
            touching = [f for f in factors if v in f[0]]
            factors = [f for f in factors if v not in f[0]]
            scope = tuple(sorted({u for vs, _ in touching for u in vs} - {v}))
            factors.append((scope, _product(touching, scope)))

        result = _product(factors, tuple(names))
        total = float(result.sum())
        if not total > 0:
            raise AssumptionViolation(
                f"bayes_net: the evidence {dict(evidence)} has probability "
                "zero under the network, so nothing can be conditioned on it.",
                recovery_hint=(
                    "Check the states named in evidence=, or fit with a "
                    "positive prior= if the combination is merely unobserved."
                ),
                diagnostics={"evidence": {k: repr(v) for k, v in evidence.items()}},
            )
        return np.asarray(result / total, dtype=float)


# ====================================================================== #
#  Construction
# ====================================================================== #


def bayes_net(
    dag: Union[DAG, str],
    data: Optional[pd.DataFrame] = None,
    *,
    cpts: Optional[Mapping[str, Any]] = None,
    prior: float = 0.0,
) -> BayesNet:
    """
    A discrete causal Bayesian network on a DAG.

    Parameters
    ----------
    dag : DAG or str
        The causal graph, or a specification :func:`statspai.dag` reads.
    data : pd.DataFrame, optional
        One column per node, each treated as categorical. The table of a
        node is the frequency of its states within each configuration of
        its parents.
    cpts : mapping, optional
        Tables written by hand, for all nodes or for those that are not
        to be estimated from ``data``. For each node one of

        - a root's distribution, ``{state: probability}``;
        - ``{parent configuration: {state: probability}}``, the
          configuration being one parent's state or a tuple of states in
          the alphabetical order of the parents' names;
        - a function of the parents returning the node's state (a
          deterministic kernel, a structural assignment) or a
          ``{state: probability}`` mapping. It is called with the parents
          as keyword arguments, or with one dict of them when it takes a
          single argument that is not a parent's name.
    prior : float, default 0.0
        Pseudo-count added to every cell before normalising (a symmetric
        Dirichlet prior; ``1`` is Laplace smoothing). At zero the tables
        are maximum-likelihood frequencies.

    Returns
    -------
    BayesNet

    Notes
    -----
    A configuration of parents that never occurs in the data has no
    frequencies. At ``prior=0`` its row is a uniform placeholder,
    ``BayesNet.unseen`` counts such rows, and a query whose answer changes
    with the placeholder warns: the data do not determine it. That is the
    positivity condition of an interventional query, checked on the query
    actually asked.

    Inference is exact (variable elimination). Its cost grows with the
    size of the largest intermediate table, so it suits networks of some
    tens of nodes with a handful of states each.

    Examples
    --------
    Estimate the kernels from data, then compare seeing with doing:

    >>> import numpy as np, pandas as pd, statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> n = 20000
    >>> g = rng.random(n) < 0.5                         # guild member
    >>> e = rng.random(n) < np.where(g, 0.8, 0.2)       # high engagement
    >>> buys = rng.random(n) < 0.2 + 0.5 * g - 0.1 * e  # purchase
    >>> df = pd.DataFrame({"G": g, "E": e, "I": buys})
    >>> net = sp.bayes_net("G -> E; G -> I; E -> I", df)
    >>> seen = net.prob({"I": True}, evidence={"E": True}) - net.prob(
    ...     {"I": True}, evidence={"E": False})
    >>> done = net.prob({"I": True}, do={"E": True}) - net.prob(
    ...     {"I": True}, do={"E": False})
    >>> bool(seen > 0.15), round(done, 1)   # association is positive; effect is -0.1
    (True, -0.1)
    """
    graph = DAG(dag) if isinstance(dag, str) else dag
    if not isinstance(graph, DAG):
        raise MethodIncompatibility(
            "bayes_net: dag must be an sp.dag(...) graph or its specification."
        )
    if prior < 0:
        raise MethodIncompatibility(f"bayes_net: prior must be >= 0, got {prior}.")
    cpts = dict(cpts or {})
    nodes = sorted(graph.nodes)
    stray = [v for v in cpts if v not in nodes]
    if stray:
        raise ColumnNotFound(
            f"bayes_net: cpts names {stray}, which are not nodes of the graph.",
            recovery_hint=f"Nodes are {nodes}.",
        )
    from_data = [v for v in nodes if v not in cpts]
    if from_data and data is None:
        raise MethodIncompatibility(
            f"bayes_net: no table for {from_data}: pass data= to estimate "
            "them or give them in cpts=."
        )
    parents = {v: tuple(sorted(graph.parents(v))) for v in nodes}
    order = _topological(parents)

    if data is not None:
        hidden = [v for v in from_data if v not in data.columns]
        # A parent defined in cpts= need not be in the data, unless a
        # child's table is estimated from it.
        hidden += [
            p
            for v in from_data
            for p in parents[v]
            if p not in data.columns and p not in hidden
        ]
        if hidden:
            raise ColumnNotFound(
                f"bayes_net: nodes {sorted(set(hidden))} are not columns of the "
                "data, and a table cannot be estimated for an unobserved "
                "variable.",
                recovery_hint=(
                    "With latent variables, identify the effect first: "
                    "sp.identify(dag, treat, y).estimate(data), which uses "
                    "only the observed columns."
                ),
                diagnostics={"missing_columns": sorted(set(hidden))},
                alternative_functions=["sp.identify"],
            )

    states: Dict[str, List[Any]] = {}
    frame: Optional[pd.DataFrame] = None
    if data is not None:
        used = [v for v in nodes if v in data.columns]
        frame = data[used].dropna()
        for v in used:
            col = frame[v]
            levels = (
                list(col.cat.categories)
                if isinstance(col.dtype, pd.CategoricalDtype)
                else sorted(pd.unique(col), key=_sort_key)
            )
            if len(levels) > _MAX_STATES:
                raise MethodIncompatibility(
                    f"bayes_net: column {v!r} takes {len(levels)} distinct "
                    "values; the network is for categorical variables.",
                    recovery_hint=(
                        f"Bin it first, e.g. pd.qcut(df[{v!r}], 3), or use "
                        "an estimator for continuous outcomes (sp.aipw, "
                        "sp.g_computation) with the adjustment set from "
                        "dag.adjustment_sets()."
                    ),
                    diagnostics={"column": v, "n_values": len(levels)},
                )
            states[v] = [_plain(s) for s in levels]

    tables: Dict[str, np.ndarray] = {}
    unseen: Dict[str, np.ndarray] = {}

    # In graph order, so that a function of the parents can enumerate the
    # parents' states.
    for v in order:
        if v in cpts:
            table, levels = _table_from_spec(v, cpts[v], parents[v], states)
            states[v] = levels
            tables[v] = table
            continue
        assert frame is not None
        shape = tuple(len(states[p]) for p in parents[v]) + (len(states[v]),)
        counts = np.zeros(shape)
        idx = tuple(_codes(frame[u], states[u]) for u in (*parents[v], v))
        np.add.at(counts, idx, 1.0)
        counts += prior
        totals = counts.sum(axis=-1, keepdims=True)
        empty = totals[..., 0] == 0
        with np.errstate(invalid="ignore", divide="ignore"):
            table = np.where(totals > 0, counts / totals, 1.0 / shape[-1])
        tables[v] = table
        if empty.any():
            unseen[v] = empty

    return BayesNet(
        graph,
        states,
        tables,
        parents,
        unseen,
        n_obs=None if frame is None else int(len(frame)),
    )


# ====================================================================== #
#  Helpers
# ====================================================================== #


def _plain(value: Any) -> Any:
    """A numpy scalar as the Python value it prints as."""
    return value.item() if isinstance(value, np.generic) else value


def _same(a: Any, b: Any) -> bool:
    """Equal as states: ``True`` and ``1`` are different labels."""
    return bool(a == b) and isinstance(a, bool) == isinstance(b, bool)


def _sort_key(value: Any) -> Tuple[str, Any]:
    """Order states of mixed type without comparing across types."""
    if isinstance(value, (bool, np.bool_)):
        return ("0", bool(value))
    if isinstance(value, (int, float, np.integer, np.floating)):
        return ("1", float(value))
    return ("2", str(value))


def _codes(column: pd.Series, levels: List[Any]) -> np.ndarray:
    lookup = {s: i for i, s in enumerate(levels)}
    return np.fromiter(
        (lookup[_plain(x)] for x in column), dtype=int, count=len(column)
    )


def _topological(parents: Mapping[str, Tuple[str, ...]]) -> List[str]:
    order: List[str] = []
    placed: set = set()
    pending = sorted(parents)
    while pending:
        ready = [v for v in pending if all(p in placed for p in parents[v])]
        if not ready:  # pragma: no cover - DAG() refuses cycles
            raise MethodIncompatibility("bayes_net: the graph has a cycle.")
        for v in ready:
            order.append(v)
            placed.add(v)
        pending = [v for v in pending if v not in placed]
    return order


def _product(factors: List[_Factor], keep: Tuple[str, ...]) -> np.ndarray:
    """Multiply factors and sum out every variable not in *keep*."""
    if not factors:
        return np.ones(())
    letters: Dict[str, int] = {}
    operands: List[Any] = []
    for scope, arr in factors:
        operands.append(arr)
        operands.append([letters.setdefault(v, len(letters)) for v in scope])
    missing = [v for v in keep if v not in letters]
    if missing:  # pragma: no cover - a queried node always has a factor
        raise MethodIncompatibility(f"bayes_net: no factor mentions {missing}.")
    operands.append([letters[v] for v in keep])
    return np.asarray(np.einsum(*operands, optimize=len(factors) > 2), dtype=float)


def _normalised(node: str, probs: Mapping[Any, Any], where: str) -> Dict[Any, float]:
    try:
        out = {_plain(k): float(p) for k, p in probs.items()}
    except (TypeError, ValueError, AttributeError) as exc:
        raise MethodIncompatibility(
            f"bayes_net: the table of {node!r}{where} must map states to "
            "probabilities."
        ) from exc
    total = sum(out.values())
    if any(p < 0 for p in out.values()) or abs(total - 1.0) > 1e-8:
        raise MethodIncompatibility(
            f"bayes_net: the probabilities of {node!r}{where} are "
            f"{list(out.values())}; they must be non-negative and sum to one "
            f"(they sum to {total:.6g})."
        )
    return out


def _call(fn: Callable[..., Any], config: Dict[str, Any]) -> Any:
    try:
        params = list(inspect.signature(fn).parameters.values())
    except (TypeError, ValueError):  # builtins without a signature
        params = []
    if (
        len(params) == 1
        and params[0].name not in config
        and params[0].kind
        in (
            params[0].POSITIONAL_ONLY,
            params[0].POSITIONAL_OR_KEYWORD,
        )
    ):
        return fn(config)
    return fn(**config)


def _table_from_spec(
    node: str,
    spec: Any,
    parents: Tuple[str, ...],
    states: Dict[str, List[Any]],
) -> Tuple[np.ndarray, List[Any]]:
    """Table and state list of *node* from a hand-written specification."""
    unknown = [p for p in parents if p not in states]
    if unknown:
        raise MethodIncompatibility(
            f"bayes_net: the table of {node!r} needs the states of its "
            f"parents {unknown}, which have neither data nor a table yet."
        )
    configs = list(itertools.product(*[states[p] for p in parents]))
    rows: List[Dict[Any, float]] = []

    if callable(spec):
        if not parents:
            raise MethodIncompatibility(
                f"bayes_net: {node!r} has no parents, so its table is a "
                "distribution {state: probability}, not a function."
            )
        for cfg in configs:
            value = _call(spec, dict(zip(parents, cfg)))
            if isinstance(value, Mapping):
                rows.append(_normalised(node, value, f" at {cfg}"))
            else:
                rows.append({_plain(value): 1.0})
    elif not parents:
        if not isinstance(spec, Mapping):
            raise MethodIncompatibility(
                f"bayes_net: the table of root {node!r} must be "
                "{state: probability}."
            )
        rows.append(_normalised(node, spec, ""))
    else:
        if not isinstance(spec, Mapping):
            raise MethodIncompatibility(
                f"bayes_net: the table of {node!r} must map each "
                f"configuration of {list(parents)} to {{state: probability}}, "
                "or be a function of the parents."
            )
        keyed = {
            (tuple(_plain(x) for x in k) if isinstance(k, tuple) else (_plain(k),)): v
            for k, v in spec.items()
        }
        absent = [cfg for cfg in configs if tuple(cfg) not in keyed]
        if absent:
            raise MethodIncompatibility(
                f"bayes_net: the table of {node!r} has no row for "
                f"{list(parents)} = {absent[:4]}"
                + (" ..." if len(absent) > 4 else "")
                + f". Keys are tuples in the order {list(parents)}.",
                diagnostics={"parents": list(parents), "n_missing": len(absent)},
            )
        extra = [k for k in keyed if k not in {tuple(c) for c in configs}]
        if extra:
            raise MethodIncompatibility(
                f"bayes_net: the table of {node!r} has rows for {extra[:4]}, "
                f"which are not configurations of {list(parents)} "
                f"(states: { {p: states[p] for p in parents} })."
            )
        for cfg in configs:
            rows.append(_normalised(node, keyed[tuple(cfg)], f" at {cfg}"))

    levels: List[Any] = list(states.get(node, []))  # the data's order first
    for row in rows:
        for s in row:
            if not any(_same(s, t) for t in levels):
                levels.append(s)
    table = np.zeros((len(rows), len(levels)))
    for i, row in enumerate(rows):
        for s, p in row.items():
            table[i, next(j for j, t in enumerate(levels) if _same(s, t))] += p
    shape = tuple(len(states[p]) for p in parents) + (len(levels),)
    return table.reshape(shape), levels
