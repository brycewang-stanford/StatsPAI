"""
Shpitser-Pearl ID algorithm for non-parametric identification of
interventional distributions P(Y | do(X)) in semi-Markovian models.

Reference
---------
Shpitser, I. & Pearl, J. (2006). "Identification of Joint Interventional
Distributions in Recursive Semi-Markovian Causal Models." AAAI.

Tian, J. & Pearl, J. (2002). "A General Identification Condition for
Causal Effects."
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from typing import Any, Union

from .._aliases import accepts_aliases
from .._result_serialize import ResultProtocolMixin

NodeInput = Union[str, Iterable[str]]
NodeSet = set[str]


@dataclass
class IdentificationResult(ResultProtocolMixin):
    """Outcome of an identification query.

    Attributes
    ----------
    identifiable : bool
        True iff P(Y | do(X)) is identifiable from the observed distribution.
    estimand : str
        Do-free formula when identifiable; a structured hedge otherwise.
    c_components : list[set[str]]
        The c-components of the ancestral semi-Markovian graph G[An(Y)].
    hedge : tuple[frozenset, frozenset] | None
        Witness C-forest pair (F, F') that proves non-identifiability.
    explanation : str
        Human-readable proof / refutation.

    Examples
    --------
    >>> import statspai as sp
    >>> g = sp.dag("Z -> X; Z -> Y; X -> Y")
    >>> res = sp.identify(g, treatment="X", outcome="Y")
    >>> isinstance(res, sp.IdentificationResult)
    True
    >>> bool(res.identifiable)
    True
    >>> "Y" in res.estimand
    True
    """

    identifiable: bool
    estimand: str
    c_components: list[NodeSet]
    hedge: tuple[frozenset[str], frozenset[str]] | None
    explanation: str

    def __repr__(self) -> str:
        status = "identifiable" if self.identifiable else "NOT identifiable"
        return f"IdentificationResult({status}: {self.estimand})"

    def summary(self) -> str:
        """Return a formatted multi-line summary of the ID query.

        Every StatsPAI result object exposes ``.summary()`` (CLAUDE.md §3.3);
        this one renders the identifiability verdict, the do-free estimand or
        the hedge witnessing non-identifiability, and the c-components of the
        ancestral semi-Markovian graph.

        Examples
        --------
        >>> import statspai as sp
        >>> g = sp.dag("Z -> X; Z -> Y; X -> Y")
        >>> res = sp.identify(g, treatment="X", outcome="Y")
        >>> "IDENTIFIABLE" in res.summary()
        True
        """
        width = 70
        verdict = "IDENTIFIABLE" if self.identifiable else "NOT IDENTIFIABLE"
        lines = [
            "=" * width,
            "  Causal Identification (Shpitser-Pearl ID)",
            "=" * width,
            "",
            f"  Verdict:   {verdict}",
        ]
        label = "Estimand" if self.identifiable else "Obstruction"
        lines.append(f"  {label}:  {self.estimand}")
        lines.append("")

        if self.c_components:
            lines.append("-" * width)
            lines.append("  C-components of G[An(Y)]")
            lines.append("-" * width)
            for comp in self.c_components:
                lines.append("    {" + ", ".join(sorted(comp)) + "}")
            lines.append("")

        if self.hedge is not None:
            f_set, f_prime = self.hedge
            lines.append("-" * width)
            lines.append("  Hedge witness (proves non-identifiability)")
            lines.append("-" * width)
            lines.append("    F  = {" + ", ".join(sorted(f_set)) + "}")
            lines.append("    F' = {" + ", ".join(sorted(f_prime)) + "}")
            lines.append("")

        if self.explanation:
            lines.append("-" * width)
            lines.append("  Explanation")
            lines.append("-" * width)
            lines.append(f"    {self.explanation}")
            lines.append("")

        lines.append("=" * width)
        return "\n".join(lines)

    def estimate(
        self,
        data: Any,
        *,
        n_boot: int = 0,
        alpha: float = 0.05,
        seed: int | None = None,
    ) -> Any:
        """
        Evaluate the estimand on categorical data: ``P(Y | do(X))``.

        Each probability in the formula is replaced by the matching
        frequency in the data and the sums are carried out. Only the
        observed variables are used, so this works where a latent
        confounder rules out fitting the whole graph (the front door, for
        one).

        Parameters
        ----------
        data : pd.DataFrame
            One column for every variable in the estimand, each treated
            as categorical.
        n_boot : int, default 0
            Nonparametric bootstrap replications. When positive, ``se``,
            ``ci_lower`` and ``ci_upper`` (percentile) are added.
        alpha : float, default 0.05
            Level of the bootstrap interval.
        seed : int, optional
            Seed for the bootstrap.

        Returns
        -------
        pd.DataFrame
            One row per combination of treatment and outcome values, with
            ``prob``, the estimate of ``P(outcome | do(treatment))``.

        Notes
        -----
        A term such as ``P(y | x', m)`` is undefined where ``(x', m)``
        never occurs. When the answer for a treatment value depends on
        such a term, the data do not determine it: those rows are
        returned as missing and an
        :class:`~statspai.exceptions.AssumptionWarning` says which. This
        is the positivity condition of the estimand, checked on the data.

        When the formula keeps a variable free that is neither treatment
        nor outcome (the algorithm may intervene on variables that cannot
        affect the outcome, and the result does not depend on their
        value in the population), it is averaged over that variable's
        observed distribution.

        Examples
        --------
        >>> import numpy as np, pandas as pd, statspai as sp
        >>> rng = np.random.default_rng(0)
        >>> n = 20000
        >>> u = rng.random(n) < 0.5                         # unobserved
        >>> x = rng.random(n) < np.where(u, 0.8, 0.2)
        >>> m = rng.random(n) < np.where(x, 0.9, 0.2)
        >>> y = rng.random(n) < 0.1 + 0.5 * m + 0.3 * u
        >>> df = pd.DataFrame({"X": x, "M": m, "Y": y})
        >>> g = sp.dag("X -> M -> Y; U -> X; U -> Y", latent=["U"])
        >>> res = sp.identify(g, "X", "Y")
        >>> est = res.estimate(df)
        >>> p1 = est.query("X == True and Y == True")["prob"].item()
        >>> p0 = est.query("X == False and Y == True")["prob"].item()
        >>> round(p1 - p0, 2)        # true effect 0.5 * (0.9 - 0.2)
        0.35
        """
        import itertools
        import warnings

        import numpy as np
        import pandas as pd

        from ..exceptions import (
            AssumptionWarning,
            ColumnNotFound,
            IdentificationFailure,
            MethodIncompatibility,
        )

        expr = getattr(self, "_expression", None)
        if not self.identifiable or expr is None:
            raise IdentificationFailure(
                "identify(...).estimate: the query is not identified, so "
                "there is no formula to evaluate.",
                recovery_hint=(
                    "Bound the effect instead (sp.manski_bounds), or look "
                    "for an instrument or a proxy."
                ),
            )
        X: tuple[str, ...] = getattr(self, "_treatment")
        Y: tuple[str, ...] = getattr(self, "_outcome")
        used = sorted(_all_vars(expr) | set(X) | set(Y))
        missing = [v for v in used if v not in data.columns]
        if missing:
            raise ColumnNotFound(
                f"identify(...).estimate: the estimand uses {missing}, which "
                "are not columns of the data.",
                diagnostics={"missing_columns": missing},
            )
        frame = data[used].dropna()
        states: dict[str, list[Any]] = {}
        codes: dict[str, Any] = {}
        for v in used:
            col = frame[v]
            if isinstance(col.dtype, pd.CategoricalDtype):
                col = col.cat.remove_unused_categories()
                levels = list(col.cat.categories)
                code = col.cat.codes.to_numpy()
            else:
                code, uniques = pd.factorize(col, sort=True)
                levels = list(uniques)
            if len(levels) > 50:
                raise MethodIncompatibility(
                    f"identify(...).estimate: column {v!r} takes "
                    f"{len(levels)} distinct values; the plug-in is for "
                    "categorical variables.",
                    recovery_hint=(
                        f"Bin it first, e.g. pd.qcut(df[{v!r}], 3). For a "
                        "continuous outcome under a backdoor or front-door "
                        "estimand use sp.aipw / sp.front_door."
                    ),
                    diagnostics={"column": v, "n_values": len(levels)},
                )
            states[v] = [s.item() if isinstance(s, np.generic) else s for s in levels]
            codes[v] = np.asarray(code)
        emp = _Empirical(codes, states)
        target = (*X, *Y)

        def value(e: _Empirical, uniform: bool) -> Any:
            scope, arr = _array(expr, e, uniform)
            parts = [(scope, arr)]
            extra = tuple(v for v in scope if v not in target)
            if extra:
                parts.append((extra, e.marginal(extra)))
            # A treatment the outcome does not respond to is not in the
            # formula: P(y | do(x)) = P(y), the same for every x.
            for v in target:
                if v not in scope:
                    parts.append(((v,), np.ones(len(e.states[v]))))
            return _contract(parts, target)

        point = value(emp, False)
        other = value(emp, True)
        shaky = np.abs(point - other) > 1e-10
        if shaky.any():
            # Any outcome cell that moves marks the whole treatment value.
            bad = shaky.reshape(shaky.shape[: len(X)] + (-1,)).any(axis=-1)
            point = np.where(bad.reshape(bad.shape + (1,) * len(Y)), np.nan, point)
            bad_x = [
                {v: states[v][i] for v, i in zip(X, idx)}
                for idx in zip(*np.nonzero(bad))
            ]
            warnings.warn(
                AssumptionWarning(
                    "identify(...).estimate: for treatment values "
                    f"{bad_x} the estimand needs a conditional probability "
                    "whose conditioning event never occurs in the data, so "
                    "the data do not determine it (positivity fails). Those "
                    "rows are returned as missing.",
                    recovery_hint=(
                        "Coarsen the variables involved so that every "
                        "combination the formula sums over is observed."
                    ),
                    diagnostics={"treatment_values": bad_x},
                ),
                stacklevel=2,
            )

        rows = list(itertools.product(*[states[v] for v in target]))
        out = pd.DataFrame(rows, columns=list(target))
        out["prob"] = point.reshape(-1)
        if n_boot > 0:
            rng = np.random.default_rng(seed)
            draws = np.empty((n_boot, point.size))
            for b in range(n_boot):
                again = emp.resample(rng.integers(0, emp.n, emp.n))
                draws[b] = value(again, False).reshape(-1)
            out["se"] = draws.std(axis=0, ddof=1)
            out["ci_lower"] = np.quantile(draws, alpha / 2, axis=0)
            out["ci_upper"] = np.quantile(draws, 1 - alpha / 2, axis=0)
            out.loc[out["prob"].isna(), ["se", "ci_lower", "ci_upper"]] = np.nan
        return out


@accepts_aliases(treat="treatment", y="outcome")
def identify(
    dag: Any,
    treatment: NodeInput,
    outcome: NodeInput,
) -> IdentificationResult:
    """Run Shpitser-Pearl ID algorithm on ``dag``.

    Parameters
    ----------
    dag : DAG
        ``statspai.dag.DAG`` instance, possibly with latent nodes
        ``_L_*`` (representing bidirected edges) or named latents declared
        with ``sp.dag(..., latent=[...])``; the latter are projected out
        with :meth:`DAG.latent_projection` first.
    treatment : str | Iterable[str]
        Set of variables X being intervened on.
    outcome : str | Iterable[str]
        Set of outcome variables Y.

    Returns
    -------
    IdentificationResult

    Examples
    --------
    >>> import statspai as sp
    >>> # Backdoor confounder -- P(Y | do(X)) is identifiable
    >>> g = sp.dag("Z -> X; Z -> Y; X -> Y")
    >>> res = sp.identify(g, treatment="X", outcome="Y")
    >>> bool(res.identifiable)
    True
    >>> # Bow arc (X <-> Y latent confounding) -- NOT identifiable
    >>> g2 = sp.dag("X -> Y; X <-> Y")
    >>> res2 = sp.identify(g2, treatment="X", outcome="Y")
    >>> bool(res2.identifiable)
    False
    >>> "hedge" in res2.estimand
    True
    """
    X = frozenset({treatment} if isinstance(treatment, str) else set(treatment))
    Y = frozenset({outcome} if isinstance(outcome, str) else set(outcome))

    # Declared latents may have parents and more than two children; the
    # algorithm below is written for bidirected edges, so project first.
    if getattr(dag, "_latent", None):
        dag = dag.latent_projection()

    V = frozenset(dag._nodes)
    if not X.issubset(V) or not Y.issubset(V):
        missing = (X | Y) - V
        raise KeyError(f"Variables not in DAG: {missing}")

    observable = frozenset(v for v in V if not _is_latent(v))
    X &= observable
    Y &= observable

    ccs = _c_components(dag, observable)

    try:
        expr = _simplify(_ID(Y, X, _subgraph(dag, observable), observable), dag)
        result = IdentificationResult(
            identifiable=True,
            estimand=_render(expr),
            c_components=[set(c) for c in ccs],
            hedge=None,
            explanation=(
                f"Query P({_fmt(Y)} | do({_fmt(X)})) is identified via "
                f"repeated c-component factorization; the resulting "
                f"expression involves only observed joint distributions."
            ),
        )
        # The expression tree behind the string, for numeric evaluation.
        object.__setattr__(result, "_expression", expr)
        object.__setattr__(result, "_treatment", _tup(X))
        object.__setattr__(result, "_outcome", _tup(Y))
        return result
    except _NotIdentifiable as exc:
        return IdentificationResult(
            identifiable=False,
            estimand=f"hedge({_fmt(exc.F)}, {_fmt(exc.F_prime)})",
            c_components=[set(c) for c in ccs],
            hedge=(exc.F, exc.F_prime),
            explanation=(
                f"P({_fmt(Y)} | do({_fmt(X)})) is NOT identifiable. "
                f"Witness hedge: F={sorted(exc.F)}, F'={sorted(exc.F_prime)}."
            ),
        )


class _NotIdentifiable(Exception):
    def __init__(self, F: Iterable[str], F_prime: Iterable[str]) -> None:
        self.F = frozenset(F)
        self.F_prime = frozenset(F_prime)


# --------------------------------------------------------------------------- #
#  Symbolic distributions
# --------------------------------------------------------------------------- #
#
# The recursion manipulates a distribution, not only a graph: line 7 replaces
# P by a product of conditionals of P and recurses on a sub-graph, and every
# later line has to marginalise or condition *that* product. The expressions
# below carry it. correctness fix (2026-10): the recursion used to pass the
# graph alone and write "P(...)" for whatever distribution was current, so a
# query that reached line 7 came back with the wrong formula; the front-door
# graph returned sum_M [P(Y) * P(M | X)], which is P(Y).


@dataclass(frozen=True)
class _Prob:
    """Observational ``P(vars | given)``."""

    vars: tuple[str, ...]
    given: tuple[str, ...] = ()


@dataclass(frozen=True)
class _Prod:
    terms: tuple[Any, ...]


@dataclass(frozen=True)
class _Sum:
    over: tuple[str, ...]
    body: Any


@dataclass(frozen=True)
class _Frac:
    num: Any
    den: Any


def _tup(s: Iterable[str]) -> tuple[str, ...]:
    return tuple(sorted(s))


def _marginal(dist: Any, V: frozenset[str], keep: frozenset[str]) -> Any:
    """``sum_{V minus keep} dist`` for a distribution over ``V``."""
    drop = V - keep
    if not drop:
        return dist
    if isinstance(dist, _Prob) and not dist.given:
        return _Prob(_tup(keep))
    return _Sum(_tup(drop), dist)


def _conditional(dist: Any, V: frozenset[str], v: str, prior: Iterable[str]) -> Any:
    """``dist(v | prior)`` for a distribution over ``V``."""
    prior_set = frozenset(prior)
    if isinstance(dist, _Prob) and not dist.given:
        return _Prob((v,), _tup(prior_set))
    num = _marginal(dist, V, prior_set | {v})
    if not prior_set:
        return num
    return _Frac(num, _marginal(dist, V, prior_set))


def _product(terms: list[Any]) -> Any:
    flat: list[Any] = []
    for t in terms:
        flat.extend(t.terms if isinstance(t, _Prod) else [t])
    return flat[0] if len(flat) == 1 else _Prod(tuple(flat))


def _render(
    expr: Any,
    names: dict[str, str] | None = None,
    reserved: set[str] | None = None,
) -> str:
    """Write ``expr`` out, priming a summation variable that is already in use.

    The front-door formula sums over the treatment inside a term that also
    has the treatment free; the inner one is written ``X'``. ``reserved``
    holds the names in use around ``expr``: the free variables of the whole
    estimand and the variables bound by enclosing sums.
    """
    names = dict(names or {})
    if reserved is None:
        reserved = set(_free_vars(expr))

    def nm(v: str) -> str:
        return names.get(v, v)

    if isinstance(expr, _Prob):
        head = ", ".join(nm(v) for v in expr.vars)
        if expr.given:
            return f"P({head} | {', '.join(nm(v) for v in expr.given)})"
        return f"P({head})"
    if isinstance(expr, _Prod):
        # Sorted, so the printed formula does not depend on set iteration.
        return " * ".join(sorted(_render(t, names, reserved) for t in expr.terms))
    if isinstance(expr, _Frac):
        num = _render(expr.num, names, reserved)
        den = _render(expr.den, names, reserved)
        return f"[{num}] / [{den}]"
    if isinstance(expr, _Sum):
        inner = dict(names)
        taken = set(reserved)
        for v in expr.over:
            label = v
            while label in taken:
                label += "'"
            inner[v] = label
            taken.add(label)
        over = ", ".join(sorted(inner[v] for v in expr.over))
        return f"sum_{{{over}}} [{_render(expr.body, inner, taken)}]"
    raise TypeError(f"unknown expression node {type(expr).__name__}")


def _free_vars(expr: Any) -> frozenset[str]:
    if isinstance(expr, _Prob):
        return frozenset(expr.vars) | frozenset(expr.given)
    if isinstance(expr, _Prod):
        out: frozenset[str] = frozenset()
        for t in expr.terms:
            out |= _free_vars(t)
        return out
    if isinstance(expr, _Frac):
        return _free_vars(expr.num) | _free_vars(expr.den)
    if isinstance(expr, _Sum):
        return _free_vars(expr.body) - frozenset(expr.over)
    raise TypeError(f"unknown expression node {type(expr).__name__}")


def _all_vars(expr: Any) -> set[str]:
    """Every variable an expression mentions, bound or free."""
    if isinstance(expr, _Prob):
        return set(expr.vars) | set(expr.given)
    if isinstance(expr, _Prod):
        return set().union(*[_all_vars(t) for t in expr.terms])
    if isinstance(expr, _Frac):
        return _all_vars(expr.num) | _all_vars(expr.den)
    if isinstance(expr, _Sum):
        return _all_vars(expr.body) | set(expr.over)
    raise TypeError(f"unknown expression node {type(expr).__name__}")


def _evaluate(expr: Any, joint: Any, values: dict[str, int]) -> float:
    """Numeric value of ``expr`` under an observational joint.

    ``joint`` maps a tuple of ``(variable, value)`` pairs, sorted by
    variable, to its probability, for binary variables. Used by the tests
    to check an estimand against the interventional distribution of a
    structural model; not part of the public API.
    """
    import itertools

    variables = sorted({v for key in joint for v, _ in key})

    def prob(fixed: dict[str, int]) -> float:
        total = 0.0
        for key, pr in joint.items():
            if all(dict(key)[v] == val for v, val in fixed.items()):
                total += pr
        return total

    if isinstance(expr, _Prob):
        given = {v: values[v] for v in expr.given}
        both = dict(given, **{v: values[v] for v in expr.vars})
        den = prob(given) if given else 1.0
        return prob(both) / den if den > 0 else 0.0
    if isinstance(expr, _Prod):
        out = 1.0
        for t in expr.terms:
            out *= _evaluate(t, joint, values)
        return out
    if isinstance(expr, _Frac):
        den = _evaluate(expr.den, joint, values)
        return _evaluate(expr.num, joint, values) / den if den > 0 else 0.0
    if isinstance(expr, _Sum):
        total = 0.0
        for combo in itertools.product((0, 1), repeat=len(expr.over)):
            total += _evaluate(
                expr.body, joint, dict(values, **dict(zip(expr.over, combo)))
            )
        return total
    raise TypeError(f"unknown expression node {type(expr).__name__} ({variables})")


def _simplify(expr: Any, dag: Any) -> Any:
    """Drop conditioning variables the graph makes irrelevant.

    Every ``_Prob`` leaf is a conditional of the observational
    distribution, so ``P(v | a, b) = P(v | b)`` whenever ``v`` and ``a``
    are d-separated given ``b`` in the full graph. The algorithm
    conditions each factor on all its predecessors in a topological
    order; most of them are usually not parents.
    """
    if isinstance(expr, _Prob):
        given = list(expr.given)
        changed = True
        while changed:
            changed = False
            for g in sorted(given):
                rest = set(given) - {g}
                if all(dag.d_separated(v, g, rest) for v in expr.vars):
                    given.remove(g)
                    changed = True
                    break
        return _Prob(expr.vars, tuple(sorted(given)))
    if isinstance(expr, _Prod):
        return _Prod(tuple(_simplify(t, dag) for t in expr.terms))
    if isinstance(expr, _Sum):
        return _Sum(expr.over, _simplify(expr.body, dag))
    if isinstance(expr, _Frac):
        return _Frac(_simplify(expr.num, dag), _simplify(expr.den, dag))
    raise TypeError(f"unknown expression node {type(expr).__name__}")


class _Empirical:
    """Marginal frequency tables of a categorical data frame."""

    def __init__(self, codes: dict[str, Any], states: dict[str, list[Any]]) -> None:
        self.codes = codes
        self.states = states
        self.n = len(next(iter(codes.values())))

    def marginal(self, names: tuple[str, ...]) -> Any:
        import numpy as np

        shape = tuple(len(self.states[v]) for v in names)
        out = np.zeros(shape)
        if names:
            np.add.at(out, tuple(self.codes[v] for v in names), 1.0)
        else:
            out += self.n
        return out / self.n

    def resample(self, idx: Any) -> "_Empirical":
        return _Empirical({v: c[idx] for v, c in self.codes.items()}, self.states)


def _array(expr: Any, emp: _Empirical, uniform: bool) -> tuple[tuple[str, ...], Any]:
    """Value of ``expr`` as an array over its free variables.

    A conditional probability whose conditioning event never occurs is
    undefined. It is filled with zero, or with a uniform distribution when
    ``uniform``; a result that differs between the two depends on it.
    """
    import numpy as np

    if isinstance(expr, _Prob):
        scope = (*expr.given, *expr.vars)
        joint = emp.marginal(scope)
        if not expr.given:
            return scope, joint
        k = len(expr.given)
        den = emp.marginal(expr.given).reshape(joint.shape[:k] + (1,) * len(expr.vars))
        fill = 1.0 / float(np.prod(joint.shape[k:])) if uniform else 0.0
        with np.errstate(invalid="ignore", divide="ignore"):
            return scope, np.where(den > 0, joint / den, fill)
    if isinstance(expr, _Prod):
        parts = [_array(t, emp, uniform) for t in expr.terms]
        scope = tuple(sorted({v for vs, _ in parts for v in vs}))
        return scope, _contract(parts, scope)
    if isinstance(expr, _Sum):
        scope, arr = _array(expr.body, emp, uniform)
        keep = tuple(v for v in scope if v not in expr.over)
        out = _contract([(scope, arr)], keep)
        for v in expr.over:  # a body constant in v sums to card(v) times it
            if v not in scope:
                out = out * len(emp.states[v])
        return keep, out
    if isinstance(expr, _Frac):
        ns, num = _array(expr.num, emp, uniform)
        ds, den = _array(expr.den, emp, uniform)
        scope = tuple(sorted(set(ns) | set(ds)))
        num_b = _lay_out(ns, num, scope)
        den_b = _lay_out(ds, den, scope)
        with np.errstate(invalid="ignore", divide="ignore"):
            return scope, np.where(den_b > 0, num_b / den_b, 0.0)
    raise TypeError(f"unknown expression node {type(expr).__name__}")


def _lay_out(scope: tuple[str, ...], arr: Any, target: tuple[str, ...]) -> Any:
    """``arr`` on the axes of ``target``, size one where it has no variable."""
    import numpy as np

    order = [v for v in target if v in scope]
    arr = np.transpose(arr, [scope.index(v) for v in order])
    return arr.reshape([arr.shape[order.index(v)] if v in order else 1 for v in target])


def _contract(parts: list[tuple[tuple[str, ...], Any]], keep: tuple[str, ...]) -> Any:
    """Product of factors, summed down to the variables in ``keep``."""
    import numpy as np

    letters: dict[str, int] = {}
    operands: list[Any] = []
    for scope, arr in parts:
        operands.append(arr)
        operands.append([letters.setdefault(v, len(letters)) for v in scope])
    operands.append([letters[v] for v in keep])
    return np.einsum(*operands)


# --------------------------------------------------------------------------- #
#  Core recursion (Shpitser-Pearl Alg. 1)
# --------------------------------------------------------------------------- #


def _ID(
    Y: frozenset[str],
    X: frozenset[str],
    dag: Any,
    V: frozenset[str],
    dist: Any = None,
) -> Any:
    """Non-parametric identification of ``P(Y | do(X))`` over observed ``V``.

    ``dist`` is the current distribution over ``V`` as a symbolic
    expression (the observational joint when omitted). Returns the
    estimand as an expression; :func:`_render` writes it out.
    """
    if dist is None:
        dist = _Prob(_tup(V))

    # Line 1: no intervention left -> marginalise.
    if not X:
        return _marginal(dist, V, Y)

    # Line 2: restrict to the ancestors of Y.
    An_Y = _ancestors_in(dag, Y, V)
    if An_Y != V:
        return _ID(Y, X & An_Y, _subgraph(dag, An_Y), An_Y, _marginal(dist, V, An_Y))

    # Line 3: intervene also on what cannot reach Y once X is fixed.
    W = (V - X) - _ancestors_in(dag, Y, V - X)
    if W:
        return _ID(Y, X | W, dag, V, dist)

    # Line 4: c-component decomposition of G minus X.
    G_minus_X = _subgraph_without_nodes(dag, X)
    ccs = _c_components(G_minus_X, V - X)
    if len(ccs) > 1:
        parts = [_ID(frozenset(S), V - frozenset(S), dag, V, dist) for S in ccs]
        body = _product(parts)
        extra = V - Y - X
        return _Sum(_tup(extra), body) if extra else body

    S = frozenset(ccs[0])
    # Line 5: the whole graph is one c-component -> a hedge.
    CV = [frozenset(c) for c in _c_components(dag, V)]
    if len(CV) == 1 and CV[0] == V:
        raise _NotIdentifiable(V, S)

    order = _topo_order(dag, V)

    # Line 6: S is a c-component of the whole graph.
    if S in CV:
        body = _product(
            [_conditional(dist, V, v, _prior(order, v)) for v in order if v in S]
        )
        extra = S - Y
        return _Sum(_tup(extra), body) if extra else body

    # Line 7: S sits strictly inside a c-component S' of the whole graph.
    # Recurse on G[S'] with the distribution
    #   prod_{v in S'} dist(v | predecessors of v in the order of V).
    for Sp in CV:
        if S < Sp:
            new_dist = _product(
                [_conditional(dist, V, v, _prior(order, v)) for v in order if v in Sp]
            )
            return _ID(Y, X & Sp, _subgraph(dag, Sp), Sp, new_dist)

    raise _NotIdentifiable(V, S)


# --------------------------------------------------------------------------- #
#  Graph utilities (latent-aware)
# --------------------------------------------------------------------------- #


def _is_latent(node: str) -> bool:
    # Only the nodes a bidirected edge or the latent projection creates.
    # A ``U_`` prefix used to count as well, which nothing documented: an
    # observed confounder named ``U_rate`` was offered as an adjustment
    # variable by ``adjustment_sets`` and treated as unmeasured here.
    return node.startswith("_L_")


def _observed_parents(dag: Any, node: str) -> NodeSet:
    out: NodeSet = set()
    for p, children in dag._edges.items():
        if node in children and not _is_latent(p):
            out.add(p)
    return out


def _bidirected_neighbors(dag: Any, node: str) -> NodeSet:
    """Return observed nodes sharing a latent parent with ``node``."""
    latent_parents = [p for p, ch in dag._edges.items() if node in ch and _is_latent(p)]
    out: NodeSet = set()
    for L in latent_parents:
        out |= {v for v in dag._edges.get(L, set()) if not _is_latent(v)}
    out.discard(node)
    return out


def _c_components(dag: Any, observed: Iterable[str]) -> list[NodeSet]:
    """Bidirected-connected components restricted to ``observed``."""
    observed = set(observed)
    unvisited = set(observed)
    components: list[NodeSet] = []
    while unvisited:
        seed = min(unvisited)
        stack = [seed]
        comp: NodeSet = set()
        while stack:
            v = stack.pop()
            if v in comp:
                continue
            comp.add(v)
            neigh = _bidirected_neighbors(dag, v) & observed
            stack.extend(neigh - comp)
        components.append(comp)
        unvisited -= comp
    return components


def _ancestors_in(
    dag: Any,
    nodes: Iterable[str],
    universe: Iterable[str],
) -> frozenset[str]:
    universe = set(universe)
    result: NodeSet = set()
    stack = list(nodes)
    while stack:
        v = stack.pop()
        if v in result or v not in universe:
            continue
        result.add(v)
        stack.extend(_observed_parents(dag, v) & universe)
    return frozenset(result)


def _subgraph(dag: Any, V: Iterable[str]) -> Any:
    """Return a fresh DAG restricted to nodes in V (plus relevant latents)."""
    from .graph import DAG as _DAG

    sub = _DAG()
    keep_obs = set(V)
    for v in keep_obs:
        sub.add_node(v)
    for p, ch in dag._edges.items():
        if _is_latent(p):
            preserved = ch & keep_obs
            if len(preserved) >= 2:
                sub._nodes.add(p)
                sub._edges.setdefault(p, set()).update(preserved)
        elif p in keep_obs:
            for c in ch:
                if c in keep_obs and not _is_latent(c):
                    sub.add_edge(p, c)
    return sub


def _subgraph_without_nodes(dag: Any, remove: Iterable[str]) -> Any:
    V = set(dag._nodes) - set(remove)
    V = {v for v in V if not _is_latent(v)}
    return _subgraph(dag, V)


def _topo_order(dag: Any, V: Iterable[str]) -> list[str]:
    V = set(V)
    indeg = {v: 0 for v in sorted(V)}
    for p, ch in dag._edges.items():
        if _is_latent(p):
            continue
        for c in ch:
            if p in V and c in V:
                indeg[c] += 1
    stack = [v for v, d in indeg.items() if d == 0]
    order: list[str] = []
    while stack:
        v = stack.pop(0)
        order.append(v)
        for c in sorted(dag._edges.get(v, set())):
            if c in V and not _is_latent(c):
                indeg[c] -= 1
                if indeg[c] == 0:
                    stack.append(c)
    if len(order) != len(V):
        order.extend(sorted(V - set(order)))
    return order


def _prior(order: list[str], v: str) -> list[str]:
    return order[: order.index(v)] if v in order else []


def _fmt(s: Iterable[str]) -> str:
    return ", ".join(sorted(s))
