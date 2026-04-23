"""LLM-DAG closed loop: iterate prior → constrained-PC → CI-validate.

The 2024-2026 line of work on *LLM-assisted causal discovery* (Kıcıman
et al. 2023; Long et al. 2023; Ban et al. 2024; Vashishtha et al. 2024)
treats a large language model as a noisy oracle over candidate edges.
The single-shot recipe — let the LLM propose a graph, accept it as is —
is brittle: LLM hallucinations leak directly into downstream
identification.

This module implements the closed loop:

    1. **Propose**.  Ask the oracle (or :func:`llm_dag_propose`) for
       directed edges with confidence scores.
    2. **Split**.  Partition the proposal into ``required`` (high
       confidence, asserted edges) and ``forbidden`` (high confidence,
       asserted *non-edges*).  Edges with weak / no signal are passed
       through as candidates only.
    3. **Constrained discovery**.  Run :func:`pc_algorithm` with the
       background-knowledge constraints injected.
    4. **Validate**.  For every required edge, run a partial-correlation
       CI test conditioning on the parents recovered by PC.  If the CI
       test rejects the edge at level ``alpha``, demote it (drop from
       ``required``) and queue the loop for another round.
    5. **Iterate**.  Repeat up to ``max_iter`` times or until no edges
       are demoted.

The result is a DAG with provenance — every retained edge carries both
an LLM confidence and a data-driven CI-test p-value.

Notes on design
---------------
- The module is pure orchestration.  No network calls.  The user owns
  the LLM client.
- ``oracle`` is a callable ``f(variables, descriptions) -> list[(from,
  to, confidence)]`` where confidence is in ``[0, 1]``.  When the
  callable returns 2-tuples, confidence defaults to 1.0.
- The validation pass is conservative: an edge is demoted only if the
  partial correlation given its parents has p-value > ``alpha``.  This
  matches the PC skeleton phase logic.

References
----------
Kıcıman, E., Ness, R., Sharma, A., & Tan, C. (2023).  "Causal reasoning
and large language models."  arXiv:2305.00050.

Long, S., Piché, A., Zantedeschi, V., Schuster, T., & Drouin, A.
(2023).  "Causal discovery with language models as imperfect experts."
arXiv:2307.02390.

Jiralerspong, T., Chen, X., More, Y., Shah, V., & Bengio, Y. (2024).
"Efficient Causal Graph Discovery Using Large Language Models."
arXiv:2402.01207. [@jiralerspong2024efficient]
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats as sp_stats


__all__ = [
    "llm_dag_constrained",
    "llm_dag_validate",
    "LLMConstrainedDAGResult",
    "DAGValidationResult",
]


# --------------------------------------------------------------------- #
#  Result classes
# --------------------------------------------------------------------- #


@dataclass
class LLMConstrainedDAGResult:
    """Output of :func:`llm_dag_constrained`.

    Attributes
    ----------
    final_edges : list of (str, str)
        Directed edges in the final CPDAG.
    edge_confidence : pd.DataFrame
        One row per candidate edge with columns
        ``edge`` (tuple), ``llm_score`` (float in [0,1] or NaN),
        ``ci_pvalue`` (float or NaN), ``retained`` (bool),
        ``source`` (one of ``'required'`` / ``'forbidden'`` / ``'ci-test'``).
    iteration_log : list of dict
        Per-iteration structured trace of which edges were proposed,
        validated, demoted.
    skeleton : pd.DataFrame
        Final undirected adjacency matrix (variables x variables).
    cpdag : pd.DataFrame
        Final CPDAG adjacency matrix.
    variables : list of str
    n_obs : int
    alpha : float
    converged : bool
        ``True`` if the loop stopped early because no edges were
        demoted in the most recent iteration.
    provenance : dict
    """
    final_edges: List[Tuple[str, str]]
    edge_confidence: pd.DataFrame
    iteration_log: List[Dict[str, Any]]
    skeleton: pd.DataFrame
    cpdag: pd.DataFrame
    variables: List[str]
    n_obs: int
    alpha: float
    converged: bool
    provenance: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            "LLM-DAG Closed Loop",
            "=" * 60,
            f"  Variables       : {', '.join(self.variables)}",
            f"  N obs           : {self.n_obs}",
            f"  Alpha           : {self.alpha}",
            f"  Iterations run  : {len(self.iteration_log)}",
            f"  Converged       : {self.converged}",
            f"  Final edges     : {len(self.final_edges)}",
        ]
        if self.final_edges:
            lines.append("  Edges:")
            for a, b in self.final_edges:
                row = self.edge_confidence[
                    self.edge_confidence['edge'].apply(lambda e: e == (a, b))
                ]
                if not row.empty:
                    score = row.iloc[0]['llm_score']
                    pval = row.iloc[0]['ci_pvalue']
                    src = row.iloc[0]['source']
                    score_s = f"{score:.2f}" if pd.notna(score) else "NA"
                    pval_s = f"{pval:.3f}" if pd.notna(pval) else "NA"
                    lines.append(
                        f"    {a} -> {b}  (llm={score_s}, ci_p={pval_s},"
                        f" src={src})"
                    )
                else:
                    lines.append(f"    {a} -> {b}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'final_edges': [list(e) for e in self.final_edges],
            'variables': self.variables,
            'n_obs': self.n_obs,
            'alpha': self.alpha,
            'converged': self.converged,
            'iteration_log': self.iteration_log,
            'edge_confidence': self.edge_confidence.to_dict(orient='records'),
            'provenance': self.provenance,
        }

    def to_dag(self):
        """Convert the final CPDAG into a :class:`statspai.dag.DAG`."""
        from ..dag import dag as _dag_factory
        if not self.final_edges:
            return _dag_factory("")
        spec = "; ".join(f"{a} -> {b}" for a, b in self.final_edges)
        return _dag_factory(spec)


@dataclass
class DAGValidationResult:
    """Output of :func:`llm_dag_validate`."""
    edge_evidence: pd.DataFrame  # edge, declared, ci_pvalue, supported
    n_supported: int
    n_unsupported: int
    alpha: float

    def summary(self) -> str:
        lines = [
            "DAG Edge Validation",
            "=" * 60,
            f"  Alpha            : {self.alpha}",
            f"  Edges supported  : {self.n_supported}",
            f"  Edges unsupported: {self.n_unsupported}",
        ]
        for _, row in self.edge_evidence.iterrows():
            edge = row['edge']
            mark = "OK" if row['supported'] else "REJECT"
            pval = row['ci_pvalue']
            pval_s = f"{pval:.3f}" if pd.notna(pval) else "NA"
            lines.append(f"    {edge[0]} -> {edge[1]}  p={pval_s}  [{mark}]")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            'alpha': self.alpha,
            'n_supported': self.n_supported,
            'n_unsupported': self.n_unsupported,
            'edges': [
                {
                    'edge': list(row['edge']),
                    'declared': bool(row['declared']),
                    'ci_pvalue': (None if pd.isna(row['ci_pvalue'])
                                  else float(row['ci_pvalue'])),
                    'supported': bool(row['supported']),
                }
                for _, row in self.edge_evidence.iterrows()
            ],
        }


# --------------------------------------------------------------------- #
#  Helpers
# --------------------------------------------------------------------- #


def _normalize_oracle_output(
    raw: Any,
) -> List[Tuple[str, str, float]]:
    """Convert raw oracle output to a list of (a, b, confidence) tuples.

    Accepts ``list[(a, b)]`` (confidence defaults to 1.0),
    ``list[(a, b, conf)]``, or any object exposing an ``edges``
    iterable of either form (so :class:`LLMDAGProposal` works directly).
    """
    if raw is None:
        return []
    edges_iter = raw
    if hasattr(raw, 'edges') and not isinstance(raw, (list, tuple, set)):
        edges_iter = raw.edges
    out: List[Tuple[str, str, float]] = []
    for item in edges_iter:
        if not isinstance(item, (list, tuple)):
            continue
        if len(item) == 2:
            a, b = item
            conf = 1.0
        elif len(item) >= 3:
            a, b, conf = item[0], item[1], item[2]
        else:
            continue
        try:
            conf = float(conf)
        except (TypeError, ValueError):
            conf = 1.0
        out.append((str(a), str(b), max(0.0, min(1.0, conf))))
    return out


def _partial_corr_pvalue(
    data: np.ndarray, i: int, j: int, conditioning: List[int]
) -> float:
    """Fisher-Z partial-correlation p-value for X_i ⟂ X_j | X_S.

    Mirrors the PC algorithm's CI test (single source of truth would
    re-export from causal_discovery.pc, but inlining keeps this module
    independent and avoids importing private helpers).
    """
    n = data.shape[0]
    cols = [i, j] + list(conditioning)
    sub = data[:, cols]
    # If conditioning set non-empty, compute partial correlation via
    # the inverse correlation matrix.
    try:
        corr = np.corrcoef(sub, rowvar=False)
    except Exception:
        return float('nan')
    if not np.all(np.isfinite(corr)):
        return float('nan')
    try:
        precision = np.linalg.pinv(corr)
    except np.linalg.LinAlgError:
        return float('nan')
    pii = precision[0, 0]
    pjj = precision[1, 1]
    pij = precision[0, 1]
    denom = np.sqrt(pii * pjj)
    if denom <= 0:
        return float('nan')
    rho = -pij / denom
    rho = max(-0.999999, min(0.999999, rho))
    # Fisher Z transform
    z = 0.5 * np.log((1 + rho) / (1 - rho))
    se = 1.0 / np.sqrt(max(n - len(conditioning) - 3, 1))
    stat = abs(z) / se
    pval = 2 * (1 - sp_stats.norm.cdf(stat))
    return float(pval)


def _parents_of(cpdag: np.ndarray, j: int) -> List[int]:
    """Indices i with i -> j in the CPDAG (directed only)."""
    d = cpdag.shape[0]
    return [i for i in range(d)
            if i != j and cpdag[i, j] == 1 and cpdag[j, i] == 0]


# --------------------------------------------------------------------- #
#  Public functions
# --------------------------------------------------------------------- #


def llm_dag_constrained(
    data: pd.DataFrame,
    variables: Optional[Sequence[str]] = None,
    descriptions: Optional[Dict[str, str]] = None,
    *,
    oracle: Optional[Callable[[Sequence[str], Dict[str, str]],
                              Any]] = None,
    alpha: float = 0.05,
    ci_test: str = 'fisherz',
    max_iter: int = 3,
    high_conf_threshold: float = 0.7,
    low_conf_threshold: float = 0.3,
    forbid_low_conf: bool = False,
    verbose: bool = False,
) -> LLMConstrainedDAGResult:
    """Closed-loop LLM-assisted causal discovery.

    Iterate **propose → constrain → CI-validate → demote** until the
    proposed required-edge set stops shrinking or ``max_iter`` is hit.

    Parameters
    ----------
    data : pd.DataFrame
        Observational data.
    variables : sequence of str, optional
        Subset of columns to include in the discovery.  Defaults to all
        numeric columns of ``data``.
    descriptions : dict, optional
        Variable name -> human-readable description (passed to the
        oracle).
    oracle : callable, optional
        Function ``f(variables, descriptions) -> list[(from, to[,
        confidence])]``.  When omitted, the loop falls back to plain
        PC discovery (data-only) and returns a single-iteration result
        with no LLM scores.
    alpha : float, default 0.05
        CI-test significance level for both PC and the validation pass.
    ci_test : {'fisherz'}, default 'fisherz'
        Conditional independence test.
    max_iter : int, default 3
        Upper bound on the number of propose-validate cycles.
    high_conf_threshold : float, default 0.7
        Minimum LLM confidence to inject the edge as a *required*
        background-knowledge constraint into PC.
    low_conf_threshold : float, default 0.3
        Maximum LLM confidence below which the edge is treated as a
        *forbidden* candidate (only when ``forbid_low_conf=True``).
    forbid_low_conf : bool, default False
        When True, low-confidence edges are forbidden in the PC skeleton
        instead of being passed through as plain candidates.  Off by
        default — most LLMs return only positive edges and we don't
        want to over-prune.
    verbose : bool, default False
        Print per-iteration progress.

    Returns
    -------
    LLMConstrainedDAGResult

    Examples
    --------
    >>> import statspai as sp
    >>> def echo_oracle(vars_, desc):
    ...     return [('X', 'Y', 0.95), ('Z', 'X', 0.9)]
    >>> r = sp.llm_dag_constrained(df, variables=['X', 'Y', 'Z'],
    ...                            oracle=echo_oracle, max_iter=3)
    >>> r.summary()
    """
    if variables is not None:
        var_list = [str(v) for v in variables]
        for v in var_list:
            if v not in data.columns:
                raise ValueError(f"Variable {v!r} not in data.columns")
    else:
        var_list = list(data.select_dtypes(include=[np.number]).columns)
    if len(var_list) < 2:
        raise ValueError("At least 2 variables required for discovery.")

    descriptions = dict(descriptions or {})
    X = data[var_list].dropna().values.astype(np.float64)
    n = X.shape[0]

    # 1) Initial proposal — call the oracle once.
    proposal: List[Tuple[str, str, float]] = []
    oracle_error: Optional[str] = None
    if oracle is not None:
        try:
            raw = oracle(var_list, descriptions)
            proposal = _normalize_oracle_output(raw)
        except Exception as exc:
            oracle_error = f"{type(exc).__name__}: {exc}"
    # Keep proposals only over known variables.
    var_set = set(var_list)
    proposal = [(a, b, c) for (a, b, c) in proposal
                if a in var_set and b in var_set and a != b]
    initial_proposal = list(proposal)

    iteration_log: List[Dict[str, Any]] = []
    converged = False
    last_skeleton = None
    last_cpdag = None

    # Track the live edge confidence — start with the LLM proposal.
    edge_state: Dict[Tuple[str, str], Dict[str, Any]] = {}
    for a, b, c in proposal:
        edge_state[(a, b)] = {
            'llm_score': c, 'ci_pvalue': float('nan'),
            'retained': True, 'source': 'candidate',
        }

    from ..causal_discovery.pc import pc_algorithm

    for it in range(max(1, int(max_iter))):
        required = [(a, b) for (a, b), s in edge_state.items()
                    if s['retained'] and s['llm_score'] >= high_conf_threshold]
        forbidden: List[Tuple[str, str]] = []
        if forbid_low_conf:
            forbidden = [(a, b) for (a, b), s in edge_state.items()
                         if s['retained']
                         and s['llm_score'] <= low_conf_threshold]

        pc_out = pc_algorithm(
            data=data, variables=var_list, alpha=alpha, ci_test=ci_test,
            forbidden=forbidden or None, required=required or None,
        )
        last_skeleton = pc_out['skeleton']
        last_cpdag = pc_out['cpdag']
        cpdag_arr = last_cpdag.to_numpy()
        idx = {v: i for i, v in enumerate(var_list)}

        # 4) Validation pass — for each retained required edge, run a
        # partial-correlation CI test conditioning on its current
        # parents in the CPDAG.  Demote edges the data rejects.
        demotions: List[Dict[str, Any]] = []
        for (a, b) in required:
            ia, ib = idx[a], idx[b]
            parents = _parents_of(cpdag_arr, ib)
            cond = [p for p in parents if p != ia]
            pval = _partial_corr_pvalue(X, ia, ib, cond)
            edge_state[(a, b)]['ci_pvalue'] = pval
            if not np.isnan(pval) and pval > alpha:
                edge_state[(a, b)]['retained'] = False
                edge_state[(a, b)]['source'] = 'demoted'
                demotions.append({'edge': (a, b), 'ci_pvalue': pval})

        for (a, b) in required:
            if edge_state[(a, b)]['retained']:
                edge_state[(a, b)]['source'] = 'required'
        for (a, b) in forbidden:
            edge_state[(a, b)]['source'] = 'forbidden'

        # Add CPDAG-discovered edges (without LLM proposal) to the
        # edge_state so they show up in edge_confidence with source
        # ``ci-test``.
        for i, var_i in enumerate(var_list):
            for j, var_j in enumerate(var_list):
                if i == j:
                    continue
                if cpdag_arr[i, j] == 1 and cpdag_arr[j, i] == 0:
                    key = (var_i, var_j)
                    if key not in edge_state:
                        edge_state[key] = {
                            'llm_score': float('nan'),
                            'ci_pvalue': float('nan'),
                            'retained': True,
                            'source': 'ci-test',
                        }

        log_entry = {
            'iter': it,
            'required_in': len(required),
            'forbidden_in': len(forbidden),
            'demoted': len(demotions),
            'demoted_edges': demotions,
            'cpdag_n_directed_edges': int(
                np.sum((cpdag_arr == 1) & (cpdag_arr.T == 0))
            ),
        }
        iteration_log.append(log_entry)
        if verbose:
            print(
                f"[iter {it}] required={len(required)}, "
                f"forbidden={len(forbidden)}, demoted={len(demotions)}"
            )
        if not demotions:
            converged = True
            break

    # Build final edges from final CPDAG (directed only) plus any
    # required-with-direction the LLM still asserts.
    final_edges: List[Tuple[str, str]] = []
    if last_cpdag is not None:
        cpdag_arr = last_cpdag.to_numpy()
        for i, var_i in enumerate(var_list):
            for j, var_j in enumerate(var_list):
                if i == j:
                    continue
                if cpdag_arr[i, j] == 1 and cpdag_arr[j, i] == 0:
                    final_edges.append((var_i, var_j))

    # edge_confidence DataFrame
    rows = []
    for (a, b), s in edge_state.items():
        rows.append({
            'edge': (a, b),
            'llm_score': s['llm_score'],
            'ci_pvalue': s['ci_pvalue'],
            'retained': s['retained'] and ((a, b) in final_edges
                                           or s['source'] == 'forbidden'),
            'source': s['source'],
        })
    edge_confidence = pd.DataFrame(rows)
    if edge_confidence.empty:
        edge_confidence = pd.DataFrame(
            columns=['edge', 'llm_score', 'ci_pvalue', 'retained', 'source']
        )

    return LLMConstrainedDAGResult(
        final_edges=final_edges,
        edge_confidence=edge_confidence,
        iteration_log=iteration_log,
        skeleton=last_skeleton if last_skeleton is not None
        else pd.DataFrame(np.zeros((len(var_list), len(var_list)),
                                   dtype=int),
                          index=var_list, columns=var_list),
        cpdag=last_cpdag if last_cpdag is not None
        else pd.DataFrame(np.zeros((len(var_list), len(var_list)),
                                   dtype=int),
                          index=var_list, columns=var_list),
        variables=var_list,
        n_obs=n,
        alpha=alpha,
        converged=converged,
        provenance={
            'oracle_error': oracle_error,
            'oracle_edges_proposed': initial_proposal,
            'high_conf_threshold': high_conf_threshold,
            'low_conf_threshold': low_conf_threshold,
            'forbid_low_conf': forbid_low_conf,
            'max_iter': max_iter,
            'ci_test': ci_test,
        },
    )


def llm_dag_validate(
    dag,
    data: pd.DataFrame,
    *,
    alpha: float = 0.05,
    ci_test: str = 'fisherz',
) -> DAGValidationResult:
    """Per-edge CI-test validation of a declared DAG.

    For every directed edge ``a -> b`` in ``dag``, run a partial-
    correlation CI test of ``a ⟂ b | parents(b) \\ {a}``.  Edges with
    p-value <= ``alpha`` are *supported* (the data did not provide
    evidence to remove them); edges with p-value > ``alpha`` are
    *unsupported* (the data is consistent with the conditional
    independence implied by removing the edge).

    Parameters
    ----------
    dag : statspai.dag.DAG or object exposing an ``edges`` attribute
        Declared causal graph.  Latent ``_L_*`` nodes are ignored.
    data : pd.DataFrame
    alpha : float, default 0.05
    ci_test : {'fisherz'}, default 'fisherz'

    Returns
    -------
    DAGValidationResult
    """
    if ci_test != 'fisherz':
        raise ValueError(f"Unknown ci_test: {ci_test!r}. Use 'fisherz'.")

    # Resolve declared edges from a DAG-like object.
    declared_edges: List[Tuple[str, str]] = []
    edges_attr = getattr(dag, 'edges', None)
    if callable(edges_attr):
        declared_edges = [tuple(e) for e in edges_attr()]
    elif edges_attr is not None:
        declared_edges = [tuple(e) for e in edges_attr]
    else:
        # Try DAG._edges (from statspai.dag.DAG)
        adj = getattr(dag, '_edges', None)
        if adj is not None:
            for parent, children in adj.items():
                for child in children:
                    declared_edges.append((parent, child))
    declared_edges = [(a, b) for a, b in declared_edges
                      if not str(a).startswith('_L_')
                      and not str(b).startswith('_L_')]

    if not declared_edges:
        return DAGValidationResult(
            edge_evidence=pd.DataFrame(
                columns=['edge', 'declared', 'ci_pvalue', 'supported']
            ),
            n_supported=0, n_unsupported=0, alpha=alpha,
        )

    nodes = sorted({n for e in declared_edges for n in e})
    nodes = [n for n in nodes if n in data.columns]
    if len(nodes) < 2:
        raise ValueError(
            "Need at least 2 declared, observable nodes that exist in `data`."
        )
    X = data[nodes].dropna().values.astype(np.float64)
    name_to_idx = {n: i for i, n in enumerate(nodes)}

    # Compute parents for each child from the declared graph.
    parent_map: Dict[str, List[str]] = {}
    for a, b in declared_edges:
        parent_map.setdefault(b, []).append(a)

    rows = []
    for a, b in declared_edges:
        if a not in name_to_idx or b not in name_to_idx:
            rows.append({
                'edge': (a, b), 'declared': True,
                'ci_pvalue': float('nan'), 'supported': False,
            })
            continue
        ia = name_to_idx[a]
        ib = name_to_idx[b]
        cond = [name_to_idx[p] for p in parent_map.get(b, [])
                if p != a and p in name_to_idx]
        pval = _partial_corr_pvalue(X, ia, ib, cond)
        supported = (not np.isnan(pval)) and (pval <= alpha)
        rows.append({
            'edge': (a, b), 'declared': True,
            'ci_pvalue': pval, 'supported': supported,
        })
    df = pd.DataFrame(rows)
    return DAGValidationResult(
        edge_evidence=df,
        n_supported=int(df['supported'].sum()),
        n_unsupported=int((~df['supported']).sum()),
        alpha=alpha,
    )
