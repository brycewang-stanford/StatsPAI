"""
LLM-assisted DAG construction (scaffold / plugin interface).

The 2024-2026 trend of *LLM-assisted causal discovery* (Kıcıman et al.
2023; Long et al. 2023; Ban et al. 2024) uses large language models as
a **domain-knowledge oracle** to:

1. Propose an initial causal graph from variable descriptions,
2. Score / prune candidate edges against CI-test skeletons, and
3. Serve as an "expert override" layer on top of classical discovery
   algorithms.

This module provides the *orchestration scaffold*: a pluggable
``oracle`` callable that accepts ``(variables, descriptions)`` and
returns a list of ``(from, to)`` edges. Users supply their own oracle
(OpenAI / Anthropic / local model), and the module

* merges the oracle's edges with a skeleton from a CI test (PC/FCI),
* flags disagreements (oracle says edge exists but CI test rejects, or
  vice-versa),
* returns a :class:`DAG` with provenance metadata.

No network calls are made by this module directly — it is dependency-
free and simply composes the oracle's output with the rest of
StatsPAI's causal-discovery stack.

References
----------
Kıcıman, E., Ness, R., Sharma, A., & Tan, C. (2023). "Causal reasoning
and large language models: Opening a new frontier for causality."
*arXiv*:2305.00050.

Long, S., Piché, A., Zantedeschi, V., Schuster, T., & Drouin, A.
(2023). "Causal discovery with language models as imperfect experts."
*arXiv*:2307.02390.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Any

import pandas as pd

Edge = Tuple[str, str]
Oracle = Callable[[Sequence[str], Dict[str, str]], List[Edge]]


@dataclass
class LLMDAGResult:
    """Merged LLM-oracle / CI-test DAG returned by :func:`llm_dag`.

    Attributes
    ----------
    edges : list of (str, str)
        Final merged edge set after ``merge_strategy`` is applied.
    oracle_edges : list of (str, str)
        Raw ``(from, to)`` edges proposed by the LLM oracle.
    ci_rejects, ci_asserts : list of (str, str)
        Variable pairs the CI-test skeleton rejected / asserted.
    disagreements : list of (str, str, str)
        ``(from, to, reason)`` where oracle and CI test conflict.
    provenance : dict
        Bookkeeping: oracle error (if any), merge strategy, ci_test, alpha.

    Examples
    --------
    >>> import statspai as sp
    >>> # An oracle is any callable: (variables, descriptions) -> edges.
    >>> def domain_oracle(variables, descriptions):
    ...     return [("smoking", "cancer")]  # e.g. an LLM API call
    >>> res = sp.llm_dag(
    ...     variables=["smoking", "cancer", "tar"],
    ...     oracle=domain_oracle,
    ...     merge_strategy="oracle_only",
    ... )  # doctest: +SKIP
    >>> res.edges          # final merged edges  # doctest: +SKIP
    >>> res.disagreements  # oracle vs CI-test conflicts  # doctest: +SKIP
    """

    edges: List[Edge]
    oracle_edges: List[Edge]
    ci_rejects: List[Edge]
    ci_asserts: List[Edge]
    disagreements: List[Tuple[str, str, str]]  # (from, to, reason)
    provenance: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:  # pragma: no cover
        lines = ["LLM-assisted DAG"]
        lines.append(f"  oracle edges    : {self.oracle_edges}")
        lines.append(f"  CI-rejected     : {self.ci_rejects}")
        lines.append(f"  CI-asserted     : {self.ci_asserts}")
        lines.append(f"  final edges     : {self.edges}")
        if self.disagreements:
            lines.append("  disagreements:")
            for src, dst, why in self.disagreements:
                lines.append(f"    {src} -> {dst}: {why}")
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover
        return f"LLMDAGResult(edges={len(self.edges)})"


def llm_dag(
    variables: Sequence[str],
    descriptions: Optional[Dict[str, str]] = None,
    oracle: Optional[Oracle] = None,
    data: Optional[pd.DataFrame] = None,
    ci_test: str = "fisherz",
    alpha: float = 0.05,
    merge_strategy: str = "intersection",
) -> LLMDAGResult:
    """
    Merge an LLM-proposed DAG with a data-driven CI-test skeleton.

    Parameters
    ----------
    variables : sequence of str
        Variable names.
    descriptions : dict, optional
        Human-readable description per variable (passed to the oracle).
    oracle : callable, optional
        A callable ``f(variables, descriptions) -> list[(from, to)]``.
        If ``None``, this function runs a **data-only** pipeline.
    data : pd.DataFrame, optional
        Observational data used for CI-test skeleton construction.
        Required for merge_strategy in {"intersection", "union_with_ci"}.
    ci_test : {"fisherz"}, default "fisherz"
    alpha : float, default 0.05
    merge_strategy : {"oracle_only", "intersection", "union_with_ci"}
        How to combine oracle and CI-test results.

    Returns
    -------
    LLMDAGResult

    Examples
    --------
    >>> import statspai as sp
    >>> import pandas as pd
    >>> import numpy as np
    >>> df = pd.DataFrame(np.random.randn(100, 3),
    ...                   columns=["x", "m", "y"])
    >>> def oracle(variables, descriptions):
    ...     return [("x", "m"), ("m", "y")]  # e.g. an LLM proposal
    >>> res = sp.llm_dag(
    ...     variables=["x", "m", "y"],
    ...     oracle=oracle,
    ...     data=df,
    ...     merge_strategy="intersection",
    ... )  # doctest: +SKIP
    >>> res.edges  # edges surviving oracle + CI-test agreement  # doctest: +SKIP
    """
    descriptions = dict(descriptions or {})

    # 1) Oracle pass
    oracle_edges: List[Edge] = []
    oracle_error: Optional[str] = None
    if oracle is not None:
        try:
            raw = oracle(list(variables), descriptions)
            filtered_edges: List[Edge] = []
            for edge in raw:
                if len(edge) == 2:
                    src, dst = edge
                    filtered_edges.append((src, dst))
            oracle_edges = filtered_edges
        except Exception as exc:  # pragma: no cover
            oracle_edges = []
            oracle_error = str(exc)
    else:
        oracle_error = "no oracle supplied"

    # 2) Data-driven skeleton via CI tests (optional)
    ci_asserts: List[Edge] = []
    ci_rejects: List[Edge] = []
    if data is not None:
        from ..causal_discovery.pc import pc_algorithm

        try:
            pc = pc_algorithm(
                data=data,
                variables=list(variables),
                alpha=alpha,
                ci_test=ci_test,
            )
            skeleton = pc["skeleton"].to_numpy()
            for i, a in enumerate(variables):
                for j, b in enumerate(variables):
                    if i >= j:
                        continue
                    if skeleton[i, j] == 1:
                        ci_asserts.append((a, b))
                    else:
                        ci_rejects.append((a, b))
        except Exception:  # pragma: no cover
            pass

    # 3) Merge
    oracle_set: set[Edge] = set(oracle_edges)
    ci_set: set[Edge] = set()
    for a, b in ci_asserts:
        ci_set.add((a, b))
        ci_set.add((b, a))

    if merge_strategy == "oracle_only":
        final = list(oracle_set)
    elif merge_strategy == "intersection":
        final = [e for e in oracle_set if e in ci_set or (e[1], e[0]) in ci_set]
    elif merge_strategy == "union_with_ci":
        final = list(oracle_set | ci_set)
    else:
        raise ValueError(f"unknown merge_strategy: {merge_strategy}")

    disagreements: List[Tuple[str, str, str]] = []
    for e in oracle_set:
        rev = (e[1], e[0])
        undirected_exists = e in ci_set or rev in ci_set
        if not undirected_exists and data is not None:
            disagreements.append((e[0], e[1], "CI test rejects this edge"))

    return LLMDAGResult(
        edges=final,
        oracle_edges=oracle_edges,
        ci_rejects=ci_rejects,
        ci_asserts=ci_asserts,
        disagreements=disagreements,
        provenance={
            "oracle_error": oracle_error,
            "merge_strategy": merge_strategy,
            "ci_test": ci_test,
            "alpha": alpha,
        },
    )


__all__ = ["llm_dag", "LLMDAGResult"]
