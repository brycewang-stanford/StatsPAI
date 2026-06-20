"""Exponential random graph models (ERGM) for ``sp.network`` — MPLE.

An ERGM places a distribution over networks,

.. math::

    P(Y = y) = \\frac{\\exp(\\theta^\\top g(y))}{\\kappa(\\theta)},

where ``g(y)`` is a vector of network statistics (edges, homophily,
triangles, …).  This module fits ``theta`` by **maximum pseudo-likelihood
estimation** (MPLE; Strauss & Ikeda 1990): the joint is approximated by the
product of dyadwise full-conditionals, each of which is logistic in the
*change statistics* ``Δg_ij`` (the change in ``g`` from toggling tie
``ij``).  MPLE therefore reduces to a logistic regression of the observed
ties on their change statistics.

Scope (honest)
--------------
* For **dyad-independent** terms (``edges``, ``nodematch``, ``nodecov``,
  ``nodefactor``, ``mutual`` with no triadic terms) MPLE coincides with the
  exact MLE, matching ``statnet::ergm``'s MPLE.
* For **dyad-dependent** terms (``triangles``) MPLE is a consistent but
  approximate estimator; its pseudo-likelihood standard errors understate
  uncertainty.  Full **MCMC-MLE** (importance-sampling the intractable
  normaliser) and **stochastic actor-oriented models (SAOM / RSiena)** for
  network *dynamics* are on the roadmap — see ``network/__init__.py``.  This
  is flagged loudly rather than silently approximated.

References
----------
Strauss, D. & Ikeda, M. (1990). "Pseudolikelihood estimation for social
networks." *JASA*, 85(409), 204-212. [@strauss1990pseudolikelihood]

Robins, G., Pattison, P., Kalish, Y. & Lusher, D. (2007). "An introduction
to exponential random graph (p*) models for social networks." *Social
Networks*, 29(2), 173-191. [@robins2007introduction]
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from .._result_serialize import ResultProtocolMixin
from ._core import as_graph
from .regression import _irls_logit

__all__ = ["ergm", "ERGMResult"]

_DYAD_DEPENDENT = {"triangles", "triangle", "ttriad"}


@dataclass
class ERGMResult(ResultProtocolMixin):
    """ERGM fit by maximum pseudo-likelihood (the ``sp.ergm`` output).

    Attributes
    ----------
    coefficients : pandas.DataFrame
        Columns ``term``, ``estimate``, ``se``, ``z``, ``p``.
    terms : list of str
    log_pseudolikelihood : float
    n_dyads : int
    directed : bool
    dyad_independent : bool
        Whether the model contains only dyad-independent terms (so MPLE = MLE).
    method : str
        ``"MPLE"``.

    Examples
    --------
    >>> import statspai as sp
    >>> g = sp.florentine_families()
    >>> res = sp.ergm(g, terms=["edges"])
    >>> # edges-only MPLE recovers the log-odds of the density:
    >>> import numpy as np
    >>> dens = sp.network_summary(g).density
    >>> bool(abs(res.coefficients.loc[0, "estimate"]
    ...          - np.log(dens / (1 - dens))) < 1e-6)
    True
    """

    _citation_keys = ("strauss1990pseudolikelihood", "robins2007introduction")

    coefficients: pd.DataFrame
    terms: List[str]
    log_pseudolikelihood: float
    n_dyads: int
    directed: bool
    dyad_independent: bool
    method: str = "MPLE"
    detail: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:  # pragma: no cover - cosmetic
        warn = (
            ""
            if self.dyad_independent
            else "\n  NOTE: dyad-dependent terms -> MPLE is approximate; "
            "SEs understate uncertainty (use MCMC-MLE when available)."
        )
        return (
            f"ERGM (maximum pseudo-likelihood, "
            f"{'dyad-independent' if self.dyad_independent else 'dyad-dependent'})\n"
            f"  n_dyads : {self.n_dyads}\n"
            f"  logPL   : {self.log_pseudolikelihood:.3f}\n"
            f"{self.coefficients.to_string(index=False)}{warn}"
        )

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return f"ERGMResult(terms={self.terms}, method='MPLE')"


def _node_attr(node_attrs: Any, name: str, labels: List[Any]) -> np.ndarray:
    if node_attrs is None:
        raise ValueError(f"term references node attribute {name!r} but node_attrs=None")
    if isinstance(node_attrs, pd.DataFrame):
        col = node_attrs[name]
        if list(col.index) != list(labels):
            try:
                col = col.reindex(labels)
            except Exception:  # positional fallback
                pass
        return np.asarray(col.to_numpy())
    if isinstance(node_attrs, dict):
        arr: np.ndarray = np.asarray(node_attrs[name])
        return arr
    raise TypeError("node_attrs must be a DataFrame or dict of arrays")


def _change_statistic(
    term: str,
    A: np.ndarray,
    mask: np.ndarray,
    directed: bool,
    node_attrs: Any,
    labels: List[Any],
) -> Tuple[str, np.ndarray]:
    """Return ``(label, change-statistic vector over masked dyads)``."""
    n = A.shape[0]
    key = term.split(":")[0].strip().lower()
    if key == "edges":
        return "edges", np.ones(int(mask.sum()))
    if key == "mutual":
        if not directed:
            raise ValueError("term 'mutual' requires a directed graph")
        return "mutual", A.T[mask]
    if key in ("triangles", "triangle", "ttriad"):
        # adding tie ij closes (A^2)_ij triangles (common neighbours)
        common = (A @ A)[mask]
        return "triangles", common
    if key in ("nodematch", "nodecov", "nodefactor", "absdiff"):
        attr_name = term.split(":", 1)[1].strip()
        x: np.ndarray = _node_attr(node_attrs, attr_name, labels).astype(float)
        Xi = np.repeat(x[:, None], n, axis=1)
        Xj = np.repeat(x[None, :], n, axis=0)
        if key == "nodematch":
            M = (Xi == Xj).astype(float)
            return f"nodematch.{attr_name}", M[mask]
        if key == "nodecov":
            M = Xi + Xj
            return f"nodecov.{attr_name}", M[mask]
        if key == "absdiff":
            M = np.abs(Xi - Xj)
            return f"absdiff.{attr_name}", M[mask]
        if key == "nodefactor":
            M = Xi + Xj
            return f"nodefactor.{attr_name}", M[mask]
    raise ValueError(
        f"unknown ERGM term {term!r}; supported: edges, mutual, triangles, "
        f"nodematch:<attr>, nodecov:<attr>, absdiff:<attr>"
    )


def ergm(
    graph: Any,
    terms: Sequence[str] = ("edges",),
    node_attrs: Any = None,
    directed: Optional[bool] = None,
    alpha: float = 0.05,
) -> ERGMResult:
    """Fit an exponential random graph model by pseudo-likelihood (MPLE).

    Parameters
    ----------
    graph : Graph or adjacency-like
        The observed network.
    terms : sequence of str, default ``("edges",)``
        Model terms.  Supported:

        * ``"edges"`` — density (the ERGM intercept).
        * ``"mutual"`` — reciprocity (directed graphs only).
        * ``"triangles"`` — transitivity (dyad-dependent → MPLE approximate).
        * ``"nodematch:<attr>"`` — uniform homophily on a node attribute.
        * ``"nodecov:<attr>"`` — main effect of a continuous attribute (sum).
        * ``"absdiff:<attr>"`` — heterophily by absolute attribute difference.
    node_attrs : DataFrame or dict, optional
        Node attributes (required by ``nodematch`` / ``nodecov`` / ``absdiff``).
        A ``DataFrame`` is aligned to node labels; a ``dict`` maps name →
        length-``n`` array in node order.
    directed : bool, optional
        Inferred from the graph when ``None``.
    alpha : float, default 0.05

    Returns
    -------
    ERGMResult

    References
    ----------
    strauss1990pseudolikelihood

    Examples
    --------
    >>> import statspai as sp
    >>> g = sp.florentine_families()
    >>> res = sp.ergm(g, terms=["edges", "triangles"])
    >>> "triangles" in res.terms
    True
    """
    g = as_graph(graph, directed=directed)
    A = g.binary()
    n = g.n_nodes
    is_dir = g.is_directed
    mask = ~np.eye(n, dtype=bool) if is_dir else np.triu(np.ones((n, n), bool), 1)
    yv = A[mask]

    cols: List[np.ndarray] = []
    names: List[str] = []
    dyad_indep = True
    for t in terms:
        if t.split(":")[0].strip().lower() in _DYAD_DEPENDENT:
            dyad_indep = False
        label, vec = _change_statistic(t, A, mask, is_dir, node_attrs, g.labels)
        names.append(label)
        cols.append(vec)
    X = np.column_stack(cols)

    if not dyad_indep:
        warnings.warn(
            "ERGM contains dyad-dependent terms (e.g. 'triangles'); MPLE is a "
            "consistent but approximate estimator and its standard errors "
            "understate uncertainty. Treat p-values as indicative; full "
            "MCMC-MLE is the roadmap.",
            stacklevel=2,
        )

    beta, se, ll, _ = _irls_logit(yv, X, max_iter=200)
    crit = float(stats.norm.ppf(1 - alpha / 2))
    z = np.divide(beta, se, out=np.zeros_like(beta), where=se > 0)
    pvals = 2 * (1 - stats.norm.cdf(np.abs(z)))
    coef_df = pd.DataFrame(
        {
            "term": names,
            "estimate": beta,
            "se": se,
            "z": z,
            "p": pvals,
            "ci_low": beta - crit * se,
            "ci_high": beta + crit * se,
        }
    )
    return ERGMResult(
        coefficients=coef_df,
        terms=names,
        log_pseudolikelihood=float(ll),
        n_dyads=int(len(yv)),
        directed=is_dir,
        dyad_independent=dyad_indep,
        detail={"alpha": alpha},
    )
