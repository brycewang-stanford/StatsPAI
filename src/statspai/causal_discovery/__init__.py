"""
Causal Discovery: Learning causal structure from observational data.

Algorithms
----------
- **NOTEARS** : NO TEARS continuous optimisation for DAG learning
  (Zheng et al. 2018). Formulates structure learning as a smooth
  optimisation problem with an acyclicity constraint.

- **PC Algorithm** : Constraint-based causal discovery using conditional
  independence tests (Spirtes, Glymour, Scheines 2000). Learns a CPDAG
  (completed partially directed acyclic graph).

References
----------
Zheng, X., Aragam, B., Ravikumar, P., & Xing, E. P. (2018).
DAGs with NO TEARS: Continuous Optimization for Structure Learning.
Advances in Neural Information Processing Systems, 31. [@zheng2018dags]

Spirtes, P., Glymour, C., & Scheines, R. (2000).
Causation, Prediction, and Search (2nd ed.). MIT Press. [@spirtes2000causation]
"""

from .notears import notears, NOTEARS
from .pc import pc_algorithm, PCAlgorithm
from .lingam import lingam, LiNGAMResult
from .ges import ges, GESResult
from .fci import fci, FCIResult
from .icp import icp, nonlinear_icp, ICPResult
from .pcmci import pcmci, PCMCIResult, partial_corr_pvalue
from .lpcmci import lpcmci, LPCMCIResult
from .dynotears import dynotears, DYNOTEARSResult
from . import _viz
from ._viz import to_networkx, to_dot, plot_dag, edge_list, shd

# ---------------------------------------------------------------- #
#  Attach .to_networkx / .to_dot / .plot to every result class so
#  ``result.plot()`` works regardless of which discovery algorithm
#  produced it. The call sites convert each dataclass's adjacency
#  storage convention into the canonical "A[i,j] != 0 ⇒ i → j" form.
# ---------------------------------------------------------------- #


def _adj_and_names_lingam(result: LiNGAMResult):
    # LiNGAM stores B[i,j] = effect of j on i, so transpose to canonical.
    return result.adjacency.T, result.names, True


def _adj_and_names_default(result):
    # Standard: A[i,j] != 0 ⇒ names[i] → names[j]
    return result.adjacency, result.names, True


def _adj_and_names_pcmci(result):
    # PCMCI stores (lag, i, j); collapse over lags by taking the maximum
    # absolute link-strength across lags so the static plot summarises
    # all temporal dependencies.
    A = result.adjacency
    if A.ndim == 3:
        collapsed = np.abs(A).max(axis=0).astype(float)
        # restore signs from lag-0 if present, else keep magnitudes
        return collapsed, result.names, True
    return A, result.names, True


def _adj_and_names_dynotears(result: DYNOTEARSResult):
    # DYNOTEARS exposes a contemporaneous W and a lagged A; the static
    # DAG plot uses W only (lagged links are best inspected in tabular form).
    W = getattr(result, "W", None)
    if W is None:
        W = getattr(result, "adjacency", None)
    return W, getattr(result, "names", None) or getattr(result, "var_names", None), True


def _adj_and_names_icp(result: ICPResult):
    # ICP returns parent sets; build adjacency from accepted parents.
    parents = getattr(result, "parents", {}) or {}
    target = getattr(result, "target", None)
    names = list(getattr(result, "names", None) or
                 getattr(result, "predictors", None) or [])
    if target is not None and target not in names:
        names = names + [target]
    name_to_idx = {nm: i for i, nm in enumerate(names)}
    k = len(names)
    A = np.zeros((k, k), dtype=float)
    if target is not None:
        for parent in parents:
            if parent in name_to_idx and target in name_to_idx:
                A[name_to_idx[parent], name_to_idx[target]] = 1.0
    return A, names, True


import numpy as np  # noqa: E402  (after the patch helpers reference it)

_ADJ_GETTERS = {
    LiNGAMResult: _adj_and_names_lingam,
    GESResult: _adj_and_names_default,
    FCIResult: _adj_and_names_default,
    DYNOTEARSResult: _adj_and_names_dynotears,
    PCMCIResult: _adj_and_names_pcmci,
    LPCMCIResult: _adj_and_names_pcmci,
    ICPResult: _adj_and_names_icp,
}


def _make_method(viz_func, transform_axes=("matrix", "names", "directed")):
    def _bound(self, *args, **kwargs):
        getter = _ADJ_GETTERS.get(type(self), _adj_and_names_default)
        A, names, directed = getter(self)
        kwargs.setdefault("directed", directed)
        return viz_func(A, names, *args, **kwargs)
    return _bound


for _cls in _ADJ_GETTERS:
    if not hasattr(_cls, "to_networkx"):
        _cls.to_networkx = _make_method(to_networkx)
    if not hasattr(_cls, "to_dot"):
        _cls.to_dot = _make_method(to_dot)
    if not hasattr(_cls, "plot"):
        _cls.plot = _make_method(plot_dag)
    if not hasattr(_cls, "edge_list"):
        _cls.edge_list = _make_method(edge_list)


__all__ = [
    'notears',
    'NOTEARS',
    'pc_algorithm',
    'PCAlgorithm',
    'lingam',
    'LiNGAMResult',
    'ges',
    'GESResult',
    'fci',
    'FCIResult',
    'icp',
    'nonlinear_icp',
    'ICPResult',
    'pcmci',
    'PCMCIResult',
    'partial_corr_pvalue',
    'lpcmci',
    'LPCMCIResult',
    'dynotears',
    'DYNOTEARSResult',
    # Shared graph helpers (also work standalone on adjacency matrices)
    'to_networkx',
    'to_dot',
    'plot_dag',
    'edge_list',
    'shd',
]
