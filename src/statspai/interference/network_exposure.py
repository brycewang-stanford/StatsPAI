"""
Exposure-mapping estimators for causal effects under network interference.

Setup
-----
* :math:`n` units with binary treatments :math:`W \\in \\{0,1\\}^n` drawn
  independently with a known probability (a Bernoulli design).
* An adjacency matrix whose row :math:`i` lists the units :math:`N_i`
  whose treatment may affect unit :math:`i`.
* An *exposure mapping* :math:`H_i(W)` that reduces the treatment vector
  to a categorical exposure depending only on :math:`W_i` and
  :math:`W_{N_i}`.

With exposure probabilities :math:`e_i(h) = P(H_i = h)` and weights
:math:`\\Gamma_i(h) = 1\\{H_i = h\\} / e_i(h)`, the average outcome under
exposure :math:`h` is estimated by the self-normalised (Hajek) mean
:math:`\\sum_i \\Gamma_i(h) Y_i / \\sum_i \\Gamma_i(h)` or by the
Horvitz-Thompson mean :math:`n^{-1} \\sum_i \\Gamma_i(h) Y_i`.

Exposure probabilities are exact for the built-in mappings (they follow
from the binomial distribution of the number of treated neighbours);
they are simulated only for a user-supplied mapping.

Variance
--------
Two units have dependent exposures only if some unit's treatment enters
both, which defines the dependency graph
:math:`G_{ij} = 1\\{(N_i \\cup \\{i\\}) \\cap (N_j \\cup \\{j\\}) \\ne
\\emptyset\\}`. The variance of an estimate with linearised terms
:math:`v_i` is estimated by :math:`n^{-2} v' G v` (``variance="hac"``).
That estimator is not guaranteed to be conservative when :math:`G` has
negative eigenvalues; ``variance="hac_psd"`` (the default) replaces
:math:`G` by its positive semidefinite part, which is always
conservative for the randomisation variance.

The four-cell mapping ``"as4"`` is

* ``c00`` : W_i = 0, no treated neighbour
* ``c10`` : W_i = 1, no treated neighbour  (direct effect)
* ``c01`` : W_i = 0, at least one treated neighbour (spillover)
* ``c11`` : W_i = 1, at least one treated neighbour

References
----------
[@aronow2017estimating], [@leung2022causal], [@gao2025causal]
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy import sparse, stats

from .._result_serialize import ResultProtocolMixin
from ..exceptions import AssumptionWarning, DataInsufficient, MethodIncompatibility

#: Largest eligible sample for which the dense eigendecomposition behind
#: ``variance="hac_psd"`` is attempted.
_PSD_MAX_N = 6000

Mapping = Union[str, Callable[[np.ndarray, np.ndarray], np.ndarray]]

# --------------------------------------------------------------------
# Result container
# --------------------------------------------------------------------


@dataclass
class NetworkExposureResult(ResultProtocolMixin):
    """Container for :func:`network_exposure` estimates.

    Attributes
    ----------
    estimates : pandas.DataFrame
        One row per exposure level: mean outcome, standard error,
        confidence interval, number of units realised at that level,
        smallest exposure probability and effective sample size.
    contrasts : pandas.DataFrame
        Differences between exposure levels with standard errors that
        account for the dependence between the two estimated means.
    exposure_levels : list of str
        The exposure categories of the mapping.
    n_obs : int
        Number of units supplied.
    n_eligible : int
        Number of units for which every exposure level has positive
        probability; all averages are over these units.

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> n = 60
    >>> A = np.zeros((n, n), dtype=int)
    >>> for i in range(n):
    ...     A[i, (i + 1) % n] = 1
    ...     A[i, (i - 1) % n] = 1
    >>> Z = (rng.random(n) < 0.5).astype(int)
    >>> Y = 1.0 + 2.0 * Z + 0.5 * (A @ Z) + rng.normal(size=n)
    >>> res = sp.network_exposure(Y, Z, A, p_treat=0.5)
    >>> isinstance(res, sp.NetworkExposureResult)
    True
    >>> res.exposure_levels
    ['c00', 'c01', 'c10', 'c11']
    """

    _citation_keys = ("aronow2017estimating", "leung2022causal", "gao2025causal")

    estimates: pd.DataFrame  # one row per exposure level
    contrasts: pd.DataFrame  # pairwise contrasts (e.g. direct, spillover)
    exposure_levels: List[str]
    n_obs: int
    p_treat: float
    design: str
    mapping: str
    detail: Dict[str, Any] = field(default_factory=dict)
    estimator: str = "hajek"
    variance: str = "hac_psd"
    n_eligible: int = 0
    alpha: float = 0.05

    def summary(self) -> str:  # pragma: no cover
        return (
            f"Network exposure estimates ({self.mapping}, {self.design})\n"
            f"  n = {self.n_obs} ({self.n_eligible} with every exposure "
            f"possible), p = {self.p_treat:.3f}\n"
            f"  estimator = {self.estimator}, variance = {self.variance}\n"
            f"{self.estimates.to_string(index=False)}\n\n"
            "Contrasts:\n"
            f"{self.contrasts.to_string(index=False)}"
        )

    def __repr__(self) -> str:  # pragma: no cover
        return f"NetworkExposureResult(levels={self.exposure_levels})"


# --------------------------------------------------------------------
# Adjacency helpers
# --------------------------------------------------------------------


def _to_adj(adj_or_edges: Any, n: Optional[int] = None) -> np.ndarray:
    """Coerce adjacency input (matrix or edge list) into a binary numpy matrix."""
    if sparse.issparse(adj_or_edges):
        A = (adj_or_edges.toarray() != 0).astype(int)
    elif isinstance(adj_or_edges, np.ndarray):
        A = (adj_or_edges != 0).astype(int)
    elif isinstance(adj_or_edges, pd.DataFrame):
        A = (adj_or_edges.to_numpy() != 0).astype(int)
    elif isinstance(adj_or_edges, (list, tuple, np.generic)):
        # np.generic is in the guard but edge lists are list/tuple at runtime.
        edges = np.asarray(list(adj_or_edges))  # type: ignore[arg-type]
        if edges.ndim != 2 or edges.shape[1] != 2:
            raise MethodIncompatibility("edge list must be (n_edges, 2)")
        if n is None:
            n = int(edges.max()) + 1
        A = np.zeros((n, n), dtype=int)
        for u, v in edges:
            A[int(u), int(v)] = 1
            A[int(v), int(u)] = 1
    else:
        raise MethodIncompatibility(
            "adjacency must be an ndarray, a sparse matrix, a DataFrame or "
            "an edge list"
        )
    if A.ndim != 2 or A.shape[0] != A.shape[1]:
        raise MethodIncompatibility("adjacency must be square")
    np.fill_diagonal(A, 0)
    return np.asarray(A)


def _as4_mapping(Z: np.ndarray, A: np.ndarray) -> np.ndarray:
    """Four-cell exposure: own treatment x any-neighbour-treated."""
    has_t_neigh = (A @ Z) > 0
    own = np.asarray(Z).astype(int)
    labels = np.array(["c00", "c01", "c10", "c11"], dtype=object)
    return np.asarray(labels[2 * own + has_t_neigh.astype(int)])


def _fraction_bins(
    counts: np.ndarray, deg: np.ndarray, thresholds: Tuple[float, ...]
) -> np.ndarray:
    frac = np.where(deg > 0, counts / np.maximum(deg, 1), 0.0)
    return np.digitize(frac, thresholds)


def _fraction_mapping(
    Z: np.ndarray,
    A: np.ndarray,
    thresholds: Tuple[float, ...] = (0.0, 0.5),
) -> np.ndarray:
    """Bin own treatment x fraction of treated neighbours."""
    deg = A.sum(axis=1).astype(float)
    own = np.asarray(Z).astype(int)
    bin_ = _fraction_bins(A @ Z, deg, thresholds)
    return np.array(
        [f"z{own[i]}_b{bin_[i]}" for i in range(own.shape[0])], dtype=object
    )


# --------------------------------------------------------------------
# Exposure probabilities
# --------------------------------------------------------------------


def _neighbour_pmf(
    d: int, own: int, p: float, n: int, n_treated: Optional[int]
) -> np.ndarray:
    """Distribution of the number of treated neighbours of a unit.

    Binomial under Bernoulli assignment. With the number treated fixed
    (``n_treated`` given), the ``d`` neighbours are a draw without
    replacement from the other ``n - 1`` units, of which
    ``n_treated - own`` are treated.
    """
    k = np.arange(d + 1)
    if n_treated is None:
        return np.asarray(stats.binom.pmf(k, d, p))
    return np.asarray(stats.hypergeom.pmf(k, n - 1, n_treated - own, d))


def _as4_probabilities(
    deg: np.ndarray, p: float, n_treated: Optional[int] = None
) -> Tuple[Dict[str, np.ndarray], List[str]]:
    """Exact exposure probabilities of the four-cell map."""
    n = deg.shape[0]
    none = {0: np.empty(n), 1: np.empty(n)}  # no neighbour treated, by own
    for d in np.unique(deg):
        for own in (0, 1):
            none[own][deg == d] = _neighbour_pmf(int(d), own, p, n, n_treated)[0]
    probs = {
        "c00": (1 - p) * none[0],
        "c01": (1 - p) * (1 - none[0]),
        "c10": p * none[1],
        "c11": p * (1 - none[1]),
    }
    return probs, sorted(probs)


def _fraction_probabilities(
    deg: np.ndarray,
    p: float,
    thresholds: Tuple[float, ...],
    n_treated: Optional[int] = None,
) -> Tuple[Dict[str, np.ndarray], List[str]]:
    """Exact exposure probabilities of the fraction map."""
    n = deg.shape[0]
    n_bins = len(thresholds) + 1
    probs: Dict[str, np.ndarray] = {}
    for own, p_own in ((0, 1 - p), (1, p)):
        bin_prob = np.zeros((n, n_bins))
        for d in np.unique(deg):
            k = np.arange(int(d) + 1)
            pmf = _neighbour_pmf(int(d), own, p, n, n_treated)
            bins = _fraction_bins(k, np.full(k.shape, d), thresholds)
            bin_prob[deg == d] = np.bincount(bins, weights=pmf, minlength=n_bins)
        for b in range(n_bins):
            col = p_own * bin_prob[:, b]
            if np.any(col > 0):
                probs[f"z{own}_b{b}"] = col
    return probs, sorted(probs)


def _simulated_probabilities(
    A: np.ndarray,
    p_treat: float,
    mapping: Callable[[np.ndarray, np.ndarray], np.ndarray],
    n_sim: int,
    rng: np.random.Generator,
    n_treated: Optional[int] = None,
) -> Tuple[Dict[str, np.ndarray], List[str]]:
    """Monte-Carlo exposure probabilities for a user-supplied mapping."""
    n = A.shape[0]
    counts: Dict[str, np.ndarray] = {}
    base = np.zeros(n, dtype=int)
    if n_treated is not None:
        base[:n_treated] = 1
    for _ in range(n_sim):
        if n_treated is None:
            Z_sim = (rng.random(n) < p_treat).astype(int)
        else:
            Z_sim = rng.permutation(base)
        labels = np.asarray(mapping(Z_sim, A), dtype=object)
        for lab in np.unique(labels):
            counts.setdefault(lab, np.zeros(n))
            counts[lab] += labels == lab
    levels = sorted(counts)
    return {lab: counts[lab] / n_sim for lab in levels}, levels


# --------------------------------------------------------------------
# Variance
# --------------------------------------------------------------------


def _dependency_graph(A: np.ndarray) -> sparse.csr_matrix:
    """``G_ij = 1`` when some unit's treatment enters both exposures."""
    B = sparse.csr_matrix(A, dtype=np.int32) + sparse.identity(
        A.shape[0], dtype=np.int32, format="csr"
    )
    G = (B @ B.T).tocsr()
    G.data = np.ones_like(G.data, dtype=float)
    return G.astype(float)


def _quadratic_forms(G: sparse.csr_matrix, V: np.ndarray, variance: str) -> np.ndarray:
    """``V' G V`` (or its PSD-adjusted version) for the columns of ``V``."""
    if variance == "hac":
        return np.asarray(V.T @ (G @ V))
    lam, U = np.linalg.eigh(G.toarray())
    P = U.T @ V
    return np.asarray((P * np.clip(lam, 0.0, None)[:, None]).T @ P)


# --------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------

_AS4_CONTRASTS = {
    "direct (c10 - c00)": ("c10", "c00"),
    "spillover (c01 - c00)": ("c01", "c00"),
    "composite (c11 - c00)": ("c11", "c00"),
    "spillover_on_treated (c11 - c10)": ("c11", "c10"),
}


def network_exposure(
    Y: Sequence[float],
    Z: Sequence[int],
    adjacency: Any,
    *,
    mapping: Mapping = "as4",
    p_treat: Optional[float] = None,
    design: str = "bernoulli",
    n_sim: int = 2000,
    seed: Optional[int] = 0,
    estimator: str = "hajek",
    variance: str = "hac_psd",
    contrasts: Optional[Sequence[Tuple[str, str]]] = None,
    thresholds: Tuple[float, ...] = (0.0, 0.5),
    min_prob: float = 0.0,
    alpha: float = 0.05,
) -> NetworkExposureResult:
    """
    Exposure-mapping estimates of direct and spillover effects on a network.

    Parameters
    ----------
    Y : array-like (n,)
        Observed outcomes.
    Z : array-like (n,) of {0,1}
        Realised treatment assignment.
    adjacency : ndarray, sparse matrix, DataFrame, or list of edges
        Row ``i`` marks the units whose treatment may affect unit ``i``.
        An edge list is read as undirected. The diagonal is ignored.
    mapping : {"as4", "fraction"} or callable
        Exposure mapping. ``"as4"`` is the four-cell partition (own
        treatment by any treated neighbour). ``"fraction"`` bins the
        share of treated neighbours at ``thresholds``. A callable
        ``f(Z, A)`` returning one label per unit is also accepted; it
        must depend on ``Z`` only through a unit's own and its
        neighbours' treatments, and its exposure probabilities are
        simulated with ``n_sim`` draws.
    p_treat : float, optional
        Treatment probability of the design. Defaults to the realised
        share of treated units; pass the design value when it is known.
    design : {"bernoulli", "complete"}
        ``"bernoulli"``: independent assignment with probability
        ``p_treat``. ``"complete"``: a fixed number of units, the number
        observed in ``Z``, is treated, every such set being equally
        likely; exposure probabilities are then hypergeometric and
        ``p_treat`` is not used.
    n_sim : int, default 2000
        Draws used to simulate exposure probabilities of a callable
        mapping. Ignored for the built-in mappings, whose probabilities
        are exact.
    seed : int, optional
        Seed for those draws.
    estimator : {"hajek", "ht"}, default "hajek"
        ``"hajek"`` divides by the sum of the inverse-probability
        weights; ``"ht"`` is the Horvitz-Thompson mean, which is
        unbiased but not invariant to adding a constant to ``Y``.
    variance : {"hac_psd", "hac"}, default "hac_psd"
        ``"hac"`` sums products of linearised terms over pairs of units
        with dependent exposures. ``"hac_psd"`` uses the positive
        semidefinite part of the dependency graph instead, which makes
        the estimate conservative for the randomisation variance.
    contrasts : sequence of (str, str), optional
        Pairs ``(h1, h0)`` for which ``mean(h1) - mean(h0)`` is
        reported. Defaults to the four named contrasts of ``"as4"`` and
        to none for other mappings.
    thresholds : tuple of float, default (0.0, 0.5)
        Bin edges of the ``"fraction"`` mapping.
    min_prob : float, default 0.0
        Units with an exposure probability at or below this value for
        some level are excluded, so that averages are over units for
        which every exposure can occur with probability above it.
    alpha : float, default 0.05
        Confidence intervals have level ``1 - alpha``.

    Returns
    -------
    NetworkExposureResult
        ``estimates`` (one row per exposure) and ``contrasts``.

    Notes
    -----
    Averages are over *eligible* units, those for which every exposure
    level has probability above ``min_prob``. A unit without neighbours
    can never have a treated neighbour, so it is not eligible under
    ``"as4"``. A warning is issued when units are dropped and when some
    eligible unit has an exposure probability below 0.01, because the
    estimates then rest on a few heavily weighted units; the
    ``ess`` column reports the effective sample size at each level.

    The variance estimators are justified for Bernoulli designs under
    the assumption that the exposure mapping is correctly specified.
    Under ``design="complete"`` the same estimators are used. They treat
    the exposures of two units with no common source of treatment as
    independent, which leaves out the weak dependence that fixing the
    number treated induces between every pair. In simulations on a
    400-node network the ``"hac_psd"`` interval covered 97 to 98% and the
    ``"hac"`` interval 92 to 95%; there is no theorem behind this case.

    References
    ----------
    [@aronow2017estimating], [@leung2022causal], [@gao2025causal]

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np
    >>> rng = np.random.default_rng(0)
    >>> n = 60
    >>> A = np.zeros((n, n), dtype=int)  # ring network: two neighbours each
    >>> for i in range(n):
    ...     A[i, (i + 1) % n] = 1
    ...     A[i, (i - 1) % n] = 1
    >>> Z = (rng.random(n) < 0.5).astype(int)
    >>> Y = 1.0 + 2.0 * Z + 0.5 * (A @ Z) + rng.normal(size=n)
    >>> res = sp.network_exposure(Y, Z, A, p_treat=0.5)
    >>> res.estimates["exposure"].tolist()
    ['c00', 'c01', 'c10', 'c11']
    >>> res.contrasts["contrast"].tolist()  # doctest: +NORMALIZE_WHITESPACE
    ['direct (c10 - c00)', 'spillover (c01 - c00)',
     'composite (c11 - c00)', 'spillover_on_treated (c11 - c10)']
    """
    Y_arr = np.asarray(Y, dtype=float)
    Z_arr = np.asarray(Z, dtype=int)
    if Y_arr.ndim != 1 or Y_arr.shape != Z_arr.shape:
        raise MethodIncompatibility("Y and Z must be vectors of the same length")
    if not np.all(np.isfinite(Y_arr)):
        raise DataInsufficient("Y contains missing or non-finite values")
    if not np.isin(Z_arr, (0, 1)).all():
        raise MethodIncompatibility("Z must be coded 0/1")
    n = Y_arr.shape[0]
    A = _to_adj(adjacency, n)
    if A.shape[0] != n:
        raise MethodIncompatibility("adjacency size must match Y/Z length")
    if design not in ("bernoulli", "complete"):
        raise MethodIncompatibility("design must be 'bernoulli' or 'complete'")
    if estimator not in ("hajek", "ht"):
        raise MethodIncompatibility("estimator must be 'hajek' or 'ht'")
    if variance not in ("hac_psd", "hac"):
        raise MethodIncompatibility("variance must be 'hac_psd' or 'hac'")
    if not 0 < alpha < 1:
        raise MethodIncompatibility("alpha must be in (0, 1)")
    if not 0 <= min_prob < 1:
        raise MethodIncompatibility("min_prob must be in [0, 1)")

    n_treated: Optional[int] = None
    if design == "complete":
        # The number treated is fixed by the design, so the treatment
        # probability is the realised share.
        n_treated = int(Z_arr.sum())
        if p_treat is not None and abs(p_treat - n_treated / n) > 1e-9:
            raise MethodIncompatibility(
                "under design='complete' the treatment probability is the "
                f"realised share {n_treated / n:.6g}; omit p_treat"
            )
        p_treat = n_treated / n
    elif p_treat is None:
        p_treat = float(Z_arr.mean())
    if not (0 < p_treat < 1):
        raise MethodIncompatibility("p_treat must be in (0, 1)")

    deg = A.sum(axis=1)
    thresholds = tuple(float(t) for t in thresholds)
    if callable(mapping):
        mapping_name = getattr(mapping, "__name__", "custom")
        exposures = np.asarray(mapping(Z_arr, A), dtype=object)
        if exposures.shape != (n,):
            raise MethodIncompatibility(
                "a callable mapping must return one label per unit"
            )
        probs, levels = _simulated_probabilities(
            A, p_treat, mapping, int(n_sim), np.random.default_rng(seed), n_treated
        )
        for lev in np.unique(exposures):
            if lev not in probs:
                # Realised but never simulated: its probability is below
                # the resolution of the simulation.
                probs[lev] = np.zeros(n)
                levels = sorted(levels + [lev])
        simulated = True
    elif mapping == "as4":
        mapping_name = "as4"
        exposures = _as4_mapping(Z_arr, A)
        probs, levels = _as4_probabilities(deg, p_treat, n_treated)
        simulated = False
    elif mapping == "fraction":
        mapping_name = "fraction"
        exposures = _fraction_mapping(Z_arr, A, thresholds)
        probs, levels = _fraction_probabilities(deg, p_treat, thresholds, n_treated)
        simulated = False
    else:
        raise MethodIncompatibility(
            "mapping must be 'as4', 'fraction' or a callable f(Z, A)"
        )

    P = np.column_stack([probs[lev] for lev in levels])
    eligible = (P > min_prob).all(axis=1)
    n_el = int(eligible.sum())
    if n_el < 2:
        raise DataInsufficient(
            "fewer than two units can receive every exposure level; "
            "check the network or coarsen the exposure mapping"
        )
    if n_el < n:
        warnings.warn(
            f"{n - n_el} of {n} units cannot receive every exposure level "
            f"with probability above {min_prob:g} (for example units "
            "without neighbours) and are excluded; estimates are averages "
            f"over the remaining {n_el} units.",
            AssumptionWarning,
            stacklevel=2,
        )
    P_el = P[eligible]
    if min_prob == 0.0 and float(P_el.min()) < 0.01:
        warnings.warn(
            "some units have an exposure probability below 0.01 (smallest: "
            f"{float(P_el.min()):.2e}), so a few units carry very large "
            "weights. Consider min_prob= to restrict to units with adequate "
            "overlap, or a coarser exposure mapping.",
            AssumptionWarning,
            stacklevel=2,
        )
    Y_el = Y_arr[eligible]
    H = np.column_stack([exposures[eligible] == lev for lev in levels]).astype(float)
    Gamma = H / P_el
    if variance == "hac_psd" and n_el > _PSD_MAX_N:
        raise MethodIncompatibility(
            f"variance='hac_psd' needs an eigendecomposition of a "
            f"{n_el} x {n_el} matrix; use variance='hac' for networks "
            f"with more than {_PSD_MAX_N} eligible units"
        )
    # Dependence can run through an excluded unit, so the dependency graph
    # is built on the full network and then restricted.
    G = _dependency_graph(A)
    if n_el < n:
        G = G[eligible][:, eligible].tocsr()

    # Means and linearised terms, one column per level.
    sum_w = Gamma.sum(axis=0)
    realised = H.sum(axis=0)
    mu = np.full(len(levels), np.nan)
    V = np.zeros_like(Gamma)
    for k in range(len(levels)):
        if realised[k] == 0:
            continue
        if estimator == "hajek":
            mu[k] = float(Gamma[:, k] @ Y_el / sum_w[k])
            V[:, k] = Gamma[:, k] * (Y_el - mu[k]) / (sum_w[k] / n_el)
        else:
            mu[k] = float(Gamma[:, k] @ Y_el / n_el)
            V[:, k] = Gamma[:, k] * Y_el
    Q = _quadratic_forms(G, V, variance) / n_el**2
    crit = float(stats.norm.ppf(1 - alpha / 2))

    rows = []
    for k, lev in enumerate(levels):
        se = float(np.sqrt(max(Q[k, k], 0.0))) if realised[k] > 0 else np.nan
        w = Gamma[:, k]
        ess = float(w.sum() ** 2 / (w @ w)) if realised[k] > 0 else 0.0
        rows.append(
            {
                "exposure": lev,
                "mean_Y(d)": mu[k],
                "se": se,
                "ci_lo": mu[k] - crit * se,
                "ci_hi": mu[k] + crit * se,
                "n_at_level": int(realised[k]),
                "min_prob": float(P_el[:, k].min()),
                "ess": ess,
            }
        )
    est = pd.DataFrame(rows)

    if contrasts is None:
        named = dict(_AS4_CONTRASTS) if mapping_name == "as4" else {}
    else:
        named = {}
        for pair in contrasts:
            if len(pair) != 2 or any(h not in levels for h in pair):
                raise MethodIncompatibility(
                    f"contrast {pair!r} must name two of the exposure "
                    f"levels {levels}"
                )
            named[f"{pair[0]} - {pair[1]}"] = (pair[0], pair[1])
    index = {lev: k for k, lev in enumerate(levels)}
    contrasts_rows = []
    for label, (a, b) in named.items():
        ia, ib = index[a], index[b]
        if realised[ia] == 0 or realised[ib] == 0:
            continue
        est_d = float(mu[ia] - mu[ib])
        var_d = float(Q[ia, ia] + Q[ib, ib] - 2 * Q[ia, ib])
        se_d = float(np.sqrt(max(var_d, 0.0)))
        z = est_d / se_d if se_d > 0 else np.nan
        contrasts_rows.append(
            {
                "contrast": label,
                "estimate": est_d,
                "se": se_d,
                "pvalue": float(2 * stats.norm.sf(abs(z))),
                "ci_lo": est_d - crit * se_d,
                "ci_hi": est_d + crit * se_d,
            }
        )
    contrasts_df = pd.DataFrame(
        contrasts_rows,
        columns=["contrast", "estimate", "se", "pvalue", "ci_lo", "ci_hi"],
    )

    _result = NetworkExposureResult(
        estimates=est,
        contrasts=contrasts_df,
        exposure_levels=list(levels),
        n_obs=n,
        p_treat=p_treat,
        design=design,
        mapping=mapping_name,
        detail={
            "adjacency_density": float(A.sum() / (n * (n - 1)) if n > 1 else 0.0),
            "max_degree": int(deg.max()) if n else 0,
            "dependency_graph_max_degree": int(G.getnnz(axis=1).max()),
            "exposure_probabilities": "simulated" if simulated else "exact",
        },
        estimator=estimator,
        variance=variance,
        n_eligible=n_el,
        alpha=alpha,
    )
    try:
        from ..output._lineage import attach_provenance as _attach_prov

        _attach_prov(
            _result,
            function="sp.interference.network_exposure",
            params={
                "mapping": mapping_name,
                "p_treat": p_treat,
                "design": design,
                "n_sim": n_sim,
                "seed": seed,
                "estimator": estimator,
                "variance": variance,
                "min_prob": min_prob,
                "alpha": alpha,
            },
            data=None,
            overwrite=False,
        )
    except Exception:  # pragma: no cover
        pass
    return _result


__all__ = ["network_exposure", "NetworkExposureResult"]
