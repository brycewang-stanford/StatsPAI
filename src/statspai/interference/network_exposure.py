"""
Aronow-Samii (2017) network exposure mappings and Horvitz-Thompson
estimators for causal effects under arbitrary interference.

Setup
-----
* :math:`N` units with binary treatment vector :math:`Z \\in \\{0,1\\}^N`
  drawn from a known design (Bernoulli or completely-randomized).
* An adjacency matrix / graph :math:`G` defining peers.
* An *exposure mapping* :math:`f: (Z, i, G) \\to \\mathcal{D}` that
  reduces the full treatment vector to a low-dimensional categorical
  exposure (e.g. ``"isolated_treated"``, ``"spillover_only"``, ...).

Average potential outcomes per exposure level :math:`d` are estimated
with the Horvitz-Thompson formula::

    \\hat\\mu(d) = (1/N) Σ_i  1{f(Z, i, G) = d} · Y_i / π_i(d)

where :math:`π_i(d) = \\Pr(f(Z, i, G) = d)` is the *exposure
probability*. Under the design, π_i(d) is computed in closed form for
Bernoulli designs and by Monte-Carlo simulation otherwise.

Standard errors follow the conservative variance bound from
Aronow & Samii (2017, Theorem 1):

    \\hat V(\\hat\\mu(d)) = (1/N²) Σ_i  Y_i² · (1 - π_i(d)) / π_i(d)²

The default mapping ``"as4"`` returns the 4-cell map from the paper:

* ``c00`` : Z_i = 0, no treated neighbours
* ``c10`` : Z_i = 1, no treated neighbours  (direct effect)
* ``c01`` : Z_i = 0, ≥1 treated neighbour   (pure spillover)
* ``c11`` : Z_i = 1, ≥1 treated neighbour   (composite)

References
----------
Aronow, P. M., & Samii, C. (2017).
"Estimating average causal effects under general interference, with
application to a social network experiment." *Annals of Applied
Statistics*, 11(4), 1912-1947. [@aronow2017estimating]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence, Tuple, Any

import numpy as np
import pandas as pd
from scipy import stats


# --------------------------------------------------------------------
# Result container
# --------------------------------------------------------------------


@dataclass
class NetworkExposureResult:
    estimates: pd.DataFrame  # one row per exposure level
    contrasts: pd.DataFrame  # pairwise contrasts (e.g. direct, spillover)
    exposure_levels: List[str]
    n_obs: int
    p_treat: float
    design: str
    mapping: str
    detail: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:  # pragma: no cover
        return (
            f"Network Exposure HT estimates ({self.mapping}, {self.design})\n"
            f"  n = {self.n_obs}, p = {self.p_treat:.3f}\n"
            f"{self.estimates.to_string(index=False)}\n\n"
            "Contrasts:\n"
            f"{self.contrasts.to_string(index=False)}"
        )

    def __repr__(self) -> str:  # pragma: no cover
        return f"NetworkExposureResult(levels={self.exposure_levels})"


# --------------------------------------------------------------------
# Adjacency helpers
# --------------------------------------------------------------------

def _to_adj(adj_or_edges, n: Optional[int] = None) -> np.ndarray:
    """Coerce adjacency input (matrix or edge list) into a binary numpy matrix."""
    if isinstance(adj_or_edges, np.ndarray):
        A = adj_or_edges.astype(int)
    elif isinstance(adj_or_edges, pd.DataFrame):
        A = adj_or_edges.to_numpy().astype(int)
    elif isinstance(adj_or_edges, (list, tuple, np.generic)):
        edges = np.asarray(list(adj_or_edges))
        if edges.ndim != 2 or edges.shape[1] != 2:
            raise ValueError("edge list must be (n_edges, 2)")
        if n is None:
            n = int(edges.max()) + 1
        A = np.zeros((n, n), dtype=int)
        for u, v in edges:
            A[int(u), int(v)] = 1
            A[int(v), int(u)] = 1
    else:
        raise TypeError("adjacency must be ndarray / DataFrame / edge-list")
    np.fill_diagonal(A, 0)
    if A.shape[0] != A.shape[1]:
        raise ValueError("adjacency must be square")
    return A


def _as4_mapping(Z: np.ndarray, A: np.ndarray) -> np.ndarray:
    """Aronow-Samii 4-cell exposure: own treatment x any-neighbor-treated."""
    has_t_neigh = (A @ Z) > 0
    out = np.empty(Z.shape[0], dtype=object)
    for i in range(Z.shape[0]):
        if Z[i] == 1 and has_t_neigh[i]:
            out[i] = "c11"
        elif Z[i] == 1:
            out[i] = "c10"
        elif has_t_neigh[i]:
            out[i] = "c01"
        else:
            out[i] = "c00"
    return out


def _fraction_mapping(Z: np.ndarray, A: np.ndarray, thresholds=(0.0, 0.5)) -> np.ndarray:
    """Bin own treatment x fraction of treated neighbours."""
    deg = A.sum(axis=1).astype(float)
    frac = np.where(deg > 0, (A @ Z) / np.maximum(deg, 1), 0.0)
    own = Z.astype(int)
    bin_ = np.digitize(frac, thresholds)  # 0 < t0 <=, 1, 2
    out = np.array([f"z{own[i]}_b{bin_[i]}" for i in range(Z.shape[0])], dtype=object)
    return out


def _exposure_probabilities(
    A: np.ndarray,
    p_treat: float,
    mapping: Callable[[np.ndarray, np.ndarray], np.ndarray],
    n_sim: int = 2000,
    rng: Optional[np.random.Generator] = None,
) -> Tuple[Dict[str, np.ndarray], List[str]]:
    """
    Monte-Carlo estimate of π_i(d) = P(f(Z, i, G) = d) under a
    Bernoulli design with treatment probability ``p_treat``.
    """
    rng = rng or np.random.default_rng(0)
    n = A.shape[0]
    counts: Dict[str, np.ndarray] = {}
    for _ in range(n_sim):
        Z_sim = (rng.random(n) < p_treat).astype(int)
        labels = mapping(Z_sim, A)
        for lab in np.unique(labels):
            counts.setdefault(lab, np.zeros(n))
            counts[lab] += (labels == lab).astype(float)
    levels = sorted(counts.keys())
    probs = {lab: counts[lab] / n_sim for lab in levels}
    return probs, levels


def _ht_estimate(
    Y: np.ndarray, exposures: np.ndarray, probs: Dict[str, np.ndarray],
    levels: List[str], min_pi: float = 1e-3,
) -> pd.DataFrame:
    n = Y.shape[0]
    rows = []
    for lev in levels:
        pi = np.clip(probs[lev], min_pi, 1.0)
        ind = (exposures == lev).astype(float)
        weights = ind / pi
        mu = float(np.mean(weights * Y))
        # Conservative Aronow-Samii Theorem 1 bound:
        #   V̂(μ̂(d)) = (1/N²) Σ_i Y_i² · (1 - π_i(d)) / π_i(d)²
        # The prior ``var = ...`` line (pre-v1.5) was dead code — its
        # own return value was overwritten by the next line and the
        # formula itself was dimensionally inconsistent.  Removed.
        var_as = float(np.sum((Y ** 2) * (1 - pi) / np.maximum(pi ** 2, 1e-12)) / n ** 2)
        se = float(np.sqrt(max(var_as, 0.0)))
        ci_lo = mu - 1.96 * se
        ci_hi = mu + 1.96 * se
        rows.append({
            "exposure": lev,
            "mean_Y(d)": mu,
            "se": se,
            "ci_lo": ci_lo,
            "ci_hi": ci_hi,
            "n_at_level": int((exposures == lev).sum()),
        })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------


def network_exposure(
    Y: Sequence[float],
    Z: Sequence[int],
    adjacency,
    *,
    mapping: str = "as4",
    p_treat: Optional[float] = None,
    design: str = "bernoulli",
    n_sim: int = 2000,
    seed: Optional[int] = 0,
) -> NetworkExposureResult:
    """
    Aronow-Samii Horvitz-Thompson estimator for arbitrary interference.

    Parameters
    ----------
    Y : array-like (n,)
        Observed outcomes.
    Z : array-like (n,) of {0,1}
        Realised treatment assignment.
    adjacency : ndarray, DataFrame, or list of edges
        Network adjacency. Diagonal is zeroed.
    mapping : {"as4", "fraction"}
        Exposure mapping. ``"as4"`` is the 4-cell Aronow-Samii partition
        (no neighbour treated × own treatment). ``"fraction"`` bins by
        share of treated neighbours.
    p_treat : float, optional
        Treatment probability for the design. Defaults to the empirical
        share of treated units.
    design : {"bernoulli"}
        Only Bernoulli is supported in this minimal release; for
        completely-randomised designs use ``p_treat = K/N``.
    n_sim : int, default 2000
        Monte-Carlo replicates used to estimate exposure probabilities.
    seed : int, optional
        RNG seed.

    Returns
    -------
    NetworkExposureResult
        Per-exposure HT means and SEs, plus contrasts for
        ``direct = mu(c10) - mu(c00)`` and ``spillover = mu(c01) - mu(c00)``
        when the AS4 mapping is used.
    """
    Y = np.asarray(Y, dtype=float)
    Z = np.asarray(Z, dtype=int)
    if Y.shape != Z.shape:
        raise ValueError("Y and Z must have the same length")
    n = Y.shape[0]
    A = _to_adj(adjacency, n)
    if A.shape[0] != n:
        raise ValueError("adjacency size must match Y/Z length")
    if design != "bernoulli":
        raise NotImplementedError("Only 'bernoulli' design is implemented")

    if p_treat is None:
        p_treat = float(Z.mean())
    if not (0 < p_treat < 1):
        raise ValueError("p_treat must be in (0, 1)")

    if mapping == "as4":
        map_fn = _as4_mapping
    elif mapping == "fraction":
        map_fn = _fraction_mapping
    else:
        raise ValueError("mapping must be 'as4' or 'fraction'")

    rng = np.random.default_rng(seed)
    exposures = map_fn(Z, A)
    probs, levels = _exposure_probabilities(A, p_treat, map_fn, n_sim=n_sim, rng=rng)

    # Make sure realised levels appear (in case MC missed rare ones).
    for lev in np.unique(exposures):
        if lev not in probs:
            probs[lev] = np.full(n, 1.0 / n_sim)
            levels.append(lev)
    levels = sorted(levels)

    est = _ht_estimate(Y, exposures, probs, levels)

    # Pairwise contrasts (for AS4)
    contrasts_rows = []
    if mapping == "as4":
        m = est.set_index("exposure")["mean_Y(d)"].to_dict()
        s = est.set_index("exposure")["se"].to_dict()

        def diff(a, b):
            if a in m and b in m:
                est_d = m[a] - m[b]
                se_d = float(np.sqrt(s[a] ** 2 + s[b] ** 2))
                z = est_d / se_d if se_d > 0 else 0.0
                return est_d, se_d, float(2 * stats.norm.sf(abs(z)))
            return None

        named = {
            "direct (c10 - c00)": ("c10", "c00"),
            "spillover (c01 - c00)": ("c01", "c00"),
            "composite (c11 - c00)": ("c11", "c00"),
            "spillover_on_treated (c11 - c10)": ("c11", "c10"),
        }
        for label, (a, b) in named.items():
            d = diff(a, b)
            if d is None:
                continue
            est_d, se_d, p = d
            contrasts_rows.append({
                "contrast": label,
                "estimate": est_d,
                "se": se_d,
                "pvalue": p,
                "ci_lo": est_d - 1.96 * se_d,
                "ci_hi": est_d + 1.96 * se_d,
            })

    contrasts = pd.DataFrame(contrasts_rows)

    return NetworkExposureResult(
        estimates=est,
        contrasts=contrasts,
        exposure_levels=levels,
        n_obs=n,
        p_treat=p_treat,
        design=design,
        mapping=mapping,
        detail={"adjacency_density": float(A.sum() / (n * (n - 1)) if n > 1 else 0.0)},
    )


__all__ = ["network_exposure", "NetworkExposureResult"]
