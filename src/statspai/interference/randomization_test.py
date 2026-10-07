"""
Randomization tests for treatment effects and spillovers on a network.

Two nested null hypotheses are covered.

``null="no_effect"``
    Treatment affects no one. Every outcome is the same under every
    assignment, so the test statistic can be recomputed under any
    re-randomization of the whole treatment vector (Fisher's test).

``null="no_spillover"``
    A unit's outcome depends only on its own treatment. Outcomes of a
    set of *focal* units are then unchanged by any re-randomization that
    leaves the focal units' own treatments fixed, so a statistic computed
    from focal outcomes can be recomputed after permuting the treatments
    of the remaining units.

The focal set must be chosen without reference to the realised
assignment (it is drawn at random from a seed here, or supplied by the
user); the permutation p-value is then valid in finite samples for any
test statistic.

References
----------
[@aronow2012general], [@athey2018exact], [@basse2019randomization]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations, product
from math import comb
from typing import Any, Callable, Dict, Optional, Sequence, Union

import numpy as np

from .._result_serialize import ResultProtocolMixin
from ..exceptions import DataInsufficient, MethodIncompatibility
from .network_exposure import _to_adj

Statistic = Union[
    str, Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], float]
]


@dataclass
class InterferenceTestResult(ResultProtocolMixin):
    """Result of :func:`interference_test`.

    Attributes
    ----------
    statistic : float
        Test statistic on the realised assignment.
    pvalue : float
        Permutation p-value ``(1 + #{T_b at least as extreme}) / (1 + B)``,
        or the exact proportion when every assignment was enumerated.
    null : str
        The hypothesis tested.
    n_focal : int
        Number of focal units (all units for ``null="no_effect"``).
    n_perm : int
        Number of alternative assignments evaluated.
    exact : bool
        Whether all admissible assignments were enumerated.

    Examples
    --------
    >>> import numpy as np
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> Z = np.array([1, 0] * 6)
    >>> Y = rng.normal(size=12) + 2.0 * Z
    >>> r = sp.interference_test(Y, Z, null="no_effect")
    >>> isinstance(r, sp.InterferenceTestResult), r.exact
    (True, True)
    """

    _citation_keys = ("aronow2012general", "athey2018exact", "basse2019randomization")

    statistic: float
    pvalue: float
    null: str
    statistic_name: str
    alternative: str
    n_obs: int
    n_focal: int
    n_perm: int
    exact: bool
    focal: np.ndarray = field(repr=False, default_factory=lambda: np.zeros(0, int))
    null_distribution: np.ndarray = field(
        repr=False, default_factory=lambda: np.zeros(0)
    )
    detail: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:  # pragma: no cover
        kind = "exact" if self.exact else "Monte Carlo"
        return (
            f"Randomization test of H0: {self.null}\n"
            f"  statistic ({self.statistic_name}) = {self.statistic:.6g}\n"
            f"  p-value ({self.alternative}, {kind}, {self.n_perm} "
            f"assignments) = {self.pvalue:.4g}\n"
            f"  n = {self.n_obs}, focal units = {self.n_focal}"
        )


def _ols_last_coefficient(X: np.ndarray, y: np.ndarray) -> float:
    beta, *_ = np.linalg.lstsq(X, y, rcond=None)
    return float(beta[-1])


def _neighbor_share(Z: np.ndarray, A: np.ndarray) -> np.ndarray:
    deg = A.sum(axis=1)
    return np.where(deg > 0, (A @ Z) / np.maximum(deg, 1), 0.0)


def _stat_difference_in_means(
    Y: np.ndarray, Z: np.ndarray, A: np.ndarray, focal: np.ndarray
) -> float:
    y, z = Y[focal], Z[focal]
    if z.min() == z.max():
        return np.nan
    return float(y[z == 1].mean() - y[z == 0].mean())


def _make_exposure_statistic(
    kind: str, A: np.ndarray
) -> Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], float]:
    """Coefficient on an exposure measure in a regression of focal outcomes.

    The regression always holds own treatment fixed; the measures that
    test a richer hypothesis also hold the share of treated neighbours
    fixed, so that the coefficient picks up only what the hypothesis
    rules out.
    """
    deg = A.sum(axis=1).astype(float)
    if kind == "weighted_share":
        # Neighbours weighted by their own number of neighbours.
        W = A * deg[None, :]
        w_tot = W.sum(axis=1)
    elif kind == "second_order_share":
        second = ((A @ A) > 0).astype(int)
        second[A > 0] = 0
        np.fill_diagonal(second, 0)
        deg2 = second.sum(axis=1).astype(float)

    def stat(Y: np.ndarray, Z: np.ndarray, A_: np.ndarray, focal: np.ndarray) -> float:
        share = _neighbor_share(Z, A)
        controls = []
        if kind == "neighbor_share":
            expo = share[focal]
        elif kind == "any_neighbor":
            expo = ((A @ Z) > 0).astype(float)[focal]
        elif kind == "weighted_share":
            expo = (np.where(w_tot > 0, (W @ Z) / np.maximum(w_tot, 1), 0.0))[focal]
            controls.append(share[focal])
        else:
            expo = (np.where(deg2 > 0, (second @ Z) / np.maximum(deg2, 1), 0.0))[focal]
            controls.append(share[focal])
        own = Z[focal].astype(float)
        cols = [np.ones(focal.shape[0])]
        if own.min() != own.max():
            cols.append(own)
        cols.extend(c for c in controls if np.ptp(c) > 0)
        cols.append(expo)
        if np.ptp(expo) == 0:
            return 0.0
        return _ols_last_coefficient(np.column_stack(cols), Y[focal])

    return stat


def interference_test(
    Y: Sequence[float],
    Z: Sequence[int],
    adjacency: Any = None,
    *,
    null: str = "no_spillover",
    statistic: Optional[Statistic] = None,
    focal: Union[None, float, Sequence[int], np.ndarray] = None,
    alternative: str = "two-sided",
    n_perm: int = 2000,
    seed: Optional[int] = 0,
) -> InterferenceTestResult:
    """
    Randomization test for treatment effects or spillovers on a network.

    Parameters
    ----------
    Y : array-like (n,)
        Observed outcomes.
    Z : array-like (n,) of {0,1}
        Realised treatment assignment, drawn independently across units
        or by fixing the number treated.
    adjacency : ndarray, sparse matrix, DataFrame, or list of edges
        Row ``i`` marks the units whose treatment may affect unit ``i``.
        Required for every hypothesis except ``"no_effect"``.
    null : {"no_spillover", "no_effect", "anonymous", "no_higher_order"}
        The hypothesis tested, from most to least restrictive:

        * ``"no_effect"``: treatment affects no outcome at all.
        * ``"no_spillover"`` (default): an outcome depends only on the
          unit's own treatment.
        * ``"anonymous"``: an outcome depends only on the unit's own
          treatment and on the *share* of its neighbours that are
          treated, not on which ones.
        * ``"no_higher_order"``: an outcome depends only on the
          treatments of the unit and of its neighbours, not on units
          further away.
    statistic : {"neighbor_share", "any_neighbor", "difference_in_means"} or callable
        ``"neighbor_share"`` (default for ``"no_spillover"``) is the
        coefficient on the share of treated neighbours in a regression
        of focal outcomes on own treatment and that share;
        ``"any_neighbor"`` uses an indicator for at least one treated
        neighbour instead. ``"difference_in_means"`` (default for
        ``"no_effect"``) is the treated-minus-control mean. A callable
        ``T(Y, Z, A, focal)`` returning a number is also accepted; it
        must use the outcomes of the focal units only. For
        ``"anonymous"`` the default is ``"weighted_share"``, the
        coefficient on the share of treated neighbours weighted by each
        neighbour's own number of neighbours, given own treatment and
        the unweighted share. For ``"no_higher_order"`` it is
        ``"second_order_share"``, the coefficient on the share treated
        among neighbours of neighbours, given own treatment and the
        share of treated neighbours.
    focal : float or array-like, optional
        The share of units drawn at random (from ``seed``, never from
        ``Z``) as focal units, or the indices of the focal units, which
        must have been chosen without looking at the realised
        assignment. The default share is 0.5 for ``"no_spillover"`` and
        0.15 for ``"no_higher_order"``. For ``"anonymous"`` focal units
        are picked in random order so that no two share a neighbour or
        are neighbours themselves, up to that share (default: as many as
        fit).
    alternative : {"two-sided", "greater", "less"}
    n_perm : int, default 2000
        Number of alternative assignments. When the number of admissible
        assignments does not exceed ``n_perm`` they are all enumerated
        and the p-value is exact.
    seed : int, optional
        Seed for the focal draw and the permutations.

    Returns
    -------
    InterferenceTestResult

    Notes
    -----
    Alternative assignments permute the treatments of the units the
    hypothesis leaves free: all units for ``"no_effect"``; the non-focal
    units for ``"no_spillover"``; the units that are neither focal nor
    neighbours of a focal unit for ``"no_higher_order"``; and, for
    ``"anonymous"``, the neighbours of each focal unit among themselves,
    which keeps every focal unit's share of treated neighbours fixed.
    Each permutation keeps the number treated fixed. Under independent
    assignment the conditional distribution
    given that number is uniform over such permutations, so the test is
    valid for both Bernoulli and completely randomized designs. It is
    not valid for designs with unequal assignment probabilities or with
    clustering.

    The hypotheses are nested, so testing them from the most restrictive
    down and stopping at the first that is not rejected needs no
    multiplicity correction.

    A larger focal set makes the statistic more precise but leaves fewer
    treatments to permute; the defaults are a compromise. The tests of
    ``"anonymous"`` and ``"no_higher_order"`` hold fixed far more of the
    assignment than the test of ``"no_spillover"`` and have
    correspondingly less power: failing to reject them is weak evidence.

    References
    ----------
    [@aronow2012general], [@athey2018exact], [@basse2019randomization]

    Examples
    --------
    >>> import numpy as np
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> n = 200
    >>> A = np.zeros((n, n), dtype=int)
    >>> for i in range(n):
    ...     A[i, (i + 1) % n] = A[(i + 1) % n, i] = 1
    >>> Z = (rng.random(n) < 0.5).astype(int)
    >>> Y = 1.0 * Z + 2.0 * (A @ Z > 0) + rng.normal(size=n)
    >>> r = sp.interference_test(Y, Z, A, n_perm=999, seed=1)
    >>> bool(r.pvalue < 0.05)
    True
    """
    Y_arr = np.asarray(Y, dtype=float)
    Z_arr = np.asarray(Z, dtype=int)
    if Y_arr.ndim != 1 or Y_arr.shape != Z_arr.shape:
        raise MethodIncompatibility("Y and Z must be vectors of the same length")
    if not np.all(np.isfinite(Y_arr)):
        raise DataInsufficient("Y contains missing or non-finite values")
    if not np.isin(Z_arr, (0, 1)).all():
        raise MethodIncompatibility("Z must be coded 0/1")
    nulls = ("no_spillover", "no_effect", "anonymous", "no_higher_order")
    if null not in nulls:
        raise MethodIncompatibility(f"null must be one of {list(nulls)}")
    if alternative not in ("two-sided", "greater", "less"):
        raise MethodIncompatibility(
            "alternative must be 'two-sided', 'greater' or 'less'"
        )
    if n_perm < 1:
        raise MethodIncompatibility("n_perm must be at least 1")
    n = Y_arr.shape[0]
    rng = np.random.default_rng(seed)

    if adjacency is None:
        if null != "no_effect":
            raise MethodIncompatibility(f"null={null!r} requires adjacency=")
        A = np.zeros((n, n), dtype=int)
    else:
        A = _to_adj(adjacency, n)
        if A.shape[0] != n:
            raise MethodIncompatibility("adjacency size must match Y/Z length")

    def draw_focal(default_share: float) -> np.ndarray:
        if focal is None or np.isscalar(focal):
            share = default_share
            if focal is not None:
                share = float(focal)  # type: ignore[arg-type]
            if not 0 < share < 1:
                raise MethodIncompatibility(
                    "focal must be a share in (0, 1) or an array of indices"
                )
            n_focal = int(round(share * n))
            if n_focal < 2 or n - n_focal < 2:
                raise DataInsufficient(
                    "too few units to split into focal and non-focal sets"
                )
            return np.asarray(np.sort(rng.choice(n, size=n_focal, replace=False)))
        idx = np.unique(np.asarray(focal, dtype=int))
        if idx.size == 0 or idx.min() < 0 or idx.max() >= n:
            raise MethodIncompatibility("focal indices must lie in [0, n)")
        return np.asarray(idx)

    # ``groups``: sets of units whose treatments are permuted among
    # themselves. The hypothesis says the focal outcomes do not change
    # under any such permutation.
    focal_idx: np.ndarray
    groups: list
    if null == "no_effect":
        focal_idx = np.arange(n)
        groups = [np.arange(n)]
    elif null == "no_spillover":
        focal_idx = draw_focal(0.5)
        mask = np.ones(n, dtype=bool)
        mask[focal_idx] = False
        groups = [np.flatnonzero(mask)]
    elif null == "no_higher_order":
        focal_idx = draw_focal(0.15)
        mask = np.ones(n, dtype=bool)
        mask[focal_idx] = False
        mask[A[focal_idx].sum(axis=0) > 0] = False
        groups = [np.flatnonzero(mask)]
    else:  # anonymous
        if focal is not None and not np.isscalar(focal):
            focal_idx = draw_focal(0.5)
            closed = A[focal_idx] + np.eye(n, dtype=int)[focal_idx]
            if (closed.sum(axis=0) > 1).any():
                raise MethodIncompatibility(
                    "for null='anonymous' no two focal units may be "
                    "neighbours or share a neighbour"
                )
        else:
            limit = n
            if focal is not None:
                limit = int(round(float(focal) * n))  # type: ignore[arg-type]
            taken = np.zeros(n, dtype=bool)
            chosen = []
            for i in rng.permutation(n):
                hood = np.flatnonzero(A[i])
                if hood.size < 2 or taken[i] or taken[hood].any():
                    continue
                chosen.append(int(i))
                taken[i] = True
                taken[hood] = True
                if len(chosen) >= limit:
                    break
            focal_idx = np.asarray(sorted(chosen), dtype=int)
        if focal_idx.size < 3:
            raise DataInsufficient(
                "fewer than three focal units with disjoint neighbourhoods "
                "of at least two units; the network is too dense or too small"
            )
        groups = [np.flatnonzero(A[i]) for i in focal_idx]

    if statistic is None:
        statistic = {
            "no_effect": "difference_in_means",
            "no_spillover": "neighbor_share",
            "anonymous": "weighted_share",
            "no_higher_order": "second_order_share",
        }[null]
    if callable(statistic):
        stat_fn = statistic
        stat_name = getattr(statistic, "__name__", "custom")
    elif statistic == "difference_in_means":
        stat_fn, stat_name = _stat_difference_in_means, statistic
    elif statistic in (
        "neighbor_share",
        "any_neighbor",
        "weighted_share",
        "second_order_share",
    ):
        if adjacency is None:
            raise MethodIncompatibility(f"statistic={statistic!r} requires adjacency=")
        stat_fn, stat_name = _make_exposure_statistic(statistic, A), statistic
    else:
        raise MethodIncompatibility(
            "statistic must be 'neighbor_share', 'any_neighbor', "
            "'weighted_share', 'second_order_share', 'difference_in_means' "
            "or a callable"
        )

    # Only groups with both treated and untreated units can change.
    groups = [g for g in groups if 0 < int(Z_arr[g].sum()) < g.size]
    if not groups:
        raise DataInsufficient(
            "the units whose treatment is re-randomized are all treated or "
            "all untreated, so there is nothing to permute"
        )
    free = np.concatenate(groups)
    t_obs = float(stat_fn(Y_arr, Z_arr, A, focal_idx))
    if not np.isfinite(t_obs):
        raise DataInsufficient(
            "the test statistic is undefined on the realised assignment"
        )

    n_assignments = 1
    for g in groups:
        n_assignments *= comb(g.size, int(Z_arr[g].sum()))
        if n_assignments - 1 > n_perm:
            break
    exact = n_assignments - 1 <= n_perm
    draws = []
    if exact:
        choices = [combinations(range(g.size), int(Z_arr[g].sum())) for g in groups]
        for pick in product(*choices):
            z_new = Z_arr.copy()
            for g, treated in zip(groups, pick):
                z_new[g] = 0
                z_new[g[list(treated)]] = 1
            draws.append(float(stat_fn(Y_arr, z_new, A, focal_idx)))
    else:
        for _ in range(int(n_perm)):
            z_new = Z_arr.copy()
            for g in groups:
                z_new[g] = rng.permutation(Z_arr[g])
            draws.append(float(stat_fn(Y_arr, z_new, A, focal_idx)))
    t_null = np.asarray(draws)
    # A draw on which the statistic is undefined counts against rejection.
    undefined = ~np.isfinite(t_null)
    tol = 1e-12 * max(1.0, abs(t_obs))
    if alternative == "two-sided":
        extreme = np.abs(t_null) >= abs(t_obs) - tol
    elif alternative == "greater":
        extreme = t_null >= t_obs - tol
    else:
        extreme = t_null <= t_obs + tol
    extreme = extreme | undefined
    if exact:
        # The enumeration includes the realised assignment itself.
        pvalue = float(extreme.mean())
    else:
        pvalue = float((1 + extreme.sum()) / (1 + t_null.size))

    return InterferenceTestResult(
        statistic=t_obs,
        pvalue=pvalue,
        null=null,
        statistic_name=stat_name,
        alternative=alternative,
        n_obs=n,
        n_focal=int(focal_idx.size),
        n_perm=int(t_null.size),
        exact=bool(exact),
        focal=focal_idx,
        null_distribution=t_null,
        detail={
            "n_rerandomized": int(free.size),
            "n_treated_rerandomized": int(Z_arr[free].sum()),
            "n_permutation_groups": len(groups),
            "n_undefined_draws": int(undefined.sum()),
        },
    )


__all__ = ["interference_test", "InterferenceTestResult"]
