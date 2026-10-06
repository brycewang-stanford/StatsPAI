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
from itertools import combinations
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
    kind: str,
) -> Callable[[np.ndarray, np.ndarray, np.ndarray, np.ndarray], float]:
    def stat(Y: np.ndarray, Z: np.ndarray, A: np.ndarray, focal: np.ndarray) -> float:
        if kind == "neighbor_share":
            expo = _neighbor_share(Z, A)[focal]
        else:
            expo = ((A @ Z) > 0).astype(float)[focal]
        own = Z[focal].astype(float)
        cols = [np.ones(focal.shape[0])]
        if own.min() != own.max():
            cols.append(own)
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
    focal: Union[float, Sequence[int], np.ndarray] = 0.5,
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
        Required for ``null="no_spillover"``.
    null : {"no_spillover", "no_effect"}, default "no_spillover"
        ``"no_spillover"`` tests that outcomes depend only on a unit's
        own treatment; ``"no_effect"`` tests that treatment affects no
        outcome at all.
    statistic : {"neighbor_share", "any_neighbor", "difference_in_means"} or callable
        ``"neighbor_share"`` (default for ``"no_spillover"``) is the
        coefficient on the share of treated neighbours in a regression
        of focal outcomes on own treatment and that share;
        ``"any_neighbor"`` uses an indicator for at least one treated
        neighbour instead. ``"difference_in_means"`` (default for
        ``"no_effect"``) is the treated-minus-control mean. A callable
        ``T(Y, Z, A, focal)`` returning a number is also accepted; it
        must use the outcomes of the focal units only.
    focal : float or array-like, default 0.5
        For ``null="no_spillover"``: the share of units drawn at random
        (from ``seed``, never from ``Z``) as focal units, or the indices
        of the focal units. Indices must have been chosen without
        looking at the realised assignment.
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
    Alternative assignments permute the treatments of the non-focal
    units (all units for ``"no_effect"``), which keeps the number treated
    fixed. Under independent assignment the conditional distribution
    given that number is uniform over such permutations, so the test is
    valid for both Bernoulli and completely randomized designs. It is
    not valid for designs with unequal assignment probabilities or with
    clustering.

    The two hypotheses are nested, so testing ``"no_effect"`` first and
    ``"no_spillover"`` only if it is rejected needs no multiplicity
    correction.

    A larger focal set makes the statistic more precise but leaves fewer
    treatments to permute; the default of one half is a compromise.

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
    if null not in ("no_spillover", "no_effect"):
        raise MethodIncompatibility("null must be 'no_spillover' or 'no_effect'")
    if alternative not in ("two-sided", "greater", "less"):
        raise MethodIncompatibility(
            "alternative must be 'two-sided', 'greater' or 'less'"
        )
    if n_perm < 1:
        raise MethodIncompatibility("n_perm must be at least 1")
    n = Y_arr.shape[0]
    rng = np.random.default_rng(seed)

    if adjacency is None:
        if null == "no_spillover":
            raise MethodIncompatibility("null='no_spillover' requires adjacency=")
        A = np.zeros((n, n), dtype=int)
    else:
        A = _to_adj(adjacency, n)
        if A.shape[0] != n:
            raise MethodIncompatibility("adjacency size must match Y/Z length")

    focal_idx: np.ndarray
    free: np.ndarray
    if null == "no_effect":
        focal_idx = np.arange(n)
        free = np.arange(n)
    else:
        if np.isscalar(focal):
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
            focal_idx = np.sort(rng.choice(n, size=n_focal, replace=False))
        else:
            focal_idx = np.unique(np.asarray(focal, dtype=int))
            if focal_idx.size == 0 or focal_idx.min() < 0 or focal_idx.max() >= n:
                raise MethodIncompatibility("focal indices must lie in [0, n)")
        mask = np.ones(n, dtype=bool)
        mask[focal_idx] = False
        free = np.flatnonzero(mask)

    if statistic is None:
        statistic = "difference_in_means" if null == "no_effect" else "neighbor_share"
    if callable(statistic):
        stat_fn = statistic
        stat_name = getattr(statistic, "__name__", "custom")
    elif statistic == "difference_in_means":
        stat_fn, stat_name = _stat_difference_in_means, statistic
    elif statistic in ("neighbor_share", "any_neighbor"):
        if adjacency is None:
            raise MethodIncompatibility(f"statistic={statistic!r} requires adjacency=")
        stat_fn, stat_name = _make_exposure_statistic(statistic), statistic
    else:
        raise MethodIncompatibility(
            "statistic must be 'neighbor_share', 'any_neighbor', "
            "'difference_in_means' or a callable"
        )

    z_free = Z_arr[free]
    n_treated_free = int(z_free.sum())
    if n_treated_free == 0 or n_treated_free == free.size:
        raise DataInsufficient(
            "the units whose treatment is re-randomized are all treated or "
            "all untreated, so there is nothing to permute"
        )
    t_obs = float(stat_fn(Y_arr, Z_arr, A, focal_idx))
    if not np.isfinite(t_obs):
        raise DataInsufficient(
            "the test statistic is undefined on the realised assignment"
        )

    n_assignments = comb(free.size, n_treated_free)
    exact = n_assignments - 1 <= n_perm
    draws = []
    if exact:
        for treated in combinations(range(free.size), n_treated_free):
            z_new = Z_arr.copy()
            z_new[free] = 0
            z_new[free[list(treated)]] = 1
            draws.append(float(stat_fn(Y_arr, z_new, A, focal_idx)))
    else:
        for _ in range(int(n_perm)):
            z_new = Z_arr.copy()
            z_new[free] = rng.permutation(z_free)
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
            "n_treated_rerandomized": n_treated_free,
            "n_undefined_draws": int(undefined.sum()),
        },
    )


__all__ = ["interference_test", "InterferenceTestResult"]
