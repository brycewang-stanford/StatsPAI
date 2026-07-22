"""
High-Dimensional Fixed Effects (HDFE) absorber.

Native Python/NumPy implementation of within-transformation (mean-sweep) for
regressions absorbing many high-cardinality fixed effects. Functionally
mirrors Stata's ``reghdfe`` (Correia 2017) and R's ``fixest::feols``
(Bergé 2018) core routine.

Algorithm
---------
The within-transformation on multiple FE groups ``G_1, ..., G_K`` is the
orthogonal projection onto the complement of the span of K group-indicator
matrices. When K = 1 this reduces to de-meaning; when K >= 2 no closed
form exists and alternating projections (method of alternating
projections, von Neumann 1933) are used:

    repeat:
        for k in 1..K:
            x <- x - mean_{G_k}(x)
    until ||dx|| < tol

This converges linearly. To get industrial speed we use the Irons-Tuck /
Aitken scalar acceleration (Δ²), which cuts iteration count by 3-10x for
the typical panel case. See Correia (2017) §2.3 for details.

Varying slopes
--------------
Besides ordinary FE dimensions, the absorber handles *varying-slope*
terms — Stata's ``absorb(i.g#c.x)`` / ``absorb(i.g##c.x)`` and fixest's
``g[[x]]`` / ``g[x]``. These absorb the columns ``x · 1[g = j]`` (one
slope per level of ``g``), optionally alongside the level dummies. The
corresponding projection residualizes ``x``-wise *within* each level
rather than de-meaning, and participates in the alternating projections
on the same footing as an ordinary FE. See :class:`SlopeSpec` and
``_hdfe_kernels.sweep_slope``.

Singleton detection
-------------------
Observations whose FE group has only one observation do not contribute to
within-variation and bias degrees-of-freedom. Iterative singleton pruning
(Correia 2015) removes them. We run a single-pass prune; subsequent rounds
only matter for K > 3 in pathological data.

Exported API
------------
``Absorber``: class that holds FE columns + weights, with ``demean`` /
``residualize`` methods.
``demean``: functional convenience wrapper.
``absorb_ols``: solve an OLS with absorbed FEs in one call (returns
coefs, SE, absorbed residuals, FE-adjusted R²).

References
----------
Correia, S. (2015). "Singletons, Cluster-Robust Standard Errors and
Fixed Effects." Working paper.
Correia, S. (2017). "Linear Models with High-Dimensional Fixed Effects."
Working paper. https://scorreia.com/research/hdfe.pdf
Bergé, L. (2018). "Efficient estimation of maximum likelihood models
with multiple fixed-effects: the R package FENmlm." CREA DP 13.
Gaure, S. (2013). "OLS with multiple high dimensional category variables."
Computational Statistics & Data Analysis, 66, 8-18.
Guimarães, P. and Portugal, P. (2010). "A simple feasible procedure to
fit models with high-dimensional fixed effects." Stata Journal, 10(4).
"""

from __future__ import annotations

import warnings
from typing import Any, List, NamedTuple, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

_VALID_SOLVERS = ("map", "lsmr", "lsqr")

# Relative tolerance below which a per-level slope normal equation counts as
# rank-deficient. Compared against that level's own uncentered second moment
# ``Σ w x²``, so it is scale-free in the units of ``x``.
_SLOPE_RANK_TOL = 1e-12


def _hdfe_kernels() -> Any:
    """Load Numba HDFE kernels on first HDFE use, not on package import."""
    from . import _hdfe_kernels as _kernels

    return _kernels


# ======================================================================
# Core demean kernel
# ======================================================================


def _factorize(fe: np.ndarray) -> Tuple[np.ndarray, int]:
    """Return integer codes in [0, G) and the group count G.

    Handles both numeric and object/string arrays. ``pd.factorize`` is
    used for speed and NaN-safety.
    """
    codes, uniq = pd.factorize(fe, sort=False, use_na_sentinel=True)
    if (codes < 0).any():
        raise ValueError(
            "HDFE: NaN values in fixed-effect column are not allowed."
        )  # pragma: no cover
    return codes.astype(np.int64), len(uniq)


def _factorize_multi(cols: Sequence[np.ndarray]) -> Tuple[np.ndarray, int]:
    """Factorize the *combination* of several label columns.

    Returns dense codes in ``[0, G)`` identifying each distinct observed
    tuple, plus ``G``.

    Implemented by combining integer codes in mixed radix rather than by
    joining string representations. String joining is not safe here: a
    separator that can appear in the data merges distinct groups, and
    ``pd.factorize`` truncates object strings at an embedded NUL byte, so
    even ``"\\0"`` — the usual "impossible" separator — silently collapses
    ``("0", "1")`` and ``("0", "2")`` into one group. Re-factorizing after
    every merge keeps the running code range at most ``n``, so the
    ``codes * G_k + c_k`` product cannot overflow int64.
    """
    codes, _ = _factorize(np.asarray(cols[0]))
    for c in cols[1:]:
        ck, Gk = _factorize(np.asarray(c))
        codes, _ = _factorize(codes * Gk + ck)
    G = int(codes.max()) + 1 if codes.size else 0
    return codes, G


def _group_mean_sweep(
    x: np.ndarray,
    codes: np.ndarray,
    counts: np.ndarray,
    weights: Optional[np.ndarray] = None,
    wsum: Optional[np.ndarray] = None,
) -> None:
    """In-place de-mean x by group codes. 2D x supported (column-wise).

    Delegates the per-column pass to the Numba-accelerated kernels in
    :mod:`_hdfe_kernels` when Numba is installed; otherwise falls back
    to a pure-NumPy ``bincount`` path. Weighted and unweighted variants
    share the same dispatch.
    """
    kernels = _hdfe_kernels()
    if x.ndim == 1:
        col = np.ascontiguousarray(x)
        if weights is None:
            kernels.sweep(col, codes, counts)
        else:
            kernels.sweep_weighted(col, weights, codes, wsum)
        if col is not x:
            x[:] = col
    else:
        if weights is None:
            for j in range(x.shape[1]):
                col = np.ascontiguousarray(x[:, j])
                kernels.sweep(col, codes, counts)
                x[:, j] = col
        else:
            for j in range(x.shape[1]):
                col = np.ascontiguousarray(x[:, j])
                kernels.sweep_weighted(col, weights, codes, wsum)
                x[:, j] = col


def _map_ap(
    x: np.ndarray,
    fe_codes: List[np.ndarray],
    counts_list: List[np.ndarray],
    weights: Optional[np.ndarray],
    wsum_list: Optional[List[np.ndarray]],
) -> None:
    """One full alternating-projection sweep over all FE dimensions (in place).

    Currently unused — :meth:`Absorber.demean` drives the AP loop through
    :func:`_group_mean_sweep_seq`. Note that this helper handles ordinary
    FE dimensions only; it does **not** apply varying-slope terms, so do
    not wire it back in without adding a ``slope_ops`` pass.
    """
    K = len(fe_codes)
    for k in range(K):
        _group_mean_sweep(
            x,
            fe_codes[k],
            counts_list[k],
            weights=weights,
            wsum=wsum_list[k] if wsum_list is not None else None,
        )


def _aitken_accelerate(x0: np.ndarray, x1: np.ndarray, x2: np.ndarray) -> np.ndarray:
    """Irons-Tuck / vector Aitken-like Δ² acceleration.

    Given three successive iterates x0, x1, x2 of a contraction map
    ``T``, return the accelerated value

        x_acc = x0 - α · (x1 - x0)

    where ``α = <dx1, d2> / <d2, d2>`` (the Minimum-Polynomial
    Extrapolation / Irons-Tuck scalar, Smith-Ford-Sidi 1987). This is
    numerically safer than element-wise Aitken (no per-element
    division blow-up) and equivalent in the limit. See Varadhan-Roland
    (2008) §3.

    Falls back to x2 if the denominator is near-zero (no acceleration
    signal).
    """
    dx1 = x1 - x0
    d2 = x2 - 2.0 * x1 + x0
    denom = float(d2 @ d2)
    if denom < 1e-30:
        return x2
    alpha = float(dx1 @ d2) / denom
    return np.asarray(x0 - alpha * dx1)


# ======================================================================
# Varying slopes
# ======================================================================


class SlopeSpec(NamedTuple):
    """A varying-slope absorbed term.

    Represents Stata's ``absorb(i.g#c.x)`` / ``absorb(i.g##c.x)`` and
    fixest's ``| g[[x]]`` / ``| g[x]``.

    Attributes
    ----------
    group : ndarray (n,)
        Raw group labels (numeric or object). Factorized internally.
    x : ndarray (n,)
        Continuous variable whose coefficient varies across levels of
        ``group``.
    with_intercept : bool
        False → absorb only the ``G`` slope columns ``x · 1[g = j]``
        (Stata ``#``, fixest ``[[x]]``). True → additionally absorb the
        ``G`` level intercepts ``1[g = j]`` (Stata ``##``, fixest
        ``[x]``).
    name : str
        Display name, used in diagnostics and warnings.

    Examples
    --------
    >>> import numpy as np
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> g = np.repeat(np.arange(4), 25)
    >>> x = rng.normal(size=100)
    >>> spec = sp.SlopeSpec(group=g, x=x, with_intercept=False, name="i.g#c.x")
    >>> spec.with_intercept
    False
    >>> spec.name
    'i.g#c.x'
    """

    group: np.ndarray
    x: np.ndarray
    with_intercept: bool = False
    name: str = "slope"


class _SlopeOp(NamedTuple):
    """Pre-factorized, pre-conditioned slope projector (internal)."""

    codes: np.ndarray
    x: np.ndarray
    gsum: np.ndarray
    xsum: np.ndarray
    inv_denom: np.ndarray
    with_intercept: bool
    n_levels: int
    n_degenerate: int
    name: str


def _slope_stats(
    codes: np.ndarray,
    x: np.ndarray,
    n_levels: int,
    with_intercept: bool,
    weights: Optional[np.ndarray],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Per-level sufficient statistics for a varying-slope projection.

    Returns ``(gsum, xsum, inv_denom, n_degenerate)`` where ``gsum`` is
    ``Σ w`` (or the group count), ``xsum`` is ``Σ w·x``, and ``inv_denom``
    is the reciprocal of the normal-equation denominator with ``0.0``
    marking rank-deficient levels.

    The denominator is ``Σ w·x²`` for a slope-only term and the *centered*
    ``Σ w·x² − (Σ w·x)² / Σ w`` when level intercepts are also absorbed.
    A level is declared rank-deficient when that denominator collapses
    relative to the level's own uncentered second moment — which is
    exactly the single-observation and zero-within-level-variance cases.
    """
    w = np.ones_like(x) if weights is None else weights
    gsum = np.bincount(codes, weights=w, minlength=n_levels)
    xsum = np.bincount(codes, weights=w * x, minlength=n_levels)
    xxsum = np.bincount(codes, weights=w * x * x, minlength=n_levels)

    if with_intercept:
        with np.errstate(divide="ignore", invalid="ignore"):
            denom = xxsum - np.where(
                gsum > 0, xsum**2 / np.where(gsum > 0, gsum, 1.0), 0.0
            )
    else:
        denom = xxsum

    # Rank test relative to the level's own uncentered second moment: a level
    # with all-zero x (xxsum == 0) or with x collinear with the intercept
    # (denom ≈ 0) is degenerate. Never divides by ~0.
    ok = denom > _SLOPE_RANK_TOL * xxsum
    inv_denom = np.zeros(n_levels, dtype=np.float64)
    np.divide(1.0, denom, out=inv_denom, where=ok)
    n_degenerate = int((~ok).sum())
    return (
        np.ascontiguousarray(gsum, dtype=np.float64),
        np.ascontiguousarray(xsum, dtype=np.float64),
        inv_denom,
        n_degenerate,
    )


def _slope_sweep(
    col: np.ndarray,
    op: _SlopeOp,
    weights: Optional[np.ndarray],
) -> None:
    """Apply one varying-slope projection to ``col`` in place (1D)."""
    kernels = _hdfe_kernels()
    if weights is None:
        kernels.sweep_slope(
            col,
            op.x,
            op.codes,
            op.gsum,
            op.xsum,
            op.inv_denom,
            op.with_intercept,
        )
    else:
        kernels.sweep_slope_weighted(
            col,
            op.x,
            weights,
            op.codes,
            op.gsum,
            op.xsum,
            op.inv_denom,
            op.with_intercept,
        )


# ======================================================================
# Singleton pruning
# ======================================================================


def _detect_singletons(
    fe: np.ndarray,
    fe_codes_raw: List[np.ndarray],
) -> np.ndarray:
    """Iteratively drop singleton observations (groups of size 1).

    Each pass: mark obs whose code in any dimension appears only once; if
    any found, drop them and recount. Repeat until stable. Returns the
    keep mask (True = keep).

    Worst-case quadratic in number of passes but in practice converges in
    <= 3 passes even on nested panels (Correia 2015).
    """
    n = fe.shape[0]
    keep = np.ones(n, dtype=bool)
    K = len(fe_codes_raw)

    while True:
        dropped = False
        for k in range(K):
            codes_k = fe_codes_raw[k][keep]
            counts_k = np.bincount(codes_k)
            single_groups = np.where(counts_k == 1)[0]
            if single_groups.size == 0:
                continue
            single_mask_local = np.isin(codes_k, single_groups)
            if single_mask_local.any():
                # Map back to global indices
                global_idx = np.where(keep)[0]
                keep[global_idx[single_mask_local]] = False
                dropped = True
        if not dropped:
            break
    return keep


# ======================================================================
# Absorber class
# ======================================================================


class Absorber:
    """Reusable HDFE demean operator.

    Build once from a DataFrame's FE columns; reuse ``demean`` to sweep any
    outcome / regressor vector or matrix. Useful when fitting many models
    that share the same absorbing FEs (e.g. event-study coefficient paths).

    Parameters
    ----------
    fe_data : DataFrame, ndarray (n, K), or None
        FE columns. Must have no NaN. May be ``None`` (or an ``(n, 0)``
        array) when only varying-slope terms are absorbed, in which case
        ``n_obs`` is inferred from ``slopes``.
    weights : ndarray (n,), optional
        Observation weights. If given, weighted means are used.
    drop_singletons : bool, default True
        If True, singleton observations (FE groups of size 1) are pruned
        before building the absorber. ``keep_mask`` stores the surviving
        rows.
    tol : float, default 1e-8
        Convergence threshold on max |dx| per iteration.
    maxiter : int, default 10000
        Maximum alternating-projection iterations.
    accelerate : bool, default True
        Enable Irons-Tuck Δ² acceleration.
    solver : {"map", "lsmr", "lsqr"}, default "map"
        Within-transformation backend. ``"map"`` uses alternating
        projections with Irons-Tuck acceleration (default, typically
        fastest on well-conditioned panels). ``"lsmr"`` / ``"lsqr"``
        delegate to ``scipy.sparse.linalg.lsmr`` / ``lsqr`` on the
        sparse FE design matrix — more robust for ill-conditioned or
        highly nested FE structures. See the migration guide for how
        this maps to ``pyreghdfe``.
    slopes : sequence of SlopeSpec, optional
        Varying-slope terms to absorb alongside ``fe_data``. Each counts
        as one more absorbed dimension in the alternating projections.
    n_obs : int, optional
        Row count, required only when ``fe_data`` is ``None`` and
        ``slopes`` is empty.

    Attributes
    ----------
    keep_mask : ndarray of bool
        Rows retained after singleton pruning. Callers must apply this
        mask to ``y``, ``X``, and any weights before passing to
        ``demean``.
    n_kept : int
        Number of surviving observations.
    n_dropped : int
        Number of singleton observations removed.
    n_fe : list of int
        Number of groups per FE dimension (post-prune).
    slope_ops : list
        Pre-conditioned varying-slope projectors (post-prune). Each
        exposes ``n_levels`` and ``n_degenerate`` (levels whose
        within-level design is rank-deficient and absorbs nothing).
    n_slope_levels : list of int
        Number of levels per varying-slope term (post-prune).

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> n = 300
    >>> fe_data = pd.DataFrame({
    ...     "firm": rng.integers(0, 20, n),
    ...     "year": rng.integers(0, 5, n),
    ... })
    >>> y = (fe_data["firm"] * 0.5 + fe_data["year"] * 0.3
    ...      + rng.normal(0, 1, n))
    >>> absorber = sp.Absorber(fe_data)
    >>> y_within = absorber.demean(y.to_numpy())  # keep_mask applied
    >>> absorber.n_fe
    [20, 5]
    >>> bool(abs(y_within.mean()) < 1e-6)
    True
    """

    __slots__ = (
        "fe_codes",
        "counts_list",
        "wsum_list",
        "weights",
        "keep_mask",
        "n_kept",
        "n_dropped",
        "n_fe",
        "slope_ops",
        "n_slope_levels",
        "tol",
        "maxiter",
        "accelerate",
        "solver",
        "_converged",
        "_iters",
    )

    def __init__(
        self,
        fe_data: Union[pd.DataFrame, np.ndarray, None],
        weights: Optional[np.ndarray] = None,
        drop_singletons: bool = True,
        tol: float = 1e-8,
        maxiter: int = 10_000,
        accelerate: bool = True,
        solver: str = "map",
        slopes: Optional[Sequence[SlopeSpec]] = None,
        n_obs: Optional[int] = None,
    ) -> None:
        if solver not in _VALID_SOLVERS:
            raise ValueError(
                f"solver={solver!r} invalid; expected one of {_VALID_SOLVERS}."
            )
        slope_specs: List[SlopeSpec] = list(slopes) if slopes else []

        if fe_data is None:
            if n_obs is None:
                if not slope_specs:
                    raise ValueError(
                        "HDFE: at least one fixed-effect column or one slope "
                        "term is required."
                    )
                n_obs = int(np.asarray(slope_specs[0].group).shape[0])
            fe_arr = np.empty((int(n_obs), 0))
        elif isinstance(fe_data, pd.DataFrame):
            fe_arr = fe_data.values
        else:
            fe_arr = np.asarray(fe_data)
            if fe_arr.ndim == 1:
                fe_arr = fe_arr.reshape(-1, 1)

        n, K = fe_arr.shape
        if K == 0 and not slope_specs:
            raise ValueError(
                "HDFE: at least one fixed-effect column or one slope term is "
                "required."
            )
        for spec in slope_specs:
            if np.asarray(spec.group).shape[0] != n or np.asarray(spec.x).shape[0] != n:
                raise ValueError(
                    f"HDFE: slope term {spec.name!r} has length "
                    f"{np.asarray(spec.group).shape[0]} but the absorber was "
                    f"built on {n} rows."
                )

        # Factorize each FE column
        fe_codes_raw: List[np.ndarray] = []
        for k in range(K):
            codes_k, _ = _factorize(fe_arr[:, k])
            fe_codes_raw.append(codes_k)

        # The categorical part of a slope term participates in singleton
        # detection exactly like an ordinary FE — this is what reghdfe does
        # (``absorb(i.g#c.x)`` on a panel with one-observation ``g`` levels
        # reports those rows in ``e(num_singletons)``).
        slope_codes_raw: List[np.ndarray] = [
            _factorize(np.asarray(spec.group))[0] for spec in slope_specs
        ]

        # Singleton pruning
        if drop_singletons:
            keep_mask = _detect_singletons(fe_arr, fe_codes_raw + slope_codes_raw)
        else:
            keep_mask = np.ones(n, dtype=bool)
        n_kept = int(keep_mask.sum())
        n_dropped = n - n_kept

        # Re-factorize on kept rows so codes are dense in [0, G)
        fe_codes: List[np.ndarray] = []
        counts_list: List[np.ndarray] = []
        n_fe: List[int] = []
        for codes_k in fe_codes_raw:
            codes_kept = codes_k[keep_mask]
            dense, uniq = pd.factorize(codes_kept, sort=False, use_na_sentinel=True)
            dense = dense.astype(np.int64)
            G = len(uniq)
            counts = np.bincount(dense, minlength=G).astype(np.float64)
            fe_codes.append(dense)
            counts_list.append(counts)
            n_fe.append(G)

        # Weighted group sums, pre-compute wsum per group
        wsum_list: Optional[List[np.ndarray]]
        self.weights: Optional[np.ndarray]
        if weights is not None:
            w_kept = np.ascontiguousarray(weights[keep_mask], dtype=np.float64)
            _wsum: List[np.ndarray] = []
            for codes_k, G in zip(fe_codes, n_fe):
                _wsum.append(np.bincount(codes_k, weights=w_kept, minlength=G))
            wsum_list = _wsum
            self.weights = w_kept
        else:
            wsum_list = None
            self.weights = None

        # Build the slope projectors on the surviving rows.
        slope_ops: List[_SlopeOp] = []
        for spec, codes_raw in zip(slope_specs, slope_codes_raw):
            codes_kept = codes_raw[keep_mask]
            dense, uniq = pd.factorize(codes_kept, sort=False, use_na_sentinel=True)
            dense = np.ascontiguousarray(dense, dtype=np.int64)
            G = len(uniq)
            x_kept = np.ascontiguousarray(
                np.asarray(spec.x, dtype=np.float64)[keep_mask]
            )
            if not np.all(np.isfinite(x_kept)):
                raise ValueError(
                    f"HDFE: slope term {spec.name!r} has non-finite values in "
                    "its continuous variable; drop or impute them first."
                )
            gsum, xsum, inv_denom, n_deg = _slope_stats(
                dense, x_kept, G, spec.with_intercept, self.weights
            )
            if n_deg:
                warnings.warn(
                    f"HDFE: slope term {spec.name!r} has {n_deg} of {G} level(s) "
                    "with a rank-deficient within-level design (single "
                    "observation, or no within-level variation in the "
                    "continuous variable). Those levels absorb nothing and are "
                    "excluded from dof_fe; the slope is not identified there.",
                    UserWarning,
                    stacklevel=2,
                )
            slope_ops.append(
                _SlopeOp(
                    codes=dense,
                    x=x_kept,
                    gsum=gsum,
                    xsum=xsum,
                    inv_denom=inv_denom,
                    with_intercept=bool(spec.with_intercept),
                    n_levels=G,
                    n_degenerate=n_deg,
                    name=spec.name,
                )
            )

        self.fe_codes = fe_codes
        self.counts_list = counts_list
        self.wsum_list = wsum_list
        self.keep_mask = keep_mask
        self.n_kept = n_kept
        self.n_dropped = n_dropped
        self.n_fe = n_fe
        self.slope_ops = slope_ops
        self.n_slope_levels = [op.n_levels for op in slope_ops]
        self.tol = tol
        self.maxiter = maxiter
        self.accelerate = accelerate
        self.solver = solver
        self._converged = False
        self._iters = 0

    # ------------------------------------------------------------------
    # Demean
    # ------------------------------------------------------------------

    def demean(
        self,
        x: np.ndarray,
        copy: bool = True,
        already_masked: bool = False,
    ) -> np.ndarray:
        """Within-transform ``x`` by sweeping out all absorbed FEs.

        Parameters
        ----------
        x : ndarray, shape (n,) or (n, p)
            Variable(s) to residualize. ``n`` must equal either the full
            input size (then ``keep_mask`` is applied) or ``n_kept``
            (when ``already_masked=True``).
        copy : bool, default True
            If True, operate on a copy; if False, modify ``x`` in place.
            Callers passing fresh arrays can set False to save memory.
        already_masked : bool, default False
            Skip application of ``keep_mask``.

        Returns
        -------
        ndarray
            Residualized ``x`` with shape ``(n_kept,)`` or ``(n_kept, p)``.
        """
        x = np.asarray(x, dtype=np.float64)
        if not already_masked:
            x = x[self.keep_mask]
        if copy:
            x = x.copy()
        if x.ndim == 1:
            x = x.reshape(-1, 1)
            squeeze = True
        else:
            squeeze = False

        fe_codes = self.fe_codes
        counts_list = self.counts_list
        wsum_list = self.wsum_list
        weights = self.weights
        slope_ops = self.slope_ops
        K = len(fe_codes)
        # A varying-slope term is one more absorbed dimension in the
        # alternating projection, exactly like an ordinary FE. An
        # intercept-bearing slope term still counts as ONE dimension because
        # the [1, x] within-level fit is done jointly (see _hdfe_kernels).
        n_dims = K + len(slope_ops)

        # Single absorbed dimension: closed-form, no iteration needed.
        if n_dims == 1:
            if K == 1:
                _group_mean_sweep(
                    x,
                    fe_codes[0],
                    counts_list[0],
                    weights,
                    wsum_list[0] if wsum_list else None,
                )
            else:
                for j in range(x.shape[1]):
                    col = np.ascontiguousarray(x[:, j])
                    _slope_sweep(col, slope_ops[0], weights)
                    x[:, j] = col
            self._converged = True
            self._iters = 1
            return x.ravel() if squeeze else x

        # K>=2: alternating projections with optional Irons-Tuck acceleration
        tol = self.tol
        maxiter = self.maxiter
        accelerate = self.accelerate

        # Krylov solvers (LSMR / LSQR) bypass the AP loop entirely: build the
        # sparse FE design matrix once and delegate the within-projection to
        # scipy. See ``_solve_krylov`` for the √w weight handling.
        if self.solver != "map":
            for j in range(x.shape[1]):
                col = x[:, j]
                r, iters, converged = _solve_krylov(
                    col,
                    fe_codes,
                    weights,
                    solver=self.solver,
                    tol=tol,
                    maxiter=maxiter,
                    slope_ops=slope_ops,
                )
                x[:, j] = r
                self._iters = max(self._iters, iters)
                self._converged = self._converged or converged
            return x.ravel() if squeeze else x

        # Per-column AP loop
        for j in range(x.shape[1]):
            col = x[:, j].copy()  # work on a copy to avoid in-place surprises
            base_scale = np.max(np.abs(col)) + 1e-30

            # Standard AP loop with periodic Irons-Tuck acceleration.
            # Every ACCEL_PERIOD sweeps, apply the vector-Aitken jump built
            # from three consecutive iterates (classic SQUAREM layout).
            accel_period = 5
            col_hist: list = []
            converged = False
            for it in range(maxiter):
                col_before = col.copy()
                _group_mean_sweep_seq(
                    col, fe_codes, counts_list, weights, wsum_list, slope_ops
                )
                dx = np.max(np.abs(col - col_before)) / base_scale
                if dx < tol:
                    converged = True
                    self._iters = max(self._iters, it + 1)
                    break
                if accelerate:
                    col_hist.append(col.copy())
                    if len(col_hist) >= 3 and (it + 1) % accel_period == 0:
                        col_acc = _aitken_accelerate(
                            col_hist[-3], col_hist[-2], col_hist[-1]
                        )
                        # Only accept the jump if it does not blow up
                        if np.max(np.abs(col_acc)) < 10 * base_scale:
                            col = col_acc
                        col_hist = []
            self._converged = self._converged or converged
            if not converged:
                self._iters = maxiter
            x[:, j] = col

        return x.ravel() if squeeze else x

    # ------------------------------------------------------------------
    # Residualize (alias + sanity)
    # ------------------------------------------------------------------

    def residualize(self, x: np.ndarray, copy: bool = True) -> np.ndarray:
        """Alias for ``demean`` — returns FE-residualized version of x."""
        return self.demean(x, copy=copy)

    def __repr__(self) -> str:
        slope = (
            f", slopes={[(op.name, op.n_levels) for op in self.slope_ops]}"
            if self.slope_ops
            else ""
        )
        return (
            f"Absorber(K={len(self.fe_codes)}, n_kept={self.n_kept}, "
            f"n_dropped={self.n_dropped}, groups={self.n_fe}{slope})"
        )


def _group_mean_sweep_seq(
    col: np.ndarray,
    fe_codes: List[np.ndarray],
    counts_list: List[np.ndarray],
    weights: Optional[np.ndarray],
    wsum_list: Optional[List[np.ndarray]],
    slope_ops: Optional[Sequence[_SlopeOp]] = None,
) -> None:
    """One full sequential sweep over all absorbed dimensions (in place, 1D).

    Ordinary FE dimensions are swept first, then any varying-slope terms.
    Alternating projections converge to the projection onto the orthogonal
    complement of the *union* of the absorbed spans regardless of the order
    in which the individual projectors are applied (von Neumann 1933), so
    slope terms participate on exactly the same footing as plain FEs.

    Dispatches to :mod:`_hdfe_kernels` for Numba-accelerated kernels.
    """
    kernels = _hdfe_kernels()
    K = len(fe_codes)
    if weights is None:
        for k in range(K):
            kernels.sweep(col, fe_codes[k], counts_list[k])
    else:
        assert wsum_list is not None
        for k in range(K):
            kernels.sweep_weighted(col, weights, fe_codes[k], wsum_list[k])
    if slope_ops:
        for op in slope_ops:
            _slope_sweep(col, op, weights)


# ======================================================================
# Krylov solvers (LSMR / LSQR) — pyreghdfe-compatible path
# ======================================================================


def _build_fe_design(
    fe_codes: List[np.ndarray],
    n_rows: int,
    slope_ops: Optional[Sequence[_SlopeOp]] = None,
) -> Any:
    """Horizontally stack the absorbed design blocks into a sparse CSR.

    Each FE dimension contributes an ``(n_rows, G_k)`` indicator block. A
    varying-slope term contributes an ``(n_rows, G)`` block holding ``x``
    in the level's column instead of ``1`` — and, when the term also
    absorbs level intercepts, a plain indicator block alongside it.
    """
    from scipy import sparse as _sp

    blocks = []
    rows = np.arange(n_rows)
    for codes in fe_codes:
        G = int(codes.max()) + 1
        data = np.ones(n_rows, dtype=np.float64)
        blocks.append(_sp.csr_matrix((data, (rows, codes)), shape=(n_rows, G)))
    for op in slope_ops or ():
        if op.with_intercept:
            blocks.append(
                _sp.csr_matrix(
                    (np.ones(n_rows), (rows, op.codes)), shape=(n_rows, op.n_levels)
                )
            )
        blocks.append(
            _sp.csr_matrix((op.x, (rows, op.codes)), shape=(n_rows, op.n_levels))
        )
    return _sp.hstack(blocks, format="csr")


def _solve_krylov(
    x: np.ndarray,
    fe_codes: List[np.ndarray],
    weights: Optional[np.ndarray],
    solver: str,
    tol: float,
    maxiter: int,
    slope_ops: Optional[Sequence[_SlopeOp]] = None,
) -> Tuple[np.ndarray, int, bool]:
    """One-column within-transformation via scipy.sparse.linalg.

    Solves ``min_α ‖W^{1/2}(x − D α)‖₂`` and returns the residual
    ``r = x − D α*`` in the **original (unweighted) scale** so that the
    downstream FWL OLS matches the MAP path byte-for-byte.
    """
    from scipy.sparse.linalg import lsmr, lsqr

    n = x.shape[0]
    D = _build_fe_design(fe_codes, n, slope_ops)

    if weights is not None:
        sw = np.sqrt(weights)
        # Row-scale D by sqrt(w); equivalent to left-multiplying by diag(sw).
        D_solve = D.multiply(sw[:, None]).tocsr()
        x_solve = sw * x
    else:
        D_solve = D
        x_solve = x

    if solver == "lsmr":
        out = lsmr(D_solve, x_solve, atol=tol, btol=tol, maxiter=maxiter)
        alpha = out[0]
        istop = out[1]
        iters = out[2]
    elif solver == "lsqr":
        out = lsqr(D_solve, x_solve, atol=tol, btol=tol, iter_lim=maxiter)
        alpha = out[0]
        istop = out[1]
        iters = out[2]
    else:  # pragma: no cover — Absorber.__init__ guards against this.
        raise ValueError(f"solver={solver!r} invalid.")

    # istop == 7 in both lsmr and lsqr means maxiter reached without convergence.
    converged = istop != 7
    # Residual in the ORIGINAL (unweighted) scale — this is what the caller
    # feeds into OLS. Using the weighted residual here would double-apply √w.
    r = x - D @ alpha
    return r, int(iters), converged


# ======================================================================
# Functional API
# ======================================================================


def demean(
    x: np.ndarray,
    fe: Union[pd.DataFrame, np.ndarray, None],
    weights: Optional[np.ndarray] = None,
    drop_singletons: bool = True,
    tol: float = 1e-8,
    maxiter: int = 10_000,
    solver: str = "map",
    slopes: Optional[Sequence[SlopeSpec]] = None,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return the within-transformed ``x`` and the singleton keep mask.

    Convenience wrapper around :class:`Absorber`. See ``Absorber`` for
    the ``solver`` kwarg semantics.

    Examples
    --------
    Sweep firm and year fixed effects out of a column:

    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> n = 200
    >>> fe = pd.DataFrame({"firm": rng.integers(0, 20, n),
    ...                    "year": rng.integers(0, 5, n)})
    >>> x = rng.normal(size=n) + 2.0 * fe["firm"].to_numpy()
    >>> xw, keep = sp.demean(x, fe)
    >>> print(xw.shape, int(keep.sum()))  # (200,) 200 — no singletons dropped
    >>> print(abs(xw.mean()) < 1e-8)      # True — FE means swept out
    """
    ab = Absorber(
        fe,
        weights=weights,
        drop_singletons=drop_singletons,
        tol=tol,
        maxiter=maxiter,
        solver=solver,
        slopes=slopes,
        n_obs=np.asarray(x).shape[0],
    )
    xw = ab.demean(x)
    return xw, ab.keep_mask


# ======================================================================
# High-level API: absorb_ols
# ======================================================================


def absorb_ols(
    y: np.ndarray,
    X: np.ndarray,
    fe: Union[pd.DataFrame, np.ndarray, None],
    weights: Optional[np.ndarray] = None,
    cluster: Optional[Union[np.ndarray, List[np.ndarray]]] = None,
    drop_singletons: bool = True,
    tol: float = 1e-8,
    maxiter: int = 10_000,
    return_absorber: bool = False,
    solver: str = "map",
    slopes: Optional[Sequence[SlopeSpec]] = None,
) -> dict:
    """OLS with absorbed high-dimensional fixed effects (reghdfe-style).

    Solves ``y = X β + Σ_k α_{g_k} + ε`` by sweeping out the FEs from
    both y and X (Frisch-Waugh-Lovell) and running OLS on residuals.

    Parameters
    ----------
    y : ndarray, shape (n,)
    X : ndarray, shape (n, p)
        Regressors *excluding* the absorbed FEs and the constant (the
        constant is absorbed by any FE dimension).
    fe : DataFrame or ndarray (n, K)
        Fixed-effect columns.
    weights : ndarray (n,), optional
        Observation weights.
    cluster : ndarray or list of ndarrays, optional
        One-way or multi-way cluster variables for robust SEs. If
        provided, returns cluster-robust SEs (one-way: Liang-Zeger
        sandwich; multi-way: inclusion-exclusion Cameron-Gelbach-Miller).
    drop_singletons : bool, default True
    tol, maxiter : float, int
        Demean convergence controls.
    return_absorber : bool, default False
        If True, also return the ``Absorber`` object for reuse.
    solver : {"map", "lsmr", "lsqr"}, default "map"
        Within-transformation backend. See :class:`Absorber`.

    Returns
    -------
    dict with keys:
        ``coef`` (p,), ``se`` (p,), ``vcov`` (p,p), ``resid`` (n_kept,),
        ``n`` (n_kept), ``df_resid``, ``dof_fe``, ``r2_within``,
        ``n_singletons_dropped``, ``converged``, ``iters``,
        ``absorber`` (if requested)

    Examples
    --------
    Two-way (firm + year) fixed effects with firm-clustered SEs:

    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> n = 500
    >>> firm = rng.integers(0, 30, n)
    >>> year = rng.integers(0, 10, n)
    >>> X = rng.normal(size=(n, 2))
    >>> y = (X @ np.array([1.5, -0.7]) + rng.normal(size=30)[firm]
    ...      + rng.normal(size=10)[year] + rng.normal(0, 0.5, n))
    >>> fe = pd.DataFrame({"firm": firm, "year": year})
    >>> out = sp.absorb_ols(y, X, fe, cluster=firm)
    >>> print(np.round(out["coef"], 2))   # [ 1.53 -0.71] — truth [1.5, -0.7]
    >>> print(out["n"], out["converged"])  # 500 True
    """
    y = np.asarray(y, dtype=np.float64).ravel()
    X = np.asarray(X, dtype=np.float64)
    if X.ndim == 1:
        X = X.reshape(-1, 1)
    n, p = X.shape
    if y.shape[0] != n:
        raise ValueError("y and X length mismatch.")  # pragma: no cover

    ab = Absorber(
        fe,
        weights=weights,
        drop_singletons=drop_singletons,
        tol=tol,
        maxiter=maxiter,
        solver=solver,
        slopes=slopes,
        n_obs=n,
    )
    yw = ab.demean(y)
    Xw = ab.demean(X)
    w = ab.weights

    # Weighted OLS on residuals
    if w is None:
        XtX = Xw.T @ Xw
        Xty = Xw.T @ yw
    else:
        Xw_w = Xw * w[:, None]
        XtX = Xw.T @ Xw_w
        Xty = Xw.T @ (yw * w)
    # Solve (use pinv fallback if near-singular)
    try:
        coef = np.linalg.solve(XtX, Xty)
    except np.linalg.LinAlgError:  # pragma: no cover
        coef = np.linalg.lstsq(XtX, Xty, rcond=None)[0]
    resid = yw - Xw @ coef

    # DOF: n - p - absorbed. Intercept-bearing FE dimensions share one
    # constant between them (hence the K-1 credit); varying-slope terms are
    # charged in full. See _absorbed_total_dof.
    dof_fe = _absorbed_total_dof(ab.n_fe, ab.slope_ops)
    df_resid = ab.n_kept - p - dof_fe
    if df_resid <= 0:
        raise ValueError(  # pragma: no cover
            f"Degrees of freedom exhausted: n_kept={ab.n_kept}, p={p}, "
            f"dof_fe={dof_fe}. Reduce regressors or drop a FE dimension."
        )

    # Within R² (FE already swept)
    if w is None:
        ss_res = float((resid**2).sum())
        y_demeaned = yw - yw.mean()
        ss_tot = float((y_demeaned**2).sum())
    else:
        ss_res = float((resid**2 * w).sum())
        y_bar_w = (yw * w).sum() / w.sum()
        ss_tot = float(((yw - y_bar_w) ** 2 * w).sum())
    r2_within = 1.0 - ss_res / ss_tot if ss_tot > 0 else np.nan

    # Variance-covariance matrix
    XtX_inv = np.linalg.inv(XtX)
    if cluster is None:
        # Classical (with DOF adjustment)
        sigma2 = ss_res / df_resid
        vcov = sigma2 * XtX_inv
    else:
        # Clip cluster arrays to the singleton-pruned subsample so shapes
        # match the within-transformed X and residuals.
        keep = ab.keep_mask
        if isinstance(cluster, list):
            cluster_sub = [np.asarray(c)[keep] for c in cluster]
        else:
            cluster_sub = np.asarray(cluster)[keep]
        dof_fe_cluster, nested_fe_in_cluster = _cluster_effective_fe_dof(
            ab.fe_codes,
            ab.n_fe,
            cluster_sub,
            ab.slope_ops,
        )
        vcov = _cluster_sandwich(
            Xw,
            resid,
            coef,
            XtX_inv,
            cluster_sub,
            df_resid=df_resid,
            weights=w,
            n_absorbed=dof_fe_cluster + p,
        )
    if cluster is None:
        dof_fe_cluster = dof_fe
        nested_fe_in_cluster = [False] * len(ab.n_fe)
    se = np.sqrt(np.maximum(np.diag(vcov), 0.0))

    out = {
        "coef": coef,
        "se": se,
        "vcov": vcov,
        "resid": resid,
        "fitted_within": Xw @ coef,  # within-prediction (excludes FE)
        "n": ab.n_kept,
        "df_resid": df_resid,
        "dof_fe": dof_fe,
        "dof_fe_cluster": dof_fe_cluster,
        "nested_fe_in_cluster": nested_fe_in_cluster,
        "r2_within": r2_within,
        "n_singletons_dropped": ab.n_dropped,
        "converged": ab._converged,
        "iters": ab._iters,
        "n_fe": ab.n_fe,
        "n_slope_levels": ab.n_slope_levels,
        "slope_names": [op.name for op in ab.slope_ops],
        "n_slope_degenerate": [op.n_degenerate for op in ab.slope_ops],
    }
    if return_absorber:
        out["absorber"] = ab
    return out


# ======================================================================
# Clustered sandwich (one-way + N-way inclusion-exclusion)
# ======================================================================


def _absorbed_fe_dof(fe_counts: List[int]) -> int:
    """Return the absorbed-FE parameter count under native HDFE conventions."""
    if not fe_counts:
        return 0
    return int(sum(int(G) for G in fe_counts) - (len(fe_counts) - 1))


def _slope_intercept_counts(slope_ops: Sequence[_SlopeOp]) -> List[int]:
    """Level counts of the intercept-bearing part of each slope term."""
    return [op.n_levels for op in slope_ops if op.with_intercept]


def _slope_dof(slope_ops: Sequence[_SlopeOp]) -> int:
    """Slope parameters actually absorbed, net of rank-deficient levels.

    A level whose within-level design is rank-deficient — an all-zero ``x``,
    or (in the intercept-bearing case) an ``x`` collinear with the constant
    — contributes a linearly dependent column that absorbs nothing, so it
    must not be charged. ``reghdfe`` reaches the same number through its
    ``Redundant`` column: ``absorb(big i.g#c.zc)`` with ``zc`` constant
    within ``g`` reports ``40 categories − 1 redundant = 39``.
    """
    return int(sum(op.n_levels - op.n_degenerate for op in slope_ops))


def _absorbed_total_dof(
    fe_counts: List[int],
    slope_ops: Sequence[_SlopeOp],
) -> int:
    """Absorbed parameter count including varying-slope terms (``e(df_a)``).

    A varying-slope term absorbs ``G`` parameters — one slope per level —
    and **not** ``G - 1``: the columns ``x · 1[g = j]`` do not contain the
    constant, so no level is redundant against the intercept. Only the
    intercept-bearing dimensions (ordinary FEs, plus the ``1[g = j]`` block
    of an ``i.g##c.x`` term) share the single constant and therefore give
    up ``K_intercept - 1`` degrees of freedom between them.

    Verified against Stata ``reghdfe``'s ``e(df_a)`` — see
    ``tests/test_hdfe_varying_slopes.py``. Example:
    ``absorb(county i.pref#c.year)`` with 30 counties and 6 prefectures
    reports ``e(df_a) = 36 = 30 + 6``, not ``35``.
    """
    intercept_counts = list(fe_counts) + _slope_intercept_counts(slope_ops)
    dof = _absorbed_fe_dof(intercept_counts) if intercept_counts else 0
    dof += _slope_dof(slope_ops)
    return int(dof)


def _codes_nested_in_cluster(
    fe_codes: np.ndarray,
    cluster_codes: np.ndarray,
    n_fe: int,
) -> bool:
    """True when every FE level maps to exactly one cluster level."""
    first_cluster = np.full(int(n_fe), -1, dtype=np.int64)
    for fe_code, cluster_code in zip(fe_codes, cluster_codes):
        previous = first_cluster[fe_code]
        if previous < 0:
            first_cluster[fe_code] = cluster_code
        elif previous != cluster_code:
            return False
    return True


def _cluster_effective_fe_dof(
    fe_codes: List[np.ndarray],
    fe_counts: List[int],
    cluster: Union[np.ndarray, List[np.ndarray]],
    slope_ops: Sequence[_SlopeOp] = (),
) -> Tuple[int, List[bool]]:
    """Return FE dof charged to CRV1 after omitting cluster-nested FEs.

    Varying-slope terms are always charged in full. A slope column
    ``x · 1[g = j]`` is not constant within a cluster even when ``g`` is
    nested in it, so the nesting redundancy that lets an ordinary FE be
    dropped does not apply — this matches ``reghdfe``, which reports
    ``e(df_a) = 6`` for ``absorb(county i.pref#c.year) vce(cluster county)``
    (county dropped as nested, the 6 slopes retained).
    """
    if not isinstance(cluster, list):
        clusters_list = [np.asarray(cluster)]
    else:
        clusters_list = [np.asarray(c) for c in cluster]

    cluster_codes_list = [_factorize(c)[0] for c in clusters_list]
    nested: List[bool] = []
    for codes, n_fe in zip(fe_codes, fe_counts):
        nested.append(
            any(
                _codes_nested_in_cluster(codes, cluster_codes, int(n_fe))
                for cluster_codes in cluster_codes_list
            )
        )
    slope_dof = _slope_dof(slope_ops)

    if not any(nested):
        return _absorbed_total_dof(fe_counts, slope_ops), nested
    effective_counts = [
        int(n_fe) for n_fe, is_nested in zip(fe_counts, nested) if not is_nested
    ]
    effective_counts += _slope_intercept_counts(slope_ops)
    if not effective_counts:
        # Every intercept-bearing dimension is nested in a cluster and so
        # drops out of the CRV1 parameter count — but the constant itself
        # is not nested away and is still an estimated parameter, so it is
        # charged its single degree of freedom on top of the slopes.
        return slope_dof + 1, nested
    return _absorbed_fe_dof(effective_counts) + slope_dof, nested


def _cluster_sandwich(
    X: np.ndarray,
    resid: np.ndarray,
    coef: np.ndarray,
    XtX_inv: np.ndarray,
    cluster: Union[np.ndarray, List[np.ndarray]],
    df_resid: int,
    weights: Optional[np.ndarray] = None,
    n_absorbed: int = 0,
) -> np.ndarray:
    """Cluster-robust variance (one-way or multi-way, PSD-corrected)."""
    if not isinstance(cluster, list):
        clusters_list = [np.asarray(cluster)]
    else:
        clusters_list = [np.asarray(c) for c in cluster]

    n, k = X.shape
    scores = X * resid[:, None] if weights is None else X * (resid * weights)[:, None]

    def _one_way(c: np.ndarray) -> np.ndarray:
        codes, _ = _factorize(c)
        G = int(codes.max()) + 1
        # Aggregate scores by cluster
        agg = np.zeros((G, k))
        np.add.at(agg, codes, scores)
        meat = agg.T @ agg
        scale = (G / max(G - 1, 1)) * ((n - 1) / max(n - n_absorbed, 1))
        return np.asarray(scale * XtX_inv @ meat @ XtX_inv)

    if len(clusters_list) == 1:
        V = _one_way(clusters_list[0])
    else:
        # N-way CGM via inclusion-exclusion over all non-empty subsets.
        from itertools import combinations

        V = np.zeros((k, k))
        M = len(clusters_list)
        for r in range(1, M + 1):
            for combo in combinations(range(M), r):
                # Intersection cluster: one group per distinct label tuple.
                # ⚠ correctness fix — this previously joined the labels with
                # "\0" and factorized the resulting strings, but pandas
                # truncates object strings at a NUL byte, so every
                # intersection collapsed back onto its first cluster
                # variable and the inclusion-exclusion terms were wrong.
                inter_codes, _ = _factorize_multi([clusters_list[i] for i in combo])
                V += ((-1) ** (r + 1)) * _one_way(inter_codes)
        # PSD correction
        eigvals, eigvecs = np.linalg.eigh(V)
        if (eigvals < 0).any():
            eigvals = np.maximum(eigvals, 0.0)
            V = eigvecs @ np.diag(eigvals) @ eigvecs.T
    return V


__all__ = [
    "Absorber",
    "SlopeSpec",
    "demean",
    "absorb_ols",
]
