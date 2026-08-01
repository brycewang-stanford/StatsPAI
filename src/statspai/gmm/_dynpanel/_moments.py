"""Design-matrix and instrument-matrix construction for dynamic-panel GMM.

Given a :class:`~._spec.DynPanelSpec` and a :class:`~._data.PanelArrays`,
build the stacked system the GMM solver consumes:

* ``dy``  — the transformed dependent variable, one entry per usable
  (unit, equation period, equation);
* ``W``   — the transformed regressors;
* ``Z``   — the instrument matrix, block-diagonal in the GMM-style columns
  and one dense column per standard ("IV-style") instrument;
* row bookkeeping (unit index, equation period, which equation) so the
  weight matrix, the robust sandwich and the Arellano-Bond serial
  correlation tests can group by unit and address neighbouring periods.

Two equations can be stacked:

**Transformed (differenced) equation** — Arellano-Bond.  Removes the fixed
effect by first-differencing; instrumented by lagged *levels*.

**Level equation** — the Blundell-Bond (1998) addition that makes this
*system* GMM.  Keeps the equation in levels and instruments it with lagged
*differences*, exploiting E[Δy_{i,t-1}(α_i + ε_{it})] = 0.  This is what
rescues identification when the series is persistent (ρ near 1), where the
lagged levels are weak instruments for the differences.

Construction is vectorised over units.  Instrument columns are enumerated
over the **global** period range rather than the observed one, which is
what makes the instrument count agree with Stata's ``e(zrank)`` /
``e(j)`` on ragged panels (a column that happens to be empty for every
unit still occupies a moment slot).

References
----------
Arellano, M. and Bond, S. (1991). *Review of Economic Studies* 58(2).
[@arellano1991some]
Blundell, R. and Bond, S. (1998). *Journal of Econometrics* 87(1),
115-143. [@blundell1998initial]
Arellano, M. and Bover, O. (1995). Another look at the instrumental
variable estimation of error-components models. *Journal of Econometrics*
68(1), 29-51. [@arellano1995another]
Roodman, D. (2009). How to do xtabond2. *Stata Journal* 9(1), 86-136 —
instrument collapsing, the levels-equation moment layout, and the ``H``
matrix used for the one-step system weight. [@roodman2009xtabond]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

import numpy as np

from ._data import PanelArrays
from ._spec import DynPanelSpec, GMMBlock, IVBlock, Term, term_name

__all__ = [
    "Design",
    "build_design",
    "first_difference_H",
    "system_H",
    "fod_weights",
    "fod_operator",
]

DIFF_EQ = 0
LEVEL_EQ = 1


@dataclass
class Design:
    """The stacked GMM system plus the bookkeeping the estimators need."""

    dy: np.ndarray
    W: np.ndarray
    Z: np.ndarray
    row_unit: np.ndarray
    row_period: np.ndarray
    row_eq: np.ndarray
    regressor_names: List[str]
    instrument_labels: List[str]
    # Column index sets for difference-in-Hansen tests, keyed by a
    # human-readable subset name ("GMM instruments for levels", "iv(...)").
    instrument_groups: Dict[str, List[int]] = field(default_factory=dict)
    unit_rows: List[np.ndarray] = field(default_factory=list)
    transform: str = "fd"
    # The (n_periods - 1, n_periods) transform operator M on the *balanced*
    # period grid, row ``a`` corresponding to stored period ``a + 1``.  H is
    # derived from it in ``_estimate.onestep_weight``.  Only materialised for
    # forward orthogonal deviations; the first difference has the closed-form
    # 2/-1 band and needs no operator.
    transform_operator: Optional[np.ndarray] = None
    # Row groups for the variance "meat"; None means group by unit.
    cluster_rows: Optional[List[np.ndarray]] = None
    _group_index_cache: Optional[Tuple[np.ndarray, np.ndarray]] = None

    @property
    def n_rows(self) -> int:
        return int(self.dy.size)

    @property
    def n_params(self) -> int:
        return int(self.W.shape[1])

    @property
    def n_instruments(self) -> int:
        return int(self.Z.shape[1])

    @property
    def n_units_used(self) -> int:
        return len(self.unit_rows)

    @property
    def meat_rows(self) -> List[np.ndarray]:
        """Row groups the sandwich "meat" is summed over.

        The units by default; a coarser grouping is installed by
        ``set_clusters``. Only the meat re-groups — the one-step weight
        ``(Z'HZ)^{-1}`` stays a within-unit object because ``H`` encodes the
        serial structure the *transform* induces, which is a property of the
        unit's own time path, not of the cluster it belongs to.
        """
        return self.unit_rows if self.cluster_rows is None else self.cluster_rows

    @property
    def n_clusters(self) -> int:
        return len(self.meat_rows)

    def set_clusters(self, unit_codes: np.ndarray) -> None:
        """Group rows by a unit-level cluster code (see ``_data``)."""
        present = np.array([self.row_unit[rows[0]] for rows in self.unit_rows])
        codes = np.asarray(unit_codes)[present]
        self.cluster_rows = [
            np.concatenate([self.unit_rows[j] for j in np.flatnonzero(codes == c)])
            for c in np.unique(codes)
        ]
        self._group_index_cache = None

    def group_index(self):
        """``(order, starts)`` for a segment sum over :attr:`meat_rows`.

        Built once and cached. Rebuilding it per call made ``reduceat`` the
        single largest cost on a 20k-unit panel.
        """
        if self._group_index_cache is None:
            groups = self.meat_rows
            codes = np.empty(self.n_rows, dtype=np.int64)
            for j, rows in enumerate(groups):
                codes[rows] = j
            order = np.argsort(codes, kind="stable")
            starts = np.searchsorted(codes[order], np.arange(len(groups)))
            self._group_index_cache = (order, starts)
        return self._group_index_cache

    @property
    def has_level_equation(self) -> bool:
        return bool(np.any(self.row_eq == LEVEL_EQ))

    @property
    def n_rows_diff(self) -> int:
        return int(np.count_nonzero(self.row_eq == DIFF_EQ))

    @property
    def n_rows_level(self) -> int:
        return int(np.count_nonzero(self.row_eq == LEVEL_EQ))


def _shift(arr: np.ndarray, lag: int) -> np.ndarray:
    """``out[:, p] = arr[:, p - lag]``, ``NaN`` where the source is absent."""
    if lag == 0:
        return arr
    out = np.full_like(arr, np.nan)
    if lag < arr.shape[1]:
        out[:, lag:] = arr[:, : arr.shape[1] - lag]
    return out


def _delta(panel: PanelArrays, term: Term) -> np.ndarray:
    """First difference of a (possibly lagged) term: ``v_{p-l} - v_{p-l-1}``."""
    arr = panel.get(term.var)
    return _shift(arr, term.lag) - _shift(arr, term.lag + 1)


def _level(panel: PanelArrays, term: Term) -> np.ndarray:
    """Level of a (possibly lagged) term: ``v_{p-l}``."""
    return _shift(panel.get(term.var), term.lag)


def first_difference_H(periods: np.ndarray) -> np.ndarray:
    """MA(1) covariance structure of first-differenced i.i.d. errors.

    ``2`` on the diagonal and ``-1`` between *consecutive* equation periods
    only, so an interior gap correctly breaks the off-diagonal link instead
    of pretending two non-adjacent differences share an error term.  This is
    the ``H`` in Stata's one-step weight ``(Σ_i Z_i' H Z_i)^{-1}``.
    """
    r = periods.size
    H = np.zeros((r, r))
    np.fill_diagonal(H, 2.0)
    adjacent = np.abs(periods[:, None] - periods[None, :]) == 1
    H[adjacent] = -1.0
    return H


def system_H(periods: np.ndarray, eqs: np.ndarray) -> np.ndarray:
    """``H`` for the stacked system, under var[ε] = I and var[α] = 0.

    With ``M`` the differencing operator, the block is

        ``[[M M', M ], [M', I]]``

    (Roodman 2009, ``xtabond2``'s default ``h(3)``), which is exactly the
    covariance of ``[Δε, ε]`` when the idiosyncratic errors are spherical
    and the fixed effect is ignored:

    * transformed × transformed: ``2`` on the diagonal, ``-1`` between
      adjacent periods;
    * transformed(p) × level(s): ``+1`` when ``s == p``, ``-1`` when
      ``s == p - 1`` — because ``Δε_p = ε_p − ε_{p-1}``;
    * level × level: the identity.

    ``h(2)`` (DPD for Gauss/Ox) zeroes the cross quadrants; ``h(1)`` uses
    the identity throughout. Only ``h(3)`` is implemented, matching
    ``xtabond2``'s default.
    """
    r = periods.size
    H = np.zeros((r, r))
    is_diff = eqs == DIFF_EQ
    is_level = ~is_diff

    dd = np.outer(is_diff, is_diff)
    same = periods[:, None] == periods[None, :]
    adj = np.abs(periods[:, None] - periods[None, :]) == 1
    H[dd & same] = 2.0
    H[dd & adj] = -1.0

    ll = np.outer(is_level, is_level)
    H[ll & same] = 1.0

    # transformed(row p) x level(col s): +1 if s == p, -1 if s == p - 1
    dl = np.outer(is_diff, is_level)
    lead = periods[None, :] == (periods[:, None] - 1)
    H[dl & same] = 1.0
    H[dl & lead] = -1.0
    ld = np.outer(is_level, is_diff)
    lag_ = periods[None, :] == (periods[:, None] + 1)
    H[ld & same] = 1.0
    H[ld & lag_] = -1.0
    return H


def fod_operator(n_periods: int) -> np.ndarray:
    """Forward orthogonal deviation operator on the *balanced* period grid.

    Row ``a`` (stored period ``a + 1``) has ``c_a`` on period ``a`` and
    ``-c_a / (T - 1 - a)`` on every later period, with
    ``c_a = sqrt((T - 1 - a) / (T - a))``.  ``M M' = I`` by construction.

    ``H`` is built from *this* operator rather than each unit's own one.
    ``xtabond2``'s help is explicit that "H always has block diagonal form,
    with all blocks the same", and the choice is observable: on the ragged
    ``abdata`` panel using each unit's own operator moves the FOD
    system-GMM coefficients by up to 13%.  The transformed *values* are
    still built from each unit's own available future periods — only the
    a-priori error covariance is the common one.
    """
    T = int(n_periods)
    M = np.zeros((max(T - 1, 0), T))
    for a in range(T - 1):
        tf = T - 1 - a
        c = np.sqrt(tf / (tf + 1.0))
        M[a, a] = c
        M[a, a + 1 :] = -c / tf
    return M


def fod_weights(periods: np.ndarray, n_periods: int) -> np.ndarray:
    """Forward orthogonal deviation operator for one unit.

    Arellano & Bover (1995): instead of differencing backwards, subtract
    from each observation the mean of all its *available future* ones,

        ``y*_t = c_t (y_t − mean_{s > t} y_s)``,
        ``c_t = sqrt(T_t / (T_t + 1))``,

    with ``T_t`` the number of future observations.  Two properties make
    this worth having:

    1. ``M M' = I``, so the one-step weight needs no MA(1) correction and
       the transformed errors stay serially uncorrelated;
    2. a gap costs one observation, not two — first differencing destroys
       the equations on *both* sides of a hole, while FOD simply averages
       over whatever future periods exist.  On a gappy panel that is a
       large efficiency gain, and it is why ``xtabond2`` offers
       ``orthogonal``.

    ``periods`` are the unit's equation-eligible periods in ascending
    order; the last one has no future and yields no row.  Returns an
    ``(len(periods) - 1, n_periods)`` matrix.

    This unit-specific operator builds the transformed *values*.  The
    a-priori error covariance ``H`` uses :func:`fod_operator`, the operator
    on the balanced grid — the two coincide on a balanced panel and differ
    on a ragged one, and ``xtabond2`` uses the balanced form for ``H``.
    """
    periods = np.asarray(periods, dtype=int)
    r = periods.size - 1
    M = np.zeros((max(r, 0), n_periods))
    for a in range(r):
        future = periods[a + 1 :]
        tf = future.size
        c = np.sqrt(tf / (tf + 1.0))
        M[a, periods[a]] = c
        M[a, future] = -c / tf
    return M


def build_design(panel: PanelArrays, spec: DynPanelSpec) -> Design:
    """Assemble the stacked equations, their regressors and their instruments.

    A (unit, period) pair contributes a **transformed** equation only when
    every level that row needs is observed for that unit — the dependent
    variable at ``p`` and ``p-1``, and each regressor term at ``p-lag`` and
    ``p-lag-1``.  It contributes a **level** equation (system GMM only) when
    ``y_p`` and every regressor level ``v_{p-lag}`` are observed.
    Availability is evaluated per variable, so a covariate that is missing
    early (as user-built or lag-operator terms necessarily are) costs only
    the equations that actually need it and never removes a level from the
    instrument pool.

    GMM-style instrument columns follow Stata's ``missing=0`` convention: an
    unavailable instrument contributes a zero rather than dropping the
    equation.
    """
    if spec.transform not in ("fd", "fod"):
        raise ValueError(
            f"transform must be 'fd' (first differences) or 'fod' (forward "
            f"orthogonal deviations), got {spec.transform!r}."
        )

    T = panel.n_periods
    terms = spec.regressor_terms
    if not terms:
        raise ValueError("the model has no regressors (lags=0 and no covariates).")

    # Standard ("IV-style") instruments are *not* subject to the missing=0
    # convention that GMM-style blocks use: Stata prints "GMM-type
    # (missing=0 ...)" only for the latter. A row whose IV column is
    # unobserved is therefore dropped, not zeroed. Invisible in the usual
    # case where every IV term is also a regressor, and decisive when it is
    # not -- e.g. Anderson-Hsiao's D.L2.y instrument, which reaches one
    # period deeper than the equation itself does.
    iv_diff_terms = [b.term for b in spec.iv_blocks if b.equation in ("diff", "both")]
    iv_level_terms = [b.term for b in spec.iv_blocks if b.equation in ("level", "both")]

    deepest = max(
        [t.lag for t in terms] + [t.lag for t in iv_diff_terms + iv_level_terms]
    )
    diff_periods = np.arange(max(deepest + 1, 1), T)
    if diff_periods.size == 0:
        raise ValueError(
            "not enough time periods for the requested lag structure: "
            f"{T} periods but the differenced equation reaches "
            f"{deepest + 1} periods back."
        )
    level_periods = np.arange(deepest, T) if spec.level_equation else np.arange(0)

    Y = panel.get(spec.y)
    fod = spec.transform == "fod"
    transform_operator: Optional[np.ndarray] = fod_operator(T) if fod else None

    # ---- transformed rows --------------------------------------------------
    if fod:
        di, d_p, d_y, d_W = _fod_rows(panel, spec, terms, deepest, T, iv_diff_terms)
        # A FOD row is *labelled* with the period of the first difference it
        # replaces (its own period + 1) so that the instrument grid, the
        # level-equation cross block and the serial-correlation test all use
        # one period index across transforms. This is xtabond2's convention:
        # with `orthogonal` it still reports GMM-type instruments L(1/.) of
        # the same variable over the same period range as `noorthogonal`.
        diff_periods = np.arange(deepest + 1, T)
    else:
        dy_grid = (Y - _shift(Y, 1))[:, diff_periods]
        ok = np.isfinite(dy_grid)
        d_grids = [_delta(panel, t)[:, diff_periods] for t in terms]
        for grid in d_grids:
            ok &= np.isfinite(grid)
        for term in iv_diff_terms:
            ok &= np.isfinite(_delta(panel, term)[:, diff_periods])
        di, dj = np.nonzero(ok)
        d_p = diff_periods[dj]
        d_y = dy_grid[di, dj]
        d_W = (
            np.column_stack([g[di, dj] for g in d_grids])
            if di.size
            else np.zeros((0, len(terms)))
        )

    # ---- level rows (system GMM) ------------------------------------------
    if level_periods.size:
        ylev_grid = Y[:, level_periods]
        okl = np.isfinite(ylev_grid)
        l_grids = [_level(panel, t)[:, level_periods] for t in terms]
        for grid in l_grids:
            okl &= np.isfinite(grid)
        for term in iv_level_terms:
            okl &= np.isfinite(_level(panel, term)[:, level_periods])
        li, lj = np.nonzero(okl)
        l_p = level_periods[lj]
        l_y = ylev_grid[li, lj]
        l_W = (
            np.column_stack([g[li, lj] for g in l_grids])
            if li.size
            else np.zeros((0, len(terms)))
        )
    else:
        li = np.zeros(0, dtype=int)
        l_p = np.zeros(0, dtype=int)
        l_y = np.zeros(0)
        l_W = np.zeros((0, len(terms)))

    if di.size + li.size == 0:
        raise ValueError(
            "no usable observations — every unit-period is missing at least "
            "one required level."
        )

    rows_i = np.concatenate([di, li])
    rows_p = np.concatenate([d_p, l_p])
    rows_eq = np.concatenate(
        [np.full(di.size, DIFF_EQ), np.full(li.size, LEVEL_EQ)]
    ).astype(int)
    dy = np.concatenate([d_y, l_y])
    W = np.vstack([d_W, l_W])
    names = [term_name(t) for t in terms]

    if spec.constant:
        # A constant differences away, so it lives only in the level rows.
        W = np.column_stack([W, (rows_eq == LEVEL_EQ).astype(float)])
        names = names + ["_cons"]

    # ---- instruments -------------------------------------------------------
    z_cols: List[np.ndarray] = []
    z_labels: List[str] = []
    groups: Dict[str, List[int]] = {}

    def _tag(name: str, start: int, stop: int) -> None:
        if stop > start:
            groups.setdefault(name, []).extend(range(start, stop))

    for block in spec.gmm_blocks:
        start = len(z_cols)
        if block.equation == "diff":
            cols, labels = _diff_gmm_columns(
                panel, block, diff_periods, rows_i, rows_p, rows_eq
            )
        else:
            cols, labels = _level_gmm_columns(
                panel, block, level_periods, rows_i, rows_p, rows_eq
            )
        z_cols.extend(cols)
        z_labels.extend(labels)
        _tag(block.label, start, len(z_cols))
        if block.equation == "level":
            _tag("GMM instruments for levels", start, len(z_cols))
    # Under a non-difference transform the standard ("IV-style") instrument
    # for the transformed equation is the *transformed* regressor, not its
    # first difference; reuse the already-built columns so the two can never
    # drift apart.
    trans_lookup = (
        {(t.var, t.lag): W[: di.size, j] for j, t in enumerate(terms)} if fod else None
    )
    iv_start = len(z_cols)
    for iv in spec.iv_blocks:
        z_cols.append(
            _iv_column(panel, iv, rows_i, rows_p, rows_eq, trans_lookup, di.size)
        )
        z_labels.append(iv.label)
    _tag(
        "iv(" + " ".join(term_name(b.term) for b in spec.iv_blocks) + ")",
        iv_start,
        len(z_cols),
    )
    if spec.constant:
        z_cols.append((rows_eq == LEVEL_EQ).astype(float))
        z_labels.append("_cons")

    if not z_cols:
        raise ValueError("the specification produced no instruments.")
    Z = np.column_stack(z_cols)

    order = np.lexsort((rows_p, rows_eq, rows_i))
    rows_i, rows_p, rows_eq = rows_i[order], rows_p[order], rows_eq[order]
    dy, W, Z = dy[order], W[order], Z[order]
    boundaries = np.flatnonzero(np.diff(rows_i)) + 1
    unit_rows = [np.asarray(g) for g in np.split(np.arange(rows_i.size), boundaries)]

    return Design(
        dy=dy,
        W=W,
        Z=Z,
        row_unit=rows_i,
        row_period=rows_p,
        row_eq=rows_eq,
        regressor_names=names,
        instrument_labels=z_labels,
        instrument_groups={k: list(v) for k, v in groups.items()},
        unit_rows=unit_rows,
        transform=spec.transform,
        transform_operator=transform_operator,
    )


def _diff_gmm_columns(
    panel: PanelArrays,
    block: GMMBlock,
    diff_periods: np.ndarray,
    rows_i: np.ndarray,
    rows_p: np.ndarray,
    rows_eq: np.ndarray,
) -> Tuple[List[np.ndarray], List[str]]:
    """Lagged *levels* instrumenting the transformed equation.

    Without ``collapse`` there is one column per ``(equation period p, lag
    distance d)`` with ``d`` in ``[lag_min, min(lag_max, p)]`` — the classic
    Arellano-Bond block-diagonal layout, whose count matches Stata's
    ``e(zrank)``.  With ``collapse`` there is one column per lag distance,
    stacking every period into the same column (Roodman 2009); empty columns
    are dropped so the count matches ``xtabond2, collapse``.
    """
    V = panel.get(block.var)
    n = rows_i.size
    on_diff = rows_eq == DIFF_EQ
    cols: List[np.ndarray] = []
    labels: List[str] = []

    if block.collapse:
        for d in range(block.lag_min, block.lag_max + 1):
            s = rows_p - d
            valid = on_diff & (s >= 0)
            if not valid.any():
                continue
            col = np.zeros(n)
            col[valid] = np.nan_to_num(V[rows_i[valid], s[valid]], nan=0.0)
            if not np.any(col):
                continue
            cols.append(col)
            labels.append(f"L{d}.{block.var}(collapsed)")
        return cols, labels

    for p in diff_periods:
        for d in range(block.lag_min, min(block.lag_max, int(p)) + 1):
            s = int(p) - d
            if s < 0:
                continue
            sel = on_diff & (rows_p == p)
            col = np.zeros(n)
            if sel.any():
                col[sel] = np.nan_to_num(V[rows_i[sel], s], nan=0.0)
            cols.append(col)
            labels.append(f"L{d}.{block.var}@t{p}")
    return cols, labels


def _level_gmm_columns(
    panel: PanelArrays,
    block: GMMBlock,
    level_periods: np.ndarray,
    rows_i: np.ndarray,
    rows_p: np.ndarray,
    rows_eq: np.ndarray,
) -> Tuple[List[np.ndarray], List[str]]:
    """Lagged *differences* instrumenting the level equation.

    The Blundell-Bond moment is E[Δv_{i,t-a}(α_i + ε_{it})] = 0 where ``a``
    is one shallower than the transformed equation's minimum lag, i.e. for
    the standard ``gmm_lags=(2, .)`` on the dependent variable the level
    instrument is ``Δy_{i,t-1}``.  Only that single difference is used:
    deeper ones are redundant given the transformed-equation moments
    (Roodman 2009, Sec. 3), which is also what ``xtabond2`` prints as
    ``D.L.n`` under "Instruments for levels equation".
    """
    V = panel.get(block.var)
    dV = V - _shift(V, 1)
    shift = max(block.lag_min - 1, 0)
    n = rows_i.size
    on_level = rows_eq == LEVEL_EQ
    cols: List[np.ndarray] = []
    labels: List[str] = []

    if block.collapse:
        s = rows_p - shift
        valid = on_level & (s >= 1)
        if not valid.any():
            return cols, labels
        col = np.zeros(n)
        col[valid] = np.nan_to_num(dV[rows_i[valid], s[valid]], nan=0.0)
        if np.any(col):
            cols.append(col)
            labels.append(f"D.L{shift}.{block.var}(collapsed,level)")
        return cols, labels

    for p in level_periods:
        s = int(p) - shift
        if s < 1:
            continue
        sel = on_level & (rows_p == p)
        col = np.zeros(n)
        if sel.any():
            col[sel] = np.nan_to_num(dV[rows_i[sel], s], nan=0.0)
        if not np.any(col):
            continue
        cols.append(col)
        labels.append(f"D.L{shift}.{block.var}@t{p}(level)")
    return cols, labels


def _iv_column(
    panel: PanelArrays,
    iv: IVBlock,
    rows_i: np.ndarray,
    rows_p: np.ndarray,
    rows_eq: np.ndarray,
    trans_lookup=None,
    n_diff_rows: int = 0,
) -> np.ndarray:
    """A single "standard" instrument column.

    ``equation='diff'`` uses the transformed value on the transformed rows
    only; ``'level'`` the level on the level rows only; ``'both'`` — which
    is ``xtabond2``'s default for ``iv()`` — puts the transformed value on
    the transformed rows *and* the level on the level rows in **one**
    column, imposing the single combined moment condition.
    """
    n = rows_i.size
    col = np.zeros(n)
    on_diff = rows_eq == DIFF_EQ
    if iv.equation in ("diff", "both"):
        key = (iv.term.var, iv.term.lag)
        if trans_lookup is not None and key in trans_lookup:
            col[:n_diff_rows] = np.nan_to_num(trans_lookup[key], nan=0.0)
        elif trans_lookup is not None:
            raise ValueError(
                f"instrument {iv.label!r} is not among the transformed "
                "regressors, so it cannot be orthogonally deviated; add it to "
                "the regressor list or use the default first-difference "
                "transform."
            )
        else:
            grid = _delta(panel, iv.term)
            sel = on_diff
            if sel.any():
                col[sel] = np.nan_to_num(grid[rows_i[sel], rows_p[sel]], nan=0.0)
    if iv.equation in ("level", "both"):
        grid = _level(panel, iv.term)
        sel = ~on_diff
        if sel.any():
            col[sel] = np.nan_to_num(grid[rows_i[sel], rows_p[sel]], nan=0.0)
    return col


def _fod_rows(panel, spec, terms, deepest, T, iv_diff_terms=()):
    """Transformed rows under forward orthogonal deviations.

    The eligible period set is the one where the equation holds *in
    levels* — ``y_t`` and every regressor level ``v_{t-lag}`` observed —
    because FOD transforms the level equation forward rather than
    differencing it backwards.  Each unit contributes one row per eligible
    period that has at least one eligible successor.
    """
    Y = panel.get(spec.y)
    level_grids = [_level(panel, t) for t in terms]
    eligible = np.isfinite(Y)
    for g in level_grids:
        eligible = eligible & np.isfinite(g)
    for term in iv_diff_terms:
        eligible = eligible & np.isfinite(_level(panel, term))
    eligible[:, :deepest] = False

    idx = np.arange(T)
    rows_i, rows_p, ys, Ws = [], [], [], []
    for u in range(panel.n_units):
        periods = idx[eligible[u]]
        M = fod_weights(periods, T)
        if M.shape[0] == 0:
            continue
        yv = np.nan_to_num(Y[u], nan=0.0)
        ys.append(M @ yv)
        Ws.append(
            np.column_stack([M @ np.nan_to_num(g[u], nan=0.0) for g in level_grids])
        )
        rows_i.append(np.full(M.shape[0], u))
        rows_p.append(periods[:-1] + 1)

    if not rows_i:
        empty = np.zeros(0, dtype=int)
        return empty, empty, np.zeros(0), np.zeros((0, len(terms)))
    return (
        np.concatenate(rows_i),
        np.concatenate(rows_p),
        np.concatenate(ys),
        np.vstack(Ws),
    )
