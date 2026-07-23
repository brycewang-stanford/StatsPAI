"""
Conley (1999) spatial — and spatio-temporal — HAC standard errors.

For cross-sectional or panel data with spatial correlation, standard errors
must account for the fact that nearby observations are correlated. This
module implements the Conley (1999) spatial heteroskedasticity and
autocorrelation consistent (HAC) variance estimator:

    V = (X'X)^{-1} Omega (X'X)^{-1}

where Omega = sum_i sum_j K(d_ij / h) * e_i * e_j * x_i x_j'

Passing ``time``, ``lag_cutoff`` and ``unit`` extends ``K`` over the time
dimension as well, giving the spatio-temporal estimator used by Hsiang (2010)
and by Stata's ``acreg``: same-unit pairs are weighted by a time kernel,
cross-unit pairs by the product of a spatial and a time kernel. See
:func:`conley` for the exact correspondence with ``acreg``'s options — in
particular that ``acreg``'s ``lag()`` governs only within-unit pairs, and
cross-unit correlation is contemporaneous unless ``lagdist()`` is set.

Two distance conventions are offered. ``distance="haversine"`` (the default,
and the historical behaviour) is great-circle distance on a sphere of radius
6371 km. ``distance="planar"`` reproduces ``acreg``'s flat 111 km/degree
convention, which is what to use for numerical parity with Stata.

For large datasets a scipy cKDTree is used for neighbour lookup, and — when
``unit`` is given — it is built on the *de-duplicated* unit coordinates rather
than on every row. Memory stays O(number of neighbour pairs with non-zero
weight); the n x n weight matrix is never materialised.

References
----------
Conley, T.G. (1999).
"GMM Estimation with Cross Sectional Dependence."
*Journal of Econometrics*, 92(1), 1-45. [@conley1999estimation]

Conley, T.G. (2008). "Spatial Econometrics." In *The New Palgrave Dictionary
of Economics*. [@conley2008spatial]

Hsiang, S.M. (2010).
"Temperatures and Cyclones Strongly Associated with Economic
Production in the Caribbean and Central America."
*PNAS*, 107(35), 15367-15372. [@hsiang2010temperatures]
"""

from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial import cKDTree

from ..core.results import EconometricResults
from ..exceptions import MethodIncompatibility
from ._psd import se_from_vcov

# Earth radius in km
_EARTH_RADIUS_KM = 6371.0

#: Kilometres per degree of latitude under ``acreg``'s planar convention.
#: ``acreg`` hard-codes 111 (see ``acreg.ado``: ``lat_scale = 111``,
#: ``lon_scale = cos(lat*pi()/180)*111``). It is *not* ``6371*pi/180``.
_ACREG_KM_PER_DEG = 111.0

#: Cap on the dense ``T x T`` within-unit time-kernel block, in elements.
#: Larger units are processed in row chunks so memory stays bounded.
_OWN_BLOCK_CHUNK = 4_000_000


def _haversine_km(
    lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray
) -> np.ndarray:
    """
    Vectorised Haversine distance in kilometres.

    Parameters are in **degrees**; conversion to radians is done internally.
    """
    lat1, lon1, lat2, lon2 = (np.radians(x) for x in (lat1, lon1, lat2, lon2))
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = np.sin(dlat / 2) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2) ** 2
    return np.asarray(2 * _EARTH_RADIUS_KM * np.arcsin(np.sqrt(a)), dtype=float)


def _latlon_to_cartesian(lat_deg: np.ndarray, lon_deg: np.ndarray) -> np.ndarray:
    """
    Convert latitude/longitude (degrees) to 3-D Cartesian coordinates on a
    unit sphere, scaled to Earth's radius in km.  This allows approximate
    Euclidean distance for cKDTree ball queries.
    """
    lat = np.radians(lat_deg)
    lon = np.radians(lon_deg)
    x = _EARTH_RADIUS_KM * np.cos(lat) * np.cos(lon)
    y = _EARTH_RADIUS_KM * np.cos(lat) * np.sin(lon)
    z = _EARTH_RADIUS_KM * np.sin(lat)
    return np.column_stack([x, y, z])


def _planar_km(
    lat_anchor: np.ndarray,
    lon_anchor: np.ndarray,
    lat_other: np.ndarray,
    lon_other: np.ndarray,
) -> np.ndarray:
    """Stata ``acreg``'s planar distance, in km.

    ``acreg`` scales latitude by a flat 111 km/degree and longitude by
    ``cos(lat_anchor) * 111``, where ``lat_anchor`` is the latitude of the
    *reference* (column) unit. The resulting distance is therefore
    **asymmetric**: ``d(a -> b) != d(b -> a)`` whenever the two points sit at
    different latitudes. This is a property of ``acreg``, faithfully
    reproduced here, not an approximation introduced by StatsPAI.
    """
    scale = np.cos(np.radians(lat_anchor)) * _ACREG_KM_PER_DEG
    dlat = _ACREG_KM_PER_DEG * (lat_anchor - lat_other)
    dlon = scale * (lon_anchor - lon_other)
    return np.asarray(np.sqrt(dlat**2 + dlon**2), dtype=float)


def _spatial_kernel(d: np.ndarray, cutoff: float, kind: str) -> np.ndarray:
    """Spatial kernel weight; exactly zero beyond ``cutoff``."""
    inside = d <= cutoff
    if kind == "uniform":
        return inside.astype(float)
    return np.maximum(0.0, 1.0 - np.abs(d / cutoff)) * inside


def _time_kernel(dt: np.ndarray, lag_cutoff: float, kind: str) -> np.ndarray:
    """Time kernel weight; exactly zero beyond ``lag_cutoff`` periods.

    The Bartlett taper is ``1 - |dt| / (lag_cutoff + 1)``, i.e. the Newey-West
    convention in which a lag of exactly ``lag_cutoff`` still carries strictly
    positive weight. ``lag_cutoff = 0`` collapses both kernels to the
    same-period indicator.
    """
    inside = dt <= lag_cutoff
    if kind == "uniform":
        return inside.astype(float)
    return np.maximum(0.0, 1.0 - dt / (lag_cutoff + 1.0)) * inside


def _unit_pair_weights(
    lat_u: np.ndarray,
    lon_u: np.ndarray,
    dist_cutoff: float,
    kernel: str,
    distance: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Ordered neighbour *unit* pairs and their spatial weights.

    Returns ``(col_unit, row_unit, weight)``, where ``weight`` is the spatial
    kernel applied to the distance from the column (reference) unit to the row
    unit. Only pairs with strictly positive weight are returned, so the memory
    footprint is O(number of neighbour unit pairs) — never O(U^2).

    The KD-tree runs on a metric that is a guaranteed *lower bound* on the
    true distance, so querying it at ``dist_cutoff`` returns a superset of the
    genuine neighbours; exact distances are then recomputed on the candidates
    and out-of-range pairs dropped. Getting that bound the wrong way round
    silently loses neighbour pairs, which is why it is spelled out below.
    """
    if distance == "haversine":
        coords = _latlon_to_cartesian(lat_u, lon_u)
        # Chord length bounding the great-circle cutoff.
        theta = dist_cutoff / _EARTH_RADIUS_KM
        radius = 2 * _EARTH_RADIUS_KM * np.sin(min(theta, np.pi) / 2)
    else:
        # The KD-tree metric must *under*-estimate the true distance, so that
        # every pair with true distance <= cutoff is returned as a candidate.
        # The anchored longitude scale is cos(lat_anchor) * 111, so scaling by
        # the smallest cos(lat) in the sample lower-bounds it for every
        # possible anchor. (Using 111 directly would over-estimate and would
        # silently drop genuine neighbours.)
        cos_min = float(np.min(np.cos(np.radians(lat_u))))
        cos_min = max(cos_min, 0.0)
        coords = np.column_stack(
            [
                _ACREG_KM_PER_DEG * lat_u,
                _ACREG_KM_PER_DEG * cos_min * lon_u,
            ]
        )
        radius = float(dist_cutoff)

    pairs = cKDTree(coords).query_pairs(r=radius, output_type="ndarray")
    empty = np.empty(0, dtype=np.intp), np.empty(0, dtype=np.intp), np.empty(0)
    if len(pairs) == 0:
        return empty

    a = pairs[:, 0].astype(np.intp)
    b = pairs[:, 1].astype(np.intp)

    if distance == "haversine":
        d = _haversine_km(lat_u[a], lon_u[a], lat_u[b], lon_u[b])
        w = _spatial_kernel(d, dist_cutoff, kernel)
        # Symmetric distance -> both orderings share one weight.
        col = np.concatenate([a, b])
        row = np.concatenate([b, a])
        ws = np.concatenate([w, w])
    else:
        # Anchored at ``a`` for the (col=a, row=b) direction and vice versa.
        w_from_a = _spatial_kernel(
            _planar_km(lat_u[a], lon_u[a], lat_u[b], lon_u[b]), dist_cutoff, kernel
        )
        w_from_b = _spatial_kernel(
            _planar_km(lat_u[b], lon_u[b], lat_u[a], lon_u[a]), dist_cutoff, kernel
        )
        col = np.concatenate([a, b])
        row = np.concatenate([b, a])
        ws = np.concatenate([w_from_a, w_from_b])

    keep = ws > 0
    if not keep.any():
        return empty
    return col[keep], row[keep], ws[keep]


def _symmetrise_planar(V: np.ndarray) -> np.ndarray:
    """Symmetrise the planar (``acreg``-convention) covariance matrix.

    ``acreg``'s anchored longitude scale makes the weight matrix genuinely
    asymmetric, so ``S' W S`` comes out asymmetric too and has to be
    symmetrised. Mata's ``_makesymmetric`` — which ``acreg`` calls — does this
    by *mirroring the lower triangle into the upper one*. That makes the
    reported off-diagonal covariances depend on the order in which the
    regressors were typed: re-running ``acreg y x2 x1`` instead of
    ``acreg y x1 x2`` moves ``Cov(x1, x2)`` from 1.6479398857745e-03 to
    1.6484267193922e-03 on our validation panel.

    We take the symmetric part ``(V + V')/2`` instead. It is invariant to
    variable order, and — because neither operation touches the diagonal — it
    yields **exactly** the same variances, hence exactly the same standard
    errors, t-statistics, p-values and confidence intervals as ``acreg``. Only
    the off-diagonal covariances differ, and only by the size of ``acreg``'s
    own order-dependence.
    """
    return 0.5 * (V + V.T)


def _spatiotemporal_meat(
    S: np.ndarray,
    lat_v: np.ndarray,
    lon_v: np.ndarray,
    unit_codes: np.ndarray,
    time_codes: np.ndarray,
    n_units: int,
    n_times: int,
    dist_cutoff: float,
    kernel: str,
    time_kernel: str,
    lag_cutoff: int,
    lag_cutoff_cross: int,
    distance: str,
) -> np.ndarray:
    """Accumulate the Conley spatial + time HAC meat, ``S' W S``.

    ``W`` is never materialised. The within-unit block and the cross-unit
    block are accumulated separately, each in O(number of non-zero weights)
    memory, so an n x n matrix is never allocated.

    Following ``acreg``, a pair drawn from the *same* unit takes the time
    kernel alone at bandwidth ``lag_cutoff`` (its spatial distance is zero by
    construction), while a pair drawn from *different* units takes the product
    of the spatial kernel and the time kernel at bandwidth
    ``lag_cutoff_cross``.
    """
    k = S.shape[1]
    meat = np.zeros((k, k), dtype=float)

    # ---- within-unit block: pure time kernel at bandwidth `lag_cutoff` ----
    order = np.argsort(unit_codes, kind="stable")
    starts = np.searchsorted(unit_codes[order], np.arange(n_units), side="left")
    ends = np.searchsorted(unit_codes[order], np.arange(n_units), side="right")

    if n_units and int(np.max(ends - starts)) <= 1:
        # Every unit contributes at most one row (a pure cross-section, or the
        # `distance="planar"` no-time path). Then the within-unit block is just
        # the diagonal, and both time kernels give weight 1 at dt = 0. Skip the
        # per-unit Python loop entirely.
        meat += S.T @ S
        starts = ends  # mark the loop below as done
    for u in range(n_units):
        rows = order[starts[u] : ends[u]]
        if rows.size == 0:
            continue
        t_u = time_codes[rows].astype(float)
        S_u = S[rows]
        # Chunk the dense T x T time-gap block so memory stays bounded even
        # for units observed over very many periods.
        chunk = max(1, int(_OWN_BLOCK_CHUNK // max(t_u.size, 1)))
        for lo in range(0, t_u.size, chunk):
            hi = min(lo + chunk, t_u.size)
            dt = np.abs(t_u[lo:hi, None] - t_u[None, :])
            w = _time_kernel(dt, lag_cutoff, time_kernel)
            meat += S_u[lo:hi].T @ (w @ S_u)

    # ---- cross-unit block: spatial kernel x time kernel ----
    lat_u = np.zeros(n_units)
    lon_u = np.zeros(n_units)
    lat_u[unit_codes] = lat_v
    lon_u[unit_codes] = lon_v

    col_u, row_u, ws = _unit_pair_weights(lat_u, lon_u, dist_cutoff, kernel, distance)
    if col_u.size == 0:
        return meat

    # (unit, time) -> row index lookup; -1 marks an unobserved cell. This is
    # O(U * T) integers, which is the natural size of a balanced panel, not
    # O(n^2).
    row_of = np.full((n_units, n_times), -1, dtype=np.int64)
    row_of[unit_codes, time_codes] = np.arange(S.shape[0], dtype=np.int64)

    n_pairs = col_u.size
    # Bound the working set to ~4M gathered observation pairs at a time.
    pair_chunk = max(1, int(_OWN_BLOCK_CHUNK // max(n_times, 1)))

    for lag in range(-int(lag_cutoff_cross), int(lag_cutoff_cross) + 1):
        wt = float(
            _time_kernel(
                np.array([abs(lag)], dtype=float), lag_cutoff_cross, time_kernel
            )[0]
        )
        if wt == 0.0:
            continue
        # Column-side periods that have a valid row-side counterpart.
        t_col = np.arange(max(0, -lag), min(n_times, n_times - lag), dtype=np.int64)
        if t_col.size == 0:
            continue
        t_row = t_col + lag

        for lo in range(0, n_pairs, pair_chunk):
            hi = min(lo + pair_chunk, n_pairs)
            cu = col_u[lo:hi]
            ru = row_u[lo:hi]
            wp = ws[lo:hi]

            idx_col = row_of[np.repeat(cu, t_col.size), np.tile(t_col, cu.size)]
            idx_row = row_of[np.repeat(ru, t_row.size), np.tile(t_row, ru.size)]
            good = (idx_col >= 0) & (idx_row >= 0)
            if not good.any():
                continue
            weights = np.repeat(wp, t_col.size)[good] * wt
            meat += (S[idx_row[good]] * weights[:, None]).T @ S[idx_col[good]]

    return meat


def conley(
    result: EconometricResults,
    data: pd.DataFrame,
    lat: str,
    lon: str,
    dist_cutoff: float,
    kernel: str = "uniform",
    alpha: float = 0.05,
    *,
    time: Optional[str] = None,
    lag_cutoff: Optional[int] = None,
    time_kernel: str = "bartlett",
    unit: Optional[str] = None,
    lag_cutoff_cross: Optional[int] = None,
    distance: str = "haversine",
) -> EconometricResults:
    """
    Compute Conley (1999) spatial — and optionally spatio-temporal — HAC
    standard errors.

    With ``time`` left at ``None`` this is the pure cross-sectional Conley
    (1999) estimator: ``V = (X'X)^-1 Omega (X'X)^-1`` with
    ``Omega = sum_i sum_j K(d_ij) e_i e_j x_i x_j'``.

    Supplying ``time``, ``lag_cutoff`` and ``unit`` switches on the
    spatio-temporal ("Conley spatial + time HAC") variant used by Hsiang
    (2010) and implemented by Stata's ``acreg``. Following ``acreg``, the two
    kinds of pair are weighted differently:

    * **same unit, different period** — the time kernel alone, at bandwidth
      ``lag_cutoff``. (Their spatial distance is zero by construction.)
    * **different unit** — the spatial kernel times the time kernel, the
      latter at bandwidth ``lag_cutoff_cross``.

    Note ``lag_cutoff_cross`` defaults to ``0``, which is ``acreg``'s own
    default: ``lag()`` governs only the within-unit serial correlation, and
    cross-unit correlation is admitted **contemporaneously only** unless
    ``lagdist()`` is set. Passing ``lag_cutoff_cross=lag_cutoff`` gives the
    fully separable space-time kernel that the phrase "spatio-temporal HAC"
    more often suggests.

    Parameters
    ----------
    result : EconometricResults
        Fitted OLS result. Must have ``data_info`` containing
        ``'X'`` (design matrix), ``'y'`` (response), and ``'residuals'``.
    data : pd.DataFrame
        Data with latitude and longitude columns (in **degrees**), row-aligned
        to the fitted sample.
    lat : str
        Column name for latitude.
    lon : str
        Column name for longitude.
    dist_cutoff : float
        Distance cutoff *h* in kilometres.  Pairs farther apart than this
        receive zero weight.
    kernel : str, default ``"uniform"``
        Spatial kernel: ``"uniform"`` (indicator) or ``"bartlett"``
        (linearly declining weight ``1 - d/h``).
    alpha : float, default 0.05
        Significance level for confidence intervals.
    time : str, optional
        Column name holding the time period. Must be integer-valued. Requires
        ``lag_cutoff`` and ``unit``.
    lag_cutoff : int, optional
        Serial-correlation bandwidth in periods, for pairs within the same
        unit. Requires ``time``.
    time_kernel : str, default ``"bartlett"``
        Time kernel: ``"bartlett"`` (``1 - |dt| / (lag_cutoff + 1)``, the
        Newey-West convention, matching ``acreg``'s ``hac bartlett``) or
        ``"uniform"`` (indicator, matching ``acreg`` without ``hac``).
    unit : str, optional
        Column name identifying the panel unit. Enables panel de-duplication:
        the spatial neighbour search runs on the distinct unit coordinates
        rather than on every row, so a unit observed T times costs one point
        in the KD-tree instead of T.
    lag_cutoff_cross : int, optional
        Time bandwidth for pairs drawn from *different* units — ``acreg``'s
        ``lagdist()``. Defaults to ``0`` (``acreg``'s default), i.e.
        contemporaneous cross-unit correlation only.
    distance : str, default ``"haversine"``
        ``"haversine"`` uses great-circle distance on a sphere of radius
        6371 km, and is symmetric. ``"planar"`` reproduces ``acreg``'s
        convention exactly: 111 km per degree of latitude and
        ``cos(lat_ref) * 111`` km per degree of longitude, anchored at the
        *reference* point, which makes the distance asymmetric; the resulting
        covariance is symmetrised the same way ``acreg`` does (Mata
        ``_makesymmetric``, which mirrors the lower triangle). Choose
        ``"planar"`` for bit-level ``acreg`` parity, ``"haversine"`` for a
        proper great-circle metric.

    Returns
    -------
    EconometricResults
        New results object carrying the Conley standard errors. ``model_info``
        records ``se_type``, ``dist_cutoff_km``, ``kernel`` and — in the
        spatio-temporal case — ``lag_cutoff``, ``lag_cutoff_cross``,
        ``time_kernel`` and ``distance``.

    Raises
    ------
    ValueError
        If ``kernel``, ``time_kernel`` or ``distance`` is not recognised, if
        ``time`` and ``lag_cutoff`` are not supplied together, if ``time`` is
        supplied without ``unit``, or if ``time`` is not integer-valued.
    MethodIncompatibility
        If a unit's coordinates are not constant across its rows.

    Notes
    -----
    Kernel-weighted HAC covariance matrices are **not** positive semi-definite
    by construction. A negative variance is reported as ``nan`` alongside a
    ``RuntimeWarning``, never clamped to zero — see
    :func:`statspai.inference._psd.se_from_vcov`. Stata's ``acreg`` reports the
    same terms as missing.

    Memory is O(number of neighbour pairs with non-zero weight); the n x n
    weight matrix is never materialised.

    Examples
    --------
    Cross-sectional Conley:

    >>> import statspai as sp
    >>> import numpy as np
    >>> import pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> n = 200
    >>> x1 = rng.normal(size=n)
    >>> x2 = rng.normal(size=n)
    >>> y = 1.0 + 0.5 * x1 - 0.3 * x2 + rng.normal(size=n)
    >>> df = pd.DataFrame({
    ...     "y": y, "x1": x1, "x2": x2,
    ...     "latitude": rng.uniform(30, 45, size=n),
    ...     "longitude": rng.uniform(-120, -100, size=n),
    ... })
    >>> result = sp.regress("y ~ x1 + x2", data=df)
    >>> c = sp.conley(result, data=df, lat="latitude", lon="longitude",
    ...               dist_cutoff=100)
    >>> print(c.summary())  # doctest: +SKIP

    Spatial + time HAC on a geo-panel, matching
    ``acreg y x1 x2, spatial latitude(lat) longitude(lon) dist(500) lag(5)
    id(id) time(t) hac bartlett``:

    >>> st = sp.conley(result, data=panel, lat="lat", lon="lon",
    ...                dist_cutoff=500, kernel="bartlett",
    ...                time="t", lag_cutoff=5, unit="id",
    ...                distance="planar")            # doctest: +SKIP

    References
    ----------
    [@conley1999estimation], [@conley2008spatial], [@hsiang2010temperatures]
    """
    if kernel not in ("uniform", "bartlett"):
        raise ValueError(f"kernel must be 'uniform' or 'bartlett', got '{kernel}'")
    if time_kernel not in ("uniform", "bartlett"):
        raise ValueError(
            f"time_kernel must be 'uniform' or 'bartlett', got '{time_kernel}'"
        )
    if distance not in ("haversine", "planar"):
        raise ValueError(f"distance must be 'haversine' or 'planar', got '{distance}'")

    # --- Fail loudly on half-specified space-time requests ---
    if (time is None) != (lag_cutoff is None):
        if time is None:
            raise ValueError(
                "conley() got lag_cutoff but no time column, so there is no "
                "time dimension to apply it to. Pass the period column as "
                "`time=` (and the panel identifier as `unit=`), e.g.\n"
                f"    sp.conley(result, data, lat={lat!r}, lon={lon!r}, "
                f"dist_cutoff={dist_cutoff!r}, time='year', "
                f"lag_cutoff={lag_cutoff!r}, unit='id')"
            )
        unit_hint = repr(unit) if unit is not None else "'id'"
        raise ValueError(
            "conley() got a time column but no lag_cutoff, so the time kernel "
            "has no bandwidth and the time dimension would be silently "
            "ignored. Pass the serial-correlation bandwidth in periods as "
            "`lag_cutoff=`, e.g.\n"
            f"    sp.conley(result, data, lat={lat!r}, lon={lon!r}, "
            f"dist_cutoff={dist_cutoff!r}, time={time!r}, lag_cutoff=5, "
            f"unit={unit_hint})"
        )
    if time is None and lag_cutoff_cross is not None:
        raise ValueError(
            "conley() got lag_cutoff_cross but no time column. "
            "lag_cutoff_cross is the cross-unit time bandwidth (Stata acreg's "
            "`lagdist()`) and is meaningless without `time=` and "
            "`lag_cutoff=`."
        )
    if time is not None and unit is None:
        raise ValueError(
            "conley() got a time column but no unit column. The spatio-"
            "temporal estimator has to tell same-unit pairs (weighted by the "
            "time kernel at lag_cutoff) apart from cross-unit pairs (weighted "
            "by the spatial kernel), so the panel identifier is required — "
            "Stata's acreg raises the same error. Pass it as `unit=`, e.g.\n"
            f"    sp.conley(result, data, lat={lat!r}, lon={lon!r}, "
            f"dist_cutoff={dist_cutoff!r}, time={time!r}, "
            f"lag_cutoff={lag_cutoff!r}, unit='id')"
        )

    # --- Extract estimation objects ---
    X = np.asarray(result.data_info["X"])
    residuals = np.asarray(result.data_info["residuals"])
    n, k = X.shape

    for col in (lat, lon, time, unit):
        if col is not None and col not in data.columns:
            raise ValueError(
                f"Column {col!r} is not in `data` (columns: "
                f"{list(data.columns)[:20]})."
            )
    if len(data) != n:
        raise MethodIncompatibility(
            f"`data` has {len(data)} rows but the fitted design matrix has {n}. "
            "`data` must be row-aligned to the estimation sample; pass the same "
            "frame (post listwise-deletion) that produced `result`."
        )

    lat_vals = data[lat].values.astype(float)
    lon_vals = data[lon].values.astype(float)

    XtX_inv = np.linalg.inv(X.T @ X)

    # --- Spatio-temporal branch ------------------------------------------
    if time is not None:
        Xe = X * residuals[:, np.newaxis]

        raw_time = data[time].to_numpy()
        t_float = np.asarray(raw_time, dtype=float)
        if not np.all(np.isfinite(t_float)):
            raise ValueError(f"Time column {time!r} contains missing values.")
        if not np.allclose(t_float, np.rint(t_float)):
            raise ValueError(
                f"Time column {time!r} must be integer-valued: lag_cutoff is "
                "measured in whole periods. Recode the period to consecutive "
                "integers (e.g. year, or a period index) before calling."
            )
        t_int = np.rint(t_float).astype(np.int64)
        time_codes = t_int - t_int.min()
        n_times = int(time_codes.max()) + 1

        unit_labels = data[unit].to_numpy()
        _, unit_codes = np.unique(unit_labels, return_inverse=True)
        unit_codes = unit_codes.astype(np.int64)
        n_units = int(unit_codes.max()) + 1

        # Panel de-duplication is only valid if a unit really is one place.
        for name, vals in ((lat, lat_vals), (lon, lon_vals)):
            first = np.zeros(n_units)
            first[unit_codes[::-1]] = vals[::-1]
            bad = ~np.isclose(vals, first[unit_codes], rtol=0, atol=0)
            if bad.any():
                offenders = np.unique(unit_labels[bad])[:5]
                raise MethodIncompatibility(
                    f"Column {name!r} is not constant within `unit`="
                    f"{unit!r}: unit(s) {list(offenders)} carry more than one "
                    f"{name} value. sp.conley treats a unit as a single fixed "
                    "location (that is what makes the panel KD-tree valid), so "
                    "moving units cannot be handled this way. Either drop "
                    "`unit=` to treat every row as its own location, or "
                    "collapse each unit to one coordinate."
                )

        # One row per (unit, time): the cross-unit block's cell lookup is
        # single-valued, so a duplicated cell would silently keep only the
        # last duplicate row in cross-unit terms while the within-unit
        # block keeps every row — an inconsistent hybrid with wrong SEs
        # (verified against a dense reference). Stata's acreg refuses
        # repeated id-time pairs for the same reason; fail loudly.
        cell = unit_codes * np.int64(n_times) + time_codes
        cell_counts = np.bincount(cell, minlength=n_units * n_times)
        if (cell_counts > 1).any():
            n_dup = int((cell_counts > 1).sum())
            dup_unit = unit_labels[cell == int(np.argmax(cell_counts))][0]
            raise ValueError(
                f"`unit`={unit!r} x `time`={time!r} does not uniquely index "
                f"the rows: {n_dup} (unit, time) cell(s) contain more than "
                f"one row (e.g. unit {dup_unit!r}). The spatio-temporal HAC "
                "requires one row per unit-period (Stata's acreg imposes the "
                "same restriction). Aggregate the data to one row per "
                "(unit, time), or pass the row-level identifier as `unit=` "
                "if each row really is its own location."
            )

        cross = 0 if lag_cutoff_cross is None else int(lag_cutoff_cross)
        if int(lag_cutoff) < 0 or cross < 0:
            raise ValueError(
                "lag_cutoff and lag_cutoff_cross must be non-negative "
                f"(got lag_cutoff={lag_cutoff}, lag_cutoff_cross={cross})."
            )

        Omega = _spatiotemporal_meat(
            Xe,
            lat_vals,
            lon_vals,
            unit_codes,
            time_codes,
            n_units,
            n_times,
            float(dist_cutoff),
            kernel,
            time_kernel,
            int(lag_cutoff),
            cross,
            distance,
        )

        V = XtX_inv @ Omega @ XtX_inv
        if distance == "planar":
            V = _symmetrise_planar(V)

        return _finalise(
            result,
            V,
            n,
            k,
            alpha,
            {
                "se_type": "conley_spatiotemporal",
                "dist_cutoff_km": dist_cutoff,
                "kernel": kernel,
                "time_kernel": time_kernel,
                "lag_cutoff": int(lag_cutoff),
                "lag_cutoff_cross": cross,
                "distance": distance,
            },
            f"Conley spatial+time HAC ({kernel} spatial / {time_kernel} time)",
        )

    # --- Planar cross-section (acreg convention, no time dimension) -------
    if distance == "planar":
        Xe = X * residuals[:, np.newaxis]
        unit_codes = np.arange(n, dtype=np.int64)
        Omega = _spatiotemporal_meat(
            Xe,
            lat_vals,
            lon_vals,
            unit_codes,
            np.zeros(n, dtype=np.int64),
            n,
            1,
            float(dist_cutoff),
            kernel,
            time_kernel,
            0,
            0,
            "planar",
        )
        V = XtX_inv @ Omega @ XtX_inv
        V = _symmetrise_planar(V)
        return _finalise(
            result,
            V,
            n,
            k,
            alpha,
            {
                "se_type": "conley_spatial",
                "dist_cutoff_km": dist_cutoff,
                "kernel": kernel,
                "distance": "planar",
            },
            f"Conley spatial HAC ({kernel} kernel)",
        )

    # --- Build Omega using cKDTree for O(n log n) neighbour lookup ---
    coords_3d = _latlon_to_cartesian(lat_vals, lon_vals)
    tree = cKDTree(coords_3d)

    # Chord length corresponding to dist_cutoff (upper bound for ball query)
    # chord = 2R sin(theta/2), theta = dist_cutoff / R
    theta = dist_cutoff / _EARTH_RADIUS_KM
    chord_cutoff = 2 * _EARTH_RADIUS_KM * np.sin(theta / 2)

    # Pre-compute X_i * e_i  (n x k)
    Xe = X * residuals[:, np.newaxis]

    # The meat of the sandwich.
    # Diagonal terms (every obs with itself, kernel weight = 1):
    #   sum_i outer(Xe_i, Xe_i)  ==  Xe.T @ Xe
    Omega = Xe.T @ Xe

    # Off-diagonal terms: only pairs within cutoff
    pairs = tree.query_pairs(r=chord_cutoff, output_type="ndarray")

    if len(pairs) > 0:
        idx_i = pairs[:, 0]
        idx_j = pairs[:, 1]

        # Exact Haversine distances for candidate pairs
        d_ij = _haversine_km(
            lat_vals[idx_i], lon_vals[idx_i], lat_vals[idx_j], lon_vals[idx_j]
        )

        # Apply distance cutoff (chord approximation may admit a few extras)
        within = d_ij <= dist_cutoff

        if kernel == "uniform":
            weights = np.ones_like(d_ij)
        else:  # bartlett
            weights = 1.0 - d_ij / dist_cutoff

        weights = weights * within  # zero out pairs beyond cutoff

        # Each pair (i,j) contributes symmetrically:
        #   weight * (outer(Xe_i, Xe_j) + outer(Xe_j, Xe_i)).
        # Summed over all pairs this is M + M.T where
        #   M = sum_p weight_p * outer(Xe_i, Xe_j) = (Xe_i * w).T @ Xe_j.
        # Zero-weight (beyond-cutoff) pairs contribute exactly nothing.
        Wi = Xe[idx_i] * weights[:, np.newaxis]
        M = Wi.T @ Xe[idx_j]
        Omega += M + M.T

    V = XtX_inv @ Omega @ XtX_inv

    return _finalise(
        result,
        V,
        n,
        k,
        alpha,
        {
            "se_type": "conley_spatial",
            "dist_cutoff_km": dist_cutoff,
            "kernel": kernel,
        },
        f"Conley spatial HAC ({kernel} kernel)",
    )


def _finalise(
    result: EconometricResults,
    V: np.ndarray,
    n: int,
    k: int,
    alpha: float,
    extra_model_info: dict,
    estimator_label: str,
) -> EconometricResults:
    """Wrap a Conley covariance matrix into a fresh results object."""
    se = pd.Series(
        se_from_vcov(V, list(result.params.index), estimator=estimator_label),
        index=result.params.index,
    )

    df_resid = n - k

    model_info = dict(result.model_info)
    model_info.update(extra_model_info)

    data_info = dict(result.data_info)
    data_info["df_resid"] = df_resid
    data_info["vcov"] = V

    new_result = EconometricResults(
        params=result.params.copy(),
        std_errors=se,
        model_info=model_info,
        data_info=data_info,
        diagnostics=dict(result.diagnostics),
    )

    # Recompute CIs at requested alpha
    t_crit = stats.t.ppf(1 - alpha / 2, df_resid)
    new_result.conf_int_lower = new_result.params - t_crit * se
    new_result.conf_int_upper = new_result.params + t_crit * se

    return new_result


def ols_conley_vcov(
    result: Any,
    data: pd.DataFrame,
    lat: str,
    lon: str,
    dist_cutoff: float,
) -> Any:
    """Conley spatial-HAC covariance for OLS (Stata ``acreg``-compatible).

    Same ``acreg`` planar-distance convention as the 2SLS ``iv_conley_vcov``:
    111 km per degree of latitude and ``cos(lat_b) * 111`` per degree of
    longitude anchored at the column point b (asymmetric weight matrix, then
    V is symmetrised). The score is the bread-applied cluster-robust
    score ``bread @ X_g' e_g`` (bread = (X'X)^{-1}), so the meat on the
    score level is ``(bread@M) (bread@M)'`` which collapses to
    ``bread @ (Σ score score') @ bread``. Matches ``acreg y x, spatial
    latitude() longitude() dist()`` to machine precision.

    The existing ``sp.conley`` (Haversine kernel) is kept for users who
    specifically want a great-circle distance; this function is the
    native ``regress(vce="conley")`` reference (matches ``acreg``).
    """
    iv = getattr(result, "data_info", None) or {}
    if not ({"X", "y", "var_names"} <= set(iv)):
        raise MethodIncompatibility(
            "ols_conley_vcov needs a result with stored "
            "data_info['X'/'y'/'var_names']; use sp.conley for the formula-reparse "
            "fallback."
        )
    y = np.asarray(iv["y"], dtype=float)
    X = np.asarray(iv["X"], dtype=float)
    names = list(iv["var_names"])
    n, k = X.shape
    for c in (lat, lon):
        if c not in data.columns or data[c].isna().any() or len(data[c]) != n:
            raise MethodIncompatibility(
                f"Coordinate column {c!r} must be present, complete, and row-"
                "aligned to the fitted sample."
            )
    lat_v = data[lat].to_numpy(dtype=float)
    lon_v = data[lon].to_numpy(dtype=float)

    bread = np.linalg.inv(X.T @ X)  # (X'X)^{-1}
    beta = bread @ (X.T @ y)
    resid = y - X @ beta

    lon_scale = np.cos(np.radians(lat_v)) * 111.0
    d_lat = lat_v[:, None] - lat_v[None, :]
    d_lon = lon_v[:, None] - lon_v[None, :]
    dist = np.sqrt((111.0 * d_lat) ** 2 + (lon_scale[None, :] * d_lon) ** 2)
    weig = (dist <= dist_cutoff).astype(float)

    score = X * resid[:, None]
    core = bread @ (score.T @ weig @ score) @ bread
    vcov = 0.5 * (core + core.T)
    se = pd.Series(
        se_from_vcov(vcov, names, estimator="Conley spatial HAC (uniform kernel)"),
        index=names,
    )
    return se
