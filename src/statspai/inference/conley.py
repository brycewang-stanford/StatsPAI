"""
Conley (1999) Spatial HAC Standard Errors.

For cross-sectional or panel data with spatial correlation, standard errors
must account for the fact that nearby observations are correlated. This
module implements the Conley (1999) spatial heteroskedasticity and
autocorrelation consistent (HAC) variance estimator:

    V = (X'X)^{-1} Omega (X'X)^{-1}

where Omega = sum_i sum_j K(d_ij / h) * e_i * e_j * x_i x_j'

Distances are computed using the Haversine formula (great-circle distance
in km). For large datasets, a scipy cKDTree is used for fast neighbor
lookup so that only pairs within the cutoff distance are evaluated.

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

from typing import Any

import numpy as np
import pandas as pd
from scipy import stats
from scipy.spatial import cKDTree

from ..core.results import EconometricResults
from ..exceptions import MethodIncompatibility

# Earth radius in km
_EARTH_RADIUS_KM = 6371.0


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


def conley(
    result: EconometricResults,
    data: pd.DataFrame,
    lat: str,
    lon: str,
    dist_cutoff: float,
    kernel: str = "uniform",
    alpha: float = 0.05,
) -> EconometricResults:
    """
    Compute Conley (1999) spatial HAC standard errors.

    Parameters
    ----------
    result : EconometricResults
        Fitted OLS result. Must have ``data_info`` containing
        ``'X'`` (design matrix), ``'y'`` (response), and ``'residuals'``.
    data : pd.DataFrame
        Data with latitude and longitude columns (in **degrees**).
    lat : str
        Column name for latitude.
    lon : str
        Column name for longitude.
    dist_cutoff : float
        Distance cutoff *h* in kilometres.  Pairs farther apart than this
        receive zero weight.
    kernel : str, default ``"uniform"``
        Kernel function: ``"uniform"`` (indicator) or ``"bartlett"``
        (linearly declining weight).
    alpha : float, default 0.05
        Significance level for confidence intervals.

    Returns
    -------
    EconometricResults
        New results object with Conley spatial standard errors.

    Examples
    --------
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
    """
    if kernel not in ("uniform", "bartlett"):
        raise ValueError(f"kernel must be 'uniform' or 'bartlett', got '{kernel}'")

    # --- Extract estimation objects ---
    X = np.asarray(result.data_info["X"])
    residuals = np.asarray(result.data_info["residuals"])
    n, k = X.shape

    lat_vals = data[lat].values.astype(float)
    lon_vals = data[lon].values.astype(float)

    XtX_inv = np.linalg.inv(X.T @ X)

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

    # --- Build new results ---
    se = pd.Series(np.sqrt(np.diag(V)), index=result.params.index)

    df_resid = n - k

    model_info = dict(result.model_info)
    model_info["se_type"] = "conley_spatial"
    model_info["dist_cutoff_km"] = dist_cutoff
    model_info["kernel"] = kernel

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
    se = pd.Series(np.sqrt(np.maximum(np.diag(vcov), 0)), index=names)
    return se
