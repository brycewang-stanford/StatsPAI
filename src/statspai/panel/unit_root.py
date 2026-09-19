"""
Panel unit root tests.

Provides LLC (Levin-Lin-Chu), IPS (Im-Pesaran-Shin), Fisher-type
(Maddala-Wu / Choi) and Hadri stationarity tests for panel data.

Equivalent to Stata's ``xtunitroot`` and R's ``plm::purtest()``. The two
references agree on the Fisher and Hadri statistics and differ, by
documented convention, on LLC and IPS (see :func:`panel_unitroot`,
``convention``).

References
----------
Levin, A., Lin, C.F. & Chu, C.S.J. (2002).
"Unit Root Tests in Panel Data: Asymptotic and Finite-Sample Properties."
*Journal of Econometrics*, 108(1), 1-24. [@levin2002unit]

Im, K.S., Pesaran, M.H. & Shin, Y. (2003).
"Testing for Unit Roots in Heterogeneous Panels."
*Journal of Econometrics*, 115(1), 53-74. [@im2003testing]
"""

import warnings
from typing import Any, Dict, Optional

import numpy as np
import pandas as pd
from scipy import stats

from .._result_serialize import ResultProtocolMixin

# ---------------------------------------------------------------------------
# Tabulated moments. Checked cell-by-cell against plm's copies (adj.levinlin,
# adj.ips.wtbar) in tests/reference_parity/test_timeseries_R_parity.py.
# ---------------------------------------------------------------------------

# Levin-Lin-Chu mean / std adjustments (mu*, sigma*) by T, per model.
_LLC_T = (25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100, 250, 500)
_LLC_ADJ = {
    "n": (
        (0.004, 1.049),
        (0.003, 1.035),
        (0.002, 1.027),
        (0.002, 1.021),
        (0.001, 1.017),
        (0.001, 1.014),
        (0.001, 1.011),
        (0.000, 1.008),
        (0.000, 1.007),
        (0.000, 1.006),
        (0.000, 1.005),
        (0.000, 1.001),
        (0.000, 1.000),
    ),
    "c": (
        (-0.554, 0.919),
        (-0.546, 0.889),
        (-0.541, 0.867),
        (-0.537, 0.850),
        (-0.533, 0.837),
        (-0.531, 0.826),
        (-0.527, 0.810),
        (-0.524, 0.798),
        (-0.521, 0.789),
        (-0.520, 0.782),
        (-0.518, 0.776),
        (-0.509, 0.742),
        (-0.500, 0.707),
    ),
    "ct": (
        (-0.703, 1.003),
        (-0.674, 0.949),
        (-0.653, 0.906),
        (-0.637, 0.871),
        (-0.624, 0.842),
        (-0.614, 0.818),
        (-0.598, 0.780),
        (-0.587, 0.751),
        (-0.578, 0.728),
        (-0.571, 0.710),
        (-0.566, 0.695),
        (-0.533, 0.603),
        (-0.500, 0.500),
    ),
}

# IPS (2003) Table 3: mean and variance of the ADF t-statistic by
# (lags 0..8) x T in _IPS_T, for the constant ('c') and trend ('ct') cases.
_IPS_T = (10, 15, 20, 25, 30, 40, 50, 60, 70, 100)
_NA = np.nan
_IPS = {
    ("c", "mean"): (
        (
            -1.504,
            -1.514,
            -1.522,
            -1.520,
            -1.526,
            -1.523,
            -1.527,
            -1.519,
            -1.524,
            -1.532,
        ),
        (
            -1.488,
            -1.503,
            -1.516,
            -1.514,
            -1.519,
            -1.520,
            -1.524,
            -1.519,
            -1.522,
            -1.530,
        ),
        (
            -1.319,
            -1.387,
            -1.428,
            -1.443,
            -1.460,
            -1.476,
            -1.493,
            -1.490,
            -1.498,
            -1.514,
        ),
        (
            -1.306,
            -1.366,
            -1.413,
            -1.433,
            -1.453,
            -1.471,
            -1.489,
            -1.486,
            -1.495,
            -1.512,
        ),
        (
            -1.171,
            -1.260,
            -1.329,
            -1.363,
            -1.394,
            -1.428,
            -1.454,
            -1.458,
            -1.470,
            -1.495,
        ),
        (_NA, _NA, -1.313, -1.351, -1.384, -1.421, -1.451, -1.454, -1.467, -1.494),
        (_NA, _NA, _NA, -1.289, -1.331, -1.380, -1.418, -1.427, -1.444, -1.476),
        (_NA, _NA, _NA, -1.273, -1.319, -1.371, -1.411, -1.423, -1.441, -1.474),
        (_NA, _NA, _NA, -1.212, -1.266, -1.329, -1.377, -1.393, -1.415, -1.456),
    ),
    ("c", "var"): (
        (1.069, 0.923, 0.851, 0.809, 0.789, 0.770, 0.760, 0.749, 0.736, 0.735),
        (1.255, 1.011, 0.915, 0.861, 0.831, 0.803, 0.781, 0.770, 0.753, 0.745),
        (1.421, 1.078, 0.969, 0.905, 0.865, 0.830, 0.798, 0.789, 0.766, 0.754),
        (1.759, 1.181, 1.037, 0.952, 0.907, 0.858, 0.819, 0.802, 0.782, 0.761),
        (2.080, 1.279, 1.097, 1.005, 0.946, 0.886, 0.842, 0.819, 0.801, 0.771),
        (_NA, _NA, 1.171, 1.055, 0.980, 0.912, 0.863, 0.839, 0.814, 0.781),
        (_NA, _NA, _NA, 1.114, 1.023, 0.942, 0.886, 0.858, 0.834, 0.795),
        (_NA, _NA, _NA, 1.164, 1.062, 0.968, 0.910, 0.875, 0.851, 0.806),
        (_NA, _NA, _NA, 1.217, 1.105, 0.996, 0.929, 0.896, 0.871, 0.818),
    ),
    ("ct", "mean"): (
        (
            -2.166,
            -2.167,
            -2.168,
            -2.167,
            -2.172,
            -2.173,
            -2.176,
            -2.174,
            -2.174,
            -2.177,
        ),
        (
            -2.173,
            -2.169,
            -2.172,
            -2.172,
            -2.173,
            -2.177,
            -2.180,
            -2.178,
            -2.176,
            -2.179,
        ),
        (
            -1.914,
            -1.999,
            -2.047,
            -2.074,
            -2.095,
            -2.120,
            -2.137,
            -2.143,
            -2.146,
            -2.158,
        ),
        (
            -1.922,
            -1.977,
            -2.032,
            -2.065,
            -2.091,
            -2.117,
            -2.137,
            -2.142,
            -2.146,
            -2.158,
        ),
        (
            -1.750,
            -1.823,
            -1.911,
            -1.968,
            -2.009,
            -2.057,
            -2.091,
            -2.103,
            -2.114,
            -2.135,
        ),
        (_NA, _NA, -1.888, -1.955, -1.998, -2.051, -2.087, -2.101, -2.111, -2.135),
        (_NA, _NA, _NA, -1.868, -1.923, -1.995, -2.042, -2.065, -2.081, -2.113),
        (_NA, _NA, _NA, -1.851, -1.912, -1.986, -2.036, -2.063, -2.079, -2.112),
        (_NA, _NA, _NA, -1.761, -1.835, -1.925, -1.987, -2.024, -2.046, -2.088),
    ),
    ("ct", "var"): (
        (1.132, 0.869, 0.763, 0.713, 0.690, 0.655, 0.633, 0.621, 0.610, 0.597),
        (1.453, 0.975, 0.845, 0.769, 0.734, 0.687, 0.654, 0.641, 0.627, 0.605),
        (1.627, 1.036, 0.882, 0.796, 0.756, 0.702, 0.661, 0.653, 0.634, 0.613),
        (2.482, 1.214, 0.983, 0.861, 0.808, 0.735, 0.688, 0.674, 0.650, 0.625),
        (3.947, 1.332, 1.052, 0.913, 0.845, 0.759, 0.705, 0.685, 0.662, 0.629),
        (_NA, _NA, 1.165, 0.991, 0.899, 0.792, 0.730, 0.705, 0.673, 0.638),
        (_NA, _NA, _NA, 1.055, 0.945, 0.828, 0.753, 0.725, 0.689, 0.650),
        (_NA, _NA, _NA, 1.145, 1.009, 0.872, 0.786, 0.747, 0.713, 0.661),
        (_NA, _NA, _NA, 1.208, 1.063, 0.902, 0.808, 0.766, 0.728, 0.670),
    ),
}

# MacKinnon (1994) approximate p-value coefficients for the single-series
# DF t-statistic (N = 1): Phi(polynomial(t)), small-p polynomial (degree 2)
# below the switch point, large-p polynomial (degree 3) above it. Same
# constants as statsmodels.tsa.adfvalues (_tau_smallps / _tau_largeps /
# _tau_stars, N = 1) and plm's p.approx = "MacKinnon1994".
_MACK94 = {
    "n": ((0.6344, 1.2378, 0.032496), (0.4797, 0.93557, -0.06999, 0.033066), -1.04),
    "c": ((2.1659, 1.4412, 0.038269), (1.7339, 0.93202, -0.12745, -0.010368), -1.61),
    "ct": ((3.2512, 1.6047, 0.049588), (2.5261, 0.61654, -0.37956, -0.060285), -2.89),
}


def mackinnon1994_pvalue(tstat: float, trend: str = "c") -> float:
    """MacKinnon (1994) approximate p-value of a Dickey-Fuller t-statistic.

    The approximation used by Stata's ``dfuller`` / ``xtunitroot fisher``
    and by ``plm::purtest(p.approx = "MacKinnon1994")``. Unlike
    ``statsmodels.tsa.adfvalues.mackinnonp`` it does not clip to 0 / 1
    outside the fitted range (plm does not either).
    """
    small, large, star = _MACK94[trend]
    x = float(tstat)
    if x <= star:
        z = small[0] + small[1] * x + small[2] * x * x
    else:
        z = large[0] + large[1] * x + large[2] * x * x + large[3] * x**3
    return float(stats.norm.cdf(z))


def _interp_T(T: float, grid: tuple) -> tuple:
    """(lo, hi, weight on hi) for linear interpolation on grid, clamped."""
    if T <= grid[0]:
        return 0, 0, 0.0
    if T >= grid[-1]:
        j = len(grid) - 1
        return j, j, 0.0
    hi = int(np.searchsorted(grid, T, side="left"))
    if grid[hi] == T:
        return hi, hi, 0.0
    lo = hi - 1
    return lo, hi, (T - grid[lo]) / (grid[hi] - grid[lo])


def _llc_adjust(T: float, trend: str, convention: str = "plm") -> tuple:
    # last row is the asymptotic value: plm interpolates towards it at
    # T = 500, Stata uses it directly for every T > 250
    if convention == "stata" and T > 250:
        return _LLC_ADJ[trend][-1]
    lo, hi, w = _interp_T(T, _LLC_T)
    tab = _LLC_ADJ[trend]
    mu = (1 - w) * tab[lo][0] + w * tab[hi][0]
    sig = (1 - w) * tab[lo][1] + w * tab[hi][1]
    return mu, sig


def _ips_moments(T: float, lags: int, trend: str) -> tuple:
    if not 0 <= lags <= 8:
        raise ValueError("IPS moments are tabulated for 0 <= lags <= 8")
    lo, hi, w = _interp_T(T, _IPS_T)
    m = _IPS[(trend, "mean")][lags]
    v = _IPS[(trend, "var")][lags]
    mu = (1 - w) * m[lo] + w * m[hi]
    var = (1 - w) * v[lo] + w * v[hi]
    if not (np.isfinite(mu) and np.isfinite(var)):
        raise ValueError(f"IPS moments are not tabulated for T = {T} with {lags} lags")
    return mu, var


class PanelUnitRootResult(ResultProtocolMixin):
    """Results from panel unit root test.

    Returned by :func:`panel_unitroot`. Holds the pooled test statistic and
    ``p``-value, the panel dimensions (``n_units``, ``n_periods``), the lag
    order, and the per-unit ADF statistics.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> N, T = 20, 30
    >>> rows = []
    >>> for i in range(N):
    ...     y = np.zeros(T)
    ...     for t in range(1, T):  # stationary AR(1), phi = 0.5
    ...         y[t] = 0.5 * y[t - 1] + rng.normal()
    ...     rows.extend({"id": i, "time": t, "y": y[t]} for t in range(T))
    >>> df = pd.DataFrame(rows)
    >>> result = sp.panel_unitroot(df, variable="y", id="id", time="time",
    ...                            test="ips")
    >>> type(result).__name__
    'PanelUnitRootResult'
    >>> result.n_units
    20
    >>> bool(0.0 <= result.p_value <= 1.0)
    True
    """

    def __init__(
        self,
        test_type: str,
        statistic: float,
        p_value: float,
        n_units: int,
        n_periods: int,
        individual_stats: Any,
        lags: Any,
        details: Optional[Dict[str, Any]] = None,
    ) -> None:
        self.test_type = test_type
        self.statistic = statistic
        self.p_value = p_value
        self.n_units = n_units
        self.n_periods = n_periods
        self.individual_stats = individual_stats
        self.lags = lags
        self.details = details or {}

    def summary(self) -> str:
        lines = [
            f"Panel Unit Root Test: {self.test_type}",
            "=" * 55,
            (
                "H0: All panels are stationary"
                if self.test_type.startswith("Hadri")
                else "H0: Panels contain unit roots"
            ),
            f"Ha: {'Some panels contain unit roots' if self.test_type.startswith('Hadri') else 'Panels are stationary'}",
            "",
            f"Statistic: {self.statistic:.4f}",
            f"P-value:   {self.p_value:.4f}",
            f"N units:   {self.n_units}",
            f"T periods: {self.n_periods}",
            "",
            f"Conclusion: {'Reject H0' if self.p_value < 0.05 else 'Fail to reject H0'} at 5%",
            "=" * 55,
        ]
        return "\n".join(lines)


def _det_matrix(n: int, trend: str, start: int = 1) -> Optional[np.ndarray]:
    if trend == "n":
        return None
    if trend == "c":
        return np.ones((n, 1))
    return np.column_stack([np.ones(n), np.arange(start, start + n, dtype=float)])


def _ols(X: np.ndarray, y: np.ndarray):
    b = np.linalg.lstsq(X, y, rcond=None)[0]
    r = y - X @ b
    return b, r


def _adf_unit(y: np.ndarray, lags: int, trend: str, dfcor: bool) -> Dict[str, Any]:
    """ADF regression of one series (Delta y_t on y_{t-1}, lagged diffs, det).

    Sample t = lags+1 .. T-1 (0-based), so n = T - lags - 1 observations.
    ``dfcor``: residual variance rss/(n - K) (True, OLS) or rss/n (False).
    Also returns the Frisch-Waugh residuals of Delta y and y_{t-1} on the
    lagged differences and deterministic terms (for LLC).
    """
    T = len(y)
    dy = np.diff(y)
    n = T - 1 - lags
    if n < 3 + lags:
        raise np.linalg.LinAlgError("too few observations")
    Y = dy[lags:]
    ylag = y[lags:-1]
    Z_parts = [dy[lags - j : T - 1 - j] for j in range(1, lags + 1)]
    det = _det_matrix(n, trend, start=lags + 2)
    if det is not None:
        Z_parts.append(det)
    Z = np.column_stack(Z_parts) if Z_parts else np.empty((n, 0))
    X = np.column_stack([ylag, Z])
    K = X.shape[1]
    if n <= K or np.linalg.matrix_rank(X) < K:
        raise np.linalg.LinAlgError("singular ADF design")
    b, r = _ols(X, Y)
    rss = float(r @ r)
    s2 = rss / (n - K) if dfcor else rss / n
    XtX_inv = np.linalg.inv(X.T @ X)
    se = float(np.sqrt(s2 * XtX_inv[0, 0]))
    if Z.shape[1] > 0:
        _, e = _ols(Z, Y)
        _, v = _ols(Z, ylag)
    else:
        e, v = Y.copy(), ylag.copy()
    return {
        "rho": float(b[0]),
        "se": se,
        "t": float(b[0] / se),
        "rss": rss,
        "n": n,
        "K": K,
        "e": e,
        "v": v,
        "dy": dy,
    }


def _bartlett_lrvar(w: np.ndarray, L: int) -> float:
    n = len(w)
    s = float(w @ w)
    for i in range(1, L + 1):
        s += 2.0 * (1.0 - i / (L + 1.0)) * float(w[i:] @ w[:-i])
    return s / n


def panel_unitroot(
    data: pd.DataFrame,
    variable: str,
    id: str = "id",
    time: str = "time",
    test: str = "ips",
    lags: Optional[int] = None,
    trend: str = "c",
    convention: str = "stata",
    dfcor: Optional[bool] = None,
    robust: bool = True,
) -> PanelUnitRootResult:
    """
    Panel unit root test.

    Equivalent to Stata's ``xtunitroot`` and R's ``plm::purtest()``.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data.
    variable : str
        Variable to test.
    id : str, default 'id'
        Unit identifier.
    time : str, default 'time'
        Time identifier.
    test : str, default 'ips'
        Test type: 'llc' (Levin-Lin-Chu), 'ips' (Im-Pesaran-Shin W-t-bar),
        'fisher' (Maddala-Wu inverse chi-squared of the per-unit ADF
        p-values; the inverse-normal, inverse-logit and modified statistics
        are in ``details``), 'hadri' (stationarity LM test).
    lags : int, optional
        Number of lagged differences in every ADF regression (Stata
        ``lags()``, plm ``lags = <int>``). If None, uses
        ``int(4 * (T / 100) ** 0.25)`` per unit (not an information
        criterion). Ignored by 'hadri'.
    trend : str, default 'c'
        'n' (none), 'c' (constant), 'ct' (constant + trend). IPS and Hadri
        require 'c' or 'ct'.
    convention : {'stata', 'plm'}, default 'stata'
        Where Stata ``xtunitroot`` and ``plm::purtest`` differ:

        * IPS: the IPS Table-3 moments are looked up at T = observations
          per panel (Stata) or T - lags - 1 (plm);
        * LLC: the LLC mean/std adjustments at T~ = T - lags - 1 (Stata) or
          T (plm); long-run-variance bandwidth ``int(3.21 T^(1/3))``
          (Stata) or ``round(3.21 T^(1/3))`` (plm); Stata normalises by the
          short-run variance rss/n and uses the se with divisor N (plm with
          ``dfcor = FALSE`` is identical on those two).
    dfcor : bool, optional
        Residual variance of the per-unit ADF regressions (IPS, Fisher,
        plm-LLC) and of the Hadri residuals: rss/(n - K) if True, rss/n if
        False. Default True for ``convention='stata'`` (OLS, as Stata's
        ``regress``) and False for ``'plm'`` (plm's default).
    robust : bool, default True
        Hadri only: heteroskedasticity-consistent version (unit-specific
        variances; Stata ``hadri, robust``, plm ``Hcons = TRUE``). False
        pools the variance (Stata default, plm ``Hcons = FALSE``).

    Returns
    -------
    PanelUnitRootResult
        ``details`` carries the intermediate quantities (LLC: delta, its
        se, t_delta, sbar, the adjustment terms; IPS: tbar, E[t], Var[t];
        Fisher: P, Z, L*, Pm and their p-values).

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np
    >>> import pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> N, T = 20, 30
    >>> rows = []
    >>> for unit in range(N):
    ...     y = np.zeros(T)
    ...     for t in range(1, T):  # stationary AR(1), phi = 0.5
    ...         y[t] = 0.5 * y[t - 1] + rng.normal()
    ...     rows.extend({"id": unit, "time": t, "y": y[t]} for t in range(T))
    >>> df = pd.DataFrame(rows)
    >>> result = sp.panel_unitroot(df, variable="y", id="id", time="time",
    ...                            test="ips")
    >>> result.n_units
    20
    >>> bool(0.0 <= result.p_value <= 1.0)
    True
    """
    if test not in ("ips", "llc", "fisher", "hadri"):
        raise ValueError(
            f"Unknown test: {test}. Use 'ips', 'llc', 'fisher', or 'hadri'."
        )
    if trend not in ("n", "c", "ct"):
        raise ValueError("trend must be 'n', 'c' or 'ct'")
    if convention not in ("stata", "plm"):
        raise ValueError("convention must be 'stata' or 'plm'")
    if dfcor is None:
        dfcor = convention == "stata"
    if test in ("ips", "hadri") and trend == "n":
        raise ValueError(
            f"test='{test}' needs trend='c' or 'ct' (no tabulated moments "
            "for the model without deterministic terms)"
        )

    units = data[id].unique()
    N = len(units)
    T_avg = data.groupby(id)[variable].count().mean()

    series = []
    n_short = 0
    for unit in units:
        y = data.loc[data[id] == unit].sort_values(time)[variable].dropna().values
        if len(y) < 5:
            n_short += 1
            continue
        series.append((unit, y.astype(float)))

    # ---------------------------------------------------------- Hadri
    if test == "hadri":
        if not series:
            raise ValueError("panel_unitroot('hadri'): no unit has >= 5 periods.")
        if n_short > 0:
            warnings.warn(
                f"panel_unitroot('hadri'): computed over {len(series)}/{N} "
                f"units. Excluded {n_short} unit(s) with <5 periods.",
                RuntimeWarning,
                stacklevel=2,
            )
        lengths = {len(y) for _, y in series}
        if len(lengths) > 1:
            raise ValueError("Hadri test requires a balanced panel")
        L = lengths.pop()
        n_det = 1 if trend == "c" else 2
        num, lm_i, rss_i = [], [], []
        for _, y in series:
            _, r = _ols(_det_matrix(L, trend), y)
            S2 = float(np.sum(np.cumsum(r) ** 2))
            rss = float(r @ r)
            s2i = rss / (L - n_det) if dfcor else rss / L
            num.append(S2 / L**2)
            lm_i.append(S2 / L**2 / s2i)
            rss_i.append(rss)
        n_u = len(series)
        if robust:
            LM = float(np.mean(lm_i))
        else:
            s2 = sum(rss_i) / (n_u * (L - n_det) if dfcor else n_u * L)
            LM = float(np.mean(num) / s2)
        mu, var = (1 / 6, 1 / 45) if trend == "c" else (1 / 15, 11 / 6300)
        Z = float(np.sqrt(n_u) * (LM - mu) / np.sqrt(var))
        return PanelUnitRootResult(
            test_type="Hadri (stationarity)",
            statistic=Z,
            p_value=float(stats.norm.sf(Z)),
            n_units=n_u,
            n_periods=int(L),
            individual_stats=pd.DataFrame({"unit": [u for u, _ in series], "LM": lm_i}),
            lags=None,
            details={"LM": LM, "mu": mu, "var": var, "robust": robust},
        )

    # ------------------------------------------------ per-unit ADF
    individual = []
    fits = []
    n_failed = 0
    for unit, y in series:
        L_i = int(np.floor(4 * (len(y) / 100) ** 0.25)) if lags is None else int(lags)
        L_i = min(L_i, len(y) // 3)
        try:
            f = _adf_unit(y, L_i, trend, dfcor)
        except np.linalg.LinAlgError:
            n_failed += 1
            individual.append(
                {"unit": unit, "t_stat": np.nan, "p_value": np.nan, "lags": L_i}
            )
            continue
        f["T"] = len(y)
        f["lags"] = L_i
        f["unit"] = unit
        f["y"] = y
        fits.append(f)
        individual.append(
            {
                "unit": unit,
                "t_stat": f["t"],
                "p_value": mackinnon1994_pvalue(f["t"], trend),
                "lags": L_i,
            }
        )
    ind_df = pd.DataFrame(individual)
    n_valid = len(fits)
    if n_valid == 0:
        raise ValueError(
            f"panel_unitroot('{test}'): no unit yielded a valid ADF statistic "
            f"({n_short}/{N} units had <5 periods, {n_failed} had a "
            f"singular ADF design). Cannot compute a panel unit-root test."
        )
    if n_short > 0 or n_failed > 0:
        warnings.warn(
            f"panel_unitroot('{test}'): computed over {n_valid}/{N} units. "
            f"Excluded {n_short} unit(s) with <5 periods and {n_failed} "
            f"unit(s) whose ADF regression was singular. The reported "
            f"statistic and n_units reflect only the {n_valid} valid units.",
            RuntimeWarning,
            stacklevel=2,
        )
    t_i = np.array([f["t"] for f in fits])

    if test == "ips":
        moments = []
        for f in fits:
            T_look = f["T"] if convention == "stata" else f["T"] - f["lags"] - 1
            moments.append(_ips_moments(T_look, f["lags"], trend))
        E = float(np.mean([m[0] for m in moments]))
        V = float(np.mean([m[1] for m in moments]))
        tbar = float(t_i.mean())
        W = float(np.sqrt(n_valid) * (tbar - E) / np.sqrt(V))
        return PanelUnitRootResult(
            test_type="Im-Pesaran-Shin (IPS)",
            statistic=W,
            p_value=float(stats.norm.cdf(W)),
            n_units=n_valid,
            n_periods=int(T_avg),
            individual_stats=ind_df,
            lags=lags,
            details={"tbar": tbar, "E_tbar": E, "Var_tbar": V},
        )

    if test == "fisher":
        p = np.array([mackinnon1994_pvalue(t, trend) for t in t_i])
        n = n_valid
        P = float(-2.0 * np.sum(np.log(p)))
        Zs = float(np.sum(stats.norm.ppf(p)) / np.sqrt(n))
        k = 3.0 * (5 * n + 4) / (np.pi**2 * n * (5 * n + 2))
        Lstar = float(np.sqrt(k) * np.sum(np.log(p / (1 - p))))
        Pm = float(np.sum(-2.0 * np.log(p) - 2.0) / (2.0 * np.sqrt(n)))
        details = {
            "P": P,
            "p_P": float(stats.chi2.sf(P, 2 * n)),
            "Z": Zs,
            "p_Z": float(stats.norm.cdf(Zs)),
            "L": Lstar,
            "df_L": 5 * n + 4,
            "p_L": float(stats.t.cdf(Lstar, 5 * n + 4)),
            "Pm": Pm,
            "p_Pm": float(stats.norm.sf(Pm)),
        }
        return PanelUnitRootResult(
            test_type="Fisher-type ADF",
            statistic=P,
            p_value=details["p_P"],
            n_units=n_valid,
            n_periods=int(T_avg),
            individual_stats=ind_df,
            lags=lags,
            details=details,
        )

    # ------------------------------------------------------------ LLC
    Ts = {f["T"] for f in fits}
    if len(Ts) > 1:
        raise ValueError("LLC test requires a balanced panel")
    T = Ts.pop()
    e_all, v_all, s_i, n_i = [], [], [], []
    for f in fits:
        if convention == "stata":
            s2_sr = f["rss"] / f["n"]
        else:
            s2_sr = f["rss"] / (f["n"] - f["K"]) if dfcor else f["rss"] / f["n"]
        sd = np.sqrt(s2_sr)
        e_all.append(f["e"] / sd)
        v_all.append(f["v"] / sd)
        n_i.append(f["n"])
        dy = f["dy"]
        if trend == "c":
            w = dy - dy.mean()
        elif trend == "ct":
            _, w = _ols(_det_matrix(len(dy), "ct"), dy)
        else:
            w = dy
        bw = 3.21 * T ** (1.0 / 3.0)
        Lh = int(bw) if convention == "stata" else int(np.floor(bw + 0.5))
        s_i.append(np.sqrt(_bartlett_lrvar(w, Lh)) / sd)
    e = np.concatenate(e_all)
    v = np.concatenate(v_all)
    Ntot = e.size
    delta = float(e @ v / (v @ v))
    rss = float(np.sum((e - delta * v) ** 2))
    if convention == "plm" and dfcor:
        se = float(np.sqrt(rss / (Ntot - 1) / (v @ v)))
    else:
        se = float(np.sqrt(rss / Ntot / (v @ v)))
    t_delta = delta / se
    sbar = float(np.mean(s_i))
    sig_e2 = rss / Ntot
    T_adj = float(np.mean(n_i)) if convention == "stata" else float(T)
    mu_adj, sig_adj = _llc_adjust(T_adj, trend, convention)
    t_star = float((t_delta - Ntot * sbar / sig_e2 * se * mu_adj) / sig_adj)
    return PanelUnitRootResult(
        test_type="Levin-Lin-Chu (LLC)",
        statistic=t_star,
        p_value=float(stats.norm.cdf(t_star)),
        n_units=n_valid,
        n_periods=int(T),
        individual_stats=ind_df,
        lags=lags,
        details={
            "delta": delta,
            "se_delta": se,
            "t_delta": float(t_delta),
            "sbar": sbar,
            "sigma_eps2": sig_e2,
            "mu_adj": mu_adj,
            "sigma_adj": sig_adj,
            "T_adj": T_adj,
        },
    )
