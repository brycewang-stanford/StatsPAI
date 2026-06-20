"""Local Projections (Jordà, AER 2005).

Estimates impulse responses by running horizon-by-horizon regressions:

    y_{t+h} - y_{t-1} = α_h + β_h · shock_t
                         + γ_h · controls_t + ε_{t,h}

for h = 0, 1, …, H. The sequence ``{β_h}`` traces the impulse response
function. Inference uses Newey-West HAC standard errors (required because
the residuals are serially correlated by construction).

This estimator is much more flexible than a VAR-based IRF — no
finite-lag structure imposed — and is the default in modern empirical
macro (Ramey 2016; Plagborg-Møller & Wolf 2021).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from ..exceptions import DataInsufficient
from .._result_serialize import ResultProtocolMixin

# --------------------------------------------------------------------- #
#  Newey-West HAC for a single regression
# --------------------------------------------------------------------- #


def _newey_west(X: np.ndarray, e: np.ndarray, lags: int) -> np.ndarray:
    """Newey-West HAC covariance matrix for an OLS regression with
    residuals ``e``. Bartlett kernel weights up to ``lags``.
    """
    n, k = X.shape
    XtX_inv = np.linalg.inv(X.T @ X)
    # meat: sum_l w_l (Gamma_l + Gamma_l') where
    # Gamma_l = (1/n) sum_t X_t e_t e_{t-l} X_{t-l}'
    omega = np.zeros((k, k))
    u = X * e[:, None]  # (n, k)
    for lag in range(lags + 1):
        if lag == 0:
            G = u.T @ u
        else:
            G = u[lag:].T @ u[:-lag]
            G = G + G.T
        w = 1.0 - lag / (lags + 1.0)
        omega += w * G
    V = XtX_inv @ omega @ XtX_inv
    return V


# --------------------------------------------------------------------- #
#  Result
# --------------------------------------------------------------------- #


@dataclass
class LocalProjectionsResult(ResultProtocolMixin):
    """Impulse-response container returned by :func:`local_projections`.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(3)
    >>> T = 200
    >>> shock = rng.normal(size=T)
    >>> y = np.zeros(T)
    >>> for t in range(1, T):
    ...     y[t] = 0.5 * y[t - 1] + 0.8 * shock[t] + rng.normal(0, 0.5)
    >>> df = pd.DataFrame({"y": y, "shock": shock})
    >>> res = sp.local_projections(df, outcome="y", shock="shock", horizons=8)
    >>> type(res).__name__
    'LocalProjectionsResult'
    >>> len(res.horizons) == 9
    True
    >>> bool(res.irf[0] > 0)
    True
    """

    horizons: np.ndarray  # (H+1,)
    irf: np.ndarray  # (H+1,)
    se: np.ndarray  # (H+1,)
    ci_lower: np.ndarray  # (H+1,)
    ci_upper: np.ndarray  # (H+1,)
    alpha: float
    shock_name: str
    outcome_name: str
    n_obs_per_horizon: np.ndarray  # (H+1,)  — usable sample per horizon

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "horizon": self.horizons,
                "irf": self.irf,
                "se": self.se,
                "ci_lower": self.ci_lower,
                "ci_upper": self.ci_upper,
                "n": self.n_obs_per_horizon,
            }
        )

    def plot(self, ax: Any = None, **kwargs: Any) -> Any:
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=(7, 4))
        ax.plot(
            self.horizons,
            self.irf,
            "-o",
            color="C0",
            label=f"IRF of {self.outcome_name}",
        )
        ax.fill_between(
            self.horizons,
            self.ci_lower,
            self.ci_upper,
            alpha=0.25,
            color="C0",
            label=f"{int((1 - self.alpha) * 100)}% CI",
        )
        ax.axhline(0, color="grey", linewidth=0.6)
        ax.set_xlabel("Horizon h")
        ax.set_ylabel(f"Response to {self.shock_name}")
        ax.legend()
        return ax

    def summary(self) -> str:
        df = self.to_frame()
        lines = [
            f"Local Projections — {self.outcome_name} on {self.shock_name}",
            "-" * 55,
            df.to_string(index=False, float_format=lambda x: f"{x: .4f}"),
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()


# --------------------------------------------------------------------- #
#  lpirfs-compatible Cholesky local projections
# --------------------------------------------------------------------- #


def _lpirfs_cholesky(
    data: pd.DataFrame,
    outcome: str,
    shock: str,
    endog_order: Optional[List[str]],
    horizons: int,
    nw_lags: Optional[int],
    alpha: float,
) -> LocalProjectionsResult:
    """Replicate ``lpirfs::lp_lin(..., shock_type=1, lags_endog_lin=1)``.

    The R implementation first estimates a reduced VAR(1), builds a
    unit-shock Cholesky matrix from the residual covariance, then runs
    horizon-specific OLS projections and maps the coefficient matrices
    through the Cholesky shock vector.
    """
    order = list(endog_order) if endog_order is not None else [outcome, shock]
    if outcome not in order:
        raise ValueError("endog_order must include outcome")
    if shock not in order:
        raise ValueError("endog_order must include shock")
    if len(order) != len(set(order)):
        raise ValueError("endog_order must not contain duplicate variables")
    missing = [col for col in order if col not in data.columns]
    if missing:
        raise KeyError(f"endog_order columns not found in data: {missing}")
    if len(data) <= horizons + 2:
        raise DataInsufficient("too few observations for Cholesky local projections")

    y_all = data.loc[:, order].to_numpy(dtype=float)
    if np.isnan(y_all).any():
        raise ValueError("Cholesky local projections require complete endog_order data")

    k_endog = y_all.shape[1]
    response_idx = order.index(outcome)
    shock_idx = order.index(shock)

    # lpirfs::create_lin_data with lags_endog_lin = 1.
    y_lin = y_all[1:, :]
    x_lin = y_all[:-1, :]
    x_var = np.column_stack([np.ones(len(x_lin)), x_lin])

    residuals = np.empty_like(y_lin)
    for k in range(k_endog):
        beta, *_ = np.linalg.lstsq(x_var, y_lin[:, k], rcond=None)
        residuals[:, k] = y_lin[:, k] - x_var @ beta

    cov = np.cov(residuals, rowvar=False, ddof=1)
    chol = np.linalg.cholesky(cov)
    shock_matrix = np.empty_like(chol)
    for j in range(k_endog):
        shock_matrix[:, j] = chol[:, j] / chol[j, j]
    shock_vec = shock_matrix[:, shock_idx]

    irf = np.empty(horizons + 1)
    se = np.empty(horizons + 1)
    n_used = np.empty(horizons + 1, dtype=int)
    irf[0] = float(shock_vec[response_idx])
    se[0] = 0.0
    n_used[0] = int(len(y_lin))

    for h in range(1, horizons + 1):
        yy = y_lin[h - 1 :, :]
        xx = x_lin[: len(x_lin) - h + 1, :]
        x_h = np.column_stack([np.ones(len(xx)), xx])
        coef = np.empty((k_endog, k_endog))
        half_width = np.empty((k_endog, k_endog))
        lag_nw = h if nw_lags is None else nw_lags
        for k in range(k_endog):
            beta, *_ = np.linalg.lstsq(x_h, yy[:, k], rcond=None)
            resid = yy[:, k] - x_h @ beta
            cov_beta = _newey_west(x_h, resid, lags=lag_nw)
            std = np.sqrt(np.maximum(np.diag(cov_beta), 0.0))
            coef[k, :] = beta[1 : k_endog + 1]
            half_width[k, :] = std[1 : k_endog + 1]
        point = coef @ shock_vec
        # lpirfs stores lower/upper bands for each coefficient before the
        # Cholesky map; for the two-variable unit-shock case this is exactly
        # the coefficient SE on the ordered shock variable, and this formula
        # also preserves the sign of nonzero Cholesky weights.
        lower = (coef - half_width) @ shock_vec
        upper = (coef + half_width) @ shock_vec
        irf[h] = float(point[response_idx])
        se[h] = float((upper[response_idx] - lower[response_idx]) / 2.0)
        n_used[h] = int(len(xx))

    z_crit = stats.norm.ppf(1 - alpha / 2)
    return LocalProjectionsResult(
        horizons=np.arange(horizons + 1),
        irf=irf,
        se=se,
        ci_lower=irf - z_crit * se,
        ci_upper=irf + z_crit * se,
        alpha=alpha,
        shock_name=shock,
        outcome_name=outcome,
        n_obs_per_horizon=n_used,
    )


# --------------------------------------------------------------------- #
#  Public entry point
# --------------------------------------------------------------------- #


def local_projections(
    data: pd.DataFrame,
    outcome: str,
    shock: str,
    controls: Optional[List[str]] = None,
    horizons: int = 20,
    nw_lags: Optional[int] = None,
    alpha: float = 0.05,
    cumulative: bool = False,
    auto_lag: bool = True,
    identification: str = "direct",
    endog_order: Optional[List[str]] = None,
) -> LocalProjectionsResult:
    """Estimate impulse responses via Jordà (2005) local projections.

    Parameters
    ----------
    data : pd.DataFrame
        Time-ordered panel (single series for now — panel extension TBD).
    outcome : str
        Column name of the outcome variable y.
    shock : str
        Column name of the shock / treatment variable.
    controls : list of str, optional
        Additional regressors taken **verbatim** from ``data``: the
        column values at time t are used directly, without re-lagging.
        If you want the lag of a control, lag it yourself before
        passing it in (e.g. ``df["unemp_lag"] = df["unemp"].shift(1)``
        and then ``controls=["unemp_lag"]``). The pre-1.16 behaviour
        silently re-lagged controls a second time on top of an auto-
        added ``y_{t-1}``, producing collinear columns and surprising
        impulse responses; see ``MIGRATION.md`` for context.
    horizons : int, default 20
        Number of horizons h = 0, 1, …, H to estimate.
    nw_lags : int, optional
        Newey-West truncation lag. Defaults to ``round(1.5 * horizons)``
        per Kilian & Kim (2011) recommendation.
    alpha : float, default 0.05
        Significance level for the confidence band.
    cumulative : bool, default False
        If ``True``, return the cumulative response
        ``y_{t+h} - y_{t-1}``. Default (False) returns ``y_{t+h}``
        directly.
    auto_lag : bool, default True
        If ``True`` (the legacy default), also adds ``y_{t-1}`` and
        ``shock_{t-1}`` as automatic regressors. Set ``False`` for a
        bare ``y_{t+h} ~ const + shock_t + controls`` specification.
        These two auto-controls were silent in the pre-1.16 docstring.
    identification : {'direct', 'lpirfs_cholesky'}, default 'direct'
        Shock-identification convention. ``'direct'`` uses the coefficient
        on the observed ``shock`` variable in each horizon regression.
        ``'lpirfs_cholesky'`` reproduces ``lpirfs::lp_lin`` with
        ``lags_endog_lin=1`` and ``shock_type=1``: the variables in
        ``endog_order`` define the Cholesky ordering, and the reported
        response is the unit structural shock for ``shock``.
    endog_order : list of str, optional
        Endogenous variable order used only when
        ``identification='lpirfs_cholesky'``. Defaults to
        ``[outcome, shock]``.

    Examples
    --------
    Impulse response of an AR(1) outcome to an observed shock:

    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(3)
    >>> T = 200
    >>> shock = rng.normal(size=T)
    >>> y = np.zeros(T)
    >>> for t in range(1, T):
    ...     y[t] = 0.5 * y[t - 1] + 0.8 * shock[t] + rng.normal(0, 0.5)
    >>> df = pd.DataFrame({"y": y, "shock": shock})
    >>> res = sp.local_projections(df, outcome="y", shock="shock", horizons=8)
    >>> print(round(float(res.irf[0]), 2))  # 0.75 — impact response, truth 0.8
    >>> print(res.to_frame().head())        # horizon, irf, se, ci bands, n
    >>> ax = res.plot()                     # IRF with Newey-West CI band
    """
    if controls is None:
        controls = []
    identification_norm = identification.lower().replace("-", "_")
    if identification_norm not in {"direct", "lpirfs_cholesky", "cholesky"}:
        raise ValueError("identification must be 'direct' or 'lpirfs_cholesky'")
    if identification_norm in {"lpirfs_cholesky", "cholesky"}:
        if controls:
            raise ValueError(
                "controls are not supported with " "identification='lpirfs_cholesky'"
            )
        if cumulative:
            raise ValueError(
                "cumulative=True is not supported with "
                "identification='lpirfs_cholesky'"
            )
        _result = _lpirfs_cholesky(
            data=data,
            outcome=outcome,
            shock=shock,
            endog_order=endog_order,
            horizons=horizons,
            nw_lags=nw_lags,
            alpha=alpha,
        )
        try:
            from ..output._lineage import attach_provenance as _attach_prov

            _attach_prov(
                _result,
                function="sp.timeseries.local_projections",
                params={
                    "outcome": outcome,
                    "shock": shock,
                    "controls": None,
                    "horizons": horizons,
                    "nw_lags": nw_lags,
                    "alpha": alpha,
                    "cumulative": cumulative,
                    "auto_lag": auto_lag,
                    "identification": identification,
                    "endog_order": (
                        list(endog_order)
                        if endog_order is not None
                        else [outcome, shock]
                    ),
                },
                data=data,
                overwrite=False,
            )
        except Exception:  # pragma: no cover
            pass
        return _result
    df = data.copy().reset_index(drop=True)
    n = len(df)
    if nw_lags is None:
        nw_lags = max(1, int(round(1.5 * horizons)))

    y = df[outcome].to_numpy(dtype=float)
    s = df[shock].to_numpy(dtype=float)
    # Auto-added pre-treatment controls (y_{t-1}, shock_{t-1}) — only
    # used when auto_lag is True. User-supplied controls are taken at
    # face value (no re-lagging) so the design matrix matches what a
    # reader of the docstring would expect; see the docstring note
    # above and tests/r_parity/34_lp for the parity contract.
    lag_y = np.concatenate([[np.nan], y[:-1]])
    lag_s = np.concatenate([[np.nan], s[:-1]])
    extra_cols = []
    for c in controls:
        col = df[c].to_numpy(dtype=float)
        extra_cols.append(col)

    irf = np.empty(horizons + 1)
    se = np.empty(horizons + 1)
    n_used = np.empty(horizons + 1, dtype=int)

    for h in range(horizons + 1):
        # LHS: y_{t+h} (or y_{t+h} - y_{t-1} if cumulative)
        if t_end := n - h:
            y_lhs = y[h:]
            lhs = y_lhs - lag_y[h:] if cumulative else y_lhs
        else:
            raise DataInsufficient("too few observations for horizon")
        shock_t = s[:t_end]
        extras_t = [arr[:t_end] for arr in extra_cols]

        cols = [np.ones(t_end), shock_t]
        if auto_lag:
            cols.append(lag_y[:t_end])
            cols.append(lag_s[:t_end])
        cols.extend(extras_t)
        X = np.column_stack(cols)
        valid = ~np.isnan(lhs) & ~np.any(np.isnan(X), axis=1)
        X_use = X[valid]
        lhs_use = lhs[valid]
        if len(X_use) <= X_use.shape[1] + 2:
            irf[h] = np.nan
            se[h] = np.nan
            n_used[h] = int(valid.sum())
            continue
        beta, *_ = np.linalg.lstsq(X_use, lhs_use, rcond=None)
        e = lhs_use - X_use @ beta
        V = _newey_west(X_use, e, lags=max(h, nw_lags))
        irf[h] = float(beta[1])  # coefficient on shock
        se[h] = float(np.sqrt(max(V[1, 1], 0)))
        n_used[h] = int(valid.sum())

    z_crit = stats.norm.ppf(1 - alpha / 2)
    _result = LocalProjectionsResult(
        horizons=np.arange(horizons + 1),
        irf=irf,
        se=se,
        ci_lower=irf - z_crit * se,
        ci_upper=irf + z_crit * se,
        alpha=alpha,
        shock_name=shock,
        outcome_name=outcome,
        n_obs_per_horizon=n_used,
    )
    try:
        from ..output._lineage import attach_provenance as _attach_prov

        _attach_prov(
            _result,
            function="sp.timeseries.local_projections",
            params={
                "outcome": outcome,
                "shock": shock,
                "controls": list(controls) if controls else None,
                "horizons": horizons,
                "nw_lags": nw_lags,
                "alpha": alpha,
                "cumulative": cumulative,
                "auto_lag": auto_lag,
                "identification": identification,
                "endog_order": (list(endog_order) if endog_order is not None else None),
            },
            data=data,
            overwrite=False,
        )
    except Exception:  # pragma: no cover
        pass
    return _result
