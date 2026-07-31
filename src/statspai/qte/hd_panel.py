"""High-dimensional panel quantile treatment effects.

Double-selection LASSO picks the controls, then a panel-appropriate quantile
regression estimates the effect at each tau.

.. warning::

    **Correctness fix in v1.21.0.**  Versions <= 1.20.0 removed unit and time
    effects by *within-demeaning* ``Y`` and ``D`` and then ran quantile
    regression on the demeaned data.  Quantile regression is not invariant to
    the within transformation, so the estimator was consistent only when the
    treatment effect was a pure location shift -- exactly the case in which a
    quantile treatment effect carries no information a mean effect does not.

    On a scale-shift design (``Y_it = u_i + (1 + d_it) e_it``, true
    ``QTE(tau) = Phi^-1(tau)``) the old code flattened the fan by ~45%:
    ``-0.707`` against a truth of ``-1.282`` at ``tau = 0.1``, and ``-0.148``
    against ``0`` at the median.

    Also fixed: the LASSO selected controls from the ``Y ~ X`` equation only
    (not Belloni-Chernozhukov-Hansen double selection), and two fallback
    paths returned a **hardcoded ``se = 0.1``** used to build confidence
    intervals.

Estimators
----------
``method='canay'`` (default)
    Canay (2011) two-step: estimate ``alpha_i`` from a mean fixed-effects
    regression, then run quantile regression on ``Y_it - alpha_hat_i``.
    **Requires the individual effect to be a pure location shift** and is a
    large-``T`` estimator (``alpha_hat_i`` carries ``O(T^-1/2)`` error).
``method='dummy_fe'``
    Quantile regression with explicit unit dummies (Koenker 2004, unshrunk).
    No location-shift restriction; incidental-parameter bias at small ``T``.
``method='pooled'``
    Pooled quantile regression, no individual effects.

References
----------
canay2011simple, koenker2004quantile, belloni2014inference, xu2025quantile
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from .._result_serialize import ResultProtocolMixin


@dataclass
class HDPanelQTEResult(ResultProtocolMixin):
    """Panel QTE at multiple quantiles with high-dimensional control selection.

    Returned by :func:`qte_hd_panel`.

    Attributes
    ----------
    quantiles, qte, se, ci_low, ci_high : ndarray
    selected_controls : list of str
        Union of the ``Y ~ X`` and ``D ~ X`` LASSO selections.
    n_obs, n_units, n_periods : int
    method, se_method : str
    diagnostics : dict
        Assumption flags, notably Canay's location-shift requirement and the
        panel length driving its bias.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(42)
    >>> rows = []
    >>> for u in range(80):
    ...     ui = rng.normal(0, 0.5)
    ...     treated = u >= 40
    ...     for t in range(10):
    ...         d = 1.0 if (treated and t >= 5) else 0.0
    ...         x1, x2, x3 = rng.normal(0, 1, 3)
    ...         y = 1.0 + 1.2 * d + 0.5 * x1 + ui + rng.normal(0, 1)
    ...         rows.append((u, t, y, d, x1, x2, x3))
    >>> df = pd.DataFrame(
    ...     rows, columns=["unit", "time", "y", "d", "x1", "x2", "x3"])
    >>> res = sp.qte_hd_panel(
    ...     df, y="y", treat="d", unit="unit", time="time",
    ...     covariates=["x1", "x2", "x3"],
    ...     quantiles=np.array([0.25, 0.5, 0.75]), se="none")
    >>> isinstance(res, sp.HDPanelQTEResult)
    True
    >>> bool(np.all(np.abs(res.qte - 1.2) < 0.4))  # true effect 1.2
    True
    """

    quantiles: np.ndarray
    qte: np.ndarray
    se: np.ndarray
    ci_low: np.ndarray
    ci_high: np.ndarray
    selected_controls: List[str]
    n_obs: int
    n_units: int = 0
    n_periods: int = 0
    method: str = "canay"
    se_method: str = "bootstrap"
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        rows = [
            f"High-Dimensional Panel QTE ({self.method})",
            "=" * 58,
            f"  N            : {self.n_obs:,}"
            f"  ({self.n_units} units x {self.n_periods} periods)",
            f"  Controls kept: {len(self.selected_controls)}"
            f" {self.selected_controls}",
            f"  SE method    : {self.se_method}",
            "",
            "  Quantile  QTE        SE        95% CI",
            "  " + "-" * 50,
        ]
        for q, t, s, lo, hi in zip(
            self.quantiles, self.qte, self.se, self.ci_low, self.ci_high
        ):
            se_txt = "     n/a" if not np.isfinite(s) else f"{s:8.4f}"
            ci_txt = (
                "        n/a"
                if not (np.isfinite(lo) and np.isfinite(hi))
                else f"[{lo:+.4f}, {hi:+.4f}]"
            )
            rows.append(f"  {q:6.2f}   {t:+.4f}  {se_txt}  {ci_txt}")
        if self.diagnostics.get("warnings"):
            rows += ["", "  Assumptions:"]
            rows += [f"    - {w}" for w in self.diagnostics["warnings"]]
        out = "\n".join(rows)
        print(out)
        return out

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "quantile": self.quantiles,
                "qte": self.qte,
                "se": self.se,
                "ci_low": self.ci_low,
                "ci_high": self.ci_high,
            }
        )

    def plot(self, ax: Any = None) -> Any:
        """Plot the QTE curve with its CI band. Returns (fig, ax)."""
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))
        else:
            fig = ax.get_figure()
        ax.plot(self.quantiles, self.qte, "o-", color="#2c7bb6", lw=2, label="QTE")
        if np.isfinite(self.ci_low).all() and np.isfinite(self.ci_high).all():
            ax.fill_between(
                self.quantiles, self.ci_low, self.ci_high, alpha=0.2, color="#2c7bb6"
            )
        ax.axhline(0, color="grey", ls="--", lw=0.8)
        ax.set_xlabel("Quantile (tau)")
        ax.set_ylabel("Treatment effect")
        ax.set_title(f"Panel QTE ({self.method})")
        ax.legend()
        fig.tight_layout()
        return fig, ax


# ══════════════════════════════════════════════════════════════════════
#  Internals
# ══════════════════════════════════════════════════════════════════════


def _double_selection(
    Y: np.ndarray,
    D: np.ndarray,
    X: np.ndarray,
    names: List[str],
    lasso_alpha: Optional[float],
) -> Tuple[np.ndarray, List[str]]:
    """Belloni-Chernozhukov-Hansen double selection.

    LASSO ``Y ~ X`` and ``D ~ X`` separately, keep the **union**. Selecting on
    the outcome equation alone (what this module did before 1.21.0) drops
    covariates that predict treatment but not the outcome -- exactly the
    omitted-variable channel double selection exists to close.

    ``X`` is standardised first, so ``lasso_alpha`` is scale-free.
    """
    from sklearn.linear_model import Lasso

    sd = X.std(axis=0)
    sd[sd == 0] = 1.0
    Xs = (X - X.mean(axis=0)) / sd

    if lasso_alpha is None:
        n, p = Xs.shape
        c, gamma = 1.1, 0.1 / max(np.log(max(n, 2)), 1.0)
        lam = c * stats.norm.ppf(1.0 - gamma / (2.0 * max(p, 1))) / np.sqrt(max(n, 1))
        lasso_alpha = float(max(lam * np.std(Y), 1e-6))

    keep = np.zeros(X.shape[1], dtype=bool)
    for target in (Y, D):
        if np.std(target) == 0:
            continue
        fit = Lasso(alpha=lasso_alpha, max_iter=5000).fit(Xs, target)
        keep |= np.abs(fit.coef_) > 1e-10

    if not keep.any():
        keep[:] = True  # never silently drop every control
    idx = np.where(keep)[0]
    return X[:, idx], [names[i] for i in idx]


def _quantreg(Y: np.ndarray, X: np.ndarray, tau: float) -> np.ndarray:
    """Quantile regression coefficients. Raises rather than fabricating."""
    import statsmodels.regression.quantile_regression as smq

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fit = smq.QuantReg(Y, X).fit(q=tau, max_iter=5000)
    return np.asarray(fit.params)


def _group_mean(v: np.ndarray, codes: np.ndarray, k: int) -> np.ndarray:
    out = np.bincount(codes, weights=v, minlength=k)
    cnt = np.bincount(codes, minlength=k).astype(float)
    cnt[cnt == 0] = 1.0
    return out / cnt


def _group_mean_2d(M: np.ndarray, codes: np.ndarray, k: int) -> np.ndarray:
    return np.column_stack([_group_mean(M[:, j], codes, k) for j in range(M.shape[1])])


def _canay_step1(
    Y: np.ndarray, W: np.ndarray, unit_codes: np.ndarray, n_units: int
) -> np.ndarray:
    """Canay (2011) step 1: individual effects from a mean FE regression."""
    Yc = Y - _group_mean(Y, unit_codes, n_units)[unit_codes]
    Wc = W - _group_mean_2d(W, unit_codes, n_units)[unit_codes]
    beta, *_ = np.linalg.lstsq(Wc, Yc, rcond=None)
    resid = Y - W @ beta
    return np.asarray(_group_mean(resid, unit_codes, n_units))


def _fit_once(
    Y: np.ndarray,
    D: np.ndarray,
    Xs: np.ndarray,
    unit_codes: np.ndarray,
    time_dummies: np.ndarray,
    n_units: int,
    taus: np.ndarray,
    method: str,
) -> np.ndarray:
    """QTE at each tau for one (possibly resampled) dataset."""
    n = len(Y)
    W = np.column_stack([D.reshape(-1, 1), Xs, time_dummies])

    if method == "canay":
        alpha = _canay_step1(Y, np.column_stack([np.ones(n), W]), unit_codes, n_units)
        Y_adj = Y - alpha[unit_codes]
        design = np.column_stack([np.ones(n), W])
    elif method == "dummy_fe":
        dummies = np.zeros((n, n_units))
        dummies[np.arange(n), unit_codes] = 1.0
        design = np.column_stack([np.ones(n), W, dummies[:, 1:]])
        Y_adj = Y
    elif method == "pooled":
        design = np.column_stack([np.ones(n), W])
        Y_adj = Y
    else:  # pragma: no cover - guarded by the public wrapper
        raise ValueError(method)

    out = np.empty(len(taus))
    for j, tau in enumerate(taus):
        out[j] = _quantreg(Y_adj, design, float(tau))[1]  # D is column 1
    return out


# ══════════════════════════════════════════════════════════════════════
#  Public API
# ══════════════════════════════════════════════════════════════════════


def qte_hd_panel(
    data: pd.DataFrame,
    y: str,
    treat: str,
    unit: str,
    time: str,
    covariates: List[str],
    quantiles: Optional[np.ndarray] = None,
    alpha: float = 0.05,
    method: str = "canay",
    lasso_alpha: Optional[float] = None,
    se: str = "bootstrap",
    n_boot: int = 200,
    seed: int = 0,
) -> HDPanelQTEResult:
    """Panel quantile treatment effects with high-dimensional controls.

    Parameters
    ----------
    data : DataFrame
        Long-format panel.
    y, treat, unit, time : str
    covariates : list of str
        Candidate control set; narrowed by double-selection LASSO.
    quantiles : array-like, optional
        Defaults to ``(0.1, 0.25, 0.5, 0.75, 0.9)``.
    alpha : float, default 0.05
    method : {'canay', 'dummy_fe', 'pooled'}
        See the module docstring. ``'canay'`` assumes the individual effect is
        a pure location shift and needs a reasonably long panel.
    lasso_alpha : float, optional
        Penalty on standardised covariates. ``None`` uses the
        Belloni-Chernozhukov-Hansen plug-in penalty.
    se : {'bootstrap', 'none'}
        ``'bootstrap'`` resamples **units**, preserving within-unit
        dependence. There is no analytic option: Canay's two-step variance
        depends on the first step, and a naive quantile-regression SE would
        understate it.
    n_boot : int, default 200
    seed : int

    Returns
    -------
    HDPanelQTEResult

    Notes
    -----
    .. versionchanged:: 1.21.0
        Rebuilt. The previous implementation within-demeaned before running
        quantile regression, selected controls from the outcome equation
        only, and fabricated ``se = 0.1`` on two fallback paths.
        See MIGRATION.md.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(42)
    >>> rows = []
    >>> for u in range(80):
    ...     ui = rng.normal(0, 0.5)
    ...     treated = u >= 40
    ...     for t in range(10):
    ...         d = 1.0 if (treated and t >= 5) else 0.0
    ...         x1, x2, x3 = rng.normal(0, 1, 3)
    ...         y = 1.0 + 1.2 * d + 0.5 * x1 + ui + rng.normal(0, 1)
    ...         rows.append((u, t, y, d, x1, x2, x3))
    >>> df = pd.DataFrame(
    ...     rows, columns=["unit", "time", "y", "d", "x1", "x2", "x3"])
    >>> res = sp.qte_hd_panel(
    ...     df, y="y", treat="d", unit="unit", time="time",
    ...     covariates=["x1", "x2", "x3"],
    ...     quantiles=np.array([0.25, 0.5, 0.75]), se="none")
    >>> bool(np.all(np.abs(res.qte - 1.2) < 0.4))  # true effect 1.2
    True

    References
    ----------
    canay2011simple, belloni2014inference, xu2025quantile
    """
    if method not in ("canay", "dummy_fe", "pooled"):
        raise ValueError(
            f"method must be 'canay', 'dummy_fe' or 'pooled', got {method!r}"
        )
    if se not in ("bootstrap", "none"):
        raise ValueError(f"se must be 'bootstrap' or 'none', got {se!r}")
    if quantiles is None:
        quantiles = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
    taus = np.atleast_1d(np.asarray(quantiles, dtype=float))
    if np.any((taus <= 0) | (taus >= 1)):
        raise ValueError("quantiles must lie strictly inside (0, 1).")

    cov = list(covariates)
    df = data[[y, treat, unit, time] + cov].dropna().reset_index(drop=True)
    Y = df[y].to_numpy(float)
    D = df[treat].to_numpy(float)
    X = df[cov].to_numpy(float)
    n = len(df)

    unit_cat = pd.Categorical(df[unit])
    unit_codes = np.asarray(unit_cat.codes, dtype=int)
    n_units = len(unit_cat.categories)
    time_cat = pd.Categorical(df[time])
    n_periods = len(time_cat.categories)
    td = np.zeros((n, n_periods))
    td[np.arange(n), np.asarray(time_cat.codes, dtype=int)] = 1.0
    time_dummies = td[:, 1:]

    Xs, sel_names = _double_selection(Y, D, X, cov, lasso_alpha)

    diag_warnings: List[str] = []
    if method == "canay":
        diag_warnings.append(
            "Canay (2011) assumes the individual effect is a pure LOCATION "
            "shift (identical alpha_i at every quantile). If unit effects "
            "differ across quantiles the estimator is inconsistent."
        )
        avg_T = n / max(n_units, 1)
        if avg_T < 5:
            msg = (
                f"qte_hd_panel(method='canay'): average T = {avg_T:.1f} is "
                "short. Canay's first-step alpha_hat_i carries O(T^-1/2) "
                "error, so the quantile estimates retain finite-T bias. "
                "Prefer T >= 10, or compare against method='dummy_fe'."
            )
            warnings.warn(msg, UserWarning, stacklevel=2)
            diag_warnings.append(msg)
    if method == "dummy_fe" and n / max(n_units, 1) < 10:
        diag_warnings.append(
            "dummy_fe with short T suffers the incidental-parameter problem."
        )

    qte_arr = _fit_once(Y, D, Xs, unit_codes, time_dummies, n_units, taus, method)

    if se == "none":
        se_arr = np.full(len(taus), np.nan)
        ci_low = np.full(len(taus), np.nan)
        ci_high = np.full(len(taus), np.nan)
    else:
        rng = np.random.default_rng(seed)
        rows_by_unit = [np.where(unit_codes == u)[0] for u in range(n_units)]
        boot = np.full((n_boot, len(taus)), np.nan)
        n_failed = 0
        for b in range(n_boot):
            pick = rng.integers(0, n_units, size=n_units)
            idx = np.concatenate([rows_by_unit[u] for u in pick])
            # Relabel resampled units, or repeated draws collapse into one
            # unit and the individual effects come out wrong.
            new_codes = np.concatenate(
                [np.full(len(rows_by_unit[u]), k) for k, u in enumerate(pick)]
            )
            try:
                boot[b] = _fit_once(
                    Y[idx],
                    D[idx],
                    Xs[idx],
                    new_codes,
                    time_dummies[idx],
                    n_units,
                    taus,
                    method,
                )
            except Exception:  # noqa: BLE001 - counted and surfaced below
                n_failed += 1
        n_ok = np.isfinite(boot).sum(axis=0)
        se_arr = np.where(n_ok >= 2, np.nanstd(boot, axis=0, ddof=1), np.nan)
        z = float(stats.norm.ppf(1 - alpha / 2))
        ci_low = qte_arr - z * se_arr
        ci_high = qte_arr + z * se_arr
        if n_failed:
            warnings.warn(
                f"qte_hd_panel: {n_failed}/{n_boot} cluster-bootstrap "
                "replications failed; SEs use the remainder. NaN SEs mean "
                "fewer than two usable replications -- they are NOT a "
                "placeholder value.",
                RuntimeWarning,
                stacklevel=2,
            )
            diag_warnings.append(f"{n_failed}/{n_boot} bootstrap draws failed")

    result = HDPanelQTEResult(
        quantiles=taus,
        qte=qte_arr,
        se=se_arr,
        ci_low=ci_low,
        ci_high=ci_high,
        selected_controls=sel_names,
        n_obs=n,
        n_units=n_units,
        n_periods=n_periods,
        method=method,
        se_method=se,
        diagnostics={
            "warnings": diag_warnings,
            "avg_T": n / max(n_units, 1),
            "n_candidate_controls": len(cov),
            "n_selected_controls": len(sel_names),
        },
    )
    try:
        from ..output._lineage import attach_provenance as _attach_prov

        _attach_prov(
            result,
            function="sp.qte.qte_hd_panel",
            params={
                "y": y,
                "treat": treat,
                "unit": unit,
                "time": time,
                "covariates": cov,
                "quantiles": list(taus),
                "alpha": alpha,
                "method": method,
                "lasso_alpha": lasso_alpha,
                "se": se,
                "n_boot": n_boot,
                "seed": seed,
            },
            data=data,
            overwrite=False,
        )
    except Exception:  # pragma: no cover - provenance is best-effort metadata
        pass
    return result
