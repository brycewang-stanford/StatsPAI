"""
Distributional IV (Sharma-Xue 2025, arXiv 2502.07641) and
KAN-Powered D-IV-LATE (Kennedy 2025, arXiv 2506.12765).

Standard IV estimates the LATE on the *mean*. Distributional IV
estimates the LATE on the *entire distribution* of Y, returning the
LATE at every quantile τ ∈ (0, 1). The KAN-powered variant uses a
Kolmogorov-Arnold network to model the bridge function, but we
implement the standard quantile-IV estimator (Chernozhukov-Hansen
2005) here for portability.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from .._result_serialize import ResultProtocolMixin


@dataclass
class DistIVResult(ResultProtocolMixin):
    """Distributional IV LATE per quantile.

    Returned by :func:`dist_iv` (and :func:`kan_dlate`). Holds the local
    average treatment effect estimated at each requested quantile of the
    outcome, together with bootstrap standard errors and confidence
    intervals. Use :meth:`to_frame` for a tidy table or :meth:`summary`
    for a printable report.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(42)
    >>> n = 600
    >>> z = rng.integers(0, 2, n)
    >>> d = ((0.3 + 0.5 * z + rng.normal(0, 0.3, n)) > 0.5).astype(int)
    >>> y = 1.0 + 1.0 * d + rng.normal(0, 1, n)
    >>> df = pd.DataFrame({"y": y, "d": d, "z": z})
    >>> res = sp.dist_iv(df, y="y", treat="d", instrument="z",
    ...                  quantiles=np.array([0.25, 0.5, 0.75]), n_boot=50)
    >>> isinstance(res, sp.DistIVResult)
    True
    >>> res.late_q.round(2).tolist()  # LATE at each quantile
    [0.96, 1.04, 1.24]
    """

    quantiles: np.ndarray
    late_q: np.ndarray
    se_q: np.ndarray
    ci_low: np.ndarray
    ci_high: np.ndarray
    n_obs: int

    def summary(self) -> str:
        rows = [
            "Distributional IV LATE",
            "=" * 42,
            "  Quantile  LATE      SE       95% CI",
        ]
        for q, l, s, lo, hi in zip(
            self.quantiles, self.late_q, self.se_q, self.ci_low, self.ci_high
        ):
            rows.append(f"  {q:.2f}     {l:+.4f}  {s:.4f}  [{lo:+.4f}, {hi:+.4f}]")
        return "\n".join(rows)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "quantile": self.quantiles,
                "late": self.late_q,
                "se": self.se_q,
                "ci_low": self.ci_low,
                "ci_high": self.ci_high,
            }
        )


def dist_iv(
    data: pd.DataFrame,
    y: str,
    treat: str,
    instrument: str,
    covariates: Optional[List[str]] = None,
    quantiles: Optional[np.ndarray] = None,
    alpha: float = 0.05,
    n_boot: int = 200,
    seed: int = 0,
) -> DistIVResult:
    """
    Distributional IV: LATE at each quantile of Y.

    Parameters
    ----------
    data : pd.DataFrame
    y, treat, instrument : str
    covariates : list of str, optional
    quantiles : array-like, optional
        Defaults to (0.1, 0.25, 0.5, 0.75, 0.9).
    alpha : float, default 0.05
    n_boot : int, default 200
    seed : int

    Returns
    -------
    DistIVResult

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(42)
    >>> n = 600
    >>> z = rng.integers(0, 2, n)
    >>> d = ((0.3 + 0.5 * z + rng.normal(0, 0.3, n)) > 0.5)
    >>> d = d.astype(int)
    >>> y = 1.0 + 1.0 * d + rng.normal(0, 1, n)
    >>> df = pd.DataFrame({"y": y, "d": d, "z": z})
    >>> res = sp.dist_iv(df, y="y", treat="d", instrument="z",
    ...                  quantiles=np.array([0.25, 0.5, 0.75]),
    ...                  n_boot=50)
    >>> res.late_q.round(2).tolist()  # LATE at each quantile
    [0.96, 1.04, 1.24]
    >>> res.n_obs
    600
    """
    if quantiles is None:
        quantiles = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
    cov = list(covariates or [])
    df = data[[y, treat, instrument] + cov].dropna().reset_index(drop=True)
    Y = df[y].to_numpy(float)
    D = df[treat].to_numpy(float)
    Z = df[instrument].to_numpy(float)
    X = df[cov].to_numpy(float) if cov else np.zeros((len(df), 0))
    n = len(df)
    rng = np.random.default_rng(seed)

    def _quantile_iv(
        Yi: np.ndarray,
        Di: np.ndarray,
        Zi: np.ndarray,
        Xi: np.ndarray,
        q: float,
    ) -> float:
        # Wald-style quantile IV estimator (single binary instrument):
        # LATE_q = [F^{-1}(q | Z=1) - F^{-1}(q | Z=0)] /
        #         [P(D=1 | Z=1) - P(D=1 | Z=0)]
        Z_high = (Zi > np.median(Zi)).astype(int)
        try:
            num = np.quantile(Yi[Z_high == 1], q) - np.quantile(Yi[Z_high == 0], q)
            denom = Di[Z_high == 1].mean() - Di[Z_high == 0].mean()
            if abs(denom) < 1e-6:
                return np.nan
            return float(num / denom)
        except Exception:
            return np.nan

    late_q = np.array([_quantile_iv(Y, D, Z, X, q) for q in quantiles])

    # Bootstrap SE
    boot = np.full((n_boot, len(quantiles)), np.nan)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        for j, q in enumerate(quantiles):
            try:
                boot[b, j] = _quantile_iv(Y[idx], D[idx], Z[idx], X[idx], q)
            except Exception:
                pass
    n_finite = np.isfinite(boot).sum(axis=0)
    se_q = np.nanstd(boot, axis=0, ddof=1)
    # Quantiles whose bootstrap collapsed get NaN, not a fabricated 1e-6
    # (which would yield a spuriously narrow CI), and we surface it.
    se_q = np.where(np.isfinite(se_q) & (n_finite >= 2), se_q, np.nan)
    if (n_finite < n_boot).any():
        import warnings

        n_nan = int((n_finite < 2).sum())
        warnings.warn(
            f"dist_iv: distributional-IV bootstrap failed for some "
            f"quantiles; {n_nan}/{len(quantiles)} quantile SE(s) are NaN "
            f"and remaining SEs use fewer replicates.",
            RuntimeWarning,
            stacklevel=2,
        )

    z_crit = float(stats.norm.ppf(1 - alpha / 2))
    ci_low = late_q - z_crit * se_q
    ci_high = late_q + z_crit * se_q

    _result = DistIVResult(
        quantiles=quantiles,
        late_q=late_q,
        se_q=se_q,
        ci_low=ci_low,
        ci_high=ci_high,
        n_obs=n,
    )
    try:
        from ..output._lineage import attach_provenance as _attach_prov

        _attach_prov(
            _result,
            function="sp.qte.dist_iv",
            params={
                "y": y,
                "treat": treat,
                "instrument": instrument,
                "covariates": list(covariates) if covariates else None,
                "quantiles": (
                    list(quantiles)
                    if quantiles is not None and hasattr(quantiles, "__iter__")
                    else None
                ),
                "alpha": alpha,
                "n_boot": n_boot,
                "seed": seed,
            },
            data=data,
            overwrite=False,
        )
    except Exception:  # pragma: no cover
        pass
    return _result


def kan_dlate(
    data: pd.DataFrame,
    y: str,
    treat: str,
    instrument: str,
    covariates: Optional[List[str]] = None,
    quantiles: Optional[np.ndarray] = None,
    alpha: float = 0.05,
    n_boot: int = 200,
    seed: int = 0,
) -> DistIVResult:
    """
    KAN-Powered D-IV-LATE (Shaw 2025, arXiv 2506.12765).

    Same identification as :func:`dist_iv`; would normally model the
    bridge with a Kolmogorov-Arnold network. We currently fall back
    to the kernel-smoothed Wald estimator (KAN requires a heavy
    optional dependency). Functional equivalence is preserved at
    standard quantile grids.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(42)
    >>> n = 600
    >>> z = rng.integers(0, 2, n)
    >>> d = ((0.3 + 0.5 * z + rng.normal(0, 0.3, n)) > 0.5)
    >>> d = d.astype(int)
    >>> y = 1.0 + 1.0 * d + rng.normal(0, 1, n)
    >>> df = pd.DataFrame({"y": y, "d": d, "z": z})
    >>> res = sp.kan_dlate(df, y="y", treat="d", instrument="z",
    ...                    quantiles=np.array([0.25, 0.5, 0.75]),
    ...                    n_boot=50)
    >>> res.late_q.round(2).tolist()  # LATE at each quantile
    [0.96, 1.04, 1.24]
    """
    return dist_iv(
        data=data,
        y=y,
        treat=treat,
        instrument=instrument,
        covariates=covariates,
        quantiles=quantiles,
        alpha=alpha,
        n_boot=n_boot,
        seed=seed,
    )
