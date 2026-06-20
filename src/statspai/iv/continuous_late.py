"""
Continuous-Instrument LATE (Zeng et al. 2025, arXiv 2504.03063).

Generalises LATE identification to continuous instruments. The
"maximal complier class" is the set of units whose treatment status
shifts most as Z varies; the LATE on this class is the limit of the
Wald ratio across instrument quantiles.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from ..core._bootstrap import bootstrap_se as _bootstrap_se
import pandas as pd
from scipy import stats


@dataclass
class ContinuousLATEResult:
    """Continuous-instrument LATE on the maximal complier class.

    Returned by :func:`sp.continuous_iv_late`. Holds the LATE estimate,
    bootstrap SE, CI, and the complier share of the maximal complier
    class. Call ``.summary()`` for a formatted report.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(42)
    >>> n = 500
    >>> z = rng.normal(size=n)
    >>> v = rng.normal(size=n)
    >>> d = (0.9 * z + v > 0).astype(float)
    >>> y = 1.2 * d + 0.5 * v + rng.normal(size=n)
    >>> df = pd.DataFrame({'y': y, 'd': d, 'z': z})
    >>> res = sp.continuous_iv_late(df, y='y', treat='d', instrument='z')
    >>> isinstance(res, sp.ContinuousLATEResult)
    True
    >>> round(res.estimate, 2)
    1.81
    """

    estimate: float
    se: float
    ci: tuple
    complier_share: float
    n_obs: int

    def summary(self) -> str:
        header = "Continuous-Instrument LATE (Maximal Complier Class)"
        bar = "=" * len(header)
        return (
            f"{header}\n"
            f"{bar}\n"
            f"  N            : {self.n_obs}\n"
            f"  Complier sh. : {self.complier_share:.3f}\n"
            f"  LATE         : {self.estimate:+.4f} (SE {self.se:.4f})\n"
            f"  95% CI       : [{self.ci[0]:+.4f}, {self.ci[1]:+.4f}]\n"
        )


def continuous_iv_late(
    data: pd.DataFrame,
    y: str,
    treat: str,
    instrument: str,
    n_quantiles: int = 4,
    alpha: float = 0.05,
    n_boot: int = 200,
    seed: int = 0,
) -> ContinuousLATEResult:
    """
    LATE with a continuous instrument via quantile-bin Wald estimator.

    Parameters
    ----------
    data : pd.DataFrame
    y, treat, instrument : str
    n_quantiles : int, default 4
        Number of quantile bins of the instrument; LATE is averaged
        across bins weighted by complier share.
    alpha : float
    n_boot : int
    seed : int

    Returns
    -------
    ContinuousLATEResult

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(42)
    >>> n = 500
    >>> z = rng.normal(size=n)
    >>> v = rng.normal(size=n)
    >>> d = (0.9 * z + v > 0).astype(float)
    >>> y = 1.2 * d + 0.5 * v + rng.normal(size=n)
    >>> df = pd.DataFrame({'y': y, 'd': d, 'z': z})
    >>> res = sp.continuous_iv_late(df, y='y', treat='d',
    ...                             instrument='z')
    >>> round(res.estimate, 2)
    1.81
    >>> round(res.complier_share, 2)  # maximal complier class
    0.35
    """
    df = data[[y, treat, instrument]].dropna().reset_index(drop=True)
    Y = df[y].to_numpy(float)
    D = df[treat].to_numpy(float)
    Z = df[instrument].to_numpy(float)
    n = len(df)
    rng = np.random.default_rng(seed)

    def _wald_per_bin(
        Yi: np.ndarray,
        Di: np.ndarray,
        Zi: np.ndarray,
    ) -> tuple[float, float]:
        z_bins = pd.qcut(Zi, q=n_quantiles, labels=False, duplicates="drop")
        unique_bins = np.sort(np.unique(z_bins))
        if len(unique_bins) < 2:
            return (
                float(np.mean(Yi[Di > Di.mean()]) - np.mean(Yi[Di <= Di.mean()])),
                1.0,
            )
        # Wald ratio for each adjacent bin, then average
        atts: list[float] = []
        weights: list[float] = []
        for k in range(len(unique_bins) - 1):
            mask = z_bins == unique_bins[k]
            mask_next = z_bins == unique_bins[k + 1]
            num = float(Yi[mask_next].mean() - Yi[mask].mean())
            denom = float(Di[mask_next].mean() - Di[mask].mean())
            if abs(denom) < 1e-6:
                continue
            atts.append(num / denom)
            weights.append(abs(denom))
        if not atts:
            return float("nan"), 0.0
        atts_arr = np.asarray(atts, dtype=float)
        weights_arr = np.asarray(weights, dtype=float)
        # Maximal complier class: pick the bin pair with the biggest
        # |denom| (most "responsive" units to Z).
        idx = int(np.argmax(weights_arr))
        return float(atts_arr[idx]), float(weights_arr[idx])

    estimate, complier = _wald_per_bin(Y, D, Z)
    boot = np.full(n_boot, np.nan)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        try:
            boot[b], _ = _wald_per_bin(Y[idx], D[idx], Z[idx])
        except Exception:  # pragma: no cover
            pass
    se = _bootstrap_se(boot, label="iv.continuous_late")
    z_crit = float(stats.norm.ppf(1 - alpha / 2))
    ci = (estimate - z_crit * se, estimate + z_crit * se)
    _result = ContinuousLATEResult(
        estimate=estimate,
        se=se,
        ci=ci,
        complier_share=complier,
        n_obs=n,
    )
    try:
        from ..output._lineage import attach_provenance as _attach_prov

        _attach_prov(
            _result,
            function="sp.iv.continuous_iv_late",
            params={
                "y": y,
                "treat": treat,
                "instrument": instrument,
                "n_quantiles": n_quantiles,
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
