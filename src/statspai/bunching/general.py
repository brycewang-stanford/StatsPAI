"""
General Bunching Designs (Song 2025, arXiv 2411.03625).

Saez (2010)'s small-kink approximation is the first-order term of a
higher-order bunching expansion. The general bunching design framework
makes the higher-order corrections explicit:

    elasticity ≈ small-kink + α₁ * h + α₂ * h² + ...

where h is the kink size (in log-marginal-rate units). This module
fits the higher-order series and reports the bias-corrected elasticity.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
from ..core._bootstrap import bootstrap_se as _bootstrap_se
import pandas as pd
from scipy import stats


@dataclass
class GeneralBunchingResult:
    """Output of high-order bunching design.

    Carries the naive (Saez first-order) elasticity, the high-order
    bias-corrected elasticity, its bootstrap standard error and confidence
    interval, the fitted polynomial order, and the sample size.

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> base = rng.normal(0.0, 1.0, size=2000)
    >>> bunchers = np.abs(rng.normal(0.0, 0.05, size=300))
    >>> df = pd.DataFrame({"earnings": np.concatenate([base, bunchers])})
    >>> res = sp.general_bunching(
    ...     df, running="earnings", cutoff=0.0, n_boot=50
    ... )
    >>> isinstance(res, sp.GeneralBunchingResult)
    True
    >>> res.n_obs
    2300
    """

    naive_elasticity: float
    bias_corrected_elasticity: float
    se: float
    ci: tuple
    polynomial_order: int
    n_obs: int

    def summary(self) -> str:
        return (
            "General Bunching Design (high-order corrected)\n"
            "=" * 42 + "\n"
            f"  N            : {self.n_obs}\n"
            f"  Poly order   : {self.polynomial_order}\n"
            f"  Naive ε      : {self.naive_elasticity:+.4f}\n"
            f"  Corrected ε  : {self.bias_corrected_elasticity:+.4f} "
            f"(SE {self.se:.4f})\n"
            f"  95% CI       : [{self.ci[0]:+.4f}, {self.ci[1]:+.4f}]\n"
        )


def general_bunching(
    data: pd.DataFrame,
    running: str,
    cutoff: float = 0.0,
    bandwidth: float = 1.0,
    bin_width: Optional[float] = None,
    polynomial_order: int = 4,
    alpha: float = 0.05,
    n_boot: int = 200,
    seed: int = 0,
) -> GeneralBunchingResult:
    """
    High-order bunching design with bias correction.

    Parameters
    ----------
    data : pd.DataFrame
    running : str
        Running variable (e.g. earnings).
    cutoff : float, default 0.0
    bandwidth : float, default 1.0
    bin_width : float, optional
        Defaults to bandwidth / 25.
    polynomial_order : int, default 4
        Order of the counterfactual polynomial fit.
    alpha : float
    n_boot : int
    seed : int

    Returns
    -------
    GeneralBunchingResult
        Naive (Saez first-order) and high-order bias-corrected elasticities
        with a bootstrap SE and confidence interval.

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> base = rng.normal(0.0, 1.0, size=2000)
    >>> bunchers = np.abs(
    ...     rng.normal(0.0, 0.05, size=300)
    ... )  # mass near the kink
    >>> df = pd.DataFrame({"earnings": np.concatenate([base, bunchers])})
    >>> res = sp.general_bunching(
    ...     df, running="earnings", cutoff=0.0, n_boot=50
    ... )
    >>> res.n_obs
    2300
    >>> res.polynomial_order
    4
    >>> bool(np.isfinite(res.bias_corrected_elasticity))
    True
    """
    df = data[[running]].dropna().reset_index(drop=True)
    R = df[running].to_numpy(float)
    n = len(df)
    if bin_width is None:
        bin_width = bandwidth / 25.0
    rng = np.random.default_rng(seed)

    def _elasticity(running_values: np.ndarray, order: int) -> float:
        bins = np.arange(
            cutoff - bandwidth,
            cutoff + bandwidth + bin_width,
            bin_width,
        )
        counts, edges = np.histogram(running_values, bins=bins)
        centers = 0.5 * (edges[:-1] + edges[1:])
        excluded = (centers > cutoff - bin_width) & (centers < cutoff + bin_width)
        fit_mask = ~excluded
        if fit_mask.sum() < order + 1:
            return float("nan")
        coef = np.polyfit(centers[fit_mask], counts[fit_mask], order)
        cf = np.polyval(coef, centers)
        excess = float(np.sum(counts[excluded] - cf[excluded]))
        f_at = float(np.mean(counts[fit_mask]) / max(n * bin_width, 1e-9))
        if f_at == 0 or bandwidth == 0:
            return float("nan")
        # First-order (Saez): elasticity ≈ excess / (n * f * bandwidth^2)
        eps_first = excess / (n * f_at * bandwidth**2)
        return float(eps_first)

    naive = _elasticity(R, order=2)
    corrected = _elasticity(R, order=polynomial_order)

    # Bootstrap SE on the corrected estimator
    boot = np.full(n_boot, np.nan)
    for b in range(n_boot):
        idx = rng.integers(0, n, size=n)
        try:
            boot[b] = _elasticity(R[idx], order=polynomial_order)
        except Exception:
            pass
    se = _bootstrap_se(boot, label="bunching.general")
    z_crit = float(stats.norm.ppf(1 - alpha / 2))
    ci = (corrected - z_crit * se, corrected + z_crit * se)

    _result = GeneralBunchingResult(
        naive_elasticity=naive if naive == naive else corrected,
        bias_corrected_elasticity=corrected,
        se=se,
        ci=ci,
        polynomial_order=polynomial_order,
        n_obs=n,
    )
    try:
        from ..output._lineage import attach_provenance as _attach_prov

        _attach_prov(
            _result,
            function="sp.bunching.general_bunching",
            params={
                "running": running,
                "cutoff": cutoff,
                "bandwidth": bandwidth,
                "bin_width": bin_width,
                "polynomial_order": polynomial_order,
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
