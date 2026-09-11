"""
Multi-cutoff and multi-score RD designs.

Implements RD designs with multiple cutoffs (rdmc) and
multiple running variables / geographic RD (rdms).

Equivalent to Stata/R's ``rdmulti`` package (Cattaneo, Keele, Titiunik &
Vazquez-Bare 2016, 2021).

References
----------
Cattaneo, M.D., Keele, L., Titiunik, R. & Vazquez-Bare, G. (2016).
"Interpreting Regression Discontinuity Designs with Multiple Cutoffs."
*Journal of Politics*, 78(4), 1229-1248. doi:10.1086/686802
[@cattaneo2016interpreting]

Cattaneo, M.D., Keele, L., Titiunik, R. & Vazquez-Bare, G. (2021).
"Extrapolating Treatment Effects in Multi-Cutoff Regression Discontinuity
Designs." *Journal of the American Statistical Association*, 116(536),
1941-1952. doi:10.1080/01621459.2020.1751646
[@cattaneo2021extrapolating]

Keele, L. & Titiunik, R. (2015).
"Geographic Boundaries as Regression Discontinuities."
*Political Analysis*, 23(1), 127-155. doi:10.1093/pan/mpu014
[@keele2015geographic]
"""

import warnings
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from .._result_serialize import ResultProtocolMixin
from ..core.results import CausalResult
from ._core import _kernel_fn


class RDMultiResult(ResultProtocolMixin):
    """Results from multi-cutoff/multi-score RD.

    Returned by :func:`rdmc`. Holds per-cutoff estimates in
    ``cutoff_results`` plus the pooled estimate / SE / CI, and exposes
    :meth:`summary` and a forest :meth:`plot`.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> n = 900
    >>> running_var = rng.uniform(0, 100, n)
    >>> cutoffs = [30, 60]
    >>> nearest = np.array([min(cutoffs, key=lambda c: abs(rv - c))
    ...                     for rv in running_var])
    >>> treated = (running_var >= nearest).astype(float)
    >>> score = 1.0 + 0.02 * running_var + 2.0 * treated + rng.normal(0, 1, n)
    >>> df = pd.DataFrame({"score": score, "running_var": running_var})
    >>> result = sp.rdmc(df, y="score", x="running_var", cutoffs=[30, 60])
    >>> isinstance(result, sp.RDMultiResult)
    True
    >>> result.n_cutoffs
    2
    """

    def __init__(
        self,
        cutoff_results: List[Dict[str, Any]],
        pooled_estimate: float,
        pooled_se: float,
        pooled_ci: Tuple[float, float],
        n_cutoffs: int,
        n_total: int,
        method: str,
    ) -> None:
        self.cutoff_results = cutoff_results  # list of dicts
        self.pooled_estimate = pooled_estimate
        self.pooled_se = pooled_se
        self.pooled_ci = pooled_ci
        self.n_cutoffs = n_cutoffs
        self.n_total = n_total
        self.method = method

    def summary(self) -> str:
        lines = [
            f"Multi-Cutoff Regression Discontinuity ({self.method})",
            "=" * 65,
            f"Number of cutoffs: {self.n_cutoffs}",
            f"Total observations: {self.n_total}",
            "",
            f"{'Cutoff':<10s} {'N':>6s} {'Estimate':>10s} {'SE':>10s} "
            f"{'95% CI':>22s} {'p-value':>10s}",
            "-" * 65,
        ]
        for cr in self.cutoff_results:
            ci = f"[{cr['ci_lower']:.4f}, {cr['ci_upper']:.4f}]"
            lines.append(
                f"{cr['cutoff']:<10.2f} {cr['n']:>6d} {cr['estimate']:>10.4f} "
                f"{cr['se']:>10.4f} {ci:>22s} {cr['p_value']:>10.4f}"
            )

        lines.append("-" * 65)
        ci_pooled = f"[{self.pooled_ci[0]:.4f}, {self.pooled_ci[1]:.4f}]"
        lines.append(
            f"{'Pooled':<10s} {self.n_total:>6d} {self.pooled_estimate:>10.4f} "
            f"{self.pooled_se:>10.4f} {ci_pooled:>22s}"
        )
        lines.append("=" * 65)
        return "\n".join(lines)

    def plot(self, ax: Optional[Any] = None, **kwargs: Any) -> Any:
        """Forest plot of cutoff-specific and pooled estimates."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:  # pragma: no cover
            raise ImportError("matplotlib required for plotting")  # pragma: no cover

        if ax is None:
            fig, ax = plt.subplots(
                figsize=(8, max(4, len(self.cutoff_results) * 0.5 + 2))
            )

        labels = [f"c = {cr['cutoff']:.1f}" for cr in self.cutoff_results] + ["Pooled"]
        estimates = [cr["estimate"] for cr in self.cutoff_results] + [
            self.pooled_estimate
        ]
        ci_lowers = [cr["ci_lower"] for cr in self.cutoff_results] + [self.pooled_ci[0]]
        ci_uppers = [cr["ci_upper"] for cr in self.cutoff_results] + [self.pooled_ci[1]]

        y_pos = range(len(labels))
        errors = [
            [e - cl for e, cl in zip(estimates, ci_lowers)],
            [cu - e for e, cu in zip(estimates, ci_uppers)],
        ]

        ax.errorbar(
            estimates,
            y_pos,
            xerr=errors,
            fmt="o",
            color="steelblue",
            capsize=3,
            elinewidth=1.5,
        )
        ax.scatter(
            [self.pooled_estimate],
            [len(labels) - 1],
            color="red",
            s=80,
            zorder=5,
            marker="D",
        )
        ax.axvline(0, color="gray", ls="--", lw=0.5)
        ax.set_yticks(list(y_pos))
        ax.set_yticklabels(labels)
        ax.set_xlabel("Treatment Effect")
        ax.set_title("Multi-Cutoff RD Estimates")
        plt.tight_layout()
        return ax


def _local_linear_rd(
    y: np.ndarray,
    x: np.ndarray,
    c: float,
    h: float,
    kernel: str = "triangular",
) -> Tuple[float, float, int]:
    """Local linear RD estimate at cutoff c with bandwidth h."""
    x_centered = x - c

    # Kernel weights (canonical definition in ._core)
    w = _kernel_fn(x_centered / h, kernel)

    mask = w > 0
    if mask.sum() < 4:
        return np.nan, np.nan, 0  # pragma: no cover

    y_m, x_m, w_m = y[mask], x_centered[mask], w[mask]
    D_m = (x_m >= 0).astype(float)

    # Local linear: y = a + b*x + tau*D + delta*D*x
    X = np.column_stack([np.ones(len(x_m)), x_m, D_m, D_m * x_m])
    W = np.diag(w_m)

    try:
        XtWX = X.T @ W @ X
        XtWy = X.T @ W @ y_m
        beta = np.linalg.solve(XtWX, XtWy)
        resid = y_m - X @ beta
        sigma2 = np.sum(w_m * resid**2) / max(mask.sum() - 4, 1)
        var_cov = sigma2 * np.linalg.inv(XtWX)
        tau = beta[2]
        se = np.sqrt(var_cov[2, 2])
    except np.linalg.LinAlgError:  # pragma: no cover
        tau, se = np.nan, np.nan  # pragma: no cover

    return float(tau), float(se), int(mask.sum())


def rdmc(
    data: pd.DataFrame,
    y: str,
    x: str,
    cutoffs: Optional[List[float]] = None,
    bandwidth: Optional[float] = None,
    kernel: str = "triangular",
    pooling: str = "ivw",
    alpha: float = 0.05,
    cutoff_var: Optional[str] = None,
    **rdrobust_kwargs: object,
) -> RDMultiResult:
    """
    Multi-cutoff RD design.

    Estimates treatment effects at multiple cutoffs and pools them.

    Two designs, and they are not the same estimator:

    * **Unit-specific cutoffs** (``cutoff_var=``): each unit faces its own
      cutoff, given by a column. This is the design of Cattaneo, Titiunik,
      Vazquez-Bare & Keele (2016) and what ``rdmulti::rdmc()`` implements --
      its ``C`` argument. Each cutoff is estimated on **only the units
      assigned to it**, by delegating to :func:`statspai.rdrobust`, so it
      inherits the CCT bandwidth cascade and robust bias correction.
    * **Shared running variable** (``cutoffs=`` alone, the default): one
      running variable crossed by several thresholds, every unit entering
      every threshold's local regression under a common rule-of-thumb
      bandwidth.

    Before 1.21 only the second was available, while the docstring claimed
    equivalence to ``rdmulti::rdmc()``. On a design with unit-specific
    cutoffs and effects of 2.0 / 5.0 / -3.0, the shared-running-variable
    path returned 0.22 / 0.51 / 0.66 -- because units belonging to one
    cutoff were pooled into every other cutoff's window, averaging the
    effects away. Pass ``cutoff_var=`` for the R-equivalent estimator.

    Parameters
    ----------
    data : pd.DataFrame
    y : str
        Outcome variable.
    x : str
        Running variable.
    cutoffs : list of float
        Cutoff values.
    bandwidth : float, optional
        Bandwidth for local polynomial. If None, uses Silverman rule.
    kernel : str, default 'triangular'
    pooling : str, default 'ivw'
        Pooling method: 'ivw' (inverse-variance weighted) or 'equal'.
    alpha : float, default 0.05

    Returns
    -------
    RDMultiResult

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> n = 900
    >>> running_var = rng.uniform(0, 100, n)
    >>> cutoffs = [30, 60]
    >>> nearest = np.array([min(cutoffs, key=lambda c: abs(rv - c))
    ...                     for rv in running_var])
    >>> treated = (running_var >= nearest).astype(float)
    >>> score = 1.0 + 0.02 * running_var + 2.0 * treated + rng.normal(0, 1, n)
    >>> df = pd.DataFrame({"score": score, "running_var": running_var})
    >>> result = sp.rdmc(df, y="score", x="running_var", cutoffs=[30, 60])
    >>> result.n_cutoffs
    2
    >>> summary_text = result.summary()
    >>> ax = result.plot()  # doctest: +SKIP
    """
    if cutoff_var is not None:
        return _rdmc_unit_cutoffs(
            data,
            y=y,
            x=x,
            cutoff_var=cutoff_var,
            cutoffs=cutoffs,
            alpha=alpha,
            kernel=kernel,
            bandwidth=bandwidth,
            **rdrobust_kwargs,
        )
    if cutoffs is None:
        raise ValueError(
            "rdmc needs either cutoffs= (shared running variable) or "
            "cutoff_var= (unit-specific cutoffs, the rdmulti::rdmc design)"
        )

    y_data = data[y].values.astype(float)
    x_data = data[x].values.astype(float)

    if bandwidth is None:
        bandwidth_value = float(1.06 * np.std(x_data) * len(x_data) ** (-1 / 5))
    else:
        bandwidth_value = float(bandwidth)

    cutoff_results = []
    z_crit = stats.norm.ppf(1 - alpha / 2)

    for c in cutoffs:
        tau, se, n_local = _local_linear_rd(y_data, x_data, c, bandwidth_value, kernel)
        p_val = 2 * stats.norm.sf(abs(tau / se)) if se > 0 else np.nan

        cutoff_results.append(
            {
                "cutoff": c,
                "estimate": tau,
                "se": se,
                "ci_lower": tau - z_crit * se,
                "ci_upper": tau + z_crit * se,
                "p_value": p_val,
                "n": n_local,
                "bandwidth": bandwidth_value,
            }
        )

    # Pool estimates
    valid = [cr for cr in cutoff_results if np.isfinite(cr["se"]) and cr["se"] > 0]
    if len(valid) > 0:
        if pooling == "ivw":
            weights = np.array([1 / cr["se"] ** 2 for cr in valid])
            weights /= weights.sum()
            pooled = sum(w * cr["estimate"] for w, cr in zip(weights, valid))
            pooled_se = np.sqrt(1 / sum(1 / cr["se"] ** 2 for cr in valid))
        else:
            pooled = np.mean([cr["estimate"] for cr in valid])
            pooled_se = np.sqrt(np.mean([cr["se"] ** 2 for cr in valid]) / len(valid))
    else:
        pooled, pooled_se = np.nan, np.nan  # pragma: no cover

    pooled_ci = (pooled - z_crit * pooled_se, pooled + z_crit * pooled_se)

    return RDMultiResult(
        cutoff_results=cutoff_results,
        pooled_estimate=pooled,
        pooled_se=pooled_se,
        pooled_ci=pooled_ci,
        n_cutoffs=len(cutoffs),
        n_total=len(y_data),
        method="Multi-Cutoff RD (rdmc)",
    )


def _rdmc_unit_cutoffs(
    data: pd.DataFrame,
    y: str,
    x: str,
    cutoff_var: str,
    cutoffs: Optional[List[float]],
    alpha: float,
    kernel: str,
    bandwidth: Optional[float],
    **rdrobust_kwargs: object,
) -> RDMultiResult:
    """``rdmulti::rdmc``'s estimator: one rdrobust fit per assigned cutoff.

    Each unit belongs to exactly one cutoff, so cutoff ``c``'s effect is
    estimated on ``data[cutoff_var] == c`` alone. Pooling follows rdmulti:
    the weights are the effective sample sizes ``Nh_c / sum(Nh)``, and the
    pooled variance is ``sum(w_c^2 * V_c)``.
    """
    from .rdrobust import rdrobust as _rdrobust

    cvals = data[cutoff_var].to_numpy()
    if cutoffs is None:
        cutoffs = sorted(pd.unique(cvals).tolist())
    z_crit = stats.norm.ppf(1 - alpha / 2)

    cutoff_results = []
    for c in cutoffs:
        sub = data.loc[cvals == c]
        if len(sub) < 10:
            raise ValueError(
                f"cutoff {c} has only {len(sub)} units assigned by "
                f"{cutoff_var!r}; too few to estimate an RD effect"
            )
        kw = dict(rdrobust_kwargs)
        if bandwidth is not None:
            kw.setdefault("h", float(bandwidth))
        fit = _rdrobust(sub, y=y, x=x, c=float(c), kernel=kernel, **kw)
        tau = float(fit.detail["estimate"][0])
        se = float(fit.detail["se"][0])
        mi = fit.model_info
        h = float(np.ravel(mi["bandwidth_h"])[0])
        xs = sub[x].to_numpy(dtype=float) - float(c)
        n_h = int(np.sum(np.abs(xs) <= h))
        cutoff_results.append(
            {
                "cutoff": float(c),
                "estimate": tau,
                "se": se,
                "ci_lower": tau - z_crit * se,
                "ci_upper": tau + z_crit * se,
                "p_value": float(fit.detail["pvalue"][0]),
                "n": n_h,
                "bandwidth": h,
                "estimate_robust": float(fit.detail["estimate"][1]),
                "se_robust": float(fit.detail["se"][1]),
            }
        )

    # rdmulti weights by effective sample size, not inverse variance.
    nh = np.array([cr["n"] for cr in cutoff_results], dtype=float)
    w = nh / nh.sum()
    pooled = float(np.sum(w * np.array([cr["estimate"] for cr in cutoff_results])))
    pooled_se = float(
        np.sqrt(np.sum(w**2 * np.array([cr["se"] ** 2 for cr in cutoff_results])))
    )
    for cr, wi in zip(cutoff_results, w):
        cr["weight"] = float(wi)

    return RDMultiResult(
        cutoff_results=cutoff_results,
        pooled_estimate=pooled,
        pooled_se=pooled_se,
        pooled_ci=(pooled - z_crit * pooled_se, pooled + z_crit * pooled_se),
        n_cutoffs=len(cutoffs),
        n_total=len(data),
        method="Multi-Cutoff RD (rdmc, unit-specific cutoffs)",
    )


def rdms(
    data: pd.DataFrame,
    y: str,
    x1: str,
    x2: str,
    cutoff1: float = 0,
    cutoff2: float = 0,
    treat: Optional[str] = None,
    bandwidth: Optional[float] = None,
    kernel: str = "triangular",
    alpha: float = 0.05,
    **rdrobust_kwargs: object,
) -> CausalResult:
    """
    Multi-score / Geographic RD design at a single boundary point.

    Collapses the two-score design onto the score the reference uses --
    Euclidean distance to the boundary point, signed by treatment status,

        ``d = sqrt((x1 - cutoff1)^2 + (x2 - cutoff2)^2) * (2 * treat - 1)``

    -- and estimates a sharp RD at ``d = 0`` by delegating to
    :func:`statspai.rdrobust`. That is what ``rdmulti::rdms()`` does, so
    the CCT bandwidth cascade, the robust bias correction and the
    heteroskedasticity-robust variance all come with it.

    Following the reference, the reported estimate is the **bias-corrected**
    point estimate with the **robust** interval (``Estimate[2]`` with
    ``ci[3, ]`` in R's return), not the conventional pair; the conventional
    numbers are in ``model_info["conventional"]``.

    .. versionchanged:: 1.27.0
       ⚠️ **Numerical output changed. Re-run anything that used this
       function.** It previously fitted a 2D local linear in ``(x1, x2)``
       inside a Euclidean window whose width came from Silverman's
       *kernel density* rule of thumb, with a homoskedastic variance and
       no bias correction. On a 6,000-row design with known effects of
       1.5 / 2.0 / 2.5 at three boundary points it used 13 / 8 / 27
       observations where ``rdmulti::rdms`` used 519 / 725 / 743, and
       returned 1.322 / **4.804** / 2.326 against the reference's
       1.396 / 1.778 / 2.463 -- the middle point 170% out, with a standard
       error of 4.21 against 0.13. See MIGRATION.md.

    .. versionadded:: 1.27.0
       ``treat=``, the treatment indicator (``zvar`` in R). Treatment on a
       two-dimensional boundary is not implied by the coordinates, which is
       why the reference requires it. Omitting it falls back to
       ``x1 >= cutoff1`` **and warns**, because that is an assumption about
       the design rather than a fact about the data.

    Parameters
    ----------
    data : pd.DataFrame
    y : str
        Outcome variable.
    x1 : str
        First running variable (e.g., latitude distance to boundary).
    x2 : str
        Second running variable (e.g., longitude distance to boundary).
    cutoff1 : float, default 0
        Cutoff for x1.
    cutoff2 : float, default 0
        Cutoff for x2.
    treat : str, optional
        Column holding the 0/1 treatment indicator (R's ``zvar``). Strongly
        recommended: without it treatment is assumed to be
        ``x1 >= cutoff1`` and a warning is issued.
    bandwidth : float, optional
        Fixed bandwidth on the signed-distance score. When omitted the CCT
        MSE-optimal cascade selects it, as the reference does.
    kernel : str, default 'triangular'
    alpha : float, default 0.05
    **rdrobust_kwargs
        Forwarded to :func:`statspai.rdrobust` (``p``, ``bwselect``,
        ``vce``, ``cluster``, ``covs``, ...).

    Returns
    -------
    CausalResult
        ``estimate`` / ``se`` / ``ci`` are the bias-corrected point estimate
        with robust inference. ``model_info`` carries the selected
        bandwidths, effective sample sizes either side, the conventional
        estimates, and ``treat_assumed`` recording whether the treatment
        convention was inferred.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(1)
    >>> n = 800
    >>> dist_lat = rng.uniform(-5, 5, n)
    >>> dist_lon = rng.uniform(-5, 5, n)
    >>> treated = (dist_lat >= 0).astype(float)
    >>> outcome = (1.0 + 1.5 * treated + 0.1 * dist_lat
    ...            + 0.1 * dist_lon + rng.normal(0, 1, n))
    >>> df = pd.DataFrame({"outcome": outcome,
    ...                    "dist_lat": dist_lat, "dist_lon": dist_lon})
    >>> result = sp.rdms(df, y="outcome", x1="dist_lat", x2="dist_lon",
    ...                  bandwidth=3.0)
    >>> summary_text = result.summary()
    """
    if treat is None:
        treat_values = (data[x1].to_numpy(dtype=float) >= cutoff1).astype(float)
        warnings.warn(
            "rdms: no treat= column given, so treatment is being taken as "
            f"({x1} >= {cutoff1}). That is an assumption about the design, "
            "not a property of the data -- a geographic boundary is rarely a "
            "threshold in one coordinate. rdmulti::rdms requires the "
            "indicator (its `zvar`) for exactly this reason. Pass treat= to "
            "state it.",
            UserWarning,
            stacklevel=2,
        )
    else:
        treat_values = data[treat].to_numpy(dtype=float)
        bad = ~np.isin(treat_values[np.isfinite(treat_values)], (0.0, 1.0))
        if bad.any():
            raise ValueError(
                f"rdms: treat= column {treat!r} must be 0/1; found "
                f"{np.unique(treat_values[np.isfinite(treat_values)])[:5]}"
            )

    y_data = data[y].to_numpy(dtype=float)
    x1_data = data[x1].to_numpy(dtype=float)
    x2_data = data[x2].to_numpy(dtype=float)

    # The reference's own construction:
    #     xc = sqrt((X - C)^2 + (X2 - C2)^2) * (2 * zvar - 1)
    # i.e. Euclidean distance to the boundary point, signed by treatment
    # status. That collapses the two-score design onto a single score, and
    # the effect at the boundary point is then an ordinary sharp RD at zero
    # on it -- which is why delegating to rdrobust is not a shortcut but the
    # thing rdmulti::rdms does.
    signed_distance = np.sqrt((x1_data - cutoff1) ** 2 + (x2_data - cutoff2) ** 2) * (
        2.0 * treat_values - 1.0
    )

    frame = pd.DataFrame({"__y": y_data, "__d": signed_distance})
    ok = np.isfinite(frame["__y"]) & np.isfinite(frame["__d"])
    n_dropped = int((~ok).sum())
    if n_dropped:
        warnings.warn(
            f"rdms: dropped {n_dropped} row(s) with a missing outcome, "
            f"coordinate or treatment indicator.",
            UserWarning,
            stacklevel=2,
        )
    frame = frame.loc[ok]

    from .rdrobust import rdrobust as _rdrobust

    fit = _rdrobust(
        frame,
        y="__y",
        x="__d",
        c=0.0,
        kernel=kernel,
        alpha=alpha,
        **({"h": float(bandwidth), "b": float(bandwidth)} if bandwidth else {}),
        **rdrobust_kwargs,
    )

    mi = fit.model_info
    # rdmulti::rdms reports the bias-corrected point estimate with the
    # robust interval (`Estimate[2]` with `ci[3, ]`), not the conventional
    # pair, so the same convention is carried here.
    robust = mi["robust"]
    return CausalResult(
        method="Multi-score RD (rdms)",
        estimand="ATE at boundary point",
        estimate=float(robust["estimate"]),
        se=float(robust["se"]),
        pvalue=float(robust.get("pvalue", np.nan)),
        ci=(float(robust["ci"][0]), float(robust["ci"][1])),
        alpha=alpha,
        n_obs=int(len(frame)),
        model_info={
            "cutoff1": cutoff1,
            "cutoff2": cutoff2,
            "kernel": mi["kernel"],
            "bandwidth_h": mi["bandwidth_h"],
            "bandwidth_b": mi["bandwidth_b"],
            "bwselect": mi["bwselect"],
            "n_effective_left": mi["n_effective_left"],
            "n_effective_right": mi["n_effective_right"],
            "conventional": mi["conventional"],
            "treat_assumed": treat is None,
            "score": "signed Euclidean distance to (cutoff1, cutoff2)",
        },
    )
