"""Roth & Sant'Anna functional-form test for parallel trends.

Parallel trends in levels and parallel trends in logs are different
assumptions, and a DiD design can satisfy one while violating the other.
Roth & Sant'Anna ask when that ambiguity goes away: when is parallel trends
insensitive to *every* strictly monotonic transformation of the outcome?

Their answer is a testable restriction. Parallel trends holds for all such
transformations if and only if the counterfactual untreated distribution the
design implies for the treated group is a genuine distribution -- in
particular, if and only if its density is non-negative everywhere. A design
whose implied density goes negative somewhere is one where the choice of
functional form is doing identifying work.

How the test is built
---------------------
Bin the outcome. For bin ``b``, define

    Y_b(i, t) = -1{Y(i, t) in b}   for untreated (i, t)
              =  0                 for treated   (i, t)

and run the Callaway-Sant'Anna estimator on ``Y_b``. Under the sign flip and
the zeroing of treated cells, the aggregated ATT of ``Y_b`` *is* the implied
counterfactual probability mass the design puts on bin ``b`` for the treated
group. Stack those across bins and the null

    H0:  implied_density[b] >= 0  for every b

is a one-sided moment-inequality problem. It is tested with a least-favourable
max-t statistic whose critical value comes from simulating the limiting normal
under the estimated correlation matrix -- the influence functions of the
per-bin aggregates supply that correlation.

A large p-value means the data give no evidence against functional-form
insensitivity. A small one means the implied density is negative somewhere by
more than sampling error can explain, so "parallel trends" is a claim about a
particular scale, and reporting the estimate in levels and in logs will not
give the same answer.

Reference implementation
------------------------
Pinned against ``didFF`` 0.1.0 (Sant'Anna's own R package) on ``did::mpdta``:
see ``tests/reference_parity/test_functional_form_parity.py`` and Track A
module ``79_didff``.

References
----------
roth2023when
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import pandas as pd

from .._result_serialize import ResultProtocolMixin
from ..exceptions import DataInsufficient, MethodIncompatibility

__all__ = ["FunctionalFormResult", "functional_form_test"]

_AGGTE_TYPES = ("simple", "group", "dynamic", "calendar")


@dataclass
class FunctionalFormResult(ResultProtocolMixin):
    """Result of :func:`functional_form_test`.

    Attributes
    ----------
    pvalue : float
        P-value for ``H0: the implied counterfactual density is non-negative
        everywhere``. Large = no evidence against functional-form
        insensitivity.
    table : pandas.DataFrame
        One row per outcome bin: ``bin_lower``, ``bin_upper``,
        ``implied_density`` and ``se``.
    statistic : float
        The max-t statistic ``max_b (-density_b / se_b)``.
    n_bins : int
        Number of bins actually used (bins with a degenerate influence
        function are dropped, as in the reference implementation).
    n_units : int
        Units contributing to the influence functions.
    aggregation : str
        Which ``aggte`` aggregation produced the per-bin estimates.
    n_sims : int
        Simulation draws behind the critical value.
    alpha : float
    diagnostics : dict
    """

    pvalue: float
    table: pd.DataFrame
    statistic: float
    n_bins: int
    n_units: int
    aggregation: str
    n_sims: int
    alpha: float = 0.05
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    method: str = "Functional-form test for parallel trends (Roth & Sant'Anna)"

    @property
    def negative_bins(self) -> pd.DataFrame:
        """The bins whose implied density came out negative."""
        return self.table[self.table["implied_density"] < 0]

    def summary(self) -> str:
        lines = [
            self.method,
            "=" * len(self.method),
            "H0: implied counterfactual density >= 0 on every bin",
            f"  bins used        : {self.n_bins}",
            f"  aggregation      : {self.aggregation}",
            f"  max-t statistic  : {self.statistic:.6f}",
            f"  p-value          : {self.pvalue:.6f}  ({self.n_sims} sims)",
            "",
        ]
        neg = self.negative_bins
        if self.pvalue < self.alpha:
            lines.append(
                f"REJECTED at {self.alpha:.2f}: the implied density is negative on "
                f"{len(neg)} bin(s) by more than sampling error explains. Parallel "
                "trends cannot hold for every monotonic transformation of the "
                "outcome, so the levels-vs-logs choice is doing identifying work "
                "and should be argued for, not assumed."
            )
        else:
            lines.append(
                "NOT rejected: no evidence against parallel trends holding for "
                "every monotonic transformation. Note this is a failure to "
                "reject, not a positive finding -- see the power of the test "
                "before reading it as reassurance."
            )
        if len(neg):
            lines.append("")
            lines.append("Bins with negative implied density:")
            lines.append(neg.to_string(index=False))
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "pvalue": self.pvalue,
            "statistic": self.statistic,
            "n_bins": self.n_bins,
            "n_units": self.n_units,
            "aggregation": self.aggregation,
            "n_sims": self.n_sims,
            "alpha": self.alpha,
            "table": self.table.to_dict(orient="records"),
            "diagnostics": dict(self.diagnostics),
        }


def _bin_edges(
    values: np.ndarray,
    n_bins: Optional[int],
    binpoints: Optional[Sequence[float]],
    context: str,
) -> np.ndarray:
    if binpoints is not None:
        edges = np.asarray(binpoints, dtype=float)
        if edges.ndim != 1 or edges.size < 2:
            raise MethodIncompatibility(
                f"{context}: `binpoints` must list at least two cut points.",
                diagnostics={"context": context, "n_binpoints": int(edges.size)},
            )
        return np.unique(edges)
    if n_bins is None:
        n_bins = 10
    n_bins = int(n_bins)
    if n_bins < 2:
        raise MethodIncompatibility(
            f"{context}: `n_bins` must be at least 2.",
            diagnostics={"context": context, "n_bins": n_bins},
        )
    lo, hi = float(np.min(values)), float(np.max(values))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi <= lo:
        raise DataInsufficient(
            f"{context}: the outcome has no spread to bin.",
            diagnostics={"context": context, "min": lo, "max": hi},
        )
    # R's `cut(x, breaks = n)` lays n+1 equally spaced edges from min to
    # max and then pushes ONLY the two outer ones out by dx/1000, so the
    # extreme observations fall strictly inside. The interior edges stay on
    # the unpadded grid, which makes the first and last bins very slightly
    # wider than the rest. Padding the whole grid instead shifts every
    # interior edge and moves the bin masses -- worth ~3e-4 per bin here.
    edges = np.linspace(lo, hi, n_bins + 1)
    pad = (hi - lo) / 1000.0
    edges[0] = lo - pad
    edges[-1] = hi + pad
    return edges


def functional_form_test(
    data: pd.DataFrame,
    y: str,
    *,
    g: str,
    t: str,
    i: str,
    n_bins: Optional[int] = 10,
    binpoints: Optional[Sequence[float]] = None,
    aggregation: str = "group",
    estimator: str = "dr",
    control_group: str = "nevertreated",
    x: Optional[List[str]] = None,
    anticipation: int = 0,
    n_sims: int = 100_000,
    alpha: float = 0.05,
    random_state: Optional[int] = 0,
) -> FunctionalFormResult:
    """Test whether parallel trends can hold for every monotonic transform.

    Parameters
    ----------
    data : DataFrame
        Long-format panel.
    y : str
        Outcome column.
    g : str
        First-treatment period; ``0`` marks never-treated units.
    t : str
        Time period column.
    i : str
        Unit identifier.
    n_bins : int, optional
        Number of equal-width outcome bins. Ignored when ``binpoints`` is
        given. The test has no power against violations that are invisible
        at the chosen resolution, so a too-coarse binning buys a large
        p-value for nothing.
    binpoints : sequence of float, optional
        Explicit bin edges, for when the outcome has natural cut points.
    aggregation : {"group", "simple", "dynamic", "calendar"}, default "group"
        Which :func:`aggte` aggregation defines "the" implied density.
        ``"group"`` is the reference implementation's default.
    estimator, control_group, x, anticipation
        Passed through to :func:`callaway_santanna`.
    n_sims : int, default 100000
        Draws used for the least-favourable critical value.
    alpha : float, default 0.05
    random_state : int, optional
        Seed for the simulation. Fixed by default so the p-value is
        reproducible; pass ``None`` for a fresh draw.

    Returns
    -------
    FunctionalFormResult

    Notes
    -----
    A failure to reject is not evidence *for* functional-form insensitivity.
    The test compares an estimated density against zero, so with few units,
    coarse bins, or a short panel it will fail to reject almost regardless of
    the truth.

    Examples
    --------
    >>> import statspai as sp
    >>> df = sp.datasets.mpdta()
    >>> res = sp.functional_form_test(
    ...     df, y="lemp", g="first_treat", t="year", i="countyreal", n_bins=6
    ... )
    >>> res.pvalue > 0.05
    True

    References
    ----------
    roth2023when
    """
    from .aggte import aggte
    from .callaway_santanna import callaway_santanna

    context = "functional_form_test"
    if aggregation not in _AGGTE_TYPES:
        raise MethodIncompatibility(
            f"{context}: `aggregation` must be one of {_AGGTE_TYPES}.",
            diagnostics={"context": context, "aggregation": aggregation},
        )
    if not 0.0 < float(alpha) < 1.0:
        raise MethodIncompatibility(
            f"{context}: `alpha` must be in (0, 1).",
            diagnostics={"context": context, "alpha": alpha},
        )
    n_sims = int(n_sims)
    if n_sims < 100:
        raise MethodIncompatibility(
            f"{context}: `n_sims` must be at least 100 for the simulated "
            "critical value to mean anything.",
            diagnostics={"context": context, "n_sims": n_sims},
        )

    for col in (y, g, t, i):
        if col not in data.columns:
            raise MethodIncompatibility(
                f"{context}: column {col!r} not in data.",
                diagnostics={"context": context, "columns": list(data.columns)},
            )

    df = data.copy()
    yv = np.asarray(df[y], dtype=float)
    if not np.all(np.isfinite(yv)):
        raise MethodIncompatibility(
            f"{context}: the outcome contains non-finite values.",
            diagnostics={
                "context": context,
                "n_nonfinite": int((~np.isfinite(yv)).sum()),
            },
        )

    # The bins are built on the UNTREATED observations only -- pre-treatment
    # periods for the treated cohorts plus every never-treated row. That is
    # the sample the counterfactual density is a density over, and using the
    # full panel instead would let post-treatment outcomes move the edges.
    gv = np.asarray(df[g], dtype=float)
    tv = np.asarray(df[t], dtype=float)
    untreated = (tv < gv) | (gv == 0)
    if not untreated.any():
        raise DataInsufficient(
            f"{context}: no untreated observations to bin over -- every row "
            "is a treated post-period, so there is no counterfactual "
            "distribution to test.",
            diagnostics={"context": context},
        )
    edges = _bin_edges(yv[untreated], n_bins, binpoints, context)
    # ``np.digitize`` with right-closed intervals mirrors R's
    # ``cut(..., right = TRUE)``: (lo, hi] with the first bin left-inclusive.
    codes = np.clip(np.digitize(yv, edges[1:-1], right=True), 0, len(edges) - 2)
    n_bin_total = len(edges) - 1

    treated_post = ~untreated

    estimates: List[float] = []
    inf_cols: List[np.ndarray] = []
    kept: List[int] = []
    _tmp = f"__ff_bin__{y}"
    for b in range(n_bin_total):
        df[_tmp] = np.where(treated_post, 0.0, -1.0 * (codes == b))
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cs = callaway_santanna(
                df,
                y=_tmp,
                g=g,
                t=t,
                i=i,
                x=x,
                estimator=estimator,
                control_group=control_group,
                base_period="universal",
                anticipation=anticipation,
                bstrap=False,
                cband=False,
            )
            agg = aggte(cs, type=aggregation, bstrap=False, cband=False)
        psi = agg.model_info.get("overall_influence_function")
        estimates.append(float(agg.estimate))
        if psi is None:
            continue
        psi = np.asarray(psi, dtype=float)
        # The reference drops bins whose influence function is numerically
        # zero: an empty bin carries no information and would make the
        # correlation matrix singular.
        if float(psi @ psi) > 1e-6:
            inf_cols.append(psi)
            kept.append(b)
    df.drop(columns=[_tmp], inplace=True, errors="ignore")

    if not inf_cols:
        raise DataInsufficient(
            f"{context}: every outcome bin produced a degenerate influence "
            "function, so the joint covariance is not estimable. Usually this "
            "means the binning is finer than the data can support.",
            diagnostics={"context": context, "n_bins": n_bin_total},
        )

    inf_matrix = np.column_stack(inf_cols)
    active = np.abs(inf_matrix).sum(axis=1) > 1e-6
    n_eff = int(active.sum())
    if n_eff < 2:
        raise DataInsufficient(
            f"{context}: fewer than two units contribute to the influence "
            "functions.",
            diagnostics={"context": context, "n_effective_units": n_eff},
        )

    # Sigmahat is the covariance of the *estimates*, so it carries 1/n twice:
    # once for the asymptotic variance and once for the sample size.
    asy_var = inf_matrix.T @ inf_matrix / n_eff
    sigma = asy_var / n_eff
    se = np.sqrt(np.diag(sigma))

    density = np.asarray([estimates[b] for b in kept], dtype=float)
    stat = float(np.max(-density / se))

    corr = _cov_to_corr(sigma)
    rng = np.random.default_rng(random_state)
    sims = rng.multivariate_normal(np.zeros(len(kept)), corr, size=n_sims)
    pvalue = float(np.mean(sims.max(axis=1) >= stat))

    table = pd.DataFrame(
        {
            "bin_lower": edges[:-1],
            "bin_upper": edges[1:],
            "implied_density": np.asarray(estimates, dtype=float),
            "se": [
                float(se[kept.index(b)]) if b in kept else np.nan
                for b in range(n_bin_total)
            ],
            "used": [b in kept for b in range(n_bin_total)],
        }
    )

    return FunctionalFormResult(
        pvalue=pvalue,
        table=table,
        statistic=stat,
        n_bins=len(kept),
        n_units=n_eff,
        aggregation=aggregation,
        n_sims=n_sims,
        alpha=float(alpha),
        diagnostics={
            "n_bins_requested": n_bin_total,
            "n_bins_dropped": n_bin_total - len(kept),
            "estimator": estimator,
            "control_group": control_group,
            "density_sum": float(np.sum(estimates)),
        },
    )


def _cov_to_corr(sigma: np.ndarray) -> np.ndarray:
    """Correlation matrix of ``sigma``, symmetrised for the simulator."""
    d = np.sqrt(np.diag(sigma))
    if np.any(d <= 0):
        raise DataInsufficient(
            "functional_form_test: a retained bin has zero estimated "
            "variance, so its correlation with the others is undefined.",
            diagnostics={"context": "functional_form_test"},
        )
    corr = sigma / np.outer(d, d)
    corr = (corr + corr.T) / 2.0
    np.fill_diagonal(corr, 1.0)
    return corr
