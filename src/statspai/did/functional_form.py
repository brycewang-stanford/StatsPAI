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
from typing import Any, Dict, List, NamedTuple, Optional, Sequence, Union

import numpy as np
import pandas as pd

from .._result_serialize import ResultProtocolMixin
from ..exceptions import DataInsufficient, MethodIncompatibility

__all__ = [
    "DistributionalDiDResult",
    "FunctionalFormResult",
    "distributional_did",
    "functional_form_test",
]

_AGGTE_TYPES = ("simple", "group", "dynamic", "calendar")

# The reference implementation bins an outcome with fewer than 20 distinct
# untreated values one-bin-per-value, and otherwise cuts it into
# ``min(20, n_distinct)`` equal-width bins.
_AUTO_BIN_MAX = 20


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

    def plot(
        self,
        ax: Any = None,
        figsize: tuple = (8, 5),
        lb: Optional[float] = None,
        ub: Optional[float] = None,
        **kwargs: Any,
    ) -> Any:
        """Bar chart of the implied counterfactual density, negatives flagged.

        This is the figure the test is really about: the p-value compresses
        into one number what the bars show directly — whether the
        distribution the design implies for the treated group's untreated
        outcome is a distribution at all. Bars below zero are the violation.

        Parameters
        ----------
        ax : matplotlib Axes, optional
        figsize : tuple, default (8, 5)
        lb, ub : float, optional
            Restrict the plotted outcome range, as ``didFF``'s ``lb_graph`` /
            ``ub_graph`` do. Bins outside are dropped from the picture only —
            the test already used all of them.
        **kwargs
            Passed to ``ax.bar``.

        Returns
        -------
        matplotlib.axes.Axes

        Examples
        --------
        >>> import statspai as sp  # doctest: +SKIP
        >>> res = sp.functional_form_test(df, y="lemp", g="first_treat",
        ...                               t="year", i="countyreal")  # doctest: +SKIP
        >>> ax = res.plot()  # doctest: +SKIP
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:  # pragma: no cover - optional dependency
            raise ImportError("matplotlib is required for plotting.")

        if ax is None:
            _, ax = plt.subplots(figsize=figsize)

        table = self.table
        level = table["level"].to_numpy(dtype=float)
        if lb is not None:
            table = table[level >= lb]
            level = table["level"].to_numpy(dtype=float)
        if ub is not None:
            table = table[level <= ub]
            level = table["level"].to_numpy(dtype=float)
        if not len(table):
            raise MethodIncompatibility(
                "functional_form_test.plot: the lb / ub window excludes every "
                "bin, so there is nothing to draw.",
                diagnostics={"lb": lb, "ub": ub, "n_bins": int(len(self.table))},
            )

        density = table["implied_density"].to_numpy(dtype=float)
        negative = density < 0
        width = kwargs.pop("width", None)
        if width is None:
            spans = np.diff(np.sort(level))
            width = float(spans[spans > 0].min()) * 0.9 if np.any(spans > 0) else 0.8

        bar_kw = dict(edgecolor="white", linewidth=0.5)
        bar_kw.update(kwargs)
        if negative.any():
            ax.bar(
                level[negative],
                density[negative],
                width=width,
                color="crimson",
                label="Negative",
                **bar_kw,
            )
        if (~negative).any():
            ax.bar(
                level[~negative],
                density[~negative],
                width=width,
                color="steelblue",
                label="Non-negative",
                **bar_kw,
            )
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Outcome")
        ax.set_ylabel("Implied density")
        ax.set_title(f"Implied counterfactual density (p = {self.pvalue:.3f})")
        ax.legend(frameon=False)
        return ax


class _Binning(NamedTuple):
    """How the outcome was discretised, and the bins that came out."""

    codes: np.ndarray  # bin index per row (ignored on treated post-rows)
    lower: np.ndarray  # per-bin lower edge (the value itself when discrete)
    upper: np.ndarray  # per-bin upper edge (the value itself when discrete)
    discrete: bool


def _bin_edges(
    values: np.ndarray,
    n_bins: Optional[int],
    binpoints: Optional[Sequence[float]],
    context: str,
) -> np.ndarray:
    """Equal-width cut points, laid out exactly as R's ``cut``."""
    if binpoints is not None:
        edges = np.asarray(binpoints, dtype=float)
        if edges.ndim != 1 or edges.size < 2:
            raise MethodIncompatibility(
                f"{context}: `binpoints` must list at least two cut points.",
                diagnostics={"context": context, "n_binpoints": int(edges.size)},
            )
        return np.asarray(np.unique(edges), dtype=float)
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
    return np.asarray(edges, dtype=float)


def _pad_binpoints(
    binpoints: Sequence[float], values: np.ndarray, context: str
) -> np.ndarray:
    """Extend user cut points to cover the outcome range, as ``didFF`` does.

    Cut points that stop short of the data would silently drop mass out of
    the density being tested, so the reference pads and warns instead.
    """
    edges = np.unique(np.asarray(binpoints, dtype=float))
    if edges.size < 1:
        raise MethodIncompatibility(
            f"{context}: `binpoints` must list at least one cut point.",
            diagnostics={"context": context, "n_binpoints": int(edges.size)},
        )
    lo, hi = float(np.min(values)), float(np.max(values))
    tol = float(np.sqrt(np.finfo(float).eps))

    if edges[0] - lo > 0 and abs(edges[0] - lo) > tol:
        edges = np.concatenate(([lo], edges))
        warnings.warn(
            f"{context}: `binpoints` do not cover the range of the outcome; "
            "padding at the lower bound.",
            UserWarning,
            stacklevel=3,
        )
    if edges[-1] - hi < 0 and abs(edges[-1] - hi) > tol:
        edges = np.concatenate((edges, [hi]))
        warnings.warn(
            f"{context}: `binpoints` do not cover the range of the outcome; "
            "padding at the upper bound.",
            UserWarning,
            stacklevel=3,
        )

    below = int(np.sum(edges <= lo))
    if below >= 1 and below < edges.size and edges[below] >= hi:
        raise MethodIncompatibility(
            f"{context}: the provided `binpoints` group every outcome value "
            "into a single bin, so there is no density to test.",
            recovery_hint="Add cut points inside the outcome range.",
            diagnostics={"context": context, "min": lo, "max": hi},
        )
    if edges.size < 2:
        raise MethodIncompatibility(
            f"{context}: `binpoints` must yield at least two cut points.",
            diagnostics={"context": context, "n_binpoints": int(edges.size)},
        )
    return np.asarray(edges, dtype=float)


def _build_bins(
    y_all: np.ndarray,
    y_untreated: np.ndarray,
    n_bins: Union[int, str, None],
    binpoints: Optional[Sequence[float]],
    context: str,
) -> _Binning:
    """Discretise the outcome the way ``didFF`` does.

    With neither ``n_bins`` nor ``binpoints`` given, the reference inspects
    the untreated outcome: fewer than 20 distinct values and it treats the
    outcome as discrete, one bin per value; otherwise it cuts into
    ``min(20, n_distinct)`` equal-width bins. An explicit ``n_bins`` always
    cuts, however few values there are.
    """
    explicit_bins = n_bins is not None and not isinstance(n_bins, str)
    if isinstance(n_bins, str) and n_bins != "auto":
        raise MethodIncompatibility(
            f"{context}: `n_bins` must be an int, None, or 'auto'; got " f"{n_bins!r}.",
            diagnostics={"context": context, "n_bins": n_bins},
        )
    if explicit_bins and binpoints is not None:
        raise MethodIncompatibility(
            f"{context}: pass only one of `n_bins` and `binpoints`.",
            recovery_hint="Drop one; `binpoints` gives explicit cut points, "
            "`n_bins` an equal-width count.",
            diagnostics={"context": context},
        )

    if binpoints is not None:
        edges = _pad_binpoints(binpoints, y_untreated, context)
        return _cut(y_all, edges)

    distinct = np.unique(y_untreated)
    if distinct.size <= 1:
        raise DataInsufficient(
            f"{context}: the outcome takes a single value among untreated "
            "observations, so there is no distribution to test.",
            diagnostics={"context": context, "n_distinct": int(distinct.size)},
        )

    if n_bins is not None and not isinstance(n_bins, str):
        return _cut(y_all, _bin_edges(y_untreated, int(n_bins), None, context))

    if distinct.size < _AUTO_BIN_MAX:
        warnings.warn(
            f"{context}: the outcome takes only {distinct.size} distinct "
            "values among untreated observations, so it is being treated as "
            "discrete (one bin per value). Pass n_bins or binpoints to bin it "
            "instead.",
            UserWarning,
            stacklevel=3,
        )
        codes = np.clip(np.searchsorted(distinct, y_all), 0, distinct.size - 1)
        return _Binning(codes=codes, lower=distinct, upper=distinct, discrete=True)

    edges = _bin_edges(y_untreated, min(_AUTO_BIN_MAX, distinct.size), None, context)
    return _cut(y_all, edges)


def _bin_att_series(
    df: pd.DataFrame,
    *,
    y: str,
    g: str,
    t: str,
    i: str,
    codes: np.ndarray,
    n_bin_total: int,
    treated_post: np.ndarray,
    distributional: bool,
    cs_kwargs: Dict[str, Any],
    agg_kwargs: Dict[str, Any],
) -> tuple:
    """Run Callaway-Sant'Anna once per outcome bin.

    Two outcomes are possible, and the difference is the whole difference
    between the two things this module computes:

    ``distributional=False``
        ``-1{Y in b}`` on untreated rows and ``0`` on treated post-rows. The
        aggregate ATT of that variable *is* the counterfactual probability
        mass the design implies for bin ``b``, which is what the
        functional-form test checks the sign of.
    ``distributional=True``
        ``+1{Y in b}`` everywhere. The aggregate ATT is then the treatment
        effect on the probability of landing in bin ``b`` — a distributional
        DiD, not a counterfactual density.

    Returns ``(estimates, influence_columns, kept_bin_indices)``. Bins whose
    influence function is numerically zero are dropped from the last two, as
    in the reference: an empty bin carries no information and would make the
    joint covariance singular.
    """
    from .aggte import aggte
    from .callaway_santanna import callaway_santanna

    estimates: List[float] = []
    inf_cols: List[np.ndarray] = []
    kept: List[int] = []
    tmp = f"__ff_bin__{y}"
    for b in range(n_bin_total):
        indicator = codes == b
        if distributional:
            df[tmp] = indicator.astype(float)
        else:
            df[tmp] = np.where(treated_post, 0.0, -1.0 * indicator)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            cs = callaway_santanna(
                df,
                y=tmp,
                g=g,
                t=t,
                i=i,
                base_period="universal",
                bstrap=False,
                cband=False,
                **cs_kwargs,
            )
            agg = aggte(cs, bstrap=False, cband=False, **agg_kwargs)
        psi = agg.model_info.get("overall_influence_function")
        estimates.append(float(agg.estimate))
        if psi is None:
            continue
        psi = np.asarray(psi, dtype=float)
        if float(psi @ psi) > 1e-6:
            inf_cols.append(psi)
            kept.append(b)
    df.drop(columns=[tmp], inplace=True, errors="ignore")
    return estimates, inf_cols, kept


def _cut(y_all: np.ndarray, edges: np.ndarray) -> _Binning:
    """Assign rows to half-open bins, mirroring R's ``cut(..., right = TRUE)``."""
    codes = np.clip(np.digitize(y_all, edges[1:-1], right=True), 0, len(edges) - 2)
    return _Binning(codes=codes, lower=edges[:-1], upper=edges[1:], discrete=False)


def functional_form_test(
    data: pd.DataFrame,
    y: str,
    *,
    g: str,
    t: str,
    i: str,
    n_bins: Union[int, str, None] = "auto",
    binpoints: Optional[Sequence[float]] = None,
    aggregation: str = "group",
    estimator: str = "dr",
    control_group: str = "nevertreated",
    x: Optional[List[str]] = None,
    weights: Optional[str] = None,
    anticipation: int = 0,
    panel: bool = True,
    allow_unbalanced_panel: bool = False,
    balance_e: Optional[int] = None,
    min_e: float = -np.inf,
    max_e: float = np.inf,
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
    n_bins : int or "auto", default "auto"
        Number of equal-width outcome bins. ``"auto"`` follows the reference
        implementation: an outcome with fewer than 20 distinct untreated
        values is treated as discrete (one bin per value, with a warning),
        otherwise it is cut into ``min(20, n_distinct)`` bins. An explicit
        integer always cuts. The test has no power against violations that
        are invisible at the chosen resolution, so a too-coarse binning buys
        a large p-value for nothing.
    binpoints : sequence of float, optional
        Explicit bin edges, for when the outcome has natural cut points.
        Padded to cover the outcome range (with a warning) if they fall
        short. Cannot be combined with an explicit ``n_bins``.
    aggregation : {"group", "simple", "dynamic", "calendar"}, default "group"
        Which :func:`aggte` aggregation defines "the" implied density.
        ``"group"`` is the reference implementation's default.
    estimator, control_group, x, weights, anticipation, panel, allow_unbalanced_panel
        Passed through to :func:`callaway_santanna`. ``weights`` names a
        sampling-weight column; leaving it unset weights every unit equally.
    balance_e, min_e, max_e
        Passed through to :func:`aggte`; they only bite when
        ``aggregation="dynamic"``, where they restrict which event times
        enter the aggregate.
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
    binning = _build_bins(yv, yv[untreated], n_bins, binpoints, context)
    codes = binning.codes
    n_bin_total = binning.lower.size

    treated_post = ~untreated

    estimates, inf_cols, kept = _bin_att_series(
        df,
        y=y,
        g=g,
        t=t,
        i=i,
        codes=codes,
        n_bin_total=n_bin_total,
        treated_post=treated_post,
        distributional=False,
        cs_kwargs=dict(
            x=x,
            weights=weights,
            estimator=estimator,
            control_group=control_group,
            anticipation=anticipation,
            panel=panel,
            allow_unbalanced_panel=allow_unbalanced_panel,
        ),
        agg_kwargs=dict(
            type=aggregation, balance_e=balance_e, min_e=min_e, max_e=max_e
        ),
    )

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
            "bin_lower": binning.lower,
            "bin_upper": binning.upper,
            # ``level`` is the bin's upper endpoint, which is what the
            # reference implementation tabulates and plots against.
            "level": binning.upper,
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
            "discrete_outcome": bool(binning.discrete),
            "weighted": weights is not None,
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
    return np.asarray(corr, dtype=float)


@dataclass
class DistributionalDiDResult(ResultProtocolMixin):
    """Result of :func:`distributional_did`.

    Attributes
    ----------
    table : pandas.DataFrame
        One row per outcome bin: ``bin_lower``, ``bin_upper``, ``level``,
        ``estimate`` (the effect on the probability of landing in that bin),
        ``se``, and ``used``.
    n_units : int
        Units contributing to the influence functions.
    aggregation : str
        Which ``aggte`` aggregation produced the per-bin estimates.
    alpha : float
    diagnostics : dict
    """

    table: pd.DataFrame
    n_units: int
    aggregation: str
    alpha: float = 0.05
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    method: str = "Distributional DiD (treatment effect on the outcome distribution)"

    def summary(self) -> str:
        lines = [
            self.method,
            "=" * len(self.method),
            "Effect on P(Y in bin), one row per bin",
            f"  bins used     : {int(self.table['used'].sum())}",
            f"  aggregation   : {self.aggregation}",
            f"  units         : {self.n_units}",
            "",
            "The per-bin effects sum to approximately zero by construction: "
            "treatment moves mass between bins, it does not create it. Read "
            "the sign pattern, not the level.",
            "",
            self.table.to_string(index=False),
        ]
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "method": self.method,
            "n_units": self.n_units,
            "aggregation": self.aggregation,
            "alpha": self.alpha,
            "table": self.table.to_dict(orient="records"),
            "diagnostics": dict(self.diagnostics),
        }

    def plot(
        self,
        ax: Any = None,
        figsize: tuple = (8, 5),
        **kwargs: Any,
    ) -> Any:
        """Bar chart of the per-bin effect with pointwise confidence intervals.

        Parameters
        ----------
        ax : matplotlib Axes, optional
        figsize : tuple, default (8, 5)
        **kwargs
            Passed to ``ax.bar``.

        Returns
        -------
        matplotlib.axes.Axes
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:  # pragma: no cover - optional dependency
            raise ImportError("matplotlib is required for plotting.")
        from scipy import stats as _stats

        if ax is None:
            _, ax = plt.subplots(figsize=figsize)

        used = self.table[self.table["used"]]
        level = used["level"].to_numpy(dtype=float)
        estimate = used["estimate"].to_numpy(dtype=float)
        se = used["se"].to_numpy(dtype=float)
        z = float(_stats.norm.ppf(1 - self.alpha / 2))

        spans = np.diff(np.sort(level))
        width = float(spans[spans > 0].min()) * 0.9 if np.any(spans > 0) else 0.8
        bar_kw = dict(color="steelblue", edgecolor="white", linewidth=0.5)
        bar_kw.update(kwargs)
        ax.bar(level, estimate, width=kwargs.pop("width", width), **bar_kw)
        ax.errorbar(
            level, estimate, yerr=z * se, fmt="none", ecolor="black", linewidth=0.8
        )
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_xlabel("Outcome")
        ax.set_ylabel("Effect on P(Y in bin)")
        ax.set_title("Distributional DiD")
        return ax


def distributional_did(
    data: pd.DataFrame,
    y: str,
    *,
    g: str,
    t: str,
    i: str,
    n_bins: Union[int, str, None] = "auto",
    binpoints: Optional[Sequence[float]] = None,
    aggregation: str = "group",
    estimator: str = "dr",
    control_group: str = "nevertreated",
    x: Optional[List[str]] = None,
    weights: Optional[str] = None,
    anticipation: int = 0,
    panel: bool = True,
    allow_unbalanced_panel: bool = False,
    balance_e: Optional[int] = None,
    min_e: float = -np.inf,
    max_e: float = np.inf,
    alpha: float = 0.05,
) -> DistributionalDiDResult:
    """Treatment effect on the **distribution** of the outcome, bin by bin.

    Where :func:`functional_form_test` asks whether the design's implied
    counterfactual density is a density at all, this asks a different
    question: how did treatment move probability mass around? Bin the
    outcome, run Callaway-Sant'Anna on each bin indicator, and read off the
    effect on ``P(Y in bin)``.

    The per-bin effects sum to approximately zero by construction — treatment
    redistributes mass, it does not create it — so the informative content is
    the *shape*: which parts of the outcome distribution gained and which
    lost. A mean ATT of zero is perfectly consistent with large offsetting
    movements in the tails, and this is what shows them.

    This is R ``didFF::distDD``. Unlike :func:`functional_form_test` it runs
    no test and returns no p-value: the reference reports point estimates and
    standard errors only.

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
    n_bins, binpoints
        Binning, exactly as in :func:`functional_form_test`. Note the bins
        here are built over the **whole** panel, not the untreated rows only:
        the object of interest is the treated group's own outcome
        distribution, so post-treatment values must be inside the support.
    aggregation, estimator, control_group, x, weights, anticipation, panel
    allow_unbalanced_panel, balance_e, min_e, max_e
        Passed through to :func:`callaway_santanna` and :func:`aggte`.
    alpha : float, default 0.05
        Level for the confidence intervals drawn by ``.plot()``.

    Returns
    -------
    DistributionalDiDResult

    Examples
    --------
    >>> import statspai as sp
    >>> df = sp.datasets.mpdta()
    >>> res = sp.distributional_did(
    ...     df, y="lemp", g="first_treat", t="year", i="countyreal", n_bins=6
    ... )
    >>> len(res.table)
    6
    >>> bool(abs(res.diagnostics["effect_sum"]) < 1e-8)  # mass is conserved
    True

    See Also
    --------
    functional_form_test : is parallel trends sensitive to the outcome scale?

    References
    ----------
    roth2023when
    """
    context = "distributional_did"
    if aggregation not in _AGGTE_TYPES:
        raise MethodIncompatibility(
            f"{context}: `aggregation` must be one of {_AGGTE_TYPES}.",
            diagnostics={"context": context, "aggregation": aggregation},
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

    # Bins span the whole panel here, which is the one structural difference
    # from the functional-form test: that test bins over untreated rows so
    # post-treatment outcomes cannot move the edges of the density it is
    # testing, while this one is *about* where treated mass ended up.
    binning = _build_bins(yv, yv, n_bins, binpoints, context)
    n_bin_total = binning.lower.size

    estimates, inf_cols, kept = _bin_att_series(
        df,
        y=y,
        g=g,
        t=t,
        i=i,
        codes=binning.codes,
        n_bin_total=n_bin_total,
        treated_post=np.zeros(len(df), dtype=bool),
        distributional=True,
        cs_kwargs=dict(
            x=x,
            weights=weights,
            estimator=estimator,
            control_group=control_group,
            anticipation=anticipation,
            panel=panel,
            allow_unbalanced_panel=allow_unbalanced_panel,
        ),
        agg_kwargs=dict(
            type=aggregation, balance_e=balance_e, min_e=min_e, max_e=max_e
        ),
    )

    if not inf_cols:
        raise DataInsufficient(
            f"{context}: every outcome bin produced a degenerate influence "
            "function, so no standard error is estimable. Usually this means "
            "the binning is finer than the data can support.",
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

    sigma = (inf_matrix.T @ inf_matrix / n_eff) / n_eff
    se = np.sqrt(np.diag(sigma))

    table = pd.DataFrame(
        {
            "bin_lower": binning.lower,
            "bin_upper": binning.upper,
            "level": binning.upper,
            "estimate": np.asarray(estimates, dtype=float),
            "se": [
                float(se[kept.index(b)]) if b in kept else np.nan
                for b in range(n_bin_total)
            ],
            "used": [b in kept for b in range(n_bin_total)],
        }
    )

    return DistributionalDiDResult(
        table=table,
        n_units=n_eff,
        aggregation=aggregation,
        alpha=float(alpha),
        diagnostics={
            "n_bins_requested": n_bin_total,
            "n_bins_dropped": n_bin_total - len(kept),
            "estimator": estimator,
            "control_group": control_group,
            "discrete_outcome": bool(binning.discrete),
            "weighted": weights is not None,
            "effect_sum": float(np.sum(estimates)),
        },
    )
