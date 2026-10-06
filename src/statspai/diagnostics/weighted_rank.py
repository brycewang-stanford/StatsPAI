"""Sensitivity analysis with weighted rank statistics in block designs.

An observational block design has ``I`` blocks of ``J`` individuals matched
for observed covariates, some treated and the rest controls. A weighted rank
statistic [@rosenbaum2023bahadur] ranks the outcomes within each block,
ranks the blocks by how dispersed their outcomes are, and sums the treated
ranks with a weight ``phi`` that grows with the block's dispersion rank.
Constant weights give the stratified Wilcoxon statistic, linear weights
Quade's statistic [@quade1979using], and weights that all but ignore the
quiet blocks give tests that are much less sensitive to unmeasured bias when
the treatment has an effect.

:func:`weighted_rank` reports the upper bound on the one-sided p-value when
the odds of treatment within a block may differ by a factor of at most
``Gamma`` because of an unobserved covariate, the ``Gamma`` at which the
bound crosses ``alpha``, and on request the corresponding bounds on the
Hodges-Lehmann estimate and confidence interval for an additive effect.
:func:`weighted_rank_power` estimates, from the data in hand, the power such
an analysis would have in a larger or smaller study.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy import integrate, special, stats

from .._input_validation import require_columns
from .._result_serialize import ResultProtocolMixin
from ..exceptions import DataInsufficient, MethodIncompatibility
from . import _sens_engine as _eng

PhiSpec = Union[str, Tuple[int, int, int], Callable[[np.ndarray], np.ndarray]]

_PHI_ALIASES = {
    "wilcoxon": "wilcoxon",
    "wilc": "wilcoxon",
    "quade": "quade",
    "u858": (8, 5, 8),
    "u868": (8, 6, 8),
    "u878": (8, 7, 8),
    "u888": (8, 8, 8),
    "mixed": "mixed",
}


# --------------------------------------------------------------------
# Block weights
# --------------------------------------------------------------------


def _u_weight(p: np.ndarray, m: int, m1: int, m2: int) -> np.ndarray:
    """The U-statistic weight ``sum_{l=m1}^{m2} l C(m,l) p^(l-1) (1-p)^(m-l)``.

    Expression (9) of [@rosenbaum2011new]: the chance, up to a constant,
    that a block is between the ``m1``-th and ``m2``-th most dispersed of
    ``m`` blocks drawn at random.
    """
    out = np.zeros_like(p, dtype=float)
    for ell in range(m1, m2 + 1):
        out += ell * special.comb(m, ell) * p ** (ell - 1) * (1.0 - p) ** (m - ell)
    return out


def _phi_label(phi: PhiSpec) -> str:
    if callable(phi):
        return getattr(phi, "__name__", "custom")
    if isinstance(phi, str):
        return phi.lower()
    m, m1, m2 = phi
    return f"u{m}{m1}{m2}" if max(m, m1, m2) < 10 else f"u({m},{m1},{m2})"


def _block_weights(p: np.ndarray, phi: PhiSpec) -> np.ndarray:
    """Weights of the blocks from their dispersion ranks ``p`` in ``(0, 1]``.

    Named weights are scaled so that the largest is one; a callable is used
    as given.
    """
    if callable(phi):
        w = np.asarray(phi(p), dtype=float)
        if w.shape != p.shape or np.any(w < 0) or not np.all(np.isfinite(w)):
            raise MethodIncompatibility(
                "phi must map the block ranks to one finite non-negative "
                "weight per block"
            )
        return w
    spec: Any = phi
    if isinstance(phi, str):
        key = phi.lower()
        if key not in _PHI_ALIASES:
            raise MethodIncompatibility(
                f"unknown phi {phi!r}; expected one of {sorted(_PHI_ALIASES)}, "
                "a triple (m, m1, m2) or a callable"
            )
        spec = _PHI_ALIASES[key]
    if spec == "wilcoxon":
        return np.ones_like(p, dtype=float)
    if spec == "quade":
        w = p.astype(float)
    elif spec == "mixed":
        w = _u_weight(p, 20, 19, 20) + _u_weight(p, 20, 19, 19)
    else:
        try:
            m, m1, m2 = (int(v) for v in spec)
        except (TypeError, ValueError):
            raise MethodIncompatibility(
                "phi must be a name, a triple (m, m1, m2) or a callable"
            ) from None
        if not 1 <= m1 <= m2 <= m:
            raise MethodIncompatibility("phi=(m, m1, m2) needs 1 <= m1 <= m2 <= m")
        w = _u_weight(p, m, m1, m2)
    top = float(w.max()) if w.size else 0.0
    return w / top if top > 0 else w


def _dispersion(y: np.ndarray, block_scale: str) -> np.ndarray:
    top = y.max(axis=1)
    if block_scale == "range":
        return np.asarray(top - y.min(axis=1))
    return np.asarray(top - (y.sum(axis=1) - top) / (y.shape[1] - 1))


def _within_scores(y: np.ndarray, scores: Optional[np.ndarray]) -> np.ndarray:
    if scores is None:
        return np.asarray(stats.rankdata(y, method="average", axis=1))
    low = stats.rankdata(y, method="min", axis=1).astype(int)
    return np.asarray(scores[low - 1])


# --------------------------------------------------------------------
# Input
# --------------------------------------------------------------------


def _as_blocks(
    y: Any,
    data: Optional[pd.DataFrame],
    treat: Optional[str],
    block: Optional[str],
    treated: Any,
    caller: str,
) -> Tuple[np.ndarray, np.ndarray]:
    """Return outcomes and 0/1 treatment indicators as ``(I, J)`` arrays."""
    if data is not None:
        if not isinstance(y, str) or treat is None or block is None:
            raise MethodIncompatibility(
                f"{caller}: with data=, pass the outcome column as y and name "
                "the treat= and block= columns"
            )
        require_columns(data, [y, treat, block], function=caller)
        d = data[[y, treat, block]].dropna()
        z = d[treat].to_numpy(dtype=float)
        if not np.all((z == 0) | (z == 1)):
            raise MethodIncompatibility(f"{caller}: the treat column must be coded 0/1")
        sizes = d.groupby(block, sort=True)[y].size()
        if sizes.nunique() != 1:
            raise MethodIncompatibility(
                f"{caller}: blocks must all have the same size; found sizes "
                f"{sorted(int(s) for s in sizes.unique())}.",
                recovery_hint=(
                    "Analyse each block size on its own and combine the "
                    "p-values with sp.truncated_product, or use "
                    "sp.rosenbaum_stratified, which takes strata of any size."
                ),
                diagnostics={"block_sizes": sizes.value_counts().to_dict()},
                alternative_functions=[
                    "sp.rosenbaum_stratified",
                    "sp.truncated_product",
                ],
            )
        d = d.sort_values(block, kind="stable")
        width = int(sizes.iloc[0])
        y_arr = d[y].to_numpy(dtype=float).reshape(-1, width)
        z_arr = d[treat].to_numpy(dtype=float).reshape(-1, width)
        return y_arr, z_arr

    if isinstance(y, str) or y is None:
        raise MethodIncompatibility(
            f"{caller}: pass a blocks-by-individuals array as y, or a column "
            "name together with data=, treat= and block="
        )
    y_arr = np.asarray(y, dtype=float)
    if y_arr.ndim != 2 or min(y_arr.shape) < 2:
        raise MethodIncompatibility(
            f"{caller}: y must have one row per block and at least two "
            "columns and two rows"
        )
    if isinstance(treated, (int, float, np.integer, np.floating)):
        k = int(treated)
        if not 1 <= k < y_arr.shape[1] or k != treated:
            raise MethodIncompatibility(
                f"{caller}: treated must be an integer between 1 and J - 1, "
                "the number of leading columns that are treated"
            )
        z_arr = np.zeros_like(y_arr)
        z_arr[:, :k] = 1.0
    else:
        z_arr = np.asarray(treated, dtype=float)
        if z_arr.shape != y_arr.shape or not np.all((z_arr == 0) | (z_arr == 1)):
            raise MethodIncompatibility(
                f"{caller}: treated must be an integer or a 0/1 array of the "
                "shape of y"
            )
    if not np.all(np.isfinite(y_arr)):
        raise MethodIncompatibility(f"{caller}: y contains missing or infinite values")
    return y_arr, z_arr


def _drop_uninformative(
    y: np.ndarray, z: np.ndarray, caller: str
) -> Tuple[np.ndarray, np.ndarray]:
    k = z.sum(axis=1)
    keep = (k > 0) & (k < z.shape[1])
    if not keep.all():
        warnings.warn(
            f"{caller}: {int((~keep).sum())} blocks have no treated unit or "
            "no control and were dropped",
            UserWarning,
            stacklevel=3,
        )
    if keep.sum() < 2:
        raise DataInsufficient(
            f"{caller}: fewer than two blocks contain both a treated unit "
            "and a control.",
            recovery_hint="Check the treat and block columns.",
            diagnostics={"n_blocks": int(keep.sum())},
            alternative_functions=[],
        )
    return y[keep], z[keep]


# --------------------------------------------------------------------
# The statistic and its bounds
# --------------------------------------------------------------------


@dataclass
class _Design:
    """What is fixed once the outcomes are fixed, whatever ``Gamma`` is."""

    weights: List[np.ndarray]  # one array of block weights per phi
    treated_sum: np.ndarray  # (I,) sum of treated within-block scores
    sorted_scores: np.ndarray  # (I, J)
    n_treated: np.ndarray  # (I,)
    sign: float

    def statistics(self) -> np.ndarray:
        return np.array([float(np.sum(w * self.treated_sum)) for w in self.weights])


def _design(
    y: np.ndarray,
    z: np.ndarray,
    phis: Sequence[PhiSpec],
    scores: Optional[np.ndarray],
    block_scale: str,
    less: bool,
) -> _Design:
    n_blocks = y.shape[0]
    p = stats.rankdata(_dispersion(y, block_scale), method="average") / n_blocks
    weights = [_block_weights(p, phi) for phi in phis]
    within = _within_scores(y, scores)
    sign = -1.0 if less else 1.0
    within = sign * within
    return _Design(
        weights=weights,
        treated_sum=(within * z).sum(axis=1),
        sorted_scores=np.sort(within, axis=1),
        n_treated=np.rint(z.sum(axis=1)).astype(int),
        sign=sign,
    )


def _bounds(design: _Design, gamma: float, kappa: float) -> List[_eng.BoundPieces]:
    """Bounds for each phi; the moment tables are shared across them."""
    tables = []
    for k in np.unique(design.n_treated):
        rows = design.n_treated == k
        mu, nu = _eng.moment_tables(design.sorted_scores[rows], int(k), gamma)
        tables.append((rows, mu, nu))
    out = []
    for w in design.weights:
        groups = [_eng._Group(w[rows], mu, nu) for rows, mu, nu in tables]
        out.append(_eng.combine(groups, kappa))
    return out


def _covariance(design: _Design, gamma: float, a: int, b: int) -> float:
    """Covariance of two weighted statistics at the separable worst case."""
    cov = 0.0
    for k in np.unique(design.n_treated):
        rows = design.n_treated == k
        mu, nu = _eng.moment_tables(design.sorted_scores[rows], int(k), gamma)
        _, v = _eng.separable_choice(mu, nu)
        cov += float(np.sum(design.weights[a][rows] * design.weights[b][rows] * v))
    return cov


def _bvn_upper(h: float, rho: float) -> float:
    """``1 - P(Z1 <= h, Z2 <= h)`` for a standard bivariate normal.

    Uses the single-integral form of the bivariate normal distribution
    function, so the result does not depend on a Monte Carlo sample.
    """
    rho = float(np.clip(rho, -1.0, 1.0))
    if not np.isfinite(h):
        return 0.0 if h > 0 else 1.0

    def integrand(theta: float) -> float:
        c = np.cos(theta)
        if c <= 0:
            return 0.0
        return float(np.exp(-h * h * (1.0 - np.sin(theta)) / (c * c)))

    area, _ = integrate.quad(integrand, 0.0, np.arcsin(rho), epsabs=0, epsrel=1e-13)
    # 1 - Phi(h)^2 written without cancellation for large h
    sf = float(stats.norm.sf(h))
    return float(max(sf * (2.0 - sf) - area / (2.0 * np.pi), 0.0))


def _joint_upper(deviates: np.ndarray, corr: np.ndarray) -> float:
    """Chance that the largest of correlated standard normals exceeds the
    largest observed deviate."""
    top = float(np.max(deviates))
    if len(deviates) == 2:
        return _bvn_upper(top, float(corr[0, 1]))
    dist = stats.multivariate_normal(
        mean=np.zeros(len(deviates)), cov=corr, allow_singular=True
    )
    return float(1.0 - dist.cdf(np.full(len(deviates), top)))


def _one_sided(
    y: np.ndarray,
    z: np.ndarray,
    phis: Sequence[PhiSpec],
    scores: Optional[np.ndarray],
    block_scale: str,
    less: bool,
    gamma: float,
    kappa: float,
    bound: str,
) -> dict:
    design = _design(y, z, phis, scores, block_scale, less)
    stat = design.statistics()
    pieces = _bounds(design, gamma, kappa)
    taylor = bound == "taylor"
    dev = np.array([p.deviate(s, taylor) for p, s in zip(pieces, stat)])
    exp = np.array([p.expectation_taylor if taylor else p.expectation for p in pieces])
    var = np.array([p.variance_taylor if taylor else p.variance for p in pieces])
    out = {
        "statistic": design.sign * stat,
        "expectation": design.sign * exp,
        "variance": var,
        "deviate": dev,
        "pvalues": stats.norm.sf(dev),
        "corr": None,
    }
    if len(phis) == 1:
        out["pvalue"] = float(out["pvalues"][0])
        return out
    k = len(phis)
    corr = np.eye(k)
    sep_var = np.array([p.variance for p in pieces])
    for a in range(k):
        for b in range(a + 1, k):
            c = _covariance(design, gamma, a, b)
            corr[a, b] = corr[b, a] = c / np.sqrt(sep_var[a] * sep_var[b])
    out["corr"] = corr
    out["pvalue"] = _joint_upper(dev, corr)
    return out


# --------------------------------------------------------------------
# The conditional test
# --------------------------------------------------------------------


def _conditional(
    y: np.ndarray, z: np.ndarray, phi: PhiSpec, less: bool, gamma: float
) -> dict:
    """Test that uses only the extreme responses of each block.

    Among the individuals who hold the largest or the smallest outcome of a
    block, the statistic is the weighted share of the treated ones that hold
    the largest. A block enters when its extremes include a treated unit
    and a control. Given which individuals are extreme, the count of
    treated maxima is noncentral hypergeometric under the sensitivity
    model, so the bound needs no approximation [@rosenbaum2025conditioning].
    """
    if less:
        y = -y
    n_blocks = y.shape[0]
    top = y.max(axis=1, keepdims=True)
    bottom = y.min(axis=1, keepdims=True)
    spread = (top - bottom)[:, 0]
    weights = _block_weights(stats.rankdata(spread, method="average") / n_blocks, phi)
    at_top = y == top
    at_bottom = y == bottom
    extreme = at_top | at_bottom
    n_ext_treated = (extreme * z).sum(axis=1)
    n_ext_control = (extreme * (1 - z)).sum(axis=1)
    decisive = (spread > 0) & (n_ext_treated >= 1) & (n_ext_control >= 1)
    if not decisive.any():
        raise DataInsufficient(
            "weighted_rank: no block has a treated unit and a control among "
            "its extreme responses.",
            recovery_hint="Use conditional=False.",
            diagnostics={"n_blocks": int(n_blocks)},
            alternative_functions=[],
        )
    w = weights[decisive] / n_ext_treated[decisive]
    n_top = at_top[decisive].sum(axis=1)
    n_bottom = at_bottom[decisive].sum(axis=1)
    drawn = np.rint(n_ext_treated[decisive]).astype(int)
    treated_top = (at_top[decisive] * z[decisive]).sum(axis=1)

    statistic = float(np.sum(w * treated_top))
    expectation = variance = 0.0
    shapes = np.stack([n_top, n_bottom, drawn], axis=1)
    for shape in np.unique(shapes, axis=0):
        rows = np.all(shapes == shape, axis=1)
        m1, m2, n = (int(v) for v in shape)
        ea, ea2 = _eng.fnch_moments(np.array([m1]), m1 + m2, n, gamma)
        expectation += float(ea[0] * np.sum(w[rows]))
        variance += float((ea2[0] - ea[0] ** 2) * np.sum(w[rows] ** 2))
    deviate = (statistic - expectation) / np.sqrt(variance) if variance > 0 else 0.0
    tied = (spread > 0) & (extreme.sum(axis=1) > 2)
    return {
        "statistic": np.array([statistic]),
        "expectation": np.array([expectation]),
        "variance": np.array([variance]),
        "deviate": np.array([deviate]),
        "pvalues": np.array([float(stats.norm.sf(deviate))]),
        "pvalue": float(stats.norm.sf(deviate)),
        "corr": None,
        "n_decisive": int(decisive.sum()),
        "n_treated_extreme_favourable": int(np.sum(treated_top >= 1)),
        "n_blocks_tied_extremes": int(tied.sum()),
    }


# --------------------------------------------------------------------
# Solving for the shift
# --------------------------------------------------------------------


def _last_true(pred: Callable[[float], bool], lo: float, hi: float) -> float:
    """Largest ``tau`` in ``[lo, hi]`` with ``pred(tau)`` true, for a
    predicate that is true up to some point and false after it."""
    if not pred(lo):
        return lo
    if pred(hi):
        return hi
    scale = max(1.0, abs(lo), abs(hi))
    while hi - lo > 1e-11 * scale:
        mid = 0.5 * (lo + hi)
        if pred(mid):
            lo = mid
        else:
            hi = mid
    return 0.5 * (lo + hi)


# --------------------------------------------------------------------
# Result
# --------------------------------------------------------------------


@dataclass
class WeightedRankResult(ResultProtocolMixin):
    """Result of :func:`weighted_rank`.

    Attributes
    ----------
    pvalue : float
        Upper bound on the p-value at the first ``gamma``.
    gamma : float
        The first (or only) value of ``Gamma`` analysed.
    gamma_critical : float
        The ``Gamma`` at which the bound equals ``alpha``: the study is
        insensitive to biases smaller than this. ``1.0`` when the test does
        not reject even without hidden bias.
    statistic, expectation, variance, deviate : float
        The weighted rank statistic, the bounds on its null moments, and the
        standardised deviate, at the first ``gamma``.
    detail : pandas.DataFrame
        One row per ``Gamma`` (and per ``phi`` when several are given).
    estimate, conf_int : tuple of float, optional
        With ``estimates=True``, the range of Hodges-Lehmann point estimates
        and the confidence interval for an additive effect at the first
        ``gamma``.

    Examples
    --------
    >>> import numpy as np
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> y = rng.normal(size=(200, 3))
    >>> y[:, 0] += 1.0
    >>> res = sp.weighted_rank(y, gamma=2.0)
    >>> type(res).__name__
    'WeightedRankResult'
    >>> res.n_blocks, res.block_size
    (200, 3)
    >>> bool(res.pvalue < 0.05)
    True
    """

    _citation_keys = ("rosenbaum2023bahadur",)

    pvalue: float
    gamma: float
    gamma_critical: float
    statistic: float
    expectation: float
    variance: float
    deviate: float
    phi: str
    alternative: str
    alpha: float
    n_blocks: int
    block_size: int
    conditional: bool = False
    bound: str = "separable"
    estimate: Optional[Tuple[float, float]] = None
    conf_int: Optional[Tuple[float, float]] = None
    detail: pd.DataFrame = field(default_factory=pd.DataFrame)
    diagnostics: dict = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            "Weighted rank sensitivity analysis",
            "==================================",
            f"Blocks x size    : {self.n_blocks} x {self.block_size}",
            f"Weights (phi)    : {self.phi}"
            + ("  (conditional on extremes)" if self.conditional else ""),
            f"Alternative      : {self.alternative}",
            f"Critical Gamma   : {self.gamma_critical:.4f}  (alpha={self.alpha})",
            "",
            self.detail.to_string(index=False, float_format=lambda v: f"{v:.6g}"),
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"WeightedRankResult(phi={self.phi}, gamma={self.gamma:g}, "
            f"pvalue={self.pvalue:.4g}, gamma_critical={self.gamma_critical:.3f})"
        )


# --------------------------------------------------------------------
# Public API
# --------------------------------------------------------------------


def weighted_rank(
    y: Any,
    *,
    data: Optional[pd.DataFrame] = None,
    treat: Optional[str] = None,
    block: Optional[str] = None,
    treated: Any = 1,
    gamma: Union[float, Sequence[float]] = 1.0,
    phi: Union[PhiSpec, List[PhiSpec]] = "u868",
    alternative: str = "greater",
    scores: Optional[Sequence[float]] = None,
    block_scale: str = "range",
    conditional: bool = False,
    estimates: bool = False,
    alpha: float = 0.05,
    bound: str = "separable",
) -> WeightedRankResult:
    """Sensitivity analysis for a block design with a weighted rank statistic.

    Tests the null hypothesis of no treatment effect in ``I`` blocks of
    ``J`` matched individuals, allowing the odds of treatment within a
    block to differ by a factor of up to ``gamma`` because of an unobserved
    covariate, and reports the largest p-value that such a bias could
    produce [@rosenbaum2023bahadur].

    Parameters
    ----------
    y : array-like of shape (I, J), or str
        Outcomes with one row per block. With ``data=``, the name of the
        outcome column instead.
    data : DataFrame, optional
        Long-format data with one row per individual. Requires ``treat``
        and ``block``; every block must have the same number of rows.
    treat, block : str, optional
        Columns of ``data`` holding the 0/1 treatment indicator and the
        block identifier.
    treated : int or array-like of shape (I, J), default 1
        For an array ``y``: the number of leading columns that hold treated
        individuals, or a 0/1 array marking them. Blocks may have different
        numbers of treated individuals.
    gamma : float or sequence of float, default 1.0
        Sensitivity parameter(s), each ``>= 1``. ``1`` is a randomization
        test.
    phi : str, tuple, callable or list of these, default "u868"
        Weight given to a block as a function of the rank of its
        dispersion, scaled to ``(0, 1]``. ``"wilcoxon"`` weights blocks
        equally (the stratified Wilcoxon test), ``"quade"`` linearly
        (Quade's test). ``"u868"``, ``"u878"``, ``"u888"`` and ``"u858"``
        are the U-statistic weights ``(m, m1, m2)`` of [@rosenbaum2011new],
        which may also be given as a triple, and ``"mixed"`` is the weight
        of [@rosenbaum2025conditioning] built from ``m = 20``. A callable
        receives the scaled ranks and returns the weights. A list of two or
        more requests the adaptive test: the largest of the standardised
        deviates is referred to their joint normal distribution, so the
        choice among the weights is paid for [@rosenbaum2012testing].
    alternative : {"greater", "less", "two-sided"}, default "greater"
        ``"greater"`` looks for treated outcomes that are larger than
        control outcomes. ``"two-sided"`` doubles the smaller one-sided
        bound.
    scores : sequence of J floats, optional
        Scores that replace the within-block ranks ``1, ..., J``; tied
        outcomes share the score of their lowest rank. ``(1, 2, 5)`` in
        blocks of three, for instance, emphasises the largest outcome.
    block_scale : {"range", "gap"}, default "range"
        Dispersion by which blocks are ranked: the range of the outcomes,
        or the gap between the largest outcome and the mean of the others.
    conditional : bool, default False
        Use only the largest and smallest outcomes of each block. The
        statistic is the weighted share of extreme treated individuals who
        hold the block maximum, among blocks whose extremes include both a
        treated individual and a control [@rosenbaum2025conditioning].
        Conditioning discards blocks but can make the analysis insensitive
        to much larger biases.
    estimates : bool, default False
        Also bound the Hodges-Lehmann estimate and the ``1 - alpha``
        confidence interval for an additive effect, by inverting the test
        after subtracting the hypothesised effect from the treated
        outcomes. The interval is one-sided unless
        ``alternative="two-sided"``.
    alpha : float, default 0.05
        Level for ``gamma_critical``, the confidence interval and the
        Taylor bound.
    bound : {"separable", "taylor"}, default "separable"
        ``"separable"`` maximises the null expectation block by block
        [@gastwirth2000asymptotic]. ``"taylor"`` is the bound of
        [@rosenbaum2018sensitivity], never smaller, which holds at level
        ``alpha`` without appeal to that approximation. The two coincide
        with one treated individual per block.

    Returns
    -------
    WeightedRankResult

    Notes
    -----
    The statistic is the sum over blocks of the block weight times the sum
    of the treated within-block scores. R's ``weightedRank::wgtRank``
    reports the same statistic divided by ``I`` (and its variance divided
    by ``I**2``); deviates and p-values are identical.

    Examples
    --------
    One treated individual (first column) and two controls per block:

    >>> import numpy as np
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> y = rng.normal(size=(200, 3))
    >>> y[:, 0] += 1.0
    >>> res = sp.weighted_rank(y, gamma=[1.0, 2.0, 3.0], phi="u878")
    >>> res.detail["Gamma"].tolist()
    [1.0, 2.0, 3.0]
    >>> bool(res.gamma_critical > 2.0)
    True

    Long-format data and an interval for the effect:

    >>> import pandas as pd
    >>> long = pd.DataFrame({
    ...     "y": y.ravel(),
    ...     "z": np.tile([1, 0, 0], 200),
    ...     "set": np.repeat(np.arange(200), 3),
    ... })
    >>> est = sp.weighted_rank("y", data=long, treat="z", block="set",
    ...                        gamma=1.5, estimates=True)
    >>> bool(est.conf_int[0] < est.estimate[0] <= est.estimate[1])
    True

    References
    ----------
    [@rosenbaum2023bahadur], [@rosenbaum2011new], [@rosenbaum2012testing],
    [@rosenbaum2025conditioning], [@gastwirth2000asymptotic],
    [@rosenbaum2018sensitivity], [@quade1979using]
    """
    caller = "weighted_rank"
    if alternative not in {"greater", "less", "two-sided"}:
        raise MethodIncompatibility(
            "alternative must be 'greater', 'less' or 'two-sided'"
        )
    if block_scale not in {"range", "gap"}:
        raise MethodIncompatibility("block_scale must be 'range' or 'gap'")
    if bound not in {"separable", "taylor"}:
        raise MethodIncompatibility("bound must be 'separable' or 'taylor'")
    if not 0 < alpha < 1:
        raise MethodIncompatibility("alpha must lie strictly between 0 and 1")
    gammas = np.atleast_1d(np.asarray(gamma, dtype=float))
    if gammas.size == 0 or np.any(~np.isfinite(gammas)) or np.any(gammas < 1):
        raise MethodIncompatibility("gamma must be >= 1")

    multiple = isinstance(phi, list)
    phis: List[PhiSpec] = list(phi) if isinstance(phi, list) else [phi]
    if multiple and len(phis) < 2:
        raise MethodIncompatibility("a list of phi needs at least two entries")
    labels = [_phi_label(p) for p in phis]

    y_arr, z_arr = _as_blocks(y, data, treat, block, treated, caller)
    y_arr, z_arr = _drop_uninformative(y_arr, z_arr, caller)
    n_blocks, size = y_arr.shape

    score_arr: Optional[np.ndarray] = None
    if scores is not None:
        score_arr = np.asarray(scores, dtype=float)
        if score_arr.shape != (size,) or np.any(np.diff(score_arr) < 0):
            raise MethodIncompatibility(
                f"scores must be {size} non-decreasing numbers, one per "
                "within-block rank"
            )
    if conditional:
        blockers = {
            "a list of phi": multiple,
            "scores=": scores is not None,
            "block_scale='gap'": block_scale == "gap",
            "estimates=True": estimates,
            "bound='taylor'": bound == "taylor",
        }
        used = [name for name, on in blockers.items() if on]
        if used:
            raise MethodIncompatibility(
                "weighted_rank: conditional=True does not combine with "
                + ", ".join(used)
                + ".",
                recovery_hint="Drop those arguments or use conditional=False.",
                diagnostics={"arguments": used},
                alternative_functions=[],
            )

    kappa = float(stats.norm.isf(alpha))

    def one_sided(less: bool, g: float, yy: np.ndarray = y_arr) -> dict:
        if conditional:
            return _conditional(yy, z_arr, phis[0], less, g)
        return _one_sided(
            yy, z_arr, phis, score_arr, block_scale, less, g, kappa, bound
        )

    def analyse(g: float) -> dict:
        if alternative == "two-sided":
            up, down = one_sided(False, g), one_sided(True, g)
            best = up if up["pvalue"] <= down["pvalue"] else down
            out = dict(best)
            out["pvalue"] = min(1.0, 2.0 * best["pvalue"])
            out["pvalues"] = np.minimum(1.0, 2.0 * best["pvalues"])
            return out
        return one_sided(alternative == "less", g)

    # A lower-tail test reports the deviate on the scale of the statistic,
    # so that it is negative when the treated outcomes are the smaller ones.
    def signed(res: dict) -> np.ndarray:
        if alternative == "less" and not conditional:
            return np.asarray(-res["deviate"])
        return np.asarray(res["deviate"])

    rows = []
    first: Optional[dict] = None
    for g in gammas:
        res = analyse(float(g))
        if first is None:
            first = res
        dev = signed(res)
        if multiple:
            for i, lab in enumerate(labels):
                rows.append(
                    {
                        "Gamma": float(g),
                        "phi": lab,
                        "pvalue": float(res["pvalues"][i]),
                        "deviate": float(dev[i]),
                        "statistic": float(res["statistic"][i]),
                        "expectation": float(res["expectation"][i]),
                        "variance": float(res["variance"][i]),
                        "pvalue_joint": res["pvalue"],
                    }
                )
        else:
            rows.append(
                {
                    "Gamma": float(g),
                    "pvalue": res["pvalue"],
                    "deviate": float(dev[0]),
                    "statistic": float(res["statistic"][0]),
                    "expectation": float(res["expectation"][0]),
                    "variance": float(res["variance"][0]),
                }
            )
    assert first is not None
    if not np.all(first["variance"] > 0):
        raise DataInsufficient(
            "weighted_rank: the statistic has no variance; the outcomes do "
            "not differ within blocks.",
            recovery_hint="Check the outcome column and the block identifier.",
            diagnostics={"n_blocks": int(n_blocks), "block_size": int(size)},
            alternative_functions=[],
        )
    detail = pd.DataFrame(rows)

    gamma_critical = _eng.solve_gamma(lambda g: analyse(g)["pvalue"], alpha)

    estimate = conf_int = None
    if estimates:
        if multiple:
            raise MethodIncompatibility(
                "weighted_rank: estimates=True needs a single phi.",
                recovery_hint="Pass one phi, or drop estimates=True.",
                diagnostics={"phi": labels},
                alternative_functions=[],
            )
        treated_y = y_arr[z_arr == 1]
        control_y = y_arr[z_arr == 0]
        span = float(y_arr.max() - y_arr.min())
        lo = float(treated_y.min() - control_y.max()) - 0.01 * span - 1e-8
        hi = float(treated_y.max() - control_y.min()) + 0.01 * span + 1e-8
        crit = float(stats.norm.isf(alpha / 2 if alternative == "two-sided" else alpha))

        def dev_up(tau: float, g: float) -> float:
            return float(one_sided(False, g, y_arr - tau * z_arr)["deviate"][0])

        def dev_down(tau: float, g: float) -> float:
            return float(one_sided(True, g, y_arr - tau * z_arr)["deviate"][0])

        cols: dict = {k: [] for k in ("hl_lower", "hl_upper", "ci_lower", "ci_upper")}
        for g in gammas:
            g = float(g)
            hl_lo = 0.5 * (
                _last_true(lambda t: dev_up(t, g) > 0, lo, hi)
                + _last_true(lambda t: dev_up(t, g) >= 0, lo, hi)
            )
            hl_hi = 0.5 * (
                _last_true(lambda t: dev_down(t, g) < 0, lo, hi)
                + _last_true(lambda t: dev_down(t, g) <= 0, lo, hi)
            )
            ci_lo, ci_hi = -np.inf, np.inf
            if alternative != "less":
                ci_lo = _last_true(lambda t: dev_up(t, g) > crit, lo, hi)
            if alternative != "greater":
                ci_hi = _last_true(lambda t: dev_down(t, g) <= crit, lo, hi)
            cols["hl_lower"].append(hl_lo)
            cols["hl_upper"].append(hl_hi)
            cols["ci_lower"].append(ci_lo)
            cols["ci_upper"].append(ci_hi)
        for name, values in cols.items():
            detail[name] = values
        estimate = (cols["hl_lower"][0], cols["hl_upper"][0])
        conf_int = (cols["ci_lower"][0], cols["ci_upper"][0])

    diagnostics: dict = {
        "n_treated_per_block": sorted(int(k) for k in np.unique(z_arr.sum(axis=1))),
    }
    if first["corr"] is not None:
        diagnostics["correlation"] = first["corr"].tolist()
    for key in (
        "n_decisive",
        "n_treated_extreme_favourable",
        "n_blocks_tied_extremes",
    ):
        if key in first:
            diagnostics[key] = first[key]

    lead = int(np.argmax(first["deviate"])) if multiple else 0
    return WeightedRankResult(
        pvalue=float(first["pvalue"]),
        gamma=float(gammas[0]),
        gamma_critical=float(gamma_critical),
        statistic=float(first["statistic"][lead]),
        expectation=float(first["expectation"][lead]),
        variance=float(first["variance"][lead]),
        deviate=float(signed(first)[lead]),
        phi="+".join(labels),
        alternative=alternative,
        alpha=alpha,
        n_blocks=int(n_blocks),
        block_size=int(size),
        conditional=bool(conditional),
        bound=bound,
        estimate=estimate,
        conf_int=conf_int,
        detail=detail,
        diagnostics=diagnostics,
    )


@dataclass
class WeightedRankPowerResult(ResultProtocolMixin):
    """Result of :func:`weighted_rank_power`.

    Attributes
    ----------
    power : pandas.DataFrame
        Estimated power of the level-``alpha`` sensitivity analysis at each
        ``Gamma``.
    mean, variance : float
        Jackknife estimates of the mean of the per-block statistic and of
        the variance of that mean in a study of the observed size.

    Examples
    --------
    >>> import numpy as np
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> y = rng.normal(size=(150, 3))
    >>> y[:, 0] += 1.0
    >>> res = sp.weighted_rank_power(y, gammas=[1.0, 2.0, 4.0])
    >>> res.power["power"].is_monotonic_decreasing
    True
    """

    _citation_keys = ("rosenbaum2023bahadur",)

    power: pd.DataFrame
    mean: float
    variance: float
    phi: str
    alpha: float
    sample_ratio: float
    n_blocks: int

    def summary(self) -> str:
        lines = [
            "Estimated power of a weighted rank sensitivity analysis",
            f"phi={self.phi}, alpha={self.alpha}, blocks observed="
            f"{self.n_blocks}, planned={self.sample_ratio * self.n_blocks:.0f}",
            "",
            self.power.to_string(index=False, float_format=lambda v: f"{v:.4f}"),
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"WeightedRankPowerResult(phi={self.phi}, n_gamma={len(self.power)})"


def weighted_rank_power(
    y: Any,
    gammas: Sequence[float],
    *,
    data: Optional[pd.DataFrame] = None,
    treat: Optional[str] = None,
    block: Optional[str] = None,
    treated: Any = 1,
    phi: PhiSpec = "u868",
    sample_ratio: float = 1.0,
    alpha: float = 0.05,
) -> WeightedRankPowerResult:
    """Estimated power of a weighted rank sensitivity analysis.

    Uses the data in hand as a pilot. The sensitivity analysis at ``Gamma``
    rejects when the statistic exceeds its bounding expectation by
    ``z_alpha`` bounding standard deviations; how often that happens depends
    on where the statistic is centred and how variable it is, both of which
    the jackknife over blocks estimates. Power is then read from the normal
    approximation, for a study with ``sample_ratio`` times as many blocks.

    The estimate answers a design question (which weights would have served
    this study best, and how many blocks a similar study needs) and is not
    a substitute for the analysis itself: choosing ``phi`` by its estimated
    power on the same outcomes and then reporting that test ignores the
    selection. Use a list of ``phi`` in :func:`weighted_rank` for a choice
    that is accounted for.

    Parameters
    ----------
    y, data, treat, block, treated
        The block design, as in :func:`weighted_rank`.
    gammas : sequence of float
        Values of the sensitivity parameter.
    phi : str, tuple or callable, default "u868"
        Block weights, as in :func:`weighted_rank`.
    sample_ratio : float, default 1.0
        Number of blocks in the planned study divided by the number
        observed.
    alpha : float, default 0.05
        Level of the one-sided test.

    Returns
    -------
    WeightedRankPowerResult

    Notes
    -----
    With ``sample_ratio = 1`` the result equals R's
    ``weightedRank::estPower``. For other ratios that function divides the
    bounding standard deviation by the ratio where its square root is
    wanted (the variance of a mean over ``s`` times as many blocks is
    ``1/s`` of the original), which overstates power when ``s > 1``; the
    computation here follows the variance.

    Examples
    --------
    >>> import numpy as np
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> y = rng.normal(size=(150, 3))
    >>> y[:, 0] += 1.0
    >>> small = sp.weighted_rank_power(y, gammas=[3.0])
    >>> large = sp.weighted_rank_power(y, gammas=[3.0], sample_ratio=4.0)
    >>> bool(large.power["power"][0] > small.power["power"][0])
    True

    References
    ----------
    [@rosenbaum2023bahadur]
    """
    caller = "weighted_rank_power"
    if not 0 < alpha < 1:
        raise MethodIncompatibility("alpha must lie strictly between 0 and 1")
    if not sample_ratio > 0:
        raise MethodIncompatibility("sample_ratio must be positive")
    gam = np.atleast_1d(np.asarray(gammas, dtype=float))
    if gam.size == 0 or np.any(gam < 1):
        raise MethodIncompatibility("gammas must be >= 1")
    y_arr, z_arr = _as_blocks(y, data, treat, block, treated, caller)
    y_arr, z_arr = _drop_uninformative(y_arr, z_arr, caller)
    n_blocks = y_arr.shape[0]
    if n_blocks < 3:
        raise DataInsufficient(
            "weighted_rank_power: the jackknife needs at least three blocks.",
            recovery_hint="Collect more blocks.",
            diagnostics={"n_blocks": int(n_blocks)},
            alternative_functions=[],
        )

    def per_block(yy: np.ndarray, zz: np.ndarray) -> float:
        d = _design(yy, zz, [phi], None, "range", False)
        return float(d.statistics()[0] / yy.shape[0])

    keep = np.ones(n_blocks, dtype=bool)
    leave_one_out = np.empty(n_blocks)
    for i in range(n_blocks):
        keep[i] = False
        leave_one_out[i] = per_block(y_arr[keep], z_arr[keep])
        keep[i] = True
    centre = float(leave_one_out.mean())
    var_mean = float((n_blocks - 1) / n_blocks * np.sum((leave_one_out - centre) ** 2))

    crit = float(stats.norm.isf(alpha))
    design = _design(y_arr, z_arr, [phi], None, "range", False)
    rows = []
    for g in gam:
        piece = _bounds(design, float(g), crit)[0]
        e_bar = piece.expectation / n_blocks
        sd_bar = np.sqrt(piece.variance) / n_blocks
        threshold = e_bar + crit * sd_bar / np.sqrt(sample_ratio)
        z = (threshold - centre) / np.sqrt(var_mean / sample_ratio)
        rows.append({"Gamma": float(g), "power": float(stats.norm.sf(z))})
    return WeightedRankPowerResult(
        power=pd.DataFrame(rows),
        mean=centre,
        variance=var_mean,
        phi=_phi_label(phi),
        alpha=alpha,
        sample_ratio=float(sample_ratio),
        n_blocks=int(n_blocks),
    )


__all__ = [
    "weighted_rank",
    "weighted_rank_power",
    "WeightedRankResult",
    "WeightedRankPowerResult",
]
