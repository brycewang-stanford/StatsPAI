"""Sensitivity analyses beyond matched pairs, and the tools that go with them.

* :func:`rosenbaum_stratified` — bounds for a treated-versus-control
  comparison within strata of any size and composition, down to a single
  stratum (the two-sample case).
* :func:`noether_test` — a sign test on the pairs with the largest
  absolute differences.
* :func:`evidence_factors` — two nearly independent tests from a design
  with two control groups, and their combination.
* :func:`truncated_product` — combination of independent p-values.
* :func:`amplify` — the two-parameter reading of a one-parameter
  sensitivity analysis.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy import special, stats

from .._input_validation import require_columns
from .._result_serialize import ResultProtocolMixin
from ..exceptions import DataInsufficient, MethodIncompatibility
from . import _sens_engine as _eng


@dataclass
class SensitivityTestResult(ResultProtocolMixin):
    """Result of :func:`rosenbaum_stratified` and :func:`noether_test`.

    Attributes
    ----------
    pvalue : float
        Upper bound on the p-value at the first ``gamma``.
    gamma : float
        The first (or only) ``Gamma`` analysed.
    gamma_critical : float
        The ``Gamma`` at which the bound equals ``alpha``.
    statistic : float
        The test statistic.
    detail : pandas.DataFrame
        One row per ``Gamma``.
    diagnostics : dict
        Counts that describe what entered the test.

    Examples
    --------
    >>> import numpy as np
    >>> import statspai as sp
    >>> rng = np.random.default_rng(1)
    >>> d = rng.normal(0.5, 1.0, 300)
    >>> res = sp.noether_test(d, gamma=2.0)
    >>> type(res).__name__
    'SensitivityTestResult'
    >>> res.diagnostics["n_pairs_used"]
    101
    """

    _citation_keys = ("rosenbaum2018sensitivity",)

    method: str
    pvalue: float
    gamma: float
    gamma_critical: float
    statistic: float
    alternative: str
    alpha: float
    detail: pd.DataFrame = field(default_factory=pd.DataFrame)
    diagnostics: dict = field(default_factory=dict)

    def summary(self) -> str:
        lines = [
            self.method,
            "=" * len(self.method),
            f"Alternative      : {self.alternative}",
            f"Critical Gamma   : {self.gamma_critical:.4f}  (alpha={self.alpha})",
        ]
        lines += [f"{k:<17}: {v}" for k, v in self.diagnostics.items()]
        lines += [
            "",
            self.detail.to_string(index=False, float_format=lambda v: f"{v:.6g}"),
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"SensitivityTestResult({self.method!r}, gamma={self.gamma:g}, "
            f"pvalue={self.pvalue:.4g}, gamma_critical={self.gamma_critical:.3f})"
        )


def _check_common(alternative: str, alpha: float, gamma: Any) -> np.ndarray:
    if alternative not in {"greater", "less", "two-sided"}:
        raise MethodIncompatibility(
            "alternative must be 'greater', 'less' or 'two-sided'"
        )
    if not 0 < alpha < 1:
        raise MethodIncompatibility("alpha must lie strictly between 0 and 1")
    gammas = np.atleast_1d(np.asarray(gamma, dtype=float))
    if gammas.size == 0 or np.any(~np.isfinite(gammas)) or np.any(gammas < 1):
        raise MethodIncompatibility("gamma must be >= 1")
    return gammas


# --------------------------------------------------------------------
# Stratified comparisons
# --------------------------------------------------------------------


def _scores_for(y: np.ndarray, codes: np.ndarray, score: str) -> np.ndarray:
    if score == "raw":
        return y
    if score == "rank":
        return np.asarray(stats.rankdata(y, method="average"))
    frame = pd.DataFrame({"y": y, "s": codes})
    if score == "stratum_rank":
        return np.asarray(frame.groupby("s")["y"].rank(method="average"))
    if score == "aligned_rank":
        centred = y - frame.groupby("s")["y"].transform("mean").to_numpy()
        return np.asarray(stats.rankdata(centred, method="average"))
    raise MethodIncompatibility(
        "score must be 'rank', 'stratum_rank', 'aligned_rank' or 'raw'"
    )


def rosenbaum_stratified(
    data: pd.DataFrame,
    y: str,
    treat: str,
    strata: Optional[str] = None,
    *,
    gamma: Union[float, Sequence[float]] = 1.0,
    score: str = "rank",
    alternative: str = "greater",
    alpha: float = 0.05,
    bound: str = "taylor",
) -> SensitivityTestResult:
    """Rosenbaum bounds for a comparison within strata of any size.

    Each stratum holds some treated individuals and some controls, in any
    numbers; the statistic is the sum of the treated individuals' scores.
    Under the null hypothesis of no effect, and allowing two individuals in
    the same stratum to differ in their odds of treatment by a factor of at
    most ``gamma``, the function bounds the one-sided p-value
    [@rosenbaum2018sensitivity]. Matched pairs and matched sets are special
    cases; with ``strata=None`` it is the two-sample analysis of
    [@rosenbaum1990sensitivity].

    Parameters
    ----------
    data : DataFrame
        One row per individual.
    y : str
        Outcome column (or, with ``score="raw"``, a column of scores).
    treat : str
        0/1 treatment indicator.
    strata : str, optional
        Stratum identifier. ``None`` treats the sample as one stratum.
    gamma : float or sequence of float, default 1.0
        Sensitivity parameter(s), ``>= 1``.
    score : {"rank", "stratum_rank", "aligned_rank", "raw"}, default "rank"
        ``"rank"`` ranks the outcomes over the whole sample (the Wilcoxon
        rank sum in a single stratum). ``"stratum_rank"`` ranks within
        strata. ``"aligned_rank"`` subtracts the stratum mean and then
        ranks over the whole sample, the aligned ranks of Hodges and
        Lehmann. ``"raw"`` uses ``y`` itself, for the permutational
        t-test or for scores computed elsewhere.
    alternative : {"greater", "less", "two-sided"}, default "greater"
    alpha : float, default 0.05
        Level for ``gamma_critical`` and for the Taylor bound.
    bound : {"taylor", "separable"}, default "taylor"
        ``"separable"`` takes, stratum by stratum, the hidden bias that
        maximises the null expectation; it is exact as the number of strata
        grows but may be slightly anti-conservative with a few large
        strata. ``"taylor"`` is a bound that holds at level ``alpha``
        whatever the strata look like. Both are in ``detail``. With a
        single stratum neither is needed: the candidate biases are searched
        for the largest p-value directly.

    Returns
    -------
    SensitivityTestResult

    Notes
    -----
    The moments of the statistic are computed exactly from Fisher's
    noncentral hypergeometric distribution. R's ``senstrat`` obtains them
    from the ``BiasedUrn`` package at its default precision of ``1e-7``
    (``method="BU"``), or from a large-sample approximation
    (``method="LS"``); its ``method="RK"`` is exact and is the one this
    function agrees with to rounding.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> s = rng.integers(0, 6, 300)
    >>> z = rng.binomial(1, 0.3 + 0.05 * s)
    >>> yv = 0.6 * z + 0.3 * s + rng.normal(size=300)
    >>> df = pd.DataFrame({"y": yv, "z": z, "s": s})
    >>> res = sp.rosenbaum_stratified(df, "y", "z", "s", gamma=[1.0, 1.5],
    ...                               score="aligned_rank")
    >>> res.diagnostics["n_strata"]
    6
    >>> bool(res.detail["pvalue"].is_monotonic_increasing)
    True

    References
    ----------
    [@rosenbaum2018sensitivity], [@rosenbaum1990sensitivity],
    [@gastwirth2000asymptotic]
    """
    gammas = _check_common(alternative, alpha, gamma)
    if bound not in {"taylor", "separable"}:
        raise MethodIncompatibility("bound must be 'taylor' or 'separable'")
    cols = [y, treat] + ([strata] if strata is not None else [])
    require_columns(data, cols, function="rosenbaum_stratified")
    d = data[cols].dropna()
    z = d[treat].to_numpy(dtype=float)
    if not np.all((z == 0) | (z == 1)):
        raise MethodIncompatibility(
            "rosenbaum_stratified: the treat column must be coded 0/1"
        )
    if strata is None:
        codes = np.zeros(len(d), dtype=int)
    else:
        codes = pd.factorize(d[strata], sort=True)[0]
    yv = d[y].to_numpy(dtype=float)
    sc = _scores_for(yv, codes, score)
    kappa = float(stats.norm.isf(alpha))

    def side(sign: float) -> Tuple[float, Any, Tuple[int, int, int]]:
        stacked, statistic, n_strata, n_t, n_c = _eng.group_strata(sign * sc, z, codes)
        if n_strata == 0:
            raise DataInsufficient(
                "rosenbaum_stratified: no stratum contains both a treated "
                "individual and a control.",
                recovery_hint="Coarsen the strata.",
                diagnostics={"n_rows": int(len(d))},
                alternative_functions=[],
            )

        def at(g: float) -> _eng.BoundPieces:
            groups = []
            for (size, n_treated), rows in stacked.items():
                mu, nu = _eng.moment_tables(rows, n_treated, g)
                groups.append(_eng._Group(np.ones(len(rows)), mu, nu))
            if n_strata == 1:
                # One stratum: the candidates can be searched directly for
                # the smallest deviate, which is the bound itself rather
                # than an approximation to it.
                mu, nu = groups[0].mu[0], groups[0].nu[0]
                with np.errstate(divide="ignore", invalid="ignore"):
                    dev = np.where(nu > 0, (statistic - mu) / np.sqrt(nu), np.inf)
                m = int(np.argmin(dev))
                return _eng.BoundPieces(mu[m], nu[m], mu[m], nu[m])
            return _eng.combine(groups, kappa)

        return statistic, at, (n_strata, n_t, n_c)

    sides = {"greater": [1.0], "less": [-1.0], "two-sided": [1.0, -1.0]}[alternative]
    built = [(sgn, *side(sgn)) for sgn in sides]
    n_strata, n_t, n_c = built[0][3]
    factor = 2.0 if alternative == "two-sided" else 1.0

    def row_at(g: float) -> dict:
        best = None
        for sgn, statistic, at, _ in built:
            piece = at(g)
            p_sep = _eng.upper_pvalue(piece.deviate(statistic))
            p_tay = _eng.upper_pvalue(piece.deviate(statistic, taylor=True))
            p_use = p_tay if bound == "taylor" else p_sep
            if best is None or p_use < best["_p"]:
                taylor = bound == "taylor"
                best = {
                    "_p": p_use,
                    "Gamma": g,
                    "pvalue": min(1.0, factor * p_use),
                    "deviate": sgn * piece.deviate(statistic, taylor),
                    "statistic": sgn * statistic,
                    "expectation": sgn
                    * (piece.expectation_taylor if taylor else piece.expectation),
                    "variance": piece.variance_taylor if taylor else piece.variance,
                    "pvalue_separable": min(1.0, factor * p_sep),
                    "pvalue_taylor": min(1.0, factor * p_tay),
                }
        assert best is not None
        best.pop("_p")
        return best

    detail = pd.DataFrame([row_at(float(g)) for g in gammas])
    gamma_critical = _eng.solve_gamma(lambda g: row_at(g)["pvalue"], alpha)
    dropped = int(pd.Series(codes).nunique()) - n_strata
    return SensitivityTestResult(
        method="Sensitivity analysis for a stratified comparison",
        pvalue=float(detail["pvalue"].iloc[0]),
        gamma=float(gammas[0]),
        gamma_critical=float(gamma_critical),
        statistic=float(detail["statistic"].iloc[0]),
        alternative=alternative,
        alpha=alpha,
        detail=detail,
        diagnostics={
            "score": score,
            "bound": bound,
            "n_strata": n_strata,
            "n_treated": n_t,
            "n_control": n_c,
            "n_strata_dropped": dropped,
        },
    )


# --------------------------------------------------------------------
# Noether's test
# --------------------------------------------------------------------


def noether_test(
    treated: Sequence[float],
    control: Optional[Sequence[float]] = None,
    *,
    f: float = 2.0 / 3.0,
    gamma: Union[float, Sequence[float]] = 1.0,
    alternative: str = "greater",
    alpha: float = 0.05,
) -> SensitivityTestResult:
    """Sign test on the matched pairs with the largest absolute differences.

    Discards the fraction ``f`` of pairs whose absolute differences are
    smallest and applies the sign test to the rest
    [@noether1973some]. Small differences are nearly as likely to be
    positive as negative whether or not there is an effect, so they add
    noise that a small bias can exploit; dropping them makes the test less
    sensitive to unmeasured bias when the effect is not tiny
    [@rosenbaum2012exact]. The bound on the p-value is exact: under a bias
    of at most ``gamma`` the number of positive differences is at most
    binomial with success probability ``gamma / (1 + gamma)``.

    Parameters
    ----------
    treated : array-like
        Treated outcome of each pair, or the treated-minus-control
        differences when ``control`` is omitted.
    control : array-like, optional
        Control outcome of each pair.
    f : float, default 2/3
        Fraction of pairs set aside, ``0 <= f < 1``. Pairs are kept when
        the rank of their absolute difference, divided by the number of
        pairs, is at least ``f``. ``0`` is the ordinary sign test.
    gamma : float or sequence of float, default 1.0
        Sensitivity parameter(s), ``>= 1``.
    alternative : {"greater", "less", "two-sided"}, default "greater"
        ``"two-sided"`` doubles the smaller one-sided bound.
    alpha : float, default 0.05
        Level for ``gamma_critical``.

    Returns
    -------
    SensitivityTestResult
        ``statistic`` is the number of positive differences among the pairs
        used; ``diagnostics["n_pairs_used"]`` is how many those are.

    Examples
    --------
    >>> import numpy as np
    >>> import statspai as sp
    >>> rng = np.random.default_rng(1)
    >>> d = rng.normal(0.5, 1.0, 300)
    >>> all_pairs = sp.noether_test(d, f=0, gamma=2.0)
    >>> top_third = sp.noether_test(d, gamma=2.0)
    >>> bool(top_third.gamma_critical > all_pairs.gamma_critical)
    True

    References
    ----------
    [@noether1973some], [@rosenbaum2012exact]
    """
    gammas = _check_common(alternative, alpha, gamma)
    if not 0 <= f < 1:
        raise MethodIncompatibility("f must satisfy 0 <= f < 1")
    d = np.asarray(treated, dtype=float)
    if control is not None:
        c = np.asarray(control, dtype=float)
        if c.shape != d.shape:
            raise MethodIncompatibility("treated and control must have the same length")
        d = d - c
    d = d[np.isfinite(d)]
    if d.ndim != 1 or d.size < 2:
        raise MethodIncompatibility("noether_test needs at least two pairs")
    ad = np.abs(d)
    use = (ad > 0) & (stats.rankdata(ad, method="average") / d.size >= f)
    n_used = int(use.sum())
    n_pos = int(np.sum(d[use] > 0))
    if n_used == 0:
        raise DataInsufficient(
            "noether_test: every pair difference is zero.",
            recovery_hint="The sign test has nothing to count.",
            diagnostics={"n_pairs": int(d.size)},
            alternative_functions=[],
        )

    def bound(g: float) -> float:
        p_hi = g / (1.0 + g)
        up = float(stats.binom.sf(n_pos - 1, n_used, p_hi))
        down = float(stats.binom.cdf(n_pos, n_used, 1.0 - p_hi))
        if alternative == "greater":
            return up
        if alternative == "less":
            return down
        return min(1.0, 2.0 * min(up, down))

    detail = pd.DataFrame(
        [{"Gamma": float(g), "pvalue": bound(float(g))} for g in gammas]
    )
    return SensitivityTestResult(
        method="Noether's sign test on the largest absolute differences",
        pvalue=float(detail["pvalue"].iloc[0]),
        gamma=float(gammas[0]),
        gamma_critical=float(_eng.solve_gamma(bound, alpha)),
        statistic=float(n_pos),
        alternative=alternative,
        alpha=alpha,
        detail=detail,
        diagnostics={
            "n_pairs": int(d.size),
            "n_pairs_used": n_used,
            "n_positive": n_pos,
            "f": float(f),
        },
    )


# --------------------------------------------------------------------
# Combining p-values, evidence factors, amplification
# --------------------------------------------------------------------


def truncated_product(pvalues: Sequence[float], trunc: float = 0.2) -> float:
    """Combine independent p-values by the truncated product method.

    The statistic is the product of the p-values that are at most
    ``trunc``; its null distribution is known in closed form
    [@zaykin2002truncated]. ``trunc=1`` is Fisher's method. Truncation
    keeps one large p-value from undoing several small ones, which matters
    when the p-values are upper bounds from a sensitivity analysis: such
    bounds tend to one as the allowed bias grows [@hsu2013effect].

    Parameters
    ----------
    pvalues : sequence of float
        Independent p-values (or p-values whose joint distribution is
        stochastically larger than uniform on the unit cube, as with
        evidence factors).
    trunc : float, default 0.2
        Truncation point, ``0 < trunc <= 1``.

    Returns
    -------
    float
        The combined p-value; ``1.0`` when no p-value is at most ``trunc``.

    Examples
    --------
    >>> import statspai as sp
    >>> round(sp.truncated_product([0.01, 0.3]), 4)
    0.0399
    >>> round(sp.truncated_product([0.01, 0.3], trunc=1.0), 4)  # Fisher
    0.0204
    >>> sp.truncated_product([0.5, 0.3])
    1.0

    References
    ----------
    [@zaykin2002truncated], [@hsu2013effect]
    """
    p = np.asarray(pvalues, dtype=float).ravel()
    if p.size == 0 or np.any(~np.isfinite(p)) or np.any((p < 0) | (p > 1)):
        raise MethodIncompatibility("pvalues must be numbers between 0 and 1")
    if not 0 < trunc <= 1:
        raise MethodIncompatibility("trunc must satisfy 0 < trunc <= 1")
    small = p[p <= trunc]
    if small.size == 0:
        return 1.0
    if np.any(small == 0):
        return 0.0
    n = p.size
    log_w = float(np.sum(np.log(small)))
    log_t = float(np.log(trunc))
    total = 0.0
    for k in range(1, n + 1):
        weight = special.comb(n, k) * (1.0 - trunc) ** (n - k)
        if weight == 0.0:
            continue
        if log_w <= k * log_t:
            # w * sum_{s<k} (k log(trunc) - log w)^s / s!, with the
            # terms formed in logs
            a = k * log_t - log_w
            terms = np.arange(k)
            inner = float(
                np.sum(np.exp(terms * np.log(a) - special.gammaln(terms + 1)))
                if a > 0
                else 1.0
            )
            total += weight * np.exp(log_w) * inner
        else:
            total += weight * trunc**k
    return float(min(1.0, total))


@dataclass
class EvidenceFactorsResult(ResultProtocolMixin):
    """Result of :func:`evidence_factors`.

    Attributes
    ----------
    pvalue : float
        Truncated product of the two factor p-value bounds.
    pvalue_treated_vs_control1, pvalue_control2_vs_others : float
        The two factors.
    detail : pandas.DataFrame
        Statistic, bounding moments and deviate of each factor.

    Examples
    --------
    >>> import numpy as np
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> y = rng.normal(size=(300, 3))
    >>> y[:, 0] += 0.8
    >>> res = sp.evidence_factors(y, gamma=1.5, upsilon=1.5)
    >>> type(res).__name__
    'EvidenceFactorsResult'
    >>> bool(res.pvalue < 0.05)
    True
    """

    _citation_keys = ("rosenbaum2023second",)

    pvalue: float
    pvalue_treated_vs_control1: float
    pvalue_control2_vs_others: float
    gamma: float
    upsilon: float
    trunc: float
    alternative: str
    n_blocks: int
    detail: pd.DataFrame = field(default_factory=pd.DataFrame)

    def summary(self) -> str:
        lines = [
            "Evidence factors with two control groups",
            "========================================",
            f"Blocks                         : {self.n_blocks}",
            f"Treated vs control 1 (Gamma={self.gamma:g})   : "
            f"p <= {self.pvalue_treated_vs_control1:.6g}",
            f"Control 2 vs others (Upsilon={self.upsilon:g}) : "
            f"p <= {self.pvalue_control2_vs_others:.6g}",
            f"Combined (truncated product at {self.trunc:g}) : "
            f"p <= {self.pvalue:.6g}",
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"EvidenceFactorsResult(pvalue={self.pvalue:.4g}, "
            f"gamma={self.gamma:g}, upsilon={self.upsilon:g})"
        )


def evidence_factors(
    y: Any,
    *,
    gamma: float = 1.0,
    upsilon: float = 1.0,
    alternative: str = "greater",
    trunc: float = 0.2,
    phi: Tuple[Any, Any] = ((8, 7, 8), (8, 8, 8)),
    scores: Sequence[float] = (1.0, 2.0, 5.0),
    block_scale: str = "gap",
) -> EvidenceFactorsResult:
    """Two evidence factors from blocks with a treated unit and two controls.

    Each block holds one treated individual, one control from a first
    control group and one from a second. The first factor compares the
    treated individual with the first control, as a matched pair. The
    second compares the second control with the other two pooled. When
    the treatment has no effect the two p-values behave as if they came
    from unrelated studies, although they share data, and the biases that
    could explain away one comparison differ from those that could explain
    away the other [@rosenbaum2023second]. The factors are combined by the
    truncated product.

    Parameters
    ----------
    y : array-like of shape (I, 3)
        Outcomes; columns are treated, first control, second control.
    gamma : float, default 1.0
        Bias allowed in the treated-versus-first-control comparison.
    upsilon : float, default 1.0
        Bias allowed in the comparison of the second control with the
        other two.
    alternative : {"greater", "less"}, default "greater"
        Direction of the effect of treatment on the outcome.
    trunc : float, default 0.2
        Truncation point of :func:`truncated_product`.
    phi : pair, default ((8, 7, 8), (8, 8, 8))
        Block weights of the two factors, as in :func:`weighted_rank`.
    scores : sequence of 3 floats, default (1, 2, 5)
        Within-block scores of the second factor. Unequal steps guard
        against the dilution that pooling the treated individual with an
        unaffected control would otherwise cause.
    block_scale : {"gap", "range"}, default "gap"
        Block dispersion used by the second factor; the first, on pairs,
        always uses the range.

    Returns
    -------
    EvidenceFactorsResult

    Examples
    --------
    >>> import numpy as np
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> y = rng.normal(size=(300, 3))
    >>> y[:, 0] += 0.8
    >>> res = sp.evidence_factors(y, gamma=1.5, upsilon=1.5)
    >>> bool(res.pvalue <= max(res.pvalue_treated_vs_control1,
    ...                        res.pvalue_control2_vs_others))
    True

    References
    ----------
    [@rosenbaum2023second], [@rosenbaum2011some], [@zaykin2002truncated]
    """
    from .weighted_rank import weighted_rank

    if alternative not in {"greater", "less"}:
        raise MethodIncompatibility("alternative must be 'greater' or 'less'")
    arr = np.asarray(y, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise MethodIncompatibility(
            "evidence_factors: y must have three columns "
            "(treated, first control, second control)"
        )
    if len(phi) != 2:
        raise MethodIncompatibility("phi must hold one weight specification per factor")
    if alternative == "less":
        arr = -arr
    first = weighted_rank(arr[:, :2], gamma=gamma, phi=phi[0])
    second = weighted_rank(
        arr[:, ::-1],
        gamma=upsilon,
        phi=phi[1],
        alternative="less",
        scores=scores,
        block_scale=block_scale,
    )
    rows = []
    for name, res in (
        ("treated_vs_control1", first),
        ("control2_vs_others", second),
    ):
        rows.append(
            {
                "factor": name,
                "bias": res.gamma,
                "pvalue": res.pvalue,
                "deviate": res.deviate,
                "statistic": res.statistic,
                "expectation": res.expectation,
                "variance": res.variance,
            }
        )
    return EvidenceFactorsResult(
        pvalue=truncated_product([first.pvalue, second.pvalue], trunc=trunc),
        pvalue_treated_vs_control1=first.pvalue,
        pvalue_control2_vs_others=second.pvalue,
        gamma=float(gamma),
        upsilon=float(upsilon),
        trunc=float(trunc),
        alternative=alternative,
        n_blocks=int(arr.shape[0]),
        detail=pd.DataFrame(rows),
    )


def amplify(
    gamma: float, treatment_odds: Union[float, Sequence[float]]
) -> Union[float, np.ndarray]:
    """Rewrite a sensitivity parameter as two odds ratios.

    A sensitivity analysis for matched pairs speaks of one number,
    ``Gamma``: an unobserved covariate that could change the odds of
    treatment by that factor and that predicts the outcome perfectly. The
    same bound applies to every unobserved covariate that multiplies the
    odds of treatment by ``Lambda`` and the odds of a positive pair
    difference in outcomes by ``Delta``, provided
    ``Gamma = (Lambda * Delta + 1) / (Lambda + Delta)``
    [@rosenbaum2009amplification]. This function returns the ``Delta`` that
    goes with each ``Lambda``.

    Parameters
    ----------
    gamma : float
        Sensitivity parameter, ``> 1``.
    treatment_odds : float or sequence of float
        ``Lambda``, each ``> gamma``.

    Returns
    -------
    float or numpy.ndarray
        ``Delta = (gamma * Lambda - 1) / (Lambda - gamma)``.

    Examples
    --------
    >>> import statspai as sp
    >>> sp.amplify(4, 7)
    9.0
    >>> sp.amplify(1.5, [2, 3, 4]).round(3).tolist()
    [4.0, 2.333, 2.0]

    References
    ----------
    [@rosenbaum2009amplification]
    """
    if not np.isscalar(gamma) or not gamma > 1:
        raise MethodIncompatibility("gamma must be a single number greater than 1")
    lam = np.asarray(treatment_odds, dtype=float)
    if np.any(~(lam > gamma)):
        raise MethodIncompatibility("every value of treatment_odds must exceed gamma")
    delta = (gamma * lam - 1.0) / (lam - gamma)
    return float(delta) if delta.ndim == 0 else delta


__all__ = [
    "rosenbaum_stratified",
    "noether_test",
    "truncated_product",
    "evidence_factors",
    "amplify",
    "SensitivityTestResult",
    "EvidenceFactorsResult",
]
