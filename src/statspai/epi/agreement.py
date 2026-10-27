"""Krippendorff's alpha for inter-coder reliability.

Krippendorff's alpha measures agreement among any number of coders on
any number of units, allows missing codings, and handles nominal,
ordinal, interval and ratio data through one difference function
[@krippendorff1970estimating; @hayes2007answering]:

    alpha = 1 - D_o / D_e

where ``D_o`` is the disagreement observed within units and ``D_e`` the
disagreement expected when the same values are paired at random.  Both
are computed from the coincidence matrix ``o_ck``: every unit with
``m_u >= 2`` codings contributes each ordered pair of its values with
weight ``1 / (m_u - 1)``.  With ``n_c = sum_k o_ck`` and
``n = sum_c n_c`` (the number of pairable values),

    alpha = 1 - (n - 1) * sum_ck o_ck d2_ck / sum_ck n_c n_k d2_ck.

Difference functions ``d2_ck``:

* nominal: 0 when ``c == k``, else 1;
* ordinal: ``(sum_{g=c..k} n_g - (n_c + n_k) / 2) ** 2`` over the ranked
  distinct values;
* interval: ``(c - k) ** 2``;
* ratio: ``((c - k) / (c + k)) ** 2``.

The bootstrap follows the algorithm used by Stata ``kalpha`` (it draws
``M`` pairs of values from the coincidence matrix, where
``M = min(25 x number of non-zero cells, n (m - 1) / 2)``, ``m`` being the
number of coders, and applies Krippendorff's corrections at -1 and 1).
A unit-resampling bootstrap is offered as the alternative.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .._result_serialize import ResultProtocolMixin
from ..exceptions import DataInsufficient, MethodIncompatibility

__all__ = ["krippendorff_alpha", "KrippendorffAlphaResult"]

_LEVELS = ("nominal", "ordinal", "interval", "ratio")
_CI_METHODS = ("krippendorff", "units")


@dataclass
class KrippendorffAlphaResult(ResultProtocolMixin):
    """Output of :func:`krippendorff_alpha`.

    Attributes
    ----------
    estimate : float
        Krippendorff's alpha.
    level : str
        Level of measurement (``'nominal'``, ``'ordinal'``, ``'interval'``
        or ``'ratio'``).
    observed_disagreement, expected_disagreement : float
        ``D_o`` and ``D_e`` per pairable value (``alpha = 1 - D_o / D_e``).
    n_units : int
        Units with at least two codings (pairable units).
    n_coders : int
        Coders (columns of the units-by-coders matrix).
    n_pairable : int
        Pairable values ``n``.
    values : list
        Distinct values, in the order of the rows of :attr:`coincidence`.
    coincidence : pandas.DataFrame
        Coincidence matrix ``o_ck``.
    ci : tuple of float or None
        Bootstrap confidence interval (``None`` without bootstrap).
    prob_below : dict
        With bootstrap: ``{min_alpha: P(alpha* < min_alpha)}``, the
        probability of failing to reach each minimum.
    boot_dist : numpy.ndarray or None
        Sorted bootstrap draws.

    Examples
    --------
    >>> import statspai as sp
    >>> res = sp.krippendorff_alpha([[1, 1], [2, 2], [3, 3], [1, 2]])
    >>> isinstance(res, sp.KrippendorffAlphaResult)
    True
    >>> round(res.estimate, 4)
    0.6667
    """

    _citation_keys: ClassVar[Tuple[str, ...]] = (
        "krippendorff1970estimating",
        "hayes2007answering",
    )

    estimate: float
    level: str
    observed_disagreement: float
    expected_disagreement: float
    n_units: int
    n_coders: int
    n_pairable: int
    values: List[Any]
    coincidence: pd.DataFrame
    ci: Optional[Tuple[float, float]] = None
    alpha: float = 0.05
    ci_method: Optional[str] = None
    n_boot: int = 0
    seed: Optional[int] = None
    prob_below: Dict[float, float] = field(default_factory=dict)
    boot_dist: Optional[np.ndarray] = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    @property
    def kalpha(self) -> float:
        """Alias of :attr:`estimate`."""
        return self.estimate

    def interpretation(self) -> str:
        """Krippendorff's rule of thumb (0.800 reliable, 0.667 tentative)."""
        a = self.estimate
        if a >= 0.8:
            return "reliable (alpha >= 0.800)"
        if a >= 0.667:
            return "tentative conclusions only (0.667 <= alpha < 0.800)"
        return "unreliable (alpha < 0.667)"

    def to_frame(self) -> pd.DataFrame:
        lo, hi = self.ci if self.ci is not None else (np.nan, np.nan)
        return pd.DataFrame(
            [
                {
                    "statistic": "krippendorff_alpha",
                    "level": self.level,
                    "estimate": self.estimate,
                    "ci_lower": lo,
                    "ci_upper": hi,
                    "n_units": self.n_units,
                    "n_coders": self.n_coders,
                    "n_pairable": self.n_pairable,
                }
            ]
        )

    def summary(self) -> str:
        lines = [
            f"Krippendorff's alpha ({self.level})",
            f"  alpha = {self.estimate:.6f}",
            f"  D_o = {self.observed_disagreement:.6g}   "
            f"D_e = {self.expected_disagreement:.6g}",
            f"  units (pairable) = {self.n_units}   coders = {self.n_coders}   "
            f"pairable values = {self.n_pairable}",
        ]
        if self.ci is not None:
            lines.append(
                f"  {100 * (1 - self.alpha):g}% CI [{self.ci[0]:.4f}, {self.ci[1]:.4f}]"
                f"   ({self.ci_method} bootstrap, {self.n_boot} reps, seed={self.seed})"
            )
            for m, q in self.prob_below.items():
                lines.append(f"  P(alpha < {m:g}) = {q:.4f}")
        lines.append(f"  Interpretation: {self.interpretation()}")
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"KrippendorffAlphaResult(level={self.level!r}, "
            f"estimate={self.estimate:.6f}, n_units={self.n_units}, "
            f"n_coders={self.n_coders})"
        )


# --------------------------------------------------------------------------- #
#  Core
# --------------------------------------------------------------------------- #


def _to_matrix(data: Any, layout: str) -> Tuple[np.ndarray, List[str]]:
    """Return an object array (units x coders) with None for missing."""
    if layout not in ("units_by_coders", "coders_by_units"):
        raise MethodIncompatibility(
            f"layout must be 'units_by_coders' or 'coders_by_units', got {layout!r}.",
            recovery_hint="Rows are units by default; use 'coders_by_units' for "
            "the irr::kripp.alpha / Stata kalpha (no transpose) orientation.",
        )
    if isinstance(data, pd.DataFrame):
        frame = data
    else:
        arr = np.asarray(data, dtype=object)
        if arr.ndim != 2:
            raise MethodIncompatibility(
                "data must be a 2-D table (units x coders).",
                recovery_hint="Pass a DataFrame or a list of rows.",
                diagnostics={"ndim": int(arr.ndim)},
            )
        frame = pd.DataFrame(arr)
    if layout == "coders_by_units":
        frame = frame.T
    coders = [str(c) for c in frame.columns]
    mat = frame.to_numpy(dtype=object)
    miss = pd.isna(frame).to_numpy()
    mat = mat.copy()
    mat[miss] = None
    return mat, coders


def _delta2(level: str, vals: np.ndarray, n_c: np.ndarray) -> np.ndarray:
    V = len(vals)
    if level == "nominal":
        return 1.0 - np.eye(V)
    if level == "ordinal":
        cum = np.concatenate([[0.0], np.cumsum(n_c)])
        i = np.arange(V)
        lo = np.minimum(i[:, None], i[None, :])
        hi = np.maximum(i[:, None], i[None, :])
        tot = cum[hi + 1] - cum[lo]
        d = (tot - (n_c[:, None] + n_c[None, :]) / 2.0) ** 2
        np.fill_diagonal(d, 0.0)
        return d
    v = vals.astype(float)
    diff = v[:, None] - v[None, :]
    if level == "interval":
        return diff**2
    ssum = v[:, None] + v[None, :]
    with np.errstate(invalid="ignore", divide="ignore"):
        d = np.where(ssum == 0, 0.0, (diff / np.where(ssum == 0, 1.0, ssum)) ** 2)
    return d


def _alpha_from_counts(
    counts: np.ndarray, level: str, vals: np.ndarray
) -> Dict[str, Any]:
    """Alpha from a units x values count matrix (pairable units only)."""
    m_u = counts.sum(axis=1)
    inv = 1.0 / (m_u - 1.0)
    coin = (counts * inv[:, None]).T @ counts
    coin[np.diag_indices_from(coin)] -= (counts * inv[:, None]).sum(axis=0)
    n_c = coin.sum(axis=1)
    n = float(n_c.sum())
    d2 = _delta2(level, vals, n_c)
    do_sum = float(np.sum(coin * d2))
    de_sum = float(n_c @ d2 @ n_c)
    return {
        "O": coin,
        "n_c": n_c,
        "n": n,
        "d2": d2,
        "D_o": do_sum / n,
        "D_e": de_sum / (n * (n - 1.0)),
        "alpha": (1.0 - (n - 1.0) * do_sum / de_sum) if de_sum > 0 else np.nan,
    }


def _krippendorff_boot(
    core: Dict[str, Any],
    n_coders: int,
    n_boot: int,
    rng: np.random.Generator,
    draws: Optional[int],
) -> Tuple[np.ndarray, int]:
    O, d2, n = core["O"], core["d2"], core["n"]
    V = O.shape[0]
    tri = np.tril_indices(V)
    P = np.where(tri[0] == tri[1], O[tri], 2.0 * O[tri]) / n
    P = np.clip(P, 0.0, None)
    P = P / P.sum()
    De = core["D_e"]  # expected disagreement per pair of values
    if draws is None:
        draws = int(min(25 * int(np.sum(O > 0)), round(n * (n_coders - 1) / 2.0)))
    draws = max(int(draws), 1)
    X = rng.multinomial(draws, P, size=n_boot)
    boot = 1.0 - (X @ d2[tri]) / (draws * De)
    boot = np.where(boot < -1.0, -1.0, boot)
    # Krippendorff's correction for draws that fall on the diagonal only
    ones = boot == 1.0
    if ones.any():
        diag_pos = int(np.sum(np.diag(O) > 0))
        if diag_pos == 1:
            boot[ones] = 0.0
        elif diag_pos > 1:
            pdiag = np.diag(O) / n
            nx = int(round(n_boot * float(np.sum(pdiag**draws))))
            if nx >= int(ones.sum()):
                boot[ones] = 0.0
            else:
                boot = np.sort(boot)
                if nx > 0:
                    boot[-nx:] = 0.0
    return np.sort(boot), draws


def krippendorff_alpha(
    data: Any,
    *,
    level: str = "nominal",
    layout: str = "units_by_coders",
    bootstrap: bool = False,
    n_boot: int = 20000,
    ci_method: str = "krippendorff",
    alpha: float = 0.05,
    min_alpha: Sequence[float] = (0.667, 0.8),
    seed: Optional[int] = None,
    draws: Optional[int] = None,
) -> KrippendorffAlphaResult:
    """Krippendorff's alpha reliability coefficient.

    Parameters
    ----------
    data : DataFrame or 2-D array-like
        Codings, one row per unit and one column per coder (the default
        ``layout``).  Missing codings are NaN / None.  Units with fewer
        than two codings are not pairable and are ignored.  Nominal
        values may be strings; the other levels need numbers.
    level : {'nominal', 'ordinal', 'interval', 'ratio'}, default 'nominal'
        Level of measurement, which selects the difference function.
        Ordinal ranks the distinct observed values in numeric order;
        ratio needs non-negative values.
    layout : {'units_by_coders', 'coders_by_units'}, default 'units_by_coders'
        Orientation of ``data``.  ``'coders_by_units'`` is the layout of R
        ``irr::kripp.alpha`` and of Stata ``kalpha`` without ``transpose``.
    bootstrap : bool, default False
        Add a bootstrap confidence interval and the probabilities of
        failing to reach ``min_alpha``.
    n_boot : int, default 20000
        Bootstrap replications (Stata ``kalpha``'s default).
    ci_method : {'krippendorff', 'units'}, default 'krippendorff'
        ``'krippendorff'`` draws pairs of values from the coincidence
        matrix (the algorithm of Stata ``kalpha``, after Hayes and
        Krippendorff [@hayes2007answering]); the interval is formed by
        the same order statistics as ``kalpha``.  ``'units'`` resamples
        units with replacement and recomputes alpha (percentile
        interval); it keeps the within-unit dependence of the codings.
    alpha : float, default 0.05
        One minus the confidence level of the bootstrap interval.
    min_alpha : sequence of float, default (0.667, 0.8)
        Thresholds for ``prob_below``.
    seed : int, optional
        Seed of the bootstrap's random-number generator
        (``numpy.random.default_rng``).  Draws do not reproduce Stata's
        stream; only the distribution is comparable.
    draws : int, optional
        ``'krippendorff'`` bootstrap only: pairs drawn per replication
        (default ``min(25 x non-zero coincidence cells, n (m - 1) / 2)``).

    Returns
    -------
    KrippendorffAlphaResult

    Raises
    ------
    DataInsufficient
        Fewer than two pairable units, or no variation among the
        pairable values (``D_e = 0``: alpha is undefined; Stata
        ``kalpha`` reports 0 by convention, R ``irr`` NaN).
    MethodIncompatibility
        Unknown level / layout / ci_method, non-numeric values for a
        metric level, negative values for ``'ratio'``.

    Examples
    --------
    Textbook example from the help file of Stata ``kalpha`` (4 coders,
    12 units, missing codings):

    >>> import numpy as np
    >>> import statspai as sp
    >>> nan = np.nan
    >>> coders = [[1, 2, 3, 3, 2, 1, 4, 1, 2, nan, nan, nan],
    ...           [1, 2, 3, 3, 2, 2, 4, 1, 2, 5, nan, 3],
    ...           [nan, 3, 3, 3, 2, 3, 4, 2, 2, 5, 1, nan],
    ...           [1, 2, 3, 3, 2, 4, 4, 1, 2, 5, 1, nan]]
    >>> res = sp.krippendorff_alpha(coders, level="nominal", layout="coders_by_units")
    >>> round(res.estimate, 3)
    0.743
    >>> round(sp.krippendorff_alpha(coders, level="interval",
    ...                             layout="coders_by_units").estimate, 3)
    0.849

    References
    ----------
    [@krippendorff1970estimating] [@hayes2007answering]
    """
    if level not in _LEVELS:
        raise MethodIncompatibility(
            f"level must be one of {_LEVELS}, got {level!r}.",
            recovery_hint="Choose the level of measurement of the codes.",
            diagnostics={"level": level},
        )
    if ci_method not in _CI_METHODS:
        raise MethodIncompatibility(
            f"ci_method must be one of {_CI_METHODS}, got {ci_method!r}.",
            recovery_hint="Use 'krippendorff' (kalpha algorithm) or 'units'.",
        )
    if not 0 < alpha < 1:
        raise MethodIncompatibility(
            f"alpha must be in (0, 1), got {alpha!r}.",
            recovery_hint="alpha is one minus the confidence level, e.g. 0.05.",
        )
    mat, coders = _to_matrix(data, layout)
    present = np.array([[v is not None for v in row] for row in mat], dtype=bool)
    observed = [v for row in mat for v in row if v is not None]
    if level == "nominal":
        try:
            vals = np.array(sorted(set(observed)), dtype=object)
        except TypeError:
            vals = np.array(sorted(set(observed), key=lambda v: str(v)), dtype=object)
    else:
        try:
            num = sorted({float(v) for v in observed})
        except (TypeError, ValueError) as exc:
            raise MethodIncompatibility(
                f"level={level!r} needs numeric codings ({exc}).",
                recovery_hint=(
                    "Use level='nominal' for text labels, or map them to numbers."
                ),
            ) from exc
        if level == "ratio" and num and num[0] < 0:
            raise MethodIncompatibility(
                "level='ratio' needs non-negative values.",
                recovery_hint="Ratio data have a true zero; use 'interval' otherwise.",
                diagnostics={"min_value": num[0]},
            )
        vals = np.array(num, dtype=float)
    pos = {v: i for i, v in enumerate(vals.tolist())}
    U, C = mat.shape
    counts = np.zeros((U, len(vals)))
    for u in range(U):
        for c in range(C):
            v = mat[u, c]
            if v is not None:
                key = v if level == "nominal" else float(v)
                counts[u, pos[key]] += 1.0
    m_u = present.sum(axis=1)
    pairable = m_u >= 2
    if int(pairable.sum()) < 2:
        raise DataInsufficient(
            f"{int(pairable.sum())} unit(s) coded by two or more coders; "
            "alpha needs at least two pairable units.",
            recovery_hint="Have more units coded by at least two coders.",
            diagnostics={"n_units": int(U), "n_pairable_units": int(pairable.sum())},
        )
    pc = counts[pairable]
    keep_vals = pc.sum(axis=0) > 0
    pc = pc[:, keep_vals]
    vals_p = vals[keep_vals]
    core = _alpha_from_counts(pc, level, vals_p)
    if not core["D_e"] > 0:
        raise DataInsufficient(
            "all pairable codings have the same value, so the expected "
            "disagreement is 0 and alpha is undefined.",
            recovery_hint=(
                "Report raw agreement instead, or code a sample with variation. "
                "Stata kalpha reports 0 here by convention; R irr returns NaN."
            ),
            diagnostics={"values": vals_p.tolist(), "n_pairable": core["n"]},
        )
    est = float(core["alpha"])
    vlist = vals_p.tolist()
    res = KrippendorffAlphaResult(
        estimate=est,
        level=level,
        observed_disagreement=float(core["D_o"]),
        expected_disagreement=float(core["D_e"]),
        n_units=int(pairable.sum()),
        n_coders=int(C),
        n_pairable=int(round(core["n"])),
        values=vlist,
        coincidence=pd.DataFrame(core["O"], index=vlist, columns=vlist),
        alpha=float(alpha),
        diagnostics={
            "n_units_total": int(U),
            "n_units_unpairable": int(U - pairable.sum()),
            "coders": coders,
        },
    )
    if not bootstrap:
        return res

    if int(n_boot) < 1:
        raise MethodIncompatibility(
            "n_boot must be a positive integer.",
            recovery_hint="e.g. n_boot=20000.",
        )
    rng = np.random.default_rng(seed)
    n_boot = int(n_boot)
    if ci_method == "krippendorff":
        if est in (0.0, 1.0):
            warnings.warn(
                f"alpha = {est:g} exactly: the pair-drawing bootstrap is degenerate "
                "here (Stata kalpha skips it too); no interval is reported.",
                UserWarning,
                stacklevel=2,
            )
            res.diagnostics["bootstrap_skipped"] = f"alpha == {est:g}"
            res.ci_method, res.seed = ci_method, seed
            return res
        boot, used_draws = _krippendorff_boot(core, C, n_boot, rng, draws)
        res.diagnostics["draws"] = used_draws
        lo_i = int(max(alpha / 2 * n_boot, 1))
        hi_i = int(np.floor((1 - alpha / 2) * n_boot) + 1)
        lo = float(boot[min(lo_i, n_boot) - 1])
        hi = float(boot[min(hi_i, n_boot) - 1])
    else:
        boot = np.empty(n_boot)
        n_undefined = 0
        idx_units = np.arange(pc.shape[0])
        for b in range(n_boot):
            take = rng.choice(idx_units, size=len(idx_units), replace=True)
            cb = pc[take]
            kv = cb.sum(axis=0) > 0
            cr = _alpha_from_counts(cb[:, kv], level, vals_p[kv])
            if not cr["D_e"] > 0:
                n_undefined += 1
                boot[b] = np.nan
            else:
                boot[b] = cr["alpha"]
        if n_undefined:
            warnings.warn(
                f"{n_undefined} of {n_boot} unit resamples had no variation "
                "(alpha undefined) and were left out of the interval.",
                UserWarning,
                stacklevel=2,
            )
            res.diagnostics["n_boot_undefined"] = n_undefined
        boot = np.sort(boot[np.isfinite(boot)])
        lo, hi = (float(q) for q in np.quantile(boot, [alpha / 2, 1 - alpha / 2]))
    res.ci = (lo, hi)
    res.ci_method = ci_method
    res.n_boot = n_boot
    res.seed = seed
    res.boot_dist = boot
    res.prob_below = {float(m): float(np.mean(boot < m)) for m in min_alpha}
    return res
