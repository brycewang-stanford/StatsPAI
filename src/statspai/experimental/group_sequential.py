"""Group sequential designs: stopping boundaries and maximum information.

``sp.group_sequential`` computes efficacy and futility stopping boundaries
for a trial analysed at ``K`` looks, the ratio of the maximum information to
that of a fixed-sample design with the same error rates, and the maximum
sample size when the fixed-design size is given. The boundary families and
their conventions follow Stata 18's ``gsbounds`` / ``gsdesign``:

* classical Wang-Tsiatis boundaries ``c_k = C t_k^(Delta - 1/2)``
  (``'obfleming'`` is ``Delta = 0``, ``'pocock'`` is ``Delta = 1/2``,
  ``'wtsiatis'`` takes ``Delta`` in [-10, 0.7]); classical futility bounds
  are ``a_k = theta sqrt(I_k) - C_f t_k^(Delta_f - 1/2)`` and meet the
  efficacy bound at the final look;
* error-spending boundaries of Lan and DeMets: ``'errobfleming'``
  (``2 - 2 Phi(z_{1-a/2} / sqrt(t))`` per tail), ``'errpocock'``
  (``a log(1 + (e - 1) t)``), ``'kdemets'`` (``a t^rho``) and ``'hsdecani'``
  (``a (1 - exp(-gamma t)) / (1 - exp(-gamma))``). Futility bounds spend the
  type II error under the alternative.

A two-sided design splits ``alpha`` equally between the tails; its futility
region is the inner interval ``(-a_k, a_k)``. Power counts rejections in the
direction of the alternative. Futility bounds are nonbinding by default: the
efficacy bounds are computed as if the trial never stopped for futility, so
the type I error is controlled whether or not futility stops are honoured.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Sequence

import numpy as np
import pandas as pd
from scipy.special import ndtr, ndtri

from .._result_serialize import ResultProtocolMixin
from ..exceptions import MethodIncompatibility
from . import _gs_core as core

__all__ = ["group_sequential", "GroupSequentialResult"]


@dataclass
class GroupSequentialResult(ResultProtocolMixin):
    """Result of :func:`group_sequential`.

    Attributes
    ----------
    boundaries : pd.DataFrame
        One row per look: ``look``, ``info_frac``, the efficacy and futility
        critical values on the z scale (``efficacy_lower`` /
        ``efficacy_upper`` / ``futility_lower`` / ``futility_upper``, NaN
        where a design has no such bound), their nominal p-values,
        cumulative ``alpha_spent`` (futility stops ignored),
        ``alpha_spent_futility`` (futility stops honoured), cumulative
        ``beta_spent`` under the alternative, and the stopping probabilities
        under the null and the alternative. With ``n_fixed`` given, ``n``
        is the cumulative sample size at each look.
    info_ratio : float
        Maximum information over the information of the fixed design with
        the same alpha and power (the sample-size inflation factor).
    power_attained : float
        Power of the design at the alternative, which equals the target.
    expected_info_h0, expected_info_h1 : float
        Expected information at stopping, as a fraction of the fixed
        design's information, under the null and the alternative (futility
        stops honoured).

    Examples
    --------
    >>> import statspai as sp
    >>> d = sp.group_sequential(n_looks=3, efficacy="obfleming")
    >>> d.boundaries["efficacy_upper"].round(4).tolist()
    [3.4711, 2.4544, 2.004]
    >>> round(d.info_ratio, 4)
    1.0174
    """

    _citation_keys = (
        "obrien1979multiple",
        "pocock1977group",
        "wang1987approximately",
        "lan1983discrete",
        "kim1987design",
        "hwang1990group",
    )

    boundaries: pd.DataFrame
    info_ratio: float
    alpha: float
    power: float
    power_attained: float
    side: str
    efficacy: Optional[str]
    futility: Optional[str]
    binding: bool
    z_fixed: float
    expected_info_h0: float
    expected_info_h1: float
    n_fixed: Optional[float] = None
    n_max: Optional[float] = None
    detail: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        eff = self.efficacy or "none (final look only)"
        fut = self.futility or "none"
        if self.futility:
            fut += ", binding" if self.binding else ", nonbinding"
        cols = [
            c
            for c in (
                "look",
                "info_frac",
                "n",
                "efficacy_lower",
                "efficacy_upper",
                "p_efficacy",
                "futility_lower",
                "futility_upper",
                "p_futility",
            )
            if c in self.boundaries and self.boundaries[c].notna().any()
        ]
        lines = [
            "Group sequential design",
            f"  Efficacy: {eff}    Futility: {fut}",
            f"  alpha = {self.alpha:g} ({self.side}), power = {self.power:g}",
            f"  Info. ratio = {self.info_ratio:.4f}"
            + (f"    N max = {self.n_max:.2f}" if self.n_max is not None else ""),
            f"  Expected info. (fraction of fixed design): "
            f"H0 {self.expected_info_h0:.4f}, H1 {self.expected_info_h1:.4f}",
            "",
            self.boundaries[cols].to_string(index=False, float_format="%.4f"),
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"GroupSequentialResult(looks={len(self.boundaries)}, "
            f"efficacy={self.efficacy!r}, futility={self.futility!r}, "
            f"info_ratio={self.info_ratio:.4f})"
        )

    def decision(self, z: Sequence[float]) -> pd.DataFrame:
        """Apply the boundaries to observed z-statistics.

        Parameters
        ----------
        z : sequence of float
            The z-statistic at each look carried out so far (at most the
            number of looks of the design), on the scale of the design: a
            positive z favours the alternative of an ``'upper'`` design and
            a negative one that of a ``'lower'`` design.

        Returns
        -------
        pd.DataFrame
            ``look``, ``z`` and ``action`` (``'continue'``,
            ``'stop: efficacy'`` or ``'stop: futility'``); rows end at the
            first stop. At the final look a z that does not cross the
            efficacy bound is a futility (acceptance) decision.
        """
        zs = np.asarray(list(z), dtype=float)
        k_max = len(self.boundaries)
        if zs.ndim != 1 or zs.size == 0 or zs.size > k_max:
            raise MethodIncompatibility(
                f"z must hold between 1 and {k_max} values (one per look).",
                recovery_hint="Pass the z-statistics of the looks done so far.",
            )
        b = self.boundaries
        rows = []
        for k, zk in enumerate(zs):
            lo, hi = b["efficacy_lower"].iloc[k], b["efficacy_upper"].iloc[k]
            flo, fhi = b["futility_lower"].iloc[k], b["futility_upper"].iloc[k]
            eff = (np.isfinite(hi) and zk >= hi) or (np.isfinite(lo) and zk <= lo)
            if self.side == "upper":
                fut = np.isfinite(flo) and zk <= flo
            elif self.side == "lower":
                fut = np.isfinite(fhi) and zk >= fhi
            else:
                fut = np.isfinite(fhi) and flo <= zk <= fhi
            if k == k_max - 1 and not eff:
                fut = True
            action = (
                "stop: efficacy" if eff else ("stop: futility" if fut else "continue")
            )
            rows.append({"look": k + 1, "z": float(zk), "action": action})
            if action != "continue":
                break
        return pd.DataFrame(rows)

    def plot(self, ax=None):  # pragma: no cover - plotting
        """Plot the boundaries against the information fraction."""
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=(6, 4))
        b = self.boundaries
        x = b["info_frac"]
        for col, style in (
            ("efficacy_upper", "o-"),
            ("efficacy_lower", "o-"),
            ("futility_upper", "s--"),
            ("futility_lower", "s--"),
        ):
            if b[col].notna().any():
                ax.plot(x, b[col], style, label=col.replace("_", " "))
        ax.axhline(0, color="grey", lw=0.5)
        ax.set_xlabel("Information fraction")
        ax.set_ylabel("Critical value (z)")
        ax.legend()
        return ax


def _info_fractions(n_looks: Optional[int], information: Any) -> np.ndarray:
    if information is not None:
        info = np.asarray(list(information), dtype=float)
        if (
            info.ndim != 1
            or info.size < 1
            or not np.all(np.isfinite(info))
            or np.any(info <= 0)
            or np.any(np.diff(info) <= 0)
        ):
            raise MethodIncompatibility(
                "information must be an increasing sequence of positive " "numbers.",
                recovery_hint="E.g. information=[0.5, 0.75, 1] or [50, 75, 100].",
                diagnostics={"information": list(np.atleast_1d(info))},
            )
        if n_looks is not None and int(n_looks) != info.size:
            raise MethodIncompatibility(
                f"n_looks={n_looks} disagrees with {info.size} information levels.",
                recovery_hint="Pass one of n_looks= or information=.",
            )
        return info / info[-1]
    k = 2 if n_looks is None else int(n_looks)
    if k < 1 or k != n_looks and n_looks is not None:
        raise MethodIncompatibility(
            f"n_looks must be a positive integer; got {n_looks!r}.",
            recovery_hint="E.g. n_looks=3 for two interim looks and a final one.",
        )
    return np.arange(1, k + 1) / k


def _classical_efficacy(
    t: np.ndarray,
    alpha: float,
    sides: int,
    delta: Optional[float],
    fut_of: Optional[Any] = None,
) -> np.ndarray:
    """Wang-Tsiatis efficacy bounds (or final-look-only when ``delta`` is None).

    ``fut_of(b_final)`` returns the futility bounds that the efficacy
    bounds must account for (binding futility); None ignores futility.
    """
    k = len(t)

    def bounds(c: float) -> np.ndarray:
        if delta is None:
            out = np.full(k, np.inf)
            out[-1] = c
            return out
        return c * t ** (delta - 0.5)

    def total(c: float) -> float:
        b = bounds(c)
        a = fut_of(b[-1]) if fut_of is not None else _no_futility(k, sides)
        path = core._Path(t, 0.0)
        s = 0.0
        for j in range(k):
            s += core.efficacy_prob(path, b[j], sides)
            if j < k - 1:
                path.advance(core.continuation(a[j], b[j], sides))
        return s - alpha

    if delta is None and fut_of is None:
        return bounds(float(ndtri(1.0 - alpha / sides)))
    return bounds(core._root(total, 0.05, core._ZMAX, "efficacy constant"))


def _no_futility(k: int, sides: int) -> np.ndarray:
    return np.full(k, 0.0 if sides == 2 else -np.inf)


def _spending_efficacy(
    t: np.ndarray,
    alpha: float,
    sides: int,
    eff: Optional[tuple],
    futility: Optional[np.ndarray],
) -> np.ndarray:
    """Error-spending efficacy bounds under H0, given (binding) futility."""
    k = len(t)
    a = futility if futility is not None else _no_futility(k, sides)
    path = core._Path(t, 0.0)
    b = np.full(k, np.inf)
    prev = 0.0
    for j in range(k):
        if eff is None:
            cum = alpha if j == k - 1 else 0.0
        else:
            cum = core.spent(eff[0], eff[2], alpha, t[j], sides)
        lower = a[j] if sides == 1 else 0.0
        b[j] = core.solve_efficacy(path, cum - prev, sides, lower)
        prev = cum
        if j < k - 1:
            path.advance(core.continuation(a[j], b[j], sides))
    return b


def _pass_h1(
    t: np.ndarray,
    info_max: float,
    sides: int,
    b: np.ndarray,
    fut: Optional[tuple],
    beta: float,
    classical_a: Optional[np.ndarray] = None,
) -> Dict[str, np.ndarray]:
    """Walk the alternative (drift 1): futility bounds, power, stop probs."""
    k = len(t)
    path = core._Path(t * info_max, 1.0)
    a = _no_futility(k, sides)
    eff_up = np.zeros(k)
    fut_p = np.zeros(k)
    stop = np.zeros(k)
    prev = 0.0
    for j in range(k):
        if j == k - 1:
            a[j] = b[j] if fut is not None else a[j]
        elif classical_a is not None:
            a[j] = classical_a[j]
        elif fut is not None:
            cum = core.spent(fut[0], fut[2], beta, t[j], 1)
            a[j] = core.solve_futility(path, cum - prev, sides, b[j])
            prev = cum
        eff_up[j] = path.above(b[j])
        eff_any = core.efficacy_prob(path, b[j], sides)
        if j < k - 1:
            fut_p[j] = core.futility_prob(path, a[j], sides)
            stop[j] = eff_any + fut_p[j]
            path.advance(core.continuation(a[j], b[j], sides))
        else:
            stop[j] = path.mass()
            fut_p[j] = stop[j] - eff_any
    return {"a": a, "power_by_look": eff_up, "fut": fut_p, "stop": stop}


def _pass_h0(
    t: np.ndarray, sides: int, a: np.ndarray, b: np.ndarray
) -> Dict[str, np.ndarray]:
    k = len(t)
    path = core._Path(t, 0.0)
    rej = np.zeros(k)
    stop = np.zeros(k)
    for j in range(k):
        rej[j] = core.efficacy_prob(path, b[j], sides)
        if j < k - 1:
            stop[j] = rej[j] + core.futility_prob(path, a[j], sides)
            path.advance(core.continuation(a[j], b[j], sides))
        else:
            stop[j] = path.mass()
    return {"reject": rej, "stop": stop}


def group_sequential(
    n_looks: Optional[int] = None,
    *,
    information: Optional[Sequence[float]] = None,
    alpha: float = 0.05,
    power: float = 0.8,
    side: str = "two-sided",
    efficacy: Optional[str] = None,
    efficacy_param: Optional[float] = None,
    futility: Optional[str] = None,
    futility_param: Optional[float] = None,
    binding: bool = False,
    n_fixed: Optional[float] = None,
) -> GroupSequentialResult:
    """Stopping boundaries and sample-size inflation of a group sequential design.

    Parameters
    ----------
    n_looks : int, optional
        Number of analyses (interim looks plus the final one), equally
        spaced in information. Default 2 when ``information`` is not given.
    information : sequence of float, optional
        Information (or sample size) at each analysis, increasing; only the
        ratios matter. Overrides equal spacing.
    alpha : float, default 0.05
        Overall (familywise) type I error over all looks.
    power : float, default 0.8
        Power at the alternative the design is sized for.
    side : {'two-sided', 'upper', 'lower'}
        ``'upper'`` / ``'lower'`` are one-sided tests in which a large
        positive / negative z is evidence for the alternative.
    efficacy : str, optional
        Efficacy boundary: ``'obfleming'``, ``'pocock'``, ``'wtsiatis'``
        (classical Wang-Tsiatis family) or ``'errobfleming'``,
        ``'errpocock'``, ``'kdemets'``, ``'hsdecani'`` (Lan-DeMets error
        spending). When neither ``efficacy`` nor ``futility`` is given the
        default is ``'obfleming'``; when only ``futility`` is given there is
        no early efficacy stopping (Stata's rule).
    efficacy_param : float, optional
        ``Delta`` for ``'wtsiatis'`` (in [-10, 0.7]), ``rho`` for
        ``'kdemets'`` (in (0, 10]), ``gamma`` for ``'hsdecani'`` (in
        [-30, 3]).
    futility : str, optional
        Futility boundary from the same class as ``efficacy`` (classical
        with classical, spending with spending).
    futility_param : float, optional
        Parameter of the futility boundary, as ``efficacy_param``.
    binding : bool, default False
        Binding futility bounds: the efficacy bounds account for futility
        stopping, which lowers them but inflates the type I error if a
        futility stop is ever overruled.
    n_fixed : float, optional
        Sample size (or number of events) of the fixed design with the same
        alpha and power, e.g. from :func:`sp.power_ttest`. When given, the
        result carries ``n_max = info_ratio * n_fixed`` and the cumulative
        size at each look (fractional; round up per arm when planning).

    Returns
    -------
    GroupSequentialResult
        ``boundaries`` (per-look critical values, nominal p-values, alpha and
        beta spent, stopping probabilities), ``info_ratio``,
        ``expected_info_h0`` / ``expected_info_h1``, and ``.decision(z)``
        to apply the bounds to observed statistics.

    Notes
    -----
    The bounds solve the crossing-probability equations exactly up to the
    numerical integration (relative error below 1e-9). Stata's ``gsbounds``
    reaches its own alpha to about 4e-9 and its bounds agree with these to
    about 1e-7 (relative); the alpha actually spent at Stata's bounds,
    evaluated here, differs from the target by less than 1e-8.

    With error-spending bounds the boundaries at an interim look depend only
    on the information fractions up to that look, so a design can be
    re-computed at the information actually observed (pass ``information=``
    with the realised fractions and the planned maximum as the last value).

    Examples
    --------
    >>> import statspai as sp
    >>> d = sp.group_sequential(n_looks=3, efficacy="errobfleming",
    ...                         futility="errpocock", side="upper")
    >>> d.boundaries[["efficacy_upper", "futility_lower"]].round(4).values.tolist()
    [[3.2001, 0.2856], [2.1408, 1.063], [1.6948, 1.6948]]
    >>> round(d.info_ratio, 4)
    1.2779
    >>> d.decision([0.1]).action.tolist()
    ['stop: futility']

    References
    ----------
    obrien1979multiple; pocock1977group; wang1987approximately;
    lan1983discrete; kim1987design; hwang1990group
    """
    if not (0 < alpha < 0.5):
        raise MethodIncompatibility(
            f"alpha must be in (0, 0.5); got {alpha!r}.",
            recovery_hint="E.g. alpha=0.05.",
        )
    if not (0.5 < power < 1):
        raise MethodIncompatibility(
            f"power must be in (0.5, 1); got {power!r}.",
            recovery_hint="E.g. power=0.8.",
        )
    side_key = str(side).strip().lower().replace("_", "-")
    side_key = {"two": "two-sided", "onesided": "upper", "one-sided": "upper"}.get(
        side_key, side_key
    )
    if side_key not in ("two-sided", "upper", "lower"):
        raise MethodIncompatibility(
            f"side must be 'two-sided', 'upper' or 'lower'; got {side!r}.",
            recovery_hint="Use side='upper' for a one-sided test of a positive effect.",
        )
    sides = 2 if side_key == "two-sided" else 1
    t = _info_fractions(n_looks, information)
    k = len(t)
    beta = 1.0 - power

    if efficacy is None and futility is None:
        efficacy = "obfleming"
    eff = (
        core.resolve_family(efficacy, efficacy_param, "efficacy")
        if efficacy is not None
        else None
    )
    fut = (
        core.resolve_family(futility, futility_param, "futility")
        if futility is not None
        else None
    )
    if eff is not None and fut is not None and eff[1] != fut[1]:
        raise MethodIncompatibility(
            "Efficacy and futility boundaries must both be classical "
            "(obfleming / pocock / wtsiatis) or both error-spending.",
            recovery_hint=(
                "E.g. efficacy='errobfleming' with futility='errpocock', or "
                "efficacy='obfleming' with futility='pocock'."
            ),
        )
    if binding and fut is None:
        raise MethodIncompatibility(
            "binding=True needs a futility boundary.",
            recovery_hint="Pass futility=..., or drop binding=True.",
        )
    if n_fixed is not None and not (np.isfinite(n_fixed) and n_fixed > 0):
        raise MethodIncompatibility(
            f"n_fixed must be a positive number; got {n_fixed!r}.",
            recovery_hint="E.g. n_fixed=sp.power_ttest(delta=0.5, power=0.8).n",
        )
    classical = (eff or fut)[1] == "classical"
    z_alpha = float(ndtri(1.0 - alpha / sides))
    info_fixed = (z_alpha + float(ndtri(power))) ** 2  # drift 1 per unit info
    eff_delta = eff[2] if (eff is not None and classical) else None

    def solve_bounds(r: float):
        """Bounds and the H1 walk at maximum information ``r * info_fixed``."""
        info_max = r * info_fixed
        if fut is None or not binding:
            b = b_nonbinding
        elif classical:
            b = _classical_efficacy(
                t, alpha, sides, eff_delta, lambda bk: classical_fut(info_max, bk)
            )
        else:
            b = _binding_spending(t, alpha, sides, eff, fut, beta, info_max)
        ca = classical_fut(info_max, b[-1]) if (classical and fut) else None
        return b, _pass_h1(t, info_max, sides, b, fut, beta, ca)

    def classical_fut(info_max: float, b_final: float) -> np.ndarray:
        drift = np.sqrt(t * info_max)
        c2 = drift[-1] - b_final
        a = drift - c2 * t ** (fut[2] - 0.5)
        if sides == 2:
            a = np.maximum(a, 0.0)
        return a

    b_nonbinding = (
        _classical_efficacy(t, alpha, sides, eff_delta)
        if classical
        else _spending_efficacy(t, alpha, sides, eff, None)
    )

    def gap(r: float) -> float:
        # Power is the probability of rejecting in the direction of the
        # alternative, except for error-spending futility bounds: those
        # designs are closed by spending exactly beta (the final futility
        # bound meets the efficacy bound), so a rejection in the wrong tail
        # of a two-sided design counts towards 1 - beta. Both rules are
        # Stata's; the wrong-tail probability is of order 1e-5.
        try:
            h1 = solve_bounds(r)[1]
        except core.InfeasibleSpending:
            # Binding futility bounds grow with the information; past some
            # point they leave the null too little probability to spend.
            # That only happens beyond the solution, where the design is
            # over-powered.
            return 1.0
        if fut is None or classical:
            return float(h1["power_by_look"].sum()) - power
        return beta - float(h1["fut"].sum())

    lo, hi = 0.5, 2.0
    while gap(lo) > 0 and lo > 1e-3:
        lo /= 2.0
    while gap(hi) < 0 and hi < 1e3:
        hi *= 2.0
    ratio = core._root(gap, lo, hi, "maximum information")
    b, h1 = solve_bounds(ratio)
    a = h1["a"]
    h0_eff = _pass_h0(t, sides, _no_futility(k, sides), b)
    h0_fut = _pass_h0(t, sides, a, b)
    cum_beta = np.cumsum(h1["fut"])
    boundaries = _boundary_frame(
        t, sides, side_key, a, b, eff, fut, h0_eff, h0_fut, h1, cum_beta
    )
    if n_fixed is not None:
        boundaries.insert(2, "n", t * ratio * float(n_fixed))
    return GroupSequentialResult(
        boundaries=boundaries,
        info_ratio=float(ratio),
        alpha=float(alpha),
        power=float(power),
        power_attained=float(h1["power_by_look"].sum()),
        side=side_key,
        efficacy=eff[0] if eff else None,
        futility=fut[0] if fut else None,
        binding=bool(binding and fut is not None),
        z_fixed=z_alpha,
        expected_info_h0=float(ratio * np.dot(t, h0_fut["stop"])),
        expected_info_h1=float(ratio * np.dot(t, h1["stop"])),
        n_fixed=None if n_fixed is None else float(n_fixed),
        n_max=None if n_fixed is None else float(ratio * n_fixed),
        detail={
            "efficacy_param": None if eff is None else eff[2],
            "futility_param": None if fut is None else fut[2],
            "info_fixed_per_unit_drift": info_fixed,
            "info_fractions": t.tolist(),
        },
    )


def _binding_spending(t, alpha, sides, eff, fut, beta, info_max) -> np.ndarray:
    """Efficacy bounds that honour binding spending futility bounds.

    The null and alternative walks advance together: the efficacy bound at a
    look depends on the earlier futility bounds and vice versa.
    """
    k = len(t)
    p0 = core._Path(t, 0.0)
    p1 = core._Path(t * info_max, 1.0)
    a = _no_futility(k, sides)
    b = np.full(k, np.inf)
    prev_a = prev_b = 0.0
    for j in range(k):
        if eff is None:
            cum = alpha if j == k - 1 else 0.0
        else:
            cum = core.spent(eff[0], eff[2], alpha, t[j], sides)
        b[j] = core.solve_efficacy(p0, cum - prev_a, sides, a[j] if sides == 1 else 0.0)
        prev_a = cum
        if j < k - 1:
            cumb = core.spent(fut[0], fut[2], beta, t[j], 1)
            a[j] = core.solve_futility(p1, cumb - prev_b, sides, b[j])
            prev_b = cumb
            region = core.continuation(a[j], b[j], sides)
            p0.advance(region)
            p1.advance(region)
    return b


def _boundary_frame(t, sides, side_key, a, b, eff, fut, h0_eff, h0_fut, h1, cum_beta):
    k = len(t)
    nan = np.full(k, np.nan)
    b_show = np.where(np.isfinite(b), b, np.nan)
    has_fut = fut is not None
    a_show = np.asarray(a, dtype=float).copy() if has_fut else nan.copy()
    if has_fut:
        a_show = np.where(np.isfinite(a_show), a_show, np.nan)
        if sides == 2:
            a_show = np.where(a_show > 0, a_show, np.nan)
    if sides == 2:
        eff_lo, eff_hi = -b_show, b_show
        fut_lo, fut_hi = -a_show, a_show
    elif side_key == "upper":
        eff_lo, eff_hi, fut_lo, fut_hi = nan, b_show, a_show, nan
    else:
        eff_lo, eff_hi, fut_lo, fut_hi = -b_show, nan, nan, -a_show
    p_eff = sides * ndtr(-b_show)
    p_fut = (sides * ndtr(-a_show)) if has_fut else nan
    return pd.DataFrame(
        {
            "look": np.arange(1, k + 1),
            "info_frac": t,
            "efficacy_lower": eff_lo,
            "efficacy_upper": eff_hi,
            "p_efficacy": p_eff,
            "futility_lower": fut_lo,
            "futility_upper": fut_hi,
            "p_futility": p_fut,
            "alpha_spent": np.cumsum(h0_eff["reject"]),
            "alpha_spent_futility": np.cumsum(h0_fut["reject"]),
            "beta_spent": cum_beta if has_fut else nan,
            "stop_prob_h0": h0_fut["stop"],
            "stop_prob_h1": h1["stop"],
        }
    )
