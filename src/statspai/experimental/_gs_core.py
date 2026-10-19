"""Numerical core of group sequential designs.

The joint law of the z-statistics at the analyses of a group sequential trial
is that of a Brownian motion with drift observed at the information levels
``I_1 < ... < I_K``: ``Z_k sqrt(I_k)`` has independent normal increments with
mean ``theta (I_k - I_{k-1})`` and variance ``I_k - I_{k-1}``. Crossing
probabilities are computed with the Armitage-McPherson-Rowe recursion: the
sub-density of ``Z_k`` on the continuation region is carried from look to look
and integrated with composite Gauss-Legendre rules (12 nodes per panel of
width 0.5 on +-12 standard deviations around the mean). The integrand is a
Gaussian mixture, so the rule is accurate to ~1e-13; refining the panels does
not move a boundary in the ninth significant digit.

Everything here works in the "upper" orientation: efficacy is a large
positive z. A lower one-sided design is the mirror image and is negated by
the caller.
"""

from __future__ import annotations

from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.optimize import brentq
from scipy.special import ndtr, ndtri

from ..exceptions import ConvergenceFailure, MethodIncompatibility

_GLX, _GLW = np.polynomial.legendre.leggauss(12)
_SPAN = 12.0  # standard deviations kept on each side of the mean
_PANEL = 0.5
_ZMAX = 40.0  # boundaries are searched in [-_ZMAX, _ZMAX]
_SQRT2PI = np.sqrt(2.0 * np.pi)

Interval = Tuple[float, float]


class InfeasibleSpending(ConvergenceFailure):
    """Binding futility left less null probability than the look must spend."""


def _nodes(lo: float, hi: float) -> Tuple[np.ndarray, np.ndarray]:
    npan = max(1, int(np.ceil((hi - lo) / _PANEL)))
    edges = np.linspace(lo, hi, npan + 1)
    a = edges[:-1, None]
    b = edges[1:, None]
    x = (a + b) / 2.0 + (b - a) / 2.0 * _GLX
    w = (b - a) / 2.0 * _GLW * np.ones_like(x)
    return x.ravel(), w.ravel()


class _Path:
    """Sub-density of ``Z_k`` on the continuation region, look by look."""

    def __init__(self, info: Sequence[float], theta: float) -> None:
        self.info = np.asarray(info, dtype=float)
        self.theta = float(theta)
        self.k = 0
        self.u = np.zeros(1)
        self.m = np.ones(1)
        self.i_prev = 0.0

    def _cond(self) -> Tuple[np.ndarray, float]:
        i_k = self.info[self.k]
        d = i_k - self.i_prev
        mean = (self.u * np.sqrt(self.i_prev) + self.theta * d) / np.sqrt(i_k)
        return mean, float(np.sqrt(d / i_k))

    def above(self, b: float) -> float:
        """P(Z_k > b and no stop before k)."""
        if b == np.inf:
            return 0.0
        mean, sd = self._cond()
        return float(np.dot(self.m, ndtr((mean - b) / sd)))

    def below(self, a: float) -> float:
        """P(Z_k < a and no stop before k)."""
        if a == -np.inf:
            return 0.0
        mean, sd = self._cond()
        return float(np.dot(self.m, ndtr((a - mean) / sd)))

    def mass(self) -> float:
        return float(self.m.sum())

    def advance(self, region: Sequence[Interval]) -> None:
        """Move to look ``k+1``, continuing on ``region`` at look ``k``."""
        mean, sd = self._cond()
        centre = self.theta * np.sqrt(self.info[self.k])
        xs: List[np.ndarray] = []
        ws: List[np.ndarray] = []
        for lo, hi in region:
            lo2, hi2 = max(lo, centre - _SPAN), min(hi, centre + _SPAN)
            if hi2 > lo2:
                x, w = _nodes(lo2, hi2)
                xs.append(x)
                ws.append(w)
        if xs and self.m.size:
            z = np.concatenate(xs)
            w = np.concatenate(ws)
            dens = np.exp(-0.5 * ((z[:, None] - mean[None, :]) / sd) ** 2) @ self.m
            self.u, self.m = z, w * dens / (sd * _SQRT2PI)
        else:
            self.u, self.m = np.zeros(0), np.zeros(0)
        self.i_prev = float(self.info[self.k])
        self.k += 1


def continuation(a: float, b: float, sides: int) -> List[Interval]:
    """Region where the trial continues: efficacy bound ``b``, futility ``a``.

    One-sided: ``(a, b)`` with ``a = -inf`` when there is no futility bound.
    Two-sided: ``(-b, -a) U (a, b)`` with ``a = 0`` for no futility bound.
    """
    if sides == 1:
        return [(a, b)] if b > a else []
    if a <= 0:
        return [(-b, b)]
    return [(-b, -a), (a, b)] if b > a else []


def efficacy_prob(path: _Path, b: float, sides: int) -> float:
    return path.above(b) + (path.below(-b) if sides == 2 else 0.0)


def futility_prob(path: _Path, a: float, sides: int) -> float:
    if sides == 1:
        return path.below(a)
    if a <= 0:
        return 0.0
    return path.below(a) - path.below(-a)


# ---------------------------------------------------------------------------
# Boundary families
# ---------------------------------------------------------------------------

#: canonical name -> (class, default parameter)
FAMILIES = {
    "obfleming": ("classical", 0.0),
    "pocock": ("classical", 0.5),
    "wtsiatis": ("classical", None),
    "errobfleming": ("spending", None),
    "errpocock": ("spending", None),
    "kdemets": ("spending", None),
    "hsdecani": ("spending", None),
}

_ALIASES = {
    "obf": "obfleming",
    "obrien_fleming": "obfleming",
    "wang_tsiatis": "wtsiatis",
    "errobf": "errobfleming",
    "ld_obf": "errobfleming",
    "lan_demets_obf": "errobfleming",
    "ld_pocock": "errpocock",
    "lan_demets_pocock": "errpocock",
    "kim_demets": "kdemets",
    "power": "kdemets",
    "hsd": "hsdecani",
    "hwang_shih_decani": "hsdecani",
}

_PARAM_RANGE = {
    "wtsiatis": (-10.0, 0.7, "Delta"),
    "kdemets": (1e-12, 10.0, "rho"),
    "hsdecani": (-30.0, 3.0, "gamma"),
}


def resolve_family(
    spec: str, param: Optional[float], role: str
) -> Tuple[str, str, float]:
    """Return ``(name, class, parameter)`` for a boundary specification."""
    name = str(spec).strip().lower().replace("-", "_")
    name = _ALIASES.get(name, name)
    if name not in FAMILIES:
        raise MethodIncompatibility(
            f"Unknown {role} boundary {spec!r}.",
            recovery_hint=(
                "Use one of 'obfleming', 'pocock', 'wtsiatis' (classical) or "
                "'errobfleming', 'errpocock', 'kdemets', 'hsdecani' "
                "(error spending)."
            ),
            diagnostics={"supported": sorted(FAMILIES)},
        )
    cls, default = FAMILIES[name]
    if default is not None:
        if param is not None and name in ("obfleming", "pocock"):
            raise MethodIncompatibility(
                f"{name!r} takes no parameter; use 'wtsiatis' with "
                f"{role}_param= to choose Delta.",
                recovery_hint=f"Drop {role}_param= or use '{role}=\"wtsiatis\"'.",
            )
        return name, cls, float(default)
    if name == "errobfleming" or name == "errpocock":
        if param is not None:
            raise MethodIncompatibility(
                f"{name!r} takes no parameter.",
                recovery_hint=f"Drop {role}_param=, or use 'kdemets' / 'hsdecani'.",
            )
        return name, cls, float("nan")
    lo, hi, label = _PARAM_RANGE[name]
    if param is None or not np.isfinite(param) or not (lo <= param <= hi):
        raise MethodIncompatibility(
            f"{name!r} needs {role}_param= ({label}) in [{lo:g}, {hi:g}]; "
            f"got {param!r}.",
            recovery_hint=f"Pass {role}_param=<{label}>, e.g. 2 for kdemets, "
            "-4 for hsdecani, 0.25 for wtsiatis.",
        )
    return name, cls, float(param)


def spent(name: str, param: float, total: float, t: float, sides: int) -> float:
    """Cumulative error spent at information fraction ``t``.

    ``total`` is the error to spend over all looks; with ``sides=2`` it is
    split equally between the two tails, and the O'Brien-Fleming-type
    function is applied to each tail at level ``total / 2``.
    """
    if t >= 1.0:
        return total
    if t <= 0.0:
        return 0.0
    if name == "errobfleming":
        a = total / sides
        return sides * 2.0 * ndtr(ndtri(a / 2.0) / np.sqrt(t))
    if name == "errpocock":
        return total * np.log1p((np.e - 1.0) * t)
    if name == "kdemets":
        return total * t**param
    if name == "hsdecani":
        if param == 0.0:
            return total * t
        return total * (-np.expm1(-param * t)) / (-np.expm1(-param))
    raise AssertionError(name)  # pragma: no cover


def _root(f: Callable[[float], float], lo: float, hi: float, what: str) -> float:
    flo, fhi = f(lo), f(hi)
    if flo == 0.0:
        return lo
    if fhi == 0.0:
        return hi
    if np.sign(flo) == np.sign(fhi):
        raise ConvergenceFailure(
            f"Could not bracket the {what}.",
            recovery_hint=(
                "The requested design is infeasible with these settings; "
                "check alpha / power, the boundary parameters and the "
                "information fractions."
            ),
            diagnostics={"lo": lo, "hi": hi, "f_lo": flo, "f_hi": fhi},
        )
    return float(brentq(f, lo, hi, xtol=1e-13, rtol=1e-15, maxiter=300))


def solve_efficacy(path: _Path, target: float, sides: int, lower: float) -> float:
    """Efficacy bound at the current look spending ``target`` under the path."""
    if target <= 0.0:
        return np.inf
    lo = 0.0 if sides == 2 else max(lower, -_ZMAX)
    if lo == -np.inf:
        lo = -_ZMAX

    def f(c: float) -> float:
        return efficacy_prob(path, c, sides) - target

    if f(lo) < 0.0:
        raise InfeasibleSpending(
            "Not enough probability left to spend at this look.",
            recovery_hint=(
                "The futility bounds remove more probability than the "
                "efficacy spending needs; use nonbinding futility or a "
                "less aggressive futility boundary."
            ),
            diagnostics={"target": target, "available": f(lo) + target},
        )
    return _root(f, lo, _ZMAX, "efficacy bound")


def solve_futility(path: _Path, target: float, sides: int, upper: float) -> float:
    """Futility bound at the current look spending ``target`` under the path."""
    hi = upper if np.isfinite(upper) else _ZMAX
    if target <= 0.0:
        return 0.0 if sides == 2 else -np.inf
    lo = 0.0 if sides == 2 else -_ZMAX

    def f(a: float) -> float:
        return futility_prob(path, a, sides) - target

    if f(hi) <= 0.0:
        return hi  # every path that has not crossed efficacy stops here
    return _root(f, lo, hi, "futility bound")
