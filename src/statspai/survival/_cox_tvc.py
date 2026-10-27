"""Time-varying coefficients for ``sp.cox``: Stata ``stcox, tvc() texp()``.

``stcox x, tvc(z) texp(g(_t))`` adds the covariate ``z * g(t)`` to the
model, whose coefficient is reported in a separate ``tvc`` equation. The
partial likelihood only evaluates covariates at failure times, so the
fit is exact after splitting every row's interval at the failure times
of its stratum that fall strictly inside it: on each piece ``(a, b]`` the
only failure time at which the piece is at risk is ``b`` (or none), and
the piece carries ``z * g(b)``. The split rows go through the
counting-process risk sets of ``sp.cox(start=)``; Efron and Breslow ties,
case weights and the robust and clustered variances are unchanged by the
split because each original row is at risk at the same failure times,
with the same weight, as before. The robust variance sums the score
residuals of a row's pieces first (Stata and R ``coxph(tt=)`` both treat
the original row as the unit), so ``N`` in Stata's ``N/(N-1)`` is the
number of original rows.

``texp`` is the function ``g``: a callable on the array of times, or a
Stata expression in ``_t`` read by a closed-grammar parser (numbers,
``_t``, ``+ - * / ^``, comparisons, parentheses and the functions
``ln log log10 exp sqrt abs min max``); nothing is passed to ``eval``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from ..exceptions import DataInsufficient, MethodIncompatibility

TexpType = Union[None, str, Callable[[np.ndarray], Any]]

#: Upper bound on the rows of the split data (each costs ~8 bytes per
#: covariate plus bookkeeping); beyond it the fit refuses instead of
#: exhausting memory.
MAX_EPISODES = 100_000_000

_TOKEN = re.compile(
    r"\s*(?:(?P<num>(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)"
    r"|(?P<name>[A-Za-z_][A-Za-z0-9_]*)"
    r"|(?P<op>>=|<=|==|!=|~=|[-+*/^()<>,]))"
)

_FUNCS_1 = {
    "ln": np.log,
    "log": np.log,
    "log10": np.log10,
    "exp": np.exp,
    "sqrt": np.sqrt,
    "abs": np.abs,
}
_FUNCS_2 = {"min": np.fmin, "max": np.fmax}


def _texp_error(text: str, why: str) -> MethodIncompatibility:
    return MethodIncompatibility(
        f"texp({text}) cannot be read: {why}.",
        recovery_hint=(
            "Write texp as an expression in _t with numbers, + - * / ^, "
            "comparisons, parentheses and ln / log / log10 / exp / sqrt / abs "
            "/ min / max (e.g. '_t', 'ln(_t)', '_t^2', '_t > 5'), or pass a "
            "Python callable g(t) -> array."
        ),
        diagnostics={"texp": text},
    )


class _TexpParser:
    """Recursive-descent reader of a Stata expression in ``_t``.

    Precedence follows Stata: comparisons < ``+ -`` < ``* /`` < unary
    minus < ``^`` (so ``-_t^2`` is ``-(_t^2)``). A chain ``a^b^c`` is
    refused rather than guessed at.
    """

    def __init__(self, text: str):
        self.text = text
        self.toks: List[tuple] = []
        pos = 0
        stripped = text.strip()
        while pos < len(stripped):
            m = _TOKEN.match(stripped, pos)
            if m is None or m.end() == pos:
                raise _texp_error(text, f"unexpected character at {stripped[pos:]!r}")
            kind = m.lastgroup
            self.toks.append((kind, m.group(kind)))
            pos = m.end()
        self.i = 0

    def _peek(self) -> Optional[tuple]:
        return self.toks[self.i] if self.i < len(self.toks) else None

    def _take(self, value: Optional[str] = None) -> tuple:
        tok = self._peek()
        if tok is None or (value is not None and tok[1] != value):
            raise _texp_error(
                self.text, f"expected {value!r}" if value else "unexpected end"
            )
        self.i += 1
        return tok

    def parse(self) -> Callable[[np.ndarray], np.ndarray]:
        if not self.toks:
            raise _texp_error(self.text, "the expression is empty")
        fn = self._comparison()
        if self._peek() is not None:
            raise _texp_error(self.text, f"unexpected {self._peek()[1]!r}")
        return fn

    def _comparison(self) -> Callable:
        left = self._additive()
        tok = self._peek()
        if tok is not None and tok[1] in (">", "<", ">=", "<=", "==", "!=", "~="):
            self.i += 1
            right = self._additive()
            op = tok[1]
            ops = {
                ">": np.greater,
                "<": np.less,
                ">=": np.greater_equal,
                "<=": np.less_equal,
                "==": np.equal,
                "!=": np.not_equal,
                "~=": np.not_equal,
            }[op]
            return lambda t, a=left, b=right, f=ops: f(a(t), b(t)).astype(float)
        return left

    def _additive(self) -> Callable:
        left = self._multiplicative()
        while (tok := self._peek()) is not None and tok[1] in ("+", "-"):
            self.i += 1
            right = self._multiplicative()
            if tok[1] == "+":
                left = lambda t, a=left, b=right: a(t) + b(t)  # noqa: E731
            else:
                left = lambda t, a=left, b=right: a(t) - b(t)  # noqa: E731
        return left

    def _multiplicative(self) -> Callable:
        left = self._unary()
        while (tok := self._peek()) is not None and tok[1] in ("*", "/"):
            self.i += 1
            right = self._unary()
            if tok[1] == "*":
                left = lambda t, a=left, b=right: a(t) * b(t)  # noqa: E731
            else:
                left = lambda t, a=left, b=right: a(t) / b(t)  # noqa: E731
        return left

    def _unary(self) -> Callable:
        tok = self._peek()
        if tok is not None and tok[1] in ("-", "+"):
            self.i += 1
            inner = self._unary()
            if tok[1] == "-":
                return lambda t, a=inner: -a(t)
            return inner
        return self._power()

    def _power(self) -> Callable:
        base = self._atom()
        tok = self._peek()
        if tok is not None and tok[1] == "^":
            self.i += 1
            # the exponent may carry its own sign: _t^-1
            exponent = self._unary_atom()
            nxt = self._peek()
            if nxt is not None and nxt[1] == "^":
                raise _texp_error(
                    self.text, "write a^b^c with parentheses, (a^b)^c or a^(b^c)"
                )
            return lambda t, a=base, b=exponent: np.power(a(t), b(t))
        return base

    def _unary_atom(self) -> Callable:
        tok = self._peek()
        if tok is not None and tok[1] in ("-", "+"):
            self.i += 1
            inner = self._unary_atom()
            return (lambda t, a=inner: -a(t)) if tok[1] == "-" else inner
        return self._atom()

    def _atom(self) -> Callable:
        tok = self._take()
        kind, val = tok
        if kind == "num":
            c = float(val)
            return lambda t, c=c: np.full(np.shape(t), c)
        if kind == "op" and val == "(":
            inner = self._comparison()
            self._take(")")
            return inner
        if kind == "name":
            if val == "_t":
                return lambda t: np.asarray(t, dtype=float)
            name = val.lower()
            if name in _FUNCS_1 or name in _FUNCS_2:
                self._take("(")
                args = [self._comparison()]
                while (nxt := self._peek()) is not None and nxt[1] == ",":
                    self.i += 1
                    args.append(self._comparison())
                self._take(")")
                if name in _FUNCS_1:
                    if len(args) != 1:
                        raise _texp_error(self.text, f"{name}() takes one argument")
                    f1 = _FUNCS_1[name]
                    return lambda t, a=args[0], f=f1: f(a(t))
                if len(args) != 2:
                    raise _texp_error(self.text, f"{name}() takes two arguments")
                f2 = _FUNCS_2[name]
                return lambda t, a=args[0], b=args[1], f=f2: f(a(t), b(t))
            raise _texp_error(
                self.text,
                f"{val!r} is not _t or a supported function (texp may only "
                "depend on analysis time)",
            )
        raise _texp_error(self.text, f"unexpected {val!r}")


def resolve_texp(texp: TexpType) -> tuple[Callable[[np.ndarray], np.ndarray], str]:
    """Return ``(g, label)`` for ``texp=``; ``None`` is Stata's ``_t``."""
    if texp is None:
        return (lambda t: np.asarray(t, dtype=float)), "_t"
    if callable(texp):
        label = getattr(texp, "__name__", "callable")
        return texp, f"<{label}>"
    if isinstance(texp, str):
        with np.errstate(all="ignore"):
            fn = _TexpParser(texp).parse()
        return fn, texp.strip()
    raise MethodIncompatibility(
        f"texp must be a string expression in _t or a callable; got {type(texp)!r}.",
        recovery_hint="Pass texp='ln(_t)' (for example) or texp=lambda t: np.log(t).",
        diagnostics={"texp_type": type(texp).__name__},
    )


def resolve_tvc(tvc: Any, data: pd.DataFrame) -> List[str]:
    """Normalise ``tvc=`` to a list of existing numeric columns."""
    names = [tvc] if isinstance(tvc, str) else list(tvc)
    if isinstance(tvc, str) and " " in tvc.strip():
        names = tvc.split()
    if not names:
        raise MethodIncompatibility(
            "tvc= is empty.",
            recovery_hint="Pass the covariate(s) whose effect varies with time, "
            "e.g. tvc=['race'].",
        )
    missing = [v for v in names if v not in data.columns]
    if missing:
        raise MethodIncompatibility(
            f"tvc= names columns that are not in the data: {missing}.",
            recovery_hint="tvc= takes column names of the data (as Stata's "
            "tvc(varlist)); build interactions or factor indicators as columns first.",
            diagnostics={"missing": missing},
        )
    if len(set(names)) != len(names):
        raise MethodIncompatibility(
            f"tvc= repeats a column: {names}.",
            recovery_hint="List each time-varying covariate once.",
        )
    return names


@dataclass
class TvcSplit:
    """Rows of the split data and their links to the input rows."""

    origin: np.ndarray  # input row of every piece
    start: np.ndarray
    stop: np.ndarray
    event: np.ndarray
    g: np.ndarray  # g(stop) on every piece


def split_at_failures(
    T: np.ndarray,
    E: np.ndarray,
    start: Optional[np.ndarray],
    strata: Optional[np.ndarray],
    g: Callable[[np.ndarray], Any],
    texp_label: str,
) -> TvcSplit:
    """Split ``(start, T]`` at the failure times of the row's stratum.

    Without ``start`` the first piece opens at ``-inf`` (every row is at
    risk from the origin, as in right-censored data).
    """
    n = len(T)
    s = np.full(n, -np.inf) if start is None else np.asarray(start, dtype=float)
    if strata is None:
        codes = np.zeros(n, dtype=np.int64)
        n_str = 1
    else:
        codes, uniq = pd.factorize(np.asarray(strata), sort=True)
        n_str = len(uniq)

    layout = []
    n_pieces = 0
    for c in range(n_str):
        rows = np.flatnonzero(codes == c)
        Ts, Es, ss = T[rows], E[rows], s[rows]
        F = np.unique(Ts[Es == 1])
        lo = np.searchsorted(F, ss, side="right")  # first failure > start
        hi = np.searchsorted(F, Ts, side="left")  # first failure >= stop
        counts = (hi - lo).clip(min=0) + 1
        n_pieces += int(counts.sum())
        layout.append((rows, Ts, Es, ss, F, lo, counts))
    if n_pieces > MAX_EPISODES:
        raise DataInsufficient(
            f"tvc= splits the data into {n_pieces:,} episodes (rows x failure "
            f"times inside their intervals), above the limit of {MAX_EPISODES:,}.",
            recovery_hint="The split grows with the number of distinct failure "
            "times: record time on a coarser grid (days -> months), which also "
            "creates the ties Stata data usually have, or fit on a subsample.",
            diagnostics={"n_episodes": n_pieces, "n_rows": n},
        )

    origin_parts: List[np.ndarray] = []
    start_parts: List[np.ndarray] = []
    stop_parts: List[np.ndarray] = []
    event_parts: List[np.ndarray] = []
    for rows, Ts, Es, ss, F, lo, counts in layout:
        total = int(counts.sum())
        first = np.cumsum(counts) - counts
        k = np.arange(total) - np.repeat(first, counts)  # piece within row
        rep_lo = np.repeat(lo, counts)
        last = k == np.repeat(counts - 1, counts)
        cut_stop = F[np.minimum(rep_lo + k, max(len(F) - 1, 0))] if len(F) else None
        stop = np.where(last, np.repeat(Ts, counts), cut_stop if len(F) else 0.0)
        cut_start = (
            F[np.clip(rep_lo + k - 1, 0, max(len(F) - 1, 0))] if len(F) else None
        )
        st = np.where(k == 0, np.repeat(ss, counts), cut_start if len(F) else 0.0)
        ev = np.where(last, np.repeat(Es, counts), 0.0)
        origin_parts.append(np.repeat(rows, counts))
        start_parts.append(st)
        stop_parts.append(stop)
        event_parts.append(ev)

    origin = np.concatenate(origin_parts)
    order = np.argsort(origin, kind="stable")  # pieces of a row stay together
    origin = origin[order]
    st = np.concatenate(start_parts)[order]
    stop = np.concatenate(stop_parts)[order]
    ev = np.concatenate(event_parts)[order]

    # g is only needed where it can matter: at the failure times. A piece
    # whose stop is not a failure time of its stratum is never at risk at
    # one, so a non-finite g there (ln(0) at a censoring time) is harmless.
    uniq_t, inv = np.unique(stop, return_inverse=True)
    with np.errstate(all="ignore"):
        g_u = np.asarray(g(uniq_t.copy()), dtype=float)
    if g_u.shape == ():
        g_u = np.full(uniq_t.shape, float(g_u))
    if g_u.shape != uniq_t.shape:
        raise MethodIncompatibility(
            f"texp({texp_label}) returned shape {g_u.shape} for {uniq_t.shape} times.",
            recovery_hint="A texp callable must map an array of times to an "
            "array of the same shape.",
        )
    g_rows = g_u[inv]
    fail_times = np.unique(T[E == 1])
    at_failure = np.isin(stop, fail_times)
    bad = at_failure & ~np.isfinite(g_rows)
    if bad.any():
        raise DataInsufficient(
            f"texp({texp_label}) is not finite at {int(np.unique(stop[bad]).size)} "
            f"failure time(s), e.g. t = {float(stop[bad][0])!r}.",
            recovery_hint="Choose a function of time defined at every failure "
            "time (ln(_t) needs _t > 0), or shift time, e.g. texp='ln(_t + 1)'.",
            diagnostics={"bad_times": np.unique(stop[bad])[:5].tolist()},
        )
    g_rows = np.where(np.isfinite(g_rows), g_rows, 0.0)
    return TvcSplit(origin=origin, start=st, stop=stop, event=ev, g=g_rows)


def tvc_labels(tvc: Sequence[str]) -> List[str]:
    """Names of the tvc coefficients: Stata's ``tvc:`` equation."""
    return [f"tvc:{v}" for v in tvc]
