"""Specification objects for dynamic-panel GMM.

A dynamic-panel GMM estimator is fully described by

1. a list of **regressor terms** — a variable plus a lag order, e.g.
   ``("k", 2)`` for ``L2.k``;
2. a list of **instrument blocks** — either *GMM-style* (a variable whose
   lagged levels form a block-diagonal set of moment conditions, one column
   per equation period × lag distance) or *standard/IV-style* (a single
   column, the transformed regressor itself).

Keeping the specification separate from the numerics is what lets
Arellano-Bond, Blundell-Bond system GMM, Anderson-Hsiao and collapsed
variants all be *configurations* of one estimator rather than separate
codepaths, and it is what makes difference-in-Hansen tests natural: drop a
block, rebuild, refit, difference the J statistics.

Lag syntax
----------
Regressor terms accept Stata's lag-operator spelling, so the canonical
Arellano-Bond (1991) employment equation is writable directly::

    x=["l(0/1).w", "l(0/2).k"]        ->  w, L.w, k, L.k, L2.k

Accepted forms (case-insensitive ``L``):

===============  ==========================================
``"k"``          contemporaneous
``"L.k"``        lag 1
``"L2.k"``       lag 2
``"L(2).k"``     lag 2
``"L(0/2).k"``   lags 0, 1, 2 (expands to three terms)
===============  ==========================================
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Tuple

__all__ = [
    "Term",
    "GMMBlock",
    "IVBlock",
    "parse_terms",
    "term_name",
    "normalize_lag_range",
]

# ``L``/``l`` optionally followed by a digit run or a parenthesised
# ``(a)`` / ``(a/b)`` range, then ``.``, then the variable name.
_LAG_RE = re.compile(
    r"""^\s*
        [Ll]
        (?:
            (?P<single>\d+)
          | \(\s*(?P<lo>\d+)\s*(?:/\s*(?P<hi>\d+)\s*)?\)
        )?
        \.
        (?P<var>\S+?)
        \s*$""",
    re.VERBOSE,
)


@dataclass(frozen=True)
class Term:
    """One regressor: ``var`` at lag ``lag`` (in *levels*, pre-transform)."""

    var: str
    lag: int = 0

    def __post_init__(self) -> None:
        if self.lag < 0:
            raise ValueError(f"negative lag {self.lag} for variable {self.var!r}")


def term_name(term: Term) -> str:
    """Stata-style display name: ``k``, ``L1.k``, ``L2.k``.

    ``L1.`` rather than Stata's bare ``L.`` so that the lagged-dependent
    rows (``L1.y``, ``L2.y``) and the lagged-covariate rows share one
    convention; this also preserves the pre-existing ``sp.xtabond`` labels.
    """
    return term.var if term.lag == 0 else f"L{term.lag}.{term.var}"


def parse_terms(items: Optional[Sequence[str]]) -> List[Term]:
    """Expand a list of lag-operator strings into concrete :class:`Term`s.

    Order is preserved and follows Stata: ``l(0/2).k`` yields ``k``,
    ``L1.k``, ``L2.k`` in that order.

    Raises
    ------
    ValueError
        On malformed lag syntax or a reversed range.  Failing loudly here
        matters: silently treating ``"L(0/2).k"`` as a column *name* would
        surface much later as a confusing KeyError.
    """
    out: List[Term] = []
    for raw in items or []:
        if not isinstance(raw, str):
            raise TypeError(
                f"regressor spec must be a string, got {type(raw).__name__}"
            )
        item = raw.strip()
        if not item:
            raise ValueError("empty regressor specification")
        m = _LAG_RE.match(item)
        if m is None:
            if "." in item and item[0] in "Ll":
                raise ValueError(
                    f"could not parse lag specification {raw!r}. Expected forms: "
                    "'k', 'L.k', 'L2.k', 'L(2).k', 'L(0/2).k'."
                )
            out.append(Term(item, 0))
            continue
        var = m.group("var")
        if m.group("single") is not None:
            out.append(Term(var, int(m.group("single"))))
        elif m.group("lo") is not None:
            lo = int(m.group("lo"))
            hi = int(m.group("hi")) if m.group("hi") is not None else lo
            if hi < lo:
                raise ValueError(
                    f"reversed lag range in {raw!r}: {lo}/{hi}. "
                    "Write the smaller lag first, e.g. 'L(0/2).k'."
                )
            out.extend(Term(var, lag) for lag in range(lo, hi + 1))
        else:
            out.append(Term(var, 1))  # bare 'L.k'
    return out


def normalize_lag_range(
    lag_range: Optional[Tuple[Optional[int], Optional[int]]],
    default_min: int,
    horizon: int,
) -> Tuple[int, int]:
    """Resolve a ``(min, max)`` instrument lag window.

    ``None`` for the minimum means "use the class default" (2 for
    endogenous, 1 for predetermined, 0 for strictly exogenous); ``None``
    for the maximum means "all available deeper lags", which is Stata's
    ``xtabond`` default and is represented as ``horizon``.
    """
    if lag_range is None:
        lo, hi = default_min, None
    else:
        lo, hi = lag_range
    lo = default_min if lo is None else int(lo)
    hi = int(horizon) if hi is None else int(hi)
    if lo < 0:
        raise ValueError(f"instrument lag minimum must be >= 0, got {lo}")
    if hi < lo:
        raise ValueError(f"instrument lag range ({lo}, {hi}) is empty (max < min).")
    return lo, hi


@dataclass
class GMMBlock:
    """Block-diagonal ("GMM-style") moment conditions for one variable.

    For the differenced equation at period ``p`` the block contributes the
    level ``var_{i, p-d}`` for every lag distance ``d`` in
    ``[lag_min, lag_max]`` that exists for that unit.

    ``collapse`` implements Roodman (2009): instead of one column per
    ``(p, d)`` pair — which grows as O(T²) and overfits the endogenous
    regressor — a single column per lag distance ``d``, summed over
    periods.  This is the standard remedy for instrument proliferation.

    ``equation`` selects which stacked equation the block instruments:
    ``'diff'`` (transformed) or ``'level'`` (Blundell-Bond system GMM,
    where the instrument is the *lagged difference* rather than the level).
    """

    var: str
    lag_min: int
    lag_max: int
    collapse: bool = False
    equation: str = "diff"
    label: str = ""

    def __post_init__(self) -> None:
        if self.equation not in ("diff", "level"):
            raise ValueError(
                f"GMMBlock.equation must be 'diff' or 'level', got {self.equation!r}"
            )
        if not self.label:
            tail = ", collapse" if self.collapse else ""
            self.label = (
                f"gmm({self.var}, {self.lag_min}/{self.lag_max}{tail}, "
                f"{self.equation})"
            )


@dataclass
class IVBlock:
    """A single "standard" instrument column built from a regressor term.

    In the differenced equation the column is ``Δ term``; in the level
    equation it is the level itself.  This is how strictly exogenous
    covariates and time dummies enter (Stata prints them under
    ``Standard:``).

    ``equation='both'`` — ``xtabond2``'s default for ``iv()`` — puts both
    into a *single* column, so the two equations share one combined moment
    condition rather than contributing two separate ones.  That is why
    ``xtabond2`` counts ``iv(w k)`` as 2 instruments in a system fit, not 4.
    """

    term: Term
    equation: str = "diff"
    label: str = ""

    def __post_init__(self) -> None:
        if self.equation not in ("diff", "level", "both"):
            raise ValueError(
                "IVBlock.equation must be 'diff', 'level' or 'both', got "
                f"{self.equation!r}"
            )
        if not self.label:
            prefix = {"diff": "D.", "level": "", "both": "D./L."}[self.equation]
            self.label = f"{prefix}{term_name(self.term)}"


@dataclass
class DynPanelSpec:
    """Everything the numerics need, resolved and validated."""

    y: str
    y_lags: int
    x_terms: List[Term] = field(default_factory=list)
    gmm_blocks: List[GMMBlock] = field(default_factory=list)
    iv_blocks: List[IVBlock] = field(default_factory=list)
    transform: str = "fd"
    level_equation: bool = False
    constant: bool = False

    @property
    def y_terms(self) -> List[Term]:
        return [Term(self.y, lag) for lag in range(1, self.y_lags + 1)]

    @property
    def regressor_terms(self) -> List[Term]:
        return self.y_terms + list(self.x_terms)

    @property
    def regressor_names(self) -> List[str]:
        return [term_name(t) for t in self.regressor_terms]
