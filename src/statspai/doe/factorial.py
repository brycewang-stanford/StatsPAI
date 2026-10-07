"""Factorial designs: ``sp.factorial_design``, ``sp.design_aberration``.

A two-level fractional factorial ``2^(k-m)`` assigns ``m`` of the ``k``
factors to interaction columns of a full factorial in the other ``k - m``.
Every such assignment (a *generator*) aliases effects with each other; the
*defining relation* lists the products of factors that equal the constant
column, the *resolution* is the length of its shortest word and the *word
length pattern* counts its words by length. Among fractions of a given
size the one whose word length pattern is smallest, compared entry by
entry from the shortest word on, has *minimum aberration*.
"""

from __future__ import annotations

import itertools
import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from .._result_serialize import ResultProtocolMixin
from ..exceptions import DataInsufficient, MethodIncompatibility, NumericalInstability

# combinations of generators times the defining words of each
_SEARCH_LIMIT = 4e9


def _letters(k: int) -> List[str]:
    alphabet = [c for c in "ABCDEFGHJKLMNOPQRSTUVWXYZ"]  # no I: it is the identity
    if k <= len(alphabet):
        return alphabet[:k]
    return [f"F{j + 1}" for j in range(k)]


def _word(mask: int, names: Sequence[str]) -> str:
    parts = [names[j] for j in range(len(names)) if mask >> j & 1]
    sep = "" if all(len(nm) == 1 for nm in names) else ":"
    return sep.join(parts)


def _parse_word(text: str, names: Sequence[str]) -> int:
    """Bit mask of the factors in ``'ABC'``, ``'A*B*C'`` or ``'A:B:C'``."""
    text = text.strip().lstrip("+")
    if not text:
        raise MethodIncompatibility("A generator has an empty side.")
    if re.search(r"[*:]", text):
        parts = [t.strip() for t in re.split(r"[*:]", text) if t.strip()]
    elif text in names:
        parts = [text]
    elif all(len(nm) == 1 for nm in names):
        parts = [c for c in text if not c.isspace()]
    else:
        raise MethodIncompatibility(
            f"Cannot read {text!r}: with factor names longer than one "
            "character write generators as 'E = A*B*C'."
        )
    mask = 0
    for part in parts:
        if part not in names:
            raise MethodIncompatibility(
                f"Generator names {part!r}, which is not a factor "
                f"({', '.join(names)})."
            )
        mask ^= 1 << list(names).index(part)
    return mask


def _span(words: Sequence[int]) -> List[int]:
    """All non-zero products of the given words."""
    out = [0]
    for w in words:
        out += [v ^ w for v in out]
    return [v for v in out if v]


def _wlp(words: Sequence[int], k: int) -> List[int]:
    counts = [0] * (k + 1)
    for w in _span(words):
        counts[bin(w).count("1")] += 1
    return counts[1:]


def _min_aberration(k: int, base: int) -> Tuple[List[int], List[int]]:
    """Generators (as masks over the base factors) with the smallest pattern."""
    m = k - base
    cand = [c for c in range(1, 1 << base) if bin(c).count("1") >= 2]
    if m > len(cand):
        raise MethodIncompatibility(
            f"{k} two-level factors do not fit in {1 << base} runs: a design "
            f"of that size holds at most {(1 << base) - 1}."
        )
    n_combo = math.comb(len(cand), m)
    if n_combo * float(1 << m) > _SEARCH_LIMIT:
        raise MethodIncompatibility(
            f"Searching the {n_combo:,} fractions of {k} factors in "
            f"{1 << base} runs is not attempted. Give the generators= of a "
            "tabulated design (Wu and Hamada 2021, Appendix 5A; Box, Hunter "
            "and Hunter 2005, Table 6.22)."
        )
    if n_combo > 2000:
        try:
            from ._factorial_core import search
        except ImportError:  # numba missing: only small searches in Python
            if n_combo > 30000:
                raise MethodIncompatibility(
                    f"Searching {n_combo:,} fractions needs numba, which could "
                    "not be imported. Give generators=."
                ) from None
        else:
            idx, pat = search(np.array(cand, dtype=np.int64), m, base, k)
            words = [cand[int(i)] | (1 << (base + j)) for j, i in enumerate(idx)]
            return words, [int(v) for v in pat]
    best: Optional[Tuple[List[int], List[int]]] = None
    for combo in itertools.combinations(cand, m):
        words = [c | (1 << (base + j)) for j, c in enumerate(combo)]
        pat = _wlp(words, k)
        if best is None or pat < best[1]:
            best = (words, pat)
    assert best is not None
    return best


_PB_ROWS = {
    12: "++-+++---+-",
    20: "++--++++-+-+----++-",
    24: "+++++-+-++--++--+-+----",
}


def _plackett_burman(n: int) -> np.ndarray:
    """Plackett-Burman design in ``n`` runs: cyclic shifts of a first row,
    closed by a row of minus signs. Checked for orthogonality on the spot."""
    row = np.array([1 if c == "+" else -1 for c in _PB_ROWS[n]], dtype=int)
    X = np.array([np.roll(row, i) for i in range(n - 1)] + [-np.ones(n - 1, dtype=int)])
    if not np.array_equal(X.T @ X, n * np.eye(n - 1, dtype=int)):
        raise NumericalInstability(  # pragma: no cover - guards the table
            f"The {n}-run Plackett-Burman array is not orthogonal."
        )
    return X


def _regular_columns(s: int, m: int) -> Tuple[np.ndarray, List[Tuple[int, ...]]]:
    """All ``(s^m - 1) / (s - 1)`` columns of the saturated regular design
    over the integers mod a prime ``s``: linear combinations of ``m`` base
    factors whose first non-zero coefficient is one."""
    grid = np.array(list(itertools.product(range(s), repeat=m)), dtype=int)
    coefs = [
        c
        for c in itertools.product(range(s), repeat=m)
        if any(c) and next(v for v in c if v) == 1
    ]
    coefs.sort(key=lambda c: (sum(1 for v in c if v), c[::-1]))
    cols = np.column_stack([(grid @ np.array(c)) % s for c in coefs])
    return cols, coefs


def _difference_matrix_6x3() -> np.ndarray:
    """A 6 x 6 matrix over the integers mod 3 in which the difference of
    any two columns takes each value twice (found by backtracking; the
    first row and column are zero)."""
    rows, cols = 6, 6
    D = np.zeros((rows, cols), dtype=int)
    options = list(itertools.product(range(3), repeat=rows - 1))

    def ok(col: np.ndarray, upto: int) -> bool:
        for j in range(upto):
            diff = (col - D[:, j]) % 3
            if not np.array_equal(np.bincount(diff, minlength=3), [2, 2, 2]):
                return False
        return True

    def fill(j: int) -> bool:
        if j == cols:
            return True
        for opt in options:
            col = np.r_[0, opt]
            if ok(col, j):
                D[:, j] = col
                if fill(j + 1):
                    return True
        return False

    if not fill(1):  # pragma: no cover - the matrix exists
        raise NumericalInstability("No difference matrix found.")
    return D


def _l18() -> np.ndarray:
    """The 18-run orthogonal array with one two-level and seven three-level
    columns (levels coded from 0), from a difference matrix: run (i, g)
    has the six-level factor at i and three-level columns D[i, j] + g; the
    six-level factor is then split into a two- and a three-level one."""
    D = _difference_matrix_6x3()
    runs = []
    for i in range(6):
        for g in range(3):
            runs.append([i // 3, i % 3] + [int((D[i, j] + g) % 3) for j in range(6)])
    return np.array(runs, dtype=int)


def _orthogonal_array(
    lev: List[int], n_runs: int, names: Sequence[str]
) -> Tuple[np.ndarray, str, List[str]]:
    """Level indices (from 0) of a fraction that is not a two-level regular
    one: a regular prime-level fraction, a Plackett-Burman design or L18."""
    k = len(lev)
    notes: List[str] = []
    s = lev[0]
    same = all(v == s for v in lev)
    if same and s == 2 and n_runs in _PB_ROWS:
        if k > n_runs - 1:
            raise MethodIncompatibility(
                f"A {n_runs}-run Plackett-Burman design holds at most "
                f"{n_runs - 1} factors."
            )
        X = (_plackett_burman(n_runs)[:, :k] + 1) // 2
        notes.append(
            "Non-regular design: main effects are orthogonal to each other "
            "and partially aliased with two-factor interactions."
        )
        return X, f"Plackett-Burman design, {n_runs} runs", notes
    if (
        n_runs == 18
        and sorted(lev)[-1] == 3
        and lev.count(2) <= 1
        and lev.count(3) <= 7
    ):
        if lev.count(2) + lev.count(3) != k:
            raise MethodIncompatibility(
                "L18 holds one 2-level and seven 3-level factors."
            )
        full = _l18()
        three = iter(range(1, 8))
        l18_cols = [0 if v == 2 else next(three) for v in lev]
        notes.append("Columns of the orthogonal array L18; strength two.")
        return full[:, l18_cols], "orthogonal array L18", notes
    prime = s in (2, 3, 5, 7)
    m = int(round(math.log(n_runs, s))) if same and prime and n_runs > 1 else 0
    if same and prime and s > 2 and s**m == n_runs and m >= 2:
        cols, coefs = _regular_columns(s, m)
        if k > cols.shape[1]:
            raise MethodIncompatibility(
                f"{n_runs} runs hold at most {cols.shape[1]} factors at {s} levels."
            )
        if k <= m:
            raise MethodIncompatibility(
                f"{k} factors at {s} levels in {n_runs} runs is not a fraction."
            )
        extra = list(range(m, cols.shape[1]))
        n_combo = math.comb(len(extra), k - m)
        chosen = list(range(k))
        if 1 < n_combo <= 1500 and n_runs <= 243:
            best = None
            for combo in itertools.combinations(extra, k - m):
                pick = list(range(m)) + list(combo)
                a = design_aberration(
                    pd.DataFrame(cols[:, pick]), max_order=min(k, 5)
                ).to_numpy()[:-2]
                key = tuple(np.round(a, 8))
                if best is None or key < best[0]:
                    best = (key, pick)
            assert best is not None
            chosen = best[1]
            notes.append("Minimum-aberration choice of columns, by search.")
        elif n_combo > 1:
            notes.append(
                "Too many column choices to search: the first columns of the "
                "saturated design are used."
            )
        gens = []
        for pos, c in enumerate(chosen[m:], start=m):
            terms = " + ".join(
                (f"{v}*" if v > 1 else "") + f"x{j + 1}"
                for j, v in enumerate(coefs[c])
                if v
            )
            gens.append(f"{names[pos]} = ({terms}) mod {s}")
        notes.append(
            "Generators on levels coded 0.." + str(s - 1) + ": " + "; ".join(gens)
        )
        return cols[:, chosen], f"{s}^({k}-{k - m}) regular fractional factorial", notes
    raise MethodIncompatibility(
        "Fractions are built for: two-level factors (regular in 2^m runs; "
        "Plackett-Burman in 12, 20 or 24 runs), factors that all have 3, 5 "
        "or 7 levels in a power of that many runs, and the 18-run array "
        "with one 2-level and up to seven 3-level factors. For other "
        "designs bring your own and check it with sp.design_aberration, or "
        "use sp.doe_optimal with candidates=."
    )


@dataclass
class FactorialDesignResult(ResultProtocolMixin):
    """A factorial design and its alias structure.

    Attributes
    ----------
    design : DataFrame
        One row per run in standard order (or run order when
        randomised). Two-level factors are coded -1 / +1 unless level
        values were given.
    kind : str
    generators : list of str
        For a fractional design, e.g. ``['E = ABCD']``.
    defining_relation : list of str
        Words equal to the identity, e.g. ``['ABCDE']``; a leading minus
        sign marks a word that equals minus the identity.
    resolution : int or None
        Length of the shortest word; ``None`` for a full factorial.
    word_length_pattern : list of int
        Number of words of length 1, 2, 3, ...
    aliases : dict
        Each main effect and two-factor interaction with the effects it
        cannot be separated from (up to ``alias_order`` factors). A
        leading minus sign means the two columns are opposite: the
        estimate is the first effect minus the second.
    model_info : dict

    Examples
    --------
    >>> import statspai as sp
    >>> d = sp.factorial_design(5, n_runs=16)
    >>> d.generators, d.resolution
    (['E = ABCD'], 5)
    """

    design: pd.DataFrame
    kind: str
    generators: List[str] = field(default_factory=list)
    defining_relation: List[str] = field(default_factory=list)
    resolution: Optional[int] = None
    word_length_pattern: List[Any] = field(default_factory=list)
    aliases: Dict[str, List[str]] = field(default_factory=dict)
    model_info: Dict[str, Any] = field(default_factory=dict)

    @property
    def n_runs(self) -> int:
        return int(self.design.shape[0])

    def to_frame(self) -> pd.DataFrame:
        """The runs as a DataFrame."""
        return self.design.copy()

    def summary(self) -> str:
        lines = [
            f"Factorial design: {self.kind}",
            "=" * 50,
            f"Runs: {self.n_runs}    Factors: {len(self.model_info['factors'])}",
        ]
        if self.generators:
            lines.append("Generators: " + ", ".join(self.generators))
            lines.append("Defining relation: I = " + " = ".join(self.defining_relation))
            lines.append(f"Resolution: {self.resolution}")
            pat = self.word_length_pattern
            lines.append(
                "Word length pattern (length 3 on): "
                + ", ".join(str(v) for v in pat[2:])
            )
            mixed = {k: v for k, v in self.aliases.items() if v}
            if mixed:
                lines.append("Aliased effects:")
                for k, v in mixed.items():
                    lines.append(f"  {k} = " + " = ".join(v))
            else:
                lines.append(
                    "No main effect or two-factor interaction is aliased with "
                    "another."
                )
        elif self.word_length_pattern:
            lines.append(f"Resolution: {self.resolution}")
            lines.append(
                "Generalized word length pattern (A1, A2, ...): "
                + ", ".join(f"{v:g}" for v in self.word_length_pattern)
            )
        for note in self.model_info.get("notes", []):
            lines.append(f"Note: {note}")
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return self.summary()


def factorial_design(
    factors: Any,
    levels: Any = 2,
    n_runs: Optional[int] = None,
    generators: Optional[Sequence[str]] = None,
    center_points: int = 0,
    replicates: int = 1,
    randomize: bool = False,
    seed: Optional[int] = None,
    alias_order: int = 2,
) -> FactorialDesignResult:
    """A full or two-level fractional factorial design.

    For an experiment that varies several factors at once: every
    combination of levels (a full factorial), or a fraction of the
    combinations chosen so that main effects and low-order interactions
    stay separable.

    Parameters
    ----------
    factors : int, list of str, or dict
        The number of factors (named ``A``, ``B``, ...), their names, or
        ``{name: [level values]}``. With a dict the design is written in
        those values and ``levels`` is ignored.
    levels : int or list of int, default 2
        Levels per factor.
    n_runs : int, optional
        Size of a fraction. Omitted, the full factorial. Available:
        two-level factors in a power of two (the minimum-aberration
        regular fraction) or in 12, 20 or 24 runs (Plackett-Burman);
        factors that all have 3, 5 or 7 levels in a power of that number
        (a regular fraction, e.g. four three-level factors in 9 runs);
        and 18 runs for one two-level and up to seven three-level factors
        (the orthogonal array L18).
    generators : list of str, optional
        Generators of a two-level fraction, e.g. ``['E = ABC', 'F =
        BCD']`` (or ``'E = A*B*C'`` when names are longer than one
        character). The factors on the left are the added ones; the
        others form the full factorial. A leading minus sign on the right
        (``'E = -ABC'``) picks the other half.
    center_points : int, default 0
        Runs at the centre added to a two-level design with numeric
        levels; they give a test for curvature and a pure-error variance.
    replicates : int, default 1
        Copies of the whole design.
    randomize : bool, default False
        Put the runs in random order (the order in which to run them).
        A ``std_order`` column keeps the standard order.
    seed : int, optional
    alias_order : int, default 2
        Largest interaction listed in ``aliases``.

    Returns
    -------
    FactorialDesignResult
        ``design``, ``generators``, ``defining_relation``, ``resolution``,
        ``word_length_pattern``, ``aliases``, ``summary()``.

    Notes
    -----
    The minimum-aberration search enumerates every choice of generators.
    It covers up to 15 factors in 32 runs and 12 in 64 (a few seconds at
    the upper end); beyond that give ``generators=`` from a published
    table. Designs found this way have the word length pattern of the
    designs returned by ``FrF2::FrF2`` in R; the generators themselves
    may differ, as minimum-aberration designs are unique only up to
    relabelling.

    For the fractions that are not two-level regular ones there is no
    defining relation in letters; ``word_length_pattern`` then holds the
    generalized pattern ``A1, A2, ...`` of ``sp.design_aberration`` and
    ``resolution`` the first non-zero entry. In a Plackett-Burman design
    every main effect is partially aliased with many two-factor
    interactions: it screens main effects and cannot separate
    interactions.

    Examples
    --------
    A half fraction of five factors in 16 runs:

    >>> import statspai as sp
    >>> d = sp.factorial_design(5, n_runs=16)
    >>> d.generators
    ['E = ABCD']
    >>> d.resolution
    5

    Seven factors in 8 runs, saturated, every two-factor interaction
    aliased with a main effect:

    >>> d = sp.factorial_design(7, n_runs=8)
    >>> d.resolution
    3

    A 2 x 3 full factorial in the units of the factors:

    >>> d = sp.factorial_design({"price": [9, 12], "ad": ["a", "b", "c"]})
    >>> d.design.shape
    (6, 2)

    References
    ----------
    box2005statistics; wu2021experiments; plackett1946design;
    joseph2025experimental
    """
    values: Optional[Dict[str, List[Any]]] = None
    if isinstance(factors, dict):
        names = [str(k) for k in factors]
        values = {str(k): list(v) for k, v in factors.items()}
        lev = [len(v) for v in values.values()]
        for nm, v in values.items():
            if len(v) < 2 or len(set(map(str, v))) != len(v):
                raise MethodIncompatibility(
                    f"Factor {nm!r} needs at least two distinct levels."
                )
    else:
        if isinstance(factors, (int, np.integer)) and not isinstance(factors, bool):
            names = _letters(int(factors))
        elif isinstance(factors, str):
            raise MethodIncompatibility(
                "factors is a number, a list of names or a {name: levels} dict."
            )
        else:
            names = [str(f) for f in factors]
        lev = (
            [int(levels)] * len(names)
            if isinstance(levels, (int, np.integer))
            else [int(v) for v in levels]
        )
    k = len(names)
    if k < 1 or len(set(names)) != k:
        raise MethodIncompatibility("Factor names must be non-empty and distinct.")
    if len(lev) != k or min(lev) < 2:
        raise MethodIncompatibility(
            "levels gives one number of levels (at least 2) per factor."
        )
    if replicates < 1 or center_points < 0:
        raise MethodIncompatibility(
            "replicates must be at least 1 and center_points non-negative."
        )
    full = int(np.prod([float(v) for v in lev]))
    two_level = all(v == 2 for v in lev)
    if generators is not None and len(generators) == 0:
        generators = None
    fractional = generators is not None or (n_runs is not None and n_runs != full)
    info: Dict[str, Any] = {"factors": names, "levels": lev, "notes": []}
    gen_text: List[str] = []
    relation: List[str] = []
    resolution: Optional[int] = None
    pattern: List[Any] = []
    levels_idx: np.ndarray
    aliases: Dict[str, List[str]] = {}
    power_of_two = n_runs is not None and n_runs >= 2 and (n_runs & (n_runs - 1)) == 0
    if fractional and generators is None and not (two_level and power_of_two):
        assert n_runs is not None
        if n_runs >= full:
            raise MethodIncompatibility(
                f"n_runs must be smaller than the full factorial ({full})."
            )
        levels_idx, kind, more = _orthogonal_array(lev, int(n_runs), names)
        info["notes"].extend(more)
        ab = design_aberration(pd.DataFrame(levels_idx), max_order=min(k, 5))
        pattern = [float(round(v, 10)) for v in ab.to_numpy()[:-2]]
        resolution = None if np.isnan(ab["resolution"]) else int(ab["resolution"])
        info["generalized_word_length"] = True
    elif fractional:
        if not two_level:
            raise MethodIncompatibility(
                "generators= are for two-level factors. Leave them out to get "
                "a three-, five- or seven-level regular fraction."
            )
        signs: Dict[int, int] = {}
        if generators is not None:
            added: List[int] = []
            words: List[int] = []
            for g in generators:
                if "=" not in str(g):
                    raise MethodIncompatibility(
                        f"A generator reads 'E = ABC'; got {g!r}."
                    )
                left, right = str(g).split("=", 1)
                neg = right.strip().startswith("-")
                lm = _parse_word(left, names)
                rm = _parse_word(right.strip().lstrip("-"), names)
                if bin(lm).count("1") != 1 or lm & rm:
                    raise MethodIncompatibility(
                        f"In {g!r} the left side must be one factor that does "
                        "not appear on the right."
                    )
                added.append(lm.bit_length() - 1)
                words.append(lm | rm)
                signs[lm.bit_length() - 1] = -1 if neg else 1
            if len(set(added)) != len(added):
                raise MethodIncompatibility("A factor has two generators.")
            base_idx = [j for j in range(k) if j not in added]
            for a, w in zip(added, words):
                if any((w >> b) & 1 for b in added if b != a):
                    raise MethodIncompatibility(
                        "Write each generator in terms of the factors that "
                        "have no generator."
                    )
            if n_runs is not None and n_runs != 1 << len(base_idx):
                raise MethodIncompatibility(
                    f"{len(generators)} generators give {1 << len(base_idx)} "
                    f"runs, not n_runs={n_runs}."
                )
            if len(_span(words)) != (1 << len(words)) - 1 or len(
                set(_span(words))
            ) != len(_span(words)):
                raise MethodIncompatibility("The generators are not independent.")
        else:
            assert n_runs is not None
            base = int(round(math.log2(n_runs))) if n_runs > 0 else 0
            if n_runs < 2 or 1 << base != n_runs or n_runs > full:
                raise MethodIncompatibility(
                    f"n_runs must be a power of two no larger than {full}; "
                    f"got {n_runs}."
                )
            words, _ = _min_aberration(k, base)
            base_idx = list(range(base))
            added = list(range(base, k))
            info["notes"].append("Minimum-aberration fraction found by search.")
        nb = len(base_idx)
        grid = np.array(list(itertools.product([-1, 1], repeat=nb)), dtype=int)
        coded = np.zeros((grid.shape[0], k), dtype=int)
        for pos, j in enumerate(base_idx):
            coded[:, j] = grid[:, pos]
        for a, w in zip(added, words):
            col = np.ones(grid.shape[0], dtype=int)
            for j in base_idx:
                if (w >> j) & 1:
                    col *= coded[:, j]
            coded[:, a] = signs.get(a, 1) * col
            rhs = _word(w & ~(1 << a), names)
            gen_text.append(f"{names[a]} = {'-' if signs.get(a, 1) < 0 else ''}{rhs}")
        # every defining word equals +1 or -1 on the fraction: the product
        # of the signs of the generators it is made of
        sign_of: Dict[int, int] = {0: 1}
        for a, w in zip(added, words):
            sign_of.update(
                {v ^ w: sg * signs.get(a, 1) for v, sg in list(sign_of.items())}
            )
        del sign_of[0]
        span = sorted(sign_of, key=lambda w: (bin(w).count("1"), w))

        def signed(mask: int, sign: int) -> str:
            return ("-" if sign < 0 else "") + _word(mask, names)

        relation = [signed(w, sign_of[w]) for w in span]
        pattern = _wlp(words, k)
        resolution = min(bin(w).count("1") for w in span)
        if resolution < 3:
            short = [_word(w, names) for w in span if bin(w).count("1") < 3]
            raise MethodIncompatibility(
                "These generators make main effects indistinguishable from "
                f"each other (defining word {short[0]}): the design has "
                f"resolution {resolution}."
            )
        effects = [
            sum(1 << j for j in combo)
            for order in range(1, min(alias_order, k) + 1)
            for combo in itertools.combinations(range(k), order)
        ]
        seen: set = set()
        for e in effects:
            if e in seen:
                continue
            # the column of e times a defining word is that of e ^ w, up to
            # the sign of the word
            mates = sorted(
                {
                    (e ^ w, sign_of[w])
                    for w in span
                    if bin(e ^ w).count("1") <= alias_order
                },
                key=lambda t: (bin(t[0]).count("1"), t[0]),
            )
            seen.update(m for m, _ in mates)
            aliases[_word(e, names)] = [signed(m, sg) for m, sg in mates]
        kind = f"2^({k}-{len(words)}) fractional factorial, resolution {resolution}"
        levels_idx = (coded + 1) // 2
    else:
        levels_idx = np.array(
            list(itertools.product(*[range(v) for v in lev])), dtype=int
        )
        kind = (
            f"2^{k} full factorial"
            if two_level
            else " x ".join(str(v) for v in lev) + " full factorial"
        )
    cols: Dict[str, Any]
    if values is not None:
        cols = {
            nm: [values[nm][i] for i in levels_idx[:, j]] for j, nm in enumerate(names)
        }
    elif two_level:
        cols = {nm: 2 * levels_idx[:, j] - 1 for j, nm in enumerate(names)}
    else:
        cols = {nm: levels_idx[:, j] + 1 for j, nm in enumerate(names)}
    design = pd.DataFrame(cols)
    design = pd.concat([design] * int(replicates), ignore_index=True)
    if center_points:
        if not two_level:
            raise MethodIncompatibility(
                "Centre points are defined for two-level designs."
            )
        try:
            centre = {nm: [float(design[nm].astype(float).mean())] for nm in names}
        except (TypeError, ValueError) as exc:
            raise MethodIncompatibility(
                "Centre points need numeric levels for every factor."
            ) from exc
        design = pd.concat(
            [design.astype(float), pd.DataFrame(centre).loc[[0] * center_points]],
            ignore_index=True,
        )
    if randomize:
        rng = np.random.default_rng(seed)
        order = rng.permutation(design.shape[0])
        design = design.iloc[order].reset_index(drop=True)
        design.insert(0, "std_order", order + 1)
    info.update(replicates=int(replicates), center_points=int(center_points))
    return FactorialDesignResult(
        design=design,
        kind=kind,
        generators=gen_text,
        defining_relation=relation,
        resolution=resolution,
        word_length_pattern=pattern,
        aliases=aliases,
        model_info=info,
    )


def design_aberration(
    design: pd.DataFrame,
    factors: Optional[Sequence[str]] = None,
    max_order: Optional[int] = None,
) -> pd.Series:
    """Generalized word length pattern of any factorial design.

    Measures how badly effects are confounded in a design with factors
    at any number of levels, regular or not: ``A_j`` is the total
    squared aliasing between the constant and all ``j``-factor
    interaction contrasts. ``A_1 = A_2 = 0`` means every pair of factors
    is balanced (an orthogonal array of strength two); ``A_3`` then
    measures the confounding of main effects with two-factor
    interactions. Between two designs of the same size the one with the
    smaller pattern, read from ``A_1`` on, is preferred.

    Parameters
    ----------
    design : DataFrame
        One row per run. Each column is treated as a categorical factor
        with the levels that appear in it.
    factors : list of str, optional
        Columns to use. Default: all.
    max_order : int, optional
        Largest ``j``. Default: ``min(number of factors, 6)``.

    Returns
    -------
    Series
        ``A1`` ... ``A<max_order>``, plus ``resolution`` (the first ``j``
        with ``A_j > 0``; NaN when all are zero) and ``strength``
        (``resolution - 1``).

    Notes
    -----
    Computed from the coincidences between runs (Xu and Wu 2001), which
    costs ``runs^2 * factors * max_order`` and never forms the contrast
    columns. For a regular two-level fraction it is the ordinary word
    length pattern. Reproduces ``DoE.base::GWLP`` in R.

    Examples
    --------
    >>> import statspai as sp
    >>> d = sp.factorial_design(4, generators=["D = ABC"]).design
    >>> a = sp.design_aberration(d)
    >>> [round(float(a[f"A{j}"]), 10) for j in (1, 2, 3, 4)]
    [0.0, 0.0, 0.0, 1.0]
    >>> int(a["resolution"])
    4

    References
    ----------
    xu2001generalized
    """
    if not isinstance(design, pd.DataFrame):
        design = pd.DataFrame(np.asarray(design))
        design.columns = [f"x{j + 1}" for j in range(design.shape[1])]
    cols = list(design.columns) if factors is None else list(factors)
    missing = [c for c in cols if c not in design.columns]
    if missing:
        raise MethodIncompatibility(
            f"Not in the design: {', '.join(map(str, missing))}."
        )
    N, p = design.shape[0], len(cols)
    if N < 2 or p < 1:
        raise DataInsufficient("A design needs at least two runs and one factor.")
    if design[cols].isna().any().any():
        raise MethodIncompatibility("The design has missing entries.")
    if N > 5000:
        raise MethodIncompatibility(
            f"{N} runs: the computation is quadratic in the number of runs."
        )
    top = min(p, 6) if max_order is None else int(max_order)
    if not 1 <= top <= p:
        raise MethodIncompatibility(f"max_order must be between 1 and {p}.")
    # E[j] holds the elementary symmetric polynomial of degree j of the
    # per-factor kernels s * 1[same level] - 1, for every pair of runs
    E = [np.ones((N, N))] + [np.zeros((N, N)) for _ in range(top)]
    for c in cols:
        codes = pd.factorize(design[c])[0]
        s = int(codes.max()) + 1
        if s < 2:
            raise MethodIncompatibility(f"Factor {c!r} has a single level.")
        K = s * (codes[:, None] == codes[None, :]).astype(float) - 1.0
        for j in range(top, 0, -1):
            E[j] += E[j - 1] * K
    out = {f"A{j}": float(E[j].sum()) / N**2 for j in range(1, top + 1)}
    res = next((j for j in range(1, top + 1) if out[f"A{j}"] > 1e-9), None)
    out["resolution"] = float(res) if res is not None else float("nan")
    out["strength"] = float(res - 1) if res is not None else float("nan")
    return pd.Series(out, name="aberration")
