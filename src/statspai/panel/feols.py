"""
Unified high-dimensional fixed-effects OLS estimator — ``sp.feols()``.

Mirrors R's ``fixest::feols`` and Stata's ``reghdfe`` top-level API:

    sp.feols("y ~ x1 + x2 | firm + year", data=df, cluster="firm")

Formula grammar
---------------
The pipe ``|`` separates regressors from absorbed fixed effects.

    "y ~ x1 + x2"                   → no FE, pure OLS (constant included)
    "y ~ x1 | firm"                 → firm FE absorbed
    "y ~ x1 + x2 | firm + year"     → two-way FE (firm, year)
    "y ~ x1 | firm + year + state"  → three-way FE

The same term grammar is accepted on **both** sides of ``|``:

    x            bare column
    c.x          explicit continuous marker
    i.f / i(f)   explicit categorical marker (base level omitted)
    a:b          interaction of a and b
    a*b          a + b + a:b (full factorial)
    f1^f2        interacted categorical — one group per level combination
    i.f#c.x      varying slope only        (Stata ``absorb(i.f#c.x)``)
    i.f##c.x     group intercepts + slope  (Stata ``absorb(i.f##c.x)``)
    f[[x]]       varying slope only        (fixest ``f[[x]]``)
    f[x]         group intercepts + slope  (fixest ``f[x]``)

Varying slopes
--------------
``i.f#c.x`` absorbs the columns ``x · 1[f = j]`` — one slope per level of
``f``, with no intercepts — matching Stata's ``#``. ``i.f##c.x`` also
absorbs the level dummies. Absorbing a slope term is equivalent by FWL to
putting the same columns on the right-hand side, and is verified against
``reghdfe`` in ``tests/test_hdfe_varying_slopes.py``.

A varying-slope term consumes ``G`` degrees of freedom (one per level),
not ``G - 1``: the slope columns do not contain the constant, so no level
is redundant against the intercept.

Anything else — arbitrary expressions such as ``np.log(x)`` or ``I(x**2)``
— is rejected with a message naming the offending term. Use ``sp.feols``
(pyfixest-backed) for the full Patsy/formulaic grammar.

Inference
---------
- Default: classical OLS SE (appropriate only when errors i.i.d.).
- ``cluster='firm'``: one-way cluster-robust (Liang-Zeger CR1).
- ``cluster=['firm', 'year']``: N-way CGM via inclusion-exclusion.
- ``cluster='firm', wild=True``: wild cluster bootstrap on top of CR1.

Returns a :class:`FEOLSResult` with coef / se / vcov / R²-within, plus
reference to the reusable ``Absorber`` (enables re-running on subsamples
or for event-study path estimation).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from itertools import combinations
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
from scipy import stats

from .._result_serialize import ResultProtocolMixin
from .hdfe import Absorber, SlopeSpec, _factorize_multi, absorb_ols


@dataclass
class FEOLSResult(ResultProtocolMixin):
    """Result of ``sp.feols()``.

    Attributes
    ----------
    params : pd.Series
        Coefficient estimates indexed by regressor name.
    std_errors : pd.Series
        Standard errors indexed by regressor name.
    vcov : np.ndarray
        Variance-covariance matrix of the coefficients.
    tvalues, pvalues : pd.Series
    conf_int_lower, conf_int_upper : pd.Series
    residuals : np.ndarray
        In-sample residuals (after FE absorption).
    fitted_within : np.ndarray
        Predicted values from X β (excludes FE contribution).
    n_obs : int
    n_singletons_dropped : int
    n_fe : List[int]
        Number of groups per absorbed FE dimension.
    dof_fe : int
        Degrees of freedom consumed by the FEs.
    df_resid : int
    r2_within : float
    se_type : str
        'iid' | 'cluster' | 'multiway_cluster' | 'wild_cluster'
    cluster_info : dict
        Metadata (cluster names, counts).
    formula : str
    absorber : Absorber
        Reusable absorber (includes ``keep_mask`` to subset rows).

    Examples
    --------
    >>> import statspai as sp
    >>> from statspai.panel.feols import feols
    >>> df = sp.mincer_wage_panel()
    >>> res = feols("log_wage ~ education + experience | period",
    ...             data=df, cluster="period")
    >>> type(res).__name__
    'FEOLSResult'
    >>> res.se_type
    'cluster'
    >>> bool(res.params["education"] > 0)
    True
    """

    params: pd.Series
    std_errors: pd.Series
    vcov: np.ndarray
    tvalues: pd.Series
    pvalues: pd.Series
    conf_int_lower: pd.Series
    conf_int_upper: pd.Series
    residuals: np.ndarray
    fitted_within: np.ndarray
    n_obs: int
    n_singletons_dropped: int
    n_fe: List[int]
    dof_fe: int
    df_resid: int
    r2_within: float
    se_type: str
    cluster_info: Dict[str, Any]
    formula: str
    absorber: Absorber
    converged: bool
    iters: int

    def summary(self) -> str:
        lines: List[str] = []
        lines.append(f"FEOLS (reghdfe-style)  |  {self.formula}")
        lines.append("=" * max(60, len(self.formula) + 25))
        lines.append(
            f"Obs: {self.n_obs:,d}   Singletons dropped: {self.n_singletons_dropped:,d}"
        )
        lines.append(
            f"Absorbed FE: groups={self.n_fe}   dof_fe={self.dof_fe}   "
            f"df_resid={self.df_resid}"
        )
        lines.append(f"R² (within) = {self.r2_within:.4f}")
        lines.append(f"SE type: {self.se_type}")
        if self.cluster_info:
            lines.append(f"Cluster info: {self.cluster_info}")
        lines.append("-" * 60)
        lines.append(
            f"{'Variable':<20}{'Estimate':>12}{'Std.Err':>12}{'t':>8}{'p':>10}"
        )
        lines.append("-" * 60)
        for name in self.params.index:
            b = self.params[name]
            se = self.std_errors[name]
            t = self.tvalues[name]
            p = self.pvalues[name]
            stars = "***" if p < 0.01 else "**" if p < 0.05 else "*" if p < 0.10 else ""
            lines.append(f"{name:<20}{b:>12.4f}{se:>12.4f}{t:>8.2f}{p:>10.4f}  {stars}")
        lines.append("=" * 60)
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()

    # Convenience: pandas-friendly attribute names
    @property
    def coef(self) -> pd.Series:
        return self.params

    @property
    def se(self) -> pd.Series:
        return self.std_errors


# ======================================================================
# Formula parsing
# ======================================================================

_FORMULA_RE = re.compile(
    r"""
    ^\s*(?P<lhs>[A-Za-z_][A-Za-z_0-9]*)\s*~\s*      # y ~
    (?P<rhs>[^|]*?)                                  # regressors
    (?:\|\s*(?P<fe>.*))?                             # optional | fe1 + fe2
    \s*$
    """,
    re.VERBOSE,
)

_NAME = r"[A-Za-z_][A-Za-z_0-9]*"

_SUPPORTED_SYNTAX = (
    "supported term syntax (both sides of '|'):\n"
    "  x                bare column\n"
    "  c.x              explicit continuous marker\n"
    "  i.f  / i(f)      explicit categorical marker\n"
    "  a:b              interaction of a and b\n"
    "  a*b              a + b + a:b (full factorial)\n"
    "  f1^f2            interacted categorical (combined group)\n"
    "  i.f#c.x          varying slope only      (Stata absorb(i.f#c.x))\n"
    "  i.f##c.x         group intercepts + slope (Stata absorb(i.f##c.x))\n"
    "  f[x]             group intercepts + slope (fixest f[x])\n"
    "  f[[x]]           varying slope only       (fixest f[[x]])"
)


@dataclass(frozen=True)
class _Atom:
    """One column reference inside an interaction, with its type marking."""

    name: str
    categorical: bool


@dataclass(frozen=True)
class _Term:
    """A single parsed formula term.

    ``kind='inter'`` covers bare columns, ``i.f``, ``a:b`` and ``f1^f2``
    uniformly: they are all interactions of one or more :class:`_Atom`.
    ``kind='slope'`` is a varying-slope term.
    """

    kind: str
    atoms: tuple = ()
    group: str = ""
    x: str = ""
    with_intercept: bool = False

    @property
    def columns(self) -> List[str]:
        if self.kind == "slope":
            return [self.group, self.x]
        return [a.name for a in self.atoms]

    @property
    def label(self) -> str:
        if self.kind == "slope":
            sep = "##" if self.with_intercept else "#"
            return f"i.{self.group}{sep}c.{self.x}"
        return ":".join(
            (f"i.{a.name}" if a.categorical else a.name) for a in self.atoms
        )

    @property
    def is_plain_name(self) -> bool:
        """True for a bare column name — the legacy-compatible fast path."""
        return (
            self.kind == "inter"
            and len(self.atoms) == 1
            and not self.atoms[0].categorical
        )


def _split_top(s: str, seps: str) -> List[str]:
    """Split ``s`` on any char in ``seps`` at bracket depth zero."""
    out: List[str] = []
    buf: List[str] = []
    depth = 0
    for ch in s:
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        if depth == 0 and ch in seps:
            out.append("".join(buf))
            buf = []
        else:
            buf.append(ch)
    out.append("".join(buf))
    return [t.strip() for t in out]


def _parse_atom(tok: str, side: str) -> _Term:
    """Parse one indivisible term (no top-level ``+``, ``*`` or ``:``)."""
    t = tok.strip()

    # --- varying slopes -----------------------------------------------------
    # Stata: i.f#c.x / i.f##c.x  (and the reversed c.x#i.f / c.x##i.f)
    m = re.match(rf"^i\.({_NAME})(##|#)c\.({_NAME})$", t)
    if m:
        return _Term(
            kind="slope",
            group=m.group(1),
            x=m.group(3),
            with_intercept=(m.group(2) == "##"),
        )
    m = re.match(rf"^c\.({_NAME})(##|#)i\.({_NAME})$", t)
    if m:
        return _Term(
            kind="slope",
            group=m.group(3),
            x=m.group(1),
            with_intercept=(m.group(2) == "##"),
        )
    # fixest: f[[x]] = slope only, f[x] = intercepts + slope
    m = re.match(rf"^({_NAME})\[\[({_NAME})\]\]$", t)
    if m:
        return _Term(kind="slope", group=m.group(1), x=m.group(2), with_intercept=False)
    m = re.match(rf"^({_NAME})\[({_NAME})\]$", t)
    if m:
        return _Term(kind="slope", group=m.group(1), x=m.group(2), with_intercept=True)

    # --- interacted categoricals: f1^f2^f3 ----------------------------------
    if "^" in t:
        parts = _split_top(t, "^")
        atoms = []
        for p in parts:
            mm = re.match(rf"^(?:i\.)?({_NAME})$", p)
            if not mm:
                raise ValueError(
                    f"feols: cannot parse {p!r} inside the interacted term "
                    f"{t!r} on the {side} of '|'. Operands of '^' must be "
                    f"column names.\n{_SUPPORTED_SYNTAX}"
                )
            atoms.append(_Atom(mm.group(1), True))
        return _Term(kind="inter", atoms=tuple(atoms))

    # --- single atoms -------------------------------------------------------
    m = re.match(rf"^i\.({_NAME})$", t) or re.match(rf"^i\(\s*({_NAME})\s*\)$", t)
    if m:
        return _Term(kind="inter", atoms=(_Atom(m.group(1), True),))
    m = re.match(rf"^c\.({_NAME})$", t)
    if m:
        return _Term(kind="inter", atoms=(_Atom(m.group(1), False),))
    m = re.match(rf"^({_NAME})$", t)
    if m:
        return _Term(kind="inter", atoms=(_Atom(m.group(1), False),))

    raise ValueError(
        f"feols: cannot parse the term {t!r} on the {side} of '|'.\n"
        f"{_SUPPORTED_SYNTAX}"
    )


def _parse_term(tok: str, side: str) -> List[_Term]:
    """Parse one ``+``-separated token, expanding ``*`` and ``:``."""
    t = tok.strip()

    # a*b*c -> all main effects and all interactions (R/Stata factorial).
    star = _split_top(t, "*")
    if len(star) > 1:
        if any(not p for p in star):
            raise ValueError(
                f"feols: malformed '*' interaction {t!r} on the {side} of "
                f"'|'.\n{_SUPPORTED_SYNTAX}"
            )
        base = [_parse_atom(p, side) for p in star]
        if any(b.kind == "slope" for b in base):
            raise ValueError(
                f"feols: '*' cannot combine varying-slope terms ({t!r} on the "
                f"{side} of '|'). Write the slope term as its own '+' term.\n"
                f"{_SUPPORTED_SYNTAX}"
            )
        out: List[_Term] = []
        for r in range(1, len(base) + 1):
            for combo in combinations(range(len(base)), r):
                atoms: tuple = sum((base[i].atoms for i in combo), ())
                out.append(_Term(kind="inter", atoms=atoms))
        return out

    # a:b -> a single interaction term
    colon = _split_top(t, ":")
    if len(colon) > 1:
        if any(not p for p in colon):
            raise ValueError(
                f"feols: malformed ':' interaction {t!r} on the {side} of "
                f"'|'.\n{_SUPPORTED_SYNTAX}"
            )
        parts = [_parse_atom(p, side) for p in colon]
        if any(p.kind == "slope" for p in parts):
            raise ValueError(
                f"feols: ':' cannot combine varying-slope terms ({t!r} on the "
                f"{side} of '|'). Write the slope term as its own '+' term.\n"
                f"{_SUPPORTED_SYNTAX}"
            )
        atoms = sum((p.atoms for p in parts), ())
        return [_Term(kind="inter", atoms=atoms)]

    return [_parse_atom(t, side)]


def _parse_side(s: str, side: str) -> List[_Term]:
    if not s.strip():
        return []
    terms: List[_Term] = []
    tokens = _split_top(s, "+")
    if any(not tok for tok in tokens):
        # An empty slot between '+' signs is a typo, not an empty side (an
        # entirely blank side short-circuits above). Do not silently drop it.
        raise ValueError(
            f"feols: dangling '+' in {s.strip()!r} on the {side} of '|'.\n"
            f"{_SUPPORTED_SYNTAX}"
        )
    for tok in tokens:
        if tok == "1":
            continue
        if tok == "0":
            raise ValueError(
                f"feols: '0' (suppress intercept) is not supported on the "
                f"{side} of '|'; the absorbed fixed effects already carry the "
                f"constant.\n{_SUPPORTED_SYNTAX}"
            )
        terms.extend(_parse_term(tok, side))
    return terms


def _parse_formula(formula: str) -> tuple[str, List[_Term], List[_Term]]:
    """Parse ``"y ~ rhs | fe"`` into a LHS name and two term lists.

    Both sides accept the same grammar (see ``_SUPPORTED_SYNTAX``). Bare
    column names parse to a single continuous :class:`_Atom` and are
    materialized through the legacy fast path, so plain formulas behave
    exactly as before.
    """
    m = _FORMULA_RE.match(formula)
    if not m:
        raise ValueError(
            f"feols: could not parse formula {formula!r}. Expected "
            f"'y ~ x1 + x2 | fe1 + fe2' with a single '~' and at most one "
            f"'|'.\n{_SUPPORTED_SYNTAX}"
        )
    lhs = m.group("lhs").strip()
    x_terms = _parse_side((m.group("rhs") or ""), "left")
    fe_terms = _parse_side((m.group("fe") or ""), "right")
    return lhs, x_terms, fe_terms


# ======================================================================
# Term materialization
# ======================================================================


def _dummies(values: pd.Series, drop_first: bool) -> tuple[np.ndarray, List[str]]:
    """Level indicators for ``values``, Stata-style (levels sorted, base first)."""
    levels = sorted(pd.unique(values.dropna()), key=lambda v: (str(type(v)), v))
    if drop_first:
        levels = levels[1:]
    cols = [(values == lv).to_numpy(dtype=np.float64) for lv in levels]
    names = [str(lv) for lv in levels]
    if not cols:
        return np.empty((len(values), 0)), []
    return np.column_stack(cols), names


def _materialize_rhs(
    df: pd.DataFrame, terms: List[_Term]
) -> tuple[np.ndarray, List[str]]:
    """Build the regressor matrix for the left side of ``|``.

    A bare column contributes itself. A categorical atom contributes
    base-omitted level indicators. An interaction multiplies its atoms
    together (expanding every categorical one). A varying-slope term
    contributes the same columns Stata's ``regress`` would build for
    ``i.f#c.x`` / ``i.f##c.x``.
    """
    # Fast path: all bare names -> exactly the legacy behaviour.
    if terms and all(t.is_plain_name for t in terms):
        names = [t.atoms[0].name for t in terms]
        return df[names].to_numpy(dtype=np.float64), names

    blocks: List[np.ndarray] = []
    names_out: List[str] = []
    for t in terms:
        if t.kind == "slope":
            g = df[t.group]
            x = df[t.x].to_numpy(dtype=np.float64)
            if t.with_intercept:
                # Stata i.f##c.x == i.f (base omitted) + c.x + i.f#c.x (base
                # omitted): the base level's slope is carried by c.x.
                dm, lv = _dummies(g, drop_first=True)
                blocks.append(dm)
                names_out += [f"{t.group}::{v}" for v in lv]
                blocks.append(x.reshape(-1, 1))
                names_out.append(t.x)
                blocks.append(dm * x[:, None])
                names_out += [f"{t.group}::{v}#{t.x}" for v in lv]
            else:
                dm, lv = _dummies(g, drop_first=False)
                blocks.append(dm * x[:, None])
                names_out += [f"{t.group}::{v}#{t.x}" for v in lv]
            continue

        block = np.ones((len(df), 1))
        labels = [""]
        for atom in t.atoms:
            if atom.categorical:
                dm, lv = _dummies(df[atom.name], drop_first=True)
                new_labels = [f"{atom.name}::{v}" for v in lv]
            else:
                dm = df[atom.name].to_numpy(dtype=np.float64).reshape(-1, 1)
                new_labels = [atom.name]
            # Column-wise (Khatri-Rao) product of what we have so far with
            # this atom's expansion.
            new_block = np.empty((len(df), block.shape[1] * dm.shape[1]))
            new_lab: List[str] = []
            k = 0
            for i in range(block.shape[1]):
                for j in range(dm.shape[1]):
                    new_block[:, k] = block[:, i] * dm[:, j]
                    k += 1
                    new_lab.append(
                        f"{labels[i]}:{new_labels[j]}" if labels[i] else new_labels[j]
                    )
            block, labels = new_block, new_lab
        blocks.append(block)
        names_out += labels

    if not blocks:
        return np.empty((len(df), 0)), []
    return np.column_stack(blocks), names_out


def _materialize_fe(
    df: pd.DataFrame, terms: List[_Term]
) -> tuple[Optional[np.ndarray], List[SlopeSpec], List[str]]:
    """Split absorbed terms into an FE label matrix and varying-slope specs.

    On the absorbed side every non-slope atom is categorical by
    construction — ``a:b`` and ``a^b`` both mean "one fixed effect per
    observed combination", matching ``fixest``'s ``fe1^fe2``.
    """
    fe_terms = [t for t in terms if t.kind != "slope"]
    slope_terms = [t for t in terms if t.kind == "slope"]

    fe_names = [t.label for t in fe_terms]
    fe_mat: Optional[np.ndarray]
    if not fe_terms:
        fe_mat = None
    elif all(len(t.atoms) == 1 for t in fe_terms):
        # Fast path: all single columns -> exactly the legacy behaviour.
        fe_mat = df[[t.atoms[0].name for t in fe_terms]].to_numpy()
        fe_names = [t.atoms[0].name for t in fe_terms]
    else:
        cols = []
        for t in fe_terms:
            if len(t.atoms) == 1:
                cols.append(df[t.atoms[0].name].to_numpy())
            else:
                # One absorbed group per distinct level combination. Uses
                # integer code combination, not string joining — see
                # hdfe._factorize_multi for why the latter is unsafe.
                cols.append(
                    _factorize_multi([df[a.name].to_numpy() for a in t.atoms])[0]
                )
        fe_mat = np.column_stack(cols)

    slopes = [
        SlopeSpec(
            group=df[t.group].to_numpy(),
            x=df[t.x].to_numpy(dtype=np.float64),
            with_intercept=t.with_intercept,
            name=t.label,
        )
        for t in slope_terms
    ]
    return fe_mat, slopes, fe_names


# ======================================================================
# feols
# ======================================================================


def feols(
    formula: str,
    data: pd.DataFrame,
    *,
    weights: Optional[Union[str, np.ndarray]] = None,
    cluster: Optional[Union[str, List[str]]] = None,
    se_type: Optional[str] = None,
    vce: Optional[str] = None,
    wild: bool = False,
    wild_n_boot: int = 999,
    wild_weight_type: str = "webb",
    wild_seed: Optional[int] = None,
    conley_lat: Optional[str] = None,
    conley_lon: Optional[str] = None,
    conley_cutoff: Optional[float] = None,
    alpha: float = 0.05,
    drop_singletons: bool = True,
    tol: float = 1e-8,
    maxiter: int = 10_000,
) -> FEOLSResult:
    """reghdfe-style OLS with high-dimensional fixed effects.

    Parameters
    ----------
    formula : str
        ``"y ~ x1 + x2 | fe1 + fe2 + fe3"``. The ``| fe...`` part is
        optional. Both sides accept bare names, ``c.x`` / ``i.f``,
        ``a:b``, ``a*b``, ``f1^f2`` and the varying-slope forms
        ``i.f#c.x`` / ``i.f##c.x`` / ``f[[x]]`` / ``f[x]`` — see the
        module docstring for the full grammar.
    data : DataFrame
    weights : str or ndarray, optional
        Observation weights. Column name or raw array.
    cluster : str or list, optional
        One-way or multi-way cluster column(s).
    se_type : {'iid', 'cluster', 'multiway_cluster', 'wild_cluster'}
        Override automatic inference of SE type. Usually inferred from
        ``cluster`` / ``wild``.
    vce : str, optional
        Canonical SE-menu keyword (matches ``sp.regress`` / ``sp.feols``):

        - ``"robust"`` / ``"hc1"`` — heteroskedasticity-robust on the
          FE-absorbed design with reghdfe's small-sample factor
          ``N/(N-k-df_a)``; matches Stata ``reghdfe ..., vce(robust)``.
        - ``"hc0"`` — no small-sample factor.
        - ``"CR2"`` / ``"CR3"`` / ``"jackknife"`` — Pustejovsky-Tipton (2018)
          bias-reduced cluster-robust on the within design (requires
          ``cluster=``, one-way); matches R ``clubSandwich::vcovCR(plm)``.
        - ``"conley"`` — Conley spatial HAC on the within design (requires
          ``conley_lat=/conley_lon=/conley_cutoff=``; Stata ``acreg`` planar
          distance convention).
        - ``"wild"`` — shorthand for ``wild=True`` (requires ``cluster=``).
    wild : bool, default False
        If True (and ``cluster`` is given), return wild-cluster-bootstrap
        p-values / CIs alongside classical cluster SE. Applied variable-
        by-variable. Only supported with a single cluster column.
    wild_n_boot : int
        Bootstrap replications.
    wild_weight_type : {'rademacher', 'webb', 'mammen'}
    wild_seed : int, optional
    conley_lat, conley_lon : str, optional
        Coordinate columns (decimal degrees) for ``vce="conley"``.
    conley_cutoff : float, optional
        Conley distance cutoff in km for ``vce="conley"``.
    alpha : float
    drop_singletons : bool
    tol, maxiter : convergence controls for the absorber.

    Returns
    -------
    FEOLSResult

    Examples
    --------
    Two-way fixed effects (firm and year) with cluster-robust SE. This
    function is exported at top level as :func:`statspai.hdfe_ols`.

    >>> import statspai as sp
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> firm_fe = rng.normal(0, 1, 40)
    >>> year_fe = rng.normal(0, 0.5, 6)
    >>> rows = []
    >>> for i in range(40):
    ...     for t in range(6):
    ...         educ = rng.normal(12, 2)
    ...         exper = rng.normal(10, 3)
    ...         lwage = (1.0 + 0.08 * educ + 0.02 * exper
    ...                  + firm_fe[i] + year_fe[t] + rng.normal(0, 0.3))
    ...         rows.append((i, t, lwage, educ, exper))
    >>> df = pd.DataFrame(rows, columns=['firm', 'year', 'lwage',
    ...                                  'educ', 'exper'])
    >>> res = sp.hdfe_ols("lwage ~ educ + exper | firm + year", data=df,
    ...                   cluster='firm')
    >>> sorted(res.coef.index.tolist())
    ['educ', 'exper']
    """
    # --- canonical vce= keyword (matches sp.regress / sp.feols) -------------
    _HC_VCE = {"robust": True, "hc_robust": True, "hc1": True, "hc0": False}
    _BR_VCE = {"cr2": 0.5, "cr3": 1.0, "jackknife": 1.0}
    _vce = vce.lower() if isinstance(vce, str) else None
    if _vce in ("wild", "wildbootstrap", "wild_cluster", "wcr", "boottest"):
        wild = True
        _vce = None
    if (
        _vce is not None
        and _vce not in _HC_VCE
        and _vce not in _BR_VCE
        and (_vce != "conley")
    ):
        raise ValueError(
            f"hdfe_ols vce={vce!r} not recognised; use 'robust'/'hc1'/'hc0', "
            "'CR2'/'CR3'/'jackknife', 'conley', or 'wild'."
        )
    if _vce is not None and weights is not None:
        raise ValueError(
            f"hdfe_ols vce={vce!r} does not support weights= — the extended "
            "SE menu is unweighted."
        )
    if _vce in _BR_VCE and (cluster is None or not isinstance(cluster, str)):
        raise ValueError(
            f"hdfe_ols vce={vce!r} requires cluster='<one column>' (one-way)."
        )
    if _vce == "conley" and (
        conley_lat is None or conley_lon is None or conley_cutoff is None
    ):
        raise ValueError(
            "hdfe_ols vce='conley' requires conley_lat=, conley_lon= and "
            "conley_cutoff= (km)."
        )

    lhs, x_terms, fe_terms = _parse_formula(formula)
    x_vars = [c for t in x_terms for c in t.columns]
    fe_vars = [t.label for t in fe_terms]

    # Collect all columns (y, x's, fe's, cluster, weight)
    cols = [lhs] + x_vars + [c for t in fe_terms for c in t.columns]
    if _vce == "conley":
        # Coordinates ride along the same dropna so they stay row-aligned.
        cols += [conley_lat, conley_lon]
    if cluster is not None:
        cluster_names = [cluster] if isinstance(cluster, str) else list(cluster)
        cols += cluster_names
    else:
        cluster_names = []
    w_col = None
    if isinstance(weights, str):
        cols.append(weights)
        w_col = weights

    df = data[list(dict.fromkeys(cols))].dropna().copy()
    if len(df) == 0:
        raise ValueError(
            "No non-missing rows remaining after dropna."
        )  # pragma: no cover

    y_arr = df[lhs].to_numpy(dtype=np.float64)
    if not x_terms:
        # Pure absorption (predict y from FE only) — trivial. Fit a constant.
        X_arr = np.ones((len(df), 1))
        x_names = ["_const"]
    else:
        X_arr, x_names = _materialize_rhs(df, x_terms)

    w_arr = None
    if w_col is not None:
        w_arr = df[w_col].to_numpy(dtype=np.float64)
    elif weights is not None:
        raw_w = np.asarray(weights, dtype=np.float64).ravel()
        if raw_w.size == len(data):
            w_arr = (
                pd.Series(raw_w, index=data.index)
                .loc[df.index]
                .to_numpy(dtype=np.float64)
            )
        elif raw_w.size == len(df):
            w_arr = raw_w
        else:
            raise ValueError(
                "weights array length must match the input data or the "
                f"post-dropna sample; got {raw_w.size}, expected {len(data)} "
                f"or {len(df)}."
            )

    if fe_terms:
        fe_mat, slope_specs, fe_vars = _materialize_fe(df, fe_terms)
    else:
        fe_mat, slope_specs = None, []
        # No FE -> fall back to plain OLS/WLS with intercept.
        if not x_terms:
            raise ValueError(
                "Need at least one regressor or one FE."
            )  # pragma: no cover
        if _vce is not None:
            raise ValueError(
                f"hdfe_ols vce={vce!r} needs at least one absorbed fixed "
                "effect; use sp.regress for the no-FE SE menu."
            )
        return _ols_no_fe(df, lhs, X_arr, x_names, w_arr, cluster_names, alpha, formula)

    cluster_arr = None
    if cluster_names:
        cluster_arr = [df[c].to_numpy() for c in cluster_names]
        if len(cluster_arr) == 1:
            cluster_arr = cluster_arr[0]

    result = absorb_ols(
        y=y_arr,
        X=X_arr,
        fe=fe_mat,
        weights=w_arr,
        cluster=cluster_arr,
        drop_singletons=drop_singletons,
        tol=tol,
        maxiter=maxiter,
        return_absorber=True,
        slopes=slope_specs,
    )

    coef = pd.Series(result["coef"], index=x_names, name="coef")
    se = pd.Series(result["se"], index=x_names, name="std_err")
    vcov = result["vcov"]

    df_resid = result["df_resid"]
    t_crit = stats.t.ppf(1 - alpha / 2, df_resid)
    t_stats = coef / se.replace(0, np.nan)
    pvals = pd.Series(
        2 * (1 - stats.t.cdf(np.abs(t_stats.fillna(0)), df_resid)),
        index=x_names,
    )
    ci_lo = coef - t_crit * se
    ci_hi = coef + t_crit * se

    # Inferred SE type
    if se_type is None:
        if cluster_names:
            se_type = "cluster" if len(cluster_names) == 1 else "multiway_cluster"
        else:
            se_type = "iid"
    cluster_info = {}
    if cluster_names:
        nested_fe_mask = list(result.get("nested_fe_in_cluster", []))
        nested_fe = [
            name for name, is_nested in zip(fe_vars, nested_fe_mask) if is_nested
        ]
        cluster_info = {
            "cluster": cluster_names,
            "n_clusters": [int(pd.Series(df[c]).nunique()) for c in cluster_names],
            "dof_fe_cluster": int(result.get("dof_fe_cluster", result["dof_fe"])),
            "nested_fe": nested_fe,
        }

    # Optional wild bootstrap
    if wild and cluster_names:
        if len(cluster_names) > 1:
            raise NotImplementedError(  # pragma: no cover
                "Wild cluster bootstrap with multi-way clustering is not yet supported."
            )
        from ..inference.wild_bootstrap import wild_cluster_bootstrap

        # Run per x_var on the absorbed model: we regress y_tilde on X_tilde.
        # Build a temporary DataFrame of within-residualized variables.
        ab = result["absorber"]
        mask = ab.keep_mask
        df_sub = df.iloc[mask].reset_index(drop=True)
        yw = ab.demean(
            df_sub[lhs].to_numpy(dtype=np.float64), copy=True, already_masked=True
        )
        Xw = ab.demean(
            _materialize_rhs(df_sub, x_terms)[0], copy=True, already_masked=True
        )
        cl_w = df_sub[cluster_names[0]].to_numpy()

        wild_data = pd.DataFrame(
            {"_y": yw, **{f"_x{i}": Xw[:, i] for i in range(Xw.shape[1])}, "_cl": cl_w}
        )
        p_wild: Dict[str, float] = {}
        ci_wild: Dict[str, tuple] = {}
        for i, name in enumerate(x_names):
            res_w = wild_cluster_bootstrap(
                wild_data,
                y="_y",
                x=[f"_x{j}" for j in range(Xw.shape[1])],
                cluster="_cl",
                test_var=f"_x{i}",
                h0=0.0,
                n_boot=wild_n_boot,
                weight_type=wild_weight_type,
                seed=wild_seed,
                alpha=alpha,
            )
            p_wild[name] = res_w["p_boot"]
            ci_wild[name] = res_w["ci_boot"]
        cluster_info["wild_p"] = p_wild
        cluster_info["wild_ci"] = ci_wild
        se_type = "wild_cluster"

    # --- extended vce menu on the FE-absorbed (within) design ---------------
    # HC (reghdfe convention), CR2/CR3/jackknife (clubSandwich plm), Conley
    # (acreg planar). Same verified estimators as sp.regress / sp.feols /
    # sp.panel — computed on the absorber's within-transformed design.
    if _vce is not None:
        from ..inference._psd import se_from_vcov
        from ..inference.jackknife import conley_vcov_matrix, cr_vcov_matrix

        ab = result["absorber"]
        mask = ab.keep_mask
        df_sub = df.iloc[mask].reset_index(drop=True)
        yw = ab.demean(
            df_sub[lhs].to_numpy(dtype=np.float64), copy=True, already_masked=True
        )
        Xw = ab.demean(
            _materialize_rhs(df_sub, x_terms)[0], copy=True, already_masked=True
        )
        n_w, k_w = Xw.shape

        if _vce in _HC_VCE:
            bread = np.linalg.inv(Xw.T @ Xw)
            e_w = yw - Xw @ (bread @ (Xw.T @ yw))
            meat = (Xw * e_w[:, None]).T @ (Xw * e_w[:, None])
            vc = bread @ meat @ bread
            if _HC_VCE[_vce]:  # HC1 with reghdfe's N/(N - k - df_a) factor
                vc = vc * (n_w / df_resid)
            vcov = vc
            se_type = f"hc_robust ({_vce}, reghdfe N/(N-k-df_a))"
            df_infer = df_resid
        elif _vce in _BR_VCE:
            codes = pd.Categorical(df_sub[cluster]).codes
            # small_sample=False matches clubSandwich CR2/CR3 for FE exactly.
            vcov = cr_vcov_matrix(
                Xw, yw, codes, power=_BR_VCE[_vce], small_sample=False
            )
            n_cl = int(codes.max()) + 1
            se_type = {
                "cr2": "CR2 cluster-robust (clubSandwich, Pustejovsky-Tipton)",
                "cr3": "CR3 cluster-robust (clubSandwich jackknife-type)",
                "jackknife": "CR3 cluster-robust (clubSandwich jackknife-type)",
            }[_vce]
            cluster_info = {"cluster": [cluster], "n_clusters": [n_cl]}
            df_infer = n_cl - 1
        else:  # conley
            vcov = conley_vcov_matrix(
                Xw,
                yw,
                df_sub[conley_lat].to_numpy(dtype=np.float64),
                df_sub[conley_lon].to_numpy(dtype=np.float64),
                float(conley_cutoff),
            )
            se_type = f"Conley spatial HAC (acreg planar, {conley_cutoff} km)"
            df_infer = n_w - k_w

        if _vce == "conley":
            # Kernel-weighted HAC is not PSD by construction; a negative
            # variance is estimator failure, not rounding, so it must not be
            # clamped to 0 (see inference/_psd.py).
            se = pd.Series(
                se_from_vcov(vcov, list(x_names), estimator=se_type),
                index=x_names,
                name="std_err",
            )
        else:
            se = pd.Series(
                np.sqrt(np.maximum(np.diag(vcov), 0)), index=x_names, name="std_err"
            )
        t_crit = stats.t.ppf(1 - alpha / 2, df_infer)
        t_stats = coef / se.replace(0, np.nan)
        pvals = pd.Series(
            2 * (1 - stats.t.cdf(np.abs(t_stats.fillna(0)), df_infer)),
            index=x_names,
        )
        ci_lo = coef - t_crit * se
        ci_hi = coef + t_crit * se

    return FEOLSResult(
        params=coef,
        std_errors=se,
        vcov=vcov,
        tvalues=t_stats.fillna(0.0),
        pvalues=pvals,
        conf_int_lower=ci_lo,
        conf_int_upper=ci_hi,
        residuals=result["resid"],
        fitted_within=result["fitted_within"],
        n_obs=result["n"],
        n_singletons_dropped=result["n_singletons_dropped"],
        n_fe=result["n_fe"],
        dof_fe=result["dof_fe"],
        df_resid=df_resid,
        r2_within=result["r2_within"],
        se_type=se_type,
        cluster_info=cluster_info,
        formula=formula,
        absorber=result["absorber"],
        converged=result["converged"],
        iters=result["iters"],
    )


# ======================================================================
# Fallback: no-FE path
# ======================================================================


def _ols_no_fe(
    df: pd.DataFrame,
    lhs: str,
    X_arr: np.ndarray,
    x_names: List[str],
    weights: Optional[np.ndarray],
    cluster_names: List[str],
    alpha: float,
    formula: str,
) -> FEOLSResult:
    """Plain OLS/WLS with intercept when no FE is absorbed."""
    y = df[lhs].to_numpy(dtype=np.float64)
    X = np.column_stack([np.ones(len(df)), X_arr])
    names = ["_const"] + list(x_names)
    n, k = X.shape

    w = None if weights is None else np.asarray(weights, dtype=np.float64).ravel()
    if w is not None:
        if w.size != n:
            raise ValueError(f"weights length {w.size} does not match n={n}.")
        if not np.all(np.isfinite(w)) or np.any(w < 0):
            raise ValueError("weights must be finite and non-negative.")
        if float(w.sum()) <= 0:
            raise ValueError("weights must have positive total mass.")

    if w is None:
        XtX = X.T @ X
        Xty = X.T @ y
    else:
        XtX = X.T @ (X * w[:, None])
        Xty = X.T @ (y * w)
    try:
        XtX_inv = np.linalg.inv(XtX)
        coef = XtX_inv @ Xty
    except np.linalg.LinAlgError:
        XtX_inv = np.linalg.pinv(XtX)
        coef = XtX_inv @ Xty
    resid = y - X @ coef
    df_resid = n - k
    ss_res_w = float((resid**2).sum()) if w is None else float((w * resid**2).sum())
    sigma2 = ss_res_w / df_resid

    if cluster_names:
        cl = (
            df[cluster_names[0]].to_numpy()
            if len(cluster_names) == 1
            else [df[c].to_numpy() for c in cluster_names]
        )
        if w is None:
            from ..inference.multiway_cluster import multiway_cluster_vcov

            vcov = multiway_cluster_vcov(X, resid, cl, df_adjust=True, n_params=k)
        else:
            from .hdfe import _cluster_sandwich

            vcov = _cluster_sandwich(
                X,
                resid,
                coef,
                XtX_inv,
                cl,
                df_resid=df_resid,
                weights=w,
                n_absorbed=k,
            )
        se_type = "cluster" if len(cluster_names) == 1 else "multiway_cluster"
    else:
        vcov = sigma2 * XtX_inv
        se_type = "iid"

    se = np.sqrt(np.maximum(np.diag(vcov), 0.0))
    t_stats = coef / np.where(se > 0, se, np.nan)
    t_crit = stats.t.ppf(1 - alpha / 2, df_resid)
    pvals = 2 * (1 - stats.t.cdf(np.abs(np.nan_to_num(t_stats)), df_resid))

    if w is None:
        y_bar = y.mean()
        ss_res = float(((y - X @ coef) ** 2).sum())
        ss_tot = float(((y - y_bar) ** 2).sum())
    else:
        y_bar = float(np.average(y, weights=w))
        ss_res = float((w * (y - X @ coef) ** 2).sum())
        ss_tot = float((w * (y - y_bar) ** 2).sum())
    r2 = 1 - ss_res / ss_tot if ss_tot > 0 else np.nan

    # Minimal Absorber stub (identity) — returned in the ``absorber``
    # field for API symmetry when the user asks for a no-FE regression.
    class _Identity:
        keep_mask = np.ones(n, dtype=bool)
        n_kept = n
        n_dropped = 0
        n_fe: list = []

    ab = _Identity()

    return FEOLSResult(
        params=pd.Series(coef, index=names),
        std_errors=pd.Series(se, index=names),
        vcov=vcov,
        tvalues=pd.Series(t_stats, index=names).fillna(0.0),
        pvalues=pd.Series(pvals, index=names),
        conf_int_lower=pd.Series(coef - t_crit * se, index=names),
        conf_int_upper=pd.Series(coef + t_crit * se, index=names),
        residuals=resid,
        fitted_within=X @ coef,
        n_obs=n,
        n_singletons_dropped=0,
        n_fe=[],
        dof_fe=0,
        df_resid=df_resid,
        r2_within=r2,
        se_type=se_type,
        cluster_info={"cluster": cluster_names} if cluster_names else {},
        formula=formula,
        absorber=ab,  # type: ignore[arg-type]
        converged=True,
        iters=0,
    )


hdfe_ols = feols  # alias for top-level namespace export (avoids collision
# with the pyfixest-backed ``sp.feols`` wrapper).


__all__ = ["feols", "hdfe_ols", "FEOLSResult"]
