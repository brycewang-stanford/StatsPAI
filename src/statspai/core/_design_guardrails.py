"""Design guardrails shared by the linear-regression entry points.

``sp.feols`` / ``sp.fepois`` / ``sp.regress`` / ``sp.hdfe_ols`` /
``sp.panel`` fit whatever they are given. Two designs make the numbers
they return answer a different question than the caller asked, and
neither shows in the output:

1. **Staggered adoption under two-way fixed effects.** A 0/1 treatment that
   switches on at different dates, with unit and time effects absorbed:
   the coefficient is a weighted average of 2x2 comparisons, some of which
   use already-treated units as controls (Goodman-Bacon 2021). With
   effects that change over time those comparisons pull the estimate away
   from any average treatment effect. The check reports the share of the
   Goodman-Bacon weight on those comparisons, computed from the cohort x
   period counts alone (no regression), which equals
   ``sp.bacon_decomposition(...)['already_treated_control_weight_share']``
   on a balanced panel.
2. **Treatment assigned at a level coarser than the standard errors.** A
   treatment constant within each state (or state x period) cell while
   the standard errors are heteroskedasticity-robust or clustered on
   individuals nested in states: the assignment is correlated within
   states and the standard error ignores it.

Each finding is a ``violations()`` entry (``model_info['design_violations']``
on results that have ``model_info``, the ``design_violations`` attribute
otherwise) with ``next_calls`` templates for the MCP layer, plus an
:class:`~statspai.exceptions.AssumptionWarning`. Estimates and standard
errors are never changed.

Only user-level calls are checked: a regression a StatsPAI estimator runs
internally (a public ``sp.*`` call inside another one) is skipped, so the
guardrails never fire on a design the user did not write. The same gate
drives the session multiplicity counter (:mod:`._spec_counter`).
"""

from __future__ import annotations

import os
import re
import sys
import warnings
from itertools import permutations
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from ._spec_counter import multiplicity_violation, record_specifications

__all__ = [
    "check_design",
    "record_search",
    "stored_design_violations",
    "bacon_forbidden_share",
    "DESIGN_VIOLATIONS_KEY",
    "FORBIDDEN_SHARE_MIN",
]

DESIGN_VIOLATIONS_KEY = "design_violations"

#: Share of Goodman-Bacon weight on already-treated controls from which the
#: staggered-adoption violation is reported. Non-staggered designs give 0.
FORBIDDEN_SHARE_MIN = 0.01

_MAX_TREATMENTS = 4
_MAX_GROUP_COLUMNS = 40
_PREFILTER_ROWS = 2_000
_TIME_NAMES = frozenset(
    {
        "year",
        "time",
        "period",
        "t",
        "date",
        "month",
        "quarter",
        "week",
        "day",
        "wave",
        "yr",
        "yq",
        "ym",
    }
)
_IDENT = re.compile(r"^[A-Za-z_][A-Za-z0-9_.]*$")
_C_TERM = re.compile(r"^C\(\s*([A-Za-z_][A-Za-z0-9_.]*)\s*(?:,.*)?\)$")
_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_.]*")

_PACKAGE = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) + os.sep
_AGENT = os.path.join(_PACKAGE, "agent") + os.sep
_SHIM = "<statspai frame-input shim"


# ---------------------------------------------------------------------- #
#  Call-site gate
# ---------------------------------------------------------------------- #


def _call_site() -> Tuple[bool, int]:
    """Whether this is a user-level call, and the warning ``stacklevel``.

    Walks out of the package. Every public ``sp.*`` function is wrapped by
    the frame-input shim, so two shim frames before the first frame outside
    the package mean the regression was called by another StatsPAI
    function. The MCP / agent layer counts as the user.
    """
    shims = 0
    level = 1
    frame = sys._getframe(1)
    while frame is not None:
        name = frame.f_code.co_filename
        if name.startswith(_SHIM):
            shims += 1
        else:
            path = os.path.abspath(name)
            if not path.startswith(_PACKAGE) or path.startswith(_AGENT):
                return shims <= 1, level
            level += 1
        frame = frame.f_back
    return shims <= 1, 1


# ---------------------------------------------------------------------- #
#  Formula reading
# ---------------------------------------------------------------------- #


def _split_top(text: str, sep: str) -> List[str]:
    out, depth, cur = [], 0, []
    for ch in text:
        if ch in "([":
            depth += 1
        elif ch in ")]":
            depth -= 1
        if ch == sep and depth == 0:
            out.append("".join(cur))
            cur = []
        else:
            cur.append(ch)
    out.append("".join(cur))
    return [t.strip() for t in out]


def _parse(
    formula: str, columns: Any, fe: Optional[Sequence[str]]
) -> Optional[Dict[str, Any]]:
    """Outcome, plain regressors, absorbed / dummied effects of ``formula``."""
    cols = set(map(str, columns))
    parts = _split_top(formula, "|")
    if len(parts) > 2 or "~" not in parts[0] or parts[0].count("~") != 1:
        return None  # IV or something this reader does not know
    lhs, rhs = (s.strip() for s in parts[0].split("~"))
    if lhs in cols:
        outcome = lhs
    else:
        names = [n for n in _NAME.findall(lhs) if n in cols]
        if len(names) != 1:
            return None
        outcome = names[0]
    regressors: List[str] = []
    mains: set = set()
    effects: List[str] = list(fe or [])
    for term in _split_top(rhs, "+"):
        if not term or term in ("0", "1", "-1"):
            continue
        m = _C_TERM.match(term)
        if m and m.group(1) in cols:
            effects.append(m.group(1))
        elif ":" in term or "*" in term:
            mains.update(n for n in _NAME.findall(term) if n in cols)
        elif _IDENT.match(term) and term in cols:
            regressors.append(term)
    if fe is None and len(parts) == 2:
        for term in _split_top(parts[1], "+"):
            if term in cols:
                effects.append(term)
    effects = [e for i, e in enumerate(effects) if e not in effects[:i]]
    return {
        "outcome": outcome,
        "regressors": regressors,
        "mains": mains,
        "fe": effects,
    }


def _cluster_columns(cluster: Any, vcov: Any, columns: Any) -> Optional[List[str]]:
    """The clustering columns, ``[]`` for HC / iid, ``None`` to skip the check."""
    cols = set(map(str, columns))
    spec = cluster
    if spec is None and isinstance(vcov, Mapping) and len(vcov) == 1:
        spec = next(iter(vcov.values()))
    if spec is None:
        if isinstance(vcov, str) and (
            vcov.lower() in ("hac", "nw", "newey-west", "dk", "ewc")
            or any(k in vcov.lower() for k in ("conley", "driscoll", "kraay"))
        ):
            return None
        return []
    if isinstance(spec, str):
        spec = [s.strip() for s in re.split(r"[+^]", spec) if s.strip()]
    try:
        names = [str(s) for s in spec]
    except TypeError:
        return None
    if not names or any(n not in cols for n in names):
        return None
    return names


def _se_label(cluster_cols: List[str], vcov: Any) -> str:
    if cluster_cols:
        return "clustered on " + " and ".join(f"'{c}'" for c in cluster_cols)
    if isinstance(vcov, str) and vcov.lower() in ("iid", "nonrobust", "classical"):
        return "classical (iid)"
    if vcov is None:
        return "not clustered"
    return "heteroskedasticity-robust, not clustered"


# ---------------------------------------------------------------------- #
#  Numerical pieces
# ---------------------------------------------------------------------- #


def _codes(col: pd.Series, sort: bool = False) -> Tuple[np.ndarray, int]:
    codes, uniques = pd.factorize(col, sort=sort)
    return codes.astype(np.int64, copy=False), len(uniques)


class _CodeCache:
    """Factorised columns of one frame, each computed once."""

    def __init__(self, frame: pd.DataFrame) -> None:
        self.frame = frame
        self._hit: Dict[str, Tuple[np.ndarray, int, bool]] = {}

    def get(self, name: str, sort: bool = False) -> Tuple[np.ndarray, int]:
        hit = self._hit.get(name)
        if hit is None or (sort and not hit[2]):
            codes, n = _codes(self.frame[name], sort=sort)
            hit = (codes, n, sort)
            self._hit[name] = hit
        return hit[0], hit[1]


def _pair(
    a: np.ndarray, n_a: int, b: np.ndarray, n_b: int
) -> Tuple[np.ndarray, int, bool]:
    """Codes of the (a, b) cells; dense (``a * n_b + b``) when that is small."""
    if n_a * n_b <= max(4 * len(a), 1 << 20):
        return a * n_b + b, n_a * n_b, True
    codes, n = _codes(pd.Series(a * n_b + b))
    return codes, n, False


def _binary(a: np.ndarray) -> bool:
    if a.dtype == bool:
        return bool(a.any() and not a.all())
    if a.dtype.kind not in "iuf":
        return False
    zero, one = a == 0, a == 1
    return bool((zero | one).all() and zero.any() and one.any())


def _within(
    cell: np.ndarray, n_cells: int, d: np.ndarray
) -> Tuple[bool, np.ndarray, np.ndarray]:
    """Whether ``d`` is constant in every cell, with per-cell ones and counts."""
    ones = np.bincount(cell, weights=d, minlength=n_cells)
    count = np.bincount(cell, minlength=n_cells)
    return bool(np.all((ones == 0) | (ones == count))), ones, count


def _nested(inner: np.ndarray, n_inner: int, outer: np.ndarray, n_outer: int) -> bool:
    """Each level of ``inner`` sits in a single level of ``outer``."""
    cell, size, dense = _pair(inner, n_inner, outer, n_outer)
    n_pairs = (
        int(np.count_nonzero(np.bincount(cell, minlength=size))) if dense else size
    )
    return n_pairs == n_inner


def _absorbing(
    d: np.ndarray, ucode: np.ndarray, n_u: int, tcode: np.ndarray, n_t: int
) -> Optional[np.ndarray]:
    """Per-row first-treated period code (``n_t`` = never), or ``None``.

    ``None`` unless ``d`` is 0 before and 1 from a unit's first treated
    period on, and switches within at least one unit.
    """
    on = d == 1
    first = np.full(n_u, n_t, dtype=np.int64)
    np.minimum.at(first, ucode[on], tcode[on])
    g = first[ucode]
    if not np.array_equal(on, tcode >= g):
        return None
    ones = np.bincount(ucode, weights=d, minlength=n_u)
    count = np.bincount(ucode, minlength=n_u)
    if not np.any((ones > 0) & (ones < count)):
        return None
    return g


def bacon_forbidden_share(g: np.ndarray, t: np.ndarray, n_t: int) -> Tuple[float, int]:
    """Goodman-Bacon weight share on already-treated controls.

    ``g`` is each row's first-treated period code (``n_t`` = never treated,
    ``0`` = treated from the first period), ``t`` its period code. The
    weights are those of R / Stata ``bacondecomp`` (and
    :func:`statspai.bacon_decomposition`), written on the cohort x period
    row counts: each comparison's weight depends only on the group sizes
    and treated shares inside its window, never on the outcome. Returns the
    normalised share of the "Later vs Earlier Treated" and "Later vs Always
    Treated" comparisons and the number of treatment cohorts.

    Every window starts or ends at a cohort's first period, so only the row
    counts of each cohort before each cohort date are needed: a K x (K + 1)
    table for K cohorts, whatever the number of periods.
    """
    present = np.zeros(n_t + 1, dtype=bool)
    present[g] = True
    levels = np.flatnonzero(present)  # sorted cohort dates; n_t = never
    k = len(levels)
    cuts = levels[levels < n_t]
    m = len(cuts)
    to_level = np.zeros(n_t + 1, dtype=np.int64)
    to_level[levels] = np.arange(k)
    to_bin = np.searchsorted(cuts, np.arange(n_t), side="right")
    tab = np.bincount(to_level[g] * (m + 1) + to_bin[t], minlength=k * (m + 1)).reshape(
        k, m + 1
    )
    # before[c, j]: rows of cohort c in periods before cuts[j]; column m: all
    before = np.concatenate(
        [np.cumsum(tab, axis=1)[:, :m], tab.sum(axis=1)[:, None]],
        axis=1,
    ).astype(float)
    pos = np.where(levels < n_t, np.searchsorted(cuts, levels), m)
    tot = before[:, m]
    own = before[np.arange(k), pos]  # rows of c before its own date
    ug = levels[:, None]  # comparison cohort (rows)
    tg = levels[None, :]  # treated cohort (columns)
    b_at_u = before[:, pos].T  # [a, b]: rows of b before a's date
    never_cmp = np.broadcast_to(ug == n_t, (k, k))
    early = (tg < ug) & ~never_cmp
    late = tg > ug
    with np.errstate(divide="ignore", invalid="ignore"):
        # treated-cohort rows and their treated share inside the window
        n_tr = np.where(early, b_at_u, np.where(late, tot[None, :] - b_at_u, tot))
        n_tr = np.where(never_cmp, tot[None, :], n_tr)
        tr_on = np.where(early, b_at_u - own[None, :], tot[None, :] - own[None, :])
        d_tr = tr_on / n_tr
        # comparison-cohort rows inside the window; treated share 0 when it
        # is the later cohort (window ends at its date), 1 when the earlier
        n_cmp = np.where(
            never_cmp,
            tot[:, None],
            np.where(early, own[:, None], tot[:, None] - own[:, None]),
        )
        n_cmp = np.broadcast_to(n_cmp, (k, k))
        n_all = n_tr + n_cmp
        sh = n_tr / n_all
        # bacondecomp's three weight formulas coincide once the comparison
        # cohort's treated share in its window is exactly 0 or 1, as it is
        # for an absorbing treatment: N^2 s (1 - s) D (1 - D)
        w = n_all**2 * sh * (1 - sh) * d_tr * (1 - d_tr)
    keep = (tg != ug) & (tg != n_t) & (tg != 0) & np.isfinite(w) & (np.abs(w) >= 1e-15)
    total = float(w[keep].sum())
    forbidden = float(w[keep & late].sum())
    n_cohorts = int(np.sum(levels < n_t))
    return (forbidden / total if total > 0 else 0.0), n_cohorts


def _time_rank(col: pd.Series, name: str) -> int:
    """Lower is more time-like; used only to break orientation ties."""
    if pd.api.types.is_datetime64_any_dtype(col) or isinstance(
        col.dtype, pd.PeriodDtype
    ):
        return 0
    return 1 if name.lower() in _TIME_NAMES else 2


# ---------------------------------------------------------------------- #
#  The two checks
# ---------------------------------------------------------------------- #


def _staggered(
    codes: _CodeCache,
    d: np.ndarray,
    fe: List[str],
    unit: Optional[str],
    time: Optional[str],
) -> Optional[Dict[str, Any]]:
    frame = codes.frame
    if unit and time:
        pairs = [(unit, time)] if unit in fe and time in fe else []
    else:
        pairs = sorted(permutations(fe, 2), key=lambda p: _time_rank(frame[p[1]], p[1]))
    found: List[Tuple[int, int, str, str, np.ndarray, np.ndarray, int]] = []
    for u, t in pairs:
        rank = _time_rank(frame[t], t)
        if found and rank > found[0][0]:
            break  # a more time-like orientation already fits
        try:
            tcode, n_t = codes.get(t, sort=True)
        except TypeError:
            continue
        ucode, n_u = codes.get(u)
        g = _absorbing(d, ucode, n_u, tcode, n_t)
        if g is not None:
            found.append((rank, n_t, u, t, g, tcode, n_t))
    if not found:
        return None
    found.sort(key=lambda r: (r[0], r[1]))
    _, _, u, t, g, tcode, n_t = found[0]
    share, n_cohorts = bacon_forbidden_share(g, tcode, n_t)
    return {"unit": u, "time": t, "share": share, "n_cohorts": n_cohorts}


def _adoption_dates(g: pd.Series, t: pd.Series, d: np.ndarray) -> bool:
    """Whether ``g`` holds adoption dates on ``t``'s scale for ``d``.

    True when ``d`` is 0 for the never-adopting levels (``g <= 0`` or after
    the last period) and otherwise depends on ``g`` only through ``t - g``:
    ``1{t >= g}`` and every event-time dummy ``1{t - g = k}``. Such a column
    groups units by when they were treated, so it is not an assignment unit,
    even where an event-time dummy is 0 for several cohorts and the cohorts
    no longer have distinct treatment paths.
    """
    if g.dtype.kind not in "iu" or t.dtype.kind not in "iuf":
        return False
    gv = g.to_numpy(dtype=np.float64)
    tv = t.to_numpy(dtype=np.float64)
    if not np.all(np.isfinite(tv)) or np.any(tv != np.round(tv)):
        return False
    t_min, t_max = float(tv.min()), float(tv.max())
    never = (gv <= 0) | (gv > t_max)
    if d[never].any() or never.all():
        return False
    dated = ~never
    if np.any(gv[dated] < t_min - (t_max - t_min)):
        return False  # not on the time axis (e.g. state codes vs years)
    diff, n_diff = _codes(pd.Series(tv[dated] - gv[dated]))
    return _within(diff, n_diff, d[dated])[0]


def _assignment_level(
    codes: _CodeCache,
    d: np.ndarray,
    candidates: List[str],
    time: Optional[str],
    cluster_cols: List[str],
) -> Optional[Dict[str, Any]]:
    """The coarsest column the treatment is assigned at, if SEs ignore it."""
    frame = codes.frame
    n = len(frame)
    tcode: Optional[np.ndarray] = None
    n_t = 1
    if time is not None:
        tcode, n_t = codes.get(time)
    # A necessary condition on a random subsample rules most columns out
    # before any full pass (a local generator: the global stream is untouched).
    m = min(n, _PREFILTER_ROWS)
    sample = np.sort(
        np.random.default_rng(20261019).choice(n, size=m, replace=False, shuffle=False)
    )
    d_s = d[sample]
    t_s = tcode[sample] if tcode is not None else None
    cl_codes = [codes.get(c) for c in cluster_cols]
    w = ((np.arange(n_t) + 1) * 0.6180339887498949) % 1.0 + 1.0
    best = None
    for name in candidates:
        if name == time:
            continue
        col = frame[name]
        if col.dtype.kind == "f":
            continue
        try:
            sc, sn = _codes(col.iloc[sample])
        except TypeError:  # unhashable cells: not a grouping column
            continue
        if sn < 3 or (sc < 0).any():
            continue  # fewer than 3 groups, or missing values
        if t_s is not None:
            sc, sn, dense = _pair(sc, sn, t_s, n_t)
            if dense:
                sn = int(np.count_nonzero(np.bincount(sc, minlength=sn)))
        if sn == m and m * m > 20 * n:
            # no two sampled rows share a cell: had the cells averaged two
            # rows or more, about m^2 / n > 20 pairs would
            continue
        if not _within(sc, int(sc.max()) + 1, d_s)[0]:
            continue
        gcode, n_g = codes.get(name)
        if gcode.min() < 0:
            continue  # missing values outside the subsample
        if tcode is None:
            cell, size, dense = gcode, n_g, True
        else:
            cell, size, dense = _pair(gcode, n_g, tcode, n_t)
        constant, ones, count = _within(cell, size, d)
        if not constant or n < 2 * int(np.count_nonzero(count)):
            continue  # varies within cells, or cells of about one row
        if tcode is None:
            n_paths = 2
        else:
            # one signature per level: the periods it is observed in and its
            # treatment in each (a missing period is not an untreated one)
            if dense:
                full = np.flatnonzero(count)
                g_c, t_c = full // n_t, full % n_t
                d_c = ones[full] / count[full]
            else:
                first = np.unique(cell, return_index=True)[1]
                g_c, t_c, d_c = gcode[first], tcode[first], d[first]
            sig = np.bincount(g_c, weights=w[t_c] * (d_c + 1.0), minlength=n_g)
            n_paths = len(np.unique(np.round(sig, 9)))
        if n_g <= n_paths:
            continue  # a cohort / treated-group column, not an assignment unit
        if time is not None and _adoption_dates(frame[name], frame[time], d):
            continue  # adoption dates: event-time dummies share all-0 paths
        if any(_nested(gcode, n_g, cc, nc) for cc, nc in cl_codes):
            continue  # clustered at this level or coarser
        if cl_codes and not any(
            nc > n_g and _nested(cc, nc, gcode, n_g) for cc, nc in cl_codes
        ):
            continue  # clusters cross the level (e.g. units x periods)
        if best is None or n_g < best["n_levels"]:
            best = {"level": name, "n_levels": int(n_g), "by_time": tcode is not None}
    return best


# ---------------------------------------------------------------------- #
#  Entry point
# ---------------------------------------------------------------------- #


def _q(name: str) -> str:
    return repr(str(name))


def check_design(
    result: Any,
    data: Any,
    *,
    formula: Any,
    entry: str,
    fe: Optional[Sequence[str]] = None,
    cluster: Any = None,
    vcov: Any = None,
    unit: Optional[str] = None,
    time: Optional[str] = None,
    weights: Any = None,
    poisson: bool = False,
) -> List[Dict[str, Any]]:
    """Run the design guardrails on a fitted regression.

    Records the findings on ``result`` (see the module docstring), warns,
    and returns them. Never changes an estimate. A no-op for calls made
    from inside another StatsPAI function and for inputs it cannot read.

    Parameters
    ----------
    result : fitted result
        ``EconometricResults`` (``model_info``) or any object that accepts
        a ``design_violations`` / ``multiplicity`` attribute.
    data : pandas.DataFrame
        The frame passed by the user (all columns, not the model frame).
    formula : str
        ``"y ~ x1 + x2"``, ``"y ~ x | fe1 + fe2"`` or with ``C(fe)`` terms.
    entry : str
        ``"sp.feols"`` etc., for messages and the refit template.
    fe : sequence of str, optional
        Absorbed effects when the formula does not carry them (``sp.panel``).
    cluster, vcov : optional
        As passed to the estimator (``cluster='state'``,
        ``vcov={'CRV1': 'state'}``, ``vcov='hetero'``, ...).
    unit, time : str, optional
        Panel identifiers when the caller knows them.
    weights : optional
        For the specification key only.
    poisson : bool
        The fit is a Poisson regression (changes the recommended estimator).
    """
    user, stacklevel = _call_site()
    if not user or not isinstance(data, pd.DataFrame) or not isinstance(formula, str):
        return []
    spec = _parse(formula, data.columns, fe)
    if spec is None:
        return []
    found: List[Dict[str, Any]] = []
    try:
        found = _run(spec, data, entry, cluster, vcov, unit, time, poisson)
    except (TypeError, ValueError, KeyError, IndexError, MemoryError) as exc:
        _put(result, "design_checks_skipped", f"{type(exc).__name__}: {exc}")
    n_used = _n_used(result)
    key = "|".join(
        str(x)
        for x in (
            entry,
            re.sub(r"\s+", "", formula),
            sorted(spec["fe"]),
            vcov if not isinstance(vcov, Mapping) else sorted(vcov.items()),
            cluster,
            weights if isinstance(weights, str) else (weights is not None),
            n_used,
        )
    )
    record = record_specifications(data, spec["outcome"], [key])
    _put(result, "multiplicity", record)
    extra = multiplicity_violation(record)
    if extra is not None:
        found.append(extra)
    if found:
        stored = _get_list(result)
        stored.extend(found)
        _put(result, DESIGN_VIOLATIONS_KEY, stored)
    _warn(found, entry, stacklevel + 1)  # + the frame of _warn itself
    return found


def record_search(result: Any, data: Any, outcome: str, keys: List[str]) -> None:
    """Record a batch of specifications (``sp.spec_curve``) on the counter.

    Stores the running count under ``multiplicity``; no violation, because
    a specification curve is the report of the search, not a hidden one.
    """
    if _call_site()[0] and isinstance(data, pd.DataFrame):
        _put(result, "multiplicity", record_specifications(data, outcome, keys))


def _run(
    spec: Dict[str, Any],
    data: pd.DataFrame,
    entry: str,
    cluster: Any,
    vcov: Any,
    unit: Optional[str],
    time: Optional[str],
    poisson: bool,
) -> List[Dict[str, Any]]:
    cols = set(map(str, data.columns))
    y = spec["outcome"]
    fe = [f for f in spec["fe"] if f in cols]
    cluster_cols = _cluster_columns(cluster, vcov, data.columns)
    used = [y] + spec["regressors"] + fe + (cluster_cols or [])
    used = [c for i, c in enumerate(used) if c not in used[:i]]
    frame = data
    mask = data[used].notna().all(axis=1).to_numpy()
    if not mask.all():
        frame = data.loc[mask]
    if len(frame) < 4:
        return []
    arrays = {c: frame[c].to_numpy() for c in spec["regressors"]}
    treatments = [
        c for c in spec["regressors"] if c not in spec["mains"] and _binary(arrays[c])
    ]
    # an explicit DiD column (did = treat * post) makes its factors controls
    for c in list(treatments):
        for a, b in permutations([t for t in treatments if t != c], 2):
            if np.array_equal(arrays[c], arrays[a] * arrays[b]):
                treatments = [t for t in treatments if t not in (a, b)]
                break
    treatments = treatments[:_MAX_TREATMENTS]
    codes = _CodeCache(frame)
    out: List[Dict[str, Any]] = []
    tool = entry.split(".")[-1]
    for name in treatments:
        d = arrays[name].astype(float)
        t_col = time if time in cols else None
        if len(fe) >= 2:
            st = _staggered(codes, d, fe, unit, time)
            if st is not None:
                t_col = st["time"]
                if st["share"] >= FORBIDDEN_SHARE_MIN:
                    out.append(_staggered_violation(name, y, st, poisson))
        if t_col is None:
            t_col = next((f for f in fe if str(f).lower() in _TIME_NAMES), None)
        if cluster_cols is None:
            continue
        candidates = list(fe) + [c for c in cluster_cols if c not in fe]
        skip = set(spec["regressors"]) | {y} | spec["mains"] | set(candidates)
        candidates += [
            c
            for c in data.columns[: _MAX_GROUP_COLUMNS * 4]
            if c not in skip and frame[c].dtype.kind != "f"
        ][:_MAX_GROUP_COLUMNS]
        level = _assignment_level(codes, d, candidates, t_col, cluster_cols)
        if level is not None:
            out.append(_cluster_violation(name, level, cluster_cols, vcov, tool, entry))
    return out


def _staggered_violation(
    d: str, y: str, st: Dict[str, Any], poisson: bool
) -> Dict[str, Any]:
    share, u, t = st["share"], st["unit"], st["time"]
    cols = {"y": y, "treat": d, "time": t, "id": u}
    call = f"sp.did(df, y={_q(y)}, treat={_q(d)}, time={_q(t)}, id={_q(u)})"
    alts = ["sp.callaway_santanna", "sp.did", "sp.sun_abraham", "sp.did_imputation"]
    if poisson:
        alts = ["sp.etwfe"] + alts
    return {
        "kind": "assumption",
        "severity": "warning",
        "test": "staggered_twfe",
        "value": round(float(share), 6),
        "threshold": FORBIDDEN_SHARE_MIN,
        "message": (
            f"'{d}' switches on at {st['n_cohorts']} different dates and the "
            f"two-way fixed-effects coefficient puts {share:.1%} of its "
            "Goodman-Bacon weight on comparisons that use already-treated "
            "units as controls. If the effect changes over time or across "
            "cohorts, the coefficient is not an average treatment effect."
            + (
                " (Weights of the linear TWFE regression on the same design; "
                "the Poisson fit has the same comparisons with other weights.)"
                if poisson
                else ""
            )
        ),
        "recovery_hint": (
            f"Use a heterogeneity-robust estimator: {call} runs Callaway-"
            "Sant'Anna on this 0/1 treatment; "
            f"sp.bacon_decomposition(df, y={_q(y)}, treat={_q(d)}, "
            f"time={_q(t)}, id={_q(u)}) shows the comparisons."
        ),
        "alternatives": alts,
        "details": {"unit": u, "time": t, "n_cohorts": st["n_cohorts"]},
        "python_call": call,
        "next_calls": [
            {
                "tool": "did",
                "arguments": dict(cols),
                "rationale": (
                    "Callaway-Sant'Anna ATT on the same 0/1 treatment "
                    "(sp.did routes staggered adoption there)."
                ),
            },
            {
                "tool": "bacon_decomposition",
                "arguments": dict(cols),
                "rationale": "Which 2x2 comparisons drive the TWFE estimate.",
            },
            {
                "tool": "callaway_santanna",
                "arguments": {"y": y, "t": t, "i": u},
                "rationale": (
                    f"Needs g: the first period with {d} == 1 for each unit "
                    "(0 for never treated)."
                ),
            },
        ],
    }


def _cluster_violation(
    d: str,
    level: Dict[str, Any],
    cluster_cols: List[str],
    vcov: Any,
    tool: str,
    entry: str,
) -> Dict[str, Any]:
    g = level["level"]
    where = f"each '{g}' x period cell" if level["by_time"] else f"each '{g}'"
    se = _se_label(cluster_cols, vcov)
    return {
        "kind": "inference",
        "severity": "warning",
        "test": "treatment_cluster_level",
        "value": {"assignment_level": g, "n_levels": level["n_levels"]},
        "threshold": None,
        "message": (
            f"'{d}' is constant within {where}: it is assigned at the '{g}' "
            f"level ({level['n_levels']} groups), but the standard errors are "
            f"{se}. They ignore the correlation of the treatment within "
            f"'{g}' and are typically far too small."
        ),
        "recovery_hint": (
            f"Refit with cluster={_q(g)} (or a wild cluster bootstrap when "
            f"there are few '{g}' groups)."
        ),
        "alternatives": [entry, "sp.wild_cluster_bootstrap"],
        "details": {"assignment_level": g, "standard_errors": se},
        "python_call": f"{entry}(..., cluster={_q(g)})",
        "next_calls": [
            {
                "tool": tool,
                "same_call": True,
                "drop_arguments": ["cluster", "vcov", "vce", "robust", "se_type"],
                "arguments": {"cluster": g},
                "rationale": f"Same model, standard errors clustered on '{g}'.",
            }
        ],
    }


# ---------------------------------------------------------------------- #
#  Storage and warnings
# ---------------------------------------------------------------------- #


def _n_used(result: Any) -> Any:
    info = getattr(result, "data_info", None)
    if isinstance(info, Mapping) and info.get("nobs") is not None:
        return info.get("nobs")
    return getattr(result, "n_obs", None)


def _put(result: Any, key: str, value: Any) -> None:
    mi = getattr(result, "model_info", None)
    if isinstance(mi, dict):
        mi[key] = value
        return
    try:
        setattr(result, key, value)
    except (AttributeError, TypeError):  # pragma: no cover - frozen result
        pass


def _get_list(result: Any) -> List[Dict[str, Any]]:
    return list(stored_design_violations(result))


def stored_design_violations(obj: Any) -> List[Dict[str, Any]]:
    """The guardrail violations recorded on a result (copies)."""
    mi = getattr(obj, "model_info", None)
    src = mi.get(DESIGN_VIOLATIONS_KEY) if isinstance(mi, Mapping) else None
    if src is None:
        src = getattr(obj, DESIGN_VIOLATIONS_KEY, None)
    if not isinstance(src, (list, tuple)):
        return []
    return [dict(v) for v in src if isinstance(v, Mapping)]


def _warn(found: List[Dict[str, Any]], entry: str, stacklevel: int) -> None:
    from ..exceptions import AssumptionWarning

    for v in found:
        if v.get("severity") not in ("warning", "error"):
            continue
        warnings.warn(
            AssumptionWarning(
                f"{entry}: {v['message']}",
                recovery_hint=v.get("recovery_hint", ""),
                diagnostics={"test": v.get("test"), "value": v.get("value")},
                alternative_functions=list(v.get("alternatives") or []),
            ),
            stacklevel=stacklevel,
        )
