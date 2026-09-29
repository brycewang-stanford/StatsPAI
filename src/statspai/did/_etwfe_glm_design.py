"""Design matrix of the nonlinear ETWFE (``sp.etwfe(family=...)``).

Mirrors Stata ``jwdid`` (Rios-Avila): which cohort x period cells get a
treatment coefficient (``cgroup``), which cells *share* one (``hettype``)
and how covariates moderate the effect (``xvar``).

``hettype`` pools the ``(g, t)`` cells into parameters:

=================  =========================  ==============================
``hettype``        parameter of cell (g, t)   covariates demeaned within
=================  =========================  ==============================
``'timecohort'``   ``(g, t)`` (saturated)     ``(g, t)`` (never: ``(0, t)``)
``'time'``         ``(t, post)``              ``(t, post3)``
``'cohort'``       ``(g, post)``              ``(g, post3)``
``'event'``        ``e = t - g``              ``e`` (never: ``e = -gap``)
``'twfe'``         ``post``                   ``post3``
=================  =========================  ==============================

``post3`` is jwdid's ``__post__``: 0 for never-treated rows and the
reference period ``g - 1``, 1 for earlier pre-periods, 2 for ``t >= g``.
Pre-period parameters exist only under ``cgroup='nevertreated'``.

A covariate ``x`` moderates each treatment parameter ``p`` as
``D_p * (x - mean_group(x))``, demeaned within the ``hettype`` groups over
the estimation sample (the cohort x period cells by default, as both
references do).  The remaining covariate terms follow the reference each
``fe`` mode reproduces:

* ``fe='unit'`` (Stata ``jwdid y x``): ``1{T = t} * x`` with the raw
  covariate for every period but the first, and -- when ``x`` varies within
  units -- ``x`` itself plus ``1{G = g} * x`` for every treated cohort;
* ``fe='cohort'`` (R ``etwfe(xvar = x)``): ``1{T = t} * (x - mean_group(x))``
  for every period but the first, nothing else.  A categorical covariate (pandas ``category``
/ ``object`` / ``bool`` dtype) becomes dummies for every level but the
first, as Stata's ``i.var``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from ..exceptions import MethodIncompatibility

HETTYPES = ("timecohort", "time", "cohort", "event", "twfe")
_HETTYPE_ALIASES = {
    None: "timecohort",
    "timecohort": "timecohort",
    "cohorttime": "timecohort",
    "full": "timecohort",
    "saturated": "timecohort",
    "time": "time",
    "calendar": "time",
    "cohort": "cohort",
    "group": "cohort",
    "event": "event",
    "twfe": "twfe",
}


def normalise_hettype(hettype: Optional[str]) -> str:
    """Validate ``hettype=`` (jwdid's ``hettype()``)."""
    key = hettype if hettype is None else str(hettype).strip().lower()
    if key not in _HETTYPE_ALIASES:
        raise MethodIncompatibility(
            f"hettype={hettype!r} is not recognised; use one of {list(HETTYPES)}.",
            recovery_hint="hettype='timecohort' (default) is the saturated "
            "cohort x period design; 'time', 'cohort', 'event' and 'twfe' "
            "share one coefficient across the cells named, as Stata jwdid "
            "hettype().",
            diagnostics={"hettype": hettype},
        )
    return _HETTYPE_ALIASES[key]


def is_categorical(s: pd.Series) -> bool:
    """Categorical covariate: pandas category, object, string or bool."""
    return bool(
        isinstance(s.dtype, pd.CategoricalDtype)
        or s.dtype == object
        or pd.api.types.is_string_dtype(s.dtype)
        or pd.api.types.is_bool_dtype(s.dtype)
    )


def _param_key(g: float, t: float, hettype: str) -> Tuple[Any, ...]:
    post = bool(t >= g)
    if hettype == "timecohort":
        return ("gt", int(g), int(t))
    if hettype == "time":
        return ("t", post, int(t))
    if hettype == "cohort":
        return ("g", int(g), post)
    if hettype == "event":
        return ("e", int(t - g))
    return ("post", post)


def _param_label(key: Tuple[Any, ...]) -> str:
    kind = key[0]
    if kind == "gt":
        return f"treat[{key[1]},{key[2]}]"
    if kind == "t":
        return f"treat[{'post' if key[1] else 'pre'},t={key[2]}]"
    if kind == "g":
        return f"treat[g={key[1]},{'post' if key[2] else 'pre'}]"
    if kind == "e":
        return f"treat[e={key[1]}]"
    return f"treat[{'post' if key[1] else 'pre'}]"


def _demean_groups(
    ft: np.ndarray, tt: np.ndarray, hettype: str, gap: float
) -> np.ndarray:
    """jwdid's ``toabshere`` groups: the cells covariates are demeaned in."""
    treated = np.isfinite(ft)
    g0 = np.where(treated, ft, 0.0)
    post3 = np.zeros(len(tt), dtype=np.int8)
    post3[treated & (tt < g0 - gap)] = 1
    post3[treated & (tt >= g0)] = 2
    if hettype == "timecohort":
        keys = pd.MultiIndex.from_arrays([g0, tt])
    elif hettype == "time":
        keys = pd.MultiIndex.from_arrays([tt, post3])
    elif hettype == "cohort":
        keys = pd.MultiIndex.from_arrays([g0, post3])
    elif hettype == "event":
        keys = pd.Index(np.where(treated, tt - g0, -gap))
    else:
        keys = pd.Index(post3)
    return pd.factorize(keys)[0]


def build_glm_design(
    df: pd.DataFrame,
    *,
    group: str,
    time: str,
    cohorts: List[float],
    periods: List[float],
    cgroup: str,
    hettype: str,
    fe_mode: str,
    xvar: Optional[List[str]],
    controls: Optional[List[str]],
) -> Dict[str, Any]:
    """Columns, cells and bookkeeping of the nonlinear ETWFE design.

    ``df`` carries ``_ft`` (first treatment, NaN for never treated) and is
    already restricted to the estimation sample.  Returns the design
    matrix and, for the aggregation step, each row's cell and covariate
    level, each cell's parameter, and the indices of the treatment columns
    (the ones set to zero for the untreated counterfactual).
    """
    ft = df["_ft"].to_numpy(dtype=float)
    tt = df[time].to_numpy(dtype=float)
    n = len(df)

    # ── cells (g, t) that carry a treatment effect ─────────────────────
    cells: List[Tuple[int, int]] = []
    cell_post: List[bool] = []
    for g_val in cohorts:
        pre = [p for p in periods if p < g_val]
        ref = pre[-1] if pre else None
        for t_val in periods:
            if cgroup == "notyet":
                if t_val < g_val:
                    continue
            elif t_val == ref:
                continue
            cells.append((int(g_val), int(t_val)))
            cell_post.append(bool(t_val >= g_val))
    cell_index = {c: i for i, c in enumerate(cells)}
    row_cell = np.full(n, -1, dtype=int)
    treated = np.isfinite(ft)
    for i in np.flatnonzero(treated):
        row_cell[i] = cell_index.get((int(ft[i]), int(tt[i])), -1)
    present = np.bincount(row_cell[row_cell >= 0], minlength=len(cells)) > 0
    keep_cells = [i for i in range(len(cells)) if present[i]]
    remap = np.full(len(cells), -1, dtype=int)
    remap[keep_cells] = np.arange(len(keep_cells))
    cells = [cells[i] for i in keep_cells]
    cell_post = [cell_post[i] for i in keep_cells]
    row_cell = np.where(row_cell >= 0, remap[np.maximum(row_cell, 0)], -1)

    # ── parameters: cells sharing a coefficient ────────────────────────
    param_keys: List[Tuple[Any, ...]] = []
    param_of_key: Dict[Tuple[Any, ...], int] = {}
    cell_param = np.empty(len(cells), dtype=int)
    for i, (g_val, t_val) in enumerate(cells):
        key = _param_key(g_val, t_val, hettype)
        if key not in param_of_key:
            param_of_key[key] = len(param_keys)
            param_keys.append(key)
        cell_param[i] = param_of_key[key]
    row_param = np.where(row_cell >= 0, cell_param[np.maximum(row_cell, 0)], -1)

    # ── covariates: columns, levels for by-level aggregation ──────────
    x_cols: List[np.ndarray] = []
    x_names: List[str] = []
    # jwdid's own covariate terms (i.tvar#c.x, ...) expand a factor as
    # levels 1..L-1: ppmlhdfe omits the last one, collinear with the
    # absorbed period effects.  The fit keeps levels 2..L (same column
    # span, so the same fit); the jwdid expansion is returned alongside
    # because Stata margins holds the absorbed period effects fixed, and
    # then the parametrisation matters (response_se='margins').
    xj_cols: List[np.ndarray] = []
    xj_names: List[str] = []
    level_codes = np.zeros(n, dtype=int)
    level_labels: List[Any] = []
    level_var: Optional[str] = None
    cat_vars = []
    for xv in xvar or []:
        s = df[xv]
        if is_categorical(s):
            cat = pd.Categorical(s)
            cats = list(cat.categories)
            if len(cats) < 2:
                raise MethodIncompatibility(
                    f"xvar {xv!r} has a single level; no heterogeneity to model.",
                    recovery_hint="Drop it from xvar.",
                    diagnostics={"xvar": xv},
                )
            codes = cat.codes
            for k, lev in enumerate(cats[1:], start=1):
                x_cols.append((codes == k).astype(float))
                x_names.append(f"{xv}[{lev}]")
            for k, lev in enumerate(cats[:-1]):
                xj_cols.append((codes == k).astype(float))
                xj_names.append(f"{xv}[{lev}]")
            cat_vars.append((xv, codes, cats))
        else:
            x_cols.append(s.to_numpy(dtype=float))
            x_names.append(str(xv))
            xj_cols.append(x_cols[-1])
            xj_names.append(str(xv))
    if len(cat_vars) == 1:
        level_var, level_codes, level_labels = (
            cat_vars[0][0],
            np.asarray(cat_vars[0][1], dtype=int),
            list(cat_vars[0][2]),
        )
    Xx = np.column_stack(x_cols) if x_cols else np.zeros((n, 0))

    x_tilde = Xx
    if Xx.shape[1]:
        pos = np.unique(np.concatenate([ft[np.isfinite(ft)], tt]))
        gap = float(np.min(np.diff(pos))) if pos.size > 1 else 1.0
        grp = _demean_groups(ft, tt, hettype, gap)
        G = int(grp.max()) + 1
        cnt = np.bincount(grp, minlength=G).astype(float)
        means = np.column_stack(
            [
                np.bincount(grp, weights=Xx[:, j], minlength=G) / cnt
                for j in range(Xx.shape[1])
            ]
        )
        x_tilde = Xx - means[grp]

    # ── assemble columns ───────────────────────────────────────────────
    cols: List[np.ndarray] = []
    names: List[str] = []
    if fe_mode == "cohort":
        cols.append(np.ones(n))
        names.append("const")
        for g_val in cohorts:
            cols.append((ft == g_val).astype(float))
            names.append(f"cohort[{int(g_val)}]")
    for t_val in periods[1:]:
        cols.append((tt == t_val).astype(float))
        names.append(f"period[{int(t_val)}]")

    treat_cols: List[int] = []
    param_col = np.empty(len(param_keys), dtype=int)
    for p, key in enumerate(param_keys):
        d = (row_param == p).astype(float)
        cols.append(d)
        names.append(_param_label(key))
        param_col[p] = len(names) - 1
        treat_cols.append(len(names) - 1)
        for j, xn in enumerate(x_names):
            cols.append(d * x_tilde[:, j])
            names.append(f"{_param_label(key)}:{xn}")
            treat_cols.append(len(names) - 1)

    if Xx.shape[1] and fe_mode == "cohort":
        # R etwfe: i(tvar, x_dm, ref = tref) -- the demeaned covariate by
        # period, and nothing else (no main effect, no cohort interaction).
        for j, xn in enumerate(x_names):
            for t_val in periods[1:]:
                cols.append((tt == t_val) * x_tilde[:, j])
                names.append(f"x[{xn}]:period[{int(t_val)}]")
    jwdid_block: Optional[np.ndarray] = None
    jwdid_alt: Optional[np.ndarray] = None
    if Xx.shape[1] and fe_mode != "cohort":
        # jwdid: i.tvar#c.(x) with the raw covariate, plus x and i.gvar#c.x
        # when x varies within units (a time-invariant x is absorbed).
        codes_u = pd.factorize(df[group])[0]
        U = int(codes_u.max()) + 1

        def _jwdid_terms(xcols, xnames, out_cols, out_names):
            for col, xn in zip(xcols, xnames):
                lo = np.full(U, np.inf)
                hi = np.full(U, -np.inf)
                np.minimum.at(lo, codes_u, col)
                np.maximum.at(hi, codes_u, col)
                span = 1e-12 * max(1.0, np.abs(col).max())
                if bool(np.any(hi - lo > span)):
                    out_cols.append(col)
                    out_names.append(f"x[{xn}]")
                    for g_val in cohorts:
                        out_cols.append((ft == g_val) * col)
                        out_names.append(f"x[{xn}]:cohort[{int(g_val)}]")
                for t_val in periods[1:]:
                    out_cols.append((tt == t_val) * col)
                    out_names.append(f"x[{xn}]:period[{int(t_val)}]")

        start = len(names)
        _jwdid_terms(list(Xx.T), x_names, cols, names)
        jwdid_block = np.arange(start, len(names))
        alt_cols: List[np.ndarray] = []
        _jwdid_terms(xj_cols, xj_names, alt_cols, [])
        jwdid_alt = np.column_stack(alt_cols) if alt_cols else None

    ctrl_names: List[str] = []
    for c in controls or []:
        cols.append(df[c].astype(float).to_numpy())
        names.append(f"control[{c}]")
        ctrl_names.append(c)

    return {
        "X": np.column_stack(cols),
        "names": names,
        "cells": cells,
        "cell_post": np.asarray(cell_post, dtype=bool),
        "cell_param": cell_param,
        "param_col": param_col,
        "param_labels": [_param_label(k) for k in param_keys],
        "row_cell": row_cell,
        "treat_cols": np.asarray(treat_cols, dtype=int),
        "x_names": x_names,
        "level_var": level_var,
        "level_codes": level_codes,
        "level_labels": level_labels,
        "ctrl_names": ctrl_names,
        "jwdid_block": jwdid_block,
        "jwdid_alt": jwdid_alt,
    }


# ── option normalisers shared by the fit and the aggregation step ──

#: Non-identity families supported by ``sp.etwfe(family=...)``, mapped to
#: their ``statsmodels`` family constructor.  ``None``/``'gaussian'`` keeps
#: the historical linear OLS path untouched.
_ETWFE_GLM_FAMILIES = {
    "poisson": "Poisson",
    "logit": "Binomial",
    "binomial": "Binomial",
}

_SCALES = ("response", "link")


def normalise_scale(scale: Optional[str]) -> str:
    """Validate ``scale=`` ('response' | 'link'; 'xb'/'eta' alias 'link')."""
    if scale is None:
        return "response"
    key = str(scale).strip().lower()
    if key in {"xb", "eta", "linear_predictor", "log"}:
        key = "link"
    if key in {"mu", "ame", "count", "probability"}:
        key = "response"
    if key not in _SCALES:
        raise MethodIncompatibility(
            f"scale={scale!r} is not recognised; use 'response' or 'link'.",
            recovery_hint="scale='link' reports the treated-observation-"
            "weighted average of the cohort x period coefficients (Stata "
            "estat simple, predict(xb)); 'response' the average marginal "
            "effect.",
            diagnostics={"scale": scale},
        )
    return key


def normalise_glm_fe(fe: Optional[str], fam_key: str) -> str:
    """Validate ``fe=`` for the nonlinear branch ('cohort' | 'unit')."""
    key = "cohort" if fe is None else str(fe).strip().lower()
    if key in {"ivar", "id", "individual"}:
        key = "unit"
    if key in {"gvar", "group", "mundlak"}:
        key = "cohort"
    if key not in {"cohort", "unit"}:
        raise MethodIncompatibility(
            f"fe={fe!r} is not recognised; use 'cohort' or 'unit'.",
            recovery_hint="fe='cohort' is R etwfe's design; fe='unit' "
            "absorbs unit fixed effects like Stata jwdid ..., "
            "method(ppmlhdfe).",
            diagnostics={"fe": fe},
        )
    if key == "unit" and fam_key not in {"poisson", "gaussian"}:
        raise MethodIncompatibility(
            f"fe='unit' is only available for family='poisson'; a {fam_key} "
            "model with unit effects suffers the incidental-parameters bias.",
            recovery_hint="Use fe='cohort' (Wooldridge's pooled Mundlak form) "
            "for binary outcomes.",
            diagnostics={"family": fam_key, "fe": fe},
        )
    return key


def _normalise_glm_cgroup(cgroup: str) -> str:
    key = str(cgroup).strip().lower()
    if key in {"notyet", "notyettreated", "not_yet"}:
        return "notyet"
    if key in {"never", "nevertreated", "never_treated"}:
        return "never"
    raise MethodIncompatibility(
        f"cgroup={cgroup!r} is not recognised; use 'notyet' or 'nevertreated'.",
        recovery_hint="cgroup='notyet' (default) or cgroup='nevertreated'.",
        diagnostics={"cgroup": cgroup},
    )
