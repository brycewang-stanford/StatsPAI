"""Array engine behind ``sp.did_multiplegt_dyn``.

``did_multiplegt_dyn.py`` defines every event of the estimator on the long
DataFrame (``_one_event`` and the helpers under it). Those definitions stay
there and remain the fallback; this module evaluates the same arithmetic on
unit x period matrices built once, because the estimator visits a few dozen
events per horizon and, under the bootstrap, does so again for every
replicate.

The contract is that nothing reported changes, to the last bit where numpy
allows it. Three things make that hold and should survive any edit here:

* every sample lists its units in the order their rows appear in the frame,
  since ``np.sum`` is pairwise and its result depends on the order;
* the cumulated treatment change is added up with the compensated sum that
  ``groupby(...).sum()`` uses;
* a bootstrap replicate is the frame ``cluster_bootstrap_draw`` would have
  built -- same call on the generator, same row order -- without building it.

``build_panel`` returns ``None`` for the layouts whose frame semantics are not
unit-level (duplicated ``(group, period)`` rows, a unit whose rows are out of
time order, missing group labels); the caller then uses the frame code.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class EventPanel:
    """A long panel as unit x period matrices (one all-missing pad column)."""

    n: int
    n_rows: int
    t_lo: int
    width: int
    t_min: float
    unit_t_min: np.ndarray
    Y: np.ndarray
    y_ok: np.ndarray
    D: np.ndarray
    obs: np.ndarray
    W: Optional[np.ndarray]
    X: Tuple[np.ndarray, ...]
    rowpos: np.ndarray
    first_row: np.ndarray
    ordered: bool
    F: np.ndarray
    never: np.ndarray
    direction: np.ndarray
    base: np.ndarray
    tcode: Optional[np.ndarray]
    elig: Optional[np.ndarray] = None
    units: Optional[pd.Index] = None
    labels: Optional[pd.Index] = None
    cluster_codes: Optional[np.ndarray] = None
    one_per_cluster: bool = True

    def col(self, t: Any) -> int:
        """Column of period ``t``; the pad column when it is out of range."""
        j = int(t) - self.t_lo
        return j if 0 <= j < self.width else -1


def _padded(n: int, width: int, fill: Any, dtype: Any) -> np.ndarray:
    return np.full((n, width + 1), fill, dtype=dtype)


def _unit_level(values: np.ndarray, codes: np.ndarray, first_row: np.ndarray) -> Any:
    """Per-unit value of a row-level column, or ``None`` if it varies in a unit."""
    per_unit = values[first_row]
    back = per_unit[codes]
    same = (back == values) | (pd.isna(back) & pd.isna(values))
    return per_unit if bool(np.all(same)) else None


def build_panel(
    df: pd.DataFrame,
    *,
    y: str,
    group: str,
    time: str,
    treatment: str,
    weights: Optional[str],
    cluster: Optional[str],
    controls: Optional[List[str]] = None,
) -> Optional[EventPanel]:
    """The matrices for ``df``, or ``None`` when the frame code must be used."""
    n_rows = len(df)
    if n_rows == 0 or "_elig" in df.columns or df[group].isna().any():
        return None
    units = pd.Index(sorted(df[group].unique()))
    codes = units.get_indexer(df[group])
    t_raw = df[time].to_numpy()
    if (codes < 0).any() or pd.isna(t_raw).any():
        return None
    t_int = t_raw.astype(np.int64)
    if not np.array_equal(t_int, t_raw):
        return None
    n = len(units)
    t_lo = int(t_int.min())
    width = int(t_int.max()) - t_lo + 1
    cols = t_int - t_lo
    key = codes.astype(np.int64) * width + cols
    order = np.argsort(key, kind="stable")
    sorted_key = key[order]
    if (sorted_key[1:] == sorted_key[:-1]).any():
        return None
    # rows of a unit in time order: what the first switch, the compensated
    # treatment sums and the bootstrap copies all read
    same_unit = codes[order][1:] == codes[order][:-1]
    if not np.all(order[1:][same_unit] > order[:-1][same_unit]):
        return None
    by_period = np.argsort(cols.astype(np.int64) * n + codes, kind="stable")
    same_period = cols[by_period][1:] == cols[by_period][:-1]
    ordered = bool(np.all(by_period[1:][same_period] > by_period[:-1][same_period]))

    rows = np.arange(n_rows)
    first_row = np.full(n, n_rows, dtype=np.int64)
    np.minimum.at(first_row, codes, rows)

    def matrix(column: str) -> np.ndarray:
        out = _padded(n, width, np.nan, float)
        out[codes, cols] = df[column].astype(float).to_numpy()
        return out

    Y = matrix(y)
    obs = _padded(n, width, False, bool)
    obs[codes, cols] = True
    rowpos = _padded(n, width, -1, np.int64)
    rowpos[codes, cols] = rows

    unit_cols = []
    for name in ("_F", "_dir", "_base"):
        per_unit = _unit_level(df[name].to_numpy(dtype=float), codes, first_row)
        if per_unit is None:
            return None
        unit_cols.append(per_unit)
    F, direction, base = unit_cols

    tcode = None
    if "_tcell" in df.columns:
        if df["_tcell"].isna().any():
            return None
        cells = sorted(df["_tcell"].unique())
        row_code = pd.Index(cells).get_indexer(df["_tcell"])
        tcode = _unit_level(row_code, codes, first_row)
        if tcode is None:
            return None

    cl_col = cluster if cluster is not None else group
    cluster_of = df.groupby(group)[cl_col].first().reindex(units)
    cluster_codes = pd.factorize(cluster_of.to_numpy())[0]
    if (cluster_codes < 0).any():
        return None

    return EventPanel(
        n=n,
        n_rows=n_rows,
        t_lo=t_lo,
        width=width,
        t_min=float(df[time].min()),
        unit_t_min=t_int[first_row],
        Y=Y,
        y_ok=~np.isnan(Y),
        D=matrix(treatment),
        obs=obs,
        W=matrix(weights) if weights is not None else None,
        X=tuple(matrix(c) for c in (controls or [])),
        rowpos=rowpos,
        first_row=first_row,
        ordered=ordered,
        F=F,
        never=np.isnan(F),
        direction=direction,
        base=base,
        tcode=tcode,
        units=units,
        labels=pd.Index(df[group]).take(first_row).rename(group),
        cluster_codes=cluster_codes,
        one_per_cluster=int(cluster_codes.max()) + 1 == n,
    )


class Resampler:
    """Cluster bootstrap replicates of a panel, as ``cluster_bootstrap_draw``.

    A replicate stacks the drawn clusters in draw order, each with its rows in
    their original order and its groups relabelled per draw, so a group drawn
    twice is two groups.
    """

    def __init__(self, panel: EventPanel, df: pd.DataFrame, cluster_col: str):
        self.panel = panel
        self.clusters = df[cluster_col].unique()
        row_cluster = pd.Index(self.clusters).get_indexer(df[cluster_col])
        unit_cluster = row_cluster[panel.first_row]
        self.members = np.argsort(unit_cluster, kind="stable")
        sizes = np.bincount(unit_cluster, minlength=len(self.clusters))
        self.sizes = sizes
        self.starts = np.cumsum(sizes) - sizes
        self._lookup = pd.Index(self.clusters)

    def draw(self, rng: np.random.Generator) -> EventPanel:
        sampled = rng.choice(self.clusters, size=len(self.clusters), replace=True)
        drawn = self._lookup.get_indexer(sampled)
        if (drawn < 0).any():
            raise KeyError("sampled cluster not in the panel")
        counts = self.sizes[drawn]
        total = int(counts.sum())
        offset = np.arange(total) - np.repeat(np.cumsum(counts) - counts, counts)
        take = self.members[np.repeat(self.starts[drawn], counts) + offset]
        shift = np.repeat(np.arange(len(drawn), dtype=np.int64), counts)
        p = self.panel
        shift_rows = shift * p.n_rows
        rowpos = p.rowpos
        if not p.ordered:
            rowpos = p.rowpos[take]
            rowpos = np.where(rowpos >= 0, rowpos + shift_rows[:, None], -1)
        return replace(
            p,
            n=total,
            t_min=float(p.unit_t_min[take].min()),
            unit_t_min=p.unit_t_min[take],
            Y=p.Y[take],
            y_ok=p.y_ok[take],
            D=p.D[take],
            obs=p.obs[take],
            W=p.W[take] if p.W is not None else None,
            X=(),
            rowpos=rowpos,
            first_row=p.first_row[take] + shift_rows,
            F=p.F[take],
            never=p.never[take],
            direction=p.direction[take],
            base=p.base[take],
            tcode=p.tcode[take] if p.tcode is not None else None,
            elig=None,
            units=None,
            labels=None,
            cluster_codes=None,
        )


def _sample(
    p: EventPanel, mask: np.ndarray, j_pre: int, j_post: int, j_anchor: int
) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """``_event_sample`` on the matrices: unit positions, change, weights."""
    ok = mask & p.y_ok[:, j_pre] & p.y_ok[:, j_post]
    if j_anchor != j_post:
        ok &= p.y_ok[:, j_anchor]
    idx: np.ndarray = np.flatnonzero(ok)
    if idx.size == 0:
        return None
    if not p.ordered:
        idx = idx[np.argsort(p.rowpos[idx, j_pre], kind="stable")]
    w: np.ndarray
    if p.W is None:
        w = np.ones(idx.size, dtype=float)
    else:
        w_raw = p.W[idx, j_anchor]
        w = np.where(np.isnan(w_raw), 0.0, w_raw)
        keep = w != 0
        if not keep.any():
            return None
        idx = idx[keep]
        w = w[keep]
    dy = p.Y[idx, j_post] - p.Y[idx, j_pre]
    return idx, dy, w


def _dose(
    p: EventPanel, idx: np.ndarray, w: np.ndarray, F: int, h: int, base_level: float
) -> float:
    """``_event_dose`` on the matrices."""
    span = range(0, h + 1) if h >= 0 else range(0, -h)
    js = np.array([j for j in (p.col(F + s) for s in span) if j >= 0], dtype=np.intp)
    if js.size == 0:
        return float("nan")
    seen = p.obs[np.ix_(idx, js)]
    if not seen.any():
        return float("nan")
    inc = np.abs(p.D[np.ix_(idx, js)] - base_level)
    # the compensated (Kahan) sum of pandas' group_sum, period by period
    total: np.ndarray = np.zeros(idx.size)
    comp: np.ndarray = np.zeros(idx.size)
    with np.errstate(invalid="ignore"):
        for k in range(len(js)):
            v = inc[:, k]
            use = ~np.isnan(v)
            term = v - comp
            new = total + term
            c_new = (new - total) - term
            c_new = np.where(np.isnan(c_new), 0.0, c_new)
            total = np.where(use, new, total)
            comp = np.where(use, c_new, comp)
    per_unit = np.where(seen.any(axis=1), total, np.nan)
    w_total = float(w.sum())
    if w_total <= 0 or not np.all(np.isfinite(per_unit)):
        return float("nan")
    return float(np.sum(w * per_unit) / w_total)


def _n_clusters(p: EventPanel, idx: np.ndarray) -> int:
    if p.one_per_cluster:
        return int(idx.size)
    assert p.cluster_codes is not None
    return int(np.unique(p.cluster_codes[idx]).size)


def _cell_centre(
    n_own: int, mean_own: float, n_pool: int, mean_pool: float
) -> Tuple[float, float]:
    if n_own >= 2:
        return mean_own, float(np.sqrt(n_own / (n_own - 1.0)))
    if n_pool >= 2:
        return mean_pool, float(np.sqrt(n_pool / (n_pool - 1.0)))
    return 0.0, 1.0


def one_event(
    p: EventPanel,
    *,
    F: Any,
    h: int,
    direction: int,
    base: Optional[float],
    tcode: Optional[int],
    control: str,
    match_baseline: bool,
    full: bool,
    need_dose: bool,
) -> Optional[Dict[str, Any]]:
    """``_one_event`` on the matrices.

    ``full=False`` returns what a point estimate needs (``delta``, ``w_sw``,
    ``n_sw``, ``sw`` and, on request, ``dose``); ``full=True`` adds the
    influence terms as ``(positions, values)`` pairs and the dose pieces.
    """
    sw = (p.F == F) & (p.direction == direction)
    if base is not None:
        sw &= p.base == base
    if tcode is not None:
        sw &= p.tcode == tcode
    if p.elig is not None:
        sw &= p.elig
    sw_all = np.flatnonzero(sw)
    if sw_all.size == 0:
        return None
    # the frame code reads the baseline off the first switcher row
    base_level = float(p.base[sw_all[np.argmin(p.first_row[sw_all])]])

    f_int = int(F)
    if h >= 0:
        t_pre, t_post = f_int - 1, f_int + h
        t_anchor = t_post
    else:
        i = -h
        t_pre, t_post = f_int - 1 - i, f_int - 1
        t_anchor = f_int - 1 + i
    if t_pre < p.t_min:
        return None
    j_pre, j_post, j_anchor = p.col(t_pre), p.col(t_post), p.col(t_anchor)

    if control == "never_treated":
        ctrl = p.never & ~sw
    else:
        ctrl = ((p.F > t_anchor) | p.never) & ~sw
    if match_baseline:
        ctrl &= p.D[:, j_pre] == base_level
    if tcode is not None:
        ctrl &= p.tcode == tcode
    if not ctrl.any():
        return None

    s_sample = _sample(p, sw, j_pre, j_post, j_anchor)
    c_sample = _sample(p, ctrl, j_pre, j_post, j_anchor)
    if s_sample is None or c_sample is None:
        return None
    sw_idx, sw_dy, sw_w = s_sample
    c_idx, c_dy, c_w = c_sample

    w_s = float(sw_w.sum())
    w_c = float(c_w.sum())
    sum_s = np.sum(sw_w * sw_dy)
    sum_c = np.sum(c_w * c_dy)
    mean_s = float(sum_s / w_s)
    mean_c = float(sum_c / w_c)
    scale = (-1.0 if h < 0 else 1.0) / direction
    out: Dict[str, Any] = {
        "delta": float(scale * (mean_s - mean_c)),
        "n_sw": int(sw_idx.size),
        "w_sw": w_s,
        "sw": sw_idx,
    }
    if need_dose or full:
        out["dose"] = _dose(p, sw_idx, sw_w, f_int, h, base_level)
    if not full:
        return out

    n_c = _n_clusters(p, c_idx)
    n_pool = _n_clusters(p, np.concatenate([sw_idx, c_idx]))
    mean_pool = float((sum_s + sum_c) / (w_s + w_c))
    e_c, dof_c = _cell_centre(n_c, mean_c, n_pool, mean_pool)

    d_at_f = p.D[sw_idx, p.col(f_int)]
    e_s = np.empty(sw_idx.size, dtype=float)
    dof_s = np.empty(sw_idx.size, dtype=float)
    for level in pd.unique(d_at_f):
        m = d_at_f == level if level == level else np.isnan(d_at_f)
        n_s = _n_clusters(p, sw_idx[m])
        mean_cell = float(np.sum(sw_w[m] * sw_dy[m]) / np.sum(sw_w[m]))
        e_s[m], dof_s[m] = _cell_centre(n_s, mean_cell, n_pool, mean_pool)

    out["psi_sw"] = (sw_w * dof_s * (sw_dy - e_s)) * scale
    out["psi_c"] = (0.0 - (w_s / w_c) * c_w * dof_c * (c_dy - e_c)) * scale
    out["c"] = c_idx

    m_x = np.zeros(len(p.X))
    for k, x in enumerate(p.X):
        xs = x[sw_idx, j_post] - x[sw_idx, j_pre]
        xc = x[c_idx, j_post] - x[c_idx, j_pre]
        m_x[k] = scale * (
            float(np.sum(sw_w * xs)) - (w_s / w_c) * float(np.sum(c_w * xc))
        )
    out["m_x"] = m_x

    lag_dose = np.zeros(max(h, 0) + 1)
    dose_now = float("nan")
    if h >= 0:
        periods = [t for t in range(f_int, t_post + 1) if p.col(t) >= 0]
        js = np.array([p.col(t) for t in periods], dtype=np.intp)
        rp = p.rowpos[np.ix_(sw_idx, js)]
        gap_all = p.D[np.ix_(sw_idx, js)] - base_level
        value = sw_w[:, None] * np.abs(gap_all)
        lags = np.broadcast_to(t_post - np.asarray(periods, dtype=int), rp.shape)
        ok_gap = (rp >= 0) & np.isfinite(gap_all)
        in_rows = np.argsort(rp[ok_gap], kind="stable")
        np.add.at(lag_dose, lags[ok_gap][in_rows], value[ok_gap][in_rows])

        gap = np.abs(p.D[sw_idx, j_post] - base_level)
        ok = np.isfinite(gap)
        if ok.any():
            dose_now = float(np.sum(sw_w[ok] * gap[ok]) / np.sum(sw_w[ok]))
    out["lag_dose"] = lag_dose
    out["dose_now"] = dose_now
    out["effects"] = scale * (sw_dy - mean_c)
    return out


def _events(p: EventPanel) -> Tuple[List[Any], Dict[Tuple[Any, int], List[float]]]:
    """Switch periods, and the baselines seen for each (period, direction)."""
    switch = np.flatnonzero(~p.never)
    switch = switch[np.argsort(p.first_row[switch], kind="stable")]
    f_values = sorted(pd.unique(p.F[switch]))
    seen = pd.DataFrame(
        {"F": p.F[switch], "dir": p.direction[switch], "base": p.base[switch]}
    ).drop_duplicates()
    bases_of: Dict[Tuple[Any, int], List[float]] = {}
    for f, d, b in seen.itertuples(index=False):
        bases_of.setdefault((f, int(d)), []).append(float(b))
    return f_values, bases_of


def common_switchers(
    p: EventPanel,
    *,
    horizons: List[int],
    control: str,
    directions: Tuple[int, ...],
    match_baseline: bool,
) -> EventPanel:
    """``same_switchers``: flag the switchers that support every effect."""
    need = sorted({-1, *(h for h in horizons if h >= 0)})
    f_int = np.where(p.never, 0, p.F).astype(np.int64)
    rows = np.arange(p.n)
    has_all = np.ones(p.n, dtype=bool)
    for h in need:
        j = f_int + h - p.t_lo
        inside = (j >= 0) & (j < p.width)
        has_all &= inside & p.obs[rows, np.where(inside, j, 0)]
    p = replace(p, elig=p.never | has_all)
    effect_h = [h for h in horizons if h >= 0]
    if len(effect_h) > 1:
        first = point_estimates(
            p,
            horizons=effect_h,
            control=control,
            directions=directions,
            normalized=False,
            match_baseline=match_baseline,
            want_switchers=True,
        )
        if "switchers" not in first:
            # what the frame code raises when there is no switcher at all
            raise KeyError("group_effects")
        contributing = first["switchers"]
        common = np.ones(p.n, dtype=bool)
        for mask in contributing:
            common &= mask
        assert p.elig is not None
        p = replace(p, elig=p.elig & (p.never | common))
    return p


def _event_grid(
    p: EventPanel, directions: Tuple[int, ...], match_baseline: bool
) -> Optional[List[Tuple[Any, int, Optional[float], Optional[int]]]]:
    f_values, bases_of = _events(p)
    if not f_values:
        return None
    cells: Tuple[Optional[int], ...] = (None,)
    if p.tcode is not None:
        cells = tuple(int(c) for c in np.unique(p.tcode))
    grid: List[Tuple[Any, int, Optional[float], Optional[int]]] = []
    for F in f_values:
        for direction in directions:
            bases: List[Optional[float]] = [None]
            if match_baseline:
                bases = list(sorted(bases_of.get((F, direction), [])))
            grid.extend((F, direction, b, c) for b in bases for c in cells)
    return grid


def point_estimates(
    p: EventPanel,
    *,
    horizons: List[int],
    control: str,
    directions: Tuple[int, ...],
    normalized: bool,
    match_baseline: bool,
    want_switchers: bool = False,
) -> Dict[str, Any]:
    """The effect at each horizon and nothing else (a bootstrap replicate)."""
    grid = _event_grid(p, directions, match_baseline)
    if grid is None:
        return {"delta": None}
    deltas: List[float] = []
    contributing: List[np.ndarray] = []
    for h in horizons:
        sum_wdelta = 0.0
        sum_wdose = 0.0
        w_total = 0.0
        mask = np.zeros(p.n, dtype=bool)
        for F, direction, base, tcode in grid:
            cell = one_event(
                p,
                F=F,
                h=h,
                direction=direction,
                base=base,
                tcode=tcode,
                control=control,
                match_baseline=match_baseline,
                full=False,
                need_dose=normalized,
            )
            if cell is None:
                continue
            sum_wdelta += cell["delta"] * cell["w_sw"]
            if normalized:
                sum_wdose += cell["dose"] * cell["w_sw"]
            w_total += cell["w_sw"]
            mask[cell["sw"]] = True
        delta_l = np.nan
        if w_total > 0:
            delta_l = sum_wdelta / w_total
            if normalized:
                dose = sum_wdose / w_total
                delta_l = delta_l / dose if np.isfinite(dose) and dose != 0 else np.nan
        deltas.append(float(delta_l) if np.isfinite(delta_l) else np.nan)
        contributing.append(mask)
    out: Dict[str, Any] = {"delta": deltas}
    if want_switchers:
        out["switchers"] = contributing
    return out
