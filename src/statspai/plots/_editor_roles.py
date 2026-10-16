"""Artist classification for the figure editor.

Every artist in a figure is given a role. Roles in ``LOCKED_ROLES`` carry
statistical content (estimates, fitted values, intervals, bars, colour-mapped
values); the editor may restyle them but never moves, hides or recolours them
in a way that changes what they say.

Two rules keep the classification on the safe side:

* A label can only move an artist to a *more* protected role. "Treatment
  group" is a data series, not a reference line; only geometry (a constant
  horizontal or vertical line) makes something a reference line.
* Whatever is not recognised is locked.
"""

from __future__ import annotations

import re
from enum import Enum, auto
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


class ArtistRole(Enum):
    """Classification of matplotlib artists by editability."""

    DATA = auto()  # Statistical data — LOCKED
    FIT = auto()  # Regression/fit lines — LOCKED
    CI = auto()  # Confidence intervals — LOCKED
    REFERENCE = auto()  # Reference lines (y=0, cutoff) — style only
    LABEL = auto()  # Titles, axis labels — fully editable
    ANNOTATION = auto()  # User annotations — fully editable
    LEGEND = auto()  # Legend — position/style editable
    SPINE = auto()  # Axis spines — visibility editable
    COSMETIC = auto()  # Purely cosmetic elements — fully editable


LOCKED_ROLES = (ArtistRole.DATA, ArtistRole.FIT, ArtistRole.CI)

_REF_LABEL = re.compile(
    r"\b(reference|zero|cutoff|onset|threshold|null|treatment|baseline)\b"
)
_FIT_LABEL = re.compile(
    r"\b(fit|trend|regression|polynomial|predicted|fitted|smoothed)\b"
)
_CI_LABEL = re.compile(r"\b(ci|confidence|bound|interval|upper|lower|band)\b")
_HAS_DIGIT = re.compile(r"\d")


def classify_figure(fig: Any) -> Dict[int, ArtistRole]:
    """Return ``{id(artist): role}`` for every artist the editor can reach."""
    roles: Dict[int, ArtistRole] = {}
    for ax in fig.get_axes():
        roles[id(ax.title)] = ArtistRole.LABEL
        roles[id(ax.xaxis.label)] = ArtistRole.LABEL
        roles[id(ax.yaxis.label)] = ArtistRole.LABEL
        for spine in ax.spines.values():
            roles[id(spine)] = ArtistRole.SPINE
        legend = ax.get_legend()
        if legend is not None:
            roles[id(legend)] = ArtistRole.LEGEND

        _classify_containers(ax, roles)
        for line in ax.get_lines():
            if id(line) not in roles:
                roles[id(line)] = _classify_line(line, ax)
        for coll in ax.collections:
            if id(coll) not in roles:
                roles[id(coll)] = _classify_collection(coll, ax)
        for patch in ax.patches:
            if id(patch) not in roles:
                roles[id(patch)] = _classify_patch(patch, ax)
        _lock_alpha_coded_spans(ax, roles)
        for image in ax.images:
            roles[id(image)] = ArtistRole.DATA
        for text in ax.texts:
            # Pre-existing text that shows a number ("0.12***", "N = 500")
            # reports a result: its wording is locked, its style is not.
            locked = bool(_HAS_DIGIT.search(text.get_text() or ""))
            roles[id(text)] = ArtistRole.DATA if locked else ArtistRole.ANNOTATION

    suptitle = getattr(fig, "_suptitle", None)
    if suptitle is not None:
        roles[id(suptitle)] = ArtistRole.LABEL
    return roles


def _classify_containers(ax: Any, roles: Dict[int, ArtistRole]) -> None:
    """Containers say exactly which artists belong to an errorbar/bar/stem."""
    from matplotlib.container import BarContainer, ErrorbarContainer, StemContainer

    for container in getattr(ax, "containers", []):
        if isinstance(container, ErrorbarContainer):
            data_line, caplines, barlinecols = container.lines
            if data_line is not None:
                roles[id(data_line)] = ArtistRole.DATA
            for cap in caplines or ():
                roles[id(cap)] = ArtistRole.CI
            for coll in barlinecols or ():
                roles[id(coll)] = ArtistRole.CI
        elif isinstance(container, BarContainer):
            for patch in container.patches:
                roles[id(patch)] = ArtistRole.DATA
            if container.errorbar is not None:
                data_line, caplines, barlinecols = container.errorbar.lines
                for cap in caplines or ():
                    roles[id(cap)] = ArtistRole.CI
                for coll in barlinecols or ():
                    roles[id(coll)] = ArtistRole.CI
        elif isinstance(container, StemContainer):
            roles[id(container.markerline)] = ArtistRole.DATA
            stems = container.stemlines
            for stem in stems if isinstance(stems, (list, tuple)) else [stems]:
                roles[id(stem)] = ArtistRole.DATA
            roles[id(container.baseline)] = ArtistRole.REFERENCE


def _is_flat(line: Any) -> bool:
    """A horizontal or vertical straight line."""
    # get_xydata() holds the unit-converted coordinates, so categorical and
    # date axes work too.
    xy = np.asarray(line.get_xydata(), dtype=float)
    if xy.ndim != 2 or xy.shape[0] < 2:
        return False
    x, y = xy[:, 0], xy[:, 1]
    fx, fy = x[np.isfinite(x)], y[np.isfinite(y)]
    if fx.size < 2 or fy.size < 2:
        return False
    return bool(np.ptp(fy) == 0 or np.ptp(fx) == 0)


def _classify_line(line: Any, ax: Any) -> ArtistRole:
    label = (line.get_label() or "").lower()
    if _is_flat(line):
        # axhline / axvline: one coordinate lives in axes space.
        if not line.get_transform().contains_branch(ax.transData):
            return ArtistRole.REFERENCE
        # In data space a two-point flat segment is a drawn reference line;
        # a longer flat series is data that happens to be constant unless
        # its label says otherwise.
        if np.asarray(line.get_xdata()).size == 2 or _REF_LABEL.search(label):
            return ArtistRole.REFERENCE
    if _FIT_LABEL.search(label):
        return ArtistRole.FIT
    if _CI_LABEL.search(label):
        return ArtistRole.CI
    return ArtistRole.DATA


def _classify_collection(coll: Any, ax: Any) -> ArtistRole:
    import matplotlib.collections as mcoll

    if isinstance(coll, mcoll.PathCollection):
        return ArtistRole.DATA
    if isinstance(coll, mcoll.LineCollection):
        return ArtistRole.CI
    if isinstance(coll, mcoll.PolyCollection):
        if not coll.get_transform().contains_branch(ax.transData):
            return ArtistRole.REFERENCE  # axvspan-style shading
        if coll.get_array() is not None:
            return ArtistRole.DATA  # hexbin and friends
        return ArtistRole.CI
    return ArtistRole.DATA


def _classify_patch(patch: Any, ax: Any) -> ArtistRole:
    # axvspan / axhspan shade a period or a region; they span the axes in one
    # direction and so are drawn in a blended transform.
    if not patch.get_transform().contains_branch(ax.transData):
        return ArtistRole.REFERENCE
    return ArtistRole.DATA


def _lock_alpha_coded_spans(ax: Any, roles: Dict[int, ArtistRole]) -> None:
    """Shaded spans of one colour whose transparency varies encode values
    (``sp.synthdid_plot`` shades each period by its time weight); lock them."""
    spans: Dict[str, List[Any]] = {}
    for patch in ax.patches:
        if roles.get(id(patch)) is ArtistRole.REFERENCE:
            spans.setdefault(_hex(patch.get_facecolor()), []).append(patch)
    for group in spans.values():
        alphas = {round(float(p.get_facecolor()[3]), 12) for p in group}
        if len(group) > 1 and len(alphas) > 1:
            for p in group:
                roles[id(p)] = ArtistRole.DATA


# ----------------------------------------------------------------------
# Extent of the locked content, used to keep axis limits honest
# ----------------------------------------------------------------------


def locked_extent(
    ax: Any, axis: str, roles: Dict[int, ArtistRole]
) -> Tuple[Optional[float], Optional[float]]:
    """Min and max, in data units, of everything locked on one axis."""
    col = 0 if axis == "x" else 1
    lo, hi = np.inf, -np.inf

    def take(values: Any) -> None:
        nonlocal lo, hi
        arr = np.asarray(values, dtype=float).ravel()
        arr = arr[np.isfinite(arr)]
        if arr.size:
            lo, hi = min(lo, float(arr.min())), max(hi, float(arr.max()))

    for line in ax.get_lines():
        if roles.get(id(line)) not in LOCKED_ROLES or not line.get_visible():
            continue
        if line.get_transform().contains_branch(ax.transData):
            xy = np.asarray(line.get_xydata(), dtype=float)
            if xy.ndim == 2 and xy.shape[0]:
                take(xy[:, col])
    for coll in ax.collections:
        if roles.get(id(coll)) not in LOCKED_ROLES or not coll.get_visible():
            continue
        box = coll.get_datalim(ax.transData)
        pts = np.asarray(box.get_points(), dtype=float)
        if np.all(np.isfinite(pts)):
            take(pts[:, col])
    for patch in ax.patches:
        if roles.get(id(patch)) not in LOCKED_ROLES or not patch.get_visible():
            continue
        if patch.get_transform().contains_branch(ax.transData):
            box = patch.get_path().get_extents(patch.get_patch_transform())
            take(np.asarray(box.get_points())[:, col])
    for image in ax.images:
        if roles.get(id(image)) in LOCKED_ROLES and image.get_visible():
            left, right, bottom, top = image.get_extent()
            take([left, right] if axis == "x" else [bottom, top])

    if not np.isfinite(lo):
        return None, None
    return lo, hi


def expand_to_cover(
    ax: Any, axis: str, low: float, high: float, data_lo: float, data_hi: float
) -> Tuple[float, float, bool]:
    """Widen ``[low, high]`` to cover the data plus a 2% margin.

    The margin is taken in the axis' scale space, so log axes get a
    multiplicative margin. Returns ``(low, high, changed)`` in the caller's
    orientation (an inverted axis stays inverted).
    """
    inverted = low > high
    lo, hi = (high, low) if inverted else (low, high)
    scale = (ax.xaxis if axis == "x" else ax.yaxis).get_transform()
    d0, d1 = scale.transform([data_lo, data_hi])
    if not (np.isfinite(d0) and np.isfinite(d1)):
        return low, high, False
    span = d1 - d0
    pad = 0.02 * span if span > 0 else 0.02 * max(abs(d0), 1.0)
    safe_lo, safe_hi = scale.inverted().transform([d0 - pad, d1 + pad])
    changed = lo > safe_lo or hi < safe_hi
    lo, hi = min(lo, float(safe_lo)), max(hi, float(safe_hi))
    return ((hi, lo) if inverted else (lo, hi)) + (changed,)


# ----------------------------------------------------------------------
# Colour groups: artists that share a colour describe one series
# ----------------------------------------------------------------------


def _hex(rgba: Any) -> str:
    import matplotlib.colors as mcolors

    return str(mcolors.to_hex(rgba, keep_alpha=False))


def single_color(artist: Any) -> Optional[str]:
    """The one colour an artist is drawn in, or ``None`` if it has several
    or its colours encode values (a colour map)."""
    import matplotlib.collections as mcoll
    import matplotlib.lines as mlines
    import matplotlib.patches as mpatches

    if isinstance(artist, mlines.Line2D):
        return _hex(artist.get_color())
    if isinstance(artist, mcoll.Collection):
        if artist.get_array() is not None:
            return None
        if isinstance(artist, mcoll.LineCollection):
            raw = artist.get_edgecolor()
        else:
            raw = artist.get_facecolor()
            if len(raw) == 0:
                raw = artist.get_edgecolor()
        cols = np.asarray(raw)
        if cols.ndim != 2 or cols.shape[0] == 0:
            return None
        hexes = {_hex(c) for c in cols}
        return hexes.pop() if len(hexes) == 1 else None
    if isinstance(artist, mpatches.Patch):
        if not artist.get_fill():
            return _hex(artist.get_edgecolor())
        return _hex(artist.get_facecolor())
    return None


def color_groups(ax: Any, roles: Dict[int, ArtistRole]) -> List[Dict[str, Any]]:
    """Group the locked artists of one axes by the colour they share.

    Returns ``[{"color": "#1f77b4", "members": [("line", 0), ("coll", 2)]}]``
    in order of first appearance. A point estimate, its error bars and its
    confidence band are normally drawn in one colour, so they land in one
    group and are recoloured together.
    """
    groups: Dict[str, List[Tuple[str, int]]] = {}
    for kind, artists in (
        ("line", ax.get_lines()),
        ("coll", ax.collections),
        ("patch", ax.patches),
    ):
        for j, artist in enumerate(artists):
            if roles.get(id(artist)) not in LOCKED_ROLES:
                continue
            color = single_color(artist)
            if color is None:
                continue
            groups.setdefault(color, []).append((kind, j))
    return [{"color": c, "members": m} for c, m in groups.items()]
