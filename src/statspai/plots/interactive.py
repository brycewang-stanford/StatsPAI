"""
Interactive figure editor for StatsPAI.

Cosmetic editing of matplotlib figures with the statistical content locked,
in the spirit of Stata's Graph Editor.

Design Principle
----------------
**Cosmetic editable, data locked.** Titles, labels, fonts, colours, layout,
legends and annotations can change. Data points, fitted lines, error bars,
confidence bands and bars cannot be moved, hidden, clipped by axis limits or
made invisible, and colour-mapped colours (which encode values) cannot be
changed.

Three guarantees, each checked by the test suite on every operation:

1. ``generate_code()`` reproduces the edited figure exactly when run after
   the code that drew it.
2. ``undo()`` and ``reset()`` return the figure to exactly what it was.
3. A saved edit list (``to_edits_json()``) replayed on a freshly drawn copy
   of the figure gives the same result.

The editor never changes global matplotlib settings.

Usage
-----
>>> import matplotlib
>>> matplotlib.use("Agg")
>>> import matplotlib.pyplot as plt
>>> import statspai as sp
>>> fig, ax = plt.subplots()        # any statspai figure works here
>>> _ = ax.plot([0, 1, 2], [0, 1, 4])
>>> editor = sp.interactive(fig)    # opens the editor where one can be shown
>>> isinstance(sp.get_code(fig), str)  # reproducible code string
True

In Jupyter notebooks ``sp.interactive(fig)`` shows an ipywidgets control
panel beside a live preview; with a GUI matplotlib backend it opens a small
control window next to the figure; elsewhere (scripts on Agg, servers,
agents) it returns the editor without touching the figure.
"""

from __future__ import annotations

import contextlib
import logging
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, Iterator, List, Literal, Optional, Sequence, Tuple

from ..exceptions import MethodIncompatibility
from . import _editor_theme  # noqa: F401  (registers apply_theme)
from ._editor_io import _action_to_op, _legacy_to_op
from ._editor_methods import EditMethods
from ._editor_ops import OPS, lit
from ._editor_presets import (  # noqa: F401  (re-exported)
    _PRESET_ALIASES,
    FONT_CHOICES,
    FONT_PRESETS,
    SIZE_PRESETS,
    _build_font_choices,
    _detect_chinese_fonts,
    _get_chinese_sans,
    _get_chinese_serif,
    _resolve_preset_fonts,
    get_font_choices,
)
from ._editor_roles import (  # noqa: F401  (ArtistRole is re-exported)
    LOCKED_ROLES,
    ArtistRole,
    classify_figure,
    color_groups,
    locked_extent,
)
from ._editor_state import Snapshot, capture, restore
from ._legend_sync import legend_sources, refresh_legend

logger = logging.getLogger(__name__)

STATE_VERSION = 2


@dataclass
class EditRecord:
    """One edit: the operation, its parameters and how to undo it.

    ``op`` and ``params`` describe the edit and are what gets saved and
    replayed. ``target_desc`` / ``property_name`` / ``old_value`` /
    ``new_value`` / ``code_line`` are kept for readers of the version-1
    format; ``code_line`` is always regenerated from ``op`` and ``params``.
    """

    target_desc: str
    property_name: str
    old_value: Any
    new_value: Any
    code_line: str
    op: Optional[str] = None
    params: Optional[Dict[str, Any]] = None
    resolved: Optional[Dict[str, Any]] = field(default=None, repr=False)
    snapshot: Optional[Snapshot] = field(default=None, repr=False, compare=False)
    slot: Optional[tuple] = field(default=None, repr=False)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to a JSON-compatible dictionary."""
        return {
            "op": self.op,
            "params": _serialize_value(self.params),
            "target": self.target_desc,
            "property": self.property_name,
            "old": _serialize_value(self.old_value),
            "new": _serialize_value(self.new_value),
            "code": self.code_line,
        }

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> "EditRecord":
        """Deserialize. The ``code`` field is kept only as text to show;
        replay and code generation use ``op`` / ``params``."""
        return cls(
            target_desc=d.get("target", ""),
            property_name=d.get("property", ""),
            old_value=d.get("old"),
            new_value=d.get("new"),
            code_line=d.get("code", ""),
            op=d.get("op"),
            params=d.get("params"),
        )


def _serialize_value(val: Any) -> Any:
    """Convert matplotlib/numpy values to JSON-safe Python types."""
    import numpy as np

    if isinstance(val, np.ndarray):
        return val.tolist()
    if isinstance(val, np.generic):
        return val.item()
    if isinstance(val, (list, tuple)):
        return [_serialize_value(v) for v in val]
    if isinstance(val, dict):
        return {str(k): _serialize_value(v) for k, v in val.items()}
    if val is None or isinstance(val, (bool, int, float, str)):
        return val
    return repr(val)


def _axvar(i: int) -> str:
    return "ax" if i == 0 else f"ax{i}"


@dataclass
class FigureEditor(EditMethods):
    """
    Editor that wraps a matplotlib figure and tracks modifications.

    Classifies every artist (data, fit, interval, reference line, label,
    ...), applies cosmetic edits as named operations, keeps an exact undo
    history and generates reproducible code.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to edit.
    protect_data : bool, default True
        Keep the statistical content intact: axis limits never hide data,
        error bars or bands; data cannot be made invisible; colour-mapped
        colours and numeric result labels cannot be changed. Setting this to
        False is strongly discouraged for statistical plots.
    coalesce : bool, default False
        Merge consecutive edits of the same property into one history entry
        (the Jupyter panel turns this on so one slider drag is one undo step).

    Attributes
    ----------
    edits : list of EditRecord
        History of all modifications.
    artist_roles : dict
        Mapping from ``id(artist)`` to :class:`ArtistRole`.
    """

    fig: Any
    protect_data: bool = True
    coalesce: bool = False
    edits: List[EditRecord] = field(default_factory=list)
    artist_roles: Dict[int, ArtistRole] = field(default_factory=dict)
    _on_refresh_callbacks: List[Any] = field(default_factory=list, repr=False)
    _redo_stack: List[EditRecord] = field(default_factory=list, repr=False)
    _groups: Dict[int, List[Dict[str, Any]]] = field(default_factory=dict, repr=False)
    _legend_sources: Dict[int, List[Optional[int]]] = field(
        default_factory=dict, repr=False
    )
    _batch_depth: int = field(default=0, repr=False)
    _dirty: bool = field(default=False, repr=False)
    _serial: int = field(default=0, repr=False)

    def __post_init__(self) -> None:
        self.artist_roles.update(classify_figure(self.fig))
        for i, ax in enumerate(self.fig.get_axes()):
            self._groups[i] = color_groups(ax, self.artist_roles)
            sources = legend_sources(ax)
            if sources is not None:
                self._legend_sources[i] = sources

    # ------------------------------------------------------------------
    # Classification
    # ------------------------------------------------------------------

    def is_editable(self, artist: Any) -> bool:
        """True unless the artist is locked statistical content."""
        if not self.protect_data:
            return True
        return (
            self.artist_roles.get(id(artist), ArtistRole.COSMETIC) not in LOCKED_ROLES
        )

    def color_groups(self, ax_index: int = 0) -> List[Dict[str, Any]]:
        """Series of one axes, grouped by the colour they were drawn in.

        Each group is ``{"color": "#1f77b4", "members": [("line", 0), ...]}``.
        Groups are fixed when the editor is created, so their numbering does
        not change as series are recoloured.
        """
        self._ax(ax_index)
        if ax_index not in self._groups:
            self._groups[ax_index] = color_groups(
                self.fig.get_axes()[ax_index], self.artist_roles
            )
        return self._groups[ax_index]

    def _ax(self, i: Any) -> Any:
        axes = self.fig.get_axes()
        if not isinstance(i, int) or not 0 <= i < len(axes):
            raise MethodIncompatibility(
                f"ax_index {i!r} out of range (figure has {len(axes)} axes)"
            )
        return axes[i]

    def _next_serial(self) -> int:
        self._serial += 1
        return self._serial

    def _refresh_legends(self, axes: List[int]) -> Dict[int, List[Optional[int]]]:
        """Rebuild the legend entries of ``axes`` after a restyle; returns
        the entry-to-artist map used for each (for the generated code)."""
        out: Dict[int, List[Optional[int]]] = {}
        all_axes = self.fig.get_axes()
        for i in axes:
            ax = all_axes[i]
            legend = ax.get_legend()
            if legend is None:
                continue
            sources = self._legend_sources.get(i)
            if sources is None or len(sources) != len(legend.get_texts()):
                sources = legend_sources(ax) or []
                self._legend_sources[i] = sources
            if not any(s is not None for s in sources):
                continue
            refresh_legend(ax, sources)
            out[i] = list(sources)
        return out

    def _get_data_range(
        self, ax: Any, axis: str
    ) -> Tuple[Optional[float], Optional[float]]:
        """Min/max of all locked content on one axis (data units)."""
        return locked_extent(ax, axis, self.artist_roles)

    # ------------------------------------------------------------------
    # Core: run one operation
    # ------------------------------------------------------------------

    def _run(self, name: str, params: Dict[str, Any]) -> Optional[EditRecord]:
        spec = OPS.get(name)
        if spec is None:
            raise MethodIncompatibility(
                f"Unknown operation {name!r}. Supported: {sorted(OPS)}"
            )
        p = spec.normalize(params)
        if "ax_index" in p and p["ax_index"] is not None:
            self._ax(p["ax_index"])
        snap = capture(self.fig, spec.axes(p, len(self.fig.get_axes())))
        try:
            resolved = spec.apply(self, dict(p))
            if resolved is not None and spec.restyles(resolved):
                legends = self._refresh_legends(_touched_axes(resolved, snap))
                if legends:
                    resolved = {**resolved, "_legend": legends}
        except Exception:
            restore(self.fig, snap)
            raise
        if resolved is None:
            return None
        target, prop = spec.legacy(resolved)
        record = EditRecord(
            target_desc=target,
            property_name=prop,
            old_value=None,
            new_value=_new_value(name, resolved),
            code_line="\n".join(spec.code(resolved, _axvar)),
            op=name,
            params=p,
            resolved=resolved,
            snapshot=snap,
            slot=spec.slot(resolved) if spec.slot else None,
        )
        last = self.edits[-1] if self.edits else None
        if (
            self.coalesce
            and last is not None
            and record.slot is not None
            and last.slot == record.slot
            and last.op == record.op
        ):
            record.snapshot = last.snapshot
            self.edits[-1] = record
        else:
            self.edits.append(record)
        self._redo_stack.clear()
        self._refresh()
        return record

    def run(self, op: str, **params: Any) -> None:
        """Apply any registered operation by name (see ``operations()``)."""
        self._run(op, params)

    @staticmethod
    def operations() -> List[Dict[str, Any]]:
        """Schema of every operation: name, description, typed parameters."""
        return [OPS[name].schema() for name in sorted(OPS)]

    @contextlib.contextmanager
    def batch(self) -> Iterator["FigureEditor"]:
        """Apply several edits with one redraw at the end."""
        self._batch_depth += 1
        try:
            yield self
        finally:
            self._batch_depth -= 1
            if self._batch_depth == 0 and self._dirty:
                self._dirty = False
                self._refresh()

    # ------------------------------------------------------------------
    # History
    # ------------------------------------------------------------------

    def undo(self) -> None:
        """Undo the last edit exactly."""
        if not self.edits:
            return
        record = self.edits.pop()
        if record.snapshot is not None:
            restore(self.fig, record.snapshot)
        self._redo_stack.append(record)
        self._refresh()

    def redo(self) -> None:
        """Redo the last undone edit."""
        if not self._redo_stack:
            return
        record = self._redo_stack.pop()
        redo_stack = list(self._redo_stack)
        coalesce, self.coalesce = self.coalesce, False
        try:
            assert record.op is not None
            self._run(record.op, dict(record.params or {}))
        finally:
            self.coalesce = coalesce
        self._redo_stack = redo_stack

    def reset(self) -> None:
        """Undo every edit, returning the figure to its original state."""
        with self.batch():
            while self.edits:
                self.undo()
        self._redo_stack.clear()
        self._refresh()

    def on_refresh(self, callback: Any) -> None:
        """Register ``callback(fig)`` to run after every change."""
        self._on_refresh_callbacks.append(callback)

    def _refresh(self) -> None:
        """Redraw where something displays the figure, and notify callbacks."""
        if self._batch_depth:
            self._dirty = True
            return
        canvas = getattr(self.fig, "canvas", None)
        if canvas is not None and _is_gui_canvas(canvas):
            try:
                canvas.draw_idle()
            except Exception:
                logger.debug("Canvas draw failed", exc_info=True)
        for cb in list(self._on_refresh_callbacks):
            try:
                cb(self.fig)
            except Exception:
                logger.debug("Refresh callback %r failed", cb, exc_info=True)

    # ------------------------------------------------------------------
    # Code generation
    # ------------------------------------------------------------------

    def _effective_edits(self) -> List[EditRecord]:
        """Edits whose effect survives: an edit is dropped when a later edit
        sets the same slot (every operation is a full setter of its slot)."""
        last_of_slot: Dict[tuple, int] = {}
        for k, e in enumerate(self.edits):
            if e.slot is not None:
                last_of_slot[e.slot] = k
        return [
            e
            for k, e in enumerate(self.edits)
            if e.slot is None or last_of_slot[e.slot] == k
        ]

    def generate_code(self, include_comment: bool = True) -> str:
        """
        Reproducible matplotlib code for all edits.

        Run it after the code that draws ``fig``; it needs only the name
        ``fig`` and defines ``ax`` (``ax1``, ... for further axes) itself.

        Returns
        -------
        str
            Python source.
        """
        if not self.edits:
            return "# No edits made"
        used: List[int] = []

        def axv(i: int) -> str:
            if i not in used:
                used.append(i)
            return _axvar(i)

        body: List[str] = []
        for e in self._effective_edits():
            assert e.op is not None and e.resolved is not None
            body.extend(OPS[e.op].code(e.resolved, axv))
            for i, sources in e.resolved.get("_legend", {}).items():
                body.append(f"sp.refresh_legend({axv(i)}, {lit(sources)})")
        preamble = [f"{_axvar(i)} = fig.axes[{i}]" for i in sorted(used)]
        if any(line.startswith("sp.") for line in body):
            preamble.insert(0, "import statspai as sp")
        lines: List[str] = []
        if include_comment:
            lines.append(
                "# --- StatsPAI figure edits (run after the code that draws `fig`) ---"
            )
        lines += preamble + body
        if include_comment:
            lines.append("# --- end StatsPAI edits ---")
        return "\n".join(lines)

    def copy_code(self) -> None:
        """Print reproducible code to stdout."""
        print(self.generate_code())

    def summary(self) -> str:
        """Return a summary of all edits."""
        if not self.edits:
            return "No edits made."
        lines = [f"StatsPAI figure editor: {len(self.edits)} edit(s)"]
        for i, edit in enumerate(self.edits, 1):
            lines.append(f"  {i}. {edit.op}: {_short(edit.params)}")
        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Serialization: saved styles, web backends
    # ------------------------------------------------------------------

    def to_edits_list(self) -> List[Dict[str, Any]]:
        """Edits as a list of dicts (``op`` + ``params``, plus v1 fields)."""
        return [e.to_dict() for e in self.edits]

    def to_edits_json(self) -> str:
        """Edits as a JSON string; replay with :meth:`apply_edits_json`."""
        import json

        return json.dumps(self.to_edits_list(), ensure_ascii=False)

    def apply_edits_list(self, edits: List[Dict[str, Any]]) -> None:
        """Replay saved edits on this figure.

        Each entry is an operation (``{"op": ..., "params": ...}``) or a
        version-1 record (``target`` / ``property`` / ``new``). A ``code``
        field is never executed or copied into the generated code. Entries
        that do not apply to this figure (a line it does not have) are
        skipped with a warning.
        """
        with self.batch():
            for d in edits:
                if d.get("op"):
                    name, params = d["op"], dict(d.get("params") or {})
                else:
                    translated = _legacy_to_op(d)
                    if translated is None:
                        warnings.warn(
                            f"Skipped unrecognised edit record {d.get('target')!r}."
                            f"{d.get('property')!r}",
                            UserWarning,
                            stacklevel=2,
                        )
                        continue
                    name, params = translated
                try:
                    self._run(name, params)
                except ValueError as exc:
                    warnings.warn(
                        f"Skipped edit {name}: {exc}", UserWarning, stacklevel=2
                    )

    def apply_edits_json(self, json_str: str) -> None:
        """Replay edits from a JSON string (see :meth:`apply_edits_list`)."""
        import json

        self.apply_edits_list(json.loads(json_str))

    def export_state(self) -> Dict[str, Any]:
        """Editor state for persistence; restore with :meth:`import_state`."""
        w, h = self.fig.get_size_inches()
        return {
            "version": STATE_VERSION,
            "edits": self.to_edits_list(),
            "figure": {"width": float(w), "height": float(h), "dpi": int(self.fig.dpi)},
            "code": self.generate_code(include_comment=False),
        }

    def import_state(self, state: Dict[str, Any]) -> None:
        """Replay a state from :meth:`export_state` (version 1 or 2)."""
        with self.batch():
            if int(state.get("version", 1)) < 2:
                fig_state = state.get("figure", {})
                if "width" in fig_state and "height" in fig_state:
                    self.set_figsize(fig_state["width"], fig_state["height"])
                if "dpi" in fig_state:
                    self.set_dpi(fig_state["dpi"])
            self.apply_edits_list(state.get("edits", []))

    def apply_actions(self, actions: List[Dict[str, Any]]) -> None:
        """Apply a list of action dicts, the API for web front ends and agents.

        Each dict has an ``action`` key naming an operation (see
        :meth:`operations`) plus its parameters, e.g.::

            {"action": "set_title", "text": "Effect of X", "ax_index": 0}
            {"action": "set_series_color", "group": 0, "color": "#D55E00"}
            {"action": "apply_theme", "theme": "aea"}

        Unknown actions or parameters raise ``ValueError``; nothing after the
        failing action is applied.
        """
        with self.batch():
            for action in actions:
                name, params = _action_to_op(action)
                self._run(name, params)

    def get_editable_schema(self) -> Dict[str, Any]:
        """Editable properties, current values, roles and the operation
        list, for building an editing panel."""
        from ._editor_io import editable_schema

        return editable_schema(self)

    def _role_name(self, artist: Any) -> str:
        return self.artist_roles.get(id(artist), ArtistRole.COSMETIC).name

    # ------------------------------------------------------------------
    # Output
    # ------------------------------------------------------------------

    def save(
        self,
        filename: str,
        dpi: int = 300,
        fmt: Optional[str] = None,
        embed_fonts: bool = True,
        **kwargs: Any,
    ) -> None:
        """Save the figure.

        Parameters
        ----------
        filename : str
            Output path ('figure.pdf', 'figure.png', ...).
        dpi : int, default 300
            Resolution for raster formats.
        fmt : str, optional
            Explicit format; inferred from the extension if omitted.
        embed_fonts : bool, default True
            Embed TrueType (Type 42) fonts in PDF/PS output instead of
            matplotlib's default Type 3 fonts, which many journals and
            submission systems reject. The figure looks the same.
        **kwargs
            Passed to ``fig.savefig()``.
        """
        from ._editor_io import save_figure

        save_figure(self.fig, filename, dpi, fmt, embed_fonts, **kwargs)

    def export(
        self,
        stem: str,
        formats: Sequence[str] = ("pdf", "png"),
        dpi: int = 300,
        embed_fonts: bool = True,
    ) -> List[str]:
        """Save ``stem.<fmt>`` for every format; returns the paths."""
        paths = []
        for fmt in formats:
            path = f"{stem}.{fmt}"
            self.save(path, dpi=dpi, fmt=fmt, embed_fonts=embed_fonts)
            paths.append(path)
        return paths

    def render_to_base64(self, dpi: int = 300, fmt: str = "png") -> str:
        """Render the figure to a base64-encoded string."""
        import base64

        return base64.b64encode(self.preview(dpi=dpi, fmt=fmt)).decode("ascii")

    def preview(
        self, dpi: int = 100, fmt: str = "png", grayscale: bool = False
    ) -> bytes:
        """Render to bytes; ``grayscale=True`` shows how the figure prints in
        black and white."""
        from ._editor_io import preview_bytes

        return preview_bytes(self.fig, dpi, fmt, grayscale)

    def check(self, min_fontsize: float = 6.0) -> List[Dict[str, Any]]:
        """Pre-submission check of the figure as it stands.

        Reports text smaller than ``min_fontsize`` points at the current
        figure size, series that cannot be told apart in grayscale, and
        series drawn almost in the background colour.

        Returns
        -------
        list of dict
            ``{"code", "severity", "message", "ax_index"}``; empty if clean.
        """
        from ._editor_check import run_checks

        return run_checks(self, min_fontsize=min_fontsize)


def _touched_axes(resolved: Dict[str, Any], snap: Snapshot) -> List[int]:
    """Axes an edit restyled: the ones it names, else all it captured."""
    if resolved.get("per_axes"):
        return [int(b["ax_index"]) for b in resolved["per_axes"]]
    if resolved.get("ax_index") is not None:
        return [int(resolved["ax_index"])]
    return [i for i, _ in snap.axes]


def _new_value(name: str, resolved: Dict[str, Any]) -> Any:
    for key in ("text", "size", "color", "visible", "alpha", "preset", "theme", "dpi"):
        if key in resolved:
            return resolved[key]
    if name in ("set_xlim", "set_ylim"):
        keys = [k for k in ("left", "right", "bottom", "top") if k in resolved]
        return tuple(resolved[k] for k in keys)
    if name == "set_figsize":
        return (resolved["width"], resolved["height"])
    return True


def _short(params: Any) -> str:
    text = repr({k: v for k, v in (params or {}).items() if v not in (None, {}, [])})
    return text if len(text) < 90 else text[:87] + "..."


def _is_gui_canvas(canvas: Any) -> bool:
    """Whether the canvas is shown in a window or widget that needs redrawing.

    Static canvases (Agg, PDF, SVG, the Jupyter inline backend) are only
    drawn when saved, so redrawing them on every edit is wasted work.
    """
    name = type(canvas).__name__
    module = type(canvas).__module__
    static = {
        "FigureCanvasAgg",
        "FigureCanvasBase",
        "FigureCanvasPdf",
        "FigureCanvasSVG",
        "FigureCanvasPS",
        "FigureCanvasPgf",
        "FigureCanvasCairo",
        "FigureCanvasTemplate",
    }
    return not (name in static or module.startswith("matplotlib_inline"))


# ------------------------------------------------------------------
# Public API
# ------------------------------------------------------------------


def interactive(
    fig: Any,
    protect_data: bool = True,
    actions: Optional[List[Dict[str, Any]]] = None,
    ui: Literal["auto", "jupyter", "window", "none"] = "auto",
) -> FigureEditor:
    """
    Open an editor for a matplotlib figure.

    Cosmetic edits only: titles, labels, fonts, colours, legend, layout and
    annotations. Data points, fitted lines, error bars and confidence bands
    stay locked, and every edit has reproducible code
    (``editor.generate_code()``).

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        The figure to edit. Typically from ``result.plot()`` or
        ``sp.binscatter()``.
    protect_data : bool, default True
        Lock data-representing elements (scatter points, regression
        lines, confidence intervals) from modification.
    actions : list of dict, optional
        Edits to apply straight away, each ``{"action": <operation>, ...}``
        (see ``FigureEditor.operations()``), e.g.
        ``[{"action": "set_title", "text": "Event study"}]``. This is how an
        agent or a script edits a figure without a user interface.
    ui : {'auto', 'jupyter', 'window', 'none'}, default 'auto'
        Where to show the controls. 'auto' picks the Jupyter panel in a
        notebook, a control window with a GUI matplotlib backend, and no
        interface otherwise (the figure is then left untouched apart from
        ``actions``).

    Returns
    -------
    FigureEditor
        The editor instance. Call ``.generate_code()`` to get
        reproducible code, or ``.copy_code()`` to print it.

    Examples
    --------
    >>> import matplotlib
    >>> matplotlib.use("Agg")
    >>> import matplotlib.pyplot as plt
    >>> import statspai as sp
    >>> fig, ax = plt.subplots()
    >>> _ = ax.plot([0, 1, 2], [0, 1, 4])
    >>> edits = [{"action": "set_title", "text": "Growth"}]
    >>> editor = sp.interactive(fig, actions=edits)
    >>> ax.get_title()
    'Growth'
    >>> print(editor.generate_code(include_comment=False))
    ax = fig.axes[0]
    ax.title.set_text('Growth')
    """
    if ui not in ("auto", "jupyter", "window", "none"):
        raise MethodIncompatibility(
            f"ui must be 'auto', 'jupyter', 'window' or 'none', got {ui!r}",
            recovery_hint="Use ui='none' to edit through actions= or the "
            "editor's methods without any interface.",
        )
    editor = getattr(fig, "_statspai_editor", None)
    if not isinstance(editor, FigureEditor) or editor.protect_data != protect_data:
        editor = FigureEditor(fig=fig, protect_data=protect_data)
        # Stored on the figure so it lives and dies with it.
        fig._statspai_editor = editor
    if actions:
        editor.apply_actions(actions)

    mode = ui
    if mode == "auto":
        if _in_jupyter():
            mode = "jupyter"
        elif _is_gui_canvas(fig.canvas):
            mode = "window"
        else:
            mode = "none"
    if mode == "jupyter":
        try:
            import ipywidgets  # noqa: F401
        except ImportError:
            warnings.warn(
                "The Jupyter editor panel needs ipywidgets "
                "(pip install ipywidgets); the editor works without it through "
                "its methods, e.g. editor.set_title(...).",
                UserWarning,
                stacklevel=2,
            )
            return editor
        from ._jupyter_editor import create_jupyter_panel

        editor._on_refresh_callbacks.clear()
        editor.coalesce = True
        create_jupyter_panel(editor)
    elif mode == "window":
        from ._script_editor import create_script_editor

        editor._on_refresh_callbacks.clear()
        create_script_editor(editor)
    return editor


def get_code(fig: Any) -> str:
    """
    Get reproducible code for all interactive edits made to a figure.

    Parameters
    ----------
    fig : matplotlib.figure.Figure

    Returns
    -------
    str
        Python code string.

    Examples
    --------
    >>> import matplotlib
    >>> matplotlib.use("Agg")
    >>> import matplotlib.pyplot as plt
    >>> import statspai as sp
    >>> fig, ax = plt.subplots()
    >>> _ = ax.plot([0, 1, 2], [0, 1, 4])
    >>> code = sp.get_code(fig)  # reproducible code for any interactive edits
    >>> isinstance(code, str)
    True
    """
    editor = getattr(fig, "_statspai_editor", None)
    if not isinstance(editor, FigureEditor):
        return "# No interactive edits found for this figure"
    return editor.generate_code()


def _in_jupyter() -> bool:
    """Detect if running in a Jupyter-like notebook (Jupyter, Colab, VSCode)."""
    try:
        from IPython import get_ipython

        shell = get_ipython()
        if shell is None:
            return False
        name = shell.__class__.__name__
        # ZMQInteractiveShell: standard Jupyter
        # Shell: Google Colab
        # Also check for ipykernel module (VSCode notebooks)
        return name in ("ZMQInteractiveShell", "Shell") or (
            hasattr(shell, "kernel") and shell.kernel is not None
        )
    except (ImportError, NameError, AttributeError):
        return False
