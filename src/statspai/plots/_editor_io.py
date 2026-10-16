"""Figure editor I/O: the editing schema, saving, previews, and the
translation of version-1 edit records and legacy action names."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, Dict, Optional, Tuple

from ..exceptions import MethodIncompatibility
from ._editor_ops import OPS, all_palettes
from ._editor_presets import (
    FONT_CHOICES,
    FONT_PRESETS,
    SIZE_PRESETS,
    _resolve_preset_fonts,
)

if TYPE_CHECKING:
    from .interactive import FigureEditor


def editable_schema(editor: "FigureEditor") -> Dict[str, Any]:
    """See ``FigureEditor.get_editable_schema``."""
    from .themes import list_themes

    axes_schema = []
    for i, ax in enumerate(editor.fig.get_axes()):
        lines_schema = [
            {
                "index": j,
                "label": line.get_label() or f"line{j}",
                "role": editor._role_name(line),
                "editable": editor.is_editable(line),
                "color": line.get_color(),
                "linewidth": line.get_linewidth(),
                "linestyle": line.get_linestyle(),
                "marker": line.get_marker(),
                "markersize": line.get_markersize(),
                "alpha": line.get_alpha(),
            }
            for j, line in enumerate(ax.get_lines())
        ]
        collections_schema = [
            {
                "index": j,
                "type": type(coll).__name__,
                "role": editor._role_name(coll),
                "editable": editor.is_editable(coll),
                "colormapped": coll.get_array() is not None,
                "alpha": coll.get_alpha(),
            }
            for j, coll in enumerate(ax.collections)
        ]
        texts_schema = [
            {
                "index": j,
                "text": t.get_text(),
                "role": editor._role_name(t),
                "visible": t.get_visible(),
                "wording_locked": not editor.is_editable(t),
            }
            for j, t in enumerate(ax.texts)
        ]
        gridlines = ax.xaxis.get_gridlines()
        legend = ax.get_legend()
        legend_info = None
        if legend is not None:
            from ._editor_ops_layout import _loc_name

            legend_info = {
                "visible": legend.get_visible(),
                "loc": _loc_name(legend),
                "labels": [t.get_text() for t in legend.get_texts()],
            }
        axes_schema.append(
            {
                "index": i,
                "title": {
                    "text": ax.get_title(),
                    "fontsize": ax.title.get_fontsize(),
                    "color": ax.title.get_color(),
                },
                "xlabel": {
                    "text": ax.get_xlabel(),
                    "fontsize": ax.xaxis.label.get_fontsize(),
                },
                "ylabel": {
                    "text": ax.get_ylabel(),
                    "fontsize": ax.yaxis.label.get_fontsize(),
                },
                "xlim": [float(v) for v in ax.get_xlim()],
                "ylim": [float(v) for v in ax.get_ylim()],
                "lines": lines_schema,
                "collections": collections_schema,
                "texts": texts_schema,
                "color_groups": [
                    {"index": g, "color": grp["color"], "members": grp["members"]}
                    for g, grp in enumerate(editor.color_groups(i))
                ],
                "spines": {n: s.get_visible() for n, s in ax.spines.items()},
                "grid": gridlines[0].get_visible() if gridlines else False,
                "legend": legend_info,
            }
        )
    w, h = editor.fig.get_size_inches()
    return {
        "axes": axes_schema,
        "figure": {
            "width": float(w),
            "height": float(h),
            "dpi": int(editor.fig.dpi),
            "facecolor": editor.fig.get_facecolor(),
        },
        "font_presets": [
            {
                "name": name,
                "font": _resolve_preset_fonts(preset)[0],
                "family": preset["family"],
                "venue": preset.get("venue", ""),
                "title_size": preset["title_size"],
                "label_size": preset["label_size"],
                "tick_size": preset["tick_size"],
            }
            for name, preset in FONT_PRESETS.items()
        ],
        "size_presets": [
            {
                "name": name,
                "desc": sp.get("desc", ""),
                "title_size": sp["title_size"],
                "label_size": sp["label_size"],
                "tick_size": sp["tick_size"],
            }
            for name, sp in SIZE_PRESETS.items()
        ],
        "themes": list_themes(),
        "palettes": all_palettes(),
        "operations": editor.operations(),
        "protect_data": editor.protect_data,
    }


def save_figure(
    fig: Any,
    filename: str,
    dpi: int,
    fmt: Optional[str],
    embed_fonts: bool,
    **kwargs: Any,
) -> None:
    import matplotlib as mpl

    save_kwargs: Dict[str, Any] = {
        "dpi": dpi,
        "bbox_inches": "tight",
        "facecolor": fig.get_facecolor(),
    }
    if fmt is not None:
        save_kwargs["format"] = fmt
    save_kwargs.update(kwargs)
    rc = {"pdf.fonttype": 42, "ps.fonttype": 42} if embed_fonts else {}
    with mpl.rc_context(rc):
        fig.savefig(filename, **save_kwargs)


def preview_bytes(fig: Any, dpi: int, fmt: str, grayscale: bool) -> bytes:
    import io

    buf = io.BytesIO()
    fig.savefig(
        buf,
        format=fmt,
        dpi=dpi,
        bbox_inches="tight",
        facecolor=fig.get_facecolor(),
    )
    data = buf.getvalue()
    if grayscale:
        if fmt != "png":
            raise MethodIncompatibility("preview(grayscale=True) needs fmt='png'")
        import matplotlib.image as mimage
        import numpy as np

        img = mimage.imread(io.BytesIO(data))
        lum = img[..., 0] * 0.299 + img[..., 1] * 0.587 + img[..., 2] * 0.114
        out = io.BytesIO()
        mimage.imsave(
            out, np.clip(lum, 0, 1), cmap="gray", vmin=0, vmax=1, format="png"
        )
        data = out.getvalue()
    return data


# ----------------------------------------------------------------------
# Version-1 records and legacy action names
# ----------------------------------------------------------------------


def _ax_index_of(target: str) -> int:
    m = re.match(r"^ax(\d+)\.", target)
    return int(m.group(1)) if m else 0


def _legacy_to_op(d: Dict[str, Any]) -> Optional[Tuple[str, Dict[str, Any]]]:
    """Translate a version-1 edit record to an operation (``None`` if the
    record has no counterpart)."""
    td, prop, val = d.get("target", ""), d.get("property", ""), d.get("new")
    i = _ax_index_of(td)
    leaf = td.split(".")[-1]
    ax = {"ax_index": i}
    if prop == "save":
        return None
    if prop == "text" and leaf in ("title", "xlabel", "ylabel"):
        return f"set_{leaf}", {"text": val, **ax}
    if prop == "fontsize" and leaf in ("title", "xlabel", "ylabel", "ticks"):
        return "set_fontsize", {"target": leaf, "size": val, **ax}
    if prop == "range" and leaf in ("xlim", "ylim"):
        lo, hi = ("left", "right") if leaf == "xlim" else ("bottom", "top")
        return f"set_{leaf}", {lo: val[0], hi: val[1], **ax}
    if prop == "visible" and ".spines." in td:
        return "set_spine_visible", {"spine": leaf, "visible": val, **ax}
    if leaf == "grid" and prop == "visible":
        return "set_grid", {"visible": val, **ax}
    if leaf == "grid" and prop == "style" and isinstance(val, dict):
        return "set_grid_style", {**val, **ax}
    if td == "fig" and prop == "figsize":
        return "set_figsize", {"width": val[0], "height": val[1]}
    if td == "fig" and prop == "dpi":
        return "set_dpi", {"dpi": val}
    if td == "fig" and prop == "tight_layout":
        return "tight_layout", {}
    if prop == "color" and leaf in ("title", "xlabel", "ylabel"):
        return "set_color", {"target": leaf, "color": val, **ax}
    if prop == "color" and leaf.startswith("line"):
        return "set_color", {"target": leaf, "color": val, **ax}
    if prop == "color" and leaf.startswith("scatter"):
        return "set_scatter_color", {"scatter_index": int(leaf[7:]), "color": val, **ax}
    if prop == "color" and leaf == "facecolor":
        target = "figure" if td == "fig.facecolor" else "axes"
        return "set_background_color", {"color": val, "target": target, **ax}
    if prop in ("linewidth", "linestyle", "marker") and leaf.startswith("line"):
        key = {"linewidth": "width", "linestyle": "style", "marker": "marker"}[prop]
        return f"set_{prop}", {"line_index": int(leaf[4:]), key: val, **ax}
    if prop == "alpha":
        return "set_alpha", {"target": leaf, "alpha": val, **ax}
    if td == "theme":
        return "apply_theme", {"theme": val}
    if leaf == "font_preset":
        return "apply_font_preset", {"preset": val, **ax}
    if leaf == "font" and prop == "family":
        if val in ("serif", "sans-serif", "monospace"):
            return "set_font", {"font_family": val, **ax}
        serif = FONT_CHOICES()["English Serif"]
        family = "serif" if str(val).split(" (")[0] in serif else "sans-serif"
        return "set_font", {"font_family": family, "font_name": val, **ax}
    if leaf == "legend" and prop == "visible":
        return "set_legend_visible", {"visible": val, **ax}
    if leaf == "legend" and prop == "properties" and isinstance(val, dict):
        return "set_legend", {**val, **ax}
    if leaf in ("xticks", "yticks") and prop == "rotation":
        return "set_tick_rotation", {"axis": leaf[0], "angle": val, **ax}
    if leaf == "title" and prop == "fontweight":
        return "set_title_weight", {"weight": val, **ax}
    if leaf == "annotation" and prop == "add" and isinstance(val, dict):
        return "add_annotation", {
            "text": val["text"],
            "xy": val["xy"],
            "kwargs": val.get("kwargs", {}),
            **ax,
        }
    return None


_ACTION_ALIASES = {
    "set_line_color": lambda a: (
        "set_color",
        {"target": f"line{a.pop('line_index')}", **a},
    ),
    "set_line_alpha": lambda a: (
        "set_alpha",
        {"target": f"line{a.pop('line_index')}", **a},
    ),
}


def _action_to_op(action: Dict[str, Any]) -> Tuple[str, Dict[str, Any]]:
    params = dict(action)
    name = params.pop("action", None)
    if not isinstance(name, str):
        raise MethodIncompatibility(
            f"Unknown action: {name!r}. Supported: "
            f"{sorted(set(OPS) | set(_ACTION_ALIASES))}"
        )
    if name in _ACTION_ALIASES:
        return _ACTION_ALIASES[name](params)
    if name not in OPS:
        raise MethodIncompatibility(
            f"Unknown action: {name!r}. Supported: "
            f"{sorted(set(OPS) | set(_ACTION_ALIASES))}"
        )
    return name, params
