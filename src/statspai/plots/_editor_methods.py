"""Typed methods of ``FigureEditor``, one per edit operation.

Each method validates nothing itself; it forwards its arguments to the
operation of the same name (``_editor_ops*``), which checks them, applies the
edit and records how to undo and reproduce it.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple

from ._editor_ops import all_palettes


class EditMethods:
    """Mixin: the public editing methods of :class:`FigureEditor`."""

    fig: Any

    def _run(self, name: str, params: Dict[str, Any]) -> Any:  # pragma: no cover
        raise NotImplementedError

    # ------------------------------------------------------------------

    def set_title(self, text: str, ax_index: int = 0, **kwargs: Any) -> None:
        """Set the axes title text (keeps its size, weight and colour)."""
        self._run("set_title", {"text": text, "ax_index": ax_index, "kwargs": kwargs})

    def set_xlabel(self, text: str, ax_index: int = 0, **kwargs: Any) -> None:
        """Set the x-axis label."""
        self._run("set_xlabel", {"text": text, "ax_index": ax_index, "kwargs": kwargs})

    def set_ylabel(self, text: str, ax_index: int = 0, **kwargs: Any) -> None:
        """Set the y-axis label."""
        self._run("set_ylabel", {"text": text, "ax_index": ax_index, "kwargs": kwargs})

    def set_suptitle(self, text: str, **kwargs: Any) -> None:
        """Set or create the figure-level title."""
        self._run("set_suptitle", {"text": text, "kwargs": kwargs})

    def set_xlim(self, left: float, right: float, ax_index: int = 0) -> None:
        """Set x-axis limits; with protect_data, never hides locked content."""
        self._run("set_xlim", {"left": left, "right": right, "ax_index": ax_index})

    def set_ylim(self, bottom: float, top: float, ax_index: int = 0) -> None:
        """Set y-axis limits; with protect_data, never hides locked content."""
        self._run("set_ylim", {"bottom": bottom, "top": top, "ax_index": ax_index})

    def set_fontsize(self, target: str, size: float, ax_index: int = 0) -> None:
        """Font size of 'title', 'xlabel', 'ylabel', 'ticks', 'legend' or
        'suptitle'."""
        self._run(
            "set_fontsize", {"target": target, "size": size, "ax_index": ax_index}
        )

    def set_font(
        self, font_family: str, font_name: Optional[str] = None, ax_index: int = 0
    ) -> None:
        """Use one font for all text of an axes.

        Parameters
        ----------
        font_family : str
            'serif', 'sans-serif', or 'monospace'.
        font_name : str, optional
            Specific font, e.g. 'Times New Roman', 'SimSun'. A display suffix
            such as ``"Songti SC (宋体)"`` is stripped.
        ax_index : int
            Which axes to apply to.
        """
        self._run(
            "set_font",
            {"font_family": font_family, "font_name": font_name, "ax_index": ax_index},
        )

    def apply_font_preset(self, preset_name: str, ax_index: int = 0) -> None:
        """Apply a font preset (font and sizes); see ``FONT_PRESETS``."""
        self._run("apply_font_preset", {"preset": preset_name, "ax_index": ax_index})

    def apply_size_preset(self, preset_name: str, ax_index: int = 0) -> None:
        """Apply a size preset without changing the font; see ``SIZE_PRESETS``."""
        self._run("apply_size_preset", {"preset": preset_name, "ax_index": ax_index})

    def set_color(self, target: str, color: Any, ax_index: int = 0) -> None:
        """Colour of 'title', 'xlabel', 'ylabel', 'suptitle' or 'line<j>'."""
        self._run("set_color", {"target": target, "color": color, "ax_index": ax_index})

    def set_series_color(self, group: int, color: Any, ax_index: int = 0) -> None:
        """Recolour one series (points, line, error bars and band together).

        ``group`` indexes :meth:`color_groups`.
        """
        self._run(
            "set_series_color", {"group": group, "color": color, "ax_index": ax_index}
        )

    def apply_palette(
        self,
        palette: str = "okabe_ito",
        colors: Optional[Sequence[Any]] = None,
        ax_index: Optional[int] = None,
    ) -> None:
        """Recolour all series with a palette (one colour per colour group)."""
        self._run(
            "apply_palette",
            {
                "palette": palette,
                "colors": list(colors) if colors is not None else None,
                "ax_index": ax_index,
            },
        )

    @staticmethod
    def palettes() -> Dict[str, List[str]]:
        """Available palettes for :meth:`apply_palette`."""
        return all_palettes()

    def set_spine_visible(
        self, spine_name: str, visible: bool, ax_index: int = 0
    ) -> None:
        """Show or hide a spine ('top', 'right', 'bottom', 'left')."""
        self._run(
            "set_spine_visible",
            {"spine": spine_name, "visible": visible, "ax_index": ax_index},
        )

    def set_grid(self, visible: bool, ax_index: int = 0, **kwargs: Any) -> None:
        """Show or hide the major grid."""
        self._run(
            "set_grid", {"visible": visible, "ax_index": ax_index, "kwargs": kwargs}
        )

    def set_grid_style(
        self,
        color: Optional[str] = None,
        alpha: Optional[float] = None,
        linestyle: Optional[str] = None,
        ax_index: int = 0,
    ) -> None:
        """Turn the grid on with a colour, alpha and/or line style."""
        self._run(
            "set_grid_style",
            {
                "color": color,
                "alpha": alpha,
                "linestyle": linestyle,
                "ax_index": ax_index,
            },
        )

    def set_figsize(self, width: float, height: float) -> None:
        """Figure size in inches."""
        self._run("set_figsize", {"width": width, "height": height})

    def set_dpi(self, dpi: int) -> None:
        """Figure DPI."""
        self._run("set_dpi", {"dpi": dpi})

    def set_legend(self, ax_index: int = 0, **kwargs: Any) -> None:
        """Legend position (``loc``), entry font size and frame."""
        if self.fig.get_axes()[ax_index].get_legend() is None:
            return
        self._run("set_legend", {"ax_index": ax_index, **kwargs})

    def set_legend_visible(self, visible: bool, ax_index: int = 0) -> None:
        """Show or hide the legend (creating it if needed)."""
        self._run("set_legend_visible", {"visible": visible, "ax_index": ax_index})

    def set_legend_labels(self, labels: Any, ax_index: int = 0) -> None:
        """Rename legend entries: a list in order, or ``{index: text}``."""
        self._run("set_legend_labels", {"labels": labels, "ax_index": ax_index})

    def set_linewidth(self, line_index: int, width: float, ax_index: int = 0) -> None:
        """Width of ``ax.get_lines()[line_index]``."""
        self._run(
            "set_linewidth",
            {"line_index": line_index, "width": width, "ax_index": ax_index},
        )

    def set_linestyle(self, line_index: int, style: str, ax_index: int = 0) -> None:
        """Style of ``ax.get_lines()[line_index]``."""
        self._run(
            "set_linestyle",
            {"line_index": line_index, "style": style, "ax_index": ax_index},
        )

    def set_marker(self, line_index: int, marker: str, ax_index: int = 0) -> None:
        """Marker of ``ax.get_lines()[line_index]``."""
        self._run(
            "set_marker",
            {"line_index": line_index, "marker": marker, "ax_index": ax_index},
        )

    def set_markersize(self, line_index: int, size: float, ax_index: int = 0) -> None:
        """Marker size of ``ax.get_lines()[line_index]``."""
        self._run(
            "set_markersize",
            {"line_index": line_index, "size": size, "ax_index": ax_index},
        )

    def set_alpha(self, target: str, alpha: float, ax_index: int = 0) -> None:
        """Transparency of 'line<j>', 'scatter<j>', 'ci<j>', 'coll<j>' or
        'patch<j>'."""
        self._run("set_alpha", {"target": target, "alpha": alpha, "ax_index": ax_index})

    def set_scatter_color(
        self, scatter_index: int, color: Any, ax_index: int = 0
    ) -> None:
        """Face colour of ``ax.collections[scatter_index]``."""
        self._run(
            "set_scatter_color",
            {"scatter_index": scatter_index, "color": color, "ax_index": ax_index},
        )

    def set_background_color(
        self, color: Any, target: str = "figure", ax_index: int = 0
    ) -> None:
        """Background of the 'figure' or the 'axes'."""
        self._run(
            "set_background_color",
            {"color": color, "target": target, "ax_index": ax_index},
        )

    def set_tick_rotation(self, axis: str, angle: float, ax_index: int = 0) -> None:
        """Rotate the tick labels of the 'x' or 'y' axis."""
        self._run(
            "set_tick_rotation", {"axis": axis, "angle": angle, "ax_index": ax_index}
        )

    def set_title_weight(self, weight: str, ax_index: int = 0) -> None:
        """Title font weight ('normal', 'bold', ...)."""
        self._run("set_title_weight", {"weight": weight, "ax_index": ax_index})

    def tight_layout(self) -> None:
        """Fit subplot margins to the labels."""
        self._run("tight_layout", {})

    def add_annotation(
        self,
        text: str,
        xy: Tuple[float, float],
        ax_index: int = 0,
        draggable: bool = False,
        **kwargs: Any,
    ) -> None:
        """Add a text annotation at data coordinates ``xy``.

        ``draggable=True`` lets it be dragged in a matplotlib GUI window
        (the drag itself is not recorded).
        """
        self._run(
            "add_annotation",
            {
                "text": text,
                "xy": xy,
                "ax_index": ax_index,
                "draggable": draggable,
                "kwargs": kwargs,
            },
        )

    def set_text(self, index: int, text: str, ax_index: int = 0) -> None:
        """Change the wording of ``ax.texts[index]`` (not numeric results
        drawn by the plotting function)."""
        self._run("set_text", {"index": index, "text": text, "ax_index": ax_index})

    def set_text_visible(self, index: int, visible: bool, ax_index: int = 0) -> None:
        """Show or hide ``ax.texts[index]``."""
        self._run(
            "set_text_visible",
            {"index": index, "visible": visible, "ax_index": ax_index},
        )

    def apply_theme(self, theme_name: str, recolor: bool = True) -> None:
        """Restyle this figure with a theme (StatsPAI, matplotlib or seaborn).

        Unlike ``sp.set_theme`` (global settings for figures drawn later),
        this changes the existing figure only.
        """
        self._run("apply_theme", {"theme": theme_name, "recolor": recolor})
