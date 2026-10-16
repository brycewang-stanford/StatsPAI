"""Rebuild legend entries after the artists they describe were restyled.

A matplotlib legend draws copies of the artists, made when the legend was
created. Recolouring a line afterwards leaves its legend entry in the old
colour, so the legend no longer says which series is which.
``refresh_legend`` rebuilds each entry with the legend's own handler, exactly
as matplotlib would have drawn it had the artist been styled that way from
the start.
"""

from __future__ import annotations

from typing import Any, List, Optional, Sequence

from ..exceptions import MethodIncompatibility


def legend_sources(ax: Any) -> Optional[List[Optional[int]]]:
    """For each legend entry, the index of the artist it was built from in
    ``ax.get_legend_handles_labels()[0]``, matched by label in order (each
    artist used once); ``None`` for entries with no labelled artist (proxy
    handles). ``None`` if the axes has no legend."""
    legend = ax.get_legend()
    if legend is None:
        return None
    _, labels = ax.get_legend_handles_labels()
    used: set = set()
    out: List[Optional[int]] = []
    for text in legend.get_texts():
        match = None
        for i, label in enumerate(labels):
            if i not in used and label == text.get_text():
                match = i
                break
        if match is not None:
            used.add(match)
        out.append(match)
    return out


def handle_boxes(legend: Any) -> List[Any]:
    """The drawing area of each legend entry, in entry order."""
    from matplotlib.offsetbox import DrawingArea, TextArea

    out: List[Any] = []

    def walk(node: Any) -> None:
        for child in node.get_children():
            if isinstance(child, DrawingArea):
                out.append(child)
            elif isinstance(child, TextArea):
                continue
            elif hasattr(child, "get_children"):
                walk(child)

    box = getattr(legend, "_legend_handle_box", None)
    if box is not None:
        walk(box)
    return out


def _handles_attr(legend: Any) -> str:
    return "legend_handles" if hasattr(legend, "legend_handles") else "legendHandles"


def refresh_legend(ax: Any, sources: Optional[Sequence[Optional[int]]] = None) -> None:
    """
    Redraw an axes' legend entries from the current style of their artists.

    A matplotlib legend copies the look of each artist when it is created;
    changing a line's colour, style or marker afterwards leaves the legend
    showing the old look. This rebuilds every entry with the legend's own
    handler, so the result is the legend matplotlib would have drawn had
    the artists been styled this way from the start. Position, frame, title,
    fonts and entry text are kept.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes whose legend to refresh. Nothing happens if it has no legend.
    sources : list of int or None, optional
        For each legend entry, the index of its artist in
        ``ax.get_legend_handles_labels()[0]`` (``None`` leaves that entry
        alone). By default entries are matched to artists by label, which
        is right unless entry texts were renamed after the legend was made.

    Returns
    -------
    None

    Examples
    --------
    >>> import matplotlib
    >>> matplotlib.use("Agg")
    >>> import matplotlib.pyplot as plt
    >>> import statspai as sp
    >>> fig, ax = plt.subplots()
    >>> line, = ax.plot([0, 1], [0, 1], color="blue", label="estimate")
    >>> _ = ax.legend()
    >>> line.set_color("red")        # the legend entry is still blue
    >>> sp.refresh_legend(ax)        # now it is red
    >>> handle = ax.get_legend().legend_handles[0]
    >>> handle.get_color()
    'red'
    """
    legend = ax.get_legend()
    if legend is None:
        return
    if sources is None:
        sources = legend_sources(ax) or []
    handles, _ = ax.get_legend_handles_labels()
    attr = _handles_attr(legend)
    entries = list(getattr(legend, attr))
    boxes = handle_boxes(legend)
    if len(sources) != len(entries):
        raise MethodIncompatibility(
            f"refresh_legend: {len(sources)} sources for {len(entries)} legend entries",
            recovery_hint=(
                "Pass one index per legend entry (None to skip an entry), or "
                "omit `sources` to match entries to artists by label."
            ),
        )
    if len(boxes) != len(entries):
        return  # a legend layout this function does not know; leave it as is
    handler_map = legend.get_legend_handler_map()
    fontsize = getattr(legend, "_fontsize", None)
    for k, src in enumerate(sources):
        if src is None:
            continue
        if not 0 <= int(src) < len(handles):
            raise MethodIncompatibility(
                f"refresh_legend: source {src} for entry {k} is not an index "
                f"into the {len(handles)} labelled artists",
                recovery_hint=(
                    "Indices refer to ax.get_legend_handles_labels()[0]; "
                    "use None for entries drawn from proxy artists."
                ),
            )
        orig = handles[int(src)]
        handler = legend.get_legend_handler(handler_map, orig)
        if handler is None:
            continue
        box = boxes[k]
        box._children = []
        entries[k] = handler.legend_artist(legend, orig, fontsize, box)
    setattr(legend, attr, entries)
    legend.stale = True
