"""``sp.gt(result)`` — :pkg:`great_tables` adapter for StatsPAI tables.

R's ``gt`` package (Iannone et al.) is the canonical publication-grade
table grammar: cell-level styling, spanners, footnote marks, themes,
multi-target rendering. Posit shipped a Python port (``great_tables``)
in 2024 and it is now stable at 0.21.

This module wires StatsPAI's existing table outputs
(:class:`RegtableResult`, :class:`PaperTables`, plain ``DataFrame``)
into a ``great_tables.GT`` instance so users can compose the polished
HTML / LaTeX / RTF / DOCX rendering on top of an estimator output
without re-implementing the table at the gt layer.

Design choices
--------------
- **Lazy import.** ``great_tables`` is **not** a hard dependency.
  Importing this module only fails (with a friendly hint) when the
  user actually calls :func:`to_gt`. The wider StatsPAI stack
  continues to import cleanly without it.
- **Pre-formatted input.** ``RegtableResult.to_dataframe()`` returns
  string-formatted cells (``"0.250***"`` / ``"(0.123)"`` /
  ``"R-squared"`` rows). We *don't* re-parse those — gt receives the
  rendered strings verbatim, so the numerical exactness of the
  underlying estimator is preserved. ``fmt_number`` would be a
  regression here (it would re-round already-rendered cells).
- **Journal preset → gt theme.** When a ``RegtableResult`` carries a
  ``template=`` (AER / QJE / Econometrica / …), we apply the matching
  font + footnote-marks + caps style via
  :meth:`great_tables.GT.opt_align_table_header` and
  :meth:`great_tables.GT.tab_options`. Star symbols are inherited
  from the StatsPAI cell strings — gt does not re-attach them.
- **Footer = StatsPAI footer**. ``rt.notes`` (and the
  ``Reproducibility:`` line when present) are emitted as
  :meth:`tab_source_note`; we don't fight gt's own footer system.

This is *opt-in* publication polish, not a replacement for
:func:`statspai.regtable` / :func:`statspai.modelsummary`. Users who
just want LaTeX / DOCX still get those directly from the result
object; ``sp.gt`` is the path to a ``great_tables.GT`` chain so they
can compose interactive HTML (``GT.show()``), embed in Quarto
(``draft.to_qmd()`` already understands gt sources), or apply
custom :meth:`tab_style` rules the StatsPAI primitives don't surface.
"""
from __future__ import annotations

from typing import TYPE_CHECKING, Any, List, Optional, Sequence, Union

import pandas as pd

if TYPE_CHECKING:  # pragma: no cover — type-only
    import great_tables as gt_pkg


__all__ = ["to_gt", "is_great_tables_available"]


# ---------------------------------------------------------------------------
# Soft dependency on great_tables
# ---------------------------------------------------------------------------

def is_great_tables_available() -> bool:
    """Return True iff ``great_tables`` can be imported in this env."""
    try:
        import great_tables  # noqa: F401
        return True
    except Exception:
        return False


def _require_gt():
    try:
        import great_tables as gt
    except ImportError as exc:  # pragma: no cover
        raise ImportError(
            "sp.gt(...) requires the 'great_tables' package. Install with:\n"
            "    pip install great_tables\n"
            "(or: pip install 'StatsPAI[publication]' once the extras "
            "ship in v1.7.2+)."
        ) from exc
    return gt


# ---------------------------------------------------------------------------
# Theme presets (journal -> gt option overrides)
# ---------------------------------------------------------------------------

# Per-journal styling that maps to ``great_tables``. Star semantics are
# inherited from the StatsPAI cell strings; here we only control fonts,
# alignment, and footnote-mark style (so headers / source notes look
# right for the target journal).
_JOURNAL_GT_STYLE = {
    # Common AER / AEJ-Applied / RestStat / RestUd / JPE convention:
    #   Times New Roman, all-caps section dividers, footnote marks
    #   are letters (a, b, c) so they don't clash with significance
    #   stars.
    "aer": {
        "font_family": "Times New Roman",
        "footnote_marks": "letters",
    },
    "aeja": {
        "font_family": "Times New Roman",
        "footnote_marks": "letters",
    },
    "restat": {
        "font_family": "Times New Roman",
        "footnote_marks": "letters",
    },
    "restud": {
        "font_family": "Times New Roman",
        "footnote_marks": "letters",
    },
    "jpe": {
        "font_family": "Times New Roman",
        "footnote_marks": "letters",
    },
    # QJE: identical font, but uses a "robust standard errors" SE
    # phrase that StatsPAI handles in the cell text — no gt change.
    "qje": {
        "font_family": "Times New Roman",
        "footnote_marks": "letters",
    },
    "econometrica": {
        "font_family": "Times New Roman",
        "footnote_marks": "letters",
    },
    "jf": {
        "font_family": "Times New Roman",
        "footnote_marks": "letters",
    },
}


def _apply_journal_theme(g, template: str):
    """Mutate ``g`` (a ``GT`` instance) with the journal's preset."""
    if not template:
        return g
    style = _JOURNAL_GT_STYLE.get(template.strip().lower())
    if not style:
        return g
    gt = _require_gt()
    # Footnote marks. ``opt_footnote_marks`` accepts predefined symbol
    # sequences ("standard", "letters", or a list of marks).
    if "footnote_marks" in style:
        try:
            g = g.opt_footnote_marks(marks=style["footnote_marks"])
        except Exception:
            # Older great_tables releases use ``opt_footnote_spec``
            # instead. We never want a styling miss to break the
            # adapter — the underlying table is still correct.
            pass
    # Font.
    if "font_family" in style:
        try:
            g = g.tab_options(table_font_names=style["font_family"])
        except Exception:
            pass
    return g


# ---------------------------------------------------------------------------
# Adapter — entry point dispatch
# ---------------------------------------------------------------------------

def to_gt(
    result: Any,
    *,
    title: Optional[str] = None,
    subtitle: Optional[str] = None,
    notes: Optional[Sequence[str]] = None,
    template: Optional[str] = None,
    rowname_col: Optional[str] = None,
    apply_theme: bool = True,
) -> "gt_pkg.GT":
    """Convert a StatsPAI table / DataFrame into a ``great_tables.GT``.

    Dispatches on the input type:

    - :class:`statspai.output.RegtableResult` — full-fidelity adapter
      that picks up the table's rendered cells, journal preset,
      title, and footer notes.
    - :class:`statspai.output.PaperTables` — flattens panels into a
      single GT with ``tab_row_group`` per panel.
    - :class:`pandas.DataFrame` — wraps verbatim. ``rowname_col``
      promotes a column to row labels.
    - Anything else with a ``to_dataframe()`` method — calls it.

    Parameters
    ----------
    result : object
        See dispatch description above.
    title : str, optional
        Override the table title. Default: pulled from
        ``result.title`` when present.
    subtitle : str, optional
        Subtitle line (``tab_header(subtitle=...)``).
    notes : sequence of str, optional
        Footer notes. When omitted, ``result.notes`` is used.
    template : str, optional
        Journal preset (``"aer"`` / ``"qje"`` / …). Default: pulled
        from ``result.template`` when present.
    rowname_col : str, optional
        Column to elevate to GT's row label position. For
        ``RegtableResult`` we default to the variable column
        automatically; for plain DataFrames, pass the column name.
    apply_theme : bool, default True
        Apply the journal-preset gt theme. Set False to keep gt's
        defaults (useful when the caller wants to compose their own
        ``tab_style`` chain).

    Returns
    -------
    great_tables.GT
        A fresh GT instance. Chain ``.tab_style(...)``, ``.opt_*``,
        ``.tab_spanner(...)``, etc. as you would in pure
        ``great_tables``.

    Examples
    --------
    >>> import statspai as sp
    >>> rt = sp.regtable(model, template="aer", title="Returns to Schooling")
    >>> g = sp.gt(rt)               # ready-to-render GT
    >>> g.as_raw_html()             # for HTML export / Quarto
    >>> g.as_latex()                # for LaTeX export

    >>> # Plain DataFrame path:
    >>> import pandas as pd
    >>> df = pd.DataFrame({"var": ["x", "y"], "M1": ["0.5***", "0.3"]})
    >>> sp.gt(df, rowname_col="var", title="Custom table")

    Raises
    ------
    ImportError
        If ``great_tables`` is not installed. Install via
        ``pip install great_tables``.
    TypeError
        If ``result`` is not adaptable to a DataFrame.
    """
    gt = _require_gt()

    # Late-bind imports so this module works in a partial install where
    # the regression_table module hasn't been touched yet.
    from .regression_table import RegtableResult, MeanComparisonResult
    from .paper_tables import PaperTables

    if isinstance(result, RegtableResult):
        return _from_regtable(
            result, gt=gt,
            title=title, subtitle=subtitle, notes=notes,
            template=template, apply_theme=apply_theme,
        )

    if isinstance(result, PaperTables):
        return _from_paper_tables(
            result, gt=gt,
            title=title, subtitle=subtitle, notes=notes,
            template=template, apply_theme=apply_theme,
        )

    if isinstance(result, MeanComparisonResult):
        return _from_dataframe(
            result.to_dataframe(), gt=gt,
            title=title or "Mean comparison",
            subtitle=subtitle, notes=notes,
            template=template, apply_theme=apply_theme,
            rowname_col=rowname_col,
        )

    if isinstance(result, pd.DataFrame):
        return _from_dataframe(
            result, gt=gt,
            title=title, subtitle=subtitle, notes=notes,
            template=template, apply_theme=apply_theme,
            rowname_col=rowname_col,
        )

    if hasattr(result, "to_dataframe") and callable(result.to_dataframe):
        return _from_dataframe(
            result.to_dataframe(), gt=gt,
            title=title or getattr(result, "title", None),
            subtitle=subtitle,
            notes=notes or getattr(result, "notes", None),
            template=template or getattr(result, "template", None),
            apply_theme=apply_theme,
            rowname_col=rowname_col,
        )

    raise TypeError(
        f"sp.gt() does not know how to convert {type(result).__name__}. "
        "Pass a RegtableResult, PaperTables, MeanComparisonResult, "
        "DataFrame, or any object with a .to_dataframe() method."
    )


# ---------------------------------------------------------------------------
# Per-source-type implementations
# ---------------------------------------------------------------------------

def _from_regtable(rt, *, gt, title, subtitle, notes, template,
                    apply_theme):
    df = rt.to_dataframe().reset_index().rename(columns={"index": ""})
    # The first column is the variable label (empty header in to_dataframe).
    # Rename to a placeholder so GT can target it as a rowname column.
    df.columns = [c if c != "" else " " for c in df.columns]

    g = gt.GT(df, rowname_col=" ")
    eff_title = title if title is not None else rt.title
    if eff_title:
        g = g.tab_header(title=eff_title, subtitle=subtitle)
    eff_notes = notes if notes is not None else rt.notes
    if eff_notes:
        for n in eff_notes:
            if n:
                g = g.tab_source_note(source_note=str(n))
    eff_template = template if template is not None else rt.template
    if apply_theme and eff_template:
        g = _apply_journal_theme(g, eff_template)
    return g


def _from_paper_tables(pt, *, gt, title, subtitle, notes, template,
                        apply_theme):
    """Flatten multi-panel PaperTables into a row-grouped GT."""
    panels = pt.panels()  # ordered dict-like
    if not panels:
        # Nothing to render — return an empty GT for predictability.
        return gt.GT(pd.DataFrame())

    # Build a long DataFrame with a "_panel" column that GT will
    # promote to row groups. PaperTables panels can have different
    # column sets — we union them, fill missing with empty strings.
    frames: List[pd.DataFrame] = []
    for panel_name, panel in panels.items():
        sub = panel.copy()
        sub.insert(0, "_panel", panel_name)
        frames.append(sub.reset_index().rename(
            columns={sub.index.name or "index": " "}
        ))
    long = pd.concat(frames, axis=0, ignore_index=True, sort=False)
    long = long.fillna("")

    g = gt.GT(long, rowname_col=" ", groupname_col="_panel")
    eff_title = title if title is not None else getattr(pt, "title", None)
    if eff_title:
        g = g.tab_header(title=eff_title, subtitle=subtitle)
    if notes:
        for n in notes:
            if n:
                g = g.tab_source_note(source_note=str(n))
    if apply_theme and template:
        g = _apply_journal_theme(g, template)
    return g


def _from_dataframe(df, *, gt, title, subtitle, notes, template,
                     apply_theme, rowname_col):
    if not isinstance(df, pd.DataFrame):
        raise TypeError(
            f"Expected DataFrame, got {type(df).__name__}"
        )
    g = (
        gt.GT(df.reset_index(drop=False) if rowname_col is None else df,
              rowname_col=rowname_col)
    )
    if title:
        g = g.tab_header(title=title, subtitle=subtitle)
    if notes:
        for n in notes:
            if n:
                g = g.tab_source_note(source_note=str(n))
    if apply_theme and template:
        g = _apply_journal_theme(g, template)
    return g
