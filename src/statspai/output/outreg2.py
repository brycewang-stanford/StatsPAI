"""Stata ``outreg2`` compatibility surface — thin facade over :func:`regtable`.

Historically this module shipped its own bespoke renderer pipeline
(``_create_regression_table``, ``_export_with_formatting``,
``_apply_apa_borders``, …) totalling ~800 lines of code that
re-implemented coefficient extraction, star formatting, three-line
table styling and Excel / Word / LaTeX export — all features that
:func:`statspai.output.regtable` already provided in publication-quality
form, with bug fixes the bespoke renderer never received (broken
``F-statistic / Trees`` row on OLS, junk ``& None & None`` LaTeX cell,
missing star legend).

In the PR-B output consolidation (see
``docs/rfc/output_pr_b_consolidation.md``) we collapse this module to
~150 lines that translate Stata-flavoured kwargs and forward to
:func:`regtable`. The user-visible API (``OutReg2`` class,
``outreg2`` function) is preserved; the *rendered output* now matches
:func:`regtable` exactly — strictly better in every case but
NOT byte-identical to the legacy output. A :class:`DeprecationWarning`
points users to ``sp.regtable(...).to_excel(...)`` for full control.

Migration notes
---------------
The following labels changed (legacy → new, all canonical to regtable):

- ``"Variables"`` header column → blank (book-tab convention)
- ``"R-squared"`` → ``"R²"``
- ``"Adj. R-squared"`` → ``"Adj. R²"``
- ``"Observations"`` → ``"N"``
- ``"F-statistic / Trees"`` → ``"F"`` (bug fix: ``"/ Trees"`` only
  applies to causal-forest results, was always wrong on OLS)
- LaTeX gains a proper star legend below the table
- ``show_se=False`` is no longer supported (regression tables without
  uncertainty are pseudo-science). Pass ``se_type="t"`` etc. via
  :func:`regtable` directly if you genuinely need an alternative cell.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional

from ..core.results import EconometricResults

_DEPRECATION_MSG_OUTREG2 = (
    "outreg2() is now a thin wrapper over sp.regtable() and will "
    "be removed in a future minor release. Migrate to "
    "sp.regtable(*models, ...) for the same output with full control "
    "over labels, journal templates, and SE formats. "
    "See docs/rfc/output_pr_b_consolidation.md for migration."
)


def _warn_outreg2_deprecation() -> None:
    warnings.warn(_DEPRECATION_MSG_OUTREG2, DeprecationWarning, stacklevel=3)


def _build_regtable(
    *,
    models: List[Any],
    model_names: Optional[List[str]],
    title: Optional[str],
    notes: Optional[List[str]],
    show_stars: bool,
    show_se: bool,
    show_tstat: bool,
    decimal_places: int,
    variable_labels: Optional[Dict[str, str]],
):
    """Translate Stata-flavoured outreg2 kwargs into a ``regtable`` call."""
    from .regression_table import regtable

    if not show_se and not show_tstat:
        warnings.warn(
            "outreg2(show_se=False) is no longer supported; the "
            "regression table will keep the standard-error row. Use "
            "sp.regtable(...) directly with se_type='t' or se_type='ci' "
            "if you need a different uncertainty cell.",
            UserWarning,
            stacklevel=3,
        )
        se_type = "se"
    elif show_tstat:
        se_type = "t"
    else:
        se_type = "se"

    labels = (
        list(model_names)
        if model_names
        else [f"Model {i + 1}" for i in range(len(models))]
    )

    return regtable(
        list(models),
        model_labels=labels,
        title=title,
        notes=notes,
        stars=show_stars,
        se_type=se_type,
        fmt=f"%.{int(decimal_places)}f",
        coef_labels=variable_labels,
    )


class OutReg2:
    """Stata-style stateful regression-table builder.

    .. deprecated::
        Use :func:`statspai.output.regtable` directly. This class now
        accumulates models and forwards to :func:`regtable` at export
        time; the rendered output matches ``regtable``'s book-tab
        style (and is no longer byte-identical to the legacy bespoke
        renderer).
    """

    def __init__(self):
        _warn_outreg2_deprecation()
        self.results: List[Any] = []
        self.model_names: List[Optional[str]] = []
        self.title: Optional[str] = "Regression Results"
        self.notes: List[str] = []

    def add_model(
        self,
        results: EconometricResults,
        name: Optional[str] = None,
    ) -> None:
        """Accumulate a model for later export."""
        self.results.append(results)
        self.model_names.append(name)

    def clear(self) -> None:
        self.results.clear()
        self.model_names.clear()
        self.notes.clear()

    def set_title(self, title: str) -> None:
        self.title = title

    def add_note(self, note: str) -> None:
        self.notes.append(note)

    # ─── exporters (forward to regtable) ───────────────────────────────

    def _resolved_model_names(self) -> List[str]:
        return [
            name if name else f"Model {i + 1}"
            for i, name in enumerate(self.model_names)
        ]

    def _table(
        self,
        *,
        show_stars: bool,
        show_se: bool,
        show_tstat: bool,
        decimal_places: int,
        variable_labels: Optional[Dict[str, str]],
    ):
        return _build_regtable(
            models=self.results,
            model_names=self._resolved_model_names(),
            title=self.title,
            notes=self.notes or None,
            show_stars=show_stars,
            show_se=show_se,
            show_tstat=show_tstat,
            decimal_places=decimal_places,
            variable_labels=variable_labels,
        )

    def to_excel(
        self,
        filename: str,
        show_stars: bool = True,
        show_se: bool = True,
        show_tstat: bool = False,
        decimal_places: int = 3,
        variable_labels: Optional[Dict[str, str]] = None,
        sheet_name: str = "Regression",  # accepted but unused by regtable
    ) -> None:
        del sheet_name  # legacy parameter, regtable picks its own sheet
        table = self._table(
            show_stars=show_stars,
            show_se=show_se,
            show_tstat=show_tstat,
            decimal_places=decimal_places,
            variable_labels=variable_labels,
        )
        table.to_excel(filename)
        print(f"Regression results exported to: {filename}")

    def to_word(
        self,
        filename: str,
        show_stars: bool = True,
        show_se: bool = True,
        show_tstat: bool = False,
        decimal_places: int = 3,
        variable_labels: Optional[Dict[str, str]] = None,
    ) -> None:
        table = self._table(
            show_stars=show_stars,
            show_se=show_se,
            show_tstat=show_tstat,
            decimal_places=decimal_places,
            variable_labels=variable_labels,
        )
        table.to_word(filename)
        print(f"Regression results exported to: {filename}")

    def to_latex(
        self,
        filename: Optional[str] = None,
        show_stars: bool = True,
        show_se: bool = True,
        show_tstat: bool = False,
        decimal_places: int = 3,
        variable_labels: Optional[Dict[str, str]] = None,
    ) -> str:
        table = self._table(
            show_stars=show_stars,
            show_se=show_se,
            show_tstat=show_tstat,
            decimal_places=decimal_places,
            variable_labels=variable_labels,
        )
        latex = table.to_latex()
        if filename:
            from pathlib import Path
            Path(filename).write_text(latex, encoding="utf-8")
            print(f"Regression results exported to: {filename}")
        return latex


def outreg2(
    *results: EconometricResults,
    filename: str,
    model_names: Optional[List[str]] = None,
    title: str = "Regression Results",
    notes: Optional[List[str]] = None,
    show_stars: bool = True,
    show_se: bool = True,
    show_tstat: bool = False,
    decimal_places: int = 3,
    variable_labels: Optional[Dict[str, str]] = None,
    format: str = "auto",
) -> Optional[str]:
    """
    Convenient export of multiple regression results (Stata-style).

    .. deprecated::
        Now a thin wrapper over :func:`statspai.output.regtable`. The
        rendered output matches ``regtable``'s book-tab style and is
        not byte-identical to the legacy bespoke renderer (label changes
        documented in the module docstring). Migrate to::

            sp.regtable(*models, model_labels=..., title=...).to_excel(filename)

    Parameters
    ----------
    *results : EconometricResults
        Multiple regression results to include.
    filename : str
        Output filename. The format is auto-detected from the
        extension: ``.xlsx`` → Excel, ``.docx`` → Word, ``.tex`` →
        LaTeX. Override with ``format=``.
    model_names : list of str, optional
        Labels for each model.
    title : str, default "Regression Results"
        Table title.
    notes : list of str, optional
        Notes appended below the table.
    show_stars : bool, default True
        Whether to show significance stars.
    show_se : bool, default True
        Whether to keep the standard-errors row. ``False`` is no
        longer supported and will emit a ``UserWarning``.
    show_tstat : bool, default False
        If ``True``, the parenthesised cell shows the *t*-statistic
        instead of the standard error.
    decimal_places : int, default 3
        Number of decimal places (mapped to ``fmt='%.<N>f'``).
    variable_labels : dict, optional
        Custom variable labels (mapped to ``regtable.coef_labels``).
    format : str, default "auto"
        ``"auto"`` (default) detects from filename extension:
        ``.xlsx`` → Excel, ``.docx`` → Word, ``.tex`` → LaTeX.
        Override with ``"excel"``, ``"word"``, or ``"latex"``.

    Returns
    -------
    str or None
        LaTeX code if ``format="latex"``, otherwise ``None``.
    """
    _warn_outreg2_deprecation()

    fmt = format.lower()
    if fmt == "auto":
        lower = filename.lower()
        if lower.endswith((".docx", ".doc")):
            fmt = "word"
        elif lower.endswith(".tex"):
            fmt = "latex"
        else:
            fmt = "excel"

    table = _build_regtable(
        models=list(results),
        model_names=model_names,
        title=title,
        notes=notes,
        show_stars=show_stars,
        show_se=show_se,
        show_tstat=show_tstat,
        decimal_places=decimal_places,
        variable_labels=variable_labels,
    )

    if fmt == "latex":
        out_path = filename if filename.endswith(".tex") else filename + ".tex"
        latex = table.to_latex()
        from pathlib import Path
        Path(out_path).write_text(latex, encoding="utf-8")
        print(f"Regression results exported to: {out_path}")
        return latex
    if fmt == "word":
        table.to_word(filename)
        print(f"Regression results exported to: {filename}")
        return None
    # default: excel
    table.to_excel(filename)
    print(f"Regression results exported to: {filename}")
    return None
