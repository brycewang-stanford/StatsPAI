"""
Cross-tabulation with statistical tests.

Equivalent to Stata's ``tab var1 var2, chi2 exact``.
Exports to text, LaTeX, Excel, Word.
"""

from typing import Optional, List, Dict, Any, Union
import numpy as np
import pandas as pd
from scipy import stats


def tab(
    data: pd.DataFrame,
    row: str,
    col: Optional[str] = None,
    output: str = 'text',
    test: bool = True,
    margins: bool = True,
    normalize: Optional[str] = None,
    title: Optional[str] = None,
) -> Union[str, pd.DataFrame]:
    """
    Cross-tabulation with chi-squared / Fisher's exact test.

    Parameters
    ----------
    data : pd.DataFrame
        Input data.
    row : str
        Row variable.
    col : str, optional
        Column variable. If None, produces a one-way frequency table.
    output : str, default 'text'
        'text', 'dataframe', 'latex', or filepath (.xlsx/.docx).
    test : bool, default True
        Include chi-squared test (and Fisher's exact for 2x2).
    margins : bool, default True
        Show row/column totals.
    normalize : str, optional
        'row', 'col', 'all', or None. Normalize to proportions.
    title : str, optional
        Table title.

    Returns
    -------
    str or pd.DataFrame

    Examples
    --------
    >>> sp.tab(df, 'treatment', 'outcome')
    >>> sp.tab(df, 'treatment', 'outcome', output='crosstab.docx')
    >>> sp.tab(df, 'treatment')  # one-way frequency
    """
    if col is None:
        return _one_way_tab(data, row, output, title)

    # Two-way cross-tabulation
    ct = pd.crosstab(
        data[row], data[col],
        margins=margins,
        margins_name='Total',
    )

    if normalize:
        norm_map = {'row': 'index', 'col': 'columns', 'all': 'all'}
        ct_norm = pd.crosstab(
            data[row], data[col],
            normalize=norm_map.get(normalize, normalize),
            margins=margins,
            margins_name='Total',
        )
        # Format as percentages
        ct_display = ct_norm.map(lambda x: f'{x:.1%}')
    else:
        ct_display = ct

    # Statistical test
    test_result = None
    if test:
        # Use raw counts (no margins)
        ct_raw = pd.crosstab(data[row], data[col])
        chi2, p_chi2, dof, expected = stats.chi2_contingency(ct_raw)
        test_result = {
            'chi2': chi2,
            'pvalue': p_chi2,
            'df': dof,
        }
        # Fisher's exact for 2x2
        if ct_raw.shape == (2, 2):
            _, p_fisher = stats.fisher_exact(ct_raw)
            test_result['fisher_pvalue'] = p_fisher

    if title is None:
        title = f'Tabulation: {row} × {col}'

    return _format_tab(ct_display, test_result, output, title)


def _one_way_tab(data, var, output, title):
    """One-way frequency table."""
    counts = data[var].value_counts().sort_index()
    pcts = data[var].value_counts(normalize=True).sort_index()

    df = pd.DataFrame({
        'Freq': counts,
        'Percent': pcts.map(lambda x: f'{x:.1%}'),
        'Cum.': pcts.cumsum().map(lambda x: f'{x:.1%}'),
    })
    df.loc['Total'] = [int(counts.sum()), '100.0%', '100.0%']

    if title is None:
        title = f'Tabulation: {var}'

    return _format_tab(df, None, output, title)


def _format_tab(df, test_result, output, title):
    """Route to output format."""
    if output == 'dataframe':
        return df

    if output.endswith('.xlsx'):
        _tab_to_excel(df, test_result, output, title)
        return f"Exported to: {output}"
    elif output.endswith('.docx'):
        _tab_to_word(df, test_result, output, title)
        return f"Exported to: {output}"
    elif output == 'latex':
        return _tab_to_latex(df, test_result, title)

    # Default: text
    lines = []
    if title:
        lines.append(title)
    lines.append('=' * 60)
    lines.append(df.to_string())
    lines.append('=' * 60)

    if test_result:
        lines.append(f"  Pearson chi2({test_result['df']}) = "
                      f"{test_result['chi2']:.4f}   Pr = {test_result['pvalue']:.4f}")
        if 'fisher_pvalue' in test_result:
            lines.append(f"  Fisher's exact                     "
                          f"Pr = {test_result['fisher_pvalue']:.4f}")

    return '\n'.join(lines)


def _tab_to_latex(df, test_result, title):
    latex = df.to_latex(caption=title)
    if test_result:
        latex += f"\n% chi2({test_result['df']}) = {test_result['chi2']:.4f}, p = {test_result['pvalue']:.4f}"
    return latex


def _tab_to_excel(df, test_result, filename, title):
    import openpyxl
    from openpyxl.styles import Font, Alignment

    wb = openpyxl.Workbook()
    ws = wb.active

    row = 1
    if title:
        ws.cell(row=row, column=1, value=title).font = Font(bold=True, size=12)
        row += 2

    # Header
    ws.cell(row=row, column=1, value='').font = Font(bold=True)
    for j, col in enumerate(df.columns, 2):
        ws.cell(row=row, column=j, value=str(col)).font = Font(bold=True)
        ws.cell(row=row, column=j).alignment = Alignment(horizontal='center')
    row += 1

    # Data
    for idx, data_row in df.iterrows():
        ws.cell(row=row, column=1, value=str(idx))
        for j, val in enumerate(data_row, 2):
            ws.cell(row=row, column=j, value=str(val)).alignment = Alignment(horizontal='center')
        row += 1

    if test_result:
        row += 1
        ws.cell(row=row, column=1,
                value=f"chi2({test_result['df']}) = {test_result['chi2']:.4f}, "
                      f"p = {test_result['pvalue']:.4f}").font = Font(italic=True, size=9)

    for c in range(1, len(df.columns) + 2):
        ws.column_dimensions[openpyxl.utils.get_column_letter(c)].width = 12
    wb.save(filename)


def _tab_to_word(df, test_result, filename, title):
    try:
        from docx import Document
        from docx.shared import Pt
        from docx.enum.text import WD_ALIGN_PARAGRAPH
        from docx.enum.table import WD_TABLE_ALIGNMENT
    except ImportError:
        raise ImportError("python-docx required. Install: pip install python-docx")

    doc = Document()

    if title:
        p = doc.add_paragraph()
        run = p.add_run(title)
        run.bold = True
        run.font.size = Pt(12)
        p.alignment = WD_ALIGN_PARAGRAPH.CENTER

    n_rows = len(df) + 1
    n_cols = len(df.columns) + 1
    table = doc.add_table(rows=n_rows, cols=n_cols)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER

    # Header
    table.rows[0].cells[0].text = ''
    for j, col in enumerate(df.columns, 1):
        table.rows[0].cells[j].text = str(col)
        for para in table.rows[0].cells[j].paragraphs:
            para.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for run in para.runs:
                run.bold = True
                run.font.size = Pt(9)

    # Data
    for i, (idx, row) in enumerate(df.iterrows()):
        table.rows[i + 1].cells[0].text = str(idx)
        for para in table.rows[i + 1].cells[0].paragraphs:
            for run in para.runs:
                run.font.size = Pt(9)
        for j, val in enumerate(row, 1):
            table.rows[i + 1].cells[j].text = str(val)
            for para in table.rows[i + 1].cells[j].paragraphs:
                para.alignment = WD_ALIGN_PARAGRAPH.CENTER
                for run in para.runs:
                    run.font.size = Pt(9)

    if test_result:
        p = doc.add_paragraph()
        run = p.add_run(
            f"Pearson chi2({test_result['df']}) = {test_result['chi2']:.4f}, "
            f"p = {test_result['pvalue']:.4f}")
        run.italic = True
        run.font.size = Pt(8)

    doc.save(filename)
