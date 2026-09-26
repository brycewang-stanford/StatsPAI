"""LaTeX exports must be escaped well enough to actually compile.

Two layers:

* :class:`TestEscapeUnit` / :class:`TestExportedSourceIsEscaped` are pure
  Python and run everywhere. They catch the class of bug that shipped
  before: a detail column called ``propensity_score`` reaching the
  ``tabular`` body as a bare ``_``, which pdflatex rejects with
  "Missing $ inserted".
* :class:`TestRealPdflatexCompile` shells out to a real TeX engine. It is
  the ground truth, and skips when no engine is installed (CI images
  usually have none).
"""

import re
import shutil
import subprocess
import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai._result_serialize import ResultProtocolMixin
from statspai.output._format import latex_escape

#: ``CausalResult.to_latex`` needs only ``threeparttable`` (it uses plain
#: ``\hline`` rules); the generic ``ResultProtocolMixin.to_latex`` emits
#: booktabs rules. Loading both keeps one preamble for both paths.
PREAMBLE = r"""\documentclass{article}
\usepackage{threeparttable}
\usepackage{booktabs}
\usepackage[T1]{fontenc}
\begin{document}
%s
\end{document}
"""

#: Characters that must never survive into a rendered data cell unescaped.
#: ``$`` is omitted only because the assertion runs per-cell after splitting,
#: and a lone ``$`` in data is already covered by the unit tests above.
FORBIDDEN_BARE = ("_", "&", "%", "#", "<", ">")


@pytest.fixture(scope="module")
def matched():
    """A PSM result whose detail table carries LaTeX-hostile column names."""
    rng = np.random.default_rng(7)
    n = 600
    x = rng.normal(size=n)
    d = (rng.random(n) < 1 / (1 + np.exp(-x))).astype(int)
    df = pd.DataFrame({"y": 0.2876 * d + 0.5 * x + rng.normal(size=n), "d": d, "x": x})
    # Deliberate landmines: underscore, ampersand, percent.
    df["pct_gain_&_loss"] = rng.normal(size=n)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.psm(df, y="y", d="d", X=["x", "pct_gain_&_loss"])


class TestEscapeUnit:
    def test_backslash_is_not_double_escaped(self):
        # The bug that motivated a single-pass regex: replacing "\" first
        # emits "{" and "}" that a later pass would escape again.
        assert latex_escape("a\\b") == "a\\textbackslash{}b"

    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("mean_treated", "mean\\_treated"),
            ("gain & loss", "gain \\& loss"),
            ("50%", "50\\%"),
            ("#1", "\\#1"),
            ("<0.001", "\\textless{}0.001"),
            (">5", "\\textgreater{}5"),
        ],
    )
    def test_specials_escaped(self, raw, expected):
        assert latex_escape(raw) == expected

    def test_missing_renders_empty(self):
        assert latex_escape(None) == ""
        assert latex_escape(np.nan) == ""

    def test_escaping_is_total_and_has_no_passthrough_mode(self):
        # Escaping a data cell is unconditional: a caller who wants markup
        # preserved must not route it through here at all. Guards against
        # reintroducing a "keep math" flag, which cannot be correct — the
        # same "_" is data in a column name and syntax inside $\beta_1$.
        assert latex_escape("$\\beta$") == "\\$\\textbackslash{}beta\\$"


class TestExportedSourceIsEscaped:
    def test_tabular_cells_have_no_bare_specials(self, matched):
        latex = matched.to_latex()
        cells = _tabular_cells(latex)
        assert cells, "expected a non-empty tabular body"
        for cell in cells:
            for ch in FORBIDDEN_BARE:
                assert not _has_bare(cell, ch), f"unescaped {ch!r} in cell {cell!r}"

    def test_notes_have_no_bare_less_than(self, matched):
        # "* p<0.1" typesets as an inverted exclamation mark under OT1.
        latex = matched.to_latex()
        notes = [ln for ln in latex.splitlines() if ln.startswith("\\item")]
        assert notes
        for line in notes:
            assert not _has_bare(line, "<")

    def test_user_caption_passes_through_verbatim(self, matched):
        # A caller who writes LaTeX means it.
        latex = matched.to_latex(caption="Effect of $D$ on $Y$")
        assert "\\caption{Effect of $D$ on $Y$}" in latex


def _tabular_cells(latex):
    """Every individual cell inside the tabular, separators stripped.

    Splitting on the column separator matters: a bare ``&`` between cells
    is correct LaTeX, so scanning raw lines would flag every table.
    """
    lines = latex.splitlines()
    try:
        start = next(
            i for i, ln in enumerate(lines) if ln.startswith("\\begin{tabular}")
        )
        end = next(i for i, ln in enumerate(lines) if ln.startswith("\\end{tabular}"))
    except StopIteration:  # pragma: no cover - guards a malformed export
        return []

    cells = []
    for line in lines[start + 1 : end]:
        if line.startswith("\\hline") or not line.strip():
            continue
        row = line.removesuffix("\\\\").strip()
        # Split on separators only — an escaped "\&" is data, not a break.
        for cell in re.split(r"(?<!\\)&", row):
            cells.append(cell.strip())
    return cells


def _has_bare(text, char):
    """True if *char* appears unescaped (not preceded by a backslash)."""
    return any(
        ch == char and (i == 0 or text[i - 1] != "\\") for i, ch in enumerate(text)
    )


@pytest.fixture(scope="module")
def generic_result():
    """A result exported by the generic Field/Value renderer.

    ``ResultProtocolMixin.to_latex`` is a second, independent LaTeX path
    from ``CausalResult.to_latex``. It escaped only underscores, and only
    in the key column, leaving every other special and the whole value
    column raw; and it dumped multi-line fields (some hold an entire
    rendered ``summary()``) into a single cell.
    """
    rng = np.random.default_rng(1)
    n = 400
    x = rng.normal(size=n)
    d = (rng.random(n) < 1 / (1 + np.exp(-x))).astype(int)
    df = pd.DataFrame(
        {"y": 0.3 * d + 0.5 * x + rng.normal(size=n), "treat": d, "x_1": x}
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.psmatch2(df, y="y", treat="treat", covariates=["x_1"])


@dataclass
class _BoolResult(ResultProtocolMixin):
    """Minimal result exercising the branches the generic table got wrong."""

    converged: bool = True
    robust: bool = False
    n: int = 5
    est: float = 538582.4


class TestGenericResultBooleans:
    def test_bool_is_not_wrapped_in_amsmath_text(self):
        # "\text{True}" is an amsmath macro. Any result with a bool field
        # produced a table that died with "Undefined control sequence"
        # unless the surrounding document happened to load amsmath.
        latex = _BoolResult().to_latex()
        assert r"\text{" not in latex
        assert "True" in latex and "False" in latex

    def test_bool_does_not_degrade_to_int(self):
        # bool is an int subclass; an int-first branch renders True as 1.
        assert "| converged | True |" in _BoolResult().to_markdown()

    def test_generated_caption_escapes_the_class_name(self):
        # The fallback caption is the class name, which is data — a class
        # called "_BoolResult" or "PSM_DiDResult" carries an underscore.
        assert "\\caption{\\_BoolResult}" in _BoolResult().to_latex()

    def test_supplied_caption_still_passes_through(self):
        latex = _BoolResult().to_latex(caption="Effect of $D$ on $Y$")
        assert "\\caption{Effect of $D$ on $Y$}" in latex

    @pytest.mark.skipif(
        shutil.which("pdflatex") is None, reason="no TeX engine installed"
    )
    def test_compiles_without_amsmath(self, tmp_path):
        _compile(_BoolResult().to_latex(), tmp_path)


class TestGenericResultTable:
    def test_no_cell_spans_multiple_lines(self, generic_result):
        latex = generic_result.to_latex()
        for cell in _tabular_cells(latex):
            assert "\n" not in cell

    def test_cells_are_escaped(self, generic_result):
        for cell in _tabular_cells(generic_result.to_latex()):
            for ch in FORBIDDEN_BARE:
                assert not _has_bare(cell, ch), f"unescaped {ch!r} in {cell!r}"


def _compile(latex, tmp_path):
    src = tmp_path / "t.tex"
    src.write_text(PREAMBLE % latex, encoding="utf-8")
    proc = subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "t.tex"],
        cwd=tmp_path,
        capture_output=True,
        text=True,
    )
    if proc.returncode != 0:
        errs = [ln for ln in proc.stdout.splitlines() if ln.startswith("!")]
        pytest.fail("pdflatex rejected the export:\n" + "\n".join(errs[:5]))


@pytest.mark.skipif(shutil.which("pdflatex") is None, reason="no TeX engine installed")
class TestRealPdflatexCompile:
    @pytest.mark.parametrize("kwargs", [{}, {"digits": 3}, {"fmt": "auto"}])
    def test_causal_result_compiles(self, matched, kwargs, tmp_path):
        _compile(matched.to_latex(**kwargs), tmp_path)

    def test_generic_result_table_compiles(self, generic_result, tmp_path):
        _compile(generic_result.to_latex(), tmp_path)
