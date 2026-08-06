"""One precision vocabulary across every exporter.

Before v1.22 each exporter took precision its own way — ``fmt="%.3f"`` on
``sumstats``, ``decimal_places=3`` on ``outreg2``, ``digits=3`` on
``fast.etable``, ``digits=4``/``6`` on the result-object exports — and none
of them understood the ``"auto"`` sentinel that ``sp.regtable`` had adopted.
Passing it produced a table full of the literal word ``auto`` rather than an
error, because ``"auto" % value`` on a template with no conversion specifier
returns the template unchanged.

These tests pin the unified contract: every exporter accepts the same
spellings, and none of them can silently emit a non-numeric cell.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.output._format import (
    AUTO,
    format_frame,
    format_pvalue,
    normalize_fmt,
    resolve_digits,
)


@pytest.fixture
def dollar_data():
    """Dollar-scale outcome, so false precision is visible."""
    rng = np.random.default_rng(20260806)
    n = 400
    df = pd.DataFrame(
        {
            "d": rng.integers(0, 2, n),
            "x": rng.normal(0, 1, n),
            "wage": rng.normal(50_000, 15_000, n),
        }
    )
    df["y"] = 15_000 * df["d"] + 2_000 * df["x"] + rng.normal(0, 9_000, n)
    return df


@pytest.fixture
def two_models(dollar_data):
    return (
        sp.regress("y ~ d", data=dollar_data),
        sp.regress("y ~ d + x", data=dollar_data),
    )


# ---------------------------------------------------------------------------
# 1. The vocabulary itself
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "spelling, expected",
    [
        (3, "%.3f"),  # R modelsummary / stargazer: digits = 3
        ("%.3f", "%.3f"),  # Stata esttab: b(%9.3f)
        ("r3", "%.3f"),  # R fixest: digits = "r3"
        ("s3", "sig:3"),  # R fixest: digits = "s3"
        ("auto", "auto"),  # StatsPAI: journal-adaptive pairing
    ],
)
def test_normalize_fmt_accepts_the_borrowed_spellings(spelling, expected):
    assert normalize_fmt(spelling) == expected


def test_resolve_digits_collapses_fmt_and_digits():
    assert resolve_digits(digits=3) == "%.3f"
    assert resolve_digits(fmt=3) == "%.3f"
    assert resolve_digits(fmt=None, digits=None, default=AUTO) == AUTO


def test_resolve_digits_rejects_both():
    with pytest.raises(sp.MethodIncompatibility, match="not both"):
        resolve_digits(fmt="%.2f", digits=2)


@pytest.mark.parametrize("bad", [2.5, object(), True])
def test_normalize_fmt_rejects_unusable_input(bad):
    """Invalid precision fails at the call site, never mid-render."""
    with pytest.raises(sp.MethodIncompatibility):
        normalize_fmt(bad)


# ---------------------------------------------------------------------------
# 2. The silent-garbage regression
# ---------------------------------------------------------------------------


def test_sumstats_auto_does_not_emit_the_literal_word(dollar_data):
    """Regression: ``fmt="auto"`` used to fill every cell with ``"auto"``.

    ``"auto" % value`` returns ``"auto"`` unchanged — no exception — so the
    table rendered as garbage that looked deliberate.
    """
    out = str(sp.sumstats(dollar_data, vars=["wage"], fmt="auto"))
    assert "auto" not in out
    assert any(ch.isdigit() for ch in out)


def test_mean_comparison_auto_does_not_emit_the_literal_word(dollar_data):
    out = str(
        sp.mean_comparison(dollar_data, variables=["wage"], group="d", fmt="auto")
    )
    assert "auto" not in out


def test_sumstats_default_avoids_false_precision(dollar_data):
    """A $50k mean must not print three decimals of noise.

    Pre-1.22 the fixed ``"%.3f"`` default rendered the mean as
    ``50228.947`` — three digits past the decimal on a dollar amount whose
    standard deviation is five figures.
    """
    row = next(
        ln
        for ln in str(sp.sumstats(dollar_data, vars=["wage"])).split("\n")
        if "wage" in ln
    )
    cells = [c for c in row.split() if any(ch.isdigit() for ch in c)]
    mean_cell = cells[1]  # after N
    assert _decimals(mean_cell) == 0, f"false precision on a $50k mean: {mean_cell!r}"
    assert "," in mean_cell, f"thousands not grouped: {mean_cell!r}"


def test_sumstats_digits_alias(dollar_data):
    a = str(sp.sumstats(dollar_data, vars=["wage"], digits=1))
    b = str(sp.sumstats(dollar_data, vars=["wage"], fmt="%.1f"))
    assert a == b


def test_sumstats_rejects_fmt_and_digits_together(dollar_data):
    with pytest.raises(sp.MethodIncompatibility, match="not both"):
        sp.sumstats(dollar_data, vars=["wage"], fmt="%.2f", digits=2)


# ---------------------------------------------------------------------------
# 3. sp.etable's fallback used to drop standard errors entirely
# ---------------------------------------------------------------------------


def test_etable_fallback_reports_standard_errors(two_models):
    """The non-pyfixest path returned bare coefficients — half a table.

    Without an SE the reader cannot judge any estimate, so the fallback now
    renders ``coef*** (se)`` like every other exporter.
    """
    txt = sp.etable(*two_models).to_string()
    assert "(" in txt and ")" in txt, f"no standard errors in fallback table:\n{txt}"


def test_etable_fallback_keeps_every_reported_term(dollar_data):
    """Auxiliary terms must survive — Tobit's ``sigma`` is a real parameter."""
    df = dollar_data.copy()
    df["ycens"] = np.maximum(
        0.0, df["x"] + np.random.default_rng(0).normal(0, 1, len(df))
    )
    r = sp.tobit(df, y="ycens", x=["x"], ll=0)
    txt = sp.etable(r).to_string()
    for term in ("const", "x", "sigma"):
        assert term in txt, f"{term!r} missing from etable fallback:\n{txt}"


def test_etable_fallback_pairs_coefficient_and_se(two_models):
    """Estimate and its own SE share a decimal place, as in sp.regtable."""
    txt = sp.etable(*two_models).to_string()
    row = next(ln for ln in txt.split("\n") if ln.strip().startswith("d"))
    cell = row.split(None, 1)[1]
    est, se = cell.split("(")[0].strip().rstrip("*"), cell.split("(")[1].rstrip(") ")
    assert _decimals(est) == _decimals(se), f"precision disagrees in {cell!r}"


def _decimals(text: str) -> int:
    text = text.replace(",", "").strip()
    return len(text.split(".")[1]) if "." in text else 0


# ---------------------------------------------------------------------------
# 4. format_frame: tidy/glance frames get the same pairing
# ---------------------------------------------------------------------------


def test_format_frame_pairs_estimate_and_se():
    df = pd.DataFrame(
        {
            "term": ["ATE"],
            "estimate": [13386.630926],
            "std_error": [843.643031],
            "conf_low": [11733.120969],
            "conf_high": [15040.140883],
        }
    )
    out = format_frame(df)
    decs = {_decimals(out[c].iloc[0]) for c in ("estimate", "std_error", "conf_low")}
    assert len(decs) == 1, f"row precision disagrees: {out.to_dict('records')}"
    assert "," in out["estimate"].iloc[0]  # thousands separated


def test_format_frame_counts_stay_integers():
    out = format_frame(pd.DataFrame({"nobs": [4000], "estimate": [1.5]}))
    assert out["nobs"].iloc[0] == "4,000"


@pytest.mark.parametrize(
    "p, expected",
    [(0.0, "<0.001"), (1e-9, "<0.001"), (0.0432, "0.043"), (0.5, "0.500")],
)
def test_format_pvalue_floors_instead_of_claiming_zero(p, expected):
    """``p = 0.000`` claims certainty no finite sample supports."""
    assert format_pvalue(p) == expected


def test_result_export_pairs_and_floors(dollar_data):
    r = sp.aipw(dollar_data, y="y", treat="d", covariates=["x"])
    md = r.to_markdown()
    assert "<0.001" in md or "0." in md
    row = next(ln for ln in md.split("\n") if "ATE" in ln)
    cells = [c.strip() for c in row.split("|") if c.strip()]
    nums = [c for c in cells[1:4] if any(ch.isdigit() for ch in c)]
    assert len({_decimals(c) for c in nums[:2]}) == 1, f"unpaired row: {row}"


def test_result_export_accepts_the_vocabulary(dollar_data):
    r = sp.aipw(dollar_data, y="y", treat="d", covariates=["x"])
    assert r.to_markdown(digits=2) == r.to_markdown(fmt="%.2f")
    with pytest.raises(sp.MethodIncompatibility, match="not both"):
        r.to_markdown(fmt="%.2f", digits=2)


def test_to_excel_stays_numeric(dollar_data, tmp_path):
    """Excel is data interchange: cells stay numbers, not display strings."""
    pytest.importorskip("openpyxl")
    r = sp.aipw(dollar_data, y="y", treat="d", covariates=["x"])
    path = tmp_path / "r.xlsx"
    r.to_excel(str(path))
    got = pd.read_excel(path)
    est = [c for c in got.columns if str(c).lower() == "estimate"]
    assert est, f"no estimate column in {list(got.columns)}"
    assert pd.api.types.is_numeric_dtype(got[est[0]])
