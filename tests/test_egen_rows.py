"""Correctness + boundary tests for the Stata-style ``egen`` row helpers.

Covers ``sp.rowmean / rowtotal / rowmax / rowmin / rowsd / rowcount`` and
``sp.outlier_indicator`` — public functions that previously had no test
coverage (CLAUDE.md §5: every public function needs a correctness *and* a
boundary test). Expected values are analytic, so these double as a guard on
the Stata missing-value semantics documented in ``utils/egen.py``:

* ``rowmean / rowmax / rowmin / rowsd`` ignore NaN; a row that is entirely
  missing yields NaN.
* ``rowtotal`` treats NaN as 0 — an all-missing row sums to 0 (Stata
  ``egen rowtotal`` convention, not ``rowmean``'s).
* ``rowcount`` counts the non-missing cells (Stata ``rownonmiss``).
"""

import numpy as np
import pandas as pd
import pytest

import statspai as sp


@pytest.fixture
def frame():
    # Row 2 (index 2) is entirely missing across a/b/c — the boundary case
    # that separates rowtotal (->0) from the NaN-returning reducers.
    return pd.DataFrame(
        {
            "a": [1.0, 2.0, np.nan, np.nan],
            "b": [3.0, np.nan, np.nan, 5.0],
            "c": [5.0, 6.0, np.nan, 7.0],
        }
    )


# --------------------------------------------------------------------------
# Correctness — analytic expected values
# --------------------------------------------------------------------------
def test_rowmean_skips_nan(frame):
    out = sp.rowmean(frame, ["a", "b", "c"])
    expected = [3.0, 4.0, np.nan, 6.0]  # (1+3+5)/3, (2+6)/2, all-NaN, (5+7)/2
    np.testing.assert_allclose(out.to_numpy(), expected)


def test_rowtotal_treats_nan_as_zero(frame):
    out = sp.rowtotal(frame, ["a", "b", "c"])
    # All-missing row sums to 0 — this is the Stata rowtotal convention and
    # the key behavioural difference from rowmean.
    np.testing.assert_allclose(out.to_numpy(), [9.0, 8.0, 0.0, 12.0])


def test_rowmax_rowmin_skip_nan(frame):
    np.testing.assert_allclose(
        sp.rowmax(frame, ["a", "b", "c"]).to_numpy(), [5.0, 6.0, np.nan, 7.0]
    )
    np.testing.assert_allclose(
        sp.rowmin(frame, ["a", "b", "c"]).to_numpy(), [1.0, 2.0, np.nan, 5.0]
    )


def test_rowsd_uses_sample_ddof(frame):
    out = sp.rowsd(frame, ["a", "b", "c"])
    # ddof=1 sample sd. Row 0: sd([1,3,5]) = 2. Row 1: sd([2,6]) = 2*sqrt(2).
    # Row 2: all-NaN -> NaN. Row 3: sd([5,7]) = sqrt(2).
    expected = [2.0, 2.0 * np.sqrt(2), np.nan, np.sqrt(2)]
    np.testing.assert_allclose(out.to_numpy(), expected)


def test_rowsd_single_value_is_nan():
    # A single non-missing value has undefined sample sd (ddof=1) -> NaN,
    # matching Stata's rowsd on a singleton row.
    df = pd.DataFrame({"a": [7.0], "b": [np.nan]})
    assert np.isnan(sp.rowsd(df, ["a", "b"]).iloc[0])


def test_rowcount_counts_nonmissing(frame):
    out = sp.rowcount(frame, ["a", "b", "c"])
    assert out.tolist() == [3, 2, 0, 2]
    assert out.dtype == np.dtype("int64")


def test_row_helpers_preserve_index():
    df = pd.DataFrame({"a": [1.0, 2.0], "b": [3.0, 4.0]}, index=["r1", "r2"])
    out = sp.rowtotal(df, ["a", "b"])
    assert list(out.index) == ["r1", "r2"]


# --------------------------------------------------------------------------
# Boundary / error handling
# --------------------------------------------------------------------------
@pytest.mark.parametrize(
    "fn", [sp.rowmean, sp.rowtotal, sp.rowmax, sp.rowmin, sp.rowsd, sp.rowcount]
)
def test_missing_column_raises(fn, frame):
    with pytest.raises(ValueError, match="not found"):
        fn(frame, ["a", "does_not_exist"])


def test_single_column_reduces_to_itself(frame):
    # Degenerate 1-column reduction: mean/max/min/total all equal the column.
    for fn in (sp.rowmean, sp.rowmax, sp.rowmin, sp.rowtotal):
        out = fn(frame, ["c"])
        np.testing.assert_allclose(
            out.to_numpy(), frame["c"].fillna(0 if fn is sp.rowtotal else np.nan)
        )


# --------------------------------------------------------------------------
# outlier_indicator
# --------------------------------------------------------------------------
def test_outlier_indicator_flags_tails():
    df = pd.DataFrame({"x": list(range(1, 101))})  # 1..100
    out = sp.outlier_indicator(df, ["x"], cuts=(5, 95))
    # nanpercentile(5)=5.95, nanpercentile(95)=95.05 -> flag <5.95 and >95.05.
    flagged = df["x"][out["x_outlier"] == 1].tolist()
    assert flagged == [1, 2, 3, 4, 5, 96, 97, 98, 99, 100]
    assert set(out["x_outlier"].unique()) <= {0, 1}


def test_outlier_indicator_preserves_input_and_adds_column():
    df = pd.DataFrame({"x": [1.0, 2.0, 3.0], "y": [10.0, 20.0, 30.0]})
    out = sp.outlier_indicator(df, ["x"])
    # Original columns untouched; new indicator column added; input not mutated.
    pd.testing.assert_series_equal(out["x"], df["x"])
    assert "x_outlier" in out.columns and "x" in df.columns
    assert "x_outlier" not in df.columns  # no in-place mutation


def test_outlier_indicator_combined_any():
    df = pd.DataFrame({"x": [0.0, 50.0, 100.0], "y": [100.0, 50.0, 0.0]})
    out = sp.outlier_indicator(df, ["x", "y"], cuts=(25, 75), combined=True)
    assert "_outlier_any" in out.columns
    # _outlier_any must equal the OR of the per-variable flags.
    expected_any = ((out["x_outlier"] + out["y_outlier"]) > 0).astype(int)
    pd.testing.assert_series_equal(out["_outlier_any"], expected_any, check_names=False)


def test_outlier_indicator_ignores_nan_rows():
    df = pd.DataFrame({"x": [1.0, np.nan, 100.0]})
    out = sp.outlier_indicator(df, ["x"], cuts=(10, 90))
    # NaN rows are never flagged as outliers.
    assert out.loc[1, "x_outlier"] == 0
