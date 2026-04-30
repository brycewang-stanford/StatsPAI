"""Tests for ``statspai.output._format`` — canonical numeric formatters."""

from __future__ import annotations

import math
import numpy as np
import pandas as pd
import pytest

from statspai.output._format import (
    format_stars,
    fmt_auto,
    fmt_int,
    fmt_val,
    is_missing,
)


# ---------------------------------------------------------------------------
# is_missing
# ---------------------------------------------------------------------------

class TestIsMissing:
    @pytest.mark.parametrize("value", [None, float("nan"), np.nan, pd.NA, pd.NaT])
    def test_missing_scalars(self, value):
        assert is_missing(value) is True

    @pytest.mark.parametrize("value", [0, 1, -1, 0.0, 1.0, "", "text"])
    def test_non_missing(self, value):
        assert is_missing(value) is False

    def test_pd_na_scalar(self):
        # pd.NA is technically not float('nan') but is NA
        assert is_missing(pd.NA) is True

    def test_nan_in_container_still_detected(self):
        # is_missing only checks the scalar itself
        arr = np.array([np.nan])
        assert is_missing(arr[0]) is True


# ---------------------------------------------------------------------------
# format_stars
# ---------------------------------------------------------------------------

class TestFormatStars:
    def test_no_stars(self):
        assert format_stars(0.15) == ""

    def test_one_star(self):
        assert format_stars(0.08) == "*"

    def test_two_stars(self):
        assert format_stars(0.03) == "**"

    def test_three_stars(self):
        assert format_stars(0.001) == "***"

    def test_exact_thresholds(self):
        # at the threshold: pvalue < level is False → no star
        assert format_stars(0.10) == ""   # not < 0.10
        assert format_stars(0.05) == "*"   # < 0.10 → one star
        assert format_stars(0.01) == "**"  # < 0.10 and < 0.05
        # just below thresholds
        assert format_stars(0.099999) == "*"
        assert format_stars(0.049999) == "**"
        assert format_stars(0.009999) == "***"

    def test_missing(self):
        assert format_stars(None) == ""
        assert format_stars(np.nan) == ""

    def test_custom_levels(self):
        result = format_stars(0.04, levels=(0.10, 0.05, 0.01))
        assert result == "**"
        # single-star threshold
        assert format_stars(0.08, levels=(0.10, 0.05)) == "*"

    def test_empty_levels(self):
        assert format_stars(0.001, levels=()) == ""

    def test_negative_pvalue(self):
        # negative p-values are nonsensical; function should treat as missing
        assert format_stars(-0.5) == ""


# ---------------------------------------------------------------------------
# fmt_auto
# ---------------------------------------------------------------------------

class TestFmtAuto:
    @pytest.mark.parametrize(
        "value, expected",
        [
            (1521.109, "1,521"),
            (-1521.109, "-1,521"),
            (590.769, "591"),
            (30.925, "30.9"),
            (3.955, "3.96"),
            # Python uses banker's rounding (round-half-to-even)
            (0.2876, "0.288"),
            (-0.0106, "-0.011"),
            (0.0, "0.000"),
        ],
    )
    def test_magnitude_brackets(self, value, expected):
        assert fmt_auto(value) == expected

    def test_missing(self):
        assert fmt_auto(None) == ""
        assert fmt_auto(np.nan) == ""

    @pytest.mark.parametrize("value", [999.9, -999.9, 0.9999, -0.9999])
    def test_boundary_values(self, value):
        result = fmt_auto(value)
        assert result != ""  # must produce some string representation


# ---------------------------------------------------------------------------
# fmt_val
# ---------------------------------------------------------------------------

class TestFmtVal:
    def test_printf_format(self):
        assert fmt_val(3.14159, "%.3f") == "3.142"

    def test_auto(self):
        assert fmt_val(0.2876, "auto") == "0.288"
        assert fmt_val(1521.0, "auto") == "1,521"

    def test_missing(self):
        assert fmt_val(None) == ""
        assert fmt_val(np.nan) == ""
        assert fmt_val(pd.NA) == ""

    def test_non_numeric(self):
        assert fmt_val("not a number") == ""
        assert fmt_val([1, 2, 3]) == ""

    def test_inf(self):
        assert fmt_val(float("inf")) == ""
        assert fmt_val(float("-inf")) == ""

    def test_zero(self):
        assert fmt_val(0.0, "%.4f") == "0.0000"


# ---------------------------------------------------------------------------
# fmt_int
# ---------------------------------------------------------------------------

class TestFmtInt:
    def test_positive(self):
        assert fmt_int(1521) == "1,521"

    def test_negative(self):
        assert fmt_int(-1521) == "-1,521"

    def test_float_rounds(self):
        assert fmt_int(1521.7) == "1,522"
        assert fmt_int(1521.3) == "1,521"

    def test_missing(self):
        assert fmt_int(None) == ""
        assert fmt_int(np.nan) == ""

    def test_non_numeric(self):
        assert fmt_int("text") == ""

    def test_zero(self):
        assert fmt_int(0) == "0"
