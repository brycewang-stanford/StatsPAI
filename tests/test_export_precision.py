"""One precision system, reachable from every exit of a result object.

Before this, the same ``CausalResult`` left by four doors at four
precisions — auto in Markdown/HTML/Word, a hard-coded ``%.4f`` in LaTeX,
six decimals in Excel, six more in ``.summary()`` — and only two of those
doors had a knob. These tests pin the agreement.
"""

import warnings
from dataclasses import dataclass

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai._result_serialize import ResultProtocolMixin
from statspai.exceptions import MethodIncompatibility
from statspai.output._format import STAT_MAX_SIGNIFICANT, fmt_statistic, format_pair
from statspai.output._journals import star_note_for


@pytest.fixture(scope="module")
def result():
    rng = np.random.default_rng(3)
    n = 2000
    x = rng.normal(size=n)
    d = (rng.random(n) < 1 / (1 + np.exp(-x))).astype(int)
    df = pd.DataFrame(
        {"y": 0.2876543 * d + 0.5 * x + rng.normal(size=n), "d": d, "x": x}
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.psm(df, y="y", d="d", X=["x"])


class TestFmtStatistic:
    @pytest.mark.parametrize(
        "value,expected",
        [
            # Unchanged from the "%.3f" this replaced — the ordinary cases.
            (0.0904, "0.090"),
            (0.9994, "0.999"),
            (10.5, "10.500"),
            (12.34, "12.340"),
            (302.955, "302.955"),
            # ...and the case that motivated it: nine significant figures,
            # the last three noise.
            (538582.398, "538,582"),
            (1234.5678, "1,234.57"),
        ],
    )
    def test_precision_is_magnitude_aware(self, value, expected):
        assert fmt_statistic(value) == expected

    def test_decimals_shrink_as_the_integer_part_grows(self):
        # The integer part is never truncated — that would misstate the
        # value. Only the decimals are negotiable, and they run out once
        # the integer part alone fills the significant-figure budget.
        seen = []
        for exponent in range(0, 9):
            rendered = fmt_statistic(1.23456789 * 10**exponent)
            integer, _, fraction = rendered.partition(".")
            int_digits = len(integer.replace(",", "").lstrip("-"))
            assert len(fraction) <= 3, rendered
            assert len(fraction) == min(3, max(0, STAT_MAX_SIGNIFICANT - int_digits))
            seen.append(len(fraction))
        # Monotone: precision is given up, never regained.
        assert seen == sorted(seen, reverse=True)
        assert seen[0] == 3 and seen[-1] == 0

    def test_missing_and_nonfinite_render_empty(self):
        assert fmt_statistic(None) == ""
        assert fmt_statistic(np.nan) == ""
        assert fmt_statistic(np.inf) == ""


class TestFormatPair:
    def test_estimate_and_se_share_one_precision(self):
        # The published-table convention: never "1.07 (0.054)".
        point, se = format_pair(1.0713, 0.0542)
        assert (point, se) == ("1.071", "0.054")

    def test_confidence_bounds_inherit_the_estimate_precision(self):
        point, se, lo, hi = format_pair(1521.6, 2.07, 1517.5, 1525.7)
        assert len({len(v.split(".")[-1]) for v in (point, lo, hi)}) == 1

    def test_explicit_fmt_applies_verbatim(self):
        assert format_pair(1.0713, 0.0542, fmt="%.4f") == ("1.0713", "0.0542")


class TestStarNote:
    def test_common_levels_unchanged(self):
        assert star_note_for((0.10, 0.05, 0.01)) == ("*** p<0.01, ** p<0.05, * p<0.10")

    def test_tight_level_is_not_rounded_out_of_existence(self):
        # A fixed "%.2f" rendered 0.001 as the self-contradicting "p<0.00".
        assert "p<0.001" in star_note_for((0.05, 0.01, 0.001))
        assert "p<0.00," not in star_note_for((0.05, 0.01, 0.001))


@dataclass
class _Generic(ResultProtocolMixin):
    """A lightweight result carried by the generic Field/Value renderer."""

    att: float = -0.094458
    se: float = 0.161388
    big: float = 538582.4
    converged: bool = True


class TestGenericRendererExitsAgree:
    """The mixin's own three display surfaces must not disagree either.

    244 classes inherit it, and it carried two conventions internally:
    ``%.4g`` in LaTeX against ``%.6g`` in Markdown and Word, so one ``att``
    read ``-0.09446`` and ``-0.094458`` from the same object.
    """

    def _value(self, text, field):
        for line in text.splitlines():
            parts = [p.strip() for p in line.replace("&", "|").split("|")]
            parts = [p for p in parts if p]
            if parts and parts[0] == field and len(parts) > 1:
                return parts[1].rstrip("\\").strip()
        return None

    @pytest.mark.parametrize("field", ["att", "se", "big"])
    def test_latex_agrees_with_markdown(self, field):
        result = _Generic()
        assert self._value(result.to_latex(), field) == self._value(
            result.to_markdown(), field
        )

    def test_large_value_is_not_forced_into_scientific_notation(self):
        # "%.4g" rendered 538582.4 as "5.386e+05" in a publication table.
        assert self._value(_Generic().to_markdown(), "big") == "538,582"

    def test_digits_is_honoured(self):
        assert self._value(_Generic().to_latex(digits=5), "att") == "-0.09446"

    def test_fmt_and_digits_together_raise(self):
        with pytest.raises(MethodIncompatibility):
            _Generic().to_latex(digits=3, fmt="%.4f")


class TestResultExitsAgree:
    """Every door out of one result object reports the same number."""

    def _att_decimals(self, text, marker):
        for line in text.splitlines():
            if marker in line:
                for token in line.replace("|", " ").split():
                    if "." in token and token.replace(".", "").strip("-").isdigit():
                        return len(token.split(".")[1])
        return None

    def test_latex_agrees_with_markdown(self, result):
        # Was 4 decimals in LaTeX against 3 in Markdown.
        assert self._att_decimals(
            result.to_latex(), "propensity"
        ) == self._att_decimals(result.to_markdown(), "propensity")

    def test_summary_agrees_with_markdown(self, result):
        # Was 6 decimals on screen against 3 in every export.
        assert self._att_decimals(result.summary(), "ATT:") == self._att_decimals(
            result.to_markdown(), "ATT"
        )

    @pytest.mark.parametrize("digits", [2, 3, 5])
    def test_digits_is_honoured_on_every_surface(self, result, digits):
        assert self._att_decimals(result.summary(digits=digits), "ATT:") == digits
        assert (
            self._att_decimals(result.to_latex(digits=digits), "propensity") == digits
        )
        assert self._att_decimals(result.to_markdown(digits=digits), "ATT") == digits

    @pytest.mark.parametrize("surface", ["summary", "to_latex", "to_markdown"])
    def test_fmt_and_digits_together_raise(self, result, surface):
        # Silently preferring one would make the other a lie.
        with pytest.raises(MethodIncompatibility):
            getattr(result, surface)(digits=3, fmt="%.4f")

    def test_pvalue_floors_consistently(self, result):
        # ".4f" printed 0.0001; every surface should now floor at <0.001
        # rather than implying a precision the p-value does not carry.
        assert "<0.001" in result.summary()
