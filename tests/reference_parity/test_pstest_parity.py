"""Stata ``pstest`` parity for ``sp.psmatch2(...).pstest()``.

``pstest`` is the table every psmatch2 user prints to justify their matched
sample, so a port that cannot reproduce it cannot be checked.  Three of its
conventions are easy to get wrong, and each one is pinned below:

1. **The post-matching standardised bias keeps the unmatched denominator.**
   Both rows divide by ``sqrt((v1u + v0u)/2)`` from the *raw* sample.  Using
   the matched variances instead makes the "after" bias incomparable to the
   "before" bias — and silently flatters or damns the match.

2. **Matched moments use Stata importance weights**, whose variance divides
   by ``Σw - 1`` rather than ``n - 1``.

3. **The Rubin block comes from pstest's own probit**, refit on the matched
   sample, *not* from psmatch2's propensity score.  Reusing ``_pscore``
   reproduces the per-covariate rows perfectly while getting Rubin's B wrong
   by ~5% and the pseudo-R² wrong outright — a failure mode that only a
   fixture like this one catches.

Fixture provenance
------------------
``_fixtures/_generate_pstest.do`` under Stata 18 MP, psmatch2 4.0.12,
pstest 4.2.2.
"""

from __future__ import annotations

import json
import pathlib
import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp

_FIXTURE_DIR = pathlib.Path(__file__).parent / "_fixtures"
_REGEN = (
    "Regenerate with tests/reference_parity/_fixtures/_generate_pstest.do "
    "under Stata 18 + psmatch2 + pstest."
)

# Per-covariate quantities agree to ~1e-14; the sample-level block involves a
# refit probit and agrees to ~1e-9.
_RTOL_COVARIATE = 1e-10
_RTOL_SUMMARY = 1e-7
# MeanBias / MedBias are accumulated by pstest into a Stata *float* variable
# (`qui g `sumbias0' = .`), so only single precision survives.
_RTOL_BIAS_SUMMARY = 1e-6


@pytest.fixture(scope="module")
def data() -> pd.DataFrame:
    path = _FIXTURE_DIR / "pstest_data.csv"
    if not path.exists():  # pragma: no cover - fixture ships with the repo
        pytest.skip(f"missing {path.name}. {_REGEN}")
    return pd.read_csv(path)


@pytest.fixture(scope="module")
def stata() -> dict:
    path = _FIXTURE_DIR / "pstest_stata.json"
    if not path.exists():  # pragma: no cover
        pytest.skip(f"missing {path.name}. {_REGEN}")
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def fitted(data):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.psmatch2(
            data, treat="d", outcome="y", covariates=["x1", "x2"], neighbor=1
        )


@pytest.fixture(scope="module")
def table(fitted):
    return fitted.pstest()


_COVARIATE_FIELDS = [
    "mean_treated_unmatched",
    "mean_control_unmatched",
    "mean_treated_matched",
    "mean_control_matched",
    "pct_bias_unmatched",
    "pct_bias_matched",
    "pct_reduction_abs_bias",
    "variance_ratio_unmatched",
    "variance_ratio_matched",
]


class TestPerCovariateRows:
    @pytest.mark.parametrize("var", ["x1", "x2"])
    @pytest.mark.parametrize("field", _COVARIATE_FIELDS)
    def test_matches_stata(self, table, stata, var, field):
        ours = float(table.table.loc[var, field])
        theirs = stata["per_covariate"][var][field]
        assert ours == pytest.approx(theirs, rel=_RTOL_COVARIATE)

    def test_post_match_bias_uses_the_unmatched_denominator(self, table, data):
        """The convention that makes before/after comparable."""
        x = data["x1"].to_numpy(dtype=float)
        t = data["d"].to_numpy(dtype=float)
        v1u = np.var(x[t == 1], ddof=1)
        v0u = np.var(x[t == 0], ddof=1)
        pooled_unmatched = np.sqrt((v1u + v0u) / 2)

        row = table.table.loc["x1"]
        implied = (
            100
            * (row["mean_treated_matched"] - row["mean_control_matched"])
            / pooled_unmatched
        )
        assert implied == pytest.approx(row["pct_bias_matched"], rel=1e-12)

    def test_reduction_is_in_absolute_bias(self, table):
        for var in ["x1", "x2"]:
            row = table.table.loc[var]
            expected = (
                -100
                * (abs(row["pct_bias_matched"]) - abs(row["pct_bias_unmatched"]))
                / abs(row["pct_bias_unmatched"])
            )
            assert row["pct_reduction_abs_bias"] == pytest.approx(expected, rel=1e-12)

    def test_treated_means_are_unchanged_by_att_matching(self, table):
        """ATT matching reweights controls only, so treated means must hold."""
        for var in ["x1", "x2"]:
            row = table.table.loc[var]
            assert row["mean_treated_matched"] == pytest.approx(
                row["mean_treated_unmatched"], rel=1e-12
            )


class TestSummaryBlock:
    @pytest.mark.parametrize("sample", ["unmatched", "matched"])
    @pytest.mark.parametrize("field", ["ps_r2", "p_chi2", "rubin_b", "rubin_r"])
    def test_matches_stata(self, table, stata, sample, field):
        ours = table.summary_stats[sample][field]
        theirs = stata["summary"][sample][field]
        assert ours == pytest.approx(theirs, rel=_RTOL_SUMMARY)

    @pytest.mark.parametrize("sample", ["unmatched", "matched"])
    @pytest.mark.parametrize("field", ["mean_bias", "median_bias"])
    def test_bias_summary_matches_stata(self, table, stata, sample, field):
        ours = table.summary_stats[sample][field]
        theirs = stata["summary"][sample][field]
        assert ours == pytest.approx(theirs, rel=_RTOL_BIAS_SUMMARY)

    def test_rubin_flags_follow_the_2001_rule(self, table):
        """B < 25 and R in [0.5, 2] is the published rule of thumb."""
        unm = table.summary_stats["unmatched"]
        mat = table.summary_stats["matched"]
        assert unm["rubin_balanced"] is False  # B = 107.4
        assert mat["rubin_balanced"] is True  # B = 15.0, R = 1.38

    def test_matching_reduced_every_headline_statistic(self, table):
        unm = table.summary_stats["unmatched"]
        mat = table.summary_stats["matched"]
        assert mat["ps_r2"] < unm["ps_r2"]
        assert mat["mean_bias"] < unm["mean_bias"]
        assert abs(mat["rubin_b"]) < abs(unm["rubin_b"])


class TestRegressionGuards:
    """These caught real bugs during development; keep them."""

    def test_rubin_block_does_not_reuse_the_psmatch2_pscore(self, table, data):
        """pstest probits; psmatch2 logits. Reusing _pscore is ~5% wrong."""
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m = sp.psmatch2(
                data, treat="d", outcome="y", covariates=["x1", "x2"], neighbor=1
            )
        p = np.clip(m.matched_data["_pscore"].to_numpy(dtype=float), 1e-12, 1 - 1e-12)
        logit_index = np.log(p / (1 - p))
        t = data["d"].to_numpy(dtype=float)
        v1 = np.var(logit_index[t == 1], ddof=1)
        v0 = np.var(logit_index[t == 0], ddof=1)
        wrong_b = (
            100
            * (logit_index[t == 1].mean() - logit_index[t == 0].mean())
            / np.sqrt((v1 + v0) / 2)
        )
        # The wrong construction is close enough to look right by eye, which
        # is exactly why it needs a fixture rather than a sanity check.
        assert abs(wrong_b - table.summary_stats["unmatched"]["rubin_b"]) > 1e-3

    def test_iw_variance_divides_by_weight_total(self, table, data):
        """Stata iweights: Var = sum w (x-m)^2 / (sum w - 1), not n - 1."""
        from statspai.matching._pstest import _iw_mean_var

        x = np.array([1.0, 2.0, 3.0])
        w = np.array([2.0, 3.0, 5.0])
        mean, var = _iw_mean_var(x, w)
        assert mean == pytest.approx((2 + 6 + 15) / 10.0)
        expected = float(np.sum(w * (x - mean) ** 2) / (w.sum() - 1))
        assert var == pytest.approx(expected)


class TestResultSurface:
    def test_summary_renders_both_blocks(self, table):
        text = str(table.summary())
        assert "Unmatched" in text and "Matched" in text
        assert "MeanBias" in text and "Ps R2" in text
        assert "Rubin 2001" in text

    def test_table_is_indexed_by_covariate_in_order(self, table):
        assert list(table.table.index) == ["x1", "x2"]

    def test_covariate_subset_is_honoured(self, fitted):
        sub = fitted.pstest(covariates=["x2"])
        assert list(sub.table.index) == ["x2"]

    def test_balance_and_pstest_are_different_by_design(self, fitted, table):
        """balance() reports StatsPAI's conventions, pstest() reports Stata's."""
        bal = fitted.balance()
        assert "smd_weighted" in bal.table.columns
        assert "pct_bias_matched" in table.table.columns
