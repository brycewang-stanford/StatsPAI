"""``sp.did_balance`` against R, including a convention divergence.

The Imbens-Rubin normalized difference is an analytic formula rather than
a package-specific algorithm, so the reference generator
(``_generate_did_balance_R.R``) computes it twice in R: once with
``cobalt::col_w_smd`` 4.6.2 and once as a direct transcription of

    (Xbar_T - Xbar_C) / sqrt((S2_T + S2_C) / 2).

Unweighted, the two R paths and StatsPAI agree to ~1e-14.

Weighted, they do not, and the reason is a genuine convention split that
this module pins rather than papers over:

* ``cobalt`` holds the denominator at the **unweighted** pooled standard
  deviation. That is the right choice for its own use case --- you
  reweight to *improve* balance, and a moving denominator would make the
  before/after comparison meaningless.
* Baker et al. (2026, §4.1) define the weighted statistic with weighted
  variances: "S2_omega,T and S2_omega,C are the sample weighted or
  unweighted variances of the covariates". There the weighted column
  describes a *different population*, not an adjustment of the same one,
  so the dispersion should be that population's too.

``sp.did_balance`` implements the paper's definition, because the paper's
Table 4 is what it reproduces. The third test below proves the gap is
exactly the denominator and nothing else, by reconstructing cobalt's
number from StatsPAI's own group means.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import statspai as sp

_FIXTURES = Path(__file__).parent / "_fixtures"
_PANEL = _FIXTURES / "did_balance_panel.csv"
_REFERENCE = _FIXTURES / "did_balance_reference.csv"

# Both sides evaluate a closed-form expression; the only slack is float
# summation order. Observed worst case is ~1e-13.
_RTOL = 1e-10


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    return pd.read_csv(_PANEL, encoding="utf-8")


@pytest.fixture(scope="module")
def reference() -> pd.DataFrame:
    return pd.read_csv(_REFERENCE, encoding="utf-8")


def _cases(weighted: bool):
    ref = pd.read_csv(_REFERENCE, encoding="utf-8")
    sub = ref[ref["weighted"].astype(bool) == weighted]
    for _, row in sub.iterrows():
        yield pytest.param(
            row["covariate"],
            row["panel"],
            float(row["norm_diff_direct"]),
            float(row["norm_diff_cobalt"]),
            id=f"{row['covariate']}-{row['panel']}",
        )


@pytest.mark.parametrize(
    "covariate,panel_name,r_direct,r_cobalt", list(_cases(weighted=False))
)
def test_unweighted_matches_cobalt_and_the_formula(
    panel, covariate, panel_name, r_direct, r_cobalt
):
    """Unweighted, there is no convention ambiguity: all three agree."""
    bal = sp.did_balance(panel, ["x", "z"], g="g", t="t", i="i")
    row = bal.table[
        (bal.table["covariate"] == covariate) & (bal.table["panel"] == panel_name)
    ].iloc[0]
    assert float(row["norm_diff"]) == pytest.approx(r_direct, rel=_RTOL)
    assert float(row["norm_diff"]) == pytest.approx(r_cobalt, rel=_RTOL)


@pytest.mark.parametrize(
    "covariate,panel_name,r_direct,r_cobalt", list(_cases(weighted=True))
)
def test_weighted_matches_the_papers_definition(
    panel, covariate, panel_name, r_direct, r_cobalt
):
    """Weighted, StatsPAI follows Baker et al. §4.1 (weighted variances)."""
    bal = sp.did_balance(panel, ["x", "z"], g="g", t="t", i="i", weights="w")
    row = bal.table[
        (bal.table["covariate"] == covariate) & (bal.table["panel"] == panel_name)
    ].iloc[0]
    assert float(row["w_norm_diff"]) == pytest.approx(r_direct, rel=_RTOL)


def test_cobalt_divergence_is_exactly_the_denominator(panel, reference):
    """The weighted gap to cobalt is a convention, not a defect.

    Reconstruct cobalt's statistic from StatsPAI's *own* weighted group
    means by swapping in the unweighted pooled standard deviation. If
    that reproduces cobalt, the numerator agrees and the entire
    difference is the denominator choice.
    """
    wtd = sp.did_balance(panel, ["x", "z"], g="g", t="t", i="i", weights="w")
    unw = sp.did_balance(panel, ["x", "z"], g="g", t="t", i="i")

    ref = reference[reference["weighted"].astype(bool)]
    assert len(ref) == 4

    for _, r in ref.iterrows():
        key = (wtd.table["covariate"] == r["covariate"]) & (
            wtd.table["panel"] == r["panel"]
        )
        w_row = wtd.table[key].iloc[0]
        u_row = unw.table[
            (unw.table["covariate"] == r["covariate"])
            & (unw.table["panel"] == r["panel"])
        ].iloc[0]

        # StatsPAI's weighted numerator, over the unweighted denominator.
        weighted_gap = float(w_row["w_mean_treated"]) - float(
            w_row["w_mean_comparison"]
        )
        unweighted_gap = float(u_row["mean_treated"]) - float(u_row["mean_comparison"])
        # denominator implied by the unweighted row
        unweighted_denom = unweighted_gap / float(u_row["norm_diff"])
        rebuilt = weighted_gap / unweighted_denom

        assert rebuilt == pytest.approx(float(r["norm_diff_cobalt"]), rel=1e-8), (
            f"{r['covariate']}/{r['panel']}: swapping in the unweighted "
            "denominator does not reproduce cobalt, so the divergence is "
            "not purely a denominator convention"
        )


def test_reference_records_the_two_r_paths_agreeing_unweighted(reference):
    """Guard the fixture itself: unweighted, the two R paths must agree.

    If cobalt and the direct formula ever diverge on the unweighted rows,
    the convention is ambiguous there too and neither can serve as *the*
    reference without saying which.
    """
    unw = reference[~reference["weighted"].astype(bool)]
    assert len(unw) == 4
    assert unw["direct_vs_cobalt_rel"].max() < 1e-12


def test_weighted_and_unweighted_are_different_statistics(panel):
    """Sanity: the fixture must actually exercise the weighting path."""
    wtd = sp.did_balance(panel, ["x", "z"], g="g", t="t", i="i", weights="w")
    gaps = (wtd.table["w_norm_diff"] - wtd.table["norm_diff"]).abs()
    assert gaps.max() > 1e-3, "fixture no longer separates weighted from unweighted"


def test_changes_panel_is_the_informative_one(panel):
    """The fixture is built level-balanced and change-imbalanced in x."""
    bal = sp.did_balance(panel, ["x"], g="g", t="t", i="i")
    lev = bal.levels.iloc[0]
    chg = bal.changes.iloc[0]
    assert abs(float(lev["norm_diff"])) < 0.25
    assert abs(float(chg["norm_diff"])) > 1.0
    assert np.isfinite(float(chg["norm_diff"]))
