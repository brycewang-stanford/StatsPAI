"""Tier D guard — pin the LaLonde classic-track replication numbers.

Part of the P1 campaign (see ``.tierd_campaign/CAMPAIGN.md``). The
``sp.replicate('lalonde_1986')`` registry ships *golden numbers* for the classic
track, but no test guarded the live 1:1 NN propensity-score-matching ATT. This
test is that missing guard: it pins all three classic LaLonde numbers to their
current deterministic values on the bundled real data, including the stable
index-anchored nearest-neighbor tie-break in ``sp.match``.

The naive and covariate-adjusted OLS values reproduce R ``MatchIt`` to the
dollar; the PSM value is the deterministic ``sp.match(method='nearest')`` output
after exact-distance ties on the binary covariates are resolved by source index.

This guard accompanies the nearest-neighbor tie-break fix; only exact-tie
matching designs can move.
"""

import pytest

import statspai as sp

COVS = ["age", "educ", "black", "hispanic", "married", "nodegree", "re74", "re75"]


@pytest.fixture(scope="module")
def lalonde():
    df, _ = sp.replicate("lalonde_1986")
    return df


def test_naive_ols_matches_matchit(lalonde):
    naive = sp.regress("re78 ~ treat", data=lalonde, robust="hc1")
    assert float(naive.params["treat"]) == pytest.approx(-635.03, abs=1.0)


def test_adjusted_ols_matches_matchit(lalonde):
    adj = sp.regress("re78 ~ treat + " + " + ".join(COVS), data=lalonde, robust="hc1")
    assert float(adj.params["treat"]) == pytest.approx(1548.24, abs=1.0)


# The index-anchored tie-break makes the matched set deterministic given the
# floating-point distance matrix: GitHub CI lands on bitwise-identical
# 1967.9396843243246 across ubuntu (OpenBLAS), windows, and macos-15 runners,
# all Python 3.9-3.13. A residual ULP-level *near*-tie sensitivity remains in
# the BLAS-computed propensity distances themselves (distinct covariate rows
# whose distances differ by ~1e-13 can swap order across BLAS builds): local
# Accelerate on macOS 26 lands on 1963.4266356756752 with both numpy 2.3/scipy
# 1.16 and numpy 2.4/scipy 1.17, ruling out a library-version cause. Pin BOTH
# observed fixed points exactly (±0.1): any algorithm change in sp.match moves
# the value off both anchors and fails loudly, while a third BLAS environment
# producing a new value also fails — surfacing it for triage instead of being
# absorbed by a wide band. Tracked in .tierd_campaign/CAMPAIGN.md.
PSM_ATT_ANCHORS = (1967.94, 1963.43)

# Since 1.39 the default keeps every tied control (ties='all'), so the 24
# treated units with exact ties on the binary covariates no longer take
# whichever control comes first. The value moves to 1968.80 and no longer
# depends on the order of the rows; the anchors above are the fixed points of
# ties='first'.
PSM_ATT_ALL_TIES = 1968.80


def test_psm_att_is_deterministic_and_pinned(lalonde):
    # Deterministic across runs; pinned to the known fixed points so any
    # tie-break / algorithm change in sp.match is caught.
    kw = dict(data=lalonde, y="re78", treat="treat", covariates=COVS, method="nearest")
    vals = [float(sp.match(**kw).estimate) for _ in range(3)]
    assert len(set(round(v, 6) for v in vals)) == 1  # deterministic within a run
    assert vals[0] == pytest.approx(PSM_ATT_ALL_TIES, abs=0.1)

    with pytest.warns(UserWarning, match="equally close matches"):
        first = float(sp.match(ties="first", **kw).estimate)
    assert any(
        first == pytest.approx(anchor, abs=0.1) for anchor in PSM_ATT_ANCHORS
    ), f"PSM ATT {first:.4f} is off both pinned anchors {PSM_ATT_ANCHORS}"


def test_default_psm_att_equals_stata_teffects(lalonde):
    """Stata 18, on the same 614 rows written with %.17g:

        teffects psmatch (re78) (treat age educ black hispanic married ///
            nodegree re74 re75, logit), atet

    gives _b = 1968.799715855857 and _se = 1126.321219230800. teffects keeps
    every tied match; so does the default here, and the two agree to the
    last digit. The value ties='first' gives equals no reference.
    """
    kw = dict(data=lalonde, y="re78", treat="treat", covariates=COVS, method="nearest")
    fit = sp.match(se_method="abadie_imbens_2016", **kw)
    assert float(fit.estimate) == pytest.approx(1968.799715855857, rel=1e-12)
    # the standard error goes through a numerically differentiated score
    assert float(fit.se) == pytest.approx(1126.321219230800, rel=1e-6)


def test_psm_att_does_not_depend_on_row_order(lalonde):
    kw = dict(y="re78", treat="treat", covariates=COVS, method="nearest")
    shuffled = lalonde.sample(frac=1.0, random_state=3).reset_index(drop=True)
    base = float(sp.match(data=lalonde, **kw).estimate)
    assert float(sp.match(data=shuffled, **kw).estimate) == pytest.approx(
        base, abs=1e-6
    )
    with pytest.warns(UserWarning, match="equally close matches"):
        moved = float(sp.match(data=shuffled, ties="first", **kw).estimate)
    # under ties='first' the same data in another order gives another ATT
    assert abs(moved - base) > 10


def test_psm_recovers_experimental_benchmark(lalonde):
    # The scientific content of DW (1999): matching recovers the ~$1,794
    # experimental benchmark and removes the negative naive selection bias.
    naive = float(
        sp.regress("re78 ~ treat", data=lalonde, robust="hc1").params["treat"]
    )
    psm = float(
        sp.match(
            data=lalonde, y="re78", treat="treat", covariates=COVS, method="nearest"
        ).estimate
    )
    assert naive < 0 < psm
    assert 1500.0 < psm < 2500.0
