"""Reference parity: pre-trend equivalence tests vs R ``fect``.

``sp.pretrends_equivalence`` ports the diagnostic panel of ``fect:::diagtest``
(Liu, Wang & Xu 2024). The point of these tests is that a failure to reject
"no pre-trend" is usually just low power (Roth 2022); the equivalence tests
reverse the null and ask whether the pre-trend is demonstrably *small*, so a
**small** p-value is the reassuring outcome.

Three statistics are pinned:

* ``f_stat`` / ``f_pvalue`` — the conventional joint Wald/F test,
  ``psi = D' S^-1 D`` scaled by ``(N_bar - k)/((N_bar - 1) k)``.
* ``f_equivalence_pvalue`` — the same statistic against a non-central ``F``
  with non-centrality ``N_bar * f_threshold``.
* ``tost_pvalue`` — two one-sided tests per pre-period against
  ``±tost_threshold``, taking the least favourable period.

What is pinned here is the **test mathematics**, not ``fect``'s counterfactual
estimator: the fixture carries ``fect``'s own pre-period estimates and
bootstrap draws, and the test re-derives its statistics from them. That is the
right boundary — StatsPAI computes these diagnostics for any event-study
result with a joint covariance (Callaway-Sant'Anna, interactive FE, ...),
rather than reimplementing ``fect``'s estimator.

Fixture
-------
``_fixtures/fect_equivalence_R.json``, generated with R 4.5.2 / ``fect``
2.4.1 on a 120-unit x 15-period staggered panel::

    out <- fect(y ~ D, data = d, index = c("id", "t"), method = "fe",
                se = TRUE, nboots = 200, parallel = FALSE, force = "two-way")
    out$test.out

References
----------
- Liu, L., Wang, Y. and Xu, Y. (2024). "A Practical Guide to Counterfactual
  Estimators for Causal Inference with Time-Series Cross-Sectional Data."
  *American Journal of Political Science*, 68(1), 160-176.
  [@liu2024practical]
- Roth, J. (2022). "Pretest with Caution: Event-Study Estimates after Testing
  for Parallel Trends." *American Economic Review: Insights*, 4(3), 305-322.
  [@roth2022pretest]
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pytest

from statspai.did._equivalence import pretrend_equivalence

_FIXTURE = pathlib.Path(__file__).parent / "_fixtures" / "fect_equivalence_R.json"


@pytest.fixture(scope="module")
def fect_case():
    if not _FIXTURE.exists():  # pragma: no cover - fixture ships with the repo
        pytest.skip(f"missing fixture: {_FIXTURE}")
    d = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    boot = np.asarray(d["boot"], dtype=float)  # (k, nboots)
    res = pretrend_equivalence(
        pre_estimates=np.asarray(d["pre_est"], dtype=float),
        pre_cov=np.cov(boot),
        n_bar=int(d["N_bar"]),
        pre_se=np.asarray(d["pre_se"], dtype=float),
        f_threshold=float(d["f_threshold"]),
        tost_threshold=float(d["tost_threshold"]),
    )
    return res, d["expect"]


def test_f_statistic_matches_fect(fect_case):
    res, exp = fect_case
    assert res.f_stat == pytest.approx(exp["f_stat"], rel=1e-10)


def test_f_pvalue_matches_fect(fect_case):
    res, exp = fect_case
    assert res.f_pvalue == pytest.approx(exp["f_p"], rel=1e-10)


def test_f_equivalence_pvalue_matches_fect(fect_case):
    """Non-central F.

    SciPy's and R's non-central-F implementations differ by ~2e-10 in
    absolute terms. The p-value here is ~2.4e-5, so that is ~1e-5 in relative
    terms — the tolerance reflects the implementations, not the port.
    """
    res, exp = fect_case
    assert res.f_equivalence_pvalue == pytest.approx(exp["f_equiv_p"], abs=1e-9)


def test_tost_pvalue_matches_fect(fect_case):
    res, exp = fect_case
    assert res.tost_pvalue == pytest.approx(exp["tost_equiv_p"], rel=1e-10)


def test_degrees_of_freedom_match_fect(fect_case):
    res, exp = fect_case
    assert res.df1 == exp["df1"]
    assert res.df2 == exp["df2"]


def test_tost_is_skipped_without_a_threshold(fect_case):
    """No universal outcome scale exists, so the TOST must not be invented."""
    res, _ = fect_case
    d = json.loads(_FIXTURE.read_text(encoding="utf-8"))
    boot = np.asarray(d["boot"], dtype=float)
    out = pretrend_equivalence(
        pre_estimates=np.asarray(d["pre_est"], dtype=float),
        pre_cov=np.cov(boot),
        n_bar=int(d["N_bar"]),
    )
    assert out.tost_pvalue is None
    assert out.f_stat == pytest.approx(res.f_stat, rel=1e-12)


def test_verdict_distinguishes_undetected_from_bounded(fect_case):
    """The whole point: 'not detected' and 'shown to be small' differ."""
    res, _ = fect_case
    assert res.f_pvalue > 0.05  # nothing detected on this panel
    assert res.tost_pvalue < 0.05  # and it *is* bounded
    assert "bounded" in res.verdict()


def test_singular_covariance_fails_loudly():
    from statspai.exceptions import DataInsufficient

    d = np.array([0.1, 0.2])
    singular = np.ones((2, 2))
    with pytest.raises(DataInsufficient, match="singular"):
        pretrend_equivalence(d, singular, n_bar=50)


def test_too_few_treated_units_fails_loudly():
    from statspai.exceptions import DataInsufficient

    with pytest.raises(DataInsufficient, match="more treated units"):
        pretrend_equivalence(np.zeros(5), np.eye(5), n_bar=4)


# ---------------------------------------------------------------------------
# End-to-end: sp.pretrends_equivalence on a fitted DiD result
# ---------------------------------------------------------------------------

_MPDTA = (
    pathlib.Path(__file__).resolve().parents[1]
    / "orig_parity"
    / "data"
    / "02_mpdta_original.csv"
)


@pytest.fixture(scope="module")
def cs_result():
    import pandas as pd

    import statspai as sp

    if not _MPDTA.exists():  # pragma: no cover - fixture ships with the repo
        pytest.skip(f"locked mpdta fixture missing: {_MPDTA}")
    mp = pd.read_csv(_MPDTA)
    return sp.callaway_santanna(mp, y="lemp", g="first_treat", t="year", i="countyreal")


def test_pretrends_equivalence_runs_on_a_cs_result(cs_result):
    """The public entry point must recover the joint covariance itself."""
    import statspai as sp

    eq = sp.pretrends_equivalence(cs_result, tost_threshold=0.05)
    assert 0.0 <= eq.f_pvalue <= 1.0
    assert 0.0 <= eq.f_equivalence_pvalue <= 1.0
    assert 0.0 <= eq.tost_pvalue <= 1.0
    assert eq.df1 >= 1 and eq.df2 >= 1
    assert isinstance(eq.verdict(), str)


def test_pretrends_equivalence_skips_tost_without_threshold(cs_result):
    import statspai as sp

    eq = sp.pretrends_equivalence(cs_result)
    assert eq.tost_pvalue is None
    assert eq.tost_threshold is None


def test_pretrends_equivalence_needs_influence_functions():
    """A diagonal covariance would invalidate the joint F test — fail loudly."""
    import statspai as sp
    from statspai.core.results import CausalResult
    from statspai.exceptions import MethodIncompatibility

    bare = CausalResult(
        method="fake",
        estimand="ATT",
        estimate=0.1,
        se=0.05,
        pvalue=0.1,
        ci=(0.0, 0.2),
        alpha=0.05,
        n_obs=100,
    )
    with pytest.raises(MethodIncompatibility, match="influence functions"):
        sp.pretrends_equivalence(bare)
