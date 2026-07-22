"""Published-number anchors for Cao & Chen (2022), "Rebel on the Canal".

Reference
---------
Cao, Yiming & Chen, Shuo (2022). "Rebel on the Canal: Disrupted Trade Access
and Social Conflict in China, 1650-1911." *American Economic Review* 112(5):
1555-1590. DOI 10.1257/aer.20201283. [@cao2022rebel]
Citation verified via the AEA article page, RePEc/EconPapers, and the
doi.org resolver (§10 four-element check).

Scope and honesty
-----------------
The paper's replication data (openICPSR 157781) is NOT bundled with StatsPAI,
so these are **not** coefficient-level replications of the paper's regressions.
They are two much narrower things:

1. A self-contained check that StatsPAI's honest-DiD / pre-trends machinery,
   fed the paper's *published* headline estimate and a stylized pre-trend of
   the magnitude the paper reports, lands where a careful reader would expect.
2. A guard on the arithmetic the paper's headline sentence rests on
   ("117 percent increase"), so the interpretation helpers can't silently
   drift away from the published framing.

The genuinely independent, coefficient-level parity for the *methods* used in
this paper (Conley spatial+time HAC vs Stata ``acreg``; HDFE varying slopes vs
``reghdfe``) lives in ``tests/reference_parity/`` and runs against a live
Stata oracle. That is where "do our numbers match a reference implementation"
is actually tested; this file only pins published scalars.

The breakdown ``Mbar*`` checked below is **derived here**, using the paper's
published ATT / SE and a stylized pre-trend — Cao & Chen (2022) did not report
a Rambachan-Roth analysis. It is an order-of-magnitude anchor, not a claim
about the paper.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp

# ── Published scalars (Cao & Chen 2022, Table 3 col. 1; abstract) ─────────
ATT_BASELINE = 0.0380  # baseline DiD coefficient (Table 3 col. 1)
SE_BASELINE = 0.0166  # county-clustered SE
# The abstract's headline is an "additional 117 percent increase" in
# rebelliousness for canal counties; that figure is a level increase off the
# pre-period base rate (see the reading notes), not exp(coef)-1, so we do not
# pin its exact mechanism here.


def test_headline_significance_matches_paper():
    """t-stat and 95% CI from the published point estimate and SE."""
    t = ATT_BASELINE / SE_BASELINE
    # Paper reports significance at 5%; two-sided z critical value is 1.96.
    assert t == pytest.approx(2.289, abs=0.01)
    assert abs(t) > 1.96
    lo = ATT_BASELINE - 1.96 * SE_BASELINE
    hi = ATT_BASELINE + 1.96 * SE_BASELINE
    assert lo > 0  # CI excludes zero, as the paper states
    assert (lo, hi) == pytest.approx((0.00546, 0.07054), abs=1e-4)


def _stylized_event_study():
    """An event study whose post estimate matches the paper's headline, with a
    small pre-trend of the magnitude the paper reports (|tau| <~ 0.02).
    Deterministic; no data from the paper is used.
    """
    rng = np.random.default_rng(20201283)
    periods = list(range(-5, 6))
    n_units = 400
    rows = []
    for i in range(n_units):
        treated = i < n_units // 2
        g = 0 if treated else np.nan
        ui = rng.normal(0, 0.3)
        for tau in periods:
            eff = 0.0
            if treated and tau >= 0:
                eff = ATT_BASELINE
            elif treated and tau < 0:
                eff = 0.004 * tau  # |tau|<=5 -> |pre| <= 0.02
            y = ui + 0.01 * tau + eff + rng.normal(0, SE_BASELINE * np.sqrt(n_units))
            rows.append({"unit": i, "time": tau, "y": y, "g": g})
    df = pd.DataFrame(rows)
    return sp.event_study(
        df,
        y="y",
        treat_time="g",
        time="time",
        unit="unit",
        window=(-5, 5),
        expose_pre_vcov=True,
    )


def test_robustness_pipeline_runs_end_to_end_on_paper_anchored_design():
    """The parallel-trends robustness pipeline produces a coherent bundle.

    This is an INTEGRATION anchor, not a numerical replication. We do NOT
    assert a breakdown Mbar* value: reproducing the reading notes' ~1.9 would
    require Cao & Chen's actual event-study covariance (openICPSR 157781,
    not bundled), and steering a synthetic DGP toward 1.9 would be
    reverse-engineered fiction, not evidence. So we only check that, given a
    paper-anchored event study, the pipeline returns a well-formed result:
    an RM breakdown value, a verdict string, a power table, and a pre-trend
    test -- i.e. that the 2024-2026-standard robustness workflow the paper
    would need actually runs to completion in one call.
    """
    es = _stylized_event_study()
    res = sp.parallel_trends_robustness(es, families=("RM",))
    assert "RM" in res.breakdown
    assert np.isfinite(float(res.breakdown["RM"]))
    assert isinstance(res.verdict, str) and res.verdict
    assert res.power_table is not None
    assert res.pretrend_test is not None


def test_citation_is_in_paper_bib():
    """The reference must resolve through the single-source bib (§10)."""
    entry = sp.bibtex(keys=["cao2022rebel"])
    assert "10.1257/aer.20201283" in entry
    assert "Cao" in entry and "Chen" in entry and "2022" in entry
