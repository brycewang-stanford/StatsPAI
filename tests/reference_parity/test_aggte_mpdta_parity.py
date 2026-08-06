"""Reference parity: ``sp.aggte`` all four aggregations vs R ``did::aggte``.

This is the guard for the three ⚠️ correctness fixes to ``sp.aggte``:

1. ``type='group'`` collapsed the per-cohort θ(g) with equal ``1/K`` weights;
   R weights each cohort by its share of treated units.
2. The ``bstrap=False`` path summed per-cell variances as if the ATT(g, t)
   cells were independent, producing standard errors ~0.64x the truth.
3. The aggregation weights are *estimated* cohort shares, but the variance
   treated them as fixed, dropping R's ``did:::wif`` term and leaving the
   standard errors up to 8% too small.

With all three fixed, every aggregation matches R ``did`` 2.3.0 on canonical
``did::mpdta`` to 10 decimal places on **both** the point estimate and the
influence-function standard error.

Reference generation
--------------------
R 4.5.2, ``did`` 2.3.0, run against the package's own ``data(mpdta)``::

    a <- att_gt(yname="lemp", tname="year", idname="countyreal",
                gname="first.treat", data=mpdta,
                control_group="nevertreated", bstrap=FALSE, cband=FALSE)
    aggte(a, type=<type>, bstrap=FALSE, cband=FALSE)

.. warning::
   When regenerating these numbers, read ``mpdta`` from the R package or cast
   the ``gname`` column to double first.  ``did::att_gt`` recodes never-treated
   units to ``Inf`` internally; if ``gname`` arrives as an **integer** column
   (which ``data.table::fread`` produces for this CSV) that assignment silently
   truncates to ``NA``, the never-treated group is lost, and ``att_gt`` returns
   6 ATT(g, t) cells instead of 12 with badly wrong aggregates.  This is an R
   data-type trap, not a StatsPAI difference — the CSV fixture below is
   byte-for-byte the same data as ``did::mpdta`` (verified column by column).

Data provenance
---------------
``tests/orig_parity/data/02_mpdta_original.csv``, SHA256
``1b789c34e12ff490b2f432217a1f70af334117523eb44d20eb842ed92a574661``;
``lemp`` and ``first_treat`` verified identical to ``did::mpdta`` row for row.

References
----------
- Callaway, B. and Sant'Anna, P.H.C. (2021). "Difference-in-Differences with
  Multiple Time Periods." *Journal of Econometrics*, 225(2), 200-230.
  [@callaway2021difference]
"""

from __future__ import annotations

import hashlib
import pathlib

import pandas as pd
import pytest

import statspai as sp

_MPDTA = (
    pathlib.Path(__file__).resolve().parents[1]
    / "orig_parity"
    / "data"
    / "02_mpdta_original.csv"
)
_MPDTA_SHA256 = "1b789c34e12ff490b2f432217a1f70af334117523eb44d20eb842ed92a574661"

# R did 2.3.0, control_group='nevertreated', base_period='varying' (R default),
# bstrap=FALSE.  (att, se) per aggregation type.
R_AGGTE = {
    "simple": (-0.0399512752, 0.0120340128),
    "group": (-0.0310182822, 0.0124460593),
    "calendar": (-0.0417004321, 0.0159718519),
    "dynamic": (-0.0772398215, 0.0199649891),
}

# R att_gt ATT(g, t) grid — the building blocks the aggregations consume.
R_ATT_GT = {
    (2004, 2004): -0.01050325,
    (2004, 2005): -0.07042316,
    (2004, 2006): -0.13725874,
    (2004, 2007): -0.10081136,
    (2006, 2004): 0.00652011,
    (2006, 2005): -0.00275082,
    (2006, 2006): -0.00459461,
    (2006, 2007): -0.04122447,
    (2007, 2004): 0.03050666,
    (2007, 2005): -0.00272589,
    (2007, 2006): -0.03108712,
    (2007, 2007): -0.02605441,
}


@pytest.fixture(scope="module")
def cs_result():
    if not _MPDTA.exists():  # pragma: no cover - fixture shipped with the repo
        pytest.skip(f"locked mpdta fixture missing: {_MPDTA}")
    digest = hashlib.sha256(_MPDTA.read_bytes()).hexdigest()
    assert (
        digest == _MPDTA_SHA256
    ), f"mpdta fixture changed; expected {_MPDTA_SHA256}, got {digest}"
    mp = pd.read_csv(_MPDTA)
    return sp.callaway_santanna(
        mp,
        y="lemp",
        g="first_treat",
        t="year",
        i="countyreal",
        control_group="nevertreated",
        base_period="varying",
    )


def test_att_gt_grid_matches_r(cs_result):
    """The ATT(g, t) building blocks match R before any aggregation."""
    got = {
        (int(g), int(t)): a
        for g, t, a in zip(
            cs_result.detail["group"], cs_result.detail["time"], cs_result.detail["att"]
        )
    }
    for key, expected in R_ATT_GT.items():
        assert key in got, f"ATT{key} missing from the (g, t) grid"
        assert got[key] == pytest.approx(
            expected, abs=1e-6
        ), f"ATT{key}: StatsPAI {got[key]:.8f} vs R {expected:.8f}"


@pytest.mark.parametrize("agg_type", sorted(R_AGGTE))
def test_aggte_point_estimate_matches_r(cs_result, agg_type):
    """All four aggregation point estimates match R to 10 decimals."""
    att_r, _ = R_AGGTE[agg_type]
    got = sp.aggte(cs_result, type=agg_type, bstrap=False)
    assert got.estimate == pytest.approx(
        att_r, abs=1e-9
    ), f"{agg_type}: StatsPAI {got.estimate:.10f} vs R {att_r:.10f}"


@pytest.mark.parametrize("agg_type", sorted(R_AGGTE))
def test_aggte_analytic_se_matches_r(cs_result, agg_type):
    """Analytic SEs now match R exactly, not merely within a few percent.

    History of this assertion, which is a record of two separate bugs:

    1. Originally the ``bstrap=False`` path treated the ATT(g, t) cells as
       independent and came in around **0.64x** these values.  Fixing that
       (aggregating through the influence functions) got within ~2.5%, and
       the tolerance was set to ``rel=0.025`` to accommodate the residual.
    2. That residual was itself a bug: the aggregation weights are
       *estimated* cohort shares, and the variance was treating them as
       fixed — R's ``did:::wif`` term was missing.  With it restored the
       agreement is exact, so the tolerance is now ~1e-9.

    If this assertion starts needing a loose ``rel=`` again, something has
    regressed — do not widen it without finding out what.
    """
    _, se_r = R_AGGTE[agg_type]
    got = sp.aggte(cs_result, type=agg_type, bstrap=False)
    assert got.se == pytest.approx(se_r, abs=1e-9), (
        f"{agg_type}: SE {got.se:.10f} vs R {se_r:.10f} " f"(ratio {got.se / se_r:.6f})"
    )


def test_group_overall_is_not_the_equal_weight_mean(cs_result):
    """The regression that motivated the fix, pinned against R directly.

    mpdta's treated cohorts are badly unbalanced (100 / 200 / 655
    observations), so the equal-weight mean and the treated-size-weighted mean
    are far apart.  Only the latter is what R reports.
    """
    got = sp.aggte(cs_result, type="group", bstrap=False)
    equal_weight = float(got.detail["att"].mean())

    assert got.estimate == pytest.approx(R_AGGTE["group"][0], abs=1e-9)
    assert got.estimate != pytest.approx(
        equal_weight, abs=1e-6
    ), "group overall collapsed back to the equal-weight mean of theta(g)"
