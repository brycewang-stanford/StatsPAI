"""``sp.event_study_vcov`` and ``sp.uniform_bands`` against R.

Track A modules 05 (Sun-Abraham), 73 (did2s) and 85 (dynamic TWFE) compare
each event-time coefficient and its standard error: the diagonal of the
event-study covariance. A simultaneous band and the HonestDiD FLCI depend
on the off-diagonal blocks too, which no Track A row reaches. This file
pins the full matrix, as ``sp.event_study_vcov`` returns it, against the
matrix each reference package produces on the same CSV bytes:

========  =====================================  ==========================
key       StatsPAI fit                           reference
========  =====================================  ==========================
cs        ``sp.callaway_santanna``               ``did::aggte(type="dynamic")``
                                                 influence function
sunab     ``sp.sun_abraham(share_variance=       ``fixest::sunab``, A V A'
          False)``
twfe      ``sp.event_study``                     ``fixest::feols(i(rel, ref=-1))``
did2s     ``sp.gardner_did(event_study=True)``   ``did2s::did2s(i(rel_year))``
========  =====================================  ==========================

The generator, ``_fixtures/_generate_event_study_vcov_R.R``, documents each
specification. For did2s it tightens fixest's demeaning tolerance from its
default 1e-6 to 1e-11: at the default the reference itself sits ~1e-7 from
the exact least-squares solution (Track A module 73's documented gap), and
the point here is to pin the exact one.

``uniform_bands`` draws its sup-t critical value by Monte Carlo, so it has
no bit-exact counterpart. The reference is ``mvtnorm::qmvnorm`` at
``abseps = 1e-6`` on the same correlation matrix, which serves as the
known value. The test is an equivalence test against it across seeds, with
the margin stated before running (see ``test_sup_t_critical_value_...``).
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
from scipy import stats

import statspai as sp

_ROOT = Path(__file__).resolve().parents[2]
_FIXTURE = Path(__file__).resolve().parent / "_fixtures" / "event_study_vcov_R.json"
_MPDTA = _ROOT / "tests" / "r_parity" / "data" / "05_sunab.csv"
_TWFE = _ROOT / "tests" / "r_parity" / "data" / "85_twfe_event_study.csv"

KEYS = ["cs", "sunab", "twfe", "did2s"]


@pytest.fixture(scope="module")
def ref():
    return json.loads(_FIXTURE.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def fits():
    mp = pd.read_csv(_MPDTA)
    tw = pd.read_csv(_TWFE)
    # sp.event_study wants NaN for never-treated; the shared CSV uses 0.
    tw["g_nan"] = tw["g"].where(tw["g"] > 0)
    common = dict(y="lemp", g="first_treat", t="year", i="countyreal")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return {
            "cs": sp.callaway_santanna(mp, **common),
            "sunab": sp.sun_abraham(mp, **common, share_variance=False),
            "twfe": sp.event_study(
                tw,
                y="y",
                treat_time="g_nan",
                time="time",
                unit="unit",
                window=(-4, 4),
                cluster="unit",
            ),
            "did2s": sp.gardner_did(
                mp,
                y="lemp",
                group="countyreal",
                time="year",
                first_treat="first_treat",
                event_study=True,
            ),
        }


@pytest.mark.parametrize("key", KEYS)
def test_event_study_vcov_matches_reference_elementwise(fits, ref, key):
    """Every entry, diagonal and off-diagonal.

    Observed worst relative gaps: cs 3.4e-15, twfe 5.1e-13, sunab 8.9e-11,
    did2s 6.7e-11 (coefficients 3.3e-10). The budget is 1e-9, inside the
    Track A default of 1e-6.
    """
    es = sp.event_study_vcov(fits[key])
    r = ref[key]
    assert es.times.tolist() == r["times"]
    assert es.joint
    np.testing.assert_allclose(es.beta, r["beta"], rtol=1e-9, atol=0)
    np.testing.assert_allclose(es.vcov, np.asarray(r["vcov"]), rtol=1e-9, atol=0)


@pytest.mark.parametrize("key", KEYS)
def test_uniform_band_is_built_on_that_covariance(fits, ref, key):
    """Given the critical value, the band is deterministic.

    ``se`` is the square root of the reference diagonal, the pointwise
    interval uses the normal quantile, and the simultaneous half-width is
    one common multiple of ``se`` across event times.
    """
    band = sp.uniform_bands(fits[key], which="all")
    r = ref[key]
    assert band["relative_time"].tolist() == r["times"]
    np.testing.assert_allclose(
        band["se"], np.sqrt(np.diag(np.asarray(r["vcov"]))), rtol=1e-9
    )
    z = stats.norm.ppf(0.975)
    np.testing.assert_allclose(band.attrs["crit_pointwise"], z, rtol=1e-15)
    half = (band["cband_upper"] - band["cband_lower"]) / (2.0 * band["se"])
    np.testing.assert_allclose(half, band.attrs["crit_uniform"], rtol=1e-12)
    np.testing.assert_allclose(
        (band["ci_upper"] - band["ci_lower"]) / (2.0 * band["se"]), z, rtol=1e-12
    )


@pytest.mark.parametrize("key", KEYS)
@pytest.mark.parametrize("which", ["all", "post"])
def test_sup_t_critical_value_is_equivalent_to_mvtnorm(fits, ref, key, which):
    """Equivalence test of the Monte Carlo critical value against qmvnorm.

    Margin, fixed before running: +/- 0.005, about 0.2 percent of a
    critical value near 2.5 and about one seed-to-seed SD at the default
    100,000 draws (measured 0.004 to 0.006 on these matrices). Twenty
    seeds; the 90 percent interval for their mean has to lie inside the
    margin (two one-sided tests at 5 percent). The reference's own error,
    abseps = 1e-6 in probability, is three orders of magnitude smaller.
    """
    truth = ref["supt"][key][which]
    draws = np.array(
        [
            sp.uniform_bands(fits[key], which=which, seed=s).attrs["crit_uniform"]
            for s in range(20)
        ]
    )
    diff = draws - truth
    half = stats.t.ppf(0.95, df=draws.size - 1) * diff.std(ddof=1) / np.sqrt(draws.size)
    lo, hi = diff.mean() - half, diff.mean() + half
    assert -0.005 < lo and hi < 0.005, (lo, hi)


def test_bands_differ_across_seeds_by_monte_carlo_error_only(fits):
    """The seed moves the critical value, never the covariance it uses."""
    a = sp.uniform_bands(fits["cs"], seed=1)
    b = sp.uniform_bands(fits["cs"], seed=2)
    np.testing.assert_array_equal(a["se"], b["se"])
    np.testing.assert_array_equal(a["att"], b["att"])
    assert a.attrs["crit_uniform"] != b.attrs["crit_uniform"]
