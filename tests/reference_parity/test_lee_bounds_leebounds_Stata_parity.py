"""``sp.lee_bounds(trimming='leebounds')`` against Stata ``leebounds``.

Reference: Stata 18, ``leebounds y d, select(s)`` (Tauchmann, SSC v1.5) on
the eight samples of ``_fixtures/leebounds_samples.csv``, from
``_fixtures/_generate_leebounds_stata.do``: six continuous outcomes stored
as float32 (either arm retained more often) and two integer outcomes with
heavy ties.

``leebounds`` stores the trimming percentage and the quantile threshold in
local macros (16 significant digits). The rounded threshold can sit one unit
in the last place beyond the data value, and then ``y >= threshold`` drops
the quantile observation: Lee's rule (``trimming='quantile'``, the default)
and Stata differ. The UCT replication (QJE 2016, Table III assets) hit this:
lower bound -2.51 under Lee's rule against Stata's -3.38. With ties the
macro value equals the data and Stata takes its fractional tie branch.
``trimming='leebounds'`` reproduces all of it; bounds and analytic SEs are
held to 1e-12 and 1e-8. Six of the eight samples differ
from Lee's rule, so the artefact is the rule rather than the exception on
float-stored outcomes.
"""

from __future__ import annotations

import json
import pathlib
import warnings

import pandas as pd
import pytest

import statspai as sp

_FIX = pathlib.Path(__file__).parent / "_fixtures"


@pytest.fixture(scope="module")
def ref():
    return json.loads((_FIX / "leebounds_stata.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def samples():
    # round_trip: pandas' default fast parser can land one ulp off the
    # value Stata reads, and a one-ulp shift is exactly what this test is
    # about.
    return pd.read_csv(_FIX / "leebounds_samples.csv", float_precision="round_trip")


@pytest.mark.parametrize("k", [str(k) for k in range(1, 9)])
def test_matches_leebounds(ref, samples, k):
    R = ref[k]
    d = samples[samples["sample"] == int(k)]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = sp.lee_bounds(
            d,
            y="y",
            treat="d",
            selection="s",
            se_method="analytic",
            trimming="leebounds",
        )
    mi = r.model_info
    assert mi["lower_bound"] == pytest.approx(R["lower"], rel=1e-12, abs=1e-12)
    assert mi["upper_bound"] == pytest.approx(R["upper"], rel=1e-12, abs=1e-12)
    assert mi["se_lower"] == pytest.approx(R["se_lower"], rel=1e-8)
    assert mi["se_upper"] == pytest.approx(R["se_upper"], rel=1e-8)
    assert mi["trimming_fraction"] == pytest.approx(R["trim"], rel=1e-12)


def test_invalid_trimming_raises(samples):
    with pytest.raises(sp.MethodIncompatibility, match="leebounds"):
        sp.lee_bounds(samples, y="y", treat="d", selection="s", trimming="stata")
