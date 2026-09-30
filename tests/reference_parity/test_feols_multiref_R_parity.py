"""``sp.feols`` accepts several reference levels in ``i()``.

fixest writes ``i(rel, ref = c(-1, -5))`` to omit two event times (a
normalisation Busting the Princelings and AI-tocracy use); pyfixest takes a
single ``ref``. ``sp.feols`` rewrites the term (merging the extra
references into the first is the same model) and maps the coefficient names
back. Reference: R fixest 0.14.0 on ``_fixtures/feols_multiref.csv``
(``_generate_feols_multiref_R.R``); names identical, coefficients and
clustered SEs to 1e-11.
"""

from __future__ import annotations

import json
import pathlib
import warnings

import pandas as pd
import pytest

import statspai as sp

_FIX = pathlib.Path(__file__).parent / "_fixtures"
R = json.loads((_FIX / "feols_multiref_R.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def data():
    return pd.read_csv(_FIX / "feols_multiref.csv")


@pytest.mark.parametrize("spelling", ["ref=[-1, -5]", "ref=c(-1, -5)"])
def test_matches_fixest(data, spelling):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = sp.feols(f"y ~ i(rel, {spelling}) | u + t", data=data, cluster="u")
    assert list(r.params.index) == R["names"]
    ref_b = pd.Series(R["b"], index=R["names"])
    ref_se = pd.Series(R["se"], index=R["names"])
    pd.testing.assert_series_equal(
        r.params[R["names"]], ref_b, check_names=False, rtol=1e-11, atol=1e-13
    )
    pd.testing.assert_series_equal(
        r.std_errors[R["names"]], ref_se, check_names=False, rtol=1e-11
    )
    assert "rel_multiref1" not in data.columns  # caller's frame untouched


def test_single_reference_and_unknown_level(data):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        one = sp.feols("y ~ i(rel, ref=[-1]) | u + t", data=data)
        plain = sp.feols("y ~ i(rel, ref=-1) | u + t", data=data)
    pd.testing.assert_series_equal(one.params, plain.params)
    with pytest.raises(sp.MethodIncompatibility, match="not a value"):
        sp.feols("y ~ i(rel, ref=[-1, -99]) | u + t", data=data)
