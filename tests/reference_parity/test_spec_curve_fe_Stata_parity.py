"""``sp.spec_curve(fe=)`` against Stata ``regress`` / ``reghdfe``.

Fixed effects are a choice dimension of the specification curve. Each FE
specification must be the ``reghdfe`` fit (the singleton firm dropped,
absorbed levels charged, the firm effects nested in the industry cluster not
charged, t with G - 1 df), and the no-FE clustered row ``regress,
vce(cluster)`` -- including its t(G - 1) p-value, which used t(n - k) before.
Reference: ``_fixtures/_generate_spec_curve_fe_Stata.do`` (Stata 18).
"""

from __future__ import annotations

import json
import pathlib

import pandas as pd
import pytest

import statspai as sp

_FIX = pathlib.Path(__file__).parent / "_fixtures"
S = json.loads((_FIX / "spec_curve_fe_Stata.json").read_text(encoding="utf-8"))
FE = {"none": "(none)", "firm": "firm", "firmyear": "firm, year"}
VCE = {"unadjusted": "nonrobust", "robust": "hc1", "cluster": "cluster"}


@pytest.fixture(scope="module")
def curve():
    df = pd.read_csv(_FIX / "spec_curve_fe.csv", float_precision="round_trip")
    res = sp.spec_curve(
        df,
        y="y",
        x="x",
        controls=[["w"]],
        fe=[[], ["firm"], ["firm", "year"]],
        se_types=["nonrobust", "hc1"],
        cluster_var="ind",
    )
    return res.results_df.set_index(["fe", "se_type"])


@pytest.mark.parametrize("key", sorted(S))
def test_matches_stata(curve, key):
    fe, vce = key.split("_")
    row = curve.loc[(FE[fe], VCE[vce])]
    ref = S[key]
    assert row["estimate"] == pytest.approx(ref["b"], rel=1e-10)
    assert row["se"] == pytest.approx(ref["se"], rel=1e-10)
    assert row["pvalue"] == pytest.approx(ref["p"], abs=1e-15, rel=1e-8)
    assert int(row["nobs"]) == ref["N"]


def test_fe_is_a_choice_dimension(curve):
    assert len(curve) == 9
