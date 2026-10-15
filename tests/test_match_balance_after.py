"""``sp.match`` reports balance after matching, and labels both tables.

Until 2026-10-16 the only balance table on the result (``detail`` and
``model_info['balance']``) was the raw sample's, unlabelled; an agent
reported the pre-matching SMDs (age -1.20, re75 -2.17 on nsw_dw) as the
balance achieved by matching.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import statspai as sp


@pytest.fixture(scope="module")
def fit():
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        nsw = sp.datasets.nsw_dw()
        return sp.match(
            nsw,
            y="re78",
            treat="treat",
            covariates=["age", "education", "re74", "re75"],
        )


def test_tables_are_labelled(fit):
    assert fit.detail.attrs["stage"] == "before matching"
    assert fit.model_info["balance_after"].attrs["stage"] == "after matching"
    assert fit.model_info["balance_before"] is fit.detail


def test_after_matching_means_are_the_weighted_matched_means(fit):
    md = fit.matched_data
    t = (md["_treated"] == 1).to_numpy()
    w = md["_weight"].fillna(0.0).to_numpy()
    after = fit.model_info["balance_after"].set_index("variable")
    for col in ("age", "education", "re74", "re75"):
        x = md[col].to_numpy(dtype=float)
        assert after.loc[col, "mean_treated"] == pytest.approx(
            np.average(x[t], weights=w[t]), abs=5e-5
        )
        assert after.loc[col, "mean_control"] == pytest.approx(
            np.average(x[~t], weights=w[~t]), abs=5e-5
        )
        sd = np.sqrt((x[t].var(ddof=1) + x[~t].var(ddof=1)) / 2)
        diff = np.average(x[t], weights=w[t]) - np.average(x[~t], weights=w[~t])
        assert after.loc[col, "smd"] == pytest.approx(diff / sd, abs=5e-5)


def test_matching_improves_balance_and_says_so(fit):
    assert fit.model_info["max_abs_smd_before"] > 2.0
    assert fit.model_info["max_abs_smd_after"] < 0.2
    diag = fit.to_dict(detail="agent")["diagnostics"]
    assert diag["max_abs_smd_after"] == fit.model_info["max_abs_smd_after"]
