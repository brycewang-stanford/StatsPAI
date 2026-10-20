"""Clean-room ``pstest`` (docs/dev/cleanroom/pstest.md): behaviour pinned by
black-box Stata ``pstest`` 4.2.2 runs beyond the committed fixture."""

from __future__ import annotations

import pathlib
import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.matching._pstest import pstest_table

_FIX = pathlib.Path(__file__).parent / "_fixtures"


@pytest.fixture(scope="module")
def matched():
    data = pd.read_csv(_FIX / "pstest_data.csv")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = sp.psmatch2(
            data, treat="d", outcome="y", covariates=["x1", "x2"], neighbor=1
        )
    return m.matched_data.assign(k=1.0), m.treat


def _run(frame, treat, covs):
    return pstest_table(
        frame, treat=treat, covariates=covs, weight_col="_weight",
        support_col="_support", pscore_col="_pscore",
    )  # fmt: skip


def test_a_constant_covariate_leaves_the_probit_and_its_lr_test_unchanged(matched):
    # Stata's probit omits the constant regressor, so pstest's LR chi2 and its
    # p-value are those of the model without it (checked on Stata 18: p before
    # matching 1.49744901164455e-14 with and without the constant on a
    # 400-row sample). The previous code counted the omitted column in the
    # degrees of freedom.
    frame, treat = matched
    _, with_k = _run(frame, treat, ["x1", "x2", "k"])
    _, without = _run(frame, treat, ["x1", "x2"])
    for sample in ("unmatched", "matched"):
        for field in ("ps_r2", "lr_chi2", "p_chi2", "rubin_b", "rubin_r"):
            assert with_k[sample][field] == pytest.approx(
                without[sample][field], rel=1e-12
            )


def test_a_constant_covariate_row_is_undefined_but_its_means_are_kept(matched):
    frame, treat = matched
    table, _ = _run(frame, treat, ["x1", "k"])
    row = table.loc["k"]
    assert row["mean_treated_unmatched"] == 1.0 and row["mean_control_matched"] == 1.0
    assert np.isnan(
        row[["pct_bias_unmatched", "t_matched", "variance_ratio_matched"]]
    ).all()
