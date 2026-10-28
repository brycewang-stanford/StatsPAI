"""sp.private_info_bound against the author's own Stata code.

Hendren's replication package (Econometrica supplement 10931) computes the
lower bound in ``makem.do``; ``_generate_private_info_bound_Stata.do`` runs
its loop body in Stata 18 on ``private_info_bound_data.csv`` (two rejection
classes, age cells nested in them, household clusters), with the
elicitation basis built as in the author's ``createhrs_all.do``. The
built-in basis with ``focal=(0, 1)`` and ``interact='agep'`` spans the same
columns, so both sides fit the same model.

Tolerances: the bound and the Wald statistic 1e-9 relative (observed 2e-11
and 1e-11; sp.probit's own stopping rule), per-person ``m_Z`` 1e-9 relative
to the largest value (observed 1.0e-10 probit, 2e-13 linear), Wald p-values
1e-8 (observed 1.2e-9 on a p of 4e-35).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import statspai as sp

FIX = Path(__file__).resolve().parent / "_fixtures"


def _ref():
    out = {}
    for line in (
        (FIX / "private_info_bound_Stata.txt").read_text(encoding="utf-8").splitlines()
    ):
        k, v = line.split(None, 1)
        out[k] = v.strip()
    return out


@pytest.mark.parametrize("model", ["probit", "linear"])
def test_matches_the_authors_makem_do(model):
    df = pd.read_csv(FIX / "private_info_bound_data.csv")
    ref = _ref()
    assert ref["stata_version"] == "18"
    rows = pd.read_csv(FIX / "private_info_bound_Stata_rows.csv").set_index("id")
    res = sp.private_info_bound(
        df,
        "loss",
        "z",
        ["x1", "x2"],
        classes="reject_list",
        cells="agegrp",
        interact="agep",
        focal=(0, 1),
        model=model,
        cluster="hhid",
    )
    for j in (1, 2):
        t = res.table.loc[j]
        assert t["estimate"] == pytest.approx(float(ref[f"{model}_M_{j}"]), rel=1e-9)
        assert t["wald_stat"] == pytest.approx(
            float(ref[f"{model}_stat_{j}"]), rel=1e-9
        )
        assert t["wald_p"] == pytest.approx(float(ref[f"{model}_p_{j}"]), rel=1e-8)
        assert t["z_terms"] == 8
    m = res.rows["m_z"].to_numpy()
    s = rows.loc[df["id"], f"m_{model}"].to_numpy()
    assert np.max(np.abs(m - s)) / np.max(np.abs(s)) < 1e-9
