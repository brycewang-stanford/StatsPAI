"""Fuzzy RD with covariates whose means are not zero, against rdrobust 4.0.0.

``rdrobust_bw`` forms the fuzzy delta-method weights at ``nu = 0`` from each
side's *levels* ``tau_Y / tau_T`` at the cutoff, and those levels depend on
where the covariates are located. Through 2026-10-18 ``sp.rdrobust`` demeaned
``covs`` before the bandwidth cascade, so on this fixture its mserd ``h`` was
4e-3 to 5e-3 (relative) away from R while ``b`` and every fixed-h estimate
matched. The older fuzzy fixture (``rdsenate_fuzzy.csv``) standardises its
covariates, which is why it never saw the gap.

Fixture and reference numbers: ``_fixtures/_generate_rd_fuzzy_covs_level_R.R``.
Tolerance: rel 1e-6, the Track A budget.
"""

from __future__ import annotations

import json
import pathlib
import warnings

import pandas as pd
import pytest

import statspai as sp

_FIX = pathlib.Path(__file__).parent / "_fixtures"
RTOL = 1e-6


@pytest.fixture(scope="module")
def rjson():
    path = _FIX / "rd_fuzzy_covs_level_R.json"
    if not path.exists():  # pragma: no cover
        pytest.skip("run _generate_rd_fuzzy_covs_level_R.R first")
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def df():
    return pd.read_csv(_FIX / "rd_fuzzy_covs_level.csv")


def _fit(df, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.rdrobust(df, y="y", x="x", c=0, fuzzy="d", covs=["z1", "z2"], **kw)


SPECS = [
    ("fuzzy_covs_p1", {}),
    ("fuzzy_covs_p2", dict(p=2)),
    ("fuzzy_covs_cluster", dict(cluster="g")),
    ("fuzzy_covs_hc1", dict(vce="hc1")),
    ("fuzzy_covs_cerrd", dict(bwselect="cerrd")),
    ("fuzzy_covs_msetwo", dict(bwselect="msetwo")),
]


@pytest.mark.parametrize("spec,kw", SPECS)
def test_fuzzy_covs_matches_r(rjson, df, spec, kw):
    ref = rjson[spec]
    info = _fit(df, **kw).model_info
    h, b = info["bandwidth_h"], info["bandwidth_b"]
    h_l, h_r = h if isinstance(h, tuple) else (h, h)
    b_l, b_r = b if isinstance(b, tuple) else (b, b)
    assert h_l == pytest.approx(ref["h_left"], rel=RTOL)
    assert h_r == pytest.approx(ref["h_right"], rel=RTOL)
    assert b_l == pytest.approx(ref["b_left"], rel=RTOL)
    assert b_r == pytest.approx(ref["b_right"], rel=RTOL)
    assert info["conventional"]["estimate"] == pytest.approx(
        ref["coef_conventional"], rel=RTOL
    )
    assert info["robust"]["estimate"] == pytest.approx(ref["coef_robust"], rel=RTOL)
    assert info["conventional"]["se"] == pytest.approx(ref["se_conventional"], rel=RTOL)
    assert info["robust"]["se"] == pytest.approx(ref["se_robust"], rel=RTOL)


def test_rdbwselect_and_rdrobust_select_the_same_h(rjson, df):
    """Both public entry points run one cascade on the same covariates."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        bw = sp.rdbwselect(df, y="y", x="x", c=0, fuzzy="d", covs=["z1", "z2"])
    assert float(bw.iloc[0]["h_left"]) == pytest.approx(
        rjson["fuzzy_covs_p1"]["h_left"], rel=RTOL
    )
