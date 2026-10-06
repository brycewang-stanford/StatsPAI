"""``sp.callaway_santanna(balance='full')`` vs R ``did`` on an unbalanced panel.

R ``did::att_gt`` (``allow_unbalanced_panel = FALSE``, its default) and
Stata ``csdid`` 2.0.0 (``bal(full)``, its default) drop every unit that is
not observed in all periods before they estimate anything. StatsPAI's
default keeps each unit in the cells where both of its periods are
observed (``balance='pair'``, the rule of ``csdid`` 1.8x), so its default
numbers differ from R's on such data. ``balance='full'`` reproduces R.

Reference: R ``did`` 2.5.1 on ``tests/r_parity/data/04_csdid.csv`` with the
rows where ``(countyreal * 7 + year) mod 11 == 0`` removed, from
``_fixtures/_generate_cs_balance_full_R.R``. Stata ``csdid`` 2.0.0
(pre-release, 2026-09-27) agreed with R to 3e-13 on the same rule in the
2026-10-06 three-way comparison.
"""

from __future__ import annotations

import json
import pathlib
import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import MethodIncompatibility

_HERE = pathlib.Path(__file__).parent
REF = json.loads(
    (_HERE / "_fixtures" / "cs_balance_full_R.json").read_text(encoding="utf-8")
)

# Same estimator on the same sample; observed differences are <= 1e-12.
RTOL = 1e-8


@pytest.fixture(scope="module")
def data():
    d = pd.read_csv(_HERE.parent / "r_parity" / "data" / "04_csdid.csv")
    d = d[(d["countyreal"] * 7 + d["year"]) % 11 != 0].reset_index(drop=True)
    assert len(d) == REF["n_rows"]
    return d


def _fit(data, control_group, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.callaway_santanna(
            data,
            y="lemp",
            g="first_treat",
            t="year",
            i="countyreal",
            estimator="dr",
            control_group=control_group,
            base_period="universal",
            **kw,
        )


@pytest.mark.parametrize("control_group", ["nevertreated", "notyettreated"])
def test_balance_full_matches_did(data, control_group):
    ref = REF[control_group]
    fit = _fit(data, control_group, balance="full")
    assert fit.model_info["balance"] == "full"
    assert fit.model_info["n_units"] == ref["n_units"]
    got = fit.detail.set_index(["group", "time"])
    cells = zip(ref["cell_group"], ref["cell_time"], ref["cell_att"], ref["cell_se"])
    for g, t, att, se in cells:
        if se is None:  # universal-base reference period
            continue
        np.testing.assert_allclose(got.loc[(g, t), "att"], att, rtol=RTOL, atol=1e-12)
        np.testing.assert_allclose(got.loc[(g, t), "se"], se, rtol=RTOL)
    for agg in ("simple", "dynamic", "group", "calendar"):
        res = sp.aggte(fit, type=agg, bstrap=False, cband=False)
        np.testing.assert_allclose(res.estimate, ref[agg]["overall_att"], rtol=RTOL)
        np.testing.assert_allclose(res.se, ref[agg]["overall_se"], rtol=RTOL)


def test_balance_full_says_what_it_dropped(data):
    with pytest.warns(UserWarning, match="balance='full' dropped"):
        fit = sp.callaway_santanna(
            data, y="lemp", g="first_treat", t="year", i="countyreal", balance="full"
        )
    n_all = data["countyreal"].nunique()
    assert (
        fit.model_info["balance_n_units_dropped"] == n_all - fit.model_info["n_units"]
    )
    assert fit.model_info["balance_n_units_dropped"] > 0


def test_default_is_pair_and_differs_from_full(data):
    """The default did not change: it is the per-comparison rule."""
    default = _fit(data, "nevertreated")
    pair = _fit(data, "nevertreated", balance="pair")
    full = _fit(data, "nevertreated", balance="full")
    assert default.model_info["balance"] == "pair"
    np.testing.assert_array_equal(default.detail["att"], pair.detail["att"])
    np.testing.assert_array_equal(default.detail["se"], pair.detail["se"])
    assert abs(default.estimate - full.estimate) > 1e-4


def test_balance_none_is_allow_unbalanced_panel(data):
    a = _fit(data, "nevertreated", balance="none")
    b = _fit(data, "nevertreated", allow_unbalanced_panel=True)
    np.testing.assert_array_equal(a.detail["att"], b.detail["att"])
    np.testing.assert_array_equal(a.detail["se"], b.detail["se"])


def test_balance_is_inert_on_a_balanced_panel():
    d = pd.read_csv(_HERE.parent / "r_parity" / "data" / "04_csdid.csv")
    fits = [_fit(d, "nevertreated", balance=b) for b in ("pair", "full", "none")]
    for f in fits[1:]:
        np.testing.assert_allclose(f.detail["att"], fits[0].detail["att"], rtol=1e-12)
        np.testing.assert_allclose(f.detail["se"], fits[0].detail["se"], rtol=1e-12)
    assert "balance_n_units_dropped" not in fits[0].model_info


def test_conflicting_or_misplaced_balance_is_refused(data):
    with pytest.raises(MethodIncompatibility, match="different estimators"):
        _fit(data, "nevertreated", balance="full", allow_unbalanced_panel=True)
    with pytest.raises(MethodIncompatibility, match="panel=True"):
        _fit(data, "nevertreated", balance="full", panel=False)
    with pytest.raises(Exception, match="balance"):
        _fit(data, "nevertreated", balance="units")


def test_balance_full_with_no_complete_unit_is_refused():
    d = pd.read_csv(_HERE.parent / "r_parity" / "data" / "04_csdid.csv")
    # every unit loses one year, and not the same one
    hole = 2003 + d["countyreal"].rank(method="dense").astype(int) % 5
    with pytest.raises(MethodIncompatibility, match="leaves no unit"):
        _fit(d[d["year"] != hole], "nevertreated", balance="full")
