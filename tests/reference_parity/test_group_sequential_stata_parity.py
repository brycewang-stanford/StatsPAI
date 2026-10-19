"""``sp.group_sequential`` against Stata 18 ``gsbounds``.

117 designs from ``_fixtures/_generate_gsbounds_Stata.do``: every boundary
family (classical O'Brien-Fleming, Pocock, Wang-Tsiatis; Lan-DeMets error
spending of O'Brien-Fleming and Pocock type, Kim-DeMets, Hwang-Shih-DeCani),
two-sided, upper and lower designs, equal and unequal information, efficacy
only, futility only, efficacy plus nonbinding or binding futility, and
several alpha / power pairs.

Stata is run with ``intpointsscale(60) infotol(1e-12)``. At its defaults it
integrates with 12r - 3 Simpson points (r = 20) and stops the bisection for
the maximum information of futility designs at ``infotol(1e-6)``; its
designs then spend beta to within about 1e-6 of the target (its own
``r(bspent)`` ends at 0.199999 for beta = 0.2) and futility bounds sit up to
3e-5 away from the exact solution. With the tolerances tightened the two
implementations agree to 3e-7 (relative) on every bound and on the
information ratio.

The one bound that does not is the first look of the error-spending
O'Brien-Fleming design with information fractions (0.1, 0.3, 0.4, 1): the
first bound has the closed form ``Phi^-1(1 - a*(0.1)/2)`` = 6.9913517 and
Stata reports 6.9913410 (it solves for it numerically in the far tail). The
test compares that bound on the probability scale, where the two differ by
2e-16.
"""

from __future__ import annotations

import json
import pathlib
import re

import numpy as np
import pytest
from scipy import stats

import statspai as sp

_FIX = pathlib.Path(__file__).parent / "_fixtures"
CASES = json.loads((_FIX / "gsbounds_Stata.json").read_text(encoding="utf-8"))

_COLS = {
    "Efficacy:Lower": "efficacy_lower",
    "Efficacy:Upper": "efficacy_upper",
    "Futility:Lower": "futility_lower",
    "Futility:Upper": "futility_upper",
}


def _spec(cmd: str, role: str):
    m = re.search(role + r"\((\w+)(?:\(([-\d.]+)\))?(, binding)?\)", cmd)
    if not m:
        return None, None, False
    return m.group(1), (float(m.group(2)) if m.group(2) else None), bool(m.group(3))


def _call(cmd: str):
    eff, eff_p, _ = _spec(cmd, "efficacy")
    fut, fut_p, binding = _spec(cmd, "futility")
    side = "upper" if " upper" in cmd else ("lower" if " lower" in cmd else "two-sided")
    kw = {}
    for name in ("alpha", "power"):
        m = re.search(name + r"\(([\d.]+)\)", cmd)
        if m:
            kw[name] = float(m.group(1))
    m = re.search(r"nlooks\((\d+)\)", cmd)
    if m:
        kw["n_looks"] = int(m.group(1))
    else:
        info = re.search(r"information\(([^)]*)\)", cmd).group(1)
        kw["information"] = [float(x) for x in info.split()]
    return sp.group_sequential(
        side=side,
        efficacy=eff,
        efficacy_param=eff_p,
        futility=fut,
        futility_param=fut_p,
        binding=binding,
        **kw,
    )


@pytest.mark.parametrize("case", CASES, ids=[c["cmd"][10:60] for c in CASES])
def test_bounds_and_information_ratio_match_gsbounds(case):
    res = _call(case["cmd"])
    assert res.info_ratio == pytest.approx(float(case["info_ratio"]), rel=1e-6)
    cols = case["cols"].split()
    rows = np.array(
        [[np.nan if v is None else float(v) for v in r] for r in case["rows"]]
    )
    np.testing.assert_allclose(res.boundaries["info_frac"], rows[:, 0], atol=1e-12)
    for j, name in enumerate(cols):
        if name not in _COLS:
            continue
        ref = rows[:, j]
        ours = res.boundaries[_COLS[name]].to_numpy()
        np.testing.assert_array_equal(np.isnan(ours), np.isnan(ref))
        ok = ~np.isnan(ref)
        rel = np.abs(ours[ok] - ref[ok]) / np.maximum(np.abs(ref[ok]), 1e-3)
        # Far-tail bounds: compare the tail probability (see module docstring).
        tail = np.abs(stats.norm.sf(np.abs(ours[ok])) - stats.norm.sf(np.abs(ref[ok])))
        assert np.all((rel <= 1e-6) | (tail <= 1e-15)), (name, ours, ref)


def test_closed_form_first_look_of_spending_design():
    # The first error-spending bound needs no integration.
    res = sp.group_sequential(information=[1, 3, 4, 10], efficacy="errobfleming")
    spent = 4 * stats.norm.sf(stats.norm.isf(0.05 / 4) / np.sqrt(0.1))
    assert res.boundaries["efficacy_upper"].iloc[0] == pytest.approx(
        stats.norm.isf(spent / 2), rel=1e-12
    )
