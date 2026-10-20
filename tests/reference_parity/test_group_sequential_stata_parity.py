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


# ---------------------------------------------------------------------------
# Expected sample size against Stata 18 ``gsdesign`` (r(ESS0), r(ESS1)),
# 16 designs from ``_fixtures/_generate_gsdesign_ess_Stata.do``.
#
# Efficacy-only and binding-futility designs agree to 1e-7. With nonbinding
# futility bounds the two report different quantities. Ours is the expected
# sample size when the futility rule is followed. Stata's is
#     sum_k n_k * (P_eff,k [futility ignored] + P_fut,k [futility followed]),
# which counts a path that crosses a futility bound and would later cross
# the efficacy bound twice. The weights then sum to more than one: in the
# futility-only design Stata's ESS1 (59.417) exceeds the maximum sample
# size (59.166). Its manual describes a third rule (drop the futility bound
# at delta = 0, keep it otherwise), which reproduces neither of its numbers.
# The test rebuilds Stata's figure from our crossing probabilities to 1e-7
# and checks ours by an identity and a simulation.
# ---------------------------------------------------------------------------

ESS = json.loads((_FIX / "gsdesign_ess_Stata.json").read_text(encoding="utf-8"))


def _ess_call(case):
    cmd = case["cmd"]
    eff, eff_p, _ = _spec(cmd, "efficacy")
    fut, fut_p, binding = _spec(cmd, "futility")
    kw = {"power": float(re.search(r"power\(([\d.]+)\)", cmd).group(1))}
    m = re.search(r"nlooks\((\d+)\)", cmd)
    info = re.search(r"information\(([^)]*)\)", cmd)
    if m:
        kw["n_looks"] = int(m.group(1))
    elif info:
        kw["information"] = [float(x) for x in info.group(1).split()]
    side = case["direction"]
    return sp.group_sequential(
        side=side,
        efficacy=eff,
        efficacy_param=eff_p,
        futility=fut,
        futility_param=fut_p,
        binding=binding,
        n_fixed=float(case["N_fixedfrac"]),
        **kw,
    )


def _crossings(res, theta, follow_futility):
    """Per-look efficacy and futility crossing probabilities at drift
    ``theta`` (0 = H0, 1 = the design alternative)."""
    from statspai.experimental import _gs_core as core
    from statspai.experimental.group_sequential import _no_futility

    b_ = res.boundaries
    t = b_["info_frac"].to_numpy()
    k = len(t)
    sides = 2 if res.side == "two-sided" else 1
    if res.side == "lower":  # mirror onto the upper scale
        b, a = -b_["efficacy_lower"].to_numpy(), -b_["futility_upper"].to_numpy()
    else:
        b = b_["efficacy_upper"].to_numpy()
        a = b_["futility_upper" if sides == 2 else "futility_lower"].to_numpy()
    b = np.where(np.isnan(b), np.inf, b)
    if res.futility is None or not follow_futility:
        a = _no_futility(k, sides)
    else:
        a = np.where(np.isnan(a), 0.0 if sides == 2 else -np.inf, a)
    info = res.info_ratio * res.detail["info_fixed_per_unit_drift"]
    path = core._Path(t * info, 1.0) if theta else core._Path(t, 0.0)
    eff, fut = np.zeros(k), np.zeros(k)
    for j in range(k):
        eff[j] = core.efficacy_prob(path, b[j], sides)
        if j < k - 1:
            fut[j] = core.futility_prob(path, a[j], sides)
            path.advance(core.continuation(a[j], b[j], sides))
        else:
            fut[j] = path.mass() - eff[j]
    return eff, fut


@pytest.mark.parametrize("case", ESS, ids=[c["cmd"][16:80] for c in ESS])
def test_expected_sample_size_against_gsdesign(case):
    res = _ess_call(case)
    assert res.info_ratio == pytest.approx(float(case["info_ratio"]), rel=1e-6)
    n = res.boundaries["n"].to_numpy()
    assert n[-1] == pytest.approx(float(case["N_max"]), rel=1e-6)
    ours = (res.expected_info_h0 * res.n_fixed, res.expected_info_h1 * res.n_fixed)
    nonbinding = res.futility is not None and not res.binding
    for theta, got, want in zip((0, 1), ours, (case["ESS0"], case["ESS1"])):
        eff_f, fut_f = _crossings(res, theta, follow_futility=True)
        # ours: futility followed; the stopping probabilities sum to one
        assert eff_f.sum() + fut_f.sum() == pytest.approx(1.0, abs=1e-10)
        assert got == pytest.approx(float(np.dot(n, eff_f + fut_f)), rel=1e-12)
        assert got <= n[-1] * (1 + 1e-12)
        if not nonbinding:
            assert got == pytest.approx(float(want), rel=1e-7)
        else:
            eff_n, _ = _crossings(res, theta, follow_futility=False)
            stata_rule = float(np.dot(n, eff_n + fut_f))
            assert stata_rule == pytest.approx(float(want), rel=1e-7)


def test_stata_nonbinding_ess_exceeds_the_maximum_sample_size():
    case = next(
        c
        for c in ESS
        if c["cmd"].endswith("power(0.9) futility(errpocock)")
        and "efficacy" not in c["cmd"]
    )
    assert float(case["ESS1"]) > float(case["N_max"])
    res = _ess_call(case)
    assert res.expected_info_h1 * res.n_fixed < float(case["N_max"])


def test_nonbinding_expected_sample_size_by_simulation():
    # Brownian motion on the information scale, futility rule followed:
    # the simulated mean sample size matches ours within 4 Monte Carlo SEs.
    case = next(c for c in ESS if "nlooks(4)" in c["cmd"] and "futility" in c["cmd"])
    res = _ess_call(case)
    b_ = res.boundaries
    t, n = b_["info_frac"].to_numpy(), b_["n"].to_numpy()
    e = b_["efficacy_upper"].fillna(np.inf).to_numpy()
    f = b_["futility_lower"].fillna(-np.inf).to_numpy()
    info = res.info_ratio * res.detail["info_fixed_per_unit_drift"]
    rng = np.random.default_rng(20261020)
    reps = 200_000
    for theta, want in ((0.0, res.expected_info_h0), (1.0, res.expected_info_h1)):
        inc = rng.normal(size=(reps, len(t))) * np.sqrt(np.diff(np.r_[0.0, t]))
        w = np.cumsum(inc, axis=1) + theta * np.sqrt(info) * t
        z = w / np.sqrt(t)
        stop = (z >= e) | (z < f)
        stop[:, -1] = True
        size = n[np.argmax(stop, axis=1)]
        se = size.std() / np.sqrt(reps)
        assert abs(size.mean() - want * res.n_fixed) < 4 * se
