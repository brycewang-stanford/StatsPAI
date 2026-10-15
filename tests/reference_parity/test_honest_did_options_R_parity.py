"""``sp.honest_did`` option values against R ``HonestDiD`` 0.2.8.

Each case changes one thing the earlier references held fixed: the target (a
post period other than the first, the post-period average, arbitrary
weights), ``alpha = 0.10``, a user-supplied relative-magnitudes grid
(``grid_lb`` / ``grid_ub`` / ``grid_points``), an event-time ``window``, and
where the event study comes from -- raw arrays
(``sp.honest_did_from_moments`` on HonestDiD's own ``BCdata_EventStudy``) or a
Callaway-Sant'Anna fit (``did::mpdta`` through ``sp.callaway_santanna``; R
runs ``did::att_gt`` + ``aggte(type = "dynamic")`` and builds sigma from its
influence functions, the pipeline of the HonestDiD README). The two event
studies are compared first: they agree to 1e-14.

**Relative magnitudes** (``honestdid_method='Conditional'``, no simulation):
both packages report the smallest and largest accepted point of the same
grid, and every bound agrees to 2e-13. HonestDiD's default ``'C-LF'`` hybrid
simulates its first-stage critical value with each side's own RNG, so two
cases compare it to within one grid step: a stochastic screen. On
``BCdata`` some sets reach the end of HonestDiD's +/-20 SD grid, where R
reports the grid end; those cases run
with ``grid_expand=False`` (HonestDiD's fixed grid). The default extends the
grid instead (``test_honest_did_moments_R_parity.py``).

**FLCI** (``method='smoothness'``). The reference is HonestDiD's own program
with three numerical approximations removed (see the generator): the exact
folded-normal quantile, ECOS at 1e-10, and the optimal ``h`` by ``optimize``
rather than a bisection with step ``|h0| eps^(1/3)``.

* The half-length -- the optimal value of the program -- agrees to 3e-10 at
  ``M > 0`` (2.3e-8 where ECOS reports ``optimal_inaccurate``), held to 1e-7.
* The interval centre does not: the half-length is flat in ``h`` at its
  minimum (it moves by 1e-11 where the centre moves by 1e-6), so the
  centre carries each solver's noise. It is held to 2e-5 of the half-length
  (measured 1.0e-5 on ``BCdata``, 2.5e-7 on ``mpdta``). This is why the map
  records the FLCI bounds as a documented difference (T4), not a 1e-6 row;
  Track A module 10 (a diagonal covariance) is the T2 row.
* At ``M = 0`` the FLCI has a closed form -- the minimum-variance estimator
  unbiased under a linear trend through the reference period, plus or minus
  ``z`` standard deviations -- computed in R with base linear algebra. The
  half-length agrees to 4e-7. The native centre misses it by up to 1.4e-3 of
  the half-length (``BCdata``, ``e = 0``): the minimum-variance subproblem is
  solved by SLSQP, whose flat directions are not converged. That is an open
  item of the native solver, pinned here so it cannot grow.

Fixture ``_fixtures/honest_did_options_R.json`` from
``_fixtures/_generate_honest_did_options_R.R``.
"""

from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import statspai as sp

_ROOT = Path(__file__).resolve().parents[2]
_REF = json.loads(
    (Path(__file__).parent / "_fixtures" / "honest_did_options_R.json").read_text(
        encoding="utf-8"
    )
)
_CASES = _REF["cases"]
_RM = sorted(
    k
    for k, c in _CASES.items()
    if c["method"] == "relative_magnitude" and c["honestdid_method"] == "Conditional"
)
_CLF = sorted(k for k, c in _CASES.items() if c.get("honestdid_method") == "C-LF")
_FLCI = sorted(k for k, c in _CASES.items() if c["method"] == "smoothness")
_WINDOW = (-3, 2)


@pytest.fixture(scope="module")
def cs():
    mp = pd.read_csv(_ROOT / "tests" / "orig_parity" / "data" / "02_mpdta_original.csv")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.callaway_santanna(
            mp,
            y="lemp",
            g="first_treat",
            t="year",
            i="countyreal",
            control_group="nevertreated",
            base_period="universal",
        )


def _run(cs, case: dict) -> pd.DataFrame:
    kw = dict(method=case["method"], m_grid=case["M"], alpha=case["alpha"])
    target = case["target"]
    if isinstance(target, dict):
        kw["e"] = target["e"]
    else:
        kw["l_vec"] = target  # 'average' or the weights
    if case["method"] == "relative_magnitude":
        kw["honestdid_method"] = case["honestdid_method"]
        if case["input"] == "bc":
            kw["grid_expand"] = False  # HonestDiD's fixed +/-20 SD grid
        if "grid" in case:
            kw.update(
                grid_lb=case["grid"]["lb"],
                grid_ub=case["grid"]["ub"],
                grid_points=case["grid"]["points"],
            )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        if case["input"] == "bc":
            inp = _REF["inputs"]["bc"]
            return sp.honest_did_from_moments(
                np.asarray(inp["betahat"]),
                np.asarray(inp["sigma"]),
                event_times=inp["times"],
                **kw,
            )
        if case["input"] == "mpdta_window":
            return sp.honest_did(cs, window=_WINDOW, **kw)
        return sp.honest_did(cs, **kw)


def test_the_cs_event_study_is_rs(cs):
    ev = sp.event_study_vcov(cs, allow_diagonal=False)
    ref = _REF["inputs"]["mpdta"]
    assert list(ev.times) == ref["times"]
    np.testing.assert_allclose(ev.beta, ref["betahat"], rtol=1e-12)
    np.testing.assert_allclose(ev.vcov, ref["sigma"], rtol=1e-10, atol=1e-16)
    win = _REF["inputs"]["mpdta_window"]["times"]
    assert win == [t for t in ref["times"] if _WINDOW[0] <= t <= _WINDOW[1]]


@pytest.mark.parametrize("cid", _RM)
def test_relative_magnitudes_conditional_set_is_honestdids(cs, cid):
    case = _CASES[cid]
    out = _run(cs, case)
    assert out.attrs["interval"] == "arp_conditional"
    np.testing.assert_allclose(out["M"], case["M"])
    np.testing.assert_allclose(out["ci_lower"], case["lb"], rtol=1e-10, atol=1e-12)
    np.testing.assert_allclose(out["ci_upper"], case["ub"], rtol=1e-10, atol=1e-12)
    scope = sp.validation_scope(out)
    assert scope["outputs"]["diagnostic"]["status"] == "reference", cid


@pytest.mark.parametrize("cid", _CLF)
def test_clf_hybrid_within_one_grid_step_of_honestdid(cs, cid):
    """A stochastic screen (S): both sides simulate the least-favourable
    critical value, each with its own RNG."""
    case = _CASES[cid]
    out = _run(cs, case)
    assert out.attrs["interval"] == "arp_c_lf"
    # HonestDiD's grid: 1,000 points over +/-20 sd of the target.
    ev = sp.event_study_vcov(cs, allow_diagonal=False)
    post = ev.vcov[np.ix_(ev.times >= 0, ev.times >= 0)]
    lv = np.asarray(case["l_vec"], dtype=float)
    step = 40.0 * float(np.sqrt(lv @ post @ lv)) / 999.0
    tol = step * (1 + 1e-9)
    np.testing.assert_allclose(out["ci_lower"], case["lb"], rtol=0, atol=tol)
    np.testing.assert_allclose(out["ci_upper"], case["ub"], rtol=0, atol=tol)
    assert sp.validation_scope(out)["status"] == "stochastic_only"


@pytest.mark.parametrize("cid", _FLCI)
def test_flci_half_length_is_the_programs_optimum(cs, cid):
    case = _CASES[cid]
    out = _run(cs, case)
    assert out.attrs["interval"] == "flci"
    half = (out["ci_upper"] - out["ci_lower"]).to_numpy() / 2
    m = np.asarray(case["M"])
    # M = 0 is the closed form, M > 0 HonestDiD's program at tight tolerance.
    np.testing.assert_allclose(
        half[m == 0], np.asarray(case["half"])[m == 0], rtol=1e-6
    )
    np.testing.assert_allclose(half[m > 0], np.asarray(case["half"])[m > 0], rtol=1e-7)


@pytest.mark.parametrize("cid", _FLCI)
def test_flci_centre_within_the_flat_optimum(cs, cid):
    case = _CASES[cid]
    out = _run(cs, case)
    centre = (out["ci_upper"] + out["ci_lower"]).to_numpy() / 2
    ref_centre = (np.asarray(case["lb"]) + np.asarray(case["ub"])) / 2
    gap = np.abs(centre - ref_centre) / np.asarray(case["half"])
    m = np.asarray(case["M"])
    assert np.all(gap[m > 0] <= 2e-5), (cid, gap)
    # The M = 0 closed form is exact; the native minimum-variance solve is
    # not (open item, see the module docstring).
    assert np.all(gap[m == 0] <= 2e-3), (cid, gap)
    scope = sp.validation_scope(out)
    assert scope["outputs"]["diagnostic"]["status"] == "disclosure", cid


def test_targets_and_alpha_change_the_answer(cs):
    """Not vacuous: each option value moves the interval."""
    base = _run(cs, _CASES["mp_rm_e0"])
    for other in ("mp_rm_avg", "mp_rm_w"):
        assert not np.allclose(base["ci_upper"], _run(cs, _CASES[other])["ci_upper"])
    full = _run(cs, _CASES["mp_rm_avg"])
    windowed = _run(cs, _CASES["mpw_rm_avg"])  # 'average' spans 0..2, not 0..3
    assert not np.allclose(full["ci_upper"], windowed["ci_upper"])
    a05 = sp.honest_did(cs, e=1, method="smoothness", m_grid=[0.01])
    a10 = _run(cs, _CASES["mp_flci_e1_a10"])
    assert a10["ci_upper"].iloc[1] < a05["ci_upper"].iloc[0]
