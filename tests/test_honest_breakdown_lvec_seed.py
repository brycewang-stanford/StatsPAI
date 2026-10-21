"""``breakdown_m(l_vec=, window=)``, the C-LF seed, and honest_did's attrs.

Fast identities on a small event study (three leads, two horizons); the R
comparison on the castle-doctrine data is in
``tests/reference_parity/test_honest_breakdown_lvec_R.py``.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest

import statspai as sp
from statspai.did._arp_sets import rm_confidence_set
from statspai.did.honest_did import event_study_from_moments
from statspai.exceptions import MethodIncompatibility

TIMES = [-4, -3, -2, 0, 1, 2]
B = np.array([0.01, -0.02, 0.005, 0.30, 0.28, 0.35])
_sd = np.array([0.03, 0.03, 0.025, 0.04, 0.05, 0.06])
_rho = 0.3 ** np.abs(np.subtract.outer(np.arange(6), np.arange(6)))
S = _rho * np.outer(_sd, _sd)


def _res(times=TIMES, b=B, s=S):
    return event_study_from_moments(b, s, event_times=times)


def _quiet(fn):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn()


@pytest.mark.parametrize(
    "method, hm",
    [
        ("smoothness", None),
        ("relative_magnitude", None),
        # the Conditional scan tests every grid point at every M: minutes
        pytest.param("relative_magnitude", "Conditional", marks=pytest.mark.slow),
    ],
)
def test_unit_l_vec_equals_single_period(method, hm):
    for e in (1,):
        lv = np.eye(3)[e]
        a = _quiet(
            lambda: sp.breakdown_m(_res(), e=e, method=method, honestdid_method=hm)
        )
        b = _quiet(
            lambda: sp.breakdown_m(_res(), l_vec=lv, method=method, honestdid_method=hm)
        )
        assert a == b


@pytest.mark.parametrize("method", ["smoothness", "relative_magnitude"])
def test_window_equals_the_windowed_event_study(method):
    keep = [1, 2, 3, 4]  # event times -3, -2, 0, 1
    sub = _res([TIMES[i] for i in keep], B[keep], S[np.ix_(keep, keep)])
    a = _quiet(
        lambda: sp.breakdown_m(_res(), window=(-3, 1), l_vec="average", method=method)
    )
    b = _quiet(lambda: sp.breakdown_m(sub, l_vec="average", method=method))
    assert a == b


@pytest.mark.parametrize(
    "method, hm",
    [
        ("smoothness", None),
        ("relative_magnitude", None),
        # the Conditional scan tests every grid point at every M: minutes
        pytest.param("relative_magnitude", "Conditional", marks=pytest.mark.slow),
    ],
)
def test_average_breakdown_is_where_the_set_reaches_zero(method, hm):
    kw = dict(method=method, honestdid_method=hm, l_vec="average")
    m_star = _quiet(lambda: sp.breakdown_m(_res(), **kw))
    assert m_star > 0
    tab = _quiet(
        lambda: sp.honest_did(_res(), m_grid=[0.997 * m_star, 1.003 * m_star], **kw)
    )
    assert list(tab["rejects_zero"]) == [True, False]


def test_clf_is_reproducible_and_uses_the_seed():
    kw = dict(method="relative_magnitude", m_grid=[0.0, 0.5], l_vec="average")
    a = _quiet(lambda: sp.honest_did(_res(), **kw))
    b = _quiet(lambda: sp.honest_did(_res(), **kw))
    assert a.equals(b)
    assert a.attrs["simulated_cv"] is True
    assert a.attrs["seed"] == 0 and a.attrs["lf_draws"] == 1000
    # the table is the grid inversion at that seed
    for seed in (0, 11):
        t = _quiet(lambda: sp.honest_did(_res(), seed=seed, **kw))
        lo, hi, _, _ = rm_confidence_set(
            B[[0, 1, 2, 3, 4, 5]],
            S,
            3,
            3,
            0.5,
            l_vec=np.full(3, 1 / 3),
            seed=seed,
            max_expand=10,
        )
        assert (t.loc[1, "ci_lower"], t.loc[1, "ci_upper"]) == (lo, hi)
        assert t.attrs["seed"] == seed


def test_conditional_and_flci_do_not_simulate():
    cond = _quiet(
        lambda: sp.honest_did(
            _res(),
            method="relative_magnitude",
            honestdid_method="Conditional",
            m_grid=[0.5],
            seed=123,
        )
    )
    base = _quiet(
        lambda: sp.honest_did(
            _res(),
            method="relative_magnitude",
            honestdid_method="Conditional",
            m_grid=[0.5],
        )
    )
    assert cond.equals(base)  # the seed is not used
    assert cond.attrs["simulated_cv"] is False and "seed" not in cond.attrs
    flci = _quiet(lambda: sp.honest_did(_res(), m_grid=[0.0]))
    assert flci.attrs["simulated_cv"] is False


def test_breakdown_in_attrs_and_next_steps():
    res = _res()
    flci = _quiet(lambda: sp.honest_did(res, l_vec="average", m_grid=[0.0]))
    want = _quiet(lambda: sp.breakdown_m(res, l_vec="average"))
    assert flci.attrs["breakdown_m"] == want
    step = flci.attrs["next_steps"][0]
    assert step["action"].startswith("sp.breakdown_m(result, ")
    assert _quiet(lambda: eval(step["action"], {"sp": sp, "result": res})) == want
    # test-inverted sets: not by default (a scan in M), on request
    rm = _quiet(lambda: sp.honest_did(res, method="relative_magnitude", m_grid=[0.0]))
    assert "breakdown_m" not in rm.attrs
    rm_bd = _quiet(
        lambda: sp.honest_did(
            res, method="relative_magnitude", m_grid=[0.0], breakdown=True
        )
    )
    assert rm_bd.attrs["breakdown_m"] == _quiet(
        lambda: sp.breakdown_m(res, method="relative_magnitude")
    )
    off = _quiet(lambda: sp.honest_did(res, m_grid=[0.0], breakdown=False))
    assert "breakdown_m" not in off.attrs


def test_breakdown_true_refused_with_r_backend():
    with pytest.raises(MethodIncompatibility):
        sp.honest_did(_res(), backend="r", breakdown=True)
