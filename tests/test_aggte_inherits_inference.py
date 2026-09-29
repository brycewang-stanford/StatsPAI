"""``sp.aggte`` inherits the fit's inference settings (R ``bstrap = NULL``).

The default used to be an unseeded bootstrap, so ``sp.aggte(fit)`` gave a
different SE on every call, and one that disagreed with the analytic SE of
the ``callaway_santanna`` fit it aggregated (found by a replication that
re-ran its notebook and saw the z statistics move).
"""

import numpy as np
import pytest

import statspai as sp


@pytest.fixture(scope="module")
def fit():
    df = sp.dgp_did(n_units=200, n_periods=8, staggered=True, seed=42)
    df["first_treat"] = df["first_treat"].fillna(0)
    return sp.callaway_santanna(df, y="y", g="first_treat", t="time", i="unit")


def test_default_is_deterministic_and_analytic(fit):
    ses = [sp.aggte(fit, type="group").se for _ in range(3)]
    assert ses[0] == ses[1] == ses[2]
    analytic = sp.aggte(fit, type="group", bstrap=False, cband=False)
    assert ses[0] == analytic.se
    assert sp.aggte(fit, type="group").model_info["bstrap"] is False


def test_unseeded_bootstrap_records_a_reproducible_seed(fit):
    a = sp.aggte(fit, type="dynamic", bstrap=True)
    seed = a.model_info["random_state"]
    assert isinstance(seed, int)
    b = sp.aggte(fit, type="dynamic", bstrap=True, random_state=seed)
    assert a.se == b.se
    np.testing.assert_array_equal(a.detail["se"], b.detail["se"])


def test_cband_or_n_boot_requests_the_bootstrap(fit):
    assert sp.aggte(fit, type="dynamic", cband=True, random_state=1).model_info[
        "bstrap"
    ]
    r = sp.aggte(fit, type="dynamic", n_boot=99, random_state=1)
    assert r.model_info["bstrap"] and r.model_info["n_boot"] == 99


def test_bootstrapped_fit_is_inherited():
    df = sp.dgp_did(n_units=150, n_periods=6, staggered=True, seed=3)
    df["first_treat"] = df["first_treat"].fillna(0)
    f = sp.callaway_santanna(
        df,
        y="y",
        g="first_treat",
        t="time",
        i="unit",
        bstrap=True,
        biters=199,
        random_state=0,
    )
    r = sp.aggte(f, type="group", random_state=0)
    assert r.model_info["bstrap"] and r.model_info["n_boot"] == 199
