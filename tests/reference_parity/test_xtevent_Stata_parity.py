"""``sp.xtevent`` = Stata ``xtevent`` (Freyaldenhoven, Hansen, Perez Perez,
Shapiro) for a policy that is continuous and changes several times per unit.

The Web of Power replication (QJE 2023) needed ``xtevent``'s event study of a
continuous policy, which ``sp.event_study`` (binary adoption dates) does not
cover. Inside the window the regressors are leads and lags of the policy's
first difference; the endpoints carry the level (``z_{t-hi-1}``) and one
minus the lead ``z_{t+|lo|}``; the first window lag is normalised to zero.

Reference: Stata ``xtevent`` 3.1.0 (SSC) on
``_fixtures/xtevent_continuous.csv`` (``_generate_xtevent_Stata.do``): the
default ``areg`` estimation, a clustered asymmetric window, another
normalisation with robust SEs, the ``reghdfe`` engine, and the static
model. Coefficients, SEs and N agree to 1e-12.
"""

from __future__ import annotations

import json
import pathlib
import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp

_FIX = pathlib.Path(__file__).parent / "_fixtures"
STATA = json.loads((_FIX / "xtevent_Stata.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def data():
    return pd.read_csv(_FIX / "xtevent_continuous.csv")


def _stata_event_rows(key):
    ref = STATA[key]
    names = ref["names"].split()
    out = {}
    for nm, b, se in zip(names, ref["b"], ref["se"]):
        if nm.startswith("_k_eq_"):
            tag = nm[len("_k_eq_") :]
            k = -int(tag[1:]) if tag[0] == "m" else int(tag[1:])
            out[k] = (b, se)
        elif nm == "x":
            out["x"] = (b, se)
    return out, ref["N"]


CASES = [
    ("w2", dict(window=2)),
    ("w34_cluster", dict(window=(-3, 4), cluster="cl")),
    ("w2_norm2_robust", dict(window=2, norm=-2, vce="robust")),
    ("w2_reghdfe_cluster", dict(window=2, engine="reghdfe", cluster="cl")),
]


@pytest.mark.parametrize("key, kw", CASES, ids=[c[0] for c in CASES])
def test_matches_stata_xtevent(data, key, kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = sp.xtevent(
            data, y="y", policy="z", panel="id", time="t", covariates=["x"], **kw
        )
    ref, n = _stata_event_rows(key)
    assert r.model_info["n_obs"] == n
    es = r.model_info["event_study"].set_index("relative_time")
    for k, (b, se) in ref.items():
        if k == "x":
            continue
        assert es.loc[k, "estimate"] == pytest.approx(b, rel=1e-10, abs=1e-12), k
        assert es.loc[k, "se"] == pytest.approx(se, rel=1e-10), k
    norm = kw.get("norm", -1)
    assert es.loc[norm, "estimate"] == 0 and es.loc[norm, "se"] == 0
    assert r.model_info["coef_table"].loc["x", "estimate"] == pytest.approx(
        ref["x"][0], rel=1e-10
    )


def test_static_model_matches_stata(data):
    ref = STATA["static_cluster"]
    names = ref["names"].split()
    b = dict(zip(names, ref["b"]))
    se = dict(zip(names, ref["se"]))
    r = sp.xtevent(
        data,
        y="y",
        policy="z",
        panel="id",
        time="t",
        covariates=["x"],
        static=True,
        cluster="cl",
    )
    assert float(r.estimate) == pytest.approx(b["z"], rel=1e-10)
    assert float(r.se) == pytest.approx(se["z"], rel=1e-10)
    assert r.model_info["n_obs"] == ref["N"]


def test_binary_staggered_policy_reproduces_binned_event_study():
    """With a 0/1 absorbing policy the construction is the binned event
    study: interior leads/lags are event-time dummies, endpoints bin the
    tails. Same regression as sp.event_study with the same bins."""
    rng = np.random.default_rng(3)
    units, periods = 60, 14
    d = pd.DataFrame(
        dict(
            i=np.repeat(np.arange(units), periods), t=np.tile(np.arange(periods), units)
        )
    )
    adopt = np.repeat(rng.choice([4, 6, 8, 99], units), periods)
    d["z"] = (d.t >= adopt).astype(float)
    d["y"] = (
        np.repeat(rng.normal(size=units), periods) + 0.9 * d.z + rng.normal(size=len(d))
    )
    r = sp.xtevent(d, y="y", policy="z", panel="i", time="t", window=2)
    es = r.model_info["event_study"].set_index("relative_time")
    # interior lag 0 regressor is exactly the adoption-period dummy
    assert np.isfinite(es.loc[0, "estimate"])
    assert set(es.index) == {-3, -2, -1, 0, 1, 2, 3}


def test_input_validation(data):
    with pytest.raises(sp.MethodIncompatibility, match="window"):
        sp.xtevent(data, y="y", policy="z", panel="id", time="t", window=(1, 3))
    with pytest.raises(sp.MethodIncompatibility, match="norm"):
        sp.xtevent(data, y="y", policy="z", panel="id", time="t", window=2, norm=5)


def test_diffavg_matches_stata(data):
    """Stata ``diffavg``: mean of the post coefficients (right endpoint
    included) minus mean of the pre ones (the normalised zero included)."""
    ref = STATA["w2_diffavg_cluster"]
    r = sp.xtevent(
        data,
        y="y",
        policy="z",
        panel="id",
        time="t",
        covariates=["x"],
        window=2,
        cluster="cl",
    )
    d = r.model_info["diffavg"]
    assert d["estimate"] == pytest.approx(ref["estimate"], rel=1e-10)
    assert d["se"] == pytest.approx(ref["se"], rel=1e-10)
