"""``tidy()`` / ``glance()`` on every result class, and dict-returning
functions turned into result objects (2026-10-19 audit).

* The ~360 classes built on ``ResultProtocolMixin`` had no ``tidy`` /
  ``glance``; they now inherit broom-style defaults.
* ``sp.bacon_decomposition`` / ``sp.evalue`` / ``sp.oster_bounds`` /
  ``sp.mi_estimate`` returned bare dicts. They return dict subclasses: the
  same keys and dict behaviour, plus ``summary()`` and the exports, and
  ``sp.regtable`` accepts the ``mi_estimate`` result.
"""

from __future__ import annotations

import importlib
import json
import math
import pickle
import pkgutil
import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai._result_serialize import ResultProtocolMixin
from statspai._result_tidy import TIDY_COLUMNS


def _quiet(fn, *a, **k):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*a, **k)


def _all_subclasses(cls):
    out = set()
    for sub in cls.__subclasses__():
        out |= {sub} | _all_subclasses(sub)
    return out


def test_every_protocol_class_has_tidy_and_glance():
    for mod in pkgutil.walk_packages(sp.__path__, "statspai."):
        if any(s in mod.name for s in ("torch", "jax", "pymc", "_skill", "mcp")):
            continue
        try:
            importlib.import_module(mod.name)
        except Exception:  # noqa: BLE001 - optional extras may be missing
            continue
    classes = _all_subclasses(ResultProtocolMixin)
    assert len(classes) >= 300
    missing = [
        c.__name__
        for c in classes
        if not (
            callable(getattr(c, "tidy", None)) and callable(getattr(c, "glance", None))
        )
    ]
    assert missing == []


@pytest.fixture(scope="module")
def cross():
    r = np.random.default_rng(0)
    n = 300
    df = pd.DataFrame({"x": r.normal(size=n), "g": r.integers(0, 20, n)})
    df["y"] = 1 + 2 * df.x + 0.1 * df.g + r.normal(size=n)
    df["dur"] = r.exponential(1.0, n)
    df["ev"] = (r.random(n) < 0.8).astype(int)
    return df


def test_vector_tidy_is_the_coefficient_table(cross):
    res = _quiet(sp.hdfe_ols, "y ~ x | g", data=cross)
    t = res.tidy()
    assert list(t.columns) == TIDY_COLUMNS
    assert t["estimate"].tolist() == res.params.tolist()
    assert t["std_error"].tolist() == res.std_errors.tolist()
    assert t["p_value"].tolist() == res.pvalues.tolist()
    t90 = res.tidy(conf_level=0.90)
    z = 1.6448536269514722
    np.testing.assert_allclose(
        t90["conf_high"] - t90["estimate"], z * res.std_errors.to_numpy(), rtol=1e-12
    )


def test_glance_is_one_row_of_scalars(cross):
    res = _quiet(sp.hdfe_ols, "y ~ x | g", data=cross)
    g = res.glance()
    assert len(g) == 1
    assert g["n_obs"].iloc[0] == len(cross)
    assert g["r2"].iloc[0] == pytest.approx(res.r2, abs=0)
    assert not any(str(c).startswith("_") for c in g.columns)


def test_tidy_refuses_without_an_estimate(cross):
    km = _quiet(sp.kaplan_meier, cross, duration="dur", event="ev")
    with pytest.raises(sp.MethodIncompatibility) as info:
        km.tidy()
    assert "glance" in info.value.recovery_hint
    assert len(km.glance()) == 1


def test_tidy_rejects_bad_conf_level(cross):
    res = _quiet(sp.hdfe_ols, "y ~ x | g", data=cross)
    with pytest.raises(sp.MethodIncompatibility):
        res.tidy(conf_level=95)


# ---------------------------------------------------------------- dict results


@pytest.fixture(scope="module")
def bacon():
    r = np.random.default_rng(3)
    rows = []
    cohort = {i: (4 if i < 15 else 6 if i < 30 else 0) for i in range(45)}
    for i in range(45):
        for t in range(1, 9):
            d = int(cohort[i] > 0 and t >= cohort[i])
            rows.append(dict(id=i, t=t, d=d, y=0.1 * i + 0.3 * t + 2 * d + r.normal()))
    df = pd.DataFrame(rows)
    return _quiet(sp.bacon_decomposition, df, y="y", treat="d", time="t", id="id")


def test_bacon_result_is_still_the_dict(bacon):
    assert isinstance(bacon, dict)
    assert set(bacon) >= {"beta_twfe", "decomposition", "summary", "weighted_sum"}
    assert bacon["beta_twfe"] == bacon.beta_twfe == bacon.estimate
    # The decomposition theorem: the weighted 2x2 estimates sum to TWFE.
    assert bacon["weighted_sum"] == pytest.approx(bacon["beta_twfe"], rel=1e-10)
    assert isinstance(bacon["summary"], pd.DataFrame)
    text = bacon.summary()
    assert f"{bacon['beta_twfe']:.6g}" in text
    assert bacon.cite() == "goodmanbacon2021difference"
    clone = pickle.loads(pickle.dumps(bacon))
    assert clone["beta_twfe"] == bacon["beta_twfe"]
    assert bacon.tidy()["estimate"].iloc[0] == bacon["beta_twfe"]
    assert "\\begin{tabular}" in bacon.to_latex()


def test_evalue_result_matches_closed_form():
    res = sp.evalue(1.5, ci=(1.2, 1.9))
    assert isinstance(res, dict)
    # VanderWeele-Ding: RR + sqrt(RR (RR - 1)) for RR >= 1.
    assert res["evalue_estimate"] == pytest.approx(
        1.5 + math.sqrt(1.5 * 0.5), rel=1e-12
    )
    assert res["evalue_ci"] == pytest.approx(1.2 + math.sqrt(1.2 * 0.2), rel=1e-12)
    json.dumps(res)
    assert f"{res['evalue_estimate']:.6g}" in res.summary()
    assert res.glance()["evalue_estimate"].iloc[0] == res["evalue_estimate"]
    with pytest.raises(sp.MethodIncompatibility):
        sp.regtable(res)


def test_oster_result_keeps_keys():
    res = sp.oster_bounds(
        beta_short=1.0, r2_short=0.1, beta_long=0.8, r2_long=0.3, r_max=0.5
    )
    assert isinstance(res, dict)
    # beta* = beta_long - delta (beta_short - beta_long) (r_max - r2_long) / (r2_long - r2_short)
    assert res["beta_adjusted"] == pytest.approx(0.8 - 1.0 * 0.2 * 0.2 / 0.2, rel=1e-12)
    assert "beta_adjusted" in res.summary()


def test_mi_estimate_goes_into_regtable():
    rng = np.random.default_rng(7)
    x1 = rng.normal(size=120)
    x2 = 0.5 * x1 + rng.normal(size=120)
    df = pd.DataFrame({"y": 1 + x1 + x2 + rng.normal(size=120), "x1": x1, "x2": x2})
    df.loc[rng.choice(120, size=15, replace=False), "x2"] = np.nan
    res = _quiet(
        sp.mi_estimate, sp.mice(df, m=3, seed=0), sp.regress, formula="y ~ x1 + x2"
    )
    assert isinstance(res, dict) and isinstance(res["params"], np.ndarray)
    assert res.params.index.tolist() == ["Intercept", "x1", "x2"]
    table = _quiet(sp.regtable, res).to_dict()["models"][0]
    for term, value in zip(res["var_names"], res["params"]):
        assert table["coefficients"][term]["estimate"] == value
    for term, value in zip(res["var_names"], res["se"]):
        assert table["coefficients"][term]["std_error"] == value
    assert table["stats"]["N"] == 120
    tidy = res.tidy()
    assert tidy["conf_low"].tolist() == list(res["ci_lower"])
