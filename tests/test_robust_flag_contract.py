"""Contract: every bool-typed ``robust=`` site rejects the string form.

``robust`` is a boolean on/off switch on eight estimators and a *string*
HC-type selector on the regression family (``robust="HC1"``).
``statspai._house_style.ROBUST_BOOL_HINTS`` names that split the
highest-impact hazard in the signature surface — and until 2026-08 it was a
*silent* hazard on seven of those eight. Only ``sp.did`` rejected the string
form; the rest read it as truthy and returned their default sandwich.

``robust="cluster"`` is the case that costs someone a result. Clustering is
a separate ``cluster=`` argument on all of these estimators, so that call
returned **unclustered** standard errors with no warning — numbers correct
for what was computed, and not what was asked for. Nothing about the output
looks wrong, which is why it survived this long.

This suite pins the guarantee for the whole list at once, driven off
``ROBUST_BOOL_HINTS`` itself, so a ninth bool-typed ``robust=`` site added
later either implements the guard or fails here.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai import _house_style as hs


def _two_period(n: int = 400, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame(
        {
            "i": np.repeat(np.arange(n // 2), 2),
            "t": np.tile([0, 1], n // 2),
            "treat": np.repeat(rng.integers(0, 2, n // 2), 2),
            "sub": np.repeat(rng.integers(0, 2, n // 2), 2),
            "x": rng.normal(size=n),
        }
    )
    df["y"] = 1.0 + 2.0 * df["treat"] * df["t"] + df["x"] + rng.normal(size=n)
    return df


def _panel(n_units: int = 40, n_periods: int = 8, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_units):
        fi = rng.normal()
        for t in range(n_periods):
            x = rng.normal()
            rows.append(
                {
                    "id": i,
                    "time": t,
                    "x": x,
                    "y": 1.0 + 0.5 * x + fi * rng.normal() + rng.normal(0, 0.3),
                }
            )
    return pd.DataFrame(rows)


def _choices(n_chid: int = 200, n_alt: int = 3, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for c in range(n_chid):
        chosen = int(np.argmax(rng.normal(size=n_alt)))
        for a in range(n_alt):
            rows.append({"chid": c, "alt": a, "y": int(a == chosen), "z": rng.normal()})
    return pd.DataFrame(rows)


def _dynpanel(n_units: int = 80, n_periods: int = 8, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    rows = []
    for i in range(n_units):
        y = rng.normal()
        for t in range(n_periods):
            y = 0.5 * y + rng.normal()
            rows.append({"id": i, "year": 1976 + t, "n": y})
    return pd.DataFrame(rows)


#: One invocation per bool-typed ``robust=`` estimator. Keyed by the name in
#: ``ROBUST_BOOL_HINTS`` so the coverage assertion below can be exhaustive.
CALLS = {
    "did": lambda f, v: f(_two_period(), y="y", treat="treat", time="t", robust=v),
    "ddd": lambda f, v: f(
        _two_period(), y="y", treat="treat", time="t", subgroup="sub", robust=v
    ),
    "did_2x2": lambda f, v: f(_two_period(), y="y", treat="treat", time="t", robust=v),
    "did_analysis": lambda f, v: f(
        _two_period(), y="y", treat="treat", time="t", robust=v
    ),
    "interactive_fe": lambda f, v: f(
        _panel(), y="y", x=["x"], id="id", time="time", robust=v
    ),
    "mixlogit": lambda f, v: f(
        _choices(), y="y", chid="chid", x_random=["z"], n_draws=50, robust=v
    ),
    "xtabond": lambda f, v: f(
        _dynpanel(), y="n", id="id", time="year", lags=1, robust=v
    ),
    "xtdpdsys": lambda f, v: f(
        _dynpanel(), y="n", id="id", time="year", lags=1, robust=v
    ),
}


def test_every_bool_robust_site_is_covered_here():
    """If someone adds a ninth entry, this suite must grow with it."""
    assert set(CALLS) == set(hs.ROBUST_BOOL_HINTS), (
        "ROBUST_BOOL_HINTS and this suite have diverged; a bool-typed "
        "`robust=` site is either untested or no longer registered."
    )


@pytest.mark.parametrize("fn", sorted(CALLS))
@pytest.mark.parametrize("bad", ["HC1", "HC3", "cluster", "robust", 1, 0, None])
def test_string_robust_is_rejected(fn, bad):
    with pytest.raises(Exception, match="(?i)boolean"):
        CALLS[fn](getattr(sp, fn), bad)


@pytest.mark.parametrize("fn", sorted(CALLS))
def test_cluster_string_names_the_real_alternative(fn):
    """The message must point at ``cluster=``, since that is where it lives."""
    with pytest.raises(Exception) as exc:
        CALLS[fn](getattr(sp, fn), "cluster")
    assert "cluster=" in str(exc.value)


@pytest.mark.parametrize("fn", sorted(CALLS))
@pytest.mark.parametrize("good", [True, False, np.True_, np.False_])
def test_booleans_still_accepted(fn, good):
    """The guard must not reject what the signature promises."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        CALLS[fn](getattr(sp, fn), good)


def test_single_implementation():
    """§4: one implementation, aliased — not five copies."""
    import importlib

    core = importlib.import_module("statspai.core._vcov")
    did_core = importlib.import_module("statspai.did._core")
    ab = importlib.import_module("statspai.gmm.arellano_bond")

    assert did_core.require_bool is core.require_bool_flag
    assert ab._require_bool is core.require_bool_flag
