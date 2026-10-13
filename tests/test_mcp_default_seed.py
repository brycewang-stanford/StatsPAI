"""Tool calls that leave a seed unset are filled and say so.

Half of the seed-taking estimators default to ``seed=None``. Through the tool
layer that made a repeated call return different numbers with nothing in the
response to reproduce the first one. The dispatcher now fills the argument,
reports it, and leaves the library default alone.
"""

from __future__ import annotations

import inspect
import json

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.agent._default_seed import (
    DEFAULT_SEED,
    DEFAULT_SEED_ENV,
    default_seed,
    inject_default_seed,
)
from statspai.agent.tools import execute_tool


@pytest.fixture(scope="module")
def frame():
    rng = np.random.default_rng(7)
    n = 300
    x = rng.normal(size=n)
    return pd.DataFrame({"y": 1.0 + 0.5 * x + rng.normal(size=n), "x": x})


def _boot(frame, **extra):
    args = {"formula": "y ~ x", "draws": 200, **extra}
    spec = inspect.signature(sp.bayes_bootstrap).parameters
    assert spec["seed"].default is None  # the premise of this file
    return execute_tool("bayes_bootstrap", args, data=frame, detail="agent")


def _numbers(out):
    """The payload without the fields that describe the call itself."""
    kept = {
        k: v
        for k, v in out.items()
        if k not in ("seed_injected", "replay", "result_card", "provenance")
    }
    return json.dumps(kept, default=str, sort_keys=True)


def test_unset_seed_is_filled_and_reported(frame):
    first = _boot(frame)
    second = _boot(frame)
    assert "error" not in first, first
    assert first["seed_injected"]["param"] == "seed"
    assert first["seed_injected"]["value"] == DEFAULT_SEED
    # The point of the fill: the same call returns the same numbers.
    assert _numbers(first) == _numbers(second)
    # ... and they are the numbers of the direct call with that seed.
    direct = sp.bayes_bootstrap("y ~ x", frame, draws=200, seed=DEFAULT_SEED)
    assert first["posterior"]["x"]["mean"] == pytest.approx(
        float(direct.params["x"]), rel=1e-12
    )
    assert f"seed={DEFAULT_SEED}" in str(first["replay"]).replace(" ", "")


def test_an_explicit_seed_is_left_alone(frame):
    out = _boot(frame, seed=5)
    assert "seed_injected" not in out
    assert _numbers(out) != _numbers(_boot(frame))


def test_an_explicit_null_is_respected():
    args = {"seed": None}
    assert inject_default_seed(sp.bayes_bootstrap, args) is None
    assert args == {"seed": None}


@pytest.mark.parametrize("value", ["off", "none", "OFF", "", "not-a-number"])
def test_environment_switch_turns_it_off(monkeypatch, value):
    monkeypatch.setenv(DEFAULT_SEED_ENV, value)
    assert default_seed() is None
    args: dict = {}
    assert inject_default_seed(sp.bayes_bootstrap, args) is None
    assert args == {}


def test_environment_sets_the_value(monkeypatch):
    monkeypatch.setenv(DEFAULT_SEED_ENV, "2026")
    args: dict = {}
    note = inject_default_seed(sp.bayes_bootstrap, args)
    assert args == {"seed": 2026} and note["value"] == 2026


def test_seed_defaults_other_than_none_are_not_touched():
    def takes_seed(data, seed=0):
        return data

    args: dict = {}
    assert inject_default_seed(takes_seed, args) is None
    assert args == {}


def test_the_fill_does_not_move_a_deterministic_estimate(monkeypatch):
    """Analytic paths must not depend on whether the seed was filled."""
    df = sp.datasets.mpdta()
    args = {"y": "lemp", "g": "first_treat", "t": "year", "i": "countyreal"}
    on = execute_tool("callaway_santanna", dict(args), data=df, detail="minimal")
    monkeypatch.setenv(DEFAULT_SEED_ENV, "off")
    off = execute_tool("callaway_santanna", dict(args), data=df, detail="minimal")
    assert "seed_injected" in on and "seed_injected" not in off
    assert on["estimate"] == off["estimate"]
    assert on["se"] == off["se"]


def test_library_default_is_unchanged(frame):
    a = sp.bayes_bootstrap("y ~ x", frame, draws=200)
    b = sp.bayes_bootstrap("y ~ x", frame, draws=200)
    assert not np.allclose(np.asarray(a.params), np.asarray(b.params))
