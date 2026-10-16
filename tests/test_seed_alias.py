"""``seed=`` is accepted wherever a public function takes a random seed.

``seed`` is the house-style spelling (``_house_style.THEMES['seed']``); the
functions that kept scikit-learn's ``random_state`` accept it through
``@accepts_aliases``. Passing both is ambiguous and must raise before any
work is done.
"""

import inspect
import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp


def _seed_aliased():
    out = []
    for name in sp.list_functions():
        obj = getattr(sp, name, None)
        if obj is None or inspect.isclass(obj) or not callable(obj):
            continue
        target = (getattr(obj, "__statspai_aliases__", {}) or {}).get("seed")
        if target:
            out.append((name, target))
    return out


ALIASED = _seed_aliased()


def test_every_seeded_function_accepts_seed():
    missing = []
    for name in sp.list_functions():
        obj = getattr(sp, name, None)
        if obj is None or inspect.isclass(obj) or not callable(obj):
            continue
        try:
            params = inspect.signature(obj).parameters
        except (TypeError, ValueError):
            continue
        aliases = getattr(obj, "__statspai_aliases__", {}) or {}
        if ("random_state" in params or "rng_seed" in params) and not (
            "seed" in params or "seed" in aliases
        ):
            missing.append(name)
    assert missing == []
    assert len(ALIASED) >= 89


@pytest.mark.parametrize("name,target", ALIASED)
def test_seed_and_its_target_together_raise(name, target):
    with pytest.raises(TypeError, match="pass only one"):
        getattr(sp, name)(seed=1, **{target: 1})


def _quiet(fn, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(**kw)


def test_seed_reaches_random_state_in_a_bootstrap():
    rng = np.random.default_rng(0)
    n = 400
    d = rng.integers(0, 2, n)
    s = (rng.uniform(size=n) < 0.6 + 0.2 * d).astype(int)
    y = np.where(s == 1, 1 + d + rng.normal(size=n), np.nan)
    df = pd.DataFrame({"y": y, "d": d, "s": s})
    kw = dict(data=df, y="y", treat="d", selection="s", n_bootstrap=30)
    a = _quiet(sp.lee_bounds, seed=5, **kw)
    b = _quiet(sp.lee_bounds, random_state=5, **kw)
    c = _quiet(sp.lee_bounds, random_state=6, **kw)
    assert a.se == b.se
    assert a.se != c.se
