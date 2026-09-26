"""Every legacy column-name spelling also accepts its house-style name.

``sp.synth_power(unit=...)`` and ``sp.panel(entity=...)`` must also take
``id=``; ``outcome=`` must also take ``y=``, and so on. The signature
lint (``scripts/signature_house_style.py``) is a ratchet on *spellings*;
this test pins that a caller using the canonical grammar is never
rejected.
"""

import inspect

import pytest

import statspai as sp
from statspai import _house_style as hs

# Legacy spellings whose meaning is a column name (or a DataFrame, for
# ``df``), so the canonical spelling is a pure rename.
_COLUMN_SPELLINGS = {
    "unit",
    "entity",
    "i",
    "panel_id",
    "outcome",
    "treatment",
    "controls",
    "covs",
    "t",
    "cluster_var",
    "sample_weight",
    "time_col",
    "df",
    "weight",
}

# (function, parameter) pairs where the legacy spelling means something else.
_FALSE_FRIENDS = {
    ("esttab", "t"): "t toggles t-statistics in the table",
    ("distance_to_feature", "unit"): "unit is the output unit, 'km' or 'm'",
    ("line_length_in_polygon", "unit"): "unit is the output unit, 'km' or 'm'",
    ("dyadic_regression", "i"): "i is the first node of a dyad, not a panel id",
}


def _gaps():
    idx = hs.alias_index()
    for name in sp.list_functions():
        f = getattr(sp, name, None)
        if not callable(f) or inspect.isclass(f):
            continue
        try:
            params = inspect.signature(f).parameters
        except (TypeError, ValueError):
            continue
        aliases = getattr(f, "__statspai_aliases__", {}) or {}
        for p in params:
            canon = idx.get(p)
            if p not in _COLUMN_SPELLINGS or not canon or canon in params:
                continue
            if (name, p) in _FALSE_FRIENDS:
                continue
            if aliases.get(canon) != p:
                yield f"{name}({p}=) does not accept {canon}="


def test_every_legacy_column_spelling_accepts_the_canonical_name():
    gaps = sorted(_gaps())
    assert not gaps, "\n".join(gaps)


def test_canonical_and_legacy_give_the_same_fit():
    import numpy as np
    import pandas as pd

    rng = np.random.default_rng(0)
    n_units, n_t = 60, 5
    ids = np.repeat(np.arange(n_units), n_t)
    df = pd.DataFrame(
        {
            "id": ids,
            "time": np.tile(np.arange(n_t), n_units),
            "x": rng.normal(size=ids.size),
            "d": rng.integers(0, 2, ids.size),
        }
    )
    df["y"] = df.d + df.x + rng.normal(size=ids.size)
    a = sp.panel(df, "y ~ d + x", entity="id", time="time")
    b = sp.panel(df, "y ~ d + x", id="id", time="time")
    assert a.params["d"] == b.params["d"]
    with pytest.raises(TypeError, match="both"):
        sp.panel(df, "y ~ d + x", id="id", entity="id", time="time")
