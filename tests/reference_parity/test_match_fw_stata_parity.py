"""Cross-language parity: ``sp.match`` / ``sp.psm`` frequency weights vs Stata.

Fixture: ``_fixtures/_generate_match_fw_stata.do`` (Stata 18 ``teffects
psmatch ... [fw=fw], atet nneighbor(1)`` and the same fit on the expanded
data). ``teffects`` matching accepts frequency weights only -- ``[pw=]``,
``[iw=]``, ``[aw=]`` give rc 101 and ``vce(cluster)`` rc 198 -- and Stata
reports the ``[fw=]`` fit and the expanded fit identically, so
``sp.match(weights=)`` is frequency weights, computed by expansion. The SE
row uses ``se_method='abadie_imbens_2016'``, teffects' estimated-score
variance, as Track A module 11 does (same ~2e-7 gap there).
"""

from __future__ import annotations

import json
import pathlib
import warnings
from functools import lru_cache
from typing import Any, Dict

import numpy as np
import pandas as pd
import pytest

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import statspai as sp

from statspai.exceptions import MethodIncompatibility

_FIX = pathlib.Path(__file__).parent / "_fixtures"
RTOL = 1e-6  # CLAUDE.md §5.1 default budget
X = ["age", "education", "black", "hispanic", "married", "re74", "re75"]


@lru_cache(maxsize=None)
def _ref() -> Dict[str, Any]:
    path = _FIX / "match_fw_stata.json"
    if not path.exists():  # pragma: no cover
        pytest.skip("run _generate_match_fw_stata.do first")
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=None)
def _data() -> pd.DataFrame:
    return pd.read_csv(_FIX / "match_fw_data.csv")


def _psm(data: pd.DataFrame, **extra: Any):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.psm(
            data,
            y="re78",
            d="treat",
            X=X,
            method="nn",
            se_method="abadie_imbens_2016",
            **extra,
        )


def test_frequency_weights_match_teffects_psmatch() -> None:
    ref = _ref()["psmatch_atet_fw"]
    res = _psm(_data(), weights="fw")
    np.testing.assert_allclose(res.estimate, ref["est"], rtol=RTOL)
    np.testing.assert_allclose(res.se, ref["se"], rtol=RTOL)
    assert res.n_obs == int(ref["N"])
    assert res.model_info["frequency_weights"] == "fw"


def test_stata_fw_is_the_expanded_fit_and_so_is_ours() -> None:
    ref = _ref()
    np.testing.assert_allclose(
        ref["psmatch_atet_fw"]["est"], ref["psmatch_atet_expanded"]["est"], rtol=1e-12
    )
    df = _data()
    expanded = df.loc[df.index.repeat(df["fw"])].reset_index(drop=True)
    a, b = _psm(df, weights="fw"), _psm(expanded)
    assert a.estimate == b.estimate and a.se == b.se


def test_dispatcher_routes_weights() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = sp.match(_data(), y="re78", treat="treat", covariates=X, weights="fw")
    assert r.model_info["frequency_weights"] == "fw"


def test_sampling_weights_and_clusters_are_refused() -> None:
    df = _data().assign(pw=lambda d: d["fw"] + 0.5, g=lambda d: np.arange(len(d)) % 20)
    with pytest.raises(MethodIncompatibility, match="frequency weights"):
        _psm(df, weights="pw")
    with pytest.raises(MethodIncompatibility, match="cluster-robust matching"):
        _psm(df, cluster="g")
