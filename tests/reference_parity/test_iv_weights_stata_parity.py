"""Cross-language parity: weighted IV (``sp.iv(weights=)``) against Stata.

Fixture: ``_fixtures/_generate_iv_weights_stata.do`` (Stata 18,
``ivregress 2sls|liml ... [aw=w], small`` under the default, ``vce(robust)``
and ``vce(cluster g)`` standard errors).  Both sides read the same CSV bytes;
twenty outcomes are missing, so the weight column must stay aligned with the
rows the IV path keeps.

Until 1.32 ``sp.iv`` / ``sp.ivreg`` accepted ``weights=`` and dropped it on
the floor, returning the unweighted estimate; ``alpha=`` was dropped the same
way.  The data make the effect of ``d`` vary with ``w`` so the weighted and
unweighted estimands differ by far more than any tolerance.
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
# CLAUDE.md §5.1 default budget; observed agreement is ~1e-14.
RTOL = 1e-6
FORMULA = "y ~ x1 + (d ~ z1 + z2)"
_VCE = {
    "unadjusted": {},
    "robust": {"vce": "robust"},
    "cluster": {"cluster": "g"},
}


@lru_cache(maxsize=None)
def _ref() -> Dict[str, Any]:
    path = _FIX / "iv_weights_stata.json"
    if not path.exists():  # pragma: no cover
        pytest.skip("run _generate_iv_weights_stata.do first")
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=None)
def _data() -> pd.DataFrame:
    path = _FIX / "iv_weights_data.csv"
    if not path.exists():  # pragma: no cover
        pytest.skip("run _generate_iv_weights_stata.do first")
    return pd.read_csv(path)


def _fit(method: str, vce: str, **extra: Any):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.iv(
            FORMULA, data=_data(), method=method, weights="w", **_VCE[vce], **extra
        )


@pytest.mark.parametrize("method", ["2sls", "liml"])
@pytest.mark.parametrize("vce", list(_VCE))
def test_weighted_iv_matches_stata(method: str, vce: str) -> None:
    ref = _ref()[f"{method}_{vce}"]
    res = _fit(method, vce)
    for term, name in (("d", "d"), ("x1", "x1"), ("_cons", "Intercept")):
        np.testing.assert_allclose(res.params[name], ref[f"b_{term}"], rtol=RTOL)
        np.testing.assert_allclose(res.std_errors[name], ref[f"se_{term}"], rtol=RTOL)
    np.testing.assert_allclose(res.diagnostics["R-squared"], ref["r2"], rtol=RTOL)
    assert res.data_info["nobs"] == int(ref["N"])
    assert res.model_info["weight_type"] == "aweight"


def test_weights_move_the_estimate() -> None:
    """Regression guard: the weight must not be silently ignored."""
    ref = _ref()
    unweighted = ref["2sls_unweighted"]["b_d"]
    weighted = ref["2sls_unadjusted"]["b_d"]
    assert abs(weighted - unweighted) > 0.05
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        plain = sp.iv(FORMULA, data=_data())
    np.testing.assert_allclose(plain.params["d"], unweighted, rtol=RTOL)
    np.testing.assert_allclose(
        _fit("2sls", "unadjusted").params["d"], weighted, rtol=RTOL
    )


def test_ivreg_array_weights_equal_column_weights() -> None:
    df = _data()
    kept = df.dropna(subset=["y"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        by_array = sp.ivreg(FORMULA, data=kept, weights=kept["w"].to_numpy())
    np.testing.assert_allclose(
        by_array.params, _fit("2sls", "unadjusted").params, rtol=1e-12
    )


def test_weights_are_scale_invariant() -> None:
    df = _data().assign(w10=lambda d: 10 * d["w"])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = sp.iv(FORMULA, data=df, weights="w")
        b = sp.iv(FORMULA, data=df, weights="w10")
    np.testing.assert_allclose(a.params, b.params, rtol=1e-12)
    np.testing.assert_allclose(a.std_errors, b.std_errors, rtol=1e-12)


def test_alpha_sets_the_reported_interval() -> None:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sp.iv(FORMULA, data=_data(), alpha=0.10)
    assert res.alpha == 0.10
    lo, hi = res.conf_int().loc["d"]
    from scipy import stats

    t = stats.t.ppf(0.95, res.data_info["df_resid"])
    np.testing.assert_allclose(hi - lo, 2 * t * res.std_errors["d"], rtol=1e-12)
    assert res.tidy().set_index("term").loc["d", "conf_high"] == pytest.approx(hi)


@pytest.mark.parametrize("bad", [{"zzz": 1}, {"small": True}, {"iv_diag": True}])
def test_unknown_options_are_rejected(bad: Dict[str, Any]) -> None:
    with pytest.raises(TypeError, match="unexpected keyword"):
        sp.iv(FORMULA, data=_data(), **bad)
    with pytest.raises(TypeError, match="unexpected keyword"):
        sp.ivreg(FORMULA, data=_data(), **bad)


def test_bad_weights_fail_loudly() -> None:
    df = _data().assign(w0=lambda d: d["w"].where(d["id"] != 3, 0.0))
    with pytest.raises(MethodIncompatibility, match="strictly positive"):
        sp.iv(FORMULA, data=df, weights="w0")
    with pytest.raises(MethodIncompatibility, match="not a column"):
        sp.iv(FORMULA, data=df, weights="nope")


def test_unsupported_weighted_paths_refuse() -> None:
    df = _data()
    with pytest.raises(MethodIncompatibility, match="absorb"):
        sp.iv(FORMULA, data=df, weights="w", absorb="g")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(MethodIncompatibility, match="weighted"):
            sp.ivreg(FORMULA, data=df, weights="w", vce="cr2", cluster="g")
