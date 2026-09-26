"""Cross-language parity: ``sp.panel(..., ssc=)`` vs Stata 18 and R fixest.

Fixtures: ``_fixtures/_generate_panel_ssc_stata.do`` (official ``xtreg`` /
``areg`` / ``regress``) and ``_generate_panel_ssc_R.R`` (fixest 0.14,
default ``ssc()``), both on ``_fixtures/panel_ssc_data.csv``: 90 units
observed 4-8 consecutive periods (540 rows), nested in 15 states, with a
state x period shock so state clustering matters.

The default ``sp.panel`` scaling is linearmodels' own and matches neither
package once SEs are robust or clustered (a factor of up to 1.40 in the
variance here). ``ssc="stata"`` / ``ssc="fixest"`` rebuild the covariance on
the same sample. Every coefficient, SE and reference distribution (t with
the package's degrees of freedom, or z) is compared; the observed gaps are
1e-14, so the SE tolerance of 1e-10 leaves four orders of margin without
admitting any convention difference (the smallest one here is 539/537 vs
539/538, 2e-3).
"""

from __future__ import annotations

import json
import math
import pathlib
import warnings

import numpy as np
import pandas as pd
import pytest

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import statspai as sp
from statspai.exceptions import MethodIncompatibility

_FIX = pathlib.Path(__file__).parent / "_fixtures"
_CLUSTER = {"cluster_id": "entity", "cluster_st": "st", "cluster_t": "time"}
_NAME = {"_cons": "const", "D.x1": "x1", "D.x2": "x2"}


def _term(name):
    # sp.panel names the CRE columns _mean_<x> / _cham_<x>_t<p>.
    if name.startswith(("mean_", "cham_")):
        return "_" + name
    return _NAME.get(name, name)


def _load(name):
    path = _FIX / name
    if not path.exists():  # pragma: no cover
        pytest.skip(f"run the generator for {name} first")
    return json.loads(path.read_text(encoding="utf-8"))


STATA = _load("panel_ssc_stata.json")
FIXEST = _load("panel_ssc_fixest.json")


@pytest.fixture(scope="module")
def data():
    return pd.read_csv(_FIX / "panel_ssc_data.csv")


def _fit(data, key, ssc):
    weighted = key.startswith("w_")
    method, vce = (key[2:] if weighted else key).split("_", 1)
    kw = {}
    if vce == "robust":
        kw["robust"] = "robust"
    elif vce != "unadjusted":
        kw["cluster"] = _CLUSTER[vce]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.panel(
            data,
            "y ~ x1 + x2",
            entity="id",
            time="t",
            method=method,
            ssc=ssc,
            weights="w" if weighted else None,
            **kw,
        )


def _check(res, ref, ref_df):
    for key, value in ref.items():
        if not key.startswith(("b_", "se_")):
            continue
        kind, term = key.split("_", 1)
        got = (res.params if kind == "b" else res.std_errors)[_term(term)]
        tol = 1e-12 if kind == "b" else 1e-10
        assert got == pytest.approx(value, rel=tol), key
    ours = res._inference_df()
    if ref_df is None:
        assert math.isinf(ours)  # z
    else:
        assert ours == ref_df
    # p-values and intervals follow the same reference distribution.
    t = res.params / res.std_errors
    from scipy import stats

    p = 2 * stats.t.sf(np.abs(t), ours)
    np.testing.assert_allclose(res.pvalues, p, rtol=1e-12)


@pytest.mark.parametrize("key", [k for k in STATA if k != "_meta"])
def test_matches_stata(data, key):
    ref = STATA[key]
    ref_df = None if ref["df_r"] < 0 else float(ref["df_r"])
    _check(_fit(data, key, "stata"), ref, ref_df)


@pytest.mark.parametrize("key", [k for k in FIXEST if k != "_meta"])
def test_matches_fixest(data, key):
    ref = FIXEST[key]
    _check(_fit(data, key, "fixest"), ref, float(ref["t_df"]))


def test_stata_and_fixest_differ_where_their_conventions_do(data):
    # xtreg's vce(robust) clusters on the panel; fixest's hetero is HC1.
    st = _fit(data, "fe_robust", "stata")
    fx = _fit(data, "fe_robust", "fixest")
    assert st.std_errors["x1"] == pytest.approx(
        _fit(data, "fe_cluster_id", "stata").std_errors["x1"], rel=1e-12
    )
    assert abs(st.std_errors["x1"] / fx.std_errors["x1"] - 1) > 1e-2
    # Time-clustered two-way: areg counts the nested period dummies, fixest
    # does not (K = 99 vs 92).
    a = _fit(data, "twoway_cluster_t", "stata").std_errors["x1"]
    b = _fit(data, "twoway_cluster_t", "fixest").std_errors["x1"]
    assert (a / b) ** 2 == pytest.approx((540 - 92) / (540 - 99), rel=1e-10)


def test_default_scaling_is_unchanged(data):
    from linearmodels.panel import PanelOLS

    p = data.set_index(["id", "t"])
    lm = PanelOLS(p.y, p[["x1", "x2"]], entity_effects=True).fit(
        cov_type="clustered", cluster_entity=True
    )
    res = sp.panel(data, "y ~ x1 + x2", entity="id", time="t", cluster="entity")
    np.testing.assert_allclose(res.std_errors, lm.std_errors, rtol=1e-14)
    assert "ssc" not in res.model_info
    assert (
        abs(
            res.std_errors["x1"] / _fit(data, "fe_cluster_id", "stata").std_errors["x1"]
            - 1
        )
        > 1e-3
    )


def test_coefficients_do_not_depend_on_ssc(data):
    base = _fit(data, "twoway_cluster_st", None)
    for ssc in ("stata", "fixest"):
        np.testing.assert_allclose(
            _fit(data, "twoway_cluster_st", ssc).params, base.params, rtol=1e-12
        )


def test_full_covariance_is_stored_for_postestimation(data):
    from statspai.postestimation._covariance import coefficient_covariance

    for ssc in (None, "stata"):
        res = _fit(data, "fe_cluster_st", ssc)
        V, se = coefficient_covariance(res)
        assert V is not None
        np.testing.assert_allclose(np.sqrt(np.diag(V)), res.std_errors, rtol=1e-12)
    res = _fit(data, "fe_cluster_st", "stata")
    V = res.data_info["vcov"]
    diff_se = math.sqrt(V.loc["x1", "x1"] + V.loc["x2", "x2"] - 2 * V.loc["x1", "x2"])
    out = sp.lincom(res, "x1 - x2")
    assert out["se"] == pytest.approx(diff_se, rel=1e-10)
    assert out["df"] == 14  # t(G - 1), as Stata's lincom after xtreg


def test_first_differences_cluster_on_time_needs_ssc(data):
    with pytest.raises(MethodIncompatibility, match="constant within each unit"):
        sp.panel(
            data, "y ~ x1 + x2", entity="id", time="t", method="fd", cluster="time"
        )
    _fit(data, "fd_cluster_t", "stata")  # the native path handles it


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"method": "re", "ssc": "fixest"}, "not defined"),
        ({"method": "be", "ssc": "fixest"}, "not defined"),
        ({"method": "fe", "ssc": "sas"}, "not supported"),
        ({"method": "ab", "ssc": "stata"}, "own conventions"),
        ({"method": "fe", "ssc": "stata", "vce": "CR2", "cluster": "id"}, "own"),
        ({"method": "fe", "ssc": "stata", "cluster": "twoway"}, "one-way"),
        ({"method": "be", "ssc": "stata", "cluster": "st"}, "conventional"),
        ({"method": "fe", "ssc": "stata", "robust": "kernel"}, "Driscoll"),
    ],
)
def test_unsupported_combinations_raise(data, kwargs, match):
    with pytest.raises(MethodIncompatibility, match=match):
        sp.panel(data, "y ~ x1 + x2", entity="id", time="t", **kwargs)
