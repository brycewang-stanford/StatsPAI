"""``sp.xtevent(trend=, proxy=)`` against Stata ``xtevent`` 3.1.0 (SSC).

References: ``_fixtures/xtevent_extensions_Stata.json`` from
``_fixtures/_generate_xtevent_extensions_Stata.do`` on the continuous
panel of the base xtevent fixture and on ``_fixtures/xtevent_binary.csv``
(``_generate_xtevent_binary_data.py``: binary staggered adoption with a
confound, two proxies and an external instrument).

T2 (1e-10): trend adjustment by GMM and by OLS (with clustering), the
proxy-IV estimator with the selected lead, given leads (including the
lead whose coefficient is ``norm``), two proxies, an external instrument
and clustering: every event-time coefficient, the proxy coefficients,
their SEs, N, and the set of normalised event times.

One departure, pinned below: with ``trend`` later than the window start,
Stata's adjusted covariance attaches the trend weights to the coefficient
at the window start instead of the ones the trend was fitted on (its
point estimates are right). Ours satisfies the delta-method formula; the
Stata standard errors are reproduced exactly by the misplaced weights.
"""

from __future__ import annotations

import json
import pathlib
import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import MethodIncompatibility

FX = pathlib.Path(__file__).parent / "_fixtures"
STATA = json.loads((FX / "xtevent_extensions_Stata.json").read_text(encoding="utf-8"))

CASES = {
    "cont_trend3_gmm": ("cont", dict(covariates=["x"], window=3, trend=-3)),
    "bin_trend4_ols": ("bin", dict(window=4, trend=-4, trend_method="ols")),
    "bin_trend3_ols_cluster": (
        "bin",
        dict(window=4, trend=-3, trend_method="ols", cluster="cl"),
    ),
    "bin_trend4_gmm": ("bin", dict(window=4, trend=-4)),
    "bin_proxy_select": ("bin", dict(window=4, proxy=["x"])),
    "bin_proxy_lead3": ("bin", dict(window=4, proxy=["x"], proxy_iv=[3])),
    "bin_proxy_lead1": ("bin", dict(window=4, proxy=["x"], proxy_iv=[1])),
    "bin_proxy_lead23_cluster": (
        "bin",
        dict(window=4, proxy=["x"], proxy_iv=[2, 3], cluster="cl"),
    ),
    "bin_proxy2_lead23": ("bin", dict(window=4, proxy=["x", "x2"], proxy_iv=[2, 3])),
    "bin_proxy_external": ("bin", dict(window=4, proxy=["x"], proxy_iv=["w"])),
}


@pytest.fixture(scope="module")
def data():
    return {
        "cont": pd.read_csv(FX / "xtevent_continuous.csv"),
        "bin": pd.read_csv(FX / "xtevent_binary.csv"),
    }


def _fit(df, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.xtevent(df, y="y", policy="z", panel="id", time="t", **kw)


def _k(name):
    tag = name[len("_k_eq_") :]
    return -int(tag[1:]) if tag[0] == "m" else int(tag[1:])


@pytest.mark.parametrize("case", sorted(CASES))
def test_matches_stata(case, data):
    kind, kw = CASES[case]
    res = _fit(data[kind], **kw)
    ref = STATA[case]
    es = res.model_info["event_study"].set_index("relative_time")
    ct = res.model_info["coef_table"]
    assert res.n_obs == ref["N"]
    zeroed = sorted(es.index[es["is_reference"]].tolist())
    assert zeroed == sorted(int(v) for v in ref["komit"].split())
    checked = 0
    for name, b, se in zip(ref["names"].split(), ref["b"], ref["se"]):
        if name.startswith("_k_eq_"):
            got_b, got_se = es.loc[_k(name), "estimate"], es.loc[_k(name), "se"]
        elif name in ("x", "x2", "_ttrend"):
            got_b, got_se = ct.loc[name, "estimate"], ct.loc[name, "se"]
        else:
            continue
        assert abs(got_b - b) < 1e-9, name
        assert abs(got_se - se) < 1e-9 * max(1.0, se), name
        checked += 1
    assert checked >= 7


def test_trend_after_window_start_is_the_delta_method(data):
    """trend=-2: phi = -delta_{-2}, so delta*_k = delta_k + (k+1) delta_{-2}."""
    raw = _fit(data["cont"], covariates=["x"], window=3, cluster="cl")
    adj = _fit(data["cont"], covariates=["x"], window=3, cluster="cl", trend=-2)
    V = raw.model_info["vcov_event"]
    es_raw = raw.model_info["event_study"].set_index("relative_time")
    es_adj = adj.model_info["event_study"].set_index("relative_time")
    for k in V.index:
        h = k + 1.0
        b = es_raw.loc[k, "estimate"] + h * es_raw.loc[-2, "estimate"]
        v = V.loc[k, k] + h**2 * V.loc[-2, -2] + 2 * h * V.loc[k, -2]
        assert abs(es_adj.loc[k, "estimate"] - b) < 1e-12
        assert abs(es_adj.loc[k, "se"] - np.sqrt(max(v, 0.0))) < 1e-12


def test_stata_trend_variance_uses_the_window_start(data):
    """Stata's SEs for trend=-2 (window starts at -3) equal the delta-method
    variance with the trend weight put on the coefficient at -3."""
    raw = _fit(data["cont"], covariates=["x"], window=3, cluster="cl")
    ref = STATA["cont_trend2_gmm_cluster"]
    V = raw.model_info["vcov_event"]
    ks = list(V.index)
    Vm = V.to_numpy()
    lam = np.zeros(len(ks))
    lam[ks.index(-3)] = -1.0  # Stata: weight of delta_{-2} placed at -3
    H = np.array([k + 1.0 for k in ks])
    A = np.eye(len(ks)) - np.outer(H, lam)
    se = np.sqrt(np.diag(A @ Vm @ A.T))
    stata_se = dict(zip(ref["names"].split(), ref["se"]))
    for j, k in enumerate(ks):
        name = f"_k_eq_m{-k}" if k < 0 else f"_k_eq_p{k}"
        assert abs(se[j] - stata_se[name]) < 1e-9 * max(1.0, stata_se[name])
    # point estimates agree with Stata
    adj = _fit(data["cont"], covariates=["x"], window=3, cluster="cl", trend=-2)
    es = adj.model_info["event_study"].set_index("relative_time")
    for name, b in zip(ref["names"].split(), ref["b"]):
        if name.startswith("_k_eq_"):
            assert abs(es.loc[_k(name), "estimate"] - b) < 1e-9


def test_refusals(data):
    with pytest.raises(MethodIncompatibility):
        _fit(data["bin"], window=4, trend=-5)  # before the window start
    with pytest.raises(MethodIncompatibility):
        _fit(data["bin"], window=4, trend=-3, norm=-2)
    with pytest.raises(MethodIncompatibility):
        _fit(data["cont"], window=3, trend=-3, trend_method="ols")  # not 0/1
    with pytest.raises(MethodIncompatibility):
        _fit(data["bin"], window=4, proxy=["x"], engine="reghdfe")
    with pytest.raises(MethodIncompatibility):
        _fit(data["bin"], window=4, proxy=["x", "x2"], proxy_iv="select")


def test_scalar_proxy_iv_is_stata_proxyiv_1(data):
    """proxy_iv=1 is Stata ``proxyiv(1)`` (it raised TypeError before 2026-10-20)."""
    res = _fit(data["bin"], window=4, proxy=["x"], proxy_iv=1)
    ref = STATA["bin_proxy_lead1"]
    es = res.model_info["event_study"].set_index("relative_time")
    for name, b, se in zip(ref["names"].split(), ref["b"], ref["se"]):
        if name.startswith("_k_eq_"):
            assert abs(es.loc[_k(name), "estimate"] - b) < 1e-9, name
            assert abs(es.loc[_k(name), "se"] - se) < 1e-9 * max(1.0, se), name


def test_missing_external_instrument_is_column_not_found(data):
    with pytest.raises(sp.exceptions.ColumnNotFound, match="nope"):
        _fit(data["bin"], window=4, proxy=["x"], proxy_iv=["nope"])
    with pytest.raises(MethodIncompatibility, match="proxy_iv must be"):
        _fit(data["bin"], window=4, proxy=["x"], proxy_iv=1.5)
