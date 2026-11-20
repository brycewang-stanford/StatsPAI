"""``result.report()`` under few clusters: the wild cluster bootstrap interval."""

import json
import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.agent import mcp_server as M


def _data(seed, G, m=60, tau=0.3):
    rng = np.random.default_rng(seed)
    g = np.repeat(np.arange(G), m)
    tr = (g < G // 2).astype(int)
    ce = rng.normal(size=G)[g]
    return pd.DataFrame(
        {"school": g, "treat": tr, "y": tau * tr + ce + rng.normal(size=G * m)}
    )


def _fit(df):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.regress("y ~ treat", data=df, cluster="school")


def test_few_clusters_report_the_inverted_wild_bootstrap_interval():
    df = _data(0, 8)
    r = _fit(df)
    rep = r.report(data=df)
    assert rep["status"] == "adjusted" and rep["term"] == "treat"
    assert rep["ci_method"] == "wild_cluster_bootstrap"
    ref = sp.wild_cluster_boot(
        r, df, "school", variable="treat", confidence_set=True, seed=0
    )
    assert rep["ci"] == pytest.approx(list(ref["ci_inverted"]))
    assert rep["conventional"]["ci"] == pytest.approx(
        [float(r.conf_int_lower["treat"]), float(r.conf_int_upper["treat"])]
    )


def test_without_data_the_report_is_a_caution():
    rep = _fit(_data(0, 8)).report()
    assert rep["status"] == "caution"
    assert rep["ci"] == rep["conventional"]["ci"]


def test_many_clusters_keep_the_conventional_interval():
    df = _data(1, 60, m=10)
    rep = _fit(df).report(data=df)
    assert rep["status"] == "conventional" and rep["basis"] == []


def test_several_regressors_report_each_slope():
    df = _data(2, 8)
    df["x"] = np.random.default_rng(5).normal(size=len(df))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        r = sp.regress("y ~ treat + x", data=df, cluster="school")
    rep = r.report(data=df)
    assert rep["estimate"] is None and set(rep["terms"]) == {"treat", "x"}
    assert len(rep["terms"]["treat"]["report_ci"]) == 2


def test_mcp_regress_computes_it_from_the_bound_data(tmp_path):
    df = _data(0, 8)
    path = tmp_path / "d.csv"
    df.to_csv(path, index=False)
    req = {
        "jsonrpc": "2.0",
        "id": 1,
        "method": "tools/call",
        "params": {
            "name": "regress",
            "arguments": {
                "formula": "y ~ treat",
                "cluster": "school",
                "data_path": str(path),
            },
        },
    }
    out = json.loads(M.handle_request(json.dumps(req)))["result"]
    payload = json.loads(out["content"][0]["text"])
    assert payload["report"]["ci_method"] == "wild_cluster_bootstrap"
    assert payload["report"]["ci"] == pytest.approx(_fit(df).report(data=df)["ci"])
