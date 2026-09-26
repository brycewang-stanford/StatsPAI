"""Cross-language parity: ``sp.qreg`` cluster-robust and kernel SEs vs qreg2.

Fixture: ``_fixtures/_generate_qreg_cluster_stata.do`` -- Stata 18 with
``qreg2`` (SSC, version 4.00), the implementation by Machado, Parente and
Santos Silva of Parente and Santos Silva (2016) [@parente2016quantile].
Official ``qreg`` refuses ``vce(cluster)``, so the method authors' module is
the reference. 600 observations in 40 clusters; errors share a cluster
component and are heteroskedastic in ``x1``.

Compared: coefficients, the full covariance matrix (off-diagonals
included), p-values and 95% intervals at quantiles 0.25 / 0.5 / 0.75 / 0.9,
clustered and unclustered, with the MAD and the Silverman bandwidth scale;
plus string cluster ids and cluster ids with missing values (qreg2 drops
those rows). Observed gaps are 1e-13; the tolerance on the covariance is
1e-10 relative to sqrt(V_ii V_jj).
"""

from __future__ import annotations

import json
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
_TERMS = {"x1": "x1", "x2": "x2", "cons": "const"}


def _load():
    path = _FIX / "qreg_cluster_stata.json"
    if not path.exists():  # pragma: no cover
        pytest.skip("run _generate_qreg_cluster_stata.do first")
    return json.loads(path.read_text(encoding="utf-8"))


STATA = _load()


@pytest.fixture(scope="module")
def data():
    return pd.read_csv(_FIX / "qreg_cluster_data.csv")


def _spec(key):
    tau = int(key[-2:]) / 100
    kw = {"quantile": tau}
    if key.startswith("cluster"):
        kw["cluster"] = {"string": "gs", "missing": "g_miss"}.get(
            key.split("_")[1], "g"
        )
    else:
        kw["vce"] = "kernel"
    if "silverman" in key:
        kw["kernel_scale"] = "silverman"
    return kw


@pytest.mark.parametrize("key", [k for k in STATA if k != "_meta"])
def test_matches_qreg2(data, key):
    ref = STATA[key]
    res = sp.qreg(data, y="y", x=["x1", "x2"], **_spec(key))
    detail = res.detail.set_index("variable")
    V = res.model_info["vcov"]
    assert res.n_obs == int(ref["N"])
    for a, ours_a in _TERMS.items():
        assert detail.loc[ours_a, "coefficient"] == pytest.approx(
            ref[f"b_{a}"], rel=1e-10, abs=1e-12
        )
        assert detail.loc[ours_a, "pvalue"] == pytest.approx(
            ref[f"p_{a}"], rel=1e-8, abs=1e-15
        )
        for b, ours_b in _TERMS.items():
            scale = np.sqrt(ref[f"V_{a}_{a}"] * ref[f"V_{b}_{b}"])
            assert abs(V.loc[ours_a, ours_b] - ref[f"V_{a}_{b}"]) < 1e-10 * scale
    # The headline interval is on x1, qreg2's t(N - k) interval.
    assert res.ci[0] == pytest.approx(ref["ll_x1"], rel=1e-10)
    assert res.ci[1] == pytest.approx(ref["ul_x1"], rel=1e-10)


def test_clustering_changes_the_variance(data):
    se = {
        k: sp.qreg(data, y="y", x=["x1", "x2"], **kw).detail.set_index("variable")["se"]
        for k, kw in (("cl", {"cluster": "g"}), ("het", {"vce": "kernel"}))
    }
    ratio = se["cl"] / se["het"]
    # The shared cluster shock loads on the intercept and on x2; a guard that
    # the scores really are summed within clusters.
    assert ratio["const"] > 1.15 and ratio["x2"] > 1.15


@pytest.mark.parametrize("vce", ["cluster g", "cluster(g)", "CLUSTER g"])
def test_stata_vce_grammar(data, vce):
    a = sp.qreg(data, y="y", x=["x1", "x2"], vce=vce)
    b = sp.qreg(data, y="y", x=["x1", "x2"], cluster="g")
    assert a.se == b.se
    assert a.model_info["n_clusters"] == 40


def test_default_is_unchanged(data):
    res = sp.qreg(data, y="y", x=["x1", "x2"])
    assert res.model_info["vce"] == "iid"
    assert "n_clusters" not in res.model_info


@pytest.mark.parametrize(
    "kwargs, match",
    [
        ({"vce": "cluster"}, "needs a cluster column"),
        ({"vce": "robust", "cluster": "g"}, "conflicts"),
        ({"vce": "cluster g", "cluster": "gs"}, "different"),
        ({"vce": "kernel", "kernel_scale": "iqr"}, "kernel_scale"),
        ({"vce": "hc1"}, "must be one of"),
    ],
)
def test_bad_requests_raise(data, kwargs, match):
    with pytest.raises(MethodIncompatibility, match=match):
        sp.qreg(data, y="y", x=["x1", "x2"], **kwargs)


def test_one_cluster_raises(data):
    with pytest.raises(MethodIncompatibility, match="at least two"):
        sp.qreg(data.assign(one=1), y="y", x=["x1", "x2"], cluster="one")


def test_bandwidth_outside_unit_interval_raises():
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"y": rng.normal(size=12), "x": rng.normal(size=12)})
    with pytest.raises(MethodIncompatibility, match="outside"):
        sp.qreg(df, y="y", x=["x"], quantile=0.95, vce="kernel")
