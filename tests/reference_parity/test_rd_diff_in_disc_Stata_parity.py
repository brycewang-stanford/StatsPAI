"""``sp.rd_diff_in_disc`` against Stata (``rdrobust`` + the pooled regression).

There was no difference-in-discontinuities estimator (Watering Down, QJE
2020): the replication subtracted two ``rdrobust`` runs by hand, whose SEs
ignore the covariance between the periods. ``rd_diff_in_disc`` fits the
Grembi, Nannicini and Troiano (2016) pooled local-linear regression, fully
interacted in side and period, with kernel weights.

Reference: Stata 18 on ``_fixtures/rd_diff_in_disc.csv`` (150 sites, pre and
post), from ``_generate_rd_diff_in_disc_Stata.do``: ``rdrobust`` in each
period with h = 0.6 (conventional estimates) and ``regress ... [aw = kernel],
vce(cluster site) | vce(robust)`` on the pooled design. Estimate, both SEs
and both period discontinuities agree to 1e-12.
"""

from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd
import pytest

import statspai as sp

_FIX = pathlib.Path(__file__).parent / "_fixtures"
R = json.loads((_FIX / "rd_diff_in_disc_Stata.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def data():
    return pd.read_csv(_FIX / "rd_diff_in_disc.csv")


def test_matches_stata(data):
    cl = sp.rd_diff_in_disc(data, y="y", x="x", post="post", h=R["h"], cluster="site")
    hc = sp.rd_diff_in_disc(data, y="y", x="x", post="post", h=R["h"])
    assert cl.n_obs == R["N"]
    assert cl.estimate == pytest.approx(R["b"], rel=1e-12)
    assert cl.se == pytest.approx(R["se_cluster"], rel=1e-12)
    assert hc.se == pytest.approx(R["se_hc1"], rel=1e-12)
    disc = cl.detail.set_index("period")["discontinuity"]
    assert disc["pre"] == pytest.approx(R["rd_pre"], rel=1e-12)
    assert disc["post"] == pytest.approx(R["rd_post"], rel=1e-12)


def test_equals_difference_of_period_rdrobust(data):
    """The point estimate is post RD minus pre RD at the common bandwidth."""
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        pre = sp.rdrobust(data[data.post == 0], y="y", x="x", h=R["h"])
        post = sp.rdrobust(data[data.post == 1], y="y", x="x", h=R["h"])

    def conv(r):
        return r.detail.set_index("method").loc["Conventional", "estimate"]

    est = sp.rd_diff_in_disc(data, y="y", x="x", post="post", h=R["h"]).estimate
    assert est == pytest.approx(conv(post) - conv(pre), rel=1e-10)


def test_default_bandwidth_and_errors(data):
    r = sp.rd_diff_in_disc(data, y="y", x="x", post="post", cluster="site")
    assert r.model_info["bandwidth_rule"].startswith("min(")
    with pytest.raises(sp.MethodIncompatibility, match="0/1"):
        sp.rd_diff_in_disc(data.assign(post=data.post * 2), y="y", x="x", post="post")


def test_covs_alias_is_covariates(data):
    d = data.assign(z=np.sin(np.arange(len(data))))
    a = sp.rd_diff_in_disc(d, y="y", x="x", post="post", h=R["h"], covariates=["z"])
    b = sp.rd_diff_in_disc(d, y="y", x="x", post="post", h=R["h"], covs=["z"])
    assert a.estimate == b.estimate and a.se == b.se
    assert "z" in a.model_info["coefficients"].index
