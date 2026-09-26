"""sp.panel honours weights= and a named cluster column, or refuses loudly.

Before 1.32 the linearmodels path dropped ``weights=`` without a word and
clustered on the entity whenever ``cluster=`` named any other column, while
the result still reported the requested column.
"""

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import MethodIncompatibility

FORMULA = "y ~ treat + x1"


@pytest.fixture(scope="module")
def pn():
    rng = np.random.default_rng(0)
    n_units, n_t = 200, 6
    ids = np.repeat(np.arange(n_units), n_t)
    tt = np.tile(np.arange(1, n_t + 1), n_units)
    st = np.repeat(np.arange(n_units) % 12, n_t)
    df = pd.DataFrame(
        {
            "id": ids,
            "time": tt,
            "st": st,
            "x1": rng.normal(size=ids.size),
            "w": rng.uniform(0.2, 3.0, ids.size),
            "treat": rng.integers(0, 2, ids.size),
        }
    )
    # State-level shocks that grow over time make state clustering matter.
    df["y"] = (
        df.treat * (1 + df.w)
        + rng.normal(size=12)[st] * tt
        + 0.5 * df.x1
        + rng.normal(size=ids.size)
    )
    df["id_copy"] = df["id"]
    return df


def _fit(pn, **kw):
    return sp.panel(pn, FORMULA, entity="id", time="time", **kw)


@pytest.mark.parametrize(
    "method, feols_formula",
    [
        ("fe", "y ~ treat + x1 | id"),
        ("twoway", "y ~ treat + x1 | id + time"),
        ("pooled", "y ~ treat + x1"),
    ],
)
def test_weighted_fit_matches_feols(pn, method, feols_formula):
    got = _fit(pn, method=method, weights="w", robust="robust")
    ref = sp.feols(feols_formula, data=pn, weights="w", vce="hetero")
    # Same WLS problem; twoway demeaning is iterative, hence 1e-8 on the SE.
    assert got.params["treat"] == pytest.approx(ref.params["treat"], abs=1e-12)
    assert got.std_errors["treat"] == pytest.approx(ref.std_errors["treat"], rel=1e-8)
    unweighted = _fit(pn, method=method, robust="robust")
    assert abs(got.params["treat"] - unweighted.params["treat"]) > 1e-3


@pytest.mark.parametrize("method", ["re", "fd", "be", "mundlak", "chamberlain", "ab"])
def test_weights_refused_without_reference(pn, method):
    with pytest.raises(MethodIncompatibility, match="weights"):
        _fit(pn, method=method, weights="w")


@pytest.mark.parametrize("method", ["fe", "twoway", "pooled", "fd", "re"])
def test_named_cluster_column_is_used(pn, method):
    by_state = _fit(pn, method=method, cluster="st")
    by_entity = _fit(pn, method=method, cluster="entity")
    # A column identical to the entity id reproduces entity clustering...
    by_copy = _fit(pn, method=method, cluster="id_copy")
    assert by_copy.std_errors["treat"] == pytest.approx(
        by_entity.std_errors["treat"], rel=1e-12
    )
    # ...and a coarser column is not silently replaced by the entity.
    assert by_state.std_errors["treat"] != pytest.approx(
        by_entity.std_errors["treat"], rel=1e-3
    )
    assert by_state.model_info["n_clusters"] == 12


def test_state_cluster_matches_feols_up_to_documented_ssc(pn):
    got = _fit(pn, method="fe", cluster="st")
    ref = sp.feols("y ~ treat + x1 | id", data=pn, cluster="st")
    # linearmodels omits fixest's G/(G-1) and (n-1)/(n-K) factors; the ratio
    # is sqrt((G-1)/G) up to the (n-1)/(n-K) term, which here is ~1e-6.
    ratio = got.std_errors["treat"] / ref.std_errors["treat"]
    assert ratio == pytest.approx(np.sqrt(11 / 12), rel=1e-5)


def test_unknown_cluster_column_raises(pn):
    with pytest.raises(MethodIncompatibility, match="not found"):
        _fit(pn, method="fe", cluster="no_such_column")


def test_gmm_rejects_non_entity_cluster(pn):
    with pytest.raises(MethodIncompatibility, match="entity"):
        sp.panel(pn, "y ~ x1", entity="id", time="time", method="ab", cluster="st")


def test_nonpositive_weights_raise(pn):
    bad = pn.assign(w=pn.w.where(pn.index != 0, 0.0))
    with pytest.raises(MethodIncompatibility, match="strictly positive"):
        _fit(bad, method="fe", weights="w")
