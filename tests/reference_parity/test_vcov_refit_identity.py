"""``sp.vcov`` / ``sp.vcov_compare``: standard errors under another ``vce=``.

The contract is an identity: ``sp.vcov(fit, v)`` returns exactly what the
estimator returns when called with ``vce=v`` -- same standard errors, same
point estimates, bit for bit -- because it *is* that call. The tests below
check the identity on the estimator families with the most-used SE menus,
and the refusals that keep a replay honest.
"""

import gc
import inspect
import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai._frame_input import bare
from statspai.core._call_record import get_call_record
from statspai.exceptions import MethodIncompatibility


def _quiet(fn, *a, **k):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*a, **k)


def _assert_identical(a, b):
    for attr in ("params", "std_errors", "estimate", "se"):
        x, y = getattr(a, attr, None), getattr(b, attr, None)
        if x is None and y is None:
            continue
        np.testing.assert_array_equal(np.asarray(x, float), np.asarray(y, float))


@pytest.fixture(scope="module")
def cross():
    rng = np.random.default_rng(0)
    n = 500
    z = rng.normal(size=n)
    u = rng.normal(size=n)
    x = 0.8 * z + 0.5 * u + rng.normal(size=n)
    y = 1 + 2 * x + u * (1 + np.abs(x))
    return pd.DataFrame(
        {
            "y": y,
            "x": x,
            "z": z,
            "w": rng.uniform(0.2, 3, n),
            "g": rng.integers(0, 40, n),
            "h": rng.integers(0, 12, n),
            "f": rng.integers(0, 30, n),
            "yb": (y > 2).astype(int),
            "c": rng.poisson(np.exp(0.2 * x)),
        }
    )


REGRESS_VCE = ["hc0", "hc1", "hc2", "hc3", "cluster g", "cr2 g", "cr3 g"]


@pytest.mark.parametrize("weights", [None, "w"])
@pytest.mark.parametrize("vce", REGRESS_VCE)
def test_regress_identity(cross, vce, weights):
    fit = _quiet(sp.regress, "y ~ x", data=cross, weights=weights)
    direct = _quiet(sp.regress, "y ~ x", data=cross, weights=weights, vce=vce)
    _assert_identical(_quiet(sp.vcov, fit, vce), direct)


def test_regress_twoway_and_positional(cross):
    fit = _quiet(sp.regress, "y ~ x", cross, "hc1")  # robust= positionally
    direct = _quiet(sp.regress, "y ~ x", data=cross, vce="cluster g h")
    _assert_identical(_quiet(sp.vcov, fit, "cluster g h"), direct)


@pytest.mark.parametrize("vce", ["robust", "cluster g"])
@pytest.mark.parametrize("model", ["logit", "probit"])
def test_binary_identity(cross, model, vce):
    fn = getattr(sp, model)
    fit = _quiet(fn, "yb ~ x", data=cross)
    _assert_identical(
        _quiet(sp.vcov, fit, vce), _quiet(fn, "yb ~ x", data=cross, vce=vce)
    )


@pytest.mark.parametrize("vce", ["hc1", "cluster g"])
def test_ivreg_identity(cross, vce):
    fit = _quiet(sp.ivreg, "y ~ (x ~ z)", data=cross)
    direct = _quiet(sp.ivreg, "y ~ (x ~ z)", data=cross, vce=vce)
    _assert_identical(_quiet(sp.vcov, fit, vce), direct)


@pytest.mark.parametrize(
    "vce,native",
    [
        ("robust", "hetero"),
        ("cluster g", {"CRV1": "g"}),
        ("cluster g h", {"CRV1": "g+h"}),
        ("cr3 g", {"CRV3": "g"}),
    ],
)
def test_feols_identity_and_stata_spelling(cross, vce, native):
    fit = _quiet(sp.feols, "y ~ x | f", data=cross)
    pyfixest_spelling = _quiet(sp.feols, "y ~ x | f", data=cross, vcov=native)
    _assert_identical(
        _quiet(sp.feols, "y ~ x | f", data=cross, vce=vce), pyfixest_spelling
    )
    _assert_identical(_quiet(sp.vcov, fit, vce), pyfixest_spelling)


def test_fepois_stata_cluster_spelling(cross):
    a = _quiet(sp.fepois, "c ~ x | f", data=cross, vce="cluster g")
    b = _quiet(sp.fepois, "c ~ x | f", data=cross, vcov={"CRV1": "g"})
    _assert_identical(a, b)


@pytest.mark.parametrize("vce", ["hc3", "cluster g", "nncluster g"])
def test_rdrobust_identity(vce):
    rng = np.random.default_rng(1)
    n = 1500
    x = rng.uniform(-1, 1, n)
    df = pd.DataFrame(
        {
            "y": x + (x >= 0) + rng.normal(0, 0.4, n),
            "x": x,
            "g": rng.integers(0, 80, n),
        }
    )
    fit = _quiet(sp.rdrobust, df, y="y", x="x")
    _assert_identical(
        _quiet(sp.vcov, fit, vce), _quiet(sp.rdrobust, df, y="y", x="x", vce=vce)
    )
    if vce != "hc3":
        kind = "hc1" if vce.startswith("cluster") else "nn"
        _assert_identical(
            _quiet(sp.rdrobust, df, y="y", x="x", vce=vce),
            _quiet(sp.rdrobust, df, y="y", x="x", vce=kind, cluster="g"),
        )


@pytest.fixture(scope="module")
def mpdta():
    df = _quiet(sp.datasets.mpdta)
    df["state"] = df["countyreal"] // 25
    return df


def test_callaway_santanna_identity(mpdta):
    kw = dict(data=mpdta, y="lemp", g="first_treat", t="year", i="countyreal")
    fit = _quiet(sp.callaway_santanna, **kw)
    _assert_identical(
        _quiet(sp.vcov, fit, "cluster state"),
        _quiet(sp.callaway_santanna, clustervars="state", **kw),
    )


def test_callaway_santanna_refuses_a_single_cluster(mpdta):
    from statspai.exceptions import DataInsufficient

    kw = dict(
        data=mpdta.assign(one=1), y="lemp", g="first_treat", t="year", i="countyreal"
    )
    with pytest.raises(DataInsufficient, match="at least two clusters"):
        _quiet(sp.callaway_santanna, clustervars="one", **kw)


@pytest.mark.parametrize("fn", ["did_imputation", "gardner_did"])
def test_imputation_did_reads_stata_cluster(mpdta, fn):
    kw = dict(
        data=mpdta, y="lemp", group="countyreal", time="year", first_treat="first_treat"
    )
    f = getattr(sp, fn)
    a = _quiet(f, vce="cluster state", **kw)
    b = _quiet(f, cluster="state", **kw)
    _assert_identical(a, b)
    _assert_identical(_quiet(sp.vcov, _quiet(f, **kw), "cluster state"), b)


CALLS = {
    "cnsreg": (("y ~ x + z",), dict(constraints="x + z = 2")),
    "nls": (("y ~ {a} + {b} * x",), dict(start={"a": 1, "b": 1})),
    "series": (("y ~ 1",), dict(x="x")),
    "threshold": (("y ~ x",), dict(threshold="z")),
}


@pytest.mark.parametrize("fn", sorted(CALLS))
def test_own_vce_estimators_read_stata_cluster(cross, fn):
    f = getattr(sp, fn)
    args, kw = CALLS[fn]
    a = _quiet(f, *args, data=cross, vce="cluster g", **kw)
    b = _quiet(f, *args, data=cross, cluster="g", **kw)
    _assert_identical(a, b)
    _assert_identical(
        _quiet(sp.vcov, _quiet(f, *args, data=cross, **kw), "cluster g"), b
    )


def test_lm_lin_reads_stata_cluster(cross):
    df = cross.assign(d=(cross["z"] > 0).astype(int))
    a = _quiet(sp.lm_lin, df, y="y", treat="d", covariates=["x"], vce="cluster g")
    b = _quiet(sp.lm_lin, df, y="y", treat="d", covariates=["x"], cluster="g")
    _assert_identical(a, b)


# ---------------------------------------------------------------------------
# Refusals and bookkeeping
# ---------------------------------------------------------------------------


def test_edited_data_is_refused(cross):
    df = cross.copy()
    fit = _quiet(sp.regress, "y ~ x", data=df)
    df.loc[0, "y"] += 10.0
    with pytest.raises(MethodIncompatibility, match="does not reproduce"):
        sp.vcov(fit, "hc1")


def test_a_different_frame_is_refused(cross):
    fit = _quiet(sp.regress, "y ~ x", data=cross)
    other = cross.sample(frac=1.0, replace=True, random_state=0)
    with pytest.raises(MethodIncompatibility, match="does not reproduce"):
        sp.vcov(fit, "hc1", data=other)


def test_collected_frame_needs_data(cross):
    df = cross.copy()
    fit = _quiet(sp.regress, "y ~ x", data=df)
    del df
    gc.collect()
    with pytest.raises(MethodIncompatibility, match="no longer exists"):
        sp.vcov(fit, "hc1")
    direct = _quiet(sp.regress, "y ~ x", data=cross, vce="hc1")
    _assert_identical(_quiet(sp.vcov, fit, "hc1", data=cross), direct)


def test_unimplemented_kind_raises_through(cross):
    fit = _quiet(
        sp.callaway_santanna,
        **dict(
            data=_quiet(sp.datasets.mpdta),
            y="lemp",
            g="first_treat",
            t="year",
            i="countyreal",
        ),
    )
    with pytest.raises(MethodIncompatibility):
        sp.vcov(fit, "hc3")


def test_result_without_record(cross):
    fit = _quiet(sp.regress, "y ~ x", data=cross)
    object.__delattr__(fit, "_statspai_call")
    with pytest.raises(MethodIncompatibility, match="carries no record"):
        sp.vcov(fit, "hc1")


def test_markout_moves_the_estimate_and_says_so(cross):
    df = cross.copy()
    df.loc[:9, "g"] = np.nan
    fit = _quiet(sp.regress, "y ~ x", data=df)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        new = sp.vcov(fit, "cluster g")
    assert new.model_info["vcov_refit"]["estimates_changed"] is True
    assert any("different point estimates" in str(w.message) for w in caught)


def test_vcov_records_what_it_did(cross):
    fit = _quiet(sp.regress, "y ~ x", data=cross)
    new = _quiet(sp.vcov, fit, "hc3")
    assert new.model_info["vcov_refit"] == {
        "function": "regress",
        "vce": "hc3",
        "replay_checked": True,
        "estimates_changed": False,
    }
    # The new result can itself be re-reported.
    _assert_identical(
        _quiet(sp.vcov, new, "hc1"), _quiet(sp.regress, "y ~ x", data=cross, vce="hc1")
    )


def test_vcov_compare_table(cross):
    fit = _quiet(sp.regress, "y ~ x", data=cross, weights="w")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        tab = sp.vcov_compare(fit, ["hc1", "cluster g", "conley"], terms=["x"])
    assert list(tab["vce"]) == ["(as fitted)", "hc1", "cluster g", "conley"]
    hc1 = _quiet(sp.regress, "y ~ x", data=cross, weights="w", vce="hc1")
    assert tab.loc[tab["vce"] == "hc1", "se"].item() == hc1.std_errors["x"]
    assert tab.loc[0, "se_ratio"] == 1.0
    assert np.isnan(tab.loc[3, "se"]) and tab.loc[3, "note"]


@pytest.fixture(scope="module")
def lp_series():
    rng = np.random.default_rng(3)
    T = 200
    shock = rng.normal(size=T)
    y = np.zeros(T)
    for t in range(1, T):
        y[t] = 0.5 * y[t - 1] + 0.8 * shock[t] + rng.normal(0, 0.5)
    d = (rng.uniform(size=T) < 0.5).astype(float)
    return pd.DataFrame({"y": y, "s": shock, "d": d})


@pytest.mark.parametrize(
    "base,vce",
    [
        ({}, "robust"),
        ({"vce": "hac"}, "robust"),
        ({"vce": "robust"}, "ols"),
        ({"state": "d", "state_transform": "none"}, "robust"),
    ],
)
def test_local_projections_identity(lp_series, base, vce):
    """One estimate per horizon (and per regime): the refit equals the
    direct call bit for bit, and the point estimates do not move."""
    fit = _quiet(sp.local_projections, lp_series, "y", "s", horizons=4, **base)
    with warnings.catch_warnings():
        warnings.simplefilter("error", sp.exceptions.StatsPAIWarning)
        new = sp.vcov(fit, vce)
    direct = _quiet(
        sp.local_projections, lp_series, "y", "s", horizons=4, **{**base, "vce": vce}
    )
    attrs = ("irf_state1", "se_state1", "irf_state0", "se_state0")
    if "state" not in base:
        attrs = ("irf", "se", "ci_lower", "ci_upper")
    for attr in attrs:
        np.testing.assert_array_equal(getattr(new, attr), getattr(direct, attr))


def test_every_vce_estimator_records_its_call():
    missing = []
    for name in sp.list_functions():
        obj = getattr(sp, name, None)
        if obj is None or inspect.isclass(obj) or not callable(obj):
            continue
        if name in {"regtable", "cross_validate", "vcov", "vcov_compare"}:
            continue
        try:
            params = inspect.signature(obj).parameters
        except (TypeError, ValueError):
            continue
        aliases = getattr(obj, "__statspai_aliases__", {}) or {}
        if "vce" not in params and "vce" not in aliases:
            continue
        # beneath the sp.* polars / Arrow input wrapper
        code = getattr(bare(obj), "__code__", None)
        if not (code and code.co_filename.endswith("_call_record.py")):
            missing.append(name)
    assert missing == []


def test_record_holds_the_frame_weakly(cross):
    df = cross.copy()
    fit = _quiet(sp.regress, "y ~ x", data=df)
    record = get_call_record(fit)
    assert record is not None and record.function == "regress"
    assert record.frames()["data"].get() is df


# ---------------------------------------------------------------------------
# regtable(vcov=) goes through the same engine  (⚠️ correctness fix)
# ---------------------------------------------------------------------------


def _se_row(table_text, line_index):
    lines = [l for l in table_text.splitlines() if "(" in l and ")" in l]
    return lines[line_index]


def test_regtable_vcov_respects_weights(cross):
    fit = _quiet(sp.regress, "y ~ x", data=cross, weights="w")
    hc1 = _quiet(sp.regress, "y ~ x", data=cross, weights="w", vce="hc1")
    text = str(_quiet(sp.regtable, fit, vcov="HC1"))
    assert f"({hc1.std_errors['x']:.3f})" in text


def test_regtable_vcov_uses_the_ml_sandwich_on_logit(cross):
    fit = _quiet(sp.logit, "yb ~ x", data=cross)
    robust = _quiet(sp.logit, "yb ~ x", data=cross, vce="robust")
    text = str(_quiet(sp.regtable, fit, vcov="robust"))
    assert f"({robust.std_errors['x']:.3f})" in text


def test_regress_provenance_records_weights_and_hac_options(cross):
    # The human-readable lineage (replication packs, regtable footers) used to
    # record a weighted fit as unweighted: weights= is a named parameter and
    # the provenance filter only looked inside **kwargs.
    fit = _quiet(sp.regress, "y ~ x", data=cross, weights="w", robust="hac", hac_lags=3)
    params = fit._provenance.params
    assert params["weights"] == "w"
    assert params["hac_lags"] == 3


def test_a_pickled_result_keeps_its_record_but_not_the_frame(cross):
    # The MCP state store and isolated fits pickle results; a weak reference
    # cannot be pickled, so the record arrives with the frame marked gone.
    import pickle

    fit = _quiet(sp.regress, "y ~ x", data=cross, weights="w")
    back = pickle.loads(pickle.dumps(fit))
    assert get_call_record(back).function == "regress"
    with pytest.raises(MethodIncompatibility, match="no longer exists"):
        sp.vcov(back, "hc1")
    direct = _quiet(sp.regress, "y ~ x", data=cross, weights="w", vce="hc1")
    _assert_identical(_quiet(sp.vcov, back, "hc1", data=cross), direct)
