"""``sp.feols`` / ``sp.fepois`` / ``sp.feglm`` on StatsPAI's own engine.

``backend='auto'`` (the default) fits plain calls natively and delegates
the rest to pyfixest. These tests pin (1) that the native fit reproduces
the delegate's numbers and result layout, (2) which calls go where, and
(3) the conventions that are easy to get wrong: missing-value rows,
singletons, separation, refusals the delegate resolves.
"""

import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.fixest import _native


@pytest.fixture(scope="module")
def panel():
    rng = np.random.default_rng(11)
    n = 3000
    df = pd.DataFrame(
        dict(
            firm=rng.integers(0, 60, n),
            year=rng.integers(0, 8, n),
            g=rng.integers(0, 30, n),
            x1=rng.normal(size=n),
            x2=rng.normal(size=n),
            w=rng.uniform(0.5, 2.0, n),
        )
    )
    # 1% singleton firms, so fixef_rm matters
    df["firm"] = np.where(rng.random(n) < 0.01, 1000 + np.arange(n), df.firm)
    df["y"] = df.x1 - df.x2 + 0.1 * (df.firm % 7) + rng.normal(size=n)
    df["c"] = rng.poisson(np.exp(0.2 * df.x1 + 0.1 * (df.firm % 5)))
    df.loc[df.firm % 60 == 3, "c"] = 0  # all-zero firms: separation
    df.loc[5, "x1"] = np.nan  # a row fixest drops
    return df


def _quiet(fn, *a, **k):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return fn(*a, **k)


VCOVS = [None, "iid", "hetero", {"CRV1": "g"}, {"CRV1": "firm"}]


@pytest.mark.parametrize("fixef_rm", ["none", "singleton"])
@pytest.mark.parametrize("weights", [None, "w"])
@pytest.mark.parametrize("vcov", VCOVS, ids=str)
@pytest.mark.parametrize(
    "fml", ["y ~ x1 + x2 | firm + year", "y ~ x1 + x2 | firm", "y ~ x1 + x2"]
)
def test_feols_native_reproduces_the_delegate(panel, fml, vcov, weights, fixef_rm):
    pytest.importorskip("pyfixest")
    kw = dict(vcov=vcov, weights=weights, fixef_rm=fixef_rm)
    a = _quiet(sp.feols, fml, data=panel, **kw)
    b = _quiet(sp.feols, fml, data=panel, backend="pyfixest", **kw)
    assert a.model_info["backend"] == "statspai-native"
    assert list(a.params.index) == [str(i) for i in b.params.index]
    np.testing.assert_allclose(a.params, b.params, rtol=1e-8)
    np.testing.assert_allclose(a.std_errors, b.std_errors, rtol=1e-6)
    np.testing.assert_allclose(a.pvalues, b.pvalues, rtol=1e-5, atol=1e-12)
    assert a.data_info["nobs"] == b.data_info["nobs"]
    assert a.data_info["df_resid"] == b.data_info["df_resid"]
    assert a.data_info.get("df_inference") == b.data_info.get("df_inference")
    for key, value in b.diagnostics.items():
        if np.isnan(value):
            assert np.isnan(a.diagnostics[key])
        else:
            assert a.diagnostics[key] == pytest.approx(value, rel=1e-8)
    # pyfixest demeans to its own tolerance: weighted two-way residuals
    # differ by ~2e-8 absolute (outcome scale ~1).
    np.testing.assert_allclose(
        a.data_info["residuals"], b.data_info["residuals"], atol=1e-6
    )


@pytest.mark.parametrize("weights", [None, "w"])
@pytest.mark.parametrize("vcov", ["hetero", {"CRV1": "g"}], ids=str)
@pytest.mark.parametrize("fml", ["c ~ x1 + x2 | firm + year", "c ~ x1 | firm"])
def test_fepois_native_standard_errors_sit_at_the_optimum(panel, fml, vcov, weights):
    """Same estimator as the delegate; the native fit runs IRLS to 1e-10,
    so its standard errors are within 1e-8 of the exact optimum (pyfixest's
    1e-8 stop leaves them up to ~1e-5 away on unweighted fits)."""
    pytest.importorskip("pyfixest")
    from statspai.fast.fepois import _fepois_fit

    a = _quiet(sp.fepois, fml, data=panel, vcov=vcov, weights=weights)
    b = _quiet(
        sp.fepois, fml, data=panel, vcov=vcov, weights=weights, backend="pyfixest"
    )
    assert a.model_info["backend"] == "statspai-native"
    assert a.data_info["nobs"] == b.data_info["nobs"]
    np.testing.assert_allclose(a.params, b.params, rtol=1e-6)
    v, cl = ("hc1", None) if vcov == "hetero" else ("cr1", "g")
    exact, _ = _fepois_fit(
        fml,
        panel.dropna(),
        vcov=v,
        cluster=cl,
        weights=weights,
        tol=1e-16,
        fe_tol=1e-14,
        maxiter=300,
        drop_singletons=False,
        note_weights=False,
    )
    se_exact = np.sqrt(np.diag(exact.vcov_matrix))
    err_native = np.max(np.abs(a.std_errors.to_numpy() - se_exact) / se_exact)
    assert err_native <= 1e-8


def test_fepois_warns_on_separation(panel):
    with pytest.warns(UserWarning, match="separation"):
        sp.fepois("c ~ x1 | firm", data=panel)


def test_missing_rows_are_dropped_as_fixest_does(panel):
    fit = _quiet(sp.feols, "y ~ x1 + x2 | firm", data=panel)
    assert fit.data_info["nobs"] == len(panel) - 1
    assert fit.model_info["n_missing_dropped"] == 1
    clean = _quiet(sp.feols, "y ~ x1 + x2 | firm", data=panel.dropna())
    np.testing.assert_array_equal(fit.params, clean.params)


@pytest.mark.parametrize(
    "fml, kw, reason",
    [
        ("y ~ x1 | firm | x2 ~ g", {}, "formula"),
        ("y ~ i(year) | firm", {}, "formula"),
        ("y ~ x1 + x2 | firm", {"vcov": "HC3"}, "vcov"),
        ("y ~ x1 + x2 | firm", {"vcov": {"CRV1": "g + year + firm"}}, "vcov"),
        ("y ~ x1 + x2 | firm", {"ssc": object()}, "ssc"),
    ],
)
def test_calls_beyond_the_native_grammar_are_delegated(panel, fml, kw, reason):
    p = _native.plan(
        fml,
        panel,
        vcov=kw.get("vcov"),
        cluster=None,
        weights=None,
        ssc=kw.get("ssc"),
        fixef_rm="none",
        extra={},
    )
    assert isinstance(p, str) and reason in p


def test_absorbed_regressor_falls_back_to_pyfixest(panel):
    """fixest drops a regressor the effects absorb; the native fit refuses
    it, so ``'auto'`` hands the call to pyfixest, which drops it."""
    pytest.importorskip("pyfixest")
    df = panel.dropna().assign(fconst=lambda d: d.firm % 3)
    fit = _quiet(sp.feols, "y ~ x1 + fconst | firm", data=df)
    assert fit.model_info["backend"] == "pyfixest"
    with pytest.raises(sp.NumericalInstability):
        sp.feglm(
            "y ~ x1 + fconst | firm",
            data=df.assign(y=(df.y > 0).astype(int)),
            family="logit",
            backend="native",
        )


def test_native_backend_refuses_what_it_cannot_fit(panel):
    with pytest.raises(sp.MethodIncompatibility, match="cannot fit"):
        sp.fepois("c ~ i(year) | firm", data=panel, backend="native")
    with pytest.raises(sp.MethodIncompatibility, match="cannot fit"):
        sp.feglm("y ~ x1 | firm", data=panel, family="gaussian", backend="native")


def test_feglm_gaussian_stays_with_pyfixest(panel):
    pytest.importorskip("pyfixest")
    fit = _quiet(sp.feglm, "y ~ x1 | firm", data=panel.dropna(), family="gaussian")
    assert fit.model_info["backend"] == "pyfixest"


def test_provenance_hash_taken_alongside_the_fit_is_the_same():
    """The input is fingerprinted on a second thread for large frames; the
    digest must equal the one taken inline."""
    rng = np.random.default_rng(0)
    n = 120_000  # x 3 columns > _HashAhead.THREAD_CELLS
    df = pd.DataFrame(
        dict(y=rng.normal(size=n), x=rng.normal(size=n), f=rng.integers(0, 50, n))
    )
    fit = sp.feols("y ~ x | f", data=df)
    assert fit._provenance.data_hash == sp.compute_data_hash(df)
    small = df.head(100)
    assert sp.feols("y ~ x | f", data=small)._provenance.data_hash == (
        sp.compute_data_hash(small)
    )


# --------------------------------------------------------------------------- #
#  Two-way clustering, i() terms and IV on the native engine
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("weights", [None, "w"])
@pytest.mark.parametrize("cl", ["g + year", "firm + year", "g + firm"])
def test_twoway_clustering_reproduces_the_delegate(panel, cl, weights):
    pytest.importorskip("pyfixest")
    kw = dict(vcov={"CRV1": cl}, weights=weights)
    a = _quiet(sp.feols, "y ~ x1 + x2 | firm + year", data=panel, **kw)
    b = _quiet(
        sp.feols, "y ~ x1 + x2 | firm + year", data=panel, backend="pyfixest", **kw
    )
    assert a.model_info["backend"] == "statspai-native"
    np.testing.assert_allclose(a.params, b.params, rtol=1e-8)
    np.testing.assert_allclose(a.std_errors, b.std_errors, rtol=1e-6)
    assert a.data_info["df_inference"] == b.data_info["df_inference"]
    assert a.data_info["df_resid"] == b.data_info["df_resid"]


@pytest.mark.parametrize(
    "fml",
    [
        "y ~ i(ev, ref=-1) | firm + year",
        "y ~ x1 + i(ev, ref=-1) | firm",
        "y ~ i(ev, x2, ref=-1) | firm + year",
        "y ~ i(lab, ref='c') + x1 | firm",
    ],
)
def test_i_terms_run_natively_with_the_delegates_names(panel, fml):
    pytest.importorskip("pyfixest")
    rng = np.random.default_rng(3)
    df = panel.dropna().assign(
        ev=rng.integers(-3, 4, len(panel) - 1),
        lab=np.array(list("abcde"))[rng.integers(0, 5, len(panel) - 1)],
    )
    a = _quiet(sp.feols, fml, data=df, vcov={"CRV1": "firm"})
    b = _quiet(sp.feols, fml, data=df, vcov={"CRV1": "firm"}, backend="pyfixest")
    assert a.model_info["backend"] == "statspai-native"
    assert list(a.params.index) == [str(i) for i in b.params.index]
    np.testing.assert_allclose(a.params, b.params, rtol=1e-8)
    np.testing.assert_allclose(a.std_errors, b.std_errors, rtol=1e-6)


@pytest.mark.parametrize("vcov", ["iid", "hetero", {"CRV1": "g"}], ids=str)
def test_iv_with_fixed_effects_runs_natively(panel, vcov):
    pytest.importorskip("pyfixest")
    rng = np.random.default_rng(5)
    df = panel.dropna()
    df = df.assign(z=df.x1 + rng.normal(size=len(df)))
    fml = "y ~ x2 | firm + year | x1 ~ z"
    a = _quiet(sp.feols, fml, data=df, vcov=vcov, fixef_rm="singleton")
    b = _quiet(
        sp.feols, fml, data=df, vcov=vcov, fixef_rm="singleton", backend="pyfixest"
    )
    assert a.model_info["backend"] == "statspai-native"
    assert list(a.params.index) == ["x2", "x1"]
    np.testing.assert_allclose(a.params, b.params, rtol=1e-8)
    np.testing.assert_allclose(a.std_errors, b.std_errors, rtol=1e-6)
    # with singletons dropped both first stages use the same sample
    assert a.diagnostics["First-stage F (x1)"] == pytest.approx(
        b.diagnostics["First-stage F (x1)"], rel=1e-6
    )
    for key in ("Hausman F-stat", "Hausman p-value"):
        assert a.diagnostics[key] == pytest.approx(b.diagnostics[key], rel=1e-6)


@pytest.mark.parametrize(
    "fml",
    [
        "y ~ x1 | csw0(firm, year)",
        "y ~ csw(x1, x2) | firm",
        "y + y2 ~ x1 + x2 | firm",
        "y + y2 ~ csw(x1, x2) | sw0(firm, year)",
        "y ~ x1 + x2 | mvsw(firm, year)",
    ],
)
def test_multiple_estimation_runs_natively_in_the_delegates_order(panel, fml):
    pytest.importorskip("pyfixest")
    df = panel.dropna().assign(y2=lambda d: d.x2 + d.y)
    a = _quiet(sp.feols, fml, data=df, vcov={"CRV1": "g"})
    b = _quiet(sp.feols, fml, data=df, vcov={"CRV1": "g"}, backend="pyfixest")
    assert isinstance(a, list) and len(a) == len(b)
    for r, q in zip(a, b):
        assert r.model_info["backend"] == "statspai-native"
        assert list(r.params.index) == [str(i) for i in q.params.index]
        np.testing.assert_allclose(r.params, q.params, rtol=1e-8)
        np.testing.assert_allclose(r.std_errors, q.std_errors, rtol=1e-6)


def test_multiple_estimation_is_all_or_nothing(panel):
    """A zero step on the regressors (``y ~ 1 | firm``) has no native fit,
    so the whole list comes from the delegate, never a mix."""
    pytest.importorskip("pyfixest")
    out = _quiet(sp.feols, "y ~ csw0(x1, x2) | firm", data=panel.dropna())
    assert {r.model_info["backend"] for r in out} == {"pyfixest"}


# --------------------------------------------------------------------------- #
#  The fixef_rm default (singletons) changes in 1.40
# --------------------------------------------------------------------------- #


@pytest.mark.parametrize("backend", ["auto", "pyfixest"])
def test_kept_singletons_announce_the_default_change(panel, backend):
    if backend == "pyfixest":
        pytest.importorskip("pyfixest")
    df = panel.dropna()
    n_single = int(
        (df.firm.map(df.firm.value_counts()) == 1).sum()
    )  # firm-only model: singletons of the one effect
    assert n_single > 0
    with pytest.warns(
        FutureWarning, match=rf"{n_single} observations .* singletons"
    ) as rec:
        sp.feols("y ~ x1 | firm", data=df, backend=backend)
    assert rec[0].filename == __file__  # points at the caller, not statspai


@pytest.mark.parametrize("fixef_rm", ["none", "singleton"])
def test_an_explicit_fixef_rm_is_silent(panel, fixef_rm):
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        sp.feols("y ~ x1 | firm", data=panel.dropna(), fixef_rm=fixef_rm)
        sp.fepois("c ~ x1 | firm", data=panel.dropna(), fixef_rm=fixef_rm)
