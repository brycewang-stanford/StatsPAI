"""Overdue removals and input checks decided for 1.41 (option scan round 5).

* ``sp.kan_dlate``, ``sp.SurrogateResult``, ``sp.qte(method=
  'quantile_regression')`` and ``sp.evalue(rare_outcome=)`` were deprecated
  with a stated removal version long past; they are gone and their old
  spellings fail with a message that names the replacement.
* ``sp.distributional_te`` bootstrap SEs use ``ddof=1``, as every other
  bootstrap SE in ``qte/`` does.
* ``sp.heckman`` refuses a selection indicator that is not 0/1.
* ``sp.ipcw`` with a missing covariate drops that row (warning, count
  recorded) instead of turning every weight into NaN.
* ``sp.absorb_ols`` warns that its multi-way ``cluster_df`` default moves to
  ``'min'`` in 1.42, and only when that default is in play.
"""

import importlib
import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import MethodIncompatibility

# --------------------------------------------------------------------- #
# Removals
# --------------------------------------------------------------------- #


def test_kan_dlate_is_gone_and_the_error_names_dist_iv():
    assert "kan_dlate" not in sp.list_functions()
    assert "kan_dlate" not in sp.__all__
    with pytest.raises(AttributeError, match="dist_iv"):
        sp.kan_dlate  # noqa: B018
    qte_mod = importlib.import_module("statspai.qte")
    assert not hasattr(qte_mod, "kan_dlate")


def test_surrogate_result_is_gone_but_the_estimators_remain():
    assert "SurrogateResult" not in sp.__all__
    with pytest.raises(AttributeError, match="CausalResult"):
        sp.SurrogateResult  # noqa: B018
    sur = importlib.import_module("statspai.surrogate")
    assert not hasattr(sur, "SurrogateResult")
    for name in ("surrogate_index", "long_term_from_short", "proximal_surrogate_index"):
        assert callable(getattr(sp, name))
        assert getattr(sp, name) is getattr(sur, name)


def _qte_frame(seed=7, n=400):
    rng = np.random.default_rng(seed)
    d = rng.integers(0, 2, n)
    return pd.DataFrame({"y": 1.0 * d + rng.normal(0, 1, n), "d": d})


def test_qte_quantile_regression_method_is_refused_with_the_replacement():
    with pytest.raises(MethodIncompatibility, match="quantile_regression") as ei:
        sp.qte(
            _qte_frame(),
            y="y",
            treatment="d",
            method="quantile_regression",
            quantiles=[0.5],
            n_boot=5,
        )
    assert "conditional_qr" in ei.value.recovery_hint
    assert "firpo_qte" in ei.value.recovery_hint


def test_qte_unknown_method_is_a_taxonomy_error_too():
    with pytest.raises(MethodIncompatibility, match="Unknown QTE method"):
        sp.qte(_qte_frame(), y="y", treatment="d", method="nope", n_boot=5)


def test_evalue_rare_outcome_keyword_is_gone():
    with pytest.raises(TypeError, match="rare_outcome"):
        sp.evalue(estimate=2.0, measure="OR", rare_outcome=True)
    with pytest.raises(TypeError, match="rare_outcome"):
        sp.evalue_from_result(
            sp.CausalResult(
                method="x",
                estimand="ATE",
                estimate=0.4,
                se=0.1,
                pvalue=0.01,
                ci=(0.2, 0.6),
                alpha=0.05,
                n_obs=100,
            ),
            measure="SMD",
            sd=1.0,
            rare_outcome=True,
        )
    # The keyword spelled ``rare`` is the one that stays.
    a = sp.evalue(estimate=2.0, measure="OR", rare=True)["evalue_estimate"]
    assert a == pytest.approx(2.0 + np.sqrt(2.0 * 1.0), rel=1e-12)


def test_evalue_rr_keeps_its_own_rare_outcome_keyword():
    # ``sp.evalue_rr`` documents ``rare_outcome`` itself; on the RR scale the
    # rare-outcome flag has no effect, so both spellings give 2 + sqrt(2).
    a = sp.evalue_rr(2.0, rr_lower=1.5, rr_upper=2.7, rare_outcome=True)
    b = sp.evalue_rr(2.0, rr_lower=1.5, rr_upper=2.7)
    assert a["evalue_estimate"] == b["evalue_estimate"]
    assert a["evalue_estimate"] == pytest.approx(2.0 + np.sqrt(2.0), rel=1e-12)


# --------------------------------------------------------------------- #
# distributional_te: bootstrap SE with ddof = 1
# --------------------------------------------------------------------- #


def test_distributional_te_bootstrap_se_is_the_ddof1_sd_of_the_draws(monkeypatch):
    dmod = importlib.import_module("statspai.qte.distributional")
    calls = []
    real = dmod._dte_ipw

    def recorder(*args, **kwargs):
        out = real(*args, **kwargs)
        calls.append(out)
        return out

    monkeypatch.setattr(dmod, "_dte_ipw", recorder)
    rng = np.random.default_rng(3)
    n = 300
    d = rng.integers(0, 2, n)
    df = pd.DataFrame({"y": 1.0 + 1.5 * d + rng.normal(0, 1, n), "d": d})
    res = sp.distributional_te(
        df,
        y="y",
        treatment="d",
        method="ipw",
        quantiles=[0.25, 0.5, 0.75],
        n_boot=12,
        seed=5,
        n_grid=30,
    )
    draws = calls[1:]  # the first call is the point estimate
    assert len(draws) == 12
    boot_dte = np.array([c["dte"] for c in draws])
    boot_qte = np.array([c["qte"] for c in draws])
    np.testing.assert_allclose(res.dte_se, boot_dte.std(axis=0, ddof=1), rtol=1e-12)
    np.testing.assert_allclose(res.qte_se, boot_qte.std(axis=0, ddof=1), rtol=1e-12)


# --------------------------------------------------------------------- #
# heckman: the selection indicator must be 0/1
# --------------------------------------------------------------------- #


def _heckman_frame(seed=1, n=400):
    rng = np.random.default_rng(seed)
    x1 = rng.normal(size=n)
    z1 = rng.normal(size=n)
    u = rng.normal(size=n)
    sel = ((0.3 + z1 + 0.5 * x1 + u) > 0).astype(int)
    y = 1 + 2 * x1 + 0.5 * u + rng.normal(size=n)
    return pd.DataFrame(
        {"y": np.where(sel == 1, y, np.nan), "x1": x1, "z1": z1, "s": sel}
    )


@pytest.mark.parametrize("method", ["twostep", "ml"])
@pytest.mark.parametrize(
    "recode", [lambda s: s + 1, lambda s: 2 * s, lambda s: s.where(s.index != 0, 3)]
)
def test_heckman_refuses_a_selection_indicator_that_is_not_0_1(method, recode):
    h = _heckman_frame()
    with pytest.raises(MethodIncompatibility, match="must be 0/1") as ei:
        sp.heckman(
            h.assign(s=recode(h["s"])),
            y="y",
            x=["x1"],
            z=["x1", "z1"],
            select="s",
            method=method,
        )
    assert ei.value.recovery_hint


def test_heckman_bool_and_missing_selection_codes_are_still_valid():
    h = _heckman_frame()
    ref = sp.heckman(h, y="y", x=["x1"], z=["x1", "z1"], select="s")
    as_bool = sp.heckman(
        h.assign(s=h["s"].astype(bool)), y="y", x=["x1"], z=["x1", "z1"], select="s"
    )
    assert as_bool.estimate == ref.estimate
    # A missing selection code drops the row (the probit never saw it).
    h2 = h.assign(s=h["s"].astype(float))
    h2.loc[0, "s"] = np.nan
    sp.heckman(h2, y="y", x=["x1"], z=["x1", "z1"], select="s")


# --------------------------------------------------------------------- #
# ipcw: missing covariates
# --------------------------------------------------------------------- #


def _ipcw_frame(seed=0, n=300):
    rng = np.random.default_rng(seed)
    b = rng.normal(size=n)
    p = 1 / (1 + np.exp(-(0.5 - 0.5 * b)))
    return pd.DataFrame(
        {
            "time": rng.exponential(5, n),
            "event": rng.binomial(1, p),
            "age": rng.normal(50, 10, n),
            "b": b,
            "v": rng.normal(size=n),
        }
    )


@pytest.mark.parametrize("method", ["pooled_logistic", "cox_ph"])
@pytest.mark.parametrize("stabilize", [True, False])
def test_ipcw_missing_covariate_drops_the_row_and_leaves_the_rest_intact(
    method, stabilize
):
    df = _ipcw_frame()
    bad = [5, 17, 40]
    df_na = df.copy()
    df_na.loc[bad[:2], "age"] = np.nan
    df_na.loc[bad[2], "v"] = np.nan  # numerator covariate
    kw = dict(
        time="time",
        event="event",
        censor_covariates=["age", "b"],
        treatment_covariates=["v"],
        method=method,
        stabilize=stabilize,
    )
    # ``v`` enters only the numerator of the stabilized weights.
    dropped = bad if stabilize else bad[:2]
    with pytest.warns(UserWarning, match=f"{len(dropped)} row"):
        res = sp.ipcw(df_na, **kw)
    # Same fit as on the complete cases, aligned back to the input rows.
    keep = ~df.index.isin(dropped)
    ref = sp.ipcw(df.loc[keep].reset_index(drop=True), **kw)
    assert res.weights.shape == (len(df),)
    assert np.isnan(res.weights[dropped]).all()
    np.testing.assert_array_equal(res.weights[keep], ref.weights)
    np.testing.assert_array_equal(res.fitted_hazards[keep], ref.fitted_hazards)
    assert res.summary_stats == ref.summary_stats
    assert res.n_dropped_missing == len(dropped)
    assert ref.n_dropped_missing == 0


def test_ipcw_unstabilized_ignores_missing_values_in_unused_numerator_columns():
    df = _ipcw_frame()
    df.loc[3, "v"] = np.nan
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        res = sp.ipcw(
            df,
            time="time",
            event="event",
            censor_covariates=["age", "b"],
            stabilize=False,
            treatment_covariates=["v"],
        )
    assert res.n_dropped_missing == 0
    assert np.isfinite(res.weights).all()


# --------------------------------------------------------------------- #
# absorb_ols: the multi-way cluster_df default
# --------------------------------------------------------------------- #


def _absorb_inputs(seed=2, n=400):
    rng = np.random.default_rng(seed)
    firm = rng.integers(0, 12, n)
    year = rng.integers(0, 6, n)
    st = rng.integers(0, 5, n)
    X = rng.normal(size=(n, 2))
    y = (
        X @ np.array([1.0, -0.5])
        + rng.normal(size=12)[firm]
        + rng.normal(size=6)[year]
        + rng.normal(size=n)
    )
    return y, X, pd.DataFrame({"firm": firm, "year": year}), firm, st


def test_absorb_ols_multiway_default_is_min_since_1_42():
    y, X, fe, firm, st = _absorb_inputs()
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        out = sp.absorb_ols(y, X, fe, cluster=[firm, st])
        pt = sp.absorb_ols(y, X, fe, cluster=[firm, st], cluster_df="per_term")
        mn = sp.absorb_ols(y, X, fe, cluster=[firm, st], cluster_df="min")
    np.testing.assert_array_equal(out["vcov"], mn["vcov"])
    assert not np.allclose(mn["vcov"], pt["vcov"], rtol=1e-6)  # the two differ


@pytest.mark.parametrize("which", ["none", "one", "one_list"])
def test_absorb_ols_default_does_not_warn_without_multiway_clustering(which):
    y, X, fe, firm, st = _absorb_inputs()
    cl = {"none": None, "one": firm, "one_list": [firm]}[which]
    with warnings.catch_warnings():
        warnings.simplefilter("error")
        sp.absorb_ols(y, X, fe, cluster=cl)


def test_hdfe_ols_never_triggers_the_absorb_ols_warning():
    y, X, fe, firm, st = _absorb_inputs()
    df = pd.DataFrame(
        {
            "y": y,
            "x1": X[:, 0],
            "x2": X[:, 1],
            "firm": firm,
            "year": fe["year"],
            "st": st,
        }
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", FutureWarning)
        sp.hdfe_ols("y ~ x1 + x2 | firm + year", data=df, cluster=["firm", "st"])
        sp.hdfe_ols(
            "y ~ x1 + x2 | firm + year",
            data=df,
            cluster=["firm", "st"],
            cluster_df="per_term",
        )


def test_absorb_ols_rejects_an_unknown_cluster_df():
    y, X, fe, firm, st = _absorb_inputs()
    with pytest.raises(MethodIncompatibility, match="cluster_df"):
        sp.absorb_ols(y, X, fe, cluster=[firm, st], cluster_df="max")
