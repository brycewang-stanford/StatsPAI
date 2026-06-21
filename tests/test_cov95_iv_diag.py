"""Coverage tests for statspai.iv.iv_diag internal branches.

Targets summary/to_frame branches (CLR/K/LTZ/bootstrap rows), serialisation
(to_excel/to_word/to_latex), vcov variants, wild cluster bootstrap, the
binary-endog LATE caveat, and iv_compare endog-name fallback paths.
"""

from __future__ import annotations

import os
import warnings

import numpy as np
import pandas as pd
import pytest

warnings.filterwarnings("ignore")

import importlib

import statspai as sp

_ivd = importlib.import_module("statspai.iv.iv_diag")


def _iv_df(n=400, seed=0, binary_endog=False):
    rng = np.random.default_rng(seed)
    z1 = rng.normal(size=n)
    z2 = rng.normal(size=n)
    x = rng.normal(size=n)
    eps = rng.normal(size=n)
    if binary_endog:
        latent = 0.8 * z1 + 0.6 * z2 + 0.5 * x + 0.6 * eps + rng.normal(0, 0.5, n)
        d = (latent > np.median(latent)).astype(float)
    else:
        d = 0.8 * z1 + 0.6 * z2 + 0.5 * eps + rng.normal(0, 0.5, n)
    y = 1.0 + 2.0 * d + 0.5 * x + eps
    cl = rng.integers(0, 8, size=n)
    return pd.DataFrame({"y": y, "d": d, "z1": z1, "z2": z2, "x": x, "cl": cl})


def test_iv_diag_full_bundle_summary_and_frame():
    df = _iv_df(seed=1)
    r = sp.iv.iv_diag(
        df,
        y="y",
        endog="d",
        instruments=["z1", "z2"],
        exog=["x"],
        n_boot=120,
        boot_methods=("pairs", "wild"),
        include_clr_ci=True,
        include_k_ci=True,
        ltz_gamma_sd=0.05,
        grid_size=81,
        random_state=3,
    )
    # to_frame exercises pairs/wild/tF/AR/CLR/K/LTZ/OLS rows
    frame = r.to_frame()
    assert isinstance(frame, pd.DataFrame)
    estimators = set(frame["estimator"])
    assert "2SLS (pairs bootstrap)" in estimators
    assert "2SLS (wild bootstrap)" in estimators
    assert "Moreira CLR set" in estimators
    assert "Kleibergen K set" in estimators
    assert "CHR plausibly-exogenous LTZ" in estimators
    # summary text exercises the same conditional blocks
    s = r.summary()
    assert "Moreira CLR" in s
    assert "Kleibergen K" in s
    assert "Plausibly exogenous" in s
    assert "Pairs bootstrap" in s
    assert "Wild bootstrap" in s
    assert r.bootstrap_se_pairs is not None
    assert r.bootstrap_se_wild is not None


def test_iv_diag_str_instruments_and_exog():
    df = _iv_df(seed=2)
    # str instrument + str exog (lines 758-762 normalisation)
    r = sp.iv.iv_diag(df, y="y", endog="d", instruments="z1", exog="x", n_boot=0)
    assert r.instruments == ["z1"]
    assert r.exog == ["x"]
    assert r.bootstrap_ci_pairs is None


def test_iv_diag_binary_endog_caveat_present():
    df = _iv_df(seed=4, binary_endog=True)
    r = sp.iv.iv_diag(
        df, y="y", endog="d", instruments=["z1", "z2"], exog=["x"], n_boot=0
    )
    assert r.tsls_late_caveat is not None
    assert "binary endogenous" in r.tsls_late_caveat
    s = r.summary()
    assert "Interpretation caveat" in s


def test_iv_diag_caveat_none_when_no_exog():
    df = _iv_df(seed=5, binary_endog=True)
    r = sp.iv.iv_diag(df, y="y", endog="d", instruments=["z1", "z2"], n_boot=0)
    assert r.tsls_late_caveat is None


def test_check_caveat_nonbinary_endog_returns_none():
    # endog has >2 unique -> returns None (line 638-639), continuous endog
    df = _iv_df(seed=6, binary_endog=False)
    assert _ivd._check_tsls_late_caveat(df, "d", ["x"], ["z1", "z2"]) is None
    # 2 unique but not {0,1} -> line 641 branch
    df2 = df.copy()
    df2["d2"] = np.where(df2["d"] > df2["d"].median(), 5.0, 3.0)
    assert _ivd._check_tsls_late_caveat(df2, "d2", ["x"], ["z1", "z2"]) is None


def test_se_2sls_robust_vcov_variants():
    df = _iv_df(seed=7)
    Y = df["y"].to_numpy()
    D = df["d"].to_numpy()
    Z = df[["z1", "z2"]].to_numpy()
    W = np.column_stack([np.ones(len(df)), df["x"].to_numpy()])
    beta, _, _ = _ivd._two_sls_point(Y, D, Z, W)
    se_hc0 = _ivd._se_2sls_robust(Y, D, Z, W, beta, vcov="HC0")
    se_hc1 = _ivd._se_2sls_robust(Y, D, Z, W, beta, vcov="HC1")
    se_other = _ivd._se_2sls_robust(Y, D, Z, W, beta, vcov="weird")  # else->scale 1
    se_classic = _ivd._se_2sls_robust(Y, D, Z, W, beta, vcov="classic")
    for s in (se_hc0, se_hc1, se_other, se_classic):
        assert np.isfinite(s) and s > 0
    # HC1 inflates relative to HC0
    assert se_hc1 >= se_hc0


def test_wild_cluster_bootstrap_path():
    df = _iv_df(seed=8)
    Y = df["y"].to_numpy()
    D = df["d"].to_numpy()
    Z = df[["z1", "z2"]].to_numpy()
    W = np.column_stack([np.ones(len(df)), df["x"].to_numpy()])
    cl = df["cl"].to_numpy()
    rng = np.random.default_rng(0)
    se, (lo, hi), B = _ivd._bootstrap_se(
        Y, D, Z, W, n_boot=200, cluster=cl, rng=rng, method="wild", alpha=0.05
    )
    assert B > 0
    assert lo <= hi


def test_bootstrap_se_unknown_method_raises():
    df = _iv_df(seed=9)
    Y = df["y"].to_numpy()
    D = df["d"].to_numpy()
    Z = df[["z1", "z2"]].to_numpy()
    W = np.ones((len(df), 1))
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError):
        _ivd._bootstrap_se(
            Y, D, Z, W, n_boot=10, cluster=None, rng=rng, method="nope", alpha=0.05
        )


def test_bootstrap_se_insufficient_successes_returns_nan():
    # n_boot small enough that successes < max(50, n_boot//4) triggers nan path
    df = _iv_df(n=120, seed=10)
    Y = df["y"].to_numpy()
    D = df["d"].to_numpy()
    Z = df[["z1", "z2"]].to_numpy()
    W = np.ones((len(df), 1))
    rng = np.random.default_rng(0)
    se, (lo, hi), B = _ivd._bootstrap_se(
        Y, D, Z, W, n_boot=10, cluster=None, rng=rng, method="pairs", alpha=0.05
    )
    assert np.isnan(se)
    assert np.isnan(lo) and np.isnan(hi)
    assert B == 10


def test_iv_diag_cluster_bootstrap_str():
    df = _iv_df(seed=11)
    r = sp.iv.iv_diag(
        df,
        y="y",
        endog="d",
        instruments=["z1", "z2"],
        exog=["x"],
        cluster="cl",
        n_boot=120,
        boot_methods=("pairs",),
        random_state=1,
    )
    assert r.bootstrap_ci_pairs is not None


def test_iv_diag_to_excel_word_latex(tmp_path):
    df = _iv_df(seed=12)
    r = sp.iv.iv_diag(
        df, y="y", endog="d", instruments=["z1", "z2"], exog=["x"], n_boot=0
    )
    xlsx = tmp_path / "out.xlsx"
    r.to_excel(str(xlsx))
    assert xlsx.exists()

    tex = r.to_latex(caption="cap", label="lab")
    assert "\\caption{cap}" in tex
    assert "\\label{lab}" in tex
    # plain latex without caption/label
    tex2 = r.to_latex()
    assert "tabular" in tex2

    docx = tmp_path / "out.docx"
    r.to_word(str(docx), title="IV bundle")
    assert docx.exists()

    assert isinstance(r.to_dict(), dict)


def test_iv_compare_basic():
    df = _iv_df(seed=13)
    out = sp.iv.iv_compare(
        formula="y ~ (d ~ z1 + z2) + x",
        data=df,
        methods=("2sls", "liml"),
        endog_name="d",
    )
    assert isinstance(out, pd.DataFrame)
    assert set(out["method"]) == {"2sls", "liml"}
    ok = out[out["status"] == "ok"]
    assert len(ok) >= 1
    assert np.all(np.isfinite(ok["estimate"].to_numpy()))


def test_iv_compare_endog_autoresolve_from_formula():
    df = _iv_df(seed=14)
    # no endog_name -> resolved from formula parse
    out = sp.iv.iv_compare(
        formula="y ~ (d ~ z1 + z2) + x",
        data=df,
        methods=("2sls",),
    )
    assert "2sls" in set(out["method"])


def test_iv_compare_jive_endog_fallback():
    # jive result class has endog FIRST and may not match canonical name;
    # exercises diagnostics-key / last-resort param fallback (1043-1061).
    df = _iv_df(seed=15)
    out = sp.iv.iv_compare(
        formula="y ~ (d ~ z1 + z2) + x",
        data=df,
        methods=("2sls", "jive"),
    )
    assert set(out["method"]) == {"2sls", "jive"}


def test_iv_compare_endog_name_mismatch_diag_fallback():
    # endog_name not in params index -> forces the "First-stage F (<endog>)"
    # diagnostics-key parse fallback (lines 1043-1049) which recovers 'd'.
    df = _iv_df(seed=19)
    out = sp.iv.iv_compare(
        formula="y ~ (d ~ z1 + z2) + x",
        data=df,
        methods=("2sls",),
        endog_name="not_a_real_col",
    )
    row = out[out["method"] == "2sls"].iloc[0]
    assert row["status"] == "ok"
    assert np.isfinite(row["estimate"])


def test_iv_diag_to_word_with_caveat(tmp_path):
    # binary endog with covariates -> caveat -> to_word adds caveat paragraph
    df = _iv_df(seed=16, binary_endog=True)
    r = sp.iv.iv_diag(
        df, y="y", endog="d", instruments=["z1", "z2"], exog=["x"], n_boot=0
    )
    assert r.tsls_late_caveat is not None
    docx = tmp_path / "caveat.docx"
    r.to_word(str(docx))
    assert docx.exists()


def test_iv_diag_cluster_array_alignment():
    # pass cluster as an ndarray of full length -> alignment branch (466-475)
    df = _iv_df(seed=17)
    cl = df["cl"].to_numpy()
    r = sp.iv.iv_diag(
        df, y="y", endog="d", instruments=["z1", "z2"], exog=["x"], cluster=cl, n_boot=0
    )
    assert np.isfinite(r.se_2sls)


def test_iv_diag_cluster_array_wrong_length_raises():
    df = _iv_df(seed=18)
    with pytest.raises(ValueError):
        sp.iv.iv_diag(
            df,
            y="y",
            endog="d",
            instruments=["z1", "z2"],
            cluster=np.ones(len(df) + 3),
            n_boot=0,
        )


def test_iv_diag_weak_first_stage_tF_inf():
    # nearly-irrelevant instruments -> F < 3.84 -> tF critical = inf (802-803)
    rng = np.random.default_rng(20)
    n = 300
    z = rng.normal(size=n)
    eps = rng.normal(size=n)
    d = 0.01 * z + eps  # essentially no first stage
    y = 1.0 + 2.0 * d + eps
    df = pd.DataFrame({"y": y, "d": d, "z": z})
    r = sp.iv.iv_diag(df, y="y", endog="d", instruments="z", n_boot=0)
    assert not np.isfinite(r.tF_critical_value)
    lo, hi = r.tF_adjusted_ci
    assert not np.isfinite(lo) and not np.isfinite(hi)
