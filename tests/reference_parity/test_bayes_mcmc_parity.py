"""MCMC diagnostics and Bayesian model averaging against R on committed files.

Everything here is a deterministic function of its input, so it is compared
digit for digit with the R packages an introductory Bayesian econometrics
course uses (Ramirez-Hassan 2026): ``coda`` for the convergence diagnostics,
``BMA`` for model averaging by BIC and ``BMS`` for model averaging under a
g-prior.

Inputs: ``_fixtures/bayes_mcmc_chains.csv``, ``bayes_mcmc_multichain.csv``
and ``bayes_bma.csv`` (synthetic, ``_generate_bayes_mcmc_data.py``).
Reference: ``bayes_mcmc_R.json`` (R 4.5.2, coda 0.19-4.1, BMA 3.18.21,
BMS 0.3.5; ``_generate_bayes_mcmc_R.R``), stored as the program wrote it.

Tolerances. ``EXACT`` (1e-9 relative) where both sides evaluate the same
closed form. ``ITER`` (1e-4) only for GLM standard errors and the gamma
family in ``bic.glm``: R's ``glm`` stops its iteration at a relative change
of 1e-8 in the deviance and builds the covariance from the weights of the
iteration before the last; the coefficients of the canonical-link models
still agree to 1e-7.

Three places where the reference prints something other than the quantity:

* ``raftery.diag`` rounds the dependence factor to three significant digits.
* ``bicreg`` computes BIC from an R-squared rounded to five decimals. The
  test shows that this is the whole difference: rounding our R-squared the
  same way reproduces its BIC to 1e-9, and its posterior probabilities
  applied to our per-model estimates reproduce its averaged coefficients.
* ``gelman.diag`` puts the number of parameters where Brooks and Gelman
  (1998) have the number of chains in the multivariate factor. The test
  rebuilds coda's number from the same eigenvalue.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import statspai as sp

FIX = Path(__file__).parent / "_fixtures"
EXACT = 1e-9
ITER = 1e-4
XN = [f"x{j}" for j in range(1, 10)]
RHS = " + ".join(XN)


def rel(a, b) -> float:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    return float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-300)))


def frame(records) -> pd.DataFrame:
    return pd.DataFrame(records)


@pytest.fixture(scope="module")
def R() -> dict:
    return json.loads((FIX / "bayes_mcmc_R.json").read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def chains() -> dict:
    full = pd.read_csv(FIX / "bayes_mcmc_chains.csv")
    return {"long": full, "short": full.iloc[:1237, :4]}


@pytest.fixture(scope="module")
def multichain() -> list:
    d = pd.read_csv(FIX / "bayes_mcmc_multichain.csv")
    return [g[["a", "b", "c"]].reset_index(drop=True) for _, g in d.groupby("chain")]


@pytest.fixture(scope="module")
def bma_data() -> pd.DataFrame:
    return pd.read_csv(FIX / "bayes_bma.csv")


# --------------------------------------------------------------------------
# coda
# --------------------------------------------------------------------------


@pytest.mark.parametrize("which", ["long", "short"])
def test_summary_matches_coda(R, chains, which):
    ref, x = R[which], chains[which]
    s = sp.mcmc_summary(x)
    st = frame(ref["stats"])
    assert rel(s["mean"], st["Mean"]) < EXACT
    assert rel(s["sd"], st["SD"]) < EXACT
    assert rel(s["naive_se"], st["Naive SE"]) < EXACT
    assert rel(s["ts_se"], st["Time-series SE"]) < EXACT
    q = s[["q2.5", "q25", "q50", "q75", "q97.5"]].to_numpy()
    assert rel(q, frame(ref["quant"]).to_numpy()) < EXACT
    assert rel(sp.mcmc_ess(x), ref["ess"]) < EXACT
    assert rel(s["ess"], ref["ess"]) < EXACT
    # the long-run variance behind both
    assert rel(s["ts_se"] ** 2 * len(x), ref["spec0"]) < EXACT


@pytest.mark.parametrize("which", ["long", "short"])
def test_geweke_matches_coda(R, chains, which):
    ref, x = R[which], chains[which]
    assert rel(sp.geweke_diag(x).table["z"], ref["geweke"]) < EXACT
    z = sp.geweke_diag(x, frac1=0.2, frac2=0.3).table["z"]
    assert rel(z, ref["geweke_23"]) < EXACT


@pytest.mark.parametrize("which", ["long", "short"])
@pytest.mark.parametrize(
    "key, kwargs", [("heidel", {}), ("heidel_strict", {"eps": 0.02, "pvalue": 0.2})]
)
def test_heidelberger_welch_matches_coda(R, chains, which, key, kwargs):
    ref = frame(R[which][key])
    t = sp.heidel_diag(chains[which], **kwargs).table
    assert (t["stationary"].astype(int).to_numpy() == ref["stest"].to_numpy()).all()
    ok = t["stationary"].to_numpy(dtype=bool)
    assert ok.any()
    assert (
        t.loc[ok, "start"].to_numpy(float) == ref.loc[ok, "start"].to_numpy(float)
    ).all()
    assert rel(t.loc[ok, "p_value"], ref.loc[ok, "pvalue"]) < EXACT
    assert rel(t.loc[ok, "mean"], ref.loc[ok, "mean"]) < EXACT
    assert rel(t.loc[ok, "halfwidth"], ref.loc[ok, "halfwidth"]) < EXACT
    ours = t.loc[ok, "halfwidth_passed"].astype(int).to_numpy()
    assert (ours == ref.loc[ok, "htest"].to_numpy(int)).all()


@pytest.mark.parametrize("which", ["long", "short"])
def test_hpd_matches_coda(R, chains, which):
    ref, x = R[which], chains[which]
    assert rel(sp.hpd_interval(x).to_numpy(), frame(ref["hpd95"]).to_numpy()) < EXACT
    got = sp.hpd_interval(x, prob=0.8).to_numpy()
    assert rel(got, frame(ref["hpd80"]).to_numpy()) < EXACT


@pytest.mark.parametrize(
    "which, key, kwargs",
    [
        ("long", "raftery_default", {}),
        ("long", "raftery_median", {"q": 0.5, "r": 0.05, "s": 0.95}),
        (
            "long",
            "raftery_decile",
            {"q": 0.1, "r": 0.02, "s": 0.9, "converge_eps": 0.01},
        ),
        ("short", "raftery_median", {"q": 0.5, "r": 0.05, "s": 0.95}),
        (
            "short",
            "raftery_decile",
            {"q": 0.1, "r": 0.02, "s": 0.9, "converge_eps": 0.01},
        ),
    ],
)
def test_raftery_lewis_matches_coda(R, chains, which, key, kwargs):
    ref = frame(R[which][key])
    t = sp.raftery_diag(chains[which], **kwargs).table
    assert (t["burnin"].to_numpy() == ref["M"].to_numpy()).all()
    assert (t["total"].to_numpy() == ref["N"].to_numpy()).all()
    assert (t["n_min"].to_numpy() == ref["Nmin"].to_numpy()).all()
    # coda prints the dependence factor to three significant digits
    ours = np.array([float(f"{v:.3g}") for v in t["dependence_factor"]])
    assert rel(ours, ref["I"]) < EXACT


def test_raftery_lewis_refuses_a_chain_shorter_than_nmin(chains):
    with pytest.raises(sp.DataInsufficient, match="3746"):
        sp.raftery_diag(chains["short"])


def test_gelman_rubin_matches_coda(R, multichain):
    ref = R["gelman"]
    g = sp.gelman_rubin(multichain)
    assert rel(g.table.to_numpy(), frame(ref["psrf"]).to_numpy()) < EXACT
    g90 = sp.gelman_rubin(multichain, confidence=0.9)
    assert rel(g90.table.to_numpy(), frame(ref["psrf90"]).to_numpy()) < EXACT
    gb = sp.gelman_rubin(multichain, autoburnin=True)
    assert rel(gb.table.to_numpy(), frame(ref["psrf_autoburnin"]).to_numpy()) < EXACT
    # multivariate factor: ours follows Brooks and Gelman, (1 + 1/chains);
    # coda has (1 + 1/parameters). Same eigenvalue.
    for res, key in ((g, "mpsrf"), (gb, "mpsrf_autoburnin")):
        m, n, p = res.settings["chains"], res.settings["draws_per_chain"], 3
        lam = (res.settings["mpsrf"] ** 2 - (1 - 1 / n)) / (1 + 1 / m)
        coda_value = np.sqrt((1 - 1 / n) + (1 + 1 / p) * lam)
        assert rel(coda_value, ref[key]) < EXACT


# --------------------------------------------------------------------------
# BMA::bicreg
# --------------------------------------------------------------------------


def _which(out) -> list:
    return ["".join(str(int(v)) for v in row) for row in out.models[XN].to_numpy()]


@pytest.mark.parametrize(
    "key, kwargs",
    [
        ("bicreg", {}),
        ("bicreg_or50", {"occam_ratio": 50}),
        ("bicreg_strict", {"occam_ratio": 100, "strict": True}),
    ],
)
def test_bicreg_window_and_rounding(R, bma_data, key, kwargs):
    ref = R[key]
    out = sp.bma(f"y ~ {RHS}", bma_data, **kwargs)
    which = _which(out)
    # 1. the same models are inside the window
    assert set(which) == set(ref["which"])
    n = len(bma_data)
    r_bic = dict(zip(ref["which"], ref["bic"]))
    r_prob = dict(zip(ref["which"], ref["postprob"]))
    # 2. R's BIC is ours with R-squared rounded to five decimals
    r2 = out.models["r2"].to_numpy()
    size = out.models["n_terms"].to_numpy()
    rebuilt = n * np.log(1 - np.round(r2, 5)) + size * np.log(n)
    assert rel(rebuilt, [r_bic[w] for w in which]) < EXACT
    # ... and that rounding moves the exact BIC only in the third decimal
    assert np.max(np.abs(out.models["bic"].to_numpy() - rebuilt)) < 5e-3
    # 3. R's probabilities on our per-model fits give R's averaged output
    p = np.array([r_prob[w] for w in which])
    coefs, ses = out._fits["coefs"], out._fits["ses"]
    mean = p @ coefs
    sd = np.sqrt(p @ (ses**2 + coefs**2) - mean**2)
    assert rel(mean, ref["postmean"]) < 1e-8
    assert rel(sd, ref["postsd"]) < 1e-8
    # 4. the exact answer is within the rounding of the reference
    assert np.max(np.abs(out.models["post_prob"].to_numpy() - p)) < 1e-3
    # (bicreg prints inclusion probabilities in percent to one decimal)
    assert np.max(np.abs(100 * out.table["pip"].to_numpy()[1:] - ref["probne0"])) < 0.16


# --------------------------------------------------------------------------
# BMA::bic.glm
# --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "key, outcome, kwargs, prob_tol, mean_tol",
    [
        ("bicglm_logit", "yb", {"family": "binomial"}, EXACT, 1e-7),
        ("bicglm_poisson", "yc", {"family": "poisson"}, EXACT, 1e-7),
        ("bicglm_gamma_log", "yg", {"family": "gamma", "link": "log"}, 1e-5, 1e-5),
        ("bicglm_gamma_inverse", "yg", {"family": "gamma"}, 1e-7, 1e-6),
    ],
)
def test_bic_glm(R, bma_data, key, outcome, kwargs, prob_tol, mean_tol):
    ref = R[key]
    out = sp.bma(f"{outcome} ~ {RHS}", bma_data, **kwargs)
    which = _which(out)
    assert which == ref["which"]  # same models, same order
    assert (
        np.max(np.abs(out.models["post_prob"].to_numpy() - ref["postprob"])) < prob_tol
    )
    assert rel(out.models["deviance"], ref["deviance"]) < 1e-8
    # BIC differs from R's by one constant (R's is relative to the saturated
    # model, ours to the null model)
    shift = out.models["bic"].to_numpy() - np.array(ref["bic"])
    assert np.ptp(shift) < 1e-4
    t = out.table
    assert np.max(np.abs(t["post_mean"].to_numpy() - ref["postmean"])) < mean_tol
    assert np.max(np.abs(t["cond_mean"].to_numpy() - ref["condpostmean"])) < mean_tol
    assert rel(t["post_sd"], ref["postsd"]) < ITER
    assert np.max(np.abs(t["cond_sd"].to_numpy() - ref["condpostsd"])) < ITER
    assert np.max(np.abs(100 * t["pip"].to_numpy()[1:] - ref["probne0"])) < 0.051


# --------------------------------------------------------------------------
# BMS::bms
# --------------------------------------------------------------------------


@pytest.mark.parametrize("g", ["uip", "bric", "ric"])
def test_gprior_matches_bms(R, bma_data, g):
    ref = R[f"bms_{g}"]
    out = sp.bma(
        f"y ~ {RHS}", bma_data, method="gprior", g=g if g != "bric" else "benchmark"
    )
    assert out.n_models == 512
    assert out.settings["g"] == ref["g"]
    t = out.table.loc[ref["names"]]
    assert np.max(np.abs(t["pip"].to_numpy() - ref["pip"])) < EXACT
    assert np.max(np.abs(t["post_mean"].to_numpy() - ref["postmean"])) < EXACT
    assert np.max(np.abs(t["post_sd"].to_numpy() - ref["postsd"])) < EXACT
    top = out.models.head(10)
    assert _which(out)[:10] == ref["top_which"]
    ours = top["post_prob"].to_numpy() / top["post_prob"].to_numpy()[0]
    theirs = np.array(ref["top_pmp"]) / ref["top_pmp"][0]
    assert rel(ours, theirs) < EXACT


def test_mc3_agrees_with_enumeration(bma_data):
    exact = sp.bma(f"y ~ {RHS}", bma_data, method="gprior")
    mc3 = sp.bma(f"y ~ {RHS}", bma_data, method="gprior", search="mc3", seed=3)
    # MC3 only decides which models are visited; their weights are exact,
    # so the two agree up to the mass of the models never visited
    missed = 1.0 - exact.models.merge(mc3.models[XN], on=XN)["post_prob"].sum()
    assert missed < 1e-3
    assert np.max(np.abs(exact.table["pip"] - mc3.table["pip"])) < 2 * missed + 1e-12
    assert np.max(np.abs(exact.table["post_mean"] - mc3.table["post_mean"])) < 2e-3


# --------------------------------------------------------------------------
# identities that need no reference
# --------------------------------------------------------------------------


def test_occam_window_is_the_exact_set(bma_data):
    """Branch and bound against brute force over all 512 models."""
    out = sp.bma(f"y ~ {RHS}", bma_data, occam_ratio=20)
    y = bma_data["y"].to_numpy()
    X = bma_data[XN].to_numpy()
    n = len(y)
    tss = ((y - y.mean()) ** 2).sum()
    bics = {}
    for code in range(512):
        cols = [j for j in range(9) if (code >> j) & 1]
        Z = np.column_stack([np.ones(n), X[:, cols]])
        e = y - Z @ np.linalg.lstsq(Z, y, rcond=None)[0]
        key = "".join("1" if j in cols else "0" for j in range(9))
        bics[key] = n * np.log(e @ e / tss) + len(cols) * np.log(n)
    best = min(bics.values())
    inside = {k for k, v in bics.items() if v <= best + 2 * np.log(20)}
    assert set(_which(out)) == inside
    got = dict(zip(_which(out), out.models["bic"]))
    assert max(abs(got[k] - bics[k]) for k in inside) < 1e-9


def test_always_and_prior_inclusion(bma_data):
    base = sp.bma(f"y ~ {RHS}", bma_data, occam_ratio=1e6)
    forced = sp.bma(f"y ~ {RHS}", bma_data, occam_ratio=1e6, always=["x4"])
    assert forced.table.loc["x4", "pip"] == pytest.approx(1.0)
    assert "x4" not in forced.models.columns
    # a prior inclusion probability multiplies the odds of every model
    # containing the term by pi / (1 - pi)
    tilted = sp.bma(
        f"y ~ {RHS}", bma_data, occam_ratio=1e9, prior_inclusion={"x9": 0.2}
    )
    wide = sp.bma(f"y ~ {RHS}", bma_data, occam_ratio=1e9)
    p0 = wide.table.loc["x9", "pip"]
    expect = 0.25 * p0 / (0.25 * p0 + (1 - p0))
    assert tilted.table.loc["x9", "pip"] == pytest.approx(expect, rel=1e-6)
    assert base.n_models <= wide.n_models


def test_categorical_term_enters_as_a_block(bma_data):
    out = sp.bma("y ~ x1 + x5 + C(grp)", bma_data, occam_ratio=1e6)
    lv = [i for i in out.table.index if i.startswith("C(grp)")]
    assert len(lv) == 2
    assert out.table.loc[lv[0], "pip"] == pytest.approx(out.table.loc[lv[1], "pip"])
    assert "C(grp)" in out.models.columns
