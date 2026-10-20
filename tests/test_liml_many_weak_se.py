"""``sp.liml(robust='many_weak')``: Hansen-Hausman-Newey (2008) standard errors.

Three kinds of evidence (no reference implementation computes the paper's
formula as written, see below):

1. Identity: Bekker (1994) / HHN in the paper's form (all regressors, all
   instruments) equals the same formula after partialling the included
   exogenous regressors out, when the leverages and ``K`` are those of the
   full instrument matrix.
2. Mechanism of the R ``ivmodel`` 1.9.1 difference (T4, not copied): its
   ``manyweakSE=TRUE`` number is reproduced to 1e-10 by the same formula
   on the partialled data with the leverages of the **regressors** ``[D, X]``
   in place of the instruments', ``K`` = excluded instruments and
   ``s2 = u'u / (n - 1 - #exog)`` (#exog counting the constant). The paper's ``p_tt`` is the diagonal of
   ``P = Z(Z'Z)^-Z'``; ivmodel's point estimates and classical SEs equal ours.
3. Simulation: 95% intervals cover the true coefficient at the nominal
   rate in a many-weak design with skewed errors.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import MethodIncompatibility
from statspai.regression._many_iv_se import hhn_vcov

F = Path(__file__).parent / "fixtures" / "many_iv"
DF = pd.read_csv(F / "many_iv.csv", float_precision="round_trip")
R = json.loads((F / "ivmodel_many_iv.json").read_text(encoding="utf-8"))
ZS = [c for c in DF if c.startswith("z")]
EXOG = ["w1", "w2"]


def _fit(fuller=None, robust="many_weak"):
    return sp.liml(
        data=DF, y="y", x_endog=["d"], x_exog=EXOG, z=ZS, fuller=fuller, robust=robust
    )


@pytest.mark.parametrize("fuller,key", [(None, "liml"), (1.0, "fuller1")])
def test_estimates_and_classical_se_equal_ivmodel(fuller, key):
    r = _fit(fuller, robust="nonrobust")
    k, est, se = R[f"{key}_classical"]
    assert r.model_info["kappa"] == pytest.approx(k, rel=1e-12)
    assert r.params["d"] == pytest.approx(est, rel=1e-10)
    assert r.std_errors["d"] == pytest.approx(se, rel=1e-10)


def _partialled():
    n = len(DF)
    W = np.column_stack([np.ones(n), DF[EXOG].to_numpy()])

    def M(A):
        return A - W @ np.linalg.lstsq(W, A, rcond=None)[0]

    return (
        n,
        W,
        M(DF["y"].to_numpy()),
        M(DF["d"].to_numpy())[:, None],
        M(DF[ZS].to_numpy()),
    )


@pytest.mark.parametrize("fuller,key", [(None, "liml"), (1.0, "fuller1")])
def test_ivmodel_many_weak_se_is_the_regressor_leverage_variant(fuller, key):
    b = _fit(fuller).params["d"]
    n, W, y, d, Zt = _partialled()
    Q = np.linalg.qr(Zt)[0]
    QDX = np.linalg.qr(np.column_stack([DF["d"].to_numpy(), W]))[0]
    V = hhn_vcov(
        y,
        d,
        np.array([b]),
        Q,
        lambda A: Q @ (Q.T @ A),
        leverage=np.einsum("ij,ij->i", QDX, QDX),
        s2_df=n - 1 - W.shape[1],
    )
    assert np.sqrt(V[0, 0]) == pytest.approx(R[f"{key}_manyweak"][2], rel=1e-10)


@pytest.mark.parametrize("fuller", [None, 1.0])
def test_paper_form_equals_partialled_form_with_full_instrument_leverage(fuller):
    r = _fit(fuller)
    n, W, y, d, Zt = _partialled()
    Zfull = np.column_stack([W, DF[ZS].to_numpy()])
    Qf = np.linalg.qr(Zfull)[0]
    Q = np.linalg.qr(Zt)[0]
    # K = all instruments enters through tau and kappa: rebuild Q with the
    # full column count by padding the partialled basis with W's basis
    Qpad = np.column_stack([Q, np.linalg.qr(W)[0]])
    V = hhn_vcov(
        y,
        d,
        np.array([r.params["d"]]),
        Qpad,
        lambda A: Q @ (Q.T @ A),
        leverage=np.einsum("ij,ij->i", Qf, Qf),
        s2_df=n - 1 - W.shape[1],
    )
    assert r.std_errors["d"] == pytest.approx(np.sqrt(V[0, 0]), rel=1e-10)


def test_bekker_part_is_invariant_to_the_leverage():
    """With normal-theory terms only (leverages enter A and B alone), the
    paper's form and ivmodel's parameterisation agree: the whole ivmodel
    difference is in A and B."""
    r = _fit()
    n, W, y, d, Zt = _partialled()
    Q = np.linalg.qr(Zt)[0]
    tau = Q.shape[1] / n
    flat = np.full(n, tau)  # p_tt = tau and kappa = tau make A = B = 0
    Vb = hhn_vcov(
        y,
        d,
        np.array([r.params["d"]]),
        Q,
        lambda A: Q @ (Q.T @ A),
        leverage=flat,
        s2_df=n - 1 - W.shape[1],
    )
    Zfull = np.column_stack([W, DF[ZS].to_numpy()])
    Qf = np.linalg.qr(Zfull)[0]
    X = np.column_stack([W, DF["d"].to_numpy()])
    tau_f = Qf.shape[1] / n
    Vf = hhn_vcov(
        DF["y"].to_numpy(),
        X,
        r.params.to_numpy(),
        Qf,
        lambda A: Qf @ (Qf.T @ A),
        leverage=np.full(n, tau_f),
    )
    assert np.sqrt(Vf[-1, -1]) == pytest.approx(np.sqrt(Vb[0, 0]), rel=1e-10)


def test_coverage_in_a_many_weak_design():
    """K = 60 instruments, n = 300, concentration parameter about 1.3 K,
    skewed homoskedastic errors; 300 replications. Nominal 95% coverage has
    a binomial SD of 1.3 points; the band is three SDs (seeded: 0.930 here,
    the classical LIML SE 0.867). A 1000-replication run with n = 500,
    K = 100 gave 0.953 (classical 0.864)."""
    rng = np.random.default_rng(20261020)
    n, K, reps = 300, 60, 300
    zs = [f"z{j}" for j in range(K)]
    hits = 0
    for _ in range(reps):
        Z = rng.normal(size=(n, K))
        w = rng.normal(size=n)
        e = rng.chisquare(2, size=n) - 2.0
        v = 0.35 * e + rng.normal(size=n)
        d = Z @ np.full(K, 0.08) + 0.5 * w + v
        y = 1.0 + d + 0.5 * w + e
        df = pd.DataFrame(Z, columns=zs).assign(w=w, d=d, y=y)
        f = sp.liml(
            data=df, y="y", x_endog=["d"], x_exog=["w"], z=zs, robust="many_weak"
        )
        hits += abs(f.params["d"] - 1.0) <= 1.959964 * f.std_errors["d"]
    assert 0.911 <= hits / reps <= 0.989


def test_cluster_is_refused():
    with pytest.raises(MethodIncompatibility, match="many_weak"):
        sp.liml(
            data=DF,
            y="y",
            x_endog=["d"],
            x_exog=EXOG,
            z=ZS,
            robust="many_weak",
            cluster="w2",
        )
