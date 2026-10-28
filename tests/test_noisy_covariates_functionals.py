"""sp.noisy_covariates: elasticity, policy effect, partially linear IV,
weighted fit and user folds (Agarwal-Singh Appendix E.3, E.4, E.6).

Evidence tier T1: known-truth recovery on low-rank designs and exact
identities (hand-written cross-fit, reductions between estimands). The
authors' own code on their ADH application is in
tests/reference_parity/test_noisy_covariates_authors_parity.py.
"""

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import DataInsufficient, MethodIncompatibility


def _factor_data(seed, n, p, r=3, noise=0.4, miss=0.15, mu=None):
    rng = np.random.default_rng(seed)
    L = rng.normal(size=(n, r))
    if mu is not None:
        L = L + np.asarray(mu)
    load = rng.normal(size=(r, p))
    X = L @ load
    Z = X + rng.normal(scale=noise, size=X.shape)
    if miss:
        Z[rng.uniform(size=Z.shape) < miss] = np.nan
    cols = [f"z{j}" for j in range(p)]
    return pd.DataFrame(Z, columns=cols), cols, L, load, rng


def test_default_outputs_unchanged():
    # Values from the pre-change code (bit-identical locally, 7 configs);
    # rtol leaves room for other BLAS builds only. Since weighted_fit
    # defaults to True when weights= is given (2026-10-28, before any
    # release), the weighted configurations pin the old projection through
    # an explicit weighted_fit=False.
    rng = np.random.default_rng(1)
    n, p, r = 300, 60, 3
    L = rng.normal(size=(n, r))
    X = L @ rng.normal(size=(r, p))
    Z = X + rng.normal(scale=0.5, size=X.shape)
    Z[rng.uniform(size=Z.shape) < 0.15] = np.nan
    cols = [f"z{j}" for j in range(p)]
    df = pd.DataFrame(Z, columns=cols)
    df["d"] = rng.binomial(1, 1 / (1 + np.exp(-L[:, 0])))
    df["u"] = rng.binomial(1, 0.5, size=n)
    df["dc"] = L[:, 0] + rng.normal(size=n)
    df["y"] = df.d * 1.5 + L[:, 0] + rng.normal(size=n)
    df["w"] = np.exp(0.3 * L[:, 1])
    pinned = {
        "ate_w": (
            dict(treat="d", weights="w", rank=3, weighted_fit=False),
            1.5764155924913292,
            0.14175155400662678,
        ),
        "ate_nc": (
            dict(treat="d", add_constant=False, rank=3, n_folds=3, seed=5),
            0.03641385872119152,
            0.06182859862986305,
        ),
        "plr_w": (
            dict(
                treat="dc",
                estimand="partially_linear",
                weights="w",
                rank=4,
                weighted_fit=False,
            ),
            0.0631496223536246,
            0.06725227886083687,
        ),
    }
    for kw, est, se in pinned.values():
        res = sp.noisy_covariates(df, "y", covariates=cols, **kw)
        np.testing.assert_allclose(res.estimate, est, rtol=1e-10)
        np.testing.assert_allclose(res.se, se, rtol=1e-10)
        assert res.model_info["weighted_fit"] is False
        assert res.model_info["fold_assignment"] == "random"
    # weights= alone now means weighted_fit=True, bit for bit.
    for kw, _, _ in pinned.values():
        if "weights" not in kw:
            continue
        kw = {k: v for k, v in kw.items() if k != "weighted_fit"}
        default = sp.noisy_covariates(df, "y", covariates=cols, **kw)
        explicit = sp.noisy_covariates(
            df, "y", covariates=cols, weighted_fit=True, **kw
        )
        assert default.estimate == explicit.estimate
        assert default.se == explicit.se
        assert default.model_info["weighted_fit"] is True


def test_elasticity_recovers_average_derivative():
    # E[Y|D,X] = 0.5 D + 0.3 D^2 + 0.4 D L1 + L0 + L2: inside the
    # dictionary (X, D X, D^2 X) because L1 is a linear function of X.
    # Truth: the sample mean of 0.5 + 0.6 D + 0.4 L1. 25 replications:
    # error SD ~ 0.03, so the MC SE of the mean error is ~ 0.006 and the
    # 0.03 bound is five of them. (At 40 replications: mean error 0.002,
    # mean SE 0.040 against an error SD of 0.030, coverage 0.975.)
    errs, cover = [], []
    for b in range(25):
        df, cols, L, _, rng = _factor_data(100 + b, n=1200, p=150)
        dose = 0.8 * L[:, 0] + rng.normal(size=len(df))
        df["dose"] = dose
        df["y"] = (
            0.5 * dose
            + 0.3 * dose**2
            + 0.4 * dose * L[:, 1]
            + L[:, 0]
            + L[:, 2]
            + rng.normal(size=len(df))
        )
        truth = float(np.mean(0.5 + 0.6 * dose + 0.4 * L[:, 1]))
        res = sp.noisy_covariates(
            df, "y", "dose", cols, estimand="elasticity", rank=3, seed=b
        )
        errs.append(res.estimate - truth)
        cover.append(res.ci[0] <= truth <= res.ci[1])
    assert abs(np.mean(errs)) < 0.03
    assert np.mean(cover) >= 0.85


def test_elasticity_identity_full_rank_equals_hand_dr():
    # No corruption, k = p: OLS on b = (1, X, D, D X, D^2, D^2 X) per
    # training fold, Riesz weight solving (B'B) eta = sum_train m with
    # m = (0, 1, 2D) blocks, DR score on the test fold.
    rng = np.random.default_rng(3)
    n, p = 400, 4
    X = rng.normal(size=(n, p))
    dose = X[:, 0] + rng.normal(size=n)
    yv = dose + 0.5 * dose**2 * X[:, 1] + X.sum(axis=1) + rng.normal(size=n)
    cols = [f"x{j}" for j in range(p)]
    df = pd.DataFrame(X, columns=cols)
    df["y"], df["dose"] = yv, dose
    with pytest.warns(sp.AssumptionWarning, match="low rank"):
        res = sp.noisy_covariates(
            df, "y", "dose", cols, estimand="elasticity", rank=p, seed=4
        )
    frng = np.random.default_rng(4)
    folds = np.empty(n, dtype=int)
    perm = frng.permutation(np.arange(n))
    folds[perm] = np.arange(n) % 2
    Xc = np.column_stack([np.ones(n), X])
    B = np.column_stack([Xc, dose[:, None] * Xc, dose[:, None] ** 2 * Xc])
    Mrow = np.column_stack([0 * Xc, Xc, 2 * dose[:, None] * Xc])
    psi = np.empty(n)
    for f in (0, 1):
        te, tr = folds == f, folds != f
        beta = np.linalg.lstsq(B[tr], yv[tr], rcond=None)[0]
        eta = np.linalg.solve(B[tr].T @ B[tr], Mrow[tr].sum(axis=0))
        psi[te] = Mrow[te] @ beta + (B[te] @ eta) * (yv[te] - B[te] @ beta)
    np.testing.assert_allclose(res.estimate, psi.mean(), rtol=1e-9)
    se = np.sqrt(np.mean((psi - psi.mean()) ** 2) / n)
    np.testing.assert_allclose(res.se, se, rtol=1e-9)


def test_policy_effect_known_truth():
    # Y = 2 + L0 - 0.5 L1 + e with E[L] = (1, 0.5, 0). Doubling every
    # covariate (t(x) = 2x) moves E[Y|X] by X beta = L0 - 0.5 L1; a shift
    # along the first loading row moves L0 by delta, so Y by delta.
    df, cols, L, load, rng = _factor_data(7, n=2000, p=200, mu=(1.0, 0.5, 0.0))
    df["y"] = 2 + L[:, 0] - 0.5 * L[:, 1] + rng.normal(size=len(df))
    double = {c: (2.0, 0.0) for c in cols}
    res = sp.noisy_covariates(
        df, "y", None, cols, estimand="policy_effect", policy=double, rank=3
    )
    truth = float(np.mean(L[:, 0] - 0.5 * L[:, 1]))
    assert abs(res.estimate - truth) < 4 * res.se
    assert res.estimand == "average policy effect"
    delta = 0.7
    shift = {c: float(delta * load[0, j]) for j, c in enumerate(cols)}
    res2 = sp.noisy_covariates(
        df, "y", None, cols, estimand="policy_effect", policy=shift, rank=3
    )
    assert abs(res2.estimate - delta) < 4 * res2.se
    assert res2.model_info["policy"]["z0"] == (1.0, delta * load[0, 0])


def test_policy_on_treat_is_shift_times_partially_linear():
    # Moving only the exactly observed regressor by c: the dictionary is
    # the partially linear one reordered and the moment is c times its
    # moment, so the score is c times the partially linear score.
    df, cols, L, _, rng = _factor_data(8, n=600, p=80)
    df["dose"] = L[:, 0] + rng.normal(size=len(df))
    df["y"] = 0.7 * df["dose"] + L[:, 1] + rng.normal(size=len(df))
    plr = sp.noisy_covariates(
        df, "y", "dose", cols, estimand="partially_linear", rank=3, seed=2
    )
    pol = sp.noisy_covariates(
        df,
        "y",
        "dose",
        cols,
        estimand="policy_effect",
        policy={"dose": 2.5},
        rank=3,
        seed=2,
    )
    np.testing.assert_allclose(pol.estimate, 2.5 * plr.estimate, rtol=1e-9)
    np.testing.assert_allclose(pol.se, 2.5 * plr.se, rtol=1e-9)


def test_partially_linear_iv_known_truth_and_identity():
    df, cols, L, _, rng = _factor_data(9, n=2000, p=200)
    n = len(df)
    u = 0.6 * L[:, 0] + rng.normal(size=n)  # instrument, confounded by L0
    v = rng.normal(size=n)
    dose = 0.8 * u + L[:, 1] + v
    df["u"], df["dose"] = u, dose
    df["y"] = -0.6 * dose + L[:, 0] + L[:, 1] + 0.8 * v + rng.normal(size=n)
    res = sp.noisy_covariates(
        df, "y", "dose", cols, estimand="partially_linear_iv", instrument="u", rank=3
    )
    assert abs(res.estimate + 0.6) < 4 * res.se
    assert abs(res.model_info["first_stage"] - 0.8) < 0.1
    assert res.estimand == "partially linear IV coefficient"
    # Instrument = treatment: the first stage is exactly one, so the ratio
    # is the partially linear coefficient.
    same = sp.noisy_covariates(
        df,
        "y",
        "dose",
        cols,
        estimand="partially_linear_iv",
        instrument="dose",
        rank=3,
        seed=1,
    )
    plr = sp.noisy_covariates(
        df, "y", "dose", cols, estimand="partially_linear", rank=3, seed=1
    )
    np.testing.assert_allclose(same.estimate, plr.estimate, rtol=1e-9)
    np.testing.assert_allclose(same.model_info["first_stage"], 1.0, rtol=1e-9)


def test_weighted_fit_targets_weighted_projection():
    # Y = tau(L1) D + L0 + e, D = 0.8 L0 + v with v independent of L.
    # The weighted partially linear coefficient is E[W tau] / E[W]; with
    # W = exp(0.5 L1), tau = 1 + 0.5 L1 it is 1.25 (lognormal tilt), which
    # weights= alone now targets. weighted_fit=False (unweighted regression;
    # the functional's moment is constant) stays near the unweighted value 1.
    df, cols, L, _, rng = _factor_data(10, n=4000, p=200)
    n = len(df)
    dose = 0.8 * L[:, 0] + rng.normal(size=n)
    w = np.exp(0.5 * L[:, 1])
    tau = 1 + 0.5 * L[:, 1]
    df["dose"], df["w"] = dose, w
    df["y"] = tau * dose + L[:, 0] + rng.normal(size=n)
    kw = dict(estimand="partially_linear", rank=3, weights="w")
    wf = sp.noisy_covariates(df, "y", "dose", cols, **kw)
    de = sp.noisy_covariates(df, "y", "dose", cols, weighted_fit=False, **kw)
    truth_w = float(np.sum(w * tau) / np.sum(w))
    assert abs(wf.estimate - truth_w) < 4 * wf.se
    assert abs(de.estimate - float(np.mean(tau))) < 4 * de.se
    assert truth_w - float(np.mean(tau)) > 4 * max(wf.se, de.se)
    assert wf.model_info["weighted_fit"] is True


def test_weighted_fit_with_constant_weights_is_identity():
    df, cols, L, _, rng = _factor_data(11, n=500, p=60)
    df["d"] = rng.binomial(1, 1 / (1 + np.exp(-L[:, 0])))
    df["y"] = 1.5 * df["d"] + L[:, 0] + rng.normal(size=len(df))
    df["one"] = 4.0
    a = sp.noisy_covariates(df, "y", "d", cols, rank=3)
    b = sp.noisy_covariates(
        df, "y", "d", cols, rank=3, weights="one", weighted_fit=True
    )
    np.testing.assert_allclose(b.estimate, a.estimate, rtol=1e-12)
    np.testing.assert_allclose(b.se, a.se, rtol=1e-12)


def test_user_folds_reproduce_internal_split():
    df, cols, L, _, rng = _factor_data(12, n=400, p=60)
    df["dose"] = L[:, 0] + rng.normal(size=len(df))
    df["y"] = 0.5 * df["dose"] + L[:, 1] + rng.normal(size=len(df))
    kw = dict(estimand="partially_linear", rank=3)
    base = sp.noisy_covariates(df, "y", "dose", cols, seed=6, **kw)
    frng = np.random.default_rng(6)
    folds = np.empty(len(df), dtype=int)
    folds[frng.permutation(np.arange(len(df)))] = np.arange(len(df)) % 2
    labels = np.where(folds == 0, "a", "b")  # any hashable labels
    df["fold"] = labels
    for given in (labels, "fold"):
        res = sp.noisy_covariates(df, "y", "dose", cols, folds=given, **kw)
        assert res.estimate == base.estimate and res.se == base.se
        assert res.model_info["fold_assignment"] == "user"


def test_user_folds_follow_dropped_rows():
    df, cols, L, _, rng = _factor_data(13, n=300, p=50)
    df["dose"] = L[:, 0] + rng.normal(size=len(df))
    df["y"] = 0.5 * df["dose"] + rng.normal(size=len(df))
    folds = np.arange(len(df)) % 3
    df.loc[df.index[:6], "y"] = np.nan
    res = sp.noisy_covariates(
        df, "y", "dose", cols, estimand="partially_linear", rank=3, folds=folds
    )
    assert res.n_obs == len(df) - 6
    assert res.model_info["n_folds"] == 3


def test_edge_cases_new_options():
    df, cols, L, _, rng = _factor_data(14, n=200, p=40)
    df["d"] = rng.binomial(1, 0.5, size=len(df))
    df["dose"] = L[:, 0] + rng.normal(size=len(df))
    df["y"] = df["dose"] + rng.normal(size=len(df))
    with pytest.raises(DataInsufficient, match="three values"):
        sp.noisy_covariates(df, "y", "d", cols, estimand="elasticity")
    with pytest.raises(MethodIncompatibility, match="policy"):
        sp.noisy_covariates(df, "y", None, cols, estimand="policy_effect")
    with pytest.raises(MethodIncompatibility, match="policy"):
        sp.noisy_covariates(df, "y", "d", cols, policy={"z0": 1.0})
    with pytest.raises(MethodIncompatibility, match="neither"):
        sp.noisy_covariates(
            df, "y", None, cols, estimand="policy_effect", policy={"nope": 1.0}
        )
    with pytest.raises(MethodIncompatibility, match="unchanged"):
        sp.noisy_covariates(
            df, "y", None, cols, estimand="policy_effect", policy={"z0": (1, 0)}
        )
    with pytest.raises(MethodIncompatibility, match="scale, shift"):
        sp.noisy_covariates(
            df, "y", None, cols, estimand="policy_effect", policy={"z0": "x"}
        )
    with pytest.raises(MethodIncompatibility, match="treat"):
        sp.noisy_covariates(df, "y", None, cols)
    with pytest.raises(MethodIncompatibility, match="instrument"):
        sp.noisy_covariates(df, "y", "dose", cols, estimand="partially_linear_iv")
    with pytest.raises(MethodIncompatibility, match="weighted_fit"):
        sp.noisy_covariates(df, "y", "d", cols, weighted_fit=True)
    with pytest.raises(MethodIncompatibility, match="one label per row"):
        sp.noisy_covariates(df, "y", "d", cols, folds=np.zeros(5))
    with pytest.raises(MethodIncompatibility, match="two distinct"):
        sp.noisy_covariates(df, "y", "d", cols, folds=np.zeros(len(df)))
    arm_folds = df["d"].to_numpy()  # each training fold holds one arm only
    with pytest.raises(DataInsufficient, match="arm"):
        sp.noisy_covariates(df, "y", "d", cols, folds=arm_folds)
    df["u0"] = 1.0
    with pytest.raises(DataInsufficient, match="instrument does not vary"):
        sp.noisy_covariates(
            df, "y", "dose", cols, estimand="partially_linear_iv", instrument="u0"
        )
