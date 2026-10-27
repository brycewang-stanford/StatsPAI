"""Option scan, round 5 (2026-10-27): ``sp.mr_bma`` limits and inputs, and
the input checks of ``sp.mr_lap`` / ``sp.mr_clust``.

The Bayes factor is pinned to Zuber et al.'s ``summary_mvMR_BF`` in
``tests/reference_parity/test_misc_sens_R_parity.py``. That script has no
model-size cap (its ``kmax`` lives in the stochastic search), so the
``max_model_size`` path is checked here against the same closed form: the
posterior of each model is proportional to ``10^(log10 BF + log10 prior)``
over the models of size ``<= max_model_size``.
"""

from __future__ import annotations

from itertools import combinations
from math import comb

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import MethodIncompatibility


def _snps(k: int, n: int = 60, seed: int = 11) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({f"beta_x{j}": rng.uniform(0.05, 0.5, n) for j in range(k)})
    df["beta_y"] = 0.3 * df["beta_x0"] - 0.2 * df["beta_x2"] + rng.normal(0, 0.03, n)
    df["se_y"] = rng.uniform(0.01, 0.05, n)
    return df


def _log10_evidence(df, exposures, kmax, sigma=0.5, p=0.5):
    se = df["se_y"].to_numpy()
    X = df[exposures].to_numpy() / se[:, None]
    Y = df["beta_y"].to_numpy() / se
    n, k = X.shape
    out = {}
    for size in range(1, kmax + 1):
        for g in combinations(range(k), size):
            Xg = X[:, list(g)]
            inv_om = Xg.T @ Xg + np.eye(size) / sigma**2
            B = np.linalg.solve(inv_om, Xg.T @ Y)
            bf = (
                -0.5 * np.log10(np.linalg.det(inv_om))
                - size * np.log10(sigma)
                - n / 2 * (np.log10(Y @ Y - B @ inv_om @ B) - np.log10(Y @ Y))
            )
            out[g] = bf + size * np.log10(p) + (k - size) * np.log10(1 - p)
    return out


def test_many_exposures_run_with_a_model_size_cap():
    df = _snps(16)
    exposures = [f"beta_x{j}" for j in range(16)]
    res = sp.mr_bma(df, exposures=exposures, max_model_size=2)
    assert len(res.model_priors) == comb(16, 1) + comb(16, 2)
    ev = _log10_evidence(df, exposures, 2)
    logs = np.array(list(ev.values()))
    post = 10 ** (logs - logs.max())
    post /= post.sum()
    np.testing.assert_allclose(res.model_priors, post, rtol=1e-9, atol=1e-300)
    # marginal inclusion = sum of posteriors of the models holding j
    marg = np.zeros(16)
    for g, pr in zip(ev, post):
        marg[list(g)] += pr
    np.testing.assert_allclose(res.marginal_inclusion.to_numpy(), marg, rtol=1e-9)


def test_model_size_cap_is_the_restricted_model_space():
    df = _snps(4)
    exposures = [f"beta_x{j}" for j in range(4)]
    full = sp.mr_bma(df, exposures=exposures)
    cap = sp.mr_bma(df, exposures=exposures, max_model_size=2)
    n_small = comb(4, 1) + comb(4, 2)
    # combinations() enumerates by size, so the first n_small models of the
    # full space are the capped space; renormalised they must agree.
    restricted = full.model_priors[:n_small] / full.model_priors[:n_small].sum()
    np.testing.assert_allclose(cap.model_priors, restricted, rtol=1e-12)


def test_too_many_models_is_refused_with_the_model_count():
    df = _snps(15)
    exposures = [f"beta_x{j}" for j in range(15)]
    with pytest.raises(MethodIncompatibility, match="32767 models"):
        sp.mr_bma(df, exposures=exposures)


@pytest.mark.parametrize("bad", [0, -1, 4, 1.5, True])
def test_invalid_max_model_size_is_refused(bad):
    df = _snps(3)
    with pytest.raises(MethodIncompatibility, match="max_model_size"):
        sp.mr_bma(df, exposures=["beta_x0", "beta_x1", "beta_x2"], max_model_size=bad)


def test_max_model_size_equal_to_k_is_the_default():
    df = _snps(3)
    ex = ["beta_x0", "beta_x1", "beta_x2"]
    a = sp.mr_bma(df, exposures=ex)
    b = sp.mr_bma(df, exposures=ex, max_model_size=3)
    np.testing.assert_array_equal(a.model_priors, b.model_priors)


@pytest.mark.parametrize("value", [0.0, -0.02, np.inf])
def test_non_positive_outcome_se_is_refused(value):
    df = _snps(3)
    df.loc[4, "se_y"] = value
    with pytest.raises(MethodIncompatibility, match="positive"):
        sp.mr_bma(df, exposures=["beta_x0", "beta_x1", "beta_x2"])


def test_duplicate_exposures_are_refused():
    df = _snps(3)
    with pytest.raises(MethodIncompatibility, match="more than once"):
        sp.mr_bma(df, exposures=["beta_x0", "beta_x0", "beta_x1"])


def test_row_order_and_index_labels_do_not_matter():
    df = _snps(4)
    ex = [f"beta_x{j}" for j in range(4)]
    base = sp.mr_bma(df, exposures=ex)
    shuffled = df.sample(frac=1.0, random_state=3)
    shuffled.index = [0] * len(shuffled)  # duplicated labels
    other = sp.mr_bma(shuffled, exposures=ex)
    np.testing.assert_allclose(other.model_priors, base.model_priors, rtol=1e-10)
    np.testing.assert_allclose(
        other.model_averaged_estimate.to_numpy(),
        base.model_averaged_estimate.to_numpy(),
        rtol=1e-10,
    )


# --------------------------------------------------------------------------
# MR frontier: mr_lap and mr_clust inputs (grapple / mr_cml / mr_raps
# already refuse these)
# --------------------------------------------------------------------------


def _summary(n: int = 25, seed: int = 8):
    rng = np.random.default_rng(seed)
    bx = rng.uniform(0.1, 0.5, n)
    sx = rng.uniform(0.02, 0.05, n)
    sy = rng.uniform(0.02, 0.05, n)
    by = 0.3 * bx + rng.normal(0, sy)
    return bx, by, sx, sy


def _bad(kind):
    bx, by, sx, sy = _summary()
    if kind == "nan_bx":
        bx[2] = np.nan
    elif kind == "inf_by":
        by[2] = np.inf
    elif kind == "zero_sy":
        sy[2] = 0.0
    elif kind == "neg_sy":
        sy[2] = -sy[2]
    elif kind == "neg_sx":
        sx[2] = -sx[2]
    return bx, by, sx, sy


@pytest.mark.parametrize("kind", ["nan_bx", "inf_by", "zero_sy", "neg_sy", "neg_sx"])
@pytest.mark.parametrize("fn", ["mr_lap", "mr_clust"])
def test_frontier_refuses_invalid_summary_statistics(fn, kind):
    # mr_lap returned NaN for a NaN beta, raised ZeroDivisionError for a
    # zero SE and squared a negative SE away; mr_clust returned an all-NaN
    # fit for the first two and accepted the third.
    with pytest.raises(MethodIncompatibility):
        getattr(sp, fn)(*_bad(kind))


def test_mr_lap_refuses_zero_exposure_se():
    bx, by, sx, sy = _summary()
    sx[0] = 0.0  # the mean F statistic would be infinite and the bias 0
    with pytest.raises(MethodIncompatibility, match="se_exposure"):
        sp.mr_lap(bx, by, sx, sy, overlap_fraction=0.5, overlap_rho=0.2)


def test_mr_clust_refuses_a_zero_exposure_effect():
    bx, by, sx, sy = _summary()
    bx[3] = 0.0  # Wald ratio undefined
    with pytest.raises(MethodIncompatibility, match="beta_exposure"):
        sp.mr_clust(bx, by, sx, sy)


def test_frontier_valid_inputs_unchanged_and_sign_invariant():
    bx, by, sx, sy = _summary()
    lap = sp.mr_lap(bx, by, sx, sy, overlap_fraction=0.4, overlap_rho=0.18)
    w = 1 / sy**2
    ivw = np.sum(w * bx * by) / np.sum(w * bx**2)
    f_mean = np.mean((bx / sx) ** 2)
    assert lap.estimate == pytest.approx(ivw - 0.4 * 0.18 / f_mean, rel=1e-12)
    s = np.where(np.arange(len(bx)) % 3 == 0, -1.0, 1.0)
    assert sp.mr_lap(
        bx * s, by * s, sx, sy, overlap_fraction=0.4, overlap_rho=0.18
    ).estimate == pytest.approx(lap.estimate, rel=1e-15)
    a = sp.mr_clust(bx, by, sx, sy)
    b = sp.mr_clust(bx * s, by * s, sx, sy)
    pd.testing.assert_frame_equal(a.cluster_estimates, b.cluster_estimates)


def test_mr_lap_alpha_is_validated_and_moves_the_interval():
    bx, by, sx, sy = _summary()
    r95 = sp.mr_lap(bx, by, sx, sy)
    r90 = sp.mr_lap(bx, by, sx, sy, alpha=0.10)
    z90 = 1.6448536269514722
    assert r90.ci_upper - r90.estimate == pytest.approx(z90 * r95.se, rel=1e-12)
    with pytest.raises(MethodIncompatibility, match="alpha"):
        sp.mr_lap(bx, by, sx, sy, alpha=1.5)
