"""Cross-fitted double machine learning estimator for spatial treatments.

Pollmann (2026, Section 4.1.1, the latent-experiment perspective) studies

    tau(d, p, m) = sum_i sum_s w_i(s, d)
                     (1{s in S} - 1{s not in S} p_s / (1 - p_s)) (Y_i - m_{s,i})
                   / sum_i sum_s w_i(s, d) 1{s in S}

with first-step estimates of the treatment probabilities ``p`` and of the
conditional mean ``m_{s,i}`` of ``Y_i`` in the absence of location ``s``,
both cross-fitted over folds of candidate locations so that the training
data are design-independent of the fold (Theorem 3).

Nuisance models follow the paper's application: a logistic regression of
the realised indicator on location covariates, and a linear outcome model
at the unit level whose regressors are the unit covariates and the
exposure to realised locations ring by ring; ``m_{s,i}`` is its
prediction with location ``s`` removed from the exposure counts.

``learner='ols'`` fits both by maximum likelihood / least squares on every
regressor. ``learner='post_lasso'`` is the paper's choice (Online Appendix:
"post-LASSO procedure with data-driven penalty (Belloni et al. 2012)"):
the propensity is ``sp.rlassologit(post=True)`` on the location
covariates, and the outcome model is fitted once per ring with the
intercept and that ring's exposure always included ("with the distance of
interest always included"). The unpenalized columns are partialled out,
``sp.rlasso`` (hdm's plug-in penalty, deterministic) selects among the
unit covariates and the other exposure columns, and the selected set is
refitted by least squares with the unpenalized columns.

For fold ``f`` the outcome model is trained on units that are neither
within ``d0`` of a fold-``f`` location nor within the support of a ring of
one; the propensity model on locations outside fold ``f`` that are not
within ``d0`` of a unit in that ring support (the sample splits of
Section 4.1.1).

The standard error is the design-based variance of Theorem 1(iv) applied
to the linearisation of ``tau(d, p_hat, m_hat)``: treated terms carry
``Y_i - m_{s,i} - tau_hat`` and control terms ``Y_i - m_{s,i}``. Theorem 3
shows that the first steps do not change the first-order distribution.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy import sparse

from ..exceptions import DataInsufficient
from ..matching.cbps import _logit_mle
from .treatments import _ring_variance

_P_CLIP = 1e-6


def dml_ring(
    wk: np.ndarray,
    E: sparse.csr_matrix,
    W: np.ndarray,
    p: np.ndarray,
    Y: np.ndarray,
    mhat: np.ndarray,
) -> Dict[str, Any]:
    """DML estimate and design-based variance for one ring.

    ``mhat`` is the (units x candidates) first-step outcome prediction.
    With ``mhat`` equal to the inverse-odds control mean everywhere this
    reproduces the ring (IPW) estimator and its variance exactly.
    """
    S = p.size
    r = p / (1.0 - p)
    Ypair = Y[:, None] - mhat
    colw = wk.sum(axis=0)
    A = float(W @ colw)
    T = float(W @ (Ypair * wk).sum(axis=0))
    C = float(((1.0 - W) * r) @ (Ypair * wk).sum(axis=0))
    est = (T - C) / A
    nbar = float(colw @ p) / S
    comp = _ring_variance(
        sparse.csr_matrix(wk), E, W, p, Y, est, 0.0, nbar, S, Ypair=Ypair
    )
    return {"est": est, "mu_t": T / A, "mu_c": C / A, "comp": comp}


def _exposure_features(
    W: np.ndarray,
    ring_weights: List[np.ndarray],
    beyond: Optional[np.ndarray],
) -> np.ndarray:
    """Exposure of every unit to realised locations, one column per ring."""
    cols = [wk @ W for wk in ring_weights]
    if beyond is not None:
        cols.append(beyond @ W)
    return np.column_stack(cols)


def _post_lasso_fit(
    X: np.ndarray,
    y: np.ndarray,
    forced: np.ndarray,
    penalty: Optional[Dict[str, Any]],
) -> Tuple[np.ndarray, np.ndarray]:
    """Post-LASSO with the ``forced`` columns of ``X`` always included.

    The forced columns are partialled out of ``y`` and of the penalized
    columns (Frisch-Waugh-Lovell), ``rlasso`` selects among the residualized
    penalized columns, and the union is refitted by least squares. Returns
    the coefficient vector over all columns of ``X`` (zero where not
    selected) and the boolean selection over the penalized columns.
    """
    from ..rlasso._core import rlasso

    pen_idx = np.flatnonzero(~forced)
    Xf = X[:, forced]
    Xp = X[:, pen_idx]
    sel = np.zeros(pen_idx.size, dtype=bool)
    if pen_idx.size:
        Bf, *_ = np.linalg.lstsq(Xf, np.column_stack([Xp, y]), rcond=None)
        R = np.column_stack([Xp, y]) - Xf @ Bf
        Xr, yr = R[:, :-1], R[:, -1]
        norms = np.sqrt((Xr**2).sum(axis=0))
        # A column with no variation left after partialling out cannot be
        # selected; rlasso's loadings would be zero for it.
        keep = norms > 1e-10 * max(float(norms.max(initial=0.0)), 1.0)
        if keep.any():
            fit = rlasso(Xr[:, keep], yr, post=True, intercept=False, penalty=penalty)
            sel[np.flatnonzero(keep)] = fit.index
    cols = np.concatenate([np.flatnonzero(forced), pen_idx[sel]])
    beta = np.zeros(X.shape[1])
    b, *_ = np.linalg.lstsq(X[:, cols], y, rcond=None)
    beta[cols] = b
    return beta, sel


def _outcome_post_lasso(
    Xbase: np.ndarray,
    F: np.ndarray,
    Y: np.ndarray,
    train: np.ndarray,
    n_rings: int,
    penalty: Optional[Dict[str, Any]],
    fold: int,
) -> Tuple[List[np.ndarray], List[int]]:
    """One post-LASSO outcome fit per ring on the training units."""
    X_tr = np.hstack([Xbase[train], F[train]])
    nb = Xbase.shape[1]
    betas: List[np.ndarray] = []
    n_sel: List[int] = []
    for k in range(n_rings):
        forced = np.zeros(X_tr.shape[1], dtype=bool)
        forced[0] = True  # intercept
        forced[nb + k] = True  # the distance of interest
        beta, sel = _post_lasso_fit(X_tr, Y[train], forced, penalty)
        n_used = int(forced.sum() + sel.sum())
        if train.sum() <= n_used:
            raise DataInsufficient(
                f"Fold {fold} leaves {int(train.sum())} units for a post-LASSO "
                f"outcome model with {n_used} selected regressors.",
                recovery_hint="Use more folds (n_folds=), narrower rings or a "
                "smaller max_effect_distance, so that more units are far from "
                "each fold's locations.",
                diagnostics={"fold": fold, "n_train": int(train.sum())},
            )
        betas.append(beta)
        n_sel.append(int(sel.sum()))
    return betas, n_sel


def dml_rings(
    D: np.ndarray,
    W: np.ndarray,
    Y: np.ndarray,
    ring_weights: List[np.ndarray],
    d0: float,
    p_known: Optional[np.ndarray],
    Zloc: Optional[np.ndarray],
    Xunit: Optional[np.ndarray],
    n_folds: int,
    rng: np.random.Generator,
    learner: str = "ols",
    lasso_penalty: Optional[Dict[str, Any]] = None,
    logit_penalty: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """Cross-fit the nuisances, then estimate every ring.

    ``lasso_penalty`` / ``logit_penalty`` override the ``penalty`` of
    ``rlasso`` / ``rlassologit`` for ``learner='post_lasso'`` (hdm
    defaults otherwise).
    """
    n, S = D.shape
    E_dense = D <= d0
    support = np.zeros_like(E_dense)
    for wk in ring_weights:
        support |= wk != 0
    # Realised locations within d0 but outside every ring's support still
    # shift outcomes; they get one more exposure column.
    beyond_mask = E_dense & ~support
    beyond = beyond_mask.astype(float) if beyond_mask.any() else None
    F = _exposure_features(W, ring_weights, beyond)
    base = [np.ones((n, 1))]
    if Xunit is not None:
        base.append(Xunit)
    Xbase = np.hstack(base)

    folds = rng.integers(n_folds, size=S)
    post_lasso = learner == "post_lasso"
    K = len(ring_weights)
    mhat = np.zeros((n, S))
    mhats = [np.zeros((n, S)) for _ in range(K)] if post_lasso else []
    sel_outcome: List[List[int]] = []
    sel_propensity: List[int] = []
    p_hat = (
        p_known.copy()
        if p_known is not None
        else np.full(S, W.mean()) if Zloc is None else np.full(S, np.nan)
    )
    n_train_units: List[int] = []
    n_train_locs: List[int] = []
    for f in range(n_folds):
        in_f = folds == f
        if not in_f.any():
            continue
        near_w = support[:, in_f].any(axis=1)  # units weighted near fold f
        near_e = E_dense[:, in_f].any(axis=1)  # units exposed to fold f
        train = np.asarray(~(near_w | near_e))
        n_train_units.append(int(train.sum()))
        cols = np.flatnonzero(in_f)
        if post_lasso:
            betas, n_sel = _outcome_post_lasso(Xbase, F, Y, train, K, lasso_penalty, f)
            sel_outcome.append(n_sel)
            fits = [(beta, mhats[k]) for k, beta in enumerate(betas)]
        else:
            X_tr = np.hstack([Xbase[train], F[train]])
            if train.sum() <= X_tr.shape[1]:
                raise DataInsufficient(
                    f"Fold {f} leaves {int(train.sum())} units to fit the "
                    f"outcome model with {X_tr.shape[1]} regressors.",
                    recovery_hint="Use more folds (n_folds=), narrower rings or "
                    "a smaller max_effect_distance, so that more units are far "
                    "from each fold's locations.",
                    diagnostics={"fold": f, "n_train": int(train.sum())},
                )
            beta, *_ = np.linalg.lstsq(X_tr, Y[train], rcond=None)
            fits = [(beta, mhat)]
        for beta, target in fits:
            b_base, b_exp = beta[: Xbase.shape[1]], beta[Xbase.shape[1] :]
            # Prediction with s removed from the exposure counts when realised.
            pred_obs = Xbase @ b_base + F @ b_exp
            for s in cols:
                shift = np.column_stack(
                    [wk[:, s] for wk in ring_weights]
                    + ([beyond[:, s]] if beyond is not None else [])
                )
                target[:, s] = pred_obs - W[s] * (shift @ b_exp)
        if Zloc is not None and p_known is None:
            blocked = E_dense[near_w].any(axis=0)
            loc_train = ~in_f & ~blocked
            n_train_locs.append(int(loc_train.sum()))
            wt = W[loc_train]
            # post-LASSO selects among the covariates, so it does not need
            # more training locations than covariates.
            too_few = loc_train.sum() <= (1 if post_lasso else Zloc.shape[1] + 1)
            if too_few or wt.min() == wt.max():
                raise DataInsufficient(
                    f"Fold {f} leaves {int(loc_train.sum())} candidate "
                    "locations, without both realised and unrealised ones, to "
                    "fit the propensity model.",
                    recovery_hint="Use more folds (n_folds=) or a smaller "
                    "max_effect_distance; or pass known probabilities with "
                    "propensity=.",
                    diagnostics={"fold": f, "n_train": int(loc_train.sum())},
                )
            if post_lasso:
                from ..rlasso._logit import rlassologit

                lfit = rlassologit(
                    Zloc[loc_train], wt, post=True, penalty=logit_penalty
                )
                sel_propensity.append(int(lfit.n_selected))
                p_hat[in_f] = lfit.predict(Zloc[in_f])
            else:
                Z1 = np.column_stack([np.ones(S), Zloc])
                gamma = _logit_mle(Z1[loc_train], wt)
                p_hat[in_f] = 1.0 / (1.0 + np.exp(-(Z1[in_f] @ gamma)))

    n_clipped = int(((p_hat < _P_CLIP) | (p_hat > 1 - _P_CLIP)).sum())
    p_hat = np.clip(p_hat, _P_CLIP, 1 - _P_CLIP)
    E = sparse.csr_matrix(E_dense)
    rings: List[Optional[Dict[str, Any]]] = []
    for k, wk in enumerate(ring_weights):
        n_t = int(((wk > 0) & (W[None, :] == 1)).sum())
        n_c = int(((wk > 0) & (W[None, :] == 0)).sum())
        if n_t == 0 or n_c == 0:
            rings.append(None)
            continue
        res = dml_ring(wk, E, W, p_hat, Y, mhats[k] if post_lasso else mhat)
        res.update(n_t=n_t, n_c=n_c)
        rings.append(res)
    return {
        "rings": rings,
        "p_hat": p_hat,
        "mhat": mhats if post_lasso else mhat,
        "folds": folds,
        "n_clipped": n_clipped,
        "n_train_units": n_train_units,
        "n_train_locations": n_train_locs,
        "learner": learner,
        "n_selected_outcome": sel_outcome,
        "n_selected_propensity": sel_propensity,
    }
