"""Shared nuclear-norm matrix-completion solver (MC-NNM).

Used by :func:`statspai.mc_panel` and :func:`statspai.mc_synth`.

Problem
-------
For an ``(N, T)`` outcome matrix ``Y`` observed on the cell set ``O``
(``obs == True``) the estimator is

    min_{mu, a, b, L}  1/2 * sum_{(i,t) in O} (Y_it - F_it)^2 + theta * ||L||_*
    F_it = [fixed effects]_it + L_it

where the fixed-effect part is unpenalised and is

* ``"two-way"``: ``a_i + b_t`` (unit and time effects),
* ``"unit"``:    ``a_i``,
* ``"time"``:    ``b_t``,
* ``"none"``:    ``0`` (pure soft-impute; no intercept either).

``theta`` is the singular-value threshold on this ``1/2``-scaled objective.
The same minimiser appears in the references under other scalings:

* R ``MCPanel::mcnnm_fit`` minimises ``(1/|O|) ||P_O(Y - F)||_F^2 +
  lambda_L ||L||_*`` (its ``update_L`` thresholds at ``lambda_L |O| / 2``),
  so ``theta = lambda_L * |O| / 2``; ``to_estimate_u`` / ``to_estimate_v``
  switch the unit / time effects.
* R ``fect(method = "mc", CV = FALSE, lambda = l)`` soft-thresholds the
  singular values of ``E / (T N)`` at ``l``, so ``theta = l * N * T``;
  ``force = "two-way"`` is the two-way case (``force = "none"`` keeps an
  unpenalised grand mean and is NOT the ``"none"`` case here).

Algorithm
---------
Soft-impute / EM (majorise-minimise): fill the unobserved cells with the
current fit, ``Z = P_O(Y) + P_O^perp(F)``; on the complete matrix remove the
fixed effects exactly by (double-)centring, ``E = C_N Z C_T``; set
``L = SVT_theta(E)`` and ``F = (Z - E) + L``.  Because centring is an
orthogonal projection and ``||P L Q||_* <= ||L||_*``, this is the exact
complete-data minimiser, so every step decreases the objective and the fixed
point is the minimiser above.  Iteration stops when the relative Frobenius
change of ``F`` falls below ``tol``.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, Optional

import numpy as np

FIXED_EFFECTS = ("two-way", "unit", "time", "none")


def _check_fixed_effects(fixed_effects: str) -> str:
    if fixed_effects not in FIXED_EFFECTS:
        raise ValueError(
            f"fixed_effects must be one of {FIXED_EFFECTS}; got {fixed_effects!r}"
        )
    return fixed_effects


def _center(Z: np.ndarray, fixed_effects: str) -> np.ndarray:
    """Remove the (unpenalised) fixed-effect part of a complete matrix."""
    if fixed_effects == "two-way":
        return (
            Z - Z.mean(axis=1, keepdims=True) - Z.mean(axis=0, keepdims=True) + Z.mean()
        )
    if fixed_effects == "unit":
        return Z - Z.mean(axis=1, keepdims=True)
    if fixed_effects == "time":
        return Z - Z.mean(axis=0, keepdims=True)
    return Z


def _svt(E: np.ndarray, theta: float, max_rank: Optional[int] = None):
    U, s, Vt = np.linalg.svd(E, full_matrices=False)
    s_thr = np.maximum(s - theta, 0.0)
    if max_rank is not None:
        s_thr[max_rank:] = 0.0
    return (U * s_thr) @ Vt, s_thr


def mc_nnm_fit(
    Y: np.ndarray,
    obs: np.ndarray,
    theta: float,
    fixed_effects: str = "two-way",
    max_iter: int = 5000,
    tol: float = 1e-10,
    max_rank: Optional[int] = None,
    init: Optional[np.ndarray] = None,
    warn: bool = True,
) -> Dict[str, Any]:
    """Fit the nuclear-norm matrix-completion model on the cells ``obs``.

    Parameters
    ----------
    Y : ndarray (N, T)
        Outcome matrix; values at unobserved cells are ignored (may be NaN).
    obs : ndarray of bool (N, T)
        True where ``Y`` enters the loss.
    theta : float
        Singular-value threshold (see module docstring for the mapping to
        ``MCPanel``'s ``lambda_L`` and ``fect``'s ``lambda``).
    fixed_effects : {"two-way", "unit", "time", "none"}
    max_iter, tol : iteration cap and relative-change stopping rule.
    max_rank : int, optional
        Hard cap on the rank of ``L`` (not part of the references' problem).
    init : ndarray (N, T), optional
        Starting fit; default is the grand mean of the observed cells
        everywhere. The minimiser does not depend on it (convex problem).
    warn : bool
        Emit a ``RuntimeWarning`` when ``max_iter`` is hit before ``tol``.

    Returns
    -------
    dict with ``fit`` (F), ``L`` (low-rank part, double-centred for
    two-way), ``fe`` (F - L), ``singular_values`` (of L), ``n_iter``,
    ``converged``, ``rel_change``.
    """
    _check_fixed_effects(fixed_effects)
    if theta < 0 or not np.isfinite(theta):
        raise ValueError(f"theta (lambda_reg) must be finite and >= 0; got {theta}")
    Y = np.asarray(Y, dtype=float)
    obs = np.asarray(obs, dtype=bool) & np.isfinite(Y)
    if obs.sum() == 0:
        raise ValueError("No observed (control) cells to fit.")
    Y0 = np.where(obs, Y, 0.0)

    if init is None:
        fill = Y0[obs].mean()
        F = np.full(Y.shape, fill)
    else:
        F = np.asarray(init, dtype=float).copy()

    converged = False
    rel = np.inf
    L = np.zeros_like(F)
    s_thr = np.zeros(min(Y.shape))
    it = 0
    for it in range(1, max_iter + 1):
        Z = np.where(obs, Y0, F)
        E = _center(Z, fixed_effects)
        L, s_thr = _svt(E, theta, max_rank)
        F_new = (Z - E) + L
        rel = float(np.linalg.norm(F_new - F) / (np.linalg.norm(F) + 1e-300))
        F = F_new
        if rel < tol:
            converged = True
            break

    if not converged and warn:
        warnings.warn(
            f"Matrix completion did not converge in {max_iter} iterations "
            f"(relative change {rel:.3g} > tol {tol:g}); increase max_iter.",
            RuntimeWarning,
            stacklevel=3,
        )
    return {
        "fit": F,
        "L": L,
        "fe": F - L,
        "singular_values": s_thr,
        "n_iter": it,
        "converged": converged,
        "rel_change": rel,
    }
