"""Analytical bias correction of FE logit / probit slopes.

``sp.feglm(family='logit' | 'probit', bias_correction=True)``. With unit
and/or period effects estimated as parameters, the slopes carry an
incidental-parameter bias of order ``1/T`` (from the unit effects) plus
``1/N`` (from the period effects). The correction is the analytical one of
Fernandez-Val and Weidner (2016) for strictly exogenous regressors
(``L = 0``, a classic panel), the estimator R ``alpaca::biasCorr``
computes; it is reproduced here from the published formulas and checked
against alpaca's output (``tests/reference_parity/test_feglm_biascorr_parity.py``).

For a binary model with index ``eta``, cdf ``F``, density ``f``,
``h = f / (F (1 - F))`` and Fisher weight ``w = f h``, let ``x~`` be the
``w``-weighted residuals of the regressors on the effects and
``H = sum w x~ x~'``. For each effect dimension ``k``,

    b_k = sum_g [ sum_{i in g} f'_i h_i x~_i / 2 ] / [ sum_{i in g} w_i ]

and the corrected slopes are ``beta~ = beta^ + H^{-1} sum_k b_k``. The
effects are then re-estimated at ``beta~`` (offset ``x' beta~``) and the
variance is evaluated there, without small-sample factors, as alpaca
reports it: ``H^{-1}`` (``vcov='iid'``), the sandwich with scores
``(y - F) h x~`` (``'hetero'``), or its one-way cluster sum (``{'CRV1':
c}``).

References
----------
fernandezval2016individual
"""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import special, stats

from ..exceptions import MethodIncompatibility, NumericalInstability


def _binary(family: str, eta: np.ndarray) -> Tuple[np.ndarray, ...]:
    """``F``, ``f``, ``f'`` and ``h`` at ``eta``."""
    if family == "logit":
        F = special.expit(eta)
        f = F * (1.0 - F)
        return F, f, f * (1.0 - 2.0 * F), np.ones_like(F)
    F = special.ndtr(eta)
    f = stats.norm.pdf(eta)
    V = np.clip(F * (1.0 - F), 1e-300, None)
    return F, f, -eta * f, f / V


def _resid(X: np.ndarray, codes: List[np.ndarray], w: np.ndarray) -> np.ndarray:
    from ..fast._fe_direct import residualise_codes

    out, ok = residualise_codes(np.array(X, dtype=float), codes, w)
    if not ok:  # pragma: no cover - the direct solve converges here
        raise NumericalInstability(
            "feglm(bias_correction=True): could not residualise on the effects.",
            recovery_hint="Use coarser fixed effects.",
        )
    return out


def _refit_effects(
    family: str,
    y: np.ndarray,
    offset: np.ndarray,
    fe_start: np.ndarray,
    codes: List[np.ndarray],
    maxiter: int = 500,
) -> np.ndarray:
    """Effects maximising the likelihood with the index offset fixed."""
    fe = fe_start.copy()
    for _ in range(maxiter):
        eta = offset + fe
        F, f, _, h = _binary(family, eta)
        w = f * h
        z = eta + (y - F) / np.where(f > 0, f, 1e-300)
        r = _resid((z - offset)[:, None], codes, w)[:, 0]
        new = (z - offset) - r
        if np.max(np.abs(new - fe)) <= 1e-12 * (1.0 + np.max(np.abs(new))):
            _ret: np.ndarray = new
            return _ret
        fe = new
    raise NumericalInstability(  # pragma: no cover - IRLS converges here
        "feglm(bias_correction=True): the effects at the corrected slopes "
        "did not converge.",
        recovery_hint="Fit without bias_correction=.",
    )


def bias_correct(
    family: str,
    y: np.ndarray,
    X: np.ndarray,
    eta: np.ndarray,
    beta: np.ndarray,
    codes: List[np.ndarray],
    vcov: str,
    cluster_codes: Any = None,
) -> Tuple[np.ndarray, np.ndarray, Dict[str, Any]]:
    """Corrected slopes, their covariance and the bias terms."""
    if len(codes) not in (1, 2):
        raise MethodIncompatibility(
            "feglm(bias_correction=True) takes one or two absorbed effects "
            f"(unit, or unit and period); got {len(codes)}.",
            recovery_hint="Absorb the unit (and period) effects only.",
        )
    F, f, fp, h = _binary(family, eta)
    w = f * h
    Xt = _resid(X, codes, w)
    H = (Xt.T * w) @ Xt
    terms = []
    for c in codes:
        G = int(c.max()) + 1
        num = np.column_stack(
            [
                np.bincount(c, weights=0.5 * fp * h * Xt[:, j], minlength=G)
                for j in range(X.shape[1])
            ]
        )
        den = np.bincount(c, weights=w, minlength=G)
        terms.append((num / den[:, None]).sum(axis=0))
    shift = np.linalg.solve(H, np.sum(terms, axis=0))
    beta_bc = beta + shift

    offset = X @ beta_bc
    fe = _refit_effects(family, y, offset, eta - X @ beta, codes)
    eta_bc = offset + fe
    F, f, _, h = _binary(family, eta_bc)
    w = f * h
    Xt = _resid(X, codes, w)
    Hinv = np.linalg.inv((Xt.T * w) @ Xt)
    if vcov == "iid":
        V = Hinv
    else:
        S = ((y - F) * h)[:, None] * Xt
        if vcov == "hc1":
            V = Hinv @ (S.T @ S) @ Hinv
        else:
            cc = np.asarray(cluster_codes, dtype=np.int64)
            G = int(cc.max()) + 1
            Sg = np.column_stack(
                [
                    np.bincount(cc, weights=S[:, j], minlength=G)
                    for j in range(S.shape[1])
                ]
            )
            V = Hinv @ (Sg.T @ Sg) @ Hinv
    info = {
        "bias_correction": "analytical (Fernandez-Val and Weidner 2016, L=0)",
        "coef_uncorrected": beta.tolist(),
        "bias_shift": shift.tolist(),
    }
    return beta_bc, V, {**info, "eta": eta_bc, "mu": F}


def apply(out: Any, p: Any, keep: np.ndarray, eta: np.ndarray, family: str) -> Any:
    """Replace the slopes, standard errors and tests of a native feglm fit
    ``out`` by their bias-corrected versions."""
    from ._native import _attach_fe, _inference

    if not p.fe:
        raise MethodIncompatibility(
            "feglm(bias_correction=True) needs absorbed effects ('| unit + period').",
            recovery_hint="Add the effects after '|', or drop bias_correction=.",
        )
    if isinstance(p.cluster, list):
        raise MethodIncompatibility(
            "feglm(bias_correction=True): two-way clustering is not offered "
            "with the bias correction.",
            recovery_hint="Cluster on one variable, or use vcov='hetero'.",
        )
    if p.weights is not None:
        raise MethodIncompatibility(
            "feglm(bias_correction=True) is defined for unweighted fits.",
            recovery_hint="Drop weights=, or drop bias_correction=.",
        )
    keep = np.asarray(keep, dtype=bool)
    rows = p.data.loc[keep] if not keep.all() else p.data
    y = rows[p.lhs].to_numpy(dtype=float)
    X = rows[p.rhs].to_numpy(dtype=float)
    codes = [pd.factorize(rows[c], sort=False)[0].astype(np.int64) for c in p.fe]
    cl = pd.factorize(rows[p.cluster], sort=False)[0] if p.vcov == "cr1" else None
    beta = out.params.to_numpy(dtype=float)
    beta_bc, V, info = bias_correct(
        family, y, X, np.asarray(eta, dtype=float), beta, codes, p.vcov, cl
    )
    params = pd.Series(beta_bc, index=out.params.index)
    se, t, pv = _inference(params, V, p, len(y), normal=True)
    out.params = params
    out.std_errors = se
    out.tvalues = t
    out.pvalues = pv
    out.data_info["var_cov"] = V
    out.data_info["fitted_values"] = info["mu"]
    out.data_info["residuals"] = y - info["mu"]
    out.data_info.pop("df_inference", None)
    out.model_info["bias_correction"] = info["bias_correction"]
    out.model_info["coef_uncorrected"] = dict(
        zip(params.index, info["coef_uncorrected"])
    )
    out.model_info["bias_shift"] = dict(zip(params.index, info["bias_shift"]))
    out.model_info["vcov_type_detail"] = (
        f"{out.model_info.get('vcov_type_detail')}; at the corrected slopes, "
        "no small-sample factors (alpaca::biasCorr)"
    )
    for key in ("Log-likelihood", "Deviance"):
        out.diagnostics.pop(key, None)
    _attach_fe(out, p, keep, info["eta"], list(p.rhs), beta_bc)
    return out


__all__ = ["apply", "bias_correct"]
