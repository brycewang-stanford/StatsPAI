"""
Choo-Siow transferable-utility matching model (``sp.choo_siow``).

Kernels are in ``_choo_siow_core.py`` (IPFP, the saturated surplus and its
delta-method covariance, the weighted Poisson GLM); the result classes in
``_choo_siow_result.py``.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from ..core._call_record import records_call
from ..exceptions import (
    AssumptionWarning,
    ColumnNotFound,
    ConvergenceWarning,
    DataInsufficient,
    MethodIncompatibility,
)
from ._choo_siow_core import nonparametric_phi, poisson_glm_fit
from ._choo_siow_result import ChooSiowResult, MatchingEquilibrium

__all__ = ["choo_siow", "ChooSiowResult", "MatchingEquilibrium"]


def _from_households(
    data: pd.DataFrame, man: str, woman: str
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    for col in (man, woman):
        if col not in data.columns:
            raise ColumnNotFound(
                f"choo_siow: column {col!r} not found.",
                recovery_hint=f"Available columns: {list(data.columns)[:20]}",
            )
    m = data[man]
    w = data[woman]
    both_missing = m.isna() & w.isna()
    if both_missing.any():
        raise DataInsufficient(
            f"choo_siow: {int(both_missing.sum())} household(s) have neither a "
            "man's nor a woman's type.",
            recovery_hint="Each row is a household: a couple, a single man "
            "(woman type missing) or a single woman (man type missing).",
        )
    men_types = pd.Index(sorted(m.dropna().unique()))
    women_types = pd.Index(sorted(w.dropna().unique()))
    couples = data.loc[m.notna() & w.notna()]
    mu = (
        pd.crosstab(couples[man], couples[woman])
        .reindex(index=men_types, columns=women_types, fill_value=0)
        .astype(float)
    )
    sm = m[w.isna()].value_counts().reindex(men_types, fill_value=0).astype(float)
    sw = w[m.isna()].value_counts().reindex(women_types, fill_value=0).astype(float)
    mu.index.name = None
    mu.columns.name = None
    return mu, sm, sw


def _from_table(
    mu: Any, singles_men: Any, singles_women: Any
) -> Tuple[pd.DataFrame, pd.Series, pd.Series]:
    if singles_men is None or singles_women is None:
        raise MethodIncompatibility(
            "choo_siow: a table input needs mu, singles_men and singles_women.",
            recovery_hint="Without singles Phi is identified only up to "
            "additive man- and woman-type terms; this model needs them.",
        )
    if isinstance(mu, pd.DataFrame):
        mu_df = mu.astype(float)
    else:
        arr = np.asarray(mu, dtype=float)
        if arr.ndim != 2:
            raise MethodIncompatibility(
                f"choo_siow: mu must be 2-D (men x women); got shape {arr.shape}.",
                recovery_hint="Rows are man types, columns woman types.",
            )
        mu_df = pd.DataFrame(arr)
    return (
        mu_df,
        _singles(singles_men, mu_df.index),
        _singles(singles_women, mu_df.columns),
    )


def _singles(values: Any, types: pd.Index) -> pd.Series:
    """Single counts aligned to ``types`` (missing types become NaN)."""
    if isinstance(values, pd.Series):
        return values.reindex(types).astype(float)
    arr = np.asarray(values, dtype=float).ravel()
    if arr.size != len(types):
        raise MethodIncompatibility(
            f"choo_siow: {arr.size} single counts for {len(types)} types.",
            recovery_hint="One count per type, in the row / column order of mu.",
        )
    return pd.Series(arr, index=types)


def _basis_array(basis: Any, mu: pd.DataFrame) -> Tuple[np.ndarray, list]:
    X, Y = mu.shape
    if isinstance(basis, dict):
        names = [str(k) for k in basis]
        mats = []
        for k, v in basis.items():
            if isinstance(v, pd.DataFrame):
                v = v.reindex(index=mu.index, columns=mu.columns)
            arr = np.asarray(v, dtype=float)
            if arr.shape != (X, Y):
                raise MethodIncompatibility(
                    f"choo_siow: basis {k!r} has shape {arr.shape}, expected {(X, Y)}.",
                    recovery_hint="Each basis function is a men x women matrix.",
                )
            mats.append(arr)
        B = np.stack(mats, axis=2)
    else:
        B = np.asarray(basis, dtype=float)
        if B.ndim != 3 or B.shape[:2] != (X, Y):
            raise MethodIncompatibility(
                f"choo_siow: basis must be a dict of {(X, Y)} matrices or an "
                f"array of shape ({X}, {Y}, K); got {B.shape}.",
                recovery_hint="e.g. basis={'same_type': np.eye(X)}",
            )
        names = [f"phi_{k}" for k in range(B.shape[2])]
    if not np.all(np.isfinite(B)):
        raise MethodIncompatibility(
            "choo_siow: basis functions contain NaN / inf.",
            recovery_hint="Every (x, y) cell needs a finite value.",
        )
    return B, names


@records_call("choo_siow")
def choo_siow(
    data: Optional[pd.DataFrame] = None,
    *,
    man: Optional[str] = None,
    woman: Optional[str] = None,
    mu: Optional[Any] = None,
    singles_men: Optional[Any] = None,
    singles_women: Optional[Any] = None,
    basis: Optional[Any] = None,
    alpha: float = 0.05,
    tol: float = 1e-12,
    maxiter: int = 200,
) -> ChooSiowResult:
    """
    Choo-Siow transferable-utility marriage-market model.

    Each man of type ``x`` and woman of type ``y`` who marry share a joint
    surplus ``Phi_xy`` plus idiosyncratic type-I extreme-value tastes over
    partner *types* (and over staying single). In the large market the
    equilibrium satisfies ``mu_xy = sqrt(mu_x0 mu_0y) exp(Phi_xy / 2)``, so
    ``Phi`` is identified from one cross-section of marriages and singles.

    Parameters
    ----------
    data : pd.DataFrame, optional
        Household-level data, one row per household: a couple (both type
        columns set), a single man (``woman`` missing) or a single woman
        (``man`` missing).
    man, woman : str, optional
        Columns holding the man's and the woman's type (with ``data``).
    mu : array-like or pd.DataFrame, optional
        Instead of ``data``: the couples table ``mu_xy``, men's types in rows.
    singles_men, singles_women : array-like or pd.Series, optional
        Counts of single men ``mu_x0`` and single women ``mu_0y`` (with
        ``mu``).
    basis : dict or ndarray, optional
        Basis functions for ``Phi_xy = sum_k beta_k phi_k(x, y)``: a dict
        ``name -> (X, Y) matrix`` or an array ``(X, Y, K)``. Omitted: the
        nonparametric (saturated) surplus.
    alpha : float, default 0.05
        Level of the intervals in :meth:`ChooSiowResult.coef_table`.
    tol : float, default 1e-12
        Newton tolerance of the parametric fit (Newton decrement).
    maxiter : int, default 200
        Newton iterations for the parametric fit.

    Returns
    -------
    ChooSiowResult
        ``phi`` / ``phi_se`` (men by women), ``params`` / ``std_errors``,
        ``fitted_mu`` and the ``counterfactual()`` solver.

    Notes
    -----
    Nonparametric: ``Phi_xy = log(mu_xy^2 / (mu_x0 mu_0y))``, the formula of
    [@choo2006who] (their total gains are ``Phi / 2``). Under multinomial
    sampling of households its delta-method covariance is
    ``4 [xy = x'y'] / mu_xy + [x = x'] / mu_x0 + [y = y'] / mu_0y``. The
    estimate does not depend on the sampling rate (the log ratio is
    homogeneous of degree zero in the counts). An empty couple cell gives
    ``-inf`` (warned); an empty single cell is an error, since it makes
    every surplus of that type infinite.

    Parametric: the weighted Poisson GLM with two-way fixed effects of
    [@galichon2024estimating] (weights 2 on couple cells, 1 on single
    cells). Its first-order conditions are the accounting identities at
    the observed margins and the moment-matching conditions
    ``sum_xy phi_k (mu_hat_xy - mu_xy(beta)) = 0``; standard errors are the
    sandwich under multinomial sampling.

    ``counterfactual()`` re-solves the equilibrium by IPFP
    [@galichon2022cupid] for new type margins and/or a new surplus.

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np
    >>> mu = np.array([[300.0, 80.0, 20.0], [90.0, 260.0, 70.0], [25.0, 85.0, 310.0]])
    >>> res = sp.choo_siow(mu=mu, singles_men=[120, 100, 90],
    ...                    singles_women=[110, 105, 95])
    >>> res.phi.shape
    (3, 3)
    >>> X, Y = mu.shape
    >>> d = np.abs(np.subtract.outer(np.arange(X), np.arange(Y)))
    >>> par = sp.choo_siow(mu=mu, singles_men=[120, 100, 90],
    ...                    singles_women=[110, 105, 95],
    ...                    basis={"const": np.ones((X, Y)), "distance": d})
    >>> list(par.params.index)
    ['const', 'distance']
    >>> eq = par.counterfactual(new_margins=([500, 500, 500], [500, 500, 500]))
    >>> bool(eq.max_margin_error < 1e-10)
    True

    References
    ----------
    [@choo2006who] [@galichon2022cupid] [@galichon2024estimating]
    """
    if not (0 < float(alpha) < 1):
        raise MethodIncompatibility(
            f"choo_siow: alpha must be in (0, 1), got {alpha!r}.",
            recovery_hint="alpha=0.05 gives 95% intervals.",
        )
    if data is not None:
        if mu is not None or singles_men is not None or singles_women is not None:
            raise MethodIncompatibility(
                "choo_siow: give either household data or a table, not both.",
                recovery_hint="data=..., man=..., woman=... OR mu=..., "
                "singles_men=..., singles_women=...",
            )
        if man is None or woman is None:
            raise MethodIncompatibility(
                "choo_siow: household data need `man` and `woman` type columns.",
                recovery_hint="man='husband_educ', woman='wife_educ'",
            )
        mu_df, sm, sw = _from_households(data, man, woman)
        source = "households"
    elif mu is not None:
        mu_df, sm, sw = _from_table(mu, singles_men, singles_women)
        source = "table"
    else:
        raise MethodIncompatibility(
            "choo_siow: no data.",
            recovery_hint="Pass household data (data, man, woman) or a table "
            "(mu, singles_men, singles_women).",
        )

    M = mu_df.to_numpy(dtype=float)
    a = sm.to_numpy(dtype=float)
    b = sw.to_numpy(dtype=float)
    X, Y = M.shape
    if X < 1 or Y < 1:
        raise DataInsufficient(
            "choo_siow: no couples table.",
            recovery_hint="Need at least one man type and one woman type.",
        )
    for name, arr in (("mu", M), ("singles_men", a), ("singles_women", b)):
        if not np.all(np.isfinite(arr)) or np.any(arr < 0):
            raise MethodIncompatibility(
                f"choo_siow: {name} must be finite and non-negative counts.",
                recovery_hint="Check for NaN (missing types after alignment) "
                "or negative entries.",
            )
    zero_single_m = [mu_df.index[i] for i in np.where(a <= 0)[0]]
    zero_single_w = [mu_df.columns[j] for j in np.where(b <= 0)[0]]
    if zero_single_m or zero_single_w:
        raise DataInsufficient(
            "choo_siow: some types have no singles "
            f"(men {zero_single_m}, women {zero_single_w}); their surplus is "
            "+inf in the data, so the model is not estimable for them.",
            recovery_hint="Merge sparse types, or drop the type from the market.",
            diagnostics={
                "men": list(map(str, zero_single_m)),
                "women": list(map(str, zero_single_w)),
            },
        )
    if M.sum() <= 0:
        raise DataInsufficient(
            "choo_siow: no couples in the data.",
            recovery_hint="The surplus is identified from couples relative to singles.",
        )

    phi_np, V_np = nonparametric_phi(M, a, b)
    cell_names = [f"{x}|{y}" for x in mu_df.index for y in mu_df.columns]
    phi_np_df = pd.DataFrame(phi_np, index=mu_df.index, columns=mu_df.columns)
    se_np = np.sqrt(np.diag(V_np)).reshape(X, Y)
    se_np_df = pd.DataFrame(se_np, index=mu_df.index, columns=mu_df.columns)
    zero_cells = [(mu_df.index[i], mu_df.columns[j]) for i, j in zip(*np.where(M <= 0))]
    diagnostics: Dict[str, Any] = {"zero_couple_cells": zero_cells}
    model_info: Dict[str, Any] = {
        "model_type": "Choo-Siow TU matching",
        "input": source,
        "n_men_types": X,
        "n_women_types": Y,
        "heterogeneity": "type-I extreme value, scale 1 (homoskedastic)",
        "sampling": "multinomial over households",
    }

    if basis is None:
        if zero_cells:
            warnings.warn(
                AssumptionWarning(
                    f"choo_siow: {len(zero_cells)} couple cell(s) are empty, so "
                    "their nonparametric surplus is -inf (standard error NaN).",
                    recovery_hint="Merge sparse types or pass `basis=` for a "
                    "parametric surplus, which uses the zeros as information.",
                    diagnostics={
                        "zero_couple_cells": [list(map(str, c)) for c in zero_cells]
                    },
                ),
                stacklevel=2,
            )
        return ChooSiowResult(
            method="nonparametric",
            phi=phi_np_df,
            phi_se=se_np_df,
            phi_nonparametric=phi_np_df,
            phi_nonparametric_se=se_np_df,
            params=pd.Series(phi_np.ravel(), index=cell_names),
            std_errors=pd.Series(np.sqrt(np.diag(V_np)), index=cell_names),
            vcov_params=pd.DataFrame(V_np, index=cell_names, columns=cell_names),
            mu=mu_df,
            singles_men=sm,
            singles_women=sw,
            fitted_mu=mu_df.copy(),
            fitted_singles_men=sm.copy(),
            fitted_singles_women=sw.copy(),
            alpha=alpha,
            model_info=model_info,
            diagnostics=diagnostics,
        )

    B, names = _basis_array(basis, mu_df)
    fit = poisson_glm_fit(B, M, a, b, tol=tol, maxiter=maxiter)
    beta = np.asarray(fit["beta"], dtype=float)
    Vb = np.asarray(fit["vcov_beta"], dtype=float)
    P = B.reshape(X * Y, -1)
    phi_par = (P @ beta).reshape(X, Y)
    phi_par_se = np.sqrt(np.clip(np.einsum("ik,kl,il->i", P, Vb, P), 0, None)).reshape(
        X, Y
    )
    if not fit["converged"]:
        warnings.warn(
            ConvergenceWarning(
                f"choo_siow: Newton did not converge in {maxiter} iterations "
                f"(max |score| {fit['max_abs_score']:.2e}).",
                recovery_hint="Raise maxiter, or rescale the basis functions.",
                diagnostics={"max_abs_score": fit["max_abs_score"]},
            ),
            stacklevel=2,
        )
    diagnostics.update(
        {
            "converged": bool(fit["converged"]),
            "n_iter": int(fit["n_iter"]),  # type: ignore[call-overload]
            "max_abs_score": float(fit["max_abs_score"]),  # type: ignore[arg-type]
            "moment_matching": pd.DataFrame(
                {
                    "observed": np.einsum("xyk,xy->k", B, M),
                    "fitted": np.einsum("xyk,xy->k", B, np.asarray(fit["fitted_mu"])),
                },
                index=names,
            ),
        }
    )
    model_info.update(
        {
            "estimator": "weighted Poisson GLM (two-way fixed effects)",
            "basis_names": names,
            "se_type": "sandwich, multinomial sampling",
        }
    )
    return ChooSiowResult(
        method="parametric",
        phi=pd.DataFrame(phi_par, index=mu_df.index, columns=mu_df.columns),
        phi_se=pd.DataFrame(phi_par_se, index=mu_df.index, columns=mu_df.columns),
        phi_nonparametric=phi_np_df,
        phi_nonparametric_se=se_np_df,
        params=pd.Series(beta, index=names),
        std_errors=pd.Series(np.sqrt(np.clip(np.diag(Vb), 0, None)), index=names),
        vcov_params=pd.DataFrame(Vb, index=names, columns=names),
        mu=mu_df,
        singles_men=sm,
        singles_women=sw,
        fitted_mu=pd.DataFrame(
            np.asarray(fit["fitted_mu"]), index=mu_df.index, columns=mu_df.columns
        ),
        fitted_singles_men=pd.Series(
            np.asarray(fit["fitted_singles_men"]), index=mu_df.index
        ),
        fitted_singles_women=pd.Series(
            np.asarray(fit["fitted_singles_women"]), index=mu_df.columns
        ),
        alpha=alpha,
        model_info=model_info,
        diagnostics=diagnostics,
        basis=B,
    )
