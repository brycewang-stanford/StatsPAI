"""
Shared machinery for proxy-variable production function estimation.

All proxy-variable estimators (Olley-Pakes 1996, Levinsohn-Petrin 2003,
Ackerberg-Caves-Frazer 2015, Wooldridge 2009) share three ingredients:

1. A polynomial expansion of (free, state, proxy) inputs to approximate
   the unobserved control function ``h(proxy, state)`` and the productivity
   process ``g(omega_{t-1})``.
2. A within-panel lag operator that respects the (id, time) index.
3. An OLS / GMM second stage that solves for the structural parameters
   conditional on a stage-one nonparametric fit.

Keep this file framework-agnostic — only NumPy / pandas / scipy. No imports
from sibling submodules so it can be reused by ``markup`` and any future
extensions (translog, GNR, etc.).
"""

from __future__ import annotations

from itertools import combinations_with_replacement
from typing import Dict, List, Sequence, Tuple

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Polynomial basis
# ---------------------------------------------------------------------------

def polynomial_basis(
    X: np.ndarray,
    degree: int = 3,
    include_intercept: bool = True,
) -> Tuple[np.ndarray, List[Tuple[int, ...]]]:
    """Build a full polynomial basis up to ``degree`` in the columns of ``X``.

    Returns the design matrix and the list of multi-indices (one per term,
    e.g. ``(1, 0, 2)`` means ``x0 * x2 * x2``). Intercept is the first column
    when ``include_intercept=True``.

    The basis is used both for the stage-1 control function (in OP/LP/ACF)
    and for the productivity Markov polynomial in the second stage.
    """
    n, k = X.shape
    terms: List[Tuple[int, ...]] = []
    cols: List[np.ndarray] = []
    if include_intercept:
        terms.append(())
        cols.append(np.ones(n))
    for d in range(1, degree + 1):
        for idx in combinations_with_replacement(range(k), d):
            terms.append(idx)
            col = np.ones(n)
            for j in idx:
                col = col * X[:, j]
            cols.append(col)
    return np.column_stack(cols), terms


# ---------------------------------------------------------------------------
# Functional form: input expansion (Cobb-Douglas vs translog)
# ---------------------------------------------------------------------------

def expand_inputs(
    X_raw: np.ndarray,
    names: Sequence[str],
    functional_form: str = "cobb-douglas",
) -> Tuple[np.ndarray, List[str]]:
    """Expand a raw input matrix according to ``functional_form``.

    Cobb-Douglas (default) returns the matrix unchanged with one column
    per input.  Translog adds the second-order terms

        0.5 * x_j**2          (own quadratic, with the 0.5 factor so that
                               the elasticity wrt x_j reads off as
                               beta_j + beta_jj * x_j + sum_k beta_jk * x_k)
        x_j * x_k  (j < k)    (cross terms)

    Returns ``(X_expanded, column_names)`` where the names follow the
    convention used by :func:`elasticities_at` and the markup helper:

        cobb-douglas   ->  ["l", "k"]                (or whatever the
                                                     caller passes)
        translog       ->  ["l", "k", "ll", "kk", "lk"]

    Parameters
    ----------
    X_raw : ndarray, shape (n, p)
        Raw input columns (free + state, in the order the caller wants
        the linear coefficients to appear in).
    names : sequence of str
        One name per raw column.  Used to label the cross terms.
    functional_form : {'cobb-douglas', 'translog'}, default 'cobb-douglas'
    """
    fform = functional_form.lower().replace("_", "-")
    if fform in ("cobb-douglas", "cd"):
        return X_raw, list(names)
    if fform != "translog":
        raise ValueError(
            f"Unknown functional_form {functional_form!r}; "
            "choose 'cobb-douglas' or 'translog'."
        )

    n, p = X_raw.shape
    cols: List[np.ndarray] = [X_raw[:, j] for j in range(p)]
    out_names: List[str] = list(names)
    # Own quadratic terms (with 0.5 factor — see docstring).
    for j in range(p):
        cols.append(0.5 * X_raw[:, j] ** 2)
        out_names.append(f"{names[j]}{names[j]}")
    # Cross terms.
    for j in range(p):
        for k in range(j + 1, p):
            cols.append(X_raw[:, j] * X_raw[:, k])
            out_names.append(f"{names[j]}{names[k]}")
    return np.column_stack(cols), out_names


def elasticities_at(
    X_raw: np.ndarray,
    names: Sequence[str],
    coef: Dict[str, float],
    functional_form: str = "cobb-douglas",
) -> pd.DataFrame:
    """Firm-time output elasticities under ``functional_form``.

    For Cobb-Douglas, returns a DataFrame with constant columns equal to
    the linear coefficients (one row per observation, repeated).

    For translog, the elasticity wrt input j is

        theta_j_it = beta_j + beta_jj * x_j_it + sum_{k != j} beta_jk * x_k_it

    where the cross-term coefficient is read by sorted name:
    ``coef["lk"]`` covers both ``l*k`` and ``k*l``.  This matches the
    naming convention from :func:`expand_inputs`.

    Parameters
    ----------
    X_raw : ndarray, shape (n, p)
        Raw input columns.
    names : sequence of str
        One name per raw column.
    coef : dict
        Coefficient map (output of :class:`ProductionResult`'s ``coef``
        attribute).  Linear keys mandatory, quadratic / cross keys
        required only for translog.
    functional_form : {'cobb-douglas', 'translog'}

    Returns
    -------
    pd.DataFrame
        Shape (n, p), columns = ``names``, values = firm-time elasticities.
    """
    n, p = X_raw.shape
    fform = functional_form.lower().replace("_", "-")
    if fform in ("cobb-douglas", "cd"):
        block = np.tile(np.asarray([coef[nm] for nm in names], dtype=float), (n, 1))
        return pd.DataFrame(block, columns=list(names))

    if fform != "translog":
        raise ValueError(
            f"Unknown functional_form {functional_form!r}; "
            "choose 'cobb-douglas' or 'translog'."
        )

    out = np.zeros((n, p))
    for j, nj in enumerate(names):
        # Linear term beta_j.
        out[:, j] = coef[nj]
        # Own quadratic: derivative of 0.5 * beta_jj * x_j**2 is beta_jj * x_j.
        own_key = f"{nj}{nj}"
        if own_key in coef:
            out[:, j] += coef[own_key] * X_raw[:, j]
        # Cross terms — look up the canonical key.  The canonical name
        # follows the order in ``names`` (matching ``expand_inputs``),
        # so for a pair (j, ki) the key is "{names[j]}{names[ki]}" if
        # j < ki else "{names[ki]}{names[j]}".
        for ki, nki in enumerate(names):
            if ki == j:
                continue
            key_jk = f"{nj}{nki}" if j < ki else f"{nki}{nj}"
            if key_jk in coef:
                out[:, j] += coef[key_jk] * X_raw[:, ki]
    return pd.DataFrame(out, columns=list(names))


# ---------------------------------------------------------------------------
# Within-panel lag
# ---------------------------------------------------------------------------

def panel_lag(
    df: pd.DataFrame,
    column: str,
    panel_id: str,
    time: str,
    lag: int = 1,
) -> pd.Series:
    """Return the within-firm lag of ``column``.

    Assumes ``time`` is comparable (int / pandas datetime). Missing lagged
    values are returned as ``NaN``; downstream callers must drop them
    consistently across regressors / instruments before fitting.
    """
    df = df.sort_values([panel_id, time])
    return df.groupby(panel_id, sort=False)[column].shift(lag)


# ---------------------------------------------------------------------------
# Stage-1 control function
# ---------------------------------------------------------------------------

def stage_one_phi(
    y: np.ndarray,
    Z: np.ndarray,
    degree: int = 3,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Fit the stage-1 polynomial control function.

    ``y_it = Phi(Z_it) + eta_it``

    where ``Z_it = (free, state, proxy)`` and ``Phi`` is approximated by a
    polynomial of degree ``degree``. Returns ``(phi_hat, eta_hat, gamma)``
    where ``gamma`` are the polynomial coefficients (used by Wooldridge GMM
    to share the same nonparametric h between stages).
    """
    P, _ = polynomial_basis(Z, degree=degree)
    gamma, *_ = np.linalg.lstsq(P, y, rcond=None)
    phi_hat = P @ gamma
    eta_hat = y - phi_hat
    return phi_hat, eta_hat, gamma


# ---------------------------------------------------------------------------
# Stage-2 productivity Markov + GMM moment evaluator
# ---------------------------------------------------------------------------

def productivity_residual(
    omega: np.ndarray,
    omega_lag: np.ndarray,
    degree: int = 3,
) -> Tuple[np.ndarray, np.ndarray]:
    """Polynomial AR for ``omega_it = g(omega_{i,t-1}) + xi_it``.

    Returns ``(xi, theta)`` where ``theta`` are the polynomial coefficients
    of ``g``.  Caller must pre-drop NaNs introduced by the lag operator.
    """
    Z = omega_lag.reshape(-1, 1)
    P, _ = polynomial_basis(Z, degree=degree)
    theta, *_ = np.linalg.lstsq(P, omega, rcond=None)
    xi = omega - P @ theta
    return xi, theta


def gmm_objective(
    beta: np.ndarray,
    *,
    phi_hat: np.ndarray,
    inputs: np.ndarray,           # shape (n, p) — same order as beta
    instruments: np.ndarray,      # shape (n, q) — q >= p
    panel_id: np.ndarray,
    time: np.ndarray,
    productivity_degree: int = 3,
    weight: np.ndarray | None = None,
) -> float:
    """Stage-2 GMM objective for OP/LP/ACF.

    Steps:
        omega_it       = phi_hat_it - inputs_it @ beta
        omega_{i,t-1}  = panel-lag of omega
        xi_it          = omega_it - g(omega_{i,t-1})  with g a polynomial
        moment(beta)   = mean(xi * instruments, axis=0)
        objective      = moment' * W * moment

    Identification uses ``q == p`` (just-identified) by default; if
    ``q > p`` the user supplies ``weight`` (e.g. an inverse-variance matrix).

    Inputs / instruments / panel_id / time must all be row-aligned.
    The function sorts the joint frame internally by ``(panel_id, time)``
    so any input order is accepted — instruments and xi stay paired.
    """
    omega = phi_hat - inputs @ beta

    # Build a single frame so omega, instruments, and panel_id/time are
    # sorted as one block — keeps moment pairing intact even if the
    # caller's data was unsorted.
    n_z = instruments.shape[1]
    df = pd.DataFrame({
        "omega": omega,
        "panel_id": panel_id,
        "time": time,
        **{f"_z{j}": instruments[:, j] for j in range(n_z)},
    })
    df = df.sort_values(["panel_id", "time"]).reset_index(drop=True)
    df["omega_lag"] = df.groupby("panel_id", sort=False)["omega"].shift(1)
    mask = df["omega_lag"].notna().to_numpy()
    if mask.sum() < 5:
        return 1e12  # penalize degenerate fits

    omega_t = df.loc[mask, "omega"].to_numpy()
    omega_l = df.loc[mask, "omega_lag"].to_numpy()
    xi, _ = productivity_residual(omega_t, omega_l, degree=productivity_degree)

    Z_used = df.loc[mask, [f"_z{j}" for j in range(n_z)]].to_numpy()
    moments = (xi[:, None] * Z_used).mean(axis=0)
    if weight is None:
        return float(moments @ moments)
    return float(moments @ weight @ moments)


# ---------------------------------------------------------------------------
# Cluster (firm) bootstrap helper
# ---------------------------------------------------------------------------

def firm_bootstrap_indices(
    panel_id: np.ndarray,
    rng: np.random.Generator,
) -> np.ndarray:
    """Return row indices for a single firm-cluster bootstrap resample.

    Standard practice for proxy-variable estimators (Wooldridge 2009 §4):
    resample firms (with replacement) and keep the entire firm history
    when sampled — preserves the within-firm time structure that AR(1)
    productivity identification depends on.
    """
    firms = pd.unique(panel_id)
    sampled = rng.choice(firms, size=len(firms), replace=True)
    rows: List[np.ndarray] = []
    # Build firm -> rows map once for O(n) bootstrap per draw.
    by_firm: dict = {}
    for i, fid in enumerate(panel_id):
        by_firm.setdefault(fid, []).append(i)
    for fid in sampled:
        rows.extend(by_firm[fid])
    return np.asarray(rows, dtype=np.int64)
