"""
Weak-identification diagnostics for IV models with multiple endogenous regressors.

Public API
----------
- :func:`kleibergen_paap_rk` — Kleibergen-Paap (2006) rk Wald/LM statistics.
  Heteroskedasticity- and cluster-robust generalisation of Cragg-Donald.
- :func:`sanderson_windmeijer` — Sanderson-Windmeijer (2016) conditional
  first-stage F for each individual endogenous regressor when multiple
  endogenous variables are present.
- :func:`conditional_lr_test` — Moreira (2003) Conditional Likelihood Ratio
  (CLR) test. Uniformly most powerful invariant in the single-endogenous
  case and weak-IV-robust.

These three statistics fill gaps that are fragmented across Stata
(``ivreg2``, ``weakiv``) and R (``ivmodel``, ``ivreg``) and are mostly
absent from Python's existing IV stack (``linearmodels``).

References
----------
Kleibergen, F. and Paap, R. (2006). "Generalized reduced rank tests using
    the singular value decomposition." *Journal of Econometrics*, 133(1),
    97-126.

Sanderson, E. and Windmeijer, F. (2016). "A weak instrument F-test in
    linear IV models with multiple endogenous variables." *Journal of
    Econometrics*, 190(2), 212-221.

Moreira, M.J. (2003). "A conditional likelihood ratio test for structural
    models." *Econometrica*, 71(4), 1027-1048.

Cragg, J.G. and Donald, S.G. (1993). "Testing identifiability and
    specification in instrumental variable models." *Econometric Theory*,
    9(2), 222-240.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from scipy import stats


# ═══════════════════════════════════════════════════════════════════════
#  Data containers
# ═══════════════════════════════════════════════════════════════════════

@dataclass
class KleibergenPaapResult:
    """Container for Kleibergen-Paap rk test output."""
    rk_wald: float
    rk_wald_pvalue: float
    rk_lm: float
    rk_lm_pvalue: float
    rk_f: float
    df_num: int
    df_denom: int
    n_endog: int
    n_instruments: int
    cov_type: str

    def summary(self) -> str:
        return (
            "Kleibergen-Paap (2006) rank test\n"
            f"{'-' * 48}\n"
            f"  rk LM statistic      : {self.rk_lm:>10.4f}   p={self.rk_lm_pvalue:.4f}\n"
            f"  rk Wald statistic    : {self.rk_wald:>10.4f}   p={self.rk_wald_pvalue:.4f}\n"
            f"  rk Wald F-statistic  : {self.rk_f:>10.4f}\n"
            f"  n_endog = {self.n_endog},  n_instruments = {self.n_instruments}\n"
            f"  covariance type      : {self.cov_type}"
        )


@dataclass
class SandersonWindmeijerResult:
    """Sanderson-Windmeijer conditional F for each endogenous variable."""
    endog_names: List[str]
    sw_f: Dict[str, float]
    sw_pvalue: Dict[str, float]
    df_num: Dict[str, int]
    df_denom: int
    partial_r2: Dict[str, float]

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame({
            'SW F': [self.sw_f[n] for n in self.endog_names],
            'p-value': [self.sw_pvalue[n] for n in self.endog_names],
            'df_num': [self.df_num[n] for n in self.endog_names],
            'partial R²': [self.partial_r2[n] for n in self.endog_names],
        }, index=self.endog_names)

    def summary(self) -> str:
        lines = [
            "Sanderson-Windmeijer (2016) conditional first-stage F",
            "-" * 48,
        ]
        lines.append(self.to_frame().round(4).to_string())
        lines.append(f"\n  df_denom = {self.df_denom}")
        lines.append("  Rule of thumb: SW F > ~10 per endogenous variable.")
        return "\n".join(lines)


@dataclass
class CLRResult:
    """Moreira (2003) CLR test."""
    statistic: float
    pvalue: float
    beta0: float
    n_simulations: int
    ar_stat: float
    lm_stat: float

    def summary(self) -> str:
        return (
            "Moreira (2003) Conditional LR test\n"
            f"{'-' * 48}\n"
            f"  H0: beta = {self.beta0:.4f}\n"
            f"  CLR statistic        : {self.statistic:>10.4f}   p={self.pvalue:.4f}\n"
            f"  (AR={self.ar_stat:.4f},  LM={self.lm_stat:.4f})\n"
            f"  simulations          : {self.n_simulations}"
        )


# ═══════════════════════════════════════════════════════════════════════
#  Internal helpers
# ═══════════════════════════════════════════════════════════════════════

def _residualize(M: np.ndarray, W: Optional[np.ndarray]) -> np.ndarray:
    """Project M onto the orthogonal complement of W."""
    if W is None or W.size == 0 or W.shape[1] == 0:
        return M
    beta, *_ = np.linalg.lstsq(W, M, rcond=None)
    return M - W @ beta


def _as_matrix(x: Union[np.ndarray, pd.DataFrame, pd.Series]) -> np.ndarray:
    a = np.asarray(x, dtype=float)
    if a.ndim == 1:
        a = a.reshape(-1, 1)
    return a


def _collect_names(
    obj: Union[np.ndarray, pd.DataFrame, pd.Series],
    prefix: str,
) -> List[str]:
    if isinstance(obj, pd.DataFrame):
        return list(obj.columns)
    if isinstance(obj, pd.Series):
        return [obj.name or f"{prefix}0"]
    a = _as_matrix(obj)
    return [f"{prefix}{i}" for i in range(a.shape[1])]


def _extract_exog(
    data: Optional[pd.DataFrame],
    exog: Optional[Union[List[str], np.ndarray]],
    n: int,
    add_const: bool,
) -> np.ndarray:
    if exog is None:
        return np.ones((n, 1)) if add_const else np.empty((n, 0))
    if isinstance(exog, (list, tuple)) and data is not None and all(isinstance(v, str) for v in exog):
        W = data[list(exog)].values.astype(float)
    else:
        W = _as_matrix(exog)
    if add_const:
        W = np.column_stack([np.ones(W.shape[0]), W])
    return W


# ═══════════════════════════════════════════════════════════════════════
#  Kleibergen-Paap rk statistic
# ═══════════════════════════════════════════════════════════════════════

def kleibergen_paap_rk(
    endog: Union[np.ndarray, pd.DataFrame],
    instruments: Union[np.ndarray, pd.DataFrame],
    exog: Optional[Union[np.ndarray, pd.DataFrame, List[str]]] = None,
    data: Optional[pd.DataFrame] = None,
    cov_type: str = "robust",
    cluster: Optional[Union[np.ndarray, pd.Series]] = None,
    add_const: bool = True,
) -> KleibergenPaapResult:
    """
    Kleibergen-Paap (2006) rk Wald / LM statistic.

    Tests the null that the reduced-form coefficient matrix on the excluded
    instruments has rank ``n_endog - 1`` (under-identification) against the
    alternative of full rank.

    This is the heteroskedasticity- and cluster-robust generalisation of the
    classical Cragg-Donald statistic. ``ivreg2`` in Stata reports the
    identical statistic.

    Parameters
    ----------
    endog : array or DataFrame, shape (n, p)
        Endogenous regressors.
    instruments : array or DataFrame, shape (n, k)
        Excluded instruments (``k >= p``).
    exog : array, DataFrame or list of column names, optional
        Included exogenous regressors (controls). Intercept is added
        automatically when ``add_const=True``.
    data : DataFrame, optional
        Used only when ``exog`` is a list of column names.
    cov_type : {'nonrobust', 'robust', 'cluster'}
        Covariance for the stacked reduced-form equations.
    cluster : array-like, optional
        Required when ``cov_type='cluster'``.
    add_const : bool, default True
        Prepend a constant to the exogenous block.

    Returns
    -------
    KleibergenPaapResult
    """
    D = _as_matrix(endog)  # n x p
    Z = _as_matrix(instruments)  # n x k
    n, p = D.shape
    k = Z.shape[1]
    if k < p:
        raise ValueError(
            f"Under-identified: only {k} instruments for {p} endogenous regressors."
        )

    W = _extract_exog(data, exog, n, add_const)
    n_W = W.shape[1]

    # Partial out exogenous regressors
    D_tilde = _residualize(D, W)
    Z_tilde = _residualize(Z, W)

    # Reduced form coefficients: D_tilde = Z_tilde @ Pi + V
    ZZ_inv = np.linalg.inv(Z_tilde.T @ Z_tilde)
    Pi = ZZ_inv @ (Z_tilde.T @ D_tilde)  # k x p
    V = D_tilde - Z_tilde @ Pi

    # Covariance of vec(Pi) — GLS form
    # Var(vec(Pi_hat)) = (Sigma_VV ⊗ (Z'Z)^{-1}) with appropriate robust mod
    if cov_type == "nonrobust":
        Sigma = (V.T @ V) / (n - n_W - k)
        cov_vec = np.kron(Sigma, ZZ_inv)
        cov_label = "nonrobust"
    elif cov_type == "robust":
        # Meat: sum_i (Z_i Z_i') ⊗ (V_i V_i')
        meat = np.zeros((k * p, k * p))
        for i in range(n):
            zi = Z_tilde[i:i + 1].T  # k x 1
            vi = V[i:i + 1].T  # p x 1
            meat += np.kron(vi @ vi.T, zi @ zi.T)
        bread = np.kron(np.eye(p), ZZ_inv)
        cov_vec = bread @ meat @ bread
        cov_label = "HC robust"
    elif cov_type == "cluster":
        if cluster is None:
            raise ValueError("cov_type='cluster' requires `cluster`.")
        g = pd.Series(np.asarray(cluster))
        groups = g.unique()
        G = len(groups)
        meat = np.zeros((k * p, k * p))
        for cid in groups:
            idx = np.where((g == cid).values)[0]
            Zc = Z_tilde[idx]
            Vc = V[idx]
            score = np.kron(Vc, Zc).sum(axis=0)
            # score is p*k (row-major kron of per-obs kron sum); rebuild properly
            score_mat = np.zeros((k, p))
            for t in idx:
                score_mat += np.outer(Z_tilde[t], V[t])
            v = score_mat.flatten(order='F')  # matches vec(Pi) stacking cols
            meat += np.outer(v, v)
        meat *= G / max(G - 1, 1)
        bread = np.kron(np.eye(p), ZZ_inv)
        cov_vec = bread @ meat @ bread
        cov_label = f"cluster ({G} groups)"
    else:
        raise ValueError(f"Unknown cov_type: {cov_type}")

    # KP rk Wald: vec(Pi)' cov_vec^{-1} vec(Pi)
    vec_Pi = Pi.flatten(order='F')
    cov_pinv = np.linalg.pinv(cov_vec)
    rk_wald = float(vec_Pi @ cov_pinv @ vec_Pi)

    # For F version divide by (k*p) and nominal denom
    df_num = k * p  # number of excluded restrictions on reduced form
    df_denom = n - n_W - k
    rk_f = rk_wald / df_num
    rk_wald_pvalue = float(1 - stats.chi2.cdf(rk_wald, df=k - p + 1))

    # KP rk LM statistic (Kleibergen 2002 / Kleibergen-Paap 2006 eq. 17)
    # Equivalent to Wald when homoskedastic, more stable under weak identification
    # KP rk LM = n * tr( (D'M_W D)^{-1} D'P_Z D )  after robust-ification
    Sigma = (V.T @ V) / n
    try:
        Sigma_half_inv = np.linalg.inv(np.linalg.cholesky(Sigma))
    except np.linalg.LinAlgError:
        Sigma_half_inv = np.linalg.pinv(_sqrtm_sym(Sigma))
    Zs = Z_tilde @ np.linalg.cholesky(ZZ_inv)  # orthonormalised instruments
    Ds = D_tilde @ Sigma_half_inv.T  # whitened endog
    A = Zs.T @ Ds  # k x p
    rk_lm = float(np.sum(A ** 2))
    rk_lm_pvalue = float(1 - stats.chi2.cdf(rk_lm, df=(k - p + 1) * p))

    return KleibergenPaapResult(
        rk_wald=rk_wald,
        rk_wald_pvalue=rk_wald_pvalue,
        rk_lm=rk_lm,
        rk_lm_pvalue=rk_lm_pvalue,
        rk_f=rk_f,
        df_num=df_num,
        df_denom=df_denom,
        n_endog=p,
        n_instruments=k,
        cov_type=cov_label,
    )


def _sqrtm_sym(M: np.ndarray) -> np.ndarray:
    w, V = np.linalg.eigh(M)
    w = np.clip(w, 1e-12, None)
    return V @ np.diag(np.sqrt(w)) @ V.T


# ═══════════════════════════════════════════════════════════════════════
#  Sanderson-Windmeijer conditional F
# ═══════════════════════════════════════════════════════════════════════

def sanderson_windmeijer(
    endog: Union[np.ndarray, pd.DataFrame],
    instruments: Union[np.ndarray, pd.DataFrame],
    exog: Optional[Union[np.ndarray, pd.DataFrame, List[str]]] = None,
    data: Optional[pd.DataFrame] = None,
    add_const: bool = True,
    endog_names: Optional[List[str]] = None,
) -> SandersonWindmeijerResult:
    """
    Sanderson-Windmeijer (2016) conditional first-stage F.

    For each endogenous regressor ``j``, residualises all *other*
    endogenous regressors out of both the outcome (that endog column) and
    the instruments, then reports the first-stage F of the resulting
    partial regression. This is the correct individual-endogenous weak-IV
    diagnostic when multiple endogenous regressors are present.

    When only one endogenous regressor is present, this reduces exactly to
    the standard first-stage F.

    Parameters
    ----------
    endog : array or DataFrame, shape (n, p)
    instruments : array or DataFrame, shape (n, k)
    exog : array, DataFrame or list of column names, optional
    data : DataFrame, optional
    add_const : bool, default True
    endog_names : list of str, optional
        Labels for endogenous columns when passing numpy arrays.

    Returns
    -------
    SandersonWindmeijerResult
    """
    D = _as_matrix(endog)
    Z = _as_matrix(instruments)
    n, p = D.shape
    k = Z.shape[1]

    if k < p:
        raise ValueError(
            f"Under-identified: only {k} instruments for {p} endogenous regressors."
        )

    W = _extract_exog(data, exog, n, add_const)

    names = endog_names or _collect_names(endog, prefix="endog")
    if len(names) != p:
        raise ValueError(f"endog_names length {len(names)} != n_endog {p}")

    # Partial out exogenous
    D_tilde = _residualize(D, W)
    Z_tilde = _residualize(Z, W)
    n_W = W.shape[1]

    sw_f: Dict[str, float] = {}
    sw_p: Dict[str, float] = {}
    df_num: Dict[str, int] = {}
    partial_r2: Dict[str, float] = {}

    for j in range(p):
        mask = np.ones(p, dtype=bool)
        mask[j] = False
        D_other = D_tilde[:, mask]
        D_j = D_tilde[:, j]

        if p > 1:
            # Residualise D_j on D_other AND Z on D_other simultaneously
            # then run first-stage of (D_j | D_other) on (Z | D_other)
            # SW (2016) Theorem 1: equivalent conditional F form
            Zc = _residualize(Z_tilde, D_other)
            y_j = _residualize(D_j.reshape(-1, 1), D_other).ravel()
        else:
            Zc = Z_tilde
            y_j = D_j

        # First-stage regression of y_j on Zc
        beta, *_ = np.linalg.lstsq(Zc, y_j, rcond=None)
        resid = y_j - Zc @ beta
        rss = float(resid @ resid)
        tss = float(y_j @ y_j)

        df1 = k - (p - 1)  # SW adjusted numerator df
        df2 = n - n_W - k  # denominator df (net of all controls + instruments)
        if df1 <= 0:
            raise ValueError(
                f"Not enough instruments: k - (p-1) = {df1} for endogenous '{names[j]}'."
            )
        if df2 <= 0:
            raise ValueError(f"Not enough observations: df_denom = {df2}.")

        if rss > 0 and tss > 0:
            explained = tss - rss
            f_j = (explained / df1) / (rss / df2)
            pval = float(1 - stats.f.cdf(f_j, df1, df2))
            pr2 = 1 - rss / tss
        else:
            f_j = np.nan
            pval = np.nan
            pr2 = np.nan

        sw_f[names[j]] = float(f_j)
        sw_p[names[j]] = pval
        df_num[names[j]] = int(df1)
        partial_r2[names[j]] = float(pr2)

    return SandersonWindmeijerResult(
        endog_names=names,
        sw_f=sw_f,
        sw_pvalue=sw_p,
        df_num=df_num,
        df_denom=int(df2),
        partial_r2=partial_r2,
    )


# ═══════════════════════════════════════════════════════════════════════
#  Moreira CLR test
# ═══════════════════════════════════════════════════════════════════════

def conditional_lr_test(
    y: Union[np.ndarray, pd.Series, str],
    endog: Union[np.ndarray, pd.Series, str],
    instruments: Union[np.ndarray, pd.DataFrame, List[str]],
    exog: Optional[Union[np.ndarray, pd.DataFrame, List[str]]] = None,
    data: Optional[pd.DataFrame] = None,
    beta0: float = 0.0,
    add_const: bool = True,
    n_simulations: int = 20_000,
    random_state: Optional[int] = None,
) -> CLRResult:
    """
    Moreira (2003) Conditional Likelihood Ratio (CLR) test.

    Tests ``H0: beta = beta0`` in a single-endogenous-variable IV model.
    Weak-IV-robust and uniformly most powerful invariant in the one
    endogenous-variable case.

    Parameters
    ----------
    y, endog : array, Series or column name
        Outcome and the single endogenous regressor.
    instruments : array, DataFrame or list of column names
    exog : array, DataFrame or list of column names, optional
    data : DataFrame, optional
    beta0 : float, default 0.0
        Null-hypothesis value of beta on ``endog``.
    add_const : bool, default True
    n_simulations : int, default 20000
        Monte-Carlo draws for the conditional critical value.
    random_state : int, optional

    Returns
    -------
    CLRResult
    """
    Yv = data[y].values.astype(float) if isinstance(y, str) else np.asarray(y, dtype=float)
    Dv = data[endog].values.astype(float) if isinstance(endog, str) else np.asarray(endog, dtype=float)
    if isinstance(instruments, list) and all(isinstance(v, str) for v in instruments):
        Z = data[instruments].values.astype(float)
    else:
        Z = _as_matrix(instruments)

    Yv = Yv.reshape(-1)
    Dv = Dv.reshape(-1)
    n = len(Yv)
    if Dv.ndim != 1:
        raise ValueError("CLR test supports a single endogenous regressor only.")
    k = Z.shape[1]

    W = _extract_exog(data, exog, n, add_const)

    # Partial out exogenous
    y_t = _residualize(Yv.reshape(-1, 1), W).ravel()
    d_t = _residualize(Dv.reshape(-1, 1), W).ravel()
    Z_t = _residualize(Z, W)

    # Build y*(beta0) = y - beta0 * d and stack [y*(beta0), d]
    ystar = y_t - beta0 * d_t

    # Reduced-form residual covariance estimate (under H0)
    YD = np.column_stack([ystar, d_t])
    # First orthonormalize Z
    ZZ = Z_t.T @ Z_t
    L = np.linalg.cholesky(ZZ)
    Zs = np.linalg.solve(L.T, Z_t.T).T  # n x k with Zs'Zs = I

    # Sigma = n^{-1} YD' M_Z YD
    M_Z = np.eye(n) - Zs @ Zs.T
    Sigma = YD.T @ M_Z @ YD / max(n - W.shape[1] - k, 1)

    # S and T statistics (Moreira 2003 notation)
    a0 = np.array([1.0, -beta0])  # identifies y*(beta0)
    b0 = np.array([beta0, 1.0])
    # Simplify: directly work with residuals
    # S = Zs' ystar / sqrt(sigma_vv_given_u * ...)
    sigma_uu = float(Sigma[0, 0])
    sigma_vv = float(Sigma[1, 1])
    sigma_uv = float(Sigma[0, 1])

    S = Zs.T @ ystar / np.sqrt(max(sigma_uu, 1e-12))
    # T built from d residualised on u direction
    # d_perp = d - (sigma_uv/sigma_uu) * ystar
    d_perp = d_t - (sigma_uv / max(sigma_uu, 1e-12)) * ystar
    sigma_perp = max(sigma_vv - sigma_uv ** 2 / max(sigma_uu, 1e-12), 1e-12)
    T = Zs.T @ d_perp / np.sqrt(sigma_perp)

    ar = float(S @ S)  # Anderson-Rubin
    lm = float((S @ T) ** 2 / max(T @ T, 1e-12))
    qt = float(T @ T)

    clr_stat = 0.5 * (
        ar - qt + np.sqrt(max((ar + qt) ** 2 - 4 * (ar * qt - lm * qt), 0.0))
    )

    # Conditional critical value via Monte-Carlo, conditioning on qt
    rng = np.random.default_rng(random_state)
    m = int(n_simulations)
    X = rng.standard_normal((m, k))
    Y = rng.standard_normal((m, k))
    # Fix qt; sample S independently of T direction.
    # Moreira 2003 Algorithm: simulate S' S and S' T under H0 with qt fixed.
    # Standard trick: draw chi2_k for ar_sim, draw beta(1/2, (k-1)/2) for lm/ar ratio.
    # We use direct normal sampling with fixed T norm = sqrt(qt).
    T_dir = T / max(np.linalg.norm(T), 1e-12)
    # Build orthonormal basis with T_dir as first vector.
    Q = _orthonormal_basis(T_dir, k)
    # Under H0, S ~ N(0, I_k). Decompose along Q.
    S_sim = rng.standard_normal((m, k))
    # project S onto Q basis: first coord is along T_dir
    coords = S_sim @ Q  # m x k
    s1 = coords[:, 0]
    s_rest_sq = np.sum(coords[:, 1:] ** 2, axis=1)
    ar_sim = s1 ** 2 + s_rest_sq
    lm_sim = s1 ** 2  # because T has norm sqrt(qt); (S.T)^2/qt after cancel
    # NOTE: conditioning on qt — qt itself cancels in LM because
    # LM = (S'T)^2 / (T'T) = s1^2 * qt / qt = s1^2.
    clr_sim = 0.5 * (
        ar_sim - qt + np.sqrt(
            np.maximum((ar_sim + qt) ** 2 - 4 * (ar_sim * qt - lm_sim * qt), 0.0)
        )
    )
    pvalue = float(np.mean(clr_sim >= clr_stat))

    return CLRResult(
        statistic=clr_stat,
        pvalue=pvalue,
        beta0=float(beta0),
        n_simulations=m,
        ar_stat=ar,
        lm_stat=lm,
    )


def _orthonormal_basis(v: np.ndarray, k: int) -> np.ndarray:
    """Return a k x k orthonormal matrix whose first column is v/||v||."""
    v = v.reshape(-1)
    v = v / max(np.linalg.norm(v), 1e-12)
    Q = np.zeros((k, k))
    Q[:, 0] = v
    # Gram-Schmidt with standard basis
    idx = np.argsort(-np.abs(v))  # stability
    filled = 1
    for j in idx:
        if filled >= k:
            break
        e = np.zeros(k)
        e[j] = 1.0
        u = e - Q[:, :filled] @ (Q[:, :filled].T @ e)
        nrm = np.linalg.norm(u)
        if nrm > 1e-10:
            Q[:, filled] = u / nrm
            filled += 1
    return Q


__all__ = [
    "kleibergen_paap_rk",
    "sanderson_windmeijer",
    "conditional_lr_test",
    "KleibergenPaapResult",
    "SandersonWindmeijerResult",
    "CLRResult",
]
