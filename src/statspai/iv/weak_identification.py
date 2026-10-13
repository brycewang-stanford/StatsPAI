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
    97-126. [@kleibergen2006generalized]

Sanderson, E. and Windmeijer, F. (2016). "A weak instrument F-test in
    linear IV models with multiple endogenous variables." *Journal of
    Econometrics*, 190(2), 212-221. [@sanderson2016weak]

Moreira, M.J. (2003). "A conditional likelihood ratio test for structural
    models." *Econometrica*, 71(4), 1027-1048. [@moreira2003conditional]

Cragg, J.G. and Donald, S.G. (1993). "Testing identifiability and
    specification in instrumental variable models." *Econometric Theory*,
    9(2), 222-240. [@cragg1993testing]
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd
from scipy import stats

from .._result_serialize import ResultProtocolMixin
from ..exceptions import DataInsufficient, IdentificationFailure, MethodIncompatibility

# ═══════════════════════════════════════════════════════════════════════
#  Data containers
# ═══════════════════════════════════════════════════════════════════════


@dataclass
class KleibergenPaapResult(ResultProtocolMixin):
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
            f"  rk LM statistic      : {self.rk_lm:>10.4f}   "
            f"p={self.rk_lm_pvalue:.4f}\n"
            f"  rk Wald statistic    : {self.rk_wald:>10.4f}   "
            f"p={self.rk_wald_pvalue:.4f}\n"
            f"  rk Wald F-statistic  : {self.rk_f:>10.4f}\n"
            f"  n_endog = {self.n_endog},  n_instruments = {self.n_instruments}\n"
            f"  covariance type      : {self.cov_type}"
        )


@dataclass
class SandersonWindmeijerResult(ResultProtocolMixin):
    """Sanderson-Windmeijer conditional F for each endogenous variable."""

    endog_names: List[str]
    sw_f: Dict[str, float]
    sw_pvalue: Dict[str, float]
    df_num: Dict[str, int]
    df_denom: int
    partial_r2: Dict[str, float]

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "SW F": [self.sw_f[n] for n in self.endog_names],
                "p-value": [self.sw_pvalue[n] for n in self.endog_names],
                "df_num": [self.df_num[n] for n in self.endog_names],
                "partial R²": [self.partial_r2[n] for n in self.endog_names],
            },
            index=self.endog_names,
        )

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
class CLRResult(ResultProtocolMixin):
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
            + (
                f"  simulations          : {self.n_simulations}"
                if self.n_simulations
                else "  p-value              : exact (conditional on T'T)"
            )
        )


# ═══════════════════════════════════════════════════════════════════════
#  Internal helpers
# ═══════════════════════════════════════════════════════════════════════


def _residualize(M: np.ndarray, W: Optional[np.ndarray]) -> np.ndarray:
    """Project M onto the orthogonal complement of W."""
    if W is None or W.size == 0 or W.shape[1] == 0:
        return M
    beta, *_ = np.linalg.lstsq(W, M, rcond=None)
    return np.asarray(M - W @ beta, dtype=float)


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
    if (
        isinstance(exog, (list, tuple))
        and data is not None
        and all(isinstance(v, str) for v in exog)
    ):
        W = data[list(exog)].values.astype(float)
    else:
        W = _as_matrix(exog)
    if add_const:
        W = np.column_stack([np.ones(W.shape[0]), W])
    return np.asarray(W, dtype=float)


# ═══════════════════════════════════════════════════════════════════════
#  Kleibergen-Paap rk statistic
# ═══════════════════════════════════════════════════════════════════════


def _reduced_form_cov(
    Z_tilde: np.ndarray,
    V: np.ndarray,
    ZZ_inv: np.ndarray,
    cov_type: str,
    cluster: Optional[Union[np.ndarray, pd.Series]],
    *,
    n: int,
    k: int,
    p: int,
    denom: float,
    ssc: float,
    cluster_ssc: bool,
) -> np.ndarray:
    """Covariance of ``vec(Pi)`` for the multivariate reduced form.

    ``V`` is whatever residual the caller wants the score variance built
    from: the unrestricted reduced-form residual for the Wald statistic,
    or the endogenous regressor itself for the ``Pi = 0`` LM statistic.
    ``ssc`` is the finite-sample factor applied to the meat (1.0 for LM).
    """
    if cov_type == "nonrobust":
        Sigma = (V.T @ V) / denom
        return np.asarray(np.kron(Sigma, ZZ_inv), dtype=float)

    # One ordering throughout: vec() stacks the columns of the (k, p)
    # matrix Pi, so entry (instrument i, endogenous j) sits at j*k + i in
    # vec(Pi), in the scores and in the bread kron(I_p, (Z'Z)^-1). Before the
    # 2026-10 fix the robust meat and the bread were instrument-major against an
    # endogenous-major vec(Pi) and cluster meat -- the same thing when
    # p = 1, a scrambled sandwich when p >= 2.
    if cov_type == "robust":
        # Meat: sum_i kron(v_i v_i', z_i z_i') = U'U with U_i = kron(v_i,
        # z_i). Vectorised: the per-row Python loop took ~1.6 s at
        # n = 100,000.
        U = (Z_tilde[:, :, None] * V[:, None, :]).transpose(0, 2, 1).reshape(n, k * p)
        meat = U.T @ U
        meat *= ssc
    else:  # cluster
        from itertools import combinations

        frame = pd.DataFrame(np.asarray(cluster))
        if frame.shape[0] != n and frame.shape[1] == n:  # pragma: no cover
            frame = frame.T
        d = frame.shape[1]
        # vec() stacks columns of the (k, p) score block, so transpose to
        # (p, k) before the row-major flatten.
        outer = (
            (Z_tilde[:, :, None] * V[:, None, :]).transpose(0, 2, 1).reshape(n, k * p)
        )
        meat = np.zeros((k * p, k * p))
        g_min = None
        for size in range(1, d + 1):
            sign = 1.0 if size % 2 == 1 else -1.0
            for cols in combinations(range(d), size):
                sub = frame.iloc[:, list(cols)]
                keys = (
                    sub.iloc[:, 0]
                    if size == 1
                    else pd.MultiIndex.from_frame(sub.astype(object))
                )
                codes, uniques = pd.factorize(keys, sort=False)
                G = len(uniques)
                if size == 1:
                    g_min = G if g_min is None else min(g_min, G)
                sums = np.zeros((G, k * p))
                np.add.at(sums, codes, outer)
                meat += sign * (sums.T @ sums)
        if cluster_ssc and g_min is not None:
            meat *= g_min / max(g_min - 1, 1)
        meat *= ssc

    bread = np.kron(np.eye(p), ZZ_inv)
    return np.asarray(bread @ meat @ bread, dtype=float)


def _psd_sqrt(M: np.ndarray) -> np.ndarray:
    """Symmetric square root of a positive semi-definite matrix."""
    w, Q = np.linalg.eigh((M + M.T) / 2.0)
    return np.asarray((Q * np.sqrt(np.clip(w, 0.0, None))) @ Q.T, dtype=float)


def _rk_statistic(
    Pi: np.ndarray, cov_vec: np.ndarray, ZtZ: np.ndarray, DtD: np.ndarray, n: int
) -> float:
    """Kleibergen-Paap rk statistic of H0: rank(Pi) = p - 1, for p >= 2.

    The construction of Kleibergen and Paap (2006, [@kleibergen2006generalized])
    as ``ranktest`` / ``ivreg2`` apply it to the first stage:

    * ``Theta = F Pi G`` with ``F (Z'Z/n)^-1 F' = I`` and ``G' (D'D/n) G =
      I``, so its singular values are the canonical correlations of the
      (partialled-out) endogenous regressors and instruments, and
      ``Var(vec Theta) = (G' kron F) Var(vec Pi) (G kron F')``;
    * singular value decomposition ``Theta = U S V'``, partitioned after
      the first ``q = p - 1`` rows / columns; ``A_perp = [U12; U22]
      U22^-1 (U22 U22')^(1/2)`` and ``B_perp = (V22 V22')^(1/2) V22'^-1
      [V12' V22']``;
    * ``lambda = A_perp' Theta B_perp'`` and ``rk = lambda' [(B_perp kron
      A_perp') Var(vec Theta) (B_perp kron A_perp')']^-1 lambda``, which
      is chi2((k - q)(p - q)) = chi2(k - p + 1) under H0.

    With ``Var(vec Pi) = Sigma_V kron (Z'Z)^-1`` this is ``n r^2 / (1 -
    r^2)`` for the smallest canonical correlation ``r`` (the Cragg-Donald
    statistic), and with ``Sigma_D`` in place of ``Sigma_V`` it is
    Anderson's ``n r^2``.
    """
    k, p = Pi.shape
    q = p - 1
    F = np.linalg.cholesky(ZtZ / n).T
    G = np.linalg.inv(np.linalg.cholesky(DtD / n)).T
    Theta = F @ Pi @ G
    T = np.kron(G.T, F)
    W_theta = T @ cov_vec @ T.T
    U, _, Vt = np.linalg.svd(Theta)
    Vm = Vt.T
    U22 = U[q:, q:]
    V22 = Vm[q:, q:]
    A_perp = U[:, q:] @ np.linalg.solve(U22, _psd_sqrt(U22 @ U22.T))
    B_perp = _psd_sqrt(V22 @ V22.T) @ np.linalg.solve(V22.T, Vm[:, q:].T)
    lam = (A_perp.T @ Theta @ B_perp.T).flatten(order="F")
    K = np.kron(B_perp, A_perp.T)
    return float(lam @ np.linalg.solve(K @ W_theta @ K.T, lam))


def _validated_cluster(cluster: object, n: int) -> pd.DataFrame:
    """Cluster ids as an (n, d) frame; refuses what would otherwise be
    answered with a number: a missing id (``pd.factorize`` codes it -1 and
    ``np.add.at`` then adds its scores to the last cluster) and a single
    cluster (the score sums to zero by the normal equations, so the
    covariance is rounding error)."""
    frame = pd.DataFrame(np.asarray(cluster))
    if frame.shape[0] != n and frame.shape[1] == n:  # pragma: no cover
        frame = frame.T
    if frame.shape[0] != n:
        raise MethodIncompatibility(
            f"cluster has {frame.shape[0]} rows; the data have {n}.",
            recovery_hint="Pass one cluster id per observation.",
        )
    n_missing = int(frame.isna().any(axis=1).sum())
    if n_missing:
        raise MethodIncompatibility(
            f"cluster has {n_missing} missing value(s).",
            recovery_hint="Drop the rows with a missing cluster id from "
            "every input first.",
            diagnostics={"n_missing": n_missing},
        )
    counts = [int(frame.iloc[:, j].nunique()) for j in range(frame.shape[1])]
    if min(counts) < 2:
        raise DataInsufficient(
            "cluster-robust rank statistics need at least two clusters; "
            f"found {min(counts)}.",
            recovery_hint="Cluster on a variable with more than one value, "
            "or use cov_type='robust'.",
            diagnostics={"n_clusters": counts},
        )
    return frame


def kleibergen_paap_rk(
    endog: Union[np.ndarray, pd.DataFrame],
    instruments: Union[np.ndarray, pd.DataFrame],
    exog: Optional[Union[np.ndarray, pd.DataFrame, List[str]]] = None,
    data: Optional[pd.DataFrame] = None,
    cov_type: str = "robust",
    cluster: Optional[Union[np.ndarray, pd.Series]] = None,
    add_const: bool = True,
    n_absorbed: int = 0,
    small: bool = True,
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
        Required when ``cov_type='cluster'``, and refused with any other
        ``cov_type`` (it used to be ignored). Missing ids and a single
        cluster are refused.
    add_const : bool, default True
        Prepend a constant to the exogenous block.
    n_absorbed : int, default 0
        Degrees of freedom consumed by absorbed fixed effects. Enters the
        finite-sample factor exactly as ``ivreg2``'s ``e(sdofminus)`` does:
        ``K = n_exog + n_instruments + n_absorbed``. Leave at 0 unless the
        reduced form was estimated on FE-residualised data.
    small : bool, default True
        Apply ``ivreg2``'s finite-sample factor to the rk **Wald**
        covariance: ``(n-K)`` in the denominator for the classical
        variance, ``n/(n-K)`` for HC-robust, and
        ``G/(G-1) * (n-1)/(n-K)`` for cluster-robust. The rk **LM**
        statistic never takes one, matching ``ranktest``.

    Returns
    -------
    KleibergenPaapResult

    Notes
    -----
    With a single endogenous regressor both statistics reproduce
    ``ivreg2``/``ranktest`` exactly: ``rk_wald`` is the Wald form built
    from the unrestricted reduced-form residuals, and ``rk_lm`` is the
    same quadratic form evaluated under the null ``Pi = 0`` (so the score
    variance uses the *unexplained* endogenous regressor) with no
    finite-sample factor.

    With two or more endogenous regressors the null is a rank restriction,
    not ``Pi = 0``, and both statistics are the Kleibergen-Paap rk form
    built from the singular value decomposition of the normalised
    reduced-form matrix (see ``_rk_statistic``), with the same two
    covariances as above. Under ``cov_type='nonrobust'`` ``rk_wald`` is the
    Cragg-Donald statistic and ``rk_lm`` Anderson's canonical-correlation
    statistic. ``rk_f`` is ``rk_wald`` divided by the number of excluded
    instruments (``df_num``), ``ivreg2``'s "Kleibergen-Paap rk Wald F";
    both p-values refer to chi2(k - p + 1). Pinned to Stata 18 ``ivreg2``
    for two and three endogenous regressors under all three covariance
    types. (Before the 2026-10 fix the p >= 2 ``rk_wald`` was the Wald statistic of
    ``Pi = 0`` and ``rk_lm`` ignored ``cov_type``.)

    Collinear excluded instruments (after partialling out ``exog``) are
    refused.
    """
    D = _as_matrix(endog)  # n x p
    Z = _as_matrix(instruments)  # n x k
    n, p = D.shape
    k = Z.shape[1]
    if k < p:
        raise IdentificationFailure(
            f"Under-identified: only {k} instruments for {p} endogenous regressors."
        )

    W = _extract_exog(data, exog, n, add_const)
    n_W = W.shape[1]

    # Partial out exogenous regressors
    D_tilde = _residualize(D, W)
    Z_tilde = _residualize(Z, W)

    from .weak_iv_ci import _check_instrument_rank

    _check_instrument_rank(
        Z_tilde,
        list(instruments.columns) if isinstance(instruments, pd.DataFrame) else None,
        what="the Kleibergen-Paap rank statistics",
    )

    # Reduced form coefficients: D_tilde = Z_tilde @ Pi + V
    ZtZ = Z_tilde.T @ Z_tilde
    try:
        ZZ_inv = np.linalg.inv(ZtZ)
    except np.linalg.LinAlgError:  # pragma: no cover
        ZZ_inv = np.linalg.pinv(ZtZ)
    Pi = ZZ_inv @ (Z_tilde.T @ D_tilde)  # k x p
    V = D_tilde - Z_tilde @ Pi

    # Covariance of vec(Pi) — GLS form
    # Var(vec(Pi_hat)) = (Sigma_VV ⊗ (Z'Z)^{-1}) with appropriate robust mod
    # ivreg2's regressor count for the reduced form: included exogenous,
    # excluded instruments, and any absorbed fixed-effect DOF.
    K_rf = n_W + k + int(n_absorbed)
    denom_rf = max(n - K_rf, 1) if small else n
    if cov_type == "cluster" and cluster is None:
        raise ValueError("cov_type='cluster' requires `cluster`.")
    if cov_type not in ("nonrobust", "robust", "cluster"):
        raise ValueError(f"Unknown cov_type: {cov_type}")
    if cluster is not None and cov_type != "cluster":
        raise MethodIncompatibility(
            f"cluster= was passed with cov_type={cov_type!r}, which does "
            "not use it.",
            recovery_hint="Pass cov_type='cluster' for cluster-robust rank "
            "statistics, or drop cluster=.",
        )
    if cov_type == "cluster":
        _validated_cluster(cluster, n)

    if cov_type == "robust":
        ssc = (n / max(n - K_rf, 1)) if small else 1.0
    elif cov_type == "cluster":
        ssc = ((n - 1) / max(n - K_rf, 1)) if small else 1.0
    else:
        ssc = 1.0

    cov_vec = _reduced_form_cov(
        Z_tilde,
        V,
        ZZ_inv,
        cov_type,
        cluster,
        n=n,
        k=k,
        p=p,
        denom=denom_rf,
        ssc=ssc,
        cluster_ssc=True,
    )
    if cov_type == "cluster":
        _cf = pd.DataFrame(np.asarray(cluster))
        _counts = [int(_cf.iloc[:, j].nunique()) for j in range(_cf.shape[1])]
        cov_label = "cluster (" + " x ".join(f"{g} groups" for g in _counts) + ")"
    elif cov_type == "robust":
        cov_label = "HC robust"
    else:
        cov_label = "nonrobust"

    vec_Pi = Pi.flatten(order="F")
    DtD = D_tilde.T @ D_tilde
    if p == 1:
        # KP rk Wald: vec(Pi)' cov_vec^{-1} vec(Pi) -- with one endogenous
        # regressor rank p - 1 = 0 is Pi = 0
        cov_pinv = np.linalg.pinv(cov_vec)
        rk_wald = float(vec_Pi @ cov_pinv @ vec_Pi)
    else:
        rk_wald = _rk_statistic(Pi, cov_vec, ZtZ, DtD, n)

    # F version: divide by the number of excluded instruments, as ivreg2's
    # "rk Wald F" does (the finite-sample factor is already in cov_vec)
    df_num = k
    df_denom = max(n - K_rf, 1)
    rk_f = rk_wald / df_num
    rk_wald_pvalue = float(stats.chi2.sf(rk_wald, df=k - p + 1))

    # KP rk LM statistic. The LM principle evaluates the score variance at
    # the restricted estimate; ranktest takes the (partialled-out)
    # endogenous regressors themselves as the "residual" there rather than
    # V, and applies no finite-sample factor -- neither G/(G-1) nor
    # (n-1)/(n-K).
    cov_null = _reduced_form_cov(
        Z_tilde,
        D_tilde,
        ZZ_inv,
        cov_type,
        cluster,
        n=n,
        k=k,
        p=p,
        denom=n,
        ssc=1.0,
        cluster_ssc=False,
    )
    if p == 1:
        # H0 is Pi = 0, so the rank test is the score test of that
        # restriction.
        rk_lm = float(vec_Pi @ np.linalg.pinv(cov_null) @ vec_Pi)
    else:
        rk_lm = _rk_statistic(Pi, cov_null, ZtZ, DtD, n)
    rk_lm_pvalue = float(stats.chi2.sf(rk_lm, df=(k - p + 1)))

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
    return np.asarray(V @ np.diag(np.sqrt(w)) @ V.T, dtype=float)


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

    For each endogenous regressor ``j``, nets the *other* endogenous
    regressors out with their 2SLS coefficients -- ``d_j`` is regressed on
    the others by 2SLS with the excluded instruments -- and tests the
    instruments in the first stage of the residual ``v_j = d_j -
    D_other @ delta_2sls``: ``F = [v_j' P_Z v_j / (k - p + 1)] / [v_j' M_Z
    v_j / (n - n_exog - k)]``, everything after partialling out the
    exogenous regressors. This is the individual-endogenous weak-IV
    diagnostic when several endogenous regressors are present, and the
    statistic Stata ``ivreg2, first`` reports as "SW F" (classical
    version; pinned to it for two and three endogenous regressors).
    Before the 2026-10 fix the other regressors were partialled out by OLS, of both
    ``d_j`` and the instruments, which is a different statistic.

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
        raise IdentificationFailure(
            f"Under-identified: only {k} instruments for {p} endogenous regressors."
        )

    W = _extract_exog(data, exog, n, add_const)

    names = endog_names or _collect_names(endog, prefix="endog")
    if len(names) != p:
        raise ValueError(
            f"endog_names length {len(names)} != n_endog {p}"
        )  # pragma: no cover

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
            # Sanderson-Windmeijer ([@sanderson2016weak]): delta is the
            # 2SLS coefficient of D_j on D_other with instruments Z,
            #   delta = (D_o' P_Z D_o)^-1 D_o' P_Z D_j,
            # and the F below tests Z in the regression of the residual
            # D_j - D_o delta on Z, with k - (p - 1) numerator degrees of
            # freedom (p - 1 directions of Z are used up by delta).
            fitted_other = Z_tilde @ np.linalg.lstsq(Z_tilde, D_other, rcond=None)[0]
            delta, *_ = np.linalg.lstsq(fitted_other, D_j, rcond=None)
            Zc = Z_tilde
            y_j = np.asarray(D_j - D_other @ delta, dtype=float)
        else:
            Zc = Z_tilde
            y_j = np.asarray(D_j, dtype=float)

        # First-stage regression of y_j on Zc
        beta, *_ = np.linalg.lstsq(Zc, y_j, rcond=None)
        resid = y_j - Zc @ beta
        rss = float(resid @ resid)
        tss = float(y_j @ y_j)

        df1 = k - (p - 1)  # SW adjusted numerator df
        # residual degrees of freedom of the first stage of v_j: delta is
        # not a regressor in it (ivreg2 reports the same n - n_exog - k)
        df2 = n - n_W - k
        if df1 <= 0:
            raise ValueError(
                f"Not enough instruments: k - (p-1) = {df1} for endogenous "
                f"'{names[j]}'."
            )
        if df2 <= 0:
            raise ValueError(
                f"Not enough observations: df_denom = {df2}."
            )  # pragma: no cover

        if rss > 0 and tss > 0:
            explained = tss - rss
            f_j = (explained / df1) / (rss / df2)
            pval = float(stats.f.sf(f_j, df1, df2))
            pr2 = 1 - rss / tss
        else:
            f_j = np.nan  # pragma: no cover
            pval = np.nan  # pragma: no cover
            pr2 = np.nan  # pragma: no cover

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
    method: str = "exact",
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
        Monte-Carlo draws, used only with ``method='simulate'``.
    random_state : int, optional
        Seed, used only with ``method='simulate'``.
    method : {'exact', 'simulate'}, default 'exact'
        ``'exact'`` integrates the conditional null distribution of CLR
        given ``T'T`` numerically -- the closed form R ``ivmodel::CLR`` and
        Stata ``weakiv`` evaluate -- so the p-value is deterministic.
        ``'simulate'`` is the pre-1.29 Monte-Carlo p-value.

    Returns
    -------
    CLRResult
    """
    if isinstance(y, str):
        if data is None:
            raise ValueError("`data` is required when `y` is a column name.")
        Yv = data[y].values.astype(float)
    else:
        Yv = np.asarray(y, dtype=float)
    if isinstance(endog, str):
        if data is None:
            raise ValueError("`data` is required when `endog` is a column name.")
        Dv = data[endog].values.astype(float)
    else:
        Dv = np.asarray(endog, dtype=float)
    if isinstance(instruments, list) and all(isinstance(v, str) for v in instruments):
        if data is None:
            raise ValueError("`data` is required when `instruments` are column names.")
        Z = data[instruments].values.astype(float)
    else:
        Z = _as_matrix(instruments)

    Yv = Yv.reshape(-1)
    Dv = Dv.reshape(-1)
    n = len(Yv)
    if Dv.ndim != 1:
        raise ValueError(
            "CLR test supports a single endogenous regressor only."
        )  # pragma: no cover
    k = Z.shape[1]

    W = _extract_exog(data, exog, n, add_const)

    from .weak_iv_ci import _refuse_missing

    _refuse_missing("conditional_lr_test", y=Yv, endog=Dv, instruments=Z, exog=W)
    if not np.isfinite(beta0):
        raise MethodIncompatibility(
            f"beta0 must be a finite number; got {beta0!r}.",
            recovery_hint="Pass the hypothesised coefficient, e.g. beta0=0.0.",
        )

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
    # Zs = Zt L'^-1 so that Zs'Zs = L^-1 (L L') L'^-1 = I. Through 1.28.0
    # this solved against L' instead of L, giving Zs = Zt L^-1 -- not
    # orthonormal once k >= 2 -- which mis-scaled every S / T statistic.
    Zs = np.linalg.solve(L, Z_t.T).T

    # Sigma = YD' M_Z YD / (n - p - k), without forming the n x n M_Z
    ZsYD = Zs.T @ YD
    Sigma = (YD.T @ YD - ZsYD.T @ ZsYD) / max(n - W.shape[1] - k, 1)

    # S and T statistics (Moreira 2003 notation), working directly with
    # residuals in the y*(beta0) and d directions.
    # S = Zs' ystar / sqrt(sigma_vv_given_u * ...)
    sigma_uu = float(Sigma[0, 0])
    sigma_vv = float(Sigma[1, 1])
    sigma_uv = float(Sigma[0, 1])

    S = Zs.T @ ystar / np.sqrt(max(sigma_uu, 1e-12))
    # T built from d residualised on u direction
    # d_perp = d - (sigma_uv/sigma_uu) * ystar
    d_perp = d_t - (sigma_uv / max(sigma_uu, 1e-12)) * ystar
    sigma_perp = max(sigma_vv - sigma_uv**2 / max(sigma_uu, 1e-12), 1e-12)
    T = Zs.T @ d_perp / np.sqrt(sigma_perp)

    ar = float(S @ S)  # Anderson-Rubin
    lm = float((S @ T) ** 2 / max(T @ T, 1e-12))
    qt = float(T @ T)

    clr_stat = 0.5 * (
        ar - qt + np.sqrt(max((ar + qt) ** 2 - 4 * (ar * qt - lm * qt), 0.0))
    )

    if method not in ("exact", "simulate"):
        raise MethodIncompatibility(
            f"method must be 'exact' or 'simulate'; got {method!r}"
        )
    if method == "exact":
        from .weak_iv_ci import _clr_conditional_pvalue

        return CLRResult(
            statistic=float(clr_stat),
            pvalue=_clr_conditional_pvalue(float(clr_stat), qt, k),
            beta0=float(beta0),
            n_simulations=0,
            ar_stat=ar,
            lm_stat=lm,
        )

    # Conditional critical value via Monte-Carlo, conditioning on qt
    rng = np.random.default_rng(random_state)
    m = int(n_simulations)
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
    ar_sim = s1**2 + s_rest_sq
    lm_sim = s1**2  # because T has norm sqrt(qt); (S.T)^2/qt after cancel
    # NOTE: conditioning on qt — qt itself cancels in LM because
    # LM = (S'T)^2 / (T'T) = s1^2 * qt / qt = s1^2.
    clr_sim = 0.5 * (
        ar_sim
        - qt
        + np.sqrt(np.maximum((ar_sim + qt) ** 2 - 4 * (ar_sim * qt - lm_sim * qt), 0.0))
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
