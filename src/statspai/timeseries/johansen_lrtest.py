"""Likelihood-ratio tests of restrictions in a cointegrated VAR.

Once the cointegration rank is chosen (:func:`statspai.johansen`), economic
hypotheses are restrictions on the cointegrating vectors ``beta`` or on the
loadings ``alpha`` of ``Delta y_t = alpha beta' y_{t-1} + ...``. With the
rank held fixed the likelihood-ratio statistics are asymptotically
chi-squared, unlike the rank tests themselves.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, List, Optional, Sequence

import numpy as np
import pandas as pd
from scipy import stats

from .._result_serialize import ResultProtocolMixin
from ..exceptions import MethodIncompatibility
from .cointegration import _JOHANSEN_TREND_ALIASES, _johansen_residuals

__all__ = ["johansen_lrtest", "JohansenLRTest"]


@dataclass
class JohansenLRTest(ResultProtocolMixin):
    """Result of :func:`johansen_lrtest`.

    Attributes
    ----------
    statistic, df, pvalue : float, int, float
        Likelihood-ratio statistic and its chi-squared reference.
    hypothesis : str
        ``'beta'``, ``'beta_known'`` or ``'loading'``.
    rank : int
        Cointegration rank held fixed under both hypotheses.
    eigenvalues, eigenvalues_restricted : ndarray
        Squared canonical correlations without and with the restriction.
    beta, loading : DataFrame
        Restricted estimates (columns ``_ce1`` ...); ``beta`` is normalised
        so the first non-zero element of each column is one when that is
        possible.
    n_used : int
        Observations entering the statistic.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> trend = np.cumsum(rng.normal(size=300))
    >>> df = pd.DataFrame({
    ...     "c": trend + rng.normal(size=300),
    ...     "y": trend + rng.normal(size=300),
    ... })
    >>> test = sp.johansen_lrtest(df, rank=1, beta_known=[1, -1])
    >>> type(test).__name__
    'JohansenLRTest'
    >>> list(test.beta.index)
    ['c', 'y']
    """

    statistic: float
    df: int
    pvalue: float
    hypothesis: str
    rank: int
    eigenvalues: np.ndarray
    eigenvalues_restricted: np.ndarray
    beta: pd.DataFrame
    loading: pd.DataFrame
    n_used: int
    lags: int
    trend: str

    def summary(self) -> str:
        what = {
            "beta": "beta = H phi (the same restrictions on every vector)",
            "beta_known": "the given vectors lie in the cointegration space",
            "loading": "alpha = A psi (restrictions on the loadings)",
        }[self.hypothesis]
        lines = [
            "Likelihood-ratio test in the cointegrated VAR",
            "=" * 60,
            f"H0: {what}",
            f"Rank: {self.rank}   Lags: {self.lags}   Trend: {self.trend}"
            f"   Used: {self.n_used}",
            "",
            f"LR statistic : {self.statistic:.4f}",
            f"chi2({self.df}) p    : {self.pvalue:.4f}",
            "",
            "Restricted cointegrating vectors (beta)",
            self.beta.to_string(float_format=lambda v: f"{v: .6f}"),
            "=" * 60,
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"JohansenLRTest({self.hypothesis}: LR={self.statistic:.4f}, "
            f"df={self.df}, pvalue={self.pvalue:.4g})"
        )


def _gen_eig(A: np.ndarray, B: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Solve ``|lambda B - A| = 0`` for symmetric ``A`` and positive
    definite ``B``; eigenvalues in decreasing order."""
    C = np.linalg.cholesky(B)
    Cinv = np.linalg.inv(C)
    M = Cinv @ A @ Cinv.T
    M = (M + M.T) / 2.0
    w, v = np.linalg.eigh(M)
    order = np.argsort(-w)
    return w[order], Cinv.T @ v[:, order]


def _matrix(value: Any, rows: int, name: str, row_names: List[str]) -> np.ndarray:
    """A restriction matrix with ``rows`` rows, from an array or from a
    DataFrame indexed by variable name."""
    if isinstance(value, pd.DataFrame):
        missing = [r for r in row_names if r not in value.index]
        if missing:
            raise MethodIncompatibility(
                f"johansen_lrtest: {name} has no row for {missing}.",
                recovery_hint=f"Index the rows by {row_names}.",
            )
        value = value.loc[row_names].to_numpy()
    M = np.asarray(value, dtype=float)
    if M.ndim == 1:
        M = M.reshape(-1, 1)
    if M.ndim != 2 or M.shape[0] != rows:
        unit = "equation" if name == "loading" else "element of the vector"
        raise MethodIncompatibility(
            f"johansen_lrtest: {name} must have {rows} rows (one per {unit}), "
            f"got shape {M.shape}.",
            recovery_hint="Each column is one direction that stays free.",
        )
    if np.linalg.matrix_rank(M) < M.shape[1]:
        raise MethodIncompatibility(
            f"johansen_lrtest: the columns of {name} are linearly dependent."
        )
    return M


def _normalise(beta: np.ndarray) -> np.ndarray:
    out = beta.copy()
    for j in range(out.shape[1]):
        nz = np.flatnonzero(np.abs(out[:, j]) > 1e-10 * np.abs(out[:, j]).max())
        if nz.size:
            out[:, j] = out[:, j] / out[nz[0], j]
    return out


def johansen_lrtest(
    data: pd.DataFrame,
    variables: Optional[Sequence[str]] = None,
    *,
    rank: int,
    lags: int = 1,
    trend: str = "c",
    beta: Any = None,
    beta_known: Any = None,
    loading: Any = None,
) -> JohansenLRTest:
    """Test linear restrictions on the cointegrating vectors or the loadings.

    Exactly one of ``beta``, ``beta_known`` and ``loading`` is given.

    ====================  =============================  ==================
    Argument              Hypothesis                     ``urca``
    ====================  =============================  ==================
    ``beta=H``            ``beta = H phi``               ``blrtest``
    ``beta_known=b``      ``beta = (b, psi)``            ``bh5lrtest``
    ``loading=A``         ``alpha = A psi``              ``alrtest``
    ====================  =============================  ==================

    Parameters
    ----------
    data : DataFrame
    variables : sequence of str, optional
        The system, in order; all numeric columns when omitted.
    rank : int
        Cointegration rank ``r``, ``1 <= r < K``, held fixed.
    lags : int, default 1
        Lagged differences, as in :func:`statspai.johansen`.
    trend : {'c', 'rc', 'n', 'rt', 'ct'}, default 'c'
        Deterministic terms, as in :func:`statspai.johansen`. In the
        restricted cases (``'rc'``, ``'rt'``) the cointegrating vector has
        ``K + 1`` elements, the last being the constant or the trend.
    beta : array (K x s) or DataFrame indexed by variable, optional
        ``H``: every cointegrating vector is a combination of its columns
        (``r <= s < K``). For example the columns ``(1, -1, 0)`` and
        ``(0, 0, 1)`` say that the first two variables enter only as their
        difference. In a restricted trend case a ``K``-row ``H`` leaves the
        deterministic coefficient free.
    beta_known : array (K x r1), optional
        Vectors asserted to be cointegrating, ``r1 <= r``; the remaining
        ``r - r1`` are estimated freely. ``(1, -1)`` tests that the
        difference of two variables is stationary given the rank. With
        ``r1 = r`` the whole cointegration space is given (``urca`` tests
        that case with ``blrtest``).
    loading : array (K x m), optional
        ``A``: the loadings are combinations of its columns (``r <= m <
        K``). Dropping the unit vector of a variable tests that it is
        weakly exogenous for the long-run parameters.

    Returns
    -------
    JohansenLRTest
        ``statistic``, ``df``, ``pvalue``, the restricted ``beta`` and
        ``loading``, and both sets of eigenvalues.

    Raises
    ------
    MethodIncompatibility
        Not exactly one hypothesis, a matrix of the wrong shape or rank, a
        restriction that leaves fewer free directions than ``rank``.

    Notes
    -----
    All three statistics come from eigenvalue problems in the product
    moments ``S_ij`` of the residuals ``R0`` (differences) and ``R1``
    (lagged levels), as the unrestricted estimator does
    [@johansen1991estimation]:

    * ``beta = H phi``: ``|lambda H'S11H - H'S10 S00^{-1} S01 H| = 0``,
      ``LR = T sum_{i<=r} ln[(1 - lambda*_i) / (1 - lambda_i)]``,
      ``df = r (K - s)``.
    * ``alpha = A psi``: the same problem after conditioning both sets of
      residuals on ``A_perp' R0``, ``df = r (K - m)``.
    * ``beta = (b, psi)``: the known directions first, then the free ones
      conditional on ``b' R1``; ``df = r1 (K - r)``.

    In a restricted trend case (``'rc'``, ``'rt'``) a ``K``-row ``beta``
    leaves the deterministic coefficient free and the degrees of freedom
    are as above. A ``K``-row ``beta_known`` is a completely known vector:
    its deterministic coefficient is zero (the relation has mean zero, or
    no trend), which is one more restriction per vector, ``df = r1 (K + 1
    - r)``. Give ``K + 1`` rows to state another value.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> trend = np.cumsum(rng.normal(size=300))
    >>> df = pd.DataFrame({
    ...     "c": trend + rng.normal(size=300),
    ...     "y": trend + rng.normal(size=300),
    ...     "r": np.cumsum(rng.normal(size=300)),
    ... })
    >>> test = sp.johansen_lrtest(df, rank=1, beta_known=[1, -1, 0])
    >>> test.df
    2
    >>> bool(test.pvalue > 0.01)   # c - y is stationary by construction
    True

    References
    ----------
    [@johansen1991estimation],
    [@johansen1992testing],
    [@neusser1991testing]
    """
    case = _JOHANSEN_TREND_ALIASES.get(str(trend).lower())
    if case is None:
        raise MethodIncompatibility(
            "johansen_lrtest: trend must be one of 'n', 'rc', 'c', 'rt', 'ct'."
        )
    given = [x is not None for x in (beta, beta_known, loading)]
    if sum(given) != 1:
        raise MethodIncompatibility(
            "johansen_lrtest: give exactly one of beta=, beta_known=, loading=.",
            recovery_hint="Test one hypothesis per call.",
        )
    if variables is None:
        variables = data.select_dtypes(include=[np.number]).columns.tolist()
    names = [str(v) for v in variables]
    Y = data[list(variables)].dropna().to_numpy(dtype=float)
    K = Y.shape[1]
    r = int(rank)
    if not 1 <= r < K:
        raise MethodIncompatibility(
            f"johansen_lrtest: rank must be between 1 and {K - 1}, got {rank}.",
            recovery_hint="Choose the rank with sp.johansen first.",
        )
    lags = int(lags)
    if lags < 0:
        raise MethodIncompatibility("johansen_lrtest: lags must be >= 0.")

    R0, R1, T = _johansen_residuals(Y, lags, case)
    k1 = R1.shape[1]  # K, or K + 1 with a restricted deterministic term
    extra = k1 - K
    beta_names = names + (["_cons"] if case == "rc" else ["_trend"] if extra else [])

    def moments(a: np.ndarray, b: np.ndarray) -> np.ndarray:
        return np.asarray(a.T @ b / T)

    S00, S01, S11 = moments(R0, R0), moments(R0, R1), moments(R1, R1)
    lam, _ = _gen_eig(S01.T @ np.linalg.solve(S00, S01), S11)
    lam = np.clip(lam, 0.0, 1.0 - 1e-15)

    def widen(M: np.ndarray) -> np.ndarray:
        """Leave the restricted deterministic coefficient free."""
        if extra and M.shape[0] == K:
            top = np.hstack([M, np.zeros((K, 1))])
            bottom = np.zeros((1, M.shape[1] + 1))
            bottom[0, -1] = 1.0
            return np.vstack([top, bottom])
        return M

    if beta is not None:
        hypothesis = "beta"
        H = _matrix(beta, K if np.shape(beta)[0] == K else k1, "beta", beta_names[:K])
        s_free = H.shape[1]
        if not r <= s_free < K + (H.shape[0] - K):
            raise MethodIncompatibility(
                f"johansen_lrtest: beta has {s_free} columns; it needs at "
                f"least rank = {r} and fewer than {H.shape[0]} to restrict "
                "anything.",
            )
        df = r * (H.shape[0] - s_free)
        H = widen(H)
        lam_r, vec = _gen_eig(
            H.T @ S01.T @ np.linalg.solve(S00, S01) @ H, H.T @ S11 @ H
        )
        lam_r = np.clip(lam_r, 0.0, 1.0 - 1e-15)
        stat = T * float(np.sum(np.log((1 - lam_r[:r]) / (1 - lam[:r]))))
        b_hat = H @ vec[:, :r]
        a_hat = S01 @ b_hat @ np.linalg.inv(b_hat.T @ S11 @ b_hat)
    elif loading is not None:
        hypothesis = "loading"
        A = _matrix(loading, K, "loading", names)
        m = A.shape[1]
        if not r <= m < K:
            raise MethodIncompatibility(
                f"johansen_lrtest: loading has {m} columns; it needs at "
                f"least rank = {r} and fewer than {K}.",
            )
        df = r * (K - m)
        # orthogonal complement of A and the map a -> (A'A)^{-1} A' a
        u, _, _ = np.linalg.svd(A, full_matrices=True)
        B = u[:, m:]
        Abar = A @ np.linalg.inv(A.T @ A)
        Rb = R0 @ B
        proj = Rb @ np.linalg.solve(Rb.T @ Rb, Rb.T)
        Ra = (R0 - proj @ R0) @ Abar
        R1b = R1 - proj @ R1
        Saa, Sa1, S11b = moments(Ra, Ra), moments(Ra, R1b), moments(R1b, R1b)
        lam_r, vec = _gen_eig(Sa1.T @ np.linalg.solve(Saa, Sa1), S11b)
        lam_r = np.clip(lam_r, 0.0, 1.0 - 1e-15)
        stat = T * float(np.sum(np.log((1 - lam_r[:r]) / (1 - lam[:r]))))
        b_hat = vec[:, :r]
        a_hat = A @ (Sa1 @ b_hat @ np.linalg.inv(b_hat.T @ S11b @ b_hat))
    else:
        hypothesis = "beta_known"
        raw = np.asarray(
            (
                beta_known.to_numpy()
                if isinstance(beta_known, pd.DataFrame)
                else beta_known
            ),
            dtype=float,
        )
        rows = raw.shape[0]
        Hk = _matrix(beta_known, K if rows == K else k1, "beta_known", beta_names[:K])
        r1 = Hk.shape[1]
        if r1 > r:
            raise MethodIncompatibility(
                f"johansen_lrtest: {r1} known vectors exceed rank = {r}."
            )
        df = r1 * (k1 - r)
        if extra and Hk.shape[0] == K:
            Hk = np.vstack([Hk, np.zeros((1, r1))])
        # known directions
        rho, _ = _gen_eig(
            Hk.T @ S01.T @ np.linalg.solve(S00, S01) @ Hk, Hk.T @ S11 @ Hk
        )
        rho = np.clip(rho, 0.0, 1.0 - 1e-15)
        # free directions, conditional on Hk' R1
        Rh = R1 @ Hk
        proj = Rh @ np.linalg.solve(Rh.T @ Rh, Rh.T)
        R0h = R0 - proj @ R0
        u, _, _ = np.linalg.svd(Hk, full_matrices=True)
        Hperp = u[:, r1:]
        R1h = (R1 - proj @ R1) @ Hperp
        S00h, S01h, S11h = moments(R0h, R0h), moments(R0h, R1h), moments(R1h, R1h)
        free = r - r1
        if free:
            lam_f, vec = _gen_eig(S01h.T @ np.linalg.solve(S00h, S01h), S11h)
            lam_f = np.clip(lam_f, 0.0, 1.0 - 1e-15)
            psi = Hperp @ vec[:, :free]
        else:
            lam_f = np.zeros(0)
            psi = np.zeros((k1, 0))
        lam_r = np.concatenate([rho[:r1], lam_f[:free]])
        stat = T * float(
            np.sum(np.log(1 - rho[:r1]))
            + np.sum(np.log(1 - lam_f[:free]))
            - np.sum(np.log(1 - lam[:r]))
        )
        b_hat = np.hstack([Hk, psi])
        a_hat = S01 @ b_hat @ np.linalg.inv(b_hat.T @ S11 @ b_hat)

    if hypothesis != "beta_known":
        b_hat = _normalise(b_hat)
        a_hat = S01 @ b_hat @ np.linalg.inv(b_hat.T @ S11 @ b_hat)
        if hypothesis == "loading":
            a_hat = A @ np.linalg.lstsq(A, a_hat, rcond=None)[0]
    cols = [f"_ce{i + 1}" for i in range(r)]
    stat = max(stat, 0.0)
    return JohansenLRTest(
        statistic=float(stat),
        df=int(df),
        pvalue=float(stats.chi2.sf(stat, df)),
        hypothesis=hypothesis,
        rank=r,
        eigenvalues=lam[:K],
        eigenvalues_restricted=np.asarray(lam_r[:r]),
        beta=pd.DataFrame(b_hat, index=beta_names, columns=cols),
        loading=pd.DataFrame(a_hat, index=names, columns=cols),
        n_used=int(T),
        lags=lags,
        trend=case,
    )
