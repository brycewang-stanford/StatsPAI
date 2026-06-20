"""CATE evaluation: RATE / AUTOC / Qini / TOC for arbitrary CATE arrays.

Yadlowsky-Fleming-Shah-Brunskill-Wager (2025, JASA, arXiv:2111.07966)
introduced the **Rank-weighted Average Treatment Effect** (RATE) as a
model-free way to evaluate any CATE-style prioritisation rule. Given a
priority score :math:`S(X_i)` (typically the estimated CATE from any
estimator — meta-learner, BCF, causal forest, neural-net) and an AIPW
pseudo-outcome :math:`\\Psi_i` constructed from the observed
:math:`(Y_i, T_i)` plus cross-fitted nuisances :math:`(\\hat m, \\hat e)`,

.. math::
    \\text{TOC}(q) = E[\\tau(X) \\mid S(X) \\ge Q_{1-q}(S)] - E[\\tau(X)],

with two scalar summaries:

- **AUTOC** (Area Under the TOC curve, unweighted):
  :math:`\\int_0^1 \\text{TOC}(q)\\,dq`.
- **Qini coefficient**: :math:`\\int_0^1 q \\cdot \\text{TOC}(q)\\,dq`.

This module exposes :func:`cate_eval` as a *backbone-agnostic* drop-in
companion: feed it any CATE estimate plus the data, and you get the
same publication-quality summary the ``grf`` package produces for its
own forest output. It cross-fits :math:`\\hat m, \\hat e` internally
unless you pass them in.

Closed-form influence-function SE follows the rank-kernel derivation in
``statspai.forest.forest_inference.rate``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from .._result_serialize import ResultProtocolMixin


@dataclass
class CATEEvalResult(ResultProtocolMixin):
    """Output of :func:`cate_eval`.

    Attributes
    ----------
    autoc : float
    autoc_se : float
    autoc_ci : (float, float)
    qini : float
    qini_se : float
    qini_ci : (float, float)
    toc_curve : pd.DataFrame
        Columns ``q`` and ``toc``; one row per quantile grid point.
    n_obs : int
    target : str
        ``"AUTOC"`` (default) or ``"QINI"``.
    method : str
        Always ``"Yadlowsky et al. 2025 (DR-RATE, IF-SE)"``.

    Examples
    --------
    >>> import numpy as np
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> n = 400
    >>> X = rng.normal(size=(n, 3))
    >>> T = rng.integers(0, 2, n)
    >>> tau = 1.0 + X[:, 0]
    >>> Y = X[:, 1] + tau * T + rng.normal(scale=0.5, size=n)
    >>> cate = 1.0 + X[:, 0]  # priority score to evaluate
    >>> res = sp.cate_eval(cate, Y, T, X=X, n_folds=3, q_grid=20, random_state=0)
    >>> isinstance(res, sp.CATEEvalResult)
    True
    >>> res.toc_curve.columns.tolist()
    ['q', 'toc']
    >>> bool(res.n_obs == n)
    True
    """

    autoc: float
    autoc_se: float
    autoc_ci: Tuple[float, float]
    qini: float
    qini_se: float
    qini_ci: Tuple[float, float]
    toc_curve: pd.DataFrame
    n_obs: int
    target: str = "AUTOC"
    alpha: float = 0.05
    method: str = "Yadlowsky et al. 2025 (DR-RATE, IF-SE)"

    def summary(self) -> str:
        return (
            "CATE Evaluation (RATE / AUTOC / Qini)\n"
            "----------------------------------------\n"
            f"  N           : {self.n_obs:,}\n"
            f"  AUTOC       : {self.autoc:+.4f} (SE {self.autoc_se:.4f}, "
            f"95% CI [{self.autoc_ci[0]:+.4f}, {self.autoc_ci[1]:+.4f}])\n"
            f"  Qini        : {self.qini:+.4f} (SE {self.qini_se:.4f}, "
            f"95% CI [{self.qini_ci[0]:+.4f}, {self.qini_ci[1]:+.4f}])\n"
            f"  Method      : {self.method}\n"
            "  Reference   : Yadlowsky S., Fleming S., Shah N.,\n"
            "                Brunskill E., Wager S. (2025). RATE. JASA.\n"
            "                arXiv:2111.07966."
        )

    def __repr__(self) -> str:  # pragma: no cover
        return self.summary()

    def plot(
        self,
        ax: Any = None,
        target: Optional[str] = None,
        figsize: Tuple[float, float] = (6.0, 4.0),
    ) -> tuple[Any, Any]:
        """Plot the TOC curve plus a dashed zero line.

        ``target`` defaults to ``self.target``; pass ``"both"`` to overlay
        AUTOC's curve with a marker for the QINI weighted area.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as e:  # pragma: no cover
            raise ImportError("matplotlib required for plot()") from e
        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.figure
        ax.plot(
            self.toc_curve["q"],
            self.toc_curve["toc"],
            color="#1f77b4",
            lw=1.8,
            label="TOC(q)",
        )
        ax.fill_between(
            self.toc_curve["q"], 0, self.toc_curve["toc"], color="#1f77b4", alpha=0.15
        )
        ax.axhline(0, color="#555", lw=0.7, ls="--")
        ax.set_xlabel(r"Quantile $q$ of priority score")
        ax.set_ylabel(r"$E[\tau(X) \mid S \ge Q_{1-q}] - E[\tau(X)]$")
        ax.set_title(
            f"TOC curve  (AUTOC = {self.autoc:+.3f},  Qini = {self.qini:+.3f})"
        )
        ax.legend(loc="upper right", fontsize=9)
        return fig, ax


def _aipw_pseudo_outcome(
    Y: np.ndarray,
    T: np.ndarray,
    e_hat: np.ndarray,
    mu1_hat: Optional[np.ndarray],
    mu0_hat: Optional[np.ndarray],
    m_hat: np.ndarray,
) -> np.ndarray:
    """AIPW Ψ_i for binary T."""
    if mu1_hat is not None and mu0_hat is not None:
        mu_T = T * mu1_hat + (1 - T) * mu0_hat
        return np.asarray(
            (mu1_hat - mu0_hat) + (T - e_hat) * (Y - mu_T) / (e_hat * (1 - e_hat)),
            dtype=float,
        )
    return np.asarray(
        T * (Y - m_hat) / e_hat - (1 - T) * (Y - m_hat) / (1 - e_hat),
        dtype=float,
    )


def _crossfit_nuisances(
    X: np.ndarray,
    Y: np.ndarray,
    T: np.ndarray,
    n_folds: int,
    random_state: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Cross-fit m̂, ê, μ̂_1, μ̂_0 with sklearn GBM defaults."""
    from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
    from sklearn.model_selection import KFold

    kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    n = len(Y)
    m_hat = np.zeros(n)
    e_hat = np.zeros(n)
    mu1 = np.zeros(n)
    mu0 = np.zeros(n)
    for tr, te in kf.split(X):
        Xtr, Xte = X[tr], X[te]
        Ytr = Y[tr]
        Ttr = T[tr]
        # m̂(X) = E[Y|X]
        m = GradientBoostingRegressor(
            n_estimators=200, max_depth=3, random_state=random_state
        ).fit(Xtr, Ytr)
        m_hat[te] = m.predict(Xte)
        # ê(X) = P(T=1|X)
        e = GradientBoostingClassifier(
            n_estimators=200, max_depth=3, random_state=random_state
        ).fit(Xtr, Ttr)
        e_hat[te] = e.predict_proba(Xte)[:, 1]
        # μ̂_1(X), μ̂_0(X)
        mask1 = Ttr == 1
        mask0 = Ttr == 0
        if mask1.any():
            m1 = GradientBoostingRegressor(
                n_estimators=200, max_depth=3, random_state=random_state
            ).fit(Xtr[mask1], Ytr[mask1])
            mu1[te] = m1.predict(Xte)
        if mask0.any():
            m0 = GradientBoostingRegressor(
                n_estimators=200, max_depth=3, random_state=random_state
            ).fit(Xtr[mask0], Ytr[mask0])
            mu0[te] = m0.predict(Xte)
    return m_hat, e_hat, mu1, mu0


def cate_eval(
    cate: np.ndarray,
    Y: np.ndarray,
    T: np.ndarray,
    X: Optional[np.ndarray] = None,
    *,
    e_hat: Optional[np.ndarray] = None,
    m_hat: Optional[np.ndarray] = None,
    mu1_hat: Optional[np.ndarray] = None,
    mu0_hat: Optional[np.ndarray] = None,
    n_folds: int = 5,
    alpha: float = 0.05,
    clip: float = 0.02,
    q_grid: int = 100,
    random_state: int = 0,
    target: str = "AUTOC",
) -> CATEEvalResult:
    r"""Evaluate any CATE estimator via RATE / AUTOC / Qini (Yadlowsky 2025).

    Parameters
    ----------
    cate : (n,) array
        Estimated CATE :math:`\hat\tau(X_i)` from any estimator.
    Y, T : (n,) arrays
        Observed outcome and binary treatment.
    X : (n, d) array, optional
        Required if any of ``e_hat / m_hat / mu1_hat / mu0_hat`` is None;
        cross-fit nuisances are estimated with GBM defaults.
    e_hat, m_hat, mu1_hat, mu0_hat : (n,) arrays, optional
        Pre-computed nuisance predictions (e.g., from the same estimator
        that produced ``cate``). If provided, no internal cross-fitting
        runs.
    n_folds : int, default 5
    alpha : float, default 0.05
    clip : float, default 0.02
        Propensity clip.
    q_grid : int, default 100
    random_state : int, default 0
    target : {"AUTOC", "QINI"}, default "AUTOC"
        Which scalar headline to emit; both are computed and returned.

    Returns
    -------
    CATEEvalResult

    Examples
    --------
    >>> # Meta-learner CATE → RATE evaluation:
    >>> ml = sp.metalearner(method="dr", ...)        # doctest: +SKIP
    >>> tau_hat = ml.predict(X)                       # doctest: +SKIP
    >>> sp.cate_eval(tau_hat, Y, T, X=X).summary()    # doctest: +SKIP
    """
    cate = np.asarray(cate, dtype=float).ravel()
    Y = np.asarray(Y, dtype=float).ravel()
    T = np.asarray(T, dtype=float).ravel()
    n = len(cate)
    if not (len(Y) == n and len(T) == n):
        raise ValueError("cate, Y, T must all have length n.")

    if e_hat is None or m_hat is None or mu1_hat is None or mu0_hat is None:
        if X is None:
            raise ValueError(
                "Provide X (so cate_eval can cross-fit nuisances) or pass "
                "e_hat / m_hat / mu1_hat / mu0_hat directly."
            )
        m_hat_cf, e_hat_cf, mu1_cf, mu0_cf = _crossfit_nuisances(
            np.asarray(X, dtype=float),
            Y,
            T,
            n_folds,
            random_state,
        )
        e_hat = e_hat_cf if e_hat is None else np.asarray(e_hat).ravel()
        m_hat = m_hat_cf if m_hat is None else np.asarray(m_hat).ravel()
        mu1_hat = mu1_cf if mu1_hat is None else np.asarray(mu1_hat).ravel()
        mu0_hat = mu0_cf if mu0_hat is None else np.asarray(mu0_hat).ravel()
    e_hat = np.clip(np.asarray(e_hat, dtype=float).ravel(), clip, 1 - clip)
    m_hat = np.asarray(m_hat, dtype=float).ravel()
    mu1_hat = np.asarray(mu1_hat, dtype=float).ravel()
    mu0_hat = np.asarray(mu0_hat, dtype=float).ravel()

    psi = _aipw_pseudo_outcome(Y, T, e_hat, mu1_hat, mu0_hat, m_hat)
    psi_bar = float(psi.mean())

    desc_order = np.argsort(-cate, kind="mergesort")
    rank = np.empty(n, dtype=np.float64)
    rank[desc_order] = np.arange(1, n + 1)
    u = rank / n

    H = np.concatenate([[0.0], np.cumsum(1.0 / np.arange(1, n + 1))])
    R_int = rank.astype(np.int64)
    w_autoc = H[n] - H[R_int - 1]
    w_qini = 1.0 - u

    phi_autoc = psi * (w_autoc - 1.0)
    phi_qini = psi * (w_qini - 0.5)

    autoc = float(phi_autoc.mean())
    qini = float(phi_qini.mean())

    def _se(phi: np.ndarray) -> float:
        c = phi - phi.mean()
        var = float(c @ c) / (n * (n - 1)) if n > 1 else float("nan")
        return float(np.sqrt(max(var, 0.0)))

    se_autoc = _se(phi_autoc)
    se_qini = _se(phi_qini)
    z = float(stats.norm.ppf(1 - alpha / 2))

    psi_sorted = psi[desc_order]
    cum = np.cumsum(psi_sorted) / np.arange(1, n + 1)
    toc_all = cum - psi_bar
    q_targets = np.linspace(1.0 / q_grid, 1.0, q_grid)
    idx_sel = np.clip((q_targets * n).astype(np.int64) - 1, 0, n - 1)
    toc_df = pd.DataFrame({"q": q_targets, "toc": toc_all[idx_sel]})

    return CATEEvalResult(
        autoc=autoc,
        autoc_se=se_autoc,
        autoc_ci=(autoc - z * se_autoc, autoc + z * se_autoc),
        qini=qini,
        qini_se=se_qini,
        qini_ci=(qini - z * se_qini, qini + z * se_qini),
        toc_curve=toc_df,
        n_obs=n,
        target=target.upper(),
        alpha=alpha,
    )
