"""Distributional IV: the LATE at every quantile of the outcome.

Standard IV identifies the LATE on the *mean*.  Distributional IV identifies
it on the *whole distribution*: for compliers,

.. math::

    QTE_c(\\tau) = F^{-1}_{Y(1)|c}(\\tau) - F^{-1}_{Y(0)|c}(\\tau)

The two complier marginal CDFs are recovered by Abadie's kappa weighting
(:func:`statspai.qte._core.complier_cdfs`); with covariates this is the
Frolich & Melly (2013) unconditional IV-QTE.

.. warning::

    **Correctness fix in v1.21.0.**  Versions <= 1.20.0 implemented a "Wald
    ratio of quantiles",
    ``[Q(tau|Z=1) - Q(tau|Z=0)] / [E(D|Z=1) - E(D|Z=0)]``.  The quantile
    operator is not linear, so the mean-Wald rescaling does not carry over
    and that expression is inconsistent for any quantile estimand.  On a
    30/50/20 always-taker/complier/never-taker design with a true complier
    ``QTE(tau) == 2.0``, the old code returned ``~4.0`` at every tau
    (n = 200,000) — a bias of exactly ``1 / Delta_p``.  See MIGRATION.md for
    the approximate ``old ~ new / Delta_p`` back-conversion.

References
----------
abadie2002bootstrap, abadie2003semiparametric, frolich2013unconditional,
holovchak2025distributional, shaw2025model
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from .._result_serialize import ResultProtocolMixin
from ._core import complier_cdfs, invert_cdf, kernel_density_at


@dataclass
class DistIVResult(ResultProtocolMixin):
    """Distributional IV: complier LATE per quantile.

    Returned by :func:`dist_iv`. Holds the complier quantile treatment effect
    at each requested quantile with standard errors and confidence intervals,
    the estimated complier share, and which SE method produced them.

    Attributes
    ----------
    quantiles : ndarray
        Probability levels.
    late_q : ndarray
        Complier QTE at each level.
    se_q, ci_low, ci_high : ndarray
        Standard errors and confidence bounds.
    complier_share : float
        ``E[kappa]``, the estimated share of compliers.
    se_method : str
        ``'analytic'`` (influence function) or ``'bootstrap'``.
    n_obs : int

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(42)
    >>> n = 4000
    >>> z = rng.integers(0, 2, n)
    >>> d = ((0.3 + 0.5 * z + rng.normal(0, 0.3, n)) > 0.5).astype(int)
    >>> y = 1.0 + 1.0 * d + rng.normal(0, 1, n)
    >>> df = pd.DataFrame({"y": y, "d": d, "z": z})
    >>> res = sp.dist_iv(df, y="y", treat="d", instrument="z",
    ...                  quantiles=np.array([0.25, 0.5, 0.75]))
    >>> isinstance(res, sp.DistIVResult)
    True
    >>> bool(np.all(np.abs(res.late_q - 1.0) < 0.3))  # true LATE = 1.0
    True
    """

    quantiles: np.ndarray
    late_q: np.ndarray
    se_q: np.ndarray
    ci_low: np.ndarray
    ci_high: np.ndarray
    n_obs: int
    complier_share: float = float("nan")
    se_method: str = "analytic"
    method: str = "Distributional IV (Abadie kappa-weighted complier QTE)"
    model_info: dict = field(default_factory=dict)

    def summary(self) -> str:
        rows = [
            self.method,
            "=" * 58,
            f"  N            : {self.n_obs:,}",
            f"  Complier share: {self.complier_share:.4f}",
            f"  SE method     : {self.se_method}",
            "",
            "  Quantile  LATE       SE        95% CI",
            "  " + "-" * 50,
        ]
        for q, l, s, lo, hi in zip(
            self.quantiles, self.late_q, self.se_q, self.ci_low, self.ci_high
        ):
            rows.append(f"  {q:6.2f}   {l:+.4f}   {s:.4f}   [{lo:+.4f}, {hi:+.4f}]")
        return "\n".join(rows)

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "quantile": self.quantiles,
                "late": self.late_q,
                "se": self.se_q,
                "ci_low": self.ci_low,
                "ci_high": self.ci_high,
            }
        )

    def plot(self, ax: Any = None) -> Any:
        """Plot the complier QTE curve with its CI band. Returns (fig, ax)."""
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))
        else:
            fig = ax.get_figure()
        ax.plot(
            self.quantiles, self.late_q, "o-", color="#2c7bb6", lw=2, label="LATE(τ)"
        )
        ax.fill_between(
            self.quantiles, self.ci_low, self.ci_high, alpha=0.2, color="#2c7bb6"
        )
        ax.axhline(0, color="grey", ls="--", lw=0.8)
        ax.set_xlabel("Quantile (τ)")
        ax.set_ylabel("Complier LATE")
        ax.set_title(self.method)
        ax.legend()
        fig.tight_layout()
        return fig, ax


# ══════════════════════════════════════════════════════════════════════
#  Internals
# ══════════════════════════════════════════════════════════════════════


def _logit_pi(X: np.ndarray, Z: np.ndarray) -> np.ndarray:
    """P(Z = 1 | X) by logistic regression, trimmed away from {0, 1}."""
    from sklearn.linear_model import LogisticRegression

    clf = LogisticRegression(max_iter=2000, solver="lbfgs", C=1e6)
    clf.fit(X, Z)
    pi = np.asarray(clf.predict_proba(X)[:, 1])
    return np.clip(pi, 0.01, 0.99)


def _first_stage_t(D: np.ndarray, Z: np.ndarray) -> float:
    """t-statistic for ``E[D|Z=1] - E[D|Z=0] = 0`` (two-sample difference)."""
    d1, d0 = D[Z == 1], D[Z == 0]
    if len(d1) < 2 or len(d0) < 2:
        return 0.0
    diff = float(d1.mean() - d0.mean())
    var = d1.var(ddof=1) / len(d1) + d0.var(ddof=1) / len(d0)
    if var <= 0:
        return 0.0 if diff == 0 else float(np.inf)
    return diff / float(np.sqrt(var))


def _point_estimate(
    Y: np.ndarray,
    D: np.ndarray,
    Z: np.ndarray,
    pi: Optional[np.ndarray],
    taus: np.ndarray,
) -> Any:
    """(late_q, complier_share, q1, q0, grid, F1, F0) or None if degenerate."""
    out = complier_cdfs(Y, D, Z, pi=pi)
    if out is None:
        return None
    grid, F1, F0, share = out
    q1 = invert_cdf(grid, F1, taus)
    q0 = invert_cdf(grid, F0, taus)
    return q1 - q0, share, q1, q0


def _analytic_se(
    Y: np.ndarray,
    D: np.ndarray,
    Z: np.ndarray,
    pi: Optional[np.ndarray],
    taus: np.ndarray,
    q1: np.ndarray,
    q0: np.ndarray,
    share: float,
) -> np.ndarray:
    """Influence-function SE for the Abadie-weighted complier QTE.

    For a weighted CDF ``F_j(y) = E[kappa_j 1{Y<=y}] / E[kappa_j]`` the
    quantile influence function is

        psi_j(tau) = -kappa_j (1{Y <= Q_j(tau)} - tau) / (P_c f_j(Q_j(tau)))

    and ``QTE(tau) = Q_1(tau) - Q_0(tau)`` gives ``psi_1 - psi_0``.
    ``SE = sd(psi) / sqrt(n)``.

    This treats ``pi(X)`` as KNOWN.  That is exact when the instrument is
    randomised with a known assignment probability, and understates variance
    when ``pi`` is estimated from covariates — which is why the covariate
    path defaults to the bootstrap instead.
    """
    from ._core import abadie_kappa

    n = len(Y)
    if pi is None:
        pi = np.full(n, float(np.mean(Z)))
    kappa_1, kappa_0 = abadie_kappa(D, Z, pi)

    f1 = kernel_density_at(Y, q1, weights=kappa_1)
    f0 = kernel_density_at(Y, q0, weights=kappa_0)

    se = np.empty(len(taus))
    for j, tau in enumerate(taus):
        if not (np.isfinite(f1[j]) and np.isfinite(f0[j])):
            se[j] = np.nan
            continue
        psi1 = -kappa_1 * ((Y <= q1[j]).astype(float) - tau) / (share * f1[j])
        psi0 = -kappa_0 * ((Y <= q0[j]).astype(float) - tau) / (share * f0[j])
        psi = psi1 - psi0
        se[j] = float(np.std(psi, ddof=1) / np.sqrt(n))
    return se


# ══════════════════════════════════════════════════════════════════════
#  Public API
# ══════════════════════════════════════════════════════════════════════


def dist_iv(
    data: pd.DataFrame,
    y: str,
    treat: str,
    instrument: str,
    covariates: Optional[List[str]] = None,
    quantiles: Optional[np.ndarray] = None,
    alpha: float = 0.05,
    se: str = "auto",
    n_boot: int = 200,
    seed: int = 0,
) -> DistIVResult:
    """Distributional IV: the complier LATE at each quantile of ``y``.

    Identifies ``F_{Y(1)|complier}`` and ``F_{Y(0)|complier}`` by Abadie
    kappa weighting and returns their quantile difference. With covariates
    this is the Frolich & Melly (2013) unconditional IV-QTE.

    Parameters
    ----------
    data : DataFrame
    y, treat, instrument : str
        Outcome, binary treatment, binary instrument.
    covariates : list of str, optional
        Conditioning set for ``P(Z=1|X)``. Unlike versions <= 1.20.0 (which
        accepted and silently ignored this argument) it now changes the
        estimate.
    quantiles : array-like, optional
        Defaults to ``(0.1, 0.25, 0.5, 0.75, 0.9)``.
    alpha : float, default 0.05
    se : {'auto', 'analytic', 'bootstrap', 'none'}, default 'auto'
        ``'auto'`` picks ``'analytic'`` without covariates and ``'bootstrap'``
        with them (the analytic influence function treats ``pi(X)`` as known).
    n_boot : int, default 200
        Bootstrap replications when ``se='bootstrap'``.
    seed : int

    Returns
    -------
    DistIVResult

    Notes
    -----
    Requires a binary instrument satisfying the standard LATE assumptions
    (random assignment, exclusion, monotonicity, non-zero first stage).
    A non-positive estimated complier share raises.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(42)
    >>> n = 4000
    >>> z = rng.integers(0, 2, n)
    >>> d = ((0.3 + 0.5 * z + rng.normal(0, 0.3, n)) > 0.5).astype(int)
    >>> y = 1.0 + 1.0 * d + rng.normal(0, 1, n)
    >>> df = pd.DataFrame({"y": y, "d": d, "z": z})
    >>> res = sp.dist_iv(df, y="y", treat="d", instrument="z",
    ...                  quantiles=np.array([0.25, 0.5, 0.75]))
    >>> bool(np.all(np.abs(res.late_q - 1.0) < 0.3))  # true LATE = 1.0
    True
    >>> res.se_method
    'analytic'

    References
    ----------
    abadie2003semiparametric, frolich2013unconditional
    """
    if quantiles is None:
        quantiles = np.array([0.1, 0.25, 0.5, 0.75, 0.9])
    taus = np.atleast_1d(np.asarray(quantiles, dtype=float))
    if np.any((taus <= 0) | (taus >= 1)):
        raise ValueError("quantiles must lie strictly inside (0, 1).")

    cov = list(covariates or [])
    df = data[[y, treat, instrument] + cov].dropna().reset_index(drop=True)
    Y = df[y].to_numpy(float)
    D = df[treat].to_numpy(float)
    Z = df[instrument].to_numpy(float)
    n = len(df)

    if not np.all(np.isin(Z, (0.0, 1.0))):
        raise ValueError("dist_iv requires a binary (0/1) instrument.")
    if not np.all(np.isin(D, (0.0, 1.0))):
        raise ValueError("dist_iv requires a binary (0/1) treatment.")

    X = df[cov].to_numpy(float) if cov else None
    pi = _logit_pi(X, Z.astype(int)) if X is not None else None

    if se == "auto":
        se = "bootstrap" if cov else "analytic"
    if se not in ("analytic", "bootstrap", "none"):
        raise ValueError(
            f"se must be 'auto', 'analytic', 'bootstrap' or 'none', got {se!r}"
        )

    point = _point_estimate(Y, D, Z, pi, taus)
    if point is None:
        raise ValueError(
            "Degenerate first stage: the Abadie complier share is ~0. "
            "Check that the instrument moves treatment and that "
            "0 < P(Z=1|X) < 1 for all observations."
        )
    late_q, share, q1, q0 = point
    if share <= 0:
        raise ValueError(
            f"Estimated complier share is {share:.4g} <= 0 — the instrument "
            "violates monotonicity (or the first stage has the wrong sign)."
        )

    # Weak first stage: everything downstream divides by `share`, so a share
    # that is positive only by sampling noise yields explosive, meaningless
    # quantile effects. Fail loudly (CLAUDE.md §7) rather than return numbers
    # that look like estimates.
    fs_t = _first_stage_t(D, Z)
    if abs(fs_t) < 2.0 or share < 0.05:
        warnings.warn(
            f"dist_iv: weak first stage (complier share {share:.4g}, "
            f"first-stage t = {fs_t:.2f}). The complier CDFs divide by this "
            "share, so the reported quantile effects and their SEs are "
            "unreliable. Check instrument relevance before using them.",
            UserWarning,
            stacklevel=2,
        )

    # ---- standard errors ------------------------------------------- #
    if se == "none":
        se_q = np.full(len(taus), np.nan)
    elif se == "analytic":
        se_q = _analytic_se(Y, D, Z, pi, taus, q1, q0, share)
    else:
        rng = np.random.default_rng(seed)
        boot = np.full((n_boot, len(taus)), np.nan)
        for b in range(n_boot):
            idx = rng.integers(0, n, size=n)
            Yb, Db, Zb = Y[idx], D[idx], Z[idx]
            if Zb.min() == Zb.max():
                continue
            # Re-fit pi on the resample: treating it as fixed is exactly the
            # variance component the bootstrap path exists to capture.
            pib = _logit_pi(X[idx], Zb.astype(int)) if X is not None else None
            pb = _point_estimate(Yb, Db, Zb, pib, taus)
            if pb is not None:
                boot[b] = pb[0]
        n_finite = np.isfinite(boot).sum(axis=0)
        se_q = np.nanstd(boot, axis=0, ddof=1)
        se_q = np.where(np.isfinite(se_q) & (n_finite >= 2), se_q, np.nan)
        if (n_finite < n_boot).any():
            warnings.warn(
                f"dist_iv: {int((n_finite < n_boot).sum())}/{len(taus)} "
                f"quantile(s) had bootstrap replications fail (degenerate "
                f"resample); those SEs use fewer replicates or are NaN.",
                RuntimeWarning,
                stacklevel=2,
            )

    z_crit = float(stats.norm.ppf(1 - alpha / 2))
    result = DistIVResult(
        quantiles=taus,
        late_q=late_q,
        se_q=se_q,
        ci_low=late_q - z_crit * se_q,
        ci_high=late_q + z_crit * se_q,
        n_obs=n,
        complier_share=share,
        se_method=se,
        model_info={"covariates": cov or None, "n_boot": n_boot, "alpha": alpha},
    )
    try:
        from ..output._lineage import attach_provenance as _attach_prov

        _attach_prov(
            result,
            function="sp.qte.dist_iv",
            params={
                "y": y,
                "treat": treat,
                "instrument": instrument,
                "covariates": cov or None,
                "quantiles": list(taus),
                "alpha": alpha,
                "se": se,
                "n_boot": n_boot,
                "seed": seed,
            },
            data=data,
            overwrite=False,
        )
    except Exception:  # pragma: no cover - provenance is best-effort metadata
        pass
    return result


def kan_dlate(
    data: pd.DataFrame,
    y: str,
    treat: str,
    instrument: str,
    covariates: Optional[List[str]] = None,
    quantiles: Optional[np.ndarray] = None,
    alpha: float = 0.05,
    n_boot: int = 200,
    seed: int = 0,
) -> DistIVResult:
    """Deprecated alias for :func:`dist_iv`.

    .. deprecated:: 1.21.0
        This function never implemented a distinct estimator — it has always
        forwarded to :func:`dist_iv` — and its docstring attributed the method
        to two different authors ("Kennedy 2025" in the module header, "Shaw
        2025" here) for the same arXiv ID. Verification against arXiv and the
        DataCite DOI registry shows arXiv:2506.12765 is *Model Risk in
        Machine-Learning Distributional IV Estimation* by **Charles Shaw**
        alone, and neither its title nor its v1 abstract mentions a
        Kolmogorov-Arnold network. Rather than keep a function that claims a
        method it does not implement, use :func:`dist_iv` directly. Scheduled
        for removal in 1.23.0; see MIGRATION.md.

    Returns
    -------
    DistIVResult
        Exactly what :func:`dist_iv` returns.

    Examples
    --------
    >>> import warnings
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(42)
    >>> n = 4000
    >>> z = rng.integers(0, 2, n)
    >>> d = ((0.3 + 0.5 * z + rng.normal(0, 0.3, n)) > 0.5).astype(int)
    >>> y = 1.0 + 1.0 * d + rng.normal(0, 1, n)
    >>> df = pd.DataFrame({"y": y, "d": d, "z": z})
    >>> qs = np.array([0.25, 0.5, 0.75])
    >>> with warnings.catch_warnings():  # deprecated: forwards to dist_iv
    ...     warnings.simplefilter("ignore", DeprecationWarning)
    ...     old = sp.kan_dlate(df, y="y", treat="d", instrument="z", quantiles=qs)
    >>> new = sp.dist_iv(df, y="y", treat="d", instrument="z", quantiles=qs)
    >>> bool(np.allclose(old.late_q, new.late_q))  # identical to dist_iv
    True
    """
    warnings.warn(
        "sp.kan_dlate is deprecated and will be removed in 1.23.0: it is a "
        "pure alias for sp.dist_iv and never implemented a KAN bridge "
        "function. Call sp.dist_iv directly. See MIGRATION.md.",
        DeprecationWarning,
        stacklevel=2,
    )
    return dist_iv(
        data=data,
        y=y,
        treat=treat,
        instrument=instrument,
        covariates=covariates,
        quantiles=quantiles,
        alpha=alpha,
        n_boot=n_boot,
        seed=seed,
    )
