"""Quantile treatment effect estimation.

Methods exposed by :func:`qte`
------------------------------
``'firpo_qte'``
    Firpo (2007) efficient **unconditional QTE**,
    ``F^-1_{Y(1)}(τ) − F^-1_{Y(0)}(τ)``, by propensity-score reweighting
    with ``D/p(X)`` and ``(1−D)/(1−p(X))``.
``'firpo_qtt'``
    Firpo (2007) **QTT**: same contrast among the treated, reweighting
    controls by ``p/(1−p)``.
``'conditional_qr'``
    Coefficient on ``D`` in a quantile regression of ``Y`` on ``D + X``
    (Koenker & Bassett 1978). A **conditional** QTE — a different estimand
    from Firpo's, with no causal reading absent rank invariance.
``'distribution'``
    IPW counterfactual-distribution estimator. Computes the **QTT**.

Also here
---------
:func:`qdid`
    **Quantile DiD (QDiD)**: the DiD contrast applied to quantiles,
    ``[Q₁₁(τ) − Q₁₀(τ)] − [Q₀₁(τ) − Q₀₀(τ)]``. Note this is *not*
    Changes-in-Changes: Athey & Imbens (2006) propose CiC and explicitly
    criticise QDiD. For CiC use :func:`statspai.cic`.

.. versionchanged:: 1.21.0
    ``method='quantile_regression'`` was labelled Firpo (2007) but computed
    the conditional estimand; it is renamed ``'conditional_qr'`` and the
    genuine Firpo estimators were added. :func:`qdid` no longer claims to
    implement Athey & Imbens (2006). See MIGRATION.md.

References
----------
firpo2007efficient, koenker1978regression, athey2006identification
"""

from __future__ import annotations

import warnings
from typing import Any, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from .._result_serialize import ResultProtocolMixin

# ══════════════════════════════════════════════════════════════════════
#  QTEResult
# ══════════════════════════════════════════════════════════════════════


class QTEResult(ResultProtocolMixin):
    """Container for quantile treatment effect estimates.

    Attributes
    ----------
    quantiles : np.ndarray
        Quantile grid.
    effects : np.ndarray
        QTE point estimates.
    se : np.ndarray
        Bootstrap / analytical standard errors.
    ci_lower, ci_upper : np.ndarray
        Confidence interval bounds.
    ate : float
        Average treatment effect (for comparison).
    method : str
        Estimation method label.
    n_obs : int
        Sample size.
    alpha : float
        Significance level.

    Examples
    --------
    A :class:`QTEResult` is produced by estimators such as :func:`qte` and
    :func:`qdid`; inspect the quantile grid, per-quantile effects and the
    mean ATE:

    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> n = 300
    >>> d = rng.integers(0, 2, n)
    >>> y = 1.0 + 1.5 * d + rng.normal(0, 1, n)
    >>> df = pd.DataFrame({"y": y, "d": d})
    >>> res = sp.qte(df, y="y", treatment="d",
    ...              quantiles=[0.25, 0.5, 0.75], n_boot=50, seed=0)
    >>> isinstance(res, sp.QTEResult)
    True
    >>> res.quantiles.tolist()
    [0.25, 0.5, 0.75]
    >>> res.effects.shape
    (3,)
    """

    def __init__(
        self,
        quantiles: np.ndarray,
        effects: np.ndarray,
        se: np.ndarray,
        ci_lower: np.ndarray,
        ci_upper: np.ndarray,
        ate: float,
        method: str,
        n_obs: int,
        alpha: float = 0.05,
        model_info: Optional[dict] = None,
        ci_lower_uniform: Optional[np.ndarray] = None,
        ci_upper_uniform: Optional[np.ndarray] = None,
        uniform_crit: float = float("nan"),
    ):
        self.quantiles = np.asarray(quantiles)
        self.effects = np.asarray(effects)
        self.se = np.asarray(se)
        self.ci_lower = np.asarray(ci_lower)
        self.ci_upper = np.asarray(ci_upper)
        self.ate = float(ate)
        self.method = method
        self.n_obs = int(n_obs)
        self.alpha = float(alpha)
        self.model_info = model_info or {}
        # Simultaneous band over the whole quantile grid. A pointwise band
        # covers each tau separately; claims like "the effect is zero at
        # every quantile" need this one. None when the estimator cannot
        # supply influence functions (bootstrap-only paths).
        self.ci_lower_uniform = (
            None if ci_lower_uniform is None else np.asarray(ci_lower_uniform)
        )
        self.ci_upper_uniform = (
            None if ci_upper_uniform is None else np.asarray(ci_upper_uniform)
        )
        self.uniform_crit = float(uniform_crit)

    # ── pretty printing ──────────────────────────────────────────── #

    @staticmethod
    def _stars(pv: float) -> str:
        if np.isnan(pv):
            return ""
        if pv < 0.01:
            return "***"
        if pv < 0.05:
            return "**"
        if pv < 0.1:
            return "*"
        return ""

    def summary(self) -> str:
        lines = []
        lines.append("━" * 64)
        lines.append(f"  {self.method}")
        lines.append("━" * 64)
        pct = int(100 * (1 - self.alpha))
        lines.append(
            f"  {'τ':>6s}  {'QTE':>10s}  {'SE':>9s}  "
            f"{'[' + str(pct) + '% CI]':>22s}"
        )
        lines.append("  " + "-" * 58)

        for i, tau in enumerate(self.quantiles):
            eff = self.effects[i]
            se_i = self.se[i]
            lo = self.ci_lower[i]
            hi = self.ci_upper[i]
            pv = 2 * (1 - stats.norm.cdf(abs(eff / se_i))) if se_i > 0 else np.nan
            s = self._stars(pv)
            lines.append(
                f"  {tau:6.2f}  {eff:>10.4f}{s:<3s}  ({se_i:.4f})  "
                f"[{lo:.4f}, {hi:.4f}]"
            )

        lines.append("  " + "-" * 58)
        lines.append(f"  ATE (mean):  {self.ate:.4f}")
        lines.append(f"  Observations: {self.n_obs:,}")
        lines.append("━" * 64)
        lines.append("  * p<0.1, ** p<0.05, *** p<0.01")
        out = "\n".join(lines)
        print(out)
        return out

    def _repr_html_(self) -> str:
        """Jupyter notebook HTML rendering."""
        pct = int(100 * (1 - self.alpha))
        rows = ""
        for i, tau in enumerate(self.quantiles):
            eff = self.effects[i]
            se_i = self.se[i]
            lo = self.ci_lower[i]
            hi = self.ci_upper[i]
            pv = 2 * (1 - stats.norm.cdf(abs(eff / se_i))) if se_i > 0 else np.nan
            s = self._stars(pv)
            rows += (
                f"<tr><td>{tau:.2f}</td><td>{eff:.4f}{s}</td>"
                f"<td>({se_i:.4f})</td><td>[{lo:.4f}, {hi:.4f}]</td></tr>\n"
            )
        return (
            f"<h4>{self.method}</h4>"
            f"<table><thead><tr><th>&tau;</th><th>QTE</th><th>SE</th>"
            f"<th>{pct}% CI</th></tr></thead><tbody>{rows}</tbody></table>"
            f"<p>ATE = {self.ate:.4f} &nbsp;|&nbsp; N = {self.n_obs:,}</p>"
        )

    def __repr__(self) -> str:
        return (
            f"QTEResult(method='{self.method}', "
            f"quantiles={list(self.quantiles)}, ate={self.ate:.4f})"
        )

    def to_frame(self) -> pd.DataFrame:
        """Tidy per-quantile table, including the uniform band when present."""
        out = pd.DataFrame(
            {
                "quantile": self.quantiles,
                "qte": self.effects,
                "se": self.se,
                "ci_lower": self.ci_lower,
                "ci_upper": self.ci_upper,
            }
        )
        if self.ci_lower_uniform is not None:
            out["ci_lower_uniform"] = self.ci_lower_uniform
            out["ci_upper_uniform"] = self.ci_upper_uniform
        return out

    # ── functional (curve-level) inference ────────────────────────── #

    def test_no_effect(self, kind: str = "ks", n_boot: int = 1000, seed: int = 0):
        """Test ``QTE(tau) = 0 at EVERY tau`` against "somewhere non-zero".

        Not what a row of pointwise p-values tests: with 19 quantiles at the
        5% level, roughly one spurious rejection is expected under the null.
        Requires ``se='analytic'``.
        """
        return self._functional_test(None, kind, n_boot, seed)

    def test_constant_effect(self, kind: str = "ks", n_boot: int = 1000, seed: int = 0):
        """Test ``QTE(tau)`` is the same at every tau against "it varies".

        Rejecting means treatment does something an average effect cannot
        express. Failing to reject means the ATE is an adequate summary.
        """
        # The estimand is the DEVIATION from the average effect, whose
        # influence function is psi_ik - mean_j(psi_ij): the component common
        # to every quantile cancels. Using the raw psi would overstate the
        # variance and make the test useless (measured: 0.000 rejection under
        # its own null).
        influence = self._influence()
        centered_est = self.effects - float(np.mean(self.effects))
        centered_inf = influence - influence.mean(axis=1, keepdims=True)
        from ._core import functional_test

        return functional_test(
            centered_est,
            centered_inf,
            null=None,
            kind=kind,
            n_boot=n_boot,
            seed=seed,
        )

    def _influence(self) -> np.ndarray:
        influence = self.model_info.get("influence")
        if influence is None:
            raise ValueError(
                "Curve-level tests need influence functions, which only the "
                "analytic standard-error path produces. Re-run with "
                "se='analytic' (available for method='firpo_qte' / "
                "'firpo_qtt')."
            )
        return np.asarray(influence, dtype=float)

    def _functional_test(self, null, kind: str, n_boot: int, seed: int):
        from ._core import functional_test

        return functional_test(
            self.effects,
            self._influence(),
            null=null,
            kind=kind,
            n_boot=n_boot,
            seed=seed,
        )

    # ── plot ──────────────────────────────────────────────────────── #

    def plot(self, ax: Any = None) -> Any:
        """QTE plot with CI bands and ATE reference line.

        Returns (fig, ax).
        """
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))
        else:
            fig = ax.get_figure()

        ax.plot(
            self.quantiles,
            self.effects,
            "o-",
            color="#2c7bb6",
            linewidth=2,
            markersize=5,
            label="QTE",
        )
        ax.fill_between(
            self.quantiles,
            self.ci_lower,
            self.ci_upper,
            alpha=0.2,
            color="#2c7bb6",
        )
        ax.axhline(0, color="grey", linestyle="--", linewidth=0.8)
        ax.axhline(
            self.ate,
            color="#d7191c",
            linestyle=":",
            linewidth=1.2,
            label=f"ATE = {self.ate:.4f}",
        )
        ax.set_xlabel("Quantile (τ)")
        ax.set_ylabel("Treatment Effect")
        ax.set_title(self.method)
        ax.legend()
        fig.tight_layout()
        return fig, ax


# ══════════════════════════════════════════════════════════════════════
#  Empirical quantile helpers
# ══════════════════════════════════════════════════════════════════════


def _quantile_func(x: np.ndarray, probs: np.ndarray) -> np.ndarray:
    xs = np.sort(x)
    cdf = np.arange(1, len(xs) + 1) / len(xs)
    return np.asarray(np.interp(probs, cdf, xs))


# ══════════════════════════════════════════════════════════════════════
#  Quantile DID
# ══════════════════════════════════════════════════════════════════════


def qdid(
    data: pd.DataFrame,
    y: str,
    group: str,
    time: str,
    quantiles: Optional[List[float]] = None,
    n_boot: int = 500,
    alpha: float = 0.05,
    seed: int = 42,
    method: str = "qdid",
) -> Any:
    """Quantile Difference-in-Differences (QDiD) and its alternatives.

    .. warning::

        **This is not Changes-in-Changes.** Versions <= 1.20.0 described and
        labelled this function as Athey & Imbens (2006) CiC. It is not:
        Athey & Imbens propose CiC *instead of* QDiD and criticise QDiD
        directly, because differencing quantiles presumes the untreated
        outcome distribution shifts by the same amount at every rank. R's
        ``qte`` package keeps ``QDiD()`` and ``CiC()`` separate for the same
        reason. The numbers never changed -- only the attribution. Use
        ``method='cic'`` for changes-in-changes.

    ``method='cic'`` delegates to :func:`statspai.cic` and returns its
    ``CausalResult``; ``qte::MDiD`` and ``ddid2`` are not implemented.

    QTE_DID(τ) = F_{11}^{-1}(τ) - F_{10}^{-1}(τ)
                - [F_{01}^{-1}(τ) - F_{00}^{-1}(τ)]

    Parameters
    ----------
    data : DataFrame
    y : str
        Outcome variable.
    group : str
        Binary group indicator (0 = control, 1 = treated).
    time : str
        Binary time indicator (0 = pre, 1 = post).
    quantiles : list of float, optional
        Defaults to ``[0.1, 0.25, 0.5, 0.75, 0.9]``.
    n_boot : int
        Bootstrap replications.
    alpha : float
        Significance level.
    seed : int
        Random seed.

    Returns
    -------
    QTEResult

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(42)
    >>> n = 400
    >>> g = rng.integers(0, 2, n)
    >>> t = rng.integers(0, 2, n)
    >>> y = (1.0 + 0.5 * g + 0.3 * t + 2.0 * g * t
    ...      + rng.normal(0, 1, n))
    >>> df = pd.DataFrame({"y": y, "g": g, "t": t})
    >>> res = sp.qdid(df, y="y", group="g", time="t",
    ...               quantiles=[0.25, 0.5, 0.75], n_boot=50)
    >>> round(res.ate, 2)  # true effect = 2.0
    2.14
    >>> np.round(res.effects, 2)  # QTE at each quantile
    array([1.96, 1.94, 2.26])
    """
    if method not in ("qdid", "cic"):
        raise ValueError(
            f"method must be 'qdid' or 'cic', got {method!r}. "
            "qte::MDiD / ddid2 are not implemented yet."
        )
    if method == "cic":
        # Delegate rather than keep a second changes-in-changes
        # implementation (CLAUDE.md §12). sp.cic takes the same signature.
        from ..did import cic as _cic

        return _cic(
            data,
            y=y,
            group=group,
            time=time,
            quantiles=quantiles,
            n_boot=n_boot,
            alpha=alpha,
            seed=seed,
        )

    if quantiles is None:
        quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
    taus = np.asarray(quantiles)

    df = data[[y, group, time]].dropna()
    gv = df[group].astype(int).values
    tv = df[time].astype(int).values
    yv = df[y].values.astype(float)

    y00 = yv[(gv == 0) & (tv == 0)]
    y01 = yv[(gv == 0) & (tv == 1)]
    y10 = yv[(gv == 1) & (tv == 0)]
    y11 = yv[(gv == 1) & (tv == 1)]

    for label, arr in [
        ("control-pre", y00),
        ("control-post", y01),
        ("treated-pre", y10),
        ("treated-post", y11),
    ]:
        if len(arr) < 2:
            raise ValueError(f"Too few observations in {label} cell ({len(arr)}).")

    def _point(
        y00_: np.ndarray,
        y01_: np.ndarray,
        y10_: np.ndarray,
        y11_: np.ndarray,
        taus_: np.ndarray,
    ) -> np.ndarray:
        q00 = _quantile_func(y00_, taus_)
        q01 = _quantile_func(y01_, taus_)
        q10 = _quantile_func(y10_, taus_)
        q11 = _quantile_func(y11_, taus_)
        return np.asarray(q11 - q10 - (q01 - q00))

    qte_point = _point(y00, y01, y10, y11, taus)
    ate = float(np.mean(y11) - np.mean(y10) - (np.mean(y01) - np.mean(y00)))

    # Bootstrap
    rng = np.random.RandomState(seed)
    idx00 = np.where((gv == 0) & (tv == 0))[0]
    idx01 = np.where((gv == 0) & (tv == 1))[0]
    idx10 = np.where((gv == 1) & (tv == 0))[0]
    idx11 = np.where((gv == 1) & (tv == 1))[0]

    boot = np.empty((n_boot, len(taus)))
    for b in range(n_boot):
        b00 = yv[rng.choice(idx00, len(idx00), replace=True)]
        b01 = yv[rng.choice(idx01, len(idx01), replace=True)]
        b10 = yv[rng.choice(idx10, len(idx10), replace=True)]
        b11 = yv[rng.choice(idx11, len(idx11), replace=True)]
        boot[b] = _point(b00, b01, b10, b11, taus)

    se = np.std(boot, axis=0, ddof=1)
    ci_lo = np.percentile(boot, 100 * alpha / 2, axis=0)
    ci_hi = np.percentile(boot, 100 * (1 - alpha / 2), axis=0)

    return QTEResult(
        quantiles=taus,
        effects=qte_point,
        se=se,
        ci_lower=ci_lo,
        ci_upper=ci_hi,
        ate=ate,
        method="Quantile DiD (QDiD)",
        n_obs=len(df),
        alpha=alpha,
        model_info={"n_boot": n_boot},
    )


# ══════════════════════════════════════════════════════════════════════
#  QTE via Quantile Regression (Firpo 2007)
# ══════════════════════════════════════════════════════════════════════


def _qreg_coef(
    y: np.ndarray, X: np.ndarray, tau: float, max_iter: int = 500, tol: float = 1e-6
) -> np.ndarray:
    """Interior-point quantile regression via iteratively reweighted LS.

    Minimises  sum rho_tau(y - X beta)  where rho_tau(u) = u*(tau - I(u<0)).
    Uses the IRLS algorithm of Koenker & d'Orey (1987).
    """
    n, k = X.shape
    # OLS start
    beta = np.linalg.lstsq(X, y, rcond=None)[0]

    for _ in range(max_iter):
        resid = y - X @ beta
        # Weights: avoid division by zero
        w = np.where(resid > 0, tau, 1 - tau) / np.maximum(np.abs(resid), 1e-8)
        W = np.diag(w)
        try:
            beta_new = np.linalg.solve(X.T @ W @ X, X.T @ W @ y)
        except np.linalg.LinAlgError:
            break
        if np.max(np.abs(beta_new - beta)) < tol:
            beta = beta_new
            break
        beta = beta_new
    return np.asarray(beta)


_QTE_METHODS = ("firpo_qte", "firpo_qtt", "conditional_qr", "distribution")


def qte(
    data: pd.DataFrame,
    y: str,
    treatment: str,
    quantiles: Optional[List[float]] = None,
    method: str = "firpo_qte",
    controls: Optional[List[str]] = None,
    n_boot: int = 500,
    alpha: float = 0.05,
    seed: int = 42,
    se: str = "auto",
) -> QTEResult:
    """Quantile treatment effect estimation.

    Parameters
    ----------
    data : DataFrame
    y : str
        Outcome variable.
    treatment : str
        Binary treatment indicator.
    quantiles : list of float, optional
        Defaults to ``[0.1, 0.25, 0.5, 0.75, 0.9]``.
    method : {'firpo_qte', 'firpo_qtt', 'conditional_qr', 'distribution'}
        Which estimand to compute. **The default changed in 1.21.0** from
        the conditional quantile regression to ``'firpo_qte'``; see Notes.

        ``'firpo_qte'``
            Firpo (2007) efficient **unconditional QTE**:
            ``F^-1_{Y(1)}(tau) - F^-1_{Y(0)}(tau)``, propensity-score
            reweighted with ``D/p`` and ``(1-D)/(1-p)``.
        ``'firpo_qtt'``
            Firpo (2007) **QTT**, i.e. the same contrast among the treated,
            reweighting controls by ``p/(1-p)``.
        ``'conditional_qr'``
            Coefficient on ``D`` in a quantile regression of ``Y`` on
            ``D + controls`` (Koenker & Bassett 1978). This is a
            **conditional** QTE and is not Firpo's estimator.
        ``'distribution'``
            IPW counterfactual-distribution estimator. This computes the
            **QTT**, not the QTE.
    controls : list of str, optional
        Covariates. Enter the propensity model for the Firpo/distribution
        methods and the design matrix for ``'conditional_qr'``.
    n_boot : int
        Bootstrap replications.
    alpha : float
        Significance level.
    seed : int
        Random seed.
    se : {'auto', 'analytic', 'bootstrap'}
        Standard-error method for the Firpo estimators. ``'auto'`` uses the
        analytic influence function without covariates and the bootstrap with
        them (the analytic form treats ``p(X)`` as known, which is
        conservative when it is estimated). Ignored by the other methods,
        which are bootstrap-only.

    Returns
    -------
    QTEResult

    Notes
    -----
    .. versionchanged:: 1.21.0
        ``method='quantile_regression'`` was documented and labelled as
        Firpo (2007) but computed the coefficient on ``D`` in a
        *conditional* quantile regression — a different estimand with no
        causal interpretation absent rank invariance. It is renamed
        ``'conditional_qr'`` (the old name still works and emits a
        ``DeprecationWarning``), the Firpo attribution is removed from it,
        and the genuine Firpo estimators are available as ``'firpo_qte'`` /
        ``'firpo_qtt'``. ``method='distribution'`` is unchanged numerically
        but is now correctly labelled QTT rather than QTE.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(42)
    >>> n = 2000
    >>> d = rng.integers(0, 2, n)
    >>> y = 1.0 + 1.5 * d + rng.normal(0, 1, n)
    >>> df = pd.DataFrame({"y": y, "d": d})
    >>> res = sp.qte(df, y="y", treatment="d", quantiles=[0.25, 0.5, 0.75])
    >>> bool(np.all(np.abs(res.effects - 1.5) < 0.25))  # true effect = 1.5
    True

    References
    ----------
    firpo2007efficient, koenker1978regression
    """
    if quantiles is None:
        quantiles = [0.1, 0.25, 0.5, 0.75, 0.9]
    taus = np.asarray(quantiles, dtype=float)
    if np.any((taus <= 0) | (taus >= 1)):
        raise ValueError("quantiles must lie strictly inside (0, 1).")

    if method == "quantile_regression":
        warnings.warn(
            "sp.qte(method='quantile_regression') is deprecated and will be "
            "removed in 1.23.0. It was labelled Firpo (2007) but computes a "
            "CONDITIONAL quantile treatment effect. Use "
            "method='conditional_qr' for the same numbers under the correct "
            "name, or method='firpo_qte' for the actual Firpo estimator. "
            "See MIGRATION.md.",
            DeprecationWarning,
            stacklevel=2,
        )
        method = "conditional_qr"

    cols = [y, treatment] + (controls or [])
    df = data[cols].dropna()
    yv = df[y].values.astype(float)
    dv = df[treatment].astype(int).values
    if not np.all(np.isin(dv, (0, 1))):
        raise ValueError("treatment must be binary (0/1).")

    if method in ("firpo_qte", "firpo_qtt"):
        return _qte_firpo(
            df, yv, dv, taus, y, controls, method, n_boot, alpha, seed, se
        )
    if method == "conditional_qr":
        return _qte_qreg(df, yv, dv, taus, y, treatment, controls, n_boot, alpha, seed)
    if method == "distribution":
        return _qte_distribution(
            df, yv, dv, taus, y, treatment, controls, n_boot, alpha, seed
        )
    raise ValueError(f"Unknown QTE method {method!r}. Use one of {_QTE_METHODS}.")


def _qte_firpo(
    df: pd.DataFrame,
    yv: np.ndarray,
    dv: np.ndarray,
    taus: np.ndarray,
    y_col: str,
    controls: Optional[List[str]],
    method: str,
    n_boot: int,
    alpha: float,
    seed: int,
    se: str,
) -> QTEResult:
    """Firpo (2007) efficient unconditional QTE / QTT."""
    from ._firpo import firpo_influence_se, firpo_quantiles, firpo_weights, logit_pscore

    estimand = "qte" if method == "firpo_qte" else "qtt"
    X = df[controls].values.astype(float) if controls else None
    pscore = logit_pscore(X, dv)

    q1, q0 = firpo_quantiles(yv, dv, pscore, taus, estimand)
    effects = q1 - q0

    w1, w0 = firpo_weights(dv, pscore, estimand)
    ate = float(np.sum(w1 * yv) / np.sum(w1) - np.sum(w0 * yv) / np.sum(w0))

    if se == "auto":
        se = "bootstrap" if controls else "analytic"
    if se not in ("analytic", "bootstrap"):
        raise ValueError(f"se must be 'auto', 'analytic' or 'bootstrap', got {se!r}")

    uni_lo = uni_hi = None
    uni_crit = float("nan")
    influence = None
    if se == "analytic":
        from ._core import uniform_band
        from ._firpo import firpo_influence_matrix

        se_arr = firpo_influence_se(yv, dv, pscore, taus, q1, q0, estimand)
        z = stats.norm.ppf(1 - alpha / 2)
        ci_lo, ci_hi = effects - z * se_arr, effects + z * se_arr
        # Simultaneous band over the whole quantile grid (WP-7). Only the
        # analytic path can supply it: it needs influence functions.
        influence = firpo_influence_matrix(yv, dv, pscore, taus, q1, q0, estimand)
        if np.isfinite(influence).all() and len(taus) > 1:
            uni_lo, uni_hi, _, uni_crit = uniform_band(
                effects, influence, alpha=alpha, seed=seed
            )
    else:
        rng = np.random.RandomState(seed)
        n = len(yv)
        boot = np.full((n_boot, len(taus)), np.nan)
        for b in range(n_boot):
            idx = rng.choice(n, n, replace=True)
            db = dv[idx]
            if db.min() == db.max():
                continue
            Xb = X[idx] if X is not None else None
            try:
                pb = logit_pscore(Xb, db)
                qb1, qb0 = firpo_quantiles(yv[idx], db, pb, taus, estimand)
                boot[b] = qb1 - qb0
            except ValueError:
                continue
        n_ok = np.isfinite(boot).sum(axis=0)
        if (n_ok < 2).any():
            warnings.warn(
                f"sp.qte({method}): bootstrap collapsed for "
                f"{int((n_ok < 2).sum())}/{len(taus)} quantile(s); those SEs "
                "are NaN.",
                RuntimeWarning,
                stacklevel=2,
            )
        se_arr = np.where(n_ok >= 2, np.nanstd(boot, axis=0, ddof=1), np.nan)
        ci_lo = np.nanpercentile(boot, 100 * alpha / 2, axis=0)
        ci_hi = np.nanpercentile(boot, 100 * (1 - alpha / 2), axis=0)

    label = (
        "Unconditional QTE (Firpo, 2007)"
        if estimand == "qte"
        else "QTT on the treated (Firpo, 2007)"
    )
    return QTEResult(
        quantiles=taus,
        effects=effects,
        se=se_arr,
        ci_lower=ci_lo,
        ci_upper=ci_hi,
        ate=ate,
        method=label,
        n_obs=len(df),
        alpha=alpha,
        model_info={
            "controls": controls,
            "estimand": estimand,
            "se_method": se,
            "n_boot": n_boot if se == "bootstrap" else None,
            "pscore_min": float(pscore.min()),
            "pscore_max": float(pscore.max()),
            "influence": influence,
        },
        ci_lower_uniform=uni_lo,
        ci_upper_uniform=uni_hi,
        uniform_crit=uni_crit,
    )


def _qte_qreg(
    df: pd.DataFrame,
    yv: np.ndarray,
    dv: np.ndarray,
    taus: np.ndarray,
    y_col: str,
    treat_col: str,
    controls: Optional[List[str]],
    n_boot: int,
    alpha: float,
    seed: int,
) -> QTEResult:
    """QTE via quantile regression."""
    # Build design matrix: [intercept, treatment, controls...]
    X_cols = [treat_col] + (controls or [])
    X = np.column_stack([np.ones(len(yv)), df[X_cols].values.astype(float)])
    treat_idx = 1  # treatment is the second column

    # Point estimates
    qte_point = np.empty(len(taus))
    for i, tau in enumerate(taus):
        beta = _qreg_coef(yv, X, tau)
        qte_point[i] = beta[treat_idx]

    ate = float(np.mean(yv[dv == 1]) - np.mean(yv[dv == 0]))

    # Bootstrap
    rng = np.random.RandomState(seed)
    boot = np.empty((n_boot, len(taus)))
    n = len(yv)
    for b in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        yb = yv[idx]
        Xb = X[idx]
        for i, tau in enumerate(taus):
            beta = _qreg_coef(yb, Xb, tau)
            boot[b, i] = beta[treat_idx]

    se = np.std(boot, axis=0, ddof=1)
    ci_lo = np.percentile(boot, 100 * alpha / 2, axis=0)
    ci_hi = np.percentile(boot, 100 * (1 - alpha / 2), axis=0)

    return QTEResult(
        quantiles=taus,
        effects=qte_point,
        se=se,
        ci_lower=ci_lo,
        ci_upper=ci_hi,
        ate=ate,
        method="Conditional QTE via Quantile Regression (Koenker & Bassett, 1978)",
        n_obs=len(df),
        alpha=alpha,
        model_info={"n_boot": n_boot, "controls": controls},
    )


def _qte_distribution(
    df: pd.DataFrame,
    yv: np.ndarray,
    dv: np.ndarray,
    taus: np.ndarray,
    y_col: str,
    treat_col: str,
    controls: Optional[List[str]],
    n_boot: int,
    alpha: float,
    seed: int,
) -> QTEResult:
    """QTE via propensity-score reweighting (distribution method)."""
    # Estimate propensity score with logistic regression
    if controls:
        Xc = df[controls].values.astype(float)
        Xp = np.column_stack([np.ones(len(yv)), Xc])
    else:
        Xp = np.ones((len(yv), 1))

    # Simple logistic via scipy
    from scipy.optimize import minimize

    def _loglik(beta: np.ndarray) -> float:
        z = Xp @ beta
        z = np.clip(z, -30, 30)
        p = 1 / (1 + np.exp(-z))
        p = np.clip(p, 1e-10, 1 - 1e-10)
        return float(-np.mean(dv * np.log(p) + (1 - dv) * np.log(1 - p)))

    beta0 = np.zeros(Xp.shape[1])
    res = minimize(_loglik, beta0, method="BFGS")
    pscore = 1 / (1 + np.exp(-np.clip(Xp @ res.x, -30, 30)))
    pscore = np.clip(pscore, 0.01, 0.99)

    def _weighted_quantiles(
        y_: np.ndarray,
        d_: np.ndarray,
        ps_: np.ndarray,
        taus_: np.ndarray,
    ) -> np.ndarray:
        """IPW-based quantile estimates for treated and counterfactual."""
        # Treated quantiles (unweighted among treated)
        y1 = y_[d_ == 1]
        q1 = _quantile_func(y1, taus_)

        # Counterfactual quantiles via IPW reweighting of controls
        y0 = y_[d_ == 0]
        w0 = ps_[d_ == 0] / (1 - ps_[d_ == 0])
        # Weighted quantile function
        order = np.argsort(y0)
        y0s = y0[order]
        w0s = w0[order]
        wcdf = np.cumsum(w0s) / np.sum(w0s)
        q0 = np.interp(taus_, wcdf, y0s)
        return np.asarray(q1 - q0)

    qte_point = _weighted_quantiles(yv, dv, pscore, taus)
    ate = float(
        np.mean(yv[dv == 1])
        - np.sum(yv[dv == 0] * pscore[dv == 0] / (1 - pscore[dv == 0]))
        / np.sum(pscore[dv == 0] / (1 - pscore[dv == 0]))
    )

    # Bootstrap
    rng = np.random.RandomState(seed)
    boot = np.empty((n_boot, len(taus)))
    n = len(yv)
    for b in range(n_boot):
        idx = rng.choice(n, n, replace=True)
        yb = yv[idx]
        db = dv[idx]
        Xpb = Xp[idx]

        # Re-estimate propensity score
        def _ll(beta: np.ndarray) -> float:
            z = Xpb @ beta
            z = np.clip(z, -30, 30)
            p = 1 / (1 + np.exp(-z))
            p = np.clip(p, 1e-10, 1 - 1e-10)
            return float(-np.mean(db * np.log(p) + (1 - db) * np.log(1 - p)))

        rb = minimize(_ll, res.x, method="BFGS")
        psb = 1 / (1 + np.exp(-np.clip(Xpb @ rb.x, -30, 30)))
        psb = np.clip(psb, 0.01, 0.99)

        if np.sum(db == 1) < 2 or np.sum(db == 0) < 2:
            boot[b] = np.nan
            continue
        boot[b] = _weighted_quantiles(yb, db, psb, taus)

    se = np.nanstd(boot, axis=0, ddof=1)
    ci_lo = np.nanpercentile(boot, 100 * alpha / 2, axis=0)
    ci_hi = np.nanpercentile(boot, 100 * (1 - alpha / 2), axis=0)

    return QTEResult(
        quantiles=taus,
        effects=qte_point,
        se=se,
        ci_lower=ci_lo,
        ci_upper=ci_hi,
        ate=ate,
        method="QTT via IPW counterfactual distribution",
        n_obs=len(df),
        alpha=alpha,
        model_info={"n_boot": n_boot, "controls": controls},
    )


__all__ = ["qdid", "qte", "QTEResult"]
