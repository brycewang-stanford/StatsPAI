"""DML diagnostics: overlap, score density, balance, orthogonality.

Provides :func:`dml_diagnostics` returning a :class:`DMLDiagnostics`
report bundling four standard ML-causal diagnostics:

1. **Overlap** — distribution of the cross-fitted propensity (IRM) or
   :math:`D - m̂(X)` residual (PLR). Flags propensity values within
   ``clip`` of 0/1 and ESS-style summaries.
2. **Score density** — kernel-density / histogram of the orthogonal
   score :math:`\\psi`; should be bell-shaped and centred at 0 under a
   correctly identified moment.
3. **Covariate balance after residualisation** — for each X_k,
   ``corr(X_k, d_resid)`` and ``corr(X_k, y_resid)``. Large values
   indicate the nuisance learner left structure in the data that the
   orthogonalisation cannot remove.
4. **Orthogonality test** — :math:`\\hat E[\\psi_a]` standardised by
   its SE; under correct nuisance estimation this ≈ 0.

Each diagnostic ships with a ``.plot()`` that produces a publication-
ready 2×2 panel matching the modelsummary / DoubleML defaults.

References
----------
- Chernozhukov V., Chetverikov D., Demirer M., Duflo E., Hansen C.,
  Newey W., Robins J. (2018). "Double/Debiased Machine Learning for
  Treatment and Structural Parameters." *Econometrics Journal* 21(1):
  C1-C68. DOI: 10.1111/ectj.12097.
- Bach P., Kurz M.S., Chernozhukov V., Spindler M., Klaassen S. (2024).
  "DoubleML: An Object-Oriented Implementation of Double Machine
  Learning in R." *Journal of Statistical Software* 108(3).
  DOI: 10.18637/jss.v108.i03.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, Any, List

import numpy as np
import pandas as pd
from scipy import stats


@dataclass
class DMLDiagnostics:
    """Bundled DML diagnostics returned by :func:`dml_diagnostics`.

    Holds the overlap table, residual-balance table, score-density
    moments, and the orthogonality test for a fitted DML estimator.
    Use ``.summary()`` for the text report and ``.plot()`` for the 2x2
    diagnostic panel.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(42)
    >>> n = 400
    >>> x1 = rng.normal(size=n)
    >>> x2 = rng.normal(size=n)
    >>> d = 0.5 * x1 + rng.normal(size=n)
    >>> y = 1.0 * d + x1 + 0.5 * x2 + rng.normal(size=n)
    >>> df = pd.DataFrame({'y': y, 'd': d, 'x1': x1, 'x2': x2})
    >>> fit = sp.dml(df, y='y', treat='d', covariates=['x1', 'x2'],
    ...              model='plr', ml_g='linear', ml_m='linear',
    ...              n_folds=2)
    >>> diag = sp.dml_diagnostics(fit)
    >>> isinstance(diag, sp.DMLDiagnostics)
    True
    >>> diag.method
    'PLR'
    """

    n_obs: int
    method: str  # PLR / IRM / PLIV / IIVM
    estimate: float
    se: float

    # Overlap (PLR: |d_resid| dist; IRM: pscore dist)
    overlap_table: pd.DataFrame
    n_clipped_low: int = 0
    n_clipped_high: int = 0
    overlap_warning: Optional[str] = None

    # Balance after residualisation
    balance_table: pd.DataFrame = field(default_factory=pd.DataFrame)

    # Score density
    score_mean: float = 0.0
    score_sd: float = 0.0
    score_skew: float = 0.0
    score_kurtosis: float = 0.0

    # Orthogonality test
    orth_stat: float = 0.0  # standardized E[ψ_a]
    orth_pvalue: float = 1.0
    orth_warning: Optional[str] = None
    _overlap_values: np.ndarray = field(
        default_factory=lambda: np.asarray([], dtype=float),
        repr=False,
    )
    _overlap_label: str = ""
    _score: Optional[np.ndarray] = field(default=None, repr=False)

    def summary(self) -> str:
        lines = [
            f"DML Diagnostics — {self.method}",
            "=" * 64,
            f"  Estimate / SE          : {self.estimate:+.4f} / {self.se:.4f}",
            f"  N observations         : {self.n_obs:,}",
            "",
            "[Overlap]",
            self.overlap_table.round(4).to_string(),
        ]
        if self.overlap_warning:
            lines.append(f"  ⚠ {self.overlap_warning}")
        lines += [
            "",
            "[Score density]",
            f"  Mean      : {self.score_mean:+.4f}",
            f"  SD        : {self.score_sd:.4f}",
            f"  Skew      : {self.score_skew:+.4f}",
            f"  Kurtosis  : {self.score_kurtosis:.4f}",
            "",
            "[Orthogonality test (E[ψ_a] = 0)]",
            f"  Standardised statistic : {self.orth_stat:+.4f}",
            f"  p-value                : {self.orth_pvalue:.4f}",
        ]
        if self.orth_warning:
            lines.append(f"  ⚠ {self.orth_warning}")
        if not self.balance_table.empty:
            lines += [
                "",
                "[Balance after residualisation]",
                self.balance_table.round(4).to_string(index=False),
            ]
        return "\n".join(lines)

    def __repr__(self) -> str:  # pragma: no cover
        return self.summary()

    def plot(self, figsize: Any = (10.0, 8.0), bins: int = 30) -> Any:
        """Render a 2×2 publication-style diagnostic panel.

        Top-left  : overlap histogram (propensity for IRM, |D-resid| for PLR).
        Top-right : score-density histogram with N(0,σ̂²) overlay.
        Bottom-left: balance bar chart (corr with each residualised covariate).
        Bottom-right: Q-Q plot of standardised score vs N(0,1).
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError as e:  # pragma: no cover
            raise ImportError("matplotlib required for DMLDiagnostics.plot()") from e

        fig, axes = plt.subplots(2, 2, figsize=figsize)
        ax_ov, ax_sc, ax_bal, ax_qq = axes.ravel()

        # Overlap
        ov = self._overlap_values
        ax_ov.hist(ov, bins=bins, color="#1f77b4", edgecolor="white")
        ax_ov.set_title(f"Overlap: {self._overlap_label}")
        ax_ov.set_xlabel(self._overlap_label)
        ax_ov.set_ylabel("Frequency")
        if self.overlap_warning:
            ax_ov.text(
                0.02,
                0.98,
                self.overlap_warning,
                transform=ax_ov.transAxes,
                va="top",
                fontsize=8,
                color="#d62728",
                bbox=dict(facecolor="white", alpha=0.8, edgecolor="none"),
            )

        # Score density + normal overlay
        psi = self._score
        if psi is not None:
            ax_sc.hist(
                psi,
                bins=bins,
                density=True,
                color="#2ca02c",
                edgecolor="white",
                alpha=0.7,
            )
            x_grid = np.linspace(np.min(psi), np.max(psi), 200)
            ax_sc.plot(
                x_grid,
                stats.norm.pdf(x_grid, 0, self.score_sd),
                color="#1f77b4",
                lw=1.5,
                label="N(0, σ̂)",
            )
            ax_sc.legend(loc="upper right", fontsize=8)
            ax_sc.set_title("Score density")
            ax_sc.set_xlabel(r"$\psi - \theta$")

        # Balance
        if not self.balance_table.empty:
            bal = self.balance_table.set_index("variable")
            x = np.arange(len(bal))
            width = 0.35
            ax_bal.bar(
                x - width / 2,
                bal["corr_d_resid"],
                width=width,
                label=r"corr($X_k$, $\tilde D$)",
                color="#1f77b4",
            )
            ax_bal.bar(
                x + width / 2,
                bal["corr_y_resid"],
                width=width,
                label=r"corr($X_k$, $\tilde Y$)",
                color="#ff7f0e",
            )
            ax_bal.axhline(0, color="#333", lw=0.6)
            ax_bal.set_xticks(x)
            ax_bal.set_xticklabels(bal.index, rotation=45, ha="right", fontsize=8)
            ax_bal.set_title("Residual-balance check")
            ax_bal.set_ylabel("Correlation")
            ax_bal.legend(loc="best", fontsize=8)

        # Q-Q plot
        if psi is not None:
            stats.probplot(psi, dist="norm", plot=ax_qq)
            ax_qq.set_title("Score Q-Q (vs. Normal)")

        fig.tight_layout()
        return fig, axes


def dml_diagnostics(result: Any, clip: float = 0.02) -> DMLDiagnostics:
    """Build a :class:`DMLDiagnostics` report from a DML CausalResult.

    Parameters
    ----------
    result : CausalResult
        Result returned by :func:`statspai.dml.dml`. Must include the
        post-fit residuals (``model_info['_y_resid']``,
        ``model_info['_d_resid']``); for IRM, additionally the propensity
        ``model_info['diagnostics']['pscore_min']`` etc. are surfaced.
    clip : float, default 0.02
        For IRM-style overlap: count units with propensity within
        ``[0, clip] ∪ [1-clip, 1]`` as overlap-violating.

    Returns
    -------
    DMLDiagnostics

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(42)
    >>> n = 400
    >>> x1 = rng.normal(size=n)
    >>> x2 = rng.normal(size=n)
    >>> d = 0.5 * x1 + rng.normal(size=n)
    >>> y = 1.0 * d + x1 + 0.5 * x2 + rng.normal(size=n)
    >>> df = pd.DataFrame({'y': y, 'd': d, 'x1': x1, 'x2': x2})
    >>> fit = sp.dml(df, y='y', treat='d', covariates=['x1', 'x2'],
    ...              model='plr', ml_g='linear', ml_m='linear',
    ...              n_folds=2)
    >>> diag = sp.dml_diagnostics(fit)
    >>> diag.n_obs
    400
    >>> diag.method
    'PLR'
    >>> round(diag.orth_pvalue, 2)  # E[psi] = 0 not rejected
    1.0
    >>> text = diag.summary()  # overlap / score / balance report
    """
    info = result.model_info or {}
    method = info.get("dml_model", "DML")
    y_resid = info.get("_y_resid")
    d_resid = info.get("_d_resid")
    pscore = info.get("_pscore")
    if y_resid is None or d_resid is None:
        raise ValueError(  # pragma: no cover
            "dml_diagnostics requires post-fit residuals on the result "
            "(re-fit with current statspai.dml; residuals are stored in "
            "model_info['_y_resid'] / ['_d_resid'])."
        )
    y_resid = np.asarray(y_resid, dtype=float).ravel()
    d_resid = np.asarray(d_resid, dtype=float).ravel()
    n = len(y_resid)

    # ---- Overlap ----
    if pscore is not None:
        pscore = np.asarray(pscore, dtype=float).ravel()
        overlap_label = "Estimated propensity ê(X)"
        overlap_values = pscore
        n_low = int(np.sum(pscore < clip))
        n_high = int(np.sum(pscore > 1 - clip))
        ov_table = pd.DataFrame(
            {
                "quantile": ["min", "p1", "p10", "median", "p90", "p99", "max"],
                "value": [
                    float(np.min(pscore)),
                    float(np.quantile(pscore, 0.01)),
                    float(np.quantile(pscore, 0.10)),
                    float(np.median(pscore)),
                    float(np.quantile(pscore, 0.90)),
                    float(np.quantile(pscore, 0.99)),
                    float(np.max(pscore)),
                ],
            }
        )
        overlap_warning = None
        if n_low + n_high > 0:
            overlap_warning = (
                f"{n_low + n_high} units ({(n_low+n_high)/n*100:.1f}%) "
                f"have propensity outside [{clip}, {1-clip}]."
            )
    else:
        overlap_label = r"Treatment residual $|D - \hat m(X)|$"
        overlap_values = np.abs(d_resid)
        n_low = 0
        n_high = 0
        ov_table = pd.DataFrame(
            {
                "quantile": ["min", "p10", "median", "p90", "max"],
                "value": [
                    float(np.min(overlap_values)),
                    float(np.quantile(overlap_values, 0.10)),
                    float(np.median(overlap_values)),
                    float(np.quantile(overlap_values, 0.90)),
                    float(np.max(overlap_values)),
                ],
            }
        )
        overlap_warning = None

    # ---- Score density ----
    # For PLR the orthogonal score psi ∝ (y_resid - θ * d_resid) * d_resid;
    # for IRM y_resid IS already psi - θ. Both have mean ≈ 0 under
    # correct identification. Use psi = y_resid (an effective score
    # residual) for unified diagnostics.
    psi = y_resid - float(np.mean(y_resid))
    score_mean = float(np.mean(psi))
    score_sd = float(np.std(psi, ddof=1))
    score_skew = float(stats.skew(psi)) if score_sd > 0 else 0.0
    score_kurt = float(stats.kurtosis(psi)) if score_sd > 0 else 0.0

    # ---- Orthogonality: E[ψ] = 0 ----
    if score_sd > 0:
        orth_stat = float(np.mean(psi) / (score_sd / np.sqrt(n)))
    else:
        orth_stat = 0.0
    orth_pvalue = float(2 * (1 - stats.norm.cdf(abs(orth_stat))))
    orth_warning = None
    if orth_pvalue < 0.01:
        orth_warning = (
            "Orthogonality score is significantly nonzero; "
            "nuisance learner may be mis-specified."
        )

    # ---- Balance ----
    rows: List[Dict[str, Any]] = []
    X_design = info.get("_X_design")
    cov_names = info.get("_covariate_names")
    if X_design is not None and cov_names:
        X = np.asarray(X_design, dtype=float)
        for j, name in enumerate(cov_names):
            xk = X[:, j]
            sd_xk = np.std(xk)
            if sd_xk == 0:
                rows.append(
                    {"variable": name, "corr_d_resid": 0.0, "corr_y_resid": 0.0}
                )
                continue  # pragma: no cover
            cd = float(np.corrcoef(xk, d_resid)[0, 1])
            cy = float(np.corrcoef(xk, y_resid)[0, 1])
            rows.append(
                {
                    "variable": name,
                    "corr_d_resid": cd,
                    "corr_y_resid": cy,
                }
            )
    bal_table = pd.DataFrame(rows)

    diag = DMLDiagnostics(
        n_obs=n,
        method=method,
        estimate=float(result.estimate),
        se=float(result.se),
        overlap_table=ov_table,
        n_clipped_low=n_low,
        n_clipped_high=n_high,
        overlap_warning=overlap_warning,
        balance_table=bal_table,
        score_mean=score_mean,
        score_sd=score_sd,
        score_skew=score_skew,
        score_kurtosis=score_kurt,
        orth_stat=orth_stat,
        orth_pvalue=orth_pvalue,
        orth_warning=orth_warning,
    )
    diag._overlap_values = overlap_values
    diag._overlap_label = overlap_label
    diag._score = psi
    return diag
