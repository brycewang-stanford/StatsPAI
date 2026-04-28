"""
Changes-in-Changes (CIC) estimator — Athey & Imbens (2006).

CIC relaxes the standard DID parallel-trends assumption to the weaker
"rank invariance" condition.  Instead of assuming additive group effects,
it uses the full outcome distributions to construct the counterfactual.

Algorithm (continuous case)
---------------------------
1. Estimate empirical CDFs for each (group, time) cell:
   F_{00}, F_{01}, F_{10}, F_{11}.
2. Counterfactual CDF for treated-post absent treatment:
   F_{Y^N,11}(y) = F_{10}( F_{00}^{-1}( F_{01}(y) ) )
3. ATT  = mean(Y_{11}) - integral of F_{Y^N,11}^{-1}
4. QTE(τ) = F_{11}^{-1}(τ) - F_{Y^N,11}^{-1}(τ)

Reference
---------
Athey, S. & Imbens, G. W. (2006).
Identification and Inference in Nonlinear Difference-in-Differences Models.
*Econometrica*, 74(2), 431-497.
"""

from __future__ import annotations

from typing import Optional, List, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from ..core.results import CausalResult


# ── Helpers ───────────────────────────────────────────────────────────

def _ecdf_values(x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Return sorted unique values and their empirical CDF."""
    xs = np.sort(x)
    cdf = np.arange(1, len(xs) + 1) / len(xs)
    return xs, cdf


def _ecdf(x: np.ndarray, grid: np.ndarray) -> np.ndarray:
    """Evaluate empirical CDF of *x* on *grid* via linear interpolation."""
    xs, cdf = _ecdf_values(x)
    return np.interp(grid, xs, cdf, left=0.0, right=1.0)


def _quantile_func(x: np.ndarray, probs: np.ndarray) -> np.ndarray:
    """Evaluate empirical quantile function (inverse CDF) at *probs*."""
    xs, cdf = _ecdf_values(x)
    # Ensure monotonicity for interp (CDF is already non-decreasing)
    return np.interp(probs, cdf, xs)


def _counterfactual_quantiles(
    y00: np.ndarray,
    y01: np.ndarray,
    y10: np.ndarray,
    grid: np.ndarray,
) -> np.ndarray:
    """Counterfactual quantile function F_{Y^N,11}^{-1}(τ).

    For each quantile τ in *grid*:
        F_{Y^N,11}^{-1}(τ) = F_{10}^{-1}( F_{00}( F_{01}^{-1}(τ) ) )

    Equivalently we build the counterfactual CDF and invert it,
    but working in quantile space is numerically more stable.
    """
    # Step 1: F_{01}^{-1}(τ) — map quantile to control-post outcome
    y_ctrl_post = _quantile_func(y01, grid)
    # Step 2: F_{00}(y) — evaluate control-pre CDF at those outcomes
    prob_ctrl_pre = _ecdf(y00, y_ctrl_post)
    # Step 3: F_{10}^{-1}(p) — map through treated-pre quantile function
    y_cf = _quantile_func(y10, prob_ctrl_pre)
    return y_cf


# ── Main estimator ────────────────────────────────────────────────────

def cic(
    data: pd.DataFrame,
    y: str,
    group: str,
    time: str,
    quantiles: Optional[List[float]] = None,
    n_boot: int = 500,
    alpha: float = 0.05,
    seed: int = 42,
    n_grid: int = 200,
) -> CausalResult:
    """Changes-in-Changes estimator (Athey & Imbens 2006).

    Parameters
    ----------
    data : DataFrame
        Panel or repeated cross-section.
    y : str
        Outcome variable.
    group : str
        Binary group indicator (0 = control, 1 = treated).
    time : str
        Binary time indicator (0 = pre, 1 = post).
    quantiles : list of float, optional
        Quantiles at which to compute QTE.  ``None`` → ATT only.
    n_boot : int
        Number of bootstrap replications for SEs / CIs.
    alpha : float
        Significance level.
    seed : int
        Random seed for reproducibility.
    n_grid : int
        Resolution of the quantile grid used internally.

    Returns
    -------
    CausalResult
    """
    df = data[[y, group, time]].dropna().copy()
    g = df[group].astype(int).values
    t = df[time].astype(int).values
    yv = df[y].values.astype(float)

    # Split into four cells
    y00 = yv[(g == 0) & (t == 0)]
    y01 = yv[(g == 0) & (t == 1)]
    y10 = yv[(g == 1) & (t == 0)]
    y11 = yv[(g == 1) & (t == 1)]

    for label, arr in [("control-pre", y00), ("control-post", y01),
                       ("treated-pre", y10), ("treated-post", y11)]:
        if len(arr) < 2:
            raise ValueError(
                f"Too few observations in the {label} cell ({len(arr)}). "
                "CIC requires data in all four (group × time) cells."
            )

    # Quantile grid
    tau_grid = np.linspace(1 / n_grid, 1 - 1 / n_grid, n_grid)

    # ── Point estimates ───────────────────────────────────────────── #
    cf_q = _counterfactual_quantiles(y00, y01, y10, tau_grid)
    att_point = np.mean(y11) - np.mean(cf_q)

    qte_taus = np.asarray(quantiles) if quantiles is not None else None
    qte_point = None
    if qte_taus is not None:
        obs_q11 = _quantile_func(y11, qte_taus)
        cf_q_at_tau = _counterfactual_quantiles(y00, y01, y10, qte_taus)
        qte_point = obs_q11 - cf_q_at_tau

    # ── Bootstrap ─────────────────────────────────────────────────── #
    rng = np.random.RandomState(seed)
    boot_att = np.empty(n_boot)
    boot_qte = np.empty((n_boot, len(qte_taus))) if qte_taus is not None else None

    idx00 = np.where((g == 0) & (t == 0))[0]
    idx01 = np.where((g == 0) & (t == 1))[0]
    idx10 = np.where((g == 1) & (t == 0))[0]
    idx11 = np.where((g == 1) & (t == 1))[0]

    for b in range(n_boot):
        b00 = yv[rng.choice(idx00, len(idx00), replace=True)]
        b01 = yv[rng.choice(idx01, len(idx01), replace=True)]
        b10 = yv[rng.choice(idx10, len(idx10), replace=True)]
        b11 = yv[rng.choice(idx11, len(idx11), replace=True)]

        bcf = _counterfactual_quantiles(b00, b01, b10, tau_grid)
        boot_att[b] = np.mean(b11) - np.mean(bcf)

        if qte_taus is not None:
            bq11 = _quantile_func(b11, qte_taus)
            bcf_tau = _counterfactual_quantiles(b00, b01, b10, qte_taus)
            boot_qte[b] = bq11 - bcf_tau

    att_se = np.std(boot_att, ddof=1)
    att_ci = (
        np.percentile(boot_att, 100 * alpha / 2),
        np.percentile(boot_att, 100 * (1 - alpha / 2)),
    )
    att_z = att_point / att_se if att_se > 0 else np.nan
    att_pvalue = float(2 * (1 - stats.norm.cdf(np.abs(att_z))))

    # ── Build detail DataFrame ────────────────────────────────────── #
    detail = None
    model_info: dict = {
        "n_control_pre": len(y00),
        "n_control_post": len(y01),
        "n_treated_pre": len(y10),
        "n_treated_post": len(y11),
        "n_boot": n_boot,
    }

    if qte_taus is not None and boot_qte is not None:
        qte_se = np.std(boot_qte, axis=0, ddof=1)
        qte_ci_lo = np.percentile(boot_qte, 100 * alpha / 2, axis=0)
        qte_ci_hi = np.percentile(boot_qte, 100 * (1 - alpha / 2), axis=0)
        qte_z = np.where(qte_se > 0, qte_point / qte_se, np.nan)
        qte_pv = 2 * (1 - stats.norm.cdf(np.abs(qte_z)))

        detail = pd.DataFrame({
            "quantile": qte_taus,
            "qte": qte_point,
            "se": qte_se,
            "ci_lower": qte_ci_lo,
            "ci_upper": qte_ci_hi,
            "pvalue": qte_pv,
        })
        model_info["qte"] = detail

    n_obs = len(df)

    result = CausalResult(
        method="Changes-in-Changes (Athey & Imbens, 2006)",
        estimand="ATT",
        estimate=float(att_point),
        se=float(att_se),
        pvalue=float(att_pvalue),
        ci=att_ci,
        alpha=alpha,
        n_obs=n_obs,
        detail=detail,
        model_info=model_info,
        _citation_key="cic",
    )

    # Attach plot helper
    result._cic_plot_data = {
        "y11": y11,
        "cf_quantiles": cf_q,
        "tau_grid": tau_grid,
        "qte_taus": qte_taus,
        "qte_point": qte_point,
        "qte_se": np.std(boot_qte, axis=0, ddof=1) if boot_qte is not None else None,
        "alpha": alpha,
    }
    _orig_plot = result.plot if hasattr(result, 'plot') else None

    def _cic_plot(ax=None):
        """CIC-specific plot: QTE plot if quantiles given, else CDF comparison."""
        import matplotlib.pyplot as plt

        pd = result._cic_plot_data
        has_qte = pd["qte_taus"] is not None

        if ax is None:
            fig, ax = plt.subplots(figsize=(8, 5))
        else:
            fig = ax.get_figure()

        if has_qte:
            taus = pd["qte_taus"]
            qte = pd["qte_point"]
            se = pd["qte_se"]
            z = stats.norm.ppf(1 - pd["alpha"] / 2)
            ax.plot(taus, qte, "o-", color="#2c7bb6", linewidth=2, label="QTE")
            ax.fill_between(
                taus, qte - z * se, qte + z * se,
                alpha=0.2, color="#2c7bb6",
            )
            ax.axhline(0, color="grey", linestyle="--", linewidth=0.8)
            ax.axhline(
                result.estimate, color="#d7191c", linestyle=":",
                linewidth=1.2, label=f"ATT = {result.estimate:.4f}",
            )
            ax.set_xlabel("Quantile (τ)")
            ax.set_ylabel("Treatment Effect")
            ax.set_title("Changes-in-Changes: Quantile Treatment Effects")
            ax.legend()
        else:
            tau = pd["tau_grid"]
            obs_q = _quantile_func(pd["y11"], tau)
            cf_q_vals = pd["cf_quantiles"]
            ax.plot(tau, obs_q, color="#2c7bb6", linewidth=2, label="Observed (treated-post)")
            ax.plot(tau, cf_q_vals, color="#d7191c", linewidth=2,
                    linestyle="--", label="Counterfactual")
            ax.set_xlabel("Quantile (τ)")
            ax.set_ylabel("Outcome")
            ax.set_title("Changes-in-Changes: Observed vs Counterfactual")
            ax.legend()

        fig.tight_layout()
        return fig, ax

    result.plot = _cic_plot

    # Custom summary for CIC
    _orig_summary = result.summary

    def _cic_summary(alpha_override=None):
        a = alpha_override or alpha
        lines = []
        lines.append("━" * 60)
        lines.append("  Changes-in-Changes (Athey & Imbens, 2006)")
        lines.append("━" * 60)
        stars = CausalResult._stars(att_pvalue)
        lines.append(f"  ATT:          {att_point:.4f}{stars}")
        lines.append(f"  Bootstrap SE: {att_se:.4f}")
        pct = int(100 * (1 - a))
        lines.append(f"  {pct}% CI:      [{att_ci[0]:.4f}, {att_ci[1]:.4f}]")
        lines.append("")

        if detail is not None:
            lines.append("  Quantile Treatment Effects:")
            for _, row in detail.iterrows():
                s = CausalResult._stars(row["pvalue"])
                lines.append(
                    f"    τ = {row['quantile']:.2f}:   "
                    f"{row['qte']:.4f}  ({row['se']:.4f}) {s}"
                )
        lines.append("━" * 60)
        lines.append(f"  Observations: {n_obs:,}")
        lines.append(f"  Bootstrap replications: {n_boot}")
        lines.append("━" * 60)
        out = "\n".join(lines)
        print(out)
        return out

    result.summary = _cic_summary
    try:
        from ..output._lineage import attach_provenance as _attach_prov
        _attach_prov(
            result,
            function="sp.did.cic",
            params={
                "y": y, "group": group, "time": time,
                "quantiles": list(quantiles) if quantiles else None,
                "n_boot": n_boot, "alpha": alpha,
                "seed": seed, "n_grid": n_grid,
            },
            data=data,
            overwrite=False,
        )
    except Exception:  # pragma: no cover
        pass
    return result


__all__ = ["cic"]
