"""
RD diagnostic and validation tools.

Provides bandwidth sensitivity analysis, covariate balance tests,
and placebo cutoff tests for regression discontinuity designs.

References
----------
Imbens, G.W. and Lemieux, T. (2008).
"Regression Discontinuity Designs: A Guide to Practice."
*Journal of Econometrics*, 142(2), 615-635.

Cattaneo, M.D., Idrobo, N. and Titiunik, R. (2020).
"A Practical Introduction to Regression Discontinuity Designs."
*Cambridge Elements*.
"""

from typing import Optional, List, Dict, Any

import numpy as np
import pandas as pd
from scipy import stats

from ..core.results import CausalResult


# ======================================================================
# Bandwidth sensitivity
# ======================================================================

def rdbwsensitivity(
    data: pd.DataFrame,
    y: str,
    x: str,
    c: float = 0,
    fuzzy: Optional[str] = None,
    p: int = 1,
    kernel: str = 'triangular',
    bw_grid: Optional[List[float]] = None,
    n_grid: int = 15,
    bw_range: tuple = (0.5, 2.0),
    alpha: float = 0.05,
    ax=None,
    figsize: tuple = (10, 6),
) -> pd.DataFrame:
    """
    Bandwidth sensitivity analysis for RD estimates.

    Re-estimates the RD effect across a grid of bandwidths to assess
    robustness. Plots point estimates with confidence intervals.

    Parameters
    ----------
    data : pd.DataFrame
    y, x : str
        Outcome and running variable names.
    c : float, default 0
        Cutoff.
    fuzzy : str, optional
        Treatment variable for fuzzy RD.
    p : int, default 1
        Polynomial order.
    kernel : str, default 'triangular'
    bw_grid : list of float, optional
        Explicit bandwidth values to evaluate. If None, auto-generates
        a grid as multiples of the MSE-optimal bandwidth.
    n_grid : int, default 15
        Number of grid points if bw_grid is None.
    bw_range : tuple, default (0.5, 2.0)
        Range of multipliers for the optimal bandwidth.
    alpha : float, default 0.05
    ax : matplotlib Axes, optional
    figsize : tuple

    Returns
    -------
    pd.DataFrame
        Columns: bandwidth, estimate, se, ci_lower, ci_upper, pvalue.
    """
    from .rdrobust import rdrobust

    # Get optimal bandwidth
    base = rdrobust(data, y=y, x=x, c=c, fuzzy=fuzzy, p=p, kernel=kernel,
                    alpha=alpha)
    h_opt = base.model_info['bandwidth_h']

    if bw_grid is None:
        lo, hi = bw_range
        bw_grid = list(np.linspace(lo * h_opt, hi * h_opt, n_grid))

    rows = []
    for bw in bw_grid:
        try:
            r = rdrobust(data, y=y, x=x, c=c, fuzzy=fuzzy, p=p,
                         kernel=kernel, h=bw, alpha=alpha)
            rows.append({
                'bandwidth': bw,
                'estimate': r.estimate,
                'se': r.se,
                'ci_lower': r.ci[0],
                'ci_upper': r.ci[1],
                'pvalue': r.pvalue,
            })
        except (ValueError, np.linalg.LinAlgError):
            continue

    result = pd.DataFrame(rows)

    # Plot if matplotlib available
    try:
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()

        ax.errorbar(result['bandwidth'], result['estimate'],
                    yerr=[result['estimate'] - result['ci_lower'],
                          result['ci_upper'] - result['estimate']],
                    fmt='o-', color='#2C3E50', markersize=4, linewidth=1,
                    capsize=3, alpha=0.8)
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8)
        ax.axvline(x=h_opt, color='#E74C3C', linestyle='--', linewidth=0.8,
                   alpha=0.7, label=f'Optimal h = {h_opt:.3f}')
        ax.set_xlabel('Bandwidth', fontsize=11)
        ax.set_ylabel('RD Estimate', fontsize=11)
        ax.set_title('Bandwidth Sensitivity', fontsize=13)
        ax.legend(fontsize=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        fig.tight_layout()
    except ImportError:
        pass

    return result


# ======================================================================
# Covariate balance test
# ======================================================================

def rdbalance(
    data: pd.DataFrame,
    x: str,
    c: float = 0,
    covs: Optional[List[str]] = None,
    p: int = 1,
    kernel: str = 'triangular',
    h: Optional[float] = None,
    alpha: float = 0.05,
) -> pd.DataFrame:
    """
    Covariate balance test at the RD cutoff.

    For each covariate, estimates the discontinuity at the cutoff using
    local polynomial regression. Pre-treatment covariates should show
    no significant jump at the cutoff if the RD design is valid.

    Parameters
    ----------
    data : pd.DataFrame
    x : str
        Running variable name.
    c : float, default 0
        Cutoff.
    covs : list of str, optional
        Covariate names to test. If None, tests all numeric columns
        except x.
    p : int, default 1
    kernel : str, default 'triangular'
    h : float, optional
        Manual bandwidth. If None, uses MSE-optimal per covariate.
    alpha : float, default 0.05

    Returns
    -------
    pd.DataFrame
        Columns: covariate, estimate, se, z, pvalue, significant.
    """
    from .rdrobust import rdrobust

    if covs is None:
        covs = [col for col in data.select_dtypes(include=[np.number]).columns
                if col != x]

    rows = []
    for cov in covs:
        try:
            r = rdrobust(data, y=cov, x=x, c=c, p=p, kernel=kernel,
                         h=h, alpha=alpha)
            rows.append({
                'covariate': cov,
                'estimate': r.estimate,
                'se': r.se,
                'z': r.estimate / r.se if r.se > 0 else 0,
                'pvalue': r.pvalue,
                'significant': r.pvalue < alpha,
            })
        except (ValueError, np.linalg.LinAlgError):
            rows.append({
                'covariate': cov,
                'estimate': np.nan,
                'se': np.nan,
                'z': np.nan,
                'pvalue': np.nan,
                'significant': np.nan,
            })

    return pd.DataFrame(rows)


# ======================================================================
# Placebo cutoff test
# ======================================================================

def rdplacebo(
    data: pd.DataFrame,
    y: str,
    x: str,
    c: float = 0,
    placebo_cutoffs: Optional[List[float]] = None,
    n_placebo: int = 10,
    side: str = 'both',
    fuzzy: Optional[str] = None,
    p: int = 1,
    kernel: str = 'triangular',
    alpha: float = 0.05,
    ax=None,
    figsize: tuple = (10, 6),
) -> pd.DataFrame:
    """
    Placebo cutoff test for RD validity.

    Estimates the RD effect at fake cutoff points where no treatment
    effect should exist. Significant effects at placebo cutoffs suggest
    the RD design may be invalid.

    Parameters
    ----------
    data : pd.DataFrame
    y, x : str
        Outcome and running variable names.
    c : float, default 0
        True cutoff.
    placebo_cutoffs : list of float, optional
        Explicit placebo cutoff values. If None, auto-generates from
        the data distribution.
    n_placebo : int, default 10
        Number of placebo cutoffs if auto-generating.
    side : str, default 'both'
        Which side of the true cutoff to place placebos:
        'left', 'right', or 'both'.
    fuzzy : str, optional
        Treatment variable for fuzzy RD.
    p : int, default 1
    kernel : str, default 'triangular'
    alpha : float, default 0.05
    ax : matplotlib Axes, optional
    figsize : tuple

    Returns
    -------
    pd.DataFrame
        Columns: cutoff, estimate, se, ci_lower, ci_upper, pvalue,
        is_true_cutoff.
    """
    from .rdrobust import rdrobust

    X = data[x].values.astype(float)

    if placebo_cutoffs is None:
        placebo_cutoffs = _auto_placebo_cutoffs(X, c, n_placebo, side)

    # Always include the true cutoff for comparison
    all_cutoffs = sorted(set(list(placebo_cutoffs) + [c]))

    rows = []
    for cutoff in all_cutoffs:
        is_true = (cutoff == c)

        if is_true:
            subset = data
        elif cutoff < c:
            subset = data[X < c]
        else:
            subset = data[X >= c]

        if len(subset) < 20:
            continue

        try:
            r = rdrobust(subset, y=y, x=x, c=cutoff, fuzzy=fuzzy,
                         p=p, kernel=kernel, alpha=alpha)
            rows.append({
                'cutoff': cutoff,
                'estimate': r.estimate,
                'se': r.se,
                'ci_lower': r.ci[0],
                'ci_upper': r.ci[1],
                'pvalue': r.pvalue,
                'is_true_cutoff': is_true,
            })
        except (ValueError, np.linalg.LinAlgError):
            continue

    result = pd.DataFrame(rows)

    # Plot
    try:
        import matplotlib.pyplot as plt

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()

        true_mask = result['is_true_cutoff']
        placebo_mask = ~true_mask

        if placebo_mask.any():
            pr = result[placebo_mask]
            ax.errorbar(pr['cutoff'], pr['estimate'],
                        yerr=[pr['estimate'] - pr['ci_lower'],
                              pr['ci_upper'] - pr['estimate']],
                        fmt='o', color='#95A5A6', markersize=5, capsize=3,
                        label='Placebo')

        if true_mask.any():
            tr = result[true_mask]
            ax.errorbar(tr['cutoff'], tr['estimate'],
                        yerr=[tr['estimate'] - tr['ci_lower'],
                              tr['ci_upper'] - tr['estimate']],
                        fmt='D', color='#E74C3C', markersize=8, capsize=4,
                        label='True cutoff', zorder=5)

        ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8)
        ax.set_xlabel('Cutoff', fontsize=11)
        ax.set_ylabel('RD Estimate', fontsize=11)
        ax.set_title('Placebo Cutoff Test', fontsize=13)
        ax.legend(fontsize=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        fig.tight_layout()
    except ImportError:
        pass

    return result


def _auto_placebo_cutoffs(
    X: np.ndarray, c: float, n: int, side: str,
) -> List[float]:
    """Generate evenly-spaced placebo cutoffs from data quantiles."""
    left_x = X[X < c]
    right_x = X[X >= c]

    cutoffs = []
    if side in ('left', 'both') and len(left_x) > 20:
        # Use 10th to 90th percentile of left side
        qs = np.linspace(10, 90, n // 2 if side == 'both' else n)
        cutoffs.extend(np.percentile(left_x, qs).tolist())

    if side in ('right', 'both') and len(right_x) > 20:
        qs = np.linspace(10, 90, n // 2 if side == 'both' else n)
        cutoffs.extend(np.percentile(right_x, qs).tolist())

    # Remove any that are too close to the true cutoff
    cutoffs = [v for v in cutoffs if abs(v - c) > np.std(X) * 0.05]

    return cutoffs


# ======================================================================
# One-stop diagnostic summary
# ======================================================================

def rdsummary(
    data: pd.DataFrame,
    y: str,
    x: str,
    c: float = 0,
    fuzzy: Optional[str] = None,
    covs: Optional[List[str]] = None,
    p: int = 1,
    kernel: str = 'triangular',
    alpha: float = 0.05,
    verbose: bool = True,
    plot: bool = False,
    full: bool = False,
) -> Dict[str, Any]:
    """
    One-stop RD diagnostic battery.

    Runs the main RD estimate plus all standard validation checks in
    a single call, returning a structured summary.

    Diagnostics run (standard):
    1. Main RD estimate (conventional + robust)
    2. Density manipulation test (CJM 2020)
    3. Covariate balance at cutoff (if covs provided)
    4. Bandwidth sensitivity

    Extended diagnostics (full=True):
    5. Honest CI (Armstrong-Kolesár 2020)
    6. Power analysis (MDE at 80%)
    7. Placebo cutoff tests (5 placebos per side)
    8. Multiple bandwidth selectors comparison

    Parameters
    ----------
    data : pd.DataFrame
    y, x : str
        Outcome and running variable.
    c : float, default 0
        Cutoff.
    fuzzy : str, optional
        Treatment variable for fuzzy RD.
    covs : list of str, optional
        Pre-treatment covariates for balance test.
    p : int, default 1
    kernel : str, default 'triangular'
    alpha : float, default 0.05
    verbose : bool, default True
        Print formatted summary to console.
    plot : bool, default False
        Generate a multi-panel diagnostic plot.
    full : bool, default False
        Run extended diagnostics (honest CI, power, placebos).

    Returns
    -------
    dict with keys:
        'estimate': CausalResult from rdrobust
        'density_test': CausalResult from rddensity
        'balance': pd.DataFrame (if covs given)
        'bw_sensitivity': pd.DataFrame
        'honest_ci': CausalResult (if full=True)
        'power': RDPowerResult (if full=True)
        'placebos': pd.DataFrame (if full=True)
        'bandwidth_comparison': pd.DataFrame (if full=True)
        'figure': matplotlib Figure (if plot=True)
    """
    from .rdrobust import rdrobust
    from ..diagnostics.rddensity import rddensity

    results: Dict[str, Any] = {}

    # 1. Main estimate
    est = rdrobust(data, y=y, x=x, c=c, fuzzy=fuzzy, p=p,
                   kernel=kernel, alpha=alpha)
    results['estimate'] = est

    # 2. Density test
    try:
        density = rddensity(data, x=x, c=c, alpha=alpha)
        results['density_test'] = density
    except (ValueError, np.linalg.LinAlgError):
        results['density_test'] = None

    # 3. Covariate balance
    if covs:
        bal = rdbalance(data, x=x, c=c, covs=covs, p=p, kernel=kernel,
                        alpha=alpha)
        results['balance'] = bal
    else:
        results['balance'] = None

    # 4. Bandwidth sensitivity
    import matplotlib
    backend = matplotlib.get_backend()
    if not plot:
        matplotlib.use('Agg')
    try:
        bws = rdbwsensitivity(data, y=y, x=x, c=c, fuzzy=fuzzy, p=p,
                              kernel=kernel, n_grid=7, alpha=alpha)
        results['bw_sensitivity'] = bws
    except Exception:
        results['bw_sensitivity'] = None
    finally:
        if not plot:
            matplotlib.use(backend)

    # --- Extended diagnostics ---
    if full:
        # 5. Honest CI
        try:
            from .honest_ci import rd_honest
            honest = rd_honest(data, y=y, x=x, c=c, kernel=kernel, alpha=alpha)
            results['honest_ci'] = honest
        except Exception:
            results['honest_ci'] = None

        # 6. Power analysis
        try:
            from .rdpower import rdpower as _rdpower
            mi = est.model_info
            n_l = mi.get('n_left', 500)
            n_r = mi.get('n_right', 500)
            bw_h = mi.get('bandwidth_h', 0.5)
            if isinstance(bw_h, tuple):
                bw_h = bw_h[0]
            power_res = _rdpower(
                tau=est.estimate,
                n_left=n_l, n_right=n_r,
                h_left=bw_h, h_right=bw_h,
                alpha=alpha,
            )
            results['power'] = power_res
        except Exception:
            results['power'] = None

        # 7. Placebo cutoff tests
        try:
            placebos = rdplacebo(data, y=y, x=x, c=c, fuzzy=fuzzy,
                                n_placebo=10, p=p, kernel=kernel, alpha=alpha)
            results['placebos'] = placebos
        except Exception:
            results['placebos'] = None

        # 8. Bandwidth comparison across methods
        try:
            from .bandwidth import rdbwselect
            bw_comp = rdbwselect(data, y=y, x=x, c=c, fuzzy=fuzzy,
                                p=p, kernel=kernel, all=True)
            results['bandwidth_comparison'] = bw_comp
        except Exception:
            results['bandwidth_comparison'] = None

    # Print summary
    if verbose:
        _print_rdsummary(results, alpha, full=full)

    # Multi-panel diagnostic plot
    if plot:
        try:
            fig = _rd_diagnostic_plot(data, y, x, c, results, alpha)
            results['figure'] = fig
        except Exception:
            results['figure'] = None

    return results


def _print_rdsummary(results: Dict[str, Any], alpha: float,
                     full: bool = False):
    """Pretty-print the RD summary."""
    est = results['estimate']
    mi = est.model_info

    print("=" * 60)
    print(f"  RD Summary ({mi['rd_type']} Design)")
    print("=" * 60)

    # Main estimate
    print(f"\n{'Estimate':>20s}: {est.estimate:.4f}")
    print(f"{'Robust SE':>20s}: {est.se:.4f}")
    print(f"{'95% CI':>20s}: [{est.ci[0]:.4f}, {est.ci[1]:.4f}]")
    print(f"{'p-value':>20s}: {est.pvalue:.4f}")
    bw = mi['bandwidth_h']
    if isinstance(bw, tuple):
        print(f"{'Bandwidth (L/R)':>20s}: {bw[0]:.4f} / {bw[1]:.4f}")
    else:
        print(f"{'Bandwidth':>20s}: {bw:.4f}")
    print(f"{'N (left/right)':>20s}: {mi['n_left']} / {mi['n_right']}")
    print(f"{'N eff (left/right)':>20s}: {mi['n_effective_left']} / "
          f"{mi['n_effective_right']}")

    # Density test
    dt = results.get('density_test')
    if dt is not None:
        sig = "*" if dt.pvalue < alpha else ""
        print(f"\n--- Density Manipulation Test (CJM 2020) ---")
        print(f"  T-stat = {dt.estimate:.3f}, p = {dt.pvalue:.4f} {sig}")
        if dt.pvalue < alpha:
            print("  WARNING: Evidence of manipulation at cutoff!")
        else:
            print("  No evidence of manipulation.")

    # Balance
    bal = results.get('balance')
    if bal is not None:
        print(f"\n--- Covariate Balance at Cutoff ---")
        n_sig = bal['significant'].sum()
        print(bal[['covariate', 'estimate', 'pvalue', 'significant']]
              .to_string(index=False))
        if n_sig > 0:
            print(f"  WARNING: {n_sig} covariate(s) show significant "
                  f"imbalance at cutoff.")
        else:
            print("  All covariates balanced.")

    # BW sensitivity
    bws = results.get('bw_sensitivity')
    if bws is not None:
        print(f"\n--- Bandwidth Sensitivity ---")
        print(bws[['bandwidth', 'estimate', 'pvalue']].to_string(index=False))
        all_sig = (bws['pvalue'] < alpha).all()
        print(f"  {'Robust' if all_sig else 'NOT robust'} across bandwidths.")

    # Extended diagnostics
    if full:
        # Honest CI
        honest = results.get('honest_ci')
        if honest is not None:
            h_mi = honest.model_info
            print(f"\n--- Honest CI (Armstrong-Kolesar 2020) ---")
            print(f"  Honest 95% CI:  [{honest.ci[0]:.4f}, {honest.ci[1]:.4f}]")
            print(f"  Naive 95% CI:   [{h_mi['naive_ci'][0]:.4f}, "
                  f"{h_mi['naive_ci'][1]:.4f}]")
            print(f"  Smoothness M:   {h_mi['M']:.4g}")
            print(f"  Bias bound:     {h_mi['bias_bound']:.4f}")

        # Power
        power = results.get('power')
        if power is not None:
            print(f"\n--- Power Analysis ---")
            print(f"  Power (current): {power.power:.2%}")
            print(f"  MDE (80% power): {power.mde:.4f}")

        # Placebos
        placebos = results.get('placebos')
        if placebos is not None:
            print(f"\n--- Placebo Cutoff Tests ---")
            n_placebo_sig = placebos.loc[
                ~placebos['is_true_cutoff'], 'pvalue'
            ].lt(alpha).sum()
            n_placebos = (~placebos['is_true_cutoff']).sum()
            print(f"  {n_placebo_sig}/{n_placebos} placebo cutoffs "
                  f"significant at {alpha:.0%}")
            if n_placebo_sig > 0:
                print("  WARNING: Significant placebo effects detected.")
            else:
                print("  No significant placebo effects. Design looks valid.")

        # Bandwidth comparison
        bw_comp = results.get('bandwidth_comparison')
        if bw_comp is not None:
            print(f"\n--- Bandwidth Comparison ---")
            print(bw_comp.to_string(index=False))

    print("\n" + "=" * 60)


def _rd_diagnostic_plot(
    data: pd.DataFrame,
    y: str, x: str, c: float,
    results: Dict[str, Any],
    alpha: float,
):
    """Generate a multi-panel RD diagnostic figure."""
    import matplotlib.pyplot as plt

    n_panels = 4
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Panel 1: RD Plot
    from .rdrobust import rdplot
    try:
        rdplot(data, y=y, x=x, c=c, ax=axes[0, 0], show_bw=True,
               title='RD Plot with Bandwidth')
    except Exception:
        axes[0, 0].set_title('RD Plot (failed)')

    # Panel 2: Density at cutoff
    from .rdrobust import rdplotdensity
    try:
        rdplotdensity(data, x=x, c=c, ax=axes[0, 1],
                      title='Density at Cutoff')
    except Exception:
        axes[0, 1].set_title('Density Plot (failed)')

    # Panel 3: Bandwidth sensitivity
    bws = results.get('bw_sensitivity')
    if bws is not None and len(bws) > 0:
        ax3 = axes[1, 0]
        est = results['estimate']
        mi = est.model_info
        h_opt = mi.get('bandwidth_h', None)

        ax3.errorbar(bws['bandwidth'], bws['estimate'],
                     yerr=[bws['estimate'] - bws['ci_lower'],
                           bws['ci_upper'] - bws['estimate']],
                     fmt='o-', color='#2C3E50', markersize=4,
                     linewidth=1, capsize=3, alpha=0.8)
        ax3.axhline(y=0, color='gray', linestyle='--', linewidth=0.8)
        if h_opt is not None:
            bw_val = h_opt[0] if isinstance(h_opt, tuple) else h_opt
            ax3.axvline(x=bw_val, color='#E74C3C', linestyle='--',
                        linewidth=0.8, alpha=0.7)
        ax3.set_xlabel('Bandwidth')
        ax3.set_ylabel('RD Estimate')
        ax3.set_title('Bandwidth Sensitivity')
        ax3.spines['top'].set_visible(False)
        ax3.spines['right'].set_visible(False)
    else:
        axes[1, 0].set_title('Bandwidth Sensitivity (N/A)')

    # Panel 4: Covariate balance or placebo cutoffs
    bal = results.get('balance')
    placebos = results.get('placebos')
    ax4 = axes[1, 1]

    if bal is not None and len(bal) > 0:
        # Covariate balance plot
        y_pos = range(len(bal))
        colors = ['#E74C3C' if s else '#2ECC71'
                  for s in bal['significant']]
        ax4.barh(list(y_pos), bal['estimate'], color=colors, alpha=0.7)
        ax4.set_yticks(list(y_pos))
        ax4.set_yticklabels(bal['covariate'], fontsize=9)
        ax4.axvline(x=0, color='gray', linestyle='--', linewidth=0.8)
        ax4.set_xlabel('Discontinuity in Covariate')
        ax4.set_title('Covariate Balance at Cutoff')
        ax4.spines['top'].set_visible(False)
        ax4.spines['right'].set_visible(False)
    elif placebos is not None and len(placebos) > 0:
        # Placebo cutoff plot
        true_mask = placebos['is_true_cutoff']
        placebo_mask = ~true_mask
        if placebo_mask.any():
            pr = placebos[placebo_mask]
            ax4.errorbar(pr['cutoff'], pr['estimate'],
                         yerr=[pr['estimate'] - pr['ci_lower'],
                               pr['ci_upper'] - pr['estimate']],
                         fmt='o', color='#95A5A6', markersize=5,
                         capsize=3, label='Placebo')
        if true_mask.any():
            tr = placebos[true_mask]
            ax4.errorbar(tr['cutoff'], tr['estimate'],
                         yerr=[tr['estimate'] - tr['ci_lower'],
                               tr['ci_upper'] - tr['estimate']],
                         fmt='D', color='#E74C3C', markersize=8,
                         capsize=4, label='True cutoff', zorder=5)
        ax4.axhline(y=0, color='gray', linestyle='--', linewidth=0.8)
        ax4.set_xlabel('Cutoff')
        ax4.set_ylabel('RD Estimate')
        ax4.set_title('Placebo Cutoff Tests')
        ax4.legend(fontsize=9)
        ax4.spines['top'].set_visible(False)
        ax4.spines['right'].set_visible(False)
    else:
        ax4.set_title('Additional Diagnostics (N/A)')
        ax4.text(0.5, 0.5, 'No covariates\nprovided',
                 ha='center', va='center', transform=ax4.transAxes,
                 fontsize=12, color='gray')

    fig.suptitle('RD Diagnostic Dashboard', fontsize=14, fontweight='bold')
    fig.tight_layout(rect=[0, 0, 1, 0.96])

    return fig
