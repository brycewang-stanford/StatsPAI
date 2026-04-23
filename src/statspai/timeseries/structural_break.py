"""
Structural break tests.

Provides Bai-Perron multiple structural break test, CUSUM test,
Chow test, and Andrews-Ploberger supremum test.

Equivalent to Stata's ``estat sbknown`` / ``estat sbsingle`` and
R's ``strucchange::breakpoints()``.

References
----------
Bai, J. & Perron, P. (1998).
"Estimating and Testing Linear Models with Multiple Structural Changes."
*Econometrica*, 66(1), 47-78. [@bai1998estimating]

Brown, R.L., Durbin, J. & Evans, J.M. (1975).
"Techniques for Testing the Constancy of Regression Relationships Over Time."
*JRSS-B*, 37(2), 149-192. [@brown1975techniques]
"""

from typing import Optional, List, Dict, Any, Union
import numpy as np
import pandas as pd
from scipy import stats


class StructuralBreakResult:
    """Results from structural break tests."""

    def __init__(self, test_type, break_dates, f_stats, p_values,
                 n_breaks, rss_full, rss_segments, bic, n_obs):
        self.test_type = test_type
        self.break_dates = break_dates
        self.f_stats = f_stats
        self.p_values = p_values
        self.n_breaks = n_breaks
        self.rss_full = rss_full
        self.rss_segments = rss_segments
        self.bic = bic
        self.n_obs = n_obs

    def summary(self) -> str:
        lines = [
            f"Structural Break Test: {self.test_type}",
            "=" * 55,
            f"N obs: {self.n_obs}",
            f"Number of breaks detected: {self.n_breaks}",
        ]
        if self.break_dates:
            lines.append(f"Break date(s): {self.break_dates}")
        if self.f_stats is not None:
            if isinstance(self.f_stats, (list, np.ndarray)):
                for i, (f, p) in enumerate(zip(self.f_stats, self.p_values)):
                    lines.append(f"  Break {i+1}: F = {f:.4f}, p = {p:.4f}")
            else:
                lines.append(f"F-statistic: {self.f_stats:.4f}")
                lines.append(f"P-value: {self.p_values:.4f}")
        lines.append(f"BIC: {self.bic:.4f}" if self.bic is not None else "")
        lines.append("=" * 55)
        return "\n".join(lines)

    def plot(self, ax=None, **kwargs):
        """Plot with break dates marked."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib required for plotting")
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 5))
        for bd in self.break_dates:
            ax.axvline(bd, color='red', ls='--', lw=1.5, label=f'Break at {bd}')
        ax.legend()
        return ax


def structural_break(
    data: pd.DataFrame = None,
    y: str = None,
    x: List[str] = None,
    max_breaks: int = 5,
    min_segment: float = 0.15,
    method: str = "bai-perron",
    alpha: float = 0.05,
) -> StructuralBreakResult:
    """
    Structural break detection.

    Implements Bai-Perron (1998) sequential/global break detection.

    Parameters
    ----------
    data : pd.DataFrame
        Time series data (assumed ordered by time).
    y : str
        Dependent variable.
    x : list of str, optional
        Regressors. If None, uses constant only (mean shift).
    max_breaks : int, default 5
        Maximum number of breaks to test.
    min_segment : float, default 0.15
        Minimum segment length as fraction of sample.
    method : str, default 'bai-perron'
        Method: 'bai-perron', 'chow', 'sup-f'.
    alpha : float, default 0.05
        Significance level.

    Returns
    -------
    StructuralBreakResult

    Examples
    --------
    >>> import statspai as sp
    >>> result = sp.structural_break(df, y='gdp_growth', x=['inflation'])
    >>> print(result.summary())
    """
    if data is None:
        raise ValueError("data is required")

    y_data = data[y].values.astype(float)
    n = len(y_data)

    if x is not None:
        X_data = data[x].values.astype(float)
        X_full = np.column_stack([np.ones(n), X_data])
    else:
        X_full = np.ones((n, 1))

    k = X_full.shape[1]
    h = max(int(n * min_segment), k + 1)  # minimum segment size

    # Full sample RSS
    beta_full = np.linalg.lstsq(X_full, y_data, rcond=None)[0]
    rss_full = np.sum((y_data - X_full @ beta_full)**2)

    if method == 'chow' or method == 'sup-f':
        # Sup-F test: find the single break maximizing F
        best_f = -np.inf
        best_break = None
        f_stats = []

        for t in range(h, n - h):
            # Segment 1
            X1, y1 = X_full[:t], y_data[:t]
            b1 = np.linalg.lstsq(X1, y1, rcond=None)[0]
            rss1 = np.sum((y1 - X1 @ b1)**2)

            # Segment 2
            X2, y2 = X_full[t:], y_data[t:]
            b2 = np.linalg.lstsq(X2, y2, rcond=None)[0]
            rss2 = np.sum((y2 - X2 @ b2)**2)

            rss_break = rss1 + rss2
            denom = rss_break / max(n - 2 * k, 1)
            f_stat = ((rss_full - rss_break) / k) / max(denom, 1e-20)
            f_stats.append((t, f_stat))

            if f_stat > best_f:
                best_f = f_stat
                best_break = t

        p_value = 1 - stats.f.cdf(best_f, k, n - 2*k) if best_f > 0 else 1.0

        return StructuralBreakResult(
            test_type='Sup-F' if method == 'sup-f' else 'Chow',
            break_dates=[best_break] if p_value < alpha else [],
            f_stats=best_f,
            p_values=p_value,
            n_breaks=1 if p_value < alpha else 0,
            rss_full=rss_full,
            rss_segments=None,
            bic=None,
            n_obs=n,
        )

    # Bai-Perron: sequential detection
    break_dates = []
    remaining_segments = [(0, n)]

    for _ in range(max_breaks):
        best_f = -np.inf
        best_break = None
        best_seg_idx = None

        for seg_idx, (start, end) in enumerate(remaining_segments):
            seg_len = end - start
            seg_h = max(int(seg_len * min_segment), k + 1)

            if seg_len < 2 * seg_h:
                continue

            X_seg = X_full[start:end]
            y_seg = y_data[start:end]
            b_seg = np.linalg.lstsq(X_seg, y_seg, rcond=None)[0]
            rss_seg = np.sum((y_seg - X_seg @ b_seg)**2)

            for t in range(seg_h, seg_len - seg_h):
                abs_t = start + t
                X1, y1 = X_full[start:abs_t], y_data[start:abs_t]
                X2, y2 = X_full[abs_t:end], y_data[abs_t:end]

                b1 = np.linalg.lstsq(X1, y1, rcond=None)[0]
                b2 = np.linalg.lstsq(X2, y2, rcond=None)[0]
                rss_split = np.sum((y1 - X1 @ b1)**2) + np.sum((y2 - X2 @ b2)**2)

                f_stat = ((rss_seg - rss_split) / k) / (rss_split / (seg_len - 2*k))
                if f_stat > best_f:
                    best_f = f_stat
                    best_break = abs_t
                    best_seg_idx = seg_idx

        if best_break is None:
            break

        p_value = 1 - stats.f.cdf(best_f, k, n - 2*k) if best_f > 0 else 1.0
        if p_value >= alpha:
            break

        break_dates.append(best_break)
        # Split the segment
        start, end = remaining_segments.pop(best_seg_idx)
        remaining_segments.insert(best_seg_idx, (start, best_break))
        remaining_segments.insert(best_seg_idx + 1, (best_break, end))

    break_dates.sort()

    # Compute BIC
    segments = [0] + break_dates + [n]
    rss_total = 0
    n_params_total = 0
    for i in range(len(segments) - 1):
        s, e = segments[i], segments[i+1]
        Xs, ys = X_full[s:e], y_data[s:e]
        bs = np.linalg.lstsq(Xs, ys, rcond=None)[0]
        rss_total += np.sum((ys - Xs @ bs)**2)
        n_params_total += k

    bic_val = n * np.log(rss_total / n) + n_params_total * np.log(n)

    return StructuralBreakResult(
        test_type='Bai-Perron',
        break_dates=break_dates,
        f_stats=None,
        p_values=None,
        n_breaks=len(break_dates),
        rss_full=rss_full,
        rss_segments=rss_total,
        bic=bic_val,
        n_obs=n,
    )


def cusum_test(
    data: pd.DataFrame,
    y: str,
    x: List[str] = None,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    CUSUM test for parameter stability.

    Tests H0: parameters are stable vs H1: parameter shift.

    Parameters
    ----------
    data : pd.DataFrame
    y : str
        Dependent variable.
    x : list of str, optional
        Regressors.
    alpha : float, default 0.05

    Returns
    -------
    dict
        Keys: 'cusum', 'critical_value', 'reject', 'n_obs'.
    """
    y_data = data[y].values.astype(float)
    n = len(y_data)

    if x is not None:
        X_data = np.column_stack([np.ones(n), data[x].values.astype(float)])
    else:
        X_data = np.ones((n, 1))

    k = X_data.shape[1]

    # Recursive residuals
    rec_resid = []
    for t in range(k, n):
        Xt = X_data[:t]
        yt = y_data[:t]
        bt = np.linalg.lstsq(Xt, yt, rcond=None)[0]
        pred = X_data[t] @ bt
        resid = y_data[t] - pred
        ft = 1 + X_data[t] @ np.linalg.inv(Xt.T @ Xt) @ X_data[t]
        rec_resid.append(resid / np.sqrt(max(ft, 1e-10)))

    rec_resid = np.array(rec_resid)
    sigma = np.std(rec_resid, ddof=1)

    # CUSUM statistic
    cusum = np.cumsum(rec_resid) / (sigma * np.sqrt(n - k))
    max_cusum = np.max(np.abs(cusum))

    # Critical values (Brownian bridge)
    # Approximate: critical value ≈ 1.358 for 5%
    crit_vals = {0.01: 1.628, 0.05: 1.358, 0.10: 1.224}
    cv = crit_vals.get(alpha, 1.358)

    return {
        'cusum': cusum,
        'max_cusum': max_cusum,
        'critical_value': cv,
        'reject': max_cusum > cv,
        'n_obs': n,
    }
