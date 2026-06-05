"""
Structural break tests.

Provides Bai-Perron multiple structural break test, CUSUM test,
Chow test, and Andrews-Ploberger supremum test.

Equivalent to Stata's ``estat sbknown`` / ``estat sbsingle`` and
R's ``strucchange::breakpoints()``.

References
----------
Andrews, D.W.K. (1993).
"Tests for Parameter Instability and Structural Change with Unknown
Change Point." *Econometrica*, 61(4), 821-856. doi:10.2307/2951764.
(Asymptotic null distribution of the sup-F / Quandt-Andrews statistic.)

Bai, J. & Perron, P. (1998).
"Estimating and Testing Linear Models with Multiple Structural Changes."
*Econometrica*, 66(1), 47-78. [@bai1998estimating]

Brown, R.L., Durbin, J. & Evans, J.M. (1975).
"Techniques for Testing the Constancy of Regression Relationships Over Time."
*JRSS-B*, 37(2), 149-192. [@brown1975techniques]
"""

from functools import lru_cache
from typing import List, Dict, Any
import numpy as np
import pandas as pd

from ..core.results import ExportMixin


_SUPF_SEED = 20240601


@lru_cache(maxsize=128)
def _supf_null_distribution(
    q: int,
    n_grid: int,
    trimming: float,
    n_reps: int = 5000,
    seed: int = _SUPF_SEED,
) -> np.ndarray:
    """Monte-Carlo sample of the Andrews (1993) sup-F limiting null law.

    Under H0 of no break, the Chow F-process satisfies
    ``q * F(lambda) -> ||BB_q(lambda)||**2 / (lambda * (1 - lambda))`` where
    ``BB_q`` is a q-vector of independent standard Brownian bridges
    (Andrews 1993, Econometrica 61(4)). The sup-F statistic therefore
    converges to the supremum of that functional over the trimmed interval
    ``[trimming, 1 - trimming]``. We draw from this limit law by simulating q
    Brownian bridges on a discrete grid whose resolution is tied to the
    break-point search (``n_grid``), which delivers correct *finite-sample*
    size -- a naive ``F(q, n-2k)`` reference over-rejects badly (it ignores
    the maximisation over candidate break points).

    Results are deterministic (fixed ``seed``) and cached per
    ``(q, n_grid, trimming)`` so a call simulates the null at most once.
    """
    rng = np.random.default_rng(seed)
    N = int(n_grid)
    lam = np.arange(1, N) / N  # interior fractions j/N; excludes lambda == 1
    mask = (lam >= trimming) & (lam <= 1.0 - trimming)
    midx = np.where(mask)[0]
    if midx.size == 0:  # degenerate trimming -> fall back to the mid-point
        midx = np.array([max(N // 2 - 1, 0)])
    denom = (lam * (1.0 - lam))[midx]
    lam_m = lam[midx]
    dt = 1.0 / N
    sups: List[np.ndarray] = []
    done = 0
    while done < n_reps:
        c = min(2500, n_reps - done)
        incr = rng.standard_normal((c, N, q)) * np.sqrt(dt)
        w = np.cumsum(incr, axis=1)              # Brownian motion, (c, N, q)
        w1 = w[:, -1, :][:, None, :]             # W(1)
        bb = w[:, midx, :] - lam_m[None, :, None] * w1   # Brownian bridge
        qproc = (bb ** 2).sum(axis=2) / denom[None, :]   # ||BB||^2 / (l(1-l))
        sups.append(qproc.max(axis=1) / q)               # F-scale supremum
        done += c
    return np.sort(np.concatenate(sups))


def _supf_pvalue(stat: float, q: int, n: int, trimming: float) -> float:
    """Asymptotic p-value for a sup-F (Quandt-Andrews) statistic.

    ``stat`` is the supremum over candidate break points of the Chow F
    statistic testing ``q`` restrictions. Returns ``P(sup-F >= stat)`` under
    the Andrews (1993) null. The reference-grid resolution is bucketed from
    the sample size so the test keeps correct size across n.
    """
    if not np.isfinite(stat) or stat <= 0:
        return 1.0
    n_grid = int(round(min(max(int(n), 120), 1500) / 50.0)) * 50
    null = _supf_null_distribution(int(q), n_grid, round(float(trimming), 3))
    return float((1 + np.count_nonzero(null >= stat)) / (null.size + 1))


class StructuralBreakResult(ExportMixin):
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
        Method: 'bai-perron', 'chow', 'sup-f'. ``'sup-f'`` and ``'chow'``
        both compute the single-break Quandt-Andrews sup-F statistic
        (maximised over candidate break points); ``'bai-perron'`` adds
        sequential supF(l+1 | l) detection.
    alpha : float, default 0.05
        Significance level.

    Returns
    -------
    StructuralBreakResult
        ``f_stats`` / ``p_values`` hold the sup-F statistic and its
        Andrews (1993) asymptotic p-value (a scalar for 'sup-f'/'chow', and a
        list -- one per detected break -- for 'bai-perron').

    Notes
    -----
    The sup-F statistic is a supremum over candidate break points, so its
    null distribution is the Andrews (1993) sup-F law, **not** ``F(k, n-2k)``.
    P-values are obtained by simulating that limit law (a q-vector Brownian
    bridge functional) on a grid tied to the sample size, which restores
    correct size -- a naive F reference rejects on ~35% of white-noise series
    at the 5% level. A known-breakpoint Chow test (fixed date, ordinary F
    distribution) is a different procedure and is not what this function
    computes.

    References
    ----------
    Andrews (1993) for the sup-F null; Bai & Perron (1998) for the sequential
    multiple-break procedure. See module docstring for full citations.

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

        # best_f is a supremum over candidate break points -> its null is the
        # Andrews (1993) sup-F law, NOT F(k, n-2k). Using the naive F CDF here
        # inflated the false-positive rate to ~35% on white noise.
        p_value = _supf_pvalue(best_f, k, n, min_segment)

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
    step_f_stats: List[float] = []
    step_p_values: List[float] = []
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

        # Sequential sup-F(l+1 | l) stopping rule. best_f is the largest Chow
        # F across segments and candidate break points, so it is referred to
        # the Andrews (1993) sup-F null (full-sample grid -> conservative for
        # later, shorter segments), replacing the naive F CDF that drove
        # spurious over-detection.
        p_value = _supf_pvalue(best_f, k, n, min_segment)
        if p_value >= alpha:
            break

        step_f_stats.append(float(best_f))
        step_p_values.append(float(p_value))
        break_dates.append(best_break)
        # Split the segment
        start, end = remaining_segments.pop(best_seg_idx)
        remaining_segments.insert(best_seg_idx, (start, best_break))
        remaining_segments.insert(best_seg_idx + 1, (best_break, end))

    # Sort breaks chronologically, keeping each break aligned with the sup-F
    # statistic / p-value that detected it.
    if break_dates:
        paired = sorted(zip(break_dates, step_f_stats, step_p_values))
        break_dates = [b for b, _, _ in paired]
        step_f_stats = [f for _, f, _ in paired]
        step_p_values = [p for _, _, p in paired]

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
        f_stats=step_f_stats if step_f_stats else None,
        p_values=step_p_values if step_p_values else None,
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
        Keys: 'cusum' (the standardized CUSUM path S_t), 'boundary' (the
        Brown-Durbin-Evans linear boundary evaluated along the path),
        'boundary_coefficient' (the BDE constant a), 'max_cusum',
        'critical_value' (= a, for back-compat), 'reject', 'n_obs'.

    Notes
    -----
    Implements the recursive-residual CUSUM test of Brown, Durbin & Evans
    (1975). Under H0 the standardized path behaves like a Brownian motion, and
    the 5% test rejects when the path crosses the *diverging linear* boundary
    ``±a·(1 + 2(t-k)/(T-k))`` with ``a = 0.948`` (``0.850`` at 10%, ``1.143`` at
    1%) -- not a flat constant. Comparing ``max|S_t|`` to a constant grossly
    over-rejects (~32% size at the nominal 5% level).

    References
    ----------
    Brown, Durbin & Evans (1975); see module docstring for the full citation.
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
        ft = 1 + X_data[t] @ np.linalg.solve(Xt.T @ Xt, X_data[t])
        rec_resid.append(resid / np.sqrt(max(ft, 1e-10)))

    rec_resid = np.array(rec_resid)
    sigma = np.std(rec_resid, ddof=1)

    # CUSUM path: S_t = cumsum(recursive residuals) / (sigma * sqrt(T - k)).
    # Under H0 this behaves like a standard Brownian motion on [0, 1].
    cusum = np.cumsum(rec_resid) / (sigma * np.sqrt(n - k))
    max_cusum = float(np.max(np.abs(cusum)))

    # Brown, Durbin & Evans (1975): reject if the path crosses the *diverging
    # linear* boundary +/- a * (1 + 2 s), s = (t - k) / (T - k) in (0, 1] --
    # NOT a constant. The crossing constant a is 0.850 / 0.948 / 1.143 at the
    # 10% / 5% / 1% level. Comparing max|S_t| to a flat constant (e.g. the
    # Kolmogorov-bridge 1.358) is a different, badly oversized test: it rejects
    # ~32% of stable series at the nominal 5% level.
    a_coef = {0.01: 1.143, 0.05: 0.948, 0.10: 0.850}.get(alpha, 0.948)
    m = len(cusum)
    s_grid = np.arange(1, m + 1) / m
    boundary = a_coef * (1.0 + 2.0 * s_grid)
    crossing = np.abs(cusum) > boundary
    reject = bool(np.any(crossing))

    return {
        'cusum': cusum,
        'max_cusum': max_cusum,
        'boundary': boundary,
        'boundary_coefficient': a_coef,
        # Back-compat key: the BDE crossing constant a. The boundary is linear,
        # so there is no single scalar critical value -- use 'boundary'.
        'critical_value': a_coef,
        'reject': reject,
        'n_obs': n,
    }
