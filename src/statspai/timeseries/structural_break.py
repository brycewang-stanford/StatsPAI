"""
Structural break tests.

Provides Bai-Perron multiple structural break test, CUSUM test,
Chow test, and Andrews-Ploberger supremum test.

The sup-F statistic corresponds to Stata's ``estat sbsingle`` (supremum
Wald = k x sup-F) and ``strucchange::Fstats``; ``method='global'`` to
``strucchange::breakpoints()``; the CUSUM test to ``estat sbcusum`` and
``strucchange::efp(type = "Rec-CUSUM")``.

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
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd

from .._result_serialize import ResultProtocolMixin

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
        w = np.cumsum(incr, axis=1)  # Brownian motion, (c, N, q)
        w1 = w[:, -1, :][:, None, :]  # W(1)
        bb = w[:, midx, :] - lam_m[None, :, None] * w1  # Brownian bridge
        qproc = (bb**2).sum(axis=2) / denom[None, :]  # ||BB||^2 / (l(1-l))
        sups.append(qproc.max(axis=1) / q)  # F-scale supremum
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


class StructuralBreakResult(ResultProtocolMixin):
    """Results from structural break tests.

    Returned by :func:`structural_break`. Holds the detected break
    point(s), the sup-F statistic(s) with their Andrews (1993)
    asymptotic p-value(s), and segment goodness-of-fit (RSS, BIC).

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> # mean shift halfway through the sample
    >>> y = np.concatenate([rng.normal(0.0, 1.0, 100),
    ...                     rng.normal(3.0, 1.0, 100)])
    >>> df = pd.DataFrame({"y": y})
    >>> res = sp.structural_break(df, y="y", method="sup-f")
    >>> isinstance(res, sp.StructuralBreakResult)
    True
    >>> res.test_type
    'Sup-F'
    >>> res.n_obs
    200
    >>> res.n_breaks >= 1
    True
    """

    def __init__(
        self,
        test_type: str,
        break_dates: list[int],
        f_stats: Any,
        p_values: Any,
        n_breaks: int,
        rss_full: float,
        rss_segments: Optional[float],
        bic: Optional[float],
        n_obs: int,
    ) -> None:
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
                    lines.append(f"  Break {i + 1}: F = {f:.4f}, p = {p:.4f}")
            else:
                lines.append(f"F-statistic: {self.f_stats:.4f}")
                lines.append(f"P-value: {self.p_values:.4f}")
        lines.append(f"BIC: {self.bic:.4f}" if self.bic is not None else "")
        lines.append("=" * 55)
        return "\n".join(lines)

    def plot(self, ax: Any = None, **kwargs: Any) -> Any:
        """Plot with break dates marked."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib required for plotting")
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 5))
        for bd in self.break_dates:
            ax.axvline(
                bd,
                color="red",
                ls="--",
                lw=1.5,
                label=f"Break at {bd}",
            )
        ax.legend()
        return ax


def structural_break(
    data: Optional[pd.DataFrame] = None,
    y: Optional[str] = None,
    x: Optional[List[str]] = None,
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
        Method: 'bai-perron', 'global', 'chow', 'sup-f'. ``'sup-f'`` and
        ``'chow'`` both compute the single-break Quandt-Andrews sup-F
        statistic (maximised over candidate break points, the
        ``strucchange::Fstats`` grid); ``'bai-perron'`` detects breaks one at
        a time (binary segmentation), each step tested with the sup-F
        statistic; ``'global'`` is the Bai-Perron global SSR minimisation by
        dynamic programming with the number of breaks chosen by BIC -- the
        estimator of ``strucchange::breakpoints(h = min_segment)``.
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
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> n = 120
    >>> inflation = rng.normal(2, 1, size=n)
    >>> shift = np.where(np.arange(n) < 60, 0.0, 2.0)  # mean shift at t=60
    >>> gdp_growth = (
    ...     1.0 + 0.3 * inflation + shift + rng.normal(0, 0.5, size=n)
    ... )
    >>> df = pd.DataFrame({"gdp_growth": gdp_growth, "inflation": inflation})
    >>> result = sp.structural_break(df, y='gdp_growth', x=['inflation'])
    >>> type(result).__name__
    'StructuralBreakResult'
    """
    if data is None:
        raise ValueError("data is required")
    assert y is not None

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
    rss_full = np.sum((y_data - X_full @ beta_full) ** 2)

    if method not in ("bai-perron", "chow", "sup-f", "global"):
        raise ValueError("method must be 'bai-perron', 'global', 'sup-f' or 'chow'")

    if method == "global":
        return _breakpoints_global(y_data, X_full, min_segment, max_breaks, rss_full)

    if method == "chow" or method == "sup-f":
        # Sup-F test: find the single break maximizing F. Candidate break
        # points t (= size of the first regime) run from floor(min_segment*n)
        # to n - floor(min_segment*n) inclusive, clamped to [k+1, n-k-1] --
        # the strucchange::Fstats(from = min_segment) convention.
        best_f = -np.inf
        best_break = None
        f_stats = []
        t_from = max(int(np.floor(min_segment * n)), k + 1)
        t_to = min(n - int(np.floor(min_segment * n)), n - k - 1)

        for t in range(t_from, t_to + 1):
            # Segment 1
            X1, y1 = X_full[:t], y_data[:t]
            b1 = np.linalg.lstsq(X1, y1, rcond=None)[0]
            rss1 = np.sum((y1 - X1 @ b1) ** 2)

            # Segment 2
            X2, y2 = X_full[t:], y_data[t:]
            b2 = np.linalg.lstsq(X2, y2, rcond=None)[0]
            rss2 = np.sum((y2 - X2 @ b2) ** 2)

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
        selected_breaks = (
            [best_break] if p_value < alpha and best_break is not None else []
        )

        res = StructuralBreakResult(
            test_type="Sup-F" if method == "sup-f" else "Chow",
            break_dates=selected_breaks,
            f_stats=best_f,
            p_values=p_value,
            n_breaks=len(selected_breaks),
            rss_full=rss_full,
            rss_segments=None,
            bic=None,
            n_obs=n,
        )
        # Wald (chi2) scale: k * F -- strucchange::Fstats' statistic and
        # Stata ``estat sbsingle``'s supremum Wald; ``sup_break`` is the
        # maximising break point (size of the first regime) whether or not
        # it is significant.
        res.sup_wald = float(k * best_f)
        res.sup_break = best_break
        res.f_path = np.array([f for _, f in f_stats])
        res.candidate_breaks = np.array([t for t, _ in f_stats])
        return res

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
            rss_seg = np.sum((y_seg - X_seg @ b_seg) ** 2)

            for t in range(seg_h, seg_len - seg_h):
                abs_t = start + t
                X1, y1 = X_full[start:abs_t], y_data[start:abs_t]
                X2, y2 = X_full[abs_t:end], y_data[abs_t:end]

                b1 = np.linalg.lstsq(X1, y1, rcond=None)[0]
                b2 = np.linalg.lstsq(X2, y2, rcond=None)[0]
                rss_split = np.sum((y1 - X1 @ b1) ** 2) + np.sum((y2 - X2 @ b2) ** 2)

                f_stat = ((rss_seg - rss_split) / k) / (rss_split / (seg_len - 2 * k))
                if f_stat > best_f:
                    best_f = f_stat
                    best_break = abs_t
                    best_seg_idx = seg_idx

        if best_break is None:
            break
        assert best_seg_idx is not None

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
        s, e = segments[i], segments[i + 1]
        Xs, ys = X_full[s:e], y_data[s:e]
        bs = np.linalg.lstsq(Xs, ys, rcond=None)[0]
        rss_total += np.sum((ys - Xs @ bs) ** 2)
        n_params_total += k

    bic_val = n * np.log(rss_total / n) + n_params_total * np.log(n)

    return StructuralBreakResult(
        test_type="Bai-Perron",
        break_dates=break_dates,
        f_stats=step_f_stats if step_f_stats else None,
        p_values=step_p_values if step_p_values else None,
        n_breaks=len(break_dates),
        rss_full=rss_full,
        rss_segments=rss_total,
        bic=bic_val,
        n_obs=n,
    )


def _segment_rss_table(y: np.ndarray, X: np.ndarray, h: int) -> np.ndarray:
    """RSS[i, j] of the OLS fit on observations i..j (0-based, inclusive).

    Computed for every start i by recursive least squares (the recursive
    residuals of the segment, as ``strucchange::breakpoints`` does), so the
    whole triangle costs O(n^2 k^2). Entries with fewer than k + 1
    observations are NaN; only segments of length >= h are ever used.
    """
    n, k = X.shape
    rss = np.full((n, n), np.nan)
    for i in range(0, n - h + 1):
        Xi, yi = X[i : i + k], y[i : i + k]
        try:
            P = np.linalg.inv(Xi.T @ Xi)
        except np.linalg.LinAlgError:
            P = None
        if P is None:
            # singular start (e.g. constant regressor over k points): fall
            # back to direct fits for this start
            for j in range(i + k, n):
                Xs, ys = X[i : j + 1], y[i : j + 1]
                b = np.linalg.lstsq(Xs, ys, rcond=None)[0]
                rss[i, j] = float(np.sum((ys - Xs @ b) ** 2))
            continue
        b = P @ (Xi.T @ yi)
        acc = float(np.sum((yi - Xi @ b) ** 2))  # exactly 0 for k points
        rss[i, i + k - 1] = acc
        for j in range(i + k, n):
            x = X[j]
            f = 1.0 + x @ P @ x
            e = y[j] - x @ b
            acc += e * e / f
            rss[i, j] = acc
            Px = P @ x
            P = P - np.outer(Px, Px) / f
            b = b + Px * (e / f)
    return rss


def _breakpoints_global(
    y: np.ndarray,
    X: np.ndarray,
    min_segment: float,
    max_breaks: Optional[int],
    rss_full: float,
) -> "StructuralBreakResult":
    """Bai-Perron global minimisation, as ``strucchange::breakpoints``.

    Minimum segment size ``h = floor(min_segment * n)`` (or ``min_segment``
    itself when >= 1); for m = 1..M the optimal partition minimises the total
    RSS by dynamic programming; the number of breaks minimises
    ``BIC = -2 logL + (k + 1)(m + 1) log n`` with
    ``logL = -n/2 (log RSS + 1 - log n + log 2 pi)``.
    Break points are the index (1-based) of the last observation of each
    regime, i.e. the size of the preceding regimes.
    """
    n, k = X.shape
    h = int(np.floor(min_segment * n)) if min_segment < 1 else int(min_segment)
    if h <= k:
        raise ValueError(
            "minimum segment size must be greater than the number of regressors"
        )
    if h > n // 2:
        raise ValueError(
            "minimum segment size must be smaller than half the number of "
            "observations"
        )
    M = int(np.ceil(n / h)) - 2
    if max_breaks is not None:
        M = max(1, min(int(max_breaks), M))
    rss = _segment_rss_table(y, X, h)

    def seg(a: int, b: int) -> float:  # 1-based inclusive a..b
        return rss[a - 1, b - 1]

    # table[m][i]: optimal RSS of observations 1..i with m breaks, and the
    # position of the last of those breaks (1-based end of regime m).
    idx = np.arange(h, n - h + 1)
    table_rss = {1: {i: seg(1, i) for i in idx}}
    table_arg: dict = {1: {}}
    for m in range(2, M + 1):
        table_rss[m], table_arg[m] = {}, {}
        for i in range(m * h, n - h + 1):
            best, arg = np.inf, None
            for j in range((m - 1) * h, i - h + 1):
                prev = table_rss[m - 1].get(j, np.nan)
                v = prev + seg(j + 1, i)
                if np.isfinite(v) and v < best:
                    best, arg = v, j
            table_rss[m][i], table_arg[m][i] = best, arg

    def extract(m: int) -> list:
        best, opt = np.inf, None
        for i in idx:
            prev = table_rss[m].get(int(i), np.nan)
            v = prev + seg(int(i) + 1, n)
            if np.isfinite(v) and v < best:
                best, opt = v, int(i)
        bps = [opt]
        for mm in range(m, 1, -1):
            bps.insert(0, table_arg[mm][bps[0]])
        return bps

    rss_by_m = [float(seg(1, n))]
    bps_by_m: list = [[]]
    for m in range(1, M + 1):
        bps = extract(m)
        edges = [0] + bps + [n]
        rss_by_m.append(
            float(sum(seg(edges[q] + 1, edges[q + 1]) for q in range(len(edges) - 1)))
        )
        bps_by_m.append(bps)
    bic = [
        n * (np.log(r) + 1 - np.log(n) + np.log(2 * np.pi))
        + (k + 1) * (m + 1) * np.log(n)
        for m, r in enumerate(rss_by_m)
    ]
    m_star = int(np.argmin(bic))
    res = StructuralBreakResult(
        test_type="Bai-Perron (global, BIC)",
        break_dates=list(bps_by_m[m_star]),
        f_stats=None,
        p_values=None,
        n_breaks=m_star,
        rss_full=rss_full,
        rss_segments=rss_by_m[m_star],
        bic=float(bic[m_star]),
        n_obs=n,
    )
    res.rss_by_breaks = np.array(rss_by_m)
    res.bic_by_breaks = np.array(bic)
    res.breaks_by_m = bps_by_m
    res.min_segment_size = h
    return res


def _bm_linear_crossing_pvalue(x: float) -> float:
    """P(sup_{0<t<=1} |W(t)| / (1 + 2t) > x) for standard Brownian motion W.

    Closed-form crossing probability of the linear boundaries
    ``+-x (1 + 2t)``, as evaluated by ``strucchange::pvalue.efp`` (Brownian
    motion, max functional; that function also switches to ``1 - 0.1465 x``
    below x = 0.3, reproduced here).
    """
    from scipy.stats import norm

    if x < 0.3:
        return float(1.0 - 0.1465 * x)
    p = 2.0 * (
        norm.sf(3.0 * x)
        + np.exp(-4.0 * x * x) * (norm.cdf(x) + norm.cdf(5.0 * x) - 1.0)
        - np.exp(-16.0 * x * x) * norm.sf(x)
    )
    return float(min(max(p, 0.0), 1.0))


@lru_cache(maxsize=32)
def _bde_boundary(alpha: float) -> float:
    """Boundary coefficient a with crossing probability alpha (5%: 0.9479)."""
    from scipy.optimize import brentq

    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0, 1)")
    return float(
        brentq(lambda a: _bm_linear_crossing_pvalue(a) - alpha, 0.3, 20.0, xtol=1e-14)
    )


def cusum_test(
    data: pd.DataFrame,
    y: str,
    x: Optional[List[str]] = None,
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
        Keys: ``'cusum'`` (standardised CUSUM path of the recursive
        residuals, ``strucchange::efp(type = "Rec-CUSUM")$process`` without
        its leading 0), ``'max_cusum'`` (its supremum in absolute value),
        ``'statistic'`` (``max_s |cusum_s| / (1 + 2 s / m)``, the statistic
        of ``strucchange::sctest`` and Stata ``estat sbcusum``),
        ``'p_value'`` (Brownian-motion crossing probability of that
        statistic), ``'boundary_coef'`` (``a`` solving crossing probability
        = ``alpha``; 0.9479 at 5%),
        ``'critical_value'`` (the Brown-Durbin-Evans crossing **boundary**,
        an array ``a * [1 + 2 s / (n - k)]`` that widens from ``a`` to ``3a``
        across the sample -- *not* a constant), ``'reject'`` (True if the path
        crosses that boundary anywhere), and ``'n_obs'``.

    Notes
    -----
    This is the recursive-residual CUSUM of Brown, Durbin & Evans (1975), as
    in R's ``strucchange::efp(type="Rec-CUSUM")``. Its boundary is linear in
    the recursion index with coefficient ``a`` (0.948 at the 5% level), not
    the constant sup\\|Brownian-bridge\\| value (1.358) that belongs to the
    OLS-CUSUM of Ploberger & Kramer (1992).

    References
    ----------
    brown1975techniques

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> n = 120
    >>> x = rng.normal(size=n)
    >>> y = 1.0 + 0.5 * x + rng.normal(scale=0.5, size=n)  # stable relation
    >>> df = pd.DataFrame({"y": y, "x": x})
    >>> res = sp.cusum_test(df, y="y", x=["x"])
    >>> sorted(res.keys())  # doctest: +NORMALIZE_WHITESPACE
    ['boundary_coef', 'critical_value', 'cusum', 'max_cusum', 'n_obs',
     'p_value', 'reject', 'statistic']
    >>> res["n_obs"]
    120
    >>> bool(res["reject"])
    False
    """
    y_data = data[y].values.astype(float)
    n = len(y_data)

    if x is not None:
        X_data = np.column_stack([np.ones(n), data[x].values.astype(float)])
    else:
        X_data = np.ones((n, 1))

    k = X_data.shape[1]

    # Recursive residuals
    rec_resid_values: list[float] = []
    for t in range(k, n):
        Xt = X_data[:t]
        yt = y_data[:t]
        bt = np.linalg.lstsq(Xt, yt, rcond=None)[0]
        pred = X_data[t] @ bt
        resid = y_data[t] - pred
        ft = 1 + X_data[t] @ np.linalg.solve(Xt.T @ Xt, X_data[t])
        rec_resid_values.append(float(resid / np.sqrt(max(ft, 1e-10))))

    rec_resid = np.array(rec_resid_values)
    sigma = np.std(rec_resid, ddof=1)
    m = rec_resid.size  # number of recursive residuals = n - k

    # Standardised CUSUM path of the recursive residuals (Brown, Durbin &
    # Evans 1975). W_t = (1/s) * sum_{j<=t} w_j; dividing by sqrt(m) puts it
    # in the units of the BDE boundary computed just below.
    cusum = np.cumsum(rec_resid) / (sigma * np.sqrt(m))
    max_cusum = float(np.max(np.abs(cusum)))

    # The Brown-Durbin-Evans crossing boundary is LINEAR in the recursion
    # index, not a constant:
    #   |W_t| > a * [sqrt(m) + 2 (t - k) / sqrt(m)]
    #   <=>  |cusum_t| > a * [1 + 2 s / m],   s = 1, ..., m.
    # The previous code compared ``max|cusum|`` against a constant 1.358; that
    # value is the sup|Brownian-bridge| critical value for the OLS-CUSUM
    # (Ploberger & Kramer 1992), a *different* test. Applied to the
    # recursive-residual CUSUM it over-rejected late breaks (true boundary
    # widens to 3a at the sample end) and under-rejected early ones. The BDE
    # boundary coefficient ``a`` solves P(crossing) = alpha (1.1430, 0.9479,
    # 0.8499 at 1%, 5%, 10%; hard-coded values rounded to three decimals,
    # used for any alpha, were replaced by this root in 1.28.x).
    a = _bde_boundary(float(alpha))
    s = np.arange(1, m + 1)
    boundary = a * (1.0 + 2.0 * s / m)
    # Test statistic: sup_s |cusum_s| / (1 + 2 s / m) (strucchange ``sctest``
    # statistic S, Stata ``estat sbcusum`` statistic), and its p-value from
    # the boundary-crossing probability of standard Brownian motion.
    stat = float(np.max(np.abs(cusum) / (1.0 + 2.0 * s / m)))

    return {
        "cusum": cusum,
        "max_cusum": max_cusum,
        # Per-recursion BDE crossing boundary (array). Replaces the old scalar
        # 1.358, which was the wrong (Brownian-bridge) critical value.
        "critical_value": boundary,
        "boundary_coef": a,
        "statistic": stat,
        "p_value": _bm_linear_crossing_pvalue(stat),
        "reject": bool(np.any(np.abs(cusum) > boundary)),
        "n_obs": n,
    }
