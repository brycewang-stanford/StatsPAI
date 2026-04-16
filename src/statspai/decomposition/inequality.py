"""
Inequality decomposition.

Supports:
- **Subgroup decomposition** (between / within) for additive inequality
  indices: Theil T, Theil L (MLD), GE(α), Atkinson, half-squared CV.
- **Gini subgroup decomposition** (Dagum 1997, Yitzhaki 1994) into
  between, within, and overlap.
- **Source decomposition** (Lerman-Yitzhaki 1985) of the Gini into
  contributions from income sources.
- **Shapley / Shorrocks (2013)** allocation of inequality to covariates
  via RIF regression + Shapley values.

References
----------
Shorrocks, A.F. (1984). "Inequality Decomposition by Population Subgroups."
*Econometrica*, 52(6), 1369-1385.

Shorrocks, A.F. (2013). "Decomposition Procedures for Distributional
Analysis: A Unified Framework Based on the Shapley Value." *Journal of
Economic Inequality*, 11, 99-126.

Dagum, C. (1997). "A New Approach to the Decomposition of the Gini
Income Inequality Ratio." *Empirical Economics*, 22, 515-531.

Lerman, R. & Yitzhaki, S. (1985). "Income Inequality Effects by Income
Source: A New Approach and Applications to the United States." *Review
of Economics and Statistics*, 67, 151-156.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from math import factorial
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from ._common import (
    add_constant,
    bootstrap_ci,
    prepare_frame,
    weighted_quantile,
    wls,
)


# ════════════════════════════════════════════════════════════════════════
# Inequality indices
# ════════════════════════════════════════════════════════════════════════

def _theil_t(y: np.ndarray, w: Optional[np.ndarray] = None) -> float:
    y = np.clip(y, 1e-12, None)
    if w is None:
        w = np.ones_like(y)
    mu = float(np.average(y, weights=w))
    return float(np.average((y / mu) * np.log(y / mu), weights=w))


def _theil_l(y: np.ndarray, w: Optional[np.ndarray] = None) -> float:
    y = np.clip(y, 1e-12, None)
    if w is None:
        w = np.ones_like(y)
    mu = float(np.average(y, weights=w))
    return float(np.log(mu) - np.average(np.log(y), weights=w))


def _ge_index(y: np.ndarray, alpha: float,
              w: Optional[np.ndarray] = None) -> float:
    """Generalised entropy GE(α)."""
    y = np.clip(y, 1e-12, None)
    if w is None:
        w = np.ones_like(y)
    mu = float(np.average(y, weights=w))
    if abs(alpha) < 1e-12:
        return _theil_l(y, w)
    if abs(alpha - 1) < 1e-12:
        return _theil_t(y, w)
    c = 1.0 / (alpha * (alpha - 1))
    return float(c * (np.average((y / mu) ** alpha, weights=w) - 1))


def _atkinson(y: np.ndarray, eps: float = 1.0,
              w: Optional[np.ndarray] = None) -> float:
    """Atkinson index for inequality aversion ε."""
    y = np.clip(y, 1e-12, None)
    if w is None:
        w = np.ones_like(y)
    mu = float(np.average(y, weights=w))
    if abs(eps - 1.0) < 1e-12:
        # A(1) = 1 - exp(E log y)/μ
        return float(1.0 - np.exp(np.average(np.log(y), weights=w)) / mu)
    # A(eps) = 1 - (E y^(1-eps))^(1/(1-eps)) / μ
    p = 1.0 - eps
    val = float(np.average(y ** p, weights=w))
    return float(1.0 - (val ** (1.0 / p)) / mu)


def _gini(y: np.ndarray, w: Optional[np.ndarray] = None) -> float:
    """Weighted Gini coefficient."""
    y = np.asarray(y, dtype=float)
    if w is None:
        w = np.ones_like(y)
    w = np.asarray(w, dtype=float)
    order = np.argsort(y)
    ys = y[order]
    ws = w[order]
    W = ws.sum()
    mu = np.average(ys, weights=ws)
    if mu <= 0:
        return float("nan")
    cum_w = np.cumsum(ws)
    F = (cum_w - 0.5 * ws) / W
    return float(2.0 * np.cov(ys, F, aweights=ws)[0, 1] / mu)


def _cv_squared_half(y: np.ndarray,
                     w: Optional[np.ndarray] = None) -> float:
    """Half squared coefficient of variation (GE(2))."""
    if w is None:
        w = np.ones_like(y)
    mu = float(np.average(y, weights=w))
    var = float(np.average((y - mu) ** 2, weights=w))
    return float(0.5 * var / (mu ** 2))


_INDEX_FN = {
    "theil_t": lambda y, w=None: _theil_t(y, w),
    "theil_l": lambda y, w=None: _theil_l(y, w),
    "mld": lambda y, w=None: _theil_l(y, w),
    "ge0": lambda y, w=None: _ge_index(y, 0.0, w),
    "ge1": lambda y, w=None: _ge_index(y, 1.0, w),
    "ge2": lambda y, w=None: _ge_index(y, 2.0, w),
    "atkinson": lambda y, w=None: _atkinson(y, 1.0, w),
    "gini": lambda y, w=None: _gini(y, w),
    "cv2": lambda y, w=None: _cv_squared_half(y, w),
}


def inequality_index(
    y: np.ndarray,
    index: str = "theil_t",
    weights: Optional[np.ndarray] = None,
    eps: float = 1.0,
    alpha: Optional[float] = None,
) -> float:
    """Compute a single inequality index."""
    y = np.asarray(y, dtype=float)
    w = np.ones_like(y) if weights is None else np.asarray(weights, dtype=float)
    if alpha is not None:
        return _ge_index(y, alpha, w)
    if index == "atkinson":
        return _atkinson(y, eps, w)
    if index in _INDEX_FN:
        return _INDEX_FN[index](y, w)
    raise ValueError(f"unknown index {index!r}")


# ════════════════════════════════════════════════════════════════════════
# Subgroup decomposition
# ════════════════════════════════════════════════════════════════════════

@dataclass
class SubgroupDecompResult:
    index: str
    total: float
    between: float
    within: float
    overlap: Optional[float]
    per_group: pd.DataFrame     # group_id, n, weight, mean, index_value, contribution

    def summary(self) -> str:
        lines = [
            "━" * 62,
            f"  Inequality Subgroup Decomposition — {self.index}",
            "━" * 62,
            f"  Total index:    {self.total: .4f}",
            f"  Between-group:  {self.between: .4f}  "
            f"({self.between / self.total * 100 if self.total != 0 else 0:.1f}%)",
            f"  Within-group:   {self.within: .4f}  "
            f"({self.within / self.total * 100 if self.total != 0 else 0:.1f}%)",
        ]
        if self.overlap is not None:
            lines.append(f"  Overlap:        {self.overlap: .4f}")
        lines.append("")
        lines.append("  Per-group:")
        lines.append(self.per_group.round(4).to_string(index=False))
        lines.append("━" * 62)
        text = "\n".join(lines)
        print(text)
        return text

    def plot(self, **kwargs):
        from .plots import inequality_subgroup_plot
        return inequality_subgroup_plot(self, **kwargs)

    def to_latex(self) -> str:
        return self.per_group.round(4).to_latex(index=False)

    def _repr_html_(self) -> str:
        return (f"<div><h3>Inequality Subgroup — {self.index}</h3>"
                f"<p>Total={self.total:.4f}, Between={self.between:.4f}, "
                f"Within={self.within:.4f}</p></div>")

    def __repr__(self) -> str:
        return (f"SubgroupDecompResult(index={self.index}, total={self.total:.4f}, "
                f"between={self.between:.4f}, within={self.within:.4f})")


def subgroup_decompose(
    data: pd.DataFrame,
    y: str,
    by: str,
    index: str = "theil_t",
    weights: Optional[Union[str, np.ndarray]] = None,
    eps: float = 1.0,
    alpha: Optional[float] = None,
) -> SubgroupDecompResult:
    """
    Subgroup decomposition (between / within) of an inequality index.

    Supported for additive GE family (theil_t, theil_l, mld, ge0, ge1,
    ge2, cv2, atkinson(ε=1)). Gini returns Dagum (1997) Gini_B / Gini_W /
    Gini_overlap.

    Parameters
    ----------
    data : pd.DataFrame
    y : str — outcome
    by : str — grouping variable
    index : str — inequality index name
    weights : str, array or None
    eps : float — Atkinson parameter
    alpha : float or None — GE parameter override
    """
    df, w = prepare_frame(data, [y, by], weights=weights)
    y_vec = df[y].to_numpy(dtype=float)
    groups = df[by].to_numpy()
    unique_g = np.unique(groups)

    total = inequality_index(y_vec, index=index, weights=w,
                             eps=eps, alpha=alpha)

    if index == "gini":
        return _gini_subgroup(y_vec, w, groups, total)

    # Additive decomposition (GE / Theil / MLD)
    per_rows = []
    within = 0.0
    mu_pool = float(np.average(y_vec, weights=w))
    W_pool = w.sum()
    for gi in unique_g:
        mask = groups == gi
        y_g = y_vec[mask]
        w_g = w[mask]
        W_g = w_g.sum()
        share_w = W_g / W_pool
        mu_g = float(np.average(y_g, weights=w_g))
        share_y = (W_g * mu_g) / (W_pool * mu_pool) if mu_pool != 0 else 0.0
        idx_g = inequality_index(y_g, index=index, weights=w_g,
                                 eps=eps, alpha=alpha)
        if index in ("theil_t", "ge1"):
            contrib = share_y * idx_g
        elif index in ("theil_l", "mld", "ge0"):
            contrib = share_w * idx_g
        elif index == "ge2" or index == "cv2":
            contrib = (share_y ** 2 / share_w) * idx_g if share_w > 0 else 0.0
        else:
            # General fallback
            contrib = share_w * idx_g
        within += contrib
        per_rows.append({
            "group": gi, "n": int(mask.sum()), "weight": W_g,
            "mean": mu_g, f"{index}_group": idx_g, "contribution": contrib,
        })
    between = total - within
    per_group = pd.DataFrame(per_rows)

    return SubgroupDecompResult(
        index=index, total=float(total), between=float(between),
        within=float(within), overlap=None, per_group=per_group,
    )


def _weighted_pairwise_mad(
    y_h: np.ndarray, w_h: np.ndarray,
    y_k: np.ndarray, w_k: np.ndarray,
) -> float:
    """
    Weighted mean absolute difference E_{w_h, w_k}[|Y_h − Y_k|].

    O((n_h + n_k) log(n_h + n_k)) via sorted-ECDF identity:

        E|X − Y| = 2 [ E_X[X · F_Y(X)] − E_X[X] · E_Y[Y · F_X(Y)/F_X(Y)] ... ]

    Closed form: sort by y; for each Y value compute fraction of the
    other sample strictly below (weighted), then use
      E|X−Y| = E_X[X (2 F_Y(X) − 1)] + E_Y[Y (1 − 2 F_X(Y))].
    This is O(n log n) in total.
    """
    if len(y_h) == 0 or len(y_k) == 0:
        return 0.0
    W_h = w_h.sum()
    W_k = w_k.sum()
    # Sort y_k for ECDF lookups; build weighted ECDF
    order_k = np.argsort(y_k)
    y_k_s = y_k[order_k]
    w_k_s = w_k[order_k]
    F_k = np.cumsum(w_k_s) / W_k   # weighted ECDF at each y_k_s
    # For each y_h[i], find F_Y(y_h[i]) = weighted share of y_k ≤ y_h[i]
    idx_h = np.searchsorted(y_k_s, y_h, side="right") - 1
    F_k_at_h = np.where(idx_h < 0, 0.0, F_k[np.clip(idx_h, 0, len(F_k) - 1)])
    # Symmetrically, F_X at each y_k
    order_h = np.argsort(y_h)
    y_h_s = y_h[order_h]
    w_h_s = w_h[order_h]
    F_h = np.cumsum(w_h_s) / W_h
    idx_k = np.searchsorted(y_h_s, y_k, side="right") - 1
    F_h_at_k = np.where(idx_k < 0, 0.0, F_h[np.clip(idx_k, 0, len(F_h) - 1)])
    # E_w|X - Y| identity:
    #   = E_h[y_h (2 F_k(y_h) − 1)] + E_k[y_k (1 − 2 F_h(y_k))]
    term_h = np.sum(w_h * y_h * (2.0 * F_k_at_h - 1.0)) / W_h
    term_k = np.sum(w_k * y_k * (1.0 - 2.0 * F_h_at_k)) / W_k
    return float(term_h + term_k)


def _gini_subgroup(
    y: np.ndarray, w: np.ndarray, groups: np.ndarray, total: float
) -> SubgroupDecompResult:
    """Dagum (1997) Gini subgroup decomposition with O(n log n) pairs."""
    unique_g = np.unique(groups)
    W = w.sum()
    mu = float(np.average(y, weights=w))

    # Per-group stats
    per_rows = []
    for gi in unique_g:
        mask = groups == gi
        y_g = y[mask]
        w_g = w[mask]
        W_g = w_g.sum()
        mu_g = float(np.average(y_g, weights=w_g))
        G_g = _gini(y_g, w_g)
        per_rows.append({
            "group": gi, "n": int(mask.sum()), "weight": W_g,
            "mean": mu_g, "gini_group": G_g,
            "contribution": (
                (W_g / W) * (W_g * mu_g / (W * mu)) * G_g if mu > 0 else 0.0
            ),
        })
    per_group = pd.DataFrame(per_rows)

    # Dagum's Gini_W = Σ_h (W_h/W)(W_h μ_h / (W μ)) G_h
    within = 0.0
    for _, row in per_group.iterrows():
        if mu > 0:
            within += (row["weight"] / W) \
                      * (row["weight"] * row["mean"] / (W * mu)) \
                      * row["gini_group"]

    # Dagum's Gross between:
    #   G_B = Σ_{h≠k} (W_h/W)(W_k/W) D_hk / (μ_h + μ_k)
    # where D_hk = E[|y_h − y_k|] is the weighted pairwise MAD.
    # Iterating unordered pairs once gives the correct total.
    between_gross = 0.0
    g_list = list(unique_g)
    for i_h, h in enumerate(g_list):
        for k in g_list[i_h + 1:]:
            mask_h = groups == h
            mask_k = groups == k
            y_h_arr, w_h_arr = y[mask_h], w[mask_h]
            y_k_arr, w_k_arr = y[mask_k], w[mask_k]
            mu_h = float(np.average(y_h_arr, weights=w_h_arr))
            mu_k = float(np.average(y_k_arr, weights=w_k_arr))
            if mu_h + mu_k <= 0:
                continue
            W_h = w_h_arr.sum() / W
            W_k = w_k_arr.sum() / W
            D_hk = _weighted_pairwise_mad(y_h_arr, w_h_arr, y_k_arr, w_k_arr)
            # Dagum uses 2 W_h W_k to count both orderings in the gross Gini
            between_gross += 2.0 * W_h * W_k * D_hk / (mu_h + mu_k)
    # Standard Gini is on [0, 1); Dagum defines total = within + between_gross
    # + transvariation (overlap). overlap = total − within − between_gross.
    overlap = total - within - between_gross

    return SubgroupDecompResult(
        index="gini", total=float(total), between=float(between_gross),
        within=float(within), overlap=float(overlap), per_group=per_group,
    )


# ════════════════════════════════════════════════════════════════════════
# Lerman-Yitzhaki source decomposition (Gini)
# ════════════════════════════════════════════════════════════════════════

@dataclass
class SourceDecompResult:
    total_gini: float
    sources: pd.DataFrame   # source, share, R, G, contribution

    def summary(self) -> str:
        lines = [
            "━" * 62,
            "  Gini Source Decomposition (Lerman-Yitzhaki 1985)",
            "━" * 62,
            f"  Total Gini: {self.total_gini: .4f}",
            "",
            self.sources.round(4).to_string(index=False),
            "━" * 62,
        ]
        text = "\n".join(lines)
        print(text)
        return text

    def plot(self, **kwargs):
        from .plots import detailed_waterfall
        return detailed_waterfall(
            self.sources, value_col="contribution",
            label_col="source",
            title="Gini Source Decomposition", **kwargs,
        )

    def to_latex(self) -> str:
        lines = [
            r"\begin{table}[htbp]", r"\centering",
            r"\caption{Gini Source Decomposition (Lerman-Yitzhaki 1985)}",
            r"\begin{tabular}{lcccc}", r"\toprule",
            r"Source & Share & $G_k$ & Gini corr. & Contribution \\",
            r"\midrule",
        ]
        for _, row in self.sources.iterrows():
            lines.append(
                f"{row['source']} & {row['share']:.4f} & "
                f"{row['gini_k']:.4f} & {row['gini_corr']:.4f} & "
                f"{row['contribution']:.4f} \\\\"
            )
        lines.extend([r"\midrule",
                      f"Total & 1.0000 & & & {self.total_gini:.4f} \\\\",
                      r"\bottomrule", r"\end{tabular}", r"\end{table}"])
        return "\n".join(lines)

    def _repr_html_(self) -> str:
        return (
            "<div style='font-family:monospace;'>"
            "<h3>Gini Source Decomposition</h3>"
            f"<p>Total Gini = {self.total_gini:.4f}</p>"
            + self.sources.round(4).to_html(index=False) + "</div>"
        )

    def __repr__(self) -> str:
        return f"SourceDecompResult(gini={self.total_gini:.4f}, "\
               f"n_sources={len(self.sources)})"


def source_decompose(
    data: pd.DataFrame,
    sources: Sequence[str],
    weights: Optional[Union[str, np.ndarray]] = None,
) -> SourceDecompResult:
    """
    Lerman-Yitzhaki (1985) Gini source decomposition.

    Total income = Σ sources. Each source's contribution is
        S_k · R_k · G_k  /  G_total
    where S_k is its share of total mean, R_k the Gini correlation with
    total rank, G_k its own Gini.
    """
    cols = list(sources)
    df, w = prepare_frame(data, cols, weights=weights)
    y_total = df[cols].sum(axis=1).to_numpy(dtype=float)
    order = np.argsort(y_total)
    ranks = np.empty_like(order, dtype=float)
    # Weighted ranks (fractional)
    W = w.sum()
    cum = np.cumsum(w[order])
    ranks[order] = (cum - 0.5 * w[order]) / W

    G_total = _gini(y_total, w)
    mu_total = float(np.average(y_total, weights=w))

    rows = []
    total_contrib = 0.0
    for s in cols:
        y_s = df[s].to_numpy(dtype=float)
        mu_s = float(np.average(y_s, weights=w))
        share = mu_s / mu_total if mu_total > 0 else 0.0
        G_s = _gini(y_s, w)
        # Gini correlation: R_k = cov(y_s, F_total) / cov(y_s, F_own)
        F_own = np.empty_like(y_s)
        order_s = np.argsort(y_s)
        cum_s = np.cumsum(w[order_s])
        F_own[order_s] = (cum_s - 0.5 * w[order_s]) / W
        cov_total = float(np.cov(y_s, ranks, aweights=w)[0, 1])
        cov_own = float(np.cov(y_s, F_own, aweights=w)[0, 1])
        R = cov_total / cov_own if cov_own != 0 else 0.0
        contrib = share * R * G_s
        total_contrib += contrib
        rows.append({
            "source": s, "share": share, "gini_k": G_s,
            "gini_corr": R, "contribution": contrib,
            "pct_of_gini": contrib / G_total * 100 if G_total != 0 else 0.0,
        })

    return SourceDecompResult(
        total_gini=float(G_total),
        sources=pd.DataFrame(rows),
    )


# ════════════════════════════════════════════════════════════════════════
# Shapley decomposition (Shorrocks 2013)
# ════════════════════════════════════════════════════════════════════════

@dataclass
class ShapleyInequalityResult:
    index: str
    total: float
    shapley: pd.DataFrame     # variable, contribution, pct

    def summary(self) -> str:
        lines = [
            "━" * 62,
            f"  Shapley Inequality Decomposition — {self.index}",
            "━" * 62,
            f"  Total index: {self.total: .4f}",
            "",
            self.shapley.round(4).to_string(index=False),
            "━" * 62,
        ]
        text = "\n".join(lines)
        print(text)
        return text

    def plot(self, **kwargs):
        from .plots import detailed_waterfall
        return detailed_waterfall(self.shapley, value_col="contribution",
                                  label_col="variable", **kwargs)

    def to_latex(self) -> str:
        lines = [
            r"\begin{table}[htbp]", r"\centering",
            f"\\caption{{Shapley Inequality Decomposition — {self.index}}}",
            r"\begin{tabular}{lcc}", r"\toprule",
            r"Variable & Contribution & \% of total \\", r"\midrule",
        ]
        for _, row in self.shapley.iterrows():
            lines.append(
                f"{row['variable']} & {row['contribution']:.4f} & "
                f"{row['pct_of_total']:.1f}\\% \\\\"
            )
        lines.extend([r"\midrule",
                      f"Total ({self.index}) & {self.total:.4f} & 100.0\\% \\\\",
                      r"\bottomrule", r"\end{tabular}", r"\end{table}"])
        return "\n".join(lines)

    def _repr_html_(self) -> str:
        return (
            "<div style='font-family:monospace;'>"
            f"<h3>Shapley Inequality — {self.index}</h3>"
            f"<p>Total = {self.total:.4f}</p>"
            + self.shapley.round(4).to_html(index=False) + "</div>"
        )

    def __repr__(self) -> str:
        return f"ShapleyInequalityResult(index={self.index}, "\
               f"total={self.total:.4f})"


def shapley_inequality(
    data: pd.DataFrame,
    y: str,
    x: Sequence[str],
    index: str = "theil_t",
    weights: Optional[Union[str, np.ndarray]] = None,
) -> ShapleyInequalityResult:
    """
    Shorrocks-Shapley decomposition of an inequality index across
    covariates.

    For each subset S ⊆ covariates, compute the *predicted* outcome
    ŷ_S = X_S · β_S (OLS) and evaluate index I(ŷ_S).  The marginal
    contribution of variable j to I is averaged over all orderings
    yielding its Shapley value φ_j.

    Parameters
    ----------
    data, y, x : as usual
    index : str
    weights : str, array or None

    Notes
    -----
    Combinatorial cost: O(2^|x|).  For |x| ≤ 10 this is fine; for
    larger x the function warns and uses a random permutation sampler.
    """
    cols = [y] + list(x)
    df, w = prepare_frame(data, cols, weights=weights)
    y_vec = df[y].to_numpy(dtype=float)
    X_raw = df[list(x)].to_numpy(dtype=float)
    k = len(x)

    idx_fn = _INDEX_FN.get(index)
    if idx_fn is None:
        raise ValueError(f"unknown index {index!r}")
    I_total = float(idx_fn(y_vec, w))

    def pred(subset_idx: Tuple[int, ...]) -> np.ndarray:
        if not subset_idx:
            return np.full_like(y_vec, y_vec.mean())
        X_s = add_constant(X_raw[:, list(subset_idx)])
        beta, _, _ = wls(y_vec, X_s, w=w)
        return X_s @ beta

    # Enumerate all subsets up to |x|≤10; else sample
    if k <= 10:
        all_subsets: List[Tuple[int, ...]] = []
        for r in range(k + 1):
            for c in combinations(range(k), r):
                all_subsets.append(c)
        v = {s: float(idx_fn(pred(s), w)) for s in all_subsets}
        # Shapley values
        contributions = np.zeros(k)
        for j in range(k):
            for s in all_subsets:
                if j in s:
                    continue
                s_with_j = tuple(sorted(s + (j,)))
                marg = v[s_with_j] - v[s]
                weight = (factorial(len(s)) * factorial(k - len(s) - 1)
                          / factorial(k))
                contributions[j] += weight * marg
    else:
        import warnings
        warnings.warn(f"|x|={k} > 10; using 500 random permutations for "
                      "Shapley approximation.")
        rng = np.random.default_rng(12345)
        contributions = np.zeros(k)
        for _ in range(500):
            perm = rng.permutation(k)
            v_prev = float(idx_fn(pred(()), w))
            subset: Tuple[int, ...] = ()
            for j in perm:
                subset = tuple(sorted(subset + (int(j),)))
                v_new = float(idx_fn(pred(subset), w))
                contributions[int(j)] += v_new - v_prev
                v_prev = v_new
        contributions /= 500.0

    df_sh = pd.DataFrame({
        "variable": list(x),
        "contribution": contributions,
        "pct_of_total": contributions / I_total * 100
        if I_total != 0 else np.zeros_like(contributions),
    })
    return ShapleyInequalityResult(index=index, total=I_total, shapley=df_sh)
