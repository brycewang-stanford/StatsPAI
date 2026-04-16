"""
Kitagawa (1955) and Das Gupta (1993) demographic decomposition.

Decompose a difference in an aggregate rate / mean between two
populations into **rate effect** (difference in category-specific
rates) and **composition effect** (difference in population
weights / subgroup shares).

Supports both two-factor (Kitagawa) and multi-factor (Das Gupta)
decomposition.

References
----------
Kitagawa, E.M. (1955). "Components of a Difference Between Two Rates."
*JASA*, 50(272), 1168-1194.

Das Gupta, P. (1993). "Standardization and Decomposition of Rates: A
User's Manual." U.S. Bureau of the Census, CDS P23-186.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from typing import Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd


# ════════════════════════════════════════════════════════════════════════
# Result
# ════════════════════════════════════════════════════════════════════════

@dataclass
class KitagawaResult:
    rate_a: float
    rate_b: float
    gap: float
    rate_effect: float
    composition_effect: float
    interaction: float
    per_cell: pd.DataFrame   # category, share_a, share_b, rate_a, rate_b,
                             # rate_contrib, comp_contrib
    method: str = "kitagawa"

    def summary(self) -> str:
        lines = [
            "━" * 62,
            f"  Kitagawa Decomposition — {self.method}",
            "━" * 62,
            f"  Rate A: {self.rate_a: .4f}",
            f"  Rate B: {self.rate_b: .4f}",
            f"  Gap (A − B):           {self.gap: .4f}",
            f"  Rate effect:           {self.rate_effect: .4f}",
            f"  Composition effect:    {self.composition_effect: .4f}",
            f"  Interaction:           {self.interaction: .4f}",
            "",
            "  Per-category:",
            self.per_cell.round(4).to_string(index=False),
            "━" * 62,
        ]
        text = "\n".join(lines)
        print(text)
        return text

    def plot(self, **kwargs):
        from .plots import detailed_waterfall
        return detailed_waterfall(
            self.per_cell,
            value_col="rate_contrib",
            label_col="category",
            **kwargs,
        )

    def to_latex(self) -> str:
        return self.per_cell.round(4).to_latex(index=False)

    def _repr_html_(self) -> str:
        return (f"<div><h3>Kitagawa Decomposition</h3>"
                f"<p>Gap={self.gap:.4f}, Rate={self.rate_effect:.4f}, "
                f"Composition={self.composition_effect:.4f}</p></div>")

    def __repr__(self) -> str:
        return (f"KitagawaResult(gap={self.gap:.4f}, "
                f"rate_effect={self.rate_effect:.4f}, "
                f"composition_effect={self.composition_effect:.4f})")


# ════════════════════════════════════════════════════════════════════════
# Kitagawa (two-factor)
# ════════════════════════════════════════════════════════════════════════

def kitagawa_decompose(
    data: pd.DataFrame,
    rate: str,
    group: str,
    by: Union[str, Sequence[str]],
    weights: Optional[str] = None,
    normalize: str = "symmetric",
) -> KitagawaResult:
    """
    Kitagawa (1955) two-factor rate decomposition.

    Parameters
    ----------
    data : pd.DataFrame
        Tidy data. Either individual-level (aggregated internally) or
        pre-aggregated cell-level with columns: `group`, `by`, `rate`, optional
        `weights` (population size in each cell).
    rate : str
        Column holding the category-specific rate (or 0/1 outcome at the
        individual level).
    group : str
        Binary group indicator.
    by : str or list of str
        Category variable(s) defining cells.
    weights : str or None
        Cell population weights. If None, each row treated as
        individual-level data (weight = 1).
    normalize : {'symmetric', 'a', 'b'}
        - 'a': rate effect evaluated at A's composition
        - 'b': rate effect evaluated at B's composition
        - 'symmetric': average (default)
    """
    by_cols = [by] if isinstance(by, str) else list(by)
    use_cols = [rate, group] + by_cols + ([weights] if weights else [])
    df = data[use_cols].dropna().copy()

    # If individual-level, aggregate by cell
    if weights is None:
        agg = (
            df.groupby([group] + by_cols)
              .agg(rate=(rate, 'mean'), pop=(rate, 'size'))
              .reset_index()
        )
        rate_col = "rate"
        pop_col = "pop"
    else:
        # Pre-aggregated
        agg = df.rename(columns={rate: "rate", weights: "pop"})
        rate_col = "rate"
        pop_col = "pop"

    # Separate into groups
    a_df = agg[agg[group] == 0].set_index(by_cols)[[rate_col, pop_col]]
    b_df = agg[agg[group] == 1].set_index(by_cols)[[rate_col, pop_col]]

    # Align on union of categories
    all_cats = sorted(set(a_df.index).union(b_df.index))
    a = a_df.reindex(all_cats).fillna(0)
    b = b_df.reindex(all_cats).fillna(0)

    pop_a_total = a[pop_col].sum()
    pop_b_total = b[pop_col].sum()
    if pop_a_total <= 0 or pop_b_total <= 0:
        raise ValueError("Zero population in one of the groups.")

    share_a = a[pop_col].to_numpy() / pop_a_total
    share_b = b[pop_col].to_numpy() / pop_b_total
    rate_a_cells = a[rate_col].to_numpy()
    rate_b_cells = b[rate_col].to_numpy()

    overall_a = float(np.sum(share_a * rate_a_cells))
    overall_b = float(np.sum(share_b * rate_b_cells))
    gap = overall_a - overall_b

    # Kitagawa decomposition
    # Rate effect at A's composition: Σ share_a (rate_a - rate_b)
    re_a = np.sum(share_a * (rate_a_cells - rate_b_cells))
    # Rate effect at B's composition
    re_b = np.sum(share_b * (rate_a_cells - rate_b_cells))
    # Composition effect at A's rates
    ce_a = np.sum((share_a - share_b) * rate_a_cells)
    # Composition effect at B's rates
    ce_b = np.sum((share_a - share_b) * rate_b_cells)

    if normalize == "a":
        rate_effect = float(re_a)
        composition_effect = float(ce_b)
    elif normalize == "b":
        rate_effect = float(re_b)
        composition_effect = float(ce_a)
    else:
        rate_effect = 0.5 * (re_a + re_b)
        composition_effect = 0.5 * (ce_a + ce_b)

    interaction = gap - rate_effect - composition_effect

    per_rows = []
    for i, cat in enumerate(all_cats):
        cat_name = str(cat) if not isinstance(cat, tuple) else " × ".join(map(str, cat))
        per_rows.append({
            "category": cat_name,
            "share_a": share_a[i], "share_b": share_b[i],
            "rate_a": rate_a_cells[i], "rate_b": rate_b_cells[i],
            "rate_contrib": 0.5 * (share_a[i] + share_b[i])
                            * (rate_a_cells[i] - rate_b_cells[i]),
            "comp_contrib": 0.5 * (share_a[i] - share_b[i])
                            * (rate_a_cells[i] + rate_b_cells[i]),
        })
    per_cell = pd.DataFrame(per_rows)

    return KitagawaResult(
        rate_a=overall_a, rate_b=overall_b, gap=gap,
        rate_effect=rate_effect, composition_effect=composition_effect,
        interaction=float(interaction), per_cell=per_cell,
        method="kitagawa",
    )


# ════════════════════════════════════════════════════════════════════════
# Das Gupta (1993) multi-factor decomposition
# ════════════════════════════════════════════════════════════════════════

@dataclass
class DasGuptaResult:
    rate_a: float
    rate_b: float
    gap: float
    factor_effects: pd.DataFrame   # factor, effect
    method: str = "das_gupta"

    def summary(self) -> str:
        lines = [
            "━" * 62,
            "  Das Gupta Multi-Factor Decomposition",
            "━" * 62,
            f"  Aggregate A: {self.rate_a: .4f}",
            f"  Aggregate B: {self.rate_b: .4f}",
            f"  Gap:         {self.gap: .4f}",
            "",
            self.factor_effects.round(4).to_string(index=False),
            "━" * 62,
        ]
        text = "\n".join(lines)
        print(text)
        return text

    def plot(self, **kwargs):
        from .plots import detailed_waterfall
        return detailed_waterfall(self.factor_effects, value_col="effect",
                                  label_col="factor", **kwargs)

    def to_latex(self) -> str:
        lines = [
            r"\begin{table}[htbp]", r"\centering",
            r"\caption{Das Gupta Multi-Factor Decomposition}",
            r"\begin{tabular}{lcc}", r"\toprule",
            r"Factor & Effect & \% of gap \\", r"\midrule",
        ]
        for _, row in self.factor_effects.iterrows():
            lines.append(
                f"{row['factor']} & {row['effect']:.4f} & "
                f"{row['pct_of_gap']:.1f}\\% \\\\"
            )
        lines.extend([r"\midrule",
                      f"Total gap & {self.gap:.4f} & 100.0\\% \\\\",
                      r"\bottomrule", r"\end{tabular}", r"\end{table}"])
        return "\n".join(lines)

    def _repr_html_(self) -> str:
        return (
            "<div style='font-family:monospace;'>"
            f"<h3>Das Gupta Multi-Factor Decomposition</h3>"
            f"<p>Aggregate A = {self.rate_a:.4f}, B = {self.rate_b:.4f}, "
            f"gap = {self.gap:.4f}</p>"
            + self.factor_effects.round(4).to_html(index=False)
            + "</div>"
        )

    def __repr__(self) -> str:
        return f"DasGuptaResult(gap={self.gap:.4f}, n_factors={len(self.factor_effects)})"


def das_gupta(
    data_a: pd.DataFrame,
    data_b: pd.DataFrame,
    factor_names: Sequence[str],
) -> DasGuptaResult:
    """
    Das Gupta (1993) multi-factor decomposition.

    Decomposes the difference in a product-form aggregate into each
    factor's contribution using symmetric averaging across all possible
    orderings.

    Parameters
    ----------
    data_a, data_b : pd.DataFrame with the same factor columns.
        Each row contributes the factor value. The aggregate for each
        group is computed as Σ_i ∏_f factor_{f,i}.

        For single-row DataFrames (one population, no stratification) the
        aggregate is simply ∏_f factor_f.
    factor_names : list of factor column names.

    Notes
    -----
    Assumes: rate = f_1 * f_2 * ... * f_m  (aggregate product form).
    For additive forms use `kitagawa_decompose`.
    """
    factors = list(factor_names)
    m = len(factors)
    if m == 0:
        raise ValueError("Need ≥1 factor.")

    # Collapse each dataframe to a single vector of factor means
    va = data_a[factors].mean().to_numpy(dtype=float)
    vb = data_b[factors].mean().to_numpy(dtype=float)

    prod_a = float(np.prod(va))
    prod_b = float(np.prod(vb))
    gap = prod_a - prod_b

    # Das Gupta effect for factor j:
    # Δ_j = mean over all orderings ρ:
    #   ∏_{k such that σ(k) < σ(j)} vA_k · (vA_j - vB_j) · ∏_{k such that σ(k) > σ(j)} vB_k
    effects = np.zeros(m)
    count = 0
    for perm in permutations(range(m)):
        count += 1
        # place factors in order perm[0], perm[1], ..., perm[m-1]
        # all factors preceding perm[i] take vA; all following take vB
        # perm[i] takes the diff
        for rank, j in enumerate(perm):
            pre = perm[:rank]
            post = perm[rank + 1:]
            contrib = np.prod(va[list(pre)]) if pre else 1.0
            contrib *= (va[j] - vb[j])
            contrib *= np.prod(vb[list(post)]) if post else 1.0
            effects[j] += contrib
    effects = effects / count

    df_fx = pd.DataFrame({
        "factor": factors,
        "effect": effects,
        "pct_of_gap": effects / gap * 100 if gap != 0 else np.zeros(m),
    })

    return DasGuptaResult(
        rate_a=prod_a, rate_b=prod_b, gap=gap,
        factor_effects=df_fx, method="das_gupta",
    )
