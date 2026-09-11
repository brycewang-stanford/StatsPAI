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
*JASA*, 50(272), 1168-1194. [@kitagawa1955components]

Das Gupta, P. (1993). "Standardization and Decomposition of Rates: A
User's Manual." U.S. Bureau of the Census, CDS P23-186.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from typing import Any, ClassVar, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from ._results import DecompResultMixin

# ════════════════════════════════════════════════════════════════════════
# Result
# ════════════════════════════════════════════════════════════════════════


@dataclass
class KitagawaResult(DecompResultMixin):
    method_name: ClassVar[str] = "Kitagawa Decomposition"
    bib_keys: ClassVar[Tuple[str, ...]] = (
        "kitagawa1955components",
        "kroger2021kitagawa",
        "oaxaca2025meets",
    )

    rate_a: float
    rate_b: float
    gap: float
    rate_effect: float
    composition_effect: float
    interaction: float
    per_cell: pd.DataFrame  # category, share_a, share_b, rate_a, rate_b,
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

    def plot(self, **kwargs: Any) -> Any:
        from .plots import detailed_waterfall

        return detailed_waterfall(
            self.per_cell,
            value_col="rate_contrib",
            label_col="category",
            **kwargs,
        )

    def to_latex(self) -> str:
        latex: str = self.per_cell.round(4).to_latex(index=False)
        return latex

    def _repr_html_(self) -> str:
        return (
            f"<div><h3>Kitagawa Decomposition</h3>"
            f"<p>Gap={self.gap:.4f}, Rate={self.rate_effect:.4f}, "
            f"Composition={self.composition_effect:.4f}</p></div>"
        )

    def __repr__(self) -> str:
        return (
            f"KitagawaResult(gap={self.gap:.4f}, "
            f"rate_effect={self.rate_effect:.4f}, "
            f"composition_effect={self.composition_effect:.4f})"
        )


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

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np
    >>> import pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> rows = []
    >>> for grp, base in [(0, 0.2), (1, 0.3)]:
    ...     for age, share in [('young', 0.5), ('mid', 0.3), ('old', 0.2)]:
    ...         n = int(400 * share) + (10 if grp == 1 else 0)
    ...         rate = base + {'young': 0.0, 'mid': 0.05, 'old': 0.1}[age]
    ...         for _ in range(n):
    ...             rows.append({'group': grp, 'age': age,
    ...                          'y': int(rng.random() < rate)})
    >>> df = pd.DataFrame(rows)
    >>> res = sp.kitagawa_decompose(df, rate='y', group='group', by='age')
    >>> # Gap = rate effect + composition effect + interaction (exact identity)
    >>> float(round(res.gap - (res.rate_effect + res.composition_effect
    ...                        + res.interaction), 10))
    0.0

    References
    ----------
    kitagawa1955components
    """
    by_cols = [by] if isinstance(by, str) else list(by)
    use_cols = [rate, group] + by_cols + ([weights] if weights else [])
    df = data[use_cols].dropna().copy()

    # If individual-level, aggregate by cell
    if weights is None:
        agg = (
            df.groupby([group] + by_cols)
            .agg(rate=(rate, "mean"), pop=(rate, "size"))
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
        per_rows.append(
            {
                "category": cat_name,
                "share_a": share_a[i],
                "share_b": share_b[i],
                "rate_a": rate_a_cells[i],
                "rate_b": rate_b_cells[i],
                "rate_contrib": 0.5
                * (share_a[i] + share_b[i])
                * (rate_a_cells[i] - rate_b_cells[i]),
                "comp_contrib": 0.5
                * (share_a[i] - share_b[i])
                * (rate_a_cells[i] + rate_b_cells[i]),
            }
        )
    per_cell = pd.DataFrame(per_rows)

    return KitagawaResult(
        rate_a=overall_a,
        rate_b=overall_b,
        gap=gap,
        rate_effect=rate_effect,
        composition_effect=composition_effect,
        interaction=float(interaction),
        per_cell=per_cell,
        method="kitagawa",
    )


# ════════════════════════════════════════════════════════════════════════
# Das Gupta (1993) multi-factor decomposition
# ════════════════════════════════════════════════════════════════════════


@dataclass
class DasGuptaResult(DecompResultMixin):
    method_name: ClassVar[str] = "Das Gupta Multi-Factor Decomposition"
    bib_keys: ClassVar[Tuple[str, ...]] = ("dasgupta1993standardization",)

    rate_a: float
    rate_b: float
    gap: float
    factor_effects: pd.DataFrame  # factor, effect
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

    def plot(self, **kwargs: Any) -> Any:
        from .plots import detailed_waterfall

        return detailed_waterfall(
            self.factor_effects, value_col="effect", label_col="factor", **kwargs
        )

    def to_latex(self) -> str:
        lines = [
            r"\begin{table}[htbp]",
            r"\centering",
            r"\caption{Das Gupta Multi-Factor Decomposition}",
            r"\begin{tabular}{lcc}",
            r"\toprule",
            r"Factor & Effect & \% of gap \\",
            r"\midrule",
        ]
        for _, row in self.factor_effects.iterrows():
            lines.append(
                f"{row['factor']} & {row['effect']:.4f} & "
                f"{row['pct_of_gap']:.1f}\\% \\\\"
            )
        lines.extend(
            [
                r"\midrule",
                f"Total gap & {self.gap:.4f} & 100.0\\% \\\\",
                r"\bottomrule",
                r"\end{tabular}",
                r"\end{table}",
            ]
        )
        return "\n".join(lines)

    def _repr_html_(self) -> str:
        html: str = self.factor_effects.round(4).to_html(index=False)
        return (
            "<div style='font-family:monospace;'>"
            f"<h3>Das Gupta Multi-Factor Decomposition</h3>"
            f"<p>Aggregate A = {self.rate_a:.4f}, B = {self.rate_b:.4f}, "
            f"gap = {self.gap:.4f}</p>" + html + "</div>"
        )

    def __repr__(self) -> str:
        return (
            f"DasGuptaResult(gap={self.gap:.4f}, n_factors={len(self.factor_effects)})"
        )


def das_gupta(
    data_a: pd.DataFrame,
    data_b: pd.DataFrame,
    factor_names: Sequence[str],
    by: Optional[Union[str, Sequence[str]]] = None,
) -> DasGuptaResult:
    """
    Das Gupta (1993) multi-factor decomposition.

    Decomposes the difference in a product-form aggregate into each
    factor's contribution using symmetric averaging across all possible
    orderings.

    Parameters
    ----------
    data_a, data_b : pd.DataFrame with the same factor columns.
        One row per stratum (e.g. age group). The aggregate for each
        population is ``R = sum_i prod_f factor_{f,i}``; with a single row
        it is simply ``prod_f factor_f``.
    factor_names : list of factor column names.
    by : str or list of str, optional
        Column(s) identifying the stratum of each row, used to pair the
        rows of ``data_a`` and ``data_b``. Both frames must contain the same
        strata. Without ``by`` the rows are paired by position, which
        requires equal lengths.

    Notes
    -----
    Assumes the aggregate is a sum over strata of products of factors,
    ``R = sum_i f_{1,i} f_{2,i} ... f_{m,i}``. Each stratum's product is
    decomposed by averaging over all orderings of the factors -- Das
    Gupta's symmetric formula -- and the per-stratum effects are summed,
    which is what ``DasGuptR::dgnpop`` does with
    ``ratefunction = "sum(f1*f2*...)"``. For additive forms use
    `kitagawa_decompose`.

    .. versionchanged:: 1.28.0
       With more than one row per population the aggregate was computed as
       the product of the factor *means* rather than the sum over strata of
       the factor products the docstring stated, so the decomposition of
       stratified data was of a different quantity. On Das Gupta's
       Table 6.5 data (four factors, six age groups; 1963 vs 1968) factor
       A's share of the gap was 0% where ``DasGuptR`` gives 36.8%, and
       factor D's +333% where it gives -52.5%. Single-row inputs are
       unchanged. Rows are now paired by ``by`` or by position, and
       unequal lengths raise.

    Examples
    --------
    >>> import statspai as sp
    >>> import pandas as pd
    >>> # Crude birth rate = fertility x share_women (product form).
    >>> data_a = pd.DataFrame({'fertility': [0.10], 'share_women': [0.50]})
    >>> data_b = pd.DataFrame({'fertility': [0.08], 'share_women': [0.55]})
    >>> res = sp.das_gupta(data_a, data_b,
    ...                    factor_names=['fertility', 'share_women'])
    >>> # Factor effects sum to the total gap (exact identity)
    >>> float(round(res.gap - res.factor_effects['effect'].sum(), 12))
    0.0

    References
    ----------
    dasgupta1993standardization
    """
    factors = list(factor_names)
    m = len(factors)
    if m == 0:
        raise ValueError("Need ≥1 factor.")

    if by is not None:
        by_cols = [by] if isinstance(by, str) else list(by)
        ka = data_a[by_cols].apply(tuple, axis=1)
        kb = data_b[by_cols].apply(tuple, axis=1)
        if ka.duplicated().any() or kb.duplicated().any():
            raise ValueError("das_gupta: `by` must identify one row per stratum.")
        if set(ka) != set(kb):
            raise ValueError(
                "das_gupta: data_a and data_b must contain the same strata; "
                f"only in A: {sorted(set(ka) - set(kb))}, "
                f"only in B: {sorted(set(kb) - set(ka))}."
            )
        fa = data_a.set_index(ka)[factors]
        fb = data_b.set_index(kb)[factors].reindex(fa.index)
    else:
        if len(data_a) != len(data_b):
            raise ValueError(
                "das_gupta: data_a and data_b have different numbers of rows "
                f"({len(data_a)} vs {len(data_b)}); pass `by=` to pair strata."
            )
        fa, fb = data_a[factors], data_b[factors]
    va = fa.to_numpy(dtype=float)  # (strata, factors)
    vb = fb.to_numpy(dtype=float)

    prod_a = float(np.sum(np.prod(va, axis=1)))
    prod_b = float(np.sum(np.prod(vb, axis=1)))
    gap = prod_a - prod_b

    # Das Gupta effect for factor j, stratum by stratum, then summed:
    # Δ_j = mean over all orderings ρ of
    #   ∏_{k before j} vA_k · (vA_j − vB_j) · ∏_{k after j} vB_k
    effects: np.ndarray = np.zeros(m)
    count = 0
    for perm in permutations(range(m)):
        count += 1
        for rank, j in enumerate(perm):
            pre = list(perm[:rank])
            post = list(perm[rank + 1 :])
            contrib = np.prod(va[:, pre], axis=1) if pre else np.ones(len(va))
            contrib = contrib * (va[:, j] - vb[:, j])
            if post:
                contrib = contrib * np.prod(vb[:, post], axis=1)
            effects[j] += float(np.sum(contrib))
    effects = effects / count

    df_fx = pd.DataFrame(
        {
            "factor": factors,
            "effect": effects,
            "pct_of_gap": effects / gap * 100 if gap != 0 else np.zeros(m),
        }
    )

    return DasGuptaResult(
        rate_a=prod_a,
        rate_b=prod_b,
        gap=gap,
        factor_effects=df_fx,
        method="das_gupta",
    )
