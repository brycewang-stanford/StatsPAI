"""Regression-table columns for scalar effects and causal forests.

``sp.regtable(ols, cf)`` used to print a blank column for a fitted
``sp.causal_forest`` (it has an empty ``params`` vector) and refused the
``ScalarEffect`` that ``cf.ate()`` / ``cf.att()`` return. A forest's
headline is its doubly robust average effect with the forest's own
standard error, so the column shows that one row: ``ATE`` for the forest
itself, ``ATT`` (or any other estimand) for a ``ScalarEffect``.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd


def is_causal_forest(result: Any) -> bool:
    from ..forest.causal_forest import CausalForest

    return isinstance(result, CausalForest)


def is_scalar_effect(result: Any) -> bool:
    from ..core.results import ScalarEffect

    return isinstance(result, ScalarEffect)


def scalar_effect_data(
    effect: Any, *, depvar: str = "", n: Optional[int] = None
) -> Any:
    """``_ModelData`` with one row, named by the effect's estimand.

    The interval is the one the effect carries (normal reference, as the
    forest computes it); ``ci_df`` is infinite so a table at another
    ``alpha`` recomputes it on the same normal reference.
    """
    from .estimates import _ModelData

    name = str(getattr(effect, "estimand", "") or "effect")
    est = float(effect)
    se = np.nan if effect.se is None else float(effect.se)
    pval = np.nan if effect.pvalue is None else float(effect.pvalue)
    lo, hi = effect.ci if effect.ci is not None else (np.nan, np.nan)
    detail = dict(getattr(effect, "detail", {}) or {})
    if n is None and detail.get("n") is not None:
        n = int(detail["n"])
    with np.errstate(divide="ignore", invalid="ignore"):
        t = est / se if np.isfinite(se) and se > 0 else np.nan
    stats = {"N": n}
    if detail.get("n_clusters"):
        stats["Clusters"] = int(detail["n_clusters"])
    return _ModelData(
        pd.Series({name: est}),
        pd.Series({name: se}),
        pd.Series({name: t}),
        pd.Series({name: pval}),
        pd.Series({name: float(lo)}),
        pd.Series({name: float(hi)}),
        stats,
        depvar,
        df_resid=None,
        ci_alpha=float(detail.get("alpha", 0.05)),
        ci_df=pd.Series({name: np.inf}),
    )


def causal_forest_data(forest: Any) -> Any:
    """The forest's ATE column: ``forest.ate()`` on its training sample."""
    info = getattr(forest, "data_info", {}) or {}
    n = info.get("nobs")
    return scalar_effect_data(
        forest.ate(), depvar="causal_forest", n=None if n is None else int(n)
    )
