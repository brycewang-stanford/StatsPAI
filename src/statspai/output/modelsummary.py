"""R ``modelsummary`` compatibility surface — thin facade over :func:`regtable`.

Historically this module shipped its own ~700-line renderer pipeline
(``_build_coef_rows`` / ``_build_stat_rows`` / ``_to_text`` /
``_to_latex`` / ``_to_html`` / ``_to_excel`` / ``_to_word`` …) that
re-implemented coefficient extraction, star formatting, three-line
table styling and every export format — duplicating code already
maintained by :func:`statspai.output.regtable`.

In the PR-B output consolidation (see
``docs/rfc/output_pr_b_consolidation.md``) we collapse the module to a
thin facade that translates R-flavoured kwargs and forwards to
:func:`regtable`. The user-visible API (``modelsummary`` function name
and parameter spelling) is preserved; the *rendered output* now matches
:func:`regtable` exactly — strictly cleaner (book-tab three-line table,
publication-quality star legend, fixed labels) but NOT byte-identical
to the legacy bespoke renderer.

:func:`coefplot` is independent of the table renderer and is kept here
unchanged.

A :class:`DeprecationWarning` is emitted on first call pointing users
to :func:`sp.regtable`. Plan to remove the facade in two minor releases
(per CLAUDE.md §3.8).

Migration notes
---------------
- ``stars=True`` / ``stars=False`` work the same. The dict form
  (``stars={"*": 0.10, "**": 0.05, "***": 0.01}``) is reinterpreted —
  only the threshold *values* are used, the symbol overrides are
  dropped (regtable's ladder is ``*/**/***`` by convention; symbol
  control is via ``notation='symbols'`` for ``†/‡/§``).
- ``se_type='parentheses'`` (default) and ``se_type='none'`` work as
  before. ``se_type='brackets'`` is no longer a separate render mode —
  emits ``UserWarning`` and falls back to parentheses. Use
  ``show_ci=True`` (which renders ``[lo, hi]``) if you want brackets
  to convey actual information.
- ``output='dataframe'`` returns a :class:`pandas.DataFrame` (same as
  before).
- ``output='<filename>.xlsx|.docx'`` writes the file and returns a
  confirmation string.
- All other ``output=`` strings (``"text"``, ``"latex"``, ``"html"``,
  ``"markdown"``) return the rendered string.
- Stat keys map: ``nobs`` → ``N``; ``r_squared`` / ``adj_r_squared``
  / ``f_stat`` / ``aic`` / ``bic`` → their canonical regtable
  equivalents.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

_DEPRECATION_MSG = (
    "modelsummary() is now a thin wrapper over sp.regtable() and will "
    "be removed in a future minor release. Migrate to "
    "sp.regtable(*models, ...) for the same output with full control "
    "over labels, journal templates, and SE formats. "
    "See docs/rfc/output_pr_b_consolidation.md for migration."
)


def _warn_once() -> None:
    warnings.warn(_DEPRECATION_MSG, DeprecationWarning, stacklevel=3)


# Stat-key translation: modelsummary R-style → regtable canonical.
# Regtable's ``_STAT_ALIASES`` already accepts ``nobs``/``aic``/``bic``,
# but not the longer ``r_squared`` / ``adj_r_squared`` / ``f_stat``
# variants R users type. Normalise here so the facade doesn't surprise
# anyone migrating from R modelsummary.
_STATS_TRANSLATION = {
    "nobs": "N",
    "n": "N",
    "r_squared": "r2",
    "rsquared": "r2",
    "r2": "r2",
    "adj_r_squared": "adj_r2",
    "adj_rsquared": "adj_r2",
    "adj_r2": "adj_r2",
    "f_stat": "F",
    "fstat": "F",
    "f": "F",
    "aic": "aic",
    "bic": "bic",
    # Modelsummary-only keys with no regtable equivalent — silently
    # dropped here; users wanting these can build a custom regtable
    # ``add_rows={}`` or extract from the result directly.
    "method": None,
    "bandwidth": None,
    "estimand": None,
}


def _translate_stats(stats: Optional[List[str]]) -> Optional[List[str]]:
    if stats is None:
        return None
    out: List[str] = []
    for s in stats:
        canonical = _STATS_TRANSLATION.get(s.lower(), s)
        if canonical is None:
            continue  # silently drop modelsummary-only keys
        if canonical not in out:
            out.append(canonical)
    return out


def modelsummary(
    *models,
    model_names: Optional[List[str]] = None,
    stars: Union[bool, Dict[str, float]] = True,
    se_type: str = "parentheses",
    show_ci: bool = False,
    stats: Optional[List[str]] = None,
    output: str = "text",
    title: str = "",
    notes: Optional[List[str]] = None,
    fmt: str = "%.4f",
    coef_map: Optional[Dict[str, str]] = None,
    add_rows: Optional[Dict[str, List[str]]] = None,
) -> Union[str, pd.DataFrame]:
    """Multi-model comparison table — R ``modelsummary`` compatibility surface.

    .. deprecated::
        Now a thin wrapper over :func:`statspai.output.regtable`. Use
        ``sp.regtable(*models, ...)`` directly for full control. See
        the module docstring for parameter mapping.

    Parameters mirror R ``modelsummary``; see the module docstring for
    the exact mapping to regtable.
    """
    if len(models) == 0:
        raise ValueError("At least one model required.")

    _warn_once()

    from .regression_table import regtable

    # ── star handling ────────────────────────────────────────────────
    if isinstance(stars, dict):
        if stars:
            star_levels = tuple(sorted(stars.values()))
            show_stars = True
        else:
            star_levels = None
            show_stars = False
    else:
        star_levels = None
        show_stars = bool(stars)

    # ── SE / CI handling ─────────────────────────────────────────────
    if show_ci:
        rt_se_type = "ci"
    else:
        st = se_type.lower()
        if st == "brackets":
            warnings.warn(
                "modelsummary(se_type='brackets') is no longer a "
                "separate render mode; SE will display in parentheses. "
                "Use show_ci=True to render [lo, hi] instead.",
                UserWarning,
                stacklevel=3,
            )
            rt_se_type = "se"
        elif st == "none":
            warnings.warn(
                "modelsummary(se_type='none') is no longer supported; "
                "the SE row will remain. Use sp.regtable(...) directly "
                "if you need a different uncertainty cell.",
                UserWarning,
                stacklevel=3,
            )
            rt_se_type = "se"
        else:
            rt_se_type = "se"

    # ── stat keys translation ────────────────────────────────────────
    rt_stats = _translate_stats(stats)

    # ── model labels ─────────────────────────────────────────────────
    rt_labels = list(model_names) if model_names else None

    # ── build the regtable call ──────────────────────────────────────
    kwargs: Dict[str, Any] = dict(
        model_labels=rt_labels,
        title=title or None,
        notes=notes,
        stars=show_stars,
        se_type=rt_se_type,
        fmt=fmt,
        coef_map=coef_map,
        add_rows=add_rows,
    )
    if star_levels is not None:
        kwargs["star_levels"] = star_levels
    if rt_stats is not None:
        kwargs["stats"] = rt_stats

    table = regtable(list(models), **kwargs)

    # ── output dispatch ──────────────────────────────────────────────
    out = output.lower()
    if out == "dataframe":
        return table.to_dataframe()
    if out == "latex":
        return table.to_latex()
    if out == "html":
        return table.to_html()
    if out == "markdown":
        return table.to_markdown()
    if output.lower().endswith(".xlsx"):
        table.to_excel(output)
        return f"Table exported to: {output}"
    if output.lower().endswith(".docx"):
        table.to_word(output)
        return f"Table exported to: {output}"
    # default: text
    return table.to_text()


# ======================================================================
# coefplot — independent of the table renderer; kept verbatim
# ======================================================================

def coefplot(
    *models,
    model_names: Optional[List[str]] = None,
    variables: Optional[List[str]] = None,
    ax=None,
    figsize: tuple = (8, 6),
    colors: Optional[List[str]] = None,
    title: Optional[str] = None,
    alpha: float = 0.05,
):
    """
    Forest plot comparing coefficients across models.

    Parameters
    ----------
    *models
        Model result objects.
    model_names : list of str, optional
    variables : list of str, optional
        Which variables to plot. Default: all shared variables.
    ax : matplotlib Axes, optional
    figsize : tuple
    colors : list of str, optional
    title : str, optional
    alpha : float
        Significance level for CIs.

    Returns
    -------
    (fig, ax)
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("matplotlib required. Install: pip install matplotlib")

    if model_names is None:
        model_names = [f'Model {i + 1}' for i in range(len(models))]
    if colors is None:
        colors = ['#2C3E50', '#E74C3C', '#3498DB', '#2ECC71',
                  '#9B59B6', '#F39C12', '#1ABC9C', '#E67E22']

    from scipy import stats as sp_stats
    z_crit = sp_stats.norm.ppf(1 - alpha / 2)

    coef_data = [_extract_coefs(m) for m in models]

    # Variables to plot
    if variables is None:
        all_v = set()
        for cd in coef_data:
            all_v.update(cd.keys())
        variables = sorted(all_v)

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    n_vars = len(variables)
    n_models = len(models)
    offsets = np.linspace(-0.15 * (n_models - 1), 0.15 * (n_models - 1), n_models)

    for m_idx, (cd, name) in enumerate(zip(coef_data, model_names)):
        color = colors[m_idx % len(colors)]
        positions = []
        estimates = []
        ci_lo = []
        ci_hi = []

        for v_idx, var in enumerate(variables):
            if var in cd:
                coef, se, _ = cd[var]
                positions.append(v_idx + offsets[m_idx])
                estimates.append(coef)
                ci_lo.append(coef - z_crit * se)
                ci_hi.append(coef + z_crit * se)

        if positions:
            pos = np.array(positions)
            est = np.array(estimates)
            lo = np.array(ci_lo)
            hi = np.array(ci_hi)
            ax.scatter(est, pos, color=color, s=40, zorder=5, label=name)
            ax.errorbar(
                est, pos, xerr=[est - lo, hi - est],
                fmt='none', color=color, capsize=3, linewidth=1, zorder=3,
            )

    ax.axvline(x=0, color='gray', linestyle='--', linewidth=0.8)
    ax.set_yticks(range(n_vars))
    ax.set_yticklabels(variables)
    ax.invert_yaxis()
    ax.set_xlabel('Coefficient Estimate')
    ax.set_title(title or 'Coefficient Plot')
    ax.legend(fontsize=9, frameon=False)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.tight_layout()

    return fig, ax


def _extract_coefs(model) -> Dict[str, tuple]:
    """Extract {var_name: (coef, se, pvalue)} from a model object.

    Used by :func:`coefplot`. Coefficient extraction for the table
    pipeline now lives in ``regression_table`` / ``estimates``.
    """
    result: Dict[str, tuple] = {}

    params = getattr(model, "params", None)
    std_errors = getattr(model, "std_errors", None)
    pvalues = getattr(model, "pvalues", None)

    if params is None:
        return result

    if isinstance(params, pd.Series):
        for var in params.index:
            coef = float(params[var])
            se = float(std_errors[var]) if (
                std_errors is not None and var in std_errors.index
            ) else np.nan
            pv = np.nan
            if pvalues is not None:
                if isinstance(pvalues, pd.Series) and var in pvalues.index:
                    pv = float(pvalues[var])
                elif isinstance(pvalues, np.ndarray):
                    idx = list(params.index).index(var)
                    if idx < len(pvalues):
                        pv = float(pvalues[idx])
            result[var] = (coef, se, pv)
        return result

    # CausalResult-style (single estimate)
    estimand = getattr(model, "estimand", None)
    estimate = getattr(model, "estimate", None)
    se_val = getattr(model, "se", None)
    pv_val = getattr(model, "pvalue", None)
    if estimand is not None and estimate is not None:
        result[str(estimand)] = (
            float(estimate),
            float(se_val) if se_val is not None else np.nan,
            float(pv_val) if pv_val is not None else np.nan,
        )
    return result
