"""``sp.fast.etable`` — fixest::etable-style side-by-side regression tables.

Phase 8 deliverable. Produces a single DataFrame (and optional LaTeX /
HTML / Markdown renderings) from multiple fitted models so users can
drop a publication-ready table into a paper without per-model rewriting.

Accepts any object that exposes:

- ``coef()``     — pd.Series of point estimates
- ``se()``       — pd.Series of standard errors
- ``n_obs`` *or* ``n`` *or* ``nobs``  — observation count
- (optional) ``r_squared`` / ``log_likelihood`` / ``deviance``

That covers ``sp.fast.fepois.FePoisResult`` and pyfixest's fitted
objects directly; for ``sp.feols`` results we wrap ``params`` /
``std_errors`` into the ``coef()`` / ``se()`` shape on the fly.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# Adapter: extract the bits we need from arbitrary result objects
# ---------------------------------------------------------------------------

def _coef_series(fit: Any) -> pd.Series:
    if hasattr(fit, "coef") and callable(fit.coef):
        out = fit.coef()
        if isinstance(out, pd.Series):
            return out
    if hasattr(fit, "params") and isinstance(fit.params, pd.Series):
        return fit.params
    if hasattr(fit, "coef_vec"):
        names = getattr(fit, "coef_names", None) or list(range(len(fit.coef_vec)))
        return pd.Series(fit.coef_vec, index=names)
    raise AttributeError(
        f"cannot extract coefficients from {type(fit).__name__}"
    )


def _se_series(fit: Any) -> pd.Series:
    if hasattr(fit, "se") and callable(fit.se):
        out = fit.se()
        if isinstance(out, pd.Series):
            return out
    for attr in ("std_errors", "bse"):
        v = getattr(fit, attr, None)
        if isinstance(v, pd.Series):
            return v
    if hasattr(fit, "vcov_matrix") and hasattr(fit, "coef_names"):
        return pd.Series(np.sqrt(np.diag(fit.vcov_matrix)),
                         index=fit.coef_names)
    raise AttributeError(
        f"cannot extract standard errors from {type(fit).__name__}"
    )


def _n_obs(fit: Any) -> Optional[int]:
    for attr in ("n_obs", "n_kept", "nobs", "n", "n_observations"):
        v = getattr(fit, attr, None)
        if isinstance(v, (int, np.integer)):
            return int(v)
    return None


def _df_residual(fit: Any) -> Optional[int]:
    """Best-effort residual-DOF lookup; ``None`` if the fit doesn't expose it."""
    for attr in ("df_residual", "df_resid"):
        v = getattr(fit, attr, None)
        if isinstance(v, (int, np.integer)):
            v = int(v)
            if v > 0:
                return v
    return None


def _maybe(fit: Any, attr: str) -> Optional[float]:
    v = getattr(fit, attr, None)
    if isinstance(v, (int, float, np.floating, np.integer)):
        return float(v)
    return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def etable(
    *fits: Any,
    names: Optional[Sequence[str]] = None,
    digits: int = 3,
    keep: Optional[Sequence[str]] = None,
    drop: Optional[Sequence[str]] = None,
    format: str = "dataframe",
    se_format: str = "paren",
    stars: bool = True,
) -> Union[pd.DataFrame, str]:
    """Stack multiple fitted models into a single comparison table.

    Parameters
    ----------
    *fits :
        One or more fitted model results.
    names : list of str, optional
        Column names for each model. Default: ``model_1, model_2, ...``.
    digits : int
        Decimal precision for coefficients and standard errors.
    keep, drop : list of str, optional
        Variable filters. ``keep`` retains only the named coefficients;
        ``drop`` removes them.
    format : {"dataframe", "latex", "html", "markdown"}
        Output format.
    se_format : {"paren", "below"}
        Whether SEs go in parentheses on the same row or as a separate
        below-row. ``"below"`` is the journal-paper standard.
    stars : bool
        Add significance stars (``*`` p<0.10, ``**`` p<0.05, ``***`` p<0.01)
        based on ``coef / se`` z-statistics.

    Returns
    -------
    DataFrame (default) or str (latex / html / markdown).
    """
    if not fits:
        raise ValueError("etable: pass at least one fitted model")
    if format not in ("dataframe", "latex", "html", "markdown"):
        raise ValueError(f"format={format!r}; expected dataframe|latex|html|markdown")
    if se_format not in ("paren", "below"):
        raise ValueError(f"se_format={se_format!r}; expected paren|below")

    if names is None:
        names = [f"({i+1})" for i in range(len(fits))]
    if len(names) != len(fits):
        raise ValueError(f"len(names)={len(names)} but len(fits)={len(fits)}")

    coefs: List[pd.Series] = [_coef_series(f) for f in fits]
    ses:   List[pd.Series] = [_se_series(f) for f in fits]
    n_obs = [_n_obs(f) for f in fits]
    df_resid = [_df_residual(f) for f in fits]

    # Star thresholds. Fall back to Normal-z when df is unavailable; use
    # Student-t two-sided critical values otherwise (matches ``stargazer`` /
    # fixest output by default).
    from scipy import stats as _stats
    threshold_cache: Dict[Optional[int], Tuple[float, float, float]] = {}

    def _stars_for(fit_idx: int) -> Tuple[float, float, float]:
        df = df_resid[fit_idx]
        if df in threshold_cache:
            return threshold_cache[df]
        if df is None:
            t10, t5, t1 = 1.645, 1.960, 2.576    # Normal-z fallback
        else:
            # Two-sided t critical values: ppf(1 - alpha/2, df)
            t10 = float(_stats.t.ppf(1 - 0.10 / 2, df))
            t5 = float(_stats.t.ppf(1 - 0.05 / 2, df))
            t1 = float(_stats.t.ppf(1 - 0.01 / 2, df))
        threshold_cache[df] = (t10, t5, t1)
        return threshold_cache[df]

    # Union of variable names across all models (preserve order of first occurrence)
    vars_all: List[str] = []
    seen = set()
    for c in coefs:
        for v in c.index:
            if v not in seen:
                vars_all.append(v)
                seen.add(v)
    if keep is not None:
        keep_set = set(keep)
        vars_all = [v for v in vars_all if v in keep_set]
    if drop is not None:
        drop_set = set(drop)
        vars_all = [v for v in vars_all if v not in drop_set]

    fmt_num = "{:." + str(digits) + "f}"

    # Build the table row-by-row
    rows: List[List[str]] = []
    row_index: List[str] = []

    for v in vars_all:
        coef_row = []
        se_row = []
        for fit_idx, (c, s) in enumerate(zip(coefs, ses)):
            if v in c.index:
                est = float(c[v])
                stderr = float(s[v]) if v in s.index else float("nan")
                star = ""
                if stars and stderr > 0:
                    z = abs(est / stderr)
                    t10, t5, t1 = _stars_for(fit_idx)
                    if z > t1:
                        star = "***"
                    elif z > t5:
                        star = "**"
                    elif z > t10:
                        star = "*"
                est_str = fmt_num.format(est) + star
                se_str = "(" + fmt_num.format(stderr) + ")"
            else:
                est_str = ""
                se_str = ""
            coef_row.append(est_str)
            se_row.append(se_str)

        if se_format == "paren":
            rows.append([a + " " + b if a else "" for a, b in zip(coef_row, se_row)])
            row_index.append(v)
        else:  # "below"
            rows.append(coef_row)
            row_index.append(v)
            rows.append(se_row)
            row_index.append("")

    # Footer: N
    rows.append([str(n) if n is not None else "" for n in n_obs])
    row_index.append("N")

    # Other footer rows if available across all models
    for label, attr in [("R²", "r_squared"), ("Log-Lik.", "log_likelihood")]:
        vals = [_maybe(f, attr) for f in fits]
        if any(v is not None for v in vals):
            rows.append([fmt_num.format(v) if v is not None else "" for v in vals])
            row_index.append(label)

    df = pd.DataFrame(rows, index=row_index, columns=list(names))

    if format == "dataframe":
        return df
    if format == "latex":
        return df.to_latex(escape=False)
    if format == "html":
        return df.to_html()
    return df.to_markdown()


__all__ = ["etable"]
