"""One-line dashboard summaries of fitted StatsPAI results.

``sp.brief(result)`` and ``result.brief()`` return a single-line
status string under ~120 characters: enough to scan a list of
results in an agent-orchestrated workflow without paying the token
cost of a full ``to_dict(detail="agent")`` payload per item.

Format
------

::

    [METHOD] estimand=ATT  est=0.412 (se=0.087)  95% CI [0.241, 0.583]  ***  N=2,000  ⚠ pretrend

Columns:

* ``[METHOD]`` — method label (truncated to 24 chars)
* ``estimand=`` — ATT / ATE / LATE / etc.
* ``est=`` — point estimate to 3 sig figs
* ``(se=...)`` — standard error
* ``95% CI [..., ...]`` — confidence interval at the result's alpha
* ``***`` / ``**`` / ``*`` — significance stars (omitted if p ≥ 0.10)
* ``N=`` — sample size with thousands separator
* ``⚠ ...`` — first ``violations()`` flag at error severity, if any

Distinct from siblings:

* :meth:`CausalResult.summary` — multi-line prose for humans (KB-scale).
* :meth:`CausalResult.to_dict` (with ``detail="minimal"``) — JSON payload
  ~ 300 chars; ``brief()`` is ~ 100 chars and human-scannable, intended
  for agent dashboards rather than tool-result payloads.
"""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
import pandas as pd

from ..workflow._degradation import record_degradation

_MAX_METHOD_LEN = 24


def _stars(pvalue: Optional[float]) -> str:
    """Significance markers — silent above α=0.10 to avoid clutter."""
    if pvalue is None or not np.isfinite(pvalue):
        return ""
    if pvalue < 0.01:
        return "***"
    if pvalue < 0.05:
        return "**"
    if pvalue < 0.10:
        return "*"
    return ""


def _fmt(x: Any, fmt: str = "{:.3g}") -> str:
    if x is None:
        return "—"
    try:
        f = float(x)
    except (TypeError, ValueError):
        return str(x)
    if not np.isfinite(f):
        return "—"
    return fmt.format(f)


def _truncate_method(name: str) -> str:
    if len(name) > _MAX_METHOD_LEN:
        return name[: _MAX_METHOD_LEN - 1] + "…"
    return name


def _violation_flag(result: Any) -> str:
    """First error-severity violation, if any. Empty string otherwise."""
    try:
        viols = result.violations() or []
    except Exception as exc:
        # ``violations()`` not implementing on a result is normal for
        # legacy types — skip silently in that common case.  But if it
        # *exists* and *raises*, surface it: the brief line is meant
        # to flag pretrend / overlap / etc. and we just lost that
        # signal for this row.
        if hasattr(result, "violations"):
            record_degradation(
                None,
                section="brief: violations() flag extraction",
                exc=exc,
                detail=f"result_type={type(result).__name__}",
            )
        return ""
    for v in viols:
        if v.get("severity") == "error":
            return f"  ⚠ {v.get('test', 'violation')}"
    # Fall back to the first warning-severity flag — agents still want
    # to see "borderline pre-trend" in a dashboard, just dimmer.
    for v in viols:
        if v.get("severity") == "warning":
            return f"  ⚠ {v.get('test', 'violation')}?"
    return ""


#: Coefficients listed when a regression designates no focal term.
_BRIEF_TERMS = 3


def brief(result: Any) -> str:
    """Render a one-line status summary of a fitted result.

    Parameters
    ----------
    result : CausalResult or EconometricResults (or any object
        exposing ``method`` / ``estimate`` / ``se`` / ``pvalue`` /
        ``ci`` / ``n_obs`` attributes).

    Returns
    -------
    str
        A single-line status string under ~120 characters. Intended
        for agent dashboards / multi-result comparisons. JSON-safe
        (it's just a string).

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np
    >>> import pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> n = 200
    >>> unit = np.repeat(np.arange(n // 2), 2)
    >>> t = np.tile([0, 1], n // 2)
    >>> treated = np.where(unit < (n // 4), 1, 0)
    >>> post = (t == 1).astype(int)
    >>> y = 1.0 + 0.8 * post + 1.5 * treated * post + rng.normal(size=n)
    >>> df = pd.DataFrame({"y": y, "treated": treated, "t": t, "unit": unit})
    >>> r = sp.did(df, y='y', treat='treated', time='t')
    >>> line = sp.brief(r)
    >>> isinstance(line, str) and "estimand=ATT" in line
    True

    See Also
    --------
    CausalResult.summary :
        Multi-line prose summary for humans.
    CausalResult.to_dict :
        Full JSON payload at minimal/standard/agent detail levels.
    """
    # Method label: CausalResult exposes ``.method`` directly;
    # EconometricResults stores it under ``model_info["method"]`` /
    # ``model_info["model_type"]``. Walk both shapes.
    method_raw = getattr(result, "method", None)
    if not method_raw:
        mi = getattr(result, "model_info", None) or {}
        method_raw = mi.get("method") or mi.get("model_type") or "?"
    method = _truncate_method(str(method_raw))

    # Causal-style result
    if hasattr(result, "estimand") and hasattr(result, "estimate"):
        estimand = getattr(result, "estimand", "")
        est = _fmt(getattr(result, "estimate", None))
        se = _fmt(getattr(result, "se", None))
        pv = getattr(result, "pvalue", None)
        try:
            pv_f = float(pv) if pv is not None else None
        except (TypeError, ValueError):
            pv_f = None
        stars = _stars(pv_f)

        ci = getattr(result, "ci", None)
        ci_str = ""
        if (
            ci is not None
            and not isinstance(ci, (pd.Series, pd.DataFrame))
            and hasattr(ci, "__len__")
            and len(ci) == 2
        ):
            alpha = getattr(result, "alpha", 0.05) or 0.05
            try:
                # ``:g`` keeps 97.5 as 97.5 (rounding printed "98% CI")
                pct = f"{100 * (1 - float(alpha)):.10g}"
            except (TypeError, ValueError):
                pct = "95"
            ci_str = f"  {pct}% CI [{_fmt(ci[0])}, {_fmt(ci[1])}]"

        n_obs = getattr(result, "n_obs", None)
        n_str = (
            f"  N={int(n_obs):,}"
            if n_obs is not None
            and isinstance(n_obs, (int, float, np.integer, np.floating))
            and np.isfinite(n_obs)
            else ""
        )

        viol = _violation_flag(result)
        stars_str = f"  {stars}" if stars else ""

        return (
            f"[{method}]"
            f"  estimand={estimand}"
            f"  est={est} (se={se})"
            f"{ci_str}"
            f"{stars_str}"
            f"{n_str}"
            f"{viol}"
        )

    # Econometric-style result (regression with multiple coefficients)
    params = getattr(result, "params", None)
    n_obs = None
    if hasattr(result, "data_info") and isinstance(result.data_info, dict):
        n_obs = result.data_info.get("nobs")

    if params is not None and hasattr(params, "index"):
        n_terms = len(params)
        # The coefficient the fit designates (its treatment, the single
        # endogenous regressor, the only regressor), else the leading
        # coefficients in model order. Until 2026-11-13 this line showed
        # the term with the smallest p-value under the label "best":
        # a focal term chosen by significance.
        focal = None
        focal_fn = getattr(result, "_focal_term", None)
        if callable(focal_fn):
            try:
                focal = focal_fn()[0]
            except Exception as exc:
                record_degradation(
                    None,
                    section="brief: focal term",
                    exc=exc,
                    detail=f"result_type={type(result).__name__}",
                )
        shown: list = []
        try:
            pvals = getattr(result, "pvalues", None)
            for i, name in enumerate(params.index):
                label = str(name)
                if label.lower() in ("intercept", "const", "_cons"):
                    continue
                if focal is not None and label != focal:
                    continue
                coef = float(params.iloc[i])
                pv = None
                if pvals is not None:
                    try:
                        pv = float(
                            pvals.iloc[i] if hasattr(pvals, "iloc") else pvals[i]
                        )
                    except (TypeError, ValueError, IndexError, KeyError):
                        pv = None
                    if pv is not None and not np.isfinite(pv):
                        pv = None
                shown.append((label, coef, pv))
        except Exception as exc:
            record_degradation(
                None,
                section="brief: coefficient extraction",
                exc=exc,
                detail=f"result_type={type(result).__name__}",
            )
            shown = []

        n_str = (
            f"  N={int(n_obs):,}"
            if n_obs is not None
            and isinstance(n_obs, (int, float, np.integer, np.floating))
            and np.isfinite(n_obs)
            else ""
        )

        def _term(label: str, coef: float, pv: Any) -> str:
            text = f"{label}={_fmt(coef)}"
            if pv is not None:
                text += f" (p={_fmt(pv, '{:.3g}')})"
            return text

        if shown and (focal is not None or len(shown) == 1):
            # a designated term, or the only regressor
            label, coef, pv = shown[0]
            stars = _stars(pv) if pv is not None else ""
            stars_str = f"  {stars}" if stars else ""
            return (
                f"[{method}]  k={n_terms}  {_term(label, coef, pv)}"
                f"{stars_str}{n_str}"
            )
        if shown:
            head = ", ".join(_term(*t) for t in shown[:_BRIEF_TERMS])
            more = len(shown) - _BRIEF_TERMS
            more_str = f", +{more} more" if more > 0 else ""
            return f"[{method}]  k={n_terms}  {head}{more_str}  (no focal term){n_str}"
        return f"[{method}]  k={n_terms}{n_str}"

    # Last resort
    return f"[{method}]  (no scalar summary available)"


__all__ = ["brief"]
