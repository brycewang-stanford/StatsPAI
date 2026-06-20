"""Auto-race entry points: sp.auto_did / sp.auto_iv.

Single-function entry points that run multiple estimators side by side
and return a leaderboard + a recommended winner.  The pattern mirrors
the existing :func:`statspai.metalearners.auto_cate` but for the
identification-strategy families (DiD, IV).

These are *new* differentiating APIs documented in the 2026-04-20 blog
post but not implemented until now.  The design goals are:

* One-line call — no per-estimator argument juggling
* Sensible defaults — defaults match the canonical recipe (CS / SA / BJS
  for DiD, 2SLS / LIML / JIVE for IV)
* Leaderboard first — users see all candidates, then pick a winner
* Graceful degradation — if one candidate crashes, the others still run

The underlying estimators are unchanged; this file is pure
orchestration + formatting logic.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from .exceptions import MethodIncompatibility

__all__ = [
    "auto_did",
    "auto_iv",
    "AutoDIDResult",
    "AutoIVResult",
]


def _validate_probability(value: Any, *, name: str, context: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise MethodIncompatibility(
            f"{context}: {name} must be a finite number in (0, 1)"
        ) from exc
    if not np.isfinite(parsed) or not (0.0 < parsed < 1.0):
        raise MethodIncompatibility(
            f"{context}: {name} must be a finite number in (0, 1)"
        )
    return parsed


def _as_string_list(
    value: Any,
    *,
    name: str,
    context: str,
    allow_none: bool = False,
) -> List[str]:
    if value is None:
        if allow_none:
            return []
        raise MethodIncompatibility(f"{context}: {name} must not be empty")
    if isinstance(value, str):
        items = [value]
    else:
        try:
            items = list(value)
        except TypeError as exc:
            raise MethodIncompatibility(
                f"{context}: {name} must be a string or sequence of strings"
            ) from exc
    if not items and not allow_none:
        raise MethodIncompatibility(f"{context}: {name} must not be empty")
    bad = [item for item in items if not isinstance(item, str) or not item]
    if bad:
        raise MethodIncompatibility(
            f"{context}: {name} entries must be non-empty strings"
        )
    return list(items)


def _normalize_methods(
    methods: Optional[List[str]],
    *,
    default: List[str],
    valid: set[str],
    context: str,
) -> List[str]:
    raw = default if methods is None else list(methods)
    if not raw:
        raise MethodIncompatibility(f"{context}: methods must not be empty")
    out = [str(m).lower() for m in raw]
    bad = [m for m in out if m not in valid]
    if bad:
        raise MethodIncompatibility(
            f"{context}: unknown methods {bad}; valid = {sorted(valid)}"
        )
    return out


def _require_columns(data: pd.DataFrame, columns: List[str], *, context: str) -> None:
    if not isinstance(data, pd.DataFrame):
        raise MethodIncompatibility(f"{context}: data must be a pandas DataFrame")
    needed = list(dict.fromkeys(columns))
    missing = [col for col in needed if col not in data.columns]
    if missing:
        raise MethodIncompatibility(
            f"{context}: data is missing required columns {missing}"
        )


def _jsonable(value: Any) -> Any:
    """Convert common NumPy/Pandas values to strict JSON primitives."""
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        value_float = float(value)
        return value_float if np.isfinite(value_float) else None
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, pd.DataFrame):
        return _jsonable(value.to_dict(orient="records"))
    if isinstance(value, pd.Series):
        return _jsonable(value.to_dict())
    if isinstance(value, np.ndarray):
        return _jsonable(value.tolist())
    if isinstance(value, dict):
        return {str(key): _jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return str(value)


def _jsonable_dict(obj: Dict[str, Any]) -> Dict[str, Any]:
    """``_jsonable`` of a dict, typed as ``Dict[str, Any]`` for callers."""
    payload: Dict[str, Any] = _jsonable(obj)
    return payload


def _candidate_status(candidates: Dict[str, Any]) -> List[Dict[str, Any]]:
    """JSON-safe status records for raw candidate result objects."""
    out: List[Dict[str, Any]] = []
    for method, candidate in candidates.items():
        if isinstance(candidate, Exception):
            out.append(
                {
                    "method": method,
                    "ok": False,
                    "error_type": type(candidate).__name__,
                    "message": str(candidate),
                }
            )
        else:
            out.append(
                {
                    "method": method,
                    "ok": True,
                    "result_type": type(candidate).__name__,
                }
            )
    return out


# =====================================================================
# Auto-DiD
# =====================================================================


@dataclass
class AutoDIDResult:
    """Leaderboard + winner from :func:`auto_did`.

    Attributes
    ----------
    leaderboard : pd.DataFrame
        One row per successful candidate, columns:
        ``method, estimate, std_error, ci_lower, ci_upper, n_obs, notes``.
        Sorted by the order in which candidates were requested.
    winner : CausalResult
        The recommended primary estimate (median-of-candidates by default).
    candidates : dict
        Raw ``CausalResult`` (or ``Exception``) for every requested method.
    selection_rule : str
        How the winner was chosen (``'median'`` / ``'first_success'`` / ...).

    Examples
    --------
    >>> import statspai as sp
    >>> df = sp.dgp_did(n_units=40, n_periods=6, staggered=True,
    ...                 n_groups=2, effect=0.5, seed=2026)
    >>> res = sp.auto_did(df, y="y", g="first_treat", t="time", i="unit")
    >>> isinstance(res, sp.AutoDIDResult)
    True
    >>> list(res.leaderboard.columns)
    ['method', 'estimate', 'std_error', 'ci_lower', 'ci_upper', 'n_obs', 'notes']
    >>> print(res.summary())  # doctest: +SKIP
    """

    leaderboard: pd.DataFrame
    winner: Any
    candidates: Dict[str, Any]
    selection_rule: str = "median"

    def summary(self) -> str:
        lines = [
            "auto_did: staggered-DiD method race",
            "=" * 60,
        ]
        lines.append(
            self.leaderboard.to_string(
                index=False,
                float_format=lambda x: f"{x:.4f}",
            )
        )
        lines.append("")
        lines.append(
            f"selected winner : {self._winner_method()} "
            f"(rule={self.selection_rule})"
        )
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe leaderboard and candidate-status payload."""
        return _jsonable_dict(
            {
                "kind": "auto_did_result",
                "selection_rule": self.selection_rule,
                "winner_method": self._winner_method(),
                "leaderboard": self.leaderboard.to_dict(orient="records"),
                "candidate_status": _candidate_status(self.candidates),
            }
        )

    def to_agent_summary(self, *, max_methods: int = 10) -> Dict[str, Any]:
        """Bounded agent-facing summary of the DiD estimator race."""
        limit = max(int(max_methods), 0)
        statuses = _candidate_status(self.candidates)
        successes = [row for row in statuses if row["ok"]]
        failures = [row for row in statuses if not row["ok"]]
        return _jsonable_dict(
            {
                "kind": "auto_did_agent_summary",
                "selection_rule": self.selection_rule,
                "winner_method": self._winner_method(),
                "leaderboard": self.leaderboard.head(limit).to_dict(orient="records"),
                "n_methods": int(len(self.leaderboard)),
                "truncated_methods": max(int(len(self.leaderboard)) - limit, 0),
                "n_successes": int(len(successes)),
                "failures": failures,
            }
        )

    def _winner_method(self) -> str:
        # Resolve the winner back to its method label by equality check
        # on the candidate dict — keeps the structure auditable.
        for k, v in self.candidates.items():
            if v is self.winner:
                return k
        return "<unresolved>"

    def __repr__(self) -> str:
        # Terse repr for list-of-results / Jupyter cell display.
        # Call `.summary()` explicitly for the full leaderboard.
        n_ok = sum(1 for v in self.candidates.values() if not isinstance(v, Exception))
        return (
            f"<AutoDIDResult: {n_ok}/{len(self.candidates)} ok, "
            f"winner={self._winner_method()} (rule={self.selection_rule})>"
        )


def auto_did(
    data: pd.DataFrame,
    y: str,
    g: str,
    t: str,
    i: str,
    *,
    x: Optional[List[str]] = None,
    methods: Optional[List[str]] = None,
    select_by: str = "median",
    alpha: float = 0.05,
) -> AutoDIDResult:
    """Run several staggered-DiD estimators side by side.

    Reports Callaway-Sant'Anna (2021), Sun-Abraham (2021), and Borusyak-
    Jaravel-Spiess (2024) by default — the three estimators whose joint
    reporting has become the 2026 top-journal standard for staggered
    adoption designs.

    Parameters
    ----------
    data : pd.DataFrame
        Panel data, long format.
    y : str
        Outcome column.
    g : str
        First-treatment-period column (0 or NaN for never-treated).
    t : str
        Calendar time column.
    i : str
        Unit identifier column.
    x : list of str, optional
        Covariates (used by CS / SA / BJS where supported).
    methods : list of str, optional
        Subset of ``{'cs', 'sa', 'bjs'}``.  Defaults to all three.
    select_by : {'median', 'first_success', 'cs', 'sa', 'bjs'}
        How to pick the winner.  ``'median'`` returns the candidate with
        the median point estimate across successes.  A method name
        returns that specific candidate (useful for forcing a baseline).
    alpha : float
        Significance level for reported CIs.

    Examples
    --------
    >>> import statspai as sp
    >>> df = sp.dgp_did(n_units=40, n_periods=6, staggered=True,
    ...                 n_groups=2, effect=0.5, seed=2026)
    >>> res = sp.auto_did(df, y="y", g="first_treat", t="time", i="unit")
    >>> sorted(res.leaderboard["method"])
    ['BJS', 'CS', 'SA']
    >>> bool(any(res.winner is v for v in res.candidates.values()))
    True

    References
    ----------
    callaway2021difference, sun2021estimating, borusyak2024revisiting
    """
    valid = {"cs", "sa", "bjs"}
    methods = _normalize_methods(
        methods, default=["cs", "sa", "bjs"], valid=valid, context="auto_did"
    )
    alpha = _validate_probability(alpha, name="alpha", context="auto_did")
    x_list = _as_string_list(x, name="x", context="auto_did", allow_none=True)
    _require_columns(data, [y, g, t, i] + x_list, context="auto_did")

    # BJS (did_imputation) interprets `g` as the first-treatment timing
    # (integer period, NaN for never-treated) — not a cohort label.
    # Fail fast if the column clearly doesn't look like a timing column
    # so users don't silently get wrong BJS answers on cohort-coded data.
    if "bjs" in methods and g in data.columns:
        g_vals = data[g].dropna().unique()
        if len(g_vals) > 0 and not pd.api.types.is_numeric_dtype(data[g]):
            raise MethodIncompatibility(
                f"auto_did: BJS branch requires column {g!r} to be a "
                "numeric first-treatment timing (not a cohort label). "
                "Either convert to the period of first treatment or drop "
                "'bjs' from `methods`."
            )

    candidates: Dict[str, Any] = {}
    rows: List[Tuple[str, float, float, float, float, int, str]] = []

    # Load submodules via importlib — NOT `from .did import <name>` which
    # would shadow the submodule with the re-exported function. Keeping a
    # live module handle lets tests monkeypatch the runner via
    # `statspai.did.callaway_santanna.callaway_santanna = broken`.
    import importlib

    _cs_mod = importlib.import_module("statspai.did.callaway_santanna")
    _sa_mod = importlib.import_module("statspai.did.sun_abraham")
    _bjs_mod = importlib.import_module("statspai.did.did_imputation")

    runners = {
        "cs": lambda: _cs_mod.callaway_santanna(
            data=data,
            y=y,
            g=g,
            t=t,
            i=i,
            x=x_list or None,
            alpha=alpha,
        ),
        "sa": lambda: _sa_mod.sun_abraham(
            data=data,
            y=y,
            g=g,
            t=t,
            i=i,
            covariates=x_list or None,
            alpha=alpha,
        ),
        # did_imputation is the BJS name in statspai; keep the mapping
        # explicit here so the leaderboard label stays 'bjs'.
        "bjs": lambda: _bjs_mod.did_imputation(
            data=data,
            y=y,
            group=i,
            time=t,
            first_treat=g,
            controls=x_list or None,
            alpha=alpha,
        ),
    }

    for m in methods:
        try:
            r = runners[m]()
            candidates[m] = r
            rows.append(
                (
                    m.upper(),
                    float(r.estimate),
                    float(getattr(r, "se", np.nan)),
                    float(r.ci[0]) if r.ci is not None else np.nan,
                    float(r.ci[1]) if r.ci is not None else np.nan,
                    int(getattr(r, "n_obs", 0) or 0),
                    "ok",
                )
            )
        except Exception as e:
            # Candidate failure is first-class; see
            # test_auto_did_degrades_when_one_candidate_fails.
            candidates[m] = e
            rows.append(
                (
                    m.upper(),
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    0,
                    f"FAILED: {type(e).__name__}: {e}",
                )
            )

    leaderboard = pd.DataFrame(
        rows,
        columns=[
            "method",
            "estimate",
            "std_error",
            "ci_lower",
            "ci_upper",
            "n_obs",
            "notes",
        ],
    )

    # --- Winner selection -------------------------------------------------
    successes = {k: v for k, v in candidates.items() if not isinstance(v, Exception)}
    if not successes:
        raise RuntimeError("auto_did: every candidate estimator failed.")

    rule = select_by.lower()
    if rule in valid and rule in successes:
        winner = successes[rule]
    elif rule in valid and rule not in successes:
        raise RuntimeError(
            f"auto_did: requested winner {rule!r} failed — "
            f"successful candidates were {list(successes)}. "
            f"Pick one of those, use select_by='first_success', or "
            f"inspect result.candidates[{rule!r}] for the exception."
        )
    elif rule == "first_success":
        winner = next(iter(successes.values()))
    elif rule == "median":
        sorted_pairs = sorted(
            successes.items(),
            key=lambda kv: float(kv[1].estimate),
        )
        mid = len(sorted_pairs) // 2
        winner = sorted_pairs[mid][1]
    else:
        raise MethodIncompatibility(
            f"auto_did: select_by={select_by!r} not recognised. "
            f"Use 'median', 'first_success', or a method name."
        )

    return AutoDIDResult(
        leaderboard=leaderboard,
        winner=winner,
        candidates=candidates,
        selection_rule=rule,
    )


# =====================================================================
# Auto-IV
# =====================================================================


@dataclass
class AutoIVResult:
    """Leaderboard + winner from :func:`auto_iv`.

    Mirrors :class:`AutoDIDResult` but for IV estimators.

    Examples
    --------
    >>> import statspai as sp
    >>> df = sp.dgp_iv(n=300, seed=7)
    >>> res = sp.auto_iv(df, y="y", endog="treatment",
    ...                  instruments="instrument", exog=["x1", "x2"])
    >>> isinstance(res, sp.AutoIVResult)
    True
    >>> list(res.leaderboard.columns)
    ['method', 'estimate', 'std_error', 'ci_lower', 'ci_upper', 'n_obs', 'notes']
    >>> print(res.summary())  # doctest: +SKIP
    """

    leaderboard: pd.DataFrame
    winner: Any
    candidates: Dict[str, Any]
    selection_rule: str = "median"

    def summary(self) -> str:
        lines = [
            "auto_iv: 2SLS / LIML / JIVE race",
            "=" * 60,
        ]
        lines.append(
            self.leaderboard.to_string(
                index=False,
                float_format=lambda x: f"{x:.4f}",
            )
        )
        lines.append("")
        lines.append(
            f"selected winner : {self._winner_method()} "
            f"(rule={self.selection_rule})"
        )
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe leaderboard and candidate-status payload."""
        return _jsonable_dict(
            {
                "kind": "auto_iv_result",
                "selection_rule": self.selection_rule,
                "winner_method": self._winner_method(),
                "leaderboard": self.leaderboard.to_dict(orient="records"),
                "candidate_status": _candidate_status(self.candidates),
            }
        )

    def to_agent_summary(self, *, max_methods: int = 10) -> Dict[str, Any]:
        """Bounded agent-facing summary of the IV estimator race."""
        limit = max(int(max_methods), 0)
        statuses = _candidate_status(self.candidates)
        successes = [row for row in statuses if row["ok"]]
        failures = [row for row in statuses if not row["ok"]]
        return _jsonable_dict(
            {
                "kind": "auto_iv_agent_summary",
                "selection_rule": self.selection_rule,
                "winner_method": self._winner_method(),
                "leaderboard": self.leaderboard.head(limit).to_dict(orient="records"),
                "n_methods": int(len(self.leaderboard)),
                "truncated_methods": max(int(len(self.leaderboard)) - limit, 0),
                "n_successes": int(len(successes)),
                "failures": failures,
            }
        )

    def _winner_method(self) -> str:
        for k, v in self.candidates.items():
            if v is self.winner:
                return k
        return "<unresolved>"

    def __repr__(self) -> str:
        n_ok = sum(1 for v in self.candidates.values() if not isinstance(v, Exception))
        return (
            f"<AutoIVResult: {n_ok}/{len(self.candidates)} ok, "
            f"winner={self._winner_method()} (rule={self.selection_rule})>"
        )


def _coef(result: Any, name: str) -> Optional[float]:
    """Extract a specific coefficient from an EconometricResults object."""
    params = getattr(result, "params", None)
    if params is None:
        return None
    if name in params.index:
        return float(params[name])
    return None


def _coef_or_nan(result: Any, name: str) -> float:
    """``_coef`` as a plain float, mapping a missing coefficient to NaN."""
    c = _coef(result, name)
    return float("nan") if c is None else c


def _se(result: Any, name: str) -> Optional[float]:
    se = getattr(result, "std_errors", None)
    if se is None:
        return None
    if name in se.index:
        return float(se[name])
    return None


def auto_iv(
    data: pd.DataFrame,
    y: str,
    endog: str,
    instruments: Any,
    *,
    exog: Optional[List[str]] = None,
    methods: Optional[List[str]] = None,
    select_by: str = "median",
    alpha: float = 0.05,
    robust: str = "nonrobust",
    cluster: Optional[str] = None,
) -> AutoIVResult:
    """Race 2SLS, LIML, and JIVE on a single-endogenous IV spec.

    Parameters
    ----------
    data : pd.DataFrame
    y : str
        Outcome column.
    endog : str
        Single endogenous regressor column.
    instruments : str or list of str
        Instrument(s).  A scalar is promoted to a one-element list.
    exog : list of str, optional
        Exogenous controls (included in all requested IV estimators).
    methods : list of str, optional
        Subset of ``{'2sls', 'liml', 'jive'}``.
    select_by : {'median', 'first_success', '2sls', 'liml', 'jive'}
    alpha : float
    robust, cluster
        Forwarded to each estimator where supported.

    Examples
    --------
    >>> import statspai as sp
    >>> df = sp.dgp_iv(n=300, seed=7)
    >>> res = sp.auto_iv(df, y="y", endog="treatment",
    ...                  instruments="instrument", exog=["x1", "x2"])
    >>> sorted(res.leaderboard["method"])
    ['2SLS', 'JIVE', 'LIML']
    >>> bool(any(res.winner is v for v in res.candidates.values()))
    True

    References
    ----------
    angrist2009mostly
    """
    instruments_list = _as_string_list(
        instruments,
        name="instruments",
        context="auto_iv",
    )
    exog_list = _as_string_list(
        exog,
        name="exog",
        context="auto_iv",
        allow_none=True,
    )
    valid = {"2sls", "liml", "jive"}
    methods = _normalize_methods(
        methods,
        default=["2sls", "liml", "jive"],
        valid=valid,
        context="auto_iv",
    )
    alpha = _validate_probability(alpha, name="alpha", context="auto_iv")
    required = [y, endog] + instruments_list + exog_list
    if cluster is not None:
        required.append(cluster)
    _require_columns(data, required, context="auto_iv")

    from .regression.iv import iv as iv_regress
    from .regression.advanced_iv import liml, jive

    # 2SLS uses formula-style; LIML / JIVE use explicit column lists.
    # statspai's IV formula convention: "y ~ (endog ~ z1 + z2) + exog1 + ..."
    instr_part = " + ".join(instruments_list)
    endog_block = f"({endog} ~ {instr_part})"
    if exog_list:
        rhs = endog_block + " + " + " + ".join(exog_list)
    else:
        rhs = endog_block
    formula = f"{y} ~ {rhs}"

    runners = {
        "2sls": lambda: iv_regress(
            formula=formula,
            data=data,
            method="2sls",
            robust=robust,
            cluster=cluster,
        ),
        "liml": lambda: liml(
            data=data,
            y=y,
            x_endog=[endog],
            x_exog=exog_list or None,
            z=instruments_list,
            robust=robust,
            cluster=cluster,
            alpha=alpha,
        ),
        "jive": lambda: jive(
            data=data,
            y=y,
            x_endog=[endog],
            x_exog=exog_list or None,
            z=instruments_list,
            robust=robust,
            cluster=cluster,
            variant="jive1",
            alpha=alpha,
        ),
    }

    candidates: Dict[str, Any] = {}
    rows = []
    for m in methods:
        try:
            r = runners[m]()
            candidates[m] = r
            coef = _coef(r, endog)
            se = _se(r, endog)
            if coef is not None and se is not None and np.isfinite(se):
                from scipy import stats as _stats

                crit = _stats.norm.ppf(1 - alpha / 2)
                ci_lo = coef - crit * se
                ci_hi = coef + crit * se
            else:
                ci_lo = ci_hi = np.nan
            # n_obs: EconometricResults sometimes stores it under different
            # keys ('n_obs' / 'nobs' / 'n'); fall back to len(data).
            # data_info can be None on some estimators — guard with `or {}`.
            n_obs = 0
            info = getattr(r, "data_info", None) or {}
            for key in ("n_obs", "nobs", "n"):
                if key in info:
                    n_obs = int(info[key])
                    break
            if n_obs == 0:
                n_obs = int(len(data))
            rows.append(
                (
                    m.upper(),
                    np.nan if coef is None else coef,
                    np.nan if se is None else se,
                    ci_lo,
                    ci_hi,
                    n_obs,
                    "ok",
                )
            )
        except Exception as e:
            # Candidate failure is first-class for the IV race as well.
            candidates[m] = e
            rows.append(
                (
                    m.upper(),
                    np.nan,
                    np.nan,
                    np.nan,
                    np.nan,
                    0,
                    f"FAILED: {type(e).__name__}: {e}",
                )
            )

    leaderboard = pd.DataFrame(
        rows,
        columns=[
            "method",
            "estimate",
            "std_error",
            "ci_lower",
            "ci_upper",
            "n_obs",
            "notes",
        ],
    )

    successes = {k: v for k, v in candidates.items() if not isinstance(v, Exception)}
    if not successes:
        raise RuntimeError("auto_iv: every candidate estimator failed.")

    rule = select_by.lower()
    if rule in valid and rule in successes:
        winner = successes[rule]
    elif rule in valid and rule not in successes:
        raise RuntimeError(
            f"auto_iv: requested winner {rule!r} failed — "
            f"successful candidates were {list(successes)}. "
            f"Pick one of those, use select_by='first_success', or "
            f"inspect result.candidates[{rule!r}] for the exception."
        )
    elif rule == "first_success":
        winner = next(iter(successes.values()))
    elif rule == "median":
        # Exclude NaN-coef successes from the median pool so sorted() is
        # deterministic (NaN compares False both ways under Python's sort).
        sortable = {
            k: v for k, v in successes.items() if np.isfinite(_coef_or_nan(v, endog))
        }
        if not sortable:
            # All successes have NaN coefs; fall back to first_success.
            winner = next(iter(successes.values()))
        else:
            sorted_pairs = sorted(
                sortable.items(),
                key=lambda kv: _coef_or_nan(kv[1], endog),
            )
            mid = len(sorted_pairs) // 2
            winner = sorted_pairs[mid][1]
    else:
        raise MethodIncompatibility(f"auto_iv: select_by={select_by!r} not recognised.")

    return AutoIVResult(
        leaderboard=leaderboard,
        winner=winner,
        candidates=candidates,
        selection_rule=rule,
    )
