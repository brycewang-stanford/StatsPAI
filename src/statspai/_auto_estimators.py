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

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

__all__ = [
    "auto_did",
    "auto_iv",
    "AutoDIDResult",
    "AutoIVResult",
]


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
                index=False, float_format=lambda x: f"{x:.4f}",
            )
        )
        lines.append("")
        lines.append(
            f"selected winner : {self._winner_method()} "
            f"(rule={self.selection_rule})"
        )
        return "\n".join(lines)

    def _winner_method(self) -> str:
        # Resolve the winner back to its method label by equality check
        # on the candidate dict — keeps the structure auditable.
        for k, v in self.candidates.items():
            if v is self.winner:
                return k
        return "<unresolved>"

    def __repr__(self) -> str:
        return self.summary()


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
    """
    methods = [m.lower() for m in (methods or ["cs", "sa", "bjs"])]
    valid = {"cs", "sa", "bjs"}
    bad = [m for m in methods if m not in valid]
    if bad:
        raise ValueError(
            f"auto_did: unknown methods {bad}; valid = {sorted(valid)}"
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
            data=data, y=y, g=g, t=t, i=i, x=x, alpha=alpha,
        ),
        "sa": lambda: _sa_mod.sun_abraham(
            data=data, y=y, g=g, t=t, i=i, covariates=x, alpha=alpha,
        ),
        # did_imputation is the BJS name in statspai; keep the mapping
        # explicit here so the leaderboard label stays 'bjs'.
        "bjs": lambda: _bjs_mod.did_imputation(
            data=data, y=y, group=g, time=t, first_treat=g,
            controls=x, alpha=alpha,
        ),
    }

    for m in methods:
        try:
            r = runners[m]()
            candidates[m] = r
            rows.append((
                m.upper(),
                float(r.estimate),
                float(getattr(r, "se", np.nan)),
                float(r.ci[0]) if r.ci is not None else np.nan,
                float(r.ci[1]) if r.ci is not None else np.nan,
                int(getattr(r, "n_obs", 0) or 0),
                "ok",
            ))
        except Exception as e:  # pragma: no cover — exercised by test_auto_did_degrades
            candidates[m] = e
            rows.append((
                m.upper(), np.nan, np.nan, np.nan, np.nan, 0,
                f"FAILED: {type(e).__name__}: {e}",
            ))

    leaderboard = pd.DataFrame(
        rows,
        columns=["method", "estimate", "std_error",
                 "ci_lower", "ci_upper", "n_obs", "notes"],
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
            successes.items(), key=lambda kv: float(kv[1].estimate),
        )
        mid = len(sorted_pairs) // 2
        winner = sorted_pairs[mid][1]
    else:
        raise ValueError(
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
                index=False, float_format=lambda x: f"{x:.4f}",
            )
        )
        lines.append("")
        lines.append(f"selected winner : {self._winner_method()} "
                     f"(rule={self.selection_rule})")
        return "\n".join(lines)

    def _winner_method(self) -> str:
        for k, v in self.candidates.items():
            if v is self.winner:
                return k
        return "<unresolved>"

    def __repr__(self) -> str:
        return self.summary()


def _coef(result, name: str) -> Optional[float]:
    """Extract a specific coefficient from an EconometricResults object."""
    params = getattr(result, "params", None)
    if params is None:
        return None
    if name in params.index:
        return float(params[name])
    return None


def _se(result, name: str) -> Optional[float]:
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
    instruments,
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
        Exogenous controls (included in all estimators).
    methods : list of str, optional
        Subset of ``{'2sls', 'liml', 'jive'}``.
    select_by : {'median', 'first_success', '2sls', 'liml', 'jive'}
    alpha : float
    robust, cluster
        Forwarded to each estimator where supported.
    """
    if isinstance(instruments, str):
        instruments_list = [instruments]
    else:
        instruments_list = list(instruments)
    exog_list = list(exog) if exog else []

    methods = [m.lower() for m in (methods or ["2sls", "liml", "jive"])]
    valid = {"2sls", "liml", "jive"}
    bad = [m for m in methods if m not in valid]
    if bad:
        raise ValueError(
            f"auto_iv: unknown methods {bad}; valid = {sorted(valid)}"
        )

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
            formula=formula, data=data, method="2sls",
            robust=robust, cluster=cluster,
        ),
        "liml": lambda: liml(
            data=data, y=y, x_endog=[endog], x_exog=exog_list or None,
            z=instruments_list, robust=robust, cluster=cluster, alpha=alpha,
        ),
        "jive": lambda: jive(
            data=data, y=y, x_endog=[endog], x_exog=exog_list or None,
            z=instruments_list, robust=robust, cluster=cluster,
            variant="jive1", alpha=alpha,
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
            rows.append((
                m.upper(),
                np.nan if coef is None else coef,
                np.nan if se is None else se,
                ci_lo, ci_hi,
                n_obs,
                "ok",
            ))
        except Exception as e:  # pragma: no cover — exercised via degradation test
            candidates[m] = e
            rows.append((
                m.upper(), np.nan, np.nan, np.nan, np.nan, 0,
                f"FAILED: {type(e).__name__}: {e}",
            ))

    leaderboard = pd.DataFrame(
        rows,
        columns=["method", "estimate", "std_error",
                 "ci_lower", "ci_upper", "n_obs", "notes"],
    )

    successes = {k: v for k, v in candidates.items()
                 if not isinstance(v, Exception)}
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
            k: v for k, v in successes.items()
            if _coef(v, endog) is not None and np.isfinite(_coef(v, endog))
        }
        if not sortable:
            # All successes have NaN coefs; fall back to first_success.
            winner = next(iter(successes.values()))
        else:
            sorted_pairs = sorted(
                sortable.items(), key=lambda kv: float(_coef(kv[1], endog)),
            )
            mid = len(sorted_pairs) // 2
            winner = sorted_pairs[mid][1]
    else:
        raise ValueError(
            f"auto_iv: select_by={select_by!r} not recognised."
        )

    return AutoIVResult(
        leaderboard=leaderboard,
        winner=winner,
        candidates=candidates,
        selection_rule=rule,
    )
