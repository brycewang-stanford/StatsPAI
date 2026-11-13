"""Remember which robustness checks a session has run on a cached fit.

``sp.audit`` reads a fitted result and nothing else, so over MCP a check
the agent ran in a later ``tools/call`` (``honest_did_from_result``,
``bacon_decomposition``, ``rdplacebo``, ...) never reached the audit:
``audit_result`` listed it as missing before and after, and an agent told
to "run what is missing" had no way to finish.

A successful follow-up is recorded on the cache entry of the fit it
belongs to, under the evidence key the audit checks. Which fit that is:

* the ``result_id`` the follow-up was given, or
* for a tool that takes the data rather than the result, every cached
  fit on the same dataset (same handle, or same file hash) with the same
  outcome column (and, for RD, the same running variable; for IV, the
  same endogenous regressor).

Nothing is recorded for a call that returned an error, and nothing marks a
check as *passed on its merits*: the entry says the check has been run and
by which call.
"""

from __future__ import annotations

from typing import Any, Dict, Optional

#: tool -> audit evidence key (``smart/audit.py`` ``evidence_paths``).
_CHECK_BY_TOOL: Dict[str, str] = {
    "honest_did_from_result": "honest_did",
    "honest_did": "honest_did",
    "breakdown_m": "honest_did",
    "bacon_decomposition": "bacon_decomposition",
    "rdbwsensitivity": "bandwidth_sensitivity",
    "rdplacebo": "placebo_cutoff",
    "anderson_rubin_test": "anderson_rubin",
    "anderson_rubin_ci": "anderson_rubin",
    "oster_delta": "oster",
    "oster_bounds": "oster",
    "sensemakr": "cinelli_hazlett",
    "evalue_from_result": "evalue",
}

#: ``sensitivity_from_result(method=...)`` -> evidence key.
_SENSITIVITY_METHOD_KEY = {
    "oster": "oster",
    "evalue": "evalue",
    "sensemakr": "cinelli_hazlett",
}

#: Entry-argument key the evidence is stored under (private: it is not an
#: argument of the fit and is dropped from replay / metadata views).
STORE_KEY = "_mcp_checks_run"

_HEADLINE_FIELDS = (
    "max_rejecting_M",
    "restriction",
    "negative_weight_share",
    "already_treated_control_weight_share",
    "ar_ci",
    "ar_pvalue",
    "lower",
    "upper",
    "evalue_estimate",
    "evalue_ci",
    "delta",
    "rv_q",
)


def evidence_key(tool: str, arguments: Dict[str, Any]) -> Optional[str]:
    """The audit evidence key a successful call to ``tool`` provides."""
    if tool == "sensitivity_from_result":
        method = str(arguments.get("method") or "evalue").lower()
        return _SENSITIVITY_METHOD_KEY.get(method)
    return _CHECK_BY_TOOL.get(tool)


def _dataset_key(prov: Any) -> Optional[str]:
    """One comparable name for a dataset: its handle, else its content hash."""
    if not isinstance(prov, dict):
        return None
    if prov.get("data_id"):
        return f"id:{prov['data_id']}"
    for node in (prov, prov.get("root") or {}):
        if isinstance(node, dict) and node.get("sha256"):
            return f"sha:{node['sha256']}"
    return None


def _roles(arguments: Dict[str, Any]) -> Dict[str, Any]:
    from ._enrichment import _followup_base_args

    return _followup_base_args(
        {k: v for k, v in arguments.items() if not str(k).startswith("_")}
    )


def _same_design(fit_args: Dict[str, Any], call_args: Dict[str, Any]) -> bool:
    """Same outcome, and same running variable / endogenous regressor."""
    fit, call = _roles(fit_args), _roles(call_args)
    if call.get("y") is None or fit.get("y") != call.get("y"):
        return False
    for role in ("running_var", "endog"):
        if role in call and role in fit and fit[role] != call[role]:
            return False
    return True


def record_followup(
    tool: str,
    arguments: Dict[str, Any],
    payload: Dict[str, Any],
    data_provenance: Optional[Dict[str, Any]] = None,
) -> int:
    """Record a successful follow-up on the fits it belongs to.

    Returns the number of cache entries annotated.
    """
    key = evidence_key(tool, arguments)
    if key is None or not isinstance(payload, dict) or payload.get("error"):
        return 0
    from ._result_cache import RESULT_CACHE

    evidence: Dict[str, Any] = {"has_run": True, "tool": tool}
    for name in _HEADLINE_FIELDS:
        value = payload.get(name)
        if isinstance(value, (int, float, str, bool)) or (
            isinstance(value, (list, tuple)) and len(value) <= 4
        ):
            evidence[name] = value

    targets = []
    rid = arguments.get("result_id")
    if isinstance(rid, str):
        targets.append(rid)
    else:
        wanted = _dataset_key(data_provenance)
        if wanted is not None:
            for other_rid, entry in RESULT_CACHE.snapshot().items():
                held = _dataset_key(entry.arguments.get("_mcp_data_provenance"))
                if held == wanted and _same_design(entry.arguments, arguments):
                    targets.append(other_rid)

    done = 0
    for target in targets:
        held_entry = RESULT_CACHE.get_entry(target)
        if held_entry is None:
            continue
        runs = dict(held_entry.arguments.get(STORE_KEY) or {})
        runs[key] = evidence
        if RESULT_CACHE.annotate(target, {STORE_KEY: runs}):
            done += 1
    return done


def checks_run(rid: Optional[str]) -> Dict[str, Any]:
    """The follow-ups recorded for ``rid`` (empty when none)."""
    if not isinstance(rid, str):
        return {}
    from ._result_cache import RESULT_CACHE

    entry = RESULT_CACHE.get_entry(rid)
    if entry is None:
        return {}
    return dict(entry.arguments.get(STORE_KEY) or {})


__all__ = ["STORE_KEY", "checks_run", "evidence_key", "record_followup"]
