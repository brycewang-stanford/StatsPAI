"""Hand-curated workflow / handle-based / citation tools.

These are the "Tier-0" tools that close the agent feedback loop:

* ``audit_result`` / ``brief_result`` / ``sensitivity_from_result`` /
  ``honest_did_from_result`` — operate on a cached result handle
  produced by an earlier tool call (``as_handle=True``). They eliminate
  the LLM having to ferry back arrays and CSV paths between turns.
* ``bibtex`` — return verified BibTeX entries from the project's
  ``paper.bib`` (the single source of truth per CLAUDE.md §10). Closes
  the citation-hallucination loophole.
* ``audit`` / ``preflight`` / ``detect_design`` / ``brief`` — explicit
  hand-curated wrappers for the smart-workflow primitives that the
  prompt templates reference. Auto-tools used to surface these with
  one-line descriptions; the bespoke schemas below give agents proper
  signposting.

Every workflow tool returns a dict shaped like the standard estimator
serializer output (``estimate`` / ``method`` / ``next_calls`` / …) so
the MCP layer doesn't need to special-case their content blocks.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Optional, Tuple

import pandas as pd

from ..exceptions import MethodIncompatibility as _MethodIncompatibility
from ._result_cache import RESULT_CACHE

# ----------------------------------------------------------------------
# Schema definitions surfaced via tool_manifest()
# ----------------------------------------------------------------------


def _result_id_schema(description: str) -> Dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "result_id": {
                "type": "string",
                "description": description,
            },
        },
        "required": ["result_id"],
    }


WORKFLOW_TOOL_SPECS: List[Dict[str, Any]] = [
    # ------------------------------------------------------------------
    # Handle-based extensions to the curated tools — break the
    # "LLM ferries arrays" anti-pattern.
    # ------------------------------------------------------------------
    {
        "name": "audit_result",
        "description": (
            "Reviewer-grade audit on a previously-fitted result. Pass "
            "the result_id returned by an earlier tool call (with "
            "as_handle=true). Returns the same checklist sp.audit() "
            "produces — every robustness check the literature expects "
            "for the design, each with status "
            "'passed|failed|missing|not_applicable' and a concrete "
            "suggest_function for the missing ones. A not_applicable "
            "check (e.g. an over-identification test on a just-identified "
            "IV fit) carries a 'reason' and is excluded from the "
            "summary's n_total."
        ),
        "input_schema": _result_id_schema(
            "Handle returned by an earlier estimator call. Must be in "
            "the server result cache (LRU-evicted; refit if missing)."
        ),
    },
    {
        "name": "brief_result",
        "description": (
            "Return the one-line agent-friendly brief for a fitted "
            "result. Uses sp.brief(). Useful when an agent wants to "
            "summarise a chained workflow without paying for the full "
            "JSON payload again."
        ),
        "input_schema": _result_id_schema("Handle to a previously-fitted result."),
    },
    {
        "name": "interpret_result",
        "description": (
            "Natural-language interpretation of a fitted result. When the "
            "connected MCP client advertised sampling, this REUSES the "
            "agent's own model (no API key) to explain the estimate, its "
            "uncertainty, and what the design does / does not identify — "
            "optionally focused by a `question` and tuned for an "
            "`audience`. With no sampling available it falls back to a "
            "deterministic structured brief: it NEVER fabricates a "
            "narrative. Every claim is grounded in the result's own "
            "numbers — the model is told not to invent estimates. Pass "
            "the result_id from an earlier as_handle=true call."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "result_id": {
                    "type": "string",
                    "description": "Handle to a previously-fitted result.",
                },
                "question": {
                    "type": "string",
                    "description": (
                        "Optional specific question to focus the "
                        "interpretation (e.g. 'is the effect "
                        "economically meaningful?')."
                    ),
                },
                "audience": {
                    "type": "string",
                    "enum": ["researcher", "policymaker", "general"],
                    "default": "researcher",
                    "description": (
                        "Tone / depth: 'researcher' (precise, names "
                        "identification assumptions), 'policymaker' "
                        "(plain, decision-focused), 'general' (no jargon)."
                    ),
                },
            },
            "required": ["result_id"],
        },
    },
    {
        "name": "sensitivity_from_result",
        "description": (
            "Run sp.sensitivity / sp.evalue / sp.oster_bounds / "
            "sp.sensemakr on a cached result. Pass method='evalue' "
            "(default) for the omitted-confounder-strength bound, "
            "'oster' for delta/R-max, 'cinelli_hazlett' for OVB bounds."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "result_id": {
                    "type": "string",
                    "description": "Handle to a fitted causal result.",
                },
                "method": {
                    "type": "string",
                    "enum": ["evalue", "oster", "cinelli_hazlett", "auto"],
                    "default": "evalue",
                },
                "benchmark_covariate": {
                    "type": "string",
                    "description": "Cinelli-Hazlett benchmark column (optional).",
                },
            },
            "required": ["result_id"],
        },
    },
    {
        "name": "honest_did_from_result",
        "description": (
            "Rambachan-Roth (2023) honest CIs on a fitted DID / "
            "event-study result. Auto-extracts betas + sigma + "
            "pre/post-period counts from the result; the LLM never "
            "ferries arrays."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "result_id": {
                    "type": "string",
                    "description": "Handle to a DID / event-study result.",
                },
                "method": {
                    "type": "string",
                    "enum": ["SD", "RM"],
                    "default": "SD",
                    "description": (
                        "SD = smoothness deviation "
                        "(Rambachan-Roth default); "
                        "RM = relative magnitude."
                    ),
                },
                "e": {
                    "type": "integer",
                    "default": 0,
                    "description": "Relative event time to audit.",
                },
                "m_bar": {
                    "type": "number",
                    "description": "Bound on deviation magnitude (optional).",
                },
            },
            "required": ["result_id"],
        },
    },
    # ------------------------------------------------------------------
    # Workflow primitives — explicit registrations so prompt templates
    # have first-class entries instead of auto-generated stubs.
    # ------------------------------------------------------------------
    {
        "name": "audit",
        "description": (
            "Reviewer-grade audit on a result. Returns the literature "
            "checklist (parallel-trends test, honest-DID, Bacon "
            "decomposition, placebo, balance, …) with status per item "
            "and the concrete suggest_function to call to fill any "
            "missing high-importance check."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "result_id": {
                    "type": "string",
                    "description": (
                        "Result handle. Required unless "
                        "you also pass a fitted result via "
                        "the result kwarg (programmatic "
                        "use)."
                    ),
                },
            },
            "required": [],
        },
    },
    {
        "name": "preflight",
        "description": (
            "Run pre-fit identification checks for a chosen method on a "
            "DataFrame. Verdict in {PASS, WARN, FAIL}. ALWAYS call this "
            "before fitting on an unfamiliar dataset to surface design "
            "problems (overlap, cohort sizes, IV first-stage F, "
            "running-variable density at the cutoff)."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "method": {
                    "type": "string",
                    "description": (
                        "Estimator name: 'did', 'rd', 'iv', "
                        "'synth', 'matching', 'dml', …"
                    ),
                },
                "y": {"type": "string", "description": "Outcome column."},
                "treatment": {"type": "string"},
                "time": {"type": "string"},
                "id": {"type": "string", "description": "Unit id column."},
                "cohort": {"type": "string"},
                "running_var": {"type": "string"},
                "instrument": {"type": "string"},
                "covariates": {
                    "type": "array",
                    "items": {"type": "string"},
                },
            },
            "required": ["method"],
        },
    },
    {
        "name": "detect_design",
        "description": (
            "Auto-detect the study design (panel / cross-section / RD "
            "/ IV-style) from column shapes and types. Returns the "
            "guessed design plus the columns that drove the inference. "
            "Call this BEFORE recommend() when the user pastes a CSV "
            "with no context."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "time_col_hint": {"type": "string"},
                "id_col_hint": {"type": "string"},
            },
            "required": [],
        },
    },
    {
        "name": "brief",
        "description": (
            "One-line agent-friendly brief for a fitted result. "
            "Cheaper than calling brief_result if you already have the "
            "result object in scope."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "result_id": {"type": "string"},
            },
            "required": [],
        },
    },
    # ------------------------------------------------------------------
    # Citation tool — the kill-switch for citation hallucination.
    # ------------------------------------------------------------------
    {
        "name": "from_stata",
        "description": (
            "Translate a single Stata command to a verified StatsPAI "
            "tool-call payload. Returns ``python_code`` (string for "
            "chat replies) AND ``arguments`` (ready-to-dispatch JSON-RPC "
            "for tools/call). Tier-1 commands: regress / xtreg / "
            "reghdfe / ivreg2 / csdid / did_imputation / synth / "
            "rdrobust; count-panel commands include nbreg / xtnbreg / "
            "ppmlhdfe. Unrecognised commands return close-match "
            "suggestions instead of guessing."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "command": {
                    "type": "string",
                    "description": (
                        "One Stata command, e.g. "
                        "'reghdfe y x, absorb(id year) "
                        "cluster(id)'. Multi-command lines "
                        "must be split by the caller."
                    ),
                },
            },
            "required": ["command"],
        },
    },
    {
        "name": "from_r",
        "description": (
            "Translate a single R / fixest / felm / did expression to a "
            "verified StatsPAI tool-call payload. Returns the same shape "
            "as from_stata. Supported callables: feols / felm / lm / "
            "att_gt / did. Pass ONE expression — no assignment, no piping."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "expression": {
                    "type": "string",
                    "description": (
                        "One R expression, e.g. "
                        "'feols(y ~ x | id^year, "
                        'data=df, cluster="id")\'.'
                    ),
                },
            },
            "required": ["expression"],
        },
    },
    {
        "name": "plot_from_result",
        "description": (
            "Render the canonical diagnostic plot for a fitted result "
            "and return it as an inline PNG image content block. "
            "MCP clients with vision (Claude Desktop, vision-capable "
            "agents) get the plot for free; clients that don't support "
            "image content see only the JSON metadata. Plot kind is "
            "auto-selected from the result type: event-study for DID, "
            "rdplot for RD, gap plot for synth, balance plot for "
            "matching, ROC for classification, etc."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "result_id": {
                    "type": "string",
                    "description": "Handle to a fitted result.",
                },
                "kind": {
                    "type": "string",
                    "description": (
                        "Override the auto-detected plot kind. "
                        "Common values: 'event_study', 'rdplot', "
                        "'synth_gap', 'love_plot', 'coef_plot'."
                    ),
                },
                "figsize": {
                    "type": "array",
                    "items": {"type": "number"},
                    "description": "Width, height in inches (default [8,5]).",
                },
            },
            "required": ["result_id"],
        },
    },
    {
        "name": "bibtex",
        "description": (
            "Return verified BibTeX entries from paper.bib (StatsPAI's "
            "single source of truth for citations). Pass one or more "
            "bib keys (e.g. 'callaway2021difference'). NEVER invent "
            "citations — call this tool instead. Unknown keys return "
            "an empty entry plus a list of close matches."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "keys": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Bib keys to look up. Most "
                        "estimators advertise their key "
                        "in agent_card.reference."
                    ),
                },
            },
            "required": ["keys"],
        },
    },
    {
        "name": "cross_validate",
        "statspai_fn": "cross_validate",
        "description": (
            "Cross-validate ONE estimand across INDEPENDENT engines "
            "(StatsPAI, pyfixest, linearmodels, DoubleML, R's fixest, Stata) "
            "and report whether they agree (AGREE / PARTIAL / DISAGREE / "
            "INSUFFICIENT). Use this to honour the cross-package "
            "reproducibility rule: trust a number only when >=2 independent "
            "implementations reproduce it. Needs a data_path."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "estimand": {
                    "type": "string",
                    "enum": ["ols", "feols", "iv", "poisson", "dml", "did"],
                    "description": "Model family to fit in every engine.",
                },
                "formula": {
                    "type": "string",
                    "description": "fixest-style 'y ~ x | fe | endog ~ z'.",
                },
                "y": {"type": "string", "description": "Outcome column."},
                "g": {
                    "type": "string",
                    "description": (
                        "DiD only: cohort / first-treatment period "
                        "(0 = never treated)."
                    ),
                },
                "t": {"type": "string", "description": "DiD only: time column."},
                "i": {
                    "type": "string",
                    "description": "DiD only: unit-id column.",
                },
                "treatment": {
                    "type": "string",
                    "description": "Focal regressor (reconciled coefficient).",
                },
                "covariates": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "fixed_effects": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "endog": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Endogenous regressors (IV).",
                },
                "instruments": {
                    "type": "array",
                    "items": {"type": "string"},
                },
                "vcov": {"type": "string"},
                "engines": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": (
                        "Engines to run, e.g. "
                        "['statspai','R::fixest','pyfixest','Stata']. "
                        "Omit for 'auto' (all installed + applicable)."
                    ),
                },
            },
            "required": ["estimand"],
        },
    },
    # ------------------------------------------------------------------
    # Discovery meta-tools. These are what make the curated MCP profile
    # usable: a client that sees only ~40 tools can still reach every
    # registered function by searching, reading the schema, and calling
    # it by name — the same discover -> describe -> call loop that
    # ``sp.search_functions`` / ``sp.describe_function`` give in Python.
    # ------------------------------------------------------------------
    {
        "name": "search_functions",
        "description": (
            "Search the StatsPAI function registry by task keywords or a "
            "short natural-language phrase (e.g. 'staggered adoption "
            "event study', 'weak instrument robust CI', 'regression "
            "discontinuity density test'). Returns ranked matches with "
            "one-line descriptions, category and validation tier. Use "
            "this first when no listed tool fits; then call "
            "describe_function for the full parameter schema and "
            "call_function to run it."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Keywords or a short task description.",
                },
                "limit": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 50,
                    "default": 10,
                    "description": "Maximum number of matches to return.",
                },
                "category": {
                    "type": "string",
                    "description": (
                        "Optional registry category filter (e.g. 'causal', "
                        "'panel', 'regression', 'diagnostics')."
                    ),
                },
            },
            "required": ["query"],
        },
    },
    {
        "name": "describe_function",
        "description": (
            "Full machine-readable metadata for one registered StatsPAI "
            "function: JSON-Schema parameters (types, defaults, enums), "
            "assumptions, pre-conditions, failure modes, alternatives, "
            "validation tier and evidence notes, plus the accepted "
            "keyword aliases. Read this before calling a function that "
            "is not in the listed tools."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Registered function name (sp.<name>).",
                },
            },
            "required": ["name"],
        },
    },
    {
        "name": "call_function",
        "description": (
            "Run ANY registered StatsPAI function by name with keyword "
            "arguments — the escape hatch for the ~1,200 functions that "
            "are not individually listed as tools. `arguments` follows "
            "the parameter schema from describe_function; column names "
            "refer to the file loaded via data_path. Unknown arguments "
            "are reported under `_unsupported_args`, never silently "
            "dropped. Supports as_handle / result_id chaining like every "
            "other tool."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "function": {
                    "type": "string",
                    "description": "Registered function name (sp.<name>).",
                },
                "arguments": {
                    "type": "object",
                    "description": (
                        "Keyword arguments for the function (JSON object)."
                    ),
                    "additionalProperties": True,
                },
            },
            "required": ["function"],
        },
    },
    # ------------------------------------------------------------------
    # Data handles. ``load_data`` turns a file / inline table into a
    # ``data_id``; ``transform_data`` derives new handles with a recorded
    # lineage; ``describe_data`` profiles any handle. Every tool accepts
    # ``data_id`` wherever it accepts ``data_path``.
    # ------------------------------------------------------------------
    {
        "name": "load_data",
        "description": (
            "Load a dataset into the server and get a data_id handle. "
            "Source: data_path (file or URL), data_records (JSON rows) or "
            "data_csv (CSV text). Returns the handle plus shape, dtypes, "
            "missing counts, the first rows and a numeric summary. Pass "
            "data_id to any later tool instead of re-sending the file; "
            "chain transform_data to filter / reshape / winsorise / impute "
            "and get a new handle whose lineage is recorded in every "
            "result's data_provenance."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "name": {
                    "type": "string",
                    "description": "Optional label recorded with the handle.",
                },
            },
            "required": [],
        },
    },
    {
        "name": "describe_data",
        "description": (
            "Profile a dataset (data_id, data_path or inline data): shape, "
            "dtypes, missing counts, head, numeric summary and — for a "
            "handle — the lineage of transforms that produced it."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "head": {
                    "type": "integer",
                    "minimum": 0,
                    "maximum": 50,
                    "default": 5,
                    "description": "Rows to echo.",
                },
            },
            "required": [],
        },
    },
    {
        "name": "transform_data",
        "description": (
            "Derive a new dataset from a data_id (or data_path / inline "
            "data) by applying `operations` in order, and return a new "
            "data_id with the lineage recorded. Operations (each an object "
            "with `op`): query {expr} (pandas DataFrame.query), select "
            "{columns}, drop {columns}, rename {mapping}, dropna "
            "{columns?}, fillna {value | mapping}, assign {column, expr} "
            "(DataFrame.eval), sort {by, ascending?}, sample {n, seed?}, "
            "winsor {columns?, cuts?=[1,99]} (replaces in place), "
            "wide_to_long {stubnames, i, j, sep?}, long_to_wide {index, "
            "columns, values}, mice {columns?, m?} (single completed "
            "dataset), function {name, arguments?} (any sp.<name> taking "
            "data= and returning a DataFrame). query / assign expressions "
            "are limited to columns (backticks for odd names), constants, "
            "operators, in / not in, math functions (log, exp, sqrt, abs, "
            "...) and col.isnull()/isin()/between()/str.contains(); other "
            "attribute access, subscripts and @locals fail with "
            "error_kind='unsafe_expression'."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "operations": {
                    "type": "array",
                    "items": {"type": "object", "additionalProperties": True},
                    "description": "Ordered list of {op, ...} objects.",
                },
                "name": {
                    "type": "string",
                    "description": "Optional label for the new handle.",
                },
            },
            "required": ["operations"],
        },
    },
    {
        "name": "route_estimator",
        "description": (
            "Route a research question to estimator calls without data. Give "
            "the family (did / iv / rd / matching / ml_causal / qte / "
            "dynamic_panel) and answers to its decision questions; get the "
            "matching registered functions with example calls, why each is "
            "right, the assumptions it adds and the guide section to read, "
            "plus the unanswered questions and the one that narrows the "
            "choice most. Call with no answers to see the questions. The "
            "full guide is readable at statspai://guide/{family}."
        ),
        "input_schema": {
            "type": "object",
            "properties": {
                "family": {
                    "type": "string",
                    "enum": [
                        "did",
                        "iv",
                        "rd",
                        "matching",
                        "ml_causal",
                        "qte",
                        "dynamic_panel",
                    ],
                    "description": "Estimator family.",
                },
                "answers": {
                    "type": "object",
                    "additionalProperties": {"type": "string"},
                    "description": "question_key -> answer (see the questions returned when omitted).",
                },
            },
            "required": ["family"],
        },
    },
]


WORKFLOW_TOOL_NAMES = frozenset(t["name"] for t in WORKFLOW_TOOL_SPECS)


def workflow_tool_manifest() -> List[Dict[str, Any]]:
    """Return manifest entries for every workflow tool."""
    return [dict(t) for t in WORKFLOW_TOOL_SPECS]


# ----------------------------------------------------------------------
# Dispatch
# ----------------------------------------------------------------------


def execute_workflow_tool(
    name: str,
    arguments: Dict[str, Any],
    *,
    data: Optional[pd.DataFrame] = None,
    detail: str = "agent",
    result_id: Optional[str] = None,
    as_handle: bool = False,
) -> Dict[str, Any]:
    """Dispatch a workflow tool call.

    Parameters
    ----------
    name : str
        One of :data:`WORKFLOW_TOOL_NAMES`.
    arguments : dict
        Tool-call arguments (already stripped of MCP-only kwargs).
    data : DataFrame, optional
        Loaded by the MCP layer for tools that need fresh data
        (``preflight``, ``detect_design``).
    detail : str
        Forwarded to result serializers.
    result_id : str, optional
        Used as the default for tools that take ``result_id`` if the
        caller didn't include it in ``arguments``.
    as_handle : bool
        Cache the new fitted result and return ``result_id`` /
        ``result_uri``.
    """
    rid_arg = arguments.get("result_id") or result_id

    if name == "bibtex":
        return _tool_bibtex(arguments)

    if name == "search_functions":
        return _tool_search_functions(arguments)

    if name == "describe_function":
        return _tool_describe_function(arguments)

    if name == "route_estimator":
        return _tool_route_estimator(arguments)

    if name == "load_data":
        return _tool_load_data(arguments, data)

    if name == "describe_data":
        return _tool_describe_data(arguments, data)

    if name == "transform_data":
        return _tool_transform_data(arguments, data)

    if name == "call_function":
        return _tool_call_function(
            arguments,
            data=data,
            detail=detail,
            result_id=rid_arg,
            as_handle=as_handle,
        )

    if name == "plot_from_result":
        return _tool_plot_from_result(rid_arg, arguments)

    if name == "from_stata":
        return _tool_from_stata(arguments)

    if name == "from_r":
        return _tool_from_r(arguments)

    if name == "detect_design":
        return _tool_detect_design(
            arguments,
            data,
            detail=detail,
            as_handle=as_handle,
        )

    if name == "preflight":
        return _tool_preflight(
            arguments,
            data,
            detail=detail,
            as_handle=as_handle,
        )

    if name == "cross_validate":
        return _tool_cross_validate(arguments, data, detail=detail)

    if name in {"audit_result", "audit"}:
        return _tool_audit(rid_arg, detail=detail)

    if name in {"brief_result", "brief"}:
        return _tool_brief(rid_arg)

    if name == "interpret_result":
        return _tool_interpret_result(rid_arg, arguments, detail=detail)

    if name == "sensitivity_from_result":
        return _tool_sensitivity_from_result(
            rid_arg, arguments, detail=detail, as_handle=as_handle
        )

    if name == "honest_did_from_result":
        return _tool_honest_did_from_result(
            rid_arg, arguments, detail=detail, as_handle=as_handle
        )

    return {
        "error": f"workflow_tool dispatch missed name {name!r}",
        "available_workflow_tools": sorted(WORKFLOW_TOOL_NAMES),
    }


# ----------------------------------------------------------------------
# Individual tool implementations
# ----------------------------------------------------------------------


def _missing_argument_error(
    *,
    tool: str,
    argument: str,
    arguments: Dict[str, Any],
    corrected: str,
    hint: str = "",
) -> Dict[str, Any]:
    """Build a *recoverable* "wrong/missing argument" payload.

    Agent-usability rule (CLAUDE.md §3.2 / §7): an error an agent cannot act
    on costs a whole turn of guessing.  Every argument error on the
    agent-facing surface therefore states three things — what was expected,
    what actually arrived, and a corrected call the agent can copy — instead
    of the bare ``"<tool> requires <arg>"`` this replaces.

    ``got`` lists the keys the caller *did* send, which is what makes a
    near-miss (``betas=`` for ``result=``, ``df=`` for ``data_path=``)
    self-diagnosing rather than a guessing game.
    """
    got = sorted(k for k, v in arguments.items() if v is not None)
    payload: Dict[str, Any] = {
        "error": (
            f"{tool}: expected argument `{argument}`, got "
            f"{got if got else 'no arguments'}"
        ),
        "expected_argument": argument,
        "got_arguments": got,
        "try": corrected,
    }
    if hint:
        payload["hint"] = hint
    return payload


def _need_result(rid: Optional[str]) -> Any:
    """Resolve a result_id to its cached object or raise a friendly dict."""
    if not rid:
        return {
            "error": "result_id is required",
            "hint": (
                "Re-run the upstream estimator with as_handle=true "
                "to get a result_id, then pass it here."
            ),
        }
    obj = RESULT_CACHE.get(rid)
    if obj is None:
        from ._result_cache import missing_result_error

        miss = missing_result_error(rid)
        # ``reason`` is the older spelling of ``miss_reason``; kept so
        # existing agents keep branching on it.
        miss["reason"] = miss["miss_reason"]
        return miss
    return obj


def _tool_audit(rid: Optional[str], *, detail: str) -> Dict[str, Any]:
    obj = _need_result(rid)
    if isinstance(obj, dict) and "error" in obj:
        return obj
    import statspai as sp

    audit_fn = getattr(sp, "audit", None)
    if audit_fn is None:
        return {"error": "sp.audit is not available in this build"}
    try:
        report = audit_fn(obj)
    except Exception as e:
        from .remediation import remediate

        return {
            "error": f"{type(e).__name__}: {e}",
            "remediation": remediate(e, context={"tool": "audit"}),
        }
    out = _audit_to_dict(report)
    out["result_id"] = rid
    return out


def _audit_to_dict(report: Any) -> Dict[str, Any]:
    """Normalize whatever sp.audit returns into a JSON-friendly dict."""
    if isinstance(report, dict):
        return dict(report)
    to_dict = getattr(report, "to_dict", None)
    if callable(to_dict):
        out = to_dict()
        if isinstance(out, dict):
            return out
    if hasattr(report, "__dict__"):
        return {k: v for k, v in vars(report).items() if not k.startswith("_")}
    return {"value": report}


def _tool_brief(rid: Optional[str]) -> Dict[str, Any]:
    obj = _need_result(rid)
    if isinstance(obj, dict) and "error" in obj:
        return obj
    import statspai as sp

    brief_fn = getattr(sp, "brief", None)
    if brief_fn is None:
        return {"error": "sp.brief is not available in this build"}
    try:
        text = brief_fn(obj)
    except Exception as e:
        from .remediation import remediate

        return {
            "error": f"{type(e).__name__}: {e}",
            "remediation": remediate(e, context={"tool": "brief"}),
        }
    return {"brief": str(text), "result_id": rid}


# ----------------------------------------------------------------------
# interpret_result — natural-language explanation, LLM-in-the-loop
# ----------------------------------------------------------------------

_AUDIENCE_TONE = {
    "researcher": (
        "for an applied econometrician: be precise and name the "
        "identification assumptions the design relies on"
    ),
    "policymaker": (
        "for a policymaker: plain language, focus on the magnitude and "
        "what it implies for decisions"
    ),
    "general": "for a general audience: no jargon, explain any term you use",
}


def _result_summary_for_interpretation(obj: Any, *, detail: str) -> Dict[str, Any]:
    """Structured, JSON-safe summary the interpretation is grounded in.

    Grounding the LLM in the result's *own* numbers (estimate / SE / CI /
    method / diagnostics) is what keeps the natural-language explanation
    honest — the model is asked to explain these, never to invent them.
    Best-effort: every extraction is optional so an exotic cached object
    still yields *something* to interpret.
    """
    summary: Dict[str, Any] = {"result_class": type(obj).__name__}

    import statspai as sp

    brief_fn = getattr(sp, "brief", None)
    if callable(brief_fn):
        try:
            summary["brief"] = str(brief_fn(obj))
        except Exception:
            # brief() is a convenience; its failure must not sink the
            # whole interpretation — the structured fields below carry
            # the load.
            pass

    try:
        from .tools import _default_serializer

        struct = _default_serializer(obj, detail=detail)
        if isinstance(struct, dict) and struct:
            summary["fields"] = struct
    except Exception as exc:
        # §3.7: surface, don't swallow — an empty "fields" would otherwise
        # read as "the result carried no structured data".
        summary["fields_error"] = f"{type(exc).__name__}: {exc}"

    return summary


def _interpretation_prompt(
    summary: Dict[str, Any], *, question: str, audience: str
) -> str:
    """Assemble the sampling prompt — anti-hallucination by construction."""
    import json as _json

    tone = _AUDIENCE_TONE.get(audience, _AUDIENCE_TONE["researcher"])
    lines = [
        f"Interpret the following StatsPAI estimation result {tone}.",
        "",
        "Ground EVERY claim in the numbers below. Do NOT invent or alter "
        "any estimate, standard error, confidence interval, p-value, or "
        "sample size. If a quantity is not present, say it is not reported "
        "rather than guessing. Keep it to 3-6 sentences.",
        "",
        "RESULT (JSON):",
        _json.dumps(summary, indent=2, default=str),
    ]
    if question:
        lines += ["", f"Focus specifically on this question: {question}"]
    return "\n".join(lines)


def _deterministic_interpretation(obj: Any, summary: Dict[str, Any]) -> str:
    """Templated narrative used when no LLM is available — no fabrication.

    Prefers the one-line ``sp.brief`` text; otherwise stitches a sentence
    or two from whatever scalar fields the serializer surfaced.
    """
    brief = summary.get("brief")
    if isinstance(brief, str) and brief.strip():
        return brief.strip()

    fields = summary.get("fields") or {}
    parts: List[str] = []
    method = fields.get("method")
    if method:
        parts.append(f"Method: {method}.")
    est = fields.get("estimate")
    se = fields.get("std_error")
    if est is not None:
        sentence = f"Point estimate: {est:.4g}"
        if se is not None:
            sentence += f" (standard error {se:.4g})"
        parts.append(sentence + ".")
    lo, hi = fields.get("conf_low"), fields.get("conf_high")
    if lo is not None and hi is not None:
        parts.append(f"95% confidence interval: [{lo:.4g}, {hi:.4g}].")
    p = fields.get("p_value")
    if p is not None:
        parts.append(f"p-value: {p:.4g}.")
    if not parts:
        return (
            f"Fitted result of type {type(obj).__name__}; it exposes no "
            "scalar estimate for a templated summary. Connect a "
            "sampling-capable MCP client for a richer interpretation."
        )
    return " ".join(parts)


def _tool_interpret_result(
    rid: Optional[str], arguments: Dict[str, Any], *, detail: str
) -> Dict[str, Any]:
    obj = _need_result(rid)
    if isinstance(obj, dict) and "error" in obj:
        return obj

    question = str(arguments.get("question") or "").strip()
    audience = arguments.get("audience") or "researcher"

    summary = _result_summary_for_interpretation(obj, detail=detail)

    out: Dict[str, Any] = {
        "result_id": rid,
        "result_class": type(obj).__name__,
        "audience": audience,
        "summary": summary,
    }
    if question:
        out["question"] = question

    # ── The wiring ──────────────────────────────────────────────────
    # resolve_llm_client() returns a SamplingLLMClient when the MCP
    # client advertised capabilities.sampling (reusing the agent's own
    # model, no API key), else None so we degrade to the deterministic
    # brief. It never raises — resolution failure means "no LLM".
    from ..causal_llm.sampling_client import resolve_llm_client

    client = resolve_llm_client()

    if client is None:
        out["interpretation"] = _deterministic_interpretation(obj, summary)
        out["backend"] = "deterministic"
        out["note"] = (
            "No MCP sampling advertised; returned a deterministic brief. "
            "Connect a sampling-capable client for a natural-language "
            "explanation that reuses the agent's own model."
        )
        return out

    prompt = _interpretation_prompt(summary, question=question, audience=audience)
    try:
        text = client.chat("user", prompt)
    except Exception as exc:
        # Mid-call sampling failure (timeout / client error). Fall back
        # LOUDLY: surface the error in the payload (CLAUDE.md §3 #7 —
        # 失败要响亮) rather than returning nothing or a wrong narrative.
        out["interpretation"] = _deterministic_interpretation(obj, summary)
        out["backend"] = "deterministic"
        out["sampling_error"] = f"{type(exc).__name__}: {exc}"
        out["note"] = (
            "MCP sampling failed mid-call; fell back to the deterministic "
            "brief. See sampling_error."
        )
        return out

    out["interpretation"] = str(text).strip()
    out["backend"] = getattr(client, "name", "mcp_sampling")
    return out


def _tool_sensitivity_from_result(
    rid: Optional[str],
    arguments: Dict[str, Any],
    *,
    detail: str,
    as_handle: bool,
) -> Dict[str, Any]:
    obj = _need_result(rid)
    if isinstance(obj, dict) and "error" in obj:
        return obj
    method = arguments.get("method", "evalue")
    benchmark = arguments.get("benchmark_covariate")

    import statspai as sp

    try:
        if method == "evalue":
            fn = getattr(sp, "evalue_from_result", None) or getattr(
                sp,
                "evalue",
                None,
            )
            result = fn(obj) if fn else None
        elif method == "oster":
            fn = getattr(sp, "oster_bounds", None)
            result = fn(obj) if fn else None
        elif method == "cinelli_hazlett":
            fn = getattr(sp, "sensemakr", None)
            kwargs = {"benchmark_covariate": benchmark} if benchmark else {}
            result = fn(obj, **kwargs) if fn else None
        else:
            fn = getattr(sp, "sensitivity", None)
            # The caller already told us which column is the treatment;
            # forward it as term= so the dashboard analyses that coefficient.
            # Without it a multi-regressor fit used to be resolved by taking
            # params.iloc[0] — the intercept — and the agent received the
            # intercept's sensitivity presented as the treatment's.
            treat_name = arguments.get("treat") or arguments.get("treatment")
            sens_kwargs = {"term": treat_name} if treat_name else {}
            result = fn(obj, **sens_kwargs) if fn else None
    except Exception as e:
        from .remediation import remediate

        return {
            "error": f"{type(e).__name__}: {e}",
            "remediation": remediate(
                e,
                context={"tool": "sensitivity_from_result"},
            ),
        }
    if result is None:
        return {
            "error": f"sensitivity method {method!r} not available in this build",
        }

    from .tools import _default_serializer

    out = _default_serializer(result, detail=detail)
    if not isinstance(out, dict):
        out = {"value": out}
    out["source_result_id"] = rid
    new_rid: Optional[str] = None
    if as_handle:
        new_rid = RESULT_CACHE.put(
            result,
            tool="sensitivity_from_result",
            arguments={"source": rid, "method": method},
        )
        out["result_id"] = new_rid
        out["result_uri"] = f"statspai://result/{new_rid}"
    from ._enrichment import enrich_payload

    # Enrichment uses the underlying sensitivity method as the tool key
    # (evalue / oster / cinelli_hazlett / sensitivity) so citations point
    # to the correct paper.
    enrich_key = (
        method
        if method in {"evalue", "oster_bounds", "sensemakr", "sensitivity"}
        else "sensitivity"
    )
    if method == "oster":
        enrich_key = "oster_bounds"
    elif method == "cinelli_hazlett":
        enrich_key = "sensemakr"
    enrich_payload(out, tool_name=enrich_key, result_id=new_rid)
    return out


def _tool_honest_did_from_result(
    rid: Optional[str],
    arguments: Dict[str, Any],
    *,
    detail: str,
    as_handle: bool,
) -> Dict[str, Any]:
    obj = _need_result(rid)
    if isinstance(obj, dict) and "error" in obj:
        return obj

    import statspai as sp

    fn = getattr(sp, "honest_did", None)
    if fn is None:
        return {"error": "sp.honest_did is not available in this build"}
    method_arg = str(arguments.get("method", "SD"))
    method_key = method_arg.lower()
    method = {
        "sd": "smoothness",
        "rm": "relative_magnitude",
        "smoothness": "smoothness",
        "relative_magnitude": "relative_magnitude",
    }.get(method_key, method_arg)
    event_time = int(arguments.get("e", 0))
    m_bar = arguments.get("m_bar")
    m_grid = [float(m_bar)] if m_bar is not None else None

    event_result = _coerce_event_study_result(obj)
    call_kwargs: Dict[str, Any] = {"e": event_time, "method": method}
    if m_grid is not None:
        call_kwargs["m_grid"] = m_grid
    try:
        result = fn(event_result, **call_kwargs)
    except Exception as exc:
        # There is deliberately no legacy fallback here.  A branch that
        # re-called ``sp.honest_did(betas=..., sigma=...,
        # num_pre_periods=..., num_post_periods=..., method=...)`` used to
        # live at this spot; that signature no longer exists — the current
        # one is ``honest_did(result, e=0, m_grid=None, method=...)`` — so
        # the branch could only ever raise ``TypeError: honest_did() got an
        # unexpected keyword argument 'betas'``, masking the real upstream
        # error behind a bogus one.  Report what actually happened instead.
        from .remediation import remediate

        rendered = ", ".join(
            [f"<result_id={rid!r}>"] + [f"{k}={v!r}" for k, v in call_kwargs.items()]
        )
        payload: Dict[str, Any] = {
            "error": f"{type(exc).__name__}: {exc}",
            "failed_call": f"sp.honest_did({rendered})",
            "hint": (
                "sp.honest_did takes a fitted result as its first positional "
                "argument: honest_did(result, e=0, m_grid=None, "
                "method='smoothness'|'relative_magnitude'). It does NOT take "
                "betas= / sigma= / num_pre_periods= / num_post_periods=. "
                "honest_did_from_result expects a result_id produced by "
                "sp.event_study / sp.callaway_santanna / sp.did_imputation / "
                "sp.sun_abraham — run one of those with as_handle=true first."
            ),
            "remediation": remediate(
                exc,
                context={"tool": "honest_did_from_result"},
            ),
        }
        betas, sigma, _n_pre, _n_post = _extract_event_study(obj)
        if betas is None or sigma is None:
            payload["diagnosis"] = (
                "no event-study coefficients + covariance could be found on "
                "the cached result, so it is very likely not an event-study "
                "/ staggered-DiD result at all."
            )
        else:
            payload["diagnosis"] = (
                "event-study coefficients were found on the cached result, so "
                "the failure is in sp.honest_did itself rather than in the "
                "shape of the upstream result."
            )
        return payload

    if isinstance(result, pd.DataFrame):
        out = {
            "method": "Rambachan-Roth (2023) honest DiD",
            "restriction": method,
            "event_time": event_time,
            "rows": result.to_dict(orient="records"),
            "max_rejecting_M": (
                float(result.loc[result["rejects_zero"], "M"].max())
                if "rejects_zero" in result and bool(result["rejects_zero"].any())
                else 0.0
            ),
        }
    else:
        from .tools import _default_serializer

        out = _default_serializer(result, detail=detail)
    if not isinstance(out, dict):
        out = {"value": out}
    out["source_result_id"] = rid
    new_rid: Optional[str] = None
    if as_handle:
        new_rid = RESULT_CACHE.put(
            result,
            tool="honest_did_from_result",
            arguments={"source": rid, "method": method, "e": event_time},
        )
        out["result_id"] = new_rid
        out["result_uri"] = f"statspai://result/{new_rid}"
    from ._enrichment import enrich_payload

    enrich_payload(out, tool_name="honest_did", result_id=new_rid)
    return out


def _coerce_event_study_result(obj: Any) -> Any:
    """Return an object shaped for the current ``sp.honest_did`` API."""
    detail = getattr(obj, "detail", None)
    if isinstance(detail, pd.DataFrame) and {"relative_time", "att", "se"} <= set(
        detail.columns
    ):
        return obj

    method = str(getattr(obj, "method", "")).lower()
    if "callaway" in method and detail is not None:
        import statspai as sp

        try:
            return sp.aggte(obj, type="dynamic", bstrap=False)
        except TypeError:
            return sp.aggte(obj, type="dynamic")
    return obj


def _extract_event_study(obj: Any) -> Tuple[Any, Any, Any, Any]:
    """Best-effort extraction of (betas, sigma, n_pre, n_post)."""
    import numpy as np

    # Direct attribute lookup
    betas = (
        getattr(obj, "event_study_betas", None)
        or getattr(obj, "betas", None)
        or getattr(obj, "coefficients", None)
    )
    sigma = (
        getattr(obj, "event_study_sigma", None)
        or getattr(obj, "sigma", None)
        or getattr(obj, "vcov", None)
    )
    n_pre = getattr(obj, "num_pre_periods", None) or getattr(obj, "n_pre", None)
    n_post = getattr(obj, "num_post_periods", None) or getattr(obj, "n_post", None)
    # Common nested shape: result.event_study has its own betas / sigma
    if betas is None or sigma is None:
        es = getattr(obj, "event_study", None)
        if es is not None:
            betas = betas or getattr(es, "betas", None)
            sigma = sigma or getattr(es, "sigma", None)
            n_pre = n_pre or getattr(es, "num_pre_periods", None)
            n_post = n_post or getattr(es, "num_post_periods", None)
    if betas is None or sigma is None:
        return None, None, None, None
    try:
        betas_arr = np.asarray(betas, dtype=float).ravel()
        sigma_arr = np.asarray(sigma, dtype=float)
        if sigma_arr.ndim == 1:
            sigma_arr = np.diag(sigma_arr)
    except Exception as exc:
        warnings.warn(
            "honest_did fallback: event-study betas/sigma on the cached "
            f"result could not be coerced to float arrays ({exc!r}); "
            "the sensitivity analysis will be skipped.",
            stacklevel=2,
        )
        return None, None, None, None
    if n_pre is None or n_post is None:
        # Heuristic: half-and-half when caller didn't tell us
        total = betas_arr.shape[0]
        n_pre_h = total // 2
        n_post_h = total - n_pre_h
        n_pre = n_pre or n_pre_h
        n_post = n_post or n_post_h
    return betas_arr, sigma_arr, n_pre, n_post


def _listify_sigma(sigma: Any) -> List[List[float]]:
    return [[float(x) for x in row] for row in sigma]


# ----------------------------------------------------------------------
# Workflow primitives that take a DataFrame
# ----------------------------------------------------------------------


def _tool_detect_design(
    arguments: Dict[str, Any],
    data: Optional[pd.DataFrame],
    *,
    detail: str,
    as_handle: bool,
) -> Dict[str, Any]:
    if data is None:
        return _missing_argument_error(
            tool="detect_design",
            argument="data_path",
            arguments=arguments,
            corrected="detect_design(data_path='panel.csv')",
        )
    import statspai as sp

    fn = getattr(sp, "detect_design", None)
    if fn is None:
        return {"error": "sp.detect_design is not available"}
    # The advertised schema uses agent-facing hint names
    # (``id_col_hint`` / ``time_col_hint``); the underlying
    # ``sp.detect_design`` takes ``unit`` / ``time``. Translate so an
    # agent following the manifest does not hit a TypeError. Honoring the
    # advertised schema in dispatch keeps schemas/*.json byte-identical.
    _HINT_MAP = {"id_col_hint": "unit", "time_col_hint": "time"}
    kwargs = {_HINT_MAP.get(k, k): v for k, v in arguments.items() if v is not None}
    try:
        out = fn(data, **kwargs)
    except Exception as e:
        from .remediation import remediate

        return {
            "error": f"{type(e).__name__}: {e}",
            "remediation": remediate(e, context={"tool": "detect_design"}),
        }
    if isinstance(out, dict):
        result_dict = dict(out)
    elif hasattr(out, "to_dict"):
        result_dict = out.to_dict()
    else:
        result_dict = {"value": str(out)}
    if as_handle:
        rid = RESULT_CACHE.put(out, tool="detect_design", arguments=arguments)
        result_dict["result_id"] = rid
        result_dict["result_uri"] = f"statspai://result/{rid}"
    return result_dict


def _tool_preflight(
    arguments: Dict[str, Any],
    data: Optional[pd.DataFrame],
    *,
    detail: str,
    as_handle: bool,
) -> Dict[str, Any]:
    if data is None:
        return _missing_argument_error(
            tool="preflight",
            argument="data_path",
            arguments=arguments,
            corrected="preflight(data_path='panel.csv', method='did')",
        )
    import statspai as sp

    fn = getattr(sp, "preflight", None)
    if fn is None:
        return {"error": "sp.preflight is not available"}
    method = arguments.get("method")
    if not method:
        return _missing_argument_error(
            tool="preflight",
            argument="method",
            arguments=arguments,
            corrected="preflight(data_path='panel.csv', method='did')",
            hint=(
                "`method` names the design you are about to run — e.g. "
                "'did', 'iv', 'rd', 'synth'."
            ),
        )
    kwargs = {k: v for k, v in arguments.items() if k != "method" and v is not None}
    try:
        out = fn(data, method, **kwargs)
    except Exception as e:
        from .remediation import remediate

        return {
            "error": f"{type(e).__name__}: {e}",
            "remediation": remediate(e, context={"tool": "preflight"}),
        }
    if isinstance(out, dict):
        result_dict = dict(out)
    elif hasattr(out, "to_dict"):
        result_dict = out.to_dict()
    else:
        result_dict = {"value": str(out), "verdict": getattr(out, "verdict", None)}
    if as_handle:
        rid = RESULT_CACHE.put(out, tool="preflight", arguments=arguments)
        result_dict["result_id"] = rid
        result_dict["result_uri"] = f"statspai://result/{rid}"
    return result_dict


def _tool_cross_validate(
    arguments: Dict[str, Any],
    data: Optional[pd.DataFrame],
    *,
    detail: str,
) -> Dict[str, Any]:
    """Run sp.cross_validate on freshly-loaded data and serialise the verdict."""
    if data is None:
        return _missing_argument_error(
            tool="cross_validate",
            argument="data_path",
            arguments=arguments,
            corrected="cross_validate(data_path='panel.csv', estimand='att')",
        )
    import statspai as sp

    fn = getattr(sp, "cross_validate", None)
    if fn is None:  # pragma: no cover - cross_validate is a core export
        return {"error": "sp.cross_validate is not available"}
    estimand = arguments.get("estimand")
    if not estimand:
        return _missing_argument_error(
            tool="cross_validate",
            argument="estimand",
            arguments=arguments,
            corrected="cross_validate(data_path='panel.csv', estimand='att')",
        )
    kwargs = {
        k: v for k, v in arguments.items() if k not in ("estimand",) and v is not None
    }
    try:
        out = fn(data, estimand, **kwargs)
    except Exception as e:
        from .remediation import remediate

        return {
            "error": f"{type(e).__name__}: {e}",
            "remediation": remediate(e, context={"tool": "cross_validate"}),
        }
    payload: Dict[str, Any] = out.to_dict(detail=detail)
    return payload


# ----------------------------------------------------------------------
# plot_from_result — emit a PNG image content block
# ----------------------------------------------------------------------

#: Map from result class-name patterns → plot kind. Highest-priority
#: match wins, so order matters: more-specific patterns first.
_PLOT_KIND_BY_CLASS: List[Tuple[str, str]] = [
    ("CallawaySantannaResult", "event_study"),
    ("EventStudyResult", "event_study"),
    ("DIDResult", "event_study"),
    ("HonestDIDResult", "honest_did"),
    ("BaconDecompositionResult", "bacon"),
    ("RDResult", "rdplot"),
    ("RDRobustResult", "rdplot"),
    ("RDDensityResult", "rddensity"),
    ("SynthResult", "synth_gap"),
    ("SynthDIDResult", "synth_gap"),
    ("MatchingResult", "love_plot"),
    ("EBalanceResult", "love_plot"),
    ("CausalForestResult", "cate_plot"),
    ("MetalearnerResult", "cate_plot"),
    ("CausalResult", "coef_plot"),
    ("EconometricResults", "coef_plot"),
]


def _detect_plot_kind(obj: Any) -> str:
    cls_name = type(obj).__name__
    for pattern, kind in _PLOT_KIND_BY_CLASS:
        if pattern in cls_name:
            return kind
    return "coef_plot"


def _render_plot_png(
    obj: Any,
    kind: str,
    figsize: Any = (8, 5),
) -> Optional[bytes]:
    """Best-effort rendering of ``obj`` to a PNG byte string.

    Returns ``None`` when matplotlib isn't installed or the result
    type doesn't expose a plot path. The caller should treat ``None``
    as "rendered nothing" and emit a JSON-only response.
    """
    try:
        import matplotlib

        matplotlib.use("Agg", force=False)
        import matplotlib.pyplot as plt
    except Exception:
        return None

    import io

    fig = None
    try:
        # Preferred: result-attached plot methods. The CausalResult class
        # exposes ``.plot()``; some result types accept a ``kind=`` kwarg.
        plot_fn = getattr(obj, "plot", None)
        if callable(plot_fn):
            try:
                ret = plot_fn(kind=kind, figsize=figsize)
            except TypeError:
                # Older signature without ``kind``/``figsize``
                try:
                    ret = plot_fn()
                except TypeError:
                    ret = None
            fig = _coerce_to_fig(ret)

        if fig is None:
            # Fallback to family-specific plot helpers on statspai.
            import statspai as sp

            helper = None
            if kind == "event_study":
                helper = (
                    getattr(sp, "event_study_table", None)
                    or getattr(sp, "enhanced_event_study_plot", None)
                    or getattr(sp, "cohort_event_study_plot", None)
                )
            elif kind == "rdplot":
                helper = getattr(sp, "rdplot", None)
            elif kind == "rddensity":
                helper = getattr(sp, "rdplotdensity", None)
            elif kind == "synth_gap":
                helper = getattr(sp, "synthdid_plot", None)
            elif kind == "love_plot":
                helper = getattr(sp, "love_plot", None) or getattr(
                    sp,
                    "balanceplot",
                    None,
                )
            elif kind == "cate_plot":
                helper = getattr(sp, "cate_plot", None)
            elif kind == "bacon":
                helper = getattr(sp, "bacon_plot", None)
            if callable(helper):
                try:
                    ret = helper(obj)
                except Exception:
                    ret = None
                fig = _coerce_to_fig(ret)

        if fig is None:
            return None

        buf = io.BytesIO()
        fig.savefig(buf, format="png", dpi=120, bbox_inches="tight")
        plt.close(fig)
        return buf.getvalue()
    except Exception:
        if fig is not None:
            try:
                plt.close(fig)
            except Exception:
                pass
        return None


def _coerce_to_fig(ret: Any) -> Any:
    """Best-effort: turn whatever a plot helper returned into a Figure."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.axes import Axes
        from matplotlib.figure import Figure
    except Exception:
        return None
    if isinstance(ret, Figure):
        return ret
    if isinstance(ret, Axes):
        return ret.figure
    if isinstance(ret, (list, tuple)) and ret:
        for item in ret:
            fig = _coerce_to_fig(item)
            if fig is not None:
                return fig
    # No useful return; rely on the active figure if matplotlib has one.
    return plt.gcf() if plt.get_fignums() else None


def _tool_plot_from_result(
    rid: Optional[str],
    arguments: Dict[str, Any],
) -> Dict[str, Any]:
    obj = _need_result(rid)
    if isinstance(obj, dict) and "error" in obj:
        return obj
    kind = arguments.get("kind") or _detect_plot_kind(obj)
    figsize = arguments.get("figsize") or (8, 5)
    if isinstance(figsize, list):
        figsize = tuple(figsize[:2]) if len(figsize) >= 2 else (8, 5)

    png = _render_plot_png(obj, kind, figsize=figsize)
    if png is None:
        return {
            "error": (
                "Could not render a plot for this result. "
                "matplotlib may not be installed, or the result "
                "class does not expose a plot path."
            ),
            "result_class": type(obj).__name__,
            "attempted_kind": kind,
            "fix": "pip install matplotlib  # or pass kind='coef_plot'",
        }
    return {
        "result_id": rid,
        "kind": kind,
        "figsize": list(figsize),
        "mime_type": "image/png",
        "image_bytes": len(png),
        # The MCP server promotes ``_plot_png`` to an image content
        # block; the underscore-prefixed key is dropped from the JSON
        # text payload so the agent doesn't see raw base64 in chat.
        "_plot_png": png,
    }


# ----------------------------------------------------------------------
# from_stata / from_r — Stata/R command translators
# ----------------------------------------------------------------------


def _tool_from_stata(arguments: Dict[str, Any]) -> Dict[str, Any]:
    cmd = arguments.get("command") or arguments.get("line") or ""
    if not isinstance(cmd, str) or not cmd.strip():
        return {
            "error": "`command` is required (one Stata command).",
            "example": {"command": "reghdfe y x, absorb(id year) cluster(id)"},
        }
    from ._translation import from_stata

    out = from_stata(cmd)
    out["source"] = "stata"
    out["input"] = cmd
    return out


def _tool_from_r(arguments: Dict[str, Any]) -> Dict[str, Any]:
    expr = arguments.get("expression") or arguments.get("line") or ""
    if not isinstance(expr, str) or not expr.strip():
        return {
            "error": "`expression` is required (one R expression).",
            "example": {"expression": 'feols(y ~ x | id, data=df, cluster="id")'},
        }
    from ._translation import from_r

    out = from_r(expr)
    out["source"] = "r"
    out["input"] = expr
    return out


# ----------------------------------------------------------------------
# bibtex tool — citation source-of-truth lookup
# ----------------------------------------------------------------------

_BIBTEX_CACHE: Optional[Dict[str, str]] = None


def _load_bibtex_index() -> Dict[str, str]:
    """Parse paper.bib once and cache key → entry text mapping.

    The parser is intentionally simple — paper.bib uses standard
    ``@article{key, ...}`` syntax with balanced braces. A heavyweight
    bibtex parser would add a dependency; this hand-rolled version
    handles every entry in the project's bib file. ``tools/bib_subset.py``
    mirrors the same brace-matching rules for the per-paper subsets.
    """
    global _BIBTEX_CACHE
    if _BIBTEX_CACHE is not None:
        return _BIBTEX_CACHE

    # Resolved via statspai._bibpath: the repo-root paper.bib under a
    # source checkout, else the byte-identical copy shipped inside the
    # wheel. Raises FileNotFoundError (never returns an empty index) when
    # neither exists — an empty bibliography would let a key "resolve" to
    # nothing and invite a fabricated citation (CLAUDE.md §10).
    from .._bibpath import read_master_bib

    text = read_master_bib()
    entries: Dict[str, str] = {}
    i = 0
    while i < len(text):
        at = text.find("@", i)
        if at < 0:
            break
        # Skip ``@string{...}`` / ``@comment{...}`` non-bib entries.
        brace = text.find("{", at)
        if brace < 0:
            break
        kind = text[at + 1 : brace].strip().lower()
        if kind in {"string", "comment", "preamble"}:
            i = brace + 1
            continue
        # Find the matching closing brace via depth counting.
        depth = 1
        j = brace + 1
        while j < len(text) and depth > 0:
            ch = text[j]
            if ch == "{":
                depth += 1
            elif ch == "}":
                depth -= 1
            j += 1
        entry = text[at:j]
        # Key is the bit between '{' and the first ',' inside the entry.
        comma = entry.find(",", brace - at)
        if comma > 0:
            key = entry[(brace - at) + 1 : comma].strip()
            if key:
                entries[key] = entry.strip()
        i = j

    _BIBTEX_CACHE = entries
    return _BIBTEX_CACHE


# ----------------------------------------------------------------------
# Data handles — load_data / describe_data / transform_data
# ----------------------------------------------------------------------


def _need_data(
    data: Optional[pd.DataFrame], *, tool: str, arguments: Dict[str, Any]
) -> Any:
    if data is None:
        return _missing_argument_error(
            tool=tool,
            argument="data_id",
            arguments=arguments,
            corrected=(
                f"{tool}(data_id='d_…')  # or data_path=..., data_records=[...], "
                "data_csv='...'"
            ),
        )
    return data


def _tool_load_data(
    arguments: Dict[str, Any], data: Optional[pd.DataFrame]
) -> Dict[str, Any]:
    from ._data_cache import describe_frame, register_frame

    frame = _need_data(data, tool="load_data", arguments=arguments)
    if isinstance(frame, dict):
        return frame
    prov = dict(arguments.get("_source_provenance") or {})
    source_id = arguments.get("_source_data_id")
    if source_id:
        # Re-registering a handle is a no-op alias: return the same id so
        # an agent cannot accidentally duplicate a frame in the cache.
        out: Dict[str, Any] = {"data_id": source_id, "aliased": True}
        out.update(describe_frame(frame))
        return out
    label = arguments.get("name")
    if isinstance(label, str) and label.strip():
        prov["label"] = label.strip()
    data_id = register_frame(frame, provenance=prov, tool="load_data")
    out = {"data_id": data_id, "data_uri": f"statspai://data/{data_id}"}
    out.update(describe_frame(frame))
    out["provenance"] = prov
    out["next_step"] = (
        "Pass data_id to any estimator (instead of data_path), or "
        "transform_data(data_id=..., operations=[...]) for a derived handle."
    )
    return out


def _tool_describe_data(
    arguments: Dict[str, Any], data: Optional[pd.DataFrame]
) -> Dict[str, Any]:
    from ._data_cache import describe_frame, lineage

    frame = _need_data(data, tool="describe_data", arguments=arguments)
    if isinstance(frame, dict):
        return frame
    head = arguments.get("head", 5)
    try:
        head = max(0, min(int(head), 50))
    except (TypeError, ValueError):
        head = 5
    out: Dict[str, Any] = describe_frame(frame, head=head)
    source_id = arguments.get("_source_data_id")
    if source_id:
        out["data_id"] = source_id
        out["lineage"] = lineage(source_id)
    prov = arguments.get("_source_provenance")
    if prov:
        out["provenance"] = prov
    return out


_TRANSFORM_OPS = (
    "query",
    "select",
    "drop",
    "rename",
    "dropna",
    "fillna",
    "assign",
    "sort",
    "sample",
    "winsor",
    "wide_to_long",
    "long_to_wide",
    "mice",
    "function",
)


class UnsafeExpression(_MethodIncompatibility):
    """A ``transform_data`` expression uses syntax outside the allowlist."""

    code = "unsafe_expression"


#: Element-wise functions ``query`` / ``assign`` expressions may call
#: (the math functions pandas ``eval`` supports).
_EXPR_FUNCS = frozenset(
    {
        "abs",
        "log",
        "log10",
        "log1p",
        "exp",
        "expm1",
        "sqrt",
        "sin",
        "cos",
        "tan",
        "sinh",
        "cosh",
        "tanh",
        "arcsin",
        "arccos",
        "arctan",
        "arcsinh",
        "arccosh",
        "arctanh",
        "arctan2",
    }
)

#: Column methods allowed as ``<column>.<method>(...)``.
_EXPR_METHODS = frozenset({"isnull", "notnull", "isna", "notna", "isin", "between"})

#: String methods allowed as ``<column>.str.<method>(...)``.
_EXPR_STR_METHODS = frozenset({"contains", "startswith", "endswith"})


def _validate_expression(expr: str) -> None:
    """Reject a ``query`` / ``assign`` expression outside a small allowlist.

    ``DataFrame.query`` / ``eval`` with ``engine="python"`` evaluate
    Python syntax, including attribute access, subscripts and local
    ``@name`` references — a model-supplied string must not reach them
    unchecked. Allowed: column names (bare or backtick-quoted), constants,
    arithmetic / comparison / boolean / bitwise operators, ``in`` /
    ``not in``, list and tuple literals, the math functions in
    :data:`_EXPR_FUNCS`, and ``col.isnull()``-style methods
    (:data:`_EXPR_METHODS`, ``col.str.contains(...)``). Everything else —
    other attribute access, dunder names, subscripts, lambdas,
    comprehensions, ``@`` references, ``@`` matrix products — raises
    :class:`UnsafeExpression`.
    """
    import ast
    import re

    # Backtick-quoted column names are opaque to Python's parser.
    masked = re.sub(r"`[^`]*`", "_statspai_bt_col_", expr)
    try:
        tree = ast.parse(masked.strip(), mode="eval")
    except SyntaxError as e:
        raise UnsafeExpression(
            f"Expression {expr!r} is not allowed: {e.msg}.",
            recovery_hint=(
                "Use column names, constants and operators only; local "
                "variable references (@name) are not supported."
            ),
        ) from e

    ops_ok = (
        ast.Expression,
        ast.BoolOp,
        ast.And,
        ast.Or,
        ast.BinOp,
        ast.Add,
        ast.Sub,
        ast.Mult,
        ast.Div,
        ast.FloorDiv,
        ast.Mod,
        ast.Pow,
        ast.BitAnd,
        ast.BitOr,
        ast.BitXor,
        ast.UnaryOp,
        ast.Not,
        ast.USub,
        ast.UAdd,
        ast.Invert,
        ast.Compare,
        ast.Eq,
        ast.NotEq,
        ast.Lt,
        ast.LtE,
        ast.Gt,
        ast.GtE,
        ast.In,
        ast.NotIn,
        ast.Is,
        ast.IsNot,
        ast.Load,
        ast.List,
        ast.Tuple,
        ast.keyword,
    )

    def _reject(node: Any, why: str) -> None:
        raise UnsafeExpression(
            f"Expression {expr!r} is not allowed: {why}.",
            recovery_hint=(
                "Allowed: columns, constants, + - * / // % **, comparisons, "
                "and/or/not, & | ~, in / not in, list literals, math "
                "functions (log, exp, sqrt, abs, ...), "
                "col.isnull()/notnull()/isna()/notna()/isin()/between(), "
                "col.str.contains()/startswith()/endswith()."
            ),
            diagnostics={"node": type(node).__name__},
        )

    def _check_name(node: Any) -> None:
        if "__" in node.id:
            _reject(node, f"name {node.id!r} contains a dunder")

    def _walk(node: Any) -> None:
        if isinstance(node, ast.Name):
            _check_name(node)
            return
        if isinstance(node, ast.Constant):
            if not isinstance(node.value, (str, int, float, bool, type(None))):
                _reject(node, "unsupported constant")
            return
        if isinstance(node, ast.Call):
            func = node.func
            if isinstance(func, ast.Name):
                if func.id not in _EXPR_FUNCS:
                    _reject(node, f"call to {func.id!r}")
            elif isinstance(func, ast.Attribute):
                owner = func.value
                if func.attr in _EXPR_METHODS and isinstance(owner, ast.Name):
                    _check_name(owner)
                elif (
                    func.attr in _EXPR_STR_METHODS
                    and isinstance(owner, ast.Attribute)
                    and owner.attr == "str"
                    and isinstance(owner.value, ast.Name)
                ):
                    _check_name(owner.value)
                else:
                    _reject(node, f"method call .{func.attr}()")
            else:
                _reject(node, "call target")
            for arg in node.args:
                _walk(arg)
            for kw in node.keywords:
                if kw.arg is None:
                    _reject(node, "**kwargs")
                _walk(kw.value)
            return
        if not isinstance(node, ops_ok):
            _reject(node, f"{type(node).__name__} syntax")
        for child in ast.iter_child_nodes(node):
            _walk(child)

    _walk(tree)


def _apply_transform(df: pd.DataFrame, step: Dict[str, Any]) -> pd.DataFrame:
    """Apply one ``transform_data`` step. Raises on a bad step."""
    from ..exceptions import MethodIncompatibility

    op = step.get("op")
    if op not in _TRANSFORM_OPS:
        raise MethodIncompatibility(
            f"Unknown transform op {op!r}.",
            recovery_hint=f"Use one of: {', '.join(_TRANSFORM_OPS)}.",
            diagnostics={"step": step},
        )

    def _cols(key: str = "columns", required: bool = True) -> List[str]:
        cols = step.get(key)
        if cols is None:
            if required:
                raise MethodIncompatibility(
                    f"op={op!r} needs `{key}`.",
                    recovery_hint=f"Add {key}=[...] to the step.",
                    diagnostics={"step": step},
                )
            return []
        if isinstance(cols, str):
            cols = [cols]
        missing = [c for c in cols if c not in df.columns]
        if missing:
            raise MethodIncompatibility(
                f"op={op!r}: column(s) not found: {missing}. "
                f"Available: {list(map(str, df.columns))[:40]}",
                recovery_hint="Check the column names with describe_data.",
                diagnostics={"missing": missing},
            )
        return list(cols)

    if op == "query":
        expr = step.get("expr")
        if not isinstance(expr, str) or not expr.strip():
            raise MethodIncompatibility(
                "op='query' needs a string `expr`.",
                recovery_hint="e.g. {'op': 'query', 'expr': 'year >= 2005 and age < 65'}",
            )
        _validate_expression(expr)
        return df.query(expr, engine="python")
    if op == "select":
        return df[_cols()]
    if op == "drop":
        return df.drop(columns=_cols())
    if op == "rename":
        mapping = step.get("mapping")
        if not isinstance(mapping, dict):
            raise MethodIncompatibility(
                "op='rename' needs `mapping` {old: new}.",
                recovery_hint="e.g. {'op': 'rename', 'mapping': {'lwage': 'y'}}",
            )
        unknown = [k for k in mapping if k not in df.columns]
        if unknown:
            raise MethodIncompatibility(
                f"op='rename': column(s) not found: {unknown}",
                recovery_hint="Check the column names with describe_data.",
            )
        return df.rename(columns=mapping)
    if op == "dropna":
        cols = _cols(required=False)
        return df.dropna(subset=cols or None)
    if op == "fillna":
        value = step.get("value", step.get("mapping"))
        if value is None:
            raise MethodIncompatibility(
                "op='fillna' needs `value` (scalar) or `mapping` {column: value}.",
                recovery_hint="e.g. {'op': 'fillna', 'mapping': {'x': 0}}",
            )
        if isinstance(value, dict):
            unknown = [k for k in value if k not in df.columns]
            if unknown:
                raise MethodIncompatibility(
                    f"op='fillna': column(s) not found: {unknown}",
                    recovery_hint="Check the column names with describe_data.",
                )
        return df.fillna(value)
    if op == "assign":
        column, expr = step.get("column"), step.get("expr")
        if not isinstance(column, str) or not isinstance(expr, str):
            raise MethodIncompatibility(
                "op='assign' needs `column` and a string `expr`.",
                recovery_hint="e.g. {'op': 'assign', 'column': 'lwage', 'expr': 'log(wage)'}",
            )
        _validate_expression(expr)
        out = df.copy()
        out[column] = out.eval(expr, engine="python")
        return out
    if op == "sort":
        by = _cols("by")
        asc = step.get("ascending", True)
        return df.sort_values(by=by, ascending=asc).reset_index(drop=True)
    if op == "sample":
        n = step.get("n")
        try:
            n = int(n)
        except (TypeError, ValueError):
            raise MethodIncompatibility(
                "op='sample' needs an integer `n`.",
                recovery_hint="e.g. {'op': 'sample', 'n': 1000, 'seed': 0}",
            )
        seed = step.get("seed", 0)
        return df.sample(n=min(n, len(df)), random_state=int(seed)).reset_index(
            drop=True
        )
    if op == "winsor":
        from ..utils.data_tools import winsor

        cols = _cols(required=False) or None
        cuts = step.get("cuts", [1, 99])
        return winsor(
            df, vars=cols, cuts=(float(cuts[0]), float(cuts[1])), replace=True
        )
    if op == "wide_to_long":
        stub = step.get("stubnames")
        i, j = step.get("i"), step.get("j")
        if not stub or not i or not j:
            raise MethodIncompatibility(
                "op='wide_to_long' needs `stubnames`, `i` and `j`.",
                recovery_hint=(
                    "e.g. {'op': 'wide_to_long', 'stubnames': ['y'], 'i': 'id', "
                    "'j': 'year', 'sep': '_'}"
                ),
            )
        return pd.wide_to_long(
            df, stubnames=stub, i=i, j=j, sep=step.get("sep", "")
        ).reset_index()
    if op == "long_to_wide":
        index, columns, values = (
            step.get("index"),
            step.get("columns"),
            step.get("values"),
        )
        if not index or not columns or not values:
            raise MethodIncompatibility(
                "op='long_to_wide' needs `index`, `columns` and `values`.",
                recovery_hint=(
                    "e.g. {'op': 'long_to_wide', 'index': 'id', 'columns': "
                    "'year', 'values': 'y'}"
                ),
            )
        wide = df.pivot(index=index, columns=columns, values=values)
        wide.columns = [f"{values}_{c}" for c in wide.columns]
        return wide.reset_index()
    if op == "mice":
        import statspai as sp

        cols = _cols(required=False) or None
        m = int(step.get("m", 5))
        res = sp.mice(df, vars=cols, m=m) if cols else sp.mice(df, m=m)
        completed = getattr(res, "completed", None) or getattr(res, "imputed", None)
        if completed is None:
            raise MethodIncompatibility(
                "sp.mice returned no completed datasets.",
                recovery_hint="Impute outside transform_data and load the result.",
            )
        first = completed[0] if isinstance(completed, (list, tuple)) else completed
        if not isinstance(first, pd.DataFrame):
            raise MethodIncompatibility(
                "sp.mice completed dataset is not a DataFrame.",
                recovery_hint="Impute outside transform_data and load the result.",
            )
        return first
    # op == "function"
    import statspai as sp

    name = step.get("name")
    if not isinstance(name, str) or not name.strip():
        raise MethodIncompatibility(
            "op='function' needs `name` (a registered sp.<name>).",
            recovery_hint="e.g. {'op': 'function', 'name': 'winsor', 'arguments': {...}}",
        )
    name = name.strip()
    if name.startswith("sp."):
        name = name[3:]
    fn = getattr(sp, name, None)
    if fn is None or not callable(fn):
        raise MethodIncompatibility(
            f"Unknown function {name!r}.",
            recovery_hint="Use search_functions to find the right name.",
        )
    kwargs = dict(step.get("arguments") or {})
    kwargs.pop("data", None)
    out = fn(data=df, **kwargs)
    if not isinstance(out, pd.DataFrame):
        raise MethodIncompatibility(
            f"sp.{name} returned {type(out).__name__}, not a DataFrame.",
            recovery_hint=(
                "transform_data only chains DataFrame-returning functions; "
                "fit estimators with call_function instead."
            ),
        )
    return out


def _tool_transform_data(
    arguments: Dict[str, Any], data: Optional[pd.DataFrame]
) -> Dict[str, Any]:
    from ..workflow._degradation import record_degradation
    from ._data_cache import describe_frame, register_frame

    frame = _need_data(data, tool="transform_data", arguments=arguments)
    if isinstance(frame, dict):
        return frame
    ops = arguments.get("operations")
    if (
        not isinstance(ops, list)
        or not ops
        or not all(isinstance(o, dict) for o in ops)
    ):
        return {
            "error": "`operations` must be a non-empty list of {op, ...} objects.",
            "ops": list(_TRANSFORM_OPS),
            "example": {
                "operations": [
                    {"op": "query", "expr": "year >= 2005"},
                    {"op": "winsor", "columns": ["wage"], "cuts": [1, 99]},
                ]
            },
        }
    df = frame
    applied: List[Dict[str, Any]] = []
    for k, step in enumerate(ops):
        before = int(len(df))
        try:
            df = _apply_transform(df, step)
        except Exception as exc:
            # A failed step aborts the chain: a partially transformed frame
            # is not what the agent asked for (CLAUDE.md §3.7).
            entry = record_degradation(
                None, section=f"transform_data step {k} ({step.get('op')})", exc=exc
            )
            envelope: Dict[str, Any] = {
                "error": f"{type(exc).__name__}: {exc}",
                "failed_step": k,
                "step": step,
                "applied": applied,
                "degradation": entry,
            }
            from ..exceptions import StatsPAIError

            if isinstance(exc, StatsPAIError):
                envelope["error_kind"] = exc.code
                envelope["recovery_hint"] = getattr(exc, "recovery_hint", None)
            return envelope
        applied.append(
            {
                "step": k,
                "op": step.get("op"),
                "arguments": {kk: vv for kk, vv in step.items() if kk != "op"},
                "n_rows_before": before,
                "n_rows_after": int(len(df)),
            }
        )
    source_id = arguments.get("_source_data_id")
    prov = dict(arguments.get("_source_provenance") or {})
    label = arguments.get("name")
    if isinstance(label, str) and label.strip():
        prov["label"] = label.strip()
    data_id = register_frame(
        df,
        provenance=prov,
        parent_id=source_id,
        operations=applied,
        tool="transform_data",
    )
    out: Dict[str, Any] = {
        "data_id": data_id,
        "data_uri": f"statspai://data/{data_id}",
        "parent_id": source_id,
        "operations": applied,
    }
    out.update(describe_frame(df))
    return out


def _tool_route_estimator(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """``route_estimator`` meta-tool: data-free estimator routing."""
    from .._routing import decision_guide, route
    from ..exceptions import StatsPAIError

    family = arguments.get("family")
    if not isinstance(family, str) or not family.strip():
        return {"error": "`family` is required.", "families": decision_guide()}
    answers = arguments.get("answers") or {}
    if not isinstance(answers, dict):
        return {"error": "`answers` must be an object of question_key -> answer."}
    try:
        if not answers:
            out: Dict[str, Any] = decision_guide(family)
            out["hint"] = (
                "Answer the questions with route_estimator(family=..., "
                "answers={key: answer, ...})."
            )
            return out
        out = route(family, **{str(k): str(v) for k, v in answers.items()})
    except StatsPAIError as exc:
        return {
            "error": str(exc),
            "error_kind": exc.code,
            "error_payload": exc.to_dict(),
        }
    out["guide_uri"] = f"statspai://guide/{family.strip().lower()}"
    return out


def _tool_search_functions(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """``search_functions`` meta-tool: ranked registry search."""
    from ..registry import search_functions

    query = arguments.get("query")
    if not isinstance(query, str) or not query.strip():
        return {
            "error": "`query` is required (keywords or a short task phrase).",
            "example": {"query": "staggered adoption event study"},
        }
    limit = arguments.get("limit", 10)
    try:
        limit = max(1, min(int(limit), 50))
    except (TypeError, ValueError):
        limit = 10
    category = arguments.get("category")
    hits = search_functions(query)
    if isinstance(category, str) and category.strip():
        cat = category.strip().lower()
        hits = [h for h in hits if str(h.get("category", "")).lower() == cat]
    return {
        "query": query,
        "n_matches": len(hits),
        "matches": hits[:limit],
        "next_step": (
            "describe_function(name=<match>) for the parameter schema, "
            "then call_function(function=<match>, arguments={...})."
        ),
    }


def _tool_describe_function(arguments: Dict[str, Any]) -> Dict[str, Any]:
    """``describe_function`` meta-tool: full registry metadata + schema."""
    from ..registry import _REGISTRY, _ensure_full_registry, describe_function
    from ..registry import function_schema as _function_schema

    name = arguments.get("name")
    if not isinstance(name, str) or not name.strip():
        return {"error": "`name` is required (a registered function name)."}
    name = name.strip()
    if name.startswith("sp."):
        name = name[3:]
    _ensure_full_registry()
    if name not in _REGISTRY:
        from difflib import get_close_matches

        return {
            "error": f"Unknown function: {name!r}.",
            "did_you_mean": get_close_matches(name, sorted(_REGISTRY), n=5, cutoff=0.6),
            "hint": "Use search_functions to find the right name.",
        }
    out: Dict[str, Any] = dict(describe_function(name))
    try:
        out["schema"] = _function_schema(name)
    except Exception as e:  # pragma: no cover - defensive
        out["schema_error"] = f"{type(e).__name__}: {e}"
    out["call_with"] = {
        "tool": "call_function",
        "arguments": {"function": name, "arguments": {"<param>": "<value>"}},
    }
    return out


def _tool_call_function(
    arguments: Dict[str, Any],
    *,
    data: Optional[pd.DataFrame],
    detail: str,
    result_id: Optional[str],
    as_handle: bool,
) -> Dict[str, Any]:
    """``call_function`` meta-tool: dispatch any registered function."""
    fn_name = arguments.get("function")
    if not isinstance(fn_name, str) or not fn_name.strip():
        return {
            "error": "`function` is required (a registered function name).",
            "example": {
                "function": "callaway_santanna",
                "arguments": {"y": "y", "g": "g", "t": "t", "i": "id"},
            },
        }
    fn_name = fn_name.strip()
    if fn_name.startswith("sp."):
        fn_name = fn_name[3:]
    if fn_name == "call_function":
        return {"error": "call_function cannot call itself."}
    inner = arguments.get("arguments") or {}
    if not isinstance(inner, dict):
        return {"error": "`arguments` must be a JSON object of keyword arguments."}
    from .tools import execute_tool

    out = execute_tool(
        fn_name,
        dict(inner),
        data=data,
        detail=detail,
        result_id=result_id,
        as_handle=as_handle,
    )
    if isinstance(out, dict):
        out.setdefault("tool", fn_name)
        out["called_via"] = "call_function"
    return out


def _tool_bibtex(arguments: Dict[str, Any]) -> Dict[str, Any]:
    from difflib import get_close_matches

    keys = arguments.get("keys") or []
    if isinstance(keys, str):
        keys = [keys]
    if not isinstance(keys, list) or not keys:
        return {
            "error": "`keys` is required (list of bib keys).",
            "example": {"keys": ["callaway2021difference", "rambachan2023more"]},
        }

    index = _load_bibtex_index()
    out_entries: Dict[str, Any] = {}
    suggestions: Dict[str, list] = {}
    for k in keys:
        k_str = str(k)
        if k_str in index:
            out_entries[k_str] = index[k_str]
        else:
            out_entries[k_str] = ""
            close = get_close_matches(k_str, list(index.keys()), n=5, cutoff=0.55)
            if close:
                suggestions[k_str] = close

    return {
        "keys": list(out_entries.keys()),
        "bibtex": out_entries,
        "unknown_keys": [k for k, v in out_entries.items() if not v],
        "suggestions": suggestions,
        "source": "paper.bib",
        "note": (
            "Empty entries mean the bib key is not in paper.bib. "
            "Do NOT fabricate — see CLAUDE.md §10."
        ),
    }


__all__ = [
    "WORKFLOW_TOOL_SPECS",
    "WORKFLOW_TOOL_NAMES",
    "workflow_tool_manifest",
    "execute_workflow_tool",
]
