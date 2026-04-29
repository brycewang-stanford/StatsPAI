"""MCP ``prompts/*`` workflow templates.

Each entry pairs a curated agent workflow description with a
``str.format_map``-friendly template body. ``prompts/get`` renders
the template; ``prompts/list`` exposes the metadata.

Decoupled from ``mcp_server.py`` so the prompt catalog can grow
without bloating the JSON-RPC dispatch file. New prompts: append a
dict to :data:`PROMPTS`. Required arguments must declare a
``description`` so the schema stays uniform — guarded by
``test_each_prompt_has_required_metadata``.
"""
from __future__ import annotations

from typing import Any, Dict, List


PROMPTS: List[Dict[str, Any]] = [
    {
        "name": "audit_did_result",
        "description": ("Run a DID estimator on a CSV, surface the "
                         "estimate, and walk through every "
                         "reviewer-checklist gap. Uses pipeline_did "
                         "to consolidate preflight + estimate + audit "
                         "+ honest-DID + Bacon into one call."),
        "arguments": [
            {"name": "data_path", "required": True,
             "description": "Absolute path to the data file."},
            {"name": "y", "required": True, "description": "Outcome column."},
            {"name": "treat", "required": True,
             "description": "Binary 0/1 treatment indicator."},
            {"name": "time", "required": True, "description": "Time column."},
        ],
        "_template": (
            "Call `pipeline_did` with data_path={data_path}, y={y}, "
            "treat={treat}, time={time}. The pipeline returns a "
            "markdown narrative with the canonical reviewer-grade "
            "DID workflow already executed (preflight, estimator, "
            "audit, honest-DID, Bacon decomposition, brief). Quote "
            "the narrative verbatim; for any high-importance check "
            "the audit flagged as missing, dispatch the corresponding "
            "entry in `next_calls`. End with a `bibtex` lookup of the "
            "keys in `citations.keys` so the user gets verified "
            "references."
        ),
    },
    {
        "name": "audit_iv_result",
        "description": ("End-to-end IV workflow: 2SLS + first-stage F + "
                         "Anderson-Rubin CI + e-value sensitivity, all "
                         "wrapped in pipeline_iv."),
        "arguments": [
            {"name": "data_path", "required": True,
             "description": "Absolute path to the data file."},
            {"name": "formula", "required": True,
             "description": "'y ~ x + (d ~ z)' Wilkinson-style."},
        ],
        "_template": (
            "Call `pipeline_iv` with data_path={data_path}, "
            "formula='{formula}'. Read `effective_F` from the "
            "response: < 10 means the Staiger-Stock weak-IV threshold "
            "is breached and you should foreground the "
            "Anderson-Rubin CI in your reply (it is in the "
            "`anderson_rubin` field) instead of the 2SLS point "
            "estimate. Cite via `bibtex(keys=...)` from the "
            "`citations.keys` list."
        ),
    },
    {
        "name": "audit_rd_result",
        "description": ("End-to-end RD workflow: rdrobust + rdplot "
                         "(image content) + density test + bandwidth "
                         "sensitivity via pipeline_rd."),
        "arguments": [
            {"name": "data_path", "required": True,
             "description": "Absolute path to the data file."},
            {"name": "y", "required": True,
             "description": "Outcome column."},
            {"name": "x", "required": True,
             "description": "Running variable column."},
            {"name": "c", "required": False,
             "description": "Cutoff value (default 0)."},
        ],
        "_template": (
            "Call `pipeline_rd` with data_path={data_path}, y={y}, "
            "x={x}, c={c}. Use the `rdplot` image content block (PNG) "
            "to anchor your reply visually. If the McCrary-style "
            "density test rejects (`rddensity` p < 0.05) flag "
            "manipulation; recommend `rdplacebo` and `rdrbounds` "
            "(emit them via `next_calls`)."
        ),
    },
    {
        "name": "design_then_estimate",
        "description": ("Given an unfamiliar CSV, auto-detect the "
                         "study design, recommend an estimator, run "
                         "it with diagnostics."),
        "arguments": [
            {"name": "data_path", "required": True,
             "description": "Absolute path to the data file."},
            {"name": "outcome", "required": True,
             "description": "Outcome column."},
            {"name": "treatment", "required": False,
             "description": "Treatment column (optional)."},
        ],
        "_template": (
            "1. Call `detect_design` with data_path={data_path}.\n"
            "2. Call `recommend` with y={outcome} (and "
            "treatment={treatment} when supplied). Read the top "
            "recommendation's `reasoning`.\n"
            "3. If the recommendation is DID/CS, call `pipeline_did`. "
            "If RD, call `pipeline_rd`. If IV, call `pipeline_iv`. "
            "Otherwise, call the recommended estimator with "
            "as_handle=true and follow up with `audit_result`.\n"
            "4. Quote the resulting `narrative`; emit the first "
            "two entries of `next_calls` for the user to consider."
        ),
    },
    {
        "name": "robustness_followup",
        "description": ("Take an existing fitted result handle and "
                         "run all high-importance follow-up "
                         "sensitivities the audit identifies as "
                         "missing."),
        "arguments": [
            {"name": "result_id", "required": True,
             "description": ("Handle from an earlier estimator call "
                              "(as_handle=true).")},
        ],
        "_template": (
            "1. Call `audit_result` with result_id={result_id}; read "
            "`items` (or `checks`) and collect every entry with "
            "status='missing' AND importance in {{'high', 'critical'}}.\n"
            "2. For each, dispatch the `suggest_function` it names. "
            "If the function takes a fitted result, pass "
            "result_id={result_id}; otherwise re-load the data via "
            "data_path.\n"
            "3. For each follow-up result, call `brief_result` and "
            "report whether the new estimate overturns the original "
            "conclusion (sign change / CI exclusion of zero)."
        ),
    },
    {
        "name": "paper_render",
        "description": ("Compose a paper-style memo from a fitted "
                         "result handle: estimate, diagnostics, "
                         "robustness, BibTeX. The output is a "
                         "ready-to-paste markdown section."),
        "arguments": [
            {"name": "result_id", "required": True,
             "description": ("Handle to a fitted result (returned by an "
                              "earlier estimator call with as_handle=true).")},
        ],
        "_template": (
            "Given result_id={result_id}:\n"
            "1. Call `brief_result` for a one-paragraph summary.\n"
            "2. Call `audit_result`; pull the audit's items with "
            "status='present' (the diagnostics that DID run) into a "
            "bulleted list.\n"
            "3. Call `plot_from_result` (auto-detects the right plot) "
            "and embed the resulting image.\n"
            "4. Call `bibtex(keys=...)` on the citation keys returned "
            "earlier; include the BibTeX bodies in a final "
            "`### References` section.\n"
            "5. Format as: '## Estimate' / '## Diagnostics' / "
            "'## Robustness' / '## Figure' / '## References'."
        ),
    },
    {
        "name": "compare_methods",
        "description": ("Run two or more estimators on the same data "
                         "and compare conclusions side by side."),
        "arguments": [
            {"name": "data_path", "required": True,
             "description": "Absolute path to the data file."},
            {"name": "y", "required": True,
             "description": "Outcome column."},
            {"name": "treat", "required": True,
             "description": "Binary treatment indicator."},
            {"name": "time", "required": False,
             "description": "Time column for panel methods."},
        ],
        "_template": (
            "Run all three: `did`, `callaway_santanna` (if cohort/id "
            "available), `did_imputation`. Use as_handle=true for "
            "each so you collect three result_ids. Then call "
            "`brief_result` on each, and report a markdown table "
            "with rows = method, columns = (estimate, 95% CI, "
            "violations flagged). Highlight any sign disagreement."
        ),
    },
    {
        "name": "policy_evaluation",
        "description": ("Causal-forest-driven policy evaluation: "
                         "fit causal_forest, summarise CATE, evaluate "
                         "a candidate policy."),
        "arguments": [
            {"name": "data_path", "required": True,
             "description": "Absolute path to the data file."},
            {"name": "formula", "required": True,
             "description": "'y ~ d | x1 + x2 + ...' (treatment | covariates)"},
        ],
        "_template": (
            "1. Call `causal_forest` with formula='{formula}' and "
            "as_handle=true.\n"
            "2. Call `cate_summary` with the result_id; report ATE + "
            "the CATE quantiles.\n"
            "3. Call `blp_test` to test whether heterogeneity is "
            "real, and `calibration_test` to check predictive quality.\n"
            "4. Call `policy_value` to estimate the value of treating "
            "everyone with positive predicted CATE."
        ),
    },
    {
        "name": "synth_full",
        "description": ("End-to-end Synthetic Control workflow: synth "
                         "fit + placebo + synthdid + permutation."),
        "arguments": [
            {"name": "data_path", "required": True,
             "description": "Absolute path to the data file."},
            {"name": "outcome", "required": True,
             "description": "Outcome column."},
            {"name": "unit", "required": True,
             "description": "Unit identifier column."},
            {"name": "time", "required": True,
             "description": "Time column."},
            {"name": "treated_unit", "required": True,
             "description": "Identifier of the treated unit."},
            {"name": "treatment_time", "required": True,
             "description": "First post-treatment period."},
        ],
        "_template": (
            "1. Call `synth` with the canonical args; as_handle=true.\n"
            "2. Call `synthdid_estimate` for the synthetic-DID "
            "alternative — the two estimates should bracket the truth.\n"
            "3. Call `synthdid_placebo` for in-space placebo "
            "inference.\n"
            "4. Call `plot_from_result` (kind='synth_gap') to "
            "visualise the treated-vs-synthetic series."
        ),
    },
    {
        "name": "decompose_inequality",
        "description": ("RIF / FFL / Oaxaca-Blinder decomposition of "
                         "an outcome gap."),
        "arguments": [
            {"name": "data_path", "required": True,
             "description": "Absolute path to the data file."},
            {"name": "y", "required": True,
             "description": "Outcome column."},
            {"name": "group", "required": True,
             "description": "Binary group indicator (e.g. gender, race)."},
            {"name": "covariates", "required": False,
             "description": "Comma-separated covariate columns."},
        ],
        "_template": (
            "Call `decompose` with method='oaxaca' (or method='rif' "
            "for distributional decomposition). Report explained vs "
            "unexplained share. If the user mentions wage gap, also "
            "run method='ffl' for the Firpo-Fortin-Lemieux variant."
        ),
    },
]


class SafeDict(dict):
    """Format-map helper that leaves unknown placeholders literal."""

    def __missing__(self, key: str) -> str:  # type: ignore[override]
        return "{" + key + "}"


def handle_prompts_list(params: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "prompts": [
            {
                "name": p["name"],
                "description": p["description"],
                "arguments": p["arguments"],
            }
            for p in PROMPTS
        ],
    }


def handle_prompts_get(params: Dict[str, Any],
                        InvalidParamsError,
                        ResourceNotFoundError) -> Dict[str, Any]:
    """Render a prompt template.

    The two error classes are passed in (rather than imported) to
    avoid a circular import — :mod:`mcp_server` owns the JSON-RPC
    error taxonomy.
    """
    name = params.get("name")
    if not isinstance(name, str):
        raise InvalidParamsError("`name` is required and must be a string")
    spec = next((p for p in PROMPTS if p["name"] == name), None)
    if spec is None:
        raise ResourceNotFoundError(
            f"Unknown prompt: {name!r}. Read prompts/list for the "
            "available templates."
        )
    args = dict(params.get("arguments") or {})
    # Validate required arguments (omit MCP would otherwise leave the
    # template with literal ``{x}`` placeholders).
    missing = [
        a["name"] for a in spec["arguments"]
        if a.get("required") and a["name"] not in args
    ]
    if missing:
        raise InvalidParamsError(
            f"prompt {name!r} missing required arguments: {missing}"
        )
    # Fill the template safely. ``str.format_map`` is single-pass —
    # it scans the *template* once for placeholders and substitutes
    # values verbatim without re-parsing the substituted text. So a
    # user value containing a literal ``{y}`` is preserved as-is in
    # the output (verified by ``test_get_with_brace_in_user_value...``).
    # ``SafeDict`` keeps unknown placeholders literal so missing
    # required-arg bugs surface instead of being silently dropped.
    template = spec["_template"]
    try:
        rendered = template.format_map(SafeDict(args))
    except Exception as e:
        raise InvalidParamsError(
            f"Failed to render prompt {name!r}: "
            f"{type(e).__name__}: {e}"
        )
    return {
        "description": spec["description"],
        "messages": [
            {"role": "user",
             "content": {"type": "text", "text": rendered}},
        ],
    }


__all__ = [
    "PROMPTS",
    "SafeDict",
    "handle_prompts_list",
    "handle_prompts_get",
]
