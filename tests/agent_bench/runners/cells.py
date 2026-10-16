"""The six experimental cells: toolset x model family.

Every cell gives the model the same two kinds of tool:

* one code-execution tool (``run_python`` or ``run_r``) whose session sits in
  a scratch directory holding the prompt's data file; and
* ``submit_answer``, a structured final answer that the grader reads.

The toolsets differ only in what the code can use (see ``sandbox.py``). The
StatsPAI cells additionally see StatsPAI's MCP tools (the ``curated``
profile), because "StatsPAI + MCP" is the configuration under test; their
definitions count toward the measured input tokens like any other tool.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Tuple

#: Estimator classes the answer schema accepts (``submit_answer.estimator``).
#: The grader maps a prompt's acceptable estimators onto these keys.
ESTIMATOR_KEYS: Tuple[str, ...] = (
    "ols",
    "ols_cluster",
    "iv_2sls",
    "callaway_santanna",
    "sun_abraham",
    "did_imputation",
    "etwfe",
    "twfe",
    "bacon",
    "honest_did",
    "rdrobust",
    "rddensity",
    "synth",
    "sdid",
    "psm",
    "ipw",
    "aipw",
    "dml_plr",
    "coxph",
    "kaplan_meier",
    "evalue",
    "sensemakr",
    "lmm",
    "glmm_logit",
    "frontier",
    "oaxaca",
    "dfl",
    "mediation",
    "hdfe_ols",
    "var",
    "local_projections",
    "other",
)


@dataclass(frozen=True)
class Cell:
    id: str
    toolset: str  # "statspai" | "pythonic" | "r"
    family: str  # "anthropic" | "openai"

    @property
    def language(self) -> str:
        return "r" if self.toolset == "r" else "python"


CELLS: Dict[str, Cell] = {
    "C1": Cell("C1", "statspai", "anthropic"),
    "C2": Cell("C2", "statspai", "openai"),
    "C3": Cell("C3", "pythonic", "anthropic"),
    "C4": Cell("C4", "pythonic", "openai"),
    "C5": Cell("C5", "r", "anthropic"),
    "C6": Cell("C6", "r", "openai"),
}

_TOOLSET_TEXT = {
    "statspai": (
        "You work in Python. The StatsPAI library is installed "
        "(`import statspai as sp`), together with pandas, numpy, scipy, "
        "statsmodels and scikit-learn. StatsPAI's MCP tools are also "
        "available; their data_path argument takes the data file's name."
    ),
    "pythonic": (
        "You work in Python with pandas, numpy, scipy, statsmodels, "
        "linearmodels, scikit-learn and DoubleML installed. StatsPAI is not "
        "available."
    ),
    "r": (
        "You work in R. Installed packages include fixest, sandwich, lmtest, "
        "AER, did, rdrobust, rddensity, Synth, synthdid, MatchIt, survival, "
        "lme4, frontier, oaxaca, mediation, sensemakr, EValue, vars, "
        "lpirf, DoubleML and HonestDiD."
    ),
}


def system_prompt(cell: Cell) -> str:
    """The system prompt: identical across cells except the toolset line."""
    return (
        "You are an applied econometrician. Answer the user's question by "
        "running code on the data file in your working directory. "
        + _TOOLSET_TEXT[cell.toolset]
        + " Each code call runs in the same session, so variables persist. "
        "When you have the answer, call submit_answer exactly once with the "
        "named values the question asks for, the estimator you used, and a "
        "one-paragraph justification. Report numbers exactly as your code "
        "printed them; do not round or estimate them by hand."
    )


def code_tool(cell: Cell) -> Dict[str, Any]:
    name = "run_r" if cell.language == "r" else "run_python"
    lang = "R" if cell.language == "r" else "Python"
    return {
        "name": name,
        "description": (
            f"Run {lang} code in the persistent session. Returns printed "
            "output and errors. Long output is truncated."
        ),
        "input_schema": {
            "type": "object",
            "properties": {"code": {"type": "string", "description": f"{lang} code"}},
            "required": ["code"],
        },
    }


def submit_tool(value_names: List[str]) -> Dict[str, Any]:
    """``submit_answer`` with the prompt's requested values as named fields."""
    props: Dict[str, Any] = {
        name: {"type": ["number", "null"], "description": f"The value of {name}."}
        for name in value_names
    }
    props["estimator"] = {
        "type": "string",
        "enum": list(ESTIMATOR_KEYS),
        "description": "The estimator the reported values come from.",
    }
    props["justification"] = {"type": "string"}
    return {
        "name": "submit_answer",
        "description": (
            "Submit the final answer. Call once, at the end. Use null for a "
            "value you could not compute."
        ),
        "input_schema": {
            "type": "object",
            "properties": props,
            "required": [*value_names, "estimator", "justification"],
        },
    }


def mcp_tools() -> List[Dict[str, Any]]:
    """StatsPAI's curated MCP tools, in the Anthropic tool format."""
    import json

    from statspai.agent import mcp_handle_request

    reply = json.loads(
        mcp_handle_request(
            json.dumps(
                {"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}}
            )
        )
    )
    tools = []
    for t in reply["result"]["tools"]:
        tools.append(
            {
                "name": t["name"],
                "description": t.get("description", ""),
                "input_schema": t.get("inputSchema", {"type": "object"}),
            }
        )
    return tools


def tools_for(cell: Cell, value_names: List[str]) -> List[Dict[str, Any]]:
    tools = [code_tool(cell), submit_tool(value_names)]
    if cell.toolset == "statspai":
        tools.extend(mcp_tools())
    return tools
