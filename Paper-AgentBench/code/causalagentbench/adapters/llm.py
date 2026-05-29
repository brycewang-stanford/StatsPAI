"""LLMAdapter — drives a real Claude/GPT agent over one of the three stacks.

This is the cell that runs conditions C1..C6. It is intentionally a thin,
auditable harness:

1. materialise the task's data to a temp CSV;
2. build a stack-aware prompt (statspai / pythonic / r);
3. call the provider at ``temperature=0`` and record token usage (M4);
4. optionally execute the returned code in a *subprocess sandbox* with a
   wall-clock timeout, capturing exec success (M3) and a machine-readable
   final answer;
5. parse the final method / point estimate / diagnostics / refusal.

Requires the relevant SDK (``anthropic`` / ``openai``) and an API key. The
production tool-loop for C1/C2 plugs in ``statspai.agent.mcp_server``;
that MCP wiring is the natural extension point and is documented in the
README. Here we use a single-turn "write code that prints CAB_RESULT"
protocol, which is provider-agnostic and fully reproducible.
"""

from __future__ import annotations

import json
import os
import re
import subprocess
import sys
import tempfile
import time
from typing import Dict, List, Optional, Tuple

from ..conditions import PYTHONIC_STACK, R_STACK
from ..schema import AgentAction, Task, Trajectory
from .base import AgentAdapter

_RESULT_SENTINEL = "CAB_RESULT="

_SYSTEM = (
    "You are a careful causal-inference analyst. You are given a research "
    "question and a CSV dataset. Decide on a valid identification strategy, "
    "estimate the effect, and run the standard diagnostics for that design. "
    "If the effect is NOT identifiable from the data, say so instead of "
    "inventing a number.\n\n"
    "Return EXACTLY ONE fenced ```python code block that, when run, prints a "
    "single line to stdout of the form:\n"
    f"  {_RESULT_SENTINEL}" + '{"method": "<estimator>", "point_estimate": <float or null>, '
    '"se": <float or null>, "diagnostics": ["<check>", ...], "refused": <true|false>}\n'
    "The CSV path is provided to your code as the variable DATA_PATH "
    "(already set); load it with pandas. Do not print anything else on that line."
)

_STACK_HINT = {
    "statspai": (
        "Use the StatsPAI package: `import statspai as sp`. Relevant entry "
        "points include sp.regress, sp.did, sp.callaway_santanna, sp.sun_abraham, "
        "sp.rdrobust, sp.ivreg, sp.detect_design, sp.recommend, sp.preflight. "
        "Prefer sp.recommend(...) to pick an estimator when unsure."
    ),
    "pythonic": (
        "Use only these packages (no StatsPAI): " + ", ".join(PYTHONIC_STACK) + "."
    ),
    "r": (
        "Write R, not Python: emit a single ```r block instead. Use these R "
        "packages: " + ", ".join(R_STACK) + ". Print the CAB_RESULT line via cat()."
    ),
}


class LLMAdapter(AgentAdapter):
    def __init__(self, condition, model: Optional[str] = None,
                 execute: bool = True, timeout_s: float = 120.0, **kw):
        super().__init__(condition, **kw)
        self.model = model
        self.execute = execute
        self.timeout_s = timeout_s

    # -- public ----------------------------------------------------------
    def run(self, task: Task, seed: int = 0) -> Trajectory:
        from .. import tasks as taskmod

        traj = Trajectory()
        t0 = time.perf_counter()
        try:
            df = taskmod.materialize(task, seed_offset=seed)
            with tempfile.TemporaryDirectory() as tmp:
                csv = os.path.join(tmp, "data.csv")
                df.to_csv(csv, index=False)
                prompt = self._build_prompt(task, df)
                text, tin, tout = self._call_provider(prompt, seed)
                traj.input_tokens, traj.output_tokens = tin, tout
                traj.raw_final = text
                traj.actions.append(AgentAction("message", "llm_reply", text))

                code, lang = _extract_code(text)
                traj.code = code
                traj.called_functions = _called_functions(code, lang)
                parsed = _parse_inline_json(text)

                if self.execute and code and lang in ("python", "r"):
                    ok, payload = _run_sandbox(code, lang, csv, self.timeout_s)
                    traj.executed_ok = ok
                    if payload:
                        parsed = payload  # executed result wins over inline
                _apply_parsed(traj, parsed)
        except Exception as exc:
            traj.error = f"{type(exc).__name__}: {exc}"
        traj.wall_clock_s = time.perf_counter() - t0
        return traj

    # -- prompt ----------------------------------------------------------
    def _build_prompt(self, task: Task, df) -> str:
        head = df.head(8).to_string(index=False)
        dtypes = ", ".join(f"{c}:{t}" for c, t in df.dtypes.astype(str).items())
        roles = ", ".join(f"{k}={v}" for k, v in task.roles.items())
        return (
            f"{_STACK_HINT[self.condition.stack]}\n\n"
            f"## Difficulty: {task.difficulty.value}\n"
            f"## Question\n{task.question}\n\n"
            f"## Column roles\n{roles}\n\n"
            f"## Data preview (DATA_PATH points to the full CSV)\n"
            f"columns: {dtypes}\n{head}\n"
        )

    # -- providers -------------------------------------------------------
    def _call_provider(self, prompt: str, seed: int) -> Tuple[str, int, int]:
        provider = self.condition.provider
        if provider == "claude":
            return _call_anthropic(prompt, self.model)
        if provider == "gpt":
            return _call_openai(prompt, self.model)
        raise ValueError(f"LLMAdapter cannot drive provider {provider!r}")


# ---------------------------------------------------------------------------
# Provider calls (lazy imports; clear errors if SDK/key missing)
# ---------------------------------------------------------------------------

def _call_anthropic(prompt: str, model: Optional[str]) -> Tuple[str, int, int]:
    try:
        import anthropic
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("pip install causalagentbench[anthropic]") from e
    client = anthropic.Anthropic()  # reads ANTHROPIC_API_KEY
    model = model or "claude-opus-4-8"
    msg = client.messages.create(
        model=model, max_tokens=4096, temperature=0,
        system=_SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    )
    text = "".join(b.text for b in msg.content if getattr(b, "type", "") == "text")
    return text, msg.usage.input_tokens, msg.usage.output_tokens


def _call_openai(prompt: str, model: Optional[str]) -> Tuple[str, int, int]:
    try:
        from openai import OpenAI
    except ImportError as e:  # pragma: no cover
        raise RuntimeError("pip install causalagentbench[openai]") from e
    client = OpenAI()  # reads OPENAI_API_KEY
    model = model or "gpt-5.3"
    resp = client.chat.completions.create(
        model=model, temperature=0,
        messages=[{"role": "system", "content": _SYSTEM},
                  {"role": "user", "content": prompt}],
    )
    text = resp.choices[0].message.content or ""
    u = resp.usage
    return text, getattr(u, "prompt_tokens", 0), getattr(u, "completion_tokens", 0)


# ---------------------------------------------------------------------------
# Parsing + sandbox execution
# ---------------------------------------------------------------------------

def _extract_code(text: str) -> Tuple[str, str]:
    m = re.search(r"```(python|r|R)\s*\n(.*?)```", text, re.DOTALL)
    if not m:
        return "", ""
    lang = m.group(1).lower()
    return m.group(2), ("r" if lang == "r" else "python")


def _called_functions(code: str, lang: str) -> List[str]:
    if lang != "python":
        return []
    out = []
    for m in re.finditer(r"\b(?:sp|statspai)\.(\w+)", code):
        out.append("statspai." + m.group(1))
    return sorted(set(out))


def _parse_inline_json(text: str) -> Optional[Dict]:
    idx = text.rfind(_RESULT_SENTINEL)
    if idx == -1:
        return None
    tail = text[idx + len(_RESULT_SENTINEL):].strip()
    return _loads_prefix(tail)


def _loads_prefix(s: str) -> Optional[Dict]:
    # parse the first balanced {...}
    depth = 0
    for i, ch in enumerate(s):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(s[: i + 1])
                except Exception:
                    return None
    return None


def _apply_parsed(traj: Trajectory, parsed: Optional[Dict]) -> None:
    if not parsed:
        return
    traj.final_method = parsed.get("method")
    pe = parsed.get("point_estimate")
    traj.final_estimate = float(pe) if isinstance(pe, (int, float)) else None
    traj.reported_diagnostics = list(parsed.get("diagnostics") or [])
    traj.refused = bool(parsed.get("refused"))


def _run_sandbox(code: str, lang: str, csv_path: str, timeout_s: float) -> Tuple[bool, Optional[Dict]]:
    """Execute agent code in a subprocess; return (exec_ok, parsed_result).

    NOTE: this runs model-written code. For production fan-out, run inside a
    container / VM. The subprocess + timeout here is a minimal guard, not a
    security boundary.
    """
    if lang == "python":
        prelude = f"DATA_PATH = {csv_path!r}\n"
        src = prelude + code
        with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as f:
            f.write(src)
            path = f.name
        cmd = [sys.executable, path]
    else:  # r
        prelude = f"DATA_PATH <- {csv_path!r}\n"
        with tempfile.NamedTemporaryFile("w", suffix=".R", delete=False) as f:
            f.write(prelude + code)
            path = f.name
        cmd = ["Rscript", path]
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s)
    except subprocess.TimeoutExpired:
        os.unlink(path)
        return False, None
    except FileNotFoundError:
        os.unlink(path)
        return False, None
    os.unlink(path)
    ok = proc.returncode == 0
    payload = None
    for line in proc.stdout.splitlines():
        if line.startswith(_RESULT_SENTINEL):
            payload = _loads_prefix(line[len(_RESULT_SENTINEL):].strip())
    return ok, payload
