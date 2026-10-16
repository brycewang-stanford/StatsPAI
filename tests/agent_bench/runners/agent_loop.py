"""One CausalAgentBench trial: model <-> tools until ``submit_answer``.

The loop is the same for every cell; only the tool set and the session's
toolset differ (``cells.py``). A trial ends when the model calls
``submit_answer``, stops calling tools, refuses, or reaches ``max_turns``.
Everything the grader needs is in the returned record: the submitted
answer, every tool call with its (truncated) output, token usage, and the
full neutral-format transcript.
"""

from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from cells import Cell, system_prompt, tools_for
from models import Model, ToolCall, Turn
from sandbox import PythonSession, RSession, make_workdir

HERE = Path(__file__).resolve().parent
DATA = HERE.parent / "data"
OUTPUT_LIMIT = 8000


def _user_message(task: Dict[str, Any]) -> str:
    files = ", ".join(task["files"]) if task["files"] else "none"
    return (
        f"{task['question']}\n\nData file(s) in your working directory: {files}.\n"
        f"{task['instruction']}"
    )


def _mcp_call(name: str, arguments: Dict[str, Any], workdir: Path) -> Dict[str, Any]:
    """Run one StatsPAI MCP tool in-process; relative data paths resolve in workdir."""
    from statspai.agent import mcp_handle_request

    args = dict(arguments)
    path = args.get("data_path")
    if isinstance(path, str) and not Path(path).is_absolute():
        args["data_path"] = str(workdir / path)
    reply = json.loads(
        mcp_handle_request(
            json.dumps(
                {
                    "jsonrpc": "2.0",
                    "id": 1,
                    "method": "tools/call",
                    "params": {"name": name, "arguments": args},
                }
            )
        )
    )
    if "error" in reply:
        return {"ok": False, "output": json.dumps(reply["error"])}
    result = reply.get("result", {})
    text = "\n".join(
        c.get("text", "") for c in result.get("content", []) if c.get("type") == "text"
    )
    return {"ok": not result.get("isError", False), "output": text}


def run_trial(
    task: Dict[str, Any],
    cell: Cell,
    model: Model,
    *,
    rep: int = 0,
    max_turns: int = 20,
    call_timeout: float = 300,
    workroot: Optional[Path] = None,
    keep_workdir: bool = False,
) -> Dict[str, Any]:
    started = time.time()
    workdir = make_workdir({f: DATA / f for f in task["files"]}, workroot)
    session = (
        RSession(workdir, timeout=call_timeout, max_chars=OUTPUT_LIMIT)
        if cell.language == "r"
        else PythonSession(
            workdir,
            allow_statspai=cell.toolset == "statspai",
            timeout=call_timeout,
            max_chars=OUTPUT_LIMIT,
        )
    )
    code_tool = "run_r" if cell.language == "r" else "run_python"
    tools = tools_for(cell, list(task["report"]))
    tool_names = {t["name"] for t in tools}
    system = system_prompt(cell)
    messages: List[Dict[str, Any]] = [{"role": "user", "text": _user_message(task)}]
    calls: List[Dict[str, Any]] = []
    submitted: Optional[Dict[str, Any]] = None
    usage = {
        "input_tokens": 0,
        "output_tokens": 0,
        "cache_read_tokens": 0,
        "cache_write_tokens": 0,
    }
    stop = "max_turns"
    turns = 0
    try:
        while turns < max_turns:
            turns += 1
            turn: Turn = model.respond(system, messages, tools)
            for key in usage:
                usage[key] += getattr(turn, key)
            messages.append(
                {
                    "role": "assistant",
                    "text": turn.text,
                    "tool_calls": turn.tool_calls,
                    "raw": turn.raw,
                }
            )
            if turn.stop_reason.startswith("refusal"):
                stop = turn.stop_reason
                break
            if not turn.tool_calls:
                stop = "no_tool_call"
                break
            for call in turn.tool_calls:
                record = _execute(call, session, workdir, code_tool, tool_names)
                calls.append(record)
                if call.name == "submit_answer" and submitted is None:
                    submitted = dict(call.arguments)
                messages.append(
                    {
                        "role": "tool",
                        "tool_call_id": call.id,
                        "name": call.name,
                        "content": record["output"],
                    }
                )
            if submitted is not None:
                stop = "submitted"
                break
    finally:
        session.close()
        if not keep_workdir:
            shutil.rmtree(workdir, ignore_errors=True)
    return {
        "task_id": task["id"],
        "level": task["level"],
        "cell": cell.id,
        "rep": rep,
        "model": model.name,
        "submitted": submitted,
        "stop": stop,
        "turns": turns,
        "usage": usage,
        "tool_calls": calls,
        "wall_seconds": round(time.time() - started, 3),
        "transcript": _serialisable(messages),
    }


def _execute(
    call: ToolCall,
    session: Any,
    workdir: Path,
    code_tool: str,
    tool_names: set,
) -> Dict[str, Any]:
    args_text = json.dumps(call.arguments, default=str)
    if call.name not in tool_names:
        return {
            "name": call.name,
            "arguments": args_text[:2000],
            "ok": False,
            "output": f"Unknown tool {call.name!r}.",
            "unknown_tool": True,
        }
    if call.name == "submit_answer":
        return {
            "name": call.name,
            "arguments": args_text,
            "ok": True,
            "output": "Answer recorded.",
        }
    if call.name == code_tool:
        result = session.run(str(call.arguments.get("code", "")))
        return {
            "name": call.name,
            "arguments": args_text[:4000],
            "ok": result.ok,
            "output": result.output,
            "timed_out": result.timed_out,
        }
    try:
        out = _mcp_call(call.name, call.arguments, workdir)
    except Exception as exc:  # noqa: BLE001 - reported to the model as a tool error
        out = {"ok": False, "output": f"{type(exc).__name__}: {exc}"}
    text = out["output"]
    if len(text) > OUTPUT_LIMIT:
        text = (
            text[:OUTPUT_LIMIT]
            + f"\n... [{len(out['output']) - OUTPUT_LIMIT} characters truncated]"
        )
    return {
        "name": call.name,
        "arguments": args_text[:4000],
        "ok": out["ok"],
        "output": text,
    }


def _serialisable(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out = []
    for m in messages:
        m = dict(m)
        if "tool_calls" in m:
            m["tool_calls"] = [
                {"id": c.id, "name": c.name, "arguments": c.arguments}
                for c in m["tool_calls"]
            ]
        out.append(m)
    return out
