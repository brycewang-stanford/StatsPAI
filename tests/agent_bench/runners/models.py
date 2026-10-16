"""Language-model back ends for CausalAgentBench.

The agent loop speaks one neutral message format; each back end translates
it to its provider's API:

* user turns: ``{"role": "user", "text": ...}``
* model turns: ``{"role": "assistant", "text": ..., "tool_calls": [ToolCall]}``
* tool results: ``{"role": "tool", "tool_call_id": ..., "name": ..., "content": ...}``

``ScriptedModel`` replays a fixed list of turns; the harness tests use it to
drive every cell end to end without a language model. ``AnthropicModel`` and
``OpenAIModel`` call the real APIs. They can only be built through
:func:`real_model`, which refuses unless the pre-registration lock verifies
(``prereg.verify``): no production trial can run before the deposit.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: Dict[str, Any]


@dataclass
class Turn:
    text: str = ""
    tool_calls: List[ToolCall] = field(default_factory=list)
    input_tokens: int = 0
    output_tokens: int = 0
    #: Prompt-cache reads and writes, counted separately from input_tokens.
    cache_read_tokens: int = 0
    cache_write_tokens: int = 0
    stop_reason: str = ""
    #: Provider content blocks as returned, replayed verbatim on the next
    #: request (thinking blocks must be passed back unchanged).
    raw: Optional[List[Dict[str, Any]]] = None


class Model:
    name = "model"

    def respond(
        self, system: str, messages: List[Dict[str, Any]], tools: List[Dict[str, Any]]
    ) -> Turn:  # pragma: no cover - interface
        raise NotImplementedError


def _approx_tokens(obj: Any) -> int:
    """Four characters per token: only for the scripted model's bookkeeping."""
    return max(1, len(json.dumps(obj, default=str)) // 4)


class ScriptedModel(Model):
    """Replays ``turns`` in order; after the last one it stops calling tools."""

    name = "scripted"

    def __init__(self, turns: Sequence[Turn]) -> None:
        self.turns = list(turns)
        self.calls = 0

    def respond(self, system, messages, tools) -> Turn:
        self.calls += 1
        if not self.turns:
            return Turn(text="(script exhausted)", stop_reason="end_turn")
        turn = self.turns.pop(0)
        turn.input_tokens = turn.input_tokens or _approx_tokens(
            [system, messages, tools]
        )
        turn.output_tokens = turn.output_tokens or _approx_tokens(
            [turn.text, [c.arguments for c in turn.tool_calls]]
        )
        return turn


# ---------------------------------------------------------------------------
# Real back ends
# ---------------------------------------------------------------------------


class AnthropicModel(Model):  # pragma: no cover - needs the API
    """Claude through the Messages API, manual tool loop.

    * ``effort`` is pinned by the pre-registration lock (the default differs
      between models, so it is never left implicit).
    * Assistant turns are replayed with their content blocks unchanged,
      thinking blocks included: dropping or editing them is a history edit
      the API may reject, and it would change what the model saw.
    * No refusal ``fallbacks``: a fallback would let another model finish
      the trial inside a cell that pre-registered one model. A refusal ends
      the trial and is scored as a failure, with its category recorded.
    """

    def __init__(self, model_id: str, effort: str, max_tokens: int = 16000) -> None:
        import anthropic

        self.client = anthropic.Anthropic()
        self.model_id = model_id
        self.name = model_id
        self.effort = effort
        self.max_tokens = max_tokens

    @staticmethod
    def _messages(messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = []
        for m in messages:
            if m["role"] == "user":
                out.append({"role": "user", "content": m["text"]})
            elif m["role"] == "assistant" and m.get("raw") is not None:
                out.append({"role": "assistant", "content": m["raw"]})
            elif m["role"] == "assistant":
                blocks: List[Dict[str, Any]] = []
                if m.get("text"):
                    blocks.append({"type": "text", "text": m["text"]})
                for c in m.get("tool_calls", []):
                    blocks.append(
                        {
                            "type": "tool_use",
                            "id": c.id,
                            "name": c.name,
                            "input": c.arguments,
                        }
                    )
                out.append({"role": "assistant", "content": blocks})
            else:
                block = {
                    "type": "tool_result",
                    "tool_use_id": m["tool_call_id"],
                    "content": m["content"],
                }
                if (
                    out
                    and out[-1]["role"] == "user"
                    and isinstance(out[-1]["content"], list)
                ):
                    out[-1]["content"].append(block)
                else:
                    out.append({"role": "user", "content": [block]})
        return out

    def respond(self, system, messages, tools) -> Turn:
        resp = self.client.messages.create(
            model=self.model_id,
            max_tokens=self.max_tokens,
            system=system,
            tools=tools,
            output_config={"effort": self.effort},
            # Tool definitions and history are re-sent every turn; caching
            # them is what any production agent does. Usage keeps cache
            # reads and writes apart so both token measures can be reported.
            cache_control={"type": "ephemeral"},
            messages=self._messages(messages),
        )
        stop = resp.stop_reason or ""
        if stop == "refusal" and getattr(resp, "stop_details", None):
            stop = f"refusal:{resp.stop_details.category}"
        text = "".join(b.text for b in resp.content if b.type == "text")
        calls = [
            ToolCall(b.id, b.name, dict(b.input))
            for b in resp.content
            if b.type == "tool_use"
        ]
        return Turn(
            text=text,
            tool_calls=calls,
            input_tokens=resp.usage.input_tokens,
            output_tokens=resp.usage.output_tokens,
            cache_read_tokens=resp.usage.cache_read_input_tokens or 0,
            cache_write_tokens=resp.usage.cache_creation_input_tokens or 0,
            stop_reason=stop,
            raw=[b.to_dict() for b in resp.content],
        )


class OpenAIModel(Model):  # pragma: no cover - needs the API
    def __init__(self, model_id: str, max_tokens: int = 4096) -> None:
        import openai

        self.client = openai.OpenAI()
        self.model_id = model_id
        self.name = model_id
        self.max_tokens = max_tokens

    @staticmethod
    def _messages(system: str, messages: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        out: List[Dict[str, Any]] = [{"role": "system", "content": system}]
        for m in messages:
            if m["role"] == "user":
                out.append({"role": "user", "content": m["text"]})
            elif m["role"] == "assistant":
                msg: Dict[str, Any] = {
                    "role": "assistant",
                    "content": m.get("text") or None,
                }
                if m.get("tool_calls"):
                    msg["tool_calls"] = [
                        {
                            "id": c.id,
                            "type": "function",
                            "function": {
                                "name": c.name,
                                "arguments": json.dumps(c.arguments),
                            },
                        }
                        for c in m["tool_calls"]
                    ]
                out.append(msg)
            else:
                out.append(
                    {
                        "role": "tool",
                        "tool_call_id": m["tool_call_id"],
                        "content": m["content"],
                    }
                )
        return out

    def respond(self, system, messages, tools) -> Turn:
        fn_tools = [
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description", ""),
                    "parameters": t["input_schema"],
                },
            }
            for t in tools
        ]
        resp = self.client.chat.completions.create(
            model=self.model_id,
            max_completion_tokens=self.max_tokens,
            tools=fn_tools,
            messages=self._messages(system, messages),
        )
        choice = resp.choices[0]
        calls = [
            ToolCall(tc.id, tc.function.name, json.loads(tc.function.arguments or "{}"))
            for tc in (choice.message.tool_calls or [])
        ]
        details = getattr(resp.usage, "prompt_tokens_details", None)
        cached = int(getattr(details, "cached_tokens", 0) or 0)
        return Turn(
            text=choice.message.content or "",
            tool_calls=calls,
            # OpenAI reports cached tokens inside prompt_tokens.
            input_tokens=resp.usage.prompt_tokens - cached,
            cache_read_tokens=cached,
            output_tokens=resp.usage.completion_tokens,
            stop_reason=choice.finish_reason or "",
        )


def real_model(family: str, lock: Optional[Dict[str, Any]] = None) -> Model:
    """A real back end, only after the pre-registration lock verifies.

    The model ID comes from the lock, never from a default: the deposit pins
    the exact release before trial 1.
    """
    from prereg import verify

    lock = lock if lock is not None else verify()
    model_id = lock["models"].get(family)
    if not model_id:
        raise RuntimeError(f"the pre-registration lock pins no {family} model")
    key = {"anthropic": "ANTHROPIC_API_KEY", "openai": "OPENAI_API_KEY"}[family]
    if not os.environ.get(key):
        raise RuntimeError(f"{key} is not set")
    if family == "anthropic":
        return AnthropicModel(model_id, effort=lock["settings"]["anthropic_effort"])
    return OpenAIModel(model_id)


class OracleModel(Model):
    """Runs a fixed piece of code, then submits the JSON it printed.

    It reacts to tool output the way a model does (the submission is read
    from the code's printed result, not known in advance), so it exercises
    the sandbox, the tool plumbing and the grader end to end. ``code_tool``
    is ``run_python`` or ``run_r``; the code must print a JSON object of the
    report fields on its last line.
    """

    name = "oracle"

    def __init__(
        self, code: str, estimator: str, code_tool: str = "run_python"
    ) -> None:
        self.code = code
        self.estimator = estimator
        self.code_tool = code_tool
        self.step = 0

    def respond(self, system, messages, tools) -> Turn:
        self.step += 1
        if self.step == 1:
            call = ToolCall("call_1", self.code_tool, {"code": self.code})
            return Turn(
                tool_calls=[call],
                input_tokens=_approx_tokens([system, messages, tools]),
                output_tokens=_approx_tokens(self.code),
                stop_reason="tool_use",
            )
        last = next(m for m in reversed(messages) if m["role"] == "tool")
        lines = [
            l for l in str(last["content"]).splitlines() if l.strip().startswith("{")
        ]
        values = json.loads(lines[-1]) if lines else {}
        args = {**values, "estimator": self.estimator, "justification": "oracle"}
        return Turn(
            tool_calls=[ToolCall("call_2", "submit_answer", args)],
            input_tokens=_approx_tokens([system, messages, tools]),
            output_tokens=_approx_tokens(args),
            stop_reason="tool_use",
        )
