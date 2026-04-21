"""
LLM client adapters for the :mod:`statspai.causal_llm` subpackage.

Exposes a uniform :class:`LLMClient` protocol plus three concrete
adapters so that :func:`sp.causal_llm.causal_mas` (and any future
agent-driven causal-inference workflow) can swap in a real model with
one keyword argument.

::

    import statspai as sp
    client = sp.causal_llm.openai_client(
        model="gpt-4o-mini", api_key="sk-...",
    )
    res = sp.causal_llm.causal_mas(variables=..., client=client)

Design constraints
------------------
* **No core-package dependency** on ``openai`` / ``anthropic``.  Both
  adapters lazily import their SDK inside ``__init__``, raising a clear
  error message if the optional extra is missing.
* **Deterministic-by-default** — temperatures default to ``0`` so unit
  tests can pin outputs.
* **Graceful retry** on transient network / rate-limit errors using
  exponential backoff (up to ``max_retries`` attempts).
* **Transcript-friendly** — every exchange is stored on the adapter
  under ``.history`` for replay / logging.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


__all__ = [
    "LLMClient",
    "openai_client",
    "anthropic_client",
    "echo_client",
]


# ---------------------------------------------------------------------------
# Minimal protocol
# ---------------------------------------------------------------------------


class LLMClient:
    """Minimal interface expected by :func:`causal_mas` and friends.

    Subclasses / adapters must implement ``chat(role, prompt)`` — a
    single-turn completion call that returns the model's plain-text
    response.  Everything else (streaming, tools, JSON mode) is
    deliberately out of scope because :func:`causal_mas` only needs a
    bag of edge proposals or critiques.
    """

    name: str = "abstract"
    history: List[Dict[str, Any]] = []

    def chat(self, role: str, prompt: str) -> str:  # pragma: no cover
        raise NotImplementedError

    # So the client is ``callable(prompt)`` for the generic case.
    def __call__(self, prompt: str) -> str:
        return self.chat("user", prompt)


# ---------------------------------------------------------------------------
# OpenAI-style adapter (v1.x Python SDK)
# ---------------------------------------------------------------------------


@dataclass
class _OpenAIClient(LLMClient):
    """Thin adapter on top of the ``openai>=1.0`` Python SDK."""

    model: str = "gpt-4o-mini"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    organization: Optional[str] = None
    temperature: float = 0.0
    max_tokens: int = 1024
    max_retries: int = 3
    system_prompt: str = (
        "You are a careful scientific assistant helping with causal "
        "inference.  Respond concisely and in the exact format requested."
    )
    name: str = "openai"
    history: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as err:  # pragma: no cover
            raise ImportError(
                "openai_client requires `pip install openai>=1.0`."
            ) from err
        key = self.api_key or os.environ.get("OPENAI_API_KEY")
        if not key:
            raise ValueError(
                "OpenAI API key missing — pass api_key=... or set "
                "OPENAI_API_KEY in the environment."
            )
        kwargs: Dict[str, Any] = {"api_key": key}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        if self.organization:
            kwargs["organization"] = self.organization
        self._client = OpenAI(**kwargs)

    def chat(self, role: str, prompt: str) -> str:
        """Run one chat-completion call and return the assistant string.

        ``role`` is used only to label the transcript (the agent's
        function — ``proposer``, ``critic``, ...).  OpenAI's own role
        field is set to ``'user'`` for the prompt; the system prompt is
        always our ``self.system_prompt``.
        """
        messages = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": f"[{role}]\n{prompt}"},
        ]
        err = None
        for attempt in range(self.max_retries):
            try:
                resp = self._client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                )
                text = resp.choices[0].message.content or ""
                self.history.append(
                    {"role": role, "prompt": prompt, "response": text,
                     "model": self.model}
                )
                return text
            except Exception as e:  # pragma: no cover - network dependent
                err = e
                time.sleep(min(2 ** attempt, 8))
        raise RuntimeError(
            f"OpenAI chat failed after {self.max_retries} attempts: {err}"
        )


def openai_client(
    model: str = "gpt-4o-mini",
    *,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    organization: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    max_retries: int = 3,
    system_prompt: Optional[str] = None,
) -> LLMClient:
    """Construct an OpenAI-compatible :class:`LLMClient`.

    Requires the optional ``openai>=1.0`` extra.  Accepts any
    ``base_url`` override so you can point this at an OpenAI-compatible
    endpoint (Azure OpenAI, vLLM, Ollama's OpenAI-compat mode, ...).

    Examples
    --------
    >>> import statspai as sp
    >>> client = sp.causal_llm.openai_client(
    ...     model="gpt-4o-mini", api_key="sk-...",
    ... )   # doctest: +SKIP
    >>> res = sp.causal_llm.causal_mas(
    ...     variables=["age","treatment","outcome"], client=client,
    ... )    # doctest: +SKIP
    """
    kwargs: Dict[str, Any] = dict(
        model=model,
        api_key=api_key,
        base_url=base_url,
        organization=organization,
        temperature=temperature,
        max_tokens=max_tokens,
        max_retries=max_retries,
    )
    if system_prompt is not None:
        kwargs["system_prompt"] = system_prompt
    return _OpenAIClient(**kwargs)


# ---------------------------------------------------------------------------
# Anthropic-style adapter (anthropic>=0.30 Python SDK)
# ---------------------------------------------------------------------------


@dataclass
class _AnthropicClient(LLMClient):
    """Adapter on top of the ``anthropic>=0.30`` Messages API."""

    model: str = "claude-opus-4-7"
    api_key: Optional[str] = None
    base_url: Optional[str] = None
    temperature: float = 0.0
    max_tokens: int = 1024
    max_retries: int = 3
    system_prompt: str = (
        "You are a careful scientific assistant helping with causal "
        "inference.  Respond concisely and in the exact format requested."
    )
    name: str = "anthropic"
    history: List[Dict[str, Any]] = field(default_factory=list)

    def __post_init__(self):
        try:
            import anthropic  # type: ignore
        except ImportError as err:  # pragma: no cover
            raise ImportError(
                "anthropic_client requires `pip install anthropic>=0.30`."
            ) from err
        key = self.api_key or os.environ.get("ANTHROPIC_API_KEY")
        if not key:
            raise ValueError(
                "Anthropic API key missing — pass api_key=... or set "
                "ANTHROPIC_API_KEY in the environment."
            )
        kwargs: Dict[str, Any] = {"api_key": key}
        if self.base_url:
            kwargs["base_url"] = self.base_url
        self._client = anthropic.Anthropic(**kwargs)

    def chat(self, role: str, prompt: str) -> str:
        err = None
        for attempt in range(self.max_retries):
            try:
                resp = self._client.messages.create(
                    model=self.model,
                    system=self.system_prompt,
                    messages=[{
                        "role": "user",
                        "content": f"[{role}]\n{prompt}",
                    }],
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                )
                # Anthropic returns a list of content blocks; join text parts.
                text = "".join(
                    getattr(block, "text", "") for block in resp.content
                )
                self.history.append(
                    {"role": role, "prompt": prompt, "response": text,
                     "model": self.model}
                )
                return text
            except Exception as e:  # pragma: no cover - network dependent
                err = e
                time.sleep(min(2 ** attempt, 8))
        raise RuntimeError(
            f"Anthropic chat failed after {self.max_retries} attempts: {err}"
        )


def anthropic_client(
    model: str = "claude-opus-4-7",
    *,
    api_key: Optional[str] = None,
    base_url: Optional[str] = None,
    temperature: float = 0.0,
    max_tokens: int = 1024,
    max_retries: int = 3,
    system_prompt: Optional[str] = None,
) -> LLMClient:
    """Construct an Anthropic-compatible :class:`LLMClient`.

    Requires the optional ``anthropic>=0.30`` extra.  Defaults to
    Claude Opus 4.7 (the latest generally-available model as of
    StatsPAI v1.3).

    Examples
    --------
    >>> import statspai as sp
    >>> client = sp.causal_llm.anthropic_client(
    ...     model="claude-sonnet-4-6", api_key="sk-ant-...",
    ... )   # doctest: +SKIP
    """
    kwargs: Dict[str, Any] = dict(
        model=model,
        api_key=api_key,
        base_url=base_url,
        temperature=temperature,
        max_tokens=max_tokens,
        max_retries=max_retries,
    )
    if system_prompt is not None:
        kwargs["system_prompt"] = system_prompt
    return _AnthropicClient(**kwargs)


# ---------------------------------------------------------------------------
# Echo client — for unit testing without any SDK
# ---------------------------------------------------------------------------


@dataclass
class _EchoClient(LLMClient):
    """Deterministic dummy client that returns a scripted response.

    The ``response_fn`` maps (role, prompt) → str.  Used by the test
    suite to verify that :func:`causal_mas` correctly parses structured
    LLM output without hitting the network.
    """

    response_fn: Callable[[str, str], str]
    name: str = "echo"
    history: List[Dict[str, Any]] = field(default_factory=list)

    def chat(self, role: str, prompt: str) -> str:
        text = self.response_fn(role, prompt)
        self.history.append(
            {"role": role, "prompt": prompt, "response": text,
             "model": "echo"}
        )
        return text


def echo_client(response_fn: Callable[[str, str], str]) -> LLMClient:
    """Deterministic scripted-response client for testing.

    >>> import statspai as sp
    >>> def scripted(role, prompt):
    ...     if role == 'proposer':
    ...         return 'age -> treatment\\ntreatment -> outcome'
    ...     return ''
    >>> client = sp.causal_llm.echo_client(scripted)
    >>> res = sp.causal_llm.causal_mas(
    ...     variables=['age','treatment','outcome'], client=client,
    ... )
    >>> ('treatment', 'outcome') in res.edges
    True
    """
    return _EchoClient(response_fn=response_fn)
