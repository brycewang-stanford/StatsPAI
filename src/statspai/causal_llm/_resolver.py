"""``get_llm_client()`` — layered resolution for LLM credentials + model.

Resolution order (first match wins):

1. **Explicit ``client=``** — already a built :class:`LLMClient` instance.
   Pass through unchanged.
2. **Explicit ``provider=`` + ``api_key=``** — construct directly.
3. **Environment variables** — auto-detect:
   - ``ANTHROPIC_API_KEY`` → Anthropic provider, default model
     :data:`DEFAULT_MODELS["anthropic"]`.
   - ``OPENAI_API_KEY`` → OpenAI provider, default model
     :data:`DEFAULT_MODELS["openai"]`.
   - When **both** are set, prefer the one named in the config file's
     ``[llm].provider``; tie-break to Anthropic.
4. **Config file** ``~/.config/statspai/llm.toml`` — provides
   ``provider`` and ``model`` defaults, but the API key still has to
   come from the environment (security: never store secrets in a
   plaintext file the user might commit).
5. **Interactive prompt** — only when ``sys.stdin.isatty()`` AND
   ``allow_interactive=True``. Prompts the user to:
   - Pick a provider from those with env-var-set keys.
   - Pick a model (default = recommended for that provider).
   Never asks for the API key over stdin (security + friction).
6. **Hard error** with a clear remediation message pointing the user
   to ``export ANTHROPIC_API_KEY=...`` or
   ``sp.causal_llm.configure_llm(...)``.

Why not store the API key in the config file?
---------------------------------------------
Industry-standard pattern (Anthropic SDK, OpenAI SDK, Hugging Face
``huggingface_hub``, AWS CLI, gcloud, kubectl): credentials live in
environment variables or platform keyring (``keyring`` Python
package), never in plaintext config files. Plaintext files get:

- Committed to dotfiles repos by accident (search GitHub for leaked
  ``.aws/credentials`` files — there are tens of thousands).
- Synced to cloud backups without obvious encryption-at-rest.
- World-readable when users forget ``chmod 600``.

Users who want OS-keyring storage can install ``keyring`` separately
and inject via ``api_key=keyring.get_password("statspai", "anthropic")``;
StatsPAI doesn't take ``keyring`` as a hard dep.
"""
from __future__ import annotations

import os
import sys
from typing import Any, Dict, Optional

from ._config import DEFAULT_MODELS, load_config


__all__ = [
    "get_llm_client",
    "list_available_providers",
    "configure_llm",
    "LLMConfigurationError",
]


_ENV_KEY_BY_PROVIDER = {
    "anthropic": "ANTHROPIC_API_KEY",
    "openai": "OPENAI_API_KEY",
}


class LLMConfigurationError(RuntimeError):
    """Raised when no LLM provider can be resolved.

    Message points the user at concrete remediation steps rather than
    the generic "no key found" — agents and CLI users alike should
    know exactly what to type next.
    """


def list_available_providers() -> Dict[str, Dict[str, Any]]:
    """Inspect the environment and return what's currently usable.

    Returns
    -------
    dict
        ``{provider_name: {"available": bool, "default_model": str,
        "env_var": str}}`` for each known provider. Useful both for
        the interactive prompt and for tools that want to surface
        available LLMs to the user.
    """
    out: Dict[str, Dict[str, Any]] = {}
    for provider, env_var in _ENV_KEY_BY_PROVIDER.items():
        out[provider] = {
            "available": bool(os.environ.get(env_var)),
            "default_model": DEFAULT_MODELS.get(provider, ""),
            "env_var": env_var,
        }
    return out


def _construct_client(
    provider: str,
    *,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    **kwargs: Any,
):
    """Build a concrete LLMClient for the named provider."""
    from .llm_clients import openai_client, anthropic_client

    eff_model = model or DEFAULT_MODELS.get(provider)
    if provider == "anthropic":
        return anthropic_client(
            model=eff_model or "claude-sonnet-4-5",
            api_key=api_key,
            **{k: v for k, v in kwargs.items()
               if k in {"temperature", "max_tokens", "max_retries",
                        "thinking_budget", "system_prompt"}},
        )
    if provider == "openai":
        return openai_client(
            model=eff_model or "gpt-4o-mini",
            api_key=api_key,
            **{k: v for k, v in kwargs.items()
               if k in {"temperature", "max_tokens", "max_retries",
                        "base_url", "organization", "system_prompt"}},
        )
    raise LLMConfigurationError(
        f"Unknown provider {provider!r}. "
        f"Supported: {sorted(_ENV_KEY_BY_PROVIDER)}."
    )


def _interactive_prompt(
    available: Dict[str, Dict[str, Any]],
    *,
    config_provider: Optional[str] = None,
    config_model: Optional[str] = None,
) -> Dict[str, str]:
    """Walk the user through picking provider + model.

    Only called when stdin is a TTY and prior layers all failed.
    Never asks for the API key — that path is closed to interactive
    input by design (security).
    """
    print("\nStatsPAI: pick an LLM provider for this session.", flush=True)
    print("Available providers (env-var key set):", flush=True)
    options = []
    for i, (name, info) in enumerate(available.items(), 1):
        marker = "✓" if info["available"] else "✗"
        suffix = "" if info["available"] else f" (set {info['env_var']} first)"
        default = " [default]" if name == config_provider else ""
        print(
            f"  {i}. {marker} {name}{default} — model={info['default_model']}"
            f"{suffix}",
            flush=True,
        )
        options.append(name)

    if not any(info["available"] for info in available.values()):
        raise LLMConfigurationError(
            "No LLM provider is configured. Set one of:\n"
            "  export ANTHROPIC_API_KEY=...\n"
            "  export OPENAI_API_KEY=...\n"
            "Then re-run, or pass an explicit `client=` instance.\n"
            "(Tip: run `sp.causal_llm.configure_llm(provider=..., model=...)` "
            "to save your provider+model preference to "
            "~/.config/statspai/llm.toml — but the API key always lives in "
            "the environment, never in the config file.)"
        )

    while True:
        try:
            raw = input("Choice [1]: ").strip() or "1"
        except (EOFError, KeyboardInterrupt):
            raise LLMConfigurationError(
                "Interactive provider selection cancelled."
            )
        try:
            idx = int(raw) - 1
            if not (0 <= idx < len(options)):
                raise ValueError
        except ValueError:
            print("  Invalid choice; enter a number.", flush=True)
            continue
        provider = options[idx]
        if not available[provider]["available"]:
            print(
                f"  {provider}'s {available[provider]['env_var']} env var "
                "is not set. Set it first, then retry.",
                flush=True,
            )
            continue
        break

    default_model = (
        config_model
        if config_provider == provider and config_model
        else available[provider]["default_model"]
    )
    try:
        model = input(f"Model [{default_model}]: ").strip() or default_model
    except (EOFError, KeyboardInterrupt):
        model = default_model

    return {"provider": provider, "model": model}


def get_llm_client(
    *,
    client: Any = None,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    allow_interactive: bool = True,
    config_path=None,
    **kwargs: Any,
):
    """Resolve an :class:`LLMClient` via layered fallback.

    See module docstring for the full resolution order. Most users
    don't call this directly — it's plumbed into
    ``sp.paper(..., llm='auto')`` and the LLM-DAG closed-loop entry
    points so the typical workflow is "set env var, forget".

    Parameters
    ----------
    client : LLMClient, optional
        Already-built client. Returned as-is. (Layer 1.)
    provider : {'anthropic', 'openai'}, optional
        Force a specific provider. (Layer 2.)
    model : str, optional
        Force a specific model. Defaults to ``DEFAULT_MODELS[provider]``.
    api_key : str, optional
        Pass-through to the constructed client. When omitted, the
        provider SDK reads from its standard env var
        (``ANTHROPIC_API_KEY`` / ``OPENAI_API_KEY``).
    allow_interactive : bool, default True
        Whether to fall back to a stdin prompt when prior layers
        fail. Set to ``False`` in agent / Jupyter contexts where
        ``input()`` would hang the kernel.
    config_path : Path, optional
        Override the config file location (mainly for testing).
    **kwargs
        Forwarded to the provider's client constructor (e.g.
        ``temperature``, ``max_tokens``, ``thinking_budget`` for
        Anthropic, ``base_url`` for OpenAI).

    Raises
    ------
    LLMConfigurationError
        When no path resolves and either ``allow_interactive=False``
        or stdin is not a TTY (so interactive can't run).
    """
    # Layer 1 — explicit client
    if client is not None:
        return client

    # Layer 2 — explicit provider + (optionally) api_key.
    if provider is not None:
        return _construct_client(
            provider, model=model, api_key=api_key, **kwargs,
        )

    # Layer 3 — env vars (auto-detect, prefer config).
    cfg = load_config(config_path)
    cfg_llm = cfg.get("llm", {}) if isinstance(cfg, dict) else {}
    cfg_provider = cfg_llm.get("provider")
    cfg_model = cfg_llm.get("model")

    available = list_available_providers()
    set_providers = [
        name for name, info in available.items() if info["available"]
    ]

    if set_providers:
        # If config names a provider AND its env var is set, use it.
        if cfg_provider in set_providers:
            chosen = cfg_provider
        elif "anthropic" in set_providers:
            chosen = "anthropic"  # tie-break
        else:
            chosen = set_providers[0]
        eff_model = (
            model
            or (cfg_model if chosen == cfg_provider else None)
            or DEFAULT_MODELS[chosen]
        )
        return _construct_client(
            chosen, model=eff_model, api_key=api_key, **kwargs,
        )

    # Layer 5 — interactive prompt (no env vars set).
    if allow_interactive and sys.stdin.isatty():
        choice = _interactive_prompt(
            available,
            config_provider=cfg_provider,
            config_model=cfg_model,
        )
        return _construct_client(
            choice["provider"], model=choice["model"],
            api_key=api_key, **kwargs,
        )

    # Layer 6 — hard error with remediation.
    raise LLMConfigurationError(
        "No LLM provider configured. Set one of these env vars:\n"
        "  export ANTHROPIC_API_KEY=...     # for Claude\n"
        "  export OPENAI_API_KEY=...        # for GPT-4 / o-series\n"
        "Then call again. Or pass an explicit `client=` instance / "
        "use `sp.causal_llm.configure_llm(provider=..., model=...)` "
        "to save your provider+model preference. "
        "API keys always come from the environment, never from the "
        "config file."
    )


def configure_llm(
    *,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    config_path=None,
):
    """Persist a provider+model preference to
    ``~/.config/statspai/llm.toml``.

    Use this for a "set once, forget" workflow when working on a
    machine with both ``ANTHROPIC_API_KEY`` and ``OPENAI_API_KEY``
    set — without it, the resolver tie-breaks to Anthropic.

    Examples
    --------
    >>> import statspai as sp                 # doctest: +SKIP
    >>> sp.causal_llm.configure_llm(          # doctest: +SKIP
    ...     provider="openai", model="gpt-4o",
    ... )
    """
    from ._config import set_preferences
    if provider is not None and provider not in _ENV_KEY_BY_PROVIDER:
        raise ValueError(
            f"Unknown provider {provider!r}. "
            f"Supported: {sorted(_ENV_KEY_BY_PROVIDER)}."
        )
    return set_preferences(
        provider=provider, model=model, path=config_path,
    )
