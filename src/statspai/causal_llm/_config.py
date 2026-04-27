"""``~/.config/statspai/llm.toml`` — persistent LLM preferences.

Stores **provider** + **model** preferences only. **Never stores API
keys** — those live in environment variables (``ANTHROPIC_API_KEY`` /
``OPENAI_API_KEY``) so secrets aren't committed to a config file the
user might accidentally share or check into a dotfiles repo.

XDG-Base-Directory compliant:
- Linux / macOS: ``${XDG_CONFIG_HOME:-~/.config}/statspai/llm.toml``
- Windows: ``%APPDATA%/statspai/llm.toml`` (best-effort)

File format::

    [llm]
    provider = "anthropic"        # or "openai"
    model = "claude-sonnet-4-5"   # provider-specific default

The resolver in :mod:`statspai.causal_llm._resolver` reads this file
exactly once per call (no global cache — easier to test, and the
overhead is negligible compared to an LLM round-trip).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Any, Dict, Optional


__all__ = [
    "config_path",
    "load_config",
    "save_config",
    "set_preferences",
    "DEFAULT_MODELS",
]


#: Provider → recommended default model. Updated whenever Anthropic /
#: OpenAI ship a new flagship; users can override per-call.
DEFAULT_MODELS: Dict[str, str] = {
    "anthropic": "claude-sonnet-4-5",
    "openai": "gpt-4o-mini",
}


def config_path() -> Path:
    """Return the platform-appropriate config file path.

    Honours ``XDG_CONFIG_HOME`` on Linux/macOS; falls back to
    ``~/.config`` if unset. Windows uses ``%APPDATA%``.
    """
    if sys.platform == "win32":
        appdata = os.environ.get("APPDATA")
        base = Path(appdata) if appdata else Path.home() / "AppData" / "Roaming"
    else:
        xdg = os.environ.get("XDG_CONFIG_HOME")
        base = Path(xdg) if xdg else Path.home() / ".config"
    return base / "statspai" / "llm.toml"


def _parse_toml(text: str) -> Dict[str, Any]:
    """Minimal TOML parser sufficient for our flat ``[llm]`` schema.

    We deliberately don't import ``tomllib`` (3.11+ only) at module
    import time — Python 3.9/3.10 users still need this to work. The
    schema is so simple that a 30-line manual parser beats pulling in
    ``tomli`` as a dep.
    """
    out: Dict[str, Any] = {}
    section = None
    for raw_line in text.splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if line.startswith("[") and line.endswith("]"):
            section = line[1:-1].strip()
            out.setdefault(section, {})
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        # Strip inline comment.
        if "#" in value and not (value.startswith('"') or value.startswith("'")):
            value = value.split("#", 1)[0].strip()
        # Strip surrounding quotes.
        if (value.startswith('"') and value.endswith('"')) or \
           (value.startswith("'") and value.endswith("'")):
            value = value[1:-1]
        # Cast booleans / ints minimally — provider+model are strings,
        # but future fields might be bools.
        if value.lower() in ("true", "false"):
            value = value.lower() == "true"
        elif value.isdigit():
            value = int(value)
        if section is None:
            out[key] = value
        else:
            out[section][key] = value
    return out


def load_config(path: Optional[Path] = None) -> Dict[str, Any]:
    """Read the TOML preferences file, or return an empty dict.

    Never raises on missing / malformed file — returns ``{}`` so
    callers can fall through cleanly to the next resolution layer.
    """
    p = path or config_path()
    if not p.exists():
        return {}
    try:
        text = p.read_text(encoding="utf-8")
    except OSError:
        return {}
    try:
        return _parse_toml(text)
    except Exception:  # pragma: no cover — defensive
        return {}


def _quote_toml(s: str) -> str:
    """Double-quote a string for TOML output, escaping ``\"`` and ``\\``."""
    return '"' + str(s).replace("\\", "\\\\").replace('"', '\\"') + '"'


def save_config(
    cfg: Dict[str, Any],
    path: Optional[Path] = None,
) -> Path:
    """Write the given preferences dict to disk.

    Creates the parent directory if needed. Overwrites any existing
    file. Format is the strict ``[llm]`` flat schema described in the
    module docstring.
    """
    p = path or config_path()
    p.parent.mkdir(parents=True, exist_ok=True)

    lines = [
        "# StatsPAI LLM preferences.",
        "# This file stores provider + model selection only — never",
        "# put API keys here. Set ANTHROPIC_API_KEY / OPENAI_API_KEY",
        "# in your environment instead.",
        "",
    ]
    for section, body in cfg.items():
        if isinstance(body, dict):
            lines.append(f"[{section}]")
            for k, v in body.items():
                if v is None:
                    continue
                if isinstance(v, bool):
                    lines.append(f"{k} = {'true' if v else 'false'}")
                elif isinstance(v, (int, float)):
                    lines.append(f"{k} = {v}")
                else:
                    lines.append(f"{k} = {_quote_toml(v)}")
            lines.append("")
        else:
            # flat top-level entry — discouraged, but write it.
            if isinstance(body, str):
                lines.append(f"{section} = {_quote_toml(body)}")
            else:
                lines.append(f"{section} = {body}")

    p.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return p


def set_preferences(
    *,
    provider: Optional[str] = None,
    model: Optional[str] = None,
    path: Optional[Path] = None,
) -> Path:
    """Convenience setter — loads, updates, saves.

    Either field can be ``None`` to leave it unchanged.
    """
    cfg = load_config(path)
    llm = cfg.setdefault("llm", {})
    if provider is not None:
        llm["provider"] = provider
    if model is not None:
        llm["model"] = model
    return save_config(cfg, path)
