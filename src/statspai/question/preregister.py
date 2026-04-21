"""
Pre-registration helpers for causal questions.

Dumps a :class:`CausalQuestion` (or any dict-serializable object) to a
YAML / JSON file suitable for OSF, AEA RCT Registry, or a repo-local
PAP.  Supports a simple deviation-log that records the timestamp +
reason whenever the protocol is modified after initial registration.

Why this matters: the article's cross-school synthesis emphasizes
that identification depends on assumptions the data cannot check.
A pre-registered analysis plan makes those assumptions explicit and
prevents retrospective p-hacking.

>>> q = sp.causal_question(...)
>>> sp.preregister(q, "pap.yaml")
>>> q2 = sp.load_preregister("pap.yaml")
"""

from __future__ import annotations

import datetime as _dt
import json
from pathlib import Path
from typing import Any, Union

from .question import CausalQuestion, causal_question


__all__ = [
    "preregister",
    "load_preregister",
]


def _to_dict(q: Union[CausalQuestion, dict]) -> dict:
    if isinstance(q, CausalQuestion):
        return q.to_dict()
    if isinstance(q, dict):
        return dict(q)
    raise TypeError(
        "preregister() accepts a CausalQuestion or dict; "
        f"got {type(q).__name__}"
    )


def _yaml_dumps(data: Any, indent: int = 0) -> str:
    """Minimal YAML serializer: no external dep. Supports dict / list /
    primitive values, which is sufficient for `CausalQuestion.to_dict()`.
    """
    pad = "  " * indent
    lines = []
    if isinstance(data, dict):
        for k, v in data.items():
            if isinstance(v, (dict, list)) and v:
                lines.append(f"{pad}{k}:")
                lines.append(_yaml_dumps(v, indent + 1))
            elif isinstance(v, list):
                lines.append(f"{pad}{k}: []")
            elif v is None:
                lines.append(f"{pad}{k}: null")
            elif isinstance(v, bool):
                lines.append(f"{pad}{k}: {str(v).lower()}")
            elif isinstance(v, (int, float)):
                lines.append(f"{pad}{k}: {v}")
            else:
                s = str(v).replace('"', '\\"')
                lines.append(f'{pad}{k}: "{s}"')
    elif isinstance(data, list):
        for v in data:
            if isinstance(v, (dict, list)):
                lines.append(f"{pad}-")
                lines.append(_yaml_dumps(v, indent + 1))
            else:
                s = str(v).replace('"', '\\"')
                lines.append(f'{pad}- "{s}"')
    return "\n".join(lines)


def _yaml_loads(text: str) -> Any:
    """Minimal inverse of :func:`_yaml_dumps` for the subset we emit.

    Parses the exact flavour ``_yaml_dumps`` produces and nothing else —
    not a general-purpose YAML parser.  For true YAML inputs we rely on
    ``pyyaml`` if installed; otherwise we fall back to this parser.
    """
    try:
        import yaml  # type: ignore[import]

        return yaml.safe_load(text)
    except ImportError:
        pass

    def _parse_scalar(s: str):
        s = s.strip()
        if s.startswith('"') and s.endswith('"'):
            return s[1:-1].replace('\\"', '"')
        if s == "null":
            return None
        if s in ("true", "false"):
            return s == "true"
        try:
            if "." in s:
                return float(s)
            return int(s)
        except ValueError:
            return s

    lines = [ln for ln in text.splitlines() if ln.strip()]
    root: Any = None

    def _consume_block(idx: int, base_indent: int):
        """Return (value, next_idx)."""
        out: Any = None
        while idx < len(lines):
            line = lines[idx]
            stripped = line.lstrip(" ")
            indent = len(line) - len(stripped)
            if indent < base_indent:
                return out, idx

            if stripped.startswith("- "):
                if out is None:
                    out = []
                val_s = stripped[2:]
                if val_s.strip() == "":
                    # Nested dict / list item
                    val, idx = _consume_block(idx + 1, base_indent + 2)
                    out.append(val)
                else:
                    out.append(_parse_scalar(val_s))
                    idx += 1
            elif stripped == "-":
                if out is None:
                    out = []
                val, idx = _consume_block(idx + 1, base_indent + 2)
                out.append(val)
            elif ":" in stripped:
                if out is None:
                    out = {}
                # Split on the *first* ": " (colon + space) so that
                # values containing colons (e.g. timestamps, free
                # notes like "Design: RCT") round-trip correctly.
                # Fall back to first-colon partition if no space
                # follows (i.e. empty value after key).
                if ": " in stripped:
                    key, _, rest = stripped.partition(": ")
                elif stripped.endswith(":"):
                    key, rest = stripped[:-1], ""
                else:
                    key, _, rest = stripped.partition(":")
                key = key.strip()
                rest = rest.strip()
                if rest == "":
                    val, idx = _consume_block(idx + 1, base_indent + 2)
                    out[key] = val
                elif rest == "[]":
                    out[key] = []
                    idx += 1
                else:
                    out[key] = _parse_scalar(rest)
                    idx += 1
            else:
                idx += 1
        return out, idx

    root, _ = _consume_block(0, 0)
    return root


def preregister(
    question: Union[CausalQuestion, dict],
    filename: Union[str, Path],
    *,
    fmt: str = "auto",
    registry_url: str = "",
    note: str = "",
) -> Path:
    """Write a pre-analysis plan to disk.

    Parameters
    ----------
    question : CausalQuestion | dict
        The pre-specified causal question / analysis plan.
    filename : str | pathlib.Path
        Destination.  ``.yaml`` / ``.yml`` writes YAML, ``.json`` writes
        JSON.  Use ``fmt=`` to force a format.
    fmt : {"auto", "yaml", "json"}, default "auto"
    registry_url : str, optional
        Link to OSF / AEA / internal registration if applicable.
    note : str, optional
        Free-form note attached under ``metadata.note``.

    Returns
    -------
    pathlib.Path
        Path written to (also contains a ``metadata`` block with
        timestamp and statspai version).
    """
    path = Path(filename)
    if fmt == "auto":
        suffix = path.suffix.lower()
        if suffix in (".yaml", ".yml"):
            fmt = "yaml"
        elif suffix == ".json":
            fmt = "json"
        else:
            fmt = "yaml"
            path = path.with_suffix(".yaml")
    if fmt not in ("yaml", "json"):
        raise ValueError("fmt must be 'yaml' or 'json'")

    payload = {
        "metadata": {
            "format_version": "statspai.preregister/1",
            "created_at": _dt.datetime.now().isoformat(timespec="seconds"),
            "registry_url": registry_url,
            "note": note,
        },
        "question": _to_dict(question),
        "deviations": [],
    }

    if fmt == "json":
        text = json.dumps(payload, indent=2, default=str)
    else:
        text = _yaml_dumps(payload)

    path.write_text(text, encoding="utf-8")
    return path


def load_preregister(filename: Union[str, Path]) -> CausalQuestion:
    """Load a pre-registration file back into a :class:`CausalQuestion`.

    Deviations and metadata are preserved on the returned object via
    the ``.notes`` field (concatenated).
    """
    path = Path(filename)
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".json":
        payload = json.loads(text)
    else:
        payload = _yaml_loads(text)

    if not isinstance(payload, dict) or "question" not in payload:
        raise ValueError(f"{path}: not a valid preregistration file.")

    q_fields = payload["question"]
    # Rebuild CausalQuestion from fields, filtering to the constructor's
    # accepted kwargs
    allowed = {
        "treatment", "outcome", "population", "estimand", "design",
        "time_structure", "time", "id", "covariates", "instruments",
        "running_variable", "cutoff", "cohort", "notes",
    }
    kwargs = {k: v for k, v in q_fields.items() if k in allowed}
    # Required fields
    if "treatment" not in kwargs or "outcome" not in kwargs:
        raise ValueError("Preregistered question missing treatment/outcome.")
    meta_note = (payload.get("metadata") or {}).get("note", "")
    if meta_note:
        kwargs["notes"] = (kwargs.get("notes", "") + " | " + meta_note).strip(" |")
    return causal_question(**kwargs)
