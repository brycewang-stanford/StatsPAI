"""The packaged skill over MCP (extension ``io.modelcontextprotocol/skills``).

StatsPAI ships an Agent Skill (``agent/_skill``, the ``statspai-analysis``
workflow) that ``statspai skill install`` copies into a skills directory.
The Skills extension (SEP-2640, Final) lets an MCP host discover and load the
same skill from the server instead, without a copy that goes stale on
``pip install -U``:

* ``skills/list`` returns one entry: the ``SKILL.md`` URI, its frontmatter
  (every field, unchanged) and the complete file manifest (URI, SHA-256
  digest and byte size of each file, computed from the bytes served);
* ``skills/get`` returns that entry by URI (unknown URI: ``-32602``);
* ``resources/read`` serves each file under ``skill://statspai-analysis/``.

Both list and get results carry ``resultType`` / ``ttlMs`` / ``cacheScope``
(added by the per-request result decoration). ``resources/directory/read``
is optional and not implemented, so the extension is declared without
``directoryRead``.
"""

from __future__ import annotations

import hashlib
from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

SKILLS_EXTENSION = "io.modelcontextprotocol/skills"
SKILL_NAME = "statspai-analysis"
SKILL_ROOT = f"skill://{SKILL_NAME}"
SKILL_MD_URI = f"{SKILL_ROOT}/SKILL.md"

_MIME = {".md": "text/markdown", ".py": "text/x-python"}


def skill_dir() -> Path:
    return Path(__file__).resolve().parent / "_skill"


def _skill_files() -> List[Tuple[str, Path]]:
    """``(relative posix path, file)`` for every file the skill serves."""
    root = skill_dir()
    out = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or "__pycache__" in path.parts:
            continue
        if path.suffix in {".pyc", ".pyo"} or path.name.startswith("."):
            continue
        out.append((path.relative_to(root).as_posix(), path))
    return out


def parse_frontmatter(text: str) -> Dict[str, Any]:
    """Frontmatter of a ``SKILL.md`` whose fields are single-line scalars.

    The packaged skill uses only ``key: value`` lines, so this avoids a
    PyYAML dependency; ``tests/test_mcp_skills_extension.py`` checks the
    result equals ``yaml.safe_load`` (what a host compares against). A line
    of any other shape raises, rather than silently producing a field the
    host would parse differently.
    """
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        raise ValueError("SKILL.md must begin with YAML frontmatter")
    fields: Dict[str, Any] = {}
    for line in lines[1:]:
        if line.strip() == "---":
            return fields
        key, sep, value = line.partition(": ")
        if not sep or not key or key != key.strip() or key.startswith("#"):
            raise ValueError(f"unsupported frontmatter line: {line!r}")
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        fields[key] = value
    raise ValueError("SKILL.md frontmatter is not closed")


@lru_cache(maxsize=1)
def skill_entry() -> Dict[str, Any]:
    """The skill entry shared by ``skills/list`` and ``skills/get``."""
    manifest = []
    frontmatter: Dict[str, Any] = {}
    for rel, path in _skill_files():
        raw = path.read_bytes()
        manifest.append(
            {
                "uri": f"{SKILL_ROOT}/{rel}",
                "digest": "sha256:" + hashlib.sha256(raw).hexdigest(),
                "size": len(raw),
            }
        )
        if rel == "SKILL.md":
            frontmatter = parse_frontmatter(raw.decode("utf-8"))
    if frontmatter.get("name") != SKILL_NAME:
        raise RuntimeError(
            f"packaged SKILL.md names {frontmatter.get('name')!r}, "
            f"expected {SKILL_NAME!r}"
        )
    return {"uri": SKILL_MD_URI, "frontmatter": frontmatter, "resources": manifest}


def handle_skills_list(params: Dict[str, Any]) -> Dict[str, Any]:
    return {"skills": [skill_entry()]}


def handle_skills_get(
    params: Dict[str, Any], *, InvalidParamsError: type
) -> Dict[str, Any]:
    uri = params.get("uri")
    if uri != SKILL_MD_URI:
        raise InvalidParamsError(
            f"Unknown skill {uri!r}. This server serves {SKILL_MD_URI}."
        )
    return {"skill": skill_entry()}


def read_skill_file(uri: str) -> Optional[Dict[str, Any]]:
    """``resources/read`` contents for a skill file, or ``None`` if unknown."""
    if not uri.startswith(SKILL_ROOT + "/"):
        return None
    rel = uri[len(SKILL_ROOT) + 1 :]
    for name, path in _skill_files():
        if name == rel:
            return {
                "contents": [
                    {
                        "uri": uri,
                        "mimeType": _MIME.get(path.suffix, "text/plain"),
                        "text": path.read_bytes().decode("utf-8"),
                    }
                ]
            }
    return None
