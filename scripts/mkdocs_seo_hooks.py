"""MkDocs hooks: page descriptions, the package version, and llms.txt.

Registered under ``hooks:`` in ``mkdocs.yml``.

* Every page without a ``description:`` front-matter field gets one drawn from
  its first prose paragraph, so search engines and AI retrievers stop seeing
  the same site-wide ``site_description`` on all 400+ pages. A hand-written
  ``description:`` always wins.
* ``config.extra.statspai_version`` is read from ``pyproject.toml`` so the
  JSON-LD block in ``overrides/main.html`` never carries a stale version.
* The repository-root ``llms.txt`` is published at the site root with its
  relative links rewritten to absolute URLs (docs pages → this site, other
  repository files → GitHub), so the file served at ``<site>/llms.txt`` has no
  dead links. The root file stays the single source.
"""

from __future__ import annotations

import re
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPO_BLOB = "https://github.com/brycewang-stanford/StatsPAI/blob/main/"
MAX_LEN = 155

_LINK = re.compile(r"!?\[([^\]]*)\]\([^)]*\)")
_REF_LINK = re.compile(r"\[([^\]]+)\]\[[^\]]*\]")
_HTML = re.compile(r"<[^>]+>")
_EMPH = re.compile(r"(\*\*|__|\*|_)(?=\S)(.+?)(?<=\S)\1")
_SKIP_PREFIXES = (
    "#",
    ">",
    "|",
    "!!!",
    "???",
    ":::",
    "<!--",
    "- ",
    "* ",
    "+ ",
    "---",
    "===",
)


def _first_paragraph(markdown: str) -> str:
    """Return the first plain prose paragraph after the title, or ''."""
    in_fence = False
    para: list[str] = []
    for raw in markdown.splitlines():
        line = raw.strip()
        if line.startswith(("```", "~~~")):
            in_fence = not in_fence
            if para:
                break
            continue
        if in_fence:
            continue
        if not line:
            if para:
                break
            continue
        if (
            line.startswith(_SKIP_PREFIXES)
            or re.match(r"^\d+\.\s", line)
            or raw.startswith("    ")
        ):
            if para:
                break
            continue
        para.append(line)
    return " ".join(para)


def _plain(text: str) -> str:
    text = _LINK.sub(r"\1", text)
    text = _REF_LINK.sub(r"\1", text)
    text = _HTML.sub("", text)
    text = text.replace("`", "")
    for _ in range(2):
        text = _EMPH.sub(r"\2", text)
    return re.sub(r"\s+", " ", text).strip()


def _truncate(text: str, limit: int = MAX_LEN) -> str:
    if len(text) <= limit:
        return text
    cut = text[: limit - 1].rsplit(" ", 1)[0].rstrip(",;:—-– ")
    return cut + "…"


def describe(markdown: str) -> str:
    """Description derived from a page's first prose paragraph ('' if none)."""
    text = _plain(_first_paragraph(markdown))
    return _truncate(text) if len(text) >= 40 else ""


def on_config(config, **kwargs):
    with open(ROOT / "pyproject.toml", "rb") as fh:
        config.extra["statspai_version"] = tomllib.load(fh)["project"]["version"]
    return config


def _api_description(markdown: str) -> str:
    """For a generated ``::: statspai.<module>`` page, the module docstring."""
    match = re.search(r"^:::\s*(statspai(?:\.\w+)*)", markdown, flags=re.MULTILINE)
    if not match:
        return ""
    import importlib

    try:
        module = importlib.import_module(match.group(1))
    except ImportError:  # an optional extra is missing: keep the site default
        return ""
    doc = (module.__doc__ or "").strip()
    summary = _plain(doc.split("\n\n", 1)[0]) if doc else ""
    lead = f"StatsPAI API reference for {match.group(1)}"
    return _truncate(f"{lead}: {summary}" if summary else f"{lead}.")


def on_page_markdown(markdown, page, config, files, **kwargs):
    if not page.meta.get("description"):
        derived = describe(markdown) or _api_description(markdown)
        if derived:
            page.meta["description"] = derived
    return markdown


def absolutize_llms(text: str, site_url: str) -> str:
    """Rewrite llms.txt relative links to absolute URLs."""
    site_url = site_url.rstrip("/") + "/"

    def repl(match: re.Match) -> str:
        label, target = match.group(1), match.group(2)
        if re.match(r"^[a-z]+://", target) or target.startswith("#"):
            return match.group(0)
        path, _, anchor = target.partition("#")
        anchor = f"#{anchor}" if anchor else ""
        if path.startswith("docs/") and path.endswith(".md"):
            page = path[len("docs/") : -len(".md")]
            page = "" if page == "index" else page + "/"
            return f"[{label}]({site_url}{page}{anchor})"
        return f"[{label}]({REPO_BLOB}{path}{anchor})"

    return re.sub(r"\[([^\]]+)\]\(([^)\s]+)\)", repl, text)


def on_post_build(config, **kwargs):
    src = ROOT / "llms.txt"
    out = Path(config["site_dir"]) / "llms.txt"
    out.write_text(
        absolutize_llms(src.read_text(encoding="utf-8"), config["site_url"]),
        encoding="utf-8",
    )
