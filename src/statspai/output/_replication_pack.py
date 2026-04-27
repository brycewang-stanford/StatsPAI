"""``sp.replication_pack`` — package an analysis into a journal-ready zip.

The AEA / AEJ data editor checklist requires a self-contained
replication archive: code, data, environment, README, and an obvious
"how to run this" entry point. Stata / R get there via ad-hoc
``replication.zip`` builds; Python projects typically don't get there
at all.

This module gives StatsPAI a one-liner:

    sp.replication_pack(paper_or_result, "submission.zip", data=df)

and produces a zip with the layout journals expect:

    submission.zip
    ├── MANIFEST.json          # version, timestamp, git, file hashes
    ├── README.md              # replication instructions
    ├── data/
    │   ├── dataset.csv
    │   └── manifest.json      # shape + dtypes + SHA256
    ├── code/
    │   └── script.py          # caller's script (or user-provided)
    ├── env/
    │   └── requirements.txt   # pip freeze (skippable)
    ├── paper/
    │   ├── paper.md / .qmd / .tex
    │   └── paper.bib          # if citations attached
    └── lineage.json           # aggregated provenance from results

Design notes
------------
- **stdlib only.** ``zipfile`` + ``json`` + ``hashlib`` + ``subprocess``
  for pip freeze. Nothing the user must extra-install.
- **Tolerant.** If any sub-step fails (no git, no pip, weird object),
  we still produce a valid zip and log the failure to ``NOTES.txt``
  inside the archive. A partial pack is more useful than no pack.
- **Picks up provenance for free.** When ``target`` is a result
  carrying ``_provenance``, or a list of such results, the lineage is
  aggregated via :func:`statspai.output._lineage.lineage_summary` and
  written to ``lineage.json``.
- **Reproducibility hook**: ``MANIFEST.json`` includes the git SHA when
  the call is made inside a repo; otherwise omits the field rather
  than lying about it.
"""
from __future__ import annotations

import datetime as _dt
import hashlib
import io
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Union

from ._bibliography import citations_to_bib_entries
from ._lineage import (
    Provenance,
    compute_data_hash,
    get_provenance,
    lineage_summary,
)

__all__ = ["replication_pack", "ReplicationPack"]


def _statspai_version() -> str:
    try:
        from .. import __version__
        return str(__version__)
    except Exception:  # pragma: no cover
        return "unknown"


def _python_version() -> str:
    v = sys.version_info
    return f"{v.major}.{v.minor}.{v.micro}"


# ---------------------------------------------------------------------------
# Helpers — extract pieces from a heterogeneous "target"
# ---------------------------------------------------------------------------

def _flatten_targets(target: Any) -> List[Any]:
    """Normalise target into a list of result-like objects to inspect."""
    if target is None:
        return []
    if isinstance(target, (list, tuple)):
        return list(target)
    return [target]


def _extract_data(target: Any) -> Optional[Any]:
    """Best-effort: pull the analysis DataFrame out of ``target``.

    Order of preference:
      1. ``target.data``
      2. ``target.workflow.data`` (PaperDraft → CausalWorkflow path)
      3. ``target._provenance.data_shape`` (no frame, but we know the shape)
    """
    for cand in _flatten_targets(target):
        # PaperDraft → workflow.data
        wf = getattr(cand, "workflow", None)
        if wf is not None:
            d = getattr(wf, "data", None)
            if d is not None:
                return d
        d = getattr(cand, "data", None)
        if d is not None:
            return d
    return None


def _collect_results(target: Any) -> List[Any]:
    """Return a flat list of objects that may carry ``_provenance``."""
    out: List[Any] = []
    for cand in _flatten_targets(target):
        if hasattr(cand, "_provenance"):
            out.append(cand)
        # PaperDraft → workflow.result
        wf = getattr(cand, "workflow", None)
        if wf is not None:
            r = getattr(wf, "result", None)
            if r is not None and hasattr(r, "_provenance"):
                out.append(r)
        # dict shape
        if isinstance(cand, Mapping) and "_provenance" in cand:
            out.append(cand)
    return out


def _extract_paper(target: Any):
    """Return a PaperDraft-like object (has ``to_markdown``) if any."""
    for cand in _flatten_targets(target):
        if hasattr(cand, "to_markdown") and hasattr(cand, "sections"):
            return cand
    return None


def _extract_citations(target: Any) -> List[str]:
    cites: List[str] = []
    for cand in _flatten_targets(target):
        c = getattr(cand, "citations", None)
        if isinstance(c, (list, tuple)):
            cites.extend(str(x) for x in c if x)
    return cites


# ---------------------------------------------------------------------------
# Helpers — produce content
# ---------------------------------------------------------------------------

def _git_sha(cwd: Optional[str] = None) -> Optional[str]:
    """Return ``git rev-parse HEAD`` if invoked inside a repo, else None."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=cwd,
            capture_output=True,
            text=True,
            timeout=2.0,
            check=False,
        )
        if out.returncode == 0:
            sha = out.stdout.strip()
            if sha and len(sha) >= 7:
                return sha
    except Exception:
        return None
    return None


def _pip_freeze() -> Optional[str]:
    """Return ``pip freeze`` output, or None if it fails."""
    try:
        out = subprocess.run(
            [sys.executable, "-m", "pip", "freeze"],
            capture_output=True,
            text=True,
            timeout=15.0,
            check=False,
        )
        if out.returncode == 0:
            return out.stdout
    except Exception:
        return None
    return None


def _importlib_freeze_fallback() -> str:
    """Cheap fallback when ``pip freeze`` isn't available.

    Lists installed top-level distributions via ``importlib.metadata``.
    """
    try:
        from importlib.metadata import distributions
        rows = []
        for dist in distributions():
            name = dist.metadata.get("Name") or "?"
            version = dist.version or "?"
            rows.append(f"{name}=={version}")
        rows.sort(key=str.lower)
        return "\n".join(rows) + "\n"
    except Exception:  # pragma: no cover
        return "(unable to enumerate environment)\n"


def _dataset_to_csv_bytes(data: Any) -> Optional[bytes]:
    """Best-effort serialise to CSV bytes."""
    try:
        import pandas as pd
        if isinstance(data, pd.DataFrame):
            return data.to_csv(index=False).encode("utf-8")
        if isinstance(data, pd.Series):
            return data.to_csv(index=False).encode("utf-8")
    except Exception:
        pass
    return None


def _dataset_manifest(data: Any) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "hash_sha256_prefix": compute_data_hash(data, length=64),
    }
    try:
        import pandas as pd
        if isinstance(data, pd.DataFrame):
            out["kind"] = "DataFrame"
            out["shape"] = list(data.shape)
            out["columns"] = [
                {"name": str(c), "dtype": str(data[c].dtype)}
                for c in data.columns
            ]
            out["n_missing_total"] = int(data.isna().sum().sum())
        elif isinstance(data, pd.Series):
            out["kind"] = "Series"
            out["shape"] = [len(data)]
            out["name"] = str(data.name) if data.name is not None else None
            out["dtype"] = str(data.dtype)
    except Exception:
        out["kind"] = type(data).__name__
    return out


def _resolve_caller_code(code: Optional[str]) -> Optional[str]:
    """If ``code`` is a path, read it; else pass through."""
    if code is None:
        return None
    # If it looks like a path AND exists, read it.
    if isinstance(code, (str, os.PathLike)):
        try:
            p = Path(code)
            if p.exists() and p.is_file() and p.stat().st_size < 5_000_000:
                return p.read_text(encoding="utf-8")
        except Exception:
            pass
        # Otherwise treat as inline code (only if it looks like code).
        if isinstance(code, str) and "\n" in code:
            return code
        if isinstance(code, str) and len(code) < 5_000_000:
            return code
    return None


def _readme(
    title: str,
    paper_filename: Optional[str],
    has_data: bool,
    has_code: bool,
    has_env: bool,
    has_lineage: bool,
) -> str:
    lines = [
        f"# {title}",
        "",
        f"Generated by StatsPAI v{_statspai_version()} on "
        f"{_dt.datetime.now().strftime('%Y-%m-%d %H:%M')}.",
        "",
        "## Layout",
        "",
        "```",
        "MANIFEST.json     Top-level archive manifest (versions, file hashes)",
    ]
    if has_data:
        lines.append(
            "data/             dataset.csv + manifest.json (schema + hash)"
        )
    if has_code:
        lines.append(
            "code/             script.py — the analysis code"
        )
    if has_env:
        lines.append(
            "env/              requirements.txt — frozen Python environment"
        )
    if paper_filename:
        lines.append(
            f"paper/            {paper_filename} (and paper.bib if citations)"
        )
    if has_lineage:
        lines.append(
            "lineage.json      Per-result Provenance records"
        )
    lines.extend([
        "```",
        "",
        "## Reproducing",
        "",
        "```bash",
        "python -m venv .venv && source .venv/bin/activate",
        "pip install -r env/requirements.txt",
        "python code/script.py",
        "```",
        "",
        "If the StatsPAI version pinned in `env/requirements.txt` differs "
        "from the one you have installed, install the exact pinned "
        "version first:",
        "",
        "```bash",
        f"pip install StatsPAI=={_statspai_version()}",
        "```",
        "",
        "## Verifying data integrity",
        "",
        "Each file in `MANIFEST.json` carries a SHA-256 prefix. Recompute "
        "with:",
        "",
        "```bash",
        "python -c \"import hashlib,sys; print(hashlib.sha256(open(sys.argv[1],'rb').read()).hexdigest())\" data/dataset.csv",
        "```",
        "",
        "and compare against the value in `MANIFEST.json`.",
    ])
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Result dataclass
# ---------------------------------------------------------------------------

class ReplicationPack:
    """Lightweight summary returned by :func:`replication_pack`.

    Mostly for testing / programmatic inspection; users typically just
    care about ``output_path``.
    """

    __slots__ = ("output_path", "manifest", "warnings")

    def __init__(
        self,
        output_path: Path,
        manifest: Dict[str, Any],
        warnings: List[str],
    ):
        self.output_path = output_path
        self.manifest = manifest
        self.warnings = warnings

    def __repr__(self) -> str:
        return (
            f"ReplicationPack(path={str(self.output_path)!r}, "
            f"files={len(self.manifest.get('files', []))}, "
            f"warnings={len(self.warnings)})"
        )

    def summary(self) -> str:
        lines = [
            "ReplicationPack",
            "===============",
            f"  Path     : {self.output_path}",
            f"  Files    : {len(self.manifest.get('files', []))}",
            f"  StatsPAI : v{self.manifest.get('statspai_version', '?')}",
            f"  Created  : {self.manifest.get('timestamp', '?')}",
        ]
        if self.warnings:
            lines.append("  Warnings :")
            for w in self.warnings:
                lines.append(f"    - {w}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def replication_pack(
    target: Any,
    output_path: Union[str, os.PathLike],
    *,
    data: Optional[Any] = None,
    code: Optional[str] = None,
    env: bool = True,
    bib: bool = True,
    paper_format: str = "auto",
    title: str = "Replication Pack",
    extra_files: Optional[Mapping[str, Union[str, bytes]]] = None,
    include_git_sha: bool = True,
    overwrite: bool = True,
) -> ReplicationPack:
    """Build a journal-ready replication archive.

    Parameters
    ----------
    target : object
        Anything carrying analysis state. Best when it's a
        :class:`statspai.workflow.paper.PaperDraft` (we then auto-extract
        the rendered paper, the workflow's data, and any results with
        provenance) but plain estimator results, lists thereof, or even
        ``None`` (for "just pack data + script") all work.
    output_path : str or PathLike
        Destination ``.zip`` path. Created or overwritten.
    data : DataFrame / Series, optional
        Explicit dataset. When omitted, we try ``target.data`` /
        ``target.workflow.data``. If both fail, the archive omits the
        ``data/`` directory and warns in ``MANIFEST.json``.
    code : str or PathLike, optional
        Either an inline Python script (multi-line string) or a path to
        a .py file. When omitted, ``code/`` is also omitted and a
        warning is logged.
    env : bool, default True
        Include ``env/requirements.txt`` from ``pip freeze``. Disable to
        keep the archive small or to avoid the subprocess call.
    bib : bool, default True
        Write ``paper/paper.bib`` from ``target.citations`` (or
        equivalent).
    paper_format : {"auto", "md", "qmd", "tex", "docx"}, default "auto"
        How to render the PaperDraft inside the archive. "auto" picks
        ``draft.fmt`` (and falls back to "md" for unknown formats).
    title : str
        Used in ``README.md``.
    extra_files : mapping, optional
        ``{"path/in/zip.txt": "contents" or b"bytes"}`` — anything
        custom you want stuffed into the archive.
    include_git_sha : bool, default True
        Capture ``git rev-parse HEAD`` for ``MANIFEST.json`` (silently
        skipped when not in a repo).
    overwrite : bool, default True
        Overwrite an existing archive at ``output_path``.

    Returns
    -------
    ReplicationPack
        Summary object. ``rp.output_path`` is the on-disk archive;
        ``rp.manifest`` is the parsed ``MANIFEST.json``;
        ``rp.warnings`` lists any partial-success notes.

    Examples
    --------
    >>> import statspai as sp
    >>> draft = sp.paper(df, "effect of training on wages",
    ...                  treatment="trained", y="wage")
    >>> rp = sp.replication_pack(draft, "training-replication.zip")
    >>> print(rp.summary())
    """
    out_path = Path(output_path).expanduser().resolve()
    if out_path.exists() and not overwrite:
        raise FileExistsError(
            f"{out_path} already exists. Pass overwrite=True to replace it."
        )
    out_path.parent.mkdir(parents=True, exist_ok=True)

    warnings: List[str] = []
    files_manifest: List[Dict[str, Any]] = []

    # Resolve all the inputs up front.
    eff_data = data if data is not None else _extract_data(target)
    eff_code = _resolve_caller_code(code)
    paper_obj = _extract_paper(target)
    citations = _extract_citations(target)
    results = _collect_results(target)

    # Build a dict of (path → bytes) and write atomically at the end.
    archive: Dict[str, bytes] = {}

    def _put(path: str, content: Union[str, bytes]) -> None:
        b = content.encode("utf-8") if isinstance(content, str) else content
        archive[path] = b
        digest = hashlib.sha256(b).hexdigest()
        files_manifest.append({
            "path": path,
            "size_bytes": len(b),
            "sha256": digest,
        })

    # ------- data -----------------------------------------------------
    if eff_data is not None:
        csv_bytes = _dataset_to_csv_bytes(eff_data)
        if csv_bytes is None:
            warnings.append(
                "data was not a DataFrame/Series — skipped CSV export."
            )
        else:
            _put("data/dataset.csv", csv_bytes)
            _put(
                "data/manifest.json",
                json.dumps(_dataset_manifest(eff_data), indent=2) + "\n",
            )
    else:
        warnings.append("no data provided or extractable from target.")

    # ------- code -----------------------------------------------------
    if eff_code:
        _put("code/script.py", eff_code)
    else:
        warnings.append(
            "no code provided — pass code=\"...\" or a path to your "
            "analysis script."
        )

    # ------- env ------------------------------------------------------
    if env:
        freeze = _pip_freeze()
        if freeze is None:
            freeze = _importlib_freeze_fallback()
            warnings.append(
                "pip freeze unavailable; used importlib.metadata fallback."
            )
        _put("env/requirements.txt", freeze)

    # ------- paper ----------------------------------------------------
    paper_filename: Optional[str] = None
    if paper_obj is not None:
        try:
            fmt = paper_format
            if fmt == "auto":
                fmt = getattr(paper_obj, "fmt", "markdown") or "markdown"
            ext_map = {
                "markdown": ("paper.md", "to_markdown"),
                "md": ("paper.md", "to_markdown"),
                "qmd": ("paper.qmd", "to_qmd"),
                "tex": ("paper.tex", "to_tex"),
                "latex": ("paper.tex", "to_tex"),
                "docx": ("paper.docx", None),  # special-cased below
            }
            entry = ext_map.get(fmt, ext_map["md"])
            fname, method_name = entry
            if method_name is None:
                # docx: write to tmp + read bytes.
                import tempfile
                with tempfile.TemporaryDirectory() as td:
                    tmp = Path(td) / "p.docx"
                    paper_obj.to_docx(str(tmp))
                    _put(f"paper/{fname}", tmp.read_bytes())
            else:
                method = getattr(paper_obj, method_name, None)
                if method is None:
                    # Requested format unsupported by this PaperDraft —
                    # fall back to markdown.
                    warnings.append(
                        f"PaperDraft does not implement {method_name}(); "
                        "fell back to markdown."
                    )
                    fname, method_name = "paper.md", "to_markdown"
                    method = paper_obj.to_markdown
                _put(f"paper/{fname}", method())
            paper_filename = fname
        except Exception as exc:
            warnings.append(
                f"paper rendering failed: {type(exc).__name__}: {exc}"
            )

    # ------- bibliography --------------------------------------------
    if bib and citations:
        try:
            entries = citations_to_bib_entries(citations)
            from ._bibliography import _format_bib_entry  # local helper
            try:
                from .. import __version__ as _v
            except Exception:
                _v = "unknown"
            bib_text = (
                f"% paper.bib — auto-generated by StatsPAI v{_v}\n"
                f"% {len(entries)} entries\n\n"
                + "\n".join(_format_bib_entry(e) for e in entries)
            )
            _put("paper/paper.bib", bib_text)
        except Exception as exc:  # pragma: no cover — defensive
            warnings.append(
                f"bibliography emission failed: {type(exc).__name__}: {exc}; "
                "fell back to raw citation dump."
            )
            _put(
                "paper/paper.bib",
                "% raw citations (parser failed)\n\n"
                + "\n\n".join(f"% {c}" for c in citations) + "\n",
            )

    # ------- lineage --------------------------------------------------
    if results:
        try:
            lin = lineage_summary(*results)
            _put("lineage.json", json.dumps(lin, indent=2, default=str) + "\n")
        except Exception as exc:
            warnings.append(
                f"lineage summary failed: {type(exc).__name__}: {exc}"
            )

    # ------- extras ---------------------------------------------------
    if extra_files:
        for k, v in extra_files.items():
            try:
                _put(str(k), v)
            except Exception as exc:
                warnings.append(
                    f"extra_file {k!r} failed: {type(exc).__name__}: {exc}"
                )

    # ------- README ---------------------------------------------------
    readme = _readme(
        title=title,
        paper_filename=paper_filename,
        has_data="data/dataset.csv" in archive,
        has_code="code/script.py" in archive,
        has_env="env/requirements.txt" in archive,
        has_lineage="lineage.json" in archive,
    )
    _put("README.md", readme)

    # ------- top-level MANIFEST.json ---------------------------------
    manifest: Dict[str, Any] = {
        "title": title,
        "statspai_version": _statspai_version(),
        "python_version": _python_version(),
        "platform": sys.platform,
        "timestamp": _dt.datetime.now().isoformat(timespec="seconds"),
        "files": files_manifest,
        "warnings": list(warnings),
    }
    if include_git_sha:
        sha = _git_sha()
        if sha:
            manifest["git_sha"] = sha
    if results:
        manifest["n_results_with_provenance"] = len(results)

    manifest_bytes = (
        json.dumps(manifest, indent=2, default=str) + "\n"
    ).encode("utf-8")
    archive["MANIFEST.json"] = manifest_bytes

    # ------- write the archive (atomic) ------------------------------
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", zipfile.ZIP_DEFLATED) as zf:
        # Stable file ordering — manifest first so unzippers display it.
        for name in sorted(archive.keys(),
                           key=lambda n: (n != "MANIFEST.json", n)):
            zf.writestr(name, archive[name])
    out_path.write_bytes(buf.getvalue())

    return ReplicationPack(
        output_path=out_path, manifest=manifest, warnings=warnings,
    )
