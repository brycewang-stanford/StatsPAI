"""``sp.doctor()`` / ``statspai doctor``: which StatsPAI, in which Python.

A machine often has several Python environments, and StatsPAI can be
installed in more than one of them at different versions. A user (or an
agent) who runs ``python3`` and gets an ``ImportError``, or an old
version, loses time finding the interpreter that has the current one.
This module answers four questions in one report:

* which StatsPAI this process imported, from where, and whether it is an
  editable install (and whether the pip metadata agrees with the code);
* which interpreter and environment (``sys.executable``, ``sys.prefix``,
  virtualenv / conda) it runs in;
* which optional backends are installed, at which version (the Rust
  kernels with their API level, pyfixest, numba, jax, torch, pymc,
  matplotlib, polars, pyarrow), and whether R and Stata are on the
  machine;
* which other ``python3`` / ``python3.x`` interpreters are on ``PATH``
  and which StatsPAI each of them has.

Nothing is imported eagerly. Optional packages are located with
``importlib.util.find_spec`` and versioned from their metadata; they are
imported only when ``check_imports=True``, and then in a child process,
so a broken extra cannot hurt the caller. Every probe has a timeout and
an error becomes a field of the report, never an exception.
"""

from __future__ import annotations

import concurrent.futures
import importlib.util
import json
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

__all__ = ["doctor", "format_doctor", "runtime_summary"]

#: Optional backends reported by the doctor: import name -> (distribution
#: name on PyPI, what it enables). Order is the display order.
BACKENDS: Dict[str, Tuple[str, str]] = {
    "statspai_hdfe": ("statspai-hdfe", "Rust kernels (HDFE, GLM IRLS, cluster meat)"),
    "pyfixest": ("pyfixest", "fixest backend (sp.feols backend='pyfixest')"),
    "numba": ("numba", "JIT kernels (forests, ETS, designs)"),
    "jax": ("jax", "GPU / JAX-accelerated estimators"),
    "torch": ("torch", "neural and deep-IV estimators"),
    "pymc": ("pymc", "PyMC Bayesian estimators"),
    "matplotlib": ("matplotlib", "plots"),
    "polars": ("polars", "polars DataFrame input"),
    "pyarrow": ("pyarrow", "Arrow / Parquet input"),
}

#: Environment variables whose value must not be printed.
_SECRET = re.compile(r"TOKEN|SECRET|PASSWORD|KEY", re.IGNORECASE)

#: Interpreter names looked up on PATH (``python3.12`` etc. are matched by
#: the pattern; ``python`` is included because on many systems it is the
#: one users type).
_INTERPRETER_NAME = re.compile(r"^python(3(\.\d+)?)?(\.exe)?$", re.IGNORECASE)

#: Code run by each probed interpreter. It must work on any Python 3 (and
#: fail cleanly on Python 2), import nothing from StatsPAI, and print one
#: JSON line.
_PROBE_CODE = (
    "import json, sys\n"
    "out = {'python_version': sys.version.split()[0], "
    "'executable': sys.executable, 'prefix': sys.prefix}\n"
    "try:\n"
    "    import importlib.util as u\n"
    "    sys.path[:] = [p for p in sys.path if p not in ('', '.')]\n"
    "    s = u.find_spec('statspai')\n"
    "    out['statspai_location'] = s.origin if s else None\n"
    "    try:\n"
    "        import importlib.metadata as m\n"
    "        out['statspai_version'] = m.version('statspai')\n"
    "    except Exception:\n"
    "        out['statspai_version'] = None\n"
    "except Exception as exc:\n"
    "    out['error'] = type(exc).__name__ + ': ' + str(exc)\n"
    "print(json.dumps(out))\n"
)


# --------------------------------------------------------------------- #
#  Small helpers
# --------------------------------------------------------------------- #


def _dist_version(dist: str) -> Optional[str]:
    try:
        from importlib.metadata import PackageNotFoundError, version
    except ImportError:  # pragma: no cover - Python < 3.8
        return None
    try:
        return version(dist)
    except PackageNotFoundError:
        return None


def _install_record() -> Dict[str, Any]:
    """How pip installed ``statspai``: editable or not, and from where.

    PEP 610 ``direct_url.json`` marks an editable install with
    ``dir_info.editable``; a wheel from PyPI has no such file.
    """
    out: Dict[str, Any] = {"metadata_version": None, "editable": False, "url": None}
    try:
        from importlib.metadata import PackageNotFoundError, distribution
    except ImportError:  # pragma: no cover
        return out
    try:
        dist = distribution("statspai")
    except PackageNotFoundError:
        out["editable"] = None
        out["note"] = "no pip metadata (running from a source tree on sys.path)"
        return out
    out["metadata_version"] = dist.version
    raw = None
    try:
        raw = dist.read_text("direct_url.json")
    except OSError:
        raw = None
    if raw:
        try:
            info = json.loads(raw)
        except ValueError:
            info = {}
        out["url"] = info.get("url")
        out["editable"] = bool((info.get("dir_info") or {}).get("editable"))
    return out


def _install_kind(install: Dict[str, Any], location: str) -> str:
    """One phrase for how the imported code got onto ``sys.path``.

    An editable install records its checkout in ``direct_url.json``. When
    the imported package is not that checkout's ``statspai`` (a
    ``PYTHONPATH`` entry or a second tree shadows it), say so: otherwise
    the report would call foreign code "editable".
    """
    editable = install.get("editable")
    if editable is None:
        return "source tree on sys.path (no pip metadata)"
    if not editable:
        return "installed package"
    url = install.get("url") or ""
    if url.startswith("file://"):
        from urllib.parse import unquote, urlparse

        root = Path(unquote(urlparse(url).path))
        here = Path(location).resolve()
        candidates = [root / "src" / "statspai", root / "statspai"]
        if not any(c.resolve() == here for c in candidates):
            return f"on sys.path ahead of the editable install at {root}"
    return "editable install"


def _environment_kind() -> Dict[str, Any]:
    in_venv = sys.prefix != getattr(sys, "base_prefix", sys.prefix)
    conda = os.environ.get("CONDA_PREFIX")
    kind = "venv" if in_venv else ("conda" if conda else "system")
    if conda and os.path.realpath(conda) == os.path.realpath(sys.prefix):
        kind = "conda"
    return {
        "kind": kind,
        "prefix": sys.prefix,
        "base_prefix": getattr(sys, "base_prefix", sys.prefix),
        "virtual_env": os.environ.get("VIRTUAL_ENV"),
        "conda_prefix": conda,
    }


def _statspai_env() -> Dict[str, str]:
    out = {}
    for key, value in sorted(os.environ.items()):
        if key.startswith("STATSPAI_") or key == "PYTHONPATH":
            out[key] = "<set>" if _SECRET.search(key) else value
    return out


def _rust_report() -> Dict[str, Any]:
    from . import _rust_backend as rb

    info = dict(rb.info())
    info["skip_rust_env"] = os.environ.get("STATSPAI_SKIP_RUST")
    info["distribution_version"] = _dist_version("statspai-hdfe")
    return info


def _backend_rows(check_imports: bool, timeout: float) -> Dict[str, Dict[str, Any]]:
    rows: Dict[str, Dict[str, Any]] = {}
    for name, (dist, purpose) in BACKENDS.items():
        try:
            spec = importlib.util.find_spec(name)
        except (ImportError, ValueError) as exc:
            spec = None
            find_error: Optional[str] = f"{type(exc).__name__}: {exc}"
        else:
            find_error = None
        row: Dict[str, Any] = {
            "installed": spec is not None,
            "version": _dist_version(dist),
            "purpose": purpose,
            "location": getattr(spec, "origin", None) if spec else None,
            # True when this process already imported it; None = not checked.
            "imports": True if name in sys.modules else None,
        }
        if find_error:
            row["error"] = find_error
        rows[name] = row
    if check_imports:
        todo = [n for n, r in rows.items() if r["installed"] and r["imports"] is None]
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            results = dict(
                zip(todo, pool.map(lambda n: _import_in_child(n, timeout), todo))
            )
        for name, (ok, err) in results.items():
            rows[name]["imports"] = ok
            if err:
                rows[name]["import_error"] = err
    return rows


def _child_env() -> Dict[str, str]:
    """Environment for child interpreters: without this shell's PYTHONPATH.

    A development checkout on ``PYTHONPATH`` would make every interpreter
    look as if it had that StatsPAI.
    """
    return {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}


def _run(
    cmd: List[str], timeout: float
) -> Tuple[Optional[str], Optional[str], Optional[int]]:
    """Run ``cmd``; ``(stdout + stderr, error, returncode)``. Never raises.

    ``close_fds=False`` and no ``cwd`` keep :mod:`subprocess` on
    ``posix_spawn``: a ``fork`` taken after numerical code ran in this
    process can crash the child on macOS.
    """
    try:
        proc = subprocess.run(  # noqa: S603 - fixed argv, no shell
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            env=_child_env(),
            close_fds=False,
        )
    except (OSError, subprocess.SubprocessError) as exc:
        return None, f"{type(exc).__name__}: {exc}", None
    return (proc.stdout or "") + (proc.stderr or ""), None, proc.returncode


def _import_in_child(name: str, timeout: float) -> Tuple[Optional[bool], Optional[str]]:
    out, err, code = _run([sys.executable, "-c", f"import {name}"], timeout)
    if err:
        return None, err
    if code != 0:
        tail = (out or "").strip().splitlines()
        return False, (tail[-1] if tail else f"exit status {code}")[:300]
    return True, None


def _which_r(timeout: float) -> Dict[str, Any]:
    path = shutil.which("Rscript")
    out: Dict[str, Any] = {"found": path is not None, "path": path, "version": None}
    if path:
        text, err, _ = _run([path, "--version"], timeout)
        if err:
            out["error"] = err
        elif text:
            m = re.search(r"version\s+(\d+\.\d+\.\d+)", text)
            out["version"] = m.group(1) if m else text.strip().splitlines()[0][:120]
    return out


def _which_stata() -> Dict[str, Any]:
    """Stata executables on PATH or in the usual install folders (not run)."""
    hits: List[str] = []
    for name in ("stata-mp", "stata-se", "stata", "StataMP", "StataSE", "xstata"):
        p = shutil.which(name)
        if p:
            hits.append(p)
    patterns: List[Tuple[Path, str]] = []
    if sys.platform == "darwin":
        patterns.append(
            (Path("/Applications"), "Stata*/Stata*.app/Contents/MacOS/stata*")
        )
    elif sys.platform.startswith("win"):
        for base in (
            os.environ.get("ProgramFiles"),
            os.environ.get("ProgramFiles(x86)"),
        ):
            if base:
                patterns.append((Path(base), "Stata*/Stata*.exe"))
    else:
        patterns.append((Path("/usr/local"), "stata*/stata*"))
    for base, pattern in patterns:
        try:
            hits.extend(str(p) for p in sorted(base.glob(pattern)) if p.is_file())
        except OSError:
            continue
    seen: List[str] = []
    for h in hits:
        if h not in seen:
            seen.append(h)
    return {"found": bool(seen), "paths": seen}


def _path_interpreters() -> List[str]:
    """Distinct interpreters named python / python3 / python3.x on PATH."""
    found: List[str] = []
    real_seen = set()
    for folder in os.environ.get("PATH", "").split(os.pathsep):
        if not folder:
            continue
        try:
            names = sorted(os.listdir(folder))
        except OSError:
            continue
        for name in names:
            if not _INTERPRETER_NAME.match(name):
                continue
            full = os.path.join(folder, name)
            if not (os.path.isfile(full) and os.access(full, os.X_OK)):
                continue
            real = os.path.realpath(full)
            if real in real_seen:
                continue
            real_seen.add(real)
            found.append(full)
    return found


def _probe(python: str, timeout: float) -> Dict[str, Any]:
    out, err, _ = _run([python, "-E", "-c", _PROBE_CODE], timeout)
    row: Dict[str, Any] = {"path": python}
    if err:
        row["error"] = err
        return row
    for line in reversed((out or "").splitlines()):
        try:
            parsed = json.loads(line)
        except ValueError:
            continue
        if isinstance(parsed, dict):
            row.update(parsed)
            return row
    tail = (out or "").strip().splitlines()
    row["error"] = tail[-1][:300] if tail else "no output"
    return row


def _probe_path(timeout: float) -> List[Dict[str, Any]]:
    candidates = _path_interpreters()
    # The same binary can serve several environments (a venv's python is a
    # symlink to its base interpreter), so "this environment" is decided by
    # sys.prefix, not by the executable's real path.
    here = os.path.realpath(sys.prefix)
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
        rows = list(pool.map(lambda p: _probe(p, timeout), candidates))
    for row in rows:
        prefix = row.get("prefix")
        row["is_current"] = bool(prefix) and os.path.realpath(prefix) == here
    return rows


# --------------------------------------------------------------------- #
#  Public API
# --------------------------------------------------------------------- #


def runtime_summary() -> Dict[str, Any]:
    """Version, interpreter and install of the running StatsPAI (cheap).

    The MCP server attaches this to its session instructions and to the
    ``describe_function`` / ``preflight`` replies, so a client can see
    which Python the server runs.
    """
    from . import __version__

    install = _install_record()
    location = str(Path(__file__).resolve().parent)
    return {
        "statspai_version": __version__,
        "python": os.path.abspath(sys.executable),
        "python_version": platform.python_version(),
        "statspai_location": location,
        "install_kind": _install_kind(install, location),
    }


def doctor(
    *,
    probe_path: bool = True,
    check_imports: bool = False,
    timeout: float = 5.0,
) -> Dict[str, Any]:
    """Report which StatsPAI this is, which Python runs it, and what else is there.

    Use it when ``import statspai`` fails in one terminal and works in
    another, when an MCP client seems to run an old version, or before
    comparing numbers with R / Stata on a new machine. The report is a
    JSON-ready dict; ``statspai doctor`` prints the same report as text
    (``--json`` for the dict) and ``python -m statspai --where`` prints
    only the first block.

    Parameters
    ----------
    probe_path : bool, default True
        Also look for other ``python`` / ``python3`` / ``python3.x``
        interpreters on ``PATH`` and ask each (in a child process, without
        this shell's ``PYTHONPATH``) which StatsPAI it has.
    check_imports : bool, default False
        Import every installed optional backend in a child process and
        report whether it succeeds. Off by default because torch, jax and
        pymc take seconds to import; without it a backend is reported as
        installed (found on ``sys.path``) with its metadata version, and
        ``imports`` is ``True`` only for those this process already
        imported.
    timeout : float, default 5.0
        Seconds allowed for each child process (interpreter probe, import
        check, ``Rscript --version``). A probe that times out is reported
        with an ``error`` field.

    Returns
    -------
    dict
        ``statspai`` (version, location, editable install, pip metadata,
        git revision), ``python`` (executable, version, implementation),
        ``environment`` (venv / conda / system, ``sys.prefix``),
        ``backends`` (one row per optional package), ``rust`` (the
        ``statspai_hdfe`` loader decision: version, ``api_version`` vs
        ``min_api_version``, ``STATSPAI_SKIP_RUST``), ``r`` and ``stata``
        (found, path, version), ``interpreters_on_path`` (one row per
        interpreter with its StatsPAI version), ``env`` (``STATSPAI_*``
        variables, secrets redacted), plus ``problems`` and ``notes``
        lists and ``ok`` (no problems).

    Examples
    --------
    >>> import statspai as sp
    >>> report = sp.doctor(probe_path=False)
    >>> report["statspai"]["version"] == sp.__version__
    True
    >>> "statspai_hdfe" in report["backends"]
    True
    """
    from . import __version__
    from ._build_info import version_info

    if (
        isinstance(timeout, bool)
        or not isinstance(timeout, (int, float))
        or timeout <= 0
    ):
        from .exceptions import MethodIncompatibility

        raise MethodIncompatibility(
            f"doctor: timeout must be a positive number of seconds, got {timeout!r}",
            recovery_hint="Pass e.g. timeout=5.0.",
            diagnostics={"timeout": repr(timeout)},
        )

    build = version_info()
    install = _install_record()
    problems: List[str] = []
    notes: List[str] = []

    statspai_block = {
        "version": __version__,
        "location": str(Path(__file__).resolve().parent),
        "editable": install.get("editable"),
        "install_kind": _install_kind(install, str(Path(__file__).resolve().parent)),
        "install_url": install.get("url"),
        "metadata_version": install.get("metadata_version"),
        "revision": build.get("revision"),
        "consistent": build.get("consistent"),
    }
    if build.get("warning"):
        problems.append(build["warning"])
    pythonpath = os.environ.get("PYTHONPATH")
    if pythonpath:
        notes.append(
            "PYTHONPATH is set; an interpreter started without it may import "
            "a different StatsPAI."
        )

    python_block = {
        "executable": os.path.abspath(sys.executable),
        "version": platform.python_version(),
        "implementation": platform.python_implementation(),
        "platform": platform.platform(),
    }

    rust = _rust_report()
    if rust.get("distribution_version") and not rust.get("available"):
        problems.append(
            f"statspai-hdfe {rust.get('distribution_version')} is installed but "
            f"not used: {rust.get('reason')}."
        )

    report: Dict[str, Any] = {
        "statspai": statspai_block,
        "python": python_block,
        "environment": _environment_kind(),
        "backends": _backend_rows(bool(check_imports), float(timeout)),
        "rust": rust,
        "r": _which_r(float(timeout)),
        "stata": _which_stata(),
        "env": _statspai_env(),
    }
    for name, row in report["backends"].items():
        if row.get("imports") is False:
            problems.append(
                f"{name} is installed but fails to import: "
                f"{row.get('import_error', 'unknown error')}"
            )

    if probe_path:
        rows = _probe_path(float(timeout))
        report["interpreters_on_path"] = rows
        others = [
            r for r in rows if not r.get("is_current") and r.get("statspai_version")
        ]
        stale = [r for r in others if r.get("statspai_version") != __version__]
        for r in stale:
            notes.append(
                f"{r['path']} (Python {r.get('python_version')}) has StatsPAI "
                f"{r['statspai_version']}, not {__version__}."
            )
        if rows and not any(r.get("is_current") for r in rows):
            notes.append(
                f"This interpreter ({python_block['executable']}) is not on PATH "
                "as python / python3; typing `python3` runs another one."
            )
    else:
        report["interpreters_on_path"] = None

    report["problems"] = problems
    report["notes"] = notes
    report["ok"] = not problems
    return report


def format_doctor(report: Dict[str, Any]) -> str:
    """Render a :func:`doctor` report as plain text."""
    s = report.get("statspai") or {}
    py = report.get("python") or {}
    env = report.get("environment") or {}
    install = s.get("install_kind")
    lines = [
        "StatsPAI doctor",
        f"  StatsPAI        {s.get('version')}  ({install})",
        f"  location        {s.get('location')}",
    ]
    if s.get("revision"):
        lines.append(f"  git revision    {s['revision']}")
    if s.get("metadata_version") and s.get("metadata_version") != s.get("version"):
        lines.append(
            f"  pip metadata    {s['metadata_version']} (differs from the code)"
        )
    lines += [
        f"  interpreter     {py.get('executable')}",
        f"  Python          {py.get('version')} ({py.get('implementation')}, "
        f"{py.get('platform')})",
        f"  environment     {env.get('kind')}  sys.prefix={env.get('prefix')}",
    ]
    rust = report.get("rust") or {}
    if rust.get("available"):
        rust_line = (
            f"statspai_hdfe {rust.get('version')} (API {rust.get('api_version')}, "
            f"needs >= {rust.get('min_api_version')}), {rust.get('threads')} threads"
        )
    else:
        rust_line = f"not used: {rust.get('reason')}"
    if rust.get("skip_rust_env"):
        rust_line += f"  [STATSPAI_SKIP_RUST={rust['skip_rust_env']}]"
    lines.append(f"  Rust kernels    {rust_line}")
    lines.append("  backends")
    for name, row in (report.get("backends") or {}).items():
        if not row.get("installed"):
            status = "not installed"
        else:
            status = row.get("version") or "installed (no metadata)"
            imp = row.get("imports")
            if imp is True:
                status += ", imports"
            elif imp is False:
                status += f", IMPORT FAILS: {row.get('import_error', '')}"
        lines.append(f"    {name:<14} {status}")
    r = report.get("r") or {}
    lines.append(
        "  Rscript         "
        + (f"{r.get('path')} ({r.get('version')})" if r.get("found") else "not found")
    )
    st = report.get("stata") or {}
    lines.append(
        "  Stata           "
        + (", ".join(st.get("paths") or []) if st.get("found") else "not found")
    )
    rows = report.get("interpreters_on_path")
    if rows is not None:
        lines.append(f"  interpreters on PATH ({len(rows)})")
        for row in rows:
            mark = "*" if row.get("is_current") else " "
            if row.get("error"):
                status = f"error: {row['error']}"
            else:
                if row.get("statspai_version"):
                    has = f"StatsPAI {row['statspai_version']}"
                elif row.get("statspai_location"):
                    has = "StatsPAI (no pip metadata)"
                else:
                    has = "no StatsPAI"
                status = f"Python {row.get('python_version')}, {has}"
            lines.append(f"   {mark} {row['path']:<44} {status}")
    for k, v in (report.get("env") or {}).items():
        lines.append(f"  env             {k}={v}")
    for p in report.get("problems") or []:
        lines.append(f"PROBLEM: {p}")
    for n in report.get("notes") or []:
        lines.append(f"note: {n}")
    lines.append("OK" if report.get("ok") else "Problems found.")
    return "\n".join(lines)


def where_text() -> str:
    """The short block ``python -m statspai --where`` prints."""
    info = runtime_summary()
    kind = _install_kind(_install_record(), info["statspai_location"])
    return "\n".join(
        [
            f"StatsPAI {info['statspai_version']}",
            f"python   {info['python']} (Python {info['python_version']})",
            f"package  {info['statspai_location']} ({kind})",
            f"prefix   {sys.prefix}",
        ]
    )
