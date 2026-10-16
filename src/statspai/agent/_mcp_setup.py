"""``statspai mcp doctor`` and ``statspai mcp config``.

The most common way an MCP setup goes wrong is not a bug in the server: the
client launches ``statspai-mcp`` by name, the name resolves on the client's
``PATH`` to a *different* Python environment than the one the user upgraded,
and the agent silently talks to an old StatsPAI. ``doctor`` finds that by
resolving every configured ``statspai`` server entry to the interpreter it
would run and asking that interpreter which StatsPAI it imports.
``config`` avoids it by writing the entry with the absolute path of the
current interpreter (``<python> -m statspai.agent.mcp_server``), so the
client runs exactly the environment the command was run from.

Nothing here talks to the network. ``doctor`` reads client configuration
files and starts the interpreters they name with ``-c "import statspai"``;
``config --write`` edits one JSON configuration file after copying it to a
timestamped backup.
"""

from __future__ import annotations

import importlib.util
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from ..exceptions import MethodIncompatibility

SERVER_KEY = "statspai"
MODULE = "statspai.agent.mcp_server"

#: Clients ``config`` knows how to describe.
CLIENTS = ("claude-code", "claude-desktop", "cursor", "windsurf", "vscode", "codex")

#: Optional dependencies worth reporting (import name -> what it enables).
OPTIONAL_DEPENDENCIES = {
    "numba": "fast kernels (forests, ETS, space-filling designs)",
    "pyarrow": "Parquet / Feather data files",
    "pyreadstat": "Stata .dta files with value labels",
    "openpyxl": "Excel data files",
    "matplotlib": "plot_from_result images",
    "pyfixest": "fixest backend",
    "torch": "neural / deep IV estimators",
    "jax": "GPU-accelerated estimators",
    "pymc": "PyMC-based Bayesian estimators",
}


# --------------------------------------------------------------------- #
#  Launch command
# --------------------------------------------------------------------- #


def launch_command(
    profile: Optional[str] = None, python: Optional[str] = None
) -> Tuple[str, List[str]]:
    """``(command, args)`` that start this environment's server."""
    args = ["-m", MODULE]
    if profile:
        args += ["--profile", profile]
    # Absolute, but symlinks kept: a venv's ``bin/python`` is a symlink to
    # the base interpreter, and resolving it would drop the venv.
    return (os.path.abspath(python or sys.executable), args)


def _env_block(data_roots: Sequence[str]) -> Dict[str, str]:
    if not data_roots:
        return {}
    roots = [str(Path(r).expanduser().resolve()) for r in data_roots]
    return {"STATSPAI_MCP_DATA_ROOTS": os.pathsep.join(roots)}


# --------------------------------------------------------------------- #
#  Client configuration locations
# --------------------------------------------------------------------- #


def _home() -> Path:
    return Path(os.environ.get("HOME") or Path.home())


def claude_desktop_config_path() -> Path:
    if sys.platform == "darwin":
        return (
            _home()
            / "Library"
            / "Application Support"
            / "Claude"
            / "claude_desktop_config.json"
        )
    if sys.platform.startswith("win"):
        base = Path(os.environ.get("APPDATA") or (_home() / "AppData" / "Roaming"))
        return base / "Claude" / "claude_desktop_config.json"
    return _home() / ".config" / "Claude" / "claude_desktop_config.json"


def config_path(client: str, cwd: Optional[Path] = None) -> Optional[Path]:
    """The file ``config --write`` edits for ``client`` (``None``: not a file)."""
    cwd = cwd or Path.cwd()
    if client == "claude-desktop":
        return claude_desktop_config_path()
    if client == "cursor":
        return _home() / ".cursor" / "mcp.json"
    if client == "windsurf":
        return _home() / ".codeium" / "windsurf" / "mcp_config.json"
    if client == "vscode":
        return cwd / ".vscode" / "mcp.json"
    return None


def _json_server_entry(
    client: str, command: str, args: List[str], env: Dict[str, str]
) -> Dict[str, Any]:
    entry: Dict[str, Any] = {"command": command, "args": args}
    if client == "vscode":
        entry = {"type": "stdio", **entry}
    if env:
        entry["env"] = env
    return entry


def _servers_key(client: str) -> str:
    return "servers" if client == "vscode" else "mcpServers"


def render_config(
    client: str,
    *,
    profile: Optional[str] = None,
    data_roots: Sequence[str] = (),
    python: Optional[str] = None,
) -> str:
    """Text the user pastes (or a command they run) to register the server."""
    if client not in CLIENTS:
        raise MethodIncompatibility(
            f"unknown client {client!r}",
            recovery_hint=f"Choose one of {list(CLIENTS)}.",
        )
    command, args = launch_command(profile, python)
    env = _env_block(data_roots)
    if client == "claude-code":
        env_flags = "".join(f" --env {k}={_shell_quote(v)}" for k, v in env.items())
        argv = " ".join(_shell_quote(a) for a in [command, *args])
        return (
            "# Register for every project (user scope); drop --scope for this "
            "project only.\n"
            f"claude mcp remove --scope user {SERVER_KEY} 2>/dev/null\n"
            f"claude mcp add --scope user{env_flags} {SERVER_KEY} -- {argv}\n"
        )
    if client == "codex":
        lines = [
            f"# ~/.codex/config.toml\n[mcp_servers.{SERVER_KEY}]",
            f"command = {json.dumps(command)}",
            f"args = {json.dumps(args)}",
        ]
        if env:
            inner = ", ".join(f"{k} = {json.dumps(v)}" for k, v in env.items())
            lines.append(f"env = {{ {inner} }}")
        return "\n".join(lines) + "\n"
    block = {
        _servers_key(client): {
            SERVER_KEY: _json_server_entry(client, command, args, env)
        }
    }
    path = config_path(client)
    return f"// {path}\n" + json.dumps(block, indent=2) + "\n"


def _shell_quote(text: str) -> str:
    if sys.platform.startswith("win"):
        return f'"{text}"' if any(c in text for c in ' "') else text
    import shlex

    return shlex.quote(text)


def write_config(
    client: str,
    *,
    profile: Optional[str] = None,
    data_roots: Sequence[str] = (),
    python: Optional[str] = None,
    path: Optional[Path] = None,
) -> Dict[str, Any]:
    """Merge the server entry into ``client``'s JSON config file.

    The existing file is copied to ``<name>.bak-<timestamp>`` first, every
    other server entry is kept, and a file that is not a JSON object is left
    untouched (``ValueError``).
    """
    target = path or config_path(client)
    if target is None:
        raise MethodIncompatibility(
            f"{client} is not configured through a JSON file this command edits.",
            recovery_hint=(
                f"Run `statspai mcp config --client {client}` and apply what "
                "it prints."
            ),
        )
    command, args = launch_command(profile, python)
    entry = _json_server_entry(client, command, args, _env_block(data_roots))
    existing: Dict[str, Any] = {}
    backup: Optional[Path] = None
    if target.exists():
        text = target.read_text(encoding="utf-8")
        if text.strip():
            loaded = json.loads(text)
            if not isinstance(loaded, dict):
                raise MethodIncompatibility(
                    f"{target} does not hold a JSON object; not edited.",
                    recovery_hint="Fix or move the file, then run --write again.",
                )
            existing = loaded
        backup = target.with_name(f"{target.name}.bak-{time.strftime('%Y%m%d%H%M%S')}")
        shutil.copy2(target, backup)
    key = _servers_key(client)
    servers = existing.get(key)
    if servers is not None and not isinstance(servers, dict):
        raise MethodIncompatibility(
            f"{target}: `{key}` is not an object; not edited.",
            recovery_hint=f"Make `{key}` a JSON object, then run --write again.",
        )
    previous = (servers or {}).get(SERVER_KEY)
    existing[key] = {**(servers or {}), SERVER_KEY: entry}
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(json.dumps(existing, indent=2) + "\n", encoding="utf-8")
    return {
        "path": str(target),
        "backup": str(backup) if backup else None,
        "entry": entry,
        "replaced": previous,
    }


# --------------------------------------------------------------------- #
#  doctor
# --------------------------------------------------------------------- #


def _probe_interpreter(
    python: str, timeout: float = 60.0, env: Optional[Dict[str, Any]] = None
) -> Dict[str, Any]:
    """Which StatsPAI does ``python`` import? (version, file) or an error.

    The probe runs the way a client would launch the server: without this
    shell's ``PYTHONPATH`` (a developer checkout on it would make every
    interpreter look up to date), with the entry's own ``env`` block on top,
    and without the current directory on ``sys.path``.

    The child is started with ``close_fds=False`` and no ``cwd`` so that
    :mod:`subprocess` can use ``posix_spawn``: a ``fork`` taken after
    numerical code ran in this process can crash in the child before
    ``exec`` on macOS (the self-test fits a model).
    """
    if not (os.path.exists(python) or shutil.which(python)):
        return {"error": f"FileNotFoundError: no interpreter at {python!r}"}
    child_env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    child_env.update({str(k): str(v) for k, v in (env or {}).items()})
    code = (
        "import json, os, sys\n"
        "sys.path[:] = [p for p in sys.path if p not in ('', os.getcwd())]\n"
        "try:\n"
        "    import statspai\n"
        "    print(json.dumps({'version': statspai.__version__, "
        "'file': statspai.__file__, 'python': sys.version.split()[0]}))\n"
        "except Exception as exc:\n"
        "    print(json.dumps({'error': type(exc).__name__ + ': ' + str(exc), "
        "'python': sys.version.split()[0]}))\n"
    )
    try:
        proc = subprocess.run(  # noqa: S603 - the interpreter a config names
            [python, "-c", code],
            capture_output=True,
            text=True,
            timeout=timeout,
            env=child_env,
            close_fds=False,
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"error": f"{type(exc).__name__}: {exc}"}
    for line in reversed(proc.stdout.splitlines()):
        try:
            out = json.loads(line)
        except ValueError:
            continue
        if isinstance(out, dict):
            return out
    last = (proc.stderr.strip().splitlines() or ["no output"])[-1]
    return {"error": f"exit status {proc.returncode}: {last}"}


def _script_interpreter(script: str) -> Optional[str]:
    """Interpreter in a console script's shebang (``None`` if not readable)."""
    try:
        with open(script, "rb") as fh:
            first = fh.readline(4096).decode("utf-8", "replace").strip()
    except OSError:
        return None
    if not first.startswith("#!"):
        return None
    parts = first[2:].strip().split()
    if not parts:
        return None
    if Path(parts[0]).name == "env" and len(parts) > 1:
        return shutil.which(parts[1])
    return parts[0]


def resolve_entry_interpreter(entry: Dict[str, Any]) -> Tuple[Optional[str], str]:
    """``(python, how)`` that a client's server entry would run."""
    command = entry.get("command")
    args = entry.get("args") or []
    if not isinstance(command, str) or not command:
        return None, "no command"
    name = Path(command).name.lower()
    if name.startswith("python") or name in {"py", "py.exe"}:
        return (shutil.which(command) or command), "python command"
    if name in {"uv", "uvx", "pipx"}:
        return None, f"launched through {name} (isolated environment; not probed)"
    script = command if os.path.isabs(command) else shutil.which(command)
    if script is None:
        return None, f"{command!r} is not on this shell's PATH"
    if name.startswith("statspai"):
        python = _script_interpreter(script)
        if python:
            return python, f"shebang of {script}"
        return None, f"cannot read the interpreter of {script}"
    if args and str(args[0]).endswith(".py"):
        return None, "custom script"
    return None, f"unrecognised command {command!r}"


def _load_json(path: Path) -> Optional[Dict[str, Any]]:
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return data if isinstance(data, dict) else None


def configured_entries(cwd: Optional[Path] = None) -> List[Dict[str, Any]]:
    """Every ``statspai`` server entry in the known client config files."""
    cwd = cwd or Path.cwd()
    found: List[Dict[str, Any]] = []
    seen: set = set()

    def take(client: str, path: Path, servers: Any, scope: str = "") -> None:
        if not isinstance(servers, dict):
            return
        for name, entry in servers.items():
            if not isinstance(entry, dict):
                continue
            # The project file is the user file when cwd is the home dir.
            key = (
                client,
                str(path.resolve()),
                name,
                scope if client == "claude-code" else "",
            )
            if key in seen:
                continue
            seen.add(key)
            command = str(entry.get("command", ""))
            args = " ".join(str(a) for a in entry.get("args") or [])
            if name.lower() == SERVER_KEY or "statspai" in f"{command} {args}":
                found.append(
                    {
                        "client": client,
                        "scope": scope,
                        "file": str(path),
                        "name": name,
                        "entry": entry,
                    }
                )

    claude_json = _home() / ".claude.json"
    data = _load_json(claude_json)
    if data:
        take("claude-code", claude_json, data.get("mcpServers"), "user")
        projects = data.get("projects")
        if isinstance(projects, dict):
            proj = projects.get(str(cwd))
            if isinstance(proj, dict):
                take("claude-code", claude_json, proj.get("mcpServers"), "local")
    for client, path, scope in [
        ("claude-code", cwd / ".mcp.json", "project"),
        ("claude-desktop", claude_desktop_config_path(), ""),
        ("cursor", _home() / ".cursor" / "mcp.json", "user"),
        ("cursor", cwd / ".cursor" / "mcp.json", "project"),
        ("windsurf", _home() / ".codeium" / "windsurf" / "mcp_config.json", ""),
        ("vscode", cwd / ".vscode" / "mcp.json", "workspace"),
    ]:
        data = _load_json(path)
        if data:
            take(client, path, data.get(_servers_key(client)), scope)
    codex = _home() / ".codex" / "config.toml"
    if codex.exists():
        try:
            import tomllib  # type: ignore[import-not-found]
        except ImportError:  # Python 3.10
            tomllib = None  # type: ignore[assignment]
        if tomllib is not None:
            try:
                parsed = tomllib.loads(codex.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                parsed = {}
            take("codex", codex, parsed.get("mcp_servers"), "user")
    return found


def _self_test() -> Dict[str, Any]:
    """In-process round trip: handshake, per-request probe, list, one call."""
    from . import _protocol
    from . import mcp_server as srv

    def rpc(method: str, params: Dict[str, Any]) -> Dict[str, Any]:
        line = json.dumps(
            {"jsonrpc": "2.0", "id": 1, "method": method, "params": params}
        )
        return json.loads(srv.handle_request(line) or "{}")

    out: Dict[str, Any] = {}
    t0 = time.perf_counter()
    init = rpc("initialize", {"protocolVersion": srv.MCP_PROTOCOL_VERSION})
    out["initialize_ok"] = "result" in init
    t1 = time.perf_counter()
    listing = rpc("tools/list", {})
    tools = (listing.get("result") or {}).get("tools") or []
    out["tools"] = len(tools)
    out["tools_list_bytes"] = len(srv._tools_list_result_json())
    out["tools_list_seconds"] = round(time.perf_counter() - t1, 3)
    meta = {
        _protocol.META_PROTOCOL_VERSION: _protocol.MODERN_PROTOCOL_VERSIONS[0],
        _protocol.META_CLIENT_CAPABILITIES: {},
    }
    discover = rpc("server/discover", {"_meta": meta})
    out["discover_ok"] = "result" in discover
    t2 = time.perf_counter()
    csv = "y,x\n" + "".join(f"{1 + 2 * i + (i % 3)},{i}\n" for i in range(30))
    call = rpc(
        "tools/call",
        {
            "name": "regress",
            "arguments": {"formula": "y ~ x", "data_csv": csv, "detail": "minimal"},
        },
    )
    result = call.get("result") or {}
    out["call_ok"] = bool(result) and not result.get("isError")
    out["call_seconds"] = round(time.perf_counter() - t2, 3)
    out["total_seconds"] = round(time.perf_counter() - t0, 3)
    if not out["call_ok"]:
        text = "".join(
            c.get("text", "") for c in result.get("content", []) if isinstance(c, dict)
        )
        out["call_error"] = text[:500] or json.dumps(call)[:500]
    return out


def run_doctor(*, self_test: bool = True, cwd: Optional[Path] = None) -> Dict[str, Any]:
    """Collect the doctor report (see :func:`format_doctor`)."""
    import statspai

    report: Dict[str, Any] = {
        "python": os.path.abspath(sys.executable),
        "python_version": platform.python_version(),
        "platform": platform.platform(),
        "statspai_version": statspai.__version__,
        "statspai_file": statspai.__file__,
        "problems": [],
        "notes": [],
    }
    problems: List[str] = report["problems"]
    notes: List[str] = report["notes"]
    here = statspai.__version__

    on_path = shutil.which("statspai-mcp")
    report["statspai_mcp_on_path"] = on_path
    if on_path:
        py = _script_interpreter(on_path)
        probe = _probe_interpreter(py) if py else {"error": "unreadable shebang"}
        report["statspai_mcp_on_path_probe"] = {"python": py, **probe}
        if probe.get("version") and probe["version"] != here:
            problems.append(
                f"`statspai-mcp` on PATH ({on_path}) runs StatsPAI "
                f"{probe['version']} from {py}, not {here} from this "
                "interpreter. A client that launches it by name gets the "
                "old version."
            )
    else:
        notes.append("`statspai-mcp` is not on this shell's PATH.")

    entries = configured_entries(cwd)
    for item in entries:
        python, how = resolve_entry_interpreter(item["entry"])
        item["resolved_python"] = python
        item["resolved_by"] = how
        if python:
            entry_env = item["entry"].get("env")
            probe = _probe_interpreter(
                python, env=entry_env if isinstance(entry_env, dict) else None
            )
            item["probe"] = probe
            scope = ", " + item["scope"] if item["scope"] else ""
            where = f"{item['client']} ({item['file']}{scope})"
            if "error" in probe:
                problems.append(
                    f"{where}: the entry runs {python}, which cannot import "
                    f"StatsPAI ({probe['error']})."
                )
            elif probe.get("version") != here:
                problems.append(
                    f"{where}: the entry runs StatsPAI {probe.get('version')} "
                    f"from {python}; this interpreter has {here}."
                )
        else:
            notes.append(f"{item['client']} entry in {item['file']}: {how}.")
    report["client_entries"] = entries
    if not entries:
        notes.append("No StatsPAI entry found in the known MCP client configs.")
    if problems:
        notes.append(
            "Fix: `statspai mcp config --client <client>` prints an entry that "
            "uses this interpreter by absolute path; add --write to apply it "
            "(JSON-config clients, with a backup)."
        )

    report["optional_dependencies"] = {
        name: importlib.util.find_spec(name) is not None
        for name in OPTIONAL_DEPENDENCIES
    }
    report["environment"] = {
        k: v for k, v in sorted(os.environ.items()) if k.startswith("STATSPAI_MCP_")
    }
    roots = os.environ.get("STATSPAI_MCP_DATA_ROOTS")
    if roots:
        for root in roots.split(os.pathsep):
            if root and not Path(root).expanduser().is_dir():
                problems.append(
                    f"STATSPAI_MCP_DATA_ROOTS entry {root!r} is not a directory."
                )
    if self_test:
        # handle_request turns any failure into a JSON-RPC error, which the
        # self-test records; nothing here needs a broad catch.
        report["self_test"] = _self_test()
        st = report["self_test"]
        if st.get("error") or not st.get("call_ok") or not st.get("discover_ok"):
            problems.append(f"Self-test failed: {json.dumps(st)[:400]}")
    report["ok"] = not problems
    return report


def format_doctor(report: Dict[str, Any]) -> str:
    lines = [
        "StatsPAI MCP doctor",
        f"  interpreter     {report['python']} (Python {report['python_version']})",
        f"  StatsPAI        {report['statspai_version']}  {report['statspai_file']}",
    ]
    probe = report.get("statspai_mcp_on_path_probe") or {}
    if report.get("statspai_mcp_on_path"):
        lines.append(
            f"  statspai-mcp    {report['statspai_mcp_on_path']} -> "
            f"{probe.get('version') or probe.get('error')}"
        )
    entries = report.get("client_entries") or []
    lines.append(f"  client entries  {len(entries)}")
    for item in entries:
        p = item.get("probe") or {}
        status = p.get("version") or p.get("error") or item.get("resolved_by")
        scope = f" [{item['scope']}]" if item.get("scope") else ""
        lines.append(f"    - {item['client']}{scope}: {item['file']} -> {status}")
    deps = report.get("optional_dependencies") or {}
    missing = [n for n, ok in deps.items() if not ok]
    lines.append(
        "  optional deps   "
        + (", ".join(n for n, ok in deps.items() if ok) or "none")
        + (f"   (missing: {', '.join(missing)})" if missing else "")
    )
    st = report.get("self_test")
    if st:
        if st.get("error"):
            lines.append(f"  self-test       ERROR {st['error']}")
        else:
            lines.append(
                f"  self-test       {st.get('tools')} tools, tools/list "
                f"{st.get('tools_list_seconds')} s ({st.get('tools_list_bytes')} B), "
                f"regress call {st.get('call_seconds')} s, "
                f"2026-07-28 probe {'ok' if st.get('discover_ok') else 'FAILED'}"
            )
    for p in report.get("problems") or []:
        lines.append(f"PROBLEM: {p}")
    for n in report.get("notes") or []:
        lines.append(f"note: {n}")
    lines.append("OK" if report.get("ok") else "Problems found.")
    return "\n".join(lines)
