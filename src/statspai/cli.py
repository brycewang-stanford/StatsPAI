"""Command-line interface for StatsPAI.

Install with ``pip install statspai`` and the ``statspai`` console
script becomes available (see ``[project.scripts]`` in pyproject.toml).

Commands
--------
    statspai list [--category CAT]
    statspai describe <name>
    statspai search <query>
    statspai help [<name>]
    statspai run <function> --data FILE [--arg k=v ...] [--format json|summary]
    statspai did|regress|ivreg|rdrobust|callaway_santanna|... --data FILE --y ...
    statspai route <family> [--answer key=value ...]
    statspai mcp [--profile core|curated|full]
    statspai version

Discovery commands delegate to sp.help / sp.list_functions /
sp.describe_function / sp.search_functions. ``run`` and the family
shortcuts go through the same dispatch layer as the MCP server
(``statspai.agent.tools.execute_tool``): the data file is loaded
server-side, arguments the function cannot bind are reported under
``_unsupported_args``, and the result is the agent payload
(``to_dict(detail=...)``) as JSON on stdout — so a shell agent gets
exactly what an MCP client gets.

Output contract (``run`` / family shortcuts):

* stdout carries exactly one strict-JSON document (RFC 8259: no ``NaN`` /
  ``Infinity``; numpy / pandas values are converted, never ``repr``-ed).
  Anything an estimator ``print``s is redirected to stderr.
* Python warnings raised during the call are attached as
  ``runtime_warnings: [{category, message}]``.
* Errors are a structured JSON object on stderr (``error``,
  ``error_kind``, ``remediation``, and ``error_payload`` for
  ``StatsPAIError``) with an exit code chosen by kind:

  ====  ==============================================================
  2     usage error (bad CLI flags, unknown function, data file unreadable)
  3     estimator error (any other failure inside the estimator)
  4     input error: ``column_not_found`` / ``missing_arguments`` /
        ``unknown_argument`` (fix the call and retry)
  5     ``missing_dependency`` (install the extra named in ``install``)
  ====  ==============================================================
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any, Dict, List, Optional, Sequence

#: Family shortcuts: ``statspai did ...`` builds its flags from the
#: registry schema of the dispatcher, so the CLI never carries a
#: hand-written flag table that could drift from the signature.
SHORTCUTS = (
    "did",
    "callaway_santanna",
    "event_study",
    "regress",
    "feols",
    "ivreg",
    "rdrobust",
    "synth",
    "dml",
    "match",
    "ipw",
    "aipw",
)

EXIT_USAGE = 2
EXIT_ESTIMATOR = 3
EXIT_INPUT = 4
EXIT_DEPENDENCY = 5

#: ``error_kind`` -> exit code. Kinds not listed exit with EXIT_ESTIMATOR.
KIND_EXIT_CODES: Dict[str, int] = {
    "column_not_found": EXIT_INPUT,
    "missing_arguments": EXIT_INPUT,
    "unknown_argument": EXIT_INPUT,
    "missing_dependency": EXIT_DEPENDENCY,
}


def _parse_value(raw: str) -> Any:
    """``k=v`` values: JSON when it parses (numbers, booleans, lists,
    objects, quoted strings), else the bare string."""
    try:
        return json.loads(raw)
    except (TypeError, ValueError):
        return raw


def _parse_kv(items: Optional[Sequence[str]], *, flag: str) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    for item in items or ():
        if "=" not in item:
            raise SystemExit(f"{flag} expects key=value, got {item!r}")
        key, raw = item.split("=", 1)
        out[key.strip()] = _parse_value(raw)
    return out


def _add_schema_args(parser: argparse.ArgumentParser, name: str) -> List[str]:
    """Add ``--<param>`` flags from the registry schema of ``name``.

    Returns the parameter names added. ``data`` is skipped (it comes from
    ``--data``); types follow the JSON schema so ``--covariates`` takes a
    JSON list and ``--alpha`` a number.
    """
    import statspai as sp

    schema = sp.function_schema(name)
    props = schema["parameters"]["properties"]
    required = set(schema["parameters"].get("required", []))
    added: List[str] = []
    for pname, prop in props.items():
        if pname == "data":
            continue
        typ = prop.get("type")
        types = typ if isinstance(typ, list) else [typ]
        desc = prop.get("description", "")
        kwargs: Dict[str, Any] = {"help": desc, "default": argparse.SUPPRESS}
        if types == ["boolean"]:
            kwargs["type"] = lambda v: (
                _parse_value(v) if v not in ("true", "false") else v == "true"
            )
            kwargs["metavar"] = "true|false"
        elif types == ["integer"]:
            kwargs["type"] = int
        elif types == ["number"]:
            kwargs["type"] = float
        elif types == ["string"] and prop.get("enum"):
            kwargs["choices"] = [str(e) for e in prop["enum"]]
        elif "array" in types or "object" in types or len(types) > 1:
            kwargs["type"] = _parse_value
            kwargs["metavar"] = "JSON"
        if pname in required and pname != "data":
            kwargs["required"] = True
        parser.add_argument(f"--{pname}", dest=pname, **kwargs)
        added.append(pname)
    return added


def _add_run_common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--data",
        "-d",
        default=None,
        help="Data file (CSV / Parquet / Feather / Stata .dta / Excel / JSON) or URL.",
    )
    parser.add_argument(
        "--columns",
        default=None,
        type=_parse_value,
        metavar="JSON",
        help='Optional column projection, e.g. \'["y","d","id"]\'.',
    )
    parser.add_argument(
        "--sample",
        default=None,
        type=int,
        help="Optional deterministic row subsample size for huge files.",
    )
    parser.add_argument(
        "--format",
        "-f",
        default="json",
        choices=["json", "summary"],
        help="json (default; the agent payload) or summary (the result's text summary).",
    )
    parser.add_argument(
        "--detail",
        default="agent",
        choices=["minimal", "standard", "agent"],
        help="Payload depth for --format json.",
    )
    parser.add_argument(
        "--out",
        "-o",
        default=None,
        help="Write the JSON payload to this file as well as stdout.",
    )
    parser.add_argument(
        "--indent",
        default=2,
        type=int,
        help="JSON indent (0 for compact).",
    )


def _run_function(
    name: str,
    arguments: Dict[str, Any],
    *,
    data: Optional[str],
    columns: Optional[List[str]],
    sample: Optional[int],
    fmt: str,
    detail: str,
    out_path: Optional[str],
    indent: int,
) -> int:
    """Shared body of ``run`` and the family shortcuts."""
    from .agent.tools import execute_tool

    df = None
    provenance = None
    if data:
        from .agent._data_loader import data_provenance, load_dataframe

        path = data if "://" in data else os.path.abspath(data)
        try:
            df = load_dataframe(path, columns=columns, sample_n=sample)
            provenance = data_provenance(path, columns=columns, sample_n=sample)
        except Exception as exc:  # loader errors are usage errors
            _emit_error(
                {
                    "error": f"{type(exc).__name__}: {exc}",
                    "error_kind": "data_load",
                    "data": data,
                }
            )
            return EXIT_USAGE

    want_summary = fmt == "summary"
    payload, runtime_warnings = _call_capturing(
        execute_tool,
        name,
        dict(arguments),
        data=df,
        detail=detail,
        as_handle=want_summary,
    )
    if not isinstance(payload, dict):
        payload = {"value": payload}
    if provenance is not None:
        payload.setdefault("data_provenance", provenance)
    if runtime_warnings:
        payload["runtime_warnings"] = runtime_warnings

    if payload.get("error"):
        kind = _error_kind(payload)
        if kind:
            payload["error_kind"] = kind
        _emit_error(payload)
        return KIND_EXIT_CODES.get(kind or "", EXIT_ESTIMATOR)

    if want_summary:
        from .agent._result_cache import RESULT_CACHE

        obj = (
            RESULT_CACHE.get(payload.get("result_id", ""))
            if payload.get("result_id")
            else None
        )
        summary = getattr(obj, "summary", None)
        if callable(summary):
            text = summary()
            print(text if isinstance(text, str) else str(text))
        else:
            print(_dumps(payload, indent=indent))
        payload.pop("result_id", None)
        payload.pop("result_uri", None)
    else:
        print(_dumps(payload, indent=indent))
    if out_path:
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write(_dumps(payload, indent=indent))
    return 0


def _call_capturing(fn: Any, *args: Any, **kwargs: Any) -> Any:
    """Run ``fn`` with stdout -> stderr and Python warnings recorded.

    Returns ``(result, runtime_warnings)`` where ``runtime_warnings`` is a
    de-duplicated list of ``{"category", "message"}`` (at most 20), the
    same shape the MCP server attaches. Redirecting stdout keeps the CLI's
    stdout a single JSON document even when an estimator prints.
    """
    import contextlib
    import warnings

    with contextlib.redirect_stdout(sys.stderr):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            result = fn(*args, **kwargs)
    recorded: List[Dict[str, str]] = []
    seen: set = set()
    for w in caught:
        key = (w.category.__name__, str(w.message))
        if key in seen:
            continue
        seen.add(key)
        recorded.append({"category": key[0], "message": key[1]})
        if len(recorded) >= 20:
            break
    return result, recorded


def _error_kind(payload: Dict[str, Any]) -> Optional[str]:
    """Pick the error kind that decides the exit code.

    A remediation category that is mechanically actionable (column typo,
    missing / unknown argument, missing dependency) wins over the raiser's
    broader taxonomy kind, so e.g. a ``MethodIncompatibility`` carrying
    ``missing_columns`` diagnostics still exits as ``column_not_found``.
    """
    rem = payload.get("remediation")
    category = rem.get("category") if isinstance(rem, dict) else None
    if isinstance(category, str) and category in KIND_EXIT_CODES:
        return category
    kind = payload.get("error_kind")
    if isinstance(kind, str) and kind:
        return kind
    return category if isinstance(category, str) else None


def _local_json_default(o: Any) -> Any:
    """Fallback encoder used only if the MCP server's encoder is unavailable."""
    import math

    try:
        import numpy as np

        if isinstance(o, np.generic):
            o = o.item()
        elif isinstance(o, np.ndarray):
            return _local_clean(o.tolist())
    except ImportError:  # pragma: no cover - numpy is a core dependency
        pass
    try:
        import pandas as pd

        if isinstance(o, pd.DataFrame):
            return _local_clean(o.to_dict(orient="list"))
        if isinstance(o, (pd.Series, pd.Index)):
            return _local_clean(o.tolist())
        if isinstance(o, (pd.Timestamp, pd.Timedelta)):
            return str(o)
    except ImportError:  # pragma: no cover - pandas is a core dependency
        pass
    if isinstance(o, float):
        return None if (math.isnan(o) or math.isinf(o)) else o
    if isinstance(o, (int, str, bool)) or o is None:
        return o
    if isinstance(o, (set, frozenset)):
        return _local_clean(sorted(o, key=str))
    return str(o)


def _local_clean(obj: Any) -> Any:
    import math

    if isinstance(obj, float):
        return None if (math.isnan(obj) or math.isinf(obj)) else obj
    if isinstance(obj, dict):
        return {k: _local_clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_local_clean(v) for v in obj]
    return obj


def _json_helpers() -> Any:
    """``(clean, default)`` — the MCP server's encoders, else local copies."""
    try:
        from .agent.mcp_server import _clean_floats, _json_default
    except ImportError:  # pragma: no cover - exercised only on refactor drift
        return _local_clean, _local_json_default
    return _clean_floats, _json_default


def _dumps(obj: Any, *, indent: Optional[int] = 2) -> str:
    """Strict JSON (``allow_nan=False``) with numpy / pandas support."""
    clean, default = _json_helpers()

    def _default(o: Any) -> Any:
        # Clean what the encoder returns: a container produced from a
        # numpy array / DataFrame may itself hold NaN.
        return clean(default(o))

    return json.dumps(
        clean(obj), indent=indent or None, default=_default, allow_nan=False
    )


def _clean(obj: Any) -> Any:
    return _json_helpers()[0](obj)


def _emit_error(payload: Dict[str, Any]) -> None:
    print(_dumps(payload, indent=2), file=sys.stderr)


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="statspai",
        description=(
            "StatsPAI CLI — discover functions, read help, search the API. "
            "All output mirrors the Python-level sp.help() / sp.list_functions()."
        ),
    )
    parser.add_argument(
        "--version", "-V", action="store_true", help="Print StatsPAI version and exit."
    )

    sub = parser.add_subparsers(dest="command", metavar="<command>")

    # list
    p_list = sub.add_parser("list", help="List registered functions.")
    p_list.add_argument(
        "--category",
        "-c",
        default=None,
        help="Filter by category (e.g. causal, panel, spatial).",
    )
    p_list.add_argument(
        "--stability",
        "-s",
        default=None,
        choices=["stable", "experimental", "deprecated"],
        help=(
            "Filter by API lifecycle tier. 'stable' = public signature "
            "locked; 'experimental' = API/method may shift; "
            "'deprecated' = scheduled for removal."
        ),
    )
    p_list.add_argument(
        "--validation",
        default=None,
        dest="validation_status",
        choices=["certified", "validated", "api_stable", "experimental", "deprecated"],
        help=(
            "Filter by numerical evidence tier. Use 'certified' for "
            "cross-language or published-reference parity evidence."
        ),
    )
    p_list.add_argument(
        "--json", action="store_true", help="Emit JSON array instead of text."
    )

    # describe
    p_desc = sub.add_parser("describe", help="Show full metadata for a function.")
    p_desc.add_argument("name", help="Function name, e.g. 'did'.")
    p_desc.add_argument(
        "--json", action="store_true", help="Emit JSON object instead of text."
    )

    # search
    p_search = sub.add_parser("search", help="Keyword search across function metadata.")
    p_search.add_argument("query", nargs="+", help="One or more keywords.")
    p_search.add_argument(
        "--json", action="store_true", help="Emit JSON array instead of text."
    )

    # help
    p_help = sub.add_parser("help", help="Show help overview, or details for a topic.")
    p_help.add_argument(
        "topic",
        nargs="?",
        default=None,
        help="Function name, category, or 'category.name' path.",
    )
    p_help.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Append full docstring after registry metadata.",
    )

    # run
    p_run = sub.add_parser(
        "run",
        help="Run any registered function on a data file and print the agent payload.",
    )
    p_run.add_argument(
        "function", help="Registered function name, e.g. callaway_santanna."
    )
    p_run.add_argument(
        "--arg",
        "-a",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Keyword argument; VALUE is parsed as JSON when possible (repeatable).",
    )
    _add_run_common(p_run)

    # family shortcuts (flags generated from the registry schema)
    for name in SHORTCUTS:
        p_fn = sub.add_parser(
            name, help=f"Run sp.{name} (flags from its registry schema)."
        )
        _add_run_common(p_fn)
        try:
            _add_schema_args(p_fn, name)
        except Exception as exc:  # pragma: no cover - registry drift
            p_fn.description = f"(schema unavailable: {exc})"

    # route
    p_route = sub.add_parser(
        "route",
        help="Route a research question to estimator calls (no data needed).",
    )
    p_route.add_argument(
        "family",
        nargs="?",
        default=None,
        help="did | iv | rd | matching | ml_causal | qte | dynamic_panel (omit to list).",
    )
    p_route.add_argument(
        "--answer",
        "-a",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="Answer to one of the family's questions (repeatable).",
    )
    p_route.add_argument(
        "--json", action="store_true", help="Emit JSON (default when piped)."
    )

    # mcp
    p_mcp = sub.add_parser("mcp", help="Start the MCP stdio server.")
    p_mcp.add_argument(
        "--profile",
        default=None,
        choices=["core", "curated", "full"],
        help="tools/list profile (default: curated, or STATSPAI_MCP_PROFILE).",
    )

    # skill
    p_skill = sub.add_parser(
        "skill",
        help="Manage the packaged Claude Code skill (statspai-analysis).",
    )
    skill_sub = p_skill.add_subparsers(dest="skill_command", metavar="<action>")
    p_install = skill_sub.add_parser(
        "install",
        help="Copy the skill into a skills directory (default ~/.claude/skills).",
    )
    p_install.add_argument(
        "--target",
        default=None,
        help="Skills directory; the skill lands in <target>/statspai-analysis/.",
    )
    p_install.add_argument(
        "--force", action="store_true", help="Overwrite an existing installation."
    )
    skill_sub.add_parser("path", help="Print the packaged skill's directory.")
    p_validate = skill_sub.add_parser(
        "validate",
        help="Run the skill's API-claim gate against the installed StatsPAI.",
    )
    p_validate.add_argument(
        "--quick",
        action="store_true",
        help="Existence + signature checks only (no fits).",
    )

    # version
    sub.add_parser("version", help="Print StatsPAI version and exit.")

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = _make_parser()
    args = parser.parse_args(argv)

    import statspai as sp

    if args.version or args.command == "version":
        print(sp.__version__)
        return 0

    if args.command is None:
        # Default: top-level overview
        print(sp.help())
        return 0

    if args.command == "list":
        names = sp.list_functions(
            category=args.category,
            stability=args.stability,
            validation_status=args.validation_status,
        )
        if args.json:
            print(json.dumps(names))
            return 0
        if not names:
            filt = []
            if args.category:
                filt.append(f"category={args.category!r}")
            if args.stability:
                filt.append(f"stability={args.stability!r}")
            if args.validation_status:
                filt.append(f"validation_status={args.validation_status!r}")
            tag = ", ".join(filt) if filt else "(no filter)"
            print(f"(no functions matching {tag})")
            return 0
        for n in sorted(names):
            print(n)
        return 0

    if args.command == "describe":
        try:
            spec = sp.describe_function(args.name)
        except KeyError as e:
            print(str(e), file=sys.stderr)
            return 2
        if args.json:
            print(_dumps(spec, indent=2))
            return 0
        print(sp.help(args.name, verbose=True))
        return 0

    if args.command == "search":
        q = " ".join(args.query)
        if args.json:
            print(_dumps(sp.search_functions(q), indent=2))
            return 0
        print(sp.help(search=q))
        return 0

    if args.command == "help":
        if args.topic is None:
            print(sp.help())
        else:
            print(sp.help(args.topic, verbose=args.verbose))
        return 0

    if args.command == "run":
        try:
            arguments = _parse_kv(args.arg, flag="--arg")
        except SystemExit as exc:
            print(str(exc), file=sys.stderr)
            return EXIT_USAGE
        if args.function not in sp.list_functions() and not callable(
            getattr(sp, args.function, None)
        ):
            import difflib

            close = difflib.get_close_matches(
                args.function, sp.list_functions(), n=5, cutoff=0.6
            )
            hits = close + [
                h["name"]
                for h in sp.search_functions(args.function)[:5]
                if h["name"] not in close
            ]
            _emit_error(
                {
                    "error": f"Unknown function {args.function!r}",
                    "did_you_mean": hits,
                    "hint": "statspai search <keywords> lists candidates.",
                }
            )
            return EXIT_USAGE
        return _run_function(
            args.function,
            arguments,
            data=args.data,
            columns=args.columns,
            sample=args.sample,
            fmt=args.format,
            detail=args.detail,
            out_path=args.out,
            indent=args.indent,
        )

    if args.command in SHORTCUTS:
        common = {
            "data",
            "columns",
            "sample",
            "format",
            "detail",
            "out",
            "indent",
            "command",
            "version",
        }
        arguments = {
            k: v for k, v in vars(args).items() if k not in common and v is not None
        }
        return _run_function(
            args.command,
            arguments,
            data=args.data,
            columns=args.columns,
            sample=args.sample,
            fmt=args.format,
            detail=args.detail,
            out_path=args.out,
            indent=args.indent,
        )

    if args.command == "route":
        try:
            answers = _parse_kv(args.answer, flag="--answer")
        except SystemExit as exc:
            print(str(exc), file=sys.stderr)
            return EXIT_USAGE
        try:
            if args.family is None:
                payload: Dict[str, Any] = sp.decision_guide()
            elif answers:
                payload = sp.route(
                    args.family, **{k: str(v) for k, v in answers.items()}
                )
            else:
                payload = sp.decision_guide(args.family)
        except Exception as exc:  # StatsPAIError -> structured
            to_dict = getattr(exc, "to_dict", None)
            _emit_error(
                to_dict()
                if callable(to_dict)
                else {"error": f"{type(exc).__name__}: {exc}"}
            )
            return EXIT_USAGE
        if args.json or not sys.stdout.isatty():
            print(_dumps(payload, indent=2))
            return 0
        _print_route(payload)
        return 0

    if args.command == "skill":
        return _skill_command(args)

    if args.command == "mcp":
        from .agent.mcp_server import main as mcp_main

        mcp_main(["--profile", args.profile] if args.profile else [])
        return 0

    parser.print_help()
    return 1


SKILL_NAME = "statspai-analysis"


def _packaged_skill_dir():
    from pathlib import Path

    return Path(__file__).resolve().parent / "agent" / "_skill"


def _skill_command(args: argparse.Namespace) -> int:
    import shutil
    from pathlib import Path

    src = _packaged_skill_dir()
    if not (src / "SKILL.md").exists():
        print(f"packaged skill not found at {src}", file=sys.stderr)
        return EXIT_USAGE
    action = args.skill_command
    if action == "path":
        print(src)
        return 0
    if action == "validate":
        import runpy

        ns = runpy.run_path(
            str(src / "validate_api_claims.py"), run_name="_statspai_skill_gate"
        )
        saved = sys.argv
        # The gate parses sys.argv itself; hand it only its own flags.
        sys.argv = ["validate_api_claims.py"] + (
            ["--quick"] if getattr(args, "quick", False) else []
        )
        try:
            return int(ns["main"]())
        finally:
            sys.argv = saved
    if action == "install":
        target = (
            Path(args.target).expanduser()
            if args.target
            else Path.home() / ".claude" / "skills"
        )
        dest = target / SKILL_NAME
        if dest.exists():
            if not args.force:
                print(
                    f"{dest} already exists; pass --force to overwrite.",
                    file=sys.stderr,
                )
                return EXIT_USAGE
            shutil.rmtree(dest)
        shutil.copytree(
            src, dest, ignore=shutil.ignore_patterns("__pycache__", "*.pyc")
        )
        n_files = sum(1 for p in dest.rglob("*") if p.is_file())
        print(f"installed {SKILL_NAME} -> {dest} ({n_files} files)")
        return 0
    print("usage: statspai skill install|path|validate", file=sys.stderr)
    return EXIT_USAGE


def _print_route(payload: Dict[str, Any]) -> None:
    """Human rendering of ``sp.route`` / ``sp.decision_guide`` output."""
    if "routes" in payload and "answers" in payload:
        print(f"family: {payload['family']}   answers: {payload['answers']}")
        for r in payload["routes"]:
            print(f"\n  -> sp.{r['call']}")
            print(f"     {r['example']}")
            print(f"     why: {r['why']}")
            if r.get("read_more"):
                print(f"     read: {payload['guide']} / {r['read_more']}")
        if payload.get("next_question"):
            q = payload["next_question"]
            print(f"\nnext question: --answer {q['key']}=<{'|'.join(q['options'])}>")
            print(f"  {q['text']}")
        return
    if "questions" in payload:
        print(f"{payload['title']}  ({payload['guide']})")
        for q in payload["questions"]:
            print(f"  {q['key']}: {q['text']}")
            for a, meaning in q["options"].items():
                print(f"      {a:<24} {meaning}")
        return
    for fam, info in payload.items():
        print(f"{fam:<14} {info['title']}  questions: {', '.join(info['questions'])}")


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
