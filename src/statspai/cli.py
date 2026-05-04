"""Command-line interface for StatsPAI.

Install with ``pip install statspai`` and the ``statspai`` console
script becomes available (see ``[project.scripts]`` in pyproject.toml).

Commands
--------
    statspai list [--category CAT]
    statspai describe <name>
    statspai search <query>
    statspai help [<name>]
    statspai version

All commands ultimately delegate to sp.help / sp.list_functions /
sp.describe_function / sp.search_functions, so behaviour matches the
Python API exactly.
"""
from __future__ import annotations

import argparse
import json
import sys
from typing import List, Optional, Sequence


def _make_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="statspai",
        description=(
            "StatsPAI CLI — discover functions, read help, search the API. "
            "All output mirrors the Python-level sp.help() / sp.list_functions()."
        ),
    )
    parser.add_argument("--version", "-V", action="store_true",
                        help="Print StatsPAI version and exit.")

    sub = parser.add_subparsers(dest="command", metavar="<command>")

    # list
    p_list = sub.add_parser("list", help="List registered functions.")
    p_list.add_argument("--category", "-c", default=None,
                        help="Filter by category (e.g. causal, panel, spatial).")
    p_list.add_argument(
        "--stability", "-s", default=None,
        choices=["stable", "experimental", "deprecated"],
        help=(
            "Filter by stability tier. 'stable' = parity-grade "
            "(numerically aligned with R/Stata or analytic reference); "
            "'experimental' = frontier-grade (implemented but not yet "
            "parity-tested or API may shift); 'deprecated' = scheduled "
            "for removal."
        ),
    )
    p_list.add_argument("--json", action="store_true",
                        help="Emit JSON array instead of text.")

    # describe
    p_desc = sub.add_parser("describe", help="Show full metadata for a function.")
    p_desc.add_argument("name", help="Function name, e.g. 'did'.")
    p_desc.add_argument("--json", action="store_true",
                        help="Emit JSON object instead of text.")

    # search
    p_search = sub.add_parser("search", help="Keyword search across function metadata.")
    p_search.add_argument("query", nargs="+", help="One or more keywords.")
    p_search.add_argument("--json", action="store_true",
                          help="Emit JSON array instead of text.")

    # help
    p_help = sub.add_parser("help", help="Show help overview, or details for a topic.")
    p_help.add_argument("topic", nargs="?", default=None,
                        help="Function name, category, or 'category.name' path.")
    p_help.add_argument("--verbose", "-v", action="store_true",
                        help="Append full docstring after registry metadata.")

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
            print(json.dumps(spec, indent=2, default=str))
            return 0
        print(sp.help(args.name, verbose=True))
        return 0

    if args.command == "search":
        q = " ".join(args.query)
        if args.json:
            print(json.dumps(sp.search_functions(q), indent=2))
            return 0
        print(sp.help(search=q))
        return 0

    if args.command == "help":
        if args.topic is None:
            print(sp.help())
        else:
            print(sp.help(args.topic, verbose=args.verbose))
        return 0

    parser.print_help()
    return 1


if __name__ == "__main__":  # pragma: no cover
    sys.exit(main())
