"""Translations of Stata's ``estat`` post-estimation subcommands.

``estat`` is one command with many subcommands, each a test with its own
defaults. Where :func:`statspai.estat` has a different default from Stata
(the Breusch-Pagan test on the fitted values in its normal-theory form, the
RESET test with powers up to four), the translation writes Stata's default
out, so the call gives Stata's number.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

from ._stata import _emit, _emit_error
from ._stata_lexer import StataCommand

__all__ = ["HANDLERS", "POSTEST"]

#: subcommand -> (full name, minimum abbreviation), from each manual entry
_SUBCOMMANDS: Tuple[Tuple[str, int], ...] = (
    ("hettest", 4),
    ("imtest", 3),
    ("ovtest", 3),
    ("bgodfrey", 3),
    ("dwatson", 3),
    ("durbinalt", 3),
    ("archlm", 6),
    ("vif", 3),
    ("ic", 2),
    ("overid", 4),
    ("firststage", 5),
    ("endogenous", 5),
    ("classification", 4),
    ("ptrends", 3),
    ("granger", 3),
    ("summarize", 2),
    ("vce", 3),
)


def _subcommand(word: str) -> Optional[str]:
    word = word.lower()
    for full, shortest in _SUBCOMMANDS:
        if shortest <= len(word) <= len(full) and full.startswith(word):
            return full
    return None


def _call(test: str, extra: Dict[str, Any], semantics: List[str]) -> Dict[str, Any]:
    args: Dict[str, Any] = {"test": test, **extra, "print_results": False}
    shown = ", ".join(f"{k}={v!r}" for k, v in args.items())
    return _emit("estat", args, f"sp.estat(result, {shown})", semantics=semantics)


def _h_estat(cmd: StataCommand) -> Dict[str, Any]:
    """``estat <subcommand> [varlist] [, options]`` -> ``sp.estat(result, ...)``."""
    if not cmd.varlist:
        return _emit_error("estat needs a subcommand", command="estat", suggestions=[])
    sub = _subcommand(cmd.varlist[0])
    rest = cmd.varlist[1:]
    opts = cmd.options
    word = cmd.varlist[0].lower()
    if word in ("simple", "group", "calendar", "event") and not rest:
        # the aggregations csdid and jwdid define; written in full by both
        from ._stata_did import did_aggregation

        return did_aggregation("estat", word, opts)
    if sub is None:
        return _emit_error(
            f"estat {cmd.varlist[0]} is not translated",
            command="estat",
            suggestions=[],
        )
    if cmd.if_cond or cmd.in_range:
        return _emit_error(
            f"estat {sub} with if / in is not translated",
            command="estat",
            suggestions=[],
        )

    if sub == "hettest":
        extra: Dict[str, Any] = {}
        if rest:
            extra["variables"] = list(rest)
        elif "rhs" in opts:
            extra["variables"] = "rhs"
        else:
            extra["variables"] = "fitted"
        opts.get("rhs")
        if "iid" in opts:
            extra["version"] = "iid"
        elif "fstat" in opts:
            extra["version"] = "fstat"
        else:
            extra["version"] = "normal"
        opts.get("normal")
        return _call(
            "hettest",
            extra,
            [
                "Stata's default is the normal-theory score test on the fitted "
                "values; sp.estat's is Koenker's N R-squared on every "
                "regressor. The call spells out which one Stata ran."
            ],
        )

    if rest:
        return _emit_error(
            f"estat {sub} takes no variable list", command="estat", suggestions=[]
        )

    if sub == "imtest":
        opts.get("white")  # printed first by Stata; it is the first row here
        opts.get("preserve")
        return _call(
            "imtest",
            {},
            [
                "The statistic is White's test; the skewness and kurtosis "
                "parts of the decomposition are in the result's 'table'."
            ],
        )

    if sub == "ovtest":
        extra = {"powers": 4}
        if "rhs" in opts:
            extra["rhs"] = True
        return _call(
            "reset",
            extra,
            [
                "estat ovtest adds the powers 2, 3 and 4 (powers=4); "
                "sp.estat's own default stops at the cube."
            ],
        )

    if sub == "bgodfrey":
        extra = {}
        raw = opts.get("lags")
        if raw is not None:
            try:
                extra["lags"] = int(str(raw).strip())
            except ValueError:
                return _emit_error(
                    f"estat bgodfrey: lags({raw}) is a list; one lag order per "
                    "call is translated",
                    command="estat",
                    suggestions=[],
                )
        if "nomiss0" in opts:
            extra["fill"] = "drop"
        # `small` asks for the F form: left unread, so it is reported
        return _call(
            "bgodfrey",
            extra,
            ["The residuals are taken in the row order of the data (time order)."],
        )

    if sub == "classification":
        extra = {}
        raw = opts.get("cutoff")
        if raw is not None:
            try:
                extra["threshold"] = float(str(raw).strip())
            except ValueError:
                return _emit_error(
                    f"estat classification: cutoff({raw}) is not a number",
                    command="estat",
                    suggestions=[],
                )
        return _call("classification", extra, [])

    if sub in ("dwatson", "vif", "ic", "overid", "firststage", "endogenous"):
        return _call(sub, {}, [])

    if sub in ("ptrends", "granger"):
        return _call(
            sub,
            {},
            [
                "After didregress / xtdidregress (sp.didregress). Stata "
                "refuses both tests when treatment dates vary; so does this."
            ],
        )

    return _emit_error(
        f"estat {sub} is not translated", command="estat", suggestions=[]
    )


HANDLERS = {"estat": _h_estat}
POSTEST = frozenset({_h_estat})
