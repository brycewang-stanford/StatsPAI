"""Run Stata command lines against a DataFrame: ``sp.stata("...", data=df)``.

:func:`statspai.from_stata` translates one command into a tool call; this
module executes the translation and returns the fitted result, so a Stata
user can paste the lines they already know.  Anything the translator could
not map faithfully (an unknown command, an option or ``if`` / ``in``
qualifier the call does not carry over, or a translation carrying a
``<placeholder>`` the user must fill in) raises instead of running a
different model.
"""

from __future__ import annotations

import inspect
import re
import warnings
from typing import Any, List, Optional

import pandas as pd

from ...exceptions import MethodIncompatibility

__all__ = ["stata"]

_PLACEHOLDER = re.compile(r"<[A-Za-z_][A-Za-z0-9_ ]*>")
_PIPE_NOTE = re.compile(r"\bpipe\b")


def _accepts(fn: Any, name: str) -> bool:
    try:
        return name in inspect.signature(fn).parameters
    except (TypeError, ValueError):  # pragma: no cover - builtins
        return False


def _uses_estimation_sample(fn: Any, result: Any) -> bool:
    """True when ``fn`` can run on ``result`` alone, averaging over the
    fitted model's own estimation sample as Stata's post-estimation
    commands do over ``e(sample)``.  Passing the raw data instead would
    bring back rows the fit dropped (missing values, markout)."""
    try:
        param = inspect.signature(fn).parameters["data"]
    except (KeyError, TypeError, ValueError):
        return False
    if param.default is inspect.Parameter.empty:
        return False
    info = getattr(result, "data_info", None) or {}
    X, names = info.get("X"), info.get("var_names")
    params = getattr(result, "params", None)
    if X is None or names is None or params is None:
        return False
    names = list(names)
    return (
        getattr(X, "ndim", 0) == 2
        and X.shape[1] == len(names)
        and names == [str(p) for p in params.index]
        and not any("[" in n or ":" in n for n in names)
    )


def _command_lines(commands: str) -> List[str]:
    lines = []
    for raw in commands.replace(";", "\n").splitlines():
        line = raw.split("//", 1)[0].strip()
        if line and not line.startswith("*"):
            lines.append(line)
    return lines


def stata(
    commands: str,
    data: Optional[pd.DataFrame] = None,
    *,
    result: Any = None,
) -> Any:
    """Execute Stata estimation / post-estimation commands on a DataFrame.

    Parameters
    ----------
    commands : str
        One Stata command, or several separated by newlines or ``;``.
        ``*`` and ``//`` comments are ignored. Post-estimation commands
        (``margins``, ``test``, ``lincom`` ...) apply to the most recent
        estimation result.
    data : pandas.DataFrame, optional
        The dataset estimation commands run on. Required unless every
        command is a post-estimation command applied to ``result``.
    result : fitted result, optional
        The estimation result that leading post-estimation commands apply
        to.

    Returns
    -------
    object
        The output of the last command: a fitted result for estimation
        commands, the post-estimation output otherwise.

    Raises
    ------
    MethodIncompatibility
        When a command is not supported by :func:`statspai.from_stata`, or
        its translation needs information the line does not carry (for
        example the panel identifier of ``xtreg, fe``, which Stata takes
        from an earlier ``xtset``).
    TypeError
        When an estimation command is run without ``data``, or a
        post-estimation command before any estimation result exists.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> df = pd.DataFrame({"x1": rng.normal(size=200), "x2": rng.normal(size=200)})
    >>> df["y"] = 1 + 0.5 * df.x1 - 0.2 * df.x2 + rng.normal(size=200)
    >>> r = sp.stata("regress y x1 x2, vce(robust)", data=df)
    >>> direct = sp.regress("y ~ x1 + x2", data=df, vce="robust")
    >>> bool((r.std_errors == direct.std_errors).all())
    True
    >>> out = sp.stata("regress y x1 x2; lincom x1 + x2", data=df)
    >>> out == sp.lincom(sp.regress("y ~ x1 + x2", data=df), "x1 + x2")
    True
    """
    import statspai as sp

    from ._stata import from_stata

    lines = _command_lines(commands)
    if not lines:
        raise ValueError("sp.stata: no command given.")
    last = result
    output: Any = result
    for line in lines:
        out = from_stata(line, columns=None if data is None else list(data.columns))
        if not out.get("ok"):
            suggestions = out.get("suggestions") or []
            raise MethodIncompatibility(
                f"sp.stata: cannot run {line!r}: {out.get('error')}",
                recovery_hint=(
                    f"Did you mean: {', '.join(suggestions)}?"
                    if suggestions
                    else "Call the sp.* function directly; sp.from_stata "
                    "lists what it translates."
                ),
                diagnostics={"command": line, "translation": out},
            )
        notes = list(out.get("notes") or [])
        blocking = [n for n in notes if _PLACEHOLDER.search(n)]
        if blocking:
            raise MethodIncompatibility(
                f"sp.stata: {line!r} cannot be run as written. {blocking[0]}",
                recovery_hint=f"Run it directly: {out.get('python_code')}",
                diagnostics={"command": line, "translation": out},
            )
        lost = list(out.get("untranslated_options") or [])
        sample = out.get("unapplied_sample")
        if lost or sample:
            what = []
            if lost:
                what.append("option(s) " + ", ".join(lost) + " are not translated")
            if sample:
                what.append(f"`{sample}` is not applied")
            raise MethodIncompatibility(
                f"sp.stata: {line!r} cannot be run as written: "
                + "; ".join(what)
                + ". Running it would fit a different model.",
                recovery_hint=(
                    (
                        "Filter the DataFrame first (Stata treats missing as "
                        "+infinity in comparisons; pandas does not) and drop "
                        "the qualifier. "
                        if sample
                        else ""
                    )
                    + f"Then run it directly: {out.get('python_code')}"
                ),
                diagnostics={"command": line, "translation": out},
            )
        code = str(out.get("python_code") or "")
        chained = out["tool"] == "marginsplot" or code.startswith(
            f"sp.{out['tool']}(result"
        )
        for note in notes:
            # "pipe the previous result" is advice for callers of
            # sp.from_stata; this runner does the piping itself.
            if chained and _PIPE_NOTE.search(note):
                continue
            warnings.warn(f"sp.stata: {note}", UserWarning, stacklevel=2)

        fn: Any = sp
        for part in str(out["tool"]).split("."):
            fn = getattr(fn, part)
        arguments = dict(out.get("arguments") or {})
        if out["tool"] == "marginsplot":
            if not isinstance(output, pd.DataFrame):
                raise TypeError(
                    f"sp.stata: {line!r} plots the output of a preceding "
                    "`margins` command in the same call."
                )
            output = fn(output)
        elif chained:
            if last is None:
                raise TypeError(
                    f"sp.stata: {line!r} is a post-estimation command; run an "
                    "estimation command first or pass result=."
                )
            if (
                _accepts(fn, "data")
                and "data" not in arguments
                and not _uses_estimation_sample(fn, last)
            ):
                if data is None and "data=df" in code:
                    raise TypeError(f"sp.stata: {line!r} needs data=<DataFrame>.")
                if data is not None:
                    arguments["data"] = data
            output = fn(last, **arguments)
        else:
            if data is None:
                raise TypeError(f"sp.stata: {line!r} needs data=<DataFrame>.")
            output = fn(data=data, **arguments)
            last = output
    return output
