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
from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from ...exceptions import MethodIncompatibility
from ._stata_datastep import DataSteps, row_mask
from ._stata_expr import StataExprError, evaluate
from ._stata_tsops import has_ts_operator, rewrite_ts_operators

__all__ = ["stata", "StataSession"]

#: Tools that describe the data without fitting anything. As in Stata,
#: where ``summarize`` is r-class and leaves ``e()`` alone, they do not
#: replace the estimation result later post-estimation commands apply to.
_DESCRIPTIVE_TOOLS = frozenset({"sumstats", "pwcorr", "ttest", "unitroot"})

#: Commands that set up the session or only print: skipping them leaves
#: every later estimate unchanged. (``use`` is not here: it replaces the
#: data, so a snippet containing it is refused.)
_SKIPPED = re.compile(
    r"\s*(?:set\s+(?:more|linesize|matsize|scheme|type\s+double)|"
    r"log\s|cap(?:ture)?\s+log\s|label\s|format\s|describe\b|desc\b|"
    r"list\b|browse\b|notes?\b|version\s|clear\s+(?:all|matrix|mata)\s*$|"
    r"macro\s+drop|eststo\s+clear|estimates\s+clear|graph\s+(?:export|save)|"
    r"pause\b|exit\s*$)",
    re.I,
)


#: Commands that draw or write a file. They do not change any estimate, so
#: the snippet goes on; a warning says the graph or file was not produced.
_NOT_PRODUCED = re.compile(
    r"\s*(?:twoway|scatter|line|histogram|hist|kdensity|tsline|graph\s+"
    r"(?:twoway|bar|box|combine)|export\s|outsheet\s|save\s|saveold\s)",
    re.I,
)

_SCALAR = re.compile(
    r"\s*sca(?:l(?:a(?:r)?)?)?\s+(?:define\s+)?([A-Za-z_]\w*)\s*=(?!=)\s*(.+)\Z",
    re.S | re.I,
)
_DISPLAY = re.compile(r"\s*di(?:s(?:p(?:l(?:a(?:y)?)?)?)?)?\s+(.+)\Z", re.S | re.I)


def _qualified(
    line: str, data: pd.DataFrame, stored: Dict[str, Dict[str, float]]
) -> pd.DataFrame:
    """``data`` restricted by the ``if`` / ``in`` qualifier of ``line``."""
    from . import _stata_options as _opts
    from ._stata_lexer import parse as _parse

    _, _, core = _opts.peel_prefixes(line)
    cmd = _parse(core)
    return data.loc[row_mask(data, cmd.if_cond, cmd.in_range, stored)]


#: Notes from sp.from_stata telling its caller to filter the data first.
_FILTER_NOTE = re.compile(r"pre-filter df|filter df first")

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


def _with_panel(
    line: str,
    out: Dict[str, Any],
    panel: Tuple[Optional[str], Optional[str]],
    columns: Any,
) -> Dict[str, Any]:
    """Fill a ``<panel_id>`` placeholder from an earlier ``xtset``.

    The handlers that need the panel id take Stata's own ``i()`` option, so
    the declaration is appended to the command and it is translated again.
    """
    from ._stata import from_stata
    from ._stata_lexer import _split_options

    unit, time = panel
    notes = " ".join(out.get("notes") or []) + str(out.get("python_code") or "")
    if unit is None or not out.get("ok") or "<panel_id>" not in notes:
        return out
    sep = " " if _split_options(line)[1] else ", "
    if time is not None:
        both = from_stata(f"{line}{sep}i({unit}) t({time})", columns=columns)
        if both.get("ok") and "t" not in (both.get("untranslated_options") or []):
            return both
    return from_stata(f"{line}{sep}i({unit})", columns=columns)


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
        One Stata command, or a do-file snippet: commands separated by
        newlines or ``;``, with ``///`` continuations, ``*`` / ``//`` /
        ``/* */`` comments and ``#delimit ;``. ``global`` / ``local``
        macros whose value is written out are expanded, and ``xtset id
        [time]`` supplies the panel of a later ``xtreg, fe`` / ``xtabond``.
        Post-estimation commands (``margins``, ``test``, ``lincom`` ...)
        apply to the most recent estimation result.

        ``if`` / ``in`` qualifiers are applied with Stata's rules for
        missing values (a missing value is larger than any number, so
        ``if x > 0`` keeps the rows where ``x`` is missing). The data steps
        ``generate``, ``replace``, ``keep``, ``drop``, ``sort``,
        ``preserve`` and ``restore`` run on a private copy of ``data``;
        ``generate`` stores single precision unless the line says
        ``double``, as Stata does. ``predict`` gives fitted values or
        residuals after a linear fit. ``scalar name = exp`` and
        ``display exp`` evaluate one numeric expression, which may use
        ``_b[x]``, ``_se[x]``, ``e(N)`` / ``e(r2)`` / ``e(r2_a)`` and the
        ``r()`` results of ``summarize``, ``test`` and ``ttest``. Session
        settings and output-only lines (``set more off``, ``log``,
        ``label``, ``describe`` ...) are skipped; graph and export commands
        are skipped with a warning. ``use`` is refused: pass the data in.

        After ``tsset time`` (or ``xtset id time``) the time-series
        operators ``L.x``, ``L2.x``, ``F.x``, ``D.x``, ``L(1/4).x`` are
        resolved against the time variable, within panel, as Stata does: a
        lag is missing where the earlier period is absent. The term
        ``L2.x`` enters the model as a column named ``x_L2``.
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
        from an earlier ``xtset`` when the snippet has none), or the snippet
        uses a loop, an undefined macro, or a macro only Stata can evaluate,
        or an expression outside the implemented set (``e(sample)``,
        ``_b[x]``, time-series operators, string functions).
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
    >>> m = sp.stata('''
    ...     global rhs "x1 x2"
    ...     regress y $rhs, ///
    ...         r            // robust
    ... ''', data=df)
    >>> bool((m.std_errors == r.std_errors).all())
    True
    >>> out = sp.stata("regress y x1 x2; lincom x1 + x2", data=df)
    >>> out == sp.lincom(sp.regress("y ~ x1 + x2", data=df), "x1 + x2")
    True
    """
    from ._stata_script import split_commands

    lines = split_commands(commands)
    if not lines:
        raise ValueError("sp.stata: no command given.")
    session = StataSession(data, result=result)
    for line in lines:
        session.run(line)
    return session.output


class StataSession:
    """State carried from one Stata command to the next.

    ``sp.stata`` is one session run over the lines of a snippet. The session
    holds what Stata would hold between commands: the data in memory (after
    ``generate`` / ``keep`` / ``sort``), the ``xtset`` declaration, the
    defined macros and the most recent estimation result.

    Attributes
    ----------
    data : pandas.DataFrame or None
        The data as the next command will see it. The caller's DataFrame is
        never modified; the first data step works on a copy.
    last : object
        The most recent estimation result (what ``test`` / ``margins`` use).
    last_data : pandas.DataFrame or None
        The rows that result was fitted on.
    output : object
        What the most recent command returned.
    """

    def __init__(self, data: Optional[pd.DataFrame] = None, *, result: Any = None):
        from ._stata_script import MacroTable

        self._steps = None if data is None else DataSteps(data)
        self._macros = MacroTable()
        self.panel: Tuple[Optional[str], Optional[str]] = (None, None)
        self.last = result
        self.last_data = data
        self.output: Any = result
        self._last_call: Optional[Dict[str, Any]] = None
        #: what earlier commands left behind: r(), e(), _b[], _se[], scalars
        self.stored: Dict[str, Dict[str, float]] = {"r": {}, "scalars": {}}
        if self._steps is not None:
            self._steps.stored = self.stored
        if result is not None:
            self._store_estimates(result)

    def value(self, expr: str) -> float:
        """A scalar Stata expression, e.g. ``_b[x] * r(mean)``."""
        frame = self.data if self.data is not None else pd.DataFrame({"_": [0.0]})
        out = evaluate(expr, frame.iloc[:1], self.stored)
        if out.dtype == object:
            raise StataExprError("the expression is a string, not a number")
        return float(out[0])

    def _store_estimates(self, result: Any) -> None:
        params = getattr(result, "params", None)
        ses = getattr(result, "std_errors", None)
        if params is None:
            return
        b = {str(k): float(v) for k, v in dict(params).items()}
        se = {} if ses is None else {str(k): float(v) for k, v in dict(ses).items()}
        for alias in ("Intercept", "const"):
            if alias in b:
                b["_cons"] = b[alias]
                if alias in se:
                    se["_cons"] = se[alias]
        e: Dict[str, float] = {}
        for key, attr in (("N", "nobs"), ("r2", "r2"), ("r2_a", "r2_adj")):
            got = getattr(result, attr, None)
            if isinstance(got, (int, float, np.integer, np.floating)):
                e[key] = float(got)
        self.stored["_b"], self.stored["_se"], self.stored["e"] = b, se, e

    def _store_r(self, tool: str, arguments: Dict[str, Any], frame: Any) -> None:
        """r() after an r-class command, at full precision."""
        r: Dict[str, float] = {}
        out = self.output
        if tool == "sumstats" and isinstance(frame, pd.DataFrame):
            names = arguments.get("vars") or list(frame.columns)
            col = frame[names[-1]]
            if pd.api.types.is_numeric_dtype(col) or pd.api.types.is_bool_dtype(col):
                x = col.dropna().to_numpy(dtype=float)
                r = {"N": float(x.size)}
                if x.size:
                    r.update(mean=float(x.mean()), sum=float(x.sum()),
                             min=float(x.min()), max=float(x.max()))  # fmt: skip
                if x.size > 1:
                    r.update(sd=float(x.std(ddof=1)), Var=float(x.var(ddof=1)))
        elif tool == "test" and isinstance(out, dict):
            df = out.get("df")
            q, resid = df if isinstance(df, tuple) else (df, None)
            r = {"p": float(out["pvalue"]), "df": float(q)}
            if out.get("distribution") == "F":
                r["F"] = float(out["statistic"])
                if resid is not None:
                    r["df_r"] = float(resid)
            else:
                r["chi2"] = float(out["statistic"])
        elif tool == "ttest":
            r = {"t": float(out.statistic), "df_t": float(out.df),
                 "p": float(out.pvalue), "se": float(out.se)}  # fmt: skip
        self.stored["r"] = r

    @property
    def data(self) -> Optional[pd.DataFrame]:
        return None if self._steps is None else self._steps.data

    def run(self, line: str) -> bool:
        """Run one command. Returns ``False`` for a line that produces no
        output (a macro definition, ``xtset``, a data step, a setting)."""
        import statspai as sp

        from ._stata import from_stata
        from ._stata_script import ScriptError, control_flow, panel_declaration

        data = self.data
        columns = None if data is None else list(data.columns)
        flow = control_flow(line)
        if flow is not None:
            raise MethodIncompatibility(
                f"sp.stata: {line!r} is control flow ({flow}); loops, programs "
                "and blocks are not run.",
                recovery_hint="Write the loop in Python around sp.stata(...) "
                "or the sp.* call.",
                diagnostics={"command": line},
            )
        try:
            if self._macros.define(line):
                return False
            line = self._macros.expand(line)
        except ScriptError as exc:
            raise MethodIncompatibility(
                f"sp.stata: cannot run {line!r}: {exc}",
                recovery_hint="Define the macro with `global name ...` / "
                "`local name ...` above the command, or write it out.",
                diagnostics={"command": line},
            ) from exc
        declared = panel_declaration(line)
        if declared is not None:
            self.panel = declared
            return False
        if _SKIPPED.match(line):
            # session settings and output-only commands: nothing to run
            return False
        if _NOT_PRODUCED.match(line):
            warnings.warn(
                f"sp.stata: skipped {line.split()[0]!r}: graphs are not drawn "
                "and files are not written. No estimate depends on it.",
                UserWarning,
                stacklevel=3,
            )
            return False
        if self.panel[1] is not None and has_ts_operator(line):
            # L.x / D.x / L(1/4).x against the declared time variable
            naming_only = re.match(r"\s*(?:test|lincom)\b", line) is not None
            try:
                line, columns_added = rewrite_ts_operators(
                    line, None if naming_only else data, self.panel
                )
                if self._steps is not None:
                    for name, values in columns_added.items():
                        self._steps.add_column(name, values, double=True)
            except StataExprError as exc:
                raise MethodIncompatibility(
                    f"sp.stata: cannot run {line!r}: {exc}.",
                    recovery_hint="Build the lag as a column in pandas and "
                    "pass the prepared DataFrame.",
                    diagnostics={"command": line},
                ) from exc
            data = self.data
            columns = None if data is None else list(data.columns)
        scalar = _SCALAR.match(line)
        show = None if scalar else _DISPLAY.match(line)
        if scalar is not None:
            expr = scalar.group(2).strip()
        elif show is not None:
            expr = show.group(1).strip()
        if scalar is not None or show is not None:
            try:
                number = self.value(expr)
            except StataExprError as exc:
                raise MethodIncompatibility(
                    f"sp.stata: cannot run {line!r}: {exc}.",
                    recovery_hint="Only a single numeric expression is "
                    "evaluated; compute it in Python from the result.",
                    diagnostics={"command": line},
                ) from exc
            if scalar:
                self.stored["scalars"][scalar.group(1)] = number
                return False
            self.output = number
            return True
        if self._steps is not None:
            try:
                if self._predict(line) or self._steps.apply(line):
                    return False
            except StataExprError as exc:
                raise MethodIncompatibility(
                    f"sp.stata: cannot run {line!r}: {exc}.",
                    recovery_hint="Do this step in pandas, pass the prepared "
                    "DataFrame as data= and drop the line.",
                    diagnostics={"command": line},
                ) from exc
        line = self._scalars_in_restriction(line)
        out = _with_panel(line, from_stata(line, columns=columns), self.panel, columns)
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
        code = str(out.get("python_code") or "")
        chained = out["tool"] == "marginsplot" or code.startswith(
            f"sp.{out['tool']}(result"
        )
        run_data = data
        if sample and not lost and not chained and data is not None:
            # apply the qualifier with Stata's missing-value rules
            try:
                run_data = _qualified(line, data, self.stored)
            except StataExprError as exc:
                raise MethodIncompatibility(
                    f"sp.stata: {line!r} cannot be run as written: "
                    f"`{sample}` is not applied ({exc}).",
                    recovery_hint="Filter the DataFrame first (Stata treats "
                    "missing as +infinity in comparisons; pandas does not) "
                    f"and drop the qualifier. Then: {out.get('python_code')}",
                    diagnostics={"command": line, "translation": out},
                ) from exc
            sample = None
            notes = [n for n in notes if not _FILTER_NOTE.search(n)]
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
        for note in notes:
            # "pipe the previous result" is advice for callers of
            # sp.from_stata; this runner does the piping itself.
            if chained and _PIPE_NOTE.search(note):
                continue
            warnings.warn(f"sp.stata: {note}", UserWarning, stacklevel=3)

        fn: Any = sp
        for part in str(out["tool"]).split("."):
            fn = getattr(fn, part)
        arguments = dict(out.get("arguments") or {})
        if out["tool"] == "marginsplot":
            if not isinstance(self.output, pd.DataFrame):
                raise TypeError(
                    f"sp.stata: {line!r} plots the output of a preceding "
                    "`margins` command in the same call."
                )
            self.output = fn(self.output)
        elif chained:
            if self.last is None:
                raise TypeError(
                    f"sp.stata: {line!r} is a post-estimation command; run an "
                    "estimation command first or pass result=."
                )
            if (
                _accepts(fn, "data")
                and "data" not in arguments
                and not _uses_estimation_sample(fn, self.last)
            ):
                if self.last_data is None and "data=df" in code:
                    raise TypeError(f"sp.stata: {line!r} needs data=<DataFrame>.")
                if self.last_data is not None:
                    # the rows the model was fitted on, not the full frame
                    arguments["data"] = self.last_data
            self.output = fn(self.last, **arguments)
            self._store_r(str(out["tool"]), arguments, None)
        else:
            if run_data is None:
                raise TypeError(f"sp.stata: {line!r} needs data=<DataFrame>.")
            self.output = fn(data=run_data, **arguments)
            if out["tool"] not in _DESCRIPTIVE_TOOLS:
                self.last = self.output
                self.last_data = run_data
                self._last_call = out
                self._store_estimates(self.output)
            else:
                self._store_r(str(out["tool"]), arguments, run_data)
        return True

    def _scalars_in_restriction(self, line: str) -> str:
        """Write named scalars into a ``test`` / ``lincom`` restriction.

        ``test d1*x + d2*x2 = 0`` with ``scalar d1 = ...`` is legal Stata;
        sp.test reads names as coefficients, so the scalars are replaced by
        their values. A name that is also a coefficient stays a coefficient.
        """
        scalars = self.stored.get("scalars") or {}
        if not scalars or not re.match(r"\s*(?:test|lincom)\b", line):
            return line
        coefficients = self.stored.get("_b") or {}
        for name, number in scalars.items():
            if name in coefficients:
                continue
            line = re.sub(
                rf"(?<![\w.]){re.escape(name)}(?![\w(\[])", repr(float(number)), line
            )
        return line

    def _predict(self, line: str) -> bool:
        """``predict [type] newvar [, xb | residuals]`` after a linear fit.

        Stata predicts for every row whose regressors are observed, not only
        the estimation sample; so does this. Only linear models whose
        coefficients are plain columns are covered -- with factor variables,
        absorbed effects or a nonlinear link the line is refused.
        """
        from ._stata_lexer import StataParseError
        from ._stata_lexer import parse as _parse

        try:
            cmd = _parse(line)
        except StataParseError:
            return False
        if cmd.command != "predict":
            return False
        if self.last is None or self._last_call is None or self._steps is None:
            raise StataExprError("`predict` needs an estimation command before it")
        if cmd.if_cond or cmd.in_range:
            raise StataExprError("`predict` with if / in is not implemented")
        kinds = [k for k in cmd.options if k]
        resid = any("residuals".startswith(k) for k in kinds)
        if (
            any(k != "xb" and not "residuals".startswith(k) for k in kinds)
            or len(kinds) > 1
        ):
            raise StataExprError(
                f"`predict` option(s) {sorted(kinds)} are not implemented; "
                "only xb and residuals are"
            )
        names = list(cmd.varlist)
        double = False
        if len(names) == 2 and names[0] in ("float", "double"):
            double = names[0] == "double"
            names = names[1:]
        if len(names) != 1:
            raise StataExprError("expected `predict [type] newvar`")
        tool = self._last_call.get("tool")
        formula = str((self._last_call.get("arguments") or {}).get("formula") or "")
        if tool not in ("regress", "ivreg") or "~" not in formula:
            raise StataExprError(
                f"`predict` after sp.{tool} is not implemented (linear models only)"
            )
        data = self._steps.data
        params = getattr(self.last, "params")
        total: np.ndarray = np.zeros(len(data))
        for term, beta in params.items():
            if term in ("Intercept", "const", "_cons"):
                total = total + float(beta)
            elif term in data.columns:
                total = total + float(beta) * data[term].to_numpy(
                    dtype=float, na_value=np.nan
                )
            else:
                raise StataExprError(
                    f"`predict`: coefficient {term!r} is not a column of the "
                    "data (factor variables and interactions are not covered)"
                )
        if resid:
            outcome = formula.split("~", 1)[0].strip()
            if outcome not in data.columns:
                raise StataExprError(f"`predict`: outcome {outcome!r} is not a column")
            total = data[outcome].to_numpy(dtype=float, na_value=np.nan) - total
        self._steps.add_column(names[0], total, double=double)
        return True
