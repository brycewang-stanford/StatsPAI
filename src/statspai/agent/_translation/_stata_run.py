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
from ._stata_session import (
    absorbed_constant,
    expand_frequency,
    prepare_weights,
    psmatch2_after,
    run_session_command,
    stata_percentile,
    teffects_after,
    teffects_before,
    xtreg_extras,
)
from ._stata_tsops import has_ts_operator, rewrite_ts_operators

__all__ = ["stata", "StataSession"]

#: Tools that describe the data without fitting anything. As in Stata,
#: where ``summarize`` is r-class and leaves ``e()`` alone, they do not
#: replace the estimation result later post-estimation commands apply to.
_DESCRIPTIVE_TOOLS = frozenset(
    {
        "sumstats", "pwcorr", "ttest", "bitest", "unitroot", "corrgram", "varsoc",
        "xtsum", "xtserial",
    }  # fmt: skip
)

#: Commands that set up the session or only print: skipping them leaves
#: every later estimate unchanged. (``use`` is not here: it replaces the
#: data, so a snippet containing it is refused.)
_SKIPPED = re.compile(
    r"\s*(?:set\s+(?:more|linesize|matsize|scheme|graphics|type\s+double)|"
    r"log\s|cap(?:ture)?\s+log\s|label\s|format\s|describe\b|desc\b|"
    r"list\b|browse\b|notes?\b|codebook\b|labelbook\b|version\s|clear\s+(?:all|matrix|mata)\s*$|"
    r"macro\s+drop|eststo\s+clear|graph\s+(?:export|save)|"
    r"return\s+list\b|ereturn\s+list\b|sysdir\b|help\s|xtdes(?:cribe)?\b|"
    r"irf\s+(?:create|set|drop|describe)\b|"
    r"pause\b)",
    re.I,
)

#: ``quietly {`` / ``capture {`` / ``noisily {``: a block that changes what
#: is shown (or whether an error stops the do-file), not what is computed
_QUIET_BLOCK = re.compile(
    r"\s*(?:qui(?:e(?:t(?:ly?)?)?)?|n(?:o(?:i(?:s(?:i(?:ly?)?)?)?)?)?|"
    r"cap(?:t(?:u(?:re?)?)?)?)\s*\{\s*$",
    re.I,
)
_CAPTURE = re.compile(
    r"\s*cap(?:t(?:u(?:re?)?)?)?\s+(?:n(?:o(?:i(?:s(?:i(?:ly?)?)?)?)?)?\s+)?"
    r"(?!log\b|program\b|\{)(\S.*)\Z",
    re.I | re.S,
)
_QUIETLY = re.compile(
    r"\s*(?:qui(?:e(?:t(?:ly?)?)?)?|noi(?:s(?:i(?:ly?)?)?)?)\s+(?!\{)(\S.*)\Z",
    re.I | re.S,
)
_EXIT = re.compile(r"\s*exit\s*(?:,\s*clear\s*)?$", re.I)


#: Commands that draw or write a file. They do not change any estimate, so
#: the snippet goes on; a warning says the graph or file was not produced.
_NOT_PRODUCED = re.compile(
    r"\s*(?:(?:twoway|tw|scatter|line|histogram|hist|kdensity|tsline|xtline|"
    r"ac|pac|rvfplot|rvpplot|avplots?|lvr2plot|qnorm|pnorm|coefplot|"
    r"teoverlap)\b|tebalance\s+(?:box|density)\b|"
    r"(?:irf|fcast)\s+(?:c?graph|ograph)\b|graph\s+"
    r"(?:twoway|bar|box|combine)|export\s|outsheet\s|save\s|saveold\s)",
    re.I,
)

#: predict's options, each with the abbreviations Stata accepts
_PREDICT_KINDS: Dict[str, str] = {"xb": "xb", "pr": "pr", "p": "pr", "hat": "leverage"}
_PREDICT_KINDS.update({"residuals"[:k]: "residuals" for k in range(1, 10)})
_PREDICT_KINDS.update({"leverage"[:k]: "leverage" for k in range(3, 9)})

_SCALAR = re.compile(
    r"\s*sca(?:l(?:a(?:r)?)?)?\s+(?:define\s+)?([A-Za-z_]\w*)\s*=(?!=)\s*(.+)\Z",
    re.S | re.I,
)
_DISPLAY = re.compile(r"\s*di(?:s(?:p(?:l(?:a(?:y)?)?)?)?)?\s+(.+)\Z", re.S | re.I)


#: `` `r(name)' `` / `` `e(name)' ``: a stored result used as a macro
_RESULT_MACRO = re.compile(r"`([re]\([A-Za-z_]\w*\))'")
#: ``local name = exp`` (not ``local name exp``, which stores the text)
_MACRO_EXPRESSION = re.compile(
    r"^(gl(?:o(?:b(?:al?)?)?)?|loc(?:al?)?)\s+([A-Za-z_]\w*)\s*=(.*)$", re.I
)


def _macro_number(value: float) -> str:
    """A number as Stata writes it into a macro, format ``%18.0g``: as many
    significant digits as fit in 17 characters beside the sign, at most 17.
    The mean 0.41178860374999998 becomes ``.41178860375`` and 1/3 becomes
    ``.3333333333333333``, so a line that reads the macro back computes
    with the number Stata computes with, not with the stored double."""
    if value != value:
        return "."
    magnitude = abs(float(value))
    text = "0"
    for digits in range(17, 0, -1):
        text = format(magnitude, f".{digits}g")
        if text.startswith("0."):
            text = text[1:]
        if len(text) <= 17:
            break
    return ("-" if value < 0 else "") + text


def _qualified(line: str, data: pd.DataFrame, stored: Dict[str, Any]) -> pd.DataFrame:
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


_FACTOR_TERM = re.compile(r"C\((\w+)(?:,[^\]]*)?\)\[(?:T\.)?(.+)\]\Z")


def _term_column(term: str, data: pd.DataFrame) -> Optional[np.ndarray]:
    """The regressor behind a coefficient name such as ``C(g)[T.2.0]:x``.

    Each ``:``-separated piece is a column or the indicator of one level of
    a factor; the regressor is their product, missing where any of the
    variables is. ``None`` when a piece is neither.
    """
    out = np.ones(len(data))
    for piece in term.split(":"):
        if piece in data.columns:
            column = data[piece].to_numpy(dtype=float, na_value=np.nan)
        else:
            m = _FACTOR_TERM.match(piece)
            if m is None or m.group(1) not in data.columns:
                return None
            raw = data[m.group(1)]
            level = m.group(2)
            if pd.api.types.is_numeric_dtype(raw) or pd.api.types.is_bool_dtype(raw):
                try:
                    hit = raw.to_numpy(dtype=float, na_value=np.nan) == float(level)
                except ValueError:
                    return None
            else:
                hit = raw.astype(str).to_numpy() == level
            column = np.where(raw.isna().to_numpy(), np.nan, hit.astype(float))
        out = out * column
    return out


def _e_scalars(result: Any) -> Dict[str, float]:
    """``e(rss)``, ``e(mss)``, ``e(rmse)``, ``e(df_r)``, ``e(df_m)``, ``e(F)``
    and ``e(ll)`` where the result carries what they are computed from."""
    out: Dict[str, float] = {}
    info = getattr(result, "data_info", None) or {}
    diag = getattr(result, "diagnostics", None) or {}
    for key, name in (("df_r", "df_resid"), ("df_m", "df_model")):
        got = info.get(name)
        if isinstance(got, (int, float, np.integer, np.floating)):
            out[key] = float(got)
    rss, tss = info.get("rss"), info.get("tss")
    if isinstance(rss, float) and isinstance(tss, float):
        out["rss"], out["mss"] = rss, tss - rss
        if out.get("df_r", 0) > 0:
            out["rmse"] = float(np.sqrt(rss / out["df_r"]))
    for key, name in (("F", "F-statistic"), ("ll", "Log-Likelihood")):
        got = diag.get(name)
        if isinstance(got, (int, float, np.integer, np.floating)) and np.isfinite(got):
            out[key] = float(got)
    return out


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
        if _EXIT.match(line):
            break  # `exit` ends a do-file; Stata runs nothing after it
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
        self._quiet_blocks = 0
        #: models named by `estimates store`: name -> (result, data, call)
        self.estimates: Dict[str, Tuple[Any, Any, Any]] = {}
        #: columns a command needed for its own run (a weight expression)
        self._scratch: list = []
        #: True while the data in memory hold random draws
        self.simulated = False
        #: programs defined by `program name ... end`: name -> body lines
        self.programs: Dict[str, list] = {}
        self._defining: Optional[Tuple[str, list]] = None
        #: `return scalar` values of the program being run
        self._returned: Optional[Dict[str, float]] = None
        self._translations: Dict[Any, Dict[str, Any]] = {}
        #: what earlier commands left behind: r(), e(), _b[], _se[], scalars.
        #: "rng" feeds rnormal() / runiform(); `set seed` reseeds it.
        self.stored: Dict[str, Any] = {
            "r": {},
            "scalars": {},
            "rng": np.random.default_rng(),
        }
        if self._steps is not None:
            self._steps.stored = self.stored
        if result is not None:
            self._store_estimates(result)

    def warn(self, message: str) -> None:
        warnings.warn(message, UserWarning, stacklevel=4)

    def use(self, data: pd.DataFrame) -> None:
        """Replace the data in memory, as Stata's ``use file, clear`` does.

        ``sp.stata`` itself never reads a file; a caller that has loaded one
        hands it over here. The panel declaration is dropped with the data.
        """
        if self._steps is None:
            self._steps = DataSteps(data)
            self._steps.stored = self.stored
        else:
            self._steps.replace_data(data.copy())
            self._steps._float = set()
        self.simulated = False
        self.panel = (None, None)
        self.stored.pop("time_var", None)
        self.stored.pop("panel_var", None)

    def append(self, other: pd.DataFrame) -> None:
        """Add the rows of ``other`` below the data (``append using``).

        Variables present on one side only are missing on the other.
        """
        if self._steps is None:
            self.use(other)
            return
        self._steps.replace_data(
            pd.concat([self._steps.data, other], ignore_index=True, sort=False)
        )

    def value(self, expr: str) -> float:
        """A scalar Stata expression, e.g. ``_b[x] * r(mean)``."""
        frame = self.data
        if frame is None or frame.empty:
            # a scalar does not need data in memory (`clear`, then `scalar`)
            columns = [] if frame is None else list(frame.columns)
            frame = pd.DataFrame({c: [np.nan] for c in columns} or {"_": [0.0]})
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
        e.update(_e_scalars(result))
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
                if x.size:
                    # what `summarize, detail` adds
                    for q in (1, 5, 10, 25, 50, 75, 90, 95, 99):
                        r[f"p{q}"] = stata_percentile(x, q)
                    dev = x - x.mean()
                    m2 = float(np.mean(dev**2))
                    if m2 > 0:
                        r["skewness"] = float(np.mean(dev**3) / m2**1.5)
                        r["kurtosis"] = float(np.mean(dev**4) / m2**2)
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
        elif tool == "bitest":
            r = {"N": float(out.n_obs), "k": float(out.successes),
                 "P_p": float(out.p_null), "p": float(out.pvalue),
                 "p_l": float(out.pvalue_less),
                 "p_u": float(out.pvalue_greater)}  # fmt: skip
            if out.k_opposite is not None:
                r["k_opp"] = float(out.k_opposite)
        self.stored["r"] = r

    @property
    def data(self) -> Optional[pd.DataFrame]:
        return None if self._steps is None else self._steps.data

    def run(self, line: str) -> bool:
        """Run one command. Returns ``False`` for a line that produces no
        output (a macro definition, ``xtset``, a data step, a setting)."""
        try:
            if _QUIET_BLOCK.match(line):
                self._quiet_blocks += 1
                return False
            if self._quiet_blocks and line.strip() == "}":
                self._quiet_blocks -= 1
                return False
            captured = _CAPTURE.match(line)
            if captured is not None:
                # `capture cmd`: an error Stata would raise (a variable that
                # is not there, `restore` without `preserve`) is swallowed;
                # a command this runner cannot translate still stops it
                try:
                    return self._run(captured.group(1))
                except MethodIncompatibility as exc:
                    if isinstance(exc.__cause__, StataExprError):
                        return False
                    raise
            quiet = _QUIETLY.match(line)
            if quiet is not None:
                line = quiet.group(1)
            return self._run(line)
        finally:
            if self._scratch and self._steps is not None:
                held = [c for c in self._scratch if c in self._steps.data.columns]
                if held:
                    self._steps.data = self._steps.data.drop(columns=held)
                self._scratch = []
            if self.stored.pop("random_draws", False):
                self.simulated = True
                warnings.warn(
                    "sp.stata: the random numbers come from numpy, not from "
                    "Stata's generator, so a simulated dataset differs from "
                    "Stata's draw by draw (same design, another sample).",
                    UserWarning,
                    stacklevel=3,
                )

    def _run(self, line: str) -> bool:
        import statspai as sp

        from ._stata import from_stata
        from ._stata_programs import program_line, run_simulate
        from ._stata_script import ScriptError, control_flow, panel_declaration

        handled_program = program_line(self, line)
        if handled_program is not None:
            return handled_program
        simulated = run_simulate(self, line)
        if simulated is not None:
            return simulated
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
            line = self._results_in_macros(line)
            if self._define_from_expression(line) or self._macros.define(line):
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
            self.stored["time_var"] = declared[1]
            self.stored["panel_var"] = declared[0]
            if self._steps is not None:
                # tsset / xtset leave the data sorted by panel and time
                keys = [k for k in declared if k and k in self._steps.data.columns]
                if keys and not self._steps.data.empty:
                    self._steps._sort(keys, None)
            return False
        if self._steps is not None:
            try:
                if self._steps.apply_label(line):
                    return False
            except StataExprError as exc:
                raise MethodIncompatibility(
                    f"sp.stata: cannot run {line!r}: {exc}.",
                    recovery_hint="Attach the labels in Python with "
                    "sp.label_var / sp.label_values and drop the line.",
                    diagnostics={"command": line},
                ) from exc
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
        fweight: Optional[str] = None
        try:
            handled = run_session_command(self, line)
            if handled is None:
                line, fweight = prepare_weights(self, line)
        except StataExprError as exc:
            raise MethodIncompatibility(
                f"sp.stata: cannot run {line!r}: {exc}.",
                recovery_hint="Do this step in pandas and pass the prepared "
                "DataFrame as data=.",
                diagnostics={"command": line},
            ) from exc
        if handled is not None:
            return handled
        data = self.data
        columns = None if data is None else list(data.columns)
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
        line = self._boundary_points(line)
        key = (line, tuple(columns or ()), self.panel)
        out = self._translations.get(key)
        if out is None:
            out = _with_panel(
                line, from_stata(line, columns=columns), self.panel, columns
            )
            if len(self._translations) < 512:
                self._translations[key] = out
        out = dict(out)
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
        if fweight is not None and run_data is not None and not chained:
            try:
                run_data = expand_frequency(run_data, fweight)
            except StataExprError as exc:
                raise MethodIncompatibility(
                    f"sp.stata: cannot run {line!r}: {exc}.",
                    recovery_hint="Check the frequency-weight variable.",
                    diagnostics={"command": line},
                ) from exc
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
            if out["tool"] == "wild_cluster_boot":
                # boottest bootstraps the clusters of the regression before it
                fitted = (getattr(self, "_last_call", None) or {}).get("arguments")
                fitted_cluster = (fitted or {}).get("cluster")
                asked = arguments.get("cluster")
                if asked is None and isinstance(fitted_cluster, str):
                    arguments["cluster"] = fitted_cluster
                elif asked is None or (
                    fitted_cluster is not None and asked != fitted_cluster
                ):
                    raise MethodIncompatibility(
                        f"sp.stata: cannot run {line!r}: "
                        + (
                            "the regression before it is not clustered and "
                            "the line names no cluster."
                            if asked is None
                            else f"it bootstraps {asked!r} but the regression "
                            f"before it is clustered on {fitted_cluster!r}."
                        ),
                        recovery_hint="Cluster the regression on one variable "
                        "(`, cluster(g)`), or call "
                        "sp.subcluster_wild_bootstrap for a bootstrap "
                        "cluster finer than the error cluster.",
                        diagnostics={"command": line, "translation": out},
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
        elif out["tool"] == "bitest" and "n" in arguments:
            # the immediate form: counts on the command line, no data
            self.output = fn(**arguments)
            self._store_r("bitest", arguments, None)
        else:
            if run_data is None:
                raise TypeError(f"sp.stata: {line!r} needs data=<DataFrame>.")
            is_teffects = re.match(r"\s*(?:\w+\s+)*teffects\b", line) is not None
            if is_teffects:
                try:
                    teffects_before(self, line, out, run_data)
                except StataExprError as exc:
                    raise MethodIncompatibility(
                        f"sp.stata: cannot run {line!r}: {exc}.",
                        recovery_hint="Drop the flagged observations "
                        "(`... if name == 0`) and run the command again.",
                        diagnostics={"command": line},
                    ) from exc
            self.output = fn(data=run_data, **arguments)
            if out["tool"] not in _DESCRIPTIVE_TOOLS:
                self.last = self.output
                self.last_data = run_data
                self._last_call = out
                self._store_estimates(self.output)
                if re.match(r"\s*(?:\w+\s*:\s*)*(?:qui\w*\s+)?xtreg\b", line):
                    xtreg_extras(self, out, run_data)
                if out["tool"] == "hdfe_ols":
                    absorbed_constant(self, out, run_data)
                if out["tool"] == "psmatch2":
                    psmatch2_after(self, run_data)
                if is_teffects:
                    try:
                        teffects_after(self, line, out, run_data)
                    except StataExprError as exc:
                        raise MethodIncompatibility(
                            f"sp.stata: cannot run {line!r}: {exc}.",
                            recovery_hint="Drop generate() and read the "
                            "matches from result.model_info['matched_data'].",
                            diagnostics={"command": line},
                        ) from exc
            else:
                self._store_r(str(out["tool"]), arguments, run_data)
        return True

    def _results_in_macros(self, line: str) -> str:
        """Write out `` `r(mean)' `` / `` `e(N)' ``: a stored result used as
        a macro. The value is the one the session holds from the command
        that left it; a result that is not there is left for the macro
        table to refuse."""

        def repl(m: "re.Match[str]") -> str:
            try:
                out = evaluate(m.group(1), pd.DataFrame({"_": [0.0]}), self.stored)
                return _macro_number(float(out[0]))
            except (StataExprError, TypeError, ValueError):
                return m.group(0)

        return _RESULT_MACRO.sub(repl, line)

    def _define_from_expression(self, line: str) -> bool:
        """``local name = exp`` / ``global name = exp`` with a numeric
        expression the session can evaluate (``r(mean)``, ``_b[x] * 2``,
        ``2010 + 5``). The macro holds the number as text, as in Stata.
        Anything else is left to the macro table, which records the value as
        unknown so that a later use is refused."""
        m = _MACRO_EXPRESSION.match(line.strip())
        if m is None:
            return False
        expr = m.group(3).strip()
        if not expr or expr.startswith(('"', '`"')):
            return False  # a string: the macro table stores the text
        from ._stata_script import ScriptError

        # Only what the session holds decides the value: stored results,
        # coefficients, scalars and arithmetic. An expression that reads the
        # data (`x[1]`, `_N`) is left to the macro table to refuse.
        if re.search(r"\b_[Nn]\b", expr):
            return False
        try:
            expr = self._macros.expand(expr)
            out = evaluate(expr, pd.DataFrame({"_": [0.0]}), self.stored)
        except (ScriptError, StataExprError):
            return False
        if out.dtype == object:
            return False
        number = float(out[0])
        table = (
            self._macros.globals
            if m.group(1).lower().startswith("g")
            else self._macros.locals
        )
        table[m.group(2)] = _macro_number(number)
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

    def _boundary_points(self, line: str) -> str:
        """Write values that ``rdms`` / ``rdmcplot`` keep in variables into
        the command.

        ``rdms y x1 x2 d, cvar(p1 p2)`` reads its boundary points from the
        leading non-missing values of ``p1`` and ``p2``; ``rdms y x,
        cvar(c) range(lo hi)`` its cutoffs and ranges the same way; and
        ``rdmcplot``'s ``pvar()`` / ``nbinsvar()`` / ... hold one setting
        per cutoff. The translation has no data, so the values are appended
        here as ``cutoff1()``, ``range1()``, ``pvec()`` and so on. Anything
        unexpected is left to the handler, which says what it needs.
        """
        data = self.data
        head = re.match(r"\s*(rdms|rdmcplot)\b", line)
        if head is None or data is None:
            return line

        def leading(name: str, as_text: bool = False) -> Optional[str]:
            if name not in data.columns:
                return None
            col = data[name].dropna()
            if as_text:
                col = col[col.astype(str).str.strip() != ""]
                return " ".join(str(v).strip() for v in col) or None
            try:
                return " ".join(repr(float(v)) for v in col) or None
            except (TypeError, ValueError):
                return None

        extra = []
        if head.group(1) == "rdms":
            cvar = re.search(r"\bc(?:var)?\(\s*(\w+)(?:\s+(\w+))?\s*\)", line)
            if cvar is None or "cutoff1(" in line:
                return line
            for i, name in enumerate(g for g in cvar.groups() if g):
                values = leading(name)
                if values is None:
                    return line
                extra.append(f"cutoff{i + 1}({values})")
            rng = re.search(r"\brange\(\s*(\w+)\s+(\w+)\s*\)", line)
            if rng is not None:
                for i, name in enumerate(rng.groups()):
                    values = leading(name)
                    if values is None:
                        return line
                    extra.append(f"range{i + 1}({values})")
        else:
            for var_opt, vec_opt, text in (
                ("pvar", "pvec", False),
                ("nbinsvar", "nbinsvec", False),
                ("nbinsrightvar", "nbinsrightvec", False),
                ("hvar", "hvec", False),
                ("binselectvar", "binselectvec", True),
            ):
                named = re.search(rf"\b{var_opt}\(\s*(\w+)\s*\)", line)
                if named is None or f"{vec_opt}(" in line:
                    continue
                values = leading(named.group(1), as_text=text)
                if values is not None:
                    extra.append(f"{vec_opt}({values})")
        return f"{line} {' '.join(extra)}" if extra else line

    def _predict(self, line: str) -> bool:
        """``predict [type] newvar [if] [, xb | residuals | leverage | pr]``.

        Stata predicts for every row whose regressors are observed, not only
        the estimation sample; so does this. Covered: the linear prediction,
        residuals and leverage after a linear fit, and the linear index or
        the probability (the default) after ``logit`` / ``probit``. A
        coefficient may be a plain column, an ``i.`` indicator or a product
        of those; with absorbed effects the line is refused.
        """
        from scipy import stats as _stats

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
        if (self._last_call or {}).get("tool") == "match":
            return self._predict_pscore(cmd)
        kinds = [_PREDICT_KINDS.get(k, k) for k in cmd.options if k]
        unknown = [k for k in kinds if k not in _PREDICT_KINDS.values()]
        if unknown or len(kinds) > 1:
            raise StataExprError(
                f"`predict` option(s) {sorted(cmd.options)} are not implemented; "
                "xb, residuals, leverage and pr are"
            )
        names = list(cmd.varlist)
        double = False
        if len(names) == 2 and names[0] in ("float", "double"):
            double = names[0] == "double"
            names = names[1:]
        if len(names) != 1:
            raise StataExprError("expected `predict [type] newvar`")
        tool = self._last_call.get("tool")
        arguments = self._last_call.get("arguments") or {}
        formula = str(arguments.get("formula") or "")
        binary = tool in ("logit", "probit")
        if tool not in ("regress", "ivreg", "logit", "probit") or "~" not in formula:
            raise StataExprError(
                f"`predict` after sp.{tool} is not implemented (linear, logit "
                "and probit models only)"
            )
        kind = kinds[0] if kinds else ("pr" if binary else "xb")
        if (kind == "pr") != binary and kind in ("pr", "residuals", "leverage"):
            raise StataExprError(f"`predict, {kind}` does not apply after sp.{tool}")
        data = self._steps.data
        params = getattr(self.last, "params")
        total: np.ndarray = np.zeros(len(data))
        design = []
        for term, beta in params.items():
            if term in ("Intercept", "const", "_cons"):
                column = np.ones(len(data))
            elif term in data.columns:
                column = data[term].to_numpy(dtype=float, na_value=np.nan)
            else:
                built = _term_column(str(term), data)
                if built is None:
                    raise StataExprError(
                        f"`predict`: coefficient {term!r} is not built from "
                        "columns of the data (plain columns, i. indicators "
                        "and their products are covered)"
                    )
                column = built
            design.append(column)
            total = total + float(beta) * column
        if kind == "residuals":
            outcome = formula.split("~", 1)[0].strip()
            if outcome not in data.columns:
                raise StataExprError(f"`predict`: outcome {outcome!r} is not a column")
            total = data[outcome].to_numpy(dtype=float, na_value=np.nan) - total
        elif kind == "leverage":
            info = getattr(self.last, "data_info", None) or {}
            X = info.get("X")
            if X is None or arguments.get("weights") or tool != "regress":
                raise StataExprError(
                    "`predict, leverage` needs an unweighted sp.regress fit"
                )
            X = np.asarray(X, dtype=float)
            rows = np.column_stack(design)
            if X.shape[1] != rows.shape[1]:
                raise StataExprError("`predict, leverage`: design does not match")
            inverse = np.linalg.pinv(X.T @ X)
            total = np.einsum("ij,jk,ik->i", rows, inverse, rows)
        elif kind == "pr":
            link = _stats.logistic.cdf if tool == "logit" else _stats.norm.cdf
            total = link(total)
        if cmd.if_cond or cmd.in_range:
            keep = row_mask(data, cmd.if_cond, cmd.in_range, self.stored)
            total = np.where(keep, total, np.nan)
        self._steps.add_column(names[0], total, double=double)
        return True

    def _predict_pscore(self, cmd: Any) -> bool:
        """``predict newvar, ps [tlevel(#)]`` after ``teffects psmatch``:
        the estimated propensity score (of treatment level 1 by default)."""
        options = dict(cmd.options)
        level = str(options.pop("tlevel", "1") or "1").strip()
        if set(options) != {"ps"} or level not in ("0", "1") or len(cmd.varlist) != 1:
            raise StataExprError(
                "after teffects psmatch only `predict newvar, ps [tlevel(0|1)]` "
                "is implemented"
            )
        info = getattr(self.last, "model_info", None) or {}
        matched = info.get("matched_data")
        data = self.last_data
        if matched is None or "_pscore" not in matched or data is None:
            raise StataExprError("`predict, ps` follows `teffects psmatch`")
        if len(matched) != len(data) or self._steps is None:
            raise StataExprError("`predict, ps`: the fit dropped rows")
        ps = matched["_pscore"].to_numpy(dtype=float)
        column = pd.Series(np.nan, index=self._steps.data.index)
        shared = data.index.intersection(column.index)
        if len(shared) != len(data):
            raise StataExprError("`predict, ps`: the data changed since the fit")
        column.loc[data.index] = ps if level == "1" else 1.0 - ps
        self._steps.add_column(cmd.varlist[0], column.to_numpy(), double=False)
        return True
