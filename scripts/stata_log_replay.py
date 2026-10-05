"""Replay a Stata log through ``sp.stata`` and compare every printed number.

``stata_corpus_scan.py`` asks whether a do-file's commands are *translated*.
This asks the harder question: do they give Stata's *numbers*? A Stata log
holds the commands that ran and what they printed, so it is its own answer
key. Each ``. command`` line is run through one ``StataSession`` (data
steps, ``if`` qualifiers, estimation, ``test`` ...), and every coefficient,
standard error, test statistic, p-value, t test, summary mean and displayed
scalar in the log is compared with what StatsPAI returns.

Usage::

    python scripts/stata_log_replay.py LOGS... --data DIR [--csv out.csv] [--strict]

``LOGS`` are ``.log`` files or folders; ``--data`` is searched recursively
for the ``.dta`` files the logs ``use`` (matched by file name).

A number counts as matched when it is within two units of the last digit
Stata printed. Each line of the report is one of

* ``ok``         the printed number is reproduced
* ``DIFF``       it is not; the first thing to look at
* ``NOT RUN``    ``sp.stata`` refused the command (the reason is given)
* ``no output``  the command ran but the log's number has no counterpart in
                 the result (``xtreg, fe`` prints ``_cons``; ``sp.feols``
                 absorbs it)
* ``random``     the data in memory were simulated (``rnormal()`` ...);
                 numpy's draws are not Stata's, so the number is not
                 comparable

Use it as a probe: a ``DIFF`` is either a translation that asks for the
wrong convention or an estimator bug, and has to be traced to which. Do not
fix what it finds by special-casing the file it was found in.
"""

from __future__ import annotations

import argparse
import contextlib
import re
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import statspai as sp
from statspai.agent._translation._stata_expr import StataExprError, coefficient_key
from statspai.agent._translation._stata_run import StataSession

NUM = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
ESTIMATION = {
    "reg", "regress", "xtreg", "areg", "probit", "logit", "poisson", "nbreg",
    "ivreg", "ivreg2", "ivregress", "tobit", "newey", "reghdfe", "prais",
    "cnsreg", "nl", "qreg",
}  # fmt: skip
SUMMARIZE = {"su", "sum", "summ", "summarize"}
#: commands compared number by number without naming the numbers
BAG = {
    "tabulate", "tabulat", "tabula", "tabul", "tabu", "tab", "ta", "tab1", "tab2",
    "mean", "proportion", "total", "ratio", "svy", "svy,", "ameans", "centile",
    "cii", "ci", "duplicates", "ranksum", "signrank", "kwallis", "spearman", "ktau",
    "ksmirnov", "median", "robvar", "oneway", "tabstat", "table", "misstable",
    "lrtest", "linktest", "by", "bysort", "bys", "count", "nestreg", "sdtest",
    "prtest", "prtesti", "bitest", "bitesti", "ztest", "sktest", "swilk", "cc",
    "cs", "tabodds", "lroc", "correlate", "corr", "pwcorr",
}  # fmt: skip


def _numbers_in(out: Any, depth: int = 0) -> List[float]:
    """Every finite number an output holds, whatever its shape."""
    found: List[float] = []
    if depth > 6 or out is None or isinstance(out, (str, bytes)):
        return found
    if isinstance(out, (bool, np.bool_)):
        return found
    if isinstance(out, (int, float, np.integer, np.floating)):
        return [float(out)] if np.isfinite(out) else found
    if isinstance(out, pd.DataFrame):
        values = out.apply(pd.to_numeric, errors="coerce").to_numpy(dtype=float)
        found.extend(float(v) for v in values.ravel() if np.isfinite(v))
        found.extend(_numbers_in(dict(out.attrs), depth + 1))
        found.extend(_numbers_in(getattr(out, "info", None), depth + 1))
        return found
    if isinstance(out, pd.Series):
        values = pd.to_numeric(out, errors="coerce").to_numpy(dtype=float)
        return [float(v) for v in values if np.isfinite(v)]
    if isinstance(out, np.ndarray):
        if out.dtype.kind in "fiu" and out.size <= 100_000:
            return [float(v) for v in out.ravel() if np.isfinite(v)]
        return found
    if isinstance(out, dict):
        for value in out.values():
            found.extend(_numbers_in(value, depth + 1))
        return found
    if isinstance(out, (list, tuple)):
        for value in out[:10_000]:
            found.extend(_numbers_in(value, depth + 1))
        return found
    for attr in ("summary_frame", "table", "tables"):
        held = getattr(out, attr, None)
        if held is not None and not callable(held):
            found.extend(_numbers_in(held, depth + 1))
    if hasattr(out, "__dict__"):
        found.extend(_numbers_in(vars(out), depth + 1))
    return found


# ------------------------------------------------------------ the data
@contextlib.contextmanager
def _readable(path: Path) -> Any:
    """``path``, or a temporary copy pandas can read when the file is in
    Stata 7's format (110), which pandas before 3.0 refuses."""
    from statspai.utils._dta_layout import release_110_as_111

    old = release_110_as_111(path)
    try:
        yield old or path
    finally:
        if old is not None:
            Path(old).unlink()


def read_dta(path: Path) -> pd.DataFrame:
    """A .dta file as Stata holds it: dates are counts of periods."""
    with _readable(path) as source:
        return pd.read_stata(source, convert_categoricals=False, convert_dates=False)


def dta_value_labels(path: Path) -> Dict[str, Dict[str, Any]]:
    """variable -> {label text: code}: Stata prints a factor level by its
    label, the session names it by its code."""
    out: Dict[str, Dict[str, Any]] = {}
    with _readable(path) as source, pd.io.stata.StataReader(source) as reader:
        sets = reader.value_labels()
        names = list(getattr(reader, "_varlist", []))
        attached = list(getattr(reader, "_lbllist", []))
    for var, label_name in zip(names, attached):
        if label_name in sets:
            # Stata truncates a long label in the table; keep both spellings
            mapping: Dict[str, Any] = {}
            for code, text in sets[label_name].items():
                mapping[str(text)] = code
                if str(text).split():  # SOEP labels its missing codes ""
                    mapping[str(text).split()[0]] = code
            out[var] = mapping
    return out


def dta_extended_missing(path: Path) -> List[str]:
    """The variables of a .dta file that hold an extended missing value
    (``.a`` ... ``.z``). The frame keeps them as NaN like ``.``; the session
    needs to know where telling them apart would matter."""
    from pandas.io.stata import StataMissingValue

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        coded = pd.read_stata(
            path, convert_categoricals=False, convert_dates=False, convert_missing=True
        )
    out = []
    for name in coded.columns:
        col = coded[name]
        if col.dtype != object:
            continue
        if any(isinstance(v, StataMissingValue) and str(v) != "." for v in col):
            out.append(str(name))
    return out


def dta_tsset(path: Path) -> Optional[str]:
    """The ``tsset`` / ``xtset`` declaration saved in a .dta file, if any.

    Stata keeps it as characteristics of ``_dta`` (``_TStvar``,
    ``_TSpanel``); pandas does not read characteristics, so the two fields
    are located in the file directly. A characteristic is stored as the
    owner's name and its own name in two fixed-width fields (33 bytes up to
    format 117, 129 from 118) followed by the text.
    """
    blob = path.read_bytes()

    def field(name: bytes) -> Optional[str]:
        at = blob.find(name + b"\x00")
        while at >= 0:
            for width in (129, 33):
                if blob[at - width : at - width + 5] == b"_dta\x00":
                    text = blob[at + width :].split(b"\x00", 1)[0]
                    return text.decode("utf-8", "replace").strip() or None
            at = blob.find(name + b"\x00", at + 1)
        return None

    time, panel = field(b"_TStvar"), field(b"_TSpanel")
    if time is None:
        return None
    return f"xtset {panel} {time}" if panel else f"tsset {time}"


# ------------------------------------------------------------- the log
_NUMBERED = re.compile(r"^\s*\d+\.(?:\s+(.*\S))?\s*$")


def _do_file_commands(path: Path) -> List[str]:
    """The commands of the do-file beside a log, for the blocks whose body
    the log does not show (``quietly { }``, ``quietly forvalues``)."""
    source = path.with_suffix(".do")
    if not source.is_file():
        return []
    from statspai.agent._translation._stata_script import split_commands

    return split_commands(source.read_text(encoding="utf-8", errors="replace"))


def _hidden_body(commands: List[str], start: int, opener: str) -> Tuple[List[str], int]:
    """Body of the block ``opener`` opens, found in the do-file from
    ``start`` on: its lines up to and including the closing brace."""
    want = " ".join(opener.split())
    for at in range(start, len(commands)):
        if " ".join(commands[at].split()) != want:
            continue
        depth, body = 1, []
        for line in commands[at + 1 :]:
            bare = line.strip()
            if bare.endswith("{") and not bare.startswith("}"):
                depth += 1
            elif bare == "}":
                depth -= 1
            body.append(line)
            if depth == 0:
                return body, at + 1 + len(body)
        return [], start
    return [], start


def parse_log(path: Path) -> List[Tuple[str, List[str]]]:
    """``[(command, lines it printed), ...]`` in the order they ran.

    A block (``forvalues ... {``) is one command. Stata echoes its body as
    numbered lines, which stay at the head of what it printed; a body line
    that follows a comment is echoed with a dot instead of a number and is
    numbered here. A block the log does not show the body of (it ran
    quietly) takes its body from the do-file next to the log, if there is
    one.
    """
    lines = path.read_text(encoding="latin-1").splitlines()
    out: List[Tuple[str, List[str]]] = []
    cmd: Optional[str] = None
    buf: List[str] = []
    depth = 0  # braces open in the block being echoed
    echoed = False  # a numbered line of that block has been seen
    for ln in lines:
        if depth > 0:
            numbered = _NUMBERED.match(ln)
            dotted = re.match(r"^\.\s+(.*\S)\s*$", ln) if numbered is None else None
            content = None
            if numbered is not None:
                content = numbered.group(1) or ""
            elif dotted is not None:
                content = dotted.group(1)
            elif ln.strip() in ("", "."):
                continue
            if ln.startswith("> ") and buf:
                # a body line wrapped by the log, or continued with ///
                buf[-1] = re.sub(r"\s*///.*$", "", buf[-1]) + " " + ln[2:].strip()
                continue
            echoed = echoed or numbered is not None
            if content is not None and echoed:
                bare = content.strip()
                if bare and not bare.startswith(("*", "//")):
                    buf.append(f"  0. {bare}")
                    if bare.endswith("{") and not bare.startswith("}"):
                        depth += 1
                    elif bare == "}":
                        depth -= 1
                continue
            depth = 0  # no echo: the body ran quietly, or this is output
        if ln.startswith(". "):
            if cmd is not None:
                out.append((cmd, buf))
            cmd, buf = ln[2:].strip(), []
            if cmd.rstrip().endswith("{"):
                depth, echoed = 1, False
        elif ln.startswith("> ") and cmd is not None and not buf:
            # a wrapped command line; `///` is the do-file's continuation mark
            cmd = re.sub(r"\s*///.*$", "", cmd) + " " + ln[2:].strip()
            depth = 1 if cmd.rstrip().endswith("{") else 0
        elif cmd is not None:
            buf.append(ln)
    if cmd is not None:
        out.append((cmd, buf))
    cleaned = []
    for c, b in out:
        c = c.rstrip(";").strip()
        if c and not c.startswith(("*", "//", "/*")):
            cleaned.append((c, b))
    commands = _do_file_commands(path)
    at = 0
    for k, (c, b) in enumerate(cleaned):
        if commands and c.endswith("{"):
            # every block moves the place in the do-file past its body, so a
            # block nested in an echoed one is not taken for a later one
            body, at = _hidden_body(commands, at, c)
            following = cleaned[k + 1][0] if k + 1 < len(cleaned) else ""
            if body and " ".join(following.split()) == " ".join(body[0].split()):
                continue  # an `if` block: the log echoes its lines as commands
            if body and not (b and _NUMBERED.match(b[0])):
                cleaned[k] = (c, [f"  0. {line.strip()}" for line in body] + b)
    return cleaned


def unit_of_last_digit(token: str) -> float:
    """Size of one unit in the last digit of a printed number."""
    mantissa, _, exponent = token.lower().partition("e")
    decimals = len(mantissa.split(".")[1]) if "." in mantissa else 0
    return 10.0 ** (int(exponent or 0) - decimals)


class Printed:
    """A number as Stata printed it: its value and its precision."""

    def __init__(self, token: str) -> None:
        self.value = float(token)
        self.unit = unit_of_last_digit(token)

    def matches(self, ours: float, rtol: float = 0.0) -> bool:
        # half a unit is rounding; the rest is float storage and, for
        # likelihood estimators, where two optimisers stop. ``rtol`` is for
        # the standard errors of a model Stata fits with `ml`: its Hessian is
        # evaluated where its own iterations stopped.
        gap = abs(ours - self.value)
        if abs(self.value) < 1e-12 and abs(ours) < 1e-12:
            return True  # zero on both sides, printed with its rounding error
        return bool(gap <= 2.0 * self.unit or gap <= rtol * abs(self.value))


_TS_ROW = re.compile(r"^((?:[LFD]\d*)+)\.$")
_TS_ONE = re.compile(r"([LFD])(\d*)")


def coefficient_table(
    buf: List[str], labels: Optional[Dict[str, Dict[str, Any]]] = None
) -> Dict[str, Tuple[Printed, Printed]]:
    """Coefficient rows by the name ``sp.stata`` gives the same term.

    Stata prints a time-series operator or a factor level under the
    variable it belongs to (``temp |`` then ``L1. |``); the session calls
    those columns ``temp_L1`` and ``C(g)[T.2]``. A level is printed by its
    value label when the variable has one; ``labels`` maps variable ->
    {label: code} to undo that.
    """
    labels = labels or {}
    rows: Dict[str, Tuple[Printed, Printed]] = {}
    group: Optional[str] = None
    for ln in buf:
        head = re.match(r"^\s*(\S+)\s*\|\s*$", ln)
        if head:
            group = head.group(1)
            continue
        if re.match(r"^\s*-+\+-+\s*$", ln):
            continue
        # a value label may hold blanks ("Got a transfer")
        m = re.match(
            rf"^\s*(\S(?:[^|]*\S)?)\s*\|\s*({NUM})\s+({NUM})\s+({NUM})\s+({NUM})", ln
        )
        if not m:
            if not ln.strip():
                group = None
            continue
        name = m.group(1)
        ts = _TS_ROW.match(name)
        if group is not None and name == "--.":
            name = group
        elif group is not None and ts:
            ops = "".join(f"{k}{n or 1}" for k, n in _TS_ONE.findall(ts.group(1)))
            name = f"{group}_{ops}"
        elif group is not None and "#" in group and _crossed(group):
            # `g#c.x |` or `d#t |`, then one row per level (`1#Quarter 4`)
            factors, rest = _crossed(group)
            # unlabelled levels of crossed factors are printed "1 1"
            levels = name.split("#") if "#" in name else name.split()
            if len(levels) != len(factors):
                levels = [name]
            codes = [
                lv if re.fullmatch(r"\d+", lv) else labels.get(f, {}).get(lv)
                for f, lv in zip(factors, levels)
            ]
            if len(levels) == len(factors) and None not in codes:
                cells = [f"C({f})[T.{c}]" for f, c in zip(factors, codes)]
                name = ":".join(cells + rest)
            else:
                group = None
                if re.fullmatch(r"c\.\w+(?:#c\.\w+)+", name):
                    name = ":".join(part[2:] for part in name.split("#"))
        elif group is not None and re.fullmatch(r"\d+", name):
            name = f"C({group})[T.{name}]"
        elif group is not None and name in labels.get(group, {}):
            name = f"C({group})[T.{labels[group][name]}]"
        else:
            group = None
            if "#" in name and re.fullmatch(r"c\.\w+(?:#c\.\w+)+", name):
                # c.x#c.z is the interaction the session calls x:z
                name = ":".join(part[2:] for part in name.split("#"))
        rows[name] = (Printed(m.group(2)), Printed(m.group(3)))
    return rows


_LEVEL = re.compile(r"C\((\w+)\)\[T\.(-?\d+)(?:\.0)?\]")


def _crossed(group: str) -> Optional[Tuple[List[str], List[str]]]:
    """``d#t#c.x`` -> (["d", "t"], ["x"]); None without a factor."""
    parts = group.split("#")
    factors = [p for p in parts if not p.startswith("c.")]
    if not factors or not all(re.fullmatch(r"(?:i\.)?\w+", f) for f in factors):
        return None
    names = [f.split(".")[-1] for f in factors]
    return names, [p[2:] for p in parts if p.startswith("c.")]


#: c.x#c.x is fitted as ``I(x ** 2)``
_POWER = re.compile(r"I\((\w+) ?\*\* ?(\d+)\)")


def stata_term(name: str) -> str:
    """A coefficient name in Stata's spelling: ``C(g)[T.2.0]:C(t)[T.3]`` and
    ``2.g#3.t`` both become ``2.g#3.t``, so a factor level generated as a
    float is found under the integer Stata prints."""
    name = _POWER.sub(lambda m: ":".join([m.group(1)] * int(m.group(2))), name)
    return coefficient_key(name)


def header_statistics(buf: List[str]) -> Dict[str, Printed]:
    """N, F, R-squared, adjusted R-squared and root MSE of a `regress` header."""
    text = "\n".join(buf)
    out: Dict[str, Printed] = {}
    for label, pattern in (
        ("N", r"Number of obs\s*=\s*([\d,]+)"),
        ("F", rf"\bF\(\s*\d+,\s*\d+\)\s*=\s*({NUM})"),
        ("r2", rf"(?<!Adj )R-squared\s*=\s*({NUM})"),
        ("r2_a", rf"Adj R-squared\s*=\s*({NUM})"),
        ("rmse", rf"Root MSE\s*=\s*({NUM})"),
    ):
        m = re.search(pattern, text)
        if m:
            out[label] = Printed(m.group(1).replace(",", ""))
    return out


# ------------------------------------------------------------- the replay
class Report:
    def __init__(self) -> None:
        self.rows: List[Dict[str, Any]] = []

    def add(self, file: str, cmd: str, what: str, status: str, **kw: Any) -> None:
        self.rows.append(
            {"file": file, "command": cmd, "what": what, "status": status, **kw}
        )

    #: set by the replay while the data in memory hold random draws
    simulated = False

    def number(
        self,
        file: str,
        cmd: str,
        what: str,
        printed: Printed,
        ours: Any,
        rtol: float = 0.0,
    ) -> None:
        if self.simulated:
            # numpy's draws are not Stata's: same design, another sample
            self.add(file, cmd, what, "random", stata=printed.value)
            return
        if ours is None or (isinstance(ours, float) and np.isnan(ours)):
            self.add(file, cmd, what, "no output", stata=printed.value)
            return
        ours = float(ours)
        status = "ok" if printed.matches(ours, rtol) else "DIFF"
        self.add(file, cmd, what, status, stata=printed.value, ours=ours,
                 units_off=abs(ours - printed.value) / printed.unit)  # fmt: skip


class Replay:
    def __init__(self, name: str, data_dirs: List[Path], report: Report) -> None:
        self.name = name
        self.data_dirs = data_dirs
        self.report = report
        self.session: Optional[StataSession] = None
        #: variable -> {value label: code} of the file in memory
        self.labels: Dict[str, Dict[str, Any]] = {}

    # -- helpers ----------------------------------------------------------
    def _find(self, cmd: str) -> Path:
        m = re.search(r'"([^"]+)"', cmd) or re.match(
            r"(?:sysuse|use|append\s+using)\s+(\S+)", cmd
        )
        if m is None:
            raise FileNotFoundError("cannot read the file name")
        stem = Path(m.group(1).split(",")[0]).name
        if not stem.lower().endswith(".dta"):
            stem += ".dta"
        for root in self.data_dirs:
            hits = [p for p in root.rglob("*.dta") if p.name.lower() == stem.lower()]
            if hits:
                return hits[0]
        raise FileNotFoundError(f"{stem} is not under --data")

    def _session(self, frame: Optional[pd.DataFrame]) -> StataSession:
        """A session on ``frame`` that keeps what a new dataset does not
        erase in Stata: macros, scalars, matrices, programs, saved data."""
        new = StataSession(frame)
        old = self.session
        if old is not None:
            new._macros.locals = old._macros.locals
            new._macros.globals = old._macros.globals
            new.programs = old.programs
            new.files = old.files
            # `use` replaces the data of the current frame only
            new.frames, new.frame, new.links = old.frames, old.frame, old.links
            new.svy = getattr(old, "svy", None) if frame is None else None
            new._temp_count = old._temp_count
            for key in ("scalars", "matrices", "rng"):
                if key in old.stored:
                    new.stored[key] = old.stored[key]
        new.file_loader = self._file
        return new

    def _file(self, name: str) -> Optional[pd.DataFrame]:
        """A .dta file under --data, for `merge ... using` and `append`."""
        for root in self.data_dirs:
            for path in root.rglob("*.dta"):
                if path.stem.lower() == name.lower():
                    return read_dta(path)
        return None

    def _loop(self, cmd: str, buf: List[str]) -> None:
        """A block the log echoes as numbered lines (``  2.  replace ...``):
        hand the opener and the body to the session, which runs it."""
        assert self.session is not None
        if not any(_NUMBERED.match(ln) for ln in buf):
            # an `if` block of a do-file: the log echoes its lines as
            # commands of their own, which feed the block one by one
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                self.session.run(cmd)
            return
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.session.run(cmd)
            try:
                for ln in buf:
                    m = _NUMBERED.match(ln)
                    if m is None:
                        break
                    if m.group(1):
                        self.session.run(m.group(1))
            finally:
                self.session._flow = None
                self.session._quiet_blocks = 0

    def _load(self, cmd: str) -> None:
        path = self._find(cmd)
        frame = read_dta(path)
        frame.attrs["_ext_missing"] = dta_extended_missing(path)
        with pd.io.stata.StataReader(path) as labelled:
            frame.attrs["_labels"] = dict(labelled.variable_labels())
            frame.attrs["_data_label"] = labelled.data_label
        self.labels = dta_value_labels(path)
        # code -> text per variable, where `decode` looks for it
        texts: Dict[str, Dict[Any, str]] = {}
        with _readable(path) as source, pd.io.stata.StataReader(source) as reader:
            sets = reader.value_labels()
            attached = dict(zip(getattr(reader, "_varlist", []),
                                getattr(reader, "_lbllist", [])))  # fmt: skip
        for var, set_name in attached.items():
            if set_name in sets and var in frame.columns:
                texts[var] = {int(k): str(v) for k, v in sets[set_name].items()}
        if texts:
            frame.attrs["_value_labels"] = texts
        self.session = self._session(frame)
        # a .dta file remembers its `tsset` / `xtset`
        declared = dta_tsset(path)
        if declared is not None:
            self.session.run(declared)

    def _import_delimited(self, cmd: str) -> None:
        """``import delimited "file.csv"``: Stata works out the delimiter
        and lower-cases the variable names (``case(lower)``, its default)."""
        m = re.match(r"import\s+delim\w*\s+(?:.*?\s)?using\s+\"?([^\s,\"]+)", cmd)
        m = m or re.match(r"import\s+delim\w*\s+\"?([^\s,\"]+)", cmd)
        if m is None:
            raise FileNotFoundError("cannot read the file name")
        name = Path(m.group(1)).name.lower()
        hits = [
            p for root in self.data_dirs for p in root.rglob("*")
            if p.is_file() and p.name.lower() == name
        ]  # fmt: skip
        if not hits:
            raise FileNotFoundError(f"{name} is not under --data")
        options = cmd.split(",", 1)[1] if "," in cmd.split('"')[-1] else ""
        with hits[0].open(encoding="utf-8", errors="replace") as fh:
            head = fh.readline()
        named = re.search(r"delim\w*\(\s*\"?(.+?)\"?\s*[,)]", options)
        if named:
            sep = {"tab": "\t", "\\t": "\t", "comma": ","}.get(
                named.group(1), named.group(1)
            )
        else:
            sep = max(",;\t|", key=head.count)
        first = head.rstrip("\r\n").split(sep)
        numeric = [bool(re.fullmatch(r"\s*-?[\d.]+\s*", cell)) for cell in first]
        named = re.search(r"varn\w*\(\s*(\w+)\s*\)", options)
        # a first row with a number in it is data, not names (Stata's rule)
        header = (
            0 if (named and named.group(1) != "nonames") or not any(numeric) else None
        )
        frame = pd.read_csv(hits[0], sep=sep, low_memory=False, header=header)
        listed = re.match(r"import\s+delim\w*\s+(.+?)\s+using\s", cmd)
        if listed:
            frame.columns = listed.group(1).split()[: frame.shape[1]]
        elif header is None:
            frame.columns = [f"v{k + 1}" for k in range(frame.shape[1])]
        if not re.search(r"case\(\s*(preserve|upper)", options):
            frame.columns = [str(c).lower() for c in frame.columns]
        self.session = self._session(frame)
        self.labels = {}

    def _input(self, cmd: str, buf: List[str]) -> None:
        """``input y w`` with the rows the log echoes as ``  1. 34 1``."""
        names = cmd.split()[1:]
        rows = []
        for ln in buf:
            m = re.match(r"^\s*\d+\.\s+(.*\S)\s*$", ln)
            if not m or m.group(1).strip() == "end":
                continue
            cells = m.group(1).split()
            if len(cells) != len(names):
                raise StataExprError("input: a row does not match the variable list")
            rows.append([np.nan if c == "." else float(c) for c in cells])
        if any(
            n.startswith("str") or not re.match(r"[A-Za-z_]\w*\Z", n) for n in names
        ):
            raise StataExprError("input: only numeric variables are read")
        self.session = self._session(pd.DataFrame(rows, columns=names))
        self.labels = {}

    def _append(self, cmd: str) -> None:
        assert self.session is not None
        if "," in cmd:
            raise FileNotFoundError("options: the session reads the file itself")
        self.session.append(read_dta(self._find(cmd)))

    # -- one command ------------------------------------------------------
    def run(self, cmd: str, buf: List[str]) -> None:
        def first_word(text: str) -> str:
            return re.split(r"[\s,]", text.strip(), maxsplit=1)[0].lower()

        opener = cmd.rstrip().endswith("{")
        if opener and (
            (buf and _NUMBERED.match(buf[0])) or re.match(r"\s*(if|else)\b", cmd)
        ):
            if self.session is None:
                self.session = self._session(None)
            self.report.simulated = bool(self.session.simulated)
            try:
                self._loop(cmd, buf)
            except Exception as exc:  # noqa: BLE001 - reported, not raised
                reason = str(exc).split("\n")[0]
                self.report.add(self.name, cmd, "-", "NOT RUN", reason=reason[:200])
            return
        word = first_word(cmd)
        quiet = False
        captured = False
        while word in ("qui", "quietly", "cap", "capture", "noi", "noisily"):
            captured = captured or word.startswith("cap")
            cmd = cmd.split(None, 1)[1]
            word = first_word(cmd)
            quiet = True
        if cmd.strip() in ("{", "}"):
            if cmd.strip() == "}" and getattr(self.session, "_flow", None) is not None:
                # the end of an `if` block whose lines were echoed one by one
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    self.session.run("}")
            return  # `quietly {`: the log does not echo what the block ran
        self.report.simulated = bool(
            self.session is not None and self.session.simulated
        )
        try:
            self._dispatch(cmd, word, buf, quiet)
        except (KeyError, FileNotFoundError, StataExprError) as exc:
            self.report.add(self.name, cmd, "-", "NOT RUN", reason=str(exc)[:200])
        except Exception as exc:  # noqa: BLE001 - every refusal belongs in the report
            if captured and isinstance(exc.__cause__, StataExprError):
                return  # an error `capture` swallows in Stata too
            reason = str(exc).split("\n")[0]
            # a command Stata itself stops at (run under `capture`): stopping
            # too is the faithful outcome
            both_stop = quiet and "Stata stops here" in reason
            self.report.add(
                self.name, cmd, "stops as Stata does" if both_stop else "-",
                "ok" if both_stop else "NOT RUN", reason=reason[:200],
            )  # fmt: skip

    def _dispatch(self, cmd: str, word: str, buf: List[str], quiet: bool) -> None:
        if word in ("use", "sysuse"):
            try:
                self._load(cmd)
            except FileNotFoundError:
                # not a file under --data: a dataset the session saved
                if self.session is None:
                    raise
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    self.session.run(cmd)
            return
        if word == "import" and re.match(r"import\s+delim", cmd):
            self._import_delimited(cmd)
            return
        if word == "input":
            self._input(cmd, buf)
            return
        if word in ("cd", "do", "exit", "#", "#delimit", "vce"):
            return
        if self.session is None:
            self.session = self._session(None)
        if word == "append":
            try:
                self._append(cmd)
            except FileNotFoundError:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    self.session.run(cmd)
            return
        if word == "program" and "drop" not in cmd.split()[:3]:
            # the log echoes the body as numbered lines: "  1.   drop _all"
            self.session.run(cmd)
            for ln in buf:
                m = re.match(r"^\s*\d+\.\s+(.*\S)\s*$", ln)
                if m:
                    self.session.run(m.group(1))
            return
        if word in ("display", "dis", "di"):
            self._display(cmd, buf)
            return
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            produced = self.session.run(cmd)
        if not produced:
            return
        out = self.session.output
        if word in SUMMARIZE:
            self._summarize(cmd, buf, out)
        elif word == "estat":
            self._estat(cmd, buf, out)
        elif word == "margins":
            if isinstance(out, pd.DataFrame) and out.attrs.get("session_margins"):
                self._bag(cmd, [ln for ln in buf if "_at:" not in ln], out)
            else:
                self._margins(cmd, buf, out)
        elif word == "hausman":
            self._hausman(cmd, buf, out)
        elif word in PANEL and not quiet:
            PANEL[word](self, cmd, buf, out)
        elif word in ("corrgram", "wntestq"):
            self._corrgram(cmd, buf, out, word)
        elif word in TIME_SERIES and not quiet:
            TIME_SERIES[word](self, cmd, buf, out)
        elif word.startswith("est") and isinstance(out, pd.DataFrame) and "AIC" in out:
            _cmp_estimates_stats(self, cmd, buf, out)
        elif _RESAMPLED.search(cmd):
            _cmp_resample(self, cmd, buf, out)
        elif word == "irf" and isinstance(out, pd.DataFrame):
            _cmp_irf(self, cmd, buf, out)
        elif word in MULTIVARIATE and not quiet:
            MULTIVARIATE[word](self, cmd, buf, out)
        elif word == "test":
            self._test(cmd, buf, out, quiet)
        elif word == "ttest":
            self._ttest(cmd, buf, out)
        elif word in ESTIMATION and not quiet:
            self._estimates(cmd, buf, out, word)
        elif word == "linktest" and not quiet:
            # the refit's coefficient table; e() stays that of the model
            self._bag(cmd, [ln for ln in buf if re.match(r"\s*(_hat|_hatsq|_cons) ", ln)],
                      {"b": dict(out.params), "se": dict(out.std_errors),
                       "t": dict(out.params / out.std_errors),
                       "ci": out.conf_int() if hasattr(out, "conf_int") else None})  # fmt: skip
        elif (word in BAG or word.rstrip(":") in BAG) and not quiet:
            self._bag(cmd, buf, out)

    def _bag(self, cmd: str, buf: List[str], out: Any) -> None:
        """Every number Stata printed in a table (after the ``|``) or after
        an ``=`` must be among the numbers of our output. For commands whose
        output is a table of statistics rather than named coefficients."""
        ours = np.array(sorted(_numbers_in(out)), dtype=float)
        position = 0
        for ln in buf:
            if ln.startswith((". ", "> ")) or re.match(r"^\s*(Note|Key)\b", ln):
                continue
            parts = []
            if "|" in ln:
                parts.append(ln.split("|", 1)[1].replace("|", " "))
            else:
                parts.extend(piece for piece in re.split(r"=", ln)[1:])
            for part in parts:
                if "|" in ln and re.search(r"[A-Za-z]{2,}", part):
                    continue  # a header row: the labels of the columns
                part = re.sub(r"(?<=\d),(?=\d{3}\b)", "", part)
                part = re.sub(r"[A-Za-z_]\w*\([^)]*\)", " ", part)  # chi2(7), F(1, 528)
                for token in re.findall(rf"(?<![\w.]){NUM}(?![\w%])", part):
                    if token in ("-", "+", "."):
                        continue
                    printed = Printed(token)
                    position += 1
                    if ours.size == 0:
                        self.report.number(
                            self.name, cmd, f"#{position}", printed, None
                        )
                        continue
                    nearest = float(ours[np.argmin(np.abs(ours - printed.value))])
                    self.report.number(self.name, cmd, f"#{position}", printed, nearest)

    def _estimates(self, cmd: str, buf: List[str], res: Any, word: str) -> None:
        params, ses = dict(res.params), dict(res.std_errors)
        if word.startswith("ivreg"):
            # `first` prints the first-stage regressions above the estimates
            start = [i for i, ln in enumerate(buf) if "Instrumental" in ln]
            buf = buf[start[-1] :] if start else buf
        odds = word == "logit" and re.search(r",.*\bor\b", cmd) is not None
        if odds:  # odds ratios: exp(b), with the delta-method standard error
            ses = {k: float(np.exp(params[k]) * v) for k, v in ses.items()}
            params = {k: float(np.exp(v)) for k, v in params.items()}
        for alias in ("Intercept", "const"):
            if alias in params:
                params["_cons"], ses["_cons"] = params[alias], ses[alias]
        assert self.session is not None
        stored_b = self.session.stored.get("_b") or {}
        stored_se = self.session.stored.get("_se") or {}
        # `xtreg, mle` is fitted by Stata's `ml`; its standard errors agree
        # to about six digits
        ml_rtol = 1e-5 if word == "xtreg" and re.search(r",.*\bmle\b", cmd) else 0.0
        # `nl` stops when the residual sum of squares changes by less than
        # 1e-5 in relative terms, which leaves about four digits of the
        # parameters when the surface is flat
        b_rtol = 1e-4 if word == "nl" else 0.0
        ml_rtol = max(ml_rtol, b_rtol)
        by_term = {stata_term(str(k)): k for k in params}
        # `encode` defines its value label inside the session, not in the file
        labels = dict(self.labels)
        steps = getattr(self.session, "_steps", None)
        sets = getattr(steps, "_label_sets", {})
        for var, set_name in getattr(steps, "_set_of", {}).items():
            if var not in labels and set_name in sets:
                mapping: Dict[str, Any] = {}
                for code, text in sets[set_name].items():
                    mapping[str(text)] = int(code)
                    mapping.setdefault(str(text).split()[0], int(code))
                labels[var] = mapping
        for name, (b, se) in coefficient_table(buf, labels).items():
            if name in ("sigma_u", "sigma_e", "rho", "/sigma", "var(e.y)"):
                continue
            if name in ("/sigma_u", "/sigma_e"):
                continue
            if word == "nl":
                name = name.lstrip("/")  # parameters are printed as /b0
            # xtreg, fe prints _cons, which the session derives
            ours_b = params.get(name, stored_b.get(name))
            ours_se = ses.get(name, stored_se.get(name))
            if ours_b is None:
                key = by_term.get(stata_term(name))
                if key is not None:
                    ours_b, ours_se = params[key], ses.get(key)
            self.report.number(self.name, cmd, f"b[{name}]", b, ours_b, rtol=b_rtol)
            self.report.number(self.name, cmd, f"se[{name}]", se, ours_se, rtol=ml_rtol)
        if word == "xtreg":
            text = "\n".join(buf)
            e = self.session.stored.get("e") or {}
            for label, pattern, key in (
                ("sigma_u", rf"^\s*sigma_u \|\s*({NUM})", "sigma_u"),
                ("sigma_e", rf"^\s*sigma_e \|\s*({NUM})", "sigma_e"),
                ("rho", rf"^\s*rho \|\s*({NUM})", "rho"),
                ("R2 within", rf"Within\s*=\s*({NUM})", "r2_w"),
                ("R2 between", rf"Between\s*=\s*({NUM})", "r2_b"),
                ("R2 overall", rf"Overall\s*=\s*({NUM})", "r2_o"),
                ("theta", rf"^theta\s*=\s*({NUM})", "theta"),
                ("corr(u_i, Xb)", rf"corr\(u_i, Xb\)\s*=\s*({NUM})", "corr"),
            ):
                m = re.search(pattern, text, re.M)
                if m and (key in e or "mle" not in cmd):
                    self.report.number(
                        self.name, cmd, label, Printed(m.group(1)), e.get(key)
                    )
        if word == "prais":
            text = "\n".join(buf)
            info = res.model_info
            for label, pattern, ours in (
                ("rho", rf"^\s*rho \|\s*({NUM})", info["rho"]),
                ("dw original", rf"\(original\)\s*=\s*({NUM})", info["dw_original"]),
                ("dw transformed", rf"\(transformed\)\s*=\s*({NUM})",
                 info["dw_transformed"]),
            ):  # fmt: skip
                m = re.search(pattern, text, re.M)
                if m:
                    self.report.number(self.name, cmd, label, Printed(m.group(1)), ours)
        if word in ("reg", "regress", "prais"):
            assert self.session is not None
            e = self.session.stored.get("e") or {}
            for label, printed in header_statistics(buf).items():
                self.report.number(self.name, cmd, f"e({label})", printed, e.get(label))

    def _pair(self, cmd: str, buf: List[str], out: Dict[str, Any]) -> None:
        """A test printed as ``chi2(k) = ...`` / ``F(a, b) = ...`` and its
        ``Prob > ...`` line."""
        text = "\n".join(buf)
        m = re.search(rf"\b(F|chi2)\(\s*(\d+)(?:,\s*\d+)?\)\s*=\s*({NUM})", text)
        if m:
            self.report.number(
                self.name, cmd, f"{m.group(1)} statistic", Printed(m.group(3)),
                out.get("statistic"),
            )  # fmt: skip
            self.report.number(
                self.name, cmd, "df", Printed(m.group(2)),
                out.get("df", out.get("df1")),
            )  # fmt: skip
        m = re.search(rf"Prob > (?:F|chi2)\s*=\s*({NUM})", text)
        if m:
            self.report.number(
                self.name, cmd, "p-value", Printed(m.group(1)), out.get("pvalue")
            )

    def _margins(self, cmd: str, buf: List[str], out: Any) -> None:
        frame = out if isinstance(out, pd.DataFrame) else getattr(out, "table", None)
        if frame is None:
            return
        key = "variable" if "variable" in frame.columns else None
        table = frame.set_index(key) if key else frame
        est = next(
            (c for c in ("dydx", "dy/dx", "margin", "estimate") if c in table), None
        )
        se = next((c for c in ("se", "std_err", "std_error") if c in table), None)
        by_term = {stata_term(str(k)): k for k in table.index}
        for name, (b, s_) in coefficient_table(buf).items():
            ours = table.loc[name] if name in table.index else None
            if ours is None and stata_term(name) in by_term:
                # a factor level: `2.region` here, `C(region)[T.2]` in the log
                ours = table.loc[by_term[stata_term(name)]]
            self.report.number(
                self.name, cmd, f"dydx[{name}]", b,
                None if ours is None or est is None else ours[est],
            )  # fmt: skip
            self.report.number(
                self.name, cmd, f"se[{name}]", s_,
                None if ours is None or se is None else ours[se],
            )  # fmt: skip

    def _hausman(self, cmd: str, buf: List[str], out: Dict[str, Any]) -> None:
        text = "\n".join(buf)
        m = re.search(r"chi2\((\d+)\)", text)
        if m:
            self.report.number(self.name, cmd, "df", Printed(m.group(1)), out["df"])
        m = re.search(rf"^\s*=\s*({NUM})\s*$", text, re.M)
        if m:
            self.report.number(
                self.name, cmd, "chi2", Printed(m.group(1)), out["statistic"]
            )
        m = re.search(rf"Prob > chi2 =\s*({NUM})", text)
        if m:
            self.report.number(
                self.name, cmd, "p-value", Printed(m.group(1)), out["pvalue"]
            )
        table = out["table"]
        for ln in buf:
            m = re.match(
                rf"^\s*(\S+)\s*\|\s*({NUM})\s+({NUM})\s+({NUM})\s+({NUM}|\.)\s*$", ln
            )
            if m and m.group(1) in table.index:
                row = table.loc[m.group(1)]
                self.report.number(
                    self.name, cmd, f"b-B[{m.group(1)}]", Printed(m.group(4)),
                    row["difference"],
                )  # fmt: skip
                if m.group(5) != ".":
                    self.report.number(
                        self.name, cmd, f"se(b-B)[{m.group(1)}]",
                        Printed(m.group(5)), row["se"],
                    )  # fmt: skip

    def _corrgram(self, cmd: str, buf: List[str], out: Any, word: str) -> None:
        text = "\n".join(buf)
        if word == "wntestq":
            last = out.iloc[-1]
            m = re.search(rf"\(Q\) statistic =\s*({NUM})", text)
            if m:
                self.report.number(self.name, cmd, "Q", Printed(m.group(1)), last["Q"])
            m = re.search(rf"Prob > chi2\((\d+)\)\s*=\s*({NUM})", text)
            if m:
                self.report.number(
                    self.name, cmd, "p-value", Printed(m.group(2)), last["Prob>Q"]
                )
            return
        for ln in buf:
            m = re.match(rf"^(\d+)\s+({NUM})\s+({NUM})\s+({NUM})\s+({NUM})", ln)
            if not m or int(m.group(1)) not in out.index:
                continue
            row = out.loc[int(m.group(1))]
            for label, group in (("AC", 2), ("PAC", 3), ("Q", 4), ("Prob>Q", 5)):
                self.report.number(
                    self.name, cmd, f"{label}[{m.group(1)}]",
                    Printed(m.group(group)), row[label],
                )  # fmt: skip

    def _estat(self, cmd: str, buf: List[str], out: Dict[str, Any]) -> None:
        text = "\n".join(buf)
        sub = cmd.split()[1].rstrip(",").lower()
        if sub.startswith("bgo"):
            m = re.search(rf"^\s*(\d+)\s*\|\s*({NUM})\s+(\d+)\s+({NUM})", text, re.M)
            if m:
                self.report.number(
                    self.name, cmd, "chi2", Printed(m.group(2)), out["statistic"]
                )
                self.report.number(
                    self.name, cmd, "p-value", Printed(m.group(4)), out["pvalue"]
                )
        elif sub.startswith("dwa"):
            m = re.search(rf"d-statistic\(.*\)\s*=\s*({NUM})", text)
            if m:
                self.report.number(
                    self.name, cmd, "d", Printed(m.group(1)), out["statistic"]
                )
        elif sub == "vif":
            table = out["vif_table"].set_index("variable")
            for ln in buf:
                m = re.match(rf"^\s*(\S+)\s*\|\s*({NUM})\s+({NUM})\s*$", ln)
                if not m:
                    continue
                name = m.group(1)
                if name == "Mean":
                    continue
                ours = table.loc[name] if name in table.index else None
                self.report.number(
                    self.name, cmd, f"VIF[{name}]", Printed(m.group(2)),
                    None if ours is None else ours["VIF"],
                )  # fmt: skip
                self.report.number(
                    self.name, cmd, f"1/VIF[{name}]", Printed(m.group(3)),
                    None if ours is None else ours["1/VIF"],
                )  # fmt: skip
            m = re.search(rf"Mean VIF\s*\|\s*({NUM})", text)
            if m:
                self.report.number(
                    self.name, cmd, "mean VIF", Printed(m.group(1)), out["mean_vif"]
                )
        elif sub == "ic":
            m = re.search(
                rf"^\s*\S+\s*\|\s*([\d,]+)\s+({NUM}|\.)\s+({NUM})\s+(\d+)"
                rf"\s+({NUM})\s+({NUM})\s*$",
                text,
                re.M,
            )
            if m:
                for label, group, key in (("ll", 3, "ll"), ("df", 4, "k"),
                                          ("AIC", 5, "AIC"), ("BIC", 6, "BIC")):  # fmt: skip
                    self.report.number(
                        self.name, cmd, label, Printed(m.group(group)), out[key]
                    )
        elif sub.startswith("imt"):
            self._pair(cmd, buf, out)
            table = out.get("table")
            for label, row in (("Skewness", "skewness"), ("Kurtosis", "kurtosis"),
                               ("Total", "total")):  # fmt: skip
                m = re.search(
                    rf"^\s*{label} \|\s*({NUM})\s+(\d+)\s+({NUM})", text, re.M
                )
                if m and table is not None:
                    self.report.number(
                        self.name, cmd, f"chi2[{row}]", Printed(m.group(1)),
                        table.loc[row, "chi2"],
                    )  # fmt: skip
                    self.report.number(
                        self.name, cmd, f"df[{row}]", Printed(m.group(2)),
                        table.loc[row, "df"],
                    )  # fmt: skip
        elif sub.startswith("endog"):
            for label, pattern, key in (
                (
                    "Durbin chi2",
                    rf"Durbin \(score\) chi2\(\d+\)\s*=\s*({NUM})",
                    "durbin",
                ),
                ("Durbin p", rf"Durbin.*\(p = ({NUM})\)", "durbin_pvalue"),
                (
                    "Wu-Hausman F",
                    rf"Wu-Hausman F\(\d+,\d+\)\s*=\s*({NUM})",
                    "statistic",
                ),
                ("Wu-Hausman p", rf"Wu-Hausman.*\(p = ({NUM})\)", "pvalue"),
            ):
                m = re.search(pattern, text)
                if m:
                    self.report.number(
                        self.name, cmd, label, Printed(m.group(1)), out.get(key)
                    )
        elif sub.startswith("overid"):
            # one line per statistic; after a robust fit Stata prints the
            # score test alone unless `forcenonrobust` asks for the others
            iid = out.get("iid_errors") or {}
            robust_fit = "sargan" in iid and iid["sargan"] != out.get("statistic")
            ours = {
                "Sargan": (iid.get("sargan"), iid.get("sargan_pvalue")),
                "Basmann": (
                    iid.get("basmann", iid.get("basmann_f")),
                    iid.get("basmann_pvalue", iid.get("basmann_f_pvalue")),
                ),
                "Anderson-Rubin": (
                    iid.get("anderson_rubin"), iid.get("anderson_rubin_pvalue"),
                ),
                "Score": (out.get("score"), out.get("score_pvalue")),
            }  # fmt: skip
            if not iid or not robust_fit:
                ours.setdefault("Sargan", (out.get("statistic"), out.get("pvalue")))
            found = re.findall(
                rf"^\s*([A-Z][\w-]*) (?:chi2|F)\([\d, ]+\)\s*=\s*({NUM})"
                rf"\s*\(p = ({NUM})\)",
                text,
                re.M,
            )
            for label, stat, p in found:
                mine = ours.get(label, (None, None))
                if label == "Sargan" and mine[0] is None:
                    mine = (out.get("statistic"), out.get("pvalue"))
                self.report.number(
                    self.name, cmd, f"{label} statistic", Printed(stat), mine[0]
                )
                self.report.number(self.name, cmd, f"{label} p", Printed(p), mine[1])
        elif sub.startswith("first"):
            m = re.search(
                rf"^\s*\S+\s*\|\s*({NUM})\s+({NUM})\s+({NUM})\s+({NUM})\s+({NUM})",
                text,
                re.M,
            )
            if m:
                self.report.number(
                    self.name, cmd, "first-stage F", Printed(m.group(4)),
                    out.get("statistic"),
                )  # fmt: skip
        elif sub.startswith("clas"):
            for label, pattern, key in (
                ("sensitivity", r"Sensitivity\s+Pr\( \+\| D\)\s+", "sensitivity"),
                ("specificity", r"Specificity\s+Pr\( -\|~D\)\s+", "specificity"),
                ("ppv", r"Positive predictive value\s+Pr\( D\| \+\)\s+", "ppv"),
                ("npv", r"Negative predictive value\s+Pr\(~D\| -\)\s+", "npv"),
                ("correct", r"Correctly classified\s+", "correctly_classified"),
            ):
                m = re.search(pattern + rf"({NUM})%", text)
                if m:
                    self.report.number(
                        self.name, cmd, label, Printed(m.group(1)), 100 * out[key]
                    )
        else:
            self._pair(cmd, buf, out)

    def _test(self, cmd: str, buf: List[str], out: Dict[str, Any], quiet: bool) -> None:
        if quiet:
            return
        text = "\n".join(buf)
        m = re.search(rf"\b(F|chi2)\(\s*\d+(?:,\s*\d+)?\)\s*=\s*({NUM})", text)
        if m:
            ours = out["chi2"] if m.group(1) == "chi2" else out["statistic"]
            self.report.number(
                self.name, cmd, f"{m.group(1)} statistic", Printed(m.group(2)), ours
            )
        m = re.search(rf"Prob > (?:F|chi2)\s*=\s*({NUM})", text)
        if m:
            self.report.number(
                self.name, cmd, "p-value", Printed(m.group(1)), out["pvalue"]
            )

    def _ttest(self, cmd: str, buf: List[str], res: Any) -> None:
        text = "\n".join(buf)
        for label, pattern, ours in (
            ("t", rf"\bt =\s*({NUM})", res.statistic),
            ("df", rf"degrees of freedom =\s*({NUM})", res.df),
            ("diff", rf"^\s*diff \|\s*(?:[\d,]+\s+)?({NUM})\s+{NUM}", res.estimate),
            ("se(diff)", rf"^\s*diff \|\s*(?:[\d,]+\s+)?{NUM}\s+({NUM})", res.se),
            ("p two-sided", rf"Pr\(\|T\| > \|t\|\) =\s*({NUM})", res.pvalue),
        ):
            m = re.search(pattern, text, re.M)
            if m:
                self.report.number(self.name, cmd, label, Printed(m.group(1)), ours)

    def _summarize(self, cmd: str, buf: List[str], out: pd.DataFrame) -> None:
        for ln in buf:
            m = re.match(
                rf"^\s*(\w+)\s*\|\s*([\d,]+)\s+({NUM})\s+({NUM})"
                rf"\s+({NUM})\s+({NUM})\s*$",
                ln,
            )
            if m and m.group(1) in out.index:
                row = out.loc[m.group(1)]
                for label, group, key in (("mean", 3, "Mean"), ("sd", 4, "Std. Dev.")):
                    self.report.number(
                        self.name, cmd, f"{label}[{m.group(1)}]",
                        Printed(m.group(group)), row[key],
                    )  # fmt: skip
        if len(out) != 1 or "Skewness" not in out.columns:
            return
        # `summarize v, detail`: percentiles on the left, moments on the right
        row = out.iloc[0]
        text = "\n".join(buf)
        percentiles = ((1, "P1"), (5, "P5"), (10, "P10"), (25, "P25"))
        percentiles += ((50, "Median"), (75, "P75"), (90, "P90"))
        percentiles += ((95, "P95"), (99, "P99"))
        for pct, key in percentiles:
            m = re.search(rf"^\s*{pct}%\s+({NUM})", text, re.M)
            if m:
                self.report.number(
                    self.name, cmd, f"p{pct}", Printed(m.group(1)), row[key]
                )
        for label, key in (("Mean", "Mean"), ("Std. Dev.", "Std. Dev."),
                           ("Variance", "Variance"), ("Skewness", "Skewness"),
                           ("Kurtosis", "Kurtosis")):  # fmt: skip
            m = re.search(rf"{re.escape(label)}\s+({NUM})\s*$", text, re.M)
            if m:
                self.report.number(
                    self.name, cmd, label.lower(), Printed(m.group(1)), row[key]
                )

    def _display(self, cmd: str, buf: List[str]) -> None:
        """``display``: the line the session prints against the line in the
        log. Numbers are compared at the precision Stata printed them; the
        text between them must be the same."""
        assert self.session is not None
        if "_result(8)" in cmd:  # pre-Stata-6 spelling of e(r2_a)
            cmd = cmd.replace("_result(8)", "e(r2_a)")
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            self.session.run(cmd)
        ours = self.session.output
        shown = " ".join(b.strip() for b in buf if b.strip() and not b.startswith(". "))
        if not shown:
            return
        if isinstance(ours, (int, float, np.integer, np.floating)):
            printed = re.findall(NUM, shown.replace(",", ""))
            if printed:
                self.report.number(
                    self.name, cmd, "displayed value", Printed(printed[-1]), float(ours)
                )
            return
        if not isinstance(ours, str):
            return
        theirs = re.sub(r"\s+", " ", shown).strip()
        mine = re.sub(r"\s+", " ", re.sub(r"\{[^{}]*\}", "", ours)).strip()
        number = re.compile(rf"(?<![\w.]){NUM}(?![\w])")
        a, b = number.split(theirs), number.split(mine)
        na, nb = number.findall(theirs), number.findall(mine)
        same = [x.strip() for x in a] == [x.strip() for x in b] and len(na) == len(nb)
        if same:
            same = all(Printed(x).matches(float(y)) for x, y in zip(na, nb))
        self.report.add(
            self.name, cmd, "displayed text", "ok" if same else "DIFF",
            stata=float("nan"), ours=float("nan"),
            reason="" if same else f"Stata: {theirs[:60]!r}  ours: {mine[:60]!r}",
        )  # fmt: skip


# ------------------------------------------------------------ panel blocks
def _cmp_xtsum(self: "Replay", cmd: str, buf: List[str], out: Any) -> None:
    variable: Optional[str] = None
    for ln in buf:
        m = re.match(
            rf"^(\S+)?\s+(overall|between|within)\s*\|\s*(?:({NUM})\s+)?({NUM})"
            rf"\s+({NUM})\s+({NUM})\s*\|",
            ln,
        )
        if not m:
            continue
        variable = m.group(1) or variable
        if (variable, m.group(2)) not in out.index:
            continue
        row = out.loc[(variable, m.group(2))]
        cells = [("sd", 4), ("min", 5), ("max", 6)]
        if m.group(3):
            cells.insert(0, ("mean", 3))
        for label, group in cells:
            self.report.number(
                self.name, cmd, f"{label}[{variable},{m.group(2)}]",
                Printed(m.group(group)), row[label],
            )  # fmt: skip


def _cmp_xtserial(self: "Replay", cmd: str, buf: List[str], out: Any) -> None:
    text = "\n".join(buf)
    m = re.search(
        rf"F\(\s*1,\s*(\d+)\)\s*=\s*({NUM})\s*\n\s*Prob > F =\s*({NUM})", text
    )
    if m:
        self.report.number(self.name, cmd, "F", Printed(m.group(2)), out["statistic"])
        self.report.number(
            self.name, cmd, "p-value", Printed(m.group(3)), out["pvalue"]
        )
    for name, (b, se) in coefficient_table(buf).items():
        key = "D." + name[: -len("_D1")] if name.endswith("_D1") else name
        ours = key in out["params"].index
        self.report.number(
            self.name, cmd, f"b[{key}]", b, out["params"][key] if ours else None
        )
        self.report.number(
            self.name, cmd, f"se[{key}]", se, out["std_errors"][key] if ours else None
        )


def _cmp_xttest0(self: "Replay", cmd: str, buf: List[str], out: Any) -> None:
    m = re.search(rf"chibar2\(01\) =\s*({NUM})", "\n".join(buf))
    if m:
        self.report.number(
            self.name, cmd, "chibar2", Printed(m.group(1)), out["statistic"]
        )


def _cmp_xtoverid(self: "Replay", cmd: str, buf: List[str], out: Any) -> None:
    m = re.search(
        rf"statistic\s+({NUM})\s+Chi-sq\((\d+)\)\s+P-value =\s*({NUM})", "\n".join(buf)
    )
    if m:
        self.report.number(
            self.name, cmd, "chi2", Printed(m.group(1)), out["statistic"]
        )
        self.report.number(self.name, cmd, "df", Printed(m.group(2)), out["df"])
        self.report.number(
            self.name, cmd, "p-value", Printed(m.group(3)), out["pvalue"]
        )


# ----------------------------------------------------------- design blocks
def _cmp_rdrobust(self: "Replay", cmd: str, buf: List[str], out: Any) -> None:
    text = "\n".join(buf)
    info = out.model_info
    for label, pattern, left, right in (
        ("N", r"Number of obs \|\s*(\d+)\s+(\d+)", info["n_left"], info["n_right"]),
        ("N eff", r"Eff\. Number of obs \|\s*(\d+)\s+(\d+)",
         info["n_effective_left"], info["n_effective_right"]),
    ):  # fmt: skip
        m = re.search(pattern, text)
        if m:
            self.report.number(
                self.name, cmd, f"{label} left", Printed(m.group(1)), left
            )
            self.report.number(
                self.name, cmd, f"{label} right", Printed(m.group(2)), right
            )
    for label, pattern, key in (("h", r"BW est\. \(h\)", "bandwidth_h"),
                                ("b", r"BW bias \(b\)", "bandwidth_b")):  # fmt: skip
        m = re.search(pattern + rf" \|\s*({NUM})\s+({NUM})", text)
        ours = info.get(key)
        if m and ours is not None:
            pair = ours if isinstance(ours, (tuple, list)) else (ours, ours)
            self.report.number(
                self.name, cmd, f"{label} left", Printed(m.group(1)), pair[0]
            )
            self.report.number(
                self.name, cmd, f"{label} right", Printed(m.group(2)), pair[1]
            )
    table = (
        out.detail.set_index("method")
        if getattr(out, "detail", None) is not None
        else None
    )
    for stata_row, ours_row in (("Conventional", "Conventional"), ("Robust", "Robust")):
        m = re.search(rf"^\s*{stata_row} \|\s*({NUM})\s+({NUM})", text, re.M)
        if not m:
            continue
        row = (
            table.loc[ours_row]
            if table is not None and ours_row in table.index
            else None
        )
        self.report.number(
            self.name, cmd, f"coef[{stata_row}]", Printed(m.group(1)),
            None if row is None else row["estimate"],
        )  # fmt: skip
        self.report.number(
            self.name, cmd, f"se[{stata_row}]", Printed(m.group(2)),
            None if row is None else row["se"],
        )  # fmt: skip


def _cmp_rddensity(self: "Replay", cmd: str, buf: List[str], out: Any) -> None:
    text = "\n".join(buf)
    m = re.search(rf"^\s*Robust \|\s*({NUM})\s+({NUM})", text, re.M)
    if m:
        self.report.number(self.name, cmd, "T", Printed(m.group(1)), out.estimate)
        self.report.number(self.name, cmd, "p-value", Printed(m.group(2)), out.pvalue)
    m = re.search(rf"BW est\. \(h\) \|\s*({NUM})\s+({NUM})", text)
    if m:
        self.report.number(
            self.name,
            cmd,
            "h left",
            Printed(m.group(1)),
            out.model_info["bandwidth_left"],
        )
        self.report.number(
            self.name,
            cmd,
            "h right",
            Printed(m.group(2)),
            out.model_info["bandwidth_right"],
        )


def _cmp_teffects(self: "Replay", cmd: str, buf: List[str], out: Any) -> None:
    text = "\n".join(buf)
    # the contrast is printed by value label when the treatment has one
    m = re.search(rf"\([^()|]+ vs [^()|]+\)\s*\|\s*({NUM})\s+({NUM})", text)
    if m:
        self.report.number(self.name, cmd, "effect", Printed(m.group(1)), out.estimate)
        # the standard error carries the logit of the propensity score, which
        # Stata fits with its own optimiser
        self.report.number(self.name, cmd, "se", Printed(m.group(2)), out.se, rtol=1e-5)
    m = re.search(r"Number of obs\s*=\s*([\d,]+)", text)
    if m:
        info = out.model_info
        # sp.match counts the arms; sp.g_computation holds n_obs on the result
        count = info.get("n_treated", 0) + info.get("n_control", 0)
        self.report.number(
            self.name, cmd, "N", Printed(m.group(1).replace(",", "")),
            count or getattr(out, "n_obs", 0),
        )  # fmt: skip


def _cmp_tebalance(self: "Replay", cmd: str, buf: List[str], out: Any) -> None:
    for m in _rows(buf, rf"^\s*(\S+)\s*\|\s*({NUM})\s+({NUM})\s+({NUM})\s+({NUM})\s*$"):
        if m.group(1) not in out.index:
            continue
        row = out.loc[m.group(1)]
        for label, group in (("std_diff_raw", 2), ("std_diff_matched", 3),
                             ("var_ratio_raw", 4), ("var_ratio_matched", 5)):  # fmt: skip
            self.report.number(
                self.name, cmd, f"{label}[{m.group(1)}]", Printed(m.group(group)),
                row[label],
            )  # fmt: skip


def _unit_codes(self: "Replay", unit: str) -> Dict[str, Any]:
    """Printed unit name -> code (Stata prints the value label, blanks removed)."""
    out: Dict[str, Any] = {}
    for text, code in self.labels.get(unit, {}).items():
        out[text.replace(" ", "")] = code
    return out


def _cmp_rcm(self: "Replay", cmd: str, buf: List[str], out: Any) -> None:
    info = out.model_info
    table = info["selection_table"]
    for m in _rows(
        buf, rf"^\s*(\d+) \|\s*({NUM})\s+({NUM})\s+({NUM})\s+({NUM})\s+({NUM})\s*$"
    ):
        k = int(m.group(1))
        if k not in table.index:
            continue
        for label, group in (
            ("aicc", 2),
            ("aic", 3),
            ("bic", 4),
            ("mbic", 5),
            ("r2", 6),
        ):
            self.report.number(
                self.name, cmd, f"{label}[K={k}]", Printed(m.group(group)),
                table.loc[k, label],
            )  # fmt: skip
    text = "\n".join(buf)
    for label, pattern, ours in (
        ("N predictors", r"Number of Predictors\s*=\s*(\d+)", info["n_selected"]),
        ("RMSE", rf"Root Mean Squared Error\s*=\s*({NUM})", info["pre_rmse"]),
        ("R-squared", rf"R-squared\s*=\s*({NUM})\s*$", info["pre_r2"]),
    ):
        m = re.search(pattern, text, re.M)
        if m:
            self.report.number(self.name, cmd, label, Printed(m.group(1)), ours)
    assert self.session is not None
    unit = self.session.panel[0] or ""
    codes = _unit_codes(self, unit)
    coefs = info["coefficients"]
    for ln in buf:
        m = re.match(rf"^\s*(\S+)\s*\|\s*({NUM})\s+({NUM})\s+({NUM})\s+({NUM})", ln)
        if not m or not ("·" in m.group(1) or m.group(1) == "_cons"):
            continue
        name = m.group(1).split("·")[-1]
        key = "_cons" if name == "_cons" else str(codes.get(name, name))
        key = key if key in coefs.index else str(float(codes.get(name, 0)))
        ours = coefs.loc[key] if key in coefs.index else None
        self.report.number(
            self.name, cmd, f"b[{name}]", Printed(m.group(2)),
            None if ours is None else ours["coef"],
        )  # fmt: skip
        self.report.number(
            self.name, cmd, f"se[{name}]", Printed(m.group(3)),
            None if ours is None else ours["se"],
        )  # fmt: skip
    # tables of (time | actual predicted effect): the first is the estimate,
    # a second one is the in-time placebo
    blocks: List[List["re.Match[str]"]] = [[]]
    for ln in buf:
        m = re.match(rf"^\s*(\S+)\s*\|\s*({NUM})\s+({NUM})\s+({NUM})\s*$", ln)
        if m and m.group(1) != "Mean":
            blocks[-1].append(m)
        elif m:
            blocks.append([])
    detail = out.detail[out.detail["post"]].reset_index(drop=True)
    for kind, rows, frame in (
        ("effect", blocks[0], detail),
        (
            "placebo-time",
            blocks[1] if len(blocks) > 1 else [],
            info.get("placebo_time"),
        ),
    ):
        if frame is None:
            continue
        for i, m in enumerate(rows):
            ours = frame.iloc[i] if i < len(frame) else None
            for label, group, col in (
                ("predicted", 3, "predicted"),
                ("effect", 4, "effect"),
            ):
                self.report.number(
                    self.name, cmd, f"{kind} {label}[{m.group(1)}]",
                    Printed(m.group(group)), None if ours is None else ours[col],
                )  # fmt: skip
    m = re.search(rf"posttreatment period is ({NUM})\.\s*$", text, re.M)
    if m:
        self.report.number(self.name, cmd, "ATT", Printed(m.group(1)), out.estimate)
    placebo = info.get("placebo_units")
    if placebo is not None:
        for m in _rows(
            buf, rf"^\s*(\S+)\s*\|\s*({NUM})\s+({NUM})\s+({NUM})\s+({NUM})\s*$"
        ):
            code = codes.get(m.group(1))
            if code is None or code not in placebo.index:
                continue
            row = placebo.loc[code]
            for label, group, col in (("pre MSPE", 2, "pre_mspe"), ("post MSPE", 3, "post_mspe"),
                                      ("ratio", 4, "ratio"), ("relative", 5, "pre_mspe_relative")):  # fmt: skip
                self.report.number(
                    self.name, cmd, f"{label}[{m.group(1)}]", Printed(m.group(group)),
                    row[col],
                )  # fmt: skip
        m = re.search(rf"as large as \S+ is ({NUM})\.", text)
        if m:
            self.report.number(
                self.name, cmd, "placebo p-value", Printed(m.group(1)), out.pvalue
            )
        rows = _rows(
            buf, rf"^\s*(\S+)\s*\|\s*({NUM})\s+({NUM})\s+({NUM})\s+({NUM})\s*$"
        )
        pointwise = [m for m in rows if codes.get(m.group(1)) is None]
        for i, m in enumerate(pointwise):
            ours = detail.iloc[i] if i < len(detail) else None
            for label, group, col in (("p two-sided", 3, "p_two_sided"),
                                      ("p right", 4, "p_right"), ("p left", 5, "p_left")):  # fmt: skip
                self.report.number(
                    self.name, cmd, f"{label}[{m.group(1)}]", Printed(m.group(group)),
                    None if ours is None else ours[col],
                )  # fmt: skip


def _cmp_synth(self: "Replay", cmd: str, buf: List[str], out: Any) -> None:
    info = out.model_info
    text = "\n".join(buf)
    m = re.search(rf"RMSPE \|\s*({NUM})", text)
    if m:
        self.report.number(
            self.name, cmd, "RMSPE", Printed(m.group(1)), info.get("pre_treatment_rmse")
        )
    assert self.session is not None
    codes = _unit_codes(self, self.session.panel[0] or "")
    by_label = {str(k).replace(" ", ""): v for k, v in codes.items()}
    weights = info.get("weights")
    ours = {} if weights is None else dict(zip(weights["unit"], weights["weight"]))
    for m in _rows(buf, rf"^\s*(.+?)\s*\|\s*({NUM})\s*$"):
        label = m.group(1).replace(" ", "")
        if label in by_label:
            self.report.number(
                self.name, cmd, f"weight[{label}]", Printed(m.group(2)),
                float(ours.get(by_label[label], 0.0)),
            )  # fmt: skip
    balance = info.get("predictor_balance")
    if balance is not None:
        rows = _rows(buf, rf"^\s*(\S+)\s*\|\s*({NUM})\s+({NUM})\s*$")
        for i, m in enumerate(rows):
            if i >= len(balance):
                break
            self.report.number(
                self.name, cmd, f"treated[{m.group(1)}]", Printed(m.group(2)),
                balance.iloc[i]["treated"],
            )  # fmt: skip
            self.report.number(
                self.name, cmd, f"synthetic[{m.group(1)}]", Printed(m.group(3)),
                balance.iloc[i]["synthetic"],
            )  # fmt: skip


_SYNTH2_SECTIONS = (
    ("Optimal Unit Weights", "weights"),
    ("Prediction results in the posttreatment", "effects"),
    ("In-space placebo test results using fake treatment units (continued", "pvalues"),
    ("In-space placebo test results using fake treatment units:", "units"),
    ("In-time placebo test results", "time"),
    ("Leave-one-out robustness test results", "loo"),
    ("Treatment Effect (LOO)", "loo-effect"),
)


def _cmp_synth2(self: "Replay", cmd: str, buf: List[str], out: Any) -> None:
    """``synth2``: fit, unit weights, effects, placebo tables, leave-one-out."""
    info = out.model_info
    text = "\n".join(buf)
    gaps = info["gap_table"]
    pre = gaps.loc[~gaps["post_treatment"], "gap"]
    for label, pattern, ours in (
        ("RMSE", rf"Root Mean Squared Error\s*=\s*({NUM})", np.sqrt((pre**2).mean())),
        ("R-squared", rf"R-squared\s*=\s*({NUM})\s*$", info.get("pre_r2")),
        ("ATT", rf"posttreatment period is ({NUM})\.\s*$", out.estimate),
        ("placebo p-value", rf"Using all control units.*?is ({NUM})\.", info.get("placebo_pvalue")),
        ("placebo p-value", rf"^Note: The probability of obtaining.*?is ({NUM})\.", info.get("placebo_pvalue")),
        ("placebo p-value (cutoff)", rf"Excluding control units.*?is ({NUM})\.", info.get("placebo_pvalue_cutoff")),
    ):  # fmt: skip
        m = re.search(pattern, text, re.M | re.S)
        if m:
            self.report.number(self.name, cmd, label, Printed(m.group(1)), ours)
    assert self.session is not None
    codes = _unit_codes(self, self.session.panel[0] or "")
    weights = dict(zip(info["weights"]["unit"], info["weights"]["weight"]))
    post = gaps[gaps["post_treatment"]].set_index("time")
    units = info.get("placebo_table")
    pvals = info.get("placebo_effects")
    pvals = None if pvals is None else pvals.set_index("time")
    fake = info.get("placebo_time")
    fake = None if fake is None else fake.set_index("time")
    loo = info.get("loo")
    loo = None if loo is None else loo.set_index("time")

    def cell(frame: Any, key: Any, col: str) -> Any:
        if frame is None or key not in frame.index:
            return None
        return frame.loc[key, col]

    section = ""
    for ln in buf:
        for title, name in _SYNTH2_SECTIONS:
            if title in ln:
                section = name
                break
        m = re.match(rf"^\s*(\S+)\s*\|((?:\s+{NUM})+)\s*$", ln)
        if not m or m.group(1) == "Mean":
            continue
        label, values = m.group(1), m.group(2).split()
        code = codes.get(label, label)
        period: Any = None
        try:
            period = int(label)
        except ValueError:
            pass
        columns: List[Tuple[str, Any]] = []
        if section == "weights" and len(values) == 1:
            columns = [("weight", float(weights.get(code, 0.0)))]
        elif section == "effects" and len(values) == 3:
            columns = [("", None), ("synthetic", cell(post, period, "synthetic")),
                       ("effect", cell(post, period, "gap"))]  # fmt: skip
        elif section == "units" and len(values) == 4:
            columns = [(name, cell(units, code, col)) for name, col in (
                ("pre MSPE", "pre_mspe"), ("post MSPE", "post_mspe"),
                ("ratio", "ratio"), ("relative", "pre_mspe_relative"))]  # fmt: skip
        elif section == "pvalues" and len(values) == 4:
            columns = [("", None)] + [(name, cell(pvals, period, col)) for name, col in (
                ("p two-sided", "p_two_sided"), ("p right", "p_right"),
                ("p left", "p_left"))]  # fmt: skip
        elif section == "time" and len(values) == 3:
            columns = [("", None), ("placebo-time synthetic", cell(fake, period, "synthetic")),
                       ("placebo-time effect", cell(fake, period, "effect"))]  # fmt: skip
        elif section == "loo" and len(values) == 4:
            columns = [("", None), ("", None),
                       ("loo synthetic min", cell(loo, period, "synthetic_min")),
                       ("loo synthetic max", cell(loo, period, "synthetic_max"))]  # fmt: skip
        elif section == "loo-effect" and len(values) == 3:
            columns = [("", None), ("loo effect min", cell(loo, period, "effect_min")),
                       ("loo effect max", cell(loo, period, "effect_max"))]  # fmt: skip
        for value, (name, ours) in zip(values, columns):
            if name:
                self.report.number(
                    self.name, cmd, f"{name}[{label}]", Printed(value), ours
                )


def _cmp_sdid(self: "Replay", cmd: str, buf: List[str], out: Any) -> None:
    """``sdid``: the ATT. Its placebo / bootstrap standard error is drawn
    at random on both sides and is not compared."""
    m = re.search(rf"^\s*\S+\s*\|\s*({NUM})\s+({NUM})\s+({NUM})", "\n".join(buf), re.M)
    if m:
        self.report.number(self.name, cmd, "ATT", Printed(m.group(1)), out.estimate)


def _cmp_boottest(self: "Replay", cmd: str, buf: List[str], out: Any) -> None:
    """``boottest``: the t statistic, and the p-value when the Rademacher
    draws were enumerated (it is then the same number on both sides)."""
    text = "\n".join(buf)
    m = re.search(rf"t\(\d+\)\s*=\s*({NUM})", text)
    if m:
        self.report.number(self.name, cmd, "t", Printed(m.group(1)), out["t_stat"])
    m = re.search(rf"Prob>\|t\|\s*=\s*({NUM})", text)
    if m and out.get("enumerated"):
        self.report.number(self.name, cmd, "p", Printed(m.group(1)), out["p_boot"])


def _cmp_psmatch2(self: "Replay", cmd: str, buf: List[str], out: Any) -> None:
    """``psmatch2``: the ATT row (difference and its standard error)."""
    m = re.search(
        rf"^\s*ATT\s*\|\s*({NUM})\s+({NUM})\s+({NUM})\s+({NUM})", "\n".join(buf), re.M
    )
    if m:
        self.report.number(self.name, cmd, "ATT", Printed(m.group(3)), out.att)
        self.report.number(self.name, cmd, "se", Printed(m.group(4)), out.se)


def _cmp_sensemakr(self: "Replay", cmd: str, buf: List[str], out: Any) -> None:
    # Treatment | Coef.  S.E.  t(H0)  R2yd.x  RV_q  RV_qa
    six = r"\s+".join([f"({NUM})"] * 6)
    for m in _rows(buf, rf"^\s*\S+\s*\|\s*{six}\s*$"):
        for label, group, key in (
            ("coef", 1, "beta_treat"),
            ("se", 2, "se_treat"),
            ("t", 3, "t_treat"),
            ("R2yd.x", 4, "partial_r2_yd"),
            ("RV_q", 5, "rv_q"),
            ("RV_qa", 6, "rv_qa"),
        ):
            self.report.number(self.name, cmd, label, Printed(m.group(group)), out[key])
        break


PANEL = {
    "sensemakr": _cmp_sensemakr,
    "sdid": _cmp_sdid,
    "boottest": _cmp_boottest,
    "psmatch2": _cmp_psmatch2,
    "synth": _cmp_synth,
    "synth2": _cmp_synth2,
    "rcm": _cmp_rcm,
    "rdrobust": _cmp_rdrobust,
    "rddensity": _cmp_rddensity,
    "teffects": _cmp_teffects,
    "tebalance": _cmp_tebalance,
    "xtsum": _cmp_xtsum,
    "xtserial": _cmp_xtserial,
    "xttest0": _cmp_xttest0,
    "xtoverid": _cmp_xtoverid,
}


# ------------------------------------------------------ time-series blocks
def _rows(buf: List[str], pattern: str) -> List["re.Match[str]"]:
    return [m for m in (re.match(pattern, ln) for ln in buf) if m]


def _cmp_dfuller(self: "Replay", cmd: str, buf: List[str], out: Any) -> None:
    text = "\n".join(buf)
    for label, pattern, ours in (
        ("Z(t)", rf"^\s*Z\(t\)\s+({NUM})", out.statistic),
        ("p-value", rf"p-value for Z\(t\) =\s*({NUM})", out.pvalue),
        ("N", r"Number of obs\s*=\s*(\d+)", out.n_obs),
    ):
        m = re.search(pattern, text, re.M)
        if m:
            self.report.number(self.name, cmd, label, Printed(m.group(1)), ours)


def _cmp_var(self: "Replay", cmd: str, buf: List[str], out: Any) -> None:
    text = "\n".join(buf)
    for label, pattern, ours in (
        ("N", r"Number of obs\s*=\s*(\d+)", out.n_obs),
        ("LL", rf"Log likelihood\s*=\s*({NUM})", out.log_likelihood),
        ("AIC", rf"\bAIC\s*=\s*({NUM})", out.aic),
        ("HQIC", rf"\bHQIC\s*=\s*({NUM})", out.hqic),
        ("SBIC", rf"\bSBIC\s*=\s*({NUM})", out.bic),
        ("FPE", rf"\bFPE\s*=\s*({NUM})", out.fpe),
        ("Det(Sigma_ml)", rf"Det\(Sigma_ml\)\s*=\s*({NUM})", out.det_sigma),
    ):
        m = re.search(pattern, text)
        if m:
            self.report.number(self.name, cmd, label, Printed(m.group(1)), ours)
    fit = out.equation_table()
    for m in _rows(buf, rf"^(\S+)\s+(\d+)\s+({NUM})\s+({NUM})\s+({NUM})\s+({NUM})\s*$"):
        if m.group(1) in fit.index:
            row = fit.loc[m.group(1)]
            for label, group, key in (("RMSE", 3, "rmse"), ("R-sq", 4, "r2"),
                                      ("chi2", 5, "chi2")):  # fmt: skip
                self.report.number(
                    self.name, cmd, f"{label}[{m.group(1)}]",
                    Printed(m.group(group)), row[key],
                )  # fmt: skip
    equation: Optional[str] = None
    group: Optional[str] = None
    for ln in buf:
        head = re.match(r"^(\S+)\s*\|\s*$", ln)
        if head:
            equation = head.group(1)
            continue
        sub = re.match(r"^\s+(\S+)\s*\|\s*$", ln)
        if sub:
            group = sub.group(1)
            continue
        m = re.match(rf"^\s*(\S+)\s*\|\s*({NUM})\s+({NUM})\s+({NUM})\s+({NUM})", ln)
        if not m or equation not in out.coefs:
            continue
        name = m.group(1)
        lag = re.fullmatch(r"L(\d+)\.", name)
        key = f"L{lag.group(1)}.{group}" if lag else name
        table = out.coefs[equation]
        ours = table.loc[key] if key in table.index else None
        self.report.number(
            self.name, cmd, f"b[{equation}:{key}]", Printed(m.group(2)),
            None if ours is None else ours["coef"],
        )  # fmt: skip
        self.report.number(
            self.name, cmd, f"se[{equation}:{key}]", Printed(m.group(3)),
            None if ours is None else ours["se"],
        )  # fmt: skip


def _cmp_varsoc(self: "Replay", cmd: str, buf: List[str], out: Any) -> None:
    for ln in buf:
        cells = [c for c in re.split(r"[|\s*]+", ln.strip()) if c]
        if not cells or not cells[0].isdigit() or int(cells[0]) not in out.index:
            continue
        lag = int(cells[0])
        row = out.loc[lag]
        try:
            numbers = [Printed(c) for c in cells[1:]]
        except ValueError:
            continue
        keys = ["LL", "FPE", "AIC", "HQIC", "SBIC"] if lag == 0 else [
            "LL", "LR", "df", "p", "FPE", "AIC", "HQIC", "SBIC"]  # fmt: skip
        for key, printed in zip(keys, numbers):
            self.report.number(self.name, cmd, f"{key}[{lag}]", printed, row[key])


def _cmp_var_table(self: "Replay", cmd: str, buf: List[str], out: Any) -> None:
    table = out["table"]
    word = cmd.split()[0].rstrip(",").lower()
    if word == "varwle":
        equation = None
        for ln in buf:
            head = re.match(r"^\s*Equation:\s*(\S+)", ln)
            if head:
                equation = head.group(1)
            m = re.match(rf"^\s*\|\s*(\d+)\s*\|\s*({NUM})\s+(\d+)\s+({NUM})", ln)
            if m and equation:
                hit = table[
                    (table.equation == equation) & (table.lag == int(m.group(1)))
                ]
                ours = hit.iloc[0] if len(hit) else None
                self.report.number(
                    self.name, cmd, f"chi2[{equation},{m.group(1)}]",
                    Printed(m.group(2)), None if ours is None else ours["chi2"],
                )  # fmt: skip
    elif word in ("varlmar", "veclmar"):
        for m in _rows(buf, rf"^\s*\|\s*(\d+)\s*\|\s*({NUM})\s+(\d+)\s+({NUM})"):
            hit = table[table.lag == int(m.group(1))]
            ours = hit.iloc[0] if len(hit) else None
            for label, group, key in (("chi2", 2, "chi2"), ("p", 4, "p")):
                self.report.number(
                    self.name, cmd, f"{label}[{m.group(1)}]",
                    Printed(m.group(group)), None if ours is None else ours[key],
                )  # fmt: skip
    elif word in ("varstable", "vecstable"):
        moduli = sorted(table["modulus"], reverse=True)
        printed = [m.group(1) for m in _rows(buf, rf"^\s*\|.*\|\s*({NUM})\s*\|\s*$")]
        for k, tok in enumerate(printed):
            self.report.number(
                self.name, cmd, f"modulus[{k + 1}]", Printed(tok),
                moduli[k] if k < len(moduli) else None,
            )  # fmt: skip
    elif word == "vargranger":
        for m in _rows(
            buf, rf"^\s*\|\s*(\S+)\s+(\S+)\s*\|\s*({NUM})\s+(\d+)\s+({NUM})"
        ):
            hit = table[(table.equation == m.group(1)) & (table.excluded == m.group(2))]
            ours = hit.iloc[0] if len(hit) else None
            for label, group, key in (("chi2", 3, "chi2"), ("p", 5, "p")):
                self.report.number(
                    self.name, cmd, f"{label}[{m.group(1)}<-{m.group(2)}]",
                    Printed(m.group(group)), None if ours is None else ours[key],
                )  # fmt: skip


def _cmp_vecrank(self: "Replay", cmd: str, buf: List[str], out: Any) -> None:
    # only the first (trace) table: `max` prints a second one
    seen = set()
    for m in _rows(
        buf,
        rf"^\s*(\d+)\s+(\d+)\s+({NUM})\s+({NUM}|\.)(?:\s+({NUM})\*?\s+({NUM}))?\s*$",
    ):
        rank = int(m.group(1))
        if rank in seen:
            continue
        seen.add(rank)
        if m.group(4) != ".":
            self.report.number(
                self.name, cmd, f"eigenvalue[{rank}]", Printed(m.group(4)),
                out.eigenvalues[rank - 1],
            )  # fmt: skip
        if m.group(5) and rank < len(out.test_stats):
            self.report.number(
                self.name, cmd, f"trace[{rank}]", Printed(m.group(5)),
                out.test_stats[rank],
            )  # fmt: skip


def _cmp_vec(self: "Replay", cmd: str, buf: List[str], out: Any) -> None:
    text = "\n".join(buf)
    for label, pattern, ours in (
        ("N", r"Number of obs\s*=\s*(\d+)", out.n_obs),
        ("LL", rf"Log likelihood\s*=\s*({NUM})", out.log_likelihood),
        ("AIC", rf"\bAIC\s*=\s*({NUM})", out.aic),
        ("HQIC", rf"\bHQIC\s*=\s*({NUM})", out.hqic),
        ("SBIC", rf"\bSBIC\s*=\s*({NUM})", out.bic),
        ("Det(Sigma_ml)", rf"Det\(Sigma_ml\)\s*=\s*({NUM})", out.det_sigma),
    ):
        m = re.search(pattern, text)
        if m:
            self.report.number(self.name, cmd, label, Printed(m.group(1)), ours)
    fit = out.equation_table()
    for m in _rows(buf, rf"^(\S+)\s+(\d+)\s+({NUM})\s+({NUM})\s+({NUM})\s+({NUM})\s*$"):
        if m.group(1) in fit.index:
            row = fit.loc[m.group(1)]
            for label, group, key in (("RMSE", 3, "rmse"), ("R-sq", 4, "r2"),
                                      ("chi2", 5, "chi2")):  # fmt: skip
                self.report.number(
                    self.name, cmd, f"{label}[{m.group(1)}]",
                    Printed(m.group(group)), row[key],
                )  # fmt: skip
    equation: Optional[str] = None
    group: Optional[str] = None
    in_beta = False
    for ln in buf:
        if re.match(r"^\s*beta \|", ln):
            in_beta = True
        head = re.match(r"^(\S+)\s*\|\s*$", ln)
        if head:
            equation = head.group(1)
            continue
        sub = re.match(r"^\s+(\S+)\s*\|\s*$", ln)
        if sub:
            group = sub.group(1)
            continue
        m = re.match(rf"^\s*(\S+)\s*\|\s*({NUM})\s+({NUM})\s+({NUM})\s+({NUM})", ln)
        if not m or equation is None:
            continue
        name = m.group(1)
        if in_beta:
            if equation in out.beta.columns and name in out.beta.index:
                self.report.number(
                    self.name, cmd, f"beta[{equation}:{name}]", Printed(m.group(2)),
                    out.beta.loc[name, equation],
                )  # fmt: skip
                self.report.number(
                    self.name, cmd, f"se beta[{equation}:{name}]", Printed(m.group(3)),
                    out.beta_se.loc[name, equation],
                )  # fmt: skip
            continue
        if equation not in out.coefs:
            continue
        op = re.fullmatch(r"L(\d*)(D?)\.", name)
        if op and group:
            lag = op.group(1)
            if op.group(2):
                key = ("LD." if lag in ("", "1") else f"L{lag}D.") + group
            else:
                key = f"L.{group}"
        else:
            key = name
        table = out.coefs[equation]
        ours = table.loc[key] if key in table.index else None
        self.report.number(
            self.name, cmd, f"b[{equation}:{key}]", Printed(m.group(2)),
            None if ours is None else ours["coef"],
        )  # fmt: skip
        self.report.number(
            self.name, cmd, f"se[{equation}:{key}]", Printed(m.group(3)),
            None if ours is None else ours["se"],
        )  # fmt: skip


def _matrix_rows(buf: List[str], start: str) -> List[List[Printed]]:
    """Numeric cells of the table rows after the line containing ``start``,
    up to the closing rule."""
    rows: List[List[Printed]] = []
    seen = False
    for ln in buf:
        if not seen:
            seen = start in ln
            continue
        if "|" not in ln:
            if rows and set(ln.strip()) <= set("-+"):
                break
            continue
        cells = ln.split("|", 1)[1].replace("|", " ").split()
        if cells and all(re.fullmatch(NUM, c) or c == "." for c in cells):
            rows.append([Printed(c) for c in cells if c != "."])
    return rows


def _cmp_pca(self: "Replay", cmd: str, buf: List[str], out: Any) -> None:
    for j, row in enumerate(_matrix_rows(buf, "Eigenvalue")):
        if j < len(out.eigenvalues):
            ours = out.eigenvalues.iloc[j]
            self.report.number(
                self.name, cmd, f"eigenvalue {j + 1}", row[0], ours["eigenvalue"]
            )
            self.report.number(
                self.name, cmd, f"proportion {j + 1}", row[-2], ours["proportion"]
            )
    for i, row in enumerate(_matrix_rows(buf, "Variable")):
        for j, cell in enumerate(row[: out.loadings.shape[1]]):
            self.report.number(
                self.name, cmd, f"loading[{i + 1},{j + 1}]", cell,
                out.loadings.iloc[i, j],
            )  # fmt: skip


def _cmp_factor(self: "Replay", cmd: str, buf: List[str], out: Any) -> None:
    # Stata's default tolerance for the iterative methods leaves four digits
    loose = 2e-4 if re.search(r",.*\b(?:ml|ipf)\b", cmd) else 0.0
    for j, row in enumerate(_matrix_rows(buf, "Eigenvalue")):
        if j < len(out.eigenvalues):
            self.report.number(
                self.name, cmd, f"eigenvalue {j + 1}", row[0],
                out.eigenvalues["eigenvalue"].iloc[j], rtol=loose,
            )  # fmt: skip
    k = out.loadings.shape[1]
    for i, row in enumerate(_matrix_rows(buf, "Variable")):
        for j, cell in enumerate(row[:k]):
            self.report.number(
                self.name, cmd, f"loading[{i + 1},{j + 1}]", cell,
                out.loadings.iloc[i, j], rtol=loose,
            )  # fmt: skip
        if len(row) > k:
            self.report.number(
                self.name, cmd, f"uniqueness[{i + 1}]", row[k],
                out.uniqueness.iloc[i], rtol=loose,
            )  # fmt: skip
    text = "\n".join(buf)
    found = re.findall(rf"Log likelihood = ({NUM})", text)  # the last: the header
    if found and out.loglik is not None:
        self.report.number(
            self.name, cmd, "log likelihood", Printed(found[-1]), out.loglik,
            rtol=1e-6,
        )  # fmt: skip
    m = re.search(rf"factors vs\. saturated:\s*chi2\(\d+\)\s*=\s*({NUM})", text)
    if m and out.lr_factors is not None:
        self.report.number(
            self.name, cmd, "LR chi2", Printed(m.group(1)), out.lr_factors["chi2"]
        )


def _cmp_estimates_stats(self: "Replay", cmd: str, buf: List[str], out: Any) -> None:
    for ln in buf:
        m = re.match(
            rf"^\s*(\S+)\s*\|\s*([\d,]+)\s+({NUM}|\.)\s+({NUM})\s+(\d+)"
            rf"\s+({NUM})\s+({NUM})\s*$",
            ln,
        )
        if not m or m.group(1) not in out.index:
            continue
        row = out.loc[m.group(1)]
        for label, group in (("N", 2), ("ll", 4), ("df", 5), ("AIC", 6), ("BIC", 7)):
            self.report.number(
                self.name, cmd, f"{label}[{m.group(1)}]",
                Printed(m.group(group).replace(",", "")), row[label],
            )  # fmt: skip


def _cmp_irf(self: "Replay", cmd: str, buf: List[str], out: Any) -> None:
    """`irf table`: the first numeric column after the step is the statistic
    (the others are its confidence limits, which are not computed here)."""
    if out.shape[1] != 1:
        return
    name = out.columns[0]
    for ln in buf:
        m = re.match(rf"^\s*(\d+)\s*\|\s*({NUM})", ln)
        if m and int(m.group(1)) in out.index:
            self.report.number(
                self.name, cmd, f"{name}[{m.group(1)}]", Printed(m.group(2)),
                out[name].loc[int(m.group(1))],
            )  # fmt: skip


_RESAMPLED = re.compile(
    r"^\s*(?:jackknife|jknife|bootstrap|bs)\b.*:|vce\(\s*(?:jack|boot)", re.I
)


def _cmp_resample(self: "Replay", cmd: str, buf: List[str], out: Any) -> None:
    """`jackknife` / `bootstrap`: the statistic and its standard error. A
    bootstrap standard error depends on the replications drawn, which are
    numpy's here and Stata's in the log."""
    random = re.search(r"\bboot|^\s*bs\b", cmd, re.I) is not None
    if hasattr(out, "params"):
        ours_b, ours_se = dict(out.params), dict(out.std_errors)
        for alias in ("Intercept", "const"):
            if alias in ours_b:
                ours_b["_cons"], ours_se["_cons"] = ours_b[alias], ours_se[alias]
    else:
        ours_b = {"_jk_1": out.estimate, "_bs_1": out.estimate}
        ours_se = {"_jk_1": out.se, "_bs_1": out.se}
    by_term = {stata_term(str(k)): k for k in ours_b}
    for name, (b, se) in coefficient_table(buf, self.labels).items():
        key = name if name in ours_b else by_term.get(stata_term(name))
        self.report.number(self.name, cmd, f"b[{name}]", b, ours_b.get(key))
        if random:
            self.report.add(self.name, cmd, f"se[{name}]", "random", stata=se.value)
        else:
            self.report.number(self.name, cmd, f"se[{name}]", se, ours_se.get(key))


MULTIVARIATE = {"pca": _cmp_pca, "factor": _cmp_factor}


def _cmp_kpss(self: "Replay", cmd: str, buf: List[str], out: Any) -> None:
    """`kpss y, maxlag(m)` prints the statistic at every lag order up to m;
    the translation asks for the last one."""
    rows = re.findall(rf"^\s*(\d+)\s+({NUM})\s*$", "\n".join(buf), re.M)
    if rows:
        self.report.number(
            self.name, cmd, f"KPSS[{rows[-1][0]}]", Printed(rows[-1][1]),
            getattr(out, "statistic", None),
        )  # fmt: skip


TIME_SERIES = {
    "kpss": _cmp_kpss,
    "vecrank": _cmp_vecrank,
    "vec": _cmp_vec,
    "veclmar": _cmp_var_table,
    "vecstable": _cmp_var_table,
    "dfuller": _cmp_dfuller,
    "var": _cmp_var,
    "varbasic": _cmp_var,
    "varsoc": _cmp_varsoc,
    "varwle": _cmp_var_table,
    "varlmar": _cmp_var_table,
    "varstable": _cmp_var_table,
    "vargranger": _cmp_var_table,
}


# ----------------------------------------------------------------- driver
def collect(paths: List[str]) -> List[Path]:
    found: List[Path] = []
    for raw in paths:
        p = Path(raw)
        found.extend(sorted(p.rglob("*.log")) if p.is_dir() else [p])
    return found


def replay(logs: List[Path], data_dirs: List[Path]) -> pd.DataFrame:
    report = Report()
    for log in logs:
        runner = Replay(f"{log.parent.name}/{log.name}", data_dirs, report)
        for cmd, buf in parse_log(log):
            runner.run(cmd, buf)
    frame = pd.DataFrame(report.rows)
    for col in ("stata", "ours", "units_off", "reason"):
        if col not in frame.columns:
            frame[col] = np.nan
    return frame


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("logs", nargs="+", help=".log files or folders")
    ap.add_argument(
        "--data", action="append", required=True, help="folder holding the .dta files"
    )
    ap.add_argument("--csv", help="write one row per comparison")
    ap.add_argument("--strict", action="store_true", help="exit 1 on any DIFF")
    args = ap.parse_args()

    frame = replay(collect(args.logs), [Path(d) for d in args.data])
    if frame.empty:
        print("nothing to compare")
        return 0
    print(f"StatsPAI {sp.__version__}\n")
    table = frame.groupby(["file", "status"]).size().unstack(fill_value=0)
    table = table.reindex(
        columns=["ok", "DIFF", "no output", "NOT RUN", "random"], fill_value=0
    )
    table.loc["TOTAL"] = table.sum()
    print(table.to_string())
    pd.set_option(
        "display.width", 220, "display.max_colwidth", 90, "display.max_rows", 400
    )
    diff = frame[frame.status == "DIFF"]
    if len(diff):
        print("\nDIFF (printed number not reproduced)")
        print(
            diff[["file", "command", "what", "stata", "ours", "units_off"]].to_string(
                index=False
            )
        )
    notrun = frame[frame.status == "NOT RUN"]
    if len(notrun):
        print("\nNOT RUN")
        for row in notrun.itertuples():
            print(f"  {row.file}: {row.command[:70]}\n      {str(row.reason)[:150]}")
    if args.csv:
        frame.to_csv(args.csv, index=False)
    return 1 if (args.strict and len(diff)) else 0


if __name__ == "__main__":
    sys.exit(main())
