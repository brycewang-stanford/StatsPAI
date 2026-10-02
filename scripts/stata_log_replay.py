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

Use it as a probe: a ``DIFF`` is either a translation that asks for the
wrong convention or an estimator bug, and has to be traced to which. Do not
fix what it finds by special-casing the file it was found in.
"""

from __future__ import annotations

import argparse
import re
import sys
import warnings
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

import statspai as sp
from statspai.agent._translation._stata_expr import StataExprError
from statspai.agent._translation._stata_run import StataSession

NUM = r"[-+]?(?:\d+\.?\d*|\.\d+)(?:[eE][-+]?\d+)?"
ESTIMATION = {
    "reg", "regress", "xtreg", "areg", "probit", "logit", "poisson", "nbreg",
    "ivreg", "ivreg2", "ivregress", "tobit", "newey", "reghdfe",
}  # fmt: skip
SUMMARIZE = {"su", "sum", "summ", "summarize"}


# ------------------------------------------------------------- the log
def parse_log(path: Path) -> List[Tuple[str, List[str]]]:
    """``[(command, lines it printed), ...]`` in the order they ran."""
    lines = path.read_text(encoding="latin-1").splitlines()
    out: List[Tuple[str, List[str]]] = []
    cmd: Optional[str] = None
    buf: List[str] = []
    for ln in lines:
        if ln.startswith(". "):
            if cmd is not None:
                out.append((cmd, buf))
            cmd, buf = ln[2:].strip(), []
        elif ln.startswith("> ") and cmd is not None and not buf:
            cmd += " " + ln[2:].strip()  # a wrapped command line
        elif cmd is not None:
            buf.append(ln)
    if cmd is not None:
        out.append((cmd, buf))
    cleaned = []
    for c, b in out:
        c = c.rstrip(";").strip()
        if c and not c.startswith("*") and not c.startswith("//"):
            cleaned.append((c, b))
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

    def matches(self, ours: float) -> bool:
        # half a unit is rounding; the rest is float storage and, for
        # likelihood estimators, where two optimisers stop
        return bool(abs(ours - self.value) <= 2.0 * self.unit)


def coefficient_table(buf: List[str]) -> Dict[str, Tuple[Printed, Printed]]:
    rows: Dict[str, Tuple[Printed, Printed]] = {}
    for ln in buf:
        m = re.match(rf"^\s*(\S+)\s*\|\s*({NUM})\s+({NUM})\s+({NUM})\s+({NUM})", ln)
        if m:
            rows[m.group(1)] = (Printed(m.group(2)), Printed(m.group(3)))
    return rows


# ------------------------------------------------------------- the replay
class Report:
    def __init__(self) -> None:
        self.rows: List[Dict[str, Any]] = []

    def add(self, file: str, cmd: str, what: str, status: str, **kw: Any) -> None:
        self.rows.append(
            {"file": file, "command": cmd, "what": what, "status": status, **kw}
        )

    def number(
        self, file: str, cmd: str, what: str, printed: Printed, ours: Any
    ) -> None:
        if ours is None or (isinstance(ours, float) and np.isnan(ours)):
            self.add(file, cmd, what, "no output", stata=printed.value)
            return
        ours = float(ours)
        status = "ok" if printed.matches(ours) else "DIFF"
        self.add(file, cmd, what, status, stata=printed.value, ours=ours,
                 units_off=abs(ours - printed.value) / printed.unit)  # fmt: skip


class Replay:
    def __init__(self, name: str, data_dirs: List[Path], report: Report) -> None:
        self.name = name
        self.data_dirs = data_dirs
        self.report = report
        self.session: Optional[StataSession] = None

    # -- helpers ----------------------------------------------------------
    def _load(self, cmd: str) -> None:
        m = re.search(r'"([^"]+)"', cmd) or re.match(r"use\s+(\S+)", cmd)
        if m is None:
            raise FileNotFoundError("cannot read the file name")
        stem = Path(m.group(1).split(",")[0]).name
        if not stem.lower().endswith(".dta"):
            stem += ".dta"
        for root in self.data_dirs:
            hits = [p for p in root.rglob("*.dta") if p.name.lower() == stem.lower()]
            if hits:
                data = pd.read_stata(hits[0], convert_categoricals=False)
                self.session = StataSession(data)
                return
        raise FileNotFoundError(f"{stem} is not under --data")

    # -- one command ------------------------------------------------------
    def run(self, cmd: str, buf: List[str]) -> None:
        word = cmd.split()[0].rstrip(",").lower()
        quiet = False
        while word in ("qui", "quietly", "cap", "capture", "noi", "noisily"):
            cmd = cmd.split(None, 1)[1]
            word = cmd.split()[0].rstrip(",").lower()
            quiet = True
        try:
            self._dispatch(cmd, word, buf, quiet)
        except (KeyError, FileNotFoundError, StataExprError) as exc:
            self.report.add(self.name, cmd, "-", "NOT RUN", reason=str(exc)[:200])
        except Exception as exc:  # noqa: BLE001 - every refusal belongs in the report
            reason = str(exc).split("\n")[0][:200]
            self.report.add(self.name, cmd, "-", "NOT RUN", reason=reason)

    def _dispatch(self, cmd: str, word: str, buf: List[str], quiet: bool) -> None:
        if word == "use":
            self._load(cmd)
            return
        if word in ("clear", "cd", "do", "exit", "#", "#delimit", "vce", "matrix"):
            return
        if self.session is None:
            if word in ("set", "log", "version"):
                return
            raise FileNotFoundError("no data in memory")
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
        elif word == "test":
            self._test(cmd, buf, out, quiet)
        elif word == "ttest":
            self._ttest(cmd, buf, out)
        elif word in ESTIMATION and not quiet:
            self._estimates(cmd, buf, out, word)

    def _estimates(self, cmd: str, buf: List[str], res: Any, word: str) -> None:
        params, ses = dict(res.params), dict(res.std_errors)
        for alias in ("Intercept", "const"):
            if alias in params:
                params["_cons"], ses["_cons"] = params[alias], ses[alias]
        for name, (b, se) in coefficient_table(buf).items():
            if name in ("sigma_u", "sigma_e", "rho", "/sigma", "var(e.y)"):
                continue
            self.report.number(self.name, cmd, f"b[{name}]", b, params.get(name))
            self.report.number(self.name, cmd, f"se[{name}]", se, ses.get(name))

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
            ("diff", rf"^\s*diff \|\s*({NUM})", res.estimate),
            ("se(diff)", rf"^\s*diff \|\s*{NUM}\s+({NUM})", res.se),
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
                self.report.number(
                    self.name,
                    cmd,
                    f"mean[{m.group(1)}]",
                    Printed(m.group(3)),
                    row["Mean"],
                )
                self.report.number(
                    self.name,
                    cmd,
                    f"sd[{m.group(1)}]",
                    Printed(m.group(4)),
                    row["Std. Dev."],
                )

    def _display(self, cmd: str, buf: List[str]) -> None:
        m = re.match(r'(?:display|dis|di)\s+"[^"]*"\s+(.+)$', cmd)
        printed = re.findall(NUM, " ".join(b for b in buf if b.strip()))
        if m is None or not printed:
            return
        expr = m.group(1).strip()
        if "_result(8)" in expr:  # pre-Stata-6 spelling of e(r2_a)
            expr = "e(r2_a)"
        if '"' in expr or "_skip" in expr:
            return  # several expressions on one line: not compared
        assert self.session is not None
        self.report.number(
            self.name,
            cmd,
            "displayed value",
            Printed(printed[-1]),
            self.session.value(expr),
        )


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
    table = table.reindex(columns=["ok", "DIFF", "no output", "NOT RUN"], fill_value=0)
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
