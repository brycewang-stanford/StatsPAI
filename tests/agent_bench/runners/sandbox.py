"""Persistent code-execution sessions for CausalAgentBench trials.

Each trial gets one session in its own scratch directory, holding the
prompt's data file. The agent's ``run_python`` / ``run_r`` calls execute in
that session and keep their state between calls (a data frame loaded in one
call is there in the next), as in a notebook.

Three toolsets differ only in what the session can import:

* ``statspai`` -- Python with StatsPAI importable;
* ``pythonic`` -- the same Python with ``import statspai`` blocked, so the
  agent works with statsmodels / linearmodels / scikit-learn / DoubleML;
* ``r`` -- an R session.

A call that exceeds its time limit kills the session; the next call starts a
fresh one and is told so. Output is truncated to ``max_chars`` so a runaway
print cannot flood the model's context.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import sys
import tempfile
import textwrap
import threading
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

#: Python driver: runs code sent as JSON lines in one persistent namespace.
_PY_DRIVER = textwrap.dedent(r"""
    import contextlib, io, json, sys, traceback

    if sys.argv[1] == "block":
        class _Blocker:
            def find_spec(self, name, path=None, target=None):
                if name == "statspai" or name.startswith("statspai."):
                    raise ModuleNotFoundError(
                        "No module named 'statspai' (not available in this toolset)"
                    )
                return None
        sys.meta_path.insert(0, _Blocker())

    namespace = {"__name__": "__main__"}
    for line in sys.stdin:
        request = json.loads(line)
        out, err = io.StringIO(), io.StringIO()
        ok = True
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
            try:
                code = compile(request["code"], "<cell>", "exec")
                exec(code, namespace)
            except BaseException:
                ok = False
                traceback.print_exc()
        reply = {"ok": ok, "stdout": out.getvalue(), "stderr": err.getvalue()}
        sys.__stdout__.write(json.dumps(reply) + "\n")
        sys.__stdout__.flush()
    """)

#: R driver: reads a line count, then that many lines of code; evaluates them
#: in the global environment and prints the output between sentinels.
_R_DRIVER = textwrap.dedent(r"""
    # The driver's own state lives in a function environment, so user code
    # evaluated in the global environment cannot clobber it (or be clobbered:
    # a loop variable named `e` here once overwrote the user's `e`).
    .statspai_bench_main <- function() {
      con <- file("stdin", open = "r")
      repeat {
        header <- readLines(con, n = 1)
        if (length(header) == 0) break
        n <- as.integer(header)
        code <- if (n > 0) readLines(con, n = n) else character(0)
        ok <- TRUE
        out <- tryCatch(
          withCallingHandlers(
            capture.output({
              exprs <- parse(text = code)
              for (i in seq_along(exprs)) {
                res <- withVisible(eval(exprs[[i]], envir = globalenv()))
                if (res$visible) print(res$value)
              }
            }, type = "output"),
            warning = function(w) {
              message("Warning: ", conditionMessage(w))
              invokeRestart("muffleWarning")
            }
          ),
          error = function(err) { ok <<- FALSE; paste("Error:", conditionMessage(err)) }
        )
        cat(out, sep = "\n")
        cat("\n<<<STATSPAI-BENCH-END", if (ok) "OK" else "ERROR", ">>>\n")
        flush(stdout())
      }
    }
    .statspai_bench_main()
    """)


@dataclass
class CallResult:
    ok: bool
    output: str
    timed_out: bool = False
    restarted: bool = False


class _Session:
    language = ""

    def __init__(self, workdir: Path, *, timeout: float, max_chars: int) -> None:
        self.workdir = Path(workdir)
        self.timeout = timeout
        self.max_chars = max_chars
        self.proc: Optional[subprocess.Popen] = None
        self._restarted = False

    # -- lifecycle -------------------------------------------------------
    def _spawn(self) -> subprocess.Popen:  # pragma: no cover - abstract
        raise NotImplementedError

    def _ensure(self) -> subprocess.Popen:
        if self.proc is None or self.proc.poll() is not None:
            self.proc = self._spawn()
        return self.proc

    def close(self) -> None:
        if self.proc is not None and self.proc.poll() is None:
            self.proc.kill()
            self.proc.wait(timeout=5)
        self.proc = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # -- one call --------------------------------------------------------
    def _exchange(self, proc: subprocess.Popen, code: str) -> Optional[CallResult]:
        raise NotImplementedError  # pragma: no cover

    def run(self, code: str) -> CallResult:
        proc = self._ensure()
        box: dict = {}

        def target() -> None:
            try:
                box["result"] = self._exchange(proc, code)
            except Exception as exc:  # noqa: BLE001 - reported to the agent
                box["result"] = CallResult(False, f"session failure: {exc}")

        worker = threading.Thread(target=target, daemon=True)
        worker.start()
        worker.join(self.timeout)
        if worker.is_alive():
            self.close()
            self._restarted = True
            return CallResult(
                False,
                f"Timed out after {self.timeout:.0f} s; the session was "
                "restarted and its variables are gone.",
                timed_out=True,
            )
        result = box.get("result") or CallResult(False, "session ended unexpectedly")
        if proc.poll() is not None and not result.ok:
            self.proc = None
            self._restarted = True
        if self._restarted and not result.timed_out:
            result.restarted = True
        text = result.output
        if len(text) > self.max_chars:
            cut = len(text) - self.max_chars
            text = text[: self.max_chars] + f"\n... [{cut} characters truncated]"
        result.output = text
        return result


class PythonSession(_Session):
    language = "python"

    def __init__(
        self,
        workdir: Path,
        *,
        allow_statspai: bool,
        timeout: float = 300,
        max_chars: int = 8000,
        python: Optional[str] = None,
    ) -> None:
        super().__init__(workdir, timeout=timeout, max_chars=max_chars)
        self.allow_statspai = allow_statspai
        self.python = python or sys.executable

    def _spawn(self) -> subprocess.Popen:
        driver = self.workdir / ".bench_driver.py"
        driver.write_text(_PY_DRIVER, encoding="utf-8")
        env = dict(os.environ)
        env["MPLBACKEND"] = "Agg"
        env["PYTHONUNBUFFERED"] = "1"
        return subprocess.Popen(
            [self.python, str(driver), "allow" if self.allow_statspai else "block"],
            cwd=self.workdir,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            env=env,
        )

    def _exchange(self, proc: subprocess.Popen, code: str) -> CallResult:
        assert proc.stdin is not None and proc.stdout is not None
        proc.stdin.write(json.dumps({"code": code}) + "\n")
        proc.stdin.flush()
        line = proc.stdout.readline()
        if not line:
            return CallResult(False, "the Python session exited")
        reply = json.loads(line)
        text = reply["stdout"]
        if reply["stderr"]:
            text = (text + "\n" if text else "") + reply["stderr"]
        return CallResult(bool(reply["ok"]), text.rstrip())


class RSession(_Session):
    language = "r"

    def __init__(
        self,
        workdir: Path,
        *,
        timeout: float = 300,
        max_chars: int = 8000,
        rscript: Optional[str] = None,
    ) -> None:
        super().__init__(workdir, timeout=timeout, max_chars=max_chars)
        self.rscript = rscript or shutil.which("Rscript") or "Rscript"

    def _spawn(self) -> subprocess.Popen:
        driver = self.workdir / ".bench_driver.R"
        driver.write_text(_R_DRIVER, encoding="utf-8")
        return subprocess.Popen(
            [self.rscript, "--vanilla", str(driver)],
            cwd=self.workdir,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

    def _exchange(self, proc: subprocess.Popen, code: str) -> CallResult:
        assert proc.stdin is not None and proc.stdout is not None
        lines = code.split("\n")
        proc.stdin.write(f"{len(lines)}\n" + "\n".join(lines) + "\n")
        proc.stdin.flush()
        collected = []
        while True:
            line = proc.stdout.readline()
            if not line:
                return CallResult(False, "".join(collected) + "\nthe R session exited")
            if line.startswith("<<<STATSPAI-BENCH-END"):
                ok = "OK" in line
                return CallResult(ok, "".join(collected).rstrip())
            collected.append(line)


def make_workdir(data_files: dict, root: Optional[Path] = None) -> Path:
    """A fresh scratch directory holding copies of ``data_files``.

    ``data_files`` maps the file name the prompt uses to a source path.
    """
    base = Path(root) if root is not None else Path(tempfile.gettempdir())
    workdir = base / f"bench-{uuid.uuid4().hex[:10]}"
    workdir.mkdir(parents=True)
    for name, source in data_files.items():
        shutil.copyfile(source, workdir / name)
    return workdir
