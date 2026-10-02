#!/usr/bin/env python3
"""Bind the Track C timings to the code they timed, not to a version number.

``tests/perf/results/*.json`` records wall-clock timings that take a quiet
machine and well over an hour to produce (``tests/perf/run_when_idle.sh``).
Tying them to the package *version* meant a documentation-only release made
them look stale: 1.34.1 and 1.34.2 changed no estimator, and the paper that
quotes the timings still had to argue file by file that they held.

This script records what a timing actually depends on. For each Track C
module it runs the module once at its smallest size, with data and results
redirected to a scratch directory, under the same profiler as
``scripts/trace_parity_provenance.py``, and stores the SHA-256 of

* the module script and the shared harness (``_common.py``, ``_data.py``),
* every StatsPAI source file whose code ran inside the timed calls
  (``import statspai`` happens before profiling, so this is the timed path,
  not the package), including the Python source behind any Numba-compiled
  function that was called, and
* the Rust HDFE backend sources,

next to the package version the committed timings were measured with. The
record is ``tests/perf/results/_timed_path.json``.

``--check`` answers one question: do the committed timings still describe
this tree? They do while none of those files has changed. A stale answer
names the files. Hashes use the ASCII-normalised bytes of
``scripts/ascii_source.py`` with the ``__version__`` line masked, so a release
bump alone does not stale them and the check also holds in the JSS archive.

The smallest size exercises the same code as the largest; only ``n`` differs.

Usage::

    python scripts/trace_perf_path.py            # trace every module, write the record
    python scripts/trace_perf_path.py 02 04      # selected modules
    python scripts/trace_perf_path.py --check    # are the committed timings current?
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import runpy
import subprocess
import sys
import tempfile
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PERF = REPO / "tests" / "perf"
RECORD = PERF / "results" / "_timed_path.json"
MODULES = ("01_hdfe", "02_csdid", "03_scm", "04_dml")
HARNESS = ("tests/perf/_common.py", "tests/perf/_data.py")
RUST_DIR = REPO / "rust" / "statspai_hdfe"

if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))

from trace_parity_provenance import _sha256  # noqa: E402


def _rust_sources() -> dict:
    files = [RUST_DIR / "Cargo.toml", *sorted((RUST_DIR / "src").rglob("*.rs"))]
    return {
        str(p.relative_to(REPO))
        .replace(os.sep, "/"): hashlib.sha256(p.read_bytes())
        .hexdigest()
        for p in files
        if p.exists()
    }


def _measured_version(stem: str) -> str | None:
    path = PERF / "results" / f"{stem}_py.json"
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return (data.get("hardware") or {}).get("statspai_version")


def _trace_child(stem: str, scratch: str) -> dict:
    """Run one perf module at its smallest size under the profiler."""
    src_root = str(REPO / "src" / "statspai")
    exercised: set = set()

    sys.path.insert(0, str(PERF))
    import _common
    import _data

    # One repetition at the smallest size, nothing written into the tree.
    _data.SIZES = {k: [min(v)] for k, v in _data.SIZES.items()}
    _data.DATA_DIR = Path(scratch) / "data"
    _data.DATA_DIR.mkdir(parents=True, exist_ok=True)
    _common.RESULTS_DIR = Path(scratch) / "results"
    _common.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    real_time_repeat = _common.time_repeat
    _common.time_repeat = lambda fn, n_reps=1, warmup=0: real_time_repeat(fn, 1, 0)

    def profiler(frame, event, arg):  # noqa: ANN001
        if event == "call":
            filename = frame.f_code.co_filename
        elif event == "c_call":
            # A Numba dispatcher is a C callable; its Python source is the
            # code that was compiled and is what a timing depends on.
            py_func = getattr(getattr(arg, "__self__", None), "py_func", None)
            code = getattr(py_func, "__code__", None)
            if code is None:
                return
            filename = code.co_filename
        else:
            return
        if filename.startswith(src_root):
            exercised.add(filename)

    import statspai  # noqa: F401  (imported before profiling: see the docstring)

    error = None
    old_argv = sys.argv
    sys.argv = [str(PERF / f"{stem}_perf.py")]
    sys.setprofile(profiler)
    try:
        runpy.run_path(str(PERF / f"{stem}_perf.py"), run_name="__main__")
    except SystemExit:
        pass
    except Exception as exc:  # recorded, never swallowed silently
        error = f"{type(exc).__name__}: {exc}"
    finally:
        sys.setprofile(None)
        sys.argv = old_argv

    repo = str(REPO) + os.sep
    return {
        "error": error,
        "exercised_sources": {
            f[len(repo) :].replace(os.sep, "/"): _sha256(Path(f))
            for f in sorted(exercised)
            if f.endswith(".py") and Path(f).exists()
        },
    }


def _trace(stem: str) -> dict:
    with tempfile.TemporaryDirectory() as scratch:
        env = dict(os.environ)
        env["PYTHONPATH"] = os.pathsep.join(
            [str(REPO / "src"), str(PERF), env.get("PYTHONPATH", "")]
        )
        env.setdefault("MPLBACKEND", "Agg")
        proc = subprocess.run(
            [sys.executable, __file__, "--child", stem, "--scratch", scratch],
            cwd=PERF,
            env=env,
            capture_output=True,
            text=True,
            timeout=7200,
        )
    marker = "@@TIMEDPATH@@"
    for line in proc.stdout.splitlines():
        if line.startswith(marker):
            rec = json.loads(line[len(marker) :])
            break
    else:
        rec = {
            "error": f"child exited {proc.returncode}: {proc.stderr[-2000:]}",
            "exercised_sources": {},
        }
    rec["script_sha256"] = _sha256(PERF / f"{stem}_perf.py")
    rec["harness_sha256"] = {rel: _sha256(REPO / rel) for rel in HARNESS}
    rec["rust_sources"] = _rust_sources()
    rec["timings_measured_with"] = _measured_version(stem)
    return rec


def stale_reasons(stem: str, rec: dict) -> list:
    """Why the committed timings of ``stem`` no longer describe this tree."""
    reasons = []
    if rec.get("error"):
        reasons.append(f"trace recorded an error: {rec['error']}")
    script = PERF / f"{stem}_perf.py"
    if rec.get("script_sha256") != (_sha256(script) if script.exists() else None):
        reasons.append(f"tests/perf/{stem}_perf.py changed")
    for rel, digest in (rec.get("harness_sha256") or {}).items():
        if not (REPO / rel).exists() or _sha256(REPO / rel) != digest:
            reasons.append(f"{rel} changed")
    sources = rec.get("exercised_sources")
    if not sources:
        reasons.append("no timed-path sources recorded")
    for rel, digest in (sources or {}).items():
        if not (REPO / rel).exists() or _sha256(REPO / rel) != digest:
            reasons.append(f"{rel} changed")
    if rec.get("rust_sources") != _rust_sources():
        reasons.append("rust/statspai_hdfe sources changed")
    if rec.get("timings_measured_with") != _measured_version(stem):
        reasons.append(
            f"tests/perf/results/{stem}_py.json was re-measured; re-trace to bind it"
        )
    return reasons


def check() -> int:
    if not RECORD.exists():
        print(f"FAIL -- {RECORD.relative_to(REPO)} is missing; run this script")
        return 1
    record = json.loads(RECORD.read_text(encoding="utf-8"))
    bad = 0
    for stem in MODULES:
        rec = record.get(stem)
        reasons = ["never traced"] if rec is None else stale_reasons(stem, rec)
        if reasons:
            bad += 1
            print(f"STALE {stem}: " + "; ".join(reasons[:6]))
        else:
            print(
                f"OK    {stem}: timings measured with StatsPAI "
                f"{rec['timings_measured_with']} still describe this tree "
                f"({len(rec['exercised_sources'])} timed-path files unchanged)"
            )
    if bad:
        print(
            "Re-run tests/perf/run_when_idle.sh for the stale modules on a quiet "
            "machine, then this script, and commit both."
        )
    return 1 if bad else 0


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("modules", nargs="*", help="module number prefixes")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--child", help=argparse.SUPPRESS)
    ap.add_argument("--scratch", help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.child:
        print("@@TIMEDPATH@@" + json.dumps(_trace_child(args.child, args.scratch)))
        return 0
    if args.check:
        return check()
    stems = [m for m in MODULES if not args.modules or m[:2] in args.modules]
    # The record binds timings to the tree that produced them, so it has to be
    # traced on that tree: tracing newer code would bind old timings to it.
    text = (REPO / "pyproject.toml").read_text(encoding="utf-8")
    tree_version = text.split('version = "', 1)[1].split('"', 1)[0]
    mismatched = {
        stem: _measured_version(stem)
        for stem in stems
        if _measured_version(stem) != tree_version
    }
    if mismatched:
        print(
            f"REFUSED -- this tree is StatsPAI {tree_version}, but the committed "
            f"timings were measured with {mismatched}. Trace on a checkout of the "
            "measured release (or re-measure with tests/perf/run_when_idle.sh)."
        )
        return 2
    record = json.loads(RECORD.read_text(encoding="utf-8")) if RECORD.exists() else {}
    failed = 0
    for stem in stems:
        rec = _trace(stem)
        record[stem] = rec
        status = "ERROR " + rec["error"] if rec.get("error") else "ok"
        failed += bool(rec.get("error"))
        print(f"{stem}: {status}; {len(rec['exercised_sources'])} timed-path files")
    RECORD.write_text(
        json.dumps({k: record[k] for k in sorted(record)}, indent=1) + "\n",
        encoding="utf-8",
    )
    print(f"wrote {RECORD.relative_to(REPO)}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
