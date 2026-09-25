#!/usr/bin/env python3
"""Call-trace audit: which implementation does each Track A row exercise?

A Track A parity row compares a StatsPAI number against an R (or Stata)
reference on identical bytes.  That comparison is evidence about a
*StatsPAI-native algorithm* only if the StatsPAI side actually ran one.  A
row whose Python side hands the computation to the R reference itself
(``backend="honestdid"``), to a port maintained by the method's authors
(``rdrobust``'s Python package), or to a third-party Python estimation
library (``linearmodels``) is still useful -- it checks the wrapper,
argument mapping and conventions -- but it is a different kind of
evidence and must be counted separately.

Reading the module source is not enough to decide this: a public
function can delegate internally (``sp.panel(method="fe")`` fits through
``linearmodels``).  So this script *runs* every Track A Python module,
with its outputs redirected to a scratch directory so no committed
fixture is touched, under a profiler that records

* every call that crosses from ``statspai`` code into a non-StatsPAI
  package other than the numerical substrate (NumPy, SciPy, pandas and
  the standard library), and
* every subprocess launched (an ``Rscript`` launch is a reference
  backend by definition).

The trace is bound to what it describes. For each module it records the
SHA-256 of the module script, of every StatsPAI source file whose code ran
while the module estimated (imports of ``statspai`` itself happen before
the profiler starts, so the set is the estimation path, not the package),
and of the committed ``<module>_py.json`` result, plus the versions of the
packages the module crossed into. Changing *any* of those files -- including
an internal delegation change behind an unchanged entry script -- makes the
trace stale, and the contract test refuses it until the module is re-traced.

The result is ``tests/r_parity/results/_implementation_trace.json``.
``tests/test_parity_implementation_provenance.py`` then asserts that the
hand-registered classification in
``tests/r_parity/compare.py::IMPLEMENTATION_PROVENANCE`` agrees with the
trace, so a module cannot silently change evidence type. A package that is
neither substrate nor in ``KNOWN`` classifies the module as
``unclassified`` -- it must be reviewed and added, never assumed native.

Usage::

    python scripts/trace_parity_provenance.py            # all modules
    python scripts/trace_parity_provenance.py 10 21 35   # selected
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
import time
from collections import Counter
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PARITY = REPO / "tests" / "r_parity"
if str(REPO / "scripts") not in sys.path:
    sys.path.insert(0, str(REPO / "scripts"))

from ascii_source import normalized_source_bytes  # noqa: E402

OUT = PARITY / "results" / "_implementation_trace.json"

#: Packages that are numerical substrate, not estimator implementations.
#: Calls into these never change a row's evidence type.
SUBSTRATE = (
    "numpy",
    "scipy",
    "pandas",
    "numba",
    "llvmlite",
    "patsy",
    "formulaic",
    "dateutil",
    "pytz",
    "tzdata",
    "joblib",
    "threadpoolctl",
    "matplotlib",
    "packaging",
    "typing_extensions",
    "importlib_metadata",
    "_distutils_hack",
    "cloudpickle",
)

#: Packages whose *estimators* would make a row a third-party or official-port
#: row.  Anything else outside the substrate is also recorded (unknown
#: packages are surfaced, never silently ignored).
KNOWN = {
    "rdrobust": "official_python_port",
    "rddensity": "official_python_port",
    "rdd": "official_python_port",
    "linearmodels": "third_party_python",
    "statsmodels": "third_party_python",
    "pyfixest": "third_party_python",
    "econml": "third_party_python",
    "doubleml": "third_party_python",
    "lifelines": "third_party_python",
    "sklearn": "nuisance_learner",
    "lightgbm": "nuisance_learner",
    "xgboost": "nuisance_learner",
}


def _top_package(filename: str) -> str | None:
    """Top-level package of a code object's file, or None for stdlib/builtin."""
    parts = Path(filename).parts
    if "site-packages" in parts:
        i = parts.index("site-packages")
        if i + 1 < len(parts):
            name = parts[i + 1]
            return name[:-3] if name.endswith(".py") else name
    return None


def _sha256(path: Path) -> str:
    # Hashed ASCII-normalized, as the JSS archive ships source files
    # (scripts/ascii_source.py), so the freshness check also holds there.
    return hashlib.sha256(normalized_source_bytes(path)).hexdigest()


def _run_one(module_path: Path) -> dict:
    """Execute one Track A module in-process under the boundary profiler."""
    src_root = str(REPO / "src" / "statspai")
    calls: Counter = Counter()
    exercised: set = set()
    launched: list[list[str]] = []

    real_popen_init = subprocess.Popen.__init__

    def popen_spy(self, args, *a, **kw):  # noqa: ANN001
        argv = [str(x) for x in (args if isinstance(args, (list, tuple)) else [args])]
        launched.append(argv)
        return real_popen_init(self, args, *a, **kw)

    def profiler(frame, event, arg):  # noqa: ANN001
        if event != "call":
            return
        callee = frame.f_code.co_filename
        if callee.startswith(src_root):
            exercised.add(callee)
            return
        caller = frame.f_back
        if caller is None or not caller.f_code.co_filename.startswith(src_root):
            return
        pkg = _top_package(callee)
        if pkg is None or pkg.split(".")[0] in SUBSTRATE:
            return
        qual = getattr(frame.f_code, "co_qualname", frame.f_code.co_name)
        mod = frame.f_globals.get("__name__", pkg)
        where = caller.f_code.co_filename[len(src_root) + 1 :]
        calls[(pkg, f"{mod}.{qual}", where)] += 1

    t0 = time.perf_counter()
    subprocess.Popen.__init__ = popen_spy  # type: ignore[method-assign]
    old_argv, old_path = sys.argv, list(sys.path)
    sys.argv = [str(module_path)]
    sys.path.insert(0, str(PARITY))
    error = None
    # Import the package before profiling starts: the files recorded as
    # exercised are then the ones on this module's estimation path, not
    # every module ``import statspai`` loads.
    import statspai  # noqa: F401

    sys.setprofile(profiler)
    try:
        runpy.run_path(str(module_path), run_name="__main__")
    except SystemExit:
        pass
    except Exception as exc:  # recorded, never swallowed silently
        error = f"{type(exc).__name__}: {exc}"
    finally:
        sys.setprofile(None)
        subprocess.Popen.__init__ = real_popen_init  # type: ignore[method-assign]
        sys.argv, sys.path[:] = old_argv, old_path

    boundary = [
        {"package": p, "callee": c, "caller": w, "n": n}
        for (p, c, w), n in sorted(calls.items())
    ]
    r_launch = [a for a in launched if any("Rscript" in x or x == "R" for x in a[:1])]
    packages = sorted({b["package"] for b in boundary})
    versions = {}
    from importlib import metadata

    for pkg in packages + ["numpy", "scipy", "pandas", "scikit-learn"]:
        dist = {"sklearn": "scikit-learn"}.get(pkg, pkg)
        try:
            versions[dist] = metadata.version(dist)
        except metadata.PackageNotFoundError:
            versions[dist] = None
    repo = str(REPO) + os.sep
    return {
        "seconds": round(time.perf_counter() - t0, 2),
        "error": error,
        "boundary_calls": boundary,
        "packages": packages,
        "rscript_launches": len(r_launch),
        "subprocesses": [a[:2] for a in launched],
        "exercised_sources": {
            f[len(repo) :].replace(os.sep, "/"): _sha256(Path(f))
            for f in sorted(exercised)
            if f.endswith(".py") and Path(f).exists()
        },
        "dependency_versions": versions,
    }


def stale_reasons(stem: str, rec: dict, root: Path = REPO) -> list:
    """Why a module's trace no longer describes the current tree ([] if current).

    The trace is stale if the entry script, any StatsPAI file on the traced
    estimation path, or the committed result changed since it was recorded,
    or if the trace predates exercised-source recording.
    """
    parity = root / "tests" / "r_parity"
    reasons = []
    script = parity / f"{stem}.py"
    if rec.get("source_sha256") != (_sha256(script) if script.exists() else None):
        reasons.append(f"entry script {script.relative_to(root)} changed")
    sources = rec.get("exercised_sources")
    if not sources:
        reasons.append("trace predates exercised-source recording")
    else:
        for rel, digest in sources.items():
            path = root / rel
            if not path.exists() or _sha256(path) != digest:
                reasons.append(f"{rel} changed")
    result = parity / "results" / f"{stem}_py.json"
    if "result_sha256" not in rec or rec["result_sha256"] != (
        _sha256(result) if result.exists() else None
    ):
        reasons.append(f"result {result.relative_to(root)} changed")
    return reasons


def _worker(stem: str, scratch: str) -> dict:
    """Run one module in a fresh interpreter so imports and state don't leak."""
    env = dict(os.environ)
    env["STATSPAI_R_PARITY_RESULTS_DIR"] = str(Path(scratch) / "results")
    env["STATSPAI_R_PARITY_DATA_DIR"] = str(Path(scratch) / "data")
    env["PYTHONPATH"] = os.pathsep.join(
        [str(REPO / "src"), str(PARITY), env.get("PYTHONPATH", "")]
    )
    env.setdefault("MPLBACKEND", "Agg")
    proc = subprocess.run(
        [sys.executable, __file__, "--child", stem],
        cwd=PARITY,
        env=env,
        capture_output=True,
        text=True,
        timeout=3600,
    )
    marker = "@@TRACE@@"
    for line in proc.stdout.splitlines():
        if line.startswith(marker):
            return json.loads(line[len(marker) :])
    return {
        "error": f"child exited {proc.returncode}: {proc.stderr[-2000:]}",
        "boundary_calls": [],
        "packages": [],
        "rscript_launches": 0,
        "subprocesses": [],
    }


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("modules", nargs="*", help="module number prefixes")
    ap.add_argument("--child", help=argparse.SUPPRESS)
    args = ap.parse_args()

    if args.child:
        path = PARITY / f"{args.child}.py"
        import warnings

        warnings.simplefilter("ignore")
        rec = _run_one(path)
        sys.stdout.write("\n@@TRACE@@" + json.dumps(rec) + "\n")
        return

    stems = sorted(p.stem for p in PARITY.glob("[0-9][0-9]*_*.py"))
    if args.modules:
        stems = [s for s in stems if s.split("_")[0] in set(args.modules)]
    existing = json.loads(OUT.read_text(encoding="utf-8")) if OUT.exists() else {}
    modules = dict(existing.get("modules", {}))
    with tempfile.TemporaryDirectory() as scratch:
        # Modules that read an already-frozen CSV read the committed copy
        # (``COMMITTED_DATA_DIR``); modules that dump one write to scratch.
        for stem in stems:
            rec = _worker(stem, scratch)
            # The trace is evidence about *this* source; a later edit to the
            # module invalidates it, and the contract test checks the hash.
            rec["source_sha256"] = _sha256(PARITY / f"{stem}.py")
            result = PARITY / "results" / f"{stem}_py.json"
            rec["result_sha256"] = _sha256(result) if result.exists() else None
            modules[stem] = rec
            status = "ERROR" if rec.get("error") else "ok"
            print(
                f"{stem:32s} {status:5s} R={rec['rscript_launches']} "
                f"pkgs={','.join(rec['packages']) or '-'}",
                flush=True,
            )
    payload = {
        "generator": "scripts/trace_parity_provenance.py",
        "python": sys.version.split()[0],
        "substrate_ignored": list(SUBSTRATE),
        "known_package_classes": KNOWN,
        "modules": dict(sorted(modules.items())),
    }
    OUT.write_text(
        json.dumps(payload, indent=1, sort_keys=False) + "\n", encoding="utf-8"
    )
    print(f"wrote {OUT.relative_to(REPO)} ({len(modules)} modules)")


if __name__ == "__main__":
    main()
