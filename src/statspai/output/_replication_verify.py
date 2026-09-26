"""``sp.verify_replication_pack`` -- does a replication archive actually rerun?

A zip that opens is not a replication. This unpacks an archive written by
:func:`sp.replication_pack` into a fresh directory, checks every file against
the SHA-256 in ``MANIFEST.json``, runs ``code/script.py`` there in a separate
Python process, and compares the numbers the rerun produces with the numbers
recorded in the pack (``results/results.json``).

The rerun's numbers are captured without touching the user's script: the
subprocess runs with ``STATSPAI_CAPTURE_RESULTS`` pointing at a file, and
every StatsPAI estimator that attaches a provenance record appends its
headline coefficients and standard errors there. Results are matched on
(function, call arguments); a match on the arguments but a different input
data fingerprint is reported, since the same call on different data is the
most common way a replication silently drifts.

The rerun uses the *current* interpreter and environment. It checks that the
code and data in the archive reproduce the recorded numbers here; installing
``env/requirements.txt`` into a clean environment first (see the pack's
README) is what makes it a third-party replication.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import subprocess
import sys
import tempfile
import time
import zipfile
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

__all__ = ["verify_replication_pack", "ReplicationVerification"]


class ReplicationVerification(dict):
    """Outcome of :func:`verify_replication_pack` (a ``dict``).

    Keys: ``status`` (``"verified"``, ``"mismatch"``, ``"rerun_failed"``,
    ``"integrity_failed"``, ``"incomplete_pack"``), ``integrity``,
    ``rerun``, ``comparison`` and ``notes``.

    Examples
    --------
    >>> import statspai as sp
    >>> sp.ReplicationVerification({"status": "verified"})["status"]
    'verified'
    """

    __slots__ = ()

    def summary(self) -> str:
        cmp_ = self.get("comparison") or {}
        lines = [
            f"Replication verification: {self.get('status')}",
            f"  integrity : {(self.get('integrity') or {}).get('status')}",
        ]
        rerun = self.get("rerun") or {}
        if rerun:
            lines.append(
                f"  rerun     : exit {rerun.get('returncode')} in "
                f"{rerun.get('seconds')} s"
            )
        if cmp_:
            lines.append(
                f"  results   : {cmp_.get('n_matched', 0)} matched, "
                f"{len(cmp_.get('mismatched', []))} mismatched, "
                f"{len(cmp_.get('missing', []))} missing in rerun, "
                f"{len(cmp_.get('extra', []))} extra"
            )
        for n in self.get("notes", []):
            lines.append(f"  note      : {n}")
        return "\n".join(lines)

    __str__ = summary

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return self.summary()


def _key(rec: Dict[str, Any]) -> str:
    return json.dumps(
        {"function": rec.get("function"), "params": rec.get("params")},
        sort_keys=True,
        default=str,
    )


def _close(a: Optional[float], b: Optional[float], rtol: float, atol: float) -> bool:
    if a is None or b is None:
        return a is None and b is None
    return math.isclose(a, b, rel_tol=rtol, abs_tol=atol)


def _compare(
    packed: List[Dict[str, Any]],
    rerun: List[Dict[str, Any]],
    rtol: float,
    atol: float,
) -> Dict[str, Any]:
    pool: Dict[str, List[Dict[str, Any]]] = {}
    for rec in rerun:
        pool.setdefault(_key(rec), []).append(rec)
    matched = 0
    mismatched: List[Dict[str, Any]] = []
    missing: List[str] = []
    data_changed: List[str] = []
    for rec in packed:
        cands = pool.get(_key(rec)) or []
        if not cands:
            missing.append(rec.get("function", "?"))
            continue
        got = cands.pop(0)
        if got.get("data_hash") != rec.get("data_hash"):
            data_changed.append(rec.get("function", "?"))
        diffs: List[Dict[str, Any]] = []
        want_nums: Dict[str, Any] = rec.get("numbers") or {}
        got_nums: Dict[str, Any] = got.get("numbers") or {}
        for name, (est, se) in want_nums.items():
            if name not in got_nums:
                diffs.append({"term": name, "issue": "absent in rerun"})
                continue
            g_est, g_se = got_nums[name]
            if not (_close(est, g_est, rtol, atol) and _close(se, g_se, rtol, atol)):
                diffs.append(
                    {
                        "term": name,
                        "packed": [est, se],
                        "rerun": [g_est, g_se],
                    }
                )
        if diffs:
            mismatched.append({"function": rec.get("function"), "differences": diffs})
        else:
            matched += 1
    extra = [r.get("function", "?") for recs in pool.values() for r in recs]
    return {
        "n_packed": len(packed),
        "n_matched": matched,
        "mismatched": mismatched,
        "missing": missing,
        "extra": extra,
        "data_changed": data_changed,
        "rtol": rtol,
        "atol": atol,
    }


def _integrity(root: Path, manifest: Dict[str, Any]) -> Dict[str, Any]:
    bad: List[str] = []
    absent: List[str] = []
    for entry in manifest.get("files", []):
        path = root / entry["path"]
        if not path.exists():
            absent.append(entry["path"])
            continue
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry.get("sha256"):
            bad.append(entry["path"])
    status = "ok" if not bad and not absent else "failed"
    return {"status": status, "hash_mismatch": bad, "absent": absent}


def verify_replication_pack(
    path: Union[str, os.PathLike],
    *,
    timeout: float = 900.0,
    rtol: float = 1e-8,
    atol: float = 1e-12,
    python: Optional[str] = None,
    keep_dir: Optional[Union[str, os.PathLike]] = None,
) -> ReplicationVerification:
    """Unpack, integrity-check, rerun and diff a replication archive.

    Parameters
    ----------
    path : str or PathLike
        Archive written by :func:`sp.replication_pack`.
    timeout : float, default 900
        Seconds allowed for ``code/script.py``.
    rtol, atol : float
        Tolerance for each recorded estimate and standard error. The
        defaults allow floating-point reordering, not a different answer.
    python : str, optional
        Interpreter for the rerun; defaults to the current one. Point it at
        a clean virtual environment built from ``env/requirements.txt`` for
        a third-party replication.
    keep_dir : str or PathLike, optional
        Unpack here (and keep it) instead of a temporary directory.

    Returns
    -------
    ReplicationVerification
        ``status`` is ``"verified"`` only when every file matches its hash,
        the script exits 0, and every recorded result is reproduced within
        tolerance. ``"incomplete_pack"`` means there is no script or no
        recorded results to check against (build with ``strict=True``).

    Examples
    --------
    >>> import os, tempfile, textwrap
    >>> import statspai as sp
    >>> df = sp.cps_wage()
    >>> fit = sp.regress("log_wage ~ education", data=df)
    >>> script = textwrap.dedent('''
    ...     import pandas as pd, statspai as sp
    ...     df = pd.read_csv("data/dataset.csv")
    ...     sp.regress("log_wage ~ education", data=df)
    ... ''')
    >>> out = os.path.join(tempfile.mkdtemp(), "pack.zip")
    >>> _ = sp.replication_pack(fit, out, data=df, code=script, env=False,
    ...                         bib=False, include_git_sha=False)
    >>> sp.verify_replication_pack(out)["status"]  # doctest: +SKIP
    'verified'
    """
    src = Path(path).expanduser().resolve()
    notes: List[str] = []
    tmp: Optional[tempfile.TemporaryDirectory] = None
    if keep_dir is not None:
        root = Path(keep_dir).expanduser().resolve()
        root.mkdir(parents=True, exist_ok=True)
    else:
        tmp = tempfile.TemporaryDirectory(prefix="statspai-verify-")
        root = Path(tmp.name).resolve()
    try:
        with zipfile.ZipFile(src) as zf:
            for member in zf.namelist():
                target = (root / member).resolve()
                if target != root and root not in target.parents:
                    raise ValueError(f"unsafe path in archive: {member!r}")
            zf.extractall(root)
        manifest = json.loads((root / "MANIFEST.json").read_text(encoding="utf-8"))
        integrity = _integrity(root, manifest)
        out: Dict[str, Any] = {"integrity": integrity, "notes": notes}

        script = root / "code" / "script.py"
        records_path = root / "results" / "results.json"
        if integrity["status"] != "ok":
            out["status"] = "integrity_failed"
            return ReplicationVerification(out)
        if not script.exists() or not records_path.exists():
            out["status"] = "incomplete_pack"
            if not script.exists():
                notes.append("no code/script.py in the archive")
            if not records_path.exists():
                notes.append(
                    "no results/results.json: the pack recorded no results with "
                    "provenance to compare against"
                )
            return ReplicationVerification(out)

        capture = root / ".statspai_rerun_results.jsonl"
        env = dict(os.environ)
        env["STATSPAI_CAPTURE_RESULTS"] = str(capture)
        env.setdefault("MPLBACKEND", "Agg")
        if python is None:
            # Rerun with the StatsPAI the caller has loaded (the script runs
            # from the unpacked archive, where a relative PYTHONPATH or an
            # editable install elsewhere would pick up a different copy).
            import statspai as _sp

            pkg_parent = str(Path(_sp.__file__).resolve().parent.parent)
            env["PYTHONPATH"] = os.pathsep.join(
                [pkg_parent] + [p for p in [env.get("PYTHONPATH")] if p]
            )
        t0 = time.perf_counter()
        try:
            proc = subprocess.run(
                [python or sys.executable, str(script)],
                cwd=str(root),
                env=env,
                capture_output=True,
                text=True,
                timeout=timeout,
            )
            rc: Optional[int] = proc.returncode
            stderr_tail = (proc.stderr or "")[-2000:]
        except subprocess.TimeoutExpired:
            rc, stderr_tail = None, f"timed out after {timeout} s"
        out["rerun"] = {
            "returncode": rc,
            "seconds": round(time.perf_counter() - t0, 2),
            "stderr_tail": stderr_tail,
            "python": python or sys.executable,
        }
        if rc != 0:
            out["status"] = "rerun_failed"
            return ReplicationVerification(out)

        packed = json.loads(records_path.read_text(encoding="utf-8"))
        rerun: List[Dict[str, Any]] = []
        if capture.exists():
            for line in capture.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    rerun.append(json.loads(line))
        comparison = _compare(packed, rerun, rtol, atol)
        out["comparison"] = comparison
        if comparison["data_changed"]:
            notes.append(
                "same call on different input data (fingerprint changed): "
                + ", ".join(comparison["data_changed"])
            )
        ok = (
            comparison["n_matched"] == comparison["n_packed"]
            and not comparison["mismatched"]
            and not comparison["missing"]
        )
        out["status"] = "verified" if ok else "mismatch"
        return ReplicationVerification(out)
    finally:
        if tmp is not None:
            tmp.cleanup()
