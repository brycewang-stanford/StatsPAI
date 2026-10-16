"""Pre-registration lock for CausalAgentBench.

Before the first production trial, the frozen design is fingerprinted into
``prompts/preregistration.lock.json``: the SHA-256 of every file that
defines what is asked, what counts as correct and how it is scored, plus
the OSF registration DOI, the exact model releases and the settings that
are not otherwise visible (effort, turn limit, reps).

``models.real_model`` calls :func:`verify` and refuses to build an API
client unless the lock exists, names a DOI, and every fingerprinted file
is byte-identical to what was locked. The mock / oracle paths never need
the lock.

Usage::

    # after the OSF deposit (Bryce):
    python runners/prereg.py lock --doi 10.17605/OSF.IO/XXXXX \\
        --anthropic-model <exact id> --openai-model <exact id> \\
        --anthropic-effort high
    python runners/prereg.py verify
    python runners/prereg.py fingerprint      # print hashes for the deposit form
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
LOCK = ROOT / "prompts" / "preregistration.lock.json"

#: Files whose bytes define the experiment.
FINGERPRINTED = (
    "prompts/_protocol.md",
    "prompts/prompts.json",
    "golds/golds.json",
    "golds/reference.R",
    "golds/reference_values.json",
    "data/MANIFEST.json",
    "runners/agent_loop.py",
    "runners/cells.py",
    "runners/sandbox.py",
    "runners/grader.py",
    "runners/models.py",
    "runners/runner.py",
)

#: Settings recorded in the lock (defaults are the pre-registered values).
DEFAULT_SETTINGS = {
    "reps": 3,
    "max_turns": 20,
    "call_timeout_seconds": 300,
    "bootstrap_B": 10000,
    "anthropic_effort": "high",
}


def fingerprint() -> Dict[str, str]:
    out = {}
    for rel in FINGERPRINTED:
        path = ROOT / rel
        out[rel] = (
            hashlib.sha256(path.read_bytes()).hexdigest()
            if path.exists()
            else "MISSING"
        )
    return out


def lock(doi: str, models: Dict[str, str], settings: Dict[str, Any]) -> Dict[str, Any]:
    if not doi:
        raise SystemExit("lock needs the OSF registration DOI (--doi)")
    if not all(models.values()):
        raise SystemExit("lock needs both exact model IDs")
    data = {
        "osf_doi": doi,
        "locked_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "models": models,
        "settings": {**DEFAULT_SETTINGS, **settings},
        "files": fingerprint(),
    }
    if "MISSING" in data["files"].values():
        raise SystemExit("a fingerprinted file is missing; build data / golds first")
    LOCK.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    return data


def verify() -> Dict[str, Any]:
    """The lock, if the design is still the one that was registered."""
    if not LOCK.exists():
        raise RuntimeError(
            "No pre-registration lock: deposit prompts/_protocol.md and the "
            "file hashes on OSF, then run `python runners/prereg.py lock`."
        )
    data = json.loads(LOCK.read_text(encoding="utf-8"))
    if not data.get("osf_doi"):
        raise RuntimeError("the lock names no OSF DOI")
    now = fingerprint()
    changed = [k for k, v in data["files"].items() if now.get(k) != v]
    if changed:
        raise RuntimeError(f"files changed since the lock: {changed}")
    return data


def main() -> int:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    lk = sub.add_parser("lock")
    lk.add_argument("--doi", required=True)
    lk.add_argument("--anthropic-model", required=True)
    lk.add_argument("--openai-model", required=True)
    lk.add_argument("--anthropic-effort", default=DEFAULT_SETTINGS["anthropic_effort"])
    sub.add_parser("verify")
    sub.add_parser("fingerprint")
    a = ap.parse_args()
    if a.cmd == "fingerprint":
        print(json.dumps(fingerprint(), indent=2))
        return 0
    if a.cmd == "verify":
        try:
            verify()
        except RuntimeError as exc:
            print(f"NOT VERIFIED: {exc}")
            return 1
        print("verified")
        return 0
    data = lock(
        a.doi,
        {"anthropic": a.anthropic_model, "openai": a.openai_model},
        {"anthropic_effort": a.anthropic_effort},
    )
    print(json.dumps(data, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
