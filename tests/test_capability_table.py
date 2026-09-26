"""``docs/capabilities.md`` is generated; it must match the code (review §7.3)."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_capability_table_is_current():
    proc = subprocess.run(
        [
            sys.executable,
            str(ROOT / "scripts" / "build_capability_table.py"),
            "--check",
        ],
        capture_output=True,
        text=True,
    )
    assert proc.returncode == 0, proc.stderr
