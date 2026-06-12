"""Benchmark regression ratchet.

Compares two ``benchmarks/run_all.py`` result files (a *baseline* and a
*current* run) and fails when any StatsPAI timing regresses beyond a
threshold. Companion to ``scripts/coverage_campaign.py`` — same spirit:
a ratchet that only ever tightens.

Two intended modes:

* **Local developer check** — compare a fresh run against the committed
  same-machine baseline::

      python benchmarks/run_all.py --quick
      python scripts/benchmark_ratchet.py --check

* **CI A/B check** (``.github/workflows/benchmarks.yml``) — the released
  PyPI wheel and the source tree are benchmarked back-to-back on the
  *same runner*, so machine speed cancels out::

      python scripts/benchmark_ratchet.py --check \
          --baseline /tmp/results_release.json \
          --current  benchmarks/results.json --threshold 1.5

Only timing fields that belong to StatsPAI itself (``sp_*_s``) are
ratcheted; reference-library timings (statsmodels / linearmodels) are
informational. Rows whose baseline timing is below ``--min-seconds``
are skipped as timer noise.

Usage::

    python scripts/benchmark_ratchet.py --check [--threshold 1.5]
    python scripts/benchmark_ratchet.py --update-baseline
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from pathlib import Path
from typing import Dict, List, Tuple

REPO = Path(__file__).resolve().parent.parent
DEFAULT_BASELINE = REPO / "benchmarks" / "baseline.json"
DEFAULT_CURRENT = REPO / "benchmarks" / "results.json"

# Timer noise floor: sub-millisecond rows flap with CPU frequency
# scaling and are meaningless to ratchet.
DEFAULT_MIN_SECONDS = 1e-3


def _row_key(row: Dict) -> Tuple:
    """Identity of a benchmark row = its integer size fields.

    Strings are deliberately excluded: fields like ``speedup_vs_sm`` are
    derived from the timings and differ between the two runs being
    compared, which would break row matching.
    """
    return tuple(
        sorted(
            (k, v)
            for k, v in row.items()
            if not k.endswith("_s")
            and isinstance(v, int)
            and not isinstance(v, bool)
        )
    )


def _sp_timings(row: Dict) -> Dict[str, float]:
    return {
        k: float(v)
        for k, v in row.items()
        if k.startswith("sp_") and k.endswith("_s") and isinstance(v, (int, float))
    }


def compare(
    baseline: Dict,
    current: Dict,
    threshold: float,
    min_seconds: float,
) -> Tuple[List[str], List[str]]:
    """Return (failures, report_lines)."""
    failures: List[str] = []
    lines: List[str] = []
    sections = [
        k for k, v in baseline.items() if isinstance(v, dict) and "rows" in v
    ]
    for section in sections:
        cur_section = current.get(section)
        if not isinstance(cur_section, dict) or "rows" not in cur_section:
            lines.append(f"~ {section}: missing from current run (skipped)")
            continue
        cur_rows = {_row_key(r): r for r in cur_section["rows"]}
        for base_row in baseline[section]["rows"]:
            key = _row_key(base_row)
            cur_row = cur_rows.get(key)
            if cur_row is None:
                lines.append(f"~ {section} {dict(key)}: row not in current run")
                continue
            for field, base_t in _sp_timings(base_row).items():
                cur_t = cur_row.get(field)
                if cur_t is None:
                    continue
                if base_t < min_seconds:
                    lines.append(
                        f"~ {section}.{field} {dict(key)}: baseline "
                        f"{base_t * 1e3:.3f} ms < noise floor (skipped)"
                    )
                    continue
                ratio = float(cur_t) / base_t
                tag = f"{section}.{field} {dict(key)}"
                msg = (
                    f"{tag}: {base_t * 1e3:.1f} ms -> {float(cur_t) * 1e3:.1f} ms "
                    f"({ratio:.2f}x)"
                )
                if ratio > threshold:
                    failures.append(msg)
                    lines.append(f"✗ {msg}")
                else:
                    lines.append(f"✓ {msg}")
    return failures, lines


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--check", action="store_true", help="compare current vs baseline")
    ap.add_argument(
        "--update-baseline",
        action="store_true",
        help="copy the current results file over the committed baseline",
    )
    ap.add_argument("--baseline", type=Path, default=DEFAULT_BASELINE)
    ap.add_argument("--current", type=Path, default=DEFAULT_CURRENT)
    ap.add_argument(
        "--threshold",
        type=float,
        default=1.5,
        help="fail when current/baseline exceeds this ratio (default 1.5)",
    )
    ap.add_argument("--min-seconds", type=float, default=DEFAULT_MIN_SECONDS)
    args = ap.parse_args()

    if args.update_baseline:
        if not args.current.exists():
            print(f"no current results at {args.current}; run benchmarks first")
            return 2
        shutil.copyfile(args.current, args.baseline)
        print(f"baseline updated from {args.current} -> {args.baseline}")
        return 0

    if not args.check:
        print("nothing to do (pass --check or --update-baseline)")
        return 2

    if not args.baseline.exists():
        print(f"no baseline at {args.baseline}; create one with --update-baseline")
        return 2
    if not args.current.exists():
        print(f"no current results at {args.current}; run benchmarks first")
        return 2

    baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
    current = json.loads(args.current.read_text(encoding="utf-8"))

    base_meta = baseline.get("meta", {})
    cur_meta = current.get("meta", {})
    print(
        f"baseline: statspai {base_meta.get('statspai_version')} "
        f"({base_meta.get('mode')}, {base_meta.get('platform')})"
    )
    print(
        f"current : statspai {cur_meta.get('statspai_version')} "
        f"({cur_meta.get('mode')}, {cur_meta.get('platform')})"
    )
    if base_meta.get("platform") != cur_meta.get("platform"):
        print(
            "warning: baseline and current were produced on different "
            "platforms; ratios are only meaningful same-machine"
        )
    print()

    failures, lines = compare(
        baseline, current, threshold=args.threshold, min_seconds=args.min_seconds
    )
    print("\n".join(lines))
    print()
    if failures:
        print(
            f"FAIL: {len(failures)} timing(s) regressed beyond "
            f"{args.threshold:.2f}x:"
        )
        for f in failures:
            print(f"  - {f}")
        return 1
    print(f"OK: no StatsPAI timing regressed beyond {args.threshold:.2f}x")
    return 0


if __name__ == "__main__":
    sys.exit(main())
