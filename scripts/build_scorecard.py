#!/usr/bin/env python3
"""Build ``docs/scorecard.md``: the evidence that StatsPAI is correct, calibrated and fast.

"Best tool" is not a measurable claim. Four things are, and a specialist
checks each of them before trusting a package with a paper:

1. **Correctness.** Does an estimator reproduce the reference
   implementation (R / Stata / the method authors' code) on the same data?
   Source: the parity index (``src/statspai/_parity_index.json``), built by
   ``scripts/build_parity_index.py`` from the Track A harness and the frozen
   reference fixtures.
2. **Calibration.** Do the reported confidence intervals cover the truth at
   their nominal rate? Source: Track B, ``tests/coverage_monte_carlo/
   results_b1000/coverage_b1000.json``.
3. **Speed.** Is it as fast as the reference on the same task and input?
   Source: Track C, ``tests/perf/results/*.json``.
4. **Honest gaps.** What has no evidence, listed by name.

Every figure on the page is read from a committed artifact; nothing is typed
by hand. The page describes the source tree it was generated from. Lines
that change the parity index may regenerate it (after
``build_parity_index.py``), but nothing requires them to between releases;
``release_gate.py --fix-census`` rewrites it and ``--check`` fails inside the
release window when it is stale, so every tagged release ships a current
page.

Usage::

    python scripts/build_scorecard.py           # write docs/scorecard.md
    python scripts/build_scorecard.py --check   # exit 1 if it is stale
"""

from __future__ import annotations

import argparse
import inspect
import json
import re
import statistics
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        _stream.reconfigure(encoding="utf-8")

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "docs" / "scorecard.md"
INDEX = REPO_ROOT / "src" / "statspai" / "_parity_index.json"
TRACK_A_RESULTS = REPO_ROOT / "tests" / "r_parity" / "results"
STATA_RESULTS = REPO_ROOT / "tests" / "stata_parity" / "results"
TRACK_B = REPO_ROOT / "tests" / "coverage_monte_carlo" / "results_b1000"
TRACK_C = REPO_ROOT / "tests" / "perf" / "results"

#: Track C modules: (file stem, label, reference package, reference side).
#: The reference side is the one ``tests/perf/compare_perf.py`` declares, so
#: the page shows the comparison the paper reports (DML against the Python
#: DoubleML, not the much slower R one).
PERF_MODULES: Tuple[Tuple[str, str, str, str], ...] = (
    ("01_hdfe", "HDFE, two absorbed FE", "fixest::feols", "R"),
    ("02_csdid", "CS-DiD ATT(g,t) + aggregation", "did::att_gt", "R"),
    ("03_scm", "Synthetic control (ADH)", "Synth::synth", "R"),
    ("04_dml", "DML PLR, linear learners, 5 folds", "DoubleML (Python)", "doubleml_py"),
)

#: Over-coverage is conservative, not a bug: the coverage test allows this
#: much above the Wilson upper bound (``test_coverage.py::_assert_calibrated``).
UPPER_SLACK = 0.02

#: Families with fewer estimators than this are left out of the gap table;
#: a 1-of-2 family says nothing about where effort should go.
MIN_FAMILY = 5


# ---------------------------------------------------------------- helpers


def _package_version() -> str:
    text = (REPO_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    match = re.search(r'^version\s*=\s*"([^"]+)"', text, re.MULTILINE)
    return match.group(1) if match else "unknown"


def _pct(num: int, den: int) -> str:
    return f"{100 * num / den:.1f}%" if den else "n/a"


def _sci(x: Optional[float]) -> str:
    return "n/a" if x is None else f"{x:.0e}" if x else "0"


def _wilson(p: float, n: int, conf: float = 0.99) -> Tuple[float, float]:
    """Wilson-score band around ``p`` for ``n`` draws.

    Same formula as ``tests/coverage_monte_carlo/test_coverage.py::
    _wilson_bounds`` (99%, so only a clearly mis-calibrated interval falls
    outside); for p = 0.95 and n = 1000 it is [0.929, 0.965].
    """
    from scipy.stats import norm

    z = norm.ppf((1 + conf) / 2)
    denom = 1 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = z * (p * (1 - p) / n + z**2 / (4 * n**2)) ** 0.5 / denom
    return centre - half, centre + half


# ------------------------------------------------------------ correctness


def _estimator_rows() -> List[Dict[str, Any]]:
    """One row per estimator: the stratum ``sp.parity_summary`` calls estimators."""
    import statspai as sp
    from statspai import registry as _registry
    from statspai._parity_taxonomy import NON_NUMERIC_CALLABLES
    from statspai.parity import INFRASTRUCTURE_CATEGORIES

    records = {
        r["function"]: r
        for r in json.loads(INDEX.read_text(encoding="utf-8"))["records"]
    }
    rows = []
    for name in sp.list_functions():
        spec = _registry._REGISTRY.get(name)
        obj = getattr(sp, name, None)
        if inspect.isclass(obj):
            continue
        if (
            spec is not None and spec.category in INFRASTRUCTURE_CATEGORIES
        ) or name in NON_NUMERIC_CALLABLES:
            continue
        module = (getattr(obj, "__module__", "") or "").split(".")
        rec = records.get(name, {})
        rows.append(
            {
                "name": name,
                "family": module[1] if len(module) > 1 else "core",
                "status": rec.get("status", "unverified"),
                "source": rec.get("source", "none"),
                "est_rel": rec.get("point_estimate_rel"),
                "se_rel": rec.get("se_rel"),
            }
        )
    return rows


def _correctness(rows: List[Dict[str, Any]]) -> List[str]:
    from statspai.parity import CROSS_LANGUAGE_STATUSES

    total = len(rows)
    by_status: Dict[str, int] = defaultdict(int)
    for row in rows:
        by_status[row["status"]] += 1
    cross = sum(by_status[s] for s in CROSS_LANGUAGE_STATUSES)
    unverified = by_status["unverified"]
    verified = total - unverified

    py_modules = sorted(
        p.stem[: -len("_py")] for p in TRACK_A_RESULTS.glob("*_py.json")
    )
    r_modules = {p.stem[: -len("_R")] for p in TRACK_A_RESULTS.glob("*_R.json")}
    stata_modules = (
        {p.stem[: -len("_Stata")] for p in STATA_RESULTS.glob("*_Stata.json")}
        if STATA_RESULTS.is_dir()
        else set()
    )
    three_way = sum(1 for m in py_modules if m in r_modules and m in stata_modules)

    track_a = [r for r in rows if r["source"] in ("track_a", "track_a_alias")]

    def _worst(gap: Any) -> Optional[float]:
        # {"R": rel, "Stata": rel}: the larger of the two sides that ran.
        values = [v for v in (gap or {}).values() if isinstance(v, (int, float))]
        return max(values) if values else None

    est = sorted(v for v in (_worst(r["est_rel"]) for r in track_a) if v is not None)
    se = sorted(v for v in (_worst(r["se_rel"]) for r in track_a) if v is not None)

    def _q(values: List[float], q: float) -> Optional[float]:
        if not values:
            return None
        return values[min(len(values) - 1, int(q * len(values)))]

    lines = [
        "## 1. Correctness against reference implementations",
        "",
        "The denominator is the estimators: result classes, plots, exporters,",
        "data loaders and other infrastructure are excluded, because they can",
        "never carry a numerical grade (`sp.parity_summary()['denominators']`).",
        "",
        "| Measure | Estimators | Share |",
        "| --- | ---: | ---: |",
        f"| Registered estimators | {total:,} | 100% |",
        f"| With numerical evidence of any kind | {verified:,} | {_pct(verified, total)} |",
        f"| Same data, matched to an R / Stata / author reference (T2) | {cross:,} | {_pct(cross, total)} |",
        f"| `bit-exact` (relative gap <= 1e-6 on estimate and SE) | {by_status['bit-exact']:,} | {_pct(by_status['bit-exact'], total)} |",
        f"| `aligned` (documented looser tolerance and mechanism) | {by_status['aligned']:,} | {_pct(by_status['aligned'], total)} |",
        f"| Known truth only (`analytical-only`) | {by_status['analytical-only']:,} | {_pct(by_status['analytical-only'], total)} |",
        f"| Published numbers reproduced (`external-replication`) | {by_status['external-replication']:,} | {_pct(by_status['external-replication'], total)} |",
        f"| No evidence (`unverified`, listed in section 4) | {unverified:,} | {_pct(unverified, total)} |",
        "",
        f"The Track A harness runs {len(py_modules)} modules on the same CSV bytes",
        f"in Python and R; {three_way} of them also run in Stata. Over the",
        f"{len(est)} estimators whose headline row comes from Track A, the",
        "relative gap to the reference (the larger of the R and Stata sides)",
        f"is as follows ({len(se)} of them report a standard error):",
        "",
        "| Quantity | Median | 90th percentile | Maximum |",
        "| --- | ---: | ---: | ---: |",
        f"| Point estimate | {_sci(_q(est, 0.5))} | {_sci(_q(est, 0.9))} | {_sci(est[-1] if est else None)} |",
        f"| Standard error | {_sci(_q(se, 0.5))} | {_sci(_q(se, 0.9))} | {_sci(se[-1] if se else None)} |",
        "",
        "Rows above 1e-6 each carry a registered mechanism",
        "(`tests/r_parity/compare.py::ROW_GAP_NOTES`); per-function detail is",
        "`sp.parity_status(name)` and the full matrix is [parity.md](parity.md).",
        "",
    ]
    return lines


# ------------------------------------------------------------- calibration


def _calibration() -> List[str]:
    path = TRACK_B / "coverage_b1000.json"
    rows = json.loads(path.read_text(encoding="utf-8"))
    body = []
    inside = 0
    for row in rows:
        lo, hi = _wilson(0.95, int(row["B"]))
        ok = lo <= row["rate"] <= hi + UPPER_SLACK
        inside += ok
        body.append(
            f"| `{row['name']}` | {row['B']:,} | {row['rate']:.3f} | "
            f"{'yes' if ok else '**no**'} | {row['se_sd_ratio']:.2f} |"
        )
    lo, hi = _wilson(0.95, int(rows[0]["B"]))
    return [
        "## 2. Calibration of confidence intervals",
        "",
        "Each row fits the estimator to B independent draws of a DGP whose",
        "true effect is known and counts how often the reported 95% interval",
        "covers it. A row passes when its coverage is at least the lower end of",
        f"the 99% Wilson band around 0.95 ({lo:.3f} for B = {rows[0]['B']:,})",
        f"and at most {UPPER_SLACK} above its upper end ({hi:.3f}); slight",
        "over-coverage is conservative, under-coverage is the failure that",
        "matters. This is the rule `test_coverage.py` applies.",
        "SE/SD near 1 means the reported standard error matches the sampling",
        "spread of the estimate.",
        "",
        f"**{inside} of {len(rows)} rows inside the band.** The rows outside it",
        "are explained in",
        "[`tests/coverage_monte_carlo/FINDINGS.md`](https://github.com/brycewang-stanford/StatsPAI/blob/main/tests/coverage_monte_carlo/FINDINGS.md);",
        "they are reported, not removed.",
        "",
        "| Estimator and DGP | B | Coverage | In band | SE/SD |",
        "| --- | ---: | ---: | :---: | ---: |",
        *body,
        "",
    ]


# ------------------------------------------------------------------- speed


def _largest(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    rows = [r for r in data.get("rows", []) if r.get("median_time_s")]
    if not rows:
        return None
    top = max(rows, key=lambda r: r["n"])
    return {
        "n": top["n"],
        "t": top["median_time_s"],
        "version": data.get("hardware", {}).get("statspai_version"),
        "platform": data.get("hardware", {}).get("machine"),
    }


def _speed() -> List[str]:
    body = []
    versions = set()
    for stem, label, ref, side in PERF_MODULES:
        py = _largest(TRACK_C / f"{stem}_py.json")
        r = _largest(TRACK_C / f"{stem}_{side}.json")
        if py is None or r is None or py["n"] != r["n"]:
            continue
        versions.add(py["version"])
        ratio = r["t"] / py["t"]
        verdict = (
            f"StatsPAI {ratio:.1f}x faster"
            if ratio >= 1.1
            else f"reference {1 / ratio:.1f}x faster" if ratio <= 1 / 1.1 else "tie"
        )
        body.append(
            f"| {label} | `{ref}` | {py['n']:,} | {py['t']:.4f} | {r['t']:.4f} | {verdict} |"
        )
    measured = ", ".join(sorted(v for v in versions if v)) or "unknown"
    return [
        "## 3. Speed against the reference on the same task",
        "",
        "Both sides read the same input file (SHA-256 checked), run on one",
        "thread, and are timed in the same session; the table shows the",
        f"largest size measured. Timings were taken on StatsPAI {measured};",
        "they are re-measured when a paper is re-anchored, not on every",
        "release. Full tables, including the agreement of the two answers:",
        "[`tests/perf/results/perf_table.md`](https://github.com/brycewang-stanford/StatsPAI/blob/main/tests/perf/results/perf_table.md).",
        "",
        "| Task | Reference | n | StatsPAI (s) | Reference (s) | Result |",
        "| --- | --- | ---: | ---: | ---: | --- |",
        *body,
        "",
    ]


# -------------------------------------------------------------------- gaps


def _gaps(rows: List[Dict[str, Any]]) -> List[str]:
    from statspai.parity import CROSS_LANGUAGE_STATUSES

    unverified = sorted(r["name"] for r in rows if r["status"] == "unverified")
    families: Dict[str, List[int]] = defaultdict(lambda: [0, 0])
    for row in rows:
        families[row["family"]][0] += 1
        families[row["family"]][1] += row["status"] in CROSS_LANGUAGE_STATUSES
    weakest = sorted(
        ((f, t, c) for f, (t, c) in families.items() if t >= MIN_FAMILY),
        key=lambda x: (x[2] / x[1], -x[1], x[0]),
    )[:12]
    names = ", ".join(f"`{n}`" for n in unverified) if unverified else "none"
    lines = [
        "## 4. Known gaps",
        "",
        f"**Estimators with no numerical evidence ({len(unverified)}):** {names}.",
        "",
        "**Families where the fewest estimators are matched to a reference.**",
        "Many of these are frontier methods (conformal, neural, LLM-assisted)",
        "with no established reference implementation, where known-truth",
        "recovery and Monte Carlo coverage are the honest ceiling.",
        "",
        "| Family | Estimators | Matched to a reference | Share |",
        "| --- | ---: | ---: | ---: |",
    ]
    lines += [f"| `{f}` | {t} | {c} | {_pct(c, t)} |" for f, t, c in weakest]
    lines += [
        "",
        "**Agent benchmark.** CausalAgentBench v2 (50 tasks, StatsPAI vs the",
        "Python stack vs R, two model providers) is pre-registered in",
        "`tests/agent_bench/`; its production run waits on the OSF deposit, so",
        "no agent-performance claim is made here yet.",
        "",
    ]
    return lines


# -------------------------------------------------------------------- page


def render() -> str:
    rows = _estimator_rows()
    lines = [
        "# Evidence scorecard",
        "",
        "<!-- AUTO-GENERATED by scripts/build_scorecard.py; do not edit. -->",
        "",
        "Generated by `python scripts/build_scorecard.py` from the committed",
        f"artifacts of the source tree at package version {_package_version()}.",
        "On `main` that tree can be ahead of the release with the same number.",
        "The page is regenerated before every release; for live numbers call",
        "`sp.parity_summary()`.",
        "",
        "No single number shows that one statistics package is better than",
        "another. What a careful user can check is whether each estimator",
        "reproduces the reference implementation, whether its intervals cover",
        "at the nominal rate, how fast it is on the same task, and what has not",
        "been checked. This page reports all four, including the failures.",
        "",
    ]
    lines += _correctness(rows)
    lines += _calibration()
    lines += _speed()
    lines += _gaps(rows)
    lines += [
        "## Reproduce",
        "",
        "```bash",
        "python tests/r_parity/compare.py                  # Track A table (R / Stata goldens committed)",
        "python tests/r_parity/verify_reproduce.py         # re-derive the R goldens (needs R)",
        "python tests/stata_parity/verify_reproduce_stata.py   # re-derive the Stata goldens (needs Stata)",
        "python tests/coverage_monte_carlo/run_b1000.py    # Track B, B = 1000",
        "bash tests/perf/run_when_idle.sh                  # Track C timings (idle machine)",
        "python scripts/build_scorecard.py                 # this page",
        "```",
        "",
    ]
    return "\n".join(lines)


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    parser.add_argument("--check", action="store_true", help="exit 1 if stale")
    args = parser.parse_args(argv)
    text = render()
    current = OUT.read_text(encoding="utf-8") if OUT.exists() else ""
    if args.check:
        if current != text:
            print(
                f"{OUT.relative_to(REPO_ROOT)} is stale: run scripts/build_scorecard.py"
            )
            return 1
        print(f"{OUT.relative_to(REPO_ROOT)} is current")
        return 0
    OUT.write_text(text, encoding="utf-8")
    print(f"wrote {OUT.relative_to(REPO_ROOT)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
