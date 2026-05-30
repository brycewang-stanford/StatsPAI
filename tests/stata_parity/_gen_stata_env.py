"""Assemble tests/stata_parity/STATA_ENVIRONMENT.md from the raw capture
files written by _capture_stata_env.do.

The Stata analogue of tests/r_parity/R_ENVIRONMENT.md + renv.lock. Stata has
no per-result packageVersion() primitive, so the per-_Stata.json provenance
block records only the engine (version/edition/OS); the community ado package
versions live here, captured verbatim from each command's `*!` banner line.
Lines with no banner are recorded as "(no *! version line)" and uninstalled
commands as "NOT INSTALLED" -- never a guessed version (CLAUDE.md §10).

    stata-mp -q -b do _capture_stata_env.do   # writes the two raw files
    python _gen_stata_env.py                   # -> STATA_ENVIRONMENT.md
"""
from __future__ import annotations

from pathlib import Path

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"
CORE = RESULTS / "_stata_core.txt"
ADO = RESULTS / "_stata_ado_versions.tsv"
OUT = HERE / "STATA_ENVIRONMENT.md"


def _read_core() -> dict:
    core = {}
    if CORE.exists():
        for line in CORE.read_text(encoding="utf-8").splitlines():
            if "=" in line:
                k, v = line.split("=", 1)
                core[k.strip()] = v.strip()
    return core


def _read_ado() -> list[tuple[str, str, str]]:
    rows = []
    if ADO.exists():
        lines = ADO.read_text(encoding="utf-8").splitlines()
        for line in lines[1:]:  # skip header
            parts = line.split("\t")
            if len(parts) >= 3:
                rows.append((parts[0], parts[1], parts[2]))
            elif len(parts) == 2:
                rows.append((parts[0], parts[1], ""))
    return rows


def main() -> None:
    core = _read_core()
    ado = _read_ado()

    lines = [
        "# Stata reference environment for the Track A parity harness",
        "",
        "> JSS reproducibility manifest (paper Section 8). Pins the exact "
        "Stata engine and community `ado` package versions under which the "
        "committed `tests/stata_parity/results/*_Stata.json` golden values "
        "were produced and **verified to reproduce bit-for-bit** by "
        "`tests/stata_parity/verify_reproduce_stata.py` "
        "(`REPRODUCIBILITY_REPORT_STATA.md`). Every `_Stata.json` also carries "
        "an inline `provenance` block with the engine fields below.",
        "",
        "Unlike R (`packageVersion()`), Stata exposes no per-command version "
        "primitive, so the `ado` versions here are the **verbatim first "
        "`*!` banner line** of each command's `.ado` file. Commands whose "
        "banner carries no version string are recorded as "
        "`(no *! version line)` and uninstalled commands as `NOT INSTALLED` "
        "— never a guessed version.",
        "",
        "## How to regenerate and verify",
        "",
        "```bash",
        "# Re-capture the engine + ado inventory:",
        "stata-mp -q -b do tests/stata_parity/_capture_stata_env.do",
        "python tests/stata_parity/_gen_stata_env.py",
        "",
        "# Re-run every Stata reference and diff each statistic against the",
        "# committed golden JSON at a 1e-9 reproducibility tolerance:",
        "python tests/stata_parity/verify_reproduce_stata.py",
        "```",
        "",
        "## Stata engine",
        "",
        "| Field | Value |",
        "|---|---|",
        f"| Version | Stata {core.get('stata_version', '?')} |",
        f"| Edition | {core.get('edition', '?')} |",
        f"| OS | {core.get('os', '?')} |",
        f"| Machine | {core.get('machine_type', '?')} |",
        f"| Executable date | {core.get('born_date', '?')} |",
        "",
        "## Community `ado` packages (verbatim `*!` banner)",
        "",
        "| Command | `*!` banner line | Path |",
        "|---|---|---|",
    ]
    for cmd, banner, path in ado:
        # Trim the absolute home prefix for portability of the printed path.
        short = path.replace(str(Path.home()), "~") if path else "—"
        banner_md = banner.replace("|", "\\|") if banner else "—"
        lines.append(f"| `{cmd}` | {banner_md} | `{short}` |")
    lines += [
        "",
        "> Built-in Stata commands used by the harness (`regress`, "
        "`ivregress`, `xtreg`, `mixed`, `melogit`, `frontier`, `xtfrontier`, "
        "`stcox`, `tobit`, `mlogit`, `ologit`, `oprobit`, `probit`, "
        "`clogit`, `nbreg`, `qreg`, `var`, `newey`, `arima`, `xtabond`, "
        "`teffects psmatch`) ship with Stata "
        f"{core.get('stata_version', '?')} {core.get('edition', '?')} and "
        "are versioned by the engine row above.",
        "",
        "---",
        "",
        "*Captured via `tests/stata_parity/_capture_stata_env.do`. Refresh "
        "whenever the Stata environment changes; the per-`_Stata.json` "
        "`provenance` block is the authoritative per-result engine record.*",
    ]
    OUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {OUT} ({len(ado)} ado packages, Stata "
          f"{core.get('stata_version','?')} {core.get('edition','?')})")


if __name__ == "__main__":
    main()
