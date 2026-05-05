"""Quick 3-way comparison helper for the StatsPAI Stata parity harness.

Run as `python3 _quick_compare.py [module]` from tests/stata_parity/. It
joins (py, R, Stata) JSONs by statistic name (with the (Intercept) /
Intercept normalisation), reports worst relative diff, and is a stand-
in until compare.py is extended for 3-way output.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path


HERE = Path(__file__).resolve().parent
PY_DIR = HERE.parent / "r_parity" / "results"
ST_DIR = HERE / "results"


def normalise(name: str) -> str:
    return name.replace("(Intercept)", "Intercept")


def load(d: Path, side: str, module: str):
    p = d / f"{module}_{side}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text(encoding="utf-8"))


def by_stat(payload):
    return {normalise(row["statistic"]): row for row in payload["rows"]}


def reldiff(a, b):
    if a is None or b is None:
        return None
    if abs(b) < 1e-12:
        return abs(a - b)
    return abs(a - b) / abs(b)


def compare(module: str) -> None:
    py = load(PY_DIR, "py", module)
    r = load(PY_DIR, "R", module)
    st = load(ST_DIR, "Stata", module)
    if not py:
        print(f"{module}: py json missing")
        return
    if not st:
        print(f"{module}: Stata json missing")
        return
    P = by_stat(py)
    R = by_stat(r) if r else {}
    S = by_stat(st)

    rows_out = []
    worst_pr = worst_ps = worst_rs = 0.0
    worst_pr_se = worst_ps_se = 0.0
    for k, p_row in P.items():
        s_row = S.get(k)
        if not s_row:
            continue
        r_row = R.get(k)
        pe, se_p = p_row.get("estimate"), p_row.get("se")
        sve = s_row.get("estimate")
        sse_se = s_row.get("se")
        re_e = r_row.get("estimate") if r_row else None
        rse_se = r_row.get("se") if r_row else None
        rel_pr = reldiff(pe, re_e)
        rel_ps = reldiff(pe, sve)
        rel_rs = reldiff(re_e, sve)
        rel_pr_se = reldiff(se_p, rse_se)
        rel_ps_se = reldiff(se_p, sse_se)
        if rel_pr is not None:
            worst_pr = max(worst_pr, rel_pr)
        if rel_ps is not None:
            worst_ps = max(worst_ps, rel_ps)
        if rel_rs is not None:
            worst_rs = max(worst_rs, rel_rs)
        if rel_pr_se is not None:
            worst_pr_se = max(worst_pr_se, rel_pr_se)
        if rel_ps_se is not None:
            worst_ps_se = max(worst_ps_se, rel_ps_se)
        rows_out.append((k, pe, re_e, sve, rel_pr, rel_ps, rel_rs))

    print(f"\n=== {module} ===")
    print(f"{'statistic':<32} {'py':>14} {'R':>14} {'Stata':>14}  rel(py-R)  rel(py-S)  rel(R-S)")
    for k, p, r_, s, rel_pr, rel_ps, rel_rs in rows_out:
        def fmt(x):
            return "—" if x is None else f"{x:>14.10f}"
        def rel(x):
            return "—" if x is None else f"{x:.2e}"
        print(f"{k:<32} {fmt(p)} {fmt(r_)} {fmt(s)}  {rel(rel_pr):>9}  {rel(rel_ps):>9}  {rel(rel_rs):>9}")
    print(f"  worst rel est: py-R={worst_pr:.2e}  py-Stata={worst_ps:.2e}  R-Stata={worst_rs:.2e}")
    print(f"  worst rel se:  py-R={worst_pr_se:.2e}  py-Stata={worst_ps_se:.2e}")


if __name__ == "__main__":
    targets = sys.argv[1:] or [p.name.replace("_Stata.json", "")
                                for p in sorted(ST_DIR.glob("*_Stata.json"))]
    for m in targets:
        compare(m)
