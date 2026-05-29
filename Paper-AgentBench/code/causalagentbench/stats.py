"""Analysis: summary tables, cluster bootstrap, and the H1..H5 tests.

The pre-registered inference is a cluster bootstrap with the *prompt* as
the cluster (B = 9,999), two-sided alpha = 0.05, Bonferroni-corrected
across the five primary hypotheses (alpha_adj = 0.01). Effect sizes are
difference-in-shares with bootstrap-percentile CIs.

Only numpy is required. Hypotheses that need a condition absent from the
supplied results are reported as "not evaluable" rather than erroring.
"""

from __future__ import annotations

from collections import defaultdict
from statistics import median
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from .conditions import CONDITIONS
from .schema import TrialResult

# Condition groupings used by the hypotheses.
STATSPAI_CELLS = ["C1", "C2"]
PYTHONIC_CELLS = ["C3", "C4"]
R_CELLS = ["C5", "C6"]


# ---------------------------------------------------------------------------
# Summary table
# ---------------------------------------------------------------------------

def _safe_mean(xs: List[Optional[float]]) -> Optional[float]:
    xs = [x for x in xs if x is not None]
    return float(np.mean(xs)) if xs else None


def summary_table(results: List[TrialResult]) -> Dict[str, Dict]:
    """Per-(condition, difficulty) aggregates of the eight metrics + M7."""
    cells: Dict[Tuple[str, str], List[TrialResult]] = defaultdict(list)
    for r in results:
        cells[(r.condition, r.difficulty)].append(r)

    table: Dict[str, Dict] = {}
    for (cond, diff), rs in sorted(cells.items()):
        m1 = _safe_mean([1.0 if r.m1_success else 0.0 for r in rs if r.m1_success is not None])
        m2 = _safe_mean([1.0 if r.m2_method_correct else 0.0 for r in rs if r.m2_method_correct is not None])
        m3 = _safe_mean([1.0 if r.m3_exec_ok else 0.0 for r in rs])
        m5 = _safe_mean([1.0 if r.m5_hallucinated else 0.0 for r in rs if r.m5_hallucinated is not None])
        m6 = _safe_mean([r.m6_diag_completeness for r in rs])
        m4 = median([r.m4_tokens for r in rs]) if rs else None
        m8 = median([r.m8_wall_s for r in rs if r.m8_wall_s is not None]) if rs else None
        table[f"{cond}/{diff}"] = dict(
            n=len(rs), m1_success=m1, m2_method=m2, m3_exec=m3,
            m5_halluc=m5, m6_diag=m6, m4_tokens_med=m4, m8_wall_med=m8,
            m7_repro_var=_m7_reproducibility(rs),
        )
    return table


def _m7_reproducibility(rs: List[TrialResult]) -> Optional[float]:
    """Mean across-seed variance of the estimate within each prompt."""
    by_task: Dict[str, List[float]] = defaultdict(list)
    for r in rs:
        if r.m7_estimate is not None:
            by_task[r.task_id].append(r.m7_estimate)
    variances = [float(np.var(v, ddof=0)) for v in by_task.values() if len(v) >= 2]
    return float(np.mean(variances)) if variances else None


def summarize(results: List[TrialResult]) -> str:
    """Render the summary table as text."""
    table = summary_table(results)
    if not table:
        return "(no results)"
    hdr = (f"{'cell':<14}{'n':>4}{'M1 succ':>9}{'M2 meth':>9}"
           f"{'M3 exec':>9}{'M5 hal':>8}{'M6 diag':>9}{'M4 tok':>9}{'M8 s':>8}")
    lines = [hdr, "-" * len(hdr)]
    for cell, v in table.items():
        def f(x, p="{:.2f}"):
            return "  n/a" if x is None else p.format(x)
        lines.append(
            f"{cell:<14}{v['n']:>4}{f(v['m1_success']):>9}{f(v['m2_method']):>9}"
            f"{f(v['m3_exec']):>9}{f(v['m5_halluc']):>8}{f(v['m6_diag']):>9}"
            f"{f(v['m4_tokens_med'],'{:.0f}'):>9}{f(v['m8_wall_med'],'{:.2f}'):>8}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Cluster bootstrap (prompt as cluster)
# ---------------------------------------------------------------------------

def _share(rs: List[TrialResult], pred: Callable[[TrialResult], Optional[bool]]) -> Optional[float]:
    vals = [pred(r) for r in rs]
    vals = [1.0 if v else 0.0 for v in vals if v is not None]
    return float(np.mean(vals)) if vals else None


def cluster_bootstrap_diff(
    group_a: List[TrialResult],
    group_b: List[TrialResult],
    pred: Callable[[TrialResult], Optional[bool]],
    B: int = 9999,
    alpha: float = 0.05,
    seed: int = 0,
) -> Dict[str, Optional[float]]:
    """Bootstrap CI for share(A) - share(B), resampling prompts (clusters).

    Returns ``{diff, lo, hi, p_two_sided}``. ``p_two_sided`` is the
    bootstrap p-value for H0: diff = 0.
    """
    a_by_task: Dict[str, List[TrialResult]] = defaultdict(list)
    b_by_task: Dict[str, List[TrialResult]] = defaultdict(list)
    for r in group_a:
        a_by_task[r.task_id].append(r)
    for r in group_b:
        b_by_task[r.task_id].append(r)
    tasks = sorted(set(a_by_task) | set(b_by_task))
    if not tasks:
        return dict(diff=None, lo=None, hi=None, p_two_sided=None)

    point = _diff(group_a, group_b, pred)
    rng = np.random.default_rng(seed)
    draws = []
    idx = np.arange(len(tasks))
    for _ in range(B):
        pick = rng.choice(idx, size=len(tasks), replace=True)
        ra, rb = [], []
        for j in pick:
            t = tasks[j]
            ra.extend(a_by_task.get(t, []))
            rb.extend(b_by_task.get(t, []))
        d = _diff(ra, rb, pred)
        if d is not None:
            draws.append(d)
    if not draws:
        return dict(diff=point, lo=None, hi=None, p_two_sided=None)
    draws = np.asarray(draws)
    lo = float(np.quantile(draws, alpha / 2))
    hi = float(np.quantile(draws, 1 - alpha / 2))
    # two-sided bootstrap p: 2 * min(share below 0, share above 0)
    p = 2 * min((draws <= 0).mean(), (draws >= 0).mean())
    return dict(diff=point, lo=lo, hi=hi, p_two_sided=float(min(p, 1.0)))


def _diff(a, b, pred) -> Optional[float]:
    sa, sb = _share(a, pred), _share(b, pred)
    if sa is None or sb is None:
        return None
    return sa - sb


# ---------------------------------------------------------------------------
# Pre-registered hypotheses H1..H5
# ---------------------------------------------------------------------------

def _cells(results: List[TrialResult], codes: List[str], difficulties=None) -> List[TrialResult]:
    out = [r for r in results if r.condition in codes]
    if difficulties:
        out = [r for r in out if r.difficulty in difficulties]
    return out


def _present(results: List[TrialResult], codes: List[str]) -> bool:
    have = {r.condition for r in results}
    return all(c in have for c in codes)


def test_hypotheses(results: List[TrialResult], B: int = 9999, seed: int = 0) -> Dict[str, Dict]:
    """Evaluate H1..H5 with prompt-cluster bootstrap + Bonferroni (5 primary).

    Hypotheses needing an absent condition are returned with
    ``evaluable=False`` rather than raising.
    """
    alpha_adj = 0.05 / 5  # Bonferroni across 5 primary hypotheses
    succ = lambda r: r.m1_success
    halluc = lambda r: r.m5_hallucinated
    out: Dict[str, Dict] = {}

    # H1: StatsPAI success >= 90% L1, 70% L2, 50% L3
    if _present(results, STATSPAI_CELLS):
        sp_cells = _cells(results, STATSPAI_CELLS)
        h1 = {}
        for diff, thresh in [("L1", 0.90), ("L2", 0.70), ("L3", 0.50)]:
            s = _share(_cells(sp_cells, STATSPAI_CELLS, [diff]), succ)
            h1[diff] = dict(success=s, threshold=thresh,
                            pass_=(s is not None and s >= thresh))
        out["H1"] = dict(evaluable=True, detail=h1,
                         pass_=all(v["pass_"] for v in h1.values()))
    else:
        out["H1"] = dict(evaluable=False, reason="needs C1,C2")

    # H2: StatsPAI exceeds Pythonic on L2-L3 success by >= 15pp
    if _present(results, STATSPAI_CELLS + PYTHONIC_CELLS):
        a = _cells(results, STATSPAI_CELLS, ["L2", "L3"])
        b = _cells(results, PYTHONIC_CELLS, ["L2", "L3"])
        bs = cluster_bootstrap_diff(a, b, succ, B=B, alpha=alpha_adj, seed=seed)
        out["H2"] = dict(evaluable=True, **bs,
                         pass_=(bs["lo"] is not None and bs["lo"] >= 0.15))
    else:
        out["H2"] = dict(evaluable=False, reason="needs C1-C4")

    # H3: StatsPAI hallucination < 5% (all trials); Pythonic > 15%
    if _present(results, STATSPAI_CELLS):
        sp_h = _share(_cells(results, STATSPAI_CELLS), halluc)
        py_h = _share(_cells(results, PYTHONIC_CELLS), halluc) if _present(results, PYTHONIC_CELLS) else None
        out["H3"] = dict(evaluable=True, statspai_halluc=sp_h, pythonic_halluc=py_h,
                         pass_statspai=(sp_h is not None and sp_h < 0.05),
                         pass_pythonic=(py_h is not None and py_h > 0.15) if py_h is not None else None)
    else:
        out["H3"] = dict(evaluable=False, reason="needs C1,C2")

    # H4: StatsPAI uses <= 60% of Pythonic tokens (same prompt+model)
    if _present(results, STATSPAI_CELLS + PYTHONIC_CELLS):
        sp_tok = median([r.m4_tokens for r in _cells(results, STATSPAI_CELLS)])
        py_tok = median([r.m4_tokens for r in _cells(results, PYTHONIC_CELLS)])
        ratio = (sp_tok / py_tok) if py_tok else None
        out["H4"] = dict(evaluable=True, statspai_tok=sp_tok, pythonic_tok=py_tok,
                         ratio=ratio, pass_=(ratio is not None and ratio <= 0.60))
    else:
        out["H4"] = dict(evaluable=False, reason="needs C1-C4")

    # H5: R-via-MCP success ~ StatsPAI but >= 1.5x tokens
    if _present(results, STATSPAI_CELLS + R_CELLS):
        a = _cells(results, R_CELLS)
        b = _cells(results, STATSPAI_CELLS)
        bs = cluster_bootstrap_diff(a, b, succ, B=B, alpha=alpha_adj, seed=seed)
        indistinguishable = (bs["lo"] is not None and bs["lo"] <= 0 <= bs["hi"])
        r_tok = median([r.m4_tokens for r in a])
        sp_tok = median([r.m4_tokens for r in b])
        tok_ratio = (r_tok / sp_tok) if sp_tok else None
        out["H5"] = dict(evaluable=True, success_diff=bs["diff"],
                         success_ci=(bs["lo"], bs["hi"]),
                         indistinguishable=indistinguishable,
                         token_ratio=tok_ratio,
                         pass_=(indistinguishable and tok_ratio is not None and tok_ratio >= 1.5))
    else:
        out["H5"] = dict(evaluable=False, reason="needs C1,C2,C5,C6")

    out["_meta"] = dict(alpha_adjusted=alpha_adj, B=B,
                        conditions_present=sorted({r.condition for r in results}))
    return out
