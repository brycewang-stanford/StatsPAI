#!/usr/bin/env python3
"""
StatsPAI — Recommendation Hit-Rate Benchmark harness.

Runs ``sp.recommend`` (and scores ``sp.audit`` catalog coverage) against the
ground-truth corpus in ``corpus.yaml`` and emits a scorecard. This is the
agent-native quality gate the moat depends on: a plausible-but-wrong
recommendation is worse than none, so we measure — across real published
designs — whether the top-k recommendation matches the estimator the paper
actually used, and whether the audit knows the robustness checks a referee
would demand.

Two metrics
-----------
1. **recommend hit-rate** (dynamic): run ``sp.recommend`` on the real data and
   check whether an econometrically-acceptable estimator appears in the
   ranked output. A *hard miss* is leading (top-1) with a disqualifying
   estimator (e.g. TWFE on a staggered design).
2. **audit catalog coverage** (static): does the ``sp.audit`` check catalog,
   for the design's method family, contain the ground-truth robustness checks?
   (Dynamic fit-based audit recall — does a fitted result actually flag them
   MISSING — is a later phase; see README roadmap.)

Usage
-----
    python3 benchmarks/recommend_hit_rate/harness.py            # human report
    python3 benchmarks/recommend_hit_rate/harness.py --json     # scorecard.json only
    python3 benchmarks/recommend_hit_rate/harness.py --check    # exit 1 if regressions

The harness is read-only w.r.t. the package; it only writes scorecard.{json,md}
into this directory.
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml  # type: ignore

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
CORPUS_PATH = HERE / "corpus.yaml"
PAPER_BIB = REPO / "paper.bib"

# --------------------------------------------------------------------------- #
#  Estimator-label normalization: prose method label -> controlled tag.
#  Ordered; first matching substring wins. The single source of judgment about
#  "what did recommend actually propose"; keep it conservative and explicit.
# --------------------------------------------------------------------------- #
_NORMALIZE_RULES: List[Tuple[str, str]] = [
    ("callaway", "callaway_santanna"),
    ("sant'anna", "callaway_santanna"),
    ("sun-abraham", "sun_abraham"),
    ("sun_abraham", "sun_abraham"),
    ("interaction-weighted", "sun_abraham"),
    ("imputation", "did_imputation"),
    ("borusyak", "borusyak_jaravel_spiess"),
    ("gardner", "gardner"),
    ("stacked", "stacked_did"),
    ("chaisemartin", "dcdh"),
    ("classic 2", "classic_2x2"),
    ("2×2", "classic_2x2"),  # 2×2
    ("2x2", "classic_2x2"),
    ("two-way fixed", "twfe"),
    ("twoway fixed", "twfe"),
    ("twfe", "twfe"),
    ("liml", "liml"),
    ("anderson", "anderson_rubin"),
    ("2sls", "twosls"),
    ("two-stage least squares", "twosls"),
    ("jive", "jive"),
    ("rdrobust", "rdrobust"),
    ("local polynomial rd", "local_polynomial_rd"),
    ("local-polynomial", "local_polynomial_rd"),
    ("local polynomial", "local_polynomial_rd"),
    ("cct", "cct"),
    ("regression discontinuity", "rdrobust"),
    ("synthetic control", "synth"),
    ("synthetic did", "synthdid"),
    ("synthetic diff", "synthdid"),
    ("synthdid", "synthdid"),
    ("augmented scm", "augsynth"),
    ("augsynth", "augsynth"),
    ("gsynth", "gsynth"),
    ("matrix completion", "mc_panel"),
    ("propensity score matching", "psm"),
    ("propensity", "psm"),
    ("double ml", "dml"),
    ("double machine", "dml"),
    ("dml", "dml"),
    ("entropy bal", "entropy_balance"),
    ("iptw", "ipw"),
    ("ipw", "ipw"),
    ("marginal structural", "ipw"),
    ("aipw", "aipw"),
    ("g-computation", "g_computation"),
    ("g computation", "g_computation"),
    ("matching", "matching"),
    ("mundlak", "mundlak"),
    ("correlated random effects", "mundlak"),
    ("panel fe", "fe"),
    ("within estimator", "fe"),
    ("fixed effect", "fe"),
    ("logit", "logit"),
    ("poisson", "poisson"),
    ("ols", "ols"),
]

# design label -> audit method family (for static catalog coverage)
_DESIGN_TO_AUDIT_FAMILY: Dict[str, str] = {
    "staggered_did": "did",
    "did_2x2": "did",
    "did": "did",
    "iv": "iv",
    "rd": "rd",
    "synth": "synth",
    "observational": "matching",
    "panel": "regression",
}


def _normalize_method(label: str) -> str:
    low = label.lower()
    for needle, tag in _NORMALIZE_RULES:
        if needle in low:
            return tag
    return f"unknown:{label.strip()[:40]}"


def _load_audit_family_map() -> Dict[str, set]:
    """family -> set(check names) from the real sp.audit catalog (no mirror)."""
    mod = importlib.import_module("statspai.smart.audit")
    checks = list(mod._CAUSAL_CHECKS) + list(mod._REGRESSION_CHECKS)
    fam: Dict[str, set] = {}
    for chk in checks:
        for f in chk.applies_to:
            fam.setdefault(f, set()).add(chk.name)
    return fam


def _verified_bib_keys() -> set:
    if not PAPER_BIB.exists():
        return set()
    text = PAPER_BIB.read_text(encoding="utf-8")
    keys = set()
    for line in text.splitlines():
        line = line.strip()
        if line.startswith("@") and "{" in line:
            keys.add(line.split("{", 1)[1].rstrip(",").strip())
    return keys


def _score_recommend(entry: Dict[str, Any], sp: Any) -> Dict[str, Any]:
    """Run recommend on the real data and score the ranked output."""
    out: Dict[str, Any] = {"id": entry["id"], "design": entry["design"]["label"]}
    loader = entry["data"]["loader"]
    args = dict(entry["data"].get("args", {}))
    gt = entry["ground_truth"]["estimator"]
    acceptable = set(gt["acceptable"])
    disqualifying = set(gt.get("disqualifying", []))

    try:
        df = eval(loader, {"__builtins__": {}}, {"sp": sp})  # noqa: S307 (trusted corpus)
    except Exception as e:  # pragma: no cover - environment dependent
        out.update(status="LOAD_ERROR", error=repr(e)[:160])
        return out

    try:
        rec = sp.recommend(df, **args)
    except Exception as e:
        out.update(status="RECOMMEND_ERROR", error=repr(e)[:160])
        return out

    labels = [r.get("method", "") for r in getattr(rec, "recommendations", [])]
    tags = [_normalize_method(x) for x in labels]
    out["detected_design"] = getattr(rec, "design", None)
    out["ranked_labels"] = labels
    out["ranked_tags"] = tags

    top1 = tags[0] if tags else None
    topk = set(tags)
    hit_top1 = top1 in acceptable
    hit_topk = bool(topk & acceptable)
    hard_miss = top1 in disqualifying

    if hard_miss:
        status = "HARD_MISS"
    elif hit_top1:
        status = "HIT"
    elif hit_topk:
        status = "PARTIAL"
    else:
        status = "MISS"

    out.update(
        status=status,
        top1_tag=top1,
        hit_top1=hit_top1,
        hit_topk=hit_topk,
        hard_miss=hard_miss,
        acceptable=sorted(acceptable),
        disqualifying=sorted(disqualifying),
    )
    return out


def _score_audit_coverage(
    entry: Dict[str, Any], fam_map: Dict[str, set]
) -> Dict[str, Any]:
    """Static: does the audit catalog (for this design family) know the
    ground-truth robustness checks a referee would demand?"""
    design = entry["design"]["label"]
    family = _DESIGN_TO_AUDIT_FAMILY.get(design, "generic")
    catalog = fam_map.get(family, set())
    gt_checks = [c["check"] for c in entry["ground_truth"].get("robustness_checks", [])]
    # Only score checks whose provenance is a real source (not citation_needed).
    scored = [
        c["check"]
        for c in entry["ground_truth"].get("robustness_checks", [])
        if c.get("provenance") != "citation_needed"
    ]
    covered = [c for c in scored if c in catalog]
    missing = [c for c in scored if c not in catalog]
    recall = (len(covered) / len(scored)) if scored else None
    return {
        "id": entry["id"],
        "audit_family": family,
        "gt_checks": gt_checks,
        "scored_checks": scored,
        "covered": covered,
        "missing_from_catalog": missing,
        "catalog_recall": recall,
    }


def _score_audit_dynamic(entry: Dict[str, Any], sp: Any) -> Dict[str, Any]:
    """Dynamic: fit the top-1 recommendation, run sp.audit on the fitted result,
    and check whether it actually surfaces the ground-truth referee checks (and
    emits an actionable suggest_function for the ones it flags missing)."""
    out: Dict[str, Any] = {"id": entry["id"]}
    scored = [
        c["check"]
        for c in entry["ground_truth"].get("robustness_checks", [])
        if c.get("provenance") != "citation_needed"
    ]
    try:
        df = eval(entry["data"]["loader"], {"__builtins__": {}}, {"sp": sp})  # noqa: S307
        rec = sp.recommend(df, **dict(entry["data"].get("args", {})))
        result = rec.run(which=0)
        card = sp.audit(result)
    except Exception as e:
        out.update(status="AUDIT_ERROR", error=repr(e)[:160], dynamic_recall=None)
        return out

    emitted = {c["name"]: c for c in card.get("checks", [])}
    covered = [c for c in scored if c in emitted]
    missing = [c for c in scored if c not in emitted]
    # actionable = emitted, flagged not-yet-done, with a function to run next.
    actionable = [
        c for c in covered
        if emitted[c].get("status") in ("missing", "failed")
        and emitted[c].get("suggest_function")
    ]
    out.update(
        status="OK",
        audit_family=card.get("method_family"),
        scored_checks=scored,
        covered=covered,
        not_emitted=missing,
        actionable_next_steps=actionable,
        dynamic_recall=(len(covered) / len(scored)) if scored else None,
    )
    return out


def run(fit: bool = True) -> Dict[str, Any]:
    import statspai as sp  # local import so --help works without the package

    corpus = yaml.safe_load(CORPUS_PATH.read_text(encoding="utf-8"))
    entries = corpus["entries"]
    fam_map = _load_audit_family_map()
    verified = _verified_bib_keys()

    # §10 guard: every cited bib_key must be in the verified paper.bib.
    citation_errors = []
    for e in entries:
        bk = e.get("paper", {}).get("bib_key")
        if bk and verified and bk not in verified:
            citation_errors.append((e["id"], bk))

    rec_rows = [_score_recommend(e, sp) for e in entries]
    audit_rows = [_score_audit_coverage(e, fam_map) for e in entries]
    audit_dyn_rows = [_score_audit_dynamic(e, sp) for e in entries] if fit else []

    n = len(rec_rows)
    n_hit1 = sum(1 for r in rec_rows if r.get("hit_top1"))
    n_hitk = sum(1 for r in rec_rows if r.get("hit_topk"))
    n_hard = sum(1 for r in rec_rows if r.get("hard_miss"))
    n_err = sum(1 for r in rec_rows if r.get("status", "").endswith("ERROR"))
    recalls = [a["catalog_recall"] for a in audit_rows if a["catalog_recall"] is not None]
    mean_recall = sum(recalls) / len(recalls) if recalls else None
    dyn_recalls = [a["dynamic_recall"] for a in audit_dyn_rows if a.get("dynamic_recall") is not None]
    mean_dyn = sum(dyn_recalls) / len(dyn_recalls) if dyn_recalls else None
    n_audit_err = sum(1 for a in audit_dyn_rows if a.get("status") == "AUDIT_ERROR")

    return {
        "corpus_version": corpus.get("corpus_version"),
        "statspai_version": getattr(sp, "__version__", "?"),
        "n_entries": n,
        "summary": {
            "hit_rate_top1": round(n_hit1 / n, 4) if n else None,
            "hit_rate_topk": round(n_hitk / n, 4) if n else None,
            "hard_miss_rate": round(n_hard / n, 4) if n else None,
            "n_errors": n_err,
            "audit_catalog_mean_recall": (
                round(mean_recall, 4) if mean_recall is not None else None
            ),
            "audit_dynamic_mean_recall": (
                round(mean_dyn, 4) if mean_dyn is not None else None
            ),
            "n_audit_errors": n_audit_err,
        },
        "citation_errors": citation_errors,
        "recommend": rec_rows,
        "audit_coverage": audit_rows,
        "audit_dynamic": audit_dyn_rows,
    }


_STATUS_GLYPH = {
    "HIT": "✓ HIT",
    "PARTIAL": "~ PARTIAL",
    "MISS": "✗ MISS",
    "HARD_MISS": "⚠ HARD-MISS",
    "LOAD_ERROR": "! LOAD-ERR",
    "RECOMMEND_ERROR": "! REC-ERR",
}


def render_markdown(card: Dict[str, Any]) -> str:
    s = card["summary"]
    lines = [
        "# Recommendation Hit-Rate Scorecard",
        "",
        f"- corpus: `{card['corpus_version']}`  |  statspai: `{card['statspai_version']}`"
        f"  |  entries: **{card['n_entries']}** (Tier-A, data-backed)",
        f"- **top-1 hit-rate: {s['hit_rate_top1']}**  |  top-k hit-rate: {s['hit_rate_topk']}"
        f"  |  hard-miss rate: {s['hard_miss_rate']}  |  errors: {s['n_errors']}",
        f"- audit catalog mean recall (static): {s['audit_catalog_mean_recall']}"
        f"  |  audit dynamic mean recall (fit+audit): {s.get('audit_dynamic_mean_recall')}"
        f"  |  audit errors: {s.get('n_audit_errors')}",
        "",
    ]
    if card["citation_errors"]:
        lines += ["> ⚠ CITATION ERRORS (bib_key not in paper.bib): "
                  + ", ".join(f"{i}:{k}" for i, k in card["citation_errors"]), ""]
    lines += [
        "## recommend hit-rate (dynamic — runs on real data)",
        "",
        "| design | id | status | detected | top-1 tag |",
        "| --- | --- | --- | --- | --- |",
    ]
    for r in card["recommend"]:
        lines.append(
            f"| {r.get('design','')} | `{r['id']}` | {_STATUS_GLYPH.get(r.get('status'), r.get('status'))}"
            f" | {r.get('detected_design','')} | `{r.get('top1_tag', r.get('error',''))}` |"
        )
    lines += [
        "",
        "## audit catalog coverage (static — does the catalog know the referee checks)",
        "",
        "| id | family | recall | missing from catalog |",
        "| --- | --- | --- | --- |",
    ]
    for a in card["audit_coverage"]:
        miss = ", ".join(a["missing_from_catalog"]) or "—"
        rc = a["catalog_recall"]
        lines.append(f"| `{a['id']}` | {a['audit_family']} | {rc} | {miss} |")
    if card.get("audit_dynamic"):
        lines += [
            "",
            "## audit recall (dynamic — fit the estimator, run sp.audit, does it ask)",
            "",
            "| id | fitted family | recall | actionable next-steps (missing→suggest_function) |",
            "| --- | --- | --- | --- |",
        ]
        for a in card["audit_dynamic"]:
            if a.get("status") == "AUDIT_ERROR":
                lines.append(f"| `{a['id']}` | — | ERR | {a.get('error','')[:60]} |")
                continue
            steps = ", ".join(a.get("actionable_next_steps", [])) or "—"
            lines.append(
                f"| `{a['id']}` | {a.get('audit_family')} | {a.get('dynamic_recall')} | {steps} |"
            )
    lines += ["", "_Generated by benchmarks/recommend_hit_rate/harness.py_"]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--json", action="store_true", help="write scorecard.json only")
    ap.add_argument("--check", action="store_true",
                    help="exit 1 on any hard-miss, error, or citation error")
    ap.add_argument("--no-fit", action="store_true",
                    help="skip the dynamic audit pass (faster; recommend hit-rate only)")
    args = ap.parse_args()

    card = run(fit=not args.no_fit)
    (HERE / "scorecard.json").write_text(json.dumps(card, indent=2), encoding="utf-8")
    md = render_markdown(card)
    (HERE / "scorecard.md").write_text(md + "\n", encoding="utf-8")

    if not args.json:
        print(md)

    if args.check:
        bad = (card["summary"]["n_errors"] > 0
               or card["summary"].get("n_audit_errors", 0) > 0
               or card["summary"]["hard_miss_rate"]
               or card["citation_errors"])
        return 1 if bad else 0
    return 0


if __name__ == "__main__":
    sys.exit(main())
