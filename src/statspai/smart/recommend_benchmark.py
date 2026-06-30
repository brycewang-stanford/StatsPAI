"""Recommendation hit-rate benchmark — the agent-native correctness gate.

``sp.recommend_benchmark()`` scores ``sp.recommend`` (and ``sp.audit`` coverage)
against a ground-truth corpus of published / archetypal empirical designs and
returns a structured scorecard. The moat is not "the agent loop runs" but "the
agent loop is right": a plausible-but-wrong recommendation is worse than none,
because it carries authority. This benchmark measures, across real designs,
whether the top-k recommendation matches the estimator the design calls for,
and whether the audit knows the robustness checks a referee would demand.

Two metrics
-----------
- **recommend hit-rate** (dynamic): run ``sp.recommend`` and check whether an
  econometrically-acceptable estimator appears in the ranked output. A *hard
  miss* is leading (top-1) with a disqualifying estimator (e.g. TWFE on a
  staggered design) — the failure that destroys trust.
- **audit recall**: static (does the ``sp.audit`` catalog know the design's
  ground-truth checks) and dynamic (fit the top-1 estimator, run ``sp.audit``,
  confirm it surfaces them with an actionable ``suggest_function``).

The corpus ships with the package (``data/recommend_corpus.yaml``); every cited
``bib_key`` resolves in ``paper.bib`` (CLAUDE.md §10). See
``benchmarks/recommend_hit_rate/`` for the CLI, scorecard, and findings log.
"""

from __future__ import annotations

import importlib
from importlib import resources
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

__all__ = ["recommend_benchmark", "render_markdown", "normalize_method"]

# --------------------------------------------------------------------------- #
#  Estimator-label normalization: prose method label -> controlled tag.
#  Ordered; first matching substring wins. The single explicit judgment about
#  "what did recommend actually propose"; kept conservative and reviewable.
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
    ("pooled did", "classic_2x2"),
    ("2×2", "classic_2x2"),
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
    # frontier design families (F-004)
    ("bunching", "bunching"),
    ("regression kink", "rkd"),
    ("rkd", "rkd"),
    ("triple difference", "ddd"),
    ("ddd", "ddd"),
    ("bartik", "bartik"),
    ("shift-share", "bartik"),
    ("oaxaca", "oaxaca"),
    ("decomposition", "oaxaca"),
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


def normalize_method(label: str) -> str:
    """Map a prose ``recommend`` method label to a controlled estimator tag.

    Parameters
    ----------
    label : str
        A ``method`` string from ``RecommendationResult.recommendations``.

    Returns
    -------
    str
        A controlled tag (e.g. ``"callaway_santanna"``), or
        ``"unknown:<label>"`` if no rule matches.

    Examples
    --------
    >>> from statspai.smart.recommend_benchmark import normalize_method
    >>> normalize_method("Callaway-Sant'Anna (2021) — staggered DID")
    'callaway_santanna'
    """
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


def _default_corpus_path() -> Optional[Path]:
    """Locate the bundled corpus (wheel) or the dev copy (source checkout)."""
    try:
        ref = (
            resources.files("statspai.smart")
            .joinpath("data")
            .joinpath("recommend_corpus.yaml")
        )
        with resources.as_file(ref) as p:
            if p.exists():
                return Path(p)
    except (ModuleNotFoundError, FileNotFoundError, AttributeError):
        pass
    # Source-checkout fallback: repo benchmarks/ copy.
    here = Path(__file__).resolve()
    for parent in here.parents:
        cand = parent / "benchmarks" / "recommend_hit_rate" / "corpus.yaml"
        if cand.exists():
            return cand
    return None


def _verified_bib_keys() -> set:
    """bib keys present in paper.bib, when running from a source checkout."""
    here = Path(__file__).resolve()
    for parent in here.parents:
        bib = parent / "paper.bib"
        if bib.exists():
            keys = set()
            for line in bib.read_text(encoding="utf-8").splitlines():
                line = line.strip()
                if line.startswith("@") and "{" in line:
                    keys.add(line.split("{", 1)[1].rstrip(",").strip())
            return keys
    return set()  # installed wheel has no paper.bib; skip the §10 cross-check


def _load_df(loader: str, sp: Any) -> Any:
    """Evaluate a corpus ``loader`` string into a DataFrame.

    The loader is a trusted, repo-internal call string authored in
    ``corpus.yaml`` (e.g. ``"sp.datasets.mpdta()"`` or
    ``"sp.dgp_did(n_units=150, ...)"``) — never user input. Builtins are
    stripped and only ``sp`` is in scope, so it cannot reach the filesystem,
    imports, or arbitrary objects. Safe by construction.
    """
    return eval(loader, {"__builtins__": {}}, {"sp": sp})  # noqa: S307  # nosec


def _score_recommend(entry: Dict[str, Any], sp: Any) -> Dict[str, Any]:
    out: Dict[str, Any] = {"id": entry["id"], "design": entry["design"]["label"]}
    gt = entry["ground_truth"]["estimator"]
    acceptable = set(gt["acceptable"])
    disqualifying = set(gt.get("disqualifying", []))
    try:
        df = _load_df(entry["data"]["loader"], sp)
    except Exception as e:  # pragma: no cover - environment dependent
        out.update(status="LOAD_ERROR", error=repr(e)[:160])
        return out
    try:
        rec = sp.recommend(df, **dict(entry["data"].get("args", {})))
    except Exception as e:
        out.update(status="RECOMMEND_ERROR", error=repr(e)[:160])
        return out

    labels = [r.get("method", "") for r in getattr(rec, "recommendations", [])]
    tags = [normalize_method(x) for x in labels]
    top1 = tags[0] if tags else None
    topk = set(tags)
    hit_top1 = top1 in acceptable
    hit_topk = bool(topk & acceptable)
    hard_miss = top1 in disqualifying
    status = (
        "HARD_MISS"
        if hard_miss
        else "HIT"
        if hit_top1
        else "PARTIAL"
        if hit_topk
        else "MISS"
    )
    out.update(
        status=status,
        detected_design=getattr(rec, "design", None),
        ranked_labels=labels,
        ranked_tags=tags,
        top1_tag=top1,
        hit_top1=hit_top1,
        hit_topk=hit_topk,
        hard_miss=hard_miss,
        acceptable=sorted(acceptable),
        disqualifying=sorted(disqualifying),
    )
    return out


def _scored_checks(entry: Dict[str, Any]) -> List[str]:
    return [
        c["check"]
        for c in entry["ground_truth"].get("robustness_checks", [])
        if c.get("provenance") != "citation_needed"
    ]


def _score_audit_coverage(
    entry: Dict[str, Any], fam_map: Dict[str, set]
) -> Dict[str, Any]:
    design = entry["design"]["label"]
    family = _DESIGN_TO_AUDIT_FAMILY.get(design, "generic")
    catalog = fam_map.get(family, set())
    scored = _scored_checks(entry)
    covered = [c for c in scored if c in catalog]
    missing = [c for c in scored if c not in catalog]
    return {
        "id": entry["id"],
        "audit_family": family,
        "covered": covered,
        "missing_from_catalog": missing,
        "catalog_recall": (len(covered) / len(scored)) if scored else None,
    }


def _score_audit_dynamic(entry: Dict[str, Any], sp: Any) -> Dict[str, Any]:
    out: Dict[str, Any] = {"id": entry["id"]}
    scored = _scored_checks(entry)
    try:
        df = _load_df(entry["data"]["loader"], sp)
        rec = sp.recommend(df, **dict(entry["data"].get("args", {})))
        result = rec.run(which=0)
        card = sp.audit(result)
    except Exception as e:
        out.update(status="AUDIT_ERROR", error=repr(e)[:160], dynamic_recall=None)
        return out
    emitted = {c["name"]: c for c in card.get("checks", [])}
    covered = [c for c in scored if c in emitted]
    actionable = [
        c
        for c in covered
        if emitted[c].get("status") in ("missing", "failed")
        and emitted[c].get("suggest_function")
    ]
    out.update(
        status="OK",
        audit_family=card.get("method_family"),
        covered=covered,
        not_emitted=[c for c in scored if c not in emitted],
        actionable_next_steps=actionable,
        dynamic_recall=(len(covered) / len(scored)) if scored else None,
    )
    return out


def recommend_benchmark(
    *,
    fit: bool = True,
    corpus_path: Optional[str] = None,
) -> Dict[str, Any]:
    """Score ``sp.recommend`` / ``sp.audit`` against the ground-truth corpus.

    Runs every corpus entry through ``sp.recommend`` on its (real or synthetic)
    data and scores whether an econometrically-acceptable estimator is
    recommended, plus whether ``sp.audit`` covers the design's ground-truth
    robustness checks. This is the public, agent-native hit-rate gate.

    Parameters
    ----------
    fit : bool, default True
        Also run the dynamic audit pass (fit each top-1 estimator and run
        ``sp.audit`` on the result). Set ``False`` for a faster
        recommend-only run.
    corpus_path : str, optional
        Path to an alternative corpus YAML. Defaults to the bundled corpus.

    Returns
    -------
    dict
        A JSON-safe scorecard with ``summary`` (``hit_rate_top1``,
        ``hard_miss_rate``, ``audit_dynamic_mean_recall`` …), ``recommend``
        (per-entry rows), ``audit_coverage``, ``audit_dynamic``, and any
        ``citation_errors``.

    Examples
    --------
    >>> from statspai.smart.recommend_benchmark import recommend_benchmark
    >>> callable(recommend_benchmark)  # card = sp.recommend_benchmark()
    True
    """
    try:
        import yaml
    except ImportError as e:  # pragma: no cover - optional dep
        raise ImportError(
            "sp.recommend_benchmark() requires pyyaml. Install with "
            "`pip install pyyaml` or `pip install statspai[dev]`."
        ) from e

    import statspai as sp

    path = Path(corpus_path) if corpus_path else _default_corpus_path()
    if path is None or not Path(path).exists():
        raise FileNotFoundError(
            "recommend_benchmark corpus not found; expected bundled "
            "statspai/smart/data/recommend_corpus.yaml or a source-checkout "
            "benchmarks/recommend_hit_rate/corpus.yaml."
        )
    corpus = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    entries = corpus["entries"]
    fam_map = _load_audit_family_map()
    verified = _verified_bib_keys()

    citation_errors = []
    for e in entries:
        bk = e.get("paper", {}).get("bib_key")
        if bk and verified and bk not in verified:
            citation_errors.append((e["id"], bk))

    gap_flags = [bool(e.get("gap_probe")) for e in entries]
    rec_rows = [_score_recommend(e, sp) for e in entries]
    for row, gp in zip(rec_rows, gap_flags):
        row["gap_probe"] = gp
    audit_rows = [_score_audit_coverage(e, fam_map) for e in entries]
    dyn_rows = [_score_audit_dynamic(e, sp) for e in entries] if fit else []

    # Headline metrics are over CORE designs — the families recommend is
    # expected to handle. Frontier (``gap_probe: true``) designs are tracked
    # separately so adding a not-yet-supported design never depresses the
    # headline hit-rate or breaks the CI ratchet. As a frontier branch lands,
    # flip its corpus entry's gap_probe off to promote it into the headline.
    core = [r for r in rec_rows if not r.get("gap_probe")]
    frontier = [r for r in rec_rows if r.get("gap_probe")]
    core_ids = {r["id"] for r in core}
    n = len(core)
    n_hit1 = sum(1 for r in core if r.get("hit_top1"))
    n_hitk = sum(1 for r in core if r.get("hit_topk"))
    n_hard = sum(1 for r in core if r.get("hard_miss"))
    n_err = sum(1 for r in core if r.get("status", "").endswith("ERROR"))
    recalls = [
        a["catalog_recall"]
        for a in audit_rows
        if a["id"] in core_ids and a["catalog_recall"] is not None
    ]
    mean_recall = sum(recalls) / len(recalls) if recalls else None
    dyn_recalls = [
        a["dynamic_recall"]
        for a in dyn_rows
        if a["id"] in core_ids and a.get("dynamic_recall") is not None
    ]
    mean_dyn = sum(dyn_recalls) / len(dyn_recalls) if dyn_recalls else None
    n_audit_err = sum(
        1 for a in dyn_rows if a["id"] in core_ids and a.get("status") == "AUDIT_ERROR"
    )
    n_frontier = len(frontier)
    n_frontier_hit = sum(1 for r in frontier if r.get("hit_top1"))

    return {
        "corpus_version": corpus.get("corpus_version"),
        "statspai_version": getattr(sp, "__version__", "?"),
        "n_entries": len(rec_rows),
        "n_core": n,
        "n_tier_a": sum(1 for e in entries if e.get("data", {}).get("tier") == "A"),
        "n_tier_b": sum(1 for e in entries if e.get("data", {}).get("tier") == "B"),
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
        "frontier": {
            "n_frontier": n_frontier,
            "n_hit": n_frontier_hit,
            "coverage": round(n_frontier_hit / n_frontier, 4) if n_frontier else None,
            "ids": [r["id"] for r in frontier],
        },
        "citation_errors": citation_errors,
        "recommend": rec_rows,
        "audit_coverage": audit_rows,
        "audit_dynamic": dyn_rows,
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
    """Render a scorecard dict (from :func:`recommend_benchmark`) as Markdown.

    Parameters
    ----------
    card : dict
        The return value of :func:`recommend_benchmark`.

    Returns
    -------
    str
        A human-readable Markdown scorecard.

    Examples
    --------
    >>> from statspai.smart.recommend_benchmark import render_markdown
    >>> callable(render_markdown)
    True
    """
    s = card["summary"]
    fr = card.get("frontier", {})
    lines = [
        "# Recommendation Hit-Rate Scorecard",
        "",
        f"- corpus: `{card['corpus_version']}`  |  statspai: `{card['statspai_version']}`"
        f"  |  entries: **{card['n_entries']}**"
        f" ({card.get('n_core', card['n_entries'])} core + {fr.get('n_frontier', 0)} frontier;"
        f" {card.get('n_tier_a', '?')} Tier-A + {card.get('n_tier_b', '?')} Tier-B)",
        f"- **core top-1 hit-rate: {s['hit_rate_top1']}**  |  top-k: {s['hit_rate_topk']}"
        f"  |  hard-miss rate: {s['hard_miss_rate']}  |  errors: {s['n_errors']}",
        f"- audit catalog mean recall (static): {s['audit_catalog_mean_recall']}"
        f"  |  audit dynamic mean recall (fit+audit): {s.get('audit_dynamic_mean_recall')}"
        f"  |  audit errors: {s.get('n_audit_errors')}",
        f"- frontier coverage (gap-probe designs recommend is being taught): "
        f"**{fr.get('coverage')}** ({fr.get('n_hit', 0)}/{fr.get('n_frontier', 0)})",
        "",
    ]
    if card["citation_errors"]:
        lines += [
            "> ⚠ CITATION ERRORS (bib_key not in paper.bib): "
            + ", ".join(f"{i}:{k}" for i, k in card["citation_errors"]),
            "",
        ]
    lines += [
        "## recommend hit-rate (dynamic — runs on real / synthetic data)",
        "",
        "_Frontier (gap-probe) designs are marked ⊕ and scored separately; they "
        "do not affect the core hit-rate or the CI ratchet._",
        "",
        "| design | id | status | detected | top-1 tag |",
        "| --- | --- | --- | --- | --- |",
    ]
    for r in card["recommend"]:
        gp = " ⊕" if r.get("gap_probe") else ""
        lines.append(
            f"| {r.get('design','')}{gp} | `{r['id']}` | {_STATUS_GLYPH.get(r.get('status'), r.get('status'))}"
            f" | {r.get('detected_design','')} | `{r.get('top1_tag', r.get('error',''))}` |"
        )
    if card.get("audit_dynamic"):
        lines += [
            "",
            "## audit recall (dynamic — fit the estimator, run sp.audit, does it ask)",
            "",
            "| id | fitted family | recall | actionable next-steps |",
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
    lines += ["", "_Generated by sp.recommend_benchmark()_"]
    return "\n".join(lines)
