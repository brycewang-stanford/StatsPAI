"""Ranking for :func:`statspai.search_functions`.

The registry search used to require at least half of a query's content words
to hit a function's name / tags / description. That rule fails exactly on the
queries an applied researcher types: domain nouns ("minimum wage employment
difference in differences") never appear in any description, so the query
returned nothing, and estimand vocabulary ("local average treatment effect")
did not reach the IV family at all.

This module keeps the original scoring (name > tag > description) and adds:

* **estimand / design vocabulary** -- multi-word phrases collapse to a
  concept token (``local average treatment effect`` -> ``late``) and each
  concept names the registered entry points that estimate it, dispatcher
  first (``synthetic control`` ranks ``sp.synth`` above its variants). A
  concept target always counts as a full match, so domain nouns around the
  design words no longer empty the result;
* **agent-card text** -- a function's own assumptions / pre-conditions /
  negative guidance are scored at low weight (they never make a function
  match on their own, they only break ties);
* **aliases rank below their canonical entry** (``sp.rdd`` below
  ``sp.rdrobust``);
* **partial fallback** -- a non-empty query with at least one term hit never
  returns ``[]``; when nothing clears the half-of-the-words bar the best
  partial matches come back flagged ``match: "partial"``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

_STOPWORDS = frozenset(
    {
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "by",
        "can",
        "do",
        "does",
        "for",
        "from",
        "how",
        "i",
        "in",
        "into",
        "is",
        "it",
        "my",
        "of",
        "on",
        "or",
        "the",
        "that",
        "this",
        "to",
        "use",
        "using",
        "want",
        "with",
        "what",
        "which",
        "when",
        "estimate",
        "estimating",
        "estimator",
        "estimation",
        "data",
        "function",
        "method",
        "model",
        "run",
        "compute",
        "should",
        "we",
        "our",
        "get",
        "find",
    }
)

SYNONYMS: Dict[str, Tuple[str, ...]] = {
    "did": ("did", "difference-in-differences", "diff-in-diff"),
    "diff": ("did",),
    "differences": ("did",),
    "rd": ("rd", "discontinuity", "rdrobust"),
    "rdd": ("rd", "discontinuity"),
    "discontinuity": ("rd", "discontinuity"),
    "iv": ("iv", "instrument", "instrumental"),
    "instrument": ("iv", "instrument", "instrumental"),
    "instruments": ("iv", "instrument", "instrumental"),
    "twfe": ("twfe", "two-way", "fixed effects"),
    "fe": ("fixed effects", "fixest", "feols", "fe"),
    "staggered": ("staggered", "callaway", "sun-abraham", "cohort"),
    "cohort": ("cohort", "staggered", "callaway"),
    "synthetic": ("synthetic", "synth"),
    "scm": ("synthetic", "synth"),
    "sc": ("synthetic", "synth"),
    "matching": ("matching", "match", "psm", "propensity"),
    "propensity": ("propensity", "psm", "ipw"),
    "heterogeneous": ("heterogeneous", "cate", "forest", "metalearner"),
    "heterogeneity": ("heterogeneous", "cate", "forest", "metalearner"),
    "cate": ("cate", "forest", "metalearner", "heterogeneous"),
    "weak": ("weak", "anderson-rubin", "anderson_rubin", "weakrobust"),
    "confidence": ("confidence", "ci"),
    "interval": ("interval", "ci"),
    "intervals": ("interval", "ci"),
    "ci": ("ci", "confidence"),
    "bandwidth": ("bandwidth", "rd"),
    "manipulation": ("mccrary", "density", "manipulation"),
    "pretrends": ("pretrend", "parallel trends", "pre-trend"),
    "pretrend": ("pretrend", "parallel trends", "pre-trend"),
    "trends": ("trend",),
    "bias": ("bias", "goodman-bacon", "bacon"),
    "covariates": ("covariate", "control", "adjust"),
    "controls": ("covariate", "control", "adjust"),
    "cluster": ("cluster",),
    "clustered": ("cluster",),
    "policy": ("policy", "treatment", "intervention"),
    "effect": ("effect", "att", "ate", "treatment"),
    "effects": ("effect", "att", "ate", "treatment"),
    "timing": ("timing", "staggered", "cohort", "event"),
    "adoption": ("adoption", "staggered", "cohort"),
    "event": ("event", "event_study", "dynamic"),
    "dml": ("dml", "double machine learning", "debiased"),
    "late": ("late", "complier", "instrument", "iv"),
    "complier": ("complier", "late", "iv"),
    "compliers": ("complier", "late", "iv"),
    "att": ("att", "treated"),
    "ate": ("ate", "average treatment effect"),
    "rkd": ("rkd", "kink"),
    "kink": ("kink", "rkd"),
    "tsls": ("2sls", "tsls", "two-stage", "iv"),
    "2sls": ("2sls", "tsls", "two-stage", "iv"),
    "psm": ("psm", "propensity", "matching"),
    "ipw": ("ipw", "inverse probability", "weighting"),
    "iptw": ("ipw", "iptw", "inverse probability"),
    "aipw": ("aipw", "doubly robust", "augmented"),
    "doublyrobust": ("doubly robust", "doubly-robust", "aipw", "dr"),
    "tmle": ("tmle", "targeted"),
    "mediation": ("mediation", "mediator", "mediate"),
    "bunching": ("bunching", "bunch"),
    "sdid": ("sdid", "synthetic difference"),
    "ddd": ("ddd", "triple"),
    "qte": ("qte", "quantile treatment"),
    "its": ("its", "interrupted"),
    "mr": ("mendelian", "mr"),
    "oaxaca": ("oaxaca", "blinder", "decomposition"),
    "power": ("power", "sample size", "mde"),
}

#: Multi-word vocabulary, longest first, collapsed to one concept token.
PHRASES: Tuple[Tuple[str, str], ...] = (
    ("conditional average treatment effects", "cate"),
    ("conditional average treatment effect", "cate"),
    ("heterogeneous treatment effects", "cate"),
    ("heterogeneous treatment effect", "cate"),
    ("treatment effect heterogeneity", "cate"),
    ("local average treatment effect", "late"),
    ("complier average causal effect", "late"),
    ("average treatment effect on the treated", "att"),
    ("average treatment effect on treated", "att"),
    ("average effect of treatment on the treated", "att"),
    ("average treatment effect", "ate"),
    ("synthetic difference in differences", "sdid"),
    ("synthetic difference-in-differences", "sdid"),
    ("synthetic diff in diff", "sdid"),
    ("difference in difference in differences", "ddd"),
    ("triple differences", "ddd"),
    ("triple difference", "ddd"),
    ("regression discontinuity design", "rd"),
    ("regression discontinuity", "rd"),
    ("regression kink design", "rkd"),
    ("regression kink", "rkd"),
    ("difference in differences", "did"),
    ("differences in differences", "did"),
    ("difference-in-differences", "did"),
    ("differences-in-differences", "did"),
    ("difference in difference", "did"),
    ("diff in diff", "did"),
    ("diff-in-diff", "did"),
    ("two stage least squares", "tsls"),
    ("two-stage least squares", "tsls"),
    ("instrumental variables", "iv"),
    ("instrumental variable", "iv"),
    ("synthetic control method", "synth"),
    ("synthetic controls", "synth"),
    ("synthetic control", "synth"),
    ("two way fixed effects", "twfe"),
    ("two-way fixed effects", "twfe"),
    ("fixed effects", "fe"),
    ("average marginal effects", "margins"),
    ("marginal effects", "margins"),
    ("marginal effect", "margins"),
    ("propensity score matching", "psm"),
    ("propensity score", "propensity"),
    ("augmented inverse probability weighting", "aipw"),
    ("augmented inverse propensity weighting", "aipw"),
    ("inverse probability of treatment weighting", "iptw"),
    ("inverse probability weighting", "ipw"),
    ("inverse propensity weighting", "ipw"),
    ("inverse probability weights", "ipw"),
    ("doubly robust", "doublyrobust"),
    ("doubly-robust", "doublyrobust"),
    ("targeted maximum likelihood", "tmle"),
    ("targeted minimum loss", "tmle"),
    ("double machine learning", "dml"),
    ("debiased machine learning", "dml"),
    ("double/debiased machine learning", "dml"),
    ("event study", "event_study"),
    ("event-study", "event_study"),
    ("mediation analysis", "mediation"),
    ("causal mediation", "mediation"),
    ("quantile treatment effects", "qte"),
    ("quantile treatment effect", "qte"),
    ("interrupted time series", "its"),
    ("mendelian randomization", "mr"),
    ("mendelian randomisation", "mr"),
    ("parallel trends", "pretrend"),
    ("pre-trends", "pretrend"),
    ("staggered adoption", "staggered"),
    ("staggered treatment", "staggered"),
    ("oaxaca blinder", "oaxaca"),
    ("blinder oaxaca", "oaxaca"),
    ("blinder-oaxaca", "oaxaca"),
    ("shift share", "bartik"),
    ("shift-share", "bartik"),
    ("power analysis", "power"),
    ("sample size", "power"),
    ("minimum detectable effect", "power"),
    ("principal stratification", "principal_strat"),
    ("front door", "frontdoor"),
    ("front-door", "frontdoor"),
    ("causal forest", "causal_forest"),
    ("causal forests", "causal_forest"),
    ("sensitivity analysis", "sensitivity"),
    ("unobserved confounding", "sensitivity"),
    ("omitted variable bias", "sensitivity"),
    ("multiple hypothesis testing", "mht"),
    ("multiple testing", "mht"),
    ("wild cluster bootstrap", "wildboot"),
)

#: Concept token -> registered entry points, most canonical first. Names that
#: are not registered are skipped at query time, so a rename degrades to "no
#: boost" rather than an error. Only estimand / design vocabulary lives here;
#: every target must be an entry point that actually estimates the concept.
CONCEPT_TARGETS: Dict[str, Tuple[str, ...]] = {
    "did": ("did", "callaway_santanna", "event_study"),
    "staggered": ("callaway_santanna", "sun_abraham", "did_imputation"),
    "event_study": ("event_study", "sun_abraham", "callaway_santanna"),
    "pretrend": ("pretrends_test", "honest_did", "event_study"),
    "twfe": ("feols", "bacon_decomposition"),
    "fe": ("feols", "panel"),
    "margins": ("margins",),
    "ddd": ("ddd",),
    "rd": ("rdrobust", "rd", "rddensity"),
    "rdd": ("rdrobust", "rd"),
    "rkd": ("rkd",),
    "kink": ("rkd", "bunching"),
    "bunching": ("bunching",),
    "iv": ("iv", "ivreg"),
    "late": ("iv", "ivreg"),
    "complier": ("iv", "ivreg"),
    "compliers": ("iv", "ivreg"),
    "tsls": ("ivreg", "iv"),
    "2sls": ("ivreg", "iv"),
    "synth": ("synth", "sdid", "augsynth"),
    "scm": ("synth", "sdid", "augsynth"),
    "sdid": ("sdid",),
    "cate": ("metalearner", "causal_forest"),
    "causal_forest": ("causal_forest",),
    "ate": ("aipw", "dml", "ipw"),
    "att": ("did", "match", "ipw"),
    "psm": ("match", "psm"),
    "matching": ("match",),
    "propensity": ("ipw", "match"),
    "ipw": ("ipw",),
    "iptw": ("msm", "ipw"),
    "aipw": ("aipw",),
    "doublyrobust": ("aipw", "drdid", "tmle"),
    "tmle": ("tmle",),
    "dml": ("dml",),
    "mediation": ("mediate",),
    "qte": ("qte",),
    "its": ("its",),
    "mr": ("mr",),
    "oaxaca": ("decompose",),
    "bartik": ("bartik",),
    "power": ("power",),
    "principal_strat": ("principal_strat",),
    "frontdoor": ("front_door",),
    "sensitivity": ("sensemakr", "evalue"),
    "mht": ("adjust_pvalues", "romano_wolf"),
    "wildboot": ("wild_cluster_bootstrap",),
}

#: Down-weight for concept words that are ambiguous on their own (a bare
#: "kink" may be a regression kink or a bunching kink; "ATT" is estimated
#: by several designs). Unlisted concepts weigh 1.0.
CONCEPT_WEIGHT: Dict[str, float] = {
    "kink": 0.5,
    "propensity": 0.6,
    "att": 0.5,
    "ate": 0.6,
    "matching": 0.8,
}

#: Weight of an agent-card hit relative to a description hit (1.0).
CARD_WEIGHT = 0.5
#: Rank below the canonical entry when a spec is an alias.
ALIAS_PENALTY = 4.0
#: Maximum number of partial matches returned by the fallback.
PARTIAL_LIMIT = 25


def search_terms(query: str) -> List[Tuple[str, Tuple[str, ...]]]:
    """Tokenise ``query`` into ``(word, alternative spellings)`` pairs."""
    lowered = " " + query.lower() + " "
    for phrase, token in PHRASES:
        lowered = lowered.replace(phrase, f" {token} ")
    raw = [w.strip(".,;:()[]'\"?!") for w in lowered.replace("_", " ").split()]
    # ``event_study`` / ``causal_forest`` / ``principal_strat`` are single
    # concept tokens; the underscore split above must not break them.
    raw = _rejoin_concepts(raw)
    terms: List[Tuple[str, Tuple[str, ...]]] = []
    for w in raw:
        if not w or w in _STOPWORDS:
            continue
        alts = tuple(dict.fromkeys((w,) + SYNONYMS.get(w, ())))
        terms.append((w, alts))
    if not terms:
        terms = [(w, (w,)) for w in raw if w]
    return terms


_JOINED = ("event_study", "causal_forest", "principal_strat")


def _rejoin_concepts(words: List[str]) -> List[str]:
    out: List[str] = []
    i = 0
    while i < len(words):
        pair = "_".join(words[i : i + 2])
        if pair in _JOINED:
            out.append(pair)
            i += 2
            continue
        out.append(words[i])
        i += 1
    return out


_CARD_TEXT_CACHE: Dict[str, Tuple[int, str]] = {}


def _card_text(spec: Any) -> str:
    """Lower-cased own agent-card text (cached per spec state)."""
    parts: List[str] = []
    for fld in ("assumptions", "pre_conditions", "not_recommended_when"):
        parts.extend(str(x) for x in getattr(spec, fld, []) or [])
    for fm in getattr(spec, "failure_modes", []) or []:
        parts.append(getattr(fm, "symptom", ""))
    key = sum(len(p) for p in parts) * 31 + len(parts)
    hit = _CARD_TEXT_CACHE.get(spec.name)
    if hit is not None and hit[0] == key:
        return hit[1]
    text = " ".join(parts).lower()
    _CARD_TEXT_CACHE[spec.name] = (key, text)
    return text


def _concept_bonus(terms: List[Tuple[str, Tuple[str, ...]]]) -> Dict[str, float]:
    bonus: Dict[str, float] = {}
    for word, _alts in terms:
        for rank, target in enumerate(CONCEPT_TARGETS.get(word, ())):
            b = max(3.0, 12.0 - 2.5 * rank) * CONCEPT_WEIGHT.get(word, 1.0)
            bonus[target] = max(bonus.get(target, 0.0), b)
    return bonus


def rank(
    registry: Dict[str, Any], query: str, *, parents: Optional[set] = None
) -> List[Dict[str, Any]]:
    """Ranked search hits over ``registry`` (``{name: FunctionSpec}``)."""
    terms = search_terms(query)
    if not terms:
        return []
    needed = max(1, (len(terms) + 1) // 2)
    query_lower = query.lower().strip()
    parents = parents or set()
    concept = _concept_bonus(terms)

    full: List[Tuple[float, int, Dict[str, Any]]] = []
    partial: List[Tuple[float, int, Dict[str, Any]]] = []
    for spec in registry.values():
        name = spec.name.lower()
        if spec.name[:1].isupper() and name != query_lower:
            # Result / exception classes are not callables an agent runs.
            continue
        name_words = set(name.replace("_", " ").split())
        desc = (spec.description or "").lower()
        tags = [t.lower() for t in (spec.tags or [])]
        tag_text = " ".join(tags)
        card = None
        score = 0.0
        hits = 0
        for word, alts in terms:
            term_score = 0.0
            for alt in alts:
                if alt == name:
                    term_score = max(term_score, 10.0)
                elif alt in name_words:
                    term_score = max(term_score, 6.0)
                elif len(alt) >= 2 and any(w.startswith(alt) for w in name_words):
                    term_score = max(term_score, 3.5)
                elif alt in name and len(alt) >= 4:
                    term_score = max(term_score, 3.0)
                if alt in tags:
                    term_score = max(term_score, 4.0)
                elif alt in tag_text:
                    term_score = max(term_score, 2.0)
                if alt in desc:
                    term_score = max(term_score, 1.0 + min(desc.count(alt), 3) * 0.25)
            if term_score > 0:
                hits += 1
                score += term_score
            elif len(word) >= 4:
                if card is None:
                    card = _card_text(spec)
                if word in card:
                    score += CARD_WEIGHT
        c_bonus = concept.get(spec.name, 0.0)
        if hits == 0 and c_bonus == 0.0:
            continue
        score += c_bonus
        if name == query_lower:
            score += 20.0
        score += hits * 2.0
        if spec.name in parents:
            score += 3.0
        if spec.validation_status == "certified":
            score += 1.0
        elif spec.validation_status == "validated":
            score += 0.5
        if name.startswith("dgp_"):
            score -= 3.0
        alias_of = getattr(spec, "alias_of", None)
        if alias_of:
            score -= ALIAS_PENALTY
        entry: Dict[str, Any] = {
            "name": spec.name,
            "description": spec.description,
            "category": spec.category,
            "stability": spec.stability,
            "validation_status": spec.validation_status,
        }
        if alias_of:
            entry["alias_of"] = alias_of
        row = (score, -len(spec.name), entry)
        if hits >= needed or c_bonus > 0.0:
            entry["match"] = "full"
            full.append(row)
        else:
            entry["match"] = "partial"
            partial.append(row)

    key = lambda x: (x[0], x[1])  # noqa: E731
    if full:
        full.sort(key=key, reverse=True)
        return [item for _, _, item in full]
    partial.sort(key=key, reverse=True)
    return [item for _, _, item in partial[:PARTIAL_LIMIT]]


__all__ = [
    "SYNONYMS",
    "PHRASES",
    "CONCEPT_TARGETS",
    "search_terms",
    "rank",
]
