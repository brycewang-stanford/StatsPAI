"""Unified help system for StatsPAI.

Aggregates four help layers:
    1. Registry metadata (sp.describe_function, sp.search_functions)
    2. Python docstrings (help(sp.xxx))
    3. Smart recommender pointers (sp.recommend, sp.compare_estimators)
    4. Submodule method catalogs (sp.decomposition.available_methods)

Usage
-----
>>> import statspai as sp
>>> sp.help()                       # top-level overview
>>> sp.help('did')                  # function detail
>>> sp.help('causal')               # category listing
>>> sp.help('causal.did')           # scoped detail
>>> sp.help(sp.regress)             # docstring fallback
>>> sp.help(search='treatment')     # keyword search
>>> sp.help('did', format='dict')   # programmatic access
"""
from __future__ import annotations

import inspect
import textwrap
from typing import Any, Dict, List, Optional, Union


# ====================================================================== #
#  Category index — groups of related functions for hierarchical help.
# ====================================================================== #
#
# NOTE: kept in sync with registry categories, but additionally covers
# areas the registry does not yet hand-register.  Used by sp.help()
# category listings and by the auto-registration fallback.

CATEGORY_DESCRIPTIONS: Dict[str, str] = {
    "regression": "OLS / IV / GLM / quantile / Tobit / Heckman / count / survival",
    "causal": "DID, RD, Synth, Matching, DML, IPW, Causal Forest, Meta-learners, DAG, g-methods",
    "panel": "Fixed/random effects, dynamic panel GMM, HDFE, interactive FE, unit roots",
    "timeseries": "VAR, ARIMA, GARCH, local projections, cointegration, structural breaks",
    "spatial": "SAR/SEM/SDM/SAC, GWR, spatial panels, LISA, Moran's I, spatial weights",
    "survey": "svydesign, svymean/glm, raking, calibration",
    "survival": "Cox, Kaplan-Meier, AFT, frailty, logrank",
    "decomposition": "Oaxaca, RIF, DFL, Machado-Mata, Kitagawa, Shapley, Gelbach",
    "diagnostics": "Sensitivity (Oster, sensemakr, E-value), specification tests, heterogeneity",
    "robustness": "Spec curve, subgroup analysis, sensitivity dashboards",
    "inference": "Bootstrap, wild cluster, Romano-Wolf, conformal, CR2/CR3, Conley",
    "output": "outreg2, modelsummary, regtable, paper_tables, coefplot",
    "plots": "binscatter, event-study plots, theme management",
    "smart": "Recommender, compare_estimators, assumption_audit, pub_ready, workflow",
    "power": "Power analysis for RCT/DID/RD/IV/cluster-RCT and MDE",
    "experimental": "Randomization, balance checks, attrition, optimal design",
    "missing": "MICE, multiple imputation, mi_estimate",
    "bayesian": "bayes_did / bayes_rd / bayes_iv / bayes_mte / BCF / policy weights",
    "postestimation": "margins, contrast, pwcompare, test, lincom",
    "agent": "LLM tool-definition surface + JSON schema export",
    "utils": "Labels, winsor, DGP simulators, read_data, describe, pwcorr",
    "datasets": "Canonical datasets (Prop99, German reunification, CPS wage, ...)",
}

# Module-path-prefix → category.  First-match wins; order matters for
# overlapping prefixes (e.g. 'causal_discovery' before 'causal').
_MODULE_CATEGORY_PREFIXES: List[tuple] = [
    ("statspai.regression", "regression"),
    ("statspai.fixest", "panel"),
    ("statspai.panel", "panel"),
    ("statspai.gmm", "panel"),
    ("statspai.multilevel", "panel"),
    ("statspai.timeseries", "timeseries"),
    ("statspai.spatial", "spatial"),
    ("statspai.survey", "survey"),
    ("statspai.survival", "survival"),
    ("statspai.decomposition", "decomposition"),
    ("statspai.diagnostics", "diagnostics"),
    ("statspai.robustness", "robustness"),
    ("statspai.inference", "inference"),
    ("statspai.output", "output"),
    ("statspai.plots", "plots"),
    ("statspai.smart", "smart"),
    ("statspai.workflow", "smart"),
    ("statspai.power", "power"),
    ("statspai.experimental", "experimental"),
    ("statspai.imputation", "missing"),
    ("statspai.bayes", "bayesian"),
    ("statspai.postestimation", "postestimation"),
    ("statspai.agent", "agent"),
    ("statspai.utils", "utils"),
    ("statspai.datasets", "datasets"),
    # Causal inference families (broad fallback)
    ("statspai.did", "causal"),
    ("statspai.rd", "causal"),
    ("statspai.synth", "causal"),
    ("statspai.matching", "causal"),
    ("statspai.dml", "causal"),
    ("statspai.deepiv", "causal"),
    ("statspai.iv", "causal"),
    ("statspai.metalearners", "causal"),
    ("statspai.neural_causal", "causal"),
    ("statspai.tmle", "causal"),
    ("statspai.bcf", "causal"),
    ("statspai.policy_learning", "causal"),
    ("statspai.conformal_causal", "causal"),
    ("statspai.dose_response", "causal"),
    ("statspai.bounds", "causal"),
    ("statspai.interference", "causal"),
    ("statspai.dtr", "causal"),
    ("statspai.multi_treatment", "causal"),
    ("statspai.msm", "causal"),
    ("statspai.proximal", "causal"),
    ("statspai.principal_strat", "causal"),
    ("statspai.mediation", "causal"),
    ("statspai.causal_impact", "causal"),
    ("statspai.bartik", "causal"),
    ("statspai.causal_discovery", "causal"),
    ("statspai.matrix_completion", "causal"),
    ("statspai.bunching", "causal"),
    ("statspai.qte", "causal"),
    ("statspai.mendelian", "causal"),
    ("statspai.mht", "inference"),
    ("statspai.dag", "causal"),
    ("statspai.selection", "regression"),
    ("statspai.nonparametric", "nonparametric"),
    ("statspai.structural", "structural"),
    ("statspai.frontier", "frontier"),
    ("statspai.causal", "causal"),
    ("statspai.core", "core"),
]


def _infer_category(obj: Any) -> str:
    """Guess category from an object's __module__."""
    mod = getattr(obj, "__module__", "") or ""
    for prefix, cat in _MODULE_CATEGORY_PREFIXES:
        if mod.startswith(prefix):
            return cat
    return "other"


# ====================================================================== #
#  HelpResult — REPL-friendly output wrapper
# ====================================================================== #


class HelpResult:
    """Rich-ish help payload: prints as text, exposes data as dict.

    Returned by :func:`help` unless ``format='dict'`` is requested,
    in which case the underlying mapping is returned directly.
    """

    __slots__ = ("text", "data")

    def __init__(self, text: str, data: Optional[Dict[str, Any]] = None):
        self.text = text
        self.data = data or {}

    def __repr__(self) -> str:
        return self.text

    def __str__(self) -> str:
        return self.text

    def _repr_html_(self) -> str:
        # Minimal Jupyter rendering: wrap text in <pre> for monospace.
        import html
        return f"<pre>{html.escape(self.text)}</pre>"


# ====================================================================== #
#  Section builders
# ====================================================================== #

_TOP_BANNER = r"""
╔══════════════════════════════════════════════════════════════════════╗
║  StatsPAI — The Agent-Native Causal Inference Toolkit for Python     ║
║  Version {version}                                                         ║
╚══════════════════════════════════════════════════════════════════════╝
"""


def _top_overview() -> str:
    """Top-level overview when sp.help() is called with no argument."""
    # Local import to avoid circular during module load.
    import statspai as sp

    lines: List[str] = []
    lines.append(_TOP_BANNER.format(version=sp.__version__).rstrip())
    lines.append("")
    lines.append("QUICK START")
    lines.append("-----------")
    lines.append("  >>> import statspai as sp")
    lines.append("  >>> result = sp.regress('y ~ x1 + x2', data=df)")
    lines.append("  >>> result = sp.did(df, y='wage', treat='treated', time='post')")
    lines.append("  >>> cf = sp.causal_forest('y ~ treat | x1 + x2', data=df)")
    lines.append("")

    lines.append("CATEGORIES  (use sp.help('<category>') for details)")
    lines.append("----------")
    from .registry import _ensure_full_registry  # noqa: WPS433
    _ensure_full_registry()
    cats = _categories_with_counts()
    width = max(len(c) for c in cats) if cats else 12
    for cat, count in sorted(cats.items()):
        desc = CATEGORY_DESCRIPTIONS.get(cat, "")
        lines.append(f"  {cat:<{width}}  ({count:>3} fns)  {desc}")
    lines.append("")

    lines.append("HELP ENTRY POINTS")
    lines.append("-----------------")
    lines.append("  sp.help()                      — this overview")
    lines.append("  sp.help('<name>')              — function detail (e.g. sp.help('did'))")
    lines.append("  sp.help('<category>')          — list functions in a category")
    lines.append("  sp.help('<category>.<name>')   — scoped function detail")
    lines.append("  sp.help(sp.regress)            — Python docstring for a callable")
    lines.append("  sp.help(search='treatment')    — keyword search")
    lines.append("  sp.help('<name>', verbose=True)— include full param schema")
    lines.append("  sp.help('<name>', format='dict')— return dict for programmatic use")
    lines.append("")

    lines.append("OTHER HELP LAYERS")
    lines.append("-----------------")
    lines.append("  sp.list_functions(category=None)   Machine-readable function index")
    lines.append("  sp.describe_function(name)         Full metadata for one function")
    lines.append("  sp.search_functions(query)         Keyword search (AND logic)")
    lines.append("  sp.function_schema(name)           OpenAI/Anthropic tool schema")
    lines.append("  sp.all_schemas()                   Bulk schema export for LLM agents")
    lines.append("  sp.recommend(data=df, y=..., treat=...)  Method advisor")
    lines.append("  from statspai.decomposition import available_methods")
    lines.append("                                     Sub-module method catalogs")
    lines.append("")
    lines.append("CLI")
    lines.append("---")
    lines.append("  $ statspai list [--category <cat>]")
    lines.append("  $ statspai describe <name>")
    lines.append("  $ statspai search <query>")
    lines.append("  $ statspai help [<name>]")
    lines.append("")
    return "\n".join(lines)


def _categories_with_counts() -> Dict[str, int]:
    """Count functions per category in the (expanded) registry."""
    from .registry import _REGISTRY  # noqa: WPS433

    counts: Dict[str, int] = {}
    for spec in _REGISTRY.values():
        counts[spec.category] = counts.get(spec.category, 0) + 1
    return counts


def _category_listing(category: str) -> Optional[str]:
    """List functions in a category. Returns None if category unknown."""
    from .registry import list_functions, _REGISTRY, _ensure_full_registry  # noqa: WPS433
    _ensure_full_registry()

    names = list_functions(category=category)
    if not names:
        return None

    lines: List[str] = []
    header = f"Category: {category}"
    lines.append(header)
    lines.append("=" * len(header))
    desc = CATEGORY_DESCRIPTIONS.get(category)
    if desc:
        lines.append(desc)
    lines.append("")
    lines.append(f"{len(names)} function(s):")
    lines.append("")

    width = max(len(n) for n in names)
    for name in sorted(names):
        spec = _REGISTRY[name]
        short = spec.description.split(".")[0][:80]
        lines.append(f"  {name:<{width}}  {short}")

    lines.append("")
    lines.append(f"Use sp.help('{category}.<name>') or sp.help('<name>') for details.")
    return "\n".join(lines)


def _function_detail(name: str, verbose: bool = False) -> Optional[str]:
    """Render function detail from registry. Returns None if unknown."""
    from .registry import _REGISTRY, _ensure_full_registry  # noqa: WPS433
    _ensure_full_registry()

    if name not in _REGISTRY:
        return None

    spec = _REGISTRY[name]
    lines: List[str] = []
    header = f"sp.{spec.name}  —  [{spec.category}]"
    lines.append(header)
    lines.append("=" * len(header))
    lines.append(textwrap.fill(spec.description, width=78))
    lines.append("")

    if spec.params:
        lines.append("Parameters")
        lines.append("----------")
        for p in spec.params:
            req = "required" if p.required else f"default={p.default!r}"
            enum = f"  ∈ {p.enum}" if p.enum else ""
            lines.append(f"  {p.name} : {p.type}  ({req}){enum}")
            if p.description:
                lines.append(f"      {p.description}")
        lines.append("")

    if spec.returns:
        lines.append(f"Returns : {spec.returns}")
        lines.append("")

    if spec.example:
        lines.append("Example")
        lines.append("-------")
        for ln in spec.example.splitlines():
            lines.append(f"  {ln}")
        lines.append("")

    if spec.tags:
        lines.append(f"Tags      : {', '.join(spec.tags)}")
    if spec.reference:
        lines.append(f"Reference : {spec.reference}")

    if verbose:
        # Append live docstring from the Python object when available.
        import statspai as sp
        obj = getattr(sp, name, None)
        if obj is not None and getattr(obj, "__doc__", None):
            lines.append("")
            lines.append("Docstring")
            lines.append("---------")
            lines.append(textwrap.dedent(obj.__doc__).strip())

    return "\n".join(lines)


def _search_results(query: str) -> str:
    """Render keyword-search hits."""
    from .registry import search_functions, _ensure_full_registry  # noqa: WPS433
    _ensure_full_registry()

    hits = search_functions(query)
    lines: List[str] = []
    header = f"Search: {query!r}  →  {len(hits)} match(es)"
    lines.append(header)
    lines.append("=" * len(header))
    if not hits:
        lines.append("(no matches — try sp.search_functions(query) with different words)")
        return "\n".join(lines)

    width = max(len(h["name"]) for h in hits)
    for h in hits[:50]:
        short = h["description"].split(".")[0][:80]
        lines.append(f"  {h['name']:<{width}}  [{h['category']}]  {short}")
    if len(hits) > 50:
        lines.append(f"  ... and {len(hits) - 50} more. Use sp.search_functions(query).")
    return "\n".join(lines)


def _callable_docstring(obj: Any) -> str:
    """Fallback: render a callable's own docstring + signature."""
    name = getattr(obj, "__name__", repr(obj))
    lines: List[str] = []
    try:
        sig = str(inspect.signature(obj))
    except (TypeError, ValueError):
        sig = "(?)"
    header = f"{name}{sig}"
    lines.append(header)
    lines.append("=" * min(len(header), 78))
    doc = inspect.getdoc(obj)
    if doc:
        lines.append(doc)
    else:
        lines.append("(no docstring)")
    return "\n".join(lines)


# ====================================================================== #
#  Public API
# ====================================================================== #

def help(  # noqa: A001 (shadow builtin by design; sp.help is our public API)
    topic: Union[str, Any, None] = None,
    *,
    search: Optional[str] = None,
    verbose: bool = False,
    format: str = "text",  # noqa: A002
) -> Union[HelpResult, Dict[str, Any], None]:
    """Unified help entry point for StatsPAI.

    Parameters
    ----------
    topic : str or callable, optional
        - ``None`` → top-level overview.
        - ``'<function_name>'`` → function detail (from registry + docstring).
        - ``'<category>'`` → list functions in category.
        - ``'<category>.<function_name>'`` → scoped function detail.
        - any callable (e.g. ``sp.regress``) → docstring fallback.
    search : str, optional
        Keyword search across names, descriptions, and tags.
    verbose : bool, default False
        If True, append live docstring after registry metadata.
    format : {'text', 'dict'}, default 'text'
        ``'text'`` returns a ``HelpResult`` that prints nicely.
        ``'dict'`` returns the underlying data mapping, suitable
        for programmatic consumers.

    Returns
    -------
    HelpResult | dict | None
    """
    if format not in ("text", "dict"):
        raise ValueError(f"format must be 'text' or 'dict', got {format!r}")

    if search is not None:
        text = _search_results(search)
        if format == "dict":
            from .registry import search_functions
            return {"query": search, "results": search_functions(search)}
        return HelpResult(text, data={"kind": "search", "query": search})

    # No topic → top-level overview
    if topic is None:
        text = _top_overview()
        if format == "dict":
            return {
                "kind": "overview",
                "version": __import__("statspai").__version__,
                "categories": _categories_with_counts(),
            }
        return HelpResult(text, data={"kind": "overview"})

    # Callable → docstring fallback
    if callable(topic) and not isinstance(topic, str):
        text = _callable_docstring(topic)
        name = getattr(topic, "__name__", None)
        # Prefer registry detail when available.
        if name is not None:
            detail = _function_detail(name, verbose=verbose)
            if detail is not None:
                text = detail
                if verbose:
                    text += "\n\n" + _callable_docstring(topic)
        if format == "dict":
            from .registry import _REGISTRY, _ensure_full_registry
            _ensure_full_registry()
            spec = _REGISTRY.get(name) if name else None
            return spec.to_dict() if spec else {"kind": "docstring", "name": name, "text": text}
        return HelpResult(text, data={"kind": "callable", "name": name})

    # String topic: category / function / scoped name
    if isinstance(topic, str):
        t = topic.strip()
        # scoped: 'causal.did'
        if "." in t:
            _cat, _, fn = t.partition(".")
            detail = _function_detail(fn, verbose=verbose)
            if detail is not None:
                if format == "dict":
                    from .registry import _REGISTRY
                    return _REGISTRY[fn].to_dict()
                return HelpResult(detail, data={"kind": "function", "name": fn})
            # scoped but unknown → fall through to suggestion

        # function name?
        detail = _function_detail(t, verbose=verbose)
        if detail is not None:
            if format == "dict":
                from .registry import _REGISTRY
                return _REGISTRY[t].to_dict()
            return HelpResult(detail, data={"kind": "function", "name": t})

        # category name?
        listing = _category_listing(t)
        if listing is not None:
            if format == "dict":
                from .registry import list_functions
                return {"kind": "category", "category": t, "functions": list_functions(t)}
            return HelpResult(listing, data={"kind": "category", "category": t})

        # Unknown — suggest nearest matches
        from difflib import get_close_matches
        from .registry import list_functions, _ensure_full_registry
        _ensure_full_registry()
        all_names = list_functions()
        suggestions = get_close_matches(t, all_names, n=5, cutoff=0.6)
        cats = list(CATEGORY_DESCRIPTIONS.keys())
        cat_suggestions = get_close_matches(t, cats, n=3, cutoff=0.6)

        lines = [f"No match for {topic!r}."]
        if suggestions:
            lines.append(f"  Did you mean: {', '.join(suggestions)}?")
        if cat_suggestions:
            lines.append(f"  Or category:  {', '.join(cat_suggestions)}?")
        lines.append("")
        lines.append("Try sp.help() for the top-level overview, "
                     "or sp.help(search='<keyword>') for keyword search.")
        text = "\n".join(lines)
        if format == "dict":
            return {
                "kind": "not_found",
                "query": topic,
                "function_suggestions": suggestions,
                "category_suggestions": cat_suggestions,
            }
        return HelpResult(text, data={"kind": "not_found", "query": topic})

    raise TypeError(
        f"sp.help() topic must be str, callable, or None; got {type(topic).__name__}"
    )
