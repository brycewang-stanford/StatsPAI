"""Closed choice lists the registry could not derive on its own.

:func:`statspai.registry._apply_schema_enrichment` fills ``enum`` from a
``Literal`` annotation, a dispatcher registry, or a docstring list that the
module source confirms. That leaves the selector arguments whose accepted
values are stated only by the code: a validation a few lines into the body, a
module constant, or a callee the argument is handed to unchanged. Without an
``enum`` an agent sees ``"Estimator or algorithm variant to use."`` and has to
guess the spelling.

Every entry here names *where the code states the set*, and
``tests/test_enum_seeds.py`` re-reads that place:

``DIRECT``
    The function validates the argument against these literals; each value
    must be spelled as a string literal in the function's own source.
``CONSTANT``
    The accepted set is a module-level constant next to the function; the
    seeded list must equal it.
``FORWARDED``
    The function passes the argument unchanged to another registered
    function (``target_param=param`` in its source); the target's ``enum``
    is reused.
``SAME_CALL``
    The name is a thin alias (``return target(*args, **kwargs)``) or the same
    object under a second name; every ``enum`` of the target is reused.
``KERNEL_USERS``
    The argument goes straight to ``rd._core._kernel_fn``; the kernels that
    function implements are the set.

Arguments left without an ``enum`` on purpose: grammars (``vce='cluster
firm'``, lavaan model text, ETS letter codes), model identifiers of external
services, and Python objects.
"""

from __future__ import annotations

from typing import Dict, List, Tuple

Key = Tuple[str, str]

DIRECT: Dict[Key, List[str]] = {
    ("structural_break", "method"): ["bai-perron", "global", "sup-f", "chow"],
    ("mr_ivw", "model"): ["default", "fixed", "random"],
    ("etregress", "method"): ["mle", "twostep"],
    ("rlasso_effect", "method"): ["partialling out", "double selection"],
    ("interactive_fe", "method"): ["iterative", "pca"],
    ("rd2d_bw", "method"): ["dpi", "rot"],
    ("zisf", "dist"): ["half-normal"],
    ("zisf", "vce"): ["oim", "opg", "robust"],
    ("lcsf", "dist"): ["half-normal"],
    ("lcsf", "vce"): ["oim", "opg", "robust"],
    ("causal_discovery", "method"): ["notears", "pc", "ges", "lingam", "fci"],
    ("survreg", "dist"): ["weibull", "exponential", "lognormal", "loglogistic"],
    ("partial_identification", "method"): ["manski", "lee", "horowitz_manski", "iv"],
    ("adjust_pvalues", "method"): [
        "bonferroni",
        "sidak",
        "holm",
        "hochberg",
        "hommel",
        "bh",
        "by",
    ],
}

#: ``(function, param) -> (module constant, values)``. A mapping constant
#: contributes its keys (``_FAMILIES``) or its distinct values
#: (``_KERNEL_ALIASES``, alias -> canonical name).
CONSTANT: Dict[Key, Tuple[str, List[str]]] = {
    ("rd2d_bw", "bwselect"): (
        "_BWSELECTS",
        [
            "mserd",
            "cerrd",
            "imserd",
            "icerrd",
            "msetwo",
            "certwo",
            "imsetwo",
            "icertwo",
        ],
    ),
    ("rd2d_bw", "vce"): ("_VCES", ["hc0", "hc1", "hc2", "hc3"]),
    ("meglm", "family"): (
        "_FAMILIES",
        ["gaussian", "binomial", "poisson", "gamma", "nbinomial"],
    ),
}

#: ``(function, param) -> (target function, target param)``.
FORWARDED: Dict[Key, Key] = {
    ("distributional_did", "estimator"): ("callaway_santanna", "estimator"),
    ("distributional_did", "control_group"): ("callaway_santanna", "control_group"),
    ("functional_form_test", "estimator"): ("callaway_santanna", "estimator"),
    ("functional_form_test", "control_group"): (
        "callaway_santanna",
        "control_group",
    ),
    ("honest_did_from_moments", "method"): ("honest_did", "method"),
    ("overlap_plot", "method"): ("propensity_score", "method"),
    ("ps_balance", "method"): ("propensity_score", "method"),
    ("geolift", "method"): ("synth", "method"),
    ("synth_report", "method"): ("synth", "method"),
    ("synth_report_to_file", "method"): ("synth_report", "method"),
    ("margins_table", "method"): ("margins", "method"),
    ("rlasso_effects", "method"): ("rlasso_effect", "method"),
    ("metafrontier", "dist"): ("frontier", "dist"),
    ("malmquist", "dist"): ("frontier", "dist"),
    ("rdbwsensitivity", "kernel"): ("rdrobust", "kernel"),
    ("rdbalance", "kernel"): ("rdrobust", "kernel"),
    ("rdplacebo", "kernel"): ("rdrobust", "kernel"),
    ("rdsummary", "kernel"): ("rdrobust", "kernel"),
    ("rdms", "kernel"): ("rdrobust", "kernel"),
}

#: ``alias -> target``: the same call under another name.
SAME_CALL: Dict[str, str] = {
    "bjs": "did_imputation",
    "borusyak_jaravel_spiess": "did_imputation",
    "boundary_rd": "rd2d",
    "geographic_rd": "rdms",
    "multi_score_rd": "rd_multi_score",
}

#: Functions whose ``kernel`` argument is evaluated by ``rd._core._kernel_fn``.
KERNEL_USERS: Tuple[str, ...] = (
    "rd_lasso",
    "rd_distribution",
    "rd_interference",
    "rd_multi_score",
    "rd_bayes_hte",
    "rd_distributional_design",
)
KERNEL_FN_CHOICES: List[str] = ["triangular", "uniform", "epanechnikov", "gaussian"]

__all__ = [
    "DIRECT",
    "CONSTANT",
    "FORWARDED",
    "SAME_CALL",
    "KERNEL_USERS",
    "KERNEL_FN_CHOICES",
]
