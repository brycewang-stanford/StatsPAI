"""
Decomposition Analysis module for StatsPAI.

Decomposition toolkit covering mean, distributional, inequality,
demographic, and causal decomposition methods under a
unified API: ``sp.decompose(method=...)``.

Methods (19 in total — Yu-Elwert added in v1.15)
-----------------------------------------------

**Mean decomposition**
- ``oaxaca`` — Blinder-Oaxaca (Blinder 1973; Oaxaca 1973) with 5
  reference-coefficient choices (A, B, pooled/Neumark, Cotton, Reimers)
- ``gelbach`` — Gelbach (2016) sequential orthogonal decomposition of
  omitted-variable bias
- ``fairlie`` — Fairlie (2005) nonlinear decomposition for logit/probit
- ``bauer_sinning`` / ``yun_nonlinear`` — Bauer-Sinning (2008) + Yun
  (2005) detailed nonlinear decomposition

**Distributional decomposition**
- ``rif`` — Recentered Influence Function regression + OB decomposition
  (Firpo-Fortin-Lemieux 2009)
- ``ffl`` — Firpo-Fortin-Lemieux (2018) two-step detailed decomposition
- ``dfl`` — DiNardo-Fortin-Lemieux (1996) reweighting
- ``machado_mata`` — Machado-Mata (2005) quantile decomposition
- ``melly`` — Melly (2005) analytical quantile decomposition
- ``cfm`` — Chernozhukov-Fernández-Val-Melly (2013) counterfactual
  distributions via distribution regression

**Inequality decomposition**
- ``subgroup`` — between/within decomposition (Theil T/L, GE, Gini,
  Atkinson, CV²)
- ``shapley_inequality`` — Shorrocks-Shapley (2013) allocation of
  inequality to covariates
- ``gini_source`` — Lerman-Yitzhaki (1985) Gini source decomposition

**Demographic / standardisation**
- ``kitagawa`` — Kitagawa (1955) two-factor rate decomposition
- ``das_gupta`` — Das Gupta (1993) multi-factor decomposition

**Causal decomposition**
- ``gap_closing`` — Lundberg (2022) gap-closing estimator
  (regression / IPW / AIPW)
- ``mediation`` — VanderWeele (2014) natural direct/indirect effects
- ``disparity`` / ``causal_jvw`` — Jackson-VanderWeele (2018) causal
  disparity decomposition
- ``yu_elwert`` — Yu & Elwert (2025) nonparametric causal decomposition
  of group disparities into baseline, prevalence, effect, and selection
  components (efficient-influence-function-based; ML-friendly)

Unified Entry
-------------
``sp.decompose(method=..., **kwargs)`` dispatches to any of the above.

Polish (v1.15)
--------------
Every result class now inherits ``DecompResultMixin``, exposing a
common ``.confint()``, ``.cite()``, ``.to_dict()``, ``.to_json()``,
``.to_excel()``, and ``.to_word()`` surface in addition to each
method's bespoke ``.summary()`` / ``.plot()`` / ``.to_latex()``.
Plots share a common palette and minimalist style via
:mod:`statspai.decomposition.plots` (forest plots, mediation forest,
Yu-Elwert mechanism plot, RIF heatmap, …).
"""

# Plots and datasets — imported for their registration side effect only.
from . import datasets as _datasets_module  # noqa: F401
from . import plots as _plots_module  # noqa: F401
from .causal import (
    DisparityDecompResult,
    GapClosingResult,
    MediationDecompResult,
    disparity_decompose,
    gap_closing,
    mediation_decompose,
)
from .cfm import CFMResult, cfm_decompose

# Convenience exports
from .datasets import chilean_households, cps_wage, disparity_panel, mincer_wage_panel

# New tier-C imports
from .dfl import DFLResult, dfl_decompose

# Unified dispatcher
from .dispatcher import available_methods, decompose
from .diversity import diversity_index
from .ffl import FFLResult, ffl_decompose
from .inequality import (
    ShapleyInequalityResult,
    SourceDecompResult,
    SubgroupDecompResult,
    inequality_index,
    shapley_inequality,
    source_decompose,
    subgroup_decompose,
)
from .kitagawa import DasGuptaResult, KitagawaResult, das_gupta, kitagawa_decompose
from .machado_mata import MachadoMataResult, machado_mata
from .melly import MellyResult, melly_decompose
from .nonlinear import NonlinearDecompResult, bauer_sinning, fairlie, yun_nonlinear

# Existing (backward-compatible) imports
from .oaxaca import GelbachResult, OaxacaResult, gelbach, oaxaca
from .rif import (
    RIFDecompositionResult,
    RIFResult,
    rif_decomposition,
    rif_values,
    rifreg,
)
from .yu_elwert import YuElwertResult, yu_elwert_decompose

__all__ = [
    # Existing (backward compat)
    "oaxaca",
    "gelbach",
    "OaxacaResult",
    "GelbachResult",
    "rifreg",
    "rif_decomposition",
    "rif_values",
    "RIFResult",
    "RIFDecompositionResult",
    # DFL
    "dfl_decompose",
    "DFLResult",
    # FFL
    "ffl_decompose",
    "FFLResult",
    # Quantile family
    "machado_mata",
    "MachadoMataResult",
    "melly_decompose",
    "MellyResult",
    "cfm_decompose",
    "CFMResult",
    # Nonlinear
    "fairlie",
    "bauer_sinning",
    "yun_nonlinear",
    "NonlinearDecompResult",
    # Inequality
    "diversity_index",
    "inequality_index",
    "subgroup_decompose",
    "source_decompose",
    "shapley_inequality",
    "SubgroupDecompResult",
    "SourceDecompResult",
    "ShapleyInequalityResult",
    # Kitagawa / Das Gupta
    "kitagawa_decompose",
    "das_gupta",
    "KitagawaResult",
    "DasGuptaResult",
    # Causal
    "gap_closing",
    "mediation_decompose",
    "disparity_decompose",
    "GapClosingResult",
    "MediationDecompResult",
    "DisparityDecompResult",
    "yu_elwert_decompose",
    "YuElwertResult",
    # Unified dispatcher
    "decompose",
    "available_methods",
    # Datasets
    "cps_wage",
    "chilean_households",
    "mincer_wage_panel",
    "disparity_panel",
]
