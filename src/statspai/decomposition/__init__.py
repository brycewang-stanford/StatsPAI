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

# Existing (backward-compatible) imports
from .oaxaca import oaxaca, gelbach, OaxacaResult, GelbachResult
from .rif import (
    rifreg,
    rif_decomposition,
    rif_values,
    RIFResult,
    RIFDecompositionResult,
)

# New tier-C imports
from .dfl import dfl_decompose, DFLResult
from .ffl import ffl_decompose, FFLResult
from .machado_mata import machado_mata, MachadoMataResult
from .melly import melly_decompose, MellyResult
from .cfm import cfm_decompose, CFMResult
from .nonlinear import (
    fairlie,
    bauer_sinning,
    yun_nonlinear,
    NonlinearDecompResult,
)
from .inequality import (
    inequality_index,
    subgroup_decompose,
    source_decompose,
    shapley_inequality,
    SubgroupDecompResult,
    SourceDecompResult,
    ShapleyInequalityResult,
)
from .kitagawa import (
    kitagawa_decompose,
    das_gupta,
    KitagawaResult,
    DasGuptaResult,
)
from .causal import (
    gap_closing,
    mediation_decompose,
    disparity_decompose,
    GapClosingResult,
    MediationDecompResult,
    DisparityDecompResult,
)
from .yu_elwert import yu_elwert_decompose, YuElwertResult

# Unified dispatcher
from .dispatcher import decompose, available_methods

# Plots and datasets — imported for their registration side effect only.
from . import plots as _plots_module  # noqa: F401
from . import datasets as _datasets_module  # noqa: F401

# Convenience exports
from .datasets import (
    cps_wage,
    chilean_households,
    mincer_wage_panel,
    disparity_panel,
)

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
