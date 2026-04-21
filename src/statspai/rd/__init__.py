"""
Regression Discontinuity (RD) module for StatsPAI.

The most comprehensive Python RD toolkit, providing:

**Core estimation:**
- Sharp, Fuzzy, and Kink RD estimation with robust bias-corrected inference (CCT 2014)
- Covariate-adjusted local polynomial estimation (Calonico et al. 2019)
- Donut-hole RD for manipulation near cutoff
- Regression Discontinuity in Time (Hausman-Rapson 2018)
- Multi-cutoff and multi-score RD (Cattaneo et al. 2024)
- Boundary discontinuity / 2D RD designs (Cattaneo-Titiunik-Yu 2025)

**Bandwidth selection:**
- MSE-optimal: mserd, msetwo, msecomb1, msecomb2
- CER-optimal: cerrd, certwo, cercomb1, cercomb2 (Calonico-Cattaneo-Farrell 2020)
- Fuzzy-specific and covariate-adjusted bandwidth selection

**Inference:**
- Honest confidence intervals (Armstrong-Kolesar 2018, 2020)
- Local randomization inference with Fisher exact tests (Cattaneo-Titiunik-VB 2016)
- Window selection and sensitivity analysis
- Rosenbaum sensitivity bounds

**Treatment effect heterogeneity:**
- CATE estimation via fully interacted local linear (Calonico et al. 2025)
- ML + RD: Causal Forest, Gradient Boosting, LASSO-assisted RD

**External validity:**
- Extrapolation away from cutoff (Angrist-Rokkanen 2015)
- Multi-cutoff extrapolation (Cattaneo et al. 2024)
- External validity diagnostics

**Diagnostics & visualization:**
- One-click diagnostic dashboard (rdsummary)
- IMSE-optimal binned scatter with pointwise CI bands
- Density manipulation testing (CJM 2020)
- Bandwidth sensitivity, covariate balance, placebo cutoff tests
- Power analysis and sample size calculations
"""

from .rdrobust import rdrobust, rdplot, rdplotdensity
from .bandwidth import rdbwselect
from .diagnostics import rdbwsensitivity, rdbalance, rdplacebo, rdsummary
from .rkd import rkd
from .honest_ci import rd_honest
from .rdit import rdit
from .rdmulti import rdmc, rdms, RDMultiResult
from .rdpower import rdpower, rdsampsi, RDPowerResult, RDSampSiResult
from .locrand import rdrandinf, rdwinselect, rdsensitivity, rdrbounds
from .hte import rdhte, rdbwhte, rdhte_lincom
from .rd2d import rd2d, rd2d_bw, rd2d_plot
from .rdml import rd_forest, rd_boost, rd_lasso, rd_cate_summary
from .extrapolate import rd_extrapolate, rd_multi_extrapolate, rd_external_validity

# v0.10 RDD frontier
from .interference import rd_interference, RDInterferenceResult
from .multi_score import rd_multi_score, MultiScoreRDResult
from .distribution_valued import rd_distribution, DistRDResult
from .bayes_hte import rd_bayes_hte, BayesRDHTEResult
from .distributional_design import rd_distributional_design, DDDResult

# User-friendly aliases
from ._aliases import (
    multi_cutoff_rd, geographic_rd, boundary_rd, multi_score_rd,
)

__all__ = [
    'rdrobust',
    'rdplot',
    'rdplotdensity',
    'rdbwselect',
    'rdbwsensitivity',
    'rdbalance',
    'rdplacebo',
    'rdsummary',
    'rkd',
    'rd_honest',
    'rdit',
    'rdmc',
    'rdms',
    'RDMultiResult',
    'rdpower',
    'rdsampsi',
    'RDPowerResult',
    'RDSampSiResult',
    'rdrandinf',
    'rdwinselect',
    'rdsensitivity',
    'rdrbounds',
    'rdhte',
    'rdbwhte',
    'rdhte_lincom',
    'rd2d',
    'rd2d_bw',
    'rd2d_plot',
    'rd_forest',
    'rd_boost',
    'rd_lasso',
    'rd_cate_summary',
    'rd_extrapolate',
    'rd_multi_extrapolate',
    'rd_external_validity',
    'multi_cutoff_rd',
    'geographic_rd',
    'boundary_rd',
    'multi_score_rd',
]
