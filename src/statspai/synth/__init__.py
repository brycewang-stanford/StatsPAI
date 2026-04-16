"""
Synthetic Control module for StatsPAI.

Unified entry point: ``synth(method=...)`` dispatches to all variants.

Variants (20 methods)
---------------------
- **classic** — Abadie, Diamond & Hainmueller (2010)
- **penalized / ridge** — Ridge-penalised SCM
- **demeaned / detrended** — Ferman & Pinto (2021)
- **unconstrained / elastic_net** — Doudchenko & Imbens (2016)
- **augmented / ascm** — Ben-Michael, Feller & Rothstein (2021)
- **sdid** — Arkhangelsky, Athey, Hirshberg, Imbens & Wager (2021)
- **factor / gsynth** — Xu (2017)
- **staggered** — Ben-Michael, Feller & Rothstein (2022)
- **mc / matrix_completion** — Athey, Bayati et al. (2021)
- **discos / distributional** — Gunsilius (2023)
- **multi_outcome** — Sun (2023)
- **scpi / prediction_interval** — Cattaneo, Feng & Titiunik (2021)
- **bayesian** — Bayesian SCM with MCMC posterior (Vives & Martinez 2024)
- **bsts / causal_impact** — Bayesian Structural Time Series (Brodersen et al. 2015)
- **penscm / abadie_lhour** — Penalized SCM with pairwise discrepancy (Abadie & L'Hour 2021)
- **fdid / forward_did** — Forward DID with optimal donor selection (Li 2024)
- **cluster** — Cluster SCM with donor grouping (Rho 2024)
- **sparse / lasso** — Sparse SCM with L1 penalties (Amjad, Shah & Shen 2018)
- **kernel / kernel_ridge** — Kernel-based nonlinear SCM

Inference
---------
- **placebo** — in-space permutation (default)
- **conformal** — Chernozhukov, Wüthrich & Zhu (2021)
- **bootstrap / jackknife** — for SDID
- **prediction intervals** — Cattaneo et al. (2021)
- **bayesian posterior** — full posterior credible intervals (Bayesian SCM)
- **bsts posterior** — Bayesian structural time series uncertainty

Diagnostics
-----------
- **synth_sensitivity()** — comprehensive robustness suite
- **synth_loo()** — leave-one-out donor analysis
- **synth_time_placebo()** — backdating tests
- **synth_donor_sensitivity()** — donor pool variation
- **synth_rmspe_filter()** — pre-RMSPE robustness
"""

# Unified dispatcher + classic SCM
from .scm import synth, SyntheticControl

# Unified plotting (replaces old synthplot with full-variant support)
from .plots import synthplot

# Variant shortcuts
from .augsynth import augsynth
from .demeaned import demeaned_synth
from .robust import robust_synth
from .gsynth import gsynth
from .staggered import staggered_synth
from .conformal import conformal_synth
from .scpi import scpi, scest, scdata
from .mc import mc_synth
from .multi_outcome import multi_outcome_synth

# Distributional Synthetic Controls
from .discos import discos, qqsynth, discos_test, discos_plot, stochastic_dominance

# Bayesian SCM
from .bayesian import bayesian_synth

# BSTS / CausalImpact
from .bsts import causal_impact, bsts_synth

# Penalized SCM (Abadie & L'Hour 2021)
from .penscm import penalized_synth

# Forward DID
from .fdid import fdid

# Cluster SCM
from .cluster import cluster_synth

# Sparse SCM
from .sparse import sparse_synth

# Kernel / Nonlinear SCM
from .kernel import kernel_synth, kernel_ridge_synth

# Multi-method comparison & auto-recommendation
from .compare import synth_compare, synth_recommend, SynthComparison

# Power analysis & sample size planning
from .power import synth_power, synth_mde, synth_power_plot

# Report generator
from .report import synth_report, synth_report_to_file

# Sensitivity & robustness diagnostics
from .sensitivity import (
    synth_loo,
    synth_time_placebo,
    synth_donor_sensitivity,
    synth_rmspe_filter,
    synth_sensitivity,
    synth_sensitivity_plot,
)

# SDID framework
from .sdid import (
    sdid,
    synthdid_estimate,
    sc_estimate,
    did_estimate,
    synthdid_placebo,
    synthdid_plot,
    synthdid_units_plot,
    synthdid_rmse_plot,
    california_prop99,
)

# Canonical SCM datasets
from .datasets import (
    german_reunification,
    basque_terrorism,
    california_tobacco,
)

__all__ = [
    # Unified entry point
    'synth',
    'SyntheticControl',
    # Variant shortcuts (original 13)
    'demeaned_synth',
    'robust_synth',
    'gsynth',
    'staggered_synth',
    'conformal_synth',
    'augsynth',
    'mc_synth',
    'multi_outcome_synth',
    # Prediction Intervals (Cattaneo et al. 2021)
    'scpi',
    'scest',
    'scdata',
    # Distributional Synthetic Controls
    'discos',
    'qqsynth',
    'discos_test',
    'discos_plot',
    'stochastic_dominance',
    # New methods (7 additions)
    'bayesian_synth',
    'causal_impact',
    'bsts_synth',
    'penalized_synth',
    'fdid',
    'cluster_synth',
    'sparse_synth',
    'kernel_synth',
    'kernel_ridge_synth',
    # SDID framework
    'sdid',
    'synthdid_estimate',
    'sc_estimate',
    'did_estimate',
    'synthdid_placebo',
    # Multi-method comparison
    'synth_compare',
    'synth_recommend',
    'SynthComparison',
    # Report generator
    'synth_report',
    'synth_report_to_file',
    # Power analysis
    'synth_power',
    'synth_mde',
    'synth_power_plot',
    # Sensitivity & robustness
    'synth_loo',
    'synth_time_placebo',
    'synth_donor_sensitivity',
    'synth_rmspe_filter',
    'synth_sensitivity',
    'synth_sensitivity_plot',
    # Plots
    'synthplot',
    'synthdid_plot',
    'synthdid_units_plot',
    'synthdid_rmse_plot',
    # Data
    'california_prop99',
    'german_reunification',
    'basque_terrorism',
    'california_tobacco',
]
