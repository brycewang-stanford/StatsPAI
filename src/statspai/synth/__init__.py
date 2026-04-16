"""
Synthetic Control module for StatsPAI.

Unified entry point: ``synth(method=...)`` dispatches to all variants.

Variants (13 methods)
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

Inference
---------
- **placebo** — in-space permutation (default)
- **conformal** — Chernozhukov, Wüthrich & Zhu (2021)
- **bootstrap / jackknife** — for SDID
- **prediction intervals** — Cattaneo et al. (2021)

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

__all__ = [
    # Unified entry point
    'synth',
    'SyntheticControl',
    # Variant shortcuts
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
    # SDID framework
    'sdid',
    'synthdid_estimate',
    'sc_estimate',
    'did_estimate',
    'synthdid_placebo',
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
]
