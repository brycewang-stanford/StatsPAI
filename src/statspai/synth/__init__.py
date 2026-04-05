"""
Synthetic Control module for StatsPAI.

Unified entry point: ``synth(method=...)`` dispatches to all variants.

Variants
--------
- **classic** — Abadie, Diamond & Hainmueller (2010)
- **penalized / ridge** — Ridge-penalised SCM
- **demeaned / detrended** — Ferman & Pinto (2021)
- **unconstrained / elastic_net** — Doudchenko & Imbens (2016)
- **augmented / ascm** — Ben-Michael, Feller & Rothstein (2021)
- **sdid** — Arkhangelsky, Athey, Hirshberg, Imbens & Wager (2021)
- **factor / gsynth** — Xu (2017)
- **staggered** — Ben-Michael, Feller & Rothstein (2022)

Inference
---------
- **placebo** — in-space permutation (default)
- **conformal** — Chernozhukov, Wüthrich & Zhu (2021)
- **bootstrap / jackknife** — for SDID
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
    # SDID framework
    'sdid',
    'synthdid_estimate',
    'sc_estimate',
    'did_estimate',
    'synthdid_placebo',
    # Plots
    'synthplot',
    'synthdid_plot',
    'synthdid_units_plot',
    'synthdid_rmse_plot',
    # Data
    'california_prop99',
]
