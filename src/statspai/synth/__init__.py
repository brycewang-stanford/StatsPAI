"""
Synthetic Control & Synthetic DID module for StatsPAI.

Provides:
- Classic Abadie-Diamond-Hainmueller SCM
- Penalized (ridge) SCM for better pre-treatment fit
- Synthetic DID, SC, and DID via the ``synthdid`` framework
- Placebo, bootstrap, and jackknife inference
- ``synthdid``-style plots (trajectory, units, RMSE)
- California Proposition 99 example dataset

References
----------
Abadie, A., Diamond, A., and Hainmueller, J. (2010).
"Synthetic Control Methods for Comparative Case Studies."
*JASA*, 105(490), 493-505.

Arkhangelsky, D., Athey, S., Hirshberg, D.A., Imbens, G.W.
and Wager, S. (2021).
"Synthetic Difference-in-Differences."
*American Economic Review*, 111(12), 4088-4118.
"""

from .scm import synth, SyntheticControl
from .augsynth import augsynth
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
    # SCM
    'synth',
    'SyntheticControl',
    # SDID framework
    'sdid',
    'synthdid_estimate',
    'sc_estimate',
    'did_estimate',
    'synthdid_placebo',
    # Plots
    'synthdid_plot',
    'synthdid_units_plot',
    'synthdid_rmse_plot',
    # Augmented SCM
    'augsynth',
    # Data
    'california_prop99',
]
