"""
Synthetic Control Method (SCM) module for StatsPAI.

Provides:
- Classic Abadie-Diamond-Hainmueller SCM
- Penalized (ridge) SCM for better pre-treatment fit
- Placebo (permutation) inference
- Gap plots and weight tables

References
----------
Abadie, A. and Gardeazabal, J. (2003).
"The Economic Costs of Conflict: A Case Study of the Basque Country."
*American Economic Review*, 93(1), 113-132.

Abadie, A., Diamond, A., and Hainmueller, J. (2010).
"Synthetic Control Methods for Comparative Case Studies."
*Journal of the American Statistical Association*, 105(490), 493-505.
"""

from .scm import synth, SyntheticControl

__all__ = ['synth', 'SyntheticControl']
