"""Design of experiments: building designs and analysing them.

``statspai.experimental`` covers the randomised controlled trial
(assignment, balance, attrition, sample size). This package covers the
design of the *runs* of an experiment or of a computer model: which
combinations of factor levels to try.

- ``factorial_design`` / ``design_aberration`` / ``factorial_effects``:
  full and fractional two-level factorials, their alias structure, and
  the analysis of unreplicated experiments.
- ``mixture_design``: factors that are shares summing to one.
- ``doe_optimal``: D-, A- and I-optimal designs for linear, nonlinear
  and generalised linear models.
- ``space_filling`` / ``design_augment`` / ``design_criteria``: Latin
  hypercube, maximum projection, maximin and uniform designs.
- ``support_points`` / ``split_data``: representative points of a
  distribution and train / test splits built from them.
- ``sobol_indices`` / ``morris_screening``: global sensitivity analysis
  of a model. ``factor_importance``: the same question asked of a data
  set, without a model.
- ``sequential_design``: Bayesian optimisation and active learning with
  a Gaussian process.
"""

from ._common import DesignResult
from .criteria import design_criteria
from .effects import FactorialEffectsResult, factorial_effects
from .factorial import FactorialDesignResult, design_aberration, factorial_design
from .importance import FactorImportanceResult, factor_importance
from .mixture import mixture_design
from .optimal import ModelDesignResult, doe_optimal
from .sensitivity import MorrisResult, SobolResult, morris_screening, sobol_indices
from .sequential import SequentialDesignResult, sequential_design
from .spacefill import design_augment, space_filling
from .support import DataSplitResult, SupportPointsResult, split_data, support_points

__all__ = [
    "space_filling",
    "design_augment",
    "design_criteria",
    "DesignResult",
    "factorial_design",
    "design_aberration",
    "FactorialDesignResult",
    "factorial_effects",
    "FactorialEffectsResult",
    "factor_importance",
    "FactorImportanceResult",
    "mixture_design",
    "doe_optimal",
    "ModelDesignResult",
    "sobol_indices",
    "SobolResult",
    "morris_screening",
    "MorrisResult",
    "support_points",
    "SupportPointsResult",
    "split_data",
    "DataSplitResult",
    "sequential_design",
    "SequentialDesignResult",
]
