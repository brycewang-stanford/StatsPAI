"""
Multilevel / mixed-effects models.

Exports
-------
mixed, MixedResult                 Linear mixed effects (Gaussian response)
meglm, melogit, mepoisson, MEGLMResult
                                   Generalised linear mixed models via
                                   Laplace approximation
icc                                Intra-class correlation with 95% CI
lrtest                             Likelihood-ratio test between two fitted
                                   mixed models (with chi-bar² boundary
                                   correction when variance components are
                                   being tested)
"""

from .lmm import MixedResult, mixed
from .glmm import MEGLMResult, meglm, melogit, mepoisson
from .diagnostics import icc
from .comparison import lrtest

__all__ = [
    "mixed",
    "MixedResult",
    "meglm",
    "melogit",
    "mepoisson",
    "MEGLMResult",
    "icc",
    "lrtest",
]
