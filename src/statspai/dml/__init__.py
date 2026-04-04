"""
Double/Debiased Machine Learning module for StatsPAI.

Implements the Chernozhukov et al. (2018) framework for causal inference
using machine learning first-stage estimators with cross-fitting.

References
----------
Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C.,
Newey, W., and Robins, J. (2018). "Double/Debiased Machine Learning for
Treatment and Structural Parameters." *Econometrics Journal*, 21(1), C1-C68.
"""

from .double_ml import dml, DoubleML

__all__ = ['dml', 'DoubleML']
