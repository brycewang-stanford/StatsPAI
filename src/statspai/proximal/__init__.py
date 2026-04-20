"""
Proximal Causal Inference (Tchetgen Tchetgen et al. 2020).

Identifies the ATE in the presence of an *unmeasured* confounder :math:`U`,
using two proxies of :math:`U`:

* :math:`Z` — "treatment-inducing confounding proxy" (independent of Y | D, U)
* :math:`W` — "outcome-inducing confounding proxy" (independent of D | U)

plus measured covariates :math:`X`.
"""

from .p2sls import proximal, ProximalCausalInference

__all__ = ['proximal', 'ProximalCausalInference']
