"""
Proximal Causal Inference (Tchetgen Tchetgen et al. 2020).

Identifies the ATE in the presence of an *unmeasured* confounder :math:`U`,
using two proxies of :math:`U`:

* :math:`Z` — "treatment-inducing confounding proxy" (independent of Y | D, U)
* :math:`W` — "outcome-inducing confounding proxy" (independent of D | U)

plus measured covariates :math:`X`.
"""

from .p2sls import proximal, ProximalCausalInference
from .negative_controls import (
    negative_control_outcome,
    negative_control_exposure,
    double_negative_control,
    NegativeControlResult,
)
from .pci_regression import proximal_regression, ProximalRegResult

# v0.10 PCI frontier
from .fortified import fortified_pci
from .bidirectional import bidirectional_pci
from .mtp import pci_mtp
from .proxy_selector import select_pci_proxies, ProxyScoreResult

__all__ = [
    'proximal',
    'ProximalCausalInference',
    'negative_control_outcome',
    'negative_control_exposure',
    'double_negative_control',
    'NegativeControlResult',
    'proximal_regression',
    'ProximalRegResult',
    'fortified_pci',
    'bidirectional_pci',
    'pci_mtp',
    'select_pci_proxies',
    'ProxyScoreResult',
]
