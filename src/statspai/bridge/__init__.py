"""
Bridging theorems for causal inference (StatsPAI v0.10).

Each "bridging theorem" pairs two seemingly different estimators on the
same target parameter and proves that — under appropriate conditions —
they identify the same quantity. Reporting both estimates side-by-side
gives doubly-robust identification: if either path's assumption holds,
the estimate is consistent.

Six bridges shipped (per arXiv 2503.11375 / 2510.26723 / 2310.18563 v6 /
2404.09117 / 2411.02771 / 2202.07234, 2022-2025):

* ``did_sc``        — DiD ≡ Synthetic Control (Shi-Athey 2025)
* ``ewm_cate``      — EWM ≡ CATE → policy (Ferman et al. 2025)
* ``cb_ipw``        — Covariate Balancing ≡ IPW × DR (Zhao-Percival 2025)
* ``kink_rdd``      — Kink-Bunching ≡ RDD (Lu-Wang-Xie 2025)
* ``dr_calib``      — Doubly Robust via Calibration (Zhang 2025)
* ``surrogate_pci`` — Long-term Surrogate ≡ PCI (Imbens-Kallus-Mao-Wang 2025, JRSS-B)

The unified entry point is ``sp.bridge(data, kind=..., **kwargs)``,
returning a :class:`BridgeResult` reporting the two path estimates,
their agreement test, and the recommended doubly-robust point estimate.
"""

from .core import bridge, BridgeResult

__all__ = ['bridge', 'BridgeResult']
