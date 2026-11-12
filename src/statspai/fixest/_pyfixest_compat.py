"""Read a pyfixest fit across the 0.60 to 0.70 attribute move.

pyfixest 0.70.0 (released 2026-11-08) moved a fit's state out of private
attributes (``fit._X``, ``fit._df_k``, ``fit._is_iv`` ...) into typed
sub-objects (``fit.within_data``, ``fit.variance_covariance``, ``fit.model``
...) and removed the old names outright: reading one raises
``AttributeError``. Every ``getattr(fit, "_X", None)`` in StatsPAI then
returned the default, so the optional pieces built on them (IV first-stage
F, Sargan and Wu-Hausman; the fixed-effect-aware residual df; the within
design for the SE menu) disappeared without an error.

pip still resolves 0.60 on Python 3.10 (0.70 needs 3.11), so both layouts
are in use. Reads of a removed private name go through :func:`pf_get`,
which tries the 0.70 location first and falls back to the 0.60 attribute.
Names pyfixest 0.70 kept as silent aliases are left as they are.
"""

from __future__ import annotations

import dataclasses
from typing import Any, Dict, Tuple

# old private name -> path of its 0.70 replacement (pyfixest's own
# deprecation registry, ``estimation/models/_deprecated_attributes.py``).
_PATHS: Dict[str, Tuple[str, ...]] = {
    "_X": ("within_data", "design"),
    "_Y": ("within_data", "response"),
    "_Z": ("within_data", "instruments"),
    "_df_k": ("variance_covariance", "df_k"),
    "_vcov": ("variance_covariance", "vcov"),
    "_is_iv": ("model", "is_iv"),
    "_f_stat_1st_stage": ("first_stage", "diagnostics", "f_stat"),
    "_p_value_1st_stage": ("first_stage", "diagnostics", "p_value"),
    "_vcov_type_detail": ("variance_covariance", "spec", "vcov_type_detail"),
    "_weights_name": ("options", "weights"),
}

_MISSING = object()


def pf_get(fit: Any, name: str, default: Any = None) -> Any:
    """``getattr(fit, name, default)`` that also finds the 0.70 location."""
    path = _PATHS.get(name)
    if path is not None:
        obj: Any = fit
        for part in path:
            obj = getattr(obj, part, _MISSING)
            if obj is _MISSING:
                break
        if obj is not _MISSING:
            return obj
    return getattr(fit, name, default)


def pf_set_vcov(fit: Any, vcov: Any) -> None:
    """Replace a fit's coefficient covariance and recompute its inference.

    0.70 keeps the covariance in a frozen ``VarianceCovariance`` dataclass
    and makes ``fit._vcov`` read-only; 0.60 has only ``fit._vcov``.
    """
    cov = getattr(fit, "variance_covariance", None)
    if cov is not None and dataclasses.is_dataclass(cov):
        fit.variance_covariance = dataclasses.replace(cov, vcov=vcov)
    else:
        fit._vcov = vcov
    fit.get_inference()
