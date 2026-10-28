"""Helpers for ``sp.recentered_iv``: the efficient instrument and Conley SEs.

Kept apart from ``recentered.py`` so that file stays readable; nothing
here is public.
"""

from __future__ import annotations

from typing import Any, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .._input_validation import check_columns
from ..exceptions import MethodIncompatibility

_CONLEY_KERNELS = ("uniform", "bartlett")
_CONLEY_DISTANCES = ("haversine", "planar")


def column_or_array(data: pd.DataFrame, value: Any, name: str) -> np.ndarray:
    """A length-``n`` float vector from a column name or an array."""
    if isinstance(value, str):
        check_columns(data, {name: value}, function="recentered_iv")
        raw = pd.to_numeric(data[value], errors="coerce")
    else:
        raw = value
    arr = np.asarray(raw, dtype=float).reshape(-1)
    if arr.size != len(data):
        raise MethodIncompatibility(
            f"{name} has {arr.size} values but data has {len(data)} rows",
            recovery_hint=f"Pass one {name} value per row of data.",
            diagnostics={f"len_{name}": int(arr.size)},
        )
    return arr


def check_efficient(
    method: str, resid_var: Any, resid_mean: Any, cov_kind: Optional[str] = None
) -> Tuple[bool, bool]:
    """Validate the efficient-instrument arguments; return which are set.

    ``cov_kind`` is ``'dense'`` when a full ``resid_cov`` matrix supplies
    the variances, in which case ``resid_mean`` needs no ``resid_var``.
    """
    has_var = resid_var is not None
    has_mean = resid_mean is not None
    if has_mean and not has_var and cov_kind != "dense":
        raise MethodIncompatibility(
            "resid_mean= needs resid_var=: the efficient instrument weighs "
            "the predictable part of the residual against its variance.",
            recovery_hint=(
                "Pass resid_var= (Var[eps_i | w]) as well; use a constant "
                "column for homoskedastic residuals."
            ),
        )
    if (has_var or has_mean or cov_kind is not None) and method != "recenter":
        raise MethodIncompatibility(
            "resid_var= / resid_mean= / resid_cov= build the efficient "
            "recentered instrument, which is defined for method='recenter' "
            "only.",
            recovery_hint=(
                "Use method='recenter', or drop resid_var / resid_mean / " "resid_cov."
            ),
            diagnostics={"method": method},
        )
    return has_var, has_mean


def efficient_transform(
    V: np.ndarray, omega: np.ndarray, psi: Optional[np.ndarray]
) -> Tuple[np.ndarray, Dict[str, float]]:
    """Apply ``E[ee' | w]^{-1} = (Omega + psi psi')^{-1}`` to the columns of V.

    ``Omega = diag(omega)``. By Sherman-Morrison the product is
    ``Omega^{-1} (v - nu rho psi)`` with
    ``rho = psi' Omega^{-1} v / psi' Omega^{-1} psi`` and
    ``nu = psi' Omega^{-1} psi / (1 + psi' Omega^{-1} psi)``, the form of
    Lemma 3 of Borusyak and Hull, NBER WP 27845 (rev. 2021), eq. 10. The map is
    linear and depends on ``w`` only, so the transformed instrument keeps
    mean zero over the shock draws.
    """
    V = np.asarray(V, dtype=float)
    vec = V.ndim == 1
    M = V.reshape(len(omega), -1)
    inv = 1.0 / omega
    out = inv[:, None] * M
    info: Dict[str, float] = {}
    if psi is not None:
        q = float(np.sum(psi * inv * psi))
        nu = q / (1.0 + q)
        out = out - np.outer(inv * psi, (psi * inv) @ M) / (1.0 + q)
        info["nu"] = nu
        info["psi_omega_psi"] = q
    return (out.ravel() if vec else out), info


def check_conley(
    vce: str,
    lat: Optional[str],
    lon: Optional[str],
    cutoff: Optional[float],
    kernel: str,
    distance: str,
    cluster: Optional[str],
    weights: Optional[str],
) -> None:
    """Validate ``vce='conley'`` and its companion arguments."""
    if vce != "conley":
        if lat is not None or lon is not None or cutoff is not None:
            raise MethodIncompatibility(
                "conley_lat / conley_lon / conley_cutoff are only used with "
                "vce='conley'",
                recovery_hint="Pass vce='conley', or drop the conley_* arguments.",
            )
        return
    if lat is None or lon is None or cutoff is None:
        raise MethodIncompatibility(
            "vce='conley' requires conley_lat=, conley_lon= and conley_cutoff=",
            recovery_hint=(
                "Name the latitude and longitude columns (degrees) and give "
                "the distance cutoff in km."
            ),
        )
    if not np.isfinite(cutoff) or cutoff <= 0:
        raise MethodIncompatibility(
            f"conley_cutoff must be a positive distance in km, got {cutoff!r}",
            recovery_hint="Pass for example conley_cutoff=500.",
        )
    if kernel not in _CONLEY_KERNELS:
        raise MethodIncompatibility(
            f"conley_kernel must be one of {_CONLEY_KERNELS}, got {kernel!r}",
            recovery_hint="Use 'uniform' or 'bartlett' (linearly decaying).",
        )
    if distance not in _CONLEY_DISTANCES:
        raise MethodIncompatibility(
            f"conley_distance must be one of {_CONLEY_DISTANCES}, got " f"{distance!r}",
            recovery_hint="Use 'haversine', or 'planar' for Stata acreg's metric.",
        )
    if cluster is not None:
        raise MethodIncompatibility(
            "vce='conley' and cluster= are alternative variances",
            recovery_hint="Drop cluster=, or use vce='robust' with cluster=.",
        )
    if weights is not None:
        raise MethodIncompatibility(
            "vce='conley' is not available with weights=: the Conley variance "
            "of weighted 2SLS is not implemented (sp.conley refuses it).",
            recovery_hint="Drop weights=, or use vce='robust' / cluster=.",
        )


def singleton_keep_mask(fe_frame: pd.DataFrame) -> np.ndarray:
    """Rows ``sp.iv(absorb=)`` keeps after dropping singletons (same rule)."""
    from ..fast.demean import demean

    _, info = demean(
        np.ones((len(fe_frame), 1)), fe_frame, drop_singletons=True, tol=1e-12
    )
    return np.asarray(info.keep_mask)


def conley_refit(
    fit: Any,
    coords: pd.DataFrame,
    cutoff: float,
    kernel: str,
    distance: str,
    alpha: float,
) -> Any:
    """``sp.conley`` on a fitted ``sp.iv`` / ``sp.regress`` result."""
    from ..inference.conley import conley

    return conley(
        fit,
        coords.reset_index(drop=True),
        "_rc_lat",
        "_rc_lon",
        float(cutoff),
        kernel=kernel,
        alpha=alpha,
        distance=distance,
    )
