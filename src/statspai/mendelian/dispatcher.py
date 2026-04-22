"""
Unified dispatcher: ``sp.mr(method=...)``

Single entry point for the Mendelian Randomization family.  Mirrors the
style of ``sp.synth`` / ``sp.decompose`` / ``sp.dml``.

The underlying estimators have heterogeneous signatures because the
single-exposure family (IVW / Egger / median / mode) consumes
SNP-summary arrays, while the multi-exposure family (MVMR / BMA /
mediation) consumes DataFrames with multiple SNP-association columns.
We expose both through one ``method=`` switch and just pass the
user's kwargs straight through to the target — no normalisation
layer that could silently massage numbers.

Examples
--------
>>> import statspai as sp
>>> # Single-exposure IVW on summary-stat arrays
>>> r = sp.mr("ivw", beta_exposure=bx, beta_outcome=by,
...            se_exposure=sx, se_outcome=sy)
>>> # All-methods convenience wrapper over a DataFrame
>>> r = sp.mr("all", data=snp_df, beta_exposure="beta_x",
...            se_exposure="se_x", beta_outcome="beta_y", se_outcome="se_y")
>>> # Multivariable MR
>>> r = sp.mr("mvmr", snp_associations=snp_df,
...            outcome="beta_y", outcome_se="se_y",
...            exposures=["beta_bmi", "beta_ldl"])
"""
from __future__ import annotations

from typing import Any, Dict, Tuple


# name -> (module_path, function_name) -- lazy-imported so importing the
# dispatcher is cheap.
_REGISTRY: Dict[str, Tuple[str, str]] = {
    # -- Single-exposure point estimators -------------------------- #
    "ivw": ("statspai.mendelian.mr", "mr_ivw"),
    "inverse_variance_weighted": ("statspai.mendelian.mr", "mr_ivw"),

    "egger": ("statspai.mendelian.mr", "mr_egger"),
    "mr_egger": ("statspai.mendelian.mr", "mr_egger"),

    "median": ("statspai.mendelian.mr", "mr_median"),
    "weighted_median": ("statspai.mendelian.mr", "mr_median"),
    # penalized_median is dispatched via kwargs (penalized=True) inside
    # the dispatcher; see below.
    "penalized_median": ("statspai.mendelian.mr", "mr_median"),

    "mode": ("statspai.mendelian.extras", "mr_mode"),
    "weighted_mode": ("statspai.mendelian.extras", "mr_mode"),
    "simple_mode": ("statspai.mendelian.extras", "mr_mode"),

    # -- All-methods convenience ----------------------------------- #
    "all": ("statspai.mendelian.mr", "mendelian_randomization"),
    "mr": ("statspai.mendelian.mr", "mendelian_randomization"),
    "mendelian_randomization": ("statspai.mendelian.mr",
                                "mendelian_randomization"),

    # -- Multi-exposure / mediation -------------------------------- #
    "mvmr": ("statspai.mendelian.multivariable", "mr_multivariable"),
    "multivariable": ("statspai.mendelian.multivariable", "mr_multivariable"),
    "mr_multivariable": ("statspai.mendelian.multivariable",
                         "mr_multivariable"),

    "mediation": ("statspai.mendelian.multivariable", "mr_mediation"),
    "two_step": ("statspai.mendelian.multivariable", "mr_mediation"),
    "mr_mediation": ("statspai.mendelian.multivariable", "mr_mediation"),

    "bma": ("statspai.mendelian.multivariable", "mr_bma"),
    "mr_bma": ("statspai.mendelian.multivariable", "mr_bma"),
    "bayesian_model_averaging": ("statspai.mendelian.multivariable", "mr_bma"),

    # -- Diagnostics (dispatched for agent-native discoverability) - #
    "presso": ("statspai.mendelian.diagnostics", "mr_presso"),
    "mr_presso": ("statspai.mendelian.diagnostics", "mr_presso"),
    "radial": ("statspai.mendelian.diagnostics", "mr_radial"),
    "mr_radial": ("statspai.mendelian.diagnostics", "mr_radial"),
    "leave_one_out": ("statspai.mendelian.diagnostics", "mr_leave_one_out"),
    "loo": ("statspai.mendelian.diagnostics", "mr_leave_one_out"),
    "steiger": ("statspai.mendelian.diagnostics", "mr_steiger"),
    "heterogeneity": ("statspai.mendelian.diagnostics", "mr_heterogeneity"),
    "pleiotropy_egger": ("statspai.mendelian.diagnostics",
                         "mr_pleiotropy_egger"),
    "f_statistic": ("statspai.mendelian.extras", "mr_f_statistic"),
    "f_stat": ("statspai.mendelian.extras", "mr_f_statistic"),

    # -- v1.6 frontier: sample-overlap / clusters / profile LL / cML - #
    "lap": ("statspai.mendelian.frontier", "mr_lap"),
    "mr_lap": ("statspai.mendelian.frontier", "mr_lap"),
    "sample_overlap": ("statspai.mendelian.frontier", "mr_lap"),

    "clust": ("statspai.mendelian.frontier", "mr_clust"),
    "mr_clust": ("statspai.mendelian.frontier", "mr_clust"),
    "clustered": ("statspai.mendelian.frontier", "mr_clust"),

    "grapple": ("statspai.mendelian.frontier", "grapple"),
    "profile_likelihood": ("statspai.mendelian.frontier", "grapple"),

    "cml": ("statspai.mendelian.frontier", "mr_cml"),
    "mr_cml": ("statspai.mendelian.frontier", "mr_cml"),
    "constrained_ml": ("statspai.mendelian.frontier", "mr_cml"),
}


def available_methods() -> list[str]:
    """Return the full list of registered MR method names (incl. aliases)."""
    return sorted(_REGISTRY.keys())


def mr(method: str = "ivw", /, **kwargs: Any) -> Any:
    """Unified entry point for the Mendelian Randomization family.

    Parameters
    ----------
    method : str, default ``"ivw"``
        The MR estimator or diagnostic to run.  Supported values are
        listed by ``sp.mr_available_methods()``.  Aliases are
        case-sensitive (lowercase).
    **kwargs
        Passed through unchanged to the target function.  See the
        referenced function's docstring for its specific signature.

    Returns
    -------
    Whatever the target function returns.  For point estimators this is
    typically a dict (``mr_ivw`` / ``mr_egger`` / ``mr_median``) or a
    dataclass (``mr_mode``).  For ``method="all"`` / ``"mr"`` you get an
    :class:`MRResult` with all three canonical methods and the Egger +
    Cochran Q diagnostics attached.

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np
    >>> bx = np.array([0.10, 0.15, 0.12])
    >>> by = np.array([0.40, 0.55, 0.48])
    >>> sx = np.array([0.02, 0.03, 0.02])
    >>> sy = np.array([0.08, 0.10, 0.09])
    >>> r = sp.mr("ivw", beta_exposure=bx, beta_outcome=by,
    ...            se_exposure=sx, se_outcome=sy)
    >>> round(r["estimate"], 2)
    4.0

    See Also
    --------
    docs/guides/mendelian_family.md : the full family guide.
    """
    if method not in _REGISTRY:
        raise ValueError(
            f"Unknown MR method {method!r}. Available: "
            + ", ".join(available_methods())
        )

    # Handle the penalized_median special case: mr_median takes
    # penalized=bool but the alias lives at the dispatcher layer.
    if method == "penalized_median":
        kwargs.setdefault("penalized", True)

    if method == "simple_mode":
        kwargs.setdefault("method", "simple")

    module_path, fn_name = _REGISTRY[method]
    import importlib
    mod = importlib.import_module(module_path)
    fn = getattr(mod, fn_name)
    return fn(**kwargs)


__all__ = ["mr", "available_methods"]
