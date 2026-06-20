"""Public entry point: :func:`cross_validate`.

Run one estimand through several independent engines and report whether they
agree. This operationalises the cross-package-validation discipline Scott
Cunningham argues for — "require at least two independent packages to estimate
the same model, and only trust the result when they agree" — as a single
callable that humans and agents share.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import pandas as pd

from ._agreement import (
    STATUS_ERROR,
    STATUS_OK,
    STATUS_UNAVAILABLE,
    EngineEstimate,
    reconcile,
    resolve_policy,
)
from ._engines import EngineAdapter, inprocess_adapters
from ._external import external_adapters
from ._result import CrossValidationResult
from ._spec import EstimandSpec


def cross_validate(
    data_or_result: Union[pd.DataFrame, Any],
    estimand: Optional[str] = None,
    *,
    formula: Optional[str] = None,
    y: Optional[str] = None,
    treatment: Optional[str] = None,
    covariates: Optional[List[str]] = None,
    fixed_effects: Optional[List[str]] = None,
    endog: Optional[List[str]] = None,
    instruments: Optional[List[str]] = None,
    cluster: Optional[Union[str, List[str]]] = None,
    weights: Optional[str] = None,
    vcov: Optional[str] = None,
    term: Optional[str] = None,
    engines: Union[str, Sequence[str]] = "auto",
    reference: Optional[str] = None,
    tol: Optional[Any] = None,
    alpha: float = 0.05,
    **extra: Any,
) -> CrossValidationResult:
    """Cross-validate one estimand across independent engines.

    Parameters
    ----------
    data_or_result : pandas.DataFrame or fitted StatsPAI result
        Either the dataset (then ``estimand`` and a model spec are required) or
        a fitted StatsPAI result, in which case the estimand, formula and
        treatment are recovered from it (see :meth:`EstimandSpec.from_result`).
    estimand : str, optional
        ``"ols"`` / ``"feols"`` / ``"iv"`` / ``"poisson"`` / ``"dml"`` (and
        aliases like ``"reg"``, ``"2sls"``, ``"fe"``). Required in data mode.
    formula : str, optional
        fixest-style: ``"y ~ x | fe | endog ~ z"``. Alternative to the
        structured ``y`` / ``treatment`` / ... arguments.
    y, treatment, covariates, fixed_effects, endog, instruments, cluster, \
weights, vcov : optional
        Structured model specification (see :class:`EstimandSpec`).
    term : str, optional
        Focal coefficient to reconcile. Defaults to ``treatment`` (or the first
        endogenous regressor for IV).
    engines : "auto" or sequence of str, default "auto"
        ``"auto"`` runs every installed engine that supports the estimand.
        A list (e.g. ``["statspai", "R::fixest", "pyfixest"]``) runs exactly
        those — missing ones are reported as ``unavailable`` rather than
        skipped silently.
    reference : str, optional
        Engine to anchor pairwise comparison on (default: ``"statspai"``).
    tol : dict or TolerancePolicy, optional
        Override the per-estimand tolerance. e.g. ``{"coef_rtol": 1e-4}`` or
        ``{"se_band": 0.5}`` for randomised estimators.
    alpha : float, default 0.05
    **extra
        Forwarded to the spec (e.g. ``n_folds`` / ``seed`` / ``model`` for DML).

    Returns
    -------
    CrossValidationResult
        Per-engine estimates, an AGREE / PARTIAL / DISAGREE / INSUFFICIENT
        verdict, reproducibility provenance and concrete next steps.

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> n = 300
    >>> x = rng.normal(size=n)
    >>> y = 1.0 + 0.5 * x + rng.normal(size=n)
    >>> df = pd.DataFrame({"y": y, "x": x})
    >>> cv = sp.cross_validate(df, "ols", formula="y ~ x", treatment="x",
    ...                        engines=["statspai", "pyfixest"])
    >>> cv.verdict in {"AGREE", "PARTIAL", "DISAGREE", "INSUFFICIENT"}
    True
    """
    if isinstance(data_or_result, pd.DataFrame):
        if estimand is None:
            raise ValueError(
                "In data mode you must pass an estimand, e.g. "
                "sp.cross_validate(df, 'ols', formula='y ~ x')."
            )
        spec = EstimandSpec.from_kwargs(
            data_or_result,
            estimand,
            formula=formula,
            y=y,
            treatment=treatment,
            covariates=covariates,
            fixed_effects=fixed_effects,
            endog=endog,
            instruments=instruments,
            cluster=cluster,
            weights=weights,
            vcov=vcov,
            term=term,
            **extra,
        )
    else:
        spec = EstimandSpec.from_result(data_or_result)
        if term:
            spec.term = term
        if vcov:
            spec.vcov = vcov

    focal = spec.focal_term()
    adapters_with_names = _resolve_adapters(spec, engines)

    estimates: List[EngineEstimate] = []
    degradations: List[Dict[str, Any]] = []
    for adapter, requested_name, explicit in adapters_with_names:
        if adapter is None:
            est = EngineEstimate(
                engine=requested_name or "unknown",
                estimand=spec.estimand,
                term=focal,
                status=STATUS_UNAVAILABLE,
                message=f"no adapter matches engine name {requested_name!r}",
            )
        else:
            est = adapter.run(spec)
        estimates.append(est)
        _maybe_record(est, explicit=explicit, sink=degradations)

    policy = resolve_policy(spec.estimand, tol)
    report = reconcile(
        estimates, policy=policy, reference=reference or "statspai", alpha=alpha
    )

    return CrossValidationResult(
        estimand=spec.estimand,
        term=focal,
        estimates=estimates,
        agreement=report,
        spec=spec.to_dict(),
        provenance=_provenance(estimates, spec=spec),
        degradations=degradations,
    )


# --------------------------------------------------------------------------- #
# Engine resolution
# --------------------------------------------------------------------------- #


def _all_adapters() -> List[EngineAdapter]:
    return inprocess_adapters() + external_adapters()


def _resolve_adapters(
    spec: EstimandSpec, engines: Union[str, Sequence[str]]
) -> List[Tuple[Optional[EngineAdapter], Optional[str], bool]]:
    """Return ``(adapter, requested_name, explicit)`` triples.

    ``explicit`` distinguishes a user-named engine (failures recorded loudly)
    from an auto-selected one (only genuine errors recorded).
    """
    pool = _all_adapters()

    if engines == "auto" or engines is None:
        selected: List[Tuple[Optional[EngineAdapter], Optional[str], bool]] = []
        for a in pool:
            sup, _ = a._supports(spec)
            avail, _ = a._available()
            if sup and avail:
                selected.append((a, a.name, False))
        return selected

    if isinstance(engines, str):
        engines = [engines]
    out: List[Tuple[Optional[EngineAdapter], Optional[str], bool]] = []
    for name in engines:
        match = next((a for a in pool if a.matches(name)), None)
        out.append((match, name, True))
    return out


def _maybe_record(
    est: EngineEstimate, *, explicit: bool, sink: List[Dict[str, Any]]
) -> None:
    """Record non-ok engines, keeping genuine failures loud.

    Errors are always recorded (a backend that *ran and crashed* is a real
    degradation). Unavailable / skipped engines are recorded only when the user
    named them explicitly — auto mode stays quiet about backends it simply did
    not pick.
    """
    if est.status == STATUS_OK:
        return
    should = est.status == STATUS_ERROR or explicit
    if not should:
        return
    from ..workflow._degradation import record_degradation

    exc = _SyntheticEngineFailure(est.message or est.status)
    record_degradation(
        sink,
        section=f"cross_validate engine '{est.engine}'",
        exc=exc,
        detail=f"status={est.status}",
    )


class _SyntheticEngineFailure(RuntimeError):
    """Carries an engine's failure message into ``record_degradation``."""


# --------------------------------------------------------------------------- #
# Provenance
# --------------------------------------------------------------------------- #

_VERSION_MODULES = {
    "statspai": "statspai",
    "pyfixest": "pyfixest",
    "linearmodels": "linearmodels",
    "doubleml": "doubleml",
}


def _data_source_provenance(spec: EstimandSpec) -> Optional[Dict[str, Any]]:
    attrs = getattr(spec.data, "attrs", {}) or {}
    raw = attrs.get("provenance")
    if isinstance(raw, dict):
        out = dict(raw)
    elif attrs.get("source"):
        out = {"source": attrs.get("source")}
    else:
        return None
    out.setdefault("n_rows", int(len(spec.data)))
    out.setdefault("columns", [str(c) for c in spec.data.columns])
    return out


def _provenance(
    estimates: List[EngineEstimate],
    *,
    spec: Optional[EstimandSpec] = None,
) -> Dict[str, Any]:
    """Capture engine versions for the engines that actually ran."""
    prov: Dict[str, Any] = {}
    if spec is not None:
        data_prov = _data_source_provenance(spec)
        if data_prov:
            prov["data"] = data_prov
    ran = {e.engine for e in estimates if e.ok}
    for engine in ran:
        mod = _VERSION_MODULES.get(engine)
        if mod:
            try:
                m = __import__(mod)
                prov[engine] = getattr(m, "__version__", "unknown")
            except Exception:  # noqa: BLE001
                prov[engine] = "unknown"
        elif engine.startswith("R::"):
            prov[engine] = _rscript_version()
        elif engine == "Stata":
            prov[engine] = "stata (subprocess)"
    return prov


def _rscript_version() -> str:
    import shutil
    import subprocess

    if shutil.which("Rscript") is None:
        return "unknown"
    try:
        out = subprocess.run(
            ["Rscript", "-e", "cat(as.character(getRversion()))"],
            capture_output=True,
            text=True,
            timeout=20,
        )
        return f"R {out.stdout.strip()}" if out.returncode == 0 else "unknown"
    except Exception:  # noqa: BLE001
        return "unknown"
