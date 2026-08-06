"""Canonical variance-estimator (vcov) specification across estimators.

StatsPAI grew two spellings for "which standard errors do you want":

* the Stata-flavoured pair ``robust="hc1", cluster="firm"`` used by
  :func:`statspai.regress`, :func:`statspai.ivreg`, and friends;
* the pyfixest/``fixest``-flavoured ``vcov="hetero"`` /
  ``vcov={"CRV1": "firm"}`` used by :func:`statspai.feols`.

Both are legitimate — users arrive from Stata and from R — but before
this module ``sp.regress(..., vcov={"CRV1": "firm"})`` was accepted and
then *silently dropped*, handing back unclustered standard errors with
no warning.  That is the single cheapest way to publish an
understated SE, so the vocabulary is now shared: every entry point
accepts both spellings and translates to its native one.

:func:`normalize_vcov` is the one shared primitive.
"""

from typing import Any, Dict, Optional, Tuple

from ..exceptions import MethodIncompatibility

__all__ = ["normalize_vcov", "reject_unknown_kwargs"]


# pyfixest / fixest scalar spellings -> StatsPAI ``robust=`` spellings.
_SCALAR_VCOV = {
    "iid": "nonrobust",
    "nonrobust": "nonrobust",
    "classical": "nonrobust",
    "hetero": "hc1",
    "hc0": "hc0",
    "hc1": "hc1",
    "hc2": "hc2",
    "hc3": "hc3",
    "hac": "hac",
    "robust": "hc1",
}

# Cluster-robust dict spellings -> (cluster small-sample kind, native vce).
_CLUSTER_VCOV = {
    "crv1": None,  # plain CR1 — the default when cluster= is set
    "cr1": None,
    "crv2": "CR2",
    "cr2": "CR2",
    "crv3": "CR3",
    "cr3": "CR3",
}


def normalize_vcov(
    *,
    vcov: Optional[Any] = None,
    robust: Optional[str] = None,
    cluster: Optional[str] = None,
    vce: Optional[str] = None,
    function: str = "this function",
) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Translate any accepted vcov spelling into ``(robust, cluster, vce)``.

    Parameters
    ----------
    vcov : str or dict, optional
        pyfixest-style specification: ``"iid"`` / ``"hetero"`` /
        ``"HC1"`` …, or ``{"CRV1": "firm"}`` / ``{"CRV3": "firm"}``.
    robust, cluster, vce : optional
        The estimator's native arguments, passed through unchanged when
        ``vcov`` is not supplied.
    function : str
        Name used in error messages.

    Returns
    -------
    (robust, cluster, vce)
        Native-spelling triple. ``vce`` is set only when the ``vcov``
        dict asked for a CR2/CR3 small-sample correction.

    Raises
    ------
    MethodIncompatibility
        If ``vcov`` is supplied alongside a conflicting ``robust`` /
        ``cluster`` / ``vce``, or if the spelling is unrecognised.
        Silently preferring one would hand back the wrong standard
        errors.
    """
    if vcov is None:
        return robust, cluster, vce

    conflicting = [
        name
        for name, value in (
            ("robust", robust),
            ("cluster", cluster),
            ("vce", vce),
        )
        # ``robust`` carries a non-None default ("nonrobust"); only an
        # explicit non-default value counts as a conflict.
        if value is not None and not (name == "robust" and value == "nonrobust")
    ]
    if conflicting:
        raise MethodIncompatibility(
            f"{function}: vcov={vcov!r} was supplied together with "
            f"{', '.join(conflicting)}. These specify the same thing in "
            "two vocabularies; pass only one.",
            recovery_hint=(f"Drop either vcov= or {'/'.join(conflicting)}."),
        )

    if isinstance(vcov, str):
        key = vcov.strip().lower()
        if key not in _SCALAR_VCOV:
            raise MethodIncompatibility(
                f"{function}: unknown vcov={vcov!r}.",
                recovery_hint=(
                    "Use one of "
                    f"{sorted(set(_SCALAR_VCOV))}, or a cluster dict such "
                    'as {"CRV1": "firm"}.'
                ),
            )
        return _SCALAR_VCOV[key], None, None

    if isinstance(vcov, dict):
        if len(vcov) != 1:
            raise MethodIncompatibility(
                f"{function}: vcov dict must hold exactly one entry, got "
                f"{sorted(vcov)}.",
                recovery_hint='Pass e.g. {"CRV1": "firm"}.',
            )
        ((kind, column),) = vcov.items()
        key = str(kind).strip().lower()
        if key not in _CLUSTER_VCOV:
            raise MethodIncompatibility(
                f"{function}: unknown cluster-robust spelling {kind!r} in "
                f"vcov={vcov!r}.",
                recovery_hint=(
                    f"Use one of {sorted(set(_CLUSTER_VCOV))}, e.g. "
                    '{"CRV1": "firm"}.'
                ),
            )
        if not isinstance(column, str):
            raise MethodIncompatibility(
                f"{function}: vcov={vcov!r} must map to a column name "
                f"(string), got {type(column).__name__}.",
                recovery_hint='Pass e.g. {"CRV1": "firm"}.',
            )
        return None, column, _CLUSTER_VCOV[key]

    raise MethodIncompatibility(
        f"{function}: vcov must be a string or a one-entry dict, got "
        f"{type(vcov).__name__}.",
        recovery_hint='Pass e.g. vcov="hetero" or vcov={"CRV1": "firm"}.',
    )


def reject_unknown_kwargs(
    kwargs: Dict[str, Any], *, function: str, known: Tuple[str, ...] = ()
) -> None:
    """Raise ``TypeError`` for leftover keyword arguments.

    Entry points that forward ``**kwargs`` down a chain used to drop
    anything unrecognised on the floor — a misspelled ``robsut="hc1"``
    or a wrong-dialect ``vcov=`` silently produced default standard
    errors. Call this once the recognised keys have been popped.
    """
    unknown = sorted(k for k in kwargs if k not in known)
    if unknown:
        raise TypeError(
            f"{function}() got unexpected keyword argument(s): "
            f"{', '.join(repr(k) for k in unknown)}. "
            "Check the spelling against the function signature — "
            "unrecognised arguments are rejected rather than ignored so "
            "a typo cannot silently change your standard errors."
        )
