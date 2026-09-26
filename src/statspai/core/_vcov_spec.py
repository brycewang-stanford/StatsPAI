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

import functools
import inspect
import os
import re
import sys
from dataclasses import dataclass
from typing import (
    Any,
    Callable,
    Dict,
    FrozenSet,
    Iterable,
    List,
    Optional,
    Tuple,
    TypeVar,
    cast,
)

from ..exceptions import MethodIncompatibility, StatsPAIWarning, warn

_F = TypeVar("_F", bound=Callable[..., Any])

__all__ = [
    "SERequest",
    "markout_clusters",
    "normalize_vcov",
    "parse_se_request",
    "reject_unknown_kwargs",
]


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


# ---------------------------------------------------------------------------
# Stata-style SE requests: vce="robust" / True / "vce(cluster firm)" / ...
# ---------------------------------------------------------------------------
#
# Before this parser each estimator read its ``robust=`` string with its own
# ad-hoc test, and the tests disagreed in ways that were invisible to the
# caller: ``logit`` treated every string other than "nonrobust" as the
# sandwich (so ``vce="cluster firm"`` quietly returned heteroskedasticity-
# robust SEs), ``ologit`` treated every string other than "robust"/"hc1" as
# model-based (so ``vce="bogus"`` quietly returned OIM SEs), and
# ``regress`` rejected ``vce="robust"`` outright.  The parser gives every
# estimator the same vocabulary and the same failure mode: a spelling is
# either understood and implemented, or the call raises.
#
# The canonical *kinds* below are estimator-agnostic.  What ``"robust"``
# means numerically is each estimator's business and follows Stata:
# HC1 on the least-squares family (``regress, vce(robust)``) and the ML
# sandwich with an N/(N-1) factor on maximum-likelihood models
# (``logit, vce(robust)``).  See ``core._vcov.ml_vcov``.

#: Spelling (lower-case) -> canonical SE kind.
_SE_KIND_SYNONYMS: Dict[str, str] = {
    # Model-based / observed-information standard errors.
    "nonrobust": "nonrobust",
    "ols": "nonrobust",
    "oim": "nonrobust",
    "iid": "nonrobust",
    "classical": "nonrobust",
    "conventional": "nonrobust",
    "unadjusted": "nonrobust",
    "homoskedastic": "nonrobust",
    # Heteroskedasticity-robust.  Stata accepts the abbreviations r / rob.
    "robust": "robust",
    "r": "robust",
    "rob": "robust",
    "hetero": "robust",
    "huber": "robust",
    "white": "hc0",
    "hc0": "hc0",
    "hc1": "hc1",
    "hc2": "hc2",
    "hc3": "hc3",
    "hac": "hac",
    # Cluster-robust.  Stata accepts cl / clu / clus / clust.
    "cluster": "cluster",
    "cl": "cluster",
    "clu": "cluster",
    "clus": "cluster",
    "clust": "cluster",
    "cr1": "cluster",
    "crv1": "cluster",
    "cr2": "cr2",
    "crv2": "cr2",
    "cr3": "cr3",
    "crv3": "cr3",
    "jackknife": "jackknife",
    "wild": "wild",
    "wildbootstrap": "wild",
    "wild_cluster": "wild",
    "wcr": "wild",
    "wre": "wild",
    "boottest": "wild",
    # Spatial.
    "conley": "conley",
}

#: Kinds that are meaningless without a cluster variable.
_NEEDS_CLUSTER: FrozenSet[str] = frozenset(
    {"cluster", "cr2", "cr3", "jackknife", "wild"}
)

#: Kinds a supplied ``cluster=`` silently upgrades to one-way clustering.
#: This is the long-standing ``robust="hc1", cluster="firm"`` idiom (and
#: Stata's own ``cluster()`` option), so it keeps working.
_UPGRADES_TO_CLUSTER: FrozenSet[str] = frozenset({"nonrobust", "robust", "hc0", "hc1"})

_VCE_WRAPPER = re.compile(r"^\s*vce\s*\((?P<body>.*)\)\s*$", re.IGNORECASE)


@dataclass(frozen=True)
class SERequest:
    """A parsed standard-error request.

    Attributes
    ----------
    kind : str
        Canonical kind: ``"nonrobust"``, ``"robust"``, ``"hc0"``–``"hc3"``,
        ``"hac"``, ``"cluster"``, ``"cr2"``, ``"cr3"``, ``"jackknife"``,
        ``"wild"`` or ``"conley"``.
    cluster : Any
        The cluster specification after merging ``cluster=`` with any
        variable named inside the ``vce`` string (``"cluster firm"``).
    spelling : Any
        What the caller passed, kept for error messages and provenance.
    """

    kind: str
    cluster: Any
    spelling: Any


def _is_str_list(value: Any) -> bool:
    return isinstance(value, (list, tuple)) and all(isinstance(v, str) for v in value)


def parse_se_request(
    robust: Any,
    cluster: Any = None,
    *,
    function: str,
    supported: Iterable[str],
    multiway: bool = False,
) -> SERequest:
    """Parse a Stata-style standard-error request into a canonical kind.

    Parameters
    ----------
    robust : None, bool or str
        The estimator's ``robust=`` value (``vce=`` has already been folded
        into it by ``accepts_aliases``).  Accepts ``True`` / ``False``, any
        spelling in the synonym table (case-insensitive), Stata's
        ``"vce(...)"`` wrapper, and a cluster variable written inline:
        ``"cluster firm"`` or, when ``multiway`` is allowed,
        ``"cluster firm year"``.
    cluster : optional
        The estimator's ``cluster=`` value.  Column names, lists of column
        names and arrays are all passed through untouched.
    function : str
        Name used in error messages.
    supported : iterable of str
        Canonical kinds the estimator implements.  Include ``"cluster"``
        when the estimator supports one-way clustering.
    multiway : bool, default False
        Whether a list of cluster variables is allowed.

    Returns
    -------
    SERequest

    Raises
    ------
    MethodIncompatibility
        When the spelling is unknown, recognised but not implemented by
        this estimator, missing its cluster variable, or in conflict with
        ``cluster=``.  Returning some other kind of standard error instead
        is exactly the silent failure this parser exists to remove.
    """
    supported_set = frozenset(supported)
    shown = ", ".join(repr(k) for k in sorted(supported_set))

    def _unknown() -> MethodIncompatibility:
        return MethodIncompatibility(
            f"{function}: Unknown robust option: {robust!r}. "
            f"Supported vce/robust values here: {shown}.",
            recovery_hint=(
                "Use a Stata-style spelling such as vce='robust', "
                "vce='cluster firm', or one of the listed values."
            ),
            diagnostics={"robust": repr(robust), "supported": sorted(supported_set)},
        )

    inline: Tuple[str, ...] = ()
    if robust is None or robust is False:
        kind = "nonrobust"
    elif robust is True:
        kind = "robust"
    elif isinstance(robust, str):
        text = robust.strip()
        wrapped = _VCE_WRAPPER.match(text)
        if wrapped is not None:
            text = wrapped.group("body").strip()
        parts = text.replace(",", " ").split()
        if not parts or parts[0].lower() not in _SE_KIND_SYNONYMS:
            raise _unknown()
        kind = _SE_KIND_SYNONYMS[parts[0].lower()]
        inline = tuple(parts[1:])
        if inline and kind not in _NEEDS_CLUSTER:
            raise MethodIncompatibility(
                f"{function}: vce={robust!r} names {list(inline)} after "
                f"{parts[0]!r}, but only cluster-type standard errors take a "
                "variable.",
                recovery_hint="Write vce='cluster firm' or pass cluster='firm'.",
            )
    else:
        raise MethodIncompatibility(
            f"{function}: robust/vce must be a string or a bool, got "
            f"{type(robust).__name__}.",
            recovery_hint="Pass e.g. vce='robust' or vce='cluster firm'.",
        )

    if inline:
        named: Any = inline[0] if len(inline) == 1 else list(inline)
        if cluster is not None:
            same = (
                cluster == named
                if isinstance(cluster, str)
                else _is_str_list(cluster) and list(cluster) == list(inline)
            )
            if not same:
                raise MethodIncompatibility(
                    f"{function}: vce={robust!r} clusters on {named!r} but "
                    f"cluster={cluster!r} was also passed. Pass the cluster "
                    "variable once.",
                    recovery_hint="Drop either the name inside vce= or cluster=.",
                )
        cluster = named

    if _is_str_list(cluster):
        if len(cluster) == 1:
            cluster = cluster[0]
        elif not multiway:
            raise MethodIncompatibility(
                f"{function}: multiway clustering on {list(cluster)} is not "
                "available; pass a single cluster variable.",
                recovery_hint="Use sp.feols / sp.regress for multiway clustering.",
            )

    if kind in _NEEDS_CLUSTER and cluster is None:
        raise MethodIncompatibility(
            f"{function}: vce={robust!r} requires cluster=<column> "
            "(a cluster variable).",
            recovery_hint="Write vce='cluster firm' or pass cluster='firm'.",
        )
    if cluster is not None:
        if kind in _UPGRADES_TO_CLUSTER:
            kind = "cluster"
        elif kind not in _NEEDS_CLUSTER:
            raise MethodIncompatibility(
                f"{function}: vce={robust!r} cannot be combined with "
                f"cluster={cluster!r}; a supplied cluster variable means "
                "cluster-robust standard errors.",
                recovery_hint="Drop cluster= or ask for vce='cluster'.",
            )

    if kind not in supported_set:
        raise MethodIncompatibility(
            f"{function}: vce={robust!r} ({kind}) is not available for this "
            f"estimator. Supported vce/robust values here: {shown}.",
            recovery_hint=f"Use one of: {shown}.",
            diagnostics={"robust": repr(robust), "kind": kind},
        )
    return SERequest(kind=kind, cluster=cluster, spelling=robust)


def _named_clusters(value: Any) -> List[str]:
    """Column names a ``robust=`` / ``vce=`` / ``cluster=`` / ``vcov=`` value
    clusters on; anything that is not a column name (arrays, bools) is
    ignored here and left to the estimator."""
    if isinstance(value, str):
        text = value.strip()
        wrapped = _VCE_WRAPPER.match(text)
        if wrapped is not None:
            text = wrapped.group("body").strip()
        parts = text.replace(",", " ").split()
        if parts and _SE_KIND_SYNONYMS.get(parts[0].lower()) in _NEEDS_CLUSTER:
            return parts[1:]
        return []
    if isinstance(value, dict):
        return [n for v in value.values() for n in _named_clusters_plain(v)]
    return []


def _named_clusters_plain(value: Any) -> List[str]:
    if isinstance(value, str):
        return [value]
    if _is_str_list(value):
        return list(value)
    return []


_PACKAGE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def _caller_stacklevel() -> int:
    """``stacklevel`` for :func:`statspai.exceptions.warn`, called from the
    wrapper below, that points at the first frame outside the package (the
    wrapper may sit under ``accepts_aliases`` or a dispatcher)."""
    level = 2  # warn() -> wrapper
    frame = sys._getframe(2)  # the wrapper's caller
    while frame is not None and os.path.abspath(frame.f_code.co_filename).startswith(
        _PACKAGE_DIR + os.sep
    ):
        level += 1
        frame = frame.f_back
    return level + 1


def markout_clusters(func: _F) -> _F:
    """Exclude rows whose cluster variable is missing, as Stata does.

    Stata's ``vce(cluster v)`` marks observations with missing ``v`` out of
    the estimation sample (R ``fixest`` does the same with a note).  Without
    this, estimators disagreed: some raised, some dropped the rows, and some
    kept them in the fit.  The decorator resolves the cluster column(s) from
    ``cluster=``, an inline ``vce="cluster v"`` / ``robust=`` spelling or a
    ``vcov={"CRV1": "v"}`` dict, drops the rows where any of them is missing,
    emits a :class:`~statspai.exceptions.StatsPAIWarning` giving the count,
    and records it as ``model_info["n_missing_cluster_dropped"]``.  Columns
    passed as arrays, and a sample in which every cluster label is missing,
    are left to the estimator.
    """
    sig = inspect.signature(func)

    @functools.wraps(func)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        try:
            bound = sig.bind_partial(*args, **kwargs)
        except TypeError:
            return func(*args, **kwargs)
        arguments = bound.arguments
        data = arguments.get("data")
        if data is None or not hasattr(data, "columns"):
            return func(*args, **kwargs)
        names: List[str] = []
        for key in ("robust", "vce", "vcov"):
            names += _named_clusters(arguments.get(key))
        names += _named_clusters_plain(arguments.get("cluster"))
        names = [n for n in dict.fromkeys(names) if n in data.columns]
        if not names:
            return func(*args, **kwargs)
        missing = data[names].isna().any(axis=1)
        n_missing = int(missing.sum())
        if n_missing == 0 or n_missing == len(data):
            return func(*args, **kwargs)
        warn(
            StatsPAIWarning,
            f"{func.__name__}: {n_missing} observation(s) with a missing "
            f"cluster variable ({', '.join(names)}) excluded from the "
            "estimation sample, as Stata's vce(cluster) does.",
            stacklevel=_caller_stacklevel(),
            recovery_hint="Drop or recode those rows before fitting to "
            "silence this warning.",
            diagnostics={"cluster": names, "n_dropped": n_missing},
        )
        arguments["data"] = data.loc[~missing]
        result = func(*bound.args, **bound.kwargs)
        info = getattr(result, "model_info", None)
        if isinstance(info, dict):
            info["n_missing_cluster_dropped"] = n_missing
            info["n_input_rows"] = int(len(data))
        return result

    return cast(_F, wrapper)
