"""Cluster-robust ``vcov`` spellings accepted by ``sp.feols`` and friends.

``sp.feols(..., vcov={"CL1": "id"})`` used to reach pyfixest unchanged and
die on its internal ``AssertionError`` ("vcov dict key must be CRV1 or
CRV3"), although the request is unambiguous and ``cluster="id"`` works.
This module maps every spelling that clearly means one-way (or multiway)
CRV1 or CRV3 clustering onto the one form the estimation paths read,
``{"CRV1": "a"}`` / ``{"CRV1": "a+b"}`` / ``{"CRV3": "a"}``:

* dict keys, any case: ``CRV1``, ``CR1``, ``CL1``, ``CL``, ``cluster``
  (and Stata's abbreviations ``clu`` / ``clus`` / ``clust``) for CRV1;
  ``CRV3``, ``CR3``, ``CL3`` for CRV3. Values: a column, ``"a+b"``, or a
  list of columns.
* strings: ``"cluster:id"``, ``"cluster(id)"``, ``"cluster = id"``,
  ``"CL1:id"``, ``"CRV3(id)"``, a one-sided formula ``"~id"`` /
  ``"~ a + b"``, and the house-style ``"cluster id"`` /
  ``"vce(cluster id)"`` that :func:`statspai.core._vcov_spec.stata_vce_to_fixest`
  already reads. A bare ``"cluster"`` next to ``cluster=`` is the same
  request as ``cluster=`` alone.

The kind names come from the shared synonym table in ``_vcov_spec`` so
there is one vocabulary. Anything that names an unknown kind raises
:class:`~statspai.exceptions.MethodIncompatibility` with a recovery hint,
and :func:`backend_rejection` turns an ``AssertionError`` raised by
pyfixest into the same exception at the call boundary.
"""

from __future__ import annotations

import functools
import inspect
import re
from typing import Any, Callable, List, Optional, Tuple, TypeVar, cast

from ..exceptions import MethodIncompatibility
from ._vcov_spec import _SE_KIND_SYNONYMS, _VCE_WRAPPER, _bare_kind

_F = TypeVar("_F", bound=Callable[..., Any])

#: Spellings (lower case) of one-way CRV1 and of CRV3 clustering.
CLUSTER_KINDS = frozenset(
    {k for k, v in _SE_KIND_SYNONYMS.items() if v == "cluster"} | {"cl1"}
)
CR3_KINDS = frozenset({k for k, v in _SE_KIND_SYNONYMS.items() if v == "cr3"} | {"cl3"})

_HINT = (
    "Pass cluster='<column>' (or vce='cluster <column>'); for CRV3 use "
    "vcov={'CRV3': '<column>'}."
)

# ``kind:vars`` / ``kind = vars`` / ``kind(vars)``
_KIND_FORM = re.compile(
    r"^\s*(?P<kind>[A-Za-z][A-Za-z0-9_]*)\s*"
    r"(?:[:=]\s*(?P<a>[^()]+?)|\(\s*(?P<b>[^()]*?)\s*\))\s*$"
)
_TILDE_FORM = re.compile(r"^\s*~\s*(?P<body>.+?)\s*$")


def _columns(body: str) -> List[str]:
    return [c for c in re.split(r"[\s+,]+", body.strip()) if c]


def house_style_vce(value: Any) -> Any:
    """Rewrite ``"cluster:id"``-type spellings into ``"cluster id"``.

    Only strings whose kind is a cluster (CRV1) or CR3 spelling are
    rewritten; ``"vce(...)"``, ``"hetero"``, ``"NW"`` and everything else
    come back unchanged for the existing parsers.
    """
    if not isinstance(value, str) or _VCE_WRAPPER.match(value):
        return value
    tilde = _TILDE_FORM.match(value)
    if tilde is not None:
        cols = _columns(tilde.group("body"))
        return "cluster " + " ".join(cols) if cols else value
    form = _KIND_FORM.match(value)
    if form is None:
        return value
    kind = form.group("kind").lower()
    cols = _columns(form.group("a") or form.group("b") or "")
    if not cols:
        return value
    if kind in CLUSTER_KINDS:
        return "cluster " + " ".join(cols)
    if kind in CR3_KINDS:
        return "cr3 " + " ".join(cols)
    return value


def _dict_columns(value: Any, vcov: Any, function: str) -> str:
    if isinstance(value, str) and _columns(value):
        return "+".join(_columns(value))
    if (
        isinstance(value, (list, tuple))
        and value
        and all(isinstance(v, str) and v.strip() for v in value)
    ):
        return "+".join(v.strip() for v in value)
    raise MethodIncompatibility(
        f"{function}: vcov={vcov!r} must map to a column name, 'a+b', or a "
        f"list of column names; got {type(value).__name__}.",
        recovery_hint=_HINT,
        diagnostics={"vcov": repr(vcov)},
    )


def fixest_vcov(vcov: Any, cluster: Any, *, function: str) -> Tuple[Any, Any]:
    """Normalise a ``vcov`` request for the fixest-style estimators.

    Returns ``(vcov, cluster)``. Cluster spellings become
    ``{"CRV1": cols}`` / ``{"CRV3": cols}``; other values are returned
    unchanged for the estimator (and pyfixest) to read.
    """
    if isinstance(vcov, dict):
        if len(vcov) != 1:
            raise MethodIncompatibility(
                f"{function}: vcov dict must hold exactly one entry, got "
                f"{sorted(map(str, vcov))}.",
                recovery_hint=_HINT,
                diagnostics={"vcov": repr(vcov)},
            )
        ((kind, value),) = vcov.items()
        key = str(kind).strip().lower()
        if key in CLUSTER_KINDS:
            small = "CRV1"
        elif key in CR3_KINDS:
            small = "CRV3"
        else:
            extra = (
                " CR2 is available as vcov='CR2' with cluster='<column>'."
                if key in ("cr2", "crv2", "cl2")
                else ""
            )
            raise MethodIncompatibility(
                f"{function}: vcov={vcov!r} names an unknown cluster-robust "
                f"variance {kind!r}. Recognised keys: CRV1 (also CR1, CL1, "
                f"cluster) and CRV3 (also CR3, CL3).{extra}",
                recovery_hint=_HINT,
                diagnostics={"vcov": repr(vcov)},
            )
        return {small: _dict_columns(value, vcov, function)}, cluster
    if isinstance(vcov, str):
        text = house_style_vce(vcov)
        if _bare_kind(text) == "cluster":
            if cluster is None:
                raise MethodIncompatibility(
                    f"{function}: vcov={vcov!r} asks for cluster-robust "
                    "standard errors but names no cluster variable.",
                    recovery_hint=_HINT,
                    diagnostics={"vcov": vcov},
                )
            # Same request as cluster= alone, so the same code path.
            return None, cluster
        return text, cluster
    return vcov, cluster


def normalises_fixest_vcov(function: str) -> Callable[[_F], _F]:
    """Apply :func:`fixest_vcov` to the ``vcov`` / ``cluster`` arguments.

    Sits above ``@markout_clusters`` so that the cluster column of a
    ``"cluster:id"`` spelling is marked out exactly as ``cluster="id"``.
    """

    def decorator(func: _F) -> _F:
        sig = inspect.signature(func)

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                bound = sig.bind_partial(*args, **kwargs)
            except TypeError:
                return func(*args, **kwargs)
            arguments = bound.arguments
            vcov = arguments.get("vcov")
            if vcov is None:
                return func(*args, **kwargs)
            new_vcov, _ = fixest_vcov(vcov, arguments.get("cluster"), function=function)
            if new_vcov is vcov:
                return func(*args, **kwargs)
            arguments["vcov"] = new_vcov
            return func(*bound.args, **bound.kwargs)

        return cast(_F, wrapper)

    return decorator


def backend_rejection(
    exc: AssertionError, vcov: Optional[Any], *, entry: str
) -> MethodIncompatibility:
    """A StatsPAI exception for an ``AssertionError`` raised by pyfixest."""
    text = (str(exc) or type(exc).__name__).rstrip(".")
    about_vcov = "vcov" in text.lower()
    return MethodIncompatibility(
        f"{entry}: the estimation backend rejected the call ({text})."
        + (f" vcov={vcov!r} is not a recognised variance." if about_vcov else ""),
        recovery_hint=(
            _HINT + " Other choices: 'iid', 'hetero', 'HC1'-'HC3'."
            if about_vcov
            else "Check the formula and arguments; the backend message is above."
        ),
        diagnostics={"backend_message": text, "vcov": repr(vcov)},
    )
