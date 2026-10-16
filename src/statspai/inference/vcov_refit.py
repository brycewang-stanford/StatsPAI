"""Standard errors under another ``vce=`` for a result you already have.

``sp.vcov(result, vce="cluster state")`` answers "what would my standard
errors be if I had clustered by state?" without the caller rebuilding the
call. It does so by re-running the recorded call (``core._call_record``)
with ``vce=`` replaced, so the numbers are by construction the ones the
estimator itself reports for that ``vce`` -- the path its parity tests pin.
An earlier shortcut that re-weighted ``X`` and the residuals on its own
(``sp.regtable(vcov=)``) ignored regression weights and applied the OLS
formula to logit fits; reusing the estimator is how that cannot recur.

Before switching ``vce``, the recorded call is replayed unchanged and must
reproduce the original estimates to ``1e-10``. That check refuses edited
data, a different frame passed as ``data=``, and estimators whose point
estimate depends on an unseeded random draw.
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from ..core._call_record import get_call_record
from ..exceptions import MethodIncompatibility, StatsPAIWarning, warn

__all__ = ["vcov", "vcov_compare"]

#: Arguments that state a standard-error choice. ``vce=`` replaces all of them.
_SE_ARGS = frozenset({"vce", "vcov", "robust", "se_type", "cluster", "clustervars"})

#: Entry points whose ``cluster=`` is not a standard-error choice.
_CLUSTER_NOT_SE = frozenset({"cross_validate"})

_RTOL = 1e-10


def _public(name: str) -> Any:
    import statspai

    fn = getattr(statspai, name, None)
    if fn is None or not callable(fn):  # pragma: no cover - registry names
        raise MethodIncompatibility(
            f"sp.vcov: sp.{name} is not a public function.",
            recovery_hint="Refit the model with vce= directly.",
        )
    return fn


def _accepts_vce(fn: Any) -> bool:
    import inspect

    try:
        params = inspect.signature(fn).parameters
    except (TypeError, ValueError):  # pragma: no cover
        return False
    aliases = getattr(fn, "__statspai_aliases__", {}) or {}
    return "vce" in params or "vce" in aliases


def _se_args(fn: Any, name: str) -> frozenset:
    """Keyword names on ``fn`` that carry the standard-error choice."""
    out = set(_SE_ARGS)
    info = getattr(fn, "__statspai_vce__", None) or {}
    out.update(info.get("native_args", ()))
    if name in _CLUSTER_NOT_SE:
        out.discard("cluster")
    aliases = getattr(fn, "__statspai_aliases__", {}) or {}
    out.update(alias for alias, target in aliases.items() if target in out)
    return frozenset(out)


def _estimates(result: Any) -> pd.Series:
    """Point estimates as a Series, the quantity the replay must reproduce."""
    params = getattr(result, "params", None)
    if isinstance(params, pd.Series) and len(params):
        return params.astype(float)
    est = getattr(result, "estimate", None)
    if est is not None and np.ndim(est) == 0:
        return pd.Series([float(est)], index=["estimate"])
    raise MethodIncompatibility(
        f"sp.vcov: a {type(result).__name__} reports neither params nor a "
        "scalar estimate, so a replay cannot be checked.",
        recovery_hint="Refit the model with vce= directly.",
    )


def _table(result: Any) -> pd.DataFrame:
    """``term, estimate, se, ci_lower, ci_upper, pvalue`` for one result."""
    params = getattr(result, "params", None)
    if isinstance(params, pd.Series) and len(params):
        idx = params.index

        def col(attr: str) -> np.ndarray:
            value = getattr(result, attr, None)
            if isinstance(value, pd.Series):
                return np.asarray(value.reindex(idx).to_numpy(float), dtype=float)
            return np.full(len(idx), np.nan)

        return pd.DataFrame(
            {
                "term": list(idx),
                "estimate": params.to_numpy(float),
                "se": col("std_errors"),
                "ci_lower": col("conf_int_lower"),
                "ci_upper": col("conf_int_upper"),
                "pvalue": col("pvalues"),
            }
        )
    est = float(_estimates(result).iloc[0])
    ci = getattr(result, "ci", None)
    lo, hi = (
        (float(ci[0]), float(ci[1]))
        if ci is not None and len(ci) == 2
        else (
            np.nan,
            np.nan,
        )
    )
    se = getattr(result, "se", None)
    pv = getattr(result, "pvalue", None)
    return pd.DataFrame(
        {
            "term": ["estimate"],
            "estimate": [est],
            "se": [float(se) if se is not None else np.nan],
            "ci_lower": [lo],
            "ci_upper": [hi],
            "pvalue": [float(pv) if pv is not None else np.nan],
        }
    )


def _same(a: pd.Series, b: pd.Series) -> bool:
    if len(a) != len(b) or list(a.index) != list(b.index):
        return False
    x, y = a.to_numpy(float), b.to_numpy(float)
    both_nan = np.isnan(x) & np.isnan(y)
    close = np.isclose(
        x,
        y,
        rtol=_RTOL,
        atol=_RTOL
        * max(1.0, float(np.nanmax(np.abs(y))) if np.isfinite(y).any() else 1.0),
    )
    return bool(np.all(both_nan | close))


def _keywords(
    fn: Any, args: Tuple[Any, ...], kwargs: Dict[str, Any]
) -> Tuple[Tuple[Any, ...], Dict[str, Any]]:
    """Move positional arguments to keywords so SE arguments can be replaced."""
    import inspect

    if not args:
        return (), dict(kwargs)
    params = list(inspect.signature(fn).parameters.values())
    if any(p.kind is p.VAR_POSITIONAL for p in params[: len(args)]):
        return args, dict(kwargs)
    named = {p.name: a for p, a in zip(params, args)}
    named.update(kwargs)
    return (), named


class _Replayer:
    """Re-runs one recorded call with ``vce=`` swapped, after checking it."""

    def __init__(self, result: Any, data: Optional[pd.DataFrame]) -> None:
        record = get_call_record(result)
        if record is None:
            raise MethodIncompatibility(
                f"sp.vcov: this {type(result).__name__} carries no record of "
                "the call that produced it (it came from an estimator without "
                "vce= support, or was built by hand).",
                recovery_hint="Refit the model with vce= directly.",
            )
        self.record = record
        self.name = record.function
        self.fn = _public(self.name)
        if not _accepts_vce(self.fn):
            raise MethodIncompatibility(
                f"sp.vcov: sp.{self.name} has no vce= argument.",
                recovery_hint=f"See sp.describe_function('{self.name}') for "
                "the standard errors it offers.",
            )
        try:
            args, kwargs = record.restore(data)
        except LookupError as exc:
            raise MethodIncompatibility(
                f"sp.vcov: {exc}", recovery_hint="Pass data=<the frame you fitted on>."
            ) from None
        self.args, self.kwargs = _keywords(self.fn, args, kwargs)
        self.original = result
        self.baseline = _estimates(result)
        self.strip = _se_args(self.fn, self.name)
        self._checked = False

    def check(self) -> None:
        if self._checked:
            return
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            replay = self.fn(*self.args, **self.kwargs)
        if not _same(_estimates(replay), self.baseline):
            raise MethodIncompatibility(
                f"sp.vcov: re-running the recorded sp.{self.name} call does "
                "not reproduce this result's estimates, so standard errors "
                "from a refit would describe a different fit. Usual causes: "
                "the data were changed after fitting, data= is not the frame "
                "that was fitted, or the estimate depends on an unseeded "
                "random draw.",
                recovery_hint="Pass the original frame as data= and a fixed "
                "seed=, or refit with vce= directly.",
                diagnostics={
                    "function": self.name,
                    "original": self.baseline.to_dict(),
                    "replay": _estimates(replay).to_dict(),
                },
            )
        self._checked = True

    def refit(self, vce: Any) -> Any:
        self.check()
        kwargs = {k: v for k, v in self.kwargs.items() if k not in self.strip}
        kwargs["vce"] = vce
        new = self.fn(*self.args, **kwargs)
        moved = not _same(_estimates(new), self.baseline)
        if moved:
            warn(
                StatsPAIWarning,
                f"sp.vcov: under vce={vce!r} sp.{self.name} returns different "
                "point estimates, not only different standard errors. The "
                "standard-error choice enters this estimator (rows with a "
                "missing cluster variable are dropped; cross-fitting folds or "
                "penalty loadings follow the clusters).",
                stacklevel=3,
                recovery_hint="Compare the estimate columns as well as the SEs.",
                diagnostics={"vce": repr(vce), "function": self.name},
            )
        info = getattr(new, "model_info", None)
        if isinstance(info, dict):
            info["vcov_refit"] = {
                "function": self.name,
                "vce": vce,
                "replay_checked": True,
                "estimates_changed": bool(moved),
            }
        return new


def vcov(result: Any, vce: Any, *, data: Optional[pd.DataFrame] = None) -> Any:
    """Re-report a fitted result under another standard-error choice.

    Re-runs the call that produced ``result`` with ``vce=`` replaced and
    returns the new result. Every other argument -- weights, fixed effects,
    covariates, seeds -- is the one you passed. Because the estimator itself
    computes the new standard errors, they are exactly what
    ``sp.<estimator>(..., vce=vce)`` returns, including its small-sample
    factors and its refusals (an estimator that does not implement a kind
    raises rather than substituting another).

    Parameters
    ----------
    result : fitted result
        Any result from an estimator that accepts ``vce=``.
    vce : str
        Stata-style request, e.g. ``"robust"``, ``"hc3"``,
        ``"cluster state"``, ``"cluster state year"`` (where multiway
        clustering is offered), ``"bootstrap"``. Which kinds a given
        estimator accepts is listed in ``sp.function_schema(name)``.
    data : DataFrame, optional
        The frame the model was fitted on. Needed only when that frame no
        longer exists (results hold it weakly).

    Returns
    -------
    result
        A new result of the same type. ``model_info['vcov_refit']`` records
        the request and whether the point estimates moved.

    Raises
    ------
    MethodIncompatibility
        When the result has no call record, the estimator has no ``vce=``,
        the requested kind is not available for it, or replaying the
        original call does not reproduce the original estimates.

    Notes
    -----
    The original call is replayed once, unchanged, before ``vce=`` is
    swapped; its estimates must match the result's to a relative ``1e-10``.
    The point estimates may legitimately move under the new request when
    the estimator uses the cluster variable for more than the variance
    (rows with a missing cluster value are dropped; cross-fitting folds and
    cluster-Lasso loadings follow the clusters); a warning says so.

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> df = pd.DataFrame({"x": rng.normal(size=200),
    ...                    "g": rng.integers(0, 40, 200)})
    >>> df["y"] = 1 + 2 * df["x"] + rng.normal(size=200)
    >>> fit = sp.regress("y ~ x", data=df)
    >>> clustered = sp.vcov(fit, "cluster g")
    >>> direct = sp.regress("y ~ x", data=df, vce="cluster g")
    >>> bool(np.allclose(clustered.std_errors, direct.std_errors, rtol=0, atol=0))
    True
    """
    return _Replayer(result, data).refit(vce)


def vcov_compare(
    result: Any,
    vce: Union[str, Sequence[Any]],
    *,
    data: Optional[pd.DataFrame] = None,
    terms: Optional[Iterable[str]] = None,
) -> pd.DataFrame:
    """Standard errors of one fit under several ``vce=`` choices, side by side.

    The first block is the result as fitted; each further block re-runs the
    call with one ``vce`` (see :func:`statspai.vcov`). A request the
    estimator does not implement is reported in the ``note`` column instead
    of stopping the table.

    Parameters
    ----------
    result : fitted result
    vce : str or list
        One request or several, e.g.
        ``["robust", "cluster state", "cluster state year"]``.
    data : DataFrame, optional
        As in :func:`statspai.vcov`.
    terms : iterable of str, optional
        Keep only these coefficients.

    Returns
    -------
    DataFrame
        One row per (request, term): ``vce``, ``term``, ``estimate``,
        ``se``, ``ci_lower``, ``ci_upper``, ``pvalue``, ``se_ratio`` (to the
        fitted SE of the same term) and ``note``.

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(1)
    >>> df = pd.DataFrame({"x": rng.normal(size=300),
    ...                    "g": rng.integers(0, 30, 300)})
    >>> df["y"] = 1 + df["x"] + rng.normal(size=300)
    >>> fit = sp.regress("y ~ x", data=df)
    >>> tab = sp.vcov_compare(fit, ["hc1", "hc3", "cluster g"], terms=["x"])
    >>> list(tab["vce"])
    ['(as fitted)', 'hc1', 'hc3', 'cluster g']
    """
    requests: List[Any] = [vce] if isinstance(vce, str) else list(vce)
    replayer = _Replayer(result, data)
    base = _table(result)
    keep = list(terms) if terms is not None else None
    blocks: List[pd.DataFrame] = []

    def add(label: str, table: pd.DataFrame, note: str = "") -> None:
        t = table.copy()
        if keep is not None:
            t = t[t["term"].isin(keep)]
        t.insert(0, "vce", label)
        t["note"] = note
        blocks.append(t)

    add("(as fitted)", base)
    for request in requests:
        try:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", StatsPAIWarning)
                new = replayer.refit(request)
        except Exception as exc:  # noqa: BLE001 - reported in the row's note
            # The replay check failing is about the whole table, not a row.
            if not replayer._checked:
                raise
            empty = base.assign(
                estimate=np.nan,
                se=np.nan,
                ci_lower=np.nan,
                ci_upper=np.nan,
                pvalue=np.nan,
            )
            message = str(exc).splitlines()[0] if str(exc) else ""
            add(str(request), empty, note=f"{type(exc).__name__}: {message}")
            continue
        moved = (
            (getattr(new, "model_info", {}) or {})
            .get("vcov_refit", {})
            .get("estimates_changed", False)
        )
        add(str(request), _table(new), note="estimates changed" if moved else "")
    out = pd.concat(blocks, ignore_index=True)
    fitted_se = dict(zip(base["term"], base["se"]))
    with np.errstate(divide="ignore", invalid="ignore"):
        out["se_ratio"] = out["se"] / out["term"].map(fitted_se).astype(float)
    cols = [
        "vce",
        "term",
        "estimate",
        "se",
        "ci_lower",
        "ci_upper",
        "pvalue",
        "se_ratio",
        "note",
    ]
    return out[cols]
