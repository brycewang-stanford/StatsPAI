"""``sp.fast.event_study`` on the Phase 1+ HDFE stack.

Phase 6 deliverable. This is a homogeneous-effects event-study estimator
(post-treatment leads/lags interacted with treatment), built on top of:

* ``sp.fast.within``    — the cached HDFE residualizer.
* ``sp.fast.i``         — event-time dummies with an explicit reference.
* ``sp.fast.crve``      — cluster-robust standard errors.

It is **not** a replacement for ``sp.callaway_santanna`` /
``sp.sun_abraham`` / ``sp.borusyak_jaravel_spiess`` — those handle
heterogeneous treatment effects with the appropriate group-specific
weights, and they are unchanged. ``sp.fast.event_study`` ships the
TWFE-style estimator on the new Rust path, suitable for designs with
homogeneous effects and as a baseline against which the heterogeneous
estimators can be diff-checked.

Design assumptions
------------------

* Two-way fixed effects: unit + time.
* Single binary treatment turning on at unit-specific event time.
* Symmetric event window around the event time; missing event-times
  outside the window are pooled into never-event indicators.
* Reference period defaults to ``t = -1`` (one period before treatment)
  per the canonical event-study convention (Borusyak-Jaravel-Spiess
  2024 §3.1).

References
----------
de Chaisemartin, C., D'Haultfœuille, X. (2020). Two-way fixed effects
estimators with heterogeneous treatment effects. AER 110(9): 2964–2996.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional, Tuple, cast

import numpy as np
import pandas as pd

from .._result_serialize import ResultProtocolMixin
from ..did.event_study import _build_bins, _resolve_ref_set
from ..exceptions import DataInsufficient, MethodIncompatibility, NumericalInstability
from ._result_protocol import jsonable as _jsonable
from ._result_protocol import tidy_records as _tidy_records
from .inference import crve as _crve
from .within import within as _within


@dataclass
class EventStudyResult(ResultProtocolMixin):
    """Outcome of :func:`event_study`."""

    formula: str
    event_times: np.ndarray
    coefs: np.ndarray
    ses: np.ndarray
    n_obs: int
    n_kept: int
    n_clusters: Optional[int]
    cluster_var: Optional[str]
    reference_event_time: int
    #: ``(start, end)`` for each estimated coefficient. Point bins have
    #: ``start == end``; with ``bin_width`` set they span several periods.
    bins: list = field(default_factory=list)
    #: ``(start, end)`` for each omitted reference bin.
    reference_bins: list = field(default_factory=list)
    bin_width: Optional[int] = None

    def tidy(self) -> pd.DataFrame:
        out = pd.DataFrame(
            {
                "event_time": self.event_times,
                "Estimate": self.coefs,
                "Std. Error": self.ses,
                "ci_lower": self.coefs - 1.96 * self.ses,
                "ci_upper": self.coefs + 1.96 * self.ses,
            }
        )
        if self.bins and any(b[0] != b[1] for b in self.bins):
            out.insert(1, "bin_start", [b[0] for b in self.bins])
            out.insert(2, "bin_end", [b[1] for b in self.bins])
            out.insert(3, "bin_label", [f"[{b[0]}, {b[1]}]" for b in self.bins])
        return out

    def plot(self, ax: Any = None) -> Any:  # pragma: no cover  - cosmetic
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:
            raise RuntimeError("matplotlib is required for .plot()") from exc
        if ax is None:
            _, ax = plt.subplots(figsize=(8, 4))
        td = self.tidy()
        ax.errorbar(
            td["event_time"],
            td["Estimate"],
            yerr=1.96 * td["Std. Error"],
            fmt="o-",
            capsize=3,
        )
        ax.axhline(0, color="black", linewidth=0.5)
        ax.axvline(-0.5, color="grey", linestyle="--", linewidth=0.5)
        ax.set_xlabel("Event time")
        ax.set_ylabel("Estimated effect")
        ax.set_title("Event study (TWFE)")
        return ax

    def summary(self) -> str:
        table = str(
            self.tidy().to_string(
                index=False,
                float_format=lambda v: f"{v:.4f}",
            )
        )
        return (
            f"sp.fast.event_study  |  N={self.n_obs:,}, kept={self.n_kept:,}, "
            f"event_times={list(self.event_times)}\n" + table
        )

    def to_dict(self) -> dict[str, Any]:
        """Lossless JSON-safe payload for the fast TWFE event study."""
        payload = {
            "kind": "fast_event_study_result",
            "model": "twfe_event_study",
            "formula": self.formula,
            "n_obs": self.n_obs,
            "n_kept": self.n_kept,
            "n_clusters": self.n_clusters,
            "cluster_var": self.cluster_var,
            "reference_event_time": self.reference_event_time,
            "event_times": self.event_times,
            "coefficients": self.coefs,
            "standard_errors": self.ses,
            "tidy": _tidy_records(self.tidy()),
        }
        return cast(dict[str, Any], _jsonable(payload))

    def to_agent_summary(self, *, max_event_times: int = 20) -> dict[str, Any]:
        """Bounded agent-facing summary for fast TWFE event-study results."""
        n_terms = len(self.event_times)
        limit = max(int(max_event_times), 0)
        rows = _tidy_records(self.tidy().head(limit))
        payload = {
            "kind": "fast_event_study_agent_summary",
            "model": "twfe_event_study",
            "formula": self.formula,
            "n_obs": self.n_obs,
            "n_kept": self.n_kept,
            "n_clusters": self.n_clusters,
            "cluster_var": self.cluster_var,
            "reference_event_time": self.reference_event_time,
            "event_times": rows,
            "n_event_times": n_terms,
            "truncated_event_times": max(n_terms - limit, 0),
        }
        return cast(dict[str, Any], _jsonable(payload))


def event_study(
    data: pd.DataFrame,
    *,
    y: str,
    unit: str,
    time: str,
    event_time: str,
    window: Optional[Tuple[int, int]] = None,
    reference: Any = -1,
    cluster: Optional[str] = None,
    drop_singletons: bool = True,
    bin_width: Optional[int] = None,
) -> EventStudyResult:
    """Two-way-FE event study on the Phase 1+ HDFE stack.

    Parameters
    ----------
    data : pd.DataFrame
    y : str
        Outcome column name.
    unit : str
        Unit identifier (e.g. firm id). Absorbed as a fixed effect.
    time : str
        Time identifier (e.g. year). Absorbed as a fixed effect.
    event_time : str
        Per-row column giving each row's relative time vs. its unit's
        event date. For never-treated units pass NaN; for treated units
        pass an integer offset (negative = pre-treatment, 0 = treatment
        period, positive = post-treatment).
    window : (lo, hi), optional
        Truncate event-time dummies to ``[lo, hi]``. Outside the window,
        rows are kept (so the unit FE still soaks up their level) but
        not given individual coefficients.
    reference : int, (str, int) or sequence of int, default -1
        Omitted reference category. Mirrors ``sp.event_study``:

        * ``int`` -- a single omitted event time (classic).
        * ``(op, bound)`` -- an interval, e.g. ``("<=", -50)`` or
          ``(">=", 20)``; every event time in range satisfying the
          comparison is pooled into the omitted base.
        * sequence of int -- an explicit omitted span, e.g. ``[-3, -2, -1]``.
    cluster : str, optional
        Column name to cluster standard errors on. Default: cluster on
        ``unit`` (the conventional choice for panel DiD).
    drop_singletons : bool
    bin_width : int, optional
        Group event times into bins of this width instead of one
        coefficient per period. Bins are anchored at the treatment
        boundary (``-1`` and ``0`` never share a bin), matching
        ``sp.event_study(bin_width=...)``. ``event_times`` then reports
        each bin's left edge; ``bins`` reports the ``(start, end)`` pairs.

    Returns
    -------
    EventStudyResult
    """
    if not isinstance(data, pd.DataFrame):
        raise MethodIncompatibility("fast.event_study: data must be a DataFrame")
    if len(data) < 1:
        raise DataInsufficient("fast.event_study: data must contain at least one row")
    roles = {"y": y, "unit": unit, "time": time, "event_time": event_time}
    bad_roles = [
        name for name, col in roles.items() if not isinstance(col, str) or not col
    ]
    if bad_roles:
        raise MethodIncompatibility(
            "fast.event_study: column arguments must be non-empty strings: "
            f"{bad_roles}"
        )
    missing = [col for col in roles.values() if col not in data.columns]
    if missing:
        raise MethodIncompatibility(
            f"fast.event_study: data missing columns: {missing}"
        )

    cluster_col = cluster if cluster is not None else unit
    if not isinstance(cluster_col, str) or not cluster_col:
        raise MethodIncompatibility(
            "fast.event_study: cluster must be a non-empty string column name"
        )
    if cluster_col not in data.columns:
        raise MethodIncompatibility(
            f"fast.event_study: data missing cluster column {cluster_col!r}"
        )
    # A plain-int reference with no binning keeps the historical code path
    # byte-for-byte; the richer specs engage the shared resolver below.
    simple_reference = (
        isinstance(reference, (int, np.integer))
        and not isinstance(reference, (bool, np.bool_))
        and bin_width is None
    )
    if simple_reference:
        reference = int(reference)
    elif isinstance(reference, (bool, np.bool_)) or not isinstance(
        reference, (int, np.integer, tuple, list, set, np.ndarray)
    ):
        # Scalars that are not integers (e.g. -1.5, None, "x") keep the
        # original error; only tuple/list/set forms reach the richer resolver.
        raise MethodIncompatibility(
            "fast.event_study: reference must be an integer event-time offset, "
            "an interval such as ('<=', -50), or a span such as [-3, -2, -1]"
        )
    if window is not None:
        try:
            lo, hi = window
        except (TypeError, ValueError) as exc:
            raise MethodIncompatibility(
                "fast.event_study: window must be a (lo, hi) pair"
            ) from exc
        if isinstance(window, (str, bytes)):
            raise MethodIncompatibility(
                "fast.event_study: window must be a (lo, hi) pair"
            )
        if not isinstance(lo, (int, np.integer)) or not isinstance(
            hi, (int, np.integer)
        ):
            raise MethodIncompatibility(
                "fast.event_study: window bounds must be integer " "event-time offsets"
            )
        lo = int(lo)
        hi = int(hi)
        if lo > hi:
            raise MethodIncompatibility(
                "fast.event_study: window lower bound must be <= upper bound"
            )
        window = (lo, hi)

    df = data.copy()
    n_obs = len(df)
    y_values = df[y].to_numpy(dtype=np.float64)
    if not np.isfinite(y_values).all():
        raise MethodIncompatibility(
            f"fast.event_study: outcome column {y!r} has non-finite values"
        )

    # Build event-time dummies (one-hot), masking NaN rows out.
    try:
        et = df[event_time].to_numpy(dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise MethodIncompatibility(
            f"fast.event_study: event_time column {event_time!r} must be "
            "numeric with NaN for never-treated rows"
        ) from exc
    if np.isinf(et).any():
        raise MethodIncompatibility(
            f"fast.event_study: event_time column {event_time!r} contains "
            "infinite values; use NaN only for never-treated rows"
        )
    finite = np.isfinite(et)
    rounded_et = np.rint(et[finite])
    if not np.allclose(et[finite], rounded_et, atol=1e-10, rtol=0.0):
        raise MethodIncompatibility(
            f"fast.event_study: event_time column {event_time!r} must contain "
            "integer event-time offsets; non-integer finite values would be "
            "silently truncated"
        )
    # cast finite rows to int for dummy labels; non-finite get a sentinel
    et_int = np.full(et.shape, np.iinfo(np.int64).min, dtype=np.int64)
    et_int[finite] = rounded_et.astype(np.int64)
    if window is not None:
        lo, hi = window
        et_int = np.where(
            finite & (et_int >= lo) & (et_int <= hi),
            et_int,
            np.iinfo(np.int64).min,
        )
        finite &= (et_int >= lo) & (et_int <= hi)

    # Use pd.Categorical so we control the level set
    if simple_reference:
        levels = sorted({v for v in et_int[finite] if v != reference})
        bins: list[tuple[int, int]] = [(int(lv), int(lv)) for lv in levels]
        ref_bins: list[tuple[int, int]] = []
    else:
        observed = [int(v) for v in et_int[finite]]
        if not observed:
            raise DataInsufficient(
                "fast.event_study: no finite event times after filtering; "
                "check event_time / window"
            )
        if window is not None:
            lo_r, hi_r = window
        else:
            lo_r, hi_r = min(observed), max(observed)
        ref_times, ref_canonical = _resolve_ref_set(reference, int(lo_r), int(hi_r))
        bin_of = _build_bins(int(lo_r), int(hi_r), bin_width)
        ref_set = set(ref_times)
        members: dict[tuple[int, int], list[int]] = {}
        for t in range(int(lo_r), int(hi_r) + 1):
            members.setdefault(bin_of[t], []).append(t)
        ref_bins = []
        for b, mem in members.items():
            in_ref = [t for t in mem if t in ref_set]
            if not in_ref:
                continue
            if len(in_ref) != len(mem):
                raise MethodIncompatibility(
                    f"fast.event_study: the reference span {sorted(ref_set)} "
                    f"cuts bin [{b[0]}, {b[1]}] in half (it omits "
                    f"{sorted(in_ref)} but keeps {sorted(set(mem) - ref_set)}). "
                    "A partially-omitted bin has no coherent interpretation. "
                    f"Align the reference to the bin edges, e.g. "
                    f"reference=('<=', {b[1]}), or drop bin_width."
                )
            ref_bins.append(b)
        bins = sorted(b for b in members if b not in set(ref_bins))
        # Keep only bins that actually carry observations.
        observed_set = set(observed)
        bins = [
            b for b in bins if any(t in observed_set for t in range(b[0], b[1] + 1))
        ]
        levels = [b[0] for b in bins]
        del ref_canonical
    if not levels:
        raise DataInsufficient(
            "fast.event_study: no event-time dummies after filtering; "
            "check event_time / window"
        )

    def _dummy_name(start: int, end: int) -> str:
        # Point bins keep the historical ``et_<k>`` name.
        return f"et_{start}" if start == end else f"et_{start}_{end}"

    dummies = pd.DataFrame(
        {
            _dummy_name(b[0], b[1]): (
                finite & (et_int >= b[0]) & (et_int <= b[1])
            ).astype(np.float64)
            for b in bins
        },
        index=df.index,
    )
    dummy_cols = list(dummies.columns)
    df_aug = pd.concat([df, dummies], axis=1)

    # FE residualisation
    wt = _within(df_aug, fe=[unit, time], drop_singletons=drop_singletons)
    y_dem, _ = wt.transform(y_values)
    X_dem = wt.transform_columns(df_aug, dummy_cols).to_numpy()

    # OLS on residualised
    XtX = X_dem.T @ X_dem
    Xty = X_dem.T @ y_dem
    try:
        bread = np.linalg.inv(XtX)
    except np.linalg.LinAlgError as exc:
        raise NumericalInstability(
            "event_study normal equations are singular. Likely cause: "
            "event-time dummies are perfectly collinear with the absorbed "
            "unit/time fixed effects after filtering."
        ) from exc
    beta = bread @ Xty
    resid = y_dem - X_dem @ beta

    # Cluster-robust SE on the residualised system. The CR1 small-sample
    # factor must charge the absorbed FE rank against residual DOF — same
    # convention as ``reghdfe`` / ``fixest``. Without it, SEs are
    # systematically too small (the FE rank is "free" parameters).
    cluster_arr = df_aug.loc[wt.keep_mask, cluster_col].to_numpy()
    fe_dof = sum(int(g) - 1 for g in wt.n_fe)
    V = _crve(
        X_dem,
        resid,
        cluster_arr,
        bread=bread,
        type="cr1",
        extra_df=fe_dof,
    )
    se = np.sqrt(np.diag(V))

    formula = f"{y} ~ event_time | {unit} + {time}  " f"(cluster: {cluster_col})"
    return EventStudyResult(
        formula=formula,
        event_times=np.asarray(levels),
        coefs=beta,
        ses=se,
        n_obs=n_obs,
        n_kept=int(wt.n_kept),
        n_clusters=int(np.unique(cluster_arr).size),
        cluster_var=cluster_col,
        reference_event_time=(
            reference if simple_reference else int(min(b[0] for b in ref_bins))
        ),
        bins=[(int(b[0]), int(b[1])) for b in bins],
        reference_bins=[(int(b[0]), int(b[1])) for b in ref_bins],
        bin_width=None if bin_width is None else int(bin_width),
    )


__all__ = ["event_study", "EventStudyResult"]
