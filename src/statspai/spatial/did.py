"""
Spatial Difference-in-Differences (Spatial DiD).

This module implements the local-spillover DiD design of Delgado and
Florax (2015) with StatsPAI's modern result surface: diagnostics,
event-study support, plotting, and report exports.

The baseline specification is a two-way fixed-effect outcome equation
augmented with the spatial lag of treatment:

.. math::

    Y_{it} = \\alpha_i + \\gamma_t + \\tau D_{it}
             + \\theta W D_{it} + X_{it}\\beta + \\varepsilon_{it}.

``direct_effect`` estimates the own-unit treatment effect, while
``spillover_effect`` estimates the effect of nearby units' treatment
exposure under a user-supplied spatial weights matrix.

References
----------
Delgado, M. S. & Florax, R. J. G. M. (2015).
"Difference-in-differences techniques for spatial data: local
autocorrelation and spatial interaction." *Economics Letters*, 137,
123-126.

Dubé, J., Legros, D., & Thériault, M. (2014).
"A spatial difference-in-differences estimator to evaluate the effect
of change in public mass transit systems on house prices."
*Transportation Research Part B*, 64, 24-40.

Roth, J., Sant'Anna, P. H. C., Bilinski, A., & Poe, J. (2023).
"What's trending in difference-in-differences? A synthesis of the
recent econometrics literature." *Journal of Econometrics*, 235(2),
2218-2244.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from ..core.results import SummaryText, _to_jsonable
from ..exceptions import DataInsufficient, MethodIncompatibility

_EPS = 1e-12
_EARTH_RADIUS_KM = 6371.0
_SE_TYPES = {"cluster", "robust", "conley"}
_CONLEY_KERNELS = {"uniform", "bartlett"}


def _column_list(columns: Optional[Sequence[str] | str], label: str) -> list[str]:
    """Normalize optional column-name input to a validated list."""
    if columns is None:
        return []
    if isinstance(columns, str):
        out = [columns]
    else:
        try:
            out = list(columns)
        except TypeError as exc:
            raise MethodIncompatibility(
                f"`{label}` must be a column name or a sequence of column names.",
                recovery_hint=f"Pass {label}='x' or {label}=['x1', 'x2'].",
                diagnostics={"argument": label, "type": type(columns).__name__},
            ) from exc
    bad = [c for c in out if not isinstance(c, str) or not c]
    if bad:
        raise MethodIncompatibility(
            f"`{label}` must contain non-empty column-name strings.",
            recovery_hint=f"Check the `{label}` argument before calling spatial_did.",
            diagnostics={"argument": label, "bad_values": [repr(c) for c in bad]},
        )
    return out


def _require_column_name(name: str, label: str) -> None:
    if not isinstance(name, str) or not name:
        raise MethodIncompatibility(
            f"`{label}` must be a non-empty column-name string.",
            recovery_hint=(
                f"Pass the name of an existing DataFrame column for `{label}`."
            ),
            diagnostics={"argument": label, "value": repr(name)},
        )


@dataclass
class SpatialDiDResult:
    """Result object for :func:`spatial_did`.

    The object intentionally exposes both a spatial-DiD-specific surface
    (``direct_effect``, ``spillover_effect``, ``total_effect``) and a
    broom/modelsummary-compatible surface (``tidy()``, ``glance()``,
    ``params``, ``std_errors``).

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> n_units, n_time = 12, 6
    >>> W = np.zeros((n_units, n_units))  # ring spatial weights over units
    >>> for i in range(n_units):
    ...     W[i, (i + 1) % n_units] = 1
    ...     W[i, (i - 1) % n_units] = 1
    >>> rows = []
    >>> for u in range(n_units):
    ...     fe = rng.normal()
    ...     for t in range(n_time):
    ...         d = 1 if (u < 6 and t >= 3) else 0  # half treated from t>=3
    ...         y = 1.0 + fe + 0.3 * t + 1.5 * d + rng.normal(scale=0.3)
    ...         rows.append({"unit": u, "time": t, "treat": d, "y": y})
    >>> df = pd.DataFrame(rows)
    >>> res = sp.spatial_did(df, y="y", treat="treat", unit="unit",
    ...                     time="time", W=W)
    >>> isinstance(res, sp.SpatialDiDResult)
    True
    >>> res.n_obs
    72
    """

    direct_effect: float
    spillover_effect: float
    se_direct: float
    se_spillover: float
    ci_direct: tuple
    ci_spillover: tuple
    pvalue_direct: float
    pvalue_spillover: float
    coefficients: pd.DataFrame
    n_obs: int
    total_effect: float = np.nan
    se_total: float = np.nan
    ci_total: tuple = (np.nan, np.nan)
    pvalue_total: float = np.nan
    alpha: float = 0.05
    se_type: str = "cluster"
    vcov: Optional[np.ndarray] = None
    detail: Dict[str, Any] = field(default_factory=dict)
    model_info: Dict[str, Any] = field(default_factory=dict)
    data_info: Dict[str, Any] = field(default_factory=dict)

    @property
    def params(self) -> pd.Series:
        coef = self.coefficients
        est_col = "estimate" if "estimate" in coef.columns else "coef"
        return pd.Series(coef[est_col].to_numpy(), index=coef["variable"].astype(str))

    @property
    def std_errors(self) -> pd.Series:
        return pd.Series(
            self.coefficients["se"].to_numpy(),
            index=self.coefficients["variable"].astype(str),
        )

    @property
    def tvalues(self) -> pd.Series:
        return self.params / self.std_errors.replace(0, np.nan)

    @property
    def pvalues(self) -> pd.Series:
        if "pvalue" in self.coefficients.columns:
            return pd.Series(
                self.coefficients["pvalue"].to_numpy(),
                index=self.coefficients["variable"].astype(str),
            )
        return pd.Series(
            2 * stats.norm.sf(np.abs(self.tvalues.to_numpy())),
            index=self.params.index,
        )

    @property
    def conf_int_lower(self) -> pd.Series:
        if "ci_lower" in self.coefficients.columns:
            return pd.Series(
                self.coefficients["ci_lower"].to_numpy(),
                index=self.coefficients["variable"].astype(str),
            )
        return self.params - stats.norm.ppf(1 - self.alpha / 2) * self.std_errors

    @property
    def conf_int_upper(self) -> pd.Series:
        if "ci_upper" in self.coefficients.columns:
            return pd.Series(
                self.coefficients["ci_upper"].to_numpy(),
                index=self.coefficients["variable"].astype(str),
            )
        return self.params + stats.norm.ppf(1 - self.alpha / 2) * self.std_errors

    @property
    def diagnostics(self) -> Dict[str, Any]:
        diagnostics = self.detail.get("diagnostics", {})
        return diagnostics if isinstance(diagnostics, dict) else {}

    def summary(self) -> str:
        lines = [
            "=" * 78,
            "  Spatial Difference-in-Differences",
            "=" * 78,
            "",
            f"  Direct effect    : {self.direct_effect: .6f}"
            f"  (SE={self.se_direct:.6f}, p={self.pvalue_direct:.4f})",
            f"  Spillover effect : {self.spillover_effect: .6f}"
            f"  (SE={self.se_spillover:.6f}, p={self.pvalue_spillover:.4f})",
            f"  Total effect     : {self.total_effect: .6f}"
            f"  (SE={self.se_total:.6f}, p={self.pvalue_total:.4f})",
            "",
            f"  Direct {int((1 - self.alpha) * 100)}% CI    : "
            f"[{self.ci_direct[0]:.6f}, {self.ci_direct[1]:.6f}]",
            f"  Spillover {int((1 - self.alpha) * 100)}% CI : "
            f"[{self.ci_spillover[0]:.6f}, {self.ci_spillover[1]:.6f}]",
            f"  Total {int((1 - self.alpha) * 100)}% CI     : "
            f"[{self.ci_total[0]:.6f}, {self.ci_total[1]:.6f}]",
            "",
            "-" * 78,
            f"  Observations: {self.n_obs:,}",
            f"  SE type     : {self.se_type}",
        ]

        diagnostics = self.diagnostics
        if diagnostics:
            lines.extend(["", "  Diagnostics", "  " + "-" * 24])
            for key in (
                "n_units",
                "n_periods",
                "treated_observation_share",
                "control_spillover_exposure_share",
                "corr_treat_spillover",
                "condition_number",
                "mean_residual_moran_i",
            ):
                if key in diagnostics:
                    val = diagnostics[key]
                    label = key.replace("_", " ")
                    if isinstance(val, (float, np.floating)):
                        lines.append(f"  {label:34s}: {val:.4f}")
                    else:
                        lines.append(f"  {label:34s}: {val}")

        warnings = self.detail.get("warnings", [])
        if warnings:
            lines.extend(["", "  Warnings", "  " + "-" * 24])
            for warning in warnings:
                lines.append(f"  - {warning}")

        es = self.detail.get("event_study")
        if isinstance(es, pd.DataFrame) and len(es) > 0:
            lines.extend(["", "-" * 78, "  Spatial Event Study", "-" * 78])
            compact = es[
                [
                    "effect",
                    "relative_time",
                    "estimate",
                    "se",
                    "ci_lower",
                    "ci_upper",
                    "pvalue",
                ]
            ].copy()
            lines.append(compact.to_string(index=False, float_format="%.4f"))

        pt = self.detail.get("pretrend_test")
        if isinstance(pt, dict) and pt:
            lines.extend(["", "  Pre-trend tests", "  " + "-" * 24])
            for effect, out in pt.items():
                if isinstance(out, dict):
                    lines.append(
                        f"  {effect:10s}: chi2({out['df']})="
                        f"{out['statistic']:.4f}, p={out['pvalue']:.4f}"
                    )

        lines.extend(["=" * 78, "* p-values use the selected covariance estimator."])
        return SummaryText("\n".join(lines))

    def tidy(
        self,
        conf_level: float = 0.95,
        include_event_study: bool = True,
    ) -> pd.DataFrame:
        alpha = 1 - conf_level
        crit = stats.norm.ppf(1 - alpha / 2)
        rows = [
            {
                "term": "direct",
                "estimate": self.direct_effect,
                "std_error": self.se_direct,
                "statistic": (
                    self.direct_effect / self.se_direct
                    if self.se_direct > 0
                    else np.nan
                ),
                "p_value": self.pvalue_direct,
                "conf_low": self.direct_effect - crit * self.se_direct,
                "conf_high": self.direct_effect + crit * self.se_direct,
                "type": "effect",
            },
            {
                "term": "spillover",
                "estimate": self.spillover_effect,
                "std_error": self.se_spillover,
                "statistic": (
                    self.spillover_effect / self.se_spillover
                    if self.se_spillover > 0
                    else np.nan
                ),
                "p_value": self.pvalue_spillover,
                "conf_low": self.spillover_effect - crit * self.se_spillover,
                "conf_high": self.spillover_effect + crit * self.se_spillover,
                "type": "effect",
            },
            {
                "term": "total",
                "estimate": self.total_effect,
                "std_error": self.se_total,
                "statistic": (
                    self.total_effect / self.se_total if self.se_total > 0 else np.nan
                ),
                "p_value": self.pvalue_total,
                "conf_low": self.total_effect - crit * self.se_total,
                "conf_high": self.total_effect + crit * self.se_total,
                "type": "effect",
            },
        ]

        if include_event_study:
            es = self.detail.get("event_study")
            if isinstance(es, pd.DataFrame) and len(es) > 0:
                for _, r in es.iterrows():
                    rows.append(
                        {
                            "term": f"{r['effect']}_event_{int(r['relative_time']):+d}",
                            "estimate": r["estimate"],
                            "std_error": r["se"],
                            "statistic": (
                                r["estimate"] / r["se"] if r["se"] > 0 else np.nan
                            ),
                            "p_value": r["pvalue"],
                            "conf_low": r["estimate"] - crit * r["se"],
                            "conf_high": r["estimate"] + crit * r["se"],
                            "type": "event_study",
                        }
                    )

        return pd.DataFrame(rows)

    def glance(self) -> pd.DataFrame:
        row = {
            "method": "Spatial Difference-in-Differences",
            "nobs": int(self.n_obs),
            "se_type": self.se_type,
            "direct_effect": self.direct_effect,
            "spillover_effect": self.spillover_effect,
            "total_effect": self.total_effect,
            "direct_p_value": self.pvalue_direct,
            "spillover_p_value": self.pvalue_spillover,
            "total_p_value": self.pvalue_total,
        }
        for key, val in self.diagnostics.items():
            if isinstance(val, (int, float, str, bool, np.integer, np.floating)):
                row[key] = val
        return pd.DataFrame([row])

    def to_dataframe(self, **kwargs: Any) -> pd.DataFrame:
        return self.tidy(**kwargs)

    def to_dict(
        self, *, detail: str = "standard", detail_head: int = 20
    ) -> Dict[str, Any]:
        if detail not in {"minimal", "standard", "agent"}:
            raise MethodIncompatibility(
                "detail must be 'minimal', 'standard', or 'agent'",
                recovery_hint=(
                    "Use detail='minimal', detail='standard', or detail='agent'."
                ),
                diagnostics={"detail": detail},
            )
        out: Dict[str, Any] = {
            "method": "Spatial Difference-in-Differences",
            "direct_effect": _to_jsonable(self.direct_effect),
            "spillover_effect": _to_jsonable(self.spillover_effect),
            "total_effect": _to_jsonable(self.total_effect),
            "se_direct": _to_jsonable(self.se_direct),
            "se_spillover": _to_jsonable(self.se_spillover),
            "se_total": _to_jsonable(self.se_total),
            "pvalue_direct": _to_jsonable(self.pvalue_direct),
            "pvalue_spillover": _to_jsonable(self.pvalue_spillover),
            "pvalue_total": _to_jsonable(self.pvalue_total),
            "ci_direct": _to_jsonable(self.ci_direct),
            "ci_spillover": _to_jsonable(self.ci_spillover),
            "ci_total": _to_jsonable(self.ci_total),
            "alpha": _to_jsonable(self.alpha),
            "n_obs": _to_jsonable(self.n_obs),
            "se_type": self.se_type,
        }
        if detail == "minimal":
            return out
        out["diagnostics"] = _to_jsonable(self.diagnostics)
        out["coefficients"] = _to_jsonable(
            self.coefficients.head(detail_head).to_dict(orient="records")
        )
        es = self.detail.get("event_study")
        if isinstance(es, pd.DataFrame):
            out["event_study"] = _to_jsonable(
                es.head(detail_head).to_dict(orient="records")
            )
        if detail == "agent":
            out["warnings"] = _to_jsonable(self.detail.get("warnings", []))
            out["pretrend_test"] = _to_jsonable(self.detail.get("pretrend_test", {}))
            out["next_steps"] = [
                {
                    "action": "Inspect spillover exposure support",
                    "reason": "Control units with positive WD identify spillover effects.",
                    "priority": "high",
                },
                {
                    "action": "Run spatial event-study or pre-trend checks",
                    "reason": "Modern DiD practice treats pre-period evidence as a diagnostic, not proof.",
                    "priority": "high",
                },
                {
                    "action": "Vary W and the spillover radius",
                    "reason": "Spatial estimands are conditional on the exposure mapping.",
                    "priority": "medium",
                },
            ]
        return out

    def to_json(self, indent: Optional[int] = None, **kwargs: Any) -> str:
        return json.dumps(self.to_dict(**kwargs), indent=indent, default=_to_jsonable)

    def to_markdown(self, path: Optional[str] = None, **kwargs: Any) -> str:
        md = str(self.tidy(**kwargs).to_markdown(index=False, floatfmt=".4f"))
        if path is not None:
            Path(path).write_text(md, encoding="utf-8")
        return md

    def to_csv(self, path: Optional[str] = None, **kwargs: Any) -> str:
        csv = str(self.tidy(**kwargs).to_csv(index=False))
        if path is not None:
            Path(path).write_text(csv, encoding="utf-8")
        return csv

    def to_latex(
        self,
        caption: Optional[str] = None,
        label: Optional[str] = None,
        path: Optional[str] = None,
        **kwargs: Any,
    ) -> str:
        table = self.tidy(**kwargs)
        latex = str(
            table.to_latex(
                index=False,
                float_format="%.4f",
                caption=caption or "Spatial difference-in-differences estimates.",
                label=label or "tab:spatial-did",
                escape=True,
            )
        )
        if path is not None:
            Path(path).write_text(latex, encoding="utf-8")
        return latex

    def to_excel(self, path: str, **kwargs: Any) -> str:
        with pd.ExcelWriter(path) as writer:
            self.tidy(**kwargs).to_excel(writer, sheet_name="effects", index=False)
            self.coefficients.to_excel(writer, sheet_name="coefficients", index=False)
            self.glance().to_excel(writer, sheet_name="diagnostics", index=False)
            es = self.detail.get("event_study")
            if isinstance(es, pd.DataFrame) and len(es) > 0:
                es.to_excel(writer, sheet_name="event_study", index=False)
        return path

    def plot(
        self,
        kind: str = "coef",
        ax: Any = None,
        figsize: tuple[int, int] = (8, 5),
        **kwargs: Any,
    ) -> Any:
        """Plot coefficient, exposure, or spatial event-study diagnostics."""
        try:
            import matplotlib.pyplot as plt
        except ImportError as exc:  # pragma: no cover
            raise ImportError("matplotlib is required for plotting") from exc

        kind = "coef" if kind == "auto" else kind
        if kind in {"coef", "effects"}:
            tab = self.tidy(include_event_study=False)
            if ax is None:
                fig, ax = plt.subplots(figsize=figsize)
            else:
                fig = ax.get_figure()
            y = np.arange(len(tab))
            ax.errorbar(
                tab["estimate"],
                y,
                xerr=[
                    tab["estimate"] - tab["conf_low"],
                    tab["conf_high"] - tab["estimate"],
                ],
                fmt="o",
                color="#1F4E79",
                ecolor="#6B8EAD",
                capsize=4,
                linewidth=1.2,
            )
            ax.axvline(0, color="0.4", linestyle="--", linewidth=0.9)
            ax.set_yticks(y)
            ax.set_yticklabels(tab["term"])
            ax.set_xlabel("Estimated effect")
            ax.set_title("Spatial DiD effects")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            fig.tight_layout()
            return fig, ax

        if kind == "exposure":
            exposure = self.detail.get("exposure_frame")
            if not isinstance(exposure, pd.DataFrame):
                raise MethodIncompatibility(
                    "Exposure diagnostics are not stored on this result.",
                    recovery_hint=(
                        "Call plot(kind='coef') or use a SpatialDiDResult "
                        "created by spatial_did."
                    ),
                    diagnostics={"kind": kind},
                )
            if ax is None:
                fig, ax = plt.subplots(figsize=figsize)
            else:
                fig = ax.get_figure()
            groups = [
                exposure.loc[exposure["treat"] == 0, "spillover_exposure"],
                exposure.loc[exposure["treat"] == 1, "spillover_exposure"],
            ]
            try:
                ax.boxplot(groups, tick_labels=["Control", "Treated"], showfliers=False)
            except TypeError:  # matplotlib < 3.9
                ax.boxplot(groups, labels=["Control", "Treated"], showfliers=False)
            ax.set_ylabel("Spatial treatment exposure (WD)")
            ax.set_title("Spillover exposure support")
            ax.spines["top"].set_visible(False)
            ax.spines["right"].set_visible(False)
            fig.tight_layout()
            return fig, ax

        if kind in {"event", "event_study"}:
            es = self.detail.get("event_study")
            if not isinstance(es, pd.DataFrame) or len(es) == 0:
                raise MethodIncompatibility(
                    "No spatial event-study estimates are available.",
                    recovery_hint=(
                        "Call spatial_did(..., event_study=True) before "
                        "plotting event-study paths."
                    ),
                    diagnostics={"kind": kind, "event_study": False},
                )
            if ax is None:
                fig, axes = plt.subplots(
                    1, 2, figsize=kwargs.pop("figsize", (11, 4)), sharey=True
                )
            else:
                axes = np.atleast_1d(ax)
                fig = axes[0].get_figure()
            for axis, effect in zip(axes, ["direct", "spillover"]):
                sub = es[es["effect"] == effect].sort_values("relative_time")
                axis.fill_between(
                    sub["relative_time"],
                    sub["ci_lower"],
                    sub["ci_upper"],
                    color="#D6E3F0",
                    alpha=0.8,
                )
                axis.plot(
                    sub["relative_time"],
                    sub["estimate"],
                    marker="o",
                    color="#1F4E79",
                    linewidth=1.5,
                )
                axis.axhline(0, color="0.4", linestyle="--", linewidth=0.8)
                axis.axvline(-0.5, color="#9A3412", linestyle=":", linewidth=1.0)
                axis.set_title(effect.title())
                axis.set_xlabel("Event time")
                axis.spines["top"].set_visible(False)
                axis.spines["right"].set_visible(False)
            axes[0].set_ylabel("Estimated effect")
            fig.tight_layout()
            return fig, axes

        raise MethodIncompatibility(
            "kind must be 'coef', 'exposure', or 'event_study'",
            recovery_hint="Use one of {'coef', 'exposure', 'event_study'}.",
            diagnostics={"kind": kind},
        )

    def __repr__(self) -> str:
        return (
            f"SpatialDiDResult(direct={self.direct_effect:+.4f}, "
            f"spillover={self.spillover_effect:+.4f}, "
            f"total={self.total_effect:+.4f})"
        )


def _as_weight_matrix(W: Any) -> Tuple[np.ndarray, Optional[Sequence[Any]]]:
    """Return a dense weights matrix and any object-level unit order."""
    id_order = getattr(W, "_id_order", None)
    try:
        if hasattr(W, "full"):
            W_full = W.full()
            if isinstance(W_full, tuple):
                W_full = W_full[0]
            W_mat = np.asarray(W_full, dtype=float)
        elif hasattr(W, "sparse"):
            W_mat = np.asarray(W.sparse.toarray(), dtype=float)
        elif hasattr(W, "toarray"):
            W_mat = np.asarray(W.toarray(), dtype=float)
        else:
            W_mat = np.asarray(W, dtype=float)
    except (TypeError, ValueError) as exc:
        raise MethodIncompatibility(
            "W must be convertible to a numeric spatial weights matrix",
            recovery_hint=(
                "Pass a square ndarray, scipy sparse matrix, or "
                "statspai.spatial.W object."
            ),
            diagnostics={"type": type(W).__name__},
        ) from exc
    if W_mat.ndim != 2 or W_mat.shape[0] != W_mat.shape[1]:
        raise MethodIncompatibility(
            "W must be a square spatial weights matrix",
            recovery_hint="Use a unit-by-unit square spatial weights matrix.",
            diagnostics={"shape": tuple(W_mat.shape)},
        )
    if not np.all(np.isfinite(W_mat)):
        raise MethodIncompatibility(
            "W must contain only finite numeric weights",
            recovery_hint="Replace NaN or infinite weights before calling spatial_did.",
            diagnostics={"shape": tuple(W_mat.shape)},
        )
    return W_mat, id_order


def _row_normalize(W_mat: np.ndarray) -> np.ndarray:
    rs = W_mat.sum(axis=1, keepdims=True)
    return np.asarray(
        np.divide(W_mat, np.where(np.abs(rs) < _EPS, 1.0, rs)),
        dtype=float,
    )


def _resolve_unit_order(
    df: pd.DataFrame,
    unit: str,
    W_id_order: Optional[Sequence[Any]],
    unit_order: Optional[Sequence[Any]],
) -> list:
    units = list(pd.unique(df[unit]))
    order = (
        list(unit_order)
        if unit_order is not None
        else (list(W_id_order) if W_id_order is not None else units)
    )
    if not units:
        raise DataInsufficient(
            "spatial_did requires at least one observed unit",
            recovery_hint="Provide non-empty panel data after dropping missing values.",
            diagnostics={"unit": unit},
        )
    if len(set(order)) != len(order):
        raise MethodIncompatibility(
            "W unit order must not contain duplicate units",
            recovery_hint="Pass each unit exactly once in `unit_order` or W.id_order.",
            diagnostics={"duplicates": [u for u in order if order.count(u) > 1]},
        )
    if set(order) != set(units):
        missing = sorted(set(units) - set(order), key=str)
        extra = sorted(set(order) - set(units), key=str)
        raise MethodIncompatibility(
            "W unit order must match data units; "
            f"missing={missing[:5]}, extra={extra[:5]}",
            recovery_hint="Align the rows/columns of W with the units in `data`.",
            diagnostics={"missing": missing, "extra": extra},
        )
    return order


def _twoway_within(
    df: pd.DataFrame,
    cols: Sequence[str],
    unit: str,
    time: str,
    *,
    tol: float = 1e-10,
    max_iter: int = 1000,
) -> pd.DataFrame:
    """Two-way within transformation, valid for balanced and unbalanced panels."""
    out = df.copy()
    Z = out[list(cols)].astype(float).copy()
    Z = Z - Z.mean(axis=0)
    for _ in range(max_iter):
        old = Z.to_numpy(copy=True)
        Z = Z - Z.groupby(out[unit], sort=False).transform("mean")
        Z = Z - Z.groupby(out[time], sort=False).transform("mean")
        if np.max(np.abs(Z.to_numpy() - old)) < tol:
            break
    for c in cols:
        out[c] = Z[c].to_numpy(dtype=float)
    return out


def _linear_fit(
    df: pd.DataFrame,
    y: str,
    x_cols: Sequence[str],
    unit: str,
    time: str,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    within = _twoway_within(df, [y, *x_cols], unit, time)
    X = np.column_stack([within[c].to_numpy(dtype=float) for c in x_cols])
    Y = within[y].to_numpy(dtype=float)
    beta, *_ = np.linalg.lstsq(X, Y, rcond=None)
    resid = Y - X @ beta
    return beta, resid, X


def _cluster_vcov(X: np.ndarray, resid: np.ndarray, cluster: np.ndarray) -> np.ndarray:
    # Canonical core sandwich (CLAUDE.md §4); CR1 correction
    # (G/max(G-1,1))*((n-1)/max(n-k,1)). pinv bread preserved.
    # Byte-identical to the prior hand-rolled sandwich for G >= 2.
    from ..core._vcov import sandwich_vcov

    XtX_inv = np.linalg.pinv(X.T @ X)
    return np.asarray(
        sandwich_vcov(XtX_inv, X * resid[:, None], clusters=cluster, correction="cr1"),
        dtype=float,
    )


def _hc1_vcov(X: np.ndarray, resid: np.ndarray) -> np.ndarray:
    n, k = X.shape
    XtX_inv = np.linalg.pinv(X.T @ X)
    meat = X.T @ ((resid**2)[:, None] * X)
    return np.asarray((n / max(n - k, 1)) * XtX_inv @ meat @ XtX_inv, dtype=float)


def _haversine_unit_distances(
    df: pd.DataFrame,
    unit: str,
    lat: str,
    lon: str,
    unit_order: Sequence[Any],
) -> np.ndarray:
    coords = df[[unit, lat, lon]].drop_duplicates(subset=[unit]).set_index(unit)
    coords = coords.reindex(unit_order)
    if coords[[lat, lon]].isna().any().any():
        raise DataInsufficient(
            "Each unit must have non-missing latitude/longitude",
            recovery_hint=(
                "Provide complete coordinates for every unit or use " "distance_matrix."
            ),
            diagnostics={"lat": lat, "lon": lon},
        )
    lat_vals = np.radians(coords[lat].to_numpy(dtype=float))
    lon_vals = np.radians(coords[lon].to_numpy(dtype=float))
    if not np.all(np.isfinite(lat_vals)) or not np.all(np.isfinite(lon_vals)):
        raise DataInsufficient(
            "Each unit must have finite latitude/longitude",
            recovery_hint="Replace NaN or infinite coordinates before using Conley SEs.",
            diagnostics={"lat": lat, "lon": lon},
        )
    dlat = lat_vals[:, None] - lat_vals[None, :]
    dlon = lon_vals[:, None] - lon_vals[None, :]
    a = (
        np.sin(dlat / 2) ** 2
        + np.cos(lat_vals[:, None]) * np.cos(lat_vals[None, :]) * np.sin(dlon / 2) ** 2
    )
    return np.asarray(
        2 * _EARTH_RADIUS_KM * np.arcsin(np.sqrt(np.minimum(a, 1.0))),
        dtype=float,
    )


def _distance_kernel(dist: np.ndarray, cutoff: float, kernel: str) -> np.ndarray:
    if not np.isfinite(cutoff) or cutoff <= 0:
        raise MethodIncompatibility(
            "conley_cutoff must be positive and finite",
            recovery_hint="Pass a positive distance cutoff for Conley standard errors.",
            diagnostics={"conley_cutoff": cutoff},
        )
    within = dist <= cutoff
    if kernel == "uniform":
        weights = within.astype(float)
    elif kernel == "bartlett":
        weights = np.maximum(1.0 - dist / cutoff, 0.0) * within
    else:
        raise MethodIncompatibility(
            "conley_kernel must be 'uniform' or 'bartlett'",
            recovery_hint="Use conley_kernel='uniform' or conley_kernel='bartlett'.",
            diagnostics={"conley_kernel": kernel},
        )
    np.fill_diagonal(weights, 1.0)
    return weights


def _conley_spatial_vcov(
    X: np.ndarray,
    resid: np.ndarray,
    df: pd.DataFrame,
    unit: str,
    time: str,
    unit_index: Dict[Any, int],
    unit_distances: np.ndarray,
    cutoff: float,
    kernel: str,
) -> np.ndarray:
    n, k = X.shape
    XtX_inv = np.linalg.pinv(X.T @ X)
    scores = X * resid[:, None]
    meat = np.zeros((k, k))
    unit_codes = df[unit].map(unit_index).to_numpy()
    for _, idx in df.groupby(time, sort=False).indices.items():
        idx_arr = np.asarray(idx, dtype=int)
        uidx = unit_codes[idx_arr]
        K = _distance_kernel(unit_distances[np.ix_(uidx, uidx)], cutoff, kernel)
        S = scores[idx_arr]
        meat += S.T @ K @ S
    scale = n / max(n - k, 1)
    return np.asarray(scale * XtX_inv @ meat @ XtX_inv, dtype=float)


def _vcov(
    *,
    X: np.ndarray,
    resid: np.ndarray,
    df: pd.DataFrame,
    unit: str,
    time: str,
    cluster: str,
    se_type: str,
    unit_index: Dict[Any, int],
    unit_distances: Optional[np.ndarray],
    conley_cutoff: Optional[float],
    conley_kernel: str,
) -> np.ndarray:
    if se_type == "cluster":
        return _cluster_vcov(X, resid, df[cluster].to_numpy())
    if se_type == "robust":
        return _hc1_vcov(X, resid)
    if se_type == "conley":
        if unit_distances is None or conley_cutoff is None:
            raise MethodIncompatibility(
                "se_type='conley' requires conley_cutoff and either "
                "coords=(lat, lon), lat/lon, or distance_matrix",
                recovery_hint=(
                    "Provide conley_cutoff plus coordinates or a distance " "matrix."
                ),
                diagnostics={"se_type": se_type, "conley_cutoff": conley_cutoff},
            )
        return _conley_spatial_vcov(
            X,
            resid,
            df,
            unit,
            time,
            unit_index,
            unit_distances,
            conley_cutoff,
            conley_kernel,
        )
    raise MethodIncompatibility(
        "se_type must be 'cluster', 'robust', or 'conley'",
        recovery_hint=("Use se_type='cluster', se_type='robust', or se_type='conley'."),
        diagnostics={"se_type": se_type},
    )


def _effect_stats(
    beta: np.ndarray,
    V: np.ndarray,
    alpha: float,
) -> Tuple[pd.DataFrame, float, float, tuple, float]:
    se = np.sqrt(np.maximum(np.diag(V), 0.0))
    crit = stats.norm.ppf(1 - alpha / 2)
    z = beta / np.where(se > 0, se, np.nan)
    p = 2 * stats.norm.sf(np.abs(z))
    coef_df = pd.DataFrame(
        {
            "coef": beta,
            "estimate": beta,
            "se": se,
            "statistic": z,
            "pvalue": p,
            "ci_lower": beta - crit * se,
            "ci_upper": beta + crit * se,
        }
    )
    total = float(beta[0] + beta[1])
    total_var = float(V[0, 0] + V[1, 1] + 2 * V[0, 1])
    se_total = float(np.sqrt(max(total_var, 0.0)))
    ci_total = (total - crit * se_total, total + crit * se_total)
    p_total = (
        float(2 * stats.norm.sf(abs(total / se_total))) if se_total > 0 else np.nan
    )
    return coef_df, total, se_total, ci_total, p_total


def _build_spatial_lags(
    df: pd.DataFrame,
    value_col: str,
    time: str,
    unit: str,
    W_norm: np.ndarray,
    unit_order: Sequence[Any],
    out_col: str,
) -> pd.Series:
    wide = df.pivot(index=unit, columns=time, values=value_col).reindex(unit_order)
    time_cols = list(wide.columns)
    mat = wide.fillna(0.0).to_numpy(dtype=float)
    lagged = W_norm @ mat
    lag_df = pd.DataFrame(lagged, index=unit_order, columns=time_cols)
    keys = pd.MultiIndex.from_frame(df[[unit, time]])
    stacked = lag_df.stack()
    stacked.index.names = [unit, time]
    return pd.Series(keys.map(stacked), index=df.index, name=out_col).astype(float)


def _event_time(
    df: pd.DataFrame,
    treat: str,
    unit: str,
    time: str,
) -> pd.Series:
    try:
        time_num = df[time].astype(float)
        time_lookup = dict(zip(df[time], time_num))
    except (TypeError, ValueError):
        ordered = sorted(pd.unique(df[time]))
        time_lookup = {v: i for i, v in enumerate(ordered)}
    tmp = df[[unit, time, treat]].copy()
    tmp["_time_num"] = tmp[time].map(time_lookup).astype(float)
    first = tmp.loc[tmp[treat] > 0].groupby(unit, sort=False)["_time_num"].min()
    rel = tmp["_time_num"] - tmp[unit].map(first)
    return rel.where(tmp[unit].isin(first.index), np.nan)


def _wald_zero(beta: np.ndarray, V: np.ndarray) -> Dict[str, Any]:
    if beta.size == 0:
        return {}
    Vinv = np.linalg.pinv(V)
    stat = float(beta.T @ Vinv @ beta)
    df = int(beta.size)
    return {"statistic": stat, "df": df, "pvalue": float(stats.chi2.sf(stat, df))}


def _spatial_event_study(
    df: pd.DataFrame,
    y: str,
    treat: str,
    unit: str,
    time: str,
    W_norm: np.ndarray,
    unit_order: Sequence[Any],
    covariates: Sequence[str],
    alpha: float,
    event_window: Optional[Tuple[int, int]],
    event_base: int,
    se_type: str,
    cluster: str,
    unit_distances: Optional[np.ndarray],
    conley_cutoff: Optional[float],
    conley_kernel: str,
) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    es_df = df.copy()
    es_df["_rel_time"] = _event_time(es_df, treat, unit, time)
    rel_obs = es_df["_rel_time"].dropna()
    if rel_obs.empty:
        return pd.DataFrame(), {}
    if event_window is None:
        lo = int(max(np.floor(rel_obs.min()), -5))
        hi = int(min(np.ceil(rel_obs.max()), 5))
    else:
        lo, hi = event_window
    event_times = [k for k in range(int(lo), int(hi) + 1) if k != event_base]
    if not event_times:
        return pd.DataFrame(), {}

    x_cols = []
    direct_cols, spill_cols = [], []
    for k in event_times:
        d_col = f"_E_{k}"
        w_col = f"_WE_{k}"
        es_df[d_col] = (es_df["_rel_time"] == k).astype(float)
        es_df[w_col] = _build_spatial_lags(
            es_df,
            d_col,
            time,
            unit,
            W_norm,
            unit_order,
            w_col,
        )
        direct_cols.append(d_col)
        spill_cols.append(w_col)
        x_cols.extend([d_col, w_col])
    x_cols.extend(covariates)

    beta, resid, X = _linear_fit(es_df, y, x_cols, unit, time)
    unit_index = {u: i for i, u in enumerate(unit_order)}
    V = _vcov(
        X=X,
        resid=resid,
        df=es_df,
        unit=unit,
        time=time,
        cluster=cluster,
        se_type=se_type,
        unit_index=unit_index,
        unit_distances=unit_distances,
        conley_cutoff=conley_cutoff,
        conley_kernel=conley_kernel,
    )
    se = np.sqrt(np.maximum(np.diag(V), 0.0))
    crit = stats.norm.ppf(1 - alpha / 2)
    rows = []
    for j, k in enumerate(event_times):
        direct_idx = x_cols.index(f"_E_{k}")
        spill_idx = x_cols.index(f"_WE_{k}")
        for effect, idx in (("direct", direct_idx), ("spillover", spill_idx)):
            b = float(beta[idx])
            s = float(se[idx])
            rows.append(
                {
                    "effect": effect,
                    "relative_time": int(k),
                    "estimate": b,
                    "se": s,
                    "ci_lower": b - crit * s,
                    "ci_upper": b + crit * s,
                    "pvalue": float(2 * stats.norm.sf(abs(b / s))) if s > 0 else np.nan,
                }
            )

    lead_times = [k for k in event_times if k < 0]
    pretrend: Dict[str, Any] = {}
    if lead_times:
        direct_lead_idx = [x_cols.index(f"_E_{k}") for k in lead_times]
        spill_lead_idx = [x_cols.index(f"_WE_{k}") for k in lead_times]
        pretrend["direct"] = _wald_zero(
            beta[direct_lead_idx],
            V[np.ix_(direct_lead_idx, direct_lead_idx)],
        )
        pretrend["spillover"] = _wald_zero(
            beta[spill_lead_idx],
            V[np.ix_(spill_lead_idx, spill_lead_idx)],
        )
    return pd.DataFrame(rows), pretrend


def _residual_moran_by_time(
    df: pd.DataFrame,
    resid: np.ndarray,
    unit: str,
    time: str,
    W_norm: np.ndarray,
    unit_order: Sequence[Any],
) -> float:
    tmp = df[[unit, time]].copy()
    tmp["_resid"] = resid
    unit_index = {u: i for i, u in enumerate(unit_order)}
    vals = []
    S0 = W_norm.sum()
    if S0 <= 0:
        return np.nan
    for _, sub in tmp.groupby(time, sort=False):
        r = np.zeros(len(unit_order))
        seen = np.zeros(len(unit_order), dtype=bool)
        for _, row in sub.iterrows():
            idx = unit_index[row[unit]]
            r[idx] = row["_resid"]
            seen[idx] = True
        if seen.sum() < 3:
            continue
        r = r - r[seen].mean()
        denom = float(r[seen] @ r[seen])
        if denom <= _EPS:
            continue
        vals.append(float(seen.sum() / S0 * (r @ W_norm @ r) / denom))
    return float(np.mean(vals)) if vals else np.nan


def _spatial_diagnostics(
    df: pd.DataFrame,
    treat: str,
    unit: str,
    time: str,
    W_mat: np.ndarray,
    W_norm: np.ndarray,
    resid: np.ndarray,
    X: np.ndarray,
    unit_order: Sequence[Any],
) -> Dict[str, Any]:
    D = df[treat].to_numpy(dtype=float)
    WD = df["_WD"].to_numpy(dtype=float)
    corr = np.corrcoef(D, WD)[0, 1] if np.std(D) > 0 and np.std(WD) > 0 else np.nan
    controls = D <= 0
    treated = D > 0
    diagnostics = {
        "n_units": int(len(unit_order)),
        "n_periods": int(df[time].nunique()),
        "n_obs": int(len(df)),
        "w_density": float(np.mean(np.abs(W_mat) > 0)),
        "w_islands": int(np.sum(np.isclose(W_mat.sum(axis=1), 0.0))),
        "treated_observation_share": float(np.mean(treated)),
        "mean_spillover_exposure": float(np.mean(WD)),
        "control_spillover_exposure_share": (
            float(np.mean(WD[controls] > 0)) if controls.any() else np.nan
        ),
        "treated_spillover_exposure_share": (
            float(np.mean(WD[treated] > 0)) if treated.any() else np.nan
        ),
        "mean_wd_control": float(np.mean(WD[controls])) if controls.any() else np.nan,
        "mean_wd_treated": float(np.mean(WD[treated])) if treated.any() else np.nan,
        "corr_treat_spillover": float(corr) if pd.notna(corr) else np.nan,
        "condition_number": float(np.linalg.cond(X.T @ X)),
        "mean_residual_moran_i": _residual_moran_by_time(
            df,
            resid,
            unit,
            time,
            W_norm,
            unit_order,
        ),
    }
    return diagnostics


def spatial_did(
    data: pd.DataFrame,
    y: str,
    treat: str,
    unit: str,
    time: str,
    W: Any,
    covariates: Optional[Sequence[str]] = None,
    cluster: Optional[str] = None,
    alpha: float = 0.05,
    *,
    se_type: str = "cluster",
    coords: Optional[Tuple[str, str]] = None,
    lat: Optional[str] = None,
    lon: Optional[str] = None,
    distance_matrix: Optional[np.ndarray] = None,
    conley_cutoff: Optional[float] = None,
    conley_kernel: str = "bartlett",
    unit_order: Optional[Sequence[Any]] = None,
    normalize_W: bool = True,
    event_study: bool = False,
    event_window: Optional[Tuple[int, int]] = None,
    event_base: int = -1,
) -> SpatialDiDResult:
    """
    Spatial DiD with direct and spillover treatment effects.

    Parameters
    ----------
    data : DataFrame
        Long panel with one row per unit-period.
    y, treat, unit, time : str
        Outcome, treatment, unit-id, and time-id columns. ``treat`` may be
        binary or a continuous exposure intensity; the spatial lag uses the
        same scale.
    W : ndarray, sparse matrix, or :class:`statspai.spatial.W`
        Spatial weights matrix over units. If a StatsPAI ``W`` object carries
        an ``id_order``, that order is respected.
    covariates : sequence of str, optional
        Additional controls included after absorbing unit and time effects.
    cluster : str, optional
        Cluster variable for ``se_type='cluster'``. Defaults to ``unit``.
    alpha : float, default 0.05
        Significance level for confidence intervals.
    se_type : {"cluster", "robust", "conley"}, default "cluster"
        Covariance estimator. ``conley`` uses spatial-HAC dependence within
        each period and requires ``conley_cutoff`` plus coordinates or a
        unit-level distance matrix.
    coords : tuple(str, str), optional
        Convenience pair ``(lat, lon)`` in degrees.
    lat, lon : str, optional
        Latitude/longitude columns in degrees.
    distance_matrix : ndarray, optional
        Unit-by-unit distances in the same units as ``conley_cutoff``.
    conley_cutoff : float, optional
        Spatial cutoff for Conley-HAC standard errors.
    conley_kernel : {"uniform", "bartlett"}, default "bartlett"
    unit_order : sequence, optional
        Explicit row/column order of ``W``.
    normalize_W : bool, default True
        Row-normalise weights before constructing ``WD``.
    event_study : bool, default False
        Estimate direct and spillover event-study paths by relative time.
    event_window : tuple(int, int), optional
        Inclusive event-time window. Defaults to observed support capped at
        [-5, 5] when ``event_study=True``.
    event_base : int, default -1
        Omitted relative-time period.

    Returns
    -------
    SpatialDiDResult
        Direct, spillover, total effects with diagnostics, plotting, and
        export helpers.

    References
    ----------
    delgado2015difference

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> n_units, n_time = 12, 6
    >>> W = np.zeros((n_units, n_units))  # ring spatial weights over units
    >>> for i in range(n_units):
    ...     W[i, (i + 1) % n_units] = 1
    ...     W[i, (i - 1) % n_units] = 1
    >>> rows = []
    >>> for u in range(n_units):
    ...     fe = rng.normal()
    ...     for t in range(n_time):
    ...         d = 1 if (u < 6 and t >= 3) else 0  # half treated from t>=3
    ...         y = 1.0 + fe + 0.3 * t + 1.5 * d + rng.normal(scale=0.3)
    ...         rows.append({"unit": u, "time": t, "treat": d, "y": y})
    >>> df = pd.DataFrame(rows)
    >>> res = sp.spatial_did(df, y="y", treat="treat", unit="unit",
    ...                     time="time", W=W)
    >>> res.coefficients["variable"].tolist()
    ['treat', 'W_treat']
    >>> res.n_obs
    72
    """
    if not isinstance(data, pd.DataFrame):
        raise MethodIncompatibility(
            "`data` must be a pandas DataFrame",
            recovery_hint="Pass a long panel DataFrame with unit and time columns.",
            diagnostics={"type": type(data).__name__},
        )
    for label, col in (("y", y), ("treat", treat), ("unit", unit), ("time", time)):
        _require_column_name(col, label)
    cov = _column_list(covariates, "covariates")
    if coords is not None:
        try:
            coord_list = list(coords)
        except TypeError as exc:
            raise MethodIncompatibility(
                "`coords` must be a pair of latitude/longitude column names",
                recovery_hint="Pass coords=('lat', 'lon') or set lat=... and lon=....",
                diagnostics={"coords": repr(coords)},
            ) from exc
        if len(coord_list) != 2:
            raise MethodIncompatibility(
                "`coords` must contain exactly two column names",
                recovery_hint="Pass coords=('lat', 'lon').",
                diagnostics={"coords": coord_list},
            )
        lat, lon = coord_list
    if lat is not None:
        _require_column_name(lat, "lat")
    if lon is not None:
        _require_column_name(lon, "lon")
    if cluster is None:
        cluster = unit
    else:
        _require_column_name(cluster, "cluster")
    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise MethodIncompatibility(
            "alpha must be between 0 and 1",
            recovery_hint="Pass a significance level such as alpha=0.05.",
            diagnostics={"alpha": alpha},
        )
    if conley_cutoff is not None and (
        not np.isfinite(conley_cutoff) or conley_cutoff <= 0
    ):
        raise MethodIncompatibility(
            "conley_cutoff must be positive and finite",
            recovery_hint="Pass a positive distance cutoff for Conley standard errors.",
            diagnostics={"conley_cutoff": conley_cutoff},
        )
    if conley_kernel not in _CONLEY_KERNELS:
        raise MethodIncompatibility(
            "conley_kernel must be 'uniform' or 'bartlett'",
            recovery_hint="Use conley_kernel='uniform' or conley_kernel='bartlett'.",
            diagnostics={"conley_kernel": conley_kernel},
        )
    if se_type not in _SE_TYPES:
        raise MethodIncompatibility(
            "se_type must be 'cluster', 'robust', or 'conley'",
            recovery_hint=(
                "Use se_type='cluster', se_type='robust', or se_type='conley'."
            ),
            diagnostics={"se_type": se_type},
        )
    if not isinstance(normalize_W, bool):
        raise MethodIncompatibility(
            "normalize_W must be True or False",
            recovery_hint="Pass a boolean for `normalize_W`.",
            diagnostics={"normalize_W": normalize_W},
        )
    if event_window is not None:
        try:
            event_window_values = tuple(event_window)
        except TypeError as exc:
            raise MethodIncompatibility(
                "event_window must be a two-element tuple",
                recovery_hint="Pass event_window=(-5, 5) or leave it as None.",
                diagnostics={"event_window": repr(event_window)},
            ) from exc
        if (
            len(event_window_values) != 2
            or event_window_values[0] > event_window_values[1]
        ):
            raise MethodIncompatibility(
                "event_window must be an ordered two-element tuple",
                recovery_hint="Pass event_window=(min_event_time, max_event_time).",
                diagnostics={"event_window": event_window_values},
            )
        event_window = (int(event_window_values[0]), int(event_window_values[1]))
    extra_keep = [cluster] if cluster not in {y, treat, unit, time, *cov} else []
    coord_keep = [
        c
        for c in (lat, lon)
        if c is not None and c not in {y, treat, unit, time, *cov, *extra_keep}
    ]
    keep = [y, treat, unit, time] + cov + extra_keep + coord_keep
    missing_cols = [c for c in keep if c not in data.columns]
    if missing_cols:
        raise MethodIncompatibility(
            f"Columns not found in data: {missing_cols}",
            recovery_hint="Check y/treat/unit/time/covariate/coordinate column names.",
            diagnostics={"missing_columns": missing_cols},
        )
    df = data[keep].dropna().sort_values([unit, time]).reset_index(drop=True)
    if df.empty:
        raise DataInsufficient(
            "No complete observations remain after dropping missing values",
            recovery_hint="Impute or remove missing values before calling spatial_did.",
            diagnostics={"required_columns": keep},
        )
    if df[unit].nunique() < 2 or df[time].nunique() < 2:
        raise DataInsufficient(
            "spatial_did requires at least two units and two time periods",
            recovery_hint="Provide panel data with variation across units and time.",
            diagnostics={
                "n_units": int(df[unit].nunique()),
                "n_periods": int(df[time].nunique()),
            },
        )

    duplicated = df.duplicated([unit, time])
    if duplicated.any():
        raise MethodIncompatibility(
            "spatial_did requires one row per unit-period; aggregate "
            "repeated cross sections before calling.",
            recovery_hint=(
                "Aggregate repeated unit-time cells or choose a "
                "repeated-cross-section estimator."
            ),
            diagnostics={"duplicate_rows": int(duplicated.sum())},
        )

    W_mat, W_id_order = _as_weight_matrix(W)
    resolved_order = _resolve_unit_order(df, unit, W_id_order, unit_order)
    if W_mat.shape[0] != len(resolved_order):
        raise MethodIncompatibility(
            "W dimensions must match number of unique units",
            recovery_hint="Align W rows/columns with the data units.",
            diagnostics={
                "W_shape": tuple(W_mat.shape),
                "n_units": len(resolved_order),
            },
        )
    W_norm = _row_normalize(W_mat) if normalize_W else W_mat.astype(float)
    unit_index = {u: i for i, u in enumerate(resolved_order)}

    warnings = []
    expected_cells = len(resolved_order) * df[time].nunique()
    if len(df) < expected_cells:
        warnings.append(
            "Panel is unbalanced; missing neighbors' treatment is treated as 0 "
            "when constructing WD for that period."
        )
    non_binary = not set(pd.unique(df[treat].dropna())).issubset(
        {0, 1, 0.0, 1.0, True, False}
    )
    if non_binary:
        warnings.append(
            "Treatment is not binary; direct and spillover coefficients are "
            "marginal effects on the supplied treatment scale."
        )

    df["_WD"] = _build_spatial_lags(
        df, treat, time, unit, W_norm, resolved_order, "_WD"
    )

    x_cols = [treat, "_WD"] + cov
    beta, resid, X = _linear_fit(df, y, x_cols, unit, time)

    unit_distances = None
    if distance_matrix is not None:
        unit_distances = np.asarray(distance_matrix, dtype=float)
        if unit_distances.shape != W_mat.shape:
            raise MethodIncompatibility(
                "distance_matrix must have the same shape as W",
                recovery_hint="Pass a unit-by-unit distance matrix aligned to W.",
                diagnostics={
                    "distance_shape": tuple(unit_distances.shape),
                    "W_shape": tuple(W_mat.shape),
                },
            )
        if not np.all(np.isfinite(unit_distances)):
            raise MethodIncompatibility(
                "distance_matrix must contain only finite distances",
                recovery_hint=(
                    "Replace NaN or infinite distances before using Conley SEs."
                ),
                diagnostics={"distance_shape": tuple(unit_distances.shape)},
            )
    elif lat is not None and lon is not None:
        unit_distances = _haversine_unit_distances(df, unit, lat, lon, resolved_order)
    if se_type == "cluster" and conley_cutoff is not None:
        se_type = "conley"

    V = _vcov(
        X=X,
        resid=resid,
        df=df,
        unit=unit,
        time=time,
        cluster=cluster,
        se_type=se_type,
        unit_index=unit_index,
        unit_distances=unit_distances,
        conley_cutoff=conley_cutoff,
        conley_kernel=conley_kernel,
    )
    coef_stats, total, se_total, ci_total, p_total = _effect_stats(beta, V, alpha)
    coef_stats.insert(0, "variable", [treat, f"W_{treat}"] + cov)
    coef_stats.insert(1, "role", ["direct", "spillover"] + ["covariate"] * len(cov))

    direct = float(beta[0])
    spill = float(beta[1])
    se_d = float(coef_stats.loc[0, "se"])
    se_s = float(coef_stats.loc[1, "se"])
    p_d = float(coef_stats.loc[0, "pvalue"])
    p_s = float(coef_stats.loc[1, "pvalue"])
    ci_d = (float(coef_stats.loc[0, "ci_lower"]), float(coef_stats.loc[0, "ci_upper"]))
    ci_s = (float(coef_stats.loc[1, "ci_lower"]), float(coef_stats.loc[1, "ci_upper"]))

    diagnostics = _spatial_diagnostics(
        df, treat, unit, time, W_mat, W_norm, resid, X, resolved_order
    )
    event_df = pd.DataFrame()
    pretrend: Dict[str, Any] = {}
    if event_study:
        event_df, pretrend = _spatial_event_study(
            df,
            y,
            treat,
            unit,
            time,
            W_norm,
            resolved_order,
            cov,
            alpha,
            event_window,
            event_base,
            se_type,
            cluster,
            unit_distances,
            conley_cutoff,
            conley_kernel,
        )

    exposure_frame = df[[unit, time, treat, "_WD"]].rename(
        columns={treat: "treat", "_WD": "spillover_exposure"}
    )
    model_info = {
        "method": "Spatial Difference-in-Differences",
        "se_type": se_type,
        "direct_effect": direct,
        "spillover_effect": spill,
        "total_effect": total,
        "covariates": cov,
        "cluster": cluster,
        "alpha": alpha,
        "normalize_W": normalize_W,
        "conley_cutoff": conley_cutoff,
        "conley_kernel": conley_kernel if se_type == "conley" else None,
        "event_study": event_df if len(event_df) else None,
        "pretrend_test": pretrend if pretrend else None,
    }
    detail = {
        "diagnostics": diagnostics,
        "warnings": warnings,
        "exposure_frame": exposure_frame,
        "W": W_norm,
        "unit_order": list(resolved_order),
    }
    if len(event_df):
        detail["event_study"] = event_df
    if pretrend:
        detail["pretrend_test"] = pretrend

    result = SpatialDiDResult(
        direct_effect=direct,
        spillover_effect=spill,
        se_direct=se_d,
        se_spillover=se_s,
        ci_direct=ci_d,
        ci_spillover=ci_s,
        pvalue_direct=p_d,
        pvalue_spillover=p_s,
        coefficients=coef_stats,
        n_obs=len(df),
        total_effect=total,
        se_total=se_total,
        ci_total=ci_total,
        pvalue_total=p_total,
        alpha=alpha,
        se_type=se_type,
        vcov=V,
        detail=detail,
        model_info=model_info,
        data_info={
            "nobs": len(df),
            "dependent_var": y,
            "df_resid": max(len(df) - X.shape[1], 1),
            "X": X,
            "residuals": resid,
            "vcov": V,
        },
    )
    try:
        from ..output._lineage import attach_provenance as _attach_prov

        _attach_prov(
            result,
            function="sp.spatial.spatial_did",
            params={
                "y": y,
                "treat": treat,
                "unit": unit,
                "time": time,
                "covariates": cov or None,
                "cluster": cluster,
                "alpha": alpha,
                "se_type": se_type,
                "coords": coords,
                "lat": lat,
                "lon": lon,
                "conley_cutoff": conley_cutoff,
                "conley_kernel": conley_kernel,
                "event_study": event_study,
                "event_window": event_window,
                "event_base": event_base,
                "W_shape": list(W_mat.shape),
            },
            data=data,
            overwrite=False,
        )
    except Exception:  # pragma: no cover
        pass
    return result


__all__ = ["spatial_did", "SpatialDiDResult"]
