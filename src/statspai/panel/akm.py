"""Abowd-Kramarz-Margolis two-way fixed effects -- ``sp.akm``.

``y_it = alpha_i + psi_J(i,t) + x_it' beta + e_it`` on a worker-firm panel:
worker and firm effects on the connected set of firms (linked by movers),
and the variance decomposition of the outcome into the worker effects,
the firm effects, twice their covariance and the residual.

The plug-in variances of estimated effects are biased upwards by their
estimation noise, which is large when few workers move (limited mobility
bias). ``variance='kss'`` subtracts the bias with the leave-one-out
estimator of Kline, Saggio and Solvsten (2020), valid under
heteroskedasticity; ``'homoskedastic'`` is the earlier correction of
Andrews et al. (2008). Both need the leave-one-out connected set (every
observation can be dropped without disconnecting its firm), which
``connected='leave_one_out'`` selects. The algebra is in
:mod:`statspai.panel._akm_kss`, the connected sets in
:mod:`statspai.panel._akm_graph`.

Covariates (and ``time`` effects, entered as indicators) are partialled
out first, as in the authors' code: their coefficients come from the full
model ``sp.hdfe_ols('y ~ x + i.time | worker + firm')`` on the estimation
sample, and the two-way decomposition is of ``y - x'b``.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd

from .._result_serialize import ResultProtocolMixin
from ..exceptions import (
    ColumnNotFound,
    DataInsufficient,
    MethodIncompatibility,
    StatsPAIWarning,
)
from . import _akm_graph as _graph
from . import _akm_kss as _kss

__all__ = ["AKMResult", "akm"]

_VARIANCES = ("kss", "homoskedastic", "plugin", "none")
_CONNECTED = ("largest", "leave_one_out", "all")
_LEVERAGE = ("auto", "exact", "jla")
#: firms above which ``leverage='auto'`` switches from dense to random
#: projection (three dense J x J matrices of doubles: 3 * 8 * J^2 bytes)
EXACT_MAX_FIRMS = 3000


@dataclass
class AKMResult(ResultProtocolMixin):
    """Result of :func:`sp.akm <statspai.akm>`.

    Attributes
    ----------
    worker_effects : DataFrame
        One row per worker: ``worker``, ``alpha`` (the worker effect,
        including the level of the outcome), ``n_obs``, ``mover``.
    firm_effects : DataFrame
        One row per firm: ``firm``, ``psi`` (person-year-weighted mean zero
        in each connected set), ``n_obs``, ``n_movers``.
    decomposition : DataFrame
        Rows ``var_y`` (of the outcome net of covariates), ``var_alpha``,
        ``var_psi``, ``cov`` (cov(alpha, psi); ``2 * cov`` enters the
        decomposition), ``var_resid``; one column per estimator computed
        (``plugin`` always; ``homoskedastic`` and ``kss`` when asked), all
        with the ``1/n`` person-year convention.
    shares : DataFrame
        ``decomposition`` divided by ``var_y`` (``cov`` doubled).
    coefficients : DataFrame or None
        Covariate (and time-indicator) coefficients and standard errors
        from the full model, ``None`` without covariates.
    sample : dict
        Observations in / dropped (missing values, outside the connected
        set, leave-one-out bridges), workers, movers, firms, components.
    leverage : dict
        ``method`` ('exact' or 'jla'), ``n_draws``, ``seed``, min / max
        ``P_ii`` (only with a corrected variance).
    data : DataFrame
        The estimation sample with ``alpha``, ``psi``, ``xb`` and ``resid``
        (and ``P_ii`` with a corrected variance), in input order.

    Examples
    --------
    >>> import statspai as sp
    >>> sp.AKMResult.__name__
    'AKMResult'
    """

    worker_effects: pd.DataFrame
    firm_effects: pd.DataFrame
    decomposition: Optional[pd.DataFrame]
    coefficients: Optional[pd.DataFrame]
    sample: Dict[str, Any]
    leverage: Dict[str, Any]
    data: pd.DataFrame
    variance: str
    connected: str
    y: str
    worker: str
    firm: str
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    _citation_keys: ClassVar[Tuple[str, ...]] = (
        "abowd1999high",
        "kline2020leave",
        "andrews2008high",
    )

    @property
    def shares(self) -> Optional[pd.DataFrame]:
        if self.decomposition is None:
            return None
        d = self.decomposition
        out = d.div(d.loc["var_y"], axis=1)
        out.loc["cov"] = 2.0 * out.loc["cov"]
        return out.rename(index={"cov": "2cov"})

    @property
    def params(self) -> pd.Series:
        """Covariate coefficients (empty without covariates); the variance
        components are in ``decomposition`` / ``components``."""
        if self.coefficients is None:
            return pd.Series(dtype=float)
        return self.coefficients["coef"].rename(None)

    @property
    def std_errors(self) -> pd.Series:
        """Standard errors of ``params``."""
        if self.coefficients is None:
            return pd.Series(dtype=float)
        return self.coefficients["std_err"].rename(None)

    @property
    def nobs(self) -> int:
        return int(self.sample["n_obs"])

    @property
    def components(self) -> pd.Series:
        """Variance components of the preferred estimator: KSS when
        computed, else homoskedastic, else plug-in (empty without a
        decomposition)."""
        if self.decomposition is None:
            return pd.Series(dtype=float)
        for col in ("kss", "homoskedastic", "plugin"):
            if col in self.decomposition.columns:
                return self.decomposition[col].rename(col)
        return pd.Series(dtype=float)  # pragma: no cover

    def felsdvreg_variables(self, normalize: bool = False) -> pd.DataFrame:
        """The variables Stata ``felsdvreg`` saves, on the rows used.

        Columns ``peff`` (worker effect), ``feff`` (firm effect), ``xb``,
        ``res``, ``mover``, ``mnum`` (rows of movers at the firm), ``pobs``
        (rows of the worker) and ``group`` (mobility group: firms without
        movers are group 0, the others are numbered in the order of their
        smallest firm id). ``felsdvreg`` sets the firm effect of the
        smallest firm of each group to zero; ``normalize=True`` instead
        makes the firm effects average zero over the group's rows (its
        ``normalize`` option). Indexed like ``data``.
        """
        d = self.data
        wc, _ = pd.factorize(d[self.worker], sort=True)
        fc, flev = pd.factorize(d[self.firm], sort=True)
        n_w, n_f = int(wc.max()) + 1, len(flev)
        pairs = np.unique(np.column_stack([wc, fc]), axis=0)
        mover_w = np.bincount(pairs[:, 0], minlength=n_w) > 1
        mover = mover_w[wc]
        mnum_f = np.bincount(fc, weights=mover.astype(float), minlength=n_f)
        comp = _graph.components(wc, fc, n_w, n_f)
        comp_f = np.zeros(n_f, dtype=np.int64)
        comp_f[fc] = comp
        has_movers = mnum_f[fc] > 0
        # groups numbered by the smallest firm (sorted id) they contain
        first_firm = pd.Series(np.arange(n_f)).groupby(comp_f).min()
        moving = sorted(c for c in first_firm.index if mnum_f[comp_f == c].sum() > 0)
        moving.sort(key=lambda c: first_firm[c])
        number = {c: k + 1 for k, c in enumerate(moving)}
        group = np.where(has_movers, [number.get(c, 0) for c in comp], 0)
        psi = d["psi"].to_numpy(dtype=float).copy()
        alpha = d["alpha"].to_numpy(dtype=float).copy()
        if not normalize:
            psi_f = np.zeros(n_f)
            psi_f[fc] = psi
            ref = psi_f[first_firm.to_numpy()]  # per component
            shift = ref[np.searchsorted(first_firm.index.to_numpy(), comp)]
            psi = psi - shift
            alpha = alpha + shift
        T = np.bincount(wc, minlength=n_w)
        return pd.DataFrame(
            {
                "peff": alpha,
                "feff": psi,
                "xb": d["xb"].to_numpy(dtype=float),
                "res": d["resid"].to_numpy(dtype=float),
                "mover": mover.astype(int),
                "mnum": mnum_f[fc].astype(int),
                "pobs": T[wc].astype(int),
                "group": group.astype(int),
            },
            index=d.index,
        )

    def fixef(self) -> Dict[str, pd.Series]:
        """Worker and firm effects as Series indexed by id (as
        ``sp.feols(...).fixef()``)."""
        return {
            self.worker: self.worker_effects.set_index("worker")["alpha"].rename(
                self.worker
            ),
            self.firm: self.firm_effects.set_index("firm")["psi"].rename(self.firm),
        }

    def summary(self) -> str:
        s = self.sample
        lines = [
            f"AKM two-way fixed effects  |  {self.y} ~ {self.worker} + {self.firm}",
            "=" * 66,
            f"Observations: {s['n_obs']:,d} used of {s['n_input']:,d}   "
            f"connected={self.connected}",
            f"Workers: {s['n_workers']:,d} (movers {s['n_movers']:,d})   "
            f"Firms: {s['n_firms']:,d}   Components: {s['n_components']}",
            "Dropped: "
            + ", ".join(f"{k}={v:,d}" for k, v in s["dropped"].items() if v),
        ]
        if self.coefficients is not None:
            lines.append("-" * 66)
            lines.append(self.coefficients.to_string(float_format=lambda v: f"{v:.4f}"))
        if self.decomposition is not None:
            lines.append("-" * 66)
            lines.append("Variance decomposition (person-year weighted, 1/n):")
            lines.append(
                self.decomposition.to_string(float_format=lambda v: f"{v:.5f}")
            )
            if self.leverage.get("method"):
                lines.append(
                    f"Leverages: {self.leverage['method']}"
                    + (
                        f" ({self.leverage['n_draws']} draws, seed "
                        f"{self.leverage['seed']})"
                        if self.leverage["method"] == "jla"
                        else ""
                    )
                    + f"   P_ii in [{self.leverage['min_P']:.3f}, "
                    f"{self.leverage['max_P']:.3f}]"
                )
        lines.append("=" * 66)
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()


def _require_columns(data: pd.DataFrame, cols: Sequence[str]) -> None:
    missing = [c for c in cols if c not in data.columns]
    if missing:
        raise ColumnNotFound(
            f"akm: column(s) {missing} not in data.",
            recovery_hint="Pass existing column names for y / worker / firm / "
            "time / covariates.",
            diagnostics={"missing": missing},
        )


def _partial_out(
    frame: pd.DataFrame,
    covariates: List[str],
    time: Optional[str],
    cluster: Optional[str],
) -> Tuple[np.ndarray, Optional[pd.DataFrame], Dict[str, Any]]:
    """``x'b`` on every row of ``frame`` from the full two-way model."""
    from .feols import _materialize_rhs, _parse_formula, feols

    work = pd.DataFrame(
        {"_y": frame["_y"].to_numpy(), "_w": frame["_w"], "_f": frame["_f"]}
    )
    rhs: List[str] = []
    names: Dict[str, str] = {}
    for j, c in enumerate(covariates):
        work[f"_c{j}"] = frame[c].to_numpy(dtype=float)
        rhs.append(f"_c{j}")
        names[f"_c{j}"] = c
    if time is not None:
        work["_t"] = frame[time].to_numpy()
        rhs.append("i._t")
    if cluster is not None:
        work["_g"] = frame[cluster].to_numpy()
    formula = "_y ~ " + " + ".join(rhs) + " | _w + _f"
    fit = feols(formula, data=work, cluster="_g" if cluster else None)
    _, x_terms, _ = _parse_formula(formula)
    X, x_names = _materialize_rhs(work, x_terms)
    b = fit.params.reindex(x_names).to_numpy(dtype=float)
    omitted = [x_names[j] for j in np.flatnonzero(~np.isfinite(b))]
    xb = X @ np.where(np.isfinite(b), b, 0.0)

    def label(n: str) -> str:
        if n.startswith("_t::"):
            return f"{time}::{n[4:]}"
        return names.get(n, n)

    coef = pd.DataFrame(
        {
            "coef": fit.params.to_numpy(dtype=float),
            "std_err": fit.std_errors.to_numpy(dtype=float),
        },
        index=[label(n) for n in fit.params.index],
    )
    info = {
        "omitted": [label(n) for n in omitted],
        "se_type": fit.se_type,
        "n_params": int(len(x_names) - len(omitted)),
    }
    return xb, coef, info


def akm(
    data: pd.DataFrame,
    y: str,
    worker: str,
    firm: str,
    *,
    time: Optional[str] = None,
    covariates: Optional[Union[str, Sequence[str]]] = None,
    connected: Optional[str] = None,
    variance: str = "kss",
    leverage: str = "auto",
    n_draws: int = 200,
    seed: Optional[int] = None,
    cluster: Optional[str] = None,
) -> AKMResult:
    """Abowd-Kramarz-Margolis worker and firm effects with a variance
    decomposition (plug-in, Andrews et al. or Kline-Saggio-Solvsten).

    Not to be confused with the Adao-Kolesar-Morales shift-share standard
    errors (``sp.shift_share_se``).

    Parameters
    ----------
    data : DataFrame
        Person-period panel, one row per worker and period.
    y : str
        Outcome (e.g. log wage).
    worker, firm : str
        Worker and firm identifiers.
    time : str, optional
        Period column; its effects enter as indicators (first period the
        base) and are partialled out with the covariates.
    covariates : str or list of str, optional
        Time-varying controls, partialled out using the full model.
    connected : {'largest', 'leave_one_out', 'all'}, optional
        Estimation sample. ``'largest'``: the largest connected set of firms
        (most firms; ties by rows). ``'leave_one_out'``: its largest subset
        in which every observation can be dropped without disconnecting the
        graph (needed for the corrected variances; it drops workers seen
        once and observations that are a firm's only link). ``'all'``:
        every observation, effects normalised within each connected set
        (``felsdvreg``'s sample). Default: ``'leave_one_out'`` with a
        corrected variance, ``'largest'`` otherwise.
    variance : {'kss', 'homoskedastic', 'plugin', 'none'}, default 'kss'
        Variance decomposition. The plug-in is always reported alongside.
    leverage : {'auto', 'exact', 'jla'}, default 'auto'
        Leverages ``P_ii`` and quadratic-form weights: ``'exact'`` from dense
        firm-by-firm matrices, ``'jla'`` from ``n_draws`` random projections
        (Rademacher, seeded by ``seed``). ``'auto'`` is exact up to
        3000 firms.
    n_draws : int, default 200
        Random projections for ``leverage='jla'`` (three sets of sparse
        solves of this size).
    seed : int, optional
        Seed of the random projections.
    cluster : str, optional
        Cluster for the covariate standard errors (reghdfe conventions).

    Returns
    -------
    AKMResult

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> rows = [(w, int(rng.integers(0, 20)), t) for w in range(300) for t in range(4)]
    >>> d = pd.DataFrame(rows, columns=["worker", "firm", "year"])
    >>> alpha = rng.normal(size=300)
    >>> d["lnw"] = alpha[d.worker] + 0.3 * d.firm / 20 + rng.normal(0, 0.2, len(d))
    >>> res = sp.akm(d, "lnw", "worker", "firm", time="year")
    >>> list(res.decomposition.columns)
    ['plugin', 'homoskedastic', 'kss']

    References
    ----------
    abowd1999high, andrews2008high, kline2020leave
    """
    if variance not in _VARIANCES:
        raise MethodIncompatibility(
            f"akm: variance={variance!r} is not one of {_VARIANCES}.",
            recovery_hint="Use 'kss', 'homoskedastic', 'plugin' or 'none'.",
            diagnostics={"variance": variance},
        )
    corrected = variance in ("kss", "homoskedastic")
    if connected is None:
        connected = "leave_one_out" if corrected else "largest"
    if connected not in _CONNECTED:
        raise MethodIncompatibility(
            f"akm: connected={connected!r} is not one of {_CONNECTED}.",
            recovery_hint="Use 'largest', 'leave_one_out' or 'all'.",
            diagnostics={"connected": connected},
        )
    if leverage not in _LEVERAGE:
        raise MethodIncompatibility(
            f"akm: leverage={leverage!r} is not one of {_LEVERAGE}.",
            recovery_hint="Use 'auto', 'exact' or 'jla'.",
            diagnostics={"leverage": leverage},
        )
    if corrected and connected != "leave_one_out":
        raise MethodIncompatibility(
            f"akm: variance={variance!r} needs connected='leave_one_out'; on "
            f"the {connected!r} set some observations have leverage 1 (a "
            "worker seen once, a firm's only link) and no leave-one-out "
            "residual.",
            recovery_hint="Drop connected= (the default selects the "
            "leave-one-out set) or use variance='plugin'.",
            diagnostics={"variance": variance, "connected": connected},
        )
    if int(n_draws) < 1:
        raise MethodIncompatibility(
            f"akm: n_draws={n_draws} must be a positive integer.",
            recovery_hint="Pass n_draws >= 1 (a few hundred is typical).",
        )
    covs = [covariates] if isinstance(covariates, str) else list(covariates or [])
    cols = (
        [y, worker, firm]
        + covs
        + ([time] if time else [])
        + ([cluster] if cluster else [])
    )
    _require_columns(data, cols)

    n_input = len(data)
    pos = np.arange(n_input)
    complete = data[list(dict.fromkeys(cols))].notna().all(axis=1).to_numpy()
    frame = pd.DataFrame(
        {"_y": data[y].to_numpy(dtype=float)[complete]},
    )
    frame["_w"] = data[worker].to_numpy()[complete]
    frame["_f"] = data[firm].to_numpy()[complete]
    for c in covs + ([time] if time else []) + ([cluster] if cluster else []):
        frame[c] = data[c].to_numpy()[complete]
    pos = pos[complete]
    wc, wlev = pd.factorize(frame["_w"], sort=True)
    fc, flev = pd.factorize(frame["_f"], sort=True)
    n_w, n_f = len(wlev), len(flev)
    if len(frame) == 0:
        raise DataInsufficient(
            "akm: no complete rows (y, worker, firm and the covariates).",
            recovery_hint="Check the columns for missing values.",
        )

    dropped = {
        "missing": int(n_input - complete.sum()),
        "not_connected": 0,
        "not_leave_one_out": 0,
    }
    comp = _graph.components(wc, fc, n_w, n_f)
    n_comp_all = int(len(np.unique(comp)))
    if connected == "all":
        keep = np.ones(len(frame), dtype=bool)
    elif connected == "largest":
        keep = _graph.largest(comp, fc)
        dropped["not_connected"] = int((~keep).sum())
    else:
        big = _graph.largest(comp, fc)
        dropped["not_connected"] = int((~big).sum())
        sub = np.flatnonzero(big)
        wc_s, _ = pd.factorize(wc[sub])
        fc_s, _ = pd.factorize(fc[sub])
        sel, _nb = _graph.leave_one_out(
            wc_s, fc_s, int(wc_s.max()) + 1, int(fc_s.max()) + 1
        )
        keep = np.zeros(len(frame), dtype=bool)
        keep[sub[sel]] = True
        dropped["not_leave_one_out"] = int(big.sum() - keep.sum())
    if keep.sum() < 3:
        raise DataInsufficient(
            f"akm: {int(keep.sum())} observation(s) left in the {connected!r} set.",
            recovery_hint="Check that workers move between firms; a "
            "leave-one-out set needs firms linked by at least two moves.",
            diagnostics={"dropped": dropped},
        )
    frame = frame.loc[keep].reset_index(drop=True)
    pos = pos[keep]

    coef_table = None
    cov_info: Dict[str, Any] = {}
    xb = np.zeros(len(frame))
    if covs or time:
        xb, coef_table, cov_info = _partial_out(frame, covs, time, cluster)
    elif cluster is not None:
        warnings.warn(
            StatsPAIWarning(
                "akm: cluster= affects only the covariate standard errors and "
                "there are no covariates.",
                recovery_hint="Drop cluster=.",
            ),
            stacklevel=2,
        )
    ytil = frame["_y"].to_numpy(dtype=float) - xb

    wc, wlev = pd.factorize(frame["_w"], sort=True)
    fc, flev = pd.factorize(frame["_f"], sort=True)
    n_w, n_f = len(wlev), len(flev)
    fit = _kss.fit_twoway(wc, fc, ytil, n_w, n_f)
    n_comp = int(len(np.unique(fit.comp_firm)))

    decomposition = None
    lev_info: Dict[str, Any] = {"method": None}
    out_frame = pd.DataFrame(
        {
            worker: frame["_w"].to_numpy(),
            firm: frame["_f"].to_numpy(),
            "alpha": fit.alpha[wc],
            "psi": fit.psi[fc],
            "xb": xb,
            "resid": fit.resid,
        },
        index=data.index[pos],
    )
    if variance != "none":
        if n_comp > 1:
            warnings.warn(
                StatsPAIWarning(
                    f"akm: the sample has {n_comp} connected sets; effects are "
                    "normalised within each, so their variances across sets "
                    "are not identified and no decomposition is reported.",
                    recovery_hint="Use connected='largest' or "
                    "'leave_one_out' for a decomposition.",
                    diagnostics={"n_components": n_comp},
                ),
                stacklevel=2,
            )
        else:
            plug = _kss.plugin_components(fit)
            table = {"plugin": plug}
            if corrected:
                method = leverage
                if method == "auto":
                    method = "exact" if n_f <= EXACT_MAX_FIRMS else "jla"
                lev = (
                    _kss.leverages_exact(fit)
                    if method == "exact"
                    else _kss.leverages_jla(fit, int(n_draws), seed)
                )
                comps = _kss.corrected_components(
                    fit, lev, n_params_extra=int(cov_info.get("n_params", 0))
                )
                table["homoskedastic"] = comps["homoskedastic"]
                if variance == "kss":
                    table["kss"] = comps["kss"]
                lev_info = {
                    "method": method,
                    "n_draws": int(n_draws) if method == "jla" else None,
                    "seed": seed if method == "jla" else None,
                    "min_P": float(lev.P.min()),
                    "max_P": float(lev.P.max()),
                }
                out_frame["P_ii"] = lev.P
            var_y = float(np.var(ytil))
            decomposition = pd.DataFrame(table)
            decomposition.loc["var_y"] = var_y
            decomposition = decomposition.loc[
                ["var_y", "var_alpha", "var_psi", "cov", "var_resid"]
            ]

    T = np.bincount(wc, minlength=n_w)
    pairs = np.unique(np.column_stack([wc, fc]), axis=0)
    n_firms_w = np.bincount(pairs[:, 0], minlength=n_w)
    mover = n_firms_w > 1
    n_movers_f = np.bincount(pairs[:, 1], weights=mover[pairs[:, 0]], minlength=n_f)
    worker_effects = pd.DataFrame(
        {"worker": wlev, "alpha": fit.alpha, "n_obs": T, "mover": mover}
    )
    firm_effects = pd.DataFrame(
        {
            "firm": flev,
            "psi": fit.psi,
            "n_obs": fit.n_f_rows.astype(int),
            "n_movers": n_movers_f.astype(int),
        }
    )
    sample = {
        "n_input": int(n_input),
        "n_obs": int(len(frame)),
        "n_workers": int(n_w),
        "n_movers": int(mover.sum()),
        "n_firms": int(n_f),
        "n_components": n_comp,
        "n_components_input": n_comp_all,
        "dropped": dropped,
    }
    return AKMResult(
        worker_effects=worker_effects,
        firm_effects=firm_effects,
        decomposition=decomposition,
        coefficients=coef_table,
        sample=sample,
        leverage=lev_info,
        data=out_frame,
        variance=variance,
        connected=connected,
        y=y,
        worker=worker,
        firm=firm,
        diagnostics={"covariates": cov_info} if cov_info else {},
    )
