"""Panel event study of a policy that can be continuous and change often.

``sp.xtevent`` follows Stata's ``xtevent`` (Freyaldenhoven, Hansen, Perez
Perez and Shapiro) [@freyaldenhoven2019event]. For a window ``(lo, hi)``
(``lo <= -1 <= 0 <= hi``) the regressors are

* ``dz_{t-k} = z_{t-k} - z_{t-k-1}`` for ``k = lo, ..., hi``;
* the right endpoint ``z_{t-hi-1}`` (all changes at least ``hi + 1``
  periods ago), labelled ``hi + 1``;
* the left endpoint ``1 - z_{t-lo}`` (all changes more than ``|lo|``
  periods ahead), labelled ``lo - 1``;

with unit and period effects, one event time (``norm``, default ``-1``)
normalised to zero. Rows whose leads or lags fall outside the panel drop
out, as in ``xtevent`` without ``impute()``. For a 0/1 absorbing policy the
regressors are the usual event-time dummies with binned endpoints.

``engine='areg'`` (``xtevent``'s default) fits by ``areg`` conventions
(absorbed unit effects charged in the small-sample factor);
``engine='reghdfe'`` by ``reghdfe``'s (``sp.hdfe_ols``).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

from .._aliases import accepts_aliases
from ..core.results import CausalResult
from ..exceptions import DataInsufficient, MethodIncompatibility


def _window(window: Union[int, Sequence[int]]) -> Tuple[int, int]:
    if isinstance(window, (int, np.integer)) and not isinstance(window, bool):
        if int(window) < 1:
            raise MethodIncompatibility("xtevent: window must be a positive integer.")
        return -int(window), int(window)
    try:
        lo, hi = (int(v) for v in window)  # type: ignore[union-attr]
    except (TypeError, ValueError):
        raise MethodIncompatibility(
            "xtevent: window must be a positive integer or a pair (lo, hi)."
        ) from None
    if lo > -1 or hi < 0:
        raise MethodIncompatibility(
            "xtevent: window (lo, hi) needs lo <= -1 and hi >= 0 " f"(got {(lo, hi)})."
        )
    return lo, hi


def _two_sided_p(stat: float, df: float) -> float:
    """Two-sided p-value on t(df), or the normal when ``df`` is infinite."""
    if not np.isfinite(stat):
        return float("nan")
    if np.isfinite(df):
        return float(2 * stats.t.sf(abs(stat), df))
    return float(2 * stats.norm.sf(abs(stat)))


def _label(k: int) -> str:
    return f"_k_eq_m{-k}" if k < 0 else f"_k_eq_p{k}"


@accepts_aliases(controls="covariates")
def xtevent(
    data: pd.DataFrame,
    y: str,
    policy: str,
    panel: str,
    time: str,
    *,
    window: Union[int, Sequence[int], None] = None,
    covariates: Optional[List[str]] = None,
    norm: int = -1,
    static: bool = False,
    engine: str = "areg",
    absorb: Optional[str] = None,
    cluster: Optional[str] = None,
    vce: Optional[str] = None,
    alpha: float = 0.05,
) -> CausalResult:
    """Event study of a (possibly continuous, repeatedly changing) policy.

    Equivalent to Stata's ``xtevent y controls, policyvar(policy)
    panelvar(panel) timevar(time) window(...)``.

    Parameters
    ----------
    data : pd.DataFrame
        Panel in long format.
    y : str
        Outcome.
    policy : str
        Policy variable ``z``: 0/1 or continuous, and free to change any
        number of times within a unit.
    panel, time : str
        Unit and (integer) period identifiers.
    window : int or (int, int)
        ``window=2`` estimates event times -2..2 with endpoints -3 and +3;
        ``window=(-3, 4)`` an asymmetric window. Required unless
        ``static=True``.
    covariates : list of str, optional
        Additional regressors (``controls=`` is accepted as an alias).
    norm : int, default -1
        Event time normalised to zero (Stata ``norm()``); must lie inside the
        window.
    static : bool, default False
        Stata ``static``: regress ``y`` on the level of ``policy`` (with the
        controls and effects) instead of an event study; the headline is
        its coefficient.
    engine : {'areg', 'reghdfe'}, default 'areg'
        Small-sample conventions of Stata ``xtevent``'s default (``areg``)
        or of its ``reghdfe`` option.
    absorb : str, optional
        Further fixed effects, ``sp.hdfe_ols`` syntax (Stata
        ``addabsorb()``); requires ``engine='reghdfe'``.
    cluster : str, optional
        Cluster variable (Stata ``vce(cluster ...)``).
    vce : {'robust'}, optional
        ``'robust'``: heteroskedasticity-robust SEs (Stata ``vce(robust)``).
    alpha : float, default 0.05

    Returns
    -------
    CausalResult
        Headline: the mean of the post-event coefficients (event times
        ``0 .. hi + 1``). ``model_info['event_study']`` has one row per
        event time (the normalised one at zero), ``['vcov_event']`` their
        covariance, ``['diffavg']`` Stata's ``diffavg`` contrast (mean of
        the post coefficients minus mean of the pre coefficients, the
        normalised one included as zero) with its SE, and
        ``['coef_table']`` every coefficient, controls included.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> d = pd.DataFrame({"i": np.repeat(np.arange(40), 12),
    ...                   "t": np.tile(np.arange(12), 40)})
    >>> d["z"] = np.repeat(rng.integers(0, 2, 40), 12) * (d["t"] >= 6)
    >>> d["y"] = d["z"] + rng.normal(size=len(d))
    >>> r = sp.xtevent(d, y="y", policy="z", panel="i", time="t", window=2)
    >>> sorted(r.model_info["event_study"]["relative_time"])
    [-3, -2, -1, 0, 1, 2, 3]

    References
    ----------
    [@freyaldenhoven2019event]
    """
    if engine not in ("areg", "reghdfe"):
        raise MethodIncompatibility("xtevent: engine must be 'areg' or 'reghdfe'.")
    if absorb is not None and engine != "reghdfe":
        raise MethodIncompatibility(
            "xtevent: absorb= (Stata addabsorb()) requires engine='reghdfe'."
        )
    if vce is not None and str(vce).lower() != "robust":
        raise MethodIncompatibility("xtevent: vce must be 'robust' (or use cluster=).")
    controls = list(covariates or [])
    need = [y, policy, panel, time] + controls + ([cluster] if cluster else [])
    missing = [c for c in need if c not in data.columns]
    if missing:
        raise MethodIncompatibility(f"xtevent: columns not found: {missing}")

    df = data[list(dict.fromkeys(need))].copy()
    df = df.sort_values([panel, time]).reset_index(drop=True)
    tt = pd.to_numeric(df[time], errors="coerce")
    if tt.isna().any() or not np.allclose(tt, np.round(tt)):
        raise MethodIncompatibility("xtevent: time must be integer periods.")

    if static:
        regs = [policy] + controls
        names_event: List[str] = []
        ks: List[int] = []
    else:
        if window is None:
            raise MethodIncompatibility("xtevent: pass window= (or static=True).")
        lo, hi = _window(window)
        if not (lo <= norm <= hi):
            raise MethodIncompatibility(
                f"xtevent: norm={norm} must be an event time inside the "
                f"window {(lo, hi)}."
            )
        z = pd.to_numeric(df[policy], errors="coerce").astype(float)
        # Leads / lags on the calendar: a gap in a unit's periods is missing.
        key = pd.MultiIndex.from_arrays([df[panel], tt.astype(np.int64)])
        zs = pd.Series(z.to_numpy(), index=key)

        def shift(s: int) -> np.ndarray:
            idx = pd.MultiIndex.from_arrays([df[panel], tt.astype(np.int64) - s])
            return zs.reindex(idx).to_numpy()

        ks = list(range(lo - 1, hi + 2))
        for k in range(lo, hi + 1):
            df[_label(k)] = shift(k) - shift(k + 1)
        df[_label(hi + 1)] = shift(hi + 1)
        df[_label(lo - 1)] = 1.0 - shift(lo)
        names_event = [_label(k) for k in ks if k != norm]
        regs = names_event + controls

    use = df.dropna(subset=[y] + regs + ([cluster] if cluster else [])).copy()
    if len(use) == 0:
        raise DataInsufficient("xtevent: no observations with every lead and lag.")
    fe = f"{panel} + {time}" + (f" + {absorb}" if absorb else "")
    fml = f"{y} ~ {' + '.join(regs)} | {fe}"
    if engine == "areg":
        from ..core.ssc_presets import ssc as _ssc
        from ..fixest.wrapper import feols as _feols

        vcov: Any = (
            {"CRV1": cluster} if cluster else ("hetero" if vce is not None else "iid")
        )
        fit = _feols(fml, use, vcov=vcov, ssc=_ssc("areg"))
        params, ses = fit.params, fit.std_errors
        V_all = fit.vcov()
        df_inf = fit._inference_df() if hasattr(fit, "_inference_df") else np.inf
    else:
        from ..panel.feols import hdfe_ols as _hdfe_ols

        kw: Dict[str, Any] = {}
        if cluster:
            kw["cluster"] = cluster
        elif vce is not None:
            kw["vce"] = "robust"
        fit = _hdfe_ols(fml, data=use, **kw)
        params, ses = fit.params, fit.std_errors
        V_all = fit.cov_params()
        df_inf = fit.df_inference
    crit = (
        stats.t.ppf(1 - alpha / 2, df_inf)
        if np.isfinite(df_inf)
        else stats.norm.ppf(1 - alpha / 2)
    )

    coef_table = pd.DataFrame(
        {"estimate": params.astype(float), "se": ses.astype(float)}
    )
    coef_table.index = [str(i) for i in coef_table.index]
    n = int(len(use))
    common = dict(alpha=alpha, n_obs=n, _citation_key="xtevent")
    if static:
        b, se = float(params[policy]), float(ses[policy])
        p = _two_sided_p(b / se, df_inf)
        return CausalResult(
            method="Panel event study, static model (xtevent)",
            estimand="Effect of policy",
            estimate=b,
            se=se,
            pvalue=p,
            ci=(b - crit * se, b + crit * se),
            detail=coef_table.reset_index(names="term"),
            model_info=dict(
                n_obs=n,
                engine=engine,
                static=True,
                coef_table=coef_table,
                formula=fml,
            ),
            **common,
        )

    Vdf = pd.DataFrame(
        np.asarray(V_all), index=coef_table.index, columns=coef_table.index
    )
    rows = []
    for k in ks:
        nm = _label(k)
        if k == norm:
            b = s = 0.0
        else:
            b, s = float(coef_table.loc[nm, "estimate"]), float(
                coef_table.loc[nm, "se"]
            )
        rows.append(
            dict(
                relative_time=k,
                estimate=b,
                att=b,
                se=s,
                ci_lower=b - crit * s,
                ci_upper=b + crit * s,
                pvalue=(_two_sided_p(b / s, df_inf) if s > 0 else np.nan),
                is_reference=k == norm,
                is_endpoint=k in (lo - 1, hi + 1),
            )
        )
    es = pd.DataFrame(rows)
    V_ev = Vdf.loc[names_event, names_event]
    V_ev.index = V_ev.columns = [k for k in ks if k != norm]

    def _contrast(w: Dict[int, float]) -> Tuple[float, float]:
        keys = [k for k in w if k != norm]
        vec = np.array([w[k] for k in keys])
        b = float(
            sum(w[k] * es.set_index("relative_time").loc[k, "estimate"] for k in keys)
        )
        v = float(vec @ V_ev.loc[keys, keys].to_numpy() @ vec)
        return b, float(np.sqrt(max(v, 0.0)))

    post = [k for k in ks if k >= 0 and k != norm]
    pre = [k for k in ks if k < 0]
    att, att_se = _contrast({k: 1.0 / len(post) for k in post})
    w_diff = {k: 1.0 / len(post) for k in post}
    for k in pre:
        w_diff[k] = w_diff.get(k, 0.0) - 1.0 / len(pre)
    d_b, d_se = _contrast(w_diff)
    p_att = _two_sided_p(att / att_se, df_inf) if att_se > 0 else np.nan
    return CausalResult(
        method="Panel event study (xtevent)",
        estimand="Mean post-event effect",
        estimate=att,
        se=att_se,
        pvalue=p_att,
        ci=(att - crit * att_se, att + crit * att_se),
        detail=es,
        model_info=dict(
            event_study=es,
            vcov_event=V_ev,
            event_study_vcov=V_ev,
            diffavg={"estimate": d_b, "se": d_se},
            coef_table=coef_table,
            window=(lo, hi),
            norm=norm,
            engine=engine,
            n_obs=n,
            n_units=int(use[panel].nunique()),
            formula=fml,
            cluster=cluster,
            df_inference=df_inf,
        ),
        **common,
    )


# Mirrors paper.bib ``freyaldenhoven2019event``.
CausalResult._CITATIONS["xtevent"] = (
    "@article{freyaldenhoven2019event,\n"
    "  title={Pre-event Trends in the Panel Event-Study Design},\n"
    "  author={Freyaldenhoven, Simon and Hansen, Christian and Shapiro, Jesse M.},\n"
    "  journal={Working paper (Federal Reserve Bank of Philadelphia)},\n"
    "  year={2019},\n"
    "  doi={10.21799/frbp.wp.2019.27}\n"
    "}"
)
