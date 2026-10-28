"""Adverse selection within an insurer's risk class (Azevedo et al. 2026).

Insurers price on covariates ``W`` but only through a coarse risk class;
consumers also know a predictor ``G`` that insurers may not use (a
polygenic index in the paper). Inside the standard risk class the spread
of the consumers' own risk is private information, and Hendren's implicit
tax summarises how much selection it creates. The pipeline has four
steps:

1. Probit of the loss ``L`` on ``W`` (Scenario 1) and on ``W`` and the
   standardized current predictor ``Gc`` (Scenario 2). For a future
   predictor ``Gf`` of known liability-scale predictive power ``R2f``
   (Scenario 3) the model ``Gf = W theta + V``, ``Gc = Gf + eps``,
   ``P(L = 1 | Gf, W) = Phi(Gf beta_g + W beta_w)`` is identified from a
   linear regression of ``Gc`` on ``W``, the probit of ``L`` on ``Gc``
   alone and the Scenario 2 probit.
2. Risk of every person in each scenario; in Scenario 3, ``n_draws``
   values of ``Gf`` per person from its normal distribution given
   ``Gc`` and ``W``.
3. The standard risk class: people whose Scenario 1 risk is between
   ``risk_class[0]`` and ``risk_class[1]`` times its mean in the sample.
4. The implicit tax curve of each scenario's risk distribution inside
   the class and its minimum up to ``upto_quantile`` (``t80``), with a
   bootstrap that repeats steps 1 to 4.

References
----------
[@azevedo2026genetic]
[@hendren2013private]
"""

from __future__ import annotations

import warnings
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

from .._input_validation import check_columns
from ..exceptions import (
    AssumptionWarning,
    DataInsufficient,
    IdentificationFailure,
    MethodIncompatibility,
)
from ._results import RiskClassSelectionResult
from .unraveling import _curve, _summaries

__all__ = ["risk_class_selection"]


def _probit_params(df: pd.DataFrame, y: str, x: List[str]) -> np.ndarray:
    """Probit coefficients ``[intercept, x...]`` of ``y`` on ``x``."""
    from ..regression.logit_probit import probit as _probit

    fit = _probit(data=df, y=y, x=x)
    dropped = fit.model_info.get("n_perfect_prediction_dropped") or 0
    if dropped:
        raise DataInsufficient(
            f"the probit of {y!r} predicts some outcomes perfectly",
            recovery_hint="Drop or merge the covariate cells that predict the "
            "loss perfectly, then rerun.",
        )
    params = fit.params.reindex(["Intercept"] + x)
    if params.isna().any():
        raise IdentificationFailure(
            f"the probit of {y!r} dropped collinear regressors "
            f"{params.index[params.isna()].tolist()}",
            recovery_hint="Remove collinear covariates.",
        )
    return np.asarray(params.to_numpy(dtype=float))


def _scenarios(r2_future: Any) -> Dict[str, float]:
    if r2_future is None:
        return {}
    if isinstance(r2_future, Mapping):
        out = {str(k): float(v) for k, v in r2_future.items()}
    elif np.ndim(r2_future) == 0:
        out = {"3": float(r2_future)}
    else:
        vals = [float(v) for v in r2_future]
        out = (
            {"3": vals[0]}
            if len(vals) == 1
            else {f"3.{i + 1}": v for i, v in enumerate(vals)}
        )
    for name, v in out.items():
        if not 0 < v < 1:
            raise MethodIncompatibility(
                f"r2_future[{name!r}] = {v} is not in (0, 1)",
                recovery_hint="Pass the liability-scale R2 of the future "
                "predictor, e.g. a SNP or twin heritability such as 0.18.",
            )
        if name in ("1", "2"):
            raise MethodIncompatibility(
                f"scenario name {name!r} is reserved",
                recovery_hint="Name future-predictor scenarios e.g. '3L', '3U'.",
            )
    return out


def _fit(
    L: np.ndarray, W: np.ndarray, gc: Optional[np.ndarray], scen3: Dict[str, float]
) -> Dict[str, Any]:
    """Step 1 on one sample: all parameters of the model."""
    n, k = W.shape
    wcols = [f"w{j}" for j in range(k)]
    df = pd.DataFrame(W, columns=wcols)
    df["_L"] = L
    out: Dict[str, Any] = {"gamma_w": _probit_params(df, "_L", wcols)}
    if gc is None:
        return out
    # Footnote 19: the current predictor is standardized (Var[Gc] = 1).
    mu, sd = float(gc.mean()), float(gc.std())
    if sd <= 0:
        raise DataInsufficient(
            "the predictor does not vary",
            recovery_hint="Pass a predictor with positive variance.",
        )
    z = (gc - mu) / sd
    df["_G"] = z
    gamma = _probit_params(df, "_L", ["_G"] + wcols)
    out.update(gc_mean=mu, gc_sd=sd, gamma=gamma)
    if not scen3:
        return out
    # Appendix A, Part 1: theta from the regression of Gc on W; the
    # residual variance is sigma_T^2 = sigma_V^2 + sigma_eps^2.
    X = np.column_stack([np.ones(n), W])
    theta, *_ = np.linalg.lstsq(X, z, rcond=None)
    sigma_t2 = float(np.mean((z - X @ theta) ** 2))
    # Part 2: liability-scale R2 of the probit of L on Gc alone.
    lam = _probit_params(df, "_L", ["_G"])[1]
    r2_c = lam**2 / (lam**2 + 1.0)
    out.update(theta=theta, sigma_t2=sigma_t2, r2_current=float(r2_c))
    future = {}
    gamma_g, gamma_w = gamma[1], np.r_[gamma[0], gamma[2:]]
    for name, r2_f in scen3.items():
        s_eps2 = 1.0 - r2_c / r2_f
        s_v2 = sigma_t2 - s_eps2  # equation (6)
        if s_eps2 < 0:
            raise IdentificationFailure(
                f"r2_future[{name!r}] = {r2_f} is below the R2 of the current "
                f"predictor ({r2_c:.4g}), so the noise variance is negative",
                recovery_hint="The future predictor must be at least as "
                "predictive as the current one; raise r2_future.",
            )
        if s_v2 <= 0:
            raise IdentificationFailure(
                f"scenario {name!r}: Var[V] = {s_v2:.4g} is not positive",
                recovery_hint="The covariates explain more of the predictor "
                "than the implied future predictor allows; check r2_future.",
            )
        if s_eps2 == 0:
            a, b, c2 = 1.0, 0.0, 0.0  # Gf = Gc
        else:
            prec = 1.0 / s_eps2 + 1.0 / s_v2
            a, b, c2 = (1.0 / s_eps2) / prec, (1.0 / s_v2) / prec, 1.0 / prec
        disc = a**2 - gamma_g**2 * c2
        if disc <= 0:
            raise IdentificationFailure(
                f"scenario {name!r}: |gamma_g| >= a / c, so beta_g is not "
                "identified (equation (8))",
                recovery_hint="Raise r2_future: the current predictor is too "
                "strong for a future predictor of this power.",
            )
        beta_g = gamma_g / np.sqrt(disc)  # equation (8)
        beta_w = gamma_w * np.sqrt(1.0 + c2 * beta_g**2) - b * theta * beta_g  # (9)
        future[name] = {
            "r2_future": r2_f,
            "sigma_eps2": float(s_eps2),
            "sigma_v2": float(s_v2),
            "a": float(a),
            "b": float(b),
            "c": float(np.sqrt(c2)),
            "beta_g": float(beta_g),
            "beta_w": beta_w,
        }
    out["future"] = future
    return out


def _risks(
    fit: Dict[str, Any],
    Wp: np.ndarray,
    gc: Optional[np.ndarray],
    risk_class: Optional[Tuple[float, float]],
    n_draws: int,
    rng: np.random.Generator,
) -> Tuple[Dict[str, Tuple[np.ndarray, np.ndarray]], np.ndarray, float, np.ndarray]:
    """Steps 2-3: risk distributions (values, weights) inside the class."""
    n = Wp.shape[0]
    X = np.column_stack([np.ones(n), Wp])
    r1 = stats.norm.cdf(X @ fit["gamma_w"])
    mean1 = float(r1.mean())
    if risk_class is None:
        inside = np.ones(n, dtype=bool)
    else:
        inside = (r1 >= risk_class[0] * mean1) & (r1 <= risk_class[1] * mean1)
    m = int(inside.sum())
    if m < 2:
        raise DataInsufficient(
            f"only {m} people fall in the risk class",
            recovery_hint="Widen risk_class or use more data.",
        )
    ones = np.ones(m)
    dist = {"1": (r1[inside], ones)}
    if gc is None:
        return dist, inside, mean1, r1
    z = (gc[inside] - fit["gc_mean"]) / fit["gc_sd"]
    Xi = X[inside]
    g = fit["gamma"]
    dist["2"] = (stats.norm.cdf(z * g[1] + Xi @ np.r_[g[0], g[2:]]), ones)
    for name, f in fit.get("future", {}).items():
        mean_gf = f["a"] * z + f["b"] * (Xi @ fit["theta"])
        gf = mean_gf[:, None] + f["c"] * rng.standard_normal((m, n_draws))
        risk = stats.norm.cdf(gf * f["beta_g"] + (Xi @ f["beta_w"])[:, None])
        dist[name] = (risk.ravel(), np.full(m * n_draws, 1.0 / n_draws))
    return dist, inside, mean1, r1


def _taxes(
    dist: Dict[str, Tuple[np.ndarray, np.ndarray]], upto: float
) -> Dict[str, Dict[str, Any]]:
    out = {}
    for name, (p, w) in dist.items():
        curve = _curve(p, w)
        s = _summaries(curve, upto)
        mean = float(np.sum(w * p) / np.sum(w))
        sd = float(np.sqrt(np.sum(w * (p - mean) ** 2) / np.sum(w)))
        out[name] = {"curve": curve, "t": s["t_min"], "mean": mean, "sd": sd}
    return out


def risk_class_selection(
    data: pd.DataFrame,
    loss: str,
    covariates: Sequence[str],
    pgi: Optional[str] = None,
    r2_future: Optional[Union[float, Sequence[float], Mapping[str, float]]] = None,
    risk_class: Optional[Tuple[float, float]] = (0.75, 1.25),
    at: Optional[Mapping[str, float]] = None,
    upto_quantile: float = 0.8,
    n_draws: int = 10,
    n_boot: int = 200,
    seed: Optional[int] = 0,
    alpha: float = 0.05,
) -> RiskClassSelectionResult:
    """Implicit tax of private risk inside an insurer's risk class.

    Implements the single-disease pipeline of Azevedo, Beauchamp and
    Karlsson Linnér (2026): estimate a probit risk model, build the risk
    distribution of the people insurers place in the standard risk class
    when consumers know only ``covariates`` (Scenario 1), also a current
    predictor ``pgi`` (Scenario 2), or a future predictor of given
    predictive power (Scenario 3), and report Hendren's minimum implicit
    tax up to the 80th percentile, ``t80``, with bootstrap intervals.

    Parameters
    ----------
    data : pandas.DataFrame
    loss : str
        0/1 indicator of the insured loss (e.g. disease by age 65).
    covariates : list of str
        Covariates ``W`` that insurers observe and use to form risk
        classes; in every scenario the consumer also knows them.
    pgi : str, optional
        Current predictor ``Gc`` that consumers know but insurers may not
        use (a polygenic index). It is standardized internally, which
        changes no risk. Needed for Scenarios 2 and 3.
    r2_future : float, list or dict, optional
        Liability-scale pseudo-R2 of a probit of the loss on a future
        predictor ``Gf`` (Assumption 5; e.g. SNP or twin heritability).
        Each value adds a Scenario 3; a dict names them, e.g.
        ``{'3L': 0.18, '3U': 0.57}``.
    risk_class : tuple of float or None, default (0.75, 1.25)
        The standard risk class is the people whose Scenario 1 risk lies
        in ``[lo * m, hi * m]`` with ``m`` its sample mean (bounds
        included). ``None`` puts everyone in one class (no underwriting).
    at : dict, optional
        Covariate values used when predicting risk, e.g. ``{'age': 65}``
        to predict the risk by age 65 (Step 2 of the paper). Estimation
        uses the observed values; the risk class and every scenario's
        risk use the overridden ones.
    upto_quantile : float, default 0.8
        The summary is the minimum implicit tax over risk values up to
        this quantile of the class's risk distribution.
    n_draws : int, default 10
        Draws of ``Gf`` per person in Scenario 3 (10 in the paper).
    n_boot : int, default 200
        Bootstrap replications (200 in the paper). Each one resamples
        people and repeats every step, including the probits and the risk
        class. ``0`` skips the bootstrap.
    seed : int, optional
        Seeds the Scenario 3 draws and the bootstrap.
    alpha : float, default 0.05

    Returns
    -------
    RiskClassSelectionResult
        ``table`` has one row per scenario with ``t80``, its bootstrap
        ``se`` and percentile interval, the mean and SD of risk in the
        class and the class size; ``curves`` holds each implicit tax
        curve; ``params`` the model parameters.

    Notes
    -----
    Scenario 3 follows Appendix A: ``theta`` and ``sigma_T^2`` come from
    the regression of the standardized ``Gc`` on ``W``; the liability
    R2 of the probit of the loss on ``Gc`` alone, ``R2c = l^2 / (l^2 + 1)``,
    gives ``sigma_eps^2 = 1 - R2c / R2f`` (equation (4)) and
    ``sigma_V^2 = sigma_T^2 - sigma_eps^2`` (equation (6)); equations (8)
    and (9) turn the Scenario 2 probit coefficients into ``beta_g`` and
    ``beta_w``. ``Gf`` given ``Gc`` and ``W`` is normal with mean
    ``a Gc + b W theta`` and variance ``c^2`` (Lemma 1). The step from the
    probit of the loss on ``Gc`` alone to ``R2c`` is exact when ``W`` is
    normal and an approximation otherwise, as in the paper. With
    ``r2_future`` equal to ``R2c`` the future predictor is the current one
    and Scenario 3 reproduces Scenario 2.

    The bootstrap reports percentile intervals; failed replications (a
    perfectly predicting probit, a non-identified Scenario 3) are dropped
    and counted in ``diagnostics['n_boot_failed']``.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> n = 4000
    >>> w = rng.normal(size=n)
    >>> gf = 0.3 * w + rng.normal(scale=0.7, size=n)
    >>> gc = gf + rng.normal(scale=0.6, size=n)
    >>> loss = (-1.5 + 0.5 * gf + 0.6 * w + rng.normal(size=n) > 0)
    >>> df = pd.DataFrame({"loss": loss.astype(int), "w": w, "gc": gc})
    >>> res = sp.risk_class_selection(df, "loss", ["w"], pgi="gc",
    ...                               r2_future=0.3, n_boot=0)
    >>> res.table.index.tolist()
    ['1', '2', '3']
    >>> bool(res.table.loc["2", "t80"] > res.table.loc["1", "t80"])
    True

    References
    ----------
    [@azevedo2026genetic]
    [@hendren2013private]
    """
    covs = [covariates] if isinstance(covariates, str) else list(covariates)
    if not covs:
        raise MethodIncompatibility(
            "covariates must name at least one variable",
            recovery_hint="Pass the variables insurers price on.",
        )
    if not 0 < upto_quantile <= 1:
        raise MethodIncompatibility(
            "upto_quantile must be in (0, 1]",
            recovery_hint="Pass e.g. upto_quantile=0.8.",
        )
    if not 0 < alpha < 1:
        raise MethodIncompatibility(
            "alpha must be in (0, 1)", recovery_hint="Pass e.g. alpha=0.05."
        )
    if risk_class is not None:
        lo, hi = (float(v) for v in risk_class)
        if not 0 <= lo < hi:
            raise MethodIncompatibility(
                f"risk_class must satisfy 0 <= lo < hi, got {risk_class!r}",
                recovery_hint="Pass e.g. risk_class=(0.75, 1.25).",
            )
        risk_class = (lo, hi)
    if int(n_draws) < 1:
        raise MethodIncompatibility(
            "n_draws must be at least 1", recovery_hint="Pass e.g. n_draws=10."
        )
    if n_boot and n_boot < 2:
        raise MethodIncompatibility(
            "n_boot must be 0 or at least 2", recovery_hint="Pass e.g. n_boot=200."
        )
    scen3 = _scenarios(r2_future)
    if scen3 and pgi is None:
        raise MethodIncompatibility(
            "r2_future needs the current predictor pgi",
            recovery_hint="Pass pgi='<column>' with r2_future.",
        )
    cols: Dict[str, Any] = {"loss": loss, "covariates": covs}
    if pgi is not None:
        cols["pgi"] = pgi
    check_columns(data, cols, function="risk_class_selection")
    at = dict(at or {})
    unknown = [c for c in at if c not in covs]
    if unknown:
        raise MethodIncompatibility(
            f"at= names variables that are not covariates: {unknown}",
            recovery_hint="Only covariates can be set for prediction.",
        )

    used = [loss] + covs + ([pgi] if pgi is not None else [])
    df = data.loc[:, list(dict.fromkeys(used))].dropna()
    n = len(df)
    if n < len(covs) + 10:
        raise DataInsufficient(
            f"only {n} complete observations",
            recovery_hint="Use fewer covariates or more data.",
        )
    L = df[loss].to_numpy(dtype=float)
    if not np.isin(L, (0.0, 1.0)).all():
        raise MethodIncompatibility(
            f"{loss!r} must be coded 0/1",
            recovery_hint=f"Recode {loss!r} as a 0/1 loss indicator.",
        )
    W = df[covs].to_numpy(dtype=float)
    Wp = W.copy()
    for col, val in at.items():
        Wp[:, covs.index(col)] = float(val)
    gc = df[pgi].to_numpy(dtype=float) if pgi is not None else None
    upto = float(upto_quantile)

    rng = np.random.default_rng(seed)
    fit = _fit(L, W, gc, scen3)
    dist, inside, mean1, r1 = _risks(fit, Wp, gc, risk_class, int(n_draws), rng)
    point = _taxes(dist, upto)
    t_nu = _summaries(_curve(r1, np.ones(n)), upto)["t_min"]

    names = list(point)
    boot = np.full((int(n_boot), len(names)), np.nan)
    n_failed = 0
    for b in range(int(n_boot)):
        idx = rng.integers(0, n, size=n)
        try:
            gb = None if gc is None else gc[idx]
            fb = _fit(L[idx], W[idx], gb, scen3)
            db, *_ = _risks(fb, Wp[idx], gb, risk_class, int(n_draws), rng)
            tb = _taxes(db, upto)
            boot[b] = [tb[s]["t"] for s in names]
        except (DataInsufficient, IdentificationFailure):
            n_failed += 1
    if n_failed:
        warnings.warn(
            f"{n_failed} of {n_boot} bootstrap replications failed and were " "dropped",
            AssumptionWarning,
            stacklevel=2,
        )

    rows = []
    for j, s in enumerate(names):
        row = {
            "t80": point[s]["t"],
            "se": np.nan,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
            "risk_mean": point[s]["mean"],
            "risk_sd": point[s]["sd"],
            "n_class": int(inside.sum()),
        }
        if n_boot and np.isfinite(boot[:, j]).sum() >= 2:
            col = boot[:, j][np.isfinite(boot[:, j])]
            row["se"] = float(np.std(col, ddof=1))
            row["ci_lower"] = float(np.quantile(col, alpha / 2))
            row["ci_upper"] = float(np.quantile(col, 1 - alpha / 2))
        rows.append(row)
    table = pd.DataFrame(rows, index=pd.Index(names, name="scenario"))

    params: Dict[str, Any] = {
        "gamma_scenario1": pd.Series(fit["gamma_w"], index=["Intercept"] + covs)
    }
    if gc is not None:
        params["gamma_scenario2"] = pd.Series(
            fit["gamma"], index=["Intercept", pgi] + covs
        )
        params["pgi_mean"] = fit["gc_mean"]
        params["pgi_sd"] = fit["gc_sd"]
    if scen3:
        params["theta"] = pd.Series(fit["theta"], index=["Intercept"] + covs)
        params["sigma_t2"] = fit["sigma_t2"]
        params["r2_current"] = fit["r2_current"]
        params["future"] = {
            s: {
                **{k: v for k, v in f.items() if k != "beta_w"},
                "beta_w": pd.Series(f["beta_w"], index=["Intercept"] + covs),
            }
            for s, f in fit["future"].items()
        }

    risk = pd.DataFrame({"risk_scenario1": r1, "in_class": inside}, index=df.index)
    return RiskClassSelectionResult(
        table=table,
        curves={s: point[s]["curve"] for s in names},
        params=params,
        risk=risk,
        risk_class=risk_class,
        mean_risk=mean1,
        t_no_underwriting=float(t_nu),
        upto_quantile=upto,
        n_boot=int(n_boot),
        alpha=alpha,
        n_obs=n,
        boot_draws=pd.DataFrame(boot, columns=names),
        diagnostics={
            "n_dropped_missing": int(len(data) - n),
            "n_boot_failed": n_failed,
            "n_draws": int(n_draws),
            "at": at,
        },
    )
