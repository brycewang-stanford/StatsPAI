"""Causal mediation with one instrument (Dippel, Ferrara and Heblich).

The design: a treatment ``T`` and a mediator ``M``, both possibly
endogenous, and a single instrument ``Z`` for the treatment. Dippel,
Ferrara and Heblich show that under their assumptions (the unobservables
that confound ``T`` and ``M`` are not the ones that confound ``M`` and
``Y``) the same instrument identifies three effects [@dippel2020causal]:

1. the total effect of ``T`` on ``Y``: IV of ``Y`` on ``T`` with ``Z``;
2. the effect of ``T`` on ``M``: IV of ``M`` on ``T`` with ``Z``;
3. the effect of ``M`` on ``Y`` holding ``T`` fixed, and the direct effect
   of ``T``: IV of ``Y`` on ``(T, M)`` with instruments ``(Z, T)``.

The indirect effect is the product of the coefficient in (2) and the
``M`` coefficient in (3); the direct effect is the ``T`` coefficient in
(3).

Numerical contract
------------------
This follows Stata's ``ivmediate`` (SSC, version 1.0.4) exactly, which was
used as a black box (``tests/reference_parity/test_iv_mediate_stata_parity.py``):

- covariates, the fixed effect and the constant are partialled out of
  ``Y``, ``T``, ``M`` and ``Z`` first (Frisch-Waugh-Lovell);
- the three equations are estimated as one just-identified GMM system with
  the equations treated as independent, so the variance of each equation
  is its own IV sandwich and the cross-equation covariances are zero. No
  small-sample factor anywhere: ``e'e / n`` without ``vce``, HC0 with
  ``vce="robust"``, and a cluster sum without ``G / (G - 1)`` with
  ``cluster=``. Covariates and absorbed levels are not subtracted from
  ``n`` either;
- the indirect effect's variance is the delta method on the product;
- the two first-stage F statistics are the Cragg-Donald Wald F (without
  ``vce``) or the Kleibergen-Paap Wald F (robust / cluster), with
  ``ivmediate``'s degrees-of-freedom factors.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

from .._result_serialize import ResultProtocolMixin
from ..core._call_record import records_call
from ..core._covariates import expands_categorical_covariates
from ..core._vcov_spec import accepts_vce
from ..exceptions import (
    AssumptionWarning,
    DataInsufficient,
    IdentificationFailure,
    MethodIncompatibility,
)

_EFFECTS = ("total", "direct", "indirect")
#: First-stage F below which the instrument is flagged as weak
#: (the Staiger-Stock rule of thumb).
_WEAK_F = 10.0


@dataclass
class IVMediateResult(ResultProtocolMixin):
    """Result of :func:`iv_mediate`.

    ``estimates`` has one row per effect (total, direct, indirect) with the
    coefficient, standard error, z statistic, p-value and confidence
    interval; ``vcov`` is their 3 x 3 covariance. ``stages`` holds the three
    underlying IV fits (coefficients and covariance), the same numbers
    ``ivmediate, full`` prints.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(1)
    >>> z, v, u = rng.standard_normal((3, 500))
    >>> t = z + v
    >>> m = 0.5 * t + v + u
    >>> df = pd.DataFrame({"y": t + m + u, "t": t, "m": m, "z": z})
    >>> res = sp.iv_mediate(df, y="y", treat="t", mediator="m", instrument="z")
    >>> type(res).__name__
    'IVMediateResult'
    >>> list(res.estimates.index)
    ['total', 'direct', 'indirect']
    """

    _citation_keys: ClassVar[Tuple[str, ...]] = ("dippel2020causal",)

    estimates: pd.DataFrame
    vcov: pd.DataFrame
    share_mediated: float
    share_direct: float
    fstat_treatment: float
    fstat_mediator: float
    n_obs: int
    vce: str
    y: str
    treat: str
    mediator: str
    instrument: str
    level: float = 0.95
    n_clusters: Optional[int] = None
    stages: Dict[str, Any] = field(default_factory=dict)
    model_info: Dict[str, Any] = field(default_factory=dict)
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    @property
    def total_effect(self) -> float:
        return float(self.estimates.loc["total", "coef"])

    @property
    def direct_effect(self) -> float:
        return float(self.estimates.loc["direct", "coef"])

    @property
    def indirect_effect(self) -> float:
        return float(self.estimates.loc["indirect", "coef"])

    def tidy(self) -> pd.DataFrame:
        """The estimates table with the effect name as a column."""
        return self.estimates.reset_index().rename(columns={"index": "effect"})

    def to_frame(self) -> pd.DataFrame:
        """The estimates table (one row per effect)."""
        return self.estimates.copy()

    def summary(self) -> str:
        e = self.estimates
        pct = int(round(100 * self.level))
        vce = {"iid": "", "robust": "robust", "cluster": "cluster"}[self.vce]
        lines = [
            "Linear IV mediation analysis (Dippel, Ferrara and Heblich)",
            "-" * 72,
            f"  Outcome    : {self.y:<20} Observations : {self.n_obs}",
            f"  Treatment  : {self.treat:<20} Instrument   : {self.instrument}",
            f"  Mediator   : {self.mediator:<20} SE           : {vce or 'iid'}"
            + (f" ({self.n_clusters} clusters)" if self.n_clusters else ""),
            "",
            f"  {'':<16}{'coef':>10}{'std.err':>11}{'z':>8}{'P>|z|':>9}"
            f"   [{pct}% conf. interval]",
        ]
        for name in _EFFECTS:
            r = e.loc[name]
            lines.append(
                f"  {name + ' effect':<16}{r['coef']:>10.4f}{r['se']:>11.4f}"
                f"{r['z']:>8.2f}{r['pvalue']:>9.3f}"
                f"   {r['ci_lower']:>9.4f} {r['ci_upper']:>9.4f}"
            )
        fname = "Cragg-Donald" if self.vce == "iid" else "Kleibergen-Paap"
        lines += [
            "",
            f"  Mediator explains {100 * self.share_mediated:.2f}% of the total "
            "effect (indirect / total).",
            f"  {fname} Wald F for the excluded instrument:",
            f"    first stage one (T on Z)    : {self.fstat_treatment:.3f}",
            f"    first stage two (M on Z | T): {self.fstat_mediator:.3f}",
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"IVMediateResult(total={self.total_effect:+.4f}, "
            f"direct={self.direct_effect:+.4f}, "
            f"indirect={self.indirect_effect:+.4f}, n={self.n_obs})"
        )


def _as_name(arg: Any, what: str) -> str:
    if isinstance(arg, (list, tuple)):
        if len(arg) != 1:
            raise MethodIncompatibility(
                f"iv_mediate takes exactly one {what}; got {len(arg)}.",
                recovery_hint=(
                    "ivmediate identifies the three effects with a single "
                    "instrument; pass one column."
                ),
            )
        arg = arg[0]
    if not isinstance(arg, str):
        raise MethodIncompatibility(
            f"{what} must be a column name, got {type(arg).__name__}.",
            recovery_hint="Pass the column name of the variable in data.",
        )
    return arg


def _partial_out(
    M: np.ndarray, W: np.ndarray, fe: Optional[np.ndarray]
) -> Tuple[np.ndarray, int]:
    """Residuals of every column of ``M`` on ``[1 or FE dummies, W]``.

    Returns the residuals and the number of parameters partialled out
    (``areg``'s ``e(N) - e(df_r)``: covariate rank plus the number of FE
    levels, or plus one for the constant).
    """
    if fe is None:
        Mc = M - M.mean(axis=0)
        Wc = W - W.mean(axis=0) if W.shape[1] else W
        k_fe = 1
    else:
        G = int(fe.max()) + 1
        counts = np.bincount(fe, minlength=G).astype(float)

        def _within(A: np.ndarray) -> np.ndarray:
            sums = np.zeros((G, A.shape[1]))
            np.add.at(sums, fe, A)
            out: np.ndarray = A - (sums / counts[:, None])[fe]
            return out

        Mc = _within(M)
        Wc = _within(W) if W.shape[1] else W
        k_fe = G
    if not W.shape[1]:
        return Mc, k_fe
    scale = max(float(np.abs(Wc).max()), 1.0)
    rank = int(np.linalg.matrix_rank(Wc, tol=1e-10 * scale * np.sqrt(Wc.shape[0])))
    b, *_ = np.linalg.lstsq(Wc, Mc, rcond=None)
    out: np.ndarray = Mc - Wc @ b
    return out, k_fe + rank


def _meat(Z: np.ndarray, e: np.ndarray, codes: Optional[np.ndarray]) -> np.ndarray:
    """``sum_i e_i^2 z_i z_i'``-type middle matrix; ``None`` codes = iid."""
    n = Z.shape[0]
    if codes is None:
        out: np.ndarray = (e @ e / n) * (Z.T @ Z)
        return out
    G = int(codes.max()) + 1
    s = np.zeros((G, Z.shape[1]))
    np.add.at(s, codes, Z * e[:, None])
    out = s.T @ s
    return out


def _iv(
    y: np.ndarray, X: np.ndarray, Z: np.ndarray, codes: Optional[np.ndarray]
) -> Tuple[np.ndarray, np.ndarray]:
    """Just-identified IV ``(Z'X)^{-1} Z'y`` and its sandwich variance."""
    A = Z.T @ X
    b = np.linalg.solve(A, Z.T @ y)
    e = y - X @ b
    Ai = np.linalg.inv(A)
    V: np.ndarray = Ai @ _meat(Z, e, codes) @ Ai.T
    return b, V


def _wald_first_stage(
    x: np.ndarray, z: np.ndarray, codes: Optional[np.ndarray]
) -> float:
    """Wald statistic for ``pi = 0`` in ``x = z pi + v`` (one instrument)."""
    zz = float(z @ z)
    pi = float(z @ x) / zz
    v = x - z * pi
    var = float(_meat(z[:, None], v, codes)[0, 0]) / zz**2
    return pi**2 / var


@records_call("iv_mediate")
@accepts_vce(
    default=None,
    native={"robust": {"robust": True}, "nonrobust": {"robust": False}},
)
@expands_categorical_covariates("covariates")
def iv_mediate(
    data: pd.DataFrame,
    y: str,
    treat: str,
    mediator: str,
    instrument: Union[str, Sequence[str]],
    *,
    covariates: Optional[Sequence[str]] = None,
    absorb: Optional[str] = None,
    cluster: Optional[str] = None,
    robust: bool = False,
    level: float = 0.95,
) -> IVMediateResult:
    """Total, direct and indirect effects with one instrument (``ivmediate``).

    Implements the instrumental-variables mediation estimator of Dippel,
    Ferrara and Heblich [@dippel2020causal], reproducing Stata's
    ``ivmediate`` (SSC) to rounding. The treatment and the mediator may both
    be endogenous; the single instrument ``instrument`` moves the treatment.
    Three IV regressions on variables with ``covariates``, the fixed effect
    ``absorb`` and the constant partialled out:

    - ``Y`` on ``T`` instrumented by ``Z``: the total effect;
    - ``M`` on ``T`` instrumented by ``Z``: the effect of ``T`` on ``M``;
    - ``Y`` on ``T`` and ``M``, instruments ``Z`` and ``T``: the direct
      effect (the ``T`` coefficient) and the effect of ``M`` on ``Y``.

    The indirect effect is the product of the second coefficient and the
    ``M`` coefficient of the third; its standard error is the delta method
    with the equations treated as independent, as ``ivmediate`` does.

    Parameters
    ----------
    data : pandas.DataFrame
    y : str
        Outcome.
    treat : str
        Treatment (may be endogenous).
    mediator : str
        Mediator (may be endogenous).
    instrument : str
        The excluded instrument for the treatment. Exactly one, as in
        ``ivmediate``.
    covariates : list of str, optional
        Exogenous controls, partialled out of every variable. Categorical
        columns are expanded into dummies.
    absorb : str, optional
        One fixed-effect identifier, absorbed like ``areg, absorb()``
        (``ivmediate``'s ``absorb()``).
    cluster : str, optional
        Cluster variable; cluster-robust variances with no small-sample
        factor (``vce(cluster)`` in ``ivmediate``). ``vce="cluster g"``
        sets it.
    robust : bool, default False
        Heteroskedasticity-robust (HC0) variances, ``vce(robust)``.
        ``vce="robust"`` sets it. Without ``robust`` or ``cluster`` the
        variances assume homoskedastic errors with ``e'e / n``.
    level : float, default 0.95
        Confidence level for the intervals.

    Returns
    -------
    IVMediateResult
        ``.estimates`` (total, direct, indirect), ``.vcov``,
        ``.share_mediated`` (indirect / total; ``ivmediate``'s "mediator
        explains ...%"), ``.share_direct`` (direct / total), the two
        first-stage F statistics and ``.stages`` (the three IV fits).
        The three effects come from separate regressions, so ``direct +
        indirect`` equals ``total`` only approximately.

    Raises
    ------
    MethodIncompatibility
        A variable is used in two roles, or more than one instrument.
    IdentificationFailure
        The treatment, mediator or instrument has no variation left after
        partialling out, or the mediator is a linear function of the
        treatment and the controls.

    Warns
    -----
    AssumptionWarning
        When either first-stage F statistic is below 10 (weak instrument).

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> n = 2000
    >>> z, v, u = rng.standard_normal((3, n))
    >>> t = 0.8 * z + v + 0.5 * rng.standard_normal(n)
    >>> m = 0.6 * t + v + u + 0.5 * rng.standard_normal(n)
    >>> yv = 0.4 * t + 0.7 * m + u + rng.standard_normal(n)
    >>> df = pd.DataFrame({"y": yv, "t": t, "m": m, "z": z})
    >>> res = sp.iv_mediate(df, y="y", treat="t", mediator="m", instrument="z")
    >>> abs(res.total_effect - 0.82) < 0.15
    True
    >>> abs(res.indirect_effect - 0.42) < 0.15
    True

    References
    ----------
    [@dippel2020causal]
    """
    if not isinstance(data, pd.DataFrame):
        raise MethodIncompatibility(
            "iv_mediate needs a pandas DataFrame.",
            recovery_hint="Pass the data as a DataFrame with named columns.",
        )
    instrument = _as_name(instrument, "instrument")
    for nm, what in ((y, "y"), (treat, "treat"), (mediator, "mediator")):
        _as_name(nm, what)
    if absorb is not None and not isinstance(absorb, str):
        raise MethodIncompatibility(
            "absorb takes one column name (ivmediate's absorb()).",
            recovery_hint="Add further fixed effects as categorical covariates.",
        )
    if not 0 < level < 1:
        raise MethodIncompatibility(
            f"level must be in (0, 1), got {level!r}.",
            recovery_hint="Use e.g. level=0.95.",
        )
    if cluster is not None and robust:
        robust = False  # cluster-robust already covers heteroskedasticity
    covs: List[str] = list(covariates or [])

    roles = {"y": [y], "treat": [treat], "mediator": [mediator]}
    roles["instrument"] = [instrument]
    roles["covariates"] = covs
    seen: Dict[str, str] = {}
    for role, cols in roles.items():
        for c in cols:
            if c in seen and seen[c] != role:
                raise MethodIncompatibility(
                    f"{c!r} is used both as {seen[c]} and as {role}.",
                    recovery_hint="Each variable can play only one role.",
                )
            seen[c] = role
    extra = [c for c in (absorb, cluster) if c is not None]
    needed = list(dict.fromkeys([y, treat, mediator, instrument] + covs + extra))
    missing = [c for c in needed if c not in data.columns]
    if missing:
        raise MethodIncompatibility(
            f"columns not in data: {missing}.",
            recovery_hint="Check the column names.",
        )

    df = data[needed]
    keep = df.notna().all(axis=1).to_numpy()
    n_dropped = int((~keep).sum())
    if n_dropped:
        warnings.warn(
            f"iv_mediate: {n_dropped} row(s) with a missing value dropped "
            "(listwise, as ivmediate does).",
            RuntimeWarning,
            stacklevel=4,
        )
        df = df.loc[keep]
    n = len(df)
    if n < 5:
        raise DataInsufficient(
            f"iv_mediate: {n} complete observation(s).",
            recovery_hint="Need more complete rows.",
        )
    num_cols = [y, treat, mediator, instrument] + covs
    try:
        V = df[num_cols].to_numpy(dtype=float)
    except (TypeError, ValueError) as exc:
        raise MethodIncompatibility(
            f"iv_mediate: non-numeric values in {num_cols}: {exc}",
            recovery_hint="Encode the variables numerically.",
        ) from exc
    fe = pd.factorize(df[absorb], sort=True)[0] if absorb is not None else None
    codes: Optional[np.ndarray] = None
    n_clusters: Optional[int] = None
    if cluster is not None:
        codes = pd.factorize(df[cluster], sort=True)[0].astype(np.intp)
        n_clusters = int(codes.max()) + 1
        if n_clusters < 2:
            raise DataInsufficient(
                "cluster must identify at least two clusters.",
                recovery_hint="Use a cluster variable with several levels.",
            )
    elif robust:
        codes = np.arange(n)
        n_clusters = n
    vce = "cluster" if cluster is not None else ("robust" if robust else "iid")

    R, k_partial = _partial_out(V[:, :4], V[:, 4:], fe)
    yr, tr, mr, zr = R.T
    raw = V[:, :4] - V[:, :4].mean(axis=0)
    for name, col, r in zip((treat, mediator, instrument), (1, 2, 3), (tr, mr, zr)):
        if float(r @ r) <= 1e-20 * max(float(raw[:, col] @ raw[:, col]), 1e-300):
            raise IdentificationFailure(
                f"iv_mediate: {name!r} has no variation left after partialling "
                "out the covariates and fixed effects.",
                recovery_hint="Drop the covariates or fixed effect it is "
                "collinear with.",
            )
    # The mediator must move independently of the treatment, otherwise the
    # third equation is not identified (M is spanned by T and the controls).
    m_on_t = mr - tr * (float(tr @ mr) / float(tr @ tr))
    if float(m_on_t @ m_on_t) <= 1e-12 * float(mr @ mr):
        raise IdentificationFailure(
            f"iv_mediate: the mediator {mediator!r} is a linear function of "
            f"the treatment {treat!r} and the controls, so its effect "
            "holding the treatment fixed is not identified.",
            recovery_hint="Use a mediator that varies given the treatment.",
            diagnostics={"residual_share": float(m_on_t @ m_on_t) / float(mr @ mr)},
        )

    z1 = zr[:, None]
    b1, V1 = _iv(yr, tr[:, None], z1, codes)
    b2, V2 = _iv(mr, tr[:, None], z1, codes)
    X3 = np.column_stack([tr, mr])
    b3, V3 = _iv(yr, X3, np.column_stack([zr, tr]), codes)
    total, direct = float(b1[0]), float(b3[0])
    a, d = float(b2[0]), float(b3[1])
    indirect = a * d
    # Equations independent: cov(b2, b3) = 0 and eq. 1 is uncorrelated with
    # the others; delta method on a * d.
    v_ind = d**2 * float(V2[0, 0]) + a**2 * float(V3[1, 1])
    c_di = a * float(V3[0, 1])
    vc = np.array(
        [
            [float(V1[0, 0]), 0.0, 0.0],
            [0.0, float(V3[0, 0]), c_di],
            [0.0, c_di, v_ind],
        ]
    )

    # First-stage F statistics (ivmediate's ranktest-based formulas).
    L = 1
    w1 = _wald_first_stage(tr, zr, codes)

    def _off_t(v: np.ndarray) -> np.ndarray:
        out: np.ndarray = v - tr * (float(tr @ v) / float(tr @ tr))
        return out

    w2 = _wald_first_stage(_off_t(mr), _off_t(zr), codes)
    if codes is None:
        f1 = w1 / n * (n - k_partial - L) / L
        f2 = w2 / n * (n - k_partial - L - 1) / L
    else:
        G = int(n_clusters or n)
        f1 = w1 / (n - 1) * (n - k_partial - L) * (G - 1) / G / L
        f2 = w2 / (n - 1) * (n - k_partial - L - 1) * (G - 1) / G / L

    est = np.array([total, direct, indirect])
    se = np.sqrt(np.diag(vc))
    zstat = est / se
    crit = float(stats.norm.ppf(0.5 + level / 2.0))
    table = pd.DataFrame(
        {
            "coef": est,
            "se": se,
            "z": zstat,
            "pvalue": 2.0 * stats.norm.sf(np.abs(zstat)),
            "ci_lower": est - crit * se,
            "ci_upper": est + crit * se,
        },
        index=list(_EFFECTS),
    )
    weak = {k: f for k, f in (("treatment", f1), ("mediator", f2)) if f < _WEAK_F}
    if weak:
        warnings.warn(
            "iv_mediate: weak instrument, first-stage F "
            + ", ".join(f"{k} {f:.2f}" for k, f in weak.items())
            + f" (below {_WEAK_F:g}); the effects and their normal-theory "
            "intervals are unreliable.",
            AssumptionWarning,
            stacklevel=4,
        )
    stages = {
        "y_on_t": {"coef": {treat: total}, "vcov": V1.tolist()},
        "m_on_t": {"coef": {treat: a}, "vcov": V2.tolist()},
        "y_on_t_m": {"coef": {treat: direct, mediator: d}, "vcov": V3.tolist()},
    }
    return IVMediateResult(
        estimates=table,
        vcov=pd.DataFrame(vc, index=list(_EFFECTS), columns=list(_EFFECTS)),
        share_mediated=indirect / total,
        share_direct=direct / total,
        fstat_treatment=float(f1),
        fstat_mediator=float(f2),
        n_obs=n,
        vce=vce,
        y=y,
        treat=treat,
        mediator=mediator,
        instrument=instrument,
        level=level,
        n_clusters=n_clusters if cluster is not None else None,
        stages=stages,
        model_info={
            "covariates": covs,
            "absorb": absorb,
            "n_partialled": k_partial,
        },
        diagnostics={"n_dropped": n_dropped, "weak_instrument": sorted(weak)},
    )
