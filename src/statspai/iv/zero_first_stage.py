"""Zero-first-stage (ZFS) test for the exclusion restriction.

An instrument's exclusion restriction is not testable in the estimation
sample — that is the whole difficulty. It *is* testable in a subsample
where the instrument has no first stage. If ``Z`` moves nothing about the
treatment there, any remaining ``Z``-``Y`` association in that subsample
has to run through a channel other than the treatment, which is exactly
what the exclusion restriction rules out.

van Kippersluis and Rietveld (2018) formalise this for Mendelian
randomization and go one step further: the direct effect estimated in the
zero-first-stage subsample can be *subtracted* from the reduced form in
the main sample, giving a pleiotropy-robust point estimate

.. math:: \\beta_{ZFS} = \\frac{\\rho_{main} - \\gamma_{zfs}}{\\pi_{main}}.

The same logic transfers wholesale to shift-share and interaction
instruments in applied micro: pick the subsample where the exposure leg
is inert (deserts for a solar-resource instrument, industries with no
national shock for a Bartik one) and read the reduced form there.

Three conditions carry the argument, and this function reports evidence
on the first two rather than assuming them:

1. the first stage really is zero in the ZFS subsample — reported with a
   confidence interval, because "insignificant" is not "zero";
2. selection into the subsample is not itself a joint consequence of
   ``Z`` and ``Y``;
3. the direct effect is homogeneous across the two subsamples — an
   assumption, not a testable claim.

References
----------
van Kippersluis, H. and Rietveld, C. A. (2018). "Pleiotropy-robust
Mendelian randomization." *International Journal of Epidemiology*, 47(4),
1279-1288. doi:10.1093/ije/dyx002 [@vankippersluis2018pleiotropy]
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from .._result_serialize import ResultProtocolMixin
from ..exceptions import DataInsufficient, MethodIncompatibility

__all__ = ["zero_first_stage", "ZeroFirstStageResult"]


@dataclass
class ZeroFirstStageResult(ResultProtocolMixin):
    """Output of :func:`zero_first_stage`.

    Attributes
    ----------
    first_stage_zfs, first_stage_zfs_se, first_stage_zfs_ci : float, float, tuple
        First stage in the zero-first-stage subsample — the premise. It
        should be indistinguishable from zero *and* small relative to
        ``first_stage_main``; the interval is what makes that judgeable.
    reduced_form_zfs, reduced_form_zfs_se, reduced_form_zfs_ci : float, float, tuple
        Reduced form in that subsample. Under the premise this *is* the
        instrument's direct effect on the outcome, i.e. the exclusion
        violation.
    first_stage_main, first_stage_main_se : float
        First stage in the estimation sample.
    beta_iv, beta_iv_se : float
        Naive IV estimate on the estimation sample (ratio of reduced form
        to first stage).
    beta_zfs_corrected, beta_zfs_corrected_se, beta_zfs_corrected_ci : float
        van Kippersluis-Rietveld corrected estimate
        ``(rho_main - gamma_zfs) / pi_main`` with bootstrap SE and
        percentile interval.
    implied_bias : float
        ``gamma / pi_main`` — how far the direct effect moves the naive
        estimate.
    n_main, n_zfs, n_boot : int
        Sample sizes and the number of bootstrap replications that
        converged.
    diagnostics : dict
        ``first_stage_not_zero`` flags a failed premise;
        ``zfs_first_stage_share_of_main`` is ``|pi_zfs| / |pi_main|``.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> n = 600
    >>> desert = rng.integers(0, 2, size=n).astype(bool)
    >>> z, u = rng.normal(size=n), rng.normal(size=n)
    >>> d = np.where(desert, 0.0, 0.9 * z) + 0.6 * u + rng.normal(size=n)
    >>> df = pd.DataFrame({"y": -0.5 * d + 0.7 * u + rng.normal(size=n),
    ...                    "d": d, "z": z, "desert": desert})
    >>> out = sp.zero_first_stage(df, y="y", endog="d", instrument="z",
    ...                           zfs="desert", n_boot=0)
    >>> isinstance(out, sp.ZeroFirstStageResult)
    True
    >>> bool(out.n_main > 0 and out.n_zfs > 0)
    True
    """

    first_stage_zfs: float
    first_stage_zfs_se: float
    first_stage_zfs_ci: tuple
    first_stage_zfs_pvalue: float
    reduced_form_zfs: float
    reduced_form_zfs_se: float
    reduced_form_zfs_ci: tuple
    reduced_form_zfs_pvalue: float
    first_stage_main: float
    first_stage_main_se: float
    beta_iv: float
    beta_iv_se: float
    beta_zfs_corrected: float
    beta_zfs_corrected_se: float
    beta_zfs_corrected_ci: tuple
    implied_bias: float
    n_main: int
    n_zfs: int
    n_boot: int
    instrument: str
    alpha: float
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def summary(self) -> str:
        lvl = int(round((1 - self.alpha) * 100))
        lines = [
            "Zero-first-stage (ZFS) exclusion test",
            "=" * 66,
            f"  instrument                 : {self.instrument}",
            f"  N (main / ZFS subsample)   : {self.n_main:,} / {self.n_zfs:,}",
            "-" * 66,
            "Premise -- the first stage must be zero where we look:",
            f"  first stage in ZFS sample  : {self.first_stage_zfs: .5f}"
            f"  (SE {self.first_stage_zfs_se:.5f})",
            f"  {lvl}% CI                    : "
            f"[{self.first_stage_zfs_ci[0]: .5f},"
            f" {self.first_stage_zfs_ci[1]: .5f}]",
            f"  first stage in main sample : {self.first_stage_main: .5f}"
            f"  (SE {self.first_stage_main_se:.5f})",
            "-" * 66,
            "Test -- any reduced form left in the ZFS sample violates exclusion:",
            f"  direct effect (gamma)      : {self.reduced_form_zfs: .5f}"
            f"  (SE {self.reduced_form_zfs_se:.5f},"
            f" p = {self.reduced_form_zfs_pvalue:.4f})",
            f"  {lvl}% CI                    : "
            f"[{self.reduced_form_zfs_ci[0]: .5f},"
            f" {self.reduced_form_zfs_ci[1]: .5f}]",
            "-" * 66,
            "Consequence:",
            f"  IV estimate (main sample)  : {self.beta_iv: .5f}"
            f"  (SE {self.beta_iv_se:.5f})",
            f"  implied bias (gamma/pi)    : {self.implied_bias: .5f}",
            f"  ZFS-corrected estimate     : {self.beta_zfs_corrected: .5f}"
            f"  (SE {self.beta_zfs_corrected_se:.5f})",
            f"  {lvl}% CI (bootstrap)        : "
            f"[{self.beta_zfs_corrected_ci[0]: .5f},"
            f" {self.beta_zfs_corrected_ci[1]: .5f}]",
            "=" * 66,
            self.verdict(),
        ]
        return "\n".join(lines)

    def verdict(self) -> str:
        """One-paragraph reading of the test."""
        if self.diagnostics.get("first_stage_not_zero"):
            return (
                "PREMISE FAILS: the instrument still has a first stage in the "
                "subsample chosen as zero-first-stage, so a reduced form there "
                "is not evidence of a direct effect. Redefine the subsample."
            )
        if self.reduced_form_zfs_pvalue < self.alpha:
            return (
                "EXCLUSION VIOLATED: the instrument moves the outcome in a "
                "subsample where it does not move the treatment. The IV "
                f"estimate is biased by about {self.implied_bias:+.4f}; the "
                "corrected estimate above nets it out under the assumption "
                "that the direct effect is the same in both subsamples."
            )
        return (
            "No detectable direct effect: the reduced form in the "
            "zero-first-stage subsample is indistinguishable from zero. This "
            "is a failure to reject, not a proof of exclusion -- read the "
            "confidence interval as the range of direct effects the data "
            "cannot rule out."
        )


def _as_list(spec: Optional[Union[str, Sequence[str]]]) -> List[str]:
    if spec is None:
        return []
    if isinstance(spec, str):
        return [t.strip() for t in spec.split("+") if t.strip()]
    return [str(t) for t in spec]


def _fit_ols(
    data: pd.DataFrame,
    lhs: str,
    rhs: List[str],
    absorb: List[str],
    cluster: List[str],
) -> Any:
    """OLS (optionally absorbing FE) returning a StatsPAI result object."""
    from .. import hdfe_ols, regress

    rhs_str = " + ".join(rhs)
    cl: Optional[Union[str, List[str]]] = cluster if cluster else None
    if absorb:
        formula = f"{lhs} ~ {rhs_str} | {' + '.join(absorb)}"
        return hdfe_ols(formula, data=data, cluster=cl)
    if cl is not None and not isinstance(cl, str):
        cl = cl[0] if len(cl) == 1 else cl
    return regress(f"{lhs} ~ {rhs_str}", data=data, cluster=cl, robust="hc1")


def zero_first_stage(
    data: pd.DataFrame,
    y: str,
    endog: str,
    instrument: str,
    zfs: Union[str, np.ndarray, pd.Series],
    exog: Optional[Union[str, Sequence[str]]] = None,
    absorb: Optional[Union[str, Sequence[str]]] = None,
    cluster: Optional[Union[str, Sequence[str]]] = None,
    alpha: float = 0.05,
    n_boot: int = 999,
    random_state: Optional[int] = None,
) -> ZeroFirstStageResult:
    """
    Zero-first-stage exclusion test with a pleiotropy-robust IV estimate.

    Tests the exclusion restriction in a subsample where the instrument has
    no first stage, and nets the estimated direct effect out of the
    main-sample IV estimate.

    Parameters
    ----------
    data : DataFrame
        Full sample: main and zero-first-stage observations together.
    y, endog, instrument : str
        Outcome, endogenous regressor, and the *single* excluded
        instrument being tested.
    zfs : str or boolean array
        Marks the zero-first-stage subsample — the observations where the
        instrument is believed to be inert for the treatment (a desert
        county for a solar-resource instrument; an industry with no
        national shock for a shift-share one). A column name is read from
        ``data`` and must be boolean or 0/1.
    exog : str or list of str, optional
        Controls entering every component regression.
    absorb : str or list of str, optional
        Fixed effects to absorb in every component regression.
    cluster : str or list of str, optional
        Clustering for the component regressions and for the bootstrap,
        which resamples whole clusters.
    alpha : float, default 0.05
        Significance level for all reported intervals.
    n_boot : int, default 999
        Bootstrap replications for the corrected estimate's standard
        error. The correction divides by an estimated first stage, so its
        sampling distribution is not well described by a naive delta
        method; resampling handles the covariance between the pieces.
        Set ``0`` to skip (the corrected SE is then ``NaN``).
    random_state : int, optional
        Bootstrap seed.

    Returns
    -------
    ZeroFirstStageResult

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> n = 800
    >>> desert = rng.integers(0, 2, size=n).astype(bool)
    >>> z = rng.normal(size=n)
    >>> u = rng.normal(size=n)
    >>> # First stage only outside the "desert" subsample.
    >>> d = np.where(desert, 0.0, 0.9 * z) + 0.6 * u + rng.normal(size=n)
    >>> y_obs = -0.5 * d + 0.7 * u + rng.normal(size=n)
    >>> df = pd.DataFrame({"y": y_obs, "d": d, "z": z, "desert": desert})
    >>> out = sp.zero_first_stage(df, y="y", endog="d", instrument="z",
    ...                           zfs="desert", n_boot=0)
    >>> bool(abs(out.first_stage_zfs) < 0.1)   # premise holds
    True

    Notes
    -----
    A failure to reject is not a clean bill of health: report the
    confidence interval on the direct effect, which is the set of
    violations the data cannot rule out. The corrected estimate additionally
    assumes the direct effect is the same in both subsamples — plausible
    when the subsample is defined by something unrelated to the outcome
    mechanism, much less so when it is defined by the outcome's own
    environment.

    References
    ----------
    [@vankippersluis2018pleiotropy]
    """
    exog_l = _as_list(exog)
    absorb_l = _as_list(absorb)
    cluster_l = _as_list(cluster) if isinstance(cluster, str) else list(cluster or [])

    if isinstance(zfs, str):
        if zfs not in data.columns:
            raise MethodIncompatibility(
                f"zfs column {zfs!r} is not in `data`.",
                diagnostics={"columns": list(data.columns)[:20]},
            )
        mask = data[zfs].to_numpy()
    else:
        mask = np.asarray(zfs)
    mask = mask.astype(bool)
    if mask.size != len(data):
        raise MethodIncompatibility(
            "zfs mask length does not match `data`.",
            diagnostics={"n_mask": int(mask.size), "n_data": int(len(data))},
        )

    needed = [y, endog, instrument] + exog_l + absorb_l + cluster_l
    seen: set = set()
    needed = [c for c in needed if not (c in seen or seen.add(c))]
    work = data.loc[:, needed].copy()
    work["__zfs"] = mask
    work = work.dropna()
    main = work.loc[~work["__zfs"]].reset_index(drop=True)
    sub = work.loc[work["__zfs"]].reset_index(drop=True)
    if len(sub) < 20 or len(main) < 20:
        raise DataInsufficient(
            "Both the main and the zero-first-stage subsample need at least "
            "20 complete observations.",
            recovery_hint="Widen the zfs definition or check for missing values.",
            diagnostics={"n_main": int(len(main)), "n_zfs": int(len(sub))},
        )

    from scipy import stats

    crit = float(stats.norm.ppf(1 - alpha / 2))
    rhs = [instrument] + exog_l

    fs_zfs = _fit_ols(sub, endog, rhs, absorb_l, cluster_l)
    rf_zfs = _fit_ols(sub, y, rhs, absorb_l, cluster_l)
    fs_main = _fit_ols(main, endog, rhs, absorb_l, cluster_l)
    rf_main = _fit_ols(main, y, rhs, absorb_l, cluster_l)

    def _pick(res: Any) -> tuple:
        return float(res.params[instrument]), float(res.std_errors[instrument])

    pi_zfs, pi_zfs_se = _pick(fs_zfs)
    gamma, gamma_se = _pick(rf_zfs)
    pi_main, pi_main_se = _pick(fs_main)
    rho_main, _ = _pick(rf_main)

    beta_iv = rho_main / pi_main if pi_main != 0 else np.nan
    beta_corr = (rho_main - gamma) / pi_main if pi_main != 0 else np.nan
    implied_bias = gamma / pi_main if pi_main != 0 else np.nan

    # Is the premise credible? Compare the ZFS first stage to the main one.
    share = abs(pi_zfs) / abs(pi_main) if pi_main != 0 else np.inf
    zfs_t = pi_zfs / pi_zfs_se if pi_zfs_se > 0 else np.nan
    premise_fails = bool(np.isfinite(zfs_t) and abs(zfs_t) > crit and share > 0.25)

    # ── Bootstrap the corrected estimate ──────────────────────────────
    boot: List[float] = []
    n_failed = 0
    if n_boot > 0:
        rng = np.random.default_rng(random_state)
        for _ in range(int(n_boot)):
            try:
                b_main = _resample(main, cluster_l, rng)
                b_sub = _resample(sub, cluster_l, rng)
                g_b, _ = _pick(_fit_ols(b_sub, y, rhs, absorb_l, cluster_l))
                p_b, _ = _pick(_fit_ols(b_main, endog, rhs, absorb_l, cluster_l))
                r_b, _ = _pick(_fit_ols(b_main, y, rhs, absorb_l, cluster_l))
                if p_b != 0:
                    boot.append((r_b - g_b) / p_b)
                else:
                    n_failed += 1
            except (
                np.linalg.LinAlgError,
                ValueError,
                KeyError,
                ZeroDivisionError,
            ):
                # A resample can be collinear or drop a whole cluster. Count
                # it rather than swallowing it: replications that fail are
                # not missing at random, so a bootstrap SE computed over the
                # survivors is only trustworthy while failures stay rare.
                n_failed += 1

        if n_failed:
            share = n_failed / float(n_boot)
            if share > 0.1:
                warnings.warn(
                    f"zero_first_stage: {n_failed}/{n_boot} bootstrap "
                    f"replications failed ({share:.0%}). The corrected "
                    "standard error is computed over the survivors and is "
                    "likely optimistic — check for near-collinear controls "
                    "or too few clusters.",
                    RuntimeWarning,
                    stacklevel=2,
                )

    if boot:
        boot_arr = np.asarray(boot, dtype=float)
        corr_se = float(np.std(boot_arr, ddof=1))
        corr_ci = (
            float(np.quantile(boot_arr, alpha / 2)),
            float(np.quantile(boot_arr, 1 - alpha / 2)),
        )
    else:
        corr_se = np.nan
        corr_ci = (np.nan, np.nan)

    beta_iv_se = abs(beta_iv) * np.sqrt(
        (pi_main_se / pi_main) ** 2 if pi_main != 0 else np.nan
    )

    return ZeroFirstStageResult(
        first_stage_zfs=pi_zfs,
        first_stage_zfs_se=pi_zfs_se,
        first_stage_zfs_ci=(pi_zfs - crit * pi_zfs_se, pi_zfs + crit * pi_zfs_se),
        first_stage_zfs_pvalue=float(2 * (1 - stats.norm.cdf(abs(zfs_t)))),
        reduced_form_zfs=gamma,
        reduced_form_zfs_se=gamma_se,
        reduced_form_zfs_ci=(gamma - crit * gamma_se, gamma + crit * gamma_se),
        reduced_form_zfs_pvalue=float(
            2 * (1 - stats.norm.cdf(abs(gamma / gamma_se))) if gamma_se > 0 else np.nan
        ),
        first_stage_main=pi_main,
        first_stage_main_se=pi_main_se,
        beta_iv=beta_iv,
        beta_iv_se=beta_iv_se,
        beta_zfs_corrected=beta_corr,
        beta_zfs_corrected_se=corr_se,
        beta_zfs_corrected_ci=corr_ci,
        implied_bias=implied_bias,
        n_main=int(len(main)),
        n_zfs=int(len(sub)),
        n_boot=len(boot),
        instrument=instrument,
        alpha=alpha,
        diagnostics={
            "first_stage_not_zero": premise_fails,
            "n_bootstrap_failed": int(n_failed),
            "zfs_first_stage_share_of_main": float(share),
            "reduced_form_main": float(rho_main),
        },
    )


def _resample(
    frame: pd.DataFrame, cluster_l: List[str], rng: np.random.Generator
) -> pd.DataFrame:
    """Bootstrap resample — whole clusters when clustering is requested."""
    if not cluster_l:
        idx = rng.integers(0, len(frame), size=len(frame))
        return frame.iloc[idx].reset_index(drop=True)
    keys = frame[cluster_l[0]]
    groups = keys.unique()
    drawn = rng.choice(groups, size=len(groups), replace=True)
    parts = [frame.loc[keys == g] for g in drawn]
    return pd.concat(parts, ignore_index=True)
