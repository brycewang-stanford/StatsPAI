"""
Thin wrappers around pyfixest estimation functions.

Each wrapper delegates to pyfixest and converts results into
StatsPAI's ``EconometricResults``, making them compatible with
``outreg2`` and the rest of the StatsPAI ecosystem.
"""

from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd

from .._aliases import accepts_aliases
from ..core.results import EconometricResults
from ..exceptions import MethodIncompatibility
from .adapter import _multi_fit_to_results, _pyfixest_to_econometric_results

#: vcov sentinels (case-insensitive) that select the wild cluster bootstrap.
_WILD_VCOV = frozenset({"wild", "wildbootstrap", "wild_cluster", "wcr", "boottest"})


def _check_pyfixest() -> Any:
    """Import pyfixest or raise a clear error."""
    try:
        import pyfixest as pf

        return pf
    except ImportError:
        raise ImportError(
            "pyfixest is required for high-dimensional fixed effects estimation.\n"
            "Install it with: pip install pyfixest\n"
            "Or install StatsPAI with the fixest extra: pip install statspai[fixest]"
        )


# --------------------------------------------------------------------------- #
#  feols — OLS / IV with high-dimensional fixed effects
# --------------------------------------------------------------------------- #


def _feols_wild(
    fml: str,
    data: pd.DataFrame,
    cluster: str,
    *,
    reps: int,
    weight_type: str,
    seed: Optional[int],
    weights: Optional[str],
    ssc: Optional[Any],
    fixef_rm: str,
    collin_tol: float,
    lean: bool,
    extra: Dict[str, Any],
) -> EconometricResults:
    """Native ``feols(..., vce="wild")`` — Stata ``boottest``-style inference.

    Fits the model once with cluster-robust (CRV1) SEs — which also stores the
    within-transformed design on the result (see ``fixest/adapter.py``) — then
    runs the WCR wild cluster bootstrap (Cameron-Gelbach-Miller 2008) on that
    partialled-out design for every non-absorbed coefficient.  Point estimates
    are the feols estimates; p-values and confidence intervals come from the
    bootstrap.  This is the verified within wild bootstrap (byte-identical to
    ``sp.regress`` on FE-demeaned data), promoted from the standalone
    ``sp.wild_cluster_boot`` to a first-class ``vce=`` option.
    """
    base = feols(
        fml,
        data,
        vcov={"CRV1": cluster},
        weights=weights,
        ssc=ssc,
        fixef_rm=fixef_rm,
        collin_tol=collin_tol,
        lean=lean,
        **extra,
    )
    if isinstance(base, list):
        raise MethodIncompatibility(
            "feols(vce='wild') does not support multiple-estimation formulas "
            "(csw/sw/csw0). Fit each model separately."
        )

    from ..inference.jackknife import wild_cluster_boot

    se = base.std_errors.copy()
    pvals = pd.Series(np.nan, index=base.params.index, dtype=float)
    ci_lo = pd.Series(np.nan, index=base.params.index, dtype=float)
    ci_hi = pd.Series(np.nan, index=base.params.index, dtype=float)
    for var in base.params.index:
        out = wild_cluster_boot(
            base,
            data,
            cluster=cluster,
            variable=str(var),
            n_boot=reps,
            weight_type=weight_type,
            seed=seed,
        )
        pvals[var] = out["p_boot"]
        ci_lo[var], ci_hi[var] = out["ci_boot"]
        se[var] = out["se_cluster"]

    base.std_errors = se
    base.pvalues = pvals
    base.conf_int_lower = ci_lo
    base.conf_int_upper = ci_hi
    base.model_info = dict(base.model_info)
    base.model_info["vcov_type"] = (
        f"wild cluster bootstrap (Cameron-Gelbach-Miller 2008, {reps} reps, "
        f"{weight_type})"
    )
    base.model_info["n_boot"] = reps
    base.model_info["cluster"] = cluster
    return base


def _feols_bias_reduced(
    fml: str,
    data: pd.DataFrame,
    kind: str,
    *,
    cluster: Optional[str],
    conley_lat: Optional[str],
    conley_lon: Optional[str],
    conley_cutoff: Optional[float],
    weights: Optional[str],
    ssc: Optional[Any],
    fixef_rm: str,
    collin_tol: float,
    lean: bool,
    extra: Dict[str, Any],
) -> EconometricResults:
    """CR2/CR3/jackknife (Pustejovsky-Tipton) or Conley spatial HAC on feols.

    Fits the FE model once (populating the stored within design), then applies
    the requested bias-reduced / spatial estimator to the partialled-out design.
    CR2/CR3 match R clubSandwich (plm); Conley matches Stata acreg on the
    FE-demeaned data.
    """
    from scipy import stats as _stats

    from ..inference.jackknife import conley_vcov_ols, cr_vcov_ols

    if kind == "conley":
        if conley_lat is None or conley_lon is None or conley_cutoff is None:
            raise MethodIncompatibility(
                "feols(vce='conley') requires conley_lat=, conley_lon=, and "
                "conley_cutoff= (planar distance cutoff in km)."
            )
        base = feols(
            fml,
            data,
            weights=weights,
            ssc=ssc,
            fixef_rm=fixef_rm,
            collin_tol=collin_tol,
            lean=lean,
            **extra,
        )
    else:
        if cluster is None:
            raise MethodIncompatibility(
                f"feols(vce={kind!r}) requires cluster=... (a cluster-robust "
                "small-sample correction)."
            )
        base = feols(
            fml,
            data,
            vcov={"CRV1": cluster},
            weights=weights,
            ssc=ssc,
            fixef_rm=fixef_rm,
            collin_tol=collin_tol,
            lean=lean,
            **extra,
        )
    if isinstance(base, list):
        raise MethodIncompatibility(
            f"feols(vce={kind!r}) does not support multiple-estimation formulas."
        )

    if kind == "conley":
        se = conley_vcov_ols(base, data, conley_lat, conley_lon, conley_cutoff)
        label = f"Conley spatial HAC (acreg planar, {conley_cutoff} km)"
    else:
        power = 0.5 if kind == "cr2" else 1.0
        cl_codes = pd.Categorical(data[cluster]).codes
        # small_sample=False matches clubSandwich's CR2/CR3 for FE (plm) exactly.
        se = cr_vcov_ols(base, cl_codes, power=power, small_sample=False)
        label = {
            "cr2": "CR2 cluster-robust (clubSandwich, Pustejovsky-Tipton 2018)",
            "cr3": "CR3 cluster-robust (clubSandwich jackknife-type)",
            "jackknife": "CR3 cluster-robust (clubSandwich jackknife-type)",
        }[kind]

    z = base.params / se
    base.std_errors = se
    base.pvalues = pd.Series(
        2 * (1 - _stats.norm.cdf(np.abs(z))), index=base.params.index
    )
    crit = _stats.norm.ppf(0.975)
    base.conf_int_lower = base.params - crit * se
    base.conf_int_upper = base.params + crit * se
    base.model_info = dict(base.model_info)
    base.model_info["vcov_type"] = label
    if cluster is not None:
        base.model_info["cluster"] = cluster
    return base


#: vcov sentinels selecting the GLM bias-reduced cluster menu (CR2/CR3/jackknife).
_GLM_BR_VCOV = frozenset({"cr2", "cr3", "jackknife"})


def _sm_glm_family(family_str: Optional[str]) -> Any:
    """Map a pyfixest family string to a statsmodels GLM family instance."""
    from statsmodels.api import families as _fam

    f = (family_str or "poisson").lower()
    if f == "poisson":
        return _fam.Poisson()
    if f in ("logit", "binomial"):
        return _fam.Binomial()
    if f == "probit":
        return _fam.Binomial(link=_fam.links.Probit())
    if f == "gaussian":
        return _fam.Gaussian()
    raise MethodIncompatibility(
        f"vce='CR2'/'CR3' not supported for family={family_str!r}; "
        "supported: poisson, logit, probit, gaussian.",
        recovery_hint="Use cluster= (CRV1) or vce='wild' instead.",
    )


def _feglm_bias_reduced(
    fml: str,
    data: pd.DataFrame,
    kind: str,
    *,
    family_str: Optional[str],
    cluster: Optional[str],
    is_pois: bool,
    ssc: Optional[Any],
    fixef_rm: str,
    collin_tol: float,
    extra: Dict[str, Any],
) -> EconometricResults:
    """CR2/CR3 (Pustejovsky-Tipton) cluster-robust SEs for fepois / feglm.

    Reproduces R ``clubSandwich::vcovCR(glm, type="CR2"/"CR3")`` to machine
    precision by fitting the GLM with the fixed effects as **dummies** and
    applying the IRLS-weighted bias-reduced adjustment (see
    :func:`statspai.inference.jackknife.glm_cr_vcov`). Unlike OLS, the
    weighted-projection FWL equivalence does *not* carry the CR2 leverage
    through fixed-effect absorption, so the dummy design is required to match
    the reference — feasible for modest FE dimensionality only.
    """
    from scipy import stats as _stats

    from ..inference.jackknife import glm_cr_vcov

    who = "fepois" if is_pois else "feglm"
    if cluster is None:
        raise MethodIncompatibility(
            f"{who}(vce={kind!r}) requires cluster=... (a cluster-robust "
            "small-sample correction).",
            recovery_hint="Pass cluster='firm' (or another id column).",
        )

    # Canonical estimate + full result object from the native estimator.
    if is_pois:
        base = fepois(
            fml,
            data,
            vcov={"CRV1": cluster},
            ssc=ssc,
            fixef_rm=fixef_rm,
            collin_tol=collin_tol,
            **extra,
        )
    else:
        base = feglm(fml, data, family=family_str, vcov={"CRV1": cluster}, **extra)
    if isinstance(base, list):
        raise MethodIncompatibility(
            f"{who}(vce={kind!r}) does not support multiple-estimation formulas."
        )

    # Parse the formula into outcome / regressors / fixed effects.
    lhs, rhs = fml.split("~", 1)
    y_var = lhs.strip()
    fe_vars: List[str] = []
    if "|" in rhs:
        cov_part, fe_part = rhs.split("|", 1)
        fe_vars = [t.strip() for t in fe_part.replace("+", " ").split() if t.strip()]
    else:
        cov_part = rhs
    cov_vars = [
        t.strip() for t in cov_part.split("+") if t.strip() and t.strip() != "1"
    ]

    need = [y_var] + cov_vars + fe_vars + [cluster]
    missing = [c for c in need if c not in data.columns]
    if missing:
        raise MethodIncompatibility(
            f"{who}(vce={kind!r}): columns {missing} not found in data."
        )
    work = data.loc[:, need].dropna().reset_index(drop=True)

    base_n = int(base.data_info.get("nobs", len(work)))
    if base_n != len(work):
        raise MethodIncompatibility(
            f"{who}(vce={kind!r}): the fitted sample ({base_n} obs) differs from "
            f"the complete-case design ({len(work)} obs) — singleton / separation "
            "dropping breaks the CR2/CR3 dummy-design alignment.",
            recovery_hint="Use cluster= (CRV1) or vce='wild' for this model.",
        )

    # Align result coefficients to design columns by name (simple regressors only).
    try:
        cov_pos = [cov_vars.index(nm) for nm in base.params.index]
    except ValueError:
        raise MethodIncompatibility(
            f"{who}(vce={kind!r}) supports simple additive regressors only; "
            "factor() / interaction terms are not yet handled.",
            recovery_hint="Use cluster= (CRV1) or vce='wild' for this model.",
        )

    y = work[y_var].to_numpy(dtype=float)
    Xc = work[cov_vars].to_numpy(dtype=float)
    blocks: List[np.ndarray] = [Xc, np.ones((len(work), 1))]
    for fe in fe_vars:
        d = pd.get_dummies(work[fe], prefix=fe, drop_first=True).astype(float).values
        if d.shape[1]:
            blocks.append(d)
    X = np.column_stack(blocks)

    # Full-dummy CR2/CR3 costs O(p^3) in the bread (p = regressors + FE dummies)
    # plus O(n_g^3) per cluster; guard against high-dimensional FE.
    if X.shape[1] >= len(work) or X.shape[1] > 1000:
        raise MethodIncompatibility(
            f"{who}(vce={kind!r}): the fixed-effects dummy design has "
            f"{X.shape[1]} columns — too many for the full-dummy CR2/CR3 "
            "(which cannot absorb the FE without losing the leverage the "
            "reference needs).",
            recovery_hint="Use cluster= (CRV1) or vce='wild' for "
            "high-dimensional fixed effects.",
        )

    fam = _sm_glm_family("poisson" if is_pois else family_str)
    codes = pd.factorize(work[cluster])[0]
    power = 0.5 if kind == "cr2" else 1.0
    se_full = glm_cr_vcov(X, y, fam, codes, power=power)
    se = pd.Series(
        [float(se_full[cov_pos[i]]) for i in range(len(cov_pos))],
        index=list(base.params.index),
    )

    z = base.params / se
    base.std_errors = se
    base.pvalues = pd.Series(
        2 * (1 - _stats.norm.cdf(np.abs(z))), index=base.params.index
    )
    crit = _stats.norm.ppf(0.975)
    base.conf_int_lower = base.params - crit * se
    base.conf_int_upper = base.params + crit * se
    base.model_info = dict(base.model_info)
    base.model_info["vcov_type"] = {
        "cr2": "CR2 cluster-robust (clubSandwich glm, Pustejovsky-Tipton 2018)",
        "cr3": "CR3 cluster-robust (clubSandwich glm jackknife-type)",
        "jackknife": "CR3 cluster-robust (clubSandwich glm jackknife-type)",
    }[kind]
    base.model_info["cluster"] = cluster
    return base


def _feglm_wild(
    fml: str,
    data: pd.DataFrame,
    *,
    family_str: Optional[str],
    cluster: Optional[str],
    is_pois: bool,
    n_boot: int,
    weight_type: str,
    seed: Optional[int],
    ssc: Optional[Any],
    fixef_rm: str,
    collin_tol: float,
    extra: Dict[str, Any],
) -> EconometricResults:
    """Score wild cluster bootstrap p-values for fepois / feglm.

    Runs the restricted (null-imposed) score wild cluster bootstrap of
    Kline-Santos (2012) — the method Stata ``boottest`` uses after ``poisson`` /
    ``logit`` — on the fixed-effects-as-dummies design (see
    :func:`statspai.inference.jackknife.glm_score_wild_boot`). It is *consistent
    with* ``boottest`` (agrees on the bootstrap p-value to ~2 decimals) but not
    bit-identical: ``boottest`` applies a specific full-model-bread /
    restricted-score studentization this canonical version does not reproduce
    exactly. Point estimates and SE/CI stay the cluster-robust (CRV1) values;
    only the p-values are replaced by the wild-bootstrap p-values.
    """
    from ..inference.jackknife import glm_score_wild_boot

    who = "fepois" if is_pois else "feglm"
    if cluster is None:
        raise MethodIncompatibility(
            f"{who}(vce='wild') requires cluster=... — the wild *cluster* "
            "bootstrap resamples cluster scores.",
            recovery_hint="Pass cluster='firm' (or another id column).",
        )

    if is_pois:
        base = fepois(
            fml,
            data,
            vcov={"CRV1": cluster},
            ssc=ssc,
            fixef_rm=fixef_rm,
            collin_tol=collin_tol,
            **extra,
        )
    else:
        base = feglm(fml, data, family=family_str, vcov={"CRV1": cluster}, **extra)
    if isinstance(base, list):
        raise MethodIncompatibility(
            f"{who}(vce='wild') does not support multiple-estimation formulas."
        )

    lhs, rhs = fml.split("~", 1)
    y_var = lhs.strip()
    fe_vars: List[str] = []
    if "|" in rhs:
        cov_part, fe_part = rhs.split("|", 1)
        fe_vars = [t.strip() for t in fe_part.replace("+", " ").split() if t.strip()]
    else:
        cov_part = rhs
    cov_vars = [
        t.strip() for t in cov_part.split("+") if t.strip() and t.strip() != "1"
    ]

    need = [y_var] + cov_vars + fe_vars + [cluster]
    missing = [c for c in need if c not in data.columns]
    if missing:
        raise MethodIncompatibility(
            f"{who}(vce='wild'): columns {missing} not found in data."
        )
    work = data.loc[:, need].dropna().reset_index(drop=True)
    if int(base.data_info.get("nobs", len(work))) != len(work):
        raise MethodIncompatibility(
            f"{who}(vce='wild'): the fitted sample differs from the complete-case "
            "design (singleton / separation dropping) — cannot align the wild "
            "bootstrap design.",
            recovery_hint="Use cluster= (CRV1) for this model.",
        )
    try:
        [cov_vars.index(nm) for nm in base.params.index]
    except ValueError:
        raise MethodIncompatibility(
            f"{who}(vce='wild') supports simple additive regressors only; "
            "factor() / interaction terms are not yet handled.",
            recovery_hint="Use cluster= (CRV1) for this model.",
        )

    y = work[y_var].to_numpy(dtype=float)
    Xc = work[cov_vars].to_numpy(dtype=float)
    blocks: List[np.ndarray] = [Xc, np.ones((len(work), 1))]
    for fe in fe_vars:
        d = pd.get_dummies(work[fe], prefix=fe, drop_first=True).astype(float).values
        if d.shape[1]:
            blocks.append(d)
    X = np.column_stack(blocks)
    if X.shape[1] >= len(work) or X.shape[1] > 1000:
        raise MethodIncompatibility(
            f"{who}(vce='wild'): the fixed-effects dummy design has "
            f"{X.shape[1]} columns — too many for the dummy-design score "
            "bootstrap (which refits the GLM per coefficient).",
            recovery_hint="Use cluster= (CRV1) for high-dimensional fixed effects.",
        )

    fam = _sm_glm_family("poisson" if is_pois else family_str)
    codes = pd.factorize(work[cluster])[0]
    pvals = {}
    for nm in base.params.index:
        out = glm_score_wild_boot(
            X,
            y,
            fam,
            codes,
            test_idx=cov_vars.index(nm),
            n_boot=n_boot,
            weight_type=weight_type,
            seed=seed,
        )
        pvals[nm] = out["p_boot"]

    base.pvalues = pd.Series(
        [pvals[nm] for nm in base.params.index], index=base.params.index
    )
    base.model_info = dict(base.model_info)
    base.model_info["vcov_type"] = (
        f"score wild cluster bootstrap (Kline-Santos 2012, {weight_type}); "
        "point estimates + SE/CI are cluster-robust CRV1"
    )
    base.model_info["cluster"] = cluster
    base.model_info["n_boot"] = n_boot
    return base


@accepts_aliases(vce="vcov")
def feols(
    fml: str,
    data: pd.DataFrame,
    vcov: Optional[Union[str, Dict[str, str]]] = None,
    *,
    weights: Optional[str] = None,
    ssc: Optional[Any] = None,
    fixef_rm: str = "none",
    collin_tol: float = 1e-6,
    lean: bool = False,
    cluster: Optional[str] = None,
    wild_reps: int = 999,
    wild_weight_type: str = "rademacher",
    seed: Optional[int] = None,
    conley_lat: Optional[str] = None,
    conley_lon: Optional[str] = None,
    conley_cutoff: Optional[float] = None,
    **kwargs: Any,
) -> Union[EconometricResults, List[EconometricResults]]:
    """
    Estimate OLS / IV with high-dimensional fixed effects via pyfixest.

    Uses the Frisch-Waugh-Lovell theorem for fast absorption of
    high-dimensional fixed effects.

    Parameters
    ----------
    fml : str
        A pyfixest-style formula. Examples:

        - ``"Y ~ X1 + X2"`` — plain OLS
        - ``"Y ~ X1 | firm + year"`` — two-way fixed effects
        - ``"Y ~ 1 | firm | X1 ~ Z1"`` — IV with fixed effects
        - ``"Y ~ X1 | csw0(firm, year)"`` — multiple estimations

    data : pd.DataFrame
        Input dataset.
    vcov : str or dict, optional
        Variance-covariance estimator.

        - ``"iid"`` — classical
        - ``"HC1"``, ``"HC2"``, ``"HC3"`` — heteroskedasticity-robust
        - ``{"CRV1": "firm"}`` — cluster-robust
        - ``{"CRV1": "firm + year"}`` — two-way clustering
    weights : str, optional
        Column name for regression weights.
    ssc : optional
        Small-sample correction via ``pyfixest.ssc()``.
    fixef_rm : str, default "none"
        How to handle singleton fixed effects:
        ``"none"`` (keep) or ``"singleton"`` (drop).
    collin_tol : float, default 1e-6
        Collinearity tolerance.
    lean : bool, default False
        If True, drop large intermediate arrays to save memory.
    **kwargs
        Additional arguments passed to ``pyfixest.feols()``.

    Returns
    -------
    EconometricResults or list of EconometricResults
        Single result for simple formulas, list for multiple estimations
        (e.g. ``csw0``/``sw``/``sw0`` syntax).

    Examples
    --------
    Two-way fixed effects with clustered SEs:

    >>> import statspai as sp
    >>> import numpy as np
    >>> import pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> n = 400
    >>> firm = rng.integers(0, 8, n)
    >>> year = rng.integers(0, 5, n)
    >>> x1 = rng.normal(size=n)
    >>> y = 1.0 + 0.5 * x1 + 0.1 * firm + 0.2 * year + rng.normal(0, 0.5, n)
    >>> df = pd.DataFrame({"y": y, "x1": x1, "firm": firm, "year": year})
    >>> res = sp.feols("y ~ x1 | firm + year", data=df, vcov={"CRV1": "firm"})  # doctest: +SKIP
    >>> "x1" in res.params.index  # doctest: +SKIP
    True

    Multiple estimation (``csw0`` returns a list, one fit per FE set):

    >>> results = sp.feols("y ~ x1 | csw0(firm, year)", data=df)  # doctest: +SKIP
    >>> summaries = [r.summary() for r in results]  # doctest: +SKIP

    Use with outreg2 to build a regression table:

    >>> r1 = sp.feols("y ~ x1", data=df)  # doctest: +SKIP
    >>> r2 = sp.feols("y ~ x1 | firm", data=df)  # doctest: +SKIP
    >>> sp.outreg2(r1, r2, filename="table.xlsx")  # doctest: +SKIP
    """
    # Wild cluster bootstrap path (Stata ``boottest`` / ``vce()``-style):
    #   sp.feols("y ~ x | firm", data=df, vce="wild", cluster="firm")
    if isinstance(vcov, str) and vcov.lower() in _WILD_VCOV:
        if cluster is None:
            raise MethodIncompatibility(
                "feols(vce='wild') requires cluster=... — the wild *cluster* "
                "bootstrap resamples residuals within clusters."
            )
        return _feols_wild(
            fml,
            data,
            cluster,
            reps=wild_reps,
            weight_type=wild_weight_type,
            seed=seed,
            weights=weights,
            ssc=ssc,
            fixef_rm=fixef_rm,
            collin_tol=collin_tol,
            lean=lean,
            extra=kwargs,
        )

    # Bias-reduced / spatial SEs on the FE-absorbed (within) design. feols stores
    # its within-transformed design on the result, so CR2/CR3 (Bell-McCaffrey /
    # jackknife-type) and Conley spatial HAC operate on the partialled-out model.
    # CR2/CR3 match R clubSandwich (plm) to machine precision; Conley matches
    # Stata acreg run on the FE-demeaned data.
    if isinstance(vcov, str) and vcov.lower() in (
        "cr2",
        "cr3",
        "jackknife",
        "conley",
    ):
        return _feols_bias_reduced(
            fml,
            data,
            vcov.lower(),
            cluster=cluster,
            conley_lat=conley_lat,
            conley_lon=conley_lon,
            conley_cutoff=conley_cutoff,
            weights=weights,
            ssc=ssc,
            fixef_rm=fixef_rm,
            collin_tol=collin_tol,
            lean=lean,
            extra=kwargs,
        )

    # Grammar convenience: `cluster="firm"` is shorthand for one-way CRV1, so the
    # SE keyword reads the same as `sp.regress(..., cluster=...)`. An explicit
    # `vcov=` always wins if both are given.
    if cluster is not None and vcov is None:
        vcov = {"CRV1": cluster}

    pf = _check_pyfixest()

    pf_kwargs: Dict[str, Any] = {
        "fml": fml,
        "data": data,
        "fixef_rm": fixef_rm,
        "collin_tol": collin_tol,
        "lean": lean,
        **kwargs,
    }
    if vcov is not None:
        pf_kwargs["vcov"] = vcov
    if weights is not None:
        pf_kwargs["weights"] = weights
    if ssc is not None:
        pf_kwargs["ssc"] = ssc

    fit = pf.feols(**pf_kwargs)

    # Multiple estimation returns FixestMulti
    if hasattr(fit, "all_fitted_models"):
        return _multi_fit_to_results(fit, vcov=None)

    return _pyfixest_to_econometric_results(fit)


# --------------------------------------------------------------------------- #
#  fepois — Poisson regression with high-dimensional fixed effects
# --------------------------------------------------------------------------- #


@accepts_aliases(vce="vcov")
def fepois(
    fml: str,
    data: pd.DataFrame,
    vcov: Optional[Union[str, Dict[str, str]]] = None,
    *,
    weights: Optional[str] = None,
    ssc: Optional[Any] = None,
    fixef_rm: str = "none",
    collin_tol: float = 1e-6,
    iwls_tol: float = 1e-8,
    iwls_maxiter: int = 25,
    cluster: Optional[str] = None,
    wild_reps: int = 9999,
    wild_weight_type: str = "rademacher",
    seed: Optional[int] = None,
    **kwargs: Any,
) -> Union[EconometricResults, List[EconometricResults]]:
    """
    Estimate Poisson regression with high-dimensional fixed effects via pyfixest.

    Parameters
    ----------
    fml : str
        pyfixest formula. E.g. ``"Y ~ X1 | firm"``.
    data : pd.DataFrame
        Input dataset.
    vcov : str or dict, optional
        Variance-covariance estimator.
    weights : str, optional
        Column name for regression weights.
    ssc : optional
        Small-sample correction.
    fixef_rm : str, default "none"
        Singleton fixed effect handling.
    collin_tol : float, default 1e-6
        Collinearity tolerance.
    iwls_tol : float, default 1e-8
        IWLS convergence tolerance.
    iwls_maxiter : int, default 25
        Max IWLS iterations.
    **kwargs
        Additional arguments passed to ``pyfixest.fepois()``.

    Returns
    -------
    EconometricResults or list of EconometricResults

    Examples
    --------
    Poisson regression (PPML-style) with firm fixed effects:

    >>> import statspai as sp
    >>> import numpy as np
    >>> import pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> n = 400
    >>> firm = rng.integers(0, 8, n)
    >>> x1 = rng.normal(size=n)
    >>> mu = np.exp(0.3 + 0.5 * x1 + 0.1 * firm)
    >>> df = pd.DataFrame({"y": rng.poisson(mu), "x1": x1, "firm": firm})
    >>> res = sp.fepois("y ~ x1 | firm", data=df)  # doctest: +SKIP
    >>> "x1" in res.params.index  # doctest: +SKIP
    True

    Bias-reduced cluster-robust SEs (matches R ``clubSandwich::vcovCR``):

    >>> res = sp.fepois("y ~ x1 | firm", data=df, vce="CR2",  # doctest: +SKIP
    ...                 cluster="firm")
    """
    # Score wild cluster bootstrap p-values (Kline-Santos 2012, boottest-style).
    if isinstance(vcov, str) and vcov.lower() in _WILD_VCOV:
        return _feglm_wild(
            fml,
            data,
            family_str="poisson",
            cluster=cluster,
            is_pois=True,
            n_boot=wild_reps,
            weight_type=wild_weight_type,
            seed=seed,
            ssc=ssc,
            fixef_rm=fixef_rm,
            collin_tol=collin_tol,
            extra=kwargs,
        )

    # Bias-reduced cluster SEs (CR2 / CR3 / jackknife) via the clubSandwich glm
    # adjustment on the FE-as-dummies design — matches R clubSandwich exactly.
    if isinstance(vcov, str) and vcov.lower() in _GLM_BR_VCOV:
        return _feglm_bias_reduced(
            fml,
            data,
            vcov.lower(),
            family_str="poisson",
            cluster=cluster,
            is_pois=True,
            ssc=ssc,
            fixef_rm=fixef_rm,
            collin_tol=collin_tol,
            extra=kwargs,
        )

    # Grammar convenience: cluster="firm" is one-way CRV1 (reads like sp.regress).
    if cluster is not None and vcov is None:
        vcov = {"CRV1": cluster}

    pf = _check_pyfixest()

    pf_kwargs: Dict[str, Any] = {
        "fml": fml,
        "data": data,
        "fixef_rm": fixef_rm,
        "collin_tol": collin_tol,
        "iwls_tol": iwls_tol,
        "iwls_maxiter": iwls_maxiter,
        **kwargs,
    }
    if vcov is not None:
        pf_kwargs["vcov"] = vcov
    if weights is not None:
        pf_kwargs["weights"] = weights
    if ssc is not None:
        pf_kwargs["ssc"] = ssc

    fit = pf.fepois(**pf_kwargs)

    if hasattr(fit, "all_fitted_models"):
        return _multi_fit_to_results(fit, vcov=None)

    return _pyfixest_to_econometric_results(fit)


# --------------------------------------------------------------------------- #
#  feglm — GLM with high-dimensional fixed effects
# --------------------------------------------------------------------------- #


@accepts_aliases(vce="vcov")
def feglm(
    fml: str,
    data: pd.DataFrame,
    family: str = "gaussian",
    vcov: Optional[Union[str, Dict[str, str]]] = None,
    *,
    cluster: Optional[str] = None,
    wild_reps: int = 9999,
    wild_weight_type: str = "rademacher",
    seed: Optional[int] = None,
    **kwargs: Any,
) -> Union[EconometricResults, List[EconometricResults]]:
    """
    Estimate GLM (logit, probit, Gaussian) with high-dimensional fixed effects.

    Parameters
    ----------
    fml : str
        pyfixest formula.
    data : pd.DataFrame
        Input dataset.
    family : str, default "gaussian"
        GLM family: ``"gaussian"``, ``"logit"``, ``"probit"``.
    vcov : str or dict, optional
        Variance-covariance estimator. Also accepts ``vce="CR2"``/``"CR3"``/
        ``"jackknife"`` (with ``cluster=``) for the clubSandwich bias-reduced
        cluster-robust SEs.
    cluster : str, optional
        Cluster id column for ``vce="CR2"/"CR3"/"jackknife"`` (also a shorthand
        for one-way ``{"CRV1": cluster}``).
    **kwargs
        Additional arguments passed to ``pyfixest.feglm()``.

    Returns
    -------
    EconometricResults or list of EconometricResults

    Examples
    --------
    Logit GLM with firm fixed effects:

    >>> import statspai as sp
    >>> import numpy as np
    >>> import pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> n = 500
    >>> firm = rng.integers(0, 8, n)
    >>> x1 = rng.normal(size=n)
    >>> p = 1.0 / (1.0 + np.exp(-(0.2 + 0.8 * x1)))
    >>> y = (rng.random(n) < p).astype(int)
    >>> df = pd.DataFrame({"y": y, "x1": x1, "firm": firm})
    >>> res = sp.feglm("y ~ x1 | firm", data=df, family="logit")  # doctest: +SKIP
    >>> "x1" in res.params.index  # doctest: +SKIP
    True
    """
    # Score wild cluster bootstrap p-values (Kline-Santos 2012, boottest-style).
    if isinstance(vcov, str) and vcov.lower() in _WILD_VCOV:
        return _feglm_wild(
            fml,
            data,
            family_str=family,
            cluster=cluster,
            is_pois=False,
            n_boot=wild_reps,
            weight_type=wild_weight_type,
            seed=seed,
            ssc=None,
            fixef_rm="none",
            collin_tol=1e-6,
            extra=kwargs,
        )

    # Bias-reduced cluster SEs (CR2 / CR3 / jackknife) via the clubSandwich glm
    # adjustment on the FE-as-dummies design — matches R clubSandwich exactly.
    if isinstance(vcov, str) and vcov.lower() in _GLM_BR_VCOV:
        return _feglm_bias_reduced(
            fml,
            data,
            vcov.lower(),
            family_str=family,
            cluster=cluster,
            is_pois=False,
            ssc=None,
            fixef_rm="none",
            collin_tol=1e-6,
            extra=kwargs,
        )

    if cluster is not None and vcov is None:
        vcov = {"CRV1": cluster}

    pf = _check_pyfixest()

    pf_kwargs: Dict[str, Any] = {"fml": fml, "data": data, "family": family, **kwargs}
    if vcov is not None:
        pf_kwargs["vcov"] = vcov

    fit = pf.feglm(**pf_kwargs)

    if hasattr(fit, "all_fitted_models"):
        return _multi_fit_to_results(fit, vcov=None)

    return _pyfixest_to_econometric_results(fit)


# --------------------------------------------------------------------------- #
#  etable — convenience re-export
# --------------------------------------------------------------------------- #


def etable(
    *results: EconometricResults,
    **kwargs: Any,
) -> Any:
    """
    Display a pyfixest-style regression table for StatsPAI results.

    If the results carry a ``_pyfixest_fit`` reference, uses pyfixest's
    native ``etable``. Otherwise falls back to a simple pandas summary.

    Parameters
    ----------
    *results : EconometricResults
        One or more fitted results.
    **kwargs
        Passed to ``pyfixest.etable()``.

    Returns
    -------
    str or DataFrame

    Examples
    --------
    Side-by-side comparison of nested specifications:

    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(1)
    >>> n = 200
    >>> df = pd.DataFrame({"x1": rng.normal(size=n), "x2": rng.normal(size=n),
    ...                    "firm": rng.integers(0, 10, n)})
    >>> df["wage"] = 1.0 + 0.5 * df["x1"] - 0.3 * df["x2"] + rng.normal(0, 0.5, n)
    >>> r1 = sp.regress("wage ~ x1", data=df)
    >>> r2 = sp.regress("wage ~ x1 + x2", data=df)
    >>> tab = sp.etable(r1, r2)  # DataFrame: one coefficient column per model
    >>> print(tab.round(2))

    With ``sp.feols`` fits (requires the ``fixest`` extra), pyfixest's
    native styled table is returned instead:

    >>> f1 = sp.feols("wage ~ x1 | firm", data=df)  # doctest: +SKIP
    >>> f2 = sp.feols("wage ~ x1 + x2 | firm", data=df)  # doctest: +SKIP
    >>> tab = sp.etable(f1, f2)  # doctest: +SKIP
    """
    pf_fits = [
        getattr(r, "_pyfixest_fit") for r in results if hasattr(r, "_pyfixest_fit")
    ]

    if pf_fits:
        pf = _check_pyfixest()
        return pf.etable(pf_fits, **kwargs)

    # Fallback: simple pandas table
    rows = {}
    for i, r in enumerate(results):
        col = f"({i + 1})"
        rows[col] = r.params
    return pd.DataFrame(rows)
