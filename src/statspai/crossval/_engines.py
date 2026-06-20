"""In-process engine adapters.

Each adapter turns an :class:`~statspai.crossval._spec.EstimandSpec` into a
normalised :class:`~statspai.crossval._agreement.EngineEstimate`, isolating the
quirks of one backend (StatsPAI native, pyfixest, linearmodels, DoubleML).

The contract: :meth:`EngineAdapter.run` **never raises**. Availability,
unsupported estimands and runtime failures all come back as an
``EngineEstimate`` with the appropriate ``status`` so the orchestrator can keep
failures loud (record them) without aborting the whole cross-validation.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from ._agreement import (
    STATUS_ERROR,
    STATUS_OK,
    STATUS_SKIPPED,
    STATUS_UNAVAILABLE,
    EngineEstimate,
)
from ._spec import EstimandSpec

# --------------------------------------------------------------------------- #
# Base
# --------------------------------------------------------------------------- #


class EngineAdapter(ABC):
    """Base class for a single backend.

    Subclasses implement :meth:`_available`, :meth:`_supports` and :meth:`_fit`.
    The public :meth:`run` wraps them with timing, availability checks and
    exception capture so a failing engine degrades to a status rather than
    crashing the run.
    """

    name: str = "base"
    #: Estimand keys this engine can express.
    supported: Tuple[str, ...] = ()
    #: Extra names this engine answers to in ``engines=[...]`` resolution.
    aliases: Tuple[str, ...] = ()

    def matches(self, name: str) -> bool:
        """Whether a user-supplied engine name selects this adapter."""
        n = (name or "").strip().lower()
        return n == self.name.lower() or n in {a.lower() for a in self.aliases}

    # -- subclass hooks --------------------------------------------------- #
    def _available(self) -> Tuple[bool, str]:
        """Return (available, reason-if-not)."""
        return True, ""

    def _supports(self, spec: EstimandSpec) -> Tuple[bool, str]:
        if self.supported and spec.estimand not in self.supported:
            return (
                False,
                f"{self.name} does not implement estimand '{spec.estimand}' "
                f"(supports: {', '.join(self.supported)}).",
            )
        return True, ""

    @abstractmethod
    def _fit(self, spec: EstimandSpec) -> EngineEstimate: ...

    # -- public ----------------------------------------------------------- #
    def run(self, spec: EstimandSpec) -> EngineEstimate:
        avail, why = self._available()
        if not avail:
            return EngineEstimate(
                engine=self.name,
                estimand=spec.estimand,
                term=_safe_term(spec),
                status=STATUS_UNAVAILABLE,
                message=why,
            )
        ok, why = self._supports(spec)
        if not ok:
            return EngineEstimate(
                engine=self.name,
                estimand=spec.estimand,
                term=_safe_term(spec),
                status=STATUS_SKIPPED,
                message=why,
            )
        t0 = time.perf_counter()
        try:
            est = self._fit(spec)
        except Exception as e:  # noqa: BLE001 — captured into status, stays loud
            return EngineEstimate(
                engine=self.name,
                estimand=spec.estimand,
                term=_safe_term(spec),
                status=STATUS_ERROR,
                message=f"{type(e).__name__}: {e}",
                elapsed_s=time.perf_counter() - t0,
            )
        if est.elapsed_s is None:
            est.elapsed_s = time.perf_counter() - t0
        return est


# --------------------------------------------------------------------------- #
# StatsPAI native
# --------------------------------------------------------------------------- #


class StatspaiAdapter(EngineAdapter):
    name = "statspai"
    supported = ("ols", "feols", "iv", "did", "poisson", "dml")

    def _fit(self, spec: EstimandSpec) -> EngineEstimate:
        import statspai as sp

        term = spec.focal_term()
        est = spec.estimand

        if est == "dml":
            if spec.y is None:
                raise ValueError("DML cross-check requires an outcome `y`.")
            dml_res = sp.dml(
                data=spec.data,
                y=spec.y,
                treat=spec.treatment,
                covariates=spec.covariates,
                model=spec.extra.get("model", "plr"),
            )
            return EngineEstimate(
                engine=self.name,
                estimand=est,
                term=spec.treatment or term,
                coef=float(dml_res.estimate),
                se=float(dml_res.se),
                pvalue=float(getattr(dml_res, "pvalue", np.nan)),
                nobs=int(len(spec.data)),
                vcov="dml",
                status=STATUS_OK,
                extra={"learners": "statspai-default"},
            )

        if est == "did":
            if spec.y is None:
                raise ValueError("DiD cross-check requires an outcome `y`.")
            x = spec.extra
            res_cs = sp.callaway_santanna(
                data=spec.data,
                y=spec.y,
                g=x["g"],
                t=x["t"],
                i=x["i"],
                control_group=x.get("control_group", "nevertreated"),
                estimator=x.get("est_method", "dr"),
                base_period=x.get("base_period", "universal"),
            )
            return EngineEstimate(
                engine=self.name,
                estimand=est,
                term="ATT",
                coef=float(res_cs.estimate),
                se=float(res_cs.se),
                nobs=int(len(spec.data)),
                vcov="cs-analytic",
                status=STATUS_OK,
                extra={"aggregation": "simple"},
            )

        res: Any
        if est == "ols":
            res = sp.regress(
                spec.build_formula(),
                data=spec.data,
                robust=_sp_robust(spec.vcov),
            )
        elif est == "feols":
            res = sp.feols(
                spec.build_formula(),
                data=spec.data,
                vcov=_sp_feols_vcov(spec.vcov, spec.cluster),
            )
        elif est == "poisson":
            res = sp.fepois(spec.build_formula(), data=spec.data)
        elif est == "iv":
            # sp.iv is a function alias that shadows the `iv` submodule at
            # runtime; fetch via getattr so the type checker doesn't resolve
            # it to the module object.
            iv_fn = getattr(sp, "iv")
            res = iv_fn(_aer_iv_formula(spec), data=spec.data)
        else:  # pragma: no cover — guarded by `supported`
            raise ValueError(f"unsupported estimand {est}")

        return _from_statspai_result(res, self.name, est, term, spec)


# --------------------------------------------------------------------------- #
# pyfixest
# --------------------------------------------------------------------------- #


class PyfixestAdapter(EngineAdapter):
    name = "pyfixest"
    supported = ("ols", "feols", "iv", "poisson")

    def _available(self) -> Tuple[bool, str]:
        try:
            import pyfixest  # noqa: F401
        except Exception as e:  # noqa: BLE001
            return False, f"pyfixest not importable: {type(e).__name__}"
        return True, ""

    def _fit(self, spec: EstimandSpec) -> EngineEstimate:
        import pyfixest as pf

        term = spec.focal_term()
        fml = _fixest_formula(spec)
        vcov = _pf_vcov(spec.vcov, spec.cluster)
        if spec.estimand == "poisson":
            m = pf.fepois(fml, data=spec.data, vcov=vcov)
        else:
            m = pf.feols(fml, data=spec.data, vcov=vcov)
        tidy = m.tidy()
        if term not in tidy.index:
            raise KeyError(
                f"focal term '{term}' not in pyfixest coefficients "
                f"{list(tidy.index)}"
            )
        row = tidy.loc[term]
        return EngineEstimate(
            engine=self.name,
            estimand=spec.estimand,
            term=term,
            coef=float(row["Estimate"]),
            se=float(row["Std. Error"]),
            tstat=float(row.get("t value", np.nan)),
            pvalue=float(row.get("Pr(>|t|)", np.nan)),
            ci_lower=float(row.get("2.5%", np.nan)),
            ci_upper=float(row.get("97.5%", np.nan)),
            nobs=int(getattr(m, "_N", len(spec.data))),
            vcov=str(vcov),
            status=STATUS_OK,
        )


# --------------------------------------------------------------------------- #
# linearmodels
# --------------------------------------------------------------------------- #


class LinearmodelsAdapter(EngineAdapter):
    name = "linearmodels"
    supported = ("ols", "feols", "iv")

    def _available(self) -> Tuple[bool, str]:
        try:
            import linearmodels  # noqa: F401
        except Exception as e:  # noqa: BLE001
            return False, f"linearmodels not importable: {type(e).__name__}"
        return True, ""

    def _supports(self, spec: EstimandSpec) -> Tuple[bool, str]:
        ok, why = super()._supports(spec)
        if not ok:
            return ok, why
        # IV + absorbed FE together is not exposed cleanly; decline rather than
        # silently fit a different model.
        if spec.estimand == "iv" and spec.fixed_effects:
            return (
                False,
                "linearmodels adapter does not cross-check IV with absorbed "
                "fixed effects; use pyfixest for that estimand.",
            )
        return True, ""

    def _fit(self, spec: EstimandSpec) -> EngineEstimate:
        from linearmodels.iv import IV2SLS
        from linearmodels.iv.absorbing import AbsorbingLS

        term = spec.focal_term()
        df = spec.data.copy()
        cov_type, cov_cfg = _lm_cov(spec.vcov, spec.cluster, df)

        res: Any  # AbsorbingLSResults | IV2SLS results — unified downstream
        if spec.estimand == "feols" and spec.fixed_effects:
            exog_cols = ([spec.treatment] if spec.treatment else []) + [
                c for c in spec.covariates
            ]
            exog_cols = [c for c in dict.fromkeys(exog_cols) if c]
            absorb = df[spec.fixed_effects].astype("category")
            res = AbsorbingLS(df[spec.y], df[exog_cols], absorb=absorb).fit(
                cov_type=cov_type, **cov_cfg
            )
        elif spec.estimand == "iv":
            exog_cols = ["const"] + [c for c in spec.covariates if c != spec.treatment]
            df["const"] = 1.0
            res = IV2SLS(
                df[spec.y],
                df[exog_cols],
                df[spec.endog],
                df[spec.instruments],
            ).fit(cov_type=cov_type, **cov_cfg)
        else:  # plain OLS via IV2SLS with no endog
            exog_cols = ["const"] + (
                ([spec.treatment] if spec.treatment else [])
                + [c for c in spec.covariates if c != spec.treatment]
            )
            df["const"] = 1.0
            exog_cols = [c for c in dict.fromkeys(exog_cols) if c]
            res = IV2SLS(df[spec.y], df[exog_cols], None, None).fit(
                cov_type=cov_type, **cov_cfg
            )

        if term not in res.params.index:
            raise KeyError(
                f"focal term '{term}' not in linearmodels params "
                f"{list(res.params.index)}"
            )
        ci = res.conf_int()
        return EngineEstimate(
            engine=self.name,
            estimand=spec.estimand,
            term=term,
            coef=float(res.params[term]),
            se=float(res.std_errors[term]),
            tstat=float(res.tstats[term]),
            pvalue=float(res.pvalues[term]),
            ci_lower=float(ci.loc[term].iloc[0]),
            ci_upper=float(ci.loc[term].iloc[1]),
            nobs=int(res.nobs),
            vcov=cov_type,
            status=STATUS_OK,
        )


# --------------------------------------------------------------------------- #
# DoubleML (statistical-mode engine)
# --------------------------------------------------------------------------- #


class DoublemlAdapter(EngineAdapter):
    name = "doubleml"
    supported = ("dml",)

    def _available(self) -> Tuple[bool, str]:
        try:
            import doubleml  # noqa: F401
            import sklearn  # noqa: F401
        except Exception as e:  # noqa: BLE001
            return False, f"doubleml/sklearn not importable: {type(e).__name__}"
        return True, ""

    def _fit(self, spec: EstimandSpec) -> EngineEstimate:
        import doubleml as dml
        from sklearn.ensemble import RandomForestRegressor

        term = spec.treatment or spec.focal_term()
        x_cols = [c for c in spec.covariates if c not in (spec.treatment,)]
        n_folds = int(spec.extra.get("n_folds", 5))
        seed = int(spec.extra.get("seed", 42))
        data = dml.DoubleMLData(spec.data, y_col=spec.y, d_cols=term, x_cols=x_cols)
        ml_l = RandomForestRegressor(n_estimators=100, random_state=seed)
        ml_m = RandomForestRegressor(n_estimators=100, random_state=seed)
        np.random.seed(seed)
        plr = dml.DoubleMLPLR(data, ml_l, ml_m, n_folds=n_folds)
        plr.fit()
        ci = plr.confint()
        return EngineEstimate(
            engine=self.name,
            estimand=spec.estimand,
            term=term,
            coef=float(plr.coef[0]),
            se=float(plr.se[0]),
            pvalue=float(plr.pval[0]),
            ci_lower=float(ci.iloc[0, 0]),
            ci_upper=float(ci.iloc[0, 1]),
            nobs=int(len(spec.data)),
            vcov="dml",
            status=STATUS_OK,
            extra={"n_folds": n_folds, "ml": "RandomForest"},
        )


# --------------------------------------------------------------------------- #
# Registry + resolution
# --------------------------------------------------------------------------- #

_INPROCESS_ADAPTERS: List[EngineAdapter] = [
    StatspaiAdapter(),
    PyfixestAdapter(),
    LinearmodelsAdapter(),
    DoublemlAdapter(),
]


def inprocess_adapters() -> List[EngineAdapter]:
    return list(_INPROCESS_ADAPTERS)


def adapter_by_name(name: str) -> Optional[EngineAdapter]:
    for a in _INPROCESS_ADAPTERS:
        if a.name == name:
            return a
    return None


# --------------------------------------------------------------------------- #
# Helpers — extraction + vcov translation
# --------------------------------------------------------------------------- #


def _from_statspai_result(
    res: Any, engine: str, estimand: str, term: str, spec: EstimandSpec
) -> EngineEstimate:
    tidy = res.tidy()
    match = tidy[tidy["term"] == term]
    if match.empty:
        # tolerate patsy-mangled names (e.g. categorical expansions)
        cand = [t for t in tidy["term"] if t == term or t.endswith(f"[{term}]")]
        if cand:
            match = tidy[tidy["term"] == cand[0]]
    if match.empty:
        raise KeyError(
            f"focal term '{term}' not in statspai coefficients " f"{list(tidy['term'])}"
        )
    row = match.iloc[0]
    nobs = getattr(res, "nobs", None)
    return EngineEstimate(
        engine=engine,
        estimand=estimand,
        term=term,
        coef=float(row["estimate"]),
        se=float(row["std_error"]),
        tstat=_get(row, "statistic"),
        pvalue=_get(row, "p_value"),
        ci_lower=_get(row, "conf_low"),
        ci_upper=_get(row, "conf_high"),
        nobs=int(nobs) if nobs is not None else int(len(spec.data)),
        vcov=spec.vcov or "default",
        status=STATUS_OK,
    )


def _fixest_formula(spec: EstimandSpec) -> str:
    """Build a pyfixest-style formula (handles FE and IV blocks).

    Endogenous regressors must NOT appear in the main RHS — fixest places them
    only in the trailing ``endog ~ instruments`` block. With no fixed effects
    the IV form is ``y ~ exog | endog ~ instr`` (no ``0`` placeholder, which
    pyfixest rejects).
    """
    endog_set = set(spec.endog)
    rhs = (
        [spec.treatment] if (spec.treatment and spec.treatment not in endog_set) else []
    ) + [c for c in spec.covariates if c != spec.treatment and c not in endog_set]
    rhs = [c for c in dict.fromkeys(rhs) if c]
    main = " + ".join(rhs) if rhs else "1"
    parts = [f"{spec.y} ~ {main}"]
    if spec.fixed_effects:
        parts.append(" + ".join(spec.fixed_effects))
    if spec.endog:
        parts.append(f"{' + '.join(spec.endog)} ~ {' + '.join(spec.instruments)}")
    return " | ".join(parts)


def _aer_iv_formula(spec: EstimandSpec) -> str:
    """Build StatsPAI's AER-style IV formula: ``y ~ exog + (endog ~ instr)``."""
    exog = [c for c in spec.covariates if c != spec.treatment]
    exog_str = " + ".join(exog) if exog else "1"
    iv_block = f"({' + '.join(spec.endog)} ~ {' + '.join(spec.instruments)})"
    return f"{spec.y} ~ {exog_str} + {iv_block}"


def _sp_robust(vcov: Optional[str]) -> str:
    if not vcov:
        return "nonrobust"
    v = vcov.lower()
    if v in ("hc1", "hc2", "hc3", "robust"):
        return "hc1" if v == "robust" else v
    if v in ("iid", "nonrobust", "classical"):
        return "nonrobust"
    return vcov


def _sp_feols_vcov(vcov: Optional[str], cluster: List[str]) -> Optional[str]:
    if cluster:
        return None  # let caller-supplied formula/cluster default apply
    if not vcov:
        return None
    return vcov


def _pf_vcov(vcov: Optional[str], cluster: List[str]) -> Union[str, Dict[str, str]]:
    if cluster:
        return {"CRV1": " + ".join(cluster)}
    if not vcov:
        return "iid"
    v = vcov.lower()
    if v in ("classical", "nonrobust"):
        return "iid"
    if v in ("robust", "hc1"):
        return "HC1"
    return vcov


def _lm_cov(
    vcov: Optional[str], cluster: List[str], df: pd.DataFrame
) -> Tuple[str, Dict[str, Any]]:
    if cluster:
        return "clustered", {"clusters": df[cluster[0]]}
    if not vcov:
        return "unadjusted", {}
    v = vcov.lower()
    if v in ("classical", "nonrobust", "iid"):
        return "unadjusted", {}
    if v in ("robust", "hc1"):
        return "robust", {}
    return v, {}


def _get(row: Any, key: str) -> Optional[float]:
    try:
        val = row[key]
    except (KeyError, IndexError):
        return None
    try:
        f = float(val)
    except (TypeError, ValueError):
        return None
    return None if (np.isnan(f) or np.isinf(f)) else f


def _safe_term(spec: EstimandSpec) -> Optional[str]:
    try:
        return spec.focal_term()
    except Exception:  # noqa: BLE001
        return spec.term or spec.treatment
