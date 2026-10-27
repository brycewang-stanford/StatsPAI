"""External-engine adapters: R (via ``Rscript``) and Stata (via batch ``do``).

These adapters shell out to a real R or Stata install, the single most
convincing cross-check StatsPAI can offer — they exercise the *exact* engines
(``fixest``, ``ivregress``/``reghdfe``) an econometrician would reach for, in a
separate process and a separate language.

Design notes
------------
* **No silent fakery.** If the binary or required package is absent the adapter
  returns ``status="unavailable"`` with a precise reason. It never substitutes a
  Python approximation for "the R number".
* **Subprocess, not in-process.** We avoid an ``rpy2`` / ``pystata`` hard
  dependency: data goes out as a temp CSV, a generated script fits the model and
  prints a tiny JSON/CSV result, which we read back. Works wherever the CLI does.
* **MCP-backed engines** (calling out to an external Stata/R MCP server such as
  SepineTam's ``stata-mcp``) are a planned third adapter family; the
  subprocess adapters here share the same :class:`EngineEstimate` contract, so
  an MCP adapter slots in without touching the reconciliation layer.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
import tempfile
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ._agreement import STATUS_OK, EngineEstimate
from ._engines import EngineAdapter, _fixest_formula
from ._spec import EstimandSpec

_DEFAULT_TIMEOUT = 120  # seconds per external fit


# --------------------------------------------------------------------------- #
# R via Rscript + fixest
# --------------------------------------------------------------------------- #


class RscriptAdapter(EngineAdapter):
    """Cross-check against R's ``fixest`` (and ``AER`` for IV) via ``Rscript``."""

    name = "R"
    aliases = ("rscript", "r::fixest", "r::feols", "r::ivreg", "r::lm")
    supported = ("ols", "feols", "iv", "poisson")
    weights_estimands = ("ols", "feols", "iv", "poisson")

    def __init__(self, timeout: int = _DEFAULT_TIMEOUT) -> None:
        self.timeout = timeout

    def _available(self) -> Tuple[bool, str]:
        if shutil.which("Rscript") is None:
            return False, "Rscript not found on PATH"
        ok, missing = _r_have_packages(["fixest", "jsonlite"])
        if not ok:
            return False, f"R packages missing: {', '.join(missing)}"
        return True, ""

    def _fit(self, spec: EstimandSpec) -> EngineEstimate:
        term = spec.focal_term()
        fml = _fixest_formula(spec)
        vcov = _r_vcov(spec.vcov, spec.cluster)
        payload = _run_r_fixest(
            spec.data,
            fml,
            term,
            vcov,
            spec.estimand,
            self.timeout,
            spec.weights or "",
            level=1.0 - spec.alpha,
        )
        return EngineEstimate(
            engine="R::fixest",
            estimand=spec.estimand,
            term=term,
            coef=_f(payload.get("coef")),
            se=_f(payload.get("se")),
            tstat=_f(payload.get("tstat")),
            pvalue=_f(payload.get("pval")),
            ci_lower=_f(payload.get("ci_lower")),
            ci_upper=_f(payload.get("ci_upper")),
            nobs=int(payload["nobs"]) if payload.get("nobs") else None,
            vcov=str(vcov),
            status=STATUS_OK,
            extra={"r_coef_name": payload.get("coef_name")},
        )


def _r_have_packages(pkgs: List[str]) -> Tuple[bool, List[str]]:
    expr = (
        "miss <- c(); "
        + "; ".join(
            f'if (!requireNamespace("{p}", quietly=TRUE)) miss <- c(miss, "{p}")'
            for p in pkgs
        )
        + '; cat(paste(miss, collapse=","))'
    )
    try:
        out = subprocess.run(
            ["Rscript", "-e", expr],
            capture_output=True,
            text=True,
            timeout=30,
        )
    except Exception:  # noqa: BLE001
        return False, pkgs
    missing = [m for m in out.stdout.strip().split(",") if m]
    return (len(missing) == 0), missing


# R script template. Fits with fixest, extracts the focal coefficient (handling
# fixest's ``fit_<endog>`` IV naming), prints a one-object JSON to stdout.
_R_SCRIPT = r"""
suppressMessages(library(fixest)); suppressMessages(library(jsonlite))
args <- commandArgs(trailingOnly=TRUE)
csv <- args[1]; fml <- args[2]; term <- args[3]; vcov <- args[4]; est <- args[5]
wt <- if (length(args) >= 6) args[6] else ""
lev <- if (length(args) >= 7) as.numeric(args[7]) else 0.95
d <- read.csv(csv)
if (startsWith(vcov, "~")) vcov <- as.formula(vcov)
w <- if (nzchar(wt)) as.formula(paste0("~", wt)) else NULL
if (est == "poisson") {
  m <- fepois(as.formula(fml), data=d, vcov=vcov, weights=w)
} else {
  m <- feols(as.formula(fml), data=d, vcov=vcov, weights=w)
}
ct <- as.data.frame(coeftable(m))
nm <- rownames(ct)
pick <- term
if (!(pick %in% nm) && (paste0("fit_", term) %in% nm)) pick <- paste0("fit_", term)
if (!(pick %in% nm)) { cat(toJSON(list(error=paste("term not found:", term,
  "have:", paste(nm, collapse=",")) ), auto_unbox=TRUE)); quit(status=0) }
row <- ct[pick, ]
ci <- tryCatch(confint(m, level=lev)[pick, ], error=function(e) c(NA, NA))
out <- list(coef=unname(row[[1]]), se=unname(row[[2]]),
            tstat=unname(row[[3]]), pval=unname(row[[4]]),
            ci_lower=unname(ci[[1]]), ci_upper=unname(ci[[2]]),
            nobs=as.integer(nobs(m)), coef_name=pick)
# digits=NA keeps full double precision — jsonlite rounds to 4 dp by default,
# which would spuriously trip the exact-mode coefficient tolerance.
cat(toJSON(out, auto_unbox=TRUE, na="null", digits=NA))
"""


def _run_r_fixest(
    data: Any,
    formula: str,
    term: str,
    vcov: str,
    estimand: str,
    timeout: int,
    weights: str = "",
    level: float = 0.95,
) -> Dict[str, Any]:
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = os.path.join(tmp, "data.csv")
        script_path = os.path.join(tmp, "fit.R")
        data.to_csv(csv_path, index=False)
        with open(script_path, "w", encoding="utf-8") as fh:
            fh.write(_R_SCRIPT)
        proc = subprocess.run(
            [
                "Rscript",
                script_path,
                csv_path,
                formula,
                term,
                vcov,
                estimand,
                weights,
                repr(float(level)),
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"Rscript failed (rc={proc.returncode}): "
                f"{proc.stderr.strip()[:400]}"
            )
        out = proc.stdout.strip()
        if not out:
            raise RuntimeError(
                f"Rscript produced no output. stderr: " f"{proc.stderr.strip()[:400]}"
            )
        payload: Dict[str, Any] = json.loads(out)
        if "error" in payload:
            raise RuntimeError(f"R: {payload['error']}")
        return payload


def _r_vcov(vcov: Optional[str], cluster: List[str]) -> str:
    if cluster:
        # A one-sided formula names the cluster variables. The bare string
        # "cluster" makes fixest cluster on the first fixed effect instead
        # (and errors when there is none).
        return "~" + " + ".join(cluster)
    if not vcov:
        return "iid"
    v = vcov.lower()
    if v in ("classical", "nonrobust", "iid"):
        return "iid"
    if v in ("robust", "hc1", "hetero"):
        return "hetero"
    return v


# --------------------------------------------------------------------------- #
# R's `did` package (Callaway–Sant'Anna) via Rscript
# --------------------------------------------------------------------------- #


class RDidAdapter(EngineAdapter):
    """Cross-check Callaway–Sant'Anna against R's ``did`` package via Rscript.

    This is the direct answer to Scott Cunningham's cross-package experiment:
    reconcile StatsPAI's ``callaway_santanna`` overall ATT against
    ``did::att_gt`` + ``aggte(type="simple")`` — the canonical R implementation.
    """

    name = "R::did"
    aliases = ("rdid", "r::csdid", "did::r")
    supported = ("did",)

    def __init__(self, timeout: int = _DEFAULT_TIMEOUT) -> None:
        self.timeout = timeout

    def _available(self) -> Tuple[bool, str]:
        if shutil.which("Rscript") is None:
            return False, "Rscript not found on PATH"
        ok, missing = _r_have_packages(["did", "jsonlite"])
        if not ok:
            return False, f"R packages missing: {', '.join(missing)}"
        return True, ""

    def _fit(self, spec: EstimandSpec) -> EngineEstimate:
        x = spec.extra
        payload = _run_r_did(
            spec.data,
            yname=spec.y,
            gname=x["g"],
            tname=x["t"],
            iname=x["i"],
            control_group=x.get("control_group", "nevertreated"),
            est_method=x.get("est_method", "dr"),
            base_period=x.get("base_period", "universal"),
            timeout=self.timeout,
        )
        return EngineEstimate(
            engine="R::did",
            estimand="did",
            term="ATT",
            coef=_f(payload.get("coef")),
            se=_f(payload.get("se")),
            nobs=int(payload["nobs"]) if payload.get("nobs") else None,
            vcov="cs-analytic",
            status=STATUS_OK,
            extra={"aggregation": "simple"},
        )


# R script: att_gt + aggte(type="simple"); bstrap=FALSE for a deterministic
# analytic SE so the verdict isn't perturbed by the multiplier-bootstrap RNG.
_R_DID_SCRIPT = r"""
suppressMessages(library(did)); suppressMessages(library(jsonlite))
a <- commandArgs(trailingOnly=TRUE)
csv<-a[1]; yn<-a[2]; gn<-a[3]; tn<-a[4]; idn<-a[5]; cg<-a[6]; em<-a[7]; bp<-a[8]
d <- read.csv(csv)
# read.csv types integer-coded cohort/time/id columns as 'integer', but did's
# att_gt returns a DIFFERENT estimate for integer vs numeric group columns
# (a real cross-package fragility). Coerce to numeric to match the in-memory
# data(mpdta) convention — and StatsPAI — exactly.
d[[gn]] <- as.numeric(d[[gn]]); d[[tn]] <- as.numeric(d[[tn]])
d[[idn]] <- as.numeric(d[[idn]])
out <- att_gt(yname=yn, tname=tn, idname=idn, gname=gn, control_group=cg,
              est_method=em, base_period=bp, bstrap=FALSE, data=d)
agg <- aggte(out, type="simple")
res <- list(coef=unname(agg$overall.att), se=unname(agg$overall.se),
            nobs=as.integer(nrow(d)))
cat(toJSON(res, auto_unbox=TRUE, na="null", digits=NA))
"""


def _run_r_did(
    data: Any,
    *,
    yname: Any,
    gname: str,
    tname: str,
    iname: str,
    control_group: str,
    est_method: str,
    base_period: str,
    timeout: int,
) -> Dict[str, Any]:
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = os.path.join(tmp, "data.csv")
        script_path = os.path.join(tmp, "did.R")
        data.to_csv(csv_path, index=False)
        with open(script_path, "w", encoding="utf-8") as fh:
            fh.write(_R_DID_SCRIPT)
        proc = subprocess.run(
            [
                "Rscript",
                script_path,
                csv_path,
                str(yname),
                gname,
                tname,
                iname,
                control_group,
                est_method,
                base_period,
            ],
            capture_output=True,
            text=True,
            timeout=timeout,
        )
        if proc.returncode != 0:
            raise RuntimeError(
                f"Rscript (did) failed (rc={proc.returncode}): "
                f"{proc.stderr.strip()[:400]}"
            )
        out = proc.stdout.strip()
        if not out:
            raise RuntimeError(
                f"Rscript (did) produced no output. stderr: "
                f"{proc.stderr.strip()[:400]}"
            )
        payload: Dict[str, Any] = json.loads(out)
        return payload


# --------------------------------------------------------------------------- #
# Stata via batch do-file
# --------------------------------------------------------------------------- #


class StataAdapter(EngineAdapter):
    """Cross-check against Stata (``regress`` / ``ivregress`` / ``reghdfe``).

    Exercised only where a Stata batch binary is on PATH. Elsewhere it returns
    ``unavailable`` — which is itself a tested degradation path.
    """

    name = "Stata"
    aliases = ("stata-mp", "stata-se", "stata::reghdfe", "stata::ivregress")
    supported = ("ols", "feols", "iv")
    weights_estimands = ("ols", "feols", "iv")

    _BINARIES = ("stata-mp", "stata-se", "stata", "StataMP", "StataSE")

    def __init__(self, timeout: int = _DEFAULT_TIMEOUT) -> None:
        self.timeout = timeout

    def _binary(self) -> Optional[str]:
        for b in self._BINARIES:
            if shutil.which(b):
                return shutil.which(b)
        env = os.environ.get("STATSPAI_STATA_BIN")
        if env and os.path.exists(env):
            return env
        return None

    def _available(self) -> Tuple[bool, str]:
        if self._binary() is None:
            return (
                False,
                "no Stata batch binary on PATH (looked for stata-mp / stata-se "
                "/ stata; set STATSPAI_STATA_BIN to override)",
            )
        return True, ""

    def _supports(self, spec: EstimandSpec) -> Tuple[bool, str]:
        ok, why = super()._supports(spec)
        if not ok:
            return ok, why
        if len(spec.cluster) > 1 and not spec.fixed_effects:
            return (
                False,
                "Stata's regress / ivregress cluster on one variable; "
                "multiway clustering is left to the reghdfe paths.",
            )
        try:
            _stata_vce(spec)
        except ValueError as exc:
            return False, str(exc)
        return True, ""

    def _fit(self, spec: EstimandSpec) -> EngineEstimate:
        term = spec.focal_term()
        binary = self._binary()
        if binary is None:  # pragma: no cover — guarded by _available()
            raise RuntimeError("Stata binary disappeared after availability check")
        coef, se, nobs, cmd = _run_stata(binary, spec, term, self.timeout)
        tstat = coef / se if se else None
        # Normal-approx inference so the report has CI/p even from a bare
        # coef/se dump; flagged via vcov label.
        from scipy import stats as _st

        crit = float(_st.norm.ppf(1.0 - spec.alpha / 2.0))
        ci_lower = coef - crit * se if se else None
        ci_upper = coef + crit * se if se else None

        pval = float(2 * _st.norm.sf(abs(tstat))) if tstat else None
        return EngineEstimate(
            engine="Stata",
            estimand=spec.estimand,
            term=term,
            coef=coef,
            se=se,
            tstat=tstat,
            pvalue=pval,
            ci_lower=ci_lower,
            ci_upper=ci_upper,
            nobs=nobs,
            vcov=spec.vcov or "default",
            status=STATUS_OK,
            extra={"stata_cmd": cmd},
        )


def _stata_vce(spec: EstimandSpec) -> str:
    """The ``vce()`` option for the requested variance ("" for the default).

    Until 2026-10 no ``vce()`` was written, so a robust or clustered
    cross-check compared the other engines with Stata's classical SE.
    """
    if spec.cluster:
        return f"vce(cluster {' '.join(spec.cluster)})"
    v = (spec.vcov or "").strip().lower()
    if v in ("", "iid", "classical", "nonrobust", "unadjusted", "ols"):
        return ""
    if v in ("robust", "hc1", "hetero"):
        return "vce(robust)"
    raise ValueError(f"Stata adapter has no translation for vcov={spec.vcov!r}.")


def _stata_command(spec: EstimandSpec) -> str:
    y = spec.y
    endog_set = set(spec.endog)
    weight = f" [aweight={spec.weights}]" if spec.weights else ""
    vce = _stata_vce(spec)
    if spec.estimand == "iv":
        endog = " ".join(spec.endog)
        instr = " ".join(spec.instruments)
        exog = [
            c
            for c in dict.fromkeys(
                ([spec.treatment] if spec.treatment else []) + list(spec.covariates)
            )
            if c and c not in endog_set
        ]
        rhs = " ".join(exog)
        # ``small``: N - K divisor and t inference, the convention of the
        # other engines (ivregress and ivreg2 default to large-sample).
        opts = " ".join(o for o in ("small", vce) if o)
        head = f"{y} {rhs} ({endog} = {instr}){weight}".replace("  ", " ")
        if spec.fixed_effects:
            absorb = " ".join(spec.fixed_effects)
            return f"ivreghdfe {head}, absorb({absorb}) {opts}"
        return f"ivregress 2sls {head}, {opts}"
    exog = [c for c in spec.covariates if c not in endog_set]
    rhs = " ".join(([spec.treatment] if spec.treatment else []) + exog)
    if spec.estimand == "feols" and spec.fixed_effects:
        absorb = " ".join(spec.fixed_effects)
        opts = " ".join(o for o in (f"absorb({absorb})", vce) if o)
        return f"reghdfe {y} {rhs}{weight}, {opts}"
    return f"regress {y} {rhs}{weight}" + (f", {vce}" if vce else "")


def _stata_do(spec: EstimandSpec, term: str, csv_path: str, out_path: str) -> str:
    """The do-file: read the CSV as doubles, fit, write full-precision results.

    ``set type double`` keeps the CSV's doubles (Stata's default reads them
    as floats, a relative error near 1e-7 that trips the exact-mode
    tolerance), and ``%21.0g`` writes 17 significant digits (``file write``
    otherwise uses ``%10.0g``).
    """
    return (
        "set type double\n"
        f'import delimited "{csv_path}", clear\n'
        f"{_stata_command(spec)}\n"
        f'file open _h using "{out_path}", write replace\n'
        'file write _h "coef,se,n" _n\n'
        f'file write _h %21.0g (_b[{term}]) "," %21.0g (_se[{term}]) "," '
        "%12.0f (e(N)) _n\n"
        "file close _h\n"
    )


def _run_stata(
    binary: str, spec: EstimandSpec, term: str, timeout: int
) -> Tuple[float, float, Optional[int], str]:
    cmd = _stata_command(spec)
    with tempfile.TemporaryDirectory() as tmp:
        csv_path = os.path.join(tmp, "data.csv")
        do_path = os.path.join(tmp, "fit.do")
        out_path = os.path.join(tmp, "out.csv")
        spec.data.to_csv(csv_path, index=False)
        do = _stata_do(spec, term, csv_path, out_path)
        with open(do_path, "w", encoding="utf-8") as fh:
            fh.write(do)
        proc = subprocess.run(
            [binary, "-b", "do", do_path],
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=tmp,
        )
        if not os.path.exists(out_path):
            raise RuntimeError(
                f"Stata produced no result (rc={proc.returncode}); cmd={cmd!r}. "
                f"Check that the estimator command is installed (e.g. reghdfe)."
            )
        import csv as _csv

        with open(out_path, encoding="utf-8") as fh:
            rows = list(_csv.DictReader(fh))
        if not rows:
            raise RuntimeError("Stata result file empty.")
        r0 = rows[0]
        nobs = int(float(r0["n"])) if r0.get("n") not in (None, "", ".") else None
        return float(r0["coef"]), float(r0["se"]), nobs, cmd


# --------------------------------------------------------------------------- #
# Registry
# --------------------------------------------------------------------------- #

_EXTERNAL_ADAPTERS: List[EngineAdapter] = [
    RscriptAdapter(),
    RDidAdapter(),
    StataAdapter(),
]


def external_adapters() -> List[EngineAdapter]:
    return list(_EXTERNAL_ADAPTERS)


def _f(x: Any) -> Optional[float]:
    if x is None:
        return None
    try:
        f = float(x)
    except (TypeError, ValueError):
        return None
    return None if (np.isnan(f) or np.isinf(f)) else f
