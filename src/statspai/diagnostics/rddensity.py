"""
Cattaneo-Jansson-Ma (2020) density discontinuity test.

Tests whether the density of the running variable is continuous at
the RD cutoff. Replaces McCrary (2008) as the modern standard for
RD manipulation testing.

Uses local polynomial density estimation on each side of the cutoff
with data-driven bandwidth selection and bias-corrected inference.

References
----------
Cattaneo, M.D., Jansson, M. and Ma, X. (2020).
"Simple Local Polynomial Density Estimators."
*Journal of the American Statistical Association*, 115(531), 1449-1455. [@cattaneo2020simple]

Cattaneo, M.D., Jansson, M. and Ma, X. (2018).
"Manipulation Testing Based on Density Discontinuity."
*The Stata Journal*, 18(1), 234-261. [@cattaneo2018manipulation]
"""

import json
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, Sequence, Tuple, Union, cast

import numpy as np
import pandas as pd
from scipy import stats

from ..core.results import CausalResult
from ..exceptions import (
    ConvergenceFailure,
    DataInsufficient,
    MethodIncompatibility,
)


def _validate_inputs(
    data: Any,
    x: Any,
    c: float,
    p: int,
    alpha: float,
    backend: Any,
) -> str:
    if not isinstance(data, pd.DataFrame):
        raise MethodIncompatibility(
            "`data` must be a pandas DataFrame.",
            recovery_hint="Pass a DataFrame containing the running variable.",
            diagnostics={"type": type(data).__name__},
        )
    if not isinstance(x, str) or not x:
        raise MethodIncompatibility(
            "`x` must be a non-empty running-variable column name.",
            recovery_hint="Pass x='running_variable'.",
            diagnostics={"x": repr(x)},
        )
    if x not in data.columns:
        raise MethodIncompatibility(
            f"Column '{x}' not found in data.",
            recovery_hint="Check `x` against data.columns.",
            diagnostics={"x": x, "available_columns": list(data.columns)},
        )
    if not np.isfinite(c):
        raise MethodIncompatibility(
            "`c` must be finite.",
            recovery_hint="Pass a finite RD cutoff.",
            diagnostics={"c": c},
        )
    if not isinstance(p, int) or isinstance(p, bool) or not 1 <= p <= 7:
        raise MethodIncompatibility(
            "p must be an integer between 1 and 7 for rddensity defaults.",
            recovery_hint="Use a local-polynomial order such as p=2.",
            diagnostics={"p": p},
        )
    if not np.isfinite(alpha) or not 0 < alpha < 1:
        raise MethodIncompatibility(
            "alpha must be between 0 and 1.",
            recovery_hint="Pass a significance level such as alpha=0.05.",
            diagnostics={"alpha": alpha},
        )
    if not isinstance(backend, str):
        raise MethodIncompatibility(
            "backend must be 'native' or 'r'.",
            recovery_hint="Use backend='native' or backend='r'.",
            diagnostics={"backend": repr(backend)},
        )
    return backend.lower().replace("-", "_")


def _running_values(data: pd.DataFrame, x: str) -> np.ndarray:
    try:
        X = data[x].values.astype(float)
    except (TypeError, ValueError) as exc:
        raise DataInsufficient(
            f"Column '{x}' must be numeric.",
            recovery_hint="Convert the running variable to numeric values.",
            diagnostics={"x": x},
        ) from exc
    return np.asarray(X[np.isfinite(X)], dtype=float)


def _validate_support(X: np.ndarray, c: float) -> Tuple[int, int, int]:
    n = len(X)
    if n < 20:
        raise DataInsufficient(
            "Need at least 20 observations.",
            recovery_hint="Provide at least 20 finite running-variable values.",
            diagnostics={"n_valid": int(n), "min_required": 20},
        )
    n_l = int(np.sum(X < c))
    n_r = int(np.sum(X >= c))
    if n_l < 5 or n_r < 5:
        raise DataInsufficient(
            "Not enough observations on each side.",
            recovery_hint="Provide at least 5 observations on each side of the cutoff.",
            diagnostics={"n_left": n_l, "n_right": n_r},
        )
    return n, n_l, n_r


def rddensity(
    data: pd.DataFrame,
    x: str,
    c: float = 0,
    p: int = 2,
    h: Optional[Union[float, Sequence[float]]] = None,
    alpha: float = 0.05,
    backend: str = "native",
) -> CausalResult:
    """
    CJM (2020) density discontinuity test for RD manipulation.

    Modern replacement for McCrary (2008). Uses local polynomial
    density estimation with bias-corrected inference.

    Parameters
    ----------
    data : pd.DataFrame
    x : str
        Running variable.
    c : float, default 0
        RD cutoff.
    p : int, default 2
        Polynomial order for density estimation.
    h : float or length-2 sequence, optional
        Bandwidth. A scalar applies the same bandwidth on both sides;
        a length-2 sequence is interpreted as ``(h_left, h_right)``.
        Default: automatic StatsPAI native pilot rule.
    alpha : float, default 0.05
    backend : {"native", "r"}, default "native"
        ``"native"`` uses StatsPAI's Python port of the default
        ``rddensity`` unrestricted triangular-kernel selector/test path.
        ``"r"`` delegates to ``rddensity::rddensity`` through
        ``Rscript`` when the R package is installed, matching the
        reference package's selector and test statistic.

    Returns
    -------
    CausalResult
        - ``estimate``: T-statistic for density discontinuity
        - ``pvalue``: p-value for H0: continuous density at cutoff
        - ``model_info``: density estimates left/right, bandwidth

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(3)
    >>> df = pd.DataFrame({'score': rng.normal(0, 1, 2000)})
    >>> result = sp.rddensity(df, x='score', c=0)
    >>> bool(0.0 <= result.pvalue <= 1.0)
    True
    >>> manipulated = result.pvalue < 0.05  # True => evidence of sorting

    Notes
    -----
    The test estimates the density from the left and right using
    local polynomial regression on the empirical CDF, then tests:

        H0: f_+(c) = f_-(c)  (no density discontinuity)
        H1: f_+(c) ≠ f_-(c)  (manipulation at cutoff)

    Native-path differences from McCrary (2008):
    - No arbitrary binning step
    - CJM/rddensity combination bandwidth selector
    - Local-polynomial density estimate on the empirical CDF, matching
      the default ``rddensity`` unrestricted triangular-kernel path

    See Cattaneo, Jansson & Ma (2020, *JASA*).
    """
    backend_norm = _validate_inputs(data, x, c, p, alpha, backend)
    if backend_norm in {"r", "rddensity", "r_reference", "r_package"}:
        return _rddensity_r_backend(data=data, x=x, c=c, p=p, h=h, alpha=alpha)
    if backend_norm not in {"native", "statspai"}:
        raise MethodIncompatibility(
            "backend must be 'native' or 'r'.",
            recovery_hint="Use backend='native' or backend='r'.",
            diagnostics={"backend": backend},
        )

    X = _running_values(data, x)
    n, n_l, n_r = _validate_support(X, c)

    X_c = X - c
    x_left = X_c[X_c < 0]
    x_right = X_c[X_c >= 0]

    h_l, h_r, h_source = _resolve_bandwidths(h, x_left, x_right, p, x_all=X_c)

    fV = _rddensity_fit(X_c, h_l=h_l, h_r=h_r, p=p)

    f_left = float(fV[0, 0])
    f_right = float(fV[1, 0])
    diff = float(fV[2, 0])
    se_diff = float(np.sqrt(fV[2, 1])) if np.isfinite(fV[2, 1]) else 0.0
    T_stat = diff / se_diff if se_diff > 0 else 0.0
    pvalue = float(2 * (1 - stats.norm.cdf(abs(T_stat))))

    z_crit = stats.norm.ppf(1 - alpha / 2)
    ci = (diff - z_crit * se_diff, diff + z_crit * se_diff)

    model_info = {
        "test": "Cattaneo-Jansson-Ma (2020)",
        "density_left": f_left,
        "density_right": f_right,
        "density_diff": diff,
        "bandwidth_left": h_l,
        "bandwidth_right": h_r,
        "bandwidth_source": h_source,
        "backend": "native",
        "validation_tier": "T2_native_reference_parity",
        "reference_backend": "rddensity",
        "validation_note": (
            "StatsPAI native rddensity mirrors the default rddensity "
            "unrestricted triangular-kernel path: rdbwdensity combination "
            "bandwidths, mass-point ECDF handling, and jackknife CJM "
            "local-polynomial density inference. backend='r' remains "
            "available for users who want to delegate directly to the R "
            "package."
        ),
        "polynomial_order": p,
        "n_left": n_l,
        "n_right": n_r,
        "cutoff": c,
    }

    _result = CausalResult(
        method="CJM (2020) Density Test",
        estimand="T-statistic (density discontinuity)",
        estimate=float(T_stat),
        se=float(se_diff),
        pvalue=pvalue,
        ci=ci,
        alpha=alpha,
        n_obs=n,
        model_info=model_info,
        _citation_key="rddensity",
    )
    try:
        from ..output._lineage import attach_provenance as _attach_prov

        _attach_prov(
            _result,
            function="sp.diagnostics.rddensity",
            params={
                "x": x,
                "c": c,
                "p": p,
                "h": h,
                "alpha": alpha,
                "backend": backend,
            },
            data=data,
            overwrite=False,
        )
    except Exception:  # pragma: no cover
        pass
    return _result


def _resolve_bandwidths(
    h: Optional[Union[float, Sequence[float]]],
    x_left: np.ndarray,
    x_right: np.ndarray,
    p: int,
    x_all: Optional[np.ndarray] = None,
) -> Tuple[float, float, str]:
    if h is None:
        if x_all is None:
            return _cjm_bandwidth(x_left, p), _cjm_bandwidth(x_right, p), "native_pilot"
        h_l, h_r = _rddensity_default_bandwidths(x_all, p)
        return h_l, h_r, "rddensity_comb"

    if np.isscalar(h):
        try:
            h_value = float(cast(Any, h))
        except (TypeError, ValueError) as exc:
            raise MethodIncompatibility(
                "h must be positive and finite.",
                recovery_hint="Pass h=None or a positive finite bandwidth.",
                diagnostics={"h": repr(h)},
            ) from exc
        if not np.isfinite(h_value) or h_value <= 0:
            raise MethodIncompatibility(
                "h must be positive and finite.",
                recovery_hint="Pass h=None or a positive finite bandwidth.",
                diagnostics={"h": h},
            )
        return h_value, h_value, "manual_scalar"

    try:
        h_values = list(cast(Sequence[float], h))
    except TypeError as exc:
        raise MethodIncompatibility(
            "h must be a scalar or a length-2 sequence (h_left, h_right).",
            recovery_hint="Pass h=0.5 or h=(0.35, 0.55).",
            diagnostics={"h": repr(h)},
        ) from exc
    if len(h_values) != 2:
        raise MethodIncompatibility(
            "h must be a scalar or a length-2 sequence (h_left, h_right).",
            recovery_hint="Pass h=0.5 or h=(0.35, 0.55).",
            diagnostics={"h_length": len(h_values)},
        )
    try:
        h_l, h_r = float(h_values[0]), float(h_values[1])
    except (TypeError, ValueError) as exc:
        raise MethodIncompatibility(
            "h_left and h_right must be positive and finite.",
            recovery_hint="Use positive finite side-specific bandwidths.",
            diagnostics={"h": [repr(v) for v in h_values]},
        ) from exc
    if not (np.isfinite(h_l) and np.isfinite(h_r)) or h_l <= 0 or h_r <= 0:
        raise MethodIncompatibility(
            "h_left and h_right must be positive and finite.",
            recovery_hint="Use positive finite side-specific bandwidths.",
            diagnostics={"h_left": h_l, "h_right": h_r},
        )
    return h_l, h_r, "manual_side_specific"


def _rddensity_r_backend(
    data: pd.DataFrame,
    x: str,
    c: float,
    p: int,
    h: Optional[Union[float, Sequence[float]]],
    alpha: float,
) -> CausalResult:
    rscript = _find_rscript()
    if rscript is None:
        raise ImportError("backend='r' requires Rscript and the R package rddensity.")

    X = _running_values(data, x)
    n, n_l, n_r = _validate_support(X, c)

    if h is None:
        h_l = h_r = None
        h_source = "rddensity_default"
    else:
        h_l, h_r, h_source = _resolve_bandwidths(h, X[X < c] - c, X[X >= c] - c, p)

    r_code = r"""
suppressPackageStartupMessages({
  if (!requireNamespace("rddensity", quietly = TRUE)) {
    stop("R package 'rddensity' is not installed")
  }
  if (!requireNamespace("jsonlite", quietly = TRUE)) {
    stop("R package 'jsonlite' is not installed")
  }
})
args <- commandArgs(trailingOnly = TRUE)
input <- args[[1]]
cutoff <- as.numeric(args[[2]])
p <- as.integer(args[[3]])
alpha <- as.numeric(args[[4]])
h_left <- as.numeric(args[[5]])
h_right <- as.numeric(args[[6]])
df <- utils::read.csv(input)
h_arg <- NULL
if (is.finite(h_left) && is.finite(h_right)) {
  h_arg <- c(h_left, h_right)
}
fit <- rddensity::rddensity(X = df$x, c = cutoff, p = p, h = h_arg)
density_left <- as.numeric(fit$hat$left)[1]
density_right <- as.numeric(fit$hat$right)[1]
density_diff <- density_right - density_left
pvalue <- as.numeric(fit$test$p_jk)[1]
zstat <- as.numeric(fit$test$t_jk)[1]
bw_left <- as.numeric(fit$h$left)[1]
bw_right <- as.numeric(fit$h$right)[1]
out <- list(
  density_left = density_left,
  density_right = density_right,
  density_diff = density_diff,
  pvalue = pvalue,
  zstat = zstat,
  bandwidth_left = bw_left,
  bandwidth_right = bw_right,
  polynomial_order = as.integer(fit$opt$p)
)
cat(jsonlite::toJSON(out, auto_unbox = TRUE, digits = 16, null = "null"))
"""

    with tempfile.TemporaryDirectory(prefix="statspai-rddensity-") as tmp:
        tmp_path = Path(tmp)
        csv_path = tmp_path / "x.csv"
        script_path = tmp_path / "run_rddensity.R"
        pd.DataFrame({"x": X}).to_csv(csv_path, index=False)
        script_path.write_text(r_code)
        h_l_arg = "NaN" if h_l is None else f"{h_l:.17g}"
        h_r_arg = "NaN" if h_r is None else f"{h_r:.17g}"
        proc = subprocess.run(
            [
                rscript,
                str(script_path),
                str(csv_path),
                f"{c:.17g}",
                str(int(p)),
                f"{alpha:.17g}",
                h_l_arg,
                h_r_arg,
            ],
            check=False,
            capture_output=True,
            text=True,
        )
    if proc.returncode != 0:
        raise ConvergenceFailure(
            "backend='r' failed while running rddensity::rddensity: "
            f"{proc.stderr.strip() or proc.stdout.strip()}",
            recovery_hint="Check the R rddensity/jsonlite installation and input data.",
            diagnostics={"returncode": int(proc.returncode)},
        )

    out: Dict[str, Any] = json.loads(proc.stdout)
    diff = float(out["density_diff"])
    zstat = float(out["zstat"])
    pvalue = float(out["pvalue"])
    se_diff = abs(diff / zstat) if np.isfinite(zstat) and abs(zstat) > 0 else 0.0
    z_crit = stats.norm.ppf(1 - alpha / 2)
    ci = (diff - z_crit * se_diff, diff + z_crit * se_diff)

    model_info = {
        "test": "Cattaneo-Jansson-Ma (2020)",
        "density_left": float(out["density_left"]),
        "density_right": float(out["density_right"]),
        "density_diff": diff,
        "bandwidth_left": float(out["bandwidth_left"]),
        "bandwidth_right": float(out["bandwidth_right"]),
        "bandwidth_source": h_source,
        "backend": "rddensity",
        "validation_tier": "reference_backend_bridge",
        "reference_backend": "rddensity",
        "validation_note": (
            "This result delegates to rddensity::rddensity. It is useful "
            "for exact reference-package numbers, but it is not counted as "
            "native Python parity evidence because the reference backend is "
            "the estimator itself."
        ),
        "polynomial_order": int(out["polynomial_order"]),
        "n_left": n_l,
        "n_right": n_r,
        "cutoff": c,
    }

    _result = CausalResult(
        method="CJM (2020) Density Test [rddensity::rddensity]",
        estimand="T-statistic (density discontinuity)",
        estimate=zstat,
        se=float(se_diff),
        pvalue=pvalue,
        ci=ci,
        alpha=alpha,
        n_obs=n,
        model_info=model_info,
        _citation_key="rddensity",
    )
    try:
        from ..output._lineage import attach_provenance as _attach_prov

        _attach_prov(
            _result,
            function="sp.diagnostics.rddensity",
            params={
                "x": x,
                "c": c,
                "p": p,
                "h": h,
                "alpha": alpha,
                "backend": "r",
            },
            data=data,
            overwrite=False,
        )
    except Exception:  # pragma: no cover
        pass
    return _result


def _find_rscript() -> Optional[str]:
    """Return an Rscript executable, including the standard macOS R path."""
    candidate = shutil.which("Rscript")
    if candidate:
        return candidate
    for path in (
        "/Library/Frameworks/R.framework/Resources/bin/Rscript",
        "/usr/local/bin/Rscript",
        "/opt/homebrew/bin/Rscript",
    ):
        if Path(path).exists():
            return path
    return None


def _cjm_bandwidth(x: np.ndarray, p: int) -> float:
    """Legacy fallback bandwidth used only when no full sample is available."""
    n = len(x)
    sd = np.std(x)
    h = 1.06 * sd * n ** (-1 / (2 * p + 3))
    return float(max(h, 0.01 * sd))


def _rddensity_unique(
    x: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Run-length encoding matching ``rddensity:::rddensityUnique``."""
    x = np.asarray(x, dtype=float)
    n = len(x)
    if n == 0:
        empty = np.array([], dtype=int)
        return np.array([], dtype=float), empty, empty, empty
    unique_vals: list[float] = []
    freqs: list[int] = []
    first: list[int] = []
    last: list[int] = []
    start = 0
    for idx in range(1, n + 1):
        if idx == n or x[idx] != x[idx - 1]:
            unique_vals.append(float(x[idx - 1]))
            freqs.append(idx - start)
            first.append(start)
            last.append(idx - 1)
            start = idx
    return (
        np.asarray(unique_vals, dtype=float),
        np.asarray(freqs, dtype=int),
        np.asarray(first, dtype=int),
        np.asarray(last, dtype=int),
    )


def _rddensity_h(x: float, p: int) -> float:
    """Probabilists' Hermite polynomial used by ``rdbwdensity``."""
    if p == 0:
        return 1.0
    if p == 1:
        return x
    if p == 2:
        return x**2 - 1
    if p == 3:
        return x**3 - 3 * x
    if p == 4:
        return x**4 - 6 * x**2 + 3
    if p == 5:
        return x**5 - 10 * x**3 + 15 * x
    if p == 6:
        return x**6 - 15 * x**4 + 45 * x**2 - 15
    if p == 7:
        return x**7 - 21 * x**5 + 105 * x**3 - 105 * x
    if p == 8:
        return x**8 - 28 * x**6 + 210 * x**4 - 420 * x**2 + 105
    if p == 9:
        return x**9 - 36 * x**7 + 378 * x**5 - 1260 * x**3 + 945 * x
    if p == 10:
        return x**10 - 45 * x**8 + 630 * x**6 - 3150 * x**4 + 4725 * x**2 - 945
    raise MethodIncompatibility(
        "p must be between 1 and 7 for rddensity defaults.",
        recovery_hint="Use p in the supported range 1..7.",
        diagnostics={"p": p},
    )


def _dnorm(x: float) -> float:
    return float(np.exp(-0.5 * x**2) / np.sqrt(2 * np.pi))


def _kernel_power_integral(power: int, low: float = 0.0, up: float = 1.0) -> float:
    """Triangular-kernel integral of x^power on [low, up]."""
    return (up ** (power + 1) - low ** (power + 1)) / (power + 1) - (
        up ** (power + 2) - low ** (power + 2)
    ) / (power + 2)


def _sgenerate(p: int) -> np.ndarray:
    out = np.zeros((p + 1, p + 1), dtype=float)
    for i in range(1, p + 2):
        for j in range(1, p + 2):
            out[i - 1, j - 1] = _kernel_power_integral(i + j - 2)
    return out


def _cgenerate(k: int, p: int) -> np.ndarray:
    out = np.zeros((p + 1, 1), dtype=float)
    for i in range(1, p + 2):
        out[i - 1, 0] = _kernel_power_integral(i + k - 1)
    return out


def _order_stat(x: np.ndarray, k: int) -> float:
    x_sorted = np.sort(np.asarray(x, dtype=float))
    return float(x_sorted[min(len(x_sorted), k) - 1])


def _rddensity_ecdf_y(n: int, freq: np.ndarray, index_last: np.ndarray) -> np.ndarray:
    y = np.arange(n, dtype=float) / (n - 1)
    return np.repeat(y[index_last], freq)


def _rddensity_default_bandwidths(
    x_centered: np.ndarray, p: int
) -> Tuple[float, float]:
    """Port of ``rdbwdensity(..., bwselect='comb')`` for the default path."""
    x = np.sort(np.asarray(x_centered, dtype=float))
    n = len(x)
    n_l = int(np.sum(x < 0))
    n_r = int(np.sum(x >= 0))
    x_unique, freq, _index_first, index_last = _rddensity_unique(x)
    n_l_unique = int(np.sum(x_unique < 0))
    n_r_unique = int(np.sum(x_unique >= 0))

    x_mu = float(np.mean(x))
    x_sd = float(np.std(x, ddof=1))
    z = x_mu / x_sd
    cb = np.array(
        [
            25884.4444444942,
            3430865.45512362,
            845007948.042626,
            330631733667.038,
            187774809656037,
            145729502641999264,
            1.4601350297445e20,
        ]
    )
    cc = np.array(
        [
            4.80000000000002,
            548.571428571555,
            100800.000000204,
            29558225.4581006,
            12896196859.6126,
            7890871468221.61,
            6467911284037581.0,
        ]
    )
    fhat_b = 1.0 / (_rddensity_h(z, p + 2) ** 2 * _dnorm(z))
    fhat_c = 1.0 / (_rddensity_h(z, p) ** 2 * _dnorm(z))
    b_n = (((2 * p + 1) / 4) * fhat_b * cb[p - 1] / n) ** (1 / (2 * p + 5)) * x_sd
    c_n = ((1 / (2 * p)) * fhat_c * cc[p - 1] / n) ** (1 / (2 * p + 1)) * x_sd

    b_n = min(b_n, float(np.max(np.abs(x_unique))))
    c_n = min(c_n, float(np.max(np.abs(x_unique))))
    b_n = max(
        b_n,
        _order_stat(np.abs(x[x < 0]), 20 + p + 3),
        _order_stat(x[x >= 0], 20 + p + 3),
        _order_stat(np.abs(x_unique[x_unique < 0]), 20 + p + 3),
        _order_stat(x_unique[x_unique >= 0], 20 + p + 3),
    )
    c_n = max(
        c_n,
        _order_stat(np.abs(x[x < 0]), 20 + p + 1),
        _order_stat(x[x >= 0], 20 + p + 1),
        _order_stat(np.abs(x_unique[x_unique < 0]), 20 + p + 1),
        _order_stat(x_unique[x_unique >= 0], 20 + p + 1),
    )

    y = _rddensity_ecdf_y(n, freq, index_last)
    mask_b = np.abs(x) <= b_n
    mask_c = np.abs(x) <= c_n
    x_b, y_b = x[mask_b], y[mask_b]
    x_c, y_c = x[mask_c], y[mask_c]
    fv_b = _rddensity_fv(
        y_b,
        x_b,
        n_l,
        n_r,
        int(np.sum(x_b < 0)),
        int(np.sum(x_b >= 0)),
        b_n,
        b_n,
        p + 2,
        p + 1,
    )
    fv_c = _rddensity_fv(
        y_c,
        x_c,
        n_l,
        n_r,
        int(np.sum(x_c < 0)),
        int(np.sum(x_c >= 0)),
        c_n,
        c_n,
        p,
        1,
    )

    hn = np.full((4, 3), np.nan, dtype=float)
    hn[:, 1] = n * c_n * fv_c[:, 1]
    s_mat = _sgenerate(p)
    c_vec = _cgenerate(p + 1, p)
    bias_factor = np.linalg.solve(s_mat, c_vec).ravel()[1]
    hn[0, 2] = fv_b[0, 3] * bias_factor * ((-1) ** p)
    hn[1, 2] = fv_b[1, 3] * bias_factor
    hn[2, 2] = hn[1, 2] - hn[0, 2]
    hn[3, 2] = hn[1, 2] + hn[0, 2]
    hn[:, 2] = hn[:, 2] ** 2
    hn[:, 0] = ((1 / (2 * p)) * hn[:, 1] / hn[:, 2] / n) ** (1 / (2 * p + 1))
    for i in range(4):
        if hn[i, 1] < 0:
            hn[i, 0] = 0.0
            hn[i, 1] = np.nan
        if not np.isfinite(hn[i, 0]):
            hn[i, 0] = 0.0

    hn[0, 0] = min(hn[0, 0], abs(float(x_unique[0])))
    hn[1, 0] = min(hn[1, 0], float(x_unique[-1]))
    hn[2, 0] = min(hn[2, 0], max(abs(float(x_unique[0])), float(x_unique[-1])))
    hn[3, 0] = min(hn[3, 0], max(abs(float(x_unique[0])), float(x_unique[-1])))
    n_min = 20 + p + 1
    h_l_min = _order_stat(np.abs(x[x < 0]), min(n_l, n_min))
    h_r_min = _order_stat(x[x >= 0], min(n_r, n_min))
    hn[0, 0] = max(hn[0, 0], h_l_min)
    hn[1, 0] = max(hn[1, 0], h_r_min)
    hn[2, 0] = max(hn[2, 0], h_l_min, h_r_min)
    hn[3, 0] = max(hn[3, 0], h_l_min, h_r_min)
    h_l_min = _order_stat(np.abs(x_unique[x_unique < 0]), min(n_l_unique, n_min))
    h_r_min = _order_stat(x_unique[x_unique >= 0], min(n_r_unique, n_min))
    hn[0, 0] = max(hn[0, 0], h_l_min)
    hn[1, 0] = max(hn[1, 0], h_r_min)
    hn[2, 0] = max(hn[2, 0], h_l_min, h_r_min)
    hn[3, 0] = max(hn[3, 0], h_l_min, h_r_min)

    h_left = float(np.median([hn[0, 0], hn[2, 0], hn[3, 0]]))
    h_right = float(np.median([hn[1, 0], hn[2, 0], hn[3, 0]]))
    return h_left, h_right


def _rddensity_fit(
    x_centered: np.ndarray, *, h_l: float, h_r: float, p: int
) -> np.ndarray:
    """Default ``rddensity`` density/test fit with q=p+1."""
    x = np.sort(np.asarray(x_centered, dtype=float))
    n = len(x)
    n_l = int(np.sum(x < 0))
    n_r = int(np.sum(x >= 0))
    _x_unique, freq, _index_first, index_last = _rddensity_unique(x)
    y = _rddensity_ecdf_y(n, freq, index_last)
    mask = (x >= -h_l) & (x <= h_r)
    x_h = x[mask]
    y_h = y[mask]
    return _rddensity_fv(
        y_h,
        x_h,
        n_l,
        n_r,
        int(np.sum(x_h < 0)),
        int(np.sum(x_h >= 0)),
        h_l,
        h_r,
        p + 1,
        1,
    )


def _rddensity_fv(
    y: np.ndarray,
    x: np.ndarray,
    n_l: int,
    n_r: int,
    n_lh: int,
    n_rh: int,
    h_l: float,
    h_r: float,
    p: int,
    s: int,
) -> np.ndarray:
    """Port of ``rddensity:::rddensity_fV`` for the default unrestricted path."""
    n = n_l + n_r
    n_h = n_lh + n_rh
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float).reshape(-1, 1)
    weights = np.empty(n_h, dtype=float)
    weights[:n_lh] = (1 + x[:n_lh] / h_l) / h_l
    weights[n_lh:] = (1 - x[n_lh:] / h_r) / h_r

    xp = np.empty((n_h, 2 * p + 2), dtype=float)
    hp = np.empty(2 * p + 2, dtype=float)
    for j in range(1, 2 * p + 3):
        col = j - 1
        if j % 2:
            power = (j - 1) // 2
            xp[:n_lh, col] = (x[:n_lh] / h_l) ** power
            xp[n_lh:, col] = 0.0
            hp[col] = h_l**power
        else:
            power = (j - 2) // 2
            xp[:n_lh, col] = 0.0
            xp[n_lh:, col] = (x[n_lh:] / h_r) ** power
            hp[col] = h_r**power

    out = np.full((4, 4), np.nan, dtype=float)
    xp_w = xp * weights[:, None]
    try:
        s_inv = np.linalg.inv(xp_w.T @ xp)
    except np.linalg.LinAlgError:
        return out
    hp_inv = np.diag(1 / hp)
    coef = hp_inv @ s_inv @ (xp_w.T @ y)
    out[0, 0] = coef[2, 0]
    out[1, 0] = coef[3, 0]
    out[2, 0] = coef[3, 0] - coef[2, 0]
    out[3, 0] = coef[3, 0] + coef[2, 0]
    out[0, 3] = coef[2 * s, 0]
    out[1, 3] = coef[2 * s + 1, 0]
    out[2, 3] = out[1, 3] - out[0, 3]
    out[3, 3] = out[1, 3] + out[0, 3]

    leverage = np.zeros((n_h, xp.shape[1]), dtype=float)
    x_unique, freq, index_first, _index_last = _rddensity_unique(x)
    del x_unique
    for col in range(xp.shape[1]):
        rev = xp_w[::-1, col]
        cumsum = np.cumsum(np.r_[0.0, rev]) / (n - 1)
        values = cumsum[:n_h][::-1]
        leverage[:, col] = np.repeat(values[index_first], freq)
    v_mat = hp_inv @ s_inv @ (leverage.T @ leverage) @ s_inv @ hp_inv
    out[0, 1] = v_mat[2, 2]
    out[1, 1] = v_mat[3, 3]
    out[2, 1] = v_mat[2, 2] + v_mat[3, 3] - 2 * v_mat[2, 3]
    out[3, 1] = v_mat[2, 2] + v_mat[3, 3] + 2 * v_mat[2, 3]
    for i in range(4):
        if np.isfinite(out[i, 1]) and out[i, 1] < 0:
            out[i, 1] = np.nan
    return out


# Citation
CausalResult._CITATIONS["rddensity"] = (
    "@article{cattaneo2020simple,\n"
    "  title={Simple Local Polynomial Density Estimators},\n"
    "  author={Cattaneo, Matias D. and Jansson, Michael and Ma, Xinwei},\n"
    "  journal={Journal of the American Statistical Association},\n"
    "  volume={115},\n"
    "  number={531},\n"
    "  pages={1449--1455},\n"
    "  year={2020},\n"
    "  publisher={Taylor \\& Francis}\n"
    "}"
)
