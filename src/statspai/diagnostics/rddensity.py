"""
Cattaneo-Jansson-Ma density discontinuity (manipulation) test.

Tests whether the density of the running variable is continuous at the
RD cutoff, the modern replacement for McCrary (2008). Each side's density
at the cutoff is the slope of a local polynomial fitted to the empirical
distribution function (cattaneo2020simple); the default test is the
robust bias-corrected statistic with jackknife standard errors and the
``comb`` data-driven bandwidths (cattaneo2018manipulation).

Clean-room rewrite of the native path: written from the two papers, the
published help pages and black-box runs of R ``rddensity`` 2.6 (GPL-2),
whose source was not read. The shared estimator primitives live in
:mod:`statspai.rd._lpdensity`; the conventions pinned by black-box
probing are recorded in ``docs/dev/cleanroom/rddensity.md``.
``backend='r'`` still delegates to the R package when it is installed.

References
----------
cattaneo2020simple
cattaneo2018manipulation
"""

import json
import shutil
import subprocess
import tempfile
from math import factorial
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

from .._aliases import accepts_aliases
from .._input_validation import check_columns
from ..core.results import CausalResult
from ..exceptions import ConvergenceFailure, DataInsufficient, MethodIncompatibility
from ..rd._lpdensity import (
    _cdf_counts,
    boundary_bias_constant,
    boundary_fit,
    kth_distance,
    normal_reference_bandwidth,
)

#: ``estimate`` / ``pvalue`` are the CJM t-test (what rddensity and Stata
#: report); ``se`` / ``ci`` are for the density difference f(c+) - f(c-),
#: whose value is ``model_info['density_diff']``. Read together,
#: ``estimate +/- z * se`` is not the interval, so both say so.
_ESTIMAND = (
    "T-statistic (density discontinuity); se and ci are on the density-"
    "difference scale (model_info['density_diff'])"
)
_SCALE = {
    "estimate": "t_statistic",
    "pvalue": "t_statistic",
    "se": "density_diff",
    "ci": "density_diff",
}


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
    check_columns(data, {"x": x}, function="sp.rddensity", extra_diagnostics={"x": x})
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


_NATIVE_NOTE = (
    "StatsPAI native rddensity mirrors the default rddensity unrestricted "
    "triangular-kernel path: rdbwdensity combination bandwidths, mass-point "
    "ECDF handling, and jackknife CJM local-polynomial density inference. "
    "backend='r' remains available for users who want to delegate directly "
    "to the R package."
)


@accepts_aliases(_strict=True, running="x", cutoff="c")
def rddensity(
    data: pd.DataFrame,
    x: str,
    c: float = 0,
    p: int = 2,
    h: Optional[Union[float, Sequence[float]]] = None,
    alpha: float = 0.05,
    backend: str = "native",
    *,
    bino_w: Optional[Union[float, Sequence[float]]] = None,
    bino_wstep: Optional[Union[float, Sequence[float]]] = None,
    bino_n: int = 20,
    bino_nw: int = 10,
    bino_p: float = 0.5,
) -> CausalResult:
    """CJM (2020) density discontinuity test for RD manipulation.

    Estimates the density of the running variable just below and just
    above the cutoff with local polynomials fitted to the empirical CDF
    and tests whether the two limits are equal. The reported statistic is
    the robust bias-corrected one: densities of order ``p + 1`` evaluated
    with bandwidths chosen for order ``p``, studentised with jackknife
    standard errors (the defaults of R/Stata ``rddensity``).

    Parameters
    ----------
    data : pandas.DataFrame
        Data containing the running variable.
    x : str
        Running-variable column (alias ``running``).
    c : float, default 0
        Cutoff (alias ``cutoff``).
    p : int, default 2
        Local polynomial order of the density estimator (1 to 7).
    h : float or (float, float), optional
        Bandwidth, or ``(h_left, h_right)``. Default: the ``comb``
        data-driven bandwidths (median of the per-side, difference and sum
        MSE-optimal choices on each side).
    alpha : float, default 0.05
        Level of ``ci`` (on the density-difference scale).
    backend : {'native', 'r'}, default 'native'
        ``'r'`` runs ``rddensity::rddensity`` through ``Rscript``.
    bino_w : float or (float, float), optional
        Half-width(s) of the first binomial-test window (below / above).
        Default: the distance to the ``bino_n``-th closest observation.
    bino_wstep : float or (float, float), optional
        Increment of the half-widths; default the first half-width(s).
    bino_n : int, default 20
        Observations in the first window when ``bino_w`` is not given.
    bino_nw : int, default 10
        Number of windows.
    bino_p : float, default 0.5
        Null probability of falling below the cutoff.

    Returns
    -------
    CausalResult
        ``estimate`` and ``pvalue`` are the robust T statistic and its
        p-value; ``se`` and ``ci`` are for the density difference
        ``f(c+) - f(c-)`` (``model_info['density_diff']``). ``model_info``
        also holds the one-sided densities and SEs, the bandwidths, the
        effective sample sizes, the order-``p`` (conventional) test and
        the exact binomial tests.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> df = pd.DataFrame({"x": rng.normal(size=2000)})
    >>> res = sp.rddensity(df, x="x", c=0)
    >>> bool(res.pvalue > 0.01)
    True

    References
    ----------
    cattaneo2020simple
    cattaneo2018manipulation
    """
    backend_key = _validate_inputs(data, x, c, p, alpha, backend)
    if backend_key in {"r", "rddensity"}:
        return _rddensity_r_backend(data, x, c, p, h, alpha)
    if backend_key != "native":
        raise MethodIncompatibility(
            "backend must be 'native' or 'r'.",
            recovery_hint="Use backend='native' or backend='r'.",
            diagnostics={"backend": backend},
        )

    X = _running_values(data, x)
    n, n_l, n_r = _validate_support(X, c)
    xs = np.sort(X, kind="stable")
    dist = xs - c
    h_l, h_r, h_source = _resolve_bandwidths(
        h, dist[dist < 0], dist[dist >= 0], p, x_all=dist
    )
    bino = _binomial_tests(
        dist, bino_n, bino_nw, w=bino_w, wstep=bino_wstep, prob=bino_p
    )

    counts = _cdf_counts(xs, mass_points=True) - 1.0  # leave-one-out
    robust = _side_pair(xs, counts, c, h_l, h_r, p + 1)
    conv = _side_pair(xs, counts, c, h_l, h_r, p)

    diff = robust["right"] - robust["left"]
    se_diff = float(np.sqrt(robust["se_left"] ** 2 + robust["se_right"] ** 2))
    if np.isfinite(diff) and np.isfinite(se_diff) and se_diff > 0:
        t_stat = diff / se_diff
        pvalue = float(2 * stats.norm.sf(abs(t_stat)))
    else:  # statistic not computable on these windows
        t_stat, se_diff, pvalue = 0.0, 0.0, 1.0
    z_crit = stats.norm.ppf(1 - alpha / 2)
    ci = (diff - z_crit * se_diff, diff + z_crit * se_diff)

    conv_diff = conv["right"] - conv["left"]
    conv_se = float(np.sqrt(conv["se_left"] ** 2 + conv["se_right"] ** 2))
    conv_t = conv_diff / conv_se if conv_se > 0 else np.nan
    conventional = {
        "density_left": conv["left"],
        "density_right": conv["right"],
        "se_left": conv["se_left"],
        "se_right": conv["se_right"],
        "se": conv_se,
        "statistic": float(conv_t),
        "pvalue": float(2 * stats.norm.sf(abs(conv_t))),
    }

    model_info = {
        "test": "Cattaneo-Jansson-Ma (2020)",
        "density_left": robust["left"],
        "density_right": robust["right"],
        "density_diff": float(diff),
        "se_left": robust["se_left"],
        "se_right": robust["se_right"],
        "conventional": conventional,
        "binomial_tests": bino,
        "bandwidth_left": float(h_l),
        "bandwidth_right": float(h_r),
        "bandwidth_source": h_source,
        "backend": "native",
        "validation_tier": "T2_native_reference_parity",
        "reference_backend": "rddensity",
        "validation_note": _NATIVE_NOTE,
        "polynomial_order": int(p),
        "n_left": n_l,
        "n_right": n_r,
        "n_eff_left": robust["n_left"],
        "n_eff_right": robust["n_right"],
        "cutoff": c,
        "scale": dict(_SCALE),
    }
    result = CausalResult(
        method="CJM (2020) Density Test",
        estimand=_ESTIMAND,
        estimate=float(t_stat),
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
            result,
            function="sp.diagnostics.rddensity",
            params={
                "x": x,
                "c": c,
                "p": p,
                "h": h,
                "alpha": alpha,
                "backend": "native",
            },
            data=data,
            overwrite=False,
        )
    except Exception:  # pragma: no cover - provenance is best effort
        pass
    return result


def _side_pair(
    xs: np.ndarray, counts: np.ndarray, c: float, h_l: float, h_r: float, order: int
) -> Dict[str, Any]:
    """Both one-sided density fits of ``order`` at the cutoff."""
    out: Dict[str, Any] = {}
    for side, h in (("left", h_l), ("right", h_r)):
        est, se, n_win = boundary_fit(xs, counts, xs, c, h, side, order)
        out[side], out[f"se_{side}"], out[f"n_{side}"] = est, se, n_win
    return out


def _resolve_bandwidths(
    h: Optional[Union[float, Sequence[float]]],
    x_left: np.ndarray,
    x_right: np.ndarray,
    p: int,
    x_all: Optional[np.ndarray] = None,
) -> Tuple[float, float, str]:
    """Validate a user bandwidth or compute the default ``comb`` pair.

    ``x_left`` / ``x_right`` / ``x_all`` are centred at the cutoff.
    """
    if h is None:
        if x_all is None:
            x_all = np.concatenate([np.asarray(x_left), np.asarray(x_right)])
        h_l, h_r = _default_bandwidths(np.sort(np.asarray(x_all, float)), p)
        return h_l, h_r, "rddensity_comb"
    if np.ndim(h) == 0:
        values = [h]
    else:
        values = list(np.ravel(np.asarray(h, dtype=object)))
        if len(values) != 2:
            raise MethodIncompatibility(
                "h must be a scalar or a length-2 sequence (h_left, h_right).",
                recovery_hint="Pass h=0.5 or h=(0.35, 0.55).",
                diagnostics={"len_h": len(values)},
            )
    try:
        bws = [float(v) for v in values]
    except (TypeError, ValueError):
        bws = [np.nan]
    if not all(np.isfinite(b) and b > 0 for b in bws):
        raise MethodIncompatibility(
            "h must be positive and finite.",
            recovery_hint="Pass h=None or a positive finite bandwidth.",
            diagnostics={"h": repr(h)},
        )
    if len(bws) == 1:
        return bws[0], bws[0], "manual_scalar"
    return bws[0], bws[1], "manual_side_specific"


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
        estimand=_ESTIMAND,
        estimate=zstat,
        se=float(se_diff),
        pvalue=pvalue,
        ci=ci,
        alpha=alpha,
        n_obs=n,
        model_info={**model_info, "scale": dict(_SCALE)},
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


def _regularize(h: float, sides: Sequence[np.ndarray], k: int) -> float:
    """Widen ``h`` to hold ``k`` observations and ``k`` distinct values on
    every side in ``sides``, then cap it at the farthest observation.

    ``sides`` are sorted absolute distances to the cutoff; the default
    ``k = 20 + p + 1`` follows the rdbwdensity help (nLocalMin /
    nUniqueMin), and a side with fewer than ``k`` observations contributes
    its farthest one. The cap, and the rule that a bandwidth the MSE
    formula cannot produce (NaN: bias not estimable) takes the lower
    bound, were found by black-box probing.
    """
    lower = 0.0
    for d in sides:
        if d.size:
            lower = max(lower, kth_distance(d, k), kth_distance(np.unique(d), k))
    h = lower if np.isnan(h) else max(h, lower)
    far = max(float(d[-1]) for d in sides if d.size)
    return min(h, far)


def _default_bandwidths(x_centered: np.ndarray, p: int) -> Tuple[float, float]:
    """``comb`` bandwidths: ``median(h_side, h_diff, h_sum)`` on each side."""
    table = _bandwidth_table(x_centered, p)
    pair = [table[k][0] for k in ("diff", "sum")]
    return (
        float(np.median([table["l"][0]] + pair)),
        float(np.median([table["r"][0]] + pair)),
    )


def _bandwidth_table(
    x_centered: np.ndarray, p: int
) -> Dict[str, Tuple[float, float, float]]:
    """MSE-optimal bandwidths of cattaneo2018manipulation (section 2.5).

    For each target (left density, right density, their difference, their
    sum) the MSE-optimal bandwidth is ``(V / (2p B^2 n))^(1/(2p+1))``:

    * ``V`` = ``n h1 se^2`` with ``se`` the jackknife SE of the order-``p``
      fit at a normal-reference pilot ``h1`` (variance of the difference
      and of the sum: ``V_left + V_right``);
    * ``B = F^(p+1)(c)/(p+1)! * e_1' S^-1 c`` with the triangular-kernel
      boundary constant and ``F^(p+1)`` from an order-``p+2`` fit at a
      second normal-reference pilot (``B_diff = B_r - B_l``,
      ``B_sum = B_r + B_l``).

    Returns ``{target: (h, V, B^2)}`` for ``l``, ``r``, ``diff``, ``sum``
    (the layout of rdbwdensity's table); ``x_centered`` is sorted.
    """
    n = x_centered.size
    xs = np.asarray(x_centered, float)
    counts = _cdf_counts(xs, mass_points=True) - 1.0
    d_left = np.sort(-xs[xs < 0])
    d_right = xs[xs >= 0]
    both = (d_left, d_right)
    k = 20 + p + 1

    h_var = _regularize(normal_reference_bandwidth(xs, 0.0, p, 1), both, k)
    h_der = _regularize(normal_reference_bandwidth(xs, 0.0, p + 2, p + 1), both, k)
    V: Dict[str, float] = {}
    B: Dict[str, float] = {}
    for side in ("left", "right"):
        _, se, _ = boundary_fit(xs, counts, xs, 0.0, h_var, side, p)
        V[side] = n * h_var * se**2
        deriv, _, _ = boundary_fit(
            xs, counts, xs, 0.0, h_der, side, p + 2, v=p + 1, with_se=False
        )
        B[side] = deriv / factorial(p + 1) * boundary_bias_constant(p, side)
    if not (np.isfinite(B["left"]) and np.isfinite(B["right"])):
        # rdbwdensity reports no bias constant for either side when one
        # side's derivative fit is not identified (black-box finding)
        B = {"left": np.nan, "right": np.nan}

    def mse_h(var: float, bias: float) -> float:
        with np.errstate(divide="ignore", invalid="ignore"):
            return float((var / (2 * p * bias**2 * n)) ** (1.0 / (2 * p + 1)))

    v_two = V["left"] + V["right"]
    cells = {
        "l": (V["left"], B["left"], (d_left,)),
        "r": (V["right"], B["right"], (d_right,)),
        "diff": (v_two, B["right"] - B["left"], both),
        "sum": (v_two, B["right"] + B["left"], both),
    }
    return {
        key: (_regularize(mse_h(var, bias), sides, k), var, bias**2)
        for key, (var, bias, sides) in cells.items()
    }


def _window_spec(
    value: Optional[Union[float, Sequence[float]]], name: str
) -> Optional[Tuple[float, float]]:
    if value is None:
        return None
    vals = np.atleast_1d(np.asarray(value, dtype=float)).ravel()
    if vals.size not in (1, 2) or not np.all(np.isfinite(vals)) or np.any(vals <= 0):
        raise MethodIncompatibility(
            f"{name} must be one positive half-width, or two (below and "
            "above the cutoff).",
            recovery_hint=f"Pass {name}=0.5 or {name}=(0.5, 1.0).",
            diagnostics={name: repr(value)},
        )
    return (float(vals[0]), float(vals[-1]))


def _binomial_tests(
    x_centered: np.ndarray,
    n_first: int = 20,
    n_windows: int = 10,
    *,
    w: Optional[Union[float, Sequence[float]]] = None,
    wstep: Optional[Union[float, Sequence[float]]] = None,
    prob: float = 0.5,
) -> pd.DataFrame:
    """Exact binomial tests of the split of observations around the cutoff.

    Follows the Stata ``rddensity`` help: the first symmetric window holds
    ``n_first`` observations (half-width = distance to the ``n_first``-th
    closest observation) and its half-width is also the increment, unless
    ``w`` / ``wstep`` are given (one value, or one per side). Window ``k``
    counts ``c - w_l <= x < c`` and ``c <= x <= c + w_r`` and tests
    ``P(below) = prob`` (cattaneo2018manipulation's companion test).
    ``half_width`` is the below-cutoff half-width; a ``half_width_right``
    column is added when the two sides differ.
    """
    cols = ["half_width", "n_left", "n_right", "pvalue"]
    xc = np.asarray(x_centered, dtype=float)
    first = _window_spec(w, "bino_w")
    step = _window_spec(wstep, "bino_wstep")
    if first is None:
        if xc.size < n_first:
            return pd.DataFrame(columns=cols)
        w0 = kth_distance(np.sort(np.abs(xc)), n_first)
        first = (w0, w0)
    if step is None:
        step = first
    rows = []
    for k in range(int(n_windows)):
        w_l = first[0] + k * step[0]
        w_r = first[1] + k * step[1]
        n_left = int(np.sum((xc < 0) & (xc >= -w_l)))
        n_right = int(np.sum((xc >= 0) & (xc <= w_r)))
        total = n_left + n_right
        pval = stats.binomtest(n_left, total, prob).pvalue if total > 0 else np.nan
        rows.append((w_l, w_r, n_left, n_right, float(pval)))
    table = pd.DataFrame(rows, columns=["half_width", "half_width_right"] + cols[1:])
    if (table["half_width"] == table["half_width_right"]).all():
        table = table.drop(columns="half_width_right")
    return table


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
