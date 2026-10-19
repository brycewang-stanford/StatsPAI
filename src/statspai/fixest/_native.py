"""StatsPAI's own engine behind ``sp.feols`` / ``sp.fepois`` (``backend='auto'``).

``sp.feols`` and ``sp.fepois`` reached R ``fixest``'s numbers through
pyfixest, an optional dependency, and paid its fixed cost on every call.
The native fit (``sp.fast.feols`` / ``sp.fast.fepois`` on the optional
Rust kernels or their NumPy fallback) computes the same estimator with
fixest's small-sample conventions. This module decides, per call, whether
the native fit can answer it exactly as the delegate would, runs it, and
reports it in the delegate's result layout.

A call is native when every part of it is something the native fit
implements: ``lhs ~ x1 + ... | fe1 + ...`` with numeric regressors and
plain fixed-effect columns, ``vcov`` in iid / hetero / one-way CRV1, no
``ssc=``, no extra pyfixest keywords. Rows with a missing value in a used
column are dropped first, as fixest does. Anything else, and a design
whose regressors are absorbed or collinear (fixest drops those columns;
the native fit refuses them), goes to pyfixest; :func:`plan` says why.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from ..core.results import EconometricResults

_NAME = re.compile(r"[A-Za-z_][A-Za-z0-9_]*\Z")
# fixest's i(factor, ref=v) / i(factor, var, ref=v), with an explicit
# scalar reference level (event-study dummies).
_I_TERM = re.compile(
    r"i\(\s*([A-Za-z_]\w*)\s*(?:,\s*([A-Za-z_]\w*)\s*)?,\s*ref\s*=\s*"
    r"(-?\d+|\"[^\"]*\"|'[^']*')\s*\)\Z"
)


@dataclass
class ITerm:
    """An ``i(factor[, var], ref=...)`` term of the formula."""

    factor: str
    var: Optional[str]
    ref: str


_HETERO = ("hetero", "hc1", "robust")


@dataclass
class Plan:
    """A call the native fit can answer, in its own terms."""

    lhs: str
    rhs: List[str]
    fe: List[str]
    vcov: str  # 'iid' | 'hc1' | 'cr1'
    cluster: Optional[Union[str, List[str]]]  # a list: two-way clustering
    weights: Optional[str]
    drop_singletons: bool
    data: pd.DataFrame
    n_missing: int
    #: internal dummy-column name -> reported name (``rel::-3``)
    renames: Dict[str, str] = field(default_factory=dict)
    #: IV: endogenous regressors and their excluded instruments
    endog: List[str] = field(default_factory=list)
    instr: List[str] = field(default_factory=list)

    @property
    def formula(self) -> str:
        rhs = " + ".join(self.rhs) if self.rhs else "1"
        if self.fe:
            return f"{self.lhs} ~ {rhs} | {' + '.join(self.fe)}"
        return f"{self.lhs} ~ 1 + {rhs}" if self.rhs else f"{self.lhs} ~ 1"

    @property
    def vcov_label(self) -> str:
        return {"iid": "iid", "hc1": "hetero", "cr1": "CRV"}[self.vcov]

    @property
    def vcov_detail(self) -> str:
        return {"iid": "iid", "hc1": "hetero", "cr1": "CRV1"}[self.vcov]


def _split_formula(
    fml: Any,
) -> Optional[Tuple[str, List[Union[str, ITerm]], List[str]]]:
    if not isinstance(fml, str) or fml.count("~") != 1 or fml.count("|") > 1:
        return None
    lhs, rest = (s.strip() for s in fml.split("~"))
    rhs_s, _, fe_s = rest.partition("|")
    tokens = [t.strip() for t in rhs_s.split("+")]
    fe = [t.strip() for t in fe_s.split("+")] if fe_s.strip() else []
    if not _NAME.match(lhs) or any(not t for t in tokens + fe):
        return None
    if fe_s and not fe_s.strip():
        return None
    rhs: List[Union[str, ITerm]] = []
    for t in tokens:
        if t == "1":
            continue
        m = _I_TERM.match(t)
        if m:
            rhs.append(ITerm(m.group(1), m.group(2), m.group(3).strip("\"'")))
        elif _NAME.match(t):
            rhs.append(t)
        else:
            return None
    if not rhs or any(not _NAME.match(t) for t in fe):
        return None
    plain = [t for t in rhs if isinstance(t, str)]
    names = [lhs] + plain + fe
    if len(set(names)) != len(names):
        return None
    return lhs, rhs, fe


_MULTI = re.compile(r"\b(sw|csw|sw0|csw0|mvsw)\b\(.+\)")


def _split_top(text: str, sep: str) -> List[str]:
    """Split on ``sep`` outside parentheses."""
    out: List[str] = []
    depth = 0
    cur: List[str] = []
    for ch in text:
        if ch == "(":
            depth += 1
        elif ch == ")":
            depth -= 1
        elif ch == sep and depth == 0:
            out.append("".join(cur).strip())
            cur = []
            continue
        cur.append(ch)
    out.append("".join(cur).strip())
    return out


def _expand_first(formula: str) -> Optional[List[str]]:
    m = _MULTI.search(formula)
    if not m:
        return None
    kind = m.group(1)
    start = formula.index("(", m.start()) + 1
    depth, pos = 1, start
    while depth:
        if formula[pos] == "(":
            depth += 1
        elif formula[pos] == ")":
            depth -= 1
        pos += 1
    args = _split_top(formula[start : pos - 1], ",")
    if kind == "mvsw":
        import itertools

        args = [
            " + ".join(c)
            for r in range(1, len(args) + 1)
            for c in itertools.combinations(args, r)
        ]
    elif kind in ("csw", "csw0"):
        args = [" + ".join(args[: i + 1]) for i in range(len(args))]
    if kind in ("sw0", "csw0", "mvsw"):
        args = ["1", *args]
    call = formula[m.start() : pos]
    return [formula.replace(call, a) for a in args]


def expand_multiple(fml: Any) -> Optional[List[str]]:
    """The models of a multiple-estimation formula, in the delegate's order.

    Several outcomes (``y1 + y2 ~ ...``) and ``sw`` / ``sw0`` / ``csw`` /
    ``csw0`` / ``mvsw`` expand left to right, depth first, as pyfixest (and
    fixest) do; the models are then grouped by their fixed-effects part in
    order of first appearance, the order pyfixest fits and lists them in.
    A fixed-effects part of ``0`` / ``1`` is dropped. ``None`` without
    multiple-estimation syntax.
    """
    if not isinstance(fml, str) or "~" not in fml:
        return None
    lhs, rhs = (t.strip() for t in fml.split("~", 1))
    deps = _split_top(lhs, "+")
    formula = f"sw({', '.join(deps)}) ~ {rhs}" if len(deps) > 1 else fml
    if len(deps) == 1 and not _MULTI.search(formula):
        return None

    def expand(f: str) -> List[str]:
        first = _expand_first(f)
        return [f] if first is None else [g for h in first for g in expand(h)]

    groups: Dict[Optional[str], List[str]] = {}
    for f in expand(formula):
        parts = [t.strip() for t in f.split("|")]
        fe = next((t for t in parts[1:] if "~" not in t), None)
        if fe in ("0", "1"):
            parts = [t for t in parts if t not in ("0", "1")]
            fe = None
        groups.setdefault(fe, []).append(" | ".join(parts))
    return [f for fs in groups.values() for f in fs]


def _split_iv(
    fml: Any,
) -> Optional[Tuple[str, List[str], List[str], List[str], List[str]]]:
    """``y ~ x | fe | d ~ z`` -> (lhs, exog, fe, endog, instruments)."""
    if not isinstance(fml, str) or fml.count("|") != 2 or fml.count("~") != 2:
        return None
    head, fe_s, iv_s = fml.split("|")
    if head.count("~") != 1 or iv_s.count("~") != 1:
        return None
    lhs, exog_s = (t.strip() for t in head.split("~"))
    endog_s, instr_s = iv_s.split("~")

    def names(text: str, allow_one: bool = False) -> Optional[List[str]]:
        toks = [t.strip() for t in text.split("+")]
        if any(not t for t in toks):
            return None
        toks = [t for t in toks if not (allow_one and t == "1")]
        return toks if all(_NAME.match(t) for t in toks) else None

    exog = names(exog_s, allow_one=True)
    fe = names(fe_s)
    endog = names(endog_s)
    instr = names(instr_s)
    if not _NAME.match(lhs) or exog is None or not fe or not endog or not instr:
        return None
    if len(instr) < len(endog):
        return None
    every = [lhs] + exog + fe + endog + instr
    if len(set(every)) != len(every):
        return None
    return lhs, exog, fe, endog, instr


def _base_columns(rhs: List[Union[str, ITerm]]) -> List[str]:
    out: List[str] = []
    for t in rhs:
        cols = [t] if isinstance(t, str) else [t.factor] + ([t.var] if t.var else [])
        out += [c for c in cols if c not in out]
    return out


def _expand_i_terms(
    rhs: List[Union[str, ITerm]], work: pd.DataFrame
) -> Union[str, Tuple[List[str], Dict[str, np.ndarray], Dict[str, str]]]:
    """Dummy columns for the ``i()`` terms, in formula order.

    Levels are sorted and the reference level left out, as the delegate
    reports them; a dummy is named ``factor::level`` (``factor::level:var``
    for an interaction).
    """
    names: List[str] = []
    cols: Dict[str, np.ndarray] = {}
    renames: Dict[str, str] = {}
    for t in rhs:
        if isinstance(t, str):
            names.append(t)
            continue
        f = work[t.factor]
        if pd.api.types.is_bool_dtype(f) or not (
            pd.api.types.is_integer_dtype(f)
            or pd.api.types.is_object_dtype(f)
            or pd.api.types.is_string_dtype(f)
        ):
            return f"i({t.factor}) on a column of dtype {f.dtype}"
        values = f.to_numpy()
        try:
            ref: Any = int(t.ref) if pd.api.types.is_integer_dtype(f) else t.ref
        except ValueError:
            return f"i({t.factor}, ref={t.ref}) on an integer column"
        codes, levels = pd.factorize(values, sort=True)
        level_list = list(levels)
        if ref not in level_list:
            return f"i({t.factor}): reference level {t.ref} not in the data"
        var = work[t.var].to_numpy(dtype=np.float64) if t.var else None
        for j, lev in enumerate(level_list):
            if lev == ref:
                continue
            internal = f"__i{len(renames)}__"
            col = (codes == j).astype(np.float64)
            cols[internal] = col * var if var is not None else col
            renames[internal] = f"{t.factor}::{lev}" + (f":{t.var}" if t.var else "")
            names.append(internal)
    return names, cols, renames


def _resolve_vcov(
    vcov: Any, cluster: Optional[str]
) -> Optional[Tuple[str, Optional[Union[str, List[str]]]]]:
    if vcov is None:
        return ("cr1", cluster) if cluster is not None else ("iid", None)
    if cluster is not None:
        # sp.feols hands ``cluster=`` to pyfixest only when vcov is unset.
        return None
    if isinstance(vcov, str):
        v = vcov.strip().lower()
        if v == "iid":
            return "iid", None
        if v in _HETERO:
            return "hc1", None
        return None
    if isinstance(vcov, dict) and len(vcov) == 1:
        ((kind, col),) = vcov.items()
        if str(kind).upper() == "CRV1" and isinstance(col, str):
            cols = [c.strip() for c in col.split("+")]
            if all(_NAME.match(c) for c in cols) and len(set(cols)) == len(cols):
                if len(cols) == 1:
                    return "cr1", cols[0]
                if len(cols) == 2:
                    return "cr1", cols
    return None


def _numeric(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s) and not pd.api.types.is_bool_dtype(s)


def plan(
    fml: Any,
    data: Any,
    *,
    vcov: Any,
    cluster: Optional[str],
    weights: Optional[str],
    ssc: Any,
    fixef_rm: str,
    extra: Dict[str, Any],
    poisson: bool = False,
    binomial: bool = False,
    multiway_ok: bool = False,
    iv_ok: bool = False,
) -> Union[Plan, str]:
    """A :class:`Plan` for this call, or the reason it goes to pyfixest."""
    if not isinstance(data, pd.DataFrame):
        return "data is not a pandas DataFrame"
    if ssc is not None:
        return "ssc= is a pyfixest object"
    if extra:
        return f"pyfixest keywords {sorted(extra)}"
    if fixef_rm not in ("none", "singleton"):
        return f"fixef_rm={fixef_rm!r}"
    iv = _split_iv(fml) if iv_ok else None
    endog: List[str] = []
    instr: List[str] = []
    rhs: List[Union[str, ITerm]]
    if iv is not None:
        lhs, exog_names, fe, endog, instr = iv
        rhs = list(exog_names)
    else:
        parts = _split_formula(fml)
        if parts is None:
            return "formula uses syntax beyond 'y ~ x1 + x2 | fe1 + fe2'"
        lhs, rhs, fe = parts
    resolved = _resolve_vcov(vcov, cluster)
    if resolved is None:
        return f"vcov={vcov!r} (cluster={cluster!r})"
    kind, cl = resolved
    if isinstance(cl, list) and not multiway_ok:
        return "two-way clustering (native for sp.feols only)"
    cl_cols = [] if cl is None else [cl] if isinstance(cl, str) else list(cl)
    base = _base_columns(rhs) + endog + instr
    used = [lhs] + [c for c in base if c != lhs] + fe
    used += [
        c for c in ([weights] if weights is not None else []) + cl_cols if c not in used
    ]
    if any(c not in data.columns for c in used):
        return "a column is missing"  # pyfixest reports it by name
    if data.columns.duplicated().any():
        return "duplicated column names"
    factors = {t.factor for t in rhs if isinstance(t, ITerm)}
    numeric = [lhs] + [c for c in base if c not in factors]
    for c in numeric + ([weights] if weights is not None else []):
        if not _numeric(data[c]):
            return f"column {c!r} is not numeric"
    # Rows fixest drops: a missing value in any used column. A missing
    # cluster id is left to pyfixest, which reports it.
    missing = np.zeros(len(data), dtype=bool)
    for c in used:
        col = data[c]
        arr = col.to_numpy()
        if arr.dtype.kind == "f":
            # One pass for the common case: every value finite.
            if np.isfinite(arr).all():
                continue
            if np.isinf(arr).any():
                return f"column {c!r} has infinite values"
        if col.hasnans:
            if c in cl_cols:
                return f"cluster column {c!r} has missing values"
            missing |= col.isna().to_numpy()
    n_missing = int(missing.sum())
    work = data.loc[~missing, used] if n_missing else data
    if len(work) == 0:
        return "no complete rows"
    if weights is not None and (work[weights].to_numpy() <= 0).any():
        return "non-positive weights"
    if poisson and (work[lhs].to_numpy() < 0).any():
        return "negative Poisson outcome"
    if binomial:
        yb = work[lhs].to_numpy()
        if ((yb < 0) | (yb > 1)).any():
            return "binomial outcome outside [0, 1]"
    renames: Dict[str, str] = {}
    if factors:
        expanded = _expand_i_terms(rhs, work)
        if isinstance(expanded, str):
            return expanded
        rhs_names, dummies, renames = expanded
        keep_cols = set(fe) | set(cl_cols) | {lhs, weights}
        frame = {c: work[c] for c in used if c not in factors or c in keep_cols}
        frame.update(dummies)
        work = pd.DataFrame(frame, index=work.index, copy=False)
    else:
        rhs_names = [t for t in rhs if isinstance(t, str)]
    return Plan(
        lhs=lhs,
        rhs=rhs_names,
        fe=fe,
        vcov=kind,
        cluster=cl,
        weights=weights,
        drop_singletons=(fixef_rm == "singleton"),
        data=work,
        n_missing=n_missing,
        renames=renames,
        endog=endog,
        instr=instr,
    )


def _refused(exc: Exception) -> bool:
    """The native fit refused a design that fixest fits by dropping columns."""
    from ..exceptions import NumericalInstability

    diag = getattr(exc, "diagnostics", None) or {}
    if isinstance(exc, NumericalInstability):
        return bool({"absorbed", "collinear"} & set(diag)) or (
            "singular" in str(exc).lower()
        )
    return False


def _singleton_obs(codes: List[np.ndarray], cards: List[int]) -> int:
    """Rows in a level of any fixed effect that has one row."""
    if not codes:
        return 0
    single = np.zeros(len(codes[0]), dtype=bool)
    for c, g in zip(codes, cards):
        single |= np.bincount(c, minlength=g)[c] == 1
    return int(single.sum())


def _model_info(p: Plan, family: str, kernel: str, fml: str) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "model_type": f"{family} (StatsPAI native)",
        "method": "High-Dimensional Fixed Effects" if p.fe else "OLS",
        "formula": fml,
        "vcov_type": p.vcov_label,
        "vcov_type_detail": p.vcov_detail,
        "clustervar": (
            []
            if p.cluster is None
            else [p.cluster] if isinstance(p.cluster, str) else list(p.cluster)
        ),
        "fixed_effects": " + ".join(p.fe) if p.fe else "None",
        "backend": "statspai-native",
        "implementation": "native",
        "fe_kernel": kernel,
        "weights_name": p.weights,
    }
    if p.n_missing:
        info["n_missing_dropped"] = p.n_missing
    return info


def _inference(
    params: pd.Series,
    V: np.ndarray,
    p: Plan,
    df_t: int,
    normal: bool,
) -> Tuple[pd.Series, pd.Series, pd.Series]:
    from scipy import stats

    se = pd.Series(np.sqrt(np.clip(np.diag(V), 0.0, None)), index=params.index)
    with np.errstate(divide="ignore", invalid="ignore"):
        t = params / se
    if normal:
        pv = 2.0 * stats.norm.sf(np.abs(t.to_numpy()))
    else:
        pv = 2.0 * stats.t.sf(np.abs(t.to_numpy()), max(df_t, 1))
    return se, t, pd.Series(pv, index=params.index)


def _attach_fe(
    out: EconometricResults,
    p: Plan,
    keep: Any,
    eta: Any,
    cols: List[str],
    beta: Any,
) -> None:
    """Keep what ``fixef()`` / ``predict(newdata)`` / margins need."""
    if not p.fe:
        return
    from ._fixef import attach

    keep = np.asarray(keep, dtype=bool)
    X = (
        p.data.loc[keep, cols].to_numpy(dtype=float)
        if cols
        else np.zeros((int(keep.sum()), 0))
    )
    attach(out, p.data, keep, p.fe, np.asarray(eta, dtype=float), X, beta)


def fit_feols(p: Plan, fml: str, *, lean: bool = False) -> EconometricResults:
    """Fit a planned ``sp.feols`` call natively, in the delegate's layout."""
    from ..fast.feols import _feols_fit

    if p.endog:
        return fit_feols_iv(p, fml, lean=lean)

    res, a = _feols_fit(
        p.formula,
        p.data,
        vcov=p.vcov,
        cluster=p.cluster,
        weights=p.weights,
        drop_singletons=p.drop_singletons,
        note_weights=False,
    )
    names = [
        "Intercept" if c == "(Intercept)" else p.renames.get(c, c)
        for c in res.coef_names
    ]
    params = pd.Series(np.asarray(res.coef_vec, dtype=float), index=names)
    V = np.asarray(res.vcov_matrix, dtype=float)
    N, k = int(res.n_kept), len(names)
    w = np.asarray(a["w"], dtype=float)
    weighted = p.weights is not None
    if p.vcov == "cr1":
        G = int(a["n_clusters"])
        df_k, df_t = a["df_k_cluster"], G - 1
    else:
        G = 0
        df_k = a["df_k"]
        df_t = N - df_k
    se, t, pv = _inference(params, V, p, df_t, normal=False)

    # R-squared family, as pyfixest computes them (weighted sums of squares).
    u = np.asarray(a["resid"], dtype=float)
    y = np.asarray(a["y"], dtype=float)
    ssu = float(w @ (u * u))
    ssy = float(w @ (y - np.average(y, weights=w)) ** 2)
    has_intercept = 1
    diagnostics: Dict[str, Any] = {}
    if p.fe:
        k_fe = int(np.sum(np.asarray(res.fe_cardinality) - 1) + 1)
        adj = (N - has_intercept) / (N - k - k_fe)
        adj_within = (N - k_fe) / (N - k - k_fe)
        y_dem = np.asarray(a["y_dem"], dtype=float)
        ssy_within = float(w @ (y_dem * y_dem))
        r2_within = 1.0 - ssu / ssy_within
        adj_r2_within = 1.0 - ssu / ssy_within * adj_within
    else:
        adj = (N - has_intercept) / (N - k)
        r2_within = adj_r2_within = float("nan")
    diagnostics["R-squared"] = 1.0 - ssu / ssy
    diagnostics["Adj. R-squared"] = 1.0 - ssu / ssy * adj
    diagnostics["R-squared (within)"] = r2_within
    diagnostics["Adj. R-squared (within)"] = adj_r2_within
    diagnostics["RMSE"] = float(np.sqrt(ssu / N))

    model_info = _model_info(p, "OLS", res.fe_kernel, fml)
    if p.fe:
        model_info["fe_solver"] = a.get("fe_solver")
    if p.fe and not p.drop_singletons:
        model_info["fe_singleton_obs"] = _singleton_obs(a["fe_codes"], a["fe_card"])
    if G:
        model_info["n_clusters"] = G
    data_info: Dict[str, Any] = {
        "nobs": N,
        "df_resid": N - max(k, df_k),
        "n_params": k,
        "var_cov": V,
        "residuals": u,
        "fitted_values": y - u,
    }
    if G:
        data_info["df_inference"] = G - 1
    if not lean:
        sw = np.sqrt(w)[:, None] if weighted else None
        X = np.asarray(a["X_dem"], dtype=float)
        yd = np.asarray(a["y_dem"], dtype=float)
        data_info["X"] = X * sw if weighted else X
        data_info["y"] = yd * sw[:, 0] if weighted else yd
        data_info["var_names"] = names
    if isinstance(p.cluster, list):
        # Two-way: fixest sets the negative eigenvalues of V_a + V_b - V_ab
        # to zero (Cameron, Gelbach and Miller 2011).
        from ..inference.jackknife import psd_adjust

        V, n_negative = psd_adjust(V)
        model_info["multiway_negative_eigenvalues"] = n_negative
        if n_negative:
            se, t, pv = _inference(params, V, p, df_t, normal=False)
            data_info["var_cov"] = V
    out = EconometricResults(
        params=params,
        std_errors=se,
        model_info=model_info,
        data_info=data_info,
        diagnostics=diagnostics,
    )
    out.tvalues = t
    out.pvalues = pv
    _attach_fe(out, p, a["keep_mask"], y - u, list(res.coef_names), params.to_numpy())
    return out


def _psd_two_way(p: Plan, V: np.ndarray) -> Tuple[np.ndarray, int]:
    """With two-way clustering, fixest's fix: negative eigenvalues of the
    covariance set to zero. Returns the matrix and how many there were."""
    if not isinstance(p.cluster, list):
        return V, 0
    from ..inference.jackknife import psd_adjust

    return psd_adjust(V)


def _crv(
    X: np.ndarray, uw: np.ndarray, dims: List[Tuple[np.ndarray, int]]
) -> np.ndarray:
    """Cluster meat; with two dimensions ``M_a + M_b - M_ab``."""
    from ..fast.inference import _cr1_meat

    meat = np.zeros((X.shape[1], X.shape[1]))
    for (codes, g), sign in zip(dims, (1.0, 1.0, -1.0)):
        meat += sign * _cr1_meat(X, uw, codes, g)
    return meat


def fit_feols_iv(p: Plan, fml: str, *, lean: bool = False) -> EconometricResults:
    """2SLS with absorbed effects, reported as ``sp.feols`` reports an IV fit.

    The outcome, exogenous regressors, endogenous regressors and excluded
    instruments are residualised on the fixed effects together (one call of
    the feols engine, which also fixes the estimation sample, the weights,
    the cluster codes and the absorbed degrees of freedom); 2SLS runs on the
    residualised columns, scaled by the square root of the weights. The
    variance is the delegate's: bread ``(X^' X^)^{-1}`` on the first-stage
    fitted design, scores ``w u x^``, fixest's small-sample factors. The
    first-stage F is the Wald statistic of the excluded instruments in the
    first-stage regression under the model's variance, divided by their
    number, with a chi-square p-value: fixest's ``ivwald1`` (pyfixest
    clusters its first stage on the first dimension only and refits it
    with singletons dropped, so its robust value can differ).
    """
    from scipy import stats

    from ..fast.feols import _feols_fit

    k_ex, k_en, k_in = len(p.rhs), len(p.endog), len(p.instr)
    cols = list(p.rhs) + p.endog + p.instr
    res, a = _feols_fit(
        f"{p.lhs} ~ {' + '.join(cols)} | {' + '.join(p.fe)}",
        p.data,
        vcov=p.vcov,
        cluster=p.cluster,
        weights=p.weights,
        drop_singletons=p.drop_singletons,
        note_weights=False,
    )
    k_all = len(cols)
    sw = np.sqrt(np.asarray(a["w"], dtype=float))
    D = np.asarray(a["X_dem"], dtype=float) * sw[:, None]
    ys = np.asarray(a["y_dem"], dtype=float) * sw
    Xe, En, Zi = D[:, :k_ex], D[:, k_ex : k_ex + k_en], D[:, k_ex + k_en :]
    Z = np.column_stack([Xe, Zi])
    X = np.column_stack([Xe, En])
    n = X.shape[0]
    k = k_ex + k_en
    ZtZ_inv = np.linalg.inv(Z.T @ Z)
    Pi = ZtZ_inv @ (Z.T @ En)
    Xhat = np.column_stack([Xe, Z @ Pi])
    bread = np.linalg.inv(Xhat.T @ Xhat)
    beta = bread @ (Xhat.T @ ys)
    u_s = ys - X @ beta  # sqrt(w)-scaled residual
    fe_dof = int(a["df_k"]) - k_all
    df_k = k + fe_dof
    dims: List[Tuple[np.ndarray, int]] = list(a["cl_dims"])
    G = int(a["n_clusters"]) if p.vcov == "cr1" else 0
    if p.vcov == "iid":
        V = bread * float(u_s @ u_s) / max(n - df_k, 1)
    elif p.vcov == "hc1":
        S = u_s[:, None] * Xhat
        V = bread @ (S.T @ S) @ bread * (n / max(n - df_k, 1))
    else:
        extra = int(a["df_k_cluster"]) - k_all
        df_k = k + extra
        V = bread @ _crv(Xhat, u_s, dims) @ bread
        V *= (G / (G - 1.0)) * ((n - 1.0) / max(n - df_k, 1))
    names = list(p.rhs) + p.endog
    params = pd.Series(beta, index=names)
    df_t = (G - 1) if G else n - df_k
    n_negative = 0
    if isinstance(p.cluster, list):
        from ..inference.jackknife import psd_adjust

        V, n_negative = psd_adjust(V)
    se, t, pv = _inference(params, V, p, df_t, normal=False)

    # First stage of the single endogenous regressor, under the model's
    # variance (first clustering dimension only).
    first: Optional[Tuple[float, Optional[float]]] = None
    if k_en == 1:
        v_s = En[:, 0] - Z @ Pi[:, 0]
        kz = Z.shape[1]
        if p.vcov == "iid":
            V1 = ZtZ_inv * float(v_s @ v_s) / max(n - (kz + fe_dof), 1)
        elif p.vcov == "hc1":
            S1 = v_s[:, None] * Z
            V1 = ZtZ_inv @ (S1.T @ S1) @ ZtZ_inv * (n / max(n - (kz + fe_dof), 1))
        else:
            # fixest's ivwald1: the first stage under the model's own
            # clustering (both dimensions when two-way, with the same
            # eigenvalue fix).
            extra1 = int(a["df_k_cluster"]) - k_all
            V1 = ZtZ_inv @ _crv(Z, v_s, dims) @ ZtZ_inv
            V1 *= (G / (G - 1.0)) * ((n - 1.0) / max(n - (kz + extra1), 1))
            if isinstance(p.cluster, list):
                from ..inference.jackknife import psd_adjust

                V1, _ = psd_adjust(V1)
        b_z = Pi[k_ex:, 0]
        V_zz = V1[k_ex:, k_ex:]
        wald = float(b_z @ np.linalg.pinv(V_zz) @ b_z)
        first = (wald / k_in, float(stats.chi2.sf(wald, k_in)))

    nan = float("nan")
    diagnostics: Dict[str, Any] = {
        "R-squared": nan,
        "Adj. R-squared": nan,
        "R-squared (within)": nan,
        "Adj. R-squared (within)": nan,
        "RMSE": nan,
    }
    model_info = _model_info(p, "IV", res.fe_kernel, fml)
    if p.fe:
        model_info["fe_solver"] = a.get("fe_solver")
    if p.fe and not p.drop_singletons:
        model_info["fe_singleton_obs"] = _singleton_obs(a["fe_codes"], a["fe_card"])
    model_info["endogenous"] = list(p.endog)
    model_info["instruments"] = list(p.instr)
    if G:
        model_info["n_clusters"] = G
    if isinstance(p.cluster, list):
        model_info["multiway_negative_eigenvalues"] = n_negative
    N = int(res.n_kept)
    u = u_s / np.where(sw > 0, sw, 1.0)
    data_info: Dict[str, Any] = {
        "nobs": N,
        "df_resid": N - max(k, df_k),
        "n_params": k,
        "var_cov": V,
        "residuals": u,
    }
    if G:
        data_info["df_inference"] = G - 1
    if not lean:
        # The 2SLS structure on the residualised columns, under the ``iv``
        # namespace sp.ivreg uses: the WRE wild bootstrap reads it there.
        # Not under X / y: the least-squares helpers (wild_cluster_boot,
        # cr2_se) would treat the structural design as an OLS one.
        data_info["iv"] = {
            "y": ys,
            "X": X,
            "W": Z,
            "exog_names": list(p.rhs),
            "endog_names": list(p.endog),
            "var_names": names,
            "n_exog": k_ex,
            "n_endog": k_en,
            "kappa": 1.0,
            "weighted": p.weights is not None,
            # the WRE bootstrap residualises each draw on the effects again
            "fe_codes": [np.asarray(c, dtype=np.int64) for c in a["fe_codes"]],
        }
        data_info["sample_index"] = p.data.index[np.asarray(a["keep_mask"], bool)]
    out = EconometricResults(
        params=params,
        std_errors=se,
        model_info=model_info,
        data_info=data_info,
        diagnostics=diagnostics,
    )
    out.tvalues = t
    out.pvalues = pv
    _attach_fe(
        out,
        p,
        a["keep_mask"],
        np.asarray(a["y"], dtype=float) - u,
        list(p.rhs) + list(p.endog),
        beta,
    )
    if p.weights is not None:
        return out  # the delegate reports IV diagnostics for unweighted fits
    from .wrapper import _iv_diagnostics_core

    _iv_diagnostics_core(
        out,
        Y=ys,
        X=X,
        Z=Z,
        u=u_s,
        names=names,
        endo=list(range(k_ex, k)),
        k_total=df_k,
        first=first,
    )
    return out


def fit_fepois(p: Plan, fml: str, *, tol: float, maxiter: int) -> EconometricResults:
    """Fit a planned ``sp.fepois`` call natively, in the delegate's layout."""
    from ..fast.fepois import _fepois_fit

    res, a = _fepois_fit(
        p.formula,
        p.data,
        vcov=p.vcov,
        cluster=p.cluster,
        weights=p.weights,
        drop_singletons=p.drop_singletons,
        # A relative deviance change of 1e-8 (fixest's and pyfixest's stop)
        # can leave the fixed effects, and so the standard errors, 1e-5
        # from the optimum; one more Newton step puts them within ~1e-9.
        tol=min(tol, 1e-10),
        maxiter=maxiter,
        note_weights=False,
    )
    names = [
        "Intercept" if c == "(Intercept)" else p.renames.get(c, c)
        for c in res.coef_names
    ]
    params = pd.Series(np.asarray(res.coef_vec, dtype=float), index=names)
    V = np.asarray(res.vcov_matrix, dtype=float)
    N, k = int(res.n_kept), len(names)
    if p.vcov == "cr1":
        G = int(a["n_clusters"])
        df_k = a["df_k_cluster"]
    else:
        G = 0
        df_k = a["df_k"]
    V, n_negative = _psd_two_way(p, V)
    se, t, pv = _inference(params, V, p, N - df_k, normal=True)
    y = np.asarray(a["y"], dtype=float)
    mu = np.asarray(a["mu"], dtype=float)
    nan = float("nan")
    diagnostics: Dict[str, Any] = {
        "R-squared": nan,
        "Adj. R-squared": nan,
        "R-squared (within)": nan,
        "Adj. R-squared (within)": nan,
        "RMSE": nan,
        "Deviance": float(res.deviance),
        "Log-likelihood": float(res.log_likelihood),
    }
    model_info = _model_info(p, "Poisson", res.fe_kernel, fml)
    if p.fe and not p.drop_singletons:
        model_info["fe_singleton_obs"] = _singleton_obs(a["fe_codes"], a["fe_card"])
    model_info["converged"] = bool(res.converged)
    model_info["iterations"] = int(res.iterations)
    if res.n_dropped_separation:
        model_info["n_dropped_separation"] = int(res.n_dropped_separation)
        import warnings

        warnings.warn(
            f"sp.fepois: {res.n_dropped_separation} observations removed "
            "because of separation (fixed-effect groups whose outcomes are "
            "all zero).",
            UserWarning,
            stacklevel=4,
        )
    if res.n_dropped_singletons:
        model_info["n_dropped_singletons"] = int(res.n_dropped_singletons)
    if G:
        model_info["n_clusters"] = G
    if isinstance(p.cluster, list):
        model_info["multiway_negative_eigenvalues"] = n_negative
    data_info: Dict[str, Any] = {
        "nobs": N,
        "df_resid": N - max(k, df_k),
        "n_params": k,
        "var_cov": V,
        "residuals": y - mu,
        "fitted_values": mu,
    }
    if G:
        data_info["df_inference"] = G - 1
    if not res.converged:
        import warnings

        from ..exceptions import ConvergenceWarning

        warnings.warn(
            f"sp.fepois: IRLS stopped after {res.iterations} iterations "
            f"without meeting iwls_tol={tol}.",
            ConvergenceWarning,
            stacklevel=4,
        )
    out = EconometricResults(
        params=params,
        std_errors=se,
        model_info=model_info,
        data_info=data_info,
        diagnostics=diagnostics,
    )
    out.tvalues = t
    out.pvalues = pv
    model_info["link"] = "log"
    _attach_fe(
        out, p, a["keep_mask"], a["eta"], list(res.coef_names), params.to_numpy()
    )
    return out


def fit_feglm(p: Plan, fml: str, *, family: str) -> EconometricResults:
    """Fit a planned ``sp.feglm`` logit / probit call natively."""
    from ..fast._feglm import feglm_fit

    res, a = feglm_fit(
        p.lhs,
        p.rhs,
        p.fe,
        p.data,
        family=family,
        vcov=p.vcov,
        cluster=p.cluster,
        weights=p.weights,
    )
    names = [
        "Intercept" if c == "(Intercept)" else p.renames.get(c, c)
        for c in res.coef_names
    ]
    params = pd.Series(np.asarray(res.coef_vec, dtype=float), index=names)
    V = np.asarray(res.vcov_matrix, dtype=float)
    N, k = int(res.n_kept), len(names)
    G = int(a["n_clusters"]) if p.vcov == "cr1" else 0
    df_k = a["df_k_cluster"] if G else a["df_k"]
    V, n_negative = _psd_two_way(p, V)
    se, t, pv = _inference(params, V, p, N - df_k, normal=True)
    nan = float("nan")
    diagnostics: Dict[str, Any] = {
        "R-squared": nan,
        "Adj. R-squared": nan,
        "R-squared (within)": nan,
        "Adj. R-squared (within)": nan,
        "RMSE": nan,
        "Deviance": float(res.deviance),
        "Log-likelihood": float(res.log_likelihood),
    }
    model_info = _model_info(p, "GLM", res.fe_kernel, fml)
    model_info["family"] = family
    model_info["converged"] = bool(res.converged)
    model_info["iterations"] = int(res.iterations)
    if res.n_dropped_separation:
        model_info["n_dropped_separation"] = int(res.n_dropped_separation)
        import warnings

        warnings.warn(
            f"sp.feglm: {res.n_dropped_separation} observations removed "
            "because their fixed-effect group has only 0 or only 1 outcomes "
            "(perfect fit), as fixest does.",
            UserWarning,
            stacklevel=4,
        )
    if G:
        model_info["n_clusters"] = G
    if isinstance(p.cluster, list):
        model_info["multiway_negative_eigenvalues"] = n_negative
    y = np.asarray(a["y"], dtype=float)
    mu = np.asarray(a["mu"], dtype=float)
    data_info: Dict[str, Any] = {
        "nobs": N,
        "df_resid": N - max(k, df_k),
        "n_params": k,
        "var_cov": V,
        "residuals": y - mu,
        "fitted_values": mu,
    }
    if G:
        data_info["df_inference"] = G - 1
    out = EconometricResults(
        params=params,
        std_errors=se,
        model_info=model_info,
        data_info=data_info,
        diagnostics=diagnostics,
    )
    out.tvalues = t
    out.pvalues = pv
    model_info["link"] = family
    _attach_fe(
        out, p, a["keep_mask"], a["eta"], list(res.coef_names), params.to_numpy()
    )
    return out


def refused(exc: Exception) -> bool:
    """Whether ``exc`` is a native refusal that pyfixest can resolve."""
    return _refused(exc)


__all__ = ["Plan", "plan", "fit_feols", "fit_fepois", "fit_feglm", "refused"]
