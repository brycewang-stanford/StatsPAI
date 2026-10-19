"""Covariate balance for binary, multi-category and continuous treatments.

:func:`bal_tab` follows the conventions of R ``cobalt::bal.tab`` (GPL,
used as a black box: documentation and outputs on shared fixtures only):

* mean differences of continuous covariates are standardized by an
  *unweighted* standard deviation, so weights cannot improve the
  statistic by changing the spread; binary covariates report the raw
  difference in proportions;
* the standard deviation is the pooled one (square root of the mean of
  the group variances, over *all* treatment groups) for the ATE, the focal
  group's for the ATT and the control group's for the ATC;
* multi-category treatments are compared pairwise (or each group against
  the full unweighted sample with ``pairwise=False``) and summarized by
  the largest absolute statistic across comparisons;
* continuous treatments report the treatment-covariate correlation, its
  weighted version being the weighted covariance over the *unweighted*
  standard deviations; standard deviations of binary covariates use the
  ``n`` divisor (``sqrt(p (1 - p))``);
* weighted variances use ``sum(w (x - m)^2) sum(w) / ((sum w)^2 - sum
  w^2)``, the effective sample size is ``(sum w)^2 / sum w^2``.
"""

from __future__ import annotations

import itertools
from typing import Any, ClassVar, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy.stats import rankdata

from .._result_serialize import ResultProtocolMixin
from ..exceptions import MethodIncompatibility
from .ps_diagnostics import _ks_stat


class BalTabResult(ResultProtocolMixin):
    """Balance table returned by :func:`sp.bal_tab`.

    Attributes
    ----------
    table : pd.DataFrame
        One row per covariate term. Binary / multi-category treatments:
        ``diff_un`` / ``diff_adj`` (multi-category: the largest absolute
        difference across comparisons), ``vr_*`` and ``ks_*``. Continuous
        treatments: ``corr_*`` (Pearson) and ``scorr_*`` (Spearman).
        ``balanced`` compares the adjusted statistic with ``threshold``.
    pairs : dict
        Multi-category treatments: one table per comparison, keyed
        ``'g1 vs. g2'``; differences are ``g1 - g2``.
    observations : pd.DataFrame
        Sample sizes (``unadjusted``) and effective sample sizes
        (``adjusted``) per treatment group.
    summary_stats : dict
        Largest adjusted statistic, number of terms over the threshold,
        the standardization used.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> df = pd.DataFrame({'x': rng.normal(size=200)})
    >>> df['g'] = np.where(df['x'] + rng.normal(size=200) > 0, 'b', 'a')
    >>> bt = sp.bal_tab(df, 'g', ['x'])
    >>> bool(abs(bt.table.loc['x', 'diff_un']) > 0.5)
    True
    """

    _citation_keys: ClassVar[Tuple[str, ...]] = ()

    def __init__(
        self,
        table: pd.DataFrame,
        pairs: Dict[str, pd.DataFrame],
        observations: pd.DataFrame,
        summary: Dict[str, Any],
    ) -> None:
        self.table = table
        self.pairs = pairs
        self.observations = observations
        self.summary_stats = summary

    def summary(self) -> str:
        s = self.summary_stats
        lines = [
            f"Balance ({s['treat_type']} treatment"
            + (f", estimand {s['estimand']}" if s.get("estimand") else "")
            + ")",
            "=" * 70,
            self.table.to_string(float_format=lambda x: f"{x:.4f}"),
            "",
            self.observations.to_string(float_format=lambda x: f"{x:.1f}"),
        ]
        if s.get("adjusted"):
            lines.append("")
            lines.append(
                f"Largest |{s['statistic']}| after weighting: "
                f"{s['max_abs_adj']:.4f}; terms over {s['threshold']}: "
                f"{s['n_imbalanced_adj']}"
            )
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()


# ----------------------------------------------------------------------
# small weighted statistics
# ----------------------------------------------------------------------


def _wvar(x: np.ndarray, w: np.ndarray) -> float:
    W = float(w.sum())
    m = float((w * x).sum()) / W
    denom = W**2 - float((w**2).sum())
    if denom <= 0:
        return float("nan")
    return float((w * (x - m) ** 2).sum()) * W / denom


def _wcov(u: np.ndarray, v: np.ndarray, w: np.ndarray) -> float:
    W = float(w.sum())
    mu = float((w * u).sum()) / W
    mv = float((w * v).sum()) / W
    denom = W**2 - float((w**2).sum())
    return float((w * (u - mu) * (v - mv)).sum()) * W / denom


def _sd(x: np.ndarray, binary: bool) -> float:
    return float(x.std(ddof=0 if binary else 1))


def _ess(w: np.ndarray) -> float:
    return float(w.sum() ** 2 / (w**2).sum()) if (w**2).sum() > 0 else float("nan")


def _vr(x1: np.ndarray, w1: np.ndarray, x2: np.ndarray, w2: np.ndarray) -> float:
    v1, v2 = _wvar(x1, w1), _wvar(x2, w2)
    if not v2 > 0:
        return float("nan")
    return v1 / v2


# ----------------------------------------------------------------------
# design
# ----------------------------------------------------------------------


def _covariate_terms(
    data: pd.DataFrame, covariates: Sequence[str]
) -> Tuple[np.ndarray, List[str], List[bool]]:
    """Numeric matrix of covariate terms. A text / categorical covariate
    becomes one indicator per level (``col_level``); a two-valued numeric
    covariate is recoded to 0/1 (0 kept if present, else the lower value
    is 0)."""
    cols: List[np.ndarray] = []
    names: List[str] = []
    is_bin: List[bool] = []
    for c in covariates:
        s = data[c]
        if isinstance(s.dtype, pd.CategoricalDtype) or not (
            pd.api.types.is_numeric_dtype(s) or pd.api.types.is_bool_dtype(s)
        ):
            levels = (
                [lv for lv in s.cat.categories if lv in set(s.unique())]
                if isinstance(s.dtype, pd.CategoricalDtype)
                else sorted(s.unique().tolist(), key=str)
            )
            for lv in levels:
                cols.append((s == lv).to_numpy(dtype=float))
                names.append(f"{c}_{lv}")
                is_bin.append(True)
            continue
        x = s.to_numpy(dtype=float)
        u = np.unique(x)
        if len(u) == 2:
            zero = 0.0 if 0.0 in u else u[0]
            x = (x != zero).astype(float)
            is_bin.append(True)
        else:
            is_bin.append(False)
        cols.append(x)
        names.append(str(c))
    return np.column_stack(cols), names, is_bin


def _resolve_weights(
    weights: Any, data: pd.DataFrame
) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
    from .weightit import WeightItResult

    meta: Dict[str, Any] = {}
    if weights is None:
        return None, meta
    if isinstance(weights, WeightItResult):
        meta = {
            "estimand": weights.estimand,
            "focal": weights.focal,
            "treat_type": weights.treat_type,
        }
        return weights.weights.to_numpy(dtype=float), meta
    if isinstance(weights, str):
        if weights not in data.columns:
            raise MethodIncompatibility(
                f"bal_tab: weights column {weights!r} not found."
            )
        return data[weights].to_numpy(dtype=float), meta
    if isinstance(weights, pd.Series) and not weights.index.equals(data.index):
        if (
            len(weights) == len(data)
            and weights.index.is_unique
            and data.index.is_unique
        ):
            weights = weights.reindex(data.index)
        else:
            raise MethodIncompatibility(
                "bal_tab: the index of weights does not match data.",
                recovery_hint="Pass the weights as a plain array in row order.",
            )
    arr = np.asarray(weights, dtype=float).ravel()
    if len(arr) != len(data):
        raise MethodIncompatibility(
            f"bal_tab: {len(arr)} weights for {len(data)} rows."
        )
    return arr, meta


def bal_tab(
    data: pd.DataFrame,
    treat: str,
    covariates: Sequence[str],
    weights: Any = None,
    *,
    estimand: Optional[str] = None,
    focal: Any = None,
    treat_type: Optional[str] = None,
    pairwise: bool = True,
    binary: str = "raw",
    continuous: str = "std",
    s_d_denom: Optional[str] = None,
    threshold: float = 0.1,
) -> BalTabResult:
    """Covariate balance before and after weighting (R ``cobalt::bal.tab``).

    Works for binary, multi-category and continuous treatments.

    Parameters
    ----------
    data : pd.DataFrame
        Analysis data.
    treat : str
        Treatment column.
    covariates : sequence of str
        Covariates; text / categorical ones are split into one indicator
        per level.
    weights : array-like, str, WeightItResult, optional
        Balancing weights (a column name, one weight per row, or the
        result of :func:`sp.weightit`, which also supplies ``estimand``,
        ``focal`` and ``treat_type``). ``None`` reports the unadjusted
        sample only. Rows with a missing weight are left out.
    estimand : {'ATE', 'ATT', 'ATC', 'ATO', 'ATM'}, optional
        Sets the default standardization: ATT uses the focal group's
        standard deviation, ATC the control group's, the others the
        pooled one. Default 'ATE'.
    focal : optional
        Focal (treated) level for ``estimand='ATT'``; binary 0/1
        treatments default to 1.
    treat_type : {'binary', 'multinomial', 'continuous'}, optional
        Override the type read from the column.
    pairwise : bool, default True
        Multi-category treatments: compare every pair of groups. With
        ``False``, compare each group with the full unweighted sample (no
        ``focal``) or the focal group with each other group.
    binary : {'raw', 'std'}, default 'raw'
        Report binary covariates as raw differences in proportion or
        standardize them.
    continuous : {'std', 'raw'}, default 'std'
        Standardize differences of continuous covariates or not.
    s_d_denom : {'pooled', 'treated', 'control', 'all'}, optional
        Override the standardization. ``'treated'`` is the focal group.
    threshold : float, default 0.1
        Balance threshold for the absolute adjusted difference (or
        correlation); terms above it get ``balanced=False``.

    Returns
    -------
    BalTabResult

    Notes
    -----
    Statistics, standardization and the multi-category summary follow
    ``cobalt`` 4.6.2 and are checked against it on shared fixtures
    (``tests/reference_parity/test_weightit_cobalt_parity.py``). Unlike
    :func:`sp.balance_diagnostics` (binary treatments, weighted standard
    deviations by default), the denominators here are always the
    unweighted ones.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(1)
    >>> n = 400
    >>> x = rng.normal(size=n)
    >>> g = rng.choice(['a', 'b', 'c'], size=n)
    >>> df = pd.DataFrame({'g': g, 'x': x + (g == 'c')})
    >>> W = sp.weightit(df, 'g', ['x'], method='ebal')
    >>> bt = sp.bal_tab(df, 'g', ['x'], weights=W)
    >>> sorted(bt.pairs)
    ['a vs. b', 'a vs. c', 'b vs. c']
    >>> bool(bt.table.loc['x', 'diff_adj'] < 1e-8)
    True

    References
    ----------
    [@greifer2026weightit]
    """
    if binary not in ("raw", "std") or continuous not in ("raw", "std"):
        raise MethodIncompatibility(
            "bal_tab: binary= and continuous= take 'raw' or 'std'."
        )
    if s_d_denom is not None and s_d_denom not in (
        "pooled",
        "treated",
        "control",
        "all",
    ):
        raise MethodIncompatibility(
            f"bal_tab: s_d_denom must be 'pooled', 'treated', 'control' or "
            f"'all'; got {s_d_denom!r}."
        )
    covariates = list(covariates)
    missing = [c for c in [treat] + covariates if c not in data.columns]
    if missing:
        raise MethodIncompatibility(
            f"bal_tab: columns {missing} not found.", diagnostics={"missing": missing}
        )
    w_all, meta = _resolve_weights(weights, data)
    estimand = str(estimand or meta.get("estimand") or "ATE").upper()
    if focal is None:
        focal = meta.get("focal")
    treat_type = treat_type or meta.get("treat_type")

    keep = data[[treat] + covariates].notna().all(axis=1).to_numpy()
    if w_all is not None:
        keep &= np.isfinite(w_all)
        if (w_all[keep] < 0).any():
            raise MethodIncompatibility("bal_tab: weights must be non-negative.")
    df = data.loc[keep]
    w = None if w_all is None else w_all[keep]
    X, names, is_bin = _covariate_terms(df, covariates)
    t = df[treat]

    if treat_type is None:
        from .weightit import _infer_treat_type

        treat_type = _infer_treat_type(t, treat)
    if treat_type == "continuous":
        return _continuous(t.to_numpy(dtype=float), X, names, is_bin, w, threshold)

    from .weightit import _levels_of

    levels = _levels_of(t)
    if len(levels) < 2:
        raise MethodIncompatibility("bal_tab: the treatment has a single level.")
    G = np.array([levels.index(v) for v in t.tolist()])
    if focal is None and estimand in ("ATT", "ATC") and set(levels) == {0, 1}:
        focal = 1 if estimand == "ATT" else 0
    if focal is not None and focal not in levels:
        raise MethodIncompatibility(
            f"bal_tab: focal={focal!r} is not a treatment level.",
            recovery_hint=f"Use one of {levels}.",
        )
    if s_d_denom is None:
        s_d_denom = {"ATT": "treated", "ATC": "control"}.get(estimand, "pooled")
    if s_d_denom == "treated" and focal is None:
        if len(levels) == 2:
            focal = levels[1]
        else:
            raise MethodIncompatibility(
                "bal_tab: standardizing by the treated group needs focal=.",
                recovery_hint=f"Pass focal= one of {levels}.",
            )
    if s_d_denom == "control" and len(levels) != 2:
        raise MethodIncompatibility(
            "bal_tab: s_d_denom='control' is defined for binary treatments."
        )

    # unweighted standard deviation per term
    sds = np.empty(X.shape[1])
    for j in range(X.shape[1]):
        x = X[:, j]
        if s_d_denom == "pooled":
            sds[j] = np.sqrt(
                np.mean([_sd(x[G == k], is_bin[j]) ** 2 for k in range(len(levels))])
            )
        elif s_d_denom == "all":
            sds[j] = _sd(x, is_bin[j])
        elif s_d_denom == "treated":
            sds[j] = _sd(x[G == levels.index(focal)], is_bin[j])
        else:
            if estimand == "ATC" and focal is not None:
                control = focal
            elif focal is not None:
                control = levels[1 - levels.index(focal)]
            else:
                control = levels[0]
            sds[j] = _sd(x[G == levels.index(control)], is_bin[j])
    standardize = np.array(
        [(binary == "std") if b else (continuous == "std") for b in is_bin]
    )

    ones = np.ones(len(G))
    if len(levels) == 2:
        # binary: treated (the ATT focal level, else the second level)
        # minus control
        hi = levels.index(focal) if (focal is not None and estimand == "ATT") else 1
        comps = [(hi, 1 - hi)]
    elif pairwise:
        comps = list(itertools.combinations(range(len(levels)), 2))
    elif focal is not None:
        f = levels.index(focal)
        comps = [(f, k) for k in range(len(levels)) if k != f]
    else:
        comps = [(k, -1) for k in range(len(levels))]  # -1 = full sample

    pairs: Dict[str, pd.DataFrame] = {}
    for g1, g2 in comps:
        rows = []
        for j, nm in enumerate(names):
            x = X[:, j]
            row: Dict[str, Any] = {"type": "binary" if is_bin[j] else "continuous"}
            for tag, wv in (("un", ones), ("adj", w)):
                if wv is None:
                    continue
                s1 = G == g1
                m1 = float(np.average(x[s1], weights=wv[s1]))
                if g2 == -1:
                    s2 = np.ones(len(G), dtype=bool)
                    w2 = ones
                else:
                    s2 = G == g2
                    w2 = wv
                m2 = float(np.average(x[s2], weights=w2[s2]))
                diff = m1 - m2
                if standardize[j]:
                    diff = diff / sds[j] if sds[j] > 0 else float("nan")
                row[f"diff_{tag}"] = diff
                row[f"vr_{tag}"] = (
                    float("nan") if is_bin[j] else _vr(x[s1], wv[s1], x[s2], w2[s2])
                )
                row[f"ks_{tag}"] = _ks_stat(x[s1], x[s2], wv[s1], w2[s2])
            rows.append(row)
        lab1 = levels[g1]
        lab2 = "All" if g2 == -1 else levels[g2]
        pairs[f"{lab1} vs. {lab2}"] = pd.DataFrame(rows, index=names)

    if len(comps) == 1:
        table = next(iter(pairs.values())).copy()
    else:
        cols: Dict[str, Any] = {
            "type": ["binary" if b else "continuous" for b in is_bin]
        }
        for tag in ("un", "adj"):
            if tag == "adj" and w is None:
                continue
            cols[f"diff_{tag}"] = np.nanmax(
                np.column_stack([p[f"diff_{tag}"].abs() for p in pairs.values()]),
                axis=1,
            )
            vrs = np.column_stack([p[f"vr_{tag}"] for p in pairs.values()])
            with np.errstate(divide="ignore", invalid="ignore"):
                vrs = np.maximum(vrs, 1.0 / vrs)
            vrs = np.where(np.isfinite(vrs), vrs, -np.inf)
            best = vrs.max(axis=1)
            cols[f"vr_{tag}"] = np.where(np.isfinite(best), best, np.nan)
            cols[f"ks_{tag}"] = np.max(
                np.column_stack([p[f"ks_{tag}"] for p in pairs.values()]), axis=1
            )
        table = pd.DataFrame(cols, index=names)

    obs = {
        "unadjusted": {str(lv): float((G == k).sum()) for k, lv in enumerate(levels)}
    }
    if w is not None:
        obs["adjusted"] = {str(lv): _ess(w[G == k]) for k, lv in enumerate(levels)}
    observations = pd.DataFrame(obs).T
    summary = {
        "treat_type": "binary" if len(levels) == 2 else "multinomial",
        "estimand": estimand,
        "focal": focal,
        "s_d_denom": s_d_denom,
        "pairwise": bool(pairwise),
        "statistic": "mean difference",
        "threshold": float(threshold),
        "adjusted": w is not None,
        "n_obs": int(len(G)),
    }
    _flag(table, "diff", w is not None, threshold, summary)
    return BalTabResult(table, pairs if len(levels) > 2 else {}, observations, summary)


def _flag(
    table: pd.DataFrame, stat: str, adjusted: bool, threshold: float, summary: Dict
) -> None:
    tag = "adj" if adjusted else "un"
    vals = table[f"{stat}_{tag}"].abs()
    table["balanced"] = vals <= threshold
    summary[f"max_abs_{tag}"] = float(vals.max())
    summary[f"n_imbalanced_{tag}"] = int((vals > threshold).sum())
    if adjusted:
        un = table[f"{stat}_un"].abs()
        summary["max_abs_un"] = float(un.max())
        summary["n_imbalanced_un"] = int((un > threshold).sum())


def _continuous(
    a: np.ndarray,
    X: np.ndarray,
    names: List[str],
    is_bin: List[bool],
    w: Optional[np.ndarray],
    threshold: float,
) -> BalTabResult:
    ones = np.ones(len(a))
    ra = rankdata(a)
    sd_a, sd_ra = float(a.std(ddof=1)), float(ra.std(ddof=1))
    rows = []
    for j in range(X.shape[1]):
        x = X[:, j]
        rx = rankdata(x)
        sd_x, sd_rx = _sd(x, is_bin[j]), _sd(rx, is_bin[j])
        row: Dict[str, Any] = {"type": "binary" if is_bin[j] else "continuous"}
        for tag, wv in (("un", ones), ("adj", w)):
            if wv is None:
                continue
            row[f"corr_{tag}"] = _wcov(a, x, wv) / (sd_a * sd_x)
            row[f"scorr_{tag}"] = _wcov(ra, rx, wv) / (sd_ra * sd_rx)
        rows.append(row)
    table = pd.DataFrame(rows, index=names)
    obs = {"unadjusted": {"total": float(len(a))}}
    if w is not None:
        obs["adjusted"] = {"total": _ess(w)}
    summary = {
        "treat_type": "continuous",
        "estimand": None,
        "statistic": "treatment-covariate correlation",
        "threshold": float(threshold),
        "adjusted": w is not None,
        "n_obs": int(len(a)),
    }
    _flag(table, "corr", w is not None, threshold, summary)
    return BalTabResult(table, {}, pd.DataFrame(obs).T, summary)
