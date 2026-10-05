"""Descriptive and survey commands of a Stata session.

* ``tabulate`` / ``tab1`` / ``tab2`` with weights, percentages, expected
  counts, tests of association and ``summarize()``;
* ``mean`` / ``proportion`` / ``total`` / ``ratio`` with ``over()``,
  weights and ``vce(cluster)``;
* ``svyset`` and the ``svy:`` prefix for those four commands, ``tabulate``
  and the regressions ``sp.svyglm`` fits; ``estat effects``;
* ``ameans``, ``centile``, ``cii``, ``misstable summarize``.

The estimators follow the Methods and formulas of [R] mean / proportion /
total / ratio and [SVY] variance estimation. Without weights and clusters
Stata treats the groups of ``over()`` as separate simple random samples
(``s / sqrt(n)`` within the group); with probability weights, clusters or
``svy`` the variance is the linearized one, which is the design variance of
``sp.svydesign`` and is computed by the same code.
"""

from __future__ import annotations

import re
import warnings
from itertools import combinations
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from ...exceptions import StatsPAIError
from ._stata_datastep import _numlist, _split_options, row_mask
from ._stata_expr import StataExprError, sample_mask
from ._stata_lexer import StataParseError
from ._stata_lexer import parse as _parse
from ._stata_manage import _flag, _leftover, _valued

if TYPE_CHECKING:  # pragma: no cover
    from ._stata_run import StataSession

__all__ = ["run_describe", "by_groups"]

_WEIGHT = re.compile(
    r"\[\s*(aw|aweights?|pw|pweights?|fw|fweights?|iw|iweights?)\s*=\s*([^\]]+)\]",
    re.I,
)


def _cmd(line: str) -> Any:
    try:
        return _parse(line)
    except StataParseError as exc:
        raise StataExprError(str(exc)) from None


def _steps(session: "StataSession") -> Any:
    """The data steps of the session; a command of this module needs data."""
    if session._steps is None:
        raise StataExprError("no data in memory")
    return session._steps


def _data(session: "StataSession") -> pd.DataFrame:
    frame: pd.DataFrame = _steps(session).data
    return frame


def _split_weight(
    session: "StataSession", line: str
) -> Tuple[str, Optional[str], Optional[np.ndarray]]:
    """``line`` without its weight clause, the kind of weight (two letters)
    and the weights."""
    m = _WEIGHT.search(line)
    if m is None:
        return line, None, None
    from ._stata_expr import evaluate

    w = evaluate(m.group(2), _data(session), session.stored)
    if w.dtype == object:
        raise StataExprError("a weight must be numeric")
    if np.any(w[~np.isnan(w)] < 0):
        raise StataExprError("negative weights are not allowed")
    rest = (line[: m.start()] + " " + line[m.end() :]).strip()
    return rest, m.group(1).lower()[:2], np.asarray(w, dtype=float)


def _numeric(data: pd.DataFrame, name: str) -> np.ndarray:
    col = data[name]
    if not (pd.api.types.is_numeric_dtype(col) or pd.api.types.is_bool_dtype(col)):
        raise StataExprError(f"{name!r} is a string variable")
    return np.asarray(col.to_numpy(dtype=float, na_value=np.nan), dtype=float)


def _level_labels(session: "StataSession", name: str, levels: Any) -> List[Any]:
    """The value labels of ``levels`` of ``name`` (the level where none)."""
    steps = _steps(session)
    table = {}
    if steps is not None:
        table = steps._label_sets.get(steps._set_of.get(name, ""), {})
    out = []
    for lv in levels:
        key = int(lv) if float(lv) == int(lv) else lv
        out.append(table.get(key, key))
    return out


# ================================================================ tabulate
def _tab_options(options: Dict[str, Any]) -> Dict[str, Any]:
    spec: Dict[str, Any] = {}
    for full, shortest in (
        ("missing", 1), ("nolabel", 3), ("sort", 4), ("nofreq", 3), ("row", 1),
        ("column", 2), ("cell", 2), ("expected", 1), ("chi2", 2), ("exact", 1),
        ("lrchi2", 2), ("v", 1), ("gamma", 1), ("taub", 1), ("all", 3),
        ("nokey", 3), ("wrap", 1), ("plot", 1), ("means", 1), ("standard", 2),
        ("freq", 1), ("obs", 3), ("nomeans", 3), ("nostandard", 3), ("noobs", 3),
        ("cchi2", 2), ("clrchi2", 3), ("first", 5),
    ):  # fmt: skip
        spec[full] = _flag(options, full, shortest)
    spec["generate"] = _valued(options, "generate", 1)
    spec["summarize"] = _valued(options, "summarize", 2)
    for name in ("matcell", "matrow", "matcol", "subpop"):
        if _valued(options, name, 4) is not None and name == "subpop":
            raise StataExprError("tabulate, subpop() is not implemented")
    _leftover(options, "tabulate")
    if spec["all"]:
        for name in ("chi2", "lrchi2", "v", "gamma", "taub"):
            spec[name] = True
    return spec


def _cell_weights(
    kind: Optional[str], w: Optional[np.ndarray], keep: np.ndarray
) -> Any:
    """What each kept row adds to its cell: 1, its frequency weight, or its
    analytic weight scaled so that the weights add up to the row count."""
    if w is None:
        return np.ones(int(keep.sum()))
    held = w[keep]
    if kind == "fw":
        if np.any(held != np.round(held)):
            raise StataExprError("may not use noninteger frequency weights")
        return held
    if kind == "aw":
        return held * (len(held) / held.sum()) if held.sum() > 0 else held
    if kind == "iw":
        return held
    raise StataExprError("tabulate does not allow pweights (Stata: r(101))")


def _whole(counts: Any) -> bool:
    values = np.asarray(counts, dtype=float)
    return bool(
        np.all(
            np.abs(values - np.round(values)) <= 1e-9 * np.maximum(1.0, np.abs(values))
        )
    )


def _summary_table(groups: pd.Series, y: np.ndarray, w: np.ndarray) -> pd.DataFrame:
    """``tabulate g, summarize(y)``: mean, standard deviation and
    frequency of y by group, with the total."""

    def row(sel: np.ndarray) -> Dict[str, float]:
        yy, ww = y[sel], w[sel]
        n = ww.sum()
        if n <= 0:
            return {"Mean": np.nan, "Std. dev.": np.nan, "Freq.": 0.0}
        mean = float(np.sum(ww * yy) / n)
        k = n if _whole(ww) else len(yy)
        var = float(np.sum(ww * (yy - mean) ** 2) / n * k / (k - 1)) if k > 1 else 0.0
        return {"Mean": mean, "Std. dev.": float(np.sqrt(var)), "Freq.": float(n)}

    out = {}
    codes = groups.to_numpy()
    for level in sorted(pd.unique(codes[~pd.isna(codes)])):
        out[level] = row(codes == level)
    out["Total"] = row(~pd.isna(codes))
    table = pd.DataFrame(out).T
    table.index.name = groups.name
    return table


def _one_table(
    session: "StataSession",
    varlist: List[str],
    mask: np.ndarray,
    kind: Optional[str],
    w: Optional[np.ndarray],
    spec: Dict[str, Any],
) -> Any:
    from ...output.tab import _with_value_labels, association_tests

    data = _data(session)
    keep = mask.copy()
    if w is not None:
        keep &= ~np.isnan(w) & (w != 0)
    summarize = spec["summarize"]
    y = None
    if summarize is not None:
        y = _numeric(data, _steps(session).expand_varlist([summarize])[0])
        keep &= ~np.isnan(y)
    if not spec["missing"]:
        for v in varlist:
            keep &= data[v].notna().to_numpy()
            if data[v].dtype == object:
                keep &= (data[v] != "").to_numpy()
    rows = data.loc[keep, varlist]
    if not spec["nolabel"]:
        rows = _with_value_labels(rows, varlist)
    if spec["missing"]:
        rows = rows.copy()
        for v in varlist:
            if rows[v].isna().any() and v in _steps(session).coded:
                raise StataExprError(
                    f"tabulate, missing: {v!r} may hold extended missing "
                    "values, which the table lists one by one and the data "
                    "keep as one"
                )
            if rows[v].isna().any():
                # the missing values are a category of their own, shown last
                levels = [lv for lv in pd.unique(rows[v].dropna())]
                try:
                    levels = sorted(levels)
                except TypeError:
                    pass
                if isinstance(rows[v].dtype, pd.CategoricalDtype):
                    levels = list(rows[v].cat.categories)
                rows[v] = pd.Categorical(
                    rows[v].astype(object).where(rows[v].notna(), "."),
                    categories=list(levels) + ["."], ordered=True,
                )  # fmt: skip
    weights = np.asarray(_cell_weights(kind, w, keep), dtype=float)
    if y is not None:
        yk = y[keep]
        if len(varlist) == 1:
            table = _summary_table(rows[varlist[0]], yk, weights)
            session.stored["r"] = {
                "N": float(weights.sum()),
                "r": float(len(table) - 1),
            }
            return table
        a, b = rows[varlist[0]], rows[varlist[1]]
        pieces = {}
        for level in sorted(pd.unique(b.dropna())):
            sel = (b == level).to_numpy()
            pieces[level] = _summary_table(a[sel], yk[sel], weights[sel])
        pieces["Total"] = _summary_table(a, yk, weights)
        return pd.concat(pieces, axis=1)
    if len(varlist) == 1:
        series = rows[varlist[0]]
        counts = (
            pd.Series(weights, index=series.index)
            .groupby(series, dropna=True, observed=True, sort=True)
            .sum()
        )
        if isinstance(counts.index, pd.CategoricalIndex):
            counts.index = counts.index.astype(object)
        if spec["sort"]:
            counts = counts.sort_values(ascending=False, kind="stable")
        total = float(counts.sum())
        whole = _whole(counts)
        table = pd.DataFrame(
            {
                "Freq.": counts.astype(int) if whole else counts,
                "Percent": 100.0 * counts / total if total else np.nan,
            }
        )
        table["Cum."] = table["Percent"].cumsum()
        table.index.name = varlist[0]
        table.attrs["N"] = total
        session.stored["r"] = {"N": total, "r": float(len(counts))}
        return table
    a, b = rows[varlist[0]], rows[varlist[1]]
    counts = pd.crosstab(a, b, values=weights, aggfunc="sum", dropna=False).fillna(0.0)
    counts = counts.loc[counts.sum(axis=1) > 0, counts.sum(axis=0) > 0]
    whole = _whole(counts)
    if whole:
        counts = counts.round().astype(int)
    table = counts.copy()
    table["Total"] = counts.sum(axis=1)
    table.loc["Total"] = table.sum(axis=0)
    n = float(counts.to_numpy().sum())
    session.stored["r"] = {"N": n, "r": float(counts.shape[0]),
                           "c": float(counts.shape[1])}  # fmt: skip
    obs = table.to_numpy(dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        if spec["row"]:
            table.attrs["row"] = pd.DataFrame(
                100.0 * obs / obs[:, [-1]], index=table.index, columns=table.columns
            )
        if spec["column"]:
            table.attrs["column"] = pd.DataFrame(
                100.0 * obs / obs[[-1], :], index=table.index, columns=table.columns
            )
        if spec["cell"]:
            table.attrs["cell"] = pd.DataFrame(
                100.0 * obs / n, index=table.index, columns=table.columns
            )
        if spec["expected"]:
            table.attrs["expected"] = pd.DataFrame(
                obs[:, [-1]] * obs[[-1], :] / n, index=table.index,
                columns=table.columns,
            )  # fmt: skip
    wanted = [k for k in ("chi2", "exact", "lrchi2", "v", "gamma", "taub") if spec[k]]
    if wanted:
        if min(counts.shape) < 2:
            raise StataExprError("tabulate: a test needs two rows and two columns")
        tests = association_tests(counts)
        table.attrs["test"] = tests
        r = session.stored["r"]
        r.update(chi2=tests["chi2"], p=tests["pvalue"], chi2_lr=tests["chi2_lr"],
                 p_lr=tests["pvalue_lr"], CramersV=tests["cramers_v"],
                 gamma=tests["gamma"], ase_gam=tests["gamma_ase"],
                 taub=tests["taub"], ase_taub=tests["taub_ase"])  # fmt: skip
        if "fisher_pvalue" in tests:
            r.update(p_exact=tests["fisher_pvalue"],
                     p1_exact=tests["fisher_pvalue_1sided"])  # fmt: skip
        elif spec["exact"]:
            raise StataExprError(
                "tabulate, exact for a table larger than 2 x 2 (the "
                "Fisher-Freeman-Halton test) is not implemented"
            )
    return table


def _tabulate(session: "StataSession", line: str, word: str) -> bool:
    """``tabulate v1 [v2] [if] [in] [weight] [, options]``, ``tab1
    varlist`` (one table per variable) and ``tab2 varlist`` (one per pair)."""
    steps = _steps(session)
    if steps is None:
        raise StataExprError("`tabulate` needs data")
    line, kind, w = _split_weight(session, line)
    cmd = _cmd(line)
    spec = _tab_options(dict(cmd.options))
    data = steps.data
    varlist = steps.expand_varlist(list(cmd.varlist))
    mask = row_mask(data, cmd.if_cond, cmd.in_range, session.stored)
    if word in ("tab1", "tab2"):
        groups = (
            [[v] for v in varlist] if word == "tab1" else
            [list(pair) for pair in combinations(varlist, 2)]
        )  # fmt: skip
        if spec["generate"]:
            raise StataExprError(f"{word}, generate() is not implemented")
        session.output = {
            " ".join(g): _one_table(session, g, mask, kind, w, spec) for g in groups
        }
        return True
    if len(varlist) not in (1, 2):
        raise StataExprError("tabulate takes one or two variables")
    if spec["generate"] is not None:
        if len(varlist) != 1:
            raise StataExprError("tabulate, generate() takes one variable")
        steps.tabulate_generate(varlist[0], spec["generate"], mask)
    session.output = _one_table(session, varlist, mask, kind, w, spec)
    return True


# ====================================================== means and totals
class Estimates(pd.DataFrame):
    """The table an estimation command prints, with ``params`` /
    ``std_errors`` so that ``_b[]`` and ``test`` find the estimates."""

    _metadata = ["info"]

    @property
    def _constructor(self) -> Any:
        return Estimates

    @property
    def params(self) -> pd.Series:
        return self["estimate"]

    @property
    def std_errors(self) -> pd.Series:
        return self["se"]


def _design(
    frame: pd.DataFrame, setup: Dict[str, Any], weights: Optional[np.ndarray]
) -> Any:
    """The ``sp.svydesign`` of ``frame`` under a ``svyset`` / weight /
    cluster specification."""
    import statspai as sp

    work = frame.copy()
    work["__w"] = 1.0 if weights is None else weights
    rules = {None: None, "missing": None, "certainty": "certainty",
             "scaled": "average", "centered": "adjust"}  # fmt: skip
    rule = rules[setup.get("singleunit")]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        design = sp.svydesign(
            work,
            weights="__w",
            strata=setup.get("strata"),
            cluster=setup.get("psu"),
            fpc=setup.get("fpc"),
            nest=True,
            lonely_psu=rule or "certainty",
        )
        if setup.get("poststrata"):
            totals = work.groupby(setup["poststrata"])[setup["postweight"]].first()
            design = design.calibrate(
                margins={setup["poststrata"]: {k: float(v) for k, v in totals.items()}}
            )
    return design


def _lonely(design: Any) -> bool:
    strata, psu = design._strata_codes, design._psu_codes
    per = pd.Series(psu).groupby(strata).nunique()
    return bool((per == 1).any())


def _estimate(
    session: "StataSession",
    kind: str,
    body: str,
    *,
    svy: Optional[Dict[str, Any]] = None,
    subpop: Optional[str] = None,
) -> Estimates:
    """``mean`` / ``proportion`` / ``total`` / ``ratio`` of ``body``."""
    from ...survey.estimators import _design_vcov

    steps = _steps(session)
    if steps is None:
        raise StataExprError(f"`{kind}` needs data")
    body, wkind, w = _split_weight(session, body)
    ratios: List[Tuple[str, str, str]] = []
    if kind == "ratio":
        head, _, option_text = _split_options(body)
        pattern = r"\(?\s*(?:(\w+)\s*:\s*)?(\w+)\s*/\s*(\w+)\s*\)?"
        spans = list(re.finditer(pattern, head))
        if not spans:
            raise StataExprError("ratio: expected `ratio (y/x)`")
        for i, m in enumerate(spans, 1):
            ratios.append((m.group(1) or f"_ratio_{i}", m.group(2), m.group(3)))
        tail = head[spans[-1].end() :]
        cmd = _cmd(f"ratio _x {tail} ," + option_text)
        names: List[str] = []
    else:
        cmd = _cmd(f"{kind} {body}")
        names = steps.expand_varlist(list(cmd.varlist))
    options = dict(cmd.options)
    over = _valued(options, "over", 1)
    vce = _valued(options, "vce", 3)
    cluster = _valued(options, "cluster", 2)
    level = float(_valued(options, "level", 1) or 95)
    citype = (_valued(options, "citype", 3) or "logit").lower()
    _valued(options, "cformat", 3)
    for display_only in ("noheader", "nolegend", "coeflegend", "nolstretch"):
        _flag(options, display_only, 4)
    _leftover(options, kind)
    if vce:
        words = vce.split()
        if words[0].startswith("cl") and len(words) == 2:
            cluster = words[1]
        elif not (words[0].startswith("lin") or words[0] == "analytic"):
            raise StataExprError(f"{kind}: vce({vce}) is not implemented")
    if citype not in ("logit", "wald", "normal"):
        raise StataExprError(f"proportion: citype({citype}) is not implemented")
    data = steps.data
    over_vars = steps.expand_varlist(over.split()) if over else []
    used = names + [v for _, a, b in ratios for v in (a, b)]
    mask = row_mask(data, cmd.if_cond, cmd.in_range, session.stored)
    complete = np.ones(len(data), dtype=bool)
    for v in used + over_vars:
        complete &= ~np.isnan(_numeric(data, v))
    setup: Dict[str, Any] = dict(svy or {})
    if svy is not None:
        if wkind is not None or cluster:
            raise StataExprError("weights and clusters are set by svyset under svy")
        if setup.get("weights"):
            w, wkind = _numeric(data, setup["weights"]), "pw"
        for v in (setup.get("strata"), setup.get("psu"), setup.get("fpc"),
                  setup.get("poststrata"), setup.get("postweight")):  # fmt: skip
            if v:
                mask &= data[v].notna().to_numpy()
    elif cluster:
        setup["psu"] = steps.expand_varlist([cluster])[0]
        mask &= data[setup["psu"]].notna().to_numpy()
    if w is not None:
        mask &= ~np.isnan(w) & (w != 0)
    domain = np.ones(len(data), dtype=bool)
    if subpop:
        text = subpop.strip()
        if text.lower().startswith("if "):
            domain = sample_mask(text[3:], data, session.stored)
        else:
            var, _, cond = text.partition(" if ")
            domain = _numeric(data, var.strip()) != 0
            domain &= ~np.isnan(_numeric(data, var.strip()))
            if cond:
                domain &= sample_mask(cond, data, session.stored)
        # outside the subpopulation a missing value does not drop the row
        sample = mask & (complete | ~domain)
    else:
        sample = mask & complete
    frame = data.loc[sample].reset_index(drop=True)
    dom = domain[sample] & complete[sample]
    n = len(frame)
    if n == 0 or not dom.any():
        raise StataExprError("no observations")
    weights: Any = None if w is None else w[sample]
    if wkind == "fw":
        if np.any(weights != np.round(weights)):
            raise StataExprError("may not use noninteger frequency weights")
        reps = weights.astype(int)
        frame = frame.loc[frame.index.repeat(reps)].reset_index(drop=True)
        dom = np.repeat(dom, reps)
        weights, n = None, len(frame)
    classical = svy is None and wkind in (None, "aw") and not setup.get("psu")
    design = _design(frame, setup, weights)
    # a poststratified design carries the adjusted weights
    ww = np.asarray(design.weights, dtype=float)
    # one row of the result per (variable or level) x over-group
    over_codes: Any = None
    if over_vars:
        over_codes = frame[over_vars].apply(tuple, axis=1).to_numpy()
    groups: List[Tuple[Any, np.ndarray]] = [(None, dom)]
    if over_vars:
        levels = sorted(set(over_codes[dom]))
        groups = [
            (lv, dom & np.asarray([bool(c == lv) for c in over_codes])) for lv in levels
        ]
    rows: List[Tuple[Any, ...]] = []
    scores: List[np.ndarray] = []
    estimates: List[float] = []
    classical_se: List[float] = []

    def add(label: Tuple[Any, ...], y: np.ndarray, x: Any,
            sel: np.ndarray, what: str) -> None:  # fmt: skip
        wy = ww * sel
        # a row outside the group adds nothing, whatever it holds
        y = np.where(sel, y, 0.0)
        x = None if x is None else np.where(sel, x, 0.0)
        n_g = float(sel.sum())
        if what == "total":
            est = float(np.sum(wy * y))
            z = wy * y
            se_c = float(np.sqrt(n_g) * np.std(y[sel], ddof=1)) if n_g > 1 else np.nan
        elif what == "ratio":
            den = float(np.sum(wy * x))
            est = float(np.sum(wy * y) / den)
            z = wy * (y - est * x) / den
            se_c = float(np.sqrt(n_g / (n_g - 1) * np.sum(z**2))) if n_g > 1 else np.nan
        else:
            den = float(np.sum(wy))
            est = float(np.sum(wy * y) / den)
            z = wy * (y - est) / den
            if what == "proportion":
                se_c = float(np.sqrt(est * (1 - est) / n_g))
            elif n_g > 1:
                var = float(np.sum(wy * (y - est) ** 2) / den * n_g / (n_g - 1))
                se_c = float(np.sqrt(var / n_g))
            else:
                se_c = np.nan
        rows.append(label)
        estimates.append(est)
        scores.append(np.where(sel, z, 0.0))
        classical_se.append(se_c)

    for lv, sel in groups:
        if kind == "ratio":
            for name, a, b in ratios:
                add((name, lv), _numeric(frame, a), _numeric(frame, b), sel, "ratio")
        elif kind == "proportion":
            for name in names:
                y = _numeric(frame, name)
                for value in sorted(set(y[dom])):
                    add((name, value, lv), (y == value).astype(float), None, sel,
                        "proportion")  # fmt: skip
        else:
            for name in names:
                add((name, lv), _numeric(frame, name), None, sel, kind)
    est = np.asarray(estimates)
    lonely = svy is not None and setup.get("singleunit") in (None, "missing")
    if classical:
        se = np.asarray(classical_se)
        df = float(n - 1)
        vcov = np.diag(se**2)
        if len(rows) > 1:
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                full = _design_vcov(np.column_stack(scores), design)
            scale = se / np.sqrt(np.clip(np.diag(full), 1e-300, None))
            vcov = full * np.outer(scale, scale)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            vcov = _design_vcov(np.column_stack(scores), design)
        se = np.sqrt(np.diag(vcov))
        n_psu = int(design._psu_codes.max()) + 1
        n_strata = int(design._strata_codes.max()) + 1
        if subpop:
            strata_in = np.unique(design._strata_codes[dom])
            n_psu = np.unique(
                design._psu_codes[np.isin(design._strata_codes, strata_in)]
            ).size
            n_strata = strata_in.size
        df = float(n_psu - n_strata)
        if lonely and _lonely(design):
            # Stata's default: a stratum with one sampling unit leaves the
            # standard errors missing (singleunit(missing))
            se = np.full(len(se), np.nan)
            vcov = np.full(vcov.shape, np.nan)
    crit = stats.t.ppf(1 - (1 - level / 100) / 2, df) if df > 0 else np.nan
    lower, upper = est - crit * se, est + crit * se
    if kind == "proportion" and citype == "logit":
        with np.errstate(divide="ignore", invalid="ignore"):
            logit = np.log(est / (1 - est))
            half = crit * se / (est * (1 - est))
            lower = 1 / (1 + np.exp(-(logit - half)))
            upper = 1 / (1 + np.exp(-(logit + half)))
    index = pd.MultiIndex.from_tuples(
        rows, names=["variable", "level", "over"][: len(rows[0])]
        if kind == "proportion" else ["variable", "over"]
    )  # fmt: skip
    table = Estimates(
        {"estimate": est, "se": se, "ci_lower": lower, "ci_upper": upper}, index=index
    )
    info: Dict[str, Any] = {"N": float(n), "df": df, "kind": kind,
                            "vcov": vcov, "labels": rows}  # fmt: skip
    if not classical:
        info["N_psu"] = float(int(design._psu_codes.max()) + 1)
        info["N_strata"] = float(int(design._strata_codes.max()) + 1)
    if svy is not None:
        info["N_pop"] = float(np.sum(ww))
        if setup.get("poststrata"):
            info["N_poststrata"] = float(frame[setup["poststrata"]].nunique())
        if subpop:
            # strata without a member of the subpopulation are left out
            inside = np.isin(design._strata_codes, strata_in)
            info["N"] = float(inside.sum())
            info["N_pop"] = float(np.sum(ww[inside]))
            info["N_sub"] = float(dom.sum())
            info["N_subpop"] = float(np.sum(ww[dom]))
            info["N_strata"] = float(n_strata)
            info["N_psu"] = float(n_psu)
        info["scores"] = scores
        info["fpc"] = bool(setup.get("fpc"))
        info["weights"] = ww
        info["domains"] = [sel for _, sel in groups]
    table.info = info
    return table


def _store_estimates(
    session: "StataSession", table: Estimates, over: List[str]
) -> None:
    """_b[] / _se[] / e() after one of the four commands, under the names
    Stata gives the estimates (``income``, ``c.income@1.sex``, ``1.pia``)."""
    b: Dict[str, float] = {}
    se: Dict[str, float] = {}
    names: List[str] = []

    def number(v: Any) -> str:
        return str(int(v)) if float(v) == int(v) else repr(float(v))

    for label, (_, row) in zip(table.info["labels"], table.iterrows()):
        if table.info["kind"] == "proportion":
            name, value, level = label
            key = f"{number(value)}.{name}"
        else:
            name, level = label
            key = name
        keys = [key]
        if level is not None:
            where = "#".join(f"{number(v)}.{g}" for v, g in zip(level, over))
            head = key if table.info["kind"] == "proportion" else f"c.{name}"
            keys = [f"{head}@{where}"]
        for k in keys:
            b[k], se[k] = float(row["estimate"]), float(row["se"])
        names.append(keys[0])
    table.info["names"] = names
    session.stored["_b"], session.stored["_se"] = b, se
    e = {"N": table.info["N"], "df_r": table.info["df"]}
    for key in ("N_psu", "N_strata", "N_pop", "N_sub", "N_subpop"):
        if key in table.info:
            e[key] = table.info[key]
    session.stored["e"] = e


_ESTIMATORS = {"mean": 4, "proportion": 4, "total": 5, "ratio": 5}


def _estimator_word(word: str) -> Optional[str]:
    for full, shortest in _ESTIMATORS.items():
        if shortest <= len(word) <= len(full) and full.startswith(word):
            return full
    return None


def _run_estimate(
    session: "StataSession", kind: str, rest: str, *, svy: bool, subpop: Optional[str]
) -> bool:
    setup = None
    if svy:
        setup = getattr(session, "svy", None)
        if setup is None:
            raise StataExprError("data not set up for svy, use svyset")
    table = _estimate(session, kind, rest, svy=setup, subpop=subpop)
    over = re.search(r"\bover\(([^)]*)\)", rest)
    over_vars = _steps(session).expand_varlist(over.group(1).split()) if over else []
    _store_estimates(session, table, over_vars)
    session.output = table
    session.last = table
    session.last_data = _steps(session).data
    session._last_call = None
    return True


# ====================================================================== svy
_SVYSET = re.compile(r"\s*svyset\b(.*)\Z", re.S | re.I)
_SVY = re.compile(
    r"\s*svy\s*(?:(?:,\s*(?P<opts>[^:]*))|(?P<vce>\s+\w[^:,]*)"
    r"(?:,\s*(?P<opts2>[^:]*))?)?"
    r":\s*(?P<cmd>.+)\Z",
    re.S | re.I,
)


def _svyset(session: "StataSession", rest: str) -> bool:
    """``svyset [psu] [weight] [, strata() fpc() singleunit() poststrata()
    postweight()]``. Later stages (``|| ssu``) do not enter the linearized
    variance unless the first stage has an fpc, and are refused then."""
    m = _WEIGHT.search(rest)
    if m is not None:
        rest = (rest[: m.start()] + " " + rest[m.end() :]).strip()
    stages = [part.strip() for part in rest.split("||")]
    cmd = _cmd("svyset " + stages[0])
    options = dict(cmd.options)
    setup: Dict[str, Any] = {
        "strata": _valued(options, "strata", 3),
        "fpc": _valued(options, "fpc", 3),
        "singleunit": _valued(options, "singleunit", 6),
        "poststrata": _valued(options, "poststrata", 5),
        "postweight": _valued(options, "postweight", 5),
    }
    vce = _valued(options, "vce", 3)
    _valued(options, "dof", 3)
    _flag(options, "clear", 5)
    _flag(options, "noclear", 7)
    _leftover(options, "svyset")
    if vce and not vce.startswith("lin"):
        raise StataExprError(f"svyset, vce({vce}): only linearized is implemented")
    if setup["singleunit"] and setup["singleunit"] not in (
        "missing", "certainty", "scaled", "centered"
    ):  # fmt: skip
        raise StataExprError(f"svyset: singleunit({setup['singleunit']}) is unknown")
    if len(stages) > 1 and setup["fpc"]:
        raise StataExprError(
            "svyset with an fpc() and a second stage: the later stages then "
            "enter the variance, which is not implemented"
        )
    if bool(setup["poststrata"]) != bool(setup["postweight"]):
        raise StataExprError("svyset: poststrata() and postweight() go together")
    steps = _steps(session)
    if steps is None:
        raise StataExprError("svyset needs data")
    psu = [v for v in cmd.varlist if v not in ("_n",)]
    if len(psu) > 1:
        raise StataExprError("svyset: one sampling-unit variable per stage")
    setup["psu"] = steps.expand_varlist(psu)[0] if psu else None
    setup["weights"] = None
    if m is not None:
        if m.group(1).lower()[:2] not in ("pw", "iw"):
            raise StataExprError("svyset takes pweights")
        setup["weights"] = steps.expand_varlist([m.group(2).strip()])[0]
    for key in ("strata", "fpc", "poststrata", "postweight"):
        if setup[key]:
            setup[key] = steps.expand_varlist([setup[key]])[0]
    setattr(session, "svy", setup)
    return False


def _svy_table(
    session: "StataSession", rest: str, setup: Dict[str, Any], subpop: Optional[str]
) -> bool:
    """``svy: tabulate v1 [v2] [, row column count se ci]``: weighted cell
    proportions with linearized standard errors and, for a two-way table,
    Pearson's statistic with the Rao-Scott second-order correction."""
    from ...survey.estimators import _design_vcov

    steps = _steps(session)
    cmd = _cmd("tabulate " + rest)
    options = dict(cmd.options)
    spec = {k: _flag(options, k, n) for k, n in (
        ("row", 3), ("column", 3), ("cell", 3), ("count", 3), ("se", 2), ("ci", 2),
        ("obs", 3), ("pearson", 3), ("percent", 3), ("missing", 3), ("null", 4),
        ("nolabel", 5), ("nomarginals", 5), ("vertical", 4), ("deff", 4),
        ("deft", 4), ("lr", 2), ("wald", 4), ("llwald", 3), ("noadjust", 5),
    )}  # fmt: skip
    _valued(options, "format", 3)
    _valued(options, "level", 1)
    _leftover(options, "svy: tabulate")
    for refused in ("lr", "wald", "llwald", "deff", "deft", "missing"):
        if spec[refused]:
            raise StataExprError(f"svy: tabulate, {refused} is not implemented")
    data = steps.data
    varlist = steps.expand_varlist(list(cmd.varlist))
    if len(varlist) not in (1, 2):
        raise StataExprError("svy: tabulate takes one or two variables")
    mask = row_mask(data, cmd.if_cond, cmd.in_range, session.stored)
    w = _numeric(data, setup["weights"]) if setup.get("weights") else np.ones(len(data))
    mask &= ~np.isnan(w) & (w != 0)
    for v in varlist + [setup.get("strata"), setup.get("psu")]:
        if v:
            mask &= data[v].notna().to_numpy()
    if subpop:
        raise StataExprError("svy, subpop(): tabulate is not implemented")
    frame = data.loc[mask].reset_index(drop=True)
    ww = w[mask]
    design = _design(frame, setup, ww)
    a = _numeric(frame, varlist[0])
    b = _numeric(frame, varlist[1]) if len(varlist) == 2 else np.zeros(len(frame))
    rlev, clev = sorted(set(a)), sorted(set(b))
    total = ww.sum()
    cells = [(r, c) for r in rlev for c in clev]
    ind = np.column_stack([((a == r) & (b == c)).astype(float) for r, c in cells])
    p = (ww[:, None] * ind).sum(axis=0) / total
    z = ww[:, None] * (ind - p) / total
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        v = _design_vcov(z, design)
    lonely = setup.get("singleunit") in (None, "missing") and _lonely(design)
    shape = (len(rlev), len(clev))
    prop = p.reshape(shape)
    n_psu = int(design._psu_codes.max()) + 1
    n_strata = int(design._strata_codes.max()) + 1
    df = float(n_psu - n_strata)
    rows = _level_labels(session, varlist[0], rlev)
    cols = (
        _level_labels(session, varlist[1], clev)
        if len(varlist) == 2
        else ["proportion"]
    )
    out: Dict[str, Any] = {
        "N": float(len(frame)),
        "N_pop": float(total),
        "df": df,
        "N_psu": float(n_psu),
        "N_strata": float(n_strata),
    }

    def frame_of(values: np.ndarray) -> pd.DataFrame:
        table = pd.DataFrame(values, index=rows, columns=cols)
        table["Total"] = table.sum(axis=1)
        table.loc["Total"] = table.sum(axis=0)
        return table

    out["cell"] = frame_of(prop)
    out["count"] = frame_of(prop * total)
    with np.errstate(divide="ignore", invalid="ignore"):
        out["row"] = frame_of(prop)
        out["row"] = out["row"].div(out["row"]["Total"], axis=0)
        out["column"] = frame_of(prop)
        out["column"] = out["column"].div(out["column"].loc["Total"], axis=1)
    # standard errors of the displayed proportions, by the delta method
    se_cell = np.sqrt(np.diag(v)).reshape(shape)

    def ratio_se(num: np.ndarray, den: np.ndarray) -> np.ndarray:
        top = (num * ww[:, None]).sum(0)
        bottom = (den * ww[:, None]).sum(0)
        zz = ww[:, None] * (num - top / bottom * den) / bottom
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            return np.asarray(np.sqrt(np.diag(_design_vcov(zz, design))), dtype=float)

    row_ind = np.column_stack([(a == r).astype(float) for r, _ in cells])
    col_ind = np.column_stack([(b == c).astype(float) for _, c in cells])
    out["se"] = {
        "cell": pd.DataFrame(se_cell, index=rows, columns=cols),
        "row": pd.DataFrame(ratio_se(ind, row_ind).reshape(shape), index=rows,
                            columns=cols),
        "column": pd.DataFrame(ratio_se(ind, col_ind).reshape(shape), index=rows,
                               columns=cols),
    }  # fmt: skip
    # the margins: for row shares the Total row holds the column shares
    col_only = np.column_stack([(b == c).astype(float) for c in clev])
    row_only = np.column_stack([(a == r).astype(float) for r in rlev])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        margins = {}
        for key, ind_m, labels in (("row", col_only, cols), ("column", row_only, rows)):
            share = (ww[:, None] * ind_m).sum(axis=0) / total
            zz = ww[:, None] * (ind_m - share) / total
            margins[key] = pd.Series(np.sqrt(np.diag(_design_vcov(zz, design))),
                                     index=labels)  # fmt: skip
    out["se"]["row_total"] = margins["row"].to_frame("Total").T
    out["se"]["column_total"] = margins["column"].to_frame("Total")
    if lonely:
        out["se"] = {k: t * np.nan for k, t in out["se"].items()}
    # confidence limits on the logit scale, as `svy: tabulate, ci` prints
    crit = stats.t.ppf(0.975, df) if df > 0 else np.nan
    out["ci"] = {}
    for key, se_table in out["se"].items():
        est = out[key.split("_")[0]].loc[se_table.index, se_table.columns]
        with np.errstate(divide="ignore", invalid="ignore"):
            logit = np.log(est / (1 - est))
            half = crit * se_table / (est * (1 - est))
            out["ci"][key] = {"lower": 1 / (1 + np.exp(-(logit - half))),
                              "upper": 1 / (1 + np.exp(-(logit + half)))}  # fmt: skip
    if len(varlist) == 2 and min(shape) > 1:
        n = float(len(frame))
        p0 = np.outer(prop.sum(axis=1), prop.sum(axis=0)).ravel()
        chi2 = float(n * np.sum((p - p0) ** 2 / p0))
        out["chi2"] = chi2
        # Rao and Scott (1984): the interaction contrasts of the log-linear
        # model, their variance under the design and under simple random
        # sampling; F = chi2 / trace(delta) on (d, d * design df), with
        # d = trace(delta)^2 / trace(delta^2)
        r_dim, c_dim = shape
        main = [np.ones(r_dim * c_dim)]
        for i in range(r_dim - 1):
            main.append(np.repeat(np.eye(r_dim)[i] - np.eye(r_dim)[-1], c_dim))
        for j in range(c_dim - 1):
            main.append(np.tile(np.eye(c_dim)[j] - np.eye(c_dim)[-1], r_dim))
        x1 = np.column_stack(main)
        inter = []
        for i in range(r_dim - 1):
            for j in range(c_dim - 1):
                inter.append(main[1 + i] * main[r_dim + j])
        x2 = np.column_stack(inter)
        base = p if not spec["null"] else p0
        v_srs = (np.diag(base) - np.outer(base, base)) / n
        d_inv = np.diag(1.0 / base)
        # the contrasts of log p that the main effects leave free
        x2t = x2 - x1 @ np.linalg.lstsq(x1, x2, rcond=None)[0]
        left = x2t.T @ d_inv
        a_srs = left @ v_srs @ left.T
        a_des = left @ v @ left.T
        delta = np.linalg.solve(a_srs, a_des)
        tr, tr2 = float(np.trace(delta)), float(np.trace(delta @ delta))
        d = tr**2 / tr2
        out["F"] = chi2 / tr if not lonely else np.nan
        out["df1"], out["df2"] = d, d * df
        out["p"] = float(stats.f.sf(out["F"], d, d * df)) if not lonely else np.nan
    session.output = out
    session.stored["r"] = {k: v for k, v in out.items() if isinstance(v, float)}
    e = {"N": out["N"], "N_pop": out["N_pop"], "df_r": out["df"],
         "N_psu": out["N_psu"], "N_strata": out["N_strata"]}  # fmt: skip
    if "F" in out:
        e.update(cun_Pear=out["chi2"], F_Pear=out["F"], df1_Pear=out["df1"],
                 df2_Pear=out["df2"], p_Pear=out["p"])  # fmt: skip
    session.stored["e"] = e
    return True


def _svy(session: "StataSession", m: "re.Match[str]") -> bool:
    setup = getattr(session, "svy", None)
    if setup is None:
        raise StataExprError("data not set up for svy, use svyset")
    if m.group("vce") and not m.group("vce").strip().lower().startswith("lin"):
        raise StataExprError(
            f"svy {m.group('vce').strip()}: only the linearized variance is "
            "implemented"
        )
    subpop = None
    opts = (m.group("opts") or m.group("opts2") or "").strip()
    if opts:
        cmd = _cmd("svy _x ," + opts)
        options = dict(cmd.options)
        subpop = _valued(options, "subpop", 3)
        _valued(options, "level", 1)
        _flag(options, "noheader", 4)
        _flag(options, "nolegend", 4)
        _leftover(options, "svy")
    inner = m.group("cmd").strip()
    word, _, rest = inner.partition(" ")
    word = word.rstrip(",").lower()
    if inner.startswith(word + ","):
        rest = inner[len(word) :]
    kind = _estimator_word(word)
    if kind is not None:
        return _run_estimate(session, kind, rest, svy=True, subpop=subpop)
    if len(word) >= 2 and "tabulate".startswith(word):
        return _svy_table(session, rest, setup, subpop)
    return _svy_model(session, inner, setup, subpop)


def _svy_model(
    session: "StataSession", inner: str, setup: Dict[str, Any], subpop: Optional[str]
) -> bool:
    """``svy: regress / logit / logistic / poisson``: ``sp.svyglm``."""
    import statspai as sp

    from ._stata import from_stata

    steps = _steps(session)
    data = steps.data
    out = from_stata(inner, columns=list(data.columns))
    family = {"regress": "gaussian", "logit": "binomial", "poisson": "poisson"}.get(
        str(out.get("tool"))
    )
    lost = list(out.get("untranslated_options") or [])
    if not out.get("ok") or family is None or lost:
        raise StataExprError(
            "svy: the command is not one sp.svyglm fits (regress, logit, "
            f"poisson without further options): {inner!r}"
        )
    formula = str(out["arguments"]["formula"])
    cmd = _cmd(inner)
    mask = row_mask(data, cmd.if_cond, cmd.in_range, session.stored)
    w = _numeric(data, setup["weights"]) if setup.get("weights") else np.ones(len(data))
    mask &= ~np.isnan(w) & (w != 0)
    names = set(re.findall(r"[A-Za-z_]\w*", formula)) & set(map(str, data.columns))
    for v in list(names) + [setup.get("strata"), setup.get("psu")]:
        if v:
            mask &= data[v].notna().to_numpy()
    domain = None
    if subpop:
        text = subpop.strip()
        domain = sample_mask(text[3:] if text.lower().startswith("if ") else text,
                             data, session.stored)[mask]  # fmt: skip
    frame = data.loc[mask].reset_index(drop=True)
    if family == "binomial":
        outcome = _numeric(frame, formula.split("~")[0].strip())
        if len(set(outcome != 0)) < 2:
            raise StataExprError(
                "outcome does not vary (logit takes 0 against any other "
                "value); Stata stops here (r(2000))"
            )
    design = _design(frame, setup, w[mask])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        result = sp.svyglm(formula, design, family=family, subpop=domain)
    if setup.get("singleunit") in (None, "missing") and _lonely(design):
        result.std_error = result.std_error * np.nan
        result.ci_lower = result.ci_lower * np.nan
        result.ci_upper = result.ci_upper * np.nan
    info = {
        "t": [float(v) for v in result.t_values],
        "p": [float(v) for v in result.p_values],
        "N": float(len(frame)),
        "N_strata": float(int(design._strata_codes.max()) + 1),
        "N_psu": float(int(design._psu_codes.max()) + 1),
        "N_pop": float(np.sum(design.weights)),
        "df_r": float(result.dof),
    }
    if family == "gaussian":
        # R-squared of the weighted fit, as `svy: regress` prints it
        import patsy

        y_mat, x_mat = patsy.dmatrices(formula, frame, return_type="dataframe")
        beta = result.estimate.reindex(x_mat.columns).to_numpy(dtype=float)
        yv = y_mat.iloc[:, 0].to_numpy(dtype=float)
        wv = np.asarray(design.weights, dtype=float)
        resid = yv - x_mat.to_numpy(dtype=float) @ beta
        centre = np.sum(wv * yv) / wv.sum()
        info["r2"] = float(1 - np.sum(wv * resid**2) / np.sum(wv * (yv - centre) ** 2))
    setattr(result, "design_info", info)
    session.output = result
    session.last, session.last_data, session._last_call = result, frame, None
    session._store_estimates(result)
    session.stored["e"].update({k: v for k, v in info.items() if isinstance(v, float)})
    return True


def _estat_effects(session: "StataSession") -> bool:
    """``estat effects`` after ``svy: mean`` / ``proportion`` / ``total`` /
    ``ratio``: DEFF, the design variance over the variance a simple random
    sample of the same number of observations would have, and DEFT, the
    square root of that ratio for sampling with replacement. The two
    differ only when ``svyset`` names an fpc."""
    table = session.last
    info: Any = getattr(table, "info", None)
    if not isinstance(table, Estimates) or "scores" not in (info or {}):
        raise StataExprError(
            "estat effects follows svy: mean / proportion / total / ratio"
        )
    n, ww = info["N"], info["weights"]
    big_n = float(ww.sum())
    fraction = n / big_n if info.get("fpc") else 0.0
    deff, deft = [], []
    for z, se in zip(info["scores"], table["se"]):
        u = z / ww  # the linearized variable: the estimate moves by sum(w u)
        centre = float(np.sum(ww * u) / big_n)
        s2 = float(n / (n - 1) * np.sum(ww * (u - centre) ** 2) / big_n)
        v_wr = big_n**2 * s2 / n
        deff.append(float(se**2 / ((1 - fraction) * v_wr)) if v_wr > 0 else np.nan)
        deft.append(float(np.sqrt(se**2 / v_wr)) if v_wr > 0 else np.nan)
    session.output = pd.DataFrame(
        {"estimate": table["estimate"], "se": table["se"], "DEFF": deff, "DEFT": deft},
        index=table.index,
    )
    return True


# ========================================================= smaller commands
def _ameans(session: "StataSession", rest: str) -> bool:
    """``ameans varlist``: arithmetic, geometric and harmonic means with
    their confidence intervals."""
    rest, kind, w = _split_weight(session, rest)
    if kind is not None:
        raise StataExprError("ameans with weights is not implemented")
    steps = _steps(session)
    cmd = _cmd("ameans " + rest)
    options = dict(cmd.options)
    level = float(_valued(options, "level", 1) or 95)
    _leftover(options, "ameans")
    data = steps.data
    mask = row_mask(data, cmd.if_cond, cmd.in_range, session.stored)
    rows = {}
    for name in steps.expand_varlist(list(cmd.varlist) or ["_all"]):
        if not pd.api.types.is_numeric_dtype(data[name]):
            continue
        x = _numeric(data, name)[mask]
        x = x[~np.isnan(x)]

        def interval(v: np.ndarray, back: Any) -> Tuple[float, float, float, int]:
            n = v.size
            if n < 2:
                return (float(back(v.mean())) if n else np.nan, np.nan, np.nan, n)
            half = (
                stats.t.ppf(1 - (1 - level / 100) / 2, n - 1) * v.std(ddof=1) / n**0.5
            )
            lo, hi = back(v.mean() - half), back(v.mean() + half)
            return float(back(v.mean())), float(min(lo, hi)), float(max(lo, hi)), n

        pos = x[x > 0]
        for label, values, back in (
            ("Arithmetic", x, lambda t: t),
            ("Geometric", np.log(pos), np.exp),
            ("Harmonic", 1.0 / pos, lambda t: 1.0 / t if t > 0 else np.nan),
        ):
            mean, lo, hi, n = interval(values, back)
            rows[(name, label)] = {"Obs": n, "Mean": mean, "ci_lower": lo,
                                   "ci_upper": hi}  # fmt: skip
    session.output = pd.DataFrame(rows).T
    return True


def _centile(session: "StataSession", rest: str) -> bool:
    """``centile varlist [, centile(numlist)]``: with ``R = (n + 1) c /
    100`` the centile interpolates between the order statistics around R;
    the default interval is the binomial-based one with interpolation."""
    steps = _steps(session)
    cmd = _cmd("centile " + rest)
    options = dict(cmd.options)
    spec = _valued(options, "centile", 1)
    level = float(_valued(options, "level", 1) or 95)
    for unsupported in ("cci", "normal", "meansd"):
        if _flag(options, unsupported, 2):
            raise StataExprError(f"centile, {unsupported} is not implemented")
    _leftover(options, "centile")
    wanted = _numlist(spec, "centile") if spec else [50.0]
    data = steps.data
    mask = row_mask(data, cmd.if_cond, cmd.in_range, session.stored)
    rows = {}
    z = stats.norm.ppf(1 - (1 - level / 100) / 2)
    for name in steps.expand_varlist(list(cmd.varlist) or ["_all"]):
        if not pd.api.types.is_numeric_dtype(data[name]):
            continue
        x = np.sort(_numeric(data, name)[mask])
        x = x[~np.isnan(x)]
        n = x.size

        def at(position: float) -> float:
            i = int(np.floor(position))
            if i < 1:
                return float(x[0])
            if i >= n:
                return float(x[-1])
            return float(x[i - 1] + (position - i) * (x[i] - x[i - 1]))

        for c in wanted:
            if n == 0:
                rows[(name, c)] = {"Obs": 0, "Percentile": c, "Centile": np.nan,
                                   "ci_lower": np.nan, "ci_upper": np.nan}  # fmt: skip
                continue
            q = c / 100.0
            # Mood and Graybill: the ranks whose binomial coverage brackets
            # the level, interpolated between
            lo_rank = n * q - z * np.sqrt(n * q * (1 - q))
            hi_rank = n * q + z * np.sqrt(n * q * (1 - q))
            dist = stats.binom(n, q)

            def bound(alpha: float, upper: bool) -> float:
                # the rank r with P(X <= r - 1) crossing alpha, interpolated
                ks = np.arange(0, n + 1)
                cdf = dist.cdf(ks)
                if not upper:
                    r = int(np.searchsorted(cdf, alpha, side="right"))
                    if r == 0:
                        return float(x[0])
                    p0, p1 = cdf[r - 1], cdf[r] if r <= n else 1.0
                    frac = (alpha - p0) / (p1 - p0) if p1 > p0 else 0.0
                    return at(r + frac) if r < n else float(x[-1])
                r = int(np.searchsorted(cdf, 1 - alpha, side="left"))
                if r >= n:
                    return float(x[-1])
                p0 = cdf[r - 1] if r > 0 else 0.0
                p1 = cdf[r]
                frac = (1 - alpha - p0) / (p1 - p0) if p1 > p0 else 0.0
                return at(r + frac)

            del lo_rank, hi_rank
            alpha = (1 - level / 100) / 2
            rows[(name, c)] = {
                "Obs": n,
                "Percentile": c,
                "Centile": at((n + 1) * q),
                "ci_lower": bound(alpha, False),
                "ci_upper": bound(alpha, True),
            }
    table = pd.DataFrame(rows).T
    session.output = table
    r = {"N": float(table["Obs"].iloc[-1]), "n_cent": float(len(wanted))}
    for i, (_, row) in enumerate(table.iloc[-len(wanted) :].iterrows(), 1):
        r[f"c_{i}"] = float(row["Centile"])
        r[f"lb_{i}"], r[f"ub_{i}"] = float(row["ci_lower"]), float(row["ci_upper"])
    session.stored["r"] = r
    return True


def _cii(session: "StataSession", rest: str) -> bool:
    """``cii means n mean sd`` and ``cii proportions n successes``."""
    import statspai as sp

    cmd = _cmd("cii " + rest)
    options = dict(cmd.options)
    level = float(_valued(options, "level", 1) or 95)
    words = list(cmd.varlist)
    if not words:
        raise StataExprError("cii: expected `cii means #obs #mean #sd`")
    sub = words[0].lower()
    try:
        numbers = [float(t) for t in words[1:]]
    except ValueError:
        raise StataExprError("cii: numbers are expected") from None
    alpha = 1 - level / 100
    if "means".startswith(sub) and len(sub) >= 4 and len(numbers) == 3:
        _leftover(options, "cii means")
        n, mean, sd = numbers
        se = sd / np.sqrt(n)
        half = stats.t.ppf(1 - alpha / 2, n - 1) * se
        session.output = pd.DataFrame(
            {"n": [n], "mean": [mean], "se": [se], "ci_lower": [mean - half],
             "ci_upper": [mean + half]}
        )  # fmt: skip
        return True
    if "proportions".startswith(sub) and len(sub) >= 4 and len(numbers) == 2:
        method = "exact"
        for name in ("exact", "wald", "wilson", "agresti", "jeffreys"):
            if _flag(options, name, 3):
                method = name
        _leftover(options, "cii proportions")
        n, k = numbers
        frame = pd.DataFrame({"y": np.r_[np.ones(int(k)), np.zeros(int(n - k))]})
        interval = getattr(sp, "ci")
        session.output = interval(
            frame, ["y"], stat="proportions", method=method, alpha=alpha
        )
        return True
    raise StataExprError(f"`cii {rest}` is not implemented")


def _misstable(session: "StataSession", rest: str) -> Optional[bool]:
    """``misstable summarize [varlist]``: the number of missing and of
    observed values of each variable that has missing values."""
    words = rest.split(None, 1)
    if not words or not "summarize".startswith(words[0].lower()):
        raise StataExprError("only `misstable summarize` is implemented")
    steps = _steps(session)
    cmd = _cmd("misstable " + (words[1] if len(words) > 1 else ""))
    data = steps.data
    mask = row_mask(data, cmd.if_cond, cmd.in_range, session.stored)
    rows = {}
    for name in steps.expand_varlist(list(cmd.varlist) or ["_all"]):
        col = data.loc[mask, name]
        if not pd.api.types.is_numeric_dtype(col):
            continue
        missing = int(col.isna().sum())
        if not missing:
            continue
        if name in steps.coded:
            raise StataExprError(
                f"misstable: {name!r} may hold extended missing values, "
                "which the table counts apart from `.` and the data do not"
            )
        held = col.dropna()
        rows[name] = {"missing": missing, "observed": int(held.size),
                      "unique": int(held.nunique()), "min": float(held.min()),
                      "max": float(held.max())}  # fmt: skip
    session.output = pd.DataFrame(rows).T
    return True


_TABLE_STATS = {
    "mean": "mean", "sd": "sd", "median": "median", "min": "min", "max": "max",
    "count": "n", "total": "sum", "sum": "sum", "variance": "variance",
    "semean": "semean", "cv": "cv", "skewness": "skewness",
    "kurtosis": "kurtosis", "iqr": "iqr", "range": "range",
    "q1": "p25", "q2": "median", "q3": "p75",
}  # fmt: skip


def _table(session: "StataSession", rest: str, raw: str) -> Optional[bool]:
    """``table (rowvars) [(colvar)], statistic(stat varlist) ...`` (Stata
    17 and later): statistics by the levels of one or two variables, with
    the totals. ``statistic(frequency)`` counts the rows of a cell,
    ``statistic(percent)`` their share of all rows."""
    import statspai as sp

    steps = _steps(session)
    rest, kind, w = _split_weight(session, rest)
    if kind is not None:
        raise StataExprError("table with weights is not implemented")
    specs = re.findall(r"\bstat(?:i(?:s(?:t(?:ic?)?)?)?)?\(([^)]*)\)", raw, re.I)
    cmd = _cmd("table " + rest)
    options = {k: v for k, v in dict(cmd.options).items()
               if not (k and len(k) >= 4 and "statistic".startswith(k))}  # fmt: skip
    nototals = _flag(options, "nototals", 5)
    _valued(options, "nformat", 3)
    _valued(options, "sformat", 3)
    _flag(options, "missing", 4)
    _leftover(options, "table")
    dims = [v for v in cmd.varlist if v not in ("(", ")", "()")]
    dims = [v.strip("()") for v in dims if v.strip("()")]
    data = steps.data
    dims = steps.expand_varlist(dims) if dims else []
    if not 1 <= len(dims) <= 2:
        raise StataExprError("table: one or two grouping variables are implemented")
    if not specs:
        specs = ["frequency"]
    mask = row_mask(data, cmd.if_cond, cmd.in_range, session.stored)
    for v in dims:
        mask &= data[v].notna().to_numpy()
    frame = data.loc[mask]
    keys = [frame[v] for v in dims]

    def cells(sub: pd.DataFrame) -> Dict[str, float]:
        out: Dict[str, float] = {}
        for spec in specs:
            words = spec.split()
            low = words[0].lower()
            if low in ("frequency", "freq"):
                out["frequency"] = float(len(sub))
                continue
            if low == "percent":
                out["percent"] = 100.0 * len(sub) / len(frame) if len(frame) else np.nan
                continue
            stat = _TABLE_STATS.get(low) or (
                low if re.fullmatch(r"p\d{1,2}", low) else None
            )
            if stat is None or len(words) < 2:
                raise StataExprError(f"table: statistic({spec}) is not implemented")
            names = steps.expand_varlist(words[1:])
            got: Any = sp.sumstats(sub, vars=names, stats=[stat], output="numeric",
                                   percentile_method="stata")  # fmt: skip
            for name in names:
                out[f"{low} {name}"] = float(got.loc[name].iloc[0])
        return out

    rows: Dict[Any, Dict[str, float]] = {}
    grouped = frame.groupby(dims if len(dims) > 1 else dims[0], sort=True)
    for level, sub in grouped:
        rows[level] = cells(sub)
    if not nototals:
        if len(dims) == 2:
            for level, sub in frame.groupby(dims[0], sort=True):
                rows[(level, "Total")] = cells(sub)
            for level, sub in frame.groupby(dims[1], sort=True):
                rows[("Total", level)] = cells(sub)
            rows[("Total", "Total")] = cells(frame)
        else:
            rows["Total"] = cells(frame)
    del keys
    table = pd.DataFrame(rows).T
    table.index.names = dims if len(dims) > 1 else [dims[0]]
    session.output = table
    return True


def _linear_after_estimates(session: "StataSession", word: str, rest: str) -> bool:
    """``lincom exp`` and ``test exp = exp`` after ``mean`` / ``proportion``
    / ``total`` / ``ratio``: a linear combination of the estimates, with
    the covariance matrix of the command and its degrees of freedom."""
    table = session.last
    info: Any = table.info
    names: List[str] = info["names"]
    cov = np.asarray(info["vcov"], dtype=float)
    body = rest.split(",")[0].strip()
    if word == "test":
        if body.startswith("(") or body.count("=") != 1:
            raise StataExprError(
                "after mean / proportion one equality is tested: "
                "`test _b[a] = _b[b]`"
            )
        left, right = body.split("=")
        body = f"({left}) - ({right})"
    for name in sorted(names, key=len, reverse=True):
        # a bare name is the estimate: c.y@1.g is _b[c.y@1.g]
        body = re.sub(
            rf"(?<![\w\[.@]){re.escape(name)}(?![\w\].@])", f"_b[{name}]", body
        )
    kept = dict(session.stored.get("_b") or {})

    def at(values: Dict[str, float]) -> float:
        session.stored["_b"] = values
        try:
            return session.value(body)
        finally:
            session.stored["_b"] = kept

    zero = {k: 0.0 for k in kept}
    constant = at(zero)
    weights = np.array([at({**zero, n: 1.0}) - constant for n in names])
    scaled = at({k: 2.0 * v for k, v in kept.items()}) - constant
    estimate = at(kept)
    if not np.isclose(scaled, 2.0 * (estimate - constant), rtol=1e-9, atol=1e-12):
        raise StataExprError(f"{word}: the expression is not linear in the estimates")
    variance = float(weights @ cov @ weights)
    se = float(np.sqrt(variance)) if variance >= 0 else np.nan
    df = float(info["df"])
    if word == "test":
        stat = estimate**2 / variance if variance > 0 else np.nan
        p_value = float(stats.f.sf(stat, 1, df))
        # the keys of sp.test's result, and Stata's r() names
        out = {
            "statistic": float(stat),
            "pvalue": p_value,
            "df": (1, df),
            "distribution": "F",
            "F": float(stat),
            "df_r": df,
            "p": p_value,
        }
    else:
        crit = stats.t.ppf(0.975, df)
        t = estimate / se
        out = {
            "estimate": float(estimate),
            "se": se,
            "t": float(t),
            "p": float(2 * stats.t.sf(abs(t), df)),
            "df": df,
            "lb": float(estimate - crit * se),
            "ub": float(estimate + crit * se),
        }
    session.output = out
    session.stored["r"] = {k: v for k, v in out.items() if isinstance(v, float)}
    if word == "test":
        session.stored["r"]["df"] = 1.0
    return True


def _exact_odds_limits(
    a: float, b: float, c: float, d: float, alpha: float
) -> Tuple[float, float]:
    """Exact confidence limits of the odds ratio of a 2 x 2 table: the odds
    ratios at which the conditional (Fisher noncentral hypergeometric)
    probability of a table at least as extreme is alpha / 2 on each side."""
    from scipy.optimize import brentq

    x, row, col, n = (
        int(round(a)),
        int(round(a + b)),
        int(round(a + c)),
        int(round(a + b + c + d)),
    )

    def tail(log_odds: float, upper: bool) -> float:
        dist = stats.nchypergeom_fisher(n, col, row, np.exp(log_odds))
        return float(dist.sf(x - 1) if upper else dist.cdf(x))

    lowest, highest = max(0, row + col - n), min(row, col)
    low = 0.0
    if x > lowest:
        low = float(np.exp(brentq(lambda t: tail(t, True) - alpha / 2, -30, 30,
                                  xtol=1e-13, rtol=1e-13)))  # fmt: skip
    high = np.inf
    if x < highest:
        high = float(np.exp(brentq(lambda t: tail(t, False) - alpha / 2, -30, 30,
                                   xtol=1e-13, rtol=1e-13)))  # fmt: skip
    return low, high


def _epitab(session: "StataSession", word: str, rest: str) -> bool:
    """``cc case exposed`` (case-control: the odds ratio with its exact
    confidence interval) and ``cs case exposed`` (cohort: risk difference
    and risk ratio), with the attributable fractions and Pearson's
    chi-squared. Both variables are 0 / 1."""
    rest, kind, w = _split_weight(session, rest)
    steps = _steps(session)
    cmd = _cmd(f"{word} {rest}")
    options = dict(cmd.options)
    level = float(_valued(options, "level", 1) or 95)
    _flag(options, "exact", 1)
    _leftover(options, word)
    if kind not in (None, "fw"):
        raise StataExprError(f"{word} takes frequency weights")
    names = steps.expand_varlist(list(cmd.varlist))
    if len(names) != 2:
        raise StataExprError(f"{word}: expected `{word} case exposed`")
    data = steps.data
    mask = row_mask(data, cmd.if_cond, cmd.in_range, session.stored)
    case, exposed = _numeric(data, names[0]), _numeric(data, names[1])
    keep = mask & ~np.isnan(case) & ~np.isnan(exposed)
    weight = np.ones(len(data)) if w is None else np.where(np.isnan(w), 0.0, w)
    case, exposed, weight = case[keep] != 0, exposed[keep] != 0, weight[keep]
    a = float(weight[case & exposed].sum())
    b = float(weight[case & ~exposed].sum())
    c = float(weight[~case & exposed].sum())
    d = float(weight[~case & ~exposed].sum())
    n = a + b + c + d
    if min(a + b, c + d, a + c, b + d) <= 0:
        raise StataExprError(f"{word}: a margin of the 2 x 2 table is empty")
    chi2 = n * (a * d - b * c) ** 2 / ((a + b) * (c + d) * (a + c) * (b + d))
    out: Dict[str, Any] = {"N": n, "chi2": float(chi2),
                           "p": float(stats.chi2.sf(chi2, 1))}  # fmt: skip
    z = stats.norm.ppf(1 - (1 - level / 100) / 2)
    if word == "cc":
        ratio = a * d / (b * c) if b * c > 0 else np.nan
        low, high = _exact_odds_limits(a, b, c, d, 1 - level / 100)
        out.update(
            **{"or": float(ratio)}, lb_or=low, ub_or=high,
            p1_exposed=a / (a + b), p0_exposed=c / (c + d),
        )  # fmt: skip
        if ratio >= 1:
            afe = (ratio - 1) / ratio
            out.update(afe=float(afe), afp=float(afe * a / (a + b)),
                       lb_afe=float((low - 1) / low) if low > 0 else np.nan,
                       ub_afe=float((high - 1) / high))  # fmt: skip
        else:
            out.update(pfe=float(1 - ratio), pfp=float((1 - ratio) * c / (c + d)),
                       lb_pfe=float(1 - high), ub_pfe=float(1 - low))  # fmt: skip
        out["p_exposed"] = (a + c) / n
    else:
        n1, n0 = a + c, b + d
        r1, r0 = a / n1, b / n0
        rd = r1 - r0
        se_rd = np.sqrt(r1 * (1 - r1) / n1 + r0 * (1 - r0) / n0)
        rr = r1 / r0 if r0 > 0 else np.nan
        se_log = np.sqrt(1 / a - 1 / n1 + 1 / b - 1 / n0) if a > 0 and b > 0 else np.nan
        lo, hi = rr * np.exp(-z * se_log), rr * np.exp(z * se_log)
        out.update(
            risk1=float(r1), risk0=float(r0), risk=float((a + b) / n),
            rd=float(rd), lb_rd=float(rd - z * se_rd), ub_rd=float(rd + z * se_rd),
            rr=float(rr), lb_rr=float(lo), ub_rr=float(hi),
        )  # fmt: skip
        if rr >= 1:
            out.update(afe=float((rr - 1) / rr), lb_afe=float((lo - 1) / lo),
                       ub_afe=float((hi - 1) / hi),
                       afp=float(((a + b) / n - r0) / ((a + b) / n)))  # fmt: skip
        else:
            out.update(pfe=float(1 - rr), lb_pfe=float(1 - hi), ub_pfe=float(1 - lo),
                       pfp=float((r0 - (a + b) / n) / r0))  # fmt: skip
    counts = pd.DataFrame([[a, b], [c, d]], index=["cases", "noncases"],
                          columns=["exposed", "unexposed"])  # fmt: skip
    counts["total"] = counts.sum(axis=1)
    counts.loc["total"] = counts.sum(axis=0)
    out["table"] = counts
    session.output = out
    session.stored["r"] = {k: v for k, v in out.items() if isinstance(v, float)}
    return True


def _anova(session: "StataSession", rest: str) -> bool:
    """``anova y g``: the one-way layout, which is ``oneway``. Models with
    several factors or interactions are ``regress`` with ``testparm``."""
    from ...inference.rank_tests import oneway

    steps = _steps(session)
    cmd = _cmd("anova " + rest)
    names = list(cmd.varlist)
    if len(names) != 2 or cmd.options or any(ch in names[1] for ch in "#|."):
        raise StataExprError(
            "anova with several terms is not implemented; fit it with "
            "`regress y i.a i.b` and test each factor with `testparm`"
        )
    names = steps.expand_varlist(names)
    data = steps.data.loc[
        row_mask(steps.data, cmd.if_cond, cmd.in_range, session.stored)
    ]
    out = oneway(data, names[0], by=names[1])
    session.output = out
    session.stored["r"] = {}
    session.stored["e"] = {
        "N": float(out.n_obs), "F": out.statistic, "r2": out.estimates["r2"],
        "rmse": out.estimates["rmse"], "mss": out.estimates["ss_between"],
        "rss": out.estimates["ss_within"], "df_m": float(out.estimates["df_between"]),
        "df_r": float(out.estimates["df_within"]),
    }  # fmt: skip
    return True


def _statsby(session: "StataSession", rest: str) -> bool:
    """``statsby name = exp ..., by(varlist) clear: command``: the command
    run in each group, and a dataset of the named results by group."""
    head, colon, command = rest.partition(":")
    if not colon:
        raise StataExprError(
            "statsby: expected `statsby exps, by(vars) clear: command`"
        )
    spec, _, option_text = head.partition(",")
    cmd = _cmd("statsby _x ," + option_text)
    options = dict(cmd.options)
    by = _valued(options, "by", 2)
    clear = _flag(options, "clear", 5)
    for display_only in ("nodots", "nolegend", "noisily", "verbose", "total"):
        if _flag(options, display_only, 4) and display_only == "total":
            raise StataExprError("statsby, total is not implemented")
    _leftover(options, "statsby")
    pairs = re.findall(r"([A-Za-z_]\w*)\s*=\s*(\([^)]*\)|\S+)", spec)
    if not pairs or not by or not clear:
        raise StataExprError(
            "statsby needs named results (`name = exp`), by() and clear"
        )
    steps = _steps(session)
    keys = steps.expand_varlist(by.split())
    full, owned = steps.data, steps._owned
    rows = []
    try:
        for level, part in full.groupby(keys if len(keys) > 1 else keys[0], sort=True):
            steps.data = part.reset_index(drop=True)
            steps.data.attrs.update(full.attrs)
            steps._owned = False
            steps._original = steps.data
            row = dict(zip(keys, level if isinstance(level, tuple) else (level,)))
            try:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    session.run(command.strip())
                for name, expr in pairs:
                    row[name] = session.value(expr)
            except (StatsPAIError, StataExprError):
                # a group the command cannot fit: Stata posts missing values
                for name, _ in pairs:
                    row[name] = np.nan
            rows.append(row)
    finally:
        steps.data, steps._owned = full, owned
    steps.replace_data(pd.DataFrame(rows, columns=keys + [n for n, _ in pairs]))
    return False


def _covariance(session: "StataSession", rest: str) -> bool:
    """``correlate varlist, covariance``: the covariance matrix on the
    rows where every variable is observed."""
    rest, kind, w = _split_weight(session, rest)
    steps = _steps(session)
    cmd = _cmd("correlate " + rest)
    options = dict(cmd.options)
    _flag(options, "covariance", 1)
    _flag(options, "means", 1)
    _leftover(options, "correlate")
    if kind not in (None, "aw", "fw"):
        raise StataExprError("correlate takes aweights and fweights")
    data = steps.data
    names = steps.expand_varlist(list(cmd.varlist) or ["_all"])
    names = [v for v in names if pd.api.types.is_numeric_dtype(data[v])]
    mask = row_mask(data, cmd.if_cond, cmd.in_range, session.stored)
    x = np.column_stack([_numeric(data, v) for v in names])
    keep = mask & ~np.isnan(x).any(axis=1)
    weights = np.ones(len(data)) if w is None else w
    keep &= ~np.isnan(weights) & (weights > 0)
    x, weights = x[keep], weights[keep]
    n = float(weights.sum()) if kind == "fw" else float(len(x))
    centred = x - (weights[:, None] * x).sum(axis=0) / weights.sum()
    cov = (centred * weights[:, None]).T @ centred / weights.sum() * n / (n - 1)
    table = pd.DataFrame(cov, index=names, columns=names)
    table.attrs["N"] = n
    session.output = table
    session.stored["r"] = {
        "N": n,
        "cov_12": float(cov[0, 1]) if len(names) > 1 else np.nan,
        "Var_1": float(cov[0, 0]),
    }
    return True


# ---------------------------------------------------------------- by groups
def check_by_keys(steps: Any, names: List[str]) -> None:
    """Refuse ``by`` on a variable whose missing values are of several
    kinds: Stata makes a group of each (``.``, ``.a``, ``.b`` ...) and
    sorts them in that order, and the data keep them as one."""
    for name in names:
        if name in steps.coded and steps.data[name].isna().any():
            raise StataExprError(
                f"by: {name!r} may hold extended missing values (.a-.z); "
                "Stata groups and sorts each kind apart, and the data keep "
                "them as one. Drop or recode the missing rows first"
            )


def by_groups(
    session: "StataSession", keys: List[str], sort: bool, order: List[str]
) -> List[Tuple[Any, np.ndarray]]:
    """The groups of ``by keys:``: (level, row positions) in sorted order.

    Without ``sort`` the data must already be in runs of the keys.
    """
    steps = _steps(session)
    unknown = [k for k in keys + order if k not in steps.data.columns]
    if unknown:
        raise StataExprError(f"by: variable(s) {unknown} are not in the data")
    if sort:
        steps._sort(keys + order, None)
    data = steps.data
    check_by_keys(steps, keys + order)
    codes = data.groupby(keys, sort=False, dropna=False).ngroup().to_numpy()
    starts: Any = (
        np.flatnonzero(np.r_[True, codes[1:] != codes[:-1]]) if len(codes) else []
    )
    if len(starts) != len(np.unique(codes)):
        raise StataExprError(
            "not sorted: `by` needs the data sorted by " + " ".join(keys)
        )
    bounds = np.r_[starts, len(codes)]
    out = []
    for i in range(len(starts)):
        rows = np.arange(bounds[i], bounds[i + 1])
        level = tuple(data.iloc[rows[0]][keys])
        out.append((level if len(keys) > 1 else level[0], rows))
    return out


_HEAD = re.compile(r"\s*([A-Za-z_]\w*)\b\s*(.*)\Z", re.S)


def run_describe(session: "StataSession", line: str) -> Optional[bool]:
    """Run ``line`` if it is one of the commands of this module."""
    m = _SVYSET.match(line)
    if m:
        return _svyset(session, m.group(1))
    m = _SVY.match(line)
    if m:
        return _svy(session, m)
    m = _HEAD.match(line)
    if m is None:
        return None
    word, rest = m.group(1).lower(), m.group(2)
    if (len(word) >= 2 and "tabulate".startswith(word)) or word in ("tab1", "tab2"):
        return _tabulate(session, line, word)
    kind = _estimator_word(word)
    if kind is not None and session._steps is not None:
        if kind == "total" and rest.lstrip().startswith("="):
            return None
        return _run_estimate(session, kind, rest, svy=False, subpop=None)
    if word in ("test", "lincom") and isinstance(session.last, Estimates):
        return _linear_after_estimates(session, word, rest)
    if word in ("cc", "cs") and session._steps is not None:
        return _epitab(session, word, rest)
    if word == "anova" and session._steps is not None:
        return _anova(session, rest)
    if word == "statsby":
        return _statsby(session, rest)
    if word == "table" and session._steps is not None:
        from ._stata import from_stata

        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            plain = from_stata(line, columns=list(_steps(session).data.columns))
        if plain.get("ok"):
            return None  # statistics of one varlist by one variable: sp.sumstats
        return _table(session, rest, line)
    if word == "estat" and re.match(r"\s*eff(?:ects)?\b", rest):
        return _estat_effects(session)
    if word in ("correlate", "corr", "cor", "corre", "correl", "correla", "correlat"):
        if re.search(
            r",.*\bc(?:o(?:v(?:a(?:r(?:i(?:a(?:n(?:ce?)?)?)?)?)?)?)?)?\b", rest
        ):
            return _covariance(session, rest)
        return None
    handler = {"ameans": _ameans, "centile": _centile, "cii": _cii,
               "misstable": _misstable}.get(word)  # fmt: skip
    if handler is not None:
        return handler(session, rest)
    return None
