"""Post-estimation commands a session runs itself.

``predict`` with the diagnostic statistics of a linear or a logistic fit,
``dfbeta``, ``estat gof``, ``lroc``, ``lrtest``, ``estat vce`` and
``logistic`` (``logit`` reported as odds ratios). Each needs the fitted
result and the rows it was fitted on, which a single translated line does
not have.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, Any, Dict, Optional

import numpy as np
import pandas as pd
from scipy import stats

from ...diagnostics.influence import influence_measures, logit_gof, logit_influence
from ._stata_datastep import row_mask
from ._stata_expr import StataExprError
from ._stata_lexer import StataParseError
from ._stata_lexer import parse as _parse

if TYPE_CHECKING:  # pragma: no cover
    from ._stata_run import StataSession

__all__ = ["run_postestimation", "record_sample"]

#: predict options after `regress` -> column of sp.influence_measures
_LINEAR = {
    "rstandard": "rstandard", "rstudent": "rstudent", "cooksd": "cooksd",
    "dfits": "dfits", "welsch": "welsch", "covratio": "covratio",
    "stdr": "stdr",
}  # fmt: skip
_LINEAR_ABBREVIATIONS = {
    "rsta": "rstandard", "rstu": "rstudent", "c": "cooksd", "cooksd": "cooksd",
    "dfi": "dfits", "w": "welsch", "covr": "covratio",
}  # fmt: skip
#: predict options after `logit` -> column of sp.logit_influence
_LOGIT = {
    "residuals": "residual", "rstandard": "rstandard", "deviance": "deviance",
    "hat": "hat", "dx2": "dx2", "ddeviance": "ddeviance", "dbeta": "dbeta",
    "number": "pattern",
}  # fmt: skip


def _steps(session: "StataSession") -> Any:
    if session._steps is None:
        raise StataExprError("no data in memory")
    return session._steps


def _full(word: str, table: Dict[str, str], shortest: Dict[str, int]) -> Optional[str]:
    for name in table:
        k = shortest.get(name, len(name))
        if k <= len(word) <= len(name) and name.startswith(word):
            return name
    return None


def record_sample(session: "StataSession", run_data: pd.DataFrame) -> None:
    """Remember which rows of the data in memory the last model used.

    The rows are those of ``run_data`` (the frame handed to the estimator,
    after ``if`` / ``in`` and the weights) that are complete on the model's
    variables. They are kept as positions in the frame in memory, and only
    as long as that frame is not replaced: a ``sort`` or a ``drop`` makes a
    new frame, and ``e(sample)`` is then no longer known.
    """
    session.stored.pop("e_sample", None)
    steps, result = session._steps, session.last
    if steps is None or result is None:
        return
    info = getattr(result, "data_info", None) or {}
    n_fit = None
    for key in ("X", "y", "fitted_values", "residuals"):
        if info.get(key) is not None:
            n_fit = len(info[key])
            break
    if n_fit is None:
        n_fit = getattr(result, "nobs", None)
    # the session's own copy: adding a column (predict, generate) then
    # keeps the frame, and with it the sample
    steps._own()
    data = steps.data
    if n_fit is None or not data.index.is_unique:
        return
    call = session._last_call or {}
    formula = str((call.get("arguments") or {}).get("formula") or "")
    names = [n for n in dict.fromkeys(re.findall(r"[A-Za-z_]\w*", formula))
             if n in run_data.columns]  # fmt: skip
    weights = (call.get("arguments") or {}).get("weights")
    if isinstance(weights, str) and weights in run_data.columns:
        names.append(weights)
    if not names:
        return
    used = run_data.index[run_data[names].notna().all(axis=1)]
    if len(used) != int(n_fit) or not used.isin(data.index).all():
        return
    mask = data.index.isin(used)
    session.stored["e_sample"] = (id(data), len(data), mask)


def estimation_sample(session: "StataSession") -> np.ndarray:
    held = session.stored.get("e_sample")
    data = _steps(session).data if session._steps is not None else None
    if held is None or data is None or held[0] != id(data) or held[1] != len(data):
        raise StataExprError(
            "the estimation sample e(sample) is not known here (the rows "
            "were sorted or dropped since the model was fitted, or the "
            "estimator does not report its sample)"
        )
    mask: np.ndarray = held[2]
    return mask


def _predict(session: "StataSession", line: str) -> Optional[bool]:
    try:
        cmd = _parse(line)
    except StataParseError:
        return None
    if cmd.command != "predict" or session.last is None or session._steps is None:
        return None
    tool = (session._last_call or {}).get("tool")
    if tool in ("mlogit", "ologit", "oprobit") and not [k for k in cmd.options if k]:
        return _predict_outcomes(session, cmd, str(tool))
    options = [k for k in cmd.options if k]
    if len(options) != 1 or cmd.options[options[0]] is not None:
        return None
    word = options[0]
    names = [n for n in cmd.varlist if n not in ("float", "double")]
    if len(names) != 1:
        return None
    column: Optional[np.ndarray] = None
    sample = None
    if tool == "regress":
        shortest = {"rstandard": 4, "rstudent": 4, "cooksd": 1, "dfits": 3,
                    "welsch": 1, "covratio": 4}  # fmt: skip
        full = _full(word, _LINEAR, shortest)
        if word in ("stdp", "stdf", "stdr"):
            return _predict_stdp(session, names[0], word, cmd)
        if full is None:
            return None
        sample = estimation_sample(session)
        column = influence_measures(session.last)[_LINEAR[full]].to_numpy()
    elif tool == "logit":
        full = _full(word, _LOGIT, {"residuals": 1, "rstandard": 2, "deviance": 2,
                                    "hat": 1, "dx2": 2, "ddeviance": 2, "dbeta": 2,
                                    "number": 1})  # fmt: skip
        if word == "stdp":
            return _predict_stdp(session, names[0], word, cmd)
        if full is None:
            return None
        sample = estimation_sample(session)
        column = logit_influence(session.last)[_LOGIT[full]].to_numpy(dtype=float)
    else:
        return None
    data = _steps(session).data
    values: Any = np.full(len(data), np.nan)
    values[sample] = column
    if cmd.if_cond or cmd.in_range:
        keep = row_mask(data, cmd.if_cond, cmd.in_range, session.stored)
        values = np.where(keep, values, np.nan)
    _steps(session).add_column(names[0], values, double="double" in cmd.varlist)
    return False


def _design_rows(session: "StataSession") -> np.ndarray:
    """The design matrix of the last fit on every row of the data."""
    from ._stata_run import _term_column

    data = _steps(session).data
    columns = []
    for term in getattr(session.last, "params").index:
        if term in ("Intercept", "const", "_cons"):
            columns.append(np.ones(len(data)))
        elif term in data.columns:
            columns.append(data[term].to_numpy(dtype=float, na_value=np.nan))
        else:
            built = _term_column(str(term), data)
            if built is None:
                raise StataExprError(
                    f"`predict`: coefficient {term!r} is not built from "
                    "columns of the data"
                )
            columns.append(built)
    return np.column_stack(columns)


def _predict_stdp(session: "StataSession", name: str, kind: str, cmd: Any) -> bool:
    """``predict se, stdp`` / ``stdf``: the standard error of the linear
    prediction (of the forecast), on every row whose regressors are
    observed, with the covariance matrix the fit reported."""
    result = session.last
    cov = getattr(result, "vcov", None)
    if cov is None:
        cov = getattr(result, "cov_params", None)
    cov = cov() if callable(cov) else cov
    if cov is None:
        raise StataExprError("`predict, stdp`: the result has no covariance matrix")
    rows = _design_rows(session)
    v = np.asarray(cov, dtype=float)
    if v.shape[0] != rows.shape[1]:
        raise StataExprError("`predict, stdp`: the design does not match")
    variance = np.einsum("ij,jk,ik->i", rows, v, rows)
    if kind in ("stdf", "stdr"):
        # s^2 (1 + h) for a forecast, s^2 (1 - h) for a residual
        info = getattr(result, "data_info", None) or {}
        e = np.asarray(info.get("residuals"), dtype=float)
        s2 = float(e @ e) / (len(e) - rows.shape[1])
        variance = s2 + variance if kind == "stdf" else s2 - variance
    with np.errstate(invalid="ignore"):
        values = np.sqrt(variance)
    data = _steps(session).data
    if cmd.if_cond or cmd.in_range:
        keep = row_mask(data, cmd.if_cond, cmd.in_range, session.stored)
        values = np.where(keep, values, np.nan)
    _steps(session).add_column(name, values, double="double" in cmd.varlist)
    return False


def _predict_outcomes(session: "StataSession", cmd: Any, tool: str) -> bool:
    """``predict p1 p2 ... pk`` after ``mlogit`` / ``ologit`` / ``oprobit``:
    the probability of each outcome, on every row whose regressors are
    observed."""
    names = [n for n in cmd.varlist if n not in ("float", "double")]
    result = session.last
    model = getattr(result, "model_info", None) or {}
    categories = list(model.get("categories") or [])
    if len(names) != len(categories):
        raise StataExprError(
            f"predict after {tool}: give one new variable per outcome "
            f"({len(categories)}), or use outcome()"
        )
    data = _steps(session).data
    params = getattr(result, "params")
    formula = str(((session._last_call or {}).get("arguments") or {}).get("formula"))
    terms = [t.strip() for t in formula.split("~", 1)[1].split("+")]
    if any(t not in data.columns for t in terms):
        raise StataExprError(f"predict after {tool}: only plain covariates are covered")
    x = np.column_stack([data[t].to_numpy(dtype=float, na_value=np.nan) for t in terms])
    if tool == "mlogit":
        base = model.get("base_category")
        index = np.zeros((len(data), len(categories)))
        for j, category in enumerate(categories):
            if category == base:
                continue
            # the equation is named by the outcome as the estimator wrote it
            label = next(
                (
                    lab
                    for lab in (category, int(float(category)), float(category))
                    if f"[{lab}]_cons" in params.index
                ),
                None,
            )
            if label is None:
                raise StataExprError(
                    f"predict after mlogit: no equation for outcome {category}"
                )
            beta = np.array([float(params[f"[{label}]{t}"]) for t in terms])
            index[:, j] = x @ beta + float(params[f"[{label}]_cons"])
        index -= index.max(axis=1, keepdims=True)
        prob = np.exp(index)
        prob /= prob.sum(axis=1, keepdims=True)
    else:
        beta = np.array([float(params[t]) for t in terms])
        cuts = np.array([float(params[f"/cut{k}"]) for k in range(1, len(categories))])
        cdf = stats.logistic.cdf if tool == "ologit" else stats.norm.cdf
        upper = np.column_stack(
            [cdf(c - x @ beta) for c in cuts] + [np.ones(len(data))]
        )
        prob = np.diff(np.column_stack([np.zeros(len(data)), upper]), axis=1)
    prob[np.isnan(x).any(axis=1)] = np.nan
    if cmd.if_cond or cmd.in_range:
        keep = row_mask(data, cmd.if_cond, cmd.in_range, session.stored)
        prob[~keep] = np.nan
    for j, name in enumerate(names):
        _steps(session).add_column(name, prob[:, j], double="double" in cmd.varlist)
    return False


_NESTREG = re.compile(
    r"\s*nestreg\s*(?:,[^:]*)?:\s*reg(?:r(?:e(?:s(?:s)?)?)?)?\s+(\S+)\s+(.+)\Z",
    re.S | re.I,
)


def _nestreg(session: "StataSession", m: "re.Match[str]") -> bool:
    """``nestreg: regress y (block 1) (block 2) ...``: the regression with
    the blocks added one after the other on the rows complete on all of
    them, and the F test of each block given the ones before it."""
    import statspai as sp

    from ._stata import from_stata

    outcome, tail = m.group(1), m.group(2)
    if "," in tail or " if " in f" {tail} " or " in " in f" {tail} ":
        raise StataExprError("nestreg with options or qualifiers is not implemented")
    blocks = re.findall(r"\(([^)]*)\)|(\S+)", tail)
    blocks = [a.strip() or b for a, b in blocks]
    data = _steps(session).data
    columns = list(data.columns)
    formulas = []
    for k in range(1, len(blocks) + 1):
        out = from_stata(f"regress {outcome} " + " ".join(blocks[:k]), columns=columns)
        if not out.get("ok"):
            raise StataExprError(f"nestreg: {out.get('error')}")
        formulas.append(str(out["arguments"]["formula"]))
    used = [n for n in dict.fromkeys(re.findall(r"[A-Za-z_]\w*", formulas[-1]))
            if n in data.columns]  # fmt: skip
    sample = data.loc[data[used].notna().all(axis=1)]
    rows, previous = [], None
    models = []
    fit: Any = None
    for k, formula in enumerate(formulas, 1):
        fit = sp.regress(formula, data=sample)
        info = fit.data_info
        # what Stata prints for each block: the regression itself
        models.append(
            {
                "b": dict(fit.params),
                "se": dict(fit.std_errors),
                "t": dict(fit.params / fit.std_errors),
                "ci": fit.conf_int() if hasattr(fit, "conf_int") else None,
                "N": float(len(sample)),
                "rss": float(info["rss"]),
                "mss": float(info["tss"]) - float(info["rss"]),
                "tss": float(info["tss"]),
                "r2": 1 - float(info["rss"]) / float(info["tss"]),
                "r2_a": 1
                - (float(info["rss"]) / float(info["df_resid"]))
                / (float(info["tss"]) / (len(sample) - 1)),
                "rmse": float(np.sqrt(float(info["rss"]) / float(info["df_resid"]))),
                "ms": [
                    (float(info["tss"]) - float(info["rss"])) / (len(fit.params) - 1),
                    float(info["rss"]) / float(info["df_resid"]),
                    float(info["tss"]) / (len(sample) - 1),
                ],
                "F": (float(info["tss"]) - float(info["rss"]))
                / (len(fit.params) - 1)
                / (float(info["rss"]) / float(info["df_resid"])),
            }
        )
        rss, df_r = float(info["rss"]), float(info["df_resid"])
        r2 = 1 - rss / float(info["tss"])
        if previous is None:
            block_df = float(len(fit.params) - 1)
            stat = (float(info["tss"]) - rss) / block_df / (rss / df_r)
            change = np.nan
        else:
            block_df = previous[1] - df_r
            stat = (previous[0] - rss) / block_df / (rss / df_r)
            change = r2 - previous[2]
        rows.append({"block": k, "F": stat, "block_df": block_df, "residual_df": df_r,
                     "p": float(stats.f.sf(stat, block_df, df_r)), "r2": r2,
                     "change_r2": change})  # fmt: skip
        previous = (rss, df_r, r2)
    table = pd.DataFrame(rows).set_index("block")
    table.attrs["models"] = models
    session.output = table
    session.last, session.last_data = fit, sample
    session._last_call = {"tool": "regress", "arguments": {"formula": formulas[-1]}}
    session._store_estimates(fit)
    return True


def _dfbeta(session: "StataSession", rest: str) -> bool:
    """``dfbeta [varlist]``: ``_dfbeta_1``, ``_dfbeta_2`` ... for the
    regressors of the last ``regress`` (or the ones named)."""
    if (session._last_call or {}).get("tool") != "regress":
        raise StataExprError("dfbeta follows regress")
    table = influence_measures(session.last)
    sample = estimation_sample(session)
    terms = [c[len("dfbeta_") :] for c in table.columns if c.startswith("dfbeta_")]
    terms = [t for t in terms if t not in ("Intercept", "const", "_cons")]
    wanted = rest.split(",")[0].split()
    if wanted:
        missing = [w for w in wanted if w not in terms]
        if missing:
            raise StataExprError(f"dfbeta: {missing} are not regressors of the model")
        terms = wanted
    data = _steps(session).data
    taken = [c for c in data.columns if re.fullmatch(r"_dfbeta_\d+", str(c))]
    start = len(taken)
    for k, term in enumerate(terms, start + 1):
        values = np.full(len(data), np.nan)
        values[sample] = table[f"dfbeta_{term}"].to_numpy()
        _steps(session).add_column(f"_dfbeta_{k}", values, double=False)
    return False


def _log_likelihood(result: Any) -> Optional[float]:
    for holder in (
        getattr(result, "model_info", None),
        getattr(result, "diagnostics", None),
    ):
        for key in ("ll", "Log-Likelihood", "log_likelihood", "llf"):
            value = (holder or {}).get(key)
            if value is not None and np.isfinite(value):
                return float(value)
    info = getattr(result, "data_info", None) or {}
    resid = info.get("residuals")
    if resid is not None:
        e = np.asarray(resid, dtype=float)
        n, rss = e.size, float(e @ e)
        return float(-0.5 * n * (np.log(2 * np.pi * rss / n) + 1))
    return None


def _lrtest(session: "StataSession", rest: str) -> bool:
    """``lrtest a [b]``: twice the difference of the log likelihoods of
    two stored models (``.`` or nothing is the last one), chi-squared on
    the difference in the number of parameters."""
    names = rest.split(",")[0].split()
    if not 1 <= len(names) <= 2:
        raise StataExprError("expected `lrtest name [name]`")
    if len(names) == 1:
        names.append(".")
    models = []
    for name in names:
        if name == ".":
            models.append(session.last)
        elif name in session.estimates:
            models.append(session.estimates[name][0])
        else:
            raise StataExprError(f"lrtest: estimation result {name!r} was not stored")
    lls: Any = [_log_likelihood(m) for m in models]
    sizes = [len(getattr(m, "params")) for m in models]
    obs = [getattr(m, "nobs", None) or len((getattr(m, "data_info", None) or {})
                                           .get("y", [])) for m in models]  # fmt: skip
    if None in lls:
        raise StataExprError("lrtest: a model does not report its log likelihood")
    if obs[0] != obs[1]:
        raise StataExprError(
            f"lrtest: the models were fitted on {obs[0]} and {obs[1]} "
            "observations; Stata stops here (r(498))"
        )
    df = abs(sizes[0] - sizes[1])
    if df == 0:
        raise StataExprError("lrtest: the models have the same number of parameters")
    big, small = (0, 1) if sizes[0] > sizes[1] else (1, 0)
    stat = 2.0 * (lls[big] - lls[small])
    out = {"chi2": float(stat), "df": float(df),
           "p": float(stats.chi2.sf(stat, df))}  # fmt: skip
    session.output = out
    session.stored["r"] = dict(out)
    return True


def _gof(session: "StataSession", rest: str) -> bool:
    m = re.search(r"group\(\s*(\d+)\s*\)", rest)
    out = logit_gof(session.last, groups=int(m.group(1)) if m else None)
    session.output = out
    session.stored["r"] = {"chi2": out["statistic"], "df": float(out["df"]),
                           "p": out["pvalue"], "N": float(out["n"])}  # fmt: skip
    return True


def _lroc(session: "StataSession") -> bool:
    from ...diagnostics.influence import roc_area

    info = getattr(session.last, "data_info", None) or {}
    if info.get("y") is None or info.get("fitted_values") is None:
        raise StataExprError("lroc follows logit / probit")
    out = roc_area(info["y"], info["fitted_values"])
    session.output = {"N": out["n"], "area": out["auc"]}
    session.stored["r"] = {"N": out["n"], "area": out["auc"]}
    return True


def _vce(session: "StataSession") -> bool:
    cov = getattr(session.last, "vcov", None)
    cov = cov() if callable(cov) else cov
    if cov is None:
        raise StataExprError("estat vce: the result has no covariance matrix")
    names = list(getattr(session.last, "params").index)
    session.output = pd.DataFrame(np.asarray(cov, dtype=float), index=names,
                                  columns=names)  # fmt: skip
    return True


def _linktest(session: "StataSession") -> bool:
    """``linktest``: the model refitted on its own linear prediction and
    the square of it; a significant ``_hatsq`` points to a misspecified
    link or functional form."""
    import statspai as sp

    tool = (session._last_call or {}).get("tool")
    info = getattr(session.last, "data_info", None) or {}
    x, y = info.get("X"), info.get("y")
    if tool not in ("regress", "logit", "probit") or x is None or y is None:
        raise StataExprError("linktest follows regress, logit or probit")
    beta = np.asarray(getattr(session.last, "params"), dtype=float)
    hat = np.asarray(x, dtype=float) @ beta
    frame = pd.DataFrame({"_y": np.asarray(y, dtype=float), "_hat": hat,
                          "_hatsq": hat**2})  # fmt: skip
    # the estimates of the model stay the last ones, as after Stata's linktest
    session.output = getattr(sp, tool)("_y ~ _hat + _hatsq", data=frame)
    return True


def _summarize_sample(session: "StataSession") -> bool:
    """``estat summarize``: the outcome and the regressors on the
    estimation sample."""
    import statspai as sp

    info = getattr(session.last, "data_info", None) or {}
    x, y = info.get("X"), info.get("y")
    if x is None or y is None:
        raise StataExprError("estat summarize: the result does not store its data")
    names = [str(n) for n in getattr(session.last, "params").index]
    frame = pd.DataFrame(np.asarray(x, dtype=float), columns=names)
    frame = frame[[n for n in names if n not in ("Intercept", "const", "_cons")]]
    frame.insert(0, "_outcome", np.asarray(y, dtype=float))
    session.output = sp.sumstats(frame, stats=["mean", "sd", "min", "max"],
                                 output="numeric")  # fmt: skip
    return True


_HEAD = re.compile(r"\s*([A-Za-z_]\w*)\b\s*(.*)\Z", re.S)


def run_postestimation(session: "StataSession", line: str) -> Optional[bool]:
    """Run ``line`` if it is one of the commands of this module."""
    nested = _NESTREG.match(line)
    if nested is not None and session._steps is not None:
        return _nestreg(session, nested)
    m = _HEAD.match(line)
    if m is None:
        return None
    word, rest = m.group(1).lower(), m.group(2)
    if word == "predict":
        return _predict(session, line)
    if session.last is None:
        return None
    if word == "dfbeta":
        return _dfbeta(session, rest)
    if word == "lrtest":
        return _lrtest(session, rest)
    if word == "lroc":
        return _lroc(session)
    if word == "linktest":
        return _linktest(session)
    if word == "estat":
        sub = rest.split(",")[0].strip().lower()
        if sub == "gof":
            return _gof(session, rest)
        if sub == "vce":
            return _vce(session)
        if sub.startswith("su"):
            return _summarize_sample(session)
    return None
