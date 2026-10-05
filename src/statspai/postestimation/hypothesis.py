"""
Post-estimation hypothesis testing.

Equivalent to Stata's ``test`` and ``lincom``.

Supports:
- Wald tests of linear restrictions ``R beta = r``: single, joint
  (``x1 = x2 = 0``), chained (``x1 = x2 = x3``), grouped
  (``(x1 = 0) (x2 = 0)``) and Stata's bare-list form ``test x1 x2``;
- linear combinations of coefficients with inference.

Both read the fit's **full** coefficient covariance matrix and its **own**
reference distribution -- F / t with the fit's degrees of freedom, or
chi-squared / z for likelihood-based fits -- as Stata's ``test`` / ``lincom``
read ``e(V)`` and ``e(df_r)``. Earlier versions rebuilt the covariance from
the standard errors as a diagonal matrix, which mis-stated every restriction
involving two correlated coefficients, and silently treated a misspelled
coefficient name as zero.
"""

from __future__ import annotations

import re
from typing import Any, List, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from ..exceptions import MethodIncompatibility
from ._covariance import inference_df, require_covariance


def test(
    result: Any,
    hypothesis: Union[str, Sequence[str]],
) -> dict[str, Any]:
    """
    Wald test for linear restrictions on coefficients.

    Parameters
    ----------
    result : EconometricResults or CausalResult
        Fitted model with ``.params`` and ``.std_errors``; restrictions that
        involve several coefficients also need its covariance matrix.
    hypothesis : str or list of str
        Hypothesis specification (Stata ``test`` syntax), or a list of
        them to be tested jointly (``["x1 = 0", "x2 = 2*x3"]``). A
        coefficient is named as the result names it, including names such
        as ``I(exper ** 2)`` or ``np.log(sales)``. Examples:
        - ``"x1 = 0"`` — beta_x1 = 0
        - ``"x1 = x2"`` — beta_x1 = beta_x2
        - ``"x1 = x2 = 0"`` — joint: beta_x1 = beta_x2 and beta_x2 = 0
        - ``"x1 x2"`` — joint: beta_x1 = 0 and beta_x2 = 0
        - ``"(x1 = 0) (x2 + x3 = 1)"`` — grouped restrictions
        - ``"x1 = 0, x2 + x3 = 1"`` — the same, in the comma-separated
          spelling of statsmodels' ``f_test``
        - ``"x1 - 2*x2 = 0"`` — linear restriction; ``_cons`` names the
          intercept whatever the estimator calls it.
        - ``"2.race = 3.race"``, ``"1.union#c.grade"`` — Stata's
          factor-variable names for the coefficients the formula engine
          calls ``C(race)[T.2]`` and ``grade:C(union)[T.1]``; ``"i.race"``
          on its own tests every level of the factor jointly (Stata's
          ``testparm i.race``).

    Returns
    -------
    dict
        ``statistic`` (F when the fit has residual degrees of freedom, else
        chi-squared), ``pvalue``, ``df`` = ``(q, df_resid or None)``,
        ``chi2`` (the Wald statistic), ``distribution`` (``"F"`` or
        ``"chi2"``) and ``hypothesis``. Linearly dependent restrictions are
        dropped, as Stata does; ``q`` counts the independent ones.

    Raises
    ------
    MethodIncompatibility
        Unknown coefficient names, restrictions without coefficients,
        inconsistent restrictions, or a multi-coefficient restriction on a
        result that carries no covariance matrix.

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> n = 200
    >>> x1, x2, x3 = rng.normal(size=n), rng.normal(size=n), rng.normal(size=n)
    >>> y = 1.0 + 0.5 * x1 + 0.5 * x2 - 0.3 * x3 + rng.normal(size=n)
    >>> df = pd.DataFrame({"y": y, "x1": x1, "x2": x2, "x3": x3})
    >>> result = sp.regress("y ~ x1 + x2 + x3", data=df)
    >>> out = sp.test(result, "x1 = x2")        # beta1 = beta2?
    >>> sorted(out)
    ['chi2', 'df', 'distribution', 'hypothesis', 'pvalue', 'statistic']
    >>> joint = sp.test(result, "x1 = x2 = 0")  # joint: beta1 = beta2 = 0?
    >>> bool(joint["pvalue"] < 0.05)
    True
    >>> restr = sp.test(result, "x1 + x2 = 1")  # beta1 + beta2 = 1?
    """
    params = _params(result)
    R, r = _parse_hypothesis(hypothesis, params)
    R, r = _independent_restrictions(R, r, str(hypothesis))
    V = require_covariance(result, R, f"test({hypothesis!r})")

    beta = params.to_numpy(dtype=float)
    diff = R @ beta - r
    meat = R @ V @ R.T
    try:
        wald = float(diff @ np.linalg.solve(meat, diff))
    except np.linalg.LinAlgError as exc:
        raise MethodIncompatibility(
            f"test({hypothesis!r}): the restricted covariance R V R' is "
            "singular, so the Wald statistic is not defined.",
            recovery_hint="Check for coefficients with zero variance.",
        ) from exc

    q = R.shape[0]
    df_resid = inference_df(result)
    hotelling = (getattr(result, "data_info", None) or {}).get("hotelling_df")
    if hotelling:
        # An orthogonal-series long-run variance with B terms (robust='ewc'):
        # W is Hotelling T2(q, B), and (B - q + 1) / (B q) * W is
        # F(q, B - q + 1).
        df2 = float(hotelling) - q + 1
        if df2 < 1:
            raise MethodIncompatibility(
                f"test({hypothesis!r}): {q} restrictions need more than "
                f"{int(hotelling)} cosine terms in the EWC covariance.",
                recovery_hint="Refit with a larger ewc_df, or test fewer "
                "restrictions.",
            )
        f_stat = wald * df2 / (float(hotelling) * q)
        return {
            "statistic": f_stat,
            "pvalue": float(sp_stats.f.sf(f_stat, q, df2)),
            "df": (q, _df_out(df2)),
            "hypothesis": hypothesis,
            "chi2": wald,
            "distribution": "F",
        }
    if np.isfinite(df_resid):
        f_stat = wald / q
        return {
            "statistic": f_stat,
            "pvalue": float(sp_stats.f.sf(f_stat, q, df_resid)),
            "df": (q, _df_out(df_resid)),
            "hypothesis": hypothesis,
            "chi2": wald,
            "distribution": "F",
        }
    return {
        "statistic": wald,
        "pvalue": float(sp_stats.chi2.sf(wald, q)),
        "df": (q, None),
        "hypothesis": hypothesis,
        "chi2": wald,
        "distribution": "chi2",
    }


def lincom(
    result: Any,
    expression: str,
    alpha: float = 0.05,
) -> dict[str, Any]:
    """
    Estimate a linear combination of coefficients with inference.

    Parameters
    ----------
    result : EconometricResults or CausalResult
        Fitted model.
    expression : str
        Linear combination (Stata ``lincom`` syntax). Examples:
        - ``"x1 + x2"`` — beta_x1 + beta_x2
        - ``"x1 - x2"`` — beta_x1 - beta_x2
        - ``"2*x1 + 3*x2"`` — 2*beta_x1 + 3*beta_x2
        - ``"_cons + x1"`` — intercept plus slope
    alpha : float, default 0.05
        Significance level.

    Returns
    -------
    dict
        ``estimate``, ``se``, ``statistic`` (t when the fit has residual
        degrees of freedom, else z; also under the historical key ``z``),
        ``df`` (``None`` for z), ``distribution`` (``"t"`` or ``"z"``),
        ``pvalue``, ``ci`` and ``expression``.

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np, pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> n = 200
    >>> x1, x2 = rng.normal(size=n), rng.normal(size=n)
    >>> y = 1.0 + 0.5 * x1 + 0.5 * x2 + rng.normal(size=n)
    >>> df = pd.DataFrame({"y": y, "x1": x1, "x2": x2})
    >>> result = sp.regress("y ~ x1 + x2", data=df)
    >>> out = sp.lincom(result, "x1 + x2")    # beta1 + beta2
    >>> sorted(out)  # doctest: +NORMALIZE_WHITESPACE
    ['ci', 'df', 'distribution', 'estimate', 'expression',
     'pvalue', 'se', 'statistic', 'z']
    >>> diff = sp.lincom(result, "x1 - x2")   # beta1 - beta2
    >>> bool(diff["se"] > 0)
    True
    """
    params = _params(result)
    c, constant = _parse_linear(*_shield_names(expression, params))
    if not np.any(c):
        raise MethodIncompatibility(
            f"lincom({expression!r}) contains no coefficients.",
            recovery_hint=f"Available terms: {list(params.index)}.",
        )
    V = require_covariance(result, c.reshape(1, -1), f"lincom({expression!r})")

    beta = params.to_numpy(dtype=float)
    estimate = float(c @ beta + constant)
    se = float(np.sqrt(max(float(c @ V @ c), 0.0)))
    statistic = estimate / se if se > 0 else float("nan")

    df_resid = inference_df(result)
    if np.isfinite(df_resid):
        pvalue = float(2 * sp_stats.t.sf(abs(statistic), df_resid))
        crit = float(sp_stats.t.ppf(1 - alpha / 2, df_resid))
        distribution, df_out = "t", _df_out(df_resid)
    else:
        pvalue = float(2 * sp_stats.norm.sf(abs(statistic)))
        crit = float(sp_stats.norm.ppf(1 - alpha / 2))
        distribution, df_out = "z", None

    return {
        "estimate": estimate,
        "se": se,
        "statistic": statistic,
        "z": statistic,
        "df": df_out,
        "distribution": distribution,
        "pvalue": pvalue,
        "ci": (estimate - crit * se, estimate + crit * se),
        "expression": expression,
    }


# ======================================================================
# Parsing helpers
# ======================================================================

_NUMBER = re.compile(r"[+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?")
_MANTISSA_E = re.compile(r"(?:\d+\.?\d*|\.\d+)[eE]")
_B_SYNTAX = re.compile(r"_b\[(.+)\]")
#: Spellings of the intercept across StatsPAI estimators and Stata.
_INTERCEPT_NAMES = ("_cons", "Intercept", "const", "(Intercept)")


def _params(result: Any) -> pd.Series:
    params = result.params
    if isinstance(params, pd.Series):
        return params
    values = np.atleast_1d(np.asarray(params, dtype=float))
    return pd.Series(values, index=[f"b{i}" for i in range(values.size)])


def _df_out(df: float) -> Any:
    return int(df) if float(df).is_integer() else float(df)


#: One factor of a Stata factor-variable coefficient name: ``2.race``,
#: ``1b.union`` (a level), ``c.grade`` (continuous), ``i.race`` (every level).
_FV_LEVEL = re.compile(r"(-?\d+(?:\.\d+)?)(?:bn|b|o|n)?\.([^\W\d]\w*)")
_FV_OTHER = re.compile(r"(?:([ci])\.)?([^\W\d]\w*)")
#: The same level as the formula engine names it: ``C(race)[T.2]``.
_PATSY_LEVEL = re.compile(
    r"C\(\s*([^\W\d]\w*)\s*(?:,[^()]*(?:\([^()]*\))?[^()]*)?\)\[T\.(.+)\]"
)


def _same_level(stata: str, patsy: str) -> bool:
    try:
        return float(stata) == float(patsy)
    except ValueError:
        return stata == patsy


def _factor_matches(name: str, names: List[str]) -> List[int]:
    """Coefficients a Stata factor-variable name refers to.

    ``2.race`` is ``C(race)[T.2]``; ``1.union#c.grade`` is the interaction
    of that level with ``grade``, in whichever order the design wrote it;
    ``i.race`` is every level of ``race``. Returns positions in ``names``.
    """
    wanted = []
    for part in name.split("#"):
        level = _FV_LEVEL.fullmatch(part)
        if level is not None:
            wanted.append(("level", level.group(2), level.group(1)))
            continue
        other = _FV_OTHER.fullmatch(part)
        if other is None:
            return []
        kind = "any" if other.group(1) == "i" else "plain"
        wanted.append((kind, other.group(2), ""))
    if all(kind == "plain" for kind, _, _ in wanted) and "#" not in name:
        if not name.startswith("c."):
            return []  # an ordinary name: not factor-variable notation
    hits = []
    for pos, candidate in enumerate(names):
        pieces = [piece.strip() for piece in str(candidate).split(":")]
        if len(pieces) != len(wanted):
            continue
        left = list(pieces)
        for kind, var, lev in wanted:
            found = None
            for piece in left:
                m = _PATSY_LEVEL.fullmatch(piece)
                if kind == "plain" and piece == var:
                    found = piece
                elif kind != "plain" and m is not None and m.group(1) == var:
                    if kind == "any" or _same_level(lev, m.group(2)):
                        found = piece
                if found is not None:
                    break
            if found is None:
                break
            left.remove(found)
        else:
            hits.append(pos)
    return hits


def _resolve_name(name: str, params: pd.Series, expression: str) -> int:
    names = list(params.index)
    if name in names:
        return names.index(name)
    b = _B_SYNTAX.fullmatch(name)
    if b is not None:
        return _resolve_name(b.group(1).strip(), params, expression)
    if name in _INTERCEPT_NAMES:
        for alias in _INTERCEPT_NAMES:
            if alias in names:
                return names.index(alias)
    shown = [str(n) for n in params.attrs.get("display", names)]
    hits = _factor_matches(name, shown)
    if len(hits) == 1:
        return hits[0]
    if len(hits) > 1:
        raise MethodIncompatibility(
            f"{name!r} in {expression!r} names {len(hits)} coefficients "
            f"({[shown[h] for h in hits]}); a restriction with '=' or an "
            "arithmetic expression needs one.",
            recovery_hint="Name one level (2.race), or list the factor on "
            "its own for a joint test (sp.test(result, 'i.race')).",
        )
    for token, real in zip(names, shown):
        expression = expression.replace(token, real)
        name = name.replace(token, real)
    raise MethodIncompatibility(
        f"Unknown coefficient {name!r} in {expression!r}.",
        recovery_hint=f"Available terms: {shown}.",
    )


def _split_terms(expression: str) -> List[Tuple[float, str]]:
    """Split ``a*x - b*y + z`` into signed terms, respecting ``1e-3``."""
    terms: List[Tuple[float, str]] = []
    sign, buf = 1.0, ""
    for ch in expression:
        if ch in "+-":
            stripped = buf.strip()
            if not stripped:
                if ch == "-":
                    sign = -sign
                continue
            if _MANTISSA_E.fullmatch(stripped.split("*")[-1].strip()):
                buf += ch  # exponent sign inside a number such as 1e-3
                continue
            terms.append((sign, stripped))
            sign, buf = (-1.0 if ch == "-" else 1.0), ""
        else:
            buf += ch
    stripped = buf.strip()
    if not stripped:
        raise MethodIncompatibility(
            f"Cannot parse {expression!r}: it is empty or ends in an operator."
        )
    terms.append((sign, stripped))
    return terms


def _parse_linear(expression: str, params: pd.Series) -> Tuple[np.ndarray, float]:
    """Parse a linear expression into ``(coefficient vector, constant)``."""
    c = np.zeros(len(params), dtype=float)
    constant = 0.0
    for sign, term in _split_terms(expression):
        factors = [f.strip() for f in term.split("*")]
        if any(not f for f in factors):
            raise MethodIncompatibility(
                f"Cannot parse term {term!r} in {expression!r}."
            )
        coef, names = sign, []
        for factor in factors:
            # `x/5`: division by a number (Stata's `lincom exp + exp2/5`)
            factor, *divisors = [part.strip() for part in factor.split("/")]
            if any(not _NUMBER.fullmatch(d) or float(d) == 0 for d in divisors):
                raise MethodIncompatibility(
                    f"{term!r} in {expression!r} divides by something other "
                    "than a non-zero number; test and lincom take linear "
                    "combinations only (sp.nlcom takes ratios of coefficients)."
                )
            for d in divisors:
                coef /= float(d)
            if _NUMBER.fullmatch(factor):
                coef *= float(factor)
            else:
                names.append(factor)
        if len(names) > 1:
            raise MethodIncompatibility(
                f"{term!r} in {expression!r} multiplies coefficients together; "
                "test and lincom take linear combinations only (use nlcom-style "
                "delta-method tools for nonlinear functions)."
            )
        if names:
            c[_resolve_name(names[0], params, expression)] += coef
        else:
            constant += coef
    return c, constant


def _parse_equality_chain(
    text: str, params: pd.Series
) -> Tuple[List[np.ndarray], List[float]]:
    sides = [s.strip() for s in text.split("=")]
    if any(not s for s in sides):
        raise MethodIncompatibility(f"Cannot parse hypothesis {text!r}.")
    parsed = [_parse_linear(s, params) for s in sides]
    rows, values = [], []
    for (c_a, k_a), (c_b, k_b) in zip(parsed[:-1], parsed[1:]):
        rows.append(c_a - c_b)
        values.append(k_b - k_a)
    return rows, values


_SIMPLE_NAME = re.compile(r"[A-Za-z_][\w.]*$")


def _shield_names(text: str, params: pd.Series) -> Tuple[str, pd.Series]:
    """Replace coefficient names that hold operators or blanks by tokens.

    ``I(exper ** 2)``, ``np.log(sales)`` and ``C(year)[T.1981]:educ`` are
    names of coefficients, but read as arithmetic by a parser of linear
    expressions. Each such name that occurs in ``text`` is swapped for a
    plain token, longest name first, and the returned parameter vector is
    indexed by the same tokens; ``attrs['display']`` keeps the real names
    for error messages.
    """
    names = [str(n) for n in params.index]
    tokens = list(names)
    order = sorted(
        (i for i, n in enumerate(names) if not _SIMPLE_NAME.match(n)),
        key=lambda i: -len(names[i]),
    )
    for i in order:
        # ``I(x**2)`` names the coefficient the design calls ``I(x ** 2)``:
        # blanks between the pieces of a name do not distinguish names.
        spellings = [names[i]]
        if "(" not in names[i] and "[" in names[i]:
            # IV and panel fits store ``np.log(x)`` as ``np.log[x]``; the
            # name written as in the formula refers to the same coefficient
            spellings.append(names[i].replace("[", "(").replace("]", ")"))
        token = f"spcoef{i}zz"
        for spelling in spellings:
            pieces = re.findall(r"\w+|[^\w\s]", spelling)
            pattern = r"\s*".join(re.escape(piece) for piece in pieces)
            if not pieces or pieces[0][0].isalnum() or pieces[0][0] == "_":
                pattern = r"(?<![\w.])" + pattern
            if pieces and (pieces[-1][-1].isalnum() or pieces[-1][-1] == "_"):
                pattern += r"(?!\w)"
            text, hits = re.subn(pattern, token, text)
            if hits:
                tokens[i] = token
    shielded = pd.Series(np.asarray(params, dtype=float), index=tokens)
    shielded.attrs["display"] = names
    return text, shielded


def _parse_hypothesis(
    hypothesis: Any,
    params: pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Parse a hypothesis string into the restriction matrix R and vector r.

    Examples:
        "x1 = 0"          -> R = [0, 1, 0, ...], r = [0]
        "x1 = x2"         -> R = [0, 1, -1, ...], r = [0]
        "x1 = x2 = 0"     -> rows x1 - x2 = 0 and x2 = 0
        "x1 x2"           -> rows x1 = 0 and x2 = 0
        "(x1 = 0) (x2 = 1)" -> rows x1 = 0 and x2 = 1
    """
    if isinstance(hypothesis, (list, tuple)):
        # a list of restrictions, tested jointly: ["x1 = 0", "x2 = 2*x3"]
        if not hypothesis or not all(isinstance(h, str) for h in hypothesis):
            raise MethodIncompatibility(
                "test() takes a hypothesis string or a list of them, e.g. "
                "['x1 = 0', 'x2 = x3']."
            )
        stacked = [_parse_hypothesis(h, params) for h in hypothesis]
        return (
            np.vstack([rows for rows, _ in stacked]),
            np.concatenate([values for _, values in stacked]),
        )
    if not isinstance(hypothesis, str):
        raise MethodIncompatibility(
            "test() takes a hypothesis string or a list of them, got "
            f"{type(hypothesis).__name__}."
        )
    text = hypothesis.strip()
    if not text:
        raise MethodIncompatibility("test() needs a hypothesis, e.g. 'x1 = 0'.")
    text, params = _shield_names(text, params)

    if text.startswith("("):
        groups = re.findall(r"\(([^()]*)\)", text)
        if not groups or re.sub(r"\(([^()]*)\)", "", text).strip():
            raise MethodIncompatibility(f"Cannot parse hypothesis {hypothesis!r}.")
    elif "=" in text and "," in text:
        # statsmodels / patsy spelling: "x1 = 0, x2 = x3" is a joint test.
        groups = [g.strip() for g in text.split(",")]
        if any(not g for g in groups):
            raise MethodIncompatibility(f"Cannot parse hypothesis {hypothesis!r}.")
    else:
        groups = [text]

    rows: List[np.ndarray] = []
    values: List[float] = []
    for group in groups:
        if "=" in group:
            g_rows, g_values = _parse_equality_chain(group, params)
        else:
            # Stata's `test x1 x2`: every listed coefficient equals zero.
            g_rows, g_values = [], []
            for name in group.replace(",", " ").split():
                # `i.race` (Stata's testparm): every level of the factor
                shown = params.attrs.get("display", list(params.index))
                several = _factor_matches(name, [str(n) for n in shown])
                if len(several) < 2 or name in params.index:
                    several = [_resolve_name(name, params, hypothesis)]
                for position in several:
                    row = np.zeros(len(params), dtype=float)
                    row[position] = 1.0
                    g_rows.append(row)
                    g_values.append(0.0)
        rows.extend(g_rows)
        values.extend(g_values)
    if not rows:
        raise MethodIncompatibility(f"Cannot parse hypothesis {hypothesis!r}.")
    return np.vstack(rows), np.asarray(values, dtype=float)


def _independent_restrictions(
    R: np.ndarray, r: np.ndarray, hypothesis: str
) -> Tuple[np.ndarray, np.ndarray]:
    """Drop linearly dependent restrictions; refuse empty or inconsistent ones."""
    kept_rows: List[np.ndarray] = []
    kept_values: List[float] = []
    for row, value in zip(R, r):
        if not np.any(row):
            raise MethodIncompatibility(
                f"test({hypothesis!r}) contains a restriction with no "
                "coefficients (e.g. 'x1 = x1').",
            )
        candidate = np.vstack(kept_rows + [row])
        if np.linalg.matrix_rank(candidate) > len(kept_rows):
            kept_rows.append(row)
            kept_values.append(float(value))
            continue
        augmented = np.column_stack([candidate, kept_values + [float(value)]])
        if np.linalg.matrix_rank(augmented) > len(kept_rows):
            raise MethodIncompatibility(
                f"test({hypothesis!r}) contains inconsistent restrictions.",
            )
    return np.vstack(kept_rows), np.asarray(kept_values, dtype=float)
