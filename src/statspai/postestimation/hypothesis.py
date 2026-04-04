"""
Post-estimation hypothesis testing.

Equivalent to Stata's ``test``, ``lincom``, ``nlcom`` commands.

Supports:
- Wald test for linear restrictions (R*beta = r)
- Joint significance tests
- Linear combinations of coefficients
"""

from typing import Optional, List, Dict, Any, Union
import numpy as np
import pandas as pd
from scipy import stats as sp_stats


def test(
    result,
    hypothesis: str,
) -> Dict[str, float]:
    """
    Wald test for linear restrictions on coefficients.

    Parameters
    ----------
    result : EconometricResults or CausalResult
        Fitted model with ``.params`` and ``.std_errors``.
    hypothesis : str
        Hypothesis specification. Examples:
        - ``"x1 = 0"`` — test if beta_x1 = 0
        - ``"x1 = x2"`` — test if beta_x1 = beta_x2
        - ``"x1 = x2 = 0"`` — joint test
        - ``"x1 + x2 = 1"`` — linear restriction

    Returns
    -------
    dict
        ``{'statistic': F, 'pvalue': p, 'df': (k, n-K), 'hypothesis': str}``

    Examples
    --------
    >>> result = sp.regress("y ~ x1 + x2 + x3", data=df)
    >>> sp.test(result, "x1 = x2")           # beta1 = beta2?
    >>> sp.test(result, "x1 = x2 = 0")       # joint: beta1 = beta2 = 0?
    >>> sp.test(result, "x1 + x2 = 1")       # beta1 + beta2 = 1?
    """
    params = result.params
    vcov = _get_vcov(result)

    R, r = _parse_hypothesis(hypothesis, params)

    # Wald statistic: (R*beta - r)' * [R*V*R']^{-1} * (R*beta - r) / q
    beta = params.values
    q = R.shape[0]  # number of restrictions

    Rb_minus_r = R @ beta - r

    try:
        meat = R @ vcov @ R.T
        wald = float(Rb_minus_r @ np.linalg.solve(meat, Rb_minus_r))
    except np.linalg.LinAlgError:
        wald = float(Rb_minus_r @ np.linalg.lstsq(meat, Rb_minus_r, rcond=None)[0])

    # F-statistic = Wald / q
    f_stat = wald / q
    df_resid = result.data_info.get('df_resid', np.inf) if hasattr(result, 'data_info') and isinstance(result.data_info, dict) else np.inf

    if np.isfinite(df_resid):
        pvalue = float(1 - sp_stats.f.cdf(f_stat, q, df_resid))
    else:
        # Chi-squared if df_resid unknown
        pvalue = float(1 - sp_stats.chi2.cdf(wald, q))

    return {
        'statistic': f_stat,
        'pvalue': pvalue,
        'df': (q, int(df_resid) if np.isfinite(df_resid) else None),
        'hypothesis': hypothesis,
        'chi2': wald,
    }


def lincom(
    result,
    expression: str,
    alpha: float = 0.05,
) -> Dict[str, float]:
    """
    Estimate a linear combination of coefficients with inference.

    Parameters
    ----------
    result : EconometricResults or CausalResult
        Fitted model.
    expression : str
        Linear combination. Examples:
        - ``"x1 + x2"`` — beta_x1 + beta_x2
        - ``"x1 - x2"`` — beta_x1 - beta_x2
        - ``"2*x1 + 3*x2"`` — 2*beta_x1 + 3*beta_x2
    alpha : float, default 0.05
        Significance level.

    Returns
    -------
    dict
        ``{'estimate': float, 'se': float, 'z': float, 'pvalue': float,
           'ci': (lower, upper), 'expression': str}``

    Examples
    --------
    >>> result = sp.regress("y ~ x1 + x2", data=df)
    >>> sp.lincom(result, "x1 + x2")         # beta1 + beta2
    >>> sp.lincom(result, "x1 - x2")         # beta1 - beta2
    """
    params = result.params
    vcov = _get_vcov(result)

    # Parse expression into coefficient vector
    c = _parse_lincom(expression, params)

    beta = params.values
    estimate = float(c @ beta)
    se = float(np.sqrt(c @ vcov @ c))
    z = estimate / se if se > 0 else 0
    pvalue = float(2 * (1 - sp_stats.norm.cdf(abs(z))))
    z_crit = sp_stats.norm.ppf(1 - alpha / 2)

    return {
        'estimate': estimate,
        'se': se,
        'z': z,
        'pvalue': pvalue,
        'ci': (estimate - z_crit * se, estimate + z_crit * se),
        'expression': expression,
    }


# ======================================================================
# Parsing helpers
# ======================================================================

def _get_vcov(result):
    """Extract variance-covariance matrix."""
    se = result.std_errors
    if isinstance(se, pd.Series):
        return np.diag(se.values**2)
    return np.diag(np.array([se])**2)


def _parse_hypothesis(hypothesis: str, params: pd.Series):
    """
    Parse hypothesis string into R matrix and r vector.

    Examples:
        "x1 = 0" → R = [0, 1, 0, ...], r = [0]
        "x1 = x2" → R = [0, 1, -1, 0, ...], r = [0]
        "x1 = x2 = 0" → R = [[0,1,0,...], [0,0,1,...]], r = [0, 0]
        "x1 + x2 = 1" → R = [0, 1, 1, 0, ...], r = [1]
    """
    k = len(params)
    var_idx = {v: i for i, v in enumerate(params.index)}

    # Handle "x1 = x2 = 0" (joint test)
    parts = [p.strip() for p in hypothesis.split('=')]

    if len(parts) >= 3:
        # Joint test: x1 = x2 = ... = value
        try:
            rhs = float(parts[-1])
        except ValueError:
            rhs = 0.0

        vars_in_test = parts[:-1] if _is_number(parts[-1]) else parts
        if _is_number(parts[-1]):
            rhs_val = float(parts[-1])
        else:
            rhs_val = 0.0
            vars_in_test = parts

        R_rows = []
        r_vals = []
        for var_expr in vars_in_test:
            row = np.zeros(k)
            c = _parse_lincom(var_expr, params)
            R_rows.append(c)
            r_vals.append(rhs_val)

        return np.array(R_rows), np.array(r_vals)

    elif len(parts) == 2:
        lhs_str, rhs_str = parts[0].strip(), parts[1].strip()

        # Parse RHS
        if _is_number(rhs_str):
            rhs_val = float(rhs_str)
            c_lhs = _parse_lincom(lhs_str, params)
            return c_lhs.reshape(1, -1), np.array([rhs_val])
        else:
            # "x1 = x2" → "x1 - x2 = 0"
            c_lhs = _parse_lincom(lhs_str, params)
            c_rhs = _parse_lincom(rhs_str, params)
            R = (c_lhs - c_rhs).reshape(1, -1)
            return R, np.array([0.0])

    raise ValueError(f"Cannot parse hypothesis: '{hypothesis}'")


def _parse_lincom(expression: str, params: pd.Series) -> np.ndarray:
    """Parse a linear combination expression into coefficient vector."""
    k = len(params)
    var_idx = {v: i for i, v in enumerate(params.index)}
    c = np.zeros(k)

    # Tokenize: split on + and -, keeping the sign
    expr = expression.strip()
    # Normalize: ensure leading + or -
    if not expr.startswith('+') and not expr.startswith('-'):
        expr = '+' + expr

    tokens = []
    current = ''
    for ch in expr:
        if ch in '+-' and current.strip():
            tokens.append(current.strip())
            current = ch
        else:
            current += ch
    if current.strip():
        tokens.append(current.strip())

    for token in tokens:
        token = token.strip()
        if not token:
            continue

        if '*' in token:
            parts = token.split('*')
            coef_str = parts[0].strip()
            var_name = parts[1].strip()
            coef_val = float(coef_str)
        else:
            # Could be "+x1" or "-x1" or just "x1"
            if token[0] in '+-':
                sign = -1.0 if token[0] == '-' else 1.0
                var_name = token[1:].strip()
            else:
                sign = 1.0
                var_name = token
            coef_val = sign

        if var_name in var_idx:
            c[var_idx[var_name]] += coef_val

    return c


def _is_number(s: str) -> bool:
    try:
        float(s)
        return True
    except ValueError:
        return False
