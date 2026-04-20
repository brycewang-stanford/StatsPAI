"""
Instrumental-variable helper utilities.

Small building blocks that plug around IV estimators which don't
accept vector-valued instruments. The canonical use case is using
:func:`sp.dml` with ``model='pliv'`` on a problem with multiple
excluded instruments: StatsPAI's PLIV reduced-form r(X) is scalar,
so we project the Z block onto its OLS first-stage fitted value and
pass that scalar index as the ``instrument`` argument.
"""

from typing import Optional, List, Union
import numpy as np
import pandas as pd


def scalar_iv_projection(
    data: pd.DataFrame,
    treat: str,
    instruments: List[str],
    covariates: Optional[List[str]] = None,
    new_col: Optional[str] = None,
    return_column: bool = False,
) -> Union[pd.DataFrame, pd.Series]:
    """
    Project a vector of instruments onto a scalar first-stage index.

    Fits the first-stage OLS regression::

        treat ~ intercept + Z1 + Z2 + ... + X1 + X2 + ...

    and returns the in-sample fitted values. This scalar "first-stage
    index" can then be passed as a single-instrument input to
    estimators that don't accept vector-valued Z (e.g. StatsPAI's
    PLIV / IIVM implementations).

    Parameters
    ----------
    data : pd.DataFrame
    treat : str
        Endogenous treatment variable name.
    instruments : list of str
        Excluded-instrument column names. Must have length ≥ 1.
    covariates : list of str, optional
        Exogenous controls to include in the first stage.
    new_col : str, optional
        Name of the column for the scalar index. Defaults to
        ``f'{treat}_iv_hat'``.
    return_column : bool, default False
        If True, return just the fitted pandas Series. If False,
        return a copy of ``data`` with the new column appended.

    Returns
    -------
    pd.DataFrame or pd.Series
        By default, ``data`` with the fitted-value column appended.
        With ``return_column=True``, returns the fitted-value Series.

    Examples
    --------
    >>> # PLIV with two excluded instruments — project first
    >>> df_aug = sp.scalar_iv_projection(
    ...     df, treat='schooling',
    ...     instruments=['quarter_of_birth', 'distance_to_college'],
    ...     covariates=['age', 'parent_edu'],
    ... )
    >>> result = sp.dml(df_aug, y='earnings', treat='schooling',
    ...                 covariates=['age', 'parent_edu'],
    ...                 model='pliv',
    ...                 instrument='schooling_iv_hat')

    Notes
    -----
    Projecting instruments onto a scalar index loses power relative to
    a full multi-instrument 2SLS when the instruments are heterogeneous,
    but the bias of the causal estimate remains valid under the same
    exogeneity conditions. A proper multi-instrument DML (vector r(X)
    and a Cragg-Donald-style first-stage test) is on the roadmap.
    """
    if not instruments:
        raise ValueError("instruments must be a non-empty list of column names")
    covariates = list(covariates or [])
    required = [treat] + list(instruments) + covariates
    missing = [c for c in required if c not in data.columns]
    if missing:
        raise ValueError(f"Columns not found in data: {missing}")

    df = data[required].dropna()
    if len(df) < len(required) + 5:
        raise ValueError(
            f"Too few complete rows ({len(df)}) for first-stage OLS "
            f"on {len(required)} variables."
        )

    Y = df[treat].values.astype(float)
    Z = df[list(instruments)].values.astype(float)
    X = df[covariates].values.astype(float) if covariates else np.zeros((len(df), 0))
    design = np.column_stack([np.ones(len(df)), Z, X])

    beta, *_ = np.linalg.lstsq(design, Y, rcond=None)
    fitted_in_sample = design @ beta

    # Build the fitted series aligned to the ORIGINAL data index.
    # Rows dropped by dropna() get NaN so the caller can detect them.
    fitted = pd.Series(np.nan, index=data.index, dtype=float)
    fitted.loc[df.index] = fitted_in_sample

    col_name = new_col or f'{treat}_iv_hat'
    if return_column:
        fitted.name = col_name
        return fitted

    out = data.copy()
    out[col_name] = fitted
    return out
