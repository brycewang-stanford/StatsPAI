"""Summary indices of several outcomes: Kling-Liebman-Katz and Anderson.

Both indices standardize each component by the mean and standard deviation
of a reference group (the control group when ``treat`` is given) and
average the z-scores. KLK weights them equally; Anderson weights component
j by the j-th row sum of the inverse covariance matrix of the z-scores, so
components that carry little information beyond the others count less.

The Anderson index reproduces Stata ``swindex`` (Schwab, Janzen, Magnan
and Thompson; SSC, distribution date 2021-02-24) with ``normby()`` set to
the control indicator, including its treatment of missing components and
its default rescaling. Departures from that ado are listed in the
docstring of :func:`summary_index`.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional, Sequence, Union

import numpy as np
import pandas as pd

from .._input_validation import check_columns
from .._result_serialize import ResultProtocolMixin
from ..core._call_record import records_call
from ..exceptions import DataInsufficient, MethodIncompatibility, NumericalInstability

_METHODS = ("klk", "anderson")
_MISSING = ("available", "impute", "listwise")
_RESCALE = ("reference", "full", "none")


def _wmean(x: np.ndarray, w: np.ndarray) -> float:
    return float(np.sum(w * x) / np.sum(w))


def _wsd(x: np.ndarray, w: np.ndarray) -> float:
    """Standard deviation with Stata's analytic-weight convention.

    ``summarize [aweight=w]``: weights rescaled to sum to n, divisor n - 1.
    With unit weights this is the ordinary sample standard deviation.
    """
    n = x.size
    if n < 2:
        return float("nan")
    m = _wmean(x, w)
    return float(np.sqrt(np.sum(w * (x - m) ** 2) / np.sum(w) * n / (n - 1)))


def _wcov(z: np.ndarray, w: np.ndarray) -> np.ndarray:
    """Covariance of the columns of ``z`` under the same convention."""
    n = z.shape[0]
    m = (w[:, None] * z).sum(axis=0) / w.sum()
    d = z - m
    return (w[:, None] * d).T @ d / w.sum() * n / (n - 1)


@dataclass
class SummaryIndexResult(ResultProtocolMixin):
    """
    A summary index and the ingredients used to build it.

    Attributes
    ----------
    index : pd.Series
        The index, aligned to ``data.index``; missing where no component
        could be used.
    components : pd.DataFrame
        One row per outcome: ``reversed``, ``mean`` and ``sd`` (reference
        group, after any sign flip), ``weight`` (raw: 1 for KLK, the row
        sum of the inverse covariance for Anderson), ``share`` (weight over
        the sum of weights), ``n_reference`` and ``n_nonmissing``.
    method : str
    treat : str or None
    control_value : Any
    missing : str
    rescale : str
        'reference', 'full' or 'none'.
    index_mean, index_sd : float
        The constants used to rescale the index (NaN when not rescaled).
    covariance : pd.DataFrame or None
        Covariance of the z-scores over complete cases (Anderson only).
    n_obs : int
        Rows with a non-missing index.
    n_complete : int
        Rows with every component observed.

    Examples
    --------
    >>> import pandas as pd
    >>> import statspai as sp
    >>> df = pd.DataFrame({"d": [0, 0, 0, 1, 1, 1],
    ...                    "a": [1.0, 2.0, 3.0, 3.0, 4.0, 5.0],
    ...                    "b": [2.0, 1.0, 3.0, 4.0, 3.0, 5.0]})
    >>> r = sp.summary_index(df, ["a", "b"], treat="d")
    >>> r.index.round(3).tolist()
    [-0.5, -0.5, 1.0, 1.5, 1.5, 3.0]
    """

    _citation_keys = (
        "kling2007experimental",
        "anderson2008multiple",
        "schwab2020constructing",
    )

    index: pd.Series
    components: pd.DataFrame
    method: str
    treat: Optional[str]
    control_value: Any
    missing: str
    rescale: str
    index_mean: float
    index_sd: float
    n_obs: int
    n_complete: int
    covariance: Optional[pd.DataFrame] = None
    diagnostics: Dict[str, Any] = field(default_factory=dict)

    def to_frame(self) -> pd.DataFrame:
        """The per-outcome table (same as ``components``)."""
        return self.components.copy()

    def summary(self) -> str:
        """Plain-text description of the index and its components."""
        name = {"klk": "Kling-Liebman-Katz", "anderson": "Anderson (GLS)"}
        ref = (
            f"{self.treat} == {self.control_value!r}"
            if self.treat is not None
            else "full sample"
        )
        lines = [
            f"Summary index: {name[self.method]}",
            f"  reference group: {ref}",
            f"  missing components: {self.missing}",
            f"  rescaled: {self.rescale}",
            f"  observations with an index: {self.n_obs}"
            f" (complete cases: {self.n_complete})",
            "",
            self.components.to_string(float_format=lambda v: f"{v:.4f}"),
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()


def _resolve_rescale(rescale: Union[bool, str, None], method: str) -> str:
    if rescale is None:
        return "reference" if method == "anderson" else "none"
    if rescale is True:
        return "reference"
    if rescale is False:
        return "none"
    if rescale in _RESCALE:
        return str(rescale)
    raise MethodIncompatibility(
        f"rescale must be None, True, False, 'reference', 'full' or 'none'; "
        f"got {rescale!r}.",
        recovery_hint="Use rescale='reference' (swindex default), 'full' "
        "(swindex fullrescale) or 'none' (swindex norescale).",
    )


@records_call("summary_index")
def summary_index(
    data: pd.DataFrame,
    outcomes: Sequence[str],
    treat: Optional[str] = None,
    *,
    method: Literal["klk", "anderson"] = "klk",
    control_value: Any = 0,
    reverse: Optional[Sequence[str]] = None,
    weights: Optional[str] = None,
    missing: Literal["available", "impute", "listwise"] = "available",
    rescale: Union[bool, Literal["reference", "full", "none"], None] = None,
) -> SummaryIndexResult:
    """
    Kling-Liebman-Katz or Anderson summary index of several outcomes.

    Each outcome is signed so that higher is better (``reverse``) and
    standardized by the mean and standard deviation of the reference group:
    the rows with ``treat == control_value``, or every row when ``treat``
    is None. ``method='klk'`` averages the z-scores with equal weights
    [@kling2007experimental]. ``method='anderson'`` weights component j by
    the j-th row sum of the inverse of the z-scores' covariance matrix
    [@anderson2008multiple], as Stata ``swindex`` does
    [@schwab2020constructing].

    Parameters
    ----------
    data : pd.DataFrame
        One row per unit.
    outcomes : list of str
        Numeric component columns.
    treat : str, optional
        Treatment column. The reference group is ``treat == control_value``.
        Rows with a missing ``treat`` still get an index and enter the
        Anderson covariance, but are not in the reference group (as rows
        with a missing ``normby()`` variable in ``swindex``).
    method : {'klk', 'anderson'}, default 'klk'
    control_value : scalar, default 0
        Value of ``treat`` that marks the reference group.
    reverse : list of str, optional
        Outcomes for which lower is better; their sign is flipped before
        standardizing (``swindex, flip()``).
    weights : str, optional
        Analytic weights. Means, standard deviations, the covariance and the
        rescaling constants are then weighted as Stata's ``[aweight=]``
        does (weights rescaled to sum to n, divisor n - 1).
    missing : {'available', 'impute', 'listwise'}, default 'available'
        'available': the index is the weighted mean of the components the
        row has, with the weights of the missing ones dropped from the
        denominator. This is what the ``swindex`` code does. (Its help file
        says missing values are set to zero, which would keep their weight
        in the denominator; the code does not.) A row with no component is
        missing. 'impute': for a row with at least one component, each
        missing z-score is replaced by the mean z-score of that component
        in the row's ``treat`` group (all rows when ``treat`` is None; a
        missing ``treat`` is its own group), the rule described by Kling,
        Liebman and Katz. 'listwise': the index is missing unless every
        component is observed. In every case the Anderson covariance uses
        the complete cases only, as ``swindex`` does.
    rescale : {None, True, False, 'reference', 'full', 'none'}, default None
        Whether to standardize the final index. 'reference' (or True)
        subtracts its mean and divides by its standard deviation in the
        reference group (``swindex`` default); 'full' uses every row
        (``swindex, fullrescale``); 'none' (or False) leaves it as the
        weighted mean of z-scores (``swindex, norescale``). None means
        'reference' for Anderson and 'none' for KLK.

    Returns
    -------
    SummaryIndexResult
        ``.index`` (a Series aligned to ``data.index``) and
        ``.components`` (means, standard deviations and weights used).

    Raises
    ------
    ColumnNotFound
        A named column is not in ``data``.
    DataInsufficient
        A component has fewer than two non-missing values or no variation
        in the reference group, the reference group is empty, or the
        Anderson covariance has fewer than two complete cases.
    NumericalInstability
        The Anderson covariance matrix is singular (redundant components).

    Notes
    -----
    Standard deviations are sample standard deviations (divisor n - 1), as
    in Stata's ``summarize``. Departures from ``swindex``: it sets an index
    that is exactly 0 to missing (a by-product of initializing the sum at
    0); here such a row keeps its 0. It stores its intermediate variables
    as float under Stata's default ``set type float``, which moves its
    index by about 1e-7; with ``set type double`` the two agree to machine
    precision. With a singular covariance it uses a generalized inverse
    (``invsym``) that depends on the order of the variables; here that is
    an error.

    Examples
    --------
    >>> import pandas as pd
    >>> import statspai as sp
    >>> df = pd.DataFrame({"d": [0, 0, 0, 1, 1, 1],
    ...                    "a": [1.0, 2.0, 3.0, 3.0, 4.0, 5.0],
    ...                    "b": [2.0, 1.0, 3.0, 4.0, 3.0, 5.0]})
    >>> sp.summary_index(df, ["a", "b"], treat="d").index.round(3).tolist()
    [-0.5, -0.5, 1.0, 1.5, 1.5, 3.0]
    >>> r = sp.summary_index(df, ["a", "b"], treat="d", method="anderson")
    >>> r.components["share"].round(3).tolist()
    [0.5, 0.5]

    References
    ----------
    [@kling2007experimental], [@anderson2008multiple],
    [@schwab2020constructing]
    """
    if not isinstance(data, pd.DataFrame):
        raise MethodIncompatibility(
            "data must be a pandas DataFrame.",
            recovery_hint="Pass the data as a DataFrame with one row per unit.",
        )
    if isinstance(outcomes, str):
        outcomes = [outcomes]
    outcomes = list(outcomes)
    if not outcomes:
        raise MethodIncompatibility(
            "outcomes is empty.",
            recovery_hint="Name at least one component column.",
        )
    if len(set(outcomes)) != len(outcomes):
        raise MethodIncompatibility(
            "outcomes contains duplicates.",
            recovery_hint="Name each component once.",
        )
    if method not in _METHODS:
        raise MethodIncompatibility(
            f"method must be 'klk' or 'anderson'; got {method!r}.",
            recovery_hint="Use method='klk' (equal weights) or "
            "method='anderson' (inverse-covariance weights, as swindex).",
        )
    if missing not in _MISSING:
        raise MethodIncompatibility(
            f"missing must be one of {_MISSING}; got {missing!r}.",
            recovery_hint="Use missing='available' (swindex), 'impute' "
            "(group-mean imputation of z-scores) or 'listwise'.",
        )
    rescale_mode = _resolve_rescale(rescale, method)
    reverse = list(reverse or [])
    cols: Dict[str, Any] = {"outcomes": outcomes}
    if treat is not None:
        cols["treat"] = treat
    if weights is not None:
        cols["weights"] = weights
    check_columns(data, cols, function="summary_index")
    stray = [c for c in reverse if c not in outcomes]
    if stray:
        raise MethodIncompatibility(
            f"reverse names columns that are not in outcomes: {stray}.",
            recovery_hint="List in reverse only components of the index.",
        )
    non_numeric = [c for c in outcomes if not pd.api.types.is_numeric_dtype(data[c])]
    if non_numeric:
        raise MethodIncompatibility(
            f"Non-numeric outcome column(s): {non_numeric}.",
            recovery_hint="Encode the components as numbers before building "
            "the index.",
        )

    n = len(data)
    Y = data[outcomes].to_numpy(dtype=float).copy()
    sign = np.array([-1.0 if c in reverse else 1.0 for c in outcomes])
    Y *= sign
    if weights is not None:
        w = data[weights].to_numpy(dtype=float)
        if np.isnan(w).any() or (w < 0).any() or not np.isfinite(w).all():
            raise MethodIncompatibility(
                "weights must be finite and non-negative, with no missing values.",
                recovery_hint="Drop or fix the rows with invalid weights.",
            )
    else:
        w = np.ones(n)

    if treat is not None:
        tr = data[treat]
        ref = (tr == control_value).to_numpy() & tr.notna().to_numpy()
        if not ref.any():
            raise DataInsufficient(
                f"No row has {treat} == {control_value!r}; the reference group "
                "is empty.",
                recovery_hint="Set control_value to the value of treat that "
                "marks the control group.",
                diagnostics={"treat_values": [str(v) for v in tr.unique()[:10]]},
            )
    else:
        ref = np.ones(n, dtype=bool)

    K = len(outcomes)
    means = np.empty(K)
    sds = np.empty(K)
    n_ref = np.empty(K, dtype=int)
    for j, c in enumerate(outcomes):
        ok = ref & ~np.isnan(Y[:, j]) & (w > 0)
        n_ref[j] = int(ok.sum())
        if n_ref[j] < 2:
            raise DataInsufficient(
                f"Outcome {c!r} has {n_ref[j]} non-missing value(s) in the "
                "reference group; its standard deviation is undefined.",
                recovery_hint="Drop this component or use a reference group "
                "where it is observed.",
                diagnostics={"outcome": c, "n_reference": n_ref[j]},
            )
        means[j] = _wmean(Y[ok, j], w[ok])
        sds[j] = _wsd(Y[ok, j], w[ok])
        if not sds[j] > 1e-12 * max(1.0, float(np.max(np.abs(Y[ok, j])))):
            raise DataInsufficient(
                f"Outcome {c!r} does not vary in the reference group.",
                recovery_hint="Drop this component: a constant cannot be "
                "standardized.",
                diagnostics={"outcome": c, "sd": float(sds[j])},
            )
    Z = (Y - means) / sds
    obs = ~np.isnan(Z)
    complete = obs.all(axis=1)

    cov_df: Optional[pd.DataFrame] = None
    if method == "anderson":
        cc = complete & (w > 0)
        if cc.sum() < 2:
            raise DataInsufficient(
                f"Only {int(cc.sum())} row(s) observe every component; the "
                "Anderson weights need the covariance of the components.",
                recovery_hint="Drop sparsely observed components or use "
                "method='klk'.",
                diagnostics={"n_complete": int(cc.sum())},
            )
        S = _wcov(Z[cc], w[cc])
        cond = np.linalg.cond(S)
        if not np.isfinite(cond) or cond > 1e12:
            raise NumericalInstability(
                "The covariance matrix of the standardized components is "
                f"singular (condition number {cond:.3g}).",
                recovery_hint="Drop a component that is a linear combination "
                "of the others, or use method='klk'.",
                diagnostics={"condition_number": float(cond)},
                alternative_functions=["summary_index"],
            )
        wt = np.linalg.solve(S, np.ones(K))
        cov_df = pd.DataFrame(S, index=outcomes, columns=outcomes)
    else:
        wt = np.ones(K)

    if missing == "impute":
        groups = (
            data[treat].astype(object).where(data[treat].notna(), "__missing__")
            if treat is not None
            else pd.Series(0, index=data.index)
        ).to_numpy()
        has_any = obs.any(axis=1)
        Zi = Z.copy()
        for g in pd.unique(groups):
            rows = groups == g
            for j in range(K):
                donors = rows & obs[:, j] & (w > 0)
                fill = rows & ~obs[:, j] & has_any
                if fill.any():
                    if not donors.any():
                        raise DataInsufficient(
                            f"Outcome {outcomes[j]!r} is never observed in "
                            f"group {g!r}, so its missing values cannot be "
                            "imputed at the group mean.",
                            recovery_hint="Use missing='available'.",
                        )
                    Zi[fill, j] = _wmean(Z[donors, j], w[donors])
        Z, obs = Zi, ~np.isnan(Zi)

    num = np.where(obs, Z * wt, 0.0).sum(axis=1)
    den = np.where(obs, wt, 0.0).sum(axis=1)
    raw = np.full(n, np.nan)
    use = obs.any(axis=1) & (den != 0)
    if missing == "listwise":
        use &= obs.all(axis=1)
    raw[use] = num[use] / den[use]

    if rescale_mode == "none":
        idx, i_mean, i_sd = raw, float("nan"), float("nan")
    else:
        base = ~np.isnan(raw) & (w > 0)
        if rescale_mode == "reference":
            base &= ref
        if base.sum() < 2:
            raise DataInsufficient(
                "Fewer than two rows of the rescaling sample have an index.",
                recovery_hint="Use rescale='none' or rescale='full'.",
            )
        i_mean = _wmean(raw[base], w[base])
        i_sd = _wsd(raw[base], w[base])
        idx = (raw - i_mean) / i_sd

    components = pd.DataFrame(
        {
            "reversed": [c in reverse for c in outcomes],
            "mean": means,
            "sd": sds,
            "weight": wt,
            "share": wt / wt.sum(),
            "n_reference": n_ref,
            "n_nonmissing": (~np.isnan(Y)).sum(axis=0),
        },
        index=pd.Index(outcomes, name="outcome"),
    )
    diagnostics: Dict[str, Any] = {}
    if method == "anderson" and (wt < 0).any():
        diagnostics["negative_weights"] = [c for c, v in zip(outcomes, wt) if v < 0]
    return SummaryIndexResult(
        index=pd.Series(idx, index=data.index, name=f"{method}_index"),
        components=components,
        method=method,
        treat=treat,
        control_value=control_value,
        missing=missing,
        rescale=rescale_mode,
        index_mean=i_mean,
        index_sd=i_sd,
        n_obs=int(np.sum(~np.isnan(idx))),
        n_complete=int(complete.sum()),
        covariance=cov_df,
        diagnostics=diagnostics,
    )


__all__: List[str] = ["summary_index", "SummaryIndexResult"]
