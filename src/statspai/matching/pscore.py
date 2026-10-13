"""Propensity score with blocks and a balancing test (Becker and Ichino).

``sp.pscore`` fits the treatment model, optionally restricts the sample to
the region of common support, cuts the score into blocks inside which
treated and control units have the same average score, and tests in every
block whether each covariate has the same mean in the two groups. It is
the routine applied papers cite when they write "the balancing property
is satisfied", and the blocks are the strata of a stratification estimator.

The procedure is the one of Stata's ``pscore`` (Becker and Ichino 2002):

1. Start from ``n_blocks`` equal intervals of the score on [0, 1].
2. In a block holding both groups, test the equality of the mean score
   between treated and controls (two-sample t test, equal variances). If
   it is rejected at ``level``, split the block at its midpoint and test
   the halves; otherwise move to the next block.
3. With the blocks final, test each covariate in each block the same way.
   The balancing property holds when no test rejects.

A block with only treated or only control units is left as it is and is
not tested: there is nothing to compare.
"""

from __future__ import annotations

from typing import Any, ClassVar, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from scipy import stats

from .._result_serialize import ResultProtocolMixin
from ..core._covariates import expands_categorical_covariates as _expands_categorical
from ..core.results import SummaryText
from ..exceptions import AssumptionWarning, DataInsufficient, MethodIncompatibility
from ..exceptions import warn as _sp_warn
from ._binary_fit import fit_binary_index
from .match import _index_by_row

__all__ = ["pscore", "PScoreResult"]


def _single(value: float) -> float:
    return float(np.float32(value))


def _pooled_t_pvalue(x: np.ndarray, treated: np.ndarray) -> float:
    """Two-sided p-value of the equal-variance two-sample t test.

    ``nan`` when the test is not defined (no degrees of freedom, or no
    variation in either group), which the caller reads as "not rejected",
    as Stata's ``r(p) < level`` does with a missing ``r(p)``.
    """
    a, b = x[~treated], x[treated]
    df = a.size + b.size - 2
    if a.size == 0 or b.size == 0 or df <= 0:
        return float("nan")
    ss = float(np.sum((a - a.mean()) ** 2) + np.sum((b - b.mean()) ** 2))
    s2 = ss / df
    if not s2 > 0:
        return float("nan")
    t = (a.mean() - b.mean()) / np.sqrt(s2 * (1.0 / a.size + 1.0 / b.size))
    return float(2.0 * stats.t.sf(abs(t), df))


class PScoreResult(ResultProtocolMixin):
    """What :func:`pscore` returns.

    Attributes
    ----------
    pscore : pandas.Series
        The fitted score, on the index of the data (missing where a row
        was not in the estimation sample).
    block : pandas.Series
        The block number of each row, from 1; missing outside the
        estimation sample and outside the common support.
    support : pandas.Series
        Whether the row is in the region of common support (all ``True``
        without ``common_support``).
    support_range : tuple of float
        Smallest and largest score among the treated.
    n_blocks : int
        The final number of blocks.
    balanced : bool
        Whether no covariate differs between treated and controls in any
        block at ``level``. ``False`` when no block holds both treated
        and control units, since nothing could then be tested
        (``n_tested_blocks`` is zero).
    n_tested_blocks : int
        Blocks holding both treated and control units.
    separated : bool
        The covariates separate treated from controls, so the treatment
        model has no maximum-likelihood fit and the scores pile up at 0
        and 1.
    unbalanced : pandas.DataFrame
        The rejections: ``variable``, ``block`` and ``pvalue``.
    blocks : pandas.DataFrame
        One row per non-empty block: its ``lower`` and ``upper`` bound
        and the numbers of controls and treated units in it.
    coefficients : pandas.DataFrame
        The treatment model: ``coef``, ``se``, ``z`` and ``pvalue``, the
        constant last. A covariate that is collinear with earlier ones is
        listed with a zero coefficient and a missing standard error.
    loglik : float
    n_obs : int
        Rows in the estimation sample.

    Examples
    --------
    >>> import statspai as sp
    >>> df = sp.datasets.nsw_lalonde()
    >>> res = sp.pscore(df, "treat", ["age", "educ", "married", "re74"])
    >>> type(res).__name__, res.n_blocks == int(res.block.max())
    ('PScoreResult', True)
    >>> list(res.blocks.columns)
    ['block', 'lower', 'upper', 'n_control', 'n_treated']
    """

    _citation_keys: ClassVar[Tuple[str, ...]] = ("becker2002estimation",)

    def __init__(
        self,
        *,
        pscore: pd.Series,
        block: pd.Series,
        support: pd.Series,
        support_range: Tuple[float, float],
        n_blocks: int,
        unbalanced: pd.DataFrame,
        blocks: pd.DataFrame,
        coefficients: pd.DataFrame,
        loglik: float,
        n_obs: int,
        treat: str,
        covariates: List[str],
        ps_model: str,
        common_support: bool,
        level: float,
        n_tested_blocks: Optional[int] = None,
        separated: bool = False,
    ) -> None:
        self.pscore = pscore
        self.block = block
        self.support = support
        self.support_range = support_range
        self.n_blocks = int(n_blocks)
        self.unbalanced = unbalanced
        self.n_tested_blocks = (
            int((blocks[["n_control", "n_treated"]] > 0).all(axis=1).sum())
            if n_tested_blocks is None
            else int(n_tested_blocks)
        )
        self.separated = bool(separated)
        self.balanced = bool(unbalanced.empty) and self.n_tested_blocks > 0
        self.blocks = blocks
        self.coefficients = coefficients
        self.loglik = float(loglik)
        self.n_obs = int(n_obs)
        self.treat = treat
        self.covariates = list(covariates)
        self.ps_model = ps_model
        self.common_support = bool(common_support)
        self.level = float(level)

    def assign(
        self,
        data: pd.DataFrame,
        pscore: str = "pscore",
        block: Optional[str] = "block",
        support: Optional[str] = "comsup",
    ) -> pd.DataFrame:
        """A copy of ``data`` with the score, block and support columns.

        Pass ``None`` for a column that is not wanted. The names are the
        ones Stata's ``pscore`` takes in ``pscore()`` and ``blockid()``
        and the ``comsup`` it creates.
        """
        out = data.copy()
        out[pscore] = self.pscore.reindex(out.index)
        if block is not None:
            out[block] = self.block.reindex(out.index)
        if support is not None:
            out[support] = self.support.reindex(out.index).astype(float)
        return out

    def summary(self) -> SummaryText:
        lo, hi = self.support_range
        lines = [
            f"Propensity score ({self.ps_model}) of {self.treat}",
            "=" * 62,
            self.coefficients.to_string(float_format=lambda v: f"{v: .6g}"),
            "",
            f"Log likelihood = {self.loglik:.5f}    N = {self.n_obs}",
        ]
        if self.common_support:
            lines.append(f"Region of common support: [{lo:.8g}, {hi:.8g}]")
        lines += [
            "",
            f"Final number of blocks: {self.n_blocks}",
            self.blocks.to_string(index=False),
            "",
        ]
        if self.separated:
            lines.append(
                "The covariates separate treated from controls: the scores "
                "are not maximum-likelihood estimates."
            )
        if self.n_tested_blocks == 0:
            lines.append(
                "The balancing property could not be tested: no block holds "
                "both treated and control units."
            )
        elif self.balanced:
            lines.append(f"The balancing property is satisfied (level {self.level:g}).")
        else:
            lines.append("The balancing property is not satisfied:")
            for row in self.unbalanced.itertuples():
                lines.append(
                    f"  {row.variable} is not balanced in block {row.block} "
                    f"(p = {row.pvalue:.4f})"
                )
        return SummaryText("\n".join(lines))

    def __repr__(self) -> str:  # pragma: no cover - display only
        return str(self.summary())


@_expands_categorical("covariates")
def pscore(
    data: pd.DataFrame,
    treat: str,
    covariates: Union[Sequence[str], str],
    *,
    ps_model: str = "logit",
    common_support: bool = False,
    level: float = 0.01,
    n_blocks: int = 5,
) -> PScoreResult:
    """Propensity score, its blocks and the balancing test.

    Fits the treatment model, cuts the fitted score into blocks in which
    treated and control units have the same mean score, and tests in every
    block whether each covariate is balanced (Becker and Ichino 2002;
    Stata ``pscore``).

    Parameters
    ----------
    data : pandas.DataFrame
    treat : str
        The 0/1 treatment.
    covariates : list of str
        The covariates of the treatment model. Powers and interactions
        are columns of ``data``. Categorical columns (``category`` dtype,
        strings, ``C(col)``) are expanded into indicators, the first
        level as the base, and each indicator is tested for balance;
        integer columns are taken as numeric.
    ps_model : {'logit', 'probit'}, default 'logit'
        The treatment model. Stata's ``pscore`` fits a probit unless it is
        given ``logit``.
    common_support : bool, default False
        Keep the rows whose score lies between the smallest and the
        largest score of the treated (``pscore, comsup``). The blocks and
        the tests are computed on those rows.
    level : float, default 0.01
        Significance level of the tests. The default is the routine's: a
        covariate is called unbalanced only on strong evidence, because
        many tests are run.
    n_blocks : int, default 5
        Number of equal intervals of [0, 1] the search starts from.

    Returns
    -------
    PScoreResult
        ``pscore``, ``block`` and ``support`` on the index of ``data``;
        ``n_blocks``, ``balanced``, ``unbalanced``, ``blocks`` and the
        ``coefficients`` of the treatment model. ``.assign(data)`` adds
        the three columns to the data.

    Notes
    -----
    The tests are two-sample t tests with a pooled variance. A block with
    only treated or only control units is not split and not tested. When
    no block holds both groups nothing is tested: ``balanced`` is then
    ``False`` and a warning says so. The top block includes a score of
    exactly one.

    When the covariates separate treated from controls the treatment
    model has no maximum-likelihood fit. A ``ConvergenceWarning`` is
    issued and ``separated`` is set; the scores, which sit at 0 and 1,
    are of no use for matching or weighting.

    The fitted score is the maximum-likelihood one. With covariates that
    predict treatment almost perfectly Stata stops its iterations earlier,
    and scores of order 1e-8 can differ from Stata's in the third digit;
    the blocks do not depend on it.

    Examples
    --------
    >>> import statspai as sp
    >>> df = sp.datasets.nsw_lalonde()
    >>> res = sp.pscore(df, "treat", ["age", "educ", "married", "re74"])
    >>> res.n_blocks >= 5, bool(res.pscore.between(0, 1).all())
    (True, True)
    >>> sorted(res.assign(df).columns.difference(df.columns))
    ['block', 'comsup', 'pscore']

    References
    ----------
    [@becker2002estimation]
    """
    if not isinstance(data, pd.DataFrame):
        raise MethodIncompatibility(
            "pscore: data must be a pandas DataFrame.",
            recovery_hint="Pass the DataFrame holding the treatment and covariates.",
        )
    if isinstance(covariates, str):
        covariates = [covariates]
    covariates = list(covariates)
    missing = [c for c in [treat] + covariates if c not in data.columns]
    if missing:
        raise MethodIncompatibility(
            f"pscore: columns not found in data: {missing}",
            recovery_hint="Check the treat and covariates column names.",
            diagnostics={"missing_columns": missing},
        )
    if not covariates:
        raise MethodIncompatibility(
            "pscore: at least one covariate is needed.",
            recovery_hint="Pass the covariates of the treatment model.",
        )
    ps_model = str(ps_model).lower()
    if ps_model not in ("logit", "probit"):
        raise MethodIncompatibility(
            f"pscore: ps_model must be 'logit' or 'probit', got {ps_model!r}.",
            recovery_hint="Use ps_model='logit' or ps_model='probit'.",
        )
    if not 0 < level < 1:
        raise MethodIncompatibility(
            f"pscore: level must lie strictly between 0 and 1, got {level!r}.",
            recovery_hint="The default is 0.01.",
        )
    n_blocks = int(n_blocks)
    if n_blocks < 1:
        raise MethodIncompatibility(
            f"pscore: n_blocks must be a positive integer, got {n_blocks!r}.",
            recovery_hint="The default is 5.",
        )

    clean = data[[treat] + covariates].dropna()
    t_raw = clean[treat].to_numpy(dtype=float)
    if not set(np.unique(t_raw)) <= {0.0, 1.0}:
        raise MethodIncompatibility(
            f"pscore: {treat!r} must be binary (0/1).",
            recovery_hint="Recode the treatment to 0/1.",
            diagnostics={"values": sorted(map(float, np.unique(t_raw)))[:10]},
        )
    treated = t_raw == 1
    if treated.all() or not treated.any():
        raise DataInsufficient(
            "pscore: both treated and control observations are needed.",
            recovery_hint="Check the treatment column.",
            diagnostics={"n_treated": int(treated.sum()), "n": int(treated.size)},
        )
    text = [
        c
        for c in covariates
        if isinstance(clean[c].dtype, pd.CategoricalDtype)
        or not (
            pd.api.types.is_numeric_dtype(clean[c])
            or pd.api.types.is_bool_dtype(clean[c])
        )
    ]
    if text:
        raise MethodIncompatibility(
            f"pscore: covariates {text} are not numeric.",
            recovery_hint="Pass covariates as a list so that categorical "
            "columns are expanded into indicators.",
            diagnostics={"columns": text},
        )
    X = clean[covariates].to_numpy(dtype=float)
    fit = fit_binary_index(X, t_raw, ps_model)
    beta = fit["beta"]
    # one evaluation per distinct row: units with the same covariates get
    # the same score to the last bit, so a later match on it sees the tie
    index = _index_by_row(np.column_stack([np.ones(len(X)), X]), beta)
    if ps_model == "probit":
        score = stats.norm.cdf(index)
    else:
        with np.errstate(over="ignore"):
            score = 1.0 / (1.0 + np.exp(-index))

    se = np.sqrt(np.clip(np.diag(fit["vcov"]), 0.0, None))
    omitted = set(fit["omitted"])
    se[[j + 1 for j in omitted]] = np.nan
    with np.errstate(divide="ignore", invalid="ignore"):
        z = beta / se
    order = list(range(1, len(beta))) + [0]
    coefficients = pd.DataFrame(
        {
            "coef": beta[order],
            "se": se[order],
            "z": z[order],
            "pvalue": 2.0 * stats.norm.sf(np.abs(z[order])),
        },
        index=covariates + ["_cons"],
    )

    lo, hi = float(score[treated].min()), float(score[treated].max())
    in_support = np.ones(score.size, dtype=bool)
    if common_support:
        in_support = (score >= lo) & (score <= hi)
    use = in_support

    # blocks: number, lower and upper bound of each row in the sample
    # (the bounds are held in single precision, as the Stata routine holds
    # them: a score within 1e-8 of a bound falls on the same side)
    block = np.zeros(score.size, dtype=int)
    lower = np.zeros(score.size)
    upper = np.zeros(score.size)
    width = 1.0 / n_blocks
    for i in range(1, n_blocks + 1):
        # the top block is closed above: a score of exactly one is in it
        below = (score < i * width) if i < n_blocks else (score <= 1.0)
        inside = use & (score >= (i - 1) * width) & below
        block[inside] = i
        lower[inside], upper[inside] = _single((i - 1) * width), _single(i * width)
    total = n_blocks
    i = 1
    while i <= total:
        here = use & (block == i)
        n_t, n_c = int((here & treated).sum()), int((here & ~treated).sum())
        if n_t == 0 or n_c == 0:
            i += 1
            continue
        p = _pooled_t_pvalue(score[here], treated[here])
        if not p < level:
            i += 1
            continue
        # the mean score differs: halve the block and test the halves
        block[use & (block > i)] += 1
        total += 1
        split = (lower[here][0] + upper[here][0]) / 2.0
        upper_half = here & (score >= split) & (score <= upper)
        block[upper_half] += 1
        upper[use & (block == i)] = _single(split)
        lower[use & (block == i + 1)] = _single(split)

    rejections: List[Dict[str, Any]] = []
    for i in range(1, total + 1):
        here = use & (block == i)
        if not (here & treated).any() or not (here & ~treated).any():
            continue
        for j, name in enumerate(covariates):
            p = _pooled_t_pvalue(X[here, j], treated[here])
            if p < level:
                rejections.append({"variable": name, "block": i, "pvalue": p})
    unbalanced = pd.DataFrame(rejections, columns=["variable", "block", "pvalue"])

    rows = []
    for i in sorted(set(block[use].tolist())):
        here = use & (block == i)
        rows.append(
            {
                "block": i,
                "lower": float(lower[here][0]),
                "upper": float(upper[here][0]),
                "n_control": int((here & ~treated).sum()),
                "n_treated": int((here & treated).sum()),
            }
        )
    blocks = pd.DataFrame(
        rows, columns=["block", "lower", "upper", "n_control", "n_treated"]
    )

    n_tested = int(((blocks["n_control"] > 0) & (blocks["n_treated"] > 0)).sum())
    if n_tested == 0:
        _sp_warn(
            AssumptionWarning,
            "pscore: no block holds both treated and control units, so the "
            "balancing property could not be tested and balanced is False. "
            "The scores of the two groups do not overlap.",
            recovery_hint="Drop the covariate that predicts treatment "
            "perfectly, or restrict the sample to where both groups occur.",
            stacklevel=3,
        )
    block_out = np.where(use, block.astype(float), np.nan)
    return PScoreResult(
        pscore=pd.Series(score, index=clean.index, name="pscore").reindex(data.index),
        block=pd.Series(block_out, index=clean.index, name="block").reindex(data.index),
        support=pd.Series(in_support, index=clean.index, name="comsup")
        .reindex(data.index)
        .fillna(False)
        .astype(bool),
        support_range=(lo, hi),
        # the highest block that holds an observation, as Stata reports it
        n_blocks=int(block[use].max()),
        unbalanced=unbalanced,
        blocks=blocks,
        coefficients=coefficients,
        loglik=fit["loglik"],
        n_obs=int(score.size),
        treat=treat,
        covariates=covariates,
        ps_model=ps_model,
        common_support=common_support,
        level=level,
        n_tested_blocks=n_tested,
        separated=bool(fit.get("separated", False)),
    )
