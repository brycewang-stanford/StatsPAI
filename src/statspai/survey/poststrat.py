"""Poststratification of model predictions to a known population."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from .._result_serialize import ResultProtocolMixin
from ..exceptions import MethodIncompatibility


@dataclass
class PoststratResult(ResultProtocolMixin):
    """Result of :func:`poststratify`.

    Attributes
    ----------
    estimate : float
        Population-weighted average of the cell predictions (posterior
        mean when draws are available).
    sd : float
        Posterior standard deviation; ``nan`` for point predictions.
    lower, upper : float
        Central posterior interval.
    table : pd.DataFrame
        The estimate overall (and by group with ``by=``): ``estimate``,
        ``median``, ``sd``, ``mad_sd``, ``lower``, ``upper``,
        ``population``.
    draws : pd.DataFrame or None
        Posterior draws of each row of ``table``, one column per row.
    cells : pd.DataFrame
        The poststratification table with each cell's predicted mean.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> cells = pd.DataFrame({"g": ["a", "b"], "N": [300, 700]})
    >>> out = sp.poststratify([0.2, 0.6], cells, count="N")
    >>> round(out.estimate, 2)
    0.48
    """

    estimate: float
    sd: float
    lower: float
    upper: float
    level: float
    table: pd.DataFrame
    cells: pd.DataFrame = field(repr=False)
    draws: Optional[pd.DataFrame] = field(default=None, repr=False)

    _citation_keys = ("gelman2020regression",)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "estimate": self.estimate,
            "sd": None if np.isnan(self.sd) else self.sd,
            "lower": None if np.isnan(self.lower) else self.lower,
            "upper": None if np.isnan(self.upper) else self.upper,
            "level": self.level,
            "table": self.table.reset_index().to_dict(orient="records"),
        }

    def summary(self) -> str:
        return "Poststratified estimate\n" + str(
            self.table.to_string(float_format=lambda v: f"{v:.4g}")
        )

    def __repr__(self) -> str:  # pragma: no cover - cosmetic
        return self.summary()


def _cell_predictions(fit: Any, cells: pd.DataFrame) -> np.ndarray:
    """Draws by cells (or 1 by cells for point predictions)."""
    epred = getattr(fit, "posterior_epred", None)
    if callable(epred):
        return np.asarray(epred(cells), dtype=float)
    predict = getattr(fit, "predict", None)
    if callable(predict):
        return np.asarray(predict(cells), dtype=float).reshape(1, -1)
    arr = np.asarray(fit, dtype=float)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2:
        raise MethodIncompatibility(
            "Predictions must be a vector (one per cell) or a matrix of "
            "draws by cells."
        )
    return arr


def poststratify(
    fit: Any,
    cells: pd.DataFrame,
    count: str = "N",
    by: Optional[Union[str, Sequence[str]]] = None,
    level: float = 0.95,
) -> PoststratResult:
    """Average model predictions over the cells of a known population.

    A sample that over- or under-represents some groups gives a biased
    mean. Fit a regression of the outcome on the variables that define
    the groups, predict the mean of every population cell, and weight
    the predictions by the population cell counts:
    ``sum_j N_j theta_j / sum_j N_j``. With a Bayesian fit the same
    average is taken draw by draw, so the estimate carries the
    uncertainty of the regression.

    Parameters
    ----------
    fit : fitted model, array of predictions, or matrix of draws
        A model with ``posterior_epred(data)`` (``sp.bayes_regress``):
        full posterior. A classical model with ``predict(data)``: point
        estimate only. Or the predictions themselves, one per cell, or
        draws by cells.
    cells : DataFrame
        The poststratification table: one row per population cell with
        the columns the model's formula needs and the cell count.
    count : str, default 'N'
        Column of ``cells`` holding the population count (or share) of
        each cell.
    by : str or list of str, optional
        Also report the estimate within the levels of these columns of
        ``cells`` (subpopulations, e.g. states).
    level : float, default 0.95
        Mass of the posterior interval.

    Returns
    -------
    PoststratResult

    Notes
    -----
    The posterior describes uncertainty in the cell means given the
    model. It takes the population counts as known and assumes that,
    within a cell, respondents resemble the people they stand for; when
    the counts are estimates or nonresponse within cells is selective,
    the true uncertainty is larger.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> poll = pd.DataFrame({"pid": rng.choice(["R", "D", "I"], 600,
    ...                                        p=[0.5, 0.3, 0.2])})
    >>> share = poll["pid"].map({"R": 0.8, "D": 0.1, "I": 0.4})
    >>> poll["vote"] = (rng.uniform(size=600) < share).astype(int)
    >>> fit = sp.bayes_regress("vote ~ C(pid)", poll, model="logit",
    ...                        draws=4000, burnin=1000, seed=1)
    >>> cells = pd.DataFrame({"pid": ["R", "D", "I"], "N": [0.33, 0.36, 0.31]})
    >>> out = sp.poststratify(fit, cells, count="N")
    >>> bool(0.35 < out.estimate < 0.50)
    True

    References
    ----------
    gelman2020regression
    """
    if not isinstance(cells, pd.DataFrame):
        raise MethodIncompatibility("cells must be a DataFrame, one row per cell.")
    if count not in cells.columns:
        raise MethodIncompatibility(
            f"cells has no column {count!r} with the population counts."
        )
    if not 0.0 < level < 1.0:
        raise MethodIncompatibility(f"level must be in (0, 1); got {level}.")
    weights = cells[count].to_numpy(dtype=float)
    if np.any(~np.isfinite(weights)) or np.any(weights < 0) or weights.sum() <= 0:
        raise MethodIncompatibility(
            "Population counts must be non-negative, finite and not all zero."
        )
    pred = _cell_predictions(fit, cells)
    if pred.shape[1] != len(cells):
        raise MethodIncompatibility(
            f"Got predictions for {pred.shape[1]} cells; the table has "
            f"{len(cells)}. A model drops cells with missing values."
        )
    has_draws = pred.shape[0] > 1
    lo = (1.0 - level) / 2.0

    def one(mask: np.ndarray) -> Dict[str, float]:
        w = weights[mask]
        if w.sum() <= 0:
            raise MethodIncompatibility("A subpopulation has zero population count.")
        d = pred[:, mask] @ w / w.sum()
        row = {"estimate": float(d.mean()), "population": float(w.sum())}
        if has_draws:
            med = float(np.median(d))
            row.update(
                median=med,
                sd=float(d.std(ddof=1)),
                mad_sd=float(1.4826 * np.median(np.abs(d - med))),
                lower=float(np.quantile(d, lo)),
                upper=float(np.quantile(d, 1.0 - lo)),
            )
        else:
            row.update(
                median=float(d[0]),
                sd=float("nan"),
                mad_sd=float("nan"),
                lower=float("nan"),
                upper=float("nan"),
            )
        row["_draws"] = d  # type: ignore[assignment]
        return row

    rows: List[Dict[str, Any]] = [one(np.ones(len(cells), dtype=bool))]
    labels: List[str] = ["overall"]
    if by is not None:
        keys = [by] if isinstance(by, str) else list(by)
        missing = [k for k in keys if k not in cells.columns]
        if missing:
            raise MethodIncompatibility(f"by columns not in cells: {missing}.")
        grouped = cells.groupby(keys, sort=True, observed=True).indices
        for key, idx in grouped.items():
            mask = np.zeros(len(cells), dtype=bool)
            mask[np.asarray(idx)] = True
            rows.append(one(mask))
            label = key if isinstance(key, tuple) else (key,)
            labels.append(", ".join(f"{k}={v}" for k, v in zip(keys, label)))
    draws = (
        pd.DataFrame({lab: r["_draws"] for lab, r in zip(labels, rows)})
        if has_draws
        else None
    )
    cols = ["estimate", "median", "sd", "mad_sd", "lower", "upper", "population"]
    table = pd.DataFrame(
        [{c: r[c] for c in cols} for r in rows], index=pd.Index(labels, name="group")
    )
    out_cells = cells.copy()
    out_cells["prediction"] = pred.mean(axis=0)
    first = table.iloc[0]
    return PoststratResult(
        estimate=float(first["estimate"]),
        sd=float(first["sd"]),
        lower=float(first["lower"]),
        upper=float(first["upper"]),
        level=level,
        table=table,
        cells=out_cells,
        draws=draws,
    )


__all__ = ["PoststratResult", "poststratify"]
