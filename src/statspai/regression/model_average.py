"""Model selection and model averaging for least-squares regressions.

Given several candidate regressions of the same outcome, this module
computes the selection criteria (AIC, BIC, leave-one-out cross-validation)
and four sets of averaging weights:

* **Mallows** weights minimise the Mallows criterion of the averaged fit,
  ``w'E'Ew + 2 s^2 k'w``, over the unit simplex, with ``E`` the matrix of
  residuals of the candidates, ``k`` their numbers of coefficients and
  ``s^2`` the error variance from the largest model;
* **jackknife** weights minimise ``w'R'Rw``, the leave-one-out
  cross-validation criterion of the averaged fit, with ``R`` the matrix of
  leave-one-out residuals. They do not need ``s^2`` and allow
  heteroskedasticity;
* **smoothed AIC / BIC** weights are proportional to ``exp(-IC / 2)``.

The averaged coefficient vector is the weighted sum of the candidates'
vectors, a coefficient counting as zero in a model that leaves it out.
"""

from __future__ import annotations

from typing import Any, Callable, ClassVar, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
from scipy import optimize

from .._result_serialize import ResultProtocolMixin
from ..exceptions import DataInsufficient, MethodIncompatibility

__all__ = ["model_average", "ModelAverageResult"]

_METHODS = ("jma", "mma", "aic", "bic")


def _simplex_qp(quadratic: np.ndarray, linear: np.ndarray) -> np.ndarray:
    """``argmin w'Qw / 2 + c'w`` subject to ``w >= 0``, ``sum(w) = 1``.

    ``Q`` is a cross-product matrix, so the problem is convex. A sequential
    quadratic programme finds the support; the weights on it are then the
    exact solution of the equality-constrained problem, which removes the
    solver's tolerance from the answer.
    """
    m = len(linear)
    scale = max(float(np.abs(quadratic).max()), 1e-300)
    Q, c = quadratic / scale, linear / scale
    start = np.full(m, 1.0 / m)
    found = optimize.minimize(
        lambda w: 0.5 * w @ Q @ w + c @ w,
        start,
        jac=lambda w: Q @ w + c,
        method="SLSQP",
        bounds=[(0.0, 1.0)] * m,
        constraints=[{"type": "eq", "fun": lambda w: w.sum() - 1.0,
                      "jac": lambda w: np.ones(m)}],
        options={"ftol": 1e-15, "maxiter": 1000},
    )  # fmt: skip
    w = np.clip(np.asarray(found.x, dtype=float), 0.0, None)
    support = w > 1e-7
    for _ in range(m):
        idx = np.flatnonzero(support)
        s = len(idx)
        # stationarity on the support with the adding-up constraint
        kkt = np.zeros((s + 1, s + 1))
        kkt[:s, :s] = Q[np.ix_(idx, idx)]
        kkt[:s, s] = kkt[s, :s] = 1.0
        rhs = np.append(-c[idx], 1.0)
        try:
            solved = np.linalg.solve(kkt, rhs)
        except np.linalg.LinAlgError:
            solved = np.linalg.lstsq(kkt, rhs, rcond=None)[0]
        exact = np.zeros(m)
        exact[idx] = solved[:s]
        if (exact[idx] >= -1e-12).all():
            # a model outside the support must not lower the objective
            gradient = Q @ exact + c + solved[s]
            enter = np.flatnonzero(~support & (gradient < -1e-10))
            if enter.size == 0:
                return np.clip(exact, 0.0, None) / np.clip(exact, 0.0, None).sum()
            support[enter[np.argmin(gradient[enter])]] = True
        else:
            support[idx[np.argmin(exact[idx])]] = False
    return w / w.sum()


class ModelAverageResult(ResultProtocolMixin):
    """Outcome of :func:`model_average`.

    Attributes
    ----------
    table : pandas.DataFrame
        One row per candidate: number of coefficients ``k``, ``sigma2``
        (mean squared residual), ``aic``, ``bic``, ``cv`` (sum of squared
        leave-one-out residuals), and the four weight columns ``w_aic``,
        ``w_bic``, ``w_mma``, ``w_jma``.
    weights : pandas.Series
        The weights of the method asked for.
    params : pandas.Series
        Averaged coefficients over the union of the candidates' terms.
    selected : dict
        The candidate each criterion picks (``'aic'``, ``'bic'``, ``'cv'``).
    fitted : numpy.ndarray
        Averaged fitted values on the estimation sample.
    fits : list
        The candidate fits, in the order given.
    method : str
    n_obs : int

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> df = pd.DataFrame(rng.normal(size=(200, 3)), columns=["x1", "x2", "x3"])
    >>> df["y"] = 1 + df.x1 + 0.2 * df.x2 + rng.normal(size=200)
    >>> res = sp.model_average(["y ~ x1", "y ~ x1 + x2", "y ~ x1 + x2 + x3"], df)
    >>> round(float(res.weights.sum()), 12)
    1.0
    >>> list(res.params.index)
    ['Intercept', 'x1', 'x2', 'x3']
    """

    _citation_keys: ClassVar[Tuple[str, ...]] = ()

    def __init__(self, **fields: Any) -> None:
        self.method: str = fields.pop("method")
        self.table: pd.DataFrame = fields.pop("table")
        self.weights: pd.Series = fields.pop("weights")
        self.params: pd.Series = fields.pop("params")
        self.selected: Dict[str, str] = fields.pop("selected")
        self.fitted: np.ndarray = fields.pop("fitted")
        self.fits: List[Any] = fields.pop("fits")
        self.n_obs: int = fields.pop("n_obs")
        self._formulas: List[str] = fields.pop("formulas")

    def average(self, statistic: Callable[[Any], float]) -> float:
        """The weighted average of ``statistic(fit)`` over the candidates:
        a marginal effect, a prediction, a function of the coefficients
        that each model defines in its own way."""
        values = np.array([float(statistic(fit)) for fit in self.fits])
        return float(values @ self.weights.to_numpy())

    def predict(self, data: pd.DataFrame) -> np.ndarray:
        """Averaged predictions for new data."""
        out = np.zeros(len(data))
        for weight, fit in zip(self.weights.to_numpy(), self.fits):
            if weight > 0:
                out += weight * np.asarray(fit.predict(data), dtype=float)
        return out

    def summary(self) -> str:
        names = {"jma": "jackknife", "mma": "Mallows", "aic": "smoothed AIC",
                 "bic": "smoothed BIC"}  # fmt: skip
        lines = [
            f"Model averaging ({names[self.method]} weights)"
            f"{'':<12}Number of obs = {self.n_obs:>8,}",
            "",
            self.table.to_string(float_format=lambda v: f"{v:.4f}"),
            "",
            "Selected: "
            + ", ".join(f"{k.upper()} -> {v}" for k, v in self.selected.items()),
            "",
            "Averaged coefficients",
            self.params.to_string(float_format=lambda v: f"{v:.6f}"),
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()


def model_average(
    formulas: Sequence[str],
    data: pd.DataFrame,
    *,
    method: str = "jma",
    sigma2_model: Optional[int] = None,
    names: Optional[Sequence[str]] = None,
) -> ModelAverageResult:
    """Model selection criteria and averaging weights for candidate
    least-squares regressions.

    Parameters
    ----------
    formulas : list of str
        The candidate regressions, all with the same outcome:
        ``["y ~ x1", "y ~ x1 + x2", "y ~ x1 + x2 + x3"]``. They need not be
        nested.
    data : pandas.DataFrame
        The data. Rows with a missing value in any variable of any
        candidate are dropped, so that all are fitted on the same sample.
    method : {'jma', 'mma', 'aic', 'bic'}, default 'jma'
        Which weights define ``weights``, ``params`` and ``fitted``:
        jackknife model averaging, Mallows model averaging, or smoothed AIC
        / BIC. All four are in ``table`` whatever the choice.
    sigma2_model : int, optional
        Position of the candidate whose residual variance (with its
        degrees-of-freedom correction) enters the Mallows criterion.
        Default: the candidate with the most coefficients.
    names : list of str, optional
        Labels of the candidates. Default ``m1``, ``m2``, ...

    Returns
    -------
    ModelAverageResult
        ``table`` (criteria and weights), ``weights``, ``params``,
        ``selected``, ``fitted``, and ``average(statistic)`` /
        ``predict(data)``.

    Notes
    -----
    Mallows weights are optimal for the averaged fit's squared error under
    homoskedasticity; jackknife weights keep that property under
    heteroskedasticity, which is why they are the default. Smoothed BIC
    weights approximate posterior model probabilities and concentrate on
    one model as the sample grows.

    No standard errors are returned. An averaged or selected estimator is
    not normally distributed around the coefficient of any one model, and
    the usual interval computed after choosing a model undercovers
    [@hansen2022econometrics, chapter 28]. Use the weights for prediction
    and for a point estimate; for inference on a coefficient, fit the
    model chosen in advance.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> x = rng.uniform(0, 1, 300)
    >>> y = np.sin(3 * x) + rng.normal(scale=0.3, size=300)
    >>> df = pd.DataFrame({"x": x, "y": y})
    >>> candidates = ["y ~ x", "y ~ x + I(x**2)", "y ~ x + I(x**2) + I(x**3)"]
    >>> res = sp.model_average(candidates, df, method="mma")
    >>> res.table[["k", "w_mma", "w_jma"]].shape
    (3, 3)
    >>> at_half = res.predict(pd.DataFrame({"x": [0.5]}))
    >>> at_half.shape
    (1,)

    References
    ----------
    hansen2022econometrics
    """
    from .ols import regress

    method = str(method).lower()
    if method not in _METHODS:
        raise MethodIncompatibility(
            f"sp.model_average: method={method!r} is not one of "
            f"{', '.join(_METHODS)}.",
            recovery_hint="Use 'jma', 'mma', 'aic' or 'bic'.",
        )
    formulas = [str(f) for f in formulas]
    if len(formulas) < 2:
        raise MethodIncompatibility(
            "sp.model_average: at least two candidate formulas are needed.",
            recovery_hint="Pass a list of formulas with the same outcome.",
        )
    labels = list(names) if names is not None else [
        f"m{j + 1}" for j in range(len(formulas))
    ]  # fmt: skip
    if len(labels) != len(formulas) or len(set(labels)) != len(labels):
        raise MethodIncompatibility(
            "sp.model_average: names= must give one distinct label per formula.",
            recovery_hint="Drop names= to use m1, m2, ...",
        )

    # one sample for all candidates: fit, then keep the rows every fit used
    first = [regress(f, data=data) for f in formulas]
    outcomes = {fit.data_info.get("dependent_var") for fit in first}
    if len(outcomes) != 1:
        raise MethodIncompatibility(
            f"sp.model_average: the formulas have different outcomes ({outcomes}).",
            recovery_hint="Averaging compares fits of one outcome.",
        )
    common = first[0].data_info["sample_index"]
    for fit in first[1:]:
        common = common.intersection(fit.data_info["sample_index"])
    if any(len(fit.data_info["sample_index"]) != len(common) for fit in first):
        sample = data.loc[common]
        fits = [regress(f, data=sample) for f in formulas]
    else:
        fits = first
    n = len(common)

    residuals, loo, k, sigma2 = [], [], [], []
    for fit in fits:
        X = np.asarray(fit.data_info["X"], dtype=float)
        e = np.asarray(fit.data_info["residuals"], dtype=float)
        if n <= X.shape[1]:
            raise DataInsufficient(
                "sp.model_average: a candidate has as many coefficients as "
                "observations.",
                recovery_hint="Drop the largest candidates.",
            )
        q = np.linalg.qr(X)[0]
        leverage = (q**2).sum(axis=1)
        residuals.append(e)
        loo.append(e / (1.0 - leverage))
        k.append(X.shape[1])
        sigma2.append(float(e @ e) / n)
    E, R = np.column_stack(residuals), np.column_stack(loo)
    k_arr, s2 = np.asarray(k, dtype=float), np.asarray(sigma2)
    aic = n * np.log(2 * np.pi * s2) + 2 * k_arr
    bic = n * np.log(2 * np.pi * s2) + np.log(n) * k_arr
    cv = (R**2).sum(axis=0)

    def smoothed(ic: np.ndarray) -> np.ndarray:
        w = np.exp(-(ic - ic.min()) / 2.0)
        return np.asarray(w / w.sum())

    big = int(np.argmax(k_arr)) if sigma2_model is None else int(sigma2_model)
    if not 0 <= big < len(fits):
        raise MethodIncompatibility(
            f"sp.model_average: sigma2_model={sigma2_model} is not a position "
            f"in the list of {len(fits)} formulas.",
            recovery_hint="Positions start at 0.",
        )
    s2_big = float(E[:, big] @ E[:, big]) / (n - k_arr[big])
    weights = {
        "aic": smoothed(aic),
        "bic": smoothed(bic),
        "mma": _simplex_qp(E.T @ E, s2_big * k_arr),
        "jma": _simplex_qp(R.T @ R, np.zeros(len(fits))),
    }
    table = pd.DataFrame(
        {
            "k": k_arr.astype(int),
            "sigma2": s2,
            "aic": aic,
            "bic": bic,
            "cv": cv,
            "w_aic": weights["aic"],
            "w_bic": weights["bic"],
            "w_mma": weights["mma"],
            "w_jma": weights["jma"],
        },
        index=labels,
    )
    chosen = weights[method]
    terms: List[str] = []
    for fit in fits:
        terms.extend(t for t in map(str, fit.params.index) if t not in terms)
    params = pd.Series(0.0, index=terms)
    for weight, fit in zip(chosen, fits):
        params = params.add(weight * fit.params.rename(index=str), fill_value=0.0)
    y = np.asarray(fits[0].data_info["y"], dtype=float)
    return ModelAverageResult(
        method=method,
        table=table,
        weights=pd.Series(chosen, index=labels),
        params=params.reindex(terms),
        selected={
            "aic": labels[int(np.argmin(aic))],
            "bic": labels[int(np.argmin(bic))],
            "cv": labels[int(np.argmin(cv))],
        },
        fitted=y - E @ chosen,
        fits=fits,
        n_obs=n,
        formulas=formulas,
    )
