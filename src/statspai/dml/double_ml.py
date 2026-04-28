"""
Double/Debiased Machine Learning (Chernozhukov et al. 2018) — dispatcher.

The per-model orthogonal scores live in dedicated files:

* :mod:`statspai.dml.plr`  — partially linear regression
* :mod:`statspai.dml.irm`  — interactive regression (binary D, ATE)
* :mod:`statspai.dml.pliv` — partially linear IV (endogenous D, IV Z)
* :mod:`statspai.dml.iivm` — interactive IV (binary D, binary Z, LATE)

This module keeps the legacy :func:`dml` / :class:`DoubleML` API by
dispatching ``model=`` to the appropriate estimator class.

References
----------
Chernozhukov, V., Chetverikov, D., Demirer, M., Duflo, E., Hansen, C.,
Newey, W., and Robins, J. (2018). "Double/Debiased Machine Learning for
Treatment and Structural Parameters." *Econometrics Journal*, 21(1), C1-C68. [@chernozhukov2018double]
"""

from typing import Optional, List, Any, Union
import pandas as pd

from ..core.results import CausalResult
from ._base import _DoubleMLBase
from .plr import DoubleMLPLR
from .irm import DoubleMLIRM
from .pliv import DoubleMLPLIV
from .iivm import DoubleMLIIVM


_MODEL_REGISTRY = {
    'plr': DoubleMLPLR,
    'irm': DoubleMLIRM,
    'pliv': DoubleMLPLIV,
    'iivm': DoubleMLIIVM,
}


def dml(
    data: pd.DataFrame,
    y: str,
    treat: str,
    covariates: List[str],
    model: str = 'plr',
    instrument: Optional[Union[str, List[str]]] = None,
    ml_g: Optional[Any] = None,
    ml_m: Optional[Any] = None,
    ml_r: Optional[Any] = None,
    n_folds: int = 5,
    n_rep: int = 1,
    alpha: float = 0.05,
) -> CausalResult:
    """
    Estimate causal effect using Double/Debiased Machine Learning.

    Parameters
    ----------
    data : pd.DataFrame
    y : str
        Outcome variable.
    treat : str
        Treatment variable.
    covariates : list of str
        High-dimensional controls X.
    model : str, default 'plr'
        DML model:

        - ``'plr'`` : partially linear (continuous or binary D)
        - ``'irm'`` : interactive regression (binary D; ATE via AIPW)
        - ``'pliv'`` : partially linear IV (endogenous D with instrument Z)
        - ``'iivm'`` : interactive IV (binary D, binary Z; LATE via Wald ratio)

    instrument : str, optional
        Scalar instrument Z. Required for ``model in {'pliv', 'iivm'}``.
    ml_g, ml_m, ml_r : sklearn estimators, optional
        Learners for outcome / treatment / instrument nuisance; sensible
        gradient-boosting defaults are selected per model.
    n_folds : int, default 5
    n_rep : int, default 1
        Repeated cross-fitting splits (median aggregation).
    alpha : float, default 0.05

    Returns
    -------
    CausalResult

    Examples
    --------
    >>> # Partially Linear Regression
    >>> result = sp.dml(df, y='wage', treat='training',
    ...                 covariates=['age', 'edu', 'exp'])

    >>> # Interactive Regression (binary treatment, ATE)
    >>> result = sp.dml(df, y='wage', treat='D', covariates=X_cols,
    ...                 model='irm')

    >>> # Partially Linear IV — endogenous D, instrument Z
    >>> result = sp.dml(df, y='earnings', treat='schooling',
    ...                 covariates=['age', 'father_edu'],
    ...                 model='pliv', instrument='quarter_of_birth')

    >>> # Interactive IV — binary D, binary Z (LATE)
    >>> result = sp.dml(df, y='earnings', treat='college',
    ...                 covariates=['age', 'ability'],
    ...                 model='iivm', instrument='lottery_win')
    """
    key = str(model).lower()
    if key not in _MODEL_REGISTRY:
        raise ValueError(
            f"model must be one of {tuple(_MODEL_REGISTRY.keys())}, got '{model}'"
        )
    estimator_cls = _MODEL_REGISTRY[key]
    estimator = estimator_cls(
        data=data, y=y, treat=treat, covariates=covariates,
        instrument=instrument,
        ml_g=ml_g, ml_m=ml_m, ml_r=ml_r,
        n_folds=n_folds, n_rep=n_rep, alpha=alpha,
    )
    _result = estimator.fit()
    try:
        from ..output._lineage import attach_provenance as _attach_prov
        _attach_prov(
            _result,
            function="sp.dml",
            params={
                "y": y, "treat": treat,
                "covariates": list(covariates),
                "model": model,
                "instrument": instrument
                              if isinstance(instrument, (str, list))
                              else None,
                "n_folds": n_folds, "n_rep": n_rep,
                "alpha": alpha,
                # Learner classes are objects — capture only their type
                # names so the provenance dict stays JSON-serialisable.
                "ml_g": type(ml_g).__name__ if ml_g is not None else None,
                "ml_m": type(ml_m).__name__ if ml_m is not None else None,
                "ml_r": type(ml_r).__name__ if ml_r is not None else None,
            },
            data=data,
            overwrite=False,
        )
    except Exception:  # pragma: no cover
        pass
    return _result


class DoubleML:
    """
    Legacy façade. Prefer the per-model classes directly for new code:
    :class:`DoubleMLPLR`, :class:`DoubleMLIRM`, :class:`DoubleMLPLIV`,
    :class:`DoubleMLIIVM`. Kept for backward compatibility.
    """

    _VALID_MODELS = tuple(_MODEL_REGISTRY.keys())

    def __init__(
        self,
        data: pd.DataFrame,
        y: str,
        treat: str,
        covariates: List[str],
        model: str = 'plr',
        instrument: Optional[Union[str, List[str]]] = None,
        ml_g: Optional[Any] = None,
        ml_m: Optional[Any] = None,
        ml_r: Optional[Any] = None,
        n_folds: int = 5,
        n_rep: int = 1,
        alpha: float = 0.05,
    ):
        key = str(model).lower()
        if key not in _MODEL_REGISTRY:
            raise ValueError(
                f"model must be one of {self._VALID_MODELS}, got '{model}'"
            )
        self.model = key
        self._impl: _DoubleMLBase = _MODEL_REGISTRY[key](
            data=data, y=y, treat=treat, covariates=covariates,
            instrument=instrument,
            ml_g=ml_g, ml_m=ml_m, ml_r=ml_r,
            n_folds=n_folds, n_rep=n_rep, alpha=alpha,
        )

    def fit(self) -> CausalResult:
        return self._impl.fit()

    # expose common attributes for legacy access
    @property
    def data(self): return self._impl.data
    @property
    def y(self): return self._impl.y
    @property
    def treat(self): return self._impl.treat
    @property
    def covariates(self): return self._impl.covariates
    @property
    def instrument(self): return self._impl.instrument
    @property
    def n_folds(self): return self._impl.n_folds
    @property
    def n_rep(self): return self._impl.n_rep
    @property
    def alpha(self): return self._impl.alpha
    @property
    def ml_g(self): return self._impl.ml_g
    @property
    def ml_m(self): return self._impl.ml_m
    @property
    def ml_r(self): return self._impl.ml_r


# Citation
CausalResult._CITATIONS['dml'] = (
    "@article{chernozhukov2018double,\n"
    "  title={Double/Debiased Machine Learning for Treatment and "
    "Structural Parameters},\n"
    "  author={Chernozhukov, Victor and Chetverikov, Denis and "
    "Demirer, Mert and Duflo, Esther and Hansen, Christian and "
    "Newey, Whitney and Robins, James},\n"
    "  journal={The Econometrics Journal},\n"
    "  volume={21},\n"
    "  number={1},\n"
    "  pages={C1--C68},\n"
    "  year={2018},\n"
    "  publisher={Oxford University Press}\n"
    "}"
)
