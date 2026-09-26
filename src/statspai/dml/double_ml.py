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
Treatment and Structural Parameters." *Econometrics Journal*, 21(1), C1-C68.
[@chernozhukov2018double]
"""

from typing import Any, List, Optional, Sequence, Union

import pandas as pd

from ..core.results import CausalResult
from ..exceptions import MethodIncompatibility
from ._base import _DoubleMLBase
from ._external_predictions import build_oof_provenance_payload
from .iivm import DoubleMLIIVM
from .irm import DoubleMLIRM
from .oof import OOFPredictions
from .pliv import DoubleMLPLIV
from .plr import DoubleMLPLR

_MODEL_REGISTRY = {
    "plr": DoubleMLPLR,
    "irm": DoubleMLIRM,
    "pliv": DoubleMLPLIV,
    "iivm": DoubleMLIIVM,
}


def dml(
    data: pd.DataFrame,
    y: str,
    treat: str,
    covariates: List[str],
    model: str = "plr",
    instrument: Optional[Union[str, List[str]]] = None,
    ml_g: Optional[Any] = None,
    ml_m: Optional[Any] = None,
    ml_r: Optional[Any] = None,
    n_folds: int = 5,
    n_rep: int = 1,
    alpha: float = 0.05,
    random_state: int = 42,
    sample_weight: Optional[Any] = None,
    fold_indices: Optional[Any] = None,
    score: Optional[str] = None,
    normalize_ipw: bool = False,
    trimming_threshold: float = 1e-2,
    *,
    cluster: Optional[str] = None,
    external_predictions: Optional["OOFPredictions"] = None,
    store_oof: bool = False,
    observation_ids: Optional[Sequence[str]] = None,
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
    ml_g, ml_m, ml_r : sklearn estimator or str, optional
        Learners for outcome / treatment / instrument nuisance. Pass any
        scikit-learn-compatible estimator, or one of the convenience
        string aliases (case-insensitive):

        - ``'rf'`` — random forest (200 trees)
        - ``'gbm'`` — gradient boosting (100 estimators, depth 3)
        - ``'lasso'`` / ``'ridge'`` — penalised linear (CV-tuned)
        - ``'linear'`` / ``'ols'`` — plain linear regression
        - ``'logistic'`` — logistic regression (classifier roles only)
        - ``'xgb'`` / ``'lgbm'`` — XGBoost / LightGBM (optional install)

        For binary treatment (``model='irm'``) ``ml_m`` is auto-coerced
        to the classifier variant; same for ``ml_r`` under ``'iivm'``.
        ``None`` falls back to the per-model gradient-boosting default.
    n_folds : int, default 5
    n_rep : int, default 1
        Repeated cross-fitting splits (median aggregation).
    alpha : float, default 0.05
    random_state : int, default 42
        Seed for the cross-fitting fold assignment. Repeat splits use
        ``random_state + rep`` so a single ``random_state`` fully
        determines the result.
    sample_weight : np.ndarray | pd.Series | str, optional
        Per-observation weights (e.g., survey/probability weights). Pass
        either a 1-D array of length ``len(data)`` or a column name
        present in ``data``. Supported for
        ``model in {'plr', 'irm', 'pliv', 'iivm'}``.
    fold_indices : array-like or str, optional
        Explicit cross-fit fold labels. Pass either a 1-D array of length
        ``len(data)`` or a column name. For internal fitting, labels must
        define exactly ``n_folds`` non-empty folds and ``n_rep`` must be 1.
        External repeated IRM predictions may use the same labels when that
        partition is equivalent to every caller repeat.
    score : str, optional
        Orthogonal score variant (DoubleML-compatible). Model-specific:

        - ``model='plr'`` : ``'partialling out'`` (default) or ``'IV-type'``.
          The IV-type score fits a third nuisance
          ``g(X) = E[Y - theta~ D | X]`` (cross-fitted on the same
          partition with a clone of ``ml_g``, ``theta~`` the preliminary
          partialling-out estimate) and solves
          ``E[(Y - D theta - g(X))(D - m(X))] = 0``, exactly as both
          DoubleML ports do. With linear nuisance learners it coincides
          with partialling out to machine precision; with regularised or
          nonlinear learners it is a distinct estimator. Versions before
          1.29.0 used ``l(X) = E[Y|X]`` in place of ``g(X)`` and did not
          reproduce DoubleML (see MIGRATION.md).
        - ``model='irm'`` : ``'ATE'`` (default) or ``'ATTE'`` (effect on
          the treated)

        ``None`` selects each model's default; the defaults reproduce the
        historical StatsPAI output exactly. Invalid for ``pliv`` /
        ``iivm`` beyond their single native score.
    normalize_ipw : bool, default False
        Self-normalize the inverse-propensity weights (Hájek-style) for
        the IPW-based models ``irm`` / ``iivm``. Matches DoubleML's
        ``normalize_ipw``. Ignored (and rejected) for ``plr`` / ``pliv``.
    trimming_threshold : float, default 0.01
        Symmetric propensity-score trimming for ``irm`` / ``iivm``: the
        estimated propensity is clipped to ``[t, 1 - t]``. The default
        ``0.01`` reproduces the historical clip and matches DoubleML's
        ``trimming_threshold`` with ``trimming_rule='truncate'``.
    cluster : str, optional
        Column identifying clusters (one-way). Cross-fitting folds are
        formed over whole clusters and the standard error is the
        cluster-robust DML variance of Chiang, Kato, Ma and Sasaki (2022),
        as DoubleML computes it for ``DoubleMLClusterData``; the estimate
        solves the fold-weighted score. Not available with
        ``external_predictions``.
    external_predictions : OOFPredictions, optional
        Python-only in-memory predictions for unweighted, unnormalised
        binary-treatment IRM ATE scoring. ``caller_declared`` training records
        are auditable declarations, not proof that training avoided leakage.
    store_oof : bool, default False
        Python-only retention switch for the supported IRM ATE scope. With
        ``store_oof=True``, use ``get_oof()`` or ``get_residuals()`` on the result.
        Ordinary result serialization omits these individual-level records.
    observation_ids : sequence of str, optional
        Python-only stable row IDs. Internal IDs are checked before complete-case
        filtering; external IDs must match the prediction record exactly.

    Returns
    -------
    CausalResult

    Notes
    -----
    Numerical parity: all four models are pinned against the upstream
    ``doubleml-for-py`` reference implementation in
    ``tests/external_parity/test_dml_python_parity.py`` — the
    partialling-out models (``plr``, ``pliv``) agree to machine
    precision under shared learners and folds. See the guide
    *"sp.dml and the DoubleML reference implementation"* in the docs
    for the full parity table and methodology.

    Declared scope boundaries (each fails loudly with a workaround in
    the error message — see the guide's "Scope and known limitations"):

    - ``pliv`` / ``iivm`` accept a **single scalar instrument**; for
      multiple excluded instruments build a first-stage index with
      ``sp.scalar_iv_projection`` first.
    - One treatment column per call (no multi-treatment joint
      inference; combine per-treatment calls with ``sp.romano_wolf``
      for simultaneous inference).
    - DML2 (pooled-moment) procedure only; PLR exposes the
      partialling-out score only.
    - Retained internal IRM fits require at least 10 rows per treatment arm in
      every training fold or raise ``DataInsufficient`` instead of using the
      legacy subgroup-mean fallback; internal explicit fold_indices require n_rep=1,
      while external repeated IRM predictions are accepted when the explicit fold
      partition is equivalent to every repeat.
    - Cluster-robust DML inference lives in ``sp.dml_panel``
      (unit-clustered panel PLR), not in this cross-sectional entry
      point.

    Default learners and coverage. When no learners are passed, the
    nuisances are fitted by gradient boosting (100 trees, depth 3). That
    default is a convenience, not a validated choice: on the Track B PLR
    design (n = 500, nonlinear nuisances) its regularisation bias is about
    half a Monte Carlo SD and the 95% interval covers 0.89 (0.94 at
    n = 2,000), while the true nuisance functions give 0.95 and a random
    forest or a spline Lasso about 0.94
    (``tests/coverage_monte_carlo/mechanisms/dml_plr_learners.py``). The
    orthogonal score makes nuisance error second order, not zero; pass
    learners suited to the problem, and check ``sp.validation_scope`` for
    which learner configurations carry evidence.

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
        raise MethodIncompatibility(
            f"dml: model must be one of {tuple(_MODEL_REGISTRY.keys())}, "
            f"got '{model}'",
            diagnostics={"model": model, "supported_models": tuple(_MODEL_REGISTRY)},
        )
    estimator_cls = _MODEL_REGISTRY[key]
    estimator = estimator_cls(
        data=data,
        y=y,
        treat=treat,
        covariates=covariates,
        instrument=instrument,
        ml_g=ml_g,
        ml_m=ml_m,
        ml_r=ml_r,
        n_folds=n_folds,
        n_rep=n_rep,
        alpha=alpha,
        random_state=random_state,
        sample_weight=sample_weight,
        fold_indices=fold_indices,
        score=score,
        normalize_ipw=normalize_ipw,
        trimming_threshold=trimming_threshold,
        cluster=cluster,
    )
    oof_provenance = build_oof_provenance_payload(
        external_predictions=external_predictions,
        store_oof=store_oof,
        observation_ids=observation_ids,
    )
    _result = estimator.fit(
        external_predictions=external_predictions,
        store_oof=store_oof,
        observation_ids=observation_ids,
    )
    try:
        from ..output._lineage import attach_provenance as _attach_prov

        provenance_params = {
            "y": y,
            "treat": treat,
            "covariates": list(estimator.covariates),
            "model": model,
            "instrument": instrument if isinstance(instrument, (str, list)) else None,
            "n_folds": n_folds,
            "n_rep": n_rep,
            "alpha": alpha,
            "random_state": int(random_state),
            "score": score,
            "normalize_ipw": normalize_ipw,
            "trimming_threshold": trimming_threshold,
            "fold_indices": fold_indices if isinstance(fold_indices, str) else None,
            "cluster": cluster,
            "sample_weight": sample_weight if isinstance(sample_weight, str) else None,
            # Keep learner objects out of the JSON-serialisable lineage record.
            "ml_g": type(ml_g).__name__ if ml_g is not None else None,
            "ml_m": type(ml_m).__name__ if ml_m is not None else None,
            "ml_r": type(ml_r).__name__ if ml_r is not None else None,
        }
        provenance_params.update(oof_provenance)
        _attach_prov(
            _result,
            function="sp.dml",
            params=provenance_params,
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

    For most workflows call the functional entry point :func:`dml`
    instead; this class wraps the same dispatcher behind a ``.fit()``
    method for code that predates the functional API.

    Examples
    --------
    >>> import numpy as np
    >>> import pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(42)
    >>> n = 400
    >>> x1 = rng.normal(size=n)
    >>> x2 = rng.normal(size=n)
    >>> d = 0.5 * x1 + rng.normal(size=n)
    >>> y = 1.0 * d + x1 + 0.5 * x2 + rng.normal(size=n)
    >>> df = pd.DataFrame({'y': y, 'd': d, 'x1': x1, 'x2': x2})
    >>> est = sp.DoubleML(df, y='y', treat='d', covariates=['x1', 'x2'],
    ...                   model='plr', ml_g='linear', ml_m='linear',
    ...                   n_folds=2)
    >>> est.model
    'plr'
    >>> res = est.fit()
    >>> isinstance(res, sp.CausalResult)
    True

    References
    ----------
    [@chernozhukov2018double]
    """

    _VALID_MODELS = tuple(_MODEL_REGISTRY.keys())

    def __init__(
        self,
        data: pd.DataFrame,
        y: str,
        treat: str,
        covariates: List[str],
        model: str = "plr",
        instrument: Optional[Union[str, List[str]]] = None,
        ml_g: Optional[Any] = None,
        ml_m: Optional[Any] = None,
        ml_r: Optional[Any] = None,
        n_folds: int = 5,
        n_rep: int = 1,
        alpha: float = 0.05,
        random_state: int = 42,
        sample_weight: Optional[Any] = None,
        fold_indices: Optional[Any] = None,
        score: Optional[str] = None,
        normalize_ipw: bool = False,
        trimming_threshold: float = 1e-2,
        cluster: Optional[str] = None,
    ):
        key = str(model).lower()
        if key not in _MODEL_REGISTRY:
            raise ValueError(  # pragma: no cover
                f"model must be one of {self._VALID_MODELS}, got '{model}'"
            )
        self.model = key
        self._impl: _DoubleMLBase = _MODEL_REGISTRY[key](
            data=data,
            y=y,
            treat=treat,
            covariates=covariates,
            instrument=instrument,
            ml_g=ml_g,
            ml_m=ml_m,
            ml_r=ml_r,
            n_folds=n_folds,
            n_rep=n_rep,
            alpha=alpha,
            random_state=random_state,
            sample_weight=sample_weight,
            fold_indices=fold_indices,
            score=score,
            normalize_ipw=normalize_ipw,
            trimming_threshold=trimming_threshold,
            cluster=cluster,
        )

    def fit(
        self,
        *,
        external_predictions: Optional["OOFPredictions"] = None,
        store_oof: bool = False,
        observation_ids: Optional[Sequence[str]] = None,
    ) -> CausalResult:
        """Fit, with optional Python-only retained IRM ATE records.

        ``store_oof=True`` enables ``get_oof()`` and ``get_residuals()``;
        ordinary serialization omits the individual-level records. The
        supported path is unweighted, unnormalised binary-treatment IRM ATE,
        and external ``caller_declared`` records remain declarations. Retained
        internal fits require at least 10 rows per treatment arm in each fold or
        raise ``DataInsufficient`` instead of using a subgroup-mean fallback;
        internal explicit fold_indices require n_rep=1, while external repeated IRM
        predictions are allowed when the explicit partition is equivalent for
        every repeat. ``external_predictions``, ``store_oof``, and
        ``observation_ids`` are Python-only controls.

        Parameters
        ----------
        external_predictions : OOFPredictions or None, default None
            In-memory predictions and declared training scopes, aligned to
            every analysis row, fold, and repeat.
        store_oof : bool, default False
            Retain complete IRM ATE records for explicit audit export.
        observation_ids : sequence of str or None, default None
            Unique input row identifiers; generated ordinals otherwise.

        Returns
        -------
        CausalResult
            The configured model's effect and uncertainty.

        Examples
        --------
        >>> import statspai as sp
        >>> callable(sp.DoubleML.fit)
        True

        References
        ----------
        See ``docs/dml_oof_audit.md`` for the audit contract.
        """
        return self._impl.fit(
            external_predictions=external_predictions,
            store_oof=store_oof,
            observation_ids=observation_ids,
        )

    # expose common attributes for legacy access
    @property
    def data(self) -> Any:
        return self._impl.data

    @property
    def y(self) -> Any:
        return self._impl.y

    @property
    def treat(self) -> Any:
        return self._impl.treat

    @property
    def covariates(self) -> Any:
        return self._impl.covariates

    @property
    def instrument(self) -> Any:
        return self._impl.instrument

    @property
    def n_folds(self) -> Any:
        return self._impl.n_folds

    @property
    def n_rep(self) -> Any:
        return self._impl.n_rep

    @property
    def alpha(self) -> Any:
        return self._impl.alpha

    @property
    def ml_g(self) -> Any:
        return self._impl.ml_g

    @property
    def ml_m(self) -> Any:
        return self._impl.ml_m

    @property
    def ml_r(self) -> Any:
        return self._impl.ml_r


# Citation
CausalResult._CITATIONS["dml"] = (
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
