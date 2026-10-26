"""
``sp.bcf`` and the BCF-style CATE approximation behind ``sp.bcf_ordinal``.

Since 1.41 :func:`bcf` is the Bayesian causal forest itself: it calls
``sp.bart_causal(method='bcf')`` (BART priors, MCMC posterior) and returns
its result. Through 1.40 it was the ensemble approximation below, which
``sp.BayesianCausalForest``, ``sp.bcf_ordinal`` and
``sp.bcf_factor_exposure`` still use.

**The approximation is not Bayesian Causal Forest.** It has no BART prior
and no MCMC posterior. It borrows the structure of Hahn, Murray and
Carvalho (2020)
-- the estimated propensity enters the prognostic model, and the
prognostic and effect functions are fitted separately -- and fits both
with ordinary ensembles:

1. Estimate propensity score e(X) = P(D=1|X).
2. Fit prognostic model: mu(X, e(X)) via ensemble (random forest).
3. Fit treatment effect model: tau(X) by regressing residuals
   Y - mu(X) on covariates, for treated units only, with shrinkage.
4. Bootstrap for posterior-like uncertainty quantification.

The intervals are percentiles of a nonparametric bootstrap of this
procedure (the propensity is not re-estimated inside the bootstrap), not
posterior credible intervals, and the numbers are not expected to match R
``bcf``. Every call says so with a ``BCFApproximationWarning``.

References
----------
Hahn, P. R., Murray, J. S., & Carvalho, C. M. (2020).
"Bayesian Regression Tree Models for Causal Inference."
Bayesian Analysis, 15(3), 965-1056. [@hahn2020bayesian]
"""

from typing import TYPE_CHECKING, Any, List, Optional

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

from .._aliases import accepts_aliases
from .._input_validation import check_columns
from ..core.results import CausalResult
from ..exceptions import DataInsufficient, MethodIncompatibility, StatsPAIWarning

if TYPE_CHECKING:  # pragma: no cover
    from ..mcmc.causal_bart import BayesCausalResult


class BCFApproximationWarning(StatsPAIWarning):
    """The BCF-style ensemble approximation is not BART-based BCF."""

    code = "bcf_approximation"


# sklearn is imported lazily inside the methods that need it so that
# ``import statspai`` doesn't pull ~245 sklearn submodules through this
# file when the user never touches bcf.


# ======================================================================
# Public API
# ======================================================================


@accepts_aliases(seed="random_state")
def bcf(
    data: pd.DataFrame,
    y: str,
    treat: str,
    covariates: List[str],
    n_trees_mu: int = 200,
    n_trees_tau: int = 50,
    n_bootstrap: Optional[int] = None,
    n_folds: Optional[int] = None,
    alpha: float = 0.05,
    random_state: Optional[int] = 42,
    **kwargs: Any,
) -> "BayesCausalResult":
    """
    Bayesian causal forest (Hahn, Murray and Carvalho 2020), MCMC posterior.

    ``sp.bcf(...)`` is ``sp.bart_causal(..., method='bcf')`` with this
    function's argument names: a prognostic forest ``mu(x, pihat)`` and a
    more strongly regularised effect forest ``tau(x)`` under BART priors,
    sampled by MCMC. The result is a
    :class:`~statspai.mcmc.causal_bart.BayesCausalResult` (posterior mean
    ``.estimate``, posterior SD ``.se``, credible interval ``.ci``,
    per-row ``.cate``, chain diagnostics in ``.diagnostics``).

    .. versionchanged:: 1.41
       Through 1.40 ``sp.bcf`` was a random-forest / gradient-boosting
       approximation with bootstrap intervals that returned a
       ``CausalResult``; it now routes to ``sp.bart_causal(method='bcf')``
       and returns its result (announced with a ``FutureWarning`` in
       1.40). The approximation is still ``sp.BayesianCausalForest(...)
       .fit()``. See MIGRATION.md.

    Parameters
    ----------
    data : pd.DataFrame
        Input data.
    y : str
        Continuous outcome.
    treat : str
        Binary treatment variable (0/1).
    covariates : list of str
        Numeric covariate names (confounders and moderators).
    n_trees_mu : int, default 200
        Trees in the prognostic forest (``n_trees_control`` of
        ``sp.bart_causal``).
    n_trees_tau : int, default 50
        Trees in the effect forest (``n_trees_moderate``).
    n_bootstrap, n_folds : None
        Options of the pre-1.41 approximation. Passing either raises
        ``MethodIncompatibility``: the posterior is sampled, not
        bootstrapped (use ``draws=`` / ``chains=``).
    alpha : float, default 0.05
        The credible interval has level ``1 - alpha``.
    random_state : int, default 42
        Seed of the chains (``seed`` of ``sp.bart_causal``); also accepted
        as ``seed=``.
    **kwargs
        Forwarded to :func:`sp.bart_causal`: ``draws``, ``burnin``,
        ``thin``, ``chains``, ``estimand``, ``propensity``,
        ``sd_control``, ``sd_moderate``, ``numcut``, ``min_leaf``,
        ``cut_grid``.

    Returns
    -------
    BayesCausalResult

    Examples
    --------
    >>> import statspai as sp, numpy as np, pandas as pd
    >>> rng = np.random.default_rng(0)
    >>> n = 200
    >>> X = rng.normal(size=(n, 2))
    >>> treatment = rng.binomial(1, 0.5, n)
    >>> outcome = 2.0 * treatment + X @ np.array([1.0, -0.5]) + rng.normal(size=n)
    >>> df = pd.DataFrame(X, columns=["x1", "x2"])
    >>> df["outcome"], df["treatment"] = outcome, treatment
    >>> result = sp.bcf(df, y="outcome", treat="treatment",
    ...                 covariates=["x1", "x2"], draws=200, burnin=200,
    ...                 chains=1)
    >>> type(result).__name__
    'BayesCausalResult'
    >>> bool(result.ci[0] < 2.0 < result.ci[1])  # true ATE = 2
    True
    >>> result.cate.shape[0]                     # individual effects
    200

    References
    ----------
    [@hahn2020bayesian]
    """
    stale = [
        name
        for name, value in (("n_bootstrap", n_bootstrap), ("n_folds", n_folds))
        if value is not None
    ]
    if stale:
        raise MethodIncompatibility(
            f"sp.bcf: {stale} belonged to the pre-1.41 ensemble "
            "approximation. Since 1.41 sp.bcf samples the Bayesian causal "
            "forest posterior (sp.bart_causal(method='bcf')), which has no "
            "bootstrap or cross-fitting.",
            recovery_hint=(
                "Drop n_bootstrap= / n_folds= (set draws= / chains= for the "
                "posterior), or call sp.BayesianCausalForest(...).fit() for "
                "the pre-1.41 approximation and its numbers."
            ),
            diagnostics={"arguments": stale},
            alternative_functions=["sp.bart_causal", "sp.BayesianCausalForest"],
        )
    clash = sorted(
        {"method", "n_trees_control", "n_trees_moderate", "level"} & set(kwargs)
    )
    if clash:
        raise MethodIncompatibility(
            f"sp.bcf: {clash} are set by sp.bcf itself (method='bcf', "
            "n_trees_mu / n_trees_tau, alpha).",
            recovery_hint=(
                "Use n_trees_mu= / n_trees_tau= / alpha=, or call "
                "sp.bart_causal directly."
            ),
            diagnostics={"arguments": clash},
            alternative_functions=["sp.bart_causal"],
        )
    if not 0.0 < alpha < 1.0:
        raise MethodIncompatibility(
            f"sp.bcf: alpha must be in (0, 1); got {alpha!r}.",
            recovery_hint="Use alpha=0.05 for a 95% credible interval.",
        )
    from ..mcmc.causal_bart import bart_causal

    result = bart_causal(
        data,
        y,
        treat,
        covariates,
        method="bcf",
        n_trees_control=n_trees_mu,
        n_trees_moderate=n_trees_tau,
        level=1.0 - alpha,
        seed=random_state,
        **kwargs,
    )
    from ..output._lineage import attach_provenance

    attach_provenance(
        result,
        function="sp.bcf",
        params={
            "y": y,
            "treat": treat,
            "covariates": list(covariates),
            "n_trees_mu": n_trees_mu,
            "n_trees_tau": n_trees_tau,
            "alpha": alpha,
            "random_state": random_state,
            **kwargs,
        },
        data=data,
        overwrite=False,
    )
    return result


def _bcf_fit(
    data: pd.DataFrame,
    y: str,
    treat: str,
    covariates: List[str],
    n_trees_mu: int = 200,
    n_trees_tau: int = 50,
    n_bootstrap: int = 200,
    n_folds: int = 5,
    alpha: float = 0.05,
    random_state: int = 42,
) -> CausalResult:
    """The ensemble approximation that :func:`bcf` was through 1.40.

    ``sp.bcf_ordinal``, ``sp.bcf_factor_exposure`` and ``sp.did_bcf`` call
    this, so the 1.41 rerouting of ``sp.bcf`` to
    ``sp.bart_causal(method='bcf')`` does not change them.
    """
    import warnings

    warnings.warn(
        BCFApproximationWarning(
            "This is the BCF-style approximation (random-forest prognostic "
            "model + gradient-boosting effect model, bootstrap intervals; "
            "sp.bcf before 1.41), not Bayesian Causal Forest: there is no "
            "BART prior or MCMC posterior and the numbers do not reproduce "
            "R bcf.",
            recovery_hint=(
                "For the Bayesian causal forest itself (BART priors, MCMC "
                "posterior) use sp.bart_causal(method='bcf'); for forest "
                "CATE with valid inference sp.causal_forest; for a doubly "
                "robust ATE sp.dml / sp.aipw / sp.tmle."
            ),
            alternative_functions=[
                "sp.bart_causal",
                "sp.causal_forest",
                "sp.dml",
                "sp.tmle",
            ],
        ),
        stacklevel=3,
    )
    est = BayesianCausalForest(
        data=data,
        y=y,
        treat=treat,
        covariates=covariates,
        n_trees_mu=n_trees_mu,
        n_trees_tau=n_trees_tau,
        n_bootstrap=n_bootstrap,
        n_folds=n_folds,
        alpha=alpha,
        random_state=random_state,
    )
    _result = est.fit()
    try:
        from ..output._lineage import attach_provenance as _attach_prov

        _attach_prov(
            _result,
            function="sp.BayesianCausalForest",
            params={
                "y": y,
                "treat": treat,
                "covariates": list(covariates),
                "n_trees_mu": n_trees_mu,
                "n_trees_tau": n_trees_tau,
                "n_bootstrap": n_bootstrap,
                "n_folds": n_folds,
                "alpha": alpha,
                "random_state": random_state,
            },
            data=data,
            overwrite=False,
        )
    except Exception:  # pragma: no cover
        pass
    return _result


# ======================================================================
# BayesianCausalForest class
# ======================================================================


class BayesianCausalForest:
    """
    BCF-style CATE approximation (random forest + boosting; not BART).

    The engine of ``sp.bcf`` through 1.40 (since 1.41 ``sp.bcf`` is
    ``sp.bart_causal(method='bcf')``) and of ``sp.bcf_ordinal`` /
    ``sp.bcf_factor_exposure``; ``fit()`` returns a
    :class:`~statspai.core.results.CausalResult` and reproduces the
    pre-1.41 ``sp.bcf`` numbers.

    Parameters
    ----------
    data : pd.DataFrame
    y : str
    treat : str
    covariates : list of str
    n_trees_mu : int
    n_trees_tau : int
    n_bootstrap : int
    n_folds : int
    alpha : float
    random_state : int

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np, pandas as pd
    >>> from statspai.bcf.bcf import BayesianCausalForest
    >>> rng = np.random.default_rng(0)
    >>> n = 120
    >>> x1 = rng.normal(size=n)
    >>> x2 = rng.normal(size=n)
    >>> e = 1 / (1 + np.exp(-0.5 * x1))
    >>> treat = (rng.uniform(size=n) < e).astype(int)
    >>> y = 1.0 + 0.5 * x1 + x2 + treat * (1.0 + x1) + rng.normal(size=n)
    >>> df = pd.DataFrame({"y": y, "treat": treat, "x1": x1, "x2": x2})
    >>> est = BayesianCausalForest(
    ...     data=df, y="y", treat="treat", covariates=["x1", "x2"],
    ...     n_trees_mu=50, n_trees_tau=25, n_bootstrap=25, n_folds=2,
    ...     random_state=0,
    ... )
    >>> result = est.fit()
    >>> type(result).__name__
    'CausalResult'
    >>> result.model_info["cate"].shape
    (120,)
    """

    def __init__(
        self,
        data: pd.DataFrame,
        y: str,
        treat: str,
        covariates: List[str],
        n_trees_mu: int = 200,
        n_trees_tau: int = 50,
        n_bootstrap: int = 200,
        n_folds: int = 5,
        alpha: float = 0.05,
        random_state: int = 42,
    ):
        self.data = data
        self.y = y
        self.treat = treat
        self.covariates = covariates
        self.n_trees_mu = n_trees_mu
        self.n_trees_tau = n_trees_tau
        self.n_bootstrap = n_bootstrap
        self.n_folds = n_folds
        self.alpha = alpha
        self.random_state = random_state

    def fit(self) -> CausalResult:
        """Fit BCF and return treatment effect estimates."""
        # Prepare data
        cols = [self.y, self.treat] + self.covariates
        check_columns(
            self.data,
            {"y": self.y, "treat": self.treat, "covariates": list(self.covariates)},
            function="sp.BayesianCausalForest",
        )

        clean = self.data[cols].dropna()
        Y = clean[self.y].values.astype(np.float64)
        D = clean[self.treat].values.astype(np.float64)
        X = clean[self.covariates].values.astype(np.float64)
        n = len(Y)

        unique_d = np.unique(D)
        if not (len(unique_d) == 2 and set(unique_d.astype(int)) == {0, 1}):
            raise ValueError("Treatment must be binary (0/1)")

        from sklearn.base import clone
        from sklearn.ensemble import (
            GradientBoostingClassifier,
            GradientBoostingRegressor,
            RandomForestRegressor,
        )
        from sklearn.model_selection import KFold

        rng = np.random.RandomState(self.random_state)

        # Step 1: Estimate propensity scores via cross-fitting
        e_hat = np.zeros(n)
        kf = KFold(
            n_splits=self.n_folds,
            shuffle=True,
            random_state=self.random_state,
        )

        for train_idx, test_idx in kf.split(X):
            prop = GradientBoostingClassifier(
                n_estimators=100, max_depth=3, random_state=self.random_state
            )
            prop.fit(X[train_idx], D[train_idx])
            e_hat[test_idx] = np.clip(
                prop.predict_proba(X[test_idx])[:, 1], 0.025, 0.975
            )

        # Step 2: Fit prognostic model mu(X, e_hat)
        X_aug = np.column_stack([X, e_hat])

        mu_model = RandomForestRegressor(
            n_estimators=self.n_trees_mu,
            max_depth=6,
            min_samples_leaf=5,
            random_state=self.random_state,
        )
        # Fit mu on control observations
        mask0 = D == 0
        mu_model.fit(X_aug[mask0], Y[mask0])
        mu_hat = mu_model.predict(X_aug)

        # Step 3: Fit treatment effect model tau(X)
        # Residuals for treated: Y_i - mu_hat(X_i, e_hat_i)
        mask1 = D == 1
        residuals_1 = Y[mask1] - mu_hat[mask1]

        tau_model = GradientBoostingRegressor(
            n_estimators=self.n_trees_tau,
            max_depth=3,
            learning_rate=0.05,
            subsample=0.8,
            min_samples_leaf=10,
            random_state=self.random_state,
        )
        tau_model.fit(X[mask1], residuals_1)
        cate = tau_model.predict(X)

        ate = float(np.mean(cate))

        # Step 4: Bootstrap for uncertainty
        boot_cate = np.zeros((self.n_bootstrap, n))
        boot_ates = np.zeros(self.n_bootstrap)

        for b in range(self.n_bootstrap):
            idx = rng.choice(n, size=n, replace=True)
            Y_b, D_b, X_b, e_b = Y[idx], D[idx], X[idx], e_hat[idx]

            X_aug_b = np.column_stack([X_b, e_b])
            mask0_b = D_b == 0
            mask1_b = D_b == 1

            if mask0_b.sum() < 5 or mask1_b.sum() < 5:
                boot_cate[b] = cate
                boot_ates[b] = ate
                continue

            mu_b = clone(mu_model)
            mu_b.fit(X_aug_b[mask0_b], Y_b[mask0_b])
            mu_pred_b = mu_b.predict(X_aug_b)

            resid_b = Y_b[mask1_b] - mu_pred_b[mask1_b]
            tau_b = clone(tau_model)
            tau_b.fit(X_b[mask1_b], resid_b)

            boot_cate[b] = tau_b.predict(X)
            boot_ates[b] = np.mean(boot_cate[b])

        se = float(np.std(boot_ates, ddof=1))
        cate_sd = np.std(boot_cate, axis=0, ddof=1)

        # Credible intervals for CATE
        cate_lower = np.percentile(boot_cate, 100 * self.alpha / 2, axis=0)
        cate_upper = np.percentile(boot_cate, 100 * (1 - self.alpha / 2), axis=0)

        if se > 0:
            z_stat = ate / se
            pvalue = float(2 * sp_stats.norm.sf(abs(z_stat)))
        else:
            pvalue = 0.0

        z_crit = sp_stats.norm.ppf(1 - self.alpha / 2)
        ci = (ate - z_crit * se, ate + z_crit * se)

        model_info = {
            "architecture": "BCF-style approximation (RF mu + GBM tau)",
            "posterior": False,
            "approximation_note": (
                "Not Bayesian Causal Forest: no BART prior or MCMC "
                "posterior; intervals are bootstrap percentiles."
            ),
            "n_trees_mu": self.n_trees_mu,
            "n_trees_tau": self.n_trees_tau,
            "n_bootstrap": self.n_bootstrap,
            "propensity_mean": float(np.mean(e_hat)),
            "cate": cate,
            "cate_sd": cate_sd,
            "cate_lower": cate_lower,
            "cate_upper": cate_upper,
            "cate_mean": float(np.mean(cate)),
            "cate_median": float(np.median(cate)),
            "cate_std": float(np.std(cate)),
            "n_treated": int(np.sum(D == 1)),
            "n_control": int(np.sum(D == 0)),
        }

        self._mu_model = mu_model
        self._tau_model = tau_model
        self._cate = cate
        self._n_features = X.shape[1]

        self._boot_cate = boot_cate
        result = CausalResult(
            method="BCF-style approximation (RF + GBM; not BART)",
            estimand="ATE",
            estimate=ate,
            se=se,
            pvalue=pvalue,
            ci=ci,
            alpha=self.alpha,
            n_obs=n,
            detail=None,
            model_info=model_info,
            _citation_key="bcf",
        )
        # Bootstrap CATE draws (n_bootstrap x n), kept off model_info so the
        # serialised result stays small; consumers such as sp.did_bcf use
        # them to form standard errors of subgroup averages.
        result._bootstrap_cate = boot_cate  # type: ignore[attr-defined]
        return result

    def effect(self, X_new: Optional[np.ndarray] = None) -> np.ndarray:
        """Predict CATE for new observations."""
        if not hasattr(self, "_tau_model"):
            raise MethodIncompatibility(
                "BayesianCausalForest.effect() requires a fitted model.",
                recovery_hint="Call fit() before requesting CATE predictions.",
            )
        if X_new is None:
            return np.asarray(self._cate, dtype=np.float64).copy()
        try:
            X_new = np.asarray(X_new, dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise MethodIncompatibility(
                "BayesianCausalForest.effect() requires numeric covariates.",
                recovery_hint="Convert CATE prediction covariates to numeric values.",
            ) from exc
        expected = getattr(self, "_n_features", None)
        if X_new.ndim == 1:
            if expected is not None and expected > 1 and X_new.size == expected:
                X_new = X_new.reshape(1, -1)
            elif expected in (None, 1):
                X_new = X_new.reshape(-1, 1)
            else:
                raise MethodIncompatibility(
                    "BayesianCausalForest.effect(): one-dimensional input does "
                    "not match the fitted covariate count.",
                    recovery_hint="Pass X_new shaped (n_samples, n_covariates).",
                    diagnostics={
                        "n_values": int(X_new.size),
                        "expected_features": expected,
                    },
                )
        elif X_new.ndim != 2:
            raise MethodIncompatibility(
                "BayesianCausalForest.effect(): X_new must be a 1D or 2D array.",
                recovery_hint="Pass X_new shaped (n_samples, n_covariates).",
                diagnostics={"x_ndim": int(X_new.ndim)},
            )
        if X_new.shape[0] == 0:
            raise DataInsufficient(
                "BayesianCausalForest.effect() received no rows.",
                recovery_hint="Pass at least one prediction row.",
            )
        if expected is not None and X_new.shape[1] != expected:
            raise MethodIncompatibility(
                "BayesianCausalForest.effect(): covariate count does not "
                "match fit().",
                recovery_hint="Use the same covariates and order used at fit time.",
                diagnostics={
                    "expected_features": expected,
                    "observed_features": int(X_new.shape[1]),
                },
            )
        if not np.isfinite(X_new).all():
            raise MethodIncompatibility(
                "BayesianCausalForest.effect(): X_new contains NaN or infinite values.",
                recovery_hint="Drop or impute non-finite CATE prediction rows.",
            )
        return np.asarray(self._tau_model.predict(X_new), dtype=np.float64)


# ======================================================================
# Citation
# ======================================================================

CausalResult._CITATIONS["bcf"] = (
    "@article{hahn2020bayesian,\n"
    "  title={Bayesian Regression Tree Models for Causal Inference: "
    "Regularization, Confounding, and Heterogeneous Effects},\n"
    "  author={Hahn, P Richard and Murray, Jared S and Carvalho, "
    "Carlos M},\n"
    "  journal={Bayesian Analysis},\n"
    "  volume={15},\n"
    "  number={3},\n"
    "  pages={965--1056},\n"
    "  year={2020}\n"
    "}"
)
