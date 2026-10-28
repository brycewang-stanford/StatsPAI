"""Result object of ``sp.ddc``."""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
from scipy import stats

from ..._result_serialize import ResultProtocolMixin
from ...exceptions import MethodIncompatibility
from ._core import controlled_transition, solve_value

_TITLES = {
    "nfxp": "Nested fixed point (Rust 1987)",
    "ccp": "Two-step CCP pseudo-likelihood (Hotz-Miller 1993)",
    "npl": "Nested pseudo-likelihood (Aguirregabiria-Mira 2002)",
}


def stationary_distribution(T: np.ndarray) -> np.ndarray:
    """Stationary distribution of a row-stochastic matrix (least squares)."""
    S = T.shape[0]
    A = np.vstack([T.T - np.eye(S), np.ones((1, S))])
    b = np.r_[np.zeros(S), 1.0]
    pi = np.linalg.lstsq(A, b, rcond=None)[0]
    pi = np.clip(pi, 0.0, None)
    return np.asarray(pi / pi.sum())


class DDCResult(ResultProtocolMixin):
    """Fitted dynamic discrete choice model.

    Attributes
    ----------
    method : str
        ``'nfxp'``, ``'ccp'`` or ``'npl'``.
    params, se : pd.Series
        Utility parameters and their standard errors.
    vcov : pd.DataFrame
        Variance matrix of ``params``.
    loglik : float
        Partial (choice) log-likelihood at ``params``, evaluated at the
        model's own fixed-point choice probabilities for every method.
    ccp : pd.DataFrame
        Model choice probabilities, one row per state, one column per action.
    ccp_first_stage : pd.DataFrame or None
        Smoothed frequency CCPs used to start CCP / NPL.
    value : pd.Series
        Integrated value function ``log sum_a exp v(s, a)`` (Euler's constant
        omitted).
    choice_values : pd.DataFrame
        ``v(s, a)``.
    transition : np.ndarray
        Transition matrices ``(J, S, S)`` used in estimation.
    transition_params : dict
        First-stage transition estimates (increment probabilities for
        ``transition='increments'``).
    beta : float
        Discount factor: fixed, or the profile-likelihood maximiser when
        ``beta`` was a list or ``'estimate'`` (see ``beta_profile``).
    n_obs, n_units, n_states : int
    converged : bool
    iterations : int
        Outer iterations (BFGS + Newton for NFXP, Newton for CCP, NPL
        iterations for NPL).
    vce : str
    diagnostics : dict

    Examples
    --------
    >>> import statspai as sp
    >>> df = sp.dgp_ddc(n_units=100, n_periods=60, seed=0)
    >>> fit = sp.ddc(df, "replace", "mileage", id="bus", time="t",
    ...              beta=0.95, n_states=90)
    >>> type(fit).__name__
    'DDCResult'
    """

    _citation_keys = ("rust1987optimal",)

    method: str
    params: pd.Series
    se: pd.Series
    vcov: pd.DataFrame
    loglik: float
    ccp: pd.DataFrame
    ccp_first_stage: Optional[pd.DataFrame]
    value: pd.Series
    choice_values: pd.DataFrame
    transition: np.ndarray
    transition_params: Dict[str, Any]
    beta: float
    n_obs: int
    n_units: Optional[int]
    n_states: int
    converged: bool
    iterations: int
    vce: str
    alpha: float
    estimand: str
    diagnostics: Dict[str, Any]

    def __init__(self, **kw: Any) -> None:
        self._phi: np.ndarray = kw.pop("_phi")
        self.__dict__.update(kw)

    # ------------------------------------------------------------------
    @property
    def tstat(self) -> pd.Series:
        return self.params / self.se

    @property
    def pvalue(self) -> pd.Series:
        return pd.Series(2 * stats.norm.sf(np.abs(self.tstat)), index=self.params.index)

    def conf_int(self, alpha: Optional[float] = None) -> pd.DataFrame:
        a = self.alpha if alpha is None else alpha
        z = stats.norm.ppf(1 - a / 2)
        return pd.DataFrame(
            {"lower": self.params - z * self.se, "upper": self.params + z * self.se}
        )

    @property
    def ci(self) -> pd.DataFrame:
        return self.conf_int()

    @property
    def beta_profile(self) -> Optional[pd.DataFrame]:
        """Log-likelihood and estimates over the ``beta`` values fitted.

        Present when ``beta`` was a list or ``'estimate'``; ``None`` otherwise.
        """
        prof = self.diagnostics.get("beta_profile")
        if prof is None:
            return None
        out = pd.DataFrame({"loglik": prof["loglik"], "converged": prof["converged"]})
        for name, vals in prof["params"].items():
            out[name] = vals
        out.index = pd.Index(prof["beta"], name="beta")
        return out

    def tidy(self, conf_level: float | None = None) -> pd.DataFrame:
        """Broom-style coefficient table."""
        ci = self.conf_int(None if conf_level is None else 1 - conf_level)
        return pd.DataFrame(
            {
                "term": self.params.index,
                "estimate": self.params.to_numpy(),
                "std_error": self.se.to_numpy(),
                "statistic": self.tstat.to_numpy(),
                "p_value": self.pvalue.to_numpy(),
                "conf_low": ci["lower"].to_numpy(),
                "conf_high": ci["upper"].to_numpy(),
            }
        )

    def summary(self) -> str:
        """Formatted coefficient table."""
        lines = [
            f"Dynamic discrete choice: {_TITLES.get(self.method, self.method)}",
            f"  beta = {self.beta:g}   states = {self.n_states}   actions = "
            f"{list(self.ccp.columns)}",
            f"  N = {self.n_obs}   units = {self.n_units}   "
            f"log-likelihood = {self.loglik:.4f}",
            f"  vce = {self.vce} "
            + (
                "(full likelihood, transitions estimated jointly)"
                if self.diagnostics.get("stage") == "full"
                else "(conditional on the first-stage transitions)"
            ),
            "",
            f"{'':<12}{'coef':>12}{'std.err':>12}{'z':>9}{'P>|z|':>9}",
        ]
        for name in self.params.index:
            lines.append(
                f"{name:<12}{self.params[name]:>12.5f}{self.se[name]:>12.5f}"
                f"{self.tstat[name]:>9.2f}{self.pvalue[name]:>9.3f}"
            )
        if self.transition_params.get("increment_probs") is not None:
            p = ", ".join(f"{x:.4f}" for x in self.transition_params["increment_probs"])
            lines += ["", f"  state increment probabilities: [{p}]"]
        if not self.converged:
            lines += ["", "  WARNING: the estimator did not report convergence"]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return f"<DDCResult method={self.method!r} K={len(self.params)} N={self.n_obs}>"

    # ------------------------------------------------------------------
    def counterfactual(
        self,
        params: Optional[Dict[str, float]] = None,
        utility_shift: Optional[np.ndarray] = None,
        beta: Optional[float] = None,
        transition: Optional[np.ndarray] = None,
    ) -> pd.DataFrame:
        """Choice probabilities and stationary distribution under a change.

        Parameters
        ----------
        params : dict, optional
            Parameter values to overwrite, e.g. ``{"RC": 5.0}``.
        utility_shift : array (S, J), optional
            Added to the flow utility of every (state, action) pair, e.g. a
            subsidy on one action.
        beta : float, optional
            Alternative discount factor.
        transition : array (J, S, S), optional
            Alternative transition matrices.

        Returns
        -------
        pd.DataFrame
            One row per state: baseline and counterfactual choice
            probabilities (``P_<action>_base`` / ``P_<action>_cf``) and the
            stationary distribution of the state under each policy
            (``stationary_base`` / ``stationary_cf``). The long-run choice
            frequencies are in ``.attrs['long_run']``.
        """
        theta = self.params.copy()
        if params:
            unknown = set(params) - set(theta.index)
            if unknown:
                raise MethodIncompatibility(
                    f"unknown parameters {sorted(unknown)}",
                    recovery_hint=f"use names from {list(theta.index)}",
                )
            for k, v in params.items():
                theta[k] = float(v)
        b = self.beta if beta is None else float(beta)
        if not 0 <= b < 1:
            raise MethodIncompatibility(
                "beta must be in [0, 1)",
                recovery_hint="pass a discount factor below one",
            )
        F = (
            self.transition
            if transition is None
            else np.asarray(transition, dtype=float)
        )
        if F.shape != self.transition.shape:
            raise MethodIncompatibility(
                f"transition must have shape {self.transition.shape}",
                recovery_hint="stack one (S, S) matrix per action",
            )
        u = self._phi @ theta.to_numpy()
        if utility_shift is not None:
            shift = np.asarray(utility_shift, dtype=float)
            if shift.shape != u.shape:
                raise MethodIncompatibility(
                    f"utility_shift must have shape {u.shape}",
                    recovery_hint="one value per (state, action)",
                )
            u = u + shift
        _, _, P_cf, _ = solve_value(u, F, b)
        P0 = self.ccp.to_numpy()
        pi0 = stationary_distribution(controlled_transition(P0, self.transition))
        pi1 = stationary_distribution(controlled_transition(P_cf, F))
        cols: Dict[str, Any] = {}
        labels: List[Any] = list(self.ccp.columns)
        for j, lab in enumerate(labels):
            cols[f"P_{lab}_base"] = P0[:, j]
            cols[f"P_{lab}_cf"] = P_cf[:, j]
        cols["stationary_base"] = pi0
        cols["stationary_cf"] = pi1
        out = pd.DataFrame(cols, index=pd.Index(range(self.n_states), name="state"))
        out.attrs["long_run"] = {
            "base": dict(zip(labels, (pi0 @ P0).tolist())),
            "counterfactual": dict(zip(labels, (pi1 @ P_cf).tolist())),
        }
        return out
