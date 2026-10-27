"""Result classes for :func:`sp.choo_siow`."""

from __future__ import annotations

from typing import Any, ClassVar, Dict, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats

from .._result_serialize import ResultProtocolMixin
from ..exceptions import MethodIncompatibility
from ._choo_siow_core import ipfp_equilibrium


class MatchingEquilibrium(ResultProtocolMixin):
    """Equilibrium of a Choo-Siow marriage market (IPFP solution).

    Attributes
    ----------
    mu : pd.DataFrame
        Couples by man type (rows) and woman type (columns).
    singles_men, singles_women : pd.Series
        Unmatched men by type, unmatched women by type.
    men_margins, women_margins : pd.Series
        Population margins the equilibrium was solved for.
    phi : pd.DataFrame
        Joint surplus used.
    n_iter : int
        IPFP iterations.
    max_margin_error : float
        Largest relative violation of ``n_x = mu_x0 + sum_y mu_xy`` /
        ``m_y = mu_0y + sum_x mu_xy``.

    Examples
    --------
    >>> import statspai as sp
    >>> import numpy as np
    >>> res = sp.choo_siow(mu=[[40, 10], [10, 40]], singles_men=[20, 20],
    ...                    singles_women=[25, 15])
    >>> eq = res.counterfactual(new_margins=([60, 80], [70, 70]))
    >>> bool(eq.max_margin_error < 1e-10)
    True
    """

    def __init__(
        self,
        mu: pd.DataFrame,
        singles_men: pd.Series,
        singles_women: pd.Series,
        men_margins: pd.Series,
        women_margins: pd.Series,
        phi: pd.DataFrame,
        n_iter: int,
        max_margin_error: float,
    ) -> None:
        self.mu = mu
        self.singles_men = singles_men
        self.singles_women = singles_women
        self.men_margins = men_margins
        self.women_margins = women_margins
        self.phi = phi
        self.n_iter = int(n_iter)
        self.max_margin_error = float(max_margin_error)
        self.n_couples = float(mu.to_numpy().sum())
        self.marriage_rate_men = self.n_couples / float(men_margins.sum())
        self.marriage_rate_women = self.n_couples / float(women_margins.sum())

    def to_frame(self) -> pd.DataFrame:
        """Long table: one row per household type (couple or single)."""
        rows = [
            {"man_type": x, "woman_type": y, "count": float(self.mu.loc[x, y])}
            for x in self.mu.index
            for y in self.mu.columns
        ]
        rows += [
            {"man_type": x, "woman_type": None, "count": float(v)}
            for x, v in self.singles_men.items()
        ]
        rows += [
            {"man_type": None, "woman_type": y, "count": float(v)}
            for y, v in self.singles_women.items()
        ]
        return pd.DataFrame(rows)

    def summary(self) -> str:
        """Formatted equilibrium table."""
        lines = [
            "Choo-Siow equilibrium (IPFP)",
            "=" * 60,
            f"Couples: {self.n_couples:.4f}   marriage rate men "
            f"{self.marriage_rate_men:.4f}, women {self.marriage_rate_women:.4f}",
            f"IPFP iterations: {self.n_iter}   max margin error: "
            f"{self.max_margin_error:.2e}",
            "",
            "Couples mu_xy (rows: men, columns: women)",
            self.mu.round(4).to_string(),
            "",
            "Single men: "
            + ", ".join(f"{k}={v:.4f}" for k, v in self.singles_men.items()),
            "Single women: "
            + ", ".join(f"{k}={v:.4f}" for k, v in self.singles_women.items()),
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"MatchingEquilibrium(couples={self.n_couples:.4g}, "
            f"types={self.mu.shape}, max_margin_error={self.max_margin_error:.1e})"
        )


def _align(values: Any, index: pd.Index, name: str) -> np.ndarray:
    if isinstance(values, pd.Series):
        missing = [i for i in index if i not in values.index]
        if missing:
            raise MethodIncompatibility(
                f"choo_siow: {name} has no entry for types {missing}.",
                recovery_hint=f"Index {name} by the same types as the fit.",
            )
        arr: np.ndarray = np.asarray(values.reindex(index), dtype=float)
    else:
        arr = np.asarray(values, dtype=float).ravel()
        if arr.size != len(index):
            raise MethodIncompatibility(
                f"choo_siow: {name} has {arr.size} entries for {len(index)} types.",
                recovery_hint=f"Give one value per type: {list(index)}.",
            )
    if not np.all(np.isfinite(arr)) or np.any(arr <= 0):
        raise MethodIncompatibility(
            f"choo_siow: {name} must be finite and strictly positive.",
            recovery_hint="Every type needs a positive population.",
        )
    return arr


class ChooSiowResult(ResultProtocolMixin):
    """Estimated Choo-Siow transferable-utility matching model.

    Attributes
    ----------
    method : str
        ``'nonparametric'`` (``Phi`` cell by cell) or ``'parametric'``
        (``Phi = sum_k beta_k phi_k``, weighted Poisson GLM).
    phi : pd.DataFrame
        Joint surplus ``Phi_xy`` of the fitted model (men by women). Choo and
        Siow's "total gains" are ``Phi / 2``.
    phi_se : pd.DataFrame
        Delta-method standard errors of ``phi`` under multinomial sampling
        of households.
    phi_nonparametric, phi_nonparametric_se : pd.DataFrame
        The saturated estimate ``log(mu_xy^2 / (mu_x0 mu_0y))`` and its
        standard errors, reported for both methods.
    params, std_errors : pd.Series
        ``beta`` (parametric) or the cells of ``phi`` labelled ``"x|y"``
        (nonparametric).
    vcov_params : pd.DataFrame
        Covariance of ``params``.
    mu : pd.DataFrame
        Observed couples; ``singles_men`` / ``singles_women`` observed
        singles; ``men_margins`` / ``women_margins`` their sums.
    fitted_mu, fitted_singles_men, fitted_singles_women
        Equilibrium of the fitted ``phi`` at the observed margins (equal to
        the data for the nonparametric fit).
    n_households : float
    model_info, diagnostics : dict

    Examples
    --------
    >>> import statspai as sp
    >>> res = sp.choo_siow(mu=[[40, 10], [10, 40]], singles_men=[20, 20],
    ...                    singles_women=[25, 15])
    >>> round(float(res.phi.iloc[0, 0]), 4)  # log(40^2 / (20 * 25))
    1.1632
    """

    _citation_keys: ClassVar[Tuple[str, ...]] = ("choo2006who", "galichon2022cupid")

    def __init__(
        self,
        *,
        method: str,
        phi: pd.DataFrame,
        phi_se: pd.DataFrame,
        phi_nonparametric: pd.DataFrame,
        phi_nonparametric_se: pd.DataFrame,
        params: pd.Series,
        std_errors: pd.Series,
        vcov_params: pd.DataFrame,
        mu: pd.DataFrame,
        singles_men: pd.Series,
        singles_women: pd.Series,
        fitted_mu: pd.DataFrame,
        fitted_singles_men: pd.Series,
        fitted_singles_women: pd.Series,
        alpha: float,
        model_info: Dict[str, Any],
        diagnostics: Dict[str, Any],
        basis: Optional[np.ndarray] = None,
    ) -> None:
        self.method = method
        self.phi = phi
        self.phi_se = phi_se
        self.phi_nonparametric = phi_nonparametric
        self.phi_nonparametric_se = phi_nonparametric_se
        self.params = params
        self.std_errors = std_errors
        self.vcov_params = vcov_params
        self.mu = mu
        self.singles_men = singles_men
        self.singles_women = singles_women
        self.men_margins = singles_men + mu.sum(axis=1)
        self.women_margins = singles_women + mu.sum(axis=0)
        self.fitted_mu = fitted_mu
        self.fitted_singles_men = fitted_singles_men
        self.fitted_singles_women = fitted_singles_women
        self.n_couples = float(mu.to_numpy().sum())
        self.n_households = float(
            self.n_couples + singles_men.sum() + singles_women.sum()
        )
        self.alpha = float(alpha)
        self.model_info = dict(model_info)
        self.diagnostics = dict(diagnostics)
        self._basis = basis
        if method == "parametric":
            # the GLM estimator has its own paper; an instance attribute
            # shadows the class default read by ``cite()``
            setattr(
                self,
                "_citation_keys",
                ("choo2006who", "galichon2022cupid", "galichon2024estimating"),
            )

    # ------------------------------------------------------------------
    def vcov(self) -> pd.DataFrame:
        """Covariance matrix of :attr:`params`."""
        return self.vcov_params

    def coef_table(self) -> pd.DataFrame:
        """Estimate, standard error, z, p-value and interval for ``params``."""
        z = self.params / self.std_errors
        crit = stats.norm.ppf(1 - self.alpha / 2)
        return pd.DataFrame(
            {
                "estimate": self.params,
                "std_error": self.std_errors,
                "z": z,
                "p_value": 2 * stats.norm.sf(np.abs(z)),
                "ci_lower": self.params - crit * self.std_errors,
                "ci_upper": self.params + crit * self.std_errors,
            }
        )

    def counterfactual(
        self,
        new_margins: Optional[Any] = None,
        *,
        phi: Optional[Any] = None,
        tol: float = 1e-12,
        maxiter: int = 100_000,
    ) -> MatchingEquilibrium:
        """Equilibrium for new population margins and/or a new surplus.

        Parameters
        ----------
        new_margins : tuple (men, women) or dict, optional
            Numbers of men by type ``n_x`` and women by type ``m_y``
            (married plus single), as arrays in type order or Series indexed
            by type; a dict uses keys ``'men'`` / ``'women'`` and may give
            only one. Default: the observed margins.
        phi : array-like or pd.DataFrame, optional
            Joint surplus to use instead of the fitted one (e.g. a
            "no sorting" surplus). Shape ``(X, Y)``.
        tol : float, default 1e-12
            Relative margin error at which IPFP stops.
        maxiter : int, default 100000

        Returns
        -------
        MatchingEquilibrium

        Examples
        --------
        >>> import statspai as sp
        >>> res = sp.choo_siow(mu=[[40, 10], [10, 40]], singles_men=[20, 20],
        ...                    singles_women=[25, 15])
        >>> same = res.counterfactual()  # observed margins: reproduces data
        >>> bool(abs(same.mu.iloc[0, 1] - 10) < 1e-8)
        True
        """
        men = self.men_margins
        women = self.women_margins
        if new_margins is not None:
            if isinstance(new_margins, dict):
                unknown = set(new_margins) - {"men", "women"}
                if unknown:
                    raise MethodIncompatibility(
                        f"choo_siow.counterfactual: unknown keys {sorted(unknown)}.",
                        recovery_hint="Use {'men': ..., 'women': ...}.",
                    )
                men_new = new_margins.get("men", men)
                women_new = new_margins.get("women", women)
            else:
                try:
                    men_new, women_new = new_margins
                except (TypeError, ValueError) as exc:
                    raise MethodIncompatibility(
                        "choo_siow.counterfactual: new_margins must be a "
                        "(men, women) pair or a dict.",
                        recovery_hint="new_margins=(n_x, m_y).",
                    ) from exc
            men = pd.Series(
                _align(men_new, self.mu.index, "men margins"), index=self.mu.index
            )
            women = pd.Series(
                _align(women_new, self.mu.columns, "women margins"),
                index=self.mu.columns,
            )
        if phi is None:
            phi_df = self.phi
        elif isinstance(phi, pd.DataFrame):
            # missing types become NaN and are rejected below
            phi_df = phi.reindex(index=self.mu.index, columns=self.mu.columns)
        else:
            arr = np.asarray(phi, dtype=float)
            if arr.shape != self.mu.shape:
                raise MethodIncompatibility(
                    f"choo_siow.counterfactual: phi has shape {arr.shape}, "
                    f"expected {self.mu.shape}.",
                    recovery_hint="Give one surplus per (man type, woman type).",
                )
            phi_df = pd.DataFrame(arr, index=self.mu.index, columns=self.mu.columns)
        P = phi_df.to_numpy(dtype=float)
        if np.isnan(P).any() or np.isposinf(P).any():
            raise MethodIncompatibility(
                "choo_siow.counterfactual: phi has NaN or +inf cells.",
                recovery_hint="-inf (a pair that never forms) is allowed; NaN is not.",
            )
        eq = ipfp_equilibrium(
            P, men.to_numpy(float), women.to_numpy(float), tol=tol, maxiter=maxiter
        )
        return MatchingEquilibrium(
            mu=pd.DataFrame(eq["mu"], index=self.mu.index, columns=self.mu.columns),
            singles_men=pd.Series(eq["singles_men"], index=self.mu.index),
            singles_women=pd.Series(eq["singles_women"], index=self.mu.columns),
            men_margins=men,
            women_margins=women,
            phi=phi_df,
            n_iter=int(eq["n_iter"]),
            max_margin_error=float(eq["max_margin_error"]),
        )

    # ------------------------------------------------------------------
    def summary(self) -> str:
        """Formatted summary of the fit."""
        width = 72
        lines = [
            "=" * width,
            f"Choo-Siow TU matching model ({self.method})".center(width),
            "=" * width,
            f"Types: {self.mu.shape[0]} men x {self.mu.shape[1]} women   "
            f"households: {self.n_households:.0f}   couples: {self.n_couples:.0f}",
        ]
        if self.method == "parametric":
            lines += [
                "Phi_xy = sum_k beta_k phi_k(x, y); weighted Poisson GLM, "
                "sandwich SEs",
                "-" * width,
                self.coef_table().round(4).to_string(),
                "-" * width,
                f"Newton iterations: {self.diagnostics.get('n_iter')}   "
                f"converged: {self.diagnostics.get('converged')}",
            ]
        lines += [
            "",
            "Joint surplus Phi_xy (rows: men, columns: women)",
            self.phi.round(4).to_string(),
            "",
            "Standard errors",
            self.phi_se.round(4).to_string(),
        ]
        zc = self.diagnostics.get("zero_couple_cells") or []
        if zc:
            lines.append(f"Empty couple cells (Phi = -inf nonparametrically): {zc}")
        lines.append("=" * width)
        return "\n".join(lines)

    def __repr__(self) -> str:
        return (
            f"ChooSiowResult(method={self.method!r}, types={self.mu.shape}, "
            f"households={self.n_households:.0f})"
        )
