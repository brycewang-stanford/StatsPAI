"""Result classes for the adverse-selection toolkit."""

from __future__ import annotations

from typing import Any, ClassVar, Dict, Optional, Tuple

import numpy as np
import pandas as pd

from .._result_serialize import ResultProtocolMixin


class PositiveCorrelationResult(ResultProtocolMixin):
    """Outcome of :func:`statspai.positive_correlation_test`.

    Attributes
    ----------
    method : str
        ``'biprobit'`` or ``'residuals'``.
    estimate : float
        The error correlation ``rho`` of the bivariate probit, or the
        correlation of the two probits' generalized residuals.
    statistic : float
        Chi-squared statistic with one degree of freedom: the likelihood
        ratio test of ``rho = 0`` (bivariate probit without clusters), the
        Wald test on ``athrho`` (with clusters), or the score statistic
        ``W`` built from the generalized residuals.
    z : float
        Signed square root of ``statistic`` (positive when coverage and
        claims are positively correlated).
    pvalue : float
        Two-sided p-value of ``statistic``.
    pvalue_greater : float
        One-sided p-value against positive correlation (the conditional
        correlation property predicted by asymmetric information).

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(0)
    >>> x = rng.normal(size=1000)
    >>> df = pd.DataFrame({"x": x,
    ...                    "cover": (x + rng.normal(size=1000) > 0).astype(int),
    ...                    "claim": (x + rng.normal(size=1000) > 0).astype(int)})
    >>> res = sp.positive_correlation_test(df, "cover", "claim", ["x"],
    ...                                    method="residuals")
    >>> type(res).__name__
    'PositiveCorrelationResult'
    """

    _citation_keys: ClassVar[Tuple[str, ...]] = ("chiappori2000testing",)

    def __init__(
        self,
        *,
        method: str,
        estimate: float,
        se: Optional[float],
        statistic: float,
        z: float,
        pvalue: float,
        pvalue_greater: float,
        test: str,
        alpha: float,
        n_obs: int,
        detail: Dict[str, Any],
        diagnostics: Dict[str, Any],
    ) -> None:
        self.method = method
        self.estimand = "conditional correlation of coverage and claims"
        self.estimate = estimate
        self.se = se
        self.statistic = statistic
        self.z = z
        self.pvalue = pvalue
        self.pvalue_greater = pvalue_greater
        self.test = test
        self.alpha = alpha
        self.n_obs = n_obs
        self.detail = detail
        self.diagnostics = diagnostics

    def summary(self) -> str:
        label = {
            "biprobit": "Bivariate probit",
            "residuals": "Generalized residuals of two probits",
        }[self.method]
        lines = [
            "Positive correlation test (Chiappori-Salanie)",
            f"Method: {label}",
            f"Number of obs = {self.n_obs}",
            f"Correlation estimate = {self.estimate:.6f}",
        ]
        if self.method == "biprobit":
            lines.append(
                f"athrho = {self.detail['athrho']:.6f} "
                f"(se {self.detail['athrho_se']:.6f})"
            )
        lines += [
            f"{self.test}: chi2(1) = {self.statistic:.4f}, "
            f"Prob > chi2 = {self.pvalue:.4f}",
            f"One-sided (positive correlation): z = {self.z:.4f}, "
            f"p = {self.pvalue_greater:.4f}",
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()


class EFCWelfareResult(ResultProtocolMixin):
    """Outcome of :func:`statspai.efc_welfare`.

    Attributes
    ----------
    estimate : float
        Deadweight loss per potential buyer from adverse (or advantageous)
        selection, in the units of ``price`` and ``cost``.
    table : pandas.DataFrame
        One row per quantity (curve coefficients, equilibrium and
        efficient price and coverage, deadweight loss) with estimate,
        standard error and confidence interval.
    curves : dict
        Intercepts and slopes of demand ``D(p)``, average cost ``AC(p)``
        and marginal cost ``MC(p)``, all linear in price.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(1)
    >>> p = rng.choice([2.0, 3.0, 4.0, 5.0], size=2000)
    >>> risk = rng.uniform(0, 10, size=2000)
    >>> buy = (risk > p + rng.normal(0, 1, 2000)).astype(int)
    >>> cost = np.where(buy == 1, 0.6 * risk + rng.normal(0, 1, 2000), np.nan)
    >>> res = sp.efc_welfare(pd.DataFrame({"p": p, "buy": buy, "cost": cost}),
    ...                      "p", "buy", "cost")
    >>> type(res).__name__
    'EFCWelfareResult'
    """

    _citation_keys: ClassVar[Tuple[str, ...]] = ("einav2010estimating",)

    def __init__(
        self,
        *,
        table: pd.DataFrame,
        curves: Dict[str, float],
        se_method: str,
        alpha: float,
        n_obs: int,
        n_covered: int,
        price_range: Tuple[float, float],
        diagnostics: Dict[str, Any],
    ) -> None:
        self.method = "Einav-Finkelstein-Cullen welfare (linear curves)"
        self.estimand = "deadweight loss per potential buyer"
        self.table = table
        self.curves = curves
        self.se_method = se_method
        self.alpha = alpha
        self.n_obs = n_obs
        self.n_covered = n_covered
        self.price_range = price_range
        self.diagnostics = diagnostics
        row = table.loc["dwl"]
        self.estimate = float(row["estimate"])
        self.se = float(row["se"])
        self.ci = (float(row["ci_lower"]), float(row["ci_upper"]))

    def summary(self) -> str:
        c = self.curves
        lines = [
            self.method,
            f"Number of obs = {self.n_obs} (covered: {self.n_covered})",
            f"Demand        D(p)  = {c['alpha']:.6g} + ({c['beta']:.6g}) p",
            f"Average cost  AC(p) = {c['gamma']:.6g} + ({c['delta']:.6g}) p",
            "Marginal cost MC(p) = AC(p) + delta * D(p) / beta",
            f"Standard errors: {self.se_method}",
            "",
            self.table.to_string(float_format=lambda v: f"{v:.6g}"),
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()

    def plot(self, ax: Any = None, n_points: int = 200) -> Any:
        """Demand, average cost and marginal cost against coverage.

        The deadweight-loss region between demand and marginal cost, from
        the equilibrium to the efficient coverage, is shaded.
        """
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=(6.4, 4.4))
        c = self.curves
        q = np.linspace(0.0, 1.0, n_points)
        inv = (q - c["alpha"]) / c["beta"]
        ac = c["gamma"] + c["delta"] * inv
        mc = ac + c["delta"] * q / c["beta"]
        ax.plot(q, inv, label="Demand")
        ax.plot(q, ac, label="Average cost")
        ax.plot(q, mc, label="Marginal cost")
        q_eq = float(self.table.loc["coverage_eq", "estimate"])
        q_eff = float(self.table.loc["coverage_eff", "estimate"])
        lo, hi = sorted((q_eq, q_eff))
        band = (q >= lo) & (q <= hi)
        if band.any():
            ax.fill_between(q[band], inv[band], mc[band], alpha=0.25, label="DWL")
        ax.axvline(q_eq, ls=":", lw=1)
        ax.axvline(q_eff, ls="--", lw=1)
        ax.set_xlabel("Share covered")
        ax.set_ylabel("Price / cost")
        ax.legend(frameon=False)
        return ax


class PooledPriceRatioResult(ResultProtocolMixin):
    """Outcome of :func:`statspai.pooled_price_ratio`.

    Attributes
    ----------
    curve : pandas.DataFrame
        One row per evaluation point ``r`` with ``quantile`` (mass strictly
        below ``r``), ``r``, ``mean_above`` (``E[P | P >= r]``), the pooled
        price ratio ``T`` and the implicit tax ``t = T - 1``.
    estimate : float
        Minimum implicit tax over points up to the ``upto_quantile``
        quantile (``t80`` with the default).
    t_min_all : float
        Infimum of the implicit tax over the whole support below one.
    no_trade : bool or None
        With ``markup`` given, whether ``t(r) >= markup`` at every point of
        the support below one (Hendren's no-trade condition).

    Examples
    --------
    >>> import numpy as np
    >>> import statspai as sp
    >>> risk = np.random.default_rng(0).beta(2, 30, size=2000)
    >>> res = sp.pooled_price_ratio(risk)
    >>> type(res).__name__
    'PooledPriceRatioResult'
    """

    _citation_keys: ClassVar[Tuple[str, ...]] = ("hendren2013private",)

    def __init__(
        self,
        *,
        curve: pd.DataFrame,
        estimate: float,
        argmin_r: float,
        t_min_all: float,
        upto_quantile: float,
        markup: Optional[float],
        no_trade: Optional[bool],
        no_trade_upto: Optional[bool],
        se: Optional[float],
        ci: Optional[Tuple[float, float]],
        alpha: float,
        n_obs: int,
        diagnostics: Dict[str, Any],
    ) -> None:
        self.method = "Hendren pooled price ratio"
        self.estimand = f"minimum implicit tax up to quantile {upto_quantile:g}"
        self.curve = curve
        self.estimate = estimate
        self.argmin_r = argmin_r
        self.t_min_all = t_min_all
        self.upto_quantile = upto_quantile
        self.markup = markup
        self.no_trade = no_trade
        self.no_trade_upto = no_trade_upto
        self.se = se
        self.ci = ci
        self.alpha = alpha
        self.n_obs = n_obs
        self.diagnostics = diagnostics

    def summary(self) -> str:
        lines = [
            "Pooled price ratio and implicit tax (Hendren 2013)",
            f"Number of obs = {self.n_obs}",
            f"Minimum implicit tax up to quantile {self.upto_quantile:g} = "
            f"{self.estimate:.6f} (at r = {self.argmin_r:.6g})",
            f"Infimum over the support below 1 = {self.t_min_all:.6f}",
        ]
        if self.ci is not None:
            lines.append(
                f"Bootstrap {100 * (1 - self.alpha):g}% interval = "
                f"[{self.ci[0]:.6f}, {self.ci[1]:.6f}]"
            )
        if self.markup is not None:
            lines.append(
                f"No-trade condition at markup {self.markup:g}: "
                f"whole support {self.no_trade}, up to quantile "
                f"{self.no_trade_upto}"
            )
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()

    def plot(self, ax: Any = None) -> Any:
        """Implicit tax ``t(r)`` against the risk quantile."""
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=(6.4, 4.4))
        ax.plot(self.curve["quantile"], self.curve["t"], label="Implicit tax t(r)")
        ax.axvline(self.upto_quantile, ls=":", lw=1)
        if self.markup is not None:
            ax.axhline(self.markup, ls="--", lw=1, label="Markup")
        ax.set_xlabel("Risk quantile")
        ax.set_ylabel("Implicit tax")
        ax.legend(frameon=False)
        return ax


class RiskClassSelectionResult(ResultProtocolMixin):
    """Outcome of :func:`statspai.risk_class_selection`.

    Attributes
    ----------
    table : pandas.DataFrame
        One row per scenario (``'1'`` covariates only, ``'2'`` with the
        current predictor, then one per future-predictor scenario) with
        ``t80`` (minimum implicit tax up to ``upto_quantile`` inside the
        risk class), bootstrap ``se``, ``ci_lower`` and ``ci_upper``, and
        the mean and SD of risk in the class.
    curves : dict of pandas.DataFrame
        Implicit tax curve of each scenario (columns as in
        :func:`statspai.pooled_price_ratio`).
    params : dict
        Probit coefficients of Scenarios 1 and 2 and, for Scenario 3,
        ``theta``, ``sigma_t2``, ``r2_current`` and per scenario
        ``sigma_eps2``, ``sigma_v2``, ``beta_g``, ``beta_w``.
    risk : pandas.DataFrame
        Scenario 1 risk of every person and whether she is in the class.
    t_no_underwriting : float
        Scenario 1 ``t80`` when everyone pays the same premium.

    Examples
    --------
    >>> import numpy as np, pandas as pd
    >>> import statspai as sp
    >>> rng = np.random.default_rng(1)
    >>> w = rng.normal(size=3000)
    >>> loss = (-1.4 + 0.7 * w + rng.normal(size=3000) > 0).astype(int)
    >>> res = sp.risk_class_selection(pd.DataFrame({"loss": loss, "w": w}),
    ...                               "loss", ["w"], n_boot=0)
    >>> type(res).__name__
    'RiskClassSelectionResult'
    """

    _citation_keys: ClassVar[Tuple[str, ...]] = (
        "azevedo2026genetic",
        "hendren2013private",
    )

    def __init__(
        self,
        *,
        table: pd.DataFrame,
        curves: Dict[str, pd.DataFrame],
        params: Dict[str, Any],
        risk: pd.DataFrame,
        risk_class: Optional[Tuple[float, float]],
        mean_risk: float,
        t_no_underwriting: float,
        upto_quantile: float,
        n_boot: int,
        alpha: float,
        n_obs: int,
        boot_draws: pd.DataFrame,
        diagnostics: Dict[str, Any],
    ) -> None:
        self.method = "Implicit tax within a risk class (Azevedo et al. 2026)"
        self.estimand = (
            f"minimum implicit tax up to quantile {upto_quantile:g} in the "
            "risk class, by information scenario"
        )
        self.table = table
        self.curves = curves
        self.params = params
        self.risk = risk
        self.risk_class = risk_class
        self.mean_risk = mean_risk
        self.t_no_underwriting = t_no_underwriting
        self.upto_quantile = upto_quantile
        self.n_boot = n_boot
        self.alpha = alpha
        self.n_obs = n_obs
        self.boot_draws = boot_draws
        self.diagnostics = diagnostics
        last = table.index[-1]
        self.estimate = float(table.loc[last, "t80"])
        self.se = float(table.loc[last, "se"])

    def summary(self) -> str:
        if self.risk_class is None:
            cls = "everyone (no underwriting)"
        else:
            cls = (
                f"risk in [{self.risk_class[0]:g}, {self.risk_class[1]:g}] x "
                f"sample mean {self.mean_risk:.6g}"
            )
        lines = [
            "Implicit tax within a risk class (Azevedo et al. 2026)",
            f"Number of obs = {self.n_obs}; class: {cls}",
            f"Scenario 1 t at quantile {self.upto_quantile:g} without "
            f"underwriting = {self.t_no_underwriting:.6f}",
            f"Bootstrap replications = {self.n_boot}",
            self.table.to_string(float_format=lambda v: f"{v:.6f}"),
        ]
        return "\n".join(lines)

    def __repr__(self) -> str:
        return self.summary()

    def plot(self, ax: Any = None) -> Any:
        """Implicit tax curves of all scenarios against the risk quantile."""
        import matplotlib.pyplot as plt

        if ax is None:
            _, ax = plt.subplots(figsize=(6.4, 4.4))
        for name, curve in self.curves.items():
            inner = curve[(curve["r"] > 0) & (curve["r"] < 1)]
            ax.plot(inner["quantile"], inner["t"], label=f"Scenario {name}")
        ax.axvline(self.upto_quantile, ls=":", lw=1)
        ax.set_xlabel("Risk quantile within the class")
        ax.set_ylabel("Implicit tax")
        ax.legend(frameon=False)
        return ax
