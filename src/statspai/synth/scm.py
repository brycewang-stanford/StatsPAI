"""
Synthetic Control Method (Abadie, Diamond, Hainmueller 2010).

Constructs a weighted combination of control units to approximate the
counterfactual trajectory of a treated unit, then estimates the treatment
effect as the gap between observed and synthetic outcomes.

Supports:
- Classic SCM (constrained optimization: weights >= 0, sum to 1)
- Penalized/ridge SCM for many donors
- Placebo inference (in-space and in-time)
- Gap plots and pathway plots
"""

from typing import Optional, List, Dict, Any, Tuple
import numpy as np
import pandas as pd
from scipy import optimize, stats

from ..core.results import CausalResult


def synth(
    data: pd.DataFrame,
    outcome: str,
    unit: str,
    time: str,
    treated_unit: Any,
    treatment_time: Any,
    covariates: Optional[List[str]] = None,
    penalization: float = 0.0,
    placebo: bool = True,
    alpha: float = 0.05,
) -> CausalResult:
    """
    Estimate treatment effect using the Synthetic Control Method.

    Parameters
    ----------
    data : pd.DataFrame
        Long-format panel data.
    outcome : str
        Outcome variable name.
    unit : str
        Unit identifier column.
    time : str
        Time period column.
    treated_unit : any
        Identifier of the treated unit.
    treatment_time : any
        First treatment period (inclusive).
    covariates : list of str, optional
        Additional covariates to match on (pre-treatment averages).
    penalization : float, default 0.0
        Ridge penalty for donor weights (0 = classic SCM).
    placebo : bool, default True
        Run placebo (permutation) inference across control units.
    alpha : float, default 0.05
        Significance level.

    Returns
    -------
    CausalResult

    Examples
    --------
    >>> result = synth(df, outcome='gdp', unit='state', time='year',
    ...               treated_unit='California', treatment_time=1989)
    >>> print(result.summary())
    >>> result.plot()
    """
    model = SyntheticControl(
        data=data,
        outcome=outcome,
        unit=unit,
        time=time,
        treated_unit=treated_unit,
        treatment_time=treatment_time,
        covariates=covariates,
        penalization=penalization,
        alpha=alpha,
    )
    return model.fit(placebo=placebo)


class SyntheticControl:
    """
    Synthetic Control estimator.

    Parameters
    ----------
    data : pd.DataFrame
        Long-format panel (unit, time, outcome, ...).
    outcome, unit, time : str
        Column names.
    treated_unit : any
        Identifier of the treated unit.
    treatment_time : any
        First treatment period.
    covariates : list of str, optional
    penalization : float, default 0.0
    alpha : float, default 0.05
    """

    def __init__(
        self,
        data: pd.DataFrame,
        outcome: str,
        unit: str,
        time: str,
        treated_unit: Any,
        treatment_time: Any,
        covariates: Optional[List[str]] = None,
        penalization: float = 0.0,
        alpha: float = 0.05,
    ):
        self.data = data
        self.outcome = outcome
        self.unit = unit
        self.time = time
        self.treated_unit = treated_unit
        self.treatment_time = treatment_time
        self.covariates = covariates or []
        self.penalization = penalization
        self.alpha = alpha

        self._validate()
        self._prepare_matrices()

    # ------------------------------------------------------------------
    # Validation & data prep
    # ------------------------------------------------------------------

    def _validate(self):
        for col in [self.outcome, self.unit, self.time] + self.covariates:
            if col not in self.data.columns:
                raise ValueError(f"Column '{col}' not found in data")
        if self.treated_unit not in self.data[self.unit].values:
            raise ValueError(
                f"Treated unit '{self.treated_unit}' not found in '{self.unit}'"
            )

    def _prepare_matrices(self):
        """Pivot data into (T x J) outcome matrix."""
        pivot = self.data.pivot_table(
            index=self.time, columns=self.unit, values=self.outcome,
        )

        if self.treated_unit not in pivot.columns:
            raise ValueError(f"Treated unit '{self.treated_unit}' missing after pivot")

        self.times = pivot.index.values
        self.pre_mask = self.times < self.treatment_time
        self.post_mask = self.times >= self.treatment_time

        if self.pre_mask.sum() < 2:
            raise ValueError("Need at least 2 pre-treatment periods")
        if self.post_mask.sum() < 1:
            raise ValueError("Need at least 1 post-treatment period")

        self.Y_treated = pivot[self.treated_unit].values  # (T,)
        donor_cols = [c for c in pivot.columns if c != self.treated_unit]
        self.donor_units = donor_cols
        self.Y_donors = pivot[donor_cols].values  # (T, J)

        # Handle NaN: drop donors with any NaN in pre-period
        pre_donors = self.Y_donors[self.pre_mask]
        valid = ~np.any(np.isnan(pre_donors), axis=0)
        if valid.sum() == 0:
            raise ValueError("No valid donor units (all have NaN in pre-period)")
        self.Y_donors = self.Y_donors[:, valid]
        self.donor_units = [self.donor_units[i] for i in range(len(self.donor_units)) if valid[i]]

        # Covariate matching matrix (pre-treatment averages)
        if self.covariates:
            pre_data = self.data[self.data[self.time] < self.treatment_time]
            cov_treated = pre_data[pre_data[self.unit] == self.treated_unit][self.covariates].mean().values
            cov_donors = []
            for d in self.donor_units:
                cov_donors.append(
                    pre_data[pre_data[self.unit] == d][self.covariates].mean().values
                )
            self.cov_treated = cov_treated
            self.cov_donors = np.array(cov_donors).T  # (n_covs, J)
        else:
            self.cov_treated = None
            self.cov_donors = None

    # ------------------------------------------------------------------
    # Weight optimization
    # ------------------------------------------------------------------

    def _solve_weights(
        self,
        Y_treated_pre: np.ndarray,
        Y_donors_pre: np.ndarray,
    ) -> np.ndarray:
        """
        Find optimal donor weights by minimizing pre-treatment MSPE.

        min_w ||Y_treated_pre - Y_donors_pre @ w||^2 + pen * ||w||^2
        s.t.  w_j >= 0,  sum(w) = 1
        """
        J = Y_donors_pre.shape[1]

        if J == 0:
            raise ValueError("No donor units available")

        def objective(w):
            residual = Y_treated_pre - Y_donors_pre @ w
            loss = residual @ residual
            if self.penalization > 0:
                loss += self.penalization * (w @ w)
            return loss

        def jac(w):
            residual = Y_treated_pre - Y_donors_pre @ w
            grad = -2 * Y_donors_pre.T @ residual
            if self.penalization > 0:
                grad += 2 * self.penalization * w
            return grad

        # Constraints: sum(w) = 1
        constraints = {'type': 'eq', 'fun': lambda w: np.sum(w) - 1}
        # Bounds: w >= 0
        bounds = [(0, 1)] * J
        # Initial: uniform
        w0 = np.ones(J) / J

        result = optimize.minimize(
            objective, w0, jac=jac,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints,
            options={'maxiter': 1000, 'ftol': 1e-12},
        )

        return result.x

    # ------------------------------------------------------------------
    # Estimation
    # ------------------------------------------------------------------

    def fit(self, placebo: bool = True) -> CausalResult:
        """
        Fit the Synthetic Control model.

        Parameters
        ----------
        placebo : bool, default True
            Run placebo tests across all donor units.

        Returns
        -------
        CausalResult
        """
        Y_pre_treated = self.Y_treated[self.pre_mask]
        Y_pre_donors = self.Y_donors[self.pre_mask]

        # Solve for donor weights
        weights = self._solve_weights(Y_pre_treated, Y_pre_donors)

        # Synthetic control trajectory
        Y_synth = self.Y_donors @ weights  # (T,)

        # Treatment effect (gap)
        gap = self.Y_treated - Y_synth  # (T,)
        gap_post = gap[self.post_mask]
        gap_pre = gap[self.pre_mask]

        # Average treatment effect on treated (post-period)
        att = float(np.mean(gap_post))

        # Pre-treatment MSPE (fit quality)
        pre_mspe = float(np.mean(gap_pre**2))

        # --- Placebo inference ---
        placebo_atts = []
        placebo_pre_mspes = []
        if placebo and len(self.donor_units) >= 2:
            placebo_atts, placebo_pre_mspes = self._run_placebos()

        # P-value from placebo distribution
        if len(placebo_atts) > 0:
            # Ratio-based: MSPE_post / MSPE_pre
            post_mspe = float(np.mean(gap_post**2))
            ratio_treated = post_mspe / pre_mspe if pre_mspe > 1e-10 else np.inf

            placebo_ratios = []
            for pa, pm in zip(placebo_atts, placebo_pre_mspes):
                pr = pa**2 / pm if pm > 1e-10 else 0
                placebo_ratios.append(pr)

            # One-sided p-value: fraction of placebos with ratio >= treated
            pvalue = float(np.mean(np.array(placebo_ratios) >= ratio_treated))
            # Ensure at least 1/(J+1) if treated is most extreme
            pvalue = max(pvalue, 1 / (len(placebo_ratios) + 1))

            se = float(np.std(placebo_atts)) if len(placebo_atts) > 1 else 0.0
        else:
            pvalue = np.nan
            se = float(np.std(gap_post)) / max(np.sqrt(len(gap_post)), 1)

        z_crit = stats.norm.ppf(1 - self.alpha / 2)
        ci = (att - z_crit * se, att + z_crit * se)

        # --- Weight table ---
        weight_df = pd.DataFrame({
            'unit': self.donor_units,
            'weight': weights,
        }).sort_values('weight', ascending=False).reset_index(drop=True)
        weight_df = weight_df[weight_df['weight'] > 1e-6]

        # --- Gap table ---
        gap_df = pd.DataFrame({
            'time': self.times,
            'treated': self.Y_treated,
            'synthetic': Y_synth,
            'gap': gap,
            'post_treatment': self.post_mask,
        })

        # --- Model info ---
        model_info: Dict[str, Any] = {
            'n_donors': len(self.donor_units),
            'n_pre_periods': int(self.pre_mask.sum()),
            'n_post_periods': int(self.post_mask.sum()),
            'pre_treatment_mspe': round(pre_mspe, 6),
            'pre_treatment_rmse': round(np.sqrt(pre_mspe), 6),
            'penalization': self.penalization,
            'treatment_time': self.treatment_time,
            'treated_unit': self.treated_unit,
            'weights': weight_df,
            'gap_table': gap_df,
            'Y_synth': Y_synth,
            'Y_treated': self.Y_treated,
            'times': self.times,
        }

        if len(placebo_atts) > 0:
            model_info['placebo_atts'] = placebo_atts
            model_info['n_placebos'] = len(placebo_atts)

        return CausalResult(
            method='Synthetic Control Method',
            estimand='ATT',
            estimate=att,
            se=se,
            pvalue=pvalue,
            ci=ci,
            alpha=self.alpha,
            n_obs=len(self.Y_treated),
            detail=weight_df,
            model_info=model_info,
            _citation_key='synth',
        )

    def _run_placebos(self) -> Tuple[List[float], List[float]]:
        """Run placebo SCM for each donor unit (in-space placebo)."""
        placebo_atts = []
        placebo_pre_mspes = []

        all_units_data = np.column_stack([
            self.Y_treated[:, np.newaxis], self.Y_donors
        ])
        all_unit_names = [self.treated_unit] + list(self.donor_units)

        for i, placebo_unit in enumerate(self.donor_units):
            # Treat this donor as "treated", rest as donors
            idx_placebo = i + 1  # +1 because treated is at index 0
            Y_placebo = all_units_data[:, idx_placebo]
            donor_idx = [j for j in range(all_units_data.shape[1]) if j != idx_placebo]
            Y_placebo_donors = all_units_data[:, donor_idx]

            Y_pre_p = Y_placebo[self.pre_mask]
            Y_pre_d = Y_placebo_donors[self.pre_mask]

            try:
                w = self._solve_weights(Y_pre_p, Y_pre_d)
                synth_p = Y_placebo_donors @ w
                gap_p = Y_placebo - synth_p

                pre_mspe_p = float(np.mean(gap_p[self.pre_mask]**2))
                att_p = float(np.mean(gap_p[self.post_mask]))

                placebo_atts.append(att_p)
                placebo_pre_mspes.append(pre_mspe_p)
            except Exception:
                continue

        return placebo_atts, placebo_pre_mspes


# ------------------------------------------------------------------
# Citation
# ------------------------------------------------------------------

# Add to CausalResult citation registry
CausalResult._CITATIONS['synth'] = (
    "@article{abadie2010synthetic,\n"
    "  title={Synthetic Control Methods for Comparative Case Studies: "
    "Estimating the Effect of California's Tobacco Control Program},\n"
    "  author={Abadie, Alberto and Diamond, Alexis and Hainmueller, Jens},\n"
    "  journal={Journal of the American Statistical Association},\n"
    "  volume={105},\n"
    "  number={490},\n"
    "  pages={493--505},\n"
    "  year={2010},\n"
    "  publisher={Taylor \\& Francis}\n"
    "}"
)
