"""
Unified results class for all econometric models
"""

from typing import Dict, Any, Optional, List, Union
import pandas as pd
import numpy as np
from scipy import stats


class EconometricResults:
    """
    Unified results class for econometric models
    
    This class provides a consistent interface for accessing results
    from different econometric estimators, similar to R's broom package.
    """
    
    def __init__(
        self,
        params: pd.Series,
        std_errors: pd.Series,
        model_info: Dict[str, Any],
        data_info: Optional[Dict[str, Any]] = None,
        diagnostics: Optional[Dict[str, Any]] = None
    ):
        """
        Initialize results object
        
        Parameters
        ----------
        params : pd.Series
            Parameter estimates with variable names as index
        std_errors : pd.Series
            Standard errors with variable names as index
        model_info : Dict[str, Any]
            Model metadata (model type, estimation method, etc.)
        data_info : Dict[str, Any], optional
            Data metadata (sample size, variable names, etc.)
        diagnostics : Dict[str, Any], optional
            Model diagnostics (R-squared, F-statistics, etc.)
        """
        self.params = params
        self.std_errors = std_errors
        self.model_info = model_info
        self.data_info = data_info or {}
        self.diagnostics = diagnostics or {}
        
        # Compute derived statistics
        self._compute_statistics()
    
    def _compute_statistics(self):
        """Compute t-statistics, p-values, and confidence intervals"""
        self.tvalues = self.params / self.std_errors
        self.pvalues = 2 * (1 - stats.t.cdf(np.abs(self.tvalues), 
                                           self.data_info.get('df_resid', np.inf)))
        
        # 95% confidence intervals by default
        alpha = 0.05
        t_crit = stats.t.ppf(1 - alpha/2, self.data_info.get('df_resid', np.inf))
        self.conf_int_lower = self.params - t_crit * self.std_errors
        self.conf_int_upper = self.params + t_crit * self.std_errors
    
    def summary(self, alpha: float = 0.05) -> str:
        """
        Generate a summary table of results
        
        Parameters
        ----------
        alpha : float, default 0.05
            Significance level for confidence intervals
            
        Returns
        -------
        str
            Formatted summary table
        """
        # Create coefficients table
        coef_table = pd.DataFrame({
            'Coefficient': self.params,
            'Std. Error': self.std_errors,
            't-statistic': self.tvalues,
            'P>|t|': self.pvalues,
            f'[{alpha/2:.3f}': self.conf_int_lower,
            f'{1-alpha/2:.3f}]': self.conf_int_upper
        })
        
        # Format the output
        output = []
        output.append("=" * 80)
        output.append(f"Model: {self.model_info.get('model_type', 'Unknown')}")
        output.append(f"Method: {self.model_info.get('method', 'Unknown')}")
        if 'dependent_var' in self.data_info:
            output.append(f"Dependent Variable: {self.data_info['dependent_var']}")
        output.append("=" * 80)
        
        # Add coefficient table
        output.append(coef_table.to_string(float_format='%.4f'))
        
        # Add model diagnostics
        if self.diagnostics:
            output.append("")
            output.append("Model Diagnostics:")
            output.append("-" * 20)
            for key, value in self.diagnostics.items():
                if isinstance(value, (int, float)):
                    output.append(f"{key:20s}: {value:.4f}")
                else:
                    output.append(f"{key:20s}: {value}")
        
        output.append("=" * 80)
        return "\n".join(output)
    
    def conf_int(self, alpha: float = 0.05) -> pd.DataFrame:
        """
        Return confidence intervals for parameters
        
        Parameters
        ----------
        alpha : float, default 0.05
            Significance level
            
        Returns
        -------
        pd.DataFrame
            Confidence intervals
        """
        t_crit = stats.t.ppf(1 - alpha/2, self.data_info.get('df_resid', np.inf))
        lower = self.params - t_crit * self.std_errors
        upper = self.params + t_crit * self.std_errors
        
        return pd.DataFrame({
            f'{alpha/2:.3f}': lower,
            f'{1-alpha/2:.3f}': upper
        }, index=self.params.index)
    
    def predict(self, data: Optional[pd.DataFrame] = None) -> np.ndarray:
        """
        Generate predictions (to be implemented by specific models)
        
        Parameters
        ----------
        data : pd.DataFrame, optional
            Data for prediction
            
        Returns
        -------
        np.ndarray
            Predicted values
        """
        raise NotImplementedError("Prediction method not implemented for this model")
    
    def residuals(self) -> Optional[np.ndarray]:
        """
        Return model residuals if available
        
        Returns
        -------
        np.ndarray or None
            Residuals
        """
        return self.data_info.get('residuals')
    
    def fitted_values(self) -> Optional[np.ndarray]:
        """
        Return fitted values if available
        
        Returns
        -------
        np.ndarray or None
            Fitted values
        """
        return self.data_info.get('fitted_values')
    
    def to_docx(self, filename: str, title: Optional[str] = None):
        """
        Export results to a Word (.docx) document.

        Parameters
        ----------
        filename : str
            Output path (.docx).
        title : str, optional
            Table title. Defaults to model type.
        """
        _result_to_docx(self, filename, title)

    def __repr__(self) -> str:
        """String representation of results"""
        model_type = self.model_info.get('model_type', 'Unknown')
        n_params = len(self.params)
        n_obs = self.data_info.get('nobs', 'Unknown')
        return f"<EconometricResults: {model_type}, {n_params} parameters, {n_obs} observations>"


class CausalResult:
    """
    Unified result object for all causal inference methods in StatsPAI.

    All causal inference estimators (DID, RD, SCM, matching, etc.) return
    this object, providing a consistent interface for summaries, plots,
    and publication-quality output.

    Parameters
    ----------
    method : str
        Name of the estimation method (displayed in summary).
    estimand : str
        What is being estimated ('ATT', 'ATE', 'LATE').
    estimate : float
        Point estimate of the main treatment effect.
    se : float
        Standard error.
    pvalue : float
        Two-sided p-value for H0: effect = 0.
    ci : tuple of (float, float)
        Confidence interval (lower, upper).
    alpha : float
        Significance level used for CI.
    n_obs : int
        Number of observations.
    detail : pd.DataFrame, optional
        Detailed estimates (e.g., group-time ATTs).
    model_info : dict, optional
        Model metadata and aggregated results.
    _influence_funcs : np.ndarray, optional
        Influence function matrix (n_units, n_estimates) for joint inference.
    _citation_key : str, optional
        Key into the citation registry.
    """

    _CITATIONS: Dict[str, str] = {
        'did_2x2': (
            "@book{angrist2009mostly,\n"
            "  title={Mostly Harmless Econometrics: An Empiricist's Companion},\n"
            "  author={Angrist, Joshua D and Pischke, J{\\\"o}rn-Steffen},\n"
            "  year={2009},\n"
            "  publisher={Princeton University Press}\n"
            "}"
        ),
        'callaway_santanna': (
            "@article{callaway2021difference,\n"
            "  title={Difference-in-differences with multiple time periods},\n"
            "  author={Callaway, Brantly and Sant'Anna, Pedro H.C.},\n"
            "  journal={Journal of Econometrics},\n"
            "  volume={225},\n"
            "  number={2},\n"
            "  pages={200--230},\n"
            "  year={2021},\n"
            "  publisher={Elsevier}\n"
            "}"
        ),
        'sun_abraham': (
            "@article{sun2021estimating,\n"
            "  title={Estimating dynamic treatment effects in event studies "
            "with heterogeneous treatment effects},\n"
            "  author={Sun, Liyang and Abraham, Sarah},\n"
            "  journal={Journal of Econometrics},\n"
            "  volume={225},\n"
            "  number={2},\n"
            "  pages={175--199},\n"
            "  year={2021},\n"
            "  publisher={Elsevier}\n"
            "}"
        ),
        'rdrobust': (
            "@article{calonico2014robust,\n"
            "  title={Robust nonparametric confidence intervals for "
            "regression-discontinuity designs},\n"
            "  author={Calonico, Sebastian and Cattaneo, Matias D "
            "and Titiunik, Rocio},\n"
            "  journal={Econometrica},\n"
            "  volume={82},\n"
            "  number={6},\n"
            "  pages={2295--2326},\n"
            "  year={2014},\n"
            "  publisher={Wiley}\n"
            "}"
        ),
        'bacon_decomposition': (
            "@article{goodman2021difference,\n"
            "  title={Difference-in-differences with variation in treatment timing},\n"
            "  author={Goodman-Bacon, Andrew},\n"
            "  journal={Journal of Econometrics},\n"
            "  volume={225},\n"
            "  number={2},\n"
            "  pages={254--277},\n"
            "  year={2021},\n"
            "  publisher={Elsevier}\n"
            "}"
        ),
    }

    def __init__(
        self,
        method: str,
        estimand: str,
        estimate: float,
        se: float,
        pvalue: float,
        ci: tuple,
        alpha: float,
        n_obs: int,
        detail: Optional[pd.DataFrame] = None,
        model_info: Optional[Dict[str, Any]] = None,
        _influence_funcs: Optional[np.ndarray] = None,
        _citation_key: Optional[str] = None,
    ):
        self.method = method
        self.estimand = estimand
        self.estimate = estimate
        self.se = se
        self.pvalue = pvalue
        self.ci = ci
        self.alpha = alpha
        self.n_obs = n_obs
        self.detail = detail
        self.model_info = model_info or {}
        self._influence_funcs = _influence_funcs
        self._citation_key = _citation_key

    # ------------------------------------------------------------------
    # Backward compatibility with EconometricResults
    # ------------------------------------------------------------------

    @property
    def params(self) -> pd.Series:
        """Treatment effect as a params Series (for outreg2 compatibility)."""
        return pd.Series({self.estimand: self.estimate})

    @property
    def std_errors(self) -> pd.Series:
        return pd.Series({self.estimand: self.se})

    @property
    def tvalues(self) -> pd.Series:
        t = self.estimate / self.se if self.se > 0 else np.nan
        return pd.Series({self.estimand: t})

    @property
    def pvalues(self) -> pd.Series:
        return pd.Series({self.estimand: self.pvalue})

    @property
    def diagnostics(self) -> Dict[str, Any]:
        return self.model_info

    @property
    def data_info(self) -> Dict[str, Any]:
        return {'nobs': self.n_obs}

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _stars(pvalue: float) -> str:
        """Significance stars."""
        if pd.isna(pvalue):
            return ""
        if pvalue < 0.01:
            return "***"
        if pvalue < 0.05:
            return "**"
        if pvalue < 0.1:
            return "*"
        return ""

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self, alpha: Optional[float] = None) -> str:
        """
        Generate a formatted text summary of the causal estimation results.

        Parameters
        ----------
        alpha : float, optional
            Override significance level for display.

        Returns
        -------
        str
        """
        alpha = alpha or self.alpha
        lines: List[str] = []

        lines.append("=" * 78)
        lines.append(f"  {self.method}")
        lines.append("=" * 78)
        lines.append("")

        stars = self._stars(self.pvalue)
        lines.append(f"  {self.estimand}:      {self.estimate: .6f} {stars}")
        lines.append(f"  Std. Error:  ({self.se:.6f})")
        pct = int(100 * (1 - alpha))
        lines.append(f"  [{pct}% CI]:    [{self.ci[0]:.6f},  {self.ci[1]:.6f}]")
        lines.append(f"  P-value:     {self.pvalue:.4f}")
        lines.append("")

        # Event study coefficients
        if 'event_study' in self.model_info:
            es = self.model_info['event_study']
            lines.append("-" * 78)
            lines.append("  Event Study Coefficients")
            lines.append("-" * 78)
            for _, row in es.iterrows():
                e = int(row['relative_time'])
                att = row['att']
                se_v = row['se']
                pv = row.get('pvalue', np.nan)
                s = self._stars(pv)
                lines.append(
                    f"  e = {e:>3d}  |  {att:>10.4f}  ({se_v:.4f}) {s}"
                )
            lines.append("")

        # Detailed estimates
        if self.detail is not None and len(self.detail) > 0:
            if 'att' in self.detail.columns:
                # Causal inference format (group-time ATTs)
                cols = [c for c in ['group', 'time', 'att', 'se',
                                    'ci_lower', 'ci_upper', 'pvalue']
                        if c in self.detail.columns]
                title_str = "Group-Time ATT Estimates"
            elif 'method' in self.detail.columns and 'estimate' in self.detail.columns:
                # RD-style inference table
                cols = [c for c in ['method', 'estimate', 'se', 'z',
                                    'pvalue', 'ci_lower', 'ci_upper']
                        if c in self.detail.columns]
                title_str = "Inference"
            elif 'coefficient' in self.detail.columns:
                # Regression format (variable / coefficient / se)
                cols = [c for c in ['variable', 'coefficient', 'se',
                                    'tstat', 'pvalue']
                        if c in self.detail.columns]
                title_str = "Regression Coefficients"
            else:
                cols = list(self.detail.columns)
                title_str = "Detailed Estimates"
            lines.append("-" * 78)
            lines.append(f"  {title_str}")
            lines.append("-" * 78)
            lines.append(
                self.detail[cols].to_string(index=False, float_format='%.4f')
            )
            lines.append("")

        # Pre-trend test
        if 'pretrend_test' in self.model_info:
            pt = self.model_info['pretrend_test']
            lines.append("-" * 78)
            lines.append(
                f"  Pre-trend Test: chi2({pt['df']}) = {pt['statistic']:.4f}, "
                f"p-value = {pt['pvalue']:.4f}"
            )
            lines.append("")

        # Model info footer
        lines.append("-" * 78)
        lines.append(f"  Observations:    {self.n_obs:,}")
        _skip = {'event_study', 'pretrend_test', 'aggregations',
                 'cohort_sizes', 'influence_funcs_matrix',
                 'conventional', 'robust'}
        for key, val in self.model_info.items():
            if key in _skip or isinstance(val, (pd.DataFrame, np.ndarray,
                                                dict, list)):
                continue
            label = key.replace('_', ' ').title()
            lines.append(f"  {label}:    {val}")
        lines.append("=" * 78)
        lines.append("  * p<0.1, ** p<0.05, *** p<0.01")

        return "\n".join(lines)

    # ------------------------------------------------------------------
    # Plotting
    # ------------------------------------------------------------------

    def event_study_plot(
        self,
        ax=None,
        title: Optional[str] = None,
        color: str = '#2C3E50',
        ci_alpha: float = 0.15,
        figsize: tuple = (10, 6),
        **kwargs,
    ):
        """
        Plot event study coefficients with confidence intervals.

        Parameters
        ----------
        ax : matplotlib.axes.Axes, optional
        title : str, optional
        color : str
        ci_alpha : float
        figsize : tuple

        Returns
        -------
        (fig, ax)
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError(
                "matplotlib required for plotting. "
                "Install: pip install matplotlib"
            )

        if 'event_study' not in self.model_info:
            raise ValueError(
                "No event study estimates. Use a staggered DID method."
            )

        es = self.model_info['event_study'].copy()

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()

        e = es['relative_time'].values
        att = es['att'].values
        lo = es['ci_lower'].values
        hi = es['ci_upper'].values

        ax.fill_between(e, lo, hi, alpha=ci_alpha, color=color)
        ax.scatter(e, att, color=color, s=40, zorder=5)
        ax.plot(e, att, color=color, linewidth=1, alpha=0.7, zorder=4)
        ax.errorbar(
            e, att,
            yerr=[att - lo, hi - att],
            fmt='none', color=color, capsize=3, linewidth=1, zorder=3,
        )

        ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8)
        ax.axvline(
            x=-0.5, color='#E74C3C', linestyle=':',
            linewidth=1, alpha=0.5, label='Treatment onset',
        )

        ax.set_xlabel('Periods Relative to Treatment', fontsize=11)
        ax.set_ylabel('Estimated Effect', fontsize=11)
        ax.set_title(title or f'Event Study: {self.method}', fontsize=13)
        ax.tick_params(labelsize=10)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.legend(fontsize=9, frameon=False)
        fig.tight_layout()
        return fig, ax

    def plot(self, type: str = 'auto', **kwargs):
        """
        Generate appropriate visualisation.

        Parameters
        ----------
        type : str
            'auto', 'event_study', or 'coefplot'.
        """
        if type == 'auto':
            if 'event_study' in self.model_info:
                return self.event_study_plot(**kwargs)
            return self._coefplot(**kwargs)
        if type == 'event_study':
            return self.event_study_plot(**kwargs)
        return self._coefplot(**kwargs)

    def _coefplot(self, ax=None, figsize=(8, 5), **kwargs):
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("matplotlib required for plotting")

        if ax is None:
            fig, ax = plt.subplots(figsize=figsize)
        else:
            fig = ax.get_figure()

        ax.errorbar(
            0, self.estimate,
            yerr=[[self.estimate - self.ci[0]], [self.ci[1] - self.estimate]],
            fmt='o', color='#2C3E50', capsize=5, markersize=8,
        )
        ax.axhline(y=0, color='gray', linestyle='--', linewidth=0.8)
        ax.set_xlim(-1, 1)
        ax.set_xticks([0])
        ax.set_xticklabels([self.estimand])
        ax.set_ylabel('Estimated Effect')
        ax.set_title(f'{self.method}: {self.estimand}')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        fig.tight_layout()
        return fig, ax

    # ------------------------------------------------------------------
    # Export
    # ------------------------------------------------------------------

    def to_latex(self, caption: Optional[str] = None,
                 label: Optional[str] = None) -> str:
        """Generate a LaTeX table of the results."""
        caption = caption or f'{self.method} Results'
        label = label or 'tab:causal_result'

        lines = [
            '\\begin{table}[htbp]',
            '\\centering',
            f'\\caption{{{caption}}}',
            f'\\label{{{label}}}',
        ]

        if self.detail is not None and len(self.detail) > 0:
            # Detect table format: causal (group/time/att) vs regression (variable/coefficient)
            if 'att' in self.detail.columns:
                cols = [c for c in ['group', 'time', 'att', 'se', 'pvalue']
                        if c in self.detail.columns]
                coef_col, star_col = 'att', 'att'
            elif 'method' in self.detail.columns and 'estimate' in self.detail.columns:
                cols = [c for c in ['method', 'estimate', 'se', 'pvalue']
                        if c in self.detail.columns]
                coef_col, star_col = 'estimate', 'estimate'
            elif 'coefficient' in self.detail.columns:
                cols = [c for c in ['variable', 'coefficient', 'se', 'pvalue']
                        if c in self.detail.columns]
                coef_col, star_col = 'coefficient', 'coefficient'
            else:
                cols = list(self.detail.columns)
                coef_col, star_col = None, None

            n_cols = len(cols)
            spec = 'l' + 'c' * (n_cols - 1)
            lines.append(f'\\begin{{tabular}}{{{spec}}}')
            lines.append('\\hline\\hline')
            hdr = {'group': 'Group', 'time': 'Time', 'att': 'ATT',
                   'se': 'Std.\\ Error', 'pvalue': 'P-value',
                   'variable': 'Variable', 'coefficient': 'Coefficient',
                   'tstat': 't-stat'}
            lines.append(' & '.join(hdr.get(c, c) for c in cols) + ' \\\\')
            lines.append('\\hline')
            for _, row in self.detail.iterrows():
                vals = []
                for c in cols:
                    v = row[c]
                    if isinstance(v, float):
                        s = self._stars(row.get('pvalue', np.nan)) if c == star_col else ''
                        vals.append(f'{v:.4f}{s}')
                    else:
                        vals.append(str(int(v)) if isinstance(v, (int, np.integer)) else str(v))
                lines.append(' & '.join(vals) + ' \\\\')
        else:
            lines.append('\\begin{tabular}{lc}')
            lines.append('\\hline\\hline')
            lines.append(
                f'{self.estimand} & '
                f'{self.estimate:.4f}{self._stars(self.pvalue)} \\\\'
            )
            lines.append(f'& ({self.se:.4f}) \\\\')

        lines += [
            '\\hline',
            f'Observations & {self.n_obs:,} \\\\',
            '\\hline\\hline',
            '\\end{tabular}',
            '\\begin{tablenotes}',
            '\\footnotesize',
            '\\item Standard errors in parentheses.',
            '\\item * p<0.1, ** p<0.05, *** p<0.01',
            '\\end{tablenotes}',
            '\\end{table}',
        ]
        return '\n'.join(lines)

    def cite(self) -> str:
        """Return BibTeX citation for the method."""
        key = self._citation_key or self.method.lower().replace(' ', '_')
        if key in self._CITATIONS:
            return self._CITATIONS[key]
        for k, v in self._CITATIONS.items():
            if k in key or key in k:
                return v
        return f"% No citation registered for method: {self.method}"

    def pretrend_test(self) -> Dict[str, Any]:
        """Return pre-trend test results (DID methods)."""
        if 'pretrend_test' not in self.model_info:
            raise ValueError("Pre-trend test not available for this method.")
        return self.model_info['pretrend_test']

    def to_docx(self, filename: str, title: Optional[str] = None):
        """
        Export results to a Word (.docx) document.

        Parameters
        ----------
        filename : str
            Output path (.docx).
        title : str, optional
            Table title. Defaults to method name.
        """
        _result_to_docx(self, filename, title)

    def __repr__(self) -> str:
        s = self._stars(self.pvalue)
        return (
            f"<CausalResult: {self.method}, {self.estimand} = "
            f"{self.estimate:.4f}{s}, SE = {self.se:.4f}, n = {self.n_obs:,}>"
        )

    def __str__(self) -> str:
        return self.summary()


# ======================================================================
# Shared Word export helper
# ======================================================================

def _result_to_docx(result, filename: str, title: Optional[str] = None):
    """
    Export a single EconometricResults or CausalResult to Word (.docx).

    Produces a one-model table with coefficients, SEs, stars, and
    diagnostics in APA format.
    """
    try:
        from docx import Document
        from docx.shared import Pt
        from docx.enum.text import WD_ALIGN_PARAGRAPH
        from docx.enum.table import WD_TABLE_ALIGNMENT
        from docx.oxml.ns import qn
        from docx.oxml import OxmlElement
    except ImportError:
        raise ImportError(
            "python-docx required for Word export. "
            "Install: pip install python-docx"
        )

    if not filename.endswith('.docx'):
        filename += '.docx'

    doc = Document()

    # Title
    if title is None:
        title = getattr(result, 'method', None) or result.model_info.get('model_type', 'Results')
    p = doc.add_paragraph()
    run = p.add_run(str(title))
    run.bold = True
    run.font.size = Pt(12)
    p.alignment = WD_ALIGN_PARAGRAPH.CENTER

    # Build rows from result
    params = result.params
    std_errors = result.std_errors
    pvalues = result.pvalues if hasattr(result, 'pvalues') else None

    def _stars(pv):
        if pv is None or (isinstance(pv, float) and np.isnan(pv)):
            return ''
        if pv < 0.01: return '***'
        if pv < 0.05: return '**'
        if pv < 0.1: return '*'
        return ''

    # Table: Variable | Coefficient | SE
    rows_data = []
    if isinstance(params, pd.Series):
        for var in params.index:
            coef = params[var]
            se = std_errors[var] if var in std_errors.index else np.nan
            if pvalues is not None and isinstance(pvalues, pd.Series) and var in pvalues.index:
                pv = float(pvalues[var])
            else:
                pv = np.nan
            rows_data.append((str(var), f'{coef:.4f}{_stars(pv)}', f'({se:.4f})'))

    n_rows = len(rows_data) * 2 + 1  # coef rows + SE rows + header
    # Actually: each variable gets 2 rows (coef, SE)
    table = doc.add_table(rows=1 + len(rows_data) * 2, cols=2)
    table.alignment = WD_TABLE_ALIGNMENT.CENTER

    # Header
    table.rows[0].cells[0].text = ''
    table.rows[0].cells[1].text = title
    for cell in table.rows[0].cells:
        for para in cell.paragraphs:
            para.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for run in para.runs:
                run.bold = True
                run.font.size = Pt(10)

    # Data
    row_idx = 1
    for var, coef_str, se_str in rows_data:
        # Coefficient row
        table.rows[row_idx].cells[0].text = var
        table.rows[row_idx].cells[1].text = coef_str
        for cell in table.rows[row_idx].cells:
            for para in cell.paragraphs:
                para.alignment = WD_ALIGN_PARAGRAPH.CENTER if cell == table.rows[row_idx].cells[1] else WD_ALIGN_PARAGRAPH.LEFT
                for run in para.runs:
                    run.font.size = Pt(9)
        row_idx += 1

        # SE row
        table.rows[row_idx].cells[0].text = ''
        table.rows[row_idx].cells[1].text = se_str
        for para in table.rows[row_idx].cells[1].paragraphs:
            para.alignment = WD_ALIGN_PARAGRAPH.CENTER
            for run in para.runs:
                run.font.size = Pt(9)
        row_idx += 1

    # Diagnostics as additional paragraph
    diag = getattr(result, 'diagnostics', {})
    n_obs = None
    if hasattr(result, 'data_info') and isinstance(result.data_info, dict):
        n_obs = result.data_info.get('nobs')
    if hasattr(result, 'n_obs'):
        n_obs = result.n_obs

    diag_lines = []
    if n_obs is not None:
        diag_lines.append(f'Observations: {n_obs:,}')
    if isinstance(diag, dict):
        for k, v in diag.items():
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                diag_lines.append(f'{k}: {v:.4f}' if isinstance(v, float) else f'{k}: {v:,}')

    if diag_lines:
        dp = doc.add_paragraph()
        for line in diag_lines:
            run = dp.add_run(line + '\n')
            run.font.size = Pt(8)

    # Significance note
    np_ = doc.add_paragraph()
    run = np_.add_run('* p<0.1, ** p<0.05, *** p<0.01')
    run.italic = True
    run.font.size = Pt(8)

    doc.save(filename)
