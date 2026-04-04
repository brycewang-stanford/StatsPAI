"""
StatsPAI: The AI-powered Statistics & Econometrics Toolkit for Python

Unified API for causal inference and econometrics:

>>> import statspai as sp
>>>
>>> # OLS regression
>>> result = sp.regress("y ~ x1 + x2", data=df)
>>>
>>> # Difference-in-Differences
>>> result = sp.did(df, y='wage', treat='treated', time='post')
>>>
>>> # Staggered DID (Callaway & Sant'Anna)
>>> result = sp.did(df, y='wage', treat='first_treat',
...                time='year', id='worker_id')
>>>
>>> # Causal Forest
>>> cf = sp.causal_forest("y ~ treatment | x1 + x2", data=df)
>>>
>>> # Publication-quality export
>>> sp.outreg2(result, filename="results.xlsx")
"""

__version__ = "0.2.0"
__author__ = "Bryce Wang"
__email__ = "bryce@copaper.ai"

from .core.results import EconometricResults, CausalResult
from .regression.ols import regress
from .regression.iv import ivreg, IVRegression
from .causal.causal_forest import CausalForest, causal_forest
from .did import did, did_2x2, callaway_santanna, sun_abraham, bacon_decomposition, honest_did, breakdown_m
from .rd import rdrobust, rdplot
from .synth import (
    synth, SyntheticControl, sdid,
    synthdid_estimate, sc_estimate, did_estimate,
    synthdid_placebo, synthdid_plot, synthdid_units_plot, synthdid_rmse_plot,
    california_prop99,
)
from .matching import match, MatchEstimator, ebalance
from .dml import dml, DoubleML
from .deepiv import deepiv, DeepIV
from .panel import panel, PanelRegression
from .causal_impact import causal_impact, CausalImpactEstimator
from .mediation import mediate, MediationAnalysis
from .bartik import bartik, BartikIV
from .output.outreg2 import OutReg2, outreg2
from .output.modelsummary import modelsummary, coefplot
from .output.sumstats import sumstats, balance_table
from .output.tab import tab
from .postestimation import margins, marginsplot, test, lincom
from .diagnostics import oster_bounds, mccrary_test, diagnose, het_test, reset_test, vif, sensemakr, rddensity, hausman_test, anderson_rubin_test
from .inference import wild_cluster_bootstrap, aipw, ri_test
from .plots import binscatter, set_theme
from .utils import label_var, label_vars, get_label, get_labels, describe, pwcorr, winsor, read_data
from .gmm import xtabond
from .regression.heckman import heckman
from .regression.quantile import qreg, sqreg
from .regression.tobit import tobit

__all__ = [
    # Core
    "EconometricResults",
    "CausalResult",
    # Regression
    "regress",
    "ivreg",
    "IVRegression",
    # DID
    "did",
    "did_2x2",
    "callaway_santanna",
    "sun_abraham",
    "bacon_decomposition",
    "honest_did",
    "breakdown_m",
    # RD
    "rdrobust",
    "rdplot",
    # Synthetic Control
    "synth",
    "SyntheticControl",
    "sdid",
    "synthdid_estimate",
    "sc_estimate",
    "did_estimate",
    "synthdid_placebo",
    "synthdid_plot",
    "synthdid_units_plot",
    "synthdid_rmse_plot",
    "california_prop99",
    # Matching
    "match",
    "MatchEstimator",
    "ebalance",
    # Double ML
    "dml",
    "DoubleML",
    # DeepIV
    "deepiv",
    "DeepIV",
    # Panel
    "panel",
    "PanelRegression",
    # Causal Impact
    "causal_impact",
    "CausalImpactEstimator",
    # Causal Forest
    "CausalForest",
    "causal_forest",
    # Output
    "OutReg2",
    "outreg2",
    "modelsummary",
    "coefplot",
    "sumstats",
    "balance_table",
    "tab",
    # Plots
    "binscatter",
    "set_theme",
    # Utils
    "label_var",
    "label_vars",
    "get_label",
    "get_labels",
    "describe",
    "pwcorr",
    "winsor",
    "read_data",
    # Dynamic Panel GMM
    "xtabond",
    "heckman",
    "qreg",
    "sqreg",
    "tobit",
    # Post-estimation
    "margins",
    "marginsplot",
    "test",
    "lincom",
    # Mediation
    "mediate",
    "MediationAnalysis",
    # Bartik IV
    "bartik",
    "BartikIV",
    # Diagnostics
    "oster_bounds",
    "mccrary_test",
    "diagnose",
    "het_test",
    "reset_test",
    "vif",
    "sensemakr",
    "rddensity",
    "hausman_test",
    "anderson_rubin_test",
    # Inference
    "wild_cluster_bootstrap",
    "aipw",
    "ri_test",
]
