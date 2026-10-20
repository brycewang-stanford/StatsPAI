"""Failure modes read from each function's own ``raise`` statements.

DO NOT EDIT BY HAND. Regenerate with::

    python scripts/gen_source_failure_modes.py

Each entry quotes a typed ``StatsPAIError`` the function raises together
with the ``recovery_hint`` written next to it. The registry uses an entry
only when the function has no hand-written or inherited failure mode.
"""

from __future__ import annotations

from typing import Dict, List

SOURCE_FAILURE_MODES: Dict[str, List[Dict[str, str]]] = {
    "ardl": [
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Use 'c', 'ct' or 'n'.",
            "symptom": "ardl: unknown trend <trend>.",
        },
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Name each regressor once and leave the outcome out of " "x=.",
            "symptom": "ardl: x=<xs> repeats a column or lists the outcome <y>. "
            "The lags of the outcome enter through lags=, not x=.",
        },
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Pass one row per period; for a panel, fit each unit "
            "separately.",
            "symptom": "ardl: the time index has repeated values.",
        },
        {
            "exception": "statspai.DataInsufficient",
            "remedy": "Check the columns.",
            "symptom": "ardl: no row has all of the series observed.",
        },
    ],
    "as_svrepdesign": [
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Use 'JK1', 'JKn', 'BRR' or 'Fay'; for "
            "bootstrap weights pass them to "
            "sp.svrepdesign(type='bootstrap').",
            "symptom": "as_svrepdesign: type=<type> is not built " "here.",
        },
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Build the design with sp.svydesign(...) first.",
            "symptom": "as_svrepdesign: design must be a SurveyDesign "
            "from sp.svydesign.",
        },
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Convert the uncalibrated design.",
            "symptom": "as_svrepdesign: a calibrated design cannot be "
            "converted (the calibration would have to be "
            "repeated on every replicate).",
        },
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Use type='JKn'.",
            "symptom": "as_svrepdesign: JK1 is for unstratified "
            "designs; this one has <H> strata.",
        },
    ],
    "bagged_forecast": [
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Pass a positive number of periods.",
            "symptom": "bagged_forecast: horizon must be at least " "1.",
        },
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "E.g. forecaster='ets'.",
            "symptom": "bagged_forecast: forecaster must be a method "
            "name or a function.",
        },
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Bind them in the function itself.",
            "symptom": "bagged_forecast: unexpected arguments "
            "<sorted(kwargs)> with a callable "
            "forecaster.",
        },
    ],
    "balance_table": [
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Use test='ttest' or 'ranksum'.",
            "symptom": "balance_table: unknown test <test>.",
        }
    ],
    "balance_vs_randomization": [
        {
            "exception": "statspai.DataInsufficient",
            "remedy": "Check the treat column.",
            "symptom": "balance_vs_randomization: need at "
            "least two treated and two control "
            "individuals.",
        }
    ],
    "bart_causal": [
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Use cut_grid='auto'.",
            "symptom": "bart_causal: cut_grid must be 'auto', 'quantile' "
            "or 'uniform'; got <cut_grid>.",
        },
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Use method='bcf' or 'bart'.",
            "symptom": "bart_causal: method must be 'bcf' or 'bart'; got " "<method>.",
        },
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Use estimand='ATE'.",
            "symptom": "bart_causal: estimand must be 'ATE', 'ATT' or "
            "'ATC'; got <estimand>.",
        },
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Use the defaults (draws=1000, burnin=1000, " "chains=4).",
            "symptom": "bart_causal: draws >= 10, burnin >= 0, thin >= "
            "1, chains >= 1.",
        },
    ],
    "bayes_arima": [
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Pass data=<DataFrame>, or pass y as an array.",
            "symptom": "y=<y> names a column, but data= was not given.",
        }
    ],
    "bds": [
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Drop them; the test needs consecutive observations.",
            "symptom": "bds: the series has missing values.",
        },
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": 'sp.bds(df, "resid")',
            "symptom": "bds: y= names the column to test.",
        },
    ],
    "best_subset": [
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Screen first with sp.lasso_select or sp.stepwise.",
            "symptom": "best_subset: <p> candidates; the exact search is "
            "limited to <_MAX_CANDIDATES>.",
        },
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Omit max_vars to search every size.",
            "symptom": "best_subset: max_vars=<max_vars> must be at " "least 1.",
        },
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Drop them from x.",
            "symptom": "best_subset: candidate(s) <bad> are constant or "
            "collinear with the forced regressors.",
        },
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Drop the redundant column(s) from x.",
            "symptom": "best_subset: the candidates are perfectly " "collinear.",
        },
    ],
    "beveridge_nelson": [
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Use ic='bic' or ic='aic'.",
            "symptom": "sp.beveridge_nelson: ic=<ic> is not 'aic' " "or 'bic'.",
        },
        {
            "exception": "statspai.DataInsufficient",
            "remedy": "Use a longer series.",
            "symptom": "sp.beveridge_nelson: <level.size> " "observations are too few.",
        },
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Add arma=True, or pass an integer max_order.",
            "symptom": "sp.beveridge_nelson: max_order=(P, Q) is "
            "the grid of the ARMA search.",
        },
        {
            "exception": "statspai.DataInsufficient",
            "remedy": "Lower max_order or use a longer series.",
            "symptom": "sp.beveridge_nelson: max_order=<pmax> with " "<n> differences.",
        },
    ],
    "binscatter": [
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Pass method='binsreg' for the CCFF (2024) " "binscatter.",
            "symptom": "binscatter: method must be 'classic' or "
            "'binsreg', not <method>",
        },
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Add the fixed effects to controls as indicator "
            "columns, or use method='classic'.",
            "symptom": "binscatter(method='binsreg') does not absorb " "fixed effects",
        },
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Use method='classic' for weighted bin means.",
            "symptom": "binscatter(method='binsreg') does not take " "weights",
        },
    ],
    "blp_instruments": [
        {
            "exception": "statspai.DataInsufficient",
            "remedy": "Drop those rows, or fill in the identifiers.",
            "symptom": "blp_instruments: missing values in the "
            "grouping column(s) <unlabelled> (column: "
            "number of rows).",
        }
    ],
    "bootstrap_series": [
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Pass a positive number of series.",
            "symptom": "bootstrap_series: n_boot must be at least " "1.",
        },
        {
            "exception": "statspai.DataInsufficient",
            "remedy": "At least two seasonal cycles (or 8 points) " "are needed.",
            "symptom": "bootstrap_series: <n> observations are too " "few.",
        },
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Leave block_size=None for the default.",
            "symptom": "bootstrap_series: block_size=<block> must "
            "be between 1 and <n>.",
        },
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Use boxcox='auto'.",
            "symptom": "bootstrap_series: boxcox=<boxcox> is not a "
            "number, 'auto' or None.",
        },
    ],
    "boxcox_lambda": [
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Shift the series or model it untransformed.",
            "symptom": "boxcox_lambda: Guerrero's method needs a "
            "strictly positive series.",
        },
        {
            "exception": "statspai.DataInsufficient",
            "remedy": "Provide a longer series.",
            "symptom": "boxcox_lambda: <n> observations give fewer "
            "than two cycles of length <m>.",
        },
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Use lower < upper, e.g. lower=-1, upper=2.",
            "symptom": "boxcox_lambda: lower=<lower> and upper=<upper> "
            "do not bracket an interval to search.",
        },
        {
            "exception": "statspai.DataInsufficient",
            "remedy": "No transformation is needed for such a series.",
            "symptom": "boxcox_lambda: the series does not vary within "
            "its cycles (it is constant), so there is no "
            "spread to stabilise.",
        },
    ],
    "cate_gain_curve": [
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Give the Series the index of data, or pass "
            "cate.to_numpy() if the values are already in "
            "row order.",
            "symptom": "cate_gain_curve: the index of the cate "
            "Series does not match the index of data, so "
            "the predictions cannot be paired with rows.",
        },
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Code the column as numbers: 0/1 for a two-arm "
            "treatment, or the dose.",
            "symptom": "cate_gain_curve: the <role> column <col> is "
            "not numeric (dtype <data[col].dtype>).",
        },
    ],
    "cate_pretrend_test": [
        {
            "alternative": "sp.did_forest",
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Fit sp.causal_forest(..., id=, time=, "
            "fe='twoway'); for staggered designs "
            "sp.did_forest() reports its own pre-trend "
            "test.",
            "symptom": "<context> needs a causal forest with "
            "fixed effects (fe=...).",
        },
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Use time_effects='common'.",
            "symptom": "<context>: time_effects must be 'common' " "or 'by_group'.",
        },
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Use a design in which units stay treated " "once treated.",
            "symptom": "<context> needs an absorbing (staggered) "
            "treatment so that leads relative to the "
            "adoption period are defined.",
        },
        {
            "exception": "statspai.DataInsufficient",
            "remedy": "Use n_groups >= 2 or pass two or more " "labels.",
            "symptom": "<context>: need at least two groups.",
        },
    ],
    "causal_discovery": [
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Choose one of: notears, pc, fci, ges, " "lingam.",
            "symptom": "Unknown causal_discovery method <method>. "
            "Expected one of: <sorted(valid)>.",
        }
    ],
    "chow_test": [
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Drop or fill them first; dropping rows silently "
            "would move the break date.",
            "symptom": "chow_test: missing values in y or x.",
        },
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "A regressor that is constant inside a regime cannot "
            "have its own coefficient there; leave it out of "
            "break_vars.",
            "symptom": "chow_test: the regressors are collinear within a " "regime.",
        },
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Every observation needs a date to be placed in a " "regime.",
            "symptom": "chow_test: missing values in the time column " "<time>.",
        },
        {
            "exception": "statspai.DataInsufficient",
            "remedy": "Move the break, or let fewer coefficients change.",
            "symptom": "chow_test: a regime of <b - a> observations cannot "
            "identify <q> coefficients.",
        },
    ],
    "ci": [
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "sp.ci(df, 'y', stat='means')",
            "symptom": "ci: stat=<stat> is not one of 'means', 'variances', 'sd', "
            "'proportions'.",
        },
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "method applies to stat='proportions'.",
            "symptom": "ci: method=<method> is not one of "
            "<list(_PROPORTION_METHODS)>.",
        },
        {
            "exception": "statspai.DataInsufficient",
            "remedy": "Check the variable.",
            "symptom": "ci: <name> has too few observations.",
        },
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Recode the variable first.",
            "symptom": "ci: <name> must be coded 0 / 1 for a proportion.",
        },
    ],
    "classical_decompose": [
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Use model='additive'.",
            "symptom": "classical_decompose: model=<model> is "
            "not 'additive' or 'multiplicative'.",
        },
        {
            "exception": "statspai.DataInsufficient",
            "remedy": "At least two full cycles and period >= 2 " "are needed.",
            "symptom": "classical_decompose: <n> observations "
            "are fewer than two cycles of period "
            "<m>.",
        },
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Use model='additive'.",
            "symptom": "classical_decompose: a multiplicative "
            "decomposition needs a strictly positive "
            "series.",
        },
    ],
    "cnsreg": [
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Use method='cls' (constrained least squares) or 'emd' "
            "(efficient minimum distance).",
            "symptom": "sp.cnsreg: method=<method> is not 'cls' or 'emd'.",
        },
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Use vce='robust', or cluster='<column>'.",
            "symptom": "sp.cnsreg: vce=<vce> is not 'ols', 'robust', 'hc2' or "
            "'hc3'.",
        },
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Pass constraints='x1 + x2 = 1', or use sp.regress.",
            "symptom": "sp.cnsreg: no constraint given.",
        },
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Remove constraints.",
            "symptom": "sp.cnsreg: <q> independent constraints on <k> "
            "coefficients leave nothing to estimate.",
        },
    ],
    "conformal_regression": [
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Drop or fill them before predicting.",
            "symptom": "conformal_regression: newdata has rows "
            "with missing regressors.",
        }
    ],
    "corrgram": [
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Pass p + q of the fitted ARMA model, or 0.",
            "symptom": "sp.corrgram: model_df=<model_df> is negative.",
        },
        {
            "exception": "statspai.DataInsufficient",
            "remedy": "Use a longer series or fewer lags.",
            "symptom": "sp.corrgram: lags=<lags> with <n> observations.",
        },
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Use pac='regression' (the default).",
            "symptom": "sp.corrgram: pac=<pac> is not 'regression' or " "'yw'.",
        },
        {
            "exception": "statspai.DataInsufficient",
            "remedy": "Autocorrelations need variation.",
            "symptom": "sp.corrgram: the series is constant.",
        },
    ],
    "ctree": [
        {
            "exception": "statspai.DataInsufficient",
            "remedy": "Drop those rows first.",
            "symptom": "ctree: the outcome '<y>' has missing values.",
        },
        {
            "exception": "statspai.DataInsufficient",
            "remedy": "A tree needs data to test on.",
            "symptom": "ctree: fewer than two observations.",
        },
    ],
    "cumulative_periodogram_test": [
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Pass e.g. alpha=0.05.",
            "symptom": "sp.<who>: alpha=<alpha> is not " "between 0 and 1.",
        },
        {
            "exception": "statspai.DataInsufficient",
            "remedy": "The test needs at least four " "observations.",
            "symptom": "sp.<who>: <n> observations.",
        },
        {
            "exception": "statspai.DataInsufficient",
            "remedy": "There is no variance to " "decompose.",
            "symptom": "sp.<who>: the series is " "constant.",
        },
    ],
    "cuped": [
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Use method='cuped' (default), 'cupac' or 'lin'.",
            "symptom": "method must be one of <list(_METHODS)>; got <method>.",
        },
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Without covariates use sp.difference_in_means.",
            "symptom": "cuped: no covariates given.",
        },
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Encode categorical covariates as indicators first.",
            "symptom": "cuped: covariates must be numeric; got <non_numeric>.",
        },
        {
            "exception": "statspai.DataInsufficient",
            "remedy": "Use fewer covariates.",
            "symptom": "cuped: <len(frame)> complete rows for "
            "<len(covariates)> covariates.",
        },
    ],
    "diversity_index": [
        {
            "exception": "statspai.DataInsufficient",
            "remedy": "Check the species and grouping columns for " "NaNs.",
            "symptom": "No records remain after dropping rows with "
            "missing group or species values.",
        },
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Pass long-format records with species= "
            "instead, or reshape to a site-by-species "
            "matrix.",
            "symptom": "Matrix input must be 1-D (one site) or 2-D "
            "(sites x species).",
        },
    ],
    "esize": [
        {
            "exception": "statspai.DataInsufficient",
            "remedy": "Check the grouping column for near-empty groups.",
            "symptom": "esize: each group needs at least two observations.",
        }
    ],
    "evidence_without_injustice": [
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Pass a function accepting a "
            "DataFrame and returning "
            "predictions.",
            "symptom": "predictor must be callable.",
        },
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Pass a function accepting (data, " "protected_value).",
            "symptom": "scm_intervention must be " "callable.",
        },
        {
            "exception": "statspai.ConvergenceFailure",
            "remedy": "Check that predictor and "
            "scm_intervention work on bootstrap "
            "resamples, or increase n_boot.",
            "symptom": "EWI bootstrap only produced "
            "<n_ok>/<n_boot> valid "
            "replicates.",
        },
        {
            "exception": "statspai.DataInsufficient",
            "remedy": "Provide at least two "
            "protected-attribute levels or "
            "explicit alternative_values.",
            "symptom": "Protected attribute <protected> "
            "has only one level; EWI test is "
            "undefined.",
        },
    ],
    "factor": [
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Use 'pf', 'ipf', 'pcf' or 'ml'.",
            "symptom": "sp.factor: method=<method> is not one of <', "
            "'.join(_METHODS)>.",
        },
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Ask for fewer factors than variables.",
            "symptom": "sp.factor: n_factors=<n_factors> is not between 1 and " "<p>.",
        },
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Pass n_factors=; compare fits with lr_factors, aic and " "bic.",
            "symptom": "sp.factor: method='ml' needs n_factors.",
        },
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Drop a variable that is a linear combination of the " "others.",
            "symptom": "sp.factor: the correlation matrix is singular.",
        },
    ],
    "forest_support": [
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Use k=10.",
            "symptom": "forest_support(): k must be a positive " "integer.",
        },
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Use quantile=0.95, alpha=0.05.",
            "symptom": "forest_support(): quantile and alpha must lie " "in (0, 1).",
        },
        {
            "exception": "statspai.DataInsufficient",
            "remedy": "Lower k.",
            "symptom": "forest_support(): fewer reference rows than " "k.",
        },
        {
            "exception": "statspai.DataInsufficient",
            "remedy": "Lower k.",
            "symptom": "forest_support(): the reference rows come "
            "from k or fewer units.",
        },
    ],
    "fourier_terms": [
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Pass the number of periods.",
            "symptom": "fourier_terms: at least one row is needed.",
        },
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "12 for monthly data, 52.18 for weekly data.",
            "symptom": "fourier_terms: period=<period> must be above " "1.",
        },
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "More harmonics than half the period are " "collinear.",
            "symptom": "fourier_terms: K=<K> must be between 1 and "
            "period/2 = <per / 2>.",
        },
    ],
    "full_match": [
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Call sp.full_match(df, y, treat, covariates).",
            "symptom": "full_match: treat= and covariates= are required.",
        },
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Use alpha=0.05 for a 95% interval.",
            "symptom": "full_match: alpha must lie strictly between 0 and "
            "1, got <alpha>.",
        },
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Use estimand='ATT'.",
            "symptom": "full_match: estimand must be ATT, ATC or ATE, got "
            "<estimand>.",
        },
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Use 'propensity', 'logit', 'mahalanobis' or " "'euclidean'.",
            "symptom": "full_match: unknown distance <distance>.",
        },
    ],
    "glmnet": [
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Other families are not implemented.",
            "symptom": "glmnet: family must be 'gaussian' or 'binomial', got "
            "<family>.",
        },
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Use rule='min'.",
            "symptom": "glmnet: rule must be 'min' or '1se', got <rule>.",
        },
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Recode the outcome.",
            "symptom": "glmnet: a binomial outcome must be coded 0/1.",
        },
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Encode categorical predictors as indicators.",
            "symptom": "glmnet: the outcome and predictors must be numeric.",
        },
    ],
    "hausman": [
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Choose one; sigmamore is the usual choice.",
            "symptom": "sp.hausman: sigmamore and sigmaless cannot both be " "set.",
        },
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Fit both models on the same regressors.",
            "symptom": "sp.hausman: the two results share no coefficient.",
        },
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Use results of sp.regress, sp.iv or sp.panel.",
            "symptom": "sp.hausman: sigmamore / sigmaless need the "
            "disturbance variance of both models, which a result "
            "does not store.",
        },
    ],
    "hierarchy": [
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": 'E.g. spec=[["country"], ["country", "state"]].',
            "symptom": "hierarchy: spec must be a non-empty list of "
            "non-empty key lists.",
        },
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Fill or drop them before aggregating.",
            "symptom": "hierarchy: <int(work[value].isna().sum())> missing "
            "values in <value>.",
        },
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Add the key that distinguishes them to the last "
            "entry of spec, or aggregate first.",
            "symptom": "hierarchy: <int(dup.sum())> duplicated (series, "
            "period) rows at the bottom level <bottom_keys>.",
        },
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Fill the gaps (e.g. with zeros for counts) first.",
            "symptom": "hierarchy: the bottom level is unbalanced (<n_bad> "
            "missing series-period cells).",
        },
    ],
    "hill_climb": [
        {
            "exception": "statspai.DataInsufficient",
            "remedy": "Drop the constant column(s).",
            "symptom": "hill_climb: column(s) <flat> are constant, so the "
            "Gaussian score of a node with zero variance is "
            "not defined.",
        },
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Pass data_type='discrete' to treat every column as "
            "categorical, or encode / drop the categorical ones "
            "and pass data_type='gaussian'.",
            "symptom": "hill_climb: the columns mix categorical and "
            "numeric types (categorical: <[c for c, f in "
            "zip(cols, flags) if f]>).",
        },
    ],
    "hypotheses": [
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": 'E.g. sp.hypotheses(fit, "x1 = 0") or '
            "sp.hypotheses(fit, joint=['x1', 'x2']).",
            "symptom": "hypotheses: pass a hypothesis (or joint=True / "
            "joint=[names]).",
        }
    ],
    "icc": [
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Refit with a random intercept only.",
            "symptom": "icc() is defined for random-intercept models only; with "
            "random slopes on <x_random> the intraclass correlation "
            "depends on the covariate values.",
        },
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Use a Gaussian (sp.mixed) or logit (sp.melogit, "
            "sp.meologit) random-intercept model.",
            "symptom": "icc() is not defined for a <family> GLMM: there is no "
            "residual variance on the linear-predictor scale (Stata's "
            "estat icc is likewise unavailable after mepoisson / "
            "menbreg / meglm gamma).",
        },
    ],
    "influence_measures": [
        {
            "exception": "statspai.DataInsufficient",
            "remedy": "The fit needs more than k + 1 " "observations.",
            "symptom": "sp.influence_measures: too few "
            "observations to leave one out.",
        }
    ],
    "interactive": [
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Use ui='none' to edit through actions= or the "
            "editor's methods without any interface.",
            "symptom": "ui must be 'auto', 'jupyter', 'window' or " "'none', got <ui>",
        }
    ],
    "jackknife": [
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Pass e.g. lambda d: sp.regress('y ~ x', data=d).",
            "symptom": "sp.jackknife: statistic must be a function of a " "DataFrame.",
        },
        {
            "exception": "statspai.DataInsufficient",
            "remedy": "Check that the statistic runs on a sample with one "
            "row removed.",
            "symptom": "sp.jackknife: fewer than three replications " "succeeded.",
        },
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Drop those rows first.",
            "symptom": "sp.jackknife: the cluster variable has missing " "values.",
        },
    ],
    "johansen_lrtest": [
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Test one hypothesis per call.",
            "symptom": "johansen_lrtest: give exactly one of beta=, "
            "beta_known=, loading=.",
        },
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Choose the rank with sp.johansen first.",
            "symptom": "johansen_lrtest: rank must be between 1 and "
            "<K - 1>, got <rank>.",
        },
        {
            "exception": "statspai.DataInsufficient",
            "remedy": "Use fewer lags or a longer sample.",
            "symptom": "johansen_lrtest: <Y.shape[0]> observations "
            "are too few for <lags> lagged difference(s) "
            "of <K> series.",
        },
    ],
    "jwdid": [
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Pass method='ppmlhdfe' for count outcomes.",
            "symptom": "jwdid(method=<method>) is not supported; use one of "
            "None, 'regress', 'ppmlhdfe', 'poisson', 'logit'.",
        },
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "predict='xb' reports the ATT in log points.",
            "symptom": "jwdid(predict=<predict>) is not supported; use None, "
            "'mu' or 'xb'.",
        },
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Drop predict=, or pass method='ppmlhdfe'.",
            "symptom": "jwdid(predict=<predict>) needs a nonlinear method(); "
            "the linear model has a single scale.",
        },
    ],
    "ktau": [
        {
            "exception": "statspai.DataInsufficient",
            "remedy": "Check the columns.",
            "symptom": "ktau: one of the variables does not vary.",
        }
    ],
    "kwallis": [
        {
            "exception": "statspai.DataInsufficient",
            "remedy": "Check the variable.",
            "symptom": "kwallis: every value is the same.",
        }
    ],
    "ljungbox": [
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Pass the number of estimated ARMA parameters.",
            "symptom": "ljungbox: model_df cannot be negative.",
        },
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "E.g. lags=10.",
            "symptom": "ljungbox: lags must be positive.",
        },
        {
            "exception": "statspai.DataInsufficient",
            "remedy": "Lower lags=.",
            "symptom": "ljungbox: <top> lags need more than <n> " "observations.",
        },
        {
            "exception": "statspai.DataInsufficient",
            "remedy": "There is no autocorrelation to test.",
            "symptom": "ljungbox: the series is constant.",
        },
    ],
    "logit_gof": [
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "groups=10 is the usual choice.",
            "symptom": "sp.logit_gof: the Hosmer-Lemeshow test needs at "
            "least 3 groups.",
        }
    ],
    "loneway": [
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "alpha=0.05 gives a 95% interval.",
            "symptom": "loneway: alpha must lie strictly between 0 and 1, "
            "got <alpha>.",
        },
        {
            "exception": "statspai.DataInsufficient",
            "remedy": "Each group needs more than one observation overall.",
            "symptom": "loneway: at least two groups and one within-group "
            "degree of freedom are needed.",
        },
        {
            "exception": "statspai.DataInsufficient",
            "remedy": "Check that the outcome is not constant.",
            "symptom": "loneway: <y> takes a single value, so there is no "
            "variation to split between and within groups.",
        },
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Drop exact=True for the asymptotic interval.",
            "symptom": "loneway: the exact interval is defined for groups of "
            "equal size.",
        },
    ],
    "lrvar": [
        {
            "exception": "statspai.DataInsufficient",
            "remedy": "The long-run variance needs variation.",
            "symptom": "sp.lrvar: a series is constant.",
        },
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Use 'bartlett', 'parzen', 'qs', 'tukey-hanning' or "
            "'truncated'.",
            "symptom": "sp.lrvar: kernel=<kernel> is not known.",
        },
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Use 0 (none) or 1.",
            "symptom": "sp.lrvar: prewhite=<prewhite> is negative.",
        },
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Drop adjust=True or set demean=True.",
            "symptom": "sp.lrvar: adjust=True corrects for the estimated mean "
            "and needs demean=True.",
        },
    ],
    "ltmle_msm": [
        {
            "exception": "statspai.DataInsufficient",
            "remedy": "Use more data.",
            "symptom": "ltmle_msm: <n> rows are too few.",
        }
    ],
    "mmhc": [
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Use sp.hill_climb(required=...) to impose it without the "
            "restriction.",
            "symptom": "mmhc: the required arc (<a>, <b>) joins two variables "
            "the tests found separable.",
        }
    ],
    "mmpc": [
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Pass data_type='discrete' or 'gaussian'.",
            "symptom": "mmpc: the columns mix categorical and numeric types "
            "(categorical: <[c for c, f in zip(cols, flags) if "
            "f]>).",
        }
    ],
    "mmqreg": [
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Pass x='col' or a list of columns.",
            "symptom": "mmqreg: give at least one regressor.",
        },
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Pass e.g. quantile=0.25 (not 25).",
            "symptom": "mmqreg: quantile=<quantile> is not strictly between 0 "
            "and 1.",
        },
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Use 'gls', 'robust' or 'cluster'.",
            "symptom": "mmqreg: vce=<vce>.",
        },
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Pass vce='cluster', cluster='<column>'.",
            "symptom": "mmqreg: vce='cluster' and cluster= go together.",
        },
    ],
    "model_average": [
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Use 'jma', 'mma', 'aic' or 'bic'.",
            "symptom": "sp.model_average: method=<method> is not one "
            "of <', '.join(_METHODS)>.",
        },
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Pass a list of formulas with the same outcome.",
            "symptom": "sp.model_average: at least two candidate "
            "formulas are needed.",
        },
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Drop names= to use m1, m2, ...",
            "symptom": "sp.model_average: names= must give one "
            "distinct label per formula.",
        },
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Averaging compares fits of one outcome.",
            "symptom": "sp.model_average: the formulas have different "
            "outcomes (<outcomes>).",
        },
    ],
    "msprt": [
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "tau is the prior standard deviation of the effect in "
            "outcome units, e.g. the effect size you hope to detect.",
            "symptom": "tau must be a positive number; got <tau>.",
        },
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "E.g. alpha=0.05.",
            "symptom": "alpha must be in (0, 1); got <alpha>.",
        },
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Drop sigma= to use the running sample variances.",
            "symptom": "sigma must be a positive number; got <sigma>.",
        },
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Compare two arms at a time.",
            "symptom": "sp.msprt: <treat> must take exactly two values; found "
            "<len(levels)>.",
        },
    ],
    "ndiffs": [
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Use 0.10, 0.05, 0.025 or 0.01.",
            "symptom": "ndiffs: alpha=<alpha> has no tabulated KPSS critical " "value.",
        },
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Use max_d=2.",
            "symptom": "ndiffs: max_d cannot be negative.",
        },
    ],
    "nlcom": [
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "alpha is 1 minus the confidence level, e.g. 0.05.",
            "symptom": "nlcom: alpha must be in (0, 1), got <alpha>.",
        },
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Write coefficient names that are not plain identifiers "
            "as _b[name].",
            "symptom": "nlcom(<expression>) is not a valid expression " "(<exc.msg>).",
        },
    ],
    "noether_test": [
        {
            "exception": "statspai.DataInsufficient",
            "remedy": "The sign test has nothing to count.",
            "symptom": "noether_test: every pair difference is zero.",
        }
    ],
    "nsdiffs": [
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Pass the seasonal period, or use sp.ndiffs.",
            "symptom": "nsdiffs: non-seasonal data (period=1) have no "
            "seasonal difference.",
        }
    ],
    "oneway": [
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Use 'bonferroni', 'sidak' or 'scheffe'.",
            "symptom": "oneway: unknown compare <compare>.",
        },
        {
            "exception": "statspai.DataInsufficient",
            "remedy": "Each group needs more than one observation overall.",
            "symptom": "oneway: no within-group degrees of freedom.",
        },
    ],
    "panel_match": [
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Pass covariates=[...] or use refinement='none'.",
            "symptom": "refinement=<refinement> needs covariates",
        },
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Use se_method='bootstrap' with qoi='ate'.",
            "symptom": "the ATE has only the bootstrap standard error",
        },
        {
            "exception": "statspai.DataInsufficient",
            "remedy": "Shorten lag, set match_missing=True, drop "
            "exact_match / restrict_control_period, or use "
            "fewer leads.",
            "symptom": "no treated observation has a matched set",
        },
        {
            "exception": "statspai.DataInsufficient",
            "remedy": "Drop placebo_test or shorten lag.",
            "symptom": "no matched set observes the outcome over the "
            "whole history window",
        },
    ],
    "pca": [
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Pass the number of components to keep, e.g. 2.",
            "symptom": "sp.pca: n_components must be a whole number, got "
            "<n_components>.",
        },
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Ask for at most as many components as variables.",
            "symptom": "sp.pca: n_components=<n_components> is not between 1 and "
            "<p>.",
        },
    ],
    "periodogram": [
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Choose one of the three estimators.",
            "symptom": "sp.periodogram: method=<method> is not 'raw', "
            "'smoothed' or 'ar'.",
        },
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Use 'cycles' (R) or 'radians' (textbook).",
            "symptom": "sp.periodogram: scale=<scale> is not 'cycles' or " "'radians'.",
        },
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Pass e.g. alpha=0.05.",
            "symptom": "sp.periodogram: alpha=<alpha> is not between 0 " "and 1.",
        },
        {
            "exception": "statspai.DataInsufficient",
            "remedy": "A spectrum needs at least four observations.",
            "symptom": "sp.periodogram: <n> observations.",
        },
    ],
    "power_ttest": [
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Use 'two-sample', 'one-sample' or 'paired'.",
            "symptom": "power_ttest: unknown type <type>.",
        },
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Use 'two-sided', 'greater' or 'less'.",
            "symptom": "power_ttest: unknown alternative <alternative>.",
        },
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "e.g. sp.power_ttest(delta=0.5, power=0.8)",
            "symptom": "power_ttest: give exactly two of n, delta and "
            "power; the third is computed (<given> given).",
        },
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "A test has power alpha when the effect is zero.",
            "symptom": "power_ttest: power must lie between alpha " "(<alpha>) and 1.",
        },
    ],
    "prais": [
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Use method='prais' (Prais-Winsten) or 'corc'.",
            "symptom": "sp.prais: method=<method> is not 'prais' or 'corc'.",
        },
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Use rhotype='regress' (the default).",
            "symptom": "sp.prais: rhotype=<rhotype> is not one of <', "
            "'.join(_RHOTYPES)>.",
        },
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Use vce='ols' or vce='robust'.",
            "symptom": "sp.prais: vce=<vce> is not 'ols', 'robust', 'hc2' or " "'hc3'.",
        },
        {
            "exception": "statspai.DataInsufficient",
            "remedy": "Drop or fill the incomplete periods so that the rows "
            "are consecutive.",
            "symptom": "sp.prais: <len(data) - n> row(s) have a missing value "
            "in the model variables, which leaves a gap in the "
            "series.",
        },
    ],
    "prtest": [
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Check the variable.",
            "symptom": "prtest: the standard error under the null is zero "
            "(every observation is the same), so the statistic is "
            "not defined.",
        },
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "sp.prtest(n=50, proportion=0.52, p=0.4)",
            "symptom": "prtest: y / by / other name columns of data; pass "
            "data, or the summary statistics n= and proportion=.",
        },
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "sp.prtest(n=50, proportion=0.52, p=0.4)",
            "symptom": "prtest: give data and y, or both n= and proportion=.",
        },
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Pass scalars for one sample, pairs for two.",
            "symptom": "prtest: n and proportion must describe the same one "
            "or two samples.",
        },
    ],
    "pscore": [
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Pass the DataFrame holding the treatment and " "covariates.",
            "symptom": "pscore: data must be a pandas DataFrame.",
        },
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Pass the covariates of the treatment model.",
            "symptom": "pscore: at least one covariate is needed.",
        },
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Use ps_model='logit' or ps_model='probit'.",
            "symptom": "pscore: ps_model must be 'logit' or 'probit', got "
            "<ps_model>.",
        },
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "The default is 0.01.",
            "symptom": "pscore: level must lie strictly between 0 and 1, got "
            "<level>.",
        },
    ],
    "ranksum": [
        {
            "exception": "statspai.DataInsufficient",
            "remedy": "Check the variable.",
            "symptom": "ranksum: every value is the same, so the ranks carry "
            "no information.",
        }
    ],
    "rd2d_plot": [
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Pass a DataFrame containing y, x1, x2, and " "treatment.",
            "symptom": "`data` must be a pandas DataFrame.",
        },
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Pass boundary=lambda x1: f(x1), or leave " "boundary=None.",
            "symptom": "boundary must be callable or None",
        },
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Use plot_type='scatter', 'heatmap', or " "'boundary_effects'.",
            "symptom": "plot_type must be 'scatter', 'heatmap', or "
            "'boundary_effects', got '<plot_type>'",
        },
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Re-estimate with rd2d(..., eval_points=...) or " "n_eval > 1.",
            "symptom": "plot_type='boundary_effects' requires a result "
            "from rd2d() with boundary eval points "
            "(approach='location' or 'distance').",
        },
    ],
    "recommend": [
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Pass id= and time= (the unit and period columns).",
            "symptom": "recommend: the data look like a panel but more "
            "than one (unit, time) pair fits: <layouts>. "
            "Recommending without knowing which would treat a "
            "panel as a cross-section.",
        }
    ],
    "refresh_legend": [
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Pass one index per legend entry (None to skip "
            "an entry), or omit `sources` to match entries "
            "to artists by label.",
            "symptom": "refresh_legend: <len(sources)> sources for "
            "<len(entries)> legend entries",
        },
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Indices refer to "
            "ax.get_legend_handles_labels()[0]; use None "
            "for entries drawn from proxy artists.",
            "symptom": "refresh_legend: source <src> for entry <k> is "
            "not an index into the <len(handles)> labelled "
            "artists",
        },
    ],
    "reset_spec_counter": [
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Pass the outcome's column name, or None to "
            "reset every outcome.",
            "symptom": "reset_spec_counter(outcome=<outcome>): "
            "outcome must be a column name (str) or "
            "None.",
        }
    ],
    "ridge": [
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Add at least one regressor to the formula.",
            "symptom": "ridge: the model has no regressors to penalise.",
        },
        {
            "exception": "statspai.DataInsufficient",
            "remedy": "Check for missing values in the outcome and regressors.",
            "symptom": "ridge: <n> complete observation(s); at least 2 are " "needed.",
        },
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Drop the constant column(s).",
            "symptom": "ridge: regressor(s) <bad> do not vary.",
        },
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Use strictly positive penalties.",
            "symptom": "ridge: lambda = 0 is least squares, which is not "
            "unique with at least as many regressors as "
            "observations.",
        },
    ],
    "rmst": [
        {
            "exception": "statspai.DataInsufficient",
            "remedy": "Check the column names.",
            "symptom": "rmst: columns not found: <missing>",
        },
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Check the weight column.",
            "symptom": "rmst: weights must be finite and non-negative.",
        },
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Use se_method='ij' or 'greenwood'.",
            "symptom": "rmst: se_method='xie_liu' is a variance of the curve, "
            "not of its area.",
        },
    ],
    "robvar": [
        {
            "exception": "statspai.DataInsufficient",
            "remedy": "Drop or merge the small group.",
            "symptom": "robvar: a group has fewer than two observations.",
        },
        {
            "exception": "statspai.DataInsufficient",
            "remedy": "Check that the outcome is not constant.",
            "symptom": "robvar: <y> takes a single value within every group, "
            "so there are no deviations whose spread could be "
            "compared.",
        },
    ],
    "rolling": [
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Use 'nonrobust' or 'hc1'.",
            "symptom": "rolling: vce must be one of <_VCE>, got <vce>.",
        },
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Drop the redundant term from the formula.",
            "symptom": "rolling: the regressors are collinear in the full " "sample.",
        },
        {
            "exception": "statspai.DataInsufficient",
            "remedy": "Use a shorter window.",
            "symptom": "rolling: window=<window> exceeds the <n> usable "
            "observations.",
        },
    ],
    "rosenbaum_stratified": [
        {
            "exception": "statspai.DataInsufficient",
            "remedy": "Coarsen the strata.",
            "symptom": "rosenbaum_stratified: no stratum "
            "contains both a treated individual and "
            "a control.",
        }
    ],
    "scdata": [
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Aggregate or drop the repeated unit-period rows first.",
            "symptom": "Duplicate (unit, time) rows in data.",
        },
        {
            "exception": "statspai.DataInsufficient",
            "remedy": "Use a later treatment_time or a longer panel.",
            "symptom": "Need at least 2 pre-treatment periods.",
        },
        {
            "exception": "statspai.DataInsufficient",
            "remedy": "treatment_time must not be after the last period.",
            "symptom": "Need at least 1 post-treatment period.",
        },
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Impute or drop the periods where the treated unit's "
            "outcome is missing.",
            "symptom": "The treated unit has missing outcome values.",
        },
    ],
    "sdtest": [
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Drop sd0=, or drop by= / other=.",
            "symptom": "sdtest: sd0= is the null of a one-sample test; two "
            "samples are tested against a ratio of one.",
        },
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Check that the variable is not constant.",
            "symptom": "sdtest: <label> has no variation, so the test is not "
            "defined.",
        },
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "sp.sdtest(df, 'y', sd0=5); for two samples pass by= or "
            "other=.",
            "symptom": "sdtest: a one-sample test needs sd0=, the positive "
            "standard deviation under the null.",
        },
    ],
    "seasonal_dummies": [
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "E.g. sp.seasonal_dummies(len(y), 4).",
            "symptom": "seasonal_dummies: period must be at least 2 "
            "and n at least 1.",
        }
    ],
    "shrinkage": [
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Drop them; a constant cannot be standardised.",
            "symptom": "shrinkage: predictors <constant> do not vary.",
        },
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Use shuffle=True, fewer folds, or drop predictors "
            "that vary in a few rows only.",
            "symptom": "shrinkage: a predictor is constant within a "
            "training fold, so it cannot be standardised "
            "there.",
        },
    ],
    "signrank": [
        {
            "exception": "statspai.DataInsufficient",
            "remedy": "Check the columns.",
            "symptom": "signrank: fewer than two pairs.",
        },
        {
            "exception": "statspai.DataInsufficient",
            "remedy": "Check the columns.",
            "symptom": "signrank: every difference is zero.",
        },
    ],
    "simple_forecast": [
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Use 'naive', 'snaive', 'drift' or 'mean'.",
            "symptom": "simple_forecast: method=<method> is not a " "benchmark method.",
        },
        {
            "exception": "statspai.DataInsufficient",
            "remedy": "Provide a longer series.",
            "symptom": "simple_forecast: the mean method needs at "
            "least 2 observations.",
        },
        {
            "exception": "statspai.DataInsufficient",
            "remedy": "Provide a longer series.",
            "symptom": "simple_forecast: the naive method needs at "
            "least 2 observations.",
        },
        {
            "exception": "statspai.DataInsufficient",
            "remedy": "Provide a longer series.",
            "symptom": "simple_forecast: the drift method needs at "
            "least 3 observations.",
        },
    ],
    "sktest": [
        {
            "exception": "statspai.DataInsufficient",
            "remedy": "Check the variable.",
            "symptom": "sktest: <name> needs at least 8 observations that are "
            "not all equal.",
        }
    ],
    "stl": [
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Non-seasonal data have no seasonal component to remove.",
            "symptom": "stl: the seasonal period must be at least 2.",
        },
        {
            "exception": "statspai.DataInsufficient",
            "remedy": "At least two full cycles are needed.",
            "symptom": "stl: <n> observations are fewer than two cycles of "
            "period <m>.",
        },
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Give one window per period, or a single one.",
            "symptom": "stl: <len(windows)> seasonal windows for <len(periods)> "
            "periods.",
        },
    ],
    "stochvol": [
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Pass data=<DataFrame>, or pass y as an array.",
            "symptom": "y=<y> names a column, but data= was not given.",
        }
    ],
    "stteffects": [
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Use 'ra', 'wra', 'ipw' or 'ipwra'.",
            "symptom": "stteffects: method must be one of "
            "<list(METHODS)>; got <method>.",
        },
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Use estimand='ATE' or 'ATET'.",
            "symptom": "stteffects: estimand must be 'ATE' or 'ATET'; got "
            "<estimand>.",
        },
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Use treat_model='logit'.",
            "symptom": "stteffects: treat_model must be 'logit' or "
            "'probit'; got <treat_model>.",
        },
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Drop censor_covariates=, or use method='wra' / " "'ipwra'.",
            "symptom": "stteffects: method='ra' handles censoring through "
            "the outcome likelihood and takes no censoring "
            "model.",
        },
    ],
    "suest": [
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Copy the column under a new name for a second equation.",
            "symptom": "Each equation needs a distinct dependent variable.",
        }
    ],
    "sumstats": [
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Use 'linear' or 'stata'.",
            "symptom": "sumstats: unknown percentile_method " "<percentile_method>.",
        },
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Analytic weights must be zero or positive.",
            "symptom": "sumstats: negative weights.",
        },
    ],
    "support_points": [
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Ask for fewer points, or use "
            "sp.split_data(method='twinning') to pick "
            "representative rows of discrete data.",
            "symptom": "support_points: the iteration broke down "
            "because several points fell on the same "
            "location. The sample has too few distinct "
            "values for <n> points.",
        }
    ],
    "svar": [
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "fit = sp.var(df, variables=[...], lags=p); sp.svar(fit, " "...)",
            "symptom": "svar: the first argument is a VAR fitted by sp.var.",
        },
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "alpha is 1 minus the coverage of the bands.",
            "symptom": "svar: alpha must be in (0, 1), got <alpha>.",
        },
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "A recursive identification is B=np.tril(np.full((K, K), "
            "np.nan)).",
            "symptom": "svar: give exactly one kind of restriction: A= / B= "
            "(short-run), long_run=, sign= or proxy=.",
        },
        {
            "exception": "statspai.DataInsufficient",
            "remedy": "Use a longer sample or fewer variables.",
            "symptom": "svar: too few observations for the residual " "covariance.",
        },
    ],
    "swilk": [
        {
            "exception": "statspai.DataInsufficient",
            "remedy": "Set them to missing (they are then dropped) or remove "
            "those rows.",
            "symptom": "swilk: <name> holds infinite values, for which the "
            "statistic is not defined.",
        },
        {
            "exception": "statspai.DataInsufficient",
            "remedy": "For a larger sample use sp.sktest.",
            "symptom": "swilk: <name> has <n> usable observations; the test "
            "covers 4 to 5,000 observations that are not all "
            "equal.",
        },
    ],
    "switchback": [
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Analyse a stretch of consecutive complete periods.",
            "symptom": "switchback: the outcome and the assignment must "
            "be observed in every period; a missing period "
            "changes the windows around it.",
        },
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Pass the design that generated the assignment: "
            "design= takes a boolean column marking the periods "
            "where the coin was flipped.",
            "symptom": "switchback: the assignment changes at period(s) "
            "<(off_design[:5] + 1).tolist()>, which are not "
            "randomization points of the stated design.",
        },
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Pass a bound that the outcome cannot exceed.",
            "symptom": "switchback: outcome_bound=<outcome_bound> is "
            "below the largest observed |<y>| (<largest>); it "
            "must bound every outcome under every "
            "assignment.",
        },
    ],
    "tighten_blocks": [
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Use ratio=1, or drop subset_cost.",
            "symptom": "<caller>: subset_cost needs ratio=1.",
        },
        {
            "alternative": "sp.two_criteria_match",
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Build the blocks first, for example with "
            "sp.two_criteria_match.",
            "symptom": "<caller>: every block must hold exactly one "
            "treated individual and at least one control.",
        },
        {
            "exception": "statspai.DataInsufficient",
            "remedy": "Lower ratio.",
            "symptom": "<caller>: some blocks have fewer than <ratio> " "controls.",
        },
    ],
    "ts_features": [
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Use scale=True.",
            "symptom": "ts_features: scale must be True or False.",
        },
        {
            "exception": "statspai.DataInsufficient",
            "remedy": "At least 12 observations are needed.",
            "symptom": "ts_features: <values.shape[0]> "
            "observations<label> are too few.",
        },
        {
            "exception": "statspai.DataInsufficient",
            "remedy": "A constant series has no features to compute.",
            "symptom": "ts_features: the series<label> is constant.",
        },
    ],
    "tscv": [
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Pass positive integers.",
            "symptom": "tscv: horizon and step must be at least 1.",
        },
        {
            "exception": "statspai.DataInsufficient",
            "remedy": "Lower initial= (or window=) or use a longer series.",
            "symptom": "tscv: the first training window (<first> observations) "
            "leaves nothing to forecast in a series of <n>.",
        },
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Use window=None for an expanding window.",
            "symptom": "tscv: window must be at least 2.",
        },
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "E.g. forecaster='ets' or lambda train, h: ...",
            "symptom": "tscv: forecaster must be a method name or a function.",
        },
    ],
    "tsfilter": [
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Use 'hp', 'bk', 'cf', 'bw' or 'hamilton'.",
            "symptom": "sp.tsfilter: method=<method> is not one of " "<_METHODS>.",
        },
        {
            "exception": "statspai.DataInsufficient",
            "remedy": "Check the column.",
            "symptom": "sp.tsfilter: the series has no observed value.",
        },
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Fill the gap or filter the longest complete stretch.",
            "symptom": "sp.tsfilter: the series has missing or infinite "
            "values between observed ones; the filters need "
            "consecutive periods.",
        },
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "method='hamilton' is one-sided by construction.",
            "symptom": "sp.tsfilter: one_sided=True is defined for "
            "method='hp', not <method>.",
        },
    ],
    "tune_causal_forest": [
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Drop them from parameters= or from the " "keywords.",
            "symptom": "tune_causal_forest: <clash> are both "
            "tuned and fixed by keyword.",
        }
    ],
    "two_stage_interference": [
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Use weighting='group' or 'individual'.",
            "symptom": "two_stage_interference: " "weighting=<weighting>.",
        },
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Pass e.g. alpha=0.05.",
            "symptom": "two_stage_interference: alpha=<alpha> " "is not in (0, 1).",
        },
        {
            "exception": "statspai.DataInsufficient",
            "remedy": "The cross-saturation effects need "
            "groups at two or more saturations; "
            "with one, use "
            "sp.difference_in_means(cluster=).",
            "symptom": "two_stage_interference: only one " "saturation in the data.",
        },
        {
            "exception": "statspai.DataInsufficient",
            "remedy": "Pool or drop the saturation levels " "with a single group.",
            "symptom": "two_stage_interference: fewer than "
            "two groups at saturation(s) "
            "<thin.index.tolist()>; the "
            "group-level variance needs two.",
        },
    ],
    "unitroot": [
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Use test='adf', 'dfgls', 'pp' or 'kpss'.",
            "symptom": "unitroot: unknown test <test>.",
        },
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Use 'c' (constant), 'ct' (constant and trend) or 'n' " "(none).",
            "symptom": "unitroot: unknown trend <trend>.",
        },
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Use test='adf', trend='n', or trend='c'.",
            "symptom": "unitroot: DF-GLS removes a constant or a trend by "
            "GLS; there is no trend='n' version.",
        },
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Use 0.01, 0.05 or 0.10.",
            "symptom": "unitroot: alpha must be in (0, 1), got <alpha>.",
        },
    ],
    "varsoc": [
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Name the endogenous series.",
            "symptom": "sp.varsoc: no variables given.",
        },
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Use trend='c'.",
            "symptom": "sp.varsoc: trend=<trend> is not 'c', 'ct' or 'n'.",
        },
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Check the names passed to exog=.",
            "symptom": "sp.varsoc: exog=<exog_names> must name columns that "
            "are not among the endogenous variables.",
        },
        {
            "exception": "statspai.DataInsufficient",
            "remedy": "Lower maxlag.",
            "symptom": "sp.varsoc: <n> observations cannot support "
            "maxlag=<maxlag> with <k> variable(s).",
        },
    ],
    "vec": [
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Use trend='c' for an unrestricted constant.",
            "symptom": "sp.vec: trend=<trend> is not one of 'c', 'rc', 'n', "
            "'rt', 'ct' (or Stata's constant / rconstant / none / "
            "rtrend / trend).",
        },
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Name the series of the system.",
            "symptom": "sp.vec needs at least two series.",
        },
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "rank = K means the levels are stationary (fit sp.var); "
            "rank = 0 means no cointegration (fit a VAR in "
            "differences).",
            "symptom": "sp.vec: lags=<lags>, rank=<rank> with <k> series; need "
            "lags >= 0 and 1 <= rank <= <k - 1>.",
        },
        {
            "exception": "statspai.DataInsufficient",
            "remedy": "Use fewer lags or a longer sample.",
            "symptom": "sp.vec: <n> observations are too few for <lags> lagged "
            "difference(s) of <k> series.",
        },
    ],
    "weighted_rank_power": [
        {
            "exception": "statspai.DataInsufficient",
            "remedy": "Collect more blocks.",
            "symptom": "weighted_rank_power: the jackknife needs "
            "at least three blocks.",
        }
    ],
    "westfall_young": [
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Drop robust=False to use CR1 by cluster, or "
            "drop cluster= for the classical t.",
            "symptom": "sp.westfall_young: robust=False (classical "
            "OLS standard errors) cannot be combined with "
            "cluster=.",
        }
    ],
    "xcorr": [
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Use prewhiten='ar' to filter each series with its own "
            "autoregression.",
            "symptom": "sp.xcorr: prewhiten=<prewhiten> is not None, 'ar' or " "'x'.",
        },
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Use ar_method='ols'.",
            "symptom": "sp.xcorr: ar_method=<ar_method> is not 'ols' or 'yw'.",
        },
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Use ic='aic' or ic='bic'.",
            "symptom": "sp.xcorr: ic=<ic> is not 'aic' or 'bic'.",
        },
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Use alpha=0.05 for a 95% band.",
            "symptom": "sp.xcorr: alpha=<alpha> is not between 0 and 1.",
        },
    ],
    "xtevent": [
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Fit them separately.",
            "symptom": "xtevent: trend= and proxy= are not combined here.",
        },
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Drop engine='reghdfe' / absorb= for the proxy " "estimator.",
            "symptom": "xtevent: proxy= is fitted as Stata xtivreg, fe "
            "(engine='areg').",
        },
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "The policy must vary within units over and above the "
            "common period shocks; check the policy column.",
            "symptom": "xtevent: the policy <policy> is collinear with the "
            "unit and period effects (and controls); its effect "
            "is not identified.",
        },
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "E.g. proxy_iv=1, proxy_iv=[1, 2] or proxy_iv=['x2'].",
            "symptom": "xtevent: proxy_iv must be 'select', a lead (int), a "
            "variable name, or a list of them (got "
            "<type(proxy_iv).__name__>).",
        },
    ],
    "xtevent_smpath": [
        {
            "exception": "statspai.DataInsufficient",
            "remedy": "Raise maxorder (at most 10) or lower level.",
            "symptom": "No polynomial of order up to <maxorder> lies "
            "in the <level> Wald region of the event-time "
            "coefficients.",
        },
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Stata's default is maxorder(10).",
            "symptom": "maxorder must be between 1 and 10.",
        },
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Use level=0.95 for a 95% Wald region.",
            "symptom": "level must lie strictly between 0 and 1.",
        },
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Omit it to use every post-event coefficient.",
            "symptom": "postwindow must be a positive integer.",
        },
    ],
    "xtoverid": [
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "The test compares within and between variation.",
            "symptom": "sp.xtoverid: no regressor varies within panels.",
        },
        {
            "exception": "statspai.DataInsufficient",
            "remedy": "Cluster on a finer level (the default is the panel "
            "identifier), or test fewer regressors.",
            "symptom": "sp.xtoverid: <G> cluster(s) cannot test "
            "<len(varying)> restriction(s): the cluster-robust "
            "covariance has rank at most <G - 1>, so the "
            "statistic would divide by zero.",
        },
    ],
    "xtserial": [
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Convert dates to a period index first.",
            "symptom": "sp.xtserial: time=<time> must count periods " "(integers).",
        },
        {
            "exception": "statspai.DataInsufficient",
            "remedy": "The test needs at least three consecutive periods " "per panel.",
            "symptom": "sp.xtserial: too few consecutive observations to "
            "difference.",
        },
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "The first-difference regression needs at least one "
            "time-varying regressor.",
            "symptom": "sp.xtserial: no regressor changes within panels.",
        },
        {
            "exception": "statspai.DataInsufficient",
            "remedy": "The test cannot be computed on this panel.",
            "symptom": "sp.xtserial: no panel has three consecutive " "periods.",
        },
    ],
    "xtsum": [
        {
            "exception": "statspai.DataInsufficient",
            "remedy": "Drop the variable.",
            "symptom": "sp.xtsum: <name> has no observed value.",
        }
    ],
    "zivot_andrews": [
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "'intercept' allows a level shift, 'trend' a "
            "change of slope, 'both' the two together.",
            "symptom": "zivot_andrews: model=<model> is not "
            "'intercept', 'trend' or 'both'.",
        },
        {
            "exception": "statspai.DataInsufficient",
            "remedy": "Use fewer lags or a longer series.",
            "symptom": "zivot_andrews: <n> observations are too few "
            "for <k_edge> lags and trim=<trim>.",
        },
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "sp.zivot_andrews(df, 'gdp').",
            "symptom": "zivot_andrews: with a DataFrame, y must name "
            "one of its columns.",
        },
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Fill or drop the gaps; the test needs a "
            "regularly spaced series.",
            "symptom": "zivot_andrews: the series has missing values "
            "inside the sample.",
        },
    ],
    "ztest": [
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "sd is the known population standard deviation.",
            "symptom": "ztest: sd must be a positive number, or one per " "sample.",
        },
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "sp.ztest(n=10, mean=88, sd=0.7, mu=85).",
            "symptom": "ztest: mean= is required with summary statistics.",
        },
    ],
}
