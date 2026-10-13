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
            "remedy": "Check the column names.",
            "symptom": "ardl: column(s) <missing_cols> are not in data.",
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
    "balance_vs_randomization": [
        {
            "exception": "statspai.DataInsufficient",
            "remedy": "Check the treat column.",
            "symptom": "balance_vs_randomization: need at "
            "least two treated and two control "
            "individuals.",
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
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Check the name.",
            "symptom": "bds: column <y> not found.",
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
            "remedy": "Check the column names.",
            "symptom": "chow_test: column(s) <missing> not found in data.",
        },
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
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Check the protected column name.",
            "symptom": "`protected` column <protected> " "not in data.",
        },
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Check admissible feature names " "against data.columns.",
            "symptom": "admissible_features not in data: " "<missing>",
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
            "remedy": "Use estimand='ATT'.",
            "symptom": "full_match: estimand must be ATT, ATC or ATE, got "
            "<estimand>.",
        },
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Use 'propensity', 'logit', 'mahalanobis' or " "'euclidean'.",
            "symptom": "full_match: unknown distance <distance>.",
        },
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Use caliper_scale='sd'.",
            "symptom": "full_match: caliper_scale must be 'sd' or " "'raw'.",
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
            "remedy": "Check spec, time= and value=.",
            "symptom": "hierarchy: columns <absent> are not in data.",
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
    ],
    "hill_climb": [
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Pass data_type='discrete' to treat every column as "
            "categorical, or encode / drop the categorical ones "
            "and pass data_type='gaussian'.",
            "symptom": "hill_climb: the columns mix categorical and "
            "numeric types (categorical: <[c for (c, f) in "
            "zip(cols, flags) if f]>).",
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
    "jackknife": [
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Pass e.g. lambda d: sp.regress('y ~ x', data=d).",
            "symptom": "sp.jackknife: statistic must be a function of a " "DataFrame.",
        },
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Pass the name of the cluster variable.",
            "symptom": "sp.jackknife: cluster=<cluster> is not a column.",
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
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Drop exact=True for the asymptotic interval.",
            "symptom": "loneway: the exact interval is defined for groups of "
            "equal size.",
        },
    ],
    "lrvar": [
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
        {
            "exception": "statspai.DataInsufficient",
            "remedy": "Use a longer series.",
            "symptom": "sp.lrvar: <n> observations are too few.",
        },
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
            "(categorical: <[c for (c, f) in zip(cols, flags) if "
            "f]>).",
        }
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
    "pca": [
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Ask for at most as many components as variables.",
            "symptom": "sp.pca: n_components=<n_components> is not between 1 and "
            "<p>.",
        }
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
            "remedy": "Check the treat and covariates column names.",
            "symptom": "pscore: columns not found in data: <missing>",
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
            "remedy": "Check y/x1/x2/treatment names against data.columns.",
            "symptom": "Column '<col>' not found in data",
        },
    ],
    "ridge": [
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
    "robvar": [
        {
            "exception": "statspai.DataInsufficient",
            "remedy": "Drop or merge the small group.",
            "symptom": "robvar: a group has fewer than two observations.",
        }
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
            "remedy": "Pass the name of a numeric column.",
            "symptom": "sumstats: the weights column <weights> is not in " "the data.",
        },
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Analytic weights must be zero or positive.",
            "symptom": "sumstats: negative weights.",
        },
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
            "(short-run), long_run= or sign=.",
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
            "remedy": "For a larger sample use sp.sktest.",
            "symptom": "swilk: <name> has <n> usable observations; the test "
            "covers 4 to 5,000 observations that are not all "
            "equal.",
        }
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
            "exception": "statspai.DataInsufficient",
            "remedy": "At least 12 observations are needed.",
            "symptom": "ts_features: <values.shape[0]> observations are " "too few.",
        },
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Use scale=True.",
            "symptom": "ts_features: scale must be True or False.",
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
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Reorder `variables` so that a series that belongs to the "
            "long-run relation comes first.",
            "symptom": "sp.vec: the cointegrating relation cannot be normalised "
            "on the first <rank> variable(s); they do not enter it.",
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
    "xtoverid": [
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "The test compares within and between variation.",
            "symptom": "sp.xtoverid: no regressor varies within panels.",
        }
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
            "exception": "statspai.DataInsufficient",
            "remedy": "The test cannot be computed on this panel.",
            "symptom": "sp.xtserial: no panel has three consecutive " "periods.",
        },
    ],
    "xtsum": [
        {
            "exception": "statspai.MethodIncompatibility",
            "remedy": "Pass the panel identifier.",
            "symptom": "sp.xtsum: id=<id> is not a column.",
        },
        {
            "exception": "statspai.DataInsufficient",
            "remedy": "Drop the variable.",
            "symptom": "sp.xtsum: <name> has no observed value.",
        },
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
