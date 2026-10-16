"""Reference solutions for the non-StatsPAI toolsets (a few tasks each).

They exist to exercise the pythonic and R sessions end to end in the
harness tests; they are not a complete solution set.
"""

PYTHONIC = {
    "L1-01": (
        "ols",
        "import json, pandas as pd, statsmodels.formula.api as smf\n"
        "d = pd.read_csv('card.csv')\n"
        "m = smf.ols('lwage ~ educ + exper + expersq + black + south + smsa', d).fit(cov_type='HC1')\n"
        "print(json.dumps({'beta_educ': m.params['educ'], 'se_educ': m.bse['educ']}))",
    ),
    "L1-10": (
        "coxph",
        "import json, pandas as pd, statsmodels.api as sm\n"
        "d = pd.read_csv('synth_survival.csv')\n"
        "m = sm.PHReg(d['time'], d[['x1','x2','treat']], status=d['event'], ties='efron').fit()\n"
        "b = dict(zip(['x1','x2','treat'], m.params))\n"
        "print(json.dumps({'beta_x1': b['x1'], 'beta_x2': b['x2'], 'beta_treat': b['treat']}))",
    ),
}

R_SOLUTIONS = {
    "L1-01": (
        "ols",
        "suppressPackageStartupMessages({library(sandwich); library(jsonlite)})\n"
        "d <- read.csv('card.csv')\n"
        "m <- lm(lwage ~ educ + exper + expersq + black + south + smsa, d)\n"
        "cat(toJSON(list(beta_educ = unname(coef(m)['educ']), "
        "se_educ = unname(sqrt(diag(vcovHC(m, 'HC1')))['educ'])), auto_unbox = TRUE, digits = NA), '\\n')",
    ),
    "L1-16": (
        "evalue",
        "suppressPackageStartupMessages({library(EValue); library(jsonlite)})\n"
        "e <- evalues.RR(2.5, 1.8, 3.2)\n"
        "cat(toJSON(list(evalue_estimate = e['E-values','point'], evalue_ci = e['E-values','lower']), "
        "auto_unbox = TRUE, digits = NA), '\\n')",
    ),
}
