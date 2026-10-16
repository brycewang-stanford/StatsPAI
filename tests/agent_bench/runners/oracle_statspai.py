"""Reference StatsPAI solutions to the 50 tasks.

Two uses:

* a scripted "agent" for the harness tests (``models.ScriptedModel`` replays
  these as ``run_python`` calls followed by ``submit_answer``), so the whole
  loop -- sandbox, tools, grader -- is exercised without a language model;
* a cross-check of StatsPAI against the R golds: an L1 task names its
  method, so a StatsPAI number outside the gold tolerance is either a
  documented convention difference or a bug.

Each entry is the code a competent agent would run; it must print a JSON
dict of the task's report fields on its last line.
"""

from __future__ import annotations

from typing import Dict, Tuple

H = (
    "import json, warnings\nimport numpy as np, pandas as pd\n"
    "import statspai as sp\nwarnings.simplefilter('ignore')\n"
)
CARD_X = "exper + expersq + black + south + smsa"
MP = "mp = pd.read_csv('mpdta.csv')\n"
CS = (
    MP + "cs = sp.callaway_santanna(mp, y='lemp', g='first_treat', t='year', "
    "i='countyreal', estimator='dr', control_group='nevertreated')\n"
)
D_COL = "mp['D'] = ((mp.first_treat > 0) & (mp.year >= mp.first_treat)).astype(int)\n"

#: id -> (estimator key, code)
SOLUTIONS: Dict[str, Tuple[str, str]] = {
    "L1-01": (
        "ols",
        f"d = pd.read_csv('card.csv')\nr = sp.regress('lwage ~ educ + {CARD_X}', data=d, vce='hc1')\n"
        "print(json.dumps({'beta_educ': r.params['educ'], 'se_educ': r.std_errors['educ']}))",
    ),
    "L1-02": (
        "iv_2sls",
        f"d = pd.read_csv('card.csv')\nr = sp.ivreg('lwage ~ (educ ~ nearc4) + {CARD_X}', data=d, vce='hc1')\n"
        "print(json.dumps({'beta_educ': r.params['educ'], 'se_educ': r.std_errors['educ']}))",
    ),
    "L1-03": (
        "callaway_santanna",
        CS + "print(json.dumps({'att': cs.estimate, 'se': cs.se}))",
    ),
    "L1-04": (
        "sun_abraham",
        MP
        + "r = sp.sun_abraham(mp, y='lemp', g='first_treat', t='year', i='countyreal')\n"
        "print(json.dumps({'att': r.estimate}))",
    ),
    "L1-05": (
        "rdrobust",
        "d = pd.read_csv('lee.csv')\nr = sp.rdrobust(d, y='voteshare_next', x='margin')\n"
        "c, rb = r.model_info['conventional'], r.model_info['robust']\n"
        "print(json.dumps({'tau_conventional': c['estimate'], 'tau_bias_corrected': rb['estimate'], 'se_robust': rb['se']}))",
    ),
    "L1-06": (
        "rddensity",
        "d = pd.read_csv('lee.csv')\nr = sp.rddensity(d, x='margin')\n"
        "print(json.dumps({'p_value': r.pvalue}))",
    ),
    "L1-07": (
        "synth",
        "d = pd.read_csv('basque.csv')\n"
        "tr = d.loc[d.treated == 1, 'region'].iloc[0]\n"
        "r = sp.synth(d, outcome='gdppc', unit='region', time='year', treated_unit=tr, treatment_time=1970)\n"
        "print(json.dumps({'avg_post_gap': r.estimate}))",
    ),
    "L1-08": (
        "psm",
        "d = pd.read_csv('nsw_dw.csv')\n"
        "r = sp.match(d, y='re78', treat='treat', covariates=['age','education','black','hispanic','married','re74','re75'], "
        "method='nearest', estimand='ATT', replace=True)\n"
        "print(json.dumps({'att': r.estimate}))",
    ),
    "L1-09": (
        "dml_plr",
        "d = pd.read_csv('card.csv')\n"
        "r = sp.dml(d, y='lwage', treat='educ', covariates=['exper','expersq','black','south','smsa'], "
        "model='plr', ml_g='linear', ml_m='linear', n_folds=5)\n"
        "print(json.dumps({'theta': r.estimate, 'se': r.se}))",
    ),
    "L1-10": (
        "coxph",
        "d = pd.read_csv('synth_survival.csv')\n"
        "r = sp.cox(data=d, duration='time', event='event', x=['x1','x2','treat'], hazard_ratio=False)\n"
        "print(json.dumps({'beta_x1': r.params['x1'], 'beta_x2': r.params['x2'], 'beta_treat': r.params['treat']}))",
    ),
    "L1-11": (
        "honest_did",
        MP
        + "cs = sp.callaway_santanna(mp, y='lemp', g='first_treat', t='year', i='countyreal', "
        "estimator='dr', control_group='nevertreated', base_period='universal')\n"
        "h = sp.honest_did(cs, e=0, m_grid=[0.05], method='smoothness')\n"
        "row = h.iloc[0]\n"
        "print(json.dumps({'ci_lower': float(row['ci_lower']), 'ci_upper': float(row['ci_upper'])}))",
    ),
    "L1-12": (
        "bacon",
        MP
        + D_COL
        + "r = sp.bacon_decomposition(mp, y='lemp', treat='D', time='year', id='countyreal')\n"
        "tab = r['decomposition'] if isinstance(r, dict) else r.decomposition\n"
        "lve = tab.loc[tab['type'].str.startswith('Later'), 'weight'].sum()\n"
        "beta = r['beta_twfe'] if isinstance(r, dict) else r.beta_twfe\n"
        "print(json.dumps({'beta_twfe': float(beta), 'weight_later_vs_earlier': float(lve)}))",
    ),
    "L1-13": (
        "etwfe",
        MP
        + "r = sp.etwfe(mp, y='lemp', group='countyreal', time='year', first_treat='first_treat')\n"
        "print(json.dumps({'att': r.estimate}))",
    ),
    "L1-14": (
        "did_imputation",
        MP
        + "r = sp.did_imputation(mp, y='lemp', group='countyreal', time='year', first_treat='first_treat')\n"
        "print(json.dumps({'att': r.estimate}))",
    ),
    "L1-15": (
        "sdid",
        "d = pd.read_csv('california.csv')\n"
        "r = sp.sdid(d, outcome='cigsale', unit='state', time='year', treated_unit='California', treatment_time=1989)\n"
        "print(json.dumps({'att': r.estimate}))",
    ),
    "L1-16": (
        "evalue",
        "r = sp.evalue(2.5, ci=(1.8, 3.2), measure='RR')\n"
        "print(json.dumps({'evalue_estimate': r['evalue_estimate'], 'evalue_ci': r['evalue_ci']}))",
    ),
    "L1-17": (
        "lmm",
        "d = pd.read_csv('panel.csv')\nr = sp.mixed(d, y='y', x_fixed=['x1'], group='gid')\n"
        "print(json.dumps({'beta_intercept': r.params.iloc[0], 'beta_x1': r.params['x1'], 'icc': sp.icc(r)['icc'] if isinstance(sp.icc(r), dict) else float(sp.icc(r))}))",
    ),
    "L1-18": (
        "glmm_logit",
        "d = pd.read_csv('logit_panel.csv')\nr = sp.melogit(d, y='y', x_fixed=['x1'], group='gid')\n"
        "print(json.dumps({'beta_intercept': r.params.iloc[0], 'beta_x1': r.params['x1']}))",
    ),
    "L1-19": (
        "frontier",
        "d = pd.read_csv('sfa.csv')\nr = sp.frontier(d, y='lny', x=['lnk','lnl'], dist='half-normal')\n"
        "eff = r.efficiency() if callable(getattr(r, 'efficiency', None)) else r.model_info['efficiency']\n"
        "print(json.dumps({'beta_lnk': r.params['lnk'], 'beta_lnl': r.params['lnl'], 'mean_efficiency': float(np.mean(eff))}))",
    ),
    "L1-20": (
        "oaxaca",
        "d = pd.read_csv('wage_gap.csv')\n"
        "o = sp.oaxaca(d, y='log_wage', group='female', x=['educ','exper']).overall\n"
        "print(json.dumps({k: o[k] for k in ('gap','endowments','coefficients','interaction')}))",
    ),
}

# L2 / L3 reuse the L1 computations, renamed to the report fields.
SOLUTIONS.update(
    {
        "L2-01": (
            "iv_2sls",
            SOLUTIONS["L1-02"][1]
            .replace("'beta_educ'", "'effect'")
            .replace("'se_educ'", "'se'"),
        ),
        "L2-02": ("callaway_santanna", SOLUTIONS["L1-03"][1]),
        "L2-03": (
            "rdrobust",
            "d = pd.read_csv('lee.csv')\nr = sp.rdrobust(d, y='voteshare_next', x='margin')\n"
            "print(json.dumps({'effect': r.estimate}))",
        ),
        "L2-04": ("psm", SOLUTIONS["L1-08"][1]),
        "L2-05": ("synth", SOLUTIONS["L1-07"][1]),
        "L2-06": ("sdid", SOLUTIONS["L1-15"][1]),
        "L2-07": (
            "twfe",
            MP
            + D_COL
            + "r = sp.feols('lemp ~ D | countyreal + year', data=mp, vce='cluster countyreal')\n"
            "print(json.dumps({'effect': r.params['D'], 'se': r.std_errors['D']}))",
        ),
        "L2-08": (
            "hdfe_ols",
            "d = pd.read_csv('trade.csv')\nr = sp.feols('y ~ x1 + x2 | firm + year', data=d)\n"
            "print(json.dumps({'beta_x1': r.params['x1'], 'beta_x2': r.params['x2']}))",
        ),
        "L2-09": (
            "bacon",
            SOLUTIONS["L1-12"][1].replace(
                ", 'weight_later_vs_earlier': float(lve)", ""
            ),
        ),
        "L2-10": ("honest_did", SOLUTIONS["L1-11"][1]),
        "L2-11": (
            "sensemakr",
            "d = pd.read_csv('nsw_dw.csv')\n"
            "r = sp.sensemakr(d, y='re78', treat='treat', controls=['age','education','black','hispanic','married','nodegree','re74','re75'])\n"
            "print(json.dumps({'rv_q': float(r['rv_q'])}))",
        ),
        "L2-12": (
            "mediation",
            "d = pd.read_csv('mediation.csv')\n"
            "r = sp.mediate(d, y='y', treat='treat', mediator='m', covariates=['x'])\n"
            "print(json.dumps({'acme': r.model_info['acme'], 'ade': r.model_info['ade']}))",
        ),
        "L2-13": ("coxph", SOLUTIONS["L1-10"][1]),
        "L2-14": (
            "lmm",
            "d = pd.read_csv('panel.csv')\nr = sp.mixed(d, y='y', x_fixed=['x1'], group='gid')\n"
            "print(json.dumps({'beta_x1': r.params['x1']}))",
        ),
        "L2-15": ("glmm_logit", SOLUTIONS["L1-18"][1]),
        "L2-16": ("frontier", SOLUTIONS["L1-19"][1]),
        "L2-17": (
            "oaxaca",
            "d = pd.read_csv('wage_gap.csv')\n"
            "r = sp.oaxaca(d, y='log_wage', group='female', x=['educ','exper'])\n"
            "print(json.dumps({'gap': r.overall['gap'], 'explained': r.overall['explained']}))",
        ),
        "L2-18": (
            "dfl",
            "d = pd.read_csv('wage_gap.csv')\n"
            "r = sp.dfl_decompose(d, y='log_wage', group='female', x=['educ','exper'])\n"
            "print(json.dumps({'counterfactual_mean': float(r.stat_cf)}))",
        ),
        "L2-19": ("evalue", SOLUTIONS["L1-16"][1]),
        "L2-20": (
            "var",
            "d = pd.read_csv('macro.csv')\nr = sp.var(d[['y1','y2']], lags=2)\n"
            "print(json.dumps({'y1_on_L1_y1': float(r.coefs['y1'].loc['L1.y1','coef']), 'y2_on_L1_y1': float(r.coefs['y2'].loc['L1.y1','coef'])}))",
        ),
    }
)

NSW_X = "['age','education','black','hispanic','married','re74','re75']"
SOLUTIONS.update(
    {
        "L3-01": (
            "iv_2sls",
            f"d = pd.read_csv('card.csv')\nr = sp.ivreg('lwage ~ (educ ~ nearc4) + {CARD_X}', data=d, vce='hc1')\n"
            "f = [v for k, v in r.diagnostics.items() if k.startswith('First-stage F (')][0]\n"
            "print(json.dumps({'effect': r.params['educ'], 'first_stage_f': float(f)}))",
        ),
        "L3-02": (
            "callaway_santanna",
            CS + "p = cs.model_info['pretrend_test']['pvalue']\n"
            "print(json.dumps({'att': cs.estimate, 'pretrend_p_value': float(p)}))",
        ),
        "L3-03": (
            "rdrobust",
            "d = pd.read_csv('lee.csv')\nr = sp.rdrobust(d, y='voteshare_next', x='margin')\n"
            "dens = sp.rddensity(d, x='margin')\n"
            "print(json.dumps({'effect': r.estimate, 'density_p_value': dens.pvalue}))",
        ),
        "L3-04": (
            "psm",
            "d = pd.read_csv('nsw_dw.csv')\n"
            f"reg = sp.regress('re78 ~ treat + ' + ' + '.join({NSW_X}), data=d, vce='hc1')\n"
            f"m = sp.match(d, y='re78', treat='treat', covariates={NSW_X}, method='nearest', estimand='ATT', replace=True)\n"
            f"ml = sp.aipw(d, y='re78', treat='treat', covariates={NSW_X}, estimand='ATT', seed=1)\n"
            "print(json.dumps({'att_regression': reg.params['treat'], 'att_matching': m.estimate, 'att_ml': ml.estimate}))",
        ),
        "L3-05": (
            "synth",
            SOLUTIONS["L1-07"][1].replace(
                "print(json.dumps({'avg_post_gap': r.estimate}))",
                "print(json.dumps({'avg_post_gap': r.estimate, 'placebo_p_value': float(r.pvalue)}))",
            ),
        ),
        "L3-06": (
            "hdfe_ols",
            "d = pd.read_csv('trade.csv')\nr = sp.feols('y ~ x1 + x2 | firm + year', data=d, vce='cluster firm')\n"
            "print(json.dumps({'beta_x1': r.params['x1'], 'se_x1': r.std_errors['x1']}))",
        ),
        "L3-07": (
            "lmm",
            "d = pd.read_csv('panel.csv')\nd['t'] = d.groupby('gid').cumcount()\n"
            "fe = sp.panel(d, 'y ~ x1', entity='gid', time='t', method='fe')\n"
            "re = sp.panel(d, 'y ~ x1', entity='gid', time='t', method='re')\n"
            "h = sp.hausman(fe, re)\n"
            "print(json.dumps({'hausman_p_value': float(h['pvalue']), 'beta_x1': float(re.params['x1'])}))",
        ),
        "L3-08": (
            "oaxaca",
            "d = pd.read_csv('wage_gap.csv')\n"
            "o = sp.oaxaca(d, y='log_wage', group='female', x=['educ','exper'])\n"
            "f = sp.dfl_decompose(d, y='log_wage', group='female', x=['educ','exper'])\n"
            "print(json.dumps({'explained_oaxaca': o.overall['explained'], 'explained_dfl': float(f.composition)}))",
        ),
        "L3-09": (
            "coxph",
            "d = pd.read_csv('synth_survival.csv')\n"
            "r = sp.cox(data=d, duration='time', event='event', x=['x1','x2','treat'], hazard_ratio=False)\n"
            "ph = r.ph_test().set_index('variable')\n"
            "print(json.dumps({'beta_treat': r.params['treat'], 'ph_global_p_value': float(ph.loc['GLOBAL','p_value'])}))",
        ),
        "L3-10": (
            "local_projections",
            "d = pd.read_csv('macro.csv')\n"
            "v = sp.var(d[['y1','y2']], lags=2)\n"
            "P = np.linalg.cholesky(np.asarray(v.sigma_u))\n"
            "irf = v.irf(periods=1, impulse='y1', response='y2')['irf']['y1 -> y2'][1] / P[0, 0]\n"
            "for k in (1, 2):\n"
            "    d[f'y1l{k}'] = d.y1.shift(k); d[f'y2l{k}'] = d.y2.shift(k)\n"
            "d['lead'] = d.y2.shift(-1)\n"
            "lp = sp.regress('lead ~ y1 + y1l1 + y1l2 + y2l1 + y2l2', data=d.dropna())\n"
            "print(json.dumps({'irf_var_h1': float(irf), 'irf_lp_h1': float(lp.params['y1'])}))",
        ),
    }
)
