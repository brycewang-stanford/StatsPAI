"""Runnable card examples for entry points changed on 2026-10-19/20.

The 2026-10-20 re-probe ran the card example of every function that gained
parameters in that batch: 31 of 32 failed with ``NameError`` (they used a
``df`` / ``fit`` / ``W`` the card never builds). Each example below builds
its data from a bundled simulator or dataset, runs in a namespace holding
only ``sp`` and binds the result to ``r``; :mod:`statspai._card_examples`
merges them into ``RUNNABLE_EXAMPLES``.
"""

from __future__ import annotations

from typing import Dict

_SIM = "import numpy as np, pandas as pd\n" "rng = np.random.default_rng(0)\n"

_STAG = (
    "df = sp.dgp_did(n_units=200, n_periods=10, staggered=True, seed=1)\n"
    "df['first_treat'] = df['first_treat'].fillna(0)\n"
)

_RING = (
    "n = 100\n"
    "W = np.zeros((n, n))  # ring: each unit linked to its two neighbours\n"
    "for i in range(n):\n"
    "    W[i, (i - 1) % n] = W[i, (i + 1) % n] = 1\n"
)

CHANGED_EXAMPLES: Dict[str, str] = {
    "stata": (
        "df = sp.cps_wage()\n"
        'r = sp.stata("regress log_wage education experience, vce(robust)", '
        "data=df)\n"
    ),
    "bjs": (
        _STAG + "r = sp.bjs(df, y='y', group='unit', time='time', "
        "first_treat='first_treat')\n"
    ),
    "didregress": (
        "df = sp.dgp_did(n_units=100, n_periods=8, seed=2)\n"
        "r = sp.didregress(df, 'y', 'treated', group='unit', time='time')\n"
    ),
    "xtevent": (
        _STAG + "r = sp.xtevent(df, y='y', policy='treated', panel='unit', "
        "time='time', window=2)\n"
    ),
    "did_2x2": (
        "df = sp.dgp_did(n_units=100, n_periods=2, seed=3)\n"
        "df['post'] = (df['time'] == 1).astype(int)\n"
        "df['g'] = df.groupby('unit')['treated'].transform('max')\n"
        "r = sp.did_2x2(df, y='y', treat='g', time='post')\n"
    ),
    "augsynth": (
        _SIM + "units, T = 12, 20\n"
        "rows = [(u, t, np.cumsum(rng.normal(size=T))[t] + u) for u in range(units)"
        " for t in range(T)]\n"
        "df = pd.DataFrame(rows, columns=['unit', 'year', 'y'])\n"
        "df.loc[(df.unit == 0) & (df.year >= 15), 'y'] += 2.0\n"
        "r = sp.augsynth(df, outcome='y', unit='unit', time='year', "
        "treated_unit=0, treatment_time=15, placebo=False)\n"
    ),
    "binscatter": (
        "import matplotlib\n"
        "matplotlib.use('Agg')\n"
        "df = sp.cps_wage()\n"
        "r = sp.binscatter(df, y='log_wage', x='experience', n_bins=15)\n"
    ),
    "sumstats": (
        "df = sp.cps_wage()\n"
        "r = sp.sumstats(df, vars=['log_wage', 'education', 'experience'], "
        "output='dataframe')\n"
    ),
    "balance_table": (
        "df = sp.cps_wage()\n"
        "r = sp.balance_table(df, treat='union', covariates=['education', "
        "'experience', 'female'], output='dataframe')\n"
    ),
    "coefplot": (
        "import matplotlib\n"
        "matplotlib.use('Agg')\n"
        "df = sp.cps_wage()\n"
        "m1 = sp.regress('log_wage ~ education', data=df)\n"
        "m2 = sp.regress('log_wage ~ education + experience', data=df)\n"
        "r = sp.coefplot(m1, m2, model_names=['M1', 'M2'], drop='Intercept')\n"
    ),
    "spillover": (
        _SIM + "cl = np.repeat(np.arange(150), 6)\n"
        "d = rng.binomial(1, 0.5, cl.size)\n"
        "peers = (pd.Series(d).groupby(cl).transform('sum') - d) / 5\n"
        "df = pd.DataFrame({'village': cl, 'd': d,\n"
        "                   'y': 1 + 1.5 * d + 2 * peers + rng.normal(size=cl.size)})\n"
        "r = sp.spillover(df, y='y', treat='d', cluster='village', n_bootstrap=50)\n"
    ),
    "causal_impact": (
        _SIM + "T = 100\n"
        "x = np.cumsum(rng.normal(size=T))\n"
        "y = 1 + 0.8 * x + rng.normal(size=T)\n"
        "y[70:] += 2.0\n"
        "df = pd.DataFrame({'t': np.arange(T), 'y': y, 'x': x})\n"
        "r = sp.causal_impact(df, y='y', time='t', intervention_time=70, "
        "covariates=['x'])\n"
    ),
    "four_way_decomposition": (
        _SIM + "n = 1000\n"
        "a = rng.integers(0, 2, n)\n"
        "m = 0.5 * a + rng.normal(size=n)\n"
        "df = pd.DataFrame({'a': a, 'm': m,\n"
        "                   'y': a + m + 0.5 * a * m + rng.normal(size=n)})\n"
        "r = sp.four_way_decomposition(df, y='y', treat='a', mediator='m')\n"
    ),
    "ltmle": (
        _SIM + "n = 800\n"
        "L0 = rng.normal(size=n)\n"
        "A0 = rng.binomial(1, 1 / (1 + np.exp(-L0)))\n"
        "L1 = 0.5 * L0 + A0 + rng.normal(size=n)\n"
        "A1 = rng.binomial(1, 1 / (1 + np.exp(-L1)))\n"
        "Y = A0 + A1 + L1 + rng.normal(size=n)\n"
        "df = pd.DataFrame({'L0': L0, 'A0': A0, 'L1': L1, 'A1': A1, 'Y': Y})\n"
        "r = sp.ltmle(df, y='Y', treatments=['A0', 'A1'], "
        "covariates_time=[['L0'], ['L1']])\n"
    ),
    "dose_response": (
        _SIM + "n = 600\n"
        "x1, x2 = rng.normal(size=(2, n))\n"
        "dose = 0.5 * x1 + rng.normal(size=n)\n"
        "df = pd.DataFrame({'x1': x1, 'x2': x2, 'dose': dose,\n"
        "                   'y': np.sin(dose) + x1 + x2 + rng.normal(size=n)})\n"
        "r = sp.dose_response(df, y='y', treat='dose', covariates=['x1', 'x2'], "
        "method='dr')\n"
    ),
    "cox": (
        _SIM + "n = 500\n"
        "age = rng.normal(size=n)\n"
        "treatment = rng.integers(0, 2, n)\n"
        "t = rng.exponential(1 / np.exp(0.5 * age - 0.7 * treatment))\n"
        "c = rng.exponential(2.0, n)\n"
        "df = pd.DataFrame({'age': age, 'treatment': treatment,\n"
        "                   'time': np.minimum(t, c),\n"
        "                   'status': (t <= c).astype(int)})\n"
        "r = sp.cox(formula='time ~ age + treatment', data=df, event='status')\n"
    ),
    "kaplan_meier": (
        _SIM + "n = 300\n"
        "t = rng.exponential(5.0, n)\n"
        "c = rng.exponential(10.0, n)\n"
        "df = pd.DataFrame({'time': np.minimum(t, c),\n"
        "                   'status': (t <= c).astype(int)})\n"
        "r = sp.kaplan_meier(data=df, duration='time', event='status', "
        "conf_type='log-log')\n"
    ),
    "finegray": (
        _SIM + "n = 600\n"
        "x = rng.normal(size=n)\n"
        "t1 = rng.exponential(1 / np.exp(0.5 * x))\n"
        "t2 = rng.exponential(1.0, n)\n"
        "c = rng.exponential(3.0, n)\n"
        "t = np.minimum.reduce([t1, t2, c])\n"
        "status = np.select([t == t1, t == t2], [1, 2], 0)\n"
        "df = pd.DataFrame({'x': x, 'time': t, 'status': status})\n"
        "r = sp.finegray(df, duration='time', event='status', x=['x'], cause=1)\n"
    ),
    "msm": (
        _SIM + "rows = []\n"
        "for i in range(300):\n"
        "    V = rng.normal(); L = rng.normal(); A_prev = 0\n"
        "    for t in range(3):\n"
        "        A = int(rng.random() < 1 / (1 + np.exp(-(0.5 * L + A_prev))))\n"
        "        Y = A + 0.5 * L + V + rng.normal()\n"
        "        rows.append((i, t, V, L, A, Y))\n"
        "        L = 0.3 * L + 0.5 * A + rng.normal(); A_prev = A\n"
        "panel = pd.DataFrame(rows, columns=['id', 't', 'V', 'L', 'A', 'Y'])\n"
        "r = sp.msm(panel, y='Y', treat='A', id='id', time='t', "
        "time_varying=['L'], baseline=['V'])\n"
    ),
    "gformula_mc": (
        _SIM + "n = 500\n"
        "L0 = rng.normal(size=n); A0 = rng.binomial(1, 0.5, n)\n"
        "L1 = L0 + A0 + rng.normal(size=n); A1 = rng.binomial(1, 0.5, n)\n"
        "Y = A0 + A1 + L1 + rng.normal(size=n)\n"
        "df = pd.DataFrame({'L0': L0, 'A0': A0, 'L1': L1, 'A1': A1, 'Y': Y})\n"
        "r = sp.gformula_mc(df, treatment_cols=['A0', 'A1'],\n"
        "                   confounder_cols=[['L0'], ['L1']], outcome_col='Y',\n"
        "                   strategy=(1, 1), control_strategy=(0, 0), "
        "n_simulations=2000, bootstrap=20, seed=0)\n"
    ),
    "weakrobust": (
        "df = sp.dgp_iv(n=500, first_stage=0.2, seed=0)\n"
        "r = sp.weakrobust(df, y='y', endog='treatment', instruments=['instrument'],"
        " exog=['x1', 'x2'], random_state=0)\n"
    ),
    "local_projections": (
        _SIM + "T = 200\n"
        "shock = rng.normal(size=T)\n"
        "y = np.zeros(T)\n"
        "for t in range(1, T):\n"
        "    y[t] = 0.5 * y[t - 1] + shock[t] + rng.normal(scale=0.5)\n"
        "df = pd.DataFrame({'y': y, 'shock': shock})\n"
        "r = sp.local_projections(df, outcome='y', shock='shock', horizons=6)\n"
    ),
    "svar": (
        _SIM + "T = 300\n"
        "e = rng.normal(size=(T, 2))\n"
        "Y = np.zeros((T, 2))\n"
        "for t in range(1, T):\n"
        "    Y[t] = 0.5 * Y[t - 1] + e[t] @ np.array([[1.0, 0.5], [0.0, 1.0]])\n"
        "fit = sp.var(pd.DataFrame(Y, columns=['a', 'b']), lags=1)\n"
        "r = sp.svar(fit, B=np.tril(np.full((2, 2), np.nan)))\n"
    ),
    "granger_causality": (
        _SIM + "T = 300\n"
        "x = rng.normal(size=T)\n"
        "y = np.r_[0.0, 0.6 * x[:-1]] + rng.normal(size=T)\n"
        "fit = sp.var(pd.DataFrame({'y': y, 'x': x}), lags=1)\n"
        "r = sp.granger_causality(fit, caused='y', causing='x')\n"
    ),
    "hypotheses": (
        "df = sp.cps_wage()\n"
        "fit = sp.regress('log_wage ~ education + experience + tenure', data=df)\n"
        "r = sp.hypotheses(fit, 'experience = tenure')\n"
    ),
    "wild_cluster_boot": (
        "df = sp.dgp_cluster_rct(n_clusters=20, cluster_size=30, seed=0)\n"
        "fit = sp.regress('y ~ treatment', data=df, cluster='cluster_id')\n"
        "r = sp.wild_cluster_boot(fit, data=df, cluster='cluster_id', "
        "variable='treatment', n_boot=499, seed=1)\n"
    ),
    "sar": (
        _SIM + _RING + "Wn = W / 2\n"
        "x = rng.normal(size=n)\n"
        "y = np.linalg.solve(np.eye(n) - 0.4 * Wn, 1 + x + rng.normal(size=n))\n"
        "df = pd.DataFrame({'y': y, 'x': x})\n"
        "r = sp.sar(W, data=df, formula='y ~ x')\n"
    ),
    "margins": (
        "df = sp.cps_wage()\n"
        "fit = sp.logit('union ~ education + experience', data=df)\n"
        "r = sp.margins(fit, data=df)\n"
    ),
    "margins_at": (
        "df = sp.cps_wage()\n"
        "fit = sp.logit('union ~ education + experience', data=df)\n"
        "r = sp.margins_at(fit, data=df, at={'experience': [1, 5, 10, 20]})\n"
    ),
    "slopes": (
        "df = sp.cps_wage()\n"
        "fit = sp.logit('union ~ education + experience', data=df)\n"
        "r = sp.slopes(fit, 'education')\n"
    ),
    "predictions": (
        "df = sp.cps_wage()\n"
        "fit = sp.logit('union ~ education + experience', data=df)\n"
        "r = sp.predictions(fit)\n"
    ),
}

__all__ = ["CHANGED_EXAMPLES"]
