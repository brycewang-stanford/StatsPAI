"""Runnable card examples for the entry points added on 2026-10-19/20.

Their cards carried the call line of the docstring example (``sp.xttobit(df,
y="hours", ...)``) without the lines that build ``df``, so an agent that
copied one hit a ``NameError`` first (2026-10-20 re-audit: 30 of 30).
Each example below simulates its own data with NumPy (or uses a bundled
dataset), runs in a namespace holding only ``sp`` and binds the result to
``r``; :mod:`statspai._card_examples` merges them into
``RUNNABLE_EXAMPLES`` and ``tests/test_card_examples.py`` runs them all.
"""

from __future__ import annotations

from typing import Dict

_SIM = "import numpy as np, pandas as pd\n" "rng = np.random.default_rng(0)\n"

NEW_EXAMPLES: Dict[str, str] = {
    "xtset": (
        _SIM + "df = pd.DataFrame({'firm': np.repeat([1, 2], 4),\n"
        "                   'year': [2001, 2002, 2004, 2005] * 2,\n"
        "                   'x': rng.normal(size=8)})\n"
        "r = sp.xtset(df, id='firm', time='year')\n"
    ),
    "lag": (
        _SIM + "df = pd.DataFrame({'firm': np.repeat([1, 2], 4),\n"
        "                   'year': [2001, 2002, 2004, 2005] * 2,\n"
        "                   'x': rng.normal(size=8)})\n"
        "r = sp.lag(df, 'x', id='firm', time='year')  # NaN after the 2003 gap\n"
    ),
    "lead": (
        _SIM + "df = pd.DataFrame({'firm': np.repeat([1, 2], 4),\n"
        "                   'year': [2001, 2002, 2004, 2005] * 2,\n"
        "                   'x': rng.normal(size=8)})\n"
        "r = sp.lead(df, 'x', id='firm', time='year')\n"
    ),
    "diff": (
        _SIM + "df = pd.DataFrame({'firm': np.repeat([1, 2], 4),\n"
        "                   'year': [2001, 2002, 2004, 2005] * 2,\n"
        "                   'x': rng.normal(size=8)})\n"
        "r = sp.diff(df, 'x', id='firm', time='year')\n"
    ),
    "reshape": (
        "import pandas as pd\n"
        "df = pd.DataFrame({'hid': [1, 1, 2, 2], 'year': [2000, 2001] * 2,\n"
        "                   'inc': [10.0, 11.0, 20.0, 22.0]})\n"
        "r = sp.reshape(df, 'wide', stubs='inc', id='hid', j='year')\n"
    ),
    "collapse": (
        "df = sp.cps_wage()\n"
        "r = sp.collapse(df, '(mean) log_wage (count) n=log_wage', by='female')\n"
    ),
    "cuped": (
        _SIM + "n = 2000\n"
        "y_pre = rng.normal(size=n)\n"
        "d = rng.integers(0, 2, n)\n"
        "df = pd.DataFrame({'d': d, 'y_pre': y_pre,\n"
        "                   'y': 0.2 * d + 0.8 * y_pre + rng.normal(size=n)})\n"
        "r = sp.cuped(df, 'y', 'd', ['y_pre'])\n"
    ),
    "msprt": (
        _SIM + "n = 1000\n"
        "d = rng.integers(0, 2, n)\n"
        "df = pd.DataFrame({'d': d, 'y': 0.3 * d + rng.normal(size=n)})\n"
        "r = sp.msprt(df, 'y', 'd', tau=0.2, sigma=1.0)\n"
    ),
    "msprt_lambda": "r = sp.msprt_lambda(0.3, 0.01, tau=0.2)\n",
    "group_sequential": (
        "r = sp.group_sequential(n_looks=3, efficacy='errobfleming',\n"
        "                        futility='errpocock', side='upper')\n"
    ),
    "gologit": (
        _SIM + "n = 2000\n"
        "age = rng.normal(size=n); income = rng.normal(size=n)\n"
        "latent = 0.8 * age + 0.5 * income + rng.logistic(size=n)\n"
        "df = pd.DataFrame({'age': age, 'income': income,\n"
        "                   'health': np.digitize(latent, [-1.0, 0.5, 2.0])})\n"
        "r = sp.gologit(df, y='health', x=['age', 'income'], npl=['age'])\n"
    ),
    "heckprobit": (
        _SIM + "n = 3000\n"
        "age, income, phone = rng.normal(size=(3, n))\n"
        "u, v = rng.multivariate_normal([0, 0], [[1, 0.5], [0.5, 1]], n).T\n"
        "asked = (0.3 + phone + 0.3 * age + v > 0).astype(int)\n"
        "insured = (0.2 + 0.5 * age + 0.5 * income + u > 0).astype(int)\n"
        "df = pd.DataFrame({'age': age, 'income': income, 'phone': phone,\n"
        "                   'asked': asked, 'insured': insured * asked})\n"
        "r = sp.heckprobit(df, y='insured', x=['age', 'income'],\n"
        "                  select='asked', z=['phone', 'age'])\n"
    ),
    "intreg": (
        _SIM + "n = 1500\n"
        "educ, exper = rng.normal(size=(2, n))\n"
        "w = 2 + 0.5 * educ + 0.3 * exper + rng.normal(size=n)\n"
        "lo = np.floor(w); hi = lo + 1.0\n"
        "df = pd.DataFrame({'educ': educ, 'exper': exper,\n"
        "                   'wage_lo': lo, 'wage_hi': hi})\n"
        "r = sp.intreg(df, y_lower='wage_lo', y_upper='wage_hi',\n"
        "              x=['educ', 'exper'])\n"
    ),
    "xttobit": (
        _SIM + "n_id, T = 200, 5\n"
        "ids = np.repeat(np.arange(n_id), T)\n"
        "wage = rng.normal(size=n_id * T)\n"
        "y = 1 + wage + rng.normal(size=n_id)[ids] + rng.normal(size=n_id * T)\n"
        "df = pd.DataFrame({'id': ids, 'wage': wage, 'hours': np.maximum(y, 0)})\n"
        "r = sp.xttobit(df, y='hours', x=['wage'], id='id', ll=0)\n"
    ),
    "xtintreg": (
        _SIM + "n_id, T = 100, 4\n"
        "ids = np.repeat(np.arange(n_id), T)\n"
        "educ = rng.normal(size=n_id * T)\n"
        "w = 2 + educ + rng.normal(size=n_id)[ids] + rng.normal(size=n_id * T)\n"
        "df = pd.DataFrame({'id': ids, 'educ': educ,\n"
        "                   'wage_lo': np.floor(w), 'wage_hi': np.floor(w) + 1})\n"
        "r = sp.xtintreg(df, y_lower='wage_lo', y_upper='wage_hi', x=['educ'],\n"
        "                id='id')\n"
    ),
    "xtivreg": (
        _SIM + "n_id, T = 300, 4\n"
        "ids = np.repeat(np.arange(n_id), T)\n"
        "z, e = rng.normal(size=(2, n_id * T))\n"
        "exper = rng.normal(size=n_id * T)\n"
        "educ = z + 0.5 * e + rng.normal(size=n_id * T)\n"
        "lwage = (0.1 * educ + 0.05 * exper + rng.normal(size=n_id)[ids] + e)\n"
        "df = pd.DataFrame({'id': ids, 'lwage': lwage, 'educ': educ,\n"
        "                   'exper': exper, 'meduc': z})\n"
        "r = sp.xtivreg(df, y='lwage', x=['exper'], endog='educ',\n"
        "               instruments=['meduc'], id='id')\n"
    ),
    "xtpcse": (
        _SIM + "countries, years = 10, 20\n"
        "df = pd.DataFrame({'country': np.repeat(np.arange(countries), years),\n"
        "                   'year': np.tile(np.arange(years), countries)})\n"
        "df['trade'] = rng.normal(size=len(df))\n"
        "df['gdp'] = 1 + 0.5 * df.trade + rng.normal(size=len(df))\n"
        "r = sp.xtpcse(df, y='gdp', x=['trade'], id='country', time='year',\n"
        "              correlation='ar1')\n"
    ),
    "sdem": (
        _SIM + "n = 200\n"
        "W = np.zeros((n, n))  # ring: each unit linked to its two neighbours\n"
        "for i in range(n):\n"
        "    W[i, (i - 1) % n] = W[i, (i + 1) % n] = 1\n"
        "x = rng.normal(size=n)\n"
        "Wn = W / W.sum(1, keepdims=True)\n"
        "u = np.linalg.solve(np.eye(n) - 0.4 * Wn, rng.normal(size=n))\n"
        "df = pd.DataFrame({'x': x, 'y': 1 + x + 0.5 * Wn @ x + u})\n"
        "r = sp.sdem(W, df, 'y ~ x')\n"
    ),
    "gns": (
        _SIM + "n = 200\n"
        "W = np.zeros((n, n))  # ring: each unit linked to its two neighbours\n"
        "for i in range(n):\n"
        "    W[i, (i - 1) % n] = W[i, (i + 1) % n] = 1\n"
        "Wn = W / W.sum(1, keepdims=True)\n"
        "x = rng.normal(size=n)\n"
        "u = np.linalg.solve(np.eye(n) - 0.3 * Wn, rng.normal(size=n))\n"
        "y = np.linalg.solve(np.eye(n) - 0.3 * Wn, 1 + x + 0.5 * Wn @ x + u)\n"
        "df = pd.DataFrame({'x': x, 'y': y})\n"
        "r = sp.gns(W, df, 'y ~ x')\n"
    ),
    "iv_wild_bootstrap": (
        _SIM + "G, m = 30, 20\n"
        "firm = np.repeat(np.arange(G), m)\n"
        "z = rng.normal(size=G * m); u = rng.normal(size=G * m)\n"
        "d = z + 0.5 * u + rng.normal(size=G * m)\n"
        "df = pd.DataFrame({'firm': firm, 'z': z, 'd': d,\n"
        "                   'y': 0.5 * d + u + rng.normal(size=G)[firm]})\n"
        "fit = sp.ivreg('y ~ (d ~ z)', data=df, cluster='firm')\n"
        "r = sp.iv_wild_bootstrap(fit, df, 'firm', 'd', n_boot=199, seed=1)\n"
    ),
    "l1_imbalance": (
        _SIM + "n = 1000\n"
        "x = rng.normal(size=n)\n"
        "df = pd.DataFrame({'x': x, 'd': (x + rng.normal(size=n) > 0).astype(int)})\n"
        "r = sp.l1_imbalance(df, 'd', ['x'])\n"
    ),
    "weightit": (
        _SIM + "n = 1000\n"
        "x1, x2 = rng.normal(size=(2, n))\n"
        "df = pd.DataFrame({'x1': x1, 'x2': x2,\n"
        "                   'dose': 0.5 * x1 - 0.3 * x2 + rng.normal(size=n)})\n"
        "r = sp.weightit(df, 'dose', ['x1', 'x2'], method='ebal')\n"
    ),
    "bal_tab": (
        _SIM + "n = 1000\n"
        "x1, x2 = rng.normal(size=(2, n))\n"
        "treat = rng.choice(['a', 'b', 'c'], n)\n"
        "df = pd.DataFrame({'x1': x1 + (treat == 'b'), 'x2': x2, 'treat': treat})\n"
        "W = sp.weightit(df, 'treat', ['x1', 'x2'], method='glm')\n"
        "r = sp.bal_tab(df, 'treat', ['x1', 'x2'], weights=W)\n"
    ),
    "mediate_paths": (
        _SIM + "n = 1000\n"
        "a = rng.integers(0, 2, n)\n"
        "m1 = 0.5 * a + rng.normal(size=n)\n"
        "m2 = 0.4 * a + 0.3 * m1 + rng.normal(size=n)\n"
        "y = 0.2 * a + 0.5 * m1 + 0.5 * m2 + rng.normal(size=n)\n"
        "df = pd.DataFrame({'a': a, 'm1': m1, 'm2': m2, 'y': y})\n"
        "r = sp.mediate_paths(df, y='y', treat='a', mediators=['m1', 'm2'],\n"
        "                     interaction=False, n_boot=50)\n"
    ),
    "ida": (
        _SIM + "a = rng.normal(size=2000); b = a + rng.normal(size=2000)\n"
        "c = b + rng.normal(size=2000)\n"
        "df = pd.DataFrame({'a': a, 'b': b, 'c': c})\n"
        "cp = pd.DataFrame([[0, 1, 0], [1, 0, 1], [0, 1, 0]],\n"
        "                  index=list('abc'), columns=list('abc'))\n"
        "r = sp.ida(df, cp, x='b', y='c')\n"
    ),
    "lmtp": (
        _SIM + "n = 1000\n"
        "W = rng.normal(size=n)\n"
        "A = W + rng.normal(size=n)\n"
        "df = pd.DataFrame({'W': W, 'A': A,\n"
        "                   'Y': 0.5 * A + W + rng.normal(size=n)})\n"
        "r = sp.lmtp(df, 'Y', ['A'], shift=1.0, baseline=['W'])\n"
    ),
    "rmst": (
        _SIM + "n = 1000\n"
        "a = rng.integers(0, 2, n)\n"
        "t = rng.exponential(np.where(a == 1, 12.0, 8.0))\n"
        "c = rng.exponential(30.0, n)\n"
        "df = pd.DataFrame({'a': a, 'time': np.minimum(t, c),\n"
        "                   'event': (t <= c).astype(int)})\n"
        "r = sp.rmst(df, 'time', 'event', group='a', tau=10)\n"
    ),
    "stteffects": (
        _SIM + "n = 1500\n"
        "x = rng.normal(size=n)\n"
        "a = (x + rng.normal(size=n) > 0).astype(int)\n"
        "t = rng.weibull(1.5, n) * np.exp(1 + 0.5 * a + 0.3 * x)\n"
        "c = rng.exponential(10.0, n)\n"
        "df = pd.DataFrame({'x': x, 'a': a, 't': np.minimum(t, c),\n"
        "                   'd': (t <= c).astype(int)})\n"
        "r = sp.stteffects(df, 't', 'd', 'a', covariates=['x'], method='ra')\n"
    ),
    "panel_match": (
        _SIM + "units, T = 60, 12\n"
        "df = pd.DataFrame({'unit': np.repeat(np.arange(units), T),\n"
        "                   'year': np.tile(np.arange(T), units)})\n"
        "start = rng.choice([6, 8, 99], units)\n"
        "df['d'] = (df.year >= start[df.unit]).astype(int)\n"
        "df['x'] = rng.normal(size=len(df))\n"
        "df['y'] = 0.5 * df.d + df.x + rng.normal(size=len(df))\n"
        "r = sp.panel_match(df, y='y', treat='d', id='unit', time='year',\n"
        "                   lag=4, lead=[0, 1], covariates=['x'])\n"
    ),
    "bart_causal": (
        _SIM + "n = 300\n"
        "x1, x2 = rng.normal(size=(2, n))\n"
        "d = (x1 + rng.normal(size=n) > 0).astype(int)\n"
        "df = pd.DataFrame({'x1': x1, 'x2': x2, 'd': d,\n"
        "                   'y': d + x1 + rng.normal(size=n)})\n"
        "r = sp.bart_causal(df, 'y', 'd', ['x1', 'x2'], draws=100, burnin=100,\n"
        "                   chains=1, seed=1)\n"
    ),
}

__all__ = ["NEW_EXAMPLES"]
