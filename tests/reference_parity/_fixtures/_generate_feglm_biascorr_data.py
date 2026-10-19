"""Write ``feglm_biascorr_data.csv`` for ``test_feglm_biascorr_parity.py``.

Then run ``_generate_feglm_biascorr_R.R`` with R ``alpaca`` (GPL, used as a
black box: only its output is read) installed in a private library.

Design: a balanced panel of 150 units over 12 periods, unit and period
effects, two regressors correlated with the effects, logit and probit
outcomes (``y``, ``yp``), and a cluster ``cl`` (30 groups of units).
"""

from pathlib import Path

import numpy as np
import pandas as pd

rng = np.random.default_rng(3)
N, T = 150, 12
i = np.repeat(np.arange(N), T)
t = np.tile(np.arange(T), N)
a = rng.normal(size=N)
g = 0.5 * rng.normal(size=T)
x1 = rng.normal(size=N * T) + 0.5 * a[i]
x2 = rng.normal(size=N * T) + 0.3 * g[t]
eta = 0.8 * x1 - 0.5 * x2 + a[i] + g[t]
y = (rng.random(N * T) < 1 / (1 + np.exp(-eta))).astype(int)
yp = (rng.normal(size=N * T) < eta).astype(int)
cl = i // 5
pd.DataFrame(dict(y=y, yp=yp, x1=x1, x2=x2, i=i, t=t, cl=cl)).to_csv(
    Path(__file__).with_name("feglm_biascorr_data.csv"),
    index=False,
    float_format="%.17g",
)
