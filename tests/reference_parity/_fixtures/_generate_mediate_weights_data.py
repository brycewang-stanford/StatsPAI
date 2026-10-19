"""Data for the weighted sp.mediate fixture (Stata 18 mediate [pw=]).

    python tests/reference_parity/_fixtures/_generate_mediate_weights_data.py
"""

import pathlib

import numpy as np, pandas as pd
rng=np.random.default_rng(20261021); n=900
x=rng.normal(size=n); g=rng.integers(0,3,n)
d=rng.binomial(1,1/(1+np.exp(-(0.4*x-0.1))),n)
m=0.5+0.7*d+0.3*x+rng.normal(size=n)
mb=rng.binomial(1,1/(1+np.exp(-(-0.3+0.9*d+0.3*x))),n)
y=1+0.4*d+0.6*m+0.3*d*m+0.2*x+rng.normal(size=n)
yb=rng.binomial(1,1/(1+np.exp(-(-0.8+0.4*d+0.5*mb+0.2*x))),n)
w=np.round(rng.gamma(2.0,0.5,n),3)+0.05
pd.DataFrame(dict(y=y,yb=yb,d=d,m=m,mb=mb,x=x,w=w)).to_csv(pathlib.Path(__file__).parent / "mediate_weights.csv",index=False,float_format="%.17g")
