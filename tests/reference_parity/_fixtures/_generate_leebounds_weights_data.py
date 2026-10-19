"""Data for the weighted ``sp.lee_bounds`` fixture (Stata ``leebounds [pw=]``).

Four samples: continuous outcomes (samples 1-2) and integer outcomes with
ties (3-4), the treated arm more retained (1, 3) or the control arm (2, 4),
positive sampling weights ``w``.

    python tests/reference_parity/_fixtures/_generate_leebounds_weights_data.py
"""

import pathlib

import numpy as np, pandas as pd
fr=[]
for k,(n,cont,flip) in enumerate([(400,True,False),(700,True,True),(500,False,False),(600,False,True)],start=1):
    rng=np.random.default_rng(20261020+k)
    d=rng.integers(0,2,n); x=rng.integers(0,2,n)
    p=np.where(d==1,0.85,0.65) if not flip else np.where(d==1,0.62,0.8)
    s=(rng.random(n)<p).astype(int)
    y=rng.normal(1+0.4*d,1,n) if cont else rng.integers(0,5,n).astype(float)
    w=np.round(rng.gamma(2.0,0.5,n),3)+0.05
    fr.append(pd.DataFrame(dict(sample=k,y=np.where(s==1,y,np.nan),d=d,s=s,x=x,w=w)))
pd.concat(fr).to_csv(pathlib.Path(__file__).parent / "leebounds_weights.csv",index=False,float_format="%.17g")
