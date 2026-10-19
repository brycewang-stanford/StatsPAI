"""Data for the random-effects spatial panel fixture (Stata spxtregress, re).

48 units on a 6 x 8 rook grid, 5 periods; spatial lag 0.3 and spatial
error 0.25, unit effects correlated with x2. spatial_panel_re_W.csv holds the
binary contiguity matrix row by row (columns w1..w48).

    python tests/reference_parity/_fixtures/_generate_spatial_panel_re_data.py
"""

import pathlib

import numpy as np, pandas as pd
rng=np.random.default_rng(5)
R,C=6,8; N=R*C; T=5
A=np.zeros((N,N))
for r in range(R):
    for c in range(C):
        for dr,dc in ((1,0),(-1,0),(0,1),(0,-1)):
            if 0<=r+dr<R and 0<=c+dc<C: A[r*C+c,(r+dr)*C+c+dc]=1
W=A/A.sum(1,keepdims=True)
lam,rho=0.3,0.25
ci=rng.normal(scale=0.7,size=N)
rows=[]
for t in range(T):
    x=rng.normal(size=N); x2=rng.normal(size=N)+0.3*ci
    u=np.linalg.solve(np.eye(N)-rho*W, rng.normal(scale=0.6,size=N))
    y=np.linalg.solve(np.eye(N)-lam*W, 1+0.8*x-0.4*x2+ci+u)
    for i in range(N): rows.append((i+1,t+1,y[i],x[i],x2[i]))
pd.DataFrame(rows,columns=["id","t","y","x","x2"]).to_csv(pathlib.Path(__file__).parent / "spatial_panel_re.csv",index=False,float_format="%.17g")
cs=pd.DataFrame({"id":np.arange(1,N+1)})
for j in range(N): cs[f"w{j+1}"]=A[:,j]
cs.to_csv(pathlib.Path(__file__).parent / "spatial_panel_re_W.csv",index=False)
