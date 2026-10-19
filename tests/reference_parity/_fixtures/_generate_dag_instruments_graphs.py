"""Random DAGs with latent confounding (bidirected edges) for the
instrument-finder fixture: 200 graphs, 5-9 nodes, an exposure x before an
outcome y, x <-> y in most of them.

    python tests/reference_parity/_fixtures/_generate_dag_instruments_graphs.py
"""

import pathlib

import numpy as np, json
out=[]
for s in range(200):
    rng=np.random.default_rng(1000+s)
    k=int(rng.integers(5,10)); order=list(rng.permutation(k)); names=[f"v{i}" for i in range(k)]
    ix=int(rng.integers(0,k-2)); iy=int(rng.integers(ix+1,k))
    names[order[ix]]="x"; names[order[iy]]="y"
    edges=[]
    for a in range(k):
        for b in range(a+1,k):
            if rng.random()<0.3: edges.append((names[order[a]],names[order[b]]))
    if rng.random()<0.6 and ("x","y") not in edges: edges.append(("x","y"))
    if not any(e[0]=="x" for e in edges): edges.append(("x","y"))
    bi=[]
    for a in range(k):
        for b in range(a+1,k):
            if rng.random()<0.1: bi.append((names[a],names[b]))
    if rng.random()<0.7 and ("x","y") not in bi and ("y","x") not in bi: bi.append(("x","y"))
    out.append({"edges":edges,"bi":bi,"nodes":names})
json.dump(out, open(pathlib.Path(__file__).parent / "dag_instruments_graphs.json", "w"))
