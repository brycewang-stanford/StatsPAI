"""Native C-LF Delta^RM breakdown values for the R bracket check.

Step 1 of 2 (then ``Rscript _generate_honest_breakdown_rm_clf.R``):
writes ``honest_breakdown_rm_clf_inputs.json`` with every case's inputs and
the native ``Mbar*`` from ``sp.breakdown_m(method='relative_magnitude')``.
"""

import json
import pathlib
import warnings

import numpy as np

import statspai as sp
from statspai.did.honest_did import event_study_from_moments

HERE = pathlib.Path(__file__).parent
rm = json.loads((HERE / "honest_rm_R.json").read_text(encoding="utf-8"))
bd = json.loads((HERE / "honest_breakdown_R.json").read_text(encoding="utf-8"))[
    "_inputs"
]

cases = {}
for key, c in rm.items():
    if key.startswith("_") or not key.endswith("|C-LF"):
        continue
    npre, npost = int(c["npre"]), int(c["npost"])
    l_vec = np.atleast_1d(np.asarray(c["l_vec"], float))
    if np.count_nonzero(l_vec) != 1:
        continue  # breakdown_m targets one event time
    e = int(np.flatnonzero(l_vec)[0])
    cases[key.split("|")[0]] = dict(
        b=c["betahat"],
        S=c["sigma"],
        npre=npre,
        npost=npost,
        e=e,
        times=list(range(-npre, 0)) + list(range(npost)),
    )
t = np.asarray(bd["event_times"])
for e in range(int((t >= 0).sum())):
    cases[f"bd_e{e}"] = dict(
        b=bd["betahat"],
        S=bd["sigma"],
        npre=int((t < 0).sum()),
        npost=int((t >= 0).sum()),
        e=e,
        times=[int(v) for v in t],
    )
with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    for c in cases.values():
        res = event_study_from_moments(
            np.asarray(c["b"], float), np.asarray(c["S"], float), event_times=c["times"]
        )
        c["mstar"] = float(sp.breakdown_m(res, e=c["e"], method="relative_magnitude"))
(HERE / "honest_breakdown_rm_clf_inputs.json").write_text(
    json.dumps(cases, indent=1), encoding="utf-8"
)
