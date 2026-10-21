"""Native breakdown values for targets ``l_vec`` / ``window`` (step 1 of 2).

Then ``Rscript _generate_honest_breakdown_lvec.R`` (run in this directory)
evaluates R ``HonestDiD`` 0.2.8 at multiples of each native value.

Cases: the castle-doctrine event study of Cheng and Hoekstra (2013) as in
the 2026-10 replication (``sp.datasets.castle_doctrine``; Callaway-Sant'Anna
with population weights, never-treated controls, universal base period,
``aggte(type='dynamic', min_e=-5, max_e=4)``), targeting the average of
e = 0..4 under ``Delta^RM`` (C-LF and Conditional) and ``Delta^SD`` (FLCI),
and the same average on ``window=(-3, 2)``; plus the ``corr42`` case of
``honest_rm_R.json`` (``l_vec = (0.5, 0.5)``).
"""

import json
import pathlib
import warnings

import numpy as np

import statspai as sp
from statspai.did._flci import event_study_moments
from statspai.did.honest_did import event_study_from_moments

HERE = pathlib.Path(__file__).parent

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    d = sp.datasets.castle_doctrine()
    d["gvar"] = d["effyear"].fillna(0).astype(int)
    cs = sp.callaway_santanna(
        d,
        y="l_homicide",
        g="gvar",
        t="year",
        i="sid",
        weights="popwt",
        estimator="reg",
        control_group="nevertreated",
        base_period="universal",
    )
    es = sp.aggte(cs, type="dynamic", min_e=-5, max_e=4)
b, S, t = event_study_moments(es)

rm = json.loads((HERE / "honest_rm_R.json").read_text(encoding="utf-8"))
c42 = rm["corr42|C-LF"]
t42 = list(range(-int(c42["npre"]), 0)) + list(range(int(c42["npost"])))

cases = {}
for name, method, hm, window in [
    ("castle_avg_rm_clf", "relative_magnitude", "C-LF", None),
    ("castle_avg_rm_cond", "relative_magnitude", "Conditional", None),
    ("castle_avg_sd_flci", "smoothness", "FLCI", None),
    ("castle_avg_rm_clf_w32", "relative_magnitude", "C-LF", (-3, 2)),
    ("castle_avg_rm_cond_w32", "relative_magnitude", "Conditional", (-3, 2)),
]:
    keep = (
        np.ones(t.size, bool) if window is None else (t >= window[0]) & (t <= window[1])
    )
    cases[name] = dict(
        b=b.tolist(),
        S=S.tolist(),
        times=[int(v) for v in t],
        window=window,
        b_win=b[keep].tolist(),
        S_win=S[np.ix_(keep, keep)].tolist(),
        npre=int((t[keep] < 0).sum()),
        npost=int((t[keep] >= 0).sum()),
        l_vec="average",
        method=method,
        hm=hm,
    )
cases["corr42_half_rm_clf"] = dict(
    b=c42["betahat"],
    S=c42["sigma"],
    times=t42,
    window=None,
    b_win=c42["betahat"],
    S_win=c42["sigma"],
    npre=int(c42["npre"]),
    npost=int(c42["npost"]),
    l_vec=[0.5, 0.5],
    method="relative_magnitude",
    hm="C-LF",
)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    for c in cases.values():
        res = event_study_from_moments(
            np.asarray(c["b"], float), np.asarray(c["S"], float), event_times=c["times"]
        )
        c["mstar"] = float(
            sp.breakdown_m(
                res,
                method=c["method"],
                honestdid_method=c["hm"],
                l_vec=c["l_vec"],
                window=c["window"],
            )
        )
        lv = c["l_vec"]
        c["l_weights"] = (
            [1.0 / c["npost"]] * c["npost"] if lv == "average" else list(lv)
        )
(HERE / "honest_breakdown_lvec_inputs.json").write_text(
    json.dumps(cases, indent=1), encoding="utf-8"
)
