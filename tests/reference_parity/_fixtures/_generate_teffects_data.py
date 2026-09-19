"""Write the shared data for the treatment-effects parity family.

Run once; the CSVs are committed so the R generator
(``_generate_teffects_R.R``), the Stata generator
(``_generate_teffects_stata.do``) and the Python test
(``test_teffects_R_parity.py``) all read identical bytes.

Files (all ``%.17g`` so nothing is lost on the round trip)
---------------------------------------------------------
teffects_cs.csv      n = 1000 cross-section, binary treatment ``d``.
    x1, x2 ~ N(0, 1); x3 ~ Bernoulli(0.4)
    d  ~ Bernoulli(expit(-0.3 + 0.6 x1 - 0.4 x2 + 0.5 x3))
    y  = 1 + 1.5 d + 0.8 x1 + 0.5 x2 - 0.7 x3 + 0.6 d x1 + N(0, 1)
         (heterogeneous in x1, so ATE != ATT)
    yb ~ Bernoulli(expit(-0.4 + 0.9 d + 0.5 x1 - 0.3 x2))
    s  ~ Bernoulli(0.55 + 0.25 d)     monotone selection (Lee / SACE)
    ys = y if s == 1 else missing
    m  = 0.3 + 0.7 d + 0.4 x1 + N(0, 1)                 mediator
    ym = 0.5 + 0.9 d + 0.6 m + 0.4 d m + 0.3 x1 + N(0, 1)
    l  = 0.2 + 0.8 d + 0.3 x1 + N(0, 1)                 exposure-induced
    m2 = 0.1 + 0.5 d + 0.4 x1 + N(0, 1)                 mediator (no l)
    yl = 0.4 + 0.7 d + 0.5 m2 + 0.6 l + 0.3 x1 + N(0, 1)
    t3 in {0, 1, 2}: multinomial logit on (x1, x2)
    ymt = 1 + 1.0 [t3 = 1] + 2.0 [t3 = 2] + 0.8 x1 + 0.5 x2 + 0.4 [t3 = 2] x1 + N(0, 1)
    tc = 1 + 0.5 x1 - 0.3 x2 + N(0, 1)                  continuous dose
    ydose = 2 + 0.8 tc - 0.1 tc^2 + 0.5 x1 + 0.3 x2 + N(0, 1)

teffects_wide.csv    n = 1000, two time points (wide).
    v  ~ N(0, 1)                                        baseline
    L0 = 0.3 v + N(0, 1)
    A0 ~ Bernoulli(expit(-0.2 + 0.5 L0 + 0.3 v))
    L1 = 0.5 L0 + 0.6 A0 + 0.2 v + N(0, 1)
    A1 ~ Bernoulli(expit(-0.1 + 0.4 L1 + 0.5 A0))
    Y  = 1 + 0.7 A0 + 0.9 A1 + 0.5 L0 + 0.6 L1 + 0.2 v + N(0, 1)
    Yb ~ Bernoulli(expit(-0.5 + 0.6 A0 + 0.5 A1 + 0.4 L0 + 0.3 L1))

teffects_long.csv    300 units x 4 periods (long), time-varying treatment.
    v ~ N(0, 1) baseline; l_t = 0.4 a_{t-1} + 0.3 v + N(0, 1) measured
    before a_t; a_t ~ Bernoulli(expit(-0.3 + 0.6 l_t + 0.8 a_{t-1}));
    y = 1 + 0.5 sum_s a_s + 0.3 v + N(0, 1), repeated within unit.
    ac_t = 0.5 + 0.4 l_t + 0.3 ac_{t-1} + N(0, 1) continuous treatment.
    yb ~ Bernoulli(expit(-1 + 0.4 sum_s a_s + 0.3 v)), repeated within unit.

teffects_wide_cens.csv  teffects_wide plus censoring after each treatment:
    C0 ~ Bernoulli(expit(2 + 0.3 L0 - 0.4 A0)) (1 = uncensored),
    C1 ~ Bernoulli(expit(2 - 0.3 L1 + 0.3 A1)) among C0 = 1; nodes after
    censoring are missing.

teffects_surv.csv    n = 500 right-censored cohort, one row per subject.
    z1 ~ N(0, 1), z2 ~ Bernoulli(0.5)
    T ~ Exp(1) * exp(-0.3 z1); C ~ Exp(mean 1.2) * exp(-0.5 z1 + 0.4 z2)
    time = min(T, C), event = 1{T <= C}   (censoring depends on z1, z2)
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

OUT = pathlib.Path(__file__).parent
rng = np.random.default_rng(20260918)


def expit(z):
    return 1.0 / (1.0 + np.exp(-z))


# --------------------------------------------------------------------------
# Cross-section
# --------------------------------------------------------------------------
n = 1000
x1 = rng.standard_normal(n)
x2 = rng.standard_normal(n)
x3 = (rng.uniform(size=n) < 0.4).astype(int)
d = (rng.uniform(size=n) < expit(-0.3 + 0.6 * x1 - 0.4 * x2 + 0.5 * x3)).astype(int)
y = 1 + 1.5 * d + 0.8 * x1 + 0.5 * x2 - 0.7 * x3 + 0.6 * d * x1 + rng.standard_normal(n)
yb = (rng.uniform(size=n) < expit(-0.4 + 0.9 * d + 0.5 * x1 - 0.3 * x2)).astype(int)
s = (rng.uniform(size=n) < 0.55 + 0.25 * d).astype(int)
ys = np.where(s == 1, y, np.nan)
m = 0.3 + 0.7 * d + 0.4 * x1 + rng.standard_normal(n)
ym = 0.5 + 0.9 * d + 0.6 * m + 0.4 * d * m + 0.3 * x1 + rng.standard_normal(n)
l_tv = 0.2 + 0.8 * d + 0.3 * x1 + rng.standard_normal(n)
m2 = 0.1 + 0.5 * d + 0.4 * x1 + rng.standard_normal(n)
yl = 0.4 + 0.7 * d + 0.5 * m2 + 0.6 * l_tv + 0.3 * x1 + rng.standard_normal(n)

eta1 = -0.2 + 0.5 * x1 - 0.3 * x2
eta2 = -0.4 - 0.4 * x1 + 0.6 * x2
den = 1 + np.exp(eta1) + np.exp(eta2)
p1, p2 = np.exp(eta1) / den, np.exp(eta2) / den
u = rng.uniform(size=n)
t3 = np.where(u < p1, 1, np.where(u < p1 + p2, 2, 0))
ymt = (
    1
    + 1.0 * (t3 == 1)
    + 2.0 * (t3 == 2)
    + 0.8 * x1
    + 0.5 * x2
    + 0.4 * (t3 == 2) * x1
    + rng.standard_normal(n)
)
tc = 1 + 0.5 * x1 - 0.3 * x2 + rng.standard_normal(n)
ydose = 2 + 0.8 * tc - 0.1 * tc**2 + 0.5 * x1 + 0.3 * x2 + rng.standard_normal(n)

cs = pd.DataFrame(
    {
        "id": np.arange(1, n + 1),
        "x1": x1,
        "x2": x2,
        "x3": x3,
        "d": d,
        "y": y,
        "yb": yb,
        "s": s,
        "ys": ys,
        "m": m,
        "ym": ym,
        "l": l_tv,
        "m2": m2,
        "yl": yl,
        "t3": t3,
        "ymt": ymt,
        "tc": tc,
        "ydose": ydose,
    }
)
cs.to_csv(OUT / "teffects_cs.csv", index=False, float_format="%.17g")

# --------------------------------------------------------------------------
# Wide, two time points
# --------------------------------------------------------------------------
v = rng.standard_normal(n)
L0 = 0.3 * v + rng.standard_normal(n)
A0 = (rng.uniform(size=n) < expit(-0.2 + 0.5 * L0 + 0.3 * v)).astype(int)
L1 = 0.5 * L0 + 0.6 * A0 + 0.2 * v + rng.standard_normal(n)
A1 = (rng.uniform(size=n) < expit(-0.1 + 0.4 * L1 + 0.5 * A0)).astype(int)
Y = 1 + 0.7 * A0 + 0.9 * A1 + 0.5 * L0 + 0.6 * L1 + 0.2 * v + rng.standard_normal(n)
Yb = (
    rng.uniform(size=n) < expit(-0.5 + 0.6 * A0 + 0.5 * A1 + 0.4 * L0 + 0.3 * L1)
).astype(int)
wide = pd.DataFrame(
    {"id": np.arange(1, n + 1), "v": v, "L0": L0, "A0": A0, "L1": L1, "A1": A1,
     "Y": Y, "Yb": Yb}
)
wide.to_csv(OUT / "teffects_wide.csv", index=False, float_format="%.17g")

# --------------------------------------------------------------------------
# Long panel, 4 periods
# --------------------------------------------------------------------------
N_ID, T = 300, 4
rows = []
for i in range(N_ID):
    vi = rng.standard_normal()
    a_prev = 0.0
    ac_prev = 0.0
    cum = 0.0
    unit = []
    for t in range(T):
        lt = 0.4 * a_prev + 0.3 * vi + rng.standard_normal()
        at = float(rng.uniform() < expit(-0.3 + 0.6 * lt + 0.8 * a_prev))
        act = 0.5 + 0.4 * lt + 0.3 * ac_prev + rng.standard_normal()
        cum += at
        unit.append([i + 1, t, vi, lt, at, act])
        a_prev, ac_prev = at, act
    yi = 1 + 0.5 * cum + 0.3 * vi + rng.standard_normal()
    for r in unit:
        rows.append(r + [yi])
long = pd.DataFrame(rows, columns=["id", "t", "v", "l", "a", "ac", "y"])
long["a"] = long["a"].astype(int)
cum_last = long.groupby("id")["a"].transform("sum").to_numpy()
v_unit = long["v"].to_numpy()
u_unit = np.repeat(rng.uniform(size=N_ID), T)
long["yb"] = (u_unit < expit(-1.0 + 0.4 * cum_last + 0.3 * v_unit)).astype(int)
long.to_csv(OUT / "teffects_long.csv", index=False, float_format="%.17g")

# --------------------------------------------------------------------------
# Right-censored cohort (one row per subject) for IPCW
# --------------------------------------------------------------------------
NS = 500
z1 = rng.standard_normal(NS)
z2 = (rng.uniform(size=NS) < 0.5).astype(int)
t_event = rng.exponential(1.0, NS) * np.exp(-0.3 * z1)
t_cens = rng.exponential(1.2, NS) * np.exp(-0.5 * z1 + 0.4 * z2)
surv = pd.DataFrame(
    {
        "id": np.arange(1, NS + 1),
        "time": np.minimum(t_event, t_cens),
        "event": (t_event <= t_cens).astype(int),
        "z1": z1,
        "z2": z2,
    }
)
surv.to_csv(OUT / "teffects_surv.csv", index=False, float_format="%.17g")

# --------------------------------------------------------------------------
# Wide data with right censoring after each treatment (LTMLE C nodes)
# --------------------------------------------------------------------------
C0 = (rng.uniform(size=n) < expit(2.0 + 0.3 * L0 - 0.4 * A0)).astype(int)
C1 = (rng.uniform(size=n) < expit(2.0 - 0.3 * L1 + 0.3 * A1)).astype(int) * C0
wc = wide.copy()
wc["C0"] = C0
wc["C1"] = C1
wc.loc[C0 == 0, ["L1", "A1"]] = np.nan
wc.loc[C1 == 0, ["Y", "Yb"]] = np.nan
wc = wc[["id", "v", "L0", "A0", "C0", "L1", "A1", "C1", "Y", "Yb"]]
wc.to_csv(OUT / "teffects_wide_cens.csv", index=False, float_format="%.17g")

print(
    f"cs n={n} treated={d.sum()} selected={s.sum()} t3={np.bincount(t3)}; "
    f"wide A0={A0.sum()} A1={A1.sum()}; long rows={len(long)}"
)
