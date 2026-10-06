"""Simulated data and system matrices for test_statespace_parity.py.

Writes ``statespace.csv`` (the observations of six models, NaN where
missing) and ``statespace_spec.json`` (their system matrices, read by both
the Python test and ``_generate_statespace_R.R``).

    python tests/reference_parity/_fixtures/_generate_statespace_data.py

Model: X_t = F_t X_{t-1} + V_t, Var V_t = Q_t; Y_t = A_t + G_t X_t + W_t,
Var W_t = R_t; X_0 ~ (x0, P0). ``P0: null`` means the stationary covariance.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
T = 80
rng = np.random.default_rng(20261006)


def simulate(spec: dict, n: int) -> np.ndarray:
    """One path of the observables; time-varying entries carry axis 0."""

    def at(key: str, t: int, ndim: int) -> np.ndarray:
        arr = np.asarray(spec[key], dtype=float)
        return arr[t] if arr.ndim == ndim + 1 else arr

    m = np.asarray(spec["F"]).shape[-1]
    x = np.asarray(spec["x0"], dtype=float)
    out = np.zeros((T, n))
    for t in range(T):
        Q = at("Q", t, 2)
        R = at("R", t, 2)
        x = at("F", t, 2) @ x + rng.multivariate_normal(np.zeros(m), Q)
        out[t] = (
            at("A", t, 1) + at("G", t, 2) @ x + rng.multivariate_normal(np.zeros(n), R)
        )
    return out


specs: dict = {}
cols: dict = {}

# 1. local level, proper prior
specs["level"] = dict(
    F=[[1.0]], G=[[1.0]], Q=[[0.3]], R=[[1.2]], A=[0.0], x0=[0.5], P0=[[4.0]]
)
cols["level"] = simulate(specs["level"], 1)[:, 0]

# 2. AR(2) plus noise, stationary initial state
specs["ar2"] = dict(
    F=[[1.1, -0.3], [1.0, 0.0]],
    G=[[1.0, 0.0]],
    Q=[[0.5, 0.0], [0.0, 0.0]],
    R=[[0.4]],
    A=[2.0],
    x0=[0.0, 0.0],
    P0=None,
)
cols["ar2"] = simulate(specs["ar2"], 1)[:, 0]

# 3. two observables, two states, correlated errors, missing values
specs["biv"] = dict(
    F=[[0.7, 0.2], [-0.1, 0.5]],
    G=[[1.0, 0.5], [0.3, 1.0]],
    Q=[[1.0, 0.3], [0.3, 0.6]],
    R=[[0.5, 0.2], [0.2, 0.8]],
    A=[1.0, -1.0],
    x0=[0.0, 0.0],
    P0=None,
)
yb = simulate(specs["biv"], 2)
yb[rng.random((T, 2)) < 0.15] = np.nan
yb[[0, 10, 11, 40, T - 1], :] = np.nan
cols["biv1"], cols["biv2"] = yb[:, 0], yb[:, 1]

# 4. regression with random-walk coefficients (time-varying G)
xreg = rng.normal(size=T)
specs["tvp"] = dict(
    F=[[1.0, 0.0], [0.0, 1.0]],
    G=[[[1.0, float(v)]] for v in xreg],
    Q=[[0.05, 0.0], [0.0, 0.02]],
    R=[[0.3]],
    A=[0.0],
    x0=[1.0, 0.5],
    P0=[[4.0, 0.0], [0.0, 1.0]],
)
cols["tvp_x"] = xreg
cols["tvp_y"] = simulate(specs["tvp"], 1)[:, 0]

# 5. singular Q and R: quarterly AR(1) in companion form, its four-quarter
#    average observed without error every fourth date, two noisy indicators
specs["mixed"] = dict(
    F=[[0.8, 0, 0, 0], [1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0]],
    G=[[0.25, 0.25, 0.25, 0.25], [2.0, 0, 0, 0], [-1.5, 0, 0, 0]],
    Q=[[0.6, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0], [0, 0, 0, 0]],
    R=[[0, 0, 0], [0, 1.5, 0], [0, 0, 2.5]],
    A=[0.5, 1.0, -2.0],
    x0=[0.0, 0.0, 0.0, 0.0],
    P0=None,
)
ym = simulate(specs["mixed"], 3)
ym[np.arange(T) % 4 != 3, 0] = np.nan
cols["mixed1"], cols["mixed2"], cols["mixed3"] = ym[:, 0], ym[:, 1], ym[:, 2]

# 6. every system matrix time-varying (pins the timing convention)
ang = 0.1 * np.arange(T)
Fv = np.array(
    [[[0.6 + 0.2 * np.sin(a), 0.1], [0.2, 0.4 + 0.3 * np.cos(a)]] for a in ang]
)
Qv = np.array([[[0.5 + 0.3 * np.cos(a) ** 2, 0.1], [0.1, 0.4]] for a in ang])
Rv = np.array([[[0.3 + 0.2 * np.sin(a) ** 2]] for a in ang])
Gv = np.array([[[1.0, np.cos(2 * a)]] for a in ang])
Av = np.array([[0.05 * t] for t in range(T)])
specs["tv"] = dict(
    F=Fv.tolist(),
    G=Gv.tolist(),
    Q=Qv.tolist(),
    R=Rv.tolist(),
    A=Av.tolist(),
    x0=[0.3, -0.2],
    P0=[[1.0, 0.2], [0.2, 2.0]],
)
ytv = simulate(specs["tv"], 1)[:, 0]
ytv[[5, 6, 30]] = np.nan
cols["tv"] = ytv

names = list(cols)
lines = [",".join(names)]
for t in range(T):
    lines.append(
        ",".join("" if np.isnan(cols[c][t]) else repr(float(cols[c][t])) for c in names)
    )
(HERE / "statespace.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
(HERE / "statespace_spec.json").write_text(json.dumps(specs), encoding="utf-8")
