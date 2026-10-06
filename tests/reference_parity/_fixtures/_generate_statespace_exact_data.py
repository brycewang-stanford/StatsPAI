"""Simulated data and system matrices for the exact diffuse parity tests.

Writes ``statespace_exact.csv`` (observations of six models, empty where
missing) and ``statespace_exact_spec.json`` (system matrices and the states
marked diffuse), read by ``test_statespace_parity.py`` and by
``_generate_statespace_exact_R.R``.

    python tests/reference_parity/_fixtures/_generate_statespace_exact_data.py

Model: X_t = F_t X_{t-1} + V_t, Var V_t = Q_t; Y_t = A_t + G_t X_t + W_t,
Var W_t = R_t. ``diffuse`` flags the states of X_0 with infinite variance;
the others start from their stationary distribution.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
T = 60
rng = np.random.default_rng(2026100603)


def simulate(spec: dict, n: int, start: list) -> np.ndarray:
    """One path of the observables; time-varying entries carry axis 0."""

    def at(key: str, t: int, ndim: int) -> np.ndarray:
        arr = np.asarray(spec[key], dtype=float)
        return arr[t] if arr.ndim == ndim + 1 else arr

    m = np.asarray(spec["F"]).shape[-1]
    x = np.asarray(start, dtype=float)
    out = np.zeros((T, n))
    for t in range(T):
        x = at("F", t, 2) @ x + rng.multivariate_normal(np.zeros(m), at("Q", t, 2))
        noise = rng.multivariate_normal(np.zeros(n), at("R", t, 2))
        out[t] = at("A", t, 1) + at("G", t, 2) @ x + noise
    return out


specs: dict = {}
cols: dict = {}

# 1. local level
specs["level"] = dict(
    F=[[1.0]], G=[[1.0]], Q=[[0.3]], R=[[1.2]], A=[0.0], x0=[0.0], diffuse=[True]
)
cols["level"] = simulate(specs["level"], 1, [5.0])[:, 0]

# 2. local linear trend: two diffuse states
specs["trend"] = dict(
    F=[[1.0, 1.0], [0.0, 1.0]],
    G=[[1.0, 0.0]],
    Q=[[0.2, 0.0], [0.0, 0.01]],
    R=[[0.8]],
    A=[0.0],
    x0=[0.0, 0.0],
    diffuse=[True, True],
)
cols["trend"] = simulate(specs["trend"], 1, [10.0, 0.3])[:, 0]

# 3. regression with random-walk coefficients (time-varying G), y missing
#    at dates 2 and 3 so that the diffuse period is stretched
xreg = rng.normal(size=T)
specs["tvp"] = dict(
    F=[[1.0, 0.0], [0.0, 1.0]],
    G=[[[1.0, float(v)]] for v in xreg],
    Q=[[0.05, 0.0], [0.0, 0.02]],
    R=[[0.3]],
    A=[0.0],
    x0=[0.0, 0.0],
    diffuse=[True, True],
)
ytvp = simulate(specs["tvp"], 1, [2.0, -1.0])[:, 0]
ytvp[[1, 2, 20]] = np.nan
cols["tvp_x"] = xreg
cols["tvp_y"] = ytvp

# 4. one diffuse random walk plus one stationary AR(1)
specs["mixed"] = dict(
    F=[[1.0, 0.0], [0.0, 0.7]],
    G=[[1.0, 1.0]],
    Q=[[0.1, 0.0], [0.0, 0.5]],
    R=[[0.2]],
    A=[0.5],
    x0=[0.0, 0.0],
    diffuse=[True, False],
)
cols["mixed"] = simulate(specs["mixed"], 1, [3.0, 0.0])[:, 0]

# 5. two observables, correlated measurement errors, two diffuse levels and
#    a stationary common component, missing values in the first periods
specs["biv"] = dict(
    F=[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 0.6]],
    G=[[1.0, 0.0, 1.0], [0.0, 1.0, -0.5]],
    Q=[[0.2, 0.05, 0.0], [0.05, 0.1, 0.0], [0.0, 0.0, 0.4]],
    R=[[0.5, 0.2], [0.2, 0.8]],
    A=[1.0, -1.0],
    x0=[0.0, 0.0, 0.0],
    diffuse=[True, True, False],
)
yb = simulate(specs["biv"], 2, [4.0, -2.0, 0.0])
yb[0, 0] = np.nan
yb[1, :] = np.nan
yb[2, 1] = np.nan
yb[3, 1] = np.nan
yb[[15, 16], 0] = np.nan
yb[30, :] = np.nan
cols["biv1"], cols["biv2"] = yb[:, 0], yb[:, 1]

# 6. every system matrix time-varying, both states diffuse, det F_1 != 1
ang = 0.1 * np.arange(T) + 0.3
Fv = np.array(
    [[[0.9 + 0.2 * np.sin(a), 0.1], [0.2, 0.8 + 0.3 * np.cos(a)]] for a in ang]
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
    x0=[0.0, 0.0],
    diffuse=[True, True],
)
ytv = simulate(specs["tv"], 1, [1.0, -1.0])[:, 0]
ytv[[0, 5, 30]] = np.nan
cols["tv"] = ytv

names = list(cols)
lines = [",".join(names)]
for t in range(T):
    lines.append(
        ",".join("" if np.isnan(cols[c][t]) else repr(float(cols[c][t])) for c in names)
    )
(HERE / "statespace_exact.csv").write_text("\n".join(lines) + "\n", encoding="utf-8")
(HERE / "statespace_exact_spec.json").write_text(json.dumps(specs), encoding="utf-8")
