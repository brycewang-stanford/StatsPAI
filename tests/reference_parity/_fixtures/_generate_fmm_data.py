"""Simulated data for the ``sp.fmm`` reference tests (Stata ``fmm``, R
``flexmix``).

Every design has well-separated classes, so the likelihood has one
maximum up to relabelling and Stata, R and StatsPAI must reach the same
point. ``cls`` is the simulated class, numbered in StatsPAI's default order
(ascending class-specific mean of the outcome); the Stata do-file starts
from it (``startvalues(classid cls)``) so Stata's labels match ours.
``g`` is a cluster id, ``fw`` an integer weight.

Run from the repository root::

    python tests/reference_parity/_fixtures/_generate_fmm_data.py
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd

HERE = pathlib.Path(__file__).parent


def _membership(rng: np.random.Generator, z: np.ndarray, gam: np.ndarray) -> np.ndarray:
    """Class draws from a multinomial logit in (1, z); class 1 is the base."""
    eta = np.column_stack([np.zeros(len(z))] + [g[0] + g[1] * z for g in gam])
    p = np.exp(eta - eta.max(axis=1, keepdims=True))
    p /= p.sum(axis=1, keepdims=True)
    u = rng.uniform(size=len(z))
    return 1 + (u[:, None] > np.cumsum(p, axis=1)).sum(axis=1)


def _common(rng: np.random.Generator, n: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "x1": rng.normal(size=n),
            "x2": rng.uniform(-1.0, 1.0, size=n),
            "z": rng.normal(size=n),
            "g": rng.integers(1, 81, size=n),
            "fw": rng.integers(1, 4, size=n),
        }
    )


def gauss2(rng: np.random.Generator) -> pd.DataFrame:
    d = _common(rng, 800)
    c = _membership(rng, d["z"].to_numpy(), np.array([[-0.3, 1.0]]))
    b = {1: (1.0, 2.0, 0.5, 1.0), 2: (7.0, -1.0, 1.0, 1.5)}
    mu = np.array(
        [b[k][0] + b[k][1] * a + b[k][2] * e for k, a, e in zip(c, d.x1, d.x2)]
    )
    sd = np.array([b[k][3] for k in c])
    d["y"] = mu + sd * rng.normal(size=len(d))
    d["cls"] = c
    return d


def gauss3(rng: np.random.Generator) -> pd.DataFrame:
    d = _common(rng, 1500)
    c = _membership(rng, d["z"].to_numpy(), np.array([[0.2, 0.8], [-0.2, -1.0]]))
    b = {1: (-4.0, 1.0, 0.8), 2: (2.0, -1.5, 1.0), 3: (9.0, 0.5, 1.2)}
    mu = np.array([b[k][0] + b[k][1] * a for k, a in zip(c, d.x1)])
    sd = np.array([b[k][2] for k in c])
    d["y"] = mu + sd * rng.normal(size=len(d))
    d["cls"] = c
    return d


def logit2(rng: np.random.Generator) -> pd.DataFrame:
    d = _common(rng, 4000)
    d["x1"] = 1.5 * d["x1"]
    c = _membership(rng, d["z"].to_numpy(), np.array([[0.2, 1.2]]))
    b = {1: (-2.5, 3.0, 0.0), 2: (2.0, -3.0, 1.5)}
    eta = np.array(
        [b[k][0] + b[k][1] * a + b[k][2] * e for k, a, e in zip(c, d.x1, d.x2)]
    )
    d["y"] = (rng.uniform(size=len(d)) < 1.0 / (1.0 + np.exp(-eta))).astype(int)
    d["cls"] = c
    return d


def pois2(rng: np.random.Generator) -> pd.DataFrame:
    d = _common(rng, 1500)
    c = _membership(rng, d["z"].to_numpy(), np.array([[0.4, -1.0]]))
    b = {1: (0.3, 0.4, -0.3), 2: (2.8, -0.5, 0.3)}
    eta = np.array(
        [b[k][0] + b[k][1] * a + b[k][2] * e for k, a, e in zip(c, d.x1, d.x2)]
    )
    d["y"] = rng.poisson(np.exp(eta))
    d["cls"] = c
    return d


def main() -> None:
    for i, (name, fn) in enumerate(
        [("gauss2", gauss2), ("gauss3", gauss3), ("logit2", logit2), ("pois2", pois2)]
    ):
        rng = np.random.default_rng(20261028 + i)
        d = fn(rng)
        cols = ["y", "x1", "x2", "z", "g", "fw", "cls"]
        d[cols].to_csv(HERE / f"fmm_{name}.csv", index=False, float_format="%.17g")


if __name__ == "__main__":
    main()
