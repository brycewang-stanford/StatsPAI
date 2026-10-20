"""CR2 / HTZ Wald test against R ``clubSandwich`` run as a black box.

``sp.fast.cluster_wald_htz`` / ``cluster_dof_wald_htz`` are a clean-room
implementation of Pustejovsky and Tipton (2018, Section 3.2) (spec:
``docs/dev/cleanroom/htz.md``). Reference: R 4.5.2, ``clubSandwich`` 0.6.2
(GPL-3), public API only (``vcovCR(type = "CR2")`` and
``Wald_test(test = "HTZ", null_constant = r)``; ``eta = df_denom + q - 1``),
on the five designs of ``_fixtures/_generate_htz_blackbox_data.py``:
balanced, unbalanced with singleton clusters, a cluster-level regressor, a
unit-by-period panel with unit and period dummies clustered by unit (where
``I - H_gg`` is singular and the CR2 adjustment needs the Moore-Penrose
inverse), and nine clusters with a high-leverage point; q = 1 to 4 and
non-zero null values (``_fixtures/_generate_htz_blackbox_R.R``).

Tolerances: ``eta``, F and ``V_R`` 1e-8 relative, p-values 1e-7 relative
with a 1e-15 absolute floor (closed-form linear algebra on identical bytes;
observed 3.4e-14 or better on every quantity). A restriction on a
coefficient that is constant within clusters (a unit dummy) is left out:
its CR2 variance is zero up to rounding, so both sides return noise.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import statspai as sp

_FIX = Path(__file__).parent / "_fixtures"
R_OUT = json.loads((_FIX / "htz_blackbox_R.json").read_text(encoding="utf-8"))
DESIGNS = pd.read_csv(_FIX / "htz_blackbox_designs.csv")


def _design(d: int):
    df = DESIGNS[DESIGNS["design"] == d].dropna(axis=1, how="any")
    rhs = [c for c in df.columns if c not in ("design", "g", "y")]
    X = np.column_stack([np.ones(len(df)), df[rhs].to_numpy(dtype=float)])
    y = df["y"].to_numpy(dtype=float)
    beta = np.linalg.lstsq(X, y, rcond=None)[0]
    return X, y - X @ beta, df["g"].to_numpy(), beta


CASES = [(d, j) for d in range(5) for j in range(len(R_OUT[f"d{d}"]["tests"]))]


@pytest.mark.filterwarnings("ignore:HTZ.*singleton:RuntimeWarning")
@pytest.mark.parametrize("d,j", CASES)
def test_htz_matches_clubsandwich_wald_test(d, j):
    ref = R_OUT[f"d{d}"]["tests"][j]
    X, e, g, beta = _design(d)
    np.testing.assert_allclose(beta, R_OUT[f"d{d}"]["beta"], rtol=1e-9, atol=1e-12)
    Rm = np.atleast_2d(np.asarray(ref["R"], dtype=float))
    r = np.atleast_1d(np.asarray(ref["r"], dtype=float))
    res = sp.fast.cluster_wald_htz(X, e, g, R=Rm, beta=beta, r=r)
    assert res.q == ref["q"]
    np.testing.assert_allclose(res.eta, ref["eta"], rtol=1e-8)
    np.testing.assert_allclose(res.F_stat, ref["F_stat"], rtol=1e-8)
    np.testing.assert_allclose(res.p_value, ref["p_value"], rtol=1e-7, atol=1e-15)
    np.testing.assert_allclose(
        res.V_R, np.atleast_2d(np.asarray(ref["V_R"], dtype=float)), rtol=1e-8
    )
    assert sp.fast.cluster_dof_wald_htz(X, g, R=Rm) == pytest.approx(res.eta, rel=1e-12)


def test_q1_eta_is_the_satterthwaite_dof_with_cross_terms():
    # Identity (eq. 13 at q = 1 reduces to eq. 11): with p_i the cluster
    # vectors (I - H)_i' A_i X_i M c / sqrt(Omega), eta = 2 / sum_ij
    # (2 (p_i'p_j)^2) * ... = (sum_i p_i'p_i)^2 / sum_ij (p_i'p_j)^2.
    X, e, g, beta = _design(2)
    c = np.zeros(X.shape[1])
    c[1] = 1.0
    n = len(g)
    M = np.linalg.inv(X.T @ X)
    H = X @ M @ X.T
    IH = np.eye(n) - H
    P = []
    for cl in np.unique(g):
        idx = np.flatnonzero(g == cl)
        B = IH[np.ix_(idx, idx)]
        vals, vecs = np.linalg.eigh(B)
        A = (vecs / np.sqrt(vals)) @ vecs.T
        p = np.zeros(n)
        p[:] = IH[:, idx] @ (A @ (X[idx] @ M @ c))
        P.append(p)
    P = np.array(P)
    gram = P @ P.T
    nu = np.trace(gram) ** 2 / np.sum(gram**2)
    eta = sp.fast.cluster_dof_wald_htz(X, g, R=c.reshape(1, -1))
    assert eta == pytest.approx(nu, rel=1e-10)
