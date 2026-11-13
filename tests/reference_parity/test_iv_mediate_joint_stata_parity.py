"""``sp.iv_mediate(equations="joint")`` against Stata's ``gmm`` + ``nlcom``.

``ivmediate`` (and ``sp.iv_mediate`` by default) sets the covariance
between coefficients of different equations to zero. ``equations="joint"``
keeps it: the three IV equations are one just-identified GMM system and
the variance is its sandwich. Official Stata estimates the same system
with ``gmm (eq1: ...) (eq2: ...) (eq3: ...), onestep winitial(identity)
vce(robust | cluster)`` and ``nlcom`` then gives (total, direct, a * d)
with the delta-method covariance. The reference numbers are in
``_fixtures/iv_mediate_joint_Stata.json``, written by
``_fixtures/_generate_iv_mediate_joint_Stata.do`` (Stata/MP 18) from the
same ``_fixtures/iv_mediate.csv`` the ``ivmediate`` parity test reads.

Tolerances. Coefficients: closed-form IV against ``gmm``'s Gauss-Newton
solution of a linear system, observed <= 3e-15 relative, ``rtol=1e-9``.
Covariance elements: observed <= 2e-12 relative, ``rtol=1e-9``.

The homoskedastic (``robust=False``) joint variance has no Stata
counterpart to compare with: ``gmm , wmatrix(unadjusted)`` has to invert
the stacked moment covariance, which is singular here because the three
residuals satisfy ``e1 = e3 + d e2`` (it then drops the second equation's
coefficient). It is checked below against the same formula written out
with Kronecker products, and by the Monte Carlo rows in
``tests/reliability/se_screen/causal/mediation.py``.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import statspai as sp

FIX = Path(__file__).parent / "_fixtures"
REF = json.loads((FIX / "iv_mediate_joint_Stata.json").read_text(encoding="utf-8"))
RTOL = 1e-9
COV = ["w1", "w2"]
SPECS = {
    "plain_robust": {"robust": True},
    "cov_robust": {"covariates": COV, "robust": True},
    "cov_cluster": {"covariates": COV, "cluster": "cl"},
}


@pytest.fixture(scope="module")
def data():
    return pd.read_csv(FIX / "iv_mediate.csv")


def _fit(data, **kw):
    return sp.iv_mediate(data, y="y", treat="t", mediator="m", instrument="z", **kw)


@pytest.mark.filterwarnings("ignore::statspai.exceptions.AssumptionWarning")
@pytest.mark.parametrize("key", list(SPECS))
def test_joint_variance_matches_stata_gmm(data, key):
    res = _fit(data, equations="joint", **SPECS[key])
    np.testing.assert_allclose(
        res.estimates["coef"].to_numpy(), REF[f"{key}_b"], rtol=RTOL
    )
    V = np.array([REF[f"{key}_V{i}"] for i in (1, 2, 3)])
    np.testing.assert_allclose(res.vcov.to_numpy(), V, rtol=RTOL)
    np.testing.assert_allclose(
        res.estimates["se"].to_numpy(), np.sqrt(np.diag(V)), rtol=RTOL
    )


@pytest.mark.filterwarnings("ignore::statspai.exceptions.AssumptionWarning")
@pytest.mark.parametrize("key", list(SPECS))
def test_joint_changes_only_the_cross_equation_terms(data, key):
    """Same coefficients, same total and direct variances as ``ivmediate``."""
    joint = _fit(data, equations="joint", **SPECS[key])
    indep = _fit(data, **SPECS[key])
    assert joint.model_info["equations"] == "joint"
    assert indep.model_info["equations"] == "independent"
    np.testing.assert_array_equal(
        joint.estimates["coef"].to_numpy(), indep.estimates["coef"].to_numpy()
    )
    for eff in ("total", "direct"):
        assert joint.vcov.loc[eff, eff] == pytest.approx(
            indep.vcov.loc[eff, eff], rel=1e-12
        )
    # ivmediate's zeros
    assert indep.vcov.loc["total", "direct"] == 0.0
    assert indep.vcov.loc["total", "indirect"] == 0.0


@pytest.mark.filterwarnings("ignore::statspai.exceptions.AssumptionWarning")
@pytest.mark.parametrize("kw", [{}, {"robust": True}, {"cluster": "cl"}])
def test_joint_variance_respects_total_equals_direct_plus_indirect(data, kw):
    """With one instrument ``total = direct + indirect`` holds exactly in
    the sample, so the joint covariance must give that difference zero
    variance. Treating the equations as independent cannot."""
    joint = _fit(data, covariates=COV, equations="joint", **kw)
    est = joint.estimates["coef"]
    assert est["total"] == pytest.approx(est["direct"] + est["indirect"], rel=1e-10)
    c = np.array([1.0, -1.0, -1.0])
    scale = float(np.trace(joint.vcov.to_numpy()))
    assert abs(c @ joint.vcov.to_numpy() @ c) <= 1e-9 * scale
    indep = _fit(data, covariates=COV, **kw)
    assert c @ indep.vcov.to_numpy() @ c > 0.1 * scale


def test_homoskedastic_joint_variance_is_the_kronecker_formula(data):
    """``A^{-1} (Sigma (x) instruments) A^{-T}`` written out independently."""
    res = _fit(data, equations="joint")
    d = data[["y", "t", "m", "z"]].to_numpy(float)
    d = d - d.mean(axis=0)  # the constant is partialled out
    y, t, m, z = d.T
    n = len(y)
    X = [t[:, None], t[:, None], np.column_stack([t, m])]
    Z = [z[:, None], z[:, None], np.column_stack([z, t])]
    dep = [y, m, y]
    beta = [np.linalg.solve(Z[j].T @ X[j], Z[j].T @ dep[j]) for j in range(3)]
    resid = np.column_stack([dep[j] - X[j] @ beta[j] for j in range(3)])
    sigma = resid.T @ resid / n
    A = np.zeros((4, 4))
    S = np.zeros((4, 4))
    sl = [slice(0, 1), slice(1, 2), slice(2, 4)]
    for j in range(3):
        A[sl[j], sl[j]] = Z[j].T @ X[j]
        for k in range(3):
            S[sl[j], sl[k]] = sigma[j, k] * (Z[j].T @ Z[k])
    Ainv = np.linalg.inv(A)
    V4 = Ainv @ S @ Ainv.T
    a, dcoef = beta[1][0], beta[2][1]
    J = np.array([[1, 0, 0, 0], [0, 0, 1, 0], [0, dcoef, 0, a]], dtype=float)
    np.testing.assert_allclose(res.vcov.to_numpy(), J @ V4 @ J.T, rtol=1e-10)


def test_unknown_equations_value_is_refused(data):
    from statspai.exceptions import MethodIncompatibility

    with pytest.raises(MethodIncompatibility, match="equations"):
        _fit(data, equations="stacked")
