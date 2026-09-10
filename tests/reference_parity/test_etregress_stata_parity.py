"""Cross-language parity: ``sp.etregress`` against Stata's ``etregress``.

Fixture: ``_fixtures/_generate_etregress_stata.do`` (Stata 18, official
``etregress``; nothing to install). It generates the data in Stata and
exports it at ``%21.16e``, so both sides read the same bytes.

Evidence structure, which is deliberately in three layers because the
three quantities are pinned to three different tolerances:

1. **``twostep`` is exact** (~5e-9 on every coefficient *and* every
   standard error). Nothing here is approximate.

2. **The MLE's likelihood, score and observed information are exact.**
   Evaluated at *Stata's own* reported parameter vector, our Hessian
   reproduces Stata's standard errors to ~1e-10
   (``test_oim_at_stata_theta_matches_stata_standard_errors``). That is
   the assertion that pins the formula, and it is independent of any
   optimiser.

3. **The MLE's optimiser stops somewhere slightly different**, ~2e-5 on
   the parameters. This is not a formula gap: at our stopping point the
   log-likelihood is *higher* than at Stata's and the gradient is ~300x
   smaller, both asserted in
   ``test_our_optimum_is_at_least_as_good_as_statas``. Stata's
   ``etregress`` stops on its own ``nrtolerance``; neither side is wrong,
   and it is the one place in this file with a budget above 1e-6.

Two regression guards matter as much as the parity itself, because both
defects this file was written to close were *silent*:

* ``method='mle'`` used to run the two-step and label it MLE. The two
  estimators are numerically distinct, so the test asserts they differ.
* ``robust=`` and ``cluster=`` were accepted and never used. The test
  asserts each changes the standard errors.
"""

from __future__ import annotations

import json
import pathlib
import warnings

import numpy as np
import pandas as pd
import pytest

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import statspai as sp

_FIX = pathlib.Path(__file__).parent / "_fixtures"
KW = dict(
    y="wage",
    x=["experience", "education"],
    treatment="union",
    z=["father_union", "region"],
)


@pytest.fixture(scope="module")
def sjson():
    path = _FIX / "etregress_stata.json"
    if not path.exists():  # pragma: no cover
        pytest.skip("run _generate_etregress_stata.do first")
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def data():
    path = _FIX / "etregress_data.csv"
    if not path.exists():  # pragma: no cover
        pytest.skip("run _generate_etregress_stata.do first")
    return pd.read_csv(path)


def _fit(data, **kw):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.etregress(data, **KW, **kw)


# ── 1. two-step: exact ─────────────────────────────────────────────────────

TWOSTEP_COEFS = [
    ("_cons", "cons"),
    ("experience", "experience"),
    ("education", "education"),
    ("union", "treat"),
    ("hazard:lambda", "hazard"),
]


@pytest.mark.parametrize("name,key", TWOSTEP_COEFS)
def test_twostep_coefficients_match_stata(sjson, data, name, key):
    res = _fit(data, method="twostep")
    assert float(res.params[name]) == pytest.approx(sjson["twostep"][key], rel=1e-6)


@pytest.mark.parametrize(
    "name,key",
    [
        ("_cons", "se_cons"),
        ("experience", "se_experience"),
        ("union", "se_treat"),
        ("hazard:lambda", "se_hazard"),
    ],
)
def test_twostep_standard_errors_match_stata(sjson, data, name, key):
    """The corrected covariance, not the OLS one.

    ``sp.etregress`` reported the naive OLS standard errors of the
    hazard-augmented regression, which ignore that the hazard is itself
    estimated. On this design they are ~11% too small — anti-conservative,
    the direction that manufactures significance.
    """
    res = _fit(data, method="twostep")
    assert float(res.std_errors[name]) == pytest.approx(sjson["twostep"][key], rel=1e-6)


def test_twostep_uncorrected_ols_se_would_be_visibly_wrong(sjson, data):
    """Quantifies what the correction is worth, so a silent removal fails."""
    res = _fit(data, method="twostep")
    n = len(data)
    hazard = None
    # Rebuild the naive OLS standard error of the same augmented design.
    W = np.column_stack(
        [
            np.ones(n),
            data[["experience", "education"]].to_numpy(float),
            data["union"].to_numpy(float),
        ]
    )
    from scipy import stats as _st

    from statspai.regression.selection import _probit_coefficients

    Z = np.column_stack([np.ones(n), data[["father_union", "region"]].to_numpy(float)])
    gamma = _probit_coefficients(Z, data["union"].to_numpy(float))
    zg = Z @ gamma
    hazard = np.where(
        data["union"].to_numpy(float) == 1,
        np.exp(_st.norm.logpdf(zg) - _st.norm.logcdf(zg)),
        -np.exp(_st.norm.logpdf(zg) - _st.norm.logcdf(-zg)),
    )
    Wa = np.column_stack([W, hazard])
    beta = np.linalg.lstsq(Wa, data["wage"].to_numpy(float), rcond=None)[0]
    resid = data["wage"].to_numpy(float) - Wa @ beta
    s2 = resid @ resid / (n - Wa.shape[1])
    naive = np.sqrt(np.diag(s2 * np.linalg.inv(Wa.T @ Wa)))[3]
    assert float(res.std_errors["union"]) == pytest.approx(
        sjson["twostep"]["se_treat"], rel=1e-6
    )
    assert (
        abs(naive - sjson["twostep"]["se_treat"]) / sjson["twostep"]["se_treat"] > 0.05
    )


# ── 2. the MLE formula, pinned independently of the optimiser ─────────────


def test_oim_at_stata_theta_matches_stata_standard_errors(sjson, data):
    """Likelihood + score + observed information, evaluated at Stata's theta.

    This is the assertion that pins the *formula*. It cannot be satisfied
    by an estimator that merely lands nearby: the Hessian is differenced
    from the analytic score at a parameter vector StatsPAI never chose.
    """
    from statspai.regression.selection import _etregress_hessian

    ref = sjson["mle"]
    n = len(data)
    yv = data["wage"].to_numpy(float)
    D = data["union"].to_numpy(float)
    W = np.column_stack(
        [np.ones(n), data[["experience", "education"]].to_numpy(float), D]
    )
    Z = np.column_stack([np.ones(n), data[["father_union", "region"]].to_numpy(float)])
    theta = np.array(
        [
            ref["cons"],
            ref["experience"],
            ref["education"],
            ref["treat"],
            ref["sel_cons"],
            ref["sel_father_union"],
            ref["sel_region"],
            ref["athrho"],
            ref["lnsigma"],
        ]
    )
    se = np.sqrt(np.diag(np.linalg.inv(_etregress_hessian(theta, yv, W, Z, D))))
    for idx, key in (
        (0, "se_cons"),
        (1, "se_experience"),
        (2, "se_education"),
        (3, "se_treat"),
    ):
        assert se[idx] == pytest.approx(ref[key], rel=1e-8)
    assert se[5] == pytest.approx(ref["se_sel_father_union"], rel=1e-8)


def test_our_optimum_is_at_least_as_good_as_statas(sjson, data):
    """Why the MLE budget below is 1e-4 rather than 1e-6.

    The residual parameter difference is where the two optimisers stopped.
    Ours stops at a higher log-likelihood with a much smaller gradient, so
    the gap is Stata's ``nrtolerance``, not a defect on our side. If a
    future change makes our optimum *worse* than Stata's, this fails even
    though the loose parity assertions would still pass.
    """
    from statspai.regression.selection import _etregress_scores

    ref = sjson["mle"]
    n = len(data)
    yv = data["wage"].to_numpy(float)
    D = data["union"].to_numpy(float)
    W = np.column_stack(
        [np.ones(n), data[["experience", "education"]].to_numpy(float), D]
    )
    Z = np.column_stack([np.ones(n), data[["father_union", "region"]].to_numpy(float)])
    theta_stata = np.array(
        [
            ref["cons"],
            ref["experience"],
            ref["education"],
            ref["treat"],
            ref["sel_cons"],
            ref["sel_father_union"],
            ref["sel_region"],
            ref["athrho"],
            ref["lnsigma"],
        ]
    )
    ll_stata, score_stata = _etregress_scores(theta_stata, yv, W, Z, D)
    res = _fit(data, method="mle")
    ours = res.params.to_numpy(float)
    ll_ours, score_ours = _etregress_scores(ours, yv, W, Z, D)

    assert ll_ours >= ll_stata
    assert np.max(np.abs(score_ours.sum(0))) < np.max(np.abs(score_stata.sum(0)))
    # And the two log-likelihoods still agree to far tighter than the
    # parameter budget, which is what "flat surface" means here.
    assert ll_ours == pytest.approx(ref["ll"], rel=1e-10)


@pytest.mark.parametrize(
    "name,key",
    [
        ("_cons", "cons"),
        ("experience", "experience"),
        ("education", "education"),
        ("union", "treat"),
        ("union:father_union", "sel_father_union"),
        ("union:region", "sel_region"),
    ],
)
def test_mle_coefficients_match_stata(sjson, data, name, key):
    res = _fit(data, method="mle")
    assert float(res.params[name]) == pytest.approx(sjson["mle"][key], rel=1e-4)


@pytest.mark.parametrize(
    "name,key",
    [
        ("_cons", "se_cons"),
        ("experience", "se_experience"),
        ("union", "se_treat"),
        ("union:father_union", "se_sel_father_union"),
    ],
)
def test_mle_standard_errors_match_stata(sjson, data, name, key):
    res = _fit(data, method="mle")
    assert float(res.std_errors[name]) == pytest.approx(sjson["mle"][key], rel=1e-4)


def test_mle_rho_sigma_lambda_match_stata(sjson, data):
    res = _fit(data, method="mle")
    ref = sjson["mle"]
    assert res.diagnostics["rho"] == pytest.approx(ref["rho"], rel=1e-4)
    assert res.diagnostics["sigma"] == pytest.approx(ref["sigma"], rel=1e-4)
    assert res.diagnostics["lambda"] == pytest.approx(ref["lambda"], rel=1e-4)


# ── 3. robust and cluster: arguments that used to be ignored ──────────────


@pytest.mark.parametrize(
    "name,key",
    [
        ("_cons", "se_cons"),
        ("experience", "se_experience"),
        ("union", "se_treat"),
        ("union:father_union", "se_sel_father_union"),
    ],
)
def test_mle_robust_standard_errors_match_stata(sjson, data, name, key):
    res = _fit(data, method="mle", robust="robust")
    assert float(res.std_errors[name]) == pytest.approx(
        sjson["mle_robust"][key], rel=1e-4
    )


@pytest.mark.parametrize(
    "name,key",
    [
        ("_cons", "se_cons"),
        ("experience", "se_experience"),
        ("union", "se_treat"),
        ("union:father_union", "se_sel_father_union"),
    ],
)
def test_mle_cluster_standard_errors_match_stata(sjson, data, name, key):
    res = _fit(data, method="mle", robust="cluster", cluster="clust")
    assert float(res.std_errors[name]) == pytest.approx(
        sjson["mle_cluster"][key], rel=1e-4
    )


# ── 4. regression guards for the two silent defects ──────────────────────


def test_mle_and_twostep_are_different_estimators(data):
    """``method='mle'`` used to be a verbatim copy of the two-step branch.

    On this design the difference in the treatment effect is ~2%; on the
    first fixture it was 10.7%. Either way, byte-identical output from the
    two methods is the signature of the defect.
    """
    mle = _fit(data, method="mle")
    ts = _fit(data, method="twostep")
    assert float(mle.params["union"]) != pytest.approx(
        float(ts.params["union"]), rel=1e-8
    )
    assert float(mle.std_errors["union"]) != pytest.approx(
        float(ts.std_errors["union"]), rel=1e-8
    )


def test_robust_and_cluster_actually_change_the_variance(data):
    """Both arguments were accepted and then never referenced again."""
    plain = _fit(data, method="mle")
    rob = _fit(data, method="mle", robust="robust")
    clu = _fit(data, method="mle", robust="cluster", cluster="clust")
    se_p = float(plain.std_errors["union"])
    assert float(rob.std_errors["union"]) != pytest.approx(se_p, rel=1e-6)
    assert float(clu.std_errors["union"]) != pytest.approx(se_p, rel=1e-6)
    assert float(clu.std_errors["union"]) != pytest.approx(
        float(rob.std_errors["union"]), rel=1e-6
    )


def test_cluster_without_column_is_refused(data):
    with pytest.raises(ValueError, match="requires cluster"):
        _fit(data, method="mle", robust="cluster")


def test_unknown_method_is_refused(data):
    with pytest.raises(ValueError, match="method must be"):
        _fit(data, method="gmm")
