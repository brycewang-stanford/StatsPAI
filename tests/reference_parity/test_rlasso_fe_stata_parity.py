"""Fixed-effects rigorous Lasso against Stata's ``rlasso, fe`` / ``pdslasso, fe``.

``sp.rlasso(absorb=)`` and ``sp.rlasso_effect(absorb=)`` are compared with
lassopack's ``rlasso ..., fe`` and pdslasso's ``pdslasso y d (x...), fe``
(Ahrens, Hansen and Schaffer) on the same CSV bytes. Reference numbers are
in ``_fixtures/rlasso_fe_Stata.json``, written by
``_fixtures/_generate_rlasso_fe_Stata.do`` (Stata/MP 18) from
``_fixtures/rlasso_fe_panel.csv`` (unbalanced panel, 70 units, 418 rows, 30
controls; ``_generate_rlasso_fe_data.py``). Two-way fixed effects are
compared with ``fe partial(i.year)``.

Tolerances. The supports must agree exactly. Given the support, the
coefficients and variances are closed-form OLS quantities on the same
within-transformed data, so they agree to rounding (observed <= 3e-15);
``rtol=1e-9`` leaves room for BLAS summation order across platforms. The
penalty level is a closed-form expression of ``n``, ``p`` and ``G``
(``rel=1e-12``).
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.exceptions import DataInsufficient, MethodIncompatibility

FIX = Path(__file__).parent / "_fixtures"
REF = json.loads((FIX / "rlasso_fe_Stata.json").read_text(encoding="utf-8"))
XS = [f"x{j + 1}" for j in range(30)]
RTOL = 1e-9


@pytest.fixture(scope="module")
def panel():
    return pd.read_csv(FIX / "rlasso_fe_panel.csv")


@pytest.mark.parametrize("var", ["d", "y"])
@pytest.mark.parametrize("vce", ["robust", "cluster"])
def test_rlasso_fe_matches_lassopack(panel, var, vce):
    key = f"rlasso_{var}_{vce}"
    fit = sp.rlasso(
        panel[XS].to_numpy(),
        panel[var].to_numpy(),
        colnames=XS,
        absorb=panel["id"].to_numpy(),
        cluster=panel["id"].to_numpy() if vce == "cluster" else None,
    )
    assert fit.lambda0 == pytest.approx(REF[f"{key}_lambda0"], rel=1e-12)
    assert fit.selected == REF[f"{key}_selected"]
    np.testing.assert_allclose(fit.beta[fit.index], REF[f"{key}_betaOLS"], rtol=RTOL)
    assert fit.absorb_levels == (70,)
    assert fit.iteration == "lassopack"
    if vce == "robust":
        assert fit.iter == REF[f"{key}_npsiiter"]


# (Stata option set, rlasso_effect keyword arguments)
_VCE = {
    "robust": {"robust": True},
    "cluster": {"cluster": "id"},
    # loptions(robust): heteroskedastic penalty, homoskedastic final variance
    "loptions": {"robust": False},
}


@pytest.mark.parametrize("opt", list(_VCE))
@pytest.mark.parametrize(
    "method, key", [("double selection", "pds"), ("partialling out", "plasso")]
)
def test_effect_matches_pdslasso_fe(panel, opt, method, key):
    res = sp.rlasso_effect(
        XS, "y", "d", method=method, data=panel, absorb="id", **_VCE[opt]
    )
    assert res.alpha == pytest.approx(REF[f"{key}_{opt}_beta"], rel=RTOL)
    assert res.se**2 == pytest.approx(REF[f"{key}_{opt}_V"], rel=RTOL)
    if method == "double selection":
        sel = [c for c, s in zip(XS, res.selection_index) if s]
        assert sel == REF[f"pds_{opt}_xselected"]
    assert res.absorb_levels == (70,)


def test_default_variances_under_absorb(panel):
    """``robust=None``: HC0 for double selection, ``e'e/n`` for partialling out."""
    ds = sp.rlasso_effect(
        XS, "y", "d", method="double selection", data=panel, absorb="id"
    )
    assert ds.se**2 == pytest.approx(REF["pds_robust_V"], rel=RTOL)
    po = sp.rlasso_effect(
        XS, "y", "d", method="partialling out", data=panel, absorb="id"
    )
    assert po.se**2 == pytest.approx(REF["plasso_loptions_V"], rel=RTOL)


@pytest.mark.parametrize("opt", ["robust", "cluster"])
@pytest.mark.parametrize(
    "method, key", [("double selection", "pds"), ("partialling out", "plasso")]
)
def test_two_way_matches_fe_partial_year(panel, opt, method, key):
    res = sp.rlasso_effect(
        XS, "y", "d", method=method, data=panel, absorb=["id", "year"], **_VCE[opt]
    )
    assert res.alpha == pytest.approx(REF[f"tw_{key}_{opt}_beta"], rel=RTOL)
    assert res.se**2 == pytest.approx(REF[f"tw_{key}_{opt}_V"], rel=RTOL)
    assert res.absorb_levels == (70, 8)


def test_two_way_rlasso_matches_fe_partial_year(panel):
    fit = sp.rlasso(
        panel[XS].to_numpy(),
        panel["d"].to_numpy(),
        colnames=XS,
        absorb=panel[["id", "year"]].to_numpy(),
    )
    assert fit.selected == REF["tw_rlasso_d_robust_selected"]
    # Stata appends the recovered year coefficients after the selected ones.
    k = fit.n_selected
    np.testing.assert_allclose(
        fit.beta[fit.index], REF["tw_rlasso_d_robust_betaOLS"][:k], rtol=RTOL
    )


def test_lassopack_iteration_reaches_lassopacks_fixed_point():
    """The panel on which hdm's iteration and lassopack's stop at different
    supports (see ``test_cluster_lasso_stata.py``): with
    ``iteration="lassopack"`` StatsPAI reaches lassopack's support."""
    ref = json.loads((FIX / "cluster_lasso_Stata.json").read_text(encoding="utf-8"))
    df = pd.read_csv(FIX / "cluster_lasso_panel_path.csv")
    cols = [f"z{j + 1}" for j in range(16)] + [f"x{j + 1}" for j in range(24)]
    kw = dict(intercept=False, colnames=cols, cluster=df["id"].to_numpy())
    hdm = sp.rlasso(df[cols].to_numpy(), df["d"].to_numpy(), **kw)
    lp = sp.rlasso(
        df[cols].to_numpy(),
        df["d"].to_numpy(),
        control={"iteration": "lassopack"},
        **kw,
    )
    assert hdm.selected != ref["path"]["selected"]
    assert lp.selected == ref["path"]["selected"]
    np.testing.assert_allclose(lp.beta[lp.index], ref["path"]["betaOLS"], rtol=RTOL)


def test_absorb_is_within_transform_without_intercept(panel):
    """``absorb=`` equals hand-demeaning plus ``intercept=False`` (the
    identity Stata shows between ``fe`` and ``nocons`` on demeaned data)."""
    cols = ["d"] + XS
    w = panel[cols] - panel.groupby("id")[cols].transform("mean")
    a = sp.rlasso(panel[XS].to_numpy(), panel["d"].to_numpy(), absorb=panel["id"])
    b = sp.rlasso(
        w[XS].to_numpy(),
        w["d"].to_numpy(),
        intercept=False,
        control={"iteration": "lassopack"},
    )
    np.testing.assert_array_equal(a.index, b.index)
    np.testing.assert_allclose(a.beta, b.beta, rtol=0, atol=1e-12)


def test_rlasso_effects_with_absorb_is_per_target(panel):
    out = sp.rlasso_effects(
        panel[["d"] + XS[:10]], "y", index=[0], data=panel, absorb="id"
    )
    one = sp.rlasso_effect(XS[:10], "y", "d", data=panel, absorb="id")
    assert out["d"].alpha == pytest.approx(one.alpha, rel=1e-12)
    assert out["d"].se == pytest.approx(one.se, rel=1e-12)


def test_absorb_argument_is_validated(panel):
    X, y = panel[XS].to_numpy(), panel["y"].to_numpy()
    with pytest.raises(MethodIncompatibility, match="needs `data`"):
        sp.rlasso_effect(X, y, panel["d"].to_numpy(), absorb="id")
    with pytest.raises(MethodIncompatibility, match="not in data"):
        sp.rlasso_effect(XS, "y", "d", data=panel, absorb="firm")
    with pytest.raises(MethodIncompatibility, match="one entry per row"):
        sp.rlasso(X, y, absorb=np.arange(5))
    ids = panel["id"].to_numpy().astype(float)
    ids[3] = np.nan
    with pytest.raises(DataInsufficient, match="missing"):
        sp.rlasso(X, y, absorb=ids)
    with pytest.raises(MethodIncompatibility, match="heteroskedastic and cluster"):
        sp.rlasso(X, y, absorb=panel["id"], penalty={"homoscedastic": True})
    with pytest.raises(MethodIncompatibility, match="iteration"):
        sp.rlasso(X, y, control={"iteration": "glmnet"})


def test_hdm_iteration_can_still_be_chosen_with_absorb(panel):
    fit = sp.rlasso(
        panel[XS].to_numpy(),
        panel["d"].to_numpy(),
        absorb=panel["id"],
        control={"iteration": "hdm"},
    )
    assert fit.iteration == "hdm"
    # Default numIter for hdm is 15; lassopack's is 2.
    assert fit.iter > 2
