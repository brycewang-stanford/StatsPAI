"""Reference parity: ``sp.cic`` vs R ``qte::CiC``.

Changes-in-Changes (Athey & Imbens 2006) had no cross-language reference in
the parity index -- ``sp.cic`` was carried as analytical-only. R's ``qte``
package ships ``CiC``, which is a direct implementation of the same estimator,
so it can serve as the reference for both the ATT and the quantile treatment
effects.

Reference generation (R 4.5.2, qte 2.0.0)::

    r <- CiC(y ~ treat, t = 2, tmin1 = 1, tname = "t", data = d,
             panel = TRUE, idname = "id", se = FALSE,
             probs = seq(0.1, 0.9, 0.1))
    r$ate ; r$qte

Fixture: ``_fixtures/cic_qte_panel.csv`` -- 600 units x 2 periods, half
treated, additive effect 2.0 on the treated-post cell.

.. note::
   ``sp.cic`` requires the period column coded 0/1 (it splits the four cells
   on ``t == 0`` / ``t == 1``), while ``qte::CiC`` takes explicit ``t`` and
   ``tmin1`` values. The fixture carries the raw 1/2 coding and the test
   derives ``post``, so the two sides see the same cells.

References
----------
Athey, S. and Imbens, G. W. (2006). "Identification and Inference in
Nonlinear Difference-in-Differences Models." *Econometrica*, 74(2), 431-497.
[@athey2006identification]
"""

from __future__ import annotations

import pathlib
import warnings

import numpy as np
import pandas as pd
import pytest

import statspai as sp

_FIXTURE = pathlib.Path(__file__).parent / "_fixtures" / "cic_qte_panel.csv"

R_ATT = 1.9075672552
R_QTE = {
    0.1: 1.64523063,
    0.2: 1.99402020,
    0.3: 2.07802174,
    0.4: 1.98327028,
    0.5: 2.00294580,
    0.6: 2.04605636,
    0.7: 2.03245592,
    0.8: 1.97001610,
    0.9: 1.61056928,
}


@pytest.fixture(scope="module")
def panel() -> pd.DataFrame:
    if not _FIXTURE.exists():  # pragma: no cover - fixture ships with the repo
        pytest.skip(f"missing fixture: {_FIXTURE}")
    df = pd.read_csv(_FIXTURE)
    df["post"] = (df["t"] == 2).astype(int)
    return df


@pytest.fixture(scope="module")
def fit(panel):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.cic(
            panel,
            y="y",
            group="treat",
            time="post",
            quantiles=sorted(R_QTE),
            n_boot=1,
            seed=0,
        )


def test_cic_att_matches_qte_package(fit):
    assert fit.estimate == pytest.approx(R_ATT, abs=1e-9)


@pytest.mark.parametrize("tau", sorted(R_QTE))
def test_cic_qte_matches_qte_package(fit, tau):
    detail = fit.detail.set_index("quantile")
    got = float(detail.loc[tau, "qte"])
    assert got == pytest.approx(
        R_QTE[tau], abs=1e-7
    ), f"tau={tau}: StatsPAI {got:.8f} vs qte::CiC {R_QTE[tau]:.8f}"


def test_cic_recovers_the_design_effect(fit):
    """The DGP is additive with effect 2.0, so CIC should land near it."""
    assert abs(fit.estimate - 2.0) < 0.15


def test_zero_bootstrap_fails_loudly(panel):
    """CIC has no analytic variance; n_boot=0 used to raise a raw IndexError
    from numpy taking a quantile of an empty array."""
    from statspai.exceptions import MethodIncompatibility

    with pytest.raises(MethodIncompatibility, match="bootstrap"):
        sp.cic(panel, y="y", group="treat", time="post", n_boot=0)


def test_bootstrap_still_produces_standard_errors(panel):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sp.cic(panel, y="y", group="treat", time="post", n_boot=50, seed=0)
    assert np.isfinite(res.se) and res.se > 0
    assert res.estimate == pytest.approx(R_ATT, abs=1e-9)
