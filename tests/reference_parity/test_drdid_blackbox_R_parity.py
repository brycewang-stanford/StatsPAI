"""Clean-room DR-DiD engines against R ``DRDID`` 1.3.0 used as a black box.

``did/_rcs.py`` and the panel 2x2 engines of ``did/callaway_santanna.py``
were rewritten on 2026-10-20 from Sant'Anna and Zhao (2020) and the DRDID
documentation (``docs/dev/cleanroom/did_drdid.md``); DRDID's source (GPL-3)
was not read. This module is the acceptance evidence: 78 estimator x design x
weighting combinations on six designs (one with comparison units trimmed at
0.995), plus four rows on a seventh design whose cell regressions are
singular. Each row checks the ATT, the SE and five projections
``mean(IF * v)`` of the influence function.

Tolerances. The outcome-regression and TWFE rows involve only least squares
and agree to ~1e-14. Every other row inherits R's logit fit, which ``glm``
stops at a relative deviance change of 1e-8 (``?glm.control``); the observed
gap is up to 5e-8 relative, so the budget is 2e-7. The improved estimators use
our exact tilting solution against DRDID's trust-region one (observed 7e-9).

Fixture: ``_generate_drdid_blackbox_data.py`` (designs) and
``_generate_drdid_blackbox_R.R`` (R 4.5.2, DRDID 1.3.0).

References
----------
[@santanna2020doubly]
"""

from __future__ import annotations

import importlib
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from statspai.did import _rcs
from statspai.did._dr_common import (
    HajekMean,
    design_matrix,
    logit_fit,
    normalized_weights,
    wls_coef,
    wls_influence,
)

# ``statspai.did.callaway_santanna`` is shadowed by the function of that name.
cs = importlib.import_module("statspai.did.callaway_santanna")

_FIX = Path(__file__).parent / "_fixtures"
_REF = pd.read_csv(_FIX / "drdid_blackbox_R.csv")

_EXACT = {"reg_did_panel", "reg_did_rc", "twfe_did_rc"}
_RTOL_EXACT = 1e-10
_RTOL_LOGIT = 2e-7


def _rc_runs(df: pd.DataFrame, weights):
    x = df[["x1", "x2"]].to_numpy()
    args = (df["y"].to_numpy(), df["post"].to_numpy(), df["D"].to_numpy(), x, weights)
    return {
        # R's convention for the pre-period comparison-regression term
        # (documented T4 divergence; the default 'delta' is the paper's).
        "drdid_rc": lambda: _rcs.drdid_rc(*args, variance_convention="reference"),
        "drdid_rc1": lambda: _rcs.drdid_rc(
            *args, locally_efficient=False, variance_convention="reference"
        ),
        "drdid_imp_rc": lambda: _rcs.drdid_imp_rc(*args),
        "drdid_imp_rc1": lambda: _rcs.drdid_imp_rc(*args, locally_efficient=False),
        "std_ipw_did_rc": lambda: _rcs.std_ipw_did_rc(*args),
        "ipw_did_rc": lambda: _rcs.ipw_did_rc(*args),
        "reg_did_rc": lambda: _rcs.reg_did_rc(*args),
        "twfe_did_rc": lambda: _rcs.twfe_did_rc(*args),
    }


def _panel_runs(df: pd.DataFrame, weights):
    dy = (df["y1"] - df["y0"]).to_numpy()
    d = df["D"].to_numpy(dtype=float)
    x = df[["x1", "x2"]].to_numpy()
    w = None if weights is None else weights / weights.mean()
    trim = 0.995
    return {
        "drdid_panel": lambda: cs._dr_att(dy, d, x, trim, w),
        "drdid_imp_panel": lambda: cs._dr_imp_att(dy, d, x, trim, w),
        "std_ipw_did_panel": lambda: cs._ipw_att(dy, d, x, trim, w),
        "ipw_did_panel": lambda: cs._ipw_abadie_att(dy, d, x, trim, w),
        "reg_did_panel": lambda: cs._reg_att(dy, d, x, w),
    }


def _cases():
    for row in _REF.itertuples(index=False):
        yield pytest.param(row, id=f"{row.design}-{row.weighting}-{row.fn}")


@pytest.mark.parametrize("ref", list(_cases()))
def test_engine_matches_drdid_black_box(ref):
    df = pd.read_csv(_FIX / f"drdid_blackbox_{ref.design}.csv")
    weights = df["w"].to_numpy() if ref.weighting == "w" else None
    runs = (_rc_runs if ref.design.startswith("rc") else _panel_runs)(df, weights)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = runs[ref.fn]()
    att, se, inf = (out.att, out.se, out.influence) if hasattr(out, "att") else out
    rtol = _RTOL_EXACT if ref.fn in _EXACT else _RTOL_LOGIT
    assert att == pytest.approx(ref.att, rel=rtol)
    assert se == pytest.approx(ref.se, rel=rtol)

    y = df["y"] if "y" in df else df["y1"] - df["y0"]
    probes = {
        "m_one": np.ones(len(df)),
        "m_x1": df["x1"].to_numpy(),
        "m_x2": df["x2"].to_numpy(),
        "m_d": df["D"].to_numpy(dtype=float),
        "m_y": y.to_numpy(dtype=float),
    }
    scale = float(np.sqrt(np.mean(np.asarray(inf) ** 2)))
    for name, v in probes.items():
        assert float(np.mean(inf * v)) == pytest.approx(
            getattr(ref, name), abs=rtol * scale * 10
        ), name


def test_fixture_covers_every_engine_and_trims():
    fns = set(_REF["fn"])
    assert len(_REF) == 82 and len(fns) == 13
    df = pd.read_csv(_FIX / "drdid_blackbox_rc_3.csv")
    x = design_matrix(df[["x1", "x2"]].to_numpy(), len(df))
    _, p = logit_fit(df["D"].to_numpy(), x, np.ones(len(df)))
    assert int(((df["D"] == 0) & (p >= 0.995)).sum()) > 5


@pytest.mark.parametrize("weighting", ["none", "w"])
def test_pre_period_regression_convention_identity(weighting):
    """``'delta'`` and ``'reference'`` differ only in the sign of one term.

    The pre-period comparison-regression estimation effect enters the
    influence function as ``s * l_00' (M_10 - M_00)`` with ``s = +1``
    (delta method, Sant'Anna-Zhao Appendix B) or ``s = -1`` (the reference
    implementation), where ``l_00`` is that regression's influence function
    and ``M_dt`` the arm-weighted covariate means of the pre-period cells.
    So the two influence functions differ by exactly ``2 l_00' (M_10 - M_00)``
    and the point estimate is identical.
    """
    df = pd.read_csv(_FIX / "drdid_blackbox_rc_1.csv")
    y, post, d = df["y"].to_numpy(), df["post"].to_numpy(), df["D"].to_numpy()
    cov = df[["x1", "x2"]].to_numpy()
    raw = df["w"].to_numpy() if weighting == "w" else None
    for efficient in (True, False):
        ref = _rcs.drdid_rc(
            y,
            post,
            d,
            cov,
            raw,
            locally_efficient=efficient,
            variance_convention="reference",
        )
        dlt = _rcs.drdid_rc(y, post, d, cov, raw, locally_efficient=efficient)
        assert dlt.att == ref.att

        n = len(df)
        w = normalized_weights(raw, n)
        x = design_matrix(cov, n)
        _, p = logit_fit(d, x, w)
        p = np.minimum(p, 1 - 1e-6)
        keep = np.where(d == 1, 1.0, (p < 0.995).astype(float))
        a10 = w * d * (1 - post)
        a00 = keep * w * p / (1 - p) * (1 - d) * (1 - post)
        rows = (d == 0) & (post == 0)
        coef = wls_coef(x, y, w, rows)
        gap = HajekMean(a10, y).grad_through_z(x) - HajekMean(a00, y).grad_through_z(x)
        expected = 2.0 * wls_influence(x, y, w, rows, coef) @ gap
        np.testing.assert_allclose(dlt.influence - ref.influence, expected, atol=1e-12)


@pytest.mark.parametrize("weighting", ["none", "w"])
def test_singular_cell_regressions_are_refused_outside_the_improved_estimators(
    weighting,
):
    """Design 4 has ``x2`` constant within two cells.

    The reference stops with a singular-design error for the traditional DR
    and OR estimators (checked in the generator) and drops the column for
    the improved ones (rows ``rc_4`` of the fixture, matched above).
    """
    from statspai.exceptions import MethodIncompatibility

    df = pd.read_csv(_FIX / "drdid_blackbox_rc_4.csv")
    weights = df["w"].to_numpy() if weighting == "w" else None
    runs = _rc_runs(df, weights)
    for fn in ("drdid_rc", "drdid_rc1", "reg_did_rc"):
        with pytest.raises(MethodIncompatibility, match="singular"):
            runs[fn]()


def test_rc_inputs_fail_loudly():
    from statspai.exceptions import DataInsufficient, MethodIncompatibility

    df = pd.read_csv(_FIX / "drdid_blackbox_rc_1.csv")
    y, post, d = df["y"].to_numpy(), df["post"].to_numpy(), df["D"].to_numpy()
    with pytest.raises(MethodIncompatibility, match="non-negative"):
        _rcs.reg_did_rc(y, post, d, None, -np.ones(len(df)))
    with pytest.raises(MethodIncompatibility, match="0/1"):
        _rcs.reg_did_rc(y, post * 2, d, None)
    with pytest.raises(MethodIncompatibility, match="variance_convention"):
        _rcs.drdid_rc(y, post, d, None, variance_convention="nope")
    keep = ~((d == 0) & (post == 0))
    with pytest.raises(DataInsufficient, match="four"):
        _rcs.std_ipw_did_rc(y[keep], post[keep], d[keep], None)
