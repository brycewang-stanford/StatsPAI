"""Reference parity: repeated cross-section DiD primitives vs R ``DRDID``.

``statspai.did._rcs`` ports the three estimators R ``did::att_gt(panel=FALSE)``
dispatches to. Confirmed from the R source of ``did:::compute.att_gt``:

    est_method = "dr"   ->  DRDID::drdid_rc
    est_method = "ipw"  ->  DRDID::std_ipw_did_rc
    est_method = "reg"  ->  DRDID::reg_did_rc

These are the building blocks for repeated-cross-section support in
``sp.callaway_santanna``; pinning them here means the (g, t) loop that will
consume them can be debugged against a known-good inner estimator.

.. note::
   ``sp.drdid(method='imp')`` is a *different* estimator — it matches
   ``DRDID::drdid_rc1`` (3.0016328005 on this fixture), not ``drdid_rc``
   (3.0026780231). The two differ in how the outcome regressions enter, so
   reusing ``sp.drdid`` for CS repeated cross-sections would have silently
   produced non-parity numbers.

Reference generation (R 4.5.2, DRDID 1.2.3)::

    drdid_rc(y=df$y, post=df$post, D=df$d, covariates=cbind(1,df$x1,df$x2))
    std_ipw_did_rc(...)
    reg_did_rc(...)

Fixture: ``_fixtures/drdid_rc_panel.csv`` — 2000 observations, repeated
cross-section (``post`` drawn independently of the unit), true ATT = 3.0.

References
----------
Sant'Anna, P.H.C. and Zhao, J. (2020). "Doubly robust
difference-in-differences estimators." *Journal of Econometrics*, 219(1),
101-122. [@santanna2020doubly]
"""

from __future__ import annotations

import pathlib

import numpy as np
import pandas as pd
import pytest

from statspai.did._rcs import drdid_rc, reg_did_rc, std_ipw_did_rc

_FIXTURE = pathlib.Path(__file__).parent / "_fixtures" / "drdid_rc_panel.csv"

# (ATT, analytic SE) from R DRDID 1.2.3
R_REFERENCE = {
    "drdid_rc": (3.0026780231, 0.0924898005),
    "std_ipw_did_rc": (3.0392849753, 0.1325097857),
    "reg_did_rc": (2.9901454617, 0.1098485192),
}

_ESTIMATORS = {
    "drdid_rc": drdid_rc,
    "std_ipw_did_rc": std_ipw_did_rc,
    "reg_did_rc": reg_did_rc,
}


@pytest.fixture(scope="module")
def rc_data():
    if not _FIXTURE.exists():  # pragma: no cover - fixture ships with the repo
        pytest.skip(f"missing fixture: {_FIXTURE}")
    df = pd.read_csv(_FIXTURE)
    x = np.column_stack([df["x1"].to_numpy(), df["x2"].to_numpy()])
    return df, x


@pytest.mark.parametrize("name", sorted(R_REFERENCE))
def test_rcs_att_matches_drdid(rc_data, name):
    df, x = rc_data
    res = _ESTIMATORS[name](df["y"], df["post"], df["d"], x)
    att_r, _ = R_REFERENCE[name]
    assert res.att == pytest.approx(
        att_r, abs=1e-8
    ), f"{name}: StatsPAI {res.att:.10f} vs DRDID {att_r:.10f}"


@pytest.mark.parametrize("name", sorted(R_REFERENCE))
def test_rcs_analytic_se_matches_drdid(rc_data, name):
    """The influence function must reproduce R's SE, not just the ATT.

    The SE is what downstream ``aggte`` aggregation and the multiplier
    bootstrap consume, so an influence function that is merely 'close' is not
    good enough.
    """
    df, x = rc_data
    res = _ESTIMATORS[name](df["y"], df["post"], df["d"], x)
    _, se_r = R_REFERENCE[name]
    assert res.se == pytest.approx(
        se_r, abs=1e-8
    ), f"{name}: SE {res.se:.10f} vs DRDID {se_r:.10f}"


@pytest.mark.parametrize("name", sorted(R_REFERENCE))
def test_influence_function_is_centred_and_reproduces_se(rc_data, name):
    """Sanity contract on the returned influence function."""
    df, x = rc_data
    res = _ESTIMATORS[name](df["y"], df["post"], df["d"], x)
    n = len(df)

    assert res.influence.shape == (n,)
    assert abs(float(res.influence.mean())) < 1e-6
    recomputed = float(np.std(res.influence, ddof=1) * np.sqrt(n - 1) / n)
    assert recomputed == pytest.approx(res.se, rel=1e-12)


def test_recovers_the_true_att(rc_data):
    """All three should land near the design effect of 3.0."""
    df, x = rc_data
    for name, fn in _ESTIMATORS.items():
        res = fn(df["y"], df["post"], df["d"], x)
        assert abs(res.att - 3.0) < 0.15, f"{name} drifted from the true ATT"


def test_missing_cell_raises(rc_data):
    """All four treatment x period cells are required — fail loudly."""
    from statspai.exceptions import DataInsufficient

    df, x = rc_data
    keep = ~((df["d"] == 1) & (df["post"] == 1))
    with pytest.raises(DataInsufficient, match="four"):
        drdid_rc(
            df.loc[keep, "y"],
            df.loc[keep, "post"],
            df.loc[keep, "d"],
            x[keep.to_numpy()],
        )


def test_no_covariates_is_accepted(rc_data):
    """Intercept-only is a valid design (unconditional parallel trends)."""
    df, _ = rc_data
    res = drdid_rc(df["y"], df["post"], df["d"], None)
    assert np.isfinite(res.att) and res.se > 0
