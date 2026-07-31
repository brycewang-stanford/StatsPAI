"""Reference parity: ``sp.qdid`` vs R ``qte::QDiD`` 1.3.1 (WP-3).

Estimator
---------
Quantile difference-in-differences,

    QTE(tau) = [Q_11(tau) - Q_10(tau)] - [Q_01(tau) - Q_00(tau)]

on ``qte::lalonde.psid.panel`` with the post period 1978 and the pre period
1975.

This is NOT changes-in-changes
------------------------------
``sp.qdid`` was documented, labelled and registered as Athey & Imbens (2006)
changes-in-changes through 1.20.0. It implements QDiD, which Athey & Imbens
propose CiC *in place of* and criticise directly -- differencing quantiles
presumes the untreated distribution shifts by the same amount at every rank.
R's ``qte`` package keeps ``QDiD()`` and ``CiC()`` as separate functions for
exactly this reason. No numbers changed; the attribution did.

``method='cic'`` now delegates to :func:`statspai.cic`, so the package has
one changes-in-changes implementation rather than three code paths that
could drift apart.

Tolerance
---------
Anchored on point estimates, which are deterministic. R's quantiles come
from ``BMisc::weighted_quantile`` (``stats::optimize`` on a piecewise-linear
check function) while ``sp.qdid`` interpolates the empirical inverse CDF, so
the two differ by plateau width -- the same convention gap documented at
length in ``test_firpo_qte_parity.py``. On this fixture the gap is at most
~152 currency units against effects running to ~8900, and the curves agree
in shape, sign and magnitude.
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


def _cic_qte(result) -> np.ndarray:
    """Per-quantile effects out of a ``CICResult`` (they live in ``.detail``)."""
    return np.asarray(result.detail["qte"], dtype=float)


@pytest.fixture(scope="module")
def rjson():
    path = _FIX / "qte_panel_R.json"
    if not path.exists():  # pragma: no cover
        pytest.skip("run _generate_qte_panel_R.R to build qte_panel_R.json")
    return json.loads(path.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def two_period_panel():
    """lalonde.psid.panel restricted to the 1975 -> 1978 contrast."""
    panel = pd.read_csv(_FIX / "qte_lalonde_panel.csv")
    sub = panel[panel["year"].isin([1975, 1978])].copy()
    sub["post"] = (sub["year"] == 1978).astype(int)
    return sub


# ── QDiD point-estimate parity ─────────────────────────────────────── #


def test_qdid_matches_r(rjson, two_period_panel):
    ref = np.asarray(rjson["qdid_nocov"]["qte"], dtype=float)
    probs = list(np.asarray(rjson["qdid_nocov"]["probs"], dtype=float))
    res = sp.qdid(
        two_period_panel,
        y="re",
        group="treat",
        time="post",
        quantiles=probs,
        n_boot=2,
    )
    scale = float(np.mean(np.abs(ref)))
    dev = np.abs(res.effects - ref)
    # Observed on this fixture: max deviation 151.6 against a mean |effect|
    # of 2465.9, i.e. 6.2% at the single worst tau, correlation 0.99984.
    # The gap is the quantile-convention difference documented in the module
    # docstring, not a difference of estimator.
    assert np.max(dev) / scale < 0.08, (
        f"max deviation {np.max(dev):.1f} on scale {scale:.1f}\n"
        f"ours {np.round(res.effects, 1)}\nR    {np.round(ref, 1)}"
    )
    # Shape agreement is the substantive claim: the curve rises across tau.
    big = np.abs(ref) > 500
    assert np.all(np.sign(res.effects[big]) == np.sign(ref[big]))
    assert np.corrcoef(res.effects, ref)[0, 1] > 0.999


def test_qdid_recovers_a_known_constant_effect():
    """Hand-set 2x2 DiD with a constant effect of 2.0 at every quantile."""
    rng = np.random.default_rng(0)
    n = 20_000
    g = rng.integers(0, 2, n)
    t = rng.integers(0, 2, n)
    y = 1.0 + 0.5 * g + 0.3 * t + 2.0 * g * t + rng.normal(0, 1, n)
    res = sp.qdid(
        pd.DataFrame({"y": y, "g": g, "t": t}),
        y="y",
        group="g",
        time="t",
        quantiles=[0.25, 0.5, 0.75],
        n_boot=2,
    )
    assert np.all(np.abs(res.effects - 2.0) < 0.15), res.effects


# ── the attribution fix ────────────────────────────────────────────── #


def test_qdid_no_longer_claims_changes_in_changes():
    """Regression guard on the mislabel, in every place it appeared."""
    rng = np.random.default_rng(1)
    n = 800
    g = rng.integers(0, 2, n)
    t = rng.integers(0, 2, n)
    y = g * t + rng.normal(size=n)
    res = sp.qdid(
        pd.DataFrame({"y": y, "g": g, "t": t}),
        y="y",
        group="g",
        time="t",
        quantiles=[0.5],
        n_boot=2,
    )
    assert "QDiD" in res.method
    assert "Athey" not in res.method and "Changes" not in res.method

    spec = sp.describe_function("qdid")
    desc = str(spec.get("description", ""))
    assert "not changes-in-changes" in desc.lower()

    doc = sp.qdid.__doc__ or ""
    assert "not Changes-in-Changes" in doc


# ── CiC delegation: one implementation, not three ──────────────────── #


def test_cic_method_delegates_to_sp_cic():
    rng = np.random.default_rng(2)
    n = 3000
    g = rng.integers(0, 2, n)
    t = rng.integers(0, 2, n)
    y = 1.0 + 0.5 * g + 0.3 * t + 1.5 * g * t + rng.normal(0, 1, n)
    df = pd.DataFrame({"y": y, "g": g, "t": t})
    taus = [0.25, 0.5, 0.75]

    via_qdid = sp.qdid(
        df,
        y="y",
        group="g",
        time="t",
        quantiles=taus,
        n_boot=5,
        seed=7,
        method="cic",
    )
    direct = sp.cic(df, y="y", group="g", time="t", quantiles=taus, n_boot=5, seed=7)
    assert type(via_qdid) is type(direct)
    # Same code path => identical numbers, not merely close.
    np.testing.assert_allclose(_cic_qte(via_qdid), _cic_qte(direct))
    assert via_qdid.estimate == pytest.approx(direct.estimate)


def test_qdid_and_cic_disagree_on_a_nonlinear_dgp():
    """The two estimators must actually differ, or the distinction is empty.

    Under a monotone but non-linear transformation of the outcome the
    untreated distribution does NOT shift by a constant at every rank, which
    is the assumption QDiD makes and CiC does not.
    """
    rng = np.random.default_rng(3)
    n = 8000
    g = rng.integers(0, 2, n)
    t = rng.integers(0, 2, n)
    latent = 1.0 + 0.6 * g + 0.8 * t + 1.0 * g * t + rng.normal(0, 1, n)
    y = np.exp(latent / 2.0)  # monotone, strongly non-linear
    df = pd.DataFrame({"y": y, "g": g, "t": t})
    taus = [0.25, 0.5, 0.75]

    q = sp.qdid(df, y="y", group="g", time="t", quantiles=taus, n_boot=2)
    c = sp.cic(df, y="y", group="g", time="t", quantiles=taus, n_boot=2, seed=0)
    assert not np.allclose(q.effects, _cic_qte(c), rtol=0.05), (q.effects, _cic_qte(c))


def test_unknown_qdid_method_raises():
    df = pd.DataFrame(
        {
            "y": [1.0, 2, 3, 4, 5, 6, 7, 8],
            "g": [0, 0, 1, 1, 0, 0, 1, 1],
            "t": [0, 1, 0, 1, 0, 1, 0, 1],
        }
    )
    with pytest.raises(ValueError, match="method must be 'qdid' or 'cic'"):
        sp.qdid(df, y="y", group="g", time="t", method="mdid")
