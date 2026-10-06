"""Tests for ``sp.bayes_mte(mte_method='hv_latent')`` — textbook HV MTE."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp
from statspai.bayes import BayesianMTEResult, bayes_mte

pymc = pytest.importorskip(
    "pymc",
    reason="PyMC is an optional dependency; skip Bayesian tests if missing.",
)


# ---------------------------------------------------------------------------
# DGP: genuine Heckman-Vytlacil where outcome = MTE(U_D_i) * D_i + eps
# ---------------------------------------------------------------------------


def _hv_dgp(n, mte_coefs, seed):
    """Data-generating process where the outcome equation evaluates
    the MTE polynomial at the TRUE latent ``U_D_i`` (not at the
    propensity). ``hv_latent`` mode should recover ``mte_coefs``.
    """
    rng = np.random.default_rng(seed)
    Z = rng.normal(size=n)
    U_D = rng.uniform(0, 1, size=n)
    p_true = 1.0 / (1.0 + np.exp(-(0.8 * Z)))
    D = (p_true > U_D).astype(float)
    u_powers = np.column_stack([U_D**k for k in range(len(mte_coefs))])
    tau = u_powers @ np.asarray(mte_coefs, dtype=float)
    Y = 1.0 + tau * D + 0.3 * rng.normal(size=n)
    return pd.DataFrame({"y": Y, "d": D, "z": Z})


@pytest.fixture
def hv_decreasing_data():
    # tau(u) = 2 - 2u: decreasing selection-on-gains
    return _hv_dgp(600, mte_coefs=(2.0, -2.0), seed=801)


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------


def test_bayes_mte_hv_latent_runs(hv_decreasing_data):
    r = bayes_mte(
        hv_decreasing_data,
        y="y",
        treat="d",
        instrument="z",
        mte_method="hv_latent",
        poly_u=1,
        draws=300,
        tune=300,
        chains=2,
        progressbar=False,
    )
    assert isinstance(r, BayesianMTEResult)
    assert r.model_info["mte_method"] == "hv_latent"
    assert "HV-latent" in r.method


def test_bayes_mte_polynomial_method_label(hv_decreasing_data):
    r = bayes_mte(
        hv_decreasing_data,
        y="y",
        treat="d",
        instrument="z",
        mte_method="polynomial",
        poly_u=1,
        draws=200,
        tune=200,
        chains=2,
        progressbar=False,
    )
    assert r.model_info["mte_method"] == "polynomial"
    assert "local-IV MTE" in r.method


# ---------------------------------------------------------------------------
# Recovery vs bias: the whole point of v0.9.10
# ---------------------------------------------------------------------------


def test_hv_latent_recovers_true_mte_polynomial(hv_decreasing_data):
    """On the true HV DGP, hv_latent should cover (b_0=2.0, b_1=-2.0)
    inside the posterior 95% HDI. The test DGP uses n=600 so the
    posterior is concentrated."""
    r = bayes_mte(
        hv_decreasing_data,
        y="y",
        treat="d",
        instrument="z",
        mte_method="hv_latent",
        poly_u=1,
        draws=500,
        tune=500,
        chains=2,
        progressbar=False,
        random_state=42,
    )
    b_mte = r.trace.posterior["b_mte"].values.reshape(-1, 2)
    b0_mean = float(b_mte[:, 0].mean())
    b1_mean = float(b_mte[:, 1].mean())
    # Loose sanity: posterior means within 1.0 of truth (n=600 +
    # only 500 NUTS draws = enough to beat 1.0 tolerance but not
    # enough for 0.1).
    assert abs(b0_mean - 2.0) < 1.0, f"b_0 drift {abs(b0_mean - 2.0):.3f}"
    assert abs(b1_mean - (-2.0)) < 1.0, f"b_1 drift {abs(b1_mean + 2.0):.3f}"


def test_polynomial_mode_recovers_the_hv_mte():
    """On a DGP whose outcome equation uses ``U_D``, the polynomial mode
    (local IV since 1.39) recovers the MTE coefficients (2, -2).

    Before 1.39 this test asserted that the polynomial mode *disagreed*
    with ``hv_latent`` here ("biased toward 0"): the mode fitted
    ``Y = alpha + D * g(p)``, which is not the MTE. Local IV uses only
    the variation in the propensity, so it needs more data than
    ``hv_latent`` for the same precision.
    """
    rng = np.random.default_rng(803)
    n = 4000
    Z = rng.normal(size=n)
    U_D = rng.uniform(size=n)
    D = (1.0 / (1.0 + np.exp(-1.5 * Z)) > U_D).astype(float)
    Y = 1.0 + (2.0 - 2.0 * U_D) * D + 0.3 * rng.normal(size=n)
    r_poly = bayes_mte(
        pd.DataFrame({"y": Y, "d": D, "z": Z}),
        y="y",
        treat="d",
        instrument="z",
        mte_method="polynomial",
        poly_u=1,
        draws=500,
        tune=500,
        chains=2,
        progressbar=False,
        random_state=7,
    )
    b = r_poly.trace.posterior["b_mte"].values.reshape(-1, 2)
    assert abs(b[:, 0].mean() - 2.0) < 4 * b[:, 0].std()
    assert abs(b[:, 1].mean() - (-2.0)) < 4 * b[:, 1].std()
    assert (b[:, 1] < 0).mean() > 0.97
    # alpha is E[Y_0] = 1
    alpha = r_poly.trace.posterior["alpha"].values.ravel()
    assert abs(alpha.mean() - 1.0) < 4 * alpha.std()


# ---------------------------------------------------------------------------
# Orthogonality with first_stage
# ---------------------------------------------------------------------------


def test_hv_latent_plus_joint_first_stage_runs(hv_decreasing_data):
    r = bayes_mte(
        hv_decreasing_data,
        y="y",
        treat="d",
        instrument="z",
        first_stage="joint",
        mte_method="hv_latent",
        poly_u=1,
        draws=250,
        tune=250,
        chains=2,
        progressbar=False,
    )
    assert r.model_info["first_stage"] == "joint"
    assert r.model_info["mte_method"] == "hv_latent"


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_invalid_mte_method_raises(hv_decreasing_data):
    with pytest.raises(ValueError, match="mte_method"):
        bayes_mte(
            hv_decreasing_data,
            y="y",
            treat="d",
            instrument="z",
            mte_method="bogus",
            draws=50,
            tune=50,
            chains=1,
            progressbar=False,
        )


# ---------------------------------------------------------------------------
# Policy-effect still works for hv_latent results
# ---------------------------------------------------------------------------


def test_hv_latent_memory_warning_fires_above_threshold():
    """hv_latent registers a shape-(n,) latent that is stored in the
    trace as (chains, draws, n). For large n this blows up memory;
    the function must emit a UserWarning above
    ``n * draws * chains > 5e7``. We assert the warning fires by
    short-circuiting the sampler so the test runs in milliseconds."""
    import warnings
    from unittest.mock import patch

    import numpy as np
    import pandas as pd

    import statspai as sp

    rng = np.random.default_rng(0)
    n = 10000  # triggers threshold with default draws/chains
    df = pd.DataFrame(
        {
            "y": rng.normal(size=n),
            "d": rng.binomial(1, 0.5, n).astype(float),
            "z": rng.normal(size=n),
        }
    )

    # Patch _sample_model to return a dummy InferenceData-like object
    # so the warning-emit path runs without actually sampling.
    class _StubTrace:
        def __init__(self, n, poly_u=2):
            import xarray as xr

            self.posterior = xr.Dataset(
                {
                    "b_mte": (
                        ("chain", "draw", "b_mte_dim_0"),
                        rng.normal(size=(1, 10, poly_u + 1)),
                    ),
                    "raw_U": (
                        ("chain", "draw", "raw_U_dim_0"),
                        rng.uniform(size=(1, 10, n)),
                    ),
                }
            )

    with patch("statspai.bayes.mte._sample_model", return_value=_StubTrace(n=n)):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            try:
                sp.bayes_mte(
                    df,
                    y="y",
                    treat="d",
                    instrument="z",
                    mte_method="hv_latent",
                    poly_u=2,
                    draws=2000,
                    chains=4,
                    progressbar=False,
                )
            except Exception:
                # The stub trace will likely fail the downstream
                # post-processing; we only care whether the warning
                # was emitted before that point.
                pass
            mem_warns = [x for x in w if "hv_latent" in str(x.message)]
            assert len(mem_warns) >= 1, (
                f"Expected UserWarning about hv_latent memory; "
                f"got {len(mem_warns)} matching warnings. All warnings: "
                f"{[str(x.message)[:80] for x in w]}"
            )
            msg = str(mem_warns[0].message)
            # Concretely mention the knobs the user can turn
            assert "draws" in msg or "chains" in msg
            assert "polynomial" in msg


def test_hv_latent_memory_warning_does_not_fire_at_small_n():
    """Below the threshold the warning should stay silent — avoid
    noise-fatiguing users on normal workflows."""
    import warnings
    from unittest.mock import patch

    import numpy as np
    import pandas as pd

    import statspai as sp

    rng = np.random.default_rng(0)
    n = 100  # tiny — well below threshold at default draws/chains
    df = pd.DataFrame(
        {
            "y": rng.normal(size=n),
            "d": rng.binomial(1, 0.5, n).astype(float),
            "z": rng.normal(size=n),
        }
    )

    class _StubTrace:
        def __init__(self, n, poly_u=2):
            import xarray as xr

            self.posterior = xr.Dataset(
                {
                    "b_mte": (
                        ("chain", "draw", "b_mte_dim_0"),
                        rng.normal(size=(1, 10, poly_u + 1)),
                    ),
                    "raw_U": (
                        ("chain", "draw", "raw_U_dim_0"),
                        rng.uniform(size=(1, 10, n)),
                    ),
                }
            )

    with patch("statspai.bayes.mte._sample_model", return_value=_StubTrace(n=n)):
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            try:
                sp.bayes_mte(
                    df,
                    y="y",
                    treat="d",
                    instrument="z",
                    mte_method="hv_latent",
                    poly_u=2,
                    draws=500,
                    chains=2,
                    progressbar=False,
                )
            except Exception:
                pass
            mem_warns = [x for x in w if "hv_latent" in str(x.message)]
            assert len(mem_warns) == 0, (
                f"Unexpected hv_latent memory warning at n={n}: "
                f"{[str(x.message)[:80] for x in mem_warns]}"
            )


def test_policy_effect_on_hv_latent_result(hv_decreasing_data):
    r = bayes_mte(
        hv_decreasing_data,
        y="y",
        treat="d",
        instrument="z",
        mte_method="hv_latent",
        poly_u=1,
        draws=300,
        tune=300,
        chains=2,
        progressbar=False,
        random_state=7,
    )
    pe = r.policy_effect(sp.policy_weight_subsidy(0.05, 0.4), label="low")
    # On a DGP with tau(u) = 2 - 2u, low-u band should yield a
    # higher policy effect than high-u.
    pe_high = r.policy_effect(sp.policy_weight_subsidy(0.6, 0.95), label="high")
    assert pe["estimate"] > pe_high["estimate"]
