"""RD must say so when a covariate is collinear with the running variable.

An RD covariate that is a smooth function of the running variable is
already absorbed by the polynomial terms the estimator fits, so the
augmented design loses rank and the covariate adjustment stops being
identified. Before this guard, StatsPAI returned a number anyway.

The reason it went unnoticed is worth keeping in the test file, because
it is not a logic bug anyone would spot by reading the code: both
StatsPAI and ``rdrobust`` try a Cholesky factorisation and fall back to a
pseudo-inverse when it fails, but NumPy's Cholesky *succeeds* on the
singular matrix where R's refuses. The fallback therefore never fired on
our side, and the meaningless solve was returned as an ordinary answer.
R, for its part, returns a different number (``covs_drop=TRUE``, its
default) or refuses outright (``covs_drop=FALSE``).

The guard raises rather than warns: the estimate is not identified, and
carrying on produced a NaN standard error on one covariate scaling and a
bare ``LinAlgError: Singular matrix`` on another -- three different
meaningless outcomes where one clear refusal belongs.

These tests assert it fires where the design really is singular and stays
quiet where the covariate is merely correlated with the running variable,
since a guard that refused ordinary covariates would be worked around
rather than heeded.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp

RANK_MSG = "rank deficient"


def _frame(n: int = 1200, seed: int = 3) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    x = rng.uniform(-100.0, 100.0, n)
    y = 0.05 * x + 4.0 * (x >= 0) + rng.normal(0.0, 2.0, n)
    return pd.DataFrame(
        {
            "y": y,
            "x": x,
            # Exactly affine in x: [1, x, z_affine] is singular by construction.
            "z_affine": x / 50.0 + 0.3,
            # Correlated with x but carrying independent variation.
            "z_ok": np.sin(x / 3.0) + rng.normal(0.0, 0.5, n),
            # Independent of x entirely.
            "z_free": rng.normal(0.0, 1.0, n),
        }
    )


@pytest.mark.parametrize("entry", ["rdbwselect", "rdrobust"])
def test_collinear_covariate_raises(entry: str) -> None:
    """The singular case must be announced, not absorbed."""
    df = _frame()
    fn = getattr(sp, entry)
    with pytest.raises(ValueError, match=RANK_MSG):
        fn(df, y="y", x="x", c=0.0, covs=["z_affine"])


@pytest.mark.parametrize("entry", ["rdbwselect", "rdrobust"])
@pytest.mark.parametrize("col", ["z_ok", "z_free"])
def test_well_posed_covariates_stay_silent(entry: str, col: str) -> None:
    """A guard that fires on ordinary covariates would just get muted."""
    df = _frame()
    fn = getattr(sp, entry)
    with warnings_as_list() as caught:
        fn(df, y="y", x="x", c=0.0, covs=[col])
    assert not [m for m in caught if RANK_MSG in m], (
        f"rank guard fired on a well-posed covariate ({col}); it would be "
        "disabled by users and stop protecting the singular case"
    )


def test_no_covariates_is_never_flagged() -> None:
    df = _frame()
    with warnings_as_list() as caught:
        sp.rdbwselect(df, y="y", x="x", c=0.0)
    assert not [m for m in caught if RANK_MSG in m]


def test_collinearity_is_detected_regardless_of_covariate_scale() -> None:
    """Rescaling a covariate cannot make a singular design look healthy.

    The check normalises columns before reading singular values; without
    that, a covariate recorded in tiny units would push the smallest
    singular value under the threshold on units alone, and one recorded in
    huge units would hide a genuine rank deficiency.
    """
    df = _frame()
    for scale in (1e-6, 1.0, 1e6):
        scaled = df.assign(z_scaled=df["z_affine"] * scale)
        with pytest.raises(ValueError, match=RANK_MSG):
            sp.rdbwselect(scaled, y="y", x="x", c=0.0, covs=["z_scaled"])


def test_higher_polynomial_order_absorbs_more_covariates() -> None:
    """A quadratic covariate is collinear once ``p`` reaches 2, not before.

    This pins the check against the *fitted* basis rather than against x
    alone -- the identification problem is created by the polynomial order
    the user chose, so the guard has to move with it.
    """
    df = _frame()
    df = df.assign(z_quad=df["x"] ** 2)
    with warnings_as_list() as caught:
        sp.rdbwselect(df, y="y", x="x", c=0.0, covs=["z_quad"], p=1)
    assert not [m for m in caught if RANK_MSG in m], "p=1 basis has no x^2 term"
    with pytest.raises(ValueError, match=RANK_MSG):
        sp.rdbwselect(df, y="y", x="x", c=0.0, covs=["z_quad"], p=2)


class warnings_as_list:
    """Collect warning messages emitted inside the block."""

    def __enter__(self) -> list:
        import warnings

        self._ctx = warnings.catch_warnings(record=True)
        self._records = self._ctx.__enter__()
        warnings.simplefilter("always")
        self._out: list = []
        return self._out

    def __exit__(self, *exc) -> None:
        self._out.extend(str(r.message) for r in self._records)
        self._ctx.__exit__(*exc)
