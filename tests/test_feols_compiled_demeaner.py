"""``sp.feols`` / ``fepois`` / ``feglm`` ask pyfixest for its compiled demeaner.

pyfixest's default numba demeaner took 108 s on 2M rows with 100k firm
effects; the rust one took 2.6 s with the coefficient equal to R fixest to
1e-15. The tests check the choice is passed through, that an explicit
choice wins, and that the estimates do not depend on it.
"""

import functools
import inspect

import numpy as np
import pandas as pd
import pytest

import statspai as sp

pf = pytest.importorskip("pyfixest")
pytest.importorskip("pyfixest.core._core_impl")


@pytest.fixture
def panel():
    rng = np.random.default_rng(7)
    n = 3000
    df = pd.DataFrame({"firm": rng.integers(0, 300, n), "year": rng.integers(0, 12, n)})
    df["x"] = rng.normal(size=n)
    df["y"] = 0.4 * df.x + 0.1 * (df.firm % 5) + rng.normal(size=n)
    df["c"] = rng.poisson(np.exp(0.3 * df.x))
    df["b"] = (df.y > 0).astype(int)
    return df


def _capture(monkeypatch, name):
    seen = {}
    real = getattr(pf, name)

    @functools.wraps(real)  # keeps the signature the wrapper inspects
    def spy(**kw):
        seen.update(kw)
        return real(**kw)

    monkeypatch.setattr(pf, name, spy)
    return seen


@pytest.mark.parametrize(
    "name, call",
    [
        ("feols", lambda d: sp.feols("y ~ x | firm + year", data=d)),
        ("fepois", lambda d: sp.fepois("c ~ x | firm + year", data=d)),
        (
            "feglm",
            lambda d: sp.feglm("b ~ x | firm + year", data=d, family="logit"),
        ),
    ],
)
def test_compiled_demeaner_is_requested(monkeypatch, panel, name, call):
    seen = _capture(monkeypatch, name)
    call(panel)
    if "demeaner" in inspect.signature(getattr(pf, name)).parameters:
        # pyfixest >= 0.60: the compiled demeaner is the default and the
        # legacy argument must not be combined with its typed ``demeaner=``.
        assert "demeaner_backend" not in seen
    else:
        assert seen["demeaner_backend"] == "rust"


def test_explicit_backend_wins_and_estimates_agree(monkeypatch, panel):
    if "demeaner" in inspect.signature(pf.feols).parameters:
        pytest.skip("pyfixest >= 0.60 takes a typed demeaner= instead")
    seen = _capture(monkeypatch, "feols")
    numba = sp.feols(
        "y ~ x | firm + year",
        data=panel,
        vcov={"CRV1": "firm"},
        demeaner_backend="numba",
    )
    assert seen["demeaner_backend"] == "numba"
    rust = sp.feols("y ~ x | firm + year", data=panel, vcov={"CRV1": "firm"})
    # Both stop at pyfixest's fixef_tol (1e-8 relative); the gap is
    # convergence noise, not a different estimator.
    np.testing.assert_allclose(rust.params, numba.params, rtol=1e-8)
    np.testing.assert_allclose(rust.std_errors, numba.std_errors, rtol=1e-8)
