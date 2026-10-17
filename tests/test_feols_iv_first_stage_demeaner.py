"""sp.feols IV: the first stage runs on the model's demeaner, once.

pyfixest's ``Feiv.first_stage`` refits the first stage through
``pyfixest.estimation.feols`` without passing ``demeaner_backend`` on, so it
ran the numba demeaner while the model ran the compiled one, and
``_note_iv_diagnostics`` then refitted it a second time. On 1,000,000 rows
with two fixed effects that was 19 of the 21 seconds of the fit.

The old computation is kept here as the reference: the first stage fitted by
pyfixest with its default (numba) demeaner. The model's own estimates do not
involve the first stage and must be unchanged bit for bit; the first-stage F
statistic comes from a different demeaner and agrees to rounding.
"""

import importlib

import numpy as np
import pandas as pd
import pytest

import statspai as sp

pf = pytest.importorskip("pyfixest")


def _data(seed: int, n: int = 1500) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    w = rng.integers(0, 120, n)
    base = rng.integers(0, 15, 120)
    f = np.where(rng.uniform(size=n) < 0.3, rng.integers(0, 15, n), base[w])
    z = rng.normal(size=n) + 0.3 * f / 15
    if seed % 2:
        z = np.round(z)
    d = 0.4 * z + rng.normal(size=120)[w] + rng.normal(size=n)
    y = d + rng.normal(size=15)[f] + rng.normal(size=n)
    return pd.DataFrame(
        dict(
            y=y,
            d=d,
            z=z,
            z2=rng.normal(size=n),
            x1=rng.normal(size=n),
            w=w,
            f=f,
            cl=rng.integers(0, 20, n),
        )
    )


CASES = [
    ("y ~ x1 | w | d ~ z", "iid"),
    ("y ~ x1 | w + f | d ~ z", {"CRV1": "w"}),
    ("y ~ x1 | w + f | d ~ z + z2", "hetero"),
    ("y ~ x1 | f | d ~ z", {"CRV1": "cl+f"}),
    ("y ~ x1 | d ~ z + z2", {"CRV1": "cl"}),
]


@pytest.mark.parametrize("seed", [0, 1, 2])
@pytest.mark.parametrize("fml,vcov", CASES)
def test_first_stage_matches_the_old_numba_refit(fml, vcov, seed):
    df = _data(seed)
    out = sp.feols(fml, data=df, vcov=vcov)

    # The model: pyfixest's compiled demeaner, as sp.feols has asked for
    # since c824af18; the first stage plays no part in these numbers. From
    # pyfixest 0.60 that demeaner is the default and demeaner_backend is a
    # legacy argument sp.feols no longer passes, so the reference is built
    # the same way (on 0.60+ the legacy path differed in the 13th digit).
    import inspect

    legacy = "demeaner" not in inspect.signature(pf.feols).parameters
    extra = {"demeaner_backend": "rust"} if legacy else {}
    model = pf.feols(fml, data=df, vcov=vcov, **extra)
    np.testing.assert_array_equal(out.params.to_numpy(), np.asarray(model._beta_hat))
    if "Multiway VCOV negative eigenvalues" not in out.diagnostics:
        # (sp.feols repairs an indefinite two-way covariance; pyfixest not)
        np.testing.assert_array_equal(
            out.std_errors.to_numpy(), np.sqrt(np.diag(model._vcov))
        )

    # Old first stage: pyfixest's default demeaner, refitted.
    old = pf.feols(fml, data=df, vcov=vcov)
    old.first_stage()
    f_old = float(old._f_stat_1st_stage)
    f_new = float(out.model_info["first_stage_f"])
    assert f_new == pytest.approx(f_old, rel=1e-12, abs=0)


def test_pyfixest_feols_is_restored_after_the_fit():
    module = importlib.import_module("pyfixest.estimation")
    before = module.feols
    sp.feols("y ~ x1 | w | d ~ z", data=_data(0), vcov="iid")
    assert module.feols is before
    with pytest.raises(Exception):
        sp.feols("y ~ x1 | w | d ~ not_a_column", data=_data(0), vcov="iid")
    assert module.feols is before


def test_first_stage_is_fitted_once():
    calls = {"n": 0}
    Feiv = importlib.import_module("pyfixest.estimation.models.feiv_").Feiv
    real = Feiv.first_stage

    def counted(self):
        calls["n"] += 1
        return real(self)

    Feiv.first_stage = counted
    try:
        sp.feols("y ~ x1 | w + f | d ~ z", data=_data(1), vcov={"CRV1": "w"})
    finally:
        Feiv.first_stage = real
    assert calls["n"] == 1
