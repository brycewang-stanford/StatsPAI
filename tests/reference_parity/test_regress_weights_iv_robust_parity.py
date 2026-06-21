"""Reference parity: weighted OLS (Stata ``aweight``) and IV robust SE vocabulary.

These tests pin two correctness/usability fixes against **Stata 18 MP** values
captured live on 2026-06-16 from an identical deterministic dataset (seed
``20260616``, regenerated in :func:`_make_data` below):

1. ``sp.regress(..., weights=col)`` previously accepted ``weights`` via
   ``**kwargs`` and *silently ignored it*, returning unweighted OLS. It now
   fits WLS with Stata ``aweight`` semantics — point estimates, classical /
   HC1-robust / clustered SEs, and R² all match ``regress y x [aw=w]`` to
   machine precision.

2. ``sp.iv(..., robust='HC1')`` previously raised ``ValueError: Unknown robust
   type: HC1`` because the IV path did not normalise the SE-type spelling the
   way ``sp.regress`` does. It now accepts case-insensitive ``HC0``–``HC3``
   plus the aliases ``True`` / ``'robust'``, and the classical / robust SEs
   match ``ivregress 2sls, small`` / ``ivregress 2sls, robust small`` (the
   finite-sample t convention StatsPAI uses) to machine precision.

Stata reference values are hard-coded constants (the gold standard); the data
is regenerated deterministically so the test is hermetic (no Stata needed at
run time).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import statspai as sp


def _make_data() -> pd.DataFrame:
    """Recreate the exact dataset used to capture the Stata 18 references."""
    rng = np.random.default_rng(20260616)
    N = 600
    n_id = 60
    idv = np.repeat(np.arange(n_id), N // n_id)
    time = np.tile(np.arange(N // n_id), n_id)
    x1 = rng.normal(0, 1, N)
    x2 = rng.normal(0, 1, N)
    x3 = rng.binomial(1, 0.4, N).astype(float)
    ufe = np.repeat(rng.normal(0, 1.0, n_id), N // n_id)
    z = rng.normal(0, 1, N)
    u = rng.normal(0, 1, N)
    d = 0.8 * z + 0.5 * u + rng.normal(0, 0.5, N)
    y = 1.0 + 0.5 * x1 - 0.3 * x2 + 0.7 * x3 + 1.2 * d + ufe + u + rng.normal(0, 1, N)
    eta_c = 0.2 + 0.3 * x1 - 0.2 * x2 + 0.4 * x3
    count_y = rng.poisson(np.exp(eta_c))
    eta_b = -0.2 + 0.6 * x1 - 0.4 * x2 + 0.5 * x3
    p = 1 / (1 + np.exp(-eta_b))
    bin_y = (rng.uniform(0, 1, N) < p).astype(int)
    w = rng.uniform(0.5, 2.0, N)
    return pd.DataFrame(
        dict(
            id=idv,
            time=time,
            y=y,
            count_y=count_y,
            bin_y=bin_y,
            x1=x1,
            x2=x2,
            x3=x3,
            d=d,
            z=z,
            w=w,
            cluster=idv,
        )
    )


# Stata 18 MP, 2026-06-16. ``regress y x1 x2 x3 [aw=w]`` (+ robust / cluster).
_STATA_AW_B = {
    "x1": 0.6127724176888111,
    "x2": -0.4707409278172578,
    "x3": 0.7003404368771670,
    "Intercept": 1.043298177949708,
}
_STATA_AW_SE_CLASSICAL = {
    "x1": 0.1021412641411457,
    "x2": 0.09965160458366089,
    "x3": 0.2040632186763977,
    "Intercept": 0.1254117342172776,
}
_STATA_AW_SE_ROBUST = {  # regress [aw=w], robust  (HC1)
    "x1": 0.09762774819731045,
    "x2": 0.1027406929076569,
    "x3": 0.2152189547520558,
    "Intercept": 0.1312642441113679,
}
_STATA_AW_SE_CLUSTER = {  # regress [aw=w], vce(cluster cluster)
    "x1": 0.09774324657474168,
    "x2": 0.1107827567078808,
    "x3": 0.1965147287425859,
    "Intercept": 0.1744999242977125,
}
_STATA_AW_R2 = 0.1013818395416003

# Stata 18 MP, 2026-06-16. ``ivregress 2sls y x1 x2 x3 (d = z), small``.
_STATA_IV_B = {
    "d": 1.346874322755188,
    "x1": 0.6334124619160230,
    "x2": -0.4979755125588632,
    "x3": 0.7402389676601552,
    "Intercept": 0.9514693178686223,
}
_STATA_IV_SE_CLASSICAL = {  # ivregress 2sls, small
    "d": 0.08686214900411880,
    "x1": 0.07083760790192627,
    "x2": 0.07036725614212229,
    "x3": 0.1440559447835836,
    "Intercept": 0.08788590871955344,
}
_STATA_IV_SE_ROBUST = {  # ivregress 2sls, robust small
    "d": 0.08632194947216774,
    "x1": 0.07062468330983299,
    "x2": 0.06484650878952905,
    "x3": 0.1463971103909562,
    "Intercept": 0.08577345199315692,
}

_RTOL = 1e-9  # machine-precision parity (point estimates / SEs)


@pytest.fixture(scope="module")
def data() -> pd.DataFrame:
    return _make_data()


def _const_key(index) -> str:
    for c in ("Intercept", "const", "_cons", "Const"):
        if c in index:
            return c
    raise AssertionError(f"no constant term found in {list(index)}")


def _assert_match(series, ref: dict, ck: str, rtol: float = _RTOL):
    for name, expected in ref.items():
        key = ck if name == "Intercept" else name
        got = float(series[key])
        assert got == pytest.approx(
            expected, rel=rtol, abs=1e-12
        ), f"{name}: {got!r} != Stata {expected!r}"


# ---------------------------------------------------------------------------
# 1. Weighted OLS == Stata aweight
# ---------------------------------------------------------------------------
class TestRegressAnalyticWeights:
    def test_weighted_classical_matches_stata_aweight(self, data):
        r = sp.regress("y ~ x1 + x2 + x3", data, weights="w")
        ck = _const_key(r.params.index)
        _assert_match(r.params, _STATA_AW_B, ck)
        _assert_match(r.std_errors, _STATA_AW_SE_CLASSICAL, ck)
        assert float(r.diagnostics["R-squared"]) == pytest.approx(
            _STATA_AW_R2, rel=1e-9
        )

    def test_weighted_robust_matches_stata_aweight_robust(self, data):
        r = sp.regress("y ~ x1 + x2 + x3", data, robust="HC1", weights="w")
        ck = _const_key(r.params.index)
        _assert_match(r.params, _STATA_AW_B, ck)
        _assert_match(r.std_errors, _STATA_AW_SE_ROBUST, ck)

    def test_weighted_cluster_matches_stata_aweight_cluster(self, data):
        r = sp.regress("y ~ x1 + x2 + x3", data, cluster="cluster", weights="w")
        ck = _const_key(r.params.index)
        _assert_match(r.params, _STATA_AW_B, ck)
        _assert_match(r.std_errors, _STATA_AW_SE_CLUSTER, ck)

    def test_weights_actually_change_estimates(self, data):
        """Regression guard against the silent-drop bug returning."""
        unw = sp.regress("y ~ x1 + x2 + x3", data).params
        wtd = sp.regress("y ~ x1 + x2 + x3", data, weights="w").params
        assert not np.allclose(unw.values, wtd.values, atol=0, rtol=0)

    @pytest.mark.parametrize(
        "bad, exc_match",
        [
            ("missing_col", "not a column"),
            ("neg", "positive"),
            ("nan", "NaN or infinite"),
        ],
    )
    def test_bad_weights_fail_loudly(self, data, bad, exc_match):
        df = data.copy()
        if bad == "neg":
            df["bw"] = -1.0
            col = "bw"
        elif bad == "nan":
            df["bw"] = np.nan
            col = "bw"
        else:
            col = "no_such_column"
        with pytest.raises(ValueError, match=exc_match):
            sp.regress("y ~ x1 + x2 + x3", df, weights=col)


# ---------------------------------------------------------------------------
# 2. IV robust SE vocabulary + ivregress ,small parity
# ---------------------------------------------------------------------------
class TestIVRobustVocabulary:
    FML = "y ~ (d ~ z) + x1 + x2 + x3"

    def test_classical_matches_ivregress_small(self, data):
        r = sp.iv(self.FML, data)
        ck = _const_key(r.params.index)
        _assert_match(r.params, _STATA_IV_B, ck)
        _assert_match(r.std_errors, _STATA_IV_SE_CLASSICAL, ck)

    @pytest.mark.parametrize("spec", ["HC1", "hc1", "robust", True])
    def test_robust_spellings_accepted_and_equal(self, data, spec):
        """All HC1 spellings must work (uppercase 'HC1' used to raise)."""
        r = sp.iv(self.FML, data, robust=spec)
        ck = _const_key(r.params.index)
        _assert_match(r.std_errors, _STATA_IV_SE_ROBUST, ck)

    def test_unknown_robust_fails_loudly(self, data):
        with pytest.raises(ValueError, match="Unknown robust option"):
            sp.iv(self.FML, data, robust="hc9")
