"""``sp.rdwinselect`` against rdlocrand 1.0, the release before two
window-construction regressions.

Running the same checks on four CRAN releases (0.9, 1.0, 1.1, 2.0):

=========================================  ====  ====  ====  ====
                                           0.9   1.0   1.1   2.0
=========================================  ====  ====  ====  ====
default first window holds ``obsmin``      yes   yes   no    no
``wmasspoints``: k-th point on each side   yes   yes   no    no
KS randomization p-value, binary variable  ok    ok    ok    1.0
=========================================  ====  ====  ====  ====

From 1.1 the first window holds ``obsmin - 1`` observations below the
cutoff and ``wmasspoints`` pairs the k-th support point above the cutoff
with the (k-1)-th below, so its first window is empty on the left (also
reported, on the read-only CRAN mirror, as cran/rdlocrand issue 1). From
2.0 the Kolmogorov-Smirnov randomization p-value is 1 on a binary variable.
The help page and Cattaneo, Idrobo & Titiunik (2024) describe what 1.0
does, so 1.0 is the reference here and the agreement is same-bytes parity
at 1e-9, not a documented divergence. The 2.0 fixture
(``rdlocrand_extensions_R.json``) covers everything those regressions do
not touch.

Fixture: ``_fixtures/_generate_rdlocrand_v1_R.R`` (needs the archived 1.0
tarball installed in a library of its own; the script says how).
"""

from __future__ import annotations

import json
import pathlib
import warnings
from itertools import combinations

import numpy as np
import pandas as pd
import pytest
from scipy import stats

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import statspai as sp

_FIX = pathlib.Path(__file__).parent / "_fixtures"
RTOL = 1e-9
_COVS = ["class", "termshouse", "termssenate"]


@pytest.fixture(scope="module")
def v1():
    path = _FIX / "rdlocrand_v1_R.json"
    if not path.exists():  # pragma: no cover
        pytest.skip("run _generate_rdlocrand_v1_R.R to build the fixture")
    out = json.loads(path.read_text(encoding="utf-8"))
    assert out["_meta"]["rdlocrand_version"] == "1.0"
    return out


@pytest.fixture(scope="module")
def senate():
    return pd.read_csv(_FIX / "rdsenate.csv")


def _assert_same(out, ref, names):
    np.testing.assert_allclose(out["window_right"], ref["w_right"], rtol=1e-12)
    assert out["n_left"].tolist() == ref["Nl"]
    assert out["n_right"].tolist() == ref["Nr"]
    np.testing.assert_allclose(out["binom_pvalue"], ref["binom"], rtol=RTOL)
    np.testing.assert_allclose(out["p_value"], ref["p_value"], rtol=RTOL)
    assert out["variable"].tolist() == [names[i - 1] for i in ref["variable"]]


@pytest.mark.parametrize(
    "key,kwargs",
    [
        ("default", {}),
        ("wobs2", {"wobs": 2}),
        ("obsmin5", {"obsmin": 5, "wobs": 3, "nwindows": 6}),
    ],
)
def test_default_window_sequences_match_rdlocrand_1_0(v1, senate, key, kwargs):
    """Windows, counts, binomial test, balance p-value, covariate."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        out = sp.rdwinselect(senate, x="margin", covs=_COVS, approx=True, **kwargs)
    _assert_same(out, v1[key], _COVS)


def test_first_window_has_obsmin_below_the_cutoff_in_1_0(v1):
    assert v1["default"]["Nl"][0] >= 10 and v1["default"]["Nr"][0] >= 10
    assert v1["obsmin5"]["Nl"][0] >= 5 and v1["obsmin5"]["Nr"][0] >= 5


def test_mass_point_windows_match_rdlocrand_1_0(v1):
    ref = v1["masspoints_toy"]
    x = np.r_[
        np.repeat(-np.arange(1, 11, dtype=float), 3),
        np.repeat(np.arange(10) + 0.5, 2),
    ]
    df = pd.DataFrame({"x": x, "z": ref["covariate"]})
    out = sp.rdwinselect(
        df, x="x", covs=["z"], wmasspoints=True, nwindows=4, approx=True
    )
    assert out["window_left"].tolist() == ref["w_left"]
    _assert_same(out, ref, ["z"])


def test_ks_on_a_binary_covariate_matches_rdlocrand_1_0(v1, senate):
    """Statistic and exact p-value to 1e-9; randomization p-value as a
    40-seed mean (S: each side's Monte Carlo SD is about 0.002)."""
    ref = v1["ks_binary"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        fits = [
            sp.rdrandinf(
                senate,
                y="dopen",
                x="margin",
                wl=-2,
                wr=2,
                statistic="ksmirnov",
                seed=s,
                ci=False,
            )
            for s in range(40)
        ]
    by_stat = fits[0].model_info["results_by_stat"]["ksmirnov"]
    assert by_stat["observed_stat"] == pytest.approx(ref["obs_stat"], rel=RTOL)
    assert by_stat["pvalue_asymptotic"] == pytest.approx(ref["asy_pvalue"], rel=RTOL)
    assert np.mean([f.pvalue for f in fits]) == pytest.approx(ref["seedmean"], abs=0.02)


# ── exact Kolmogorov-Smirnov p-value with ties ──────────────────────────


def test_exact_ks_pvalue_with_ties_is_the_permutation_distribution():
    from statspai.rd import _locrand_core as core

    a = np.array([0, 0, 1, 1, 1.0])
    b = np.array([0, 1, 1, 2, 2, 2.0])
    y = np.r_[a, b]
    lab = np.r_[np.ones(5), np.zeros(6)].astype(int)
    obs = core.ks_from_labels(y, lab[None, :])[0]
    hits = total = 0
    for idx in combinations(range(11), 5):
        relabelled = np.zeros(11, dtype=int)
        relabelled[list(idx)] = 1
        hits += core.ks_from_labels(y, relabelled[None, :])[0] >= obs - 1e-12
        total += 1
    assert core.ks_exact_pvalue(a, b, obs) == pytest.approx(hits / total, rel=1e-12)


@pytest.mark.parametrize("m,n", [(8, 11), (30, 25), (63, 57)])
def test_exact_ks_pvalue_without_ties_is_the_classical_one(m, n):
    from statspai.rd import _locrand_core as core

    rng = np.random.default_rng(m + n)
    a, b = rng.normal(size=m), rng.normal(0.4, 1, size=n)
    ref = stats.ks_2samp(a, b, method="exact")
    assert core.ks_exact_pvalue(a, b, ref.statistic) == pytest.approx(
        ref.pvalue, rel=1e-10
    )
