"""Cross-language parity: ``sp.aipw`` / ``sp.ipw`` weights and clusters vs Stata.

Fixture: ``_fixtures/_generate_teffects_design_stata.do`` (Stata 18
``teffects aipw``, logit propensity + per-arm linear outcome). Both sides
read the same CSV bytes. ``sp.aipw`` is run with ``cross_fit=False,
se_method='sandwich'``, the parametric estimator ``teffects`` computes.

Stata's weight conventions, established while building this fixture:

* ``teffects aipw`` refuses ``[pw=]`` and accepts ``[iw=]``.
* Under ``vce(cluster c)`` the iweighted sandwich sums the weighted scores
  within clusters -- the sampling-weight sandwich -- with no ``G/(G-1)``
  factor. So ``sp.aipw(weights=)`` (pweight robust) is pinned against
  ``[iw=w], vce(cluster id)``: one observation per cluster.
* Under ``vce(robust)`` Stata reads iweights as frequencies (effective
  N = sum of w). That is a different quantity; ``iw_robust`` pins our
  reconstruction of it so the convention gap is documented, not assumed.
"""

from __future__ import annotations

import json
import pathlib
import warnings
from functools import lru_cache
from typing import Any, Dict

import numpy as np
import pandas as pd
import pytest

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import statspai as sp

from statspai.exceptions import MethodIncompatibility

_FIX = pathlib.Path(__file__).parent / "_fixtures"
# CLAUDE.md §5.1 default budget; observed agreement is ~1e-14.
RTOL = 1e-6
KW: Dict[str, Any] = dict(
    y="y",
    treat="d",
    covariates=["x1", "x2", "x3"],
    cross_fit=False,
    se_method="sandwich",
)
CASES = {
    "plain": {},
    "weights": {"weights": "w"},
    "cluster": {"cluster": "g"},
    "weights_cluster": {"weights": "w", "cluster": "g"},
}


@lru_cache(maxsize=None)
def _ref() -> Dict[str, Any]:
    path = _FIX / "teffects_design_stata.json"
    if not path.exists():  # pragma: no cover
        pytest.skip("run _generate_teffects_design_stata.do first")
    return json.loads(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=None)
def _data() -> pd.DataFrame:
    path = _FIX / "teffects_design_data.csv"
    if not path.exists():  # pragma: no cover
        pytest.skip("run _generate_teffects_design_stata.do first")
    return pd.read_csv(path)


def _fit(**extra: Any):
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return sp.aipw(_data(), **KW, **extra)


@pytest.mark.parametrize("case", list(CASES))
def test_aipw_design_matches_teffects(case: str) -> None:
    ref = _ref()[case]
    res = _fit(**CASES[case])
    po = res.model_info["potential_outcome_means"]
    po_se = res.model_info["potential_outcome_means_se"]
    np.testing.assert_allclose(res.estimate, ref["ate"], rtol=RTOL)
    np.testing.assert_allclose(res.se, ref["ate_se"], rtol=RTOL)
    for arm in (0, 1):
        np.testing.assert_allclose(po[arm], ref[f"po{arm}"], rtol=RTOL)
        np.testing.assert_allclose(po_se[arm], ref[f"po{arm}_se"], rtol=RTOL)
    assert res.n_obs == int(ref["N"])


def test_weights_move_the_ate() -> None:
    ref = _ref()
    assert abs(ref["weights"]["ate"] - ref["plain"]["ate"]) > 0.1


def test_stata_iweight_robust_is_the_frequency_sandwich() -> None:
    """Pin the documented convention gap rather than leave it implicit.

    Stata's ``[iw=w], vce(robust)`` treats w as frequencies. Rebuilding that
    from sp.aipw's own influence rows (u_i^2 / omega_i, effective N = sum w)
    reproduces Stata; it is not the sampling-weight SE sp.aipw reports.
    """
    from statspai.inference.aipw import _aipw_stacked_if, _fit_outcome, _fit_propensity

    df = _data()
    X = df[["x1", "x2", "x3"]].to_numpy(float)
    D = df["d"].to_numpy(float)
    Y = df["y"].to_numpy(float)
    w = df["w"].to_numpy(float)
    n = len(Y)
    om = w * n / w.sum()
    e = _fit_propensity(X, D, X, om)
    m1 = _fit_outcome(X[D == 1], Y[D == 1], X, om[D == 1])
    m0 = _fit_outcome(X[D == 0], Y[D == 0], X, om[D == 0])
    p1 = m1 + D * (Y - m1) / e
    p0 = m0 + (1 - D) * (Y - m0) / (1 - e)
    i1, i0 = _aipw_stacked_if(
        X, D, Y, e, m1, m0, p1 - np.mean(om * p1), p0 - np.mean(om * p0), om
    )
    u = i1 - i0
    freq_se = np.sqrt(np.sum(u**2 / om)) / n * np.sqrt(n / w.sum())
    np.testing.assert_allclose(freq_se, _ref()["iw_robust"]["ate_se"], rtol=RTOL)
    assert abs(_fit(weights="w").se / freq_se - 1) > 0.1


def test_unweighted_default_path_is_unchanged() -> None:
    """weights/cluster unset must reproduce the pre-1.32 numbers exactly."""
    res = _fit()
    ref = _ref()["plain"]
    np.testing.assert_allclose(res.estimate, ref["ate"], rtol=1e-12)
    np.testing.assert_allclose(res.se, ref["ate_se"], rtol=1e-12)


def test_cross_fit_influence_path_accepts_design_options() -> None:
    df = _data()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        base = sp.aipw(df, y="y", treat="d", covariates=["x1", "x2", "x3"])
        clus = sp.aipw(df, y="y", treat="d", covariates=["x1", "x2", "x3"], cluster="g")
        wt = sp.aipw(df, y="y", treat="d", covariates=["x1", "x2", "x3"], weights="w")
    assert clus.estimate == pytest.approx(base.estimate, rel=1e-12)
    # The cluster shock enters both arms additively, so it cancels in the
    # ATE contrast (Stata's cluster ATE SE is smaller too) but not in the
    # potential-outcome means, whose SE must grow.
    se_po = clus.model_info["potential_outcome_means_se"][0]
    assert se_po > 2 * base.model_info["potential_outcome_means_se"][0]
    assert abs(wt.estimate - base.estimate) > 0.1
    assert wt.model_info["n_clusters"] is None and clus.model_info["n_clusters"] == 60


def test_bad_design_inputs_fail_loudly() -> None:
    df = _data().assign(w0=lambda d: d["w"].where(d["id"] != 5, 0.0))
    with pytest.raises(MethodIncompatibility, match="strictly positive"):
        sp.aipw(df, weights="w0", **KW)
    with pytest.raises(MethodIncompatibility, match="not found"):
        sp.aipw(df, cluster="nope", **KW)
    with pytest.raises(MethodIncompatibility, match="two clusters"):
        sp.aipw(df.assign(one=1), cluster="one", **KW)


# --------------------------------------------------------------------------
# sp.ipw(se_method="sandwich") vs teffects ipw (pweights are allowed there)
# --------------------------------------------------------------------------

IPW_SPECS = {
    "plain": {},
    "pw": {"weights": "w"},
    "cluster": {"cluster": "g"},
    "pw_cluster": {"weights": "w", "cluster": "g"},
}


@pytest.mark.parametrize("stat,estimand", [("ate", "ATE"), ("atet", "ATT")])
@pytest.mark.parametrize("tag", list(IPW_SPECS))
def test_ipw_sandwich_matches_teffects_ipw(stat: str, estimand: str, tag: str) -> None:
    ref = _ref()[f"ipw_{stat}_{tag}"]
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        res = sp.ipw(
            _data(),
            "y",
            "d",
            ["x1", "x2", "x3"],
            estimand=estimand,
            se_method="sandwich",
            **IPW_SPECS[tag],
        )
    np.testing.assert_allclose(res.estimate, ref["est"], rtol=RTOL)
    np.testing.assert_allclose(res.se, ref["se"], rtol=RTOL)


def test_ipw_cluster_bootstrap_resamples_clusters() -> None:
    df = _data()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        a = sp.ipw(df, "y", "d", ["x1", "x2", "x3"], n_bootstrap=50, seed=1)
        b = sp.ipw(
            df, "y", "d", ["x1", "x2", "x3"], n_bootstrap=50, seed=1, cluster="g"
        )
        c = sp.ipw(df, "y", "d", ["x1", "x2", "x3"], n_bootstrap=50, seed=1)
    assert a.estimate == b.estimate
    assert a.se == c.se  # the unclustered default path is unchanged
    assert b.se != a.se and b.model_info["n_clusters"] == 60


def test_ipw_sandwich_refuses_trimmed_or_unnormalised_weights() -> None:
    with pytest.raises(MethodIncompatibility, match="normalize=True and trim=0"):
        sp.ipw(_data(), "y", "d", ["x1"], se_method="sandwich", trim=0.05)
    with pytest.raises(MethodIncompatibility, match="normalize=True and trim=0"):
        sp.ipw(_data(), "y", "d", ["x1"], se_method="sandwich", normalize=False)
