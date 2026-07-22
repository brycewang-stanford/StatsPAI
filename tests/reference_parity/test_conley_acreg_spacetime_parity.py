"""Reference parity: ``sp.conley`` spatial + time HAC vs Stata ``acreg``.

Oracle
------
Stata 18 MP with ``acreg`` (Colella, Lalive, Sakalli & Thoenig). The frozen
covariance matrices below were produced by running ``acreg`` on the synthetic
geo-panel that :func:`panel` regenerates from ``default_rng(7)``. The exact
commands are recorded next to each oracle matrix.

What is asserted
----------------
1. **Standard errors** match ``acreg`` to ~1e-14 relative (asserted at 1e-9,
   target was 1e-6). This is the number users consume.
2. The **full weight matrix** is exact, not merely its diagonal: sandwiching
   ``_spatiotemporal_meat`` and applying Mata's ``_makesymmetric`` (mirror the
   lower triangle) *in acreg's own variable order* reproduces ``acreg``'s
   ``e(V)`` entrywise.

Known, deliberate difference
----------------------------
``acreg``'s anchored longitude scale makes ``S' W S`` genuinely asymmetric, and
Mata's ``_makesymmetric`` resolves that by mirroring the lower triangle. That
makes the reported off-diagonal covariances depend on the order the regressors
were typed in — see :func:`test_acreg_offdiagonals_are_variable_order_dependent`,
which pins ``acreg`` disagreeing with *itself* on ``Cov(x1, x2)`` in the 4th
significant digit. StatsPAI takes the order-invariant symmetric part instead.
Neither operation touches the diagonal, so variances, standard errors,
t-statistics and confidence intervals are unaffected.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd
import pytest

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    import statspai as sp
    from statspai.inference.conley import _spatiotemporal_meat

# --- acreg oracle, in acreg's variable order: [x1, x2, _cons] ---------------
# Common prefix: acreg y x1 x2, spatial latitude(lat) longitude(lon) dist(40)
ACREG_V = {
    # ... lag(3) id(id) time(t) hac bartlett
    "VA": [
        [3.82006329506092042e-03, 1.64793988577453809e-03, 1.80343581564648399e-04],
        [1.64793988577453809e-03, 4.18850194323838569e-03, 1.14659094860633076e-04],
        [1.80343581564648399e-04, 1.14659094860633076e-04, 2.81615402962693932e-03],
    ],
    # ... lag(3) id(id) time(t) hac
    "VB": [
        [4.10467468402558530e-03, 2.86454722258372039e-03, 7.04887319928316331e-04],
        [2.86454722258372039e-03, 5.21788942543751494e-03, -7.64869410428604295e-05],
        [7.04887319928316331e-04, -7.64869410428604295e-05, 1.71738748956464958e-03],
    ],
    # ... (spatial only, no panel options)
    "VC": [
        [3.95497307537070054e-04, 2.03161148247304243e-03, 8.52389070687592011e-05],
        [2.03161148247304243e-03, 4.00954474586823380e-03, 1.03540011153783508e-03],
        [8.52389070687592011e-05, 1.03540011153783508e-03, 8.91192194814947555e-04],
    ],
    # ... lag(0) id(id) time(t) hac bartlett
    "VD": [
        [4.14336880497642116e-03, 1.48427212445348920e-03, 3.22229288432896485e-04],
        [1.48427212445348920e-03, 4.52501096272790119e-03, 5.30738249328419672e-04],
        [3.22229288432896485e-04, 5.30738249328419672e-04, 2.48105315430945323e-03],
    ],
    # ... lag(5) id(id) time(t) hac bartlett
    "VE": [
        [3.84900890446635176e-03, 1.51896019789701125e-03, 1.93431544935067297e-04],
        [1.51896019789701125e-03, 4.26068499086779563e-03, -4.88308309987940876e-05],
        [1.93431544935067297e-04, -4.88308309987940876e-05, 2.96681524973849793e-03],
    ],
    # ... lag(3) lagdist(3) id(id) time(t) hac bartlett
    "VF": [
        [3.83978144832212076e-03, 1.72784008660007652e-03, 7.86997183023635093e-04],
        [1.72784008660007652e-03, 4.61250705901261604e-03, 4.96672929057995877e-04],
        [7.86997183023635093e-04, 4.96672929057995877e-04, 2.95400336016454507e-03],
    ],
    # ... lag(3) lagdist(2) id(id) time(t) hac
    "VG": [
        [3.55309212551536578e-03, 3.18765748146412203e-03, 1.55569642695396271e-03],
        [3.18765748146412203e-03, 3.83443100701409212e-03, 4.23944407276262186e-04],
        [1.55569642695396271e-03, 4.23944407276262186e-04, 1.62933801072749217e-03],
    ],
}

# sp.conley kwargs reproducing each acreg command.
CASES = {
    "VA": dict(kernel="bartlett", time="t", lag_cutoff=3, unit="id"),
    "VB": dict(kernel="uniform", time="t", lag_cutoff=3, unit="id"),
    "VC": dict(kernel="uniform"),
    "VD": dict(kernel="bartlett", time="t", lag_cutoff=0, unit="id"),
    "VE": dict(kernel="bartlett", time="t", lag_cutoff=5, unit="id"),
    "VF": dict(
        kernel="bartlett", time="t", lag_cutoff=3, lag_cutoff_cross=3, unit="id"
    ),
    "VG": dict(kernel="uniform", time="t", lag_cutoff=3, lag_cutoff_cross=2, unit="id"),
}


@pytest.fixture(scope="module")
def panel():
    """The exact synthetic geo-panel that was fed to Stata.

    25 units x 12 periods, coordinates confined to a ~0.5 degree box so the
    haversine and planar conventions nearly coincide.
    """
    rng = np.random.default_rng(7)
    n_units, n_periods = 25, 12
    lat0 = rng.uniform(40.0, 40.5, n_units)
    lon0 = rng.uniform(-100.0, -99.5, n_units)
    rows = [
        (u + 1, t, lat0[u], lon0[u])
        for u in range(n_units)
        for t in range(1, n_periods + 1)
    ]
    df = pd.DataFrame(rows, columns=["id", "t", "lat", "lon"])
    n = len(df)
    df["x1"] = rng.normal(size=n)
    df["x2"] = rng.normal(size=n)
    df["y"] = 1.0 + 0.5 * df.x1 - 0.3 * df.x2 + rng.normal(size=n)
    return df


@pytest.fixture(scope="module")
def fitted(panel):
    return sp.regress("y ~ x1 + x2", data=panel)


def _to_acreg_order(mat, names):
    """Permute a StatsPAI [Intercept, x1, x2] matrix into acreg's order."""
    perm = [names.index("x1"), names.index("x2"), names.index("Intercept")]
    return np.asarray(mat)[np.ix_(perm, perm)]


@pytest.mark.parametrize("case", sorted(CASES))
def test_standard_errors_match_acreg(panel, fitted, case):
    """SEs match acreg to machine precision (observed max rel err ~5e-15)."""
    res = sp.conley(
        fitted,
        panel,
        "lat",
        "lon",
        dist_cutoff=40,
        distance="planar",
        **CASES[case],
    )
    got = np.sqrt(
        np.diag(_to_acreg_order(res.data_info["vcov"], list(res.params.index)))
    )
    want = np.sqrt(np.diag(np.array(ACREG_V[case])))
    assert np.all(np.isfinite(got))
    # Target tolerance was 1e-6; the implementation reaches ~1e-14, so assert
    # something that would actually catch a regression.
    np.testing.assert_allclose(got, want, rtol=1e-9, atol=0)


@pytest.mark.parametrize("case", ["VA", "VD", "VE", "VF", "VG"])
def test_full_weight_matrix_is_exact_not_just_the_diagonal(panel, fitted, case):
    """Reproduce acreg's e(V) entrywise, including its _makesymmetric artifact.

    This is the strong test: it pins every off-diagonal too, which is only
    possible if the entire n x n weight matrix (spatial kernel, time kernel,
    same-unit vs cross-unit split, anchored asymmetric distance) is exact.
    """
    kw = CASES[case]
    X = np.asarray(fitted.data_info["X"])
    resid = np.asarray(fitted.data_info["residuals"])
    names = list(fitted.params.index)

    unit_codes = pd.factorize(panel["id"], sort=True)[0].astype(np.int64)
    time_codes = (panel["t"].to_numpy() - panel["t"].min()).astype(np.int64)

    meat = _spatiotemporal_meat(
        X * resid[:, None],
        panel["lat"].to_numpy(float),
        panel["lon"].to_numpy(float),
        unit_codes,
        time_codes,
        int(unit_codes.max()) + 1,
        int(time_codes.max()) + 1,
        40.0,
        kw["kernel"],
        "bartlett",
        int(kw["lag_cutoff"]),
        int(kw.get("lag_cutoff_cross") or 0),
        "planar",
    )
    bread = np.linalg.inv(X.T @ X)
    V = _to_acreg_order(bread @ meat @ bread, names)
    # Mata _makesymmetric: mirror the lower triangle into the upper one.
    V = np.tril(V) + np.tril(V, -1).T
    np.testing.assert_allclose(V, np.array(ACREG_V[case]), rtol=1e-9, atol=0)


def test_acreg_offdiagonals_are_variable_order_dependent():
    """Document why StatsPAI does not copy acreg's _makesymmetric.

    Same data, same command, regressors typed in the opposite order:

        acreg y x1 x2, spatial ... dist(40) lag(3) id(id) time(t) hac bartlett
            -> Cov(x1, x2) = 1.6479398857745e-03
        acreg y x2 x1, spatial ... dist(40) lag(3) id(id) time(t) hac bartlett
            -> Cov(x1, x2) = 1.6484267193922e-03

    while SE(x1) is 6.1806660604347e-02 under both. Mirroring an asymmetric
    matrix is order-dependent; taking its symmetric part is not, and both
    leave the diagonal alone.
    """
    cov_x1x2_order_12 = 1.6479398857745e-03
    cov_x1x2_order_21 = 1.6484267193922e-03
    se_x1_order_12 = 6.1806660604347e-02
    se_x1_order_21 = 6.1806660604347e-02

    # Off-diagonals disagree in the 4th significant digit ...
    rel = abs(cov_x1x2_order_21 - cov_x1x2_order_12) / cov_x1x2_order_12
    assert 1e-5 < rel < 1e-2
    # ... but the standard errors are identical.
    assert se_x1_order_12 == se_x1_order_21


def test_planar_and_haversine_nearly_agree_on_a_small_extent(panel, fitted):
    """On a ~0.5 degree box the two distance conventions should barely differ.

    They are genuinely different metrics (great-circle at R=6371 km vs acreg's
    flat 111 km/degree), so this is a sanity band, not an equality.
    """
    kw = dict(kernel="bartlett", time="t", lag_cutoff=3, unit="id")
    se_planar = sp.conley(
        fitted, panel, "lat", "lon", dist_cutoff=40, distance="planar", **kw
    ).std_errors
    se_hav = sp.conley(
        fitted, panel, "lat", "lon", dist_cutoff=40, distance="haversine", **kw
    ).std_errors
    np.testing.assert_allclose(
        se_hav.to_numpy(), se_planar.to_numpy(), rtol=0.05, atol=0
    )
